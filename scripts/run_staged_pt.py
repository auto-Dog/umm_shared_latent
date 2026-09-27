#!/usr/bin/env python
"""Staged full-mix PT runner for the flaky hf-mirror host.

The full 600K+600K mix cannot be streamed from the mirror in one long run: the
mirror drops multi-GB parquet transfers mid-flight and hf_transfer has no read
timeout, so the run wedges for hours. Instead we download a bounded batch of
shards per stage (resumable curl), train on them from local disk (zero network),
delete them, and repeat. One global LR schedule (max_steps=75000) is kept across
all stages via Trainer checkpoint resume (StopAtStepCallback + ignore_data_skip,
see src/umm_uniquery/train.py).

Usage:
  UNIQUERY_PYTHON=... CUDA_VISIBLE_DEVICES=1,3 python scripts/run_staged_pt.py \
      --base-config configs/local_pt_ivl3.yaml \
      --output-root outputs/pt_ivl3_staged
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import yaml

# omni_files, cc12m_files, omni_rows, cc12m_rows
StageData = tuple[list[Path], list[Path], int, int]

ROOT = Path(__file__).resolve().parent.parent
MIRROR = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
OMNI_REPO = "TIGER-Lab/OmniEdit-Filtered-1.2M"
OMNI_SHARDS = 571  # data/train-%05d-of-00571.parquet
OMNI_TOTAL_QUOTA = 600_000
CC12M_REPO = "pixparse/cc12m-wds"
# Verified via the mirror API (2026-09-25): 2176 tars, cc12m-train-0000.tar ..
# cc12m-train-2175.tar, consecutive, no gaps. ~5041 rows/tar (tar 0 measured)
# -> ~11M rows total, matching the dataset viewer. Nominal CC12M is 12M.
CC12M_TARS = 2176  # cc12m-train-%04d.tar
CC12M_TOTAL_QUOTA = 600_000

OMNI_PER_STAGE = 6
# 3 tars/stage so the 120-tar CC12M quota (600K rows) fits inside the 48
# omni stages instead of trailing ~72 cc12m-only stages.
CC12M_PER_STAGE = 3


def omni_url(shard: int) -> str:
    return (
        f"{MIRROR}/datasets/{OMNI_REPO}/resolve/main/"
        f"data/train-{shard:05d}-of-{OMNI_SHARDS:05d}.parquet"
    )


def cc12m_url(tar: int) -> str:
    return (
        f"{MIRROR}/datasets/{CC12M_REPO}/resolve/main/"
        f"cc12m-train-{tar:04d}.tar"
    )


def expected_size(url: str) -> int:
    last_err: Exception | None = None
    for _ in range(3):
        try:
            out = subprocess.run(
                ["curl", "-sIL", "--max-time", "60", "-o", "/dev/null", "-w", "%{http_code} %{content_length}",
                 url],
                capture_output=True, text=True,
            )
            parts = out.stdout.split()
            code = parts[0] if parts else ""
            if code != "200":
                raise RuntimeError(f"HEAD -> HTTP {code}")
            if len(parts) > 1 and parts[1]:
                return int(parts[1])
            # Some responses omit Content-Length on HEAD (chunked). Probe the total
            # via a 1-byte range request instead of downloading the whole body.
            rng = subprocess.run(
                ["curl", "-sSL", "-r", "0-0", "-D", "-", "-o", "/dev/null", "--max-time", "60", url],
                capture_output=True, text=True,
            )
            for line in rng.stdout.splitlines():
                if line.lower().startswith("content-range:"):
                    return int(line.rsplit("/", 1)[1].strip())
            raise RuntimeError("no size header on HEAD or range request")
        except RuntimeError as exc:
            last_err = exc
            time.sleep(5)
    raise RuntimeError(f"Could not determine size of {url}: {last_err}")


def download_file(url: str, target: Path, expected: int) -> None:
    """Resumable, retried download with size verification against the remote HEAD.

    Size is accepted as >= the HEAD value, not ==: the mirror occasionally serves
    a slightly different revision of a shard than the HEAD reported (observed:
    train-00015 at 5.142GB vs 5.110GB HEAD — still a complete, valid parquet).
    Downstream parquet_rows/tar_rows re-validate integrity on the actual file.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 6):
        if target.is_file() and target.stat().st_size >= expected:
            return
        result = subprocess.run(
            [
                "curl", "-sSL", "-C", "-", "--retry", "20", "--retry-delay", "5",
                "--retry-all-errors", "--max-time", "900", "-o", str(target), url,
            ],
            capture_output=True, text=True,
        )
        if target.is_file() and target.stat().st_size >= expected:
            return
        print(f"  [dl] attempt {attempt} incomplete "
              f"({target.stat().st_size if target.exists() else 0}/{expected} bytes); retrying", flush=True)
        if result.returncode != 0:
            time.sleep(10)
    raise RuntimeError(f"Failed to download {url} after 5 attempts")


def parquet_rows(path: Path) -> int:
    """Full-scan a parquet file, returning its row count.

    A metadata-only read cannot see torn pages: a shard that downloaded fine
    by size but is corrupt mid-file (observed 2026-09-27 on omni_00209, from a
    disk-full window) still opens for metadata but throws
    `Deserializing page header failed` on a full read — which then crashed the
    trainer ~20 min later. Scanning every page here catches that at download
    time instead.
    """
    return pq.read_table(path).num_rows


def tar_rows(path: Path) -> int:
    count = 0
    with tarfile.open(path, "r:*") as tar:
        for member in tar:
            if member.name.endswith(".jpg"):
                count += 1
    return count


def read_global_step(checkpoint_dir: Path) -> int:
    state = yaml.safe_load((checkpoint_dir / "trainer_state.json").read_text())
    return int(state["global_step"])


def build_stage_config(base: dict, out: Path, idx: int, omni_files: list[str],
                       omni_rows: int, cc12m_files: list[str], cc12m_rows: int,
                       stage_start_step: int, max_steps: int) -> Path:
    import copy
    cfg = copy.deepcopy(base)
    # Per-stage quota multiples of the effective batch so no trailing drop_last.
    def quota(rows: int) -> int:
        return max(16, (rows // 16) * 16)

    cfg["data"]["sources"] = [
        {
            "kind": "omniedit_local",
            "path": omni_files,
            "split": "train",
            "sample_count": quota(omni_rows),
            "seed_offset": idx,
        },
    ]
    # Omni-only stages (40-47) carry no CC12M tars; omit the source entirely —
    # load_dataset rejects an empty data_files list.
    if cc12m_files:
        cfg["data"]["sources"].append(
            {
                "kind": "cc12m_wds",
                "path": cc12m_files,
                "split": "train",
                "sample_count": quota(cc12m_rows),
                "seed_offset": idx,
            }
        )
    cfg["training"].update(
        {
            "output_dir": str(out),
            "run_name": f"uniquery-pt-ivl3-stage-{idx:02d}",
            "max_steps": max_steps,
            "stage_start_step": stage_start_step,
            "ignore_data_skip": True,
            "save_steps": 500,
            "save_total_limit": 2,
        }
    )
    path = ROOT / "configs" / f"stage_ivl3_{idx:02d}_generated.yaml"
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, default_flow_style=False, allow_unicode=True)
    return path


def latest_checkpoint(output_dir: Path) -> Path | None:
    candidates = sorted(
        (p for p in output_dir.glob("checkpoint-*") if p.is_dir()),
        key=lambda p: int(p.name.split("-")[1]),
        reverse=True,
    )
    for candidate in candidates:
        if (candidate / "trainer_state.json").is_file() and (
            (candidate / "optimizer.pt").is_file()
            or (candidate / "scheduler.pt").is_file()
        ):
            return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", default="configs/local_pt_ivl3.yaml")
    parser.add_argument("--output-root", default="outputs/pt_ivl3_staged")
    parser.add_argument("--staging-root", default="/data/mingjun/pt_stage")
    parser.add_argument("--nproc", type=int, default=2)
    parser.add_argument("--omni-per-stage", type=int, default=OMNI_PER_STAGE)
    parser.add_argument("--cc12m-per-stage", type=int, default=CC12M_PER_STAGE)
    parser.add_argument("--max-stages", type=int, default=0, help="0 = until quotas are exhausted")
    parser.add_argument("--from-stage", type=int, default=0)
    parser.add_argument("--gpus", default="1,3")
    args = parser.parse_args()

    python = os.environ.get("UNIQUERY_PYTHON", "python")
    base_path = ROOT / args.base_config
    base = yaml.safe_load(base_path.read_text())
    output_root = ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    staging_root = Path(args.staging_root)

    effective_batch = (
        int(base["training"]["per_device_train_batch_size"])
        * int(base["training"].get("gradient_accumulation_steps", 1))
        * args.nproc
    )
    total_samples = sum(s["sample_count"] for s in base["data"]["sources"])
    max_steps = int(base["training"].get("max_steps") or -(-total_samples // effective_batch))

    # Exact per-stage shard/tar assignment across the full quotas. Row counts are
    # approximate per shard (~2100) / tar (~5041); the stage quota is pinned from
    # the *actual* downloaded row counts, so the plan only needs the totals.
    omni_total = min(-(-OMNI_TOTAL_QUOTA // 2100), OMNI_SHARDS)  # ~286 shards
    cc12m_total = min(-(-CC12M_TOTAL_QUOTA // 5041), CC12M_TARS)  # 120 tars (120*5041=604,920)
    stage_plan: list[tuple[list[int], list[int]]] = []
    shard = tar = 0
    while shard < omni_total or tar < cc12m_total:
        count_omni = min(args.omni_per_stage, omni_total - shard)
        count_cc12m = min(args.cc12m_per_stage, cc12m_total - tar)
        stage_plan.append((list(range(shard, shard + count_omni)), list(range(tar, tar + count_cc12m))))
        shard += count_omni
        tar += count_cc12m

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = args.gpus
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env.setdefault("HF_ENDPOINT", MIRROR)
    # Reduce allocator fragmentation on the 24GB cards shared with other jobs.
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    stage_start_step = 0
    if args.from_stage > 0:
        ckpt = latest_checkpoint(output_root)
        if ckpt is None:
            raise SystemExit(f"--from-stage {args.from_stage} but no checkpoint found in {output_root}")
        stage_start_step = read_global_step(ckpt)

    total_stages = len(stage_plan)
    print(f"[staged] {total_stages} stages, batch={effective_batch}, max_steps={max_steps}, "
          f"starting at stage {args.from_stage}", flush=True)

    def prepare_stage(idx: int) -> StageData:
        """Download one plan entry into its stage dir and count rows."""
        stage_dir = staging_root / f"stage_{idx:02d}"
        omni_shards_idx, cc12m_tars_idx = stage_plan[idx]
        omni_files: list[Path] = []
        cc12m_files: list[Path] = []
        for shard in omni_shards_idx:
            url = omni_url(shard)
            target = stage_dir / f"omni_{shard:05d}.parquet"
            print(f"  [dl] omni shard {shard}/{OMNI_SHARDS - 1} ...", flush=True)
            download_file(url, target, expected_size(url))
            # Verify the full file (not just the footer); a corrupt download
            # crashes training ~20 min later, so catch it now and re-fetch.
            for attempt in range(1, 4):
                try:
                    parquet_rows(target)
                    break
                except Exception as exc:  # noqa: BLE001
                    print(f"  [dl] shard {shard} corrupt on verify (attempt {attempt}/3): "
                          f"{exc.__class__.__name__}; re-downloading", flush=True)
                    target.unlink(missing_ok=True)
                    download_file(url, target, expected_size(url))
                    if attempt == 3:
                        raise
            omni_files.append(target)
        for tar in cc12m_tars_idx:
            url = cc12m_url(tar)
            target = stage_dir / f"cc12m_{tar:04d}.tar"
            print(f"  [dl] cc12m tar {tar}/{CC12M_TARS - 1} ...", flush=True)
            download_file(url, target, expected_size(url))
            cc12m_files.append(target)
        omni_rows = sum(parquet_rows(f) for f in omni_files)
        cc12m_rows = sum(tar_rows(f) for f in cc12m_files)
        print(f"  [data] stage {idx}: omni rows={omni_rows}, cc12m rows={cc12m_rows}", flush=True)
        return omni_files, cc12m_files, omni_rows, cc12m_rows

    class Prefetch:
        """Downloads the next stage's data on a daemon thread during training."""

        def __init__(self) -> None:
            self._result: StageData | None = None
            self._error: BaseException | None = None

        def start(self, idx: int) -> Prefetch:
            def _run() -> None:
                try:
                    self._result = prepare_stage(idx)
                except BaseException as exc:  # noqa: BLE001 — surfaced on join()
                    self._error = exc
            self._thread = threading.Thread(target=_run, daemon=True)
            self._thread.start()
            return self

        def join(self) -> StageData:
            # Block until the background download finishes, then surface its
            # error or result. A plain assert here (old code) crashed the whole
            # runner if training finished before the prefetch thread: the assert
            # fires while _error/_result are both still None, the daemon thread
            # dies with the process, and its curl keeps writing the stage dir —
            # so a later sync-fallback prepare_stage runs curl against the same
            # file in parallel, tearing it. Waiting instead guarantees the
            # download is done before we build/train on its data.
            self._thread.join()
            if self._error is not None:
                raise self._error
            return self._result  # type: ignore[return-value]

    # Prime the first stage synchronously: nothing is training yet, so overlap is
    # impossible — the downloads would sit idle on the critical path.
    current = prepare_stage(args.from_stage)

    for idx in range(args.from_stage, len(stage_plan)):
        if args.max_stages and idx - args.from_stage >= args.max_stages:
            print(f"[staged] max_stages reached at stage {idx}; stopping", flush=True)
            break
        stage_dir = staging_root / f"stage_{idx:02d}"
        omni_files, cc12m_files, omni_rows, cc12m_rows = current

        print(f"=== stage {idx}/{len(stage_plan) - 1} ===", flush=True)
        stage_cfg = build_stage_config(
            base, output_root, idx,
            [str(f) for f in omni_files], omni_rows,
            [str(f) for f in cc12m_files], cc12m_rows,
            stage_start_step, max_steps,
        )
        # Pre-download the next stage on a background thread while this stage
        # trains, so the mirror's ~40 min/stage never idles the GPUs.
        prefetch: Prefetch | None = None
        if idx + 1 < len(stage_plan):
            prefetch = Prefetch().start(idx + 1)
            print(f"  [prefetch] stage {idx + 1} downloading in background", flush=True)

        print(f"  [run] launching stage training -> {output_root}", flush=True)
        completed = subprocess.run(
            [
                python, "-m", "umm_uniquery.resilient_launch",
                "--config", str(stage_cfg),
                "--nproc-per-node", str(args.nproc),
            ],
            cwd=ROOT, env=env,
        )
        if completed.returncode != 0:
            if prefetch is not None:
                prefetch.join()
            raise SystemExit(
                f"[staged] stage {idx} training FAILED (exit {completed.returncode}); "
                f"raw data kept at {stage_dir}; resume with --from-stage {idx}"
            )

        ckpt = latest_checkpoint(output_root)
        if ckpt is None:
            raise SystemExit(f"[staged] stage {idx} finished but no resumable checkpoint in {output_root}")
        new_step = read_global_step(ckpt)
        quota_omni = (omni_rows // 16) * 16
        quota_cc12m = (cc12m_rows // 16) * 16
        expected_step = stage_start_step + (quota_omni + quota_cc12m) // effective_batch
        if new_step != expected_step:
            print(f"  [warn] expected global_step {expected_step}, got {new_step}", flush=True)
        stage_start_step = new_step
        print(f"  [staged] stage {idx} done -> global_step {new_step}", flush=True)

        shutil.rmtree(stage_dir, ignore_errors=True)
        print(f"  [clean] deleted {stage_dir}", flush=True)

        # Take the pre-downloaded next stage; if the background download failed
        # (mirror), fall back to a synchronous retry before training it.
        if prefetch is not None:
            try:
                current = prefetch.join()
            except BaseException:
                print(f"  [prefetch] stage {idx + 1} background download failed; "
                      f"retrying synchronously", flush=True)
                current = prepare_stage(idx + 1)

    print(f"[staged] complete: global_step {stage_start_step}", flush=True)


if __name__ == "__main__":
    main()
