"""Generate GenEval images with the umm_uniquery cc12m T2I checkpoint.

Mirrors Janus's `generate_geneval_janus.py` output layout so the shared
GenEval eval stage (evaluate_images.py + summary_scores.py) can score it:

  <IMAGES_DIR>/%05d/metadata.jsonl          (single JSON object, json.dump)
  <IMAGES_DIR>/%05d/samples/%05d.png        (4 samples, indexed by j)

Resumable: a %05d folder that already has all N_SAMPLES images is skipped.
Seeds are a function of (prompt index, sample index), so a resumed run
regenerates partial folders identically.

Run:  CFG10_GPU=3 python infer/generate_geneval_umm.py   (conda: uniquery)
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

# Must be set before importing torch.
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CFG10_GPU", "3")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from umm_uniquery.config import load_config
from umm_uniquery.modeling import UniQueryInternVL3Model

PROJECT = "/home/mingjun/umm_uniquery"
CHECKPOINT = "/home/mingjun/models/t2i_cc12m_pt_1007_16667"
PROMPTS_FILE = "/home/mingjun/geneval/prompts/evaluation_metadata.jsonl"
IMAGES_DIR = Path(PROJECT) / "outputs" / "geneval_cc12m" / "images"

N_SAMPLES = 4        # GenEval convention
CHUNK = 4            # prompts per generate_t2i call (x4 samples = 16 gens/call)
STEPS = 20
GUIDANCE = 4.5


def main() -> None:
    with open(PROMPTS_FILE) as f:
        entries = [json.loads(line) for line in f if line.strip()]
    total = len(entries)

    config = load_config(Path(PROJECT) / "configs" / "local_pt_ivl3_cfg10.yaml")
    model = UniQueryInternVL3Model(config["model"])
    model.load_adapter(CHECKPOINT)
    model.to("cuda").eval()

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"prompts={total}  samples/prompt={N_SAMPLES}  target={total * N_SAMPLES} images")
    print(f"output: {IMAGES_DIR}")

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    done_total = 0
    for start in range(0, total, CHUNK):
        remaining = []
        for idx in range(start, min(start + CHUNK, total)):
            i = start + (idx - start)
            folder = IMAGES_DIR / f"{i:05d}"
            have = sum((folder / "samples" / f"{j:05d}.png").exists() for j in range(N_SAMPLES))
            if have >= N_SAMPLES:
                done_total += N_SAMPLES
            else:
                remaining.append((i, entries[i]))

        if not remaining:
            print(f"  prompt {start:4d}-{start + CHUNK - 1:4d}: complete, skip")
            continue

        prompts4 = []
        metas = []
        for i, e in remaining:
            for j in range(N_SAMPLES):
                prompts4.append(e["prompt"])
                metas.append((i, j))

        # Seed keyed by (prompt idx, sample idx): resumable runs stay identical.
        generator = [
            torch.Generator(device="cuda").manual_seed(42 + i * N_SAMPLES + j)
            for (i, j) in metas
        ]
        images = model.generate_t2i(
            prompts4,
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE,
            generator=generator,
        )

        for (i, j), img in zip(metas, images):
            folder = IMAGES_DIR / f"{i:05d}"
            sampdir = folder / "samples"
            sampdir.mkdir(parents=True, exist_ok=True)
            img.save(sampdir / f"{j:05d}.png")
            mpath = folder / "metadata.jsonl"
            if not mpath.exists():
                with open(mpath, "w") as f:
                    json.dump(entries[i], f)

        done_total += len(images)
        elapsed = time.time() - t0
        rate = elapsed / done_total
        eta = (total * N_SAMPLES - done_total) * rate
        print(
            f"  prompt {start:4d}-{start + len(remaining) - 1:4d}: "
            f"+{len(images)} -> {done_total}/{total * N_SAMPLES} "
            f"({rate:.2f}s/img, eta {eta / 60:.0f}m)"
        )

    dt = time.time() - t0
    print(f"\nDone {done_total} images in {dt:.0f}s ({dt / done_total:.2f}s/img) -> {IMAGES_DIR}")


if __name__ == "__main__":
    main()
