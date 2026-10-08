from __future__ import annotations

import glob
import io
import os
import random
import re
from collections.abc import Iterator
from typing import Any

from datasets import load_dataset
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info

from umm_uniquery.registry import DATA_SOURCES

# Shard-file extensions used to expand a Hub repo id into its ordered shard list for
# the resume fast-forward. Local glob/`data_files` lists are used verbatim; a source
# can override this with `shard_suffix`.
_SHARD_EXTENSIONS = (".tar", ".tar.gz", ".tgz", ".parquet", ".arrow")
_HF_ENDPOINT = "https://huggingface.co"


def _natural_key(name: str) -> tuple[tuple[int, Any], ...]:
    """Sort key that orders numeric chunks numerically (``...-0179`` before ``...-1000``).

    Each chunk is tagged with an int/str marker so heterogeneous names never compare an
    ``int`` against a ``str`` (which raises in Python 3).
    """
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part)
        for part in re.split(r"(\d+)", name)
    )


def _looks_like_local(path: str) -> bool:
    return path.startswith(("/", ".", "~")) or glob.has_magic(path) or os.path.exists(path)


def _hub_shard_url(repo_id: str, filename: str) -> str:
    endpoint = os.environ.get("HF_ENDPOINT", _HF_ENDPOINT).rstrip("/")
    return f"{endpoint}/datasets/{repo_id}/resolve/main/{filename}"


def _image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, bytes):
        return Image.open(io.BytesIO(value)).convert("RGB")
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        if value.get("path"):
            return Image.open(value["path"]).convert("RGB")
    if isinstance(value, str):
        return Image.open(value).convert("RGB")
    raise TypeError(f"Unsupported image value: {type(value)!r}")


def _last_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return _last_text(value[-1]) if value else ""
    return str(value)


@DATA_SOURCES.register("cc12m")
def standardize_cc12m(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "task": "t2i",
        "prompt": _last_text(row.get("txt", row.get("caption"))),
        "source_images": [],
        "target_image": _image(row.get("jpg", row.get("image"))),
    }


# Local WebDataset tar entries carry the same jpg/txt keys as the HF parquet rows.
DATA_SOURCES.register("cc12m_wds")(standardize_cc12m)


# BLIP3o-Pretrain-Long-Caption tars carry the same jpg/txt keys as CC12M
# (verified: sa_%06d.tar entries are `sa_*.jpg` + `sa_*.txt`), so the caption->image
# standardizer applies unchanged.
DATA_SOURCES.register("blip3o")(standardize_cc12m)


@DATA_SOURCES.register("omniedit")
def standardize_omniedit(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "task": "edit",
        "prompt": _last_text(row.get("edited_prompt_list", row.get("caption"))),
        "source_images": [_image(row.get("src_img", row.get("source_image")))],
        "target_image": _image(row.get("edited_img", row.get("target_image"))),
    }


# Staged runs pin OmniEdit parquet shards downloaded to local disk; the parquet
# loader reads them with zero network and yields the same columns as the hub rows.
DATA_SOURCES.register("omniedit_local")(standardize_omniedit)


@DATA_SOURCES.register("metaquery_instruct")
def standardize_metaquery(row: dict[str, Any]) -> dict[str, Any]:
    raw_sources = row.get("source_images", row.get("input_images", [])) or []
    if not isinstance(raw_sources, (list, tuple)):
        raw_sources = [raw_sources]
    return {
        "task": "instruction",
        "prompt": _last_text(row.get("prompt", row.get("caption"))),
        "source_images": [_image(image) for image in raw_sources],
        "target_image": _image(row.get("target_image", row.get("image"))),
    }


def _draw_source(rng: random.Random, remaining: list[int]) -> int:
    """Pick the next source index from the quota-weighted deterministic mixture."""
    draw = rng.randrange(sum(remaining))
    cumulative = 0
    for index, count in enumerate(remaining):
        cumulative += count
        if draw < cumulative:
            return index
    return len(remaining) - 1  # unreachable while quotas are positive


def plan_consumption(
    seed: int, quotas: list[int], start_offset: int
) -> tuple[list[int], list[int], random.Random]:
    """Replay the mixture's source-selection RNG over the first `start_offset` draws.

    Returns ``(per_source_consumed, remaining_quotas, rng)`` with ``rng`` left positioned
    exactly at draw ``start_offset``. Source selection depends only on the seed and the
    quotas (never on the image/label data), so this replay is pure Python and cheap: it
    tells each source how many samples to fast-forward by instead of draining — and
    re-decoding — the whole interleaved prefix on resume.
    """
    remaining = [int(quota) for quota in quotas]
    rng = random.Random(seed)
    counts = [0] * len(remaining)
    total = sum(remaining)
    for _ in range(max(0, min(int(start_offset), total))):
        index = _draw_source(rng, remaining)
        counts[index] += 1
        remaining[index] -= 1
    return counts, remaining, rng


class ExactStreamingMixture(IterableDataset):
    """Deterministic, bounded streaming mixture without materializing source datasets.

    Baseline configs intentionally use one dataloader worker. Accelerate performs the
    process-level split while this class preserves exact source quotas and restart order.

    ``start_offset`` is the number of *global* mixture samples already consumed by earlier
    training steps (see ``train._resolve_data_start_offset``). On resume the mixture
    replays its source-selection RNG (``plan_consumption``) to learn how many samples each
    source already contributed, then fast-forwards every source stream to that position —
    dropping whole shards where possible so the resumed run never re-downloads the
    consumed prefix. Source selection stays exact; the per-source position lands only
    *approximately* (shard granularity), which is the intended trade-off: near-zero
    restart cost for a handful of boundary samples.
    """

    def __init__(
        self,
        sources: list[dict[str, Any]],
        seed: int,
        shuffle_buffer: int,
        start_offset: int = 0,
    ):
        super().__init__()
        self.sources = sources
        self.seed = seed
        self.shuffle_buffer = shuffle_buffer
        self.total_samples = sum(int(source["sample_count"]) for source in sources)
        self.start_offset = max(0, int(start_offset))
        self._shard_cache: dict[int, list[str] | None] = {}

    def set_start_offset(self, start_offset: int) -> bool:
        """Fast-forward the next iteration by `start_offset` global samples.

        Returns ``True`` when a positive offset was applied, meaning the mixture advances
        its own streams past the consumed prefix. The caller must then tell HF Trainer NOT
        to run its own ``skip_first_batches`` (``ignore_data_skip=True``): doing both would
        skip the prefix twice and leave a hole in the training data.
        """
        self.start_offset = max(0, int(start_offset))
        return self.start_offset > 0

    def _shard_files(self, index: int, source: dict[str, Any]) -> list[str] | None:
        """Ordered shard-file list for a source, or ``None`` when it cannot be resolved.

        Resolution order: an explicit ``data_files`` list/glob, an explicit ``path``
        list/glob, then a Hub dataset repo id listed through ``HfApi`` (filtered to the
        shard extensions, or the source's ``shard_suffix``). The result is cached because
        listing a repo hits the network and the list never changes during a run.
        """
        if index in self._shard_cache:
            return self._shard_cache[index]
        raw = source.get("data_files") or source["path"]
        patterns = [raw] if isinstance(raw, str) else [str(item) for item in raw]
        suffix = source.get("shard_suffix")
        suffixes = (str(suffix),) if suffix else _SHARD_EXTENSIONS
        files: list[str] = []
        for pattern in patterns:
            if _looks_like_local(pattern):
                files.extend(sorted(glob.glob(pattern), key=_natural_key))
                continue
            try:
                from huggingface_hub import HfApi

                api = HfApi(endpoint=os.environ.get("HF_ENDPOINT", _HF_ENDPOINT))
                names = [
                    name
                    for name in api.list_repo_files(pattern, repo_type="dataset")
                    if name.endswith(suffixes)
                ]
            except Exception:  # noqa: BLE001 - any Hub failure degrades to a row-wise skip
                self._shard_cache[index] = None
                return None
            names.sort(key=_natural_key)
            files.extend(_hub_shard_url(pattern, name) for name in names)
        result = files or None
        self._shard_cache[index] = result
        return result

    def _fast_forward(
        self, index: int, source: dict[str, Any], skip_rows: int
    ) -> tuple[dict[str, Any], int]:
        """Drop fully-consumed shards, returning the rewritten source + residual row skip.

        Two triggers drop whole tar/parquet files instead of streaming past them — which
        is what avoids re-downloading the consumed prefix:

        * ``shard_start`` (explicit, per-source override): drop the first N shards and
          start there, ignoring the derived ``skip_rows``.
        * ``shard_samples`` (rows per shard) + a positive ``skip_rows``: drop the shards
          the consumed rows fully cover, then skip the remainder within the next shard.

        Without a resolvable shard list (or without either hint) this degrades to a plain
        row-wise skip over the full stream.
        """
        shard_start = source.get("shard_start")
        shard_samples = source.get("shard_samples")
        if skip_rows <= 0 and shard_start is None:
            return source, 0
        if shard_start is None and not shard_samples:
            return source, skip_rows
        files = self._shard_files(index, source)
        if not files:
            return source, skip_rows
        if shard_start is not None:
            dropped = int(shard_start)
            if dropped < 0:
                raise ValueError(f"shard_start must be non-negative, got {dropped}")
            if dropped >= len(files):
                raise ValueError(
                    f"shard_start={dropped} but only {len(files)} shards available for "
                    f"source {source['path']!r}"
                )
            residual = 0
        else:
            per_shard = max(1, int(shard_samples))
            # Keep at least one shard so the tail is never empty.
            dropped = min(skip_rows // per_shard, len(files) - 1)
            residual = max(0, skip_rows - dropped * per_shard)
        if dropped <= 0:
            return source, skip_rows
        return {**source, "data_files": files[dropped:]}, residual

    def _load_stream(self, source: dict[str, Any]):
        data_files = source.get("data_files")
        if source["kind"] == "cc12m_wds":
            # Local WebDataset tar files are true streaming: no shard download, entries
            # are read straight from the archive (see data/sources/cc12m_wds in configs).
            return load_dataset(
                "webdataset",
                data_files=data_files or source["path"],
                split=source.get("split", "train"),
                streaming=True,
            )
        if source["kind"] == "omniedit_local":
            # Stage-pinned local parquet shards: same columns as the hub rows, read
            # with zero network (see scripts/run_staged_pt.py for the staging loop).
            return load_dataset(
                "parquet",
                data_files=data_files or source["path"],
                split=source.get("split", "train"),
                streaming=True,
            )
        kwargs: dict[str, Any] = {
            "path": source["path"],
            "split": source.get("split", "train"),
            "streaming": True,
        }
        if source.get("name"):
            kwargs["name"] = source["name"]
        if data_files:
            kwargs["data_files"] = data_files
        return load_dataset(**kwargs)

    def _build_iterator(
        self, index: int, source: dict[str, Any], worker_id: int, skip_rows: int
    ) -> Iterator[dict[str, Any]]:
        source, residual = self._fast_forward(index, source, skip_rows)
        stream = self._load_stream(source)
        stream = stream.shuffle(
            seed=self.seed + int(source.get("seed_offset", 0)),
            buffer_size=self.shuffle_buffer,
        )
        if worker_id:
            stream = stream.shard(num_shards=worker_id + 1, index=worker_id)
        if residual > 0:
            stream = stream.skip(residual)
        standardize = DATA_SOURCES.get(source["kind"])
        # Standardising inside the source iterator (and dropping bad rows here) keeps the
        # outer RNG drawing exactly once per yielded item, so `plan_consumption` stays
        # aligned with the live loop.
        for row in stream:
            try:
                yield standardize(row)
            except (OSError, TypeError, ValueError, KeyError):
                continue

    def __iter__(self) -> Iterator[dict[str, Any]]:
        worker = get_worker_info()
        if worker is not None and worker.num_workers != 1:
            raise RuntimeError(
                "ExactStreamingMixture requires dataloader_num_workers=0 for deterministic "
                "Accelerate sharding; use dataset-side WebDataset shards to scale I/O."
            )
        worker_id = 0 if worker is None else worker.id
        quotas = [int(source["sample_count"]) for source in self.sources]
        # Replay the selection RNG to learn how many samples each source already
        # contributed before `start_offset`, then fast-forward every stream to that
        # position. `rng` and `remaining` carry the correct state into the live loop.
        counts, remaining, rng = plan_consumption(self.seed, quotas, self.start_offset)
        iterators = [
            self._build_iterator(index, source, worker_id, counts[index])
            for index, source in enumerate(self.sources)
        ]
        while sum(remaining) > 0:
            source_index = _draw_source(rng, remaining)
            try:
                item = next(iterators[source_index])
            except StopIteration as exc:
                raise RuntimeError(
                    f"Streaming source {self.sources[source_index]['path']} exhausted before its "
                    f"sample_count={self.sources[source_index]['sample_count']} quota."
                ) from exc
            remaining[source_index] -= 1
            yield item

