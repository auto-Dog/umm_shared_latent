from __future__ import annotations

import io
import random
from collections.abc import Iterator
from typing import Any

from datasets import load_dataset
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info

from umm_uniquery.registry import DATA_SOURCES


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


class ExactStreamingMixture(IterableDataset):
    """Deterministic, bounded streaming mixture without materializing source datasets.

    Baseline configs intentionally use one dataloader worker. Accelerate performs the
    process-level split while this class preserves exact source quotas and restart order.
    """

    def __init__(self, sources: list[dict[str, Any]], seed: int, shuffle_buffer: int):
        super().__init__()
        self.sources = sources
        self.seed = seed
        self.shuffle_buffer = shuffle_buffer
        self.total_samples = sum(int(source["sample_count"]) for source in sources)

    def _source_iterator(self, source: dict[str, Any], worker_id: int) -> Iterator[dict[str, Any]]:
        if source["kind"] == "cc12m_wds":
            # Local WebDataset tar files are true streaming: no shard download, entries
            # are read straight from the archive (see data/sources/cc12m_wds in configs).
            stream = load_dataset(
                "webdataset",
                data_files=source["path"],
                split=source.get("split", "train"),
                streaming=True,
            )
        elif source["kind"] == "omniedit_local":
            # Stage-pinned local parquet shards: same columns as the hub rows, read
            # with zero network (see scripts/run_staged_pt.py for the staging loop).
            stream = load_dataset(
                "parquet",
                data_files=source["path"],
                split=source.get("split", "train"),
                streaming=True,
            )
        else:
            kwargs: dict[str, Any] = {
                "path": source["path"],
                "split": source.get("split", "train"),
                "streaming": True,
            }
            if source.get("name"):
                kwargs["name"] = source["name"]
            if source.get("data_files"):
                kwargs["data_files"] = source["data_files"]
            stream = load_dataset(**kwargs)
        stream = stream.shuffle(
            seed=self.seed + int(source.get("seed_offset", 0)),
            buffer_size=self.shuffle_buffer,
        )
        if worker_id:
            stream = stream.shard(num_shards=worker_id + 1, index=worker_id)
        standardize = DATA_SOURCES.get(source["kind"])
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
        iterators = [self._source_iterator(source, worker_id) for source in self.sources]
        remaining = [int(source["sample_count"]) for source in self.sources]
        rng = random.Random(self.seed)
        while sum(remaining) > 0:
            draw = rng.randrange(sum(remaining))
            cumulative = 0
            source_index = 0
            for index, count in enumerate(remaining):
                cumulative += count
                if draw < cumulative:
                    source_index = index
                    break
            try:
                item = next(iterators[source_index])
            except StopIteration as exc:
                raise RuntimeError(
                    f"Streaming source {self.sources[source_index]['path']} exhausted before its "
                    f"sample_count={self.sources[source_index]['sample_count']} quota."
                ) from exc
            remaining[source_index] -= 1
            yield item

