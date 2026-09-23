from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

from umm_uniquery.config import load_config
from umm_uniquery.modeling import UniQueryModel


def _read_prompts(path: Path) -> list[tuple[str, str]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for index, raw_line in enumerate(handle):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("{"):
                item = json.loads(line)
                records.append((str(item.get("id", index)), item["prompt"]))
            else:
                records.append((str(index), line))
    return records


def _safe_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip(".")
    return cleaned or "sample"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate T2I samples from a UniQuery adapter")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompts", required=True, help="Text or JSONL with id/prompt fields")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config)
    model = UniQueryModel(config["model"])
    model.load_adapter(args.checkpoint)
    model.to("cuda").eval()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = _read_prompts(Path(args.prompts))

    for start in range(0, len(records), args.batch_size):
        batch = records[start : start + args.batch_size]
        generators = [
            torch.Generator(device="cuda").manual_seed(args.seed + start + offset)
            for offset in range(len(batch))
        ]
        images = model.generate_t2i(
            [prompt for _, prompt in batch],
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=generators,
        )
        for (sample_id, _), image in zip(batch, images):
            image.save(output_dir / f"{_safe_id(sample_id)}.png")


if __name__ == "__main__":
    main()
