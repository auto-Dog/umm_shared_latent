"""Shared helpers for single-point probe generation.

Conventions (see probes/README.md):
- Prompts live in probes/prompts/*.jsonl, one JSON object per line:
      {"id": "...", "prompt": "...", "note": "..."}
- Output goes to probes/outputs/<model>/<prompt_id>/:
      meta.jsonl          one line per generated sample (prompt, seed, params)
      <seed>.png          the generated image
- Each generator script is self-contained and takes --prompts/--out/--gpu.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_PROMPTS = Path(__file__).resolve().parents[1] / "prompts" / "umbrella_baseball.jsonl"

# Fixed seeds so runs are reproducible across models/machines.
DEFAULT_SEEDS = [42, 2026, 2027, 2028]


def load_prompts(path: Path) -> list[dict]:
    with open(path) as f:
        prompts = [json.loads(l) for l in f if l.strip()]
    if not prompts:
        raise SystemExit(f"no prompts in {path}")
    return prompts


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS,
                        help="jsonl prompt registry (default: umbrella_baseball.jsonl)")
    parser.add_argument("--out", type=Path, required=True,
                        help="output dir, e.g. probes/outputs/<model>")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                        help="generation seeds (default: 42 2026 2027 2028)")
    parser.add_argument("--gpu", type=str, default="1",
                        help="physical GPU id (parsed before torch import)")
    return parser


def write_meta(out: Path, prompt_id: str, rows: list[dict]) -> None:
    """Append one meta.jsonl line per sample under out/<prompt_id>/."""
    d = out / prompt_id
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "meta.jsonl", "a") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
