from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from transformers import TrainerCallback


_CHECKPOINT_PATTERN = re.compile(r"^checkpoint-(\d+)$")


def _checkpoint_step(path: Path) -> int:
    match = _CHECKPOINT_PATTERN.match(path.name)
    return int(match.group(1)) if match else -1


def is_resumable_checkpoint(path: str | Path) -> bool:
    """Reject incomplete checkpoints left by OOM/process death during a save."""

    path = Path(path)
    if not path.is_dir() or _checkpoint_step(path) < 0:
        return False
    core_files = (
        path / "adapter_model.safetensors",
        path / "trainer_state.json",
    )
    if not all(candidate.is_file() for candidate in core_files):
        return False
    if (path / "checkpoint_complete.json").is_file():
        return True
    required = (
        path / "scheduler.pt",
    )
    if not all(candidate.is_file() for candidate in required):
        return False
    optimizer_saved = (path / "optimizer.pt").is_file() or any(
        child.is_dir() and child.name.startswith("global_step") for child in path.iterdir()
    )
    return optimizer_saved


def find_latest_resumable_checkpoint(output_dir: str | Path) -> str | None:
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        return None
    candidates = sorted(
        (
            child
            for child in output_dir.iterdir()
            if child.is_dir() and _CHECKPOINT_PATTERN.match(child.name)
        ),
        key=_checkpoint_step,
        reverse=True,
    )
    for candidate in candidates:
        if is_resumable_checkpoint(candidate):
            return str(candidate.resolve())
    return None


class CheckpointCompletionCallback(TrainerCallback):
    """Writes a completion marker only after Trainer saved every checkpoint component."""

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return control
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "global_step": state.global_step,
            "complete": True,
        }
        temporary = checkpoint_dir / "checkpoint_complete.json.tmp"
        marker = checkpoint_dir / "checkpoint_complete.json"
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(marker)
        return control
