from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
from transformers import Trainer


class UniQueryTrainer(Trainer):
    """Trainer that checkpoints only the ~0.1B adapter, not frozen foundation weights."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._component_sums: dict[str, float] = {}
        self._component_count = 0

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ):
        outputs = model(**inputs)
        loss = outputs["loss"]
        self._component_count += 1
        for name, value in outputs.items():
            if name != "loss" and name.endswith("_loss"):
                self._component_sums[name] = self._component_sums.get(name, 0.0) + float(value)
        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        if "loss" in logs and self._component_count:
            for name, total in self._component_sums.items():
                logs[name] = total / self._component_count
            self._component_sums.clear()
            self._component_count = 0
        super().log(logs, start_time)

    def _save(self, output_dir: str | None = None, state_dict=None):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        model = self.accelerator.unwrap_model(self.model)
        model.save_adapter(output_dir)
        model.processor.save_pretrained(output_dir)
        torch.save(self.args, Path(output_dir) / "training_args.bin")
        with (Path(output_dir) / "trainable_parameters.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump({"count": model.trainable_parameter_count}, handle, indent=2)

    def _load_from_checkpoint(self, resume_from_checkpoint: str, model=None):
        target = model or self.model
        target = self.accelerator.unwrap_model(target)
        target.load_adapter(resume_from_checkpoint)
