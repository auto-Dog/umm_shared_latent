from __future__ import annotations

from typing import Any, Protocol

import torch


class SFTFormatter(Protocol):
    """Contract for a future interleaved reasoning/answer supervision formatter."""

    def __call__(self, example: dict[str, Any]) -> dict[str, Any]: ...


class RewardFunction(Protocol):
    """Contract for future text, image, and modality-balance RL rewards."""

    def __call__(
        self,
        prompts: list[str],
        text_outputs: list[str],
        image_outputs: torch.Tensor | None,
        references: list[dict[str, Any]],
    ) -> torch.Tensor: ...


class StageBuilder(Protocol):
    """Contract implemented by PT/DM today and by future SFT/RL stages."""

    def __call__(self, config: dict[str, Any], model: torch.nn.Module) -> Any: ...

