from __future__ import annotations

from collections.abc import Callable
from typing import Any


class Registry:
    """Small explicit registry used by data, loss, SFT, and RL extension points."""

    def __init__(self, name: str):
        self.name = name
        self._items: dict[str, Callable[..., Any]] = {}

    def register(self, name: str):
        def decorator(item: Callable[..., Any]):
            if name in self._items:
                raise KeyError(f"{name!r} is already registered in {self.name}")
            self._items[name] = item
            return item

        return decorator

    def get(self, name: str) -> Callable[..., Any]:
        try:
            return self._items[name]
        except KeyError as exc:
            choices = ", ".join(sorted(self._items)) or "<empty>"
            raise KeyError(f"Unknown {self.name} {name!r}; available: {choices}") from exc


DATA_SOURCES = Registry("data source")
STAGES = Registry("training stage")
REWARD_BUILDERS = Registry("RL reward builder")
SFT_FORMATTERS = Registry("SFT formatter")

