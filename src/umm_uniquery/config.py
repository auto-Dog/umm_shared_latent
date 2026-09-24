from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _read_config(path: Path, seen: set[Path]) -> dict[str, Any]:
    path = path.resolve()
    if path in seen:
        raise ValueError(f"Circular config inheritance detected at {path}")
    seen.add(path)
    with path.open("r", encoding="utf-8") as handle:
        current = yaml.safe_load(handle) or {}
    parent = current.pop("extends", None)
    if parent is None:
        return current
    base = _read_config((path.parent / parent).resolve(), seen)
    return _deep_merge(base, current)


def _parse_value(raw: str) -> Any:
    return yaml.safe_load(raw)


def _set_dotted(config: dict[str, Any], expression: str) -> None:
    if "=" not in expression:
        raise ValueError(f"Override must have KEY=VALUE form: {expression}")
    dotted_key, raw_value = expression.split("=", 1)
    cursor = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        child = cursor.setdefault(part, {})
        if not isinstance(child, dict):
            raise ValueError(f"Cannot set {dotted_key}: {part} is not a mapping")
        cursor = child
    cursor[parts[-1]] = _parse_value(raw_value)


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    config = _read_config(Path(path), set())
    for expression in overrides or []:
        _set_dotted(config, expression)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    required = ("stage", "model", "data", "training")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing required config sections: {missing}")
    if not config["data"].get("streaming", False):
        raise ValueError("This baseline requires data.streaming=true")
    sources = config["data"].get("sources", [])
    if not sources:
        raise ValueError("At least one streaming data source is required")
    for source in sources:
        if int(source.get("sample_count", 0)) <= 0:
            raise ValueError(f"source.sample_count must be positive: {source}")
    model_cfg = config["model"]
    backbone = model_cfg.get("backbone", "qwen")
    if backbone == "internvl3":
        if not model_cfg.get("ivl3_id"):
            raise ValueError("model.ivl3_id is required for the internvl3 backbone")
        if not model_cfg.get("sana_id"):
            raise ValueError("model.sana_id is required for the internvl3 backbone")
    else:
        max_pixels = int(model_cfg.get("max_pixels", 1_000_000))
        if max_pixels > 1_000_000:
            raise ValueError(
                "model.max_pixels must not exceed 1,000,000 for the Qwen2.5-VL backbone"
            )
        if max_pixels <= 0:
            raise ValueError("model.max_pixels must be positive")
        min_pixels = int(model_cfg.get("min_pixels", 0))
        if min_pixels <= 0 or min_pixels > max_pixels:
            raise ValueError(
                "model.min_pixels must be positive and no greater than model.max_pixels"
            )
    connector = model_cfg.get("connector", {})
    if connector.get("type") == "light_transformer":
        hidden = int(connector["hidden_size"])
        heads = int(connector["num_attention_heads"])
        if hidden % heads != 0:
            raise ValueError("connector.hidden_size must be divisible by num_attention_heads")
    recovery = config["training"].get("failure_recovery", {})
    if int(recovery.get("max_restarts", 0)) < 0:
        raise ValueError("training.failure_recovery.max_restarts must be non-negative")
    if float(recovery.get("monitor_interval_seconds", 1)) <= 0:
        raise ValueError("training.failure_recovery.monitor_interval_seconds must be positive")
