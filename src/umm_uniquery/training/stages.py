from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from umm_uniquery.data import ExactStreamingMixture, UniQueryCollator
from umm_uniquery.registry import STAGES


@dataclass
class StageComponents:
    dataset: ExactStreamingMixture
    collator: UniQueryCollator


def _diffusion_stage(config: dict[str, Any], model: Any) -> StageComponents:
    data = config["data"]
    dataset = ExactStreamingMixture(
        sources=data["sources"],
        seed=int(config.get("seed", 42)),
        shuffle_buffer=int(data.get("shuffle_buffer", 10_000)),
    )
    collator = UniQueryCollator(
        processor=model.processor,
        image_size=int(data.get("image_size", 512)),
        system_prompt=data.get(
            "system_prompt",
            "You will be given an image or its caption. Please describe the content of the image in detail in your own words.",
        ),
        query_suffix=model.query_suffix,
        max_input_text_tokens=int(data.get("max_input_text_tokens", 256)),
    )
    return StageComponents(dataset=dataset, collator=collator)


STAGES.register("pt")(_diffusion_stage)
STAGES.register("dm")(_diffusion_stage)
