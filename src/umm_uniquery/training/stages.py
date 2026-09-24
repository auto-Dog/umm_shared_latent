from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from umm_uniquery.data import (
    ExactStreamingMixture,
    InternVL3Collator,
    UniQueryCollator,
)
from umm_uniquery.registry import STAGES

# This is a text-to-image (generation) stage: the LLM must turn the caption into
# image query tokens, not describe an image back.
_DEFAULT_SYSTEM_PROMPT = (
    "You are a text-to-image generation model. Generate the image described by "
    "the user prompt. The image query tokens in the sequence represent the image "
    "to generate."
)


@dataclass
class StageComponents:
    dataset: ExactStreamingMixture
    collator: UniQueryCollator | InternVL3Collator


def _diffusion_stage(config: dict[str, Any], model: Any) -> StageComponents:
    data = config["data"]
    dataset = ExactStreamingMixture(
        sources=data["sources"],
        seed=int(config.get("seed", 42)),
        shuffle_buffer=int(data.get("shuffle_buffer", 10_000)),
    )
    system_prompt = data.get("system_prompt", _DEFAULT_SYSTEM_PROMPT)
    if config["model"].get("backbone") == "internvl3":
        collator = InternVL3Collator(
            tokenizer=model.tokenizer,
            image_size=int(data.get("image_size", 512)),
            system_prompt=system_prompt,
            query_suffix=model.query_suffix,
            max_input_text_tokens=int(data.get("max_input_text_tokens", 256)),
        )
    else:
        collator = UniQueryCollator(
            processor=model.processor,
            image_size=int(data.get("image_size", 512)),
            system_prompt=system_prompt,
            query_suffix=model.query_suffix,
            max_input_text_tokens=int(data.get("max_input_text_tokens", 256)),
        )
    return StageComponents(dataset=dataset, collator=collator)


STAGES.register("pt")(_diffusion_stage)
STAGES.register("dm")(_diffusion_stage)
