from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
from PIL import Image

# OpenUni drops the text condition to this fixed prompt with probability
# `cfg_dropout` (their `unconditional=0.1`); the CFG template's text is
# "Generate an image." in prompt_template.
_CFG_PROMPT = "Generate an image."


def _center_crop_resize(image: Image.Image, size: int) -> torch.Tensor:
    image = image.convert("RGB")
    width, height = image.size
    scale = size / min(width, height)
    resized = image.resize((round(width * scale), round(height * scale)), Image.Resampling.BICUBIC)
    left = (resized.width - size) // 2
    top = (resized.height - size) // 2
    cropped = resized.crop((left, top, left + size, top + size))
    array = np.asarray(cropped, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


class UniQueryCollator:
    def __init__(
        self,
        processor: Any,
        image_size: int,
        system_prompt: str,
        query_suffix: str,
        max_input_text_tokens: int = 256,
        cfg_dropout: float = 0.0,
    ):
        self.processor = processor
        self.image_size = image_size
        self.system_prompt = system_prompt
        self.query_suffix = query_suffix
        self.max_input_text_tokens = max_input_text_tokens
        self.cfg_dropout = cfg_dropout

    def _truncate_text(self, text: str) -> str:
        tokenizer = self.processor.tokenizer
        token_ids = tokenizer(
            text=text, return_tensors="pt", padding=False
        ).input_ids[0, : self.max_input_text_tokens]
        return tokenizer.decode(token_ids)

    def _prompt(self, example: dict[str, Any]) -> str:
        text = self._truncate_text(example["prompt"])
        if self.cfg_dropout > 0 and random.random() < self.cfg_dropout:
            text = _CFG_PROMPT
        content = [
            {"type": "image"} for _ in example["source_images"]
        ] + [{"type": "text", "text": text}]
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            {"role": "user", "content": content},
        ]
        prompt = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        return prompt + self.query_suffix

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        prompts = [self._prompt(example) for example in examples]
        # Qwen's processor consumes images in the same flattened order as the image
        # placeholders across the text batch. Nested per-example lists caused shape
        # and dispatch failures in the remote runtime.
        images = [image for example in examples for image in example["source_images"]]
        processor_kwargs: dict[str, Any] = {
            "text": prompts,
            "return_tensors": "pt",
            "padding": True,
        }
        if images:
            processor_kwargs["images"] = images
        encoded = self.processor(**processor_kwargs)
        batch = dict(encoded)
        batch["target_pixels"] = torch.stack(
            [_center_crop_resize(example["target_image"], self.image_size) for example in examples]
        )
        # Keep Trainer batches tensor-only. Accelerate's batch broadcast rejects
        # string lists, and task metadata is not consumed by the baseline loss.
        return batch


class InternVL3Collator:
    """Text-only collator for the InternVL3-1B MetaQuery backbone.

    Builds the InternVL3 chat prompt (system block, user turn, open assistant turn)
    plus the MetaQuery query_suffix and tokenizes with the plain HF tokenizer; the
    InternVL ViT is not part of the text-to-image query path. Encoding is text-only,
    so batches are already tensor-only and Accelerate-broadcast friendly.
    """

    def __init__(
        self,
        tokenizer: Any,
        image_size: int,
        query_suffix: str,
        system_prompt: str = (
            "You are a text-to-image generation model. Generate the image described "
            "by the user prompt. The image query tokens in the sequence represent "
            "the image to generate."
        ),
        max_input_text_tokens: int = 256,
        cfg_dropout: float = 0.0,
    ):
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.system_prompt = system_prompt
        self.query_suffix = query_suffix
        self.max_input_text_tokens = max_input_text_tokens
        self.cfg_dropout = cfg_dropout

    def _truncate_text(self, text: str) -> str:
        token_ids = self.tokenizer(
            text=text, return_tensors="pt", add_special_tokens=False
        ).input_ids[0, : self.max_input_text_tokens]
        return self.tokenizer.decode(token_ids)

    def _prompt(self, example: dict[str, Any]) -> str:
        user = self._truncate_text(example["prompt"])
        if self.cfg_dropout > 0 and random.random() < self.cfg_dropout:
            user = _CFG_PROMPT
        return (
            "<|im_start|>system\n"
            f"{self.system_prompt}"
            "<|im_end|>\n<|im_start|>user\n"
            f"{user}"
            "<|im_end|>\n<|im_start|>assistant\n"
            + self.query_suffix
        )

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        prompts = [self._prompt(example) for example in examples]
        encoded = self.tokenizer(
            text=prompts, return_tensors="pt", padding=True, add_special_tokens=True
        )
        batch = dict(encoded)
        batch["target_pixels"] = torch.stack(
            [_center_crop_resize(example["target_image"], self.image_size) for example in examples]
        )
        return batch
