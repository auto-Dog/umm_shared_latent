from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image


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
    ):
        self.processor = processor
        self.image_size = image_size
        self.system_prompt = system_prompt
        self.query_suffix = query_suffix
        self.max_input_text_tokens = max_input_text_tokens

    def _truncate_text(self, text: str) -> str:
        tokenizer = self.processor.tokenizer
        token_ids = tokenizer(
            text=text, return_tensors="pt", padding=False
        ).input_ids[0, : self.max_input_text_tokens]
        return tokenizer.decode(token_ids)

    def _prompt(self, example: dict[str, Any]) -> str:
        content = [
            {"type": "image"} for _ in example["source_images"]
        ] + [{"type": "text", "text": self._truncate_text(example["prompt"])}]
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
