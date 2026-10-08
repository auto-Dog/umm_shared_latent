"""T2I inference with the cc12m PT adapter (pt_ivl3_cfg10 / step 16667).

Loads the InternVL3-backbone UniQuery model + adapter weights and generates
512x512 images. Mirrors `sample.py` but selects UniQueryInternVL3Model (sample.py
hardcodes the Qwen variant, which cannot load this InternVL3 checkpoint).
"""
from __future__ import annotations

import os
import sys
import time

# Must be set before importing torch.
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CFG10_GPU", "3")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from pathlib import Path

import torch
from umm_uniquery.config import load_config
from umm_uniquery.modeling import UniQueryInternVL3Model

PROJECT = "/home/mingjun/umm_uniquery"
CHECKPOINT = "/home/mingjun/models/t2i_cc12m_pt_1007_16667"
OUTPUT_DIR = Path(PROJECT) / "outputs" / "infer_cfg10_t2i"

PROMPTS = [
    "a photo of a baseball glove below an umbrella",  # the training eval probe prompt
    "a red panda sitting on a wooden board, looking at the camera",
    "a snowy mountain peak at sunrise with pink clouds",
    "an astronaut riding a horse on the moon, highly detailed",
    "一只戴着红色围巾的柯基犬在雪地里奔跑",
    "a bowl of ramen with soft-boiled egg and chashu, studio lighting",
]


def main() -> None:
    torch.set_grad_enabled(False)
    config = load_config(Path(PROJECT) / "configs" / "local_pt_ivl3_cfg10.yaml")
    model = UniQueryInternVL3Model(config["model"])
    model.load_adapter(CHECKPOINT)
    model.to("cuda").eval()

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Connector params: {sum(p.numel() for p in model.connector.parameters()):,}")
    print(f"Trainable count: {model.trainable_parameter_count:,}")
    print(f"Sana caption_channels: {model.transformer.config.caption_channels}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    images = model.generate_t2i(
        PROMPTS,
        num_inference_steps=20,
        guidance_scale=4.5,
        generator=[torch.Generator(device="cuda").manual_seed(42 + i) for i in range(len(PROMPTS))],
    )
    dt = time.time() - t0
    for idx, (prompt, image) in enumerate(zip(PROMPTS, images)):
        path = OUTPUT_DIR / f"{idx:02d}_{prompt[:40].replace('/', '_')}.png"
        image.save(path)
        print(f"[{idx}] {prompt[:60]!r} -> {path.name}  ({dt / len(PROMPTS):.1f}s/img)")
    print(f"\nTotal {len(PROMPTS)} images in {dt:.1f}s ({dt / len(PROMPTS):.1f}s/img) -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
