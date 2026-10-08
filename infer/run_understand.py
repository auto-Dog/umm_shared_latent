#!/usr/bin/env python3
"""Multimodal *understanding* with the MetaQuery (cc12m PT) adapter.

Key mechanism: inject the learnable query tokens (BOI + 256 queries + EOI)
DIRECTLY into the PreFill embedding stream, at the exact position they were
trained at -- right after "<|im_start|>assistant\n". The ViT image tokens sit
earlier (user turn, <IMG_CONTEXT> positions), and causal attention means the
query tokens attend to image + question during prefill, so the model can
"benefit from" the image tokens when it starts writing the answer. Text answers
are then generated autoregressively with model.ivl3.generate() (the official
InternVL3 wrapper, which injects ViT features internally).

A no-query baseline (plain VQA format) is run for the same cases to show the
effect of the query injection.

Results are written to outputs/infer_cfg10_understand/.
"""
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CFG10_GPU", "1")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torchvision.transforms as T
from PIL import Image
from umm_uniquery.config import load_config
from umm_uniquery.modeling import UniQueryInternVL3Model
from umm_uniquery.modeling.internvl3_model import _INTERNVL3_SYSTEM, _INTERNVL3_USER

PROJECT = "/home/mingjun/umm_uniquery"
CHECKPOINT = sys.argv[1] if len(sys.argv) > 1 else "/home/mingjun/models/t2i_cc12m_pt_1007_16667"
OUTPUT_DIR = Path(PROJECT) / "outputs" / "infer_cfg10_understand"
EXAMPLE_IMAGES = Path("/home/mingjun/OpenUni/OpenUni_MME/models/InternVL3-1B/examples")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"

UNDERSTAND_SYSTEM = (
    "You are a helpful, harmless multimodal assistant. Answer the user's "
    "question about the provided image concisely and accurately."
)
SEP_TOKEN = "<|im_end|>"


# ---- image preprocessing (identical to run_mm.py) ----
def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff, best_ratio, area = float("inf"), (1, 1), width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff, best_ratio = ratio_diff, ratio
        elif ratio_diff == best_ratio_diff:
            if abs(ratio[0] * ratio[1] - area) < abs(best_ratio[0] * best_ratio[1] - area):
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1) for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed = []
    for i in range(blocks):
        box = (
            (i % target_aspect_ratio[0]) * image_size,
            (i // target_aspect_ratio[0]) * image_size,
            ((i % target_aspect_ratio[0]) + 1) * image_size,
            ((i // target_aspect_ratio[0]) + 1) * image_size,
        )
        processed.append(resized_img.crop(box))
    if use_thumbnail and len(processed) != 1:
        processed.append(image.resize((image_size, image_size)))
    return processed


def pil_to_pixel_values(image, input_size=448, max_num=12):
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, max_num=max_num)
    return torch.stack([transform(img) for img in images])


@torch.no_grad()
def answer(model, image, question, use_query=True, max_new_tokens=128):
    """One (image, question) -> text answer via PreFill injection.

    position (env UNDERSTAND_POS):
      "assistant" -- query_suffix appended after "<|im_start|>assistant\n"
                     (their T2I-trained position)
      "user"      -- query_suffix appended at the end of the user turn,
                     right before "<|im_end|>" (after image + question)
    """
    position = os.environ.get("UNDERSTAND_POS", "assistant")
    device = next(model.connector.parameters()).device
    dtype = next(model.connector.parameters()).dtype
    eos_id = model.tokenizer.convert_tokens_to_ids(SEP_TOKEN)
    gen_cfg = model.ivl3.language_model.generation_config
    gen_cfg.pad_token_id = eos_id
    gen_cfg.eos_token_id = eos_id

    # image -> ViT-compatible pixel values (dynamic 448 preprocess)
    pv = pil_to_pixel_values(image)
    num_patches = pv.shape[0]
    pixel_values = pv.to(device, dtype=dtype)

    if use_query and position == "user":
        # queries sit after image + question inside the user turn: causal
        # attention still lets them see both, then the assistant turn is a
        # normal text continuation.
        user_input = f"<image>\n{question}" + model.query_suffix
    else:
        user_input = f"<image>\n{question}"
    base = _INTERNVL3_SYSTEM.format(system=UNDERSTAND_SYSTEM) + _INTERNVL3_USER.format(input=user_input)
    if use_query and position == "assistant":
        # learnable query tokens injected right after "<|im_start|>assistant\n"
        # (their trained position) -> attend to image + question at prefill.
        base += model.query_suffix
    query = base
    image_tokens = (
        "<img>"
        + IMG_CONTEXT_TOKEN * model.ivl3.num_image_token * num_patches
        + "</img>"
    )
    query = query.replace("<image>", image_tokens, 1)

    encoded = model.tokenizer(query, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    n_prefill = input_ids.shape[1]

    outputs = model.ivl3.generate(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_id,
        pad_token_id=eos_id,
    )
    text = model.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    answer_text = text.split(SEP_TOKEN)[0].strip()
    return {
        "answer": answer_text,
        "use_query": use_query,
        "num_patches": num_patches,
        "prefill_tokens": n_prefill,
        "gen_tokens": outputs.shape[-1],
    }


def main() -> None:
    config = load_config(Path(PROJECT) / "configs" / "local_pt_ivl3_cfg10.yaml")
    model = UniQueryInternVL3Model(config["model"])
    model.load_adapter(CHECKPOINT)
    model.to("cuda").eval()
    model.ivl3.img_context_token_id = model.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    print(f"GPU: {torch.cuda.get_device_name(0)} | num_image_token: {model.ivl3.num_image_token} | queries: {model.num_queries}")

    cases = [
        ("image1.jpg", "What animal is in the image?", "red panda"),
        ("image1.jpg", "What is the animal sitting on?", "a wooden board / platform"),
        ("image1.jpg", "What color is the animal's fur?", "reddish-brown / orange"),
        ("image2.jpg", "What is happening in the image?", "a person feeding a panda"),
        ("image2.jpg", "What is the person giving the panda?", "bamboo"),
    ]

    position = os.environ.get("UNDERSTAND_POS", "assistant")
    results = []
    for img_name, question, expected in cases:
        image = Image.open(EXAMPLE_IMAGES / img_name)
        with_query = answer(model, image, question, use_query=True)
        no_query = answer(model, image, question, use_query=False)
        entry = {
            "image": str(EXAMPLE_IMAGES / img_name),
            "question": question,
            "expected": expected,
            "answer_with_query": with_query["answer"],
            "answer_baseline": no_query["answer"],
            "num_patches": with_query["num_patches"],
            "prefill_tokens_with_query": with_query["prefill_tokens"],
            "prefill_tokens_baseline": no_query["prefill_tokens"],
        }
        results.append(entry)
        print(f"\n=== {img_name} | q: {question}")
        print(f"  [with query ] {entry['answer_with_query']!r}")
        print(f"  [baseline    ] {entry['answer_baseline']!r}")
        print(f"  [expected    ] {expected!r}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = "sft" if "sft" in CHECKPOINT else "cc12m"
    out_file = OUTPUT_DIR / f"results_{tag}_{position}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"checkpoint": CHECKPOINT, "position": position, "cases": results},
                  f, ensure_ascii=False, indent=2)
    print(f"\nsaved -> {out_file}")


if __name__ == "__main__":
    main()
