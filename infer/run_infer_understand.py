#!/usr/bin/env python3
"""Multimodal understanding inference for UMM(MetaQuery) adapters.

Verified injection mechanism (user-end learnable query):
    <|im_start|>system ... <|im_end|>
    <|im_start|>user\n<image tokens>\n<question>
        + <begin_of_img><img0..255><end_of_img>        <- query_suffix
    <|im_end|>
    <|im_start|>assistant\n                            <- text answer generated here

The 256 learnable query tokens sit at the END of the user turn, right before
<|im_end|>: under causal attention they attend to the ViT image tokens and the
question during PreFill ("benefit from" the image), and the assistant turn
starts with a normal text-continuation prior. This is the empirically
verified-working position -- injecting the queries inside the assistant turn
(their T2I-trained spot) degenerates the LLM into token repetition
(see outputs/infer_cfg10_understand/README.md).

Usage:
  single case:
    python infer/run_infer_understand.py --image a.jpg --question "What animal is this?"
  batch cases (JSON list of {"image": ..., "question": ..., "expected": ...}):
    python infer/run_infer_understand.py --cases cases.json --out out.json
  demo (5 example QA pairs on the InternVL3 example images):
    python infer/run_infer_understand.py --demo
  ablation without query injection (baseline VQA format):
    add --no-query

Env: CFG10_GPU selects the GPU (must be set before torch import), default 1.
"""
import argparse
import json
import os
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

PROJECT = Path("/home/mingjun/umm_uniquery")
CHECKPOINT_CC12M = "/home/mingjun/models/t2i_cc12m_pt_1007_16667"
CHECKPOINT_SFT = "/home/mingjun/models/t2i_blip3o_sft_1007_9000"
EXAMPLE_IMAGES = Path("/home/mingjun/OpenUni/OpenUni_MME/models/InternVL3-1B/examples")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
SEP_TOKEN = "<|im_end|>"

UNDERSTAND_SYSTEM = (
    "You are a helpful, harmless multimodal assistant. Answer the user's "
    "question about the provided image concisely and accurately."
)

DEMO_CASES = [
    ("image1.jpg", "What animal is in the image?", "red panda"),
    ("image1.jpg", "What is the animal sitting on?", "a wooden board / platform"),
    ("image1.jpg", "What color is the animal's fur?", "reddish-brown / orange"),
    ("image2.jpg", "What is happening in the image?", "a person feeding a panda"),
    ("image2.jpg", "What is the person giving the panda?", "bamboo"),
]


# ---- image preprocessing (identical to run_mm.py / run_understand.py) ----
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
def answer(model, image, question, use_query=True, system=UNDERSTAND_SYSTEM,
           max_new_tokens=128, do_sample=False, temperature=1.0):
    """(image, question) -> text answer. user-end query injection at PreFill."""
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

    # user turn = <image tokens> + question + (optional) query_suffix, all
    # before <|im_end|> so the queries see image+question at PreFill.
    user_input = f"<image>\n{question}"
    if use_query:
        user_input += model.query_suffix
    base = _INTERNVL3_SYSTEM.format(system=system) + _INTERNVL3_USER.format(input=user_input)
    image_tokens = (
        "<img>" + IMG_CONTEXT_TOKEN * model.ivl3.num_image_token * num_patches + "</img>"
    )
    query = base.replace("<image>", image_tokens, 1)

    encoded = model.tokenizer(query, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    n_prefill = input_ids.shape[1]

    kwargs = dict(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        do_sample=do_sample,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_id,
        pad_token_id=eos_id,
    )
    if do_sample:
        kwargs["temperature"] = temperature
    outputs = model.ivl3.generate(**kwargs)

    text = model.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    answer_text = text.split(SEP_TOKEN)[0].strip()
    return {
        "answer": answer_text,
        "use_query": use_query,
        "num_patches": num_patches,
        "prefill_tokens": n_prefill,
        "gen_tokens": outputs.shape[-1],
    }


def load_model(checkpoint):
    config = load_config(PROJECT / "configs" / "local_pt_ivl3_cfg10.yaml")
    model = UniQueryInternVL3Model(config["model"])
    model.load_adapter(checkpoint)
    model.to("cuda").eval()
    model.ivl3.img_context_token_id = model.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description="UMM multimodal understanding inference (user-end query injection)")
    ap.add_argument("--checkpoint", default=CHECKPOINT_CC12M,
                    help=f"adapter dir (default: {CHECKPOINT_CC12M}; SFT: {CHECKPOINT_SFT})")
    ap.add_argument("--image", help="image path (single-case mode, with --question)")
    ap.add_argument("--question", help="question text (single-case mode, with --image)")
    ap.add_argument("--cases", help="JSON file: list of {\"image\", \"question\", \"expected\"?}")
    ap.add_argument("--demo", action="store_true", help="run the 5 example QA pairs")
    ap.add_argument("--out", help="write results JSON to this path (else print to stdout)")
    ap.add_argument("--system", default=UNDERSTAND_SYSTEM, help="system prompt override")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--do-sample", action="store_true", help="sample instead of greedy")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--no-query", action="store_true", help="ablation: plain VQA, no query injection")
    args = ap.parse_args()

    n_modes = sum([bool(args.image), bool(args.cases), args.demo])
    if n_modes != 1:
        ap.error("exactly one of --image, --cases, --demo is required")

    if args.image and not args.question:
        ap.error("--question is required with --image")
    if args.question and not args.image:
        ap.error("--image is required with --question")

    model = load_model(args.checkpoint)
    print(f"checkpoint: {args.checkpoint}")
    print(f"GPU: {torch.cuda.get_device_name(0)} | num_image_token: {model.ivl3.num_image_token}"
          f" | queries: {model.num_queries} | use_query: {not args.no_query}")

    if args.demo:
        cases = [{"image": str(EXAMPLE_IMAGES / n), "question": q, "expected": e}
                 for n, q, e in DEMO_CASES]
    elif args.cases:
        with open(args.cases, encoding="utf-8") as f:
            cases = json.load(f)
    else:
        cases = [{"image": args.image, "question": args.question}]

    results = []
    for c in cases:
        image = Image.open(c["image"])
        r = answer(model, image, c["question"], use_query=not args.no_query,
                   system=args.system, max_new_tokens=args.max_new_tokens,
                   do_sample=args.do_sample, temperature=args.temperature)
        entry = {
            "image": c["image"],
            "question": c["question"],
            "answer": r["answer"],
            "use_query": r["use_query"],
            "num_patches": r["num_patches"],
            "prefill_tokens": r["prefill_tokens"],
            "gen_tokens": r["gen_tokens"],
        }
        if "expected" in c:
            entry["expected"] = c["expected"]
        results.append(entry)
        print(f"\n=== {Path(c['image']).name} | q: {c['question']}")
        print(f"  [answer ] {r['answer']}")
        if "expected" in c:
            print(f"  [expect ] {c['expected']}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"checkpoint": args.checkpoint, "cases": results},
                      f, ensure_ascii=False, indent=2)
        print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
