"""Multimodal (image-conditioned) generation with the cc12m PT adapter.

The repo's InternVL3 model variant routes only text through the LLM in
`encode_context` (the T2I path deliberately bypasses the ViT). This script adds
the image-conditioned path WITHOUT touching the repo: it extracts ViT features,
injects them into the LLM input embeddings at the <IMG_CONTEXT> positions, slices
the MetaQuery BOI..EOI span from the full hidden states, and feeds the connector
-> Sana pipeline (reusing `generate_t2i`'s pipeline construction).

NOTE: this adapter is pure CC12M text-to-image pretrained; it never saw image
input during training. The image condition is therefore out-of-distribution and
results are exploratory — the point is to see whether the frozen InternVL3 ViT
conditioning influences generation at all.
"""
from __future__ import annotations

import os

# Must be set before importing torch.
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CFG10_GPU", "3")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import time
from pathlib import Path

import torch
import torchvision.transforms as T
from diffusers import SanaPipeline
from PIL import Image
from umm_uniquery.config import load_config
from umm_uniquery.modeling import UniQueryInternVL3Model
from umm_uniquery.modeling.internvl3_model import _SanaStubTokenizer, _INTERNVL3_SYSTEM, _INTERNVL3_USER

PROJECT = "/home/mingjun/umm_uniquery"
CHECKPOINT = "/home/mingjun/models/t2i_cc12m_pt_1007_16667"
OUTPUT_DIR = Path(PROJECT) / "outputs" / "infer_cfg10_mm"
EXAMPLE_IMAGES = Path("/home/mingjun/OpenUni/OpenUni_MME/models/InternVL3-1B/examples")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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
def generate_multimodal(model, prompts, cond_images, num_inference_steps=20,
                        guidance_scale=4.5, generator=None):
    """Image-conditioned generation: (image, text) -> 512x512 image."""
    device = next(model.connector.parameters()).device
    dtype = next(model.connector.parameters()).dtype
    system_prompt = model.model_config.get(
        "generation_system_prompt",
        "You are a text-to-image generation model. Generate the image described by the user prompt.",
    )
    img_context_token_id = model.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    model.ivl3.img_context_token_id = img_context_token_id

    # ---- build tokenized prompts (batch size = len(prompts)) ----
    queries, pixel_values_list, num_patches_list = [], [], []
    for prompt, cond_image in zip(prompts, cond_images):
        pv = pil_to_pixel_values(cond_image)
        num_patches = pv.shape[0]
        pixel_values_list.append(pv)
        num_patches_list.append(num_patches)
        user_input = f"<image>\n{prompt}"
        query = (
            _INTERNVL3_SYSTEM.format(system=system_prompt)
            + _INTERNVL3_USER.format(input=user_input)
            + model.query_suffix
        )
        image_tokens = "<img>" + "<IMG_CONTEXT>" * model.ivl3.num_image_token * num_patches + "</img>"
        queries.append(query.replace("<image>", image_tokens, 1))
    encoded = model.tokenizer(queries, return_tensors="pt", padding=True)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    # ---- images -> ViT features -> LLM embeddings (mirror InternVLChatModel.forward) ----
    pixel_values = torch.cat(pixel_values_list, dim=0).to(device, dtype=dtype)
    vit_embeds = model.ivl3.extract_feature(pixel_values)
    language_model = model.ivl3.language_model
    input_embeds = language_model.get_input_embeddings()(input_ids).clone()
    B, N, C = input_embeds.shape
    flat_embeds = input_embeds.reshape(B * N, C)
    flat_ids = input_ids.reshape(B * N)
    selected = flat_ids == img_context_token_id
    n_selected = int(selected.sum().item())
    n_vit = vit_embeds.reshape(-1, C).shape[0]  # (num_patches, num_image_token, C) -> flattened rows
    print(f"[mm] {n_selected} image tokens injected, {n_vit} ViT rows "
          f"(={n_vit // model.ivl3.num_image_token} patches x {model.ivl3.num_image_token}); "
          f"num_patches_list={num_patches_list}")
    assert n_selected == n_vit, f"image token count {n_selected} != ViT rows {n_vit}"
    flat_embeds[selected] = vit_embeds.reshape(-1, C)
    hidden = language_model.model(
        inputs_embeds=flat_embeds.reshape(B, N, C),
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    ).last_hidden_state

    # ---- slice BOI..EOI query span -> connector (same as encode_context) ----
    boi_mask = input_ids == model.boi_token_id
    eoi_mask = input_ids == model.eoi_token_id
    boi_positions = torch.where(boi_mask)[1]
    eoi_positions = torch.where(eoi_mask)[1]
    seq_positions = torch.arange(N, device=device).unsqueeze(0)
    query_mask = (seq_positions > boi_positions[:, None]) & (seq_positions < eoi_positions[:, None])
    assert torch.all(query_mask.sum(dim=1) == model.num_queries)
    context = hidden[query_mask].view(B, model.num_queries, hidden.shape[-1])
    query_mask_ones = torch.ones(context.shape[:2], device=device, dtype=torch.bool)
    prompt_embeds, _ = model.connector(context, query_mask_ones)

    # ---- Sana pipeline (same tail as generate_t2i) ----
    negative = [
        _INTERNVL3_SYSTEM.format(system=system_prompt)
        + _INTERNVL3_USER.format(input="")
        + model.query_suffix
        for _ in prompts
    ]
    neg_enc = model.tokenizer(negative, return_tensors="pt", padding=True)
    neg_input_ids = neg_enc["input_ids"].to(device)
    neg_attention_mask = neg_enc["attention_mask"].to(device)
    neg_embeds, _ = model.encode_queries(input_ids=neg_input_ids, attention_mask=neg_attention_mask)

    attention = torch.ones(prompt_embeds.shape[:2], device=device, dtype=torch.bool)
    pipeline = SanaPipeline(
        transformer=model.transformer,
        scheduler=model.inference_scheduler,
        vae=model.vae,
        text_encoder=None,
        tokenizer=_SanaStubTokenizer(),
    )
    result = pipeline(
        prompt=None, negative_prompt=None,
        prompt_embeds=prompt_embeds,
        prompt_attention_mask=attention,
        negative_prompt_embeds=neg_embeds,
        negative_prompt_attention_mask=attention,
        height=512, width=512,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        complex_human_instruction=None,
        use_resolution_binning=False,
        output_type="pil",
    )
    return result.images


def main() -> None:
    torch.set_grad_enabled(False)
    config = load_config(Path(PROJECT) / "configs" / "local_pt_ivl3_cfg10.yaml")
    model = UniQueryInternVL3Model(config["model"])
    model.load_adapter(CHECKPOINT)
    model.to("cuda").eval()
    print(f"GPU: {torch.cuda.get_device_name(0)} | num_image_token: {model.ivl3.num_image_token}")

    # (image, prompt) pairs — the prompt semantics are the same as T2I; the image is
    # an extra condition the model was NOT trained on.
    cases = [
        (EXAMPLE_IMAGES / "image1.jpg", "a red panda sitting on a wooden board, looking at the camera"),
        (EXAMPLE_IMAGES / "image2.jpg", "a person feeding a giant panda bamboo in a zoo"),
        (EXAMPLE_IMAGES / "image1.jpg", "a cute red panda in a snowy forest, high quality"),
    ]
    prompts = [c[1] for c in cases]
    cond_images = [Image.open(c[0]).convert("RGB") for c in cases]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for i, (img_path, _) in enumerate(cases):
        cond_images[i].save(OUTPUT_DIR / f"cond_{i}.png")

    t0 = time.time()
    images = generate_multimodal(
        model, prompts, cond_images,
        num_inference_steps=20, guidance_scale=4.5,
        generator=[torch.Generator(device="cuda").manual_seed(100 + i) for i in range(len(prompts))],
    )
    dt = time.time() - t0
    for idx, (prompt, image) in enumerate(zip(prompts, images)):
        path = OUTPUT_DIR / f"mm_{idx:02d}_{prompt[:30].replace('/', '_')}.png"
        image.save(path)
        print(f"[mm {idx}] {prompt[:50]!r} -> {path.name} ({dt / len(prompts):.1f}s/img)")
    print(f"\nTotal {len(prompts)} MM images in {dt:.1f}s -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
