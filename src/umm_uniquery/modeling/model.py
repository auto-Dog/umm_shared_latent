from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from diffusers import (
    AutoencoderKL,
    AutoencoderDC,
    DPMSolverMultistepScheduler,
    FlowMatchEulerDiscreteScheduler,
    SanaPipeline,
    SanaTransformer2DModel,
)
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from safetensors.torch import load_file, save_file
from torch import nn
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from .connector import build_connector
from .losses import LatentAlignmentHead, query_diversity_loss


class UniQueryModel(nn.Module):
    """Frozen Qwen2.5-VL and Sana joined by learnable queries and a light connector."""

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.config = config
        dtype = getattr(torch, config.get("torch_dtype", "bfloat16"))
        attention_backend = config.get("attention_backend", "sdpa")

        self.mllm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config["mllm_id"],
            torch_dtype=dtype,
            attn_implementation=attention_backend,
        )
        self.processor = AutoProcessor.from_pretrained(
            config["mllm_id"],
            min_pixels=config.get("min_pixels", 256 * 28 * 28),
            max_pixels=min(int(config.get("max_pixels", 1_000_000)), 1_000_000),
        )
        self.processor.tokenizer.padding_side = "left"
        self.image_token_id = int(
            getattr(
                self.mllm.config,
                "image_token_id",
                self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>"),
            )
        )
        self.num_queries = int(config["connector"].get("num_queries", 256))
        tokenizer = self.processor.tokenizer
        original_vocab_size = self.mllm.get_input_embeddings().num_embeddings
        # Keep MetaQuery's exact vocabulary construction order: first reserve model
        # rows for BOI/EOI/query tokens, then bring the tokenizer up to the model's
        # original vocabulary size, and finally append the MetaQuery special tokens.
        try:
            self.mllm.resize_token_embeddings(
                original_vocab_size + self.num_queries + 2,
                mean_resizing=False,
            )
        except TypeError:
            self.mllm.resize_token_embeddings(original_vocab_size + self.num_queries + 2)
        if len(tokenizer) < original_vocab_size:
            tokenizer.add_special_tokens(
                {
                    "additional_special_tokens": [
                        f"<pad_token_{index}>"
                        for index in range(original_vocab_size - len(tokenizer))
                    ]
                }
            )
        metaquery_tokens = (
            ["<begin_of_img>", "<end_of_img>"]
            + [f"<img{index}>" for index in range(self.num_queries)]
        )
        tokenizer.add_special_tokens(
            {"additional_special_tokens": metaquery_tokens}
        )
        if len(tokenizer) != self.mllm.get_input_embeddings().num_embeddings:
            raise ValueError(
                "MetaQuery tokenizer/model vocabulary mismatch after adding special tokens: "
                f"tokenizer={len(tokenizer)}, model="
                f"{self.mllm.get_input_embeddings().num_embeddings}"
            )
        self.boi_token_id = tokenizer.convert_tokens_to_ids("<begin_of_img>")
        self.eoi_token_id = tokenizer.convert_tokens_to_ids("<end_of_img>")
        query_token_ids = tokenizer.convert_tokens_to_ids(
            [f"<img{index}>" for index in range(self.num_queries)]
        )
        trainable_token_ids = [self.boi_token_id, self.eoi_token_id] + query_token_ids
        expected_ids = list(
            range(trainable_token_ids[0], trainable_token_ids[0] + len(trainable_token_ids))
        )
        if trainable_token_ids != expected_ids:
            raise ValueError("MetaQuery special token IDs must be contiguous")
        self.metaquery_token_start = trainable_token_ids[0]
        self.metaquery_token_end = trainable_token_ids[-1]
        if self.metaquery_token_start != original_vocab_size:
            raise ValueError(
                "MetaQuery special tokens must occupy the newly resized embedding rows"
            )
        self.query_suffix = (
            "\n<begin_of_img>"
            + "".join(f"<img{index}>" for index in range(self.num_queries))
            + "<end_of_img><|im_end|>"
        )
        sana_id = config["sana_id"]
        self.transformer = SanaTransformer2DModel.from_pretrained(
            sana_id, subfolder="transformer", torch_dtype=dtype
        )
        vae_id = config["vae_id"]
        if "sana" in vae_id.lower() or "dc-ae" in vae_id.lower():
            try:
                self.vae = AutoencoderDC.from_pretrained(vae_id, torch_dtype=dtype)
            except (OSError, ValueError):
                self.vae = AutoencoderDC.from_pretrained(
                    vae_id, subfolder="vae", torch_dtype=dtype
                )
        else:
            try:
                self.vae = AutoencoderKL.from_pretrained(vae_id, torch_dtype=dtype)
            except (OSError, ValueError):
                self.vae = AutoencoderKL.from_pretrained(
                    vae_id, subfolder="vae", torch_dtype=dtype
                )
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            sana_id, subfolder="scheduler"
        )
        self.inference_scheduler = DPMSolverMultistepScheduler.from_pretrained(
            sana_id, subfolder="scheduler"
        )

        self.mllm.requires_grad_(False)
        self.transformer.requires_grad_(False)
        self.vae.requires_grad_(False)
        # This is the original MetaQuery learnable-query mechanism. The resized
        # embedding parameter participates in autograd, while the hook zeros every
        # pre-existing Qwen row so only BOI/EOI/<img_i> rows learn.
        embedding_weight = self.mllm.get_input_embeddings().weight
        embedding_weight.requires_grad_(True)

        def freeze_qwen_embedding_rows(gradient: torch.Tensor) -> torch.Tensor:
            gradient[:original_vocab_size].zero_()
            return gradient

        embedding_weight.register_hook(freeze_qwen_embedding_rows)
        self.mllm.config.use_cache = False
        self.mllm.model.config.use_sliding_window = False
        self.mllm.model.config.sliding_window = None
        # MetaQuery exposes the final Qwen hidden states through `.logits` and avoids
        # materializing vocabulary logits during diffusion training.
        self.mllm.lm_head = nn.Identity()

        context_size = int(self.mllm.config.hidden_size)
        output_size = int(self.transformer.config.caption_channels)
        self.connector = build_connector(config["connector"], context_size, output_size)

        loss_config = config.get("losses", {})
        self.flow_weight = float(loss_config.get("flow_weight", 1.0))
        self.diversity_weight = float(loss_config.get("query_diversity_weight", 0.0))
        self.alignment_weight = float(loss_config.get("latent_alignment_weight", 0.0))
        self.alignment_head = None
        if self.alignment_weight > 0:
            query_size = int(config["connector"]["hidden_size"])
            latent_channels = int(getattr(self.transformer.config, "in_channels", 32))
            self.alignment_head = LatentAlignmentHead(query_size, latent_channels)

        if config.get("gradient_checkpointing", True):
            self.mllm.gradient_checkpointing_enable({"use_reentrant": False})
            self.transformer.enable_gradient_checkpointing()

    def train(self, mode: bool = True):
        super().train(mode)
        # The frozen VAE needs no backward graph. Qwen/Sana remain in training mode so
        # their checkpointed activations can pass gradients to Query/Connector inputs.
        self.vae.eval()
        return self

    @property
    def trainable_parameter_count(self) -> int:
        # Count the effective trainable rows, not the frozen prefix of the single
        # dense embedding Parameter retained for MetaQuery compatibility.
        embedding_weight = self.mllm.get_input_embeddings().weight
        count = (self.num_queries + 2) * embedding_weight.shape[1]
        embedding_id = id(embedding_weight)
        count += sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad and id(parameter) != embedding_id
        )
        return count

    def encode_context(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        qwen_pixel_values: torch.Tensor | list[torch.Tensor] | None = None,
        qwen_image_grid_thw: torch.Tensor | list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": False,
            "return_dict": True,
        }
        qwen_pixel_values, qwen_image_grid_thw = self._prepare_qwen_vision_inputs(
            input_ids=input_ids,
            pixel_values=qwen_pixel_values,
            image_grid_thw=qwen_image_grid_thw,
        )
        if qwen_pixel_values is not None:
            kwargs["pixel_values"] = qwen_pixel_values
            kwargs["image_grid_thw"] = qwen_image_grid_thw
        output = self.mllm(**kwargs)
        hidden_states = output.logits

        boi_mask = input_ids == self.boi_token_id
        eoi_mask = input_ids == self.eoi_token_id
        if not torch.all(boi_mask.sum(dim=1) == 1) or not torch.all(
            eoi_mask.sum(dim=1) == 1
        ):
            raise ValueError("Each prompt must contain exactly one MetaQuery BOI and EOI token")
        boi_positions = torch.where(boi_mask)[1]
        eoi_positions = torch.where(eoi_mask)[1]
        sequence_positions = torch.arange(
            input_ids.shape[1], device=input_ids.device
        ).unsqueeze(0)
        query_mask = (sequence_positions > boi_positions[:, None]) & (
            sequence_positions < eoi_positions[:, None]
        )
        counts = query_mask.sum(dim=1)
        if not torch.all(counts == self.num_queries):
            raise ValueError(
                f"Each MetaQuery span must contain {self.num_queries} tokens; got {counts.tolist()}"
            )
        return hidden_states[query_mask].view(
            input_ids.shape[0], self.num_queries, hidden_states.shape[-1]
        )

    def _prepare_qwen_vision_inputs(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor | list[torch.Tensor] | None,
        image_grid_thw: torch.Tensor | list[torch.Tensor] | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if pixel_values is None and image_grid_thw is None:
            if torch.any(input_ids == self.image_token_id):
                raise ValueError("Qwen input_ids contain image tokens but pixel_values are missing")
            return None, None
        if pixel_values is None or image_grid_thw is None:
            raise ValueError("Qwen pixel_values and image_grid_thw must be provided together")

        if isinstance(pixel_values, (list, tuple)):
            if not pixel_values:
                raise ValueError("Qwen pixel_values list must not be empty")
            chunks = []
            for value in pixel_values:
                if value.ndim == 3 and value.shape[0] == 1:
                    value = value.squeeze(0)
                if value.ndim != 2:
                    raise ValueError(
                        "Each processed Qwen pixel_values tensor must have shape "
                        "[num_patches, patch_dim]"
                    )
                chunks.append(value)
            pixel_values = torch.cat(chunks, dim=0)
        elif pixel_values.ndim == 3 and pixel_values.shape[0] == 1:
            pixel_values = pixel_values.squeeze(0)
        if pixel_values.ndim != 2:
            raise ValueError(
                "Qwen pixel_values must be processor output with shape "
                "[num_patches, patch_dim], not raw [batch, channels, height, width] images"
            )

        if isinstance(image_grid_thw, (list, tuple)):
            if not image_grid_thw:
                raise ValueError("Qwen image_grid_thw list must not be empty")
            image_grid_thw = torch.cat(
                [value.reshape(-1, 3) for value in image_grid_thw], dim=0
            )
        image_grid_thw = image_grid_thw.reshape(-1, 3).to(
            device=input_ids.device, dtype=torch.long
        )
        patches_per_image = image_grid_thw.prod(dim=1)
        expected_patches = int(patches_per_image.sum().item())
        if pixel_values.shape[0] != expected_patches:
            raise ValueError(
                "Qwen pixel/grid mismatch: "
                f"pixel_values has {pixel_values.shape[0]} patches, "
                f"but image_grid_thw describes {expected_patches}"
            )

        patch_size = int(self.mllm.config.vision_config.patch_size)
        spatial_pixels = image_grid_thw[:, 1] * image_grid_thw[:, 2] * (patch_size**2)
        if torch.any(spatial_pixels > 1_000_000):
            raise ValueError(
                "Qwen image exceeds the 1,000,000 pixel hard limit after processor resize: "
                f"{spatial_pixels.tolist()}"
            )

        spatial_merge = int(self.mllm.config.vision_config.spatial_merge_size)
        merge_area = spatial_merge**2
        if torch.any(patches_per_image % merge_area != 0):
            raise ValueError(
                "Each Qwen image grid must be divisible by spatial_merge_size squared"
            )
        expected_image_tokens = int((patches_per_image // merge_area).sum().item())
        actual_image_tokens = int((input_ids == self.image_token_id).sum().item())
        if actual_image_tokens != expected_image_tokens:
            raise ValueError(
                "Qwen image token/grid mismatch: "
                f"input_ids has {actual_image_tokens} image tokens, "
                f"but image_grid_thw requires {expected_image_tokens}"
            )

        visual = getattr(self.mllm, "visual", None)
        if visual is None:
            visual = getattr(self.mllm.model, "visual", None)
        if visual is None:
            raise AttributeError(
                "Qwen2.5-VL visual encoder was not found on the loaded Transformers model"
            )
        vision_dtype = next(visual.parameters()).dtype
        pixel_values = pixel_values.to(device=input_ids.device, dtype=vision_dtype)
        return pixel_values, image_grid_thw

    @torch.no_grad()
    def pixels_to_latents(self, target_pixels: torch.Tensor) -> torch.Tensor:
        encoded = self.vae.encode(target_pixels.to(dtype=next(self.vae.parameters()).dtype))
        if isinstance(self.vae, AutoencoderKL):
            latents = encoded.latent_dist.sample()
        elif isinstance(self.vae, AutoencoderDC):
            latents = encoded.latent
        else:
            raise ValueError(f"Unsupported VAE type: {type(self.vae)}")
        shift = getattr(self.vae.config, "shift_factor", None)
        if shift is not None:
            latents = latents - shift
        return latents * self.vae.config.scaling_factor

    def encode_queries(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        qwen_pixel_values: torch.Tensor | list[torch.Tensor] | None = None,
        qwen_image_grid_thw: torch.Tensor | list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context = self.encode_context(
            input_ids,
            attention_mask,
            qwen_pixel_values,
            qwen_image_grid_thw,
        )
        query_mask = torch.ones(
            context.shape[:2], device=context.device, dtype=torch.bool
        )
        return self.connector(context, query_mask)

    def _get_sigmas(self, timesteps: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        schedule_timesteps = self.noise_scheduler.timesteps.to(timesteps.device)
        sigmas = self.noise_scheduler.sigmas.to(device=timesteps.device, dtype=latents.dtype)
        indices = [(schedule_timesteps == timestep).nonzero().item() for timestep in timesteps]
        sigma = sigmas[indices].flatten()
        while sigma.ndim < latents.ndim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def compute_flow_loss(
        self,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        noise = torch.randn_like(latents)
        batch_size = latents.shape[0]
        weighting_scheme = "uniform"
        density = compute_density_for_timestep_sampling(
            weighting_scheme=weighting_scheme,
            batch_size=batch_size,
            logit_mean=0.0,
            logit_std=1.0,
            mode_scale=1.29,
        )
        indices = (density * self.noise_scheduler.config.num_train_timesteps).long()
        timesteps = self.noise_scheduler.timesteps[indices].to(latents.device)
        sigmas = self._get_sigmas(timesteps, latents)
        noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
        prediction = self.transformer(
            hidden_states=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_attention_mask,
            return_dict=False,
        )[0]
        target = noise - latents
        weighting = compute_loss_weighting_for_sd3(
            weighting_scheme=weighting_scheme, sigmas=sigmas
        )
        return (
            (weighting.float() * (prediction.float() - target.float()).square())
            .reshape(batch_size, -1)
            .mean(dim=1)
            .mean()
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_pixels: torch.Tensor,
        qwen_pixel_values: torch.Tensor | list[torch.Tensor] | None = None,
        qwen_image_grid_thw: torch.Tensor | list[torch.Tensor] | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        if qwen_pixel_values is not None and pixel_values is not None:
            raise ValueError("Pass qwen_pixel_values or legacy pixel_values, not both")
        if qwen_image_grid_thw is not None and image_grid_thw is not None:
            raise ValueError("Pass qwen_image_grid_thw or legacy image_grid_thw, not both")
        qwen_pixel_values = (
            qwen_pixel_values if qwen_pixel_values is not None else pixel_values
        )
        qwen_image_grid_thw = (
            qwen_image_grid_thw if qwen_image_grid_thw is not None else image_grid_thw
        )
        latents = self.pixels_to_latents(target_pixels)
        prompt_embeds, query_states = self.encode_queries(
            input_ids=input_ids,
            attention_mask=attention_mask,
            qwen_pixel_values=qwen_pixel_values,
            qwen_image_grid_thw=qwen_image_grid_thw,
        )
        query_mask = torch.ones(
            prompt_embeds.shape[:2], device=prompt_embeds.device, dtype=torch.bool
        )
        flow = self.compute_flow_loss(latents, prompt_embeds, query_mask)
        total = self.flow_weight * flow
        output = {"flow_loss": flow.detach()}

        if self.diversity_weight > 0:
            diversity = query_diversity_loss(query_states)
            total = total + self.diversity_weight * diversity
            output["query_diversity_loss"] = diversity.detach()
        if self.alignment_head is not None:
            alignment = self.alignment_head(query_states, latents)
            total = total + self.alignment_weight * alignment
            output["latent_alignment_loss"] = alignment.detach()
        output["loss"] = total
        return output

    @torch.no_grad()
    def generate_t2i(
        self,
        prompts: list[str],
        negative_prompt: str = "",
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 20,
        guidance_scale: float = 4.5,
        generator: torch.Generator | list[torch.Generator] | None = None,
    ) -> list[Any]:
        """Generate a T2I batch for evaluation; edit/image guidance is a later stage."""

        system_prompt = self.config.get(
            "generation_system_prompt",
            "You will be given an image or its caption. Please describe the "
            "content of the image in detail in your own words.",
        )

        def render_prompt(text: str) -> str:
            conversation = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "text", "text": text}]},
            ]
            prompt = self.processor.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
            return prompt + self.query_suffix

        positive = [render_prompt(prompt) for prompt in prompts]
        negative = [render_prompt(negative_prompt) for _ in prompts]
        encoded = self.processor(
            text=positive + negative, return_tensors="pt", padding=True
        )
        device = next(self.connector.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        prompt_embeds, _ = self.encode_queries(
            input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"]
        )
        batch_size = len(prompts)
        attention = torch.ones(
            prompt_embeds.shape[:2], device=device, dtype=torch.bool
        )
        pipeline = SanaPipeline(
            transformer=self.transformer,
            scheduler=self.inference_scheduler,
            vae=self.vae,
            text_encoder=None,
            tokenizer=None,
        )
        result = pipeline(
            prompt=None,
            negative_prompt=None,
            prompt_embeds=prompt_embeds[:batch_size],
            prompt_attention_mask=attention[:batch_size],
            negative_prompt_embeds=prompt_embeds[batch_size:],
            negative_prompt_attention_mask=attention[batch_size:],
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            complex_human_instruction=None,
            use_resolution_binning=False,
            output_type="pil",
        )
        return result.images

    def adapter_state_dict(self) -> dict[str, torch.Tensor]:
        prefixes = ("connector.", "alignment_head.")
        state = {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in self.state_dict().items()
            if name.startswith(prefixes)
        }
        state["metaquery_embeddings"] = (
            self.mllm.get_input_embeddings()
            .weight[self.metaquery_token_start : self.metaquery_token_end + 1]
            .detach()
            .cpu()
            .contiguous()
        )
        return state

    def save_adapter(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_file(self.adapter_state_dict(), str(output_dir / "adapter_model.safetensors"))
        with (output_dir / "adapter_config.json").open("w", encoding="utf-8") as handle:
            json.dump(self.config, handle, indent=2, ensure_ascii=False)

    def load_adapter(self, checkpoint_dir: str | Path, strict: bool = True) -> None:
        state = load_file(str(Path(checkpoint_dir) / "adapter_model.safetensors"))
        metaquery_embeddings = state.pop("metaquery_embeddings", None)
        # Backward-compatible with the short-lived independent-Parameter format.
        if metaquery_embeddings is None:
            metaquery_embeddings = state.pop("query_token_embeddings", None)
        if metaquery_embeddings is not None:
            expected_shape = (
                self.num_queries + 2,
                self.mllm.get_input_embeddings().weight.shape[1],
            )
            if tuple(metaquery_embeddings.shape) != expected_shape:
                raise RuntimeError(
                    "MetaQuery embedding shape mismatch; "
                    f"expected={expected_shape}, got={tuple(metaquery_embeddings.shape)}"
                )
            with torch.no_grad():
                self.mllm.get_input_embeddings().weight[
                    self.metaquery_token_start : self.metaquery_token_end + 1
                ].copy_(metaquery_embeddings)
        missing, unexpected = self.load_state_dict(state, strict=False)
        relevant_missing = [
            key
            for key in missing
            if key.startswith(("connector.", "alignment_head."))
        ]
        if strict and (metaquery_embeddings is None or relevant_missing or unexpected):
            raise RuntimeError(
                "Adapter mismatch; "
                f"missing_metaquery_embeddings={metaquery_embeddings is None}, "
                f"missing={relevant_missing}, unexpected={unexpected}"
            )
