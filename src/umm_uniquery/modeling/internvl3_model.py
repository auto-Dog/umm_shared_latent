from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from diffusers import (
    AutoencoderDC,
    AutoencoderKL,
    DPMSolverMultistepScheduler,
    FlowMatchEulerDiscreteScheduler,
    SanaPipeline,
    SanaTransformer2DModel,
)
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from safetensors.torch import load_file, save_file
from torch import nn
from transformers import AutoTokenizer

from .connector import build_connector
from .internvl3.modeling_internvl_chat import InternVLChatModel
from .internvl3.modeling_intern_vit import has_flash_attn
from .losses import LatentAlignmentHead, query_diversity_loss

# Prompt preamble matching InternVL3's chat template (see the tokenizer's
# chat_template.jinja): optional system block, then the user turn and an open
# assistant turn. The MetaQuery query_suffix is appended after this.
_INTERNVL3_SYSTEM = "<|im_start|>system\n{system}<|im_end|>\n"
_INTERNVL3_USER = "<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n"


class _SanaStubTokenizer:
    """Minimal stand-in for SanaPipeline when prompt_embeds bypass text encoding."""

    padding_side = "right"


class UniQueryInternVL3Model(nn.Module):
    """InternVL3-1B backbone with the same special-token MetaQuery as the Qwen model.

    The Qwen variant encodes BOI/EOI/query tokens as new vocabulary rows and slices
    the frozen LLM hidden states between BOI and EOI. This class reproduces that
    mechanism on InternVL3's Llama language model so both backbones share identical
    MetaQuery semantics and adapter checkpoint format. Images (via the InternVL ViT)
    are not part of the text-to-image query path; like the OpenUni t2i stage, only the
    LLM text forward is used.
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        # Named model_config, not config: HF Trainer integrations call
        # model.config.to_json_string() and would fail on a plain dict.
        self.model_config = config
        dtype = getattr(torch, config.get("torch_dtype", "bfloat16"))
        ivl3_id = config["ivl3_id"]
        backend = config.get("attention_backend", "sdpa")
        use_flash_attn = backend == "flash_attention_2" and has_flash_attn

        self.ivl3 = InternVLChatModel.from_pretrained(
            ivl3_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            use_flash_attn=use_flash_attn,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            ivl3_id, trust_remote_code=True, padding_side="right"
        )
        self.num_queries = int(config["connector"].get("num_queries", 256))
        tokenizer = self.tokenizer
        language_model = self.ivl3.language_model
        original_vocab_size = language_model.get_input_embeddings().num_embeddings
        # Mirror the Qwen variant's exact MetaQuery vocabulary construction: reserve
        # model rows for BOI/EOI/query tokens, bring the tokenizer up to the model's
        # original vocabulary size, then append the MetaQuery special tokens.
        try:
            language_model.resize_token_embeddings(
                original_vocab_size + self.num_queries + 2,
                mean_resizing=False,
            )
        except TypeError:
            language_model.resize_token_embeddings(
                original_vocab_size + self.num_queries + 2
            )
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
        if len(tokenizer) != language_model.get_input_embeddings().num_embeddings:
            raise ValueError(
                "MetaQuery tokenizer/model vocabulary mismatch after adding special tokens: "
                f"tokenizer={len(tokenizer)}, model="
                f"{language_model.get_input_embeddings().num_embeddings}"
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
            + "<end_of_img>"
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

        self.ivl3.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.train_flow_model = bool(config.get("train_flow_model", False))
        self.transformer.requires_grad_(self.train_flow_model)
        # Original MetaQuery learnable-query mechanism: the resized embedding parameter
        # participates in autograd, while the hook zeros every pre-existing InternVL3
        # row so only BOI/EOI/<img_i> rows learn.
        embedding_weight = language_model.get_input_embeddings().weight
        embedding_weight.requires_grad_(True)

        def freeze_internvl3_embedding_rows(gradient: torch.Tensor) -> torch.Tensor:
            gradient[:original_vocab_size].zero_()
            return gradient

        embedding_weight.register_hook(freeze_internvl3_embedding_rows)
        language_model.config.use_cache = False

        context_size = int(self.ivl3.config.llm_config.hidden_size)
        output_size = int(self.transformer.config.caption_channels)
        # The frozen LLM emits bf16 hidden states; run the connector in the same dtype
        # (OpenUni's reference trains it under bfloat16 as well) so standalone forward
        # and generation work without an autocast wrapper.
        self.connector = build_connector(config["connector"], context_size, output_size).to(dtype)

        loss_config = config.get("losses", {})
        self.flow_weight = float(loss_config.get("flow_weight", 1.0))
        self.diversity_weight = float(loss_config.get("query_diversity_weight", 0.0))
        self.alignment_weight = float(loss_config.get("latent_alignment_weight", 0.0))
        self.alignment_head = None
        if self.alignment_weight > 0:
            query_size = int(config["connector"]["hidden_size"])
            latent_channels = int(getattr(self.transformer.config, "in_channels", 32))
            self.alignment_head = LatentAlignmentHead(query_size, latent_channels).to(dtype)

        if config.get("gradient_checkpointing", True):
            language_model.gradient_checkpointing_enable({"use_reentrant": False})
            self.transformer.enable_gradient_checkpointing()

    def train(self, mode: bool = True):
        super().train(mode)
        # The frozen VAE needs no backward graph. InternVL3/Sana remain in training
        # mode so their checkpointed activations can pass gradients to Query/Connector.
        self.vae.eval()
        return self

    @property
    def trainable_parameter_count(self) -> int:
        # Count the effective trainable rows, not the frozen prefix of the single
        # dense embedding Parameter retained for MetaQuery compatibility.
        embedding_weight = self.ivl3.language_model.get_input_embeddings().weight
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
    ) -> torch.Tensor:
        # InternVL3's InternVLChatModel.forward unconditionally runs extract_feature on
        # pixel_values, so the MetaQuery t2i path bypasses it and drives the frozen
        # Llama directly (matching OpenUni's `self.llm.model(...)` call).
        output = self.ivl3.language_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        hidden_states = output.last_hidden_state

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

    def encode_queries(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context = self.encode_context(input_ids, attention_mask)
        query_mask = torch.ones(
            context.shape[:2], device=context.device, dtype=torch.bool
        )
        return self.connector(context, query_mask)

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
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        latents = self.pixels_to_latents(target_pixels)
        prompt_embeds, query_states = self.encode_queries(
            input_ids=input_ids, attention_mask=attention_mask
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

        system_prompt = self.model_config.get(
            "generation_system_prompt",
            "You are a text-to-image generation model. Generate the image described "
            "by the user prompt. The image query tokens in the sequence represent "
            "the image to generate.",
        )

        def render_prompt(text: str) -> str:
            return (
                _INTERNVL3_SYSTEM.format(system=system_prompt)
                + _INTERNVL3_USER.format(input=text)
                + self.query_suffix
            )

        positive = [render_prompt(prompt) for prompt in prompts]
        negative = [render_prompt(negative_prompt) for _ in prompts]
        encoded = self.tokenizer(
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
            # diffusers 0.32 unconditionally touches self.tokenizer.padding_side in
            # encode_prompt even when prompt_embeds are passed in.
            tokenizer=_SanaStubTokenizer(),
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
        if self.train_flow_model:
            # Store the Sana flow weights under their top-level `transformer.`
            # module prefix (not the bare submodule keys) so they round-trip
            # through self.load_state_dict() on resume.
            state.update(
                {
                    name: tensor.detach().cpu().contiguous()
                    for name, tensor in self.state_dict().items()
                    if name.startswith("transformer.")
                }
            )
        state["metaquery_embeddings"] = (
            self.ivl3.language_model.get_input_embeddings()
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
            json.dump(self.model_config, handle, indent=2, ensure_ascii=False)

    def load_adapter(self, checkpoint_dir: str | Path, strict: bool = True) -> None:
        state = load_file(str(Path(checkpoint_dir) / "adapter_model.safetensors"))
        metaquery_embeddings = state.pop("metaquery_embeddings", None)
        if metaquery_embeddings is not None:
            expected_shape = (
                self.num_queries + 2,
                self.ivl3.language_model.get_input_embeddings().weight.shape[1],
            )
            if tuple(metaquery_embeddings.shape) != expected_shape:
                raise RuntimeError(
                    "MetaQuery embedding shape mismatch; "
                    f"expected={expected_shape}, got={tuple(metaquery_embeddings.shape)}"
                )
            with torch.no_grad():
                self.ivl3.language_model.get_input_embeddings().weight[
                    self.metaquery_token_start : self.metaquery_token_end + 1
                ].copy_(metaquery_embeddings)
        # Older finetune checkpoints stored the Sana flow weights as bare
        # submodule keys (`transformer_blocks.*`). Re-add the top-level
        # `transformer.` prefix so load_state_dict can map them instead of
        # reporting every weight as unexpected (which trips the strict check).
        transformer_keys = set(self.transformer.state_dict().keys())
        if any(name in transformer_keys for name in state):
            state = {
                (f"transformer.{name}" if name in transformer_keys else name): tensor
                for name, tensor in state.items()
            }
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
