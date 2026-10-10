from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import _calculate_fan_in_and_fan_out


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.float().pow(2).mean(-1, keepdim=True)
        normalized = hidden_states * torch.rsqrt(variance + self.eps).to(hidden_states.dtype)
        return normalized * self.weight


class Attention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout = dropout

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, query_len, hidden = query.shape
        key_len = key_value.shape[1]
        query_states = self.q_proj(query).view(batch, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(key_value).view(batch, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(key_value).view(batch, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        attention_mask = None
        if key_padding_mask is not None:
            attention_mask = key_padding_mask[:, None, None, :].to(torch.bool)
        output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        output = output.transpose(1, 2).reshape(batch, query_len, hidden)
        return self.out_proj(output)


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class QueryConnectorLayer(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, num_heads: int, dropout: float):
        super().__init__()
        self.self_norm = RMSNorm(hidden_size)
        self.ffn_norm = RMSNorm(hidden_size)
        self.self_attention = Attention(hidden_size, num_heads, dropout)
        self.mlp = SwiGLU(hidden_size, intermediate_size)

    def forward(
        self,
        queries: torch.Tensor,
        query_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        normalized = self.self_norm(queries)
        queries = queries + self.self_attention(normalized, normalized, query_mask)
        return queries + self.mlp(self.ffn_norm(queries))


class LightQueryConnector(nn.Module):
    """Six-layer bidirectional query transformer with a Sana projection."""

    def __init__(
        self,
        context_size: int,
        output_size: int,
        hidden_size: int = 896,
        intermediate_size: int = 4096,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 14,
        dropout: float = 0.0,
        **_: object,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.context_projection = nn.Linear(context_size, hidden_size, bias=False)
        self.layers = nn.ModuleList(
            QueryConnectorLayer(hidden_size, intermediate_size, num_attention_heads, dropout)
            for _ in range(num_hidden_layers)
        )
        self.final_norm = RMSNorm(hidden_size)
        self.output_projection = nn.Linear(hidden_size, output_size, bias=False)

    def forward(
        self, context: torch.Tensor, context_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        queries = self.context_projection(context)
        for layer in self.layers:
            queries = layer(queries, context_mask)
        queries = self.final_norm(queries)
        return self.output_projection(queries), queries


class LinearQueryConnector(nn.Module):
    """Low-capacity architecture ablation that has no transformer layers."""

    def __init__(self, context_size: int, output_size: int, hidden_size: int, **_: object):
        super().__init__()
        self.context_projection = nn.Linear(context_size, hidden_size, bias=False)
        self.output_projection = nn.Linear(hidden_size, output_size, bias=False)

    def forward(
        self, context: torch.Tensor, context_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.context_projection(context)
        return self.output_projection(hidden), hidden


def _trunc_normal_(tensor, mean, std, a, b):
    # From PyTorch official master (mirrors OpenUni's modeling_connector.py).
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        raise ValueError("mean is more than 2 std from [a, b] in _trunc_normal_.")

    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)
    tensor.uniform_(2 * l - 1, 2 * u - 1)
    tensor.erfinv_()
    tensor.mul_(std * math.sqrt(2.0))
    tensor.add_(mean)
    tensor.clamp_(min=a, max=b)


def trunc_normal_tf_(tensor, mean: float = 0.0, std: float = 1.0, a: float = -2.0, b: float = 2.0):
    with torch.no_grad():
        _trunc_normal_(tensor, 0, 1.0, a, b)
        tensor.mul_(std).add_(mean)


def _variance_scaling_(tensor, scale=1.0, mode="fan_in", distribution="normal"):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    denom = fan_in if mode == "fan_in" else fan_out if mode == "fan_out" else (fan_in + fan_out) / 2
    variance = scale / denom
    if distribution == "truncated_normal":
        trunc_normal_tf_(tensor, std=math.sqrt(variance) / 0.87962566103423978)
    elif distribution == "normal":
        with torch.no_grad():
            tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        with torch.no_grad():
            tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def _lecun_normal_(tensor):
    _variance_scaling_(tensor, mode="fan_in", distribution="truncated_normal")


def _openuni_init_weights(module: nn.Module) -> None:
    """Exact OpenUni init scheme (src/models/connector/modeling_connector.py:48-72)."""
    if isinstance(module, OpenUniAttention):
        nn.init.xavier_uniform_(module.q_proj.weight)
        nn.init.xavier_uniform_(module.k_proj.weight)
        nn.init.xavier_uniform_(module.v_proj.weight)
        nn.init.xavier_uniform_(module.out_proj.weight)
        for bias in (module.q_proj.bias, module.k_proj.bias,
                     module.v_proj.bias, module.out_proj.bias):
            nn.init.zeros_(bias)
    elif isinstance(module, OpenUniMLP):
        nn.init.xavier_uniform_(module.fc1.weight)
        nn.init.xavier_uniform_(module.fc2.weight)
        nn.init.normal_(module.fc1.bias, std=1e-6)
        nn.init.normal_(module.fc2.bias, std=1e-6)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    elif isinstance(module, (nn.Linear, nn.Conv2d)):
        _lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class OpenUniAttention(nn.Module):
    """CLIP-style multi-head attention copied from OpenUni's ConnectorAttention."""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        if self.head_dim * num_heads != hidden_size:
            raise ValueError(f"embed_dim {hidden_size} must be divisible by num_heads {num_heads}")
        self.scale = self.head_dim**-0.5
        self.dropout = dropout
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        # Upcast attention to fp32 (OpenUni's ConnectorAttention does the same).
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, q_len, self.embed_dim)
        return self.out_proj(attn_output)


class OpenUniMLP(nn.Module):
    """Two-layer GELU MLP copied from OpenUni's ConnectorMLP."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(hidden_states)))


class OpenUniEncoderLayer(nn.Module):
    """Pre-norm transformer layer: LayerNorm -> self-attn -> residual; LayerNorm -> MLP -> residual."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.embed_dim = hidden_size
        self.self_attn = OpenUniAttention(hidden_size, num_heads, dropout)
        self.layer_norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.mlp = OpenUniMLP(hidden_size, intermediate_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class OpenUniConnectorEncoder(nn.Module):
    """Port of OpenUni's ConnectorEncoder (modeling_connector.py:479-507) with its init.

    Runs bidirectional self-attention over the 256 query tokens. The mask is the
    user-side bool [B, L] convention (all-ones in practice); it is converted to the
    additive 4D float mask the attention layers expect.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        attention_dropout: float = 0.0,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            OpenUniEncoderLayer(
                hidden_size, intermediate_size, num_attention_heads,
                attention_dropout, layer_norm_eps,
            )
            for _ in range(num_hidden_layers)
        )
        self.gradient_checkpointing = False
        self.apply(_openuni_init_weights)

    def forward(
        self,
        hidden_states: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if context_mask is not None:
            if context_mask.dtype == torch.bool or context_mask.dtype == torch.uint8:
                additive = (1 - context_mask.float()).unsqueeze(1).unsqueeze(1) * torch.finfo(
                    hidden_states.dtype
                ).min
            else:
                additive = context_mask
        else:
            additive = None
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer.__call__, hidden_states, use_reentrant=False
                )
            else:
                hidden_states = layer(hidden_states)
        return hidden_states


class OpenUniQueryConnector(nn.Module):
    """OpenUni connector: CLIP-style 6-layer encoder + separate Sana projector.

    Mirrors OpenUni's enc_proj path (llm2dit = projector(connector(x))): the frozen
    LLM hidden states (896-dim for InternVL3-1B) are processed directly by the
    encoder stack at hidden_size, then a single Linear projects to Sana's
    caption_channels. An input projection is added only when the backbone's hidden
    size differs from hidden_size (e.g. Qwen2.5-VL-3B at 2048-dim).
    """

    def __init__(
        self,
        context_size: int,
        output_size: int,
        hidden_size: int = 896,
        intermediate_size: int = 3072,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 14,
        attention_dropout: float = 0.0,
        dropout: float | None = None,
        num_queries: int | None = None,
        layer_norm_eps: float = 1e-5,
        **_: object,
    ):
        super().__init__()
        if dropout is not None:
            attention_dropout = dropout
        self.hidden_size = hidden_size
        self.input_projection = (
            nn.Linear(context_size, hidden_size) if context_size != hidden_size else None
        )
        self.encoder = OpenUniConnectorEncoder(
            hidden_size, intermediate_size, num_hidden_layers,
            num_attention_heads, attention_dropout, layer_norm_eps,
        )
        self.projector = nn.Linear(hidden_size, output_size)

    def forward(
        self, context: torch.Tensor, context_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.input_projection is not None:
            context = self.input_projection(context)
        hidden = self.encoder(context, context_mask)
        return self.projector(hidden), hidden


_MAE_LAYER_PREFIX = "vit.encoder.layer.{index}."
# MAE ViT encoder block tensor names (transformers 4.49 layout, q/k/v split) mapped
# to the OpenUni encoder layer attributes. The two blocks share the exact format:
# pre-norm LayerNorm -> MHSA -> residual, then LayerNorm -> GELU MLP -> residual.
_MAE_TO_CONNECTOR = [
    ("layer_norm1", "layernorm_before"),
    ("layer_norm2", "layernorm_after"),
    ("self_attn.q_proj", "attention.attention.query"),
    ("self_attn.k_proj", "attention.attention.key"),
    ("self_attn.v_proj", "attention.attention.value"),
    ("self_attn.out_proj", "attention.output.dense"),
    ("mlp.fc1", "intermediate.dense"),
    ("mlp.fc2", "output.dense"),
]


def _mae_weights(mae_id: str) -> dict[str, torch.Tensor]:
    """Load a MAE ViT checkpoint's state dict from a local dir or the Hub.

    Prefers the local ``model.safetensors`` / ``pytorch_model.bin`` (the offline
    path used on the mirror-only hosts), and falls back to ``ViTMAEModel`` so
    ``mae_id`` can also be a Hub repo id (downloaded through HF_ENDPOINT).
    """
    from pathlib import Path

    from safetensors.torch import load_file

    directory = Path(mae_id)
    safetensors_path = directory / "model.safetensors"
    if safetensors_path.is_file():
        return load_file(str(safetensors_path))
    bin_path = directory / "pytorch_model.bin"
    if bin_path.is_file():
        state = torch.load(str(bin_path), map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        if not isinstance(state, dict):
            raise ValueError(f"Unexpected pytorch_model.bin contents from {mae_id}")
        return state
    # Remote fallback: mirrors OpenUni's offline-first convention but still lets
    # hosts without a local copy pull the checkpoint through the HF mirror.
    from transformers import ViTMAEModel

    return ViTMAEModel.from_pretrained(mae_id, torch_dtype=torch.float32).state_dict()


def load_mae_encoder_weights(
    encoder: OpenUniConnectorEncoder, mae_id: str, layers: list[int]
) -> None:
    """Seed the connector encoder from MAE-pretrained ViT encoder blocks.

    ``layers`` maps connector layer i to MAE encoder layer ``layers[i]``, so a
    6-layer connector can take the first half of the 12-layer vit-mae-base encoder
    (``[0..5]``) or the whole stack (``[0..11]``). Everything else — the input
    projection, the Sana projector, the final RMS norm — keeps its OpenUni init.
    """
    weights = _mae_weights(mae_id)
    with torch.no_grad():
        for connector_index, mae_index in enumerate(layers):
            prefix = _MAE_LAYER_PREFIX.format(index=mae_index)
            layer = encoder.layers[connector_index]
            for target, source in _MAE_TO_CONNECTOR:
                destination = layer
                for part in target.split("."):
                    destination = getattr(destination, part)
                weight = weights.get(prefix + source + ".weight")
                if weight is None:
                    raise KeyError(
                        f"MAE checkpoint {mae_id!r} is missing {prefix + source}.weight; "
                        "is it a ViT-MAE model?"
                    )
                destination.weight.copy_(weight)
                bias = weights.get(prefix + source + ".bias")
                if bias is not None:
                    destination.bias.copy_(bias)


class OpenUniMAEQueryConnector(OpenUniQueryConnector):
    """OpenUni connector seeded from a MAE-pretrained ViT encoder.

    Natural-image-predicting (MAE) pre-training improves downstream tasks, so this
    variant boots the query transformer from MAE encoder weights instead of the
    random OpenUni init: the connector blocks share the ViT encoder's exact
    pre-norm LayerNorm -> MHSA -> GELU MLP format, which lets the weights be copied
    directly. hidden_size/num_attention_heads must match the chosen MAE checkpoint
    (768/12 for vit-mae-base); when the backbone hidden size differs,
    OpenUniQueryConnector inserts an input projection automatically.
    """

    def __init__(
        self,
        mae_id: str,
        mae_layers: list[int] | None = None,
        **kwargs: object,
    ):
        super().__init__(**kwargs)
        layers = (
            list(mae_layers)
            if mae_layers is not None
            else list(range(len(self.encoder.layers)))
        )
        load_mae_encoder_weights(self.encoder, mae_id, layers)


def build_connector(config: dict, context_size: int, output_size: int) -> nn.Module:
    connector_type = config.get("type", "light_transformer")
    kwargs = {key: value for key, value in config.items() if key != "type"}
    if connector_type == "light_transformer":
        return LightQueryConnector(context_size=context_size, output_size=output_size, **kwargs)
    if connector_type == "linear":
        return LinearQueryConnector(context_size=context_size, output_size=output_size, **kwargs)
    if connector_type == "openuni":
        return OpenUniQueryConnector(context_size=context_size, output_size=output_size, **kwargs)
    if connector_type == "openuni_mae":
        return OpenUniMAEQueryConnector(context_size=context_size, output_size=output_size, **kwargs)
    raise ValueError(f"Unknown connector type: {connector_type}")
