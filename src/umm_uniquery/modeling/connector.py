from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


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


def build_connector(config: dict, context_size: int, output_size: int) -> nn.Module:
    connector_type = config.get("type", "light_transformer")
    kwargs = {key: value for key, value in config.items() if key != "type"}
    if connector_type == "light_transformer":
        return LightQueryConnector(context_size=context_size, output_size=output_size, **kwargs)
    if connector_type == "linear":
        return LinearQueryConnector(context_size=context_size, output_size=output_size, **kwargs)
    raise ValueError(f"Unknown connector type: {connector_type}")
