from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def query_diversity_loss(query_states: torch.Tensor) -> torch.Tensor:
    normalized = F.normalize(query_states.float(), dim=-1)
    gram = normalized @ normalized.transpose(1, 2)
    identity = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype)
    off_diagonal = gram - identity.unsqueeze(0)
    return off_diagonal.square().mean()


class LatentAlignmentHead(nn.Module):
    """Predicts target VAE latent statistics from the unified query representation."""

    def __init__(self, query_size: int, latent_channels: int):
        super().__init__()
        self.projection = nn.Linear(query_size, latent_channels * 2)

    def forward(self, query_states: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        query_summary = query_states.mean(dim=1)
        target = torch.cat(
            [latents.float().mean(dim=(2, 3)), latents.float().std(dim=(2, 3))],
            dim=-1,
        )
        prediction = self.projection(query_summary.float())
        return F.smooth_l1_loss(prediction, target)

