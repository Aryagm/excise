"""Hard-concrete channel gates (Louizos et al., 2018)."""

import math

import torch

GAMMA, ZETA, BETA = -0.1, 1.1, 2.0 / 3.0
_SHIFT = BETA * math.log(-GAMMA / ZETA)


class HardConcreteGates(torch.nn.Module):
    def __init__(self, n_layers: int, d_ff: int, init: float = 3.0):
        super().__init__()
        self.log_alpha = torch.nn.Parameter(torch.full((n_layers, d_ff), init))

    def sample_all(self) -> list[torch.Tensor]:
        """One stochastic gate vector per layer.

        Must be called OUTSIDE any gradient-checkpointed region: in-region
        RNG produces a different graph on recomputation and breaks backward.
        """
        out = []
        for li in range(self.log_alpha.shape[0]):
            la = self.log_alpha[li]
            u = torch.rand_like(la).clamp_(1e-6, 1 - 1e-6)
            s = torch.sigmoid((u.log() - (-u).log1p() + la) / BETA)
            out.append((s * (ZETA - GAMMA) + GAMMA).clamp(0, 1))
        return out

    def p_open(self) -> torch.Tensor:
        return torch.sigmoid(self.log_alpha - _SHIFT)

    def expected_open(self) -> torch.Tensor:
        return self.p_open().mean()

    def topk_mask(self, k_total: int) -> torch.Tensor:
        """Binary mask keeping the k_total highest-scoring channels."""
        flat = self.log_alpha.detach().flatten()
        idx = torch.topk(flat, k_total).indices
        mask = torch.zeros_like(flat)
        mask[idx] = 1.0
        return mask.view_as(self.log_alpha)
