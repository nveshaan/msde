import torch

from ..common.utils import DEFAULT_DEVICE
from .base import DensityWeights

DENSITY_NORMALIZATIONS = ("none", "max", "minmax", "rank")


class UniformWeights(DensityWeights):
    """Every point attracts equally (q_j = 1): the distance-only baseline."""

    def __init__(self):
        super().__init__()

    def compute(self, X, stability=None):
        return torch.ones(X.shape[0], dtype=X.dtype, device=X.device)


class GravityWeights(DensityWeights):
    """The density-stability attraction factor ``q_j = rho_j^alpha * s_j^beta``.

    rho comes from any ``DensityWeights`` strategy (``density``); s is the
    per-point stability score in [0, 1] given to ``forward(..., stability=s)``
    (higher = more stable). Without stability scores, s = 1 for every point.

    Preprocessing, applied before the exponents so that q is well scaled:
    density_normalization : how rho is rescaled
        - "none":   as is
        - "max":    rho / max(rho), in (0, 1]
        - "minmax": (rho - min) / (max - min), in [0, 1]
        - "rank":   rank / n, in (0, 1], robust to outliers and scale
    density_floor : lower bound on rho after normalising, keeping it positive
        so that ``alpha`` can be any real number.
    stability_floor : lower bound on s. 0 (default) lets zero stability remove
        a point's attraction entirely; a small positive value keeps it faintly
        attractive.
    """

    uses_stability = True

    def __init__(self, density, alpha=1.0, beta=1.0, density_normalization="max",
                 density_floor=1e-6, stability_floor=0.0, device=DEFAULT_DEVICE):
        super().__init__()
        if not isinstance(density, DensityWeights) or density.uses_stability:
            raise ValueError("density must be a DensityWeights strategy that produces rho (e.g. KDEWeights)")
        if density_normalization not in DENSITY_NORMALIZATIONS:
            raise ValueError(f"density_normalization must be one of {DENSITY_NORMALIZATIONS}; got {density_normalization!r}")
        if not density_floor > 0:
            raise ValueError("density_floor must be > 0")
        if not 0.0 <= stability_floor <= 1.0:
            raise ValueError("stability_floor must be in [0, 1]")
        self.density = density
        self.density_normalization = density_normalization
        self.density_floor = density_floor
        self.stability_floor = stability_floor
        self.register_scalar("alpha", alpha, device)
        self.register_scalar("beta", beta, device)

    def normalize_density(self, rho):
        mode = self.density_normalization
        if mode == "max":
            rho = rho / rho.max().clamp_min(torch.finfo(rho.dtype).tiny)
        elif mode == "minmax":
            lo, hi = rho.min(), rho.max()
            rho = (rho - lo) / (hi - lo).clamp_min(torch.finfo(rho.dtype).tiny)
        elif mode == "rank":
            ranks = rho.argsort().argsort().to(rho.dtype) + 1.0
            rho = ranks / rho.shape[0]
        return rho.clamp_min(self.density_floor)

    @torch.no_grad()
    def compute(self, X, stability=None):
        rho = self.normalize_density(self.density.compute(X))
        if stability is None:
            s = torch.ones_like(rho)
        else:
            s = stability.to(device=rho.device, dtype=rho.dtype).clamp(self.stability_floor, 1.0)
        return rho.pow(self.alpha) * s.pow(self.beta)
