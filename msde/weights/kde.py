import math

import torch

from ..common.utils import DEFAULT_DEVICE
from .base import DensityWeights

BANDWIDTH_RULES = ("scott", "silverman")


class KDEWeights(DensityWeights):
    """Gaussian kernel density estimate of every point, as the density prior.

    weight_i = sum_{j != i} exp(-||x_i - x_j||^2 / (2 h^2))

    The sum leaves out the point itself (leave-one-out) and has no
    normalisation constant, so it is a soft neighbour count in ``[0, n-1]``:
    the same scale as ``SparseEmpirical``'s counts, and independent of ``d``.
    Only the relative weights matter to the movement step.

    Cost: ``O(n^2 d)`` time, ``O(chunk_size * n)`` working space. This is the
    cheaper alternative to ``SparseEmpirical``, which pays an extra
    ``O(P * n^2 * q)`` for its graph-space counting.

    bandwidth : the kernel width ``h``, either a positive number or a rule
        computed from the data on every call, using ``s`` = the mean
        per-feature standard deviation:
        - "scott":     h = s * n^(-1/(d+4))
        - "silverman": h = s * (n * (d + 2) / 4)^(-1/(d+4))
        In high dimensions both rules give a bandwidth close to ``s``, so the
        estimate is very smooth; pass a number to sharpen it.
    chunk_size : rows of the n x n distance matrix held at once.
    """

    def __init__(self, bandwidth="scott", chunk_size=2048, device=DEFAULT_DEVICE):
        super().__init__()
        if isinstance(bandwidth, str):
            if bandwidth not in BANDWIDTH_RULES:
                raise ValueError(
                    f"bandwidth must be a positive number or one of {BANDWIDTH_RULES}; got {bandwidth!r}"
                )
        elif isinstance(bandwidth, bool) or not bandwidth > 0:
            raise ValueError(f"bandwidth must be a positive number or one of {BANDWIDTH_RULES}; got {bandwidth!r}")
        if chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        self.bandwidth = bandwidth
        self.chunk_size = chunk_size
        self.device_name = device

    def resolve_bandwidth(self, X):
        """The numeric bandwidth used for ``X`` (applies the rule if one was given)."""
        if not isinstance(self.bandwidth, str):
            return float(self.bandwidth)
        n, d = X.shape
        scale = X.std(dim=0).mean().item() if n > 1 else 1.0
        if self.bandwidth == "scott":
            factor = n ** (-1.0 / (d + 4))
        else:
            factor = (n * (d + 2) / 4.0) ** (-1.0 / (d + 4))
        # A degenerate sample (all points equal) has zero spread; any positive h works.
        return max(scale * factor, 1e-12)

    @torch.no_grad()
    def compute(self, X):
        n = X.shape[0]
        inv_two_h2 = 1.0 / (2.0 * self.resolve_bandwidth(X) ** 2)
        weights = torch.empty(n, dtype=X.dtype, device=X.device)
        for start in range(0, n, self.chunk_size):
            end = min(n, start + self.chunk_size)
            k = torch.exp(-torch.cdist(X[start:end], X).square() * inv_two_h2)    # (c, n)
            # Leave-one-out: drop each row's own entry exactly (cdist leaves rounding noise there).
            rows = torch.arange(end - start, device=X.device)
            k[rows, rows + start] = 0.0
            weights[start:end] = k.sum(dim=1)
        return weights
