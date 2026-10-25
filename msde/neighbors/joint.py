import torch

from ..common.kernel import estimate_bandwidth, gaussian_kernel
from ..common.knn import METRICS
from ..common.utils import DEFAULT_DEVICE
from .base import NeighborSelector, Neighbors


class JointKNN(NeighborSelector):
    """Neighbours chosen by a joint distance / density / stability score.

    a_ij = q_j * K(d_ij),  K(d) = exp(-d^2 / 2h^2),  N_k(i) = TopK_{j != i} a_ij

    where ``q_j`` is the per-point attraction factor from the weights component
    (``rho_j^alpha * s_j^beta`` with ``GravityWeights``). Unlike a distance-only
    k-NN, a dense, stable point can outrank a closer unstable or sparse one.
    This is the *selection* step only; the attraction weights are computed
    separately by the movement step.

    Every point is a candidate (no distance pre-filter), so the cost is
    ``O(n^2 d)`` time and ``O(chunk_size * n)`` working space. A point is
    never its own neighbour.

    Zero scores: candidates are ranked by score first; any slots left once the
    positive scores run out (including all of them, if every score is zero) are
    filled by the nearest remaining candidates.

    h : kernel bandwidth, a positive number, or "auto" for the median distance
        to the k nearest neighbours (re-estimated on every ``select`` call).
    metric : "euclidean" or "cosine"; ``Neighbors.dist`` holds those distances.
    """

    def __init__(self, k=30, h="auto", metric="euclidean", device=DEFAULT_DEVICE, chunk_size=2048):
        super().__init__()
        if metric not in METRICS:
            raise ValueError(f"metric must be one of {sorted(METRICS)}; got {metric!r}")
        if isinstance(h, str):
            if h != "auto":
                raise ValueError(f"h must be a positive number or 'auto'; got {h!r}")
        elif isinstance(h, bool) or not h > 0:
            raise ValueError(f"h must be a positive number or 'auto'; got {h!r}")
        if chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        self.k = k
        self.h = h
        self.metric = metric
        self.device_name = device
        self.chunk_size = chunk_size

    @torch.no_grad()
    def select(self, X, labels=None, corpus_labels=None, base_weights=None):
        n = X.shape[0]
        if isinstance(self.k, bool) or not isinstance(self.k, int) or not 1 <= self.k <= n - 1:
            raise ValueError(f"k={self.k!r} is invalid for {n} points; expected an integer 1 <= k <= {n - 1}")
        if base_weights is None:
            base_weights = torch.ones(n, dtype=X.dtype, device=X.device)
        h = (estimate_bandwidth(X, self.k, self.metric, self.device_name)
             if self.h == "auto" else float(self.h))
        if labels is not None and corpus_labels is None:
            corpus_labels = labels
        dist_fn = METRICS[self.metric]

        indices = torch.empty((n, self.k), dtype=torch.long, device=X.device)
        dist = torch.empty((n, self.k), dtype=X.dtype, device=X.device)
        valid = torch.empty((n, self.k), dtype=torch.bool, device=X.device)
        for start in range(0, n, self.chunk_size):
            end = min(n, start + self.chunk_size)
            d = dist_fn(X[start:end], X)                                  # (c, n)
            score = base_weights.unsqueeze(0) * gaussian_kernel(d, h)     # a_ij
            # Positive scores rank first, by score; zero scores fall back to nearest.
            key = torch.where(score > 0, score, -d)

            excluded = torch.zeros_like(d, dtype=torch.bool)
            rows = torch.arange(end - start, device=X.device)
            excluded[rows, rows + start] = True                           # never its own neighbour
            if labels is not None:
                excluded |= labels[start:end].unsqueeze(1) != corpus_labels.unsqueeze(0)
            key = key.masked_fill(excluded, float("-inf"))

            top, idx = key.topk(self.k, dim=1)
            indices[start:end] = idx
            dist[start:end] = d.gather(1, idx)
            valid[start:end] = torch.isfinite(top)

        if labels is None:
            return Neighbors(indices=indices, dist=dist)
        return Neighbors(indices=indices, dist=dist, valid=valid, orphan_mask=~valid.any(dim=1))
