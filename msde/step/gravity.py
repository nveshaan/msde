import torch

from ..common.kernel import estimate_bandwidth, gaussian_kernel, row_neighbor_dist
from ..common.knn import METRICS
from ..common.utils import DEFAULT_DEVICE
from .base import MovementStep


class GravityBarycenter(MovementStep):
    """Attraction-weighted neighbourhood mean, then a step of size ``eta``.

    w_ij = q_j * K(d_ij)                    (the attraction weights)
    mu_i = sum_j w_ij x_j / (sum_j w_ij + eps)
    x_i' = x_i + eta * (mu_i - x_i)

    ``q_j`` is the per-point attraction factor from the weights component and
    ``K(d) = exp(-d^2 / 2h^2)``. Distances are recomputed from the current
    positions on every iteration, and every point is updated from the same
    snapshot (no sequential in-place updates). The neighbours themselves come
    from the neighbours component and are only refreshed when the loop
    recomputes them (``recompute_neighbors``).

    A point whose weights sum to at most ``eps`` (all neighbours have no
    attraction, or none are valid under the label mask) is left where it is,
    rather than being pulled towards the origin by the ``eps`` in the
    denominator.

    eta : step size, 0 < eta <= 1; ``forward``'s ``gate`` scales it further.
    h : kernel bandwidth, a positive number, or "auto" for the median distance
        to the k nearest neighbours of the current points (re-estimated each
        time neighbours are recomputed).
    metric : "euclidean" or "cosine" distance inside the kernel.
    """

    def __init__(self, eta=0.1, h="auto", metric="euclidean", eps=1e-8,
                 device=DEFAULT_DEVICE, chunk_size=4096):
        super().__init__()
        if not 0 < eta <= 1:
            raise ValueError(f"eta must be in (0, 1]; got {eta!r}")
        if metric not in METRICS:
            raise ValueError(f"metric must be one of {sorted(METRICS)}; got {metric!r}")
        if isinstance(h, str):
            if h != "auto":
                raise ValueError(f"h must be a positive number or 'auto'; got {h!r}")
        elif isinstance(h, bool) or not h > 0:
            raise ValueError(f"h must be a positive number or 'auto'; got {h!r}")
        if not eps > 0:
            raise ValueError("eps must be > 0")
        self.h = h
        self.metric = metric
        self.eps = eps
        self.device_name = device
        self.chunk_size = chunk_size
        self.register_scalar("eta", eta, device)

    @torch.no_grad()
    def prepare(self, X, base_weights, neighbors):
        indices = neighbors.indices
        q = base_weights[indices]                                    # q_j of each selected neighbour
        if neighbors.valid is not None:
            q = q * neighbors.valid
        h = (estimate_bandwidth(X, indices.shape[1], self.metric, self.device_name)
             if self.h == "auto" else float(self.h))
        return indices, q, h

    @torch.no_grad()
    def step(self, X, state, gate=1.0):
        indices, q, h = state
        new = torch.empty_like(X)
        for start in range(0, X.shape[0], self.chunk_size):
            end = min(X.shape[0], start + self.chunk_size)
            x = X[start:end]
            xj = X[indices[start:end]]                               # (c, k, d), from the snapshot X
            w = q[start:end] * gaussian_kernel(row_neighbor_dist(x, xj, self.metric), h)
            total = w.sum(dim=1)
            mu = (w.unsqueeze(-1) * xj).sum(dim=1) / (total + self.eps).unsqueeze(-1)
            mu = torch.where((total > self.eps).unsqueeze(-1), mu, x)
            new[start:end] = x + self.eta * gate * (mu - x)
        return new, (new - X).norm(dim=1)
