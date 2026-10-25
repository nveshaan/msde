import torch

from .knn import METRICS, torch_knn


def gaussian_kernel(d, h):
    """K(d) = exp(-d^2 / (2 h^2))."""
    return torch.exp(-d.square() / (2.0 * h * h))


def row_neighbor_dist(X, Xj, metric="euclidean"):
    """Distance from each row ``X[i]`` to its own neighbours ``Xj[i]``: (n, d), (n, k, d) -> (n, k)."""
    if metric == "euclidean":
        return (X.unsqueeze(1) - Xj).norm(dim=-1)
    if metric == "cosine":
        sim = torch.nn.functional.cosine_similarity(X.unsqueeze(1), Xj, dim=-1)
        return (1.0 - sim).clamp_min(0.0)
    raise ValueError(f"metric must be one of {sorted(METRICS)}; got {metric!r}")


@torch.no_grad()
def estimate_bandwidth(X, k, metric="euclidean", device="cpu"):
    """A data-driven kernel width: the median distance to the ``k`` nearest
    neighbours (the point itself excluded) over all points."""
    n = X.shape[0]
    if n < 2:
        return 1.0
    kk = min(k, n - 1)
    _, dist = torch_knn(X, kk + 1, device=device, metric=metric)   # column 0 is the point itself
    return max(dist[:, 1:].median().item(), 1e-12)
