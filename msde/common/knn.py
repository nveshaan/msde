import torch
import torch.nn.functional as F

from .utils import DEFAULT_DEVICE


def _euclidean_dist(X_chunk, X_full):
    return torch.cdist(X_chunk, X_full)


def _cosine_dist(X_chunk, X_full):
    """Cosine distance 1 - cos(x, y), in [0, 2]."""
    sim = F.normalize(X_chunk, dim=1) @ F.normalize(X_full, dim=1).T
    return (1.0 - sim).clamp_min(0.0)


METRICS = {"euclidean": _euclidean_dist, "cosine": _cosine_dist}

_KNN_CHUNK_KERNEL_CACHE = {}
_MASKED_KNN_CHUNK_KERNEL_CACHE = {}


def _make_knn_chunk_kernel(k, metric):
    dist_fn = METRICS[metric]

    def _kernel(X_chunk, X_full):
        d = dist_fn(X_chunk, X_full)              # (chunk, n) - compares to ALL N
        return d.topk(k, largest=False, dim=1)
    return torch.compile(_kernel, fullgraph=True)


def _get_knn_chunk_kernel(k, metric):
    """Return the compiled unmasked k-NN kernel for ``k`` and ``metric``."""
    key = (k, metric)
    if key not in _KNN_CHUNK_KERNEL_CACHE:
        _KNN_CHUNK_KERNEL_CACHE[key] = _make_knn_chunk_kernel(k, metric)
    return _KNN_CHUNK_KERNEL_CACHE[key]


def _make_masked_knn_chunk_kernel(k, metric):
    """Return a compiled k-NN kernel that masks cross-class distances."""
    dist_fn = METRICS[metric]

    def _kernel(X_chunk, X_full, labels_chunk, labels_full):
        d = dist_fn(X_chunk, X_full)                              # (chunk, n)
        same_class = labels_chunk.unsqueeze(1) == labels_full.unsqueeze(0)
        d = d.masked_fill(~same_class, float("inf"))
        return d.topk(k, largest=False, dim=1)
    return torch.compile(_kernel, fullgraph=True)


def _get_masked_knn_chunk_kernel(k, metric):
    """Return the cached masked k-NN kernel for ``k`` and ``metric``."""
    key = (k, metric)
    if key not in _MASKED_KNN_CHUNK_KERNEL_CACHE:
        _MASKED_KNN_CHUNK_KERNEL_CACHE[key] = _make_masked_knn_chunk_kernel(k, metric)
    return _MASKED_KNN_CHUNK_KERNEL_CACHE[key]


def torch_knn(X, k, device=DEFAULT_DEVICE, chunk_size=8192, labels=None, corpus_labels=None,
              metric="euclidean"):
    """Compute exact chunked k-NN indices and distances under ``metric``
    ("euclidean" or "cosine" distance)."""
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {sorted(METRICS)}; got {metric!r}")
    X_full = X

    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError(f"k must be a positive integer; got {k!r}")
    corpus_size = X_full.shape[0]
    if k < 1 or k > corpus_size:
        raise ValueError(
            f"k={k} is invalid for k-NN size {corpus_size}; "
            f"expected 1 <= k <= {corpus_size}"
        )

    if labels is not None:
        # corpus_labels: labels of X's points as neighbours (defaults to ``labels``)
        full_labels = corpus_labels if corpus_labels is not None else labels
        chunk_kernel = _get_masked_knn_chunk_kernel(k, metric)
    else:
        full_labels = None
        chunk_kernel = _get_knn_chunk_kernel(k, metric)

    n = X.shape[0]
    all_idx = torch.empty((n, k), dtype=torch.long, device=device)
    all_dist = torch.empty((n, k), dtype=X.dtype, device=device)

    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        if labels is not None:
            dists, idx = chunk_kernel(X[start:end], X_full, labels[start:end], full_labels)
        else:
            dists, idx = chunk_kernel(X[start:end], X_full)
        all_idx[start:end] = idx
        all_dist[start:end] = dists

    return all_idx, all_dist


def compute_fixed_knn(X, k, device=DEFAULT_DEVICE, chunk_size=4096, labels=None, corpus_labels=None,
                      metric="euclidean"):
    indices, _ = torch_knn(
        X, k, device=device, chunk_size=chunk_size,
        labels=labels, corpus_labels=corpus_labels, metric=metric,
    )
    return indices
