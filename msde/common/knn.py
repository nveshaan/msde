import torch

from .utils import DEFAULT_DEVICE, DIFF_EPS


_KNN_CHUNK_KERNEL_CACHE = {}
_MASKED_KNN_CHUNK_KERNEL_CACHE = {}


def _make_knn_chunk_kernel(k):
    def _kernel(X_chunk, X_full):
        d = torch.cdist(X_chunk, X_full)          # (chunk, n) - compares to ALL N
        return d.topk(k, largest=False, dim=1)
    return torch.compile(_kernel, fullgraph=True)


def _get_knn_chunk_kernel(k):
    """Return the compiled unmasked k-NN kernel for ``k``."""
    if k not in _KNN_CHUNK_KERNEL_CACHE:
        _KNN_CHUNK_KERNEL_CACHE[k] = _make_knn_chunk_kernel(k)
    return _KNN_CHUNK_KERNEL_CACHE[k]


def _make_masked_knn_chunk_kernel(k):
    """Return a compiled k-NN kernel that masks cross-class distances."""
    def _kernel(X_chunk, X_full, labels_chunk, labels_full):
        d = torch.cdist(X_chunk, X_full)                          # (chunk, n)
        same_class = labels_chunk.unsqueeze(1) == labels_full.unsqueeze(0)
        d = d.masked_fill(~same_class, float("inf"))
        return d.topk(k, largest=False, dim=1)
    return torch.compile(_kernel, fullgraph=True)


def _get_masked_knn_chunk_kernel(k):
    """Return the cached masked k-NN kernel for ``k``."""
    if k not in _MASKED_KNN_CHUNK_KERNEL_CACHE:
        _MASKED_KNN_CHUNK_KERNEL_CACHE[k] = _make_masked_knn_chunk_kernel(k)
    return _MASKED_KNN_CHUNK_KERNEL_CACHE[k]


def _make_soft_topk_weight_kernel():
    def _kernel(X, corpus, indices, temperature):
        sel = corpus[indices]                                    # (n, k, d)
        d_selected = (X.unsqueeze(1) - sel).norm(dim=-1)          # (n, k)
        boundary = d_selected.max(dim=1, keepdim=True).values     # (n, 1) -- the k-th neighbour's distance
        return torch.sigmoid((boundary - d_selected) / temperature)
    return torch.compile(_kernel, fullgraph=True)


SOFT_TOPK_WEIGHT_KERNEL = _make_soft_topk_weight_kernel()


def torch_knn(X, k, device=DEFAULT_DEVICE, chunk_size=8192, corpus=None, labels=None, corpus_labels=None):
    """Compute exact chunked k-NN indices and distances."""
    X_full = corpus if corpus is not None else X

    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError(f"k must be a positive integer; got {k!r}")
    corpus_size = X_full.shape[0]
    if k < 1 or k > corpus_size:
        raise ValueError(
            f"k={k} is invalid for k-NN corpus size {corpus_size}; "
            f"expected 1 <= k <= {corpus_size}"
        )

    if labels is not None:
        if corpus is None:
            full_labels = corpus_labels if corpus_labels is not None else labels
        elif corpus_labels is not None:
            full_labels = corpus_labels
        else:
            raise ValueError(
                "corpus_labels must be given alongside labels when corpus is also given "
                "(cross-manifold masked k-NN) -- there's no default correspondence between "
                "X's labels and corpus's labels."
            )
        chunk_kernel = _get_masked_knn_chunk_kernel(k)
    else:
        full_labels = None
        chunk_kernel = _get_knn_chunk_kernel(k)

    n = X.shape[0]
    all_idx = torch.empty((n, k), dtype=torch.long, device=device)
    
    # Store dists in a list and cat them to preserve autograd graph 
    # (inplace assignments on requires_grad tensors throw errors)
    all_dist_list = []

    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        if labels is not None:
            dists, idx = chunk_kernel(X[start:end], X_full, labels[start:end], full_labels)
        else:
            dists, idx = chunk_kernel(X[start:end], X_full)
        all_idx[start:end] = idx
        all_dist_list.append(dists)

    all_dist = torch.cat(all_dist_list, dim=0)
    return all_idx, all_dist


def compute_fixed_knn(X, k, device=DEFAULT_DEVICE, chunk_size=4096, corpus=None, labels=None, corpus_labels=None):
    indices, _ = torch_knn(
        X, k, device=device, chunk_size=chunk_size, corpus=corpus,
        labels=labels, corpus_labels=corpus_labels,
    )
    return indices



def smooth_pairwise_dist(X, Y):
    """Pairwise Euclidean distances with a smooth zero-distance guard."""
    diff = X.unsqueeze(1) - Y.unsqueeze(0)
    return torch.sqrt(diff.square().sum(dim=-1) + DIFF_EPS**2)


def masked_soft_knn_weights(
    X,
    corpus,
    k,
    temperature,
    labels=None,
    corpus_labels=None,
    exclude_self=False,
    iterations=20,
):
    """Continuous relaxation of k-NN over *all* candidate points.

    For every query row i we solve for tau_i so the sigmoid occupancies have
    total mass approximately k. Unlike topk, every candidate receives a
    differentiable (possibly very small) membership weight.
    """
    if temperature is None:
        raise ValueError("A positive temperature is required for differentiable soft k-NN")
    temperature = torch.as_tensor(temperature, dtype=X.dtype, device=X.device)
    temperature = torch.sqrt(temperature.square() + DIFF_EPS**2)

    d = smooth_pairwise_dist(X, corpus)
    valid = torch.ones(d.shape, dtype=torch.bool, device=d.device)

    if exclude_self:
        if X.shape[0] != corpus.shape[0]:
            raise ValueError("exclude_self=True requires query and corpus to have the same size")
        valid = valid & ~torch.eye(X.shape[0], dtype=torch.bool, device=d.device)

    if labels is not None:
        if corpus_labels is None:
            corpus_labels = labels
        same_class = labels.unsqueeze(1) == corpus_labels.unsqueeze(0)
        valid = valid & same_class

    valid_f = valid.to(d.dtype)
    valid_count = valid_f.sum(dim=1)
    target = torch.minimum(
        valid_count,
        torch.full_like(valid_count, float(k)),
    )

    mean_d = (d * valid_f).sum(dim=1) / (valid_count + DIFF_EPS)
    tau = mean_d

    max_tau_step = 4.0 * (mean_d.detach().mean() + 1.0)
    max_tau_step = torch.as_tensor(max_tau_step, dtype=d.dtype, device=d.device)

    for _ in range(iterations):
        membership = torch.sigmoid((tau.unsqueeze(1) - d) / temperature) * valid_f
        residual = membership.sum(dim=1) - target
        derivative = (
            membership * (1.0 - membership)
        ).sum(dim=1) / temperature
        raw_step = residual / (derivative + DIFF_EPS)
        step = max_tau_step * torch.tanh(raw_step / (max_tau_step + DIFF_EPS))
        tau = tau - step

    membership = torch.sigmoid((tau.unsqueeze(1) - d) / temperature) * valid_f
    return d, membership, valid
