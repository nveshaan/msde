import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

DEFAULT_DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def _prepare_scalar(value, name, device, dtype, learnable):
    if value is None:
        return None

    scalar = torch.as_tensor(value, device=device, dtype=dtype)
    if scalar.numel() != 1:
        raise ValueError(f"{name} must be a scalar")

    if learnable:
        if not scalar.is_leaf or not scalar.requires_grad:
            scalar = scalar.detach().clone().requires_grad_(True)
    else:
        scalar = scalar.detach()
    return scalar

def _configure_logging(log_file):
    """Configure this module's file-only logger."""
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()

    logger.propagate = False

    if log_file:
        handler = logging.FileHandler(log_file)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
    else:
        logger.addHandler(logging.NullHandler())
        logger.setLevel(logging.CRITICAL + 1)


_SPARSE_MM_SUPPORT_CACHE = {}

def _sparse_mm_supported(device):
    """Return whether sparse matrix multiplication works on ``device``."""
    if device in _SPARSE_MM_SUPPORT_CACHE:
        return _SPARSE_MM_SUPPORT_CACHE[device]
    try:
        idx = torch.zeros((2, 1), dtype=torch.long, device=device)
        vals = torch.ones(1, device=device)
        t = torch.sparse_coo_tensor(idx, vals, (1, 1), device=device).coalesce()
        torch.sparse.mm(t, torch.ones(1, 1, device=device))
        supported = True
    except Exception:
        supported = False
    _SPARSE_MM_SUPPORT_CACHE[device] = supported
    return supported


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


_SOFT_TOPK_WEIGHT_KERNEL = _make_soft_topk_weight_kernel()


def torch_knn(X, k, device=DEFAULT_DEVICE, chunk_size=8192, corpus=None, labels=None, corpus_labels=None):
    """Compute exact chunked k-NN indices and distances."""
    X_full = corpus if corpus is not None else X

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


def _make_symmetrize_kernel():
    def _kernel(knn_dists, knn_indices, rhos, sigma, n, k):
        dists_shifted = torch.clamp(knn_dists - rhos[:, None], min=0.0)
        weights = torch.exp(-dists_shifted / sigma[:, None])

        rows = torch.arange(n, device=knn_dists.device).repeat_interleave(k)
        cols = knn_indices.reshape(-1)
        vals = weights.reshape(-1)

        fwd_keys = rows * n + cols
        rev_keys = cols * n + rows

        sort_idx = torch.argsort(fwd_keys)
        sorted_keys = fwd_keys[sort_idx]
        sorted_vals = vals[sort_idx]

        pos = torch.searchsorted(sorted_keys, rev_keys)
        pos = torch.clamp(pos, max=sorted_keys.shape[0] - 1)
        matched = sorted_keys[pos] == rev_keys
        w_rev = torch.where(matched, sorted_vals[pos], torch.zeros_like(vals))

        w_sym = vals + w_rev - vals * w_rev
        return rows, cols, w_sym

    return torch.compile(_kernel, fullgraph=True)


_SYMMETRIZE_KERNEL = _make_symmetrize_kernel()


_SOFT_PRUNE_MARGIN_SIGMAS = 6.0


def fuzzy_simplicial_set_torch(knn_indices, knn_dists, n, n_neighbors, n_epochs=200,
                                device=DEFAULT_DEVICE, soft_prune=False,
                                soft_prune_temperature=None):
    """Build a sparse fuzzy graph from k-NN results."""
    if soft_prune and soft_prune_temperature is None:
        raise ValueError("soft_prune=True requires soft_prune_temperature")
    k = n_neighbors
    target = float(torch.log2(torch.tensor(k, dtype=torch.float32)))

    with torch.no_grad():
        mask = knn_dists > 0
        rhos_ng = torch.where(mask, knn_dists, torch.tensor(float("inf"), device=device))
        rhos_ng = torch.clamp(rhos_ng.min(dim=1).values, min=1e-8)

        lo = torch.full((n,), 1e-20, device=device)
        hi = torch.full((n,), 1e3, device=device)
        sigma_ng = torch.ones(n, device=device)
        
        dists_shifted_ng = torch.clamp(knn_dists - rhos_ng[:, None], min=0.0)
        dists_shifted_tail = dists_shifted_ng[:, 1:]

        for _ in range(64):
            vals = torch.exp(-dists_shifted_tail / sigma_ng[:, None])
            vals_sum = vals.sum(dim=1)

            converged = (vals_sum - target).abs() < 1e-5
            too_high = (vals_sum > target) & ~converged
            too_low = (vals_sum < target) & ~converged

            hi = torch.where(too_high, sigma_ng, hi)
            lo = torch.where(too_low, sigma_ng, lo)
            sigma_ng = torch.where(too_high, (lo + sigma_ng) / 2.0, sigma_ng)
            sigma_ng = torch.where(
                too_low,
                torch.where(hi >= 1e3, sigma_ng * 2.0, (sigma_ng + hi) / 2.0),
                sigma_ng,
            )

            if bool(converged.all()):
                break

    rhos = rhos_ng.detach()
    sigma = torch.clamp(sigma_ng.detach(), min=1e-10)

    rows, cols, w_sym = _SYMMETRIZE_KERNEL(knn_dists, knn_indices, rhos, sigma, n, k)

    threshold = w_sym.max() / max(n_epochs, 1)

    if soft_prune:
        keep_threshold = threshold.detach() - _SOFT_PRUNE_MARGIN_SIGMAS * soft_prune_temperature
        active = torch.nonzero(w_sym.detach() >= keep_threshold, as_tuple=True)[0]
        gate = torch.sigmoid((w_sym - threshold) / soft_prune_temperature)
        w_gated = w_sym * gate
        return rows[active], cols[active], w_gated[active]

    active = torch.nonzero(w_sym >= threshold, as_tuple=True)[0]
    return rows[active], cols[active], w_sym[active]


def _build_sparse_similarity(X, n_neighbors, n_epochs, device, soft_prune=False,
                              soft_prune_temperature=None):
    """Build the sparse fuzzy similarity graph used for density weights."""
    n = X.shape[0]
    knn_idx, knn_dist = torch_knn(X, n_neighbors, device=device)
    rows, cols, vals = fuzzy_simplicial_set_torch(
        knn_idx, knn_dist, n, n_neighbors, n_epochs, device,
        soft_prune=soft_prune, soft_prune_temperature=soft_prune_temperature,
    )
    return torch.sparse_coo_tensor(
        torch.stack([rows, cols]), vals, size=(n, n), device=device,
        check_invariants=False,
    ).coalesce()


def _sparse_similarity_layout(S):
    """Prepare row offsets and squared norms for streamed graph passes."""
    idx = S.indices()
    rows_sorted, cols_sorted, vals_sorted = idx[0], idx[1], S.values()
    n = S.shape[0]
    row_ptr = torch.searchsorted(
        rows_sorted, torch.arange(0, n + 1, device=S.device)
    ).tolist()
    row_norm_sq = torch.zeros(n, dtype=torch.float32, device=S.device)
    row_norm_sq.scatter_add_(0, rows_sorted, vals_sorted ** 2)
    return rows_sorted, cols_sorted, vals_sorted, row_ptr, row_norm_sq


def _make_pairwise_dist_from_cross():
    def _kernel(row_norm_sq, idx, cross):
        sq_dist = row_norm_sq[:, None] + row_norm_sq[idx][None, :] - 2.0 * cross
        return torch.sqrt(torch.clamp(sq_dist, min=0.0))
    return torch.compile(_kernel, fullgraph=True)


_PAIRWISE_DIST_KERNEL = _make_pairwise_dist_from_cross()


def _chunk_pairwise_dist(S, rows_sorted, cols_sorted, vals_sorted, row_ptr, row_norm_sq, start, end):
    """Compute distances to one graph-row chunk."""
    n = S.shape[0]
    device = S.device
    lo, hi = row_ptr[start], row_ptr[end]
    c = end - start

    chunk_dense = torch.zeros((c, n), dtype=row_norm_sq.dtype, device=device)
    chunk_dense[rows_sorted[lo:hi] - start, cols_sorted[lo:hi]] = vals_sorted[lo:hi]

    cross = torch.sparse.mm(S, chunk_dense.T)
    idx = torch.arange(start, end, device=device)
    dist = _PAIRWISE_DIST_KERNEL(row_norm_sq, idx, cross)      # compiled
    return dist, idx


def radius_counts_chunked(S, rows_sorted, cols_sorted, vals_sorted, row_ptr, row_norm_sq,
                          eps_tensor, chunk_size, temperature=None):
    """Count graph-space neighbors within ``eps`` using row chunks."""
    n = S.shape[0]
    counts = torch.zeros(n, dtype=torch.float32, device=S.device)
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        dist, _ = _chunk_pairwise_dist(
            S, rows_sorted, cols_sorted, vals_sorted, row_ptr, row_norm_sq, start, end
        )
        if temperature is not None:
            chunk_counts = torch.sigmoid((eps_tensor - dist) / temperature).sum(dim=0) - torch.sigmoid(eps_tensor / temperature)
        else:
            chunk_counts = (dist < eps_tensor).sum(dim=0).float() - 1.0
            
        counts[start:end] = chunk_counts
    return counts


def _min_max_dist_chunked(S, rows_sorted, cols_sorted, vals_sorted, row_ptr, row_norm_sq, chunk_size):
    """Return global non-self minimum and maximum graph distances."""
    n = S.shape[0]
    device = S.device
    running_max = torch.tensor(float("-inf"), device=device)
    running_min = torch.tensor(float("inf"), device=device)
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        dist, idx = _chunk_pairwise_dist(
            S, rows_sorted, cols_sorted, vals_sorted, row_ptr, row_norm_sq, start, end
        )
        self_mask = torch.zeros_like(dist, dtype=torch.bool)
        self_mask[idx, torch.arange(idx.shape[0], device=device)] = True
        running_max = torch.maximum(running_max, dist.masked_fill(self_mask, float("-inf")).max())
        running_min = torch.minimum(running_min, dist.masked_fill(self_mask, float("inf")).min())
    return running_max.item(), running_min.item()


def _binary_search_eps_chunked(S, rows_sorted, cols_sorted, vals_sorted, row_ptr, row_norm_sq,
                               low, high, threshold, required, chunk_size, tol=1e-4, max_iter=50):
    """Find the smallest radius satisfying the density condition."""
    lo, hi = low, high
    result = high
    found = False
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        counts = radius_counts_chunked(
            S, rows_sorted, cols_sorted, vals_sorted, row_ptr, row_norm_sq, torch.tensor(mid, device=S.device), chunk_size
        )
        satisfied = int((counts > threshold).sum().item())
        if satisfied >= required:
            result, found, hi = mid, True, mid
        else:
            lo = mid
        if abs(hi - lo) < tol:
            break
    return result, found


class _ImplicitEpsFn(torch.autograd.Function):
    """Differentiate the epsilon selected by the chunked search implicitly."""

    @staticmethod
    def forward(ctx, vals_sorted, rows_sorted, cols_sorted, row_ptr, n,
                threshold, required, chunk_size, count_temperature,
                threshold_temperature, min_dist, max_dist):
        device = vals_sorted.device
        with torch.no_grad():
            vals_d = vals_sorted.detach()
            row_norm_sq = torch.zeros(n, dtype=torch.float32, device=device)
            row_norm_sq.scatter_add_(0, rows_sorted, vals_d ** 2)
            S_hard = torch.sparse_coo_tensor(
                torch.stack([rows_sorted, cols_sorted]), vals_d, size=(n, n),
                device=device, check_invariants=False,
            ).coalesce()

            used_threshold, used_required = threshold, required
            eps_value, found = _binary_search_eps_chunked(
                S_hard, rows_sorted, cols_sorted, vals_d, row_ptr, row_norm_sq,
                min_dist, max_dist, used_threshold, used_required, chunk_size,
            )
            if not found:
                used_threshold = max(1, threshold // 2)
                used_required = required // 2
                eps_value, found = _binary_search_eps_chunked(
                    S_hard, rows_sorted, cols_sorted, vals_d, row_ptr, row_norm_sq,
                    min_dist, max_dist, used_threshold, used_required, chunk_size,
                )
            if not found:
                eps_value = max_dist

        eps_star = torch.tensor(eps_value, dtype=vals_sorted.dtype, device=device)

        ctx.save_for_backward(vals_sorted, rows_sorted, cols_sorted, eps_star)
        ctx.row_ptr = row_ptr
        ctx.n = n
        ctx.chunk_size = chunk_size
        ctx.count_temperature = count_temperature
        ctx.threshold_temperature = threshold_temperature
        ctx.used_threshold = used_threshold
        ctx.used_required = used_required
        ctx.converged = found
        return eps_star

    @staticmethod
    def backward(ctx, grad_output):
        vals_sorted, rows_sorted, cols_sorted, eps_star = ctx.saved_tensors

        if not ctx.converged:
            return (torch.zeros_like(vals_sorted),
                    None, None, None, None, None, None, None, None, None, None, None)

        n = ctx.n
        with torch.enable_grad():
            vals_ = vals_sorted.detach().requires_grad_(True)
            eps_ = eps_star.detach().requires_grad_(True)
            row_norm_sq = torch.zeros(n, dtype=torch.float32, device=vals_.device)
            row_norm_sq.scatter_add_(0, rows_sorted, vals_ ** 2)
            S_soft = torch.sparse_coo_tensor(
                torch.stack([rows_sorted, cols_sorted]), vals_, size=(n, n),
                device=vals_.device, check_invariants=False,
            ).coalesce()
            counts = radius_counts_chunked(
                S_soft, rows_sorted, cols_sorted, vals_, ctx.row_ptr, row_norm_sq,
                eps_, ctx.chunk_size, temperature=ctx.count_temperature,
            )
            residual = (
                torch.sigmoid((counts - ctx.used_threshold) / ctx.threshold_temperature).sum()
                - ctx.used_required
            )
            dR_deps, dR_dvals = torch.autograd.grad(residual, (eps_, vals_))

        # Guard against a near-zero denominator (flat residual w.r.t. eps)
        # without flipping its sign.
        eps_floor = 1e-8
        sign = torch.where(dR_deps >= 0, 1.0, -1.0)
        dR_deps_safe = torch.where(dR_deps.abs() < eps_floor, sign * eps_floor, dR_deps)

        d_eps_d_vals = -dR_dvals / dR_deps_safe          # implicit function theorem
        grad_vals = grad_output * d_eps_d_vals

        return (grad_vals, None, None, None, None, None, None, None, None, None, None, None)


def _calculate_eps_from_similarity(S, rows_sorted, cols_sorted, vals_sorted, row_ptr, row_norm_sq,
                                   n, nbd_sample_count_threshold,
                                   satisfiability_proportion, chunk_size,
                                   differentiable=False, count_temperature=None,
                                   threshold_temperature=None):
    """Calculate epsilon from a prepared sparse similarity layout."""
    max_dist, min_dist = _min_max_dist_chunked(
        S, rows_sorted, cols_sorted, vals_sorted, row_ptr, row_norm_sq, chunk_size
    )
    threshold = (
        max(1, n - 1)
        if nbd_sample_count_threshold >= n
        else nbd_sample_count_threshold
    )
    required = int(satisfiability_proportion * n)

    if differentiable:
        if count_temperature is None or threshold_temperature is None:
            raise ValueError(
                "differentiable=True requires both count_temperature and "
                "threshold_temperature"
            )
        return _ImplicitEpsFn.apply(
            vals_sorted, rows_sorted, cols_sorted, row_ptr, n,
            threshold, required, chunk_size,
            count_temperature, threshold_temperature, min_dist, max_dist,
        )

    eps_value, found = _binary_search_eps_chunked(
        S, rows_sorted, cols_sorted, vals_sorted, row_ptr, row_norm_sq,
        min_dist, max_dist, threshold, required, chunk_size,
    )

    if not found:
        eps_value, found = _binary_search_eps_chunked(
            S, rows_sorted, cols_sorted, vals_sorted, row_ptr, row_norm_sq,
            min_dist, max_dist,
            max(1, threshold // 2), required // 2, chunk_size,
        )

    if not found:
        eps_value = max_dist

    return torch.tensor(eps_value, dtype=torch.float32, device=S.device)


def compute_weights_from_similarity_chunked(S, n, nbd_sample_count_threshold,
                                            satisfiability_proportion, max_iters_weight_count,
                                            chunk_size, temperature=None, eps=None, layout=None,
                                            differentiable_eps=False, eps_count_temperature=None,
                                            eps_threshold_temperature=None):
    """Compute empirical weights with streamed graph-distance passes."""
    if layout is None:
        layout = _sparse_similarity_layout(S)
    rows_sorted, cols_sorted, vals_sorted, row_ptr, row_norm_sq = layout

    if eps is None:
        if differentiable_eps:
            eps_tensor = _calculate_eps_from_similarity(
                S, rows_sorted, cols_sorted, vals_sorted, row_ptr, row_norm_sq,
                n, nbd_sample_count_threshold,
                satisfiability_proportion, chunk_size,
                differentiable=True, count_temperature=eps_count_temperature,
                threshold_temperature=eps_threshold_temperature,
            )
        else:
            with torch.no_grad():
                eps_tensor = _calculate_eps_from_similarity(
                    S, rows_sorted, cols_sorted, vals_sorted, row_ptr, row_norm_sq,
                    n, nbd_sample_count_threshold,
                    satisfiability_proportion, chunk_size,
                )
    else:
        eps_tensor = eps

    delta = (eps_tensor - 1e-6) / max_iters_weight_count
    total_counts = torch.zeros(n, dtype=torch.float32, device=S.device)
    eps_running = eps_tensor

    for _ in range(max_iters_weight_count):
        total_counts = total_counts + radius_counts_chunked(
            S, rows_sorted, cols_sorted, vals_sorted, row_ptr, row_norm_sq, eps_running, chunk_size, temperature
        )
        eps_running = eps_running - delta

    return total_counts / max_iters_weight_count


_DENSE_WEIGHT_STEP_CACHE = {}


def _make_dense_weight_step(has_temperature):
    def _step(dist, eps_running, temperature):
        if has_temperature:
            return torch.sigmoid((eps_running - dist) / temperature).sum(dim=1) - torch.sigmoid(eps_running / temperature)
        else:
            return (dist < eps_running).sum(dim=1).float() - 1.0
    return torch.compile(_step, fullgraph=True)


def _get_dense_weight_step(has_temperature):
    """Return the cached compiled dense weight step."""
    if has_temperature not in _DENSE_WEIGHT_STEP_CACHE:
        _DENSE_WEIGHT_STEP_CACHE[has_temperature] = _make_dense_weight_step(has_temperature)
    return _DENSE_WEIGHT_STEP_CACHE[has_temperature]


def compute_weights_from_similarity_dense(S, n, nbd_sample_count_threshold,
                                          satisfiability_proportion, max_iters_weight_count,
                                          temperature=None, eps=None):
    """Compute empirical weights with dense pairwise distances."""
    S_dense = S.to_dense()
    dist = torch.cdist(S_dense, S_dense)

    if eps is None:
        with torch.no_grad():
            mask = torch.eye(n, dtype=torch.bool, device=S.device)
            valid_dists = dist.masked_fill(mask, float('inf'))
            min_dist = valid_dists.min().item()
            valid_dists_max = dist.masked_fill(mask, float('-inf'))
            max_dist = valid_dists_max.max().item()

            threshold = max(1, n - 1) if nbd_sample_count_threshold >= n else nbd_sample_count_threshold
            required = int(satisfiability_proportion * n)

            def check_eps(eps_val):
                c = (dist < eps_val).sum(dim=1) - 1
                return int((c > threshold).sum().item())

            lo, hi = min_dist, max_dist
            eps_val = hi
            found = False
            for _ in range(50):
                mid = (lo + hi) / 2.0
                if check_eps(mid) >= required:
                    eps_val, found, hi = mid, True, mid
                else:
                    lo = mid
                if abs(hi - lo) < 1e-4:
                    break

            if not found:
                relaxed_thresh = max(1, threshold // 2)
                relaxed_required = required // 2
                lo, hi = min_dist, max_dist
                for _ in range(50):
                    mid = (lo + hi) / 2.0
                    if check_eps(mid) >= relaxed_required:
                        eps_val, found, hi = mid, True, mid
                    else:
                        lo = mid
                    if abs(hi - lo) < 1e-4:
                        break
                        
            if not found:
                eps_val = max_dist

        eps_tensor = torch.tensor(eps_val, dtype=torch.float32, device=S.device)
    else:
        eps_tensor = eps

    delta = (eps_tensor - 1e-6) / max_iters_weight_count
    total_counts = torch.zeros(n, dtype=torch.float32, device=S.device)
    eps_running = eps_tensor

    step_fn = _get_dense_weight_step(temperature is not None)
    for _ in range(max_iters_weight_count):
        total_counts = total_counts + step_fn(dist, eps_running, temperature)
        eps_running = eps_running - delta

    return total_counts / max_iters_weight_count


def get_empirical_weights(
    X,
    nbd_sample_count_threshold=5,
    max_iters_weight_count=4,
    satisfiability_proportion=0.3,
    n_neighbors=15,
    n_epochs=200,
    device=DEFAULT_DEVICE,
    use_chunking=True,
    chunk_size=2048,
    temperature=None,
    eps=None,
    soft_prune=False,
    soft_prune_temperature=None,
    differentiable_eps=False,
    eps_count_temperature=None,
    eps_threshold_temperature=None,
):
    n = X.shape[0]
    S = _build_sparse_similarity(
        X, n_neighbors, n_epochs, device,
        soft_prune=soft_prune, soft_prune_temperature=soft_prune_temperature,
    )

    if differentiable_eps and not use_chunking:
        raise ValueError(
            "differentiable_eps=True (implicit-differentiation eps) is only "
            "implemented for the chunked path -- pass use_chunking=True, or "
            "set differentiable_eps=False to use the dense path as before."
        )

    if use_chunking:
        return compute_weights_from_similarity_chunked(
            S, n, nbd_sample_count_threshold, satisfiability_proportion,
            max_iters_weight_count, chunk_size, temperature, eps,
            differentiable_eps=differentiable_eps,
            eps_count_temperature=eps_count_temperature,
            eps_threshold_temperature=eps_threshold_temperature,
        )
    else:
        return compute_weights_from_similarity_dense(
            S, n, nbd_sample_count_threshold, satisfiability_proportion,
            max_iters_weight_count, temperature, eps
        )


def _build_sparse_weight_matrix(indices_i64, w_norm, n_rows, n_cols, device):
    """Build the sparse row-normalized neighbor matrix."""
    k = indices_i64.shape[1]
    rows = torch.arange(n_rows, device=device).repeat_interleave(k)
    cols = indices_i64.reshape(-1)
    vals = w_norm.reshape(-1)
    return torch.sparse_coo_tensor(
        torch.stack([rows, cols]), vals, size=(n_rows, n_cols), device=device,
        check_invariants=False,
    ).coalesce()


_MOVEMENT_GATE_EPS = 1e-8


def _make_movement_kernel(clipping, clip_mode, needs_gather, low_precision, low_precision_dtype=torch.bfloat16,
                           smooth_movement_gate=False):
    """Build the compiled dense movement kernel."""

    def _kernel(X, indices, w_or_barycenter, learning_rate, alpha, gate, corpus):
        n, k = indices.shape
        neighbor_pos = None

        if needs_gather:
            neighbor_pos = corpus[indices]
            if low_precision:
                lp_w = w_or_barycenter.to(low_precision_dtype)
                lp_neighbors = neighbor_pos.to(low_precision_dtype)
                barycenter = (lp_w.unsqueeze(-1) * lp_neighbors).sum(dim=1).to(X.dtype)
            else:
                barycenter = (w_or_barycenter.unsqueeze(-1) * neighbor_pos).sum(dim=1)
        else:
            barycenter = w_or_barycenter

        diff = barycenter - X
        dist_move = diff.norm(dim=1)

        if clipping and clip_mode > 0:
            if neighbor_pos is None:
                neighbor_pos = corpus[indices]
            dists = torch.cdist(X.unsqueeze(1), neighbor_pos).squeeze(1)
            median_dist = dists.sort(dim=1).values[:, k // 2]
            delta = (alpha * median_dist).clamp_min(1e-8)
            if clip_mode == 1:
                effective_step = dist_move * (delta / (delta + dist_move))
            else:
                effective_step = torch.minimum(dist_move, delta)
        else:
            effective_step = dist_move

        dist_move_safe = dist_move.clamp_min(1e-12)
        scale = (learning_rate * effective_step / dist_move_safe).unsqueeze(-1)
        step = gate * scale * diff

        if smooth_movement_gate:
            movement_gate = dist_move / (dist_move + _MOVEMENT_GATE_EPS)
            revised_d = X + movement_gate.unsqueeze(-1) * step
            change = movement_gate * dist_move
        else:
            moved = dist_move >= _MOVEMENT_GATE_EPS
            updated = X + step
            revised_d = torch.where(moved.unsqueeze(-1), updated, X)
            change = torch.where(moved, dist_move, torch.zeros_like(dist_move))

        return revised_d, change

    return torch.compile(_kernel, fullgraph=True)


def _make_shift_kernel(clipping, clip_mode, use_sparse, low_precision, low_precision_dtype=torch.bfloat16,
                        smooth_movement_gate=False):
    """Build one cached shift step for a fixed configuration."""
    movement_kernel = _make_movement_kernel(
        clipping, clip_mode, needs_gather=not use_sparse,
        low_precision=low_precision, low_precision_dtype=low_precision_dtype,
        smooth_movement_gate=smooth_movement_gate,
    )

    if use_sparse:
        def _shift(X, indices, w_or_W, learning_rate, alpha, gate, corpus):
            barycenter = torch.sparse.mm(w_or_W, corpus)
            return movement_kernel(X, indices, barycenter, learning_rate, alpha, gate, corpus)
    else:
        def _shift(X, indices, w_or_W, learning_rate, alpha, gate, corpus):
            return movement_kernel(X, indices, w_or_W, learning_rate, alpha, gate, corpus)

    return _shift


_SHIFT_KERNEL_CACHE = {}


def _get_shift_kernel(clipping, clip_mode, use_sparse, low_precision, smooth_movement_gate=False):
    key = (clipping, clip_mode, use_sparse, low_precision, smooth_movement_gate)
    if key not in _SHIFT_KERNEL_CACHE:
        _SHIFT_KERNEL_CACHE[key] = _make_shift_kernel(
            clipping, clip_mode, use_sparse, low_precision, smooth_movement_gate=smooth_movement_gate,
        )
    return _SHIFT_KERNEL_CACHE[key]


# ---------------------------------------------------------------------------
# Strict end-to-end differentiable path
# ---------------------------------------------------------------------------
# The original MSDE path deliberately uses a hard sparse graph. That path is
# retained above for speed/backwards compatibility. The helpers below are a
# continuous relaxation of the same ingredients:
#   * soft k-NN is obtained by solving a smooth occupancy equation
#       sum_j sigmoid((tau_i - d_ij) / T) = k
#     rather than calling topk;
#   * fuzzy graph union is computed densely, with no data-dependent indexing;
#   * rho and sigma are obtained by fixed, differentiable Newton iterations;
#   * epsilon is obtained by a smooth root solve over soft counts;
#   * pruning is a sigmoid gate, never a nonzero()/hard active-edge subset;
#   * movement uses dense weighted barycenters and a smooth zero-movement gate;
#   * all iteration counts are fixed (no data-dependent early stopping).
#
# This path is O(N^2) in memory/time for an N-point manifold. That cost is the
# unavoidable trade-off for exact end-to-end gradients through neighbor
# membership and graph topology without a learned/approximate sparse kernel.

_DIFF_EPS = 1e-8
_DIFF_LOG_SCALE_EPS = 1e-6
_DIFF_SOLVER_ITERATIONS = 25
_DEFAULT_DIFF_TEMPERATURE = 0.1


def _smooth_pairwise_dist(X, Y):
    """Pairwise Euclidean distances with a smooth zero-distance guard."""
    diff = X.unsqueeze(1) - Y.unsqueeze(0)
    return torch.sqrt(diff.square().sum(dim=-1) + _DIFF_EPS**2)


def _masked_soft_knn_weights(
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
    temperature = torch.sqrt(temperature.square() + _DIFF_EPS**2)

    d = _smooth_pairwise_dist(X, corpus)
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

    mean_d = (d * valid_f).sum(dim=1) / (valid_count + _DIFF_EPS)
    tau = mean_d

    max_tau_step = 4.0 * (mean_d.detach().mean() + 1.0)
    max_tau_step = torch.as_tensor(max_tau_step, dtype=d.dtype, device=d.device)

    for _ in range(iterations):
        membership = torch.sigmoid((tau.unsqueeze(1) - d) / temperature) * valid_f
        residual = membership.sum(dim=1) - target
        derivative = (
            membership * (1.0 - membership)
        ).sum(dim=1) / temperature
        raw_step = residual / (derivative + _DIFF_EPS)
        step = max_tau_step * torch.tanh(raw_step / (max_tau_step + _DIFF_EPS))
        tau = tau - step

    membership = torch.sigmoid((tau.unsqueeze(1) - d) / temperature) * valid_f
    return d, membership, valid


def _soft_fuzzy_graph(
    X,
    k,
    temperature,
    prune_temperature,
    n_epochs,
    exclude_self=True,
    sigma_iterations=20,
    knn_iterations=20,
):
    """Build a dense, differentiable fuzzy graph and density prior."""
    d, membership, valid = _masked_soft_knn_weights(
        X,
        X,
        k,
        temperature,
        exclude_self=exclude_self,
        iterations=knn_iterations,
    )
    valid_f = valid.to(d.dtype)
    valid_count = valid_f.sum(dim=1)

    # Smooth approximation to UMAP's rho = first non-zero neighbor distance.
    safe_d = d + (~valid).to(d.dtype) * 1.0e6
    rho_temp = temperature
    rho = -rho_temp * torch.logsumexp(-safe_d / rho_temp, dim=1)
    rho = torch.where(valid_count > 0, rho, torch.zeros_like(rho))

    # Smooth positive approximation to relu(d-rho).
    offset = rho_temp * F.softplus((d - rho.unsqueeze(1)) / rho_temp)

    # Smooth replacement for the rho/sigma binary search. We solve in log(sigma)
    # so positivity is guaranteed, and bound each Newton step with tanh rather
    # than a hard clamp.
    sigma_target = torch.minimum(
        valid_count,
        torch.full_like(valid_count, float(torch.log2(torch.tensor(max(k, 1), dtype=d.dtype)))),
    )
    # Prevent an impossible exact target when fewer points than the desired
    # target are available.
    sigma_target = sigma_target * (valid_count > 0).to(d.dtype)
    init_sigma = (offset * valid_f).sum(dim=1) / (valid_count + _DIFF_EPS)
    log_sigma = torch.log(init_sigma + _DIFF_LOG_SCALE_EPS)

    max_log_sigma_step = 2.0
    for _ in range(sigma_iterations):
        sigma = torch.exp(log_sigma)
        q = torch.exp(-offset / sigma) * valid_f
        residual = q.sum(dim=1) - sigma_target
        derivative = (q * offset / sigma).sum(dim=1)
        raw_step = residual / (derivative + _DIFF_EPS)
        step = max_log_sigma_step * torch.tanh(raw_step / max_log_sigma_step)
        log_sigma = log_sigma - step

    sigma = torch.exp(log_sigma)
    fuzzy = membership * torch.exp(-offset / sigma)

    # Dense fuzzy union: w + w^T - w*w^T. No edge matching/index search.
    w_sym = fuzzy + fuzzy.transpose(0, 1) - fuzzy * fuzzy.transpose(0, 1)

    # Soft pruning. The threshold uses a smooth log-sum-exp approximation of
    # max() and then a sigmoid gate; importantly, there is no active-edge
    # nonzero()/hard mask.
    prune_temperature = torch.as_tensor(
        prune_temperature, dtype=X.dtype, device=X.device
    )
    prune_temperature = torch.sqrt(prune_temperature.square() + _DIFF_EPS**2)
    smooth_max = prune_temperature * torch.logsumexp(
        w_sym.reshape(-1) / prune_temperature, dim=0
    )
    threshold = smooth_max / max(n_epochs, 1)
    gate = torch.sigmoid((w_sym - threshold) / prune_temperature)
    w_soft = w_sym * gate

    return w_soft


def _smooth_epsilon_from_graph(
    graph,
    nbd_sample_count_threshold,
    satisfiability_proportion,
    count_temperature,
    threshold_temperature,
    iterations=25,
):
    """Differentiate through epsilon selection using a smooth fixed-point solve."""
    n = graph.shape[0]
    d_graph = _smooth_pairwise_dist(graph, graph)
    offdiag = ~torch.eye(n, dtype=torch.bool, device=graph.device)
    valid = offdiag.to(d_graph.dtype)

    count_temperature = torch.as_tensor(
        count_temperature, dtype=graph.dtype, device=graph.device
    ).square().add(_DIFF_EPS**2).sqrt()
    threshold_temperature = torch.as_tensor(
        threshold_temperature, dtype=graph.dtype, device=graph.device
    ).square().add(_DIFF_EPS**2).sqrt()

    mean_dist = (d_graph * valid).sum() / (valid.sum() + _DIFF_EPS)
    log_eps = torch.log(mean_dist + _DIFF_LOG_SCALE_EPS)

    threshold = float(max(1, n - 1) if nbd_sample_count_threshold >= n else nbd_sample_count_threshold)
    target_fraction = torch.as_tensor(
        float(satisfiability_proportion), dtype=graph.dtype, device=graph.device
    )

    max_log_eps_step = 2.0
    for _ in range(iterations):
        eps = torch.exp(log_eps)
        soft_counts = (
            torch.sigmoid((eps - d_graph) / count_temperature) * valid
        ).sum(dim=1)
        satisfaction = torch.sigmoid(
            (soft_counts - threshold) / threshold_temperature
        )
        residual = satisfaction.mean() - target_fraction

        dcount_deps = (
            torch.sigmoid((eps - d_graph) / count_temperature)
            * (1.0 - torch.sigmoid((eps - d_graph) / count_temperature))
            / count_temperature
            * valid
        ).sum(dim=1)
        dsat_dcount = satisfaction * (1.0 - satisfaction) / threshold_temperature
        dres_deps = (dsat_dcount * dcount_deps).mean()
        dres_dlogeps = dres_deps * eps
        raw_step = residual / (dres_dlogeps + _DIFF_EPS)
        step = max_log_eps_step * torch.tanh(raw_step / max_log_eps_step)
        log_eps = log_eps - step

    return torch.exp(log_eps), d_graph, valid


def _differentiable_empirical_weights(
    X,
    n_neighbors,
    n_epochs,
    nbd_sample_count_threshold,
    satisfiability_proportion,
    max_iters_weight_count,
    temperature,
    eps=None,
    eps_threshold_temperature=None,
    eps_solver_iterations=25,
):
    """Fully differentiable analogue of the empirical density-weight pass."""
    if temperature is None:
        raise ValueError(
            "fully_differentiable=True requires temperature > 0 so all counting "
            "operations remain smooth"
        )
    if eps_threshold_temperature is None:
        eps_threshold_temperature = temperature

    graph = _soft_fuzzy_graph(
        X,
        n_neighbors,
        temperature,
        temperature,
        n_epochs,
        knn_iterations=eps_solver_iterations,
    )

    if eps is None:
        eps_tensor, graph_dist, valid_graph = _smooth_epsilon_from_graph(
            graph,
            nbd_sample_count_threshold,
            satisfiability_proportion,
            temperature,
            eps_threshold_temperature,
            iterations=eps_solver_iterations,
        )
    else:
        eps_tensor = eps
        graph_dist = _smooth_pairwise_dist(graph, graph)
        valid_graph = (
            ~torch.eye(graph.shape[0], dtype=torch.bool, device=graph.device)
        ).to(graph.dtype)

    # Sample the soft count at a fixed sequence of positive radii. Unlike the
    # old implementation there is no discrete indicator and no self subtraction
    # because the fixed diagonal is excluded by valid_graph.
    total_counts = torch.zeros(X.shape[0], dtype=X.dtype, device=X.device)
    steps = max(max_iters_weight_count, 1)
    for i in range(steps):
        frac = (i + 0.5) / steps
        eps_running = eps_tensor * (1.0 - frac) + _DIFF_LOG_SCALE_EPS * frac
        counts = (
            torch.sigmoid((eps_running - graph_dist) / (torch.sqrt(torch.as_tensor(temperature, dtype=X.dtype, device=X.device).square() + _DIFF_EPS**2)))
            * valid_graph
        ).sum(dim=1)
        total_counts = total_counts + counts

    return total_counts / steps


def _soft_minimum(a, b, temperature):
    """Smooth approximation to min(a,b)."""
    return 0.5 * (
        a + b - torch.sqrt((a - b).square() + temperature**2)
    )


def _differentiable_shift_step(
    X,
    corpus,
    base_weights,
    k,
    learning_rate,
    alpha,
    gate,
    temperature,
    labels=None,
    corpus_labels=None,
    reference_mode=False,
    clipping=False,
    clip_mode=0,
    smooth_iterations=20,
):
    """One fully differentiable MSDE movement step."""
    d, membership, _ = _masked_soft_knn_weights(
        X,
        corpus,
        k,
        temperature,
        labels=labels,
        corpus_labels=corpus_labels,
        exclude_self=not reference_mode,
        iterations=smooth_iterations,
    )

    # base_weights are row-wise density priors for the corpus, exactly like the
    # old selected-neighbor weights, but every corpus point now participates
    # with a continuous soft-kNN membership weight.
    w = membership * base_weights.unsqueeze(0)
    mass = w.sum(dim=1)
    numerator = w @ corpus
    barycenter = numerator / (mass.unsqueeze(1) + _DIFF_EPS)

    # A smooth validity/orphan gate: if mass -> 0, the proposed movement -> 0.
    valid_gate = mass / (mass + _DIFF_EPS)
    diff = barycenter - X
    dist_move = torch.sqrt(diff.square().sum(dim=1) + _DIFF_EPS**2)

    effective_step = dist_move
    if clipping and clip_mode > 0:
        neighbor_mean_dist = (membership * d).sum(dim=1) / (membership.sum(dim=1) + _DIFF_EPS)
        delta = alpha * neighbor_mean_dist
        if clip_mode == 1:
            effective_step = dist_move * (delta / (delta + dist_move + _DIFF_EPS))
        else:
            effective_step = _soft_minimum(
                dist_move,
                delta,
                torch.sqrt(torch.as_tensor(temperature, dtype=X.dtype, device=X.device).square() + _DIFF_EPS**2),
            )
    elif clipping:
        effective_step = dist_move

    movement_gate = dist_move / (dist_move + _DIFF_EPS)
    scale = learning_rate * effective_step / (dist_move + _DIFF_EPS)
    step = gate * valid_gate.unsqueeze(1) * movement_gate.unsqueeze(1) * scale.unsqueeze(1) * diff
    revised = X + step
    change = valid_gate * movement_gate * dist_move
    return revised, change


class MeanShiftDensityEnhancement(torch.nn.Module):
    def __init__(
        self,
        k=30,
        nbd_sample_count_threshold=30,
        learning_rate=0.3,
        max_iters_shift=5,
        shift_threshold=0.0001,
        clipping=False,
        clip_mode=0,
        alpha=0.5,
        device=DEFAULT_DEVICE,
        keep_trajectory=False,
        log_file=None,
        use_chunking=False,
        weight_chunk_size=4096,
        temperature=None,
        eps=None,
        enable_gradients=True,
        learn_temperature=False,
        learn_learning_rate=False,
        learn_alpha=False,
        learn_eps=False,
        use_sparse_shift=True,
        low_precision_barycenter=False,
        recompute_neighbors=0,
        smooth_movement_gate=False,
        fully_differentiable=False,
        # Legacy differentiability controls are retained for source compatibility.
        # In fully_differentiable mode they are derived automatically from
        # `temperature` and should normally not be specified.
        differentiable_solver_iterations=None,
        use_soft_topk=False,
        soft_topk_temperature=None,
        learn_soft_topk_temperature=False,
        X=None,
        labels=None,
    ):
        """
        smooth_movement_gate : legacy control for the original sparse path.
            In ``fully_differentiable=True`` mode the movement gate is always
            smooth, so this option is not needed.
        fully_differentiable : when True, use the dense continuous relaxation of
            the entire MSDE computation. This is the only switch needed to turn
            on end-to-end differentiability. Set ``temperature`` to control the
            smoothness of neighbour assignment and density counting. The same
            temperature is automatically reused for every relaxation in this
            mode; solver iteration counts and pruning/movement temperatures are
            chosen internally. The differentiable path is O(N^2).
            Typical usage is simply::

                msde = MeanShiftDensityEnhancement(
                    k=30, temperature=0.1, fully_differentiable=True
                )

            No separate soft-top-k, pruning, epsilon, or solver-temperature
            hyperparameters are needed.
        differentiable_solver_iterations : legacy/advanced override. In normal
            use leave this as None; the differentiable path uses a fixed internal
            solver budget.
        use_soft_topk, soft_topk_temperature, learn_soft_topk_temperature :
            legacy controls for the intermediate partially-differentiable path.
            They are ignored in ``fully_differentiable=True`` mode because soft
            neighbour assignment is enabled automatically.
        X : optional (m, d) tensor -- a fixed reference manifold. When
            given, forward() no longer shifts its input against itself:
            every point of the (different) X passed to forward() is
            instead shifted with respect to neighbours found in *this* X,
            which stays fixed for the lifetime of the module. When omitted
            (default), forward() behaves as before -- each call's X is
            shifted with respect to itself.
        labels : optional (m,) integer tensor/array-like, aligned with X.
            Class labels for the reference manifold. When given (requires
            X to also be given here), forward() restricts neighbour search
            to same-class points for any call where forward() is also
            given labels: a point of forward()'s input can only be shifted
            towards reference points sharing its label (forward()'s
            labels are optional per-call -- see forward()'s docstring).
            Can also be set or changed later via
            set_reference_manifold(X_ref_or_None, labels=...) without
            passing X here.
        """
        super().__init__()

        # In fully differentiable mode, `temperature` is the single relaxation
        # hyperparameter. Everything else is derived internally.
        if fully_differentiable and temperature is None:
            temperature = _DEFAULT_DIFF_TEMPERATURE
        if temperature is not None and float(torch.as_tensor(temperature).detach().cpu()) <= 0:
            raise ValueError("temperature must be > 0")
        if learn_temperature and temperature is None:
            raise ValueError("learn_temperature=True requires an initial temperature")
        if learn_eps and temperature is None:
            raise ValueError("learn_eps=True requires temperature to enable differentiable weights")
        if use_soft_topk and soft_topk_temperature is None and not fully_differentiable:
            raise ValueError("use_soft_topk=True requires an explicit soft_topk_temperature")
        if learn_soft_topk_temperature and not use_soft_topk and not fully_differentiable:
            raise ValueError("learn_soft_topk_temperature=True requires use_soft_topk=True")
        if fully_differentiable and learn_eps and eps is None:
            raise ValueError(
                "fully_differentiable=True with learn_eps=True requires an explicit initial eps"
            )
        if differentiable_solver_iterations is not None:
            if not isinstance(differentiable_solver_iterations, int) or differentiable_solver_iterations < 1:
                raise ValueError("differentiable_solver_iterations must be a positive integer")
        if recompute_neighbors is not None and (
            not isinstance(recompute_neighbors, int) or isinstance(recompute_neighbors, bool) or recompute_neighbors < 0
        ):
            raise ValueError(
                "recompute_neighbors must be None, 0 (never recompute after the "
                "first iteration -- original fixed-graph behaviour), or a "
                "positive int N (recompute the neighbour graph every N "
                "iterations, i.e. at iter_count 0, N, 2N, ...)."
            )
        self.k = k
        self.nbd_sample_count_threshold = nbd_sample_count_threshold
        self.max_iters_shift = max_iters_shift
        self.shift_threshold = shift_threshold
        self.clipping = clipping
        self.clip_mode = clip_mode
        self.device_name = device
        self.keep_trajectory = keep_trajectory
        self.use_chunking = use_chunking
        self.weight_chunk_size = weight_chunk_size
        self.enable_gradients = enable_gradients
        self.log_file = log_file
        self.learn_eps = learn_eps
        self.smooth_movement_gate = (True if fully_differentiable else smooth_movement_gate)
        self.fully_differentiable = fully_differentiable
        self.differentiable_solver_iterations = (
            differentiable_solver_iterations if differentiable_solver_iterations is not None
            else _DIFF_SOLVER_ITERATIONS
        )
        self.use_soft_topk = (True if fully_differentiable else use_soft_topk)
        # Normalize None -> 0 so the forward()-loop guard can treat both as
        # "falsy => never recompute after the first iteration" uniformly.
        self.recompute_neighbors = recompute_neighbors or 0

        # Resolved once per instance (device doesn't change afterward), not
        # per forward() call: whether to use the sparse-spmm barycenter path
        # (no per-iteration (n, k, d) gather) vs. the dense gather fallback.
        self.use_sparse_shift = use_sparse_shift and _sparse_mm_supported(device)
        self.low_precision_barycenter = low_precision_barycenter
        self._shift_kernel = _get_shift_kernel(
            self.clipping, self.clip_mode, self.use_sparse_shift, self.low_precision_barycenter,
            smooth_movement_gate=self.smooth_movement_gate,
        )

        dtype = next(
            (value.dtype for value in (learning_rate, alpha, temperature, eps, soft_topk_temperature)
             if isinstance(value, torch.Tensor) and value.is_floating_point()),
            torch.get_default_dtype(),
        )
        # In fully differentiable mode there is deliberately one public
        # smoothness parameter: `temperature`. The legacy
        # `learn_soft_topk_temperature=True` flag therefore also means
        # "learn the single temperature" for backwards compatibility.
        if fully_differentiable:
            soft_topk_temperature = temperature
            learn_temperature = learn_temperature or learn_soft_topk_temperature

        scalar_options = (
            ("learning_rate", learning_rate, learn_learning_rate),
            ("alpha", alpha, learn_alpha),
            ("temperature", temperature, learn_temperature),
            ("eps", eps, learn_eps),
        )
        if not fully_differentiable:
            scalar_options = scalar_options + (
                ("soft_topk_temperature", soft_topk_temperature, learn_soft_topk_temperature),
            )

        self.learnable_parameters = {}
        for name, value, learnable in scalar_options:
            if name == "eps" and learnable and value is None:
                self.register_buffer(name, None)
                continue

            scalar = _prepare_scalar(
                value, name, device, dtype,
                enable_gradients and learnable,
            )
            if enable_gradients and learnable:
                parameter = torch.nn.Parameter(scalar)
                setattr(self, name, parameter)
                self.learnable_parameters[name] = parameter
            else:
                self.register_buffer(name, scalar)

        # In fully differentiable mode, expose the legacy soft-top-k parameter
        # name as an alias to the single temperature Parameter. It is the same
        # object, so there is still only one learnable smoothness hyperparameter.
        if fully_differentiable:
            self.soft_topk_temperature = self.temperature
            if enable_gradients and learn_soft_topk_temperature:
                self.learnable_parameters["soft_topk_temperature"] = self.temperature

        # Reference-manifold mode: X given here is a fixed corpus that
        # every future forward(X_new) shifts X_new against, instead of
        # X_new shifting against itself. See set_reference_manifold() for
        # updating/clearing it after construction.
        self.set_reference_manifold(X, labels=labels)

        _configure_logging(log_file)

    def _set_ref_labels(self, labels, n_ref):
        """Validate and (re)register self._ref_labels. `labels=None` clears it."""
        if labels is not None:
            ref_labels = torch.as_tensor(labels, device=self.device_name).detach().long()
            if ref_labels.dim() != 1 or ref_labels.shape[0] != n_ref:
                raise ValueError(
                    f"labels must be 1D with length matching the reference manifold "
                    f"(n_ref={n_ref}); got shape {tuple(ref_labels.shape)}"
                )
            self._buffers.pop("_ref_labels", None)
            self.register_buffer("_ref_labels", ref_labels)
        else:
            self._buffers.pop("_ref_labels", None)
            self.register_buffer("_ref_labels", None)

    def set_reference_manifold(self, X, labels=None):
        """
        Set, replace, clear, or (re)label the fixed reference manifold used
        by forward() -- outside of and after __init__.

        X : (n_ref, d) tensor/array-like, or None.
            If not None, becomes the new self.X_ref: forward()'s future
            calls will shift their (different) input against neighbours
            found in this fixed manifold instead of against themselves.
            Its neighbour-density weights are (re)computed once here, not
            per forward() call, since the manifold is fixed until this
            method is called again. Stored detached/no-grad -- it's meant
            as fixed reference data, not something trained via forward()'s
            gradients. `labels` is applied alongside it (see below).
            If None and `labels` is also None, clears any existing
            reference manifold (and any stored labels): forward() reverts
            to shifting its input against itself.
            If None but `labels` is given, X_ref itself is left exactly as
            it is (no recompute of X_ref or its base weights) -- only its
            labels are set/replaced. This requires a reference manifold to
            already be set; call set_reference_manifold(X, labels=...)
            (or pass X at __init__) first if not.
        labels : optional (n_ref,) integer tensor/array-like, aligned with
            X (or with the existing self.X_ref, if X is None here). When
            given, forward() restricts neighbour search to same-class
            points -- see forward()'s docstring. Pass labels=None
            (default) to leave existing labels alone when X is given, or
            to clear everything when X is also None.
        """
        if X is not None:
            X_ref = torch.as_tensor(
                X, device=self.device_name
            ) if self.fully_differentiable else torch.as_tensor(X, device=self.device_name).detach()
            if X_ref.dim() != 2:
                raise ValueError(
                    f"X (reference manifold) must be 2D (n_ref, d); got shape {tuple(X_ref.shape)}"
                )
            self._buffers.pop("X_ref", None)
            self.register_buffer("X_ref", X_ref)
            if self.fully_differentiable:
                # The strict path recomputes the reference density prior inside
                # forward() so it remains connected to learnable temperatures/eps
                # and to a reference tensor that still requires gradients.
                self._ref_base_weights = None
            else:
                with torch.no_grad():
                    self._ref_base_weights = self._compute_base_weights(self.X_ref).detach()
            self._set_ref_labels(labels, X_ref.shape[0])
        elif labels is not None:
            if getattr(self, "X_ref", None) is None:
                raise ValueError(
                    "set_reference_manifold(X=None, labels=...) updates labels on an "
                    "existing reference manifold, but none is set. Call "
                    "set_reference_manifold(X, labels=...) (or pass X at __init__) first."
                )
            self._set_ref_labels(labels, self.X_ref.shape[0])
        else:
            self._buffers.pop("X_ref", None)
            self.register_buffer("X_ref", None)
            self._set_ref_labels(None, 0)
            self._ref_base_weights = None

    def _compute_base_weights(self, X):
        """
        The "how typical/dense is this point" prior, one weight per row of
        `X`. Factored out of forward() so the same logic can be run either
        on forward()'s own X (self-shift mode) or once, at init time, on
        the fixed reference manifold (reference-shift mode) -- see
        __init__ and forward() below.
        """
        if self.learn_eps and self.eps is None:
            similarity = _build_sparse_similarity(X, 15, 200, self.device_name)
            layout = _sparse_similarity_layout(similarity)
            with torch.no_grad():
                eps_initial = _calculate_eps_from_similarity(
                    similarity,
                    *layout,
                    X.shape[0],
                    self.nbd_sample_count_threshold,
                    0.3,
                    self.weight_chunk_size,
                )
            self._buffers.pop("eps", None)
            self.eps = torch.nn.Parameter(eps_initial.to(dtype=X.dtype))
            self.learnable_parameters["eps"] = self.eps

            if self.use_chunking:
                base_weights_t = compute_weights_from_similarity_chunked(
                    similarity,
                    X.shape[0],
                    self.nbd_sample_count_threshold,
                    0.3,
                    4,
                    self.weight_chunk_size,
                    self.temperature,
                    self.eps,
                    layout=layout,
                )
            else:
                base_weights_t = compute_weights_from_similarity_dense(
                    similarity,
                    X.shape[0],
                    self.nbd_sample_count_threshold,
                    0.3,
                    4,
                    self.temperature,
                    self.eps,
                )
        else:
            base_weights_t = get_empirical_weights(
                X,
                nbd_sample_count_threshold=self.nbd_sample_count_threshold,
                max_iters_weight_count=4,
                satisfiability_proportion=0.3,
                chunk_size=self.weight_chunk_size,
                device=self.device_name,
                use_chunking=self.use_chunking,
                temperature=self.temperature,
                eps=self.eps,
            )
        return base_weights_t

    def _forward_fully_differentiable(self, X, gate=1.0, labels=None):
        """Execute the continuous end-to-end differentiable MSDE path."""
        if not self.enable_gradients:
            X = X.detach()

        reference_mode = self.X_ref is not None
        ref_labels = self._ref_labels if reference_mode else None

        if reference_mode:
            if X.dim() != 2 or X.shape[1] != self.X_ref.shape[1]:
                raise ValueError(
                    f"X passed to forward() has feature dim {tuple(X.shape[1:])}, "
                    f"which doesn't match the reference manifold's feature dim "
                    f"{tuple(self.X_ref.shape[1:])}."
                )
            corpus = self.X_ref
        else:
            corpus = X

        if labels is not None and reference_mode and ref_labels is None:
            raise ValueError(
                "labels was passed to forward(), but the reference manifold has no labels"
            )

        masked = labels is not None
        if masked:
            labels = torch.as_tensor(labels, device=self.device_name).detach().long()
            if labels.dim() != 1 or labels.shape[0] != X.shape[0]:
                raise ValueError(
                    f"labels must be 1D with length matching X (n={X.shape[0]}); "
                    f"got shape {tuple(labels.shape)}"
                )

        # A single temperature controls every smooth relaxation.
        count_temperature = self.temperature
        knn_temperature = count_temperature
        eps_threshold_temperature = count_temperature

        # Density prior. In reference mode this is recomputed in differentiable
        # mode so learnable temperatures/eps still participate in the graph.
        base_weights = _differentiable_empirical_weights(
            corpus,
            self.k,
            200,
            self.nbd_sample_count_threshold,
            0.3,
            4,
            count_temperature,
            eps=self.eps,
            eps_threshold_temperature=eps_threshold_temperature,
            eps_solver_iterations=self.differentiable_solver_iterations,
        )

        shifted_dataset = X.clone()
        total_distance = torch.zeros(
            X.shape[0], dtype=X.dtype, device=X.device
        )
        trajectory = [shifted_dataset.clone()] if self.keep_trajectory else []

        # Fixed iteration count: no .item()-based convergence branch can change
        # the computational graph.
        for iter_count in range(self.max_iters_shift):
            corpus_for_shift = self.X_ref if reference_mode else shifted_dataset
            corpus_labels = ref_labels if reference_mode else labels
            shifted_dataset, change = _differentiable_shift_step(
                shifted_dataset,
                corpus_for_shift,
                base_weights,
                self.k,
                self.learning_rate,
                self.alpha,
                gate,
                knn_temperature,
                labels=labels if masked else None,
                corpus_labels=corpus_labels if masked else None,
                reference_mode=reference_mode,
                clipping=self.clipping,
                clip_mode=self.clip_mode,
                smooth_iterations=self.differentiable_solver_iterations,
            )
            total_distance = total_distance + change
            if self.keep_trajectory:
                trajectory.append(shifted_dataset.clone())

            # Logging only; this scalar never controls execution.
            logger.debug(
                f"Differentiable iter {iter_count + 1}: "
                f"mean change = {float(change.detach().mean()):.6f}"
            )

        return shifted_dataset, total_distance, trajectory

    def forward(self, X, gate=1.0, labels=None):
        """
        Run MSDE and return shifted data, movement, and trajectory.

        If a reference manifold was passed at init time (or via
        set_reference_manifold()), every point of this X is shifted with
        respect to neighbours found in that fixed reference manifold
        (self.X_ref never moves). Otherwise X is shifted with respect to
        itself, as before.

        labels : optional (n,) integer tensor/array-like, aligned with X.
            Class labels for this call's X. When given, neighbour search
            is restricted per-class -- each point can only be shifted
            towards same-class neighbours (found in self.X_ref if a
            reference manifold is set, otherwise in X itself).
            - Self-shift mode: pass labels here only; per-point classes
              are compared against each other within X.
            - Reference-shift mode: if the reference manifold has labels
              (set via the `labels=` argument to __init__ or
              set_reference_manifold()), passing labels here is optional
              -- omit them to do an unmasked (all-classes) shift for this
              call even though the reference is labelled, or pass them to
              mask per-class as usual. If the reference manifold does NOT
              have labels, passing labels here is an error: there is
              nothing on the reference side to mask against.
            A point with fewer than k same-class neighbours available
            simply gets fewer effective neighbours (the rest contribute
            zero weight); a point with *no* same-class neighbours at all
            is left unmoved for that call (no valid direction to shift
            it in) rather than shifted towards a meaningless barycenter.
        """
        if self.fully_differentiable:
            return self._forward_fully_differentiable(X, gate=gate, labels=labels)

        if not self.enable_gradients:
            X = X.detach()

        reference_mode = self.X_ref is not None
        ref_labels = self._ref_labels if reference_mode else None

        if reference_mode:
            if X.dim() != 2 or X.shape[1] != self.X_ref.shape[1]:
                raise ValueError(
                    f"X passed to forward() has feature dim {tuple(X.shape[1:])}, "
                    f"which doesn't match the reference manifold's feature dim "
                    f"{tuple(self.X_ref.shape[1:])}. Reference-shift mode requires "
                    f"both to live in the same feature space; point count and "
                    f"order may still differ freely."
                )
            base_weights_t = self._ref_base_weights
        else:
            base_weights_t = self._compute_base_weights(X)

        if labels is not None and reference_mode and ref_labels is None:
            raise ValueError(
                "labels was passed to forward(), but the reference manifold has no "
                "labels (none were given to __init__ or set_reference_manifold()). "
                "There's nothing on the reference side to mask against -- either call "
                "set_reference_manifold(self.X_ref, labels=...) to label the reference "
                "manifold, or drop labels= from this forward() call."
            )

        masked = labels is not None
        if masked:
            labels = torch.as_tensor(labels, device=self.device_name).detach().long()
            if labels.dim() != 1 or labels.shape[0] != X.shape[0]:
                raise ValueError(
                    f"labels must be 1D with length matching X (n={X.shape[0]}); "
                    f"got shape {tuple(labels.shape)}"
                )

        n_samples = X.shape[0]
        shifted_dataset = X.clone()
        total_distance = torch.zeros(n_samples, device=self.device_name)
        trajectory = [shifted_dataset.clone()] if self.keep_trajectory else []

        logger.info(
            f"Computing fixed k-NN (k={self.k}) in feature space on {self.device_name} ..."
        )

        indices_fixed = None
        w_or_W = None
        orphan_mask = None    # (n,) bool -- points with zero same-class neighbours this recompute

        corpus_size = self.X_ref.shape[0] if reference_mode else n_samples
        corpus_labels = ref_labels if reference_mode else labels   # self mode: corpus IS X, so its labels are `labels`

        for iter_count in range(self.max_iters_shift):
            corpus_for_shift = self.X_ref if reference_mode else shifted_dataset

            if iter_count == 0 or (
                self.recompute_neighbors and iter_count % self.recompute_neighbors == 0
            ):
                with torch.no_grad():
                    knn_kwargs = {}
                    if masked:
                        knn_kwargs["labels"] = labels
                        if reference_mode:
                            knn_kwargs["corpus_labels"] = corpus_labels

                    indices_fixed_i64 = compute_fixed_knn(
                        shifted_dataset.detach(), self.k, device=self.device_name,
                        corpus=(self.X_ref.detach() if reference_mode else None),
                        **knn_kwargs,
                    )
                    indices_fixed = indices_fixed_i64.to(torch.int32)

                w = base_weights_t[indices_fixed_i64]                        # (n, k)

                if self.use_soft_topk:
                    soft_topk_w = _SOFT_TOPK_WEIGHT_KERNEL(
                        shifted_dataset, corpus_for_shift, indices_fixed_i64, self.soft_topk_temperature
                    )
                    w = w * soft_topk_w

                if masked:
                    valid = corpus_labels[indices_fixed_i64] == labels.unsqueeze(1)   # (n, k) bool
                    w = w * valid
                    has_any_valid = valid.any(dim=1)                          # (n,)
                    orphan_mask = ~has_any_valid
                else:
                    orphan_mask = None

                denom = w.sum(dim=1, keepdim=True).clamp_min(1e-6)           # (n, 1)
                w_norm = w / denom                                           # fold the divide in once, not per iteration

                if self.use_sparse_shift:
                    w_or_W = _build_sparse_weight_matrix(
                        indices_fixed_i64, w_norm, n_samples, corpus_size, self.device_name
                    )
                else:
                    w_or_W = w_norm

            revised_d, change = self._shift_kernel(
                shifted_dataset,
                indices_fixed,
                w_or_W,
                self.learning_rate,
                self.alpha,
                gate,
                corpus_for_shift,
            )

            if orphan_mask is not None and orphan_mask.any():
                revised_d = torch.where(orphan_mask.unsqueeze(1), shifted_dataset, revised_d)
                change = torch.where(orphan_mask, torch.zeros_like(change), change)

            total_distance = total_distance + change
            shifted_dataset = revised_d

            if self.keep_trajectory:
                trajectory.append(shifted_dataset.clone())

            mean_change = change.mean().item()
            logger.debug(f"Iter {iter_count + 1}: mean change = {mean_change:.6f}")

            if mean_change < self.shift_threshold:
                logger.info(f"Converged at iteration {iter_count + 1}.")
                break

        return shifted_dataset, total_distance, trajectory


mean_shift_density_enhancement = MeanShiftDensityEnhancement