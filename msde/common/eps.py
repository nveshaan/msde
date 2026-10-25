import torch

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
                          eps_tensor, chunk_size):
    """Count graph-space neighbors within ``eps`` using row chunks."""
    n = S.shape[0]
    counts = torch.zeros(n, dtype=torch.float32, device=S.device)
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        dist, _ = _chunk_pairwise_dist(
            S, rows_sorted, cols_sorted, vals_sorted, row_ptr, row_norm_sq, start, end
        )
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


def calculate_eps_from_similarity(S, rows_sorted, cols_sorted, vals_sorted, row_ptr, row_norm_sq,
                                   n, nbd_sample_count_threshold,
                                   satisfiability_proportion, chunk_size):
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
