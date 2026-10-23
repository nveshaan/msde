import torch

from .knn import smooth_pairwise_dist
from .utils import DIFF_EPS, DIFF_LOG_SCALE_EPS


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


def calculate_eps_from_similarity(S, rows_sorted, cols_sorted, vals_sorted, row_ptr, row_norm_sq,
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



def smooth_epsilon_from_graph(
    graph,
    nbd_sample_count_threshold,
    satisfiability_proportion,
    count_temperature,
    threshold_temperature,
    iterations=25,
):
    """Differentiate through epsilon selection using a smooth fixed-point solve."""
    n = graph.shape[0]
    d_graph = smooth_pairwise_dist(graph, graph)
    offdiag = ~torch.eye(n, dtype=torch.bool, device=graph.device)
    valid = offdiag.to(d_graph.dtype)

    count_temperature = torch.as_tensor(
        count_temperature, dtype=graph.dtype, device=graph.device
    ).square().add(DIFF_EPS**2).sqrt()
    threshold_temperature = torch.as_tensor(
        threshold_temperature, dtype=graph.dtype, device=graph.device
    ).square().add(DIFF_EPS**2).sqrt()

    mean_dist = (d_graph * valid).sum() / (valid.sum() + DIFF_EPS)
    log_eps = torch.log(mean_dist + DIFF_LOG_SCALE_EPS)

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
        raw_step = residual / (dres_dlogeps + DIFF_EPS)
        step = max_log_eps_step * torch.tanh(raw_step / max_log_eps_step)
        log_eps = log_eps - step

    return torch.exp(log_eps), d_graph, valid
