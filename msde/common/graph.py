import torch
import torch.nn.functional as F

from .knn import masked_soft_knn_weights, torch_knn
from .utils import DEFAULT_DEVICE, DIFF_EPS, DIFF_LOG_SCALE_EPS


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


def build_sparse_similarity(X, n_neighbors, n_epochs, device, soft_prune=False,
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


def sparse_similarity_layout(S):
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



def soft_fuzzy_graph(
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
    d, membership, valid = masked_soft_knn_weights(
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
    init_sigma = (offset * valid_f).sum(dim=1) / (valid_count + DIFF_EPS)
    log_sigma = torch.log(init_sigma + DIFF_LOG_SCALE_EPS)

    max_log_sigma_step = 2.0
    for _ in range(sigma_iterations):
        sigma = torch.exp(log_sigma)
        q = torch.exp(-offset / sigma) * valid_f
        residual = q.sum(dim=1) - sigma_target
        derivative = (q * offset / sigma).sum(dim=1)
        raw_step = residual / (derivative + DIFF_EPS)
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
    prune_temperature = torch.sqrt(prune_temperature.square() + DIFF_EPS**2)
    smooth_max = prune_temperature * torch.logsumexp(
        w_sym.reshape(-1) / prune_temperature, dim=0
    )
    threshold = smooth_max / max(n_epochs, 1)
    gate = torch.sigmoid((w_sym - threshold) / prune_temperature)
    w_soft = w_sym * gate

    return w_soft
