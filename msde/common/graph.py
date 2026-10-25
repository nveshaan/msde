import torch

from .knn import torch_knn
from .utils import DEFAULT_DEVICE


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


def fuzzy_simplicial_set_torch(knn_indices, knn_dists, n, n_neighbors, n_epochs=200,
                                device=DEFAULT_DEVICE):
    """Build a sparse fuzzy graph from k-NN results."""
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

    active = torch.nonzero(w_sym >= threshold, as_tuple=True)[0]
    return rows[active], cols[active], w_sym[active]


def build_sparse_similarity(X, n_neighbors, n_epochs, device):
    """Build the sparse fuzzy similarity graph used for density weights."""
    n = X.shape[0]
    knn_idx, knn_dist = torch_knn(X, n_neighbors, device=device)
    rows, cols, vals = fuzzy_simplicial_set_torch(
        knn_idx, knn_dist, n, n_neighbors, n_epochs, device)
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
