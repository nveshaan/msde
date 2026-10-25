import torch

from ..common.eps import calculate_eps_from_similarity, radius_counts_chunked
from ..common.graph import build_sparse_similarity, sparse_similarity_layout
from ..common.utils import DEFAULT_DEVICE
from .base import DensityWeights


def compute_weights_from_similarity_chunked(S, n, nbd_sample_count_threshold,
                                            satisfiability_proportion, max_iters_weight_count,
                                            chunk_size, eps=None, layout=None):
    """Compute empirical weights with streamed graph-distance passes."""
    if layout is None:
        layout = sparse_similarity_layout(S)
    rows_sorted, cols_sorted, vals_sorted, row_ptr, row_norm_sq = layout

    if eps is None:
        eps_tensor = calculate_eps_from_similarity(
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
            S, rows_sorted, cols_sorted, vals_sorted, row_ptr, row_norm_sq, eps_running, chunk_size
        )
        eps_running = eps_running - delta

    return total_counts / max_iters_weight_count


def _make_dense_weight_step():
    def _step(dist, eps_running):
        return (dist < eps_running).sum(dim=1).float() - 1.0
    return torch.compile(_step, fullgraph=True)


_DENSE_WEIGHT_STEP = _make_dense_weight_step()


def compute_weights_from_similarity_dense(S, n, nbd_sample_count_threshold,
                                          satisfiability_proportion, max_iters_weight_count,
                                          eps=None):
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

    for _ in range(max_iters_weight_count):
        total_counts = total_counts + _DENSE_WEIGHT_STEP(dist, eps_running)
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
    eps=None,
):
    """Empirical density weights of the rows of ``X`` from a sparse fuzzy graph."""
    n = X.shape[0]
    S = build_sparse_similarity(X, n_neighbors, n_epochs, device)

    if use_chunking:
        return compute_weights_from_similarity_chunked(
            S, n, nbd_sample_count_threshold, satisfiability_proportion,
            max_iters_weight_count, chunk_size, eps,
        )
    return compute_weights_from_similarity_dense(
        S, n, nbd_sample_count_threshold, satisfiability_proportion,
        max_iters_weight_count, eps,
    )


class SparseEmpirical(DensityWeights):
    """Empirical density weights from a hard sparse fuzzy graph.

    Counts the graph-space neighbours within a radius ``eps`` (fixed if given,
    otherwise found by bisection) averaged over ``max_iters_weight_count``
    shrinking radii.
    """

    def __init__(
        self,
        nbd_sample_count_threshold=30,
        n_neighbors=15,
        n_epochs=200,
        max_iters_weight_count=4,
        satisfiability_proportion=0.3,
        use_chunking=False,
        chunk_size=4096,
        eps=None,
        device=DEFAULT_DEVICE,
    ):
        super().__init__()
        self.nbd_sample_count_threshold = nbd_sample_count_threshold
        self.n_neighbors = n_neighbors
        self.n_epochs = n_epochs
        self.max_iters_weight_count = max_iters_weight_count
        self.satisfiability_proportion = satisfiability_proportion
        self.use_chunking = use_chunking
        self.chunk_size = chunk_size
        self.device_name = device
        self.register_scalar("eps", eps, device)

    @torch.no_grad()
    def compute(self, X):
        return get_empirical_weights(
            X,
            nbd_sample_count_threshold=self.nbd_sample_count_threshold,
            max_iters_weight_count=self.max_iters_weight_count,
            satisfiability_proportion=self.satisfiability_proportion,
            n_neighbors=self.n_neighbors,
            n_epochs=self.n_epochs,
            device=self.device_name,
            use_chunking=self.use_chunking,
            chunk_size=self.chunk_size,
            eps=self.eps,
        )
