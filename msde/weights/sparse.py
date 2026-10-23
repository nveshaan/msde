import torch

from ..common.eps import calculate_eps_from_similarity, radius_counts_chunked
from ..common.graph import build_sparse_similarity, sparse_similarity_layout
from ..common.utils import DEFAULT_DEVICE
from .base import DensityWeights


def compute_weights_from_similarity_chunked(S, n, nbd_sample_count_threshold,
                                            satisfiability_proportion, max_iters_weight_count,
                                            chunk_size, temperature=None, eps=None, layout=None,
                                            differentiable_eps=False, eps_count_temperature=None,
                                            eps_threshold_temperature=None):
    """Compute empirical weights with streamed graph-distance passes."""
    if layout is None:
        layout = sparse_similarity_layout(S)
    rows_sorted, cols_sorted, vals_sorted, row_ptr, row_norm_sq = layout

    if eps is None:
        if differentiable_eps:
            eps_tensor = calculate_eps_from_similarity(
                S, rows_sorted, cols_sorted, vals_sorted, row_ptr, row_norm_sq,
                n, nbd_sample_count_threshold,
                satisfiability_proportion, chunk_size,
                differentiable=True, count_temperature=eps_count_temperature,
                threshold_temperature=eps_threshold_temperature,
            )
        else:
            with torch.no_grad():
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
    S = build_sparse_similarity(
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


class SparseEmpirical(DensityWeights):
    """Empirical density weights from a hard sparse fuzzy graph.

    Non-differentiable w.r.t. graph topology but differentiable through the
    k-NN distances, and through ``temperature`` / ``eps`` when those are
    learnable (``temperature`` turns hard counts into sigmoid counts).
    ``eps=None`` with ``learn_eps=True`` initialises a learnable eps from the
    data on first use. The weights of a reference manifold are cached.
    """

    cache_reference = True

    def __init__(
        self,
        nbd_sample_count_threshold=30,
        n_neighbors=15,
        n_epochs=200,
        max_iters_weight_count=4,
        satisfiability_proportion=0.3,
        use_chunking=False,
        chunk_size=4096,
        temperature=None,
        eps=None,
        learn_temperature=False,
        learn_eps=False,
        soft_prune=False,
        soft_prune_temperature=None,
        differentiable_eps=False,
        eps_count_temperature=None,
        eps_threshold_temperature=None,
        device=DEFAULT_DEVICE,
    ):
        super().__init__()
        if temperature is not None and float(torch.as_tensor(temperature).detach().cpu()) <= 0:
            raise ValueError("temperature must be > 0")
        if learn_temperature and temperature is None:
            raise ValueError("learn_temperature=True requires an initial temperature")
        if learn_eps and temperature is None:
            raise ValueError("learn_eps=True requires temperature to enable differentiable weights")
        if differentiable_eps and not use_chunking:
            raise ValueError(
                "differentiable_eps=True (implicit-differentiation eps) is only "
                "implemented for the chunked path -- pass use_chunking=True, or "
                "set differentiable_eps=False to use the dense path."
            )
        self.nbd_sample_count_threshold = nbd_sample_count_threshold
        self.n_neighbors = n_neighbors
        self.n_epochs = n_epochs
        self.max_iters_weight_count = max_iters_weight_count
        self.satisfiability_proportion = satisfiability_proportion
        self.use_chunking = use_chunking
        self.chunk_size = chunk_size
        self.learn_eps = learn_eps
        self.soft_prune = soft_prune
        self.soft_prune_temperature = soft_prune_temperature
        self.differentiable_eps = differentiable_eps
        self.eps_count_temperature = eps_count_temperature
        self.eps_threshold_temperature = eps_threshold_temperature
        self.device_name = device

        dtype = next(
            (v.dtype for v in (temperature, eps) if isinstance(v, torch.Tensor) and v.is_floating_point()),
            torch.get_default_dtype(),
        )
        self.register_scalar("temperature", temperature, learn_temperature, device, dtype)
        self.register_scalar("eps", eps, learn_eps and eps is not None, device, dtype)

    def compute(self, corpus):
        if self.learn_eps and self.eps is None:
            similarity = build_sparse_similarity(
                corpus, self.n_neighbors, self.n_epochs, self.device_name,
                soft_prune=self.soft_prune, soft_prune_temperature=self.soft_prune_temperature,
            )
            layout = sparse_similarity_layout(similarity)
            with torch.no_grad():
                eps_initial = calculate_eps_from_similarity(
                    similarity,
                    *layout,
                    corpus.shape[0],
                    self.nbd_sample_count_threshold,
                    self.satisfiability_proportion,
                    self.chunk_size,
                )
            self.replace_scalar_with_parameter("eps", eps_initial.to(dtype=corpus.dtype))

            if self.use_chunking:
                return compute_weights_from_similarity_chunked(
                    similarity,
                    corpus.shape[0],
                    self.nbd_sample_count_threshold,
                    self.satisfiability_proportion,
                    self.max_iters_weight_count,
                    self.chunk_size,
                    self.temperature,
                    self.eps,
                    layout=layout,
                )
            return compute_weights_from_similarity_dense(
                similarity,
                corpus.shape[0],
                self.nbd_sample_count_threshold,
                self.satisfiability_proportion,
                self.max_iters_weight_count,
                self.temperature,
                self.eps,
            )

        return get_empirical_weights(
            corpus,
            nbd_sample_count_threshold=self.nbd_sample_count_threshold,
            max_iters_weight_count=self.max_iters_weight_count,
            satisfiability_proportion=self.satisfiability_proportion,
            n_neighbors=self.n_neighbors,
            n_epochs=self.n_epochs,
            device=self.device_name,
            use_chunking=self.use_chunking,
            chunk_size=self.chunk_size,
            temperature=self.temperature,
            eps=self.eps,
            soft_prune=self.soft_prune,
            soft_prune_temperature=self.soft_prune_temperature,
            differentiable_eps=self.differentiable_eps,
            eps_count_temperature=self.eps_count_temperature,
            eps_threshold_temperature=self.eps_threshold_temperature,
        )
