"""Mean-shift density enhancement (MSDE) on the unit sphere, in a single file.

Every point of an embedding set is repeatedly moved towards a weighted mean of its neighbours until the
movement converges. The pipeline, in the order ``MeanShiftDensityEnhancement.forward`` runs it:

1. **Normalise** the embeddings to unit length. They are re-normalised right after every shift, so every
   state the loop passes through (and the output) lies on the sphere.
2. **Whitening** (optional, off by default): centre the points and rescale them by the inverse square root of
   their covariance, so distances become Mahalanobis distances and correlated, unevenly scaled directions no
   longer dominate. It can be switched on separately for the density and for the neighbour selection
   (``whiten_density``, ``whiten_selection``); the step itself always works on the unit sphere.
3. **Density**: an empirical density per point, computed once on the initial points (whitened if
   ``whiten_density``). A fuzzy k-nearest-neighbour graph is built; points are then compared as rows of that
   sparse graph (sparse matmul), and a point's density is the average number of other points within a
   radius, over a few shrinking radii.
4. **Select neighbours**: every other point is scored by ``NeighborScore`` (distance x density x stability;
   distance measured in the whitened space if ``whiten_selection``, else on the sphere) and the ``k``
   best-scoring ones are kept. A dense, stable point can outrank a closer, sparse one.
5. **Step**: each point moves a fraction ``learning_rate`` towards the weighted mean of its selected neighbours
   (weights from ``NeighborWeight``), computed with one sparse matmul ``W @ X``; it is then put back on the sphere.
6. **Repeat** (optionally re-selecting neighbours every ``recompute_neighbors`` iterations) until the mean
   per-point movement on the sphere drops below ``shift_threshold`` or ``max_iters_shift`` is reached.

Stability scores (values in [0, 1], higher = more stable) start at 1.0 for every point. Pass a ``stability_fn`` to
``forward`` and they are updated after every iteration from that iteration's step vectors; the updated scores feed
the next neighbour weights and, whenever neighbours are reselected, the selection score. The selection score, the
neighbour weights and the stability update can each be replaced by a callable (which carries its own parameters)
passed to ``forward``; all three receive the index of the current iteration and the iteration cap, so they can
change over the run.

Only torch is required.
"""

import logging

import torch

logger = logging.getLogger(__name__)

DEFAULT_DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

LABEL_MODES = ("same_label", "zero_label")


# ---------------------------------------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------------------------------------
def _raise_dynamo_recompile_limit(limit=128):
    """Compiled kernels built from one factory share a code object, and dynamo counts recompiles per code
    object (default limit 8). Using several values of ``k`` or several dtypes would otherwise hit the limit,
    which is a hard error under ``fullgraph=True``."""
    try:
        import torch._dynamo.config as dynamo_config
        for name in ("recompile_limit", "cache_size_limit"):
            if hasattr(dynamo_config, name):
                setattr(dynamo_config, name, max(getattr(dynamo_config, name), limit))
    except Exception:       # a torch build without dynamo: nothing to raise
        pass


_raise_dynamo_recompile_limit()

_SPARSE_MM_SUPPORT_CACHE = {}


def sparse_mm_supported(device):
    """Whether sparse-times-dense matrix multiplication works on ``device`` (probed once per device)."""
    device = str(device)
    if device not in _SPARSE_MM_SUPPORT_CACHE:
        try:
            idx = torch.zeros((2, 1), dtype=torch.long, device=device)
            t = torch.sparse_coo_tensor(idx, torch.ones(1, device=device), (1, 1), device=device).coalesce()
            torch.sparse.mm(t, torch.ones(1, 1, device=device))
            _SPARSE_MM_SUPPORT_CACHE[device] = True
        except Exception:
            _SPARSE_MM_SUPPORT_CACHE[device] = False
    return _SPARSE_MM_SUPPORT_CACHE[device]


def unit(X, eps=1e-12):
    """Rows scaled to unit length (a zero row stays zero)."""
    return X / X.norm(dim=1, keepdim=True).clamp_min(eps)


@torch.no_grad()
def whiten_points(X, regularization=1e-6, return_transform=False):
    """Centre ``X`` and rescale it by the inverse square root of its covariance.

    Euclidean distances between the returned rows are Mahalanobis distances of ``X``. The covariance is
    accumulated and decomposed in float64 on the CPU: forming it in float32 squares the condition number
    and loses the small eigenvalues of typical embeddings. ``regularization`` is a ridge added to the
    eigenvalues, relative to their mean, which keeps near-degenerate directions (fewer points than features,
    constant features) from being blown up. The result is cast back to ``X``'s dtype before it is moved to
    its device (some devices have no float64).

    With ``return_transform`` the result is ``(whitened, mean (d,), W (d, d))``, ``whitened == (X - mean) @ W``
    up to rounding, so that other points can be mapped into the same space.
    """
    X64 = X.cpu().double()
    Xc = X64 - X64.mean(0, keepdim=True)
    evals, evecs = torch.linalg.eigh(Xc.T @ Xc / max(X.shape[0] - 1, 1))
    evals = evals.clamp_min(0.0)
    scale = (evals + regularization * evals.mean()).clamp_min(torch.finfo(torch.float64).tiny).rsqrt()
    whitened = (Xc @ (evecs * scale)).to(X.dtype).to(X.device)
    if not return_transform:
        return whitened
    return whitened, X64.mean(0).to(X.dtype).to(X.device), (evecs * scale).to(X.dtype).to(X.device)


# ---------------------------------------------------------------------------------------------------------
# Exact k-nearest neighbours
# ---------------------------------------------------------------------------------------------------------
_KNN_KERNELS = {}


def _knn_kernel(k):
    if k not in _KNN_KERNELS:
        def _kernel(X_chunk, X_full):
            return torch.cdist(X_chunk, X_full).topk(k, largest=False, dim=1)     # (chunk, n): all-pairs
        _KNN_KERNELS[k] = torch.compile(_kernel, fullgraph=True)
    return _KNN_KERNELS[k]


@torch.no_grad()
def knn(X, k, chunk_size=8192):
    """Exact Euclidean k-nearest neighbours of every row among all rows (itself included at distance 0).

    Returns ``(indices (n, k) int64, distances (n, k))``. A point's distance to itself is set to exactly 0:
    the matmul-based ``cdist`` leaves platform-dependent rounding noise there that would otherwise be
    mistaken for the nearest-neighbour distance.
    """
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= X.shape[0]:
        raise ValueError(f"k={k!r} is invalid for {X.shape[0]} points; expected an integer 1 <= k <= {X.shape[0]}")
    X = X.detach()
    n = X.shape[0]
    kernel = _knn_kernel(k)
    all_idx = torch.empty((n, k), dtype=torch.long, device=X.device)
    all_dist = torch.empty((n, k), dtype=X.dtype, device=X.device)
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        dists, idx = kernel(X[start:end], X)
        rows = torch.arange(start, end, device=X.device).unsqueeze(1)
        all_idx[start:end] = idx
        all_dist[start:end] = dists.masked_fill(idx == rows, 0.0)
    return all_idx, all_dist


@torch.no_grad()
def median_neighbor_distance(X, k):
    """Median distance from a point to its ``k`` nearest other points: a data-driven kernel width."""
    n = X.shape[0]
    if n < 2:
        return 1.0
    _, dist = knn(X, min(k, n - 1) + 1)                                  # column 0 is the point itself
    return max(dist[:, 1:].median().item(), 1e-12)


# ---------------------------------------------------------------------------------------------------------
# Density: fuzzy neighbour graph -> radius counts in graph space (sparse matmul)
# ---------------------------------------------------------------------------------------------------------
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


def _fuzzy_graph(knn_indices, knn_dists, n, k, n_epochs):
    """Sparse fuzzy graph (COO rows, cols, values) from k-NN results: per-point scale ``sigma`` by bisection,
    fuzzy-union symmetrisation, and removal of edges weaker than ``max / n_epochs``."""
    device = knn_dists.device
    target = float(torch.log2(torch.tensor(k, dtype=torch.float32)))

    mask = knn_dists > 0
    rhos = torch.where(mask, knn_dists, torch.tensor(float("inf"), device=device))
    rhos = torch.clamp(rhos.min(dim=1).values, min=1e-8)

    lo = torch.full((n,), 1e-20, device=device)
    hi = torch.full((n,), 1e3, device=device)
    sigma = torch.ones(n, device=device)
    shifted_tail = torch.clamp(knn_dists - rhos[:, None], min=0.0)[:, 1:]

    for _ in range(64):
        vals_sum = torch.exp(-shifted_tail / sigma[:, None]).sum(dim=1)
        converged = (vals_sum - target).abs() < 1e-5
        too_high = (vals_sum > target) & ~converged
        too_low = (vals_sum < target) & ~converged

        hi = torch.where(too_high, sigma, hi)
        lo = torch.where(too_low, sigma, lo)
        sigma = torch.where(too_high, (lo + sigma) / 2.0, sigma)
        sigma = torch.where(too_low, torch.where(hi >= 1e3, sigma * 2.0, (sigma + hi) / 2.0), sigma)

        if bool(converged.all()):
            break

    sigma = torch.clamp(sigma, min=1e-10)
    rows, cols, w_sym = _SYMMETRIZE_KERNEL(knn_dists, knn_indices, rhos, sigma, n, k)
    active = torch.nonzero(w_sym >= w_sym.max() / max(n_epochs, 1), as_tuple=True)[0]
    return rows[active], cols[active], w_sym[active]


def _similarity_graph(X, n_neighbors, n_epochs):
    """The fuzzy neighbour graph of ``X`` as a coalesced sparse (n, n) matrix."""
    n = X.shape[0]
    knn_idx, knn_dist = knn(X, n_neighbors)
    rows, cols, vals = _fuzzy_graph(knn_idx, knn_dist, n, n_neighbors, n_epochs)
    return torch.sparse_coo_tensor(
        torch.stack([rows, cols]), vals, size=(n, n), device=X.device, check_invariants=False,
    ).coalesce()


def _graph_layout(S):
    """Row offsets and squared row norms of the sparse graph, for streamed passes."""
    idx = S.indices()
    rows_sorted, cols_sorted, vals_sorted = idx[0], idx[1], S.values()
    n = S.shape[0]
    row_ptr = torch.searchsorted(rows_sorted, torch.arange(0, n + 1, device=S.device)).tolist()
    row_norm_sq = torch.zeros(n, dtype=torch.float32, device=S.device)
    row_norm_sq.scatter_add_(0, rows_sorted, vals_sorted ** 2)
    return rows_sorted, cols_sorted, vals_sorted, row_ptr, row_norm_sq


def _make_pairwise_dist_kernel():
    def _kernel(row_norm_sq, idx, cross):
        sq_dist = row_norm_sq[:, None] + row_norm_sq[idx][None, :] - 2.0 * cross
        return torch.sqrt(torch.clamp(sq_dist, min=0.0))
    return torch.compile(_kernel, fullgraph=True)


_PAIRWISE_DIST_KERNEL = _make_pairwise_dist_kernel()


def _graph_chunk_dist(S, layout, start, end):
    """Graph-space distances from the rows ``start:end`` to all rows. The cross terms are one sparse-times-dense
    matmul, ``S @ chunk.T``, which is where the density computation spends its time."""
    rows_sorted, cols_sorted, vals_sorted, row_ptr, row_norm_sq = layout
    n = S.shape[0]
    lo, hi = row_ptr[start], row_ptr[end]
    chunk_dense = torch.zeros((end - start, n), dtype=row_norm_sq.dtype, device=S.device)
    chunk_dense[rows_sorted[lo:hi] - start, cols_sorted[lo:hi]] = vals_sorted[lo:hi]
    cross = torch.sparse.mm(S, chunk_dense.T)
    idx = torch.arange(start, end, device=S.device)
    return _PAIRWISE_DIST_KERNEL(row_norm_sq, idx, cross), idx


def _radius_counts(S, layout, radius, chunk_size):
    """For every row, the number of other rows within ``radius`` in graph space."""
    n = S.shape[0]
    counts = torch.zeros(n, dtype=torch.float32, device=S.device)
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        dist, _ = _graph_chunk_dist(S, layout, start, end)
        counts[start:end] = (dist < radius).sum(dim=0).float() - 1.0
    return counts


def _graph_distance_range(S, layout, chunk_size):
    """Global (max, min) distance between two different rows in graph space."""
    n = S.shape[0]
    running_max = torch.tensor(float("-inf"), device=S.device)
    running_min = torch.tensor(float("inf"), device=S.device)
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        dist, idx = _graph_chunk_dist(S, layout, start, end)
        self_mask = torch.zeros_like(dist, dtype=torch.bool)
        self_mask[idx, torch.arange(idx.shape[0], device=S.device)] = True
        running_max = torch.maximum(running_max, dist.masked_fill(self_mask, float("-inf")).max())
        running_min = torch.minimum(running_min, dist.masked_fill(self_mask, float("inf")).min())
    return running_max.item(), running_min.item()


def _search_radius(S, layout, low, high, threshold, required, chunk_size, tol=1e-4, max_iter=50):
    """Smallest radius at which at least ``required`` rows have more than ``threshold`` others inside it."""
    lo, hi, result, found = low, high, high, False
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        counts = _radius_counts(S, layout, torch.tensor(mid, device=S.device), chunk_size)
        if int((counts > threshold).sum().item()) >= required:
            result, found, hi = mid, True, mid
        else:
            lo = mid
        if abs(hi - lo) < tol:
            break
    return result, found


@torch.no_grad()
def density_weights(X, nbd_sample_count_threshold=30, n_neighbors=15, n_epochs=200, max_iters_weight_count=4,
                    satisfiability_proportion=0.3, chunk_size=2048, whiten=False, whiten_regularization=1e-6):
    """Empirical density of every row of ``X``, returned as an (n,) tensor (larger = denser).

    With ``whiten`` the points are whitened first (see ``whiten_points``; ``whiten_regularization`` is its ridge),
    so the graph and the density are computed in Mahalanobis geometry instead of Euclidean. A fuzzy ``n_neighbors``-nearest-neighbour graph of ``X`` is built and its rows are compared as sparse
    vectors (sparse matmul, ``chunk_size`` rows at a time). The radius is the smallest one at which at least
    ``satisfiability_proportion`` of the points have more than ``nbd_sample_count_threshold`` others inside
    it (found by bisection, relaxed once if that fails). The density of a point is its number of others
    within the radius, averaged over ``max_iters_weight_count`` radii shrinking linearly to zero.
    """
    if whiten:
        X = whiten_points(X, whiten_regularization)
    n = X.shape[0]
    S = _similarity_graph(X, n_neighbors, n_epochs)
    layout = _graph_layout(S)

    max_dist, min_dist = _graph_distance_range(S, layout, chunk_size)
    threshold = max(1, n - 1) if nbd_sample_count_threshold >= n else nbd_sample_count_threshold
    required = int(satisfiability_proportion * n)
    radius, found = _search_radius(S, layout, min_dist, max_dist, threshold, required, chunk_size)
    if not found:
        radius, found = _search_radius(S, layout, min_dist, max_dist, max(1, threshold // 2), required // 2, chunk_size)
    if not found:
        radius = max_dist
    radius = torch.tensor(radius, dtype=torch.float32, device=S.device)

    delta = (radius - 1e-6) / max_iters_weight_count
    total = torch.zeros(n, dtype=torch.float32, device=S.device)
    for _ in range(max_iters_weight_count):
        total = total + _radius_counts(S, layout, radius, chunk_size)
        radius = radius - delta
    return total / max_iters_weight_count


# ---------------------------------------------------------------------------------------------------------
# Neighbour selection and neighbour weights (replaceable callables)
# ---------------------------------------------------------------------------------------------------------
def _check_strength(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not value >= 0 or value == float("inf"):
        raise ValueError(f"{name} must be a finite number >= 0; got {value!r}")
    return float(value)


class NeighborScore:
    """How attractive every candidate is to a point: ``density^a * stability^b * exp(-c * d^2 / (2 h^2))``.

    The default selection score. ``a``, ``b`` and ``c`` are the density, stability and distance strengths
    (>= 0; 0 switches that factor off, larger values weight it more). ``d`` is the distance between the point and
    the candidate in the selection space (Euclidean on the sphere, or Mahalanobis if the selection whitens) and
    ``h`` the kernel width.

    A replacement must have the same call signature and return a score of the same shape as ``dist``:
    ``score(dist, density, stability, h, iteration, n_iters)`` with ``dist`` a (chunk, n) block of distances
    from a chunk of points to all points, ``density`` and ``stability`` the (n,) per-candidate vectors, ``h`` the
    kernel width in use, ``iteration`` the 0-based index of the current iteration and ``n_iters`` the iteration
    cap (``max_iters_shift``; the run may stop earlier by convergence). It is called whenever neighbours are
    selected. Larger means more attractive; scores <= 0 mean "no attraction" and are only used to fill slots
    once the positive scores run out (the nearest remaining candidates are taken). Scores must be finite.
    """

    def __init__(self, distance_strength=1.0, density_strength=1.0, stability_strength=1.0):
        self.distance_strength = _check_strength("distance_strength", distance_strength)
        self.density_strength = _check_strength("density_strength", density_strength)
        self.stability_strength = _check_strength("stability_strength", stability_strength)

    def __call__(self, dist, density, stability, h, iteration, n_iters):
        kernel = torch.exp(-self.distance_strength * dist.square() / (2.0 * h * h))
        return density.pow(self.density_strength) * stability.pow(self.stability_strength) * kernel


class NeighborWeight:
    """How much each selected neighbour pulls its point in the weighted mean: ``density^a * stability^b``.

    The default neighbour weight (distance is ignored, so every selected neighbour counts according to how
    dense and stable it is). ``a`` and ``b`` are the density and stability strengths (>= 0).

    A replacement must have the same call signature, ``weight(dist, density, stability, h, iteration, n_iters)``,
    with the three tensors of shape (n, k) for the k selected neighbours of every point (``dist`` the current
    distances to them in the selection space, ``stability`` their current scores), ``h`` the kernel width in
    use, ``iteration`` the 0-based index of the iteration whose step will use the weights and ``n_iters`` the
    iteration cap (``max_iters_shift``). It returns non-negative finite (n, k) weights, normalised per row
    afterwards, so only their ratios matter. A weight function you pass is evaluated at the start of every
    iteration; the default is evaluated only when something it depends on can change.
    """

    def __init__(self, density_strength=1.0, stability_strength=1.0):
        self.density_strength = _check_strength("density_strength", density_strength)
        self.stability_strength = _check_strength("stability_strength", stability_strength)

    def __call__(self, dist, density, stability, h, iteration, n_iters):
        return density.pow(self.density_strength) * stability.pow(self.stability_strength)


@torch.no_grad()
def select_neighbors(X, density, stability, k, h, score_fn, whiten=False, whiten_regularization=1e-6,
                     labels=None, corpus_labels=None, chunk_size=1024, iteration=0, n_iters=1):
    """The ``k`` best-scoring other points for every point of ``X``.

    Distances are Euclidean on ``X``, or, with ``whiten``, Euclidean in the whitened space of the current ``X``
    (Mahalanobis; ``whiten_regularization`` is the ridge of ``whiten_points``). All other points are candidates (no
    distance pre-filter); a point is never its own neighbour. Candidates with a positive score rank by score,
    the remaining slots (if any) go to the nearest candidates. With ``labels`` a point may only pick points
    whose ``corpus_labels`` equal its label.

    ``iteration`` and ``n_iters`` are handed on to ``score_fn``.

    Returns ``(indices (n, k), dist (n, k), valid (n, k) bool or None, orphan (n,) bool or None, h, transform)``
    where ``valid`` marks picks that respect the label mask, ``orphan`` marks points with no valid pick at all,
    ``dist`` the distances to the picks in the selection space, ``h`` the kernel width used (``"auto"`` resolves
    to the median distance to the ``k`` nearest neighbours in that space) and ``transform`` is ``None`` or, with
    ``whiten``, the ``(mean, W)`` that maps points into the selection space (see ``neighbor_distances``).
    """
    n = X.shape[0]
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= n - 1:
        raise ValueError(f"k={k!r} is invalid for {n} points; expected an integer 1 <= k <= {n - 1}")
    transform = None
    if whiten:
        Z, mean, W = whiten_points(X, whiten_regularization, return_transform=True)
        transform = (mean, W)
    else:
        Z = X
    if h == "auto":
        h = median_neighbor_distance(Z, k)
    if labels is not None and corpus_labels is None:
        corpus_labels = labels

    indices = torch.empty((n, k), dtype=torch.long, device=X.device)
    dist = torch.empty((n, k), dtype=X.dtype, device=X.device)
    valid = torch.empty((n, k), dtype=torch.bool, device=X.device)
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        d = torch.cdist(Z[start:end], Z)                                     # (c, n) distances in the selection space
        score = score_fn(d, density, stability, h, iteration, n_iters)
        if score.shape != d.shape or not bool(torch.isfinite(score).all()):
            raise ValueError("the selection score must be finite and have the shape of the distance block "
                             f"{tuple(d.shape)}; got {tuple(score.shape)}")
        key = torch.where(score > 0, score, -d)                              # positives first, then nearest

        excluded = torch.zeros_like(d, dtype=torch.bool)
        rows = torch.arange(end - start, device=X.device)
        excluded[rows, rows + start] = True                                  # never its own neighbour
        if labels is not None:
            excluded |= labels[start:end].unsqueeze(1) != corpus_labels.unsqueeze(0)
        top, idx = key.masked_fill(excluded, float("-inf")).topk(k, dim=1)

        indices[start:end] = idx
        dist[start:end] = d.gather(1, idx)
        valid[start:end] = torch.isfinite(top)

    if labels is None:
        return indices, dist, None, None, h, transform
    return indices, dist, valid, ~valid.any(dim=1), h, transform


@torch.no_grad()
def neighbor_distances(X, indices, transform=None, chunk_size=256):
    """Current distances (n, k) from every point of ``X`` to its selected neighbours ``indices``, in the selection
    space: Euclidean on ``X``, or, if the selection whitened, Euclidean after mapping the points with the
    ``(mean, W)`` transform of that selection."""
    Z = X if transform is None else (X - transform[0]) @ transform[1]
    out = torch.empty(indices.shape, dtype=X.dtype, device=X.device)
    for start in range(0, X.shape[0], chunk_size):
        end = min(X.shape[0], start + chunk_size)
        out[start:end] = (Z[start:end].unsqueeze(1) - Z[indices[start:end]]).norm(dim=-1)
    return out


# ---------------------------------------------------------------------------------------------------------
# Weighted-mean step
# ---------------------------------------------------------------------------------------------------------
@torch.no_grad()
def prepare_step(indices, dist, density, stability, h, weight_fn, valid, n, iteration=0, n_iters=1):
    """Normalised (n, k) neighbour weights and, where the device supports it, the sparse (n, n) weight matrix.

    ``weight_fn`` is evaluated here with the (n, k) tensors of the selected neighbours. The label-validity mask
    is applied and every row is normalised to sum to 1 (a row with no weight at all stays zero, which leaves its
    point in place once the step is renormalised onto the sphere).
    """
    w = weight_fn(dist, density[indices], stability[indices], h, iteration, n_iters)
    if w.shape != indices.shape or not bool(torch.isfinite(w).all()) or bool((w < 0).any()):
        raise ValueError("the neighbour weights must be finite, non-negative and have shape "
                         f"{tuple(indices.shape)}; got {tuple(w.shape)}")
    if valid is not None:
        w = w * valid
    w = w / w.sum(dim=1, keepdim=True).clamp_min(torch.finfo(w.dtype).tiny)

    W = None
    if sparse_mm_supported(w.device):
        k = indices.shape[1]
        rows = torch.arange(n, device=w.device).repeat_interleave(k)
        W = torch.sparse_coo_tensor(
            torch.stack([rows, indices.reshape(-1)]), w.reshape(-1), size=(n, n), device=w.device,
            check_invariants=False,
        ).coalesce()
    return indices, w, W


@torch.no_grad()
def weighted_mean(X, indices, w, W, chunk_size=512):
    """Row-wise weighted mean of the selected neighbours: one sparse matmul ``W @ X`` (or, on a device without
    sparse matmul, a gather over row chunks)."""
    if W is not None:
        return torch.sparse.mm(W, X)
    out = torch.empty_like(X)
    for start in range(0, X.shape[0], chunk_size):
        end = min(X.shape[0], start + chunk_size)
        out[start:end] = (w[start:end].unsqueeze(-1) * X[indices[start:end]]).sum(dim=1)
    return out


# ---------------------------------------------------------------------------------------------------------
# Stability update
# ---------------------------------------------------------------------------------------------------------
@torch.no_grad()
def update_stability(stability_fn, step, density, stability, h, iteration=0, n_iters=1):
    """New (n,) stability scores from the latest step vectors.

    ``stability_fn(step, density, stability, h, iteration, n_iters)`` gets the (n, d) displacement each point made
    in the iteration that just finished, the (n,) density, the (n,) scores before this update, the kernel width
    ``h``, the 0-based index of that iteration and the iteration cap, and must return finite (n,) scores within
    [0, 1].
    """
    new = stability_fn(step, density, stability, h, iteration, n_iters)
    if not isinstance(new, torch.Tensor) or new.shape != stability.shape:
        raise ValueError(f"the stability scores must be a tensor of shape {tuple(stability.shape)}; "
                         f"got {tuple(getattr(new, 'shape', ()))}")
    if not (bool(torch.isfinite(new).all()) and bool((new >= 0).all()) and bool((new <= 1).all())):
        raise ValueError("stability scores must be finite and within [0, 1]")
    return new.to(device=stability.device, dtype=stability.dtype)


# ---------------------------------------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------------------------------------
class MeanShiftDensityEnhancement(torch.nn.Module):
    """Mean-shift density enhancement of unit-normalised embeddings (see the module docstring).

    Parameters
    ----------
    k : number of neighbours each point is pulled towards.
    nbd_sample_count_threshold, n_neighbors, n_epochs, max_iters_weight_count, satisfiability_proportion,
    weight_chunk_size : the density computation (see ``density_weights``).
    learning_rate : fraction of the way a point moves towards its weighted mean each iteration.
    density_strength, stability_strength, distance_strength : the strengths of the default selection score
        and neighbour weight (see ``NeighborScore`` and ``NeighborWeight``); >= 0, 0 switches a factor off.
    h : kernel width of the distance term, a positive number in the units of the selection space (whitened
        units if ``whiten_selection``), or "auto" (median distance to the ``k`` nearest neighbours,
        re-estimated every time neighbours are selected).
    whiten_density, whiten_selection : whether the density computation / the neighbour selection measure
        distances in whitened (Mahalanobis) instead of Euclidean geometry. Both default to False.
    whiten_regularization : the ridge of the whitening (see ``whiten_points``); only used when one of the two
        flags is set.
    recompute_neighbors : 0 or None selects neighbours once, from the initial points; a positive N reselects
        every N iterations (iterations 0, N, 2N, ...), re-estimating the whitening (if on) on the current
        points and re-evaluating the neighbour weights.
    max_iters_shift, shift_threshold : the loop stops when the mean per-point movement on the sphere is below
        ``shift_threshold`` (convergence) or after ``max_iters_shift`` iterations (a safety cap).
    keep_trajectory : also return the unit-norm state after every iteration.
    label_mode : how ``labels`` given to ``forward`` restrict neighbours: "same_label" (a point's neighbours
        must share its label) or "zero_label" (neighbours must have label 0, whatever the point's own label).
    chunk_size : rows of the n x n distance blocks held at once during neighbour selection.
    device : where the computation runs.
    """

    def __init__(
        self,
        k=50,
        nbd_sample_count_threshold=30,
        n_neighbors=15,
        n_epochs=200,
        max_iters_weight_count=4,
        satisfiability_proportion=0.3,
        weight_chunk_size=2048,
        learning_rate=0.3,
        density_strength=1.0,
        stability_strength=1.0,
        distance_strength=1.0,
        h="auto",
        whiten_density=False,
        whiten_selection=False,
        whiten_regularization=1e-6,
        recompute_neighbors=0,
        max_iters_shift=1000,
        shift_threshold=1e-4,
        keep_trajectory=False,
        label_mode="same_label",
        chunk_size=1024,
        device=DEFAULT_DEVICE,
    ):
        super().__init__()
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError(f"k must be a positive integer; got {k!r}")
        if isinstance(h, str):
            if h != "auto":
                raise ValueError(f"h must be a positive number or 'auto'; got {h!r}")
        elif isinstance(h, bool) or not h > 0:
            raise ValueError(f"h must be a positive number or 'auto'; got {h!r}")
        if recompute_neighbors is not None and (
            not isinstance(recompute_neighbors, int) or isinstance(recompute_neighbors, bool) or recompute_neighbors < 0
        ):
            raise ValueError("recompute_neighbors must be None, 0 (select once) or a positive integer N "
                             "(reselect every N iterations)")
        if not learning_rate > 0:
            raise ValueError("learning_rate must be > 0")
        if not isinstance(max_iters_shift, int) or max_iters_shift < 1:
            raise ValueError("max_iters_shift must be a positive integer")
        if not shift_threshold >= 0:
            raise ValueError("shift_threshold must be >= 0")
        if not whiten_regularization >= 0:
            raise ValueError("whiten_regularization must be >= 0")
        if not isinstance(whiten_density, bool) or not isinstance(whiten_selection, bool):
            raise ValueError("whiten_density and whiten_selection must be booleans")
        if label_mode not in LABEL_MODES:
            raise ValueError(f"label_mode must be one of {LABEL_MODES}; got {label_mode!r}")
        if chunk_size < 1 or weight_chunk_size < 1:
            raise ValueError("chunk_size and weight_chunk_size must be positive")

        self.k = k
        self.nbd_sample_count_threshold = nbd_sample_count_threshold
        self.n_neighbors = n_neighbors
        self.n_epochs = n_epochs
        self.max_iters_weight_count = max_iters_weight_count
        self.satisfiability_proportion = satisfiability_proportion
        self.weight_chunk_size = weight_chunk_size
        self.learning_rate = learning_rate
        self.h = h
        self.whiten_density = whiten_density
        self.whiten_selection = whiten_selection
        self.whiten_regularization = whiten_regularization
        self.recompute_neighbors = recompute_neighbors or 0
        self.max_iters_shift = max_iters_shift
        self.shift_threshold = shift_threshold
        self.keep_trajectory = keep_trajectory
        self.label_mode = label_mode
        self.chunk_size = chunk_size
        self.device_name = device
        self.default_score = NeighborScore(distance_strength, density_strength, stability_strength)
        self.default_weight = NeighborWeight(density_strength, stability_strength)

    def _labels(self, labels, n):
        """``(query_labels, neighbor_labels)`` for ``labels`` given to ``forward`` (or ``(None, None)``)."""
        if labels is None:
            return None, None
        labels = torch.as_tensor(labels, device=self.device_name).detach().long()
        if labels.dim() != 1 or labels.shape[0] != n:
            raise ValueError(f"labels must be 1D with length n={n}; got shape {tuple(labels.shape)}")
        return (labels if self.label_mode == "same_label" else torch.zeros_like(labels)), labels

    @torch.no_grad()
    def forward(self, X, gate=1.0, labels=None, stability_fn=None, score_fn=None, weight_fn=None):
        """Run MSDE on the rows of ``X`` and return ``(shifted, total_distance, trajectory)``.

        shifted : (n, d) unit-norm points after the last iteration.
        total_distance : (n,) the distance each point moved, summed over iterations (chords on the sphere).
        trajectory : list of unit-norm (n, d) states (initial state, then after each iteration) if
            ``keep_trajectory`` is set, otherwise empty.

        gate : multiplier on the step size for this call.
        labels : optional (n,) integer labels restricting neighbour selection (see ``label_mode``). A point
            with no valid neighbour is left where it is.
        stability_fn : optional callable that updates the stability scores (higher = more stable, in [0, 1]) after
            every iteration; it carries its own parameters. Called as
            ``stability_fn(step, density, stability, h, iteration, n_iters)`` with ``step`` the (n, d) displacement
            every point made in the iteration that just finished (new unit position minus old; zero for a point
            left in place), ``density`` the (n,) density, ``stability`` the (n,) scores before this update (all 1.0
            at the start), ``h`` the kernel width in use, ``iteration`` the 0-based index of that iteration and
            ``n_iters`` the iteration cap (``max_iters_shift``); it returns the new (n,) scores, finite and within
            [0, 1]. The first selection and the first step always use scores of 1.0 (there is no step yet). The
            updated scores are used by the next iteration's neighbour weights and by every later neighbour
            selection. Without a ``stability_fn`` the scores stay at 1.0. The final scores are available as
            ``stability_`` afterwards.
        score_fn : optional replacement for the selection score (see ``NeighborScore``; it carries its own
            parameters and receives ``iteration`` and ``n_iters``). It is used for every neighbour selection,
            including the initial one: with ``recompute_neighbors=0`` it defines the single neighbour graph the
            run uses, otherwise it is called again at every reselection.
        weight_fn : optional replacement for the neighbour weights of the weighted mean (see ``NeighborWeight``;
            it carries its own parameters and receives ``iteration`` and ``n_iters``). A function you pass is
            evaluated at the start of **every** iteration, with the current distances to the selected neighbours
            (in the selection space) and the current stability scores, so the weights follow the moving points
            and the stability updates. The default is used when not given; it only depends on the density and
            the stability, so it is re-evaluated only when a ``stability_fn`` is given.
        """
        X = torch.as_tensor(X)
        if X.dim() != 2 or not X.is_floating_point():
            raise ValueError(f"X must be a 2D floating-point tensor; got shape {tuple(X.shape)}, dtype {X.dtype}")
        X = X.detach().to(self.device_name)
        n = X.shape[0]
        if not 1 <= self.k <= n - 1 or not 1 <= self.n_neighbors <= n:
            raise ValueError(f"k={self.k} and n_neighbors={self.n_neighbors} need k <= n-1 and n_neighbors <= n "
                             f"for n={n} points")

        stability = torch.ones(n, dtype=X.dtype, device=X.device)           # every point starts fully stable
        query_labels, neighbor_labels = self._labels(labels, n)
        score_fn = score_fn if score_fn is not None else self.default_score
        weights_every_iteration = weight_fn is not None or stability_fn is not None
        weight_fn = weight_fn if weight_fn is not None else self.default_weight
        n_iters = self.max_iters_shift

        x = unit(X)                                                          # on the sphere from the start
        density = density_weights(
            x, self.nbd_sample_count_threshold, self.n_neighbors, self.n_epochs, self.max_iters_weight_count,
            self.satisfiability_proportion, self.weight_chunk_size,
            whiten=self.whiten_density, whiten_regularization=self.whiten_regularization,
        ).clamp_min(0.0).to(x.dtype)

        total_distance = torch.zeros(n, dtype=x.dtype, device=x.device)
        trajectory = [x.clone()] if self.keep_trajectory else []
        state, indices, dist, valid, orphan, h, transform = None, None, None, None, None, None, None
        for iteration in range(n_iters):
            selected = iteration == 0 or bool(self.recompute_neighbors and iteration % self.recompute_neighbors == 0)
            if selected:
                indices, dist, valid, orphan, h, transform = select_neighbors(
                    x, density, stability, self.k, self.h, score_fn,
                    whiten=self.whiten_selection, whiten_regularization=self.whiten_regularization,
                    labels=query_labels, corpus_labels=neighbor_labels, chunk_size=self.chunk_size,
                    iteration=iteration, n_iters=n_iters,
                )
            elif weights_every_iteration:
                dist = neighbor_distances(x, indices, transform)             # distances at the current positions
            if selected or weights_every_iteration:
                state = prepare_step(indices, dist, density, stability, h, weight_fn, valid, n, iteration, n_iters)

            mean = weighted_mean(x, *state[:3])
            moved = unit(x + self.learning_rate * gate * (mean - x))         # step, then back onto the sphere
            if orphan is not None:
                moved = torch.where(orphan.unsqueeze(1), x, moved)
            step = moved - x                                                 # this iteration's step vectors
            change = step.norm(dim=1)

            total_distance = total_distance + change
            x = moved
            if self.keep_trajectory:
                trajectory.append(x.clone())
            mean_change = change.mean().item()
            logger.debug("iteration %d: mean movement %.6f", iteration + 1, mean_change)
            if stability_fn is not None:
                stability = update_stability(stability_fn, step, density, stability, h, iteration, n_iters)
            if mean_change < self.shift_threshold:
                logger.info("converged after %d iterations", iteration + 1)
                break

        self.stability_ = stability                                          # the final scores (1.0 without a stability_fn)
        return x, total_distance, trajectory


mean_shift_density_enhancement = MeanShiftDensityEnhancement
