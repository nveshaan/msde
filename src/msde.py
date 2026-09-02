import logging
import torch

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
    """
    Route all logging from this module to `log_file` only.
    If log_file is None, logging is disabled entirely (no console output).
    Safe to call repeatedly (e.g. on every top-level entry point call).
    """
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
    """
    Sparse-op support probe (cached per device string)
    
    torch.sparse_coo_tensor / torch.sparse.mm historically raised
    NotImplementedError on the MPS backend; support has landed on some
    recent builds but isn't guaranteed for every torch/macOS combination.
    Probe once per device, cache the result, and let callers fall back to
    the dense gather-based path instead of hard-crashing.
    """
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


def _make_knn_chunk_kernel(k):
    def _kernel(X_chunk, X_full):
        d = torch.cdist(X_chunk, X_full)          # (chunk, n) - compares to ALL N
        return d.topk(k, largest=False, dim=1)
    return torch.compile(_kernel, fullgraph=True)


def _get_knn_chunk_kernel(k):
    """
    Cached per k (closure constant -- topk's k must be a compile-time
    constant). torch_knn is called with the same k repeatedly (k=self.k for
    the shift graph, k=15 for the fuzzy-similarity graph), so this is a
    compile-once-reuse-many-forward()-calls win. The trailing, possibly
    undersized chunk (when n isn't a multiple of chunk_size) triggers one
    extra recompile for that distinct shape the first time it's seen, then
    is cached same as any other shape.
    """
    if k not in _KNN_CHUNK_KERNEL_CACHE:
        _KNN_CHUNK_KERNEL_CACHE[k] = _make_knn_chunk_kernel(k)
    return _KNN_CHUNK_KERNEL_CACHE[k]


def torch_knn(X, k, device=DEFAULT_DEVICE, chunk_size=8192):
    """
    Exact brute-force k-NN on GPU via chunked cdist + topk.
    Maintains autograd flow for distances if X requires_grad.
    
    MEMORY NOTE: This does NOT "batch" the dataset mathematically. Every point
    finds its true neighbors across the *entire* dataset X (O(N) search space). 
    chunk_size merely streams the outer loop to bound peak VRAM to O(chunk_size * N).
    
    Returns
    -------
    knn_indices : LongTensor (n, k) on device
    knn_dists   : FloatTensor (n, k) on device
    """
    n = X.shape[0]
    all_idx = torch.empty((n, k), dtype=torch.long, device=device)
    
    # Store dists in a list and cat them to preserve autograd graph 
    # (inplace assignments on requires_grad tensors throw errors)
    all_dist_list = []
    chunk_kernel = _get_knn_chunk_kernel(k)

    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        dists, idx = chunk_kernel(X[start:end], X)
        all_idx[start:end] = idx
        all_dist_list.append(dists)

    all_dist = torch.cat(all_dist_list, dim=0)
    return all_idx, all_dist


def compute_fixed_knn(X, k, device=DEFAULT_DEVICE, chunk_size=4096):
    indices, _ = torch_knn(X, k, device=device, chunk_size=chunk_size)
    return indices


def _make_symmetrize_kernel():
    def _kernel(knn_dists, knn_indices, rhos, sigma, n, k):
        dists_shifted = torch.clamp(knn_dists - rhos[:, None], min=0.0)
        weights = torch.exp(-dists_shifted / sigma[:, None])

        rows = torch.arange(n, device=knn_dists.device).repeat_interleave(k)
        cols = knn_indices.reshape(-1)
        vals = weights.reshape(-1)

        # Symmetrize on GPU (Autograd safe through fancy indexing)
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


# Single config (no branching on any Python constant), so one instance
# suffices -- built once at import time.
_SYMMETRIZE_KERNEL = _make_symmetrize_kernel()


def fuzzy_simplicial_set_torch(knn_indices, knn_dists, n, n_neighbors, n_epochs=200,
                                device=DEFAULT_DEVICE):
    """
    GPU port of UMAP's smooth-knn-dist construction.
    Modified to ensure gradients can flow back through knn_dists.
    """
    k = n_neighbors
    target = float(torch.log2(torch.tensor(k, dtype=torch.float32)))

    # Compute binary search targets (rho and sigma) WITHOUT tracking gradients
    # so we don't build a massive, unstable autograd graph inside the loop.
    # NOT compiled: inherently sequential (each step depends on the last)
    # and syncs via bool(...) every iteration -- compile can't remove that.
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

    # Now back in autograd land, recompute the actual weights using the fixed 
    # rho and sigma constants, allowing gradients to flow to knn_dists and thus X.
    # Compiled: fixed-shape dense ops all the way through symmetrization.
    rhos = rhos_ng.detach()
    sigma = torch.clamp(sigma_ng.detach(), min=1e-10)

    rows, cols, w_sym = _SYMMETRIZE_KERNEL(knn_dists, knn_indices, rhos, sigma, n, k)

    # NOT compiled: nonzero()'s output size is data-dependent, which
    # fullgraph=True compilation can't handle as a fixed-shape graph.
    threshold = w_sym.max() / max(n_epochs, 1)
    active = torch.nonzero(w_sym >= threshold, as_tuple=True)[0]

    return rows[active], cols[active], w_sym[active]


# ---------------------------------------------------------------------------
# Empirical weight computation
#
# The algorithm runs on the *entire* dataset as a single logical batch --
# every point is compared against every other point, exactly like an
# unbounded-memory O(n^2) implementation would. What's bounded is memory,
# not the input: the similarity graph is stored sparse (only the ~n*k
# nonzero edges, not a dense n x n matrix), and the O(n^2) pairwise-distance
# work needed for the eps radius search is streamed in row-chunks (a la
# `torch_knn`'s cdist chunking) so peak memory is O(chunk_size * n) instead
# of O(n^2). The binary search for eps is inherently sequential (each step's
# bounds depend on the previous step's result), so it stays a scalar
# Python-level loop -- but each step's condition check is itself chunked.
# ---------------------------------------------------------------------------

def _build_sparse_similarity(X, n_neighbors, n_epochs, device):
    """Fuzzy simplicial set stored as a sparse (n, n) COO tensor (~n*k
    nonzeros) instead of a dense matrix."""
    n = X.shape[0]
    knn_idx, knn_dist = torch_knn(X, n_neighbors, device=device)
    rows, cols, vals = fuzzy_simplicial_set_torch(
        knn_idx, knn_dist, n, n_neighbors, n_epochs, device
    )
    return torch.sparse_coo_tensor(
        torch.stack([rows, cols]), vals, size=(n, n), device=device,
        check_invariants=False,
    ).coalesce()


def _sparse_similarity_layout(S):
    """
    Precompute a CSR-like layout (sorted rows/cols/vals + row offset
    pointers) once, so every chunked pass below can slice an arbitrary
    row-range via plain Python indexing (`row_ptr[start:end]`) instead of
    re-running `torch.searchsorted` (and a GPU sync) on every chunk.
    """
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


# Single config, device/shape-agnostic (torch.compile guards+recompiles per
# shape internally) -- built once at import time.
_PAIRWISE_DIST_KERNEL = _make_pairwise_dist_from_cross()


def _chunk_pairwise_dist(S, rows_sorted, cols_sorted, vals_sorted, row_ptr, row_norm_sq, start, end):
    """
    Distances from every point (n) to the points in row-range [start, end),
    computed without ever materializing a dense (n, n) matrix:
    dense-ify just this row chunk from the sparse graph, get dot products
    against all n rows via one sparse @ dense matmul, then finish with
    ||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b>.
    Returns dist (n, chunk) and idx (chunk,) -- the absolute row indices.

    The sparse.mm stays eager (torch.compile can't wrap a sparse-tensor
    argument, same limitation as the shift kernel's spmm); only the
    dense distance-finishing math after it is compiled. This function is
    called up to ~50x per eps binary-search call, per chunk, so the
    compiled part gets reused heavily across a single forward() and across
    the many forward() calls in a bench loop.
    """
    n = S.shape[0]
    device = S.device
    lo, hi = row_ptr[start], row_ptr[end]
    c = end - start

    chunk_dense = torch.zeros((c, n), dtype=row_norm_sq.dtype, device=device)
    chunk_dense[rows_sorted[lo:hi] - start, cols_sorted[lo:hi]] = vals_sorted[lo:hi]

    # Matrix multiply guarantees we compute similarity against ALL `N` points
    cross = torch.sparse.mm(S, chunk_dense.T)                 # eager
    idx = torch.arange(start, end, device=device)
    dist = _PAIRWISE_DIST_KERNEL(row_norm_sq, idx, cross)      # compiled
    return dist, idx


def radius_counts_chunked(S, rows_sorted, cols_sorted, vals_sorted, row_ptr, row_norm_sq,
                          eps_tensor, chunk_size, temperature=None):
    """
    Count neighbours within radius `eps` for every point. Every point is
    still compared against every other point (same result as a full
    O(n^2) computation); only the peak memory is bounded, to
    O(chunk_size * n), by streaming over row-chunks.
    """
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
    """Global max/min pairwise distance (excluding self-pairs), streamed
    over row-chunks -- seeds the binary search bounds."""
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
    """
    Scalar binary search for the smallest eps satisfying the density
    condition -- inherently sequential (each step's bounds depend on the
    previous step), same as the original. Each condition check runs
    chunked, so the search stays within a fixed memory budget regardless
    of n, and gives the exact same eps a full O(n^2) search would.
    """
    lo, hi = low, high
    result = high
    found = False
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        # Hard count needed during search
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


def _calculate_eps_from_similarity(S, rows_sorted, cols_sorted, vals_sorted, row_ptr, row_norm_sq,
                                   n, nbd_sample_count_threshold,
                                   satisfiability_proportion, chunk_size):
    """
    Takes the sparse-similarity layout (rows_sorted, cols_sorted,
    vals_sorted, row_ptr, row_norm_sq) as input instead of rebuilding it
    via _sparse_similarity_layout -- callers that already have the layout
    (compute_weights_from_similarity_chunked, MeanShiftDensityEnhancement's
    learn_eps branch in forward()) no longer pay for a second
    searchsorted + scatter_add pass over the whole graph just to get eps.
    """
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


def compute_weights_from_similarity_chunked(S, n, nbd_sample_count_threshold,
                                            satisfiability_proportion, max_iters_weight_count,
                                            chunk_size, temperature=None, eps=None, layout=None):
    """
    Chunked (memory-bounded, exact) replacement for the original dense
    per-batch weight computation. Operates on the entire similarity graph
    `S` as a single logical batch.
    """
    if layout is None:
        layout = _sparse_similarity_layout(S)
    rows_sorted, cols_sorted, vals_sorted, row_ptr, row_norm_sq = layout

    if eps is None:
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
    """
    Cached per has_temperature (a fixed, per-model-instance setting) --
    baked as a closure constant rather than passed as a runtime arg so the
    `if` never appears inside the compiled graph.
    """
    if has_temperature not in _DENSE_WEIGHT_STEP_CACHE:
        _DENSE_WEIGHT_STEP_CACHE[has_temperature] = _make_dense_weight_step(has_temperature)
    return _DENSE_WEIGHT_STEP_CACHE[has_temperature]


def compute_weights_from_similarity_dense(S, n, nbd_sample_count_threshold,
                                          satisfiability_proportion, max_iters_weight_count,
                                          temperature=None, eps=None):
    """
    Unbatched (memory-heavy but fast) empirical weight computation.
    Prioritizes O(1) ops via huge dense matrices, bypassing chunking.
    Used when use_chunking=False. Peak VRAM footprint is O(N^2).
    """
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
    eps=None
):
    n = X.shape[0]
    S = _build_sparse_similarity(X, n_neighbors, n_epochs, device)

    if use_chunking:
        return compute_weights_from_similarity_chunked(
            S, n, nbd_sample_count_threshold, satisfiability_proportion,
            max_iters_weight_count, chunk_size, temperature, eps
        )
    else:
        return compute_weights_from_similarity_dense(
            S, n, nbd_sample_count_threshold, satisfiability_proportion,
            max_iters_weight_count, temperature, eps
        )


# ---------------------------------------------------------------------------
# Core shift kernel
#
# Two structurally different kernels, selected once per model instance
# (not per call):
#
#   sparse path : barycenter = torch.sparse.mm(W, X), where W is the
#                 (n, n) row-normalized neighbour-weight matrix, built once
#                 (indices_fixed + w_norm are loop-invariant across shift
#                 iterations). No (n, k, d) gather every iteration.
#   gather path : original X[indices] gather + weighted sum. Used when
#                 sparse ops aren't supported on the target device (see
#                 _sparse_mm_supported) or explicitly disabled.
#
# IMPORTANT: torch.compile cannot trace a function that takes a sparse
# tensor as an argument at all ("Attempted to wrap sparse Tensor") --
# this is a hard limitation, unrelated to whether the *eager* op runs fine
# (which _sparse_mm_supported checks). So torch.sparse.mm is always called
# eagerly, outside any compiled region; only the movement math that follows
# (dense-only: diff, clipping, update) is compiled.
#
# clipping / clip_mode / low_precision are captured as plain Python closure
# constants, not passed as runtime tensor/scalar arguments, so each
# compiled kernel is one straight-line graph with no runtime branch and no
# per-call guard/recompile risk.
# ---------------------------------------------------------------------------

def _build_sparse_weight_matrix(indices_i64, w_norm, n, device):
    """
    (n, n) sparse COO weight matrix from a fixed (n, k) neighbour index
    table and its row-normalized weights. Built once per forward() call
    (indices_fixed and w_norm don't change across shift iterations), then
    reused every iteration via torch.sparse.mm(W, X) in place of a fresh
    gather. NOTE: sparse_coo_tensor indices must be int64 -- pass the
    int64 copy of indices_fixed here, not the int32 gather-path copy.
    """
    k = indices_i64.shape[1]
    rows = torch.arange(n, device=device).repeat_interleave(k)
    cols = indices_i64.reshape(-1)
    vals = w_norm.reshape(-1)
    return torch.sparse_coo_tensor(
        torch.stack([rows, cols]), vals, size=(n, n), device=device,
        check_invariants=False,
    ).coalesce()


def _make_movement_kernel(clipping, clip_mode, needs_gather, low_precision, low_precision_dtype=torch.bfloat16):
    """
    Compiled, dense-only kernel: given X and (for the gather path) the
    fixed neighbour indices, produces the barycenter -- via gather+weighted
    sum if needs_gather, or takes a precomputed barycenter tensor if not
    (sparse path, where torch.sparse.mm already ran eagerly) -- then does
    the movement math (diff, magnitude, optional clipping, update). Never
    sees a sparse tensor, so it's safe to wrap in fullgraph=True.

    Called as:
      needs_gather=True  (dense path):  _kernel(X, indices, w_or_barycenter, learning_rate, alpha, gate)
      needs_gather=False (sparse path): _kernel(X, indices, w_or_barycenter, learning_rate, alpha, gate)
    In both cases the 3rd argument is either the (n, k) weight tensor
    (gather path) or the already-computed (n, d) barycenter (sparse path);
    which one it is is fixed by needs_gather, a closure constant.
    """

    def _kernel(X, indices, w_or_barycenter, learning_rate, alpha, gate):
        n, k = indices.shape
        neighbor_pos = None

        if needs_gather:
            neighbor_pos = X[indices]                                  # (n, k, d) -- one gather
            if low_precision:
                lp_w = w_or_barycenter.to(low_precision_dtype)
                lp_neighbors = neighbor_pos.to(low_precision_dtype)
                barycenter = (lp_w.unsqueeze(-1) * lp_neighbors).sum(dim=1).to(X.dtype)
            else:
                barycenter = (w_or_barycenter.unsqueeze(-1) * neighbor_pos).sum(dim=1)
        else:
            barycenter = w_or_barycenter                                # already computed via sparse.mm outside

        diff = barycenter - X
        dist_move = diff.norm(dim=1)
        moved = dist_move >= 1e-8

        if clipping and clip_mode > 0:
            if neighbor_pos is None:
                neighbor_pos = X[indices]           # only extra gather needed under the sparse path
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

        updated = X + gate * scale * diff
        revised_d = torch.where(moved.unsqueeze(-1), updated, X)
        change = torch.where(moved, dist_move, torch.zeros_like(dist_move))
        return revised_d, change

    return torch.compile(_kernel, fullgraph=True)


def _make_shift_kernel(clipping, clip_mode, use_sparse, low_precision, low_precision_dtype=torch.bfloat16):
    """
    Build the top-level shift step for a fixed
    (clipping, clip_mode, use_sparse, low_precision) configuration. Call
    once per model instance and reuse -- not per shift iteration.

    Returns a plain (uncompiled) Python function -- it does the eager
    torch.sparse.mm when use_sparse, then delegates to a compiled
    dense-only movement kernel. The wrapper itself must stay uncompiled:
    that's what keeps the sparse tensor from ever entering a compiled
    region.
    """
    movement_kernel = _make_movement_kernel(
        clipping, clip_mode, needs_gather=not use_sparse,
        low_precision=low_precision, low_precision_dtype=low_precision_dtype,
    )

    if use_sparse:
        def _shift(X, indices, w_or_W, learning_rate, alpha, gate):
            barycenter = torch.sparse.mm(w_or_W, X)          # eager -- torch.compile can't wrap sparse tensors
            return movement_kernel(X, indices, barycenter, learning_rate, alpha, gate)
    else:
        def _shift(X, indices, w_or_W, learning_rate, alpha, gate):
            return movement_kernel(X, indices, w_or_W, learning_rate, alpha, gate)

    return _shift


# Cache kernels across model instances that share the same config -- avoids
# both a redundant Python closure build and (for the compiled inner
# movement kernel) a redundant torch.compile trace for identical configs.
_SHIFT_KERNEL_CACHE = {}


def _get_shift_kernel(clipping, clip_mode, use_sparse, low_precision):
    key = (clipping, clip_mode, use_sparse, low_precision)
    if key not in _SHIFT_KERNEL_CACHE:
        _SHIFT_KERNEL_CACHE[key] = _make_shift_kernel(clipping, clip_mode, use_sparse, low_precision)
    return _SHIFT_KERNEL_CACHE[key]


# ---------------------------------------------------------------------------
# Core shift loop (Fully Differentiable)
# ---------------------------------------------------------------------------

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
    ):
        super().__init__()

        if learn_temperature and temperature is None:
            raise ValueError("learn_temperature=True requires an explicit initial temperature")
        if learn_eps and temperature is None:
            raise ValueError("learn_eps=True requires temperature to enable differentiable weights")
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
        # Normalize None -> 0 so the forward()-loop guard can treat both as
        # "falsy => never recompute after the first iteration" uniformly.
        self.recompute_neighbors = recompute_neighbors or 0

        # Resolved once per instance (device doesn't change afterward), not
        # per forward() call: whether to use the sparse-spmm barycenter path
        # (no per-iteration (n, k, d) gather) vs. the dense gather fallback.
        self.use_sparse_shift = use_sparse_shift and _sparse_mm_supported(device)
        self.low_precision_barycenter = low_precision_barycenter
        self._shift_kernel = _get_shift_kernel(
            self.clipping, self.clip_mode, self.use_sparse_shift, self.low_precision_barycenter
        )

        dtype = next(
            (value.dtype for value in (learning_rate, alpha, temperature, eps)
             if isinstance(value, torch.Tensor) and value.is_floating_point()),
            torch.get_default_dtype(),
        )
        scalar_options = (
            ("learning_rate", learning_rate, learn_learning_rate),
            ("alpha", alpha, learn_alpha),
            ("temperature", temperature, learn_temperature),
            ("eps", eps, learn_eps),
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

        _configure_logging(log_file)

    def forward(self, X, gate=1.0):
        """Run MSDE and return shifted data, movement, and trajectory."""
        if not self.enable_gradients:
            X = X.detach()

        if self.learn_eps and self.eps is None:
            similarity = _build_sparse_similarity(X, 15, 200, self.device_name)
            # Build once, reuse for both the eps calc below and the chunked weight pass that follows
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

        n_samples = X.shape[0]
        shifted_dataset = X.clone()
        total_distance = torch.zeros(n_samples, device=self.device_name)
        trajectory = [shifted_dataset.clone()] if self.keep_trajectory else []

        logger.info(
            f"Computing fixed k-NN (k={self.k}) in feature space on {self.device_name} ..."
        )

        # indices_fixed / w_norm / w_or_W are (re)built either once, before
        # the loop (recompute_neighbors=0/None -- original "fixed neighbour
        # graph" behaviour, cheapest), or every N iterations
        # (recompute_neighbors=N -- periodic adaptive mean-shift: the
        # neighbour graph tracks shifted_dataset every N steps instead of
        # every step, damping the runaway-collapse feedback loop that
        # recomputing every single iteration produces -- see below). The
        # block itself is identical in every case; only *when* it runs
        # changes, via the `iter_count == 0 or iter_count % self.recompute_neighbors == 0`
        # guard below. iter_count == 0 always (re)builds it regardless of N,
        # since indices_fixed/w_or_W don't exist yet on the first pass.
        #
        # Note base_weights_t itself is NOT recomputed here even when
        # recompute_neighbors is set -- it comes from a separate, far more
        # expensive pipeline (get_empirical_weights's eps binary search over
        # the *original* X) and is treated as a static "how typical is this
        # point" prior. Only the neighbour topology used for the barycenter
        # gather/spmm is refreshed. Recomputing base_weights_t on the
        # shifted data would be possible too, but multiplies the dominant
        # cost of forward() by (max_iters_shift // N) -- do that only if you
        # have a specific reason the density prior itself needs to track the
        # shift, not just the neighbour set.
        indices_fixed = None
        w_or_W = None

        for iter_count in range(self.max_iters_shift):
            if iter_count == 0 or (
                self.recompute_neighbors and iter_count % self.recompute_neighbors == 0
            ):
                with torch.no_grad():
                    # int64 needed for torch.sparse_coo_tensor's index tensor;
                    # int32 halves gather bandwidth for the dense fallback path
                    # and for the clipping-only gather under the sparse path.
                    # Safe given n <= ~100k always fits comfortably in int32's
                    # range -- if you ever run this on datasets north of
                    # ~2^31 points, keep indices_fixed as int64 instead.
                    indices_fixed_i64 = compute_fixed_knn(
                        shifted_dataset.detach(), self.k, device=self.device_name
                    )
                    indices_fixed = indices_fixed_i64.to(torch.int32)

                w = base_weights_t[indices_fixed_i64]                        # (n, k)
                denom = w.sum(dim=1, keepdim=True).clamp_min(1e-6)           # (n, 1)
                w_norm = w / denom                                           # fold the divide in once, not per iteration

                if self.use_sparse_shift:
                    w_or_W = _build_sparse_weight_matrix(
                        indices_fixed_i64, w_norm, n_samples, self.device_name
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
            )

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