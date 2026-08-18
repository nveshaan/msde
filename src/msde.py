import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)

DEFAULT_DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


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


# ---------------------------------------------------------------------------
# k-NN (brute-force, GPU, chunked over the query dimension)
# ---------------------------------------------------------------------------

def torch_knn(X, k, device=DEFAULT_DEVICE, chunk_size=4096):
    """
    Exact brute-force k-NN on GPU via chunked cdist + topk.

    Includes the point itself as its own nearest neighbour (distance 0),
    matching the convention of pynndescent / umap.nearest_neighbors that
    the original code relied on.

    Assumes X is already a tensor on `device`; no-op copy if so.

    Returns
    -------
    knn_indices : LongTensor (n, k) on device
    knn_dists   : FloatTensor (n, k) on device
    """
    X = torch.as_tensor(X, dtype=torch.float32, device=device)
    n = X.shape[0]

    all_idx = torch.empty((n, k), dtype=torch.long, device=device)
    all_dist = torch.empty((n, k), dtype=torch.float32, device=device)

    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        d = torch.cdist(X[start:end], X)          # (chunk, n)
        dists, idx = d.topk(k, largest=False, dim=1)
        all_idx[start:end] = idx
        all_dist[start:end] = dists

    return all_idx, all_dist


def compute_fixed_knn(X, k, device=DEFAULT_DEVICE, chunk_size=4096):
    """
    Compute fixed k-NN indices directly in feature space.
    Drop-in GPU replacement for the original NNDescent-based version.
    Returns a torch.LongTensor (kept on `device`) so it can be reused
    across shift iterations without host<->device round trips.
    """
    indices, _ = torch_knn(X, k, device=device, chunk_size=chunk_size)
    return indices


def compute_knn_dists(X, indices):
    """
    Compute pairwise distances from X to its fixed k-NN (by index).
    X       : FloatTensor (n, d) on device
    indices : LongTensor  (n, k) on same device
    Returns : FloatTensor (n, k) on same device
    """
    diff = X.unsqueeze(1) - X[indices]     # (n, k, d)
    return diff.norm(dim=-1)


# ---------------------------------------------------------------------------
# Fuzzy simplicial set (UMAP graph), fully vectorized on GPU
# ---------------------------------------------------------------------------

def fuzzy_simplicial_set_torch(knn_indices, knn_dists, n, n_neighbors, n_epochs=200,
                                device=DEFAULT_DEVICE):
    """
    GPU port of UMAP's smooth-knn-dist + membership-strength + fuzzy-union
    construction. All N per-point binary searches for sigma run in lockstep,
    using boolean masks to freeze converged points instead of branching.
    """
    k = n_neighbors
    target = float(np.log2(k))

    knn_dists_t = torch.as_tensor(knn_dists, dtype=torch.float32, device=device)
    knn_indices_t = torch.as_tensor(knn_indices, dtype=torch.long, device=device)

    # Rho: distance to nearest non-zero neighbor
    mask = knn_dists_t > 0
    rhos = torch.where(mask, knn_dists_t, torch.tensor(float("inf"), device=device))
    rhos = torch.clamp(rhos.min(dim=1).values, min=1e-8)

    # Binary search for sigma, vectorized over all N points
    lo = torch.full((n,), 1e-20, device=device)
    hi = torch.full((n,), 1e3, device=device)
    sigma = torch.ones(n, device=device)
    dists_shifted = torch.clamp(knn_dists_t - rhos[:, None], min=0.0)
    dists_shifted_tail = dists_shifted[:, 1:]  # skip j=0 (rho contributor)

    for _ in range(64):
        vals = torch.exp(-dists_shifted_tail / sigma[:, None])
        vals_sum = vals.sum(dim=1)

        converged = (vals_sum - target).abs() < 1e-5
        too_high = (vals_sum > target) & ~converged
        too_low = (vals_sum < target) & ~converged

        hi = torch.where(too_high, sigma, hi)
        lo = torch.where(too_low, sigma, lo)
        sigma = torch.where(too_high, (lo + sigma) / 2.0, sigma)
        sigma = torch.where(
            too_low,
            torch.where(hi >= 1e3, sigma * 2.0, (sigma + hi) / 2.0),
            sigma,
        )

        if bool(converged.all()):
            break

    # Edge weights
    weights = torch.exp(-dists_shifted / torch.clamp(sigma[:, None], min=1e-10))

    rows = torch.arange(n, device=device).repeat_interleave(k)
    cols = knn_indices_t.reshape(-1)
    vals = weights.reshape(-1)

    # Symmetrize on GPU: P = A + A^T - A * A^T
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
        check_invariants=False,  # rows/cols are valid knn indices by construction
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
    ).tolist()  # one sync, not one per chunk
    row_norm_sq = torch.zeros(n, dtype=torch.float32, device=S.device)
    row_norm_sq.scatter_add_(0, rows_sorted, vals_sorted ** 2)
    return rows_sorted, cols_sorted, vals_sorted, row_ptr, row_norm_sq


def _chunk_pairwise_dist(S, rows_sorted, cols_sorted, vals_sorted, row_ptr, row_norm_sq, start, end):
    """
    Distances from every point (n) to the points in row-range [start, end),
    computed without ever materializing a dense (n, n) matrix:
    dense-ify just this row chunk from the sparse graph, get dot products
    against all n rows via one sparse @ dense matmul, then finish with
    ||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b>.
    Returns dist (n, chunk) and idx (chunk,) -- the absolute row indices.
    """
    n = S.shape[0]
    device = S.device
    lo, hi = row_ptr[start], row_ptr[end]
    c = end - start

    chunk_dense = torch.zeros((c, n), dtype=row_norm_sq.dtype, device=device)
    chunk_dense[rows_sorted[lo:hi] - start, cols_sorted[lo:hi]] = vals_sorted[lo:hi]

    cross = torch.sparse.mm(S, chunk_dense.T)                       # (n, c)
    idx = torch.arange(start, end, device=device)
    sq_dist = row_norm_sq[:, None] + row_norm_sq[idx][None, :] - 2.0 * cross
    dist = torch.sqrt(torch.clamp(sq_dist, min=0.0))                # (n, c)
    return dist, idx


def radius_counts_chunked(S, rows_sorted, cols_sorted, vals_sorted, row_ptr, row_norm_sq,
                           eps, chunk_size):
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
        counts[start:end] = (dist < eps).sum(dim=0).float() - 1.0  # exclude self
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
        counts = radius_counts_chunked(
            S, rows_sorted, cols_sorted, vals_sorted, row_ptr, row_norm_sq, mid, chunk_size
        )
        satisfied = int((counts > threshold).sum().item())  # one sync per iteration
        if satisfied >= required:
            result, found, hi = mid, True, mid
        else:
            lo = mid
        if abs(hi - lo) < tol:
            break
    return result, found


def compute_weights_from_similarity_chunked(S, n, nbd_sample_count_threshold,
                                             satisfiability_proportion, max_iters_weight_count,
                                             chunk_size):
    """
    Chunked (memory-bounded, exact) replacement for the original dense
    per-batch weight computation. Operates on the entire similarity graph
    `S` as a single logical batch.
    """
    rows_sorted, cols_sorted, vals_sorted, row_ptr, row_norm_sq = _sparse_similarity_layout(S)

    max_dist, min_dist = _min_max_dist_chunked(
        S, rows_sorted, cols_sorted, vals_sorted, row_ptr, row_norm_sq, chunk_size
    )

    threshold = max(1, n - 1) if nbd_sample_count_threshold >= n else nbd_sample_count_threshold
    required = int(satisfiability_proportion * n)

    eps, found = _binary_search_eps_chunked(
        S, rows_sorted, cols_sorted, vals_sorted, row_ptr, row_norm_sq,
        min_dist, max_dist, threshold, required, chunk_size,
    )

    if not found:
        relaxed_thresh = max(1, threshold // 2)
        relaxed_required = required // 2
        eps, found = _binary_search_eps_chunked(
            S, rows_sorted, cols_sorted, vals_sorted, row_ptr, row_norm_sq,
            min_dist, max_dist, relaxed_thresh, relaxed_required, chunk_size,
        )

    if not found:
        eps = max_dist

    delta = (eps - 1e-6) / max_iters_weight_count
    total_counts = torch.zeros(n, dtype=torch.float32, device=S.device)
    eps_running = eps
    for _ in range(max_iters_weight_count):
        total_counts += radius_counts_chunked(
            S, rows_sorted, cols_sorted, vals_sorted, row_ptr, row_norm_sq, eps_running, chunk_size
        )
        eps_running -= delta

    return total_counts / max_iters_weight_count


def get_empirical_weights(
    X,
    nbd_sample_count_threshold=5,
    max_iters_weight_count=4,
    satisfiability_proportion=0.3,
    n_neighbors=15,
    random_state=42,   # kept for signature compatibility; unused (exact GPU knn)
    n_epochs=200,
    device=DEFAULT_DEVICE,
    chunk_size=2048,
):
    """
    Compute empirical density weights for each point over the *full*
    dataset as a single batch -- no data is split or approximated.

    Memory is bounded by chunking the O(n^2) distance computation itself
    (sparse similarity graph + row-chunked distances), not by chunking
    the input, so results are identical to an unbounded-memory run. Lower
    `chunk_size` trades speed for a smaller peak-memory footprint.

    Returns a FloatTensor (n,) on `device` -- kept as a tensor (not numpy)
    so callers can chain it into further GPU work without a round trip.
    """
    X = torch.as_tensor(X, dtype=torch.float32, device=device)
    n = X.shape[0]

    S = _build_sparse_similarity(X, n_neighbors, n_epochs, device)

    return compute_weights_from_similarity_chunked(
        S, n, nbd_sample_count_threshold, satisfiability_proportion,
        max_iters_weight_count, chunk_size,
    )


# ---------------------------------------------------------------------------
# Core shift kernel — density-weighted barycenter shift, fully vectorized on GPU
# ---------------------------------------------------------------------------

def shift_data(
    X,
    indices,
    dists,
    base_weights,
    learning_rate,
    clipping=False,
    clip_mode=0,       # 0 = no clipping, 1 = soft, 2 = hard
    alpha=0.5,
):
    """
    One DMSL shift step: move each point toward the barycenter of its k
    fixed neighbours, weighted only by each neighbour's base (empirical
    density) weight — no distance-based (t-kernel) term. Vectorized
    PyTorch version — no Python-level loop over samples.

    Parameters
    ----------
    X             : FloatTensor (n, d)      on device
    indices       : LongTensor  (n, k)      on device
    dists         : FloatTensor (n, k)      on device (used only for
                                             clipping's median-distance cap)
    base_weights  : FloatTensor (n,)        on device
    learning_rate : float
    clipping      : bool
    clip_mode     : int   0=none, 1=soft, 2=hard
    alpha         : float

    Returns
    -------
    revised_d : FloatTensor (n, d) on device
    change    : FloatTensor (n,)   on device
    """
    n, k = indices.shape

    # --- weighted barycenter ---
    w = base_weights[indices]                                   # (n, k)
    denom = w.sum(dim=1, keepdim=True).clamp_min(1e-6)          # (n, 1)

    neighbor_pos = X[indices]                                   # (n, k, d)
    revised_d = (w.unsqueeze(-1) * neighbor_pos).sum(dim=1) / denom   # (n, d)

    # --- movement magnitude ---
    diff = revised_d - X
    dist_move = diff.norm(dim=1)                                # (n,)
    change = dist_move.clone()

    moved = dist_move >= 1e-8   # points that actually move

    if clipping and clip_mode > 0:
        median_dist = dists.sort(dim=1).values[:, k // 2]        # (n,) matches tmp[k//2]
        delta = (alpha * median_dist).clamp_min(1e-8)

        if clip_mode == 1:
            effective_step = dist_move * (delta / (delta + dist_move))
        else:
            effective_step = torch.minimum(dist_move, delta)
    else:
        effective_step = dist_move

    dist_move_safe = dist_move.clamp_min(1e-12)  # avoid /0 for unmoved points
    scale = (learning_rate * effective_step / dist_move_safe).unsqueeze(-1)  # (n, 1)

    updated = X + scale * diff
    revised_d = torch.where(moved.unsqueeze(-1), updated, X)
    change = torch.where(moved, change, torch.zeros_like(change))

    return revised_d, change


# ---------------------------------------------------------------------------
# Core shift loop
# ---------------------------------------------------------------------------

def get_shift_fast(
    X,
    k,
    nbd_sample_count_threshold,
    learning_rate,
    max_iters_shift,
    shift_threshold,
    clipping=False,
    clip_mode=0,
    alpha=0.5,
    device=DEFAULT_DEVICE,
    keep_trajectory=False,
    log_file=None,
    weight_chunk_size=2048,
):
    """
    Run the DMSL shift loop on GPU.

    k-NN indices are computed once via brute-force GPU k-NN and reused
    across all iterations (only `dists` and positions change per step).
    Everything stays as a device tensor end-to-end; the only host<->device
    transfers are the required input upload and the final output download
    (plus optional per-iteration trajectory snapshots, opt-in only).

    weight_chunk_size : int — row-chunk size used by the empirical-weight
                              eps search (memory/speed knob only; does not
                              change the result). Lower it if you hit an
                              out-of-memory error on `get_empirical_weights`.
    """
    _configure_logging(log_file)

    X_t = torch.as_tensor(X, dtype=torch.float32, device=device)

    base_weights_t = get_empirical_weights(
        X_t,
        nbd_sample_count_threshold=nbd_sample_count_threshold,
        max_iters_weight_count=4,
        satisfiability_proportion=0.3,
        chunk_size=weight_chunk_size,
        device=device,
    )

    n_samples = X_t.shape[0]
    shifted_dataset = X_t.clone()
    total_distance = torch.zeros(n_samples, device=device)
    trajectory = [shifted_dataset.clone().cpu().numpy()] if keep_trajectory else []

    logger.info(f"Computing fixed k-NN (k={k}) in feature space on {device} ...")
    indices_fixed = compute_fixed_knn(X_t, k, device=device)

    with torch.no_grad():
        for iter_count in range(max_iters_shift):
            dists = compute_knn_dists(shifted_dataset, indices_fixed)

            revised_d, change = shift_data(
                shifted_dataset,
                indices_fixed,
                dists,
                base_weights_t,
                learning_rate,
                clipping,
                clip_mode,
                alpha,
            )

            total_distance += change
            shifted_dataset = revised_d
            if keep_trajectory:
                trajectory.append(shifted_dataset.clone().cpu().numpy())

            mean_change = change.mean().item()   # single sync point per iter
            logger.debug(f"Iter {iter_count + 1}: mean change = {mean_change:.6f}")

            if mean_change < shift_threshold:
                logger.info(f"Converged at iteration {iter_count + 1}.")
                break

    shifted_dataset_np = shifted_dataset.cpu().numpy()
    total_distance_np = total_distance.cpu().numpy()

    return shifted_dataset_np, total_distance_np, trajectory


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def mean_shift_density_enhancement(
    X,
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
    weight_chunk_size=2048,
):
    """
    Density-Mode Shift Learning (DMSL) — main entry point, GPU (PyTorch).

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    k : int
    nbd_sample_count_threshold : int
    learning_rate : float
    max_iters_shift : int
    shift_threshold : float
    clipping  : bool  — enable per-step clipping; pass True for trajectory
                        runs only, leave False for clustering runs
    clip_mode : int   — 0=none, 1=soft (smooth saturation), 2=hard (strict cap)
    alpha     : float — step cap = alpha * local_median_neighbour_dist
    device    : str   — 'mps' or 'cuda' or 'cpu'
    keep_trajectory : bool — if False, skips per-iteration device->host
                              copies of the full dataset (faster, lower
                              host memory, for runs that don't need it)
    log_file  : str or None — path to write logs to; if None, logging is
                              disabled entirely (nothing is logged anywhere)
    weight_chunk_size : int — memory/speed knob for the empirical-weight
                              eps search; lower it under tight VRAM. Does
                              not change the computed weights, since every
                              point is still compared against every other
                              point regardless of chunk size.

    Returns
    -------
    data_shifted   : np.ndarray
    total_distance : np.ndarray
    trajectory     : list of np.ndarray
    """
    data_shifted, total_distance, trajectory = get_shift_fast(
        X, k, nbd_sample_count_threshold,
        learning_rate, max_iters_shift, shift_threshold,
        clipping=clipping,
        clip_mode=clip_mode,
        alpha=alpha,
        device=device,
        keep_trajectory=keep_trajectory,
        log_file=log_file,
        weight_chunk_size=weight_chunk_size,
    )
    return data_shifted, total_distance, trajectory