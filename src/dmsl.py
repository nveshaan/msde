"""
cellflow/msde.py
----------------
Density-Mode Shift Learning (DMSL) — the core trajectory algorithm.

Previously called MSDE (Mean Shift Diffusion Embedding). The algorithm has
been significantly updated in v7: it now uses UMAP-space fixed k-NN indices
(computed once on the original data) and a smooth t-distribution kernel for
the shift, replacing the earlier Gaussian-mean-shift kernel.

Change log (v7)
---------------
- UMAP is computed internally once during get_shift_fast() and its k-NN
  index is reused across all shift iterations (no redundant re-computation).
- The shift kernel is now a heavy-tailed t-distribution kernel combined
  with empirical density weights, giving smoother, more robust trajectories.
- The fitted umap_model is no longer returned by get_shift_fast() /
  mean_shift_manifold_learning() so the pipeline can reuse it instead of
  fitting UMAP a second time on the same data (Change 2).

Public API
----------
mean_shift_manifold_learning(X, ...)   <- main entry point
get_shift_fast(X, ...)                 <- lower-level, exposes weights
get_empirical_weights(X, ...)          <- weight computation
get_cell_trajectory(trajectory, id)   <- single-cell path extraction
"""

import logging
import numpy as np
from scipy.spatial import cKDTree
from umap.umap_ import fuzzy_simplicial_set, nearest_neighbors
from scipy.sparse import csr_matrix
from numba import njit, prange
from pynndescent import NNDescent
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Neighbourhood / density helpers
# ---------------------------------------------------------------------------

def count_points_within_radius(X, tree, epsilon):
    neighbors = tree.query_ball_tree(tree, epsilon)
    counts = np.array([len(pts) - 1 for pts in neighbors])
    return counts


def max_min_distances_kdtree(X):
    tree = cKDTree(X)
    dists, _ = tree.query(X, k=X.shape[0])
    all_distances = dists[:, 1:].flatten()
    return np.max(all_distances), np.min(all_distances)


def binary_search_condition(low, high, condition, tol=1e-4, max_iter=50):
    result = None
    for _ in range(max_iter):
        mid = (low + high) / 2
        if condition(mid):
            result = mid
            high = mid
        else:
            low = mid
        if abs(high - low) < tol:
            break
    return result


def condition_formulation(point_in_radius_counts, nbd_sample_count_threshold, satisfiability_proportion):
    if_satisfied = (np.where(point_in_radius_counts > nbd_sample_count_threshold)[0]) * 1
    return np.sum(if_satisfied) >= satisfiability_proportion


# ---------------------------------------------------------------------------
# Empirical weight computation
# ---------------------------------------------------------------------------

def get_empirical_weights(
    X,
    nbd_sample_count_threshold=5,
    max_iters_weight_count=4,
    satisfiability_proportion=0.3,
    n_neighbors=15,
    metric='euclidean',
    random_state=42,
    batch_size=1000,
):
    """
    Compute empirical density weights for each cell.

    Uses UMAP fuzzy simplicial sets to estimate local similarity structure,
    then performs a binary-search radius sweep to count neighbourhood
    occupancy, yielding a weight proportional to local density.
    """

    def umap_graph_similarity(X_batch):
        knn_indices, knn_dists, _ = nearest_neighbors(
            X_batch,
            n_neighbors=n_neighbors,
            metric=metric,
            metric_kwds={},
            angular=False,
            random_state=random_state,
            low_memory=True,
            use_pynndescent=True,
        )
        G, _, _ = fuzzy_simplicial_set(
            X_batch,
            n_neighbors=n_neighbors,
            random_state=random_state,
            metric=metric,
            knn_indices=knn_indices,
            knn_dists=knn_dists,
            angular=False,
            set_op_mix_ratio=1.0,
            local_connectivity=1.0,
            apply_set_operations=True,
            verbose=False,
        )
        return G.toarray() if isinstance(G, csr_matrix) else G

    def compute_weights_from_similarity(sim, X_batch):
        tree = cKDTree(sim)
        max_dist, min_dist = max_min_distances_kdtree(sim)

        threshold = (
            max(1, len(X_batch) - 1)
            if nbd_sample_count_threshold >= len(X_batch)
            else nbd_sample_count_threshold
        )
        effective_required = int(satisfiability_proportion * len(X_batch))

        eps = binary_search_condition(
            min_dist, max_dist,
            lambda mid: condition_formulation(
                count_points_within_radius(sim, tree, mid),
                threshold,
                effective_required,
            ),
        )

        if eps is None:
            relaxed_thresh = max(1, threshold // 2)
            relaxed_prop = effective_required // 2
            eps = binary_search_condition(
                min_dist, max_dist,
                lambda mid: condition_formulation(
                    count_points_within_radius(sim, tree, mid),
                    relaxed_thresh,
                    relaxed_prop,
                ),
            )

        if eps is None:
            eps = max_dist

        delta = (eps - 1e-6) / max_iters_weight_count
        all_counts = []
        for _ in range(max_iters_weight_count):
            counts = count_points_within_radius(sim, tree, eps)
            all_counts.append(counts)
            eps -= delta

        return np.mean(all_counts, axis=0)

    if len(X) <= 3 * n_neighbors:
        sim = umap_graph_similarity(X)
        return compute_weights_from_similarity(sim, X)

    effective_batch_size = min(batch_size, max(n_neighbors * 3, 100))
    total_batches = (len(X) + effective_batch_size - 1) // effective_batch_size

    weights_all = []
    for batch_idx in range(total_batches):
        start = batch_idx * effective_batch_size
        end = min(len(X), start + effective_batch_size)
        X_batch = X[start:end]
        sim = umap_graph_similarity(X_batch)
        weights_batch = compute_weights_from_similarity(sim, X_batch)
        weights_all.append(weights_batch)

    return np.concatenate(weights_all)


# ---------------------------------------------------------------------------
# Numba JIT kernel — t-distribution smooth shift (v7)
# ---------------------------------------------------------------------------

@njit(fastmath=True, parallel=True)
def shift_data_smooth_tkernel(
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
    One DMSL shift step using a heavy-tailed t-distribution kernel.

    Parameters
    ----------
    X             : ndarray (n_samples, n_features)
    indices       : ndarray (n_samples, k)  — fixed k-NN indices
    dists         : ndarray (n_samples, k)  — current distances to k-NN
    base_weights  : ndarray (n_samples,)    — empirical density weights
    learning_rate : float
    clipping      : bool  — enable step-size clipping (trajectory use only)
    clip_mode     : int   — 0=none, 1=soft (smooth saturation), 2=hard (strict cap)
    alpha         : float — local scale factor: delta = alpha * median_dist

    Returns
    -------
    revised_d : ndarray (n_samples, n_features) — shifted positions
    change    : ndarray (n_samples,)            — per-cell Euclidean displacement
    """
    n, k = indices.shape
    d = X.shape[1]

    revised_d = np.empty_like(X)
    change = np.empty(n)

    for i in prange(n):
        # --- t-kernel bandwidth (mean neighbour dist) ---
        sigma = 0.0
        for j in range(k):
            sigma += dists[i, j]
        sigma /= k
        if sigma < 1e-6:
            sigma = 1e-6

        # --- weighted barycenter ---
        denom = 0.0
        weights_local = np.empty(k)

        for j in range(k):
            dist = dists[i, j]
            w = 1.0 / (1.0 + (dist * dist) / (sigma * sigma))
            w *= base_weights[indices[i, j]]
            weights_local[j] = w
            denom += w

        if denom < 1e-6:
            denom = 1e-6

        for t in range(d):
            acc = 0.0
            for j in range(k):
                acc += weights_local[j] * X[indices[i, j], t]
            acc /= denom
            revised_d[i, t] = acc

        # --- movement magnitude ---
        dist_move = 0.0
        for t in range(d):
            diff = revised_d[i, t] - X[i, t]
            dist_move += diff * diff
        dist_move = np.sqrt(dist_move)

        change[i] = dist_move

        if dist_move < 1e-8:
            for t in range(d):
                revised_d[i, t] = X[i, t]
            continue

        # --- clipping block (trajectory only) ---
        if clipping and clip_mode > 0:
            # local scale: median of neighbour distances
            # Numba-compatible median: sort a copy, pick middle element
            tmp = np.empty(k)
            for j in range(k):
                tmp[j] = dists[i, j]
            # simple insertion sort (k is typically small, e.g. 30–100)
            for a in range(1, k):
                key = tmp[a]
                b = a - 1
                while b >= 0 and tmp[b] > key:
                    tmp[b + 1] = tmp[b]
                    b -= 1
                tmp[b + 1] = key
            median_dist = tmp[k // 2]
            delta = alpha * median_dist
            if delta < 1e-8:
                delta = 1e-8

            if clip_mode == 1:
                # soft: smooth saturation — asymptotically approaches delta
                effective_step = dist_move * (delta / (delta + dist_move))
            else:
                # hard: strict cap
                if dist_move < delta:
                    effective_step = dist_move
                else:
                    effective_step = delta
        else:
            effective_step = dist_move

        # --- apply learning rate and update ---
        scale = learning_rate * effective_step / dist_move
        for t in range(d):
            revised_d[i, t] = X[i, t] + scale * (revised_d[i, t] - X[i, t])

    return revised_d, change


# ---------------------------------------------------------------------------
# k-NN helpers
# ---------------------------------------------------------------------------

def compute_fixed_knn(X, k):
    """
    Compute fixed k-NN indices directly in HVG space using NNDescent.

    UMAP is no longer fitted here — it is the pipeline's responsibility
    to fit a visualisation UMAP separately with user-controlled parameters.
    """
    index = NNDescent(
        X,
        n_neighbors=k,
        metric="euclidean",
        random_state=42,
    )
    indices, _ = index.neighbor_graph
    indices = indices.astype(np.int64)
    return indices


@njit(fastmath=True, parallel=True)
def compute_knn_dists(X, indices):
    """Compute pairwise distances from X to its k-NN (by index)."""
    n, k = indices.shape
    d = X.shape[1]
    dists = np.empty((n, k), dtype=np.float32)

    for i in prange(n):
        for j in range(k):
            idx = indices[i, j]
            dist = 0.0
            for t in range(d):
                diff = X[i, t] - X[idx, t]
                dist += diff * diff
            dists[i, j] = np.sqrt(dist)

    return dists


# ---------------------------------------------------------------------------
# Single-cell path extraction
# ---------------------------------------------------------------------------

def get_cell_trajectory(trajectory_list, cell_id):
    """
    Extract the trajectory of a single cell across all DMSL iterations.

    Parameters
    ----------
    trajectory_list : list of np.ndarray
    cell_id : int

    Returns
    -------
    path : np.ndarray, shape (n_steps, n_features)
    """
    return np.array([pos[cell_id] for pos in trajectory_list])


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
    return_weights=False,
    weights=None,
    clipping=False,
    clip_mode=0,
    alpha=0.5,
):
    """
    Run the DMSL shift loop.

    k-NN indices are computed directly in HVG space via NNDescent (no UMAP).
    UMAP is fitted independently by the pipeline with user-controlled
    parameters and is no longer returned here.

    Parameters
    ----------
    clipping  : bool  — enable per-step clipping (trajectory runs only)
    clip_mode : int   — 0=none, 1=soft (smooth saturation), 2=hard (strict cap)
    alpha     : float — delta = alpha * local_median_dist

    Returns
    -------
    shifted_dataset : np.ndarray
    base_weights    : np.ndarray  (only when return_weights=True)
    total_distance  : np.ndarray
    trajectory      : list of np.ndarray
    """
    if weights is None:
        base_weights = get_empirical_weights(
            X,
            nbd_sample_count_threshold=nbd_sample_count_threshold,
            max_iters_weight_count=4,
            satisfiability_proportion=0.3,
            batch_size=1000,
        )
    else:
        base_weights = weights

    n_samples = X.shape[0]
    shifted_dataset = X.copy()
    total_distance = np.zeros(n_samples)
    trajectory = [shifted_dataset.copy()]

    logger.info(f"Computing fixed k-NN (k={k}) in HVG space …")
    indices_fixed = compute_fixed_knn(X, k)

    for iter_count in tqdm(range(max_iters_shift), desc="DMSL shifts"):
        indices = indices_fixed
        dists = compute_knn_dists(shifted_dataset, indices)

        revised_d, change = shift_data_smooth_tkernel(
            shifted_dataset,
            indices,
            dists,
            base_weights,
            learning_rate,
            clipping,
            clip_mode,
            alpha,
        )

        total_distance += change
        shifted_dataset = revised_d
        trajectory.append(shifted_dataset.copy())

        logger.debug(f"Iter {iter_count + 1}: mean change = {change.mean():.6f}")

        if change.mean() < shift_threshold:
            logger.info(f"Converged at iteration {iter_count + 1}.")
            break

    if return_weights:
        return shifted_dataset, base_weights, total_distance, trajectory
    else:
        return shifted_dataset, total_distance, trajectory


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def mean_shift_manifold_learning(
    X,
    k=30,
    nbd_sample_count_threshold=30,
    learning_rate=0.3,
    max_iters_shift=5,
    shift_threshold=0.0001,
    return_weights=False,
    clipping=False,
    clip_mode=0,
    alpha=0.5,
):
    """
    Density-Mode Shift Learning (DMSL) — main entry point.

    The function name is kept as mean_shift_manifold_learning for backwards
    compatibility with existing pipeline code.  The new name for the
    algorithm is DMSL (Density-Mode Shift Learning).

    UMAP is no longer fitted or returned here.  The pipeline fits its own
    visualisation UMAP independently via embed_umap(), giving users full
    control over n_neighbors, min_dist, and other UMAP parameters.

    Parameters
    ----------
    X : np.ndarray, shape (n_cells, n_features)
    k : int
    nbd_sample_count_threshold : int
    learning_rate : float
    max_iters_shift : int
    shift_threshold : float
    return_weights : bool
    clipping  : bool  — enable per-step clipping; pass True for trajectory
                        runs only, leave False for clustering runs
    clip_mode : int   — 0=none, 1=soft (smooth saturation), 2=hard (strict cap)
    alpha     : float — step cap = alpha * local_median_neighbour_dist

    Returns (return_weights=False)
    -------
    data_shifted   : np.ndarray
    total_distance : np.ndarray
    trajectory     : list of np.ndarray

    Returns (return_weights=True)
    -------
    data_shifted   : np.ndarray
    weights        : np.ndarray
    total_distance : np.ndarray
    trajectory     : list of np.ndarray
    """
    if not return_weights:
        data_shifted, total_distance, trajectory = get_shift_fast(
            X, k, nbd_sample_count_threshold,
            learning_rate, max_iters_shift, shift_threshold,
            return_weights=False,
            clipping=clipping,
            clip_mode=clip_mode,
            alpha=alpha,
        )
        return data_shifted, total_distance, trajectory

    else:
        data_shifted, weights, total_distance, trajectory = get_shift_fast(
            X, k, nbd_sample_count_threshold,
            learning_rate, max_iters_shift, shift_threshold,
            return_weights=True,
            clipping=clipping,
            clip_mode=clip_mode,
            alpha=alpha,
        )
        return data_shifted, weights, total_distance, trajectory
