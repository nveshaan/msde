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


def _scheduled_gate(scheduler, step, total_steps, start, end, device, dtype):
    if scheduler is None:
        return torch.as_tensor(1.0, device=device, dtype=dtype)

    progress = 1.0 if total_steps <= 1 else step / (total_steps - 1)
    if scheduler == "linear":
        value = progress
    elif scheduler == "cosine":
        value = 0.5 * (1.0 - torch.cos(torch.tensor(progress * torch.pi)))
    elif callable(scheduler):
        value = scheduler(step, total_steps)
    else:
        raise ValueError("gate_scheduler must be None, 'linear', 'cosine', or callable")

    value = torch.as_tensor(value, device=device, dtype=dtype)
    if value.numel() != 1:
        raise ValueError("gate_scheduler must return a scalar")
    return start + (end - start) * value


# ---------------------------------------------------------------------------
# k-NN (brute-force, GPU, chunked over the query dimension)
# ---------------------------------------------------------------------------

def torch_knn(X, k, device=DEFAULT_DEVICE, chunk_size=4096):
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

    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        d = torch.cdist(X[start:end], X)          # (chunk, n) - compares to ALL N
        dists, idx = d.topk(k, largest=False, dim=1)
        all_idx[start:end] = idx
        all_dist_list.append(dists)

    all_dist = torch.cat(all_dist_list, dim=0)
    return all_idx, all_dist


def compute_fixed_knn(X, k, device=DEFAULT_DEVICE, chunk_size=4096):
    indices, _ = torch_knn(X, k, device=device, chunk_size=chunk_size)
    return indices


@torch.compile(fullgraph=True)
def compute_knn_dists(X, indices):
    """
    Compute pairwise distances from X to its fixed k-NN (by index).
    Differentiable w.r.t X.
    """
    diff = X.unsqueeze(1) - X[indices]     # (n, k, d)
    return diff.norm(dim=-1)


# ---------------------------------------------------------------------------
# Fuzzy simplicial set (UMAP graph), fully vectorized on GPU
# ---------------------------------------------------------------------------

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
    rhos = rhos_ng.detach()
    sigma = torch.clamp(sigma_ng.detach(), min=1e-10)
    
    dists_shifted = torch.clamp(knn_dists - rhos[:, None], min=0.0)
    weights = torch.exp(-dists_shifted / sigma[:, None])

    rows = torch.arange(n, device=device).repeat_interleave(k)
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

    # Matrix multiply guarantees we compute similarity against ALL `N` points
    cross = torch.sparse.mm(S, chunk_dense.T)
    idx = torch.arange(start, end, device=device)
    sq_dist = row_norm_sq[:, None] + row_norm_sq[idx][None, :] - 2.0 * cross
    dist = torch.sqrt(torch.clamp(sq_dist, min=0.0))
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


def _calculate_eps_from_similarity(S, n, nbd_sample_count_threshold,
                                   satisfiability_proportion, chunk_size):
    rows_sorted, cols_sorted, vals_sorted, row_ptr, row_norm_sq = _sparse_similarity_layout(S)
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
                                            chunk_size, temperature=None, eps=None):
    """
    Chunked (memory-bounded, exact) replacement for the original dense
    per-batch weight computation. Operates on the entire similarity graph
    `S` as a single logical batch.
    """
    rows_sorted, cols_sorted, vals_sorted, row_ptr, row_norm_sq = _sparse_similarity_layout(S)

    if eps is None:
        with torch.no_grad():
            eps_tensor = _calculate_eps_from_similarity(
                S, n, nbd_sample_count_threshold,
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
    
    for _ in range(max_iters_weight_count):
        if temperature is not None:
            counts = torch.sigmoid((eps_running - dist) / temperature).sum(dim=1) - torch.sigmoid(eps_running / temperature)
        else:
            counts = (dist < eps_running).sum(dim=1).float() - 1.0
            
        total_counts = total_counts + counts
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
# ---------------------------------------------------------------------------

@torch.compile(fullgraph=True)
def shift_data(
    X,
    indices,
    w,
    denom,
    learning_rate,
    clipping=False,
    clip_mode=0,       
    alpha=0.5,
    gate=1.0,
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
    gate          : scalar multiplier applied to the update step

    Returns
    -------
    revised_d : FloatTensor (n, d) on device
    change    : FloatTensor (n,)   on device
    """
    n, k = indices.shape

    # --- weighted barycenter ---
    neighbor_pos = X[indices]                                   # (n, k, d)
    revised_d = (w.unsqueeze(-1) * neighbor_pos).sum(dim=1) / denom   # (n, d)

    # --- movement magnitude ---
    diff = revised_d - X
    dist_move = diff.norm(dim=1)                                
    
    moved = dist_move >= 1e-8   

    if clipping and clip_mode > 0:
        dists = (X.unsqueeze(1) - neighbor_pos).norm(dim=-1)
        # sort preserves autograd graph for the extracted elements
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
    
    # Use torch.where to avoid breaking gradients with in-place assignments
    revised_d = torch.where(moved.unsqueeze(-1), updated, X)
    change = torch.where(moved, dist_move, torch.zeros_like(dist_move))

    return revised_d, change


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
        gate_scheduler=None,
        gate_start=0.0,
        gate_end=1.0,
        device=DEFAULT_DEVICE,
        keep_trajectory=False,
        log_file=None,
        use_chunking=False,
        weight_chunk_size=2048,
        temperature=None,
        eps=None,
        enable_gradients=True,
        learn_temperature=False,
        learn_learning_rate=False,
        learn_alpha=False,
        learn_eps=False,
    ):
        super().__init__()

        if learn_temperature and temperature is None:
            raise ValueError("learn_temperature=True requires an explicit initial temperature")
        if learn_eps and temperature is None:
            raise ValueError("learn_eps=True requires temperature to enable differentiable weights")
        self.k = k
        self.nbd_sample_count_threshold = nbd_sample_count_threshold
        self.max_iters_shift = max_iters_shift
        self.shift_threshold = shift_threshold
        self.clipping = clipping
        self.clip_mode = clip_mode
        if gate_scheduler is not None and not isinstance(gate_scheduler, str) and not callable(gate_scheduler):
            raise TypeError("gate_scheduler must be None, 'linear', 'cosine', or callable")
        if isinstance(gate_scheduler, str) and gate_scheduler not in ("linear", "cosine"):
            raise ValueError("gate_scheduler must be None, 'linear', 'cosine', or callable")
        self.gate_scheduler = gate_scheduler
        self.gate_start = gate_start
        self.gate_end = gate_end
        self.device_name = device
        self.keep_trajectory = keep_trajectory
        self.use_chunking = use_chunking
        self.weight_chunk_size = weight_chunk_size
        self.enable_gradients = enable_gradients
        self.log_file = log_file
        self.learn_eps = learn_eps

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

    def forward(self, X):
        """Run MSDE and return shifted data, movement, and trajectory."""
        if not self.enable_gradients:
            X = X.detach()

        if self.learn_eps and self.eps is None:
            similarity = _build_sparse_similarity(X, 15, 200, self.device_name)
            with torch.no_grad():
                eps_initial = _calculate_eps_from_similarity(
                    similarity,
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

        with torch.no_grad():
            indices_fixed = compute_fixed_knn(X, self.k, device=self.device_name)

        w = base_weights_t[indices_fixed]                                   # (n, k)
        denom = w.sum(dim=1, keepdim=True).clamp_min(1e-6)          # (n, 1)

        for iter_count in range(self.max_iters_shift):
            gate = _scheduled_gate(
                self.gate_scheduler,
                iter_count,
                self.max_iters_shift,
                self.gate_start,
                self.gate_end,
                shifted_dataset.device,
                shifted_dataset.dtype,
            )

            revised_d, change = shift_data(
                shifted_dataset,
                indices_fixed,
                w, denom,
                self.learning_rate,
                self.clipping,
                self.clip_mode,
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