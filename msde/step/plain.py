import torch

from ..common.utils import DEFAULT_DEVICE, sparse_mm_supported
from .base import MovementStep


def _build_sparse_weight_matrix(indices_i64, w_norm, n_rows, n_cols, device):
    """Build the sparse row-normalized neighbor matrix."""
    k = indices_i64.shape[1]
    rows = torch.arange(n_rows, device=device).repeat_interleave(k)
    cols = indices_i64.reshape(-1)
    vals = w_norm.reshape(-1)
    return torch.sparse_coo_tensor(
        torch.stack([rows, cols]), vals, size=(n_rows, n_cols), device=device,
        check_invariants=False,
    ).coalesce()


_MOVEMENT_GATE_EPS = 1e-8


def _make_movement_kernel(clipping, clip_mode, needs_gather, low_precision, low_precision_dtype=torch.bfloat16):
    """Build the compiled dense movement kernel."""

    def _kernel(X, indices, w_or_barycenter, learning_rate, alpha, gate):
        n, k = indices.shape
        neighbor_pos = None

        if needs_gather:
            neighbor_pos = X[indices]
            if low_precision:
                lp_w = w_or_barycenter.to(low_precision_dtype)
                lp_neighbors = neighbor_pos.to(low_precision_dtype)
                barycenter = (lp_w.unsqueeze(-1) * lp_neighbors).sum(dim=1).to(X.dtype)
            else:
                barycenter = (w_or_barycenter.unsqueeze(-1) * neighbor_pos).sum(dim=1)
        else:
            barycenter = w_or_barycenter

        diff = barycenter - X
        dist_move = diff.norm(dim=1)

        if clipping and clip_mode > 0:
            if neighbor_pos is None:
                neighbor_pos = X[indices]
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
        step = gate * scale * diff

        moved = dist_move >= _MOVEMENT_GATE_EPS
        updated = X + step
        revised_d = torch.where(moved.unsqueeze(-1), updated, X)
        change = torch.where(moved, dist_move, torch.zeros_like(dist_move))

        return revised_d, change

    return torch.compile(_kernel, fullgraph=True)


def _make_shift_kernel(clipping, clip_mode, use_sparse, low_precision, low_precision_dtype=torch.bfloat16):
    """Build one cached shift step for a fixed configuration."""
    movement_kernel = _make_movement_kernel(
        clipping, clip_mode, needs_gather=not use_sparse,
        low_precision=low_precision, low_precision_dtype=low_precision_dtype,
    )

    if use_sparse:
        def _shift(X, indices, w_or_W, learning_rate, alpha, gate):
            barycenter = torch.sparse.mm(w_or_W, X)
            return movement_kernel(X, indices, barycenter, learning_rate, alpha, gate)
    else:
        def _shift(X, indices, w_or_W, learning_rate, alpha, gate):
            return movement_kernel(X, indices, w_or_W, learning_rate, alpha, gate)

    return _shift


_SHIFT_KERNEL_CACHE = {}


def _get_shift_kernel(clipping, clip_mode, use_sparse, low_precision):
    key = (clipping, clip_mode, use_sparse, low_precision)
    if key not in _SHIFT_KERNEL_CACHE:
        _SHIFT_KERNEL_CACHE[key] = _make_shift_kernel(clipping, clip_mode, use_sparse, low_precision)
    return _SHIFT_KERNEL_CACHE[key]


class PlainBarycenter(MovementStep):
    """Barycenter step over the selected neighbours.

    Weights are the density prior of each selected neighbour, times the
    label-validity mask, normalised per row. ``use_sparse_shift`` computes the
    barycenter with a sparse matmul (no per-iteration (n, k, d) gather) when the
    device supports it.

    clipping / clip_mode : 0 = off, 1 = soft-saturating step, 2 = hard cap at
        ``alpha`` times the median neighbour distance.
    low_precision_barycenter : gather-mode barycenter in bfloat16.
    """

    def __init__(
        self,
        learning_rate=0.3,
        alpha=0.5,
        clipping=False,
        clip_mode=0,
        use_sparse_shift=True,
        low_precision_barycenter=False,
        device=DEFAULT_DEVICE,
    ):
        super().__init__()
        self.clipping = clipping
        self.clip_mode = clip_mode
        self.device_name = device
        # Resolved once per instance: sparse-spmm barycenter vs dense gather.
        self.use_sparse_shift = use_sparse_shift and sparse_mm_supported(device)
        self.low_precision_barycenter = low_precision_barycenter

        self.register_scalar("learning_rate", learning_rate, device)
        self.register_scalar("alpha", alpha, device)
        self._shift_kernel = _get_shift_kernel(
            clipping, clip_mode, self.use_sparse_shift, low_precision_barycenter,
        )

    @torch.no_grad()
    def prepare(self, X, base_weights, neighbors):
        indices = neighbors.indices                                   # (n, k) int64
        w = base_weights[indices]                                     # (n, k)
        if neighbors.valid is not None:
            w = w * neighbors.valid

        denom = w.sum(dim=1, keepdim=True).clamp_min(1e-6)            # (n, 1)
        w_norm = w / denom                                            # fold the divide in once, not per iteration

        if self.use_sparse_shift:
            w_or_W = _build_sparse_weight_matrix(
                indices, w_norm, X.shape[0], X.shape[0], self.device_name
            )
        else:
            w_or_W = w_norm
        return indices.to(torch.int32), w_or_W

    @torch.no_grad()
    def step(self, X, state, gate=1.0):
        indices, w_or_W = state
        return self._shift_kernel(
            X, indices, w_or_W, self.learning_rate, self.alpha, gate,
        )
