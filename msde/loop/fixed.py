from .base import ShiftLoop


class FixedLoop(ShiftLoop):
    """Always run exactly ``max_iters_shift`` iterations: no data-dependent
    branching, so the computational graph is fixed. Neighbours are recomputed
    every ``recompute_neighbors`` iterations (default 1: every iteration)."""

    def __init__(self, max_iters_shift=5, recompute_neighbors=1, keep_trajectory=False):
        super().__init__(max_iters_shift, recompute_neighbors, keep_trajectory)
