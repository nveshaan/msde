from .base import ShiftLoop


class EarlyStopLoop(ShiftLoop):
    """Iterate up to ``max_iters_shift`` times, stopping once the mean movement
    drops below ``shift_threshold`` (a data-dependent ``.item()`` branch, so not
    meant for end-to-end differentiable graphs). Neighbours are computed once
    unless ``recompute_neighbors`` is a positive N."""

    def __init__(self, max_iters_shift=5, shift_threshold=0.0001, recompute_neighbors=0,
                 keep_trajectory=False):
        super().__init__(max_iters_shift, recompute_neighbors, keep_trajectory)
        self.shift_threshold = shift_threshold

    def converged(self, change):
        return change.mean().item() < self.shift_threshold
