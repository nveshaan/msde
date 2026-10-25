from ..common.utils import ScalarModule


class MovementStep(ScalarModule):
    """Moves points towards a weighted barycenter of their neighbours.

    A step is split in two phases so the loop can reuse expensive work between
    neighbour recomputations:

    prepare -- called whenever neighbours are (re)computed; turns ``Neighbors``
        and the density prior into an opaque ``state`` (e.g. a normalised sparse
        weight matrix).
    step -- called every iteration with that ``state``.
    """

    def prepare(self, X, base_weights, neighbors):
        """Return the opaque state used by the following ``step`` calls."""
        raise NotImplementedError

    def step(self, X, state, gate=1.0):
        """Return ``(X_new, change)``; ``change`` is the per-point movement (n,)."""
        raise NotImplementedError
