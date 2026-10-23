from ..common.utils import ParametrizedModule


class MovementStep(ParametrizedModule):
    """Moves points towards a weighted barycenter of their neighbours.

    A step is split in two phases so the loop can reuse expensive work between
    neighbour recomputations:

    prepare -- called whenever neighbours are (re)computed; turns ``Neighbors``
        and the density prior into an opaque ``state`` (e.g. a normalised sparse
        weight matrix).
    step -- called every iteration with that ``state``.
    """

    accepts = None  # the Neighbors form this step consumes: "hard" or "soft"

    def prepare(self, X, corpus, base_weights, neighbors):
        """Return the opaque state used by the following ``step`` calls."""
        raise NotImplementedError

    def step(self, X, corpus, state, gate=1.0):
        """Return ``(X_new, change)``; ``change`` is the per-point movement (n,)."""
        raise NotImplementedError
