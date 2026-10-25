from ..common.utils import ScalarModule


class DensityWeights(ScalarModule):
    """Produces the "how typical/dense is this point" prior: one weight per point."""

    # Whether compute() uses the per-point stability scores passed to forward().
    uses_stability = False

    def compute(self, X, stability=None):
        """Return the (n,) per-point weights of the rows of ``X``.

        ``stability`` is the optional (n,) stability score in ``[0, 1]`` given
        to forward(); only strategies with ``uses_stability = True`` read it.
        """
        raise NotImplementedError
