from ..common.utils import ScalarModule


class DensityWeights(ScalarModule):
    """Produces the "how typical/dense is this point" prior: one weight per point."""

    def compute(self, X):
        """Return the (n,) density weights of the rows of ``X``."""
        raise NotImplementedError
