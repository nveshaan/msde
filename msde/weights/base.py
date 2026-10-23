import torch

from ..common.utils import ParametrizedModule


class DensityWeights(ParametrizedModule):
    """Produces the "how typical/dense is this point" prior: one weight per corpus row."""

    # When True, the weights of a fixed reference manifold are computed once
    # (no-grad, detached) in set_reference() and reused by every forward().
    # When False they are recomputed on every call, so they stay connected to
    # learnable parameters and to a reference tensor that requires grad.
    cache_reference = False

    def __init__(self):
        super().__init__()
        self._ref_weights = None

    def compute(self, corpus):
        """Return the (m,) density weights of the rows of ``corpus``."""
        raise NotImplementedError

    def set_reference(self, X_ref):
        """Set (or clear, with None) the fixed reference manifold."""
        self._ref_weights = None
        if X_ref is not None and self.cache_reference:
            with torch.no_grad():
                self._ref_weights = self.compute(X_ref).detach()

    def weights_for(self, corpus, reference=False):
        if reference and self._ref_weights is not None:
            return self._ref_weights
        return self.compute(corpus)
