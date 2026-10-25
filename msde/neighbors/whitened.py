import torch

from ..common.utils import DEFAULT_DEVICE
from .chunked import ChunkedKNN


class WhitenedKNN(ChunkedKNN):
    """Exact top-k neighbours in whitened space, i.e. by Mahalanobis distance.

    The points are centred and rescaled by the inverse square root of their
    covariance, so every direction has unit variance and no correlations
    remain; the search is then Euclidean. The ranking is invariant to any
    invertible linear transform of the features (rescaling or mixing columns
    does not change the neighbours), and ``Neighbors.dist`` holds Mahalanobis
    distances.

    The covariance is estimated from the points being searched, once per
    ``select`` call (so it follows the points as they shift), over all points
    regardless of labels. Only the neighbour choice changes: the density
    weights and the movement step still work in the original feature space.

    regularization : ridge added to the covariance eigenvalues, relative to
        their mean. It keeps near-degenerate directions (fewer points than
        features, or constant features) from being blown up. The ridge is not
        invariant to a change of basis, so the invariance above is exact only
        while every eigenvalue is far above ``regularization * mean(eigenvalue)``.
        For very ill-conditioned data (eigenvalues spanning ~1e6) lower it,
        e.g. 1e-9, in exchange for less protection on degenerate directions.
        0 is allowed for full-rank data.
    """

    metric = "euclidean"

    def __init__(self, k=30, regularization=1e-6, device=DEFAULT_DEVICE, chunk_size=4096, exclude_self=False):
        super().__init__(k=k, device=device, chunk_size=chunk_size, exclude_self=exclude_self)
        if not regularization >= 0:
            raise ValueError(f"regularization must be >= 0; got {regularization!r}")
        self.regularization = regularization

    @torch.no_grad()
    def transform(self, X):
        n = X.shape[0]
        Xc = X - X.mean(dim=0, keepdim=True)
        cov = (Xc.T @ Xc) / max(n - 1, 1)
        # The d x d eigendecomposition is tiny; do it on the CPU in float64
        # (not every device implements eigh, and float32 loses small eigenvalues).
        # Move first, then upcast: MPS cannot convert to float64, not even on the way to the CPU.
        evals, evecs = torch.linalg.eigh(cov.cpu().double())
        evals = evals.clamp_min(0.0)
        ridge = self.regularization * evals.mean()
        scale = (evals + ridge).clamp_min(torch.finfo(torch.float64).tiny).rsqrt()
        W = (evecs * scale).to(X.dtype).to(X.device)                 # PCA rotation, then unit-variance scaling (cast before moving, for MPS)
        return Xc @ W
