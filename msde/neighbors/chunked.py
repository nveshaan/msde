import torch

from ..common.knn import torch_knn
from ..common.utils import DEFAULT_DEVICE
from .base import NeighborSelector, Neighbors


class ChunkedKNN(NeighborSelector):
    """Exact chunked top-k neighbours under a distance ``metric``, optionally
    restricted by labels. With ``exclude_self=True`` a point is never its own
    neighbour (the default keeps it, at distance 0, as its first neighbour).
    Subclasses set ``metric`` and may override
    ``transform`` to search in a transformed space."""

    metric = None   # a key of common.knn.METRICS

    def __init__(self, k=30, device=DEFAULT_DEVICE, chunk_size=4096, exclude_self=False):
        super().__init__()
        self.k = k
        self.exclude_self = exclude_self
        self.device_name = device
        self.chunk_size = chunk_size

    def transform(self, X):
        """The points the search runs on (default: ``X`` itself)."""
        return X

    @torch.no_grad()
    def select(self, X, labels=None, corpus_labels=None, base_weights=None):
        k = self.k + 1 if self.exclude_self else self.k
        indices, dist = torch_knn(
            self.transform(X), k, device=self.device_name, chunk_size=self.chunk_size,
            labels=labels, corpus_labels=corpus_labels, metric=self.metric,
        )
        if self.exclude_self:
            indices, dist = _drop_self(indices, dist)

        valid = orphan_mask = None
        if labels is not None:
            valid = corpus_labels[indices] == labels.unsqueeze(1)    # (n, k) bool
            orphan_mask = ~valid.any(dim=1)                           # (n,)

        return Neighbors(indices=indices, dist=dist, valid=valid, orphan_mask=orphan_mask)


def _drop_self(indices, dist):
    """Remove each row's own index from its k+1 neighbours, leaving exactly k.

    A row whose own index is not among its k+1 (possible with many exact
    duplicates, or when the label mask removed it) loses its farthest entry
    instead.
    """
    n = indices.shape[0]
    is_self = indices == torch.arange(n, device=indices.device).unsqueeze(1)
    keep = ~is_self
    keep[~is_self.any(dim=1), -1] = False
    k = indices.shape[1] - 1
    return indices[keep].view(n, k), dist[keep].view(n, k)
