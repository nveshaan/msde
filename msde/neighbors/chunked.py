import torch

from ..common.knn import torch_knn
from ..common.utils import DEFAULT_DEVICE
from .base import NeighborSelector, Neighbors


class ChunkedKNN(NeighborSelector):
    """Exact chunked top-k neighbours under a distance ``metric``, optionally
    restricted by labels. Subclasses set ``metric`` and may override
    ``transform`` to search in a transformed space."""

    metric = None   # a key of common.knn.METRICS

    def __init__(self, k=30, device=DEFAULT_DEVICE, chunk_size=4096):
        super().__init__()
        self.k = k
        self.device_name = device
        self.chunk_size = chunk_size

    def transform(self, X):
        """The points the search runs on (default: ``X`` itself)."""
        return X

    @torch.no_grad()
    def select(self, X, labels=None, corpus_labels=None):
        indices, dist = torch_knn(
            self.transform(X), self.k, device=self.device_name, chunk_size=self.chunk_size,
            labels=labels, corpus_labels=corpus_labels, metric=self.metric,
        )

        valid = orphan_mask = None
        if labels is not None:
            valid = corpus_labels[indices] == labels.unsqueeze(1)    # (n, k) bool
            orphan_mask = ~valid.any(dim=1)                           # (n,)

        return Neighbors(indices=indices, dist=dist, valid=valid, orphan_mask=orphan_mask)
