import torch

from ..common.knn import SOFT_TOPK_WEIGHT_KERNEL, torch_knn
from ..common.utils import DEFAULT_DEVICE
from .base import NeighborSelector, Neighbors


class HardKNN(NeighborSelector):
    """Exact top-k neighbours (non-differentiable index selection).

    soft_topk_temperature : if given, each selected neighbour is additionally
        weighted by a sigmoid of its distance relative to the k-th neighbour's;
        this carries gradients to the query/corpus positions and, if
        ``learn_soft_topk_temperature``, to the temperature itself.
    """

    form = "hard"

    def __init__(self, k=30, soft_topk_temperature=None, learn_soft_topk_temperature=False,
                 device=DEFAULT_DEVICE, chunk_size=4096):
        super().__init__()
        if learn_soft_topk_temperature and soft_topk_temperature is None:
            raise ValueError("learn_soft_topk_temperature=True requires an explicit soft_topk_temperature")
        self.k = k
        self.device_name = device
        self.chunk_size = chunk_size
        self.register_scalar(
            "soft_topk_temperature", soft_topk_temperature, learn_soft_topk_temperature, device,
        )

    def select(self, X, corpus, labels=None, corpus_labels=None, exclude_self=False):
        self_mode = corpus is X
        with torch.no_grad():
            indices, dist = torch_knn(
                X.detach(), self.k, device=self.device_name, chunk_size=self.chunk_size,
                corpus=None if self_mode else corpus.detach(),
                labels=labels, corpus_labels=corpus_labels,
            )

        weights = None
        if self.soft_topk_temperature is not None:
            weights = SOFT_TOPK_WEIGHT_KERNEL(X, corpus, indices, self.soft_topk_temperature)

        valid = orphan_mask = None
        if labels is not None:
            valid = corpus_labels[indices] == labels.unsqueeze(1)    # (n, k) bool
            orphan_mask = ~valid.any(dim=1)                           # (n,)

        return Neighbors(
            form="hard", dist=dist, indices=indices, valid=valid,
            orphan_mask=orphan_mask, weights=weights,
        )
