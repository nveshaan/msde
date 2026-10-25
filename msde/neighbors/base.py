from dataclasses import dataclass
from typing import Optional

import torch

from ..common.utils import ScalarModule


@dataclass
class Neighbors:
    """Result of a neighbour-selection pass.

    indices -- (n, k) int64 rows of X selected as neighbours of each point.
    dist -- (n, k) distances to them.
    valid -- (n, k) bool, False where a selected row violates the label mask
        (None when no labels were given).
    orphan_mask -- (n,) bool, True for points with no valid neighbour at all
        (None when no labels were given).
    """

    indices: torch.Tensor
    dist: torch.Tensor
    valid: Optional[torch.Tensor] = None
    orphan_mask: Optional[torch.Tensor] = None


class NeighborSelector(ScalarModule):
    """Chooses, for every point, which other points it is pulled towards."""

    def select(self, X, labels=None, corpus_labels=None, base_weights=None):
        """Return ``Neighbors`` for the rows of ``X`` among the rows of ``X``.

        ``labels`` are the query-side labels (already mapped by the label
        policy) and ``corpus_labels`` the neighbour-side labels; when
        ``labels`` is given, a point may only reach points with a matching label.
        ``base_weights`` is the (n,) per-point attraction factor from the weights
        component; distance-only selectors ignore it.
        """
        raise NotImplementedError
