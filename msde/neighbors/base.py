from dataclasses import dataclass
from typing import Optional

import torch

from ..common.utils import ParametrizedModule


@dataclass
class Neighbors:
    """Result of a neighbour-selection pass.

    ``form`` says which fields are populated, and a ``MovementStep`` declares
    which form it consumes:

    hard -- ``indices`` (n, k) int64 corpus rows, ``dist`` (n, k), optional
        ``valid`` (n, k) bool (label mask), ``orphan_mask`` (n,) bool (points
        with no valid neighbour), optional ``weights`` (n, k) soft membership.
    soft -- ``membership`` (n, m) over every corpus point, ``dist`` (n, m),
        ``valid`` (n, m) bool.
    """

    form: str
    dist: Optional[torch.Tensor] = None
    indices: Optional[torch.Tensor] = None
    valid: Optional[torch.Tensor] = None
    orphan_mask: Optional[torch.Tensor] = None
    weights: Optional[torch.Tensor] = None
    membership: Optional[torch.Tensor] = None


class NeighborSelector(ParametrizedModule):
    """Chooses, for every query point, which corpus points it is pulled towards."""

    form = None  # "hard" or "soft": the Neighbors form this selector produces

    def select(self, X, corpus, labels=None, corpus_labels=None, exclude_self=False):
        """Return ``Neighbors`` for the rows of ``X`` against ``corpus``.

        ``labels`` are the query-side labels (already mapped by the label
        policy) and ``corpus_labels`` the corpus-side labels; when ``labels`` is
        given, a point may only reach corpus points with a matching label.
        ``corpus is X`` in self-shift mode. ``exclude_self`` is a hint that
        query i must not be its own neighbour (only meaningful when
        ``corpus is X``); strategies may ignore it.
        """
        raise NotImplementedError
