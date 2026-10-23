import logging
from dataclasses import dataclass
from typing import Optional

import torch

from ..common.utils import ParametrizedModule, logger


@dataclass
class LoopContext:
    """Everything a loop needs besides the points being shifted.

    ``reference`` is the fixed corpus in reference-shift mode (None in
    self-shift mode, where the corpus is the points themselves). Labels are
    None when no masking applies: ``query_labels`` are already mapped by the
    label policy, ``corpus_labels`` are the corpus side's.
    """

    neighbors: torch.nn.Module
    step: torch.nn.Module
    base_weights: torch.Tensor
    reference: Optional[torch.Tensor] = None
    query_labels: Optional[torch.Tensor] = None
    corpus_labels: Optional[torch.Tensor] = None
    gate: float = 1.0


class ShiftLoop(ParametrizedModule):
    """Drives the neighbours -> step iteration.

    ``run`` is the shared skeleton; strategies override the policy hooks
    (``should_recompute``, ``converged``) and the ``preprocess`` /
    ``postprocess`` hooks, which wrap the iteration and default to identity.
    """

    def __init__(self, max_iters_shift=5, recompute_neighbors=0, keep_trajectory=False):
        super().__init__()
        if recompute_neighbors is not None and (
            not isinstance(recompute_neighbors, int) or isinstance(recompute_neighbors, bool)
            or recompute_neighbors < 0
        ):
            raise ValueError(
                "recompute_neighbors must be None, 0 (never recompute after the "
                "first iteration -- fixed-graph behaviour), or a positive int N "
                "(recompute the neighbours every N iterations, i.e. at iter_count "
                "0, N, 2N, ...)."
            )
        self.max_iters_shift = max_iters_shift
        # Normalize None -> 0 so None and 0 both mean "never recompute".
        self.recompute_neighbors = recompute_neighbors or 0
        self.keep_trajectory = keep_trajectory

    def should_recompute(self, iter_count):
        return iter_count == 0 or bool(
            self.recompute_neighbors and iter_count % self.recompute_neighbors == 0
        )

    def converged(self, change):
        return False

    def preprocess(self, X, ctx):
        """Transform the points before shifting (default: identity)."""
        return X

    def postprocess(self, result, ctx):
        """Transform ``(shifted, total_distance, trajectory)`` (default: identity)."""
        return result

    def run(self, X, ctx):
        """Return ``(shifted_dataset, total_distance, trajectory)``."""
        X = self.preprocess(X, ctx)
        reference = ctx.reference
        masked = ctx.query_labels is not None

        shifted = X.clone()
        total_distance = torch.zeros(X.shape[0], dtype=X.dtype, device=X.device)
        trajectory = [shifted.clone()] if self.keep_trajectory else []

        state = None
        orphan_mask = None    # (n,) bool -- points with zero valid neighbours at the last recompute

        for iter_count in range(self.max_iters_shift):
            corpus = reference if reference is not None else shifted

            if self.should_recompute(iter_count):
                neighbors = ctx.neighbors.select(
                    shifted, corpus,
                    labels=ctx.query_labels if masked else None,
                    corpus_labels=ctx.corpus_labels if masked else None,
                    exclude_self=reference is None,
                )
                state = ctx.step.prepare(shifted, corpus, ctx.base_weights, neighbors)
                orphan_mask = neighbors.orphan_mask

            revised, change = ctx.step.step(shifted, corpus, state, ctx.gate)

            if orphan_mask is not None and orphan_mask.any():
                revised = torch.where(orphan_mask.unsqueeze(1), shifted, revised)
                change = torch.where(orphan_mask, torch.zeros_like(change), change)

            total_distance = total_distance + change
            shifted = revised

            if self.keep_trajectory:
                trajectory.append(shifted.clone())

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Iter {iter_count + 1}: mean change = {float(change.detach().mean()):.6f}")

            if self.converged(change):
                logger.info(f"Converged at iteration {iter_count + 1}.")
                break

        return self.postprocess((shifted, total_distance, trajectory), ctx)
