import torch

from ..common.utils import DEFAULT_DEVICE, DIFF_EPS
from .base import MovementStep


def _soft_minimum(a, b, temperature):
    """Smooth approximation to min(a,b)."""
    return 0.5 * (
        a + b - torch.sqrt((a - b).square() + temperature**2)
    )


class SoftBarycenter(MovementStep):
    """Differentiable barycenter step over soft neighbour memberships.

    Every corpus point contributes with weight ``membership * base_weights``;
    a point with no mass is left unmoved by a smooth validity gate, and the
    zero-movement gate is smooth. ``temperature`` is only used by
    ``clip_mode=2`` (soft minimum).
    """

    accepts = "soft"

    def __init__(
        self,
        learning_rate=0.3,
        alpha=0.5,
        clipping=False,
        clip_mode=0,
        temperature=0.1,
        learn_learning_rate=False,
        learn_alpha=False,
        learn_temperature=False,
        device=DEFAULT_DEVICE,
    ):
        super().__init__()
        self.clipping = clipping
        self.clip_mode = clip_mode
        dtype = next(
            (v.dtype for v in (learning_rate, alpha, temperature)
             if isinstance(v, torch.Tensor) and v.is_floating_point()),
            torch.get_default_dtype(),
        )
        self.register_scalar("learning_rate", learning_rate, learn_learning_rate, device, dtype)
        self.register_scalar("alpha", alpha, learn_alpha, device, dtype)
        self.register_scalar("temperature", temperature, learn_temperature, device, dtype)

    def prepare(self, X, corpus, base_weights, neighbors):
        # base_weights are row-wise density priors for the corpus; every corpus
        # point participates with its continuous soft-kNN membership weight.
        w = neighbors.membership * base_weights.unsqueeze(0)
        return neighbors.dist, neighbors.membership, w

    def step(self, X, corpus, state, gate=1.0):
        d, membership, w = state
        mass = w.sum(dim=1)
        numerator = w @ corpus
        barycenter = numerator / (mass.unsqueeze(1) + DIFF_EPS)

        # A smooth validity/orphan gate: if mass -> 0, the proposed movement -> 0.
        valid_gate = mass / (mass + DIFF_EPS)
        diff = barycenter - X
        dist_move = torch.sqrt(diff.square().sum(dim=1) + DIFF_EPS**2)

        effective_step = dist_move
        if self.clipping and self.clip_mode > 0:
            neighbor_mean_dist = (membership * d).sum(dim=1) / (membership.sum(dim=1) + DIFF_EPS)
            delta = self.alpha * neighbor_mean_dist
            if self.clip_mode == 1:
                effective_step = dist_move * (delta / (delta + dist_move + DIFF_EPS))
            else:
                effective_step = _soft_minimum(
                    dist_move,
                    delta,
                    torch.sqrt(torch.as_tensor(self.temperature, dtype=X.dtype, device=X.device).square() + DIFF_EPS**2),
                )

        movement_gate = dist_move / (dist_move + DIFF_EPS)
        scale = self.learning_rate * effective_step / (dist_move + DIFF_EPS)
        step = gate * valid_gate.unsqueeze(1) * movement_gate.unsqueeze(1) * scale.unsqueeze(1) * diff
        revised = X + step
        change = valid_gate * movement_gate * dist_move
        return revised, change
