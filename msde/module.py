import torch

from .common.labels import LabelPolicy, as_label_tensor
from .common.utils import DEFAULT_DEVICE, configure_logging
from .loop import EarlyStopLoop, LoopContext
from .neighbors import EuclideanKNN
from .step import PlainBarycenter
from .weights import SparseEmpirical


class MeanShiftDensityEnhancement(torch.nn.Module):
    """Mean-shift density enhancement composed of four swappable components.

    neighbors : ``NeighborSelector`` -- which points each point is pulled towards.
    weights   : ``DensityWeights``   -- the per-point density prior.
    step      : ``MovementStep``     -- how a point moves towards the weighted barycenter.
    loop      : ``ShiftLoop``        -- the iteration (schedule, stopping, pre/post-processing).

    Each is optional; the defaults are EuclideanKNN / SparseEmpirical /
    PlainBarycenter / EarlyStopLoop. Nothing here is differentiable: forward()
    runs under ``torch.no_grad()``.

    label_mode : "same_label" (default) or "zero_label" -- see ``LabelPolicy``.
    """

    def __init__(
        self,
        neighbors=None,
        weights=None,
        step=None,
        loop=None,
        label_mode="same_label",
        device=DEFAULT_DEVICE,
        log_file=None,
    ):
        super().__init__()
        self.neighbors = neighbors if neighbors is not None else EuclideanKNN(device=device)
        self.weights = weights if weights is not None else SparseEmpirical(device=device)
        self.step = step if step is not None else PlainBarycenter(device=device)
        self.loop = loop if loop is not None else EarlyStopLoop()

        self.label_policy = LabelPolicy(label_mode)
        self.device_name = device
        self.log_file = log_file
        configure_logging(log_file)

    @property
    def label_mode(self):
        return self.label_policy.mode

    @torch.no_grad()
    def forward(self, X, gate=1.0, labels=None, stability=None):
        """
        Run MSDE and return ``(shifted, total_distance, trajectory)``.

        labels : optional (n,) integer tensor/array-like, aligned with X.
            When given, neighbour search is restricted according to
            ``label_mode``. A point with fewer than k valid neighbours simply
            gets fewer effective neighbours; a point with *no* valid neighbours
            is left unmoved rather than shifted towards a meaningless
            barycenter.
        stability : optional (n,) per-point stability scores in [0, 1] (higher =
            more stable), read by weights strategies with ``uses_stability``
            (``GravityWeights``). Passing it to any other weights strategy is
            an error rather than being silently ignored.
        """
        if stability is not None:
            if not self.weights.uses_stability:
                raise ValueError(
                    f"stability was given, but {type(self.weights).__name__} does not use it; "
                    f"use a weights strategy with uses_stability (e.g. GravityWeights)"
                )
            stability = torch.as_tensor(stability, device=X.device, dtype=X.dtype)
            if stability.dim() != 1 or stability.shape[0] != X.shape[0]:
                raise ValueError(f"stability must be 1D with length n={X.shape[0]}; got shape {tuple(stability.shape)}")
            if not (torch.isfinite(stability).all() and (stability >= 0).all() and (stability <= 1).all()):
                raise ValueError("stability scores must be finite and within [0, 1]")

        query_labels = corpus_labels = None
        if labels is not None:
            corpus_labels = as_label_tensor(labels, X.shape[0], self.device_name)
            query_labels = self.label_policy.query_labels(corpus_labels)

        X = self.loop.preprocess(X)
        ctx = LoopContext(
            neighbors=self.neighbors,
            step=self.step,
            base_weights=self.weights.compute(X, stability=stability),
            query_labels=query_labels,
            corpus_labels=corpus_labels,
            gate=gate,
        )
        return self.loop.run(X, ctx)


mean_shift_density_enhancement = MeanShiftDensityEnhancement
