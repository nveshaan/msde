import torch

from .common.labels import LabelPolicy, as_label_tensor
from .common.utils import DEFAULT_DEVICE, configure_logging
from .loop import EarlyStopLoop, LoopContext
from .neighbors import HardKNN
from .step import PlainBarycenter
from .weights import SparseEmpirical


class MeanShiftDensityEnhancement(torch.nn.Module):
    """Mean-shift density enhancement composed of four swappable components.

    neighbors : ``NeighborSelector`` -- which corpus points each point is pulled towards.
    weights   : ``DensityWeights``   -- the per-corpus-point density prior.
    step      : ``MovementStep``     -- how a point moves towards the weighted barycenter.
    loop      : ``ShiftLoop``        -- the iteration (schedule, stopping, pre/post-processing).

    Each is optional; the defaults are the hard-neighbour / sparse-density /
    plain-barycenter / early-stopping combination. A component may be
    non-differentiable; ``neighbors.form`` must match ``step.accepts``.

    X : optional (m, d) tensor -- a fixed reference manifold. When given,
        forward() shifts its (different) input against neighbours found in
        this X, which stays fixed. When omitted, forward() shifts its input
        against itself. Detached unless ``reference_grad=True``. Can be set or
        changed later via ``set_reference_manifold``.
    labels : optional (m,) integer labels aligned with X (requires X). Lets
        forward() restrict neighbour search per class -- see forward().
    label_mode : "same_label" (default) or "zero_label" -- see ``LabelPolicy``.
    reference_grad : keep autograd history on the reference manifold X.
    enable_gradients : when False, forward() detaches its input and all
        existing component parameters are frozen.
    """

    def __init__(
        self,
        neighbors=None,
        weights=None,
        step=None,
        loop=None,
        X=None,
        labels=None,
        label_mode="same_label",
        device=DEFAULT_DEVICE,
        enable_gradients=True,
        reference_grad=False,
        log_file=None,
    ):
        super().__init__()
        self.neighbors = neighbors if neighbors is not None else HardKNN(device=device)
        self.weights = weights if weights is not None else SparseEmpirical(device=device)
        self.step = step if step is not None else PlainBarycenter(device=device)
        self.loop = loop if loop is not None else EarlyStopLoop()

        if self.neighbors.form != self.step.accepts:
            raise ValueError(
                f"{type(self.neighbors).__name__} produces {self.neighbors.form!r} neighbours "
                f"but {type(self.step).__name__} consumes {self.step.accepts!r} neighbours"
            )

        self.label_policy = LabelPolicy(label_mode)
        self.device_name = device
        self.enable_gradients = enable_gradients
        self.reference_grad = reference_grad
        self.log_file = log_file

        if not enable_gradients:
            for parameter in self.parameters():
                parameter.requires_grad_(False)

        self.set_reference_manifold(X, labels=labels)
        configure_logging(log_file)

    @property
    def label_mode(self):
        return self.label_policy.mode

    @property
    def learnable_parameters(self):
        """Learnable scalars of all components, keyed ``"<component>.<name>"``."""
        return {
            f"{component_name}.{name}": parameter
            for component_name in ("neighbors", "weights", "step", "loop")
            for name, parameter in getattr(self, component_name).learnable_parameters.items()
        }

    def _set_ref_labels(self, labels, n_ref):
        """Validate and (re)register self._ref_labels. `labels=None` clears it."""
        self._buffers.pop("_ref_labels", None)
        if labels is not None:
            labels = as_label_tensor(labels, n_ref, self.device_name, what="the reference manifold")
        self.register_buffer("_ref_labels", labels)

    def set_reference_manifold(self, X, labels=None):
        """
        Set, replace, clear, or (re)label the fixed reference manifold used
        by forward() -- outside of and after __init__.

        X : (n_ref, d) tensor/array-like, or None.
            If not None, becomes the new self.X_ref: forward()'s future
            calls will shift their (different) input against neighbours
            found in this fixed manifold instead of against themselves.
            ``weights.set_reference`` is told, so a caching strategy computes
            the manifold's density weights once here.
            If None and `labels` is also None, clears any existing
            reference manifold (and any stored labels).
            If None but `labels` is given, X_ref itself is left exactly as
            it is -- only its labels are set/replaced. This requires a
            reference manifold to already be set.
        labels : optional (n_ref,) integer tensor/array-like, aligned with
            X (or with the existing self.X_ref, if X is None here).
        """
        if X is not None:
            X_ref = torch.as_tensor(X, device=self.device_name)
            if not self.reference_grad:
                X_ref = X_ref.detach()
            if X_ref.dim() != 2:
                raise ValueError(
                    f"X (reference manifold) must be 2D (n_ref, d); got shape {tuple(X_ref.shape)}"
                )
            self._buffers.pop("X_ref", None)
            self.register_buffer("X_ref", X_ref)
            self.weights.set_reference(self.X_ref)
            self._set_ref_labels(labels, X_ref.shape[0])
        elif labels is not None:
            if getattr(self, "X_ref", None) is None:
                raise ValueError(
                    "set_reference_manifold(X=None, labels=...) updates labels on an "
                    "existing reference manifold, but none is set. Call "
                    "set_reference_manifold(X, labels=...) (or pass X at __init__) first."
                )
            self._set_ref_labels(labels, self.X_ref.shape[0])
        else:
            self._buffers.pop("X_ref", None)
            self.register_buffer("X_ref", None)
            self._set_ref_labels(None, 0)
            self.weights.set_reference(None)

    def forward(self, X, gate=1.0, labels=None):
        """
        Run MSDE and return ``(shifted, total_distance, trajectory)``.

        If a reference manifold is set, every point of this X is shifted with
        respect to neighbours found in that fixed manifold (self.X_ref never
        moves). Otherwise X is shifted with respect to itself.

        labels : optional (n,) integer tensor/array-like, aligned with X.
            When given, neighbour search is restricted per ``label_mode``
            (found in self.X_ref if a reference manifold is set, otherwise
            in X itself).
            - Self-shift mode: pass labels here only.
            - Reference-shift mode: if the reference manifold has labels,
              passing labels here is optional -- omit them to do an unmasked
              shift for this call. If the reference manifold does NOT have
              labels, passing labels here is an error.
            A point with fewer than k valid neighbours simply gets fewer
            effective neighbours; a point with *no* valid neighbours is left
            unmoved rather than shifted towards a meaningless barycenter.
        """
        if not self.enable_gradients:
            X = X.detach()

        reference_mode = self.X_ref is not None

        if reference_mode:
            if X.dim() != 2 or X.shape[1] != self.X_ref.shape[1]:
                raise ValueError(
                    f"X passed to forward() has feature dim {tuple(X.shape[1:])}, "
                    f"which doesn't match the reference manifold's feature dim "
                    f"{tuple(self.X_ref.shape[1:])}. Reference-shift mode requires "
                    f"both to live in the same feature space; point count and "
                    f"order may still differ freely."
                )
            if labels is not None and self._ref_labels is None:
                raise ValueError(
                    "labels was passed to forward(), but the reference manifold has no "
                    "labels (none were given to __init__ or set_reference_manifold()). "
                    "There's nothing on the reference side to mask against -- either call "
                    "set_reference_manifold(self.X_ref, labels=...) to label the reference "
                    "manifold, or drop labels= from this forward() call."
                )

        query_labels = corpus_labels = None
        if labels is not None:
            labels = as_label_tensor(labels, X.shape[0], self.device_name)
            query_labels = self.label_policy.query_labels(labels)
            # Self-shift: the corpus is X itself, so its labels are `labels`.
            corpus_labels = self._ref_labels if reference_mode else labels

        corpus = self.X_ref if reference_mode else X
        ctx = LoopContext(
            neighbors=self.neighbors,
            step=self.step,
            base_weights=self.weights.weights_for(corpus, reference=reference_mode),
            reference=self.X_ref if reference_mode else None,
            query_labels=query_labels,
            corpus_labels=corpus_labels,
            gate=gate,
        )
        return self.loop.run(X, ctx)


mean_shift_density_enhancement = MeanShiftDensityEnhancement
