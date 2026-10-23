import torch

from ..common.eps import smooth_epsilon_from_graph
from ..common.graph import soft_fuzzy_graph
from ..common.knn import smooth_pairwise_dist
from ..common.utils import DEFAULT_DEVICE, DIFF_EPS, DIFF_LOG_SCALE_EPS, DIFF_SOLVER_ITERATIONS
from .base import DensityWeights


def _differentiable_empirical_weights(
    X,
    n_neighbors,
    n_epochs,
    nbd_sample_count_threshold,
    satisfiability_proportion,
    max_iters_weight_count,
    temperature,
    eps=None,
    eps_threshold_temperature=None,
    eps_solver_iterations=25,
):
    """Fully differentiable analogue of the empirical density-weight pass."""
    if temperature is None:
        raise ValueError(
            "fully_differentiable=True requires temperature > 0 so all counting "
            "operations remain smooth"
        )
    if eps_threshold_temperature is None:
        eps_threshold_temperature = temperature

    graph = soft_fuzzy_graph(
        X,
        n_neighbors,
        temperature,
        temperature,
        n_epochs,
        knn_iterations=eps_solver_iterations,
    )

    if eps is None:
        eps_tensor, graph_dist, valid_graph = smooth_epsilon_from_graph(
            graph,
            nbd_sample_count_threshold,
            satisfiability_proportion,
            temperature,
            eps_threshold_temperature,
            iterations=eps_solver_iterations,
        )
    else:
        eps_tensor = eps
        graph_dist = smooth_pairwise_dist(graph, graph)
        valid_graph = (
            ~torch.eye(graph.shape[0], dtype=torch.bool, device=graph.device)
        ).to(graph.dtype)

    # Sample the soft count at a fixed sequence of positive radii. Unlike the
    # old implementation there is no discrete indicator and no self subtraction
    # because the fixed diagonal is excluded by valid_graph.
    total_counts = torch.zeros(X.shape[0], dtype=X.dtype, device=X.device)
    steps = max(max_iters_weight_count, 1)
    for i in range(steps):
        frac = (i + 0.5) / steps
        eps_running = eps_tensor * (1.0 - frac) + DIFF_LOG_SCALE_EPS * frac
        counts = (
            torch.sigmoid((eps_running - graph_dist) / (torch.sqrt(torch.as_tensor(temperature, dtype=X.dtype, device=X.device).square() + DIFF_EPS**2)))
            * valid_graph
        ).sum(dim=1)
        total_counts = total_counts + counts

    return total_counts / steps


class SoftEmpirical(DensityWeights):
    """Fully differentiable density weights: soft k-NN graph, smooth eps solve
    and sigmoid counts (O(n^2)). Recomputed on every call, including for a
    reference manifold, so gradients reach ``temperature``/``eps`` and the
    reference tensor."""

    cache_reference = False

    def __init__(
        self,
        n_neighbors=30,
        n_epochs=200,
        nbd_sample_count_threshold=30,
        satisfiability_proportion=0.3,
        max_iters_weight_count=4,
        temperature=0.1,
        eps=None,
        learn_temperature=False,
        learn_eps=False,
        eps_threshold_temperature=None,
        solver_iterations=DIFF_SOLVER_ITERATIONS,
        device=DEFAULT_DEVICE,
    ):
        super().__init__()
        if temperature is None or float(torch.as_tensor(temperature).detach().cpu()) <= 0:
            raise ValueError("SoftEmpirical requires temperature > 0 so all counting operations remain smooth")
        if learn_eps and eps is None:
            raise ValueError("learn_eps=True requires an explicit initial eps")
        if solver_iterations < 1:
            raise ValueError("solver_iterations must be a positive integer")
        self.n_neighbors = n_neighbors
        self.n_epochs = n_epochs
        self.nbd_sample_count_threshold = nbd_sample_count_threshold
        self.satisfiability_proportion = satisfiability_proportion
        self.max_iters_weight_count = max_iters_weight_count
        self.solver_iterations = solver_iterations

        dtype = next(
            (v.dtype for v in (temperature, eps, eps_threshold_temperature)
             if isinstance(v, torch.Tensor) and v.is_floating_point()),
            torch.get_default_dtype(),
        )
        self.register_scalar("temperature", temperature, learn_temperature, device, dtype)
        self.register_scalar("eps", eps, learn_eps, device, dtype)
        self.register_scalar("eps_threshold_temperature", eps_threshold_temperature, False, device, dtype)

    def compute(self, corpus):
        threshold_temperature = self.eps_threshold_temperature
        if threshold_temperature is None:
            threshold_temperature = self.temperature
        return _differentiable_empirical_weights(
            corpus,
            self.n_neighbors,
            self.n_epochs,
            self.nbd_sample_count_threshold,
            self.satisfiability_proportion,
            self.max_iters_weight_count,
            self.temperature,
            eps=self.eps,
            eps_threshold_temperature=threshold_temperature,
            eps_solver_iterations=self.solver_iterations,
        )
