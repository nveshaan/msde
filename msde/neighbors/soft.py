from ..common.knn import masked_soft_knn_weights
from ..common.utils import DEFAULT_DEVICE, DIFF_SOLVER_ITERATIONS
from .base import NeighborSelector, Neighbors


class SoftKNN(NeighborSelector):
    """Differentiable k-NN: sigmoid memberships over *all* corpus points whose
    total mass per query is ~k (O(n*m) memory)."""

    form = "soft"

    def __init__(self, k=30, temperature=0.1, learn_temperature=False,
                 solver_iterations=DIFF_SOLVER_ITERATIONS, device=DEFAULT_DEVICE):
        super().__init__()
        if temperature is None:
            raise ValueError("SoftKNN requires a positive temperature")
        if solver_iterations < 1:
            raise ValueError("solver_iterations must be a positive integer")
        self.k = k
        self.solver_iterations = solver_iterations
        self.register_scalar("temperature", temperature, learn_temperature, device)

    def select(self, X, corpus, labels=None, corpus_labels=None, exclude_self=False):
        dist, membership, valid = masked_soft_knn_weights(
            X, corpus, self.k, self.temperature,
            labels=labels, corpus_labels=corpus_labels,
            exclude_self=exclude_self, iterations=self.solver_iterations,
        )
        return Neighbors(form="soft", dist=dist, membership=membership, valid=valid)
