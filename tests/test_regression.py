"""Component compositions must reproduce the pre-refactor monolith's outputs.

Fixtures come from tests/make_fixtures.py (run against the old src/msde.py).
"""
import pathlib

import pytest
import torch

from msde import (
    EarlyStopLoop, EuclideanKNN, MeanShiftDensityEnhancement, PlainBarycenter, SparseEmpirical,
)

FIXTURES = torch.load(pathlib.Path(__file__).with_name("fixtures.pt"))
DEVICE = "cpu"


def make_data():
    g = torch.Generator().manual_seed(0)
    X = torch.cat([torch.randn(100, 4, generator=g), torch.randn(100, 4, generator=g) + 4.0])
    y = torch.cat([torch.zeros(100), torch.ones(100)]).long()
    return X, y


def hard(iters=4, k=10, knn=None, weights=None, step=None, loop=None, **kw):
    return MeanShiftDensityEnhancement(
        neighbors=EuclideanKNN(k=k, device=DEVICE, **(knn or {})),
        weights=SparseEmpirical(nbd_sample_count_threshold=10, device=DEVICE, **(weights or {})),
        step=PlainBarycenter(device=DEVICE, **(step or {})),
        loop=EarlyStopLoop(max_iters_shift=iters, **(loop or {})),
        device=DEVICE, **kw,
    )


BUILDERS = {
    "plain": lambda: hard(),
    "same_label": lambda: hard(),
    "zero_label": lambda: hard(label_mode="zero_label"),
    "clip": lambda: hard(step=dict(clipping=True, clip_mode=2)),
    "recompute": lambda: hard(loop=dict(recompute_neighbors=2)),
}


@pytest.mark.parametrize("name", list(BUILDERS))
def test_matches_monolith(name):
    X, y = make_data()
    case = FIXTURES[name]
    model = BUILDERS[name]()
    shifted, dist, _ = model(X.clone(), labels=y if case["labels"] else None)
    torch.testing.assert_close(shifted, case["shifted"], atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(dist, case["dist"], atol=1e-5, rtol=1e-4)


def test_forward_is_not_differentiable():
    X, _ = make_data()
    shifted, dist, _ = hard()(X.clone().requires_grad_(True))
    assert not shifted.requires_grad and not dist.requires_grad


def test_new_step_needs_no_edits_elsewhere():
    """A toy variant plugs in by subclassing the base class only."""
    from msde import MovementStep

    class Frozen(MovementStep):
        def prepare(self, X, base_weights, neighbors):
            return None

        def step(self, X, state, gate=1.0):
            return X, torch.zeros(X.shape[0], dtype=X.dtype)

    X, _ = make_data()
    model = hard(step=None)
    model.step = Frozen()
    shifted, dist, _ = model(X.clone())
    torch.testing.assert_close(shifted, X)
    assert dist.sum() == 0


def test_loop_hooks():
    from msde import EarlyStopLoop

    class Doubling(EarlyStopLoop):
        def postprocess(self, result, ctx):
            shifted, dist, traj = result
            return shifted * 2, dist, traj

    X, _ = make_data()
    base, _, _ = hard()(X.clone())
    model = hard()
    model.loop = Doubling(max_iters_shift=4)
    out, _, _ = model(X.clone())
    torch.testing.assert_close(out, base * 2)


def test_cosine_knn_matches_bruteforce():
    from msde import CosineKNN

    X, y = make_data()
    sel = CosineKNN(k=5, device=DEVICE)
    for labels in (None, y):
        nb = sel.select(X, labels=labels, corpus_labels=labels)
        sim = torch.nn.functional.normalize(X, dim=1) @ torch.nn.functional.normalize(X, dim=1).T
        d = 1.0 - sim
        if labels is not None:
            d = d.masked_fill(labels[:, None] != labels[None, :], float("inf"))
        expected = d.topk(5, largest=False, dim=1)
        torch.testing.assert_close(nb.dist, expected.values, atol=1e-5, rtol=1e-4)
        assert (nb.dist >= 0).all()


def test_cosine_knn_ignores_row_scale_and_differs_from_euclidean():
    from msde import CosineKNN

    X, _ = make_data()
    scale = torch.rand(X.shape[0], 1) * 9 + 1
    cos = CosineKNN(k=5, device=DEVICE)
    a = cos.select(X).indices
    b = cos.select(X * scale).indices
    assert (a == b).float().mean() > 0.99          # scale-invariant (up to fp ties)
    eu = EuclideanKNN(k=5, device=DEVICE).select(X * scale).indices
    assert (b != eu).any()


def test_cosine_model_runs_with_labels():
    from msde import CosineKNN

    X, y = make_data()
    model = MeanShiftDensityEnhancement(
        neighbors=CosineKNN(k=10, device=DEVICE),
        weights=SparseEmpirical(nbd_sample_count_threshold=10, device=DEVICE),
        step=PlainBarycenter(device=DEVICE),
        loop=EarlyStopLoop(max_iters_shift=3),
        device=DEVICE,
    )
    shifted, dist, _ = model(X.clone(), labels=y)
    assert shifted.shape == X.shape and torch.isfinite(shifted).all()


def test_unknown_metric_rejected():
    from msde.common.knn import torch_knn

    X, _ = make_data()
    with pytest.raises(ValueError, match="metric"):
        torch_knn(X, 3, device=DEVICE, metric="manhattan")
