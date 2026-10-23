"""Component compositions must reproduce the pre-refactor monolith's outputs.

Fixtures come from tests/make_fixtures.py (run against the old src/msde.py).
"""
import pathlib

import pytest
import torch

from msde import (
    EarlyStopLoop, FixedLoop, HardKNN, MeanShiftDensityEnhancement, PlainBarycenter,
    SoftBarycenter, SoftEmpirical, SoftKNN, SparseEmpirical,
)

FIXTURES = torch.load(pathlib.Path(__file__).with_name("fixtures.pt"))
DEVICE = "cpu"


def make_data():
    g = torch.Generator().manual_seed(0)
    X = torch.cat([torch.randn(100, 4, generator=g), torch.randn(100, 4, generator=g) + 4.0])
    y = torch.cat([torch.zeros(100), torch.ones(100)]).long()
    Xref = torch.randn(150, 4, generator=g) + 1.0
    return X, y, Xref


def hard(iters=4, k=10, knn=None, weights=None, step=None, loop=None, **kw):
    return MeanShiftDensityEnhancement(
        neighbors=HardKNN(k=k, device=DEVICE, **(knn or {})),
        weights=SparseEmpirical(nbd_sample_count_threshold=10, device=DEVICE, **(weights or {})),
        step=PlainBarycenter(device=DEVICE, **(step or {})),
        loop=EarlyStopLoop(max_iters_shift=iters, **(loop or {})),
        device=DEVICE, **kw,
    )


def soft(**kw):
    return MeanShiftDensityEnhancement(
        neighbors=SoftKNN(k=10, temperature=0.3, device=DEVICE),
        weights=SoftEmpirical(n_neighbors=10, nbd_sample_count_threshold=10, temperature=0.3, device=DEVICE),
        step=SoftBarycenter(temperature=0.3, device=DEVICE),
        loop=FixedLoop(max_iters_shift=3),
        device=DEVICE, **kw,
    )


BUILDERS = {
    "plain": lambda ref: hard(),
    "same_label": lambda ref: hard(),
    "zero_label": lambda ref: hard(label_mode="zero_label"),
    "reference": lambda ref: hard(X=ref),
    "clip": lambda ref: hard(step=dict(clipping=True, clip_mode=2)),
    "recompute": lambda ref: hard(loop=dict(recompute_neighbors=2)),
    "soft_topk": lambda ref: hard(
        knn=dict(soft_topk_temperature=0.5), weights=dict(temperature=0.5)),
    "full_diff": lambda ref: soft(),
    "full_diff_labels": lambda ref: soft(),
}


@pytest.mark.parametrize("name", list(FIXTURES))
def test_matches_monolith(name):
    X, y, Xref = make_data()
    case = FIXTURES[name]
    model = BUILDERS[name](Xref)
    shifted, dist, _ = model(X.clone(), labels=y if case["labels"] else None)
    atol = 1e-4 if name.startswith("full_diff") else 1e-5
    torch.testing.assert_close(shifted.detach(), case["shifted"], atol=atol, rtol=1e-4)
    torch.testing.assert_close(dist.detach(), case["dist"], atol=atol, rtol=1e-4)


def test_soft_components_carry_gradients():
    X, y, _ = make_data()
    model = MeanShiftDensityEnhancement(
        neighbors=SoftKNN(k=10, temperature=0.3, learn_temperature=True, device=DEVICE),
        weights=SoftEmpirical(n_neighbors=10, nbd_sample_count_threshold=10, temperature=0.3,
                              eps=1.0, learn_eps=True, device=DEVICE),
        step=SoftBarycenter(temperature=0.3, device=DEVICE),
        loop=FixedLoop(max_iters_shift=2),
        device=DEVICE,
    )
    assert set(model.learnable_parameters) == {"neighbors.temperature", "weights.eps"}
    X = X.clone().requires_grad_(True)
    shifted, _, _ = model(X)
    shifted.square().sum().backward()
    assert X.grad is not None
    for name, p in model.learnable_parameters.items():
        assert p.grad is not None, name


def test_mismatched_components_rejected():
    with pytest.raises(ValueError, match="consumes"):
        MeanShiftDensityEnhancement(
            neighbors=SoftKNN(k=10, device=DEVICE), step=PlainBarycenter(device=DEVICE), device=DEVICE,
        )


def test_new_step_needs_no_edits_elsewhere():
    """A toy variant plugs in by subclassing the base class only."""
    from msde import MovementStep

    class Frozen(MovementStep):
        accepts = "hard"

        def prepare(self, X, corpus, base_weights, neighbors):
            return None

        def step(self, X, corpus, state, gate=1.0):
            return X, torch.zeros(X.shape[0], dtype=X.dtype)

    X, _, _ = make_data()
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

    X, _, _ = make_data()
    base, _, _ = hard()(X.clone())
    model = hard()
    model.loop = Doubling(max_iters_shift=4)
    out, _, _ = model(X.clone())
    torch.testing.assert_close(out, base * 2)
