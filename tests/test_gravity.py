"""Gravity-based enhancement: each piece checked against the spec's formulas."""
import math

import pytest
import torch

from msde import (
    CosineKNN, EarlyStopLoop, EuclideanKNN, GravityBarycenter, GravityWeights, JointKNN, KDEWeights,
    MeanShiftDensityEnhancement, Neighbors, SparseEmpirical, UniformWeights, WhitenedKNN,
)
from msde.common.kernel import estimate_bandwidth
from tests.test_regression import make_data

DEVICE = "cpu"


def sets(idx):
    return [frozenset(r.tolist()) for r in idx]


def cloud(n=60, d=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g).double()


# ---------------------------------------------------------------- exclude_self
@pytest.mark.parametrize("cls", [EuclideanKNN, CosineKNN, WhitenedKNN])
def test_exclude_self_never_returns_the_point_itself(cls):
    X = cloud().float()
    nb = cls(k=5, exclude_self=True, device=DEVICE).select(X)
    assert nb.indices.shape == (60, 5) and nb.dist.shape == (60, 5)
    assert not (nb.indices == torch.arange(60).unsqueeze(1)).any()
    with_self = cls(k=6, device=DEVICE).select(X)           # self + its 5 real neighbours
    assert sets(nb.indices) == [frozenset(r.tolist()) - {i} for i, r in enumerate(with_self.indices)]


def test_exclude_self_with_many_duplicates_and_labels():
    X = torch.zeros(12, 3)                                  # all identical: self may not be in the top k+1
    X[10:] = 5.0
    nb = EuclideanKNN(k=3, exclude_self=True, device=DEVICE).select(X)
    assert nb.indices.shape == (12, 3)
    assert not (nb.indices == torch.arange(12).unsqueeze(1)).any()
    y = torch.arange(12) % 2
    nb = EuclideanKNN(k=3, exclude_self=True, device=DEVICE).select(X, labels=y, corpus_labels=y)
    assert not (nb.indices == torch.arange(12).unsqueeze(1)).any() and nb.valid.all()


# -------------------------------------------------------------------- JointKNN
def joint_reference(X, q, k, h):
    n = X.shape[0]
    d = torch.cdist(X, X)
    a = q.unsqueeze(0) * torch.exp(-d.square() / (2 * h * h))
    a.fill_diagonal_(-1.0)
    return a.topk(k, dim=1).indices


def test_joint_knn_matches_the_spec_formula():
    X = cloud()
    q = torch.rand(60, dtype=torch.float64) + 0.05
    nb = JointKNN(k=6, h=1.3, device=DEVICE).select(X, base_weights=q)
    assert sets(nb.indices) == sets(joint_reference(X, q, 6, 1.3))
    assert not (nb.indices == torch.arange(60).unsqueeze(1)).any()
    torch.testing.assert_close(nb.dist, torch.cdist(X, X).gather(1, nb.indices))


def test_joint_knn_is_chunk_invariant_and_reduces_to_knn_for_uniform_q():
    X = cloud()
    q = torch.rand(60, dtype=torch.float64) + 0.05
    a = JointKNN(k=6, h=1.3, chunk_size=7, device=DEVICE).select(X, base_weights=q)
    b = JointKNN(k=6, h=1.3, chunk_size=1000, device=DEVICE).select(X, base_weights=q)
    assert sets(a.indices) == sets(b.indices)
    uniform = JointKNN(k=6, h=1.3, device=DEVICE).select(X)           # q = 1: the kernel is monotone in d
    plain = EuclideanKNN(k=6, exclude_self=True, device=DEVICE).select(X)
    assert sets(uniform.indices) == sets(plain.indices)


def test_a_farther_stable_point_can_outrank_a_closer_unstable_one():
    # point 0 at the origin; candidate 1 is close but has no attraction, candidate 2 is farther but attractive
    X = torch.tensor([[0.0], [1.0], [2.0], [50.0]])
    q = torch.tensor([1.0, 0.01, 1.0, 1.0])
    joint = JointKNN(k=1, h=2.0, device=DEVICE).select(X, base_weights=q)
    assert joint.indices[0, 0].item() == 2
    plain = EuclideanKNN(k=1, exclude_self=True, device=DEVICE).select(X)      # distance only: Approach 1 selection
    assert plain.indices[0, 0].item() == 1


def test_joint_knn_zero_score_fallbacks_to_nearest():
    X = cloud()
    nearest = EuclideanKNN(k=6, exclude_self=True, device=DEVICE).select(X)
    all_zero = JointKNN(k=6, h=1.3, device=DEVICE).select(X, base_weights=torch.zeros(60, dtype=torch.float64))
    assert sets(all_zero.indices) == sets(nearest.indices)
    # only 2 attractive candidates: they come first, the rest are the nearest remaining
    q = torch.zeros(60, dtype=torch.float64)
    q[[7, 8]] = 1.0
    nb = JointKNN(k=6, h=100.0, device=DEVICE).select(X, base_weights=q)
    for i in range(60):
        row = set(nb.indices[i].tolist())
        assert {7, 8} - {i} <= row
        rest = row - {7, 8}
        d = torch.cdist(X[i:i + 1], X)[0]
        d[[i, 7, 8]] = float("inf")
        assert rest == set(d.topk(len(rest), largest=False).indices.tolist())


def test_joint_knn_labels_cosine_and_validation():
    X, y = make_data()
    nb = JointKNN(k=5, h="auto", metric="cosine", device=DEVICE).select(X, labels=y, corpus_labels=y)
    assert nb.valid.all() and (y[nb.indices] == y.unsqueeze(1)).all()
    assert not (nb.indices == torch.arange(X.shape[0]).unsqueeze(1)).any()
    # zero_label mapping: query labels differ from the neighbour-side labels
    nb = JointKNN(k=5, h=1.0, device=DEVICE).select(X, labels=torch.zeros_like(y), corpus_labels=y)
    assert (y[nb.indices] == 0).all()
    for bad in (dict(h="x"), dict(h=0), dict(h=-1.0), dict(metric="l1"), dict(chunk_size=0)):
        with pytest.raises(ValueError):
            JointKNN(device=DEVICE, **bad)
    with pytest.raises(ValueError, match="k="):
        JointKNN(k=60, device=DEVICE).select(cloud())


def test_estimate_bandwidth_is_the_median_knn_distance():
    X = cloud()
    d = torch.cdist(X, X)
    d.fill_diagonal_(float("inf"))
    expected = d.topk(4, largest=False).values.median().item()
    assert abs(estimate_bandwidth(X, 4) - expected) < 1e-9


# ------------------------------------------------------------ GravityBarycenter
def step_reference(X, idx, q, h, eta, eps, metric="euclidean"):
    out = X.clone()
    for i in range(X.shape[0]):
        Xj = X[idx[i]]
        if metric == "euclidean":
            d = (Xj - X[i]).norm(dim=1)
        else:
            d = 1 - (Xj @ X[i]) / (Xj.norm(dim=1) * X[i].norm())
        w = q[idx[i]] * torch.exp(-d.square() / (2 * h * h))
        mu = (w.unsqueeze(1) * Xj).sum(0) / (w.sum() + eps)
        if w.sum() <= eps:
            mu = X[i]
        out[i] = X[i] + eta * (mu - X[i])
    return out


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_gravity_step_matches_the_spec_formula(metric):
    X = cloud()
    q = torch.rand(60, dtype=torch.float64)
    idx = EuclideanKNN(k=6, exclude_self=True, device=DEVICE).select(X).indices
    step = GravityBarycenter(eta=0.3, h=1.1, metric=metric, device=DEVICE)
    state = step.prepare(X, q, Neighbors(indices=idx, dist=torch.zeros(60, 6)))
    new, change = step.step(X, state, gate=1.0)
    torch.testing.assert_close(new, step_reference(X, idx, q, 1.1, step.eta.double().item(), 1e-8, metric))
    torch.testing.assert_close(change, (new - X).norm(dim=1))
    half, _ = step.step(X, state, gate=0.5)                       # gate scales eta
    torch.testing.assert_close(half - X, 0.5 * (new - X))


def test_gravity_step_zero_weight_points_stay_and_updates_use_a_snapshot():
    X = cloud()
    idx = EuclideanKNN(k=6, exclude_self=True, device=DEVICE).select(X).indices
    step = GravityBarycenter(eta=0.5, h=1.0, device=DEVICE)
    q = torch.ones(60, dtype=torch.float64)
    q[idx[3]] = 0.0                                               # nothing attracts point 3
    new, _ = step.step(X, step.prepare(X, q, Neighbors(indices=idx, dist=None)))
    torch.testing.assert_close(new[3], X[3])
    assert (new[4] != X[4]).any()
    # snapshot semantics: permuting the points permutes the result identically
    perm = torch.randperm(60, generator=torch.Generator().manual_seed(2))
    inv = perm.argsort()
    Xp = X[perm]
    idxp = inv[idx[perm]]                                         # the same graph, relabelled
    newp, _ = step.step(Xp, step.prepare(Xp, q[perm], Neighbors(indices=idxp, dist=None)))
    torch.testing.assert_close(newp, new[perm])


def test_gravity_step_pulls_towards_the_attractive_neighbour():
    X = torch.tensor([[0.0], [-1.0], [1.0]])
    idx = torch.tensor([[1, 2], [0, 2], [0, 1]])
    step = GravityBarycenter(eta=1.0, h=10.0, device=DEVICE)
    for q, side in ((torch.tensor([1.0, 0.0, 1.0]), 1), (torch.tensor([1.0, 1.0, 0.0]), -1)):
        new, _ = step.step(X, step.prepare(X, q, Neighbors(indices=idx, dist=None)))
        assert new[0, 0].sign().item() == side                    # point 0 moves to the attractive side


def test_gravity_step_validation():
    for bad in (dict(eta=0), dict(eta=1.5), dict(h=0), dict(h="x"), dict(metric="l1"), dict(eps=0)):
        with pytest.raises(ValueError):
            GravityBarycenter(device=DEVICE, **bad)


# --------------------------------------------------------------- GravityWeights
def test_gravity_weights_formula_normalisations_and_floors():
    X = cloud().float()
    rho = KDEWeights(1.0, device=DEVICE).compute(X)
    s = torch.rand(60)
    for mode, expect in (
        ("none", rho),
        ("max", rho / rho.max()),
        ("minmax", (rho - rho.min()) / (rho.max() - rho.min())),
        ("rank", (rho.argsort().argsort() + 1).float() / 60),
    ):
        gw = GravityWeights(KDEWeights(1.0, device=DEVICE), alpha=1.5, beta=0.5,
                            density_normalization=mode, device=DEVICE)
        torch.testing.assert_close(gw.compute(X, stability=s), expect.clamp_min(1e-6).pow(1.5) * s.pow(0.5))
    gw = GravityWeights(KDEWeights(1.0, device=DEVICE), density_normalization="none", device=DEVICE)
    torch.testing.assert_close(gw.compute(X), rho.clamp_min(1e-6))                 # no stability: s = 1
    zero = torch.zeros(60)
    assert (gw.compute(X, stability=zero) == 0).all()                              # zero stability removes attraction
    floored = GravityWeights(KDEWeights(1.0, device=DEVICE), density_normalization="none",
                             stability_floor=0.2, device=DEVICE)
    assert (floored.compute(X, stability=zero) > 0).all()


def test_gravity_weights_validation_and_uniform():
    with pytest.raises(ValueError, match="density"):
        GravityWeights(object(), device=DEVICE)
    with pytest.raises(ValueError, match="density"):
        GravityWeights(GravityWeights(KDEWeights(1.0, device=DEVICE), device=DEVICE), device=DEVICE)
    for bad in (dict(density_normalization="x"), dict(density_floor=0), dict(stability_floor=-0.1),
                dict(stability_floor=1.5)):
        with pytest.raises(ValueError):
            GravityWeights(KDEWeights(1.0, device=DEVICE), device=DEVICE, **bad)
    assert (UniformWeights().compute(cloud()) == 1).all()


# ----------------------------------------------------- module + ablation study
def gravity_model(selection, weights, **kw):
    return MeanShiftDensityEnhancement(
        neighbors=selection, weights=weights, step=GravityBarycenter(eta=0.2, h="auto", device=DEVICE),
        loop=EarlyStopLoop(max_iters_shift=2, **kw), device=DEVICE,
    )


def test_stability_argument_validation():
    X, y = make_data()
    model = MeanShiftDensityEnhancement(weights=KDEWeights(1.0, device=DEVICE), device=DEVICE)
    with pytest.raises(ValueError, match="does not use it"):
        model(X, stability=torch.ones(X.shape[0]))
    gw = GravityWeights(KDEWeights(1.0, device=DEVICE), device=DEVICE)
    model = gravity_model(EuclideanKNN(k=5, exclude_self=True, device=DEVICE), gw)
    with pytest.raises(ValueError, match="length"):
        model(X, stability=torch.ones(3))
    for bad in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError, match="within"):
            model(X, stability=torch.full((X.shape[0],), bad))
    shifted, _, _ = model(X.clone(), stability=torch.rand(X.shape[0]))
    assert torch.isfinite(shifted).all()


def test_three_way_ablation_runs_and_differs():
    X, y = make_data()
    s = torch.rand(X.shape[0], generator=torch.Generator().manual_seed(4))
    density = lambda: KDEWeights(1.0, device=DEVICE)
    runs = {
        "1 distance only": gravity_model(EuclideanKNN(k=8, exclude_self=True, device=DEVICE), UniformWeights()),
        "2 approach 1": gravity_model(EuclideanKNN(k=8, exclude_self=True, device=DEVICE),
                                      GravityWeights(density(), device=DEVICE)),
        "3 approach 2": gravity_model(JointKNN(k=8, h="auto", device=DEVICE),
                                      GravityWeights(density(), device=DEVICE)),
    }
    out = {name: m(X.clone(), labels=y, stability=s)[0] if "1" not in name else m(X.clone(), labels=y)[0]
           for name, m in runs.items()}
    for name, o in out.items():
        assert o.shape == X.shape and torch.isfinite(o).all() and (o != X).any(), name
    names = list(out)
    for a in range(3):
        for b in range(a + 1, 3):
            assert (out[names[a]] - out[names[b]]).abs().max() > 1e-4, (names[a], names[b])


def test_stability_steers_the_shift():
    # a line of points; making the right half stable and the left half unstable pulls the middle rightwards
    X = torch.linspace(-3, 3, 41).unsqueeze(1)
    mid = 20
    def moved(stability):
        model = gravity_model(EuclideanKNN(k=6, exclude_self=True, device=DEVICE),
                              GravityWeights(UniformWeights(), beta=2.0, device=DEVICE))
        return model(X.clone(), stability=stability)[0][mid, 0] - X[mid, 0]
    right = torch.where(X[:, 0] > 0, 1.0, 0.0)
    assert moved(right) > 0 > moved(1 - right)
    assert abs(moved(torch.ones(41))) < 1e-6                      # symmetric neighbourhood: no net pull
