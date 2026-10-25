"""Generate regression fixtures from the pre-refactor monolith (src/msde.py).

Historical: fixtures.pt was generated once, at the commit that added it, with
`PYTHONPATH=. uv run python tests/make_fixtures.py`. The monolith has since been
split into the msde/ package (see git history), so this script no longer runs
on the current tree; tests/test_regression.py is what consumes fixtures.pt.
"""
import torch
from src.msde import MeanShiftDensityEnhancement

CASES = {
    "plain": dict(kwargs=dict(k=10, nbd_sample_count_threshold=10, max_iters_shift=4), labels=False),
    "same_label": dict(kwargs=dict(k=10, nbd_sample_count_threshold=10, max_iters_shift=4), labels=True),
    "zero_label": dict(kwargs=dict(k=10, nbd_sample_count_threshold=10, max_iters_shift=4, label_mode="zero_label"), labels=True),
    "clip": dict(kwargs=dict(k=10, nbd_sample_count_threshold=10, max_iters_shift=4, clipping=True, clip_mode=2), labels=False),
    "recompute": dict(kwargs=dict(k=10, nbd_sample_count_threshold=10, max_iters_shift=4, recompute_neighbors=2), labels=False),
}


def make_data():
    g = torch.Generator().manual_seed(0)
    X = torch.cat([torch.randn(100, 4, generator=g), torch.randn(100, 4, generator=g) + 4.0])
    y = torch.cat([torch.zeros(100), torch.ones(100)]).long()
    Xref = torch.randn(150, 4, generator=g) + 1.0
    return X, y, Xref


if __name__ == "__main__":
    X, y, Xref = make_data()
    out = {}
    for name, c in CASES.items():
        kw = dict(c["kwargs"], device="cpu")
        if c.get("ref"):
            kw["X"] = Xref
        m = MeanShiftDensityEnhancement(**kw)
        shifted, dist, _ = m(X.clone(), labels=y if c["labels"] else None)
        out[name] = dict(kwargs=c["kwargs"], ref=bool(c.get("ref")), labels=c["labels"],
                         shifted=shifted.detach(), dist=dist.detach())
        print(name, float(dist.sum()))
    torch.save(out, "tests/fixtures.pt")
