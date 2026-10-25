"""Regenerate tests/fixtures.pt from the current package.

Run after an intentional numerical change: `uv run python tests/make_fixtures.py`
(then review the diff of the outputs before committing the new fixtures).
"""
import warnings

import torch

warnings.filterwarnings("ignore")

from tests.test_regression import BUILDERS, FIXTURES_PATH, make_data

LABELLED = {"same_label", "zero_label"}

if __name__ == "__main__":
    X, y = make_data()
    out = {}
    for name, build in BUILDERS.items():
        labels = y if name in LABELLED else None
        shifted, dist, _ = build()(X.clone(), labels=labels)
        out[name] = dict(labels=labels is not None, shifted=shifted, dist=dist)
        print(name, float(dist.sum()))
    torch.save(out, FIXTURES_PATH)
