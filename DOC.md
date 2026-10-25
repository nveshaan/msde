# MSDE

MSDE (mean-shift density enhancement) computes an empirical density weight for each point, selects neighbours, and repeatedly moves each point toward a weighted neighbour barycenter.

It is built from **four swappable components**, each with a base class and one or more strategies. Everything runs under `torch.no_grad()`; nothing is differentiable.

| Component | Base class | Question it answers | Strategies |
| --- | --- | --- | --- |
| `msde.neighbors` | `NeighborSelector` | Which points is each point pulled towards? | `EuclideanKNN`, `CosineKNN` |
| `msde.weights` | `DensityWeights` | How typical/dense is each point? | `SparseEmpirical` |
| `msde.step` | `MovementStep` | How does a point move towards the weighted barycenter? | `PlainBarycenter` |
| `msde.loop` | `ShiftLoop` | How is the iteration run (schedule, stopping, pre/post-processing)? | `EarlyStopLoop` |

Code shared by more than one component lives in `msde.common` (`knn`, `graph`, `eps`, `labels`, `utils`). Components never import from each other.

```python
import torch
from msde import (
    MeanShiftDensityEnhancement, EuclideanKNN, SparseEmpirical, PlainBarycenter, EarlyStopLoop,
)

X = torch.randn(10_000, 128, device="cuda")

model = MeanShiftDensityEnhancement(
    neighbors=EuclideanKNN(k=30, device="cuda"),
    weights=SparseEmpirical(nbd_sample_count_threshold=30, use_chunking=True, chunk_size=2048, device="cuda"),
    step=PlainBarycenter(learning_rate=0.3, device="cuda"),
    loop=EarlyStopLoop(max_iters_shift=5),
    device="cuda",
)
shifted, movement, trajectory = model(X)
```

Every component is optional; omitted ones default to the combination above. `shifted` has shape `(n, d)`, `movement` has shape `(n,)` (accumulated movement), and `trajectory` is empty unless the loop has `keep_trajectory=True`.

## Notation and cost model

- `n`: points; `d`: feature dimension; `k`: shift neighbours; `q`: `SparseEmpirical.n_neighbors` (default 15); `c`: chunk size; `I_s`: shift iterations; `r`: neighbour rebuilds (`r=1` for `recompute_neighbors=0`, else `<= ceil(I_s/p)`).
- `EuclideanKNN` / `CosineKNN`: `O(n^2*d)` time even when chunked; `O(c*n + n*k)` working space.
- `SparseEmpirical`: the k-NN stage is `O(n^2*d)`; the fuzzy graph has `E ~ 2*n*q` edges. Each graph row is an `n`-dimensional sparse vector, so one chunked radius-count pass costs `O(E*n) = O(n^2*q)` time (a sparse-times-dense matmul per chunk) with `O(c*n)` working space. The pipeline runs about 20 such passes: one min/max pass, a bisection for epsilon (`~log2(range/1e-4)`, at most 50, plus a relaxed retry when it fails) and `max_iters_weight_count` weight passes. Total `O(n^2*(d + P*q))` with `P ~ 20`. The dense variant materialises `S` and calls `cdist` on `n`-dimensional rows: `O(n^3)` time and `O(n^2)` memory.
- `PlainBarycenter`: sparse mode ~`O(n^2*d)`; gather mode `O(n*k*d)`; clipping adds `O(n*k*d)`.
- `torch.compile` is used for dense kernels; the first call for a new shape/configuration includes compilation, so do not benchmark it.

## Components

### Neighbours (`msde.neighbors`)

`select(X, labels, corpus_labels) -> Neighbors`, where `Neighbors` is a dataclass with `indices (n,k)`, `dist (n,k)`, and, when labels are given, `valid (n,k)` and `orphan_mask (n,)`.

| Strategy | Options |
| --- | --- |
| `EuclideanKNN` | `k` (must satisfy `1 <= k <= n`, validated before the compiled `topk`), `chunk_size`, `device`. Ranks by Euclidean distance. |
| `CosineKNN` | Same options. Ranks by cosine distance `1 - cos(x, y)` (in `[0, 2]`), so it ignores each point's scale; `Neighbors.dist` holds cosine distances. Only the neighbour choice changes: density weights and the movement step still work in the original feature space. |

Both share `ChunkedKNN` (`msde.neighbors.chunked`), which only needs a `metric` key from `msde.common.knn.METRICS`; add a metric there and a two-line subclass to get a new one.

### Weights (`msde.weights`)

`compute(X) -> (n,)`.

| Strategy | Options |
| --- | --- |
| `SparseEmpirical` | `nbd_sample_count_threshold`, `n_neighbors` (15), `n_epochs` (200), `max_iters_weight_count` (4), `satisfiability_proportion` (0.3), `use_chunking` / `chunk_size`, `eps` (fixed radius; otherwise found by bisection), `device` |

### Step (`msde.step`)

Two phases so the loop can reuse work between neighbour recomputations: `prepare(X, base_weights, neighbors) -> state` (on every recompute) and `step(X, state, gate) -> (X_new, change)` (every iteration).

| Strategy | Options |
| --- | --- |
| `PlainBarycenter` | `learning_rate`, `alpha`, `clipping`, `clip_mode` (0 off, 1 saturating, 2 hard cap at `alpha` x median neighbour distance), `use_sparse_shift` (sparse matmul vs gather), `low_precision_barycenter`, `device` |

### Loop (`msde.loop`)

`ShiftLoop.run` is the shared skeleton: `preprocess` -> iterate (recompute neighbours / `prepare` on schedule, `step`, orphan freeze, trajectory) -> `postprocess`. Strategies override `should_recompute`, `converged`, `preprocess`, `postprocess`.

| Strategy | Options |
| --- | --- |
| `EarlyStopLoop` | `max_iters_shift`, `shift_threshold` (stop when mean movement is below it), `recompute_neighbors` (`0`/`None`: once; positive `p`: at iterations 0, p, 2p, ...), `keep_trajectory` |

## `MeanShiftDensityEnhancement` options

`neighbors`, `weights`, `step`, `loop` (components), plus:

| Option | Effect |
| --- | --- |
| `labels` (per call) | Integer labels passed to `forward(X, gate=1.0, labels=None)`. |
| `label_mode` | `"same_label"` (default) or `"zero_label"`; see [Label masking](#label-masking). |
| `gate` (per call) | Multiplier on the shift step for that call. |
| `log_file` | Logging is silent when `None`; with a path, this package's messages are written there and propagation to root handlers is disabled. |

## Label masking

Labels passed to `forward(X, labels=y)` restrict which points can be neighbours:

| `label_mode` | Point with label 0 | Point with label 1, 2, 3, ... |
| --- | --- | --- |
| `"same_label"` (default) | label-0 neighbours | neighbours with its own label |
| `"zero_label"` | label-0 neighbours | label-0 neighbours |

```python
model = MeanShiftDensityEnhancement(label_mode="zero_label")
shifted, movement, _ = model(X, labels=y)   # every point shifts towards label-0 points
```

- The k-NN strategies set cross-class distances to `inf` before `topk`. With fewer than `k` eligible points the extras get zero weight; with none, the point is an orphan and the loop leaves it unmoved.
- `k` is validated against the total number of points, not the number eligible under the mask.
- Density weights are not masked; only neighbour eligibility depends on labels.
- `zero_label` needs label-0 points in `X`; otherwise every point is an orphan and nothing moves.

## Writing a new strategy

Subclass the component's base class and override the documented methods; nothing else changes. A step that never moves points:

```python
from msde import MovementStep

class Frozen(MovementStep):
    def prepare(self, X, base_weights, neighbors):
        return None                       # opaque state reused by step()

    def step(self, X, state, gate=1.0):
        return X, torch.zeros(X.shape[0], dtype=X.dtype, device=X.device)
```

Use `self.register_scalar(name, value, device)` (from `ScalarModule`, the shared base) to store a scalar hyperparameter as a buffer. Loop pre/post-processing:

```python
class Doubling(EarlyStopLoop):
    def postprocess(self, result, ctx):
        shifted, dist, trajectory = result
        return shifted * 2, dist, trajectory
```

## Standalone functions

For profiling or custom pipelines: `msde.common.knn` (`torch_knn`, `compute_fixed_knn`), `msde.common.graph` (`fuzzy_simplicial_set_torch`), `msde.common.eps` (`radius_counts_chunked`, `calculate_eps_from_similarity`), and `msde.weights.sparse` (`get_empirical_weights`, `compute_weights_from_similarity_chunked`, `compute_weights_from_similarity_dense`).

## Failure modes and checks

- The k-NN strategies validate `1 <= k <= n` before calling `torch.topk`; in a masked run `n` is still the total number of points.
- Label tensors must be integer-like, 1-D and aligned with the points.
- Benchmark after the first call per shape: compiled kernels compile lazily.
