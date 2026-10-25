# MSDE

MSDE (mean-shift density enhancement) computes an empirical density weight for each point, selects neighbours, and repeatedly moves each point toward a weighted neighbour barycenter.

It is built from **four swappable components**, each with a base class and one or more strategies. Everything runs under `torch.no_grad()`; nothing is differentiable.

| Component | Base class | Question it answers | Strategies |
| --- | --- | --- | --- |
| `msde.neighbors` | `NeighborSelector` | Which points is each point pulled towards? | `EuclideanKNN`, `CosineKNN`, `WhitenedKNN`, `JointKNN` |
| `msde.weights` | `DensityWeights` | How typical/dense is each point? | `SparseEmpirical`, `KDEWeights`, `GravityWeights`, `UniformWeights` |
| `msde.step` | `MovementStep` | How does a point move towards the weighted barycenter? | `PlainBarycenter`, `GravityBarycenter` |
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
- `EuclideanKNN` / `CosineKNN` / `WhitenedKNN`: `O(n^2*d)` time even when chunked; `O(c*n + n*k)` working space. `WhitenedKNN` adds `O(n*d^2 + d^3)` for the covariance and its eigendecomposition.
- `SparseEmpirical`: the k-NN stage is `O(n^2*d)`; the fuzzy graph has `E ~ 2*n*q` edges. Each graph row is an `n`-dimensional sparse vector, so one chunked radius-count pass costs `O(E*n) = O(n^2*q)` time (a sparse-times-dense matmul per chunk) with `O(c*n)` working space. The pipeline runs about 20 such passes: one min/max pass, a bisection for epsilon (`~log2(range/1e-4)`, at most 50, plus a relaxed retry when it fails) and `max_iters_weight_count` weight passes. Total `O(n^2*(d + P*q))` with `P ~ 20`. The dense variant materialises `S` and calls `cdist` on `n`-dimensional rows: `O(n^3)` time and `O(n^2)` memory.
- `JointKNN`: `O(n^2*d)` time (every point is a candidate) and `O(c*n)` working space; with `h="auto"` it adds one k-NN pass to estimate the bandwidth.
- `KDEWeights`: `O(n^2*d)` time and `O(c*n)` working space, i.e. one pairwise-distance pass. This is the cost of `SparseEmpirical`'s k-NN stage alone, without its `O(P*n^2*q)` graph-space counting.
- `PlainBarycenter`: sparse mode ~`O(n^2*d)`; gather mode `O(n*k*d)`; clipping adds `O(n*k*d)`.
- `torch.compile` is used for dense kernels; the first call for a new shape/configuration includes compilation, so do not benchmark it.

## Components

### Neighbours (`msde.neighbors`)

`select(X, labels, corpus_labels, base_weights) -> Neighbors` (`base_weights` is the weights component's per-point output; distance-only selectors ignore it), where `Neighbors` is a dataclass with `indices (n,k)`, `dist (n,k)`, and, when labels are given, `valid (n,k)` and `orphan_mask (n,)`.

| Strategy | Options |
| --- | --- |
| `EuclideanKNN` | `k` (must satisfy `1 <= k <= n`, validated before the compiled `topk`), `chunk_size`, `device`, `exclude_self` (default `False`: a point is its own first neighbour at distance 0; `True`: it is never its own neighbour, and exactly `k` others are returned). Ranks by Euclidean distance. Every k-NN strategy in this list except `JointKNN` takes the same options. |
| `CosineKNN` | Same options. Ranks by cosine distance `1 - cos(x, y)` (in `[0, 2]`), so it ignores each point's scale; `Neighbors.dist` holds cosine distances. Only the neighbour choice changes: density weights and the movement step still work in the original feature space. |
| `WhitenedKNN` | `k`, `chunk_size`, `device`, `regularization` (ridge on the covariance eigenvalues, relative to their mean; default `1e-6`). Centres and whitens the points with their covariance, then searches by Euclidean distance, i.e. ranks by Mahalanobis distance; `Neighbors.dist` holds Mahalanobis distances. The ranking is invariant to invertible linear maps of the features. The covariance is estimated from all points on every `select` call (so it follows the shifting points), regardless of labels. The ridge keeps degenerate directions (fewer points than features, constant columns) from being amplified, but it is not basis-invariant: lower it (e.g. `1e-9`) for very ill-conditioned data. The `d x d` eigendecomposition runs on the CPU in float64. |
| `JointKNN` | `k`, `h` (kernel bandwidth: a positive number, or `"auto"` = median distance to the `k` nearest neighbours, re-estimated on each call), `metric` (`"euclidean"` or `"cosine"`), `chunk_size`, `device`. Selects `TopK_{j != i} q_j * exp(-d_ij^2 / 2h^2)`, where `q_j` is the weights component's output, so a dense, stable point can outrank a closer unstable one. Never returns the point itself. Zero scores: positive scores rank first and any remaining slots go to the nearest candidates. See [Gravity-based enhancement](#gravity-based-enhancement). |

All three share `ChunkedKNN` (`msde.neighbors.chunked`): a subclass sets a `metric` key from `msde.common.knn.METRICS` and may override `transform(X)` to search in a transformed space (`WhitenedKNN` does).

### Weights (`msde.weights`)

`compute(X, stability=None) -> (n,)`. `stability` is the optional per-point score passed to `forward(..., stability=s)`; only strategies with `uses_stability = True` read it, and `forward` raises if it is given to any other.

| Strategy | Options |
| --- | --- |
| `SparseEmpirical` | `nbd_sample_count_threshold`, `n_neighbors` (15), `n_epochs` (200), `max_iters_weight_count` (4), `satisfiability_proportion` (0.3), `use_chunking` / `chunk_size`, `eps` (fixed radius; otherwise found by bisection), `device` |
| `KDEWeights` | `bandwidth` (a positive number, or `"scott"` (default) / `"silverman"` computed from the data on each call), `chunk_size`, `device`. Gaussian KDE with leave-one-out: `weight_i = sum_{j != i} exp(-\|x_i - x_j\|^2 / 2h^2)`, an unnormalised soft neighbour count in `[0, n-1]`, on the same scale as `SparseEmpirical`'s counts. In high dimensions the rules give `h` close to the per-feature spread, so the estimate is very smooth; pass a number to sharpen it. |
| `GravityWeights` | `density` (any `DensityWeights` strategy giving `rho`, e.g. `KDEWeights`), `alpha`, `beta`, `density_normalization` (`"none"`, `"max"` (default), `"minmax"`, `"rank"`), `density_floor` (default `1e-6`), `stability_floor` (default `0`). Returns `q_j = rho_j^alpha * s_j^beta` with `s` from `forward(..., stability=s)`; without scores `s = 1`. |
| `UniformWeights` | none. Returns 1 for every point (`q_j = 1`), the distance-only baseline. |

### Step (`msde.step`)

Two phases so the loop can reuse work between neighbour recomputations: `prepare(X, base_weights, neighbors) -> state` (on every recompute) and `step(X, state, gate) -> (X_new, change)` (every iteration).

| Strategy | Options |
| --- | --- |
| `PlainBarycenter` | `learning_rate`, `alpha`, `clipping`, `clip_mode` (0 off, 1 saturating, 2 hard cap at `alpha` x median neighbour distance), `use_sparse_shift` (sparse matmul vs gather), `low_precision_barycenter`, `device` |
| `GravityBarycenter` | `eta` (step size, `0 < eta <= 1`; `gate` scales it per call), `h` (positive number or `"auto"`), `metric` (`"euclidean"` / `"cosine"`), `eps`, `chunk_size`, `device`. Computes `w_ij = q_j * exp(-d_ij^2 / 2h^2)`, `mu_i = sum_j w_ij x_j / (sum_j w_ij + eps)` and `x_i' = x_i + eta * (mu_i - x_i)`. A point whose weights sum to at most `eps` stays where it is. Distances are recomputed from the current positions each iteration, from a snapshot. |

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
| `stability` (per call) | Optional `(n,)` scores in `[0, 1]` (higher = more stable), used by `GravityWeights`. |
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

## Gravity-based enhancement

Shift points towards stable, dense local regions. With density `rho_i`, stability `s_i` in `[0, 1]`, and `K(d) = exp(-d^2 / 2h^2)`, the attraction factor is `q_j = rho_j^alpha * s_j^beta`. Selection and attraction weighting are separate components, so the study's three configurations differ in exactly one component:

```python
from msde import (MeanShiftDensityEnhancement as MSDE, EuclideanKNN, JointKNN, KDEWeights,
                  GravityWeights, UniformWeights, GravityBarycenter, EarlyStopLoop)

density = lambda: KDEWeights("scott")
step = lambda: GravityBarycenter(eta=0.05, h="auto")
loop = lambda: EarlyStopLoop(max_iters_shift=1)       # start with one iteration

# 1. distance-only selection and attraction
m1 = MSDE(neighbors=EuclideanKNN(k=20, exclude_self=True), weights=UniformWeights(), step=step(), loop=loop())
# 2. Approach 1: distance-only selection, density/stability-weighted attraction
m2 = MSDE(neighbors=EuclideanKNN(k=20, exclude_self=True), weights=GravityWeights(density()), step=step(), loop=loop())
# 3. Approach 2: joint selection and attraction
m3 = MSDE(neighbors=JointKNN(k=20, h="auto"), weights=GravityWeights(density()), step=step(), loop=loop())

shifted, movement, _ = m3(X, stability=s)             # s: (n,) in [0, 1]
```

| Spec | Component |
| --- | --- |
| `k`, distance metric | `k` and `metric` of the neighbours component (`GravityBarycenter`/`JointKNN` take `metric` for the kernel distance) |
| `alpha`, `beta`, density / stability preprocessing | `GravityWeights` (`alpha`, `beta`, `density_normalization`, `density_floor`, `stability_floor`) |
| `h` | `JointKNN.h` and `GravityBarycenter.h`; set both to the same number, or leave both `"auto"` (same estimate) |
| `eta`, `eps` | `GravityBarycenter` |
| Number of iterations, recomputing neighbours | the loop's `max_iters_shift` and `recompute_neighbors` |
| `rho` | any density strategy passed as `GravityWeights(density=...)` |

Decisions the spec left open:

- **Self-neighbours** are excluded (`exclude_self=True` on the k-NN strategies; `JointKNN` always).
- **Zero attraction:** with `stability_floor=0`, zero stability removes a point's attraction completely. A small positive floor keeps it faintly attractive. `rho` is floored at `density_floor` so any `alpha` is valid.
- **Zero weight sum:** a point whose weights sum to at most `eps` is not moved. The literal formula would pull it towards the origin.
- **`JointKNN` zero scores:** positive scores rank first; remaining slots (or all of them, if every score is zero) go to the nearest remaining candidates.
- **`JointKNN` locality:** all points are candidates, with no distance pre-filter, so selection is global in principle and local only through `h`.
- **Stability is a destination property:** it sets how strongly a point attracts others, never how much it resists moving.
- **Neighbours are fixed** between recomputations while distances in the kernel follow the current positions; set `recompute_neighbors=1` on the loop to refresh the neighbour set every iteration.
- **Normalise embeddings** after shifting yourself if the downstream model needs unit length, and consider `metric="cosine"` for unit-normalised embeddings.

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
