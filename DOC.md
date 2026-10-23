# MSDE

MSDE (mean-shift density enhancement) computes an empirical density weight for each point, selects neighbours, and repeatedly moves each point toward a weighted neighbour barycenter.

It is built from **four swappable components**. Each has a base class and one or more strategies, and any of them may be non-differentiable:

| Component | Base class | Question it answers | Strategies |
| --- | --- | --- | --- |
| `msde.neighbors` | `NeighborSelector` | Which corpus points is each point pulled towards? | `HardKNN`, `SoftKNN` |
| `msde.weights` | `DensityWeights` | How typical/dense is each corpus point? | `SparseEmpirical`, `SoftEmpirical` |
| `msde.step` | `MovementStep` | How does a point move towards the weighted barycenter? | `PlainBarycenter`, `SoftBarycenter` |
| `msde.loop` | `ShiftLoop` | How is the iteration run (schedule, stopping, pre/post-processing)? | `EarlyStopLoop`, `FixedLoop` |

Code shared by more than one component lives in `msde.common` (`knn`, `graph`, `eps`, `labels`, `utils`). Components never import from each other.

```python
import torch
from msde import (
    MeanShiftDensityEnhancement, HardKNN, SparseEmpirical, PlainBarycenter, EarlyStopLoop,
)

X = torch.randn(10_000, 128, device="cuda")

model = MeanShiftDensityEnhancement(
    neighbors=HardKNN(k=30, device="cuda"),
    weights=SparseEmpirical(nbd_sample_count_threshold=30, use_chunking=True, chunk_size=2048, device="cuda"),
    step=PlainBarycenter(learning_rate=0.3, device="cuda"),
    loop=EarlyStopLoop(max_iters_shift=5),
    device="cuda",
)
shifted, movement, trajectory = model(X)
```

Every component is optional; omitted ones default to the combination above. `shifted` has shape `(n, d)`, `movement` has shape `(n,)` (accumulated movement), and `trajectory` is empty unless the loop has `keep_trajectory=True`.

An end-to-end differentiable configuration is just a different composition:

```python
T = 0.1
model = MeanShiftDensityEnhancement(
    neighbors=SoftKNN(k=30, temperature=T),
    weights=SoftEmpirical(n_neighbors=30, nbd_sample_count_threshold=30, temperature=T),
    step=SoftBarycenter(temperature=T),
    loop=FixedLoop(max_iters_shift=5),
)
```

Mixed compositions are allowed when shapes work (e.g. `HardKNN` + `SoftEmpirical` + `PlainBarycenter` keeps hard neighbours but differentiable density weights). The one hard rule: `neighbors.form` must equal `step.accepts` (`"hard"` or `"soft"`), checked at construction.

## Notation and cost model

- `n`: query points; `m`: reference-manifold points (`m=n` in self-shift); `d`: feature dimension; `k`: neighbours of the shift graph; `q`: `SparseEmpirical.n_neighbors` (default 15); `c`: chunk size; `I_s`: shift iterations; `r`: neighbour rebuilds (`r=1` for `recompute_neighbors=0`, else `<= ceil(I_s/p)`).

- `HardKNN`: `O(n*m*d)` time even when chunked; `O(c*m + n*k)` working space.
- `SparseEmpirical`: sparse graph `O(n*q)`; each chunked radius-count pass is `O(n^2*d)` with `O(c*n)` working space, up to 50 passes per epsilon search (plus a relaxed retry). The dense variant is `O(n^2*d)` time and `O(n^2)` memory.
- `PlainBarycenter`: sparse mode ~`O(n*m*d)`; gather mode `O(n*k*d)`; clipping adds `O(n*k*d)`.
- `SoftKNN` / `SoftEmpirical` / `SoftBarycenter`: dense `O(n*m)` memory and `O(I_s*n*m*d)` time. For self-shift this is quadratic in `n`: exact gradients through neighbour membership cannot coexist with the sparse `O(n*k)` structure.
- Autograd retains intermediates, so peak training memory can exceed these forward estimates.
- `torch.compile` is used for dense kernels; the first call for a new shape/configuration includes compilation, so do not benchmark it.

## Components

### Neighbours (`msde.neighbors`)

`select(X, corpus, labels, corpus_labels, exclude_self) -> Neighbors`. `Neighbors` is a dataclass with `form="hard"` (`indices (n,k)`, `dist`, `valid`, `orphan_mask`, optional `weights`) or `form="soft"` (`membership (n,m)`, `dist`, `valid`).

| Strategy | Options |
| --- | --- |
| `HardKNN` | `k` (must satisfy `1 <= k <= corpus size`, validated before the compiled `topk`), `soft_topk_temperature` (reweights selected neighbours by a sigmoid of their distance to the k-th neighbour; carries gradients to positions), `learn_soft_topk_temperature`, `chunk_size`, `device` |
| `SoftKNN` | `k` (target membership mass per point, not a hard limit), `temperature`, `learn_temperature`, `solver_iterations`, `device` |

### Weights (`msde.weights`)

`compute(corpus) -> (m,)`. `cache_reference` says whether a fixed reference manifold's weights are computed once in `set_reference` (detached, no-grad) or recomputed every call.

| Strategy | Options |
| --- | --- |
| `SparseEmpirical` (caches reference) | `nbd_sample_count_threshold`, `n_neighbors` (15), `n_epochs` (200), `max_iters_weight_count` (4), `satisfiability_proportion` (0.3), `use_chunking` / `chunk_size`, `temperature` (sigmoid counts when set, hard indicators when `None`), `eps`, `learn_temperature`, `learn_eps` (with `eps=None` the epsilon is initialised from the data on first use), `soft_prune` / `soft_prune_temperature`, `differentiable_eps` (chunked only) with `eps_count_temperature` / `eps_threshold_temperature` |
| `SoftEmpirical` (recomputes) | `n_neighbors`, `n_epochs`, `nbd_sample_count_threshold`, `satisfiability_proportion`, `max_iters_weight_count`, `temperature`, `eps`, `learn_temperature`, `learn_eps` (requires explicit initial `eps`), `eps_threshold_temperature`, `solver_iterations` |

### Step (`msde.step`)

Two phases so the loop can reuse work between neighbour recomputations: `prepare(X, corpus, base_weights, neighbors) -> state` (on every recompute) and `step(X, corpus, state, gate) -> (X_new, change)` (every iteration).

| Strategy | Options |
| --- | --- |
| `PlainBarycenter` (accepts `"hard"`) | `learning_rate`, `alpha`, `learn_learning_rate`, `learn_alpha`, `clipping`, `clip_mode` (0 off, 1 saturating, 2 hard cap at `alpha` x median neighbour distance), `use_sparse_shift`, `low_precision_barycenter`, `smooth_movement_gate` |
| `SoftBarycenter` (accepts `"soft"`) | `learning_rate`, `alpha`, `clipping`, `clip_mode` (1 saturating, 2 soft minimum), `temperature` (only used by `clip_mode=2`), `learn_*` flags. Smooth zero-movement gate and a continuous validity gate for points with no neighbour mass |

### Loop (`msde.loop`)

`ShiftLoop.run` is the shared skeleton: `preprocess` -> iterate (recompute neighbours / `prepare` on schedule, `step`, orphan freeze, trajectory) -> `postprocess`. Strategies override `should_recompute`, `converged`, `preprocess`, `postprocess`.

| Strategy | Options |
| --- | --- |
| `EarlyStopLoop` | `max_iters_shift`, `shift_threshold` (stop when mean movement is below it; a data-dependent `.item()` branch), `recompute_neighbors` (`0`/`None`: once; positive `p`: at iterations 0, p, 2p, ...), `keep_trajectory` |
| `FixedLoop` | `max_iters_shift`, `recompute_neighbors` (default 1: every iteration), `keep_trajectory`. No data-dependent branching, so the graph has fixed control flow |

## `MeanShiftDensityEnhancement` options

`neighbors`, `weights`, `step`, `loop` (components), plus:

| Option | Effect |
| --- | --- |
| `X` | `None`: self-shift. `(m,d)`: reference-manifold mode; every `forward` input is shifted against this fixed corpus. Settable later with `set_reference_manifold(X, labels=None)`. |
| `labels` | Integer labels for the reference manifold; per-call labels go to `forward(X, gate=1.0, labels=None)`. |
| `label_mode` | `"same_label"` (default) or `"zero_label"`; see [Label masking](#label-masking). |
| `reference_grad` | Keep autograd history on the reference tensor (default `False`: detached). Pair with a non-caching weights strategy such as `SoftEmpirical`, which recomputes the reference prior every call so gradients reach both queries and reference. |
| `enable_gradients` | `False` detaches `forward`'s input and freezes existing component parameters. |
| `gate` (per call) | Multiplier on the shift step for that call. |
| `log_file` | Logging is silent when `None`; with a path, this package's messages are written there and propagation to root handlers is disabled. |

`model.learnable_parameters` collects every learnable scalar, keyed `"<component>.<name>"` (e.g. `"neighbors.temperature"`). A scalar is a `torch.nn.Parameter` only when its `learn_*` flag is set, else a buffer. Passing an existing `nn.Parameter` as a value registers that very object, so one learnable temperature can be shared across components.

## Label masking

Labels passed to `forward(X, labels=y)` restrict which corpus points (X itself, or `X_ref` in reference mode) can be neighbours:

| `label_mode` | Point with label 0 | Point with label 1, 2, 3, ... |
| --- | --- | --- |
| `"same_label"` (default) | label-0 neighbours | neighbours with its own label |
| `"zero_label"` | label-0 neighbours | label-0 neighbours |

```python
model = MeanShiftDensityEnhancement(label_mode="zero_label")
shifted, movement, _ = model(X, labels=y)   # every point shifts towards label-0 points
```

- Reference mode: if the reference has labels, per-call labels are optional (omit them for an unmasked shift); if it has none, passing labels is an error.
- `HardKNN`: cross-class distances are set to `inf` before `topk`. With fewer than `k` eligible points the extras get zero weight; with none, the point is an orphan and the loop leaves it unmoved.
- `SoftKNN`: membership is multiplied by the eligibility mask with target mass `min(eligible count, k)`; with no eligible mass `SoftBarycenter`'s validity gate scales the step to zero.
- `k` is validated against the unmasked corpus size.
- Density weights are not masked; only neighbour eligibility depends on labels.
- Label equality is discrete by necessity; the geometry and weighting stay differentiable.
- `zero_label` needs label-0 points in the corpus; otherwise every point is an orphan and nothing moves.

## Writing a new strategy

Subclass the component's base class and override the documented methods; nothing else changes. A step that never moves points:

```python
from msde import MovementStep

class Frozen(MovementStep):
    accepts = "hard"                      # which Neighbors form it consumes

    def prepare(self, X, corpus, base_weights, neighbors):
        return None                       # opaque state reused by step()

    def step(self, X, corpus, state, gate=1.0):
        return X, torch.zeros(X.shape[0], dtype=X.dtype, device=X.device)
```

Use `self.register_scalar(name, value, learnable, device)` (from `ParametrizedModule`, the shared base) for hyperparameters that may be learnable. Loop pre/post-processing:

```python
class Doubling(EarlyStopLoop):
    def postprocess(self, result, ctx):
        shifted, dist, trajectory = result
        return shifted * 2, dist, trajectory
```

## Standalone functions

For profiling or custom pipelines: `msde.common.knn` (`torch_knn`, `compute_fixed_knn`, `masked_soft_knn_weights`), `msde.common.graph` (`fuzzy_simplicial_set_torch`, `soft_fuzzy_graph`), `msde.common.eps` (`radius_counts_chunked`, `calculate_eps_from_similarity`, `smooth_epsilon_from_graph`), and `msde.weights.sparse` (`get_empirical_weights`, `compute_weights_from_similarity_chunked`, `compute_weights_from_similarity_dense`).

## Failure modes and checks

- `HardKNN` validates `1 <= k <= corpus_size` before calling `torch.topk`; in a masked run the corpus size is still the unmasked total.
- Soft strategies require a positive `temperature`; `learn_eps=True` in `SoftEmpirical` requires an explicit initial `eps`; `learn_temperature=True` needs an initial temperature.
- `SparseEmpirical(differentiable_eps=True)` requires `use_chunking=True`.
- A `neighbors`/`step` pair with different forms raises `ValueError` at construction.
- Reference and query feature dimensions must match; label tensors must be integer-like, 1-D and aligned with their points.
- Dense soft computation can be prohibitively expensive for large `n` or `m`; use the hard/sparse components when gradients through graph construction are unnecessary.
- Benchmark after the first call per shape: compiled kernels compile lazily.
