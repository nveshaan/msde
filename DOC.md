# MSDE

MSDE (mean-shift density enhancement) computes an empirical density weight for each point, builds a fuzzy k-nearest-neighbor graph, and repeatedly moves each point toward a weighted neighbor barycenter.

The main public API is `MeanShiftDensityEnhancement`:

```python
import torch
from msde import MeanShiftDensityEnhancement

X = torch.randn(10_000, 128, device="cuda")

# Fast/legacy path
model = MeanShiftDensityEnhancement(
    k=30,
    device="cuda",
    use_chunking=True,
    weight_chunk_size=2048,
)

shifted, movement, trajectory = model(X)
```

`shifted` has shape `(n, d)`, `movement` has shape `(n,)` and contains accumulated movement magnitudes, and `trajectory` is empty unless `keep_trajectory=True`.

## Notation and cost model

- `n`: number of query points in a self-shift or weight computation.
- `m`: number of points in a fixed reference manifold.
- `d`: feature dimension.
- `k`: neighbors used by the shift graph.
- `q`: neighbors used by the empirical-weight graph. The model uses `q=15` internally.
- `c`: `chunk_size` or `weight_chunk_size`.
- `I_w`: `max_iters_weight_count` in the standalone weight API. The model uses `4`.
- `I_s`: number of shift iterations, bounded by `max_iters_shift`.
- `r`: number of neighbor-graph rebuilds. With `recompute_neighbors=0`, `r=1`; with period `p`, `r <= ceil(I_s/p)`.

Costs below count distance arithmetic and omit constant factors. Exact k-NN remains an all-corpus search even when chunked:

- k-NN time: `O(n * m * d)` for a query set of size `n` and corpus of size `m`; self-search uses `m=n`.
- k-NN working space: `O(c * m + n * k)` for the distance chunk, outputs, and indices.
- Sparse graph storage: `O(n * q)` edges, plus `O(n)` row metadata.
- Chunked empirical-weight passes: each radius-count pass is `O(n^2 * d)` in the graph embedding dimension and uses `O(c * n)` working space. Epsilon search performs up to 50 passes, plus a possible relaxed retry.
- Dense empirical weights: `O(n^2 * d)` time and `O(n^2)` distance storage, in addition to the dense similarity matrix.
- Shift step: sparse mode is approximately `O(n * m * d)` for sparse matrix multiplication; gather mode is `O(n * k * d)`. Clipping adds `O(n * k * d)` distance work.
- Autograd can retain intermediate tensors, so peak training memory can exceed these forward working-space estimates.

`torch.compile` is used for dense kernels. The first call for a new shape/configuration can include compilation and should not be used as a steady-state benchmark.

## Implementation semantics

- The default path preserves the original sparse/hard graph semantics. Exact `topk`, hard graph membership, hard radius counts when `temperature=None`, and detached binary searches remain discrete in that path.
- `fully_differentiable=True` switches the class to a continuous dense relaxation. Neighbor assignment is soft over the full query-corpus distance matrix rather than a hard k-NN index set; density weights use sigmoid counts; rho/sigma/epsilon are obtained with differentiable fixed-point/Newton iterations; graph pruning is a smooth gate rather than structural `nonzero()` pruning; movement and clipping are smooth; orphan handling is a continuous validity gate; and the shift loop does not use data-dependent early stopping. This removes the hard value/graph decisions that otherwise block gradients.
- In the differentiable path, `temperature` is the single public relaxation/smoothness hyperparameter. Internal temperatures and solver budgets are derived automatically. The path is dense and therefore substantially more expensive than the sparse legacy path.
- Label masks remain discrete by necessity: class equality determines whether a reference point is eligible. The differentiable path still uses the label mask as a fixed eligibility mask, but zero eligible mass is handled by a continuous validity gate rather than an `orphan_mask`/`torch.where` freeze.
- `torch.compile` is used for suitable dense kernels. The first call for a new shape/configuration can include compilation and should not be used as a steady-state benchmark.
- Scalar options are validated as one-element tensors, moved to `device`, and detached when fixed. A scalar becomes a learnable `torch.nn.Parameter` only when both its `learn_*` flag and `enable_gradients` are enabled; otherwise it is stored as a buffer.
- In `fully_differentiable=True` reference-manifold mode, the reference tensor is kept connected to autograd and its density prior is recomputed inside the differentiable forward path, so gradients can propagate through both query and reference embeddings. In the legacy path, the reference manifold is detached and its density weights are cached when set.
- The module logger is silent when `log_file=None`. When a path is supplied, it writes this module's messages to that file and disables propagation to console/root handlers.

## `MeanShiftDensityEnhancement` configuration

### Neighbor graph and geometry

| Option | Configurations and effect | Use when / watch out |
| --- | --- | --- |
| `k` | Positive integer number of legacy shift neighbors. | Larger values smooth the legacy hard graph but increase its gather/graph cost. In fully differentiable mode, `k` controls the softness target of the continuous neighbor assignment rather than selecting exactly `k` hard indices. |
| `n_neighbors` | Not a constructor option. The legacy density graph uses `15` neighbors and `200` fuzzy-set epochs. | Use `get_empirical_weights` directly when the standalone legacy density graph must use different internal settings. |
| `recompute_neighbors` | Legacy path: `0`/`None` builds the graph once; positive `p` rebuilds periodically. | In fully differentiable mode there is no hard neighbor topology to rebuild; each shift iteration evaluates the continuous assignment directly. |
| `X` | `None`: self-shift; `(m,d)` tensor/array: reference-manifold mode. | In fully differentiable mode the reference can remain connected to autograd. In the legacy path it is detached and cached. |
| `labels` | Integer labels can restrict eligible neighbors to the same class. | Label equality itself is discrete. Fully differentiable mode smooths values and geometry, but does not make class membership differentiable. |

### Weight computation

| Option | Configurations and effect | Use when / watch out |
| --- | --- | --- |
| `use_chunking` | Legacy path: chunked sparse graph processing vs dense baseline. | The fully differentiable path is dense by design; `use_chunking` does not turn it back into hard sparse k-NN. |
| `weight_chunk_size` | Legacy streamed-pass chunk size. | Primarily relevant to the legacy path. |
| `nbd_sample_count_threshold` | Density threshold used to determine the target neighborhood mass/radius. | Larger values generally broaden the density support. |
| `satisfiability_proportion` | Legacy standalone proportion parameter; the class's differentiable path uses the same density-conditioning concept internally. | Most users can leave the default unchanged. |
| `temperature` | The primary smoothness control. | For end-to-end training, use a positive value such as `0.05–0.2` and tune it based on gradient scale and how sharp you want neighbor assignments to be. |
| `eps` | Optional fixed epsilon. | Supplying `eps` avoids epsilon solving. In fully differentiable mode, a fixed epsilon is still differentiable downstream with respect to embeddings and other learnable parameters. |
| `learn_eps` | Makes epsilon learnable. | In fully differentiable mode, give an explicit initial `eps`; epsilon participates in the continuous computation graph. |
| `differentiable_eps` and `eps_*_temperature` | Legacy advanced controls for the partially differentiable epsilon helper. | Do not use these in normal fully differentiable mode. |
| `soft_prune` and `soft_prune_temperature` | Legacy advanced controls for sparse soft pruning. | Soft pruning is automatic in fully differentiable mode. |

### Shift update

| Option | Configurations and effect | Use when / watch out |
| --- | --- | --- |
| `learning_rate` | Scalar multiplier on the shift step. | Can be made learnable with `learn_learning_rate=True`. |
| `learn_learning_rate` | Registers `learning_rate` as a parameter. | Useful for end-to-end optimization; keep the learned value in a sensible range. |
| `alpha` | Clipping scale. | Only affects clipping. |
| `learn_alpha` | Registers `alpha` as a parameter. | Useful when clipping is enabled and its scale should be learned. |
| `clipping` | Enables step clipping. | In fully differentiable mode clipping remains smooth. |
| `clip_mode` | Legacy mode selector; fully differentiable mode internally uses smooth clipping. | Use the legacy setting only when intentionally using the legacy path. |
| `use_sparse_shift` | Legacy sparse-vs-gather representation choice. | Not the representation used by the fully differentiable path. |
| `low_precision_barycenter` | Legacy gather-mode precision optimization. | Not used by the fully differentiable path. |
| `max_iters_shift` | Number of shift iterations in the fixed differentiable loop. | More iterations mean a deeper autograd graph and higher compute/memory cost. |
| `shift_threshold` | Legacy early-stop threshold. | Not used to terminate the fully differentiable loop. |
| `gate` | Per-call multiplier on the shift step. | Useful for scaling one call without changing stored parameters. |

### Differentiability

| Option | Configurations and effect | Use when / watch out |
| --- | --- | --- |
| `fully_differentiable` | `False`: existing sparse/hard MSDE path. `True`: continuous dense end-to-end path. | Set this to `True` when gradients must flow through neighbor assignment, density estimation, graph weighting, pruning, and the iterative shift. |
| `temperature` | Single public relaxation hyperparameter in fully differentiable mode. | Smaller values approximate sharper/harder decisions; larger values produce smoother gradients. |
| `enable_gradients` | Preserves gradients when `True`; detaches the query input when `False`. | Keep `True` for end-to-end training. |
| `learn_temperature` | Makes `temperature` a learnable parameter. | Useful when the model should learn its smoothness; constrain positivity in the optimizer if necessary. |
| `keep_trajectory` | Stores the initial state and each completed differentiable shift step. | Can dominate memory because the trajectory retains `(I_s+1)*n*d` values. |
| `log_file` | Enables module logging when a path is supplied. | Logging is disabled by default. |

### What `fully_differentiable=True` changes

The strict differentiable path removes the main value/graph discontinuities from the numerical computation:

1. **Neighbor selection:** continuous soft assignment over the full query-corpus distance matrix replaces hard `topk`.
2. **Density counts:** sigmoid radius counts replace `(distance < epsilon)`.
3. **Density parameters:** rho, sigma, and epsilon use fixed differentiable solver iterations rather than hard bisection control flow.
4. **Pruning:** no data-dependent sparse `nonzero()` selection is used; edge weights are smoothly gated.
5. **Movement:** the zero-movement safeguard is continuous.
6. **Clipping:** minimum-like behavior is smoothed.
7. **Orphans:** zero-neighbor mass is handled by a continuous validity factor instead of a hard freeze.
8. **Iteration control:** the differentiable shift loop runs a fixed number of iterations; mean movement is logged but does not determine execution.

The class therefore exposes a deliberately small public interface:

```python
model = MeanShiftDensityEnhancement(
    k=30,
    temperature=0.1,
    fully_differentiable=True,
)
```

There is normally no need to specify soft-top-k temperature, pruning temperature, epsilon-count temperature, epsilon-threshold temperature, or solver iteration count.

## Reference-manifold mode

```python
# Fully differentiable reference mode
reference = torch.randn(50_000, 128, device="cuda", requires_grad=True)

model = MeanShiftDensityEnhancement(
    X=reference,
    k=30,
    temperature=0.1,
    fully_differentiable=True,
    device="cuda",
)

queries = torch.randn(2_000, 128, device="cuda", requires_grad=True)

shifted, movement, _ = model(queries)

loss = shifted.square().mean()
loss.backward()
```

In fully differentiable reference mode, the reference manifold remains connected to autograd and its density prior is recomputed as part of the differentiable forward computation. Therefore gradients can propagate into both `queries` and `reference`.

In the legacy reference mode, the reference manifold is detached and its empirical density weights are computed when the reference is set.

Labels still represent discrete class membership. When labels are supplied, cross-class reference points are excluded by a fixed eligibility mask; the geometry and weighting remain differentiable with respect to the floating-point inputs.

## Standalone components

These functions remain available for profiling or custom pipelines:

- `torch_knn(X, k, corpus=None, labels=None, corpus_labels=None)`: exact legacy chunked k-NN; returns hard indices and distances.
- `compute_fixed_knn(...)`: same hard legacy search, returning only indices.
- `fuzzy_simplicial_set_torch(...)`: legacy sparse fuzzy-graph COO builder.
- `get_empirical_weights(...)`: legacy density-weight pipeline.
- `compute_weights_from_similarity_chunked(...)`: legacy streamed sparse density weighting.
- `compute_weights_from_similarity_dense(...)`: legacy dense baseline.
- `radius_counts_chunked(...)`: legacy radius counting; positive `temperature` selects sigmoid counts, while `temperature=None` uses hard indicators.

For end-to-end differentiability, prefer the `MeanShiftDensityEnhancement(..., fully_differentiable=True)` class path rather than assembling these legacy hard/sparse components manually.

## Practical recipes

### Small, fast baseline

```python
model = MeanShiftDensityEnhancement(
    k=15,
    max_iters_shift=3,
    use_chunking=False,
    use_sparse_shift=False,
    device="cuda",
)
```

Use this legacy configuration when speed and sparse graph structure matter more than differentiating through graph construction.

### Large dataset with bounded working memory

```python
model = MeanShiftDensityEnhancement(
    k=30,
    use_chunking=True,
    weight_chunk_size=1024,
    use_sparse_shift=True,
    recompute_neighbors=0,
    device="cuda",
)
```

This is still the legacy sparse path. Chunking controls peak working memory but does not change its all-corpus asymptotic search cost.

### End-to-end differentiable objective

```python
model = MeanShiftDensityEnhancement(
    k=30,
    temperature=0.1,
    fully_differentiable=True,
)

shifted, movement, _ = model(X)

loss = loss_fn(shifted)
loss.backward()
```

This is the recommended API for learning an upstream encoder jointly with MSDE. `temperature` is the main additional hyperparameter to tune.

### Learning the MSDE hyperparameters too

```python
model = MeanShiftDensityEnhancement(
    k=30,
    temperature=0.1,
    learning_rate=0.3,
    alpha=0.5,
    fully_differentiable=True,
    learn_temperature=True,
    learn_learning_rate=True,
    learn_alpha=True,
)
```

All three scalars can then receive gradients from the downstream objective.

## Cost model for the fully differentiable path

The continuous relaxation removes the hard top-k bottleneck by evaluating query-corpus distances densely.

For query size `n`, corpus size `m`, and feature dimension `d`:

- Dense distance construction: approximately `O(n*m*d)` time and `O(n*m)` distance memory.
- Continuous neighbor assignment and density weighting: additional `O(n*m)`–scale tensor work.
- Shift iteration: approximately `O(I_s*n*m*d)` when each iteration evaluates the continuous query-corpus geometry.
- Peak autograd memory can be substantially larger than forward working memory because intermediate tensors from every differentiable iteration may be retained.

For self-shift, `m=n`, so the dominant cost is quadratic in the number of points.

This is the fundamental trade-off for obtaining gradients through neighbor membership: a genuinely continuous relaxation cannot preserve the exact sparse `O(n*k)` graph structure at the same time.

## Failure modes and checks

- `fully_differentiable=True` requires floating-point feature tensors and uses a positive temperature.
- `k` still controls the target neighbor mass of the continuous assignment; it is not a hard limit on the number of candidates considered.
- `learn_temperature=True` requires an initial temperature or uses the differentiable-mode default when appropriate.
- `learn_eps=True` in fully differentiable mode requires an explicit initial `eps`.
- Reference and query feature dimensions must match.
- Label tensors must be integer-like and aligned with their corresponding points.
- Label equality remains discrete; only the geometric/weight computation is relaxed.
- The differentiable path deliberately avoids data-dependent early stopping so that the autograd graph has fixed control flow.
- Dense continuous computation can become prohibitively expensive for large `n` or `m`; use the legacy sparse path when full graph-construction gradients are unnecessary.
- Benchmark after the first call for each relevant shape because compiled kernels may compile lazily.
