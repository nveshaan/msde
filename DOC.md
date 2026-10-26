# MSDE

Mean-shift density enhancement (MSDE) of embeddings, in one file: [`src/msde.py`](src/msde.py). Every point is repeatedly moved towards a weighted mean of its neighbours until the movement converges, and the embeddings stay on the unit sphere throughout. Only `torch` is required.

```python
import torch
from msde import MeanShiftDensityEnhancement

X = torch.randn(3000, 512)                      # any (n, d) float embeddings; they are normalised internally

model = MeanShiftDensityEnhancement(k=50, learning_rate=0.33, device="mps")   # "cpu" / "cuda" / "mps"
shifted, movement, trajectory = model(X)
```

`shifted` is the `(n, d)` unit-norm result, `movement` the `(n,)` distance each point travelled (summed chord lengths on the sphere), and `trajectory` a list of unit-norm states (only if `keep_trajectory=True`). Everything runs under `torch.no_grad()`: nothing is differentiable.

## Pipeline

`forward` runs these stages in order.

1. **Normalise** the input to unit length. Points are re-normalised immediately after every shift, so every state the loop passes through, every trajectory entry and the output are unit-norm. (A zero vector has no direction: it is pulled along by its neighbours and is unit-norm after the first shift.)
2. **Whitening (optional, off by default).** With `whiten_density` and/or `whiten_selection` set, the points are centred and rescaled by the inverse square root of their covariance, so that Euclidean distance between whitened points is the Mahalanobis distance and correlated or unevenly scaled directions stop dominating. The two flags are independent: one switches the density computation, the other the neighbour selection. The step always works on the unit sphere. The covariance is accumulated and decomposed in float64 on the CPU (forming it in float32 squares the condition number and loses the small eigenvalues of typical embeddings), with a ridge added to the eigenvalues (see [The ridge](#the-ridge)); the result is cast back to the input dtype before it is moved to the device (some devices have no float64).
3. **Density** of every point, computed **once** on the initial points (whitened if `whiten_density`); the UMAP fuzzy graph built along the way is kept and handed to the score, weight and stability functions:
   - build an exact `n_neighbors`-nearest-neighbour graph and turn it into a sparse fuzzy graph (per-point scale found by bisection, symmetrised by fuzzy union, weak edges dropped);
   - compare points as rows of that sparse graph: distances come from one sparse-times-dense matmul `S @ chunk.T` per chunk of rows;
   - find the smallest radius at which at least `satisfiability_proportion` of the points have more than `nbd_sample_count_threshold` others inside it (bisection, relaxed once if it fails);
   - a point's density is its number of other points within the radius, averaged over `max_iters_weight_count` radii shrinking linearly to zero.
4. **Select neighbours.** In the geometry of the current points (Euclidean on the sphere, or Mahalanobis if `whiten_selection`), every other point is scored by `NeighborScore` and the `k` best are kept: with the default score `density^a * stability^b * exp(-c * d^2 / (2 h^2))`, so a dense, stable point can outrank a closer, sparse one. All points are candidates (no distance pre-filter) and a point is never its own neighbour. Candidates with a positive score rank by score; if fewer than `k` have one, the remaining slots go to the nearest candidates.
5. **Weights.** Each selected neighbour gets a weight from `NeighborWeight` (default `density^a * stability^b`); weights of picks that violate the label mask are zeroed and rows are normalised to sum to 1. The default weights are computed when neighbours are selected (and again after every iteration if a `stability_fn` is given); a `weight_fn` you pass is evaluated at the start of every iteration, with the current distances to the selected neighbours.
6. **Step.** `mean = W @ X` with `W` a sparse `(n, n)` matrix holding the `n*k` normalised weights (one sparse matmul), then `x' = normalise(x + learning_rate * gate * (mean - x))`.
7. **Repeat** until the mean per-point movement on the sphere falls below `shift_threshold` (convergence) or `max_iters_shift` iterations have run. With `recompute_neighbors = N > 0`, stages 2 and 4 (the whitening, if on, and the neighbour selection) are redone every `N` iterations on the current points (iterations 0, N, 2N, ...); the density is never recomputed. A `weight_fn` and a `stability_fn`, if given, run in every iteration regardless of `recompute_neighbors`.

## Constructor

| Argument | Default | Meaning |
| --- | --- | --- |
| `k` | 50 | Neighbours each point is pulled towards; needs `k <= n - 1`. |
| `nbd_sample_count_threshold` | 30 | Density: how many others must lie within the radius for a point to "satisfy" it. |
| `n_neighbors` | 15 | Density: neighbours of the fuzzy graph (`<= n`). |
| `n_epochs` | 200 | Density: edges weaker than `max / n_epochs` are dropped from the graph. |
| `max_iters_weight_count` | 4 | Density: number of shrinking radii averaged. |
| `satisfiability_proportion` | 0.3 | Density: fraction of points that must satisfy the threshold. |
| `weight_chunk_size` | 2048 | Density: rows of graph-space distances held at once. |
| `learning_rate` | 0.3 | Fraction of the way a point moves towards its weighted mean per iteration. |
| `distance_strength` | 1.0 | Strength `c` of the distance term of the default selection score. |
| `density_strength` | 1.0 | Strength `a` of the density term, in the default selection score and neighbour weight. |
| `stability_strength` | 1.0 | Strength `b` of the stability term, in the default selection score and neighbour weight. |
| `h` | `"auto"` | Kernel width of the distance term, in the units of the selection space (whitened units if `whiten_selection`): a positive number, or `"auto"` = median distance to the `k` nearest neighbours, re-estimated at every selection. |
| `whiten_density` | `False` | Compute the density on whitened points (Mahalanobis geometry). |
| `whiten_selection` | `False` | Select neighbours by distances in whitened space, re-estimated from the current points at every selection. |
| `whiten_regularization` | 1e-6 | Relative ridge of the whitening; only used when one of the two flags is set (see [The ridge](#the-ridge)). |
| `recompute_neighbors` | 0 | 0 / `None`: select neighbours once; `N > 0`: reselect every `N` iterations. |
| `max_iters_shift` | 1000 | Safety cap on iterations; the loop normally stops earlier by convergence (see [Speed](#speed) for typical counts). |
| `shift_threshold` | 1e-4 | Convergence: stop when the mean per-point movement is below this. |
| `keep_trajectory` | `False` | Also return the unit-norm state after every iteration. |
| `label_mode` | `"same_label"` | How `labels` restrict neighbours: `"same_label"` (neighbours share the point's label) or `"zero_label"` (neighbours must have label 0, whatever the point's own label). |
| `chunk_size` | 1024 | Rows of the `n x n` distance blocks held at once during selection. |
| `device` | auto | `"mps"`, `"cuda"` or `"cpu"`; inputs are moved there. |

A strength is a number `>= 0`; 0 switches that factor off and larger values weight it more (they are exponents on density and stability, and a multiplier on the squared distance in the kernel).

## `forward(X, gate=1.0, labels=None, stability_fn=None, score_fn=None, weight_fn=None)`

| Argument | Meaning |
| --- | --- |
| `X` | `(n, d)` floating-point embeddings. |
| `gate` | Multiplier on the step size for this call (0 leaves the points where they are). |
| `labels` | `(n,)` integer labels restricting neighbour selection (see `label_mode`). A point with no valid neighbour is left where it is. |
| `stability_fn` | Optional callable that updates the stability scores (`(n,)`, in `[0, 1]`, higher = more stable) after every iteration from that iteration's step vectors. Without one, every point stays at 1.0. See [Stability](#stability). |
| `score_fn` | Replacement for the selection score. Called whenever neighbours are selected, including the initial selection, so with `recompute_neighbors=0` it defines the single neighbour graph the run uses. |
| `weight_fn` | Replacement for the neighbour weights of the weighted mean. Evaluated at the start of **every** iteration, with the current distances and stability scores. The default is used when it is not given. |

### Replacing the selection score, the neighbour weights and the stability update

All three are plain callables that carry their own parameters. All three receive the **0-based index of the current iteration** and the **iteration cap** `n_iters` (`max_iters_shift`; a run may stop earlier by convergence), so they can change over the run, and the **UMAP fuzzy graph** that was computed for the density (below). The defaults are the callable classes `NeighborScore(distance_strength, density_strength, stability_strength)` and `NeighborWeight(density_strength, stability_strength)`.

```python
score_fn(dist, density, stability, h, iteration, n_iters, graph) -> score
    # dist: (chunk, n) distances from a chunk of points to all points; density, stability: (n,) per candidate
    # graph: (chunk, n) rows of the UMAP fuzzy graph, aligned with dist
    # h: kernel width in use; score: (chunk, n), finite; larger = more attractive;
    # <= 0 means "no attraction" (only used to fill remaining slots)

weight_fn(dist, density, stability, h, iteration, n_iters, graph) -> weights
    # dist, density, stability, graph: (n, k), for the k selected neighbours of every point
    # (dist = their CURRENT distances in the selection space, stability = their current scores,
    #  graph = fuzzy-graph weight between each point and each selected neighbour, 0 if there is no such edge)
    # weights: (n, k), finite and >= 0; normalised per row afterwards

stability_fn(step, density, stability, h, iteration, n_iters, graph) -> stability
    # step: (n, d) displacement of every point in the iteration that just finished; density, stability: (n,)
    # graph: the full (n, n) sparse fuzzy graph
    # returns (n,) scores in [0, 1]
```

| function | called | `iteration` is |
| --- | --- | --- |
| `score_fn` | whenever neighbours are selected: at the start, and every `recompute_neighbors` iterations after | the iteration whose step will use the selected neighbours |
| `weight_fn` | at the start of every iteration, after any selection | the iteration whose step will use the weights |
| `stability_fn` | after every iteration | the iteration that just finished |

**The fuzzy graph.** The density is computed from a UMAP-style fuzzy `n_neighbors`-nearest-neighbour graph of the initial points (of the whitened points if `whiten_density`): every point has a per-point scale, an edge to each of its `n_neighbors` nearest points, the two directions of a pair are combined by fuzzy union, and edges weaker than `max / n_epochs` are dropped. Weights are in `(0, 1]`. There is one entry per directed nearest-neighbour edge (about `n_neighbors` per point), so the matrix is **not** symmetric: on 3,240 ARMD OCT embeddings (CLIP) about 58% of the edges also exist in the reverse direction. It is computed **once**, before the first iteration, and is the same for the whole run. Each function gets the view that fits its other arguments: the `(chunk, n)` dense rows aligned with `dist` for `score_fn` (0 where there is no edge), the `(n, k)` values for the selected `(point, neighbour)` pairs for `weight_fn` (0 where a selected neighbour is not a graph neighbour; this differs from the graph neighbours because selection also looks at density), and the whole `(n, n)` coalesced sparse tensor for `stability_fn`.

Within an iteration the order is: select neighbours (if due) → weights → step → stability update. A function with an older signature (without `iteration`, `n_iters` or `graph`) raises a `TypeError`. `dist` is measured in the selection space: Euclidean on the sphere, or whitened (Mahalanobis) if `whiten_selection`; for `weight_fn` it is recomputed from the current positions every iteration (with whitening, the transform from the last selection is reused). Shape or value violations raise `ValueError`. Examples:

```python
from msde import MeanShiftDensityEnhancement, NeighborScore, NeighborWeight
import torch

model = MeanShiftDensityEnhancement(k=30, recompute_neighbors=1)

# weaker emphasis on distance, stronger on stability, chosen per call
shifted, _, _ = model(X, score_fn=NeighborScore(distance_strength=0.5, density_strength=1, stability_strength=2))

# a custom score with its own parameter: ignore distance beyond a cutoff
def cutoff_score(dist, density, stability, h, iteration, n_iters, graph, cutoff=3.0):
    return density * stability * (dist < cutoff * h)

# distance-aware neighbour weights: attraction decays with the points' current distance
def gravity_weight(dist, density, stability, h, iteration, n_iters, graph):
    return density * stability * torch.exp(-dist.square() / (2 * h * h))
shifted, _, _ = model(X, weight_fn=gravity_weight)

# a schedule: let the distance matter more and more as the run progresses
class AnnealedScore:
    def __init__(self, start=0.2, end=3.0):
        self.start, self.end = start, end
    def __call__(self, dist, density, stability, h, iteration, n_iters, graph):
        strength = self.start + (self.end - self.start) * iteration / max(n_iters - 1, 1)
        return NeighborScore(distance_strength=strength)(dist, density, stability, h, iteration, n_iters, graph)
shifted, _, _ = model(X, score_fn=AnnealedScore())

# using the fuzzy graph: neighbours that are also graph neighbours count extra
def connected_weight(dist, density, stability, h, iteration, n_iters, graph):
    return density * stability * (0.1 + graph)
shifted, _, _ = model(X, weight_fn=connected_weight)

# a stability that compares a point's step with the average step of its graph neighbours (a sparse matmul)
def moves_like_neighbours(step, density, stability, h, iteration, n_iters, graph):
    weight_sum = torch.sparse.mm(graph, torch.ones(step.shape[0], 1, device=step.device)).clamp_min(1e-9)
    local = torch.sparse.mm(graph, step) / weight_sum
    return torch.exp(-(step - local).norm(dim=1) / step.norm(dim=1).median().clamp_min(1e-9))
shifted, _, _ = model(X, stability_fn=moves_like_neighbours)
```

With `recompute_neighbors=0` the neighbour graph is fixed after the first selection, so a `score_fn` is called once; the weights and the stability keep following the moving points through `weight_fn` and `stability_fn`.

### Stability

Every point starts with stability 1.0. A `stability_fn` (a callable that carries its own parameters) updates the scores **after every iteration** from what the points just did; its signature is given above. `step` is the displacement each point made in the iteration that just finished (new unit position minus old; zero for a point left in place), `stability` the scores before this update (all 1.0 at the start), so running updates are possible.

- **Timing.** The first neighbour selection and the first step use scores of 1.0, since there is no step yet. After that the updated scores feed the neighbour weights of the next iteration (re-evaluated every iteration) and, whenever neighbours are reselected (`recompute_neighbors > 0`), the selection score. With `recompute_neighbors=0` the neighbour graph itself is fixed by the initial scores, but the weights on it still follow the stability.
- **Result.** The final scores are available afterwards as `model.stability_` (all 1.0 without a `stability_fn`).
- **A fixed set of scores** can be supplied with a closure, `stability_fn=lambda step, density, stability, h, iteration, n_iters, graph: my_scores`; they take effect from the second iteration on.
- **Cost.** Per iteration a `stability_fn` adds one call, and the weights are rebuilt (live distances `O(n k d)`, plus `O(n d^2)` for the whitening transform if `whiten_selection`, and the sparse weight matrix `O(n k log(n k))`). A `weight_fn` costs the same without the call. Nothing is added when neither is given, and the results are then identical to a run without these features.

```python
import torch

def movers_unstable(step, density, stability, h, iteration, n_iters, graph):
    # a point that moves much more than the typical point in an iteration is unstable
    size = step.norm(dim=1)
    return torch.exp(-size / size.median().clamp_min(1e-9))

class RunningStability:
    # an exponential moving average of the same score, with its own parameter
    def __init__(self, momentum=0.5):
        self.momentum = momentum
    def __call__(self, step, density, stability, h, iteration, n_iters, graph):
        size = step.norm(dim=1)
        target = torch.exp(-size / size.median().clamp_min(1e-9))
        return (1 - self.momentum) * stability + self.momentum * target

model = MeanShiftDensityEnhancement(k=30, recompute_neighbors=1)
shifted, movement, _ = model(X, stability_fn=RunningStability(momentum=0.3))
scores = model.stability_            # (n,) in [0, 1]
```

## The ridge

`whiten_regularization` is the **ridge** of the whitening. Whitening divides every principal direction of the data by its standard deviation, `1 / sqrt(lambda_i)` for eigenvalue `lambda_i`, so that all directions get unit variance. Directions with a tiny eigenvalue (mostly noise, in typical embeddings) would be stretched enormously and end up as important as the real structure. The ridge adds a floor to the eigenvalues before dividing:

```
scale_i = 1 / sqrt(lambda_i + r * mean(lambda))        r = whiten_regularization
```

- `r` small (the default `1e-6`): essentially full whitening; every direction with an eigenvalue above `r * mean(lambda)` gets unit weight, however small it is.
- `r` large (1 or more): the floor is at least as large as the average eigenvalue, so weak directions are no longer amplified and whitening fades towards no whitening; as `r` grows, all directions are scaled by about the same factor and distances become proportional to Euclidean ones. Intermediate values give partial whitening: strong directions are equalised, weak ones stay weak.
- It also keeps the computation finite when there are fewer points than dimensions or constant features (zero eigenvalues).

It has no effect unless `whiten_density` or `whiten_selection` is on.

## Speed

The two heavy matrix products are sparse matmuls, kept on purpose:

- the density's radius counting (`S @ chunk.T`, about 20 passes), which dominates the density cost;
- the weighted mean `W @ X` of every iteration.

If a device cannot do sparse matmul (probed once per device), the weighted mean falls back to a chunked gather. Measured for one weighted-mean step with `k = 50`:

| device | `n` | `d` | sparse `W @ X` | gather | gather / sparse |
| --- | --- | --- | --- | --- | --- |
| CPU | 3,240 | 512 | 34 ms | 235 ms | 7.0x |
| CPU | 3,240 | 1152 | 88 ms | 528 ms | 6.0x |
| CPU | 16,000 | 512 | 362 ms | 1164 ms | 3.2x |
| Apple MPS | 3,240 | 512 | 2.7 ms | 45 ms | 17x |
| Apple MPS | 3,240 | 1152 | 3.0 ms | 101 ms | 34x |
| Apple MPS | 16,000 | 512 | 11 ms | 222 ms | 20x |

Cost model (`n` points of dimension `d`; `P` about 20 counting passes, `q` = `n_neighbors`):

- whitening (only if a flag is set): `O(n d^2 + d^3)` in float64 on the CPU, once for the density and at every neighbour reselection for the selection;
- density: `O(n^2 d)` for the k-NN graph plus `O(P n^2 q)` for the counting passes, once;
- selection: `O(n^2 d)` time and `O(chunk_size * n)` working memory per selection (plus one extra k-NN pass when `h="auto"`);
- each iteration: one sparse matmul, `O(n k d)`.

With the defaults (no whitening), 402 of 405 benchmark runs (27 datasets x 5 encoders x 3 seeds, 200-4,000 points, `d` 512-1152) converged within the cap: a median of 50 iterations, 90th percentile 164, and some cells needing several hundred (the largest converged run took 988). The three that did not converge were the same dataset/encoder pair and had a final mean movement of about 1.5e-4, just above the threshold. About 19% of runs needed more than 100 iterations, so a cap of 100 stopped them early, and for some cells that changed the resulting scores noticeably. On an Apple MPS device the average run took about 0.7 s (whitening off). `torch.compile` is used for the k-NN, graph and distance kernels, so the first call for a new shape includes compilation time.

## Things to know

- **The movement can be informative in either direction.** If you use `movement` as an anomaly score, its AUROC can be well below 0.5 (anomalies move *less*) for some encoders and above 0.5 for others; judge it by `max(auroc, 1 - auroc)` and fix the direction per encoder.
- **Density is computed once**, from the initial points (whitened if `whiten_density`), and is not refreshed by `recompute_neighbors`.
- **Whitening needs `n` comfortably above `d`** (only relevant when it is on). With fewer points than dimensions the covariance is singular and the ridge decides the scale of the null directions; the result is finite but noisy. Embeddings with a very wide eigenvalue spread (1e8 and beyond) are handled in float64, but the smallest directions are still governed by `whiten_regularization`.
- **`h` is in the units of the selection space.** With `whiten_selection` a numeric `h` is not comparable with raw embedding distances; `"auto"` adapts to the data either way.
- **Stability is computed during the run, not supplied as an array.** It starts at 1.0 and changes only if you pass a `stability_fn`; the first selection and step always use 1.0. To use scores computed elsewhere, return them from a `stability_fn` (they then act from the second iteration).
- **A point whose weights are all zero stays put** after the step, up to float rounding (its mean is the zero vector, so the step scales it by `1 - learning_rate` and re-normalisation restores its direction).
- **Small differences across devices.** CPU and MPS agree to about `1e-4` on the converged output; the movement AUROC differed by 0.003 in one run (iterated float32 arithmetic).
