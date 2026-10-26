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
3. **Density** of every point, computed **once** on the initial points (whitened if `whiten_density`):
   - build an exact `n_neighbors`-nearest-neighbour graph and turn it into a sparse fuzzy graph (per-point scale found by bisection, symmetrised by fuzzy union, weak edges dropped);
   - compare points as rows of that sparse graph: distances come from one sparse-times-dense matmul `S @ chunk.T` per chunk of rows;
   - find the smallest radius at which at least `satisfiability_proportion` of the points have more than `nbd_sample_count_threshold` others inside it (bisection, relaxed once if it fails);
   - a point's density is its number of other points within the radius, averaged over `max_iters_weight_count` radii shrinking linearly to zero.
4. **Select neighbours.** In the geometry of the current points (Euclidean on the sphere, or Mahalanobis if `whiten_selection`), every other point is scored by `NeighborScore` and the `k` best are kept: with the default score `density^a * stability^b * exp(-c * d^2 / (2 h^2))`, so a dense, stable point can outrank a closer, sparse one. All points are candidates (no distance pre-filter) and a point is never its own neighbour. Candidates with a positive score rank by score; if fewer than `k` have one, the remaining slots go to the nearest candidates.
5. **Weights.** Each selected neighbour gets a weight from `NeighborWeight` (default `density^a * stability^b`), evaluated at selection time; weights of picks that violate the label mask are zeroed and rows are normalised to sum to 1.
6. **Step.** `mean = W @ X` with `W` a sparse `(n, n)` matrix holding the `n*k` normalised weights (one sparse matmul), then `x' = normalise(x + learning_rate * gate * (mean - x))`.
7. **Repeat** until the mean per-point movement on the sphere falls below `shift_threshold` (convergence) or `max_iters_shift` iterations have run. With `recompute_neighbors = N > 0`, stages 2, 4 and 5 are redone every `N` iterations on the current points (iterations 0, N, 2N, ...); the density is never recomputed.

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

## `forward(X, gate=1.0, labels=None, stability=None, score_fn=None, weight_fn=None)`

| Argument | Meaning |
| --- | --- |
| `X` | `(n, d)` floating-point embeddings. |
| `gate` | Multiplier on the step size for this call (0 leaves the points where they are). |
| `labels` | `(n,)` integer labels restricting neighbour selection (see `label_mode`). A point with no valid neighbour is left where it is. |
| `stability` | `(n,)` scores in `[0, 1]`, higher = more stable. Default: 1.0 for every point. Fixed for the whole run; never updated by it. |
| `score_fn` | Replacement for the selection score. Used for **every** neighbour selection, including the initial one, so with `recompute_neighbors=0` it defines the single neighbour graph the run uses. |
| `weight_fn` | Replacement for the neighbour weights of the weighted mean. Optional; the default is used when it is not given. |

### Replacing the selection score and the neighbour weights

Both are plain callables that carry their own parameters; the defaults are the callable classes `NeighborScore(distance_strength, density_strength, stability_strength)` and `NeighborWeight(density_strength, stability_strength)`.

```python
score_fn(dist, density, stability, h) -> score        # dist: (chunk, n) distances from a chunk of points to all points
                                                      # density, stability: (n,) per candidate; h: kernel width in use
                                                      # score: (chunk, n), finite; larger = more attractive;
                                                      # <= 0 means "no attraction" (only used to fill remaining slots)

weight_fn(dist, density, stability, h) -> weights     # all (n, k): the k selected neighbours of every point
                                                      # (dist = distances in the selection space at selection time)
                                                      # weights: (n, k), finite and >= 0; normalised per row afterwards
```

`dist` is measured in the selection space: Euclidean on the sphere, or whitened (Mahalanobis) if `whiten_selection`. Shape or value violations raise `ValueError`. Examples:

```python
from msde import MeanShiftDensityEnhancement, NeighborScore, NeighborWeight
import torch

model = MeanShiftDensityEnhancement(k=30, recompute_neighbors=1)

# stronger emphasis on stability, weaker on distance, chosen per call
shifted, _, _ = model(X, stability=s, score_fn=NeighborScore(distance_strength=0.5, density_strength=1, stability_strength=2))

# a custom score with its own parameter: ignore distance beyond a cutoff
def cutoff_score(dist, density, stability, h, cutoff=3.0):
    return density * stability * (dist < cutoff * h)

# distance-aware neighbour weights (attraction decays with distance); refreshed whenever neighbours are reselected
def gravity_weight(dist, density, stability, h):
    return density * stability * torch.exp(-dist.square() / (2 * h * h))
shifted, _, _ = model(X, weight_fn=gravity_weight)
```

`weight_fn` is evaluated at each neighbour selection, with the distances of that moment. With `recompute_neighbors=0` that is once, at the start; to let distance-dependent weights follow the moving points, enable `recompute_neighbors`.

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
- **Stability is an input, not an output.** The run never changes it. To use a computed stability (for example from a first MSDE pass), pass it in `stability=`.
- **A point whose weights are all zero stays put** after the step, up to float rounding (its mean is the zero vector, so the step scales it by `1 - learning_rate` and re-normalisation restores its direction).
- **Small differences across devices.** CPU and MPS agree to about `1e-4` on the converged output; the movement AUROC differed by 0.003 in one run (iterated float32 arithmetic).
