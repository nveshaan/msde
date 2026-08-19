import time
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.dmsl import mean_shift_manifold_learning
from src.msde import MeanShiftDensityEnhancement

# Select Apple Silicon MPS device if available
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
MAX_ITER = 100
DIMS_NP = np.array([3, 8, 16, 32, 64, 128, 512, 1024])


def build_manifold_torch(points_torch: torch.Tensor, dim: int) -> torch.Tensor:
    base = torch.stack(
        [torch.sin(points_torch), torch.cos(points_torch), points_torch], dim=1
    )
    repeats = math.ceil(dim / base.shape[1])
    return base.repeat(1, repeats)[:, :dim]


def build_manifold_numpy(points_np: np.ndarray, dim: int) -> np.ndarray:
    base = np.stack([np.sin(points_np), np.cos(points_np), points_np], axis=1)
    repeats = math.ceil(dim / base.shape[1])
    return np.tile(base, (1, repeats))[:, :dim]

msde = MeanShiftDensityEnhancement(
    k=50,
    max_iters_shift=MAX_ITER,
    device=device,
    keep_trajectory=True,
    enable_gradients=False,
)

# 1. Define sweep ranges explicitly
DIMS_LIST = DIMS_NP.tolist()
n_points_np = np.array([64, 128, 512, 1024, 4096, 4096 * 2, 4096 * 4])
n_points_list = n_points_np.tolist()
n_dims = len(DIMS_LIST)
n_sizes = len(n_points_list)

# 2. Containers for benchmark metrics (dim x n_points)
results = {
    "standard": {
        "time": np.full((n_dims, n_sizes), np.nan, dtype=np.float64),
        "iters": np.full((n_dims, n_sizes), np.nan, dtype=np.float64),
        "time_per_step": np.full((n_dims, n_sizes), np.nan, dtype=np.float64),
    },
    "pytorch": {
        "time": np.full((n_dims, n_sizes), np.nan, dtype=np.float64),
        "iters": np.full((n_dims, n_sizes), np.nan, dtype=np.float64),
        "time_per_step": np.full((n_dims, n_sizes), np.nan, dtype=np.float64),
    },
}

# Warmup run (smallest dim and n) for fairer timing
warmup_dim = DIMS_LIST[0]
warmup_n = n_points_list[0]
points_torch = torch.linspace(0, 5 * math.pi, warmup_n, device=device)
manifold_tensor = build_manifold_torch(points_torch, warmup_dim)
points_np = np.linspace(0, 5 * np.pi, warmup_n)
manifold_np = build_manifold_numpy(points_np, warmup_dim)

_, _, _, _ = mean_shift_manifold_learning(
    manifold_np, k=50, max_iters_shift=500
)

if device.type == "mps":
    torch.mps.synchronize()

_, _, _ = msde(manifold_tensor)

if device.type == "mps":
    torch.mps.synchronize()

shifted_std_last = None
shifted_pt_last = None

for dim_idx, dim in enumerate(DIMS_LIST):
    print(f"\n===== Benchmarking dim = {dim} =====")

    for size_idx, n in enumerate(n_points_list):
        print(f"Benchmarking dim={dim}, N={n}...")

        # Data generation
        # --- PyTorch Tensor for MSDE ---
        points_torch = torch.linspace(0, 5 * math.pi, n, device=device)
        manifold_tensor = build_manifold_torch(points_torch, dim)

        # --- NumPy Array for DMSL ---
        points_np = np.linspace(0, 5 * np.pi, n)
        manifold_np = build_manifold_numpy(points_np, dim)

        # --- Standard (DMSL) using NumPy array ---
        start = time.time()
        shifted_std, _, _, traj_std = mean_shift_manifold_learning(
            manifold_np, k=50, max_iters_shift=MAX_ITER
        )
        elapsed_std = time.time() - start
        iters_std = len(traj_std)

        results["standard"]["time"][dim_idx, size_idx] = elapsed_std
        results["standard"]["iters"][dim_idx, size_idx] = iters_std
        results["standard"]["time_per_step"][dim_idx, size_idx] = (
            elapsed_std / iters_std if iters_std else float("nan")
        )

        # --- PyTorch (MSDE) using selected device ---
        if device.type == "mps":
            torch.mps.synchronize()

        start = time.time()
        shifted_pt, _, traj_pt = msde(manifold_tensor)

        if device.type == "mps":
            torch.mps.synchronize()

        elapsed_pt = time.time() - start
        iters_pt = len(traj_pt)

        results["pytorch"]["time"][dim_idx, size_idx] = elapsed_pt
        results["pytorch"]["iters"][dim_idx, size_idx] = iters_pt
        results["pytorch"]["time_per_step"][dim_idx, size_idx] = (
            elapsed_pt / iters_pt if iters_pt else float("nan")
        )

        shifted_std_last = shifted_std
        shifted_pt_last = shifted_pt

# 3. Calculate and print speedup metrics
std_tps = results["standard"]["time_per_step"]
pt_tps = results["pytorch"]["time_per_step"]
speedups_total = np.divide(
    std_tps,
    pt_tps,
    out=np.full_like(std_tps, np.nan),
    where=np.isfinite(pt_tps) & (pt_tps > 0),
)

valid_speedups = speedups_total[np.isfinite(speedups_total) & (speedups_total > 0)]
if valid_speedups.size > 0:
    mean_speedup = float(np.mean(valid_speedups))
    geom_mean_speedup = float(np.exp(np.mean(np.log(valid_speedups))))
    max_flat_idx = int(np.nanargmax(speedups_total))
    max_dim_idx, max_size_idx = np.unravel_index(max_flat_idx, speedups_total.shape)
    max_speedup_val = float(speedups_total[max_dim_idx, max_size_idx])
else:
    mean_speedup = float("nan")
    geom_mean_speedup = float("nan")
    max_dim_idx, max_size_idx = 0, 0
    max_speedup_val = float("nan")

print("\n" + "=" * 45)
for dim_idx, dim in enumerate(DIMS_LIST):
    for size_idx, n in enumerate(n_points_list):
        s = speedups_total[dim_idx, size_idx]
        print(f"dim={dim:<4} N={n:<6}: {s:.2f}x speedup")
print("-" * 45)
print(f"Arithmetic Mean Speedup : {mean_speedup:.2f}x")
print(f"Geometric Mean Speedup  : {geom_mean_speedup:.2f}x")
print(
    "Maximum Speedup "
    f"(dim={DIMS_LIST[max_dim_idx]}, N={n_points_list[max_size_idx]}): "
    f"{max_speedup_val:.2f}x"
)
print("=" * 45 + "\n")

# 4. Visualization & Saving
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

def plot_heatmap(ax: plt.Axes, data: np.ndarray, title: str, cbar_label: str, cmap: str) -> None:
    image = ax.imshow(data, aspect="auto", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("Number of Points (N)")
    ax.set_ylabel("Dimension")
    ax.set_xticks(np.arange(n_sizes))
    ax.set_xticklabels(n_points_list)
    ax.set_yticks(np.arange(n_dims))
    ax.set_yticklabels(DIMS_LIST)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(cbar_label)


plot_heatmap(
    axes[0, 0],
    np.log10(np.maximum(results["standard"]["time_per_step"], 1e-16)),
    "log10 Time/Step: Standard (DMSL)",
    "log10(seconds / step)",
    "viridis",
)
plot_heatmap(
    axes[0, 1],
    np.log10(np.maximum(results["pytorch"]["time_per_step"], 1e-16)),
    "log10 Time/Step: PyTorch (MSDE)",
    "log10(seconds / step)",
    "plasma",
)
plot_heatmap(
    axes[1, 0],
    speedups_total,
    "Speedup Heatmap (Standard / PyTorch)",
    "speedup (x)",
    "cividis",
)

speedup_per_dim = np.nanmean(speedups_total, axis=1)
axes[1, 1].plot(DIMS_LIST, speedup_per_dim, "o-")
axes[1, 1].set_title("Mean Speedup vs Dimension")
axes[1, 1].set_xlabel("Dimension")
axes[1, 1].set_ylabel("Mean speedup (x)")
axes[1, 1].grid(True, ls="--", alpha=0.5)

plt.tight_layout()
output_file = "mean_shift_benchmark.png"
plt.savefig(output_file, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Benchmark plot saved to '{output_file}'.")

# 5. Interactive 3D comparison for the final benchmark combo (first 3 dims only)
shifted_std_plot = np.asarray(shifted_std_last)[:, :3]
shifted_pt_plot = shifted_pt_last.detach().cpu().numpy()[:, :3]

interactive_fig = make_subplots(
    rows=1,
    cols=2,
    specs=[[{"type": "scene"}, {"type": "scene"}]],
    subplot_titles=("Shifted Standard (DMSL)", "Shifted PyTorch (MSDE)"),
)

interactive_fig.add_trace(
    go.Scatter3d(
        x=shifted_std_plot[:, 0],
        y=shifted_std_plot[:, 1],
        z=shifted_std_plot[:, 2],
        mode="markers",
        marker={"size": 3, "color": shifted_std_plot[:, 2], "colorscale": "Viridis"},
        name="DMSL",
    ),
    row=1,
    col=1,
)
interactive_fig.add_trace(
    go.Scatter3d(
        x=shifted_pt_plot[:, 0],
        y=shifted_pt_plot[:, 1],
        z=shifted_pt_plot[:, 2],
        mode="markers",
        marker={"size": 3, "color": shifted_pt_plot[:, 2], "colorscale": "Plasma"},
        name="MSDE",
    ),
    row=1,
    col=2,
)

interactive_fig.update_layout(
    title="Interactive 3D Shifted Manifold Comparison",
    scene={"xaxis_title": "X", "yaxis_title": "Y", "zaxis_title": "Z"},
    scene2={"xaxis_title": "X", "yaxis_title": "Y", "zaxis_title": "Z"},
)
interactive_output_file = "shifted_manifold_comparison.html"
interactive_fig.write_html(interactive_output_file, auto_open=True)
print(f"Interactive 3D plot saved to '{interactive_output_file}'.")