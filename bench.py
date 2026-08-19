import time
import math
import torch
import numpy as np
import matplotlib.pyplot as plt

from src.dmsl import mean_shift_manifold_learning
from src.msde import MeanShiftDensityEnhancement

# Select Apple Silicon MPS device if available
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

msde = MeanShiftDensityEnhancement(
    k=50,
    max_iters_shift=500,
    device=device,
    keep_trajectory=True,
    enable_gradients=False
)

# 1. Define point counts explicitly
n_points_np = np.array([64, 128, 512, 1024, 4096, 4096 * 2, 4096 * 4])
n_points_list = n_points_np.tolist()

# 2. Containers for benchmark metrics
results = {
    "standard": {"time": [], "iters": [], "time_per_step": []},
    "pytorch": {"time": [], "iters": [], "time_per_step": []},
}

for n in [64]:
    # Data generation
    # --- PyTorch Tensor for MSDE ---
    points_torch = torch.linspace(0, 5 * math.pi, n, device=device)
    manifold_tensor = torch.stack(
        [torch.sin(points_torch), torch.cos(points_torch), points_torch], dim=1
    )

    # --- NumPy Array for DMSL ---
    points_np = np.linspace(0, 5 * np.pi, n)
    manifold_np = np.stack(
        [np.sin(points_np), np.cos(points_np), points_np], axis=1
    )

    # --- Standard (DMSL) using NumPy array ---
    _, _, _, traj_std = mean_shift_manifold_learning(
        manifold_np, k=50, max_iters_shift=500
    )

    # --- PyTorch (MSDE) using MPS Tensor ---
    if device.type == "mps":
        torch.mps.synchronize()

    _, _, traj_pt = msde(manifold_tensor)

    if device.type == "mps":
        torch.mps.synchronize()

for n in n_points_list:
    print(f"Benchmarking N = {n}...")

    # Data generation
    # --- PyTorch Tensor for MSDE ---
    points_torch = torch.linspace(0, 5 * math.pi, n, device=device)
    manifold_tensor = torch.stack(
        [torch.sin(points_torch), torch.cos(points_torch), points_torch], dim=1
    )

    # --- NumPy Array for DMSL ---
    points_np = np.linspace(0, 5 * np.pi, n)
    manifold_np = np.stack(
        [np.sin(points_np), np.cos(points_np), points_np], axis=1
    )

    # --- Standard (DMSL) using NumPy array ---
    start = time.time()
    _, _, _, traj_std = mean_shift_manifold_learning(
        manifold_np, k=50, max_iters_shift=500
    )
    elapsed_std = time.time() - start
    iters_std = len(traj_std)

    results["standard"]["time"].append(elapsed_std)
    results["standard"]["iters"].append(iters_std)
    results["standard"]["time_per_step"].append(
        elapsed_std / iters_std if iters_std else float("nan")
    )

    # --- PyTorch (MSDE) using MPS Tensor ---
    if device.type == "mps":
        torch.mps.synchronize()

    start = time.time()
    _, _, traj_pt = msde(manifold_tensor)

    if device.type == "mps":
        torch.mps.synchronize()

    elapsed_pt = time.time() - start
    iters_pt = len(traj_pt)

    results["pytorch"]["time"].append(elapsed_pt)
    results["pytorch"]["iters"].append(iters_pt)
    results["pytorch"]["time_per_step"].append(
        elapsed_pt / iters_pt if iters_pt else float("nan")
    )

# 3. Calculate and print speedup metrics using PyTorch
std_times = torch.tensor(results["standard"]["time_per_step"], dtype=torch.float64)
pt_times = torch.tensor(results["pytorch"]["time_per_step"], dtype=torch.float64)
speedups_total = std_times / pt_times

mean_speedup = torch.mean(speedups_total).item()
geom_mean_speedup = torch.exp(torch.mean(torch.log(speedups_total))).item()
max_speedup_val, max_idx = torch.max(speedups_total, dim=0)

print("\n" + "=" * 45)
for n, s in zip(n_points_list, speedups_total.tolist()):
    print(f"N = {n:<6}: {s:.2f}x speedup")
print("-" * 45)
print(f"Arithmetic Mean Speedup : {mean_speedup:.2f}x")
print(f"Geometric Mean Speedup  : {geom_mean_speedup:.2f}x")
print(f"Maximum Speedup (N={n_points_list[max_idx.item()]}): {max_speedup_val.item():.2f}x")
print("=" * 45 + "\n")

# 4. Visualization & Saving
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Total Runtime
axes[0].plot(n_points_list, results["standard"]["time"], "o-", label="Standard (DMSL)")
axes[0].plot(n_points_list, results["pytorch"]["time"], "s--", label="PyTorch (MSDE MPS)")
axes[0].set_xscale("log", base=2)
axes[0].set_yscale("log")
axes[0].set_title("Total Execution Time")
axes[0].set_xlabel("Number of Points (N)")
axes[0].set_ylabel("Time (seconds)")
axes[0].set_xticks(n_points_list)
axes[0].get_xaxis().set_major_formatter(plt.ScalarFormatter())
axes[0].grid(True, which="both", ls="--", alpha=0.5)
axes[0].legend()

# Plot 2: Total Iterations
axes[1].plot(n_points_list, results["standard"]["iters"], "o-", label="Standard (DMSL)")
axes[1].plot(n_points_list, results["pytorch"]["iters"], "s--", label="PyTorch (MSDE MPS)")
axes[1].set_xscale("log", base=2)
axes[1].set_title("Convergence Iterations")
axes[1].set_xlabel("Number of Points (N)")
axes[1].set_ylabel("Iterations (len(traj))")
axes[1].set_xticks(n_points_list)
axes[1].get_xaxis().set_major_formatter(plt.ScalarFormatter())
axes[1].grid(True, which="both", ls="--", alpha=0.5)
axes[1].legend()

# Plot 3: Time Per Step
axes[2].plot(n_points_list, results["standard"]["time_per_step"], "o-", label="Standard (DMSL)")
axes[2].plot(n_points_list, results["pytorch"]["time_per_step"], "s--", label="PyTorch (MSDE MPS)")
axes[2].set_xscale("log", base=2)
axes[2].set_yscale("log")
axes[2].set_title("Time per Iteration Step")
axes[2].set_xlabel("Number of Points (N)")
axes[2].set_ylabel("Seconds / Step")
axes[2].set_xticks(n_points_list)
axes[2].get_xaxis().set_major_formatter(plt.ScalarFormatter())
axes[2].grid(True, which="both", ls="--", alpha=0.5)
axes[2].legend()

plt.tight_layout()
output_file = "mean_shift_benchmark.png"
plt.savefig(output_file, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Benchmark plot saved to '{output_file}'.")