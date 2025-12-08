# %% [markdown]
# # Annealing Optimization Demo
#
# This notebook demonstrates the sequential quadratic programming (SQP) solver with
# simulated annealing for fault slip rate inversion with coupling constraints.
#
# The annealing approach helps find better solutions by iteratively:
# 1. Solving a convex relaxation of the problem
# 2. Tightening bounds towards feasibility
# 3. Occasionally loosening bounds to escape local minima

# %%
# Configuration for high-DPI displays and parallel processing
import os

os.environ["RAYON_NUM_THREADS"] = "4"

# For Jupyter/IPython, uncomment these lines:
# %config InlineBackend.figure_format = "retina"

# %%
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

import celeri
from celeri.plot import plot_mesh

# Suppress verbose logging during optimization
logger.disable("celeri")

# %% [markdown]
# ## Load the Japan model
#
# Build the model and operators from the Japan configuration file.

# %%
from pathlib import Path

# Get the directory of this script
script_dir = Path(__file__).parent.resolve()
config_file = script_dir / "../data/config/japan_config.json"

print("Building model...")
model = celeri.build_model(config_file)

print(f"Model has {len(model.meshes)} meshes")
print(f"Total mesh points: {model.total_mesh_points}")
print(f"Segment mesh indices: {model.segment_mesh_indices}")

# %%
# Build operators (this is the slow step)
print("Building operators...")
operators = celeri.build_operators(model, eigen=True, tde=True)
print("Operators ready!")

# %% [markdown]
# ## Run SQP2 solver with annealing
#
# The `solve_sqp2` function runs the sequential quadratic programming solver.
# With annealing enabled, it performs multiple passes to escape local minima.

# %%
# Solver configuration
solve_kwargs = dict(
    solver="CLARABEL",
    equilibrate_enable=False,
    direct_solve_method="faer",
    ignore_dpp=True,
)

# Run optimization with annealing
print("Running SQP2 optimization with annealing...")
estimation = celeri.solve_sqp2(
    model,
    verbose=True,
    max_iter=100,
    solve_kwargs=solve_kwargs,
    objective="qr_sum_of_squares",
    operators=operators,
    annealing_enabled=True,
    annealing_schedule=[0.125, 0.125, 0.125],
)

print(f"\nOptimization complete!")
print(f"Total iterations: {len(estimation.trace.objective)}")
print(f"Final out-of-bounds: {estimation.trace.out_of_bounds[-1]}")

# %% [markdown]
# ## Plot convergence
#
# Visualize how the objective function and constraint violations evolve
# during optimization.

# %%
trace = estimation.trace

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Objective function
ax = axes[0, 0]
ax.semilogy(trace.objective_norm2, "b.-")
ax.set_xlabel("Iteration")
ax.set_ylabel("Residual 2-norm")
ax.set_title("Objective Function")
ax.grid(True, alpha=0.3)

# Out-of-bounds count
ax = axes[0, 1]
ax.plot(trace.out_of_bounds, "r.-")
ax.set_xlabel("Iteration")
ax.set_ylabel("Count")
ax.set_title("Out-of-Bounds Velocities")
ax.grid(True, alpha=0.3)

# Non-convex constraint loss
ax = axes[1, 0]
losses = trace.nonconvex_constraint_loss
ax.semilogy([max(1e-10, loss) for loss in losses], "g.-")
ax.set_xlabel("Iteration")
ax.set_ylabel("Constraint Loss")
ax.set_title("Non-convex Constraint Violation")
ax.grid(True, alpha=0.3)

# Iteration time
ax = axes[1, 1]
ax.bar(range(len(trace.iter_time)), trace.iter_time, color="purple", alpha=0.7)
ax.set_xlabel("Iteration")
ax.set_ylabel("Time (s)")
ax.set_title(f"Iteration Time (total: {trace.total_time:.1f}s)")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("convergence.png", dpi=150)
plt.close()
print("Saved convergence.png")

# %% [markdown]
# ## Plot estimation summary
#
# Overview of observed vs. modeled velocities and estimated slip rates.

# %%
celeri.plot_estimation_summary(model, estimation)
plt.savefig("estimation_summary.png", dpi=150)
plt.close()
print("Saved estimation_summary.png")

# %% [markdown]
# ## Plot TDE slip rates on meshes
#
# Visualize the estimated slip rates on each triangular mesh element.


# %%
def plot_tde_slip_rates(estimation, component="strike_slip"):
    """Plot TDE slip rates for all meshes."""
    meshes = estimation.model.meshes
    if component == "strike_slip":
        rates = estimation.tde_strike_slip_rates
        title = "Strike-Slip Rates (mm/yr)"
    else:
        rates = estimation.tde_dip_slip_rates
        title = "Dip-Slip Rates (mm/yr)"

    if rates is None:
        print("No TDE rates available")
        return

    # Calculate global color range
    all_values = np.concatenate([np.ravel(vals) for vals in rates.values()])
    vmax = np.max(np.abs(all_values))
    vmin = -vmax

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot each mesh
    pc = None
    for mesh_idx, mesh in enumerate(meshes):
        pc = plot_mesh(mesh, fill_value=rates[mesh_idx], ax=ax, vmin=vmin, vmax=vmax)

    # Plot segments
    segment = estimation.model.segment
    for i in range(len(segment)):
        color = "k" if segment.dip.iloc[i] == 90.0 else "r"
        ax.plot(
            [segment.lon1.iloc[i], segment.lon2.iloc[i]],
            [segment.lat1.iloc[i], segment.lat2.iloc[i]],
            color,
            linewidth=0.5,
        )

    ax.set_xlim(estimation.model.config.lon_range)
    ax.set_ylim(estimation.model.config.lat_range)
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)

    if pc is not None:
        plt.colorbar(pc, ax=ax, shrink=0.7, label="mm/yr")

    plt.tight_layout()
    return fig


# %%
fig = plot_tde_slip_rates(estimation, "strike_slip")
if fig:
    fig.savefig("strike_slip_rates.png", dpi=150)
    plt.close(fig)
    print("Saved strike_slip_rates.png")

# %%
fig = plot_tde_slip_rates(estimation, "dip_slip")
if fig:
    fig.savefig("dip_slip_rates.png", dpi=150)
    plt.close(fig)
    print("Saved dip_slip_rates.png")

# %% [markdown]
# ## Plot coupling ratios
#
# The coupling ratio is the elastic slip rate divided by the kinematic slip rate.
# Values between 0 and 1 indicate partial coupling.


# %%
def plot_coupling_ratios(estimation, component="strike_slip"):
    """Plot coupling ratios for all meshes."""
    meshes = estimation.model.meshes
    # Only segment meshes have coupling data
    segment_mesh_indices = estimation.model.segment_mesh_indices

    if component == "strike_slip":
        coupling = estimation.tde_strike_slip_rates_coupling_smooth
        title = "Strike-Slip Coupling Ratio"
    else:
        coupling = estimation.tde_dip_slip_rates_coupling_smooth
        title = "Dip-Slip Coupling Ratio"

    if coupling is None:
        print("No coupling data available")
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    pc = None
    for mesh_idx in segment_mesh_indices:
        mesh = meshes[mesh_idx]
        # Clip coupling to [0, 1] for visualization
        values = np.clip(coupling[mesh_idx], 0, 1)
        # Use center=0.5 for coupling ratios (0 to 1 scale)
        pc = plot_mesh(
            mesh, fill_value=values, ax=ax, vmin=0, vmax=1, cmap="viridis", center=0.5
        )

    # Plot segments
    segment = estimation.model.segment
    for i in range(len(segment)):
        color = "k" if segment.dip.iloc[i] == 90.0 else "r"
        ax.plot(
            [segment.lon1.iloc[i], segment.lon2.iloc[i]],
            [segment.lat1.iloc[i], segment.lat2.iloc[i]],
            color,
            linewidth=0.5,
        )

    ax.set_xlim(estimation.model.config.lon_range)
    ax.set_ylim(estimation.model.config.lat_range)
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)

    if pc is not None:
        plt.colorbar(pc, ax=ax, shrink=0.7, label="Coupling")

    plt.tight_layout()
    return fig


# %%
fig = plot_coupling_ratios(estimation, "strike_slip")
if fig:
    fig.savefig("strike_slip_coupling.png", dpi=150)
    plt.close(fig)
    print("Saved strike_slip_coupling.png")

# %%
fig = plot_coupling_ratios(estimation, "dip_slip")
if fig:
    fig.savefig("dip_slip_coupling.png", dpi=150)
    plt.close(fig)
    print("Saved dip_slip_coupling.png")

# %% [markdown]
# ## Plot evolution of slip rates during optimization
#
# Visualize how the kinematic and elastic slip rates evolve through iterations.
# This shows the annealing process tightening bounds towards feasibility.


# %%
def plot_slip_rate_evolution(trace, mesh_idx=0, n_points=50):
    """Plot how slip rates evolve during optimization.

    Args:
        trace: MinimizerTrace from solve_sqp2
        mesh_idx: Index of mesh to visualize
        n_points: Number of random points to show trajectories for
    """
    n_iters = len(trace.slip_rates)
    if n_iters < 2:
        print("Not enough iterations to show evolution")
        return

    # Extract slip rates over iterations for this mesh
    ss_kinematic_history = []
    ss_elastic_history = []
    ds_kinematic_history = []
    ds_elastic_history = []

    for i in range(n_iters):
        slip_rate = trace.slip_rates[i][mesh_idx]
        ss_kinematic = slip_rate.strike_slip.kinematic_numpy(smooth=True)
        ss_elastic = slip_rate.strike_slip.elastic_numpy()
        ds_kinematic = slip_rate.dip_slip.kinematic_numpy(smooth=True)
        ds_elastic = slip_rate.dip_slip.elastic_numpy()

        if ss_kinematic is None or ds_kinematic is None:
            print(f"Mesh {mesh_idx} has no kinematic slip rates (not a segment mesh)")
            return

        ss_kinematic_history.append(ss_kinematic)
        ss_elastic_history.append(ss_elastic)
        ds_kinematic_history.append(ds_kinematic)
        ds_elastic_history.append(ds_elastic)

    # Convert to arrays: shape (n_iters, n_points)
    ss_kinematic_history = np.array(ss_kinematic_history)
    ss_elastic_history = np.array(ss_elastic_history)
    ds_kinematic_history = np.array(ds_kinematic_history)
    ds_elastic_history = np.array(ds_elastic_history)

    n_mesh_points = ss_kinematic_history.shape[1]
    n_points = min(n_points, n_mesh_points)

    # Randomly select points to visualize
    rng = np.random.default_rng(42)
    point_indices = rng.choice(n_mesh_points, size=n_points, replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Strike-slip
    ax = axes[0]
    for idx in point_indices:
        ax.plot(
            ss_kinematic_history[:, idx],
            ss_elastic_history[:, idx],
            ".-",
            alpha=0.3,
            markersize=2,
            linewidth=0.5,
        )
    # Final points
    ax.scatter(
        ss_kinematic_history[-1, point_indices],
        ss_elastic_history[-1, point_indices],
        c="red",
        s=10,
        zorder=10,
        label="Final",
    )
    # Reference lines for coupling = 0 and coupling = 1
    lim = max(
        np.abs(ss_kinematic_history).max() * 1.1, np.abs(ss_elastic_history).max() * 1.1
    )
    ax.plot([-lim, lim], [0, 0], "k--", alpha=0.3, label="coupling=0")
    ax.plot([-lim, lim], [-lim, lim], "k-", alpha=0.3, label="coupling=1")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Kinematic Strike-Slip (mm/yr)")
    ax.set_ylabel("Elastic Strike-Slip (mm/yr)")
    ax.set_title(f"Strike-Slip Evolution (mesh {mesh_idx})")
    ax.legend(loc="upper left")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Dip-slip
    ax = axes[1]
    for idx in point_indices:
        ax.plot(
            ds_kinematic_history[:, idx],
            ds_elastic_history[:, idx],
            ".-",
            alpha=0.3,
            markersize=2,
            linewidth=0.5,
        )
    # Final points
    ax.scatter(
        ds_kinematic_history[-1, point_indices],
        ds_elastic_history[-1, point_indices],
        c="red",
        s=10,
        zorder=10,
        label="Final",
    )
    # Reference lines
    lim = max(
        np.abs(ds_kinematic_history).max() * 1.1, np.abs(ds_elastic_history).max() * 1.1
    )
    ax.plot([-lim, lim], [0, 0], "k--", alpha=0.3, label="coupling=0")
    ax.plot([-lim, lim], [-lim, lim], "k-", alpha=0.3, label="coupling=1")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Kinematic Dip-Slip (mm/yr)")
    ax.set_ylabel("Elastic Dip-Slip (mm/yr)")
    ax.set_title(f"Dip-Slip Evolution (mesh {mesh_idx})")
    ax.legend(loc="upper left")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# %%
# Plot evolution for first mesh
fig = plot_slip_rate_evolution(trace, mesh_idx=0)
if fig:
    fig.savefig("slip_rate_evolution.png", dpi=150)
    plt.close(fig)
    print("Saved slip_rate_evolution.png")

# %% [markdown]
# ## Plot out-of-bounds evolution by mesh
#
# See how each mesh converges to feasibility.

# %%
oob_history = np.array(trace.out_of_bounds_detailed)  # shape: (n_iter, n_meshes, 2)

fig, ax = plt.subplots(figsize=(10, 5))

n_meshes = oob_history.shape[1]
for mesh_idx in range(n_meshes):
    # Sum strike-slip and dip-slip out-of-bounds
    total_oob = oob_history[:, mesh_idx, 0] + oob_history[:, mesh_idx, 1]
    ax.plot(total_oob, ".-", label=f"Mesh {mesh_idx}")

ax.set_xlabel("Iteration")
ax.set_ylabel("Out-of-Bounds Count")
ax.set_title("Out-of-Bounds Evolution by Mesh")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("oob_by_mesh.png", dpi=150)
plt.close()
print("Saved oob_by_mesh.png")

# %% [markdown]
# ## Compare with and without annealing
#
# Run the solver without annealing to compare the results.

# %%
print("Running SQP2 optimization WITHOUT annealing...")
estimation_no_anneal = celeri.solve_sqp2(
    model,
    verbose=True,
    max_iter=100,
    solve_kwargs=solve_kwargs,
    objective="qr_sum_of_squares",
    operators=operators,
    annealing_enabled=False,
)

print(f"\nWithout annealing:")
print(f"  Total iterations: {len(estimation_no_anneal.trace.objective)}")
print(f"  Final out-of-bounds: {estimation_no_anneal.trace.out_of_bounds[-1]}")
print(f"  Final objective: {estimation_no_anneal.trace.objective_norm2[-1]:.6f}")

print(f"\nWith annealing:")
print(f"  Total iterations: {len(estimation.trace.objective)}")
print(f"  Final out-of-bounds: {estimation.trace.out_of_bounds[-1]}")
print(f"  Final objective: {estimation.trace.objective_norm2[-1]:.6f}")

# %%
# Compare convergence
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
ax.semilogy(estimation_no_anneal.trace.objective_norm2, "b.-", label="No annealing")
ax.semilogy(estimation.trace.objective_norm2, "r.-", label="With annealing")
ax.set_xlabel("Iteration")
ax.set_ylabel("Residual 2-norm")
ax.set_title("Objective Comparison")
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(estimation_no_anneal.trace.out_of_bounds, "b.-", label="No annealing")
ax.plot(estimation.trace.out_of_bounds, "r.-", label="With annealing")
ax.set_xlabel("Iteration")
ax.set_ylabel("Count")
ax.set_title("Out-of-Bounds Comparison")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("comparison.png", dpi=150)
plt.close()
print("Saved comparison.png")

# %% [markdown]
# ## Summary
#
# This notebook demonstrated:
#
# 1. **Model building**: Loading configuration and building operators
# 2. **SQP2 solver**: Running constrained optimization with coupling constraints
# 3. **Annealing**: Multi-pass optimization to escape local minima
# 4. **Visualization**: Convergence, slip rates, and coupling ratios
#
# The annealing approach helps find solutions that satisfy the non-convex coupling
# constraints while minimizing the data misfit. By iteratively tightening and
# loosening bounds, the solver can explore different regions of the solution space.

print("\n=== Demo complete! ===")
