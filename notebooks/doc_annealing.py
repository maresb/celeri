# %% [markdown]
# # Annealing Optimization Demo
#
# This notebook demonstrates the annealing feature in Celeri's optimization.
# Annealing allows the solver to continue optimizing after all constraints are satisfied,
# gradually loosening bounds to explore better solutions.

# %%
import os

os.environ["RAYON_NUM_THREADS"] = "4"

import matplotlib.pyplot as plt
import numpy as np

import celeri
from celeri.optimize import MinimizerTrace

# %%
# Configure matplotlib for high-quality figures
plt.rcParams["figure.dpi"] = 150

# %%
# Load the Japan model
config_file = "data/config/japan_config.json"
model = celeri.build_model(config_file)
operators = celeri.build_operators(model, eigen=True, tde=True)

# %%
# Run optimization with annealing enabled
solve_kwargs = dict(
    solver="CLARABEL",
    equilibrate_enable=False,
    direct_solve_method="faer",
    ignore_dpp=True,
)

estimation = celeri.solve_sqp2(
    model,
    operators=operators,
    verbose=True,
    solve_kwargs=solve_kwargs,
    objective="qr_sum_of_squares",
    max_iter=100,
    annealing_enabled=True,
    annealing_schedule=[0.125, 0.125, 0.125],
)

trace: MinimizerTrace = estimation.trace

# %%
# Display convergence information
print(f"Total iterations: {len(trace.params)}")
print(f"Final out-of-bounds: {trace.out_of_bounds[-1]}")
print(f"Final objective (L2 norm): {trace.objective_norm2[-1]:.6e}")

# Find iterations where we're in bounds (annealing phase)
in_bounds_iterations = [
    i for i, oob in enumerate(trace.out_of_bounds) if oob == 0
]
if in_bounds_iterations:
    # Note: in_bounds_iterations uses 0-based indexing, but we display as iteration number
    print(f"Annealing started at iteration: {in_bounds_iterations[0] + 1}")
    print(f"Annealing iterations: {len(in_bounds_iterations)}")

# %%
# Plot convergence metrics
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

iterations = np.arange(len(trace.params))

# Objective function (L2 norm)
axes[0, 0].semilogy(iterations, trace.objective_norm2, "o-", markersize=4)
axes[0, 0].set_xlabel("Iteration")
axes[0, 0].set_ylabel("Objective (L2 norm)")
axes[0, 0].set_title("Objective Function Convergence")
axes[0, 0].grid(True, alpha=0.3)
if in_bounds_iterations:
    axes[0, 0].axvline(
        in_bounds_iterations[0], color="r", linestyle="--", alpha=0.5, label="Annealing starts"
    )
    axes[0, 0].legend()

# Out-of-bounds count
axes[0, 1].plot(iterations, trace.out_of_bounds, "o-", markersize=4, color="orange")
axes[0, 1].set_xlabel("Iteration")
axes[0, 1].set_ylabel("Out-of-bounds count")
axes[0, 1].set_title("Constraint Satisfaction")
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].axhline(0, color="g", linestyle="--", alpha=0.5)

# Non-convex constraint loss
axes[1, 0].semilogy(iterations, trace.nonconvex_constraint_loss, "o-", markersize=4, color="purple")
axes[1, 0].set_xlabel("Iteration")
axes[1, 0].set_ylabel("Non-convex constraint loss")
axes[1, 0].set_title("Non-convex Constraint Loss")
axes[1, 0].grid(True, alpha=0.3)

# Iteration time
axes[1, 1].plot(iterations, trace.iter_time, "o-", markersize=4, color="teal")
axes[1, 1].set_xlabel("Iteration")
axes[1, 1].set_ylabel("Time (seconds)")
axes[1, 1].set_title("Iteration Time")
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Helper function to create an Estimation from a state vector
def estimation_from_state_vector(
    state_vector: np.ndarray, model, operators
) -> celeri.Estimation:
    """Create an Estimation object from a state vector."""
    return celeri.build_estimation(model, operators, state_vector)

# %%
# Plot TDE slip rate evolution for selected iterations
def plot_tde_slip_evolution(
    trace: MinimizerTrace,
    iterations_to_plot: list[int] | None = None,
    mesh_idx: int = 0,
):
    """Plot TDE slip rates at different iterations."""
    if iterations_to_plot is None:
        # Select iterations: start, middle, end, and a few during annealing
        n_iter = len(trace.params)
        iterations_to_plot = [0, n_iter // 4, n_iter // 2]
        in_bounds = [i for i, oob in enumerate(trace.out_of_bounds) if oob == 0]
        if in_bounds:
            iterations_to_plot.extend([
                in_bounds[0],  # First in-bounds iteration
                in_bounds[len(in_bounds) // 2],  # Middle of annealing
                in_bounds[-1],  # Final iteration
            ])
        iterations_to_plot = sorted(set(iterations_to_plot))[:6]  # Limit to 6 plots

    n_plots = len(iterations_to_plot)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    meshes = trace.model.meshes
    segment = trace.model.segment
    lon_range = trace.model.config.lon_range
    lat_range = trace.model.config.lat_range

    for idx, iter_num in enumerate(iterations_to_plot):
        ax = axes[idx]

        # Get TDE slip rates for this iteration
        state_vector = trace.params[iter_num]
        iter_estimation = estimation_from_state_vector(
            state_vector, trace.model, trace.minimizer.operators
        )
        tde_slip_rates = iter_estimation.tde_strike_slip_rates
        if tde_slip_rates is None:
            raise ValueError(f"TDE slip rates are None for iteration {iter_num}")

        # Plot fault segments
        for i in range(len(segment)):
            color = "k" if segment.dip.iloc[i] == 90.0 else "r"
            ax.plot(
                [segment.lon1.iloc[i], segment.lon2.iloc[i]],
                [segment.lat1.iloc[i], segment.lat2.iloc[i]],
                f"-{color}",
                linewidth=0.5,
            )

        # Plot TDE slip rates for the selected mesh
        mesh = meshes[mesh_idx]
        x_coords = mesh.points[:, 0]
        y_coords = mesh.points[:, 1]
        vertex_array = np.asarray(mesh.verts)

        xy = np.c_[x_coords, y_coords]
        verts = xy[vertex_array]

        import matplotlib.collections
        pc = matplotlib.collections.PolyCollection(
            verts, edgecolor="none", cmap="rainbow"
        )

        slip_values = tde_slip_rates[mesh_idx]
        pc.set_array(slip_values)
        pc.set_clim([-10, 10])  # Fixed range for comparison
        ax.add_collection(pc)

        # Add colorbar
        plt.colorbar(pc, ax=ax, label="Slip (mm/yr)")

        # Set limits and aspect
        ax.set_xlim([lon_range[0], lon_range[1]])
        ax.set_ylim([lat_range[0], lat_range[1]])
        ax.set_aspect("equal", adjustable="box")

        # Title with iteration info
        oob = trace.out_of_bounds[iter_num]
        obj = trace.objective_norm2[iter_num]
        phase = "Annealing" if oob == 0 else "Constrained"
        ax.set_title(
            f"Iteration {iter_num} ({phase})\n"
            f"OOB: {oob}, Obj: {obj:.2e}"
        )

    # Hide unused subplots
    for idx in range(len(iterations_to_plot), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.show()

# %%
# Plot evolution for the first mesh
plot_tde_slip_evolution(trace, mesh_idx=0)

# %%
# Create an interactive visualization function
def create_interactive_slider(trace: MinimizerTrace, mesh_idx: int = 0):
    """Create an interactive slider to view different iterations."""
    try:
        from IPython.display import display
        from ipywidgets import IntSlider, interact
    except ImportError:
        print("IPython widgets not available. Skipping interactive visualization.")
        return

    def plot_iteration(n: int):
        fig, ax = plt.subplots(figsize=(12, 10))

        meshes = trace.model.meshes
        segment = trace.model.segment
        lon_range = trace.model.config.lon_range
        lat_range = trace.model.config.lat_range

        # Plot fault segments
        for i in range(len(segment)):
            color = "k" if segment.dip.iloc[i] == 90.0 else "r"
            ax.plot(
                [segment.lon1.iloc[i], segment.lon2.iloc[i]],
                [segment.lat1.iloc[i], segment.lat2.iloc[i]],
                f"-{color}",
                linewidth=0.5,
            )

        # Get TDE slip rates
        state_vector = trace.params[n]
        iter_estimation = estimation_from_state_vector(
            state_vector, trace.model, trace.minimizer.operators
        )
        tde_slip_rates = iter_estimation.tde_strike_slip_rates
        if tde_slip_rates is None:
            raise ValueError(f"TDE slip rates are None for iteration {n}")

        # Plot mesh
        mesh = meshes[mesh_idx]
        x_coords = mesh.points[:, 0]
        y_coords = mesh.points[:, 1]
        vertex_array = np.asarray(mesh.verts)

        xy = np.c_[x_coords, y_coords]
        verts = xy[vertex_array]

        import matplotlib.collections
        pc = matplotlib.collections.PolyCollection(
            verts, edgecolor="none", cmap="rainbow"
        )

        slip_values = tde_slip_rates[mesh_idx]
        pc.set_array(slip_values)
        pc.set_clim([-10, 10])
        ax.add_collection(pc)
        plt.colorbar(pc, ax=ax, label="Slip (mm/yr)")

        ax.set_xlim([lon_range[0], lon_range[1]])
        ax.set_ylim([lat_range[0], lat_range[1]])
        ax.set_aspect("equal", adjustable="box")

        oob = trace.out_of_bounds[n]
        obj = trace.objective_norm2[n]
        phase = "Annealing" if oob == 0 else "Constrained"
        ax.set_title(
            f"Iteration {n} ({phase})\n"
            f"Out-of-bounds: {oob}, Objective: {obj:.6e}"
        )

        plt.tight_layout()
        plt.show()

    interact(
        plot_iteration,
        n=IntSlider(
            min=0,
            max=len(trace.params) - 1,
            step=1,
            description="Iteration:",
            continuous_update=False,
        ),
    )

# %%
# Uncomment to use interactive slider (requires IPython)
# create_interactive_slider(trace, mesh_idx=0)

# %%
# Summary statistics
print("=" * 60)
print("Annealing Summary")
print("=" * 60)
print(f"Total iterations: {len(trace.params)}")
print(f"Constrained phase iterations: {len([oob for oob in trace.out_of_bounds if oob > 0])}")
print(f"Annealing phase iterations: {len([oob for oob in trace.out_of_bounds if oob == 0])}")
print(f"\nInitial objective: {trace.objective_norm2[0]:.6e}")
print(f"Final objective: {trace.objective_norm2[-1]:.6e}")
print(f"Improvement: {(1 - trace.objective_norm2[-1] / trace.objective_norm2[0]) * 100:.2f}%")
print(f"\nTotal time: {trace.total_time:.2f} seconds")
print(f"Average iteration time: {np.mean(trace.iter_time):.3f} seconds")
