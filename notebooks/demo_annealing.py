"""Demonstrate annealing optimization with simplified plotting.

This script shows how the annealing feature works by running optimization
with an annealing schedule and visualizing how the TDE slip rates evolve
across iterations.

Compared to the old benchmark_convex_solvers.ipynb notebook, this script:
- Uses the simplified solve_sqp2() API instead of manual problem building
- Leverages the MinimizerTrace class for automatic iteration tracking
- Uses build_estimation() helper to extract TDE slip rates from any iteration
- Creates cleaner, more focused visualizations without copy-paste boilerplate

The annealing feature allows the solver to continue optimizing after all
constraints are satisfied by temporarily loosening bounds, helping escape
local minima and find better solutions.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from celeri.config import get_config
from celeri.model import Model
from celeri.optimize import solve_sqp2
from celeri.plot import plot_mesh
from celeri.solve import build_estimation

logger.disable("celeri")


def plot_tde_slip_rates(
    model: Model,
    estimation,
    *,
    title: str = "TDE Slip (strike-slip)",
    vmin: float | None = None,
    vmax: float | None = None,
):
    """Plot TDE slip rates on meshes with segment overlay."""
    segment = model.segment
    meshes = model.meshes
    lon_range = model.config.lon_range
    lat_range = model.config.lat_range

    # Plot segments
    plt.figure(figsize=(12, 10))
    plt.title(title)
    for i in range(len(segment)):
        if segment.dip[i] == 90.0:
            plt.plot(
                [segment.lon1[i], segment.lon2[i]],
                [segment.lat1[i], segment.lat2[i]],
                "-k",
                linewidth=0.5,
            )
        else:
            plt.plot(
                [segment.lon1[i], segment.lon2[i]],
                [segment.lat1[i], segment.lat2[i]],
                "-r",
                linewidth=0.5,
            )

    plt.xlim([lon_range[0], lon_range[1]])
    plt.ylim([lat_range[0], lat_range[1]])
    plt.gca().set_aspect("equal", adjustable="box")

    # Plot TDE slip rates
    if estimation.tde_strike_slip_rates is not None:
        all_values = np.concatenate(
            [np.ravel(vals) for vals in estimation.tde_strike_slip_rates.values()]
        )
        if vmin is None:
            vmin = np.min(all_values)
        if vmax is None:
            vmax = np.max(all_values)

        ax = plt.gca()
        pc = None
        for i in range(len(meshes)):
            if i in estimation.tde_strike_slip_rates:
                pc = plot_mesh(
                    meshes[i],
                    fill_value=estimation.tde_strike_slip_rates[i],
                    ax=ax,
                    vmin=vmin,
                    vmax=vmax,
                )
        if pc is not None:
            plt.colorbar(pc, ax=ax, label="slip (mm/yr)")


def plot_annealing_convergence(trace):
    """Plot convergence metrics across annealing iterations."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Objective function
    axes[0, 0].plot(trace.objective_norm2, "o-")
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Objective (L2 norm)")
    axes[0, 0].set_title("Objective Function")
    axes[0, 0].grid(True)

    # Out of bounds
    axes[0, 1].plot(trace.out_of_bounds, "o-", color="red")
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("Out of bounds count")
    axes[0, 1].set_title("Constraint Violations")
    axes[0, 1].grid(True)

    # Constraint loss
    axes[1, 0].plot(trace.nonconvex_constraint_loss, "o-", color="orange")
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylabel("Constraint loss")
    axes[1, 0].set_title("Non-convex Constraint Loss")
    axes[1, 0].grid(True)

    # Iteration time
    axes[1, 1].plot(trace.iter_time, "o-", color="green")
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].set_ylabel("Time (s)")
    axes[1, 1].set_title("Iteration Time")
    axes[1, 1].grid(True)

    plt.tight_layout()


def main():
    """Run annealing optimization and create visualizations."""
    # Load configuration and build model
    # Path is relative to workspace root
    from pathlib import Path
    
    # Get workspace root (parent of notebooks directory)
    workspace_root = Path(__file__).parent.parent
    config_path = workspace_root / "data" / "config" / "japan_config.json"
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            "Please ensure you're running from the workspace root or that the data files exist."
        )
    
    config = get_config(str(config_path))
    model = Model.from_config(config)

    # Configure solver settings
    solve_kwargs = dict(
        solver="CLARABEL",
        equilibrate_enable=False,
        direct_solve_method="faer",
        ignore_dpp=True,
    )

    # Run optimization with annealing
    print("Running optimization with annealing...")
    estimation = solve_sqp2(
        model,
        max_iter=100,
        verbose=True,
        solve_kwargs=solve_kwargs,
        objective="qr_sum_of_squares",
        annealing_enabled=True,
        annealing_schedule=[0.125, 0.125, 0.125],
    )

    # Get the trace for iteration-by-iteration analysis
    trace = estimation.trace
    if trace is None:
        print("No trace available - cannot plot convergence")
        return

    print(f"\nOptimization completed in {len(trace.params)} iterations")
    print(f"Final objective: {trace.objective_norm2[-1]:.6e}")
    print(f"Final out-of-bounds: {trace.out_of_bounds[-1]}")

    # Create output directory in workspace root
    output_dir = workspace_root / "notebooks" / "annealing_output"
    output_dir.mkdir(exist_ok=True)
    
    # Plot convergence metrics
    print("\nPlotting convergence metrics...")
    plot_annealing_convergence(trace)
    output_file = output_dir / "annealing_convergence.png"
    plt.savefig(str(output_file), dpi=150, bbox_inches="tight")
    print(f"Saved: {output_file}")

    # Plot final TDE slip rates
    print("\nPlotting final TDE slip rates...")
    plot_tde_slip_rates(trace.model, estimation, title="Final TDE Slip (strike-slip)")
    output_file = output_dir / "annealing_final_slip.png"
    plt.savefig(str(output_file), dpi=150, bbox_inches="tight")
    print(f"Saved: {output_file}")

    # Plot a few key iterations to show evolution
    print("\nPlotting evolution at key iterations...")
    key_iterations = []
    # Find iterations where out-of-bounds goes to zero (annealing starts)
    for i, oob in enumerate(trace.out_of_bounds):
        if i > 0 and trace.out_of_bounds[i - 1] > 0 and oob == 0:
            key_iterations.append(i)
    # Also include first, middle, and last
    key_iterations = [0] + key_iterations[:3] + [len(trace.params) - 1]
    key_iterations = sorted(set(key_iterations))[:5]  # Limit to 5 plots

    fig, axes = plt.subplots(1, len(key_iterations), figsize=(5 * len(key_iterations), 5))
    if len(key_iterations) == 1:
        axes = [axes]

    # Get value range from final iteration
    final_estimation = build_estimation(
        trace.model, trace.minimizer.operators, trace.params[-1]
    )
    if final_estimation.tde_strike_slip_rates is not None:
        all_final_values = np.concatenate(
            [np.ravel(vals) for vals in final_estimation.tde_strike_slip_rates.values()]
        )
        vmin, vmax = np.min(all_final_values), np.max(all_final_values)
    else:
        vmin, vmax = -10, 10

    for idx, iter_num in enumerate(key_iterations):
        ax = axes[idx]
        # Create estimation for this iteration
        iter_estimation = build_estimation(
            trace.model, trace.minimizer.operators, trace.params[iter_num]
        )

        # Plot segments
        segment = trace.model.segment
        for i in range(len(segment)):
            if segment.dip[i] == 90.0:
                ax.plot(
                    [segment.lon1[i], segment.lon2[i]],
                    [segment.lat1[i], segment.lat2[i]],
                    "-k",
                    linewidth=0.5,
                )
            else:
                ax.plot(
                    [segment.lon1[i], segment.lon2[i]],
                    [segment.lat1[i], segment.lat2[i]],
                    "-r",
                    linewidth=0.5,
                )

        ax.set_xlim(
            [trace.model.config.lon_range[0], trace.model.config.lon_range[1]]
        )
        ax.set_ylim(
            [trace.model.config.lat_range[0], trace.model.config.lat_range[1]]
        )
        ax.set_aspect("equal", adjustable="box")

        # Plot TDE slip rates
        if iter_estimation.tde_strike_slip_rates is not None:
            for i in range(len(trace.model.meshes)):
                if i in iter_estimation.tde_strike_slip_rates:
                    plot_mesh(
                        trace.model.meshes[i],
                        fill_value=iter_estimation.tde_strike_slip_rates[i],
                        ax=ax,
                        vmin=vmin,
                        vmax=vmax,
                    )

        oob = trace.out_of_bounds[iter_num]
        obj = trace.objective_norm2[iter_num]
        ax.set_title(
            f"Iter {iter_num}\nOOB: {oob}, Obj: {obj:.2e}",
            fontsize=10,
        )

    plt.tight_layout()
    output_file = output_dir / "annealing_evolution.png"
    plt.savefig(str(output_file), dpi=150, bbox_inches="tight")
    print(f"Saved: {output_file}")

    print("\nDone!")


if __name__ == "__main__":
    main()
