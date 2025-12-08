# %% [markdown]
"""
# Annealing demonstration

Lightweight remake of the earlier `benchmark_convex_solvers.ipynb`, but using
the modern Celeri API so we can explore the SQP2 annealing workflow with almost
no boilerplate. The notebook:

- loads a compact Japan test config (toggle to the full config if desired),
- runs `solve_sqp2` with an annealing schedule,
- plots iteration diagnostics (objective and out-of-bounds counts),
- shows a map of strike-slip rates, and
- provides a slider to inspect any iteration without rebuilding operators.
"""

# %%
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from ipywidgets import IntSlider, interact
from matplotlib.collections import PolyCollection

import celeri
from celeri import build_estimation, build_model
from celeri.config import Config
from celeri.operators import Operators, build_operators
from celeri.optimize import MinimizerTrace, solve_sqp2
from celeri.plot import plot_estimation_summary
from celeri.solve import Estimation

# %%
# Keep Clarabel deterministic across runs.
os.environ.setdefault("RAYON_NUM_THREADS", "4")

_CURRENT_FILE = Path(__file__) if "__file__" in globals() else Path.cwd()
PROJECT_ROOT = _CURRENT_FILE.resolve().parents[1]

SMALL_CONFIG = PROJECT_ROOT / "tests" / "test_japan_config.json"
FULL_CONFIG = PROJECT_ROOT / "data" / "config" / "japan_config.json"

# Set to False to run the full-resolution Japan example.
USE_SMALL_EXAMPLE = False
CONFIG_PATH = SMALL_CONFIG if USE_SMALL_EXAMPLE else FULL_CONFIG

# Annealing looseness values in mm/yr.
ANNEALING_SCHEDULE: list[float] = [0.25, 0.25, 0.125]
MAX_ITER = 12
SOLVE_KWARGS: dict[str, object] = {"solver": "CLARABEL", "ignore_dpp": True}

# %%
# Build model + operators once and reuse them for every visualization.
config: Config = celeri.get_config(CONFIG_PATH)
model = build_model(config)
operators: Operators = build_operators(model, eigen=True)

# %%
# Run SQP2 with annealing enabled.
annealed: Estimation = solve_sqp2(
    model,
    operators=operators,
    annealing_enabled=True,
    annealing_schedule=ANNEALING_SCHEDULE,
    max_iter=MAX_ITER,
    solve_kwargs=SOLVE_KWARGS,
    verbose=True,
)
if annealed.trace is None:
    raise RuntimeError("Expected a MinimizerTrace on the estimation object.")
trace: MinimizerTrace = annealed.trace

# %%
def trace_to_frame(trace: MinimizerTrace) -> pd.DataFrame:
    """Return a tidy view of iteration metrics."""
    return pd.DataFrame(
        {
            "iteration": np.arange(len(trace.objective_norm2), dtype=int),
            "objective_norm2": trace.objective_norm2,
            "out_of_bounds": trace.out_of_bounds,
            "nonconvex_loss": trace.nonconvex_constraint_loss,
            "iter_time_s": trace.iter_time,
            "total_time_s": np.cumsum(trace.iter_time),
        }
    )


progress = trace_to_frame(trace)
progress

# %%
def plot_progress(frame: pd.DataFrame) -> None:
    """Plot residual norm and out-of-bounds counts per iteration."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    axes[0].plot(frame["iteration"], frame["objective_norm2"], marker="o")
    axes[0].set_title("Objective 2-norm")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("||r||₂")

    axes[1].plot(frame["iteration"], frame["out_of_bounds"], marker="o")
    axes[1].set_title("Velocities out of bounds")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Count")

    for ax in axes:
        ax.grid(True, linestyle=":", linewidth=0.7)


plot_progress(progress)

# %%
def _mesh_poly_collection(
    mesh,
    values: npt.NDArray[np.floating],
    *,
    vmin: float,
    vmax: float,
    cmap: str = "coolwarm",
) -> PolyCollection:
    """Create a colored mesh for plotting.

    Args:
        values: (n_tde,) strike- or dip-slip rates in mm/yr for the mesh.
    """
    xy = mesh.points[:, :2]  # (n_nodes, 2) geographic coordinates
    verts = xy[np.asarray(mesh.verts)]
    pc = PolyCollection(verts, edgecolor="none", cmap=cmap)
    pc.set_array(values)
    pc.set_clim(vmin, vmax)
    return pc


def plot_tde_component(
    estimation: Estimation,
    *,
    mesh_indices: Iterable[int] | None = None,
    component: str = "strike",
    clim: tuple[float, float] | None = None,
    title: str = "TDE strike-slip (mm/yr)",
) -> None:
    """Plot strike- or dip-slip rates on all meshes."""
    if component not in {"strike", "dip"}:
        raise ValueError("component must be 'strike' or 'dip'")

    values_dict = (
        estimation.tde_strike_slip_rates
        if component == "strike"
        else estimation.tde_dip_slip_rates
    )
    if values_dict is None:
        raise ValueError("Estimation is missing TDE slip rates.")

    mesh_indices = list(mesh_indices or estimation.model.segment_mesh_indices)
    stacked = np.concatenate([values_dict[idx] for idx in mesh_indices])
    vmin, vmax = clim if clim is not None else (stacked.min(), stacked.max())

    fig, ax = plt.subplots(figsize=(10, 8))
    for idx in mesh_indices:
        mesh = estimation.model.meshes[idx]
        pc = _mesh_poly_collection(mesh, values_dict[idx], vmin=vmin, vmax=vmax)
        ax.add_collection(pc)

        # Mesh outline for context
        x_edge = mesh.points[mesh.ordered_edge_nodes[:, 0], 0]
        y_edge = mesh.points[mesh.ordered_edge_nodes[:, 0], 1]
        ax.plot(
            np.append(x_edge, x_edge[0]),
            np.append(y_edge, y_edge[0]),
            color="black",
            linewidth=0.7,
        )

    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")
    plt.colorbar(pc, ax=ax, label="Slip (mm/yr)")
    plt.show()


plot_tde_component(annealed, component="strike", title="Annealed strike-slip (final)")

# %%
# Standard velocity + slip summary for the annealed solution.
plot_estimation_summary(model, annealed, quiver_scale=model.config.quiver_scale)

# %%
@lru_cache(maxsize=None)
def estimation_at(iteration: int) -> Estimation:
    """Rebuild an Estimation for a specific iteration (cached).

    The cached array shapes match the state vector shape (n_params,), and the
    resulting `tde_*` dictionaries have one entry per mesh with (n_tde,) values.
    """
    if iteration >= len(trace.params):
        raise IndexError(f"iteration {iteration} exceeds available steps")
    params = trace.params[iteration]
    return build_estimation(model, operators, params)


def view_iteration(iteration: int) -> None:
    """Interactive viewer for slip during the SQP2 iterations."""
    est = estimation_at(iteration)
    oob = trace.out_of_bounds[iteration]
    title = f"Iteration {iteration} — {oob} out of bounds"
    plot_tde_component(est, component="strike", title=title)


interact(
    view_iteration,
    iteration=IntSlider(
        min=0,
        max=len(trace.params) - 1,
        step=1,
        value=len(trace.params) - 1,
        description="Iteration",
        continuous_update=False,
    ),
)
