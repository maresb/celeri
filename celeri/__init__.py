"""Celeri - A crustal block modeling package.

This module uses lazy imports for heavy dependencies to reduce startup time.
The following modules are loaded lazily when their exports are first accessed:
- celeri.optimize (cvxpy)
- celeri.optimize_sqp (cvxpy)  
- celeri.solve_mcmc (pymc, arviz)
- celeri.plot (matplotlib, cartopy)

Essential modules for CLI startup are loaded eagerly:
- celeri.cli, celeri.config, celeri.model, celeri.solve, celeri.output
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

# Essential imports for CLI startup - these are loaded eagerly
from celeri.celeri_util import (
    align_velocities,
    diagnose_matrix,
    get_2component_index,
    get_3component_index,
    get_keep_index_12,
    get_logger,
    get_newest_run_folder,
    get_segment_oblique_projection,
    get_transverse_projection,
    interleave2,
    interleave3,
    read_run,
    wrap2360,
)
from celeri.cli import parse_args, process_args
from celeri.config import Config, get_config
from celeri.mesh import Mesh, MeshConfig
from celeri.model import (
    Model,
    assign_block_labels,
    build_model,
    create_output_folder,
    process_sar,
    process_segment,
    process_station,
    read_data,
)
from celeri.operators import (
    Operators,
    build_operators,
    get_block_strain_rate_to_velocities_partials,
    get_eigenvalues_and_eigenvectors,
    get_full_dense_operator,
    get_full_dense_operator_eigen,
    get_global_float_block_rotation_partials,
    get_mogi_to_velocities_partials,
    get_qp_all_inequality_operator_and_data_vector,
    get_qp_slip_rate_inequality_operator_and_data_vector,
    get_qp_tde_inequality_operator_and_data_vector,
    get_rotation_to_slip_rate_partials,
    get_rotation_to_tri_slip_rate_partials,
    get_rotation_to_velocities_partials,
    get_segment_station_operator_okada,
    get_slip_rate_bounds,
    get_slip_rate_constraints,
    get_tde_to_velocities_single_mesh,
    get_weighting_vector_single_mesh_for_col_norms,
)
from celeri.output import write_output
from celeri.solve import (
    Estimation,
    assemble_and_solve_dense,
    build_and_solve_dense,
    build_and_solve_dense_no_meshes,
    build_estimation,
    lsqlin_qp,
)
from celeri.spatial import (
    get_okada_displacements,
    get_shared_sides,
    get_tde_to_velocities,
    get_tri_displacements,
)

# Type checking imports (no runtime cost)
if TYPE_CHECKING:
    from celeri.optimize import solve_sqp2
    from celeri.optimize_sqp import plot_iterative_convergence, solve_sqp
    from celeri.plot import (
        get_default_plotting_options,
        plot_coastlines,
        plot_common_elements,
        plot_coupling,
        plot_coupling_evolution,
        plot_estimation_summary,
        plot_fault_geometry,
        plot_input_summary,
        plot_land,
        plot_matrix_abs_log,
        plot_mesh,
        plot_mesh_mode,
        plot_residuals,
        plot_rotation_components,
        plot_segment_displacements,
        plot_segment_rates,
        plot_strain_rate_components_for_block,
        plot_tde_boundary_condition_labels,
        plot_vel_arrows_elements,
        plot_vels,
    )
    from celeri.solve_mcmc import solve_mcmc

try:
    from importlib.metadata import version

    __version__ = version("celeri")
except Exception:
    __version__ = "unknown"

# Lazy import mappings: attribute_name -> (module_path, attribute_name_in_module)
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # From celeri.optimize
    "solve_sqp2": ("celeri.optimize", "solve_sqp2"),
    # From celeri.optimize_sqp
    "solve_sqp": ("celeri.optimize_sqp", "solve_sqp"),
    "plot_iterative_convergence": ("celeri.optimize_sqp", "plot_iterative_convergence"),
    # From celeri.solve_mcmc
    "solve_mcmc": ("celeri.solve_mcmc", "solve_mcmc"),
    # From celeri.plot
    "get_default_plotting_options": ("celeri.plot", "get_default_plotting_options"),
    "plot_coastlines": ("celeri.plot", "plot_coastlines"),
    "plot_common_elements": ("celeri.plot", "plot_common_elements"),
    "plot_coupling": ("celeri.plot", "plot_coupling"),
    "plot_coupling_evolution": ("celeri.plot", "plot_coupling_evolution"),
    "plot_estimation_summary": ("celeri.plot", "plot_estimation_summary"),
    "plot_fault_geometry": ("celeri.plot", "plot_fault_geometry"),
    "plot_input_summary": ("celeri.plot", "plot_input_summary"),
    "plot_land": ("celeri.plot", "plot_land"),
    "plot_matrix_abs_log": ("celeri.plot", "plot_matrix_abs_log"),
    "plot_mesh": ("celeri.plot", "plot_mesh"),
    "plot_mesh_mode": ("celeri.plot", "plot_mesh_mode"),
    "plot_residuals": ("celeri.plot", "plot_residuals"),
    "plot_rotation_components": ("celeri.plot", "plot_rotation_components"),
    "plot_segment_displacements": ("celeri.plot", "plot_segment_displacements"),
    "plot_segment_rates": ("celeri.plot", "plot_segment_rates"),
    "plot_strain_rate_components_for_block": (
        "celeri.plot",
        "plot_strain_rate_components_for_block",
    ),
    "plot_tde_boundary_condition_labels": (
        "celeri.plot",
        "plot_tde_boundary_condition_labels",
    ),
    "plot_vel_arrows_elements": ("celeri.plot", "plot_vel_arrows_elements"),
    "plot_vels": ("celeri.plot", "plot_vels"),
}


def __getattr__(name: str):
    """Lazily import heavy modules when their exports are first accessed."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """List all available attributes including lazy imports."""
    return list(__all__)


__all__ = [
    "Config",
    "Estimation",
    "Mesh",
    "MeshConfig",
    "Model",
    "Operators",
    "__version__",
    "align_velocities",
    "assemble_and_solve_dense",
    "assign_block_labels",
    "build_and_solve_dense",
    "build_and_solve_dense_no_meshes",
    "build_estimation",
    "build_model",
    "build_operators",
    "create_output_folder",
    "diagnose_matrix",
    "get_2component_index",
    "get_3component_index",
    "get_block_strain_rate_to_velocities_partials",
    "get_config",
    "get_default_plotting_options",
    "get_eigenvalues_and_eigenvectors",
    "get_full_dense_operator",
    "get_full_dense_operator_eigen",
    "get_global_float_block_rotation_partials",
    "get_keep_index_12",
    "get_logger",
    "get_mogi_to_velocities_partials",
    "get_newest_run_folder",
    "get_okada_displacements",
    "get_qp_all_inequality_operator_and_data_vector",
    "get_qp_slip_rate_inequality_operator_and_data_vector",
    "get_qp_tde_inequality_operator_and_data_vector",
    "get_rotation_to_slip_rate_partials",
    "get_rotation_to_tri_slip_rate_partials",
    "get_rotation_to_velocities_partials",
    "get_segment_oblique_projection",
    "get_segment_station_operator_okada",
    "get_shared_sides",
    "get_slip_rate_bounds",
    "get_slip_rate_constraints",
    "get_tde_to_velocities",
    "get_tde_to_velocities_single_mesh",
    "get_transverse_projection",
    "get_tri_displacements",
    "get_weighting_vector_single_mesh_for_col_norms",
    "interleave2",
    "interleave3",
    "lsqlin_qp",
    "parse_args",
    "plot_coastlines",
    "plot_common_elements",
    "plot_coupling",
    "plot_coupling_evolution",
    "plot_estimation_summary",
    "plot_fault_geometry",
    "plot_input_summary",
    "plot_iterative_convergence",
    "plot_land",
    "plot_matrix_abs_log",
    "plot_mesh",
    "plot_mesh_mode",
    "plot_residuals",
    "plot_rotation_components",
    "plot_segment_displacements",
    "plot_segment_rates",
    "plot_strain_rate_components_for_block",
    "plot_tde_boundary_condition_labels",
    "plot_vel_arrows_elements",
    "plot_vels",
    "process_args",
    "process_sar",
    "process_segment",
    "process_station",
    "read_data",
    "read_run",
    "solve_mcmc",
    "solve_sqp",
    "solve_sqp2",
    "wrap2360",
    "write_output",
]
