from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

# NOTE ON STARTUP LATENCY
# -----------------------
# Historically, `celeri.__init__` eagerly imported a large portion of the package
# to provide a convenience "flat" API. That made `import celeri` (and therefore
# CLIs like `celeri-solve`) slow, because it pulled in heavy optional deps like
# matplotlib/cartopy/cvxpy even when they weren't used.
#
# We keep the same public API, but resolve symbols lazily on first access.

try:
    from importlib.metadata import version as _version

    __version__ = _version("celeri")
except Exception:  # pragma: no cover
    __version__ = "unknown"

# Public symbols -> (module, attribute)
_EXPORTS: dict[str, tuple[str, str]] = {
    # celeri_util
    "align_velocities": ("celeri.celeri_util", "align_velocities"),
    "diagnose_matrix": ("celeri.celeri_util", "diagnose_matrix"),
    "get_2component_index": ("celeri.celeri_util", "get_2component_index"),
    "get_3component_index": ("celeri.celeri_util", "get_3component_index"),
    "get_keep_index_12": ("celeri.celeri_util", "get_keep_index_12"),
    "get_logger": ("celeri.celeri_util", "get_logger"),
    "get_newest_run_folder": ("celeri.celeri_util", "get_newest_run_folder"),
    "get_segment_oblique_projection": ("celeri.celeri_util", "get_segment_oblique_projection"),
    "get_transverse_projection": ("celeri.celeri_util", "get_transverse_projection"),
    "interleave2": ("celeri.celeri_util", "interleave2"),
    "interleave3": ("celeri.celeri_util", "interleave3"),
    "read_run": ("celeri.celeri_util", "read_run"),
    "wrap2360": ("celeri.celeri_util", "wrap2360"),

    # cli
    "parse_args": ("celeri.cli", "parse_args"),
    "process_args": ("celeri.cli", "process_args"),

    # config
    "Config": ("celeri.config", "Config"),
    "get_config": ("celeri.config", "get_config"),

    # mesh
    "Mesh": ("celeri.mesh", "Mesh"),
    "MeshConfig": ("celeri.mesh", "MeshConfig"),

    # model
    "Model": ("celeri.model", "Model"),
    "assign_block_labels": ("celeri.model", "assign_block_labels"),
    "build_model": ("celeri.model", "build_model"),
    "create_output_folder": ("celeri.model", "create_output_folder"),
    "process_sar": ("celeri.model", "process_sar"),
    "process_segment": ("celeri.model", "process_segment"),
    "process_station": ("celeri.model", "process_station"),
    "read_data": ("celeri.model", "read_data"),

    # operators
    "Operators": ("celeri.operators", "Operators"),
    "build_operators": ("celeri.operators", "build_operators"),
    "get_block_strain_rate_to_velocities_partials": ("celeri.operators", "get_block_strain_rate_to_velocities_partials"),
    "get_eigenvalues_and_eigenvectors": ("celeri.operators", "get_eigenvalues_and_eigenvectors"),
    "get_full_dense_operator": ("celeri.operators", "get_full_dense_operator"),
    "get_full_dense_operator_eigen": ("celeri.operators", "get_full_dense_operator_eigen"),
    "get_global_float_block_rotation_partials": ("celeri.operators", "get_global_float_block_rotation_partials"),
    "get_mogi_to_velocities_partials": ("celeri.operators", "get_mogi_to_velocities_partials"),
    "get_qp_all_inequality_operator_and_data_vector": ("celeri.operators", "get_qp_all_inequality_operator_and_data_vector"),
    "get_qp_slip_rate_inequality_operator_and_data_vector": ("celeri.operators", "get_qp_slip_rate_inequality_operator_and_data_vector"),
    "get_qp_tde_inequality_operator_and_data_vector": ("celeri.operators", "get_qp_tde_inequality_operator_and_data_vector"),
    "get_rotation_to_slip_rate_partials": ("celeri.operators", "get_rotation_to_slip_rate_partials"),
    "get_rotation_to_tri_slip_rate_partials": ("celeri.operators", "get_rotation_to_tri_slip_rate_partials"),
    "get_rotation_to_velocities_partials": ("celeri.operators", "get_rotation_to_velocities_partials"),
    "get_segment_station_operator_okada": ("celeri.operators", "get_segment_station_operator_okada"),
    "get_slip_rate_bounds": ("celeri.operators", "get_slip_rate_bounds"),
    "get_slip_rate_constraints": ("celeri.operators", "get_slip_rate_constraints"),
    "get_tde_to_velocities_single_mesh": ("celeri.operators", "get_tde_to_velocities_single_mesh"),
    "get_weighting_vector_single_mesh_for_col_norms": ("celeri.operators", "get_weighting_vector_single_mesh_for_col_norms"),

    # optimization
    "solve_sqp2": ("celeri.optimize", "solve_sqp2"),
    "plot_iterative_convergence": ("celeri.optimize_sqp", "plot_iterative_convergence"),
    "solve_sqp": ("celeri.optimize_sqp", "solve_sqp"),

    # output
    "write_output": ("celeri.output", "write_output"),

    # plotting (heavy deps: matplotlib/cartopy)
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
    "plot_strain_rate_components_for_block": ("celeri.plot", "plot_strain_rate_components_for_block"),
    "plot_tde_boundary_condition_labels": ("celeri.plot", "plot_tde_boundary_condition_labels"),
    "plot_vel_arrows_elements": ("celeri.plot", "plot_vel_arrows_elements"),
    "plot_vels": ("celeri.plot", "plot_vels"),

    # solve
    "Estimation": ("celeri.solve", "Estimation"),
    "assemble_and_solve_dense": ("celeri.solve", "assemble_and_solve_dense"),
    "build_and_solve_dense": ("celeri.solve", "build_and_solve_dense"),
    "build_and_solve_dense_no_meshes": ("celeri.solve", "build_and_solve_dense_no_meshes"),
    "build_estimation": ("celeri.solve", "build_estimation"),
    "lsqlin_qp": ("celeri.solve", "lsqlin_qp"),

    # mcmc
    "solve_mcmc": ("celeri.solve_mcmc", "solve_mcmc"),

    # spatial
    "get_okada_displacements": ("celeri.spatial", "get_okada_displacements"),
    "get_shared_sides": ("celeri.spatial", "get_shared_sides"),
    "get_tde_to_velocities": ("celeri.spatial", "get_tde_to_velocities"),
    "get_tri_displacements": ("celeri.spatial", "get_tri_displacements"),
}


def __getattr__(name: str) -> Any:  # pragma: no cover
    """Lazily resolve public attributes on first access."""
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as e:  # pragma: no cover
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from e

    module = import_module(module_name)
    value = getattr(module, attr_name)

    # Cache on module for subsequent attribute access.
    globals()[name] = value
    return value


def __dir__() -> list[str]:  # pragma: no cover
    return sorted(set(list(globals().keys()) + list(_EXPORTS.keys())))


if TYPE_CHECKING:  # pragma: no cover
    # For type checkers/IDE completion only.
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
    from celeri.optimize import solve_sqp2
    from celeri.optimize_sqp import plot_iterative_convergence, solve_sqp
    from celeri.output import write_output
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
    from celeri.solve import (
        Estimation,
        assemble_and_solve_dense,
        build_and_solve_dense,
        build_and_solve_dense_no_meshes,
        build_estimation,
        lsqlin_qp,
    )
    from celeri.solve_mcmc import solve_mcmc
    from celeri.spatial import (
        get_okada_displacements,
        get_shared_sides,
        get_tde_to_velocities,
        get_tri_displacements,
    )


__all__ = ["__version__", *_EXPORTS.keys()]
