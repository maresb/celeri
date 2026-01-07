#!/usr/bin/env python3
from __future__ import annotations

from loguru import logger


@logger.catch(reraise=True)
def main():
    # NOTE: Keep imports inside main() to reduce CLI startup latency.
    # Importing the top-level `celeri` package is intentionally avoided here because
    # it re-exports plotting/optimization helpers that pull in heavy dependencies.
    from celeri.celeri_util import get_logger
    from celeri.cli import parse_args, process_args
    from celeri.config import get_config
    from celeri.model import build_model
    from celeri.output import write_output

    # Process arguments
    args = parse_args()

    # Read in command file and start logging
    config = get_config(args.config_file_name)
    logger = get_logger(config)
    process_args(config, args)
    model = build_model(config)

    if config.repl:
        import IPython

    if config.solve_type == "dense":
        # Classic dense solve
        logger.info("Dense build and solve")
        from celeri.solve import build_and_solve_dense

        estimation = build_and_solve_dense(model)
    elif config.solve_type == "dense_no_meshes":
        # Classic dense solve with no meshes
        logger.info("Dense build and solve (no meshes)")
        from celeri.solve import build_and_solve_dense_no_meshes

        estimation = build_and_solve_dense_no_meshes(model)
    elif config.solve_type == "qp":
        from celeri.operators import build_operators
        from celeri.optimize_sqp import solve_sqp

        operators = build_operators(model, tde=True, eigen=True)
        estimation = solve_sqp(model, operators)
    elif config.solve_type == "qp2":
        # Bounded solve
        logger.info("Quadratic programming with KL modes")
        from celeri.optimize import solve_sqp2

        estimation = solve_sqp2(model)
    elif config.solve_type == "mcmc":
        # MCMC solve
        logger.info("MCMC solve")
        from celeri.solve_mcmc import solve_mcmc

        estimation = solve_mcmc(model)
    else:
        raise ValueError(f"Unknown solve type: {config.solve_type}")

    # Write output
    write_output(estimation)

    # Drop into ipython REPL
    if config.repl:
        import IPython

        IPython.embed(banner1="")


if __name__ == "__main__":
    main()
