# AGENTS.md

## Cursor Cloud specific instructions

### Overview

`celeri` is a Python 3.13 scientific computing library for earthquake cycle kinematics. It is a single-package project with no databases, Docker, or external services. All data is file-based.

### Environment

- **Package manager**: [pixi](https://pixi.sh/) (conda-forge + PyPI hybrid). See `pixi.toml` for all dependency definitions.
- **Python**: 3.13 (managed by pixi, not the system Python).
- All commands should be run via `pixi run <command>` (e.g., `pixi run pytest`, `pixi run ruff check`).

### Common commands

| Task | Command |
|---|---|
| Lint | `pixi run ruff check celeri/ tests/` |
| Format check | `pixi run ruff format --check celeri/ tests/` |
| Run all tests | See CI partitioning below |
| Run celeri-solve | `cd data/config && pixi run celeri-solve <config>.json` |

### Test partitioning (matching CI)

Tests are split into three groups in CI (`.github/workflows/test.yml`):

- **other**: `pixi run pytest ./tests/ --ignore=./tests/test_solve_dense.py --ignore=./tests/test_optimize_sqp.py --ignore=./tests/test_optimize.py --ignore=./tests/test_cli.py`
- **solve**: `pixi run pytest ./tests/test_solve_dense.py --arraydiff`
- **optimize**: `pixi run pytest ./tests/test_optimize_sqp.py ./tests/test_optimize.py`

### Gotchas

- The "other" test group includes `test_closure.py` which prints very large numpy arrays via `loguru` warnings (due to `addopts = "-s"` in `pyproject.toml`). If you pipe test output (e.g., with `tee`), the pipe buffer can fill and block the process indefinitely. Run tests directly without piping, or redirect to a file with `>` instead of `|`.
- `celeri-solve` with `"repl": true` in the config file will launch an IPython REPL after the solve completes. Set `"repl": false` or kill the process after confirming the solve succeeded.
- The pixi lockfile format warning (`uses an older format (v6)`) is harmless and can be ignored.
- Some tests involve heavy elastic operator computation and can take several minutes on a single CPU.
