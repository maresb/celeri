# PR Review Issues and Observations

## Summary

This document records observations noted during the review and refactoring of the test reorganization PR.

## Observations

### 1. Missing trailing newline in test_output_files.py

**Severity**: Minor (style)

The file `tests/test_output_files.py` is missing a trailing newline at the end of the file. While POSIX conventions suggest text files should end with a newline, this was preserved to maintain consistency with the original PR.

### 2. Executable permission on cascadia.msh

**Severity**: Minor (file permissions)

The file `tests/data/mesh/cascadia.msh` has executable permissions (0755) set, which is unusual for a mesh data file. Mesh files are typically plain data and don't need to be executable. Consider changing to 0644 in a future cleanup.

### 3. Test parametrization reduction in test_cli.py

**Observation**: The original test_cli.py used a cross-product of 3 config files × 5 solve types = 15 test combinations, but then skipped some. The new version uses explicit 10 test combinations, removing:
- All `dense` solve type tests (presumably moved elsewhere or redundant)
- `qp` solve type for non-test configs (marked as "very slow")

This is an intentional optimization to reduce CI time, but should be documented somewhere that these combinations are still valid but excluded from regular CI.

### 4. Large test data files

**Observation**: Several large mesh files were added:
- `cascadia.msh` (~10,000 lines)
- `graham_nshm23_..._segmesh0.msh` (~2,500 lines)
- `graham_nshm23_..._segmesh1.msh` (~3,200 lines)
- `wna_segment0.csv` and `wna_segment1.csv` (~4,000 lines each)

Consider whether these should be stored in Git LFS for better repository performance.

### 5. Reference file naming convention

**Observation**: The reference files use a naming convention that encodes test parameters:
- `test_dense_sol_test_japan_config-False-False.txt`
- `test_operator_tde_to_velocities_test_japan_config.txt`

The parameter encoding uses `True`/`False` strings which works but could be more explicit (e.g., `eigen_True_tde_False`).

### 6. Internal API usage in tests

**Observation**: The `test_smart_segment_recompute` test uses internal APIs (`_OperatorBuilder`, `_store_elastic_operators`, `_hash_elastic_operator_input`) prefixed with underscore, indicating they're considered internal/private. This tight coupling to internal implementation could make the test fragile to refactoring.

## Recommendations

1. Consider adding the trailing newline to `test_output_files.py` in a future commit
2. Fix file permissions on `cascadia.msh` to 0644
3. Evaluate Git LFS for large test data files
4. Consider adding a comment in test_cli.py explaining which test combinations were removed and why
5. Consider whether the internal API usage in `test_smart_segment_recompute` should be replaced with public API calls if available
