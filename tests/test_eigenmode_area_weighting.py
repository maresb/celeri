"""Test area weighting in eigenmode computation using proper mathematical formulation.

This tests the corrected implementation using generalized eigenvalue problems
for proper eigenmodes on meshes with area weighting.
"""

import numpy as np
import pytest

from celeri.operators import get_eigenvalues_and_eigenvectors


def test_constant_weighting_backward_compatibility():
    """Test that constant weighting matches the original behavior."""
    # Create simple test data
    x = np.array([0.0, 1.0, 0.5])
    y = np.array([0.0, 0.0, 1.0])
    z = np.array([0.0, 0.0, 0.0])
    areas = np.array([1.0, 2.0, 0.5])
    
    # Get eigenvalues with constant weighting (original behavior)
    evals_const, evecs_const = get_eigenvalues_and_eigenvectors(
        n_eigenvalues=2, x=x, y=y, z=z, distance_exponent=1.0,
        areas=areas, area_weighting="constant"
    )
    
    # Get eigenvalues without area weighting (should be same as constant)
    evals_none, evecs_none = get_eigenvalues_and_eigenvectors(
        n_eigenvalues=2, x=x, y=y, z=z, distance_exponent=1.0
    )
    
    # Results should be identical
    np.testing.assert_allclose(evals_const, evals_none)
    np.testing.assert_allclose(np.abs(evecs_const), np.abs(evecs_none))


def test_area_weighting_changes_result():
    """Test that area weighting produces different results from constant weighting."""
    # Create test data with varying areas
    x = np.array([0.0, 1.0, 0.5])
    y = np.array([0.0, 0.0, 1.0])
    z = np.array([0.0, 0.0, 0.0])
    areas = np.array([1.0, 10.0, 0.1])  # Very different areas
    
    # Get eigenvalues with area weighting (generalized eigenvalue problem)
    evals_area, evecs_area = get_eigenvalues_and_eigenvectors(
        n_eigenvalues=2, x=x, y=y, z=z, distance_exponent=1.0,
        areas=areas, area_weighting="area"
    )
    
    # Get eigenvalues with constant weighting
    evals_const, evecs_const = get_eigenvalues_and_eigenvectors(
        n_eigenvalues=2, x=x, y=y, z=z, distance_exponent=1.0,
        areas=areas, area_weighting="constant"
    )
    
    # Results should be different (with significantly different areas)
    assert not np.allclose(evals_area, evals_const)


def test_inverse_area_weighting():
    """Test that inverse area weighting works correctly."""
    # Create test data
    x = np.array([0.0, 1.0, 0.5])
    y = np.array([0.0, 0.0, 1.0])
    z = np.array([0.0, 0.0, 0.0])
    areas = np.array([1.0, 2.0, 0.5])
    
    # Get eigenvalues with inverse area weighting
    evals_inv, evecs_inv = get_eigenvalues_and_eigenvectors(
        n_eigenvalues=2, x=x, y=y, z=z, distance_exponent=1.0,
        areas=areas, area_weighting="inverse"
    )
    
    # Results should be finite (no NaN from division by zero)
    assert np.all(np.isfinite(evals_inv))
    assert np.all(np.isfinite(evecs_inv))


def test_zero_area_handling():
    """Test that zero areas are handled correctly without producing NaN."""
    # Create test data with one zero area
    x = np.array([0.0, 1.0, 0.5])
    y = np.array([0.0, 0.0, 1.0])
    z = np.array([0.0, 0.0, 0.0])
    areas = np.array([1.0, 0.0, 0.5])  # One zero area
    
    # Test with area weighting
    evals_area, evecs_area = get_eigenvalues_and_eigenvectors(
        n_eigenvalues=2, x=x, y=y, z=z, distance_exponent=1.0,
        areas=areas, area_weighting="area"
    )
    
    # Test with inverse area weighting
    evals_inv, evecs_inv = get_eigenvalues_and_eigenvectors(
        n_eigenvalues=2, x=x, y=y, z=z, distance_exponent=1.0,
        areas=areas, area_weighting="inverse"
    )
    
    # Results should be finite (no NaN from zero area or division by zero)
    assert np.all(np.isfinite(evals_area))
    assert np.all(np.isfinite(evecs_area))
    assert np.all(np.isfinite(evals_inv))
    assert np.all(np.isfinite(evecs_inv))


def test_eigenvector_orthogonality():
    """Test that eigenvectors are orthogonal with respect to the mass matrix."""
    # Create test data with varying areas
    x = np.array([0.0, 1.0, 0.5, 2.0])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    z = np.array([0.0, 0.0, 0.0, 0.0])
    areas = np.array([1.0, 4.0, 0.25, 2.0])

    # Get eigenvectors with area weighting
    evals_area, evecs_area = get_eigenvalues_and_eigenvectors(
        n_eigenvalues=3, x=x, y=y, z=z, distance_exponent=1.0,
        areas=areas, area_weighting="area"
    )

    # Check orthogonality with respect to mass matrix: v_i^T M v_j = 0 for i != j
    mass_matrix = np.diag(areas)
    
    # Test orthogonality between first two eigenvectors
    orthogonality_01 = evecs_area[:, 0].T @ mass_matrix @ evecs_area[:, 1]
    orthogonality_02 = evecs_area[:, 0].T @ mass_matrix @ evecs_area[:, 2]
    orthogonality_12 = evecs_area[:, 1].T @ mass_matrix @ evecs_area[:, 2]
    
    # Should be approximately zero (within numerical precision)
    assert abs(orthogonality_01) < 1e-12
    assert abs(orthogonality_02) < 1e-12
    assert abs(orthogonality_12) < 1e-12


def test_mass_matrix_normalization():
    """Test that eigenvectors are properly normalized with respect to mass matrix."""
    # Create test data
    x = np.array([0.0, 1.0, 0.5])
    y = np.array([0.0, 0.0, 1.0])
    z = np.array([0.0, 0.0, 0.0])
    areas = np.array([1.0, 4.0, 0.25])

    # Get eigenvectors with area weighting
    evals_area, evecs_area = get_eigenvalues_and_eigenvectors(
        n_eigenvalues=2, x=x, y=y, z=z, distance_exponent=1.0,
        areas=areas, area_weighting="area"
    )

    # Check that eigenvectors are normalized: v_i^T M v_i = 1
    mass_matrix = np.diag(areas)
    
    norm_0 = evecs_area[:, 0].T @ mass_matrix @ evecs_area[:, 0]
    norm_1 = evecs_area[:, 1].T @ mass_matrix @ evecs_area[:, 1]
    
    # Should be approximately 1
    np.testing.assert_allclose(norm_0, 1.0, rtol=1e-12)
    np.testing.assert_allclose(norm_1, 1.0, rtol=1e-12)