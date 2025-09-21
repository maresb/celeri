"""Test area weighting in eigenmode computation."""

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
    
    # Get eigenvalues with area weighting
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


def test_uniform_areas_gives_constant_behavior():
    """Test that uniform areas with area weighting scales eigenvalues proportionally."""
    # Create test data with uniform areas
    x = np.array([0.0, 1.0, 0.5])
    y = np.array([0.0, 0.0, 1.0])
    z = np.array([0.0, 0.0, 0.0])
    uniform_areas = np.array([2.0, 2.0, 2.0])  # All same area

    # Get eigenvalues with area weighting
    evals_area, evecs_area = get_eigenvalues_and_eigenvectors(
        n_eigenvalues=2, x=x, y=y, z=z, distance_exponent=1.0,
        areas=uniform_areas, area_weighting="area"
    )

    # Get eigenvalues with constant weighting
    evals_const, evecs_const = get_eigenvalues_and_eigenvectors(
        n_eigenvalues=2, x=x, y=y, z=z, distance_exponent=1.0,
        areas=uniform_areas, area_weighting="constant"
    )

    # With uniform areas, eigenvalues should be scaled by the area value
    area_value = uniform_areas[0]
    expected_ratio = area_value  # sqrt(area * area) = area
    np.testing.assert_allclose(evals_area / evals_const, expected_ratio, rtol=1e-12)
    
    # Eigenvectors should have the same magnitudes
    np.testing.assert_allclose(np.abs(evecs_area), np.abs(evecs_const), rtol=1e-12)


def test_inverse_area_uniform_gives_inverse_scaling():
    """Test that inverse area weighting with uniform areas scales by 1/area."""
    # Create test data with uniform areas
    x = np.array([0.0, 1.0, 0.5])
    y = np.array([0.0, 0.0, 1.0])
    z = np.array([0.0, 0.0, 0.0])
    uniform_areas = np.array([4.0, 4.0, 4.0])  # All same area

    # Get eigenvalues with inverse area weighting
    evals_inv, _ = get_eigenvalues_and_eigenvectors(
        n_eigenvalues=2, x=x, y=y, z=z, distance_exponent=1.0,
        areas=uniform_areas, area_weighting="inverse"
    )

    # Get eigenvalues with constant weighting
    evals_const, _ = get_eigenvalues_and_eigenvectors(
        n_eigenvalues=2, x=x, y=y, z=z, distance_exponent=1.0,
        areas=uniform_areas, area_weighting="constant"
    )

    # With uniform areas and inverse weighting, eigenvalues should be scaled by 1/area
    area_value = uniform_areas[0]
    expected_ratio = 1.0 / area_value
    np.testing.assert_allclose(evals_inv / evals_const, expected_ratio, rtol=1e-12)