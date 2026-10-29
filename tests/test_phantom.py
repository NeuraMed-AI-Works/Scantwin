"""
Unit tests for Phantom and Lesion generation modules.
"""

import pytest
import numpy as np

from simulation.phantom import generate_shepp_logan, generate_disk_phantom
from simulation.lesion import insert_lesion


def test_shepp_logan_dimensions_and_values():
    """Verify Shepp-Logan generation across varying grid resolutions."""
    for size in [64, 128, 256]:
        phantom = generate_shepp_logan(size=size)
        assert phantom.shape == (size, size), f"Expected shape ({size}, {size}), got {phantom.shape}"
        assert phantom.dtype == np.float64
        assert np.all(np.isfinite(phantom)), "Phantom contains non-finite values (NaN/Inf)"
        assert np.min(phantom) >= 0.0, f"Negative attenuation found: {np.min(phantom)}"
        assert np.max(phantom) <= 1.0, f"Attenuation exceeds 1.0: {np.max(phantom)}"


def test_disk_phantom():
    """Verify geometric disk phantom properties."""
    disk = generate_disk_phantom(size=100, outer_radius_fraction=0.4, inner_radius_fraction=0.1)
    assert disk.shape == (100, 100)
    assert np.all(np.isfinite(disk))
    assert np.isclose(disk[50, 50], 0.9)  # Center should have inner attenuation
    assert np.isclose(disk[0, 0], 0.0)    # Corner should be air (zero)


def test_insert_lesion_geometry_and_contrast():
    """Verify lesion insertion accuracy, boundary checks, and masks."""
    size = 128
    phantom = generate_shepp_logan(size=size)
    center = (70, 60)
    radius = 6.0
    contrast = 0.10

    mod_phantom, lesion_mask, bg_mask, meta = insert_lesion(
        phantom=phantom,
        center=center,
        radius=radius,
        contrast=contrast,
    )

    assert mod_phantom.shape == (size, size)
    assert lesion_mask.shape == (size, size)
    assert bg_mask.shape == (size, size)

    # Disjoint ROIs
    assert not np.any(lesion_mask & bg_mask), "Lesion mask and background mask must be disjoint!"

    # Verify centroid pixel is within lesion mask
    assert lesion_mask[center[0], center[1]], "Centroid pixel must belong to lesion mask"

    # Verify contrast was added
    assert mod_phantom[center[0], center[1]] > phantom[center[0], center[1]]

    # Verify metadata
    assert meta["num_lesion_pixels"] > 0
    assert meta["num_background_pixels"] > 0


def test_lesion_out_of_bounds_raises():
    """Verify that attempting to place a lesion outside image bounds raises ValueError."""
    phantom = np.zeros((100, 100))
    with pytest.raises(ValueError):
        insert_lesion(phantom, center=(150, 50))
