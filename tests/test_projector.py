"""
Unit tests for Radon Forward Projection module.
"""

import pytest
import numpy as np

from simulation.phantom import generate_shepp_logan
from simulation.projector import compute_radon_projection


def test_radon_projection_dimensions():
    """Verify projection matrix dimensions and angle sampling."""
    size = 128
    phantom = generate_shepp_logan(size=size)

    for n_proj in [30, 90, 180]:
        sinogram, theta = compute_radon_projection(phantom, num_projections=n_proj)
        assert sinogram.shape == (size, n_proj), f"Expected shape ({size}, {n_proj}), got {sinogram.shape}"
        assert len(theta) == n_proj
        assert np.isclose(theta[0], 0.0)
        assert theta[-1] < 180.0
        assert np.all(np.isfinite(sinogram)), "Sinogram contains NaN or Inf"
        assert np.all(sinogram >= 0.0), "Line integrals cannot be negative"


def test_radon_invalid_inputs():
    """Verify error handling on invalid inputs."""
    phantom = np.zeros((64, 64))
    with pytest.raises(ValueError):
        compute_radon_projection(phantom, num_projections=1)

    with pytest.raises(ValueError):
        compute_radon_projection(np.zeros((64, 64, 3)), num_projections=60)
