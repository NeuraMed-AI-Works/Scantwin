"""
Unit tests for Reconstruction methods (FBP, SART, Engine).
"""

import pytest
import numpy as np

from simulation.phantom import generate_shepp_logan
from simulation.projector import compute_radon_projection
from reconstruction.fbp import reconstruct_fbp
from reconstruction.sart import reconstruct_sart
from reconstruction.reconstruction_engine import reconstruct


def test_fbp_filters():
    """Verify that all standard FBP filters execute and produce valid outputs."""
    size = 64
    phantom = generate_shepp_logan(size=size)
    sinogram, theta = compute_radon_projection(phantom, num_projections=60)

    for filt in ["ramp", "shepp-logan", "cosine", "hamming", "hann"]:
        recon = reconstruct_fbp(sinogram, theta, filter_name=filt)
        assert recon.shape == (size, size)
        assert np.all(np.isfinite(recon))


def test_sart_reconstruction():
    """Verify SART iterative reconstruction."""
    size = 64
    phantom = generate_shepp_logan(size=size)
    sinogram, theta = compute_radon_projection(phantom, num_projections=30)

    recon = reconstruct_sart(sinogram, theta, iterations=1)
    assert recon.shape == (size, size)
    assert np.all(np.isfinite(recon))


def test_reconstruction_engine_unified():
    """Verify unified reconstruction entry point."""
    size = 64
    phantom = generate_shepp_logan(size=size)
    sinogram, theta = compute_radon_projection(phantom, num_projections=45)

    recon, meta = reconstruct(sinogram, theta, method="FBP", filter_name="ramp")
    assert recon.shape == (size, size)
    assert meta["method"] == "FBP"
    assert "elapsed_time_ms" in meta
    assert meta["elapsed_time_ms"] >= 0.0

    with pytest.raises(ValueError):
        reconstruct(sinogram, theta, method="INVALID_METHOD")
