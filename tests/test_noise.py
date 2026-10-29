"""
Unit tests for Photon Noise Model and Acquisition Simulator.
"""

import pytest
import numpy as np

from simulation.phantom import generate_shepp_logan
from simulation.projector import compute_radon_projection
from simulation.noise_model import apply_photon_noise
from simulation.acquisition import simulate_acquisition


def test_noise_finite_and_non_negative():
    """Verify that noisy sinogram contains only finite, non-negative values."""
    phantom = generate_shepp_logan(size=64)
    sinogram, _ = compute_radon_projection(phantom, num_projections=30)

    for i0 in [1e3, 1e4, 1e5]:
        noisy = apply_photon_noise(sinogram, incident_photons=i0, seed=42)
        assert noisy.shape == sinogram.shape
        assert np.all(np.isfinite(noisy)), f"Non-finite values found at I0={i0}"
        assert np.all(noisy >= 0.0), f"Negative line integrals found at I0={i0}"


def test_noise_reproducibility_with_seed():
    """Identical seeds must yield identical Poisson realizations."""
    sinogram = np.ones((50, 30)) * 0.5
    noisy1 = apply_photon_noise(sinogram, incident_photons=1e4, seed=123)
    noisy2 = apply_photon_noise(sinogram, incident_photons=1e4, seed=123)
    np.testing.assert_array_equal(noisy1, noisy2)


def test_noise_different_seeds_produce_variability():
    """Different random seeds must yield distinct Poisson realizations."""
    sinogram = np.ones((50, 30)) * 0.5
    noisy1 = apply_photon_noise(sinogram, incident_photons=1e4, seed=123)
    noisy2 = apply_photon_noise(sinogram, incident_photons=1e4, seed=456)
    assert not np.array_equal(noisy1, noisy2)


def test_noise_variance_decreases_with_higher_i0():
    """
    Statistical physics test:
    Lower photon budgets (e.g. 1e3) must exhibit greater sample variance
    around the true line integral than higher photon budgets (e.g. 1e6).
    """
    p_true = 0.5
    sinogram = np.full((100, 100), p_true, dtype=np.float64)

    noisy_low_exp = apply_photon_noise(sinogram, incident_photons=1e3, seed=42)
    noisy_high_exp = apply_photon_noise(sinogram, incident_photons=1e6, seed=42)

    var_low = float(np.var(noisy_low_exp))
    var_high = float(np.var(noisy_high_exp))

    assert var_low > var_high, (
        f"Expected lower photon budget to have higher variance: "
        f"var(1e3)={var_low:.6f} vs var(1e6)={var_high:.6f}"
    )


def test_simulate_acquisition_metadata():
    """Verify acquisition wrapper returns correct metadata and proxies."""
    phantom = generate_shepp_logan(size=64)
    noisy, ideal, theta, meta = simulate_acquisition(
        phantom,
        num_projections=90,
        photon_budget=5e4,
        seed=10,
    )

    assert noisy.shape == ideal.shape
    assert len(theta) == 90
    assert meta["num_projections"] == 90
    assert meta["photon_budget"] == 5e4
    assert meta["relative_photon_exposure_pct"] == 50.0  # 5e4 / 1e5 = 50%
    assert meta["projection_fraction_pct"] == 25.0       # 90 / 360 = 25%
    assert 0.0 < meta["normalized_acquisition_cost"] < 1.0


def test_negative_i0_raises():
    """Verify negative photon budget raises ValueError."""
    sinogram = np.ones((10, 10))
    with pytest.raises(ValueError):
        apply_photon_noise(sinogram, incident_photons=-100)
