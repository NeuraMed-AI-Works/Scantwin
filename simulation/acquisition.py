"""
CT Acquisition Simulator Module
===============================
Simulates the physical acquisition stage, integrating forward Radon projection,
sparse-view sampling, and photon-limited noise models.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np

from config import (
    REFERENCE_I0,
    REFERENCE_PROJECTIONS,
    DEFAULT_ANGULAR_RANGE_DEG,
)
from simulation.projector import compute_radon_projection
from simulation.noise_model import apply_photon_noise


def simulate_acquisition(
    phantom: np.ndarray,
    num_projections: int = 180,
    photon_budget: float = 1e5,
    seed: Optional[int] = 42,
    angular_range: Tuple[float, float] = DEFAULT_ANGULAR_RANGE_DEG,
    circle: bool = True,
    electronic_noise_sigma: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Simulate a complete CT acquisition process.

    Parameters
    ----------
    phantom : np.ndarray
        2D linear attenuation phantom slice.
    num_projections : int, optional
        Number of projection angles sampled (default 180).
    photon_budget : float, optional
        Relative incident photon budget I0 (default 1e5).
    seed : int, optional
        Random seed for Poisson noise reproducibility.
    angular_range : tuple of (float, float), optional
        Projection angular span in degrees.
    circle : bool, optional
        Masking outside inscribed circle.
    electronic_noise_sigma : float, optional
        Detector electronic noise std dev.

    Returns
    -------
    noisy_sinogram : np.ndarray
        Sinogram including simulated Poisson transmission noise.
    ideal_sinogram : np.ndarray
        Noise-free continuous forward Radon projection.
    theta : np.ndarray
        Array of projection angles in degrees.
    metadata : dict
        Acquisition metrics and relative exposure proxies.
    """
    # 1. Forward continuous Radon projection
    ideal_sinogram, theta = compute_radon_projection(
        phantom,
        num_projections=num_projections,
        angular_range=angular_range,
        circle=circle,
    )

    # 2. Add Poisson transmission noise
    noisy_sinogram = apply_photon_noise(
        ideal_sinogram,
        incident_photons=photon_budget,
        seed=seed,
        electronic_noise_sigma=electronic_noise_sigma,
    )

    # 3. Compute relative exposure and acquisition proxies
    rel_exposure_pct = (photon_budget / REFERENCE_I0) * 100.0
    proj_fraction_pct = (num_projections / REFERENCE_PROJECTIONS) * 100.0
    acq_cost = 0.7 * (photon_budget / REFERENCE_I0) + 0.3 * (num_projections / REFERENCE_PROJECTIONS)

    metadata: Dict[str, Any] = {
        "num_projections": num_projections,
        "photon_budget": photon_budget,
        "seed": seed,
        "relative_photon_exposure_pct": rel_exposure_pct,
        "projection_fraction_pct": proj_fraction_pct,
        "normalized_acquisition_cost": acq_cost,
    }

    return noisy_sinogram, ideal_sinogram, theta, metadata
