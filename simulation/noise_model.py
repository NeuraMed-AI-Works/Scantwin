"""
CT Photon Counting and Transmission Noise Model
================================================
Simulates physical X-ray transmission Poisson statistics based on the Beer-Lambert law.

MATHEMATICAL FORMULATION:
1. Ideal projected line integral from 2D discrete Radon transform:
       p = integral mu(x, y) dl

2. Conversion to physical attenuation thickness:
   The discrete Radon transform sums pixel values without spatial dimension units.
   In diagnostic CT, the maximum effective attenuation path length across a soft tissue
   head/body slice is approximately 3.0 to 5.0 (e.g. 20 cm diameter * 0.2 cm^-1 linear
   attenuation coefficient). We map the peak line integral to a physical optical thickness
   p_max_phys = 4.0:
       s_scale = p_max_phys / max(p)
       p_phys = p * s_scale

3. Transmitted mean photon count through ray path:
       I_bar = I0 * exp(-p_phys)
   where I0 represents the relative incident photon budget proxy per ray.

4. Stochastic Poisson photon detection:
       I_meas ~ Poisson(I_bar)

5. Additive electronic readout noise (optional, default disabled):
       I_detected = I_meas + Normal(0, sigma_readout^2)

6. Logarithmic line integral recovery with stability guards:
       p_noisy_phys = -ln( max(I_detected, epsilon) / I0 )
       p_noisy = max(p_noisy_phys / s_scale, 0.0)

DISCLAIMER:
I0 serves strictly as a simulated photon budget proxy for exposure-tradeoff analysis.
It is NOT patient radiation dose in mSv and must not be used for clinical dosimetry.
"""

from typing import Optional
import numpy as np


def apply_photon_noise(
    sinogram: np.ndarray,
    incident_photons: float = 1e5,
    seed: Optional[int] = None,
    electronic_noise_sigma: float = 0.0,
    max_physical_line_integral: float = 4.0,
    eps: float = 1.0,
) -> np.ndarray:
    """
    Apply Beer-Lambert transmission Poisson noise to Radon line integrals.

    Parameters
    ----------
    sinogram : np.ndarray
        Ideal 2D line integral array p >= 0 of shape (detectors, angles).
    incident_photons : float, optional
        Relative incident photon budget I0 (default is 1e5).
        Lower values correspond to low-exposure acquisitions with higher Poisson quantum mottle.
    seed : int, optional
        Random seed for deterministic reproducibility.
    electronic_noise_sigma : float, optional
        Standard deviation of Gaussian electronic readout noise (default 0.0 = disabled).
    max_physical_line_integral : float, optional
        Target maximum optical thickness for Beer-Lambert attenuation (default 4.0).
    eps : float, optional
        Minimum transmitted photon count cutoff to guard against log(0) singularities (default 1.0).

    Returns
    -------
    noisy_sinogram : np.ndarray
        Noisy recovered line integral projection data of the same shape as input.
    """
    if incident_photons <= 0:
        raise ValueError(f"Incident photon budget I0 must be positive, got {incident_photons}")

    rng = np.random.default_rng(seed)

    # 1. Map dimensionless discrete Radon sum to realistic physical line integral [0, max_physical_line_integral]
    p_clamped = np.maximum(sinogram, 0.0)
    p_max = float(np.max(p_clamped))
    scale_factor = (max_physical_line_integral / p_max) if p_max > 1e-12 else 1.0

    p_physical = p_clamped * scale_factor

    # 2. Expected transmitted photon intensity via Beer-Lambert attenuation
    transmitted_mean = incident_photons * np.exp(-p_physical)

    # 3. Stochastic Poisson sampling of transmitted quanta
    measured_counts = rng.poisson(lam=transmitted_mean).astype(np.float64)

    # 4. Optional electronic readout noise
    if electronic_noise_sigma > 0.0:
        readout_noise = rng.normal(loc=0.0, scale=electronic_noise_sigma, size=measured_counts.shape)
        measured_counts += readout_noise

    # 5. Numerical stability guard: clamp photon counts
    measured_counts_safe = np.maximum(measured_counts, eps)

    # 6. Inverse log transmission recovery and rescale to Radon coordinates
    p_noisy_physical = -np.log(measured_counts_safe / incident_photons)
    noisy_sinogram = np.maximum(p_noisy_physical / scale_factor, 0.0)

    return noisy_sinogram.astype(np.float64)
