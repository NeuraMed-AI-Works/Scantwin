"""
Simultaneous Algebraic Reconstruction Technique (SART) Module
=============================================================
Iterative algebraic solver for parallel-beam CT reconstruction.
Solves the linear system p = A * x through successive projection and back-projection
residual updates.

DISCLAIMER:
SART is used here as an academic iterative reconstruction baseline and does not represent
proprietary commercial iterative algorithms (e.g. ASiR, AIDR 3D, IMR).
"""

from typing import Optional
import numpy as np
import skimage.transform


def reconstruct_sart(
    sinogram: np.ndarray,
    theta: np.ndarray,
    iterations: int = 1,
    relaxation: float = 0.15,
    image_init: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Reconstruct a 2D CT slice using the Simultaneous Algebraic Reconstruction Technique (SART).

    Parameters
    ----------
    sinogram : np.ndarray
        2D projection data of shape (num_detectors, num_angles).
    theta : np.ndarray
        1D array of projection angles in degrees.
    iterations : int, optional
        Number of full SART iterations across all projection angles (default is 1).
    relaxation : float, optional
        Relaxation parameter weighting ray updates (default 0.15).
    image_init : np.ndarray, optional
        Initial estimate of the image (defaults to zeros if None).

    Returns
    -------
    reconstruction : np.ndarray
        2D floating-point reconstructed image array.
    """
    if iterations < 1:
        raise ValueError(f"Iterations must be >= 1, got {iterations}")

    reconstruction = image_init
    for _ in range(iterations):
        reconstruction = skimage.transform.iradon_sart(
            sinogram,
            theta=theta,
            image=reconstruction,
            relaxation=relaxation,
        )

    return reconstruction.astype(np.float64)
