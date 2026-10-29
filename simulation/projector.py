"""
CT Forward Projection Module
============================
Simulates parallel-beam CT projections via the 2D Radon transform.

A line integral through the attenuation coefficient distribution mu(x, y)
at angle theta and radial detector offset s is defined as:

    p(theta, s) = integral_{ray(theta, s)} mu(x, y) dl

The collection of all line integrals across angles 0 <= theta < 180 degrees
forms the sinogram.
"""

from typing import Tuple
import numpy as np
import skimage.transform


def compute_radon_projection(
    image: np.ndarray,
    num_projections: int = 180,
    angular_range: Tuple[float, float] = (0.0, 180.0),
    circle: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute parallel-beam forward projections (sinogram) of a 2D image.

    Parameters
    ----------
    image : np.ndarray
        2D floating-point image of linear attenuation values, shape (H, W).
    num_projections : int, optional
        Number of angular view samples (default is 180).
    angular_range : tuple of (float, float), optional
        Angular range in degrees [start, stop). Default is (0.0, 180.0).
    circle : bool, optional
        If True, rays outside the inscribed circle of the image are zeroed out (standard in CT).
        Default is True.

    Returns
    -------
    sinogram : np.ndarray
        2D floating-point projection data of shape (num_detector_bins, num_projections).
        - Axis 0 (rows): detector radial position coordinate `s`
        - Axis 1 (columns): projection angle index corresponding to `theta`
    theta : np.ndarray
        1D array of sampled projection angles in degrees, shape (num_projections,).
    """
    if image.ndim != 2:
        raise ValueError(f"Expected 2D image array, got {image.ndim}D array with shape {image.shape}.")

    if num_projections < 2:
        raise ValueError(f"Number of projections must be >= 2, got {num_projections}.")

    theta_start, theta_stop = angular_range
    # Evenly spaced projection angles across [theta_start, theta_stop)
    theta = np.linspace(theta_start, theta_stop, num_projections, endpoint=False)

    # Compute continuous parallel-beam Radon transform using scikit-image
    sinogram = skimage.transform.radon(image, theta=theta, circle=circle)

    return sinogram.astype(np.float64), theta
