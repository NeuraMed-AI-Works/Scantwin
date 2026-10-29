"""
Filtered Back Projection (FBP) Module
=====================================
Direct analytical inversion of the 2D Radon transform using the Fourier Slice Theorem:
    Sinogram -> 1D Frequency Ramp/Window Filtering -> Back Projection -> Spatial Reconstruction

DISCLAIMER:
Research and educational prototype. FBP implementation uses parallel-beam geometry.
"""

from typing import Optional
import numpy as np
import skimage.transform


def reconstruct_fbp(
    sinogram: np.ndarray,
    theta: np.ndarray,
    filter_name: str = "ramp",
    circle: bool = True,
    output_size: Optional[int] = None,
) -> np.ndarray:
    """
    Reconstruct a 2D CT slice from a parallel-beam sinogram using Filtered Back Projection.

    Parameters
    ----------
    sinogram : np.ndarray
        2D array of projection line integrals of shape (num_detectors, num_angles).
    theta : np.ndarray
        1D array of projection angles in degrees.
    filter_name : str, optional
        Filter applied to the Fourier projections:
        - "ramp": Standard Ram-Lak filter (preserves high spatial frequencies, sensitive to noise)
        - "shepp-logan": Multiplies ramp by sinc function (moderates high frequency noise)
        - "cosine": Smooth roll-off based on cosine function
        - "hamming": Hamming window for reduced ringing artifacts
        - "hann": Hann window with high noise suppression
        Default is "ramp".
    circle : bool, optional
        If True, zero out reconstruction outside the inscribed circular field of view.
        Default is True.
    output_size : int, optional
        Desired output image dimension. If None, defaults to sinogram.shape[0].

    Returns
    -------
    reconstruction : np.ndarray
        2D floating-point reconstructed image array.
    """
    valid_filters = ["ramp", "shepp-logan", "cosine", "hamming", "hann"]
    filter_key = filter_name.lower().strip()
    if filter_key not in valid_filters:
        raise ValueError(f"Filter '{filter_name}' not supported. Choose from {valid_filters}")

    # skimage.transform.iradon executes:
    # 1. 1D FFT along detector axis
    # 2. Multiplication by Fourier filter
    # 3. Inverse 1D FFT
    # 4. Discrete backprojection across angles theta
    reconstruction = skimage.transform.iradon(
        sinogram,
        theta=theta,
        output_size=output_size,
        filter_name=filter_key,
        circle=circle,
    )

    return reconstruction.astype(np.float64)
