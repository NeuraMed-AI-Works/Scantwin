"""
Normalization and Scaling Utilities
===================================
Maintains separation between raw physical CT attenuation values,
metric calculation scales, and display normalization.
"""

from typing import Tuple, Optional
import numpy as np


def normalize_for_display(
    image: np.ndarray,
    window_range: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """
    Produce a [0.0, 1.0] normalized copy of an image for screen display.

    Parameters
    ----------
    image : np.ndarray
        Raw CT attenuation or reconstruction array.
    window_range : tuple of (float, float), optional
        Optional (vmin, vmax) window levels. If None, derived from min/max.

    Returns
    -------
    display_image : np.ndarray
        Float array scaled to [0.0, 1.0] for visualization only.
    """
    if window_range is not None:
        vmin, vmax = window_range
    else:
        vmin = float(np.min(image))
        vmax = float(np.max(image))

    if vmax - vmin < 1e-12:
        return np.zeros_like(image, dtype=np.float64)

    clipped = np.clip(image, vmin, vmax)
    return ((clipped - vmin) / (vmax - vmin)).astype(np.float64)


def compute_absolute_error_map(
    reference: np.ndarray,
    reconstruction: np.ndarray,
) -> np.ndarray:
    """
    Compute absolute pixel-wise difference map: |reference - reconstruction|.
    """
    return np.abs(reference - reconstruction).astype(np.float64)


def compute_error_statistics(
    reference: np.ndarray,
    reconstruction: np.ndarray,
) -> dict:
    """
    Calculate comprehensive pixel-wise error metrics.

    Returns
    -------
    stats : dict
        error_map, mse, rmse, max_abs_error, mean_abs_error.
    """
    err_map = compute_absolute_error_map(reference, reconstruction)
    mse = float(np.mean(err_map ** 2))
    rmse = float(np.sqrt(mse))
    max_err = float(np.max(err_map))
    mean_err = float(np.mean(err_map))

    return {
        "error_map": err_map,
        "mse": mse,
        "rmse": rmse,
        "max_abs_error": max_err,
        "mean_abs_error": mean_err,
    }
