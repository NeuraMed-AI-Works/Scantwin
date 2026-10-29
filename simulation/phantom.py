"""
Phantom Generation Module
=========================
Generates 2D digital phantoms for CT acquisition and reconstruction simulation.
Supports standard Shepp-Logan phantoms and geometric phantoms for testing.
"""

from typing import Tuple
import numpy as np
import skimage.data
import skimage.transform


def generate_shepp_logan(size: int = 256) -> np.ndarray:
    """
    Generate a 2D Shepp-Logan phantom at arbitrary spatial resolution.

    The Shepp-Logan phantom simulates the linear attenuation distribution of a
    human head slice with soft tissue, ventricles, and skull features.

    Parameters
    ----------
    size : int, optional
        Target height and width in pixels (default is 256).

    Returns
    -------
    phantom : np.ndarray
        2D float64 array of shape (size, size) with normalized attenuation values in [0.0, 1.0].
    """
    if size <= 0:
        raise ValueError(f"Phantom size must be positive, got {size}")

    # Standard scikit-image Shepp-Logan is 400x400 float64
    base_phantom = skimage.data.shepp_logan_phantom()

    if size == 400:
        return base_phantom.astype(np.float64)

    # Resample to the requested dimensions
    resized = skimage.transform.resize(
        base_phantom,
        (size, size),
        order=1,
        mode="reflect",
        preserve_range=True,
        anti_aliasing=True,
    )
    # Ensure strict floating point bounds [0.0, 1.0]
    return np.clip(resized, 0.0, 1.0).astype(np.float64)


def generate_disk_phantom(
    size: int = 256,
    outer_radius_fraction: float = 0.45,
    inner_radius_fraction: float = 0.20,
    attenuation_outer: float = 0.5,
    attenuation_inner: float = 0.9,
) -> np.ndarray:
    """
    Generate a simple geometric concentric disk phantom for unit testing and calibration.

    Parameters
    ----------
    size : int, optional
        Grid size in pixels (default 256).
    outer_radius_fraction : float, optional
        Outer disk radius as a fraction of image size.
    inner_radius_fraction : float, optional
        Inner disk radius as a fraction of image size.
    attenuation_outer : float, optional
        Attenuation value of the outer ring.
    attenuation_inner : float, optional
        Attenuation value of the inner target.

    Returns
    -------
    phantom : np.ndarray
        2D float64 array of shape (size, size).
    """
    if size <= 0:
        raise ValueError(f"Size must be positive, got {size}")

    coords = np.linspace(-0.5, 0.5, size)
    x, y = np.meshgrid(coords, coords)
    radius_map = np.sqrt(x**2 + y**2)

    phantom = np.zeros((size, size), dtype=np.float64)
    phantom[radius_map <= outer_radius_fraction] = attenuation_outer
    phantom[radius_map <= inner_radius_fraction] = attenuation_inner

    return phantom
