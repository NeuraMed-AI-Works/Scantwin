"""
Simple post-filters shared by the X-ray and ultrasound models
=============================================================
``apply_post_filter(image, kind, strength)``

* ``none``        - identity
* ``gaussian``    - Gaussian smoothing, sigma = 0.7 * strength pixels
* ``median``      - median filter, window = 2 * strength + 1 pixels
* ``anisotropic`` - Perona-Malik edge-preserving diffusion, 3 * strength iterations
                    (a deliberately simple "anisotropic-style" smoother, not a tuned denoiser)

All filters keep the input intensity scale, so metrics against the ground truth stay comparable.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

POST_FILTERS = ("none", "gaussian", "median", "anisotropic")


def anisotropic_diffusion(image: np.ndarray, iterations: int, kappa: float = 0.08, step: float = 0.2) -> np.ndarray:
    """Perona-Malik diffusion (exponential conductance) with reflecting borders."""
    u = np.asarray(image, dtype=np.float64).copy()
    for _ in range(max(0, int(iterations))):
        p = np.pad(u, 1, mode="edge")
        d_n = p[:-2, 1:-1] - u
        d_s = p[2:, 1:-1] - u
        d_w = p[1:-1, :-2] - u
        d_e = p[1:-1, 2:] - u
        u = u + step * sum(np.exp(-((d / kappa) ** 2)) * d for d in (d_n, d_s, d_w, d_e))
    return u


def apply_post_filter(image: np.ndarray, kind: str, strength: int = 1) -> np.ndarray:
    """Apply one of ``POST_FILTERS`` to a 2D image."""
    kind = str(kind).lower()
    strength = max(1, int(strength))
    if kind == "none":
        return np.asarray(image, dtype=np.float64)
    if kind == "gaussian":
        return ndimage.gaussian_filter(np.asarray(image, dtype=np.float64), sigma=0.7 * strength)
    if kind == "median":
        return ndimage.median_filter(np.asarray(image, dtype=np.float64), size=2 * strength + 1)
    if kind == "anisotropic":
        return anisotropic_diffusion(image, iterations=3 * strength)
    raise ValueError(f"Unknown post-filter '{kind}'. Choose from {POST_FILTERS}.")
