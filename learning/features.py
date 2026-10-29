"""
Image descriptors used by the protocol model
============================================
Seven cheap, modality-agnostic descriptors computed inside the valid field of view (circle for
CT / Other, full frame for X-ray / MRI / ultrasound). They describe *how hard an image is to acquire
and reconstruct* (edges, texture, size of the object), not what disease it might show. Image values
are the [0, 1] intensity proxy produced by ``simulation.datasets``.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from scipy import ndimage

FEATURE_NAMES: List[str] = [
    "mean_intensity",
    "std_intensity",
    "foreground_fraction",
    "edge_density",
    "entropy",
    "high_freq_ratio",
    "foreground_mean",
]


def compute_image_features(image: np.ndarray, fov: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Return the descriptor dict for one 2D image in [0, 1]; ``fov`` is the valid-pixel mask (default: full frame)."""
    if image.ndim != 2:
        raise ValueError("image must be 2D")
    if fov is None:
        fov = np.ones(image.shape, dtype=bool)
    vals = image[fov]
    fg = vals > 0.05
    grad = np.hypot(ndimage.sobel(image, axis=0), ndimage.sobel(image, axis=1))
    hist, _ = np.histogram(vals, bins=32, range=(0.0, 1.0))
    p = hist / max(hist.sum(), 1)
    entropy = float(-np.sum(p[p > 0] * np.log2(p[p > 0])))

    power = np.abs(np.fft.fftshift(np.fft.fft2(image * fov))) ** 2
    h, w = power.shape
    yy, xx = np.meshgrid(np.arange(h) - h / 2, np.arange(w) - w / 2, indexing="ij")
    radius = np.hypot(yy, xx) / (min(h, w) / 2)
    total = power.sum()
    hf = float(power[radius > 0.5].sum() / total) if total > 0 else 0.0

    feats = {
        "mean_intensity": float(vals.mean()),
        "std_intensity": float(vals.std()),
        "foreground_fraction": float(fg.mean()),
        "edge_density": float(grad[fov].mean()),
        "entropy": entropy,
        "high_freq_ratio": hf,
        "foreground_mean": float(vals[fg].mean()) if fg.any() else 0.0,
    }
    if not all(np.isfinite(v) for v in feats.values()):
        raise ValueError("Non-finite image feature; check the image.")
    return feats
