"""
Synthetic Lesion Insertion Module
=================================
Injects known synthetic low-contrast targets into digital phantoms
for quantitative detectability and contrast-to-noise ratio (CNR) assessment.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np


def insert_lesion(
    phantom: np.ndarray,
    center: Optional[Tuple[int, int]] = None,
    radius: float = 8.0,
    contrast: float = 0.08,
    shape: str = "circle",
    eccentricity: float = 1.0,
    background_gap: float = 3.0,
    background_thickness: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Insert a synthetic target (lesion) of known geometry and contrast into a 2D phantom.

    Parameters
    ----------
    phantom : np.ndarray
        2D input attenuation array.
    center : tuple of (int, int), optional
        Lesion centroid (row, col) in pixel coordinates.
        Defaults to an anatomical soft-tissue region in the Shepp-Logan phantom (~55% row, ~45% col).
    radius : float, optional
        Radius (or semi-major axis) in pixels. Default is 8.0.
    contrast : float, optional
        Contrast offset relative to background attenuation.
        Can be positive (hyper-attenuating) or negative (hypo-attenuating).
    shape : str, optional
        Target geometry: "circle" or "ellipse".
    eccentricity : float, optional
        Ratio of semi-minor axis to semi-major axis for elliptical targets (default 1.0).
    background_gap : float, optional
        Margin in pixels between lesion boundary and background annulus.
    background_thickness : float, optional
        Radial thickness of the background evaluation annulus in pixels.

    Returns
    -------
    modified_phantom : np.ndarray
        Phantom with the synthetic target inserted.
    lesion_mask : np.ndarray
        Boolean 2D mask where True indicates pixels inside the lesion ROI.
    background_mask : np.ndarray
        Boolean 2D mask where True indicates background reference pixels.
    metadata : dict
        Descriptive properties of the lesion and surrounding ROI.
    """
    height, width = phantom.shape

    if center is None:
        # Default soft tissue location within homogeneous brain parenchyma (~64% row, ~28% col)
        center = (int(height * 0.64), int(width * 0.28))

    row_c, col_c = center

    if not (0 <= row_c < height and 0 <= col_c < width):
        raise ValueError(f"Lesion center {center} is outside image dimensions ({height}, {width}).")

    y_indices, x_indices = np.ogrid[:height, :width]
    dy = y_indices - row_c
    dx = x_indices - col_c

    # Normalized distance metric for circle/ellipse
    ry = radius
    rx = radius * max(0.1, eccentricity)

    dist_sq = (dy / ry) ** 2 + (dx / rx) ** 2
    lesion_mask = dist_sq <= 1.0

    # Background annulus defined between (radius + background_gap) and (radius + background_gap + background_thickness)
    r_annulus_inner = radius + background_gap
    r_annulus_outer = radius + background_gap + background_thickness

    dist_annulus_inner = (dy / r_annulus_inner) ** 2 + (dx / (r_annulus_inner * max(0.1, eccentricity))) ** 2
    dist_annulus_outer = (dy / r_annulus_outer) ** 2 + (dx / (r_annulus_outer * max(0.1, eccentricity))) ** 2

    raw_background_mask = (dist_annulus_inner > 1.0) & (dist_annulus_outer <= 1.0)

    # Exclude external air/vacuum (< 0.05 attenuation) from background reference
    background_mask = raw_background_mask & (phantom > 0.05)

    if np.sum(lesion_mask) == 0:
        raise ValueError("Lesion radius is too small to cover any pixels.")

    # Create modified phantom with contrast added inside lesion ROI
    modified_phantom = phantom.copy()
    modified_phantom[lesion_mask] = np.clip(modified_phantom[lesion_mask] + contrast, 0.0, 1.0)

    # Calculate actual ground-truth local contrast
    mu_lesion_gt = float(np.mean(modified_phantom[lesion_mask]))
    mu_bg_gt = float(np.mean(modified_phantom[background_mask])) if np.any(background_mask) else 0.0
    true_contrast = mu_lesion_gt - mu_bg_gt

    metadata: Dict[str, Any] = {
        "center": center,
        "radius": radius,
        "nominal_contrast": contrast,
        "true_ground_truth_contrast": true_contrast,
        "num_lesion_pixels": int(np.sum(lesion_mask)),
        "num_background_pixels": int(np.sum(background_mask)),
        "shape": shape,
    }

    return modified_phantom, lesion_mask, background_mask, metadata
