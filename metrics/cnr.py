"""
Contrast-to-Noise Ratio (CNR) and Contrast Recovery Module
==========================================================
Computes objective regional detectability proxies:
    CNR = |mu_lesion - mu_background| / sigma_background

DISCLAIMER:
CNR serves as a mathematical physical signal-to-noise metric and does not represent
a clinical observer study or radiological diagnostic accuracy.
"""

from typing import Dict, Optional
import numpy as np


def compute_cnr(
    image: np.ndarray,
    lesion_mask: np.ndarray,
    background_mask: np.ndarray,
    eps: float = 1e-8,
) -> float:
    """
    Calculate the Contrast-to-Noise Ratio (CNR) using ROI masks.

    Parameters
    ----------
    image : np.ndarray
        2D reconstructed CT image.
    lesion_mask : np.ndarray
        Boolean 2D mask for target/lesion pixels.
    background_mask : np.ndarray
        Boolean 2D mask for local background pixels.
    eps : float, optional
        Numerical stability epsilon to prevent zero division.

    Returns
    -------
    cnr : float
        Calculated CNR value.
    """
    if not np.any(lesion_mask):
        raise ValueError("Lesion mask contains zero active pixels.")
    if not np.any(background_mask):
        raise ValueError("Background mask contains zero active pixels.")

    lesion_vals = image[lesion_mask]
    bg_vals = image[background_mask]

    mu_lesion = float(np.mean(lesion_vals))
    mu_bg = float(np.mean(bg_vals))
    sigma_bg = float(np.std(bg_vals))

    if sigma_bg < eps:
        # If background noise is zero, contrast is either perfectly resolved or both are identical
        if abs(mu_lesion - mu_bg) > eps:
            return 100.0  # High finite upper ceiling
        return 0.0

    cnr = abs(mu_lesion - mu_bg) / (sigma_bg + eps)
    return float(cnr)


def compute_roi_statistics(
    image: np.ndarray,
    lesion_mask: np.ndarray,
    background_mask: np.ndarray,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """
    Compute comprehensive regional attenuation and noise statistics for lesion and background ROIs.

    Returns
    -------
    stats : dict
        mean_lesion, mean_background, std_background, delta_contrast, and cnr.
    """
    if not np.any(lesion_mask) or not np.any(background_mask):
        return {
            "mean_lesion": 0.0,
            "mean_background": 0.0,
            "std_background": 0.0,
            "delta_contrast": 0.0,
            "cnr": 0.0,
        }

    lesion_vals = image[lesion_mask]
    bg_vals = image[background_mask]

    mu_lesion = float(np.mean(lesion_vals))
    mu_bg = float(np.mean(bg_vals))
    sigma_bg = float(np.std(bg_vals))

    if sigma_bg < eps:
        cnr = 100.0 if abs(mu_lesion - mu_bg) > eps else 0.0
    else:
        cnr = abs(mu_lesion - mu_bg) / (sigma_bg + eps)

    return {
        "mean_lesion": mu_lesion,
        "mean_background": mu_bg,
        "std_background": sigma_bg,
        "delta_contrast": abs(mu_lesion - mu_bg),
        "cnr": float(cnr),
    }


def compute_contrast_recovery(
    reconstruction: np.ndarray,
    ground_truth: np.ndarray,
    lesion_mask: np.ndarray,
    background_mask: np.ndarray,
    eps: float = 1e-8,
) -> float:
    """
    Compute Contrast Recovery Coefficient (CRC):
        CRC = (mu_lesion_recon - mu_bg_recon) / (mu_lesion_gt - mu_bg_gt)

    Evaluates how accurately the reconstruction preserves true physical contrast
    without partial volume or filtering degradation.
    """
    delta_recon = float(np.mean(reconstruction[lesion_mask]) - np.mean(reconstruction[background_mask]))
    delta_gt = float(np.mean(ground_truth[lesion_mask]) - np.mean(ground_truth[background_mask]))

    if abs(delta_gt) < eps:
        return 1.0 if abs(delta_recon) < eps else 0.0

    return float(delta_recon / (delta_gt + eps))
