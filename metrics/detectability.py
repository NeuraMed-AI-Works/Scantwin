"""
Lesion Detectability Module
===========================
Evaluates experimental target visibility criteria based on objective CNR thresholds.

DISCLAIMER:
Labels strictly denote physical detectability criteria (e.g. "Target visibility
criterion satisfied"). This does NOT denote clinical diagnosis, cancer detection,
or patient diagnostic outcome.
"""

from typing import Dict, Any
import numpy as np

from metrics.cnr import compute_cnr, compute_contrast_recovery


def assess_detectability(
    reconstruction: np.ndarray,
    ground_truth: np.ndarray,
    lesion_mask: np.ndarray,
    background_mask: np.ndarray,
    cnr_threshold: float = 3.0,
) -> Dict[str, Any]:
    """
    Assess target visibility against a researcher-defined CNR threshold.

    Parameters
    ----------
    reconstruction : np.ndarray
        Reconstructed CT image.
    ground_truth : np.ndarray
        Ideal ground truth image.
    lesion_mask : np.ndarray
        Target mask.
    background_mask : np.ndarray
        Background ROI mask.
    cnr_threshold : float, optional
        Minimum acceptable CNR for visibility (default 3.0).

    Returns
    -------
    result : dict
        Contains CNR, contrast recovery, visibility status label, and threshold.
    """
    cnr = compute_cnr(reconstruction, lesion_mask, background_mask)
    crc = compute_contrast_recovery(reconstruction, ground_truth, lesion_mask, background_mask)

    satisfied = cnr >= cnr_threshold
    status_label = (
        "✓ Experimental criterion satisfied"
        if satisfied
        else "✗ Experimental criterion not satisfied"
    )

    return {
        "cnr": cnr,
        "contrast_recovery": crc,
        "cnr_threshold": cnr_threshold,
        "satisfied": satisfied,
        "status_label": status_label,
    }
