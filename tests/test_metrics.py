"""
Unit tests for Image Quality and Detectability metrics.
"""

import pytest
import numpy as np

from metrics.image_quality import (
    compute_mse,
    compute_rmse,
    compute_psnr,
    compute_ssim,
    evaluate_image_quality,
)
from metrics.cnr import compute_cnr, compute_contrast_recovery
from metrics.detectability import assess_detectability


def test_metrics_identity():
    """Identical reference and test images must produce ideal metric values."""
    img = np.ones((64, 64), dtype=np.float64) * 0.5
    img[20:40, 20:40] = 0.8

    assert compute_mse(img, img) == 0.0
    assert compute_rmse(img, img) == 0.0
    assert compute_psnr(img, img) >= 99.0
    assert np.isclose(compute_ssim(img, img), 1.0)


def test_metrics_sensitivity_to_degradation():
    """Adding disturbance should strictly degrade MSE and SSIM."""
    img = np.ones((64, 64), dtype=np.float64) * 0.5
    noisy = img + np.random.normal(0, 0.05, img.shape)

    mse = compute_mse(img, noisy)
    ssim = compute_ssim(img, noisy)

    assert mse > 0.0
    assert ssim < 1.0


def test_cnr_controlled():
    """Verify CNR calculation on a synthetic binary image."""
    img = np.zeros((50, 50), dtype=np.float64)
    lesion_mask = np.zeros((50, 50), dtype=bool)
    bg_mask = np.zeros((50, 50), dtype=bool)

    lesion_mask[10:20, 10:20] = True
    bg_mask[30:40, 30:40] = True

    # Case 1: lesion = 0.8, bg mean = 0.4 with std = 0.1
    np.random.seed(42)
    img[lesion_mask] = 0.8
    img[bg_mask] = 0.4 + np.random.normal(0, 0.1, np.sum(bg_mask))

    cnr = compute_cnr(img, lesion_mask, bg_mask)
    expected_cnr = (0.8 - 0.4) / 0.1  # ~ 4.0
    assert 3.0 <= cnr <= 5.0, f"Expected CNR around 4.0, got {cnr}"

    # Detectability assessment
    det_res = assess_detectability(img, img, lesion_mask, bg_mask, cnr_threshold=3.0)
    assert det_res["satisfied"] is True
    assert det_res["status_label"] == "✓ Experimental criterion satisfied"


def test_dimension_mismatch_raises():
    """Verify error on mismatched dimensions."""
    ref = np.zeros((32, 32))
    test = np.zeros((64, 64))
    with pytest.raises(ValueError):
        compute_mse(ref, test)
