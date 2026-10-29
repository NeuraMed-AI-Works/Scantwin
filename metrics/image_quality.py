"""
Image Quality Metrics Module
============================
Calculates objective fidelity metrics:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Peak Signal-to-Noise Ratio (PSNR)
- Structural Similarity Index Measure (SSIM)

NORMALIZATION METHODOLOGY:
To maintain rigorous scientific validity, metrics are computed directly on the
unscaled reconstruction arrays matched to the reference ground truth scale.
Reconstructions are NOT independently min-max normalized prior to evaluation,
as artificial scaling masks loss of absolute CT attenuation fidelity.
"""

from typing import Dict, Optional
import numpy as np
import skimage.metrics


def compute_mse(
    reference: np.ndarray,
    test: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """
    Compute Mean Squared Error (MSE) between reference and test images.

    Parameters
    ----------
    reference : np.ndarray
        Ground truth reference 2D image.
    test : np.ndarray
        Reconstructed 2D test image.
    mask : np.ndarray, optional
        Boolean mask defining region of evaluation (e.g. interior FOV).

    Returns
    -------
    mse : float
        Mean squared difference.
    """
    if reference.shape != test.shape:
        raise ValueError(f"Shape mismatch: reference {reference.shape} vs test {test.shape}")

    if mask is not None:
        diff = reference[mask] - test[mask]
    else:
        diff = reference - test

    return float(np.mean(diff ** 2))


def compute_rmse(
    reference: np.ndarray,
    test: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """
    Compute Root Mean Squared Error (RMSE).

    Parameters
    ----------
    reference : np.ndarray
        Ground truth reference 2D image.
    test : np.ndarray
        Reconstructed 2D test image.
    mask : np.ndarray, optional
        Boolean mask for evaluation region.

    Returns
    -------
    rmse : float
        Square root of MSE.
    """
    return float(np.sqrt(compute_mse(reference, test, mask=mask)))


def compute_psnr(
    reference: np.ndarray,
    test: np.ndarray,
    data_range: Optional[float] = None,
    mask: Optional[np.ndarray] = None,
) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) in decibels (dB).

    PSNR = 10 * log10( (data_range^2) / MSE )

    Parameters
    ----------
    reference : np.ndarray
        Ground truth reference 2D image.
    test : np.ndarray
        Reconstructed 2D test image.
    data_range : float, optional
        Dynamic range of reference signal (max - min). If None, computed from reference.
    mask : np.ndarray, optional
        Evaluation mask.

    Returns
    -------
    psnr : float
        PSNR value in dB (capped at 100.0 dB for identical images).
    """
    mse = compute_mse(reference, test, mask=mask)
    if mse == 0.0:
        return 100.0

    if data_range is None:
        data_range = float(np.max(reference) - np.min(reference))
        if data_range == 0.0:
            data_range = 1.0

    psnr = 10.0 * np.log10((data_range ** 2) / mse)
    return float(min(psnr, 100.0))   # cap: numerically (near-)identical images


def compute_ssim(
    reference: np.ndarray,
    test: np.ndarray,
    data_range: Optional[float] = None,
) -> float:
    """
    Compute Structural Similarity Index (SSIM) using scikit-image.

    Parameters
    ----------
    reference : np.ndarray
        Ground truth reference 2D image.
    test : np.ndarray
        Reconstructed 2D test image.
    data_range : float, optional
        Dynamic range of reference signal. If None, computed from reference.

    Returns
    -------
    ssim : float
        SSIM value in [-1.0, 1.0].
    """
    if reference.shape != test.shape:
        raise ValueError(f"Shape mismatch: reference {reference.shape} vs test {test.shape}")

    if data_range is None:
        data_range = float(np.max(reference) - np.min(reference))
        if data_range == 0.0:
            data_range = 1.0

    ssim_val = skimage.metrics.structural_similarity(
        reference,
        test,
        data_range=data_range,
    )
    return float(ssim_val)


def evaluate_image_quality(
    reference: np.ndarray,
    reconstruction: np.ndarray,
    data_range: Optional[float] = None,
    mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Calculate full suite of standard image quality metrics.

    Returns
    -------
    metrics : dict
        Dictionary containing MSE, RMSE, PSNR, and SSIM.
    """
    mse = compute_mse(reference, reconstruction, mask=mask)
    rmse = float(np.sqrt(mse))
    psnr = compute_psnr(reference, reconstruction, data_range=data_range, mask=mask)
    ssim = compute_ssim(reference, reconstruction, data_range=data_range)

    return {
        "mse": mse,
        "rmse": rmse,
        "psnr": psnr,
        "ssim": ssim,
    }
