"""
Plotting and Visualization Utilities
====================================
Generates publication-quality figures using matplotlib for simulation,
reconstruction, and image quality evaluation.
"""

from typing import Dict, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np

from utils.normalization import compute_absolute_error_map


def plot_phase1_overview(
    ground_truth: np.ndarray,
    sinogram: np.ndarray,
    theta: np.ndarray,
    reconstruction: np.ndarray,
    metrics: Dict[str, float],
    lesion_mask: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Generate a 4-panel comparative figure for Phase 1 verification:
    1. Ground Truth Phantom
    2. Parallel-Beam Sinogram
    3. FBP Reconstruction
    4. Absolute Error Map

    Parameters
    ----------
    ground_truth : np.ndarray
        Ground truth attenuation image.
    sinogram : np.ndarray
        Sinogram projection array.
    theta : np.ndarray
        Projection angles in degrees.
    reconstruction : np.ndarray
        Reconstructed CT slice.
    metrics : dict
        Computed metrics (MSE, RMSE, PSNR, SSIM, CNR).
    lesion_mask : np.ndarray, optional
        Boolean mask of lesion target to outline.
    save_path : str, optional
        Filepath to save figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    error_map = compute_absolute_error_map(ground_truth, reconstruction)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), dpi=150)

    # 1. Ground Truth
    im0 = axes[0].imshow(ground_truth, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("Ground Truth Phantom", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("X (pixels)")
    axes[0].set_ylabel("Y (pixels)")
    if lesion_mask is not None and np.any(lesion_mask):
        axes[0].contour(lesion_mask, colors="red", linewidths=1.0, levels=[0.5])
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # 2. Sinogram
    im1 = axes[1].imshow(
        sinogram,
        cmap="viridis",
        aspect="auto",
        extent=[theta[0], theta[-1], sinogram.shape[0], 0],
    )
    axes[1].set_title(f"Sinogram ({len(theta)} views)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Projection Angle θ (°)")
    axes[1].set_ylabel("Detector Bin s")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. FBP Reconstruction
    im2 = axes[2].imshow(reconstruction, cmap="gray", vmin=0.0, vmax=1.0)
    axes[2].set_title("FBP Reconstruction", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("X (pixels)")
    axes[2].set_ylabel("Y (pixels)")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    # 4. Error Map
    im3 = axes[3].imshow(error_map, cmap="inferno", vmin=0.0)
    axes[3].set_title("Absolute Error Map |GT - Recon|", fontsize=12, fontweight="bold")
    axes[3].set_xlabel("X (pixels)")
    axes[3].set_ylabel("Y (pixels)")
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    # Annotate summary metrics at bottom
    cnr_str = f" | CNR: {metrics['cnr']:.2f}" if "cnr" in metrics else ""
    summary_text = (
        f"MSE: {metrics['mse']:.2e} | RMSE: {metrics['rmse']:.4f} | "
        f"PSNR: {metrics['psnr']:.2f} dB | SSIM: {metrics['ssim']:.4f}{cnr_str}"
    )
    fig.suptitle(f"ScanTwin — Milestone 1 CT Pipeline Verification\n{summary_text}", fontsize=13, y=1.03)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    return fig
