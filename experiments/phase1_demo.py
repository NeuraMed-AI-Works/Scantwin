"""
Phase 1 Demonstration and Sanity Verification Script
===================================================
Executes the foundational CT acquisition and reconstruction pipeline:
1. Virtual Shepp-Logan phantom generation (256x256)
2. Synthetic low-contrast lesion insertion
3. Continuous parallel-beam Radon forward projection (Sinogram)
4. Filtered Back Projection (FBP) analytical reconstruction
5. Metric evaluation (MSE, RMSE, PSNR, SSIM, CNR)
6. Error map computation and 4-panel visual reporting

DISCLAIMER:
Research and educational prototype. ScanTwin uses simplified CT acquisition and
exposure models and is not intended for clinical diagnosis, patient-specific
dosimetry, or scanner protocol prescription.
"""

import os
import sys

# Ensure repository root is in python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import numpy as np

from config import RESEARCH_DISCLAIMER
from simulation.phantom import generate_shepp_logan
from simulation.lesion import insert_lesion
from simulation.projector import compute_radon_projection
from reconstruction.reconstruction_engine import reconstruct
from metrics.image_quality import evaluate_image_quality
from metrics.cnr import compute_cnr
from metrics.detectability import assess_detectability
from utils.plotting import plot_phase1_overview


def run_phase1_demo():
    print("=" * 70)
    print("ScanTwin: Milestone 1 Verification Pipeline")
    print("=" * 70)
    print(RESEARCH_DISCLAIMER)
    print("-" * 70)

    # 1. Generate Virtual Phantom
    size = 256
    print(f"[Step 1] Generating 2D Shepp-Logan phantom ({size}x{size})...")
    base_phantom = generate_shepp_logan(size=size)

    # 2. Insert Synthetic Low-Contrast Lesion
    lesion_radius = 8.0
    lesion_contrast = 0.08
    print(f"[Step 2] Inserting synthetic target (r={lesion_radius} px, contrast={lesion_contrast})...")
    gt_phantom, lesion_mask, bg_mask, lesion_meta = insert_lesion(
        base_phantom,
        radius=lesion_radius,
        contrast=lesion_contrast,
    )
    print(f"         Lesion active pixels: {lesion_meta['num_lesion_pixels']}")
    print(f"         Background ROI pixels: {lesion_meta['num_background_pixels']}")

    # 3. Compute Forward Projections (Sinogram)
    num_projections = 180
    print(f"[Step 3] Computing parallel-beam Radon transform ({num_projections} angles)...")
    sinogram, theta = compute_radon_projection(
        gt_phantom,
        num_projections=num_projections,
        angular_range=(0.0, 180.0),
        circle=True,
    )
    print(f"         Sinogram dimensions: {sinogram.shape} (detectors x angles)")

    # 4. FBP Reconstruction
    print("[Step 4] Reconstructing slice using Filtered Back Projection (ramp filter)...")
    recon_image, recon_meta = reconstruct(
        sinogram,
        theta,
        method="FBP",
        filter_name="ramp",
        circle=True,
    )
    print(f"         Reconstruction completed in {recon_meta['elapsed_time_ms']:.1f} ms")
    print(f"         Recon intensity range: [{recon_meta['min_val']:.4f}, {recon_meta['max_val']:.4f}]")

    # 5. Quantitative Metric Assessment
    print("[Step 5] Computing quantitative fidelity and detectability metrics...")
    quality_metrics = evaluate_image_quality(gt_phantom, recon_image)
    cnr_val = compute_cnr(recon_image, lesion_mask, bg_mask)
    quality_metrics["cnr"] = cnr_val

    detectability = assess_detectability(
        recon_image,
        gt_phantom,
        lesion_mask,
        bg_mask,
        cnr_threshold=3.0,
    )

    print("-" * 70)
    print(f"  MSE  : {quality_metrics['mse']:.6e}")
    print(f"  RMSE : {quality_metrics['rmse']:.6f}")
    print(f"  PSNR : {quality_metrics['psnr']:.2f} dB")
    print(f"  SSIM : {quality_metrics['ssim']:.4f}")
    print(f"  CNR  : {quality_metrics['cnr']:.2f}")
    print(f"  Target Visibility Status: {detectability['status_label']} (Threshold: {detectability['cnr_threshold']})")
    print("-" * 70)

    # 6. Save Overview Figure
    os.makedirs(os.path.join(PROJECT_ROOT, "results", "figures"), exist_ok=True)
    figure_path = os.path.join(PROJECT_ROOT, "results", "figures", "phase1_demo.png")
    print(f"[Step 6] Saving publication figure to: {figure_path}")
    plot_phase1_overview(
        ground_truth=gt_phantom,
        sinogram=sinogram,
        theta=theta,
        reconstruction=recon_image,
        metrics=quality_metrics,
        lesion_mask=lesion_mask,
        save_path=figure_path,
    )
    print("Verification complete. Figure saved successfully.")
    print("=" * 70)

    return quality_metrics


if __name__ == "__main__":
    run_phase1_demo()
