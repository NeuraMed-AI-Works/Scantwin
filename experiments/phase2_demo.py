"""
ScanTwin Phase 2 Final Validation Demonstration Script
======================================================
Executes the final 3-way head-to-head comparison:
- CASE A: Reference Baseline (180 views, I0 = 1e5, FBP)
- CASE B: Extreme Low Quality (30 views, I0 = 1e3, FBP)
- CASE C: Optimized Protocol (discovered via Bayesian optimizer)

Computes, prints, and visualizes:
Ground Truth | Sinogram | Reconstructed CT | Error Map
SSIM | PSNR | RMSE | CNR | Relative Exposure | Feasibility

DISCLAIMER:
Research and educational prototype. ScanTwin uses simplified CT acquisition and
exposure models and is not intended for clinical diagnosis, patient-specific
dosimetry, or scanner protocol prescription.
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import RESEARCH_DISCLAIMER, AcquisitionProtocol
from simulation.phantom import generate_shepp_logan
from simulation.lesion import insert_lesion
from simulation.acquisition import simulate_acquisition
from reconstruction.reconstruction_engine import reconstruct
from metrics.image_quality import evaluate_image_quality
from metrics.cnr import compute_cnr, compute_roi_statistics
from digital_twin.twin import CTDigitalTwin
from optimization.protocol_optimizer import ProtocolOptimizer
from utils.normalization import compute_absolute_error_map


def run_phase2_demonstration(size=256, save_outputs=True):
    print("=" * 75)
    print("ScanTwin Phase 2: Final Scientific Validation Demonstration")
    print("=" * 75)
    print(RESEARCH_DISCLAIMER)
    print("-" * 75)

    # 1. Virtual Phantom & Target Setup
    base = generate_shepp_logan(size=size)
    phantom, lesion_mask, bg_mask, lesion_meta = insert_lesion(base, radius=8.0, contrast=0.08)

    twin = CTDigitalTwin(phantom, lesion_mask, bg_mask, lesion_meta)

    # 2. Case A: Reference Baseline (180 views, I0 = 1e5, FBP shepp-logan)
    proto_a = AcquisitionProtocol(photon_budget=1e5, num_projections=180, reconstruction_method="FBP", fbp_filter="shepp-logan")
    state_a, recon_a, sino_a = twin.evaluate_protocol(proto_a, target_ssim_min=0.88, target_cnr_min=3.0)

    # 3. Case B: Extreme Low Quality (30 views, I0 = 1e3, FBP ramp)
    proto_b = AcquisitionProtocol(photon_budget=1e3, num_projections=30, reconstruction_method="FBP", fbp_filter="ramp")
    state_b, recon_b, sino_b = twin.evaluate_protocol(proto_b, target_ssim_min=0.88, target_cnr_min=3.0)

    # 4. Case C: AI Protocol Optimizer Search
    print("Running Bayesian optimizer to identify optimal candidate satisfying SSIM >= 0.88, CNR >= 3.0...")
    optimizer = ProtocolOptimizer(twin, target_ssim_min=0.88, target_cnr_min=3.0, seed=42)
    best_opt_state, _, opt_summary = optimizer.optimize(n_trials=20, n_realizations=1)
    print(f"Optimizer Summary: {opt_summary}")

    if best_opt_state is not None:
        proto_c = AcquisitionProtocol(
            photon_budget=best_opt_state.photon_budget,
            num_projections=best_opt_state.num_projections,
            reconstruction_method=best_opt_state.reconstruction_method,
            fbp_filter=best_opt_state.fbp_filter,
            sart_iterations=best_opt_state.sart_iterations,
        )
    else:
        # Fallback candidate for demonstration
        proto_c = AcquisitionProtocol(photon_budget=2.5e4, num_projections=150, reconstruction_method="FBP", fbp_filter="shepp-logan")

    state_c, recon_c, sino_c = twin.evaluate_protocol(proto_c, target_ssim_min=0.88, target_cnr_min=3.0)

    # 5. Format Comparison Table
    cases = [
        {"Case": "CASE A: Reference Baseline", "state": state_a, "recon": recon_a, "sino": sino_a},
        {"Case": "CASE B: Extreme Low Quality", "state": state_b, "recon": recon_b, "sino": sino_b},
        {"Case": "CASE C: AI-Optimized Candidate", "state": state_c, "recon": recon_c, "sino": sino_c},
    ]

    records = []
    for c in cases:
        st = c["state"]
        rel_red = max(0.0, 100.0 - st.relative_photon_exposure_pct)
        records.append({
            "Case": c["Case"],
            "Photon Budget (I0)": f"{st.photon_budget:.1e}",
            "Views": st.num_projections,
            "Reconstruction": f"{st.reconstruction_method} ({st.fbp_filter})",
            "SSIM": f"{st.ssim:.4f}",
            "PSNR (dB)": f"{st.psnr:.2f}",
            "RMSE": f"{st.rmse:.4f}",
            "CNR": f"{st.cnr:.2f}",
            "Rel. Exposure": f"{st.relative_photon_exposure_pct:.1f}%",
            "Exposure Reduction": f"{rel_red:.1f}%",
            "Feasible?": "YES" if st.is_feasible else "NO",
        })

    table_df = pd.DataFrame(records)
    print("-" * 75)
    print(table_df.to_string(index=False))
    print("-" * 75)

    if save_outputs:
        out_dir = os.path.join(PROJECT_ROOT, "results", "phase2")
        os.makedirs(out_dir, exist_ok=True)
        csv_file = os.path.join(out_dir, "phase2_demonstration.csv")
        table_df.to_csv(csv_file, index=False)

        # Multi-row figure: 3 cases x 4 columns (Ground Truth, Sinogram, Recon, Error Map)
        fig, axes = plt.subplots(3, 4, figsize=(18, 13), dpi=150)

        for row_idx, c in enumerate(cases):
            st = c["state"]
            recon_img = c["recon"]
            sino_img = c["sino"]
            err_map = compute_absolute_error_map(phantom, recon_img)

            # Col 0: Ground Truth
            axes[row_idx, 0].imshow(phantom, cmap="gray", vmin=0.0, vmax=1.0)
            axes[row_idx, 0].contour(lesion_mask, colors="red", linewidths=1.0, levels=[0.5])
            axes[row_idx, 0].set_title(f"{c['Case']}\nGround Truth", fontsize=10, fontweight="bold")
            axes[row_idx, 0].axis("off")

            # Col 1: Sinogram
            axes[row_idx, 1].imshow(sino_img, cmap="viridis", aspect="auto")
            axes[row_idx, 1].set_title(f"Sinogram ({st.num_projections} views, I₀={st.photon_budget:.1e})", fontsize=10)
            axes[row_idx, 1].set_xlabel("Projection Angle")
            axes[row_idx, 1].set_ylabel("Detector Bin")

            # Col 2: Reconstruction
            axes[row_idx, 2].imshow(recon_img, cmap="gray", vmin=0.0, vmax=1.0)
            axes[row_idx, 2].contour(lesion_mask, colors="red", linewidths=0.8, levels=[0.5])
            axes[row_idx, 2].set_title(f"Reconstructed CT\nSSIM: {st.ssim:.3f} | CNR: {st.cnr:.2f}", fontsize=10)
            axes[row_idx, 2].axis("off")

            # Col 3: Error Map
            axes[row_idx, 3].imshow(err_map, cmap="inferno", vmin=0.0, vmax=0.35)
            axes[row_idx, 3].set_title(f"Error Map |GT - Recon|\nRMSE: {st.rmse:.4f} | Feasible: {'YES' if st.is_feasible else 'NO'}", fontsize=10)
            axes[row_idx, 3].axis("off")

        fig.suptitle("ScanTwin Phase 2 Validation Demonstration: Reference vs Extreme Low Quality vs AI-Optimized", fontsize=13, fontweight="bold", y=0.995)
        plt.tight_layout()
        fig_file = os.path.join(out_dir, "phase2_demonstration.png")
        plt.savefig(fig_file, bbox_inches="tight")
        plt.close()
        print(f"Saved demonstration plot to: {fig_file}")

    print("=" * 75)
    return table_df


if __name__ == "__main__":
    run_phase2_demonstration()
