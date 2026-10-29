"""
Experiment C: Reconstruction Algorithm Comparison (FBP vs SART)
==============================================================
Direct comparative study contrasting analytical Filtered Back Projection (FBP)
against algebraic iterative reconstruction (SART) under four distinct regimes:
1. High-Quality Acquisition (180 views, I0 = 1e5)
2. Photon-Limited Acquisition (180 views, I0 = 5e3)
3. Sparse-View Acquisition (45 views, I0 = 1e5)
4. Combined Degradation (45 views, I0 = 5e3)

DISCLAIMER:
SART represents an academic iterative baseline. Results demonstrate general properties
of algebraic iterative methods versus standard analytical filtering.
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

from config import RESEARCH_DISCLAIMER
from simulation.phantom import generate_shepp_logan
from simulation.lesion import insert_lesion
from simulation.acquisition import simulate_acquisition
from reconstruction.reconstruction_engine import reconstruct
from metrics.image_quality import evaluate_image_quality
from metrics.cnr import compute_cnr


def run_reconstruction_comparison(size=256, save_outputs=True):
    print("=" * 70)
    print("Experiment C: FBP vs SART Reconstruction Comparison")
    print("=" * 70)
    print(RESEARCH_DISCLAIMER)

    base = generate_shepp_logan(size=size)
    phantom, lesion_mask, bg_mask, _ = insert_lesion(base, radius=8.0, contrast=0.08)

    regimes = [
        {"name": "1. High-Quality", "views": 180, "i0": 1e5},
        {"name": "2. Photon-Limited", "views": 180, "i0": 5e3},
        {"name": "3. Sparse-View", "views": 45, "i0": 1e5},
        {"name": "4. Combined Degradation", "views": 45, "i0": 5e3},
    ]

    records = []
    recon_images = {}

    for regime in regimes:
        name = regime["name"]
        views = regime["views"]
        i0 = regime["i0"]
        print(f"Simulating regime: {name} (N={views}, I0={i0:.1e})...")

        noisy_sino, _, theta, meta = simulate_acquisition(
            phantom,
            num_projections=views,
            photon_budget=i0,
            seed=42,
        )

        # FBP Ramp
        fbp_recon, fbp_meta = reconstruct(noisy_sino, theta, method="FBP", filter_name="ramp")
        fbp_q = evaluate_image_quality(phantom, fbp_recon)
        fbp_cnr = compute_cnr(fbp_recon, lesion_mask, bg_mask)

        # SART (2 iterations)
        sart_recon, sart_meta = reconstruct(noisy_sino, theta, method="SART", iterations=2)
        sart_q = evaluate_image_quality(phantom, sart_recon)
        sart_cnr = compute_cnr(sart_recon, lesion_mask, bg_mask)

        recon_images[name] = {"FBP": fbp_recon, "SART": sart_recon}

        records.append({
            "Regime": name,
            "Algorithm": "FBP (ramp)",
            "Views": views,
            "Photon Budget (I0)": f"{i0:.1e}",
            "SSIM": fbp_q["ssim"],
            "PSNR (dB)": fbp_q["psnr"],
            "RMSE": fbp_q["rmse"],
            "CNR": fbp_cnr,
            "Elapsed Time (ms)": fbp_meta["elapsed_time_ms"],
        })

        records.append({
            "Regime": name,
            "Algorithm": "SART (2 iter)",
            "Views": views,
            "Photon Budget (I0)": f"{i0:.1e}",
            "SSIM": sart_q["ssim"],
            "PSNR (dB)": sart_q["psnr"],
            "RMSE": sart_q["rmse"],
            "CNR": sart_cnr,
            "Elapsed Time (ms)": sart_meta["elapsed_time_ms"],
        })

    df = pd.DataFrame(records)

    if save_outputs:
        os.makedirs(os.path.join(PROJECT_ROOT, "results", "tables"), exist_ok=True)
        os.makedirs(os.path.join(PROJECT_ROOT, "results", "figures"), exist_ok=True)

        table_path = os.path.join(PROJECT_ROOT, "results", "tables", "reconstruction_comparison.csv")
        df.to_csv(table_path, index=False)
        print(f"Saved results table to: {table_path}")

        # Plot 4x2 visual grid: row = regime, col 0 = FBP, col 1 = SART
        fig, axes = plt.subplots(4, 2, figsize=(10, 16), dpi=150)

        for row_idx, regime in enumerate(regimes):
            name = regime["name"]
            fbp_img = recon_images[name]["FBP"]
            sart_img = recon_images[name]["SART"]

            # Filter dataframe for this regime
            df_fbp = df[(df["Regime"] == name) & (df["Algorithm"].str.startswith("FBP"))].iloc[0]
            df_sart = df[(df["Regime"] == name) & (df["Algorithm"].str.startswith("SART"))].iloc[0]

            axes[row_idx, 0].imshow(fbp_img, cmap="gray", vmin=0.0, vmax=1.0)
            axes[row_idx, 0].set_title(
                f"{name} — FBP\nSSIM: {df_fbp['SSIM']:.3f} | CNR: {df_fbp['CNR']:.2f}",
                fontsize=10,
            )
            axes[row_idx, 0].axis("off")

            axes[row_idx, 1].imshow(sart_img, cmap="gray", vmin=0.0, vmax=1.0)
            axes[row_idx, 1].set_title(
                f"{name} — SART (2 iter)\nSSIM: {df_sart['SSIM']:.3f} | CNR: {df_sart['CNR']:.2f}",
                fontsize=10,
            )
            axes[row_idx, 1].axis("off")

        fig.suptitle(
            "ScanTwin Experiment C: FBP vs SART Comparison Across Acquisition Regimes",
            fontsize=13,
            fontweight="bold",
            y=0.995,
        )
        plt.tight_layout()

        fig_path = os.path.join(PROJECT_ROOT, "results", "figures", "reconstruction_comparison.png")
        plt.savefig(fig_path, bbox_inches="tight")
        print(f"Saved figure to: {fig_path}")

    return df


if __name__ == "__main__":
    run_reconstruction_comparison()
