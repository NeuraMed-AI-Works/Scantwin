"""
Experiment A: Photon Budget vs Image Quality Sweep
===================================================
Investigates the effect of varying incident photon budget I0 (relative exposure proxy)
on objective CT reconstruction quality (MSE, RMSE, PSNR, SSIM) and lesion CNR,
holding projection view count constant.

STATISTICAL RIGOR:
Evaluates multiple Poisson noise realizations per exposure level to calculate
mean and standard deviation.
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


def run_dose_quality_sweep(
    photon_levels=(1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6),
    num_projections=180,
    n_realizations=5,
    size=256,
    save_outputs=True,
):
    print("=" * 70)
    print("Experiment A: Incident Photon Budget Sweep")
    print("=" * 70)
    print(RESEARCH_DISCLAIMER)

    # Generate Phantom and synthetic target
    base = generate_shepp_logan(size=size)
    phantom, lesion_mask, bg_mask, _ = insert_lesion(base, radius=8.0, contrast=0.08)

    records = []

    for i0 in photon_levels:
        print(f"Evaluating photon budget I0 = {i0:.1e} ({n_realizations} realizations)...")
        ssims, psnrs, rmses, cnrs = [], [], [], []

        for r in range(n_realizations):
            seed = int(i0) + r * 101
            noisy_sino, _, theta, meta = simulate_acquisition(
                phantom,
                num_projections=num_projections,
                photon_budget=i0,
                seed=seed,
            )
            recon, _ = reconstruct(noisy_sino, theta, method="FBP", filter_name="shepp-logan")
            q = evaluate_image_quality(phantom, recon)
            c = compute_cnr(recon, lesion_mask, bg_mask)

            ssims.append(q["ssim"])
            psnrs.append(q["psnr"])
            rmses.append(q["rmse"])
            cnrs.append(c)

        records.append({
            "photon_budget": i0,
            "projections": num_projections,
            "relative_exposure_pct": (i0 / 1e5) * 100.0,
            "ssim_mean": float(np.mean(ssims)),
            "ssim_std": float(np.std(ssims)),
            "psnr_mean": float(np.mean(psnrs)),
            "psnr_std": float(np.std(psnrs)),
            "rmse_mean": float(np.mean(rmses)),
            "rmse_std": float(np.std(rmses)),
            "cnr_mean": float(np.mean(cnrs)),
            "cnr_std": float(np.std(cnrs)),
        })

    df = pd.DataFrame(records)

    if save_outputs:
        os.makedirs(os.path.join(PROJECT_ROOT, "results", "tables"), exist_ok=True)
        os.makedirs(os.path.join(PROJECT_ROOT, "results", "figures"), exist_ok=True)

        table_path = os.path.join(PROJECT_ROOT, "results", "tables", "dose_quality_sweep.csv")
        df.to_csv(table_path, index=False)
        print(f"Saved results table to: {table_path}")

        # Plot 2x2 multi-panel figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=150)

        # SSIM vs Photon Budget
        axes[0, 0].errorbar(df["photon_budget"], df["ssim_mean"], yerr=df["ssim_std"], fmt="o-", capsize=4, color="#1f77b4")
        axes[0, 0].set_xscale("log")
        axes[0, 0].set_xlabel("Relative Incident Photon Budget I₀ (log scale)")
        axes[0, 0].set_ylabel("SSIM (Mean ± 1 SD)")
        axes[0, 0].set_title("SSIM vs Simulated Photon Budget", fontweight="bold")
        axes[0, 0].grid(True, which="both", ls="--", alpha=0.5)

        # CNR vs Photon Budget
        axes[0, 1].errorbar(df["photon_budget"], df["cnr_mean"], yerr=df["cnr_std"], fmt="s-", capsize=4, color="#2ca02c")
        axes[0, 1].set_xscale("log")
        axes[0, 1].set_xlabel("Relative Incident Photon Budget I₀ (log scale)")
        axes[0, 1].set_ylabel("Lesion CNR (Mean ± 1 SD)")
        axes[0, 1].set_title("Lesion CNR vs Simulated Photon Budget", fontweight="bold")
        axes[0, 1].axhline(3.0, color="red", linestyle=":", label="Visibility Criterion (CNR=3.0)")
        axes[0, 1].legend()
        axes[0, 1].grid(True, which="both", ls="--", alpha=0.5)

        # PSNR vs Photon Budget
        axes[1, 0].errorbar(df["photon_budget"], df["psnr_mean"], yerr=df["psnr_std"], fmt="^-", capsize=4, color="#d62728")
        axes[1, 0].set_xscale("log")
        axes[1, 0].set_xlabel("Relative Incident Photon Budget I₀ (log scale)")
        axes[1, 0].set_ylabel("PSNR (dB) (Mean ± 1 SD)")
        axes[1, 0].set_title("PSNR vs Simulated Photon Budget", fontweight="bold")
        axes[1, 0].grid(True, which="both", ls="--", alpha=0.5)

        # RMSE vs Photon Budget
        axes[1, 1].errorbar(df["photon_budget"], df["rmse_mean"], yerr=df["rmse_std"], fmt="d-", capsize=4, color="#9467bd")
        axes[1, 1].set_xscale("log")
        axes[1, 1].set_xlabel("Relative Incident Photon Budget I₀ (log scale)")
        axes[1, 1].set_ylabel("RMSE (Mean ± 1 SD)")
        axes[1, 1].set_title("RMSE vs Simulated Photon Budget", fontweight="bold")
        axes[1, 1].grid(True, which="both", ls="--", alpha=0.5)

        fig.suptitle(
            "ScanTwin Experiment A: Photon Budget vs CT Image Quality\n(Constant 180 Projections, Shepp-Logan Filter)",
            fontsize=13,
            fontweight="bold",
        )
        plt.tight_layout()

        fig_path = os.path.join(PROJECT_ROOT, "results", "figures", "dose_quality_sweep.png")
        plt.savefig(fig_path, bbox_inches="tight")
        print(f"Saved figure to: {fig_path}")

    return df


if __name__ == "__main__":
    run_dose_quality_sweep()
