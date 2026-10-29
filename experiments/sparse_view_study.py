"""
Experiment B: Sparse-View CT Sampling Study
===========================================
Investigates the impact of projection angular undersampling (sparse views)
on reconstruction artifacts and quantitative metrics, holding photon budget constant.

Demonstrates streaking and angular undersampling degradation in Filtered Back Projection.
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


def run_sparse_view_study(
    view_counts=(30, 45, 60, 90, 120, 180, 360),
    photon_budget=1e5,
    n_realizations=3,
    size=256,
    save_outputs=True,
):
    print("=" * 70)
    print("Experiment B: Sparse-View CT Undersampling Study")
    print("=" * 70)
    print(RESEARCH_DISCLAIMER)

    base = generate_shepp_logan(size=size)
    phantom, lesion_mask, bg_mask, _ = insert_lesion(base, radius=8.0, contrast=0.08)

    records = []

    for views in view_counts:
        print(f"Evaluating projection view count = {views} ({n_realizations} realizations)...")
        ssims, psnrs, rmses, cnrs = [], [], [], []

        for r in range(n_realizations):
            seed = views * 100 + r
            noisy_sino, _, theta, meta = simulate_acquisition(
                phantom,
                num_projections=views,
                photon_budget=photon_budget,
                seed=seed,
            )
            recon, _ = reconstruct(noisy_sino, theta, method="FBP", filter_name="ramp")
            q = evaluate_image_quality(phantom, recon)
            c = compute_cnr(recon, lesion_mask, bg_mask)

            ssims.append(q["ssim"])
            psnrs.append(q["psnr"])
            rmses.append(q["rmse"])
            cnrs.append(c)

        records.append({
            "num_projections": views,
            "photon_budget": photon_budget,
            "projection_fraction_pct": (views / 360.0) * 100.0,
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

        table_path = os.path.join(PROJECT_ROOT, "results", "tables", "sparse_view_study.csv")
        df.to_csv(table_path, index=False)
        print(f"Saved results table to: {table_path}")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

        # SSIM vs View Count
        axes[0].errorbar(df["num_projections"], df["ssim_mean"], yerr=df["ssim_std"], fmt="o-", capsize=4, color="#1f77b4")
        axes[0].set_xlabel("Projection View Count (θ samples across 180°)")
        axes[0].set_ylabel("SSIM (Mean ± 1 SD)")
        axes[0].set_title("SSIM vs Angular Sampling Density", fontweight="bold")
        axes[0].grid(True, ls="--", alpha=0.5)

        # CNR vs View Count
        axes[1].errorbar(df["num_projections"], df["cnr_mean"], yerr=df["cnr_std"], fmt="s-", capsize=4, color="#2ca02c")
        axes[1].set_xlabel("Projection View Count (θ samples across 180°)")
        axes[1].set_ylabel("Lesion CNR (Mean ± 1 SD)")
        axes[1].set_title("Lesion CNR vs Angular Sampling Density", fontweight="bold")
        axes[1].axhline(3.0, color="red", linestyle=":", label="Visibility Criterion (CNR=3.0)")
        axes[1].legend()
        axes[1].grid(True, ls="--", alpha=0.5)

        fig.suptitle(
            f"ScanTwin Experiment B: Sparse-View CT Sampling Study\n(Constant Photon Budget I₀ = {photon_budget:.1e}, FBP Ramp Filter)",
            fontsize=13,
            fontweight="bold",
        )
        plt.tight_layout()

        fig_path = os.path.join(PROJECT_ROOT, "results", "figures", "sparse_view_study.png")
        plt.savefig(fig_path, bbox_inches="tight")
        print(f"Saved figure to: {fig_path}")

    return df


if __name__ == "__main__":
    run_sparse_view_study()
