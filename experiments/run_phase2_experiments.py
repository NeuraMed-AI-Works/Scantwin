"""
ScanTwin Phase 2 Automated Experimental Suite
=============================================
Executes the four rigorous quantitative validation studies for Phase 2:
1. Experiment A: Photon Budget vs Image Quality Study (holding views=180, FBP)
2. Experiment B: Sparse-View CT Sampling Study (holding I0=1e5, FBP)
3. Experiment C: Reconstruction Algorithm Comparison (FBP vs SART across 4 regimes)
4. Experiment D: Combined Grid Study (3x3 grid of I0 x Views with 2D Heatmaps)

Saves all tables (.csv) and figures (.png) to results/phase2/ with exact specified filenames.

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

from config import RESEARCH_DISCLAIMER
from simulation.phantom import generate_shepp_logan
from simulation.lesion import insert_lesion
from simulation.acquisition import simulate_acquisition
from reconstruction.reconstruction_engine import reconstruct
from metrics.image_quality import evaluate_image_quality
from metrics.cnr import compute_cnr, compute_roi_statistics

RESULTS_PHASE2_DIR = os.path.join(PROJECT_ROOT, "results", "phase2")
os.makedirs(RESULTS_PHASE2_DIR, exist_ok=True)


def run_experiment_a(phantom, lesion_mask, bg_mask, n_realizations=5):
    """
    Experiment A: Photon Budget Study
    Vary I0 across 1e3 to 1e6 holding projection count = 180 (FBP shepp-logan).
    """
    print("-" * 70)
    print("Running Phase 2 — Experiment A: Photon Budget Study...")
    photon_levels = [1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6]
    num_views = 180

    records = []
    for i0 in photon_levels:
        ssims, psnrs, rmses, cnrs = [], [], [], []
        for r in range(n_realizations):
            seed = int(i0) + r * 101
            noisy_sino, _, theta, _ = simulate_acquisition(
                phantom, num_projections=num_views, photon_budget=i0, seed=seed
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
            "projections": num_views,
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
    csv_path = os.path.join(RESULTS_PHASE2_DIR, "photon_budget_study.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Plot 1: photon_vs_ssim.png
    plt.figure(figsize=(7, 5), dpi=150)
    plt.errorbar(df["photon_budget"], df["ssim_mean"], yerr=df["ssim_std"], fmt="o-", capsize=4, color="#1f77b4", lw=2)
    plt.xscale("log")
    plt.xlabel("Simulated Incident Photon Budget I₀ (log scale)")
    plt.ylabel("SSIM (Mean ± 1 SD)")
    plt.title("SSIM vs Simulated Photon Budget (180 Projections, FBP)", fontweight="bold")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    p_ssim_path = os.path.join(RESULTS_PHASE2_DIR, "photon_vs_ssim.png")
    plt.savefig(p_ssim_path)
    plt.close()
    print(f"Saved: {p_ssim_path}")

    # Plot 2: photon_vs_cnr.png
    plt.figure(figsize=(7, 5), dpi=150)
    plt.errorbar(df["photon_budget"], df["cnr_mean"], yerr=df["cnr_std"], fmt="s-", capsize=4, color="#2ca02c", lw=2)
    plt.xscale("log")
    plt.axhline(3.0, color="red", linestyle=":", label="Experimental Criterion (CNR=3.0)")
    plt.xlabel("Simulated Incident Photon Budget I₀ (log scale)")
    plt.ylabel("Lesion CNR (Mean ± 1 SD)")
    plt.title("Lesion CNR vs Simulated Photon Budget (180 Projections, FBP)", fontweight="bold")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    p_cnr_path = os.path.join(RESULTS_PHASE2_DIR, "photon_vs_cnr.png")
    plt.savefig(p_cnr_path)
    plt.close()
    print(f"Saved: {p_cnr_path}")

    return df


def run_experiment_b(phantom, lesion_mask, bg_mask, n_realizations=4):
    """
    Experiment B: Sparse-View CT Study
    Vary projection views in [30, 60, 90, 120, 180, 360] holding I0=1e5 (FBP ramp).
    """
    print("-" * 70)
    print("Running Phase 2 — Experiment B: Sparse-View CT Study...")
    view_counts = [30, 60, 90, 120, 180, 360]
    photon_budget = 1e5

    records = []
    for views in view_counts:
        ssims, psnrs, rmses, cnrs = [], [], [], []
        for r in range(n_realizations):
            seed = views * 100 + r
            noisy_sino, _, theta, _ = simulate_acquisition(
                phantom, num_projections=views, photon_budget=photon_budget, seed=seed
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
    csv_path = os.path.join(RESULTS_PHASE2_DIR, "sparse_view_study.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Plot 1: views_vs_ssim.png
    plt.figure(figsize=(7, 5), dpi=150)
    plt.errorbar(df["num_projections"], df["ssim_mean"], yerr=df["ssim_std"], fmt="o-", capsize=4, color="#1f77b4", lw=2)
    plt.xlabel("Projection View Count (θ samples across [0, 180°))")
    plt.ylabel("SSIM (Mean ± 1 SD)")
    plt.title("SSIM vs Projection Sampling Density (I₀ = 10⁵, FBP)", fontweight="bold")
    plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    v_ssim_path = os.path.join(RESULTS_PHASE2_DIR, "views_vs_ssim.png")
    plt.savefig(v_ssim_path)
    plt.close()
    print(f"Saved: {v_ssim_path}")

    # Plot 2: views_vs_cnr.png
    plt.figure(figsize=(7, 5), dpi=150)
    plt.errorbar(df["num_projections"], df["cnr_mean"], yerr=df["cnr_std"], fmt="s-", capsize=4, color="#2ca02c", lw=2)
    plt.axhline(3.0, color="red", linestyle=":", label="Experimental Criterion (CNR=3.0)")
    plt.xlabel("Projection View Count (θ samples across [0, 180°))")
    plt.ylabel("Lesion CNR (Mean ± 1 SD)")
    plt.title("Lesion CNR vs Projection Sampling Density (I₀ = 10⁵, FBP)", fontweight="bold")
    plt.legend()
    plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    v_cnr_path = os.path.join(RESULTS_PHASE2_DIR, "views_vs_cnr.png")
    plt.savefig(v_cnr_path)
    plt.close()
    print(f"Saved: {v_cnr_path}")

    return df


def run_experiment_c(phantom, lesion_mask, bg_mask):
    """
    Experiment C: Reconstruction Comparison (FBP vs SART)
    Compares FBP and SART under:
    1. High-quality (180 views, I0=1e5)
    2. Low-photon (180 views, I0=5e3)
    3. Sparse-view (45 views, I0=1e5)
    4. Combined low-photon + sparse-view (45 views, I0=5e3)
    """
    print("-" * 70)
    print("Running Phase 2 — Experiment C: Reconstruction Comparison (FBP vs SART)...")
    regimes = [
        {"name": "1. High-Quality", "views": 180, "i0": 1e5},
        {"name": "2. Low-Photon", "views": 180, "i0": 5e3},
        {"name": "3. Sparse-View", "views": 45, "i0": 1e5},
        {"name": "4. Low-Photon + Sparse", "views": 45, "i0": 5e3},
    ]

    records = []
    recon_images = {}

    for regime in regimes:
        name = regime["name"]
        views = regime["views"]
        i0 = regime["i0"]

        noisy_sino, _, theta, _ = simulate_acquisition(
            phantom, num_projections=views, photon_budget=i0, seed=42
        )

        # FBP ramp
        fbp_recon, fbp_meta = reconstruct(noisy_sino, theta, method="FBP", filter_name="ramp")
        fbp_q = evaluate_image_quality(phantom, fbp_recon)
        fbp_cnr = compute_cnr(fbp_recon, lesion_mask, bg_mask)

        # SART 2 iterations
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
    csv_path = os.path.join(RESULTS_PHASE2_DIR, "reconstruction_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Figure: reconstruction_comparison.png (4 rows x 2 cols)
    fig, axes = plt.subplots(4, 2, figsize=(10, 16), dpi=150)
    for r_idx, regime in enumerate(regimes):
        name = regime["name"]
        fbp_img = recon_images[name]["FBP"]
        sart_img = recon_images[name]["SART"]

        df_fbp = df[(df["Regime"] == name) & (df["Algorithm"].str.startswith("FBP"))].iloc[0]
        df_sart = df[(df["Regime"] == name) & (df["Algorithm"].str.startswith("SART"))].iloc[0]

        axes[r_idx, 0].imshow(fbp_img, cmap="gray", vmin=0.0, vmax=1.0)
        axes[r_idx, 0].set_title(f"{name} — FBP\nSSIM: {df_fbp['SSIM']:.3f} | CNR: {df_fbp['CNR']:.2f}", fontsize=10)
        axes[r_idx, 0].axis("off")

        axes[r_idx, 1].imshow(sart_img, cmap="gray", vmin=0.0, vmax=1.0)
        axes[r_idx, 1].set_title(f"{name} — SART (2 iter)\nSSIM: {df_sart['SSIM']:.3f} | CNR: {df_sart['CNR']:.2f}", fontsize=10)
        axes[r_idx, 1].axis("off")

    fig.suptitle("ScanTwin Phase 2: FBP vs SART Reconstruction Comparison", fontsize=13, fontweight="bold", y=0.995)
    plt.tight_layout()
    fig_path = os.path.join(RESULTS_PHASE2_DIR, "reconstruction_comparison.png")
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_path}")

    return df


def run_experiment_d(phantom, lesion_mask, bg_mask):
    """
    Experiment D: Combined Acquisition Grid Study
    Evaluates a 3x3 matrix:
    I0 in [1e3, 1e4, 1e5] x Views in [60, 120, 180].
    Generates ssim_heatmap.png and cnr_heatmap.png.
    """
    print("-" * 70)
    print("Running Phase 2 — Experiment D: Combined Acquisition Study (3x3 Grid)...")
    photon_grid = [1e3, 1e4, 1e5]
    views_grid = [60, 120, 180]

    records = []
    ssim_matrix = np.zeros((len(photon_grid), len(views_grid)))
    cnr_matrix = np.zeros((len(photon_grid), len(views_grid)))

    for i_idx, i0 in enumerate(photon_grid):
        for v_idx, views in enumerate(views_grid):
            noisy_sino, _, theta, meta = simulate_acquisition(
                phantom, num_projections=views, photon_budget=i0, seed=42
            )
            recon, _ = reconstruct(noisy_sino, theta, method="FBP", filter_name="shepp-logan")
            q = evaluate_image_quality(phantom, recon)
            c = compute_cnr(recon, lesion_mask, bg_mask)

            ssim_matrix[i_idx, v_idx] = q["ssim"]
            cnr_matrix[i_idx, v_idx] = c

            records.append({
                "photon_budget": i0,
                "num_projections": views,
                "relative_exposure_pct": (i0 / 1e5) * 100.0,
                "projection_fraction_pct": (views / 360.0) * 100.0,
                "ssim": q["ssim"],
                "psnr": q["psnr"],
                "rmse": q["rmse"],
                "cnr": c,
            })

    df = pd.DataFrame(records)
    csv_path = os.path.join(RESULTS_PHASE2_DIR, "combined_study.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Heatmap 1: ssim_heatmap.png
    plt.figure(figsize=(7, 6), dpi=150)
    plt.imshow(ssim_matrix, cmap="viridis", aspect="auto")
    plt.colorbar(label="SSIM")
    plt.xticks(range(len(views_grid)), views_grid)
    plt.yticks(range(len(photon_grid)), [f"{p:.0e}" for p in photon_grid])
    plt.xlabel("Projection Views (N)")
    plt.ylabel("Incident Photon Budget I₀")
    plt.title("SSIM Grid: Incident Photons vs Projection Views", fontweight="bold")
    for i in range(len(photon_grid)):
        for j in range(len(views_grid)):
            plt.text(j, i, f"{ssim_matrix[i, j]:.3f}", ha="center", va="center", color="white" if ssim_matrix[i, j] < 0.75 else "black", fontweight="bold")
    plt.tight_layout()
    h_ssim_path = os.path.join(RESULTS_PHASE2_DIR, "ssim_heatmap.png")
    plt.savefig(h_ssim_path)
    plt.close()
    print(f"Saved: {h_ssim_path}")

    # Heatmap 2: cnr_heatmap.png
    plt.figure(figsize=(7, 6), dpi=150)
    plt.imshow(cnr_matrix, cmap="plasma", aspect="auto")
    plt.colorbar(label="Lesion CNR")
    plt.xticks(range(len(views_grid)), views_grid)
    plt.yticks(range(len(photon_grid)), [f"{p:.0e}" for p in photon_grid])
    plt.xlabel("Projection Views (N)")
    plt.ylabel("Incident Photon Budget I₀")
    plt.title("Lesion CNR Grid: Incident Photons vs Projection Views", fontweight="bold")
    for i in range(len(photon_grid)):
        for j in range(len(views_grid)):
            plt.text(j, i, f"{cnr_matrix[i, j]:.2f}", ha="center", va="center", color="white" if cnr_matrix[i, j] < 5.0 else "black", fontweight="bold")
    plt.tight_layout()
    h_cnr_path = os.path.join(RESULTS_PHASE2_DIR, "cnr_heatmap.png")
    plt.savefig(h_cnr_path)
    plt.close()
    print(f"Saved: {h_cnr_path}")

    return df


def run_all_phase2_experiments():
    print("=" * 70)
    print("ScanTwin: Phase 2 Automated Experimental Validation Suite")
    print("=" * 70)
    print(RESEARCH_DISCLAIMER)

    size = 256
    print(f"Generating 2D Shepp-Logan Phantom ({size}x{size}) with synthetic target...")
    base = generate_shepp_logan(size=size)
    phantom, lesion_mask, bg_mask, _ = insert_lesion(base, radius=8.0, contrast=0.08)

    df_a = run_experiment_a(phantom, lesion_mask, bg_mask, n_realizations=5)
    df_b = run_experiment_b(phantom, lesion_mask, bg_mask, n_realizations=4)
    df_c = run_experiment_c(phantom, lesion_mask, bg_mask)
    df_d = run_experiment_d(phantom, lesion_mask, bg_mask)

    print("=" * 70)
    print(f"All Phase 2 experimental studies completed successfully!")
    print(f"Artifacts saved in: {RESULTS_PHASE2_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    run_all_phase2_experiments()
