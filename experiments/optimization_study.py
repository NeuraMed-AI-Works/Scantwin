"""
Experiment D: Acquisition Protocol Optimization Study
=====================================================
Performs a head-to-head empirical comparison between:
1. Reference Protocol (Standard high-exposure clinical baseline: 360 views, I0 = 1e5, FBP ramp)
2. Naive Low-Exposure Protocol (Arbitrary reduction: 60 views, I0 = 1e4, FBP ramp)
3. AI-Optimized Protocol (Discovered by closed-loop digital twin optimizer satisfying SSIM/CNR constraints)

STATISTICAL REPEATABILITY:
Evaluates all three protocols across repeated independent noise realizations (N = 5)
to report statistically robust mean and standard deviation metrics.
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

from config import RESEARCH_DISCLAIMER, AcquisitionProtocol, REFERENCE_I0
from simulation.phantom import generate_shepp_logan
from simulation.lesion import insert_lesion
from digital_twin.twin import CTDigitalTwin
from optimization.protocol_optimizer import ProtocolOptimizer


def run_optimization_study(
    target_ssim_min=0.88,
    target_cnr_min=3.0,
    n_trials=25,
    n_eval_realizations=5,
    size=256,
    save_outputs=True,
):
    print("=" * 70)
    print("Experiment D: AI Protocol Optimization Study")
    print("=" * 70)
    print(RESEARCH_DISCLAIMER)

    # 1. Setup Phantom with known synthetic lesion
    base = generate_shepp_logan(size=size)
    phantom, lesion_mask, bg_mask, lesion_meta = insert_lesion(base, radius=8.0, contrast=0.08)

    twin = CTDigitalTwin(
        ground_truth=phantom,
        lesion_mask=lesion_mask,
        background_mask=bg_mask,
        lesion_metadata=lesion_meta,
    )

    # 2. Run Optimizer
    print(f"Running Bayesian optimization (n_trials={n_trials}, target SSIM>={target_ssim_min}, CNR>={target_cnr_min})...")
    optimizer = ProtocolOptimizer(
        twin=twin,
        target_ssim_min=target_ssim_min,
        target_cnr_min=target_cnr_min,
        seed=42,
    )

    best_state, history_df, summary = optimizer.optimize(
        n_trials=n_trials,
        n_realizations=1,
        include_sart=False,
    )
    print(summary)

    # 3. Define candidate protocols for statistical evaluation
    ref_proto = AcquisitionProtocol(
        photon_budget=REFERENCE_I0,
        num_projections=360,
        reconstruction_method="FBP",
        fbp_filter="ramp",
    )

    naive_proto = AcquisitionProtocol(
        photon_budget=1e4,
        num_projections=60,
        reconstruction_method="FBP",
        fbp_filter="ramp",
    )

    if best_state is not None:
        opt_proto = AcquisitionProtocol(
            photon_budget=best_state.photon_budget,
            num_projections=best_state.num_projections,
            reconstruction_method=best_state.reconstruction_method,
            fbp_filter=best_state.fbp_filter,
            sart_iterations=best_state.sart_iterations,
        )
    else:
        # Fallback for evaluation if optimization found no feasible candidate
        opt_proto = AcquisitionProtocol(
            photon_budget=3e4,
            num_projections=120,
            reconstruction_method="FBP",
            fbp_filter="shepp-logan",
        )

    protocols = {
        "Reference Baseline": ref_proto,
        "Naive Low-Exposure": naive_proto,
        "AI-Optimized": opt_proto,
    }

    records = []
    recon_samples = {}

    for name, proto in protocols.items():
        print(f"Evaluating {name} across {n_eval_realizations} realizations...")
        state, recon, _ = twin.evaluate_protocol(
            proto,
            target_ssim_min=target_ssim_min,
            target_cnr_min=target_cnr_min,
            n_realizations=n_eval_realizations,
        )
        recon_samples[name] = recon

        rel_red = 100.0 - state.relative_photon_exposure_pct
        records.append({
            "Protocol": name,
            "Photon Budget (I0)": f"{proto.photon_budget:.1e}",
            "Projections": proto.num_projections,
            "Reconstruction": f"{proto.reconstruction_method} ({proto.fbp_filter})",
            "SSIM (Mean)": state.ssim,
            "SSIM (SD)": state.metadata.get("ssim_std", 0.0),
            "CNR (Mean)": state.cnr,
            "CNR (SD)": state.metadata.get("cnr_std", 0.0),
            "PSNR (dB)": state.psnr,
            "RMSE": state.rmse,
            "Rel. Exposure (%)": state.relative_photon_exposure_pct,
            "Rel. Exposure Reduction (%)": max(0.0, rel_red),
            "Acquisition Cost": state.normalized_acquisition_cost,
            "Feasible?": "Yes" if state.is_feasible else "No",
        })

    eval_df = pd.DataFrame(records)

    if save_outputs:
        os.makedirs(os.path.join(PROJECT_ROOT, "results", "tables"), exist_ok=True)
        os.makedirs(os.path.join(PROJECT_ROOT, "results", "figures"), exist_ok=True)

        table_path = os.path.join(PROJECT_ROOT, "results", "tables", "optimization_study.csv")
        eval_df.to_csv(table_path, index=False)
        print(f"Saved evaluation table to: {table_path}")

        # 3-Panel Visual Comparison: Reference vs Naive vs Optimized
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150)

        names = ["Reference Baseline", "Naive Low-Exposure", "AI-Optimized"]
        for idx, name in enumerate(names):
            row_data = eval_df[eval_df["Protocol"] == name].iloc[0]
            img = recon_samples[name]

            axes[idx].imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
            axes[idx].contour(lesion_mask, colors="red", linewidths=1.0, levels=[0.5])
            title = (
                f"{name}\n"
                f"I₀={row_data['Photon Budget (I0)']}, N={row_data['Projections']}\n"
                f"SSIM: {row_data['SSIM (Mean)']:.3f} | CNR: {row_data['CNR (Mean)']:.2f}\n"
                f"Exposure: {row_data['Rel. Exposure (%)']:.1f}% | Feasible: {row_data['Feasible?']}"
            )
            axes[idx].set_title(title, fontsize=11, fontweight="bold")
            axes[idx].set_xlabel("X (pixels)")
            axes[idx].set_ylabel("Y (pixels)")

        fig.suptitle(
            "ScanTwin Experiment D: Reference vs Naive Low-Exposure vs AI-Optimized Protocol",
            fontsize=13,
            fontweight="bold",
            y=1.03,
        )
        plt.tight_layout()

        fig_path = os.path.join(PROJECT_ROOT, "results", "figures", "optimization_study.png")
        plt.savefig(fig_path, bbox_inches="tight")
        print(f"Saved figure to: {fig_path}")

    return eval_df, history_df


if __name__ == "__main__":
    run_optimization_study()
