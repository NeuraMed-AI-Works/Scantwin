"""Tab: Automated Experiments A-D (modality-aware); results saved to results/<modality>/ with timestamps."""

from __future__ import annotations

import glob
import os

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from experiments.modality_experiments import (ROOT, combined_grid, reconstruction_comparison, results_dir, save_experiment, sweep)
from ui.common import scope_box
from ui.compat import stretch


def _fig_line(df, x, title, xlabel, x_log):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), dpi=110)
    for ax, y, lab in ((axes[0], "ssim", "SSIM"), (axes[1], "rmse", "RMSE")):
        ax.errorbar(df[x], df[f"{y}_mean"], yerr=df[f"{y}_std"].fillna(0), marker="o", capsize=3)
        ax.set_xlabel(xlabel); ax.set_ylabel(f"{lab} (mean ± SD)"); ax.grid(alpha=0.3)
        if x_log:
            ax.set_xscale("log")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


def render(ctx) -> None:
    m = ctx.modality
    ax = m.sweep_axes()
    st.subheader(f"Automated Research Experiments ({m.name})")
    scope_box(m)
    st.markdown(f"Each run is a real simulation on the current object; results are saved as CSV + PNG under "
                f"`results/{m.name.lower().replace(' ', '-')}/` with a timestamp, and **earlier runs are never overwritten**.")
    exp = st.radio("Experiment", [
        f"A — Primary sweep: {ax['primary']['label']}",
        f"B — Secondary sweep: {ax['secondary']['label']}",
        "C — Reconstruction-side comparison (shared measurement)",
        f"D — Combined grid: {ax['primary']['label']} × {ax['secondary']['label']}",
    ], key="exp_choice")
    n_real = st.slider("Noise realizations per point", 1, 10, 3, key="exp_n")
    if st.button("▶ Run and save experiment", type="primary", key="exp_go"):
        bar = st.progress(0.0)
        cb = lambda d, t: bar.progress(d / t)
        if exp.startswith("A") or exp.startswith("B"):
            a = ax["primary"] if exp.startswith("A") else ax["secondary"]
            df = sweep(ctx.twin, ctx.protocol, a["param"], a["values"], n_real, ctx.seed, cb)
            xcol = a["param"] if exp.startswith("B") else "relative_exposure_pct"
            fig = _fig_line(df, xcol, f"{m.name}: {a['label']}", a["label"] if exp.startswith("B") else m.cost_labels["exposure"] + " (%)",
                            a["x_log"] and exp.startswith("A") and m.name != "MRI")
            name = "experiment_A_primary" if exp.startswith("A") else "experiment_B_secondary"
        elif exp.startswith("C"):
            df = reconstruction_comparison(ctx.twin, ctx.protocol, ctx.seed, ctx.ideal_mode)
            fig, a2 = plt.subplots(figsize=(6, 3), dpi=110)
            a2.bar(df["variant"], df["ssim"]); a2.set_ylabel("SSIM"); a2.set_title("Reconstruction comparison")
            plt.setp(a2.get_xticklabels(), rotation=20, ha="right"); fig.tight_layout()
            name = "experiment_C_reconstruction"
        else:
            df = combined_grid(ctx.twin, ctx.protocol, ax["primary"]["param"], ax["primary"]["values"],
                               ax["secondary"]["param"], ax["secondary"]["values"], max(1, min(n_real, 3)), ctx.seed, cb)
            piv = df.pivot_table(index=ax["secondary"]["param"], columns=ax["primary"]["param"], values="ssim_mean")
            fig, a3 = plt.subplots(figsize=(6.5, 3.6), dpi=110)
            im = a3.imshow(piv.values, cmap="viridis", aspect="auto", origin="lower")
            a3.set_xticks(range(len(piv.columns))); a3.set_xticklabels([f"{c:g}" for c in piv.columns], rotation=30)
            a3.set_yticks(range(len(piv.index))); a3.set_yticklabels([f"{c:g}" for c in piv.index])
            a3.set_xlabel(ax["primary"]["label"]); a3.set_ylabel(ax["secondary"]["label"]); a3.set_title("SSIM (mean)")
            plt.colorbar(im, ax=a3); fig.tight_layout()
            name = "experiment_D_combined"
        paths = save_experiment(m.name, name, df, fig)
        st.session_state[f"exp_{m.name}"] = (df, fig, paths)
    if f"exp_{m.name}" in st.session_state:
        df, fig, paths = st.session_state[f"exp_{m.name}"]
        st.dataframe(df, **stretch(st.dataframe))
        st.pyplot(fig, **stretch(st.pyplot))
        st.success("Saved: " + ", ".join(os.path.relpath(p, ROOT) for p in paths.values()))
    prev = sorted(glob.glob(os.path.join(results_dir(m.name), "*.csv")))
    with st.expander(f"Earlier saved runs for {m.name} ({len(prev)})"):
        for p in prev[::-1]:
            st.write(os.path.relpath(p, ROOT))
    if m.name == "CT":
        p2 = os.path.join(ROOT, "results", "phase2")
        if os.path.isdir(p2):
            with st.expander("Phase 2 CT validation artifacts (pre-computed with the original CT-only scripts)"):
                for f in sorted(glob.glob(os.path.join(p2, "*.png"))):
                    st.image(f, caption=os.path.basename(f))
