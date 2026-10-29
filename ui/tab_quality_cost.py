"""Tab: Quality vs Cost Lab — sweeps of the modality's own cost axis (mean ± SD over noise realizations)."""

from __future__ import annotations

import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from experiments.modality_experiments import ROOT, sweep
from ui.common import scope_box
from ui.compat import stretch


def _line(df, x, ymean, ystd, title, xlabel, ylabel, x_log, hline=None, hline_label=""):
    sub = df.dropna(subset=[ymean])
    fig = go.Figure(go.Scatter(x=sub[x], y=sub[ymean], mode="lines+markers",
                               error_y=dict(type="data", array=sub[ystd].fillna(0.0), visible=True), name=ylabel))
    if hline is not None:
        fig.add_hline(y=hline, line_dash="dot", line_color="red", annotation_text=hline_label)
    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel, height=340, margin=dict(l=10, r=10, t=50, b=10))
    if x_log:
        fig.update_xaxes(type="log")
    return fig


def render(ctx) -> None:
    m = ctx.modality
    ax = m.sweep_axes()
    st.subheader(f"Quality vs Cost Laboratory ({m.name})")
    scope_box(m)
    st.markdown(f"Sweeps one protocol parameter while all other settings come from the sidebar. The x axis shows the modality's own "
                f"cost proxy: **{m.cost_labels['exposure']}**. Each point is the mean ± 1 SD over independent noise realizations.")
    which = st.radio("Sweep", [f"Primary: {ax['primary']['label']}", f"Secondary: {ax['secondary']['label']}"], horizontal=True,
                     key="qc_which")
    axis = ax["primary"] if which.startswith("Primary") else ax["secondary"]
    mode = st.radio("Noise realizations", ["Quick mode (3)", "Research mode (10)"], horizontal=True, key="qc_n")
    n_real = 3 if "Quick" in mode else 10
    store = f"qc_{m.name}"
    if st.button("🔬 RUN LIVE SWEEP", type="primary", key="qc_go"):
        bar = st.progress(0.0)
        df = sweep(ctx.twin, ctx.protocol, axis["param"], axis["values"], n_real, ctx.seed,
                   progress=lambda d, t: bar.progress(d / t))
        st.session_state[store] = (df, axis, n_real)
    if store in st.session_state:
        df, a, n = st.session_state[store]
        xcol = "relative_exposure_pct" if a["param"] in ("photon_budget", "n_frames", "acceleration") else a["param"]
        xlabel = (m.cost_labels["exposure"] + " (%)") if xcol == "relative_exposure_pct" else a["label"]
        x_log = a["x_log"] and xcol == "relative_exposure_pct" and m.name != "MRI"
        st.markdown(f"#### Results — sweep of {a['label']} (N = {n} realizations)")
        c1, c2, c3 = st.columns(3)
        c1.plotly_chart(_line(df, xcol, "ssim_mean", "ssim_std", "SSIM", xlabel, "SSIM (mean ± SD)", x_log, ctx.min_ssim, "SSIM criterion"),
                        **stretch(st.plotly_chart))
        if df["cnr_mean"].notna().any():
            c2.plotly_chart(_line(df, xcol, "cnr_mean", "cnr_std", "CNR", xlabel, "Target CNR (mean ± SD)", x_log, ctx.min_cnr, "CNR criterion"),
                            **stretch(st.plotly_chart))
        else:
            c2.info("CNR: n/a — insert a synthetic target in the sidebar.")
        c3.plotly_chart(_line(df, xcol, "rmse_mean", "rmse_std", "RMSE", xlabel, "RMSE (mean ± SD)", x_log), **stretch(st.plotly_chart))
        st.dataframe(df, **stretch(st.dataframe))
        st.download_button("Download sweep (CSV)", df.to_csv(index=False).encode("utf-8"), file_name=f"scantwin_sweep_{m.name}.csv",
                           mime="text/csv")
    elif m.name == "CT":
        p_dose = os.path.join(ROOT, "results", "phase2", "photon_budget_study.csv")
        if os.path.exists(p_dose):
            st.markdown("#### Pre-computed CT validation sweep (Phase 2, 180 views)")
            d = pd.read_csv(p_dose)
            st.plotly_chart(_line(d, "photon_budget", "ssim_mean", "ssim_std", "Experiment A: SSIM vs photon budget I₀",
                                  "Photon budget I₀ (log)", "SSIM (mean ± SD)", True, ctx.min_ssim, "SSIM criterion"), **stretch(st.plotly_chart))
