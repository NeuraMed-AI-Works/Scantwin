"""Tab: Live Acquisition — real-time streaming digital twin for every modality."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from simulation.live import first_frame_meeting_criteria, stream_modality, stream_series
from ui.common import draw_image, fmt, metric_grid, scope_box, show_fig
from ui.compat import stretch

STREAM_TEXT = {
    "CT": "projection views arrive progressively (golden-angle order)",
    "Other": "projection views arrive progressively (golden-angle order, generic proxy)",
    "MRI": "phase-encode lines arrive progressively, centre of k-space first, then outward",
    "Ultrasound": "compounded frames accumulate one by one — speckle visibly falls",
    "X-ray": "the photon budget is ramped up — exposure accumulates and noise visibly falls",
}


def _render_frame(frame, rows, ph, ctx):
    m = frame.metrics
    has_t = not np.isnan(m["cnr"])
    ph["progress"].progress(min(1.0, frame.acquired / frame.total),
                            text=f"{frame.label} | frame computed in {frame.elapsed_ms:.0f} ms")
    with ph["images"].container():
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), dpi=105)
        gray = ctx.run.view.cmap == "gray"
        view = frame.partial_measurement
        draw_image(axes[0], view, f"{ctx.run.view.title} (received so far)", cmap=ctx.run.view.cmap,
                   vmin=0.0 if gray else None, vmax=float(max(1.0, np.nanmax(view))) if gray and np.isfinite(view).any() else None,
                   axis_off=False)
        axes[0].set_xlabel(ctx.run.view.xlabel, fontsize=7)
        axes[0].set_ylabel(ctx.run.view.ylabel, fontsize=7)
        draw_image(axes[1], frame.recon, f"Live reconstruction — {frame.label}", lesion_mask=frame.lesion_mask)
        err = np.abs(frame.truth - frame.recon)
        draw_image(axes[2], err, f"|Truth − Recon| (max {err.max():.3f})", cmap="inferno", vmin=0.0, vmax=0.35)
        plt.tight_layout()
        show_fig(fig)
    with ph["metrics"].container():
        metric_grid([
            (frame.unit.capitalize(), f"{frame.acquired}/{frame.total}"),
            ("SSIM", fmt(m["ssim"], ".3f"), f"{m['ssim'] - ctx.min_ssim:+.3f} vs criterion"),
            ("CNR", fmt(m["cnr"], ".2f") if has_t else "n/a", f"{m['cnr'] - ctx.min_cnr:+.2f} vs criterion" if has_t else None),
            ("PSNR", fmt(m["psnr"], ".1f", " dB")),
            ("RMSE", fmt(m["rmse"], ".4f")),
            (ctx.modality.cost_labels["exposure_short"], f"{frame.relative_photon_exposure_pct:.1f}%"),
        ], cols=3)
    with ph["charts"].container():
        hist = pd.DataFrame(rows).set_index("Frame")
        g1, g2 = st.columns(2)
        g1.caption("SSIM per frame")
        g1.line_chart(hist[["SSIM"]], height=170)
        if has_t:
            g2.caption("CNR per frame")
            g2.line_chart(hist[["CNR"]], height=170)


def render(ctx) -> None:
    m = ctx.modality
    st.subheader(f"Live Acquisition — real-time {m.name} digital twin")
    scope_box(m)
    st.markdown(
        f"Watch the simulated acquisition work. **Live scan**: {STREAM_TEXT[m.name]}; after every batch the image is "
        "re-reconstructed and re-measured. **Live slice stream**: a series is acquired slice after slice with the current "
        "protocol. Uses the sidebar protocol and criteria. Every number is computed by the simulation as the frame is produced; "
        "the final live frame equals a normal acquisition with the same seed."
    )
    mode = st.radio("Stream type", ["Live scan (data arrives progressively)", "Live slice stream (series)"], horizontal=True,
                    key="live_mode")
    c1, c2 = st.columns(2)
    n_frames = c1.slider("Frames per scan", 4, 16, 8, key="live_n", help="Intermediate reconstructions in a live scan.")
    delay = c2.slider("Delay between frames (ms)", 0, 1000, 150, 50, key="live_delay", help="Pause after each frame so changes are visible.")
    series_ok = len(ctx.slices) > 1
    is_series = mode.startswith("Live slice")
    if is_series and not series_ok:
        st.info("The selected dataset has a single slice. Choose **Synthetic slice series** or upload a stack / volume.")
    run_live = st.button("▶ Start live acquisition", type="primary", key="live_go")

    ph = {k: st.empty() for k in ("progress", "images", "metrics", "charts")}
    summary = st.empty()
    store = f"live_{m.name}"
    if run_live and not (is_series and not series_ok):
        collected, rows = [], []
        proto = ctx.protocol
        if not is_series:
            gen = stream_modality(m, ctx.truth, ctx.lesion_mask, ctx.bg_mask, proto, seed=ctx.seed, n_frames=n_frames,
                                  ideal=ctx.ideal_mode, frame_delay_s=delay / 1000.0)
        else:
            gen = stream_series(m, ctx.slices, proto, lesion_on=ctx.target_on, lesion_radius=ctx.sb.target_radius,
                                lesion_contrast=ctx.sb.target_contrast, seed=ctx.seed, ideal=ctx.ideal_mode,
                                frame_delay_s=delay / 1000.0)
        for fr in gen:
            collected.append(fr)
            c = fr.metrics["cnr"]
            rows.append({"Frame": fr.index + 1, "Label": fr.label, "Acquired": fr.acquired, "SSIM": fr.metrics["ssim"],
                         "PSNR (dB)": fr.metrics["psnr"], "RMSE": fr.metrics["rmse"], "CNR": None if np.isnan(c) else c,
                         "Rel. exposure/time proxy (%)": fr.relative_photon_exposure_pct, "Cost": fr.cost,
                         "Frame time (ms)": fr.elapsed_ms})
            _render_frame(fr, rows, ph, ctx)
        st.session_state[store] = (collected[-1], list(rows), is_series, collected)
    elif store in st.session_state:
        last, rows, is_series_shown, collected = st.session_state[store]
        _render_frame(last, rows, ph, ctx)
        is_series = is_series_shown
    else:
        rows, collected = [], []

    if rows:
        df = pd.DataFrame(rows)
        with summary.container():
            if not is_series and collected:
                hit = first_frame_meeting_criteria(collected, ctx.min_ssim, ctx.eff_min_cnr, ctx.min_psnr)
                if hit is None:
                    st.error("Experimental criteria were not satisfied at any checkpoint of this scan.")
                else:
                    st.success(f"Experimental criteria first satisfied at **{hit.acquired} of {hit.total} {hit.unit}** "
                               f"({hit.scan_progress_pct:.0f}% of this scan). This is the earliest *tested checkpoint* in this "
                               "simulation, not a clinical stopping rule.")
            st.dataframe(df, **stretch(st.dataframe))
            st.download_button("Download live log (CSV)", df.to_csv(index=False).encode("utf-8"),
                               file_name=f"scantwin_live_{m.name}.csv", mime="text/csv")
