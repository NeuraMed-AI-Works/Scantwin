"""Tab: Digital Twin — one protocol end to end."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from ui.common import draw_image, fmt, metric_grid, pipeline_breadcrumb, scope_box, show_fig, tooltip_text


def render(ctx) -> None:
    m, run = ctx.modality, ctx.run
    labels, tips = m.cost_labels, tooltip_text()
    st.subheader(f"{m.name} Image Formation and Reconstruction Pipeline")
    pipeline_breadcrumb(m)
    scope_box(m)

    cols = st.columns(4)
    with cols[0]:
        st.markdown("**1. Ground truth**")
        fig, ax = plt.subplots(figsize=(3.8, 3.8), dpi=110)
        draw_image(ax, ctx.truth, lesion_mask=ctx.lesion_mask, bg_mask=ctx.bg_mask)
        show_fig(fig)
        if ctx.target_on:
            st.caption("🔴 target ROI | 🔵 background ROI")
    with cols[1]:
        mode = "ideal, noise-free" if ctx.ideal_mode else "noisy"
        st.markdown(f"**2. {run.view.title} ({mode})**")
        fig, ax = plt.subplots(figsize=(3.8, 3.8), dpi=110)
        v = run.view.image
        gray = run.view.cmap == "gray"
        draw_image(ax, v, cmap=run.view.cmap, vmin=0.0 if gray else None,
                   vmax=float(max(1.0, np.nanmax(v))) if gray else None, axis_off=False)
        ax.set_xlabel(run.view.xlabel, fontsize=8)
        ax.set_ylabel(run.view.ylabel, fontsize=8)
        show_fig(fig)
        st.caption(run.view.note)
    with cols[2]:
        st.markdown(f"**3. {labels['recon']}**")
        fig, ax = plt.subplots(figsize=(3.8, 3.8), dpi=110)
        draw_image(ax, run.recon, lesion_mask=ctx.lesion_mask)
        show_fig(fig)
        st.caption(f"{m.describe_protocol(run.protocol)} — reconstructed in {run.recon_meta.get('elapsed_time_ms', 0.0):.1f} ms")
    with cols[3]:
        st.markdown("**4. Absolute error map**")
        fig, ax = plt.subplots(figsize=(3.8, 3.8), dpi=110)
        im = draw_image(ax, run.err_stats["error_map"], cmap="inferno", vmin=0.0, vmax=0.35)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        show_fig(fig)
        st.caption(f"Max error {run.err_stats['max_abs_error']:.3f} | mean {run.err_stats['mean_abs_error']:.4f}")

    st.markdown("### Quantitative fidelity & feasibility")
    q = run.quality
    metric_grid([
        ("SSIM", fmt(q["ssim"], ".4f"), f"{q['ssim'] - ctx.min_ssim:+.3f} vs criterion", tips["ssim"]),
        ("CNR", fmt(run.cnr, ".2f"), (f"{run.cnr - ctx.min_cnr:+.2f} vs criterion" if run.cnr is not None else None), tips["cnr"]),
        ("PSNR", fmt(q["psnr"], ".2f", " dB"), None, tips["psnr"]),
        ("RMSE", fmt(q["rmse"], ".4f"), None, tips["rmse"]),
        (labels["exposure"], f"{run.cost.exposure_pct:.1f}%", None, "Relative to the modality's reference protocol (= 100 %)."),
        ("Acquisition cost", f"{run.cost.cost:.3f}", f"{run.cost.cost - 1.0:+.3f} vs reference", tips["cost"]),
    ], cols=3)

    left, right = st.columns(2)
    with left:
        rs = run.roi_stats
        body = ("n/a — no synthetic target inserted." if rs is None else
                f"• Target mean: <code>{rs['mean_lesion']:.4f}</code><br>• Background mean: <code>{rs['mean_background']:.4f}</code><br>"
                f"• Background noise (σ): <code>{rs['std_background']:.4f}</code><br>• Local contrast |Δ|: <code>{rs['delta_contrast']:.4f}</code>")
        st.markdown(f'<div class="stat-card"><strong>Regional statistics:</strong><br>{body}</div>', unsafe_allow_html=True)
    with right:
        crit = [f"SSIM {fmt(q['ssim'], '.3f')} / {ctx.min_ssim:.2f}"]
        if ctx.min_psnr is not None:
            crit.append(f"PSNR {fmt(q['psnr'], '.1f')} / {ctx.min_psnr:.1f} dB")
        if run.cnr is not None:
            crit.append(f"CNR {fmt(run.cnr, '.2f')} / {ctx.min_cnr:.1f}")
        css = "status-feasible" if run.feasible else "status-infeasible"
        txt = ("✅ ALL EXPERIMENTAL CRITERIA SATISFIED" if run.feasible else "⚠️ CRITERIA NOT SATISFIED") + " (" + "; ".join(crit) + ")"
        st.markdown(f'<div class="stat-card"><strong>Experimental criteria:</strong><br><div style="margin-top:8px;" '
                    f'class="{css}">{txt}</div></div>', unsafe_allow_html=True)
        r = ctx.ref_state
        st.caption(f"Reference protocol ({m.describe_protocol(r.protocol)}): SSIM {fmt(r.ssim, '.3f')}, "
                   f"{'satisfies' if r.is_feasible else 'does not satisfy'} the criteria.")
