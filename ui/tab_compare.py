"""Tab: Protocol Comparison (A vs B) for the selected modality."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from digital_twin.pipeline import run_single
from ui.common import draw_image, fmt, render_protocol_controls, scope_box, show_fig
from ui.compat import stretch


def _alt_defaults(m):
    """A = reference protocol, B = a clearly cheaper one (first cheaper grid entry near 10% cost)."""
    ref = m.reference_protocol()
    grid = m.training_grid()
    cheap = min(grid, key=lambda p: abs(m.acquisition_cost(p).cost - 0.1))
    return ref, cheap


def render(ctx) -> None:
    m = ctx.modality
    st.subheader(f"Controlled comparison: Protocol A vs Protocol B ({m.name})")
    scope_box(m)
    st.markdown("Head-to-head comparison of two simulated protocols on the same object and the same noise seed.")
    da, db = _alt_defaults(m)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### 🔵 Protocol A")
        pa = render_protocol_controls(m, "cmpA", da)
    with col_b:
        st.markdown("#### 🟠 Protocol B")
        pb = render_protocol_controls(m, "cmpB", db)

    rows, runs = [], {}
    for tag, p in (("A", pa), ("B", pb)):
        r = run_single(m, ctx.truth, p, 42, False, ctx.lesion_mask, ctx.bg_mask, ctx.constraints)
        runs[tag] = r
        rows.append({"Protocol": tag, "Parameters": m.describe_protocol(r.protocol), "SSIM": fmt(r.quality["ssim"], ".4f"),
                     "CNR": fmt(r.cnr, ".2f"), "PSNR (dB)": fmt(r.quality["psnr"], ".2f"), "RMSE": fmt(r.quality["rmse"], ".4f"),
                     f"{m.cost_labels['exposure_short']} (%)": f"{r.cost.exposure_pct:.1f}", "Cost": f"{r.cost.cost:.3f}",
                     "Criteria met?": "✓ YES" if r.feasible else "✗ NO"})

    for col, tag in zip(st.columns(2), ("A", "B")):
        r = runs[tag]
        with col:
            st.markdown(f"**Protocol {tag}: {m.describe_protocol(r.protocol)}**")
            fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), dpi=110)
            gray = r.view.cmap == "gray"
            draw_image(axes[0], r.view.image, r.view.title, cmap=r.view.cmap, vmin=0.0 if gray else None, vmax=None, axis_off=False)
            draw_image(axes[1], r.recon, f"{tag}: SSIM {r.quality['ssim']:.3f}")
            draw_image(axes[2], r.err_stats["error_map"], f"Error (RMSE {r.quality['rmse']:.4f})", cmap="inferno", vmin=0.0, vmax=0.35)
            plt.tight_layout()
            show_fig(fig)
    st.dataframe(pd.DataFrame(rows), hide_index=True, **stretch(st.dataframe))
