"""Tab: Reconstruction Analysis — reconstruction-side alternatives on one shared measurement."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from digital_twin.pipeline import compare_recon_variants
from ui.common import draw_image, fmt, scope_box, show_fig

TITLES = {
    "CT": "Analytical (FBP filters) vs algebraic iterative (SART) reconstruction",
    "Other": "Analytical (FBP filters) vs algebraic iterative (SART) reconstruction (generic proxy)",
    "MRI": "Plain zero-filled reconstruction vs Hann k-space window",
    "X-ray": "Post-filter options on the inverse-log radiograph",
    "Ultrasound": "Post-filter options on the compounded B-mode image",
}


def render(ctx) -> None:
    m = ctx.modality
    st.subheader(f"Reconstruction Analysis — {TITLES[m.name]}")
    scope_box(m)
    st.markdown("All variants use the SAME simulated measurement (same protocol, same seed); only the reconstruction-side setting changes.")
    with st.spinner("Computing comparative reconstructions..."):
        res = compare_recon_variants(m, ctx.truth, ctx.protocol, ctx.seed, ctx.ideal_mode, ctx.lesion_mask, ctx.bg_mask)
    cols = st.columns(len(res))
    for col, r in zip(cols, res):
        with col:
            st.markdown(f"#### {r['label']}")
            fig, axes = plt.subplots(1, 2, figsize=(6, 3.2), dpi=110)
            draw_image(axes[0], r["recon"], "Reconstruction")
            draw_image(axes[1], r["err"], "Absolute error", cmap="inferno", vmin=0.0, vmax=0.35)
            plt.tight_layout()
            show_fig(fig)
            st.write(f"**SSIM** {fmt(r['quality']['ssim'], '.4f')} | **CNR** {fmt(r['cnr'], '.2f')} | "
                     f"**PSNR** {fmt(r['quality']['psnr'], '.2f', ' dB')} | **Runtime** {r['ms']:.1f} ms")
