"""Shared UI helpers: CSS, header, metric cards (3 x 2 grid, never truncated), figures, protocol controls, info boxes."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from config import APP_SUBTITLE, DATA_USE_WARNING, RESEARCH_DISCLAIMER
from modalities import Modality
from ui.compat import stretch

CSS = """
<style>
.main-header {font-size: 2.1rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0;}
.sub-header {font-size: 1.02rem; color: #4B5563; margin-bottom: 12px;}
.disclaimer-banner {background:#FEF3C7; border-left:5px solid #F59E0B; padding:9px 14px; color:#92400E;
    font-size:.85rem; border-radius:4px; margin-bottom:14px;}
.scope-box {background:#F0FDF4; border-left:5px solid #16A34A; padding:9px 14px; color:#14532D;
    font-size:.86rem; border-radius:4px; margin:6px 0 12px 0;}
.pipeline-breadcrumb {background:#EFF6FF; border:1px solid #BFDBFE; border-radius:6px; padding:8px 12px;
    font-size:.84rem; color:#1E40AF; margin-bottom:15px; font-family:monospace;}
.status-feasible {background:#D1FAE5; color:#065F46; font-weight:bold; padding:7px 14px; border-radius:6px; display:inline-block;}
.status-infeasible {background:#FEE2E2; color:#991B1B; font-weight:bold; padding:7px 14px; border-radius:6px; display:inline-block;}
.stat-card {background:#F8FAFC; border:1px solid #E2E8F0; border-radius:6px; padding:10px 14px; margin-top:10px;}
/* metric cards: never cut a value off */
[data-testid="stMetricValue"], [data-testid="stMetricValue"] > div {font-size:1.45rem !important; overflow:visible !important;
    text-overflow:clip !important; white-space:normal !important; word-break:break-word;}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] > div {overflow:visible !important; text-overflow:clip !important;
    white-space:normal !important;}
[data-testid="stMetricDelta"], [data-testid="stMetricDelta"] > div {overflow:visible !important; text-overflow:clip !important;
    white-space:normal !important;}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def header() -> None:
    st.markdown('<div class="main-header">🔬 ScanTwin</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-header">{APP_SUBTITLE}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="disclaimer-banner"><strong>Research Disclaimer:</strong> {RESEARCH_DISCLAIMER}</div>',
                unsafe_allow_html=True)


def scope_box(modality: Modality) -> None:
    st.markdown(f'<div class="scope-box"><strong>Model scope — {modality.name}:</strong> {modality.model_scope_text}</div>',
                unsafe_allow_html=True)


def pipeline_breadcrumb(modality: Modality) -> None:
    st.markdown(f'<div class="pipeline-breadcrumb">{modality.cost_labels["pipeline"]}</div>', unsafe_allow_html=True)


def fmt(value: Optional[float], spec: str = ".3f", suffix: str = "") -> str:
    """'n/a' for None or non-finite numbers: nothing invented, no NaN ever reaches the screen."""
    try:
        v = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(v):
        return "n/a"
    return format(v, spec) + suffix


def metric_grid(items: Sequence[Tuple], cols: int = 3) -> None:
    """items: (label, value_str, delta_or_None, help_or_None); laid out ``cols`` per row."""
    for start in range(0, len(items), cols):
        cells = st.columns(cols)
        for cell, it in zip(cells, items[start:start + cols]):
            label, value = it[0], it[1]
            delta = it[2] if len(it) > 2 else None
            help_ = it[3] if len(it) > 3 else None
            cell.metric(label, value, delta=delta, help=help_)


def show_fig(fig) -> None:
    st.pyplot(fig, **stretch(st.pyplot))
    plt.close(fig)


def draw_image(ax, img: np.ndarray, title: str = "", cmap: str = "gray", vmin: Optional[float] = 0.0,
               vmax: Optional[float] = 1.0, lesion_mask=None, bg_mask=None, axis_off: bool = True) -> Any:
    masked = np.ma.masked_invalid(img)
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad("#111111")
    im = ax.imshow(masked, cmap=cm, vmin=vmin, vmax=vmax, aspect="auto" if cmap != "gray" else "equal")
    if lesion_mask is not None:
        ax.contour(lesion_mask, colors="red", linewidths=0.9, levels=[0.5])
    if bg_mask is not None:
        ax.contour(bg_mask, colors="cyan", linewidths=0.7, levels=[0.5])
    ax.set_title(title, fontsize=9)
    if axis_off:
        ax.axis("off")
    return im


def tooltip_text() -> Dict[str, str]:
    return {
        "ssim": "Structural similarity to the ground truth (1 = identical). Higher is better.",
        "psnr": "Peak signal-to-noise ratio vs the ground truth, in dB. Higher is better.",
        "rmse": "Root-mean-square error vs the ground truth, in image-intensity units [0-1]. Lower is better.",
        "cnr": "Contrast-to-noise ratio of the inserted synthetic target: |mean(target) - mean(background)| / std(background). 'n/a' when no target is inserted.",
        "cost": "Normalized acquisition cost; the modality's reference protocol = 1.000. A proxy, never dose or real scan time.",
    }


# ------------------------------------------------------------------------------------------
# Protocol controls generated from Modality.protocol_space()
# ------------------------------------------------------------------------------------------
def render_protocol_controls(modality: Modality, key_prefix: str, defaults: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Draw one widget per active ParamSpec in the current Streamlit container and return the protocol dict."""
    defaults = {**modality.default_protocol(), **(defaults or {})}
    protocol: Dict[str, Any] = {}
    for sp in modality.protocol_space():
        protocol[sp.name] = defaults[sp.name]
        if not sp.active(protocol):
            continue
        key = f"{key_prefix}_{modality.name}_{sp.name}"
        label = f"{sp.label} ({sp.unit})" if sp.unit else sp.label
        if sp.kind == "ordered":
            opts = list(sp.options)
            val = defaults[sp.name] if defaults[sp.name] in opts else opts[0]
            protocol[sp.name] = st.select_slider(label, options=opts, value=val, format_func=sp.format, key=key, help=sp.help)
        elif sp.kind == "categorical":
            opts = list(sp.options)
            idx = opts.index(defaults[sp.name]) if defaults[sp.name] in opts else 0
            protocol[sp.name] = st.selectbox(label, opts, index=idx, key=key, help=sp.help)
        else:
            protocol[sp.name] = int(st.slider(label, int(sp.min), int(sp.max), int(defaults[sp.name]), int(sp.step or 1),
                                              key=key, help=sp.help))
    return protocol


def quickstart_panel() -> None:
    """Where to get a few small real images per modality (sources verified for this project)."""
    with st.expander("📚 Quick-start datasets — where to get a few real images"):
        st.warning(DATA_USE_WARNING)
        st.markdown(
            """
**MedMNIST+ on Zenodo** (CC BY 4.0; DermaMNIST is non-commercial): <https://zenodo.org/record/10519652>
Use the **`_64`** files (the plain unsuffixed files are only 28×28; `_128` / `_224` are very large). Upload the `.npz` directly:
the `train_images` array is used for training and `test_images` for testing.

| Modality | File to download and upload |
|---|---|
| CT (abdominal organ slices) | `organcmnist_64.npz`, `organsmnist_64.npz` or `organamnist_64.npz` |
| X-ray (chest) | `pneumoniamnist_64.npz` (also `chestmnist_64.npz`) |
| Ultrasound (breast, per the MedMNIST documentation) | `breastmnist_64.npz` |
| MRI | No 2D MRI set from MedMNIST has been verified — use the Decathlon volumes below or your own de-identified images |

**Medical Segmentation Decathlon** (CT and MRI volumes in NIfTI, CC BY-SA 4.0): <https://registry.opendata.aws/msd/>,
bucket `s3://msd-for-monai` (list it with `aws s3 ls --no-sign-request s3://msd-for-monai/`).
**Check which task is MRI before using it** (the task descriptions on the registry page say which modality each one is).
Upload a `.nii` / `.nii.gz` volume (needs `nibabel`); up to 8 evenly spaced axial slices with tissue are used, with one
shared intensity mapping for the whole volume. For a **CT** volume the soft-tissue HU window 400/40 is applied.

**Your own images** (de-identified, `.dcm`, `.png`, `.jpg`, `.tif`, `.npy`) are always allowed.

Images are resampled to the chosen grid (128×128 for training and analysis), zero-padded to square and, for CT/Other,
masked to the inscribed circle. ScanTwin downloads nothing automatically.
"""
        )
