"""
ScanTwin: An Intelligent Digital Twin for Exposure-Aware Optimization of Medical Image Acquisition and Reconstruction
====================================================================================================================
Thin orchestrator: page config, CSS, sidebar, shared context and tab dispatch. All tab bodies live in ``ui/``.

DISCLAIMER:
Research and educational prototype. ScanTwin uses simplified image-acquisition and
exposure models and is not intended for clinical diagnosis, patient-specific
dosimetry, or scanner/protocol prescription.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st

st.set_page_config(page_title="ScanTwin | Medical Image Acquisition Digital Twin", page_icon="🔬", layout="wide",
                   initial_sidebar_state="expanded")

from ui import (tab_compare, tab_digital_twin, tab_experiments, tab_live, tab_methodology, tab_optimizer,  # noqa: E402
                tab_quality_cost, tab_recon, tab_train_analyze)
from ui.common import header, inject_css  # noqa: E402
from ui.context import build_context  # noqa: E402
from ui.sidebar import render_sidebar  # noqa: E402

inject_css()
header()
sb = render_sidebar()
ctx = build_context(sb)

if ctx.dataset_error:
    st.warning(ctx.dataset_error)
st.caption(f"**{ctx.modality.icon} {ctx.modality.name}** — ground truth: **{ctx.active_slice.name}** "
           f"({'real image data' if ctx.active_slice.is_real else 'synthetic / image data'}). {ctx.active_slice.note}")

TABS = [
    ("🎓 Start Here / Train & Analyze", tab_train_analyze),
    ("🖥️ Digital Twin", tab_digital_twin),
    ("⚡ Live Acquisition", tab_live),
    ("⚖️ Protocol Comparison (A vs B)", tab_compare),
    ("🔄 Reconstruction Analysis", tab_recon),
    ("📈 Quality vs Cost Lab", tab_quality_cost),
    ("🧠 Protocol Optimizer", tab_optimizer),
    ("🔬 Automated Experiments", tab_experiments),
    ("📖 Methodology & Validation", tab_methodology),
]
for container, (_, module) in zip(st.tabs([t for t, _ in TABS]), TABS):
    with container:
        module.render(ctx)
