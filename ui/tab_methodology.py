"""Tab: Methodology & Validation — modality-specific model-scope box plus the project documentation."""

from __future__ import annotations

import os

import streamlit as st

from ui.common import scope_box

DOCS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")
FILES = {
    "Numerical validation report": "validation_report.md",
    "Methodology pipeline": "methodology.md",
    "Mathematical equations (per modality)": "equations.md",
    "Assumptions": "assumptions.md",
    "Limitations": "limitations.md",
}


def render(ctx) -> None:
    m = ctx.modality
    st.subheader("Scientific Methodology & Numerical Validation")
    scope_box(m)
    st.markdown(f"**Selected modality: {m.name}** — {m.description}")
    st.caption("The documents below cover every modality; look for the section named after the selected modality.")
    choice = st.radio("Section", list(FILES), horizontal=True, key="doc_choice")
    path = os.path.join(DOCS, FILES[choice])
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            st.markdown(f.read())
    else:
        st.warning("Documentation file not found.")
