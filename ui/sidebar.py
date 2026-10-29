"""Sidebar: global modality selector, dataset / phantom, target, protocol, criteria."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import streamlit as st

from modalities import Modality, get_modality, modality_names
from simulation.datasets import DATA_DIR, SUPPORTED_EXTENSIONS, list_local_datasets
from ui.common import render_protocol_controls

UPLOAD_TYPES = [e.lstrip(".") for e in SUPPORTED_EXTENSIONS if e != ".nii.gz"] + ["gz"]


@dataclass
class SidebarState:
    modality: Modality
    source: str
    resolution: int
    upload_name: str
    upload_bytes: bytes
    target_on: bool
    target_radius: float
    target_contrast: float
    ideal_mode: bool
    seed: int
    protocol: Dict[str, Any]
    min_ssim: float
    min_psnr: Optional[float]
    min_cnr: float


def render_sidebar() -> SidebarState:
    with st.sidebar:
        st.header("⚙️ Simulation Controls")
        name = st.radio(
            "Imaging modality", modality_names(), horizontal=False, key="modality_sel",
            help="Drives the protocol controls, physics model, raw-data view, cost labels and wording of every tab.",
        )
        modality = get_modality(name)
        st.caption(modality.description)
        k = modality.name

        with st.expander("1. Dataset / Phantom & Target", expanded=True):
            local = list_local_datasets()
            options = (["Shepp-Logan", "Concentric Disks", "Synthetic slice series"]
                       + [f"Local file: {p.name}" for p in local] + ["Upload image / volume"])
            source = st.selectbox("Dataset / Phantom", options, index=0, key=f"src_{k}",
                                  help="Analytical phantoms, a small synthetic series, files placed in data/, or an upload. "
                                       "Nothing is downloaded automatically. Use only public, de-identified or licensed images.")
            resolution = st.select_slider("Grid resolution (pixels)", options=[128, 256], value=128, key="res",
                                          help="128 keeps everything fast; 256 is slower but sharper.")
            up_name, up_bytes = "", b""
            if source.startswith("Upload"):
                up = st.file_uploader("Image, stack or volume", type=UPLOAD_TYPES, key=f"up_{k}",
                                      help=".dcm, .nii/.nii.gz (needs nibabel), .npy/.npz, .png/.jpg/.tif")
                if up is not None:
                    up_name, up_bytes = up.name, up.getvalue()
            target_on = st.checkbox("Insert synthetic target", value=True, key=f"tgt_{k}",
                                    help="Adds a known low-contrast disc so a contrast-to-noise ratio (CNR) can be measured.")
            if target_on:
                radius = st.slider("Target radius (px)", 3.0, 18.0, 8.0 if resolution == 256 else 5.0, 0.5, key=f"tr_{k}",
                                   help="Radius of the inserted synthetic target.")
                contrast = st.slider("Target contrast offset", 0.01, 0.25, 0.08, 0.01, key=f"tc_{k}",
                                     help="Intensity offset of the target relative to its surroundings.")
            else:
                radius, contrast = 8.0, 0.0

        with st.expander(f"2. {modality.name} acquisition & reconstruction", expanded=True):
            ideal_mode = st.radio("Measurement shown / reconstructed", ["Noisy (stochastic)", "Ideal (noise-free)"], index=0,
                                  key=f"mode_{k}",
                                  help="Switch between the noisy measurement and the ideal noise-free measurement of the same protocol.") .startswith("Ideal")
            protocol = render_protocol_controls(modality, "sb")
            seed = int(st.number_input("Random noise seed", value=42, step=1, key=f"seed_{k}",
                                       help="Same seed + same parameters = identical result; another seed = another noise realization."))

        with st.expander("3. Experimental quality criteria", expanded=True):
            dc = modality.default_criteria
            min_ssim = st.slider("Minimum SSIM", 0.30, 0.99, float(dc["min_ssim"]), 0.01, key=f"mssim_{k}",
                                 help="User-set experimental criterion (not a clinical threshold).")
            use_psnr = st.checkbox("Also require a minimum PSNR", value=False, key=f"upsnr_{k}")
            min_psnr = st.slider("Minimum PSNR (dB)", 10.0, 50.0, float(dc["min_psnr"]), 0.5, key=f"mpsnr_{k}") if use_psnr else None
            min_cnr = st.slider("Minimum CNR (needs a target)", 0.5, 10.0, float(dc["min_cnr"]), 0.1, key=f"mcnr_{k}",
                                help="Applied only when a synthetic target is inserted.")
    return SidebarState(modality, source, int(resolution), up_name, up_bytes, bool(target_on), float(radius), float(contrast),
                        bool(ideal_mode), seed, protocol, float(min_ssim), min_psnr, float(min_cnr))
