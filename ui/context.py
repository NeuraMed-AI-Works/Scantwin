"""Builds the shared context (dataset, target, twin, single-protocol run) from the sidebar state; cached where deterministic."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import streamlit as st

from digital_twin.pipeline import SingleRun, run_single
from digital_twin.state import TwinState
from digital_twin.twin import DigitalTwin
from modalities import Modality, get_modality
from simulation.datasets import (DATA_DIR, SliceData, file_extension, load_slices, load_synthetic,
                                 suggest_lesion_center, synthetic_series)
from simulation.lesion import insert_lesion
from ui.sidebar import SidebarState


@dataclass
class Ctx:
    sb: SidebarState
    modality: Modality
    slices: List[SliceData]
    slice_idx: int
    active_slice: SliceData
    truth: np.ndarray
    lesion_mask: Optional[np.ndarray]
    bg_mask: Optional[np.ndarray]
    lesion_meta: Dict[str, Any]
    twin: DigitalTwin
    run: SingleRun
    ref_state: TwinState
    constraints: Dict[str, Any]
    dataset_error: Optional[str] = None

    # convenient aliases
    @property
    def protocol(self): return self.sb.protocol
    @property
    def seed(self): return self.sb.seed
    @property
    def ideal_mode(self): return self.sb.ideal_mode
    @property
    def target_on(self): return self.lesion_mask is not None
    @property
    def min_ssim(self): return self.sb.min_ssim
    @property
    def min_psnr(self): return self.sb.min_psnr
    @property
    def min_cnr(self): return self.sb.min_cnr
    @property
    def eff_min_cnr(self): return self.sb.min_cnr if self.target_on else None


@st.cache_data(show_spinner=False)
def get_dataset_slices(source: str, res: int, fov: bool, is_ct: bool, upload_name: str = "", upload_bytes: bytes = b""):
    if source in ("Shepp-Logan", "Concentric Disks"):
        return [load_synthetic(source, res, fov_circle=fov)]
    if source == "Synthetic slice series":
        return synthetic_series(res, n_slices=5, fov_circle=fov)
    if source.startswith("Local file: "):
        path = DATA_DIR / source.split(": ", 1)[1]
        ext = file_extension(path.name)
        return load_slices(path, size=res, fov_circle=fov, is_hu=True if (is_ct and ext in (".nii", ".nii.gz")) else None)
    ext = file_extension(upload_name)
    return load_slices(upload_bytes, size=res, name=upload_name, fov_circle=fov,
                       is_hu=True if (is_ct and ext in (".nii", ".nii.gz")) else None)


@st.cache_data(show_spinner=False)
def _cached_run(modality_name: str, truth: np.ndarray, protocol_json: str, seed: int, ideal: bool,
                lmask, bmask, constraints_json: str) -> SingleRun:
    return run_single(get_modality(modality_name), truth, json.loads(protocol_json), seed, ideal, lmask, bmask,
                      json.loads(constraints_json))


def build_context(sb: SidebarState) -> Ctx:
    modality = sb.modality
    err = None
    try:
        if sb.source.startswith("Upload"):
            if not sb.upload_bytes:
                slices = [load_synthetic("Shepp-Logan", sb.resolution, fov_circle=modality.uses_circular_fov)]
                err = "Upload an image or volume to use it; showing Shepp-Logan meanwhile."
            else:
                slices = get_dataset_slices(sb.source, sb.resolution, modality.uses_circular_fov, modality.name == "CT",
                                            sb.upload_name, sb.upload_bytes)
        else:
            slices = get_dataset_slices(sb.source, sb.resolution, modality.uses_circular_fov, modality.name == "CT")
    except (ValueError, ImportError, OSError) as exc:
        slices = [load_synthetic("Shepp-Logan", sb.resolution, fov_circle=modality.uses_circular_fov)]
        err = f"Could not load dataset ({exc}); showing Shepp-Logan instead."

    idx = 0
    if len(slices) > 1:
        with st.sidebar:
            idx = st.slider("Slice index", 1, len(slices), 1, key="slice_idx") - 1
    sl = slices[min(idx, len(slices) - 1)]
    base = sl.image

    if sb.target_on:
        centre = None if sl.name == "Shepp-Logan" else suggest_lesion_center(base, radius=sb.target_radius)
        truth, lmask, bmask, lmeta = insert_lesion(base, center=centre, radius=sb.target_radius, contrast=sb.target_contrast)
    else:
        truth, lmask, bmask, lmeta = base.copy(), None, None, {}

    twin = DigitalTwin(modality, truth, lmask, bmask, lmeta)
    constraints = {"min_ssim": sb.min_ssim, "min_psnr": sb.min_psnr, "min_cnr": sb.min_cnr if sb.target_on else None}
    proto = {k: (v.item() if hasattr(v, "item") else v) for k, v in sb.protocol.items()}
    run = _cached_run(modality.name, truth, json.dumps(proto, sort_keys=True), sb.seed, sb.ideal_mode, lmask, bmask,
                      json.dumps(constraints))
    ref_state, _, _ = twin.evaluate_protocol(modality.reference_protocol(), target_ssim_min=sb.min_ssim,
                                             target_cnr_min=constraints["min_cnr"], target_psnr_min=sb.min_psnr, seed=42)
    return Ctx(sb, modality, slices, idx, sl, truth, lmask, bmask, lmeta, twin, run, ref_state, constraints, err)
