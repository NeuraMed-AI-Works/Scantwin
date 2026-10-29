"""
Single-protocol pipeline used by the interactive tabs (keeps computation out of the UI modules).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from digital_twin.twin import is_protocol_feasible
from metrics.cnr import compute_cnr, compute_roi_statistics
from metrics.detectability import assess_detectability
from metrics.image_quality import evaluate_image_quality
from modalities import Modality, get_modality
from modalities.base import CostInfo, MeasurementView
from utils.normalization import compute_error_statistics


@dataclass
class SingleRun:
    protocol: Dict[str, Any]
    ideal: Any
    noisy: Any
    meta: Dict[str, Any]
    active: Any
    recon: np.ndarray
    recon_meta: Dict[str, Any]
    view: MeasurementView
    metric_ref: np.ndarray
    quality: Dict[str, float]
    err_stats: Dict[str, Any]
    roi_stats: Optional[Dict[str, float]]
    cnr: Optional[float]
    feasible: bool
    cost: CostInfo


def run_single(
    modality: Modality,
    truth: np.ndarray,
    protocol: Dict[str, Any],
    seed: int,
    ideal_mode: bool,
    lesion_mask: Optional[np.ndarray],
    background_mask: Optional[np.ndarray],
    constraints: Dict[str, Any],
) -> SingleRun:
    proto = modality.normalize_protocol(protocol)
    proto.pop("seed", None)
    ideal, noisy, meta = modality.acquire(truth, proto, seed)
    active = ideal if ideal_mode else noisy
    recon, rmeta = modality.reconstruct(active, proto, meta)
    ref = modality.metric_reference(truth, proto)
    quality = evaluate_image_quality(ref, recon)
    has_target = lesion_mask is not None and background_mask is not None
    roi = compute_roi_statistics(recon, lesion_mask, background_mask) if has_target else None
    cnr = compute_cnr(recon, lesion_mask, background_mask) if has_target else None
    feasible = is_protocol_feasible({"ssim": quality["ssim"], "psnr": quality["psnr"], "cnr": cnr}, constraints)
    return SingleRun(
        protocol=proto, ideal=ideal, noisy=noisy, meta=meta, active=active, recon=recon, recon_meta=rmeta,
        view=modality.measurement_view(active, meta, proto), metric_ref=ref, quality=quality,
        err_stats=compute_error_statistics(ref, recon), roi_stats=roi, cnr=cnr, feasible=feasible,
        cost=modality.acquisition_cost(proto),
    )


def compare_recon_variants(
    modality: Modality, truth: np.ndarray, protocol: Dict[str, Any], seed: int, ideal_mode: bool,
    lesion_mask: Optional[np.ndarray], background_mask: Optional[np.ndarray],
) -> List[Dict[str, Any]]:
    """Same measurement, different reconstruction-side settings (fair comparison)."""
    proto = modality.normalize_protocol(protocol)
    proto.pop("seed", None)
    ideal, noisy, meta = modality.acquire(truth, proto, seed)
    meas = ideal if ideal_mode else noisy
    out = []
    for label, vp in modality.recon_variants(proto):
        vp = modality.normalize_protocol(vp)
        recon, rmeta = modality.reconstruct(meas, vp, meta)
        ref = modality.metric_reference(truth, vp)
        q = evaluate_image_quality(ref, recon)
        cnr = compute_cnr(recon, lesion_mask, background_mask) if lesion_mask is not None and background_mask is not None else None
        out.append({"label": label, "protocol": vp, "recon": recon, "ref": ref, "quality": q, "cnr": cnr,
                    "ms": float(rmeta.get("elapsed_time_ms", 0.0)),
                    "err": np.abs(ref - recon)})
    return out
