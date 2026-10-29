"""
Live (Real-Time) Acquisition Engine
===================================
Generators that emit one result frame at a time so a UI can render the digital twin while a
simulated scan is in progress, for every modality.

``stream_modality``
    One object, one protocol. The WHOLE scan is generated once (single seed) and its data is then
    *revealed* progressively; after each batch the partial data are reconstructed and measured against the
    ground truth. What "progressively" means depends on the modality:

    * CT          projection views arrive in golden-angle order
    * MRI         phase-encode lines arrive centre first, then outward
    * Ultrasound  compounded frames accumulate one by one (speckle visibly falls)
    * X-ray       photon budget is ramped up (exposure accumulates, noise visibly falls)

    The final frame equals a normal (non-live) acquisition with the same seed and protocol.

``stream_series`` / ``stream_slices``
    A series of slices, each acquired with the same protocol, emitted slice by slice.

All numbers come from real simulation calls; nothing here is hardcoded or interpolated.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence

import numpy as np

from config import REFERENCE_PROJECTIONS
from digital_twin.twin import is_protocol_feasible
from metrics.cnr import compute_cnr
from metrics.image_quality import evaluate_image_quality
from modalities import Modality, get_modality
from modalities.base import geometric_steps
from simulation.datasets import SliceData, suggest_lesion_center
from simulation.lesion import insert_lesion
from simulation.ordering import GOLDEN_RATIO_CONJUGATE, acquisition_order  # noqa: F401  (re-exported)


@dataclass
class LiveFrame:
    """One emitted state of the running digital twin."""

    index: int
    label: str
    acquired: int                       # units of data received so far
    total: int                          # units in the complete scan
    unit: str                           # "views" | "k-space lines" | "frames" | "% of photon budget" | "slice"
    truth: np.ndarray
    partial_measurement: np.ndarray     # 2D raw-data view; NaN = not yet acquired
    recon: np.ndarray
    metrics: Dict[str, float]           # ssim, psnr, rmse, cnr (cnr is NaN without a target)
    relative_photon_exposure_pct: float  # the modality's headline exposure / time proxy (% of reference)
    cost: float                         # normalized cost of the data acquired so far
    elapsed_ms: float
    modality: str = "CT"
    lesion_mask: Optional[np.ndarray] = None
    background_mask: Optional[np.ndarray] = None

    # --- CT-era aliases kept for backwards compatibility
    @property
    def views_acquired(self) -> int:
        return self.acquired

    @property
    def views_total(self) -> int:
        return self.total

    @property
    def partial_sinogram(self) -> np.ndarray:
        return self.partial_measurement

    @property
    def projection_fraction_pct(self) -> float:
        return 100.0 * self.acquired / REFERENCE_PROJECTIONS

    @property
    def scan_progress_pct(self) -> float:
        return 100.0 * self.acquired / self.total


def measure(
    truth: np.ndarray,
    recon: np.ndarray,
    lesion_mask: Optional[np.ndarray],
    background_mask: Optional[np.ndarray],
) -> Dict[str, float]:
    """Image-quality metrics plus target CNR (NaN when no target is present; the UI shows 'n/a')."""
    quality = evaluate_image_quality(truth, recon)
    cnr = (
        compute_cnr(recon, lesion_mask, background_mask)
        if lesion_mask is not None and background_mask is not None
        else float("nan")
    )
    return {"ssim": quality["ssim"], "psnr": quality["psnr"], "rmse": quality["rmse"], "cnr": float(cnr)}


def default_view_steps(total_views: int, n_frames: int = 8) -> List[int]:
    """Roughly geometric schedule of cumulative view counts ending exactly at ``total_views`` (CT)."""
    first = max(4, total_views // 24)
    steps = np.unique(np.round(np.geomspace(first, total_views, n_frames)).astype(int))
    steps[-1] = total_views
    return [int(s) for s in steps]


def stream_modality(
    modality: Modality,
    truth: np.ndarray,
    lesion_mask: Optional[np.ndarray],
    background_mask: Optional[np.ndarray],
    protocol: Dict[str, Any],
    seed: int = 42,
    steps: Optional[Sequence[int]] = None,
    n_frames: int = 8,
    ideal: bool = False,
    frame_delay_s: float = 0.0,
) -> Iterator[LiveFrame]:
    """Yield a frame after each batch of data has been acquired (see module docstring)."""
    proto = modality.normalize_protocol(protocol)
    proto.pop("seed", None)
    steps = list(steps) if steps is not None else modality.default_live_steps(proto, truth.shape, n_frames)
    session = modality.live_session(truth, proto, int(seed), ideal, steps)
    if not steps or steps[-1] != session.total or any(s < 1 or s > session.total for s in steps):
        raise ValueError(f"steps must lie within [1, {session.total}] and end at {session.total}")
    metric_ref = modality.metric_reference(truth, proto)

    for idx, n_seen in enumerate(steps):
        t0 = time.perf_counter()
        st = session.state_at(int(n_seen))
        metrics = measure(metric_ref, st["recon"], lesion_mask, background_mask)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        yield LiveFrame(
            index=idx, label=f"{n_seen}/{session.total} {session.unit}", acquired=int(n_seen), total=int(session.total),
            unit=session.unit, truth=metric_ref, partial_measurement=st["view"], recon=st["recon"], metrics=metrics,
            relative_photon_exposure_pct=float(st["exposure_pct"]), cost=float(st["cost"]), elapsed_ms=elapsed_ms,
            modality=modality.name, lesion_mask=lesion_mask, background_mask=background_mask,
        )
        if frame_delay_s > 0:
            time.sleep(frame_delay_s)


def stream_series(
    modality: Modality,
    slices: Sequence[SliceData],
    protocol: Dict[str, Any],
    lesion_on: bool = True,
    lesion_radius: float = 6.0,
    lesion_contrast: float = 0.08,
    seed: int = 42,
    ideal: bool = False,
    frame_delay_s: float = 0.0,
) -> Iterator[LiveFrame]:
    """Acquire and reconstruct a series slice by slice with one protocol, yielding one frame per slice."""
    proto = modality.normalize_protocol(protocol)
    proto.pop("seed", None)
    cost = modality.acquisition_cost(proto)
    for i, sl in enumerate(slices):
        t0 = time.perf_counter()
        if lesion_on:
            centre = suggest_lesion_center(sl.image, radius=lesion_radius)
            truth, lmask, bmask, _ = insert_lesion(sl.image, center=centre, radius=lesion_radius, contrast=lesion_contrast)
        else:
            truth, lmask, bmask = sl.image, None, None
        ideal_m, noisy_m, meta = modality.acquire(truth, proto, seed + i)
        meas = ideal_m if ideal else noisy_m
        recon, _ = modality.reconstruct(meas, proto, meta)
        metric_ref = modality.metric_reference(truth, proto)
        metrics = measure(metric_ref, recon, lmask, bmask)
        view = modality.measurement_view(meas, meta, proto).image
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        yield LiveFrame(
            index=i, label=sl.name, acquired=1, total=1, unit="slice", truth=metric_ref, partial_measurement=view,
            recon=recon, metrics=metrics, relative_photon_exposure_pct=cost.exposure_pct, cost=cost.cost,
            elapsed_ms=elapsed_ms, modality=modality.name, lesion_mask=lmask, background_mask=bmask,
        )
        if frame_delay_s > 0:
            time.sleep(frame_delay_s)


# ---------------------------------------------------------------------------------------------
# CT-specific entry points kept with their original signatures (used by tests and older scripts)
# ---------------------------------------------------------------------------------------------
def _ct_protocol(photon_budget: float, num_views: int, method: str, filter_name: str, iterations: int) -> Dict[str, Any]:
    return {"photon_budget": float(photon_budget), "num_projections": int(num_views), "reconstruction_method": method,
            "fbp_filter": filter_name, "sart_iterations": int(iterations)}


def stream_scan(
    truth: np.ndarray,
    lesion_mask: Optional[np.ndarray],
    background_mask: Optional[np.ndarray],
    photon_budget: float,
    total_views: int,
    view_steps: Optional[Sequence[int]] = None,
    seed: int = 42,
    method: str = "FBP",
    filter_name: str = "ramp",
    iterations: int = 2,
    ideal: bool = False,
    frame_delay_s: float = 0.0,
) -> Iterator[LiveFrame]:
    """CT live scan: projection views arrive progressively in golden-angle order."""
    steps = list(view_steps) if view_steps is not None else default_view_steps(total_views)
    if steps[-1] != total_views or any(s < 2 or s > total_views for s in steps):
        raise ValueError("view_steps must be within [2, total_views] and end at total_views")
    return stream_modality(
        get_modality("CT"), truth, lesion_mask, background_mask,
        _ct_protocol(photon_budget, total_views, method, filter_name, iterations),
        seed=seed, steps=steps, ideal=ideal, frame_delay_s=frame_delay_s,
    )


def stream_slices(
    slices: Sequence[SliceData],
    photon_budget: float,
    num_views: int,
    lesion_on: bool = True,
    lesion_radius: float = 6.0,
    lesion_contrast: float = 0.08,
    seed: int = 42,
    method: str = "FBP",
    filter_name: str = "ramp",
    iterations: int = 2,
    frame_delay_s: float = 0.0,
) -> Iterator[LiveFrame]:
    """CT slice stream (see :func:`stream_series` for other modalities)."""
    return stream_series(
        get_modality("CT"), slices, _ct_protocol(photon_budget, num_views, method, filter_name, iterations),
        lesion_on=lesion_on, lesion_radius=lesion_radius, lesion_contrast=lesion_contrast, seed=seed,
        frame_delay_s=frame_delay_s,
    )


def first_frame_meeting_criteria(
    frames: Sequence[LiveFrame],
    min_ssim: float,
    min_cnr: Optional[float],
    min_psnr: Optional[float] = None,
) -> Optional[LiveFrame]:
    """Earliest frame satisfying the experimental criteria (CNR only when a target exists); None if never met."""
    constraints = {"min_ssim": min_ssim, "min_cnr": min_cnr, "min_psnr": min_psnr}
    for fr in frames:
        m = {"ssim": fr.metrics["ssim"], "psnr": fr.metrics["psnr"]}
        if not np.isnan(fr.metrics["cnr"]):
            m["cnr"] = fr.metrics["cnr"]
        if is_protocol_feasible(m, constraints):          # single authoritative feasibility function
            return fr
    return None
