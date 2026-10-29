"""
CT modality (parallel-beam)
===========================
Wraps the original ScanTwin CT code; behaviour is identical to the Phase 1/2 pipeline:

    Radon transform -> Beer-Lambert -> Poisson counts -> log recovery -> FBP / SART

Protocol: photon budget I0, number of views, reconstruction method, FBP filter, SART iterations.
Cost: 0.7 * (I0 / I0_ref) + 0.3 * (views / views_ref), reference (I0=1e5, 360 views) = 1.0.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from config import (
    REFERENCE_I0,
    REFERENCE_PROJECTIONS,
    SUPPORTED_FBP_FILTERS,
)
from modalities.base import CostInfo, LiveSession, MeasurementView, Modality, ParamSpec, geometric_steps
from reconstruction.reconstruction_engine import reconstruct as ct_reconstruct
from simulation.noise_model import apply_photon_noise
from simulation.ordering import acquisition_order
from simulation.projector import compute_radon_projection

_RADON_CACHE: "OrderedDict[tuple, tuple]" = OrderedDict()
_RADON_CACHE_MAX = 24


def _cached_radon(image: np.ndarray, views: int) -> Tuple[np.ndarray, np.ndarray]:
    """Ideal sinogram, cached by image content and view count (deterministic, so safe to reuse)."""
    key = (image.shape, hash(np.ascontiguousarray(image).tobytes()), int(views))
    hit = _RADON_CACHE.get(key)
    if hit is not None:
        _RADON_CACHE.move_to_end(key)
        return hit
    out = compute_radon_projection(image, num_projections=int(views))
    _RADON_CACHE[key] = out
    if len(_RADON_CACHE) > _RADON_CACHE_MAX:
        _RADON_CACHE.popitem(last=False)
    return out


class _CTLive(LiveSession):
    unit = "views"

    def __init__(self, modality: "CTModality", truth: np.ndarray, protocol: Dict[str, Any], seed: int, ideal: bool):
        self.m = modality
        self.protocol = protocol
        ideal_s, noisy_s, meta = modality.acquire(truth, protocol, seed)
        self.full = ideal_s if ideal else noisy_s
        self.theta = meta["theta"]
        self.total = int(protocol["num_projections"])
        self.order = acquisition_order(self.total)

    def state_at(self, n: int) -> Dict[str, Any]:
        seen = np.sort(self.order[:n])
        p = self.protocol
        recon, _ = ct_reconstruct(
            self.full[:, seen], self.theta[seen], method=p["reconstruction_method"],
            filter_name=p["fbp_filter"], iterations=int(p["sart_iterations"]), circle=True,
        )
        view = np.full(self.full.shape, np.nan)
        view[:, seen] = self.full[:, seen]
        cost = 0.7 * (p["photon_budget"] / REFERENCE_I0) + 0.3 * (n / REFERENCE_PROJECTIONS)
        return {"view": view, "recon": recon, "exposure_pct": 100.0 * p["photon_budget"] / REFERENCE_I0, "cost": cost}


class CTModality(Modality):
    name = "CT"
    icon = "🩻"
    uses_circular_fov = True
    description = "Parallel-beam computed tomography: photon budget, number of views, reconstruction method."
    model_scope_text = (
        "Simplified 2D parallel-beam CT: Radon line integrals, monochromatic Beer-Lambert attenuation and Poisson "
        "photon counting, reconstructed by filtered back projection or SART. No scatter, beam hardening, fan/cone "
        "geometry, detector response or patient motion. The photon budget I₀ is a relative exposure proxy, NOT dose "
        "in mSv. This is a teaching model, not a validated scanner simulation."
    )
    cost_labels = {
        "exposure": "Relative photon exposure",
        "exposure_short": "Rel. photon exposure",
        "cost": "Normalized acquisition cost",
        "measurement": "Sinogram",
        "recon": "Reconstructed slice",
        "pipeline": ("Ground-truth slice ➔ Forward projector ➔ Ideal sinogram ➔ Beer-Lambert transmission ➔ "
                     "Poisson noise ➔ Noisy sinogram ➔ Reconstruction ➔ Quality assessment"),
        "reduction": "Relative simulated photon-exposure/cost reduction",
    }
    model_param_names = ("photon_budget", "num_projections", "fbp_filter")

    # ------------------------------------------------------------------ protocol
    def protocol_space(self) -> List[ParamSpec]:
        return [
            ParamSpec("photon_budget", "Incident photon budget I₀", "ordered",
                      (1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6), 1e5, short="I0", unit="photons/ray (relative proxy)",
                      numeric=True, log=True, fmt="{:.0e}",
                      help="Simulated incident photons per ray. A relative exposure proxy; lower values give more Poisson quantum mottle."),
            ParamSpec("num_projections", "Projection views", "ordered", (30, 45, 60, 90, 120, 180, 360), 180,
                      short="views", unit="views over 180°", numeric=True,
                      help="Angular samples over [0°, 180°). Too few views produce streak artifacts."),
            ParamSpec("reconstruction_method", "Reconstruction algorithm", "categorical", ("FBP", "SART"), "FBP",
                      short="recon", help="Analytical filtered back projection or algebraic iterative SART."),
            ParamSpec("fbp_filter", "Fourier apodization filter", "categorical", tuple(SUPPORTED_FBP_FILTERS),
                      "shepp-logan", short="filter", when=("reconstruction_method", "FBP"),
                      help="Frequency-domain filter on the projections. Smoother filters trade resolution for noise."),
            ParamSpec("sart_iterations", "SART iterations", "int", (), 2, short="SART it", min=1, max=6, step=1,
                      when=("reconstruction_method", "SART"),
                      help="Number of full algebraic projection/back-projection sweeps."),
        ]

    def reference_protocol(self) -> Dict[str, Any]:
        return {"photon_budget": REFERENCE_I0, "num_projections": REFERENCE_PROJECTIONS,
                "reconstruction_method": "FBP", "fbp_filter": "ramp", "sart_iterations": 1}

    def training_grid(self) -> List[Dict[str, Any]]:
        return [
            {"photon_budget": i0, "num_projections": v, "reconstruction_method": "FBP", "fbp_filter": f, "sart_iterations": 1}
            for v in (30, 60, 90, 120, 180, 360) for i0 in (1e3, 3e3, 1e4, 3e4, 1e5) for f in ("ramp", "hann")
        ]

    def suggest_protocol(self, trial: Any, include_alt_recon: bool = False) -> Dict[str, Any]:
        i0 = trial.suggest_float("photon_budget", 1e3, 1e6, log=True)
        views = trial.suggest_int("num_projections", 30, 360, step=15)
        methods = ["FBP", "SART"] if include_alt_recon else ["FBP"]
        method = trial.suggest_categorical("reconstruction_method", methods)
        if method == "FBP":
            filt = trial.suggest_categorical("fbp_filter", list(SUPPORTED_FBP_FILTERS))
            iters = 1
        else:
            filt = "ramp"
            iters = trial.suggest_int("sart_iterations", 1, 4)
        return {"photon_budget": i0, "num_projections": views, "reconstruction_method": method,
                "fbp_filter": filt, "sart_iterations": iters}

    def sweep_axes(self) -> Dict[str, Dict[str, Any]]:
        return {
            "primary": {"param": "photon_budget", "label": "Photon budget I₀", "values": [1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6],
                        "x_log": True},
            "secondary": {"param": "num_projections", "label": "Projection views", "values": [30, 45, 60, 90, 120, 180, 360],
                          "x_log": False},
        }

    def recon_variants(self, protocol: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
        base = dict(protocol)
        out: List[Tuple[str, Dict[str, Any]]] = []
        for f in ("ramp", "shepp-logan", "hann"):
            out.append((f"FBP ({f})", {**base, "reconstruction_method": "FBP", "fbp_filter": f}))
        iters = int(base.get("sart_iterations", 2)) if base.get("reconstruction_method") == "SART" else 2
        out.append((f"SART ({iters} it)", {**base, "reconstruction_method": "SART", "sart_iterations": iters}))
        return out

    # ------------------------------------------------------------------ physics
    def acquisition_key(self, protocol: Dict[str, Any]) -> Tuple:
        return (float(protocol["photon_budget"]), int(protocol["num_projections"]))

    def acquire(self, image: np.ndarray, protocol: Dict[str, Any], seed: Optional[int] = 42):
        views = int(protocol["num_projections"])
        i0 = float(protocol["photon_budget"])
        ideal, theta = _cached_radon(image, views)
        noisy = apply_photon_noise(ideal, incident_photons=i0, seed=seed)
        cost = self.acquisition_cost(protocol)
        meta = {
            "theta": theta, "num_projections": views, "photon_budget": i0, "seed": seed,
            "relative_photon_exposure_pct": cost.exposure_pct,
            "projection_fraction_pct": 100.0 * views / REFERENCE_PROJECTIONS,
            "normalized_acquisition_cost": cost.cost,
        }
        return ideal, noisy, meta

    def reconstruct(self, measurement, protocol: Dict[str, Any], meta: Dict[str, Any]):
        return ct_reconstruct(
            measurement, meta["theta"], method=protocol.get("reconstruction_method", "FBP"),
            filter_name=protocol.get("fbp_filter", "ramp"), iterations=int(protocol.get("sart_iterations", 1)),
            circle=True,
        )

    def acquisition_cost(self, protocol: Dict[str, Any]) -> CostInfo:
        exposure = float(protocol["photon_budget"]) / REFERENCE_I0
        views = int(protocol["num_projections"]) / REFERENCE_PROJECTIONS
        return CostInfo(
            cost=0.7 * exposure + 0.3 * views, exposure_pct=100.0 * exposure,
            components={"relative_photon_exposure": exposure, "view_fraction": views},
        )

    def measurement_view(self, measurement, meta, protocol) -> MeasurementView:
        return MeasurementView(
            np.asarray(measurement), "Sinogram", "Projection angle index (θ: 0 → 180°)", "Detector bin s", "viridis",
            note=f"{int(protocol['num_projections'])} views across 180°",
        )

    # ------------------------------------------------------------------ live
    def live_session(self, truth, protocol, seed, ideal, steps) -> LiveSession:
        return _CTLive(self, truth, protocol, seed, ideal)

    def default_live_steps(self, protocol, image_shape, n_frames) -> List[int]:
        total = int(protocol["num_projections"])
        return geometric_steps(max(4, total // 24), total, n_frames)


class OtherModality(CTModality):
    """'Other' 2D images: the CT parallel-beam model used as a clearly labelled *generic proxy*."""

    name = "Other"
    icon = "🖼️"
    description = "Any other 2D image: pushed through the generic parallel-beam projection proxy (not modality physics)."
    model_scope_text = (
        "PROXY ONLY. For image types without a dedicated model (e.g. endoscopy, microscopy, photographs) ScanTwin uses "
        "the generic CT-style parallel-beam projection + Poisson model purely as an image-formation proxy. Results "
        "describe how this simulated projection/reconstruction chain degrades the image, not how the original "
        "imaging device behaves. Not a validated model of anything."
    )
    cost_labels = dict(CTModality.cost_labels, **{
        "measurement": "Projection data (proxy)",
        "recon": "Reconstructed image (proxy)",
        "pipeline": ("Image ➔ Generic projection proxy ➔ Ideal projections ➔ Transmission/Poisson proxy ➔ "
                     "Noisy projections ➔ Reconstruction ➔ Quality assessment"),
    })
