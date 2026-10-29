"""
X-ray (projection radiography) modality
=======================================
Single projection image, no tomographic reconstruction.

    p = k * image                         (image intensity proxy -> line-integral map, max(p) = 4)
    I_mean = I0 * exp(-p)                 (Beer-Lambert, monochromatic)
    I ~ Poisson(I_mean)                   (quantum noise)
    p_noisy = -ln(max(I, 1) / I0)         (log recovery)
    image_hat = max(p_noisy / k, 0)       (back to image scale)

"Reconstruction" is this inverse log transform plus an optional denoising post-filter
(none / gaussian / median / anisotropic, selectable strength).

Cost = I0 / I0_ref (relative photon exposure proxy); reference I0_ref = 1e5. Not patient dose.
Model limits: no scatter, no anti-scatter grid, no polychromatic spectrum, no detector MTF or
electronic noise, no geometric magnification, no heel effect.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from modalities.base import CostInfo, LiveSession, MeasurementView, Modality, ParamSpec, geometric_steps
from simulation.postfilter import POST_FILTERS, apply_post_filter

XRAY_REFERENCE_I0 = 1e5
MAX_LINE_INTEGRAL = 4.0


def _scale(image: np.ndarray) -> float:
    peak = float(np.max(image))
    return MAX_LINE_INTEGRAL / peak if peak > 1e-12 else 1.0


def inverse_log(counts: np.ndarray, i0: float, k: float) -> np.ndarray:
    """-ln(max(counts, 1) / I0) / k, clipped at 0 (same guards as the CT noise model)."""
    p = -np.log(np.maximum(counts, 1.0) / i0)
    return np.maximum(p / k, 0.0)


class _XRayLive(LiveSession):
    """Exposure accumulates: photons of the full-dose image arrive in nested binomial slices."""

    unit = "% of photon budget"

    def __init__(self, modality: "XRayModality", truth, protocol, seed, ideal, steps):
        self.m, self.protocol, self.ideal = modality, protocol, ideal
        ideal_c, noisy_c, meta = modality.acquire(truth, protocol, seed)
        self.meta = meta
        self.total = 100
        steps = sorted(set(int(s) for s in steps) | {100})
        self.steps = steps
        self.shots: Dict[int, np.ndarray] = {}
        if ideal:
            for s in steps:
                self.shots[s] = ideal_c * (s / 100.0)
        else:
            rng = np.random.default_rng(int(seed) + 10_000)
            remaining = noisy_c.astype(np.int64)
            acc = np.zeros_like(remaining)
            prev = 0.0
            for s in steps:
                f = s / 100.0
                p = 1.0 if f >= 1.0 else (f - prev) / (1.0 - prev)
                got = rng.binomial(remaining, min(1.0, max(0.0, p)))
                acc = acc + got
                remaining = remaining - got
                self.shots[s] = acc.astype(np.float64)
                prev = f
            self.shots[100] = noisy_c.astype(np.float64)    # final frame is exactly the batch acquisition

    def state_at(self, n: int) -> Dict[str, Any]:
        counts = self.shots[int(n)]
        i0_eff = self.protocol["photon_budget"] * n / 100.0
        meta = dict(self.meta, photon_budget=i0_eff)
        recon, _ = self.m.reconstruct(counts, dict(self.protocol, photon_budget=i0_eff), meta)
        view = counts / max(i0_eff, 1e-12)
        exposure = 100.0 * i0_eff / XRAY_REFERENCE_I0
        return {"view": view, "recon": recon, "exposure_pct": exposure, "cost": exposure / 100.0}


class XRayModality(Modality):
    name = "X-ray"
    icon = "☢️"
    uses_circular_fov = False
    description = "Projection radiography: photon budget and an optional denoising post-filter."
    model_scope_text = (
        "Simplified radiography: the image is treated as a map of line integrals, transmitted through a monochromatic "
        "Beer-Lambert model with Poisson photon counting; 'reconstruction' is the inverse log transform plus an "
        "optional post-filter. No scatter or anti-scatter grid, polychromatic spectrum, detector blur, electronic "
        "noise or geometric effects. Photon budget is a relative exposure proxy, NOT patient dose. Not a validated "
        "detector simulation."
    )
    cost_labels = {
        "exposure": "Relative photon exposure",
        "exposure_short": "Rel. photon exposure",
        "cost": "Normalized acquisition cost",
        "measurement": "Raw radiograph (transmitted counts / I₀)",
        "recon": "Radiograph after inverse-log + post-filter",
        "pipeline": ("Image ➔ Line-integral map ➔ Beer-Lambert transmission ➔ Poisson noise ➔ Raw radiograph ➔ "
                     "Inverse log + post-filter ➔ Quality assessment"),
        "reduction": "Relative simulated photon-exposure reduction",
    }
    model_param_names = ("photon_budget", "post_filter", "filter_strength")
    default_criteria = {"min_ssim": 0.90, "min_psnr": 28.0, "min_cnr": 3.0}

    def protocol_space(self) -> List[ParamSpec]:
        return [
            ParamSpec("photon_budget", "Incident photon budget I₀", "ordered",
                      (1e2, 3e2, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6), 1e5, short="I0", unit="photons/pixel (relative proxy)",
                      numeric=True, log=True, fmt="{:.0e}",
                      help="Photons per detector pixel before the patient (relative exposure proxy). Fewer photons -> more quantum mottle."),
            ParamSpec("post_filter", "Denoising post-filter", "categorical", ("none", "gaussian", "median"), "none",
                      short="post", help="Optional spatial filter after the inverse log. Smooths noise but also blurs edges."),
            ParamSpec("filter_strength", "Filter strength", "ordered", (1, 2, 3), 1, short="strength",
                      numeric=True, when=("post_filter", ("gaussian", "median")),
                      help="Gaussian sigma = 0.7 x strength px; median window = 2 x strength + 1 px. Ignored when the filter is 'none'."),
        ]

    def reference_protocol(self) -> Dict[str, Any]:
        return {"photon_budget": XRAY_REFERENCE_I0, "post_filter": "none", "filter_strength": 1}

    def training_grid(self) -> List[Dict[str, Any]]:
        grid = []
        for i0 in (1e2, 3e2, 1e3, 3e3, 1e4, 3e4, 1e5):
            grid.append({"photon_budget": i0, "post_filter": "none", "filter_strength": 1})
            for f in ("gaussian", "median"):
                for s in (1, 2):
                    grid.append({"photon_budget": i0, "post_filter": f, "filter_strength": s})
        return grid

    def suggest_protocol(self, trial, include_alt_recon: bool = False) -> Dict[str, Any]:
        i0 = trial.suggest_float("photon_budget", 1e2, 1e5, log=True)
        filt = trial.suggest_categorical("post_filter", ["none", "gaussian", "median"])
        strength = trial.suggest_int("filter_strength", 1, 3) if filt != "none" else 1
        return {"photon_budget": i0, "post_filter": filt, "filter_strength": strength}

    def sweep_axes(self) -> Dict[str, Dict[str, Any]]:
        return {
            "primary": {"param": "photon_budget", "label": "Photon budget I₀", "values": [1e2, 3e2, 1e3, 3e3, 1e4, 3e4, 1e5],
                        "x_log": True},
            "secondary": {"param": "filter_strength", "label": "Post-filter strength", "values": [1, 2, 3], "x_log": False},
        }

    def recon_variants(self, protocol: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
        s = int(protocol.get("filter_strength", 1))
        return [
            ("No post-filter", {**protocol, "post_filter": "none"}),
            (f"Gaussian (strength {s})", {**protocol, "post_filter": "gaussian", "filter_strength": s}),
            (f"Median (strength {s})", {**protocol, "post_filter": "median", "filter_strength": s}),
        ]

    # ------------------------------------------------------------------ physics
    def acquisition_key(self, protocol):
        return (float(protocol["photon_budget"]),)

    def acquire(self, image: np.ndarray, protocol: Dict[str, Any], seed: Optional[int] = 42):
        i0 = float(protocol["photon_budget"])
        if i0 <= 0:
            raise ValueError("Photon budget must be positive.")
        k = _scale(image)
        mean_counts = i0 * np.exp(-k * np.maximum(image, 0.0))
        rng = np.random.default_rng(seed)
        noisy = rng.poisson(mean_counts).astype(np.float64)
        cost = self.acquisition_cost(protocol)
        meta = {"k": k, "photon_budget": i0, "seed": seed, "relative_photon_exposure_pct": cost.exposure_pct,
                "normalized_acquisition_cost": cost.cost}
        return mean_counts, noisy, meta

    def reconstruct(self, measurement, protocol: Dict[str, Any], meta: Dict[str, Any]):
        t0 = time.perf_counter()
        img = inverse_log(np.asarray(measurement, dtype=np.float64), float(protocol["photon_budget"]), meta["k"])
        strength = int(protocol.get("filter_strength", 1))
        img = apply_post_filter(img, protocol.get("post_filter", "none"), strength)
        return img, {"elapsed_time_ms": (time.perf_counter() - t0) * 1000.0, "post_filter": protocol.get("post_filter", "none")}

    def acquisition_cost(self, protocol: Dict[str, Any]) -> CostInfo:
        exposure = float(protocol["photon_budget"]) / XRAY_REFERENCE_I0
        return CostInfo(exposure, 100.0 * exposure, {"relative_photon_exposure": exposure})

    def measurement_view(self, measurement, meta, protocol) -> MeasurementView:
        i0 = max(float(protocol["photon_budget"]), 1e-12)
        return MeasurementView(np.asarray(measurement) / i0, "Raw radiograph (counts / I₀)", "Detector column", "Detector row",
                               "gray", note="Bright = more photons reached the detector (less attenuation)")

    # ------------------------------------------------------------------ live
    def live_session(self, truth, protocol, seed, ideal, steps) -> LiveSession:
        return _XRayLive(self, truth, protocol, seed, ideal, steps)

    def default_live_steps(self, protocol, image_shape, n_frames) -> List[int]:
        return geometric_steps(3, 100, n_frames)
