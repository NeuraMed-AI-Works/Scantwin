"""
Ultrasound modality (speckle model)
===================================
NOT a wave-propagation (Field-II-style) simulation. The image is an echogenicity/reflectivity map r >= 0.

    frame_k = r * s_k + e_k,   s_k ~ Exponential(1) (fully developed speckle), e_k ~ N(0, 0.01^2)
    compounded = mean_k(frame_k)  over N independent frames (transmit angles)   -> speckle variance ~ 1/N
    display = [post-filter](compounded), then optional log compression  L(x) = ln(1 + g x) / ln(1 + g), g = 100

Metrics are computed in the DISPLAYED domain: with log compression on, the ground truth is compressed with the
same L, so SSIM/PSNR values with compression on/off are not strictly comparable.

Cost = N / N_ref (acquisition-time proxy), N_ref = 32 frames. Reference: 32 frames, log off, no filter.
Not modelled: depth attenuation, beamforming, point-spread function, correlated speckle between
compounded angles, TGC, probe geometry, motion.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from modalities.base import CostInfo, LiveSession, MeasurementView, Modality, ParamSpec, geometric_steps
from simulation.postfilter import apply_post_filter

US_N_REF = 32
US_LOG_GAIN = 100.0
US_ELECTRONIC_SIGMA = 0.01


def log_compress(x: np.ndarray, gain: float = US_LOG_GAIN) -> np.ndarray:
    return np.log1p(gain * np.maximum(x, 0.0)) / np.log1p(gain)


def _display(compounded: np.ndarray, protocol: Dict[str, Any]) -> np.ndarray:
    img = apply_post_filter(np.maximum(compounded, 0.0), protocol.get("post_filter", "none"), int(protocol.get("filter_strength", 1)))
    if protocol.get("log_compression", "off") == "on":
        img = log_compress(img)
    return img


class _USLive(LiveSession):
    unit = "frames"

    def __init__(self, modality: "UltrasoundModality", truth, protocol, seed, ideal):
        self.m, self.protocol, self.ideal = modality, protocol, ideal
        ideal_f, noisy_f, meta = modality.acquire(truth, protocol, seed)
        self.frames = ideal_f if ideal else noisy_f
        self.total = int(protocol["n_frames"]) if not ideal else int(protocol["n_frames"])

    def state_at(self, n: int) -> Dict[str, Any]:
        n_eff = min(n, self.frames.shape[0])
        compounded = self.frames[:n_eff].mean(axis=0)
        recon = _display(compounded, self.protocol)
        last = self.frames[n_eff - 1]
        view = log_compress(last) if self.protocol.get("log_compression") == "on" else last
        cost = n / US_N_REF
        return {"view": view, "recon": recon, "exposure_pct": 100.0 * cost, "cost": cost}


class UltrasoundModality(Modality):
    name = "Ultrasound"
    icon = "🔊"
    uses_circular_fov = False
    description = "Speckle model: number of compounded frames, log compression, post-filter."
    model_scope_text = (
        "Speckle-statistics model, NOT an acoustic simulation: echogenicity × exponentially distributed speckle per "
        "frame, averaged over N independent compounded frames (speckle variance falls as 1/N), with optional log "
        "compression and a simple post-filter. No depth attenuation, beamforming, point-spread function, correlated "
        "speckle between angles, TGC or probe geometry. Cost is an acquisition-time proxy (frames). Metrics are "
        "computed in the displayed (possibly log-compressed) domain."
    )
    cost_labels = {
        "exposure": "Relative acquisition-time proxy (frames)",
        "exposure_short": "Acq.-time proxy",
        "cost": "Normalized acquisition cost (N / N_ref)",
        "measurement": "Single B-mode frame",
        "recon": "Compounded B-mode image",
        "pipeline": ("Echogenicity map ➔ × speckle (Exp(1)) per frame ➔ + electronic noise ➔ N frames ➔ Compounding "
                     "(mean) ➔ Post-filter ➔ Log compression ➔ Quality assessment"),
        "reduction": "Relative simulated acquisition-time reduction",
    }
    model_param_names = ("n_frames", "log_compression", "post_filter", "filter_strength")
    # Speckle limits SSIM against the clean reflectivity map, so the starting criteria are lower.
    default_criteria = {"min_ssim": 0.50, "min_psnr": 20.0, "min_cnr": 2.0}

    def protocol_space(self) -> List[ParamSpec]:
        return [
            ParamSpec("n_frames", "Compounded frames N", "ordered", (1, 2, 4, 8, 16, 32), 8, short="N", unit="frames",
                      numeric=True, log=True,
                      help="Independent frames (transmit angles) averaged. Speckle variance falls as 1/N; acquisition time rises as N."),
            ParamSpec("log_compression", "Log compression", "categorical", ("off", "on"), "off", short="log",
                      help="B-mode log compression of the amplitude. Metrics are computed in the displayed domain."),
            ParamSpec("post_filter", "Post-filter", "categorical", ("none", "median", "anisotropic"), "none", short="post",
                      help="Median or simple Perona-Malik anisotropic smoothing of the compounded image."),
            ParamSpec("filter_strength", "Filter strength", "ordered", (1, 2, 3), 1, short="strength", numeric=True,
                      when=("post_filter", ("median", "anisotropic")),
                      help="Median window = 2 x strength + 1 px; anisotropic = 3 x strength diffusion iterations. Ignored for 'none'."),
        ]

    def reference_protocol(self) -> Dict[str, Any]:
        return {"n_frames": US_N_REF, "log_compression": "off", "post_filter": "none", "filter_strength": 1}

    def training_grid(self) -> List[Dict[str, Any]]:
        grid = []
        for n in (1, 2, 4, 8, 16, 32):
            for log in ("off", "on"):
                grid.append({"n_frames": n, "log_compression": log, "post_filter": "none", "filter_strength": 1})
                for f in ("median", "anisotropic"):
                    for s in (1, 2):
                        grid.append({"n_frames": n, "log_compression": log, "post_filter": f, "filter_strength": s})
        return grid

    def suggest_protocol(self, trial, include_alt_recon: bool = False) -> Dict[str, Any]:
        n = trial.suggest_categorical("n_frames", [1, 2, 4, 8, 16, 32])
        log = trial.suggest_categorical("log_compression", ["off", "on"])
        filt = trial.suggest_categorical("post_filter", ["none", "median", "anisotropic"])
        strength = trial.suggest_int("filter_strength", 1, 3) if filt != "none" else 1
        return {"n_frames": n, "log_compression": log, "post_filter": filt, "filter_strength": strength}

    def sweep_axes(self) -> Dict[str, Dict[str, Any]]:
        return {
            "primary": {"param": "n_frames", "label": "Compounded frames N", "values": [1, 2, 4, 8, 16, 32], "x_log": True},
            "secondary": {"param": "filter_strength", "label": "Post-filter strength", "values": [1, 2, 3], "x_log": False},
        }

    def recon_variants(self, protocol: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
        s = int(protocol.get("filter_strength", 1))
        return [
            ("No post-filter", {**protocol, "post_filter": "none"}),
            (f"Median (strength {s})", {**protocol, "post_filter": "median", "filter_strength": s}),
            (f"Anisotropic (strength {s})", {**protocol, "post_filter": "anisotropic", "filter_strength": s}),
        ]

    def metric_reference(self, truth: np.ndarray, protocol: Dict[str, Any]) -> np.ndarray:
        return log_compress(truth) if protocol.get("log_compression", "off") == "on" else truth

    # ------------------------------------------------------------------ physics
    def acquisition_key(self, protocol):
        return (int(protocol["n_frames"]),)

    def acquire(self, image: np.ndarray, protocol: Dict[str, Any], seed: Optional[int] = 42):
        n = int(protocol["n_frames"])
        if n < 1:
            raise ValueError("n_frames must be >= 1")
        r = np.maximum(np.asarray(image, dtype=np.float64), 0.0)
        rng = np.random.default_rng(seed)
        frames = np.empty((n,) + r.shape, dtype=np.float64)
        for k in range(n):                               # per-frame draws: frames of N=8 are a prefix of N=32
            speckle = rng.exponential(1.0, size=r.shape)
            frames[k] = r * speckle + US_ELECTRONIC_SIGMA * rng.standard_normal(r.shape)
        cost = self.acquisition_cost(protocol)
        meta = {"n_frames": n, "seed": seed, "relative_acquisition_time_pct": cost.exposure_pct,
                "normalized_acquisition_cost": cost.cost}
        return r[None], frames, meta

    def reconstruct(self, measurement, protocol: Dict[str, Any], meta: Dict[str, Any]):
        t0 = time.perf_counter()
        compounded = np.asarray(measurement, dtype=np.float64).mean(axis=0)
        img = _display(compounded, protocol)
        return img, {"elapsed_time_ms": (time.perf_counter() - t0) * 1000.0}

    def acquisition_cost(self, protocol: Dict[str, Any]) -> CostInfo:
        frac = int(protocol["n_frames"]) / US_N_REF
        return CostInfo(frac, 100.0 * frac, {"frame_fraction": frac})

    def measurement_view(self, measurement, meta, protocol) -> MeasurementView:
        frame = np.asarray(measurement)[0]
        if protocol.get("log_compression", "off") == "on":
            frame = log_compress(frame)
        note = "single frame (first of N)" if np.asarray(measurement).shape[0] > 1 else "noise-free echogenicity (no speckle)"
        return MeasurementView(frame, "B-mode frame (single, un-compounded)", "Lateral position", "Depth", "gray", note=note)

    # ------------------------------------------------------------------ live
    def live_session(self, truth, protocol, seed, ideal, steps) -> LiveSession:
        return _USLive(self, truth, protocol, seed, ideal)

    def default_live_steps(self, protocol, image_shape, n_frames) -> List[int]:
        total = int(protocol["n_frames"])
        return list(range(1, total + 1)) if total <= max(2, n_frames) else geometric_steps(1, total, n_frames)
