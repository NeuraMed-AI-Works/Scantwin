"""
MRI modality (Cartesian k-space)
================================
The image is the magnitude of the transverse magnetization (phase 0).

    k = fftshift(fft2(image, norm="ortho"))                 (centred, unitary)
    mask: fully sampled centre block (half of the acquired lines, >= 8) + random outer phase-encode
          lines -> about N/R lines in total (acceleration factor R)
    k_meas = mask * (k + sigma * (n_re + i n_im)),  sigma = sigma0 / sqrt(NEX)
    recon  = | ifft2( window * k_meas ) |                    (zero-filled magnitude, Rician-type noise)

``window`` is an optional Hann k-space window. No density compensation is applied (plain zero filling),
so skipped outer lines mainly cost resolution/aliasing, not intensity scale. Phase-encode direction =
rows; every sampled line is complete.

Cost (scan-time proxy) = (sampled lines / total lines) * NEX / NEX_ref with NEX_ref = 4; the sampled-line
fraction is nominally 1 / R. Reference: R = 1, NEX = 4, no window -> 1.0.

Not modelled: coil sensitivities, parallel imaging, compressed sensing, T1/T2 contrast physics,
relaxation, field inhomogeneity, motion, phase. Learning-based or compressed-sensing reconstruction
would be future work.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from modalities.base import CostInfo, LiveSession, MeasurementView, Modality, ParamSpec

MRI_SIGMA0 = 0.015       # k-space noise std per real/imag component at NEX = 1 (image intensity units, arbitrary)
MRI_NEX_REF = 4
CENTER_SHARE = 0.5       # share of the acquired lines forming the fully sampled centre block
MIN_CENTER_LINES = 8


def _kspace(image: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.fft2(np.asarray(image, dtype=np.float64), norm="ortho"))


def _image_from_k(k: np.ndarray) -> np.ndarray:
    return np.abs(np.fft.ifft2(np.fft.ifftshift(k), norm="ortho"))


def sampling_plan(n_lines: int, accel: int, rng: np.random.Generator) -> Dict[str, Any]:
    """Boolean line mask (+ unit weights, kept for clarity) for acceleration ``accel`` (R >= 1)."""
    accel = max(1, int(accel))
    target = int(np.clip(round(n_lines / accel), 1, n_lines))
    centre_n = int(min(target, max(MIN_CENTER_LINES, round(CENTER_SHARE * target))))
    mid = n_lines // 2
    c0 = mid - centre_n // 2
    centre = np.arange(c0, c0 + centre_n)
    outer = np.setdiff1d(np.arange(n_lines), centre)
    n_out = target - centre_n
    picked = rng.choice(outer, size=n_out, replace=False) if n_out > 0 else np.array([], dtype=int)
    mask = np.zeros(n_lines, dtype=bool)
    mask[centre] = True
    mask[picked] = True
    weights = mask.astype(np.float64)
    # order of arrival: centre first, then outward by distance from the centre line
    sampled = np.flatnonzero(mask)
    order = sampled[np.argsort(np.abs(sampled - mid), kind="stable")]
    return {"mask": mask, "weights": weights, "order": order, "n_sampled": int(mask.sum())}


def kspace_window(shape: Tuple[int, int], kind: str) -> np.ndarray:
    if kind == "none":
        return np.ones(shape)
    if kind == "hann":
        return np.outer(np.hanning(shape[0] + 2)[1:-1], np.hanning(shape[1] + 2)[1:-1])
    raise ValueError(f"Unknown k-space filter '{kind}'.")


class _MRILive(LiveSession):
    unit = "k-space lines"

    def __init__(self, modality: "MRIModality", truth, protocol, seed, ideal):
        self.m, self.protocol = modality, protocol
        ideal_k, noisy_k, meta = modality.acquire(truth, protocol, seed)
        self.k = ideal_k if ideal else noisy_k
        self.meta = meta
        self.plan = meta["plan"]
        self.order = self.plan["order"]
        self.total = len(self.order)
        self.window = kspace_window(truth.shape, protocol["kspace_filter"])

    def state_at(self, n: int) -> Dict[str, Any]:
        rows = self.order[:n]
        kk = np.zeros_like(self.k)
        kk[rows, :] = self.k[rows, :]
        w = np.zeros(self.k.shape[0])
        w[rows] = self.plan["weights"][rows]
        recon = _image_from_k(kk * w[:, None] * self.window)
        view = np.full(self.k.shape, np.nan)
        view[rows, :] = np.log10(1.0 + np.abs(self.k[rows, :]))
        frac = n / self.k.shape[0]
        cost = frac * self.protocol["nex"] / MRI_NEX_REF
        return {"view": view, "recon": recon, "exposure_pct": 100.0 * cost, "cost": cost}


class MRIModality(Modality):
    name = "MRI"
    icon = "🧲"
    uses_circular_fov = False
    description = "Cartesian k-space MRI: acceleration factor, number of excitations (NEX), k-space filter."
    model_scope_text = (
        "Simplified Cartesian MRI: the image is the magnitude of transverse magnetization, Fourier-transformed to "
        "k-space, undersampled along the phase-encode direction (fully sampled centre + random outer lines), "
        "corrupted by complex Gaussian noise that shrinks as 1/√NEX, and reconstructed by zero-filled inverse FFT "
        "magnitude (Rician-type noise) with optional Hann k-space window. No coil sensitivities, parallel imaging, "
        "compressed sensing, T1/T2 contrast physics, motion or field effects. The cost is a scan-time proxy, not a "
        "real sequence time. Not a validated scanner simulation."
    )
    cost_labels = {
        "exposure": "Relative scan-time proxy",
        "exposure_short": "Scan-time proxy",
        "cost": "Normalized acquisition cost (scan-time proxy)",
        "measurement": "k-space (log-magnitude)",
        "recon": "Magnitude image",
        "pipeline": ("Image magnitude ➔ FFT (k-space) ➔ Phase-encode undersampling ➔ Complex Gaussian noise (σ/√NEX) ➔ "
                     "Sampled k-space ➔ Zero-filled inverse FFT magnitude ➔ Quality assessment"),
        "reduction": "Relative simulated scan-time reduction",
    }
    model_param_names = ("acceleration", "nex", "kspace_filter")
    default_criteria = {"min_ssim": 0.85, "min_psnr": 25.0, "min_cnr": 3.0}

    def protocol_space(self) -> List[ParamSpec]:
        return [
            ParamSpec("acceleration", "Acceleration factor R", "ordered", (1, 2, 3, 4, 6, 8), 2, short="R",
                      unit="x fewer phase-encode lines", numeric=True, fmt="{}",
                      help="Keep about 1/R of the phase-encode lines (centre always fully sampled). Larger R = faster scan, more aliasing."),
            ParamSpec("nex", "Excitations / averages (NEX)", "ordered", (1, 2, 4), 2, short="NEX", unit="averages",
                      numeric=True,
                      help="Signal averages. Noise standard deviation falls as 1/√NEX; scan time rises linearly."),
            ParamSpec("kspace_filter", "k-space filter", "categorical", ("none", "hann"), "none", short="filter",
                      help="Hann window on k-space: lowers noise and Gibbs ringing, but blurs."),
        ]

    def reference_protocol(self) -> Dict[str, Any]:
        return {"acceleration": 1, "nex": MRI_NEX_REF, "kspace_filter": "none"}

    def training_grid(self) -> List[Dict[str, Any]]:
        return [{"acceleration": r, "nex": n, "kspace_filter": f}
                for r in (1, 2, 3, 4, 6, 8) for n in (1, 2, 4) for f in ("none", "hann")]

    def suggest_protocol(self, trial, include_alt_recon: bool = False) -> Dict[str, Any]:
        return {
            "acceleration": trial.suggest_categorical("acceleration", [1, 2, 3, 4, 6, 8]),
            "nex": trial.suggest_categorical("nex", [1, 2, 4]),
            "kspace_filter": trial.suggest_categorical("kspace_filter", ["none", "hann"]),
        }

    def sweep_axes(self) -> Dict[str, Dict[str, Any]]:
        return {
            "primary": {"param": "acceleration", "label": "Acceleration factor R", "values": [1, 2, 3, 4, 6, 8], "x_log": False},
            "secondary": {"param": "nex", "label": "NEX (averages)", "values": [1, 2, 4], "x_log": False},
        }

    def recon_variants(self, protocol: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
        return [("No k-space filter", {**protocol, "kspace_filter": "none"}),
                ("Hann k-space window", {**protocol, "kspace_filter": "hann"})]

    # ------------------------------------------------------------------ physics
    def acquisition_key(self, protocol):
        return (int(protocol["acceleration"]), int(protocol["nex"]))

    def acquire(self, image: np.ndarray, protocol: Dict[str, Any], seed: Optional[int] = 42):
        rng = np.random.default_rng(seed)
        n_lines = image.shape[0]
        plan = sampling_plan(n_lines, int(protocol["acceleration"]), rng)
        k = _kspace(image)
        sigma = MRI_SIGMA0 / np.sqrt(max(1, int(protocol["nex"])))
        noise = sigma * (rng.standard_normal(k.shape) + 1j * rng.standard_normal(k.shape))
        m = plan["mask"][:, None]
        ideal = np.where(m, k, 0.0)
        noisy = np.where(m, k + noise, 0.0)
        cost = self.acquisition_cost(protocol)
        meta = {"plan": plan, "sigma": float(sigma), "seed": seed, "sampled_lines": plan["n_sampled"],
                "sampled_line_fraction": plan["n_sampled"] / n_lines, "relative_scan_time_pct": cost.exposure_pct,
                "normalized_acquisition_cost": cost.cost}
        return ideal, noisy, meta

    def reconstruct(self, measurement, protocol: Dict[str, Any], meta: Dict[str, Any]):
        t0 = time.perf_counter()
        window = kspace_window(measurement.shape, protocol.get("kspace_filter", "none"))
        w = meta["plan"]["weights"][:, None]
        img = _image_from_k(measurement * w * window)
        return img, {"elapsed_time_ms": (time.perf_counter() - t0) * 1000.0}

    def acquisition_cost(self, protocol: Dict[str, Any]) -> CostInfo:
        lines = 1.0 / max(1, int(protocol["acceleration"]))
        nex = int(protocol["nex"]) / MRI_NEX_REF
        cost = lines * nex
        return CostInfo(cost, 100.0 * cost, {"sampled_line_fraction": lines, "nex_fraction": nex})

    def measurement_view(self, measurement, meta, protocol) -> MeasurementView:
        mag = np.log10(1.0 + np.abs(measurement))
        mag = np.where(meta["plan"]["mask"][:, None], mag, np.nan)
        return MeasurementView(mag, "k-space (log₁₀(1+|k|))", "Frequency-encode index kx", "Phase-encode line ky", "magma",
                               note=f"{meta['plan']['n_sampled']} of {measurement.shape[0]} phase-encode lines sampled (dark = skipped)")

    # ------------------------------------------------------------------ live
    def live_session(self, truth, protocol, seed, ideal, steps) -> LiveSession:
        return _MRILive(self, truth, protocol, seed, ideal)

    def default_live_steps(self, protocol, image_shape, n_frames) -> List[int]:
        from modalities.base import geometric_steps

        n_lines = image_shape[0]
        total = int(np.clip(round(n_lines / max(1, int(protocol["acceleration"]))), 1, n_lines))
        return geometric_steps(max(2, total // 16), total, n_frames)
