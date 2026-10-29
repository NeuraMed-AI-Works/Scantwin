"""
Digital Twin Simulation Core
============================
Modality-agnostic forward pipeline:

    ground truth -> modality.acquire (ideal / noisy measurement) -> modality.reconstruct
                 -> quality metrics (+ synthetic-target CNR) -> feasibility -> TwinState

``is_protocol_feasible`` is THE single authoritative feasibility function; nothing else in the
code base re-implements that logic.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from config import DEFAULT_MIN_CNR, DEFAULT_MIN_SSIM, AcquisitionProtocol
from digital_twin.state import TwinState
from metrics.cnr import compute_cnr, compute_contrast_recovery
from metrics.image_quality import evaluate_image_quality
from modalities import Modality, get_modality
from modalities.base import finite_or_raise


def protocol_to_dict(protocol: Any) -> Dict[str, Any]:
    """Accept a plain dict or the legacy CT ``AcquisitionProtocol`` dataclass."""
    if isinstance(protocol, dict):
        return dict(protocol)
    if isinstance(protocol, AcquisitionProtocol):
        return {
            "photon_budget": protocol.photon_budget,
            "num_projections": protocol.num_projections,
            "reconstruction_method": protocol.reconstruction_method,
            "fbp_filter": protocol.fbp_filter,
            "sart_iterations": protocol.sart_iterations,
            "seed": protocol.seed,
        }
    raise TypeError(f"Unsupported protocol type {type(protocol)!r}")


def is_protocol_feasible(metrics: Dict[str, Any], constraints: Dict[str, Any]) -> bool:
    """
    Authoritative single-source-of-truth feasibility evaluation.

    Parameters
    ----------
    metrics : dict
        Measured metrics. Must contain ``ssim``; ``psnr`` is needed only if ``min_psnr`` is set; ``cnr`` is
        used only when it is present AND finite (``None``/NaN = no synthetic target, CNR criterion skipped).
    constraints : dict
        ``min_ssim`` (default 0.88), optional ``min_cnr`` (default 3.0; ``None`` disables) and optional
        ``min_psnr`` (``None``/absent disables). These are user-set experimental criteria, not clinical thresholds.
    """
    def _num(v):
        return v is not None and np.isfinite(v)

    min_ssim = constraints.get("min_ssim", DEFAULT_MIN_SSIM)
    min_cnr = constraints.get("min_cnr", DEFAULT_MIN_CNR)
    min_psnr = constraints.get("min_psnr", None)

    ssim_v = metrics.get("ssim", None)
    ssim_ok = _num(ssim_v) and ssim_v >= min_ssim

    cnr_v = metrics.get("cnr", None)
    cnr_ok = True if (cnr_v is None or not _num(cnr_v) or min_cnr is None) else cnr_v >= min_cnr

    if min_psnr is None:
        psnr_ok = True
    else:
        psnr_v = metrics.get("psnr", None)
        psnr_ok = _num(psnr_v) and psnr_v >= min_psnr

    return bool(ssim_ok and cnr_ok and psnr_ok)


class DigitalTwin:
    """Virtual twin of one imaging chain (acquisition + reconstruction) and one known object."""

    def __init__(
        self,
        modality: Modality,
        ground_truth: np.ndarray,
        lesion_mask: Optional[np.ndarray] = None,
        background_mask: Optional[np.ndarray] = None,
        lesion_metadata: Optional[Dict[str, Any]] = None,
    ):
        self.modality = modality
        self.ground_truth = np.asarray(ground_truth, dtype=np.float64)
        self.lesion_mask = lesion_mask
        self.background_mask = background_mask
        self.lesion_metadata = lesion_metadata or {}
        self.height, self.width = self.ground_truth.shape

    @property
    def has_target(self) -> bool:
        return self.lesion_mask is not None and self.background_mask is not None

    def evaluate_protocol(
        self,
        protocol: Any,
        target_ssim_min: float = DEFAULT_MIN_SSIM,
        target_cnr_min: Optional[float] = DEFAULT_MIN_CNR,
        n_realizations: int = 1,
        target_psnr_min: Optional[float] = None,
        seed: Optional[int] = None,
        ideal: bool = False,
    ) -> Tuple[TwinState, np.ndarray, Any]:
        """
        Run acquisition + reconstruction ``n_realizations`` times (seeds ``seed, seed+1, ...``) and average the metrics.

        Returns ``(state, last_reconstruction, last_measurement)``.
        """
        proto = self.modality.normalize_protocol(protocol_to_dict(protocol))
        proto_seed = proto.pop("seed", None)
        base_seed = int(seed if seed is not None else (proto_seed if proto_seed is not None else 42))

        mses, rmses, psnrs, ssims, cnrs, crcs, times = [], [], [], [], [], [], []
        last_recon, last_meas = None, None
        metric_ref = self.modality.metric_reference(self.ground_truth, proto)

        for i in range(max(1, int(n_realizations))):
            ideal_m, noisy_m, meta = self.modality.acquire(self.ground_truth, proto, base_seed + i)
            meas = ideal_m if ideal else noisy_m
            recon, rmeta = self.modality.reconstruct(meas, proto, meta)
            if not np.all(np.isfinite(recon)):
                raise ValueError("Reconstruction contains non-finite values.")
            q = evaluate_image_quality(metric_ref, recon)
            mses.append(finite_or_raise(q["mse"], "MSE"))
            rmses.append(finite_or_raise(q["rmse"], "RMSE"))
            psnrs.append(finite_or_raise(q["psnr"], "PSNR"))
            ssims.append(finite_or_raise(q["ssim"], "SSIM"))
            if self.has_target:
                cnrs.append(finite_or_raise(compute_cnr(recon, self.lesion_mask, self.background_mask), "CNR"))
                crcs.append(finite_or_raise(
                    compute_contrast_recovery(recon, metric_ref, self.lesion_mask, self.background_mask), "CRC"))
            times.append(float(rmeta.get("elapsed_time_ms", 0.0)))
            last_recon, last_meas = recon, meas

        n = max(1, int(n_realizations))
        mean_ssim, mean_psnr = float(np.mean(ssims)), float(np.mean(psnrs))
        mean_cnr = float(np.mean(cnrs)) if cnrs else None
        cost = self.modality.acquisition_cost(proto)

        constraints = {"min_ssim": target_ssim_min, "min_cnr": target_cnr_min, "min_psnr": target_psnr_min}
        metrics = {"ssim": mean_ssim, "psnr": mean_psnr, "cnr": mean_cnr}
        feasible = is_protocol_feasible(metrics, constraints)

        if feasible:
            label = "Feasible (experimental criteria satisfied)"
        else:
            viol = []
            if mean_ssim < target_ssim_min:
                viol.append(f"SSIM {mean_ssim:.3f} < {target_ssim_min:.3f}")
            if mean_cnr is not None and target_cnr_min is not None and mean_cnr < target_cnr_min:
                viol.append(f"CNR {mean_cnr:.2f} < {target_cnr_min:.2f}")
            if target_psnr_min is not None and mean_psnr < target_psnr_min:
                viol.append(f"PSNR {mean_psnr:.1f} < {target_psnr_min:.1f} dB")
            label = f"Infeasible ({', '.join(viol)})"

        state = TwinState(
            modality=self.modality.name,
            protocol=proto,
            seed=base_seed,
            image_size=self.height,
            lesion_present=self.has_target,
            lesion_radius=self.lesion_metadata.get("radius"),
            lesion_contrast=self.lesion_metadata.get("nominal_contrast"),
            lesion_center=self.lesion_metadata.get("center"),
            mse=float(np.mean(mses)), rmse=float(np.mean(rmses)), psnr=mean_psnr, ssim=mean_ssim,
            cnr=mean_cnr, contrast_recovery=float(np.mean(crcs)) if crcs else None,
            elapsed_reconstruction_ms=float(np.mean(times)),
            relative_photon_exposure_pct=cost.exposure_pct,
            normalized_acquisition_cost=cost.cost,
            cost_components=dict(cost.components),
            target_ssim_min=target_ssim_min, target_cnr_min=target_cnr_min, target_psnr_min=target_psnr_min,
            is_feasible=feasible, feasibility_status_label=label,
            metadata={
                "n_realizations": n,
                "ssim_std": float(np.std(ssims)) if n > 1 else 0.0,
                "cnr_std": float(np.std(cnrs)) if (n > 1 and cnrs) else 0.0,
                "psnr_std": float(np.std(psnrs)) if n > 1 else 0.0,
                "rmse_std": float(np.std(rmses)) if n > 1 else 0.0,
            },
        )
        return state, last_recon, last_meas


class CTDigitalTwin(DigitalTwin):
    """Backwards-compatible CT twin (``CTDigitalTwin(ground_truth, lesion_mask, background_mask, lesion_metadata)``)."""

    def __init__(self, ground_truth, lesion_mask=None, background_mask=None, lesion_metadata=None):
        super().__init__(get_modality("CT"), ground_truth, lesion_mask, background_mask, lesion_metadata)


def evaluate_protocol(
    twin: DigitalTwin,
    protocol: Any,
    constraints: Optional[Dict[str, Any]] = None,
    n_realizations: int = 1,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Modality-independent structured evaluation. Returns
    ``ssim, psnr, rmse, cnr (None without a target), relative_exposure, cost, feasible`` (+ extras).
    """
    constraints = constraints or {}
    state, _, _ = twin.evaluate_protocol(
        protocol,
        target_ssim_min=constraints.get("min_ssim", DEFAULT_MIN_SSIM),
        target_cnr_min=constraints.get("min_cnr", DEFAULT_MIN_CNR),
        target_psnr_min=constraints.get("min_psnr"),
        n_realizations=n_realizations,
        seed=seed,
    )
    out: Dict[str, Any] = {
        "ssim": state.ssim, "psnr": state.psnr, "rmse": state.rmse, "cnr": state.cnr,
        "relative_exposure": state.relative_photon_exposure_pct,
        "cost": state.normalized_acquisition_cost,
        "feasible": state.is_feasible,
    }
    if state.projection_fraction_pct is not None:
        out["projection_fraction"] = state.projection_fraction_pct
    return out


def evaluate_protocol_dict(
    twin: DigitalTwin,
    photon_budget: float = 1e5,
    projection_count: int = 180,
    reconstruction_method: str = "FBP",
    filter_name: str = "ramp",
    sart_iterations: int = 1,
    seed: int = 42,
    target_ssim_min: float = DEFAULT_MIN_SSIM,
    target_cnr_min: float = DEFAULT_MIN_CNR,
    n_realizations: int = 1,
) -> Dict[str, Any]:
    """Legacy CT-style signature kept for backwards compatibility; delegates to :func:`evaluate_protocol`."""
    proto = {
        "photon_budget": float(photon_budget), "num_projections": int(projection_count),
        "reconstruction_method": reconstruction_method, "fbp_filter": filter_name,
        "sart_iterations": sart_iterations, "seed": seed,
    }
    return evaluate_protocol(twin, proto, {"min_ssim": target_ssim_min, "min_cnr": target_cnr_min}, n_realizations)
