"""
Optimization Objective Formulation
==================================
Constrained objective for Optuna, for any modality.

    minimize   Cost(protocol)               (normalized acquisition cost, reference = 1.0)
    subject to SSIM >= SSIM_min,  [PSNR >= PSNR_min],  [CNR >= CNR_min if a synthetic target exists]

Penalty formulation for the gradient-free solver:
    Loss = Cost + (0 if feasible else PenaltyBase + lambda1*max(0, SSIM_min - SSIM)
                                      + lambda2*max(0, CNR_min - CNR) + lambda3*max(0, PSNR_min - PSNR)/10)
Feasibility itself always comes from ``digital_twin.twin.is_protocol_feasible`` (via the twin).
"""

from __future__ import annotations

from typing import Callable, Optional

from digital_twin.controller import DigitalTwinController
from digital_twin.twin import DigitalTwin


def create_optuna_objective(
    twin: DigitalTwin,
    controller: DigitalTwinController,
    target_ssim_min: float = 0.90,
    target_cnr_min: Optional[float] = 3.0,
    n_realizations: int = 1,
    include_sart: bool = True,
    lambda_ssim: float = 15.0,
    lambda_cnr: float = 2.0,
    penalty_base: float = 10.0,
    target_psnr_min: Optional[float] = None,
) -> Callable:
    """Factory returning ``objective(trial)``. ``include_sart`` enables the modality's alternative reconstruction
    (SART for CT); other modalities ignore it."""
    modality = twin.modality

    def objective(trial) -> float:
        protocol = modality.suggest_protocol(trial, include_alt_recon=include_sart)
        protocol["seed"] = trial.number + 100

        state, _, _ = twin.evaluate_protocol(
            protocol,
            target_ssim_min=target_ssim_min,
            target_cnr_min=target_cnr_min,
            target_psnr_min=target_psnr_min,
            n_realizations=n_realizations,
        )
        controller.record_state(state)

        trial.set_user_attr("ssim", state.ssim)
        trial.set_user_attr("cnr", state.cnr)
        trial.set_user_attr("psnr", state.psnr)
        trial.set_user_attr("is_feasible", state.is_feasible)
        trial.set_user_attr("normalized_cost", state.normalized_acquisition_cost)
        trial.set_user_attr("relative_exposure_pct", state.relative_photon_exposure_pct)

        base_cost = state.normalized_acquisition_cost
        if state.is_feasible:
            return base_cost

        penalty = penalty_base + lambda_ssim * max(0.0, target_ssim_min - state.ssim)
        if state.cnr is not None and target_cnr_min is not None:
            penalty += lambda_cnr * max(0.0, target_cnr_min - state.cnr)
        if target_psnr_min is not None:
            penalty += max(0.0, target_psnr_min - state.psnr) / 10.0
        return base_cost + penalty

    return objective
