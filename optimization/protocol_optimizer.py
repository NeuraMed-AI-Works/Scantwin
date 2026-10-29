"""
Intelligent Acquisition Protocol Optimizer
==========================================
Bayesian optimization (Tree-structured Parzen Estimator via Optuna) of the minimal-cost acquisition
protocol of ANY modality that satisfies user-defined experimental image-quality criteria. A result is
reported only if a simulation on the digital twin confirms it with ``is_protocol_feasible``.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import pandas as pd

from config import DEFAULT_MIN_CNR, DEFAULT_MIN_SSIM, NO_FEASIBLE_MESSAGE
from digital_twin.controller import DigitalTwinController
from digital_twin.state import TwinState
from digital_twin.twin import DigitalTwin
from optimization.objective import create_optuna_objective


def _optuna():
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("The protocol optimizer needs 'optuna': pip install optuna") from exc
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return optuna


class ProtocolOptimizer:
    """Searches the modality's protocol space for the cheapest protocol meeting the criteria."""

    def __init__(
        self,
        twin: DigitalTwin,
        target_ssim_min: float = DEFAULT_MIN_SSIM,
        target_cnr_min: Optional[float] = DEFAULT_MIN_CNR,
        seed: int = 42,
        target_psnr_min: Optional[float] = None,
    ):
        self.twin = twin
        self.modality = twin.modality
        self.target_ssim_min = target_ssim_min
        self.target_cnr_min = target_cnr_min
        self.target_psnr_min = target_psnr_min
        self.seed = seed
        self.controller = DigitalTwinController(twin, target_ssim_min, target_cnr_min, target_psnr_min)
        self.study = None

    def _criteria_text(self) -> str:
        parts = [f"SSIM >= {self.target_ssim_min:.2f}"]
        if self.target_psnr_min is not None:
            parts.append(f"PSNR >= {self.target_psnr_min:.1f} dB")
        if self.twin.has_target and self.target_cnr_min is not None:
            parts.append(f"CNR >= {self.target_cnr_min:.2f}")
        return " and ".join(parts)

    def optimize(
        self,
        n_trials: int = 25,
        n_realizations: int = 1,
        include_sart: bool = False,
        progress_callback: Optional[Callable[[int, int, Optional[TwinState]], None]] = None,
    ) -> Tuple[Optional[TwinState], pd.DataFrame, str]:
        """
        Run the constrained search. Returns ``(best_feasible_state | None, history_dataframe, summary_message)``.
        ``include_sart`` enables the modality's alternative reconstruction (SART for CT; ignored elsewhere).
        """
        optuna = _optuna()
        self.controller.evaluate_reference_baseline()
        self.study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=self.seed))

        objective_fn = create_optuna_objective(
            twin=self.twin, controller=self.controller,
            target_ssim_min=self.target_ssim_min, target_cnr_min=self.target_cnr_min,
            target_psnr_min=self.target_psnr_min,
            n_realizations=n_realizations, include_sart=include_sart,
        )

        def trial_callback(study, trial):
            if progress_callback is not None:
                progress_callback(trial.number + 1, n_trials, self.controller.get_best_feasible_protocol())

        self.study.optimize(objective_fn, n_trials=n_trials, callbacks=[trial_callback] if progress_callback else None)

        best = self.controller.get_best_feasible_protocol()
        history = self.controller.get_history_dataframe()
        if best is None:
            msg = (f"{NO_FEASIBLE_MESSAGE} None of the {n_trials} tested configurations satisfied {self._criteria_text()}.")
        else:
            label = self.modality.cost_labels["reduction"]
            ref_cost = self.controller.reference_state.normalized_acquisition_cost
            red = 100.0 * (1.0 - best.normalized_acquisition_cost / ref_cost)
            msg = (f"Feasible protocol found ({self.modality.describe_protocol(best.protocol)}). {label}: {red:.1f}% "
                   f"(normalized cost {best.normalized_acquisition_cost:.3f} vs reference {ref_cost:.3f}) while "
                   f"satisfying {self._criteria_text()}: SSIM = {best.ssim:.4f}"
                   + (f", CNR = {best.cnr:.2f}" if best.cnr is not None else "") + ". "
                   "Simulation-confirmed on the digital twin; not a clinical protocol.")
        return best, history, msg
