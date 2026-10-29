"""
Closed-Loop Digital Twin Controller
===================================
Coordinates the digital twin and the optimizer: reference baseline, history of evaluated
protocols, best feasible protocol, and the Reference / Optimized / Naive comparison table.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from config import DEFAULT_MIN_CNR, DEFAULT_MIN_SSIM
from digital_twin.state import TwinState
from digital_twin.twin import DigitalTwin


def fmt_opt(value: Optional[float], spec: str = ".2f") -> str:
    """Format a number, or 'n/a' for None / non-finite values (nothing invented, nothing NaN in the UI)."""
    try:
        v = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "n/a"
    if v != v or v in (float("inf"), float("-inf")):
        return "n/a"
    return format(v, spec)


class DigitalTwinController:
    """Closed-loop controller orchestrating evaluation, history and comparison."""

    def __init__(
        self,
        twin: DigitalTwin,
        target_ssim_min: float = DEFAULT_MIN_SSIM,
        target_cnr_min: Optional[float] = DEFAULT_MIN_CNR,
        target_psnr_min: Optional[float] = None,
    ):
        self.twin = twin
        self.modality = twin.modality
        self.target_ssim_min = target_ssim_min
        self.target_cnr_min = target_cnr_min
        self.target_psnr_min = target_psnr_min
        self.history: List[TwinState] = []
        self.reference_state: Optional[TwinState] = None

    def evaluate_reference_baseline(self) -> TwinState:
        """Evaluate the modality's reference protocol (cost = 1.0) with the current criteria."""
        state, _, _ = self.twin.evaluate_protocol(
            self.modality.reference_protocol(),
            target_ssim_min=self.target_ssim_min,
            target_cnr_min=self.target_cnr_min,
            target_psnr_min=self.target_psnr_min,
            n_realizations=1,
            seed=42,
        )
        self.reference_state = state
        return state

    def record_state(self, state: TwinState) -> None:
        self.history.append(state)

    def get_history_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([s.to_flat_dict() for s in self.history])

    def get_best_feasible_protocol(self) -> Optional[TwinState]:
        """Lowest normalized cost among states that satisfy all criteria (None if none do)."""
        feasible = [s for s in self.history if s.is_feasible]
        if not feasible:
            return None
        return min(feasible, key=lambda s: s.normalized_acquisition_cost)

    def _row(self, label: str, st: TwinState) -> Dict[str, Any]:
        labels = self.modality.cost_labels
        return {
            "Protocol": label,
            "Parameters": self.modality.describe_protocol(st.protocol),
            "SSIM": fmt_opt(st.ssim, ".4f"),
            "CNR": fmt_opt(st.cnr, ".2f"),
            "PSNR (dB)": fmt_opt(st.psnr, ".2f"),
            f"{labels['exposure_short']} (%)": f"{st.relative_photon_exposure_pct:.1f}%",
            "Acquisition Cost": f"{st.normalized_acquisition_cost:.3f}",
            "Feasible?": "Yes" if st.is_feasible else "No",
        }

    def compare_protocols(self, optimized_state: Optional[TwinState],
                          naive_low_state: Optional[TwinState] = None) -> pd.DataFrame:
        """Reference, optimized (if found) and naive low-cost protocols side by side."""
        if self.reference_state is None:
            self.evaluate_reference_baseline()
        rows = [self._row("Reference Baseline", self.reference_state)]
        if optimized_state is not None:
            rows.append(self._row("AI-Optimized", optimized_state))
        if naive_low_state is not None:
            rows.append(self._row("Naive Low-Cost", naive_low_state))
        return pd.DataFrame(rows)
