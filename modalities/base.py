"""
Modality abstraction
====================
Every imaging modality in ScanTwin implements :class:`Modality`. The digital twin, evaluator,
optimizer, experiments, learning code and UI only talk to this interface, so the same
"simulate acquisition -> reconstruct -> measure quality -> search cheaper settings" loop works for
CT, X-ray, MRI, ultrasound and the generic "Other" proxy.

Conventions shared by all modalities
------------------------------------
* A *protocol* is a plain ``dict`` ``{param_name: value}`` (see :meth:`Modality.protocol_space`).
* Images are 2D float arrays in [0, 1] ("intensity proxy"), already resampled/normalized by
  ``simulation.datasets``. Ground truth and reconstruction live on the SAME scale; metrics are
  computed directly on that scale (data range = ground-truth max - min).
* ``acquire`` returns ``(ideal_measurement, noisy_measurement, meta)``; identical seed and
  parameters give identical output, a different seed gives a different noise realization.
* ``acquisition_cost`` is normalized so the modality's reference protocol costs exactly 1.0.
* Nothing here is a validated scanner model: each modality is a simplified image-formation model
  (see ``model_scope_text``). Costs are *proxies*, never patient dose.

DISCLAIMER: research/educational prototype; not for clinical use.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------------------
# Small data classes
# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ParamSpec:
    """One controllable acquisition/reconstruction parameter."""

    name: str
    label: str
    kind: str                         # "ordered" (select-slider), "categorical" (selectbox), "int" (slider)
    options: Tuple[Any, ...] = ()     # allowed values for "ordered" / "categorical"
    default: Any = None
    help: str = ""
    short: str = ""                   # compact name for tables, e.g. "I0"
    unit: str = ""                    # unit or proxy note shown next to the label
    numeric: bool = False             # True -> numeric ML feature; False -> one-hot encoded
    log: bool = False                 # log10-transform when used as a numeric ML feature
    fmt: str = "{}"                   # display format for values
    min: Optional[float] = None       # "int" kind only
    max: Optional[float] = None
    step: Optional[float] = None
    when: Optional[Tuple[str, Any]] = None   # active only if protocol[when[0]] == when[1] (or is in it, if a tuple/list)

    def format(self, value: Any) -> str:
        try:
            return self.fmt.format(value)
        except (ValueError, TypeError):
            return str(value)

    def active(self, protocol: Dict[str, Any]) -> bool:
        if self.when is None:
            return True
        want = self.when[1]
        have = protocol.get(self.when[0])
        return have in want if isinstance(want, (tuple, list, set)) else have == want


@dataclass
class CostInfo:
    """Normalized acquisition cost (reference = 1.0) and its named components."""

    cost: float
    exposure_pct: float               # the modality's headline exposure/time proxy, % of reference
    components: Dict[str, float] = field(default_factory=dict)


@dataclass
class MeasurementView:
    """How to draw the raw measurement (sinogram, k-space, radiograph, B-mode frame)."""

    image: np.ndarray                 # 2D real array; NaN marks data not acquired yet
    title: str
    xlabel: str = ""
    ylabel: str = ""
    cmap: str = "viridis"
    note: str = ""


class LiveSession(ABC):
    """A scan prepared once (one seed) whose data is *revealed* progressively."""

    unit: str = "units"               # what accumulates: "views", "k-space lines", "frames", "% of photon budget"
    total: int = 1

    @abstractmethod
    def state_at(self, n: int) -> Dict[str, Any]:
        """Return ``{"view": 2D array (NaN = unacquired), "recon": 2D array, "exposure_pct": float, "cost": float}``
        for the data collected after ``n`` units."""


# ----------------------------------------------------------------------------------------
# Modality base class
# ----------------------------------------------------------------------------------------
class Modality(ABC):
    """Interface implemented by CT, X-ray, MRI, ultrasound and the generic proxy."""

    name: str = "?"
    description: str = ""
    model_scope_text: str = ""
    uses_circular_fov: bool = False
    supports_lesion_cnr: bool = True
    icon: str = "🖼️"

    #: wording used by the UI (never hardcode "CT" in the UI; read these instead)
    cost_labels: Dict[str, str] = {
        "exposure": "Relative exposure proxy",
        "exposure_short": "Rel. exposure",
        "cost": "Normalized acquisition cost",
        "measurement": "Raw measurement",
        "recon": "Reconstruction",
        "pipeline": "Ground truth ➔ Acquisition ➔ Measurement ➔ Noise ➔ Reconstruction ➔ Quality assessment",
        "reduction": "Relative simulated cost reduction",
    }

    #: default experimental criteria shown in the UI (arbitrary starting values, user-adjustable, never clinical)
    default_criteria: Dict[str, float] = {"min_ssim": 0.88, "min_psnr": 28.0, "min_cnr": 3.0}

    #: protocol parameters used as features by the learned surrogate (must vary in the training grid)
    model_param_names: Tuple[str, ...] = ()

    # ------------------------------------------------------------------ protocol description
    @abstractmethod
    def protocol_space(self) -> List[ParamSpec]:
        """Controllable parameters with labels, units/proxies, bounds, defaults and tooltips."""

    @abstractmethod
    def reference_protocol(self) -> Dict[str, Any]:
        """High-quality baseline protocol; its normalized cost is exactly 1.0."""

    @abstractmethod
    def training_grid(self) -> List[Dict[str, Any]]:
        """Small list of protocols simulated per training image (<= ~60)."""

    def search_grid(self) -> List[Dict[str, Any]]:
        """Candidate protocols the recommender may choose from (defaults to the training grid)."""
        return self.training_grid()

    @abstractmethod
    def suggest_protocol(self, trial: Any, include_alt_recon: bool = False) -> Dict[str, Any]:
        """Sample a protocol from an Optuna trial (constrained, modality search space)."""

    @abstractmethod
    def sweep_axes(self) -> Dict[str, Dict[str, Any]]:
        """``{"primary": {...}, "secondary": {...}}``; each ``{"param", "label", "values", "x_log"}``.
        ``primary`` is the cost axis swept in the Quality-vs-Cost lab."""

    @abstractmethod
    def recon_variants(self, protocol: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
        """Reconstruction-side alternatives on the same acquisition: ``[(label, protocol), ...]``."""

    # ------------------------------------------------------------------ physics
    @abstractmethod
    def acquire(self, image: np.ndarray, protocol: Dict[str, Any], seed: Optional[int] = 42
                ) -> Tuple[Any, Any, Dict[str, Any]]:
        """``(ideal_measurement, noisy_measurement, meta)``."""

    @abstractmethod
    def reconstruct(self, measurement: Any, protocol: Dict[str, Any], meta: Dict[str, Any]
                    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """``(reconstruction, recon_meta)``; ``recon_meta['elapsed_time_ms']`` is always present."""

    @abstractmethod
    def acquisition_cost(self, protocol: Dict[str, Any]) -> CostInfo:
        """Normalized cost (reference = 1.0) with named components."""

    @abstractmethod
    def measurement_view(self, measurement: Any, meta: Dict[str, Any], protocol: Dict[str, Any]) -> MeasurementView:
        """Display form of the raw measurement."""

    def acquisition_key(self, protocol: Dict[str, Any]) -> Tuple:
        """Parameters that change the *acquisition* (everything else only changes reconstruction).
        Used to reuse simulated measurements across protocols that share them."""
        return tuple(sorted((k, str(v)) for k, v in protocol.items()))

    def metric_reference(self, truth: np.ndarray, protocol: Dict[str, Any]) -> np.ndarray:
        """Ground truth in the scale the reconstruction is displayed in (identity by default)."""
        return truth

    # ------------------------------------------------------------------ live
    @abstractmethod
    def live_session(self, truth: np.ndarray, protocol: Dict[str, Any], seed: int, ideal: bool,
                     steps: Sequence[int]) -> LiveSession:
        """Prepare a whole scan once; data is revealed progressively by ``LiveSession.state_at``."""

    @abstractmethod
    def default_live_steps(self, protocol: Dict[str, Any], image_shape: Tuple[int, int], n_frames: int) -> List[int]:
        """Cumulative unit counts for the live stream; the last one equals ``LiveSession.total``."""

    # ------------------------------------------------------------------ helpers (shared)
    def fov_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        """Boolean mask of valid pixels (circle for projection-based modalities)."""
        if self.uses_circular_fov and shape[0] == shape[1]:
            from simulation.datasets import circular_fov_mask

            return circular_fov_mask(shape[0])
        return np.ones(shape, dtype=bool)

    def spec(self, name: str) -> ParamSpec:
        for s in self.protocol_space():
            if s.name == name:
                return s
        raise KeyError(name)

    def default_protocol(self) -> Dict[str, Any]:
        return {s.name: s.default for s in self.protocol_space()}

    def normalize_protocol(self, protocol: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Fill missing parameters with defaults and drop unknown keys other than ``seed``."""
        base = self.default_protocol()
        out = dict(base)
        for k, v in (protocol or {}).items():
            if k in base or k == "seed":
                out[k] = v
        return out

    def active_params(self, protocol: Dict[str, Any]) -> List[ParamSpec]:
        return [s for s in self.protocol_space() if s.active(protocol)]

    def describe_protocol(self, protocol: Dict[str, Any]) -> str:
        parts = [f"{s.short or s.name}={s.format(protocol[s.name])}" for s in self.active_params(protocol) if s.name in protocol]
        return ", ".join(parts)

    def protocol_key(self, protocol: Dict[str, Any]) -> str:
        return "|".join(f"{s.name}={protocol.get(s.name)}" for s in self.active_params(protocol))

    def reference_cost(self) -> CostInfo:
        return self.acquisition_cost(self.reference_protocol())

    def encode_protocols(self, df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
        """Numeric design columns for the surrogate from a table of protocols (one row each)."""
        cols: Dict[str, Any] = {}
        for name in self.model_param_names:
            sp = self.spec(name)
            col = df[name]
            if sp.numeric:
                vals = col.astype(float).to_numpy()
                cols[f"{prefix}{name}"] = np.log10(vals) if sp.log else vals
            else:
                for opt in sp.options:
                    cols[f"{prefix}{name}={opt}"] = (col.astype(str) == str(opt)).astype(float).to_numpy()
        return pd.DataFrame(cols, index=df.index)

    def info(self) -> Dict[str, Any]:
        return {"name": self.name, "description": self.description, "scope": self.model_scope_text}

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<Modality {self.name}>"


def finite_or_raise(value: float, label: str) -> float:
    """Explicit NaN/Inf guard: never let a non-finite number travel further."""
    v = float(value)
    if not np.isfinite(v):
        raise ValueError(f"Non-finite value in {label}.")
    return v


def geometric_steps(first: int, total: int, n_frames: int) -> List[int]:
    """Roughly geometric cumulative schedule ending exactly at ``total``."""
    first = max(1, min(first, total))
    steps = np.unique(np.round(np.geomspace(first, total, max(2, n_frames))).astype(int))
    steps = steps[steps >= 1]
    steps[-1] = total
    return [int(s) for s in steps]
