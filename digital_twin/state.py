"""
Digital Twin State Module
=========================
Structured, serializable state of the digital twin after evaluating one protocol on one object:
the protocol, measured image-quality metrics, acquisition-cost proxies and feasibility against the
user's experimental criteria. Works for every modality; CT-style attribute names
(``photon_budget``, ``num_projections`` ...) remain available as read-only conveniences.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class TwinState:
    """State snapshot of the digital twin for one evaluated protocol."""

    modality: str = "CT"
    protocol: Dict[str, Any] = field(default_factory=dict)
    seed: Optional[int] = 42

    # object / target
    image_size: int = 256
    lesion_present: bool = False
    lesion_radius: Optional[float] = None
    lesion_contrast: Optional[float] = None
    lesion_center: Optional[tuple] = None

    # measured quality (CNR / contrast recovery are None when no synthetic target exists)
    mse: float = 0.0
    rmse: float = 0.0
    psnr: float = 0.0
    ssim: float = 0.0
    cnr: Optional[float] = None
    contrast_recovery: Optional[float] = None
    elapsed_reconstruction_ms: float = 0.0

    # cost proxies (reference protocol = 1.0)
    relative_photon_exposure_pct: float = 100.0      # the modality's headline exposure / time proxy, % of reference
    normalized_acquisition_cost: float = 1.0
    cost_components: Dict[str, float] = field(default_factory=dict)

    # criteria and verdict
    target_ssim_min: float = 0.90
    target_cnr_min: Optional[float] = 3.0
    target_psnr_min: Optional[float] = None
    is_feasible: bool = False
    feasibility_status_label: str = "Uninitialized"

    metadata: Dict[str, Any] = field(default_factory=dict)

    # ---- convenience -------------------------------------------------------------------
    @property
    def relative_exposure_pct(self) -> float:
        return self.relative_photon_exposure_pct

    def _p(self, key: str, default: Any = None) -> Any:
        return self.protocol.get(key, default)

    @property
    def photon_budget(self):            # CT / X-ray
        return self._p("photon_budget")

    @property
    def num_projections(self):          # CT
        return self._p("num_projections")

    @property
    def reconstruction_method(self):    # CT
        return self._p("reconstruction_method")

    @property
    def fbp_filter(self):               # CT
        return self._p("fbp_filter")

    @property
    def sart_iterations(self):          # CT
        return self._p("sart_iterations")

    @property
    def projection_fraction_pct(self):  # CT
        frac = self.cost_components.get("view_fraction")
        return None if frac is None else 100.0 * frac

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_flat_dict(self) -> Dict[str, Any]:
        """One flat record (protocol parameters become columns) for DataFrames / CSV."""
        d = self.to_dict()
        proto = d.pop("protocol")
        comps = d.pop("cost_components")
        meta = d.pop("metadata")
        flat = {**d, **proto}
        flat.update({f"cost_{k}": v for k, v in comps.items()})
        flat.update({k: v for k, v in meta.items() if isinstance(v, (int, float, str, bool))})
        return flat

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TwinState":
        return cls(**data)


CTTwinState = TwinState   # backwards-compatible name
