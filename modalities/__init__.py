"""
Imaging modalities
==================
``MODALITIES`` maps a display name to its class; ``get_modality(name)`` returns a (cached) instance.
"""

from __future__ import annotations

from typing import Dict, List, Type

from modalities.base import CostInfo, LiveSession, MeasurementView, Modality, ParamSpec
from modalities.ct import CTModality, OtherModality
from modalities.mri import MRIModality
from modalities.ultrasound import UltrasoundModality
from modalities.xray import XRayModality

MODALITIES: Dict[str, Type[Modality]] = {
    "CT": CTModality,
    "X-ray": XRayModality,
    "MRI": MRIModality,
    "Ultrasound": UltrasoundModality,
    "Other": OtherModality,
}

_INSTANCES: Dict[str, Modality] = {}


def modality_names() -> List[str]:
    return list(MODALITIES)


def get_modality(name: str) -> Modality:
    if name not in MODALITIES:
        raise KeyError(f"Unknown modality '{name}'. Choose from {modality_names()}.")
    if name not in _INSTANCES:
        _INSTANCES[name] = MODALITIES[name]()
    return _INSTANCES[name]


__all__ = [
    "CostInfo", "LiveSession", "MeasurementView", "Modality", "ParamSpec", "MODALITIES",
    "CTModality", "OtherModality", "XRayModality", "MRIModality", "UltrasoundModality",
    "get_modality", "modality_names",
]
