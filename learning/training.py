"""
Training orchestration: images (each tagged with a modality) -> per-modality models (+ pooled fallback)
=======================================================================================================
``train_models`` simulates every image with its own modality's twin, then trains one surrogate per modality that has
at least ``MIN_TRAIN_IMAGES`` images. If some modality has fewer images, a pooled model over all images (modality as a
feature) is trained as well and used for those modalities, with an explicit warning.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from learning.protocol_model import (
    MIN_TRAIN_IMAGES,
    POOLED_KEY,
    ProtocolModel,
    build_training_table,
    model_filename,
    save_model,
    train_protocol_model,
)
from modalities import get_modality
from simulation.datasets import SliceData, load_synthetic, synthetic_series


@dataclass
class TrainingResult:
    models: Dict[str, ProtocolModel] = field(default_factory=dict)      # key: modality name or "pooled"
    table: Optional[pd.DataFrame] = None
    warnings: List[str] = field(default_factory=list)
    saved_paths: Dict[str, str] = field(default_factory=dict)
    pooled_for: List[str] = field(default_factory=list)                 # modalities served only by the pooled model

    def model_for(self, modality: str) -> Optional[ProtocolModel]:
        if modality in self.models:
            return self.models[modality]
        pooled = self.models.get(POOLED_KEY)
        return pooled if pooled is not None and modality in pooled.covered_modalities else None


def train_models(
    images: Sequence[np.ndarray],
    names: Sequence[str],
    modalities: Sequence[str],
    seed: int = 0,
    n_estimators: int = 200,
    models_dir: Optional[str] = None,
    progress: Optional[Callable[[int, int, str, str, pd.DataFrame], None]] = None,
) -> TrainingResult:
    """Train from images; ``progress(done, total, image_name, modality, grid)`` fires after every image."""
    if not (len(images) == len(names) == len(modalities)):
        raise ValueError("images, names and modalities must have equal length")
    res = TrainingResult()
    total, done = len(images), 0
    frames = []
    for mod_name in dict.fromkeys(modalities):                      # keep order, unique
        idx = [i for i, m in enumerate(modalities) if m == mod_name]
        modality = get_modality(mod_name)

        def _cb(d, t, name, grid, _mod=mod_name, _base=done):
            if progress:
                progress(_base + d, total, name, _mod, grid)

        frames.append(build_training_table(modality, [images[i] for i in idx], [names[i] for i in idx], seed=seed, progress=_cb))
        done += len(idx)
    table = pd.concat(frames, ignore_index=True)
    res.table = table

    counts = table.groupby("modality")["image"].nunique().to_dict()
    thin = [m for m, c in counts.items() if c < MIN_TRAIN_IMAGES]
    for mod_name, c in counts.items():
        if c >= MIN_TRAIN_IMAGES:
            res.models[mod_name] = train_protocol_model(table, modality=mod_name, seed=seed, n_estimators=n_estimators)
    if thin:
        if table["image"].nunique() >= MIN_TRAIN_IMAGES:
            res.models[POOLED_KEY] = train_protocol_model(table, modality=None, seed=seed, n_estimators=n_estimators)
            res.pooled_for = thin
            res.warnings.append(
                f"Modalities with fewer than {MIN_TRAIN_IMAGES} images ({', '.join(thin)}) are served by a pooled model "
                "(modality used as a feature). Add more images of those modalities for a dedicated model.")
        else:
            res.warnings.append(f"Too few images in total to train a model for: {', '.join(thin)}.")
    if models_dir:
        os.makedirs(models_dir, exist_ok=True)
        for key, mdl in res.models.items():
            path = os.path.join(models_dir, model_filename(key))
            save_model(mdl, path)
            res.saved_paths[key] = path
    return res


def builtin_demo_images(modality_name: str, size: int = 128, n_slices: int = 8):
    """Always-available SYNTHETIC training set: a warped Shepp-Logan series plus the two analytic phantoms."""
    modality = get_modality(modality_name)
    circ = modality.uses_circular_fov
    series: List[SliceData] = synthetic_series(size, n_slices=n_slices, fov_circle=circ)
    series += [load_synthetic("Shepp-Logan", size, fov_circle=circ), load_synthetic("Concentric Disks", size, fov_circle=circ)]
    return [s.image for s in series], [s.name for s in series]
