"""
Modality-aware automated experiments (A-D)
==========================================
A  primary cost-axis sweep (photon budget / acceleration / frames ...), mean +- SD over noise realizations
B  secondary-axis sweep (views / NEX / filter strength ...)
C  reconstruction-side comparison on one shared measurement
D  combined primary x secondary grid (heatmap data)

Every function returns a DataFrame computed by real simulations. ``save_experiment`` writes CSV (+ optional PNG)
to ``results/<modality>/`` with a timestamp in the file name, so earlier runs are NEVER overwritten.
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from digital_twin.pipeline import compare_recon_variants
from digital_twin.twin import DigitalTwin
from modalities import Modality

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _stats(values: List[Optional[float]]) -> tuple:
    v = [x for x in values if x is not None]
    return (float(np.mean(v)), float(np.std(v))) if v else (None, None)


def _point(twin: DigitalTwin, protocol: Dict[str, Any], n_real: int, seed: int) -> Dict[str, Any]:
    ss, ps, rs, cs = [], [], [], []
    for r in range(n_real):
        st, _, _ = twin.evaluate_protocol(protocol, n_realizations=1, seed=seed + 101 * r, target_cnr_min=None)
        ss.append(st.ssim); ps.append(st.psnr); rs.append(st.rmse); cs.append(st.cnr)
    cost = twin.modality.acquisition_cost(twin.modality.normalize_protocol(protocol))
    row = {"cost": cost.cost, "relative_exposure_pct": cost.exposure_pct}
    for name, vals in (("ssim", ss), ("psnr", ps), ("rmse", rs), ("cnr", cs)):
        row[f"{name}_mean"], row[f"{name}_std"] = _stats(vals)
    return row


def sweep(twin: DigitalTwin, base: Dict[str, Any], param: str, values: Sequence[Any], n_real: int = 3, seed: int = 42,
          progress: Optional[Callable[[int, int], None]] = None) -> pd.DataFrame:
    rows = []
    for i, v in enumerate(values):
        proto = dict(base, **{param: v})
        rows.append({param: v, "modality": twin.modality.name, "n_realizations": n_real, **_point(twin, proto, n_real, seed)})
        if progress:
            progress(i + 1, len(values))
    return pd.DataFrame(rows)


def combined_grid(twin: DigitalTwin, base: Dict[str, Any], p1: str, v1: Sequence[Any], p2: str, v2: Sequence[Any],
                  n_real: int = 2, seed: int = 42, progress: Optional[Callable[[int, int], None]] = None) -> pd.DataFrame:
    rows, total, k = [], len(v1) * len(v2), 0
    for a in v1:
        for b in v2:
            proto = dict(base, **{p1: a, p2: b})
            rows.append({p1: a, p2: b, "modality": twin.modality.name, **_point(twin, proto, n_real, seed)})
            k += 1
            if progress:
                progress(k, total)
    return pd.DataFrame(rows)


def reconstruction_comparison(twin: DigitalTwin, protocol: Dict[str, Any], seed: int = 42, ideal: bool = False) -> pd.DataFrame:
    res = compare_recon_variants(twin.modality, twin.ground_truth, protocol, seed, ideal, twin.lesion_mask, twin.background_mask)
    return pd.DataFrame([{"variant": r["label"], "ssim": r["quality"]["ssim"], "psnr": r["quality"]["psnr"],
                          "rmse": r["quality"]["rmse"], "cnr": r["cnr"], "recon_ms": r["ms"]} for r in res])


def results_dir(modality_name: str) -> str:
    d = os.path.join(ROOT, "results", modality_name.lower().replace(" ", "-"))
    os.makedirs(d, exist_ok=True)
    return d


def save_experiment(modality_name: str, name: str, df: pd.DataFrame, fig=None) -> Dict[str, str]:
    """Write CSV (+PNG) under results/<modality>/ with a timestamp; never overwrites an earlier run."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    base = os.path.join(results_dir(modality_name), f"{name}_{stamp}")
    n = 1
    while os.path.exists(base + ".csv"):
        n += 1
        base = os.path.join(results_dir(modality_name), f"{name}_{stamp}_{n}")
    paths = {"csv": base + ".csv"}
    df.to_csv(paths["csv"], index=False)
    if fig is not None:
        fig.savefig(base + ".png", dpi=130, bbox_inches="tight")
        paths["png"] = base + ".png"
    return paths
