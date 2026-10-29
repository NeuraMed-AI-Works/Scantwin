"""Markdown report of a Train & Analyze session (every number comes from the stored results)."""

from __future__ import annotations

from typing import Any, List, Sequence

import pandas as pd

from config import NO_FEASIBLE_MESSAGE, RESEARCH_DISCLAIMER
from learning.protocol_model import ProtocolModel, Recommendation
from modalities import get_modality


def results_table(items: Sequence[tuple]) -> pd.DataFrame:
    """items: (image_name, Recommendation) -> flat CSV-ready table."""
    rows = []
    for name, rec in items:
        m = get_modality(rec.modality)
        row: dict = {"image": name, "modality": rec.modality, "status": rec.status,
                     "criteria_min_ssim": rec.criteria.min_ssim, "criteria_min_psnr": rec.criteria.min_psnr,
                     "criteria_min_cnr": rec.criteria.min_cnr, "novelty_z": rec.novelty_z,
                     "reference_params": m.describe_protocol(rec.reference["protocol"]),
                     "reference_ssim": rec.reference["ssim"], "reference_psnr": rec.reference["psnr"],
                     "reference_rmse": rec.reference["rmse"], "reference_cnr": rec.reference["cnr"],
                     "reference_cost": rec.reference["cost"]}
        if rec.chosen:
            c = rec.chosen
            row.update({"recommended_params": m.describe_protocol(c["protocol"]), "recommended_ssim": c["ssim"],
                        "recommended_psnr": c["psnr"], "recommended_rmse": c["rmse"], "recommended_cnr": c["cnr"],
                        "recommended_cost": c["cost"], "predicted_ssim": c["pred_ssim"],
                        "cost_reduction_pct": rec.cost_reduction_pct})
        rows.append(row)
    return pd.DataFrame(rows)


def build_markdown_report(items: Sequence[tuple], model: ProtocolModel) -> str:
    def f(v: Any, s: str = ".3f") -> str:
        return "n/a" if v is None or v != v else format(v, s)

    lines: List[str] = ["# ScanTwin analysis report", "", f"> {RESEARCH_DISCLAIMER}", "",
                        f"Model: `{model.modality}` | training images: {model.n_images} | "
                        f"CV SSIM MAE {f(model.cv['ssim_mae'])} (mean baseline {f(model.cv['ssim_mae_baseline'])}) | "
                        f"beats baseline: {'yes' if model.cv['beats_baseline'] else 'no'}", ""]
    for w in model.warnings:
        lines.append(f"- WARNING: {w}")
    for name, rec in items:
        m = get_modality(rec.modality)
        lines += ["", f"## {name} ({rec.modality})", ""]
        if rec.novelty_z > 3:
            lines.append(f"- Image differs from the training set (max z = {rec.novelty_z:.1f}); prediction less reliable.")
        r = rec.reference
        lines.append(f"- Reference: {m.describe_protocol(r['protocol'])} - SSIM {f(r['ssim'])}, PSNR {f(r['psnr'], '.2f')} dB, "
                     f"CNR {f(r['cnr'], '.2f')}, cost {f(r['cost'])}")
        if rec.chosen:
            c = rec.chosen
            lines.append(f"- Recommended (simulation-confirmed): {m.describe_protocol(c['protocol'])} - SSIM {f(c['ssim'])}, "
                         f"PSNR {f(c['psnr'], '.2f')} dB, CNR {f(c['cnr'], '.2f')}, cost {f(c['cost'])}, "
                         f"relative simulated cost reduction {f(rec.cost_reduction_pct, '.1f')}%")
        else:
            lines.append(f"- {NO_FEASIBLE_MESSAGE}")
    lines += ["", "_Criteria are user-set experimental criteria, not clinical thresholds. Costs are relative proxies, not dose._"]
    return "\n".join(lines)
