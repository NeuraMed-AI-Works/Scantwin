"""Generate docs/validation_report.md from REAL simulation runs (no hardcoded numbers).

Usage:  python scripts/generate_validation_report.py [--size 64] [--realizations 3]

For every modality it (1) checks the simulation is reproducible, (2) sweeps the primary cost
parameter and reports measured SSIM/PSNR/CNR mean +- std over noise realizations, (3) checks the
ideal (noise-free) pipeline. Every number in the output file comes from this run on the built-in
synthetic phantom; the file records the settings, date and package versions.
"""
import argparse
import datetime as dt
import os
import platform
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import RESEARCH_DISCLAIMER  # noqa: E402
from digital_twin.twin import DigitalTwin  # noqa: E402
from experiments.modality_experiments import sweep  # noqa: E402
from metrics.image_quality import evaluate_image_quality  # noqa: E402
from modalities import get_modality, modality_names  # noqa: E402
from simulation.datasets import circular_fov_mask  # noqa: E402
from simulation.lesion import insert_lesion  # noqa: E402
from simulation.phantom import generate_shepp_logan  # noqa: E402


def md_table(df):
    """Markdown table without optional dependencies (None -> n/a)."""
    def cell(v):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "n/a"
        return f"{v:.4g}" if isinstance(v, (float, np.floating)) else str(v)
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--realizations", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "validation_report.md"))
    a = ap.parse_args()

    base = generate_shepp_logan(a.size)
    base_ct = base * circular_fov_mask(a.size)           # projection-based modalities need a circular FOV
    lines = [
        "# Validation report (generated)", "",
        f"> {RESEARCH_DISCLAIMER}", "",
        f"Generated {dt.datetime.now().isoformat(timespec='seconds')} by `scripts/generate_validation_report.py` "
        f"(python {platform.python_version()}, numpy {np.__version__}). Phantom: built-in synthetic Shepp-Logan "
        f"{a.size}x{a.size} with an inserted low-contrast target; {a.realizations} noise realizations per point "
        "(mean ± std). **Every number below was measured by this run; nothing is typed in.** Synthetic phantom only, "
        "so these numbers describe the simulator, not any scanner or patient data.", "",
    ]
    for name in modality_names():
        m = get_modality(name)
        phantom, lesion, bg, meta = insert_lesion(base_ct if m.uses_circular_fov else base,
                                                  radius=max(3.0, a.size / 16), contrast=0.15)
        twin = DigitalTwin(m, phantom, lesion, bg, meta)
        ref = m.reference_protocol()
        lines += [f"## {name}", "", f"Reference protocol: `{m.describe_protocol(ref)}` (normalized cost = {m.reference_cost().cost:.3f}).", ""]
        # reproducibility check
        i1, n1, _ = m.acquire(phantom, ref, 11)
        i2, n2, _ = m.acquire(phantom, ref, 11)
        _, n3, _ = m.acquire(phantom, ref, 12)
        lines += [f"- Same seed reproduces the noisy measurement exactly: **{bool(np.array_equal(n1, n2))}**; "
                  f"different seed differs: **{not np.array_equal(n1, n3)}**."]
        ideal, _, meta_a = m.acquire(phantom, ref, 11)
        rec, _ = m.reconstruct(ideal, ref, meta_a)
        q = evaluate_image_quality(m.metric_reference(phantom, ref), rec)
        lines += [f"- Noise-free (ideal) pipeline at the reference protocol: SSIM = {q['ssim']:.4f}, RMSE = {q['rmse']:.4f}.", ""]
        ax = m.sweep_axes()["primary"]
        df = sweep(twin, ref, ax["param"], ax["values"], n_real=a.realizations)
        lines += [f"### Sweep of `{ax['param']}` (other parameters at the reference)", "", md_table(df), ""]
    lines += ["## Notes", "",
              "- Differences between modalities in absolute SSIM are NOT comparable: the noise models and normalizations differ.",
              "- Re-run the script after changing any modality constant; the report is overwritten with the new measurements."]
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
