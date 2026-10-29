# Methodology

> Research and educational prototype; not for clinical use.

## Closed loop (all modalities)
ground truth -> **acquire** (modality physics + noise) -> raw measurement -> **reconstruct** -> quality assessment
(SSIM, PSNR, RMSE, CNR of a synthetic target) -> digital twin state -> **optimizer / recommender** -> new protocol.

Everything is reached through the `Modality` interface (`modalities/base.py`): `protocol_space`, `acquire`,
`reconstruct`, `acquisition_cost`, `measurement_view`, `live_session`, `suggest_protocol`.

## Cost (reference = 1.0)
| Modality | Cost | Headline proxy |
|---|---|---|
| CT / Other | 0.7 I0/I0ref + 0.3 views/views_ref | relative photon budget |
| X-ray | I0/I0ref | relative photon budget |
| MRI | (1/R) * NEX / NEX_ref | scan-time proxy |
| Ultrasound | N / N_ref | frame count |

## Optimization
Optuna TPE samples protocols from the modality's search space. The objective minimizes normalized cost with a penalty for
violating the criteria; `is_protocol_feasible` is the only feasibility authority. If no trial is feasible the app reports
"No feasible protocol found within the current search space."

## Learning ("Train from my images")
1. For each training image, simulate the modality's small protocol grid (<= 60 protocols) with the digital twin.
2. Features = image descriptors (intensity statistics, gradient energy, entropy, ...) + encoded protocol parameters.
3. RandomForest predicts SSIM and PSNR. Evaluation: grouped cross-validation (no image in both train and test fold) vs a mean baseline.
4. Recommendation: candidates meeting the criteria *as predicted* are ranked by cost, the cheapest few are **re-simulated by the twin**,
   and the reference protocol is always included. Only simulation-confirmed protocols are reported.
5. Models are saved as `models/protocol_model_<modality>.joblib`.

## Analyze
New test images are analyzed with the matching modality model: one result card per image (recommended protocol or the
no-feasible message, simulated metrics, relative simulated cost reduction), CSV download, optional Markdown report.

## Live acquisition
Whole scan is generated once with one seed and **revealed progressively**: CT views in golden-angle order, MRI k-space lines centre-first,
ultrasound frames accumulating, X-ray exposure ramp (photon budget fraction), plus a slice stream over a series. The last live frame equals the batch result (tested).

## Reproducibility
Seeds are explicit everywhere; `pytest` covers physics trends, reproducibility, live == batch, feasibility logic and learning.
`scripts/generate_validation_report.py` regenerates `validation_report.md` from real runs.
