"""
Protocol model: learn good acquisition settings per image and modality from real images
=======================================================================================
Training data are produced by the ScanTwin digital twin itself. For every training image and its modality we
simulate that modality's small protocol grid (``Modality.training_grid``), measure SSIM / PSNR / RMSE against the
image, and fit a random-forest surrogate

    quality(image descriptors, protocol parameters) -> SSIM, PSNR

For a new (test) image the surrogate ranks the modality's candidate protocols by normalized acquisition cost among
those predicted to satisfy the criteria; the cheapest candidates are then CONFIRMED by running the digital twin on
the test image. A recommendation is reported ONLY if a simulation confirms it with the single authoritative
feasibility function (``digital_twin.twin.is_protocol_feasible``, reached through ``DigitalTwin``).

Scope and honesty
-----------------
* The learner predicts reconstruction quality of a *simulation*; it does not diagnose anything.
* One model is trained per modality that has at least ``MIN_TRAIN_IMAGES`` images; modalities with fewer images share a
  pooled model (modality as a feature) and a clear warning is raised.
* Each modality uses its own simplified image-formation model, not validated scanner physics.
* With few training images the surrogate is weak; image-grouped cross-validation and a mean-predictor baseline are
  reported so this is visible.

DISCLAIMER: research/educational prototype; not for clinical use.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold

from config import NO_FEASIBLE_MESSAGE
from digital_twin.twin import DigitalTwin, is_protocol_feasible
from learning.features import FEATURE_NAMES, compute_image_features
from metrics.image_quality import evaluate_image_quality
from modalities import Modality, get_modality
from simulation.datasets import suggest_lesion_center
from simulation.lesion import insert_lesion

MIN_TRAIN_IMAGES: int = 3
RECOMMENDED_TRAIN_IMAGES: int = 8
POOLED_KEY = "pooled"
METRIC_COLUMNS = ["ssim", "psnr", "rmse", "cost", "relative_exposure_pct"]


# ----------------------------------------------------------------------------------------
# Criteria / target configuration
# ----------------------------------------------------------------------------------------
@dataclass
class Criteria:
    """User-set EXPERIMENTAL criteria (never clinical thresholds)."""

    min_ssim: float = 0.88
    min_psnr: Optional[float] = None
    min_cnr: Optional[float] = None       # applies only when a synthetic target is inserted

    def constraints(self, has_target: bool) -> Dict[str, Any]:
        return {"min_ssim": self.min_ssim, "min_psnr": self.min_psnr, "min_cnr": self.min_cnr if has_target else None}

    def key(self) -> tuple:
        return (self.min_ssim, self.min_psnr, self.min_cnr)


@dataclass
class TargetConfig:
    """Synthetic target inserted into the test image so CNR can be evaluated (optional)."""

    radius: float = 6.0
    contrast: float = 0.08

    def key(self) -> tuple:
        return (self.radius, self.contrast)


# ----------------------------------------------------------------------------------------
# Grid simulation
# ----------------------------------------------------------------------------------------
def _protocol_frame(modality: Modality, protocols: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    names = [s.name for s in modality.protocol_space()]
    return pd.DataFrame([{n: p.get(n) for n in names} for p in protocols])


def simulate_grid(
    modality: Modality,
    image: np.ndarray,
    seed: int = 0,
    protocols: Optional[Sequence[Dict[str, Any]]] = None,
) -> pd.DataFrame:
    """Run the digital twin over a protocol grid (default: the modality's training grid) for one image.

    Measurements are reused between protocols that share the same acquisition (same seed), so only the
    reconstruction differs between them, as in a real re-reconstruction.
    """
    protos = [modality.normalize_protocol(p) for p in (protocols if protocols is not None else modality.training_grid())]
    cache: Dict[tuple, Any] = {}
    rows = []
    for proto in protos:
        proto = {k: v for k, v in proto.items() if k != "seed"}
        key = modality.acquisition_key(proto)
        if key not in cache:
            _, noisy, meta = modality.acquire(image, proto, seed)
            cache[key] = (noisy, meta)
        noisy, meta = cache[key]
        recon, _ = modality.reconstruct(noisy, proto, meta)
        q = evaluate_image_quality(modality.metric_reference(image, proto), recon)
        cost = modality.acquisition_cost(proto)
        rows.append({**{s.name: proto.get(s.name) for s in modality.protocol_space()},
                     "ssim": q["ssim"], "psnr": q["psnr"], "rmse": q["rmse"],
                     "cost": cost.cost, "relative_exposure_pct": cost.exposure_pct})
    return pd.DataFrame(rows)


def build_training_table(
    modality: Modality,
    images: Sequence[np.ndarray],
    names: Sequence[str],
    seed: int = 0,
    progress: Optional[Callable[[int, int, str, pd.DataFrame], None]] = None,
) -> pd.DataFrame:
    """Simulate the modality's grid for every training image and attach descriptors. Progress callback per image."""
    frames = []
    for k, (img, name) in enumerate(zip(images, names)):
        grid = simulate_grid(modality, img, seed=seed + k)
        feats = compute_image_features(img, modality.fov_mask(img.shape))
        for key, val in feats.items():
            grid[key] = val
        grid["image"], grid["modality"] = name, modality.name
        frames.append(grid)
        if progress:
            progress(k + 1, len(images), name, grid)
    return pd.concat(frames, ignore_index=True)


# ----------------------------------------------------------------------------------------
# Surrogate model
# ----------------------------------------------------------------------------------------
def design_matrix(df: pd.DataFrame, modalities: Sequence[str], pooled: bool, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Numeric features: image descriptors + (modality one-hot if pooled) + protocol encodings from each modality."""
    parts = [df[FEATURE_NAMES].astype(float).reset_index(drop=True)]
    df = df.reset_index(drop=True)
    if pooled:
        parts.append(pd.DataFrame({f"modality={m}": (df["modality"] == m).astype(float) for m in modalities}))
    enc_cols: Dict[str, np.ndarray] = {}
    for m in modalities:
        mod = get_modality(m)
        prefix = f"{m}|" if pooled else ""
        rows = df.index[df["modality"] == m]
        if len(rows) == 0:
            for c in mod.encode_protocols(_protocol_frame(mod, [mod.reference_protocol()]), prefix).columns:
                enc_cols.setdefault(c, np.zeros(len(df)))
            continue
        e = mod.encode_protocols(df.loc[rows].reset_index(drop=True), prefix)
        for c in e.columns:
            col = enc_cols.setdefault(c, np.zeros(len(df)))
            col[rows] = e[c].to_numpy()
    parts.append(pd.DataFrame(enc_cols))
    x = pd.concat(parts, axis=1)
    if columns is not None:
        x = x.reindex(columns=columns, fill_value=0.0)
    return x


@dataclass
class ProtocolModel:
    """Trained surrogate plus everything needed to audit it."""

    forest: RandomForestRegressor
    modality: str                         # modality name, or "pooled"
    covered_modalities: List[str]
    feature_columns: List[str]
    feature_mean: Dict[str, float]
    feature_std: Dict[str, float]
    n_images: int
    n_images_by_modality: Dict[str, int]
    cv: Dict[str, Any]
    importances: Dict[str, float]
    warnings: List[str] = field(default_factory=list)
    trained_at: float = field(default_factory=time.time)

    @property
    def pooled(self) -> bool:
        return self.modality == POOLED_KEY

    @property
    def model_id(self) -> str:
        return f"{self.modality}:{self.trained_at:.3f}:{self.n_images}"

    @property
    def modalities(self) -> List[str]:        # name kept from the Phase-3 UI
        return self.covered_modalities

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        x = design_matrix(df, self.covered_modalities, self.pooled, self.feature_columns)
        return self.forest.predict(x.to_numpy(dtype=float))        # columns: ssim, psnr

    def novelty(self, feats: Dict[str, float]) -> float:
        """Largest |z-score| of a test image's descriptors vs the training images (> 3 = unlike the training data)."""
        z = [abs(feats[k] - self.feature_mean[k]) / (self.feature_std[k] + 1e-9) for k in FEATURE_NAMES]
        return float(max(z))


def train_protocol_model(
    table: pd.DataFrame,
    modality: Optional[str] = None,
    seed: int = 0,
    n_estimators: int = 200,
) -> ProtocolModel:
    """
    Fit the surrogate; evaluate with image-grouped cross-validation against a mean-predictor baseline.

    ``modality`` given -> a per-modality model trained on that modality's rows only. ``None`` -> a pooled model over
    every row with the modality as a feature.
    """
    pooled = modality is None
    sub = table if pooled else table[table["modality"] == modality]
    n_images = sub["image"].nunique()
    if n_images < MIN_TRAIN_IMAGES:
        raise ValueError(f"Need at least {MIN_TRAIN_IMAGES} training images, got {n_images}.")
    mods = sorted(sub["modality"].unique().tolist())
    sub = sub.reset_index(drop=True)
    xdf = design_matrix(sub, mods, pooled)
    x = xdf.to_numpy(dtype=float)
    y = sub[["ssim", "psnr"]].to_numpy(dtype=float)
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError("Training table contains non-finite values.")

    groups = sub["image"].to_numpy()
    n_splits = min(5, n_images)
    preds, base = np.zeros_like(y), np.zeros_like(y)
    for tr, te in GroupKFold(n_splits=n_splits).split(x, y, groups):
        rf = RandomForestRegressor(n_estimators=n_estimators, random_state=seed, n_jobs=-1, min_samples_leaf=2)
        rf.fit(x[tr], y[tr])
        preds[te] = rf.predict(x[te])
        base[te] = y[tr].mean(axis=0)
    mae, mae_base = np.abs(preds - y).mean(axis=0), np.abs(base - y).mean(axis=0)
    ss_res, ss_tot = ((y - preds) ** 2).sum(axis=0), ((y - y.mean(axis=0)) ** 2).sum(axis=0)
    cv = {
        "n_splits": n_splits,
        "ssim_mae": float(mae[0]), "ssim_mae_baseline": float(mae_base[0]), "ssim_r2": float(1 - ss_res[0] / ss_tot[0]),
        "psnr_mae": float(mae[1]), "psnr_mae_baseline": float(mae_base[1]), "psnr_r2": float(1 - ss_res[1] / ss_tot[1]),
        "beats_baseline": bool(mae[0] < mae_base[0]),
    }

    forest = RandomForestRegressor(n_estimators=n_estimators, random_state=seed, n_jobs=-1, min_samples_leaf=2)
    forest.fit(x, y)
    per_image = sub.groupby("image")[FEATURE_NAMES].first()
    warns: List[str] = []
    if n_images < RECOMMENDED_TRAIN_IMAGES:
        warns.append(f"Only {n_images} training images (8+ recommended): treat predictions as rough.")
    if not cv["beats_baseline"]:
        warns.append("The model does not beat a simple mean predictor in cross-validation; add more varied training images. "
                     "Recommendations are still confirmed by simulation before being reported.")
    if pooled:
        warns.append("Pooled model (modality used as a feature) because at least one modality has fewer than "
                     f"{MIN_TRAIN_IMAGES} images; predictions for those modalities are less reliable.")
    return ProtocolModel(
        forest=forest,
        modality=POOLED_KEY if pooled else str(modality),
        covered_modalities=mods,
        feature_columns=list(xdf.columns),
        feature_mean={k: float(per_image[k].mean()) for k in FEATURE_NAMES},
        feature_std={k: float(per_image[k].std(ddof=0)) for k in FEATURE_NAMES},
        n_images=int(n_images),
        n_images_by_modality={m: int(sub[sub["modality"] == m]["image"].nunique()) for m in mods},
        cv=cv,
        importances=dict(zip(xdf.columns, map(float, forest.feature_importances_))),
        warnings=warns,
    )


# ----------------------------------------------------------------------------------------
# Recommendation for a test image
# ----------------------------------------------------------------------------------------
@dataclass
class Recommendation:
    """Outcome of analysing one test image."""

    status: str                      # "confirmed" | "not_confirmed" | "none_predicted"
    message: str
    modality: str
    features: Dict[str, float]
    novelty_z: float
    criteria: Criteria
    reference: Dict[str, Any]        # protocol + simulated metrics at the reference protocol
    chosen: Optional[Dict[str, Any]] = None
    candidates: Optional[pd.DataFrame] = None   # predicted table for the whole search grid
    verified: Optional[pd.DataFrame] = None     # simulated check of the cheapest predicted-feasible candidates
    cost_reduction_pct: Optional[float] = None
    exposure_reduction_pct: Optional[float] = None
    truth: Optional[np.ndarray] = None          # image used as ground truth (incl. synthetic target if requested)
    lesion_mask: Optional[np.ndarray] = None
    background_mask: Optional[np.ndarray] = None
    reference_recon: Optional[np.ndarray] = None
    chosen_recon: Optional[np.ndarray] = None
    seed: int = 123


def _state_record(state, recon) -> Dict[str, Any]:
    return {
        "protocol": dict(state.protocol), "ssim": state.ssim, "psnr": state.psnr, "rmse": state.rmse,
        "cnr": state.cnr, "cost": state.normalized_acquisition_cost,
        "relative_exposure_pct": state.relative_photon_exposure_pct, "feasible": state.is_feasible,
    }


def recommend_protocol(
    model: ProtocolModel,
    modality: Modality,
    image: np.ndarray,
    criteria: Criteria,
    target: Optional[TargetConfig] = None,
    verify_top: int = 8,
    seed: int = 123,
) -> Recommendation:
    """Predict, rank by cost, then confirm the cheapest predicted-feasible protocols (and the reference) by simulation."""
    feats = compute_image_features(image, modality.fov_mask(image.shape))
    novelty = model.novelty(feats)

    # --- ground truth for confirmation (optionally with a synthetic target so CNR can be evaluated)
    if target is not None and modality.supports_lesion_cnr:
        centre = suggest_lesion_center(image, radius=target.radius)
        truth, lmask, bmask, lmeta = insert_lesion(image, center=centre, radius=target.radius, contrast=target.contrast)
    else:
        truth, lmask, bmask, lmeta = image, None, None, {}
    twin = DigitalTwin(modality, truth, lmask, bmask, lmeta)
    has_target = twin.has_target
    cons = criteria.constraints(has_target)

    def evaluate(protocol):
        return twin.evaluate_protocol(protocol, target_ssim_min=cons["min_ssim"], target_cnr_min=cons["min_cnr"],
                                      target_psnr_min=cons["min_psnr"], n_realizations=1, seed=seed)

    # --- reference (always simulated, always part of the search space)
    ref_proto = modality.normalize_protocol(modality.reference_protocol())
    ref_state, ref_recon, _ = evaluate(ref_proto)
    reference = _state_record(ref_state, ref_recon)

    # --- candidates predicted by the surrogate
    grid = [modality.normalize_protocol(p) for p in modality.search_grid()]
    if modality.protocol_key(ref_proto) not in {modality.protocol_key(p) for p in grid}:
        grid.append(ref_proto)
    cand = _protocol_frame(modality, grid)
    for key, val in feats.items():
        cand[key] = val
    cand["modality"] = modality.name
    pred = model.predict(cand)
    cand["pred_ssim"], cand["pred_psnr"] = pred[:, 0], pred[:, 1]
    costs = [modality.acquisition_cost(p) for p in grid]
    cand["cost"] = [c.cost for c in costs]
    cand["relative_exposure_pct"] = [c.exposure_pct for c in costs]
    cand["protocol_key"] = [modality.protocol_key(p) for p in grid]
    ref_key = modality.protocol_key(ref_proto)
    cand["is_reference"] = cand["protocol_key"] == ref_key
    # the surrogate predicts SSIM / PSNR only; CNR (needs the synthetic target) is checked by simulation
    cand["predicted_feasible"] = [
        is_protocol_feasible({"ssim": s, "psnr": p}, {"min_ssim": criteria.min_ssim, "min_psnr": criteria.min_psnr, "min_cnr": None})
        for s, p in zip(cand["pred_ssim"], cand["pred_psnr"])
    ]
    cand = cand.sort_values(["cost", "relative_exposure_pct"]).reset_index(drop=True)

    # --- confirm by simulation: cheapest predicted-feasible candidates, then the reference
    todo = cand[cand["predicted_feasible"] & ~cand["is_reference"]].head(verify_top)
    rows, recons = [], {}
    for _, r in todo.iterrows():
        proto = {s.name: r[s.name] for s in modality.protocol_space()}
        proto = _native(proto)
        state, recon, _ = evaluate(proto)
        rows.append({**r.to_dict(), "actual_ssim": state.ssim, "actual_psnr": state.psnr, "actual_rmse": state.rmse,
                     "actual_cnr": state.cnr, "confirmed": bool(state.is_feasible)})
        recons[r["protocol_key"]] = recon
    ref_row = cand[cand["is_reference"]].iloc[0].to_dict()
    rows.append({**ref_row, "actual_ssim": ref_state.ssim, "actual_psnr": ref_state.psnr, "actual_rmse": ref_state.rmse,
                 "actual_cnr": ref_state.cnr, "confirmed": bool(ref_state.is_feasible)})
    recons[ref_key] = ref_recon
    ver = pd.DataFrame(rows).sort_values(["cost", "relative_exposure_pct"]).reset_index(drop=True)

    common = dict(modality=modality.name, features=feats, novelty_z=novelty, criteria=criteria, reference=reference,
                  candidates=cand, verified=ver, truth=truth, lesion_mask=lmask, background_mask=bmask,
                  reference_recon=ref_recon, seed=seed)
    ok = ver[ver["confirmed"]]
    if ok.empty:
        any_predicted = bool(cand["predicted_feasible"].any())
        status = "not_confirmed" if any_predicted else "none_predicted"
        detail = ("The model predicted feasible protocols but simulation did not confirm any of them, and the reference "
                  "protocol itself does not satisfy the criteria." if any_predicted else
                  "No protocol in the search grid is predicted to satisfy the criteria, and the reference protocol "
                  "itself does not satisfy them.")
        return Recommendation(status, f"{NO_FEASIBLE_MESSAGE} {detail}", **common)

    best = ok.iloc[0].to_dict()
    proto_best = _native({s.name: best[s.name] for s in modality.protocol_space()})
    chosen = {"protocol": proto_best, "ssim": best["actual_ssim"], "psnr": best["actual_psnr"], "rmse": best["actual_rmse"],
              "cnr": best["actual_cnr"], "cost": best["cost"], "relative_exposure_pct": best["relative_exposure_pct"],
              "pred_ssim": best["pred_ssim"], "pred_psnr": best["pred_psnr"], "is_reference": bool(best["is_reference"])}
    cost_red = 100.0 * (1.0 - best["cost"] / reference["cost"])
    exp_red = 100.0 - best["relative_exposure_pct"] / max(reference["relative_exposure_pct"], 1e-12) * 100.0
    msg = ("Lowest-cost protocol, among those checked by simulation, that satisfies the experimental criteria."
           if not best["is_reference"] else
           "Only the reference protocol satisfied the criteria among the candidates checked by simulation (no cost reduction).")
    return Recommendation("confirmed", msg, chosen=chosen, chosen_recon=recons[best["protocol_key"]],
                          cost_reduction_pct=float(cost_red), exposure_reduction_pct=float(exp_red), **common)


def _native(proto: Dict[str, Any]) -> Dict[str, Any]:
    """numpy scalars -> python scalars (cleaner caches, JSON and display)."""
    out = {}
    for k, v in proto.items():
        out[k] = v.item() if hasattr(v, "item") else v
    return out


# ----------------------------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------------------------
def model_filename(key: str) -> str:
    return f"protocol_model_{key.lower().replace(' ', '-')}.joblib"


def save_model(model: ProtocolModel, path: str) -> None:
    import joblib

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str) -> ProtocolModel:
    """Load a model file. ONLY load files you created yourself: joblib/pickle can execute arbitrary code."""
    import joblib

    obj = joblib.load(path)
    if not isinstance(obj, ProtocolModel):
        raise ValueError("File is not a ScanTwin protocol model.")
    return obj


def list_saved_models(models_dir: str) -> List[Path]:
    d = Path(models_dir)
    return sorted(d.glob("protocol_model_*.joblib")) if d.exists() else []
