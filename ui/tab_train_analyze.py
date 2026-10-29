"""Tab 1: Start Here / Train & Analyze — the guided workflow (modality + data -> train -> test images -> results)."""

from __future__ import annotations

import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from config import DATA_USE_WARNING, NO_FEASIBLE_MESSAGE
from learning.protocol_model import (Criteria, MIN_TRAIN_IMAGES, ProtocolModel, TargetConfig, list_saved_models, load_model,
                                     recommend_protocol)
from learning.report import build_markdown_report, results_table
from learning.training import TrainingResult, builtin_demo_images, train_models
from modalities import get_modality, modality_names
from simulation.datasets import file_extension, load_slices, synthetic_series
from ui.common import draw_image, fmt, metric_grid, quickstart_panel, scope_box, show_fig
from ui.compat import stretch
from ui.sidebar import UPLOAD_TYPES

LEARN_SIZE = 128
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def _read_uploads(files, mods, npz_key):
    images, names, out_mods, errors = [], [], [], []
    for f, mod in zip(files, mods):
        m = get_modality(mod)
        ext = file_extension(f.name)
        try:
            for sl in load_slices(f.getvalue(), size=LEARN_SIZE, name=f.name, npz_key=npz_key, fov_circle=m.uses_circular_fov,
                                  is_hu=True if (mod == "CT" and ext in (".nii", ".nii.gz")) else None):
                images.append(sl.image); names.append(sl.name); out_mods.append(mod)
        except (ValueError, ImportError, OSError) as exc:
            errors.append(f"{f.name}: {exc}")
    return images, names, out_mods, errors


@st.cache_data(show_spinner=False)
def _analyze_cached(name: str, image: np.ndarray, modality_name: str, crit_key: tuple, target_key, model_id: str, seed: int, _model):
    crit = Criteria(*crit_key)
    target = TargetConfig(*target_key) if target_key else None
    return recommend_protocol(_model, get_modality(modality_name), image, crit, target, seed=seed)


def _store_models(res: TrainingResult) -> None:
    models = st.session_state.setdefault("models", {})
    models.update(res.models)
    st.session_state["train_info"] = res


def _model_for(modality: str):
    models = st.session_state.get("models", {})
    if modality in models:
        return models[modality]
    pooled = models.get("pooled")
    return pooled if pooled is not None and modality in pooled.covered_modalities else None


def _render_model_cards() -> None:
    models = st.session_state.get("models", {})
    info = st.session_state.get("train_info")
    for key, mdl in models.items():
        cv = mdl.cv
        st.markdown(f"##### Model: `{key}` — modalities {', '.join(mdl.covered_modalities)}")
        metric_grid([
            ("Training images", str(mdl.n_images)),
            ("CV SSIM error (MAE)", fmt(cv["ssim_mae"], ".3f"), f"{cv['ssim_mae'] - cv['ssim_mae_baseline']:+.3f} vs mean baseline"),
            ("CV R² (SSIM)", fmt(cv["ssim_r2"], ".2f")),
            ("CV PSNR error (MAE, dB)", fmt(cv["psnr_mae"], ".2f")),
            ("Beats mean baseline?", "yes" if cv["beats_baseline"] else "no"),
            ("Protocol grid per image", str(int(info.table[info.table["image"] == info.table["image"].iloc[0]].shape[0])) if info and info.table is not None else "—"),
        ], cols=3)
        for w in mdl.warnings:
            st.warning(w)
        with st.expander(f"What the `{key}` model relies on (feature importance)"):
            st.bar_chart(pd.Series(mdl.importances).sort_values())
        if info is not None and key in info.saved_paths:
            st.caption(f"Saved to `{os.path.relpath(info.saved_paths[key], os.path.dirname(MODELS_DIR))}`")


def render(ctx) -> None:
    gm = ctx.modality
    st.subheader("Start here — learn from real images, then analyze new ones")
    st.markdown(
        "**How it works.** (1) choose the modality and your datasets; (2) the digital twin simulates a small grid of acquisition "
        "protocols for every training image and a model learns how image content and settings determine reconstruction quality — "
        "**separately for each modality**; (3) you provide **test images**; (4) for each one ScanTwin predicts which protocols meet "
        "your criteria, ranks them by cost and **confirms the cheapest by simulation**. Nothing here diagnoses disease."
    )
    st.warning(DATA_USE_WARNING)

    # ------------------------------------------------------------------ Step 1
    st.markdown("### Step 1 — Modality and training data")
    scope_box(gm)
    quickstart_panel()
    src = st.radio("Training data", ["Upload my images", "Built-in synthetic demo set"], horizontal=True, key="ta_src",
                   help="The demo set is SYNTHETIC (warped Shepp-Logan series + phantoms): a way to try the app without any download.")
    train_files, train_mods = [], []
    if src.startswith("Upload"):
        train_files = st.file_uploader(
            f"Training images (at least {MIN_TRAIN_IMAGES} per modality, 8+ recommended; stacks, volumes and MedMNIST .npz count as several)",
            type=UPLOAD_TYPES, accept_multiple_files=True, key="ta_train_up",
            help="A MedMNIST .npz uses its train_images; a NIfTI volume gives up to 8 axial slices.") or []
        if train_files:
            meta = pd.DataFrame({"file": [f.name for f in train_files], "modality": [gm.name] * len(train_files)})
            edited = st.data_editor(meta, hide_index=True, disabled=["file"], key=f"ta_meta_{len(train_files)}_{gm.name}",
                                    column_config={"modality": st.column_config.SelectboxColumn("modality", options=modality_names(), required=True)})
            train_mods = edited["modality"].tolist()
    else:
        st.info(f"Demo training on {gm.name}: 10 synthetic images ({LEARN_SIZE}×{LEARN_SIZE}).")
    st.caption(f"Resolution for training and analysis: {LEARN_SIZE}×{LEARN_SIZE} (resampled, zero-padded to square"
               f"{', masked to a circle for CT/Other' if gm.uses_circular_fov else ''}).")

    c1, c2 = st.columns([1, 2])
    do_train = c1.button("🎓 Train", type="primary", key="ta_train", disabled=(src.startswith("Upload") and not train_files))
    saved = list_saved_models(MODELS_DIR)
    with c2:
        if saved:
            pick = st.selectbox("Saved models", [p.name for p in saved], key="ta_saved")
            st.caption("⚠️ Only load model files you created yourself — joblib/pickle files can execute code.")
            if st.button("Load selected model", key="ta_load"):
                try:
                    mdl = load_model(str(next(p for p in saved if p.name == pick)))
                    st.session_state.setdefault("models", {})[mdl.modality] = mdl
                    st.session_state["train_info"] = None
                except Exception as exc:  # noqa: BLE001 - show any load problem to the user
                    st.error(f"Could not load model: {exc}")

    # ------------------------------------------------------------------ Step 2
    if do_train:
        if src.startswith("Upload"):
            images, names, mods, errs = _read_uploads(train_files, train_mods, "train_images")
            for e in errs:
                st.error(e)
        else:
            images, names = builtin_demo_images(gm.name, LEARN_SIZE)
            mods, errs = [gm.name] * len(images), []
        if len(images) < MIN_TRAIN_IMAGES:
            st.error(f"Need at least {MIN_TRAIN_IMAGES} readable training images; got {len(images)}.")
        else:
            st.markdown("### Step 2 — Training (live)")
            bar = st.progress(0.0, text="Simulating protocol grids…")
            live_tbl, rows = st.empty(), []

            def on_image(done, total, name, mod, grid):
                rows.append({"image": name, "modality": mod, "protocols simulated": len(grid),
                             "SSIM best": grid["ssim"].max(), "SSIM worst": grid["ssim"].min(),
                             "cost range": f"{grid['cost'].min():.3f} – {grid['cost'].max():.3f}"})
                bar.progress(done / total, text=f"Simulated {done}/{total}: {name} ({mod})")
                live_tbl.dataframe(pd.DataFrame(rows), **stretch(st.dataframe))

            with st.spinner("Simulating, then fitting and cross-validating the model(s)…"):
                res = train_models(images, names, mods, seed=int(ctx.seed), models_dir=MODELS_DIR, progress=on_image)
            bar.progress(1.0, text="Training complete")
            for w in res.warnings:
                st.warning(w)
            _store_models(res)
            st.session_state["trained_demo"] = not src.startswith("Upload")
            if res.table is not None:
                st.download_button("Download training table (CSV)", res.table.to_csv(index=False).encode("utf-8"),
                                   file_name="scantwin_training_table.csv", mime="text/csv")

    if not st.session_state.get("models"):
        st.caption("Train a model (or load a saved one) to unlock Step 3, where you will be asked for test images.")
        return
    st.markdown("### Model quality")
    _render_model_cards()

    # ------------------------------------------------------------------ Step 3
    st.markdown("### Step 3 — Test images")
    st.success("**Training done — please upload the test image(s) you want analyzed.**")
    t1, t2 = st.columns(2)
    test_mod_name = t1.selectbox("Modality of the test images", modality_names(), index=modality_names().index(gm.name), key="ta_test_mod")
    tm = get_modality(test_mod_name)
    dc = tm.default_criteria
    min_ssim = t2.slider("Experimental criterion: minimum SSIM", 0.30, 0.99, float(dc["min_ssim"]), 0.01, key=f"ta_ssim_{test_mod_name}")
    u1, u2, u3 = st.columns(3)
    use_psnr = u1.checkbox("Also require minimum PSNR", key=f"ta_upsnr_{test_mod_name}")
    min_psnr = u1.slider("Minimum PSNR (dB)", 10.0, 50.0, float(dc["min_psnr"]), 0.5, key=f"ta_psnr_{test_mod_name}") if use_psnr else None
    use_tgt = u2.checkbox("Insert a synthetic target (enables CNR)", value=False, key=f"ta_tgt_{test_mod_name}",
                          help="CNR is then checked by simulation (the model itself predicts SSIM and PSNR).")
    tr, tc = (u2.slider("Target radius (px)", 3.0, 12.0, 5.0, 0.5, key=f"ta_tr_{test_mod_name}"),
              u2.slider("Target contrast", 0.01, 0.25, 0.08, 0.01, key=f"ta_tc_{test_mod_name}")) if use_tgt else (5.0, 0.08)
    min_cnr = u3.slider("Minimum CNR", 0.5, 10.0, float(dc["min_cnr"]), 0.1, key=f"ta_cnr_{test_mod_name}") if use_tgt else None
    st.caption("These are user-set experimental criteria, never clinical thresholds.")

    test_files = st.file_uploader("Upload test image(s)", type=UPLOAD_TYPES, accept_multiple_files=True, key="ta_test_up",
                                  help="A MedMNIST .npz uses its separate test_images array.") or []
    use_demo_test = False
    if st.session_state.get("trained_demo"):
        use_demo_test = st.checkbox("Use a built-in SYNTHETIC test image (demo)", value=not test_files, key="ta_demo_test")

    model = _model_for(test_mod_name)
    if model is None:
        st.error(f"No model covers {test_mod_name}. Train with {test_mod_name} images (or the demo set) first.")
        return
    if model.pooled:
        st.warning(f"{test_mod_name} is served by the pooled model — predictions are less reliable.")
    elif test_mod_name not in model.covered_modalities:
        st.warning("Modality mismatch between the model and the test images.")

    items: List[tuple] = []
    for f in test_files:
        imgs, names, _, errs = _read_uploads([f], [test_mod_name], "test_images")
        for e in errs:
            st.error(e)
        items += list(zip(names, imgs))
    if use_demo_test:
        demo = synthetic_series(LEARN_SIZE, 13, fov_circle=tm.uses_circular_fov)[6]
        items.append((demo.name + " (synthetic)", demo.image))
    if not items:
        st.info("Waiting for test images…")
        return

    # ------------------------------------------------------------------ Step 4
    st.markdown("### Step 4 — Results")
    crit = Criteria(min_ssim, min_psnr, min_cnr)
    results = []
    for name, img in items:
        with st.spinner(f"Analyzing {name}…"):
            rec = _analyze_cached(name, img, test_mod_name, crit.key(), (tr, tc) if use_tgt else None, model.model_id,
                                  int(ctx.seed) + 123, model)
        results.append((name, rec))
        _render_result(name, img, rec, tm)
    tbl = results_table(results)
    d1, d2 = st.columns(2)
    d1.download_button("Download all results (CSV)", tbl.to_csv(index=False).encode("utf-8"),
                       file_name="scantwin_test_analysis.csv", mime="text/csv")
    d2.download_button("Download report (Markdown)", build_markdown_report(results, model).encode("utf-8"),
                       file_name="scantwin_report.md", mime="text/markdown")


def _render_result(name, img, rec, tm) -> None:
    st.markdown(f"#### {name}")
    if rec.novelty_z > 3.0:
        st.warning(f"This image differs strongly from the training images (max descriptor z-score {rec.novelty_z:.1f}); "
                   "the prediction is less reliable.")
    ref, ch = rec.reference, rec.chosen
    a, b, c = st.columns(3)
    for col, im, cap in ((a, rec.truth, "Test image" + (" (+ synthetic target)" if rec.lesion_mask is not None else "")),
                         (b, rec.reference_recon, f"Reference result — SSIM {ref['ssim']:.3f}"),
                         (c, rec.chosen_recon, f"Recommended result — SSIM {ch['ssim']:.3f}" if ch else "No confirmed protocol")):
        with col:
            if im is None:
                st.info(cap)
                continue
            fig, ax = plt.subplots(figsize=(3.4, 3.4), dpi=100)
            draw_image(ax, np.clip(im, 0, 1), cap, lesion_mask=rec.lesion_mask if im is rec.truth else None)
            show_fig(fig)
    rows = [{"": "Reference", "Parameters": tm.describe_protocol(ref["protocol"]), "SSIM": fmt(ref["ssim"], ".4f"),
             "PSNR (dB)": fmt(ref["psnr"], ".2f"), "RMSE": fmt(ref["rmse"], ".4f"), "CNR": fmt(ref["cnr"], ".2f"),
             tm.cost_labels["exposure_short"] + " (%)": f"{ref['relative_exposure_pct']:.2f}", "Normalized cost": f"{ref['cost']:.3f}"}]
    if ch:
        rows.append({"": "Recommended", "Parameters": tm.describe_protocol(ch["protocol"]), "SSIM": fmt(ch["ssim"], ".4f"),
                     "PSNR (dB)": fmt(ch["psnr"], ".2f"), "RMSE": fmt(ch["rmse"], ".4f"), "CNR": fmt(ch["cnr"], ".2f"),
                     tm.cost_labels["exposure_short"] + " (%)": f"{ch['relative_exposure_pct']:.2f}", "Normalized cost": f"{ch['cost']:.3f}"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, **stretch(st.dataframe))
    if ch:
        st.success(f"**{rec.message}** {tm.cost_labels['reduction']}: **{rec.cost_reduction_pct:.1f}%** (computed from the reference above). "
                   f"Predicted SSIM {ch['pred_ssim']:.3f}, simulated SSIM {ch['ssim']:.3f}. Simulated proxy only — not a clinical protocol.")
        with st.expander("Error maps (heatmap): reference vs recommended"):
            fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.2), dpi=100)
            ref_img = tm.metric_reference(rec.truth, ref["protocol"])
            draw_image(axes[0], np.abs(ref_img - rec.reference_recon), "|truth − reference|", cmap="inferno", vmin=0, vmax=0.35)
            draw_image(axes[1], np.abs(ref_img - rec.chosen_recon), "|truth − recommended|", cmap="inferno", vmin=0, vmax=0.35)
            show_fig(fig)
    else:
        st.error(f"**{NO_FEASIBLE_MESSAGE}**")
        st.caption(rec.message)
    if rec.verified is not None:
        with st.expander("Candidates checked by simulation (predicted vs simulated)"):
            v = rec.verified
            cols = [s.name for s in tm.protocol_space() if s.name in v.columns] + ["cost", "pred_ssim", "actual_ssim", "actual_psnr", "actual_cnr", "confirmed"]
            st.dataframe(v[cols], hide_index=True, **stretch(st.dataframe))
    with st.expander("Image descriptors"):
        st.json({k: round(float(v), 4) for k, v in rec.features.items()})
