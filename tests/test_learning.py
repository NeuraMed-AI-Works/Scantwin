"""Tests for image descriptors and the per-modality learned protocol model (small, fast settings)."""

import numpy as np

from learning.features import FEATURE_NAMES, compute_image_features
from learning.protocol_model import (
    MIN_TRAIN_IMAGES,
    Criteria,
    TargetConfig,
    build_training_table,
    recommend_protocol,
    simulate_grid,
    train_protocol_model,
)
from learning.training import train_models
from modalities import get_modality
from simulation.datasets import circular_fov_mask, synthetic_series

NAMES = ["CT", "X-ray", "MRI", "Ultrasound"]


def _images(n=4, size=48):
    out = []
    for i, s in enumerate(synthetic_series(size, n_slices=n)):
        img = np.clip(s.image * (0.6 + 0.15 * i), 0, 1)   # vary contrast so images are not identical
        img[~circular_fov_mask(size)] = 0.0
        out.append(img)
    return out


def test_features_are_finite_and_complete():
    f = compute_image_features(_images(1)[0])
    assert set(f) == set(FEATURE_NAMES) and all(np.isfinite(v) for v in f.values())


def test_simulate_grid_physics_trends_ct():
    ct = get_modality("CT")
    img = _images(1)[0]
    protos = [dict(ct.reference_protocol(), photon_budget=i0, num_projections=v) for i0 in (1e3, 1e5) for v in (30, 90)]
    g = simulate_grid(ct, img, seed=1, protocols=protos)
    assert len(g) == 4 and np.all(np.isfinite(g[["ssim", "psnr", "rmse", "cost"]].to_numpy()))
    hi = g[(g.photon_budget == 1e5) & (g.num_projections == 90)].iloc[0]
    lo = g[(g.photon_budget == 1e3) & (g.num_projections == 30)].iloc[0]
    assert hi.ssim > lo.ssim and hi.rmse < lo.rmse and hi.cost > lo.cost


def test_grid_is_small_and_cost_is_normalized_for_every_modality():
    for name in NAMES + ["Other"]:
        m = get_modality(name)
        assert 0 < len(m.training_grid()) <= 60
        assert abs(m.reference_cost().cost - 1.0) < 1e-12
        assert max(m.acquisition_cost(p).cost for p in m.training_grid()) <= 1.0 + 1e-12


def test_too_few_images_rejected():
    imgs = _images(2)
    table = build_training_table(get_modality("CT"), imgs, ["a", "b"])
    try:
        train_protocol_model(table, modality="CT")
    except ValueError as e:
        assert str(MIN_TRAIN_IMAGES) in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_train_and_recommend_is_confirmed_by_simulation_for_each_modality():
    imgs = _images(4)
    for name in NAMES:
        m = get_modality(name)
        seen = []
        table = build_training_table(m, imgs[:3], ["a", "b", "c"], progress=lambda d, t, n, g: seen.append((d, t, n)))
        assert seen[-1][:2] == (3, 3) and set(table.modality) == {name}
        model = train_protocol_model(table, modality=name, n_estimators=25)
        assert model.modality == name and "beats_baseline" in model.cv
        crit = Criteria(min_ssim=0.3)
        rec = recommend_protocol(model, m, imgs[3], crit, target=TargetConfig(4.0, 0.1), verify_top=3)
        assert rec.status in ("confirmed", "not_confirmed")
        if rec.status == "confirmed":
            assert rec.chosen["ssim"] >= 0.3                    # confirmed = simulated, not just predicted
            assert rec.chosen["cost"] <= rec.reference["cost"] + 1e-12
            assert rec.verified["confirmed"].any()
            assert rec.cost_reduction_pct is not None and rec.cost_reduction_pct >= -1e-9
        impossible = recommend_protocol(model, m, imgs[3], Criteria(min_ssim=1.01), verify_top=2)
        assert impossible.status in ("none_predicted", "not_confirmed") and impossible.chosen is None
        assert "No feasible protocol found within the current search space." in impossible.message


def test_per_modality_and_pooled_models():
    imgs = _images(4)
    res = train_models(imgs[:3] + imgs[:2], ["a", "b", "c", "d", "e"], ["CT"] * 3 + ["MRI"] * 2, n_estimators=20)
    assert "CT" in res.models and "pooled" in res.models           # MRI has only 2 images -> pooled fallback
    assert res.pooled_for == ["MRI"] and res.warnings
    assert res.model_for("MRI").pooled and not res.model_for("CT").pooled
    assert res.model_for("Ultrasound") is None
