"""
Automated Sanity-Check and Scientific Integrity Test Suite (Phase 2)
===================================================================
Rigorous verification of the 10 mandatory physical and numerical behaviors:
- Test 1: Ground truth compared against itself (MSE ≈ 0, SSIM ≈ 1).
- Test 2: Fixed random seed produces identical noisy sinogram.
- Test 3: Distinct random seeds produce distinct noisy sinograms.
- Test 4: Low photon budget exhibits statistically greater noise variance than high photon budget.
- Test 5: Very sparse views produce measurable reconstruction degradation compared to dense views.
- Test 6: Increasing lesion contrast produces higher CNR.
- Test 7: When lesion is disabled, CNR calculation handles absent masks safely without crash.
- Test 8: Unachievable optimizer constraints report no feasible protocol.
- Test 9: All computed metrics (MSE, RMSE, PSNR, SSIM, CNR) evaluate to finite real numbers.
- Test 10: Zero NaNs or Infs in complete digital twin forward simulation pipeline.
"""

import pytest
import numpy as np

from config import DEFAULT_MIN_SSIM, DEFAULT_MIN_CNR
from simulation.phantom import generate_shepp_logan
from simulation.lesion import insert_lesion
from simulation.projector import compute_radon_projection
from simulation.noise_model import apply_photon_noise
from simulation.acquisition import simulate_acquisition
from reconstruction.reconstruction_engine import reconstruct
from metrics.image_quality import evaluate_image_quality, compute_mse, compute_ssim
from metrics.cnr import compute_cnr, compute_roi_statistics
from digital_twin.twin import CTDigitalTwin, evaluate_protocol_dict, is_protocol_feasible
from optimization.protocol_optimizer import ProtocolOptimizer


def test_1_phantom_identity():
    """Test 1: Phantom compared against itself must yield MSE ≈ 0 and SSIM ≈ 1."""
    p = generate_shepp_logan(64)
    mse = compute_mse(p, p)
    ssim = compute_ssim(p, p)
    assert np.isclose(mse, 0.0), f"Identity MSE should be 0, got {mse}"
    assert np.isclose(ssim, 1.0), f"Identity SSIM should be 1.0, got {ssim}"


def test_2_seed_reproducibility():
    """Test 2: Same random seed produces bit-for-bit identical noisy sinogram."""
    p = generate_shepp_logan(64)
    sino, _ = compute_radon_projection(p, num_projections=30)
    noisy_a = apply_photon_noise(sino, incident_photons=1e4, seed=777)
    noisy_b = apply_photon_noise(sino, incident_photons=1e4, seed=777)
    np.testing.assert_array_equal(noisy_a, noisy_b)


def test_3_distinct_seed_stochasticity():
    """Test 3: Different random seeds produce distinct Poisson noise realizations."""
    p = generate_shepp_logan(64)
    sino, _ = compute_radon_projection(p, num_projections=30)
    noisy_a = apply_photon_noise(sino, incident_photons=1e4, seed=111)
    noisy_b = apply_photon_noise(sino, incident_photons=1e4, seed=222)
    assert not np.array_equal(noisy_a, noisy_b)


def test_4_low_photon_greater_noise_variance():
    """Test 4: Low photon budget (1e3) has statistically greater variance than high (1e6)."""
    p = generate_shepp_logan(64)
    sino, _ = compute_radon_projection(p, num_projections=30)
    noisy_low = apply_photon_noise(sino, incident_photons=1e3, seed=42)
    noisy_high = apply_photon_noise(sino, incident_photons=1e6, seed=42)
    var_low = float(np.var(noisy_low))
    var_high = float(np.var(noisy_high))
    assert var_low > var_high, f"Expected var(1e3) > var(1e6), got {var_low} vs {var_high}"


def test_5_sparse_views_degrade_reconstruction():
    """Test 5: Very sparse views (30 views) degrade reconstruction compared to dense views (180 views)."""
    p = generate_shepp_logan(64)
    sino_dense, theta_dense = compute_radon_projection(p, num_projections=180)
    sino_sparse, theta_sparse = compute_radon_projection(p, num_projections=30)

    recon_dense, _ = reconstruct(sino_dense, theta_dense, method="FBP")
    recon_sparse, _ = reconstruct(sino_sparse, theta_sparse, method="FBP")

    ssim_dense = compute_ssim(p, recon_dense)
    ssim_sparse = compute_ssim(p, recon_sparse)

    assert ssim_dense > ssim_sparse, (
        f"Dense sampling SSIM ({ssim_dense:.4f}) should exceed sparse SSIM ({ssim_sparse:.4f})"
    )


def test_6_contrast_increase_improves_cnr():
    """Test 6: Increasing synthetic target contrast should generally produce higher CNR."""
    base = generate_shepp_logan(64)
    p_low, l_low, bg_low, _ = insert_lesion(base, radius=6.0, contrast=0.04)
    p_high, l_high, bg_high, _ = insert_lesion(base, radius=6.0, contrast=0.16)

    sino_low, theta = compute_radon_projection(p_low, num_projections=60)
    sino_high, _ = compute_radon_projection(p_high, num_projections=60)

    recon_low, _ = reconstruct(sino_low, theta, method="FBP")
    recon_high, _ = reconstruct(sino_high, theta, method="FBP")

    cnr_low = compute_cnr(recon_low, l_low, bg_low)
    cnr_high = compute_cnr(recon_high, l_high, bg_high)

    assert cnr_high > cnr_low, f"Higher contrast CNR ({cnr_high:.2f}) should exceed low ({cnr_low:.2f})"


def test_7_lesion_disabled_handled_safely():
    """Test 7: When lesion is absent or empty, ROI stats return safe zero structure without crash."""
    img = np.zeros((32, 32))
    empty_mask = np.zeros((32, 32), dtype=bool)
    stats = compute_roi_statistics(img, empty_mask, empty_mask)
    assert stats["cnr"] == 0.0
    assert stats["delta_contrast"] == 0.0


def test_8_unachievable_optimizer_constraint():
    """Test 8: Impossible optimizer constraints report no feasible protocol."""
    base = generate_shepp_logan(64)
    p, l, bg, meta = insert_lesion(base, radius=4.0, contrast=0.02)
    twin = CTDigitalTwin(p, l, bg, meta)
    optimizer = ProtocolOptimizer(twin, target_ssim_min=0.999, target_cnr_min=99.0, seed=42)
    best_state, _, summary = optimizer.optimize(n_trials=3)
    assert best_state is None
    assert "No feasible protocol found" in summary


def test_9_all_metrics_finite():
    """Test 9: All computed metrics evaluate to finite real numbers."""
    base = generate_shepp_logan(64)
    p, l, bg, _ = insert_lesion(base, radius=5.0, contrast=0.08)
    noisy_sino, _, theta, _ = simulate_acquisition(p, num_projections=45, photon_budget=1e4)
    recon, _ = reconstruct(noisy_sino, theta)
    q = evaluate_image_quality(p, recon)
    cnr = compute_cnr(recon, l, bg)

    for k, v in q.items():
        assert np.isfinite(v), f"Metric {k} is non-finite: {v}"
    assert np.isfinite(cnr), f"CNR is non-finite: {cnr}"


def test_10_no_nans_or_infs_in_forward_pipeline():
    """Test 10: Digital twin structured dictionary output contains zero NaNs or Infs."""
    base = generate_shepp_logan(64)
    twin = CTDigitalTwin(base)
    res = evaluate_protocol_dict(twin, photon_budget=1e4, projection_count=30, seed=42)

    for k, v in res.items():
        if isinstance(v, float):
            assert not np.isnan(v), f"Field {k} returned NaN"
            assert not np.isinf(v), f"Field {k} returned Inf"
