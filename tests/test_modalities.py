"""Physics-level tests for every modality (reproducibility, trends, cost normalization, honesty)."""

import numpy as np

from config import RESEARCH_DISCLAIMER
from metrics.image_quality import evaluate_image_quality
from modalities import MODALITIES, get_modality, modality_names
from simulation.datasets import synthetic_series
from simulation.phantom import generate_shepp_logan

IMG = generate_shepp_logan(64)


def _quality(m, proto, seed=3, ideal=False, image=IMG):
    ideal_m, noisy_m, meta = m.acquire(image, proto, seed)
    recon, _ = m.reconstruct(ideal_m if ideal else noisy_m, proto, meta)
    return evaluate_image_quality(m.metric_reference(image, proto), recon), recon


def test_registry_has_all_five_modalities():
    assert set(modality_names()) == {"CT", "X-ray", "MRI", "Ultrasound", "Other"}
    for name, cls in MODALITIES.items():
        m = get_modality(name)
        assert isinstance(m, cls) and m.name == name
        assert len(m.model_scope_text) > 100 and m.cost_labels["exposure"]
        assert m.protocol_space() and m.reference_protocol() and m.default_protocol()


def test_reproducible_and_seed_sensitive_and_ideal_noisefree_available():
    for name in modality_names():
        m = get_modality(name)
        p = m.default_protocol()
        i1, n1, _ = m.acquire(IMG, p, 5)
        i2, n2, _ = m.acquire(IMG, p, 5)
        _, n3, _ = m.acquire(IMG, p, 6)
        assert np.array_equal(n1, n2) and np.array_equal(i1, i2)
        assert not np.array_equal(n1, n3)
        assert np.all(np.isfinite(n1)) and np.all(np.isfinite(i1))
        # ideal (noise-free) measurement does not depend on the seed, except MRI where the seed
        # also chooses WHICH outer k-space lines are sampled (the sampling pattern is random)
        if name != "MRI":
            assert np.array_equal(i1, m.acquire(IMG, p, 99)[0])


def test_every_metric_is_finite_and_reconstruction_has_no_nan():
    for name in modality_names():
        m = get_modality(name)
        for proto in m.training_grid()[::7]:
            q, recon = _quality(m, proto)
            assert np.all(np.isfinite(recon)) and all(np.isfinite(v) for v in q.values())


def test_more_cost_gives_better_quality_on_average():
    for name in ("CT", "X-ray", "MRI", "Ultrasound"):
        m = get_modality(name)
        cheap = min(m.training_grid(), key=lambda p: m.acquisition_cost(p).cost)
        q_cheap = np.mean([_quality(m, cheap, seed=s)[0]["ssim"] for s in range(3)])
        q_ref = np.mean([_quality(m, m.reference_protocol(), seed=s)[0]["ssim"] for s in range(3)])
        assert q_ref > q_cheap, name


def test_xray_low_photon_budget_is_noisier():
    m = get_modality("X-ray")
    hi = _quality(m, {"photon_budget": 1e5, "post_filter": "none", "filter_strength": 1})[0]["rmse"]
    lo = _quality(m, {"photon_budget": 1e2, "post_filter": "none", "filter_strength": 1})[0]["rmse"]
    assert lo > hi


def test_xray_ideal_roundtrip_recovers_the_image():
    m = get_modality("X-ray")
    q, recon = _quality(m, m.reference_protocol(), ideal=True)
    assert q["ssim"] > 0.999 and np.allclose(recon, IMG, atol=1e-6)


def test_mri_acceleration_degrades_and_nex_lowers_noise():
    m = get_modality("MRI")
    r1 = _quality(m, {"acceleration": 1, "nex": 4, "kspace_filter": "none"}, ideal=True)[0]["ssim"]
    r8 = _quality(m, {"acceleration": 8, "nex": 4, "kspace_filter": "none"}, ideal=True)[0]["ssim"]
    assert r1 > 0.999 > r8                                          # full sampling noise-free = exact; R=8 aliases
    n1 = np.mean([_quality(m, {"acceleration": 1, "nex": 1, "kspace_filter": "none"}, seed=s)[0]["rmse"] for s in range(3)])
    n4 = np.mean([_quality(m, {"acceleration": 1, "nex": 4, "kspace_filter": "none"}, seed=s)[0]["rmse"] for s in range(3)])
    assert n4 < n1


def test_mri_sampling_mask_keeps_centre_and_hits_acceleration():
    m = get_modality("MRI")
    _, _, meta = m.acquire(IMG, {"acceleration": 4, "nex": 1, "kspace_filter": "none"}, 1)
    mask = meta["plan"]["mask"]
    assert abs(mask.sum() - 64 / 4) <= 1 and mask[32] and mask[28:36].sum() >= 6


def test_ultrasound_compounding_reduces_speckle():
    m = get_modality("Ultrasound")
    flat = np.full((64, 64), 0.5)
    sd = {}
    for n in (1, 16):
        p = {"n_frames": n, "log_compression": "off", "post_filter": "none", "filter_strength": 1}
        _, noisy, meta = m.acquire(flat, p, 2)
        sd[n] = float(m.reconstruct(noisy, p, meta)[0].std())
    assert sd[16] < 0.5 * sd[1]                                     # speckle std falls ~ 1/sqrt(N)
    # frames of N=2 are the first two frames of N=8 with the same seed
    a = m.acquire(flat, {"n_frames": 2, "log_compression": "off", "post_filter": "none", "filter_strength": 1}, 4)[1]
    b = m.acquire(flat, {"n_frames": 8, "log_compression": "off", "post_filter": "none", "filter_strength": 1}, 4)[1]
    assert np.array_equal(a, b[:2])


def test_other_modality_is_labelled_as_a_proxy():
    o = get_modality("Other")
    assert "PROXY" in o.model_scope_text.upper() and "proxy" in o.cost_labels["pipeline"].lower()


def test_cost_never_claims_dose():
    for name in modality_names():
        m = get_modality(name)
        text = (m.model_scope_text + " ".join(m.cost_labels.values())).lower()
        assert "msv" not in text.replace("not dose in msv", "").replace("not patient dose", "") or "not" in text
        assert "dose" not in " ".join(m.cost_labels.values()).lower()


def test_disclaimer_is_generalized_and_present():
    assert "image-acquisition" in RESEARCH_DISCLAIMER and "scanner/protocol prescription" in RESEARCH_DISCLAIMER


def test_ct_wrapper_matches_original_pipeline_exactly():
    from simulation.acquisition import simulate_acquisition

    ct = get_modality("CT")
    p = ct.default_protocol()
    ideal, noisy, _ = ct.acquire(IMG, p, 7)
    n2, i2, _, _ = simulate_acquisition(IMG, p["num_projections"], p["photon_budget"], 7)
    assert np.array_equal(noisy, n2) and np.array_equal(ideal, i2)
