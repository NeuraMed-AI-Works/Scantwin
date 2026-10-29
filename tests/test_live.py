"""Tests for the real-time (streaming) acquisition engine."""

import numpy as np

from reconstruction.reconstruction_engine import reconstruct
from simulation.acquisition import simulate_acquisition
from simulation.datasets import synthetic_series
from simulation.lesion import insert_lesion
from simulation.live import (
    acquisition_order,
    default_view_steps,
    first_frame_meeting_criteria,
    stream_scan,
    stream_slices,
)
from simulation.phantom import generate_shepp_logan


def _truth(size=96):
    base = generate_shepp_logan(size)
    return insert_lesion(base, radius=5.0, contrast=0.1)[:3]


def test_acquisition_order_is_a_permutation_with_spread_prefixes():
    order = acquisition_order(64)
    assert sorted(order.tolist()) == list(range(64))
    prefix = np.sort(order[:8])
    gaps = np.diff(np.concatenate([prefix, [prefix[0] + 64]]))
    assert gaps.max() <= 20  # 8 views cover the 180 degrees without a big hole (uniform would be 8)


def test_view_steps_end_at_total_and_increase():
    steps = default_view_steps(180, 8)
    assert steps[-1] == 180 and steps == sorted(set(steps)) and steps[0] >= 4


def test_stream_scan_final_frame_equals_batch_acquisition():
    truth, lmask, bmask = _truth()
    frames = list(stream_scan(truth, lmask, bmask, 1e5, 60, view_steps=[10, 30, 60], seed=7))
    assert [f.views_acquired for f in frames] == [10, 30, 60]
    noisy, _, theta, _ = simulate_acquisition(truth, 60, 1e5, 7)
    batch, _ = reconstruct(noisy, theta)
    assert np.allclose(batch, frames[-1].recon)  # live final frame == normal scan, same seed


def test_stream_scan_is_reproducible_and_seed_sensitive():
    truth, lmask, bmask = _truth()
    kw = dict(photon_budget=1e4, total_views=40, view_steps=[20, 40])
    a = list(stream_scan(truth, lmask, bmask, seed=1, **kw))
    b = list(stream_scan(truth, lmask, bmask, seed=1, **kw))
    c = list(stream_scan(truth, lmask, bmask, seed=2, **kw))
    assert np.array_equal(a[-1].recon, b[-1].recon)
    assert not np.array_equal(a[-1].recon, c[-1].recon)


def test_more_views_generally_improve_quality_and_metrics_are_finite():
    truth, lmask, bmask = _truth()
    frames = list(stream_scan(truth, lmask, bmask, 1e5, 120, view_steps=[6, 120], seed=3))
    for f in frames:
        assert all(np.isfinite(v) for v in f.metrics.values())
        assert np.all(np.isfinite(f.recon))
    assert frames[-1].metrics["ssim"] > frames[0].metrics["ssim"]
    assert frames[-1].metrics["rmse"] < frames[0].metrics["rmse"]


def test_partial_sinogram_marks_unacquired_views():
    truth, lmask, bmask = _truth()
    f = list(stream_scan(truth, lmask, bmask, 1e5, 40, view_steps=[10, 40], seed=0))[0]
    acquired_cols = ~np.all(np.isnan(f.partial_sinogram), axis=0)
    assert acquired_cols.sum() == 10


def test_no_lesion_gives_nan_cnr_not_a_fake_number():
    truth = generate_shepp_logan(64)
    f = list(stream_scan(truth, None, None, 1e5, 30, view_steps=[30], seed=0))[0]
    assert np.isnan(f.metrics["cnr"])
    assert np.isfinite(f.metrics["ssim"])


def test_first_frame_meeting_criteria_uses_authoritative_logic():
    truth, lmask, bmask = _truth()
    frames = list(stream_scan(truth, lmask, bmask, 1e5, 90, view_steps=[6, 30, 90], seed=3))
    assert first_frame_meeting_criteria(frames, 1.1, 0.0) is None  # SSIM > 1 is impossible
    hit = first_frame_meeting_criteria(frames, 0.0, 0.0)
    assert hit is frames[0]


def test_stream_slices_one_frame_per_slice():
    series = synthetic_series(64, n_slices=3)
    frames = list(stream_slices(series, 5e4, 60, lesion_radius=4.0, seed=0))
    assert len(frames) == 3
    assert all(np.isfinite(f.metrics["ssim"]) and np.isfinite(f.metrics["cnr"]) for f in frames)
    assert frames[0].label != frames[1].label


def test_every_modality_streams_progressively_and_final_frame_equals_batch():
    from modalities import get_modality, modality_names
    from simulation.live import stream_modality

    truth, lmask, bmask = _truth(64)
    for name in modality_names():
        m = get_modality(name)
        proto = m.normalize_protocol(m.default_protocol())
        frames = list(stream_modality(m, truth, lmask, bmask, proto, seed=7, n_frames=5))
        assert len(frames) >= 2 and frames[-1].acquired == frames[-1].total
        assert [f.acquired for f in frames] == sorted(set(f.acquired for f in frames))
        for f in frames:
            assert all(np.isfinite(v) for v in f.metrics.values()) and np.all(np.isfinite(f.recon))
        _, noisy, meta = m.acquire(truth, proto, 7)
        batch, _ = m.reconstruct(noisy, proto, meta)
        assert np.allclose(batch, frames[-1].recon, atol=1e-9), name


def test_series_stream_works_for_non_ct_modalities():
    from modalities import get_modality
    from simulation.live import stream_series

    m = get_modality("MRI")
    frames = list(stream_series(m, synthetic_series(48, n_slices=2, fov_circle=False), m.default_protocol(), lesion_radius=4.0))
    assert len(frames) == 2 and all(np.isfinite(f.metrics["ssim"]) for f in frames)
