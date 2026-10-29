"""Tests for the minimal dataset module (synthetic series, file loaders, lesion placement)."""

import io

import numpy as np

from simulation.datasets import (
    circular_fov_mask,
    list_local_datasets,
    load_slices,
    suggest_lesion_center,
    synthetic_series,
)
from simulation.lesion import insert_lesion
from simulation.phantom import generate_shepp_logan


def _raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc:
        return True
    return False


def test_synthetic_series_shape_range_and_variation():
    series = synthetic_series(64, n_slices=5)
    assert len(series) == 5
    for sl in series:
        assert sl.image.shape == (64, 64)
        assert np.all(np.isfinite(sl.image))
        assert 0.0 <= sl.image.min() and sl.image.max() <= 1.0
        assert not sl.is_real
    assert not np.allclose(series[0].image, series[2].image)  # slices genuinely differ
    assert np.all(series[0].image[~circular_fov_mask(64)] == 0.0)


def test_npy_2d_loader_normalizes_and_masks_fov():
    base = generate_shepp_logan(96)
    buf = io.BytesIO()
    np.save(buf, base * 1500.0 - 300.0)  # arbitrary scale/offset, like raw CT values
    out = load_slices(buf.getvalue(), size=64, name="slice.npy")
    assert len(out) == 1 and out[0].image.shape == (64, 64)
    assert out[0].is_real
    assert 0.0 <= out[0].image.min() and out[0].image.max() <= 1.0
    assert np.all(out[0].image[~circular_fov_mask(64)] == 0.0)


def test_npy_stack_is_limited_and_shares_one_mapping():
    base = generate_shepp_logan(48)
    stack = np.stack([base, 0.5 * base] * 6)  # 12 slices -> 8 kept
    buf = io.BytesIO()
    np.save(buf, stack)
    out = load_slices(buf.getvalue(), size=48, name="stack.npy")
    assert len(out) == 8
    # one shared mapping: the half-intensity slice must stay dimmer than the full one
    assert out[1].image.max() < out[0].image.max()


def test_loader_rejects_bad_input():
    assert _raises(ValueError, load_slices, b"abc", 32, "x.xyz")
    assert _raises(ValueError, load_slices, b"abc", 32)  # bytes without a name
    bad = io.BytesIO()
    np.save(bad, np.array([[np.nan, 1.0], [0.0, 1.0]]))
    assert _raises(ValueError, load_slices, bad.getvalue(), 32, "nan.npy")


def test_lesion_center_is_in_tissue_and_background_is_homogeneous():
    for size, radius in [(128, 5.0), (256, 8.0)]:
        img = generate_shepp_logan(size)
        centre = suggest_lesion_center(img, radius=radius)
        truth, lmask, bmask, meta = insert_lesion(img, center=centre, radius=radius, contrast=0.08)
        assert lmask.any() and bmask.any()
        assert not (lmask & bmask).any()
        assert img[lmask].min() > 0.05  # lesion sits in tissue, not air
        assert truth[bmask].std() < 0.05  # ground-truth background is flat enough for CNR to mean something
        assert meta["true_ground_truth_contrast"] > 0.05


def test_list_local_datasets_handles_missing_dir(tmp_path=None):
    from pathlib import Path

    assert list_local_datasets(Path("/definitely/not/here")) == []


def test_fov_circle_is_optional_and_source_is_real():
    base = generate_shepp_logan(48)
    buf = io.BytesIO()
    np.save(buf, base)
    sq = load_slices(buf.getvalue(), size=48, name="s.npy", fov_circle=False)[0]
    ci = load_slices(buf.getvalue(), size=48, name="s.npy", fov_circle=True)[0]
    assert sq.is_real and sq.source == "real"
    assert np.all(ci.image[~circular_fov_mask(48)] == 0.0)


def test_nifti_extension_and_missing_nibabel_message():
    from simulation.datasets import file_extension

    assert file_extension("Scan.NII.GZ") == ".nii.gz" and file_extension("a.nii") == ".nii"
    try:
        import nibabel  # noqa: F401
    except ImportError:
        try:
            load_slices(b"x", 32, "v.nii.gz")
        except ImportError as exc:
            assert "nibabel" in str(exc)
        else:
            raise AssertionError("expected ImportError")


def test_nifti_slices_use_one_shared_mapping_with_fake_reader():
    import sys
    import types

    vol = np.zeros((40, 40, 30), np.float32)
    yy, xx = np.mgrid[:40, :40]
    for k in range(5, 25):
        vol[:, :, k] = ((yy - 20) ** 2 + (xx - 20) ** 2 < 150) * (100 + 10 * k)

    class _Hdr:
        def get_zooms(self):
            return (1.0, 1.0, 2.0)

    class _Img:
        header = _Hdr()

        def get_fdata(self, dtype=None):
            return vol.astype(dtype or np.float64)

    fake = types.ModuleType("nibabel")
    fake.load = lambda path: _Img()
    saved = sys.modules.get("nibabel")
    sys.modules["nibabel"] = fake
    try:
        out = load_slices(b"x", 48, "vol.nii.gz", fov_circle=False)
    finally:
        if saved is None:
            sys.modules.pop("nibabel", None)
        else:
            sys.modules["nibabel"] = saved
    assert 1 < len(out) <= 8
    peaks = [float(o.image.max()) for o in out]
    assert peaks == sorted(peaks) and peaks[-1] > peaks[0]          # brighter slices stay brighter: one mapping
