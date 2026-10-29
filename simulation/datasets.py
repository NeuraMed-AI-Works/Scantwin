"""
Minimal Dataset Module
======================
Supplies the digital twin with ground-truth images (any modality) from a deliberately small set of sources:

* ``synthetic``  - Shepp-Logan / disk phantoms and a synthetic series (always available).
* ``real``       - real images that YOU place in ``data/`` or upload in the app
                   (``.dcm``, ``.nii`` / ``.nii.gz`` via optional nibabel, ``.npy`` 2D/3D, ``.npz`` MedMNIST bundles,
                   ``.png``, ``.jpg``, ``.tif``).

No dataset is downloaded automatically. ScanTwin does not ship patient data. Use only public,
de-identified or licensed images.

Normalization (conceptually separate from display normalization, see utils/normalization.py)
-------------------------------------------------------------------------------------------
* CT DICOM with RescaleSlope/Intercept (or any file loaded with ``is_hu=True``) -> Hounsfield units -> fixed
  soft-tissue window [HU_WINDOW_MIN, HU_WINDOW_MAX] mapped linearly to [0, 1].
* Everything else (MRI, X-ray, ultrasound, photographs, MedMNIST ...) -> if already within [0, 1] it is kept;
  otherwise ONE robust percentile stretch (0.5 - 99.5 of the non-zero voxels) over the WHOLE stack / volume is used,
  so slices of one volume keep their relative brightness.
* The result is an *intensity proxy* in [0, 1], not a calibrated attenuation / signal / echogenicity value.
* Images are zero-padded to square and resampled (anti-aliased, bilinear) to the chosen grid. For projection-based
  modalities (CT / Other) pixels outside the inscribed circular field of view are set to 0 because the Radon model
  uses ``circle=True``; MRI, X-ray and ultrasound keep the full square.

DISCLAIMER: research/educational use only; not for clinical diagnosis.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import ndimage
from skimage.transform import resize

from simulation.phantom import generate_shepp_logan, generate_disk_phantom

DATA_DIR: Path = Path(__file__).resolve().parent.parent / "data"
SUPPORTED_EXTENSIONS = (".dcm", ".nii", ".nii.gz", ".npy", ".npz", ".png", ".jpg", ".jpeg", ".tif", ".tiff")

# Fixed soft-tissue-like HU window (centre 40, width 400) for DICOM input.
HU_WINDOW_MIN: float = -160.0
HU_WINDOW_MAX: float = 240.0

MAX_STACK_SLICES: int = 8  # keeps datasets minimal and live streaming fast
AIR_THRESHOLD: float = 0.05  # same air cutoff used by the lesion background mask


@dataclass
class SliceData:
    """A single ground-truth slice plus provenance."""

    image: np.ndarray
    name: str
    source: str  # "synthetic" | "real" | "image"
    note: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_real(self) -> bool:
        return self.source in ("real", "real_ct")


# ----------------------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------------------
def circular_fov_mask(size: int) -> np.ndarray:
    """Boolean mask of the reconstruction circle, defined exactly as scikit-image's radon(circle=True) does."""
    centre = size // 2
    coords = np.arange(size) - centre
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    return (yy**2 + xx**2) <= (size // 2) ** 2


def _fit_to_size(image: np.ndarray, size: int) -> np.ndarray:
    """Resize (anti-aliased) to a square grid, zero-padding to keep the aspect ratio."""
    h, w = image.shape
    side = max(h, w)
    padded = np.zeros((side, side), dtype=np.float64)
    r0, c0 = (side - h) // 2, (side - w) // 2
    padded[r0 : r0 + h, c0 : c0 + w] = image
    if side != size:
        padded = resize(padded, (size, size), order=1, mode="reflect", anti_aliasing=True, preserve_range=True)
    return padded


def _finalize(image01: np.ndarray, size: int, fov_circle: bool = True) -> np.ndarray:
    fitted = _fit_to_size(image01, size)
    fitted = np.clip(fitted, 0.0, 1.0)
    if fov_circle:
        fitted[~circular_fov_mask(size)] = 0.0
    return fitted.astype(np.float64)


def file_extension(name: str) -> str:
    """Lower-case extension, treating ``.nii.gz`` as one extension."""
    low = str(name).lower()
    return ".nii.gz" if low.endswith(".nii.gz") else Path(low).suffix


# ----------------------------------------------------------------------------------------
# Lesion placement for arbitrary anatomy
# ----------------------------------------------------------------------------------------
def suggest_lesion_center(
    image: np.ndarray,
    radius: float = 8.0,
    background_gap: float = 3.0,
    background_thickness: float = 5.0,
) -> Tuple[int, int]:
    """
    Pick a lesion centre deep inside soft tissue so the background annulus stays in the object.

    Candidates are body pixels whose distance to air exceeds the lesion + annulus extent. Among
    them, the centre with the most homogeneous ground-truth footprint (lowest local std) is
    chosen, so the heuristic works for any anatomy, including user-supplied real slices.
    """
    body = ndimage.binary_fill_holes(image > AIR_THRESHOLD)
    dist = ndimage.distance_transform_edt(body)
    needed = radius + background_gap + background_thickness
    candidates = dist >= needed
    if not np.any(candidates):               # object too small: fall back to the deepest pixel
        return tuple(int(v) for v in np.unravel_index(int(np.argmax(dist)), dist.shape))

    # Local mean/std over the lesion + annulus footprint. A homogeneous footprint keeps the
    # ground-truth background variance (and hence the CNR denominator) from being dominated by
    # anatomy edges rather than by noise.
    win = int(2 * np.ceil(needed) + 1)
    local_mean = ndimage.uniform_filter(image, size=win, mode="constant")
    local_sq = ndimage.uniform_filter(image**2, size=win, mode="constant")
    local_std = np.sqrt(np.maximum(local_sq - local_mean**2, 0.0))
    tissue = candidates & (local_mean > 2.0 * AIR_THRESHOLD)
    pool = tissue if np.any(tissue) else candidates
    score = np.where(pool, local_std, np.inf)
    # Tie-break towards the body interior (largest distance) among near-equally flat spots.
    flat = score <= (np.min(score) + 1e-3)
    idx = np.where(flat, dist, -np.inf)
    row, col = np.unravel_index(int(np.argmax(idx)), idx.shape)
    return int(row), int(col)


# ----------------------------------------------------------------------------------------
# Synthetic sources
# ----------------------------------------------------------------------------------------
def synthetic_series(size: int = 128, n_slices: int = 5, fov_circle: bool = True) -> List[SliceData]:
    """
    A small synthetic 'series': one Shepp-Logan base slice scaled and rotated slice-to-slice.

    This mimics anatomy changing along the scan axis for live-streaming demonstrations. It is
    SYNTHETIC and carries no anatomical meaning.
    """
    if n_slices < 1:
        raise ValueError("n_slices must be >= 1")
    base = generate_shepp_logan(size)
    centre = (size - 1) / 2.0
    out: List[SliceData] = []
    for i in range(n_slices):
        t = 0.0 if n_slices == 1 else i / (n_slices - 1)
        scale = 1.0 - 0.22 * abs(2.0 * t - 1.0)          # 1.00 (ends) -> 0.78 (middle)
        angle = np.deg2rad(-8.0 + 16.0 * t)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]]) / scale
        offset = np.array([centre, centre]) - matrix @ np.array([centre, centre])
        warped = ndimage.affine_transform(base, matrix, offset=offset, order=1, mode="constant", cval=0.0)
        out.append(
            SliceData(
                image=_finalize(warped, size, fov_circle),
                name=f"Synthetic slice {i + 1}/{n_slices}",
                source="synthetic",
                note="Synthetic series derived from Shepp-Logan (no anatomical meaning).",
                meta={"scale": float(scale), "rotation_deg": float(np.rad2deg(angle))},
            )
        )
    return out


def load_synthetic(kind: str, size: int, fov_circle: bool = False) -> SliceData:
    if kind == "Shepp-Logan":
        img = generate_shepp_logan(size)
    elif kind == "Concentric Disks":
        img = generate_disk_phantom(size)
    else:
        raise ValueError(f"Unknown synthetic phantom '{kind}'")
    if fov_circle:
        img = np.where(circular_fov_mask(size), img, 0.0)
    return SliceData(img, kind, "synthetic", "Analytical phantom.")


# ----------------------------------------------------------------------------------------
# Real-data readers
# ----------------------------------------------------------------------------------------
def _read_bytes(source: Union[str, Path, bytes]) -> Tuple[bytes, str]:
    if isinstance(source, (str, Path)):
        path = Path(source)
        return path.read_bytes(), path.name
    raise TypeError("source must be a path")


def _read_nifti(raw: bytes, ext: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Read a NIfTI volume (needs optional nibabel). Returns a 3D array (x, y, z) and metadata."""
    try:
        import nibabel as nib  # optional dependency
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("Reading .nii / .nii.gz files needs the optional 'nibabel' package: pip install nibabel") from exc
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"volume{ext}"
        path.write_bytes(raw)
        img = nib.load(str(path))
        vol = np.asarray(img.get_fdata(dtype=np.float32), dtype=np.float32)
        zooms = [float(z) for z in img.header.get_zooms()[:3]]
    if vol.ndim == 4:                       # (x, y, z, t) -> first volume
        vol = vol[..., 0]
    if vol.ndim != 3:
        raise ValueError(f"Expected a 3D NIfTI volume, got shape {vol.shape}")
    return vol, {"voxel_size_mm": zooms, "volume_shape": tuple(int(v) for v in vol.shape)}


def _select_axial_slices(vol: np.ndarray, max_slices: int = MAX_STACK_SLICES) -> np.ndarray:
    """Up to ``max_slices`` evenly spaced axial slices (last axis) that actually contain tissue, rotated for display."""
    finite = np.isfinite(vol)
    if not finite.all():
        raise ValueError("Image contains NaN/Inf values.")
    lo, hi = float(vol.min()), float(vol.max())
    if hi - lo < 1e-12:
        raise ValueError("Volume has no usable dynamic range.")
    thresh = lo + 0.05 * (hi - lo)
    content = np.array([(vol[:, :, k] > thresh).mean() for k in range(vol.shape[2])])
    valid = np.flatnonzero(content > 0.02)
    if len(valid) == 0:
        valid = np.arange(vol.shape[2])
    pick = valid[np.unique(np.linspace(0, len(valid) - 1, min(max_slices, len(valid))).astype(int))]
    return np.stack([np.rot90(vol[:, :, int(k)]) for k in pick]).astype(np.float64)
    try:
        import pydicom  # optional dependency
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "Reading .dcm files needs the optional 'pydicom' package: pip install pydicom"
        ) from exc
    ds = pydicom.dcmread(io.BytesIO(raw))
    arr = ds.pixel_array.astype(np.float64)
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):      # RGB DICOM -> luminance
        arr = arr[..., :3].mean(axis=-1)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    hu = arr * slope + intercept
    meta = {
        "modality": str(getattr(ds, "Modality", "?")),
        "pixel_spacing_mm": [float(v) for v in getattr(ds, "PixelSpacing", [])] or None,
        "rows": int(getattr(ds, "Rows", arr.shape[0])),
        "cols": int(getattr(ds, "Columns", arr.shape[-1])),
    }
    return hu, meta


def _read_image(raw: bytes) -> np.ndarray:
    from PIL import Image

    img = Image.open(io.BytesIO(raw))
    arr = np.asarray(img).astype(np.float64)
    if arr.ndim == 3:  # RGB(A) -> luminance proxy
        arr = arr[..., :3].mean(axis=-1)
    return arr


def _percentile_range(values: np.ndarray) -> Tuple[float, float]:
    """Robust (0.5 - 99.5) percentile range of the non-zero values (all values if everything is zero)."""
    flat = np.asarray(values).ravel()
    if flat.size > 4_000_000:                       # subsample huge volumes; the mapping needs no more precision
        flat = flat[:: flat.size // 4_000_000 + 1]
    nz = flat[flat != 0]
    use = nz if nz.size > 0.01 * flat.size else flat
    p_lo, p_hi = np.percentile(use, [0.5, 99.5])
    return float(p_lo), float(p_hi)


def _to_unit_range(stack: np.ndarray, is_hu: bool, ref_range: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """
    Map a (n, h, w) stack to [0, 1] using ONE mapping for the whole stack (or for the whole volume the stack was
    sampled from, via ``ref_range``).
    """
    if is_hu:
        return np.clip((stack - HU_WINDOW_MIN) / (HU_WINDOW_MAX - HU_WINDOW_MIN), 0.0, 1.0)
    lo, hi = float(np.min(stack)), float(np.max(stack))
    if ref_range is None and lo >= 0.0 and hi <= 1.0:
        return stack
    p_lo, p_hi = ref_range if ref_range is not None else _percentile_range(stack)
    if p_hi - p_lo < 1e-12:
        raise ValueError("Image has no usable dynamic range.")
    return np.clip((stack - p_lo) / (p_hi - p_lo), 0.0, 1.0)


def load_slices(
    source: Union[str, Path, bytes],
    size: int = 128,
    name: Optional[str] = None,
    npz_key: str = "train_images",
    fov_circle: bool = True,
    is_hu: Optional[bool] = None,
) -> List[SliceData]:
    """
    Load 1..MAX_STACK_SLICES real slices from a file path or raw bytes.

    Parameters
    ----------
    source : path or bytes
        File path, or raw bytes (then ``name`` must carry the extension, e.g. an uploaded file).
    size : int
        Target square grid size.
    npz_key : str
        Array used from an ``.npz`` bundle (``train_images`` or ``test_images`` for MedMNIST files).
    fov_circle : bool
        Zero everything outside the inscribed circle (needed by the parallel-beam CT / Other models only).
    is_hu : bool, optional
        Force (True) or forbid (False) the Hounsfield-unit window. ``None`` = automatic: only a CT DICOM is HU.
    """
    if isinstance(source, (bytes, bytearray)):
        if not name:
            raise ValueError("`name` (with extension) is required when loading from bytes.")
        raw, fname = bytes(source), name
    else:
        raw, fname = _read_bytes(source)
    ext = file_extension(fname)
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{ext}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}")

    meta: Dict[str, Any] = {}
    hu = bool(is_hu) if is_hu is not None else False
    ref_range: Optional[Tuple[float, float]] = None
    nifti_stack: Optional[np.ndarray] = None
    if ext == ".dcm":
        arr, meta = _read_dicom(raw)
        hu = bool(is_hu) if is_hu is not None else (meta.get("modality", "CT") == "CT")
        source_kind = "real"
    elif ext in (".nii", ".nii.gz"):
        vol, meta = _read_nifti(raw, ext)
        if not hu:
            ref_range = _percentile_range(vol)           # ONE mapping for the whole volume
        nifti_stack = _select_axial_slices(vol)
        arr = nifti_stack
        source_kind = "real"
    elif ext == ".npy":
        arr = np.load(io.BytesIO(raw), allow_pickle=False).astype(np.float64)
        source_kind = "real"
    elif ext == ".npz":
        # MedMNIST-style bundle: use `npz_key` (train_images / test_images) or the first array.
        bundle = np.load(io.BytesIO(raw), allow_pickle=False)
        key = npz_key if npz_key in bundle.files else bundle.files[0]
        arr = np.asarray(bundle[key]).astype(np.float64)
        if arr.ndim == 4:  # (n, h, w, channels) -> luminance
            arr = arr.mean(axis=-1)
        source_kind = "real"
    else:
        arr = _read_image(raw)
        source_kind = "image"

    if arr.ndim == 2:
        stack = arr[None]
    elif arr.ndim == 3:
        idx = np.unique(np.linspace(0, arr.shape[0] - 1, min(MAX_STACK_SLICES, arr.shape[0])).astype(int))
        stack = arr[idx]
    else:
        raise ValueError(f"Expected a 2D slice or 3D stack, got shape {arr.shape}")
    if not np.all(np.isfinite(stack)):
        raise ValueError("Image contains NaN/Inf values.")

    unit = _to_unit_range(stack, hu, ref_range)
    note = (
        "User-supplied real slice (HU window 400/40)." if hu
        else "User-supplied image/array, percentile-normalized to [0, 1]."
    )
    stem = fname[:-7] if fname.lower().endswith(".nii.gz") else Path(fname).stem
    return [
        SliceData(_finalize(unit[i], size, fov_circle), f"{stem} [{i + 1}/{len(unit)}]", source_kind, note, dict(meta))
        for i in range(len(unit))
    ]


def list_local_datasets(data_dir: Path = DATA_DIR) -> List[Path]:
    """Files placed in ``data/`` that ScanTwin can read."""
    if not data_dir.exists():
        return []
    return sorted(p for p in data_dir.iterdir() if file_extension(p.name) in SUPPORTED_EXTENSIONS)
