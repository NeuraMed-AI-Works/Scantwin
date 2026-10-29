"""Export pydicom's small public CT test slice to data/CT_small.npy (run locally, needs pydicom)."""
from pathlib import Path

import numpy as np

try:
    import pydicom
    from pydicom.data import get_testdata_file
except ImportError:
    raise SystemExit("Install first:  pip install pydicom")

src = get_testdata_file("CT_small.dcm")
if src is None:
    raise SystemExit("pydicom's CT_small.dcm test file was not found in this installation.")
ds = pydicom.dcmread(src)
hu = ds.pixel_array.astype(np.float64) * float(getattr(ds, "RescaleSlope", 1)) + float(getattr(ds, "RescaleIntercept", 0))
out = Path(__file__).resolve().parent.parent / "data" / "CT_small.npy"
out.parent.mkdir(exist_ok=True)
np.save(out, hu)
print(f"Saved {out}  shape={hu.shape}  HU range=({hu.min():.0f}, {hu.max():.0f})  modality={ds.Modality}")
