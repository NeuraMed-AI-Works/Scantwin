# ScanTwin data folder (intentionally minimal)

ScanTwin ships **no patient data** and **downloads nothing automatically**.

Built in (no files needed): Shepp-Logan, concentric disks, a 5-slice synthetic series, and a
built-in synthetic demo set for the "Train from my images" workflow.

To use real images drop files here or upload them in the app:

| Format | Notes |
|---|---|
| `.dcm` | needs `pydicom`; converted to HU only if the selected modality is CT, then a fixed window -> [0, 1] |
| `.nii` / `.nii.gz` | needs `nibabel` (optional); 3D volumes -> up to 8 axial slices with ONE shared intensity mapping |
| `.npy` / `.npz` | 2D slice or 3D stack (max 8 slices kept); percentile-normalized |
| `.png` `.jpg` `.tif` | 8/16-bit; percentile-normalized |

## Quick-start sources (you download them yourself)

* **MedMNIST+ on Zenodo** - small 2D/3D image sets in `.npz`; the `_64` files (64x64) are the
  convenient size. Use the `train_images` array of a 2D set (for example PneumoniaMNIST = chest
  X-ray, BreastMNIST = ultrasound, OrganAMNIST/OrganCMNIST/OrganSMNIST = CT). Check each set's licence
  (several are CC BY 4.0; some sets carry other terms).
* **Medical Segmentation Decathlon (MSD)** - NIfTI volumes. MRI tasks: Task01 BrainTumour,
  Task02 Heart, Task04 Hippocampus, Task05 Prostate. CT tasks: Task03 Liver, Task06 Lung,
  Task07 Pancreas, Task08 HepaticVessel, Task09 Spleen, Task10 Colon. Licence: CC BY-SA 4.0.

The app's quick-start panel lists these sources and links; **it never fetches anything for you.**

Only use data that is public, de-identified and licensed for your use. Images are resampled to the
chosen grid and zero-padded to square. The value range becomes an *intensity proxy in [0, 1]*,
not a calibrated physical quantity. Research/education only; not for clinical use.

Sample CT slice (optional, run once on a machine with internet):

    pip install pydicom
    python scripts/fetch_sample_ct.py
