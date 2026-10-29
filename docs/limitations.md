# Limitations

> Research and educational prototype; not for clinical diagnosis, patient-specific dosimetry, or scanner/protocol prescription.

1. **Simplified physics.** Every modality omits most real effects (see `assumptions.md`). Absolute quality numbers describe the simulator only.
2. **No dose.** "Relative exposure proxy" and "normalized acquisition cost" are unit-free ratios to a reference protocol. They are not mSv, Gy or seconds.
3. **Arbitrary noise levels.** Photon budgets, MRI sigma0 and ultrasound speckle/electronic noise are round numbers, not calibrated.
4. **Modalities are not comparable with each other.** SSIM = 0.9 in X-ray and in ultrasound mean different things.
5. **Images are treated as ground truth.** A low-quality or already-compressed input limits what the twin can show. Real images are normalized to an intensity proxy; real attenuation, T1/T2 or echogenicity values are not recovered.
6. **Learned model scope.** The random-forest surrogate predicts simulated SSIM/PSNR from image descriptors and protocol parameters. It can be wrong for images unlike its training set; the app shows a novelty check, a modality-mismatch warning, and grouped cross-validation against a mean baseline. It is never used alone: every recommendation is re-simulated.
7. **Small data.** Training needs >= 3 images per modality (8+ recommended). Pooled fallbacks carry weaker evidence.
8. **Only 2D slices.** Volumes are reduced to at most 8 axial slices.
9. **Synthetic target.** CNR refers to a synthetic low-contrast disk inserted by the app, not a real lesion. ScanTwin does not detect or classify disease.
10. **Not validated.** No comparison with measured scanner data has been made.
11. **Model files.** Saved `.joblib` models are Python pickles; loading one executes code. Load only files you created yourself.
12. **Data.** Use only public, de-identified, licensed data. ScanTwin ships and downloads none.
13. **Environment.** The UI was developed against Streamlit/Optuna APIs; run `pytest` and the app on your machine to confirm your installed versions.
