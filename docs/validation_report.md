# Validation report (generated)

> Research and educational prototype. ScanTwin uses simplified image-acquisition and exposure models and is not intended for clinical diagnosis, patient-specific dosimetry, or scanner/protocol prescription.

Generated 2026-10-26T10:28:46 by `scripts/generate_validation_report.py` (python 3.13.17, numpy 2.5.3). Phantom: built-in synthetic Shepp-Logan 64x64 with an inserted low-contrast target; 3 noise realizations per point (mean ± std). **Every number below was measured by this run; nothing is typed in.** Synthetic phantom only, so these numbers describe the simulator, not any scanner or patient data.

## CT

Reference protocol: `I0=1e+05, views=360, recon=FBP, filter=ramp` (normalized cost = 1.000).

- Same seed reproduces the noisy measurement exactly: **True**; different seed differs: **True**.
- Noise-free (ideal) pipeline at the reference protocol: SSIM = 0.9476, RMSE = 0.0383.

### Sweep of `photon_budget` (other parameters at the reference)

| photon_budget | modality | n_realizations | cost | relative_exposure_pct | ssim_mean | ssim_std | psnr_mean | psnr_std | rmse_mean | rmse_std | cnr_mean | cnr_std |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1000 | CT | 3 | 0.307 | 1 | 0.9018 | 0.0008673 | 27.74 | 0.007636 | 0.04005 | 3.521e-05 | 0.8609 | 0.008879 |
| 3000 | CT | 3 | 0.321 | 3 | 0.9306 | 0.0003396 | 27.97 | 0.02423 | 0.03902 | 0.0001087 | 0.8531 | 0.007827 |
| 1e+04 | CT | 3 | 0.37 | 10 | 0.9424 | 0.0002876 | 28.08 | 0.01032 | 0.0385 | 4.575e-05 | 0.8489 | 0.006536 |
| 3e+04 | CT | 3 | 0.51 | 30 | 0.9459 | 0.0002059 | 28.11 | 0.002232 | 0.03838 | 9.864e-06 | 0.8518 | 0.002372 |
| 1e+05 | CT | 3 | 1 | 100 | 0.9469 | 3.725e-05 | 28.12 | 0.00217 | 0.03834 | 9.58e-06 | 0.851 | 0.0007559 |
| 3e+05 | CT | 3 | 2.4 | 300 | 0.9474 | 1.378e-05 | 28.12 | 0.001222 | 0.03834 | 5.397e-06 | 0.8502 | 0.001607 |
| 1e+06 | CT | 3 | 7.3 | 1000 | 0.9475 | 1.567e-05 | 28.12 | 0.0004987 | 0.03834 | 2.201e-06 | 0.851 | 0.000797 |

## X-ray

Reference protocol: `I0=1e+05, post=none` (normalized cost = 1.000).

- Same seed reproduces the noisy measurement exactly: **True**; different seed differs: **True**.
- Noise-free (ideal) pipeline at the reference protocol: SSIM = 1.0000, RMSE = 0.0000.

### Sweep of `photon_budget` (other parameters at the reference)

| photon_budget | modality | n_realizations | cost | relative_exposure_pct | ssim_mean | ssim_std | psnr_mean | psnr_std | rmse_mean | rmse_std | cnr_mean | cnr_std |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100 | X-ray | 3 | 0.001 | 0.1 | 0.7337 | 0.001093 | 27.8 | 0.1622 | 0.03978 | 0.0007477 | 0.6763 | 0.03157 |
| 300 | X-ray | 3 | 0.003 | 0.3 | 0.8709 | 0.001947 | 32.88 | 0.2426 | 0.02216 | 0.0006243 | 0.6262 | 0.04294 |
| 1000 | X-ray | 3 | 0.01 | 1 | 0.953 | 0.001056 | 38.22 | 0.2418 | 0.01198 | 0.0003368 | 0.713 | 0.007297 |
| 3000 | X-ray | 3 | 0.03 | 3 | 0.9835 | 0.0003673 | 43.07 | 0.08115 | 0.006854 | 6.398e-05 | 0.6887 | 0.01306 |
| 1e+04 | X-ray | 3 | 0.1 | 10 | 0.9949 | 8.494e-05 | 48.33 | 0.2588 | 0.003743 | 0.0001117 | 0.6935 | 0.005594 |
| 3e+04 | X-ray | 3 | 0.3 | 30 | 0.9982 | 7.039e-05 | 53.03 | 0.09545 | 0.002179 | 2.403e-05 | 0.6981 | 0.00304 |
| 1e+05 | X-ray | 3 | 1 | 100 | 0.9995 | 1.146e-05 | 58.24 | 0.09174 | 0.001196 | 1.268e-05 | 0.6962 | 0.0006348 |

## MRI

Reference protocol: `R=1, NEX=4, filter=none` (normalized cost = 1.000).

- Same seed reproduces the noisy measurement exactly: **True**; different seed differs: **True**.
- Noise-free (ideal) pipeline at the reference protocol: SSIM = 1.0000, RMSE = 0.0000.

### Sweep of `acceleration` (other parameters at the reference)

| acceleration | modality | n_realizations | cost | relative_exposure_pct | ssim_mean | ssim_std | psnr_mean | psnr_std | rmse_mean | rmse_std | cnr_mean | cnr_std |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | MRI | 3 | 1 | 100 | 0.8726 | 0.001433 | 40.45 | 0.06557 | 0.009267 | 6.987e-05 | 0.6941 | 0.004431 |
| 2 | MRI | 3 | 0.5 | 50 | 0.7114 | 0.02917 | 24.7 | 0.8068 | 0.05705 | 0.005114 | 0.6688 | 0.01644 |
| 3 | MRI | 3 | 0.3333 | 33.33 | 0.6358 | 0.0144 | 21.4 | 0.3316 | 0.08314 | 0.003131 | 0.6753 | 0.01733 |
| 4 | MRI | 3 | 0.25 | 25 | 0.5891 | 0.01267 | 20.03 | 0.2561 | 0.09737 | 0.002879 | 0.516 | 0.02788 |
| 6 | MRI | 3 | 0.1667 | 16.67 | 0.5781 | 0.003851 | 19.31 | 0.07353 | 0.1057 | 0.0008968 | 0.4917 | 0.0178 |
| 8 | MRI | 3 | 0.125 | 12.5 | 0.5957 | 0.001438 | 19.03 | 0.0003849 | 0.1092 | 4.838e-06 | 0.4894 | 0.006546 |

## Ultrasound

Reference protocol: `N=32, log=off, post=none` (normalized cost = 1.000).

- Same seed reproduces the noisy measurement exactly: **True**; different seed differs: **True**.
- Noise-free (ideal) pipeline at the reference protocol: SSIM = 1.0000, RMSE = 0.0000.

### Sweep of `n_frames` (other parameters at the reference)

| n_frames | modality | n_realizations | cost | relative_exposure_pct | ssim_mean | ssim_std | psnr_mean | psnr_std | rmse_mean | rmse_std | cnr_mean | cnr_std |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Ultrasound | 3 | 0.03125 | 3.125 | 0.4761 | 0.007332 | 13.17 | 0.05505 | 0.2144 | 0.001356 | 0.258 | 0.1521 |
| 2 | Ultrasound | 3 | 0.0625 | 6.25 | 0.5908 | 0.002757 | 16.01 | 0.1433 | 0.1546 | 0.00254 | 0.3686 | 0.1822 |
| 4 | Ultrasound | 3 | 0.125 | 12.5 | 0.6898 | 0.005001 | 19.15 | 0.3584 | 0.1078 | 0.004392 | 0.5066 | 0.1047 |
| 8 | Ultrasound | 3 | 0.25 | 25 | 0.7696 | 0.006098 | 21.97 | 0.3112 | 0.0779 | 0.002786 | 0.5388 | 0.02974 |
| 16 | Ultrasound | 3 | 0.5 | 50 | 0.8346 | 0.00162 | 24.73 | 0.2143 | 0.05667 | 0.001386 | 0.6315 | 0.02773 |
| 32 | Ultrasound | 3 | 1 | 100 | 0.8892 | 0.001078 | 27.83 | 0.1935 | 0.03964 | 0.0008806 | 0.6982 | 0.03026 |

## Other

Reference protocol: `I0=1e+05, views=360, recon=FBP, filter=ramp` (normalized cost = 1.000).

- Same seed reproduces the noisy measurement exactly: **True**; different seed differs: **True**.
- Noise-free (ideal) pipeline at the reference protocol: SSIM = 0.9476, RMSE = 0.0383.

### Sweep of `photon_budget` (other parameters at the reference)

| photon_budget | modality | n_realizations | cost | relative_exposure_pct | ssim_mean | ssim_std | psnr_mean | psnr_std | rmse_mean | rmse_std | cnr_mean | cnr_std |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1000 | Other | 3 | 0.307 | 1 | 0.9018 | 0.0008673 | 27.74 | 0.007636 | 0.04005 | 3.521e-05 | 0.8609 | 0.008879 |
| 3000 | Other | 3 | 0.321 | 3 | 0.9306 | 0.0003396 | 27.97 | 0.02423 | 0.03902 | 0.0001087 | 0.8531 | 0.007827 |
| 1e+04 | Other | 3 | 0.37 | 10 | 0.9424 | 0.0002876 | 28.08 | 0.01032 | 0.0385 | 4.575e-05 | 0.8489 | 0.006536 |
| 3e+04 | Other | 3 | 0.51 | 30 | 0.9459 | 0.0002059 | 28.11 | 0.002232 | 0.03838 | 9.864e-06 | 0.8518 | 0.002372 |
| 1e+05 | Other | 3 | 1 | 100 | 0.9469 | 3.725e-05 | 28.12 | 0.00217 | 0.03834 | 9.58e-06 | 0.851 | 0.0007559 |
| 3e+05 | Other | 3 | 2.4 | 300 | 0.9474 | 1.378e-05 | 28.12 | 0.001222 | 0.03834 | 5.397e-06 | 0.8502 | 0.001607 |
| 1e+06 | Other | 3 | 7.3 | 1000 | 0.9475 | 1.567e-05 | 28.12 | 0.0004987 | 0.03834 | 2.201e-06 | 0.851 | 0.000797 |

## Notes

- Differences between modalities in absolute SSIM are NOT comparable: the noise models and normalizations differ.
- Re-run the script after changing any modality constant; the report is overwritten with the new measurements.
