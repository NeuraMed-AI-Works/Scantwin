# Assumptions (per modality "model scope")

> Research and educational prototype. ScanTwin uses simplified image-acquisition and exposure models and is
> not intended for clinical diagnosis, patient-specific dosimetry, or scanner/protocol prescription.

ScanTwin does not simulate real scanners. Each modality is a deliberately small image-formation model.
All images are 2D arrays in [0, 1] (an *intensity proxy*). All costs are *proxies normalized so the
reference protocol = 1.0*; none is a dose in mSv/Gy and none is a real scan time.

## Shared
* The "ground truth" is the loaded or synthetic image itself. It is the image the simulator tries to recover, not anatomy.
* Same seed + same parameters = identical noise; a different seed = a different noise realization.
* Quality = SSIM, PSNR (capped at 100 dB), RMSE, and CNR of an inserted synthetic low-contrast target.
  CNR is **n/a** when no target exists; it is never replaced by 0.
* Non-finite numbers (NaN/Inf) are rejected or shown as n/a, never passed on silently.
* Noise levels (photon budget, sigma0, speckle) are arbitrary round numbers chosen for a readable range. They are not calibrated to any device.
* Default criteria (min SSIM / PSNR / CNR) are experimental starting values, editable in the sidebar. They are **not** clinical acceptance thresholds.
* The feasibility decision is made by exactly one function (`digital_twin.twin.is_protocol_feasible`).

## CT
Parallel-beam Radon transform, monochromatic Beer-Lambert, Poisson counts, FBP (several filters) or SART.
No scatter, beam hardening, fan/cone geometry, detector response, motion. Exposure proxy = photon budget I0.
Cost = 0.7 * I0/I0_ref + 0.3 * views/views_ref.

## X-ray
Image treated as a line-integral map p = k * image (max 4); Poisson counts; inverse log; optional post-filter
(none / gaussian / median). No scatter, grid, spectrum, detector blur or electronic noise. Cost = I0/I0_ref.

## MRI
Magnitude image -> centred orthonormal FFT -> Cartesian undersampling (fully sampled centre block of 50 % of the
acquired lines, at least 8, plus random outer lines) -> complex Gaussian noise sigma0/sqrt(NEX) -> zero-filled
inverse FFT magnitude (optional Hann window). No coils, parallel imaging, compressed sensing, T1/T2 physics, motion
or field inhomogeneity. Cost = (1/R) * NEX / NEX_ref, a scan-time proxy.

## Ultrasound
Speckle statistics only: echogenicity * Exp(1) speckle per frame + small electronic noise (0.01), mean of N frames,
optional post-filter (median / anisotropic) and log compression. Not an acoustic simulation: no depth attenuation,
beamforming, PSF, probe geometry or TGC. Metrics are computed in the displayed domain. Cost = N / N_ref.

## Other
A generic proxy that reuses the CT-style projection model. It describes how this simulated chain degrades an image,
not how the original imaging device behaves.

## Learning
* The surrogate learns from **simulated** outcomes on the user's own images; it learns the simulator, not a scanner.
* Models are per modality (>= 3 images). With fewer images a pooled model with modality as a feature is used and flagged.
* A recommendation is reported only if the digital twin simulation confirms it with `is_protocol_feasible`; otherwise the message is
  "No feasible protocol found within the current search space."
* No disease classification is performed anywhere.
