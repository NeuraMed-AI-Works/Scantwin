# Equations

> Research and educational prototype; simplified models, not validated against any scanner.

## CT (and "Other" proxy)
Line integrals p(θ,s) = ∫ μ dl (Radon transform). Photons I ~ Poisson(I0 · exp(−p)). Recovered p̂ = −ln(max(I,1)/I0).
Reconstruction: FBP (ramp, shepp-logan, cosine, hamming, hann filters) or SART. Cost = 0.7 I0/I0ref + 0.3 N/Nref.

## X-ray
p = k · x with k = 4 / max(x). I ~ Poisson(I0 · exp(−p)). x̂ = −ln(max(I,1)/I0)/k, optional post-filter (Gaussian / median).
Cost = I0/I0ref.

## MRI
K = F{x} (centred, orthonormal). Sampled lines S (centre block + random outer). K_meas = S · (K + σ (n_re + i n_im)), σ = σ0/√NEX.
x̂ = |F⁻¹{ S · K_meas · w }| with w = 1 or a Hann window. Cost = (1/R) · NEX/NEX_ref.

## Ultrasound
Frame f: y_f = x · e_f + ε_f, e_f ~ Exp(1), ε_f ~ N(0, 0.01²). ŷ = mean_f y_f (variance of speckle ∝ 1/N). Optional log compression
g(y) = log(1 + G y)/log(1 + G), G = 100, and post-filter (median / anisotropic Perona-Malik). Cost = N/Nref.

## Metrics
SSIM (skimage, data range = max − min of truth), PSNR = 20 log10(range/RMSE) capped at 100 dB, RMSE, CNR = |μ_target − μ_bg| / σ_bg
(n/a when no target exists; if σ_bg ≈ 0 a finite ceiling of 100 is returned when means differ, else 0). Relative simulated cost reduction = (1 − cost_chosen/cost_ref) · 100 %.
