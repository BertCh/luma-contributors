# Pooled county spatial-IV covariance sensitivity

## Model and construction

The `county-panel` story adds a covariance-only sensitivity for its pooled, 63-row spatial 2SLS fit. The structural table is `[poverty, log density, W y]`; the excluded-instrument table is `[poverty, log density, W poverty, W log density]`, with an implicit intercept in both. A GPU pass constructs `W y` and `W X` by applying the exact row-standardized Queen CSR rows used by the panel estimator to the corresponding county observations in each decade. The covariance contributor reads the pooled model's actual structural residual buffer. It is not applied to the separate unit-effects 2SLS fit or to the likelihood estimators.

Four GPU covariance outputs are computed from that same point estimate and residual vector:

- HC0;
- HC1, multiplying HC0 by `63 / (63 - 4) = 63 / 59`, where four coefficients include the intercept;
- full dense Gaussian space-time HAC kernels with 50 km or 100 km spatial scale and a 10-year temporal scale.

The HAC kernel is `exp(-d² / (2 h²) - Δt² / (2 × 10²))`, evaluated between the 63 rows using the New Jersey polygons' interior label anchors projected to the scene's azimuthal-equidistant metre frame. The temporal coordinates are the three decennial periods. This is a product of Gaussian kernels on Euclidean spatial coordinates and time, so the unrounded kernel is positive semidefinite; float32 storage and computation are an implementation approximation. The kernels are full, symmetric, have unit diagonal, and are separate from the model's Queen matrix.

## Interpretation and limitations

Changing the selector changes only the covariance matrix and its diagonal standard errors; it does not refit or select a different estimator. HC0/HC1 do not account for serial or spatial dependence. The 50 km, 100 km, and 10-year scales are authored sensitivity values, not fitted or validated bandwidths. Matrix entries and standard errors are descriptive; the scene does not threshold p-values or claim calibrated coverage, valid instruments, or causal identification. The series has only 21 units and three observed decades.

The method follows the distinction between spatial-lag IV point estimation and robust covariance in PySAL's [spatial-lag IV tutorial](https://pysal.org/spreg/notebooks/14_IV_estimation_spatial_lag.html) and [`GM_Lag` API](https://pysal.org/spreg/generated/spreg.GM_Lag.html). An independent NumPy reference is recorded in `reference-county-iv-covariance.py` and `county-iv-covariance-reference.json` in this research directory; those calculations are a numerical comparison target, not production GPU acceptance.
