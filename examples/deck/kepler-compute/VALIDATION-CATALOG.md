<!--
SPDX-License-Identifier: MIT
SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
-->

# Validation, regression and covariance catalog

Layers → Analysis → Contributor catalog exposes twelve additional GPU operations. Each form accepts built-in samples, pasted GeoJSON or a native dataset with a geometry field. Numeric properties are selected by name. Run results show the dispatched contributor, row values and a separate diagnostics table; every table can be staged in Sources and published. Invocations preserve parameter values and input content hashes.

| Operation | Inputs and controls | Tables |
| --- | --- | --- |
| GTWR / multiscale GTWR | Response, comma-separated predictors, numeric time in seconds, integer Unix-second origin, kernel, fixed bandwidth and time scale. Multiscale adds per-component bandwidth/time-scale lists (intercept first) and backfitting iterations. | Local coefficients, fitted values, residuals, statuses; GTWR fit statistics or per-iteration backfitting scores. |
| Space-time kriging | Sample values, optional observation error variances, explicit query points/times, metric covariance, anisotropy, simple/ordinary mean. | Prediction, latent conditional variance, pivot ratio, retained neighbors, status; model and solve counts. |
| Vecchia likelihood / Fisher / space-time likelihood | Known-mean residuals, optional observation error variances, covariance and conditioning-neighbor count. Enable **Fit covariance parameters** for the library optimizer. | Site terms/statuses; conditionals or Fisher scores/information; total likelihood, covariance, convergence, work used and failure counts. |
| Area of applicability | Separate training and query predictor rows. | Query dissimilarity/applicability, training dissimilarity and training threshold/status diagnostics. |
| Split / geographic conformal intervals | Separate held-out calibration observations/predictions and query predictions; optional positive scales, miscoverage. Geographic adds neighbor count, Gaussian spatial bandwidth and query self weight. | Lower/upper endpoints, status bits; calibration and unbounded/invalid counts. |
| Ensemble scores | Observation and comma-separated ensemble member columns, optional fair correction and variogram order. | Per-site CRPS; field energy and variogram scores. Variogram pairs are consecutive source rows in both directions. |
| Gaussian scores | Observations, predictive means and predictive variances. | Per-site CRPS; scored/invalid counts and mean CRPS. |
| FDR mask | One point per test or raster cell center, p-value column, adjustment method and alpha. | Adjusted p-values and significance; tested/significant counts and critical p-value. |

Comma-separated lists allow up to 11 regression predictors, 16 applicability predictors or 64 ensemble members, with at most 64 characters per name. The predictor name `intercept` is reserved in the regression forms. Geographic variogram fitting uses the same validity bounds as the empirical fitter (longitude ±180°, latitude ±90°), subject to finite local projection. Space-time Matern fits enforce the nugget floor against total sill; inadmissible optimizer proposals are penalized.

Catalog inputs are limited to 2,048 rows each to bound exhaustive work. Spatial operations use a shared local meter frame for geographic data, or native planar units. Numeric times subtract the declared origin before Float32 rounding. Missing numeric values become NaN inputs; contributor statuses determine exclusions. Source indices remain unchanged. Nonfinite outputs, including unbounded conformal endpoints, become native nulls with status bits. Conformal calibration must use held-out predictions; geographic weighting does not establish distribution-free spatial coverage. Applicability describes predictor support. Kriging variance is conditional on the covariance and omits fitting uncertainty.

**Configure ordinary kriging grid** also offers GPU Vecchia likelihood and Fisher-scoring variogram fitting. These use a fitted constant mean, eight earlier spatial neighbors in source order, exponential or Matern 3/2 covariance and a local meter frame, for 3–2,048 valid samples with a varying response. Fewer than three valid samples or a response with no variance are rejected before GPU fitting with a specific error message. Fit diagnostics show convergence and work used. Apply commits explicit covariance values and the fitting method to the replayable recipe. The manual model menu also includes Matern 3/2 and Matern 5/2. Space-time covariance fitting stays in the catalog because the ordinary grid has no time axis. The space-time wrapper uses a metric joint covariance with fixed anisotropy; other library covariance families and automatic GTWR bandwidth search remain outside these forms.

Validation coverage is in `modules/kepler/test/catalog-validation.node.spec.ts` and `catalog-validation-gpu.spec.ts`. Browser scripts `scripts/analysis-catalog.mjs`, `catalog-discovery.mjs` and `kriging-analysis.mjs` take an optional URL (default `http://127.0.0.1:5318/?rows=256`). The GPU and browser checks require the driver's WebGPU Chromium environment.
