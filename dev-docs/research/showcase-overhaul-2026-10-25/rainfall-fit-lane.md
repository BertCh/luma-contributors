# Rainfall statistics CLI lane

## Independent numerical verifier inputs

The backend comparison remains in the existing `rainfall-interpolation` story. Its inputs are the
selected GHCN-Daily variable, the `robust` estimator toggle, current maximum lag, and the existing
`weighting` and `variogramModel` controls. `GPUVariogram` produces four direction sectors and 16
lags. For Matheron semivariance and mean distance, the omnidirectional reduction remains the
pair-count-weighted sector mean.

For robust Cressie–Hawkins bins, the sector semivariance is not pooled by an arithmetic mean. For
each finite sector bin with positive pair count `n`, the reduction recovers
`rootMean = (2 * (0.457 + 0.494 / n) * sectorGammaCH)^(1/4)`, pools `n * rootMean` over total pair
count `N`, then computes `gammaCH = (pooledRootMean)^4 / (2 * (0.457 + 0.494 / N))`. The formula
matches the contributor formula documented in `modules/graph/src/gpu-dataframe/pair-statistics/gpu-variogram.ts`.
The selected empirical distances, semivariances and pair counts are rounded once to Float32/UInt32;
the identical arrays feed `fitVariogramModel` and all GPU fits.

The GPU search uses 200 log-range steps and 60 refinement iterations over `[minimum positive lag /
10, maximum lag * 3]`, matching the CPU fitter's domain and iteration count. GPU output row order is
`spherical`, `exponential`, `gaussian`. For each row, `parameters` is `[nugget, partial sill,
effective range, weighted residual]`; `diagnostics` is `[usable bins, unweighted SSE, AIC, relative
log-range bracket]`; `status` is 0 interior solution, 1 too few usable bins, 2 numerical failure,
or 3 boundary/iteration-budget candidate. Status 3 is not an identified range; statuses 1 and 2 are
withheld. AIC describes bin residuals only. It is not a spatial likelihood and should not be
compared across different empirical bins.

The `variogramInputBins` readout contains one exact Float32 input row per lag as
`[distance_m, semivariance, pair_count]`. `fitComparison` exposes CPU and selected GPU
`[nugget, sill, range_m, weighted_residual]` to seven significant digits, alongside status,
diagnostics and human-readable differences. `variogramChart` overlays the CPU and GPU fits and
provides its chart table. The omnidirectional reduction runs on the CPU after the existing small
variogram readback; the bounded inputs are uploaded to the separate GPU fit graph. This contribution
is source-wired only; it has not been GPU-executed or numerically accepted in a browser.

## Conditional leave-one-out diagnostics

The `kriging` story step uses `addInterpolationDiagnosticsRecipe` with the selected variable, gauge
positions, ordinary-kriging parameters, neighborhood size and search settings. A request is allowed
only after an actual CPU variogram fit exists, current empirical bins have completed readback,
`variogramReader` is idle, and pair counts have not overflowed. An unavailable fit leaves the
request dirty and reports why; fallback kriging parameters never feed LOO. Activating hold-out
validation pauses the diagnostic.

`leaveOneOutChart` is an observed-versus-predicted scatter in common measurement units, and its
numeric table exposes both values. `leaveOneOutValues` provides every station's source identifier,
observed value, prediction, prediction variance, residual and standardized residual to seven
significant digits. `leaveOneOutMetrics` repeats the first three CSV source identifiers as fixed
probes and reports bias, MAE, RMSE and valid counts. The status reports recipe invalid/uncertain
counts and the count with positive prediction variance; standardized values are only meaningful
when variance is positive. No probabilities or Gaussian coverage claims are inferred.

The variogram is fitted on the selected gauge set before leave-one-out prediction. These are
conditional diagnostics, not leakage-resistant spatial cross-validation or calibrated deployment
error. The separate random and spatially blocked hold-out methods remain separate.

## Scoped verification

- Rainfall-only TypeScript check passed for the compute and scene files with the showcase compiler settings.
- The full showcase TypeScript check was attempted and currently fails in unrelated
  `statistics/chicago-population-support` files: four union-type errors for `built-up` and
  `uniform-area` values.
- Scoped Biome check/write passed for `rainfall-interpolation.compute.ts` and `rainfall-interpolation.scene.ts`.
- Story-step check passed for `interpolation`; strict cartography check passed for `interpolation`.
- No GPU or browser acceptance claim; neither was run.
