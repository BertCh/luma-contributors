# Model-based spatial statistics: code review

Scope: `modules/analysis/src/gpu-spatial-analysis/{spatial-regression, spatial-interpolation, point-process, study-window, spatial-validation, spatial-interaction, catchment-accessibility, location-allocation, segregation}` and `modules/analysis/src/dataframe-analysis/pair-statistics`, plus their specs. Read-only review: nothing was built, typechecked or run. Paths below are relative to `modules/analysis/src` unless they start with `test/` (that is, `modules/analysis/test`) or `dev-docs/`.

Tags: **CONFIRMED** means I traced the code path. **SUSPECTED** means the code path is real but the size of the numerical effect is my estimate and was not measured.

Depth: I read every `index.ts` and the main class of most operations. I went deeper into the spatial ML log-determinant, the kriging solvers, the GWR and MGWR solves, conditional simulation, the panels and the cross and space-time variograms. Location-allocation, segregation, catchment and the study window were only skimmed for their limits.

---

## 1. Inventory

Abbreviations: P = predictors without the intercept, n = rows, k = neighbours, SE = standard error, LOO = leave-one-out. "Spec" lists the main spec files only; many classes also appear in `test/analysis-core/kernel-audit/cases-*.ts`.

| Symbol | File | Operation | Solver | Capacity limits | Uncertainty output | Reference oracle | Main spec |
|---|---|---|---|---|---|---|---|
| `GPUOrdinaryLeastSquares` | spatial-regression/gpu-ordinary-least-squares.ts | OLS plus spreg diagnostics | Centred, correlation-scaled f32 Cholesky; Kahan tile merges; pivot tolerance 1e-3; precision warning when cond > 1e4 | P ≤ 15 | Coefficient SE, t, JB, BP, KB, VIF, Belsley index | spreg OLS (`spreg-ols-reference.ts`) | ordinary-least-squares*.spec.ts |
| `GPUSpatialLagMaximumLikelihood` | gpu-spatial-lag-maximum-likelihood.ts | spreg ML_Lag | Concentrated likelihood; 1025-point derivative grid plus 32 bisection steps; information matrix by 16-wide Cholesky | P ≤ 14; any n (chunked W); exact traces ≤ 4096 rows by default (≤ 16384 explicit); eigenvalue log-det ≤ 1024 rows | SE of (b, ρ); boundary, series-limited and approximation flags | spreg ML_Lag; f64 CPU mirror | spatial-lag-maximum-likelihood.spec.ts, spatial-maximum-likelihood.node.spec.ts |
| `GPUSpatialErrorMaximumLikelihood` | gpu-spatial-error-maximum-likelihood.ts | spreg ML_Error | Same search; λ–β covariance is zero by construction | P ≤ 15; same trace limits | SE of β and λ | spreg ML_Error | spatial-error-maximum-likelihood.spec.ts |
| `GPUSpatialErrorGM` | gpu-spatial-error-gm.ts | Kelejian–Prucha GM / homoskedastic GM | Moment equations; FGLS | P ≤ 15 | β SE; **λ has no SE** (doc line 183) | spreg GM_Error* | spatial-error-gm*.spec.ts |
| `GPUSpatialTwoStageLeastSquares` | gpu-spatial-two-stage-least-squares.ts | GM_Lag / 2SLS | Normal equations, Cholesky | P ≤ 8 | Base covariance only (HC/cluster are roadmap ST01) | spreg | spatial-two-stage-least-squares*.spec.ts |
| `GPUSpatialWeightsTraces` | gpu-spatial-weights-traces.ts | tr(W^q), 1'W^q 1 | Exact unit probes or 64 Rademacher (Hutchinson) probes | q ≤ 256; exact ≤ 16384 rows | Per-power Hutchinson SE | `oracles/statistics-spatial-weights-traces.py` | spatial-weights-traces*.spec.ts |
| `GPUSpatialLagImpacts` | gpu-spatial-lag-impacts.ts | LeSage–Pace impacts | Truncated series | q ≤ 256 | **Point estimates only** | spreg spat_impacts='full' | spatial-lag-impacts.spec.ts |
| `GPUSpatialRegressionDiagnostics` | gpu-spatial-regression-diagnostics.ts | Moran on residuals, LM tests | Moments | — | Test p-values | spreg | spatial-regression-diagnostics*.spec.ts |
| `GPUSpatialPanelRegression` | gpu-spatial-panel-regression.ts | Panel spatial-lag 2SLS (pooled / unit FE) | Within transform, then 2SLS on block W | Balanced, period-major; 2 ≤ T ≤ 1024; N·T < 2^24 | Homoskedastic SE | none named | panel-fixed-effects*.spec.ts |
| `GPUSpatialPanelErrorMaximumLikelihood` | gpu-spatial-panel-error-maximum-likelihood.ts | Panel_FE_Error | Within transform, then the error ML on I_T ⊗ W | Balanced; T ≤ 1024; N·T ≤ 65536 | SE; LM panel diagnostics | spreg Panel_FE_Error | coverage-estimators.spec.ts, statistics-upstream.spec.ts |
| `GPUSpatialPanelLagMaximumLikelihood` | gpu-spatial-panel-lag-maximum-likelihood.ts | Panel_RE_Lag / unit FE lag | Dense entity determinant; alternating φ / ρ | **N ≤ 32, T ≤ 32, P ≤ 7** | β/ρ covariance; Hausman | spreg 1.8.3 | statistics-advanced*.spec.ts |
| `GPUSeeminglyUnrelatedRegression` | gpu-seemingly-unrelated-regression.ts | SUR / SURerrorGM (one step) | Joint Cholesky ≤ 16 | 2–4 equations, P ≤ 7, n ≤ 65536 | Joint covariance | spreg SUR | ordinary-sur-parallel*.spec.ts |
| `GPUProbitRegression` | gpu-probit-regression.ts | Non-spatial probit plus spatial residual tests | Newton / IRLS | n ≤ 65536, P ≤ 11, ≤ 256 iterations | Expected-information SE | spreg Probit | probit-regression.node.spec.ts |
| `GPUSpatialRegimeRegression` / `GPURestrictedRegimeRegression` | gpu-*regime-regression.ts | OLS_Regimes | Centred within each regime, Cholesky | Restricted: P ≤ 7, ≤ 8 regimes; plain: ≤ 4096 regimes; n ≤ 65536 | Restricted: covariance, Chow, Wald; plain: **point estimates only** | spreg OLS_Regimes | geospatial-expansion.spec.ts, coverage-estimators.spec.ts |
| `GPUGeographicallyWeightedRegression` | gpu-geographically-weighted-regression.ts, gwr-fit-wgsl.ts | GWR with golden-section / AICc bandwidth | Design centred on the focal row; diagonally equilibrated f32 Cholesky of X'WX | P ≤ 11 (12 coefficients); n ≤ 65536 brute force, ≤ 1,048,576 indexed (bounded kernels only); adaptive k ≤ 128 | Local SE, t, hat, local R², optional local condition number | mgwr | geographically-weighted-regression*.spec.ts |
| `GPUGeneralizedGeographicallyWeightedRegression` | gpu-generalized-geographically-weighted-regression.ts | Poisson and logistic GWR | Bounded IRLS, equilibrated Cholesky | P ≤ 24; same row limits | Local SE, LOO | mgwr | adaptive-generalized-gwr*.spec.ts |
| `GPUMultiscaleGeographicallyWeightedRegression` | gpu-multiscale-geographically-weighted-regression.ts | MGWR backfitting | Dense O(iterations · P · n²) | ≤ 32 candidate bandwidths per variable; **full inference only for n ≤ 128** | Conditional SE (known σ²), or full-Jacobian SE / effective parameters (n ≤ 128) | mgwr 2.2.1 | mgwr-convergence.node.spec.ts, local-regression-limits.spec.ts |
| `GPULocalCollinearity` | gpu-local-collinearity.ts | mgwr local_collinearity | Jacobi eigenvalues | P ≤ 12 | VIF, VDP, condition index | mgwr | coverage-statistics.node.spec.ts |
| `GPUGeographicallyWeightedRegressionNonstationarityTest` | gpu-geographically-weighted-regression-nonstationarity-test.ts | Monte Carlo nonstationarity test | Equilibrated Cholesky | P ≤ 7 | p-values | — | geographically-weighted-regression-nonstationarity*.spec.ts |
| `GPUKriging` | spatial-interpolation/gpu-kriging.ts | Raster ordinary kriging | Normalized (k+1) system, partial-pivot Gaussian elimination; pivot ratio ≥ 1e-4 | k ≤ 16; ring walk over a grid index; per-frame variogram | Kriging variance, LOO prediction and variance, uncertain count | f64 CPU oracle | gpu-kriging.spec.ts, spatial-interpolation.node.spec.ts |
| `GPULocalKriging` | spatial-interpolation/gpu-local-kriging.ts | Simple, ordinary, universal, external-drift, block and co-kriging | Ruiz-equilibrated partial-pivot LU (`equilibrated-solve-wgsl.ts`); pivot ratio ≥ **1e-5** | k ≤ 32, ≤ 8 drift columns, ≤ 64 support points, 2–4 variables; brute-force scan unless `neighborhoods` is supplied; planar only; **model parameters are compile time** | Variance and pivot ratio per query | GSTools, PyKrige, gstat | interpolation-coverage*.spec.ts, interpolation-advanced.spec.ts |
| `GPULocalRBF` | spatial-interpolation/gpu-local-rbf.ts | SciPy RBFInterpolator, local | Equilibrated LU | k ≤ 32, polynomial degree ≤ 2 | None | SciPy | prediction-scale-robustness.spec.ts |
| `GPUNaturalNeighbor` | spatial-interpolation/gpu-natural-neighbor.ts | Sibson coordinates | Geometric | ≤ 64 neighbours, 256 triangles, 2^24 sites | None | CGAL | interpolation-advanced.spec.ts |
| `GPUInverseDistanceWeighting` | spatial-interpolation/gpu-inverse-distance-weighting.ts | IDW | Weighted mean | grid index | None | gstat / GDAL style | parity-wp1-*idw*.spec.ts |
| `GPUConditionalGaussianSimulation` | spatial-interpolation/gpu-conditional-gaussian-simulation.ts, conditional-simulation-blocked.ts | Exact conditional Gaussian simulation | Dense f32 Cholesky of the joint (n+q)² covariance; absolute tolerance 1e-6·(sill+nugget) | n + q ≤ 4096 (64 MiB factor) | Realizations | f64 reference | conditional-simulation-*.spec.ts |
| `GPUVariogram` | pair-statistics/gpu-variogram.ts | Matheron and Cressie–Hawkins, directional | Exact 64-bit fixed-point pair sums | lags × directions ≤ 512 | Pair counts | `oracles/dataframe-variogram.py`, gstat | gpu-variogram*.spec.ts |
| `GPUVariogramModelFit` | pair-statistics/gpu-variogram-model-fit.ts | WLS fit | Log-range ladder plus golden section; closed-form NNLS nugget and sill | spherical, exponential, Gaussian only | Residual AIC; boundary status 3 | GSTools | variogram-model-fit-*.spec.ts |
| `GPUCrossVariogram` / `GPUSpaceTimeVariogram` | pair-statistics/gpu-cross-variogram.ts | Cross and space-time empirical semivariance | Dense all-pairs | n ≤ 16384; spatial × temporal bins ≤ 128 | Pair counts | Small hand fixtures plus a reference in the spec | test/gpu-spatial-analysis/spatial-regression/geospatial-expansion*.spec.ts |
| `GPURipley`, `GPURipleyDistanceFunctions`, `GPURipleyEnvelope` | pair-statistics/gpu-ripley*.ts | K, L, pcf, F, G, J | Exact integer pair histograms | ≤ 256 radii | CSR envelopes | spatstat | gpu-ripley*.spec.ts, parity-wp6-*.spec.ts |
| `GPUSpatialCorrelogram`, `GPUPointPatternIndices`, `GPUMarkedPointStatistics` | pair-statistics/ | Correlogram, point-pattern indices, cross-type and marked K | Pair histogram | various | Tests and p-values where defined | spatstat, pointpats | gpu-spatial-correlogram*.spec.ts, geospatial-expansion*.spec.ts |
| `GPUInhomogeneousPointPattern` | point-process/gpu-inhomogeneous-point-pattern.ts | Kinhom (border, translation), pcf, Epanechnikov intensity | Dense O(n² · radii) | **≤ 4096 events**, ≤ 256 radii; fixed bandwidth (no bw.diggle / bw.ppl) | None beyond envelopes | spatstat | spatstat-parity.spec.ts, point-process*.spec.ts |
| `GPUPointPatternEnvelope` (with internal rank envelope) | point-process/gpu-point-pattern-envelope.ts | Pointwise, DCLF and extreme-rank envelopes | Order statistics | ≤ 4096 curve points and simulations | Global p-value | spatstat | point-process-advanced.spec.ts |
| `GPUSpatialBalancedSample` / `GPUContinuousSpatialSample` | point-process/ | GRTS / balanced sampling | — | 4096 rows; 4096-vertex frames | Inclusion probabilities | spsurvey style | point-process*.spec.ts, continuous-spatial-sample.node.spec.ts |
| `GPUStudyWindowPreparation` | study-window/ | Polygon window preparation | Exact plus sampled grid | ≤ 1,048,576 sample cells | — | — | gpu-study-window-preparation*.spec.ts |
| `GPUSpatialCrossValidation` | spatial-validation/gpu-spatial-cross-validation.ts | blockCV folds and buffers | — | n ≤ 65536, ≤ 64 folds; cluster/NNDM design ≤ 256 rows | Fold labels | blockCV 4.0.1 | test/gpu-spatial-analysis/spatial-regression/coverage-statistics*.spec.ts |
| `GPUSpatialCrossValidatedOLS` | spatial-validation/gpu-spatial-cross-validated-ols.ts | Training-only OLS per fold | OLS | P ≤ 15 | Held-out metrics | NumPy | spatial-validation-advanced.spec.ts |
| `GPUNestedInterpolationValidation` | spatial-validation/gpu-nested-interpolation-validation.ts | Nested variogram / RBF selection | Local solves | **n ≤ 256**, ≤ 16 folds, ≤ 16 candidates, k ≤ 32 | Held-out scores | NumPy / SciPy | spatial-validation-advanced.spec.ts |
| `GPUSpatialInteraction` | spatial-interaction/gpu-spatial-interaction.ts | spint gravity, four constraint modes | Profile grid or golden section; IPF | ≤ 4096 pairs, ≤ 128 origins and destinations, 1 covariate | **None** (doc line 71) | spint | coverage-estimators.spec.ts |
| `GPUCatchmentAccessibility`, `GPUHuffTradeAreas` | catchment-accessibility/ | 2SFCA, Huff | Weighted sums | CSR | None | `oracles/network-cost-weights-2sfca.py`, Huff oracle | gpu-catchment-accessibility.spec.ts, parity-wp1-huff*.spec.ts |
| `GPULocationAllocation`, `GPUBoundedFacilityAllocation` | location-allocation/ | p-median, maximum coverage, set cover | Greedy plus Teitz–Bart swaps | Binding-size bounds; bounded variant ≤ 32 facilities | Status words | `oracles/p-median-oracle.py` | gpu-location-allocation.spec.ts, facility-parallel*.spec.ts |
| `GPUSegregation` | segregation/gpu-segregation.ts | PySAL indices | Reductions | ≤ 16 groups | Inference via the decomposition recipe | PySAL segregation | gpu-segregation*.spec.ts |

---

## 2. Numerical findings (question 1)

**N1. Small dense solves are float32 everywhere. The safeguards differ by family.** CONFIRMED.
- OLS (`gpu-ordinary-least-squares.ts`, doc lines 126–170) is the best protected:
  - two-pass centring with Kahan merges;
  - correlation scaling;
  - Cholesky pivot tolerance 1e-3, so status 1 at VIF > 1e6 (`ordinary-least-squares-parameters.ts:21`);
  - Belsley index by one-sided Jacobi;
  - precision warning at cond > 1e4.
- The spatial ML estimators reuse OLS for `e0` and `e1`, and propagate OLS status and conditioning.
- The GWR family uses `factorEquilibrated_P` (`local-weighted-least-squares-wgsl.ts:70-84`). It does diagonal equilibration and then `choleskyFactor_P` (`spatial-regression-solve.ts:40-42`), which rejects only `diagonal <= 0` or non-finite. There is **no relative pivot tolerance**. See V1.
- Kriging and RBF use Ruiz equilibration plus partial-pivot LU, and reject when a pivot is below 1e-6 of the largest entry or the pivot ratio is below 1e-5 (`equilibrated-solve-wgsl.ts:19-20`). `GPUKriging` uses a separate solver with a ratio of 1e-4. See V2.
- Conditional simulation uses a dense f32 Cholesky with an absolute tolerance of 1e-6·(sill+nugget). See N4.

**N2. Spatial ML log-determinant.** CONFIRMED.
- Default: the series `ln|I-ρW| = -Σ_{q≤Q} t^q tr((W/s)^q)/q`, with Q = 128 and t = ρ·‖W‖∞. It is evaluated by Horner's rule in f32 from `GPUSpatialWeightsTraces`.
- Traces are exact (one unit probe per row) up to 4096 rows by default and up to 16384 rows on request. Above that they are Hutchinson estimates with 64 Rademacher probes (`spatial-weights-traces-parameters.ts:10-27`, `gpu-spatial-lag-maximum-likelihood.ts:338-343`).
- So for n > 4096 the default log-likelihood is stochastic. The summary marks this with `APPROXIMATION = 1` and the stochastic error bound `getSpatialLogDeterminantErrorBound`.
- Truncation forces a search limit: |ρ|·‖W‖∞ < about 0.93 for Q = 128 (doc lines 255–265). Above it the fit reports `SERIES_LIMITED` / `AT_BOUNDARY` rather than a biased interior ρ. `tailExtrapolation` (geometric model of the last three traces, `spatial-series-tail.ts`) extends the limit to 0.99 but is **off by default**.
- `logDeterminant: 'exact'` uses CPU float64 Householder + QL eigenvalues of the symmetrized W (`spatial-exact-log-determinant.ts`). It is capped at **1024 rows** and needs W to be similar to a symmetric matrix. The GPU then sums `log(1-ρλ)` in f32 with Kahan compensation.
  - Caveat (SUSPECTED): the Kahan compensation in `getExactLogDeterminantWGSL` relies on the shader compiler not reassociating. Metal fast-math may remove it. With n ≤ 1024 the uncompensated f32 error is still small.
- Even in exact mode, the ρ variance's `tr3 = tr((WA)'WA)` and `(I-ρW)^{-1}Xb` still come from Q-step truncated series (doc lines 199–203, code lines 809–841, 924–970).
  - I checked whether this biases the variance near ρ ≈ 0.98: the omitted tail of a row-standardized W is close to a constant per connected component, which the centred intercept and the correlation-based pr2_e absorb. I do **not** report it as a defect.
- **Untested regime.** The only stochastic-versus-exact tests use the 252-row `lag_12_21` scene with **256** probes:
  - `test/gpu-spatial-analysis/spatial-regression/spatial-maximum-likelihood.node.spec.ts:356-380`;
  - `test/gpu-spatial-analysis/spatial-regression/spatial-lag-maximum-likelihood.spec.ts:347-371`.
  
  The default production regime (n > 4096, 64 probes) is exercised only for node selection (`test/gpu-spatial-analysis/spatial-regression/spatial-maximum-likelihood-nodes.node.spec.ts:218`, 5000 rows, no numerical check).
  - My scale estimate (SUSPECTED): for a k-neighbour row-standardized W, the Hutchinson SE of tr(W^q) is about √(2n/(k·64)). At n = 1e5 that is about 20–25 per power. The resulting ρ error is about 1e-3, a sizeable fraction of SE(ρ) at that n. No test pins it.

**N3. Kriging conditioning.** CONFIRMED that it is checked. Silent failure is SUSPECTED only at the margin.
- Near-duplicate points under a Gaussian model with no nugget give identical f32 rows. The pivot test then fails and the cell gets status 2, or NaN in `GPUKriging`. That is a loud failure.
- `GPUKriging` coalesces exact duplicates (doc lines 196–200). `GPULocalKriging` reports exact duplicates as singular and does not jitter them (doc lines 105–113).
- The risk is the gap between the two tolerances (V2).
- Variance is computed as `prior − w'c − μ'drift`, clamped at −1e-4 of the prior (`gpu-local-kriging.ts:444`). Cancellation noise is about k·eps·sill, which is harmless.

**N4. Conditional simulation, dense f32 Cholesky up to 4096².** CONFIRMED tolerance; scaling effect SUSPECTED.
- The tolerance is absolute: 1e-6·(sill+nugget) (`gpu-conditional-gaussian-simulation.ts:438-445`, `conditional-simulation-blocked.ts:64-69,130-139`).
- The backward error of an f32 Cholesky grows like n·eps·‖A‖. At n+q ≈ 4096 that is about 2e-4·sill, two orders of magnitude above the tolerance.
- For dense query sets under smooth covariances (Gaussian, or exponential with a long range), late query pivots will go below −tolerance and fail explicitly well before the 4096 cap.
- Small positive query pivots in (0, tol] are kept as `sqrt(value)`, so noisy loadings can enter the realizations. The duplicate-alias branch in the blocked path zeroes only exact duplicates.
- Box–Muller uses 24-bit uniforms, so normals are truncated at about 5.9σ. That is harmless.

**N5. Coordinates are f32 at the storage boundary.** CONFIRMED.
- `GPUKriging`'s doc says local offsets mean "UTM-sized absolute coordinates do not quantize" distances (`gpu-kriging.ts:192-195`). But sample positions are read from a `float32x2` view and only then shifted (`position - origin`, line 720).
- So samples are already quantized to the f32 ulp of their absolute magnitude: about 0.5 m at a UTM northing of 5e6. The offset helps only the query (cell) positions.
- `GPUVariogram`, `GPUCrossVariogram`, `GPULocalKriging` and GWR all take absolute f32 positions with no origin shift.
- `GPUSpaceTimeVariogram` takes **times as f32 with no origin**. Epoch seconds (about 1.7e9) are quantized to 128 s; epoch milliseconds to about 2 minutes. Nothing in the doc warns about this.

**N6. Variogram accumulation is exact.** CONFIRMED.
- `GPUVariogram` uses 64-bit fixed-point pair sums with a robust unit (doc lines 183–208). It is deterministic and well bounded.
- `GPUCrossVariogram` / `GPUSpaceTimeVariogram` do *not* share it. They use plain f32 per-(focus, bin) partials summed serially over rows (`gpu-cross-variogram.ts:170`). That is adequate at n ≤ 16384, but inconsistent with the rest of the module.

**N7. Variogram fitting.** CONFIRMED.
- Only the range is nonlinear: a 200-step log ladder plus golden section, with closed-form nonnegative nugget and sill.
- Boundary optima return status 3.
- Families are limited to spherical, exponential and Gaussian (`variogram-model.ts:6`). There is no Matérn or stable model, no nested structures and no anisotropy, either here or in `getPredictionCovarianceWGSL` (`gpu-local-kriging.ts:468-477`).

---

## 3. Correctness and validity findings

**V1. GWR, generalized GWR and the nonstationarity test call an ill-conditioned local system "OK".** CONFIRMED.
- `fitLocation` (`gwr-fit-wgsl.ts`, the `factorEquilibrated_P` call) returns `ok` whenever every Cholesky pivot of the equilibrated (unit-diagonal) X'WX is positive.
- The status doc says SINGULAR means "singular, ill-conditioned, or the bandwidth was invalid" (`geographically-weighted-regression-parameters.ts:121-125`). No conditioning test exists.
- The optional `localConditionNumber` is only reported (`gpu-geographically-weighted-regression.ts:1310`). It never changes status.
- Failing scenario: P = 2 and x2 = x1 + δ inside a small bandwidth, giving a local correlation of 1 − 1e-6.
  - The equilibrated last pivot is about 2e-6, against f32 rounding of about 1e-7 in the off-diagonal.
  - β1 and β2 then carry errors of order 5–50%, with status 0 and finite, plausible SEs.
- OLS would return status 1 for the same data (pivot tolerance 1e-3).
- Recommendation: apply the OLS tolerance to the equilibrated factor, or set a status bit from `getLocalConditionNumber` above about 1e3 (mgwr flags a local condition number > 30).

**V2. `GPULocalKriging` accepts systems that `GPUKriging`'s own measurement calls inaccurate.** CONFIRMED.
- `gpu-kriging.ts:51-59` records that, with a Gaussian model, nugget 0.001 and 12 clustered samples, a pivot ratio of 3e-5 gave **weight errors of 5e-2**. It therefore sets the threshold at 1e-4.
- The shared equilibrated solver used by `GPULocalKriging` and `GPULocalRBF` accepts ratios down to **1e-5** (`equilibrated-solve-wgsl.ts:20`), with systems up to 40 × 40 (32 neighbours plus 8 drift columns).
- For SPD covariance blocks, the pivot ratio is only a lower bound on 1/cond. The true error can be larger than the ratio implies.
- `output.values.z` exposes the pivot ratio, so callers can filter. The status word still says 0.

**V3. `GPUSpaceTimeVariogram` has no zero-lag classes and redoes all pair work per bin.** CONFIRMED.
- `gpu-cross-variogram.ts:123,138-148`: one invocation per (focus, bin) rescans every later row and keeps only the pairs in its bin. That is O(n²·bins) distance evaluations: 1.7e10 at n = 16384 and 128 bins.
- Bins are equal-width from 0. Distance-0 pairs (same station, different times) fall into spatial bin 0 together with every pair in `[0, maxDistance/L)`.
  - So the pure temporal variogram γ(0, u), needed to fit separable, metric or sum-metric models (gstat `variogramST` → `fit.StVariogram`), cannot be separated.
  - Time lag 0 can be isolated only by choosing `maximumTime/T` below the sampling interval.
- Test evidence is a 3-point fixture plus a small reference (`test/gpu-spatial-analysis/spatial-regression/geospatial-expansion.spec.ts:172-193,502-511`). There is no gstat `variogramST` parity.
- The coverage matrix has no row for `GPUCrossVariogram`, `GPUSpaceTimeVariogram` or `GPUMarkedPointStatistics`.

**V4. Panel lag ML is a toy-sized "reference" mode, and the coverage matrix misnames it.** CONFIRMED.
- `gpu-spatial-panel-lag-maximum-likelihood.ts:66,79-81` caps it at 32 entities, 32 periods and 7 predictors.
- `dev-docs/coverage-matrix.md:143` (ST03.random) attributes Panel_RE_Lag to `GPUSpatialPanelErrorMaximumLikelihood`. It should be `GPUSpatialPanelLagMaximumLikelihood`, and the row should state the cap.

**V5. MGWR full inference is limited to n ≤ 128.** CONFIRMED.
- The check is at `gpu-multiscale-geographically-weighted-regression.ts:162-167`. Above 128 rows, only conditional SEs with a caller-supplied σ² are available, and they exclude backfit and bandwidth-selection uncertainty (doc lines 89–100).
- `dev-docs/coverage-matrix.md:146` (ST06) claims "full MGWR backfit Jacobian uncertainty" without the cap.

**V6. Adaptive GWR bandwidth is capped at k ≤ 128.** CONFIRMED.
- `GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_NEIGHBOR_COUNT = 128` (`geographically-weighted-regression-parameters.ts:67`). `getBandwidthForValue` rejects larger k.
- mgwr's golden-section search on n ≥ 5–10k often selects adaptive k in the hundreds. Here those bandwidths are unreachable, and the search optimum is clipped to the ladder.

**V7. Stale or contradicting documentation in point processes.** CONFIRMED.
- `gpu-ripley.ts:183` says "cross-type and inhomogeneous K are not implemented". `GPUMarkedPointStatistics` (cross-type K/L) and `GPUInhomogeneousPointPattern` (Kinhom) now exist.
- Roadmap PP01/PP02 (`dev-docs/roadmap.md` §2.2) says "Intensity estimation … Not implemented. `gpu-ripley.ts:183` rejects inhomogeneous K". Line 183 is a doc comment, not a rejection.
- The coverage matrix (line 165) lists PP01 as implemented with Epanechnikov intensity. The roadmap row should be narrowed to what is still missing:
  - bandwidth selection (bw.diggle, bw.ppl);
  - intensity with covariates (ppm);
  - more than 4096 events.

**V8. Kriging and variogram model-family gap against named references.** CONFIRMED absence.
- gstat, GSTools and PyKrige all default to or offer Matérn (and stable, power and nested models) and geometric anisotropy. None exists here (N7).
- This is the largest validity gap for "parity" claims beyond the three families tested.

**V9. Local kriging parameters are compile-time.** CONFIRMED.
- `gpu-local-kriging.ts:37` ("counts and model parameters specialize the graph"); `getPredictionCovarianceWGSL` inlines sill, range and nugget as WGSL literals (line 477).
- Any likelihood or cross-validation loop over covariance parameters therefore recompiles per evaluation.
- `GPUKriging` takes per-frame variogram parameters, but only ordinary kriging with k ≤ 16.

**V10. Scope notes against references** (CONFIRMED from the docs in the code).
- `GPUSpatialErrorGM` reports λ without an SE (spreg reports one).
- `GPUSpatialLagImpacts` has no simulated impact SEs (spreg and spatialreg can give them).
- `GPUSpatialInteraction` has one covariate and no coefficient uncertainty (spint gives the GLM SEs).
- `GPUProbitRegression` is non-spatial.
- Panels must be balanced, with no time effects (`gpu-spatial-panel-regression.ts:59`).
- These overlap the roadmap rows ST01, ST04/05, ST03 and ST11 and are listed only for completeness.

---

## 4. Scaling walls (question 2)

| Contributor | Wall | Kind |
|---|---|---|
| Spatial lag / error ML, GM | Exact traces ≤ 4096 by default (16384 maximum); eigenvalue log-det ≤ 1024 (CPU O(n³)); Q ≤ 256; P ≤ 14 / 15 | Above 4096 the likelihood is stochastic. Each fit costs about Q·(1 + probe batches) dispatches |
| Panel lag ML | N ≤ 32, T ≤ 32, P ≤ 7 | Dense determinant |
| Panel error ML | N·T ≤ 65536; block I_T ⊗ W is materialized, so traces run on N·T rows instead of N | Avoidable T-fold work |
| GWR | Brute force n ≤ 65536 (O(n²) per bandwidth candidate); indexed ≤ 1,048,576 only for bisquare or cut-off Gaussian kernels; adaptive k ≤ 128; P ≤ 11; ≤ 64 search iterations | Gaussian kernels without a cutoff visit every row |
| Generalized GWR | P ≤ 24; same row limits; IRLS ≤ 128 iterations | Private 25×25 arrays (register pressure) |
| MGWR | O(iterations·P·n²); full inference n ≤ 128 | Dense |
| `GPUKriging` | k ≤ 16, compile time | Indexed ring walk, scales |
| `GPULocalKriging` / RBF | k ≤ 32; brute-force O(queries·samples) unless a CSR `neighborhoods` is supplied | Model parameters compile time |
| Conditional simulation | n + q ≤ 4096; dense (n+q)² f32 binding | Practical f32 failure earlier (N4) |
| Natural neighbour | ≤ 64 neighbours, 256 triangles | Bounded per query |
| `GPUCrossVariogram` / `GPUSpaceTimeVariogram` | n ≤ 16384, ≤ 128 bins, O(n²·bins) | Brute force; not on the indexed pair histogram |
| `GPUVariogram`, Ripley family | Grid-indexed pair histogram; ≤ 2048 accumulators per workgroup; ≤ 512 bins, ≤ 256 radii | Scales |
| Inhomogeneous K, envelopes, balanced sampling | ≤ 4096 events / curves / rows | Dense |
| Nested interpolation validation | n ≤ 256 | Dense |
| CV design (cluster, NNDM, kNNDM) | ≤ 256 rows | Dense |
| Spatial interaction | ≤ 4096 OD pairs, 128 × 128 | Bounded |
| SUR | ≤ 4 equations × 7 predictors | 16-wide Cholesky |

Compile-time unrolling: Q, P, k, support count, lag and bin counts and candidate counts are all compile-time constants, so changing them rebuilds the graph. The spatial lag ML emits `Q − 1` separate Horner dispatch nodes (`gpu-spatial-lag-maximum-likelihood.ts:809`), so graph size grows linearly with Q.

---

## 5. Time-dimension readiness (question 4)

**What exists**
- **Panels** (`GPUSpatialPanelRegression`, `...ErrorMaximumLikelihood`, `...LagMaximumLikelihood`): balanced, period-major arrays (`period·N + entity`) with one time-invariant entity W, replicated into a block-diagonal CSR. T is limited to 2–1024 (lag ML: 32).
  - Unit effects only (random effects for lag ML).
  - No time effects, dynamic lags, unbalanced panels or serial correlation.
- **`GPUSpaceTimeVariogram`**: empirical space × time-lag bins only (V3).
- Outside this scope, but reusable: Knox and Mantel tests (`space-time-tests/`), emerging hot spots, the space-time permutation scan, `GPURasterTemporal`, and trajectory operators.

**What is absent** (CONFIRMED by grep: no `vgmST`, product-sum, sum-metric, GTWR, Hawkes / ETAS, space-time K, or space-time intensity anywhere in `modules/analysis/src`):
- space-time covariance models;
- space-time variogram fitting;
- space-time kriging;
- GTWR;
- the space-time K function;
- spatio-temporal intensity.

| Candidate | Reuse | New work | Size |
|---|---|---|---|
| ST covariance models (separable, product-sum, metric, sum-metric) | `getPredictionCovarianceWGSL` pattern; the spatial families | `covariance(vec3, vec3)`; parameters per frame rather than as literals | S for the WGSL; M including per-frame parameters |
| ST variogram binning done properly | `pair-histogram.ts` (indexed, exact 64-bit) with time as an extra binding, temporal bins as slots, and explicit h = 0 / u = 0 classes | Replace the brute-force `GPUCrossVariogram` path; time origin offset | S–M |
| ST variogram fitting | `GPUVariogramModelFit` structure (ladder plus closed-form linear part) | Multi-parameter nonlinear search (2–3 ranges, an anisotropy ratio κ); the linear sills still have a closed form | M |
| ST kriging | `GPULocalKriging` system assembly and `equilibrated-solve-wgsl.ts`; `neighborhoods` CSR input | Space-time neighbour selection (no 3D grid index today; the scan-all path works but is O(n·q)); ST covariance | M |
| GTWR | `fitLocation`, `getCandidateLoopWGSL`, golden-section bandwidth | Combined distance `sqrt(d_s² + μ d_t²)`; a second bandwidth or ratio; time-aware index (2D grid plus a time filter) | M |
| Space-time K / pcf | Pair histogram with slots = radius × time bins; Knox's time sorting; study-window border distances | Temporal edge correction, CSR × uniform-time null | S–M |
| Space-time intensity (kernel) | `GPUInhomogeneousPointPattern` Epanechnikov plus window mass | Separable temporal kernel and time-window normalization; lift the 4096 limit by indexing | M |
| Panel time effects / dynamic panels | Panel CSR replication, within transform | Two-way demeaning (S); dynamic (lagged y_{t−1}, Wy_{t−1}) needs GMM or bias correction (L) | S / L |

---

## 6. Feasibility of the methods in question 7

| Method | Existing pieces | Gaps | Verdict |
|---|---|---|---|
| **Vecchia / NNGP likelihood and prediction** | `GPULocalKriging` simple mode = conditional Gaussian mean and variance given ≤ 32 neighbours; accepts caller CSR `neighborhoods`, so "previous points in the ordering" can be passed; observation-error variance supported; equilibrated solve; GPU sort available | (a) Covariance parameters are compile-time (V9), so every optimizer step recompiles: they must move to a parameter buffer. (b) Max–min ordering and an ordered-kNN builder. (c) A reduction of Σ[log σ_i² + r_i²/σ_i²]. (d) An outer optimizer (CPU Nelder–Mead or L-BFGS, or a GPU ladder like the variogram fit). (e) Matérn. (f) Status 2 at a pivot ratio of 1e-5 must count as a likelihood failure, not be skipped | **M**. Highest-value addition; also gives scalable prediction |
| Fixed-rank kriging / random Fourier features | OLS (P ≤ 15, ridge); blocked dense Cholesky from conditional simulation (≤ 4096²) | No dense regression wider than 16 columns; the conditional-simulation Cholesky is coupled to the covariance assembly. RFF with D in the hundreds needs a general blocked X'X + λI solve | **M** (factor out the blocked Cholesky); L if done properly with Matérn spectral sampling |
| Scalable GWR (FastGWR) | Already per-location O(k) work, GPU-resident hat diagonal, golden-section AICc, grid index to 1M rows | Adaptive k ≤ 128 (V6); Gaussian kernel without a cutoff is O(n²); no conditioning flag (V1) | **S**: lift k (private arrays scale as k only in the neighbour list), add a cutoff default, add a conditioning status |
| SVC via eigenvector spatial filtering | Sparse W matvec kernels (traces, Horner); OLS | Needs leading eigenvectors of MWM: GPU Lanczos or block power iteration with re-orthogonalization (f32 loses orthogonality quickly); OLS P ≤ 15 limits the eigenvectors × covariates (the SVC-ESF design has P·L columns) | **L** |
| Conformal prediction intervals | `GPUSpatialCrossValidation` (65536 rows, buffered blocks), `GPUSpatialCrossValidatedOLS` held-out residuals, `GPUKriging` LOO prediction and variance for every sample, GPU sort | Quantile of absolute or standardized held-out residuals per fold, plus interval assembly; spatially localized (weighted) conformal needs kernel-weighted quantiles | **S** for split / CV+ with OLS and `GPUKriging`; **M** for localized; kriging inside nested validation is limited to n ≤ 256 |
| Log-Gaussian Cox process | Intensity kernels; window preparation; conditional simulation ≤ 4096; Poisson IRLS in generalized GWR | No latent-field inference (Laplace / INLA / MCMC), no circulant embedding or SPDE on a lattice, no Poisson GLM with offset at grid scale | **L** |
| Hawkes / ETAS | Pair histogram (bounded space neighbourhood), Knox time sorting, Philox streams, Epanechnikov background | Triggering sums over earlier events need ordered space-time pair enumeration; EM declustering (p_ij per pair) is parallel but needs per-event normalization and an outer iteration loop; magnitude terms (ETAS) | **M** for the space-time Hawkes EM with a fixed kernel; **L** for ETAS with parameter MLE |

---

## 7. Recommendations, ranked

1. **Add a conditioning status to the GWR family (V1).** Reuse the OLS pivot tolerance on the equilibrated factor, or the existing `getLocalConditionNumber`. Today a nearly collinear local design returns status 0 with wrong coefficients. Small change; correctness.
2. **Unify kriging acceptance (V2).**
   - Either raise `SYSTEM_PIVOT_RATIO_TOLERANCE` to 1e-4, or justify 1e-5 with a measurement made after equilibration, as was done for `GPUKriging`.
   - Add a test with a Gaussian model, nugget 1e-3 and 32 clustered neighbours that asserts either status 2 or a weight error below 1e-3 against f64.
3. **Pin the default large-n stochastic ML regime (N2).** Add one GPU or CPU-mirror test at n ≥ 8192 with 64 probes, comparing ρ and SE(ρ) against exact traces (16384-row exact mode is available), across several seeds. Record the spread next to `APPROXIMATION`.
4. **Fix the coverage matrix and roadmap rows:**
   - ST03.random: class name and the 32 × 32 × 7 cap (V4);
   - ST06: MGWR full inference n ≤ 128 (V5);
   - PP01: what is actually missing (V7);
   - add rows for `GPUCrossVariogram`, `GPUSpaceTimeVariogram` and `GPUMarkedPointStatistics`;
   - add the n ≤ 256 limit of nested validation and CV design to ST10;
   - correct the `gpu-ripley.ts:183` and `gpu-kriging.ts:192-195` doc claims (N5).
5. **Move covariance parameters to per-frame buffers in `GPULocalKriging` (V9) and add Matérn (V8).** This unblocks likelihood fitting, Vecchia, nested CV without recompiling, and space-time covariances.
6. **Rebuild `GPUSpaceTimeVariogram` on the indexed pair histogram.** Give it explicit h = 0 and u = 0 classes and a time origin (V3, N5). Then add space-time covariance families and a fit (section 5). This is the cheapest route to real space-time geostatistics.
7. **Conformal intervals on the existing CV machinery (S).** `GPUSpatialCrossValidatedOLS` residuals and `GPUKriging` LOO standardized errors, reduced to fold quantiles.
8. **Lift the adaptive GWR limit k ≤ 128 (V6).** Default a Gaussian cutoff so the indexed path applies. Together with 1 this gives a FastGWR-class tool.
9. **Vecchia likelihood (M)** on top of 5: ordered-kNN CSR, parameter buffer, likelihood reduction, CPU outer optimizer.
10. **Add a relative or condition-aware tolerance to the conditional-simulation Cholesky (N4).** It should scale with n·eps, or the class should document the practical size limit for smooth covariances. Add a test at n + q = 4096 with an exponential model and a long range.

Not repeated here: roadmap ST01, ST02, ST04/05, ST06b, ST07–ST11, PP04/05 and IN04–IN06.
