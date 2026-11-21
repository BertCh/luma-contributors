---
title: "GPU Spatial Analysis: Regression"
sidebar_label: "Regression"
description: "Regression contributors in @luma-contributors/analysis/gpu-spatial-analysis."
---


# Regression

Accuracy figures on this page describe the recorded fixtures and device, under their stated
conditioning and scale. They are not guarantees for every GPU. Newly added numerical regression
cases require the queued native-device qualification before extending those claims.


### `GPUOrdinaryLeastSquares`

Fits a linear model of one response on predictor columns by least squares and reports coefficients, standard errors, t statistics and diagnostics.

::lens{scenario=gpu-spatial-analysis.ordinary-least-squares rung=result caption="Diabetes on three predictors across 788 tracts, R squared 0.67; compare each estimate with its standard error."}

Ordinary least squares with an intercept and a full regression report, in one prebuilt graph:
coefficients, standard errors, t statistics, R2, adjusted R2, sigma squared, log-likelihood, AIC,
BIC, optional per-row residual and fitted columns, the Jarque-Bera normality test, the
Koenker-Bassett and Breusch-Pagan heteroskedasticity tests, and collinearity and precision
diagnostics (Belsley condition number, correlation condition number, maximum VIF, precision
warning). `getGPUOrdinaryLeastSquaresModelResult(readback, {predictorNames})` turns the read-back
arrays into the shared `GPURegressionModelResult` (coefficient table with Student t p-values on
`n - p` degrees of freedom, `sigmaSquared = RSS / (n - p)` and `sigmaSquaredMaximumLikelihood = RSS / n`,
R2, log-likelihood, AIC and BIC equal to spreg, conditioning, and the tests `jarqueBera`,
`breuschPagan`, `koenkerBassett`).
When coefficient buffers have spare capacity, pass `{predictorCount, predictorNames}` to the
decoder. It reads only the active `predictorCount + 1` intercept-first entries and validates that
all three coefficient arrays hold that prefix; unused capacity does not add model parameters.
The decoder captures each array reference once before validation and reads that same reference
throughout decoding, including accessor-backed readbacks. Each coefficient p-value uses its reported statistic.

```ts
import type {Device} from '@luma.gl/core';
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
import {GPUOrdinaryLeastSquares, getGPUOrdinaryLeastSquaresParameterValues, type GPUOrdinaryLeastSquaresProps} from '@luma-contributors/analysis/gpu-spatial-analysis';

// Fragment inputs: allocate/upload these views in the caller's graph.
declare const device: Device;
declare const graph: GPUCommandGraph; // owns every view in inputs
declare const inputs: GPUOrdinaryLeastSquaresProps;
const {predictors, response, mask} = inputs;
const {coefficients, standardErrors, tStatistics, summary, residuals, fitted} = inputs.output;
const status = inputs.output.status.code;
const ridgeBuffer = new GPUParameterBuffer(device, {id: 'ols-ridge', format: 'float32', length: 1});
ridgeBuffer.write(getGPUOrdinaryLeastSquaresParameterValues({ridgeLambda: 0}));
graph.add(
  new GPUOrdinaryLeastSquares({
    predictors, // GraphDataView<'float32'>, row-major row * predictorCount + column
    response, // GraphDataView<'float32'>
    mask, // optional GraphDataView<'uint32'>; zero excludes the row
    parameters: ridgeBuffer.importToGraph(graph), // optional: getGPUOrdinaryLeastSquaresParameterValues({ridgeLambda})
    predictorCount: 3, // compile-time, 1 to 15 (16-wide Cholesky); the intercept is added internally
    output: {coefficients, standardErrors, tStatistics, summary, status: {code: status}, residuals, fitted}
  })
);
```

- Supply `predictorCount` predictor columns **without** a constant column. The intercept is added
  internally; an explicit all-ones column is singular. `predictors.length` is exactly
  `response.length * predictorCount`, and a supplied mask has exactly `response.length` uint32
  entries. All are single packed views. Coefficient outputs accept capacity at least
  `predictorCount + 1`; summary at least 22, status at least one, and each requested row output
  at least `response.length`. Unwritten capacity tails are unspecified.
- The direct constructor may omit `predictorCount`: it infers the exact integer ratio
  `predictors.length / response.length` for nonempty response rows, still within 1–15.
  Named operations require an explicit count. Length alone cannot distinguish row-major
  interleaving from concatenated columns: use `GPUOrdinaryLeastSquares.createDesign` when
  starting with separate columns. It preserves the single-column view identity; multiple
  equal-length columns are gathered/transposed on the supplied graph in coefficient order.
- Rows with any non-finite predictor or response value, or a zero mask, are excluded.
- `coefficients`, `standardErrors`, `tStatistics`: `predictorCount + 1` floats, intercept first.
- `summary` (22 floats, `GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH`), indexed by
  `GPU_ORDINARY_LEAST_SQUARES_SUMMARY_*`: row count, R2, adjusted R2, sigma squared, log-likelihood,
  AIC, BIC, Jarque-Bera, its p-value, Koenker-Bassett, its p-value, RSS, TSS, skewness, kurtosis,
  ridge lambda, Breusch-Pagan, its p-value, `CONDITION_NUMBER`, `CORRELATION_CONDITION_NUMBER`,
  `MAXIMUM_VARIANCE_INFLATION_FACTOR`, `PRECISION_WARNING`. The slots 9 and 10 used to be called
  `BREUSCH_PAGAN`; they were always the Koenker-Bassett statistic, and `BREUSCH_PAGAN` now names the
  classic test in slots 16 and 17.
- The predictor count is capped at 15 (16 coefficients with the intercept) by the 16-wide
  function-local Cholesky factorization of the solve; a larger design needs a tiled solver.
- `status.code` (uint32): 0 ok, 1 singular or ill-conditioned (a predictor within `1 - 1e-6` of the span
  of the others, or constant), 2 too few rows (`n <= predictorCount + 1`). A failed fit leaves every
  statistic and per-row column NaN (the row count is still written).
- `residuals` and `fitted` are optional per-row columns (NaN for excluded rows).
- Optional `predictions` has one value per input row, including masked rows and rows with a missing
  response, provided their predictors are finite. It uses the fit's training-only centers, slopes
  and mean correction; a failed fit or nonfinite predictors produce NaN. The operation form and
  CPU twin expose the same output. Raw float32 intercepts can lose response-level information at
  large predictor origins, so use these centered predictions for held-out evaluation.
  The OLS GPU operation and CPU twin capture all graph names and optional output presence at
  construction, including the nested status name. Reusing or editing the caller's props/output
  map cannot rename a declared prediction or disable its writer; construct a new operation to
  change ports. Input graph values remain per-run data.
  The direct `GPUOrdinaryLeastSquares` contributor also captures its design width, input views,
  optional outputs and nested status view at construction; buffer contents remain per-frame data.
  Both named operation forms require positive `rowCount` at construction and a resolved response
  of exactly that length. Mismatches are rejected before fitting, even when row outputs are omitted.
For separately stored predictors, this adapter creates the required observation-row layout.
All views and outputs below belong to the supplied graph; no buffer is read back:

```ts
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {GPUOrdinaryLeastSquares, type GPUOrdinaryLeastSquaresProps} from '@luma-contributors/analysis/gpu-spatial-analysis';

declare const graph: GPUCommandGraph;
declare const firstColumn: GraphDataView<'float32'>;
declare const secondColumn: GraphDataView<'float32'>;
declare const response: GraphDataView<'float32'>;
declare const output: GPUOrdinaryLeastSquaresProps['output'];
const predictors = GPUOrdinaryLeastSquares.createDesign(graph, [firstColumn, secondColumn]);
// If the rows are joined-table observations, keep the join's completeness evidence.
declare const joinedTable: Parameters<typeof GPUOrdinaryLeastSquares.addJoinedTableFit>[2];
const composition = GPUOrdinaryLeastSquares.addJoinedTableFit(
  graph, {predictors, response, output}, joinedTable
);
// Retain composition.status beside the fit; composition.mask is its training-row mask.
```

`addJoinedTableFit` combines `joinedTable.matched`, the required one-word `status.complete`,
optional `status.candidateOverflow` and an optional caller mask in left observation-row order.
Incomplete joins exclude all training rows rather than fitting a retained subset. It adds the
fit to that same graph and returns `{fit, mask, status}`; retain the returned join status with
the regression outputs. For an ordinary complete table, use
`graph.add(new GPUOrdinaryLeastSquares({predictors, response, output}))` instead.

- Conditioning never changes `status`: a warning is not a failure and the fit is still reported.
  `CONDITION_NUMBER` is the Belsley number (spreg `OLS.mulColli`): `sqrt(lambdaMax / lambdaMin)` of the
  raw `[1, X]` Gram matrix scaled to unit diagonal (rebuilt from the centered moments and means; values
  above about 30 mean harmful collinearity; ridge is not included). `CORRELATION_CONDITION_NUMBER` is
  `lambdaMax / lambdaMin` of the predictor correlation matrix the float32 Cholesky factors (with the
  ridge penalty), and `MAXIMUM_VARIANCE_INFLATION_FACTOR` the largest diagonal of its inverse. Float32
  coefficients lose about `correlationConditionNumber * 6e-8` relative accuracy, so
  `PRECISION_WARNING` is 1 when the correlation condition number exceeds 1e4
  (`GPU_REGRESSION_PRECISION_CONDITION_LIMIT`, about three of seven digits lost). Eigenvalues come from a
  cyclic Jacobi routine in float32, so the condition numbers are indicative for very ill-conditioned
  designs (the GPU specs allow 2e-2 to 1e-1 relative against the float64 oracle and spreg). They match spreg `mulColli` and `diagnostics.vif` (max
  over the predictor columns, computed on the centered design) in the float64 oracle.
- The optional ridge penalty `lambda >= 0` is added to the centered normal matrix of the
  predictors (never the intercept) and changes between frames without rebuilding the graph. Ridge
  slope standard errors are `sqrt(diag(sigma2 * (X'X + lambda I)^-1))`, where `X` is the
  centered predictor design. The matrix inside the diagonal is the approximate slope covariance;
  the intercept is unpenalized and its variance also accounts for predictor means. These errors
  ignore shrinkage bias and are not the ridge sandwich covariance. The reader reports the applied
  positive penalty as `regularization: {kind: 'ridge', lambda}` and
  `inferenceMethod: 'ridge-approximate'`. Its reported t statistics and Student t p-values retain
  that approximate inference interpretation; unpenalized fits use `ordinary-student-t`. Likelihood and information
  criteria use the fitted residuals and the ordinary parameter count; they do not correct for
  shrinkage or substitute an effective ridge degrees of freedom. The parameter packer and shader
  treat nonpositive/nonfinite lambda as zero. A positive finite host request must stay positive
  and finite when packed as float32, otherwise the packer rejects it. The final scaled-residual
  likelihood writer evaluates `log(sum((residual/scale)^2)/n) + 2*log(scale)` for nonzero residuals:
  positive residual variance is not floored, including below `2^-126` or when raw squares
  underflow. Only an exact zero-residual fit uses the finite log-variance fallback
  `log(1.17549435e-38)` (approximately `log(2^-126)`), unlike GTWR's exact-fit infinities.
  This fallback does not change published residual variance or RSS.


Numerics: no float atomics. Per-tile partial sums (one invocation per tile of consecutive rows, in
row order) are merged in a fixed pairwise order (a float Kahan compensation is folded to a plain
sum by the shader compiler of at least one supported GPU). Each tile first selects a complete
training-row origin and sums differences from it. The merge shifts those partials to the first
nonempty tile's origin before averaging; empty tiles contribute zero. Above 4,096 tiles, bounded
group metadata passes retain that same first usable global origin and exact uint32 counts, followed
by group sums about the global origin and a pairwise merge over groups. Main and Breusch–Pagan
means, moments and residual sums all use bounded merges. Metadata adds two uint32 words per group;
floating group buffers are included in allocation preflight. These extra nodes appear only above
4,096 tiles. This preserves small
variances when many nearly equal predictors have a large coordinate origin. Means first, then centered
cross-products about those means; the solve runs on the correlation-scaled normal matrix with a
single-invocation Cholesky. The residual sum of squares comes from the residual pass, not from
`Syy - b'Sxy`, so it stays accurate when R2 is close to 1. A float32 mean misses the true mean of
its column by up to a float32 step; the moment pass also sums the deviations of every column from
its stored mean, and that remainder is carried through the cross-products, the intercept and the
residuals, so the residuals sum to zero and the residual sum of squares matches float64 also when
the response mean is thousands of residual deviations (residuals of 0.002 around 10,000: within
1e-4, where a float32 mean alone gave 4% to 20% too much). Results are bitwise reproducible for a
given `tileRowCount` (default `max(64, ceil(rows / 4096))`).
Mean corrections are divided by the row count before multiplication, with the quotient fixed by
an integer bit cast, so an overflowing product of deviation sums cannot erase a finite variance.
Fitted values and held-out predictions also fix the combined centered contribution and mean
correction with an integer bit cast before adding the response mean. This prevents Metal fast
math from adding small correction terms to a large mean separately and rounding them away.
For diagnostics, a residual reduction records the largest absolute residual, then two additional
tile/merge nodes accumulate moments of `e / max(abs(e))`. The auxiliary regression uses the
centered response `(abs(e) - scale) / scale * (abs(e) / scale + 1)`, with the first difference
fixed against arithmetic regrouping. It equals `e² / scale² - 1` and preserves small squared-magnitude
differences that separately dividing and squaring near one can erase. Raw residuals and RSS retain
their original units. Standardized skewness, kurtosis,
Jarque–Bera, Koenker–Bassett and classic Breusch–Pagan avoid raw fourth-power overflow/underflow
when their underlying residuals and RSS are representable in float32.

AIC and BIC count `predictorCount + 1` parameters (GeoDa convention); the log-likelihood is
Gaussian with `sigma2 = RSS / n`. Each estimator on this page counts as its reference library does:
OLS the intercept and predictors (`p`), `GPUSpatialLagMaximumLikelihood` `p + 1` (it adds `rho`),
`GPUSpatialErrorMaximumLikelihood` `p` (not `lambda`), and the GWR family `tr(S) + 1` (the error
variance). Put the models on one count before comparing criteria: each uncounted parameter adds 2
to AIC and `ln(n)` to the Schwarz criterion. Jarque-Bera is `n/6 (S^2 + (K - 3)^2 / 4)` with p-value
`exp(-JB / 2)`. Both heteroskedasticity tests use the auxiliary regression of `e^2` on `[1, X]` with
`R2_aux`, and a chi-square p-value on `predictorCount` degrees of freedom computed by the regularized
upper incomplete gamma with a fixed iteration cap. Koenker-Bassett (studentized, valid without
normality) is `n R2_aux`; the classic Breusch-Pagan (assumes normal errors) is
`0.5 R2_aux (sum e^4 / s2^2 - n) = 0.5 R2_aux n (kurtosis - 1)` with `s2 = RSS / n`, which is spreg
`breusch_pagan` with `z = [1, X]`. Both match spreg 1.9.1 (`breusch_pagan`, `koenker_bassett` with
`z = reg.x`; spreg's default `z` is the squared design). With skewed errors they differ several
times over, and Jarque-Bera rejecting normality is the cue to read Koenker-Bassett instead.
Classic BP is evaluated directly from the auxiliary explained sum of squares divided by
`2 * (RSS/n)^2`. For positive constant squared residuals it is 0 with p-value 1; studentized KB
remains NaN because the auxiliary response has zero variance. An exact zero-residual fit leaves
both tests undefined.
The GPU gates BP on its positive normalized squared sum. A nonzero fit can therefore retain its
dimensionless BP result even when individual raw squares underflow to zero; an exact fit still
leaves BP undefined. This does not reconstruct underflowed raw RSS values.
`getChiSquareSurvival(value, degreesOfFreedom)` is the float64 CPU mirror; its tail is 1 at
nonpositive values, 0 at positive infinity and NaN for NaN input. Degrees of freedom must be a
positive integer. The GPU auxiliary-test tail uses the same boundary conventions.
The host uses converged series/continued fractions with a shape-dependent budget, a stable
Stirling prefactor, and a uniform incomplete-gamma expansion for shapes at least `1e8`.
It throws on host-budget exhaustion rather than returning a partial probability. The GPU retains
its 200-iteration budget for its at-most-15 degrees of freedom.
`getStudentTTwoSidedPValue` / `getStudentTSurvival` (float64 regularized incomplete beta, pinned to
`scipy.stats.t.sf`) back the model-result p-values.
Student-t evaluation forms the small complementary beta argument directly near zero and retains
log arguments for huge finite statistics, preserving representable tails even when `t*t` overflows.
For degrees of freedom at least `1e10`, it integrates the normal-density expansion through
`df^-2`, avoiding branch selection or tail loss from a beta argument rounded to one.
The CPU singular-value helper transposes wide matrices, skips stagnant rotations and bounds
Jacobi sweeps by `max(32, 4 * min(rows, columns))`; its range contains only actual singular values.
It obtains column norms with `hypot` and computes rotations from unit-column products and
scaled norm ratios, preserving small singular values whose squared column norms underflow.

The shared model result derives global corrected AIC as `AIC + 2 k (k + 1) / (n - k - 1)`,
where `k` is the information-criterion parameter count. It returns `null` when `n <= k + 1`
or the model has no likelihood. Estimator-specific corrected-AIC overrides are preserved.

#### How it computes

{/* lens:GPUOrdinaryLeastSquares */}

::lens{scenario=gpu-spatial-analysis.ordinary-least-squares rung=formula caption="Follow 6 equations: normal equations, intercept, sigma squared, standard errors, t statistics and R squared."}
::lens{scenario=gpu-spatial-analysis.ordinary-least-squares rung=kernel caption="Inspect the tile, merge, solve, residual and finish kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.ordinary-least-squares rung=graph caption="Trace the accumulation, solve, residual and report nodes of one fit on 788 tracts."}

{/* parameters: GPUOrdinaryLeastSquares generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUOrdinaryLeastSquares` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPUOrdinaryLeastSquares`.

- Packer: `getGPUOrdinaryLeastSquaresParameterValues(settings?)`, writes `GPU_ORDINARY_LEAST_SQUARES_PARAMETER_LENGTH` (1) `float32` values.

Settings of `getGPUOrdinaryLeastSquaresParameterValues`:

| Field | Type | Description |
|---|---|---|
| `ridgeLambda` (optional) | `number` | Ridge penalty in squared predictor units, added as `lambda * I` to the centered normal matrix of the predictors (never to the intercept). A value that is not finite and positive means no penalty. Positive finite requests must remain normal and finite in float32 (at least 2^-126); smaller penalties cannot be applied reliably on Metal. Defaults to 0. |

{/* /parameters */}

### `GPUGeographicallyWeightedRegression`

The catalog CPU fallback centers each local weighted system and scales predictor columns
before solving; coefficients, standard errors and hat diagnostics are transformed back to
the caller's coordinates. A large predictor offset alone does not make a local fit singular.

Fits a separate weighted regression at every location and writes local coefficients, standard errors, t values and fit statistics, choosing the bandwidth by AICc.

::lens{scenario=gpu-spatial-analysis.geographically-weighted-regression rung=result caption="Diabetes beside the local poverty coefficient, 0.002 to 0.34 across 788 tracts; compare where it is strongest."}

Pass optional `spatialContext` to `getGPUGeographicallyWeightedRegressionParameterValues(settings)` to warn on short fixed `bandwidths`; see [Precision of longitude/latitude positions](./concepts.md#precision-of-longitudelatitude-positions).

Geographically weighted regression with AICc bandwidth selection (Fotheringham, Brunsdon and
Charlton 2002; mgwr, ArcGIS GWR). For every included row `i` it fits a weighted least squares
`y ~ 1 + x` over all included rows `j` with weights `K(d_ij / h)` and writes local coefficients,
local standard errors, t values and a significance mask, local R-squared, fitted values, residuals and
the hat-matrix diagonal.

The contributor captures construction options and output maps at construction. Mutating those
caller objects later does not alter validated shader shapes or bindings; changing input buffer
contents continues to update the next encoding.

```ts
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
import {GPUGeographicallyWeightedRegression, getGPUGeographicallyWeightedRegressionParameterLength, getGPUGeographicallyWeightedRegressionParameterValues, type GPUGeographicallyWeightedRegressionProps} from '@luma-contributors/analysis/gpu-spatial-analysis';

// Fragment inputs: allocate/upload these views in the caller's graph.
declare const graph: GPUCommandGraph;
declare const device: import('@luma.gl/core').Device;
declare const inputs: GPUGeographicallyWeightedRegressionProps;
const {positions, predictors, response} = inputs;
const {coefficients, standardErrors, tValues, significant, selectedBandwidth, summary} = inputs.output;
const parameters = new GPUParameterBuffer(device, {
  id: 'gwr-parameters',
  format: 'float32',
  length: getGPUGeographicallyWeightedRegressionParameterLength(16)
});
graph.add(
  new GPUGeographicallyWeightedRegression({
    positions, // GraphDataView<'float32x2'>, planar (or lon/lat degrees with spatialContext)
    predictors, // row-major row * predictorCount + column
    predictorCount: 2,
    response,
    parameters: parameters.importToGraph(graph),
    maximumBandwidthCount: 16,
    output: {coefficients, standardErrors, tValues, significant, selectedBandwidth, summary}
  })
);
parameters.write(
  getGPUGeographicallyWeightedRegressionParameterValues(
    {kernel: 'gaussian', bandwidthMode: 'adaptive', bandwidths: [20, 40, 60, 80]},
    16
  )
);
```

- Kernels: `'gaussian'` `exp(-0.5 (d/h)^2)` (unbounded unless `gaussianCutoff` is set) and `'bisquare'`
  `(1 - (d/h)^2)^2` for `d < h`.
- Bandwidths: `'fixed'` (ladder values are distances) or `'adaptive'` (ladder values are neighbour
  counts `k`, `h` is 1.00001 times the distance to the k-th nearest included row, the row itself
  counted as in mgwr; `k` in `[2, maximumNeighborCount]`, at most 128).
- Bandwidth ladder: up to `maximumBandwidthCount` (at most 32) candidates in the parameter buffer.
  Each candidate's AICc, `n ln(RSS/n) + n ln(2 pi) + n (n + tr S) / (n - 2 - tr S)`, is evaluated on
  the GPU and the outputs are written at the selected bandwidth. A ladder of one value is a fixed
  bandwidth. Candidates with a non-positive bandwidth, invalid `k` or `n - 2 - tr S <= 0` score NaN.
- Singular locations (changed): a location whose local system is singular (for example a rare dummy
  that is constant in its neighbourhood) makes only that row NaN (coefficients, standard errors,
  fitted value, residual, hat; `localStatus` 1). It no longer voids the whole candidate. A candidate's
  `n`, RSS, `tr S` and AICc are taken over its non-singular rows, and `bandwidthSingularRowCounts`
  reports the count per candidate. Selection: among the candidates with the fewest singular rows the
  lowest AICc wins (ties to the lowest index), so a bandwidth with singular rows is chosen only when
  every valid candidate has some. This is a deliberate deviation from mgwr, which raises on a singular
  local system.
- `localStatus` 1 (`SINGULAR`) also covers an ill-conditioned local design. The local normal equations
  are scaled to unit diagonal (a correlation matrix, on the design centred at the focus), and a Cholesky
  pivot below `GPU_ORDINARY_LEAST_SQUARES_PIVOT_TOLERANCE` (1e-3, the OLS limit) is singular, so a
  local predictor within about `1 - 1e-6` of the span of the earlier columns (variance inflation factor
  above 1e6) is rejected exactly where `GPUOrdinaryLeastSquares` reports status 1. Before this test only
  a non-positive pivot failed and a nearly collinear design returned coefficients with status 0. The
  same test applies to the generalized GWR information matrix and the nonstationarity test refits.
  Well-conditioned fits are unchanged.
- Refinement (new, default on; `refinement: false` keeps the fixed ladder, a number from 1 to 4 sets
  the pass count, default 2): after the caller ladder, a GPU kernel writes a new ladder that zooms
  around the best candidate between its nearest evaluated neighbours (it never extrapolates beyond the
  caller's range), the same kernels evaluate it, and the overall best wins. Adaptive ladders hold
  integers: when the integers inside the bracket number at most `maximumBandwidthCount` all of them
  are tried, so the result is the exact integer optimum of the bracket; a wider bracket is sampled at
  equal spacing and the next pass narrows it. Fixed ladders are `maximumBandwidthCount` equally spaced
  distances per pass. Each pass costs up to one more ladder evaluation, so the default is about three
  times the work of the bare ladder; shorten the ladder rather than turning refinement off when cost
  matters. `selectedBandwidth` is `[candidate index, value]`; indices from `maximumBandwidthCount` up
  are refined candidates, and `candidateTable` lists every candidate of every pass. Every pass reads its
  ladder through a GPU pass cursor, so N and N + 1 passes compile the same pipelines. On the
  144-row test scene (adaptive bisquare) mgwr's golden section with its default bounds `[40 + 2 p, n]`
  stops at `k = 47`, while its exhaustive integer AICc optimum, which the refinement finds from a
  five-value ladder, is `k = 26` (and mgwr's golden section with bounds widened to 6..144 agrees).
  The AICc curve in `k` is not unimodal there (local minimum near 30), which is why a bracket that
  fits the ladder is enumerated rather than searched.
- Golden-section search (new, opt-in `bandwidthSearch: 'golden-section'`; the default `'ladder'` is
  unchanged): mgwr's `Sel_BW(...).search('golden_section')` as a fixed-depth loop on the GPU, about
  12 to 20 candidate evaluations instead of a ladder of up to 32 per pass, no read-back. The first two
  `bandwidths` values of the parameters are the bounds `[minimum, maximum]`; a value of 0 or less
  takes mgwr's default for that bound, so `[0, 0]` is mgwr's default search, and a single value is a
  fixed, unselected bandwidth. Defaults are computed on the GPU from the included rows: adaptive
  `[min(40 + 2 (predictorCount + 1), n), n]` with `n` capped at `maximumNeighborCount`. Capping the
  default lower bound as well keeps small scenes inside their feasible neighbor counts. Fixed bounds are
  `[min_dist / 2, 2 max_dist]` with `min_dist` the smallest positive row-to-row distance and
  `max_dist` the largest pairwise distance (exact without a grid index; with one, the largest
  distance from a row to a corner of the grid extent, an upper bound that costs at most an
  iteration or two). The interior points are mgwr's `a + 0.38197 (c - a)` and `c - 0.38197 (c - a)`,
  rounded half to even to integers for adaptive bandwidths. Iteration 1 evaluates both points, every
  later iteration compares the two scores, shrinks the bracket and evaluates the one new interior
  point with the same candidate kernels (the other is reused). The search stops when the two
  interior AICc values differ by at most `max(1e-6, 4e-6 |AICc|)`, mgwr's `1e-6` with a float32 noise
  floor (the AICc sums `n` float32 terms; `1e-6` alone never fires at AICc values of hundreds), or
  after `maximumSearchIterations` (default 30, at most 64, compile-time). The selected bandwidth is
  the winner of the last comparison, as mgwr returns it; fewest singular rows first, then the lower
  AICc, an invalid score loses to a valid one and two invalid scores stop the search. The iteration
  index is a GPU cursor and the loop is gated by an indirect dispatch that the one-thread iteration
  kernel zeroes on convergence, so N and N + 1 iterations compile the same pipelines and converged
  iterations launch nothing. `candidateTable` then has `maximumSearchIterations + 1` rows in
  evaluation order (rows 0 and 1 are the first interior points, row `i + 1` the point of iteration
  `i`; unreached rows are NaN), `selectedBandwidth[0]` is the table row of the selection,
  `bandwidthScores` and `bandwidthSingularRowCounts` hold its first `maximumBandwidthCount` rows, and
  the optional `searchState` output reports the bracket, iteration count and whether the tolerance
  (not the cap) stopped it (`GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SEARCH_STATE`). `refinement` is
  ignored. Needs `maximumBandwidthCount >= 2` and `maximumNeighborCount >= 2`. Pinned to mgwr 2.2.1
  on the 144-row scene (bisquare and Gaussian, adaptive and fixed): the adaptive `k` is mgwr's exactly,
  fixed bandwidths agree within the AICc-flatness tolerance of the float32 search, and the float64 CPU
  mirror of the state machine reproduces mgwr's evaluated bandwidths in order. mgwr rounds its fixed
  answer to two decimals; this contributor returns the unrounded value.
- Gaussian cutoff (new, `gaussianCutoff` in bandwidth multiples, compile-time, default exact
  unbounded Gaussian): the weight of a row farther than `gaussianCutoff * h` is exactly 0 (fixed and
  adaptive), and the Gaussian kernel then uses the grid index like bisquare, `O(n * neighbours)`
  instead of `O(n^2)` (mgwr has no cutoff). The omitted weight of a row is at most `exp(-c^2 / 2)` of
  the self weight: 1.1e-2 at `c = 3`, 3.4e-4 at 4, 3.7e-6 at 5, 1.5e-8 at 6. For a locally uniform
  density the omitted share of the weight sum is that value and of the weighted second moment
  `(1 + c^2 / 2) exp(-c^2 / 2)` (3.0e-3 at 4, 5.0e-5 at 5, 2.8e-7 at 6), which bounds the relative
  effect on the local sums up to the local conditioning; on the test scene the coefficient error stays
  below 20 times that share. Recommended: 5 (`GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_RECOMMENDED_GAUSSIAN_CUTOFF`),
  where the effect is below the float32 accumulation error of the fits; use 4 when 3e-3 is acceptable
  and 6 only to match the exact Gaussian to float32. The cutoff does not change the bisquare kernel.
- Local inference (new): `standardErrors`, `tValues` and `significant` (all laid out like
  `coefficients`) are mgwr `bse`, `tvalues` and `filter_tvals != 0`. `bse = sqrt(diag(C C') sigma2)`
  with `C C' = (X'WX)^-1 X'W^2 X (X'WX)^-1` (one more weighted Gram matrix in the same neighbourhood
  pass) and `sigma2 = RSS / (n - tr S)` (mgwr `sigma2_v1`). The summary carries `ENP = tr S`,
  `SIGMA_SQUARED`, `LOG_LIKELIHOOD`, the Da Silva-Fotheringham adjusted alpha `alpha * k / tr S`
  (`ADJUSTED_ALPHA`, `k` design columns, `alpha` 0.05 unless the per-frame `significanceLevel` says
  otherwise) and the critical value `t.ppf(1 - adjusted / 2, n - 1)` (`CRITICAL_T`). The Student t
  quantile is inverted on the GPU (continued-fraction incomplete beta with a bisection; the float64
  CPU mirror `getStudentTCriticalValue` is pinned to `scipy.stats.t.ppf` to 1e-9 and the GPU summary is
  checked against mgwr), because the mask needs it there.
  `significant` is 1 where `|t| > CRITICAL_T`. The bandwidth is selected on the observed data and the
  inference is conditional on it, as in mgwr. Pinned to mgwr 2.2.1 (fixed and adaptive, planar and
  spherical); adaptive fits agree to about 5e-4 because mgwr scales the k-th neighbour distance by
  1.0000001 and this contributor by 1.00001.
- `getGPUGeographicallyWeightedRegressionModelResult({summary, selectedBandwidth})` returns the shared
  `GPURegressionModelResult` (`kind: 'geographically-weighted-regression'`, empty `coefficients`,
  `parameterCount = tr S`, `rowCount = n`, `sigmaSquared = RSS / (n - tr S)`, `logLikelihood`,
  `akaikeInformationCriterion` `= -2 llf + 2 (tr S + 1)` and `correctedAkaikeInformationCriterion`
  (mgwr `llf`, `aic`, `aicc`), `rSquared`) plus `effectiveParameterCount`, `singularRowCount`,
  `includedRowCount`, `adjustedAlpha`, `criticalT` and `selectedBandwidth`.
- `spatialContext` (new): omitted is planar. It measures distance, so `local-lambert-equal-area` frames throw. `{coordinateSpace: 'longitude-latitude', metric:
  'great-circle', units: 'meters'}` makes `positions` `(longitude, latitude)` degrees and every
  distance (kernels, fixed bandwidths, the adaptive k-th neighbour) a haversine distance on a sphere of
  `sphereRadius` (default the mean Earth radius 6,371,008.8 m); bandwidths are then metres. Rows with a
  latitude outside [-90, 90] are excluded. The grid index is planar-only and is not built for a
  great-circle context, so every location scans every included row and the 65,536-row scan limit
  applies; the cell walk is not made exact for longitude wrap-around. mgwr `spherical=True` uses a
  6371.0 km radius and kilometres, so pass `sphereRadius: 6371000` and metres to match it.
- Predictors: 1 to 11 (12 design columns; it was 7). No kernel binds more than eight storage buffers,
  because the per-row results (including standard errors) share one row-state buffer; the limit is
  per-thread private memory (`2 P^2 + 6 P` floats in the fit). The 11-predictor shader has not been run
  on every adapter: lower `GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_PREDICTOR_COUNT` if one
  rejects it.
- Outputs: `coefficients` (`row * (predictorCount + 1) + column`, column 0 intercept), optional
  `standardErrors`, `tValues`, `significant`, `localR2` (`1 - sum_j w_ij (y_j - x_j' beta_i)^2 / sum_j w_ij
  (y_j - ybar_i)^2`: the residuals are those of local model `i` at its neighbours `j`, not each neighbour's own
  GWR residual, so a package that uses the latter gives different values), `fitted`, `residuals`, `hatDiagonal`,
  `localStatus` (0 ok, 1 singular, 2 excluded), `bandwidthScores` (AICc of the caller ladder),
  `bandwidthSingularRowCounts`, `candidateTable`, `selectedBandwidth` (`[index, value]`),
  `localConditionNumber` and `summary` (`GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY`: RSS, `tr S`,
  AICc, global R-squared, n, has-valid-candidate flag, singular rows, sigma2, adjusted alpha, critical
  t, log-likelihood, included rows; 12 floats, it was 6).
  Local and global R-squared are NaN for a constant response, including nonzero constants. Response
  moments use differences from a retained response before averaging. Parallel variance reductions
  choose a nonempty tile as their response-moment anchor. Candidate `OBSERVATION_COUNT` counts
  successful fits; invalid-bandwidth and singular rows remain in `INCLUDED_ROW_COUNT` but
  do not enter fitted statistics. An adaptive count above the available rows therefore reports
  zero fitted observations and NaN AICc.

- `localConditionNumber` is the mgwr `local_collinearity` local condition number: the singular-value
  ratio of the kernel-weighted, column-normalized design (above 30 flags collinearity). It is pinned to
  mgwr 2.2.1 (bisquare and gaussian, fixed and adaptive). The weighted design rows are folded into a
  triangular factor by Givens rotations and its singular values come from one-sided Jacobi (the
  helpers of `GPULocalCollinearity`), so the two classes report the same number to about
  `float32 epsilon * condition`, also for a predictor with a large mean. The former Gram-matrix
  route was 17% off for a predictor of 1500 +- 3 and meaningless for a projected northing.
- Rows with a zero mask or a non-finite position, predictor or response are excluded as
  calibration locations and as neighbours.
- Numerics: the design is centred on the focal row and Jacobi-equilibrated before the shared
  Cholesky solve, so large coordinates or badly scaled predictors stay well conditioned in f32.
  A predictor that is constant over a neighbourhood makes that location singular.
  Construction also captures nested spatial-context metadata, including the sphere radius.
  The shared Student-t evaluator keeps extreme representable tails in log space and brackets
  critical values across the floating-point exponent range. Degrees of freedom remain capped
  at 200,000: the resulting approximation is about 1.2e-5 at a two-sided tail of 0.05 and grows
  toward more extreme tails; it is not an error bounded by float32 rounding.
  Direct tails at large degrees of freedom use a stable normal-tail evaluation after inverting
  the third-order t-quantile expansion; NaN t inputs propagate NaN.
- Cost: one thread per location scanning every row per candidate: `O(n^2 * ladder * p^2)`, at most
  65536 rows without the grid index (see `indexGridSize` below) (about 80 ms for 4096 rows and 8 candidates on the
  test adapter). Reductions use fixed-order 256-row tiles, so results are bitwise reproducible.
  Above 1,048,576 rows, candidate tiles compensate the RSS and hat-trace sums before the global
  reduction. Single-predictor fits on this route also retain low words in the local normal moments
  before publishing float32 scores and fit outputs.
- Grid index: `indexGridSize` (default: automatic from 1024 rows; `false` disables) bins rows into a uniform grid so
  each location scans nearby cells only. The scan limit is 65,536 rows; indexed execution supports
  up to 16,777,216 rows subject to device buffer and scratch limits. Explicit exact Gaussian support
  is limited to 1,048,576 rows. These are the
  `GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_ROW_COUNT`,
  `GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_INDEXED_ROW_COUNT`, and
  `GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_DENSE_INDEXED_ROW_COUNT` limits respectively. The
  `'bisquare'` fixed-bandwidth fits sum in cell order (about 1e-5 relative to the earlier row-order sums, still
  deterministic). Gaussian kernels use the grid only with a `gaussianCutoff`.

Decode `localStatus` with `decodeGPUGeographicallyWeightedRegressionStatus` and compare codes with
`GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_STATUS`. It reports each local weighted least squares
fit. A successful local fit does not establish bandwidth-search convergence: the golden-section
`searchState` reports that separately. MGWR has different row codes and a separate decoder for
the coupled backfit outcome; GWR code 1 means singular, while MGWR code 1 means excluded.
The parameter packer requires every supplied bandwidth to remain finite in float32 and an explicit
`significanceLevel` to remain strictly between 0 and 1 in float32. It rejects values that round to
0 or 1, and leaves a supplied target unchanged on validation failure. Only an omitted level encodes
the default sentinel 0. Bandwidths follow the `ArrayLike` indexed contract: the validated length
is read once, exactly that many indexed values are copied before any target write, and an attached
iterator cannot change the count or overwrite spare target capacity.

#### How it computes

{/* lens:GPUGeographicallyWeightedRegression */}

::lens{scenario=gpu-spatial-analysis.geographically-weighted-regression rung=datum bind=page datum=300 caption="Inspect tract 300: its local coefficients, t value and local R squared."}
::lens{scenario=gpu-spatial-analysis.geographically-weighted-regression rung=formula caption="Follow the bisquare weight, local fit, hat value, AICc and local t equations."}
::lens{scenario=gpu-spatial-analysis.geographically-weighted-regression rung=kernel caption="Inspect the grid, candidate, golden-section and local-fit kernels with retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.geographically-weighted-regression rung=graph caption="Trace the golden-section search over 10 candidate bandwidths and the final local fits over 788 tracts."}

{/* parameters: GPUGeographicallyWeightedRegression generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUGeographicallyWeightedRegression` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPUGeographicallyWeightedRegression`.

- Packer: `getGPUGeographicallyWeightedRegressionParameterValues(settings, maximumBandwidthCount?)`, writes `getGPUGeographicallyWeightedRegressionParameterLength(...)` `float32` values.
  - `maximumBandwidthCount` (`number`): Ladder capacity of the contributor (its `maximumBandwidthCount`).

Settings of `getGPUGeographicallyWeightedRegressionParameterValues`:

| Field | Type | Description |
|---|---|---|
| `spatialContext` (optional) | `GPUSpatialContext` | Spatial context of the positions; when it is longitude/latitude, a distance under 10 float32 quanta (about 17 m) logs a warning (see Precision of longitude/latitude positions). Does not change the packed values. |
| `kernel` (optional) | `'gaussian' \| 'bisquare'` | Weight kernel. Defaults to `'bisquare'`. |
| `bandwidthMode` (optional) | `'fixed' \| 'adaptive'` | Meaning of the ladder values. Defaults to `'fixed'`. |
| `bandwidths` | `ArrayLike<number>` | Candidate bandwidths: distances (`'fixed'`) or neighbour counts `k` (`'adaptive'`, rounded to integers). One value is a fixed bandwidth; several are searched by AICc (and refined, see `refinementPassCount`). With `bandwidthSearch: 'golden-section'` the first two values are the search bounds `[minimum, maximum]` (a value of 0 or less takes mgwr's default for that bound); further values are ignored. |
| `significanceLevel` (optional) | `number` | Two-sided significance level of the adjusted alpha and critical t value. Its float32 representation must be in (0, 1). Defaults to 0.05. Per frame. |

{/* /parameters */}

### `GPUGeographicallyWeightedRegressionNonstationarityTest`

Tests whether each GWR coefficient varies across space more than permuted data would, with a Monte Carlo pseudo p-value per coefficient.

::lens{scenario=gpu-spatial-analysis.geographically-weighted-regression-nonstationarity-test rung=result caption="Observed against permuted spread of four coefficients; poverty pseudo p = 0.03, the other three 0.01."}

Monte Carlo test of coefficient non-stationarity (mgwr `spatial_variability`, GWR4). It permutes the
observations among the locations with the Philox-keyed swap-or-not shuffle of `GPUGlobalPermutationTest`
(the earlier Feistel bijection was non-uniform on small row counts), refits
at the selected bandwidth and compares the standard deviation of each local coefficient with the observed
one. The pseudo p-value is `(g + 1) / (P + 1)`. Outputs are `table` (six floats per coefficient), `summary`
(`GPU_GWR_NONSTATIONARITY_SUMMARY`, `GPU_GWR_NONSTATIONARITY_TABLE`, `GPU_GWR_NONSTATIONARITY_TABLE_STRIDE`)
and an optional `standardDeviations` (the reference distribution). The per-permutation standard deviations
match mgwr 2.2.1 to 3e-15 (CPU oracle) and 2e-3 (GPU). The bandwidth stays at the value the regression
selected (mgwr re-selects it for every permutation), and that value was selected on the observed data, so
the test is conditional on that selected bandwidth. `selectedBandwidth` is read from slot 1 (the selected value),
so it also works when the regression refined its bandwidth beyond the caller ladder. It is planar-only
(a great-circle `spatialContext` throws, with a hint to project first) and takes 1 to 7 predictors even
though the regression accepts 11. The cost is `O(P * n^2 * p^2)` with a full scan (no grid index).

A coefficient's observed spread and every requested permuted spread must be finite before its
pseudo p-value, exceedance count or simulated z score is published. If any is undefined, those
three fields are NaN. An undefined simulation is never treated as a non-exceedance. The simulated
mean and population standard deviation remain diagnostic summaries of the finite draws;
`summary.failedFitCount` still counts failed local refits. For example, eight included locations
with five wholly failed permutations report 40 failed refits and NaN inference, rather than
an apparently significant `1 / 6` p-value. Clearing the failure on a later encoding restores
the usual add-one inference. A per-frame permutation count outside `[1, maximumPermutations]`
also gives NaN inference and NaN `summary.permutationCount`; it cannot certify a clamped test.
No simulated draws run for an invalid request: the observed distribution row remains available,
all simulated rows are cleared to NaN, and simulated mean/SD are NaN. Simulated moments are
centered on the first finite draw, so identical nonzero draws give SD zero and z NaN.

The observed and permuted spread merges preserve their historical serial order through 4,096 tiles.
Above that, they use the hierarchical pairwise order described below for row-tile reductions. The
per-batch partial budget and device binding limits can bind before the `2^24` tile limit.
Batch sizing budgets four commands per batch above 4,096 tiles (including the group pass),
and retains the historical three-command sizing through 4,096. For example, 262,143 permutations
at 4,097 one-row tiles and one predictor use 35 runs per batch and a 32,040-command upper estimate.

#### How it computes

{/* lens:GPUGeographicallyWeightedRegressionNonstationarityTest */}

::lens{scenario=gpu-spatial-analysis.geographically-weighted-regression-nonstationarity-test rung=formula caption="Follow the observed spread, exceedance count, pseudo p-value and permuted z score."}
::lens{scenario=gpu-spatial-analysis.geographically-weighted-regression-nonstationarity-test rung=kernel caption="Inspect the permuted refit, spread merge and finish kernels with retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.geographically-weighted-regression-nonstationarity-test rung=graph caption="Trace the GWR fit, 99 permuted refits and the finish step across 788 tracts."}

{/* parameters: GPUGeographicallyWeightedRegressionNonstationarityTest generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUGeographicallyWeightedRegressionNonstationarityTest` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPUGeographicallyWeightedRegressionNonstationarityTest`.

- Packer: `getGPUGeographicallyWeightedRegressionParameterValues(settings, maximumBandwidthCount?)`, writes `getGPUGeographicallyWeightedRegressionParameterLength(...)` `float32` values.
  - Input port: `bandwidthParameters`.
  - `maximumBandwidthCount` (`number`): Ladder capacity of the contributor (its `maximumBandwidthCount`).
- Packer: `getGPUPermutationParameterValues(settings, options?)`, writes `GPU_PERMUTATION_PARAMETER_LENGTH` (4) `uint32` values.
  - Input port: `parameters`.
  - `options` (`{maximumPermutations?: number}`)

Settings of `getGPUGeographicallyWeightedRegressionParameterValues`:

| Field | Type | Description |
|---|---|---|
| `spatialContext` (optional) | `GPUSpatialContext` | Spatial context of the positions; when it is longitude/latitude, a distance under 10 float32 quanta (about 17 m) logs a warning (see Precision of longitude/latitude positions). Does not change the packed values. |
| `kernel` (optional) | `'gaussian' \| 'bisquare'` | Weight kernel. Defaults to `'bisquare'`. |
| `bandwidthMode` (optional) | `'fixed' \| 'adaptive'` | Meaning of the ladder values. Defaults to `'fixed'`. |
| `bandwidths` | `ArrayLike<number>` | Candidate bandwidths: distances (`'fixed'`) or neighbour counts `k` (`'adaptive'`, rounded to integers). One value is a fixed bandwidth; several are searched by AICc (and refined, see `refinementPassCount`). With `bandwidthSearch: 'golden-section'` the first two values are the search bounds `[minimum, maximum]` (a value of 0 or less takes mgwr's default for that bound); further values are ignored. |
| `significanceLevel` (optional) | `number` | Two-sided significance level of the adjusted alpha and critical t value. Its float32 representation must be in (0, 1). Defaults to 0.05. Per frame. |

Settings of `getGPUPermutationParameterValues`:

| Field | Type | Description |
|---|---|---|
| `seed` | `number` | Random seed, an integer in `[0, 2^53)`. Results are a pure function of the seed, the inputs and `permutationCount`, independent of the dispatch shape. Per frame. |
| `permutationCount` | `number` | Number of permutations `P`, in `[1, maximumPermutations]` of the contributor. The count is never clamped: a value outside that range makes the kernels write NaN p-values (local: rows are not tested and `output.status.overflow` is set; global: the `permutationCount` field of `output.summary` is NaN). Validate on the CPU with the contributor's `getParameterValues` or the `maximumPermutations` option of `getGPUPermutationParameterValues`. |
| `significanceLevel` (optional) | `number` | Significance level for the local mask, strictly in (0, 1) after float32 rounding. Defaults to 0.05. Ignored by the global test. |

{/* /parameters */}

### Reading regression status words

Import the model-specific `...StatusFields` decoder with its producer from
`@luma-contributors/analysis/gpu-spatial-analysis`. Each accepts one numeric word and returns
`{code, name, ok, excluded, failed, converged}`; unknown words never imply success.
`converged: null` means that this vocabulary does not encode iterative convergence. Existing
string-returning decoders remain available. Read a uint32 port as uint32 before passing its word.

| Decoder | Output scope and interpretation |
| --- | --- |
| `decodeGPUOrdinaryLeastSquaresStatusFields` | `output.status.code`, one model: 0 ok, 1 singular/ill-conditioned, 2 too few rows. |
| `decodeGPUGeographicallyWeightedRegressionStatusFields` | `output.localStatus`, one focal row: 0 ok, 1 singular (including invalid bandwidth), 2 excluded. Selection/search state is separate. |
| `decodeGPUGeographicallyTemporallyWeightedRegressionStatusFields` | Dense/indexed GTWR `output.localStatus`, same row vocabulary; independent of the summary reader status. |
| `decodeGPUMultiscaleGeographicallyTemporallyWeightedRegressionStatusFields` | MGTWR `output.localStatus`, same row vocabulary. Row 0 does not certify backfit convergence: inspect `scores` and the separate bandwidth-search status. |
| `decodeGPUMultiscaleGeographicallyWeightedRegressionStatusFields` | MGWR `output.status.codes`, per result row: 0 converged, 1 excluded, 2 singular, 3 iteration limit with retained finite last estimate, 4 numerical failure. Component search exhaustion is a separate port. |
| `decodeGPUGeneralizedGeographicallyWeightedRegressionStatusFields` | GGWR `output.status.codes`, per result row: 0 converged, 1 excluded, 2 singular, 3 retained unfinished iterate, 4 numerical failure, 5 constant boundary response, 6 witnessed complete/quasi separation. |
| `decodeGPUMultiscaleGeographicallyWeightedRegressionInferenceStatusFields` | `inference.output.status`, one inference result, independent of row success: 0 ok, 1 unsuccessful fit, 2 invalid inference. |
| `decodeGPUSpatialLagMaximumLikelihoodStatusFields` / `decodeGPUSpatialErrorMaximumLikelihoodStatusFields` | Each model's `output.status.code`: 0 ok, 1 singular, 2 too few rows, 3 unusable fit. Code 3 combines nonfinite/incomplete inputs, invalid moments/intervals and (for lag ML) rejected optima; it cannot diagnose a unique cause. Inspect named summary/boundary/convergence fields. |
| `decodeGPUSpatialErrorGMStatusFields` / `decodeGPUSpatialTwoStageLeastSquaresStatusFields` | Each model's `output.status.code`; GM code 3 also covers zero OLS residuals, while 2SLS code 1 combines singular and early nonfinite solves. |
| `decodeGPUSpatialRegressionDiagnosticsStatusFields` | Diagnostics `output.status.code`, independently of fit: 0 ok, 1 singular, 2 too few rows, 3 degenerate/incomplete weights, 4 invalid residual variance/input. |
| `decodeGPUInstrumentalVariablesCovarianceStatusFields` | IV `output.status.code`, independently of fit success: 0 ok, 1 singular projection, 2 insufficient rows, 3 nonfinite input **or computed covariance result**, 4 invalid HAC kernel, 5 invalid leverage, 6 invalid clusters, 7 failed PSD certificate. |
| `decodeGPURestrictedRegimeRegressionStatusFields` | Restricted fit `output.status`, independently of `output.robust.status`: 0 ok, 1 insufficient rows, 2 restricted singular/nonfinite result, 3 unrestricted singular/nonfinite result. Codes 2/3 can follow successful solves when residual sums become nonfinite. |
| `decodeGPUSeeminglyUnrelatedRegressionStatusFields` | SUR system `output.status`: 0 ok, 1 insufficient rows, 2 singular design, 3 singular error covariance, 4 not converged with retained usable estimate (`failed: false`), 5 invalid parameters. |
| `decodeGPUSpatialSeeminglyUnrelatedRegressionStatusFields` | Spatial SUR `spatial.output.status`, one word per equation: error-GM 0–3 plus 4 inadmissible weights; distinct from the SUR system status. |
| `decodeGPUProbitRegressionStatusFields` | Probit `output.status`: 0 passed both accepted/full-step convergence checks, 1 insufficient rows, 2 singular, 3 retained unfinished iterate, 4 certified separation/boundary or invalid nonbinary response. `tolerance` must be positive and at most `MAXIMUM_PROBIT_TOLERANCE` (1e-2). The per-coefficient accepted- and full-step certificate uses the smaller of the requested tolerance and 1e-5. |
| `decodeGPUProbitSpatialDiagnosticsStatusFields` | Probit `spatialDiagnostics.output.status`: 0 ok, 1 unsuccessful model, 2 invalid/incomplete weights, 3 degenerate test. |
| `decodeGPUProbitSpatialLagStatusFields` | Probit `spatialLag.output.status`: 0 ok, 1 model not converged, 2 singular first stage, 3 singular second stage, 4 rho out of range, 5 invalid weights. |
| `decodeGPUSpatialPanelRegressionStatusFields` / `decodeGPUSpatialPanelErrorMaximumLikelihoodStatusFields` | Each panel model's `output.status.code`, using its underlying 2SLS/error-ML vocabulary; panel 2SLS code 2 also includes nonpositive residual degrees of freedom. |
| `decodeGPUSpatialPanelErrorDiagnosticsStatusFields` | Panel-error `diagnostics.status`: 0 ok, 1 unsuccessful/incomplete pooled fit, 2 degenerate tests. |
| `decodeGPUSpatialPanelLagMaximumLikelihoodStatusFields` | Panel-lag `output.status`: 0 converged, 1 invalid/incomplete inputs, 2 singular/invalid variance, 3 unfinished or lower-bound phi. Check boundary bits and covariance availability; code 3 is not a general retained-estimate certificate. |
| `decodeGPUSpatialPanelHausmanStatusFields` | `hausman.status`: 0 ok, 1 unsuccessful fit, 2 invalid covariance difference. |

A retained iteration-limit estimate is not a converged fit. MGWR/GGWR and probit code 3, and
SUR system code 4, are explicitly separate from failed/excluded rows. ML code 3 instead marks
an unusable model; finite residuals retained on a rejection do not establish model success.
See [coverage contracts](./coverage-statistics.md) for the separate covariance, probit and SUR
ports, and [panel fixed effects](./panel-fixed-effects.md) for the panel 2SLS result domain.
SUR publishes its optional completed-fit count in `output.iteration.status.iterationCount`.
Probit's two optional spatial stages read `weights.status.overflow` and
`weights.status.complete`. Overflow or a zero completeness certificate suppresses that stage's
spatial results while the ordinary probit fit remains available.

Read each inference status independently. HAC kernel loss can leave 2SLS point coefficients,
structural residuals and classical summary diagnostics usable while its covariance status is 4
and robust SE/z/p columns are NaN. Restricted-regime robust status 0 additionally requires
an identified restricted fit with positive residual degrees of freedom, a valid complete kernel
when HAC is selected, and resolved positive-definite covariance for every tested contrast.
Its separate robust codes 1/2/3 mean unavailable restricted fit, invalid/incomplete kernel,
or invalid covariance/contrast rank, respectively; nonzero codes suppress all robust results.
The unrestricted comparison can fail without suppressing usable restricted robust inference.
Classical status 1 retains precedence when the unrestricted sample has insufficient residual
degrees of freedom, including when the attempted restricted solve also fails; robust status
then independently reports whether that restricted fit is usable.

Restricted-regime HAC compares `requiredCount` with the terminal CSR offset and allocation
capacity; a supplied `status.count` must equal that terminal offset. Zero `complete`, overflow,
candidate loss, nonzero `unresolvedCount` or an unconverged kernel suppresses robust inference
with status 2; clearing loss restores it. `uncertainCount` counts fallback-resolved
classifications and alone does not suppress inference. Unresolved contrast pivots, rank-one
joint contrasts or exact-fit residual roundoff report robust status 3, covariance NaN and
Wald rows `[NaN,0,NaN]`. Identified small-residual and dense-kernel contrasts remain usable.
CPU certification uses order-independent response medians and a conditioned predictor basis
when intercepts vary; predictor medians and row-position maps are computed once. GPU
certification projects signed influences into the contrast basis before taking absolute values;
tile error applies after CSR cancellation. Optional robust inference stores `2*k+3` float32
score words per original response row for `k` restricted coefficients, including masked rows.
That binding can limit row count alongside the predictor binding and total 2^24-row limit.
Panel-error diagnostic status is likewise separate from the ML fit: code 1 suppresses pooled
tests after entity-weight loss, and code 2 suppresses degenerate tests, including exact pooled
zero RSS or unresolved projected spatial-fit information. A usable ML fit alone does not certify
these tests. Positive `complete` certificates have inverse polarity to overflow flags; an
absent optional certificate never supplies evidence that missing candidates were complete.

### `GPUSpatialRegressionDiagnostics`

Tests an ordinary least squares fit for spatial dependence with LM-lag, LM-error, robust, SARMA and residual Moran's I tests.

::lens{scenario=gpu-spatial-analysis.spatial-regression-diagnostics rung=result caption="Six tests on the diabetes OLS residuals: LM-lag 638 against LM-error 595, then robust 170 against 127."}

Maximum: `rows * (predictorCount + 3)` is at most `getGPUSpatialRegressionDiagnosticsMaximumRowTableValueCount(device)` (2^25 on a default device); the per-tile moment table is at most `getGPUSpatialRegressionDiagnosticsMaximumTileMomentCount(device)` values.

Spatial diagnostics for an OLS fit (spreg `LMtests` and `MoranRes`): LM-lag, LM-error, robust LM-lag,
robust LM-error, LM-SARMA, and Moran's I of the residuals (mean, variance, z, two-sided normal p).

- Props: `weights` (a square `GPUSpatialWeights`; any sparsity pattern is accepted, directed kNN included,
  and row-standardized weights work; `W'` is formed with `GPUSpatialWeightsTranspose`),
  `predictors` (row-major `n x k`, no intercept), `response`, `residuals` (OLS residuals with intercept,
  for example `GPUOrdinaryLeastSquares` `output.residuals`), `predictorCount` (1 to 15), optional `mask`,
  `restandardizeRows`, `tileRowCount` and `output: {tests, summary, status: {code}}`.
- Masked and missing fits: a row is analyzed when its `mask` entry is not zero and its response, residual
  and predictors are finite; pass the OLS `mask` and the NaN residuals of excluded rows do the rest. Excluded
  rows are removed from the whole analysis: `n`, `S0`, `T`, every Gram matrix and every `W`-product run over
  the fitted rows, and links from fitted rows to excluded rows are dropped. That is `libpysal.weights.w_subset`
  on the fitted ids. With `restandardizeRows: true` each fitted row's remaining weights are rescaled to sum
  to 1 (`w_subset` then `transform = 'r'`; use it after masking row-standardized weights); the default
  keeps the surviving weights as given. Rows left with no fitted neighbor keep an all-zero row. Results
  match spreg 1.9.1 `OLS(..., spat_diag=True, moran=True)` on the fitted subset in both modes.
- `tests` (18 f32): six rows of `[statistic, df, p]`: LM_LAG, LM_ERROR, ROBUST_LM_LAG, ROBUST_LM_ERROR
  (df 1), LM_SARMA (df 2) and MORAN_RESIDUALS (the statistic is z, the df slot is 0, p is two-sided). The
  row index constants are `GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TEST_*`.
- `summary` (13 f32): `n` (analyzed rows), `sigma2 = e'e / n`, `T = tr(W'W + WW)`, `J`, `S0`, Moran `I`,
  `E[I]`, `Var[I]`, `z`, `p`, `e'We`, `e'Wy` and `EXCLUDED_ROW_COUNT` (`..._SUMMARY_*`). The excluded
  count is a summary slot, not a status port, because `status.code` is this operator's single result code; it is
  also written when the status is not 0. Before this change a masked fit reported status 4 for the whole
  result.
- The spatial regression recipe publishes one `uint32` solver `code` and one
  `status.invalidCount` for each of its OLS and diagnostics stages. The OLS count comes from the
  number of rows outside its complete cases. The diagnostics count also reflects any rows with
  unusable OLS residuals. Optional GWR publishes one `rowCodes` value per observation.
- `status.code`: 0 ok, 1 singular, 2 too few analyzed rows (`n <= predictorCount + 2`, also a failed OLS fit; OLS
  itself fits `n = predictorCount + 2` rows, so on exactly that many rows the fit succeeds and the diagnostics do not),
  3 incomplete weights (`weights.status.overflow` or `candidateOverflow`) or degenerate weights,
  4 non-finite moments or exactly zero residuals. Incomplete weights suppress every test statistic
  and p-value, and the recipe returns specification decision 4 (undetermined). Both operation forms
  accept the optional weights status inputs and recover when the flags clear. Failed tests have NaN
  statistics and p-values; degrees of freedom and analyzed/excluded row counts remain defined.
  Conditioning slots are not reported here (see the OLS fit).
  The direct GPU contributor also treats a supplied `weights.status.complete = 0` or
  `requiredCount` above the retained terminal offset, supplied `count` or capacity as loss
  with code 3, even beside clear overflow flags. A missing optional certificate is not a
  supplied completeness guarantee. Captured certificate contents are reread each submission.

Diagnostics and both operation forms capture the declared input and output maps, including nested
weights/status fields and inherited getters, at construction. Mutating the caller's maps later
cannot remove a loss input or redirect the validated output; buffer contents remain per-run data.

Sums are fixed-order tile sums with a pairwise merge on centered columns, and the solve is a unit-diagonal
scaled Cholesky, so there are no atomics and no readback. Moran's variance and `J` are f32 with
cancellation (about 1e-2 relative on the variance and z). The statistics match spreg 1.9.1 (`OLS` with
`spat_diag` and `moran`) on a lattice, a symmetrized kNN scene and two directed row-standardized kNN scenes
(`k = 3`, 60 and 90 rows, asymmetric patterns) to 2e-3 relative. Earlier versions accumulated `W'Z` over each
row's own neighbors only, which biased the Moran variance for asymmetric patterns; that is fixed.

The fitted-lag contribution to `J` is accumulated as a squared projection-residual norm rather
than a difference of squared norms. Both its right-hand side and residual pass subtract the same
observed fitted-lag origin before projection, retaining variation under a large response offset.
When the original Moran trace expression is within 64 machine epsilons of cancellation, both
forms reevaluate variance as `2(n/S0)^2 ||MSM - tr(MS)M/d||²_F / [d(d+2)]`, where
`S=(W+W')/2`, `M` removes the design and `d=n-p`. An observed complete-graph component is removed
before projection to preserve tiny perturbations. Exact degeneracy gives variance zero and NaN
z/p; a small positive variance retains inference. This fallback costs `O(n² p)` work (plus sparse
weight lookups) and `O(np+p²)` GPU storage, and runs only for cancellation-sized trace results.

The 16-row `spatial-regression-diagnostics-audit.oracle.py` cases use dense annihilator matrices
for all six tests and every summary slot, with masked rows, repeated predictor rows, islands,
both remaining-weight conventions, shifted predictors and response rescaling. Node and headless
audit specs compare against the same committed independent values.

```ts
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUOrdinaryLeastSquares, GPUSpatialRegressionDiagnostics, type GPUOrdinaryLeastSquaresProps, type GPUSpatialRegressionDiagnosticsProps} from '@luma-contributors/analysis/gpu-spatial-analysis';

// Fragment inputs: allocate/upload these views in the caller's graph.
declare const graph: GPUCommandGraph;
declare const fit: GPUOrdinaryLeastSquaresProps & {output: {residuals: NonNullable<GPUOrdinaryLeastSquaresProps['output']['residuals']>}};
declare const diagnostics: GPUSpatialRegressionDiagnosticsProps;
const {predictors, response, mask} = fit;
const {coefficients, standardErrors, tStatistics, residuals, summary: olsSummary} = fit.output;
const olsCode = fit.output.status.code;
const {weights} = diagnostics;
const {tests, summary} = diagnostics.output;
const code = diagnostics.output.status.code;
graph.add(new GPUOrdinaryLeastSquares({predictors, response, mask, predictorCount: 2, output: {
  coefficients, standardErrors, tStatistics, summary: olsSummary, residuals, status: {code: olsCode}
}}));
graph.add(new GPUSpatialRegressionDiagnostics({
  weights, predictors, response, residuals, mask, restandardizeRows: true, predictorCount: 2,
  output: {tests, summary, status: {code}}
}));
```

#### How it computes

{/* lens:GPUSpatialRegressionDiagnostics */}

::lens{scenario=gpu-spatial-analysis.spatial-regression-diagnostics rung=formula caption="Follow the LM-error, LM-lag, robust, SARMA and Moran's I equations."}
::lens{scenario=gpu-spatial-analysis.spatial-regression-diagnostics rung=kernel caption="Inspect the tile, merge and finish kernels with retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.spatial-regression-diagnostics rung=graph caption="Trace queen weights, the OLS fit and the six diagnostics over 788 tracts."}

### Weight completeness in spatial model fits

`GPUSpatialErrorGM`, `GPUSpatialTwoStageLeastSquares`, `GPUSpatialWeightsTraces`,
`GPUSpatialLagMaximumLikelihood` and `GPUSpatialErrorMaximumLikelihood` read the supplied
weights' `status.overflow` and `status.candidateOverflow` on every submission. A supplied
`status.complete` is a positive certificate: zero means loss even when both overflow flags
are clear; nonzero does not override another loss signal. A missing certificate supplies no
evidence of completeness and is not required. For chunked weights the direct GPU contributors
read both the root status and each chunk's status. If any loss word is nonzero or a supplied
certificate is zero,
all dependent float outputs are NaN; fitted models also publish `status.code = 3` (unusable,
non-finite/suppressed result) and clear `status.converged` when present. Traces suppress
`infinityNorm`, powers, sums, standard errors and Gram traces; `approximation` retains its
sampling meaning. A later clean submission recomputes usable outputs on the same graph.
Count-only status is also honored: `requiredCount > count` or a requirement above the terminal
CSR offset proves loss, even when spare allocated slots exist. Chunk-root retained count is the
saturating sum of child terminal offsets; each child is also checked separately. Aggregate
capacity and retained sums saturate at `0xffffffff`, the largest uint32 requirement. Counts alone never certify intermediate
candidate completeness; any supplied loss flags are still checked.
Suppression writes only the documented logical entries in each output view. Spare caller
capacity beyond those entries is preserved, including trailing table, summary, residual and
trace values; scalar status outputs write only their first word.
Only the status fields declared by each contributor are captured or written: traces expose
`approximation`, GM and 2SLS expose `code`, and likelihood fits expose `code` and optional
`converged`. Additional fields on a wider caller status map are ignored.
Trace status always has the declared named-map shape, including when the caller's map happens
to expose a `buffer` property; scalar-view handling is reserved for ports that declare it.
Both likelihood contributors require any `powerTraceStandardErrors` destination to be packed
float32 with at least `maximumPower` entries. Invalid destinations are rejected at construction,
so a corrected destination can be declared on the same graph with the same contributor ID.

The GPU named operation forms and CPU twins currently require a complete CSR without a
`weights.status` map. They reject status-bearing weights explicitly, including chunk statuses,
because those forms have no loss-flag input ports. Use the direct GPU contributors to consume
producer ports with per-submission loss flags. Do not remove the flags from a possibly incomplete port.
When importing weights into another graph, import the producer's status buffers along with the
CSR arrays. If a different producer does not write a previously used loss flag, clear that flag
before each submission of the replacement producer, including reuse of a cached graph.
The spatial-regression recipe uses the separate diagnostics contributor, which already handles
its own weights loss flags.
Diagnostics and residual Moran also honor retained-count loss: diagnostics return code 3,
the recipe reports decision 4 (undetermined), and spatial statistics are unavailable. Named
GPU/CPU diagnostics accept `weights.status.count` and `weights.status.requiredCount` graph names.

These five contributors capture declared props (including inherited getters), nested output/status
maps, weights maps and chunk boundaries at construction. Later edits to those maps do not move
validated bindings; GPU buffer contents remain per-frame inputs.

The five regression lens scenarios capture both `weightsOverflow` and the contiguity producer's
`weightsCandidateOverflow`. The readback-to-record boundary refuses either loss word even when
the other is zero, so an incomplete adjacency cannot become a recorded result.

### `GPUSpatialTwoStageLeastSquares`

Estimates a spatial lag model by two-stage least squares, instrumenting the lagged response with neighbouring predictors, and tests the residuals.

::lens{scenario=gpu-spatial-analysis.spatial-two-stage-least-squares rung=result caption="Five rows: intercept, three predictors and rho = 0.55 (se 0.027); compare each with its standard error."}

The spatial lag model `y = rho W y + X b + e` by two-stage least squares (spreg `GM_Lag` with `w_lags = 1`),
with instruments `[1, X, WX]`. Props: `weights`, `predictors`, `response`, `predictorCount` (1 to 8 with
`instrumentOrder: 1`, 1 to 5 with `instrumentOrder: 2`: the constant is centered out, so the solve factors `(order + 1) k`
instrument columns in a 16-wide Cholesky; the order-2 cap was previously 8 in the check but failed when building the
graph), optional `tileRowCount`, `instrumentOrder`, `robust` and `output: {table, summary, status: {code}, residuals?}`.
Construction permits `tileCount = ceil(rows / tileRowCount) <= 2^24`, subject to device binding limits.
Above 4,096 tiles the sum and norm merges use the hierarchical order below.
`GPUSpatialTwoStageLeastSquaresOperation` applies this check when lowered to its GPU contributor.
The CPU twin ignores GPU tiling. In `computeSpatialWeightsTracesOnCPU`, a CSR with
`offsets`, `neighbors` and `weights` is one numerical matrix even when it also carries
recursive loss metadata in `chunks`. Loss evidence at the root and in metadata children
is still checked. Numerical chunked input instead supplies `rowCount` and contiguous
`{rowStart, rowCount, weights}` chunks covering every row; only this numerical chunked
layout rejects Gram traces.

The named CPU trace operation refuses completeness certificate ports, including flat overflow
and count aliases; use the pure function with concrete certificates or the GPU contributor with
GPU status views.
No kernel binds more than 8 storage buffers.
`robust: {method: 'HC0' | 'HC1' | 'HC2' | 'HC3' | 'CLUSTER' | 'HAC', kernel?, clusters?, clusterCount?, output: {covariance, status: {code}}}` replaces the standard errors, z
statistics and p-values of `table` by those of a heteroskedasticity-consistent, one-way clustered CR1 or spatial HAC covariance
(`GPUInstrumentalVariablesCovariance`; HAC needs `kernel`, a symmetric positive semidefinite weights CSR with a unit
diagonal, distinct from the model weights) and writes the full `(k + 2) x (k + 2)` covariance and its own status.
The [instrumental-variable covariance guide](instrumental-variables-covariance.md) defines the
HC corrections, projected-design leverage, cluster-key limits, PSD certificate and every solver
status for the direct contributor and both operation forms.
The HAC kernel's `status.overflow` and `status.candidateOverflow` suppress only robust inference
with covariance code 4; point estimates and classical diagnostics remain available. Both named
operation forms accept loss graph-value names under `robust.kernel.status` and recover when the
flags clear on a later run. The pure CPU function accepts the corresponding arrays at the same path.
Count-proven HAC loss has the same covariance code 4 and preserves the point fit. Named forms
also accept `robust.kernel.status.count` and `robust.kernel.status.requiredCount`. The pure CPU
function compares the requirement with retained CSR slots.
The direct GPU HAC path also consumes a supplied `robust.kernel.status.complete`: zero
suppresses covariance with code 4 even if the overflow flags are clear and counts fit.
The model's coefficients, structural residuals and classical summary remain usable; the robust
covariance and table SE/z/p columns become NaN. Clearing kernel loss restores inference on a
later submission without changing the point fit. Use only the status input names declared by
the selected operation form; a direct GPU kernel's nested certificate is a separate port.
The point estimates and the `summary` diagnostics keep their classical definitions. The displayed
lens and its classical formula retain the default without `robust`. Its recorded `robust.method`
choice is `none`, explicitly distinguishing classical covariance from the optional robust methods.
`instrumentOrder: 2` (compile-time, default 1) adds `W^2 X` (spreg `w_lags=2`, instruments `[1, X, WX,
W^2 X]`), evaluated per row from the two-hop neighborhood with no extra buffers; it is pinned to spreg
1.9.1 `GM_Lag(w_lags=2)` on three scenes.

- `table`: `(k + 2) x 4` f32 rows of `[coefficient, se, z, two-sided normal p]` in spreg order: the
  intercept, then `X`, then `rho` last.
- `summary` (16, `GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_*`): `n`, `sigma2 = u'u / n` (spreg `sig2n`, the `GM_Lag`
  default), `u'u`, the pseudo R2 (squared correlation of `y` and `y - u`), Moran's `I` of `u`, the
  Anselin-Kelejian statistic and its p-value (slots 0 to 6). The statistic follows spreg `akTest(case='gen')`; any
  sparsity pattern is accepted (`W'u` comes from `GPUSpatialWeightsTranspose`), and it matches spreg 1.9.1 on
  directed kNN scenes. The total weight `S0` cancels from the statistic, which is computed as
  `(u'Wu / sigma2)^2 / (T + 4 a / sigma2)`: for signed weights whose total is zero the statistic is defined
  (0.95057 on an eight-row scene with weights of +1 and -1, as the cancelled formula on spreg's fit; it was
  NaN), while Moran's `I`, which divides by `S0`, is NaN. Diagnostics (slots 7 to 15; none changes `status`):
  - Conditioning of the second stage: the Belsley condition number of the raw `[1, PZ]` Gram matrix (`Z = [X, Wy]`,
    `P` the projection on the instruments), the correlation condition number of `Z'PZ` and its largest variance
    inflation factor, the correlation condition number of the instrument system `H'H` (the instrument-projection
    system), and a precision warning (1.0) when either correlation condition number exceeds 1e4 (float32
    normal equations may have lost three of seven digits). Computed by the Jacobi eigenvalue routine of
    `spatial-regression-solve`.
  - First-stage strength: the classical joint F statistic and the partial R2 of the excluded instruments (`WX`, and
    `W^2 X` for order 2) in the regression of `Wy` on `[1, X, WX, ...]`; `F = ((ESS_full - ESS_restricted) / (m - k)) /
    (RSS_full / (n - m - 1))`, partial R2 `= (ESS_full - ESS_restricted) / (TSS - ESS_restricted)`. The usual
    weak-instrument rule of thumb is `F < 10`. Both are the non-robust (homoskedastic) forms and are pinned to a numpy
    computation of the same definitions. F is NaN when the first-stage denominator has no
    residual degrees of freedom (`n <= m + 1`); the point fit and partial R2 can remain valid.
  - Admissibility of `rho`: `||W||_inf` (largest absolute row sum) and a flag that is 1.0 when `|rho| ||W||_inf >= 1`,
    that is when `I - rho W` is not guaranteed invertible and the reduced form and impacts are unreliable
    (`(-1, 1)` for a row-standardized `W`; sufficient, not necessary, for other `W`). It is a flag, not a failure:
    2SLS is unconstrained, the estimate is still reported, and the reader decides what a boundary or explosive `rho`
    means for the model.
- `status.code`: 0 ok, 1 singular or ill-conditioned (including non-finite inputs, and a predictor that is constant
  at float32 resolution, as for the other estimators), 2 too few rows, 3 non-finite sums or results suppressed for incomplete weights.
- `residuals`: the structural residuals `u = y - Z delta`, which can be passed to later diagnostics. They are
  formed on the centered columns and sum to zero: the part of each mean that float32 cannot hold is carried
  through the fit, and the moments are corrected to moments about the true means.
  A late failure can retain finite GPU residuals, including a saturated fit with code 2; check
  the point-fit code before passing them to downstream diagnostics. Spare output capacity is
  allowed; robust covariance consumes only the active row prefix and preserves its view offset.
  The explorer and showcase write separate map display buffers under the same submission's
  fit code: failed fits display no residuals. Showcase tooltips use the code read with each result
  and suppress failed-fit residuals; a later successful submission restores both displays.
- The lag of the response is formed about the response mean, `W y = W (y - c) + c s_i`, with the row sums `s_i`
  of the float32 weights carried exactly (rows without neighbours included); a sum of `w * y` over the raw
  response is rounded to a float32 step of the response, which cost up to 4.2% of `sigma2` and 45% of the
  Anselin-Kelejian statistic for residuals of 0.002 to 0.004 around 10,000.
- One step of iterative refinement follows the solve (three more nodes). The float32 solve lands a few float32
  steps of `rho` from the estimate. Where the lag differs between rows by the response mean (rows without
  neighbours, row sums that differ) the two-stage `rho` is not at the minimum of `u'u`, and three and five such
  steps were 3% and 0.8% of `sigma2` (the float32 value nearest to the estimate would have cost 0.07% and
  0.03%: it was the solve, not the resolution of `rho`). The refinement forms the residuals at the float32
  coefficients without the rounding of their products, sums `H'u` and solves the same system for the
  correction. The correction is kept beside the float32 coefficients: `residuals`, `sigma2`, the standard
  errors and the Anselin-Kelejian statistic are those of the corrected estimate, and `table` holds the float32
  nearest to it.
- Measured on the test GPU against spreg `GM_Lag` and a centered float64 solve, for residuals of 0.002 to
  0.004 around 10,000:
  - rows that share a row sum, exactly or up to the rounding of the weights (ring at 0.5, six neighbours at
    float32 1/6, binary weights with row sum 2, row sums of 1 and 1.0000000298, `rho` 0.2 to 0.8, and a second
    predictor near 4,096): `sigma2` within 2.8e-6, the Anselin-Kelejian statistic within 2.4e-4, `rho` within
    6e-4 standard errors;
  - row sums that differ with such a response: three rows without neighbours, `sigma2` within 9.2e-6 and the
    statistic within 9.1e-6; row sums of 2 and 3, 9.5e-7 and 4.9e-7;
  - slopes two hundred times the residual deviation over the predictor range: `sigma2` within 1.6e-4, because
    the fitted values are rounded to about 1% of a residual.
  `W X` (the instruments) is still summed from the raw predictors.

The coefficients, standard errors, `sigma2`, pseudo R2 and Anselin-Kelejian statistic match spreg on a
144-row scene within 5e-3 (coefficients) to 2e-2 (z, AK). The default standard errors are the classical
`sigma2 (Z'PZ)^-1`; the `robust` prop gives HC0/HC1/HC2/HC3, clustered CR1 or spatial HAC ones.
`GM_Combo` and maximum-likelihood models are not provided here. For the spatial error model see
[`GPUSpatialErrorGM`](#gpuspatialerrorgm).

The 12-row `spatial-two-stage-least-squares-audit.oracle.py` fixtures independently form the dense
SVD instrument projector, classical and robust covariance matrices, first-stage strength and
Anselin-Kelejian diagnostics. Both instrument orders include repeated predictor rows, an island,
response rescaling and shifted predictors. Node and headless audit specs compare every output
against these literals.

`getGPUSpatialTwoStageLeastSquaresModelResult(readback, {predictorCount?, predictorNames?})` turns the read-back `table`, `summary`
and `status` into the shared `GPURegressionModelResult`: rows `CONSTANT`, the predictors and `W_y` (role
`'spatial-lag'`), z statistics with float64 normal p-values, `sigmaSquaredMaximumLikelihood = u'u / n` (the scale
of the standard errors, `standardErrorSigmaSquared: 'maximum-likelihood'`), `pseudoRSquared`, no likelihood,
`tests.anselinKelejian` and conditioning (the correlation condition number is the larger of the second-stage
and instrument values, so `illConditioned` equals the precision warning).
For a robust table, also supply `readback.robust: {method, status}` (present on the pure CPU result).
The decoder labels its errors `'robust'` and returns a separate `covarianceInference` method and
status. The point-fit `status.ok` remains independent of covariance success or failure.
`getGPUSpatialTwoStageLeastSquaresDiagnostics(summary)` returns the first-stage F, partial R2, `||W||_inf` and the
`rho` admissibility flag. The flag is `false` for a measured 0, `true` for a measured 1, and
`null` when unavailable (including the NaN summary of an incomplete-weight fit with code 3).
Pass the fitted `predictorCount` when the read-back table has spare capacity; only the active
`(predictorCount + 2) * 4` prefix is decoded. Omit it only for an exact-length table. Predictor
names label columns and do not determine the fitted count.

#### How it computes

{/* lens:GPUSpatialTwoStageLeastSquares */}

::lens{scenario=gpu-spatial-analysis.spatial-two-stage-least-squares rung=formula caption="Follow the instrumented estimator, residual variance, z statistics and Anselin-Kelejian test."}
::lens{scenario=gpu-spatial-analysis.spatial-two-stage-least-squares rung=kernel caption="Inspect the moment, solve, residual and finish kernels with retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.spatial-two-stage-least-squares rung=graph caption="Trace queen weights, the two stages and the Anselin-Kelejian test over 788 tracts."}

### `GPUSpatialWeightsTraces`

Computes the traces and sums of powers of a weights matrix, the inputs of spatial log-determinants and impacts.

::lens{scenario=gpu-spatial-analysis.spatial-weights-traces rung=result caption="Traces of 12 queen-weight powers fall from 122 at q = 2 to 19; sums stay at 788."}

Powers of a spatial weights matrix: `powerTraces[q-1] = tr((W/s)^q)` and `powerSums[q-1] = 1' (W/s)^q 1` with normalization scale `s = output.infinityNorm`. `infinityNorm` is the scale used to form powers:
it retains the largest float32 absolute row sum when that sum is positive and finite, and uses 1
when the computed norm is zero or nonfinite. With a positive finite norm, the series runs in
`rho * s` with bounded normalized powers. Powers cover
`q = 1..Q`. These are the inputs of the spatial-lag log-determinant `ln|I - rho W| = -sum_q rho^q tr(W^q) / q`
and of the impact multipliers (see `GPUSpatialLagImpacts`).
For example, one `3.2e38` edge per row gives a GPU scale near `3.19999998e38`. At a scale of
2¹²⁶ or greater, the GPU’s division path rescales both operands to keep the reciprocal normal.

For optional cubic and quartic Gram probe moments, the GPU checks the exponents of live nonzero
weights and the row norm. It keeps the original scalar calculation when the operands are normal
and the exponent bounds, including headroom for edge, row and probe reductions, fall within
`[-114, 123]`. All-zero weights also take the scalar route. Otherwise it uses raw weights and
carries separate exponents through the reductions. For `x = Wz`, the protected probe terms are
`x'(Wx)` and `||W'x||²`. The significands and published results have float32 precision. Final
results can still overflow at publication, and the other two Gram moments use raw float32 products.

Props: `weights` (any square CSR: asymmetric,
not standardized, self-loops allowed), `maximumPower` (`Q`, compile-time, 1 to 256), optional `mode`,
`probeCount`, `parameters` (per-frame `[seed]`, `getGPUSpatialWeightsTracesParameterValues({seed})`; omitted means seed 0),
`includeGramTraces`, `tileRowCount`, `probeBatchSize` and
`output: {powerTraces, powerSums, gramTraces?, status?: {approximation?}, infinityNorm?}` (float32 views of at least `Q`, `Q`, 4 values
and a uint32 scalar).
Construction permits at most `REGRESSION_TILE_MAXIMUM_TILE_COUNT` (`2^24`) tiles, in both exact and stochastic modes,
subject to device binding limits. Above 4,096 tiles sum and maximum partials use hierarchical merges.
For whole weights the count is `ceil(rows / tileRowCount)`; for chunked weights it is the sum of
`ceil(chunk.rowCount / tileRowCount)` over chunks. The default whole-CSR layout targets 1024 tiles;
explicit one-row tiles and chunk boundaries can exceed that target without hitting a 4,096-tile ceiling.
`GPUSpatialWeightsTracesOperation` applies the same check during lowering, including chunked inputs.
The CPU twin ignores GPU tiling. In `computeSpatialWeightsTracesOnCPU`, a CSR with
`offsets`, `neighbors` and `weights` is one numerical matrix even when it also carries
recursive loss metadata in `chunks`. Loss evidence at the root and in metadata children
is still checked. Numerical chunked input instead supplies `rowCount` and contiguous
`{rowStart, rowCount, weights}` chunks covering every row; only this numerical chunked
layout rejects Gram traces.

The named CPU trace operation refuses completeness certificate ports, including flat overflow
and count aliases; use the pure function with concrete certificates or the GPU contributor with
GPU status views.
The contributor captures chunk boundaries and nested weight-view references at construction;
later edits to the caller's chunk array or port maps do not change the validated dispatch layout.
GPU buffer contents remain per-frame inputs.

- `mode: 'exact'` runs one unit-vector chain `v_q = W v_{q-1}` per row and sums the diagonal entries
  `v_q[i]`: exact up to float32 rounding. With mode omitted, it is selected for at most 4,096
  total rows (`GPU_SPATIAL_WEIGHTS_TRACES_AUTO_EXACT_MAXIMUM_ROWS`); above that the default is
  stochastic. Explicit `mode: 'exact'` accepts at most 16,384 total rows
  (`GPU_SPATIAL_WEIGHTS_TRACES_EXACT_MAXIMUM_ROWS`), subject to the command and device limits.
  The work is `rows * Q * nnz`, which at 4096 rows,
  `Q = 256` and 8 neighbors is about 3e10 multiply-adds in about 520 dispatches, while 16384 rows (the limit
  of an explicit `'exact'`) already costs 8200 dispatches and 5e11 multiply-adds.
  `GPUSpatialWeightsTraces.getExecutionPlan(props, device.limits)` checks the selected route,
  chunk-local tiles, probe scratch, command schedule and device bindings before construction.
- `mode: 'stochastic'` is Hutchinson estimation with `probeCount` (default 64) Rademacher probes from Philox
  (counter `(probe, row / 4, 0, 'PROB')`, key `(seed, 'TRAC')`, the seed read on the GPU, so changing it never
  recompiles; a seeded run is reproducible and
  the test-only CPU reference in `modules/analysis/test/reference` replays the same probes). The standard error of `tr(A)` is at most
  `sqrt(2 / probeCount) ||A||_F`. `output.status.approximation` is 1 (it is 0 in exact mode).
- `powerSums` is always exact: one chain from the ones vector.
- `gramTraces` (when `includeGramTraces`, 4 values) are `[tr(W'W), tr(W'W W), tr((W'W)^2), sum_i c_i^2]` with
  `c_i = sum_l w_li^2`: the moments of the Kelejian-Prucha and Anselin-Kelejian variance formulas. `tr(W'W)` and
  `sum c_i^2` are exact sums (the column sums use `GPUSpatialWeightsTranspose`); the other two are
  `E[(Wz)'(WWz)]` and `E[||W'(Wz)||^2]` over the same probes, exact in exact mode.

Probes run in batches whose two ping-pong vectors stay within 64 MiB. Every dot product is a fixed-order tile
sum with a pairwise merge, so there are no float atomics and results are bit-identical across batch sizes and
runs. The shaders do not depend on the power, the batch or the probe index: per-power and per-batch tables
are 64-element aligned sub-views and a GPU batch table holds each batch's first probe and probe count, so
the pipeline count does not grow with `maximumPower` or the number of batches. A `W` with spectral radius above 1 grows as `radius^q` and can overflow float32 for large `Q`.

Row-tile merge order (also used by error GM, 2SLS and both likelihood estimators): through 4,096
tiles, existing nodes, IDs and summation order are unchanged. Above 4,096, consecutive groups of
at most 4,096 tiles reduce independently, in tile order. Each sum uses a binary-counter cascade:
merge equal-size blocks as each tile arrives, then fold leftover blocks from smallest to largest.
Apply that same cascade to the group totals in group order. Maximum slots use maxima at both levels.
The additional binding holds `ceil(tileCount / 4096) * termCount` float32 words, included in allocation
preflight. Two levels support `REGRESSION_TILE_MAXIMUM_TILE_COUNT = 4096^2 = 2^24` tiles, so the tile
limit does not bind before the float32 row-count limit, including one row per tile. Device buffer
limits and each contributor's other work limits still apply. Results are reproducible for a fixed layout.
`estimatedCommandNodeCount` on weights traces, error GM and both likelihood contributors includes chunk
dispatches, probe batches, group passes and nested fit/trace work. Construction rejects estimates
above the shared 32,768-command budget before declaring resources. Reduce powers or probe counts,
increase `probeBatchSize` where available, or combine small CSR chunks to reduce command count.
The two-level tile limit does not grant an exception to this separate budget.

CPU float64 helpers: `getSpatialLogDeterminantFromTraces(rho, powerTraces, infinityNorm)`, `getSpatialMultiplierTraceFromTraces(rho, powerTraces,
n, infinityNorm)` (`= tr((I - rho W)^-1) = n + sum_q rho^q tr(W^q)`),
`getSpatialMultiplierTotalFromPowerSums(rho, powerSums, n, infinityNorm)` and the truncation bounds
`getSpatialLogDeterminantTruncationBound(rho, Q, n, infinityNorm)` and
`getSpatialMultiplierTruncationBound(rho, Q, n, infinityNorm)`. Pass the traces' published normalization scale
`infinityNorm` (including the all-zero fallback 1) for raw weights. A spectral radius alone cannot bound total multipliers
on directed weights (a directed path has radius zero but nonzero omitted powers). WGSL generators
`getSpatialLogDeterminantWGSL(Q, bindingName)`, `getSpatialMultiplierTraceWGSL(Q, bindingName, n)` and
`getSpatialMultiplierTotalWGSL(Q, bindingName, n)` define `getSpatialLogDeterminant(rho)` and friends by
Horner's rule over an f32 storage binding. The series require `|rho| * spectralRadius(W) < 1` (`|rho| < 1` for
a row-standardized `W`) and are truncated. With `t = |rho| * infinityNorm < 1`, the log-determinant
error is below `n t^(Q+1) / ((Q+1)(1 - t))` and the multiplier error below `n t^(Q+1) / (1 - t)`.
Choose `Q` from `t^Q` (`Q = 30` is accurate to 1e-9 at `t = 0.5` but needs about 300 terms at `t = 0.9`).

The traces match `np.trace(np.linalg.matrix_power(W, q))` for `q = 1..12` on a row-standardized lattice, a
binary lattice and two directed kNN scenes (numpy 2.5.4), and the log-determinant series matches
`np.linalg.slogdet` within the stated bound.

```ts
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUSpatialWeightsTraces, type GPUSpatialWeightsTracesProps} from '@luma-contributors/analysis/gpu-spatial-analysis';

// Fragment inputs: allocate/upload these views in the caller's graph.
declare const graph: GPUCommandGraph;
declare const inputs: GPUSpatialWeightsTracesProps;
const {weights} = inputs;
const {powerTraces, powerSums, infinityNorm} = inputs.output;
graph.add(new GPUSpatialWeightsTraces({
  weights, maximumPower: 30, output: {powerTraces, powerSums, infinityNorm}   // exact for <= 4096 rows
}));
```

#### How it computes

{/* lens:GPUSpatialWeightsTraces */}

::lens{scenario=gpu-spatial-analysis.spatial-weights-traces rung=formula caption="Follow the infinity norm, the power traces and the power sums."}
::lens{scenario=gpu-spatial-analysis.spatial-weights-traces rung=kernel caption="Inspect the norm, scaling, probe-chain and merge kernels with retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.spatial-weights-traces rung=graph caption="Trace queen weights, 788 unit-vector chains and 12 power steps."}

{/* parameters: GPUSpatialWeightsTraces generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUSpatialWeightsTraces` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPUSpatialWeightsTraces`.

- Packer: `getGPUSpatialWeightsTracesParameterValues(settings?)`, writes `GPU_SPATIAL_WEIGHTS_TRACES_PARAMETER_LENGTH` (1) `uint32` values.

Settings of `getGPUSpatialWeightsTracesParameterValues`:

| Field | Type | Description |
|---|---|---|
| `seed` (optional) | `number` | Philox seed of the stochastic probes, an integer in `[0, 2^32)`. Defaults to 0. |

{/* /parameters */}

### `GPUSpatialLagImpacts`

Turns spatial lag coefficients and weights traces into average direct, indirect and total effects of each predictor.

::lens{scenario=gpu-spatial-analysis.spatial-lag-impacts rung=result caption="Direct, indirect and total effects of three predictors; poverty's total is 0.42, 61% of it indirect."}

Average direct, indirect and total impacts of the predictors of a spatial lag model (LeSage and Pace 2009):
`direct = b_k tr((I - rho W)^-1) / n`, `total = b_k 1'(I - rho W)^-1 1 / n`, `indirect = total - direct`, from the
truncated series `n + sum_{q<=Q} rho^q tr(W^q)` and `n + sum_{q<=Q} rho^q 1'W^q 1` evaluated in one kernel. Props:
`coefficients` and `coefficientStride` (the `table` of `GPUSpatialTwoStageLeastSquares` with stride 4, or any
fit with a coefficient per row), `rhoIndex` (`predictorCount + 1` for the 2SLS table), optional
`firstPredictorIndex` (default 1, so the intercept is excluded), `predictorCount`, `rowCount`, `maximumPower`,
`traces: {powerTraces, powerSums, infinityNorm}` (the outputs of `GPUSpatialWeightsTraces`), optional `status` (a fit's `output.status.code`) and `rhoBound`
(default 1), and `output: {impacts}`: `predictorCount * 3` f32 values `[direct, indirect, total]` per
predictor.

Impacts are NaN when the fit `status` is nonzero, `rho` is not finite or
`|rho| * infinityNorm >= rhoBound`. The bound applies to the scaled parameter: the default 1
keeps the series convergent for raw and row-standardized weights alike. With raw weights,
do not divide `rhoBound` by the spectral radius again.

The contributor and both operation forms capture scalar options and the trace/output port maps
at construction. Later edits to those caller objects do not change the validated series length,
coefficient layout or output destination. GPU buffer contents remain per-frame inputs.

The direct impact matches spreg 1.9.1 `GM_Lag(..., spat_impacts='full')`, and so does the total for a
row-standardized `W` (spreg uses `1 / (1 - rho)`, which is exact only there; this node uses the exact
`1'W^q1` sums, so it is also right for other weights). spreg's `'simple'` method fixes the direct multiplier at 1
and is not reproduced; its `'power'` method stops after the first term below tolerance, which is wrong on
bipartite lattices where odd traces vanish. Standard errors of the impacts (simulation of `(rho, b)` from the
coefficient covariance) are not provided: the 2SLS contributor does not output the covariance matrix.

```ts
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUSpatialTwoStageLeastSquares, GPUSpatialWeightsTraces, GPUSpatialLagImpacts, type GPUSpatialTwoStageLeastSquaresProps, type GPUSpatialLagImpactsProps} from '@luma-contributors/analysis/gpu-spatial-analysis';

// Fragment inputs: allocate/upload these views in the caller's graph.
declare const graph: GPUCommandGraph;
declare const fit: GPUSpatialTwoStageLeastSquaresProps;
declare const downstream: GPUSpatialLagImpactsProps;
const {weights, predictors, response} = fit;
const {table, summary} = fit.output;
const status = fit.output.status.code;
const {rowCount} = downstream;
const {powerTraces, powerSums, infinityNorm} = downstream.traces;
const {impacts} = downstream.output;
graph.add(new GPUSpatialTwoStageLeastSquares({
  weights, predictors, response, predictorCount: 2, output: {table, summary, status: {code: status}}
}));
graph.add(new GPUSpatialWeightsTraces({weights, maximumPower: 30, output: {powerTraces, powerSums, infinityNorm}}));
graph.add(new GPUSpatialLagImpacts({
  coefficients: table, coefficientStride: 4, rhoIndex: 3, predictorCount: 2,
  rowCount, maximumPower: 30, traces: {powerTraces, powerSums, infinityNorm}, status, output: {impacts}
}));
```

#### How it computes

{/* lens:GPUSpatialLagImpacts */}

::lens{scenario=gpu-spatial-analysis.spatial-lag-impacts rung=formula caption="Follow the direct, total and indirect effect series in rho."}
::lens{scenario=gpu-spatial-analysis.spatial-lag-impacts rung=kernel caption="Inspect the impacts kernel with its retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.spatial-lag-impacts rung=graph caption="Trace the likelihood fit, its power traces and the impacts over 788 tracts."}

### `GPUSpatialErrorGM`

Estimates a spatial error model by generalized moments, then a feasible GLS regression on spatially filtered variables.

::lens{scenario=gpu-spatial-analysis.spatial-error-gm rung=result caption="Five rows: intercept, three predictors and lambda; lambda = 0.83 with standard error 0.015."}

The spatial error model `y = X b + u`, `u = lambda W u + e` by generalized moments. Props: `weights` (any square
CSR; no symmetry or standardization needed), `predictors`, `response`, `predictorCount` (1 to 15, the OLS limit and
the 16-wide Cholesky; no kernel binds more than 8 storage buffers, so the previous cap of 8 was not needed),
`estimator`, optional `tileRowCount`, `traceMode`, `traceProbeCount`, `traceParameters` (per-frame seed, as `GPUSpatialWeightsTraces` `parameters`) and
`output: {table, summary, status: {code}, residuals?}`.
For either estimator, construction permits at most `2^24` tiles, subject to device binding limits:
`ceil(rows / tileRowCount)` for whole weights, or the sum of `ceil(chunk.rowCount / tileRowCount)`
over chunks. Above 4,096 tiles the sum and maximum merges use the hierarchical order above.
CPU twins do not use tile merges and retain their existing row limits.
`GPUSpatialErrorGMOperation` applies the whole-CSR tile-count check during lowering.
The contributor captures chunk boundaries and nested weight-view references at construction,
so later edits to the caller's chunk layout cannot change the validated tile count.

**Default change.** `estimator` is `'homoskedastic'` by default (spreg 1.9.1 `GM_Error_Hom`, `A1 = 'hom_sc'`,
one iteration: Drukker, Egger and Prucha 2013, Anselin 2011), which gives `lambda` a standard error, z and p-value.
The previous behavior is `estimator: 'kelejian-prucha'` (spreg `GM_Error`, Kelejian and Prucha 1998 and 1999), whose
`lambda` row keeps NaN for se, z and p; in both cases the standard errors of the betas condition on the
estimated `lambda`.

- Steps (both): OLS of `y` on `[1, X]` gives `u`; moments of `u` give `lambda`, minimized globally on a 1024-interval
  grid with bisection of every bracketed minimum (spreg runs a local L-BFGS-B from zero; the two agree whenever the
  objective has a single basin); OLS of `y - lambda W y` on the filtered `[1, X]` (the constant is filtered too, as
  `get_spFilter` does) gives the coefficients.
- `lambda` bound: `0.99 / ||W||_inf` with `||W||_inf` the largest absolute row sum, computed on the GPU. It equals
  spreg's fixed 0.99 for a row-standardized `W` and keeps `I - lambda W` invertible for any `W`; a fixed 0.99
  admitted explosive `lambda` for raw weights. A solution on the bound is flagged
  (`..._SUMMARY_LAMBDA_AT_BOUND`) and `..._SUMMARY_LAMBDA_BOUND` reports the bound.
- `homoskedastic` replicates spreg step for step: (a) `lambda1` minimizes the unweighted
  `|G [l, l^2]' - g|^2` of the moments `u'A1u`, `u'A2u` with `A1 = (W'W - (t/n) I) / (1 + (t/n)^2)`, `t = tr(W'W)`,
  `A2 = (W + W') / 2`; (b) the FGLS betas at `lambda1` are the reported betas and give `u1 = y - X b1`; (c) the
  moments are recomputed on `u1` and `psi`, the variance of the moments (from `sigma2`, `mu3`, `mu4` of
  `u1 - lambda1 W u1` and the traces `tr W`, `tr W^2`, `tr W'W`, `tr W'WW`, `tr (W'W)^2`, `sum_i c_i^2` of
  [`GPUSpatialWeightsTraces`](#gpuspatialweightstraces)), weights the second minimization
  `(G p - g)' psi^-1 (G p - g)` giving `lambda`; (d) `Var(b) = sigma2 (Xs'Xs)^-1` with `Xs` and `u_s` filtered at
  `lambda` (`sigma2 = u_s'u_s / n`) and `Var(lambda) = (j' psi^-1 j)^-1 / n`, `j = G [1, 2 lambda]'`. The
  closed forms replace spreg's `n x n` `A1`, `A2` and are verified against spreg's own `psi`, `G`, `lambda` and
  standard errors (the betas-`lambda` covariance of spreg is not formed; it changes no reported number).
  Pinned to spreg on a row-standardized lattice, an asymmetric kNN scene and a non-row-standardized (binary) lattice.
- Traces are exact up to 4096 rows and Hutchinson estimates above (`traceMode` overrides). With stochastic traces
  the weighted second-step `lambda`, its standard error and the betas' standard errors are approximate and
  `..._SUMMARY_APPROXIMATION` is 1.0. Reported betas and residuals come from the exact first step and are unaffected.
- `table`: `(k + 2) x 4` f32 rows of `[coefficient, se, z, two-sided normal p]`: the intercept, `X`, then
  `lambda` last.
- `summary` (13, `GPU_SPATIAL_ERROR_GM_SUMMARY_*`): `n`, `lambda`, `sigma2 = e'e / n` (spreg `sig2n`), pseudo R2,
  the residual sum of squares `u'u` and the moment objective at the solution, then conditioning of the final
  filtered regression (Belsley condition number of the raw filtered `[1, X]`, correlation condition number of the
  factored Gram matrix, largest predictor variance inflation factor, and a precision warning above a correlation
  condition number of 1e4 that never changes `status`), the `lambda` bound, the approximation flag and the
  at-bound flag.
- `status.code`: 0 ok, 1 singular (OLS or filtered fit), 2 too few rows (`n <= k + 2`),
  3 non-finite or unidentified moments (including `W` without links).
  An exactly affine response has zero innovations and cannot identify lambda; the CPU twin
  certifies this relation on the original input values and returns code 3. The GPU proposes
  coefficients from at most 256 interpolation rows (falling back to OLS coefficients), plus
  a common-denominator rational proposal. Continued fractions use at most 24 steps, denominators
  up to 65536 per slope and a common denominator up to `2^24`; the intercept numerator is formed
  from an original row. Independent integer checks on **every** original row verify either one
  complete dyadic relation or one complete common-denominator relation, including `7y=x`.
  No residual-size threshold suppresses a genuine one-ULP departure. This sufficient GPU
  certificate requires finite normal nonzero product operands and a product-exponent span of
  at most 200; unsupported or undiscovered relations leave ordinary GM in effect. The CPU
  exact-rank certificate is broader.
  Homoskedastic GPU covariance
  inversion scales its 2x2 moment matrix before taking determinants, preserving response-scale
  equivariance when the covariance entries fit float32.
  Pseudo R² normalizes covariance by the two standard deviations before squaring. Runtime
  integer bit barriers preserve both divisions against shader-compiler reassociation, avoiding
  overflow when the correlation is finite but the products of sums of squares exceed float32.
- `residuals`: `u = y - X b`, formed in the centered model as `(y - ybar) - a0 - (x - xbar) b`, so they and the
  filtered residuals behind `sigma2` keep their accuracy when the response mean is far above them (residuals of
  0.002 to 0.004 around 10,000 match spreg in float64 within 1e-4 in `sigma2` and `u'u`).
  A later report failure can retain finite residuals from the first filtered fit; check `status.code` before use.

`getGPUSpatialErrorGMModelResult(readback, {predictorCount?, predictorNames?})` turns the read-back `table`, `summary` and `status`
into the shared `GPURegressionModelResult`: rows `CONSTANT`, the predictors and `lambda` (role `'spatial-error'`),
z statistics with float64 normal p-values, `sigmaSquaredMaximumLikelihood = e'e / n` (and
`residualSumOfSquares = n sigma2` of the filtered residuals, the scale of the standard errors), `pseudoRSquared`,
no likelihood and the conditioning above.
Pass the fitted `predictorCount` for a table with spare capacity; padding after the active
`(predictorCount + 2) * 4` values is ignored. Omitting the count requires an exact-length table.

Accumulations use the fixed-order tile and pairwise-merge scheme of `GPUOrdinaryLeastSquares`, so results are
deterministic. The spec matches spreg on a 10x10 lattice and an asymmetric kNN scene (`k = 4`, 80 rows):
`lambda` and the coefficients within 5e-3, standard errors within 1e-2 and z within 2e-2 (relative), for both
estimators. The `GM_Error_Het` variant and the Kelejian-Prucha weighted estimator are not provided.

The 12-row `spatial-error-gm-audit.oracle.py` fixtures independently form dense quadratic moments,
use SciPy scalar minimization and NumPy SVD fits, and include repeated predictor rows, an island,
shifted/rescaled inputs and seeded stochastic traces. Node and headless audit specs compare the
complete table, summary, residuals and status against the same literals.
The homoskedastic estimator requires whole weights because its Gram traces need a whole-CSR
transpose; use whole weights or `estimator: 'kelejian-prucha'` for chunked input.

```ts
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUSpatialErrorGM, type GPUSpatialErrorGMProps} from '@luma-contributors/analysis/gpu-spatial-analysis';

// Fragment inputs: allocate/upload these views in the caller's graph.
declare const graph: GPUCommandGraph;
declare const inputs: GPUSpatialErrorGMProps;
const {weights, predictors, response} = inputs;
const {table, summary} = inputs.output;
const status = inputs.output.status.code;
graph.add(new GPUSpatialErrorGM({weights, predictors, response, predictorCount: 2, output: {table, summary, status: {code: status}}}));
```

#### How it computes

{/* lens:GPUSpatialErrorGM */}

::lens{scenario=gpu-spatial-analysis.spatial-error-gm rung=formula caption="Follow the moment system, the lambda search and the filtered regression."}
::lens{scenario=gpu-spatial-analysis.spatial-error-gm rung=kernel caption="Inspect the moment, lambda scan, filtered regression and finish kernels with retained WGSL."}
::lens{scenario=gpu-spatial-analysis.spatial-error-gm rung=graph caption="Trace the OLS stage, traces, moment steps and filtered fit over 788 tracts."}

### `GPUSpatialLagMaximumLikelihood`

Estimates a spatial lag model by maximum likelihood, searching rho on the concentrated likelihood with a trace-series log-determinant.

::lens{scenario=gpu-spatial-analysis.spatial-lag-maximum-likelihood rung=result caption="Five rows: intercept, three predictors and rho; rho = 0.65 with standard error 0.021."}

Maximum-likelihood estimator of the spatial lag model `y = rho W y + X b + e` (spreg 1.9.1 `ML_Lag`, `method='full'`),
with the exact log-determinant replaced by a truncated trace series. Nothing is read back. Props: `weights` (any
square CSR), `predictors`, `response`, `predictorCount` (1 to 14), `maximumPower` (series length `Q`, 2 to 256,
default 128), `traceMode`, `traceProbeCount`, `traceParameters` (per-frame seed), optional `tileRowCount` and
`output: {table, summary, status: {code, converged?}, residuals?, powerTraces?, powerSums?, infinityNorm?}`. Provide `powerTraces`,
`powerSums` and `infinityNorm` together to share the traces with a downstream [`GPUSpatialLagImpacts`](#gpuspatiallagimpacts).
The bundled trace contributor uses the [standalone normalization and power calculation](#gpuspatialweightstraces).
Its `infinityNorm` retains the maximum absolute row sum computed in float32 when positive and finite.
A zero or nonfinite computed norm gives scale 1.
Main moment tiles default to `max(64, ceil(rowCount / 4096))` rows. Lag main moments span global
rows; its CSR norm stages preserve chunk boundaries. Error-ML main moments and norm stages
both preserve chunk boundaries, so their tile count sums `ceil(chunk.rowCount / tileRowCount)`
over nonempty chunks. The main width's 4,096 target is not a rejection at 4,097 tiles.
Main and nested merges use bounded groups of at most 4,096 partials, accepting at most
16,777,216 (`4096^2`) tiles, subject to bindings and the independent 32,768-command budget.
The likelihood contributor resolves its nested trace layout at construction, starting with
`max(32, ceil(rowCount / 1024))` rows per tile independently of the main width and doubling it if chunk boundaries produce more than 4096 tiles.
Historical width choices are retained when the nonempty-chunk count can fit one level.
If that count already exceeds 4096, the initial width is kept (widening cannot remove a chunk),
and the trace child uses hierarchical merges up to `REGRESSION_TILE_MAXIMUM_TILE_COUNT` (`2^24`)
tiles. The row, binding and command-graph work limits still apply. The captured layout is used
for both allocation planning and the trace child;
weights port maps and chunk boundaries are also captured, while GPU contents remain per-frame inputs.
The GPU operation form supports whole CSR and applies its tile checks when lowered. Use the
contributor directly for chunked CSR; mixed-executor operations require packed input vectors.
CPU twins do not use trace tiling and retain their float64 reduction order.

- Steps: the OLS residuals `e0`, `e1` of `y` and of the lag of `y` on `[1, X]` (two `GPUOrdinaryLeastSquares`
  compositions) give the concentrated likelihood `-n/2 ln(s2(rho)) + ln|I - rho W|`,
  `s2 = (e0 - rho e1)'(e0 - rho e1) / n`. The lag is formed about the response mean,
  `W y = W (y - ybar) + ybar s_i`, with the row sums `s_i` of the float32 weights carried exactly (rows without
  neighbours included), because a sum of `w * y` over the raw response is rounded to a float32 step of the
  response. The search evaluates `u'u` as `S + e1'e1 (rho - rho*)^2`, with `S` the sum of squares of
  `e0 - rho0 e1` at `rho0 = e0'e1 / e1'e1` summed per row and `rho*` its minimizer, not as
  `e0'e0 - 2 rho e0'e1 + rho^2 e1'e1`: where the model fits well (`u'u` below 1e-7 of `e0'e0`) that difference
  of three float32 sums is rounding, and the search then failed (status 3) or returned a wrong `rho` with
  status 0. The log-determinant is `-sum_{q<=Q} rho^q tr(W^q) / q` from
  [`GPUSpatialWeightsTraces`](#gpuspatialweightstraces) (exact traces up to 4096 rows, Hutchinson above).
  `rho` is the global maximum on `[-B, B]`: one kernel evaluates the analytic derivative on 1025 grid points,
  bisects (32 steps) every positive-to-non-positive sign change on the derivative, and keeps the best candidate
  (an end point counts when the derivative points outward). It is deterministic and has no iteration budget.
  `b = b0 - rho b1`, `s2 = u'u / n`.
- Variance: spreg's information matrix of `(b, rho, s2)` with `tr1 = tr(W A)` and `tr2 = tr((W A)^2)` as series in
  `rho` of the power traces, and `tr3 = tr((W A)' W A)`, `A = (I - rho W)^-1`, from a second probe pass at the
  estimated `rho` (Horner `u <- z + rho W u` per probe, exact unit probes up to 4096 rows, Rademacher above; the
  probes of `GPUSpatialWeightsTraces`). The `s2` row is eliminated by its Schur complement, the matrix is built on
  centered predictors and on `W A X b` about its mean (two more nodes; for a response with a large mean the raw
  column cancels against the intercept in float32, which gave standard errors of `rho` and of the intercept a
  half to a third of spreg's, or NaN), unit-scaled and factored with the shared Cholesky, and the intercept
  variance is mapped back to the uncentered model. A positive definite failure leaves the standard errors NaN
  (`..._SUMMARY_VARIANCE_DEFINED` 0) without changing `status`. The coefficient block equals spreg's `vm`.
- Search interval and truncation. `B = min(0.99, seriesLimit) / ||W||_inf` with `||W||_inf` the largest absolute row
  sum computed on the GPU (0.99 for a row-standardized `W`; keeps `I - rho W` invertible for any `W`, whereas spreg's
  fixed `(-1, 1)` admits explosive `rho` for raw weights). The series omits a tail below
  `n s^(Q+1) / ((Q+1) (1 - s))`, `s = |rho| ||W||_inf` (worst case, `|tr(W^q)| <= n s^q`;
  `getSpatialLogDeterminantTruncationBound`), and omitting it biases the likelihood toward large positive `rho`, so
  the interval also stops at the `seriesLimit` where the worst-case tail reaches `1e-5 n`
  (`getSpatialMaximumLikelihoodSeriesLimit`: about 0.93 for `Q = 128`, 0.96 for `Q = 256`). Beyond it the fit is a
  boundary solution (`..._SUMMARY_AT_BOUNDARY`, `..._SERIES_LIMITED`, `converged` 0); raise `maximumPower` when
  `rho` is near the bound. The default `Q = 128` keeps the worst-case truncation below `1e-7 n` log-likelihood units
  at `|rho| <= 0.9` (below `1e-6 |LL|`) at the cost of about `Q (1 + probe batches)` dispatches; the probe pass costs
  `rows x Q x nnz` multiply-adds in exact mode. `..._SUMMARY_TRUNCATION_BOUND` reports the remainder at the estimate.
- Approximation. Above 4096 rows the traces and the `tr3` pass are Hutchinson estimates (`..._SUMMARY_APPROXIMATION`
  1.0): `rho` itself then carries stochastic error (about 1e-3 with 256 probes on a 144-row lattice), unlike
  `GPUSpatialErrorGM` where only the standard error of `lambda` is approximate.
- `table`: `(k + 2) x 4` f32 rows of `[coefficient, se, z, two-sided normal p]`: the intercept, `X`, then `rho` last
  (spreg `W_y`). It is the layout `GPUSpatialLagImpacts` reads (`coefficientStride: 4`, `rhoIndex: predictorCount + 1`).
- `summary` (22, `GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_*`): `n`, `rho`, `sigma2 = u'u / n`, full log-likelihood
  `-n/2 ln s2 + ln|I - rho W| - n/2 (ln 2 pi + 1)` (spreg `logll`), AIC and Schwarz with spreg's `k = predictorCount + 2`
  (`rho` counted), `pr2` (squared correlation of `y` and `y - u`), `pr2_e` (of `y` and `(I - rho W)^-1 X b`), `u'u`,
  conditioning of the design (Belsley number, correlation condition number, largest VIF, precision warning above
  1e4; never changes `status`), the bound, the approximation, converged, at-boundary and series-limited flags, the
  truncation bound, variance-defined, the number of local maxima found on the grid (more than 1 is multimodal) and
  `ln|I - rho W|`.
- `status.code`: 0 ok, 1 singular design, 2 too few rows (`n <= k + 2`), 3 non-finite predictor, response or weight, no
  admissible interval, a residual sum of squares that reaches zero inside the search interval, or an interior
  `rho` that fails the first-order condition: with `u'u` and `u'e1` summed per row from the published
  residuals and `t = u'e1 / u'u`, the derivative `g = n t - tr(W A)` of the concentrated log-likelihood and its
  curvature `c = n (e1'e1 / u'u - 2 t^2) + tr((W A)^2)` must satisfy `|g| <= sqrt(0.25 c) + 4 c step`, that
  is, `rho` is within half a standard error plus four float32 steps of `rho` of a stationary point. Only
  ratios of the sums are formed, so a response multiplied by 1e-10, 1 or 1e10 gives the same status and the
  same `rho` to 6e-8 (the products of two sums of the earlier form overflowed at 1e10 and rejected a correct
  fit). The four steps are the float32 resolution of `rho` and of the sums: without them the float32 nearest
  to the maximum failed the test when the standard error of `rho` is below one step (a 64-row scene with a
  standard error of 4.6e-9; `rho` is now returned within two steps, but `s2` is 33 times the float64 value
  there, the standard errors 5.7 times and the log-likelihood 111 low, and no output flags it. The cause is
  the coefficients: the residuals are `e0 - rho e1` of two float32 OLS fits, each with its own rounded slope,
  and 91% of the published residual sum of squares lies in the span of `[1, x]` (the published slope is one
  float32 step from the float64 slope). The float32 floor of the residual arithmetic is 1.04 to 1.65 times,
  and the published residuals with their component in the span of `[1, x]` removed give 2.9 times; the kernel
  does not refine the coefficients as the two-stage kernel does). The check catches a search that ended away from a stationary point: a `rho`
  moved by 1e-4 on a high-fit scene, or by one standard error on an ordinary one, is status 3, and one moved
  by a fifth of a standard error is accepted. It shares the trace series with the search and cannot tell a
  local maximum from the global one. The outputs are then NaN, except `residuals`, which keep those of the
  rejected `rho`. `status.converged` (optional) is 1 for an interior maximum. A stationary maximum
  exactly at either endpoint is a boundary candidate and reports `converged = 0`.
  Stationary endpoint minima do not increment the local-maximum count; a flat profile keeps
  one lower-bound candidate. A zero endpoint derivative does not replace an isolated interior maximum.
  After a stationary lower-endpoint minimum, the first scan cell starts with its inward
  derivative sign, retaining a maximum inside that cell before the next grid point.

The upper endpoint's inward derivative is checked independently of the penultimate grid
  derivative, so a stationary upper maximum in the last cell remains a candidate without
  counting an already bracketed boundary twice.
  Two degenerate cases have a likelihood and are fitted as spreg `ML_Lag` fits them: a minimum of `u'u` that
  is zero but lies outside the search interval (`u'u = e1'e1 (rho - rho*)^2` is positive on the interval), and
  `e1'e1 = 0` (the lag column lies in the span of `[1, X]`, `u'u` is constant and the estimate is the maximum
  of the log-determinant). Both were status 3.
- `residuals`: `u = y - rho W y - X b`, as `e0 - rho e1` of the two OLS fits, whose residuals sum to zero (see
  the OLS numerics above). The mean of the squared residuals at the final `rho` gives `s2`.
  For varying `W y`, all rows use the same centered lag calculation. Measured on the test GPU against the float64 maximum of the concentrated likelihood
  (the estimator of spreg `ML_Lag`, whose own search is within 5e-7 of it in `rho`), for residuals of 0.002 to
  0.004 around a response mean of 10,000:
  - rows that share a row sum, exactly or up to the rounding of the weights (ring at 0.5, six neighbours at
    float32 1/6, binary weights with row sum 2, row sums of 1 and 1.0000000298, `rho` 0.2 to 0.8): `s2` within
    1.1e-5 and the log-likelihood within 0.004 (before the lag was formed about the mean: up to 4.2% and 12.5);
  - three rows without neighbours, `u'u` at 2e-10 of `e0'e0`: `s2` within 4e-5 and `rho` within 0.12 standard
    errors, a float32 step of `rho` (before: `s2` 2,800 times too large with status 0);
  - slopes two hundred times the residual deviation over the predictor range (`u'u` at 3e-9 of `e0'e0`): `s2`
    within 3.4e-4 and `rho` within 0.04 standard errors on 24 scenes, because the fitted values are rounded to
    1% to 3% of a residual (before: 14 of the 24 were status 3 and 3 were wrong with status 0);
  - row sums that differ by more than rounding with such a response (row sums 2 and 3): the lag differs
    between rows by the response mean and the lag column is rounded to a float32 step of that; `s2` within
    2.5e-4.

Exact log-determinant (opt-in): for ordinary lag ML, `logDeterminant: 'exact'` with
`eigenvalues` (the `rowCount` eigenvalues of `W` as float32, in a buffer disjoint from all outputs) replaces the truncated trace series of
`ln|I - rho W|`, `tr1` and `tr2` by `sum ln|1 - rho lambda|`, `sum lambda / (1 - rho lambda)` and its square sum, so there is
no truncation error and no series limit on the search interval (`..._SUMMARY_TRUNCATION_BOUND` is 0). It needs
`W` similar to a symmetric matrix (symmetric pattern and `w_ij d_i = w_ji d_j`, for example row-standardized
symmetric contiguity or distance weights) and at most `GPU_SPATIAL_EXACT_LOG_DETERMINANT_MAXIMUM_ROW_COUNT` (1024)
rows, because `getSpatialWeightsEigenvalues(offsets, neighbors, weights, rowCount)` runs a dense CPU symmetric
eigen-decomposition (about `4/3 n^3` operations). The third trace of the variance still comes from the probe pass.
Direct lag/error estimators retain one eigenvalue per response row. Panel-error forms infer
a repeated entity spectrum from N supplied values and apply repetition privately; there is
no public repetition-count option.
The helper rejects nonpositive or nonfinite recovered similarity scales and checks reversibility with tolerance
1e-4 relative to each transformed pair's magnitude, without an absolute floor; rescaling the weights does not
relax that check. Scale recovery uses exponent-scaled products when an intermediate overflows or underflows,
so a positive finite final similarity scale survives. The standalone `getSpatialExactLogDeterminant(rho, eigenvalues)` accepts negative factors
and returns the log absolute determinant (`rho = 2`, eigenvalues `[1, -1]` gives `log(3)`). A zero factor
returns `-Infinity`. Products throughout the unit-product cancellation region are checked using their multiplication residual before declaring
singularity; finite overflowing products contribute the operands' log magnitudes. True singularity takes priority
regardless of spectrum order. The CPU helper uses `log1p` for positive factors to preserve small terms; the GPU uses
a short series near zero for the same purpose.
The default stays `'series'`.

The panel-error wrapper additionally accepts a short entity spectrum: for `N` entities and
`T` periods, supply either `N` values (`N <= 1,024` on GPU) or the expanded `N*T` spectrum
(`N*T <= 1,024` on GPU). Its nested error-ML estimator repeats the short spectrum's determinant
and both trace derivatives `T` times; it does not read a short view as `N*T` values.
Equivalent representations preserve the panel-error outputs up to reduction rounding, subject
to the panel's independent total-row, design and device limits. CPU panel-error forms accept
both lengths without the GPU spectrum cap. See the [panel-error contract](./coverage-statistics.md#regimes-panels-and-sur)
for its full Gaussian likelihood; panel-lag retains its distinct likelihood normalization.

`getGPUSpatialLagMaximumLikelihoodModelResult(readback, {predictorCount?, predictorNames?})` builds the shared
`GPURegressionModelResult` (`kind: 'spatial-lag-maximum-likelihood'`, rows `CONSTANT`, predictors and `W_y` with role
`'spatial-lag'`, z statistics with float64 normal p-values, `standardErrorSigmaSquared: 'maximum-likelihood'`,
`logLikelihood`, and AIC and Schwarz equal to spreg's); `getGPUSpatialMaximumLikelihoodDiagnostics(summary)` returns
the flags and `pr2_e`. Its boolean diagnostics (`approximation`, `converged`, `atBoundary`,
`seriesLimited`, `varianceDefined`) return `null` when the summary flag is unavailable,
including code-3 loss suppression; measured 0 and 1 remain `false` and `true`.
At most 14 predictors: the `(predictorCount + 2)`-wide information matrix must fit the
16-wide Cholesky. Inputs must be finite.
For a capacity-sized table, pass the fitted `predictorCount` to decode only its active
`(predictorCount + 2) * 4` values and compute criteria with the fitted parameter count. Omitting
it requires an exact-length table.

Verified against spreg `ML_Lag(method='full', spat_impacts='full')` on a 12x12 lattice, a binary (raw) lattice, a
six-predictor lattice and two kNN scenes by a float64 mirror (`rho` to 4e-8, coefficients 3e-7, standard errors 1e-7,
z, log-likelihood, AIC, Schwarz, `pr2`, `pr2_e`, conditioning and `tr1`, `tr2`, `tr3`); the GPU spec checks the float32
output against the same pins at 2e-3 (`rho`, coefficients) and 1e-2 (standard errors), and chains the fit into
`GPUSpatialLagImpacts`. spreg's total multiplier `1 / (1 - rho)` is exact only for a row-standardized `W`; for other
weights the impacts here use the exact `1'W^q 1` sums.

```ts
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUSpatialLagMaximumLikelihood, GPUSpatialLagImpacts, type GPUSpatialLagMaximumLikelihoodProps, type GPUSpatialLagImpactsProps} from '@luma-contributors/analysis/gpu-spatial-analysis';

// Fragment inputs: allocate/upload these views in the caller's graph.
declare const graph: GPUCommandGraph;
declare const fit: GPUSpatialLagMaximumLikelihoodProps;
declare const downstream: GPUSpatialLagImpactsProps;
const {weights, predictors, response} = fit;
const {table, summary} = fit.output;
const status = fit.output.status.code;
const {rowCount} = downstream;
const {powerTraces, powerSums, infinityNorm} = downstream.traces;
const {impacts} = downstream.output;
graph.add(new GPUSpatialLagMaximumLikelihood({weights, predictors, response, predictorCount: 2,
  output: {table, summary, status: {code: status}, powerTraces, powerSums, infinityNorm}}));
graph.add(new GPUSpatialLagImpacts({coefficients: table, coefficientStride: 4, rhoIndex: 3, predictorCount: 2,
  rowCount, maximumPower: 128, traces: {powerTraces, powerSums, infinityNorm}, status, output: {impacts}}));
```

#### How it computes

{/* lens:GPUSpatialLagMaximumLikelihood */}

::lens{scenario=gpu-spatial-analysis.spatial-lag-maximum-likelihood rung=formula caption="Follow the concentrated likelihood, its log-determinant series and the rho search."}
::lens{scenario=gpu-spatial-analysis.spatial-lag-maximum-likelihood rung=kernel caption="Inspect the OLS, trace, rho search and variance kernels with retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.spatial-lag-maximum-likelihood rung=graph caption="Trace the two OLS fits, 128 trace powers and the rho search over 788 tracts."}

### `GPUSpatialErrorMaximumLikelihood`

Estimates a spatial error model by maximum likelihood, searching lambda on the concentrated likelihood with a trace-series log-determinant.

::lens{scenario=gpu-spatial-analysis.spatial-error-maximum-likelihood rung=result caption="Five rows: intercept, three predictors and lambda; lambda = 0.89 with standard error 0.018."}

Maximum-likelihood estimator of the spatial error model `y = X b + u`, `u = lambda W u + e` (spreg 1.9.1
`ML_Error`, `method='full'`), with the same truncated-series log-determinant, search and approximation rules as
`GPUSpatialLagMaximumLikelihood`. Props: `weights`, `predictors`, `response`, `predictorCount` (1 to 15),
`maximumPower`, `traceMode`, `traceProbeCount`, `traceParameters` (per-frame seed), optional `tileRowCount` and
`output: {table, summary, status: {code, converged?}, residuals?}`.
Nested trace layout resolution and the `2^24` hierarchical tile limit follow
`GPUSpatialLagMaximumLikelihood`, including captured weights metadata and allocation layout.

- Steps: a first tile pass fits OLS on the centered columns and gives `e0 = (y - mean) - (X - mean) b0`. With
  `Z = [1, X - mean, e0]`, three tile passes (fixed pairwise merge) accumulate once
  `G0 = Z'Z`, `G1 + G1' = Z'WZ + (Z'WZ)'` and `G2 = (WZ)'(WZ)`. Using `e0` for the response leaves the minimum over
  the coefficients unchanged (the filtered regression estimates `b - b0`) and keeps `ee(lambda)` accurate in float32
  when `R2` is close to 1: with the raw response `yy - 2 b'r + b'Ab` cancels to `1 - R2` of its terms. The Gram matrix of the filtered columns
  `Z - lambda W Z` (the filtered constant `1 - lambda sum_j w_ij` is column 0, as in spreg) is then exactly
  `G0 - lambda (G1 + G1') + lambda^2 G2`, so scanning `lambda` needs no pass over the rows. Each grid point solves
  one unit-scaled Cholesky system for `ee(lambda)`, and the derivative follows from the envelope theorem
  (`ee' = yy' - 2 b'r' + b'A'b`); the search is the one of the lag model on `-n/2 ln(ee / n) + ln|I - lambda W|`.
- Variance: `Var(b) = s2 (Xs'Xs)^-1` (centered form mapped back to the uncentered intercept) and
  `Var(lambda) = 1 / (tr2 + tr3 - 2 tr1^2 / n)`, the Schur complement of spreg's 2 x 2 information matrix of
  `(lambda, s2)`, with the traces of `W (I - lambda W)^-1` (`tr3` from the probe pass at the estimated `lambda`). As in
  spreg the coefficient standard errors condition on `lambda`.
- `table`: `(k + 2) x 4` rows, `lambda` last. `summary`: the 22 slots of the lag model; `sigma2 = e'e / n` of the
  filtered residuals, `pr2` of `y` and `X b`, `pr2_e` NaN, `u'u` of `u = y - X b`, and the conditioning of the final
  filtered regression (Belsley number of the raw `[c, X - lambda W X]`, correlation condition number and largest
  predictor VIF of the unit-scaled filtered system). AIC and Schwarz use spreg's `k = predictorCount + 1`: spreg's
  `ML_Error` does not count `lambda` (`k = x.shape[1]`), unlike `ML_Lag`, which counts `rho`.
- `status.code` as the lag model (the cap is 15 predictors, the OLS and Cholesky limit).

Pseudo R2 is the squared correlation of `y` and `X b`. It is evaluated as a product of ratios
of directly accumulated centered fitted-value moments, preserving weak predictions and response-scale invariance when the individual moments fit
float32 even if their products would overflow or underflow. Constant predictions have undefined
correlation and report NaN.
The shader removes residual means from its deviation moments as well, preserving correlation
when predictor translation makes its mean round to a neighboring binary32 value. The CPU lag
fit certifies exactly zero predictor/response covariance from original dyadic observations,
before rounding means such as `-2/7`; this preserves the constant-fit NaN convention.

The CPU lag fit also certifies constant original `W*y` before centering and preserves its
zero slope contribution through prediction. The GPU certifies equality of original row-product
sums with integer accumulators, including cancellation and equal nonzero lags, and preserves a
certified constant through fitting and prediction. Constant predictions retain NaN in summary
slot 6 even with varying row sums. The witness uses a separate buffer, retaining the prior
row-sum binding ceiling.

For the likelihood models, summary slot 18 is a truncation bound: with tail extrapolation it
is the plain-series remainder bound plus the absolute applied tail correction. The tail model
remains a heuristic; its correction cannot reduce this bound. With exact retained traces this
bounds the reported log-determinant's truncation error. With stochastic traces, sampling error
in the retained terms is separate and can exceed slot 18; `powerTraceStandardErrors` describe
that uncertainty and do not provide a deterministic bound on a realized sampling error.

With `logDeterminant: 'exact'`, positive residual variance and actual scalar-identity weights `W = c I`, the concentrated
likelihood is constant: filtering scales the residual variance by `(1 - c lambda)^2`, exactly
cancelling the log-determinant. Lambda and the innovation variance cannot be identified separately.
Both forms choose the lower search bound, report one boundary candidate and `converged = 0`,
retain the coefficients and their inference, and publish NaN for the lambda standard error,
z statistic and p-value with the variance-defined summary flag set to 0. This holds for every `maximumPower`,
including odd lengths; a positive truncated inverse-series variance is not evidence of identifiability.
The certificate checks the CSR itself, including chunks, rather than trusting rounded-equal
eigenvalues. Finite-series mode continues to fit its requested approximate likelihood.

For the lag model the same scalar certificate chooses one lower-bound candidate with
`converged = 0` and variance-defined flag 0. Beta and rho are jointly unidentified: all their
standard errors, z statistics and p-values are NaN, while point estimates remain available.

For a certified scalar profile the GPU solves the unfiltered Gram and computes coefficient
inference from the unfiltered residual variance. Only the reported innovation variance is
rescaled by `(1-c lambda)^2`. This avoids cancellation between nearly equal Gram matrices when
negative scalar weights make the lower-bound filter scale 0.01; coefficient p-values retain the
same unfiltered-model definition.

An exactly affine response has zero residual variance and no finite Gaussian ML maximum. The CPU
operation certifies this with exact integer rank arithmetic and returns code 3. GPU float32 OLS can
create a small positive residual on such data; this rejection path still requires GPU qualification.
Use `CPUSpatialErrorMaximumLikelihoodOperation` when exact perfect-fit classification is required.

`getGPUSpatialErrorMaximumLikelihoodModelResult(readback, {predictorCount?, predictorNames?})` builds the shared model result
(`kind: 'spatial-error-maximum-likelihood'`, `lambda` role `'spatial-error'`, `residualSumOfSquares = n sigma2` of the
filtered residuals, AIC and Schwarz equal to spreg's). Verified against `ML_Error(method='full')` on a
row-standardized lattice, an asymmetric kNN scene and a binary lattice (float64 mirror to 1e-7; GPU spec at 2e-3 and
1e-2, not yet measured).
Pass the fitted `predictorCount` for a table with spare capacity; padding after the active
`(predictorCount + 2) * 4` values is ignored. Omitting it requires an exact-length table.

```ts
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUSpatialErrorMaximumLikelihood, type GPUSpatialErrorMaximumLikelihoodProps} from '@luma-contributors/analysis/gpu-spatial-analysis';

// Fragment inputs: allocate/upload these views in the caller's graph.
declare const graph: GPUCommandGraph;
declare const inputs: GPUSpatialErrorMaximumLikelihoodProps;
const {weights, predictors, response} = inputs;
const {table, summary} = inputs.output;
const status = inputs.output.status.code;
graph.add(new GPUSpatialErrorMaximumLikelihood({weights, predictors, response, predictorCount: 2,
  output: {table, summary, status: {code: status}}}));
```

#### How it computes

{/* lens:GPUSpatialErrorMaximumLikelihood */}

::lens{scenario=gpu-spatial-analysis.spatial-error-maximum-likelihood rung=formula caption="Follow the filtered Gram matrix, the concentrated likelihood and the lambda search."}
::lens{scenario=gpu-spatial-analysis.spatial-error-maximum-likelihood rung=kernel caption="Inspect the Gram, trace, lambda search and variance kernels with retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.spatial-error-maximum-likelihood rung=graph caption="Trace the Gram passes, 128 trace powers and the lambda search over 788 tracts."}

### `GPUGeographicallyTemporallyWeightedRegression`

Geographically and temporally weighted regression (GTWR; Huang, Wu and Barry 2010) at a fixed or adaptive bandwidth. Every row is fitted by weighted least squares with kernel weights on the space-time distance `sqrt(ds^2 + tau dt^2)`, where `ds` is the planar distance, `dt` the difference of the times in ticks and `tau` the time scale (the `tau` of the Python package mgtwr, whose `GTWRKernel` and `GTWR.fit` are the oracle). The kernel is Gaussian `exp(-z^2 / 2)` or bisquare `(1 - z^2)^2` with `z = d / h`, chosen at compile time. `bandwidth` and `timeScale` are per-frame (`getGPUGTWRParameterValues`), so a caller can move through the `(bandwidth, timeScale)` plane without recompiling. With `adaptive: true`, bandwidth is an integer count of usable neighbours, including the focal observation and zero-distance ties, from 1 to 128. The local bandwidth is the kth space-time distance multiplied by `1.0000001` (mgtwr convention). A count greater than the usable population, a noninteger count, or a zero kth distance reports `SINGULAR`. Masked padding never enters the neighbour list.

Set `bandwidthSearch: {parameters, maximumIterations}` to select both bandwidth and time scale on the GPU. Pack the six per-frame values with `getGPUGTWRBandwidthSearchParameterValues({bandwidthBounds: [minimum, maximum], timeScaleBounds: [minimumScale, maximumScale], absoluteTolerance: 1e-3, relativeTolerance: 1e-3})`. The outer golden-section search minimizes the time-scale profile; each profile evaluation runs an inner golden-section bandwidth search. Each scalar search evaluates both endpoints and up to `maximumIterations` interior pairs (default 8, maximum 32). Adaptive bandwidth bounds must be integers in [1, 128]; time-scale bounds may include zero. Equal bounds fix an axis. This is a bounded reference algorithm: the profile is assumed unimodal, and nonsmooth adaptive objectives may have other minima. It does not promise an exhaustive global optimum.

`output.selectedParameters` optionally receives the winning `[bandwidth, timeScale]` pair. `output.status` accepts the canonical `iterationCount`, `converged` and `iterationLimitReached` uint32 ports. The count is outer time-scale comparisons; convergence requires the outer interval and the winning inner interval to meet `width <= max(absoluteTolerance, relativeTolerance * max(abs(lower), abs(upper)))` (at least one neighbour for the adaptive inner interval). Adaptive interior comparisons use distinct counts whenever possible; the final interval of at most three counts is also considered resolved once every count in it has been evaluated. Limit flags include the winning inner search. Candidates with singular rows or nonfinite AICc cannot win by dropping observations; exact score ties choose the smaller time scale, then the smaller bandwidth. At the limit, the best finite evaluated model is fitted and published. If no finite candidate exists, selected settings are NaN and included rows report `SINGULAR`; no convergence is claimed. Invalid raw GPU bounds follow the same path. Per-frame settings and bounds do not recompile; row counts, kernel, adaptive mode and budgets are compile-time. `estimatedCommandNodeCount` includes the nested search and construction rejects combinations above 32,768 commands. Finished searches gate their later fit dispatches on the GPU.

`times` are float32 ticks after the origin of a `temporalContext` that declares `float32-relative` encoding with an origin epoch; the unit is whatever the context declares, and `timeScale` converts one tick into spatial units squared per tick squared. The local system is solved in coordinates centered on the focal row with the diagonally equilibrated Cholesky factorization shared with the GWR family, so a rank-deficient or ill-conditioned local design reports `SINGULAR` for that row (NaN coefficients, fitted value and hat) instead of an unstable fit; `EXCLUDED` marks a masked row or one with a non-finite value. `summary` has the layout of `GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY` over the non-singular rows: residual sum of squares, trace of the hat matrix, AICc (the formula of `GPUGeographicallyWeightedRegression`, mgwr `aicc`: `-2 llf + 2 n (tr S + 1) / (n - tr S - 2)`, NaN when `n - tr S - 2 <= 0`, so a GWR and a GTWR fit are comparable and a degenerate bandwidth cannot win a profile search; mgtwr's own `aicc`, `-2 llf + 2 (tr S + 1) (n - 1) / (n - tr S - 1)`, is smaller by `2 n (k + 1) / (n - k - 2) - 2 (k + 1) (n - 1) / (n - k - 1)` with `k = tr S`: 2.15 at `n = 200`, `k = 60`), R squared, observation count, singular row count, sigma squared, log-likelihood and included row count.

Use `getGPUGeographicallyTemporallyWeightedRegressionModelResult({summary})` for either dense
or indexed readback, including historical summaries. The reader accepts numeric float32
elements, an `ArrayBuffer`, or a logical `Uint8Array` byte slice (including a Node `Buffer`),
preserving its byte offset and length even when unaligned:

```ts
import {getGPUGeographicallyTemporallyWeightedRegressionModelResult} from '@luma-contributors/analysis/gpu-spatial-analysis';

function readGTWRSummary(summaryBytes: Uint8Array | ArrayBuffer) {
  return getGPUGeographicallyTemporallyWeightedRegressionModelResult({summary: summaryBytes});
}
```

For an `ArrayBuffer` read, first construct `new Uint8Array(buffer, logicalByteOffset, logicalByteLength)`.
The reader validates whole float32 byte lengths and the minimum 12-element (48-byte) summary.
For `n=4`, `RSS=16` and `tr(S)=2`, the decoded row count is 4, status is 0 and Gaussian
log-likelihood is approximately `-8.448342855`; AICc is `null` because its correction denominator
is zero. Slot 5 (`HAS_VALID_CANDIDATE`) is unpublished
(NaN); the GWR reader instead interprets that slot as a bandwidth-selection certificate and must
not be used for GTWR. The GTWR reader's summary `status.code` is 0 when `rowCount > 0` usable
rows remain, otherwise 1. It does not certify every local fit or search convergence.
`singularRowCount` counts included rows whose local solve failed; `includedRowCount` includes
those rows, whereas `rowCount` counts only usable rows. Excluded rows are outside both counts;
count excluded rows separately from per-row `localStatus`, rather than treating singular rows
as excluded or interpreting allocation capacity as an observation count.

An unavailable AICc (`n - tr(S) - 2 <= 0`) becomes `null` without discarding a finite published
likelihood. True zero-RSS fits publish `+Infinity` log-likelihood and, when that correction
denominator is positive, `-Infinity` AICc. These infinities are retained by the reader. GTWR
computes residuals and fit statistics from the centered solve before fitted responses are rounded
back to the original response magnitude: a rounded fitted value equal to an observation does not
certify zero RSS. Original-space intercepts retain a compensated correction while predictor and
response origins cancel. This differs from GWR's floored finite candidate-selection objective.

The fit is float32 and planar with a native metric (`spatialContext`, default planar/native;
other coordinate spaces are rejected), with at most 11 predictors. The reference route,
`searchStructure: 'brute-force'`, visits all training rows for each focal fit
(`O(rows^2 x predictors^2)`) and accepts at most 32,768 rows (`GPU_GTWR_MAXIMUM_ROW_COUNT`).
The grid route accepts at most 16,777,216 rows (`GPU_GTWR_MAXIMUM_INDEXED_ROW_COUNT`), subject
to device bindings and scratch allocations. It requires compact support (bisquare, or Gaussian
with a positive finite float32 `gaussianCutoff`) and no `bandwidthSearch`. The default is grid
when these conditions hold, otherwise brute force. An explicit ineligible grid request is rejected.
`gaussianCutoff` is in bandwidth multiples: weights are zero when space-time distance exceeds
`gaussianCutoff * bandwidth` on either route. It deliberately changes Gaussian support; omitting
it retains untruncated Gaussian reference fits. Grid candidate traversal uses spatial bounds
and tests the full space-time distance. Bandwidth/time-scale search retains the reference route's
32,768-row bound and its separate 32,768-command budget. `gtwr-bandwidth-search.spec.ts` compares adaptive fits and search AICc with independent NumPy fits and exhaustive grids, including ties, padding, purely spatial bounds, singular/empty candidates and iteration limits. `gtwr-mgtwr.spec.ts` compares coefficients, fitted values, residuals, hat diagonals and summary statistics with mgtwr 2.0.5 output for four configurations (Gaussian and bisquare, time scale 0, 1.5, 2 and 8) and checks time-shift invariance, a larger time scale changing the fit, rank-deficient designs and masked rows. mgtwr declares no licence, so it is executed as an oracle (`test/oracles/gtwr-mgtwr-oracle.py`) and no code is copied.

The small independent `temporal-gwr-audit.oracle.py` fixture uses weighted QR to check every
fit, hat, residual, summary and search-status output with tied distances and masked rows.
Dense and indexed GTWR accumulate response differences from a local response anchor, then restore
the anchor after the centered solve; a large response intercept does not erase a small slope.
The exact `y=1000000+x` regression checks slope 1 and zero residuals directly.
It also checks changed units: scaling space by `a` and time ticks by `b` requires bandwidth
`a*h` and time scale `tau*a²/b²` for the same fit. Adaptive neighbor counts stay unchanged.
Node specs interpret emitted WGSL; the headless companion checks dense and indexed fits against
the same literals.
GTWR and multiscale GTWR capture compile-time props, context metadata, nested output/status
records and search budgets at construction. Mutating the caller's original props does not add
sweeps or change validated ports; the captured GPU views still read their current buffer contents.

{/* lens:GPUGeographicallyTemporallyWeightedRegression */}

### `GPUMultiscaleGeographicallyTemporallyWeightedRegression`

Multiscale GTWR by backfitting, following mgtwr `multi_bws`. A GTWR fit at the initial `(bandwidth, timeScale)` starts the components `beta_j x_j` (the intercept is component 0). Each iteration visits the components in order and smooths the partial residual `beta_j x_j + err` with a one-covariate local regression at that component's own bandwidth and time scale, `beta_ij = sum_l w_il x_lj r_l / sum_l w_il x_lj^2`, updating `err` before the next component (Gauss-Seidel order). `iterationCount` (1 to 32) is compile-time and every iteration runs; `output.scores` holds mgtwr's convergence score `sqrt(sum (XB_new - XB_old)^2 / n / sum (sum_j XB_new)^2)` for each iteration, so a caller sees where mgtwr would have stopped. Without `bandwidthSearch`, bandwidths are supplied by the caller (`getGPUMGTWRParameterValues`). With it, the initial GTWR fit and every component's partial-residual, no-intercept local regression select their bandwidth and time scale using the nested profile search above. Component AICc uses the partial response, residual sum of squares and smoother trace `sum_i x_ij^2 / sum_l w_il x_lj^2`. `adaptive` also applies to every component. The search bounds are shared by the initial and component searches. `output.selectedParameters` has the same `2 + 2 (predictorCount + 1)` layout as the input parameters, holding the initial pair and the last pair for each component. `output.status` sums outer comparisons over all searches, ANDs convergence and ORs limit flags. Backfitting itself still executes its fixed iteration count and publishes its score history. Command-budget validation includes every search, so larger combinations of backfit and search budgets are refused at construction. A row whose initial local fit is singular starts from zero components, and a component smoother whose weighted `x^2` sum is not positive leaves a NaN coefficient (status `SINGULAR`) and contributes nothing.

The spec compares the final coefficients, residuals and the whole score history with `mgtwr.function.multi_bws` run with fixed per-component bandwidths and time scales for two configurations that converge in five iterations.
The separate `temporal-gwr-audit.oracle.py` uses a simultaneous component block solve on a
balanced two-site fixture. It checks every output independently, including a nonzero first
convergence score, the zero second score, selected parameter pairs, singleton-search status
aggregation, masked rows and space/time/predictor rescaling.
Multiscale GTWR refuses `gaussianCutoff` to keep the exact Gaussian/bisquare kernel in initialization
and every component, and refuses `searchStructure` because component smoothing uses bounded
exhaustive traversal. Omit those props for multiscale fitting; use
`GPUGeographicallyTemporallyWeightedRegression` for truncated or indexed single-scale fits.

See [local variability and regression extensions](./statistical-expansion.md) for the expanded
statistics, fixed, finite-grid or golden-section-selected multiscale fits, generalized local models, pooled panels and regimes.

{/* lens:GPUMultiscaleGeographicallyTemporallyWeightedRegression */}

## MGWR bandwidth selection and limits

`GPUMultiscaleGeographicallyWeightedRegression` and its GPU/CPU operation forms accept
`bandwidthSelection: {criterion: 'AICc', searchMethod: 'golden-section', maximumSearchIterations: 30}`.
`bandwidthCandidates` then holds two positive finite distance bounds per component, in
`[interceptLower, interceptUpper, predictor0Lower, predictor0Upper, ...]` order, with
`bandwidthCandidateCount: 2`. Equal bounds select a fixed bandwidth; reversed or invalid bounds
invalidate the fit. The existing finite-grid mode remains the default, as does the
`'leave-one-out'` criterion. Candidate/bound values can change per frame.

Each backfit component selects its bandwidth from the scalar regression on its current partial
response, following [PySAL mgwr multi_bw](https://github.com/pysal/mgwr/blob/master/mgwr/search.py).
For AICc, include the focal observation and compute
`n log(max(RSS / n, 1e-30)) + n log(2π) + n (n + tr(H_j)) / (n - tr(H_j) - 2)`.
`H_j` is that scalar partial-response smoother, and `n` counts complete, unmasked rows.
The GPU accumulates power-of-two-scaled residual squares and recovers `RSS / n` using that
included count, so an overflowing intermediate RSS does not reject a representable variance.
A nonpositive denominator or any failed local fit rejects the candidate. Grid ties select the
first candidate; golden-section ties keep the lower interior point. Golden-section shares GWR's
`delta = 0.38197` and score tolerance `max(1e-6, 4e-6 max(abs(scores)))`, with at most 64 search
iterations. MGWR also evaluates the upper endpoint and keeps it if it is better than the
interior winner. A feasible endpoint keeps an initially infeasible interior bracket searchable.
If the search budget is exhausted, it returns the better of that endpoint and the last interior
winner. The
contributor retains OLS initialization (zero with full inference), its relative-contribution
backfit stopping rule, and unrounded distance bandwidths; it does not reproduce PySAL's initial
single-bandwidth GWR fit, SOC stopping rule, or two-decimal bandwidth rounding.

For the **full MGWR model**, the trace is `tr(S) = sum_j tr(R_j)`, where `R_j` is the coupled
component response operator propagated through the backfit, conditional on the selected bandwidth
history (Li and Fotheringham, 2020, [*Computational improvements to multi-scale geographically
weighted regression*](https://doi.org/10.1080/13658816.2020.1720692)). This is the trace in
`inference.output.summary[0]`, and each `tr(R_j)` is in `effectiveParameters`. Summing independent
scalar `H_j` traces would give a different full-model effective parameter count. Standard errors
remain conditional on bandwidth history; bandwidth confidence intervals and full selection
uncertainty are not provided.

Planar bisquare and explicitly truncated Gaussian fits automatically use the shared GWR grid from
1,024 rows. The square dimensions are `ceil(sqrt(rows / 8))`, capped at 1,024 per axis.
`indexGridSize: false` forces the dense path. Untruncated Gaussian and great-circle
fits retain the dense fallback, including an explicit grid with an untruncated Gaussian kernel.
Limits are 65,536 dense rows and 1,048,576 indexed rows, subject to bindings; 24 centered
predictors (`p`), hence 25 design columns including the intercept; 128 backfit sweeps;
32 finite-grid candidates per component; and the shared 32,768 command-node budget.
The convergence reduction has at most 256 partial groups; this is neither a predictor nor a
candidate capacity. The
selection score buffer needs `rows * candidateWidth * 16` bytes in one storage binding
(`candidateWidth = 3` for golden-section: two interior points and the upper endpoint), checked against the device limit. Full inference retains
both `n <= 128` and `n^3 * (p + 1) * maximumIterations <= 268435456`,
where `n = response.length` and `p = predictorCount`. The CPU twin uses float64 internally. Both backends honor
`maximumIterations` (default 32); finite unfinished fits publish coefficients and predictions with
status 3, while inference remains unavailable until convergence. The CPU accepts budgets above
the GPU's 128-sweep cap. Near-tied scores can produce different bandwidths within float32 error.
Float32 coefficient or prediction overflow produces numerical-failure code 4 and NaN outputs.
The optional `output.status.componentSearchIterationLimitReached` has one flag per design
column; it reports any exhausted golden-section search independently of backfit convergence,
accumulates across sweeps, and resets each execution. Both operation constructors snapshot their
sweep budgets. The website labels finite limit estimates separately from exclusions and failures.

The independent [NumPy fixture generator](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/test/gpu-spatial-analysis/spatial-regression/multiscale-bandwidth-selection-fixtures.py)
records the initialization and stopping differences and checks scalar selection plus coupled traces.

MGWR construction uses exact logical lengths, unlike the capacity-prefix OLS/GWR outputs.
For `n = response.length` and `p = predictorCount`, inputs are packed row-major predictors
(`n * p`), `n` positions/responses and an optional `n`-row mask. `parameters` has capacity
**at least `p + 2`**: tolerance followed by `p + 1` component bandwidths. When supplied,
`bandwidthCandidates` has exactly `(p + 1) * bandwidthCandidateCount` float32 entries;
golden-section uses two bounds per component and `bandwidthCandidateCount: 2`.

| MGWR output port | Exact logical element count | Element type |
| --- | ---: | --- |
| `output.coefficients` | `n * (p + 1)` | float32, row-major, intercept first |
| `output.fitted` | `n` | float32 |
| `output.status.codes` | `n` | uint32, primary result rows |
| `output.status.iterationCount` | `1` | uint32, completed sweeps |
| `output.selectedBandwidths` (optional) | `p + 1` | float32 |
| `output.conditionalStandardErrors` (optional) | `n * (p + 1)` | float32; requires exactly one float32 `noiseVariance` |
| `output.status.componentSearchIterationLimitReached` (optional) | `p + 1` | uint32 |
| `inference.output.standardErrors` | `n * (p + 1)` | float32 |
| `inference.output.effectiveParameters` | `p + 1` | float32 |
| `inference.output.summary` | `4` | float32 |
| `inference.output.status` | `1` | uint32 |

The full-inference ports apply only when `inference` is requested, under both guards above.
`conditionalStandardErrors` is the separate fixed-noise conditional route; requesting it alone
does not request the coupled full-inference response operators.

Planar finite-support GGWR automatically builds a grid above 65,536 rows when `indexGridSize` is
omitted. `indexGridSize: false` forces dense traversal, with a cap of 65,536 rows. An explicit
grid with fixed, untruncated Gaussian support accepts at most 1,048,576 rows, subject to bindings.
This is an input/allocation bound: each focal fit and each IRLS iteration still visits every
training row, so neighborhood traversal is quadratic even with that explicit grid. Bisquare and
Gaussian with a cutoff use finite-support cell traversal and accept up to 16,777,216 indexed
rows, subject to bindings. GGWR local inference uses its selected fit route's row and
device-binding limits.

These GGWR fit capacities do not enlarge bandwidth-selection work limits. AICc grid and golden
searches using dense or untruncated Gaussian traversal are capped at **4,096 total input slots**
(`GENERALIZED_GWR_MAXIMUM_SEARCH_ROWS`), even with an explicit grid. Finite-support indexed
selection retains its 16,777,216-row bound subject to bindings and command budget. See
[AICc bandwidth-selection limits](./adaptive-generalized-gwr.md#aicc-golden-section-bandwidth-selection).
