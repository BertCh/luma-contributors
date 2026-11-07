# Local variability, point patterns and spatial regression extensions

These contributors build GPU command-graph nodes. They allocate graph-owned scratch, write caller-owned
outputs, and never read results back or fit a hidden CPU model. Import local variability and regression
from `@luma-contributors/graph/gpu-spatial-analysis`; import the pair statistics from
`@luma-contributors/graph/gpu-dataframe`.

## Local variability

`GPULocalSpatialHeteroscedasticity` computes LOSH using a CSR `weights` graph, scalar `values`, an
optional `mask`, and caller-owned `output.statistic` and `output.status` views. `residualPower` is 2
by default (squared local residuals), or 1 (absolute local residuals). It computes the weighted local
mean, the residual power for each observation, then a weighted local mean of neighboring residual
powers divided by their global mean. The result is descriptive; it does not supply a chi-square or
permutation p-value.

Positive weights are normalized over available neighbors. Masked and nonfinite observations are
excluded. Islands have no residual and are excluded from the global residual mean. In the final
neighbor reduction, undefined residuals are removed and the remaining weights are normalized again.
Thus masked/island behavior is explicit and extends the complete-data `esda.LOSH` reference contract.

`GPUMultivariateLocalGeary` accepts row-major values and `variableCount` (1–32). It selects complete
rows jointly, standardizes each variable using its population variance, and averages the
row-standardized local Geary statistic over variables. Complete-case islands participate in global
standardization but have undefined local statistics. A constant variable makes the result undefined.
Randomization must preserve the multivariate row pairing; this contributor does not reuse a
univariate permutation test or manufacture p-values.

Both require per-row `output.status`:

| Code | Meaning |
| --- | --- |
| 0 | Defined statistic |
| 1 | Excluded observation or LOSH island |
| 2 | Undefined statistic, including constant fields or invalid accumulated arithmetic |
| 3 | Incomplete upstream weights |

`addLocalVariabilityRecipe` composes LOSH and local Moran on the same weights, values and mask. The
returned outputs keep LOSH and its status separate from Moran's statistic, z-score, raw and adjusted
p-values, quadrants, and inference status. Multiple-testing correction applies only to the Moran
family. The recipe preserves upstream overflow and the invalid count from each statistical stage.
It does not create a joint significance score or inferential LOSH classification.

## Marked and cross-type point patterns

`GPUMarkedPointStatistics` supplies cross-type K/L, caller-intensity-weighted inhomogeneous K,
annular pair correlation and product-mark correlation in a rectangular window. Inputs are planar
`positions`, optional `types` and `typePair`, optional `marks` and `intensities`, a mask, and six
float32 parameters:

```text
[minimumX, minimumY, maximumX, maximumY, maximumRadius, minimumIntensity]
```

With `radiusCount` bins, the upper radii are `maximumRadius * (bin + 1) / radiusCount`.
`output.statistics` contains four scalar columns per bin, with indices in
`GPU_MARKED_POINT_STATISTICS_COLUMNS`: K, L, pair correlation and mark correlation. `output.pairCounts`
contains cumulative directed pair counts; self-pairs are excluded. `output.status` is 0 when the
K estimate is defined, 1 for an invalid window or empty category, and 2 for nonfinite accumulation.
An empty annulus can produce a valid K with undefined correlations; inspect those float outputs too.

The default `edgeCorrection: 'translation'` uses the rectangle's overlap with its translated copy.
`'none'` explicitly disables correction. Pairs with zero overlap are excluded. Without supplied
intensity the normalization uses `area / (nFocus * nNeighbor)`, including same-type `n²` normalization.
With intensity, the estimator is the sum of `1 / (intensity_i * intensity_j * overlapArea)`;
nonpositive, nonfinite and below-threshold intensities are excluded. Intensity estimation remains an
explicit upstream step.

Pair correlation is an annular-area estimate, not a kernel-smoothed `pcf` estimator. Mark correlation
uses weighted mean products divided by the product of category mean marks; zero mean products and
empty shells are undefined. Supplying intensity weights both the pair and mark sums. These choices
must be matched before comparing another package's output.

## Cross and space-time variograms

`GPUCrossVariogram` computes the signed mean of `(x_i - x_j) * (y_i - y_j) / 2` for each distance bin.
`secondValues` defaults to `values`, yielding the Matheron semivariogram. Negative cross-semivariances
are retained. `GPUSpaceTimeVariogram` additionally requires `times` and `temporalLagCount`; bins are
spatial-major. Parameters are `[maximumDistance, maximumTimeLag]` (the second entry is unused without
times). Both limits must be positive and finite.

Bins are left-closed and right-open, except the final upper edge is included. Each unordered pair is
counted once. Output views are `semivariances`, `pairCounts`, and per-bin `status` (0 defined, 1 empty
or invalid lag parameters, 2 nonfinite arithmetic). Complete-case exclusion includes positions,
both values, optional times, and the mask. The implementation bounds row count to 16,384 and bin
count to 128. It uses deterministic dense pair traversal and row partials; it does not claim the
scaling of the indexed univariate `GPUVariogram`.

## Generalized geographically weighted regression

`GPUGeneralizedGeographicallyWeightedRegression` supports `family: 'poisson'` and `'logistic'`, with
an intercept plus 1–11 predictors, finite planar positions, an optional mask and fixed-distance or adaptive-neighbor
bandwidths. Poisson response values are nonnegative integer counts, and logistic response values are
0 or 1. An optional `offset` is additive on the linear predictor; use log exposure for Poisson.
Parameters are `[bandwidth, relativeCoefficientTolerance]`. The kernel is `'gaussian'` (default) or
`'bisquare'`. With `bandwidthMode: 'adaptive'`, the first parameter is an integer neighbor count bounded by `maximumNeighborCount`; see [Adaptive generalized GWR](./adaptive-generalized-gwr.md). All-zero Poisson or single-class logistic neighborhoods report status 5.

Each local model uses Newton/IRLS with Cholesky and a bounded likelihood step-halving search.
`maximumIterations` defaults to 32 and is bounded to 128. Outputs are row-major `coefficients`
(intercept first), `fitted`, and `status: {codes, iterationCount}`. Both status columns have one row per observation; `iterationCount` records the number of local IRLS steps. Only converged fits publish coefficients.
There is no implicit regularization or separation correction. Poisson predictors outside [-30,30]
are reported as numerical failures; they are not silently clamped to another likelihood. Logistic
predictors are bounded to [-80,80] for evaluation. All-zero count neighborhoods, singular designs,
and separated binary outcomes can fail or exhaust the budget.

## Multiscale geographically weighted regression

`GPUMultiscaleGeographicallyWeightedRegression` implements Gaussian-response additive backfitting,
with a separate distance bandwidth for every variable, including the intercept. Parameters are:

```text
[relativeContributionTolerance, interceptBandwidth, predictorBandwidth0, ...]
```

The algorithm starts at zero coefficient surfaces, updates one variable across every location, then
moves to the next variable. A sweep converges when the largest relative change in any fitted component
is within tolerance. The row selection is complete-case. Singular component fits invalidate the
whole coupled backfit. Non-converged coefficient surfaces are not published as completed results.
`output.status.codes` has one row per observation; `output.status.iterationCount` is one scalar
containing the number of completed backfit sweeps.

For automatic bounded selection, supply column-major `bandwidthCandidates` and
`bandwidthCandidateCount` (1–32). Every component update evaluates the candidate bandwidths by
leave-one-out partial-response squared error, selects the smallest total error, then fits the
component using that bandwidth. Ties select the first candidate. An invalid candidate is rejected;
if none is valid, fitting fails. `output.selectedBandwidths` exposes the final choices. This is a
finite-grid CV contract, distinct from mgwr's continuous AICc or adaptive-neighbor bandwidth search.

Optional `output.conditionalStandardErrors` requires a scalar `noiseVariance`. It computes
`sqrt(noiseVariance * sum(linearSmootherWeights²))` for each local univariate coefficient, holding
other component surfaces and the selected bandwidths fixed. This is conditional submodel uncertainty,
not full MGWR covariance: uncertainty from backfitting, bandwidth selection and noise estimation is
not included. The variance must be supplied explicitly; negative/nonfinite variance produces NaNs.

For full backfit uncertainty, supply `inference: {varianceConvention, output}` with
`standardErrors[n*p]`, `effectiveParameters[p]`, `summary[4]`, and `status[1]`, where p includes
the intercept. Every coefficient update propagates its full response Jacobian through all other
component surfaces, using the actual selected bandwidth history. The combined smoother S yields
summary `[trace(S), trace(SᵀS), sigma2, residualDegreesOfFreedom]`. `residual-df` (default) estimates
sigma² as SSE/(n-trace(S)); `smoother-df` uses SSE/(n-2trace(S)+trace(SᵀS)). Standard errors use
the complete backfit Jacobian and estimated variance; effective parameters are each component's
smoother trace. Inference status is 0 valid, 1 unsuccessful coupled fit, or 2 invalid degrees of
freedom/variance. Failed inference produces NaNs.

This matches mgwr's bandwidth-conditioned uncertainty, including coupling among coefficient
surfaces; it does not add uncertainty from reselecting bandwidths under resampling. Full inference
requires n ≤ 128 and `n³*p*maximumIterations<=268435456`, and allocates two n²p Jacobian buffers.
Distance bandwidths are fixed; adaptive-neighbor and continuous AICc optimization are separate
contracts. Direct mgwr 2.2.1 fixtures cover fixed bandwidths, both variance conventions, and an
explicit finite-grid LOO policy composed from actual mgwr local fits. The latter is not advertised
as parity with Sel_BW's different default AICc selection policy.

Both iterative regressions use `GPU_ITERATIVE_REGRESSION_STATUS`:

| Code | Meaning |
| --- | --- |
| 0 | Converged |
| 1 | Excluded observation |
| 2 | Singular or insufficient local design |
| 3 | Iteration limit reached |
| 4 | Numerical failure or invalid numerical parameters |

## Pooled spatial panels and regimes

`GPUSpatialPanelRegression` defaults to pooled spatial-lag 2SLS with a
common intercept, slopes and spatial autoregressive coefficient across periods and entities.
Inputs are balanced and period-major, with a single time-invariant entity weights graph. It expands
`I_periods ⊗ W` on GPU and delegates estimation to `GPUSpatialTwoStageLeastSquares`; output layout,
instrument-order options and solver diagnostics are those of that contributor. There are no
cross-period edges. The default `effects: 'pooled'` retains this behavior. `effects: 'unit'` adds a GPU within transformation; see [Unit fixed effects](./panel-fixed-effects.md) for absorbed intercepts, degrees of freedom and diagnostic limits. Random effects and dynamic panels are unsupported.
Its classical pooled covariance is not clustered by entity or robust to serial correlation.

`GPUSpatialRegimeRegression` fits a separate intercept and every slope for each caller-supplied
spatial regime. `regimes` contains integer IDs in `[0, regimeCount)`; other IDs and masked/nonfinite
rows are excluded. It publishes regime-major `coefficients`, row-level `fitted`, regime `counts`,
and regime `status` (0 fitted, 1 insufficient observations, 2 singular/nonfinite design). Predictors
are centered within each regime before solving. This is fully varying regime OLS, with no lag/error
coefficient, cross-regime equality restriction, covariance output or Chow test.

## Evidence and performance limits

The committed generator
[`geospatial-expansion-statistics.py`](../../../../modules/graph/test/oracles/geospatial-expansion-statistics.py)
records package versions, synthetic input provenance, float32 rounding and reference types. It uses
direct esda LOSH/Geary, direct constrained-bandwidth mgwr backfitting, direct spreg panel and regime
fits, independent local statsmodels GLMs, and explicit statsmodels/NumPy compositions for grid CV
and conditional uncertainty. SciPy distance/NumPy bin compositions are labeled as compositions,
not direct spatstat or scikit-gstat parity.

Focused browser tests compare these references and exercise exact bin endpoints, signed
cross-covariance, masks, upstream overflow, constant variables, empty regimes, singularity, forced
non-convergence, candidate selection, conditional uncertainty and recipe composition. Node tests
parse the generated WGSL and check the default eight-storage-buffer limit, including all optional
inputs. These are correctness checks, not evidence of a speedup.

The new pair traversal is `O(rows² * bins)`. Local GLMs are quadratic per iteration; multiscale
backfitting is quadratic per sweep and variable, multiplied by candidate count when selecting
bandwidths. Row limits bound the declared shape but do not promise a suitable interactive latency
at the maximum shape. Benchmark compilation, GPU execution, scratch memory and repeated updates
on the intended device and workload before choosing these paths over established CPU libraries.

Adaptive neighbor bandwidths and degenerate-response status are detailed in [Adaptive generalized GWR](./adaptive-generalized-gwr.md). Conditional fixed-residual LOSH inference has a separate [residual permutation contract](./losh-residual-permutation.md).
