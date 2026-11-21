# Local variability, point patterns and spatial regression extensions

These contributors build GPU command-graph nodes. They allocate graph-owned scratch, write caller-owned
outputs, and never read results back or fit a hidden CPU model. Import local variability and regression
from `@luma-contributors/analysis/gpu-spatial-analysis`; import the pair statistics from
`@luma-contributors/analysis/dataframe-analysis`.

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
family. The recipe preserves both upstream loss flags and the invalid count from each statistical
stage. Candidate-only loss suppresses both LOSH (including normalized residuals) and Moran, with
INCOMPLETE LOSH status; both recover on resubmission when the loss flags clear.
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
K estimate is defined, 1 for an invalid window or empty category, 2 for nonfinite accumulation,
and 3 when the cumulative directed pair count exceeds `0xffffffff`. In that case the count
saturates at `0xffffffff` and all four statistics are NaN. An exact count of `0xffffffff` is valid.
An empty annulus can produce a valid K with undefined correlations; inspect those float outputs too.

The default `edgeCorrection: 'translation'` uses the rectangle's overlap with its translated copy.
`'none'` explicitly disables correction. Pairs with zero overlap are excluded. Without supplied
intensity the normalization is `area / (nFocus * nNeighbor)` for two different types (spatstat `Kcross`)
and `area / (n * (n - 1))` when focus and neighbor are the same set (no `types`, or an equal `typePair`),
as `GPURipley` and spatstat `Kest`; one point of a type has no same-type K (NaN, status 1).
With intensity, the estimator is the sum of `1 / (intensity_i * intensity_j * overlapArea)`;
nonpositive, nonfinite and below-threshold intensities are excluded. Intensity estimation remains an
explicit upstream step.

Pair correlation is an annular-area estimate, not a kernel-smoothed `pcf` estimator. Mark correlation
uses weighted mean products divided by the product of category mean marks; zero mean products and
empty shells are undefined. Supplying intensity weights both the pair and mark sums. These choices
must be matched before comparing another package's output.

The default `searchStructure: 'grid'` uses a `[256, 256]` maximum cell lattice; optional
`gridSize: [columns, rows]` changes those dimensions. Cells are at least `maximumRadius` wide,
and each focus visits its 3 x 3 neighboring cells. The partial table permits
`rows * radiusCount * 3 <= 2^24` float32 entries, further constrained by the device's storage
binding size. Device limits for the partials and cell offsets are checked before scratch allocation.

`searchStructure: 'brute-force'` explicitly selects the dense reference, limited to 16,384 rows;
it cannot be combined with `gridSize`. Counts and radius edges are unchanged. Indexed sums use
cell order and a fixed tree of at most 256 entries per invocation. Category means reduce marks
above `2^100` separately at scale `2^-24`, divide by the category count, then restore their scale.
This keeps a representable mean finite even when the raw mark sum would overflow. Smaller marks
remain unscaled, preserving their contribution when large opposite marks cancel. Means and sums
can differ from the reference's running means and serial sums within the
float32 comparison tolerance (`1e-6 + 1e-4 * |reference|` in the traversal regression).

Large or clustered neighborhoods still require quadratic pair work: a radius reaching the window's
extent collapses the lattice to one cell. A finite radius whose square overflows float32 also uses
one cell, retaining the finite-distance predicate and the reference estimator instead of losing all
pairs. Invalid or nonpositive radii remain empty. This fallback preserves correctness; it does not
bound the pair-walk work for broad neighborhoods.

## Cross and space-time variograms

`GPUCrossVariogram` is deprecated in favor of `GPUSpaceTimeVariogram`. It remains a forwarding class through the shared indexed empirical variogram front end, preserving its original props, `[maximumDistance, maximumTimeLag]` parameter layout and `lagCount * temporalLagCount` output bins. Its first bin includes zero lag; the newer class gives each axis an explicit zero-lag class. `secondValues` still defaults to `values`, giving the Matheron semivariogram, and negative cross-semivariances are retained. The adapter computes complete-case bounds on the GPU, including both variables, coordinates, mask and optional time. It fits the lattice to those bounds each encoding; omitted `gridSize` now uses `[256, 256]`. The default and explicit lattices now share the same limits: `rows * bins <= 2^24`, at most
128 bins, exact float32 row IDs and the device's storage binding size. Omission of `gridSize` no
longer imposes the former dense reference's 16,384-row cap.

`GPUCrossVariogram` retains floating-point legacy arithmetic on every frame: it forms pair differences from the original values, accumulates per-row partials through the shared cell index, and combines them with bounded tree reductions. Centering can erase a small raw difference beside an unrelated value outlier, and even a finite global fixed-point scale can round a valid pair term to zero. The legacy path therefore does not center or quantize its values. Neighboring cell ranges are merged in ascending input-row order before summation, retaining the former default's signed cancellation independently of lattice dimensions. The fixed tree over per-row partials can round differently from the former serial finishing sum. Status is 0 valid, 1 empty/invalid parameters, 2 nonfinite accumulated arithmetic, or 3 when a bin's pair count exceeds `0xffffffff`. Overflow saturates the count at `0xffffffff` and publishes NaN semivariance; an exact count of `0xffffffff` remains valid. The compatibility path retains the original `rows * bins` scratch bound, with no row-sized serial finishing loop.

The legacy adapter accepts positive finite radii whose square or cell margin overflows float32. It also accepts finite bounds whose width or height overflows on subtraction. These cases use a single-cell lattice, with cell lookup avoiding the overflowing subtraction and the original finite-distance filter applied to every candidate pair. Such frames can visit all pairs. Normal frames retain the fitted lattice and neighboring-cell traversal.

`GPUSpaceTimeVariogram` retains its shared fixed-point pair histogram, accumulating positive and negative cross terms separately and distinguishing integer overflow as status 3.

`GPUSpaceTimeVariogram` is built on the indexed pair histogram of `GPUVariogram`, so its cost follows
the pairs in neighboring lattice cells instead of `rows^2`. Class 0 of each axis is the exact zero lag
(identical float32 coordinates; identical float32 relative times). Positive lags fall in
`1 + min(floor(lag / maximum * count), count - 1)`, left-closed and right-open with the final upper
edge included. Output has `(lagCount + 1) * (temporalLagCount + 1)` bins, spatial-major:
`spatialClass * (temporalLagCount + 1) + temporalClass`. Bin 0 is coincident points at the same time;
the first row (spatial class 0) is the purely temporal variogram, the first column the purely spatial
one. Each unordered pair is counted once. Parameters are written with
`getGPUSpaceTimeVariogramParameterValues({bounds, maximumDistance, maximumTimeLag})`; rows outside
`bounds` are excluded, and `gridSize` only affects speed. At most 409 bins are supported, and rows are
limited as for `GPUVariogram`.

`times` must be relative to a float64 origin chosen on the CPU, in one unit: float32 spaces raw Unix
epoch seconds 128 s apart and epoch milliseconds about 2 minutes apart, which collapses short lags.
Build them with `getGPUSpaceTimeRelativeTimes(times, origin, requiredResolution)` from
`gpu-spatial-analysis`, which reports the remaining float32 resolution and can throw when the span is
too long for the lags you need. Times closer than that resolution fall into the zero-lag class.

Output views are `semivariances`, `pairCounts`, and per-bin `status` (0 defined, 1 empty or invalid
lag parameters, 2 nonfinite arithmetic, 3 integer accumulation overflow). Complete-case exclusion
includes positions, both values, times, and the mask. Terms are accumulated as exact integers (the
positive and negative parts apart), with a fixed-point unit set by the typical cross term, so results
are reproducible. Parity with gstat `variogramST` is not yet established: the spec compares with an
in-repo double-precision brute-force reference, and a gstat fixture is pending.

## Generalized geographically weighted regression

`GPUGeneralizedGeographicallyWeightedRegression` supports `family: 'poisson'` and `'logistic'`, with
an intercept plus 1–24 predictors, finite planar or great-circle positions, an optional mask and fixed-distance or adaptive-neighbor
bandwidths. Poisson response values are nonnegative integer counts, and logistic response values are
0 or 1. An optional `offset` is additive on the linear predictor; use log exposure for Poisson.
Parameters are `[bandwidth, relativeCoefficientTolerance]`. The kernel is `'gaussian'` (default) or
`'bisquare'`. With `bandwidthMode: 'adaptive'`, the first parameter is an integer neighbor count bounded by `maximumNeighborCount`; see [Adaptive generalized GWR](./adaptive-generalized-gwr.md). All-zero Poisson or single-class logistic neighborhoods report status 5.

Each local model uses Newton/IRLS with Cholesky and a bounded likelihood step-halving search.
`maximumIterations` defaults to 32 and is bounded to 128. Outputs are row-major `coefficients`
(intercept first), `fitted`, and `status: {codes, iterationCount}`. Both status columns have one row per observation; `iterationCount` records the number of local IRLS steps. Converged fits and finite iteration-limit iterates publish coefficients.
There is no implicit regularization or separation correction. Poisson linear predictors outside [-30,30]
are reported as numerical failures; they are not silently clamped to another likelihood. Logistic
training linear predictors are bounded to [-80,80] for likelihood evaluation. Prediction-only
logistic queries accept any finite link, including offset 100 with a float32 probability of 1
and zero influence. Large Poisson exposure/count unit mismatches use a weighted exposure-aware
intercept start. Prediction-only design conditioning uses training rows. `iterationCount` counts
accepted coefficient updates, so a singular first solve reports zero completed updates.
All-zero count neighborhoods, singular designs,
and separated outcomes have explicit diagnostics; status 6 names a witnessed complete/quasi-separation
direction. AICc grid/golden-section selection and its named search status are described in
[Adaptive generalized GWR](./adaptive-generalized-gwr.md).

## Multiscale geographically weighted regression

`GPUMultiscaleGeographicallyWeightedRegression` implements Gaussian-response additive backfitting,
with a separate distance bandwidth for every variable, including the intercept. Parameters are:

```text
[relativeContributionTolerance, interceptBandwidth, predictorBandwidth0, ...]
```

The algorithm starts every coefficient surface at the global ordinary least squares coefficients
(at zero when full `inference` is requested, because its propagation starts from zero surfaces),
updates one variable across every location, then moves to the next variable. This starting point is
this kernel's own: a package that starts from a single-bandwidth GWR fit reaches the same fixed point
when it is unique, with different iteration counts and a different result if the iteration limit
stops the backfitting first. The starting fit reduces complete-case counts, centered sums and
normal-equation moments in bounded row tiles, merging sums in a fixed pairwise order. Responses
are scaled by a data-derived power of two before subtraction in the mean and moment passes;
the fit restores the original response units when publishing coefficients. Centered mean
contributions are also divided by the included-row count before accumulation. These steps avoid
opposite-sign response subtraction overflow and unnormalized deviation-sum overflow. A valid
observation anchors centering even when the leading or trailing rows are excluded. Only the small
coefficient system is solved in one invocation; singular or insufficient global designs start at zero.
A sweep converges when the largest relative change in any fitted component
is within tolerance. The row selection is complete-case. Singular component fits invalidate the
whole coupled backfit. `maximumIterations` bounds completed sweeps on both GPU and CPU (default
32; GPU maximum 128). An exhausted finite fit publishes its last coefficients and fitted values
with code 3. Excluded, singular, and numerically failed fits publish NaNs. Conditional and full
inference require convergence and remain NaN at the iteration limit; full inference status is 1.
`output.status.codes` has one row per observation; `output.status.iterationCount` is one scalar
containing the number of completed backfit sweeps. The CPU twin computes in float64 and accepts
budgets above the GPU maximum, while honoring every explicit budget and the shared default.
Both operation constructors snapshot their validated sweep budgets and selection settings;
mutating the caller's props later does not change subsequent executions. Code 0 and code 3 carry
finite float32 coefficients and fitted values on every included row. A coefficient or prediction
that would overflow the public float32 output invalidates the coupled fit with code 4, suppresses
all included coefficients/predictions to NaN, and leaves inference unavailable (status 1).

Planar bisquare and truncated Gaussian kernels can prune candidates through a grid, with at most
1,048,576 rows on that route. Dense traversal permits at most 65,536 rows, including untruncated
Gaussian fits even when an explicit `indexGridSize` is supplied. Great-circle fits also use dense
traversal. An indexing option cannot lift the bound when the kernel still visits every row.

For automatic bounded selection, supply column-major `bandwidthCandidates` and
`bandwidthCandidateCount` (1–32). The default is finite-grid leave-one-out partial-response
squared error; ties select the first candidate. `bandwidthSelection.criterion` also accepts
`'AICc'`, scored from each scalar partial-response smoother as in mgwr `multi_bw`.
`bandwidthSelection.searchMethod` accepts `'finite-grid'` (default) or `'golden-section'`.
Golden-section requires `bandwidthCandidateCount: 2`: each column gives positive lower and upper
distance bounds. Its `maximumSearchIterations` is 1–64 and all selection work shares the command
node budget. Invalid candidates lose to valid ones; no valid candidate invalidates the fit.
Selection runs during every component update. `output.selectedBandwidths` exposes final choices.
Omitting candidates preserves fixed per-component bandwidths, as in the displayed lens.
Optional `output.status.componentSearchIterationLimitReached` has one uint32 flag per design
column, intercept first. A flag is one if any golden-section search for that component exhausted
its search budget during this submission. Flags accumulate across backfit sweeps and reset on
every execution. Fixed bandwidths and finite-grid selection produce zero flags. Both operation
forms accept graph value names for this port; the CPU function always returns the corresponding
`componentSearchIterationLimitReached` array. A finite selected bandwidth remains available at
the search limit. Backfit convergence and bandwidth-history-conditioned inference remain
independently valid: backfit code 0 and inference status 0 may coexist with a search-limit flag 1.

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
freedom/variance, including a finite CPU variance that overflows its public float32 representation.
Invalid inference retains its diagnostic summary (which may contain Infinity on overflow) and
suppresses dependent standard errors/effective parameters to NaN.

This matches mgwr's bandwidth-conditioned uncertainty, including coupling among coefficient
surfaces; it does not add uncertainty from reselecting bandwidths under resampling. Full inference
requires n ≤ 128 and `n³*p*maximumIterations<=268435456`, and allocates two n²p Jacobian buffers.
Bandwidths are distances; adaptive-neighbor MGWR selection is not supported. Golden-section is
a bounded continuous search, not a global-optimum certificate for a non-unimodal score. Direct mgwr 2.2.1 fixtures cover fixed bandwidths, both variance conventions, and an
explicit finite-grid LOO policy composed from actual mgwr local fits. The latter is not advertised
as parity with Sel_BW's different default AICc selection policy.
The separate `multiscale-gwr-audit.oracle.py` solves the simultaneous component normal equations
and their response operators as one block linear system. Its tied two-site fixture checks every
CPU and GPU operation output against independent literals, including both variance conventions,
conditional errors, excluded rows, exactly two completed sweeps and coordinate/predictor rescaling.
The stopping convention checked here is maximum relative component change; it differs from
PySAL mgwr's aggregate score-of-change stopping criterion.

MGWR exports `GPU_MULTISCALE_GEOGRAPHICALLY_WEIGHTED_REGRESSION_STATUS` and
`decodeGPUMultiscaleGeographicallyWeightedRegressionStatus`. Its codes retain the existing numbers
shared with the first five `GPU_ITERATIVE_REGRESSION_STATUS` values used by generalized GWR:

| Code | Meaning |
| --- | --- |
| 0 | Converged |
| 1 | Excluded observation |
| 2 | Singular or insufficient local design |
| 3 | Iteration limit reached |
| 4 | Numerical failure or invalid numerical parameters |

For MGWR, the code on each included row describes the entire coupled backfit; one failed component
invalidates all included rows. An excluded row always receives code 1. These codes do not report
whether a component bandwidth search reached its own search tolerance. GWR `localStatus`, MGWR
backfit convergence, and bandwidth-search convergence describe separate facts. In particular,
GWR code 1 means singular, and must be decoded with the GWR decoder rather than the MGWR decoder.
The website readout uses the MGWR decoder to distinguish converged rows, excluded rows, displayed
finite iteration-limit estimates, and failed rows.

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
and response view identities, regime/predictor counts and the output map are snapshotted at
construction; changing the caller's props later does not change dispatch bounds or bindings.
Declared fields are captured by name, including inherited/non-enumerable mask getters, which
are read once at construction. A prototype-supplied mask selects the same complete cases as an own field.
Input buffer contents remain per-frame data. Predictors
are centered within each regime before solving. Cross-products and predictions carry the correction
for rounded float32 means; fitted values use centered predictors rather than the rounded raw
intercept. Means are accumulated as differences from the first complete row of each regime,
preserving variance in nearly constant projected-coordinate predictors. The raw intercept is
formed from stored means before adding its separate mean correction; corrections are divided
by the row count before multiplying deviation sums, preserving finite large-scale moments.
Prediction fixes the combined centered contribution and mean correction with an integer bit
cast before adding the response mean, preserving representable fitted levels on Metal.
This is fully varying regime OLS, with no lag/error
coefficient, cross-regime equality restriction, covariance output or Chow test. The per-regime
moments are accumulated in parallel and in a fixed order (a stable regime sort, then one workgroup
per regime), so results are reproducible; rows are limited to 2^24 and the device's binding size.

## Evidence and performance limits

The committed generator
[`geospatial-expansion-statistics.py`](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/test/oracles/geospatial-expansion-statistics.py)
records package versions, synthetic input provenance, float32 rounding and reference types. It uses
direct esda LOSH/Geary, direct constrained-bandwidth mgwr backfitting, direct spreg panel and regime
fits, independent local statsmodels GLMs, and explicit statsmodels/NumPy compositions for grid CV
and conditional uncertainty. SciPy distance/NumPy bin compositions are labeled as compositions,
not direct spatstat or scikit-gstat parity.

To predict values at query locations or generate conditional realizations, see
[Interpolation and simulation](./interpolation-expansion.md) for operation-specific examples of
local kriging, radial basis functions, natural-neighbor interpolation and conditional Gaussian
simulation. Local kriging can bind per-frame covariance parameters.
Default simulations rebuild preparation on each encoding.
With `reusePreparation: true`, write the changed geometry and call `invalidatePreparation()` before
the next encoding to refresh preparation using the covariance captured at construction. In both
cases, changing the covariance model or its parameters requires a new simulation and graph.

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
