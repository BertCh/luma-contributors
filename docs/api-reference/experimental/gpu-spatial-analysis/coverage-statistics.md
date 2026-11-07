# Bounded statistical estimation and validation

These contributors extend the existing spatial statistics and regression graph with explicit
numerical, missingness and convergence contracts. All computation is recorded in the caller's
GPU graph. They neither submit work nor read buffers back, and have no CPU fallback. Inputs and
outputs are caller-owned packed views; the graph owns transient workspace. Each generated kernel
stays within the default eight-storage-buffer limit.

The supported modes below are **bounded subsets** of the ST01–ST11 backlog. They do not establish
complete PySAL, mgwr, blockCV or spint parity. Exact capacity counts, converged numerical estimates,
Monte Carlo inference, exact leave-one-out model selection and bounded continuous profile optimization are different guarantees.

| Backlog | API | Delivered contract | Remaining scope |
| --- | --- | --- | --- |
| ST01 | `GPUInstrumentalVariablesCovariance`; `GPUSpatialTwoStageLeastSquares.robust` | Full HC0, HC1 or spatial-HAC IV covariance; integrated robust SE, z and normal p-value table | HC2/HC3, clustered covariance, automatic HAC PSD certification |
| ST02 | `GPURestrictedRegimeRegression` | Any common/varying intercept and slopes; full restricted covariance; joint nested-model Chow F/degrees and coefficient/joint Wald chi-square tails | spatial/heteroskedastic restriction tests |
| ST03 | `GPUSpatialPanelErrorMaximumLikelihood` | Balanced unit fixed-effects spatial-error Gaussian ML and pooled LM error and robust LM lag/error tests | Random/time effects, Hausman tests, clustered/serial inference |
| ST04 | `GPUSeeminglyUnrelatedRegression` | One-step feasible SUR and Kelejian–Prucha spatial-error SUR, joint error covariance and full system coefficient covariance | iterated SUR and small-sample inference |
| ST05 | `GPUProbitRegression` | Binary probit, expected-information standard errors, average marginal effects, bounded likelihood optimization; Pinkse, Kelejian–Prucha and Pinkse–Slade residual tests | Latent spatial probit |
| ST06 | `GPULocalCollinearity`; `GPUGeneralizedGeographicallyWeightedRegression` | Weighted VIF, intercept-inclusive design condition/variance decomposition; exact LOO bandwidth search; Poisson/logistic local SE and influence | Multiscale backfitting uncertainty and additional selection criteria |
| ST07 | `GPUMultivariateLocalGearyPermutationTest` | Joint-vector conditional randomization, replay, equal-tail or esda minimum-tail p-values and optional Bonferroni adjustment | Exact upstream classification-label parity and additional correction families |
| ST08 | `GPURankMarkov` | Full-rank/geographic-rank transitions, common complete panels and absolute-rank mobility | More than 64 rank states, fractional ties and incomplete-panel interpolation |
| ST09 | `GPUFuzzyContiguity`, `GPUIslandAttachment` | Planar filled-polygon separation and nearest original-island attachment, with added-edge provenance | Geographic coordinates, topology repair and GEOS robust-predicate parity |
| ST10 | `GPUSpatialCrossValidation`, `GPUSpatialCrossValidatedOLS`, `GPUNestedInterpolationValidation` | Block/group/environmental folds, buffers, bounded NNDM/kNNDM, training-only OLS and nested covariance/RBF selection with held-out metrics | [Explicit design, optimizer and capacity contracts](./spatial-validation-advanced.md) |
| ST11 | `GPUSpatialInteraction` | Four constraint modes with grid or continuous bounded profile calibration, IPF, deviance and marginal residuals | multiple covariates, coefficient uncertainty |

## IV covariance and robust spatial 2SLS

`GPUInstrumentalVariablesCovariance` takes structural design `Z`, instrument design `H` and
structural residuals from the same fitted IV model. Both designs omit the implicit intercept.
The coefficient count is 1–15; the instrument count is between the coefficient count and 16;
there are at most 65,536 observations. All estimation rows must be finite. There is no silent
complete-case change after a model has already been fitted.

The covariance uses the projected-design influence matrix, not an OLS sandwich applied to
endogenous regressors. Columns are centered and norm-scaled before both Cholesky solves. The
intercept influence is recovered with its cross terms, so the output includes the full
`(coefficientCount + 1)²` covariance in intercept-first row-major order.

- `HC0` uses squared residual scores without a finite-sample multiplier.
- `HC1` multiplies HC0 by `n / (n - p)`, where `p` includes the intercept.
- `HAC` takes a separate spatial kernel, without a finite-sample multiplier. It must be symmetric,
  nonnegative, positive semidefinite and have unit diagonal. Bounds, finite weights, symmetry
  and diagonal are checked on GPU. Positive semidefiniteness is a caller precondition.

Covariance status is `0` success, `1` singular projection, `2` insufficient observations,
`3` nonfinite inputs/results and `4` invalid HAC kernel. Failed covariance entries are NaN.

For spatial 2SLS, set `robust: {method, kernel?, output: {covariance, status}}`. Existing point
estimates and classical summary diagnostics remain unchanged; the coefficient table's SE, z and
p-value columns use the robust covariance. The robust status is separate from the estimator's
status. A robust covariance does not change the classical Anselin–Kelejian diagnostic's assumptions.

## Regimes, panels and SUR

`GPURestrictedRegimeRegression.common` includes the intercept followed by each predictor.
A true entry estimates one shared coefficient; a false entry estimates one coefficient per regime.
Output order is source-column-major, with regime order inside each varying column.
The unrestricted model is limited to 16 coefficients; its complete cases are exactly those of the
restricted model. Covariance uses the restricted model's pooled `SSE / (n - p)` estimate.
The Chow output contains `[F, numeratorDegrees, denominatorDegrees]`. Optional `stability` contains `[Wald chi-square, degrees, upper-tail p]` for each source coefficient and then jointly. It tests equality of the varying regime coefficients, matching `spreg.Chow`; common columns are `[NaN, 0, NaN]`. This is a different hypothesis from the nested-model F test.
Empty or singular regimes are explicit, and a singular unrestricted fit does not erase a valid
restricted coefficient estimate. The joint test assumes independent homoskedastic normal errors.

`GPUSpatialPanelErrorMaximumLikelihood` takes the existing spatial-error ML arguments plus
`periodCount`. Data are balanced and period-major; the supplied weights describe only the entity
axis. Unit means are removed on GPU, and `I_T ⊗ W` supplies the repeated spatial determinant.
The objective and variance divisor use `N*T`, matching the profiled fixed-effects Gaussian ML
convention. This is not REML or a short-panel bias correction. It inherits the existing bounded
trace-series search, boundary and approximation diagnostics. The absorbed intercept table row and
information criteria are NaN. Predictors constant through time are singular after demeaning.
Optional `diagnostics.tests` returns statistic/p-value pairs for pooled-panel LM error, robust LM lag and robust LM error. These fit pooled OLS on the original panel and use its residual variance, spatial lags and projected spatial fitted values, matching spreg 1.8.3. Diagnostic status is 0 valid, 1 failed/incomplete pooled fit, or 2 degenerate/nonfinite test information. The robust tests require positive spatial fitted-value information beyond the weight trace. They are specification tests, not clustered uncertainty for the FE estimator.

`GPUSeeminglyUnrelatedRegression` requires 2–4 equations with the same predictor count but
potentially different predictor values. Data are equation-major, then row-major; observation IDs
must be aligned explicitly. Any nonfinite value in any equation excludes that observation from all
equations. The total number of coefficients, including each intercept, is at most 16.
Equation OLS residuals estimate a **joint** cross-equation covariance with divisor `n`, followed
by one full feasible-GLS solve. The output covariance is the conditional GLS information inverse
using that first-stage error covariance. Singular design and singular error covariance have
separate status codes. This is one-step FGLS, not an iterated likelihood fit.

Optional `spatial` selects the `spreg.SURerrorGM` convention: equation OLS residuals estimate Kelejian–Prucha spatial-error parameters; filtering transforms responses, designs and original OLS residuals before the joint GLS solve. The intercept and full covariance are transformed back to original units, and reported residuals are structural residuals. This selected mode requires complete aligned rows and nonnegative row-standardized zero-diagonal weights without islands; it rejects a mask or caller-supplied `initialResiduals`. Each equation has its own parameter and GM status; status 4 denotes inadmissible weights. Nonspatial `initialResiduals` allows an explicit first-stage residual covariance, with the same common complete-case mask.

## Probit and local collinearity

`GPUProbitRegression` uses centered/scaled predictors, Fisher scoring and monotone likelihood
backtracking. It accepts 1–11 predictors and at most 65,536 rows. Complete cases are selected
jointly; included responses must be exactly 0 or 1. The expected-information SEs are model based.
Marginal effects average the normal density over estimation rows, then multiply by each slope;
they are not effects evaluated at the means.

Its summary is `[completeRows, logLikelihood, deviance, iterations]`. Status `0` means converged,
`1` insufficient data, `2` singular information, `3` iteration/line-search exhaustion and `4`
nonbinary data or separated/unsupported tails. Nonconverged finite iterates remain available with
status 3, but their standard errors are NaN. The f32 normal CDF is floored at `1e-7`; all-zero/all-one
outcomes and linear predictors outside ±12 are rejected explicitly. Optional `spatialDiagnostics` uses the fitted probit probability and generalized/standardized residuals to report `[Pinkse chi-square, p, Kelejian–Prucha z, two-sided p, Pinkse–Slade chi-square, p]`. Weights must have a sorted zero-diagonal CSR; the induced common complete-case graph is used. Diagnostic status is 0 valid, 1 failed/no-case model, 2 invalid weights, 3 zero test information. This does not estimate latent spatial probit.

`GPULocalCollinearity` uses supplied local estimation weights, including diagonal weights if the
fit uses the focal observation. Every predictor uses the same complete neighbors. VIFs are the
diagonal of the inverse weighted correlation matrix; the condition index is the square root of
its maximum/minimum eigenvalue ratio. The intercept is excluded from this correlation calculation. Paired optional `designConditionNumbers` and `varianceDecomposition` reproduce mgwr's intercept-inclusive column-normalized design convention: multiply each design row by normalized spatial weight, not its square root, and report the variance proportion associated with the smallest singular value. A bounded Jacobi eigensolver returns status 3 on exhaustion; status 1 means insufficient neighbors and 2 singular/nonfinite correlation or design. The eigenvalue threshold is `1e-7` of the largest eigenvalue, an explicit f32 conditioning limit.

`GPUGeneralizedGeographicallyWeightedRegression.inference` returns local scale-one Poisson/logistic standard errors and diagonal influence. The covariance is `A^-1 X' V W² X A^-1`, with `A = X' V W X`; it matches mgwr GLM local uncertainty and does not assume that spatial weights are frequency weights. Failed fits retain NaN inference.

`bandwidthSelection` supplies 2–32 positive candidate bandwidths and outputs their exact leave-one-out mean deviance, the selected bandwidth and status (0 selected, 1 no valid candidate). Adaptive candidates must be integer neighbor counts within `maximumNeighborCount`. Each inner fit excludes its own focus, including from adaptive neighbor selection. Every eligible training row must converge for a candidate to be valid; ties choose the first candidate. This explicit criterion differs from mgwr `Sel_BW` AICc or approximate CV criteria. The pinned oracle executes mgwr local GLM solves with the self weight removed to verify the exact candidate scores.

For nested spatial evaluation, supply one outer fold's **training mask** and set `predictExcluded: true`. Bandwidth selection, centering/solves and inference use only training rows, while finite held-out predictor/position queries receive predictions without reading their responses. The direct test changes held-out counts from zero to 100,000 and verifies unchanged candidate scores, selected bandwidth, coefficients and predictions. Apply this graph separately to each fold; score held-out responses afterward. Finite query offsets remain caller inputs.

## Joint randomization and rank dynamics

`GPUMultivariateLocalGearyPermutationTest` preserves the existing `GPUMultivariateLocalGeary`
statistic API and adds a separate inference producer. It supports at most 256 observations,
16 variables and 4096 permutations (`maximumPermutations`, fixed when compiling the graph). A common complete-case mask applies to all variables;
population standardization is performed once. Active local weights are row-standardized.
The focal vector is fixed while distinct **whole neighbor vectors** are sampled without replacement.
Independent column permutations would destroy the cross-variable dependence and are not used.

The conservative equal-tail Monte Carlo p-value is
`min(1, 2 * min((lower + 1)/(P + 1), (upper + 1)/(P + 1)))`, counting ties in both tails.
`tail: 'esda-min-tail'` instead computes `(min(upper, P-upper)+1)/(P+1)`, where `upper` counts simulated values greater than or equal to the observed value. This reproduces esda's undoubled, tie-asymmetric tail convention; equal-tail remains the default.
Optional Bonferroni adjustment uses all included rows, including islands, as the family.
Classes are 0 nonsignificant, 1 significant similarity, 2 significant dissimilarity and
`0xffffffff` unknown. Constant variables invalidate the joint statistic; islands are unknown.
Optional simulations expose every row/replicate draw for replay. Seed words and the shared
Philox conditional-permutation purpose determine the stream.

`GPURankMarkov` follows the pinned giddy 2.3.6 rank conventions with zero-based descending ranks.
Equal values receive distinct ranks; default `tieOrder: 'giddy'` assigns the higher descending rank to the later original row, matching ascending ordinal ranks reversed by giddy. `tieOrder: 'row-id'` preserves earlier-row-first ties explicitly. Any missing value removes the row
from **all** periods, avoiding changing rank populations over time. Full rank follows a location
through rank states; geographic rank follows the occupant of a fixed rank through location states.
The original `n × n` output capacity is retained when rows are excluded. Empty transition rows
stay zero. Optional existing Markov diagnostics retain their ergodicity assumptions; optional
mobility reports mean absolute rank movement per adjacent period, NaN below two complete rows.

## Weights and spatial evaluation

`GPUFuzzyContiguity` uses the existing feature→polygon→ring→vertex layout, with a packed
`float32x2` position view. Polygon distance combines segment distances with even-odd filled-part
containment, preserving holes and multipart features. Rings close implicitly. `tolerance` is the
maximum separation, equivalent geometrically to intersecting exact round buffers of half that
radius; it does not build approximated buffer polygons or use grid snapping.
Features are limited to 512 and total vertices to 65,536. A dense feature-pair distance workspace
is explicit. Feature status distinguishes valid, empty and malformed/nonfinite input. Simple valid
rings are required; self intersections and near-degenerate f32 predicates are not repaired.
Edges are binary and ordered, with `added = 1` only for strictly positive separation.
Output overflow reports the exact required neighbor count.

`GPUIslandAttachment` selects nearest finite supplied representatives for **original empty CSR
rows**, with lowest-ID distance ties. Selection is simultaneous, so islands may attach to each
other. Reciprocal insertion is the default; directed insertion is explicit. It preserves existing
weights and uses weight 1 only for added edges. Empty means no stored slots, not a zero-weight row.
Nearest comparisons use a shared distance scale to avoid squared-distance overflow or underflow
for finite float32 representatives. No representative is inferred, and no implicit normalization
is performed. Truncated CSR remains
ordered and bounded, with required counts, attachment IDs and per-edge provenance.

`GPUSpatialCrossValidation` hashes explicit block IDs and a uint32 seed to fold IDs. Assignment
is deterministic but not balanced; empty folds remain visible. Every finite unmasked position
appears in exactly one test fold. A training point is excluded when its planar distance from any
held-out point is less than or equal to `bufferDistance`. Distance zero also excludes duplicate
positions in different blocks. Invalid coordinates enter neither training nor testing.
The baseline performs exact all-pairs checks and is not an indexed large-domain performance claim.

`GPUSpatialCrossValidatedOLS` composes those masks with a separate OLS fit for each fold. OLS
centering and all estimation statistics use training rows only. It predicts held-out rows using
that fold's coefficients and reports `[scoredRows, missingTestRows, MSE, MAE, bias, R²]`.
Bias is prediction minus observation. Metric status bits report no scored rows, missing coverage
and undefined R² for constant test outcomes. Changing held-out responses cannot change that fold's
fit. Generalized GWR supplies a training-only `bandwidthSelection` graph as described above; [`GPUNestedInterpolationValidation`](./spatial-validation-advanced.md) supplies reusable inner-fold covariance/variogram and RBF bandwidth selection, outer refitting and held-out metrics using those masks.

## Calibrated spatial interaction

`GPUSpatialInteraction` fits a one-covariate Poisson mean proportional to
`exp(coefficient * covariate)` over an explicit finite coefficient grid or continuous bounded search. Supported constraints
are unconstrained total mass, origin totals, destination totals and both margins. The latter uses
bounded iterative proportional fitting. Default `calibration: 'grid'` retains a finite-grid optimum. `calibration: 'bounded'` takes exactly two ascending coefficient endpoints and uses golden-section profile optimization, with `maximumCalibrationIterations` (1–64, default 32) and `coefficientTolerance` (default `1e-5`). The profiled deviance is optimized over the explicit interval; optimum-boundary and f32 flat-objective behavior remain visible in the interval. Candidates with exponent spans above 80 are rejected explicitly.

Rows are explicit OD count cells. `pairState` distinguishes structural zero (0), observed (1) and
missing (2). Missing cells are excluded from both support and margins and predict NaN; structural
zeros must have observed flow zero and predict zero. Observed zero counts remain in the support.
Duplicate supplied OD rows are independent cells; aggregate duplicates explicitly when appropriate.
Zero observed margins remain zero. At most 4096 cells, 128 origins, 128 destinations and 128
coefficient candidates are supported; candidate predictions occupy explicit transient storage.

The selected summary is `[coefficient, deviance, maximumRelativeMarginError, iterations]`.
Each candidate exposes deviance, error, iterations and status. Unconverged candidates are never
selected as converged fits. If none succeeds, predictions are NaN with a failure status.
Optional origin/destination residuals are observed minus predicted margins. A successful fit is
conditional on the specified support and search domain; coefficient uncertainty is unavailable. Optional bounded-only `output.calibration` is `[lower, upper, width, converged]`. Status 4 retains the best finite fit but reports an exhausted coefficient search; inner IPF failure remains separate. The final two candidates are exposed in bounded mode. Direct spint GLM fixtures cover all four constraint modes with the same explicit support.

## Verification and evidence limits

The generator `modules/analysis/test/oracles/statistics-coverage.py` pins NumPy 2.2.6 and SciPy 1.15.3.
It constructs independent dense IV sandwiches, nested regime designs, block SUR GLS, probit
likelihood/information, profiled panel-error likelihood and interaction IPF references. These are
**composed numerical references**. Separately, `statistics-upstream.py` actually executes pinned spreg 1.8.3, mgwr 2.2.1, esda 2.7.0, giddy 2.3.6, libpysal 4.13.0 and spint 1.0.7 (with scikit-learn 1.7.2). Its checked-in fixture records full covariance, diagnostics, local inference, exact LOO scores, tied ranks, weights and calibrated flows. `statistics-upstream.spec.ts` compares GPU results directly against those outputs; it does not relabel the composed NumPy oracle as library parity.

`coverage-statistics.spec.ts` tests robust inference, singular/nonfinite/kernel failures, spatial
fold invariants, both rank conventions and island capacity/provenance. `coverage-estimators.spec.ts`
adds numerical estimator comparisons, forced probit exhaustion, joint-vector permutation replay,
fuzzy distance/provenance/capacity and an actual held-out-response perturbation test. The node suite
parses and audits generated WGSL and checks storage binding budgets. Native GPU execution results
and final repository gates are recorded in the implementation report; constructors and shader
parsing alone do not validate numerical execution.

`statistics-performance-benchmark.spec.ts` records preparation/upload, graph compilation, first and repeated resident execution, readback and graph transient bytes for increasing SUR and training-only bandwidth-selection workloads. These are device-specific bounded observations, not a universal throughput guarantee or driver-memory peak. Broader stochastic calibration/coverage studies, full topology adversaries and browser-family scaling remain additional evidence.

## Executable examples

These small deterministic synthetic examples run the actual GPU producers and capture inputs, estimates and status outputs. They illustrate the bounded contracts above, not geographic observations or performance evidence.

### `GPUInstrumentalVariablesCovariance`

{/* lens:GPUInstrumentalVariablesCovariance */}

::lens{scenario=gpu-spatial-analysis.instrumental-variables-covariance rung=result caption="Inspect the estimates and explicit status outputs on synthetic inputs."}

::lens{scenario=gpu-spatial-analysis.instrumental-variables-covariance rung=graph caption="Inspect the executed GPU command graph."}

### `GPURestrictedRegimeRegression`

{/* lens:GPURestrictedRegimeRegression */}

::lens{scenario=gpu-spatial-analysis.restricted-regime-regression rung=result caption="Inspect the estimates and explicit status outputs on synthetic inputs."}

::lens{scenario=gpu-spatial-analysis.restricted-regime-regression rung=graph caption="Inspect the executed GPU command graph."}

### `GPUSpatialPanelErrorMaximumLikelihood`

{/* lens:GPUSpatialPanelErrorMaximumLikelihood */}

::lens{scenario=gpu-spatial-analysis.spatial-panel-error-maximum-likelihood rung=result caption="Inspect the estimates and explicit status outputs on synthetic inputs."}

::lens{scenario=gpu-spatial-analysis.spatial-panel-error-maximum-likelihood rung=graph caption="Inspect the executed GPU command graph."}

### `GPUSeeminglyUnrelatedRegression`

{/* lens:GPUSeeminglyUnrelatedRegression */}

::lens{scenario=gpu-spatial-analysis.seemingly-unrelated-regression rung=result caption="Inspect the estimates and explicit status outputs on synthetic inputs."}

::lens{scenario=gpu-spatial-analysis.seemingly-unrelated-regression rung=graph caption="Inspect the executed GPU command graph."}

### `GPUProbitRegression`

{/* lens:GPUProbitRegression */}

::lens{scenario=gpu-spatial-analysis.probit-regression rung=result caption="Inspect the estimates and explicit status outputs on synthetic inputs."}

::lens{scenario=gpu-spatial-analysis.probit-regression rung=graph caption="Inspect the executed GPU command graph."}

### `GPULocalCollinearity`

{/* lens:GPULocalCollinearity */}

::lens{scenario=gpu-spatial-analysis.local-collinearity rung=result caption="Inspect the estimates and explicit status outputs on synthetic inputs."}

::lens{scenario=gpu-spatial-analysis.local-collinearity rung=graph caption="Inspect the executed GPU command graph."}

### `GPUMultivariateLocalGearyPermutationTest`

{/* lens:GPUMultivariateLocalGearyPermutationTest */}

::lens{scenario=gpu-spatial-analysis.multivariate-local-geary-permutation-test rung=result caption="Inspect the estimates and explicit status outputs on synthetic inputs."}

::lens{scenario=gpu-spatial-analysis.multivariate-local-geary-permutation-test rung=graph caption="Inspect the executed GPU command graph."}

### `GPURankMarkov`

{/* lens:GPURankMarkov */}

::lens{scenario=gpu-spatial-analysis.rank-markov rung=result caption="Inspect the estimates and explicit status outputs on synthetic inputs."}

::lens{scenario=gpu-spatial-analysis.rank-markov rung=graph caption="Inspect the executed GPU command graph."}

### `GPUFuzzyContiguity`

{/* lens:GPUFuzzyContiguity */}

::lens{scenario=gpu-spatial-analysis.fuzzy-contiguity rung=result caption="Inspect the estimates and explicit status outputs on synthetic inputs."}

::lens{scenario=gpu-spatial-analysis.fuzzy-contiguity rung=graph caption="Inspect the executed GPU command graph."}

### `GPUIslandAttachment`

{/* lens:GPUIslandAttachment */}

::lens{scenario=gpu-spatial-analysis.island-attachment rung=result caption="Inspect the estimates and explicit status outputs on synthetic inputs."}

::lens{scenario=gpu-spatial-analysis.island-attachment rung=graph caption="Inspect the executed GPU command graph."}

### `GPUSpatialCrossValidation`

{/* lens:GPUSpatialCrossValidation */}

::lens{scenario=gpu-spatial-analysis.spatial-cross-validation rung=result caption="Inspect the estimates and explicit status outputs on synthetic inputs."}

::lens{scenario=gpu-spatial-analysis.spatial-cross-validation rung=graph caption="Inspect the executed GPU command graph."}

### `GPUSpatialCrossValidatedOLS`

{/* lens:GPUSpatialCrossValidatedOLS */}

::lens{scenario=gpu-spatial-analysis.spatial-cross-validated-ols rung=result caption="Inspect the estimates and explicit status outputs on synthetic inputs."}

::lens{scenario=gpu-spatial-analysis.spatial-cross-validated-ols rung=graph caption="Inspect the executed GPU command graph."}

### `GPUSpatialInteraction`

{/* lens:GPUSpatialInteraction */}

::lens{scenario=gpu-spatial-analysis.spatial-interaction rung=result caption="Inspect the estimates and explicit status outputs on synthetic inputs."}

::lens{scenario=gpu-spatial-analysis.spatial-interaction rung=graph caption="Inspect the executed GPU command graph."}

## Recorded reference and scaling run

The separate `statistics-blockcv.R` generator actually executes blockCV 4.0.1 `cv_spatial` with
explicit user polygons/predefined folds and `cv_buffer` with projected distances. GPU masks are
compared exactly against both. The seed hash is our explicit deterministic assignment rule;
this does not claim identity with blockCV's random balancing optimizer.

The checked-in `statistics-performance-baseline.json` records the Apple WebGPU run. Resident
median encode/submit/queue-completion times were:

| Workload | Rows | Resident median | Graph transient bytes |
| --- | ---: | ---: | ---: |
| Two-equation SUR | 64 | 1.8 ms | 0 |
| Two-equation SUR | 1,024 | 4.8 ms | 0 |
| Two-equation SUR | 16,384 | 63.8 ms | 0 |
| Three-candidate Poisson LOO search | 32 | 2.5 ms | 1,928 |
| Three-candidate Poisson LOO search | 128 | 7.0 ms | 7,688 |
| Three-candidate Poisson LOO search | 512 | 17.5 ms | 30,728 |

Caller-owned input/output buffers are counted separately in the fixture. Zero graph transients
for SUR does not mean zero GPU memory: it writes caller outputs and uses shader-local scratch.
Preparation includes allocation/upload and fixture creation. First execution includes deferred
pipeline work; cached resident measurements are not cold-driver performance. Compilation, readback,
input/output bytes and result certificates are retained with every observation.


## Random and unit-effects spatial lag maximum likelihood

`GPUSpatialPanelLagMaximumLikelihood` fits `y = rho Wy + X beta + unitEffect + error`
on complete balanced panels in period-major order. The random-effects fit alternates analytic
variance-component shrinkage `phi` with an exact dense log-determinant profile search for `rho`;
unit effects use within-entity demeaning. The declared capacity is 2–32 entities, 2–32 periods
and 1–7 predictors. Weights must be nonnegative, row-standardized, zero-diagonal CSR with sorted
unique neighbors and no islands. Missing observations invalidate the complete fit.

`effects` defaults to `random`. `maximumIterations` (1–128, default 64) bounds outer sweeps;
`maximumSearchIterations` (1–64, default 32) bounds each golden-section search. `tolerance`
(default 1e-5) controls phi change and final rho bracket width. Rho is restricted to [-.999,.999],
and random phi to [1e-5,1]. Boundary summary bit 1 flags rho near either endpoint; bit 2 flags
random phi near either endpoint. Unit phi is exactly zero and does not set that bit.

With `p = predictorCount + 1`, output shapes are: `coefficients[p+2]` (intercept, slopes, rho,
phi), `covariance[(p+1)²]`, `summary[6]`, `status[1]`, and observation-sized `fitted`/`residuals`.
Unit intercept and its covariance row/column are NaN. Covariance matches spreg's conditional
`varb` expected-information block for beta/rho, excluding phi and innovation variance.
Summary is `[sigma2, reportedLogLikelihood, sweeps, phiChange, rhoBracketWidth, boundaryBits]`.
The log likelihood deliberately follows spreg 1.8.3 reporting: unit effects retain its additional
`-NT/2 * log(NT)` normalization; comparing random and unit reported values as a likelihood-ratio
test is inappropriate. Fitted values and innovations use the quasi-demeaned response convention.

Fit status is 0 converged, 1 invalid/incomplete inputs, 2 singular or nonpositive/nonfinite
innovation variance, and 3 iteration/search exhaustion. Exhaustion exposes provisional estimates
and diagnostics but NaN covariance. Other failures never publish completed inference.

Optional `hausman: {tests, status}` additionally fits unit effects and compares slopes/rho using
spreg's `RE.varb - FE.varb` convention. `tests[3]` is `[statistic, df, p]`. The compatibility df
is `predictorCount + 2` (spreg's `RE.k`, including intercept and rho), **not** the contrast rank
`predictorCount + 1`. Status is 0 valid, 1 unsuccessful constituent fit, or 2 non-positive-definite
covariance difference/nonfinite comparison; invalid comparisons return NaNs. This explicitly
rejects negative Hausman statistics instead of interpreting them as chi-square inference.

The pinned `statistics-advanced.py` generator executes spreg 1.8.3 `Panel_RE_Lag`, `Panel_FE_Lag`
and `panel_Hausman`, and mgwr 2.2.1 full response-smoother inference. Native tests compare panel
estimates, conditional covariance, innovations, reported likelihood and valid/invalid Hausman cases,
and exercise missing data, singular within-design, zero variance and both iteration budgets.

{/* lens:GPUSpatialPanelLagMaximumLikelihood */}

::lens{scenario=gpu-spatial-analysis.spatial-panel-lag-maximum-likelihood rung=result caption="Inspect random-effects spatial lag estimates and Hausman diagnostics."}

::lens{scenario=gpu-spatial-analysis.spatial-panel-lag-maximum-likelihood rung=graph caption="Inspect the executed panel likelihood and diagnostic kernels."}
