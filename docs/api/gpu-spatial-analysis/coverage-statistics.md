# Bounded statistical estimation and validation

Import from `@luma-contributors/analysis/gpu-spatial-analysis`.

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
| ST01 | `GPUInstrumentalVariablesCovariance`; `GPUSpatialTwoStageLeastSquares.robust` | HC0/HC1/HC2/HC3, one-way clustered CR1 and spatial-HAC IV covariance; robust SE/z/p table and returned-covariance PSD certification | Entire HAC-kernel PSD certification and multi-way clustering |
| ST02 | `GPURestrictedRegimeRegression` | Common/varying intercept and slopes; pooled covariance and Chow F; optional HC0/HC1 or spatial-HAC covariance and coefficient/joint Wald chi-square tails | — |
| ST03 | `GPUSpatialPanelErrorMaximumLikelihood` | Balanced unit, time and two-way fixed-effects spatial-error Gaussian ML; CPU/operation forms; pooled LM error and robust LM lag/error tests | Independent-unit random effects, larger period-effect designs, Hausman tests, clustered/serial inference |
| ST04 | `GPUSeeminglyUnrelatedRegression` | One-step or iterated feasible SUR and fixed-GM spatial-error SUR, joint error covariance and full system coefficient covariance | small-sample inference |
| ST05 | `GPUProbitRegression` | Binary probit, expected-information errors, average marginal effects and three spatial residual tests; optional weighted one-step latent spatial-lag linearization with HC0 errors | Constrained spatial maximum likelihood and strong-dependence bias correction |
| ST06 | `GPULocalCollinearity`; `GPUGeneralizedGeographicallyWeightedRegression`; `GPUMultiscaleGeographicallyWeightedRegression` | Weighted collinearity diagnostics; exact GLM LOO bandwidth search; MGWR finite-grid/golden-section LOO or AICc selection and bandwidth-conditioned backfit inference | MGWR adaptive-neighbor selection and uncertainty from bandwidth reselection |
| ST07 | `GPUMultivariateLocalGearyPermutationTest` | Joint-vector conditional randomization, replay, equal-tail or esda minimum-tail p-values and optional Bonferroni adjustment | Exact upstream classification-label parity and additional correction families |
| ST08 | `GPURankMarkov` | Full-rank/geographic-rank transitions, common complete panels and absolute-rank mobility | Fractional ties and incomplete-panel interpolation; optional diagnostics retain 64 states |
| ST09 | `GPUFuzzyContiguity`, `GPUIslandAttachment` | Planar filled-polygon separation and nearest original-island attachment, with added-edge provenance | Geographic coordinates, topology repair and GEOS robust-predicate parity |
| ST10 | `GPUSpatialCrossValidation`, `GPUSpatialCrossValidatedOLS`, `GPUNestedInterpolationValidation` | Block/group/environmental folds, buffers, bounded NNDM/kNNDM, training-only OLS and nested covariance/RBF selection with held-out metrics | [Explicit design, optimizer and capacity contracts](./spatial-validation-advanced.md) |
| ST11 | `GPUSpatialInteraction` | Four constraint modes with grid or continuous bounded profile calibration, IPF, deviance and marginal residuals | multiple covariates, coefficient uncertainty |

## IV covariance and robust spatial 2SLS

`GPUInstrumentalVariablesCovariance` takes structural design `Z`, instrument design `H` and
structural residuals from the same fitted IV model. Both designs omit the implicit intercept.
The coefficient count is 1–15; the instrument count is between the coefficient count and 16;
there are at most 2^24 observations (float32 counts), and fewer when the design, instrument or
score rows exceed the device's storage binding size. All estimation rows must be finite. There is
no silent complete-case change after a model has already been fitted.
For `n` estimation rows, `design` is `n * coefficientCount`, `instruments` is
`n * instrumentCount`, and `residuals`/optional `clusters` have `n` entries. Inputs are packed
row-major float32 designs and residuals (uint32 cluster keys). `output.covariance` has
`(coefficientCount + 1)^2` float32 values, and `output.status.code` is one uint32 word.

Every row sum (column means, the centered moments `H'H`, `H'Z`, `diag(Z'Z)` and the meat matrix)
is a row-tile reduction merged in a fixed pairwise order, so results are reproducible; one
invocation solves the small projection system. The HAC kernel is applied over its CSR neighbors,
one invocation per row (`O(nnz · p)`), never over all pairs.

The covariance uses the projected-design influence matrix, not an OLS sandwich applied to
endogenous regressors. Columns are centered and norm-scaled before both Cholesky solves. The
intercept influence is recovered with its cross terms, so the output includes the full
`(coefficientCount + 1)²` covariance in intercept-first row-major order.

- `HC0` uses squared residual scores without a finite-sample multiplier.
- `HC1` multiplies HC0 by `n / (n - p)`, where `p` includes the intercept.
- `HC2` and `HC3` divide residuals by `sqrt(1-h)` and `1-h`, respectively, using projected-design
  leverage including the intercept. A denominator at most 1e-6 or leverage below -1e-6 is invalid.
- `CLUSTER` requires dense `uint32` `clusters` and `clusterCount` capacity; only occupied groups
  count in the CR1 multiplier `G/(G-1) * (n-1)/(n-p)`. At least two groups must be occupied.
- `HAC` takes a separate spatial kernel, without a finite-sample multiplier. It must be symmetric,
  nonnegative, positive semidefinite and have unit diagonal. Bounds, finite weights, symmetry
  and diagonal are checked on GPU. Positive semidefiniteness is a caller precondition.

The returned covariance is certified PSD to relative tolerance 1e-5 with a shifted Cholesky
test; zero and rank-deficient covariance are accepted. This does not certify the entire HAC kernel.
Covariance status is `0` success, `1` singular projection, `2` insufficient observations,
`3` nonfinite inputs/results, `4` invalid HAC kernel, `5` invalid leverage, `6` invalid clusters
and `7` a failed PSD certificate. Failed covariance entries are NaN.

For spatial 2SLS, set `robust: {method, kernel?, clusters?, clusterCount?, output: {covariance, status: {code}}}`.
The coefficient table's SE, z and p-value columns use the robust covariance. The estimator
publishes point estimates and classical Anselin–Kelejian diagnostics with their own status.
For HAC, the direct GPU contributor and both named operation forms accept the optional positive
certificate `robust.kernel.status.complete`. A zero certificate, a nonzero `overflow` or
`candidateOverflow`, or recorded count loss gives covariance code 4 and NaN robust covariance and
table SE/z/p columns. Point coefficients, structural residuals and classical summary diagnostics
remain usable. A nonzero certificate still requires the other supplied loss checks to pass. The
graph reads the kernel views on each submission, so restoring complete contents restores robust
inference on the same graph.

## Regimes, panels and SUR

SUR, including spatial SUR, supports 16,777,216 aligned observation rows per equation with tiled
reductions, subject to bindings. Panel-error ML accepts at most 16,777,216 (`2^24`) total panel
rows and `periodCount` in `[2, 16,777,216]`, with `entityCount * periodCount <= 16,777,216`.
These are different from `GPUSpatialPanelRegression` (pooled/unit 2SLS): 2–1,024 periods and
at most 16,777,215 (`2^24 - 1`) total rows. Repeated edge allocations and device storage bindings
can impose smaller bounds on either panel model. See [panel fixed effects](./panel-fixed-effects.md)
and the separate [panel-lag ML routes](#gpuspatialpanellagmaximumlikelihood).

`GPURestrictedRegimeRegression.common` includes the intercept followed by each predictor.
A true entry estimates one shared coefficient; a false entry estimates one coefficient per regime.
Output order is source-column-major, with regime order inside each varying column.
With `n` response rows and `p = predictorCount`, supply `n * p` row-major predictors and `n`
uint32 regime IDs, with 1–7 predictors and 2–8 regimes. `common` has `p + 1` booleans.
If `k` counts shared columns once and varying columns `regimeCount` times, outputs contain
`k` float32 coefficients, `k*k` covariance values, `n` fitted values, three Chow values and
one uint32 status word. Optional stability has `(p + 2) * 3` float32 values.
The unrestricted model is limited to 16 coefficients; its complete cases are exactly those of the
restricted model. Covariance uses the restricted model's pooled `SSE / (completeRowCount - k)` estimate,
where `k` is the restricted coefficient count defined above and `completeRowCount` counts the
shared complete estimation rows.
The Chow output contains `[F, numeratorDegrees, denominatorDegrees]`. Optional `stability` contains `[Wald chi-square, degrees, upper-tail p]` for each source coefficient and then jointly. It tests equality of the varying regime coefficients, matching `spreg.Chow`; common columns are `[NaN, 0, NaN]`. This is a different hypothesis from the nested-model F test.
Empty or singular regimes are explicit, and a singular unrestricted fit does not erase a valid
restricted coefficient estimate. The joint test assumes independent homoskedastic normal errors.
Optional `robust: {type: 'white', smallSample?: boolean}` adds HC0 covariance
`B X' diag(e²) X B`, where `B = (X'X)^-1`, using the restricted model's residuals.
`smallSample: true` selects HC1, multiplying by `n / (n - k)` for the complete-case count `n`
and restricted parameter count `k`. Alternatively, `robust: {type: 'hac', kernelWeights}`
uses Kelejian–Prucha covariance `B X' diag(e) K diag(e) X B`, as in `spreg` spatial HAC.
Supply `output.robust` together with `robust`: `covariance` has the raw-design layout of the
classical covariance; `chow` is `[joint Wald chi-square, degrees, upper-tail p]` for equality
of all varying coefficients; `stability` has the same per-column and joint rows as classical
stability. The existing `output.chow` remains the nested-model F test. Neither robust test
uses an F distribution. Omitting `robust` preserves the existing results and command graph.
Robust inference requires an identified restricted model and positive restricted residual degrees
of freedom (`n > k`); it does not require the unrestricted comparison to fit. Classical status
can therefore be `3` (unrestricted singular/nonfinite result) or `1` (no unrestricted residual degrees) while
robust status is zero. Classical restricted codes 2/3 also cover nonfinite restricted/unrestricted
computed residual sums, including after successful solves; neither code proves singularity alone.
Classical outputs keep their existing status-dependent availability.

HAC kernels must be square on the original response row space, with sorted unique CSR neighbors,
finite nonnegative weights, exact symmetry and unit diagonal, including masked rows. Build a
fixed-bandwidth kernel with self weights using the spatial-weights contributors, or upload one.
Row-normalized kernels and metadata declaring asymmetry or no self weights are refused at
construction; use an unstandardized symmetric kernel with self weights. GPU validation checks
CSR values, symmetry and the diagonal. Masked/incomplete observations have zero scores in both
kernel dimensions; weights are never renormalized, and an island retains its diagonal term.
Supplied nonzero `overflow`, `candidateOverflow`, `iterationLimitReached`, `iterationLimit`, `invalidInput`, `invalidCount`,
`uncertainCount` or `approximation`, or a supplied `converged = 0` or `complete = 0`, suppresses
robust inference. `requiredCount` above the terminal CSR offset, supplied `count` or allocated
neighbor capacity also suppresses inference, even without an overflow port; equality is accepted
only when all those bounds hold. A supplied `count` below the retained terminal offset is
inconsistent and invalid. Spare allocation cannot hide missing kernel edges. Every supplied kernel
status port must contain exactly one uint32 value in the CPU and GPU forms. Explicit `undefined`
optional status entries are omitted by the operation forms.

`output.robust.status` uses the exported `RESTRICTED_REGIME_ROBUST_*` constants:
`OK = 0`; `FIT_UNAVAILABLE = 1` for a failed restricted fit or no restricted residual degrees;
`INVALID_KERNEL = 2` for an
incomplete or invalid kernel; `INVALID_COVARIANCE = 3` for a nonfinite covariance or a tested
contrast covariance that is not positive definite, including a singular contrast covariance.
A usable robust test requires finite covariance and a numerically positive-definite covariance
of **every tested contrast**, with information above the uncertainty from both residual formation
and score accumulation. Positive marginal variances alone do not establish joint rank. Both forms
check the full contrast factorization, including its remaining Schur pivots, against propagated
uncertainty. An all-ones HAC kernel has zero meat by the OLS normal equations; a two-block
kernel can have rank-one meat and positive marginal variances while still invalidating a
two-degree joint test. Exact-fit residual rounding is likewise insufficient information for
HC0 or identity HAC. These paths report robust status 3 rather than a significant Wald result;
small representable nonzero residuals remain eligible when their information is resolved.
Nonzero status suppresses every robust float result (test rows become `[NaN, 0, NaN]`), while
classical outputs retain their meaning. Common-column and no-varying-column joint tests are
`[NaN, 0, NaN]` even with status zero. Residuals and contrasts use the shifted basis; covariance
is transformed to raw coefficients. Shifting a common-slope predictor leaves the tests unchanged.
Shifting a varying-slope predictor under a common intercept changes the model; under varying
intercepts it preserves slope tests and the joint test, but changes the intercept-only hypothesis.
The CPU twin and graph operation accept the same estimator; CPU calls supply kernel arrays,
operations supply graph names, and contributors supply `GPUSpatialWeights` views.
Joint intercept/slope contrast rows are rebased before covariance factorization in both forms.
Output views must have disjoint logical ranges. Disjoint aligned prefixes may be shared across
separate dispatches, such as fitted values and robust Chow, but prediction cannot bind the same
physical buffer read-only for coefficients and writable for fitted values.

Rows (at most 2^24, and within the device's binding size for the predictors) are reduced in
parallel: per-regime means and centered moments come from a stable regime sort and one workgroup
per regime, one invocation solves both nested systems, and the residual sums of squares are
row-tile sums merged in a fixed pairwise order.
Robust scores and their residual-error bounds additionally require a single binding of
`response.length * (2*k + 3)` float32 words (`response.length * (2*k + 3) * 4` bytes),
including masked original rows; the device's
storage binding limit can therefore reduce the row limit when robust inference is requested.

For 37 original rows and three restricted coefficients this score binding needs 1,332 bytes;
a 1,331-byte storage-binding limit rejects it. The complete-case count `n` determines degrees
of freedom and HC1 correction, rather than allocation.
The sandwich reduction uses the same fixed row tiles, visiting HAC neighbors in CSR order.
The legacy `fitted` port evaluates raw float32 coefficients; large predictor/intercept terms can
lose precision through cancellation. The shifted residual and inference paths avoid that loss.

Both systems are solved in a shifted basis, never from raw cross-products, so a predictor with a
large mean (a year, an elevation, a projected northing) costs no precision. With a varying
intercept the predictors and the response are centered on their regime means; with a common
intercept the response and the common-slope predictors are centered on their pooled means. A
varying slope under a common intercept is not invariant to a shift, so when such a predictor has a
mean larger than its spread the fit changes basis instead: `x - mean` takes the place of the
constant and the raw columns `x 1[g]` stay (`1 = (sum_g x 1[g] - (x - mean)) / mean`), which spans
the same model with a conditioning that does not depend on the size of the mean. The published
coefficients and covariance are those of the raw design. Regime means are float32, so the sums of
the deviations from them are carried through the moments and every difference of a value and its
shift is fixed before it meets a small term (the shader compiler of the test GPU otherwise
regroups `(y - mean) - fit` as `y - (mean + fit)`): a response whose mean is far above its
residuals keeps its Chow statistic. The Chow numerator is the direct sum of
squared differences of the two fits (equal to `SSE_restricted - SSE_unrestricted`), and each Wald
contrast is formed from differences of shifts before it meets the coefficients. A system is
singular (status 2 restricted, 3 unrestricted) when a Cholesky pivot of the unit-diagonal normal
matrix is below `1e-3`, the limit of `GPUOrdinaryLeastSquares`; `GPUSpatialRegimeRegression`,
`GPUSeeminglyUnrelatedRegression`, `GPUProbitRegression` and
`GPUSpatialPanelLagMaximumLikelihood` apply the same limit.

`GPUSpatialPanelErrorMaximumLikelihood` takes the existing spatial-error ML arguments plus
`periodCount` and `effects: 'unit' | 'time' | 'twoWay' | 'random'` (default `'unit'`). Data are balanced
and period-major; the supplied weights describe only the entity axis. Unit effects remove each
entity's mean over periods. Time effects remove each period's mean over entities. Two-way effects
use `value - entityMean - periodMean + grandMean`. `I_T ⊗ W` supplies the repeated spatial determinant.
The objective and variance divisor use `NT = N*T`. The published full Gaussian likelihood is
`logL = -NT/2 * (log(2*pi*sigma2) + 1) + T * log|I - lambda W|`.
The recorded spreg panel-error `logll` instead equals `logL - NT/2 * log(NT)` (an offset
of approximately −92.90882426 at `NT=48`); do not silently compare these two normalizations.
This panel-error convention is distinct from [panel-lag ML](#gpuspatialpanellagmaximumlikelihood),
which retains that additional normalization. For unit effects,
`getGPUSpatialPanelErrorMaximumLikelihoodModelResult(readback, {entityCount, predictorCount})`
counts `k = N + predictorCount + 1` (absorbed entity intercepts, slopes and lambda), uses
`df = NT - N - predictorCount - 1` and filtered RSS `NT * sigma2`. It preserves the absorbed
NaN intercept, the published likelihood and unavailable AIC/Schwarz/AICc as `null`. This is not REML or a short-panel bias correction. It inherits the existing bounded
trace-series search, boundary and approximation diagnostics. The absorbed intercept table row and
information criteria are NaN. A predictor in the absorbed effect space is singular (fit status 1).
For time/two-way effects the filtered regression also profiles period contrasts: cross-entity
centering and a general spatial matrix do not commute. Omitting these contrasts gives the wrong
likelihood on asymmetric weights or panels with islands. The bounded 16-column solve requires
`predictorCount + periodCount - 1 <= 15` in these modes; unit effects retain the total-row period
limit. Slopes and lambda keep their original table positions; period intercepts are absorbed and
not returned. Residuals are structural residuals after profiling the selected effects. Conditioning
slots describe the complete filtered design, including the period contrasts in time/two-way mode.
Pseudo R squared uses the within-transformed response and its fitted values. Any nonfinite
observation (including an entire period with no complete rows) gives fit status 3; missing panels
are unsupported. Fit status 2 means too few rows. A boundary fit retains status 0 but sets
`summary[16] = 1`, `summary[15] = 0` and optional `status.converged = 0`; its lambda standard error
does not describe an interior maximum. These conventions apply to every fixed-effects mode.

For larger period-effect designs, use `computeSpatialPanelErrorMaximumLikelihoodOnCPU` or
`CPUSpatialPanelErrorMaximumLikelihoodOperation` with the same `effects`, `periodCount` and
`predictorCount`. Both CPU forms are checked against dense Gaussian ML on a 4-entity, 16-period
asymmetric graph with an island, including every table, summary, residual and diagnostic output.
The CPU solve explicitly stores the period contrasts; its memory and factorization work grow
with the period count. Removing the GPU guard alone is insufficient: the shared filtered
likelihood solve still accepts at most 16 columns, including the implicit intercept.

`effects: 'random'` is deliberately refused. For **independent** unit effects with
`A = I - lambda W` and `phi = sigma_e / sqrt(T sigma_u² + sigma_e²)`, within contrasts have
covariance `sigma_e² A⁻¹ A⁻ᵀ`, while the normalized between contrast has covariance
`sigma_e² [A⁻¹ A⁻ᵀ + (phi⁻² - 1) I]`. Its likelihood needs the extra determinant
`log|I + (phi⁻² - 1) A Aᵀ|` and its corresponding GLS inverse, beyond `T log|A|`.
The existing bounded powers of W cannot supply that determinant for general asymmetric W.
Scalar quasi-demeaning as in the panel-lag random model would instead spatially filter the unit
effects, fitting a different covariance. Use a dense independent-unit random spatial-error ML
estimator for this mode; no approximation or random-effects result is published here.
For example, the two 3×3 nilpotent graphs with links `1→2` and `1→2→3` have infinity norm 1
and every power trace zero. At `lambda = 0.4`, `phi = 0.5`, their extra log-determinants are
respectively `4.188441885601216` and `4.2205424637463995`. Thus even identical normalized
power traces and infinity norm cannot determine the independent-unit likelihood. All four
forms refuse `effects: 'random'` with the reason and the dense-estimator alternative; the GPU
operation validates this when lowering constructs its contributor.

`GPUSpatialPanelErrorMaximumLikelihoodOperation` and `CPUSpatialPanelErrorMaximumLikelihoodOperation`
share named ports, including optional diagnostics, residuals, convergence, trace errors and weights
loss flags. They add `entityCount`. With `logDeterminant: 'exact'`, both forms accept either
the `N` eigenvalues of the entity weights `W`, or the `N*T` eigenvalues of the expanded
block-diagonal panel graph (each entity eigenvalue repeated `T` times). GPU execution limits
the **supplied spectrum** to 1,024 values: the entity representation permits `N <= 1,024`
even when `N*T > 1,024`; the expanded representation requires `N*T <= 1,024`.
Both remain subject to the total-panel-row, design, command and device-binding limits above.
The CPU exact route accepts both lengths without imposing the GPU spectrum cap.
For the entity representation the nested error-ML estimator multiplies the log determinant,
`sum eigenvalue / (1 - lambda * eigenvalue)` and
`sum (eigenvalue / (1 - lambda * eigenvalue))^2` by `T`. Thus both the determinant and its
first two parameter derivatives describe the expanded graph. Equivalent entity and expanded
spectra produce equivalent tables, likelihood summaries, structural residuals and diagnostics,
up to floating-point reduction order. The panel-error Gaussian likelihood normalization above
is unchanged; [panel-lag ML](#gpuspatialpanellagmaximumlikelihood) keeps its separate normalization.
`computeSpatialPanelErrorMaximumLikelihoodOnCPU` accepts typed arrays and returns every port.
Completeness is evaluated on the original entity CSR, before period expansion: a nonzero loss
flag or a `requiredCount` above its retained terminal offset, supplied `count`, or capacity
suppresses the fit with code 3 and pooled diagnostics with code 1. Expanding four retained
entity edges over two periods does not make a five-edge requirement complete. Named GPU/CPU
forms accept `weights.status.count` and `weights.status.requiredCount`; clearing loss recomputes
the fit and tests. Count formats and ownership are checked before panel resources are declared,
so correcting a foreign count view permits the same-ID retry on the original graph.
The direct GPU panel contributors also capture a supplied entity `weights.status.complete`:
zero suppresses the fit and pooled diagnostics with those same codes, even beside clear
overflow flags. The guard is folded before expansion and reused by the pooled tests; no
certificate is inferred when this optional port is absent. Restoring it to nonzero permits
recomputation only when the other loss/count checks are also clear.
Its float64 QR drops GPU dimensional caps; it keeps the bounded trace-series objective and
1024-interval stationary-point search. It is a small reference fit, not an exhaustive global search.
Optional `diagnostics.tests` returns statistic/p-value pairs for pooled-panel LM error, robust LM lag and robust LM error. These fit pooled OLS on the original panel and use its residual variance, spatial lags and projected spatial fitted values, matching spreg 1.8.3. That variance is `RSS / (n - p)` (spreg `ols.sig2`, `n = N*T`, `p = predictorCount + 1`), where the cross-section LM tests of `GPUSpatialRegressionDiagnostics` use `RSS / n` (spreg `LMtests`): on the same residuals and block-diagonal weights a panel LM error is `((n - p) / n)^2` times the cross-section one, so compare each with its own reference and not with the other. Diagnostic status is 0 valid, 1 failed/incomplete pooled fit, or 2 degenerate/nonfinite test information. The robust tests require positive spatial fitted-value information beyond the weight trace. They are specification tests, not clustered uncertainty for the FE estimator.

These diagnostic codes are exported as `GPU_PANEL_ERROR_DIAGNOSTIC_STATUS_OK`,
`GPU_PANEL_ERROR_DIAGNOSTIC_STATUS_INCOMPLETE` and `GPU_PANEL_ERROR_DIAGNOSTIC_STATUS_DEGENERATE`.
Both forms remove the constant component of the projection input and treat a projected
spatial-fit residual norm at most `1e-6` of that centered lagged-fit norm
as numerical rank zero, returning diagnostic status 2 and six NaNs. Positive projection
roundoff alone is insufficient to identify the robust tests.
The pooled regression is formed on stably centered data. Its spatial fitted vector retains
`mean(y)*(W1-(W1)[0])`, so a varying row-sum contribution remains signal. A common response
shift cannot change the rank decision when `W1` is constant. Exact pooled normal-score zeros
also reject a fit made entirely of rounding error without imposing a small-slope cutoff.
Both `weights.status.overflow` and `weights.status.candidateOverflow` suppress all pooled tests
with diagnostic status 1, alongside suppression of the ML fit with fit status 3.
An exactly affine pooled response has `RSS = 0`: all three LM statistics are undefined and
the diagnostic status is 2 with all six test values NaN. The CPU forms certify this on the
original float32 observations using exact arithmetic; positive QR rounding noise cannot turn
an exact pooled fit into valid test information. This does not impose a small-variance cutoff.
The GPU also checks the original input bits for an affine witness `y = a + b*x_j`, including
constant responses. A certified witness invalidates a rounding-created successful ML fit with
fit status 3, convergence 0, NaN table/residuals and a summary containing only the panel row count.
Successful pooled tests become diagnostic status 2 with six NaNs. Existing incomplete/singular
statuses retain precedence. This reuses the exact integer collinearity predicate: normal/zero
float32 coordinates must fit its 200-bit product-exponent span; an unsupported or uncertified
case continues through the existing numerical fit. The CPU Gram certificate covers more general
multivariate/effect-space exact fits. Seeded power-trace standard errors remain meaningful on a
zero-variance fit failure and are retained by both forms. Small positive variances are not rejected.
An additional GPU certificate compares exact within-entity differences (within-period differences
for time effects) against a common slope. It can reject an absorbed-shift exact ML fit while
leaving its valid pooled diagnostics intact. This is a sufficient single-column certificate;
the CPU checks full predictor rank with exact balanced-panel contrasts before testing exact fits.
Pooled robust tests retain projected information separately from the weight trace and use the
direct residual/projected-fit score, avoiding cancellation at small fitted slopes.
LM error divides its residual-lag score by the variance before squaring, preserving response-unit
invariance without overflowing two factors that should cancel. No diagnostic status 0 is
published unless all three statistics and probabilities are finite.
CPU stochastic trace errors are also retained for missing-response and singular-design failures;
incomplete weights still suppress them. Exact scalar-identity weights select the negative-boundary
flat-profile tie, one candidate, convergence 0 and undefined lambda inference. Exactly orthogonal
CPU predictions have zero slopes and undefined pseudo R² when fitted values are constant.

`GPUSeeminglyUnrelatedRegression` requires 2–4 equations with the same predictor count but
potentially different predictor values. Data are equation-major, then row-major; observation IDs
must be aligned explicitly. Any nonfinite value in any equation excludes that observation from all
equations. The total number of coefficients, including each intercept, is at most 16.
Equation OLS residuals estimate a **joint** cross-equation covariance with divisor `n`, followed
by one full feasible-GLS solve. The output covariance is the conditional GLS information inverse
using that first-stage error covariance. Singular design and singular error covariance have
separate status codes. Omitting `iteration` retains this one-step result.
Residual covariance terms preserve the usual float32 product/divisor order when the product is
finite. If an individual product overflows, the GPU fixes one residual divided by `n` before
multiplication, so a representable covariance is not rejected solely for that intermediate
overflow. Truly nonfinite or singular covariance still reports status 3.
Computed residuals at their arithmetic error floor are treated as zero before covariance
factorization: the bound is `8 * width * epsilon * (abs(centered response) + sum(abs(fitted terms)))`,
where `width = predictorCount + 1`,
using float32 epsilon on GPU and float64 epsilon on CPU. It scales with the response and fit;
there is no absolute small-variance cutoff. An exactly fitted equation is singular status 3
before any GLS solve (count zero), including with an iteration budget.

Optional `iteration: {maximumIterations, tolerance?}` repeats joint residual covariance (divisor
complete-case `n`) and GLS on a fixed command graph, with no host readback. The budget includes
the first GLS fit and is 1–256 on the GPU; the CPU twin obeys the requested budget without that
graph-size cap. The default tolerance is `1e-5`. Optional root `parameters` is a one-element
float32 tolerance view, packed with `getGPUSeeminglyUnrelatedRegressionParameterValues` and
written via `GPUParameterBuffer.write` between encodings; it overrides `iteration.tolerance`
without recompiling. Named operations accept its graph-value name and CPU inputs accept its
Float32Array. Omitted views upload the convenience setting through the same runtime layout,
so changing that convenience value does not change WGSL. GPU tolerance must round to a finite normal
positive float32 (at least `2^-126`, approximately `1.17549435e-38`); smaller values are refused
at construction because rounding or subnormal flushing can turn the stopping threshold into
zero. CPU convenience tolerance accepts every positive finite JavaScript number (including
`1e40`) and compares in float64. Shared parameter views retain the GPU layout/validity rule.
An invalid runtime parameter reports `GPU_SUR_STATUS_INVALID_PARAMETERS` (5), count zero and
NaN numeric fit outputs. The parameter length is `GPU_SUR_PARAMETER_LENGTH` (1).
Every round's four kernels use separate GPU-written indirect arguments; finished rounds dispatch
no solve or tile work. Bounded prelude batches reset those arguments once per frame.
Convergence requires
`max_j(abs(new[j]-old[j])/max(1,abs(old[j]))) < tolerance`, initially comparing against equation
OLS coefficients. These coefficients are compared in original predictor coordinates for ordinary SUR and in
original filtered predictor coordinates for spatial SUR, with the response anchored at its first
value in the spatial comparison. Centering used by the numerical solve does not redefine the
stopping criterion. If initial residuals are supplied (including internally in spatial mode),
round 1 only consumes that seed: convergence requires at least one subsequent GLS fit using
residuals from the fitted coefficients. A one-round seeded budget therefore reports status 4;
refreshing to singular covariance reports status 3 and retains count 1.
A nonspatial fixed point solves the Gaussian ML equations; convergence does not
certify a global optimum. Optional uint32 `output.iteration.status.iterationCount` (length 1) counts completed GLS
fits and is available in one-step mode too. Joint `output.status` is 0 successful one-step or
converged iteration, 1 insufficient complete rows, 2 singular design, 3 singular residual
covariance, 4 budget exhausted, or 5 invalid runtime parameters. Status 4 retains the last coefficients and residuals.
`residualCovariance` is the covariance **used by** the last completed GLS round, and `covariance`
is its conditional GLS information inverse; they are not recomputed from the last residuals.
If an iterated round encounters a singular covariance, status 3 replaces the fit with NaN numeric
outputs; the count retains the number of earlier successful rounds. Legacy one-step status 3
continues to retain the singular residual covariance for diagnosis.

Optional `spatial` selects the `spreg.SURerrorGM` convention: equation OLS residuals estimate Kelejian–Prucha spatial-error parameters; filtering transforms responses, designs and original OLS residuals before the joint GLS solve. The intercept and full covariance are transformed back to original units, and reported residuals are structural residuals. This selected mode requires complete aligned rows and nonnegative row-standardized zero-diagonal weights without islands; it rejects a mask or caller-supplied `initialResiduals`. The shared weights are checked once with bounded row tiles, and the resulting admissibility flag applies to every equation. Each equation has its own parameter and GM status; status 4 denotes inadmissible weights. Nonspatial `initialResiduals` allows an explicit first-stage residual covariance, with the same common complete-case mask.

With both `spatial` and `iteration`, GM parameters remain at their first-stage values while GLS
iterates on the filtered system. Only the first round uses the filtered original OLS residuals;
later rounds use residuals from the current filtered system coefficients. This is conditional
iterated FGLS with fixed GM parameters, rather than joint maximum likelihood for spatial lambda.
An invalid first-stage parameter suppresses every row of its filtered equation, including
islands with no neighbor terms. The joint fit then reports status 1 with NaN numeric outputs;
equation statuses retain the GM failure or invalid-weight reason.
The direct GPU contributor consumes both spatial weight `overflow` and `candidateOverflow`:
either flag makes every equation inadmissible (equation status 4, joint status 1). CPU and named
operation forms refuse any `spatial.weights.status` object because they cannot consume these
ports; use the direct GPU contributor for status-bearing weights.
Rows within the accepted `1e-5` row-sum tolerance are normalized consistently before GM moments
and every filter. Responses are shifted by their first observation before OLS, GM and filtering, then that
origin is restored to the structural intercept. This keeps the filtered intercept constant and
preserves slope/covariance invariance under a constant response shift; structural residuals use
the shifted fit on GPU to avoid cancellation against a large published intercept.
The GPU first stage retains the low words of the lagged residual moments. It projects the two
variance moments before the lambda search, where their small difference can affect a covariance
transformed through a large predictor origin.

## Probit and local collinearity

`GPUProbitRegression` uses centered/scaled predictors, Fisher scoring and monotone likelihood
backtracking. It accepts 1–11 predictors and at most 2^24 rows (fewer when the predictors, or
the diagnostic residual rows, exceed the device's binding size). Each iteration is a row-tile
pass for the score and information, a one-invocation solve, one row-tile pass that evaluates all
24 halved line-search steps, and a one-invocation step choice; iterations are unrolled
(`estimatedCommandNodeCount`, about `6 · maximumIterations`) and gated on the GPU once the fit
stops. Complete cases are selected
jointly; included responses must be exactly 0 or 1. The expected-information SEs are model based.
Marginal effects average the normal density over estimation rows, then multiply by each slope;
they are not effects evaluated at the means.

For `n` rows and `p` predictor columns, inputs have `n * p` packed row-major float32 predictor
values, `n` responses and an optional `n`-word uint32 mask; the intercept is implicit. Outputs
have `p + 1` float32 coefficients and standard errors, `n` fitted probabilities, `p` marginal
effects, four summary values and one uint32 status word.
Its summary is `[completeRows, logLikelihood, deviance, iterations]`. Status `0` means converged,
`1` insufficient data, `2` singular information, `3` not converged and `4` nonbinary data (a complete response whose stored float32 bits are not +0, -0 or exactly 1.0,
so a subnormal value is nonbinary in both forms) or
separated data. Three invariants hold:

- Status 0 requires both the accepted step and the full, unshortened Fisher-scoring direction
  to satisfy `abs(step[j]) <= toleranceEffective * (1 + abs(candidate[j]))` for **every**
  coefficient `j`, using the same candidate coefficient for the accepted and full directions
  in the centered/scaled fit coordinates. A large coefficient cannot relax another coefficient's
  threshold. No tested direction may have been certified as
  separating or left undecided. A tiny backtracked step alone cannot certify convergence.
- Status 4 for binary data means separated, and only with a certificate: a direction of the
  coefficients that provably moves no row towards the wrong side and some row towards the right
  side. Data that are not separated are never status 4.
- Everything else that is not status 1 or 2 is status 3, with the last iterate written (and NaN
  standard errors): iterations or line-search steps exhausted, a step pinned by the bound on
  misclassified rows, a linear predictor that overflowed, or a small step on data that float32
  could not tell from separated. An ordinary well-conditioned fit is never status 3 at the
  default settings. Separated data whose certificate the search does not find are status 3 (or 2,
  when the information matrix of the runaway fit loses rank first), never 0.

Status 3 writes the iterate however large its linear predictors are: 2,001 rows that converge in 7
steps, stopped after 3 with a correctly classified row beyond a linear predictor of 16, are status
3 with their iterate. Every other failure writes NaN coefficients. All-zero/all-one outcomes are
status 4.

The GPU class, GPU operation and CPU forms accept positive finite `tolerance` up to
`MAXIMUM_PROBIT_TOLERANCE` (1e-2) and default to 1e-5. On CPU,
`toleranceEffective = min(tolerance, 1e-5)`. GPU execution rounds the resolved value to float32
and floors it at the smallest normal float32.
`toleranceEffective = max(Math.fround(min(tolerance, 1e-5)), 2^-126)`, avoiding a zero or
flushed-subnormal threshold. Requests in `[1e-5, 1e-2]` share the default certificate. Smaller
requests tighten it subject to the GPU floor. Each route uses
its effective threshold for both accepted and full steps, independently of separation tests.

`fitted`, `marginalEffects` and the diagnostic residuals are computed from the centered, scaled
design of the fit and its estimate, not from the published raw coefficients: the raw intercept is
`a - sum mean_j b_j` in float32, which for a predictor near 2^24 with a spread of five float32
steps is rounded to a step of 0.5. Rebuilt from the raw coefficients such a fit had fitted
probabilities of 0.185, 0.421, 0.691 where statsmodels has 0.136, 0.345, 0.618, and a marginal
effect 2.7% low; they now agree to 1e-7. The slopes were right throughout.

The likelihood of a row is `log Phi(s eta)` (`s = +1` for a response of 1, `-1` for 0), with `Phi`
evaluated through `erfc` so that its lower tail keeps relative accuracy down to the smallest normal
f32; the score and the information weight are ratios of the density to the two tail probabilities,
and no `1 - p` is formed. A row whose probability is within float32 rounding of 0 or 1 therefore
still pulls the fit as the likelihood requires. (`fitted` and the spatial diagnostics use
probabilities floored at `1e-7`.)

- A row on the right side of the fit is not bounded: once its probability rounds to 1 it adds
  nothing to the likelihood. 2,000 rows from a latent index `0.2 + 2x` plus one correctly classified
  row at `x = 9` (a linear predictor of 18.9 at the maximum) return the maximum of the other rows,
  slope 2.072, with status 0, as statsmodels `Probit` does. So does a fit whose maximum has a largest
  linear predictor of 12.06.
- A row on the wrong side leaves the float32 range of `Phi` near 13, so the line search accepts no
  step that takes such a row beyond ±12. A fit is pinned when that bound, not the likelihood,
  shortened the accepted step and the shortened step is below the effective tolerance while the full Fisher
  step is not: status 3 with the pinned iterate (it was status 4 with NaN coefficients; the data
  are not separated). The same 2,000 rows plus a response of 0 at `x = 9` are such a fit (the
  float64 maximum is a slope of 1.447 with that row at 13.2; the iterate has a slope of 1.314 with
  that row at 12); a response of 0 at `x = 4` is not (slope 1.862, that row at 7.7, returned with
  status 0).
- The same full-step check applies when likelihood backtracking, rather than the ±12 bound,
  shortened the step: a small accepted move with a large full direction remains status 3.
- Separated data have no finite maximum: some direction of the coefficients moves every row to the
  right side or leaves it where it is. The fit certifies separation by exhibiting such a direction,
  not by the size of a linear predictor, and a direction counts only when the sign of every row's
  move is known. There is no tolerance: a move towards the wrong side disqualifies a direction
  however small it is, and a row whose sign float32 cannot decide is not counted as standing still.
  Three kinds of direction are tested, each reduced over the rows with maxima of flags, which are
  exact whatever the row count:
  - A column that separates at its lowest or highest value: the rows above the lowest value of a
    column (or below its highest) all have the same response. Raw values are compared, which is
    exact, and the test does not depend on the fit. An indicator of a group of rows that all have
    a response of 1 is certified this way at any `maximumIterations` and any `tolerance`: one or
    three rows of 600, an indicator among eleven predictors, every second row, one row of
    100,000, three of 200,000, five of 400,000, five and twenty of 1,000,000.
  - A single-column threshold from the Fisher direction. With its rounding taken out (a column
    whose largest contribution to any row is below 1e-4 of the largest contribution of any column
    is dropped; the intercept is set so that the rows at an end of the kept columns do not move
    when their move is below that level), a direction with one column left is
    `sign · (x - value)`. Rows are compared with `value`, and in the next pass with the two data
    values next to it, so a column that separates at an interior value where both responses occur
    is certified one iteration after the direction points there: quasi-complete separation, and a
    three-valued predictor whose rows with both responses are at its middle value (100,002 rows;
    status 3 at the defaults before, status 2 with 128 iterations and status 0 with a tolerance
    of 1e-3).
  - The Fisher direction itself, for separation by a combination of columns. The move of a row
    takes its sign from the float32 value only when that is beyond the evaluation error bound,
    `(predictorCount + 5) 2^-23` of the sum of the magnitudes of the row's terms; a row all of
    whose terms are exactly zero stands still; any other row is undecided and the direction is
    not a certificate. Complete separation by two predictors is certified this way. Quasi-complete
    separation by a combination of columns, whose tied rows have moves that cancel to rounding, is
    not: 606 such rows ended as status 2 at the defaults and status 3 with 8 or 16 iterations.
  A certificate is kept once found, and a certified fit is status 4 however it ends: iterations
  exhausted, a singular information matrix (without a certificate that is status 2), a failed line
  search, an overflowing linear predictor, or a step below `tolerance`. statsmodels `Probit` does
  not converge on any of these scenes, and a linear program over directions (SciPy `linprog`)
  finds a separating direction on each.
- What the certificates need. A column that separates at an end of its range: nothing. Complete
  separation by one predictor: a budget of two iterations; by two predictors: seven;
  quasi-complete separation at an interior value: five (measured). With a smaller
  `maximumIterations` those are status 3. The `tolerance` no longer matters for an indicator (one
  row of 600 at a tolerance of 1e-2 was status 0 with a coefficient of 3.94 and a standard error
  of 5.39; it is status 4).
- Data that are not separated are never status 4. 1,000 rows `(x = -1, y = 0)`, 2,000 rows
  `(1, 1)` and the rows `(-1e-5, 1)` and `(1e-5, 0)` have a finite maximum at a slope of 6.02; the
  first Fisher direction moves the two central rows towards the wrong side by 1.5e-5 of their own
  terms, which a tolerance of 1e-3 took for separation (status 4 at every budget). They now
  converge in 23 steps to the statsmodels maximum (slope within 2e-6 standard errors), and are
  status 3 with their iterate when the budget is smaller. 60 samples of 25 to 69 rows with one to
  four predictors that a linear program and statsmodels agree are not separated are all status 0,
  as are an indicator group with both outcomes at 600, 20,000 and 100,000 rows and two rare-event
  fits.
- A candidate direction that moves no row towards the wrong side as far as float32 can tell but
  leaves some row undecided is remembered, and a later step below `tolerance` is then status 3,
  not 0. No scene of the specs needs this guard.

Optional `spatialDiagnostics` uses the fitted probit probability and generalized/standardized residuals to report `[Pinkse chi-square, p, Kelejian–Prucha z, two-sided p, Pinkse–Slade chi-square, p]`. Weights must have a sorted zero-diagonal CSR starting at offset zero, with finite nonnegative weights, including rows excluded by the mask; the induced common complete-case graph is used without renormalization. Diagnostic status is 0 valid, 1 failed/no-case model, 2 invalid/incomplete weights, 3 zero test information. Nonzero loss, invalidity, uncertainty or approximation flags in `weights.status`, or a zero `complete` or `converged` word, suppress the tests. A `requiredCount` exceeding the accessible CSR span, written `count` or allocated edge capacity also proves loss. Status views contain exactly one uint32 value; `iterationCount` is metadata. These residual tests do not estimate a spatial-lag coefficient.

The diagnostic definitions are explicit. Write `e = y-p`, `v = p(1-p)`, `g = phi(X beta)`,
`a = g e/v`, `s = e/sqrt(v)`, `J = mean(g²/v)` and `T = tr(W'W + WW)`.
Pinkse is `(a'Wa/J)²/T`, Pinkse–Slade is `(s'Ws)²/T`; both use chi-square(1) upper tails.
Kelejian–Prucha is the signed normal statistic `e'We / sqrt(sum_ij (wij² + wij wji) vi vj)`
with a two-sided p-value. The diagnostic probabilities are clamped to `[1e-7, 1-1e-7]`.

### Optional latent spatial-lag probit

Supply `spatialLag: {weights, output: {coefficients, standardErrors, status}}` to estimate
`y* = rho W y* + X beta + e`, `y = 1[y* > 0]`. Both coefficient vectors have
`predictorCount + 2` float32 values: intercept, predictor slopes, rho. The ordinary `output`
continues to report the independent probit. `GPUProbitRegressionOperation`,
`CPUProbitRegressionOperation` and `computeProbitRegressionOnCPU` accept the corresponding option.
Named operations accept `weights.status` fields as graph-value names. The pure CPU input accepts
the same fields as one-element uint32 arrays. A zero `complete` word, a nonzero `overflow` word,
or a `requiredCount` exceeding the accessible CSR span, supplied `count` or allocated capacity
suppresses the affected spatial stage.

The implemented Klier–McMillen-style probability-residual variant fits an ordinary probit
`beta0`, then forms `eta = X beta0`, `p = Phi(eta)`, `g = phi(eta)`, `u = y-p`.
The derivatives at `rho = 0` are `G_beta = g X` and `G_rho = g W eta`.
The error variance has zero derivative at zero for zero-diagonal W. The first-order equation is
`u + G_beta beta0 = [G_beta, G_rho] [beta, rho] + error`.
Multiply its target and gradient columns by `1/sqrt(p(1-p))` (using the fitted-probability clamp).
Project these weighted gradients onto **unweighted** instruments `[X, WX, W²X]`, including
the intercept in X and its lags W1 and W²1. Retain each lagged intercept when it carries
independent information (masks and islands can make its row sums vary). Construct the spatial
blocks from the centered/scaled ordinary design and center every completed nonconstant
instrument column before accumulating Gram moments; retain the ordinary intercept. These
changes preserve the full instrument span and avoid dropping independent columns solely due
to a large predictor origin. Regress the weighted target on projected gradients `H` in one
step. No further spatial iterations run.

Drop redundant instrument columns in declared order at a relative Cholesky factor pivot of
`1e-3` (squared pivot ratio `1e-6`); require at least as many retained columns as coefficients.
The second stage uses the same pivot limit. Errors use the HC0 sandwich
`(H'H)^-1 H'diag(r²)H (H'H)^-1`, where `r` is the weighted **structural** residual,
`weightedTarget - weightedGradient * estimate`, without a finite-sample correction.
The fit uses centered/scaled gradients and transforms beta and its covariance back to raw units.
Instruments preserve the span of raw X, WX and W²X. Masks and nonfinite observations remove both
the focus row and its neighbor contributions, without renormalizing W even when its metadata
declares a row transform. Islands remain in the ordinary and spatial fits, with zero spatial
products. A zero W cannot identify rho. An incomplete weights status suppresses the spatial fit.
Before normalizing the rho gradient column, require W eta to exceed the propagated rounding
floor of the ordinary fit on at least one row. The GPU floor in eta is
`(predictorCount + 8) * 2^-23 * [1 + sum_j |design_j| (1 + |beta0_j|)]`; the CPU uses
`64 * Number.EPSILON` for the leading factor. Propagate this floor through W, so rescaling
W does not turn ordinary-fit roundoff into identified dependence. Failure gives spatial status 3.
The CPU rounds all spatial coefficients/errors to float32 before classifying finiteness and
the rho boundary: a published rho equal to ±1 has status 4; a float32 overflow suppresses both
vectors with status 3. Empty edge arrays are supported: all-island diagnostics have status 3
and the spatial estimate has status 2. Writable ranges across all three output maps must be
disjoint. The contributor's copied configuration and nested port maps are frozen.

`GPU_PROBIT_SPATIAL_LAG_STATUS` names the scalar status codes:

| Code | Name | Result |
| --- | --- | --- |
| 0 | `OK` | Estimate and errors published |
| 1 | `MODEL_NOT_CONVERGED` | Ordinary fit status is nonzero; spatial outputs NaN |
| 2 | `SINGULAR_FIRST_STAGE` | Insufficient independent instruments or failed projection; NaN |
| 3 | `SINGULAR_SECOND_STAGE` | Projected gradients singular or numerical result nonfinite; NaN |
| 4 | `RHO_OUT_OF_RANGE` | `abs(rho) >= 1`; unconstrained estimate and errors published |
| 5 | `INVALID_WEIGHTS` | Malformed CSR (including excluded focus rows), negative/nonfinite weights or incomplete input status; NaN |

All existing GPU row/predictor/iteration limits apply. The optional design table additionally
requires `(4 * predictorCount + 6) * rowCount * 4` bytes in one storage binding; first-stage tile
partials require `(3k+3)(4k+5)+1` float32 words per tile for `k = predictorCount`. The CPU twin
does not impose GPU shape limits and keeps the explicit ordinary-fit iteration budget.

This is a local linearization about `rho = 0`, intended for small to moderate dependence;
strong dependence can produce substantial bias and estimates outside the stable interval.
The committed independent SciPy/NumPy oracle also simulates 2,000 rows for 50 seeds, normal
predictors/slopes `[0.8,-0.45]`, intercept 0.25, standard-normal errors and four equally weighted
ring neighbors. At true rho 0.2 the mean estimate is **0.21738**, sample SD **0.06593**, range
**[0.07631,0.34502]**. At true rho 0.6 it is **0.85474**, SD **0.08858**, range
**[0.64587,1.04845]**. This specified probability-residual/unweighted-instrument variant shows
upward bias in that experiment; it does not establish a general bias towards zero or numerical
equivalence to `McSpatial::spprobit`. The oracle records the exact weighting and commands.

The [seed-level simulation report](https://github.com/visgl/luma-contributors/blob/master/modules/analysis/test/gpu-spatial-analysis/spatial-regression/probit-spatial-lag-simulation.json)
retains all 50 estimates at each rho, seeds 0–49, library versions and model settings.
The Monte Carlo standard errors of the two reported means are **0.00932** and **0.01253**.
All 50 estimates at rho 0.6 exceed 0.6 (the smallest is 0.64587), so the proposed
bias-towards-zero description is contradicted on this experiment. The measured direction is
specific to this variant and design; it is not a universal directional-bias theorem.
Regenerate the complete report with:

```sh
OPENBLAS_NUM_THREADS=1 /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/spatial-regression/probit-spatial-lag.oracle.py --simulation
```

The independent mathematical oracle and named CPU-operation callback are checked without a
device on the 20- and 60-row fixtures, including every estimate, error, fitted row, effect,
diagnostic and status. The separate headless spec covers GPU execution of those same fixtures;
a passing CPU callback or WGSL audit does not qualify its hardware results.

<a id="local-collinearity-contract"></a>

`GPULocalCollinearity` uses supplied local estimation weights, including diagonal weights if the
fit uses the focal observation. For `n` weight rows and `p` predictor columns, supply exactly
`n * p` row-major float32 predictors and an optional `n`-word uint32 mask; `p` is 1–12.
Outputs are `n * p` float32 VIFs, `n` float32 correlation condition numbers and `n` uint32
statuses. The paired optional design outputs have `n` condition numbers and `n * (p + 1)`
variance proportions. Row count and CSR capacity remain subject to weight indexing and device
bindings. Every predictor uses the same complete neighbors. VIFs are the
diagonal of the inverse weighted correlation matrix; the condition index is the square root of
its maximum/minimum eigenvalue ratio. The intercept is excluded from this correlation calculation. Paired optional `designConditionNumbers` and `varianceDecomposition` reproduce mgwr's intercept-inclusive column-normalized design convention: multiply each design row by normalized spatial weight, not its square root, and report the variance proportion associated with the smallest singular value. A bounded one-sided Jacobi singular-value solver returns status 3 on exhaustion; status 1 means insufficient complete neighbors and 2 singular/nonfinite correlation or design. A smallest singular value at most `1e-7` of the largest is numerically singular, an explicit f32 conditioning limit.

Predictor means use differences from a complete neighbor before averaging, so translating identical
columns preserves singular status 2, capped correlation condition number, and NaN VIFs.
If supplied weights report `overflow` or `candidateOverflow`, or `requiredCount` exceeds `count`
or neighbor capacity or the retained terminal CSR offset, every local collinearity diagnostic
is NaN and every row has status 1.
This applies to root and per-chunk metadata. Restoring complete counts and clearing the flags
restores normal computation on the next encoding. All output ranges must be disjoint, including
`conditionNumbers` and `designConditionNumbers` written by different dispatches. Construction captures the predictor count,
weights descriptors and output maps; buffer contents remain per-frame inputs.
The direct GPU path also captures optional root/per-chunk `status.complete` certificates.
A supplied zero suppresses every float diagnostic and sets every row status to 1, even
beside clear overflow flags. Absent certificates supply no completeness guarantee.
The supplied certificate's contents remain per-submission inputs and can recover on a later run.

`GPUGeneralizedGeographicallyWeightedRegression.inference` returns local scale-one Poisson/logistic standard errors and diagonal influence. The covariance is `A^-1 X' V W² X A^-1`, with `A = X' V W X`; it matches mgwr GLM local uncertainty and does not assume that spatial weights are frequency weights. Prediction-only queries have zero diagonal influence because their responses are excluded from training. Failed fits retain NaN inference, including Poisson queries outside the `[-30, 30]` linear-predictor limit (status 4).

`bandwidthSelection` supplies 2–32 positive candidate bandwidths and outputs their exact leave-one-out mean deviance, the selected bandwidth and status (0 selected, 1 no valid candidate). Adaptive candidates must be integer neighbor counts within `maximumNeighborCount`. Each inner fit excludes its own focus, including from adaptive neighbor selection. Every eligible training row must converge for a candidate to be valid; ties choose the first candidate. The logistic deviance of a row is taken from its leave-one-out linear predictor, `2 log(1 + exp(-eta))` for a response of 1 and `2 log(1 + exp(eta))` for 0, so a confident prediction scores finitely (the float32 fitted mean is exactly 1 from `eta` of about 17 and cannot carry it). This is the default `criterion: 'loo-deviance'`. Optional `criterion: 'aicc'` uses in-sample fits and the mgwr Poisson/Binomial deviance/influence correction. `method: 'golden-section'` searches bounded fixed or adaptive AICc bandwidths with GPU gates and named convergence/budget status; see [Adaptive generalized GWR](./adaptive-generalized-gwr.md). The pinned oracle executes mgwr local GLM solves with the self weight removed to verify the exact candidate scores.

For nested spatial evaluation, supply one outer fold's **training mask** and set `predictExcluded: true`. Bandwidth selection, centering/solves and inference use only training rows, while finite held-out predictor/position queries receive predictions without reading their responses. The direct test changes held-out counts from zero to 100,000 and verifies unchanged candidate scores, selected bandwidth, coefficients and predictions. Apply this graph separately to each fold; score held-out responses afterward. Finite query offsets remain caller inputs.

## Joint randomization and rank dynamics

`GPUMultivariateLocalGearyPermutationTest` preserves the existing `GPUMultivariateLocalGeary`
statistic API and adds a separate inference producer. It supports at most 256 observations,
16 variables and 4096 permutations (`permutationCount`, fixed when compiling the graph). A common complete-case mask applies to all variables;
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
`rankingMethod: 'auto'` chooses pairwise ranking through 64 rows and sorted ranking above
that; explicit `'pairwise'` retains the 64-row reference bound. Sorted ranking accepts at most
65,535 logical states/rows, 65,536 periods and 2,147,483,648 panel values, subject to dense
`n * n` transition output, input/sort allocations and device bindings. Optional transition
diagnostics retain their own 64-state bound: requesting `output.steadyStates`,
`output.stateSparsity` or any `output.status` group constrains the job to 64 logical states,
even with sorted ranking. `output.homogeneity` is unavailable at every size because Rank Markov
has only one conditioning group. `output.status.converged` requires `output.steadyStates`.
For more than 64 rows, use `rankingMethod: 'sorted'` only when the diagnostic/status outputs
are omitted and every dense binding fits; optional `output.mobility`
does not impose the diagnostic cap. Check the actual props and device before constructing:

```ts
import type {Device} from '@luma.gl/core';
import {GPURankMarkov, type GPURankMarkovProps} from '@luma-contributors/analysis/gpu-spatial-analysis';

declare const props: GPURankMarkovProps;
declare const device: Device;
const plan = GPURankMarkov.getExecutionPlan(props, device.limits);
if (!plan.feasible) throw new Error('Rank Markov route or bindings exceed the reported constraints');
const contributor = new GPURankMarkov(props);
```

The plan reports `route`, `feasible`, `constraints`, `bindings` and `deviceChecked`; omitting
device limits leaves `deviceChecked` false. Dense counts and probabilities each require `n*n`
elements, regardless of the ranking route or complete-row count. The separate CPU reference
drops GPU pairwise/diagnostic and period caps but retains a 65,535-row dense-square bound;
its output sizes and total transition opportunities must still fit uint32.
The original `n × n` output capacity is retained when rows are excluded. Empty transition rows
stay zero. Optional existing Markov diagnostics retain their ergodicity assumptions; optional
mobility reports mean absolute rank movement per adjacent period, NaN below two complete rows.

## Weights and spatial evaluation

<a id="fuzzy-contiguity-contract"></a>

`GPUFuzzyContiguity` uses the existing feature→polygon→ring→vertex layout, with a packed
`float32x2` position view. Polygon distance combines segment distances with even-odd filled-part
containment, preserving holes and multipart features. Rings close implicitly. `tolerance` is the
maximum separation, equivalent geometrically to intersecting exact round buffers of half that
radius; it does not build approximated buffer polygons or use grid snapping.
Feature IDs/offsets permit at most 4,294,967,294 features; two-word position indexing permits
at most 2,147,483,647 vertices. Packed geometry, pair scratch, BVH dispatches and actual device
storage bindings impose smaller feasible limits. Required pairs and CSR prefixes saturate at
`0xffffffff`; overflow remains visible. Outputs are sorted binary CSR, one uint32 `featureStatus`
per feature, and one uint32 `added` per retained neighbor slot. Candidate pairs come from a BVH over
feature bounds expanded by the tolerance, and only candidates run the exact distance test, so no
feature-pair workspace is stored. The BVH leaves are sorted along a Morton curve of the bound
centers, so the search cost does not depend on the input order (16,384 shuffled squares take about
5 ms on the headless test GPU, against 110 ms with leaves in source order); the emitted pairs are
radix-sorted by row and neighbor, so each row stays ascending. The query box widening scales with
both the coordinates and the tolerance, so a pair the distance test accepts is found from both
features and the CSR stays symmetric. Feature status distinguishes valid, empty and malformed/nonfinite input. Simple valid
rings are required; self intersections and near-degenerate f32 predicates are not repaired.
Edges are binary and ordered, with `added = 1` only for strictly positive separation.
Output overflow reports final CSR capacity loss. The required neighbor count is exact while
it fits uint32; once the true requirement exceeds `0xffffffff`, the published `requiredCount` is a lower bound
saturated at `0xffffffff`, rather than the exact requirement.

<a id="island-attachment-contract"></a>

`GPUIslandAttachment` selects nearest finite supplied representatives for **original empty CSR
rows**, with lowest-ID distance ties. Selection is simultaneous, so islands may attach to each
other. Reciprocal insertion is the default; directed insertion is explicit. It preserves existing
weights and uses weight 1 only for added edges. Empty means no stored slots, not a zero-weight row.
Nearest comparisons use a shared distance scale to avoid squared-distance overflow or underflow
for finite float32 representatives. `nearestSearch` selects how the nearest is found and never
changes the result: `'scan'` tests every representative per island (islands × rows), `'grid'` takes
the 16 nearest candidates of each island from a `GPUNeighborSearch` grid and decides among them with
the scan's own comparison, and `'auto'` (the default) scans up to
`GPU_ISLAND_ATTACHMENT_AUTO_SCAN_MAXIMUM_ROWS` (4,096) rows and uses the grid above. Ties go to the
lowest row on both paths. An island whose candidate list cannot prove the answer (the farthest
candidate is not clearly farther than the chosen one, or the coordinates are so large that the grid
bounds or squared distances overflow) rescans every representative, so the result always equals the
scan's. A dedicated pass compacts unresolved islands and uses a 256-lane workgroup per island,
in batches of at most 65,536 representatives (at most 256 comparisons per lane per dispatch).
The CSR row-domain bound is 4,294,967,294. Over the **whole grid job**, both
`rows * 16 <= 4294967295` (candidate-product ceiling 268,435,455 rows) and
`4096 + 3 * ceil(rows / 65536) * ceil(rows / 16777215) <= 32768`
(command schedule) must hold. The current schedule admits 100,663,290 rows and rejects
100,663,291, before device limits; fitting the candidate product alone is insufficient.
The planner also checks `inputEdgeCapacity + rows * (reciprocal ? 2 : 1) <= 4294967295`
and actual storage bindings/allocations. Explicit `'scan'` accepts at most 65,535 total rows
(`GPU_ISLAND_ATTACHMENT_SCAN_MAXIMUM_ROWS`), distinct from auto's 4,096-row crossover
(`GPU_ISLAND_ATTACHMENT_AUTO_SCAN_MAXIMUM_ROWS`). Use the public class-static planner:

```ts
import type {Device} from '@luma.gl/core';
import {GPUIslandAttachment, type GPUIslandAttachmentProps} from '@luma-contributors/analysis/gpu-spatial-analysis';

declare const props: GPUIslandAttachmentProps;
declare const device: Device;
const plan = GPUIslandAttachment.getExecutionPlan(props, device.limits);
if (!plan.feasible) throw new Error('Island attachment exceeds the reported route constraints');
const contributor = new GPUIslandAttachment(props);
```

The plan's `constraints` report the failed numeric checks; `deviceChecked` is false when
device limits are omitted. Planning does not replace graph ownership, alias or runtime-data
validation. Degenerate grid inputs can require exhaustive batched work. That
fallback is never silent: the optional `output.status.uncertainCount` holds the exact number of
islands that used it (always 0 with `'scan'`). A count near the island count means the
representatives are too tied (15 or more at one address) or too large for the candidate list to
help: deduplicate representatives, or use `'scan'` only through its 65,535-row cap. On 60,000 shuffled rows with 25 percent islands the grid took
7 ms against 52 ms for the scan (Apple GPU, second run, readback included). Writing the output costs
each row's own degree. No representative is
inferred, and no implicit normalization is performed. Truncated CSR remains
ordered and bounded, with required counts, attachment IDs and per-edge provenance.
Inputs are planar `float32x2` representatives in the entity row space and sorted-unique CSR.
`output.attachment` is one uint32 nearest row per source row (`0xffffffff` if none needed/possible);
`output.added` is one uint32 provenance flag per retained edge. Distances use native coordinate
units; no implicit projection is applied.

Island attachment carries either input neighborhood loss flag into
`outputWeights.status.candidateOverflow`, using a graph-owned word when no caller destination
is supplied. A supplied candidate word is overwritten on every submission, including recovery;
`overflow` reports only the new destination's capacity loss. Attachments derived from an
incomplete input remain incomplete even when their new destination fits.
Input loss views are captured at construction; replacing the caller's status object does not
replace those views. Updating their GPU contents still changes the propagated flag per submission.
Island attachment captures all declared input/output views, including count destinations, at
construction. Preflight, shader bindings and the returned description use those same identities;
replacing caller port objects does not redirect writes or introduce foreign views into preflight.
An optional island-attachment `status.count` receives the retained CSR slot count on every
submission, including zero for an empty result.
Destination status fields other than `count`, `overflow`, `requiredCount`, `candidateOverflow`
and `uncertainCount` are refused; omit them and run the relevant diagnostic contributor afterward.

<a id="spatial-cross-validation-contract"></a>

`GPUSpatialCrossValidation` deals explicit blocks to folds in a seeded random order. Every block
that holds a finite unmasked row draws one Philox word keyed by the uint32 seed and the block ID;
the blocks are ranked by that word (ties by block ID) and the block of rank `r` goes to fold
`r % foldCount`. Fold sizes therefore differ by at most one block (not one row), no fold is empty
when there are at least `foldCount` such blocks, and a different seed gives a different order.
With fewer blocks than folds the empty folds remain visible in `status`. The fold of a block depends
on the seed and on the set of blocks that hold a usable row: masking rows without emptying a block
changes nothing, masking whole blocks re-deals the rest (ranking masked blocks too would leave folds
empty), so `GPUNestedInterpolationValidation` deals its inner folds anew for each outer fold. Every finite unmasked position
appears in exactly one test fold. A training point is excluded when its planar distance from any
held-out point is less than or equal to `bufferDistance`. Distance zero also excludes duplicate
positions in different blocks. Invalid coordinates enter neither training nor testing.
The baseline performs exact all-pairs checks and is not an indexed large-domain performance claim.
For `n` rows and `f` folds, `positions`, `blocks`, optional `mask` and `output.folds` each have
exactly `n` entries; training/testing are uint32 masks of exactly `f * n` entries in fold-major
order. `output.counts` has `3 * f` uint32 words (training/test/buffered-out counts per fold), and
`output.status` has `f` words (bit1 empty training, bit2 empty test). The block route accepts
1–65,536 rows and 2–64 folds; storage bindings may lower these bounds. Optional design routes
have the smaller [design-specific limits](./spatial-validation-advanced.md). `bufferDistance`
is a nonnegative distance in the positions' native planar units, at most `1e18`.

<a id="cross-validated-ols-contract"></a>

`GPUSpatialCrossValidatedOLS` composes those masks with a separate OLS fit for each fold. OLS
centering and all estimation statistics use training rows only. It predicts held-out rows using
that fold's centered slopes and mean correction, preserving precision at large predictor origins,
and reports `[scoredRows, missingTestRows, MSE, MAE, bias, R²]`.
Bias is prediction minus observation. Metric status has bit 1 for no scored rows, bit 2 for
missing outcomes/predictions, bit 4 for undefined R² with constant test outcomes, and
bit 8 for nonfinite metric arithmetic. Bit 8 suppresses MSE, MAE, bias and R² to NaN while
retaining scored/missing counts. Changing held-out responses cannot change that fold's
fit. With `n` rows, `f` folds and `p` supplied predictor columns, predictors are row-major
`n * p`, response has `n` values, coefficients have `f * (p + 1)` float32 values (intercept
first), predictions have `f * n` float32 values, and `fitStatus`/`metricStatus` each have `f`
uint32 words. `metrics` has `6 * f` float32 values. Limits combine the fold route above and
OLS's 1–15 predictor columns, subject to device bindings; fold masks use the source row domain.
Generalized GWR supplies a training-only `bandwidthSelection` graph as described above; [`GPUNestedInterpolationValidation`](./spatial-validation-advanced.md) supplies reusable inner-fold covariance/variogram and RBF bandwidth selection, outer refitting and held-out metrics using those masks.

## Calibrated spatial interaction

`GPUSpatialInteraction` fits a one-covariate Poisson mean proportional to
`exp(coefficient * covariate)` over an explicit finite coefficient grid or continuous bounded search. Supported constraints
are unconstrained total mass, origin totals, destination totals and both margins. The latter uses
bounded iterative proportional fitting. Default `calibration: 'grid'` retains a finite-grid optimum. `calibration: 'bounded'` takes exactly two ascending coefficient endpoints and uses golden-section profile optimization, with `maximumCalibrationIterations` (1–64, default 32) and `coefficientTolerance` (default `1e-5`). The profiled deviance is optimized over the explicit interval; optimum-boundary and f32 flat-objective behavior remain visible in the interval. Candidates with exponent spans above 80 are rejected explicitly.

Rows are explicit OD count cells. `pairState` distinguishes structural zero (0), observed (1) and
missing (2). Missing cells are excluded from both support and margins and predict NaN; structural
zeros must have observed flow zero and predict zero. Observed zero counts remain in the support.
Duplicate supplied OD rows are independent cells; aggregate duplicates explicitly when appropriate.
Zero observed margins remain zero. Dense routes support at most 4096 cells, 128 origins, 128
destinations and 128 coefficient candidates. The sparse doubly constrained route uses explicit
pair tables within its device and command budget; candidate predictions occupy transient storage.

A numerically successful profile does not establish coefficient identification. A constant
covariate is absorbed by the total-mass intercept; an origin-only covariate under origin
constraints, or a destination-only covariate under destination constraints, is absorbed by those
margins. Under both margins an additive `originTerm + destinationTerm` is also absorbed.
Multiple coefficients can then produce the same fitted flows and deviance. Report the support,
constraints and search interval, and do not interpret a selected grid endpoint as an identified
gravity effect or as coefficient uncertainty.

Optional `output.coefficientStatus` is one uint32 word, independent of prediction/fit success.
Decode its typed words or logical byte read with `decodeGPUSpatialInteractionCoefficientStatus`
(`Uint32Array`, `Uint8Array` or `ArrayBuffer`). Sampled codes 2/3 require finite-grid calibration
with at least two distinct coefficients and every candidate successful; bounded calibration
without a structural proof remains `undetermined`:

| Code | Evidence | Meaning and limit |
| --- | --- | --- |
| 0 | `undetermined` | No conclusive structural proof or usable sampled comparison; includes missing/failed fits or insufficient distinct sampled coefficients. |
| 1 | `structurally-absorbed` | Exact incidence constraints prove absorption by the intercept, one profiled margin, or jointly profiled origin/destination margins on the observed support, including disconnected components. |
| 2 | `sampled-flat` | Successful sampled coefficients have equal deviance within the comparison tolerance below; this does not prove a globally flat profile or structural absorption. |
| 3 | `sampled-varying` | The successful sampled profile varies; this does not prove unique/global identification, good conditioning or finite uncertainty. |

The decoder returns `code`, `evidence`, `structurallyAbsorbed`, `sampledFlatProfile` and
`sampledVaryingProfile`. A fit may have status 0 and coefficient status 1. Retain both words;
neither sampled evidence value is an identification certificate. Structural checks use exact
float32 dyadic covariates on observed support: signed zeros agree, while subnormal bit patterns
remain distinct. With both margins, each connected component is independently rooted and
the certificate checks that every observed cell has `covariate = originPotential + destinationPotential`.
Missing and structural-zero cells do not constrain these potentials.

Sampled comparisons use published float32 deviances with tolerance
`8 * 2^-23 * max(1, abs(first), abs(second))`. Coefficients must be distinct at evaluation
precision: float32 after rounding on GPU, float64 on CPU. A grid that collapses to one
evaluated coefficient is `undetermined` in the absence of a structural proof, even if its
supplied values differ. A structural absorption certificate takes precedence and still reports
code 1 for a single-point or collapsed grid.
GPU one-margin checks take linear work in the pair count after sorting. Joint checks use
`V = originCount + destinationCount` grouped propagation rounds, `O(V * (pairCount + V))` work
after grouping, including visits to declared vertices with no observed pairs in every round,
and `O(pairCount + V)` storage. Their `256 + 2*V` command allowance and exact-potential
scratch are checked together with the fit against the 32,768-command and device budgets;
use the CPU reference if this optional certificate cannot fit. CPU propagation takes linear
work in the pair count plus group count. These costs belong to the optional coefficient
evidence, independently of whether the prediction fit succeeds.
The shorter `O(pairCount * V)` work bound applies when `V = O(pairCount)`; declaring many
empty groups still incurs the vertex visits.

For example, four observed cells `(origin, destination) = (0,0), (0,1), (1,0), (1,1)` with
flows `[1,3,2,6]` and covariate `[1,1,7,7]` under origin constraints fit `[2,2,4,4]` at
either coefficient in `[0,1]` or its reversal `[1,0]`. Their Poisson deviance is
`2 * (log(0.5) + 3*log(1.5) + 2*log(0.5) + 6*log(1.5))`, published as `3.13948894`.
The coefficient is structurally absorbed despite prediction success; reversing the grid
does not turn the selected coefficient into identification evidence.


The selected summary is `[coefficient, deviance, maximumRelativeMarginError, iterations]`.
Each candidate exposes deviance, error, iterations and status. Unconverged candidates are never
selected as converged fits. If none succeeds, predictions are NaN with a failure status.
Optional origin/destination residuals are observed minus predicted margins. A successful fit is
conditional on the specified support and search domain; coefficient uncertainty is unavailable. Optional bounded-only `output.calibration` is `[lower, upper, width, converged]`. Status 4 retains the best finite fit but reports an exhausted coefficient search; inner IPF failure remains separate. The final two candidates are exposed in bounded mode. Direct spint GLM fixtures cover all four constraint modes with the same explicit support.

The CPU twin honors explicit `maximumIterations` and `maximumCalibrationIterations` budgets.
Dense and sparse GPU Poisson deviances use a cancellation-free near-fit series so a small
positive deviance does not become an exact-fit tie. When both bounded-search probes fail their
inner balancing budget, CPU and GPU preserve the original interval and explore its endpoints
and successive dyadic subdivisions using the remaining outer budget; a valid incumbent is
retained. The inner limit is unchanged. A feasible fit is required before the converged slot of
`output.calibration` can be 1, even if the interval is already narrow. An exact zero-deviance fit certifies the global
lower bound and closes the interval at that coefficient (CPU permits float64 roundoff at zero).
Otherwise unresolved portions remain visible in the interval and status 4 retains a feasible
exploratory fit when the search budget ends.
An unfinished balancing candidate has candidate code 1 and cannot win; if no candidate converges,
status is 3 and predictions, summary and marginal residuals are NaN. Exhausting only the coefficient
search gives status 4, retains the best final fit and bracket, and suppresses marginal
residuals. Without explicit budgets the ordinary CPU float64 solve continues to tolerance or numerical
stagnation, without the GPU iteration caps; exploratory recovery after failed inner fits defaults
to 32 outer rounds. The contributor and both operation forms snapshot
settings and coefficient arrays at construction.

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

[Inputs, outputs and limits](#iv-covariance-and-robust-spatial-2sls).

{/* lens:GPUInstrumentalVariablesCovariance */}

::lens{scenario=gpu-spatial-analysis.instrumental-variables-covariance rung=result caption="Inspect the estimates and explicit status outputs on synthetic inputs."}

::lens{scenario=gpu-spatial-analysis.instrumental-variables-covariance rung=graph caption="Inspect the executed GPU command graph."}

### `GPURestrictedRegimeRegression`

[Inputs, outputs and limits](#regimes-panels-and-sur).

{/* lens:GPURestrictedRegimeRegression */}

::lens{scenario=gpu-spatial-analysis.restricted-regime-regression rung=result caption="Inspect the estimates and explicit status outputs on synthetic inputs."}

::lens{scenario=gpu-spatial-analysis.restricted-regime-regression rung=graph caption="Inspect the executed GPU command graph."}

### `GPUSpatialPanelErrorMaximumLikelihood`

Tiled moments and reductions support 16,777,216 total rows, subject to device bindings. Panel period
count in unit mode is bounded by the total row count. Time/two-way GPU effects also require
`predictorCount + periodCount - 1 <= 15` for the profiled period contrasts.

{/* lens:GPUSpatialPanelErrorMaximumLikelihood */}

::lens{scenario=gpu-spatial-analysis.spatial-panel-error-maximum-likelihood rung=result caption="Inspect the estimates and explicit status outputs on synthetic inputs."}

::lens{scenario=gpu-spatial-analysis.spatial-panel-error-maximum-likelihood rung=graph caption="Inspect the executed GPU command graph."}

### `GPUSeeminglyUnrelatedRegression`

Tiled moments and reductions support 16,777,216 aligned rows, subject to device bindings.
The system supports 2–4 equations and at most 16 coefficients, including equation intercepts.

{/* lens:GPUSeeminglyUnrelatedRegression */}

::lens{scenario=gpu-spatial-analysis.seemingly-unrelated-regression rung=result caption="Inspect the estimates and explicit status outputs on synthetic inputs."}

::lens{scenario=gpu-spatial-analysis.seemingly-unrelated-regression rung=graph caption="Inspect the executed GPU command graph."}

{/* parameters: GPUSeeminglyUnrelatedRegression generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUSeeminglyUnrelatedRegression` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPUSeeminglyUnrelatedRegression`.

- Packer: `getGPUSeeminglyUnrelatedRegressionParameterValues(settings?)`, writes `GPU_SUR_PARAMETER_LENGTH` (1) `float32` values.

Settings of `getGPUSeeminglyUnrelatedRegressionParameterValues`:

| Field | Type | Description |
|---|---|---|
| `tolerance` (optional) | `number` | - |

{/* /parameters */}

### `GPUProbitRegression`

[Inputs, outputs and limits](#probit-and-local-collinearity).

{/* lens:GPUProbitRegression */}

::lens{scenario=gpu-spatial-analysis.probit-regression rung=result caption="Inspect the estimates and explicit status outputs on synthetic inputs."}

::lens{scenario=gpu-spatial-analysis.probit-regression rung=graph caption="Inspect the executed GPU command graph."}

### `GPULocalCollinearity`

[Inputs, outputs and limits](#local-collinearity-contract).

{/* lens:GPULocalCollinearity */}

::lens{scenario=gpu-spatial-analysis.local-collinearity rung=result caption="Inspect the estimates and explicit status outputs on synthetic inputs."}

::lens{scenario=gpu-spatial-analysis.local-collinearity rung=graph caption="Inspect the executed GPU command graph."}

### `GPUMultivariateLocalGearyPermutationTest`

See [runtime parameters and inference semantics](./operators-statistics.md#gpumultivariatelocalgearypermutationtest).

{/* lens:GPUMultivariateLocalGearyPermutationTest */}

::lens{scenario=gpu-spatial-analysis.multivariate-local-geary-permutation-test rung=result caption="Inspect the estimates and explicit status outputs on synthetic inputs."}

::lens{scenario=gpu-spatial-analysis.multivariate-local-geary-permutation-test rung=graph caption="Inspect the executed GPU command graph."}

### `GPURankMarkov`

Sorted ranks above 64 rows support up to 65,535 logical states, subject to dense transition-output
bindings. Use `GPURankMarkov.getExecutionPlan(props, device.limits)` before construction.
The [rank-dynamics contract](#joint-randomization-and-rank-dynamics) lists the diagnostic/status
outputs that retain the separate 64-state bound and the CPU dense-square limit.

{/* lens:GPURankMarkov */}

::lens{scenario=gpu-spatial-analysis.rank-markov rung=result caption="Inspect the estimates and explicit status outputs on synthetic inputs."}

::lens{scenario=gpu-spatial-analysis.rank-markov rung=graph caption="Inspect the executed GPU command graph."}

### `GPUFuzzyContiguity`

[Inputs, outputs and limits](#fuzzy-contiguity-contract).

{/* lens:GPUFuzzyContiguity */}

::lens{scenario=gpu-spatial-analysis.fuzzy-contiguity rung=result caption="Inspect the estimates and explicit status outputs on synthetic inputs."}

::lens{scenario=gpu-spatial-analysis.fuzzy-contiguity rung=graph caption="Inspect the executed GPU command graph."}

### `GPUIslandAttachment`

[Inputs, outputs and limits](#island-attachment-contract).

{/* lens:GPUIslandAttachment */}

::lens{scenario=gpu-spatial-analysis.island-attachment rung=result caption="Inspect the estimates and explicit status outputs on synthetic inputs."}

::lens{scenario=gpu-spatial-analysis.island-attachment rung=graph caption="Inspect the executed GPU command graph."}

### `GPUSpatialCrossValidation`

[Inputs, outputs and limits](#spatial-cross-validation-contract).

{/* lens:GPUSpatialCrossValidation */}

::lens{scenario=gpu-spatial-analysis.spatial-cross-validation rung=result caption="Inspect the estimates and explicit status outputs on synthetic inputs."}

::lens{scenario=gpu-spatial-analysis.spatial-cross-validation rung=graph caption="Inspect the executed GPU command graph."}

### `GPUSpatialCrossValidatedOLS`

[Inputs, outputs and limits](#cross-validated-ols-contract).

{/* lens:GPUSpatialCrossValidatedOLS */}

::lens{scenario=gpu-spatial-analysis.spatial-cross-validated-ols rung=result caption="Inspect the estimates and explicit status outputs on synthetic inputs."}

::lens{scenario=gpu-spatial-analysis.spatial-cross-validated-ols rung=graph caption="Inspect the executed GPU command graph."}

### `GPUSpatialInteraction`

Dense routes retain 4,096 pairs and 128 groups. Sparse doubly constrained balancing accepts explicit
pair tables, subject to device bindings and the command-node budget; incomplete balancing is
reported.

{/* lens:GPUSpatialInteraction */}

::lens{scenario=gpu-spatial-analysis.spatial-interaction rung=result caption="Inspect the estimates and explicit status outputs on synthetic inputs."}

::lens{scenario=gpu-spatial-analysis.spatial-interaction rung=graph caption="Inspect the executed GPU command graph."}

## Recorded reference and scaling run

The separate `statistics-blockcv.R` generator actually executes blockCV 4.0.1 `cv_spatial` with
explicit user polygons/predefined folds and `cv_buffer` with projected distances. The GPU design
is compared with both as a partition: every GPU fold must be one blockCV fold, with the same test
rows and the same training rows, whatever number the fold carries. Blocks are dealt to folds in a
seeded Philox order (see above), which is our explicit deterministic assignment rule;
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

### `GPUSpatialPanelLagMaximumLikelihood`

`GPUSpatialPanelLagMaximumLikelihood`: Auto selects tiled unit effects when entity eigenvalues are
supplied. Tiled fits support 1,024 entities and 16,777,216 total rows, subject to device bindings;
reference fits retain 32 entities and 32 periods.

`GPUSpatialPanelLagMaximumLikelihood` fits `y = rho Wy + X beta + unitEffect + error`
on complete balanced panels in period-major order. The random-effects fit alternates analytic
variance-component shrinkage `phi` with an exact dense log-determinant profile search for `rho`;
unit effects use within-entity demeaning. The reference capacity is 2–32 entities, 2–32 periods and 1–7 predictors. With
`fitMode` selecting tiled unit effects and a supplied entity spectrum, the bound is 1,024 entities
and 16,777,216 total rows, subject to device bindings. Weights must be nonnegative, row-standardized, zero-diagonal CSR with sorted
unique neighbors and no islands. Missing observations invalidate the complete fit.

Both GPU routes and the CPU twin refuse incomplete weights: nonzero
`weights.status.overflow[0]` (final capacity loss) or
`weights.status.candidateOverflow[0]` (candidate loss) returns status 1 with NaN coefficients,
covariance, summary, fitted values and residuals. Clearing the input flags permits a subsequent
submission to fit normally; the two input flags retain their distinct meanings.
On both GPU routes and the CPU twin, the accessible retained CSR span is
`offsets[entityCount] - offsets[0]`; unused leading slots do not count as retained edges.
An available `requiredCount` must fit both edge capacity and this span, even when `count`
is supplied; a supplied `count` must equal that span. For offsets `[2,4,6,8]`, the span is
6: `count = requiredCount = 6` passes the count checks, while both set to 8 fail despite the
terminal offset being 8. Spare capacity cannot certify edges absent from the accessible CSR.
An over-capacity `count` also fails. These count
checks apply even when no overflow scalar is supplied and suppress dependent Hausman results.
Both count views must belong to the encoding graph and be disjoint from every fit and Hausman
output, as required for the overflow inputs.

`CPUSpatialPanelLagMaximumLikelihoodOperation` uses the same `fitMode` and `eigenvalues` ports.
Auto selects the supplied spectrum only for unit effects; `reference` rejects a spectrum and
`tiled` requires one. Supply exactly one eigenvalue per entity: a length mismatch throws, while
a nonfinite value or an absolute value above float32(1.00001) returns status 1 and NaN outputs.
The CPU objective and determinant derivatives use the supplied spectrum; the caller must ensure
it belongs to the entity weights matrix. The pure `computeSpatialPanelLagMaximumLikelihoodOnCPU`
accepts the spectrum as `input.eigenvalues` and the mode as `options.fitMode`; with no spectrum
it evaluates the dense CSR determinant. GPU-specific size caps do not limit the CPU fit.
Explicit `maximumSearchIterations` and `maximumIterations` budgets bind the CPU fit: exhausting
a search budget with a rho bracket wider than `tolerance`, or an outer budget before convergence,
returns status 3 with the current coefficients, fitted values and residuals; covariance is withheld.
One golden-section step on `[-0.999, 0.999]` leaves a bracket about 1.23483 wide. When budgets are
explicit and the bracket meets tolerance, the CPU includes a retained interval endpoint and
applies the one-sided optimality test there. A converged constrained maximum is not budget
exhaustion. When budgets are
omitted, the CPU search continues to binary64 resolution and alternation continues to convergence
or a repeated state. Residual-square sums use compensated float64 accumulation so repeating
periods preserves the estimate and scales covariance by the inverse repetition count.

The contributor and both operation forms capture declared fit options, spectrum presence and
graph port/status maps by property read at construction, including inherited or non-enumerable
getters. Each getter is read once; changing its return value later cannot remove a loss flag.
Mutating the original props cannot change the declared spectrum dependency; the spectrum's
values are read anew on each execution. A graph spectrum-length error starts with the configured
operation ID; the standalone CPU function uses its own function name.

The fit runs on centered data: the pooled mean of each predictor and of the response is removed
before quasi-demeaning, so a predictor with a large mean (a year, an elevation, an income) costs
no precision. Only the intercept depends on the shift; it is mapped back as
`intercept + (1 - rho) mean(y) - slopes' mean(x)` and its covariance row and column are
transformed to match. Normal and information matrices are factored after diagonal equilibration,
and a Cholesky pivot of the unit-diagonal matrix below `1e-3` (the limit of
`GPUOrdinaryLeastSquares`) is status 2.
Unit effects factor the within design directly. The pooled initialization is used only by
random effects to initialize `phi`, in both GPU and CPU forms; collinear entity offsets cannot invalidate an identifiable
within design through that unused pooled solve.

An exact zero-RSS minimum at an admissible `rho` has no finite Gaussian maximum: a
nearby golden-section point with positive rounding-sized variance is not a successful fit.
Both GPU routes and both operation forms also run a sufficient original-input certificate.
It checks temporal differences against `X beta + rho W y` with exact integer products,
using 24 rounded slope/rho candidates from the fit.
For a single predictor, exact cross-products compare temporal differences of `y-rho Wy` with
those of `x` at each candidate rho (and at zero), independently of the fitted slope. A
non-dyadic slope such as `1/3` is detected both without and with spatial dependence.
A certified zero within variance returns status 2 with every floating output NaN;
dependent Hausman outputs are status 1/all NaN.
There is no variance threshold. The GPU certificate supports normal/zero operands with
at most a 168-bit span between triple-product exponents; unsupported operands yield no
certificate. Single-column cross-products use four factors and a 140-bit exponent-span
envelope; the separate original three-factor witnesses retain their 168-bit envelope.
The CPU uses BigInt and retains subnormals and larger spans. This is a
sufficient degeneracy check, not a complete symbolic rank test. The independent
`panel-lag-poles.oracle.py` uses Fraction arithmetic on 8-row copy, spatial-lag and weighted
fixtures, including a perturbation that must not be certified and affine copies.
Separately, if squared moments underflow to zero in float32, the GPU can reject before
publishing any floating diagnostic (status 2, all NaN). The CPU can retain a finite float64
likelihood while its variance rounds to zero on publication. This numerical refusal does
not certify a mathematical zero-variance pole; failed-fit diagnostics can differ by precision.

The concentrated variance is formed directly. With `e0`, `e1` the residuals of the quasi-demeaned
response and of its lag on the predictors, `rho0 = e0'e1 / e1'e1` and `r = e0 - rho0 e1` summed
per row, `u'u = S + e1'e1 (rho - rho*)^2`, two non-negative terms; the search and the published
`sigma2`, log-likelihood and covariance all use it. The former `e0'e0 - 2 rho e0'e1 + rho^2 e1'e1`
of three float32 sums is rounding once the model fits well. Measured on the test GPU against spreg
1.9.1 `Panel_FE_Lag` and `Panel_RE_Lag` (16 entities, 16 periods, `rho` 0.4, residual deviation
0.004): with `u'u` at 3e-5 of `e0'e0` the published `sigma2` was 1% to 2% off and is now within
1e-5; at 3e-7, 36% low and 96% high, now within 6.3e-5; at 3e-9 (eight scenes, both effects) it
was 10 to 241 times too large with status 0 in 9 of the 16 fits, negative with status 2 in one
and status 3 in six, and is now within 1.7e-3 with status 0 in all 16, where the float32 rounding
of the fitted values (0.6% of a residual per row) is what remains. The lag of the response is summed from differences rounded as written, with the
row sums of the float32 weights carried exactly.

During the `rho` search, both GPU routes remove the rho-independent log-variance constant and
evaluate small relative variance increments with a series. Otherwise float32 can round nearby
variances to the same value, so a narrow search bracket can still surround a biased estimate.
The published log-likelihood retains the full log variance and its existing constants.
After eight coarse golden steps, both GPU routes guide the remaining golden brackets by
the analytic derivative
`n (rho-rho*) / (S/(e1'e1)+(rho-rho*)^2) + T tr(W (I-rho W)^-1)`.
The reference route computes the trace with LU column solves; tiled unit effects use the
supplied spectrum. This avoids differences of rounded log determinants. The same explicit
step budget and bracket geometry apply; this is still a local bounded profile method.
CPU float64 profiling and GPU profiling can take different numbers of outer phi sweeps;
there is no one-sweep bound on their difference.

Entity effects survive the pooled centering. Each column is stored as its within deviation
(centered value less its entity mean) and its entity mean, and the quasi-demeaned value
`value - (1 - phi) mean` is formed as `deviation + phi mean`, whose terms do not cancel; the
normal equations of each step are the polynomial in `phi` of three moment sums taken once.
Both routes form within deviations with the shared split-center transform: scaled two-part
temporal sums preserve a large common level and isolated extreme observations, and the high
and low mean parts are subtracted separately. A time-invariant column therefore has exactly
zero deviation, including a non-power-of-two period count. A first-observation reference is
not used: with temporal values `[2^25,-2^25,1]` it would erase the final within value `2/3`.
The pooled-centered entity mean is carried separately for random effects.
The reference route reduces these moments in a fixed pairwise order; tiled unit effects use
the same order inside each tile and when merging tiles. Serial row sums could bias the slope
after rho was accurate by losing small predictor squares following larger ones. The 32-row
integer moment oracle in `panel-lag-pack` covers this loss in unit, random and tiled routes.
Formed as written, `(value - mean) - fitted` was regrouped by the shader compiler of the test GPU
as `value - (mean + fitted)`: with entity effects of +3000 and -3000 (16 entities, 16 periods,
unit effects, residual deviation 0.004) all 256 published residuals were multiples of the float32
step of 3000 and `rho` was 0.026 standard errors from spreg `Panel_FE_Lag`; an earlier fix
reached 0.002. A later Metal regression exposed a separate lag-packing loss (0.012 standard
errors): forming `Wy` on large entity levels before demeaning erased small spatial variation.
The reference pack now forms `W` times the within response, then stores lagged entity means
separately. This uses the exact commutation of temporal demeaning with fixed entity weights.
The 8-row `panel-lag-pack` oracle covers every fit output and exact half-step lag preservation,
with shifted/rescaled copies. Float32 entity means can still leave constant residual offsets;
the pack no longer rounds the within lag to the step of the large entity effect.

`effects` defaults to `random`. `maximumIterations` (1–128, default 64) bounds outer sweeps;
`maximumSearchIterations` (1–64, default 32) bounds each golden-section search. `tolerance`
(default 1e-5) controls phi change and final rho bracket width; the change of phi is held against
it as an absolute change for a phi of 0.01 or more and in proportion to `phi / 0.01` below. Rho is
restricted to [-.999,.999], and random phi to [1e-8,1]. Boundary summary bit 1 flags rho near
either endpoint; bit 2 flags random phi at either endpoint; bit 4 (with bit 2 and status 3) flags
a phi whose estimate is below 1e-8 and was held there: the published numbers are then those of
the constrained problem, not the maximum, and there is no covariance. Unit phi is exactly zero
and sets neither bit. The lower limit was 1e-5 and was reported as status 0 with bit 2 only:
entity effects of +3000 and -3000 with residuals of 0.004 (spreg `Panel_RE_Lag` phi 2.5e-7) came
out with `sigma2` 105 times spreg's and `rho` 3.5 standard errors off; that fit is now status 0
with phi within 0.2% of spreg, `rho` within 0.009 standard errors, the intercept within 0.2 and
`sigma2` within 1.1% (the float32 entity means). float32 data cannot carry residuals much below
1e-6 of an entity effect, so with 16 periods phi stays above about 1e-8 on data the kernel can
resolve (effects of 16,000,000 with residuals of 0.5: spreg phi 9.1e-9, status 3).

`fitted` is the quasi-demeaned response at the published phi less the residual
(`fitted + residuals = y - (1 - phi) mean_i(y)`). A phi inside the tolerance of the alternation
but `d` away from spreg's moves `fitted` by `d` times the entity mean of the response and leaves
`residuals` alone: with a response mean of 10,000 and residuals of 0.004, a phi within 4.5e-7 of
spreg's gives fitted values 1.2 residual deviations from spreg's.

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
On a singular conditional design, status 2 can retain initial or pooled coefficients and
summary values as failure diagnostics, while covariance, fitted values and residuals are NaN.
Those diagnostic values are not ML estimates and can differ between CPU and GPU initialization.
The all-floating-outputs-NaN requirement above applies to certified exact poles, not every
status-2 path. Check fit status before interpreting coefficients or variance.

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
