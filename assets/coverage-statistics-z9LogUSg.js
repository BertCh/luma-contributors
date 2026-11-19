var e={slug:`api/gpu-spatial-analysis/coverage-statistics`,title:`Bounded statistical estimation and validation`,html:`<h1 id="bounded-statistical-estimation-and-validation">Bounded statistical estimation and validation</h1>
<p>These contributors extend the existing spatial statistics and regression graph with explicit
numerical, missingness and convergence contracts. All computation is recorded in the caller's
GPU graph. They neither submit work nor read buffers back, and have no CPU fallback. Inputs and
outputs are caller-owned packed views; the graph owns transient workspace. Each generated kernel
stays within the default eight-storage-buffer limit.</p>
<p>The supported modes below are <strong>bounded subsets</strong> of the ST01–ST11 backlog. They do not establish
complete PySAL, mgwr, blockCV or spint parity. Exact capacity counts, converged numerical estimates,
Monte Carlo inference, exact leave-one-out model selection and bounded continuous profile optimization are different guarantees.</p>
<table>
<thead>
<tr>
<th>Backlog</th>
<th>API</th>
<th>Delivered contract</th>
<th>Remaining scope</th>
</tr>
</thead>
<tbody>
<tr>
<td>ST01</td>
<td><code>GPUInstrumentalVariablesCovariance</code>; <code>GPUSpatialTwoStageLeastSquares.robust</code></td>
<td>HC0/HC1/HC2/HC3, one-way clustered CR1 and spatial-HAC IV covariance; robust SE/z/p table and returned-covariance PSD certification</td>
<td>Entire HAC-kernel PSD certification and multi-way clustering</td>
</tr>
<tr>
<td>ST02</td>
<td><code>GPURestrictedRegimeRegression</code></td>
<td>Any common/varying intercept and slopes; full restricted covariance; joint nested-model Chow F/degrees and coefficient/joint Wald chi-square tails</td>
<td>spatial/heteroskedastic restriction tests</td>
</tr>
<tr>
<td>ST03</td>
<td><code>GPUSpatialPanelErrorMaximumLikelihood</code></td>
<td>Balanced unit fixed-effects spatial-error Gaussian ML and pooled LM error and robust LM lag/error tests</td>
<td>Random/time effects, Hausman tests, clustered/serial inference</td>
</tr>
<tr>
<td>ST04</td>
<td><code>GPUSeeminglyUnrelatedRegression</code></td>
<td>One-step feasible SUR and Kelejian–Prucha spatial-error SUR, joint error covariance and full system coefficient covariance</td>
<td>iterated SUR and small-sample inference</td>
</tr>
<tr>
<td>ST05</td>
<td><code>GPUProbitRegression</code></td>
<td>Binary probit, expected-information standard errors, average marginal effects, tail-accurate likelihood with separation reported as a status; Pinkse, Kelejian–Prucha and Pinkse–Slade residual tests</td>
<td>Latent spatial probit</td>
</tr>
<tr>
<td>ST06</td>
<td><code>GPULocalCollinearity</code>; <code>GPUGeneralizedGeographicallyWeightedRegression</code>; <code>GPUMultiscaleGeographicallyWeightedRegression</code></td>
<td>Weighted collinearity diagnostics; exact GLM LOO bandwidth search; MGWR finite-grid/golden-section LOO or AICc selection and bandwidth-conditioned backfit inference</td>
<td>MGWR adaptive-neighbor selection and uncertainty from bandwidth reselection</td>
</tr>
<tr>
<td>ST07</td>
<td><code>GPUMultivariateLocalGearyPermutationTest</code></td>
<td>Joint-vector conditional randomization, replay, equal-tail or esda minimum-tail p-values and optional Bonferroni adjustment</td>
<td>Exact upstream classification-label parity and additional correction families</td>
</tr>
<tr>
<td>ST08</td>
<td><code>GPURankMarkov</code></td>
<td>Full-rank/geographic-rank transitions, common complete panels and absolute-rank mobility</td>
<td>More than 64 rank states, fractional ties and incomplete-panel interpolation</td>
</tr>
<tr>
<td>ST09</td>
<td><code>GPUFuzzyContiguity</code>, <code>GPUIslandAttachment</code></td>
<td>Planar filled-polygon separation and nearest original-island attachment, with added-edge provenance</td>
<td>Geographic coordinates, topology repair and GEOS robust-predicate parity</td>
</tr>
<tr>
<td>ST10</td>
<td><code>GPUSpatialCrossValidation</code>, <code>GPUSpatialCrossValidatedOLS</code>, <code>GPUNestedInterpolationValidation</code></td>
<td>Block/group/environmental folds, buffers, bounded NNDM/kNNDM, training-only OLS and nested covariance/RBF selection with held-out metrics</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/spatial-validation-advanced">Explicit design, optimizer and capacity contracts</a></td>
</tr>
<tr>
<td>ST11</td>
<td><code>GPUSpatialInteraction</code></td>
<td>Four constraint modes with grid or continuous bounded profile calibration, IPF, deviance and marginal residuals</td>
<td>multiple covariates, coefficient uncertainty</td>
</tr>
</tbody>
</table>
<h2 id="iv-covariance-and-robust-spatial-2sls">IV covariance and robust spatial 2SLS</h2>
<p><code>GPUInstrumentalVariablesCovariance</code> takes structural design <code>Z</code>, instrument design <code>H</code> and
structural residuals from the same fitted IV model. Both designs omit the implicit intercept.
The coefficient count is 1–15; the instrument count is between the coefficient count and 16;
there are at most 2^24 observations (float32 counts), and fewer when the design, instrument or
score rows exceed the device's storage binding size. All estimation rows must be finite. There is
no silent complete-case change after a model has already been fitted.</p>
<p>Every row sum (column means, the centered moments <code>H'H</code>, <code>H'Z</code>, <code>diag(Z'Z)</code> and the meat matrix)
is a row-tile reduction merged in a fixed pairwise order, so results are reproducible; one
invocation solves the small projection system. The HAC kernel is applied over its CSR neighbors,
one invocation per row (<code>O(nnz · p)</code>), never over all pairs.</p>
<p>The covariance uses the projected-design influence matrix, not an OLS sandwich applied to
endogenous regressors. Columns are centered and norm-scaled before both Cholesky solves. The
intercept influence is recovered with its cross terms, so the output includes the full
<code>(coefficientCount + 1)²</code> covariance in intercept-first row-major order.</p>
<ul>
<li><code>HC0</code> uses squared residual scores without a finite-sample multiplier.</li>
<li><code>HC1</code> multiplies HC0 by <code>n / (n - p)</code>, where <code>p</code> includes the intercept.</li>
<li><code>HC2</code> and <code>HC3</code> divide residuals by <code>sqrt(1-h)</code> and <code>1-h</code>, respectively, using projected-design
leverage including the intercept. A denominator at most 1e-6 or leverage below -1e-6 is invalid.</li>
<li><code>CLUSTER</code> requires dense <code>uint32</code> <code>clusters</code> and <code>clusterCount</code> capacity; only occupied groups
count in the CR1 multiplier <code>G/(G-1) * (n-1)/(n-p)</code>. At least two groups must be occupied.</li>
<li><code>HAC</code> takes a separate spatial kernel, without a finite-sample multiplier. It must be symmetric,
nonnegative, positive semidefinite and have unit diagonal. Bounds, finite weights, symmetry
and diagonal are checked on GPU. Positive semidefiniteness is a caller precondition.</li>
</ul>
<p>The returned covariance is certified PSD to relative tolerance 1e-5 with a shifted Cholesky
test; zero and rank-deficient covariance are accepted. This does not certify the entire HAC kernel.
Covariance status is <code>0</code> success, <code>1</code> singular projection, <code>2</code> insufficient observations,
<code>3</code> nonfinite inputs/results, <code>4</code> invalid HAC kernel, <code>5</code> invalid leverage, <code>6</code> invalid clusters
and <code>7</code> a failed PSD certificate. Failed covariance entries are NaN.</p>
<p>For spatial 2SLS, set <code>robust: {method, kernel?, clusters?, clusterCount?, output: {covariance, status}}</code>. Existing point
estimates and classical summary diagnostics remain unchanged; the coefficient table's SE, z and
p-value columns use the robust covariance. The robust status is separate from the estimator's
status. A robust covariance does not change the classical Anselin–Kelejian diagnostic's assumptions.</p>
<h2 id="regimes-panels-and-sur">Regimes, panels and SUR</h2>
<p>SUR, including spatial SUR, and panel-error fits support 16,777,216 total rows with tiled
reductions; panel periods cannot exceed the total row count. Device bindings may impose lower limits.</p>
<p><code>GPURestrictedRegimeRegression.common</code> includes the intercept followed by each predictor.
A true entry estimates one shared coefficient; a false entry estimates one coefficient per regime.
Output order is source-column-major, with regime order inside each varying column.
The unrestricted model is limited to 16 coefficients; its complete cases are exactly those of the
restricted model. Covariance uses the restricted model's pooled <code>SSE / (n - p)</code> estimate.
The Chow output contains <code>[F, numeratorDegrees, denominatorDegrees]</code>. Optional <code>stability</code> contains <code>[Wald chi-square, degrees, upper-tail p]</code> for each source coefficient and then jointly. It tests equality of the varying regime coefficients, matching <code>spreg.Chow</code>; common columns are <code>[NaN, 0, NaN]</code>. This is a different hypothesis from the nested-model F test.
Empty or singular regimes are explicit, and a singular unrestricted fit does not erase a valid
restricted coefficient estimate. The joint test assumes independent homoskedastic normal errors.
Rows (at most 2^24, and within the device's binding size for the predictors) are reduced in
parallel: per-regime means and centered moments come from a stable regime sort and one workgroup
per regime, one invocation solves both nested systems, and the residual sums of squares are
row-tile sums merged in a fixed pairwise order.</p>
<p>Both systems are solved in a shifted basis, never from raw cross-products, so a predictor with a
large mean (a year, an elevation, a projected northing) costs no precision. With a varying
intercept the predictors and the response are centered on their regime means; with a common
intercept the response and the common-slope predictors are centered on their pooled means. A
varying slope under a common intercept is not invariant to a shift, so when such a predictor has a
mean larger than its spread the fit changes basis instead: <code>x - mean</code> takes the place of the
constant and the raw columns <code>x 1[g]</code> stay (<code>1 = (sum_g x 1[g] - (x - mean)) / mean</code>), which spans
the same model with a conditioning that does not depend on the size of the mean. The published
coefficients and covariance are those of the raw design. Regime means are float32, so the sums of
the deviations from them are carried through the moments and every difference of a value and its
shift is fixed before it meets a small term (the shader compiler of the test GPU otherwise
regroups <code>(y - mean) - fit</code> as <code>y - (mean + fit)</code>): a response whose mean is far above its
residuals keeps its Chow statistic. The Chow numerator is the direct sum of
squared differences of the two fits (equal to <code>SSE_restricted - SSE_unrestricted</code>), and each Wald
contrast is formed from differences of shifts before it meets the coefficients. A system is
singular (status 2 restricted, 3 unrestricted) when a Cholesky pivot of the unit-diagonal normal
matrix is below <code>1e-3</code>, the limit of <code>GPUOrdinaryLeastSquares</code>; <code>GPUSpatialRegimeRegression</code>,
<code>GPUSeeminglyUnrelatedRegression</code>, <code>GPUProbitRegression</code> and
<code>GPUSpatialPanelLagMaximumLikelihood</code> apply the same limit.</p>
<p><code>GPUSpatialPanelErrorMaximumLikelihood</code> takes the existing spatial-error ML arguments plus
<code>periodCount</code>. Data are balanced and period-major; the supplied weights describe only the entity
axis. Unit means are removed on GPU, and <code>I_T ⊗ W</code> supplies the repeated spatial determinant.
The objective and variance divisor use <code>N*T</code>, matching the profiled fixed-effects Gaussian ML
convention. This is not REML or a short-panel bias correction. It inherits the existing bounded
trace-series search, boundary and approximation diagnostics. The absorbed intercept table row and
information criteria are NaN. Predictors constant through time are singular after demeaning.
Optional <code>diagnostics.tests</code> returns statistic/p-value pairs for pooled-panel LM error, robust LM lag and robust LM error. These fit pooled OLS on the original panel and use its residual variance, spatial lags and projected spatial fitted values, matching spreg 1.8.3. That variance is <code>RSS / (n - p)</code> (spreg <code>ols.sig2</code>, <code>n = N*T</code>, <code>p = predictorCount + 1</code>), where the cross-section LM tests of <code>GPUSpatialRegressionDiagnostics</code> use <code>RSS / n</code> (spreg <code>LMtests</code>): on the same residuals and block-diagonal weights a panel LM error is <code>((n - p) / n)^2</code> times the cross-section one, so compare each with its own reference and not with the other. Diagnostic status is 0 valid, 1 failed/incomplete pooled fit, or 2 degenerate/nonfinite test information. The robust tests require positive spatial fitted-value information beyond the weight trace. They are specification tests, not clustered uncertainty for the FE estimator.</p>
<p><code>GPUSeeminglyUnrelatedRegression</code> requires 2–4 equations with the same predictor count but
potentially different predictor values. Data are equation-major, then row-major; observation IDs
must be aligned explicitly. Any nonfinite value in any equation excludes that observation from all
equations. The total number of coefficients, including each intercept, is at most 16.
Equation OLS residuals estimate a <strong>joint</strong> cross-equation covariance with divisor <code>n</code>, followed
by one full feasible-GLS solve. The output covariance is the conditional GLS information inverse
using that first-stage error covariance. Singular design and singular error covariance have
separate status codes. This is one-step FGLS, not an iterated likelihood fit.</p>
<p>Optional <code>spatial</code> selects the <code>spreg.SURerrorGM</code> convention: equation OLS residuals estimate Kelejian–Prucha spatial-error parameters; filtering transforms responses, designs and original OLS residuals before the joint GLS solve. The intercept and full covariance are transformed back to original units, and reported residuals are structural residuals. This selected mode requires complete aligned rows and nonnegative row-standardized zero-diagonal weights without islands; it rejects a mask or caller-supplied <code>initialResiduals</code>. The shared weights are checked once with bounded row tiles, and the resulting admissibility flag applies to every equation. Each equation has its own parameter and GM status; status 4 denotes inadmissible weights. Nonspatial <code>initialResiduals</code> allows an explicit first-stage residual covariance, with the same common complete-case mask.</p>
<h2 id="probit-and-local-collinearity">Probit and local collinearity</h2>
<p><code>GPUProbitRegression</code> uses centered/scaled predictors, Fisher scoring and monotone likelihood
backtracking. It accepts 1–11 predictors and at most 2^24 rows (fewer when the predictors, or
the diagnostic residual rows, exceed the device's binding size). Each iteration is a row-tile
pass for the score and information, a one-invocation solve, one row-tile pass that evaluates all
24 halved line-search steps, and a one-invocation step choice; iterations are unrolled
(<code>estimatedCommandNodeCount</code>, about <code>6 · maximumIterations</code>) and gated on the GPU once the fit
stops. Complete cases are selected
jointly; included responses must be exactly 0 or 1. The expected-information SEs are model based.
Marginal effects average the normal density over estimation rows, then multiply by each slope;
they are not effects evaluated at the means.</p>
<p>Its summary is <code>[completeRows, logLikelihood, deviance, iterations]</code>. Status <code>0</code> means converged,
<code>1</code> insufficient data, <code>2</code> singular information, <code>3</code> not converged and <code>4</code> nonbinary data or
separated data. Three invariants hold:</p>
<ul>
<li>Status 0 is a maximum.</li>
<li>Status 4 for binary data means separated, and only with a certificate: a direction of the
coefficients that provably moves no row towards the wrong side and some row towards the right
side. Data that are not separated are never status 4.</li>
<li>Everything else that is not status 1 or 2 is status 3, with the last iterate written (and NaN
standard errors): iterations or line-search steps exhausted, a step pinned by the bound on
misclassified rows, a linear predictor that overflowed, or a small step on data that float32
could not tell from separated. An ordinary well-conditioned fit is never status 3 at the
default settings. Separated data whose certificate the search does not find are status 3 (or 2,
when the information matrix of the runaway fit loses rank first), never 0.</li>
</ul>
<p>Status 3 writes the iterate however large its linear predictors are: 2,001 rows that converge in 7
steps, stopped after 3 with a correctly classified row beyond a linear predictor of 16, are status
3 with their iterate. Every other failure writes NaN coefficients. All-zero/all-one outcomes are
status 4.</p>
<p><code>fitted</code>, <code>marginalEffects</code> and the diagnostic residuals are computed from the centered, scaled
design of the fit and its estimate, not from the published raw coefficients: the raw intercept is
<code>a - sum mean_j b_j</code> in float32, which for a predictor near 2^24 with a spread of five float32
steps is rounded to a step of 0.5. Rebuilt from the raw coefficients such a fit had fitted
probabilities of 0.185, 0.421, 0.691 where statsmodels has 0.136, 0.345, 0.618, and a marginal
effect 2.7% low; they now agree to 1e-7. The slopes were right throughout.</p>
<p>The likelihood of a row is <code>log Phi(s eta)</code> (<code>s = +1</code> for a response of 1, <code>-1</code> for 0), with <code>Phi</code>
evaluated through <code>erfc</code> so that its lower tail keeps relative accuracy down to the smallest normal
f32; the score and the information weight are ratios of the density to the two tail probabilities,
and no <code>1 - p</code> is formed. A row whose probability is within float32 rounding of 0 or 1 therefore
still pulls the fit as the likelihood requires. (<code>fitted</code> and the spatial diagnostics use
probabilities floored at <code>1e-7</code>.)</p>
<ul>
<li>A row on the right side of the fit is not bounded: once its probability rounds to 1 it adds
nothing to the likelihood. 2,000 rows from a latent index <code>0.2 + 2x</code> plus one correctly classified
row at <code>x = 9</code> (a linear predictor of 18.9 at the maximum) return the maximum of the other rows,
slope 2.072, with status 0, as statsmodels <code>Probit</code> does. So does a fit whose maximum has a largest
linear predictor of 12.06.</li>
<li>A row on the wrong side leaves the float32 range of <code>Phi</code> near 13, so the line search accepts no
step that takes such a row beyond ±12. A fit is pinned when that bound, not the likelihood,
shortened the accepted step and the shortened step is below <code>tolerance</code> while the full Fisher
step is not: status 3 with the pinned iterate (it was status 4 with NaN coefficients; the data
are not separated). The same 2,000 rows plus a response of 0 at <code>x = 9</code> are such a fit (the
float64 maximum is a slope of 1.447 with that row at 13.2; the iterate has a slope of 1.314 with
that row at 12); a response of 0 at <code>x = 4</code> is not (slope 1.862, that row at 7.7, returned with
status 0).</li>
<li>Separated data have no finite maximum: some direction of the coefficients moves every row to the
right side or leaves it where it is. The fit certifies separation by exhibiting such a direction,
not by the size of a linear predictor, and a direction counts only when the sign of every row's
move is known. There is no tolerance: a move towards the wrong side disqualifies a direction
however small it is, and a row whose sign float32 cannot decide is not counted as standing still.
Three kinds of direction are tested, each reduced over the rows with maxima of flags, which are
exact whatever the row count:
<ul>
<li>A column that separates at its lowest or highest value: the rows above the lowest value of a
column (or below its highest) all have the same response. Raw values are compared, which is
exact, and the test does not depend on the fit. An indicator of a group of rows that all have
a response of 1 is certified this way at any <code>maximumIterations</code> and any <code>tolerance</code>: one or
three rows of 600, an indicator among eleven predictors, every second row, one row of
100,000, three of 200,000, five of 400,000, five and twenty of 1,000,000.</li>
<li>A single-column threshold from the Fisher direction. With its rounding taken out (a column
whose largest contribution to any row is below 1e-4 of the largest contribution of any column
is dropped; the intercept is set so that the rows at an end of the kept columns do not move
when their move is below that level), a direction with one column left is
<code>sign · (x - value)</code>. Rows are compared with <code>value</code>, and in the next pass with the two data
values next to it, so a column that separates at an interior value where both responses occur
is certified one iteration after the direction points there: quasi-complete separation, and a
three-valued predictor whose rows with both responses are at its middle value (100,002 rows;
status 3 at the defaults before, status 2 with 128 iterations and status 0 with a tolerance
of 1e-3).</li>
<li>The Fisher direction itself, for separation by a combination of columns. The move of a row
takes its sign from the float32 value only when that is beyond the evaluation error bound,
<code>(predictorCount + 5) 2^-23</code> of the sum of the magnitudes of the row's terms; a row all of
whose terms are exactly zero stands still; any other row is undecided and the direction is
not a certificate. Complete separation by two predictors is certified this way. Quasi-complete
separation by a combination of columns, whose tied rows have moves that cancel to rounding, is
not: 606 such rows ended as status 2 at the defaults and status 3 with 8 or 16 iterations.
A certificate is kept once found, and a certified fit is status 4 however it ends: iterations
exhausted, a singular information matrix (without a certificate that is status 2), a failed line
search, an overflowing linear predictor, or a step below <code>tolerance</code>. statsmodels <code>Probit</code> does
not converge on any of these scenes, and a linear program over directions (SciPy <code>linprog</code>)
finds a separating direction on each.</li>
</ul>
</li>
<li>What the certificates need. A column that separates at an end of its range: nothing. Complete
separation by one predictor: a budget of two iterations; by two predictors: seven;
quasi-complete separation at an interior value: five (measured). With a smaller
<code>maximumIterations</code> those are status 3. The <code>tolerance</code> no longer matters for an indicator (one
row of 600 at a tolerance of 1e-2 was status 0 with a coefficient of 3.94 and a standard error
of 5.39; it is status 4).</li>
<li>Data that are not separated are never status 4. 1,000 rows <code>(x = -1, y = 0)</code>, 2,000 rows
<code>(1, 1)</code> and the rows <code>(-1e-5, 1)</code> and <code>(1e-5, 0)</code> have a finite maximum at a slope of 6.02; the
first Fisher direction moves the two central rows towards the wrong side by 1.5e-5 of their own
terms, which a tolerance of 1e-3 took for separation (status 4 at every budget). They now
converge in 23 steps to the statsmodels maximum (slope within 2e-6 standard errors), and are
status 3 with their iterate when the budget is smaller. 60 samples of 25 to 69 rows with one to
four predictors that a linear program and statsmodels agree are not separated are all status 0,
as are an indicator group with both outcomes at 600, 20,000 and 100,000 rows and two rare-event
fits.</li>
<li>A candidate direction that moves no row towards the wrong side as far as float32 can tell but
leaves some row undecided is remembered, and a later step below <code>tolerance</code> is then status 3,
not 0. No scene of the specs needs this guard.</li>
</ul>
<p>Optional <code>spatialDiagnostics</code> uses the fitted probit probability and generalized/standardized residuals to report <code>[Pinkse chi-square, p, Kelejian–Prucha z, two-sided p, Pinkse–Slade chi-square, p]</code>. Weights must have a sorted zero-diagonal CSR; the induced common complete-case graph is used. Diagnostic status is 0 valid, 1 failed/no-case model, 2 invalid weights, 3 zero test information. This does not estimate latent spatial probit.</p>
<p><code>GPULocalCollinearity</code> uses supplied local estimation weights, including diagonal weights if the
fit uses the focal observation. Every predictor uses the same complete neighbors. VIFs are the
diagonal of the inverse weighted correlation matrix; the condition index is the square root of
its maximum/minimum eigenvalue ratio. The intercept is excluded from this correlation calculation. Paired optional <code>designConditionNumbers</code> and <code>varianceDecomposition</code> reproduce mgwr's intercept-inclusive column-normalized design convention: multiply each design row by normalized spatial weight, not its square root, and report the variance proportion associated with the smallest singular value. A bounded Jacobi eigensolver returns status 3 on exhaustion; status 1 means insufficient neighbors and 2 singular/nonfinite correlation or design. The eigenvalue threshold is <code>1e-7</code> of the largest eigenvalue, an explicit f32 conditioning limit.</p>
<p><code>GPUGeneralizedGeographicallyWeightedRegression.inference</code> returns local scale-one Poisson/logistic standard errors and diagonal influence. The covariance is <code>A^-1 X' V W² X A^-1</code>, with <code>A = X' V W X</code>; it matches mgwr GLM local uncertainty and does not assume that spatial weights are frequency weights. Failed fits retain NaN inference.</p>
<p><code>bandwidthSelection</code> supplies 2–32 positive candidate bandwidths and outputs their exact leave-one-out mean deviance, the selected bandwidth and status (0 selected, 1 no valid candidate). Adaptive candidates must be integer neighbor counts within <code>maximumNeighborCount</code>. Each inner fit excludes its own focus, including from adaptive neighbor selection. Every eligible training row must converge for a candidate to be valid; ties choose the first candidate. The logistic deviance of a row is taken from its leave-one-out linear predictor, <code>2 log(1 + exp(-eta))</code> for a response of 1 and <code>2 log(1 + exp(eta))</code> for 0, so a confident prediction scores finitely (the float32 fitted mean is exactly 1 from <code>eta</code> of about 17 and cannot carry it). This is the default <code>criterion: 'loo-deviance'</code>. Optional <code>criterion: 'aicc'</code> uses in-sample fits and the mgwr Poisson/Binomial deviance/influence correction. <code>method: 'golden-section'</code> searches bounded fixed or adaptive AICc bandwidths with GPU gates and named convergence/budget status; see <a href="/luma-contributors/docs/api/gpu-spatial-analysis/adaptive-generalized-gwr">Adaptive generalized GWR</a>. The pinned oracle executes mgwr local GLM solves with the self weight removed to verify the exact candidate scores.</p>
<p>For nested spatial evaluation, supply one outer fold's <strong>training mask</strong> and set <code>predictExcluded: true</code>. Bandwidth selection, centering/solves and inference use only training rows, while finite held-out predictor/position queries receive predictions without reading their responses. The direct test changes held-out counts from zero to 100,000 and verifies unchanged candidate scores, selected bandwidth, coefficients and predictions. Apply this graph separately to each fold; score held-out responses afterward. Finite query offsets remain caller inputs.</p>
<h2 id="joint-randomization-and-rank-dynamics">Joint randomization and rank dynamics</h2>
<p><code>GPUMultivariateLocalGearyPermutationTest</code> preserves the existing <code>GPUMultivariateLocalGeary</code>
statistic API and adds a separate inference producer. It supports at most 256 observations,
16 variables and 4096 permutations (<code>maximumPermutations</code>, fixed when compiling the graph). A common complete-case mask applies to all variables;
population standardization is performed once. Active local weights are row-standardized.
The focal vector is fixed while distinct <strong>whole neighbor vectors</strong> are sampled without replacement.
Independent column permutations would destroy the cross-variable dependence and are not used.</p>
<p>The conservative equal-tail Monte Carlo p-value is
<code>min(1, 2 * min((lower + 1)/(P + 1), (upper + 1)/(P + 1)))</code>, counting ties in both tails.
<code>tail: 'esda-min-tail'</code> instead computes <code>(min(upper, P-upper)+1)/(P+1)</code>, where <code>upper</code> counts simulated values greater than or equal to the observed value. This reproduces esda's undoubled, tie-asymmetric tail convention; equal-tail remains the default.
Optional Bonferroni adjustment uses all included rows, including islands, as the family.
Classes are 0 nonsignificant, 1 significant similarity, 2 significant dissimilarity and
<code>0xffffffff</code> unknown. Constant variables invalidate the joint statistic; islands are unknown.
Optional simulations expose every row/replicate draw for replay. Seed words and the shared
Philox conditional-permutation purpose determine the stream.</p>
<p><code>GPURankMarkov</code> follows the pinned giddy 2.3.6 rank conventions with zero-based descending ranks.
Equal values receive distinct ranks; default <code>tieOrder: 'giddy'</code> assigns the higher descending rank to the later original row, matching ascending ordinal ranks reversed by giddy. <code>tieOrder: 'row-id'</code> preserves earlier-row-first ties explicitly. Any missing value removes the row
from <strong>all</strong> periods, avoiding changing rank populations over time. Full rank follows a location
through rank states; geographic rank follows the occupant of a fixed rank through location states.
The original <code>n × n</code> output capacity is retained when rows are excluded. Empty transition rows
stay zero. Optional existing Markov diagnostics retain their ergodicity assumptions; optional
mobility reports mean absolute rank movement per adjacent period, NaN below two complete rows.</p>
<h2 id="weights-and-spatial-evaluation">Weights and spatial evaluation</h2>
<p><code>GPUFuzzyContiguity</code> uses the existing feature→polygon→ring→vertex layout, with a packed
<code>float32x2</code> position view. Polygon distance combines segment distances with even-odd filled-part
containment, preserving holes and multipart features. Rings close implicitly. <code>tolerance</code> is the
maximum separation, equivalent geometrically to intersecting exact round buffers of half that
radius; it does not build approximated buffer polygons or use grid snapping.
Features are limited to 65,536 and total vertices to 65,536. Candidate pairs come from a BVH over
feature bounds expanded by the tolerance, and only candidates run the exact distance test, so no
feature-pair workspace is stored. The BVH leaves are sorted along a Morton curve of the bound
centers, so the search cost does not depend on the input order (16,384 shuffled squares take about
5 ms on the headless test GPU, against 110 ms with leaves in source order); the emitted pairs are
radix-sorted by row and neighbor, so each row stays ascending. The query box widening scales with
both the coordinates and the tolerance, so a pair the distance test accepts is found from both
features and the CSR stays symmetric. Feature status distinguishes valid, empty and malformed/nonfinite input. Simple valid
rings are required; self intersections and near-degenerate f32 predicates are not repaired.
Edges are binary and ordered, with <code>added = 1</code> only for strictly positive separation.
Output overflow reports the exact required neighbor count.</p>
<p><code>GPUIslandAttachment</code> selects nearest finite supplied representatives for <strong>original empty CSR
rows</strong>, with lowest-ID distance ties. Selection is simultaneous, so islands may attach to each
other. Reciprocal insertion is the default; directed insertion is explicit. It preserves existing
weights and uses weight 1 only for added edges. Empty means no stored slots, not a zero-weight row.
Nearest comparisons use a shared distance scale to avoid squared-distance overflow or underflow
for finite float32 representatives. <code>nearestSearch</code> selects how the nearest is found and never
changes the result: <code>'scan'</code> tests every representative per island (islands × rows), <code>'grid'</code> takes
the 16 nearest candidates of each island from a <code>GPUNeighborSearch</code> grid and decides among them with
the scan's own comparison, and <code>'auto'</code> (the default) scans up to
<code>GPU_ISLAND_ATTACHMENT_SCAN_MAXIMUM_ROW_COUNT</code> (4,096) rows and uses the grid above. Ties go to the
lowest row on both paths. An island whose candidate list cannot prove the answer (the farthest
candidate is not clearly farther than the chosen one, or the coordinates are so large that the grid
bounds or squared distances overflow) rescans every representative, so the result always equals the
scan's; such islands cost the scan, one pass over all rows (up to 65,535) in a single invocation. That
fallback is never silent: the optional <code>output.status.uncertainCount</code> holds the exact number of
islands that used it (always 0 with <code>'scan'</code>). A count near the island count means the
representatives are too tied (15 or more at one address) or too large for the candidate list to
help: use <code>'scan'</code> or deduplicate. On 60,000 shuffled rows with 25 percent islands the grid took
7 ms against 52 ms for the scan (Apple GPU, second run, readback included). Writing the output costs
each row's own degree. No representative is
inferred, and no implicit normalization is performed. Truncated CSR remains
ordered and bounded, with required counts, attachment IDs and per-edge provenance.</p>
<p><code>GPUSpatialCrossValidation</code> deals explicit blocks to folds in a seeded random order. Every block
that holds a finite unmasked row draws one Philox word keyed by the uint32 seed and the block ID;
the blocks are ranked by that word (ties by block ID) and the block of rank <code>r</code> goes to fold
<code>r % foldCount</code>. Fold sizes therefore differ by at most one block (not one row), no fold is empty
when there are at least <code>foldCount</code> such blocks, and a different seed gives a different order.
With fewer blocks than folds the empty folds remain visible in <code>status</code>. The fold of a block depends
on the seed and on the set of blocks that hold a usable row: masking rows without emptying a block
changes nothing, masking whole blocks re-deals the rest (ranking masked blocks too would leave folds
empty), so <code>GPUNestedInterpolationValidation</code> deals its inner folds anew for each outer fold. Every finite unmasked position
appears in exactly one test fold. A training point is excluded when its planar distance from any
held-out point is less than or equal to <code>bufferDistance</code>. Distance zero also excludes duplicate
positions in different blocks. Invalid coordinates enter neither training nor testing.
The baseline performs exact all-pairs checks and is not an indexed large-domain performance claim.</p>
<p><code>GPUSpatialCrossValidatedOLS</code> composes those masks with a separate OLS fit for each fold. OLS
centering and all estimation statistics use training rows only. It predicts held-out rows using
that fold's centered slopes and mean correction, preserving precision at large predictor origins,
and reports <code>[scoredRows, missingTestRows, MSE, MAE, bias, R²]</code>.
Bias is prediction minus observation. Metric status bits report no scored rows, missing coverage
and undefined R² for constant test outcomes. Changing held-out responses cannot change that fold's
fit. Generalized GWR supplies a training-only <code>bandwidthSelection</code> graph as described above; <a href="/luma-contributors/docs/api/gpu-spatial-analysis/spatial-validation-advanced"><code>GPUNestedInterpolationValidation</code></a> supplies reusable inner-fold covariance/variogram and RBF bandwidth selection, outer refitting and held-out metrics using those masks.</p>
<h2 id="calibrated-spatial-interaction">Calibrated spatial interaction</h2>
<p><code>GPUSpatialInteraction</code> fits a one-covariate Poisson mean proportional to
<code>exp(coefficient * covariate)</code> over an explicit finite coefficient grid or continuous bounded search. Supported constraints
are unconstrained total mass, origin totals, destination totals and both margins. The latter uses
bounded iterative proportional fitting. Default <code>calibration: 'grid'</code> retains a finite-grid optimum. <code>calibration: 'bounded'</code> takes exactly two ascending coefficient endpoints and uses golden-section profile optimization, with <code>maximumCalibrationIterations</code> (1–64, default 32) and <code>coefficientTolerance</code> (default <code>1e-5</code>). The profiled deviance is optimized over the explicit interval; optimum-boundary and f32 flat-objective behavior remain visible in the interval. Candidates with exponent spans above 80 are rejected explicitly.</p>
<p>Rows are explicit OD count cells. <code>pairState</code> distinguishes structural zero (0), observed (1) and
missing (2). Missing cells are excluded from both support and margins and predict NaN; structural
zeros must have observed flow zero and predict zero. Observed zero counts remain in the support.
Duplicate supplied OD rows are independent cells; aggregate duplicates explicitly when appropriate.
Zero observed margins remain zero. Dense routes support at most 4096 cells, 128 origins, 128
destinations and 128 coefficient candidates. The sparse doubly constrained route uses explicit
pair tables within its device and command budget; candidate predictions occupy transient storage.</p>
<p>The selected summary is <code>[coefficient, deviance, maximumRelativeMarginError, iterations]</code>.
Each candidate exposes deviance, error, iterations and status. Unconverged candidates are never
selected as converged fits. If none succeeds, predictions are NaN with a failure status.
Optional origin/destination residuals are observed minus predicted margins. A successful fit is
conditional on the specified support and search domain; coefficient uncertainty is unavailable. Optional bounded-only <code>output.calibration</code> is <code>[lower, upper, width, converged]</code>. Status 4 retains the best finite fit but reports an exhausted coefficient search; inner IPF failure remains separate. The final two candidates are exposed in bounded mode. Direct spint GLM fixtures cover all four constraint modes with the same explicit support.</p>
<h2 id="verification-and-evidence-limits">Verification and evidence limits</h2>
<p>The generator <code>modules/analysis/test/oracles/statistics-coverage.py</code> pins NumPy 2.2.6 and SciPy 1.15.3.
It constructs independent dense IV sandwiches, nested regime designs, block SUR GLS, probit
likelihood/information, profiled panel-error likelihood and interaction IPF references. These are
<strong>composed numerical references</strong>. Separately, <code>statistics-upstream.py</code> actually executes pinned spreg 1.8.3, mgwr 2.2.1, esda 2.7.0, giddy 2.3.6, libpysal 4.13.0 and spint 1.0.7 (with scikit-learn 1.7.2). Its checked-in fixture records full covariance, diagnostics, local inference, exact LOO scores, tied ranks, weights and calibrated flows. <code>statistics-upstream.spec.ts</code> compares GPU results directly against those outputs; it does not relabel the composed NumPy oracle as library parity.</p>
<p><code>coverage-statistics.spec.ts</code> tests robust inference, singular/nonfinite/kernel failures, spatial
fold invariants, both rank conventions and island capacity/provenance. <code>coverage-estimators.spec.ts</code>
adds numerical estimator comparisons, forced probit exhaustion, joint-vector permutation replay,
fuzzy distance/provenance/capacity and an actual held-out-response perturbation test. The node suite
parses and audits generated WGSL and checks storage binding budgets. Native GPU execution results
and final repository gates are recorded in the implementation report; constructors and shader
parsing alone do not validate numerical execution.</p>
<p><code>statistics-performance-benchmark.spec.ts</code> records preparation/upload, graph compilation, first and repeated resident execution, readback and graph transient bytes for increasing SUR and training-only bandwidth-selection workloads. These are device-specific bounded observations, not a universal throughput guarantee or driver-memory peak. Broader stochastic calibration/coverage studies, full topology adversaries and browser-family scaling remain additional evidence.</p>
<h2 id="executable-examples">Executable examples</h2>
<p>These small deterministic synthetic examples run the actual GPU producers and capture inputs, estimates and status outputs. They illustrate the bounded contracts above, not geographic observations or performance evidence.</p>
<h3 id="gpuinstrumentalvariablescovariance"><code>GPUInstrumentalVariablesCovariance</code></h3>
<p>{/* lens:GPUInstrumentalVariablesCovariance */}</p>
<p>::lens{scenario=gpu-spatial-analysis.instrumental-variables-covariance rung=result caption="Inspect the estimates and explicit status outputs on synthetic inputs."}</p>
<p>::lens{scenario=gpu-spatial-analysis.instrumental-variables-covariance rung=graph caption="Inspect the executed GPU command graph."}</p>
<h3 id="gpurestrictedregimeregression"><code>GPURestrictedRegimeRegression</code></h3>
<p>{/* lens:GPURestrictedRegimeRegression */}</p>
<p>::lens{scenario=gpu-spatial-analysis.restricted-regime-regression rung=result caption="Inspect the estimates and explicit status outputs on synthetic inputs."}</p>
<p>::lens{scenario=gpu-spatial-analysis.restricted-regime-regression rung=graph caption="Inspect the executed GPU command graph."}</p>
<h3 id="gpuspatialpanelerrormaximumlikelihood"><code>GPUSpatialPanelErrorMaximumLikelihood</code></h3>
<p>Tiled moments and reductions support 16,777,216 total rows, subject to device bindings. Panel period
count is bounded by the total row count.</p>
<p>{/* lens:GPUSpatialPanelErrorMaximumLikelihood */}</p>
<p>::lens{scenario=gpu-spatial-analysis.spatial-panel-error-maximum-likelihood rung=result caption="Inspect the estimates and explicit status outputs on synthetic inputs."}</p>
<p>::lens{scenario=gpu-spatial-analysis.spatial-panel-error-maximum-likelihood rung=graph caption="Inspect the executed GPU command graph."}</p>
<h3 id="gpuseeminglyunrelatedregression"><code>GPUSeeminglyUnrelatedRegression</code></h3>
<p>Tiled moments and reductions support 16,777,216 total rows, subject to device bindings. Panel period
count is bounded by the total row count.</p>
<p>{/* lens:GPUSeeminglyUnrelatedRegression */}</p>
<p>::lens{scenario=gpu-spatial-analysis.seemingly-unrelated-regression rung=result caption="Inspect the estimates and explicit status outputs on synthetic inputs."}</p>
<p>::lens{scenario=gpu-spatial-analysis.seemingly-unrelated-regression rung=graph caption="Inspect the executed GPU command graph."}</p>
<h3 id="gpuprobitregression"><code>GPUProbitRegression</code></h3>
<p>{/* lens:GPUProbitRegression */}</p>
<p>::lens{scenario=gpu-spatial-analysis.probit-regression rung=result caption="Inspect the estimates and explicit status outputs on synthetic inputs."}</p>
<p>::lens{scenario=gpu-spatial-analysis.probit-regression rung=graph caption="Inspect the executed GPU command graph."}</p>
<h3 id="gpulocalcollinearity"><code>GPULocalCollinearity</code></h3>
<p>{/* lens:GPULocalCollinearity */}</p>
<p>::lens{scenario=gpu-spatial-analysis.local-collinearity rung=result caption="Inspect the estimates and explicit status outputs on synthetic inputs."}</p>
<p>::lens{scenario=gpu-spatial-analysis.local-collinearity rung=graph caption="Inspect the executed GPU command graph."}</p>
<h3 id="gpumultivariatelocalgearypermutationtest"><code>GPUMultivariateLocalGearyPermutationTest</code></h3>
<p>See <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-statistics#gpumultivariatelocalgearypermutationtest">runtime parameters and inference semantics</a>.</p>
<p>{/* lens:GPUMultivariateLocalGearyPermutationTest */}</p>
<p>::lens{scenario=gpu-spatial-analysis.multivariate-local-geary-permutation-test rung=result caption="Inspect the estimates and explicit status outputs on synthetic inputs."}</p>
<p>::lens{scenario=gpu-spatial-analysis.multivariate-local-geary-permutation-test rung=graph caption="Inspect the executed GPU command graph."}</p>
<h3 id="gpurankmarkov"><code>GPURankMarkov</code></h3>
<p>Sorted ranks above 64 rows support up to 65,535 logical states, subject to dense transition-output
bindings. Explicit pairwise ranking and transition diagnostics retain 64.</p>
<p>{/* lens:GPURankMarkov */}</p>
<p>::lens{scenario=gpu-spatial-analysis.rank-markov rung=result caption="Inspect the estimates and explicit status outputs on synthetic inputs."}</p>
<p>::lens{scenario=gpu-spatial-analysis.rank-markov rung=graph caption="Inspect the executed GPU command graph."}</p>
<h3 id="gpufuzzycontiguity"><code>GPUFuzzyContiguity</code></h3>
<p>{/* lens:GPUFuzzyContiguity */}</p>
<p>::lens{scenario=gpu-spatial-analysis.fuzzy-contiguity rung=result caption="Inspect the estimates and explicit status outputs on synthetic inputs."}</p>
<p>::lens{scenario=gpu-spatial-analysis.fuzzy-contiguity rung=graph caption="Inspect the executed GPU command graph."}</p>
<h3 id="gpuislandattachment"><code>GPUIslandAttachment</code></h3>
<p>{/* lens:GPUIslandAttachment */}</p>
<p>::lens{scenario=gpu-spatial-analysis.island-attachment rung=result caption="Inspect the estimates and explicit status outputs on synthetic inputs."}</p>
<p>::lens{scenario=gpu-spatial-analysis.island-attachment rung=graph caption="Inspect the executed GPU command graph."}</p>
<h3 id="gpuspatialcrossvalidation"><code>GPUSpatialCrossValidation</code></h3>
<p>{/* lens:GPUSpatialCrossValidation */}</p>
<p>::lens{scenario=gpu-spatial-analysis.spatial-cross-validation rung=result caption="Inspect the estimates and explicit status outputs on synthetic inputs."}</p>
<p>::lens{scenario=gpu-spatial-analysis.spatial-cross-validation rung=graph caption="Inspect the executed GPU command graph."}</p>
<h3 id="gpuspatialcrossvalidatedols"><code>GPUSpatialCrossValidatedOLS</code></h3>
<p>{/* lens:GPUSpatialCrossValidatedOLS */}</p>
<p>::lens{scenario=gpu-spatial-analysis.spatial-cross-validated-ols rung=result caption="Inspect the estimates and explicit status outputs on synthetic inputs."}</p>
<p>::lens{scenario=gpu-spatial-analysis.spatial-cross-validated-ols rung=graph caption="Inspect the executed GPU command graph."}</p>
<h3 id="gpuspatialinteraction"><code>GPUSpatialInteraction</code></h3>
<p>Dense routes retain 4,096 pairs and 128 groups. Sparse doubly constrained balancing accepts explicit
pair tables, subject to device bindings and the command-node budget; incomplete balancing is
reported.</p>
<p>{/* lens:GPUSpatialInteraction */}</p>
<p>::lens{scenario=gpu-spatial-analysis.spatial-interaction rung=result caption="Inspect the estimates and explicit status outputs on synthetic inputs."}</p>
<p>::lens{scenario=gpu-spatial-analysis.spatial-interaction rung=graph caption="Inspect the executed GPU command graph."}</p>
<h2 id="recorded-reference-and-scaling-run">Recorded reference and scaling run</h2>
<p>The separate <code>statistics-blockcv.R</code> generator actually executes blockCV 4.0.1 <code>cv_spatial</code> with
explicit user polygons/predefined folds and <code>cv_buffer</code> with projected distances. The GPU design
is compared with both as a partition: every GPU fold must be one blockCV fold, with the same test
rows and the same training rows, whatever number the fold carries. Blocks are dealt to folds in a
seeded Philox order (see above), which is our explicit deterministic assignment rule;
this does not claim identity with blockCV's random balancing optimizer.</p>
<p>The checked-in <code>statistics-performance-baseline.json</code> records the Apple WebGPU run. Resident
median encode/submit/queue-completion times were:</p>
<table>
<thead>
<tr>
<th>Workload</th>
<th align="right">Rows</th>
<th align="right">Resident median</th>
<th align="right">Graph transient bytes</th>
</tr>
</thead>
<tbody>
<tr>
<td>Two-equation SUR</td>
<td align="right">64</td>
<td align="right">1.8 ms</td>
<td align="right">0</td>
</tr>
<tr>
<td>Two-equation SUR</td>
<td align="right">1,024</td>
<td align="right">4.8 ms</td>
<td align="right">0</td>
</tr>
<tr>
<td>Two-equation SUR</td>
<td align="right">16,384</td>
<td align="right">63.8 ms</td>
<td align="right">0</td>
</tr>
<tr>
<td>Three-candidate Poisson LOO search</td>
<td align="right">32</td>
<td align="right">2.5 ms</td>
<td align="right">1,928</td>
</tr>
<tr>
<td>Three-candidate Poisson LOO search</td>
<td align="right">128</td>
<td align="right">7.0 ms</td>
<td align="right">7,688</td>
</tr>
<tr>
<td>Three-candidate Poisson LOO search</td>
<td align="right">512</td>
<td align="right">17.5 ms</td>
<td align="right">30,728</td>
</tr>
</tbody>
</table>
<p>Caller-owned input/output buffers are counted separately in the fixture. Zero graph transients
for SUR does not mean zero GPU memory: it writes caller outputs and uses shader-local scratch.
Preparation includes allocation/upload and fixture creation. First execution includes deferred
pipeline work; cached resident measurements are not cold-driver performance. Compilation, readback,
input/output bytes and result certificates are retained with every observation.</p>
<h2 id="random-and-unit-effects-spatial-lag-maximum-likelihood">Random and unit-effects spatial lag maximum likelihood</h2>
<p><code>GPUSpatialPanelLagMaximumLikelihood</code>: Auto selects tiled unit effects when entity eigenvalues are
supplied. Tiled fits support 1,024 entities and 16,777,216 total rows, subject to device bindings;
reference fits retain 32 entities and 32 periods.</p>
<p><code>GPUSpatialPanelLagMaximumLikelihood</code> fits <code>y = rho Wy + X beta + unitEffect + error</code>
on complete balanced panels in period-major order. The random-effects fit alternates analytic
variance-component shrinkage <code>phi</code> with an exact dense log-determinant profile search for <code>rho</code>;
unit effects use within-entity demeaning. The reference capacity is 2–32 entities, 2–32 periods and 1–7 predictors. With
<code>fitMode</code> selecting tiled unit effects and a supplied entity spectrum, the bound is 1,024 entities
and 16,777,216 total rows, subject to device bindings. Weights must be nonnegative, row-standardized, zero-diagonal CSR with sorted
unique neighbors and no islands. Missing observations invalidate the complete fit.</p>
<p>Both GPU routes and the CPU twin refuse incomplete weights: nonzero
<code>weights.status.overflow[0]</code> (final capacity loss) or
<code>weights.status.candidateOverflow[0]</code> (candidate loss) returns status 1 with NaN coefficients,
covariance, summary, fitted values and residuals. Clearing the input flags permits a subsequent
submission to fit normally; the two input flags retain their distinct meanings.</p>
<p><code>CPUSpatialPanelLagMaximumLikelihoodOperation</code> uses the same <code>fitMode</code> and <code>eigenvalues</code> ports.
Auto selects the supplied spectrum only for unit effects; <code>reference</code> rejects a spectrum and
<code>tiled</code> requires one. Supply exactly one eigenvalue per entity: a length mismatch throws, while
a nonfinite value or an absolute value above float32(1.00001) returns status 1 and NaN outputs.
The CPU objective and determinant derivatives use the supplied spectrum; the caller must ensure
it belongs to the entity weights matrix. The pure <code>computeSpatialPanelLagMaximumLikelihoodOnCPU</code>
accepts the spectrum as <code>input.eigenvalues</code> and the mode as <code>options.fitMode</code>; with no spectrum
it evaluates the dense CSR determinant. GPU-specific size caps do not limit the CPU fit.
Explicit <code>maximumSearchIterations</code> and <code>maximumIterations</code> budgets bind the CPU fit: exhausting
a search budget with a rho bracket wider than <code>tolerance</code>, or an outer budget before convergence,
returns status 3 with the current coefficients, fitted values and residuals; covariance is withheld.
One golden-section step on <code>[-0.999, 0.999]</code> leaves a bracket about 1.23483 wide. When budgets are
omitted, the CPU search continues to binary64 resolution and alternation continues to convergence
or a repeated state. Residual-square sums use compensated float64 accumulation so repeating
periods preserves the estimate and scales covariance by the inverse repetition count.</p>
<p>The contributor and both operation forms capture declared fit options, spectrum presence and
graph port/status maps by property read at construction, including inherited or non-enumerable
getters. Each getter is read once; changing its return value later cannot remove a loss flag.
Mutating the original props cannot change the declared spectrum dependency; the spectrum's
values are read anew on each execution. A graph spectrum-length error starts with the configured
operation ID; the standalone CPU function uses its own function name.</p>
<p>The fit runs on centered data: the pooled mean of each predictor and of the response is removed
before quasi-demeaning, so a predictor with a large mean (a year, an elevation, an income) costs
no precision. Only the intercept depends on the shift; it is mapped back as
<code>intercept + (1 - rho) mean(y) - slopes' mean(x)</code> and its covariance row and column are
transformed to match. Normal and information matrices are factored after diagonal equilibration,
and a Cholesky pivot of the unit-diagonal matrix below <code>1e-3</code> (the limit of
<code>GPUOrdinaryLeastSquares</code>) is status 2.</p>
<p>The concentrated variance is formed directly. With <code>e0</code>, <code>e1</code> the residuals of the quasi-demeaned
response and of its lag on the predictors, <code>rho0 = e0'e1 / e1'e1</code> and <code>r = e0 - rho0 e1</code> summed
per row, <code>u'u = S + e1'e1 (rho - rho*)^2</code>, two non-negative terms; the search and the published
<code>sigma2</code>, log-likelihood and covariance all use it. The former <code>e0'e0 - 2 rho e0'e1 + rho^2 e1'e1</code>
of three float32 sums is rounding once the model fits well. Measured on the test GPU against spreg
1.9.1 <code>Panel_FE_Lag</code> and <code>Panel_RE_Lag</code> (16 entities, 16 periods, <code>rho</code> 0.4, residual deviation
0.004): with <code>u'u</code> at 3e-5 of <code>e0'e0</code> the published <code>sigma2</code> was 1% to 2% off and is now within
1e-5; at 3e-7, 36% low and 96% high, now within 6.3e-5; at 3e-9 (eight scenes, both effects) it
was 10 to 241 times too large with status 0 in 9 of the 16 fits, negative with status 2 in one
and status 3 in six, and is now within 1.7e-3 with status 0 in all 16, where the float32 rounding
of the fitted values (0.6% of a residual per row) is what remains. The lag of the response is summed from differences rounded as written, with the
row sums of the float32 weights carried exactly.</p>
<p>During the <code>rho</code> search, the tiled GPU route removes the rho-independent log-variance constant and
evaluate small relative variance increments with a series. Otherwise float32 can round nearby
variances to the same value, so a narrow search bracket can still surround a biased estimate.
The published log-likelihood retains the full log variance and its existing constants.</p>
<p>Entity effects survive the pooled centering. Each column is stored as its within deviation
(centered value less its entity mean) and its entity mean, and the quasi-demeaned value
<code>value - (1 - phi) mean</code> is formed as <code>deviation + phi mean</code>, whose terms do not cancel; the
normal equations of each step are the polynomial in <code>phi</code> of three moment sums taken once.
Formed as written, <code>(value - mean) - fitted</code> was regrouped by the shader compiler of the test GPU
as <code>value - (mean + fitted)</code>: with entity effects of +3000 and -3000 (16 entities, 16 periods,
unit effects, residual deviation 0.004) all 256 published residuals were multiples of the float32
step of 3000 and <code>rho</code> was 0.026 standard errors from spreg <code>Panel_FE_Lag</code>; it is now within
0.002. What float32 leaves there: the entity means are sequential float32 sums, off by up to
6e-4, which leaves a constant per entity in the residuals and <code>sigma2</code> 0.6% high (1.1% before),
and the lag of the centered response is rounded to a float32 step of the effect, which leaves the
within-entity part of a residual within 7e-5.</p>
<p><code>effects</code> defaults to <code>random</code>. <code>maximumIterations</code> (1–128, default 64) bounds outer sweeps;
<code>maximumSearchIterations</code> (1–64, default 32) bounds each golden-section search. <code>tolerance</code>
(default 1e-5) controls phi change and final rho bracket width; the change of phi is held against
it as an absolute change for a phi of 0.01 or more and in proportion to <code>phi / 0.01</code> below. Rho is
restricted to [-.999,.999], and random phi to [1e-8,1]. Boundary summary bit 1 flags rho near
either endpoint; bit 2 flags random phi at either endpoint; bit 4 (with bit 2 and status 3) flags
a phi whose estimate is below 1e-8 and was held there: the published numbers are then those of
the constrained problem, not the maximum, and there is no covariance. Unit phi is exactly zero
and sets neither bit. The lower limit was 1e-5 and was reported as status 0 with bit 2 only:
entity effects of +3000 and -3000 with residuals of 0.004 (spreg <code>Panel_RE_Lag</code> phi 2.5e-7) came
out with <code>sigma2</code> 105 times spreg's and <code>rho</code> 3.5 standard errors off; that fit is now status 0
with phi within 0.2% of spreg, <code>rho</code> within 0.009 standard errors, the intercept within 0.2 and
<code>sigma2</code> within 1.1% (the float32 entity means). float32 data cannot carry residuals much below
1e-6 of an entity effect, so with 16 periods phi stays above about 1e-8 on data the kernel can
resolve (effects of 16,000,000 with residuals of 0.5: spreg phi 9.1e-9, status 3).</p>
<p><code>fitted</code> is the quasi-demeaned response at the published phi less the residual
(<code>fitted + residuals = y - (1 - phi) mean_i(y)</code>). A phi inside the tolerance of the alternation
but <code>d</code> away from spreg's moves <code>fitted</code> by <code>d</code> times the entity mean of the response and leaves
<code>residuals</code> alone: with a response mean of 10,000 and residuals of 0.004, a phi within 4.5e-7 of
spreg's gives fitted values 1.2 residual deviations from spreg's.</p>
<p>With <code>p = predictorCount + 1</code>, output shapes are: <code>coefficients[p+2]</code> (intercept, slopes, rho,
phi), <code>covariance[(p+1)²]</code>, <code>summary[6]</code>, <code>status[1]</code>, and observation-sized <code>fitted</code>/<code>residuals</code>.
Unit intercept and its covariance row/column are NaN. Covariance matches spreg's conditional
<code>varb</code> expected-information block for beta/rho, excluding phi and innovation variance.
Summary is <code>[sigma2, reportedLogLikelihood, sweeps, phiChange, rhoBracketWidth, boundaryBits]</code>.
The log likelihood deliberately follows spreg 1.8.3 reporting: unit effects retain its additional
<code>-NT/2 * log(NT)</code> normalization; comparing random and unit reported values as a likelihood-ratio
test is inappropriate. Fitted values and innovations use the quasi-demeaned response convention.</p>
<p>Fit status is 0 converged, 1 invalid/incomplete inputs, 2 singular or nonpositive/nonfinite
innovation variance, and 3 iteration/search exhaustion. Exhaustion exposes provisional estimates
and diagnostics but NaN covariance. Other failures never publish completed inference.</p>
<p>Optional <code>hausman: {tests, status}</code> additionally fits unit effects and compares slopes/rho using
spreg's <code>RE.varb - FE.varb</code> convention. <code>tests[3]</code> is <code>[statistic, df, p]</code>. The compatibility df
is <code>predictorCount + 2</code> (spreg's <code>RE.k</code>, including intercept and rho), <strong>not</strong> the contrast rank
<code>predictorCount + 1</code>. Status is 0 valid, 1 unsuccessful constituent fit, or 2 non-positive-definite
covariance difference/nonfinite comparison; invalid comparisons return NaNs. This explicitly
rejects negative Hausman statistics instead of interpreting them as chi-square inference.</p>
<p>The pinned <code>statistics-advanced.py</code> generator executes spreg 1.8.3 <code>Panel_RE_Lag</code>, <code>Panel_FE_Lag</code>
and <code>panel_Hausman</code>, and mgwr 2.2.1 full response-smoother inference. Native tests compare panel
estimates, conditional covariance, innovations, reported likelihood and valid/invalid Hausman cases,
and exercise missing data, singular within-design, zero variance and both iteration budgets.</p>
<p>{/* lens:GPUSpatialPanelLagMaximumLikelihood */}</p>
<p>::lens{scenario=gpu-spatial-analysis.spatial-panel-lag-maximum-likelihood rung=result caption="Inspect random-effects spatial lag estimates and Hausman diagnostics."}</p>
<p>::lens{scenario=gpu-spatial-analysis.spatial-panel-lag-maximum-likelihood rung=graph caption="Inspect the executed panel likelihood and diagnostic kernels."}</p>`};export{e as default};