var e={slug:`api/gpu-spatial-analysis/operators-regression`,title:`GPU Spatial Analysis: Regression`,html:`<h1 id="regression">Regression</h1>
<h3 id="gpuordinaryleastsquares"><code>GPUOrdinaryLeastSquares</code></h3>
<p>Fits a linear model of one response on predictor columns by least squares and reports coefficients, standard errors, t statistics and diagnostics.</p>
<p>::lens{scenario=gpu-spatial-analysis.ordinary-least-squares rung=result caption="Diabetes on three predictors across 788 tracts, R squared 0.67; compare each estimate with its standard error."}</p>
<p>Ordinary least squares with an intercept and a full regression report, in one prebuilt graph:
coefficients, standard errors, t statistics, R2, adjusted R2, sigma squared, log-likelihood, AIC,
BIC, optional per-row residual and fitted columns, the Jarque-Bera normality test, the
Koenker-Bassett and Breusch-Pagan heteroskedasticity tests, and collinearity and precision
diagnostics (Belsley condition number, correlation condition number, maximum VIF, precision
warning). <code>getGPUOrdinaryLeastSquaresModelResult(readback, {predictorNames})</code> turns the read-back
arrays into the shared <code>GPURegressionModelResult</code> (coefficient table with Student t p-values on
<code>n - p</code> degrees of freedom, <code>sigmaSquared = RSS / (n - p)</code> and <code>sigmaSquaredMaximumLikelihood = RSS / n</code>,
R2, log-likelihood, AIC and BIC equal to spreg, conditioning, and the tests <code>jarqueBera</code>,
<code>breuschPagan</code>, <code>koenkerBassett</code>).</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> graph</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUCommandGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(device);</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">  new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUOrdinaryLeastSquares</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    predictors, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// GraphDataView&#x3C;'float32'>, row-major row * predictorCount + column</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    response, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// GraphDataView&#x3C;'float32'></span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    mask, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// optional GraphDataView&#x3C;'uint32'>; zero excludes the row</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    parameters: ridgeBuffer.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph), </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// optional: getGPUOrdinaryLeastSquaresParameterValues({ridgeLambda})</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    predictorCount: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// compile-time, 1 to 15 (16-wide Cholesky); the intercept is added internally</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    output: {coefficients, standardErrors, tStatistics, summary, status: {code: status}, residuals, fitted}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  })</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">);</span></span></code></pre>
<ul>
<li>Rows with any non-finite predictor or response value, or a zero mask, are excluded.</li>
<li><code>coefficients</code>, <code>standardErrors</code>, <code>tStatistics</code>: <code>predictorCount + 1</code> floats, intercept first.</li>
<li><code>summary</code> (22 floats, <code>GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH</code>), indexed by
<code>GPU_ORDINARY_LEAST_SQUARES_SUMMARY_*</code>: row count, R2, adjusted R2, sigma squared, log-likelihood,
AIC, BIC, Jarque-Bera, its p-value, Koenker-Bassett, its p-value, RSS, TSS, skewness, kurtosis,
ridge lambda, Breusch-Pagan, its p-value, <code>CONDITION_NUMBER</code>, <code>CORRELATION_CONDITION_NUMBER</code>,
<code>MAXIMUM_VARIANCE_INFLATION_FACTOR</code>, <code>PRECISION_WARNING</code>. The slots 9 and 10 used to be called
<code>BREUSCH_PAGAN</code>; they were always the Koenker-Bassett statistic, and <code>BREUSCH_PAGAN</code> now names the
classic test in slots 16 and 17.</li>
<li>The predictor count is capped at 15 (16 coefficients with the intercept) by the 16-wide
function-local Cholesky factorization of the solve; a larger design needs a tiled solver.</li>
<li><code>status.code</code> (uint32): 0 ok, 1 singular or ill-conditioned (a predictor within <code>1 - 1e-6</code> of the span
of the others, or constant), 2 too few rows (<code>n &#x3C;= predictorCount + 1</code>). A failed fit leaves every
statistic and per-row column NaN (the row count is still written).</li>
<li><code>residuals</code> and <code>fitted</code> are optional per-row columns (NaN for excluded rows).</li>
<li>Optional <code>predictions</code> has one value per input row, including masked rows and rows with a missing
response, provided their predictors are finite. It uses the fit's training-only centers, slopes
and mean correction; a failed fit or nonfinite predictors produce NaN. The operation form and
CPU twin expose the same output. Raw float32 intercepts can lose response-level information at
large predictor origins, so use these centered predictions for held-out evaluation.
The OLS GPU operation and CPU twin capture all graph names and optional output presence at
construction, including the nested status name. Reusing or editing the caller's props/output
map cannot rename a declared prediction or disable its writer; construct a new operation to
change ports. Input graph values remain per-run data.</li>
<li>Conditioning never changes <code>status</code>: a warning is not a failure and the fit is still reported.
<code>CONDITION_NUMBER</code> is the Belsley number (spreg <code>OLS.mulColli</code>): <code>sqrt(lambdaMax / lambdaMin)</code> of the
raw <code>[1, X]</code> Gram matrix scaled to unit diagonal (rebuilt from the centered moments and means; values
above about 30 mean harmful collinearity; ridge is not included). <code>CORRELATION_CONDITION_NUMBER</code> is
<code>lambdaMax / lambdaMin</code> of the predictor correlation matrix the float32 Cholesky factors (with the
ridge penalty), and <code>MAXIMUM_VARIANCE_INFLATION_FACTOR</code> the largest diagonal of its inverse. Float32
coefficients lose about <code>correlationConditionNumber * 6e-8</code> relative accuracy, so
<code>PRECISION_WARNING</code> is 1 when the correlation condition number exceeds 1e4
(<code>GPU_REGRESSION_PRECISION_CONDITION_LIMIT</code>, about three of seven digits lost). Eigenvalues come from a
cyclic Jacobi routine in float32, so the condition numbers are indicative for very ill-conditioned
designs (the GPU specs allow 2e-2 to 1e-1 relative against the float64 oracle and spreg). They match spreg <code>mulColli</code> and <code>diagnostics.vif</code> (max
over the predictor columns, computed on the centered design) in the float64 oracle.</li>
<li>The optional ridge penalty <code>lambda >= 0</code> is added to the centered normal matrix of the
predictors (never the intercept) and changes between frames without rebuilding the graph. Ridge
standard errors are <code>sigma2 (X'X + lambda I)^-1</code> and ignore the shrinkage bias.</li>
</ul>
<p>Numerics: no float atomics. Per-tile partial sums (one invocation per tile of consecutive rows, in
row order) are merged in a fixed pairwise order (a float Kahan compensation is folded to a plain
sum by the shader compiler of at least one supported GPU). Each tile first selects a complete
training-row origin and sums differences from it. The merge shifts those partials to the first
nonempty tile's origin before averaging; empty tiles contribute zero. This preserves small
variances when many nearly equal predictors have a large coordinate origin. Means first, then centered
cross-products about those means; the solve runs on the correlation-scaled normal matrix with a
single-invocation Cholesky. The residual sum of squares comes from the residual pass, not from
<code>Syy - b'Sxy</code>, so it stays accurate when R2 is close to 1. A float32 mean misses the true mean of
its column by up to a float32 step; the moment pass also sums the deviations of every column from
its stored mean, and that remainder is carried through the cross-products, the intercept and the
residuals, so the residuals sum to zero and the residual sum of squares matches float64 also when
the response mean is thousands of residual deviations (residuals of 0.002 around 10,000: within
1e-4, where a float32 mean alone gave 4% to 20% too much). Results are bitwise reproducible for a
given <code>tileRowCount</code> (default <code>max(64, ceil(rows / 4096))</code>).
Mean corrections are divided by the row count before multiplication, with the quotient fixed by
an integer bit cast, so an overflowing product of deviation sums cannot erase a finite variance.
Fitted values and held-out predictions also fix the combined centered contribution and mean
correction with an integer bit cast before adding the response mean. This prevents Metal fast
math from adding small correction terms to a large mean separately and rounding them away.</p>
<p>AIC and BIC count <code>predictorCount + 1</code> parameters (GeoDa convention); the log-likelihood is
Gaussian with <code>sigma2 = RSS / n</code>. Each estimator on this page counts as its reference library does:
OLS the intercept and predictors (<code>p</code>), <code>GPUSpatialLagMaximumLikelihood</code> <code>p + 1</code> (it adds <code>rho</code>),
<code>GPUSpatialErrorMaximumLikelihood</code> <code>p</code> (not <code>lambda</code>), and the GWR family <code>tr(S) + 1</code> (the error
variance). Put the models on one count before comparing criteria: each uncounted parameter adds 2
to AIC and <code>ln(n)</code> to the Schwarz criterion. Jarque-Bera is <code>n/6 (S^2 + (K - 3)^2 / 4)</code> with p-value
<code>exp(-JB / 2)</code>. Both heteroskedasticity tests use the auxiliary regression of <code>e^2</code> on <code>[1, X]</code> with
<code>R2_aux</code>, and a chi-square p-value on <code>predictorCount</code> degrees of freedom computed by the regularized
upper incomplete gamma with a fixed iteration cap. Koenker-Bassett (studentized, valid without
normality) is <code>n R2_aux</code>; the classic Breusch-Pagan (assumes normal errors) is
<code>0.5 R2_aux (sum e^4 / s2^2 - n) = 0.5 R2_aux n (kurtosis - 1)</code> with <code>s2 = RSS / n</code>, which is spreg
<code>breusch_pagan</code> with <code>z = [1, X]</code>. Both match spreg 1.9.1 (<code>breusch_pagan</code>, <code>koenker_bassett</code> with
<code>z = reg.x</code>; spreg's default <code>z</code> is the squared design). With skewed errors they differ several
times over, and Jarque-Bera rejecting normality is the cue to read Koenker-Bassett instead.
<code>getChiSquareSurvival(value, degreesOfFreedom)</code> is the float64 CPU mirror, and
<code>getStudentTTwoSidedPValue</code> / <code>getStudentTSurvival</code> (float64 regularized incomplete beta, pinned to
<code>scipy.stats.t.sf</code>) back the model-result p-values.</p>
<p>The shared model result derives global corrected AIC as <code>AIC + 2 k (k + 1) / (n - k - 1)</code>,
where <code>k</code> is the information-criterion parameter count. It returns <code>null</code> when <code>n &#x3C;= k + 1</code>
or the model has no likelihood. Estimator-specific corrected-AIC overrides are preserved.</p>
<h4 id="how-it-computes">How it computes</h4>
<p>{/* lens:GPUOrdinaryLeastSquares */}</p>
<p>::lens{scenario=gpu-spatial-analysis.ordinary-least-squares rung=formula caption="Follow 6 equations: normal equations, intercept, sigma squared, standard errors, t statistics and R squared."}
::lens{scenario=gpu-spatial-analysis.ordinary-least-squares rung=kernel caption="Inspect the tile, merge, solve, residual and finish kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.ordinary-least-squares rung=graph caption="Trace the accumulation, solve, residual and report nodes of one fit on 788 tracts."}</p>
<p>{/* parameters: GPUOrdinaryLeastSquares generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpuordinaryleastsquares-parameters"><code>GPUOrdinaryLeastSquares</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUOrdinaryLeastSquaresParameterValues(settings?)</code>, writes <code>GPU_ORDINARY_LEAST_SQUARES_PARAMETER_LENGTH</code> (1) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPUOrdinaryLeastSquaresParameterValues</code>:</p>
<table>
<thead>
<tr>
<th>Field</th>
<th>Type</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>ridgeLambda</code> (optional)</td>
<td><code>number</code></td>
<td>Ridge penalty in squared predictor units, added as <code>lambda * I</code> to the centered normal matrix of the predictors (never to the intercept). A value that is not finite and positive means no penalty. Defaults to 0.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="gpugeographicallyweightedregression"><code>GPUGeographicallyWeightedRegression</code></h3>
<p>Fits a separate weighted regression at every location and writes local coefficients, standard errors, t values and fit statistics, choosing the bandwidth by AICc.</p>
<p>::lens{scenario=gpu-spatial-analysis.geographically-weighted-regression rung=result caption="Diabetes beside the local poverty coefficient, 0.002 to 0.34 across 788 tracts; compare where it is strongest."}</p>
<p>Pass optional <code>spatialContext</code> to <code>getGPUGeographicallyWeightedRegressionParameterValues(settings)</code> to warn on short fixed <code>bandwidths</code>; see <a href="/luma-contributors/docs/api/gpu-spatial-analysis/concepts#precision-of-longitudelatitude-positions">Precision of longitude/latitude positions</a>.</p>
<p>Geographically weighted regression with AICc bandwidth selection (Fotheringham, Brunsdon and
Charlton 2002; mgwr, ArcGIS GWR). For every included row <code>i</code> it fits a weighted least squares
<code>y ~ 1 + x</code> over all included rows <code>j</code> with weights <code>K(d_ij / h)</code> and writes local coefficients,
local standard errors, t values and a significance mask, local R-squared, fitted values, residuals and
the hat-matrix diagonal.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> parameters</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUParameterBuffer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(device, {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  id: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'gwr-parameters'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  format: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'float32'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  length: </span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">getGPUGeographicallyWeightedRegressionParameterLength</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">16</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">  new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUGeographicallyWeightedRegression</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    positions, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// GraphDataView&#x3C;'float32x2'>, planar (or lon/lat degrees with spatialContext)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    predictors, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// row-major row * predictorCount + column</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    predictorCount: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    response,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    parameters: parameterBuffer.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    maximumBandwidthCount: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">16</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    output: {coefficients, standardErrors, tValues, significant, selectedBandwidth, summary}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  })</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">);</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">parameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">write</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">  getGPUGeographicallyWeightedRegressionParameterValues</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    {kernel: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'gaussian'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, bandwidthMode: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'adaptive'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, bandwidths: [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">20</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">40</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">60</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">80</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">]},</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">    16</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  )</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">);</span></span></code></pre>
<ul>
<li>
<p>Kernels: <code>'gaussian'</code> <code>exp(-0.5 (d/h)^2)</code> (unbounded unless <code>gaussianCutoff</code> is set) and <code>'bisquare'</code>
<code>(1 - (d/h)^2)^2</code> for <code>d &#x3C; h</code>.</p>
</li>
<li>
<p>Bandwidths: <code>'fixed'</code> (ladder values are distances) or <code>'adaptive'</code> (ladder values are neighbour
counts <code>k</code>, <code>h</code> is 1.00001 times the distance to the k-th nearest included row, the row itself
counted as in mgwr; <code>k</code> in <code>[2, maximumNeighborCount]</code>, at most 128).</p>
</li>
<li>
<p>Bandwidth ladder: up to <code>maximumBandwidthCount</code> (at most 32) candidates in the parameter buffer.
Each candidate's AICc, <code>n ln(RSS/n) + n ln(2 pi) + n (n + tr S) / (n - 2 - tr S)</code>, is evaluated on
the GPU and the outputs are written at the selected bandwidth. A ladder of one value is a fixed
bandwidth. Candidates with a non-positive bandwidth, invalid <code>k</code> or <code>n - 2 - tr S &#x3C;= 0</code> score NaN.</p>
</li>
<li>
<p>Singular locations (changed): a location whose local system is singular (for example a rare dummy
that is constant in its neighbourhood) makes only that row NaN (coefficients, standard errors,
fitted value, residual, hat; <code>localStatus</code> 1). It no longer voids the whole candidate. A candidate's
<code>n</code>, RSS, <code>tr S</code> and AICc are taken over its non-singular rows, and <code>bandwidthSingularRowCounts</code>
reports the count per candidate. Selection: among the candidates with the fewest singular rows the
lowest AICc wins (ties to the lowest index), so a bandwidth with singular rows is chosen only when
every valid candidate has some. This is a deliberate deviation from mgwr, which raises on a singular
local system.</p>
</li>
<li>
<p><code>localStatus</code> 1 (<code>SINGULAR</code>) also covers an ill-conditioned local design. The local normal equations
are scaled to unit diagonal (a correlation matrix, on the design centred at the focus), and a Cholesky
pivot below <code>GPU_ORDINARY_LEAST_SQUARES_PIVOT_TOLERANCE</code> (1e-3, the OLS limit) is singular, so a
local predictor within about <code>1 - 1e-6</code> of the span of the earlier columns (variance inflation factor
above 1e6) is rejected exactly where <code>GPUOrdinaryLeastSquares</code> reports status 1. Before this test only
a non-positive pivot failed and a nearly collinear design returned coefficients with status 0. The
same test applies to the generalized GWR information matrix and the nonstationarity test refits.
Well-conditioned fits are unchanged.</p>
</li>
<li>
<p>Refinement (new, default on; <code>refinement: false</code> keeps the fixed ladder, a number from 1 to 4 sets
the pass count, default 2): after the caller ladder, a GPU kernel writes a new ladder that zooms
around the best candidate between its nearest evaluated neighbours (it never extrapolates beyond the
caller's range), the same kernels evaluate it, and the overall best wins. Adaptive ladders hold
integers: when the integers inside the bracket number at most <code>maximumBandwidthCount</code> all of them
are tried, so the result is the exact integer optimum of the bracket; a wider bracket is sampled at
equal spacing and the next pass narrows it. Fixed ladders are <code>maximumBandwidthCount</code> equally spaced
distances per pass. Each pass costs up to one more ladder evaluation, so the default is about three
times the work of the bare ladder; shorten the ladder rather than turning refinement off when cost
matters. <code>selectedBandwidth</code> is <code>[candidate index, value]</code>; indices from <code>maximumBandwidthCount</code> up
are refined candidates, and <code>candidateTable</code> lists every candidate of every pass. Every pass reads its
ladder through a GPU pass cursor, so N and N + 1 passes compile the same pipelines. On the
144-row test scene (adaptive bisquare) mgwr's golden section with its default bounds <code>[40 + 2 p, n]</code>
stops at <code>k = 47</code>, while its exhaustive integer AICc optimum, which the refinement finds from a
five-value ladder, is <code>k = 26</code> (and mgwr's golden section with bounds widened to 6..144 agrees).
The AICc curve in <code>k</code> is not unimodal there (local minimum near 30), which is why a bracket that
fits the ladder is enumerated rather than searched.</p>
</li>
<li>
<p>Golden-section search (new, opt-in <code>bandwidthSearch: 'golden-section'</code>; the default <code>'ladder'</code> is
unchanged): mgwr's <code>Sel_BW(...).search('golden_section')</code> as a fixed-depth loop on the GPU, about
12 to 20 candidate evaluations instead of a ladder of up to 32 per pass, no read-back. The first two
<code>bandwidths</code> values of the parameters are the bounds <code>[minimum, maximum]</code>; a value of 0 or less
takes mgwr's default for that bound, so <code>[0, 0]</code> is mgwr's default search, and a single value is a
fixed, unselected bandwidth. Defaults are computed on the GPU from the included rows: adaptive
<code>[40 + 2 (predictorCount + 1), n]</code> with <code>n</code> capped at <code>maximumNeighborCount</code> (mgwr's <code>n</code> cannot be
evaluated beyond the neighbour list; identical for <code>n &#x3C;= maximumNeighborCount</code>), fixed
<code>[min_dist / 2, 2 max_dist]</code> with <code>min_dist</code> the smallest positive row-to-row distance and
<code>max_dist</code> the largest pairwise distance (exact without a grid index; with one, the largest
distance from a row to a corner of the grid extent, an upper bound that costs at most an
iteration or two). The interior points are mgwr's <code>a + 0.38197 (c - a)</code> and <code>c - 0.38197 (c - a)</code>,
rounded half to even to integers for adaptive bandwidths. Iteration 1 evaluates both points, every
later iteration compares the two scores, shrinks the bracket and evaluates the one new interior
point with the same candidate kernels (the other is reused). The search stops when the two
interior AICc values differ by at most <code>max(1e-6, 4e-6 |AICc|)</code>, mgwr's <code>1e-6</code> with a float32 noise
floor (the AICc sums <code>n</code> float32 terms; <code>1e-6</code> alone never fires at AICc values of hundreds), or
after <code>maximumSearchIterations</code> (default 30, at most 64, compile-time). The selected bandwidth is
the winner of the last comparison, as mgwr returns it; fewest singular rows first, then the lower
AICc, an invalid score loses to a valid one and two invalid scores stop the search. The iteration
index is a GPU cursor and the loop is gated by an indirect dispatch that the one-thread iteration
kernel zeroes on convergence, so N and N + 1 iterations compile the same pipelines and converged
iterations launch nothing. <code>candidateTable</code> then has <code>maximumSearchIterations + 1</code> rows in
evaluation order (rows 0 and 1 are the first interior points, row <code>i + 1</code> the point of iteration
<code>i</code>; unreached rows are NaN), <code>selectedBandwidth[0]</code> is the table row of the selection,
<code>bandwidthScores</code> and <code>bandwidthSingularRowCounts</code> hold its first <code>maximumBandwidthCount</code> rows, and
the optional <code>searchState</code> output reports the bracket, iteration count and whether the tolerance
(not the cap) stopped it (<code>GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SEARCH_STATE</code>). <code>refinement</code> is
ignored. Needs <code>maximumBandwidthCount >= 2</code> and <code>maximumNeighborCount >= 2</code>. Pinned to mgwr 2.2.1
on the 144-row scene (bisquare and Gaussian, adaptive and fixed): the adaptive <code>k</code> is mgwr's exactly,
fixed bandwidths agree within the AICc-flatness tolerance of the float32 search, and the float64 CPU
mirror of the state machine reproduces mgwr's evaluated bandwidths in order. mgwr rounds its fixed
answer to two decimals; this contributor returns the unrounded value.</p>
</li>
<li>
<p>Gaussian cutoff (new, <code>gaussianCutoff</code> in bandwidth multiples, compile-time, default exact
unbounded Gaussian): the weight of a row farther than <code>gaussianCutoff * h</code> is exactly 0 (fixed and
adaptive), and the Gaussian kernel then uses the grid index like bisquare, <code>O(n * neighbours)</code>
instead of <code>O(n^2)</code> (mgwr has no cutoff). The omitted weight of a row is at most <code>exp(-c^2 / 2)</code> of
the self weight: 1.1e-2 at <code>c = 3</code>, 3.4e-4 at 4, 3.7e-6 at 5, 1.5e-8 at 6. For a locally uniform
density the omitted share of the weight sum is that value and of the weighted second moment
<code>(1 + c^2 / 2) exp(-c^2 / 2)</code> (3.0e-3 at 4, 5.0e-5 at 5, 2.8e-7 at 6), which bounds the relative
effect on the local sums up to the local conditioning; on the test scene the coefficient error stays
below 20 times that share. Recommended: 5 (<code>GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_RECOMMENDED_GAUSSIAN_CUTOFF</code>),
where the effect is below the float32 accumulation error of the fits; use 4 when 3e-3 is acceptable
and 6 only to match the exact Gaussian to float32. The cutoff does not change the bisquare kernel.</p>
</li>
<li>
<p>Local inference (new): <code>standardErrors</code>, <code>tValues</code> and <code>significant</code> (all laid out like
<code>coefficients</code>) are mgwr <code>bse</code>, <code>tvalues</code> and <code>filter_tvals != 0</code>. <code>bse = sqrt(diag(C C') sigma2)</code>
with <code>C C' = (X'WX)^-1 X'W^2 X (X'WX)^-1</code> (one more weighted Gram matrix in the same neighbourhood
pass) and <code>sigma2 = RSS / (n - tr S)</code> (mgwr <code>sigma2_v1</code>). The summary carries <code>ENP = tr S</code>,
<code>SIGMA_SQUARED</code>, <code>LOG_LIKELIHOOD</code>, the Da Silva-Fotheringham adjusted alpha <code>alpha * k / tr S</code>
(<code>ADJUSTED_ALPHA</code>, <code>k</code> design columns, <code>alpha</code> 0.05 unless the per-frame <code>significanceLevel</code> says
otherwise) and the critical value <code>t.ppf(1 - adjusted / 2, n - 1)</code> (<code>CRITICAL_T</code>). The Student t
quantile is inverted on the GPU (continued-fraction incomplete beta with a bisection; the float64
CPU mirror <code>getStudentTCriticalValue</code> is pinned to <code>scipy.stats.t.ppf</code> to 1e-9 and the GPU summary is
checked against mgwr), because the mask needs it there.
<code>significant</code> is 1 where <code>|t| > CRITICAL_T</code>. The bandwidth is selected on the observed data and the
inference is conditional on it, as in mgwr. Pinned to mgwr 2.2.1 (fixed and adaptive, planar and
spherical); adaptive fits agree to about 5e-4 because mgwr scales the k-th neighbour distance by
1.0000001 and this contributor by 1.00001.</p>
</li>
<li>
<p><code>getGPUGeographicallyWeightedRegressionModelResult({summary, selectedBandwidth})</code> returns the shared
<code>GPURegressionModelResult</code> (<code>kind: 'geographically-weighted-regression'</code>, empty <code>coefficients</code>,
<code>parameterCount = tr S</code>, <code>rowCount = n</code>, <code>sigmaSquared = RSS / (n - tr S)</code>, <code>logLikelihood</code>,
<code>akaikeInformationCriterion</code> <code>= -2 llf + 2 (tr S + 1)</code> and <code>correctedAkaikeInformationCriterion</code>
(mgwr <code>llf</code>, <code>aic</code>, <code>aicc</code>), <code>rSquared</code>) plus <code>effectiveParameterCount</code>, <code>singularRowCount</code>,
<code>includedRowCount</code>, <code>adjustedAlpha</code>, <code>criticalT</code> and <code>selectedBandwidth</code>.</p>
</li>
<li>
<p><code>spatialContext</code> (new): omitted is planar. It measures distance, so <code>local-lambert-equal-area</code> frames throw. <code>{coordinateSpace: 'longitude-latitude', metric: 'great-circle', units: 'meters'}</code> makes <code>positions</code> <code>(longitude, latitude)</code> degrees and every
distance (kernels, fixed bandwidths, the adaptive k-th neighbour) a haversine distance on a sphere of
<code>sphereRadius</code> (default the mean Earth radius 6,371,008.8 m); bandwidths are then metres. Rows with a
latitude outside [-90, 90] are excluded. The grid index is planar-only and is not built for a
great-circle context, so every location scans every included row and the 65,536-row scan limit
applies; the cell walk is not made exact for longitude wrap-around. mgwr <code>spherical=True</code> uses a
6371.0 km radius and kilometres, so pass <code>sphereRadius: 6371000</code> and metres to match it.</p>
</li>
<li>
<p>Predictors: 1 to 11 (12 design columns; it was 7). No kernel binds more than eight storage buffers,
because the per-row results (including standard errors) share one row-state buffer; the limit is
per-thread private memory (<code>2 P^2 + 6 P</code> floats in the fit). The 11-predictor shader has not been run
on every adapter: lower <code>GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_PREDICTOR_COUNT</code> if one
rejects it.</p>
</li>
<li>
<p>Outputs: <code>coefficients</code> (<code>row * (predictorCount + 1) + column</code>, column 0 intercept), optional
<code>standardErrors</code>, <code>tValues</code>, <code>significant</code>, <code>localR2</code> (<code>1 - sum_j w_ij (y_j - x_j' beta_i)^2 / sum_j w_ij (y_j - ybar_i)^2</code>: the residuals are those of local model <code>i</code> at its neighbours <code>j</code>, not each neighbour's own
GWR residual, so a package that uses the latter gives different values), <code>fitted</code>, <code>residuals</code>, <code>hatDiagonal</code>,
<code>localStatus</code> (0 ok, 1 singular, 2 excluded), <code>bandwidthScores</code> (AICc of the caller ladder),
<code>bandwidthSingularRowCounts</code>, <code>candidateTable</code>, <code>selectedBandwidth</code> (<code>[index, value]</code>),
<code>localConditionNumber</code> and <code>summary</code> (<code>GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY</code>: RSS, <code>tr S</code>,
AICc, global R-squared, n, has-valid-candidate flag, singular rows, sigma2, adjusted alpha, critical
t, log-likelihood, included rows; 12 floats, it was 6).</p>
</li>
<li>
<p><code>localConditionNumber</code> is the mgwr <code>local_collinearity</code> local condition number: the singular-value
ratio of the kernel-weighted, column-normalized design (above 30 flags collinearity). It is pinned to
mgwr 2.2.1 (bisquare and gaussian, fixed and adaptive). The weighted design rows are folded into a
triangular factor by Givens rotations and its singular values come from one-sided Jacobi (the
helpers of <code>GPULocalCollinearity</code>), so the two classes report the same number to about
<code>float32 epsilon * condition</code>, also for a predictor with a large mean. The former Gram-matrix
route was 17% off for a predictor of 1500 +- 3 and meaningless for a projected northing.</p>
</li>
<li>
<p>Rows with a zero mask or a non-finite position, predictor or response are excluded as
calibration locations and as neighbours.</p>
</li>
<li>
<p>Numerics: the design is centred on the focal row and Jacobi-equilibrated before the shared
Cholesky solve, so large coordinates or badly scaled predictors stay well conditioned in f32.
A predictor that is constant over a neighbourhood makes that location singular.</p>
</li>
<li>
<p>Cost: one thread per location scanning every row per candidate: <code>O(n^2 * ladder * p^2)</code>, at most
65536 rows without the grid index (see <code>indexGridSize</code> below) (about 80 ms for 4096 rows and 8 candidates on the
test adapter). Reductions use fixed-order 256-row tiles, so results are bitwise reproducible.</p>
</li>
<li>
<p>Grid index: <code>indexGridSize</code> (default: automatic from 1024 rows; <code>false</code> disables) bins rows into a uniform grid so
each location scans nearby cells only. The row cap rises from 65,536 to 1,048,576 with the grid, and
<code>'bisquare'</code> fixed-bandwidth fits sum in cell order (about 1e-5 relative to the earlier row-order sums, still
deterministic). Gaussian kernels use the grid only with a <code>gaussianCutoff</code>.</p>
</li>
</ul>
<p>Decode <code>localStatus</code> with <code>decodeGPUGeographicallyWeightedRegressionStatus</code> and compare codes with
<code>GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_STATUS</code>. It reports each local weighted least squares
fit. A successful local fit does not establish bandwidth-search convergence: the golden-section
<code>searchState</code> reports that separately. MGWR has different row codes and a separate decoder for
the coupled backfit outcome; GWR code 1 means singular, while MGWR code 1 means excluded.
The parameter packer requires every supplied bandwidth to remain finite in float32 and an explicit
<code>significanceLevel</code> to remain strictly between 0 and 1 in float32. It rejects values that round to
0 or 1, and leaves a supplied target unchanged on validation failure. Only an omitted level encodes
the default sentinel 0. Bandwidths follow the <code>ArrayLike</code> indexed contract: the validated length
is read once, exactly that many indexed values are copied before any target write, and an attached
iterator cannot change the count or overwrite spare target capacity.</p>
<h4 id="how-it-computes-1">How it computes</h4>
<p>{/* lens:GPUGeographicallyWeightedRegression */}</p>
<p>::lens{scenario=gpu-spatial-analysis.geographically-weighted-regression rung=datum bind=page datum=300 caption="Inspect tract 300: its local coefficients, t value and local R squared."}
::lens{scenario=gpu-spatial-analysis.geographically-weighted-regression rung=formula caption="Follow the bisquare weight, local fit, hat value, AICc and local t equations."}
::lens{scenario=gpu-spatial-analysis.geographically-weighted-regression rung=kernel caption="Inspect the grid, candidate, golden-section and local-fit kernels with retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.geographically-weighted-regression rung=graph caption="Trace the golden-section search over 10 candidate bandwidths and the final local fits over 788 tracts."}</p>
<p>{/* parameters: GPUGeographicallyWeightedRegression generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpugeographicallyweightedregression-parameters"><code>GPUGeographicallyWeightedRegression</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUGeographicallyWeightedRegressionParameterValues(settings, maximumBandwidthCount?)</code>, writes <code>getGPUGeographicallyWeightedRegressionParameterLength(...)</code> <code>float32</code> values.
<ul>
<li><code>maximumBandwidthCount</code> (<code>number</code>): Ladder capacity of the contributor (its <code>maximumBandwidthCount</code>).</li>
</ul>
</li>
</ul>
<p>Settings of <code>getGPUGeographicallyWeightedRegressionParameterValues</code>:</p>
<table>
<thead>
<tr>
<th>Field</th>
<th>Type</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>spatialContext</code> (optional)</td>
<td><code>GPUSpatialContext</code></td>
<td>Spatial context of the positions; when it is longitude/latitude, a distance under 10 float32 quanta (about 17 m) logs a warning (see Precision of longitude/latitude positions). Does not change the packed values.</td>
</tr>
<tr>
<td><code>kernel</code> (optional)</td>
<td><code>GPUGeographicallyWeightedRegressionKernel</code></td>
<td>Weight kernel. Defaults to <code>'bisquare'</code>.</td>
</tr>
<tr>
<td><code>bandwidthMode</code> (optional)</td>
<td><code>GPUGeographicallyWeightedRegressionBandwidthMode</code></td>
<td>Meaning of the ladder values. Defaults to <code>'fixed'</code>.</td>
</tr>
<tr>
<td><code>bandwidths</code></td>
<td><code>ArrayLike&#x3C;number></code></td>
<td>Candidate bandwidths: distances (<code>'fixed'</code>) or neighbour counts <code>k</code> (<code>'adaptive'</code>, rounded to integers). One value is a fixed bandwidth; several are searched by AICc (and refined, see <code>refinementPassCount</code>). With <code>bandwidthSearch: 'golden-section'</code> the first two values are the search bounds <code>[minimum, maximum]</code> (a value of 0 or less takes mgwr's default for that bound); further values are ignored.</td>
</tr>
<tr>
<td><code>significanceLevel</code> (optional)</td>
<td><code>number</code></td>
<td>Two-sided significance level of the adjusted alpha and critical t value. Its float32 representation must be in (0, 1). Defaults to 0.05. Per frame.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="gpugeographicallyweightedregressionnonstationaritytest"><code>GPUGeographicallyWeightedRegressionNonstationarityTest</code></h3>
<p>Tests whether each GWR coefficient varies across space more than permuted data would, with a Monte Carlo pseudo p-value per coefficient.</p>
<p>::lens{scenario=gpu-spatial-analysis.geographically-weighted-regression-nonstationarity-test rung=result caption="Observed against permuted spread of four coefficients; poverty pseudo p = 0.03, the other three 0.01."}</p>
<p>Monte Carlo test of coefficient non-stationarity (mgwr <code>spatial_variability</code>, GWR4). It permutes the
observations among the locations with the Philox-keyed swap-or-not shuffle of <code>GPUGlobalPermutationTest</code>
(the earlier Feistel bijection was non-uniform on small row counts), refits
at the selected bandwidth and compares the standard deviation of each local coefficient with the observed
one. The pseudo p-value is <code>(g + 1) / (P + 1)</code>. Outputs are <code>table</code> (six floats per coefficient), <code>summary</code>
(<code>GPU_GWR_NONSTATIONARITY_SUMMARY</code>, <code>GPU_GWR_NONSTATIONARITY_TABLE</code>, <code>GPU_GWR_NONSTATIONARITY_TABLE_STRIDE</code>)
and an optional <code>standardDeviations</code> (the reference distribution). The per-permutation standard deviations
match mgwr 2.2.1 to 3e-15 (CPU oracle) and 2e-3 (GPU). The bandwidth stays at the value the regression
selected (mgwr re-selects it for every permutation), and that value was selected on the observed data, so
the test is conditional on it, as mgwr's is. <code>selectedBandwidth</code> is read from slot 1 (the selected value),
so it also works when the regression refined its bandwidth beyond the caller ladder. It is planar-only
(a great-circle <code>spatialContext</code> throws, with a hint to project first) and takes 1 to 7 predictors even
though the regression accepts 11. The cost is <code>O(P * n^2 * p^2)</code> with a full scan (no grid index).</p>
<h4 id="how-it-computes-2">How it computes</h4>
<p>{/* lens:GPUGeographicallyWeightedRegressionNonstationarityTest */}</p>
<p>::lens{scenario=gpu-spatial-analysis.geographically-weighted-regression-nonstationarity-test rung=formula caption="Follow the observed spread, exceedance count, pseudo p-value and permuted z score."}
::lens{scenario=gpu-spatial-analysis.geographically-weighted-regression-nonstationarity-test rung=kernel caption="Inspect the permuted refit, spread merge and finish kernels with retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.geographically-weighted-regression-nonstationarity-test rung=graph caption="Trace the GWR fit, 99 permuted refits and the finish step across 788 tracts."}</p>
<p>{/* parameters: GPUGeographicallyWeightedRegressionNonstationarityTest generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpugeographicallyweightedregressionnonstationaritytest-parameters"><code>GPUGeographicallyWeightedRegressionNonstationarityTest</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUGeographicallyWeightedRegressionParameterValues(settings, maximumBandwidthCount?)</code>, writes <code>getGPUGeographicallyWeightedRegressionParameterLength(...)</code> <code>float32</code> values.
<ul>
<li>Input port: <code>bandwidthParameters</code>.</li>
<li><code>maximumBandwidthCount</code> (<code>number</code>): Ladder capacity of the contributor (its <code>maximumBandwidthCount</code>).</li>
</ul>
</li>
<li>Packer: <code>getGPUPermutationParameterValues(settings, options?)</code>, writes <code>GPU_PERMUTATION_PARAMETER_LENGTH</code> (4) <code>uint32</code> values.
<ul>
<li>Input port: <code>parameters</code>.</li>
<li><code>options</code> (<code>{maximumPermutations?: number}</code>)</li>
</ul>
</li>
</ul>
<p>Settings of <code>getGPUGeographicallyWeightedRegressionParameterValues</code>:</p>
<table>
<thead>
<tr>
<th>Field</th>
<th>Type</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>spatialContext</code> (optional)</td>
<td><code>GPUSpatialContext</code></td>
<td>Spatial context of the positions; when it is longitude/latitude, a distance under 10 float32 quanta (about 17 m) logs a warning (see Precision of longitude/latitude positions). Does not change the packed values.</td>
</tr>
<tr>
<td><code>kernel</code> (optional)</td>
<td><code>GPUGeographicallyWeightedRegressionKernel</code></td>
<td>Weight kernel. Defaults to <code>'bisquare'</code>.</td>
</tr>
<tr>
<td><code>bandwidthMode</code> (optional)</td>
<td><code>GPUGeographicallyWeightedRegressionBandwidthMode</code></td>
<td>Meaning of the ladder values. Defaults to <code>'fixed'</code>.</td>
</tr>
<tr>
<td><code>bandwidths</code></td>
<td><code>ArrayLike&#x3C;number></code></td>
<td>Candidate bandwidths: distances (<code>'fixed'</code>) or neighbour counts <code>k</code> (<code>'adaptive'</code>, rounded to integers). One value is a fixed bandwidth; several are searched by AICc (and refined, see <code>refinementPassCount</code>). With <code>bandwidthSearch: 'golden-section'</code> the first two values are the search bounds <code>[minimum, maximum]</code> (a value of 0 or less takes mgwr's default for that bound); further values are ignored.</td>
</tr>
<tr>
<td><code>significanceLevel</code> (optional)</td>
<td><code>number</code></td>
<td>Two-sided significance level of the adjusted alpha and critical t value. Its float32 representation must be in (0, 1). Defaults to 0.05. Per frame.</td>
</tr>
</tbody>
</table>
<p>Settings of <code>getGPUPermutationParameterValues</code>:</p>
<table>
<thead>
<tr>
<th>Field</th>
<th>Type</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>seed</code></td>
<td><code>number</code></td>
<td>Random seed, an integer in <code>[0, 2^53)</code>. Results are a pure function of the seed, the inputs and <code>permutationCount</code>, independent of the dispatch shape. Per frame.</td>
</tr>
<tr>
<td><code>permutationCount</code></td>
<td><code>number</code></td>
<td>Number of permutations <code>P</code>, in <code>[1, maximumPermutations]</code> of the contributor. The count is never clamped: a value outside that range makes the kernels write NaN p-values (local: rows are not tested and <code>output.status.overflow</code> is set; global: the <code>permutationCount</code> field of <code>output.summary</code> is NaN). Validate on the CPU with the contributor's <code>getParameterValues</code> or the <code>maximumPermutations</code> option of <code>getGPUPermutationParameterValues</code>.</td>
</tr>
<tr>
<td><code>significanceLevel</code> (optional)</td>
<td><code>number</code></td>
<td>Significance level for the local <code>significant</code> mask. Defaults to 0.05. Ignored by the global test.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="gpuspatialregressiondiagnostics"><code>GPUSpatialRegressionDiagnostics</code></h3>
<p>Tests an ordinary least squares fit for spatial dependence with LM-lag, LM-error, robust, SARMA and residual Moran's I tests.</p>
<p>::lens{scenario=gpu-spatial-analysis.spatial-regression-diagnostics rung=result caption="Six tests on the diabetes OLS residuals: LM-lag 638 against LM-error 595, then robust 170 against 127."}</p>
<p>Maximum: <code>rows * (predictorCount + 3)</code> is at most <code>getGPUSpatialRegressionDiagnosticsMaximumRowTableValueCount(device)</code> (2^25 on a default device); the per-tile moment table is at most <code>getGPUSpatialRegressionDiagnosticsMaximumTileMomentCount(device)</code> values.</p>
<p>Spatial diagnostics for an OLS fit (spreg <code>LMtests</code> and <code>MoranRes</code>): LM-lag, LM-error, robust LM-lag,
robust LM-error, LM-SARMA, and Moran's I of the residuals (mean, variance, z, two-sided normal p).</p>
<ul>
<li>Props: <code>weights</code> (a square <code>GPUSpatialWeights</code>; any sparsity pattern is accepted, directed kNN included,
and row-standardized weights work; <code>W'</code> is formed with <code>GPUSpatialWeightsTranspose</code>),
<code>predictors</code> (row-major <code>n x k</code>, no intercept), <code>response</code>, <code>residuals</code> (OLS residuals with intercept,
for example <code>GPUOrdinaryLeastSquares</code> <code>output.residuals</code>), <code>predictorCount</code> (1 to 15), optional <code>mask</code>,
<code>restandardizeRows</code>, <code>tileRowCount</code> and <code>output: {tests, summary, status: {code}}</code>.</li>
<li>Masked and missing fits: a row is analyzed when its <code>mask</code> entry is not zero and its response, residual
and predictors are finite; pass the OLS <code>mask</code> and the NaN residuals of excluded rows do the rest. Excluded
rows are removed from the whole analysis: <code>n</code>, <code>S0</code>, <code>T</code>, every Gram matrix and every <code>W</code>-product run over
the fitted rows, and links from fitted rows to excluded rows are dropped. That is <code>libpysal.weights.w_subset</code>
on the fitted ids. With <code>restandardizeRows: true</code> each fitted row's remaining weights are rescaled to sum
to 1 (<code>w_subset</code> then <code>transform = 'r'</code>; use it after masking row-standardized weights); the default
keeps the surviving weights as given. Rows left with no fitted neighbor keep an all-zero row. Results
match spreg 1.9.1 <code>OLS(..., spat_diag=True, moran=True)</code> on the fitted subset in both modes.</li>
<li><code>tests</code> (18 f32): six rows of <code>[statistic, df, p]</code>: LM_LAG, LM_ERROR, ROBUST_LM_LAG, ROBUST_LM_ERROR
(df 1), LM_SARMA (df 2) and MORAN_RESIDUALS (the statistic is z, the df slot is 0, p is two-sided). The
row index constants are <code>GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TEST_*</code>.</li>
<li><code>summary</code> (13 f32): <code>n</code> (analyzed rows), <code>sigma2 = e'e / n</code>, <code>T = tr(W'W + WW)</code>, <code>J</code>, <code>S0</code>, Moran <code>I</code>,
<code>E[I]</code>, <code>Var[I]</code>, <code>z</code>, <code>p</code>, <code>e'We</code>, <code>e'Wy</code> and <code>EXCLUDED_ROW_COUNT</code> (<code>..._SUMMARY_*</code>). The excluded
count is a summary slot, not a status port, because <code>status.code</code> is this operator's single result code; it is
also written when the status is not 0. Before this change a masked fit reported status 4 for the whole
result.</li>
<li><code>status.code</code>: 0 ok, 1 singular, 2 too few analyzed rows (<code>n &#x3C;= predictorCount + 2</code>, also a failed OLS fit; OLS
itself fits <code>n = predictorCount + 2</code> rows, so on exactly that many rows the fit succeeds and the diagnostics do not),
3 incomplete weights (<code>weights.status.overflow</code> or <code>candidateOverflow</code>) or degenerate weights,
4 non-finite moments or exactly zero residuals. Incomplete weights suppress every test statistic
and p-value, and the recipe returns specification decision 4 (undetermined). Both operation forms
accept the optional weights status inputs and recover when the flags clear. Failed tests have NaN
statistics and p-values; degrees of freedom and analyzed/excluded row counts remain defined.
Conditioning slots are not reported here (see the OLS fit).</li>
</ul>
<p>Diagnostics and both operation forms capture the declared input and output maps, including nested
weights/status fields and inherited getters, at construction. Mutating the caller's maps later
cannot remove a loss input or redirect the validated output; buffer contents remain per-run data.</p>
<p>Sums are fixed-order tile sums with a pairwise merge on centered columns, and the solve is a unit-diagonal
scaled Cholesky, so there are no atomics and no readback. Moran's variance and <code>J</code> are f32 with
cancellation (about 1e-2 relative on the variance and z). The statistics match spreg 1.9.1 (<code>OLS</code> with
<code>spat_diag</code> and <code>moran</code>) on a lattice, a symmetrized kNN scene and two directed row-standardized kNN scenes
(<code>k = 3</code>, 60 and 90 rows, asymmetric patterns) to 2e-3 relative. Earlier versions accumulated <code>W'Z</code> over each
row's own neighbors only, which biased the Moran variance for asymmetric patterns; that is fixed.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUOrdinaryLeastSquares</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({predictors, response, mask, predictorCount: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, output: {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  coefficients, standardErrors, tStatistics, summary: olsSummary, residuals, status: {code: olsCode}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}}));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUSpatialRegressionDiagnostics</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  weights, predictors, response, residuals, mask, restandardizeRows: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, predictorCount: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {tests, summary, status: {code}}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span></code></pre>
<h4 id="how-it-computes-3">How it computes</h4>
<p>{/* lens:GPUSpatialRegressionDiagnostics */}</p>
<p>::lens{scenario=gpu-spatial-analysis.spatial-regression-diagnostics rung=formula caption="Follow the LM-error, LM-lag, robust, SARMA and Moran's I equations."}
::lens{scenario=gpu-spatial-analysis.spatial-regression-diagnostics rung=kernel caption="Inspect the tile, merge and finish kernels with retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.spatial-regression-diagnostics rung=graph caption="Trace queen weights, the OLS fit and the six diagnostics over 788 tracts."}</p>
<h3 id="weight-completeness-in-spatial-model-fits">Weight completeness in spatial model fits</h3>
<p><code>GPUSpatialErrorGM</code>, <code>GPUSpatialTwoStageLeastSquares</code>, <code>GPUSpatialWeightsTraces</code>,
<code>GPUSpatialLagMaximumLikelihood</code> and <code>GPUSpatialErrorMaximumLikelihood</code> read the supplied
weights' <code>status.overflow</code> and <code>status.candidateOverflow</code> on every submission. For chunked
weights they read both the root status and each chunk's status. If any loss word is nonzero,
all dependent float outputs are NaN; fitted models also publish <code>status.code = 3</code> (unusable,
non-finite/suppressed result) and clear <code>status.converged</code> when present. Traces suppress
<code>infinityNorm</code>, powers, sums, standard errors and Gram traces; <code>approximation</code> retains its
sampling meaning. A later clean submission recomputes usable outputs on the same graph.</p>
<p>The GPU named operation forms and CPU twins currently require a complete CSR without a
<code>weights.status</code> map. They reject status-bearing weights explicitly, including chunk statuses,
because those forms have no loss-flag input ports. Use the direct GPU contributors to consume
producer ports with per-submission loss flags. Do not remove the flags from a possibly incomplete port.
The spatial-regression recipe uses the separate diagnostics contributor, which already handles
its own weights loss flags.</p>
<p>These five contributors capture declared props (including inherited getters), nested output/status
maps, weights maps and chunk boundaries at construction. Later edits to those maps do not move
validated bindings; GPU buffer contents remain per-frame inputs.</p>
<p>The five regression lens scenarios capture both <code>weightsOverflow</code> and the contiguity producer's
<code>weightsCandidateOverflow</code>. The readback-to-record boundary refuses either loss word even when
the other is zero, so an incomplete adjacency cannot become a recorded result.</p>
<h3 id="gpuspatialtwostageleastsquares"><code>GPUSpatialTwoStageLeastSquares</code></h3>
<p>Estimates a spatial lag model by two-stage least squares, instrumenting the lagged response with neighbouring predictors, and tests the residuals.</p>
<p>::lens{scenario=gpu-spatial-analysis.spatial-two-stage-least-squares rung=result caption="Five rows: intercept, three predictors and rho = 0.55 (se 0.027); compare each with its standard error."}</p>
<p>The spatial lag model <code>y = rho W y + X b + e</code> by two-stage least squares (spreg <code>GM_Lag</code> with <code>w_lags = 1</code>),
with instruments <code>[1, X, WX]</code>. Props: <code>weights</code>, <code>predictors</code>, <code>response</code>, <code>predictorCount</code> (1 to 8 with
<code>instrumentOrder: 1</code>, 1 to 5 with <code>instrumentOrder: 2</code>: the constant is centered out, so the solve factors <code>(order + 1) k</code>
instrument columns in a 16-wide Cholesky; the order-2 cap was previously 8 in the check but failed when building the
graph), optional <code>tileRowCount</code>, <code>instrumentOrder</code>, <code>robust</code> and <code>output: {table, summary, status: {code}, residuals?}</code>.
Construction requires <code>tileCount = ceil(rows / tileRowCount) &#x3C;= 4096</code> for the serial norm merge.
Increase <code>tileRowCount</code> if an explicit layout exceeds this limit; the default layout still reaches <code>2^24</code> rows.
<code>GPUSpatialTwoStageLeastSquaresOperation</code> applies this check when lowered to its GPU contributor.
The CPU twin ignores GPU tiling and does not impose this tile-count cap.
No kernel binds more than 8 storage buffers.
<code>robust: {method: 'HC0' | 'HC1' | 'HC2' | 'HC3' | 'CLUSTER' | 'HAC', kernel?, clusters?, clusterCount?, output: {covariance, status}}</code> replaces the standard errors, z
statistics and p-values of <code>table</code> by those of a heteroskedasticity-consistent, one-way clustered CR1 or spatial HAC covariance
(<code>GPUInstrumentalVariablesCovariance</code>; HAC needs <code>kernel</code>, a symmetric positive semidefinite weights CSR with a unit
diagonal, distinct from the model weights) and writes the full <code>(k + 2) x (k + 2)</code> covariance and its own status.
The <a href="/luma-contributors/docs/api/gpu-spatial-analysis/instrumental-variables-covariance">instrumental-variable covariance guide</a> defines the
HC corrections, projected-design leverage, cluster-key limits, PSD certificate and every solver
status for the direct contributor and both operation forms.
The point estimates and the <code>summary</code> diagnostics keep their classical definitions. The displayed
lens and its classical formula retain the default without <code>robust</code>. Its recorded <code>robust.method</code>
choice is <code>none</code>, explicitly distinguishing classical covariance from the optional robust methods.
<code>instrumentOrder: 2</code> (compile-time, default 1) adds <code>W^2 X</code> (spreg <code>w_lags=2</code>, instruments <code>[1, X, WX, W^2 X]</code>), evaluated per row from the two-hop neighborhood with no extra buffers; it is pinned to spreg
1.9.1 <code>GM_Lag(w_lags=2)</code> on three scenes.</p>
<ul>
<li><code>table</code>: <code>(k + 2) x 4</code> f32 rows of <code>[coefficient, se, z, two-sided normal p]</code> in spreg order: the
intercept, then <code>X</code>, then <code>rho</code> last.</li>
<li><code>summary</code> (16, <code>GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_*</code>): <code>n</code>, <code>sigma2 = u'u / n</code> (spreg <code>sig2n</code>, the <code>GM_Lag</code>
default), <code>u'u</code>, the pseudo R2 (squared correlation of <code>y</code> and <code>y - u</code>), Moran's <code>I</code> of <code>u</code>, the
Anselin-Kelejian statistic and its p-value (slots 0 to 6). The statistic follows spreg <code>akTest(case='gen')</code>; any
sparsity pattern is accepted (<code>W'u</code> comes from <code>GPUSpatialWeightsTranspose</code>), and it matches spreg 1.9.1 on
directed kNN scenes. The total weight <code>S0</code> cancels from the statistic, which is computed as
<code>(u'Wu / sigma2)^2 / (T + 4 a / sigma2)</code>: for signed weights whose total is zero the statistic is defined
(0.95057 on an eight-row scene with weights of +1 and -1, as the cancelled formula on spreg's fit; it was
NaN), while Moran's <code>I</code>, which divides by <code>S0</code>, is NaN. Diagnostics (slots 7 to 15; none changes <code>status</code>):
<ul>
<li>Conditioning of the second stage: the Belsley condition number of the raw <code>[1, PZ]</code> Gram matrix (<code>Z = [X, Wy]</code>,
<code>P</code> the projection on the instruments), the correlation condition number of <code>Z'PZ</code> and its largest variance
inflation factor, the correlation condition number of the instrument system <code>H'H</code> (the instrument-projection
system), and a precision warning (1.0) when either correlation condition number exceeds 1e4 (float32
normal equations may have lost three of seven digits). Computed by the Jacobi eigenvalue routine of
<code>spatial-regression-solve</code>.</li>
<li>First-stage strength: the classical joint F statistic and the partial R2 of the excluded instruments (<code>WX</code>, and
<code>W^2 X</code> for order 2) in the regression of <code>Wy</code> on <code>[1, X, WX, ...]</code>; <code>F = ((ESS_full - ESS_restricted) / (m - k)) / (RSS_full / (n - m - 1))</code>, partial R2 <code>= (ESS_full - ESS_restricted) / (TSS - ESS_restricted)</code>. The usual
weak-instrument rule of thumb is <code>F &#x3C; 10</code>. Both are the non-robust (homoskedastic) forms and are pinned to a numpy
computation of the same definitions.</li>
<li>Admissibility of <code>rho</code>: <code>||W||_inf</code> (largest absolute row sum) and a flag that is 1.0 when <code>|rho| ||W||_inf >= 1</code>,
that is when <code>I - rho W</code> is not guaranteed invertible and the reduced form and impacts are unreliable
(<code>(-1, 1)</code> for a row-standardized <code>W</code>; sufficient, not necessary, for other <code>W</code>). It is a flag, not a failure:
2SLS is unconstrained, the estimate is still reported, and the reader decides what a boundary or explosive <code>rho</code>
means for the model.</li>
</ul>
</li>
<li><code>status.code</code>: 0 ok, 1 singular or ill-conditioned (including non-finite inputs, and a predictor that is constant
at float32 resolution, as for the other estimators), 2 too few rows, 3 non-finite sums or results suppressed for incomplete weights.</li>
<li><code>residuals</code>: the structural residuals <code>u = y - Z delta</code>, which can be passed to later diagnostics. They are
formed on the centered columns and sum to zero: the part of each mean that float32 cannot hold is carried
through the fit, and the moments are corrected to moments about the true means.</li>
<li>The lag of the response is formed about the response mean, <code>W y = W (y - c) + c s_i</code>, with the row sums <code>s_i</code>
of the float32 weights carried exactly (rows without neighbours included); a sum of <code>w * y</code> over the raw
response is rounded to a float32 step of the response, which cost up to 4.2% of <code>sigma2</code> and 45% of the
Anselin-Kelejian statistic for residuals of 0.002 to 0.004 around 10,000.</li>
<li>One step of iterative refinement follows the solve (three more nodes). The float32 solve lands a few float32
steps of <code>rho</code> from the estimate. Where the lag differs between rows by the response mean (rows without
neighbours, row sums that differ) the two-stage <code>rho</code> is not at the minimum of <code>u'u</code>, and three and five such
steps were 3% and 0.8% of <code>sigma2</code> (the float32 value nearest to the estimate would have cost 0.07% and
0.03%: it was the solve, not the resolution of <code>rho</code>). The refinement forms the residuals at the float32
coefficients without the rounding of their products, sums <code>H'u</code> and solves the same system for the
correction. The correction is kept beside the float32 coefficients: <code>residuals</code>, <code>sigma2</code>, the standard
errors and the Anselin-Kelejian statistic are those of the corrected estimate, and <code>table</code> holds the float32
nearest to it.</li>
<li>Measured on the test GPU against spreg <code>GM_Lag</code> and a centered float64 solve, for residuals of 0.002 to
0.004 around 10,000:
<ul>
<li>rows that share a row sum, exactly or up to the rounding of the weights (ring at 0.5, six neighbours at
float32 1/6, binary weights with row sum 2, row sums of 1 and 1.0000000298, <code>rho</code> 0.2 to 0.8, and a second
predictor near 4,096): <code>sigma2</code> within 2.8e-6, the Anselin-Kelejian statistic within 2.4e-4, <code>rho</code> within
6e-4 standard errors;</li>
<li>row sums that differ with such a response: three rows without neighbours, <code>sigma2</code> within 9.2e-6 and the
statistic within 9.1e-6; row sums of 2 and 3, 9.5e-7 and 4.9e-7;</li>
<li>slopes two hundred times the residual deviation over the predictor range: <code>sigma2</code> within 1.6e-4, because
the fitted values are rounded to about 1% of a residual.
<code>W X</code> (the instruments) is still summed from the raw predictors.</li>
</ul>
</li>
</ul>
<p>The coefficients, standard errors, <code>sigma2</code>, pseudo R2 and Anselin-Kelejian statistic match spreg on a
144-row scene within 5e-3 (coefficients) to 2e-2 (z, AK). The default standard errors are the classical
<code>sigma2 (Z'PZ)^-1</code>; the <code>robust</code> prop gives HC0/HC1/HC2/HC3, clustered CR1 or spatial HAC ones.
<code>GM_Combo</code> and maximum-likelihood models are not provided here. For the spatial error model see
<a href="#gpuspatialerrorgm"><code>GPUSpatialErrorGM</code></a>.</p>
<p><code>getGPUSpatialTwoStageLeastSquaresModelResult(readback, {predictorCount?, predictorNames?})</code> turns the read-back <code>table</code>, <code>summary</code>
and <code>status</code> into the shared <code>GPURegressionModelResult</code>: rows <code>CONSTANT</code>, the predictors and <code>W_y</code> (role
<code>'spatial-lag'</code>), z statistics with float64 normal p-values, <code>sigmaSquaredMaximumLikelihood = u'u / n</code> (the scale
of the standard errors, <code>standardErrorSigmaSquared: 'maximum-likelihood'</code>), <code>pseudoRSquared</code>, no likelihood,
<code>tests.anselinKelejian</code> and conditioning (the correlation condition number is the larger of the second-stage
and instrument values, so <code>illConditioned</code> equals the precision warning).
<code>getGPUSpatialTwoStageLeastSquaresDiagnostics(summary)</code> returns the first-stage F, partial R2, <code>||W||_inf</code> and the
<code>rho</code> admissibility flag.
Pass the fitted <code>predictorCount</code> when the read-back table has spare capacity; only the active
<code>(predictorCount + 2) * 4</code> prefix is decoded. Omit it only for an exact-length table. Predictor
names label columns and do not determine the fitted count.</p>
<h4 id="how-it-computes-4">How it computes</h4>
<p>{/* lens:GPUSpatialTwoStageLeastSquares */}</p>
<p>::lens{scenario=gpu-spatial-analysis.spatial-two-stage-least-squares rung=formula caption="Follow the instrumented estimator, residual variance, z statistics and Anselin-Kelejian test."}
::lens{scenario=gpu-spatial-analysis.spatial-two-stage-least-squares rung=kernel caption="Inspect the moment, solve, residual and finish kernels with retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.spatial-two-stage-least-squares rung=graph caption="Trace queen weights, the two stages and the Anselin-Kelejian test over 788 tracts."}</p>
<h3 id="gpuspatialweightstraces"><code>GPUSpatialWeightsTraces</code></h3>
<p>Computes the traces and sums of powers of a weights matrix, the inputs of spatial log-determinants and impacts.</p>
<p>::lens{scenario=gpu-spatial-analysis.spatial-weights-traces rung=result caption="Traces of 12 queen-weight powers fall from 122 at q = 2 to 19; sums stay at 788."}</p>
<p>Powers of a spatial weights matrix: <code>powerTraces[q-1] = tr((W/s)^q)</code> and <code>powerSums[q-1] = 1' (W/s)^q 1</code> with <code>s = ||W||_inf</code> (largest absolute row sum, written to <code>output.infinityNorm</code>; series then run in <code>rho * s</code>, which keeps the float32 powers bounded for raw weights) for
<code>q = 1..Q</code>. These are the inputs of the spatial-lag log-determinant <code>ln|I - rho W| = -sum_q rho^q tr(W^q) / q</code>
and of the impact multipliers (see <code>GPUSpatialLagImpacts</code>). Props: <code>weights</code> (any square CSR: asymmetric,
not standardized, self-loops allowed), <code>maximumPower</code> (<code>Q</code>, compile-time, 1 to 256), optional <code>mode</code>,
<code>probeCount</code>, <code>parameters</code> (per-frame <code>[seed]</code>, <code>getGPUSpatialWeightsTracesParameterValues({seed})</code>; omitted means seed 0),
<code>includeGramTraces</code>, <code>tileRowCount</code>, <code>probeBatchSize</code> and
<code>output: {powerTraces, powerSums, gramTraces?, status?: {approximation?}, infinityNorm?}</code> (float32 views of at least <code>Q</code>, <code>Q</code>, 4 values
and a uint32 scalar).
Construction requires at most 4096 tiles for the serial norm merge, in both exact and stochastic modes.
For whole weights the count is <code>ceil(rows / tileRowCount)</code>; for chunked weights it is the sum of
<code>ceil(chunk.rowCount / tileRowCount)</code> over chunks. Increase <code>tileRowCount</code> or combine small chunks
if the count exceeds 4096. The default whole-CSR layout targets 1024 tiles and still reaches <code>2^24</code> rows
in stochastic mode. Chunk boundaries can exceed the cap even with the default tile width.
<code>GPUSpatialWeightsTracesOperation</code> applies the same check during lowering, including chunked inputs.
The CPU twin ignores GPU tiling and does not impose this tile-count cap.
The contributor captures chunk boundaries and nested weight-view references at construction;
later edits to the caller's chunk array or port maps do not change the validated dispatch layout.
GPU buffer contents remain per-frame inputs.</p>
<ul>
<li><code>mode: 'exact'</code> runs one unit-vector chain <code>v_q = W v_{q-1}</code> per row and sums the diagonal entries
<code>v_q[i]</code>: exact up to float32 rounding. It is the default for at most 4096 rows
(<code>GPU_SPATIAL_WEIGHTS_TRACES_MAXIMUM_EXACT_ROW_COUNT</code>): the work is <code>rows * Q * nnz</code>, which at 4096 rows,
<code>Q = 256</code> and 8 neighbors is about 3e10 multiply-adds in about 520 dispatches, while 16384 rows (the limit
of an explicit <code>'exact'</code>) already costs 8200 dispatches and 5e11 multiply-adds.</li>
<li><code>mode: 'stochastic'</code> is Hutchinson estimation with <code>probeCount</code> (default 64) Rademacher probes from Philox
(counter <code>(probe, row / 4, 0, 'PROB')</code>, key <code>(seed, 'TRAC')</code>, the seed read on the GPU, so changing it never
recompiles; a seeded run is reproducible and
the test-only CPU reference in <code>modules/analysis/test/reference</code> replays the same probes). The standard error of <code>tr(A)</code> is at most
<code>sqrt(2 / probeCount) ||A||_F</code>. <code>output.status.approximation</code> is 1 (it is 0 in exact mode).</li>
<li><code>powerSums</code> is always exact: one chain from the ones vector.</li>
<li><code>gramTraces</code> (when <code>includeGramTraces</code>, 4 values) are <code>[tr(W'W), tr(W'W W), tr((W'W)^2), sum_i c_i^2]</code> with
<code>c_i = sum_l w_li^2</code>: the moments of the Kelejian-Prucha and Anselin-Kelejian variance formulas. <code>tr(W'W)</code> and
<code>sum c_i^2</code> are exact sums (the column sums use <code>GPUSpatialWeightsTranspose</code>); the other two are
<code>E[(Wz)'(WWz)]</code> and <code>E[||W'(Wz)||^2]</code> over the same probes, exact in exact mode.</li>
</ul>
<p>Probes run in batches whose two ping-pong vectors stay within 64 MiB. Every dot product is a fixed-order tile
sum with a pairwise merge, so there are no float atomics and results are bit-identical across batch sizes and
runs. The shaders do not depend on the power, the batch or the probe index: per-power and per-batch tables
are 64-element aligned sub-views and a GPU batch table holds each batch's first probe and probe count, so
the pipeline count does not grow with <code>maximumPower</code> or the number of batches. A <code>W</code> with spectral radius above 1 grows as <code>radius^q</code> and can overflow float32 for large <code>Q</code>.</p>
<p>CPU float64 helpers: <code>getSpatialLogDeterminantFromTraces(rho, powerTraces)</code>, <code>getSpatialMultiplierTraceFromTraces(rho, powerTraces, n)</code> (<code>= tr((I - rho W)^-1) = n + sum_q rho^q tr(W^q)</code>) and the truncation bounds
<code>getSpatialLogDeterminantTruncationBound</code> and <code>getSpatialMultiplierTruncationBound</code>. WGSL generators
<code>getSpatialLogDeterminantWGSL(Q, bindingName)</code>, <code>getSpatialMultiplierTraceWGSL(Q, bindingName, n)</code> and
<code>getSpatialMultiplierTotalWGSL(Q, bindingName, n)</code> define <code>getSpatialLogDeterminant(rho)</code> and friends by
Horner's rule over an f32 storage binding. The series require <code>|rho| * spectralRadius(W) &#x3C; 1</code> (<code>|rho| &#x3C; 1</code> for
a row-standardized <code>W</code>) and are truncated: the log-determinant error is below
<code>n |rho|^(Q+1) / ((Q+1)(1 - |rho|))</code> and the multiplier error below <code>n |rho|^(Q+1) / (1 - |rho|)</code>, so choose <code>Q</code>
from <code>|rho|^Q</code> (<code>Q = 30</code> is accurate to 1e-9 at <code>|rho| = 0.5</code> but needs about 300 terms at <code>|rho| = 0.9</code>).</p>
<p>The traces match <code>np.trace(np.linalg.matrix_power(W, q))</code> for <code>q = 1..12</code> on a row-standardized lattice, a
binary lattice and two directed kNN scenes (numpy 2.5.4), and the log-determinant series matches
<code>np.linalg.slogdet</code> within the stated bound.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUSpatialWeightsTraces</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  weights, maximumPower: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">30</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, output: {powerTraces, powerSums, infinityNorm}   </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// exact for &#x3C;= 4096 rows</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span></code></pre>
<h4 id="how-it-computes-5">How it computes</h4>
<p>{/* lens:GPUSpatialWeightsTraces */}</p>
<p>::lens{scenario=gpu-spatial-analysis.spatial-weights-traces rung=formula caption="Follow the infinity norm, the power traces and the power sums."}
::lens{scenario=gpu-spatial-analysis.spatial-weights-traces rung=kernel caption="Inspect the norm, scaling, probe-chain and merge kernels with retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.spatial-weights-traces rung=graph caption="Trace queen weights, 788 unit-vector chains and 12 power steps."}</p>
<p>{/* parameters: GPUSpatialWeightsTraces generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpuspatialweightstraces-parameters"><code>GPUSpatialWeightsTraces</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUSpatialWeightsTracesParameterValues(settings?)</code>, writes <code>GPU_SPATIAL_WEIGHTS_TRACES_PARAMETER_LENGTH</code> (1) <code>uint32</code> values.</li>
</ul>
<p>Settings of <code>getGPUSpatialWeightsTracesParameterValues</code>:</p>
<table>
<thead>
<tr>
<th>Field</th>
<th>Type</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>seed</code> (optional)</td>
<td><code>number</code></td>
<td>Philox seed of the stochastic probes, an integer in <code>[0, 2^32)</code>. Defaults to 0.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="gpuspatiallagimpacts"><code>GPUSpatialLagImpacts</code></h3>
<p>Turns spatial lag coefficients and weights traces into average direct, indirect and total effects of each predictor.</p>
<p>::lens{scenario=gpu-spatial-analysis.spatial-lag-impacts rung=result caption="Direct, indirect and total effects of three predictors; poverty's total is 0.42, 61% of it indirect."}</p>
<p>Average direct, indirect and total impacts of the predictors of a spatial lag model (LeSage and Pace 2009):
<code>direct = b_k tr((I - rho W)^-1) / n</code>, <code>total = b_k 1'(I - rho W)^-1 1 / n</code>, <code>indirect = total - direct</code>, from the
truncated series <code>n + sum_{q&#x3C;=Q} rho^q tr(W^q)</code> and <code>n + sum_{q&#x3C;=Q} rho^q 1'W^q 1</code> evaluated in one kernel. Props:
<code>coefficients</code> and <code>coefficientStride</code> (the <code>table</code> of <code>GPUSpatialTwoStageLeastSquares</code> with stride 4, or any
fit with a coefficient per row), <code>rhoIndex</code> (<code>predictorCount + 1</code> for the 2SLS table), optional
<code>firstPredictorIndex</code> (default 1, so the intercept is excluded), <code>predictorCount</code>, <code>rowCount</code>, <code>maximumPower</code>,
<code>traces: {powerTraces, powerSums, infinityNorm}</code> (the outputs of <code>GPUSpatialWeightsTraces</code>), optional <code>status</code> (a fit's <code>output.status.code</code>) and <code>rhoBound</code>
(default 1), and <code>output: {impacts}</code>: <code>predictorCount * 3</code> f32 values <code>[direct, indirect, total]</code> per
predictor.</p>
<p>Impacts are NaN when the fit <code>status</code> is nonzero, <code>rho</code> is not finite or <code>|rho| >= rhoBound</code>. For a <code>W</code> that
is not row-standardized set <code>rhoBound = 1 / spectralRadius</code>, so the series cannot diverge silently.</p>
<p>The direct impact matches spreg 1.9.1 <code>GM_Lag(..., spat_impacts='full')</code>, and so does the total for a
row-standardized <code>W</code> (spreg uses <code>1 / (1 - rho)</code>, which is exact only there; this node uses the exact
<code>1'W^q1</code> sums, so it is also right for other weights). spreg's <code>'simple'</code> method fixes the direct multiplier at 1
and is not reproduced; its <code>'power'</code> method stops after the first term below tolerance, which is wrong on
bipartite lattices where odd traces vanish. Standard errors of the impacts (simulation of <code>(rho, b)</code> from the
coefficient covariance) are not provided: the 2SLS contributor does not output the covariance matrix.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUSpatialTwoStageLeastSquares</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  weights, predictors, response, predictorCount: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, output: {table, summary, status: {code: status}}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUSpatialWeightsTraces</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({weights, maximumPower: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">30</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, output: {powerTraces, powerSums, infinityNorm}}));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUSpatialLagImpacts</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  coefficients: table, coefficientStride: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">4</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, rhoIndex: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, predictorCount: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  rowCount, maximumPower: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">30</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, traces: {powerTraces, powerSums, infinityNorm}, status, output: {impacts}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span></code></pre>
<h4 id="how-it-computes-6">How it computes</h4>
<p>{/* lens:GPUSpatialLagImpacts */}</p>
<p>::lens{scenario=gpu-spatial-analysis.spatial-lag-impacts rung=formula caption="Follow the direct, total and indirect effect series in rho."}
::lens{scenario=gpu-spatial-analysis.spatial-lag-impacts rung=kernel caption="Inspect the impacts kernel with its retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.spatial-lag-impacts rung=graph caption="Trace the likelihood fit, its power traces and the impacts over 788 tracts."}</p>
<h3 id="gpuspatialerrorgm"><code>GPUSpatialErrorGM</code></h3>
<p>Estimates a spatial error model by generalized moments, then a feasible GLS regression on spatially filtered variables.</p>
<p>::lens{scenario=gpu-spatial-analysis.spatial-error-gm rung=result caption="Five rows: intercept, three predictors and lambda; lambda = 0.83 with standard error 0.015."}</p>
<p>The spatial error model <code>y = X b + u</code>, <code>u = lambda W u + e</code> by generalized moments. Props: <code>weights</code> (any square
CSR; no symmetry or standardization needed), <code>predictors</code>, <code>response</code>, <code>predictorCount</code> (1 to 15, the OLS limit and
the 16-wide Cholesky; no kernel binds more than 8 storage buffers, so the previous cap of 8 was not needed),
<code>estimator</code>, optional <code>tileRowCount</code>, <code>traceMode</code>, <code>traceProbeCount</code>, <code>traceParameters</code> (per-frame seed, as <code>GPUSpatialWeightsTraces</code> <code>parameters</code>) and
<code>output: {table, summary, status: {code}, residuals?}</code>.
For either estimator, construction requires at most 4096 tiles for the serial norm merge:
<code>ceil(rows / tileRowCount)</code> for whole weights, or the sum of <code>ceil(chunk.rowCount / tileRowCount)</code>
over chunks. Increase <code>tileRowCount</code> or combine small chunks if this limit is exceeded.
The default whole-CSR layout still reaches <code>2^24</code> rows; chunk boundaries can exceed the cap even
with the default tile width. CPU twins do not use tile merges and retain their existing row limits.
<code>GPUSpatialErrorGMOperation</code> applies the whole-CSR tile-count check during lowering.
The contributor captures chunk boundaries and nested weight-view references at construction,
so later edits to the caller's chunk layout cannot change the validated tile count.</p>
<p><strong>Default change.</strong> <code>estimator</code> is <code>'homoskedastic'</code> by default (spreg 1.9.1 <code>GM_Error_Hom</code>, <code>A1 = 'hom_sc'</code>,
one iteration: Drukker, Egger and Prucha 2013, Anselin 2011), which gives <code>lambda</code> a standard error, z and p-value.
The previous behavior is <code>estimator: 'kelejian-prucha'</code> (spreg <code>GM_Error</code>, Kelejian and Prucha 1998 and 1999), whose
<code>lambda</code> row keeps NaN for se, z and p; in both cases the standard errors of the betas condition on the
estimated <code>lambda</code>.</p>
<ul>
<li>Steps (both): OLS of <code>y</code> on <code>[1, X]</code> gives <code>u</code>; moments of <code>u</code> give <code>lambda</code>, minimized globally on a 1024-interval
grid with bisection of every bracketed minimum (spreg runs a local L-BFGS-B from zero; the two agree whenever the
objective has a single basin); OLS of <code>y - lambda W y</code> on the filtered <code>[1, X]</code> (the constant is filtered too, as
<code>get_spFilter</code> does) gives the coefficients.</li>
<li><code>lambda</code> bound: <code>0.99 / ||W||_inf</code> with <code>||W||_inf</code> the largest absolute row sum, computed on the GPU. It equals
spreg's fixed 0.99 for a row-standardized <code>W</code> and keeps <code>I - lambda W</code> invertible for any <code>W</code>; a fixed 0.99
admitted explosive <code>lambda</code> for raw weights. A solution on the bound is flagged
(<code>..._SUMMARY_LAMBDA_AT_BOUND</code>) and <code>..._SUMMARY_LAMBDA_BOUND</code> reports the bound.</li>
<li><code>homoskedastic</code> replicates spreg step for step: (a) <code>lambda1</code> minimizes the unweighted
<code>|G [l, l^2]' - g|^2</code> of the moments <code>u'A1u</code>, <code>u'A2u</code> with <code>A1 = (W'W - (t/n) I) / (1 + (t/n)^2)</code>, <code>t = tr(W'W)</code>,
<code>A2 = (W + W') / 2</code>; (b) the FGLS betas at <code>lambda1</code> are the reported betas and give <code>u1 = y - X b1</code>; (c) the
moments are recomputed on <code>u1</code> and <code>psi</code>, the variance of the moments (from <code>sigma2</code>, <code>mu3</code>, <code>mu4</code> of
<code>u1 - lambda1 W u1</code> and the traces <code>tr W</code>, <code>tr W^2</code>, <code>tr W'W</code>, <code>tr W'WW</code>, <code>tr (W'W)^2</code>, <code>sum_i c_i^2</code> of
<a href="#gpuspatialweightstraces"><code>GPUSpatialWeightsTraces</code></a>), weights the second minimization
<code>(G p - g)' psi^-1 (G p - g)</code> giving <code>lambda</code>; (d) <code>Var(b) = sigma2 (Xs'Xs)^-1</code> with <code>Xs</code> and <code>u_s</code> filtered at
<code>lambda</code> (<code>sigma2 = u_s'u_s / n</code>) and <code>Var(lambda) = (j' psi^-1 j)^-1 / n</code>, <code>j = G [1, 2 lambda]'</code>. The
closed forms replace spreg's <code>n x n</code> <code>A1</code>, <code>A2</code> and are verified against spreg's own <code>psi</code>, <code>G</code>, <code>lambda</code> and
standard errors (the betas-<code>lambda</code> covariance of spreg is not formed; it changes no reported number).
Pinned to spreg on a row-standardized lattice, an asymmetric kNN scene and a non-row-standardized (binary) lattice.</li>
<li>Traces are exact up to 4096 rows and Hutchinson estimates above (<code>traceMode</code> overrides). With stochastic traces
the standard error of <code>lambda</code> is approximate and <code>..._SUMMARY_APPROXIMATION</code> is 1.0 (betas, <code>lambda</code> and the
betas' standard errors still use the exact moment sums).</li>
<li><code>table</code>: <code>(k + 2) x 4</code> f32 rows of <code>[coefficient, se, z, two-sided normal p]</code>: the intercept, <code>X</code>, then
<code>lambda</code> last.</li>
<li><code>summary</code> (13, <code>GPU_SPATIAL_ERROR_GM_SUMMARY_*</code>): <code>n</code>, <code>lambda</code>, <code>sigma2 = e'e / n</code> (spreg <code>sig2n</code>), pseudo R2,
the residual sum of squares <code>u'u</code> and the moment objective at the solution, then conditioning of the final
filtered regression (Belsley condition number of the raw filtered <code>[1, X]</code>, correlation condition number of the
factored Gram matrix, largest predictor variance inflation factor, and a precision warning above a correlation
condition number of 1e4 that never changes <code>status</code>), the <code>lambda</code> bound, the approximation flag and the
at-bound flag.</li>
<li><code>status.code</code>: 0 ok, 1 singular (OLS or filtered fit), 2 too few rows (<code>n &#x3C;= k + 2</code>), 3 non-finite (including <code>W</code>
without links).</li>
<li><code>residuals</code>: <code>u = y - X b</code>, formed in the centered model as <code>(y - ybar) - a0 - (x - xbar) b</code>, so they and the
filtered residuals behind <code>sigma2</code> keep their accuracy when the response mean is far above them (residuals of
0.002 to 0.004 around 10,000 match spreg in float64 within 1e-4 in <code>sigma2</code> and <code>u'u</code>).</li>
</ul>
<p><code>getGPUSpatialErrorGMModelResult(readback, {predictorCount?, predictorNames?})</code> turns the read-back <code>table</code>, <code>summary</code> and <code>status</code>
into the shared <code>GPURegressionModelResult</code>: rows <code>CONSTANT</code>, the predictors and <code>lambda</code> (role <code>'spatial-error'</code>),
z statistics with float64 normal p-values, <code>sigmaSquaredMaximumLikelihood = e'e / n</code> (and
<code>residualSumOfSquares = n sigma2</code> of the filtered residuals, the scale of the standard errors), <code>pseudoRSquared</code>,
no likelihood and the conditioning above.
Pass the fitted <code>predictorCount</code> for a table with spare capacity; padding after the active
<code>(predictorCount + 2) * 4</code> values is ignored. Omitting the count requires an exact-length table.</p>
<p>Accumulations use the fixed-order tile and pairwise-merge scheme of <code>GPUOrdinaryLeastSquares</code>, so results are
deterministic. The spec matches spreg on a 10x10 lattice and an asymmetric kNN scene (<code>k = 4</code>, 80 rows):
<code>lambda</code> and the coefficients within 5e-3, standard errors within 1e-2 and z within 2e-2 (relative), for both
estimators. The <code>GM_Error_Het</code> variant and the Kelejian-Prucha weighted estimator are not provided.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUSpatialErrorGM</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({weights, predictors, response, predictorCount: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, output: {table, summary, status: {code: status}}}));</span></span></code></pre>
<h4 id="how-it-computes-7">How it computes</h4>
<p>{/* lens:GPUSpatialErrorGM */}</p>
<p>::lens{scenario=gpu-spatial-analysis.spatial-error-gm rung=formula caption="Follow the moment system, the lambda search and the filtered regression."}
::lens{scenario=gpu-spatial-analysis.spatial-error-gm rung=kernel caption="Inspect the moment, lambda scan, filtered regression and finish kernels with retained WGSL."}
::lens{scenario=gpu-spatial-analysis.spatial-error-gm rung=graph caption="Trace the OLS stage, traces, moment steps and filtered fit over 788 tracts."}</p>
<h3 id="gpuspatiallagmaximumlikelihood"><code>GPUSpatialLagMaximumLikelihood</code></h3>
<p>Estimates a spatial lag model by maximum likelihood, searching rho on the concentrated likelihood with a trace-series log-determinant.</p>
<p>::lens{scenario=gpu-spatial-analysis.spatial-lag-maximum-likelihood rung=result caption="Five rows: intercept, three predictors and rho; rho = 0.65 with standard error 0.021."}</p>
<p>Maximum-likelihood estimator of the spatial lag model <code>y = rho W y + X b + e</code> (spreg 1.9.1 <code>ML_Lag</code>, <code>method='full'</code>),
with the exact log-determinant replaced by a truncated trace series. Nothing is read back. Props: <code>weights</code> (any
square CSR), <code>predictors</code>, <code>response</code>, <code>predictorCount</code> (1 to 14), <code>maximumPower</code> (series length <code>Q</code>, 2 to 256,
default 128), <code>traceMode</code>, <code>traceProbeCount</code>, <code>traceParameters</code> (per-frame seed), optional <code>tileRowCount</code> and
<code>output: {table, summary, status: {code, converged?}, residuals?, powerTraces?, powerSums?, infinityNorm?}</code>. Provide <code>powerTraces</code>,
<code>powerSums</code> and <code>infinityNorm</code> together to share the traces with a downstream <a href="#gpuspatiallagimpacts"><code>GPUSpatialLagImpacts</code></a>.
The likelihood contributor resolves its nested trace layout at construction, starting with the
trace default width and doubling it if chunk boundaries produce more than 4096 tiles.
More than 4096 nonempty chunks cannot fit and are rejected at construction, before graph resources
are declared. The captured layout is used for both allocation planning and the trace child;
weights port maps and chunk boundaries are also captured, while GPU contents remain per-frame inputs.
The GPU operation form applies these checks when lowered; CPU twins do not use trace tiling.</p>
<ul>
<li>Steps: the OLS residuals <code>e0</code>, <code>e1</code> of <code>y</code> and of the lag of <code>y</code> on <code>[1, X]</code> (two <code>GPUOrdinaryLeastSquares</code>
compositions) give the concentrated likelihood <code>-n/2 ln(s2(rho)) + ln|I - rho W|</code>,
<code>s2 = (e0 - rho e1)'(e0 - rho e1) / n</code>. The lag is formed about the response mean,
<code>W y = W (y - ybar) + ybar s_i</code>, with the row sums <code>s_i</code> of the float32 weights carried exactly (rows without
neighbours included), because a sum of <code>w * y</code> over the raw response is rounded to a float32 step of the
response. The search evaluates <code>u'u</code> as <code>S + e1'e1 (rho - rho*)^2</code>, with <code>S</code> the sum of squares of
<code>e0 - rho0 e1</code> at <code>rho0 = e0'e1 / e1'e1</code> summed per row and <code>rho*</code> its minimizer, not as
<code>e0'e0 - 2 rho e0'e1 + rho^2 e1'e1</code>: where the model fits well (<code>u'u</code> below 1e-7 of <code>e0'e0</code>) that difference
of three float32 sums is rounding, and the search then failed (status 3) or returned a wrong <code>rho</code> with
status 0. The log-determinant is <code>-sum_{q&#x3C;=Q} rho^q tr(W^q) / q</code> from
<a href="#gpuspatialweightstraces"><code>GPUSpatialWeightsTraces</code></a> (exact traces up to 4096 rows, Hutchinson above).
<code>rho</code> is the global maximum on <code>[-B, B]</code>: one kernel evaluates the analytic derivative on 1025 grid points,
bisects (32 steps) every positive-to-non-positive sign change on the derivative, and keeps the best candidate
(an end point counts when the derivative points outward). It is deterministic and has no iteration budget.
<code>b = b0 - rho b1</code>, <code>s2 = u'u / n</code>.</li>
<li>Variance: spreg's information matrix of <code>(b, rho, s2)</code> with <code>tr1 = tr(W A)</code> and <code>tr2 = tr((W A)^2)</code> as series in
<code>rho</code> of the power traces, and <code>tr3 = tr((W A)' W A)</code>, <code>A = (I - rho W)^-1</code>, from a second probe pass at the
estimated <code>rho</code> (Horner <code>u &#x3C;- z + rho W u</code> per probe, exact unit probes up to 4096 rows, Rademacher above; the
probes of <code>GPUSpatialWeightsTraces</code>). The <code>s2</code> row is eliminated by its Schur complement, the matrix is built on
centered predictors and on <code>W A X b</code> about its mean (two more nodes; for a response with a large mean the raw
column cancels against the intercept in float32, which gave standard errors of <code>rho</code> and of the intercept a
half to a third of spreg's, or NaN), unit-scaled and factored with the shared Cholesky, and the intercept
variance is mapped back to the uncentered model. A positive definite failure leaves the standard errors NaN
(<code>..._SUMMARY_VARIANCE_DEFINED</code> 0) without changing <code>status</code>. The coefficient block equals spreg's <code>vm</code>.</li>
<li>Search interval and truncation. <code>B = min(0.99, seriesLimit) / ||W||_inf</code> with <code>||W||_inf</code> the largest absolute row
sum computed on the GPU (0.99 for a row-standardized <code>W</code>; keeps <code>I - rho W</code> invertible for any <code>W</code>, whereas spreg's
fixed <code>(-1, 1)</code> admits explosive <code>rho</code> for raw weights). The series omits a tail below
<code>n s^(Q+1) / ((Q+1) (1 - s))</code>, <code>s = |rho| ||W||_inf</code> (worst case, <code>|tr(W^q)| &#x3C;= n s^q</code>;
<code>getSpatialLogDeterminantTruncationBound</code>), and omitting it biases the likelihood toward large positive <code>rho</code>, so
the interval also stops at the <code>seriesLimit</code> where the worst-case tail reaches <code>1e-5 n</code>
(<code>getSpatialMaximumLikelihoodSeriesLimit</code>: about 0.93 for <code>Q = 128</code>, 0.96 for <code>Q = 256</code>). Beyond it the fit is a
boundary solution (<code>..._SUMMARY_AT_BOUNDARY</code>, <code>..._SERIES_LIMITED</code>, <code>converged</code> 0); raise <code>maximumPower</code> when
<code>rho</code> is near the bound. The default <code>Q = 128</code> keeps the worst-case truncation below <code>1e-7 n</code> log-likelihood units
at <code>|rho| &#x3C;= 0.9</code> (below <code>1e-6 |LL|</code>) at the cost of about <code>Q (1 + probe batches)</code> dispatches; the probe pass costs
<code>rows x Q x nnz</code> multiply-adds in exact mode. <code>..._SUMMARY_TRUNCATION_BOUND</code> reports the remainder at the estimate.</li>
<li>Approximation. Above 4096 rows the traces and the <code>tr3</code> pass are Hutchinson estimates (<code>..._SUMMARY_APPROXIMATION</code>
1.0): <code>rho</code> itself then carries stochastic error (about 1e-3 with 256 probes on a 144-row lattice), unlike
<code>GPUSpatialErrorGM</code> where only the standard error of <code>lambda</code> is approximate.</li>
<li><code>table</code>: <code>(k + 2) x 4</code> f32 rows of <code>[coefficient, se, z, two-sided normal p]</code>: the intercept, <code>X</code>, then <code>rho</code> last
(spreg <code>W_y</code>). It is the layout <code>GPUSpatialLagImpacts</code> reads (<code>coefficientStride: 4</code>, <code>rhoIndex: predictorCount + 1</code>).</li>
<li><code>summary</code> (22, <code>GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_*</code>): <code>n</code>, <code>rho</code>, <code>sigma2 = u'u / n</code>, full log-likelihood
<code>-n/2 ln s2 + ln|I - rho W| - n/2 (ln 2 pi + 1)</code> (spreg <code>logll</code>), AIC and Schwarz with spreg's <code>k = predictorCount + 2</code>
(<code>rho</code> counted), <code>pr2</code> (squared correlation of <code>y</code> and <code>y - u</code>), <code>pr2_e</code> (of <code>y</code> and <code>(I - rho W)^-1 X b</code>), <code>u'u</code>,
conditioning of the design (Belsley number, correlation condition number, largest VIF, precision warning above
1e4; never changes <code>status</code>), the bound, the approximation, converged, at-boundary and series-limited flags, the
truncation bound, variance-defined, the number of local maxima found on the grid (more than 1 is multimodal) and
<code>ln|I - rho W|</code>.</li>
<li><code>status.code</code>: 0 ok, 1 singular design, 2 too few rows (<code>n &#x3C;= k + 2</code>), 3 non-finite predictor, response or weight, no
admissible interval, a residual sum of squares that reaches zero inside the search interval, or an interior
<code>rho</code> that fails the first-order condition: with <code>u'u</code> and <code>u'e1</code> summed per row from the published
residuals and <code>t = u'e1 / u'u</code>, the derivative <code>g = n t - tr(W A)</code> of the concentrated log-likelihood and its
curvature <code>c = n (e1'e1 / u'u - 2 t^2) + tr((W A)^2)</code> must satisfy <code>|g| &#x3C;= sqrt(0.25 c) + 4 c step</code>, that
is, <code>rho</code> is within half a standard error plus four float32 steps of <code>rho</code> of a stationary point. Only
ratios of the sums are formed, so a response multiplied by 1e-10, 1 or 1e10 gives the same status and the
same <code>rho</code> to 6e-8 (the products of two sums of the earlier form overflowed at 1e10 and rejected a correct
fit). The four steps are the float32 resolution of <code>rho</code> and of the sums: without them the float32 nearest
to the maximum failed the test when the standard error of <code>rho</code> is below one step (a 64-row scene with a
standard error of 4.6e-9; <code>rho</code> is now returned within two steps, but <code>s2</code> is 33 times the float64 value
there, the standard errors 5.7 times and the log-likelihood 111 low, and no output flags it. The cause is
the coefficients: the residuals are <code>e0 - rho e1</code> of two float32 OLS fits, each with its own rounded slope,
and 91% of the published residual sum of squares lies in the span of <code>[1, x]</code> (the published slope is one
float32 step from the float64 slope). The float32 floor of the residual arithmetic is 1.04 to 1.65 times,
and the published residuals with their component in the span of <code>[1, x]</code> removed give 2.9 times; the kernel
does not refine the coefficients as the two-stage kernel does). The check catches a search that ended away from a stationary point: a <code>rho</code>
moved by 1e-4 on a high-fit scene, or by one standard error on an ordinary one, is status 3, and one moved
by a fifth of a standard error is accepted. It shares the trace series with the search and cannot tell a
local maximum from the global one. The outputs are then NaN, except <code>residuals</code>, which keep those of the
rejected <code>rho</code>. <code>status.converged</code> (optional) is 1 for an interior maximum.
Two degenerate cases have a likelihood and are fitted as spreg <code>ML_Lag</code> fits them: a minimum of <code>u'u</code> that
is zero but lies outside the search interval (<code>u'u = e1'e1 (rho - rho*)^2</code> is positive on the interval), and
<code>e1'e1 = 0</code> (the lag column lies in the span of <code>[1, X]</code>, <code>u'u</code> is constant and the estimate is the maximum
of the log-determinant). Both were status 3.</li>
<li><code>residuals</code>: <code>u = y - rho W y - X b</code>, as <code>e0 - rho e1</code> of the two OLS fits, whose residuals sum to zero (see
the OLS numerics above). Measured on the test GPU against the float64 maximum of the concentrated likelihood
(the estimator of spreg <code>ML_Lag</code>, whose own search is within 5e-7 of it in <code>rho</code>), for residuals of 0.002 to
0.004 around a response mean of 10,000:
<ul>
<li>rows that share a row sum, exactly or up to the rounding of the weights (ring at 0.5, six neighbours at
float32 1/6, binary weights with row sum 2, row sums of 1 and 1.0000000298, <code>rho</code> 0.2 to 0.8): <code>s2</code> within
1.1e-5 and the log-likelihood within 0.004 (before the lag was formed about the mean: up to 4.2% and 12.5);</li>
<li>three rows without neighbours, <code>u'u</code> at 2e-10 of <code>e0'e0</code>: <code>s2</code> within 4e-5 and <code>rho</code> within 0.12 standard
errors, a float32 step of <code>rho</code> (before: <code>s2</code> 2,800 times too large with status 0);</li>
<li>slopes two hundred times the residual deviation over the predictor range (<code>u'u</code> at 3e-9 of <code>e0'e0</code>): <code>s2</code>
within 3.4e-4 and <code>rho</code> within 0.04 standard errors on 24 scenes, because the fitted values are rounded to
1% to 3% of a residual (before: 14 of the 24 were status 3 and 3 were wrong with status 0);</li>
<li>row sums that differ by more than rounding with such a response (row sums 2 and 3): the lag differs
between rows by the response mean and the lag column is rounded to a float32 step of that; <code>s2</code> within
2.5e-4.</li>
</ul>
</li>
</ul>
<p>Exact log-determinant (opt-in, also <code>GPUSpatialErrorMaximumLikelihood</code>): <code>logDeterminant: 'exact'</code> with
<code>eigenvalues</code> (the <code>rowCount</code> eigenvalues of <code>W</code> as float32) replaces the truncated trace series of
<code>ln|I - rho W|</code>, <code>tr1</code> and <code>tr2</code> by <code>sum ln|1 - rho lambda|</code>, <code>sum lambda / (1 - rho lambda)</code> and its square sum, so there is
no truncation error and no series limit on the search interval (<code>..._SUMMARY_TRUNCATION_BOUND</code> is 0). It needs
<code>W</code> similar to a symmetric matrix (symmetric pattern and <code>w_ij d_i = w_ji d_j</code>, for example row-standardized
symmetric contiguity or distance weights) and at most <code>GPU_SPATIAL_EXACT_LOG_DETERMINANT_MAXIMUM_ROW_COUNT</code> (1024)
rows, because <code>getSpatialWeightsEigenvalues(offsets, neighbors, weights, rowCount)</code> runs a dense CPU symmetric
eigen-decomposition (about <code>4/3 n^3</code> operations). The third trace of the variance still comes from the probe pass.
The helper rejects nonpositive or nonfinite recovered similarity scales and checks reversibility with tolerance
1e-4 relative to each transformed pair's magnitude, without an absolute floor; rescaling the weights does not
relax that check. Scale recovery uses exponent-scaled products when an intermediate overflows or underflows,
so a positive finite final similarity scale survives. The standalone <code>getSpatialExactLogDeterminant(rho, eigenvalues)</code> accepts negative factors
and returns the log absolute determinant (<code>rho = 2</code>, eigenvalues <code>[1, -1]</code> gives <code>log(3)</code>). A zero factor
returns <code>-Infinity</code>. A product that rounds to one is checked using its multiplication residual before declaring
singularity; finite overflowing products contribute the operands' log magnitudes. True singularity takes priority
regardless of spectrum order. The CPU helper uses <code>log1p</code> for positive factors to preserve small terms; the GPU uses
a short series near zero for the same purpose.
The default stays <code>'series'</code>.</p>
<p><code>getGPUSpatialLagMaximumLikelihoodModelResult(readback, {predictorCount?, predictorNames?})</code> builds the shared
<code>GPURegressionModelResult</code> (<code>kind: 'spatial-lag-maximum-likelihood'</code>, rows <code>CONSTANT</code>, predictors and <code>W_y</code> with role
<code>'spatial-lag'</code>, z statistics with float64 normal p-values, <code>standardErrorSigmaSquared: 'maximum-likelihood'</code>,
<code>logLikelihood</code>, and AIC and Schwarz equal to spreg's); <code>getGPUSpatialMaximumLikelihoodDiagnostics(summary)</code> returns
the flags and <code>pr2_e</code>. At most 14 predictors: the <code>(predictorCount + 2)</code>-wide information matrix must fit the
16-wide Cholesky. Inputs must be finite.
For a capacity-sized table, pass the fitted <code>predictorCount</code> to decode only its active
<code>(predictorCount + 2) * 4</code> values and compute criteria with the fitted parameter count. Omitting
it requires an exact-length table.</p>
<p>Verified against spreg <code>ML_Lag(method='full', spat_impacts='full')</code> on a 12x12 lattice, a binary (raw) lattice, a
six-predictor lattice and two kNN scenes by a float64 mirror (<code>rho</code> to 4e-8, coefficients 3e-7, standard errors 1e-7,
z, log-likelihood, AIC, Schwarz, <code>pr2</code>, <code>pr2_e</code>, conditioning and <code>tr1</code>, <code>tr2</code>, <code>tr3</code>); the GPU spec checks the float32
output against the same pins at 2e-3 (<code>rho</code>, coefficients) and 1e-2 (standard errors), and chains the fit into
<code>GPUSpatialLagImpacts</code>. spreg's total multiplier <code>1 / (1 - rho)</code> is exact only for a row-standardized <code>W</code>; for other
weights the impacts here use the exact <code>1'W^q 1</code> sums.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUSpatialLagMaximumLikelihood</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({weights, predictors, response, predictorCount: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {table, summary, status: {code: status}, powerTraces, powerSums, infinityNorm}}));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUSpatialLagImpacts</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({coefficients: table, coefficientStride: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">4</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, rhoIndex: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, predictorCount: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  rowCount, maximumPower: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">128</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, traces: {powerTraces, powerSums, infinityNorm}, status, output: {impacts}}));</span></span></code></pre>
<h4 id="how-it-computes-8">How it computes</h4>
<p>{/* lens:GPUSpatialLagMaximumLikelihood */}</p>
<p>::lens{scenario=gpu-spatial-analysis.spatial-lag-maximum-likelihood rung=formula caption="Follow the concentrated likelihood, its log-determinant series and the rho search."}
::lens{scenario=gpu-spatial-analysis.spatial-lag-maximum-likelihood rung=kernel caption="Inspect the OLS, trace, rho search and variance kernels with retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.spatial-lag-maximum-likelihood rung=graph caption="Trace the two OLS fits, 128 trace powers and the rho search over 788 tracts."}</p>
<h3 id="gpuspatialerrormaximumlikelihood"><code>GPUSpatialErrorMaximumLikelihood</code></h3>
<p>Estimates a spatial error model by maximum likelihood, searching lambda on the concentrated likelihood with a trace-series log-determinant.</p>
<p>::lens{scenario=gpu-spatial-analysis.spatial-error-maximum-likelihood rung=result caption="Five rows: intercept, three predictors and lambda; lambda = 0.89 with standard error 0.018."}</p>
<p>Maximum-likelihood estimator of the spatial error model <code>y = X b + u</code>, <code>u = lambda W u + e</code> (spreg 1.9.1
<code>ML_Error</code>, <code>method='full'</code>), with the same truncated-series log-determinant, search and approximation rules as
<code>GPUSpatialLagMaximumLikelihood</code>. Props: <code>weights</code>, <code>predictors</code>, <code>response</code>, <code>predictorCount</code> (1 to 15),
<code>maximumPower</code>, <code>traceMode</code>, <code>traceProbeCount</code>, <code>traceParameters</code> (per-frame seed), optional <code>tileRowCount</code> and
<code>output: {table, summary, status: {code, converged?}, residuals?}</code>.
Nested trace layout resolution and the 4096-nonempty-chunk construction limit follow
<code>GPUSpatialLagMaximumLikelihood</code>, including captured weights metadata and allocation layout.</p>
<ul>
<li>Steps: a first tile pass fits OLS on the centered columns and gives <code>e0 = (y - mean) - (X - mean) b0</code>. With
<code>Z = [1, X - mean, e0]</code>, three tile passes (fixed pairwise merge) accumulate once
<code>G0 = Z'Z</code>, <code>G1 + G1' = Z'WZ + (Z'WZ)'</code> and <code>G2 = (WZ)'(WZ)</code>. Using <code>e0</code> for the response leaves the minimum over
the coefficients unchanged (the filtered regression estimates <code>b - b0</code>) and keeps <code>ee(lambda)</code> accurate in float32
when <code>R2</code> is close to 1: with the raw response <code>yy - 2 b'r + b'Ab</code> cancels to <code>1 - R2</code> of its terms. The Gram matrix of the filtered columns
<code>Z - lambda W Z</code> (the filtered constant <code>1 - lambda sum_j w_ij</code> is column 0, as in spreg) is then exactly
<code>G0 - lambda (G1 + G1') + lambda^2 G2</code>, so scanning <code>lambda</code> needs no pass over the rows. Each grid point solves
one unit-scaled Cholesky system for <code>ee(lambda)</code>, and the derivative follows from the envelope theorem
(<code>ee' = yy' - 2 b'r' + b'A'b</code>); the search is the one of the lag model on <code>-n/2 ln(ee / n) + ln|I - lambda W|</code>.</li>
<li>Variance: <code>Var(b) = s2 (Xs'Xs)^-1</code> (centered form mapped back to the uncentered intercept) and
<code>Var(lambda) = 1 / (tr2 + tr3 - 2 tr1^2 / n)</code>, the Schur complement of spreg's 2 x 2 information matrix of
<code>(lambda, s2)</code>, with the traces of <code>W (I - lambda W)^-1</code> (<code>tr3</code> from the probe pass at the estimated <code>lambda</code>). As in
spreg the coefficient standard errors condition on <code>lambda</code>.</li>
<li><code>table</code>: <code>(k + 2) x 4</code> rows, <code>lambda</code> last. <code>summary</code>: the 22 slots of the lag model; <code>sigma2 = e'e / n</code> of the
filtered residuals, <code>pr2</code> of <code>y</code> and <code>X b</code>, <code>pr2_e</code> NaN, <code>u'u</code> of <code>u = y - X b</code>, and the conditioning of the final
filtered regression (Belsley number of the raw <code>[c, X - lambda W X]</code>, correlation condition number and largest
predictor VIF of the unit-scaled filtered system). AIC and Schwarz use spreg's <code>k = predictorCount + 1</code>: spreg's
<code>ML_Error</code> does not count <code>lambda</code> (<code>k = x.shape[1]</code>), unlike <code>ML_Lag</code>, which counts <code>rho</code>.</li>
<li><code>status.code</code> as the lag model (the cap is 15 predictors, the OLS and Cholesky limit).</li>
</ul>
<p><code>getGPUSpatialErrorMaximumLikelihoodModelResult(readback, {predictorCount?, predictorNames?})</code> builds the shared model result
(<code>kind: 'spatial-error-maximum-likelihood'</code>, <code>lambda</code> role <code>'spatial-error'</code>, <code>residualSumOfSquares = n sigma2</code> of the
filtered residuals, AIC and Schwarz equal to spreg's). Verified against <code>ML_Error(method='full')</code> on a
row-standardized lattice, an asymmetric kNN scene and a binary lattice (float64 mirror to 1e-7; GPU spec at 2e-3 and
1e-2, not yet measured).
Pass the fitted <code>predictorCount</code> for a table with spare capacity; padding after the active
<code>(predictorCount + 2) * 4</code> values is ignored. Omitting it requires an exact-length table.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUSpatialErrorMaximumLikelihood</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({weights, predictors, response, predictorCount: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {table, summary, status: {code: status}}}));</span></span></code></pre>
<h4 id="how-it-computes-9">How it computes</h4>
<p>{/* lens:GPUSpatialErrorMaximumLikelihood */}</p>
<p>::lens{scenario=gpu-spatial-analysis.spatial-error-maximum-likelihood rung=formula caption="Follow the filtered Gram matrix, the concentrated likelihood and the lambda search."}
::lens{scenario=gpu-spatial-analysis.spatial-error-maximum-likelihood rung=kernel caption="Inspect the Gram, trace, lambda search and variance kernels with retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.spatial-error-maximum-likelihood rung=graph caption="Trace the Gram passes, 128 trace powers and the lambda search over 788 tracts."}</p>
<h3 id="gpugeographicallytemporallyweightedregression"><code>GPUGeographicallyTemporallyWeightedRegression</code></h3>
<p>Geographically and temporally weighted regression (GTWR; Huang, Wu and Barry 2010) at a fixed or adaptive bandwidth. Every row is fitted by weighted least squares with kernel weights on the space-time distance <code>sqrt(ds^2 + tau dt^2)</code>, where <code>ds</code> is the planar distance, <code>dt</code> the difference of the times in ticks and <code>tau</code> the time scale (the <code>tau</code> of the Python package mgtwr, whose <code>GTWRKernel</code> and <code>GTWR.fit</code> are the oracle). The kernel is Gaussian <code>exp(-z^2 / 2)</code> or bisquare <code>(1 - z^2)^2</code> with <code>z = d / h</code>, chosen at compile time. <code>bandwidth</code> and <code>timeScale</code> are per-frame (<code>getGPUGTWRParameterValues</code>), so a caller can move through the <code>(bandwidth, timeScale)</code> plane without recompiling. With <code>adaptive: true</code>, bandwidth is an integer count of usable neighbours, including the focal observation and zero-distance ties, from 1 to 128. The local bandwidth is the kth space-time distance multiplied by <code>1.0000001</code> (mgtwr convention). A count greater than the usable population, a noninteger count, or a zero kth distance reports <code>SINGULAR</code>. Masked padding never enters the neighbour list.</p>
<p>Set <code>bandwidthSearch: {parameters, maximumIterations}</code> to select both bandwidth and time scale on the GPU. Pack the six per-frame values with <code>getGPUGTWRBandwidthSearchParameterValues({bandwidthBounds: [minimum, maximum], timeScaleBounds: [minimumScale, maximumScale], absoluteTolerance: 1e-3, relativeTolerance: 1e-3})</code>. The outer golden-section search minimizes the time-scale profile; each profile evaluation runs an inner golden-section bandwidth search. Each scalar search evaluates both endpoints and up to <code>maximumIterations</code> interior pairs (default 8, maximum 32). Adaptive bandwidth bounds must be integers in [1, 128]; time-scale bounds may include zero. Equal bounds fix an axis. This is a bounded reference algorithm: the profile is assumed unimodal, and nonsmooth adaptive objectives may have other minima. It does not promise an exhaustive global optimum.</p>
<p><code>output.selectedParameters</code> optionally receives the winning <code>[bandwidth, timeScale]</code> pair. <code>output.status</code> accepts the canonical <code>iterationCount</code>, <code>converged</code> and <code>iterationLimitReached</code> uint32 ports. The count is outer time-scale comparisons; convergence requires the outer interval and the winning inner interval to meet <code>width &#x3C;= max(absoluteTolerance, relativeTolerance * max(abs(lower), abs(upper)))</code> (at least one neighbour for the adaptive inner interval). Adaptive interior comparisons use distinct counts whenever possible; the final interval of at most three counts is also considered resolved once every count in it has been evaluated. Limit flags include the winning inner search. Candidates with singular rows or nonfinite AICc cannot win by dropping observations; exact score ties choose the smaller time scale, then the smaller bandwidth. At the limit, the best finite evaluated model is fitted and published. If no finite candidate exists, selected settings are NaN and included rows report <code>SINGULAR</code>; no convergence is claimed. Invalid raw GPU bounds follow the same path. Per-frame settings and bounds do not recompile; row counts, kernel, adaptive mode and budgets are compile-time. <code>estimatedCommandNodeCount</code> includes the nested search and construction rejects combinations above 32,768 commands. Finished searches gate their later fit dispatches on the GPU.</p>
<p><code>times</code> are float32 ticks after the origin of a <code>temporalContext</code> that declares <code>float32-relative</code> encoding with an origin epoch; the unit is whatever the context declares, and <code>timeScale</code> converts one tick into spatial units squared per tick squared. The local system is solved in coordinates centered on the focal row with the diagonally equilibrated Cholesky factorization shared with the GWR family, so a rank-deficient or ill-conditioned local design reports <code>SINGULAR</code> for that row (NaN coefficients, fitted value and hat) instead of an unstable fit; <code>EXCLUDED</code> marks a masked row or one with a non-finite value. <code>summary</code> has the layout of <code>GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY</code> over the non-singular rows: residual sum of squares, trace of the hat matrix, AICc (the formula of <code>GPUGeographicallyWeightedRegression</code>, mgwr <code>aicc</code>: <code>-2 llf + 2 n (tr S + 1) / (n - tr S - 2)</code>, NaN when <code>n - tr S - 2 &#x3C;= 0</code>, so a GWR and a GTWR fit are comparable and a degenerate bandwidth cannot win a profile search; mgtwr's own <code>aicc</code>, <code>-2 llf + 2 (tr S + 1) (n - 1) / (n - tr S - 1)</code>, is smaller by <code>2 n (k + 1) / (n - k - 2) - 2 (k + 1) (n - 1) / (n - k - 1)</code> with <code>k = tr S</code>: 2.15 at <code>n = 200</code>, <code>k = 60</code>), R squared, observation count, singular row count, sigma squared, log-likelihood and included row count.</p>
<p>The fit is exhaustive over rows (<code>O(rows^2 x predictors^2)</code>, at most 32,768 rows and 11 predictors), float32 and planar with a native metric (<code>spatialContext</code>, default planar/native; other coordinate spaces are rejected). <code>gtwr-bandwidth-search.spec.ts</code> compares adaptive fits and search AICc with independent NumPy fits and exhaustive grids, including ties, padding, purely spatial bounds, singular/empty candidates and iteration limits. <code>gtwr-mgtwr.spec.ts</code> compares coefficients, fitted values, residuals, hat diagonals and summary statistics with mgtwr 2.0.5 output for four configurations (Gaussian and bisquare, time scale 0, 1.5, 2 and 8) and checks time-shift invariance, a larger time scale changing the fit, rank-deficient designs and masked rows. mgtwr declares no licence, so it is executed as an oracle (<code>test/oracles/gtwr-mgtwr-oracle.py</code>) and no code is copied.</p>
<p>{/* lens:GPUGeographicallyTemporallyWeightedRegression */}</p>
<h3 id="gpumultiscalegeographicallytemporallyweightedregression"><code>GPUMultiscaleGeographicallyTemporallyWeightedRegression</code></h3>
<p>Multiscale GTWR by backfitting, following mgtwr <code>multi_bws</code>. A GTWR fit at the initial <code>(bandwidth, timeScale)</code> starts the components <code>beta_j x_j</code> (the intercept is component 0). Each iteration visits the components in order and smooths the partial residual <code>beta_j x_j + err</code> with a one-covariate local regression at that component's own bandwidth and time scale, <code>beta_ij = sum_l w_il x_lj r_l / sum_l w_il x_lj^2</code>, updating <code>err</code> before the next component (Gauss-Seidel order). <code>iterationCount</code> (1 to 32) is compile-time and every iteration runs; <code>output.scores</code> holds mgtwr's convergence score <code>sqrt(sum (XB_new - XB_old)^2 / n / sum (sum_j XB_new)^2)</code> for each iteration, so a caller sees where mgtwr would have stopped. Without <code>bandwidthSearch</code>, bandwidths are supplied by the caller (<code>getGPUMGTWRParameterValues</code>). With it, the initial GTWR fit and every component's partial-residual, no-intercept local regression select their bandwidth and time scale using the nested profile search above. Component AICc uses the partial response, residual sum of squares and smoother trace <code>sum_i x_ij^2 / sum_l w_il x_lj^2</code>. <code>adaptive</code> also applies to every component. The search bounds are shared by the initial and component searches. <code>output.selectedParameters</code> has the same <code>2 + 2 (predictorCount + 1)</code> layout as the input parameters, holding the initial pair and the last pair for each component. <code>output.status</code> sums outer comparisons over all searches, ANDs convergence and ORs limit flags. Backfitting itself still executes its fixed iteration count and publishes its score history. Command-budget validation includes every search, so larger combinations of backfit and search budgets are refused at construction. A row whose initial local fit is singular starts from zero components, and a component smoother whose weighted <code>x^2</code> sum is not positive leaves a NaN coefficient (status <code>SINGULAR</code>) and contributes nothing.</p>
<p>The spec compares the final coefficients, residuals and the whole score history with <code>mgtwr.function.multi_bws</code> run with fixed per-component bandwidths and time scales for two configurations that converge in five iterations.</p>
<p>See <a href="/luma-contributors/docs/api/gpu-spatial-analysis/statistical-expansion">local variability and regression extensions</a> for the expanded
statistics, fixed, finite-grid or golden-section-selected multiscale fits, generalized local models, pooled panels and regimes.</p>
<p>{/* lens:GPUMultiscaleGeographicallyTemporallyWeightedRegression */}</p>
<h2 id="mgwr-bandwidth-selection-and-limits">MGWR bandwidth selection and limits</h2>
<p><code>GPUMultiscaleGeographicallyWeightedRegression</code> and its GPU/CPU operation forms accept
<code>bandwidthSelection: {criterion: 'AICc', searchMethod: 'golden-section', maximumSearchIterations: 30}</code>.
<code>bandwidthCandidates</code> then holds two positive finite distance bounds per component, in
<code>[interceptLower, interceptUpper, predictor0Lower, predictor0Upper, ...]</code> order, with
<code>bandwidthCandidateCount: 2</code>. Equal bounds select a fixed bandwidth; reversed or invalid bounds
invalidate the fit. The existing finite-grid mode remains the default, as does the
<code>'leave-one-out'</code> criterion. Candidate/bound values can change per frame.</p>
<p>Each backfit component selects its bandwidth from the scalar regression on its current partial
response, following <a href="https://github.com/pysal/mgwr/blob/master/mgwr/search.py">PySAL mgwr multi_bw</a>.
For AICc, include the focal observation and compute
<code>n log(max(RSS / n, 1e-30)) + n log(2π) + n (n + tr(H_j)) / (n - tr(H_j) - 2)</code>.
<code>H_j</code> is that scalar partial-response smoother, and <code>n</code> counts complete, unmasked rows.
The GPU accumulates power-of-two-scaled residual squares and recovers <code>RSS / n</code> using that
included count, so an overflowing intermediate RSS does not reject a representable variance.
A nonpositive denominator or any failed local fit rejects the candidate. Grid ties select the
first candidate; golden-section ties keep the lower interior point. Golden-section shares GWR's
<code>delta = 0.38197</code> and score tolerance <code>max(1e-6, 4e-6 max(abs(scores)))</code>, with at most 64 search
iterations. MGWR also evaluates the upper endpoint and keeps it if it is better than the
interior winner. A feasible endpoint keeps an initially infeasible interior bracket searchable.
If the search budget is exhausted, it returns the better of that endpoint and the last interior
winner. The
contributor retains OLS initialization (zero with full inference), its relative-contribution
backfit stopping rule, and unrounded distance bandwidths; it does not reproduce PySAL's initial
single-bandwidth GWR fit, SOC stopping rule, or two-decimal bandwidth rounding.</p>
<p>For the <strong>full MGWR model</strong>, the trace is <code>tr(S) = sum_j tr(R_j)</code>, where <code>R_j</code> is the coupled
component response operator propagated through the backfit, conditional on the selected bandwidth
history (Li and Fotheringham, 2020, <a href="https://doi.org/10.1080/13658816.2020.1720692"><em>Computational improvements to multi-scale geographically
weighted regression</em></a>). This is the trace in
<code>inference.output.summary[0]</code>, and each <code>tr(R_j)</code> is in <code>effectiveParameters</code>. Summing independent
scalar <code>H_j</code> traces would give a different full-model effective parameter count. Standard errors
remain conditional on bandwidth history; bandwidth confidence intervals and full selection
uncertainty are not provided.</p>
<p>Planar bisquare and explicitly truncated Gaussian fits automatically use the shared GWR grid at
1024 rows: square dimensions <code>ceil(sqrt(rows / 8))</code>, capped at 1024 per axis. <code>indexGridSize</code>
overrides the dimensions, and <code>false</code> forces the dense path. Untruncated Gaussian and great-circle
fits retain the dense fallback, including an explicit grid with an untruncated Gaussian kernel.
Limits are 65,536 dense rows, 1,048,576 indexed rows, 24 predictors,
128 backfit sweeps, 32 finite-grid candidates and the shared 32,768 command-node budget. The
selection score buffer needs <code>rows * candidateWidth * 16</code> bytes in one storage binding
(<code>candidateWidth = 3</code> for golden-section: two interior points and the upper endpoint), checked against the device limit. Full inference retains
its 128-row and cubic-work limits. The CPU twin uses float64 internally. Both backends honor
<code>maximumIterations</code> (default 32); finite unfinished fits publish coefficients and predictions with
status 3, while inference remains unavailable until convergence. The CPU accepts budgets above
the GPU's 128-sweep cap. Near-tied scores can produce different bandwidths within float32 error.
Float32 coefficient or prediction overflow produces numerical-failure code 4 and NaN outputs.
The optional <code>output.status.componentSearchIterationLimitReached</code> has one flag per design
column; it reports any exhausted golden-section search independently of backfit convergence,
accumulates across sweeps, and resets each execution. Both operation constructors snapshot their
sweep budgets. The website labels finite limit estimates separately from exclusions and failures.</p>
<p>The independent <a href="https://github.com/visgl/luma-contributors/blob/master/modules/analysis/test/gpu-spatial-analysis/spatial-regression/multiscale-bandwidth-selection-fixtures.py">NumPy fixture generator</a>
records the initialization and stopping differences and checks scalar selection plus coupled traces.</p>`};export{e as default};