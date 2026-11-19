var e={slug:`api/gpu-spatial-analysis/statistical-expansion`,title:`Local variability, point patterns and spatial regression extensions`,html:`<h1 id="local-variability-point-patterns-and-spatial-regression-extensions">Local variability, point patterns and spatial regression extensions</h1>
<p>These contributors build GPU command-graph nodes. They allocate graph-owned scratch, write caller-owned
outputs, and never read results back or fit a hidden CPU model. Import local variability and regression
from <code>@luma-contributors/analysis/gpu-spatial-analysis</code>; import the pair statistics from
<code>@luma-contributors/analysis/dataframe-analysis</code>.</p>
<h2 id="local-variability">Local variability</h2>
<p><code>GPULocalSpatialHeteroscedasticity</code> computes LOSH using a CSR <code>weights</code> graph, scalar <code>values</code>, an
optional <code>mask</code>, and caller-owned <code>output.statistic</code> and <code>output.status</code> views. <code>residualPower</code> is 2
by default (squared local residuals), or 1 (absolute local residuals). It computes the weighted local
mean, the residual power for each observation, then a weighted local mean of neighboring residual
powers divided by their global mean. The result is descriptive; it does not supply a chi-square or
permutation p-value.</p>
<p>Positive weights are normalized over available neighbors. Masked and nonfinite observations are
excluded. Islands have no residual and are excluded from the global residual mean. In the final
neighbor reduction, undefined residuals are removed and the remaining weights are normalized again.
Thus masked/island behavior is explicit and extends the complete-data <code>esda.LOSH</code> reference contract.</p>
<p><code>GPUMultivariateLocalGeary</code> accepts row-major values and <code>variableCount</code> (1–32). It selects complete
rows jointly, standardizes each variable using its population variance, and averages the
row-standardized local Geary statistic over variables. Complete-case islands participate in global
standardization but have undefined local statistics. A constant variable makes the result undefined.
Randomization must preserve the multivariate row pairing; this contributor does not reuse a
univariate permutation test or manufacture p-values.</p>
<p>Both require per-row <code>output.status</code>:</p>
<table>
<thead>
<tr>
<th>Code</th>
<th>Meaning</th>
</tr>
</thead>
<tbody>
<tr>
<td>0</td>
<td>Defined statistic</td>
</tr>
<tr>
<td>1</td>
<td>Excluded observation or LOSH island</td>
</tr>
<tr>
<td>2</td>
<td>Undefined statistic, including constant fields or invalid accumulated arithmetic</td>
</tr>
<tr>
<td>3</td>
<td>Incomplete upstream weights</td>
</tr>
</tbody>
</table>
<p><code>addLocalVariabilityRecipe</code> composes LOSH and local Moran on the same weights, values and mask. The
returned outputs keep LOSH and its status separate from Moran's statistic, z-score, raw and adjusted
p-values, quadrants, and inference status. Multiple-testing correction applies only to the Moran
family. The recipe preserves both upstream loss flags and the invalid count from each statistical
stage. Candidate-only loss suppresses both LOSH (including normalized residuals) and Moran, with
INCOMPLETE LOSH status; both recover on resubmission when the loss flags clear.
It does not create a joint significance score or inferential LOSH classification.</p>
<h2 id="marked-and-cross-type-point-patterns">Marked and cross-type point patterns</h2>
<p><code>GPUMarkedPointStatistics</code> supplies cross-type K/L, caller-intensity-weighted inhomogeneous K,
annular pair correlation and product-mark correlation in a rectangular window. Inputs are planar
<code>positions</code>, optional <code>types</code> and <code>typePair</code>, optional <code>marks</code> and <code>intensities</code>, a mask, and six
float32 parameters:</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span>[minimumX, minimumY, maximumX, maximumY, maximumRadius, minimumIntensity]</span></span></code></pre>
<p>With <code>radiusCount</code> bins, the upper radii are <code>maximumRadius * (bin + 1) / radiusCount</code>.
<code>output.statistics</code> contains four scalar columns per bin, with indices in
<code>GPU_MARKED_POINT_STATISTICS_COLUMNS</code>: K, L, pair correlation and mark correlation. <code>output.pairCounts</code>
contains cumulative directed pair counts; self-pairs are excluded. <code>output.status</code> is 0 when the
K estimate is defined, 1 for an invalid window or empty category, 2 for nonfinite accumulation,
and 3 when the cumulative directed pair count exceeds <code>0xffffffff</code>. In that case the count
saturates at <code>0xffffffff</code> and all four statistics are NaN. An exact count of <code>0xffffffff</code> is valid.
An empty annulus can produce a valid K with undefined correlations; inspect those float outputs too.</p>
<p>The default <code>edgeCorrection: 'translation'</code> uses the rectangle's overlap with its translated copy.
<code>'none'</code> explicitly disables correction. Pairs with zero overlap are excluded. Without supplied
intensity the normalization is <code>area / (nFocus * nNeighbor)</code> for two different types (spatstat <code>Kcross</code>)
and <code>area / (n * (n - 1))</code> when focus and neighbor are the same set (no <code>types</code>, or an equal <code>typePair</code>),
as <code>GPURipley</code> and spatstat <code>Kest</code>; one point of a type has no same-type K (NaN, status 1).
With intensity, the estimator is the sum of <code>1 / (intensity_i * intensity_j * overlapArea)</code>;
nonpositive, nonfinite and below-threshold intensities are excluded. Intensity estimation remains an
explicit upstream step.</p>
<p>Pair correlation is an annular-area estimate, not a kernel-smoothed <code>pcf</code> estimator. Mark correlation
uses weighted mean products divided by the product of category mean marks; zero mean products and
empty shells are undefined. Supplying intensity weights both the pair and mark sums. These choices
must be matched before comparing another package's output.</p>
<p>The default <code>searchStructure: 'grid'</code> uses a <code>[256, 256]</code> maximum cell lattice; optional
<code>gridSize: [columns, rows]</code> changes those dimensions. Cells are at least <code>maximumRadius</code> wide,
and each focus visits its 3 x 3 neighboring cells. The partial table permits
<code>rows * radiusCount * 3 &#x3C;= 2^24</code> float32 entries, further constrained by the device's storage
binding size. Device limits for the partials and cell offsets are checked before scratch allocation.</p>
<p><code>searchStructure: 'brute-force'</code> explicitly selects the dense reference, limited to 16,384 rows;
it cannot be combined with <code>gridSize</code>. Counts and radius edges are unchanged. Indexed sums use
cell order and a fixed tree of at most 256 entries per invocation. Category means reduce marks
above <code>2^100</code> separately at scale <code>2^-24</code>, divide by the category count, then restore their scale.
This keeps a representable mean finite even when the raw mark sum would overflow. Smaller marks
remain unscaled, preserving their contribution when large opposite marks cancel. Means and sums
can differ from the reference's running means and serial sums within the
float32 comparison tolerance (<code>1e-6 + 1e-4 * |reference|</code> in the traversal regression).</p>
<p>Large or clustered neighborhoods still require quadratic pair work: a radius reaching the window's
extent collapses the lattice to one cell. A finite radius whose square overflows float32 also uses
one cell, retaining the finite-distance predicate and the reference estimator instead of losing all
pairs. Invalid or nonpositive radii remain empty. This fallback preserves correctness; it does not
bound the pair-walk work for broad neighborhoods.</p>
<h2 id="cross-and-space-time-variograms">Cross and space-time variograms</h2>
<p><code>GPUCrossVariogram</code> is deprecated in favor of <code>GPUSpaceTimeVariogram</code>. It remains a forwarding class through the shared indexed empirical variogram front end, preserving its original props, <code>[maximumDistance, maximumTimeLag]</code> parameter layout and <code>lagCount * temporalLagCount</code> output bins. Its first bin includes zero lag; the newer class gives each axis an explicit zero-lag class. <code>secondValues</code> still defaults to <code>values</code>, giving the Matheron semivariogram, and negative cross-semivariances are retained. The adapter computes complete-case bounds on the GPU, including both variables, coordinates, mask and optional time. It fits the lattice to those bounds each encoding; omitted <code>gridSize</code> now uses <code>[256, 256]</code>. The default and explicit lattices now share the same limits: <code>rows * bins &#x3C;= 2^24</code>, at most
128 bins, exact float32 row IDs and the device's storage binding size. Omission of <code>gridSize</code> no
longer imposes the former dense reference's 16,384-row cap.</p>
<p><code>GPUCrossVariogram</code> retains floating-point legacy arithmetic on every frame: it forms pair differences from the original values, accumulates per-row partials through the shared cell index, and combines them with bounded tree reductions. Centering can erase a small raw difference beside an unrelated value outlier, and even a finite global fixed-point scale can round a valid pair term to zero. The legacy path therefore does not center or quantize its values. Neighboring cell ranges are merged in ascending input-row order before summation, retaining the former default's signed cancellation independently of lattice dimensions. The fixed tree over per-row partials can round differently from the former serial finishing sum. Status is 0 valid, 1 empty/invalid parameters, 2 nonfinite accumulated arithmetic, or 3 when a bin's pair count exceeds <code>0xffffffff</code>. Overflow saturates the count at <code>0xffffffff</code> and publishes NaN semivariance; an exact count of <code>0xffffffff</code> remains valid. The compatibility path retains the original <code>rows * bins</code> scratch bound, with no row-sized serial finishing loop.</p>
<p>The legacy adapter accepts positive finite radii whose square or cell margin overflows float32. It also accepts finite bounds whose width or height overflows on subtraction. These cases use a single-cell lattice, with cell lookup avoiding the overflowing subtraction and the original finite-distance filter applied to every candidate pair. Such frames can visit all pairs. Normal frames retain the fitted lattice and neighboring-cell traversal.</p>
<p><code>GPUSpaceTimeVariogram</code> retains its shared fixed-point pair histogram, accumulating positive and negative cross terms separately and distinguishing integer overflow as status 3.</p>
<p><code>GPUSpaceTimeVariogram</code> is built on the indexed pair histogram of <code>GPUVariogram</code>, so its cost follows
the pairs in neighboring lattice cells instead of <code>rows^2</code>. Class 0 of each axis is the exact zero lag
(identical float32 coordinates; identical float32 relative times). Positive lags fall in
<code>1 + min(floor(lag / maximum * count), count - 1)</code>, left-closed and right-open with the final upper
edge included. Output has <code>(lagCount + 1) * (temporalLagCount + 1)</code> bins, spatial-major:
<code>spatialClass * (temporalLagCount + 1) + temporalClass</code>. Bin 0 is coincident points at the same time;
the first row (spatial class 0) is the purely temporal variogram, the first column the purely spatial
one. Each unordered pair is counted once. Parameters are written with
<code>getGPUSpaceTimeVariogramParameterValues({bounds, maximumDistance, maximumTimeLag})</code>; rows outside
<code>bounds</code> are excluded, and <code>gridSize</code> only affects speed. At most 409 bins are supported, and rows are
limited as for <code>GPUVariogram</code>.</p>
<p><code>times</code> must be relative to a float64 origin chosen on the CPU, in one unit: float32 spaces raw Unix
epoch seconds 128 s apart and epoch milliseconds about 2 minutes apart, which collapses short lags.
Build them with <code>getGPUSpaceTimeRelativeTimes(times, origin, requiredResolution)</code> from
<code>gpu-spatial-analysis</code>, which reports the remaining float32 resolution and can throw when the span is
too long for the lags you need. Times closer than that resolution fall into the zero-lag class.</p>
<p>Output views are <code>semivariances</code>, <code>pairCounts</code>, and per-bin <code>status</code> (0 defined, 1 empty or invalid
lag parameters, 2 nonfinite arithmetic, 3 integer accumulation overflow). Complete-case exclusion
includes positions, both values, times, and the mask. Terms are accumulated as exact integers (the
positive and negative parts apart), with a fixed-point unit set by the typical cross term, so results
are reproducible. Parity with gstat <code>variogramST</code> is not yet established: the spec compares with an
in-repo double-precision brute-force reference, and a gstat fixture is pending.</p>
<h2 id="generalized-geographically-weighted-regression">Generalized geographically weighted regression</h2>
<p><code>GPUGeneralizedGeographicallyWeightedRegression</code> supports <code>family: 'poisson'</code> and <code>'logistic'</code>, with
an intercept plus 1–24 predictors, finite planar or great-circle positions, an optional mask and fixed-distance or adaptive-neighbor
bandwidths. Poisson response values are nonnegative integer counts, and logistic response values are
0 or 1. An optional <code>offset</code> is additive on the linear predictor; use log exposure for Poisson.
Parameters are <code>[bandwidth, relativeCoefficientTolerance]</code>. The kernel is <code>'gaussian'</code> (default) or
<code>'bisquare'</code>. With <code>bandwidthMode: 'adaptive'</code>, the first parameter is an integer neighbor count bounded by <code>maximumNeighborCount</code>; see <a href="/luma-contributors/docs/api/gpu-spatial-analysis/adaptive-generalized-gwr">Adaptive generalized GWR</a>. All-zero Poisson or single-class logistic neighborhoods report status 5.</p>
<p>Each local model uses Newton/IRLS with Cholesky and a bounded likelihood step-halving search.
<code>maximumIterations</code> defaults to 32 and is bounded to 128. Outputs are row-major <code>coefficients</code>
(intercept first), <code>fitted</code>, and <code>status: {codes, iterationCount}</code>. Both status columns have one row per observation; <code>iterationCount</code> records the number of local IRLS steps. Converged fits and finite iteration-limit iterates publish coefficients.
There is no implicit regularization or separation correction. Poisson predictors outside [-30,30]
are reported as numerical failures; they are not silently clamped to another likelihood. Logistic
predictors are bounded to [-80,80] for evaluation. All-zero count neighborhoods, singular designs,
and separated outcomes have explicit diagnostics; status 6 names a witnessed complete/quasi-separation
direction. AICc grid/golden-section selection and its named search status are described in
<a href="/luma-contributors/docs/api/gpu-spatial-analysis/adaptive-generalized-gwr">Adaptive generalized GWR</a>.</p>
<h2 id="multiscale-geographically-weighted-regression">Multiscale geographically weighted regression</h2>
<p><code>GPUMultiscaleGeographicallyWeightedRegression</code> implements Gaussian-response additive backfitting,
with a separate distance bandwidth for every variable, including the intercept. Parameters are:</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span>[relativeContributionTolerance, interceptBandwidth, predictorBandwidth0, ...]</span></span></code></pre>
<p>The algorithm starts every coefficient surface at the global ordinary least squares coefficients
(at zero when full <code>inference</code> is requested, because its propagation starts from zero surfaces),
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
whole coupled backfit. <code>maximumIterations</code> bounds completed sweeps on both GPU and CPU (default
32; GPU maximum 128). An exhausted finite fit publishes its last coefficients and fitted values
with code 3. Excluded, singular, and numerically failed fits publish NaNs. Conditional and full
inference require convergence and remain NaN at the iteration limit; full inference status is 1.
<code>output.status.codes</code> has one row per observation; <code>output.status.iterationCount</code> is one scalar
containing the number of completed backfit sweeps. The CPU twin computes in float64 and accepts
budgets above the GPU maximum, while honoring every explicit budget and the shared default.
Both operation constructors snapshot their validated sweep budgets and selection settings;
mutating the caller's props later does not change subsequent executions. Code 0 and code 3 carry
finite float32 coefficients and fitted values on every included row. A coefficient or prediction
that would overflow the public float32 output invalidates the coupled fit with code 4, suppresses
all included coefficients/predictions to NaN, and leaves inference unavailable (status 1).</p>
<p>Planar bisquare and truncated Gaussian kernels can prune candidates through a grid, with at most
1,048,576 rows on that route. Dense traversal permits at most 65,536 rows, including untruncated
Gaussian fits even when an explicit <code>indexGridSize</code> is supplied. Great-circle fits also use dense
traversal. An indexing option cannot lift the bound when the kernel still visits every row.</p>
<p>For automatic bounded selection, supply column-major <code>bandwidthCandidates</code> and
<code>bandwidthCandidateCount</code> (1–32). The default is finite-grid leave-one-out partial-response
squared error; ties select the first candidate. <code>bandwidthSelection.criterion</code> also accepts
<code>'AICc'</code>, scored from each scalar partial-response smoother as in mgwr <code>multi_bw</code>.
<code>bandwidthSelection.searchMethod</code> accepts <code>'finite-grid'</code> (default) or <code>'golden-section'</code>.
Golden-section requires <code>bandwidthCandidateCount: 2</code>: each column gives positive lower and upper
distance bounds. Its <code>maximumSearchIterations</code> is 1–64 and all selection work shares the command
node budget. Invalid candidates lose to valid ones; no valid candidate invalidates the fit.
Selection runs during every component update. <code>output.selectedBandwidths</code> exposes final choices.
Omitting candidates preserves fixed per-component bandwidths, as in the displayed lens.
Optional <code>output.status.componentSearchIterationLimitReached</code> has one uint32 flag per design
column, intercept first. A flag is one if any golden-section search for that component exhausted
its search budget during this submission. Flags accumulate across backfit sweeps and reset on
every execution. Fixed bandwidths and finite-grid selection produce zero flags. Both operation
forms accept graph value names for this port; the CPU function always returns the corresponding
<code>componentSearchIterationLimitReached</code> array. A finite selected bandwidth remains available at
the search limit. Backfit convergence and bandwidth-history-conditioned inference remain
independently valid: backfit code 0 and inference status 0 may coexist with a search-limit flag 1.</p>
<p>Optional <code>output.conditionalStandardErrors</code> requires a scalar <code>noiseVariance</code>. It computes
<code>sqrt(noiseVariance * sum(linearSmootherWeights²))</code> for each local univariate coefficient, holding
other component surfaces and the selected bandwidths fixed. This is conditional submodel uncertainty,
not full MGWR covariance: uncertainty from backfitting, bandwidth selection and noise estimation is
not included. The variance must be supplied explicitly; negative/nonfinite variance produces NaNs.</p>
<p>For full backfit uncertainty, supply <code>inference: {varianceConvention, output}</code> with
<code>standardErrors[n*p]</code>, <code>effectiveParameters[p]</code>, <code>summary[4]</code>, and <code>status[1]</code>, where p includes
the intercept. Every coefficient update propagates its full response Jacobian through all other
component surfaces, using the actual selected bandwidth history. The combined smoother S yields
summary <code>[trace(S), trace(SᵀS), sigma2, residualDegreesOfFreedom]</code>. <code>residual-df</code> (default) estimates
sigma² as SSE/(n-trace(S)); <code>smoother-df</code> uses SSE/(n-2trace(S)+trace(SᵀS)). Standard errors use
the complete backfit Jacobian and estimated variance; effective parameters are each component's
smoother trace. Inference status is 0 valid, 1 unsuccessful coupled fit, or 2 invalid degrees of
freedom/variance. Failed inference produces NaNs.</p>
<p>This matches mgwr's bandwidth-conditioned uncertainty, including coupling among coefficient
surfaces; it does not add uncertainty from reselecting bandwidths under resampling. Full inference
requires n ≤ 128 and <code>n³*p*maximumIterations&#x3C;=268435456</code>, and allocates two n²p Jacobian buffers.
Bandwidths are distances; adaptive-neighbor MGWR selection is not supported. Golden-section is
a bounded continuous search, not a global-optimum certificate for a non-unimodal score. Direct mgwr 2.2.1 fixtures cover fixed bandwidths, both variance conventions, and an
explicit finite-grid LOO policy composed from actual mgwr local fits. The latter is not advertised
as parity with Sel_BW's different default AICc selection policy.</p>
<p>MGWR exports <code>GPU_MULTISCALE_GEOGRAPHICALLY_WEIGHTED_REGRESSION_STATUS</code> and
<code>decodeGPUMultiscaleGeographicallyWeightedRegressionStatus</code>. Its codes retain the existing numbers
shared with the first five <code>GPU_ITERATIVE_REGRESSION_STATUS</code> values used by generalized GWR:</p>
<table>
<thead>
<tr>
<th>Code</th>
<th>Meaning</th>
</tr>
</thead>
<tbody>
<tr>
<td>0</td>
<td>Converged</td>
</tr>
<tr>
<td>1</td>
<td>Excluded observation</td>
</tr>
<tr>
<td>2</td>
<td>Singular or insufficient local design</td>
</tr>
<tr>
<td>3</td>
<td>Iteration limit reached</td>
</tr>
<tr>
<td>4</td>
<td>Numerical failure or invalid numerical parameters</td>
</tr>
</tbody>
</table>
<p>For MGWR, the code on each included row describes the entire coupled backfit; one failed component
invalidates all included rows. An excluded row always receives code 1. These codes do not report
whether a component bandwidth search reached its own search tolerance. GWR <code>localStatus</code>, MGWR
backfit convergence, and bandwidth-search convergence describe separate facts. In particular,
GWR code 1 means singular, and must be decoded with the GWR decoder rather than the MGWR decoder.
The website readout uses the MGWR decoder to distinguish converged rows, excluded rows, displayed
finite iteration-limit estimates, and failed rows.</p>
<h2 id="pooled-spatial-panels-and-regimes">Pooled spatial panels and regimes</h2>
<p><code>GPUSpatialPanelRegression</code> defaults to pooled spatial-lag 2SLS with a
common intercept, slopes and spatial autoregressive coefficient across periods and entities.
Inputs are balanced and period-major, with a single time-invariant entity weights graph. It expands
<code>I_periods ⊗ W</code> on GPU and delegates estimation to <code>GPUSpatialTwoStageLeastSquares</code>; output layout,
instrument-order options and solver diagnostics are those of that contributor. There are no
cross-period edges. The default <code>effects: 'pooled'</code> retains this behavior. <code>effects: 'unit'</code> adds a GPU within transformation; see <a href="/luma-contributors/docs/api/gpu-spatial-analysis/panel-fixed-effects">Unit fixed effects</a> for absorbed intercepts, degrees of freedom and diagnostic limits. Random effects and dynamic panels are unsupported.
Its classical pooled covariance is not clustered by entity or robust to serial correlation.</p>
<p><code>GPUSpatialRegimeRegression</code> fits a separate intercept and every slope for each caller-supplied
spatial regime. <code>regimes</code> contains integer IDs in <code>[0, regimeCount)</code>; other IDs and masked/nonfinite
rows are excluded. It publishes regime-major <code>coefficients</code>, row-level <code>fitted</code>, regime <code>counts</code>,
and regime <code>status</code> (0 fitted, 1 insufficient observations, 2 singular/nonfinite design). Predictors
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
per regime), so results are reproducible; rows are limited to 2^24 and the device's binding size.</p>
<h2 id="evidence-and-performance-limits">Evidence and performance limits</h2>
<p>The committed generator
<a href="../../../modules/analysis/test/oracles/geospatial-expansion-statistics.py"><code>geospatial-expansion-statistics.py</code></a>
records package versions, synthetic input provenance, float32 rounding and reference types. It uses
direct esda LOSH/Geary, direct constrained-bandwidth mgwr backfitting, direct spreg panel and regime
fits, independent local statsmodels GLMs, and explicit statsmodels/NumPy compositions for grid CV
and conditional uncertainty. SciPy distance/NumPy bin compositions are labeled as compositions,
not direct spatstat or scikit-gstat parity.</p>
<p>Focused browser tests compare these references and exercise exact bin endpoints, signed
cross-covariance, masks, upstream overflow, constant variables, empty regimes, singularity, forced
non-convergence, candidate selection, conditional uncertainty and recipe composition. Node tests
parse the generated WGSL and check the default eight-storage-buffer limit, including all optional
inputs. These are correctness checks, not evidence of a speedup.</p>
<p>The new pair traversal is <code>O(rows² * bins)</code>. Local GLMs are quadratic per iteration; multiscale
backfitting is quadratic per sweep and variable, multiplied by candidate count when selecting
bandwidths. Row limits bound the declared shape but do not promise a suitable interactive latency
at the maximum shape. Benchmark compilation, GPU execution, scratch memory and repeated updates
on the intended device and workload before choosing these paths over established CPU libraries.</p>
<p>Adaptive neighbor bandwidths and degenerate-response status are detailed in <a href="/luma-contributors/docs/api/gpu-spatial-analysis/adaptive-generalized-gwr">Adaptive generalized GWR</a>. Conditional fixed-residual LOSH inference has a separate <a href="/luma-contributors/docs/api/gpu-spatial-analysis/losh-residual-permutation">residual permutation contract</a>.</p>`};export{e as default};