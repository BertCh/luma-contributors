var e={slug:`api/gpu-spatial-analysis/space-time-statistics`,title:`Space-time weights, lagged and differential local Moran, cross-correlation`,html:`<h1 id="space-time-weights-lagged-and-differential-local-moran-cross-correlation">Space-time weights, lagged and differential local Moran, cross-correlation</h1>
<p>STX-02. This section belongs under <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-dynamics">operators-dynamics.md</a>; it lives in its own
file until that file is free of other edits. Import from <code>@luma-contributors/analysis/gpu-spatial-analysis</code>.
Cube layouts are those of <code>GPUCubeLayoutTranspose</code>: <code>'cell-major'</code> is <code>cell * periodCount + period</code>,
<code>'period-major'</code> is <code>period * cellCount + cell</code>.</p>
<h2 id="space-time-weights-gpuspatialweightsalgebra-operation-spacetime">Space-time weights: <code>GPUSpatialWeightsAlgebra</code> operation <code>'spaceTime'</code></h2>
<p><code>T (x) W</code>: a spatial weights matrix <code>W</code> crossed with a temporal lag matrix <code>T</code>. The result is the ordinary
sparse <code>GPUSpatialWeights</code> CSR over <code>cellCount * periodCount</code> rows (ascending neighbor IDs per row), so the
permutation tests, <code>GPUSpatialLag</code>, global statistics and multiple testing read it unchanged.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUSpatialWeightsAlgebra</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  operation: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'spaceTime'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  weights, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// W over cellCount cells</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  periodCount,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  layout: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'period-major'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// or 'cell-major'</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  lags: [{lag: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}, {lag: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, weight: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0.5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, includeSelf: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// rows = cellCount * periodCount, with status.overflow</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span></code></pre>
<p>Row (cell <code>i</code>, period <code>t</code>) lists, for each lag <code>l</code>, the cells <code>j</code> of row <code>i</code> of <code>W</code> at period <code>t - l</code> with
weight <code>T * w_ij</code>. Lag 0 is the contemporaneous neighborhood, lags above 0 look into the past, negative lags
look ahead. Entries outside <code>[0, periodCount)</code> are dropped, so the first <code>max(lag)</code> periods of a pure lag
matrix are islands. <code>includeSelf</code> adds the same cell at the lagged period with spatial weight 1 (<code>I + W</code>
for that lag) and replaces a diagonal slot of <code>W</code>; it is refused at lag 0. Lags must be distinct (at most
16) and temporal weights positive. <code>output.distances</code> is not supported. The result keeps the <code>'raw'</code> or
<code>'binary'</code> transform of <code>W</code> only for unit temporal weights without <code>includeSelf</code>; otherwise it is
<code>'unknown'</code>, so row-standardize afterwards with <code>GPUSpatialWeightsTransform</code> when the permutation test
should see rows that sum to 1.</p>
<p>Reference: a dense Kronecker sum built independently in the spec (both layouts, negative lags, self
entries, overflow).</p>
<h2 id="lagged-and-differential-local-moran-gpuspacetimelocalmoran">Lagged and differential local Moran: <code>GPUSpaceTimeLocalMoran</code></h2>
<p>Like analytic local Moran, permutation scoring is unavailable when either supplied neighborhood
loss word (<code>weights.status.overflow</code> or <code>candidateOverflow</code>) is nonzero. Every p-value and
observed statistic is NaN, and the local permutation completeness word is 1 for either loss stage.
Inspect the input flags separately to identify the loss stage. Clearing
the flags with complete weights restores inference on the same graph.</p>
<p>The Knox, Mantel and Jacquez contributors also suppress inference on either pair/neighborhood
loss word: their entire float summary is NaN, float statistics are NaN, integer count statistics
are zero, and an optional permutations-used output is zero. The Mantel operation and CPU twin
accept <code>pairs.status: {overflow?, candidateOverflow?}</code> graph-value names; the pure CPU function
accepts the corresponding Uint32Array words in <code>input.status</code>. These conditions are read afresh
on each submission; the two input loss meanings remain distinct.
Suppression writes only the published layout: <code>GPU_SPACE_TIME_SUMMARY_LENGTH</code> summary words,
<code>maximumPermutations + 1</code> statistic words and one permutations-used word. Larger caller views
retain their spare words on complete, suppressed and recovered submissions.
Only <code>overflow</code> and <code>candidateOverflow</code> suppress CPU Mantel; capacity metadata such as
<code>requiredCount</code> is ignored. The decoder represents a suppressed zero-draw frame with
<code>permutationCount: 0, available: false</code>; positive draw counts retain their existing lower bounds.</p>
<p>GeoDa definitions (the GeoDa workbook lab 6c (multivariate local statistics) and the GeoDa 1.22 differential
Moran):</p>
<ul>
<li><code>variant: 'lagged'</code>: bivariate local Moran of one variable at two times,
<code>I_i = z_{i,t} sum_j w_ij z_{j,t-lag}</code> (<code>direction: 'inward'</code>, default); <code>'outward'</code> swaps the roles
(<code>z_{i,t-lag}</code> with the neighbors at <code>t</code>). It runs <code>GPULocalPermutationTest</code> <code>'localBivariateMoran'</code>
with the permuted variable being the second one.</li>
<li><code>variant: 'differential'</code>: the univariate local Moran of the change <code>y_t - y_{t-lag}</code> (GeoDa documents the
differential Moran as the Moran statistic of the difference variable). It runs <code>'localMoran'</code>.</li>
</ul>
<p>The recipe slices period <code>period</code> and <code>period - lag</code> out of the cube (one kernel), then forwards every other
prop (<code>parameters</code>, <code>maximumPermutations</code>, <code>alternative</code>, <code>islands</code>, <code>multipleTesting</code>, <code>output</code>, ...) to the
permutation test. Scaling is esda's: <code>I = (n - 1) z_x lag(z_y) / sum z_x^2</code>, which is <code>(n - 1) / n</code> times the
GeoDa z product. Permutations are conditional on the focal row and drawn over the cells of the period.</p>
<p>GeoDa's caveat applies: the lagged bivariate form ignores in-situ correlation between the two times, so
High-High can reflect temporal inertia rather than diffusion.</p>
<p>The whole cube at once is a composition, not a class: build <code>'spaceTime'</code> weights with <code>lags: [{lag: 1}]</code>
and run <code>localBivariateMoran</code> with <code>values = secondValues = cube</code>. The permutation then draws over the
pooled cube (a different null from the per-period recipe), the first period is an island, and the
standardization is pooled. Use <code>islands: 'keep'</code>.</p>
<p>References: esda 2.10.0 <code>Moran_Local_BV</code> and <code>Moran_Local</code> (fixture
<code>modules/analysis/test/oracles/fixtures/space-time-statistics.json</code>, p-values within five binomial standard
errors), an independent double-precision CPU twin, and the Philox twin for exceedance counts.</p>
<p>{/* lens:GPUSpaceTimeLocalMoran */}</p>
<h2 id="cross-correlation-per-location-gpuspacetimecrosscorrelation">Cross-correlation per location: <code>GPUSpaceTimeCrossCorrelation</code></h2>
<p><code>GPUSpaceTimeCrossCorrelation</code>: periodCount is at most 2,147,483,647; cube and correlation address
products must fit uint32.</p>
<p>For each cell and lag <code>l</code> (distinct integers, <code>|l| &#x3C;= periodCount - 4</code>, at most 64) the Pearson correlation of
the overlapping pairs <code>(first[t], second[t + l])</code>, so a positive lag asks whether <code>second</code> follows <code>first</code>.
The p-value is Fisher's z test, <code>erfc(|atanh r| sqrt(n - 3) / sqrt 2)</code>. It assumes serially independent
observations: autocorrelated series give p-values that are too small, so difference or prewhiten first.
Pairs with a non-finite value, fewer than 4 overlapping periods or a constant series are NaN and outside the
family. <code>multipleTesting</code> (default <code>'benjamini-hochberg'</code>; use <code>'benjamini-yekutieli'</code> for arbitrary
dependence, which neighboring cells and adjacent lags are) adjusts across every tested (cell, lag) pair
through <code>GPUMultipleTestingAdjustment</code>. Outputs are indexed <code>cell * lags.length + lagIndex</code>: <code>correlations</code>,
<code>pValues</code>, optional <code>adjustedPValues</code>, <code>pairCounts</code>, <code>testedCount</code>. Both layouts are read in place.
The optional outputs are independent: <code>testedCount</code> is a one-row uint32 count of finite raw p-values
even without <code>adjustedPValues</code>. It is rewritten on every submission, including zero when no pair
can be tested. <code>pairCounts</code> records overlap lengths even for untestable constant or missing series.
<code>testedCount</code> may share a buffer with <code>pValues</code> in disjoint ranges, with or without adjusted p-values;
counting uses private storage and a separate publication pass.</p>
<p>Reference: numpy <code>corrcoef</code>, scipy <code>norm.sf</code> for the Fisher p-values and <code>false_discovery_control</code> for BH
(same fixture), plus an independent double-precision implementation with an exact <code>erfc</code>. f32 sums, so
correlations agree to 1e-4 and Fisher p-values to 1e-5 absolute plus 1e-3 relative.</p>
<p>{/* lens:GPUSpaceTimeCrossCorrelation */}</p>`};export{e as default};