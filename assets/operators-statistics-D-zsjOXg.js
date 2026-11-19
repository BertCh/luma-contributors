var e={slug:`api/gpu-spatial-analysis/operators-statistics`,title:`GPU Spatial Analysis: Spatial statistics`,html:`<h1 id="spatial-statistics">Spatial statistics</h1>
<p>Space-time clustering (ST-DBSCAN and ST-OPTICS through the optional <code>times</code> of <code>GPUSpatialClustering</code> and
<code>GPUOPTICS</code>) is described in <a href="/luma-contributors/docs/api/gpu-spatial-analysis/space-time-clustering">Space-time clustering</a>. Space-time statistics on
cubes are in <a href="/luma-contributors/docs/api/gpu-spatial-analysis/space-time-statistics">Space-time statistics</a>.</p>
<h3 id="gpuglobalspatialstatistics"><code>GPUGlobalSpatialStatistics</code></h3>
<p>Scalable rows are bounded by 4,294,963,199 and actual device bindings. Additive uint32 count outputs
preserve exact populations; float32 statistics remain rounded. Tiled reductions are the default;
mode reference retains its previous limit.</p>
<p>Summarise a tract column with global Moran’s I, Geary’s C and General G from queen weights, each with an analytic z-score and p-value.</p>
<p>::lens{scenario=gpu-spatial-analysis.global-spatial-statistics rung=result caption="Global Moran I = 0.60 (z 29.8), Geary and G for 791 Chicago tracts, with p-values."}</p>
<p>Global spatial autocorrelation from a <code>GPUSpatialWeights</code> CSR and a value column, each statistic with
its analytic expectation, variance, z-score and normal p-value (<code>alternative</code>, default two-sided).
Pick the statistics with <code>statistics</code> (compile-time). The <code>output.summary</code> layout (<code>GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT</code>) is fixed,
and blocks that were not requested hold NaN. <code>values</code> and <code>secondValues</code> may be <code>float32</code> or
<code>uint32</code> (for example the <code>uint32</code> counts of <code>addPointsInPolygonsChoroplethRecipe</code>); counts are
converted to f32 by one node, exact below 2^24, so the results equal those of an f32 copy bit for bit.</p>
<ul>
<li><code>moran</code>: Moran's I with esda's <code>VI_norm</code> and <code>VI_rand</code> (sample kurtosis).</li>
<li><code>geary</code>: Geary's C with <code>VC_norm</code> and <code>VC_rand</code>.</li>
<li><code>getisOrdG</code>: Getis-Ord General G, which needs non-negative ratio-scale data: when any analysis
value is negative the whole G block is NaN. Its randomization variance
uses the Getis–Ord analytical moments, evaluated in centered form to avoid f32 cancellation in
<code>E[G^2] - E[G]^2</code>. Both <code>varianceNormality</code> and <code>varianceRandomization</code> report this variance;
their z and p fields use the same normal approximation. These are analytical results, not
sampled permutations. <a href="https://pysal.org/esda/_modules/esda/getisord.html">esda.G</a> calls these
<code>VG</code> and <code>z_norm</code> and binarizes its weights. Match it with binary weights and <code>islands: 'keep'</code>.
This card retains row-standardized weights for all three statistics, so its G is the weighted
generalization. The former NaN normality fields were unpopulated slots, not failed arithmetic.
Zeros are allowed when the pair-product denominator is positive; a zero denominator makes G
and its variance undefined. Positive constant values give G = E[G], variance zero and NaN z/p.</li>
<li><code>bivariateMoran</code>: <code>(n / S0) sum w_ij zx_i zy_j / sqrt(sum zx^2 sum zy^2)</code>, which equals esda's
<code>Moran_BV</code> for row-standardized weights. The exact variance under randomization of <code>y</code> is
<code>E = 0</code>. esda itself offers only permutation inference.</li>
<li><code>joinCount</code>: BB, BW and WW joins of a binary column (nonzero is black) over binarized weights.
Counts are exact integers, also available as ordered-pair <code>output.joinCounts</code>. Cliff-Ord randomization
moments are given for BB and BW.</li>
</ul>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUGlobalSpatialStatistics</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  weights, values, statistics: [</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'moran'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'geary'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'getisOrdG'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'joinCount'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {summary, joinCounts}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// summary[GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.moran + GPU_GLOBAL_SPATIAL_STATISTIC_FIELD.zRandomization]</span></span></code></pre>
<p>Neighbors, islands and re-standardization follow the shared rules of the local statistics (see
<code>GPUHotSpotAnalysis</code> and <code>GPULocalMoran</code>):</p>
<ul>
<li><strong>Neighbors.</strong> A neighbor is a slot with a positive weight to another analysis row. Zero-weight
slots, the row's own slot and out-of-range IDs are not neighbors.</li>
<li><strong>Re-standardization.</strong> Weights are used as given, except when <code>weights.information.transform</code> is <code>'row'</code>:
each row is then divided by the sum of its active weights everywhere (<code>S0</code>, <code>S1</code>, <code>S2</code>, the reverse
weights <code>w'_ji</code> and the cross products), so masked, missing and island neighbors drop out and the
row still sums to 1 (esda on the subset <code>W</code>, re-row-standardized).</li>
<li><strong>Islands.</strong> An island is an included row (mask nonzero, every value finite) with no active slot.
The rule looks at out-links only; for symmetric weights it equals the old in-or-out rule.
<code>islands: 'exclude'</code> (default, compile-time) removes islands from <code>n</code>, the moments, <code>S0</code>, <code>S1</code>,
<code>S2</code> and from the neighbors of other rows, in one pass with no cascade (a row whose only neighbors
were islands keeps its place in <code>n</code> and adds no weight). <code>islands: 'keep'</code> is esda's convention.
The summary <code>islandCount</code> reports the islands under either policy. esda parity needs <code>'keep'</code>.</li>
<li><strong>Incomplete weights.</strong> If either <code>weights.status.overflow</code> or <code>weights.status.candidateOverflow</code>
is nonzero, the weight sums, mean, variance and every statistic block (including analytic
p-values) are NaN. Population, island and black counts remain available; <code>output.joinCounts</code>
holds 0 as an unavailable value, not an observed zero. Display consumers must check both loss
flags: unavailable means or lags have no quadrant, and join counts are unavailable until recovery.
This applies to packed and chunked weights in both <code>mode: 'tiled'</code> and
<code>mode: 'reference'</code>, even when no output status is requested. Pass the producer's <code>outputWeights</code>
directly so its loss flags travel with the CSR buffers.
Observed joins use up to one fractional digit: asymmetric weights can produce half joins.
The showcase tooltip reads the published join class (the no-class sentinel means no data),
and the quadrant legend counts unavailable rows separately from its reported island total.</li>
<li><strong>Loss diagnostics.</strong> Optional <code>output.status.overflow</code> and <code>output.status.candidateOverflow</code>
are separate uint32 scalars. They forward the producer's corresponding flags as 0 or 1,
and write 0 for an absent input flag. Overflow reports final capacity loss; candidateOverflow
reports intermediate loss. Contiguity pair-only loss reports <code>[0, 1]</code>, even if the final slots
are ample. The statistic forwards these causes independently.
Each is overwritten on every submission, so
inference resumes when the producer clears both flags. Increasing final output capacity cannot
recover dropped candidates. The summary layout, scalar allocation ownership and baseline limit
of eight storage bindings per kernel are unchanged.
Input <code>weights.status.overflow</code> needs at least one uint32 word; spare words are ignored.</li>
<li><strong>Chunk status.</strong> Loss flags must be aggregated on outer <code>weights.status</code>. Local Moran,
hot-spot, local Geary and both global modes reject loss flags on individual CSR chunks at
construction, even if the outer port is also present, rather than silently ignore them.</li>
<li><strong><code>alternative</code></strong> (<code>'two-sided'</code> default, <code>'greater'</code>, <code>'lesser'</code>; compile-time) selects
the tail of every analytic p field, including the join counts: <code>'greater'</code> is the upper tail of z
(for example positive autocorrelation of Moran's I), <code>'lesser'</code> the lower tail. <code>'directed'</code> and
<code>'folded'</code> are permutation conventions and throw. <code>getInferenceMetadata()</code> returns a
<code>GPUInferenceMetadata</code> for a legend. Under non-Gaussian data prefer the randomization fields.</li>
</ul>
<p>The summary block reports <code>n</code>, <code>S0</code>, <code>S1</code>, <code>S2</code>, the mean, the variance, the black count and the
island count. <code>n</code> is counted in <code>u32</code> and the mean gets a second-pass correction
(<code>mean += sum(x - mean) / n</code>), so a large offset does not bias the variance; the row count must be
at most 4,294,963,199 for scalable routes, with device binding limits and exact uint32 counts.
The statistics do not change when the values are multiplied by a constant, so the values are divided
by a power of two near their mean magnitude before the moment sums (an exact operation; only the
reported mean and variance are scaled back) and the weights enter the variances as <code>S1 / S0^2</code> and
<code>S2 / S0^2</code>. Large data therefore keeps its inference: 40,000 values near 300,000, a standard
deviation of 5e7, or 1.2e7 rows with 60 neighbors no longer overflow float32 in the General G,
Moran or Geary variances.
<code>w_ji</code> comes from a binary search of row <code>j</code>. Column sums come from a deterministic transpose: a
stable sort of slots by neighbor. Per-row partials are reduced by fixed-shape workgroup trees, and
join counts by u32 atomics. Results are bitwise reproducible, and values, weights and the mask
change per frame without a rebuild. Inference is NaN when <code>n &#x3C; 4</code>, <code>S0 = 0</code>, or the values are
constant (G reports zero variance for positive constant values, with NaN z/p). Constant values give NaN Moran's I and Geary's C: the centering corrects the f32 mean
quotient with one <code>fma</code> remainder step, so constant data centers to exactly 0 on every adapter.
<code>E[I]</code> and <code>E[C]</code> are NaN below two rows (join counts of an empty input stay 0).</p>
<h4 id="how-it-computes">How it computes</h4>
<p>{/* lens:GPUGlobalSpatialStatistics */}</p>
<p>::lens{scenario=gpu-spatial-analysis.global-spatial-statistics rung=datum bind=page datum=420 caption="Inspect tract 420: its poverty value is one of 791 rows entering the global sums."}
::lens{scenario=gpu-spatial-analysis.global-spatial-statistics rung=formula caption="Follow 4 equations: Moran’s I, its expectation, the randomisation z-score and the two-sided p-value."}
::lens{scenario=gpu-spatial-analysis.global-spatial-statistics rung=kernel caption="Inspect the row-gather, column-sum and finalize kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.global-spatial-statistics rung=graph caption="Trace queen weights, row standardisation and the fixed-order reductions into 52 summary values."}</p>
<h3 id="gpuhotspotanalysis-and-gpulocalmoran"><code>GPUHotSpotAnalysis</code> and <code>GPULocalMoran</code></h3>
<p><code>GPULocalMoran</code>: Scalable rows are bounded by 4,294,963,199 and actual device bindings. Additive
uint32 count outputs preserve exact populations; float32 statistics remain rounded.</p>
<p><code>GPUHotSpotAnalysis</code>: Scalable rows are bounded by 4,294,963,199 and actual device bindings.
Additive uint32 count outputs preserve exact populations; float32 statistics remain rounded.</p>
<p><code>GPUHotSpotAnalysis</code>, <code>GPULocalMoran</code> and <code>GPULocalGeary</code> accept optional uint32 scalars
<code>output.status.invalidCount</code>, <code>output.status.overflow</code> and <code>output.status.candidateOverflow</code>.
<code>invalidCount</code> counts selected input rows with nonfinite values; masked rows, islands and
undefined analytic tests do not count. <code>overflow</code> and <code>candidateOverflow</code> forward the producer's
corresponding final-capacity and intermediate-loss flags.
Each is 0 or 1,
and is 0 when its input port is absent. All requested status scalars are overwritten on every
encoding, independently of multiple-testing correction. Candidate loss is kept separate: increasing
the final capacity cannot recover candidates already dropped upstream.</p>
<p><code>GPUHotSpotAnalysis.selfWeight</code> accepts a numeric default or a packed float32 view from
<code>getGPUHotSpotAnalysisParameterValues</code> hosted in a <code>GPUParameterBuffer</code>. Updating that buffer
between encodings switches between Gi (<code>0</code>) and Gi* (positive), or changes the focal weight,
without rebuilding the graph. Numeric defaults and the packer reject positive weights that
round below the smallest normal float32 (<code>2^-126</code>), preventing underflow or GPU flushing from
silently changing Gi* to Gi. Exactly zero remains valid for Gi.
The existing <code>parameters</code> view still controls fixed moments;
its layout and the fixed confidence bins are unchanged.</p>
<p>Score tract values against neighbouring values with local Moran’s I, analytic conditional-randomisation z-scores, and corrected significance.</p>
<p>::lens{scenario=gpu-spatial-analysis.local-moran rung=result caption="Compare poverty percentages and local I across 791 Chicago tracts; significance is reported separately."}</p>
<p><code>GPUHotSpotAnalysis</code>: Score each tract with Getis-Ord Gi* from binary queen weights, returning z-scores, 90, 95 and 99% confidence bins and corrected p-values.</p>
<p>::lens{scenario=gpu-spatial-analysis.hot-spot-analysis rung=result caption="Gi* z-scores for 791 Chicago tracts; 265 are significant at 95% after Benjamini-Hochberg correction."}</p>
<p>Local spatial autocorrelation on caller-supplied square <code>weights</code>. Rows count when their <code>mask</code> is
nonzero and their value is finite; excluded rows are neither foci nor neighbors. <code>n</code>, the mean <code>X</code>
and the population standard deviation <code>S</code> are taken over the analysis rows (see islands below), or
from per-frame <code>fixedMoments</code>. The mean gets a second-pass correction (<code>mean += sum(x - mean) / n</code>),
and the centered values are the stored first-pass deviations minus the mean of those deviations (one
scheme with <code>GPUGlobalSpatialStatistics</code>, the permutation tests and <code>GPUEmergingHotSpots</code>), so a large
offset biases neither the variance nor the z-scores; <code>n</code> is counted in <code>u32</code>. Sums run in slot order
with fixed-order reductions, so outputs are bitwise reproducible. Analytic p-values are never exactly
0: they are floored at the smallest normal float32 (1.17549435e-38), the same rule in the global,
local, hot-spot and emerging-hot-spot statistics.</p>
<p>Shared semantics (also <code>GPULocalGeary</code>):</p>
<ul>
<li>
<p><strong>Inputs and outputs.</strong> <code>values</code> is <code>float32</code> or <code>uint32</code> (counts are converted to f32 by one
node, exact below 2^24), so the <code>uint32</code> counts of a points-in-polygons aggregation chain without a
cast. Every caller-owned result view lives in one <code>output</code> bag (<code>GPUHotSpotAnalysisOutput</code>,
<code>GPULocalMoranOutput</code>, <code>GPULocalGearyOutput</code>); the CPU settings type is
<code>GPUSpatialAutocorrelationSettings</code>.</p>
</li>
<li>
<p><strong>Neighbors.</strong> A neighbor is a slot with a positive weight that points at another included row.
Zero-weight slots, the focal row's own slot and out-of-range IDs are not neighbors: they are not
counted in <code>neighborCounts</code> or <code>minimumNeighbors</code>. If either <code>weights.status.overflow</code> or
<code>weights.status.candidateOverflow</code> is nonzero, the weights are incomplete. Floating statistic
and inference outputs (including raw and adjusted p-values) are NaN, and significance classes
(<code>quadrants</code> or <code>bins</code>) are 0. This also suppresses ungated Moran quadrants. The statistic's
own optional status forwards both producer flags. <code>neighborCounts</code> is 0 for every row during
loss, meaning unavailable, not that the retained CSR has no neighbors. Population/island counts
retain their meanings; <code>invalidCount</code> still counts nonfinite input values rather than
incomplete neighborhoods. When loss clears, active-neighbor counts are recomputed normally.</p>
</li>
<li>
<p><strong>Re-standardization.</strong> Weights are used as given, except when <code>weights.information.transform</code> is
<code>'row'</code>: then each row's weights are divided by their sum over the active neighbors, so masked or
missing neighbors drop out and the row still sums to 1 (esda on the subset <code>W</code>, re-row-standardized,
not a lag that shrinks toward 0). Without masking this changes nothing beyond rounding. Weights
with no declared transform are never rescaled.</p>
</li>
<li>
<p><strong>Islands.</strong> An island is an included row with no neighbor among the included rows.
<code>islands: 'exclude'</code> (default, compile-time) removes islands from <code>n</code>, the mean and the variance and
writes NaN for every output. <code>islands: 'keep'</code> is the esda convention: islands stay in the moments;
their local statistic is the formula value (local I and the spatial lag are 0, local Geary is 0),
and z and p are NaN because they are not tested. The decision is one pass with no cascade: a row
whose only neighbors were islands has no neighbor among the analysis rows, gets NaN, and stays in
<code>n</code>. <code>globalStatistics</code> is <code>[n, mean, variance, sd, islandCount]</code> under either policy. esda
parity specs use <code>'keep'</code>.</p>
</li>
<li>
<p><strong><code>minimumNeighbors</code></strong> (default 1, compile-time integer). Rows with fewer neighbors get NaN z and p
and bin or quadrant 0; the moments are unchanged. The default only rejects rows without a
neighbor, since no row can have a z-score from its own value. Esri recommends at least 8
neighbors for Gi* (pass <code>minimumNeighbors: 8</code>); the normal approximation of local Moran is poor
below about 4.</p>
</li>
<li>
<p><strong><code>multipleTesting</code></strong> (<code>'none'</code> default, <code>'bonferroni'</code>, <code>'holm'</code>, <code>'benjamini-hochberg'</code>,
<code>'benjamini-yekutieli'</code>, compile-time) adjusts the two-sided p-values through
<code>GPUMultipleTestingAdjustment</code>; the family is every row with a defined z-score, independent of any
viewport. The optional <code>output.adjustedPValues</code> holds the adjusted values (equal to <code>pValues</code> for
<code>'none'</code>).
<code>getInferenceMetadata(settings)</code> takes the same <code>GPUSpatialAutocorrelationSettings</code> that are packed
into <code>parameters</code> (validated identically) and returns a <code>GPUInferenceMetadata</code> (two-sided,
analytic, the method, the significance level, the tested-rows definition) for building a legend.</p>
</li>
<li>
<p><strong>Legend contract.</strong> Bins and quadrants mean the same thing for every correction method. A Gi*
bin of level <code>L</code> means the adjusted p-value (the plain p-value for <code>'none'</code>) is at most <code>0.10</code>,
<code>0.05</code> or <code>0.01</code> for <code>L = 1, 2, 3</code>, and the sign is the sign of z. A local Moran quadrant is the
sign pair of the centered value and the lag, reported only where the adjusted p-value is at most
<code>significanceLevel</code>. With <code>'none'</code>, Gi* bins use the critical values <code>1.645, 1.960, 2.576</code>,
which is the same rule.</p>
</li>
<li>
<p><code>GPUHotSpotAnalysis</code> writes the Getis-Ord Gi* z-score per row for arbitrary weights:
<code>z_i = sum_j w_ij (x_j - X) / (S sqrt((n S1_i - W_i^2) / (n - 1)))</code>, with <code>W_i = sum_j w_ij</code> and
<code>S1_i = sum_j w_ij^2</code>. The focal row enters through <code>selfWeight</code> (default 1,
classic Gi*), never through the matrix; a row's own value alone never produces a z-score. The
z-score is unchanged when every value is shifted by a constant. It equals esda's
<code>G_Local(star=True)</code> analytic <code>Zs</code> for positive data (esda normalizes by the sum of values, which
flips the sign when the mean is negative). <code>selfWeight: 0</code> is the exact Ord-Getis Gi (esda
<code>G_Local(star=False)</code>), which standardizes with the mean and variance of the <code>n - 1</code> other
observations: <code>z_i = (sum_j w_ij x_j - W_i mu_i) / (s_i sqrt(((n - 1) S1_i - W_i^2) / (n - 2)))</code>.
With row-standardized weights <code>selfWeight: 1</code> weighs the focal row as much as all its neighbors,
so prefer <code>0</code> there. Optional outputs: the two-sided p-value, the adjusted p-value, the neighbor
count (the focal row not included) and a <code>sint32</code> confidence bin in <code>-3..3</code> (99/95/90% hot or cold).</p>
</li>
<li>
<p><code>GPULocalMoran</code> writes the z-score of local Moran's I with the analytic moments of conditional
randomization, the null esda's permutation test samples: <code>E[L_i] = W_i mu</code>,
<code>Var[L_i] = sigma^2 (N S1_i - W_i^2) / (N - 1)</code> over the other <code>N = n - 1</code> rows. Optional outputs
are <code>localI</code> (<code>(n - 1) c_i L_i / sum c^2</code>, esda's <code>Is</code>; NaN when the values are constant, 0 for
kept islands), the spatial lag <code>L_i = sum_j w_ij c_j</code>,
the p-value, the adjusted p-value, the neighbor count and the esda quadrant code (1 HH, 2 LH, 3 LL,
4 HL; 0 when not significant at the per-frame <code>significanceLevel</code>).
<code>quadrantGating</code> (<code>'analytic'</code> default or <code>'none'</code>, compile-time) chooses what <code>quadrants</code> holds. With
<code>'analytic'</code> it is nonzero only where the analytic p-value (adjusted when <code>multipleTesting</code> is set) is
significant. With <code>'none'</code> it
is the ungated quadrant from the signs of the centered value and the spatial lag, for every included
row with a nonzero centered value and lag, and 0 otherwise, so another test can gate it, for example the
<code>significant</code> mask of <code>GPULocalPermutationTest</code> (esda's <code>q</code> with <code>p_sim</code>). <code>zScores</code> and <code>pValues</code> do
not change. <code>'none'</code> cannot be combined with <code>multipleTesting</code> (the constructor throws).
Local Moran significance does not depend on <code>|c_i|</code>: a row near the mean can be significant.</p>
</li>
</ul>
<p>Checked against esda 2.10 on 200 points with 6-nearest-neighbor weights: <code>localI</code> matches <code>Is</code> to
3e-7, the z-score tracks esda's 99,999-permutation <code>z_sim</code> (correlation 0.999997, largest gap 0.03),
and Gi* matches <code>Zs</code> to 1e-7. A 60-row fixture with an island and masked rows pins <code>G_Local</code> (star
and non-star), <code>Moran_Local</code>, <code>Geary_Local</code> and the statsmodels adjustments for <code>islands: 'keep'</code>,
<code>'exclude'</code> and the re-standardized masked case.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> search</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUNeighborSearch</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  mode: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'radius'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, gridSize: [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">256</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">256</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">], positions,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  parameters: searchParameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph), output: {</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">weights, status: {overflow}}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(search);</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUHotSpotAnalysis</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({weights: search.outputWeights, values, parameters, output: {zScores, bins, pValues}}));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPULocalMoran</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({weights: search.outputWeights, values, parameters, output: {zScores, quadrants}}));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">searchParameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">write</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">getGPUNeighborSearchParameterValues</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({bounds, radius: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">500</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span></code></pre>
<p>Pass the producer's <code>outputWeights</code> directly. Rebuilding a weights object from only its CSR buffers
drops the search's completeness signals. For a line of four points spaced one unit apart, radius
1.1 requires six directed neighbor slots: capacity one reports overflow and suppresses Moran
inference; capacity six permits the defined tests. After candidate loss, even ample final capacity
must leave inference unavailable until the producer reports a complete neighborhood.</p>
<h4 id="how-it-computes-1">How it computes</h4>
<p>{/* lens:GPUHotSpotAnalysis <em>/}
{/</em> lens:GPULocalMoran */}</p>
<p><code>GPULocalMoran</code>:</p>
<p>::lens{scenario=gpu-spatial-analysis.local-moran rung=datum bind=page datum=420 caption="Inspect tract 420: its value, neighbour lag, local I, and corrected p-value."}
::lens{scenario=gpu-spatial-analysis.local-moran rung=formula caption="Follow 2 equations: local I and its conditional-randomisation z-score."}
::lens{scenario=gpu-spatial-analysis.local-moran rung=kernel caption="Inspect 791 focal invocations and the retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.local-moran rung=graph caption="Trace queen weights, row standardisation, local moments, and FDR across 791 tracts."}</p>
<p><code>GPUHotSpotAnalysis</code>:</p>
<p>::lens{scenario=gpu-spatial-analysis.hot-spot-analysis rung=datum bind=page datum=420 view=neighbourhood-view caption="Inspect tract 420: its queen neighbours, Gi* z-score, corrected p-value and confidence bin."}
::lens{scenario=gpu-spatial-analysis.hot-spot-analysis rung=formula caption="Follow the Gi* z-score: a weighted sum of centred values over its neighbour-count spread."}
::lens{scenario=gpu-spatial-analysis.hot-spot-analysis rung=kernel caption="Inspect 791 focal invocations, the moment passes, FDR adjustment and classify kernel."}
::lens{scenario=gpu-spatial-analysis.hot-spot-analysis rung=graph caption="Trace queen weights, moments, Gi* gather, Benjamini-Hochberg adjustment and confidence bins."}</p>
<p>{/* parameters: GPUHotSpotAnalysis generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpuhotspotanalysis-parameters"><code>GPUHotSpotAnalysis</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUHotSpotAnalysisParameterValues(settings?)</code>, writes <code>GPU_HOT_SPOT_ANALYSIS_PARAMETER_LENGTH</code> (1) <code>float32</code> values.</li>
<li>Packer: <code>getGPUSpatialAutocorrelationParameterValues(settings?)</code>, writes <code>GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH</code> (8) <code>float32</code> values.
<ul>
<li>Input port: <code>parameters</code>.</li>
</ul>
</li>
</ul>
<p>Settings of <code>getGPUHotSpotAnalysisParameterValues</code>:</p>
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
<td><code>selfWeight</code> (optional)</td>
<td><code>number</code></td>
<td>Non-negative focal weight. Zero selects Gi, a positive value selects Gi*. Positive weights must round to a normal float32 (at least 2^-126); smaller values are rejected rather than silently switching statistics. Defaults to 1.</td>
</tr>
</tbody>
</table>
<p>Settings of <code>getGPUSpatialAutocorrelationParameterValues</code>:</p>
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
<td><code>significanceLevel</code> (optional)</td>
<td><code>number</code></td>
<td>Two-sided significance level of <code>GPULocalMoran</code> quadrants, in <code>(0, 1)</code>, applied to the adjusted p-value when <code>multipleTesting</code> is set. Defaults to <code>0.05</code>. <code>GPUHotSpotAnalysis</code> ignores it and always reports 90/95/99% bins.</td>
</tr>
<tr>
<td><code>fixedMoments</code> (optional)</td>
<td><code>GPUSpatialAutocorrelationFixedMoments</code></td>
<td>Optional moments that replace the moments of the current rows.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<p>{/* parameters: GPULocalMoran generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpulocalmoran-parameters"><code>GPULocalMoran</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUSpatialAutocorrelationParameterValues(settings?)</code>, writes <code>GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH</code> (8) <code>float32</code> values.
<ul>
<li>Input port: <code>parameters</code>.</li>
</ul>
</li>
</ul>
<p>Settings of <code>getGPUSpatialAutocorrelationParameterValues</code>:</p>
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
<td><code>significanceLevel</code> (optional)</td>
<td><code>number</code></td>
<td>Two-sided significance level of <code>GPULocalMoran</code> quadrants, in <code>(0, 1)</code>, applied to the adjusted p-value when <code>multipleTesting</code> is set. Defaults to <code>0.05</code>. <code>GPUHotSpotAnalysis</code> ignores it and always reports 90/95/99% bins.</td>
</tr>
<tr>
<td><code>fixedMoments</code> (optional)</td>
<td><code>GPUSpatialAutocorrelationFixedMoments</code></td>
<td>Optional moments that replace the moments of the current rows.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="gpulocalgeary"><code>GPULocalGeary</code></h3>
<p>Scalable rows are bounded by 4,294,963,199 and actual device bindings. Additive uint32 count outputs
preserve exact populations; float32 statistics remain rounded.</p>
<p>Measure how much each tract differs from its neighbours with local Geary c, where small values mean similar surroundings.</p>
<p>::lens{scenario=gpu-spatial-analysis.local-geary rung=result caption="Local Geary c across 791 Chicago tracts, mean 0.81; small values mark tracts like their neighbours."}</p>
<p>Neighbor, row re-standardization and <code>islands</code> semantics follow <code>GPULocalMoran</code>. The output is
<code>output.statistic</code>; the parameters and the permutation adapter are described under
Additional spatial statistics below.</p>
<h4 id="how-it-computes-2">How it computes</h4>
<p>{/* lens:GPULocalGeary */}</p>
<p>::lens{scenario=gpu-spatial-analysis.local-geary rung=datum bind=page datum=420 caption="Inspect tract 420: its value, its queen neighbours’ values and the resulting c."}
::lens{scenario=gpu-spatial-analysis.local-geary rung=formula caption="Follow the rule: squared differences to each neighbour, weighted, divided by the variance."}
::lens{scenario=gpu-spatial-analysis.local-geary rung=kernel caption="Inspect 791 focal invocations over the retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.local-geary rung=graph caption="Trace queen weights, row standardisation, moments and the local Geary kernel."}</p>
<h3 id="gpulocalspatialheteroscedasticity"><code>GPULocalSpatialHeteroscedasticity</code></h3>
<p>Scalable rows are bounded by 4,294,963,199 and actual device bindings. Additive uint32 count outputs
preserve exact populations; float32 statistics remain rounded.</p>
<p>Descriptive LOSH divides each row's weighted neighbor residual power by the study-wide mean
residual power. Positive usable neighbors are normalized within each row. Masked rows, nonfinite
values and islands have no residual; an all-zero or absent residual field leaves the statistic
undefined. The per-row <code>output.status</code> codes remain <code>0</code> defined, <code>1</code> excluded, <code>2</code> undefined and
<code>3</code> incomplete weights.</p>
<p>Either upstream overflow or candidateOverflow (including a flag on an individual chunk) makes
all LOSH statistics and normalized residuals NaN, every per-row status INCOMPLETE, and residualCount
zero. The same compiled graph recovers after both input loss causes clear. The local-variability
recipe therefore suppresses LOSH consistently with its Moran stage.</p>
<p>The global mean uses parallel maximum, count and sum reductions. Residuals are divided by their
maximum before summation, so a finite mean does not overflow merely because its unscaled total
would overflow. Exponent-scaled quotients avoid underflow of a large divisor's reciprocal;
the mean is reconstructed from a bounded mantissa and two finite power-of-two scales.
No invocation scans all rows to form the mean. Float32 results may differ from
the former running mean through reduction order.</p>
<p>Per-frame <code>residualPower</code> accepts a packed float32 view from
<code>getGPULocalSpatialHeteroscedasticityParameterValues</code> hosted in a <code>GPUParameterBuffer</code>; write
<code>1</code> or <code>2</code> between encodings to switch absolute deviations and squared deviations. Numeric
powers remain supported as uploaded defaults. View shapes, optional outputs and chunk topology
are compile-time. The <a href="/luma-contributors/docs/api/gpu-spatial-analysis/losh-residual-permutation">conditional residual inference</a> wrapper
accepts the same parameter view.</p>
<p>{/* parameters: GPULocalSpatialHeteroscedasticity generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpulocalspatialheteroscedasticity-parameters"><code>GPULocalSpatialHeteroscedasticity</code> parameters</h4>
<ul>
<li>Packer: <code>getGPULocalSpatialHeteroscedasticityParameterValues(settings?)</code>, writes <code>GPU_LOCAL_SPATIAL_HETEROSCEDASTICITY_PARAMETER_LENGTH</code> (1) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPULocalSpatialHeteroscedasticityParameterValues</code>:</p>
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
<td><code>residualPower</code> (optional)</td>
<td><code>1 | 2</code></td>
<td>Absolute-deviation power: 1 for absolute deviation, 2 for variance. Defaults to 2.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="gpumultivariatelocalgearypermutationtest"><code>GPUMultivariateLocalGearyPermutationTest</code></h3>
<p>Conditional randomization tests multivariate local Geary statistics over complete, unmasked
rows. Islands and incomplete rows remain untested; incomplete weights invalidate inference.</p>
<p>The exact and tiled strategies accept an optional <code>parameters</code> uint32 view from
<code>getGPUMultivariateLocalGearyPermutationTestParameterValues</code>, hosted in a <code>GPUParameterBuffer</code>.
Its three words are <code>[seedLow, seedHigh, float32 bits of significance]</code>; both Philox key words
retain all 32 bits. Updating this view changes the seed and classification threshold on the same
compiled graph, including with multiple-testing correction. When the view is present it overrides
numeric <code>seed</code> and <code>significance</code> defaults; omitting it uploads those defaults through the same
parameter-buffer layout. Use the packer to validate each frame's settings.</p>
<p>Values, mask, CSR contents and parameter contents are per-frame. View shapes, variable count,
maximum permutations, strategy, tile sizes, maximum neighbors, tail convention, correction choice
and optional outputs stay compile-time. The tiled simulation stages the parameter words beside
the valid-row count to keep each kernel within eight storage-buffer bindings. The conditional
randomization, tie handling and multiple-testing formulas are unchanged.</p>
<p>{/* parameters: GPUMultivariateLocalGearyPermutationTest generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpumultivariatelocalgearypermutationtest-parameters"><code>GPUMultivariateLocalGearyPermutationTest</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUMultivariateLocalGearyPermutationTestParameterValues(settings?)</code>, writes <code>GPU_MULTIVARIATE_LOCAL_GEARY_PERMUTATION_TEST_PARAMETER_LENGTH</code> (3) <code>uint32</code> values.</li>
</ul>
<p>Settings of <code>getGPUMultivariateLocalGearyPermutationTestParameterValues</code>:</p>
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
<td><code>readonly [number, number]</code></td>
<td>Two unsigned 32-bit Philox key words, including all high bits. Defaults to [0, 0].</td>
</tr>
<tr>
<td><code>significance</code> (optional)</td>
<td><code>number</code></td>
<td>Significance threshold in (0, 1), compared as float32. Defaults to 0.05.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="gpuassuncaoreisstandardization"><code>GPUAssuncaoReisStandardization</code></h3>
<p>Standardize each tract's rate against the pooled rate and Poisson noise, turning event counts and populations into one z-score per tract.</p>
<p>::lens{scenario=gpu-spatial-analysis.assuncao-reis-standardization rung=result caption="Standardized poverty rates of 791 tracts range from -1.5 to 3.3; the 3 zero-population tracts stay missing."}</p>
<p>Empirical-Bayes standardized rates of Assuncao and Reis (1999), the transformation behind
<code>esda.Moran_Rate</code> and <code>esda.Moran_Local_Rate</code>. It writes one standardized value per row and no
spatial statistic; Moran statistics of rate data come from composing it with the existing nodes.
Lives in <code>spatial-autocorrelation/gpu-rate-standardization</code>.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> standardization</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUAssuncaoReisStandardization</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  events, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// float32 or uint32 view (counts are converted to f32, exact below 2^24)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  populations, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// float32 or uint32 view, population at risk</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  mask, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// optional uint32</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    standardizedRates, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// float32 view, NaN for rows outside the analysis</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    statistics, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// optional, [n, b, s2, a, totalPopulation]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    status: {invalidCount} </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// optional uint32 scalar</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  }</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span></code></pre>
<p>Definition (identical to <code>esda.smoothing.assuncao_rate</code>), over the <code>m</code> valid rows with rates
<code>p_i = e_i / n_i</code>: <code>b = sum e / sum n</code>, <code>s2 = sum n_i (p_i - b)^2 / sum n</code>,
<code>a = s2 - b / (sum n / m)</code>, <code>v_i = a + b / n_i</code> (replaced by <code>b / n_i</code> where <code>v_i &#x3C; 0</code>), and
<code>z_i = (p_i - b) / sqrt(v_i)</code>.</p>
<ul>
<li><strong>Negative prior variance.</strong> <code>a</code> is negative when the rates vary less than Poisson noise predicts.
esda does not clamp <code>a</code>: it keeps the negative <code>a</code> wherever <code>v_i</code> stays non-negative and falls
back to the pure Poisson variance <code>b / n_i</code> only for rows where <code>v_i &#x3C; 0</code>. This node does the
same, for parity, and <code>statistics[3]</code> reports the unclamped <code>a</code>. The EB smoothing in
<code>rate-smoothing</code> clamps the prior variance at zero instead (a different, also standard
convention for shrinkage); the two are not meant to agree when <code>a &#x3C; 0</code>.</li>
<li><strong>Missing rows.</strong> A row is valid when it is unmasked, its events are finite and non-negative, and
its population is finite and positive. Other rows write NaN, are excluded from <code>b</code>, <code>s2</code>, <code>m</code>
and the mean population, and are excluded by the downstream statistic. <code>output.status.invalidCount</code>
counts the unmasked rows that were rejected. Rows where <code>v_i</code> is not positive (for example <code>b = 0</code>)
write NaN, as esda's 0 / 0 does.</li>
<li><strong>Precision.</strong> <code>m</code> is a u32; sums are fixed-order two-level workgroup trees (no float atomics,
bitwise reproducible); <code>b</code> gets a second-pass correction and <code>s2</code> is formed from centered
deviations. Scalable count/index bounds and device limits apply.</li>
</ul>
<p>Composition (no statistic code is duplicated):</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> standardizedRates</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> createTransientView</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'rates'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'float32'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, rows);</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUAssuncaoReisStandardization</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({events, populations, output: {standardizedRates}}));</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// EB Moran's I (esda.Moran_Rate, transformation 'r'): row-standardized weights.</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">  new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUGlobalSpatialStatistics</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    weights, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// weights.information.transform === 'row'</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    values: standardizedRates,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    statistics: [</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'moran'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    islands: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'keep'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// esda's convention</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    output: {summary}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  })</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">);</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// EB local Moran (esda.Moran_Local_Rate): GPULocalMoran, or GPULocalPermutationTest('localMoran').</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">  new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPULocalMoran</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    weights,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    values: standardizedRates,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    parameters,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    islands: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'keep'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    output: {zScores, localI}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  })</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">);</span></span></code></pre>
<p><code>summary[moran + statistic]</code>, <code>summary[moran + expected]</code> and <code>summary[moran + zRandomization]</code>
equal <code>Moran_Rate.I</code>, <code>.EI</code> and <code>.z_rand</code> (esda recomputes the Moran of the standardized values and
ignores the EB sampling variance); <code>localI</code> equals <code>Moran_Local_Rate.Is</code>. Row-standardized weights
are re-standardized over the active neighbors, so NaN rows drop out of the neighborhoods of the
rows that remain.</p>
<h4 id="how-it-computes-3">How it computes</h4>
<p>{/* lens:GPUAssuncaoReisStandardization */}</p>
<p>::lens{scenario=gpu-spatial-analysis.assuncao-reis-standardization rung=datum bind=page datum=420 caption="Inspect tract 420: its events, population, crude rate and standardized z."}
::lens{scenario=gpu-spatial-analysis.assuncao-reis-standardization rung=formula caption="Follow z: the tract rate minus the pooled rate, over its Poisson-plus-prior spread."}
::lens{scenario=gpu-spatial-analysis.assuncao-reis-standardization rung=kernel caption="Inspect the three block-sum passes, the mean correction and the standardize kernel."}
::lens{scenario=gpu-spatial-analysis.assuncao-reis-standardization rung=graph caption="Trace the totals, mean correction, variance and per-tract standardization passes."}</p>
<h3 id="gpumultipletestingadjustment"><code>GPUMultipleTestingAdjustment</code></h3>
<p>Exact integer ranks support at most 2,147,483,648 rows, subject to published sort and device
bindings.</p>
<p>Adjust a column of per-tract p-values for multiple testing, returning corrected values that are significant when at most alpha.</p>
<p>::lens{scenario=gpu-spatial-analysis.multiple-testing-adjustment rung=result caption="Hot spot p-values of 754 tested tracts before and after Benjamini-Hochberg: 369 pass 0.05 unadjusted, 265 adjusted."}</p>
<p>Adjusts a column of per-row p-values for multiple testing on the GPU. <code>method</code> is <code>'none'</code>,
<code>'bonferroni'</code>, <code>'holm'</code>, <code>'benjamini-hochberg'</code> or <code>'benjamini-yekutieli'</code>, and
<code>output.adjustedPValues</code> holds the same numbers as statsmodels <code>multipletests</code> (<code>bonferroni</code>, <code>holm</code>, <code>fdr_bh</code>, <code>fdr_by</code>), in
<code>[0, 1]</code>. <code>getMultipleTestingAdjustedPValues</code> is the float64 CPU reference with the same semantics.</p>
<ul>
<li>
<p><strong>Permutation p-value floor.</strong> A permutation p-value is never below <code>m / (P + 1)</code> (<code>m</code> is 2 for the
<code>'two-sided'</code> default, else 1), so with <code>P = 999</code> the two-sided floor is 0.002. Then
Bonferroni and Holm cannot reject in a family of more than 25 rows at <code>alpha = 0.05</code>, and with
10,000 rows Benjamini-Hochberg needs 400 rows at the floor before it can reject. Nothing warns at run
time. <code>getPermutationRejectionFloor({permutationCount, alternative, rowCount, alpha, method})</code>
is the CPU helper that answers it before you run: it returns <code>pValueFloor</code>, <code>canReject</code>,
<code>minimumRowsAtFloor</code> (Infinity when <code>canReject</code> is false) and, for Bonferroni and Holm,
<code>maximumRowCount</code>. For Benjamini-Yekutieli the requirement is multiplied by the harmonic sum of
<code>rowCount</code>. Raise <code>permutationCount</code> (the floor needs <code>P + 1 >= m rowCount / alpha</code> for Bonferroni
and Holm) when <code>canReject</code> is false. Sequential stopping does not change the floor.</p>
</li>
<li>
<p><strong>Family.</strong> The family is every tested row: a row whose p-value is finite and in <code>[0, 1]</code>. NaN
rows are not tested and get NaN. There is no mask prop by design: run the statistic and this node
on the full analysis set and apply a viewport or display mask only when rendering, otherwise
panning changes the family size <code>m</code> and makes the visible rows look more or less significant.</p>
</li>
<li>
<p><strong><code>minimumFamilySize</code>.</strong> <code>m = max(testedRows, minimumFamilySize)</code>; rows without a p-value count as
<code>p = 1</code>. Use it to pin <code>m</code> when the tested set can shrink (for example an analysis mask that
follows the viewport). The optional <code>output.testedCount</code> reports the <code>m</code> actually used.
The GPU contributor accepts <code>minimumFamilySize</code> up to 2^31. The host helper
<code>getMultipleTestingAdjustedPValues</code> and CPU twin <code>computeMultipleTestingAdjustmentOnCPU</code> accept
up to 2^32 - 1. Their allocation and sorting depend only on the supplied p-value rows, including
when the logical family is larger. For Benjamini-Yekutieli the host harmonic factor uses at most
64 terms, then an Euler-Maclaurin expansion with remainder below binary64 rounding.</p>
</li>
<li>
<p><strong>Which method.</strong> Local statistics of neighboring rows share observations, so their p-values are
dependent. Benjamini-Hochberg only controls the false discovery rate under independence or
positive regression dependence; use <code>'benjamini-yekutieli'</code> (Benjamini-Hochberg times
<code>c(m) = sum 1/i</code>, a cost of about <code>ln m</code> in power) for controlled error under arbitrary
dependence. Holm controls the family-wise error rate and is uniformly more powerful than
Bonferroni.</p>
</li>
<li>
<p><strong>Significance.</strong> For every method a row is significant at level <code>alpha</code> exactly when
<code>adjustedPValue &#x3C;= alpha</code>, so 0.10 / 0.05 / 0.01 bins are consistent by construction. Tied
p-values get identical adjusted values.</p>
</li>
<li>
<p><strong>Cost.</strong> <code>none</code> and Bonferroni are O(n). The step procedures add a 30-bit <code>GPUSort</code> plus a
deterministic cumulative maximum (Holm) or minimum (Benjamini) scan in workgroup memory, a few
linear kernels in all. At most 2,147,483,648 rows (ranks are exact uint32 integers), within device limits; results are
independent of scheduling.</p>
</li>
</ul>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUMultipleTestingAdjustment</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  pValues, method: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'benjamini-yekutieli'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, output: {adjustedPValues}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// significant at 5%: adjustedPValues[i] &#x3C;= 0.05</span></span></code></pre>
<h4 id="how-it-computes-4">How it computes</h4>
<p>{/* lens:GPUMultipleTestingAdjustment */}</p>
<p>::lens{scenario=gpu-spatial-analysis.multiple-testing-adjustment rung=datum bind=page datum=420 view=fdr-staircase caption="Inspect tract 420: its p-value, rank in the sorted family and adjusted p-value."}
::lens{scenario=gpu-spatial-analysis.multiple-testing-adjustment rung=formula caption="Follow the step-up rule: a running minimum of p times m over rank."}
::lens{scenario=gpu-spatial-analysis.multiple-testing-adjustment rung=kernel caption="Inspect the count, 30-bit sort, scan and scatter kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.multiple-testing-adjustment rung=graph caption="Trace the hot spot p-values through counting, sorting, the cumulative minimum and scatter."}</p>
<h3 id="gpurasterfalsediscoveryratemask"><code>GPURasterFalseDiscoveryRateMask</code></h3>
<p>Exact integer ranks support at most 2,147,483,648 cells, subject to sort and device binding limits.</p>
<p>Field significance of a p-value raster: adjust every tested cell for multiple testing and mask the cells that survive.</p>
<p>Run it on the p-value output of a Mann-Kendall, Welch or correlation operator so a trend map shows only
cells that remain significant after correction (Wilks 2016: testing every grid point without a
multiplicity adjustment overstates the result). It composes <code>GPUMultipleTestingAdjustment</code> on the cells
whose p-value is finite and in <code>[0, 1]</code> and whose optional <code>validity</code> is nonzero, so the family is exactly
the valid cells and is not changed by a display mask. <code>method</code> is <code>'benjamini-hochberg'</code> (default),
<code>'benjamini-yekutieli'</code> (for dependent local statistics), <code>'holm'</code> or <code>'bonferroni'</code>.</p>
<ul>
<li><code>output.significant</code> is 1 where <code>adjusted &#x3C;= alpha</code>, which is statsmodels' <code>multipletests</code> <code>reject</code>.
<code>alpha</code> is a per-frame parameter (<code>getGPURasterFalseDiscoveryRateMaskParameterValues</code>).</li>
<li>Optional outputs: <code>adjustedPValues</code>, <code>testedCount</code> (the family size <code>m</code>), <code>significantCount</code>,
<code>criticalPValue</code> (the largest raw p-value that is significant, the Benjamini-Hochberg step-up
threshold; NaN when none) and <code>histogram</code>, the counts of tested p-values over equal bins of <code>[0, 1]</code>
by the published <code>GPURasterHistogram</code>. A histogram that is flat means no signal; a spike near zero
means there is some.</li>
<li><strong>Reference.</strong> The adjusted p-values reuse the statsmodels fixtures of <code>GPUMultipleTestingAdjustment</code>;
the mask is checked against those, against the 15 p-values of Benjamini and Hochberg (1995, four
rejections at q = 0.05) and against the textbook step-up and step-down rules written independently.
statsmodels itself was not run for the mask.</li>
<li><strong>float32 limit.</strong> Adjusted p-values are float32, so a cell whose adjusted value is within a few ulp of
<code>alpha</code> may fall on either side of statsmodels' float64 decision.</li>
<li>At most 2,147,483,648 cells, subject to sort and device limits. This is a field operation on a linear cell array, so it needs no raster grid.</li>
</ul>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPURasterFalseDiscoveryRateMask</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  cellCount: width </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">*</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> height,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  pValues: trendPValues,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  validity: trendValidity,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  method: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'benjamini-hochberg'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  parameters: alphaParameters, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// getGPURasterFalseDiscoveryRateMaskParameterValues({alpha: 0.05})</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {significant, criticalPValue}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span></code></pre>
<p>{/* lens:GPURasterFalseDiscoveryRateMask */}</p>
<p>{/* parameters: GPURasterFalseDiscoveryRateMask generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpurasterfalsediscoveryratemask-parameters"><code>GPURasterFalseDiscoveryRateMask</code> parameters</h4>
<ul>
<li>Packer: <code>getGPURasterFalseDiscoveryRateMaskParameterValues(settings)</code>, writes <code>GPU_RASTER_FALSE_DISCOVERY_RATE_MASK_PARAMETER_LENGTH</code> (1) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPURasterFalseDiscoveryRateMaskParameterValues</code>:</p>
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
<td><code>alpha</code></td>
<td><code>number</code></td>
<td>Significance level (false discovery rate for the Benjamini procedures), in <code>(0, 1]</code>.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="gpulocalpermutationtest-and-gpuglobalpermutationtest"><code>GPULocalPermutationTest</code> and <code>GPUGlobalPermutationTest</code></h3>
<p>For recording one row's raw conditional draws, import <code>createLocalPermutationReplay</code> and
<code>LocalPermutationReplayProps</code> from <code>@luma-contributors/analysis/gpu-spatial-analysis</code>.
Add the test to the same recording graph first, then add the returned command-node producer.
Replay requires whole-CSR weights: exactly one chunk covering all test rows. Multi-chunk
weights are rejected when constructing the replay, even after the test has been lowered;
repeating the lowering step cannot enable that unsupported form. A supported whole-CSR test
that has not been lowered instead reports that the test must be lowered first.
The helper borrows the test's prepared inputs and replays its full permutation budget, including
when sequential testing stopped early. Use <code>output.permutationCounts[row]</code> from the test to select
the prefix used by its decision. A zero replay-info count means the row is unavailable or invalid;
suppress its distribution. Captures use <code>gpu-replay</code> provenance and name the pinned row.
The helper snapshots the test, row and both output view references before validating them;
later edits to the caller's props or nested output map cannot replace a validated binding.
Upload new values to the captured row view to change the pin. Row-domain metadata belongs to
the capture; the producer has no <code>keySpace</code> option. The scenario reads <code>replayInfo</code> alongside
<code>nullDistribution</code>, omits a zero-count replay and publishes exactly the reported prefix (for
example five values when five draws were requested into a 99-value allocation). Missing or
out-of-capacity replay counts are rejected before sidecar publication.</p>
<p><code>GPUGlobalPermutationTest</code>: Scalable rows are bounded by 4,294,963,199 and actual device bindings.
Additive uint32 count outputs preserve exact populations; float32 statistics remain rounded.</p>
<p><code>GPULocalPermutationTest</code>: Scalable rows are bounded by 4,294,963,199 and actual device bindings.
Additive uint32 count outputs preserve exact populations; float32 statistics remain rounded.</p>
<p><code>GPULocalPermutationTest</code>: Test each tract’s local Moran I against 99 conditional permutations, returning pseudo p-values, adjusted values and a significance mask.</p>
<p>::lens{scenario=gpu-spatial-analysis.local-permutation-test rung=result caption="Two-sided pseudo p-values of 791 tracts from 99 permutations; 303 are at most 0.05, none after adjustment."}</p>
<p><code>GPUGlobalPermutationTest</code>: Test global Moran’s I against 99 relabelings of the tracts, returning the pseudo p-value and the simulated reference distribution.</p>
<p>::lens{scenario=gpu-spatial-analysis.global-permutation-test rung=result caption="Observed Moran I = 0.60 against 99 simulated values near 0; pseudo p = 0.02, the smallest possible."}</p>
<p>Maximum (<code>GPUGlobalPermutationTest</code>): <code>(maximumPermutations + 1) * ceil(rows / 2048)</code> is at most <code>GPU_GLOBAL_PERMUTATION_TEST_MAXIMUM_PAIR_BLOCK_COUNT</code> (2^24 - 1), so 2^20 permutations allow 15 row blocks (30720 rows). Without <code>stopAfterExceedances</code> the permutations are not chunked (one dispatch); with it they run in batches (see below).</p>
<p>Permutation inference (pseudo p-values) over a <code>GPUSpatialWeights</code> CSR, following PySAL esda's
<code>p_sim</code> with the corrections below. Both contributors read a uint32 parameter view written with
<code>getGPUPermutationParameterValues</code> from a <code>GPUPermutationSettings</code> object
<code>{seed, permutationCount, significanceLevel}</code>. Changing the seed, the permutation count (up to the
compile-time <code>maximumPermutations</code>), the values or the weights never rebuilds the graph. <code>values</code>
and <code>secondValues</code> are <code>float32</code> or <code>uint32</code> (converted to f32, exact below 2^24), and every result
view lives in the <code>output</code> bag (<code>GPULocalPermutationTestOutput</code>, <code>GPUGlobalPermutationTestOutput</code>).</p>
<ul>
<li><code>GPULocalPermutationTest</code> does conditional randomization for <code>localMoran</code>, <code>localGeary</code>, <code>localG</code>,
<code>localGStar</code> (a self weight of 1), <code>localBivariateMoran</code>, <code>localLeeL</code>, <code>localJoinCount</code> and
<code>localBivariateJoinCount</code> (see <strong>Bivariate, join-count and Lee statistics</strong> below). For each tested row and permutation it draws an ordered sample
of <code>k_i</code> distinct other analysis rows (a partial Fisher-Yates over their compacted positions) and
assigns them to the row's weight slots. It writes <code>exceedances</code>, <code>pseudoPValues</code>, an optional
<code>observed</code> statistic, an optional <code>adjustedPValues</code>, and a <code>significant</code> mask. Rows with more than
<code>maximumNeighbors</code> (64 at most) neighbors are not tested and set <code>output.status.overflow</code>.
<strong>Negative data.</strong> <code>localG</code> and <code>localGStar</code> are defined for non-negative data: the ranking
drops the denominator as a positive constant, which fails for negative values (with
<code>alternative: 'greater'</code> the p-value would be the lower tail). Values are only known on the GPU, so
when any analysis row is negative no row is tested: <code>exceedances</code> is
<code>GPU_LOCAL_PERMUTATION_NOT_TESTED</code>, <code>pseudoPValues</code>, <code>adjustedPValues</code> and <code>observed</code> are NaN,
<code>significant</code> is 0, <code>output.status.overflow</code> is 1 and the optional <code>output.status.invalidCount</code>
(one uint32, 0 for every other statistic) holds the number of negative analysis rows. Valid
non-negative data is unchanged. Zero is allowed.</li>
<li><code>GPUGlobalPermutationTest</code> handles <code>moran</code>, <code>geary</code>, <code>getisOrdG</code>, <code>bivariateMoran</code> (<code>y</code>
permuted) and <code>leesL</code> (Lee's spatial Pearson, <code>x</code> and <code>y</code> permuted together).
<code>output.summary</code> (<code>GPU_GLOBAL_PERMUTATION_SUMMARY</code>) holds the observed statistic, <code>p_sim</code>, the
exceedance count, the permutation count used, the simulated mean and standard deviation, <code>z_sim</code>,
<code>p_z_sim</code>, the minimum and maximum, and the island and analysis row counts (<code>islandCount</code>,
<code>analysisCount</code>). Optional outputs are the <code>referenceDistribution</code> and its <code>histogram</code> (GeoDa's
permutation plot). <strong>Negative data.</strong> <code>getisOrdG</code> is defined for non-negative data, like the
analytical G of <code>GPUGlobalSpatialStatistics</code> (which also makes its G block NaN when any analysis
value is negative). When any analysis row is negative the observed statistic, every p-value,
<code>zSimulated</code>, the simulated summaries and the <code>referenceDistribution</code> are NaN, the <code>histogram</code> is
all zero, and the optional <code>output.status.invalidCount</code> (one uint32, 0 for every other statistic)
holds the number of negative analysis rows.</li>
</ul>
<p><strong>Sequential stopping (<code>stopAfterExceedances</code>, opt-in).</strong> Besag and Clifford (1991, "Sequential Monte
Carlo p-values", Biometrika 78(2)): set the compile-time integer <code>h</code> in <code>[1, maximumPermutations]</code>
and each test consumes its permutations in index order, stopping at the first <code>L</code> whose exceedance
count (the count the p-value uses for the chosen <code>alternative</code>: <code>g</code>, <code>l</code> or <code>min(g, l)</code>) equals <code>h</code>.
Then <code>p = min(1, m h / L)</code> (<code>m</code> is 2 for <code>'two-sided'</code>, else 1), <code>exceedances</code> is <code>h</code> and <code>L</code> is
reported: <code>GPULocalPermutationTest</code> writes it per row to the optional <code>output.permutationCounts</code>
(uint32; <code>P</code> for a row that never reached <code>h</code>, <code>GPU_LOCAL_PERMUTATION_NOT_TESTED</code> for an untested
row; the output requires <code>stopAfterExceedances</code>), and <code>GPUGlobalPermutationTest</code> writes it to the
<code>permutationCount</code> field of <code>output.summary</code>. A row or test that never reaches <code>h</code> keeps the
standard <code>min(m (M + 1), P + 1) / (P + 1)</code>, so with <code>h > P</code> the results equal the unstopped run bit
for bit; reaching <code>h</code> exactly at <code>L = P</code> counts as a stop (<code>p = m h / P</code>). The Philox stream of
permutation <code>p</code> is unchanged, so the first <code>L</code> permutations are those of the full run and <code>L</code> does not
depend on the dispatch shape. The <code>'folded'</code> alternative is rejected (its statistic needs the mean
of the whole run). Omitting the prop keeps today's behavior and kernels unchanged.</p>
<ul>
<li>Local: each row loops over permutations in its own invocation and <code>break</code>s at <code>L</code>, so a clearly
non-extreme row costs about <code>h / (tail probability)</code> permutations instead of <code>P</code>. <code>significant</code>,
<code>adjustedPValues</code> and the Geary <code>classes</code> use the stopped p-value.</li>
<li>Global: the permutations run in fixed batches of <code>permutationBatchSize</code> when provided, otherwise
<code>getGPUGlobalPermutationBatchSize(rows, maximumPermutations)</code> (about 128 row-block workgroups per
batch so the GPU stays occupied, at most 64 batches). The batch is also the most permutations
computed past <code>L</code>. <code>permutationBatchSize</code> materializes each permuted variable once per row and
permutation before the pair kernel gathers its values. It accepts an integer in
<code>[1, maximumPermutations + 1]</code>, at most <code>2^25</code> elements per materialized variable and at most
4096 unrolled batches; <code>estimatedCommandNodeCount</code> also enforces the shared command node budget.
With or without early stopping, the shuffle sequence and sum order are unchanged, so the results
are bit-identical to the inline shuffle for the same seed. The observed statistic runs separately
before the first sequential batch. After each batch a one-workgroup
node scans that batch's tail flags in permutation order, keeps the running counts in a GPU state
word and zeroes the indirect dispatches of the remaining batches, so the stopping index does not
depend on the batch size. <code>referenceDistribution</code>, <code>histogram</code>, <code>simulatedMean</code>,
<code>simulatedStandardDeviation</code>, <code>zSimulated</code>, <code>minimum</code> and <code>maximum</code> use the first <code>L</code> permutations
only (<code>referenceDistribution</code> is NaN past <code>L</code>).</li>
<li>Typical use is <code>h</code> between 10 and 20 with large <code>P</code> (a LISA map with <code>P = 999</code>, <code>h = 10</code>): rows
with <code>p</code> far above 0.05 stop after tens of permutations while the significant rows run to <code>P</code>.</li>
</ul>
<p><strong>Tail (<code>alternative</code>).</strong> The type is the shared <code>GPUInferenceAlternative</code> and the default is now <code>'two-sided'</code>: with <code>g = #{sim >= observed}</code>
and <code>l = #{sim &#x3C;= observed}</code>, <code>p = min(2 (min(g, l) + 1), P + 1) / (P + 1)</code>. A row (or a global
statistic) whose simulations all tie the observed value has <code>g = l = P</code> and <code>p = 1</code>. <code>'directed'</code>
(<code>p = (min(g, l) + 1) / (P + 1)</code>) is esda's legacy default and is anti-conservative: the tail is
chosen from the data and not doubled, so under the null <code>p</code> is about twice too small. Its tie
handling deviates from esda on purpose: esda uses <code>min(g, P - g)</code>, and <code>P - g</code> counts only
simulations strictly below the observed value, so when simulations tie the observed value (discrete
data, zero-inflated counts, a constant neighborhood) esda reports the smallest attainable
<code>p = 1/(P+1)</code> (for 20 equal simulations: 0.0476 for a tied observed value), while <code>min(g, l)</code> reports</p>
<ol>
<li>The two agree whenever no simulation equals the observed value (pinned against esda 2.10.0).
<code>'folded'</code> matches esda's undirected test exactly; <code>'greater'</code> and <code>'lesser'</code>
match esda's one-sided tests. Do not combine <code>'directed'</code> with <code>multipleTesting</code>.</li>
</ol>
<p><strong>Neighbors, islands and re-standardization.</strong> One rule, shared with the other Stream A statistics:
a neighbor slot is <em>active</em> when its neighbor is a valid row other than the row itself, its weight is
positive, and the neighbor is an included row (mask nonzero, finite value). Zero-weight slots are not
neighbors: they are not counted in <code>k_i</code> and are never drawn. An <em>island</em> is an included row with no
active slot. <code>islands: 'exclude'</code> (default) drops islands from the analysis: NaN outputs, and they
are not counted in <code>n</code>, the mean, the variance or <code>S0</code>, and not neighbors of other rows. There is no
cascade: a row whose only neighbors were islands stays in <code>n</code> with no neighbor, so it is not tested
(global tests add 0 to <code>S0</code> for it). <code>islands: 'keep'</code> reproduces esda: islands stay in <code>n</code> and the
moments, their local statistic is the formula value with a zero lag (<code>observed</code>, for example local
I = 0), but they are never tested (<code>exceedances</code> is <code>GPU_LOCAL_PERMUTATION_NOT_TESTED</code>, p-value NaN).
Parity specs against esda use <code>islands: 'keep'</code>. Island counts are reported in the optional
one-row <code>output.islandCount</code> (local) or <code>summary[GPU_GLOBAL_PERMUTATION_SUMMARY.islandCount]</code> (global).
When <code>weights.information.transform</code> is <code>'row'</code>, each analysis row's active weights are re-standardized to
sum to 1, in the lags and in <code>S0</code>, so a masked or missing neighbor drops out instead of shrinking
the row sum below 1 (numerically identical when nothing is dropped); other transforms are used as
given. If <code>weights.status.overflow</code> or <code>weights.status.candidateOverflow</code> is nonzero the weights are
incomplete: no row is analyzed (statistics NaN), and the local test's existing completeness flag
<code>output.status.overflow</code> is 1 for either loss stage, including a flag on a chunk. Inspect the two input flags separately to identify
the loss stage. The mean is a fixed-order tree sum followed by a
second-pass correction of the deviations, and <code>n</code> is an exact uint32, so large-offset data keep
<code>sum(z) = 0</code> to f32 accuracy.</p>
<p>Local permutation construction rejects an estimated command graph above 32,768 nodes, before
allocating its scratch buffers. Its conservative bound is <code>256 + 3 * C + T + 2 * L</code>, where <code>C</code> is
the number of CSR chunks (one for whole weights), <code>T</code> is the total row-tile count and <code>L</code> is the
number of distinct final/candidate loss views across the outer weights and all chunks. Each loss
view needs both a preparation pass and a public-completeness writer; shared views count once.</p>
<p><strong>Permutation count.</strong> The count is never clamped. A per-frame <code>permutationCount</code> outside
<code>[1, maximumPermutations]</code> writes NaN p-values: the local test leaves every row untested and sets
<code>output.status.overflow</code>; the global test writes NaN for the <code>permutationCount</code> field of the summary,
the p-values and the simulated summaries (the observed statistic is still reported). Validate on the
CPU with <code>getGPUPermutationParameterValues(settings, {maximumPermutations})</code> or the contributor's
<code>getParameterValues(settings)</code>, both of which throw.</p>
<p><strong>Multiple testing (local).</strong> At 999 permutations the smallest two-sided p-value is 0.002, so a large
family may be unable to reject after correction: check <code>getPermutationRejectionFloor</code> (see
<code>GPUMultipleTestingAdjustment</code>). <code>multipleTesting</code> is <code>'none'</code> (default), <code>'bonferroni'</code>, <code>'holm'</code>,
<code>'benjamini-hochberg'</code> or <code>'benjamini-yekutieli'</code>, run through <code>GPUMultipleTestingAdjustment</code> over the
pseudo p-values of all tested rows (never a display mask). <code>output.adjustedPValues</code> receives the
adjusted values, and <code>output.significant</code> is <code>adjustedPValue &#x3C;= level</code>; with <code>'none'</code> it is the exact
cross-multiplied integer comparison <code>p &#x3C;= level</code>.
<code>getInferenceMetadata(settings)</code> on either contributor (the same <code>GPUPermutationSettings</code> that are
packed, validated identically)
returns a <code>GPUInferenceMetadata</code> (alternative, permutation count, seed, multiple-testing method,
significance level and a <code>testedRowsDefinition</code> sentence) for legends.</p>
<p><strong>Bivariate, join-count and Lee statistics.</strong> Definitions below use <code>c = x - mean</code> over the analysis
rows, <code>SS = sum c^2</code>, <code>n</code> analysis rows and the weights as given (re-standardized for <code>'row'</code>
weights). Parity: statistics against esda 2.10 at f32 accuracy, p-values against the twin of the
Philox draws, plus a Monte Carlo bound against esda <code>p_sim</code> (R = 9999).</p>
<ul>
<li><code>localBivariateMoran</code> (esda <code>Moran_Local_BV</code>, needs <code>secondValues</code>): <code>I_i = (n - 1) c^x_i sum_j w_ij c^y_j / sqrt(SS_x SS_y)</code>; the focus keeps <code>x</code>, the drawn neighbors supply <code>y</code>. The optional
<code>quadrants</code> output is esda's <code>q</code> in the PySAL scheme (1 high-high, 2 low-high, 3 low-low, 4
high-low in <code>x_i</code> and the lag of <code>y</code>); it is not gated by significance.</li>
<li><code>localLeeL</code> (esda <code>Spatial_Pearson_Local</code>, needs <code>secondValues</code>): <code>L_i = n (sum_j w_ij c^x_j)(sum_j w_ij c^y_j) / sqrt(SS_x SS_y)</code>. A drawn row supplies both its <code>x</code> and its <code>y</code> (the pair stays
together), the focus value does not enter. For row-standardized weights the mean of the local
values is the global <code>L</code>. esda's own p-value is the <code>'directed'</code> one (anti-conservative); the
default here is <code>'two-sided'</code>.</li>
<li><code>leesL</code> in <code>GPUGlobalPermutationTest</code> (esda <code>Spatial_Pearson</code>, element <code>[0, 1]</code>): <code>L = z'W'Wz / 1'W'W1 = n sum_i (sum_j w_ij c^x_j)(sum_j w_ij c^y_j) / (sqrt(SS_x SS_y) sum_i (sum_j w_ij)^2)</code>,
with the rows of <code>(x, y)</code> permuted together by the swap-or-not shuffle (esda permutes the rows of
<code>Z</code>). Note that esda's global version does not row-standardize <code>W</code>; here <code>'row'</code> weights are
re-standardized over active slots, others are used as given. The smoothing factors esda reports
at <code>[0, 0]</code> and <code>[1, 1]</code> are not exposed. (<code>GPUSpatialPearson</code> is a different statistic: the
weighted correlation across edges.)</li>
<li><code>localJoinCount</code> (esda <code>Join_Counts_Local</code>, binary <code>values</code>): <code>LJC_i = x_i sum_j x_j</code> over the active
neighbors with binary weights (the weight values are ignored), where a value is a join only when
it is exactly 1. Only rows with <code>LJC_i > 0</code> are tested; the others get NaN p-values and
<code>GPU_LOCAL_PERMUTATION_NOT_TESTED</code>, as esda sets their <code>p_sim</code> to NaN. The tail that detects
clustering is <code>'greater'</code>. With ties (discrete statistics) esda's <code>'two-sided'</code> p-value
(percentile based) is not <code>2 min(g, l)</code>; the parity specs therefore use <code>'greater'</code>.</li>
<li><code>localBivariateJoinCount</code> (esda <code>Join_Counts_Local_BV</code>, <code>joinCase</code>): <code>'CLC'</code> (co-location) counts
neighbors with <code>x_j = y_j = 1</code> for a focus with <code>x_i = y_i = 1</code>; <code>'BJC'</code> (no co-location) counts
neighbors with <code>x_j = 0, y_j = 1</code> for a focus with <code>x_i = 1, y_i = 0</code>. A drawn row supplies its <code>x</code>
and <code>y</code> together. <strong>Deviation from esda for <code>'BJC'</code>:</strong> esda observes the count above but
simulates <code>x_i sum_j y_j</code>, which ignores <code>y_i = 0</code> and <code>x_j = 0</code>, so its p-values do not test the
statistic it reports (on the specs' fixture they differ by up to 0.09 at R = 9999). This
implementation simulates the observed statistic; <code>'CLC'</code> is identical to esda.</li>
<li><code>localGeary</code> classification (<code>classes</code> output, <code>GPU_LOCAL_GEARY_CLASS</code>; Anselin 2019, esda
<code>Geary_Local(labels=True)</code>): with <code>m</code> the mean of the local Geary values over the analysis rows, a
significant row (<code>p &#x3C;= level</code>, or <code>adjustedPValue &#x3C;= level</code> with <code>multipleTesting</code>) is 1
(high-high, positive association: <code>c_i &#x3C; m</code> and <code>x_i</code> above the mean), 2 (low-low: <code>c_i &#x3C; m</code>, <code>x_i</code>
below the mean) or 3 (negative association: <code>c_i > m</code>); an insignificant row is 4 and a row that
is not tested (NaN p-value), or exactly at a mean, is 0 (esda: NaN). esda's own docstring calls
codes 1 and 2 "outlier" and "cluster"; the numeric rule is ported, not the words. Small <code>c_i</code> means
similar neighbors, so the tail of clustering is <code>'lesser'</code> (also the tail declared by
<code>GPU_LOCAL_GEARY_PERMUTATION_ADAPTER</code>): with <code>'lesser'</code> class 3 is empty because negative
association never has a small lower-tail p-value, with <code>'greater'</code> classes 1 and 2 are empty, and
with the default <code>'two-sided'</code> both appear. The classification is not available analytically.</li>
</ul>
<p><strong>Global shuffle.</strong> Each permutation relabels the analysis values through a keyed swap-or-not
shuffle (Hoang, Morris and Rogaway 2012, <code>6 ceil(log2 n) + 6</code> rounds, round keys and salts from
Philox 4x32-10 on the seed and the permutation index), so no sort runs per permutation and nothing
of size <code>P x n</code> is stored. The earlier four-round Feistel network is no longer used by the global
test because it is not uniform on small domains: with 400000 draws the marginal of <code>pi(0)</code> and the
joint placement of fixed pairs reject uniformity far below the 0.1 percent level for domains of
10 to 50 elements (joint chi-square p-values below 1e-100; marginal p-values from 4e-4 at <code>n = 50</code>
down to 1e-139 at <code>n = 10</code>), while swap-or-not passes (all 150 chi-square p-values above 0.0023,
none below 1e-3). At 20000 draws on rook lattices and random kNN graphs with <code>n</code> = 30 to 5000
both shuffles reproduce the analytic randomization mean and variance of Moran's I (<code>E = -1/(n-1)</code>,
esda <code>VI_rand</code>) to within about 2.7 standard errors, so the Feistel bias is a small-domain,
joint-placement effect that the variance check alone does not see; it is removed rather than
tolerated. The Knox and Mantel tests and the GWR non-stationarity test use the same swap-or-not
shuffle.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> parameters</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUParameterBuffer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(device, {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  id: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'permutations'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, format: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'uint32'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, length: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">GPU_PERMUTATION_PARAMETER_LENGTH</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  values: </span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">getGPUPermutationParameterValues</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    {seed: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, permutationCount: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">99</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">},</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    {maximumPermutations: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">999</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  )</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> test</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPULocalPermutationTest</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  weights, values, statistic: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'localMoran'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, parameters: parameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  maximumPermutations: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">999</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, multipleTesting: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'benjamini-yekutieli'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, islands: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'exclude'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    exceedances, pseudoPValues, significant, adjustedPValues, islandCount, status: {overflow}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  }</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(test);</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// Preview with 99 permutations, then refine on idle:</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> settings</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {seed: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, permutationCount: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">999</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">};</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">parameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">write</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(test.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">getParameterValues</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(settings));</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> metadata</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> test.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">getInferenceMetadata</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(settings);</span></span></code></pre>
<h4 id="how-it-computes-5">How it computes</h4>
<p>{/* lens:GPULocalPermutationTest <em>/}
{/</em> lens:GPUGlobalPermutationTest */}</p>
<p><code>GPULocalPermutationTest</code>:</p>
<p>::lens{scenario=gpu-spatial-analysis.local-permutation-test rung=datum bind=page datum=420 view=fdr-staircase caption="Inspect tract 420: its observed I, exceedance count, pseudo p-value and 99-draw null distribution."}
::lens{scenario=gpu-spatial-analysis.local-permutation-test rung=formula caption="Follow the pseudo p-value: doubled smaller tail count plus one, over permutations plus one."}
::lens{scenario=gpu-spatial-analysis.local-permutation-test rung=kernel caption="Inspect the per-tract Philox draw loop and tail counting with retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.local-permutation-test rung=graph caption="Trace weights, moments, 99 permutation draws, classification and Benjamini-Hochberg adjustment."}</p>
<p><code>GPUGlobalPermutationTest</code>:</p>
<p>::lens{scenario=gpu-spatial-analysis.global-permutation-test rung=datum bind=page datum=420 caption="Inspect tract 420: its poverty value, one of 791 relabeled across 99 permutations."}
::lens{scenario=gpu-spatial-analysis.global-permutation-test rung=formula caption="Follow the pseudo p-value and the simulated z-score from the 99 pair sums."}
::lens{scenario=gpu-spatial-analysis.global-permutation-test rung=kernel caption="Inspect the pair-sum, shuffle and finalize kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.global-permutation-test rung=graph caption="Trace weights, 100 pair sums, reduction over row blocks and the finalize summary."}</p>
<p>{/* parameters: GPUGlobalPermutationTest generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpuglobalpermutationtest-parameters"><code>GPUGlobalPermutationTest</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUPermutationParameterValues(settings, options?)</code>, writes <code>GPU_PERMUTATION_PARAMETER_LENGTH</code> (4) <code>uint32</code> values.
<ul>
<li>Input port: <code>parameters</code>.</li>
<li><code>options</code> (<code>{maximumPermutations?: number}</code>)</li>
</ul>
</li>
<li>Packer: <code>instance.getParameterValues(settings)</code>, writes the <code>uint32</code> parameter view.</li>
</ul>
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
<p>{/* parameters: GPULocalPermutationTest generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpulocalpermutationtest-parameters"><code>GPULocalPermutationTest</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUPermutationParameterValues(settings, options?)</code>, writes <code>GPU_PERMUTATION_PARAMETER_LENGTH</code> (4) <code>uint32</code> values.
<ul>
<li>Input port: <code>parameters</code>.</li>
<li><code>options</code> (<code>{maximumPermutations?: number}</code>)</li>
</ul>
</li>
<li>Packer: <code>instance.getParameterValues(settings)</code>, writes the <code>uint32</code> parameter view.</li>
</ul>
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
<h3 id="gpuspatialpearson-and-gpugammastatistic"><code>GPUSpatialPearson</code> and <code>GPUGammaStatistic</code></h3>
<p><code>GPUSpatialPearson</code>: Scalable rows are bounded by 4,294,963,199 and actual device bindings. Additive
uint32 count outputs preserve exact populations; float32 statistics remain rounded.</p>
<p><code>GPUGammaStatistic</code>: Scalable rows are bounded by 4,294,963,199 and actual device bindings. Additive
uint32 count outputs preserve exact populations; float32 statistics remain rounded.</p>
<p><code>GPUSpatialPearson</code>: Correlate a tract variable with a second variable at its neighbours, returning one edge-weighted Pearson coefficient.</p>
<p>::lens{scenario=gpu-spatial-analysis.spatial-pearson rung=result caption="Edge-weighted Pearson r = 0.30 between poverty and neighbouring uninsured rates over 791 tracts."}</p>
<p><code>GPUGammaStatistic</code>: Reduce neighbour pairs to one Gamma value: the weighted mean cross-product, squared difference or absolute difference.</p>
<p>::lens{scenario=gpu-spatial-analysis.gamma-statistic rung=result caption="Weighted mean squared poverty difference across the directed queen pairs of 791 tracts is 214 points squared."}</p>
<p>Both statistics sum over directed edges with positive weight to another included row, in three
fixed-order stages. Neither reports an analytic p-value; their permutation adapters and the other
details are described under Additional spatial statistics below.</p>
<h4 id="how-it-computes-6">How it computes</h4>
<p>{/* lens:GPUSpatialPearson <em>/}
{/</em> lens:GPUGammaStatistic */}</p>
<p><code>GPUSpatialPearson</code>:</p>
<p>::lens{scenario=gpu-spatial-analysis.spatial-pearson rung=datum bind=page datum=420 caption="Inspect tract 420: its poverty value is one of 791 foci whose neighbours supply y."}
::lens{scenario=gpu-spatial-analysis.spatial-pearson rung=formula caption="Follow the edge-weighted correlation: covariance over the product of weighted standard deviations."}
::lens{scenario=gpu-spatial-analysis.spatial-pearson rung=kernel caption="Inspect the moments, centered and statistic kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.spatial-pearson rung=graph caption="Trace queen weights, row standardisation and the three-stage edge reduction."}</p>
<p><code>GPUGammaStatistic</code>:</p>
<p>::lens{scenario=gpu-spatial-analysis.gamma-statistic rung=datum bind=page datum=420 caption="Inspect tract 420: its poverty value enters every directed edge to its neighbours."}
::lens{scenario=gpu-spatial-analysis.gamma-statistic rung=formula caption="Follow the Gamma ratio: weighted pair terms over total edge weight."}
::lens{scenario=gpu-spatial-analysis.gamma-statistic rung=kernel caption="Inspect the moments, centered and statistic kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.gamma-statistic rung=graph caption="Trace queen weights, row standardisation and the three-stage edge reduction."}</p>
<h3 id="additional-spatial-statistics-and-bounded-optimization">Additional spatial statistics and bounded optimization</h3>
<p>Rose rows retain the uint32 ceiling 4,294,967,295. There is no separate 2^24 group cap:
<code>groupCount * summaryStride</code> and <code>groupCount * circular.binCount</code> must each fit uint32 and
device bindings. Empirical-Bayes contributors allow 4,294,967,039 rows. Scan accepted count totals
are bounded by 4,294,967,294, reserving <code>0xffffffff</code> for overflow.</p>
<p><code>GPURoseStatistic</code>: Supports radians/degrees and positive-x counterclockwise or positive-y clockwise
conventions. Row, group and flattened output addressing are bounded before construction.</p>
<p><code>GPULocalGeary</code> writes source-aligned <code>sum_j w_ij (z_i - z_j)^2</code> values to <code>output.statistic</code>
(<code>GPULocalGearyOutput</code>, with the optional <code>globalStatistics</code>), with the neighbor, re-standardization and <code>islands</code> semantics of
<code>GPULocalMoran</code> (an island is NaN under <code>'exclude'</code>, the default, and 0 under <code>'keep'</code>, esda's
convention; a constant variable gives NaN). It has no analytic classification: use permutation
inference (<code>GPULocalPermutationTest</code> with <code>statistic: 'localGeary'</code> and <code>classes</code> gives the Anselin
2019 positive / negative association classes, see above). <code>GPU_LOCAL_GEARY_PERMUTATION_ADAPTER</code> declares the <code>'lesser'</code> tail, because small <code>c_i</code>
means similar neighbors (a cluster). <code>GPUSpatialPearson</code> is the weighted Pearson correlation across
directed spatial edges <code>(x_i, y_j)</code>. <code>GPUGammaStatistic</code> reports the weighted mean centered
cross-product, squared difference, or absolute difference over those edges, into the one-row
<code>output.statistic</code> (<code>GPUEdgeAssociationOutput</code>; <code>float32</code> or <code>uint32</code> values). Both skip the focal
row's own slot and zero-weight slots, and reduce the edges in three fixed-order stages (per-workgroup
moments, per-workgroup centered sums around the combined means, one combining workgroup), so they use
the whole device and stay deterministic. The local Geary, Pearson, and two Gamma forms expose
<code>GPUPermutationStatisticAdapter</code> descriptors that name a statistic of the existing permutation
contributors; they do not duplicate permutation loops. Only two of them test the same statistic:
local Geary, and squared-difference Gamma (Geary's numerator over a constant). The cross-product
Gamma adapter runs the <code>'moran'</code> test and the Pearson adapter the <code>'bivariateMoran'</code> test. These
order permutations like the edge statistic only on regular weights: every row or every column
with the same total weight for Gamma (for example row-standardized weights), every column with the
same total for Pearson. On other graphs they are tests of Moran's I and of bivariate Moran's I.
Example: on the star 0-1, 0-2, 0-3 with the edge 3-4 and binary weights, the values
(7, 2, 4, 1, 11) and (11, 2, 1, 7, 4) have cross-product Gamma -10 and -11.0625 but Moran pair sums
-80 and -64. <code>GPUGammaStatistic</code> is not <code>esda.Gamma</code> (a plain sum, uncentered by default, not
divided by the edge weight), and <code>GPUSpatialPearson</code> is not <code>esda.Spatial_Pearson</code> (Lee's L, see
<code>leesL</code>).</p>
<p><code>GPURoseStatistic</code> consumes planar line starts and ends. Its required
<code>GPUCircularStatisticsContract</code> makes directional versus axial interpretation, radians or degrees, angular
origin, and equal-width bin count explicit. It reports count, mean angle, resultant length,
circular variance, Rayleigh z, and the weighted Rose bins for every group, in
<code>output: {summary, bins}</code>. The descriptive mean, resultant length, variance and bins use the
positive finite <code>weights</code>; invalid lines or weights are excluded. Rayleigh z uses <strong>each valid
segment once</strong>, with unweighted resultant $\\bar R_0$ and valid segment count $n$:</p>
<p>$$z = n\\bar R_0^2 = \\frac{(\\sum_i\\cos\\phi_i)^2 + (\\sum_i\\sin\\phi_i)^2}{n}.$$</p>
<p>Here $\\phi_i=\\theta_i$ for directional data and $2\\theta_i$ for axial data. This corrects the
previous use of total descriptive weight as sample size: metres of street are not independent
observations, and changing their units must not change z. Weighted descriptive <code>resultantLength</code>
is distinct from $\\bar R_0$. No p-value is returned. Under independent uniform directions the
usual <a href="https://github.com/circstat/circstat-matlab/blob/master/circ_rtest.m">Rayleigh approximation</a>
is $p\\approx e^{-z}[1+(2z-z^2)/(4n)-(24z-132z^2+76z^3-9z^4)/(288n^2)]$;
spatially dependent street segments do not establish that null assumption. This planar rose is
separate from <a href="https://pysal.org/giddy/generated/giddy.directional.Rose.html">giddy Rose</a>, which
forms directional LISA vectors and uses permutations of sector counts for inference.</p>
<p><code>GPUAZPRegions</code> is a greedy AZP with batched moves: a bounded, deterministic local search from
caller-supplied contiguous labels with no tabu list, annealing or reactive escape, so it stops at the
first local optimum and the result depends on <code>initialLabels</code> (for several starts, run several
instances from different initial partitions and keep the smallest <code>objective</code>). Each round every row
finds its best improving move (the largest reduction of the within-region sum of squares from leaving
its region for an adjacent one) and the round applies an independent set of them at once: every region
takes part in at most one selected move, chosen by gain and then by a stable seed-keyed row order
(two atomic-max claim passes over the regions), so the selected gains add up exactly and the objective
falls strictly every round. The best move of a round is always selected, so a round without a move is
the local optimum. Contiguity is kept by a <strong>local articulation test</strong> instead of a search of the whole
source region: a row may leave region A only when its A-neighbors stay connected to each other through
rows of A within two steps of it (at most <code>GPU_AZP_LOCAL_NEIGHBORHOOD_CAPACITY</code> rows, scanning neighbor
lists of at most <code>GPU_AZP_MAXIMUM_SCAN_DEGREE</code> entries; a larger neighborhood or a hub refuses the
move). The test is sufficient, never necessary, so a region never becomes disconnected, but a safe
move whose reconnecting path is longer is refused: the result is a local optimum with respect to the
moves that pass the test, not necessarily to every contiguity-preserving move (on random lattice
partitions the test refuses about 1% of the safe moves). A row only moves to the region of one of its
neighbors, which keeps the destination connected. The initial labels are checked on the GPU as well: a
region that is not connected adds its extra components to <code>invalidCount</code> and the search does not run.
The operator publishes <code>output: {labels, objective, status}</code>: the final within-region sum of squares
(the same quantity as <code>withinSumOfSquares</code> of <code>GPURegionPartitionEvaluation</code> on the same table,
recomputed from the returned labels) plus packed iteration, convergence, iteration-limit, and
invalid-count status. <code>iterationCount</code> counts the rounds that applied at least one move, not single
moves; <code>maximumIterations</code> is a hard compile-time bound on rounds (a round applies many moves, so far
fewer are needed than the single moves of earlier versions). The seed is per frame: the one-word
<code>parameters</code> view written with <code>getGPUAZPRegionsParameterValues({seed})</code> fixes tie order, so a new seed
is a buffer write. Optional <code>extensiveValues</code> and <code>extensiveThreshold</code> forbid a move that would leave
the source region below the threshold (the max-p constraint). A multi-start AZP is deferred: one
instance already costs <code>maximumIterations</code> rounds of row-parallel work, so parallel starts would
multiply the dispatch count rather than fill idle lanes. There is no row cap: private memory is
<code>O(GPU_AZP_LOCAL_NEIGHBORHOOD_CAPACITY)</code> and the work per row and round is bounded by the row's
neighbor lists, so only the <code>rows x columnCount</code> attribute table and the region record table must each
fit one 128 MiB binding (<code>getGPUAZPMaximumTableWordCount(device)</code> words, 33,554,432 on a default
device). After the local optimum the one-thread round-finish kernel zeroes a GPU-written indirect
dispatch, so the remaining rounds launch only that kernel.</p>
<p>The AZP CPU twin obeys the same explicit <code>maximumIterations</code> bound. Its pure function defaults to
256 rounds when the budget is omitted. A successful batch on the last permitted round reports
<code>converged = 0</code> and <code>iterationLimitReached = 1</code>; a no-move round must execute to prove convergence.</p>
<p><code>GPUAZPRegions</code>, <code>GPUMaxPRegions</code> and <code>GPUWardRegions</code> share two input rules. <strong>Attributes are
standardized by default</strong> (<code>standardize</code>, default true): each column becomes a z-score over its finite
rows with the population standard deviation (divisor <code>n</code>, the <code>sklearn</code> <code>StandardScaler</code> convention),
a constant column becomes zero, and <code>objective</code> is in z-score units. Pass <code>standardize: false</code> for the
raw-unit behavior of earlier versions (a default change). <strong>Adjacency is undirected</strong>: row <code>i</code> is
adjacent to <code>j</code> when either row lists the other, so directed (for example k-nearest-neighbor) weights
are symmetrized inside the operator with <code>GPUSpatialWeightsTranspose</code>; weights whose
<code>information.symmetric</code> is <code>true</code> skip that pass. All three take a per-frame <code>uint32</code> <code>parameters</code>
view holding <code>[seed]</code> (<code>getGPUAZPRegionsParameterValues</code>, <code>getGPUWardRegionsParameterValues</code>,
<code>getGPUMaxPRegionsParameterValues</code>) and write <code>output: {labels, objective, status}</code>.</p>
<p><code>GPUWardRegions</code> begins with one region per row and repeatedly merges the adjacent pair with the
smallest Ward increase in within-region squared error. <code>targetRegionCount</code> and <code>maximumIterations</code>
bound the search; disconnected weights can stop it before the target, which is reported through
<code>output.status</code> (<code>GPU_WARD_STATUS</code>).</p>
<p>The Ward CPU twin obeys the same explicit round budget, including zero. With zero rounds and valid
input it returns the singleton partition and reports the iteration limit unless the target is already
met. Invalid input sets <code>invalidCount</code> and leaves both <code>converged</code> and <code>iterationLimitReached</code> zero,
including at a zero-round budget, for both merge schedules and operation forms. The pure CPU function
defaults to <code>rowCount - targetRegionCount</code> rounds when the budget is omitted.</p>
<p>Ward captures declared options, output maps and adjacency status fields by property read at
construction, including inherited/non-enumerable getters. Later caller-map changes cannot remove
a declared loss flag; buffer contents remain per-frame inputs.</p>
<p>Ward suppresses partition outputs if either <code>weights.status.overflow</code> or
<code>weights.status.candidateOverflow</code> is nonzero: every label is <code>0xffffffff</code>, <code>objective</code> is NaN,
and iteration/convergence fields are zero. The flags retain their separate final-output and
candidate-loss meanings; neither contributes to <code>invalidCount</code>. Both operation forms accept the
same optional <code>weights.status</code> graph values; the pure CPU function accepts corresponding uint32
arrays in <code>input.status</code>. Clearing both flags on a later submission permits merging again.</p>
<p><code>GPUWardRegions</code> has a compile-time <code>mergeSchedule</code>. The default <code>'sequential'</code> applies one merge per
unrolled iteration (always the globally cheapest adjacent pair), so reaching <code>k</code> regions from <code>n</code> rows
needs <code>n - k</code> iterations. <code>'mutual-nearest'</code> makes every iteration a round: each region finds its cheapest
adjacent region (Ward increase, ties by the same seed-keyed pair order), and every pair that chose each
other merges at once. Many graphs need far fewer rounds than sequential merges, but a star graph
needs <code>n - k</code> rounds because only the center and its chosen leaf are reciprocal. <code>maximumIterations</code>
is an upper bound on rounds, not a logarithmic convergence guarantee. When a round would pass
<code>targetRegionCount</code>, exact radix selection keeps only the cheapest pairs in the total order. Above
4,096 rows, counting and byte-wise radix histograms run across multiple workgroups. <code>status[iterationCount]</code>
still counts merges, <code>objective</code> is still the sum of the merges' Ward increases (the within-region sum of
squares of <code>labels</code>), and a <code>status</code> view of five words receives the number of rounds in word
<code>GPU_WARD_ROUND_COUNT_STATUS_WORD</code>. It is opt-in because the result can differ from <code>'sequential'</code>. Without
contiguity constraints (complete adjacency) Ward's linkage is reducible, so both schedules build the same
dendrogram and agree whenever the target is a natural cut (every merge below it is cheaper than every
merge above it, as for well separated clusters), which the <code>scipy.cluster.hierarchy.ward</code> fixture checks.
At other targets a round can merge a costly reciprocal pair before a cheaper pair that exists only after
other merges of the same round, so the partitions differ; with contiguity constraints Ward can also have
inversions (a merged region can be cheaper to join than its parts), which makes them differ further.
Either result is a valid contiguous partition with an exact objective.</p>
<p>Both schedules contract a GPU-resident graph of unique undirected region pairs after each merge or
round. Internal edges and duplicate boundary pairs are discarded, edge dispatches shrink with the
active edge count, and one Ward cost is shared by both endpoints. Costs are recomputed only when an
endpoint's region moments changed. Merges update parent links; dense row labels are published once
at the end with race-free pointer doubling. These changes retain each schedule's seeded tie order
and moment arithmetic. The sequential schedule still has <code>n - k</code> dependent merge decisions, and
contraction adds scratch and dispatch overhead, so small inputs do not imply a speedup.</p>
<p><code>GPUMaxPRegions</code> is a parallel multi-seed Max-P: <code>startCount</code> (compile time, default 16, at most 64)
independent constructions share every dispatch. Each start plants many region seeds at once and grows all
regions together in gated rounds, so there is no row cap. A <em>seed wave</em> plants a seed at every
unassigned row that is a local minimum of its Philox 4x32-10 key of <code>(row, start; seed)</code> (seed read from
<code>parameters</code>) among the unassigned rows next to it, is not next to a growing region and is below the
wave's key cut; the first wave cuts to about <code>0.4 / s</code> of the rows, with <code>s = ceil(threshold / mean(extensiveValues))</code> the expected region size, later waves cut 1.3 times as many, and a wave starts
when no region is growing or <code>ceil(s / 2)</code> rounds after the previous one. Each growing region then claims
one frontier row per round: rows offer themselves with a key made of their number of unassigned
neighbors (fewest first, which keeps regions compact) and their squared distance to the region mean, a
region keeps its smallest key and a row offered by several regions joins the smallest. Region sums are
updated by their single owner, so no float atomics are involved. A region that reaches the threshold
stops, and a region with no frontier left under the threshold is marked failed. The rows of failed
regions and rows that no region reached then join the most similar adjacent completed region in
label-propagation rounds; a failed region with no completed neighbor stays and is reported in the
status. The returned start has the fewest under-threshold regions, then the most regions, then the
smallest within-region sum of squares of the grown regions (before enclave rows join), then the lowest
start index; the status reports the region count, any infeasible remainder, the <code>selectedStart</code> and the
<code>improvementIterationCount</code> (successful rounds applying at least one move, rather than reassigned
rows; formerly named <code>improvementMoveCount</code>), and <code>objective</code> is the exact within-region sum of squares of the returned
labels. <code>maximumIterations</code> bounds refinement rounds on both CPU and GPU; a round can move several
independent rows. With <code>maximumIterations > 0</code> (default 0) the selected construction is refined by greedy AZP
moves that keep every region at or above the threshold, which lowers <code>objective</code> at the same region
count (the AZP phase reads the same <code>parameters</code> view); spopt's <code>MaxPHeuristic</code> anneals instead.</p>
<p>Both CPU and GPU limit growth to <code>min(maximumGrowthSteps, 2 * rowCount + 2)</code> rounds and enclave
assignment to <code>min(maximumGrowthSteps, rowCount + 1)</code> rounds. At the limit they publish the current
partition: unreached rows become singleton regions, unfinished regions contribute to
<code>infeasibleRegionCount</code>, and <code>feasible</code> is zero while any such region remains.</p>
<p>The rounds are unrolled to <code>maximumGrowthSteps</code> (compile time, default 256, at most 4096) and gated on
the GPU, so rounds after the construction has settled launch nothing and every dispatch is
<code>O(startCount * (rowCount + neighborCount))</code>. A region claims one row per round, so a region needing more
rows than <code>maximumGrowthSteps</code> cannot complete and is reported under the threshold; seed waves are spaced
half an expected region apart, so raise <code>maximumGrowthSteps</code> (nine command nodes per round) when the
threshold needs regions of more than about <code>maximumGrowthSteps / 4</code> rows. Scratch memory is
<code>startCount * rows * (columnCount + 3) * 4</code> bytes for the region table and must fit one storage binding
(<code>startCount * rows * 16</code> bytes for the claim records), which is the only size limit. Against <code>spopt</code>'s
<code>MaxPHeuristic</code> (test/oracles/statistics-max-p.py, spopt 0.8.0, lattices and Delaunay graphs of 144 to
900 rows) the construction reaches 0.975 to 1.04 of spopt's best region count with every region at the
threshold and contiguous; its objective before the optional AZP improvement is up to 1.55 times spopt's
annealed objective.</p>
<p><code>GPULocationAllocation</code> consumes a row-major demand-by-facility cost matrix and implements bounded
greedy p-median, maximum-coverage, and set-covering selection. Pair it with
<code>GPUNetworkCostMatrix</code> for network costs, or provide any dense cost matrix (<code>costs</code>) or a sparse one
(<code>sparseCosts</code>, below). Outputs include selected facility flags, nearest assignments, the objective, uncovered demand count, completion, and invalid
input count. The contributor keeps selection on the GPU, but it does not claim the integer-programming
optimum or application-specific facility constraints of a packaged solver.</p>
<p>Set covering scores a facility by the number of uncovered demand rows it would cover, not by their
weight: as in <code>spopt</code>'s <code>LSCP</code>, every reachable row must be covered, including a row whose
<code>demandWeights</code> entry is 0. <code>demandWeights</code> enters the p-median and maximum-coverage objectives only.</p>
<p>A <code>+Infinity</code> cost marks an unreachable demand-facility pair. Unreachable demand is never summed
into the p-median objective: the objective is the weighted cost of the reachable rows, the
unreachable rows are counted in <code>uncoveredDemandCount</code> (not <code>invalidCount</code>, which marks invalid input
and stops the run), their assignment is <code>0xffffffff</code> and their <code>assignedCosts</code> entry is <code>+Infinity</code>.
Greedy and swap steps compare p-median candidates lexicographically, first by the demand weight left
unreachable and then by the reachable cost, so one unreachable row cannot swamp the float32 cost
differences of the others.</p>
<p>After the greedy construction, p-median and maximum coverage run a Teitz-Bart vertex-substitution
(swap) phase, bounded by <code>maximumSwapIterations</code> (compile-time, integer 0 to 1000, default 100; 0 is
pure greedy; set covering has no swap phase). Each iteration scores every (selected facility, unselected
candidate) swap in parallel, applies the single best strictly improving swap (ties go to the lowest
selected facility, then the lowest candidate) and stops when none improves. A swap must gain more than
1e-6 of the objective scale, which keeps f32 summation noise from cycling. It needs
<code>maximumFacilityCount * facilityCount &#x3C;= 2^24</code> (scores are kept per pair). The <code>status</code> output now has seven words: after the four earlier ones,
<code>swapIterationCount</code> (swaps applied), <code>converged</code> (1 when no improving swap remains, so the result is a
swap-local optimum) and <code>iterationLimitReached</code> (1 when the budget was used up, so local optimality is
not verified); all three are 0 when the phase is off. Behavior change: swaps are on by default, so
p-median and coverage results can differ from (and are never worse than) the old greedy result, and
<code>output.status</code> must hold seven words. Greedy maximum coverage alone has the <code>(1 - 1/e)</code> guarantee
relative to the optimum (Nemhauser, Wolsey and Fisher), which the swap phase preserves because it never
lowers the covered weight; greedy p-median has no constant-factor guarantee. Either way the result is a
local optimum, not the integer-programming optimum. It is checked against a brute-force optimum on small
instances, and p-median against a randomized weighted fixture with <code>+Infinity</code> entries whose exact optimum
comes from the integer program solved by <code>scipy.optimize.milp</code> (generator and package versions in
<code>test/oracles/p-median-oracle.py</code>).</p>
<p>Swap scoring uses the Whitaker fast-interchange gain decomposition in the form of Resende and Werneck
("A fast swap-based local search procedure for location problems", Annals of Operations Research,
2007). With <code>d1</code> and <code>d2</code> the cost of a demand row to its nearest and second-nearest selected facility,
<code>w</code> its weight and <code>c(i, j)</code> the cost of candidate <code>i</code> to row <code>j</code>, swapping out <code>r</code> for <code>i</code> changes the
p-median cost by <code>gain(i) - loss(r) + extra(i, r)</code>, where <code>gain(i)</code> sums <code>w * max(0, d1 - c(i, j))</code> over
all rows, <code>loss(r)</code> sums <code>w * (d2 - d1)</code> over the rows served by <code>r</code>, and <code>extra(i, r)</code> sums
<code>w * (d2 - max(c(i, j), d1))</code> over the rows served by <code>r</code> with <code>c(i, j) &#x3C; d2</code>. Unreachable rows keep the
lexicographic rule: a separate reachable-weight term (which <code>d2 = +Infinity</code> and a newly reached row
feed) takes precedence over the cost term. Maximum coverage has the same shape with <code>gain(i)</code> the weight
of the uncovered rows <code>i</code> covers, <code>loss(r)</code> the weight covered only by <code>r</code>, and <code>extra(i, r)</code> the part of
it that <code>i</code> also covers. One thread scores all <code>maximumFacilityCount</code> swaps of one candidate in a single
ascending pass over that candidate's costs, writing only its own pair entries (no atomics), and a
fixed workgroup tree sums the objective scale (sparse input: <code>loss(r)</code> too), so an iteration costs <code>O(facilityCount * demandCount)</code> work (dense) instead of <code>O(maximumFacilityCount * facilityCount * demandCount)</code> and the
sums do not depend on scheduling. The earlier per-pair kernels summed per-row differences; the
decomposition sums the same terms grouped differently, so the applied swap is the same except where
float32 summation order changes a near-tie, which the 1e-6 threshold guards (the small integer
fixtures in the specs sum exactly, and the specs require identical swaps). The dense kernel visits every row of a
candidate, so each row served by <code>r</code> adds its own net term <code>w * (min(max(c, d1), d2) - d1)</code> (and the
reachable weight it loses if neither <code>d2</code> nor <code>c</code> reaches it) instead of a separate <code>loss(r)</code> and
<code>extra(i, r)</code>; no large <code>d2</code> terms cancel, which a spec with <code>d2</code> about <code>1e7 * d1</code> checks in float32.
The sparse kernel cannot visit unlisted pairs, so it sums <code>loss(r)</code> per selected facility (one
workgroup each) and adds <code>extra(i, r)</code>. For a fully reachable solution the comparison cap described
below bounds the individual terms; float32 summation order can still change near-ties.</p>
<p>Every finite nonnegative cost is reachable, including values at or above <code>1e30</code>; only <code>+Infinity</code>
marks an unreachable pair. This rule applies to greedy scores, swaps and final assignments in both
representations. Assignment ties choose the lowest facility row. A coverage threshold of <code>+Infinity</code>
covers all reachable pairs and never covers unreachable pairs.</p>
<p>P-median greedy scores and swap cost totals, gains and tolerance comparisons use a common binary
scale when the complete weighted demand sum risks overflow. The cost exponent uses costs and
weights from the same demand row and excludes zero products, so unrelated maxima on
noncontributing rows do not erase small improvements. For p-median swaps, when every positive-weight
demand row is already reachable, the exponent instead bounds the weighted nearest costs for that
iteration, with headroom for the sparse sum of capped terms. Weighted comparison costs are capped
at twice that scaled objective before differences are formed: a larger cost cannot improve the
current solution. This prevents unused, enormous
fallback costs from erasing a small gain through scaling or sparse loss/extra cancellation. A swap
that can reach new positive-weight demand retains the full input product scale. These bounds affect
comparisons only; assigned costs and the reported objective remain in caller units.
Swap reach weights and maximum-coverage swap gains, losses, totals and tolerance likewise share a
weight exponent. The tolerance remains relative to the total demand weight, including uncovered
rows, even when its raw sum overflows.
This allows a representable improvement even when the previous objective overflows float32.
Assigned costs and the final objective remain in caller units;
score arithmetic retains the usual float32 rounding.
The CPU twin honors the same explicit or default swap budget. It reports a limit after the last
permitted applied swap, and convergence only when a permitted iteration finds no improvement.</p>
<p><code>sparseCosts</code> replaces the dense matrix with CSR rows: <code>offsets</code> (<code>demandCount + 1</code> non-decreasing
<code>uint32</code>, from 0 to the entry count), <code>facilities</code> (strictly ascending and below <code>facilityCount</code> within a
row) and <code>costs</code> (<code>float32</code>), with at least one entry; a pair that is not listed is unreachable
(<code>+Infinity</code>), and a coverage distance that is <code>+Infinity</code> does not cover unlisted pairs. Exactly one of
<code>costs</code> and <code>sparseCosts</code> is given. Every operation (greedy p-median, maximum coverage, set covering and
the swap phase) supports it and gives the result of the equivalent dense matrix. Violations of the CSR
structure, NaN or negative costs count in <code>invalidCount</code>. The GPU builds the facility-major transpose
once (a stable radix sort of the entries by facility and a binary search of the segment offsets), which
the facility-side loops read, so greedy scores and swap candidates scan only the listed pairs and the
work scales with the entry count; hub rows are validated with the shared load-balanced loop, and the
demand loops scan the shorter of the row and the selected-facility list. The dense <code>demand rows x facilityCount</code> limit no longer applies: the entry count is at most
<code>getGPULocationAllocationMaximumSparseEntryCount(device)</code> (16,777,216 on a default device, the 8-byte
transpose entry in one 128 MiB binding), the demand row count <code>offsets.length - 1</code> at most
<code>getGPULocationAllocationMaximumDemandCount(device)</code>, and the facility count and swap pair limits are
unchanged.</p>
<p>Once no swap improves, the one-thread apply kernel zeroes a GPU-written indirect dispatch, so a
converged run launches only that kernel per remaining iteration instead of six; maximum coverage and
set covering do the same for the greedy steps after one that chose nothing.</p>
<p><code>alternative</code> (compile-time; a change recompiles) selects the tail, with esda's definitions. With <code>R</code>
permutations, <code>g</code> is the number of simulated values <code>>= observed</code> and <code>l</code> the number <code>&#x3C;= observed</code>:</p>
<table>
<thead>
<tr>
<th><code>alternative</code></th>
<th>pseudo p-value</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>'directed'</code> (default)</td>
<td><code>(min(g, R - g) + 1) / (R + 1)</code>, the previous behavior</td>
</tr>
<tr>
<td><code>'greater'</code></td>
<td><code>(g + 1) / (R + 1)</code></td>
</tr>
<tr>
<td><code>'lesser'</code></td>
<td><code>(l + 1) / (R + 1)</code></td>
</tr>
<tr>
<td><code>'two-sided'</code></td>
<td><code>min(2 (min(g, l) + 1) / (R + 1), 1)</code></td>
</tr>
<tr>
<td><code>'folded'</code></td>
<td><code>(f + 1) / (R + 1)</code>, <code>f</code> = number of simulated values with <code>abs(sim - mean) >= abs(observed - mean)</code></td>
</tr>
</tbody>
</table>
<p><code>exceedances</code> holds the numerator count of the chosen tail, and <code>significant</code> and the Benjamini-Hochberg
path use the same p-value. <code>'two-sided'</code> doubles the p-value against <code>'directed'</code> and cannot go below
<code>2 / (R + 1)</code>, so raise <code>permutations</code> when testing a small level.</p>
<p><code>'folded'</code> matches esda 2.10 exactly: <code>mean</code> is the mean of the <code>R</code> simulated statistics (the observed
value is excluded, as esda computes it from the reference distribution), ties count, and the test is
symmetric about that mean. The kernels run two sweeps over the same deterministic stream (a sum, then
the count); the other tails keep one. <code>'two-sided'</code> is not esda's definition: esda 2.10 places the
observed value at a percentile of the simulated statistics and counts simulations beyond the
interpolated percentiles, which would need every simulated value of a row stored and sorted. The two
can differ by a few counts (a 20-permutation fixture gives 3/21 in esda and 2/21 here). Use <code>'folded'</code>
when an undirected test must agree with esda.</p>
<p>On skewed data <code>'folded'</code> can flag far fewer rows than <code>'two-sided'</code>. With local Moran's I over a
right-skewed variable (hot spots on a low background), a low-value row's conditional distribution is
left-skewed, so its observed value sits in the short tail and the long tail holds simulations just as
far from the mean. Low-low rows that <code>'greater'</code> flags then mostly fail <code>'folded'</code>, while high-high rows
pass. esda 2.10 behaves the same: on a 1,500-point hot-spot field it flags 211 rows with <code>'folded'</code> and
551 with <code>'two-sided'</code>.</p>
<p>Random numbers come from Philox 4x32-10 counter streams, keyed by the 64-bit seed with the counter
<code>(block, row, permutation, tag)</code>. Results are therefore a pure function of the seed and inputs,
independent of the dispatch shape. The observed and simulated statistics run through the same f32
code, positive constant factors are dropped from the comparisons, and exceedances are integers. The
significance and BH tests compare correctly rounded f32 products, never quotients. Global pair sums
use fixed-shape workgroup trees. Every output is bitwise reproducible for a seed.</p>
<p>Maximum: all limits derive from the device's <code>maxStorageBufferBindingSize</code>; the values below are for a default device (128 MiB). <code>GPUAZPRegions</code> and <code>GPUWardRegions</code> take at most <code>getGPURegionalizationMaximumSlotCount(device)</code> (<code>2^24</code>) <code>weights.neighbors</code> slots (AZP symmetrized adjacency and Ward contracted edge records hold 8 bytes per slot in one binding; Ward also reserves a hash table at exactly twice the input slot capacity) and <code>GPUMaxPRegions</code> at most <code>getGPUMaxPMaximumSlotCount(device)</code> (<code>2^23</code>, its AZP refinement symmetrizes again); <code>GPUAZPRegions</code> tables are at most <code>getGPUAZPMaximumTableWordCount(device)</code> (<code>2^25</code>) words and Ward's <code>rows x (columnCount + 1)</code> region table is at most <code>getGPUWardMaximumRegionTableWordCount(device)</code> (<code>2^25</code>). <code>GPULocationAllocation</code> takes at most <code>getGPULocationAllocationMaximumFacilityCount(device)</code> (<code>2^24</code>) facilities, <code>getGPULocationAllocationMaximumDemandCount(device)</code> (<code>11184810</code>) demand rows and <code>getGPULocationAllocationMaximumCostCount(device)</code> (<code>2^25</code>) cost entries.</p>
<p><code>GPUAZPRegions</code>: Improve contiguous starting regions by moving border tracts, returning region labels, the within-region sum of squares and status words.</p>
<p>::lens{scenario=gpu-spatial-analysis.azp-regions rung=result caption="Eight community areas regrouped by poverty; a 64-round budget allows longer local searches."}</p>
<p>The former 16-round budget stopped while improving moves remained; 32 rounds were also insufficient.
The example now allows 64 rounds, including a no-move round to establish convergence. A round
can move only one tract per participating region; the budget is not a count of full boundary sweeps.
Check <code>status[GPU_AZP_STATUS.converged]</code> before interpreting the result as a local optimum under the bounded
connectivity test.</p>
<p><code>GPUMaxPRegions</code>: Grow as many contiguous regions as possible that each reach a minimum extensive total, returning labels, the within-region sum of squares and status words.</p>
<p>::lens{scenario=gpu-spatial-analysis.max-p-regions rung=result caption="11 regions of at least 40,000 residents grown over 136 tracts; compare poverty with region labels."}</p>
<p><code>GPUWardRegions</code>: Merge adjacent tracts by smallest Ward cost into a target region count, returning labels and the within-region sum of squares.</p>
<p>::lens{scenario=gpu-spatial-analysis.ward-regions rung=result caption="Ward merges 136 tracts into five contiguous regions in 131 merges; the largest holds 91 tracts."}</p>
<p><code>GPULocationAllocation</code>: Choose p facilities from candidate tracts to minimize population-weighted cost, returning selected flags, nearest-facility assignments and the objective.</p>
<p>::lens{scenario=gpu-spatial-analysis.location-allocation rung=result caption="Five facilities chosen from 136 tracts, each tract coloured by its nearest; weighted mean distance 1.24 km."}</p>
<p><code>GPURoseStatistic</code>: Summarize planar line directions as a weighted rose, returning count, mean angle, resultant length, circular variance, Rayleigh z and sector bins.</p>
<p>::lens{scenario=gpu-spatial-analysis.rose-statistic rung=result caption="Axial rose of 894 street segments in 18 sectors; east carries 42.5 km, north 41.3 km."}</p>
<p>The regionalization and location-allocation cards use community areas 1 to 8 (136 contiguous north-side
tracts) of the Chicago tract data, so the unrolled algorithms stay small. Their label values are
nominal identifiers, not scores. The location-allocation cost matrix is straight-line kilometres between
tract vertex-mean centres, prepared on the CPU as the input; the figure does not draw it.</p>
<h4 id="how-it-computes-7">How it computes</h4>
<p>{/* lens:GPUAZPRegions <em>/}
{/</em> lens:GPULocationAllocation <em>/}
{/</em> lens:GPUMaxPRegions */}</p>
<p><code>GPUAZPRegions</code>:</p>
<p>::lens{scenario=gpu-spatial-analysis.azp-regions rung=datum bind=page datum=40 caption="Inspect tract 40: its poverty value, starting community area and AZP region."}
::lens{scenario=gpu-spatial-analysis.azp-regions rung=formula caption="Follow the AZP objective and the gain of moving one tract between regions."}
::lens{scenario=gpu-spatial-analysis.azp-regions rung=kernel caption="Inspect the standardize, candidate-move, claim and apply kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.azp-regions rung=graph caption="Trace queen weights, standardization, up to 64 rounds of batched moves and the final objective."}</p>
<p><code>GPUMaxPRegions</code>:</p>
<p>::lens{scenario=gpu-spatial-analysis.max-p-regions rung=datum bind=page datum=40 caption="Inspect tract 40: its poverty, population and the max-p region that claimed it."}
::lens{scenario=gpu-spatial-analysis.max-p-regions rung=formula caption="Follow the max-p rule: regions meet the population threshold, then the sum of squares decides."}
::lens{scenario=gpu-spatial-analysis.max-p-regions rung=kernel caption="Inspect the growth, enclave and selection kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.max-p-regions rung=graph caption="Trace queen weights, eight parallel starts, growth rounds and the selection of the best start."}</p>
<p><code>GPULocationAllocation</code>:</p>
<p>::lens{scenario=gpu-spatial-analysis.location-allocation rung=datum bind=page datum=40 caption="Inspect tract 40: its population weight, assigned facility and straight-line distance to it."}
::lens{scenario=gpu-spatial-analysis.location-allocation rung=formula caption="Follow the p-median objective: weight times distance to the nearest selected facility."}
::lens{scenario=gpu-spatial-analysis.location-allocation rung=kernel caption="Inspect the greedy choice, swap scoring and finalize kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.location-allocation rung=graph caption="Trace the greedy construction, up to eight swap iterations and the final objective reduction."}</p>
<p>{/* lens:GPURoseStatistic <em>/}
{/</em> lens:GPUWardRegions */}</p>
<p><code>GPUWardRegions</code>:</p>
<p>::lens{scenario=gpu-spatial-analysis.ward-regions rung=datum bind=page datum=40 caption="Inspect tract 40: its poverty value and the Ward region it merged into."}
::lens{scenario=gpu-spatial-analysis.ward-regions rung=formula caption="Follow the Ward merge cost: size-weighted squared distance between adjacent region means."}
::lens{scenario=gpu-spatial-analysis.ward-regions rung=kernel caption="Inspect the candidate, merge and relabel kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.ward-regions rung=graph caption="Trace queen weights, standardization and up to 32 mutual-nearest merge rounds (23 ran) toward five regions."}</p>
<p><code>GPURoseStatistic</code>:</p>
<p>::lens{scenario=gpu-spatial-analysis.rose-statistic rung=datum bind=page datum=100 caption="Inspect segment 100: its endpoints and length weight, one of 894 rows entering the rose."}
::lens{scenario=gpu-spatial-analysis.rose-statistic rung=formula caption="Follow the circular moments: weighted moments describe street length; unweighted directions give standard Rayleigh z."}
::lens{scenario=gpu-spatial-analysis.rose-statistic rung=kernel caption="Inspect the tiled bin and moment reduction kernel with its retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.rose-statistic rung=graph caption="Trace the single-pass reduction from 894 segments into five summary words and 18 sector bins."}</p>
<p>{/* parameters: GPUAZPRegions generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpuazpregions-parameters"><code>GPUAZPRegions</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUAZPRegionsParameterValues(settings?)</code>, writes <code>GPU_AZP_REGIONS_PARAMETER_LENGTH</code> (1) <code>uint32</code> values.</li>
</ul>
<p>Settings of <code>getGPUAZPRegionsParameterValues</code>:</p>
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
<td>Reproducible tie-order seed, an integer in <code>[0, 2^32 - 1]</code>. Defaults to 0. Record the seed used to create <code>initialLabels</code> next to it.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<p>{/* parameters: GPUMaxPRegions generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpumaxpregions-parameters"><code>GPUMaxPRegions</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUMaxPRegionsParameterValues(settings?)</code>, writes <code>GPU_MAX_P_REGIONS_PARAMETER_LENGTH</code> (1) <code>uint32</code> values.</li>
</ul>
<p>Settings of <code>getGPUMaxPRegionsParameterValues</code>:</p>
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
<td>Reproducible seed of the random seed-row order of every start (and of the improvement phase's tie order), an integer in <code>[0, 2^32 - 1]</code>. Defaults to 0.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<p>{/* parameters: GPUWardRegions generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpuwardregions-parameters"><code>GPUWardRegions</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUWardRegionsParameterValues(settings?)</code>, writes <code>GPU_WARD_REGIONS_PARAMETER_LENGTH</code> (1) <code>uint32</code> values.</li>
</ul>
<p>Settings of <code>getGPUWardRegionsParameterValues</code>:</p>
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
<td>Reproducible tie-order seed, an integer in <code>[0, 2^32 - 1]</code>. Defaults to 0.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="gpuspatialclustering"><code>GPUSpatialClustering</code></h3>
<p>Cluster planar points by density (DBSCAN), returning compact cluster labels, core flags and the cluster count.</p>
<p>::lens{scenario=gpu-spatial-analysis.spatial-clustering rung=result caption="DBSCAN with a 250 m radius and six-point cores finds 34 clusters among 2,344 observations; noise stays unclustered."}</p>
<p>Maximum: <code>gridSize</code> has at most <code>getGPUGridIndexMaximumCellCount(device)</code> (<code>2^25 - 1</code> on a default device: <code>cells + 1</code> uint32 cell offsets in one 128 MiB binding) cells; with <code>output.clusterCentroids</code> and sorted sums the cluster capacity is at most <code>GPU_SPATIAL_CLUSTERING_MAXIMUM_SORTED_CLUSTER_COUNT</code> (<code>2^24 - 1</code>, one 256-thread workgroup per cluster).</p>
<p>DBSCAN density clustering of planar points. Bounds, <code>epsilon</code>, and <code>minimumPoints</code> are per-frame
parameters (<code>getGPUSpatialClusteringParameterValues</code>). <code>gridSize</code> is only a compile-time maximum
lattice. Each encoding derives an active lattice whose cells are at least <code>epsilon</code> wide, so the
3×3 neighborhood search is always exact and <code>epsilon</code> changes never recompile. The neighbor scans
read positions and core flags from copies in grid-cell order, so the points of one cell are read
from consecutive memory whatever the input order. Clusters are joined
by a lock-free union-find that hooks larger roots under smaller ones. Labels are therefore
canonical: compact cluster IDs follow the smallest core row of each cluster, and a border point
joins the adjacent cluster with the smallest root. Outputs are per-point labels, roots, and core
flags, the cluster count, a compact list of cluster representatives, and cluster sizes and
centroids, all in the <code>output</code> bag. The settings type is <code>GPUSpatialClusteringSettings</code>.</p>
<p>With zero positions, only cluster counts, sizes, centroids and compact status scalars are written.
The zero-length per-point views may share their unused input buffer; no dummy storage row is
required. Writable cluster outputs still need valid storage bindings and disjoint logical ranges;
sizes and centroids may share an aligned binding prefix because separate passes fill them.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUSpatialClustering</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  positions, parameters: parameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph), gridSize: [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">128</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">128</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {labels, clusterCount, clusters: {ids: clusterRoots, status: {count, overflow}}, clusterSizes, clusterCentroids}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">parameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">write</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">getGPUSpatialClusteringParameterValues</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({bounds, epsilon: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">25</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, minimumPoints: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">8</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span></code></pre>
<p>With <code>minimumPoints = 1</code> every point is a core point, so clusters are the connected components of the
graph that joins points within <code>epsilon</code>, with no noise. That is PostGIS <code>ST_ClusterWithin(geom, distance)</code> (single-linkage clustering by distance, with the same transitive chaining); larger values
are <code>ST_ClusterDBSCAN(geom, eps, minpoints)</code>. <code>GPUSpatialClustering</code> writes labels only: pass them to
<a href="#gpugroupgeometry"><code>GPUGroupGeometry</code></a> and <a href="#gpugroupconvexhull"><code>GPUGroupConvexHull</code></a> for cluster
bounds, centers and outlines.</p>
<p><code>denseBoxShortcut: true</code> enables the FDBSCAN dense-box shortcut. Points are binned into boxes of side
<code>0.7 * epsilon</code>, and a box with at least <code>minimumPoints</code> points makes all of its points core without a
neighbor scan. Labels, roots and core flags are identical to the plain algorithm. The union kernel
still scans neighbors, so the measured gain on 200,000 clustered points is only 3 to 9 percent, and
the option is off by default. Points whose box index exceeds 2^16 skip the shortcut.</p>
<p><code>output.drawInstanceCount</code> (a one-row <code>uint32</code>, needs <code>output.clusters</code>) receives the clamped cluster count, so a layer
can draw one instance per cluster: import an indirect draw record's instance count with
<code>graph.importGPUData(id, drawCommands.getInstanceCountData(0))</code> and pass the view.</p>
<p><code>sumOrder</code> defaults to <code>'sorted'</code> (stable sort of members by cluster, then a fixed-order segmented sum):
because labels are canonical, <code>clusterCentroids</code> is bitwise reproducible across encodings on one device,
and cluster sizes are exact either way. <code>'atomic'</code> skips the sorts but uses float atomics, so last bits vary
between runs. Behavior change: last-bit differences versus the previous atomic default.</p>
<p><code>spatialContext</code> (optional) declares the coordinate space of <code>positions</code>. Omitted or planar, the search is
Euclidean in position units. It measures distance, so <code>local-lambert-equal-area</code> frames throw. <code>{coordinateSpace: 'longitude-latitude', metric: 'great-circle'}</code> clusters
<code>[longitude, latitude]</code> degrees with <code>epsilon</code> a great-circle distance in meters (kilometers with
<code>units: 'kilometers'</code>, or any <code>sphereRadius</code>; the mean Earth radius by default). Then <code>bounds</code> are
<code>[minLon, minLat, maxLon, maxLat]</code> degrees inside <code>[-180, 180] x [-90, 90]</code>. The neighbor test is the
haversine term against <code>sin^2(epsilon / (2 R))</code>, and each encoding sizes its cells to
<code>epsilon / R</code> of latitude by <code>2 asin(sin(epsilon / (2 R)) / cos(maxAbsLatitude))</code> of longitude (the whole
extent in one column near a pole), so the 3x3 search stays exact. The search wraps at the
antimeridian: points on the two sides of +-180 are neighbours when their great-circle distance is within
<code>epsilon</code>, so a cluster straddling +-180 is one cluster, as in <code>GPUOPTICS</code>. Cluster centroids remain arithmetic
means of the coordinates, so shift longitudes into a continuous range first if the centroid of a straddling
cluster is needed. <code>denseBoxShortcut</code>
is planar only, and other metrics (rhumb, ellipsoidal) throw. Geographic results match scikit-learn
<code>DBSCAN(metric='haversine')</code> on the test scene.
Planar positions are float32, so the representable spacing near a coordinate of magnitude <code>M</code> is about
<code>M * 1.2e-7</code>: UTM northings near 5e6 are only 0.5 m apart, and an <code>epsilon</code> below a few such spacings
(a few metres) is unreliable. Subtract a local origin from the coordinates (and from <code>bounds</code>) before
upload so the spacing stays far below <code>epsilon</code>.
Pass optional <code>spatialContext</code> to <code>getGPUSpatialClusteringParameterValues(settings)</code> to warn on short <code>epsilon</code> values; see <a href="/luma-contributors/docs/api/gpu-spatial-analysis/concepts#precision-of-longitudelatitude-positions">Precision of longitude/latitude positions</a>.</p>
<h4 id="how-it-computes-8">How it computes</h4>
<p>{/* lens:GPUSpatialClustering */}</p>
<p>::lens{scenario=gpu-spatial-analysis.spatial-clustering rung=datum bind=page datum=1000 caption="Inspect observation 1000: its position, core flag and cluster label."}
::lens{scenario=gpu-spatial-analysis.spatial-clustering rung=formula caption="Follow the core test: observations within epsilon, counted against the minimum points."}
::lens{scenario=gpu-spatial-analysis.spatial-clustering rung=kernel caption="Inspect the grid-binned core, union and label kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.spatial-clustering rung=graph caption="Trace the cell grid, core flags, lock-free union-find and compact labels over 2,344 observations."}</p>
<p>{/* parameters: GPUSpatialClustering generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpuspatialclustering-parameters"><code>GPUSpatialClustering</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUSpatialClusteringParameterValues(settings)</code>, writes <code>GPU_SPATIAL_CLUSTERING_PARAMETER_LENGTH</code> (8) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPUSpatialClusteringParameterValues</code>:</p>
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
<td><code>bounds</code></td>
<td><code>readonly [number, number, number, number]</code></td>
<td>Inclusive <code>[minX, minY, maxX, maxY]</code> extent of the data. Points outside it, or with a non-finite coordinate, are excluded and labeled noise. Bounds should cover the data: they size the active cell lattice, and a tighter extent makes the neighbor search cheaper. Bounds with <code>maxX &#x3C; minX</code> or <code>maxY &#x3C; minY</code> exclude every point.</td>
</tr>
<tr>
<td><code>epsilon</code></td>
<td><code>number</code></td>
<td>Neighbor radius. Two points are neighbors when their distance is <code>&#x3C;= epsilon</code>. Positive.</td>
</tr>
<tr>
<td><code>minimumPoints</code></td>
<td><code>number</code></td>
<td>Integer <code>>= 1</code>. A point is a core point when its epsilon-neighborhood, including the point itself, holds at least this many points.</td>
</tr>
<tr>
<td><code>timeEpsilon</code> (optional)</td>
<td><code>number</code></td>
<td>Space-time clustering (ST-DBSCAN): two points are neighbors only when, besides the spatial test, their relative times differ by at most this many ticks of the contributor's <code>temporalContext</code> unit. Positive. Required when the contributor has <code>times</code>, and ignored otherwise. Stored at index 6.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="gpuspatialkmeans"><code>GPUSpatialKMeans</code></h3>
<p>Partition planar points into k clusters by Lloyd iterations, returning labels, centers, sizes, squared distances, inertia and status.</p>
<p>::lens{scenario=gpu-spatial-analysis.spatial-k-means rung=result caption="k-means assigns 2,344 observations to six clusters; inertia 3.1e9 m² after 14 iterations, converged."}</p>
<p>Deterministic k-means over planar points (scikit-learn <code>KMeans</code>, PostGIS <code>ST_ClusterKMeans</code>). Props:
<code>positions</code>, optional planar <code>spatialContext</code> (longitude/latitude throws; project first; it measures
distance, so <code>local-lambert-equal-area</code> frames also throw), <code>k</code>
(1 to 256), <code>maximumIterations</code> (1 to 64), <code>initialization</code> (<code>'kmeans++'</code> by default, or <code>'first-valid'</code>),
<code>restartCount</code> (1 to 32, default 1), the per-frame <code>parameters</code> view
(<code>getGPUSpatialKMeansParameterValues({seed, tolerance})</code>, <code>GPU_SPATIAL_KMEANS_PARAMETER_LENGTH</code> <code>uint32</code> words) and the
<code>output</code> bag: <code>labels</code> (cluster or noise), <code>centers</code> (<code>k</code> rows), optional <code>sizes</code>, <code>squaredDistances</code> (the
explain column), <code>inertia</code> (a one-row <code>float32</code>: the total inertia, the sum of <code>squaredDistances</code>),
<code>status.iterationCount</code> and <code>status.converged</code>.
Behavior change: the default <code>initialization</code> was <code>'first-valid'</code>, which is order dependent and can place
centers next to each other; <code>'kmeans++'</code> matches scikit-learn. Each point goes to the nearest center with lowest-ID ties, means are fixed-order chunked
reductions (no sort). Results are bitwise reproducible per seed. <code>'kmeans++'</code> uses <code>log</code>, which WGSL does not
pin down across devices.
Behavior change (2026-11-10): the <code>'kmeans++'</code> draws are Philox 4x32-10 blocks per <code>(row, round)</code> under the
registered purpose <code>kMeansPlusPlusSeeding</code>, mapped to the open interval <code>(0, 1)</code>; they were a private integer hash
whose largest value mapped to exactly 1. The centers a given <code>seed</code> selects therefore differ from earlier builds.
When every remaining point coincides with a chosen center, the next center repeats the lowest valid row (finite, not
NaN) and its empty cluster is relocated; a center is NaN only when it could not be initialized at all.</p>
<p>Behavior change (empty clusters): a center that receives no point is no longer left in place. After every
assignment step the empty clusters (lowest ID first) are relocated like scikit-learn's
<code>_relocate_empty_clusters_dense</code>: they take the points with the largest squared distance to their assigned
center, farthest first, and the donor clusters' counts and sums lose those points before the means are
taken. Ties in distance go to the lowest row. scikit-learn's <code>argpartition</code> leaves the order among the
relocated points unspecified; here it is always descending distance. If every point sits on its center
nothing is relocated, and a cluster that is still empty (or whose only member was taken) moves to the mean of
the largest cluster (lowest ID on ties). A center that is NaN because it could not be initialized (fewer valid
points than <code>k</code>) is never relocated. The selection is a single-workgroup iterated argmax with no atomics and
no readback; it costs <code>O(n E)</code> for <code>E</code> empty clusters and nothing when none is empty.</p>
<p><code>restartCount</code> runs that many independent seeded initializations, each with its own Lloyd iterations, and
returns the one with the lowest total inertia (a fixed-order compensated sum; a tie goes to the lowest
restart index). Restart <code>r</code> uses the seed <code>getGPUSpatialKMeansRestartSeed(seed, r)</code> (restart 0 uses <code>seed</code>), so any
single restart can be reproduced with <code>restartCount: 1</code> and that seed. The default of 1 matches
scikit-learn's <code>n_init='auto'</code> for k-means++. Restarts are separate node chains in the graph, so graph size
and GPU time grow linearly with <code>restartCount</code>; <code>status.iterationCount</code>, <code>status.converged</code> and <code>inertia</code> describe the
returned restart.
<code>restartCount</code> above 1 needs <code>'kmeans++'</code> (<code>'first-valid'</code> has no randomness).</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUSpatialKMeans</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({positions, k: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">8</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, maximumIterations: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">20</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, initialization: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'kmeans++'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  parameters, output: {labels, centers, sizes}}));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">parameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">write</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">getGPUSpatialKMeansParameterValues</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({seed: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span></code></pre>
<p><code>maximumIterations</code> is a cap: a cap that is too low reports <code>status.converged</code> 0 (first-valid initialization needs 20 to 30
iterations on taxi data at <code>k = 16</code>), and the largest shift usually falls straight to exactly 0, a true
Lloyd fixed point, so a tolerance relative to the data extent is enough. <code>tolerance</code> (per frame, default 0) is a convergence threshold on the largest center shift. After an
iteration whose shift is at most <code>tolerance</code>, the one-thread convergence check zeroes a GPU-written
indirect dispatch, so the later iterations' assign, reduction and relocation kernels do not launch; the
result equals running every iteration when <code>tolerance</code> is 0 (an exact fixed point). The optional one-row
<code>status.iterationCount</code> and <code>status.converged</code> outputs report the iterations used and the flag. The seed and tolerance
are <code>parameters</code> words and the k-means++ round and the restart index come from a GPU cursor word, so the graph compiles one pipeline per kernel at any <code>k</code> and
<code>restartCount</code>, and restart scoring is a two-stage (per-chunk, then fixed-order) reduction.</p>
<h4 id="how-it-computes-9">How it computes</h4>
<p>{/* lens:GPUSpatialKMeans */}</p>
<p>::lens{scenario=gpu-spatial-analysis.spatial-k-means rung=datum bind=page datum=1000 caption="Inspect observation 1000: its cluster label and squared distance to that center."}
::lens{scenario=gpu-spatial-analysis.spatial-k-means rung=formula caption="Follow the inertia and the center update: squared distance to the nearest center, then member means."}
::lens{scenario=gpu-spatial-analysis.spatial-k-means rung=kernel caption="Inspect the assignment, chunked reduction and center-update kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.spatial-k-means rung=graph caption="Trace k-means++ seeding and 20 unrolled Lloyd iterations down to the inertia reduction."}</p>
<p>{/* parameters: GPUSpatialKMeans generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpuspatialkmeans-parameters"><code>GPUSpatialKMeans</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUSpatialKMeansParameterValues(settings?)</code>, writes <code>GPU_SPATIAL_KMEANS_PARAMETER_LENGTH</code> (2) <code>uint32</code> values.</li>
</ul>
<p>Settings of <code>getGPUSpatialKMeansParameterValues</code>:</p>
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
<td>Seed of the <code>'kmeans++'</code> draws, an integer in <code>[0, 2^32 - 1]</code>. Default 0. Restart <code>r</code> is seeded with <code>getGPUSpatialKMeansRestartSeed(seed, r)</code>.</td>
</tr>
<tr>
<td><code>tolerance</code> (optional)</td>
<td><code>number</code></td>
<td>Convergence tolerance: an iteration converges when the largest center shift (Euclidean, in position units) is at most this value. Finite and <code>>= 0</code>; default <code>0</code>, which stops only at an exact fixed point and so gives the same result as running every iteration. A positive value trades exactness for fewer iterations. After convergence the remaining iterations' assign, reduction and relocation kernels are skipped through a GPU-written indirect dispatch, so each one costs only a one-thread convergence check.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="gpuhierarchicaldensityclustering"><code>GPUHierarchicalDensityClustering</code></h3>
<p>Cluster planar points by density at every scale (HDBSCAN), returning compact cluster labels, core distances and the cluster count. Optional relative times and a per-frame time scale enable <a href="/luma-contributors/docs/api/gpu-spatial-analysis/space-time-clustering#st-hdbscan">scaled ST-HDBSCAN</a> through the same contributor.</p>
<p>::lens{scenario=gpu-spatial-analysis.hierarchical-density-clustering rung=result caption="HDBSCAN finds 19 clusters of different density among 782 observations without a radius; the rest is noise."}</p>
<p>HDBSCAN density clustering of planar points, fully on the GPU, matching scikit-learn's <code>HDBSCAN</code> with its
defaults (<code>cluster_selection_method='eom'</code>, <code>allow_single_cluster=False</code>, <code>cluster_selection_epsilon=0</code>,
Euclidean) up to a relabeling of the clusters. Props: <code>positions</code> (at most
<code>GPU_HIERARCHICAL_DENSITY_CLUSTERING_MAXIMUM_ROW_COUNT</code>, 2^20 rows), the compile-time <code>minimumClusterSize</code>
(<code>min_cluster_size</code>, at least 2) and <code>minimumSamples</code> (<code>min_samples</code>, the point itself counted, 1 to 32,
default <code>min(minimumClusterSize, 32)</code>), and <code>output: {labels, coreDistances?, status?: {count}}</code>. Labels are
compact cluster IDs or <code>0xffffffff</code> for noise; non-finite points are noise.</p>
<p>The pipeline: core distances from a <code>GPUNeighborSearch</code> k-nearest self join over the extent of the finite points (the point itself is the first sample); the exact minimum spanning tree over the mutual
reachability distance by Boruvka (unrolled <code>ceil(log2 rows) + 1</code> rounds of the same kernels, gated on the GPU
component count; each point walks a Hilbert-sorted BVH and skips nodes that hold only its own component);
the single-linkage dendrogram by parallel parent splicing over the sorted tree, with Euler-tour pointer
doubling for subtree sizes; condensing by pointer jumping; stability as a fixed-order segmented sum; and
excess-of-mass selection by heavy-path tree contraction. The tree passes use GPU convergence flags and
indirect dispatch gates, including on deep chains; the row maximum bounds transient storage.
Where mutual reachability distances tie, several minimum spanning trees exist and the GPU breaks ties by point
rows, which can condense a few points differently from scikit-learn: on the oracle scenes
(<code>test/oracles/spatial-clustering-hdbscan.py</code>) the spanning-tree weights are identical and the adjusted Rand
index is at least 0.98. Membership probabilities, outlier scores, <code>allow_single_cluster</code> and <code>'leaf'</code>
selection are not provided.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUHierarchicalDensityClustering</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  positions, minimumClusterSize: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">15</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, minimumSamples: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {labels, coreDistances, status: {count: clusterCount}}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span></code></pre>
<h4 id="how-it-computes-10">How it computes</h4>
<p>{/* lens:GPUHierarchicalDensityClustering */}</p>
<p>::lens{scenario=gpu-spatial-analysis.hierarchical-density-clustering rung=datum bind=page datum=300 caption="Inspect observation 300: its core distance and cluster label."}
::lens{scenario=gpu-spatial-analysis.hierarchical-density-clustering rung=formula caption="Follow core distance, mutual reachability and cluster stability, which decide the surviving clusters."}
::lens{scenario=gpu-spatial-analysis.hierarchical-density-clustering rung=kernel caption="Inspect the k-nearest, Boruvka, dendrogram, condensing and selection kernels with retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.hierarchical-density-clustering rung=graph caption="Trace core distances, the spanning tree, the dendrogram and excess-of-mass selection over 782 observations."}</p>
<p>{/* parameters: GPUHierarchicalDensityClustering generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpuhierarchicaldensityclustering-parameters"><code>GPUHierarchicalDensityClustering</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUHierarchicalDensityClusteringParameterValues(settings)</code>, writes <code>GPU_HIERARCHICAL_DENSITY_CLUSTERING_PARAMETER_LENGTH</code> (4) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPUHierarchicalDensityClusteringParameterValues</code>:</p>
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
<td><code>timeScale</code></td>
<td><code>number</code></td>
<td>Spatial position units per temporal-context tick. Finite and non-negative; zero ignores time distances.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="gpuspatialweightsminimumspanningtree-gpuskaterregions-and-gpuregionpartitionevaluation"><code>GPUSpatialWeightsMinimumSpanningTree</code>, <code>GPUSkaterRegions</code>, and <code>GPURegionPartitionEvaluation</code></h3>
<p><code>GPUSpatialWeightsMinimumSpanningTree</code>: Find the minimum spanning forest of a contiguity graph by attribute dissimilarity, returning tree-edge flags, component labels and edge costs.</p>
<p>Either weights loss flag suppresses the forest: tree flags are zero, component labels are
<code>0xffffffff</code> (unavailable), and compact count/requiredCount are zero. The producer's <code>output</code>
port carries compact candidateOverflow for either upstream cause; compact overflow only reports
compact destination capacity. SKATER on the same weights suppresses labels (<code>0xffffffff</code>),
region count (zero/unavailable), cut IDs (no cut) and gains (NaN). Clearing loss restores normal
results on resubmission. Showcase/explorer region and coloring readouts withhold definite results
while either producer flag is set; a sentinel is never a region, color or observed count.
<code>GPUSkaterRegionsOperation</code> and <code>CPUSkaterRegionsOperation</code> accept the two graph value names under
<code>weights.status</code>. The pure <code>computeSkaterRegionsOnCPU</code> accepts the corresponding uint32 arrays
under <code>input.status</code>, alongside its flat CSR arrays, and uses the same suppression sentinels.
Both operation forms read the supplied loss values on every run of a reused compiled graph;
clearing a loss word restores normal outputs on the next run.
Pass the complete producer weights port to every forest and partition stage, including REDCAP
and Ward; a clean supplied status has the same result as omitted status. MST's packed
<code>output.status</code> requires <code>GPU_MINIMUM_SPANNING_TREE_STATUS.length</code> words (now two):
<code>invalidCount</code> stays at word zero and <code>candidateOverflow</code> at word one reports either upstream
adjacency loss cause as 0/1, even without compact edges. Increase old one-word allocations to
the named length. Compact <code>output.edges.status.candidateOverflow</code> reports the same adjacency loss.
SKATER consumes the forest's loss word through <code>treeStatus</code>, in both operation forms and its CPU
twin. On clean and suppressed frames it writes one region-count word and
<code>maximumRegionCount - 1</code> cut entries; spare output capacity stays untouched.
Forest, SKATER, LOSH and multivariate Geary capture their declared props and nested port maps at
construction, including inherited getters and masks. Change GPU buffer contents to resubmit;
construct a new contributor to change a view or destination.
Both named SKATER operations accept <code>weights.status.overflow</code> and <code>candidateOverflow</code> value names
and apply this suppression before CPU forest traversal or GPU cuts. The pure CPU twin accepts the
same scalar arrays in <code>input.status</code>. A missing forest label during loss is unavailable, not an
invalid-tree exception. Only one <code>regionCount</code> word and <code>maximumRegionCount - 1</code> cut-log words are
written, including during suppression; spare destination words are preserved.</p>
<p>::lens{scenario=gpu-spatial-analysis.spatial-weights-minimum-spanning-tree rung=result caption="The 135 tree edges joining 136 tracts, drawn between centroids over poverty and coloured by dissimilarity; total 29.6."}</p>
<p><code>GPUSkaterRegions</code>: Cut a spanning tree greedily where the within-region sum of squares falls most, returning region labels, cut edges and cut gains.</p>
<p>::lens{scenario=gpu-spatial-analysis.skater-regions rung=result caption="Target five regions, with at least five tracts per cut side to exclude singleton regions."}</p>
<p>This example sets <code>minimumRegionSize: 5</code> (spopt's <code>floor</code>) because the default of one permits
singleton regions. Each cut maximizes SSD reduction among cuts satisfying that floor; it does
not balance region sizes and may stop below the target if no feasible cut remains.</p>
<p><code>GPURegionPartitionEvaluation</code>: Score any label column by within-region, between-region and total sum of squares, region sizes, boundary share and pseudo-F.</p>
<p>::lens{scenario=gpu-spatial-analysis.region-partition-evaluation rung=result caption="Nine summary words for the eight community areas as a partition of poverty; pseudo-F is 13.7."}</p>
<p>These cards use community areas 1 to 8 (136 contiguous north-side Chicago tracts). The spanning-tree card
draws tree edges as segments between GPU-computed tract centroids.</p>
<p>Maximum: SKATER keeps <code>weights.neighbors rows x (2 + columnCount)</code> arc statistics, at most <code>getGPUSkaterMaximumStatisticCount(device)</code> (<code>2^25</code> on a default device); <code>GPURegionPartitionEvaluation</code> takes at most <code>getGPURegionPartitionEvaluationMaximumRowCount(device)</code> (<code>2^24</code>) rows, <code>getGPURegionPartitionEvaluationMaximumLabelCapacity(device)</code> (<code>2^25 - 1</code>) labels and <code>labelCapacity x columnCount</code> at most <code>getGPURegionPartitionEvaluationMaximumRegionValueCount(device)</code> (<code>2^25</code>).</p>
<p>Spatially constrained regionalization over a contiguity graph (PySAL spopt <code>SpanningForest</code> and SKATER,
Assuncao et al. 2006).</p>
<ul>
<li><code>GPUSpatialWeightsMinimumSpanningTree</code>: Boruvka minimum spanning forest over a symmetric
<code>GPUSpatialWeights</code> (every link in both rows; for directed weights build the union with
<code>GPUSpatialWeightsTranspose</code> first, unlike the AZP, Max-P and Ward operators, which symmetrize
themselves because the tree's edge IDs are the caller's CSR slots). The edge cost is the squared Euclidean distance between (optionally standardized)
attribute rows. A total order of <code>(cost, lower-row CSR slot)</code> makes the forest unique and equal to
Kruskal under that order, with integer atomics and no float races. <code>output</code> holds <code>treeEdgeFlags</code>
(per CSR slot), <code>componentLabels</code> (the minimum row of each tree) and, optionally, a compact <code>edges</code>
list (<code>ids</code> plus <code>status: {count, overflow, requiredCount?, candidateOverflow?}</code>) with <code>edgeEndpoints</code>, <code>edgeDissimilarities</code> and <code>standardizedValues</code>. It unrolls <code>ceil(log2 rows)</code> rounds. Tied costs
can pick different edges than scipy's traversal order, with the same total cost.
Non-finite attribute entries are skipped as in <code>GPUAZPRegions</code>, <code>GPUWardRegions</code> and <code>GPUMaxPRegions</code>:
the standardization mean and deviation use the finite entries of each column, so one NaN no longer
makes every edge cost NaN, and the entry stays non-finite in <code>standardizedValues</code>. Every link that
touches a row with a non-finite entry costs <code>+Infinity</code>, so it joins the forest last and the forest
still spans each connected component. The optional <code>output.status</code> stores the skipped-entry count,
before standardization, at <code>GPU_MINIMUM_SPANNING_TREE_STATUS.invalidCount</code>; its second word,
<code>GPU_MINIMUM_SPANNING_TREE_STATUS.candidateOverflow</code>, independently reports adjacency loss.
Unlike the three solvers above, the tree is still built when the count is nonzero. Pass the status
view to <code>GPUSkaterRegions</code> as <code>treeStatus</code>: like those solvers, SKATER then makes no cut while the
count is nonzero (each tree stays one region and every <code>cutEdges</code> entry is <code>GPU_SKATER_NO_CUT</code>).
A nonzero adjacency-loss word suppresses labels/count/cuts/gains instead of publishing a partition.</li>
<li><code>GPUSkaterRegions</code>: greedy SKATER cuts of that tree. Each step evaluates the sum-of-squared-deviation
reduction of every tree edge in parallel (Euler-tour side sums) and applies the best one, ties to the
lowest slot, while honoring <code>minimumRegionSize</code>. The reduction is computed in its direct form
<code>nA nB / (nA + nB) |mean A - mean B|^2</code> from the two sides' sizes and attribute sums (the Ward increase), not as
a difference of three sums of squares, so a small gain inside a quiet region whose mean is far from zero is not
lost to float32 rounding; the side sums carry the exact rounding error of each addition. What remains is the
last place of the region's attribute sum spread over the smaller side, which centred attributes keep small. The target region count and the minimum size are
per-frame <code>parameters</code> words written with
<code>getGPUSkaterRegionsParameterValues({regionCount, minimumRegionSize})</code>
(<code>GPU_SKATER_REGIONS_PARAMETER_LENGTH</code> words); at most <code>min(maximumRegionCount - 1, rows - 1)</code> steps
are compiled, so the
partition with k regions is a prefix of the one with k + 1. <code>output</code> holds <code>labels</code> (the row index of
each region's top), <code>regionCount</code>, <code>cutEdges</code> and <code>cutGains</code>. The partitions equal spopt <code>SpanningForest</code> (squared-Euclidean metric) exactly, including
floors and islands; spopt's <code>islands='increase'</code> adds the tree count to <code>n_clusters</code>, while here the
target counts every region. SKATER does not center attributes itself: the tree's <code>standardize</code>
(default true) does, and <code>standardizedValues</code> should be passed as the SKATER <code>values</code> so the cuts and
the tree share a metric and the f32 subtree-sum subtraction stays well conditioned.
The optional cut logs still contain <code>maximumRegionCount - 1</code> entries; entries beyond the attainable
cuts are <code>GPU_SKATER_NO_CUT</code> and zero gain.</li>
<li><code>GPURegionPartitionEvaluation</code>: scores any label column (SKATER, <code>GPUSpatialClustering</code>, <code>GPUSpatialKMeans</code>):
region count, within, between and total sum of squared deviations, smallest and largest region, the
fraction of weights-graph links that cross regions (a compactness proxy) and the number of ignored
rows, plus per-label sizes and within-region deviations. The summary also holds <code>pseudoF</code>, the
Calinski-Harabasz statistic <code>(between / (k - 1)) / (within / (n - k))</code>, pinned against
<code>sklearn.metrics.calinski_harabasz_score</code> (NaN where undefined, 1 when the within-region dispersion
is zero, as sklearn). <code>output</code> holds <code>summary</code> (<code>GPU_REGION_PARTITION_EVALUATION_LAYOUT</code>: <code>withinSumOfSquares</code>,
<code>betweenSumOfSquares</code>, <code>totalSumOfSquares</code>, ...) and the optional <code>regionSizes</code> and
<code>regionWithinSumOfSquares</code>. The cost is <code>O(labelCapacity * rows)</code>.
<code>betweenSumOfSquares</code> is computed directly as <code>sum over regions of n_r |mean_r - mean|^2</code>, not as
<code>total - within</code>, so it and <code>pseudoF</code> keep their digits when the partition explains a small share of the total
(it equals <code>total - within</code> up to float32 rounding).
A labelled row with a NaN or infinite value in any column is left out of every statistic, as an unlabelled row
is: it is counted in <code>ignoredRows</code>, and the optional one-row <code>output.invalidRowCount</code> says how many such rows
there were. One missing value removes one row; it no longer turns every statistic into NaN.</li>
</ul>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUSpatialWeightsMinimumSpanningTree</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({weights, values, columnCount: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {treeEdgeFlags, componentLabels, standardizedValues}}));</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// skater: a GPUParameterBuffer written with getGPUSkaterRegionsParameterValues({regionCount: 8})</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUSkaterRegions</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({weights, treeEdgeFlags, componentLabels, values: standardizedValues,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  columnCount: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, parameters: skater.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph), maximumRegionCount: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">12</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {labels, regionCount}}));</span></span></code></pre>
<p>SKATER, Ward, AZP and Max-p expose <code>estimatedCommandNodeCount</code> and reject construction when this
upper bound exceeds the shared 32,768-node limit. The estimate includes transitive adjacency,
sort/scan, contiguity and nested refinement nodes. Published primitives have named conservative
allowances, checked against emitted graphs in node tests. A GPU dispatch gate saves execution
after convergence; the construction budget still counts every unrolled round.
SKATER's bound includes loss aggregation and each requested suppression pass. With loss ports
and all four outputs, 4,094 rows and <code>maximumRegionCount: 4094</code> require 32,773 nodes and are refused.
Ward and AZP snapshot their iteration budgets (and Ward's schedule) before validation. Ward's
mutual-nearest estimate uses 18 nodes per round through 4,096 input rows and 45 per round above
that threshold, matching the cooperative and radix selection routes respectively.</p>
<p>Cost: the publish and apply kernels are single-threaded and each SKATER step is
<code>O(sum of side sizes)</code>, which suits thousands of units and is quadratic on path-like trees of 100,000
rows or more. Once a step finds no cut or the target region count is reached, the one-thread kernels
zero a GPU-written indirect dispatch, so the remaining steps launch only their one-thread kernels. The
step index comes from a GPU cursor word, so every step shares one pipeline per kernel.</p>
<h4 id="how-it-computes-11">How it computes</h4>
<p>{/* lens:GPUSpatialWeightsMinimumSpanningTree <em>/}
{/</em> lens:GPUSkaterRegions <em>/}
{/</em> lens:GPURegionPartitionEvaluation */}</p>
<p><code>GPUSpatialWeightsMinimumSpanningTree</code>:</p>
<p>::lens{scenario=gpu-spatial-analysis.spatial-weights-minimum-spanning-tree rung=datum bind=page datum=40 caption="Inspect tract 40: its poverty value, standardized value and component label in the tree."}
::lens{scenario=gpu-spatial-analysis.spatial-weights-minimum-spanning-tree rung=formula caption="Follow the edge cost: squared difference of standardized attribute rows across each link."}
::lens{scenario=gpu-spatial-analysis.spatial-weights-minimum-spanning-tree rung=kernel caption="Inspect the standardize, edge-cost, cheapest-edge and relabel kernels with retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.spatial-weights-minimum-spanning-tree rung=graph caption="Trace queen weights, edge costs and eight Boruvka rounds that merge trees."}</p>
<p><code>GPUSkaterRegions</code>:</p>
<p>::lens{scenario=gpu-spatial-analysis.skater-regions rung=datum bind=page datum=40 caption="Inspect tract 40: its poverty value, tree component and SKATER region."}
::lens{scenario=gpu-spatial-analysis.skater-regions rung=formula caption="Follow the cut gain: region sum of squares minus the sums of its two sides."}
::lens{scenario=gpu-spatial-analysis.skater-regions rung=kernel caption="Inspect the Euler-tour, cut-gain, best-cut and apply kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.skater-regions rung=graph caption="Trace queen weights, the spanning tree and up to seven greedy cut steps."}</p>
<p><code>GPURegionPartitionEvaluation</code>:</p>
<p>::lens{scenario=gpu-spatial-analysis.region-partition-evaluation rung=datum bind=page datum=40 caption="Inspect tract 40: its poverty value and community-area label in the evaluated partition."}
::lens{scenario=gpu-spatial-analysis.region-partition-evaluation rung=formula caption="Follow within, between and total sums of squares into the Calinski-Harabasz pseudo-F."}
::lens{scenario=gpu-spatial-analysis.region-partition-evaluation rung=kernel caption="Inspect the sort, region-sum, deviation and summary kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.region-partition-evaluation rung=graph caption="Trace queen weights, the label sort, per-region sums and the nine-word summary."}</p>
<p>{/* parameters: GPUSkaterRegions generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpuskaterregions-parameters"><code>GPUSkaterRegions</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUSkaterRegionsParameterValues(settings)</code>, writes <code>GPU_SKATER_REGIONS_PARAMETER_LENGTH</code> (2) <code>uint32</code> values.</li>
</ul>
<p>Settings of <code>getGPUSkaterRegionsParameterValues</code>:</p>
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
<td><code>regionCount</code></td>
<td><code>number</code></td>
<td>Target region count, clamped to <code>maximumRegionCount</code> on the GPU. A value at or below the number of forest trees cuts nothing.</td>
</tr>
<tr>
<td><code>minimumRegionSize</code> (optional)</td>
<td><code>number</code></td>
<td>Minimum region size in rows. Defaults to 1.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="gpugroupgeometry"><code>GPUGroupGeometry</code></h3>
<p>Summarize the geometry of each label group, returning counts, bounds, centres, medoids, standard distances and ellipses.</p>
<p>::lens{scenario=gpu-spatial-analysis.group-geometry rung=result caption="Mean centres of the 34 DBSCAN clusters of 2,344 observations, sized by member count; the largest has 553."}</p>
<p>Planar only: <code>spatialContext</code> defaults to native planar coordinates and a longitude/latitude context
throws at construction (project first with <code>createGPULocalFrame</code> and <code>GPULocalFrameProjection</code>).
Geographic group centroids live in <code>GPUGeometryMeasures</code>.
Declares <code>measure: 'distance'</code>: a Lambert equal-area local frame throws; use an
azimuthal-equidistant frame.</p>
<p>Per-label summaries over a <code>uint32</code> label column, for example the output of <code>GPUSpatialClustering</code>:
count, bounds <code>[minX, minY, maxX, maxY]</code>, mean center, weighted center, weight sum, medoid, standard
deviational ellipse and standard distance.</p>
<ul>
<li>Props: <code>positions</code> (<code>float32x2</code>), <code>labels</code>, <code>groupCount</code>, optional <code>noiseLabel</code> and <code>weights</code>,
<code>parameters</code> (the <code>getGPUGeographicDistributionParameterValues</code> layout: origin, standard deviations,
ellipse convention), <code>medoidMaximumGroupSize</code> (default 4096) and <code>output</code>.</li>
<li>Outputs (all optional, at least one): <code>counts</code>, <code>bounds</code> (<code>float32x4</code>), <code>meanCenters</code>,
<code>weightedCenters</code> and <code>weightSums</code> (need <code>weights</code>), <code>medoidIndices</code>, <code>ellipses</code>
(<code>[angle, sigmaX, sigmaY]</code>, weighted when <code>weights</code> is given), <code>standardDistances</code> and <code>status.overflow</code>.</li>
<li>Labels at or above <code>groupCount</code> (for example DBSCAN noise, <code>0xffffffff</code>), a label equal to
<code>noiseLabel</code>, and rows with a non-finite coordinate are excluded. Empty groups give count 0, NaN
floats and <code>GPU_GROUP_GEOMETRY_NO_MEDOID</code>.</li>
<li>The medoid is the included row with the smallest summed Euclidean distance to the group, lowest row
on ties, unweighted. It is quadratic per group, so a group larger than <code>medoidMaximumGroupSize</code> gets
no medoid and sets <code>output.status.overflow</code>.</li>
<li>Sums are fixed-order, so results are bitwise reproducible.</li>
</ul>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUGroupGeometry</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  positions, labels: clusterLabels, groupCount: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">64</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  parameters: parameterBuffer.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {counts, bounds, meanCenters, medoidIndices, ellipses}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span></code></pre>
<h4 id="how-it-computes-12">How it computes</h4>
<p>{/* lens:GPUGroupGeometry */}</p>
<p>::lens{scenario=gpu-spatial-analysis.group-geometry rung=datum bind=page datum=1000 caption="Inspect observation 1000: its cluster label, the group it adds to."}
::lens{scenario=gpu-spatial-analysis.group-geometry rung=formula caption="Follow the group mean centre and the medoid, the member with the smallest summed distance."}
::lens{scenario=gpu-spatial-analysis.group-geometry rung=kernel caption="Inspect the sort, segmented bounds, centre, ellipse and medoid kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.group-geometry rung=graph caption="Trace DBSCAN labels, the sort by group, segmented reductions and the medoid search."}</p>
<p>{/* parameters: GPUGroupGeometry generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpugroupgeometry-parameters"><code>GPUGroupGeometry</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUGeographicDistributionParameterValues(settings?)</code>, writes <code>GPU_GEOGRAPHIC_DISTRIBUTION_PARAMETER_LENGTH</code> (8) <code>float32</code> values.
<ul>
<li>Input port: <code>parameters</code>.</li>
</ul>
</li>
</ul>
<p>Settings of <code>getGPUGeographicDistributionParameterValues</code>:</p>
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
<td><code>origin</code> (optional)</td>
<td><code>readonly [number, number]</code></td>
<td>Local origin <code>[x, y]</code> subtracted from every position before any sum, so float32 sums keep precision when the planar coordinates are large (for example Web Mercator meters). Choose a point near the data. Defaults to <code>[0, 0]</code>.</td>
</tr>
<tr>
<td><code>standardDeviations</code> (optional)</td>
<td><code>number</code></td>
<td>Number of standard deviations that scale <code>standardDistances</code>, <code>ellipses</code> and the vertex rings (ArcGIS offers 1, 2 and 3). Any positive number is accepted. Defaults to <code>1</code>.</td>
</tr>
<tr>
<td><code>ellipseConvention</code> (optional)</td>
<td><code>GPUGeographicDistributionEllipseConvention</code></td>
<td>Axis scaling of the ellipse: <code>'root-two'</code> scales the axes by <code>sqrt(2)</code> as the ArcGIS Directional Distribution tool does, <code>'standard'</code> uses the plain standard deviation along each principal axis. Defaults to <code>'root-two'</code>.</td>
</tr>
<tr>
<td><code>orientationOnly</code> (optional)</td>
<td><code>boolean</code></td>
<td>Treat lines as undirected (angles are doubled before averaging, so a line and its reverse agree). Only affects <code>directionalMeans</code>. Defaults to <code>false</code>.</td>
</tr>
<tr>
<td><code>medianTolerance</code> (optional)</td>
<td><code>number</code></td>
<td>Absolute tolerance, in position units, under which the last Weiszfeld step counts as converged for <code>medianConverged</code>. The iteration stops once every non-empty group's last step is at most this value. Non-negative. Defaults to <code>GPU_GEOGRAPHIC_DISTRIBUTION_DEFAULT_MEDIAN_TOLERANCE</code>.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="gpugroupconvexhull"><code>GPUGroupConvexHull</code></h3>
<p>Outline each label group by its convex hull, returning hull vertices in CSR layout with emitted counts and true sizes.</p>
<p>::lens{scenario=gpu-spatial-analysis.group-convex-hull rung=result caption="Convex hulls of the 34 DBSCAN clusters, one outline per cluster; the largest has 15 vertices."}</p>
<p>Planar only: a longitude/latitude <code>spatialContext</code> throws at construction; project first with
<code>createGPULocalFrame</code> and <code>GPULocalFrameProjection</code>.</p>
<p>Convex hull per label (Andrew monotone chain), for cluster and hot-region outlines and trajectory
footprints (PostGIS <code>ST_ConvexHull</code> per group, <code>GeoSeries.convex_hull</code>). Props: <code>positions</code>, <code>labels</code>,
<code>groupCount</code>, optional <code>noiseLabel</code>, <code>maximumVerticesPerGroup</code> (at least 3), <code>vertexCapacity</code> and
<code>output</code>: <code>vertexIndices</code>, optional <code>vertexPositions</code>, <code>offsets</code> (<code>groupCount + 1</code>), <code>counts</code>, optional
<code>sizes</code> (true hull sizes) and <code>status.overflow</code> (0 or 1). Optional one-row diagnostics
<code>status.groupOverflow</code> and <code>status.vertexOverflow</code> identify the per-group and total vertex limits,
respectively. Every supplied status is rewritten each frame, including empty groups.</p>
<p><code>status.overflow</code> previously packed bits 1 and 2; code testing those bits must use the corresponding
named diagnostic. The exported <code>GPU_GROUP_CONVEX_HULL_GROUP_OVERFLOW</code> and
<code>GPU_GROUP_CONVEX_HULL_TOTAL_OVERFLOW</code> constants remain available but are deprecated; they no longer
apply to <code>status.overflow</code>.</p>
<ul>
<li>Hulls are counter-clockwise, not closed, and start at the smallest <code>(x, y)</code> vertex. Collinear points
are excluded, equal points collapse to the lowest row, one distinct point gives one vertex, and
all-collinear points give the two extremes.</li>
<li>Orientation is exact: coordinates are snapped to a 2^29 lattice with one global power-of-two scale,
and the sign is computed with 64-bit integer arithmetic. Output positions are the original floats.</li>
<li>The lattice is global, derived from the extent of all rows rather than per group. The hull is exact
for the snapped points, not for the raw floats, and adding a distant group (or any row that widens
the extent) can change the lattice scale and so the hull of an unrelated small group.</li>
<li>A hull over the per-group cap is dropped whole. Groups are admitted in group order until the total
capacity runs out, and the first group that does not fit and all later ones are dropped (<code>counts</code> 0,
<code>sizes</code> still true).</li>
<li><code>extremePointFilter</code> (default <code>true</code> up to 65536 groups) is an Akl-Toussaint filter before the sorts:
per group it finds the eight extreme lattice points (min and max of x, y, x + y and x - y) and gives
every row strictly inside their octagon (exact test) the excluded-row key. Boundary and collinear
rows are kept, so the hull is bit for bit the same with <code>true</code> or <code>false</code>. <code>GPUSort</code> has a fixed
length and cannot be skipped from the GPU, so the three sorts still cover every row; the saving is in
the prefilter and chain, which skip the discarded run. The extreme tables hold 16 words per group, so
forcing it on above 65536 groups throws.</li>
<li><code>prefilterLevels</code> (0 to 4, <code>GPU_GROUP_CONVEX_HULL_MAXIMUM_PREFILTER_LEVELS</code>; default 2 from 8192 rows)
discards points inside nested octagon-like filters before the chain. For one 500k-point disk the median
went from 551 ms (level 0) to 36, 8.8 and 7.6 ms (levels 1 to 3); 64 groups of 500k: 33 to 7.3 ms. With
4096 small groups the levels cost 2 to 3 ms (3.0 ms at level 0), so set 0 there.</li>
<li>The chain is one invocation per group, so a group whose points are all on the hull stays serial. For a
concave outline, close a raster <code>GPUDistanceField</code>; exact alpha shapes stay on the CPU.</li>
</ul>
<h4 id="how-it-computes-13">How it computes</h4>
<p>{/* lens:GPUGroupConvexHull */}</p>
<p>::lens{scenario=gpu-spatial-analysis.group-convex-hull rung=datum bind=page datum=1000 caption="Inspect observation 1000: its cluster label, the group whose hull it may bound."}
::lens{scenario=gpu-spatial-analysis.group-convex-hull rung=formula caption="Follow the exact lattice orientation test that keeps only counter-clockwise hull turns."}
::lens{scenario=gpu-spatial-analysis.group-convex-hull rung=kernel caption="Inspect the lattice, extreme-point filter, sort and monotone-chain kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.group-convex-hull rung=graph caption="Trace DBSCAN labels, three sort passes, the per-group chain and the hull vertex emission."}</p>
<h3 id="gpuneighborhoodsummary"><code>GPUNeighborhoodSummary</code></h3>
<p>Summarise each tract's neighbourhood from a spatial weights table, returning a row-standardised mean of values and the entropy of neighbour labels.</p>
<p>::lens{scenario=gpu-spatial-analysis.neighborhood-summary rung=result caption="Mean poverty of each tract's queen neighbours across 791 tracts, from row-standardised weights; range 4.2 to 57.5%."}</p>
<p>Per-row summaries of a neighborhood over a <code>GPUSpatialWeights</code> CSR (momepy <code>describe</code>, generalized and
categorical spatial lag). The members of row <code>i</code>, in fixed order, are the focal row (with
<code>includeFocal</code>, weight <code>focalWeight</code>, default 1) and then the active CSR slots in slot order. A slot
is active when its weight is positive; zero-weight slots, a listed self slot, out-of-range slots,
zero-mask neighbors and, for numeric statistics, non-finite values are skipped. Zero-weight slots are
therefore not members (<code>count</code>, <code>min</code>, <code>max</code>, <code>median</code> and the categorical outputs ignore them),
which differs from libpysal <code>lag_spatial</code> and momepy when zero weights are listed. When
<code>weights.information.transform</code> is <code>'row'</code> the slot weights are divided by the sum of the active slot
weights, so <code>sum</code> (the lag) stays a weighted average when neighbors are masked or non-finite (the
focal weight is unchanged). If <code>weights.status.overflow</code> is nonzero every output is NaN. A masked-out focal row gives <code>count</code> 0 and NaN elsewhere. With <code>k</code> members, weights <code>w</code> and
values <code>x</code>:</p>
<ul>
<li><code>count</code> is <code>k</code>, <code>weightSum</code> is <code>W</code>, <code>sum</code> is <code>sum w x</code> (the lag) and <code>mean</code> is <code>sum / W</code> (NaN when
<code>W = 0</code>).</li>
<li><code>min</code> and <code>max</code> are unweighted. <code>standardDeviation</code> is the weighted population deviation about the
mean, NaN when <code>W = 0</code>.</li>
<li><code>median</code> is unweighted (the mean of the two middle values for even <code>k</code>) for rows with at most
<code>maximumNeighbors</code> members (default 32, at most 64). Larger rows give NaN and set
<code>output.status.overflow</code>, which is required when <code>median</code> is requested.</li>
<li><code>output.modes</code> (<code>uint32</code>) is the category with the largest weighted frequency, ties to the lowest category
(<code>GPU_NEIGHBORHOOD_SUMMARY_NO_MODE</code> for none). <code>output.entropy</code> is the Shannon entropy (natural log) of the
weighted category shares. Categorical cost is <code>O(k^2)</code> per row.</li>
</ul>
<p><code>statistics</code> fixes the order of the <code>output.table</code> columns (no repeats): column <code>c</code> of row <code>i</code> is at
<code>i * statistics.length + c</code>. <code>values</code> may be <code>float32</code> or <code>uint32</code> counts (one conversion node,
exact below 2^24). The numeric and categorical parts are two kernels, each below the eight
storage buffer limit. Sums run in member order in f32, so results are deterministic.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUNeighborhoodSummary</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  weights, values, categories, includeFocal: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  statistics: [</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'count'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'mean'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'standardDeviation'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'median'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {table, modes, entropy, status: {overflow}}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span></code></pre>
<h4 id="how-it-computes-14">How it computes</h4>
<p>{/* lens:GPUNeighborhoodSummary */}</p>
<p>::lens{scenario=gpu-spatial-analysis.neighborhood-summary rung=datum bind=page datum=420 caption="Inspect tract 420: its neighbours, their weights and the weighted mean."}
::lens{scenario=gpu-spatial-analysis.neighborhood-summary rung=formula caption="Follow the neighbour mean and label entropy: weighted sums over the active slots."}
::lens{scenario=gpu-spatial-analysis.neighborhood-summary rung=kernel caption="Inspect the numeric and categorical kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.neighborhood-summary rung=graph caption="Trace queen weights, row standardisation and the neighbourhood summary kernels."}</p>
<h3 id="gpuempiricalbayesrates"><code>GPUEmpiricalBayesRates</code></h3>
<p>Shrink each tract's crude rate toward the pooled rate according to its population, returning smoothed rates, raw rates and moments.</p>
<p>::lens{scenario=gpu-spatial-analysis.empirical-bayes-rates rung=result caption="Crude and empirical-Bayes poverty rates of 791 tracts around a pooled 25.3%; small populations shrink the most."}</p>
<p>Global empirical-Bayes standardization and smoothing of rates (esda <code>assuncao_rate</code>, as used by
<code>Moran_Rate</code> and <code>Moran_Local_Rate</code> with <code>adjusted=True</code>, and <code>Empirical_Bayes</code>). Over the included rows
(nonzero <code>mask</code>, finite event count <code>e</code>, finite population <code>b > 0</code>; esda gives NaN for <code>b = 0</code>, here the
row is excluded): <code>y = e / b</code>, <code>m = sum e / sum b</code>, <code>s2 = sum b (y - m)^2 / sum b</code>, <code>a = s2 - m / mean(b)</code>.</p>
<p><code>events</code> and <code>populations</code> are <code>float32</code> or <code>uint32</code> (for example the <code>uint32</code> <code>counts</code> of
<code>addPointsInPolygonsChoroplethRecipe</code>; one conversion node, exact below 2^24). Outputs go in <code>output</code>:</p>
<ul>
<li><code>output.standardizedRates</code>: <code>z = (y - m) / sqrt(v)</code> with <code>v = a + m / b</code>, replaced by <code>m / b</code> when negative
(esda's guard).</li>
<li><code>output.smoothedRates</code>: <code>r = w y + (1 - w) m</code> with <code>w = a / (a + m / b)</code>. By default (<code>clampPriorVariance: true</code>)
<code>a</code> is clamped at 0 (Marshall 1991, GeoDa), so <code>w</code> stays in <code>[0, 1]</code> and <code>a &#x3C;= 0</code> (counts no more dispersed
than Poisson, common for small <code>n</code> or low counts) shrinks every rate fully to <code>m</code>. esda <code>Empirical_Bayes</code>
does not clamp, so with <code>a &#x3C; 0</code> its <code>w</code> leaves <code>[0, 1]</code> and rates are pushed away from <code>m</code>, possibly below
zero. Pass <code>clampPriorVariance: false</code> for exact esda parity. This is a default change.</li>
<li><code>standardizedRates</code> never clamps <code>a</code>; it keeps esda's <code>assuncao_rate</code> guard (<code>v = m / b</code> when <code>a + m / b &#x3C; 0</code>).</li>
<li>Optional <code>output.rawRates</code> and <code>output.summary</code> (<code>GPU_EMPIRICAL_BAYES_SUMMARY</code>, 8 floats: count, event sum,
population sum, pooled rate, weighted rate variance, prior variance, clamped prior variance). The
<code>priorVariance</code> slot stays unclamped (esda <code>ebi_a</code>); <code>clampedPriorVariance</code> is <code>max(a, 0)</code>.</li>
</ul>
<p>Sums are fixed-order two-level trees, so results are bitwise reproducible. Excluded rows are NaN, which
<code>GPULocalMoran</code>, <code>GPUGlobalSpatialStatistics</code> and the permutation tests exclude. Write
<code>standardizedRates</code> to a transient view and pass it as <code>values</code>. For the neighborhood-pooled variant see
<a href="#gpuspatialempiricalbayesrates"><code>GPUSpatialEmpiricalBayesRates</code></a>.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUEmpiricalBayesRates</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({events, populations, output: {standardizedRates: z}}));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">  new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUGlobalSpatialStatistics</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({weights, values: z, statistics: [</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'moran'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">], output: {summary}})</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">);</span></span></code></pre>
<h4 id="how-it-computes-15">How it computes</h4>
<p>{/* lens:GPUEmpiricalBayesRates */}</p>
<p>::lens{scenario=gpu-spatial-analysis.empirical-bayes-rates rung=datum bind=page datum=420 caption="Inspect tract 420: its crude rate, shrinkage weight and smoothed rate."}
::lens{scenario=gpu-spatial-analysis.empirical-bayes-rates rung=formula caption="Follow the shrinkage: a weighted mix of the tract rate and the pooled rate."}
::lens{scenario=gpu-spatial-analysis.empirical-bayes-rates rung=kernel caption="Inspect the partial sums, totals, variance and per-tract rates kernels."}
::lens{scenario=gpu-spatial-analysis.empirical-bayes-rates rung=graph caption="Trace the two-pass reduction into the summary and the per-tract rates."}</p>
<h3 id="gpuspatialempiricalbayesrates"><code>GPUSpatialEmpiricalBayesRates</code></h3>
<p>Pool each tract with its queen neighbours into a local prior, returning spatial rates and shrunken rates per tract.</p>
<p>::lens{scenario=gpu-spatial-analysis.spatial-empirical-bayes-rates rung=result caption="Spatial empirical-Bayes poverty rates of 791 tracts pooled over queen neighbourhoods, spanning 2.8% to 75.9%."}</p>
<p>Neighborhood-pooled rate smoothing (esda 2.10 <code>Spatial_Rate</code> and <code>Spatial_Empirical_Bayes</code>, libpysal 4.15
weights). The neighborhood of an included row <code>i</code> is <code>i</code> plus its active neighbors (slots with positive
weight to another row, the shared neighbor rule) that are included
(nonzero <code>mask</code>, finite <code>events</code>, finite <code>populations > 0</code>); only the weights' structure counts (esda sets
<code>transform = 'b'</code>; a slot stored with weight 0 is not a neighbor here). With <code>E = sum e</code>, <code>B = sum b</code> and <code>n</code> members: the spatial rate is <code>E / B</code>
(<code>Spatial_Rate.r</code>), the local prior mean is <code>m = E / B</code>, the local variance is
<code>s2 = sum b (e / b - m)^2 / B</code>, <code>a = max(s2 - m / (B / n), 0)</code>, and the smoothed rate is
<code>w e_i / b_i + (1 - w) m</code> with <code>w = a / (a + m / b_i)</code>. <code>events</code> and <code>populations</code> are <code>float32</code> or
<code>uint32</code>. Outputs are the optional <code>output.spatialRates</code>, <code>output.smoothedRates</code>, <code>output.priorMeans</code> and
<code>output.priorVariances</code> (NaN for excluded rows). One invocation reduces each row
over its ascending slots, so results are bitwise reproducible. Isolates smooth to their own rate.</p>
<p>esda 2.10.0 <code>Spatial_Empirical_Bayes</code> itself sets <code>r_var[r_var &#x3C; 0] = 0</code>, so the clamp matches esda and needs
no option (unlike the global <code>Empirical_Bayes</code>, which does not clamp; see
<a href="#gpuempiricalbayesrates"><code>GPUEmpiricalBayesRates</code></a>). The spec pins esda 2.10.0 on a 6x6 queen lattice (spatial
rate within 1e-5 relative, smoothed within 2e-4 in f32) and on low-dispersion counts where the local <code>a</code>
clamps. Deviations: a neighborhood with no events gives <code>0 / 0 = NaN</code> in esda and 0 here; rows excluded by the
mask or a non-positive population are skipped as neighbors instead of raising; a neighbor ID equal to the
focus row is skipped, where esda would count it twice.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUSpatialEmpiricalBayesRates</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({events, populations, weights, output: {smoothedRates}}));</span></span></code></pre>
<h4 id="how-it-computes-16">How it computes</h4>
<p>{/* lens:GPUSpatialEmpiricalBayesRates */}</p>
<p>::lens{scenario=gpu-spatial-analysis.spatial-empirical-bayes-rates rung=datum bind=page datum=420 caption="Inspect tract 420: its neighbourhood totals, local prior mean and smoothed rate."}
::lens{scenario=gpu-spatial-analysis.spatial-empirical-bayes-rates rung=formula caption="Follow the local shrinkage: the tract rate mixed with its neighbourhood prior mean."}
::lens{scenario=gpu-spatial-analysis.spatial-empirical-bayes-rates rung=kernel caption="Inspect 791 invocations, each summing its own and its neighbours' events and populations."}
::lens{scenario=gpu-spatial-analysis.spatial-empirical-bayes-rates rung=graph caption="Trace queen weights into the neighbourhood-pooled smoothing kernel."}</p>
<h3 id="gpuemerginghotspots"><code>GPUEmergingHotSpots</code></h3>
<p>Classify a space-time cube into ArcGIS emerging hot and cold spot categories, returning Gi* z-scores, Mann-Kendall trends and categories.</p>
<p>::lens{scenario=gpu-spatial-analysis.emerging-hot-spots rung=result caption="Emerging hot spot category of 224 cells over twelve months; all 224 are no pattern: the 35 cells that would be sporadic hot under the older rule have no significant final month."}</p>
<p>ArcGIS-style emerging hot spot analysis over a dense space-time cube, on a regular lattice or on any
spatial weights.
<code>values</code> is indexed <code>cell * sliceCount + slice</code> (the <code>GPUTemporalReduction</code> layout, <code>cell = row * gridWidth + column</code>); a <code>float32</code> view treats NaN as a missing bin, a <code>uint32</code> view (for example
<code>GPUTemporalReduction</code> <code>counts</code>) is read as <code>f32(count)</code>. An optional per-cell <code>mask</code> excludes cells.
<code>cellCount * sliceCount</code> must fit one storage binding (2^25 bins at the default 128 MiB
<code>maxStorageBufferBindingSize</code>); <code>getCommandNodes</code> throws above it.</p>
<p>Pass exactly one neighborhood: <code>gridWidth</code> and <code>gridHeight</code> for the lattice, with a per-frame
<code>radius</code> in cells, or <code>weights</code> over the cells (rows are cells) plus a compile-time <code>selfWeight</code>,
which works on H3 cells, polygons or points. With binary weights equal to the lattice disc, weights
mode reproduces lattice mode.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> hotSpots</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUEmergingHotSpots</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  values, gridWidth: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">64</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, gridHeight: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">64</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, sliceCount: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">24</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, maximumRadius: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">4</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, parameters,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {giZScores, trendZ, trendP, trendS, category, hotSliceCount, coldSliceCount}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(hotSpots);</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> settings</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {radius: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, temporalWindow: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, confidenceLevel: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0.95</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">};</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">parameterBuffer.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">write</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(hotSpots.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">getParameterValues</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(settings));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">console.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">log</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(hotSpots.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">getInferenceMetadata</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(settings));</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// {alternative: 'two-sided', permutationCount: 0, multipleTesting: 'benjamini-hochberg',</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">//  significanceLevel: 0.05, testedRowsDefinition: 'valid space-time bins'}</span></span></code></pre>
<p>Defaults changed: <code>multipleTesting</code> is <code>'benjamini-hochberg'</code> (it was no correction), <code>confidenceLevel</code> is 0.95
(it was 0.9), the trend level is <code>1 - confidenceLevel</code>, <code>minimumTrendSliceCount</code> is 10, and
<code>finalSlice</code> is <code>'require'</code>. Pass <code>multipleTesting: 'none'</code> for the previous rule.</p>
<p>Passes (deterministic, no float atomics):</p>
<ol>
<li>Global <code>n</code>, mean and standard deviation of valid bins: fixed 1024-bin block sums added in order.</li>
<li>Space-time Gi* per bin: the spatial neighbors (lattice cells with <code>dx^2 + dy^2 &#x3C;= radius^2</code>, or
the cell's weights row plus the cell itself at <code>selfWeight</code>) over the current and <code>temporalWindow</code>
previous slices, <code>z = sum w (x - X) / (S sqrt((n S1 - W^2) / (n - 1)))</code> over valid bins.</li>
<li>Two-sided normal p-values <code>erfc(|z| / sqrt(2))</code> of every valid Gi* bin, adjusted by
<code>GPUMultipleTestingAdjustment</code> over the whole family of valid bins of the cube (never a display
mask). A bin is hot or cold when its adjusted p-value is at most <code>1 - confidenceLevel</code>, with the
sign of z. With <code>multipleTesting: 'none'</code> the rule is <code>|z| >= criticalZ</code> with the same two-sided
critical values, and the p-value passes are skipped unless <code>output.pValues</code> or
<code>output.adjustedPValues</code> is requested. Bins within a temporal window overlap and neighboring cells share observations, so the
tests are dependent: Benjamini-Hochberg is only guaranteed under positive dependence, and
<code>'benjamini-yekutieli'</code> controls the false discovery rate under arbitrary dependence at a cost in
power of about <code>ln m</code>. The correction needs fewer than 2^24 bins.</li>
<li>Per-cell Mann-Kendall on the finite z series: exact integer <code>S</code>, tie-corrected variance
<code>[n(n-1)(2n+5) - sum t(t-1)(2t+5)] / 18</code> (ties by exact f32 equality), continuity-corrected z, and a
two-sided p-value from the Numerical Recipes <code>erfcc</code> fit. Consecutive Gi* values share
<code>temporalWindow</code> slices, so the series is serially correlated and this p-value is anti-conservative
(ESRI has the same property): treat the trend level as a screening threshold. Cells with fewer
than <code>minimumTrendSliceCount</code> finite slices (default 10, ESRI guidance) get NaN <code>trendZ</code> and
<code>trendP</code>.</li>
<li>Classification into the 17 <code>GPU_EMERGING_HOT_SPOTS_CATEGORIES</code> (0 = no pattern, hot 1..8, cold 9..16:
new, consecutive, intensifying, persistent, diminishing, sporadic, oscillating, historical) plus
<code>UNDETERMINED</code> (17).</li>
</ol>
<p>Sporadic follows current ArcGIS Pro: a sporadic hot (cold) spot has a statistically significant hot
(cold) final slice, with fewer than 90% hot (cold) slices and no cold (hot) slice. A cell with only
hot (or only cold) history and a non-significant final slice is therefore <code>NO_PATTERN</code>, not sporadic.
Earlier versions labelled such a cell sporadic, and the default changed. Pass
<code>sporadicRequiresFinalSlice: false</code> to the packer (a per-frame setting, no rebuild) to reproduce results
computed under the older rule; it changes only that neutral-final case.</p>
<p><code>UNDETERMINED</code> marks a cell whose category cannot be decided: with <code>finalSlice: 'require'</code> (the
default) a cell with data whose last slice is missing, and a cell that is persistent in one direction
(Intensifying, Persistent or Diminishing, hot or cold) but has fewer finite slices than
<code>minimumTrendSliceCount</code>. It is never silently reported as Persistent. <code>finalSlice: 'last-valid'</code>
keeps the old behavior, where the last finite slice stands in for the final slice, so a stale cell can
be labelled New or Persistent for a period it has no data in. Cells without any finite slice stay
<code>NO_PATTERN</code>. The optional <code>output.lastValidSlice</code> gives each cell's last finite slice index
(<code>GPU_EMERGING_HOT_SPOTS_NO_SLICE</code>, <code>0xffffffff</code>, for none).</p>
<p>Legend contract: category codes never change meaning with <code>multipleTesting</code>, <code>confidenceLevel</code> or
<code>finalSlice</code>; those options only change which bins are hot or cold and which cells are undetermined.
<code>getInferenceMetadata(settings)</code> returns a plain <code>GPUInferenceMetadata</code> for the legend, derived from the
same <code>GPUEmergingHotSpotsSettings</code> that <code>getParameterValues(settings)</code> packs (<code>significanceLevel</code> is
<code>1 - confidenceLevel</code>). <code>getParameterValues</code> takes <code>multipleTesting</code> from the node and rejects a
different <code>settings.multipleTesting</code>, so the packed values and the legend cannot disagree.</p>
<p><code>criticalZ</code> is only accepted with <code>multipleTesting: 'none'</code> (the parameter packer throws otherwise).
Without a correction <code>confidenceLevel</code> must be 0.9, 0.95 or 0.99 or <code>criticalZ</code> must be given; with a
correction any level in <code>(0, 1)</code> works.</p>
<p>Per-frame parameters (radius, temporal window, confidence level, critical z, trend significance,
persistence fraction, minimum trend length, the Sporadic rule) live in a parameter buffer and never rebuild the graph.
Compile-time: <code>gridWidth</code>, <code>gridHeight</code>, <code>sliceCount</code> (at most 256), <code>maximumRadius</code> (at most 32),
<code>multipleTesting</code>, <code>finalSlice</code>, value format, optional views. Outputs are caller-owned fields of
<code>output</code> (<code>GPUEmergingHotSpotsOutput</code>): <code>giZScores</code> (per bin), <code>trendZ</code>, <code>trendP</code>, <code>trendS</code>, <code>category</code>,
<code>hotSliceCount</code>, <code>coldSliceCount</code> (per cell), optional <code>pValues</code> and <code>adjustedPValues</code> (per bin),
optional <code>lastValidSlice</code> (per cell) and optional <code>globalStatistics</code>
<code>[n, mean, variance, standardDeviation]</code>.</p>
<h4 id="how-it-computes-17">How it computes</h4>
<p>{/* lens:GPUEmergingHotSpots */}</p>
<p>::lens{scenario=gpu-spatial-analysis.emerging-hot-spots rung=datum bind=page datum=100 caption="Inspect cell 100: its trend statistics, hot and cold month counts and category."}
::lens{scenario=gpu-spatial-analysis.emerging-hot-spots rung=formula caption="Follow space-time Gi*, the Mann-Kendall trend and the adjusted p-value rule for hot and cold months."}
::lens{scenario=gpu-spatial-analysis.emerging-hot-spots rung=kernel caption="Inspect the moment, Gi*, adjustment, Mann-Kendall and classification kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.emerging-hot-spots rung=graph caption="Trace the global moments, Gi* per bin, Benjamini-Hochberg adjustment and per-cell classification."}</p>
<p>{/* parameters: GPUEmergingHotSpots generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpuemerginghotspots-parameters"><code>GPUEmergingHotSpots</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUEmergingHotSpotsParameterValues(settings)</code>, writes <code>GPU_EMERGING_HOT_SPOTS_PARAMETER_LENGTH</code> (8) <code>float32</code> values.</li>
<li>Packer: <code>instance.getParameterValues(settings)</code>, writes the <code>float32</code> parameter view.</li>
</ul>
<p>Settings of <code>getGPUEmergingHotSpotsParameterValues</code>:</p>
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
<td><code>radius</code> (optional)</td>
<td><code>number</code></td>
<td>Lattice mode: spatial neighborhood radius in lattice cells. Cell <code>(dx, dy)</code> offsets with <code>dx * dx + dy * dy &#x3C;= radius * radius</code> are neighbors (the focal cell included, so <code>0</code> is the focal cell alone). Clamped to the compile-time <code>maximumRadius</code>. Ignored in weights mode. Defaults to <code>0</code>.</td>
</tr>
<tr>
<td><code>temporalWindow</code></td>
<td><code>number</code></td>
<td>Temporal window <code>k >= 0</code>: neighbors span the current slice and the <code>k</code> previous slices.</td>
</tr>
<tr>
<td><code>confidenceLevel</code> (optional)</td>
<td><code>number</code></td>
<td>Confidence level of hot and cold bins, in <code>(0, 1)</code>. A bin is hot or cold when its adjusted two-sided p-value is at most <code>1 - confidenceLevel</code>. With <code>multipleTesting: 'none'</code> only <code>0.9</code>, <code>0.95</code> and <code>0.99</code> (the tabulated critical z values) or an explicit <code>criticalZ</code> are accepted. Defaults to <code>0.95</code> (it was <code>0.9</code>).</td>
</tr>
<tr>
<td><code>criticalZ</code> (optional)</td>
<td><code>number</code></td>
<td>Explicit critical z that overrides <code>confidenceLevel</code> for hot and cold bins. Only meaningful without a correction: it requires <code>multipleTesting: 'none'</code> and throws otherwise, because a fixed critical z cannot express a family-wise or false discovery rate threshold.</td>
</tr>
<tr>
<td><code>multipleTesting</code> (optional)</td>
<td><code>GPUMultipleTestingMethod</code></td>
<td>The compile-time <code>multipleTesting</code> of the node these settings feed. It does not change the packed values; it makes the packer reject a <code>criticalZ</code> combined with a correction and decide whether <code>confidenceLevel</code> must be tabulated. Defaults to <code>'benjamini-hochberg'</code>, like the node. <code>GPUEmergingHotSpots.getParameterValues</code> fills it from the node and rejects a different value.</td>
</tr>
<tr>
<td><code>trendSignificanceLevel</code> (optional)</td>
<td><code>number</code></td>
<td>Two-sided p-value at or below which the Mann-Kendall trend is significant, in <code>(0, 1)</code>. Defaults to <code>1 - confidenceLevel</code>. Consecutive Gi* z values share <code>temporalWindow</code> slices, so the series is serially correlated and this p-value is anti-conservative (ESRI has the same property); treat it as a screening threshold.</td>
</tr>
<tr>
<td><code>minimumTrendSliceCount</code> (optional)</td>
<td><code>number</code></td>
<td>Cells with fewer finite slices than this get NaN <code>trendZ</code> and <code>trendP</code>, and a persistent cell that would need the trend decision gets <code>UNDETERMINED</code>. A non-negative integer; <code>0</code> disables the guard. Defaults to 10, ESRI's guidance for Mann-Kendall.</td>
</tr>
<tr>
<td><code>persistentFraction</code> (optional)</td>
<td><code>number</code></td>
<td>Fraction of slices that makes a spot persistent or historical, in <code>(0.5, 1]</code>. Default <code>0.9</code>.</td>
</tr>
<tr>
<td><code>sporadicRequiresFinalSlice</code> (optional)</td>
<td><code>boolean</code></td>
<td>Whether Sporadic hot and cold need a statistically significant hot (cold) final slice, as current ArcGIS Pro defines them. Defaults to <code>true</code>. <code>false</code> restores the older rule that also labels a cell with a non-significant final slice Sporadic when it has only hot (cold) history, so results computed before the change can be reproduced. Only that neutral-final case changes; a location that fails the new test is <code>NO_PATTERN</code>.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="addspacetimehotspotsrecipe"><code>addSpaceTimeHotSpotsRecipe</code></h3>
<p>Turn timestamped events into an emerging hot spot map, returning the month-by-cell cube, Gi* z-scores, trends and categories.</p>
<p>::lens{scenario=gpu-spatial-analysis.space-time-hot-spots-recipe rung=result caption="Hot spot categories of 224 cells from 2,344 timestamped observations over twelve months; none is sporadic hot because no cell has a significant hot final month."}</p>
<p>Timestamped events to an emerging hot spot map: <code>GPUCalendarBuckets</code> decodes the event times, an
adapter bins every event into a dense <code>cell * sliceCount + slice</code> cube, <code>GPUEmergingHotSpots</code>
classifies it and an ordinal <code>GPUColorScale</code> colors the categories (18 palette entries, 0 to 17).</p>
<p>For <code>cells.kind: 'lattice'</code>, optional <code>cells.spatialContext</code> describes positions and bounds, defaults to
native planar, and rejects longitude/latitude. <code>result.spatialContext</code> returns this resolved context;
with caller-supplied cell IDs it carries <code>cells.weights.spatialContext</code> when declared.</p>
<p>Slices are absolute time steps, the ArcGIS "time step interval", so the Mann-Kendall trend runs over
a monotone axis:</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> spaceTime</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> addSpaceTimeHotSpotsRecipe</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph, {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  timestamps, calendarParameters, cells,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  settings: {temporalWindow: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, radius: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">1.5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">},</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  parameters: emergingParameters, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// GPUParameterBuffer the recipe fills from settings</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  slices: {step: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'month'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, interval: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, origin: {year: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">2022</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, month: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">11</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}, count: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">12</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">},</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  multipleTesting: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'benjamini-yekutieli'</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">spaceTime.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">setSettings</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({temporalWindow: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, radius: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">1.5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}); </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// next frame, no rebuild</span></span></code></pre>
<p>Slice <code>floor((stepIndex(t) - stepIndex(origin)) / (interval * stepsPerUnit))</code> holds the events at
local time <code>t</code> (the same UTC-offset semantics as <code>GPUCalendarBuckets</code>). <code>stepIndex</code> counts whole
steps: <code>year</code>, <code>year * 4 + quarter - 1</code>, <code>year * 12 + month - 1</code>, days since the epoch for <code>'day'</code> and
<code>'week'</code> (a week is 7 days anchored to the origin day), hours since the epoch for <code>'hour'</code>. It is
integer arithmetic on the GPU (Hinnant's <code>days_from_civil</code>) and on the CPU
(<code>getSpaceTimeSliceIndex</code>). Events before the origin or past the last slice are skipped and counted
in the <code>slice-range</code> status stage (<code>invalidCount</code>, or the <code>outputs.status.skippedEventCount</code> override). The sorted path's
<code>cube-counts</code> stage also reports <code>requiredCount</code> (the <code>outputs.status.groupRequiredCount</code> output).</p>
<p>Optional <code>timeZone: GPUCalendarBucketsTimeZone</code> forwards a transition table from
<code>getGPUTimeZoneTransitionTable</code> to <code>GPUCalendarBuckets</code>, which searches it per row on the GPU.
It is mutually exclusive with <code>utcOffsets</code>; setting both throws. With an <code>'hour'</code> step, local
slices on transition days are 0 h or 2 h long, so use UTC for hourly cubes.</p>
<p>Cyclic fields (month of year, weekday, hour of day, day of year) are no longer accepted: <code>month = 1</code>
would pool every January of a multi-year range, and a trend over hour or weekday is meaningless.
Cyclic profiles belong in a non-trend summary such as <code>GPUCalendarBuckets</code> <code>hourWeekdayCounts</code>.</p>
<p><code>multipleTesting</code> (default <code>'benjamini-hochberg'</code>) and <code>finalSlice</code> (default <code>'require'</code>) pass through
to <code>GPUEmergingHotSpots</code>. The per-frame <code>settings</code> (<code>GPUSpaceTimeHotSpotsRecipeSettings</code>: the emerging
hot spot settings without <code>multipleTesting</code>) are packed by the recipe into its <code>parameters</code>
<code>GPUParameterBuffer</code>, and <code>result.inference</code> (a plain <code>GPUInferenceMetadata</code> object) is
<code>GPUEmergingHotSpots.getInferenceMetadata</code> of the same object; <code>result.setSettings</code> repacks both. The
result also has optional <code>pValues</code>, <code>adjustedPValues</code> and <code>lastValidSlice</code> outputs.</p>
<h4 id="how-it-computes-18">How it computes</h4>
<p>{/* lens:addSpaceTimeHotSpotsRecipe */}</p>
<p>::lens{scenario=gpu-spatial-analysis.space-time-hot-spots-recipe rung=formula caption="Follow the binning: event time to month slice, position to lattice cell, counts per cell and slice."}
::lens{scenario=gpu-spatial-analysis.space-time-hot-spots-recipe rung=kernel caption="Inspect the calendar decode, cube binning and emerging hot spot kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.space-time-hot-spots-recipe rung=graph caption="Trace calendar decoding, the atomic cube count and the emerging hot spot chain."}</p>
<h3 id="gpugeographicdistribution"><code>GPUGeographicDistribution</code></h3>
<p>Measure the weighted centre, spread and ellipse of tract centroids, returning mean centre, standard distance and ellipse per group.</p>
<p>::lens{scenario=gpu-spatial-analysis.geographic-distribution rung=result caption="Population-weighted standard distance of 791 Chicago tract centroids: 11,343 m in the local frame."}</p>
<p><code>GPUGeographicDistribution</code> computes the ArcGIS "Measuring Geographic Distributions" statistics
over planar points, per group or per masked selection: counts, weight sums, mean centre, Weiszfeld
median centre, standard distance, standard deviational ellipse, linear directional mean, and
polygon rings for the ellipse and the standard-distance circle. Every statistic is Euclidean:
<code>spatialContext</code> (default native planar) rejects <code>longitude-latitude</code> at construction with a
"project positions to a planar frame first" error, and <code>GPURoseStatistic</code> does the same. A
spherical mean centre (normalised mean of 3D unit vectors) and a great-circle standard distance are
deferred, because they need a second implementation of every reduction pass in both the chunked and
the sorted path.</p>
<p>Declares <code>measure: 'distance'</code>: a Lambert equal-area local frame throws; use an
azimuthal-equidistant frame.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> parameters</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUParameterBuffer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(device, {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  id: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'distribution-parameters'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  format: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'float32'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  length: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">GPU_GEOGRAPHIC_DISTRIBUTION_PARAMETER_LENGTH</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">  new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUGeographicDistribution</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    positions, weights, groupIds, groupCount: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">8</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, mask,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    parameters: parameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    output: {meanCenters, ellipses, ellipseVertices}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  })</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">);</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">parameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">write</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">  getGPUGeographicDistributionParameterValues</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({origin: [x0, y0], standardDeviations: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">})</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">);</span></span></code></pre>
<ul>
<li>Inputs: <code>positions</code> (<code>float32x2</code>, fewer than 2^24 rows), optional <code>weights</code> (a non-finite,
negative or zero weight excludes the row), <code>groupIds</code> with compile-time <code>groupCount</code> (ids at or
above it are excluded), <code>mask</code>, and <code>lineEnds</code> for the directional mean.</li>
<li>Per-frame parameters (no recompile): <code>origin</code> (local origin subtracted before every sum for
float32 stability), <code>standardDeviations</code>, <code>ellipseConvention</code> (<code>'root-two'</code> scales the axes by
<code>sqrt(2)</code>, <code>'standard'</code> does not), <code>orientationOnly</code>, <code>medianTolerance</code>.</li>
<li>Median centre: Weiszfeld steps from the mean centre, all groups together, for at most the
compile-time <code>maximumMedianIterations</code> (default 24). Behavior change: the steps stop after the first one
in which every non-empty group moved at most <code>medianTolerance</code> (a GPU-written indirect dispatch skips the
rest), instead of always running <code>maximumMedianIterations</code> steps; <code>medianConverged</code> is then 1 for every
non-empty group.</li>
<li>Divisors are the weight sum, as in ArcGIS (not <code>n - 1</code>). Second moments are central (a mean
pass, then a pass about the mean), never <code>E[x^2] - E[x]^2</code>. No small-sample degrees-of-freedom
correction is applied, so for small <code>n</code> the spread is biased low by <code>sqrt((n - 1) / n)</code>; there is
no confidence interval or significance test.</li>
<li><code>ellipses</code> holds <code>[angle, sigmaX, sigmaY]</code> per group. As in ArcGIS the "y" axis is the long one
(<code>sigmaY >= sigmaX</code>). <code>angle</code> is counter-clockwise from +x of the short x axis, so the long axis
points at <code>angle + pi / 2</code>; a ring vertex is <code>M + R(angle) (sigmaX cos t, sigmaY sin t)</code>.</li>
<li><code>directionalMeans</code> holds <code>[meanAngle, circularVariance, meanLength]</code> (angles counter-clockwise
from +x). With <code>orientationOnly</code>, angles are doubled before averaging and the mean angle lies in
<code>(-pi/2, pi/2]</code>.</li>
<li><code>ellipseVertices</code> and <code>circleVertices</code> are counter-clockwise rings of <code>polygonVertexCount</code>
(default 64) vertices per group, not closed, ready for a deck.gl <code>PolygonLayer</code> or <code>PathLayer</code>.</li>
<li>The median centre runs at most <code>maximumMedianIterations</code> (default 24) Weiszfeld steps from the mean
centre; <code>medianConverged</code> is 1 when the last step moved at most <code>medianTolerance</code>.</li>
<li>Empty groups report 0 counts and weight sums and NaN in every other output.</li>
<li>Determinism: rows are sorted by group once and every sum, including each Weiszfeld step, is a
fixed-order segmented tree sum. No float atomics; repeated encodings are bitwise identical.</li>
</ul>
<h4 id="how-it-computes-19">How it computes</h4>
<p>{/* lens:GPUGeographicDistribution */}</p>
<p>::lens{scenario=gpu-spatial-analysis.geographic-distribution rung=datum bind=page datum=420 caption="Inspect tract 420: its centroid and population weight enter every group sum."}
::lens{scenario=gpu-spatial-analysis.geographic-distribution rung=formula caption="Follow the standard distance and ellipse: weighted second moments about the mean centre."}
::lens{scenario=gpu-spatial-analysis.geographic-distribution rung=kernel caption="Inspect the moment passes, the spread kernel and the ellipse ring kernel."}
::lens{scenario=gpu-spatial-analysis.geographic-distribution rung=graph caption="Trace the weighted sums into the mean centre, spread and ellipse vertices."}</p>
<p>{/* parameters: GPUGeographicDistribution generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpugeographicdistribution-parameters"><code>GPUGeographicDistribution</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUGeographicDistributionParameterValues(settings?)</code>, writes <code>GPU_GEOGRAPHIC_DISTRIBUTION_PARAMETER_LENGTH</code> (8) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPUGeographicDistributionParameterValues</code>:</p>
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
<td><code>origin</code> (optional)</td>
<td><code>readonly [number, number]</code></td>
<td>Local origin <code>[x, y]</code> subtracted from every position before any sum, so float32 sums keep precision when the planar coordinates are large (for example Web Mercator meters). Choose a point near the data. Defaults to <code>[0, 0]</code>.</td>
</tr>
<tr>
<td><code>standardDeviations</code> (optional)</td>
<td><code>number</code></td>
<td>Number of standard deviations that scale <code>standardDistances</code>, <code>ellipses</code> and the vertex rings (ArcGIS offers 1, 2 and 3). Any positive number is accepted. Defaults to <code>1</code>.</td>
</tr>
<tr>
<td><code>ellipseConvention</code> (optional)</td>
<td><code>GPUGeographicDistributionEllipseConvention</code></td>
<td>Axis scaling of the ellipse: <code>'root-two'</code> scales the axes by <code>sqrt(2)</code> as the ArcGIS Directional Distribution tool does, <code>'standard'</code> uses the plain standard deviation along each principal axis. Defaults to <code>'root-two'</code>.</td>
</tr>
<tr>
<td><code>orientationOnly</code> (optional)</td>
<td><code>boolean</code></td>
<td>Treat lines as undirected (angles are doubled before averaging, so a line and its reverse agree). Only affects <code>directionalMeans</code>. Defaults to <code>false</code>.</td>
</tr>
<tr>
<td><code>medianTolerance</code> (optional)</td>
<td><code>number</code></td>
<td>Absolute tolerance, in position units, under which the last Weiszfeld step counts as converged for <code>medianConverged</code>. The iteration stops once every non-empty group's last step is at most this value. Non-negative. Defaults to <code>GPU_GEOGRAPHIC_DISTRIBUTION_DEFAULT_MEDIAN_TOLERANCE</code>.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="gpuregionstatistics"><code>GPURegionStatistics</code></h3>
<p>The compact <code>output.selectedRows.status.overflow</code> reports final selection capacity loss;
optional <code>candidateOverflow</code> reports truncated region input or grid candidate loss. Its <code>requiredCount</code> counts
selected rows reaching compaction and is a lower bound when candidates were lost.</p>
<p><code>GPURegionStatistics</code>: Select tract centroids inside a lasso and reduce them to counts, sum, mean, extremes and a histogram.</p>
<p>::lens{scenario=gpu-spatial-analysis.region-statistics rung=result caption="Tracts whose centroids fall inside a lasso: 240 of 791 Chicago tracts."}</p>
<p><code>GPUPickRegionMask</code>: Convert a region-pick result into a deduplicated 0/1 mask over the source rows.</p>
<p>::lens{scenario=gpu-spatial-analysis.pick-region-mask rung=result caption="20 tracts of community area 8, picked by a supplied region-pick result."}</p>
<p><code>GPURegionMask</code>: Test every point against a rectangle or lasso, writing one 0/1 mask value per row.</p>
<p>::lens{scenario=gpu-spatial-analysis.region-mask rung=result caption="Tracts whose centroids fall inside a rectangle: 180 of 791 Chicago tracts."}</p>
<p>Selects points inside a per-frame rectangle, circle, lasso polygon (world space or screen pixels
through a view-projection transform), pick region (<code>GPUIndexPickingTarget.addRegionPass</code> result),
or caller mask, and reduces them into selected count, value count, sum, mean, minimum, maximum, and
a fixed-bin histogram packed into one small <code>output.summary</code> buffer. Optional stable selected IDs
(<code>output.selectedRows</code>, a <code>GPUCompactOutput</code>) and a 0/1 <code>output.mask</code> stay on the GPU for
highlighting. <code>GPURegionStatisticsReadback</code> reads the summary through a
<code>GPUReadbackRing</code> without stalling. <code>GPURegionMask</code> and <code>GPUPickRegionMask</code> are exported as
standalone building blocks.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPURegionStatistics</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  region: {kind: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'polygon'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, vertices: lasso, vertexCount: lassoCount.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    screenTransform: transform.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph)},</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  positions, values, histogram: {binCount: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">16</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">},</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {summary: graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importGPUData</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'summary'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, summaryData), selectedRows}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span></code></pre>
<p>Rectangle and lasso regions are inclusive, as luma.gl <code>GPUPointSpatialQuery</code> is for bounds and
polygons: a point exactly on the boundary is selected. Lasso membership is even-odd with an exact
side test. A point is on an edge when it lies in the edge's bounding box and the exact orientation
sign is zero; otherwise a straddling edge counts when the exact orientation sign puts the point
strictly on its inner side, so no f32 crossing abscissa decides the result.</p>
<p><code>GPURegionMask</code> and <code>GPUPickRegionMask</code> write <code>output: {mask, status: {overflow}}</code>
(<code>GPURegionMaskOutput</code>); <code>GPUPickRegionMask</code> reads the packed pick result from <code>picks</code>, as does a
<code>{kind: 'pick-region', picks}</code> region.</p>
<p>Coordinates: <code>spatialContext</code> defaults to native planar. Rectangle, lasso and pick regions are
topological tests and also accept <code>longitude-latitude</code> (metric <code>'none'</code>): edges are straight lines
in degree space and require linear edges. Prepare polygons that cross ±180 or enclose a pole with
<code>GPULonLatPolygonPreparation</code>. Optional <code>output.status.antimeridianUncertainCount</code> and
<code>output.status.poleUncertainCount</code> report unprepared rings in <code>GPURegionMask</code> and
<code>GPURegionStatistics</code>. These one-row <code>uint32</code> counters are cleared and rewritten every encoding
for world-space lon/lat lassos; planar contexts, screen-space lassos and other region kinds leave
them untouched. A lasso consumes one ring at a time, including an empty prepared copy through a
zero active vertex count. A <code>'radius'</code> region is a planar distance in
position units, so it throws at construction in <code>longitude-latitude</code> (the region kind is
compile-time); project positions to a planar frame first, or use a rectangle or lasso. Geographic
metrics are rejected. Radius selections measure distance and reject <code>local-lambert-equal-area</code> frames. <code>GPURegionMask</code> takes the same <code>spatialContext</code>; <code>GPUPickRegionMask</code> reads no
positions and has none.</p>
<p>An optional <code>spatialIndex: {kind: 'grid', index, candidateCapacity}</code> takes a caller-built
<code>GPUGridIndex</code> over the same positions and reduces only candidates from the cells a world-space
rectangle, circle, or lasso touches. Results match the full scan except for float reassociation in
<code>sum</code> and <code>mean</code>; the <code>candidatesTruncated</code> flag reports capacity overflow or rows missing from the
index.</p>
<p><strong>Summary words.</strong> <code>output.summary</code> holds <code>8 + binCount</code> <code>uint32</code> words (<code>GPU_REGION_STATISTICS_SUMMARY_LAYOUT</code>):
0 <code>selectedCount</code>, 1 <code>valueCount</code>, 2 <code>sum</code> (f32 bits), 3 <code>mean</code> (f32 bits), 4 <code>minimum</code> (f32 bits), 5 <code>maximum</code>
(f32 bits), 6 <code>histogramOutsideCount</code>, 7 <code>flags</code>, then the bins. <code>flags</code> bits (<code>GPU_REGION_STATISTICS_FLAGS</code>):
1 <code>selectionTruncated</code> (<code>output.selectedRows.ids</code> too small), 2 <code>regionTruncated</code> (lasso vertex count above
capacity, or pick-region overflow), 4 <code>candidatesTruncated</code> (grid candidate stage incomplete); other bits are 0.
<code>decodeGPURegionStatistics</code> and <code>GPURegionStatisticsReadback</code> decode the words. The same flags are also available
as named one-row status scalars beside the header, which keeps its words: optional <code>output.status.overflow</code> is 1
when bit 1 or 2 is set and <code>output.status.candidateOverflow</code> is 1 when bit 4 is set (it is not folded into
<code>overflow</code>). Optional <code>output.selectedRows.status.candidateOverflow</code> reports bits 2 or 4 separately
from <code>output.selectedRows.status.overflow</code>; <code>selectedRows.status.requiredCount</code> counts selected rows
reaching final compaction and is a lower bound when the region input or grid candidates were truncated.</p>
<p>An optional <code>output.drawInstanceCount</code> (one-row <code>uint32</code> view, for example the instance-count word
of an indirect draw record) receives the same capacity-clamped selected count as
<code>output.selectedRows.status.count</code> on every encoding, so selected rows can be drawn instanced through
<code>output.selectedRows.ids</code> without a readback. It requires <code>output.selectedRows</code>; statistics-only
graphs read the count from <code>output.summary</code>.</p>
<h4 id="how-it-computes-20">How it computes</h4>
<p>{/* lens:GPURegionStatistics <em>/}
{/</em> lens:GPUPickRegionMask <em>/}
{/</em> lens:GPURegionMask */}</p>
<p><code>GPURegionStatistics</code>:</p>
<p>::lens{scenario=gpu-spatial-analysis.region-statistics rung=datum bind=page datum=420 caption="Inspect tract 420: its centroid and value, and whether the lasso selects it."}
::lens{scenario=gpu-spatial-analysis.region-statistics rung=formula caption="Follow the summary: selected finite values give the count, sum and mean."}
::lens{scenario=gpu-spatial-analysis.region-statistics rung=kernel caption="Inspect the lasso mask, count, sum, extent and summary kernels."}
::lens{scenario=gpu-spatial-analysis.region-statistics rung=graph caption="Trace the mask, count and reduction nodes into the summary buffer."}</p>
<p><code>GPUPickRegionMask</code>:</p>
<p>::lens{scenario=gpu-spatial-analysis.pick-region-mask rung=datum bind=page datum=420 caption="Inspect tract 420: which pick pairs name it and the mask value they set."}
::lens{scenario=gpu-spatial-analysis.pick-region-mask rung=formula caption="Follow the rule: a row is 1 when a pick pair of the chosen batch names it."}
::lens{scenario=gpu-spatial-analysis.pick-region-mask rung=kernel caption="Inspect the clear, scatter and overflow kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.pick-region-mask rung=graph caption="Trace the mask clear, the pick scatter and the overflow copy."}</p>
<p><code>GPURegionMask</code>:</p>
<p>::lens{scenario=gpu-spatial-analysis.region-mask rung=datum bind=page datum=420 caption="Inspect tract 420: its centroid against the rectangle bounds."}
::lens{scenario=gpu-spatial-analysis.region-mask rung=formula caption="Follow the rule: a point is 1 inside the inclusive, valid rectangle on both axes."}
::lens{scenario=gpu-spatial-analysis.region-mask rung=kernel caption="Inspect 791 point tests with the retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.region-mask rung=graph caption="Trace the overflow flag and the single region mask kernel."}</p>
<h3 id="gpuzonalstatistics"><code>GPUZonalStatistics</code></h3>
<p>Reduce tract values to one count, sum, mean and deviation per community area, using per-row zone membership.</p>
<p>::lens{scenario=gpu-spatial-analysis.zonal-statistics rung=result caption="People below the poverty line summed over 77 community areas from 791 tracts; total 688,314."}</p>
<p>Choropleth aggregation. Each point is joined to the polygon feature that contains it (an internal
<code>GPUPointInPolygonJoin</code>), or the caller supplies per-point feature rows. The points are then
reduced per feature to count, valid-value count, sum, weighted sum and weight sum, mean, minimum,
maximum, density (count divided by a caller area or a GPU shoelace area), and a feature-value
<code>[min, max]</code> extent for color scales.</p>
<p>Boundary points follow <code>boundaryRule</code> (passed to the join). The default <code>'half-open'</code> counts a point
on an edge or vertex shared by coverage features exactly once, whatever the feature row order; points
on the outer right or top edge of the coverage are counted by no feature. <code>'closed'</code> counts every
boundary point (smallest containing row wins) and <code>'open'</code> only strict interiors.</p>
<p>Polygon features use the four-level <code>GPUPolygonGeometryPort</code> layout (<code>positions</code>, <code>featureOffsets</code>,
<code>polygonOffsets</code>, <code>ringOffsets</code>; one output row per feature, multipart features included) with the
join options; their <code>sourceIds</code> are not read. Point positions are the top-level <code>positions</code>.</p>
<p>The output views you pass select the statistics at compile time. Empty features give a sum of 0 and
NaN for the mean, minimum and maximum. <code>sumOrder: 'sorted'</code> stably sorts points by feature and
reduces each segment in a fixed tree order, so sums are bitwise reproducible at a bounded cost. It is
the default when every point-rate input (positions, values, weights, <code>pointFeatureRows</code>) is packed.
Chunked inputs fall back to <code>sumOrder: 'atomic'</code>, whose float atomics can vary in the last bits
between runs and serialize when many points share a feature.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUZonalStatistics</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  features: {kind: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'polygons'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, positions: polygonPositions, featureOffsets, polygonOffsets,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    ringOffsets, candidateCapacity: positions.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">length</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> *</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">},</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  positions, values: incomes, weights: households,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {counts, means, densities, extent, extentStatistic: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'mean'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, status: {overflow}}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span></code></pre>
<p><code>output.status.uncertainCount</code> (one <code>uint32</code> row) reports the join's uncertain candidate pairs. A nonzero
value means some points could not be classified and the per-feature statistics may be low. It is
always 0 for caller-supplied feature rows.</p>
<p>Optional <code>output.status.candidateOverflow</code> is 1 when the join's candidate capacity or BVH leaf capacity was too
small, and optional <code>output.candidateTruncated</code> (polygon features only; one packed <code>uint32</code> per point) flags the
points whose membership may be incomplete: the points whose candidates were dropped, or every point after a BVH
leaf overflow. Both are the <code>GPUPointInPolygonJoin</code> outputs passed through; <code>candidateOverflow</code> is always 0 for
caller-supplied feature rows.</p>
<ul>
<li>Variance. <code>output.variances</code> and <code>output.standardDeviations</code> are the unbiased estimates in two
passes (mean first, then <code>sum w (v - m)^2</code>), so there is no <code>E[v^2] - E[v]^2</code> cancellation. Unweighted it
is <code>ddof = 1</code> (<code>numpy.var(v, ddof=1)</code>). With <code>weights</code>, which act as reliability weights, it is
<code>sum w (v - m)^2 / (V1 - V2 / V1)</code> with <code>V1 = sum w</code>, <code>V2 = sum w^2</code>, equal to
<code>numpy.cov(v, aweights=w, ddof=1)</code>. A feature with fewer than two valid rows, or a non-positive
denominator, gives NaN, never 0. Variances need packed point views (not chunked vectors) and follow
<code>sumOrder</code>. Checked against numpy in <code>zonal_variance.py</code>.</li>
<li>Weighted density. <code>output.weightedDensities</code> is <code>weightSums / area</code> (needs <code>weights</code>); the area rules
of <code>densities</code> apply (finite and positive, else NaN).</li>
<li>Coordinates. <code>spatialContext</code> accepts planar (default) or <code>longitude-latitude</code> with metric <code>'none'</code>.
Membership runs <code>GPUPointInPolygonJoin</code> with a containment-only context derived from <code>spatialContext</code>
(metric <code>'none'</code>), so in degrees it is topological with straight edges (not geodesics) and polygons
require linear edges. Prepare polygons that cross ±180 or enclose a pole with
<code>GPULonLatPolygonPreparation</code>; optional one-row <code>uint32</code>
<code>output.status.antimeridianUncertainCount</code> / <code>output.status.poleUncertainCount</code> report unprepared
rings and are cleared every encoding. Planar contexts and caller-supplied feature rows leave
those counters untouched. The shoelace area of degrees is
wrong, so a longitude-latitude context rejects GPU areas (<code>densities</code>, <code>weightedDensities</code>,
<code>featureAreas</code> without <code>areas</code>) at construction: pass spherical <code>areas</code> in square meters, for example
from <code>GPUGeometryMeasures</code>. GPU-computed planar areas and densities measure area and reject
<code>local-azimuthal-equidistant</code> frames; supplied <code>areas</code> and means-only configurations do not trigger
this area-frame restriction. Counts, sums, means, extrema and variances need no area.</li>
<li>Not provided: median and quantiles (they need a per-feature sort of the values), weighted minima and
maxima, fractional polygon coverage (a point belongs to a feature or it does not).</li>
</ul>
<h4 id="how-it-computes-21">How it computes</h4>
<p>{/* lens:GPUZonalStatistics */}</p>
<p>::lens{scenario=gpu-spatial-analysis.zonal-statistics rung=datum bind=page datum=420 caption="Inspect tract 420: its zone row, value and share of the zone sum."}
::lens{scenario=gpu-spatial-analysis.zonal-statistics rung=formula caption="Follow the zone sum, mean and deviation: finite values grouped by zone row."}
::lens{scenario=gpu-spatial-analysis.zonal-statistics rung=kernel caption="Inspect the sorted sums and the mean and deviation kernels."}
::lens{scenario=gpu-spatial-analysis.zonal-statistics rung=graph caption="Trace the stable sort by zone, segmented sums, means and deviations."}</p>
<h3 id="gpufocalstatistics"><code>GPUFocalStatistics</code></h3>
<p>Summarise each raster cell's moving window, writing the window mean, deviation and valid-cell count per cell.</p>
<p>::lens{scenario=gpu-spatial-analysis.focal-statistics rung=result caption="Window mean elevation (radius 3) on a 120 m terrain lattice; means span 1,083 to 1,531 m."}</p>
<p>Moving-window statistics over a float32 raster, like ArcGIS <code>FocalStatistics</code> and GRASS
<code>r.neighbors</code>: mean, sum, min, max, range, population standard deviation, and the count of valid
cells. The radius, window shape, minimum count, and center nodata policy are per-frame.
A cell is nodata when it is not finite (NaN or an infinity), equals the optional <code>noDataValue</code>, or
has a zero <code>validity</code> word; nodata cells are left out of every window. (Before 2026-11-10 an
infinity was kept as a value, which gave an infinite mean and a NaN standard deviation.)
The operator is raster-only: it reads a lattice of values and never positions, so it takes no
<code>spatialContext</code>, its <code>grid</code> (<code>GPURasterMetadata</code>) only sizes the raster, and its radius is in
cells, not ground units. Cell size and shape (for example
lon/lat cells that shrink with latitude) are the caller's concern.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">  new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUFocalStatistics</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    values: raster, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// row-major float32, NaN = nodata</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    validity, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// optional uint32, 0 = nodata</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    noDataValue: </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">-</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">9999</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// optional sentinel</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    grid, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// GPURasterMetadata: width * height cells, any affine</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    maximumRadius: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">16</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// compile-time cap, at most 64</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    parameters: focalParameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph), </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// 4 float32</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    output: {mean, standardDeviation, min, max, counts} </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// any non-empty subset</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  })</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">);</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">focalParameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">write</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">  getGPUFocalStatisticsParameterValues</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({radius: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, shape: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'circle'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, minimumCount: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">})</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">); </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// no recompile</span></span></code></pre>
<ul>
<li>Square windows cover <code>|dx|, |dy| &#x3C;= floor(radius)</code>; circles keep offsets with
<code>dx^2 + dy^2 &#x3C;= radius^2</code>. The radius is clamped to <code>maximumRadius</code>.</li>
<li>Window cells outside the raster and nodata cells are skipped, so edge windows are clipped.
A cell is nodata (NaN) with fewer than <code>max(minimumCount, 1)</code> valid cells, or when its own value
is nodata and <code>propagateCenterNoData</code> is set. <code>count</code> is written for every cell.</li>
<li>Direct gather in fixed row-major window order: deterministic, no atomics. Standard deviation
subtracts the window mean in a second pass. Moments and extremes are separate kernels, each
scheduled only when one of its outputs is requested.</li>
<li>Cost is <code>(2r + 1)^2</code> reads per cell; see Status and limitations for the still-open separable and
summed-area variants.</li>
</ul>
<h4 id="how-it-computes-22">How it computes</h4>
<p>{/* lens:GPUFocalStatistics */}</p>
<p>::lens{scenario=gpu-spatial-analysis.focal-statistics rung=datum bind=page datum=528 caption="Inspect cell 528: its window, mean, deviation and valid-cell count."}
::lens{scenario=gpu-spatial-analysis.focal-statistics rung=formula caption="Follow the window mean and population deviation over the valid cells of each window."}
::lens{scenario=gpu-spatial-analysis.focal-statistics rung=kernel caption="Inspect 1,024 gather invocations reading each window in row-major order."}
::lens{scenario=gpu-spatial-analysis.focal-statistics rung=graph caption="Trace the single moments kernel from the elevation raster to the window statistics."}</p>
<p>{/* parameters: GPUFocalStatistics generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpufocalstatistics-parameters"><code>GPUFocalStatistics</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUFocalStatisticsParameterValues(settings)</code>, writes <code>GPU_FOCAL_STATISTICS_PARAMETER_LENGTH</code> (4) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPUFocalStatisticsParameterValues</code>:</p>
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
<td><code>radius</code></td>
<td><code>number</code></td>
<td>Window radius in cells. A square window covers <code>|dx|, |dy| &#x3C;= floor(radius)</code>; a circle covers offsets with <code>dx^2 + dy^2 &#x3C;= radius^2</code> inside that square. Clamped to the contributor's compile-time <code>maximumRadius</code>.</td>
</tr>
<tr>
<td><code>shape</code> (optional)</td>
<td><code>GPUFocalStatisticsShape</code></td>
<td>Window shape. Defaults to <code>'square'</code>.</td>
</tr>
<tr>
<td><code>minimumCount</code> (optional)</td>
<td><code>number</code></td>
<td>Cells whose window has fewer valid cells are nodata. Defaults to 1.</td>
</tr>
<tr>
<td><code>propagateCenterNoData</code> (optional)</td>
<td><code>boolean</code></td>
<td>When true, a nodata center cell yields nodata statistics. Defaults to false.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="gpusimilarlocations"><code>GPUSimilarLocations</code></h3>
<p>Rank rows by weighted distance in standardized attribute space to a reference, returning ranks, distances and the best row ids.</p>
<p>::lens{scenario=gpu-spatial-analysis.similar-locations rung=result caption="Distance of 136 north-side tracts to one reference tract in five indicators; nearest 0.44, reference blank."}</p>
<p>Maximum: <code>columnCount</code> is at most <code>GPU_SIMILAR_LOCATIONS_MAXIMUM_COLUMN_COUNT</code> (one 256-lane reduction per column within a uint32 dispatch) and the value table at most <code>getGPUSimilarLocationsMaximumValueCount(device)</code> (<code>2^25</code> on a default device) float32 values.</p>
<p>Ranks rows by weighted Euclidean distance, in standardized attribute space, to a reference row or the mean
of several selected rows (the ArcGIS Similarity Search task; checked against NumPy, SciPy and scikit-learn distances). Props: <code>values</code> (row-major <code>rowCount * columnCount</code>), <code>columnCount</code>,
<code>referenceMask</code> (a <code>uint32</code> mask of the reference rows), optional <code>mask</code>, per-frame <code>parameters</code>
(<code>getGPUSimilarLocationsParameterValues</code>: result count, most or least similar, reference exclusion and
attribute weights), <code>standardization</code> (<code>'zscore'</code>, population, or <code>'rank'</code>, percentile; compile-time) and
<code>resultCapacity</code> (compile-time size of <code>output.topIds</code>).</p>
<ul>
<li>Outputs: <code>ranks</code> (0 is the most similar; unranked rows hold <code>GPU_SIMILAR_LOCATIONS_NO_RANK</code>),
<code>distances</code> (NaN when unranked), <code>topIds</code> (the best rows in rank order) and <code>status.count</code>.</li>
<li>Rows with a non-finite attribute or a zero mask are never ranked. A stable <code>GPUSort</code> on the distance bits
keeps ties on the lowest ID.</li>
<li><code>'rank'</code> standardization uses a quadratic scan, so it suits tens of thousands of rows, not millions.</li>
</ul>
<h4 id="how-it-computes-23">How it computes</h4>
<p>{/* lens:GPUSimilarLocations */}</p>
<p>::lens{scenario=gpu-spatial-analysis.similar-locations rung=datum bind=page datum=60 caption="Inspect tract 60: its distance to the reference tract and its similarity rank."}
::lens{scenario=gpu-spatial-analysis.similar-locations rung=formula caption="Follow the z-scores, the reference vector and the weighted Euclidean distance to it."}
::lens{scenario=gpu-spatial-analysis.similar-locations rung=kernel caption="Inspect the column statistics, standardize, reference, distance, sort and rank kernels with retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.similar-locations rung=graph caption="Trace validation, z-scores, the reference mean, distances and the stable sort into ranks."}</p>
<p>{/* parameters: GPUSimilarLocations generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpusimilarlocations-parameters"><code>GPUSimilarLocations</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUSimilarLocationsParameterValues(settings, columnCount)</code>, writes <code>getGPUSimilarLocationsParameterLength(...)</code> <code>float32</code> values.
<ul>
<li><code>columnCount</code> (<code>number</code>): Number of attributes of the contributor.</li>
</ul>
</li>
</ul>
<p>Settings of <code>getGPUSimilarLocationsParameterValues</code>:</p>
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
<td><code>resultCount</code></td>
<td><code>number</code></td>
<td>Number of top-ranked rows reported in <code>output.topIds</code>, clamped to the compile-time capacity.</td>
</tr>
<tr>
<td><code>direction</code> (optional)</td>
<td><code>'most' | 'least'</code></td>
<td><code>'most'</code> ranks the nearest rows first; <code>'least'</code> ranks the farthest first. Defaults to <code>'most'</code>.</td>
</tr>
<tr>
<td><code>excludeReference</code> (optional)</td>
<td><code>boolean</code></td>
<td>Excludes the reference (selected) rows from the ranking. Defaults to true.</td>
</tr>
<tr>
<td><code>weights</code> (optional)</td>
<td><code>ArrayLike&#x3C;number></code></td>
<td>Non-negative weight per attribute. Defaults to 1 for every attribute.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="gpulocaloutlierfactor"><code>GPULocalOutlierFactor</code></h3>
<p>Score each point's local density against its neighbours' over a kNN table, returning k-distance, density, LOF and an outlier mask.</p>
<p>::lens{scenario=gpu-spatial-analysis.local-outlier-factor rung=result caption="Local outlier factor of 2,344 observations from 10 neighbours; 75 exceed 3 and mark sparse points."}</p>
<p>Local Outlier Factor (Breunig 2000; geo <code>OutlierDetection</code>, scikit-learn <code>LocalOutlierFactor</code>, Sedona
<code>ST_LocalOutlierFactor</code>) over a kNN table. <code>neighbors</code> is a <code>GPUSpatialWeights</code> with <code>distances</code>, typically
<code>GPUNeighborSearch</code> in <code>'knn'</code> mode as a self join; the weight values are unused. Per row it writes
<code>kDistance</code> (the largest neighbor distance), <code>localReachabilityDensity = 1 / (mean max(kDistance_j, d_ij) + densityFloor)</code> and <code>lof = mean(lrd_j) / lrd_i</code>, which equals <code>-negative_outlier_factor_</code>, all in the <code>output</code> bag. Optional
outputs are an <code>outlier</code> mask (<code>lof > threshold</code>) and a one-row <code>outlierCount</code> (integer atomics, deterministic).
<code>parameters</code> (<code>getGPULocalOutlierFactorParameterValues({threshold, densityFloor = 1e-10})</code>) change per
frame without recompiling. There is no default <code>threshold</code>: LOF has no null distribution, so no cut is
calibrated (the former hidden 1.5 is gone). Omit it and the mask and count are all 0 while <code>lof</code> is
unaffected; otherwise pick a cut from the score distribution. LOF is exploratory, not inferential: no
p-value, nothing "significant". Choose <code>k</code> (the neighbor count of the table) of at least 10 and larger than
the smallest group that should count as a cluster rather than as outliers (Breunig 2000), and compare
scores only at the same <code>k</code>. The input must be a true kNN list; with ties at the k-th distance a list cut
at exactly <code>k</code> entries changes the k-distance neighborhood (as in scikit-learn). The class reads only
distances; <code>neighbors.spatialContext</code> describes their length unit, which also applies to <code>densityFloor</code>.
Optional <code>spatialContext</code> declares the floor's context and is checked against <code>neighbors.spatialContext</code>
with <code>assertCompatibleGPUSpatialContexts</code> when both are supplied. LOF itself is a unit-free ratio. The floor keeps coincident clusters finite at density <code>1 / densityFloor</code>, and
points whose neighbors are such duplicates score enormous values, as in scikit-learn, so use a <code>k</code> above
the duplicate count or a larger floor. A row with no neighbors gets kDistance 0, density 0 and lof 1 and is
never flagged. A radius-mode table works, but the k-distance is then the largest in-band distance. Against
a pinned scikit-learn run (24 points with <code>k = 5</code>, and a duplicate case) kDistance was within 2e-6 relative
and lof within 1e-4, with the mask and count exact.</p>
<h4 id="how-it-computes-24">How it computes</h4>
<p>{/* lens:GPULocalOutlierFactor */}</p>
<p>::lens{scenario=gpu-spatial-analysis.local-outlier-factor rung=datum bind=page datum=1000 caption="Inspect observation 1000: its k-distance, density, LOF and outlier flag."}
::lens{scenario=gpu-spatial-analysis.local-outlier-factor rung=formula caption="Follow the reachability distance, local density and the ratio of neighbour density to its own."}
::lens{scenario=gpu-spatial-analysis.local-outlier-factor rung=kernel caption="Inspect the k-distance, density, score and outlier-mask kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.local-outlier-factor rung=graph caption="Trace the ten-nearest-neighbour search into k-distance, density, LOF and the thresholded mask."}</p>
<p>See <a href="/luma-contributors/docs/api/gpu-spatial-analysis/statistical-expansion">local variability and regression extensions</a> for the expanded
statistics, fixed, finite-grid or golden-section-selected multiscale fits, generalized local models, pooled panels and regimes.</p>
<p>{/* parameters: GPULocalOutlierFactor generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpulocaloutlierfactor-parameters"><code>GPULocalOutlierFactor</code> parameters</h4>
<ul>
<li>Packer: <code>getGPULocalOutlierFactorParameterValues(settings?)</code>, writes <code>GPU_LOCAL_OUTLIER_FACTOR_PARAMETER_LENGTH</code> (4) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPULocalOutlierFactorParameterValues</code>:</p>
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
<td><code>threshold</code> (optional)</td>
<td><code>number</code></td>
<td>Rows with <code>lof > threshold</code> are flagged in the <code>outlier</code> mask and counted. There is no default: LOF has no null distribution, so no cut is calibrated (Breunig et al. 2000 give none; values near 1 are as dense as their neighbors, well above 1 locally sparse, and the scale depends on <code>k</code> and the data). Choose it from the score distribution, for example a top fraction. When omitted the packed threshold is NaN, no row is flagged, the <code>outlier</code> mask is all 0 and <code>outlierCount</code> is 0; the <code>lof</code> scores are unaffected. The flag is exploratory, not a significance statement.</td>
</tr>
<tr>
<td><code>densityFloor</code> (optional)</td>
<td><code>number</code></td>
<td>Added to each row's mean reachability distance before inverting it, so a cluster of coincident points (mean reachability 0) gets a large finite density <code>1 / densityFloor</code> instead of infinity. Must be positive and finite. Defaults to scikit-learn's <code>1e-10</code>. Raise it (for example to the coordinate resolution) to make duplicate-heavy data less extreme. Its unit is the length unit of <code>neighbors.spatialContext</code>; declare the context of this value through <code>GPULocalOutlierFactorProps.spatialContext</code> to check compatibility.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>`};export{e as default};