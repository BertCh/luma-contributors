var e={slug:`api/gpu-spatial-analysis/quantile-lisa`,title:`GPUQuantileLISA`,html:`<h1 id="gpuquantilelisa">GPUQuantileLISA</h1>
<p>{/* lens:GPUQuantileLISA */}</p>
<p>::lens{scenario=gpu-spatial-analysis.quantile-lisa rung=graph}</p>
<p><code>GPUQuantileLISA</code> is a spatial-autocorrelation contributor exported from
<code>@luma-contributors/analysis/gpu-spatial-analysis</code>; the recipes barrel retains its compatibility
export. It composes exact resident column quantiles, a selected-class indicator and
conditional local join-count permutations for one or more scalar variables. Add it to a <code>GPUCommandGraph</code>
and submit the compiled graph explicitly. It never submits work, reads buffers or transfers
ownership of inputs.</p>
<p>Supply packed <code>float32</code> <code>values</code>, optional packed <code>uint32</code> <code>mask</code>, square CSR spatial <code>weights</code>,
a two-word resident <code>uint32</code> <code>quantileParameters</code> view <code>[k, q]</code>, the existing resident permutation
<code>parameters</code> view (packed with <code>getGPUPermutationParameterValues</code>), and caller-owned outputs.
For multivariate analysis, add <code>additionalVariables: [{values, quantileParameters}, ...]</code>.
The primary pair comes first; every variable has the same row count and its own <code>[k, q]</code> controls.
The common <code>mask</code> applies to every variable. Allocate <code>boundaries</code> with <code>2 * variableCount</code> words
and <code>validCount</code> with <code>variableCount</code> words; all other outputs remain source-sized or scalar.
For example, <code>additionalVariables: [{values: income, quantileParameters: incomeQuantiles}]</code>
intersects the primary variable's selected class with income's independently selected class.</p>
<p><code>k</code> is the number of quantiles, <code>q</code> the selected class (one-based); <code>2 &#x3C;= k &#x3C; values.length</code> and
<code>1 &#x3C;= q &#x3C;= k</code>. Values, weights, masks, quantile settings, seed, permutation count and significance
level can change between submissions without rebuilding. <code>maximumPermutations</code>,
<code>maximumNeighbors</code> (default 32, at most 64) and view layouts are graph topology.</p>
<h2 id="quantiles-and-ties">Quantiles and ties</h2>
<p>For <code>n</code> finite, unmasked values sorted as <code>x</code>, a boundary at probability <code>p</code> interpolates at
<code>h = clamp(n*p - 0.5, 0, n-1)</code> (Hyndman–Fan type 5). A row belongs when
<code>lower &#x3C;= value &#x3C; upper</code>. Exterior bounds are negative and positive infinity. A tie exactly at
an interior boundary belongs to the higher class. Ties can make a selected class empty; a
constant column belongs entirely to the final class. These rules follow the pinned
<a href="https://github.com/GeoDaCenter/libgeoda/blob/71238cb58979032048081434ce6e36a53e254280/gda_sa.cpp#L233">libgeoda gda_quantilelisa</a>
and <a href="https://github.com/GeoDaCenter/libgeoda/blob/71238cb58979032048081434ce6e36a53e254280/GenUtils.cpp#L2080">QuantileBreaks and percentile</a>.
The desktop GeoDa dialog has a different upper-inclusive tie rule; this API follows libgeoda.</p>
<p>The resident selector uses type-7 quantiles after GPU conversion of type-5 ranks to probabilities.
Arithmetic and outputs are float32, so interpolated boundaries can differ from GeoDa's float64
arithmetic by rounding. Values arbitrarily close to a rounded boundary can change membership.
At most 2^24 - 1 packed source rows and 64 variables are supported. The shared eight-column
inference matrix needs <code>32 * values.length</code> bytes in one storage binding (at most 4,194,304 rows
on a device with a 128 MiB binding limit). <code>estimatedCommandNodeCount</code> bounds graph compilation
cost and is checked against the shared command-node budget at construction. Chunked inputs
require explicit caller packing.</p>
<h2 id="missing-values-islands-and-inference">Missing values, islands and inference</h2>
<p>NaN, infinities and zero-masked rows are omitted independently from each variable before
quantiling. Joint membership is the intersection of all selected-class indicators. Inference
includes only rows finite in every variable and retained by the common mask. This does not
quantile the complete-case subset: a finite value remains in its own marginal quantile
distribution even when another variable is missing in that row. The pinned
upstream <code>QuantileBreaks</code> implementation ignores its undefined mask when computing breaks;
this API deliberately corrects that defect. <code>validCount</code> counts finite unmasked rows per variable before
island exclusion. The quantile distribution includes islands; inference excludes them using the
shared noncascading island policy. Only finite positive-weight, nonself slots count as neighbors;
join counts treat each such slot as a binary edge, regardless of the weight magnitude.</p>
<p>The observed statistic is <code>membership[i] * sum(membership[j])</code>. Only rows with a nonzero
observed count are tested. Conditional permutations draw distinct analysis rows without the
focus row, preserving its neighbor count. The shared Philox stream replays exactly for the same
seed and inputs. It draws uniformly without replacement; pinned GeoDa's complete mode rounds
<code>hash * (n-1)</code> to an integer, giving endpoints half the sampling mass of interior rows. The GPU
deliberately uses the shared uniform sampler and does not reproduce that biased Thomas Wang stream.</p>
<p>The pseudo p-value follows pinned GeoDa's legacy directed rule:
<code>(min(upperExceedances, P - upperExceedances) + 1) / (P + 1)</code>, including equality in the upper tail.
This legacy rule is anti-conservative; it does not provide a two-sided error-control guarantee.
A constant final class illustrates the legacy defect: all draws tie, producing <code>1/(P+1)</code> rather than 1. No multiple-testing adjustment is applied. Untested rows have NaN p-values and the shared
<code>GPU_LOCAL_PERMUTATION_NOT_TESTED</code> exceedance sentinel. Empty selected classes are valid results.</p>
<h2 id="outputs-and-completeness">Outputs and completeness</h2>
<ul>
<li><code>membership</code>: joint selected-class binary indicator, zero when any variable is missing.</li>
<li><code>boundaries</code>: <code>[lower, upper]</code> pairs in primary-then-additional variable order, including exterior infinities.</li>
<li><code>validCount</code>: source count per variable in the same order.</li>
<li><code>observed</code>, <code>significant</code>: optional shared inference outputs.</li>
<li><code>exceedances</code>, <code>pseudoPValues</code>: required shared inference outputs.</li>
<li><code>classes</code>: <code>GPU_QUANTILE_LISA_CLASS</code>: 0 not significant, 1 significant, 2 undefined/masked,
3 isolated. Unlike upstream <code>GetClusterIndicators</code>, which collapses exceptional rows to zero,
these retain the label distinctions declared by its join-count class.</li>
<li><code>invalidQuantile</code>: one when any variable's resident k/q is invalid; membership is zero and no row is tested.</li>
<li><code>status.overflow</code>: shared inference invalid-count, neighbor-capacity or upstream weights-overflow
flag. Optional <code>status.invalidCount</code> is forwarded by shared inference (zero for these binary
inputs). Read <code>invalidQuantile</code> and <code>status.overflow</code> before presenting results. A capacity
failure is not an isolated row.</li>
</ul>
<p>Every terminal output must be a distinct packed caller-owned view to permit readback. Internal
scratch is graph-owned. Inputs and outputs must not alias. Destruction follows normal graph and
buffer ownership; the contributor does not destroy borrowed inputs.</p>
<h2 id="scope-and-evidence">Scope and evidence</h2>
<p>There is no existing CPU twin. Multivariate output implements the selected-quantile intersection
and joint local join count. An empty joint class yields zero membership and observed counts on
analysis rows, untested p-values and no significant classes. Pinned libgeoda's <code>MultiJoinCount</code>
has an additional bivariate no-colocation fallback when the intersection is empty; this API keeps
the explicit intersection definition and does not switch statistics. Missing-mask handling follows
the per-variable omission and joint validity rule above.</p>
<p>The independent NumPy fixture generator
<code>modules/analysis/test/oracles/multivariate-quantile-lisa-numpy.py</code> uses <code>numpy.quantile</code> with
<code>method='hazen'</code> and exhaustively enumerates conditional combinations. Its fixtures cover two
and three variables, ties, empty intersections, empty marginal classes, different missing rows,
common masking and islands. GPU tests compare joint observed counts with those fixtures and
exact sampled exceedances with the established independent uniform Philox oracle; resident
replay, invalid settings in the first and last variable, invalid permutation counts, padding slots
and neighbor overflow are covered.</p>
<p>Numerical GPU tests cover type-5 boundaries versus type 7, negative values, constant and tied
columns, empty classes, missing values, islands, invalid controls, neighbor capacity and replay.
The sorted test oracle independently computes boundaries; the established Philox oracle checks
exact exceedances, pseudo p-values and significance. The reproducible <code>modules/analysis/test/oracles/quantile-lisa-geoda.py</code> fixture runs actual
PyGeoDa 0.1.3 with 9,999 permutations. GPU observed counts match exactly; p-values are checked
with a broad 0.2 absolute tolerance because both the random streams and endpoint sampling differ.
Exact inference is asserted against the independent uniform Philox oracle, including the directed
tail and its all-ties degeneracy.</p>
<p>{/* parameters: GPUQuantileLISA generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h2 id="gpuquantilelisa-parameters"><code>GPUQuantileLISA</code> parameters</h2>
<ul>
<li>Packer: <code>getGPUPermutationParameterValues(settings, options?)</code>, writes <code>GPU_PERMUTATION_PARAMETER_LENGTH</code> (4) <code>uint32</code> values.
<ul>
<li>Input port: <code>parameters</code>.</li>
<li><code>options</code> (<code>{maximumPermutations?: number}</code>)</li>
</ul>
</li>
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
<p>{/* /parameters */}</p>`};export{e as default};