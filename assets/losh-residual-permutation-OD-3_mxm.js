var e={slug:`api/gpu-spatial-analysis/losh-residual-permutation`,title:`Conditional LOSH residual inference`,html:`<h1 id="conditional-losh-residual-inference">Conditional LOSH residual inference</h1>
<p><code>GPULOSHResidualPermutationTest</code> combines descriptive local spatial heteroscedasticity
(LOSH) with a conditional residual-label randomization test. This is a distinct null
model. It does not implement esda LOSH inference and does not permute the original
observations and refit their local means on every draw.</p>
<h2 id="statistic-and-null-model">Statistic and null model</h2>
<p>For finite, unmasked rows, compute a local positive-weight neighbor mean. Self links,
nonfinite or nonpositive weights and excluded neighbors do not contribute. A row with
no eligible neighbor has no residual. Set <code>r[i] = abs(y[i] - localMean[i]) ** a</code>, where
<code>residualPower</code> is 1 or 2 (default 2), and divide every finite residual by their global
mean. The optional <code>output.normalizedResiduals</code> exposes that field. A zero or nonfinite
global residual mean yields NaN residual outputs and no inferential tests.</p>
<p>The descriptive statistic is the weighted mean of eligible normalized neighbor
residuals. In a conditional draw for focus <code>i</code>, retain its residual and assign distinct
other finite residual labels without replacement to its active neighbor slots, in CSR
slot order. The graph, residual field, denominator and slot weights stay fixed. Finite
residuals remain eligible donors even when they have no eligible residual neighbor of
their own. Each row has its own conditional experiment; these draws do not constitute
one shared global permutation across all rows.</p>
<p>The implementation delegates randomization to the existing <code>GPULocalPermutationTest</code>
<code>localG</code> kernel. Its unnormalized weighted residual lag is a positive row-specific
multiple of LOSH, so ordering, ties and tail counts agree. The <code>localG</code> observed output
is deliberately not exposed as LOSH. The public <code>output.statistic</code> comes from the
descriptive LOSH contributor.</p>
<h2 id="parameters-and-outputs">Parameters and outputs</h2>
<p>Supply <code>values</code>, <code>weights</code>, optional <code>mask</code>, optional <code>residualPower</code>, <code>parameters</code>,
<code>maximumPermutations</code>, optional <code>maximumNeighbors</code> (default 32, maximum 64), and an
explicit <code>alternative</code>: <code>greater</code>, <code>lesser</code> or <code>two-sided</code>. Construct <code>parameters</code> with
<code>getGPUPermutationParameterValues({seed, permutationCount, significanceLevel})</code>.</p>
<p>Seed is a nonnegative safe integer. Draws use the existing Philox counter stream and
partial Fisher–Yates sampler. Repeated inputs, seed and count reproduce the same draw
sequence; seed/count buffers can change between frames without rebuilding the graph.
The active count must be in <code>[1, maximumPermutations]</code>. Counts are never clamped.</p>
<p>For <code>P</code> draws, count <code>g = count(simulated >= observed)</code> and
<code>l = count(simulated &#x3C;= observed)</code>. Ties enter both tails:</p>
<table>
<thead>
<tr>
<th>Tail</th>
<th>Unadjusted pseudo p-value</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>greater</code></td>
<td><code>(g + 1) / (P + 1)</code></td>
</tr>
<tr>
<td><code>lesser</code></td>
<td><code>(l + 1) / (P + 1)</code></td>
</tr>
<tr>
<td><code>two-sided</code></td>
<td><code>min(1, 2 * (min(g, l) + 1) / (P + 1))</code></td>
</tr>
</tbody>
</table>
<p>Caller-owned <code>output.inference</code> contains <code>exceedances</code>, <code>pseudoPValues</code>, optional
<code>adjustedPValues</code>, optional <code>significant</code>, and <code>status.overflow</code>. The required
<code>output.status</code> is the separate per-row descriptive <code>GPU_LOCAL_VARIABILITY_STATUS</code>.
Descriptive success does not imply inferential completion: inspect both statuses.
Untested rows have NaN p-values, exceedances <code>GPU_LOCAL_PERMUTATION_NOT_TESTED</code>, and
zero significance. Invalid permutation counts or incomplete upstream weights invalidate
all inference; exceeding the neighbor bound invalidates affected rows and sets overflow.
All output views must be mutually disjoint and disjoint from input views.</p>
<p><code>multipleTesting</code> uses the shared adjustment contributor; its family is every finite
tested p-value. Excluded, undefined and untested rows do not enter the family. <code>none</code>,
Bonferroni, Holm, Benjamini–Hochberg and Benjamini–Yekutieli follow the existing API.
Bonferroni/Holm address familywise error; BY is the conservative FDR choice under
arbitrary dependence. BH requires the usual dependence assumptions. Any error-control
interpretation also requires the residual-label exchangeability null to be appropriate;
residuals obtained from overlapping fitted neighborhoods are not automatically
exchangeable under an arbitrary raw-observation null.</p>
<h2 id="evidence-and-limits">Evidence and limits</h2>
<p><code>test/oracles/losh-backlog-inference.py</code> uses only Python standard-library exact rational
arithmetic and enumerates every ordered distinct donor assignment for a five-row graph.
Its fixture records descriptive residuals/statistics and exact inclusive null-tail
probabilities. GPU tests compare seeded Monte Carlo estimates with those exhaustive
probabilities, verify Bonferroni composition, and exercise undefined residuals,
upstream incompleteness, invalid counts and neighbor-capacity failure. Node tests parse
maximal-option kernels and enforce the default eight storage-binding limit and output
alias rejection. These tests establish this conditional composition, not parity with
an external LOSH inference library or multivariate joint randomization.</p>
<p>All numerical stages remain on the GPU, with no hidden readback or CPU fallback.
The descriptive LOSH denominator uses parallel maximum, count and sum reductions; its
scaled sum avoids overflow of the unscaled residual total. A resident <code>residualPower</code> parameter
view can change the descriptive power between encodings. Multivariate joint inference and
raw-observation LOSH refitting are outside this API.</p>`};export{e as default};