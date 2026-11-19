var e={slug:`api/gpu-spatial-analysis/sinkhorn-divergence`,title:`Sinkhorn divergence between spatial distributions (STM-09)`,html:`<h1 id="sinkhorn-divergence-between-spatial-distributions-stm-09">Sinkhorn divergence between spatial distributions (STM-09)</h1>
<p><code>GPUSinkhornDivergence</code> measures how different two mass fields on the same grid are, in the units of squared distance: the debiased entropic optimal transport cost (the Sinkhorn divergence of Genevay et al. 2018 and Feydy et al. 2019). It is zero for identical fields and grows as mass has to move farther, which makes it a verification score for forecast and observed fields that does not suffer the "double penalty" of cell-by-cell error, and a comparison of two density maps.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">import</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  getGPUSinkhornDivergenceParameterValues,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  GPUSinkhornDivergence</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">from</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> '@luma-contributors/analysis/gpu-spatial-analysis'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">;</span></span></code></pre>
<p>The classes are in <code>src/gpu-spatial-analysis/optimal-transport/</code>.</p>
<h2 id="definition">Definition</h2>
<p>For masses <code>a</code> and <code>b</code> on the grid, normalized to sum to one, and the squared Euclidean cost <code>C</code> between cell centers,</p>
<pre><code>OT(a, b) = min over couplings P of  &#x3C;P, C> + epsilon KL(P | a x b)
S(a, b)  = OT(a, b) - OT(a, a) / 2 - OT(b, b) / 2
</code></pre>
<p><code>S</code> is zero when <code>a = b</code>, never negative, and symmetric. Plain <code>OT(a, a)</code> is not zero (the entropic bias), which is why the divergence subtracts it. Both masses are normalized, so a difference in total mass is not seen, and there is no unbalanced (KL-marginal) variant. The POT function <code>ot.bregman.empirical_sinkhorn_divergence</code> (0.9.7.post1) is a different quantity: its documentation says it does not include the entropic terms, and it differs from <code>S</code> by 0.05 on the <code>epsilon</code> 0.5 fixture case.</p>
<h2 id="gpusinkhorndivergence"><code>GPUSinkhornDivergence</code></h2>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> sinkhorn</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUSinkhornDivergence</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  grid,             </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// GPURasterMetadata: only width and height are used</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  source,           </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// GraphDataView&#x3C;'float32'>, width * height masses in raster row order</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  target,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  parameters,       </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// GraphDataView&#x3C;'float32'> from getGPUSinkhornDivergenceParameterValues</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  maximumIterations: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">100</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    divergence,     </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// 1 row</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    components,     </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// optional, 3 rows: OT(a, b), OT(a, a), OT(b, b)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    marginalError,  </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// optional, 1 row</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    status: {converged, iterationLimitReached, iterationCount, invalidCount, approximation}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  }</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(sinkhorn);</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">getGPUSinkhornDivergenceParameterValues</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  cellWidth: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  cellHeight: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  epsilon: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  tolerance: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">1e-4</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  epsilonStart: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">16</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  scalingFactor: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0.8</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span></code></pre>
<p>Per-frame (no recompile): the contents of <code>parameters</code> (cell width and height, <code>epsilon</code>, <code>tolerance</code>, <code>epsilonStart</code>, <code>scalingFactor</code>) and of both mass fields. Compile-time (a new graph): the grid size, <code>maximumIterations</code>, <code>blurRadius</code>, and which optional outputs exist. Extra prop: <code>id</code>, <code>spatialContext</code> (planar/native only).</p>
<ul>
<li><code>parameters</code> is <code>[cellWidth, cellHeight, epsilon, tolerance, epsilonStart, scalingFactor]</code> (<code>GPU_SINKHORN_DIVERGENCE_PARAMETER_LENGTH</code> is 6). <code>epsilon</code> is in squared position units. Iteration <code>k</code> uses <code>max(epsilon, epsilonStart * scalingFactor^k)</code>; the defaults (<code>epsilonStart = epsilon</code>, <code>scalingFactor = 1</code>) are no schedule. <code>tolerance</code> defaults to <code>GPU_SINKHORN_DIVERGENCE_DEFAULT_TOLERANCE</code> (1e-4). The packer throws on a value that is not positive and finite, on <code>epsilonStart &#x3C; epsilon</code> and on <code>scalingFactor</code> outside <code>(0, 1]</code>.</li>
<li>Masses must be non-negative and finite. A negative or non-finite cell is counted in <code>status.invalidCount</code> and treated as zero. If either field has no mass, or the parameters are unusable, <code>divergence</code> and the other values are NaN, <code>iterationCount</code> is 0 and neither <code>converged</code> nor <code>iterationLimitReached</code> is set.</li>
<li>Each field is normalized independently after rescaling by a power of two chosen from its largest finite mass. The multiplier stays normal and finite and every rescaled mass is below four, so even a raw total beyond float32 range is supported. For example, <code>[3, 2]</code> and <code>[2, 3]</code> give the same distributions when either or both fields are multiplied by <code>2^126</code>; every input remains finite. The CPU twin uses float64 totals and supports the same case.</li>
<li><code>components</code> are reported because the divergence is a difference of them and carries their rounding.</li>
<li><code>marginalError</code> is the L1 distance between the cross plan's target marginal and the normalized target masses, measured before the last target-side update. <code>converged</code> is one only when the cross plan and both symmetric self plans each have L1 marginal error at most <code>tolerance</code> before their updates, with the regularization at its final value. The symmetric plans have equal row and column residuals. A small <code>marginalError</code> alone does not certify convergence. <code>iterationLimitReached</code> is one when <code>maximumIterations</code> (1 to <code>GPU_SINKHORN_DIVERGENCE_MAXIMUM_ITERATIONS</code>, 1000; default <code>GPU_SINKHORN_DIVERGENCE_DEFAULT_MAXIMUM_ITERATIONS</code>, 100) ran out first, in which case the latest potentials are still used.</li>
<li>The residual pass must use the exact packed final <code>epsilon</code>. A distinct <code>epsilonStart</code> with <code>scalingFactor = 1</code> never reaches it and exhausts the budget, even if it differs by only one float32 step. The website scene displays <code>iteration limit reached (estimate)</code> alongside its distance and cost readout when this flag is set; it clears the indication after a converged run.</li>
<li><code>blurRadius</code> (compile-time, default the longer side minus one, which is exact) limits each 1D blur window. A smaller radius drops Gibbs kernel terms beyond it. It is exact when the dropped terms are negligible against the nearest mass and sets <code>status.approximation</code>; a cell with no mass in its window gets a huge potential, so keep the window large against the gaps between masses.</li>
<li>A positive-mass cell unreachable through the blur window contributes its entire normalized mass to the residual. Repeated unreachable potentials do not count as convergence. For example, <code>[1, 0, 0]</code> to <code>[0, 0, 1]</code> on a <code>3 x 1</code> grid with <code>blurRadius = 1</code> has cross marginal error <code>1</code> and reaches the iteration limit for tolerance below <code>1</code>.</li>
<li>Grid sides are limited to <code>GPU_SINKHORN_DIVERGENCE_MAXIMUM_GRID_SIDE</code> (2048) because one blur invocation walks a row or a column; cells to 2^24 (float32 counts). <code>estimatedCommandNodeCount</code> is <code>10 + 7 * maximumIterations</code>, an upper bound including both optional publication passes. The two additional fixed nodes find the per-field maxima before the rescaled mass sums.</li>
</ul>
<p>{/* lens:GPUSinkhornDivergence */}</p>
<p>{/* parameters: GPUSinkhornDivergence generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h3 id="gpusinkhorndivergence-parameters"><code>GPUSinkhornDivergence</code> parameters</h3>
<ul>
<li>Packer: <code>getGPUSinkhornDivergenceParameterValues(settings)</code>, writes <code>GPU_SINKHORN_DIVERGENCE_PARAMETER_LENGTH</code> (6) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPUSinkhornDivergenceParameterValues</code>:</p>
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
<td><code>cellWidth</code></td>
<td><code>number</code></td>
<td>Cell width in position units. Positive and finite.</td>
</tr>
<tr>
<td><code>cellHeight</code></td>
<td><code>number</code></td>
<td>Cell height in position units. Positive and finite.</td>
</tr>
<tr>
<td><code>epsilon</code></td>
<td><code>number</code></td>
<td>Entropic regularization, in squared position units (the cost is the squared distance between cell centers). Positive. In float32 the log-domain exponents reach <code>maximum cost / epsilon</code>, and each carries an absolute rounding error of about <code>6e-8</code> times that, so keep <code>epsilon</code> at least about <code>1e-3</code> times the squared grid diagonal.</td>
</tr>
<tr>
<td><code>tolerance</code> (optional)</td>
<td><code>number</code></td>
<td>Largest L1 marginal error allowed in each of the cross plan and both symmetric self plans before their updates at the final epsilon. Defaults to <code>GPU_SINKHORN_DIVERGENCE_DEFAULT_TOLERANCE</code>.</td>
</tr>
<tr>
<td><code>epsilonStart</code> (optional)</td>
<td><code>number</code></td>
<td>First regularization of the epsilon schedule, at least <code>epsilon</code>. Defaults to <code>epsilon</code> (no scaling).</td>
</tr>
<tr>
<td><code>scalingFactor</code> (optional)</td>
<td><code>number</code></td>
<td>Factor in <code>(0, 1]</code> applied to the regularization each iteration until it reaches <code>epsilon</code>: iteration <code>k</code> uses <code>max(epsilon, epsilonStart * scalingFactor^k)</code>. Defaults to 1.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h2 id="how-it-runs">How it runs</h2>
<p>In the log domain the dual iteration is <code>g_j = -epsilon * LSE_i(f_i / epsilon + log a_i - C_ij / epsilon)</code> and the same for <code>f</code>. The Gibbs kernel <code>exp(-C / epsilon)</code> is separable in the two grid axes, so each log-sum-exp over all cells is two 1D passes, one along rows and one along columns: the raster convolution pattern in the log semiring, at <code>O(cells * (width + height))</code> per update instead of <code>O(cells^2)</code>. The three problems <code>(a, b)</code>, <code>(a, a)</code> and <code>(b, b)</code> run in one batch, the symmetric ones with the averaged update <code>f &#x3C;- (f + T f) / 2</code>. Each first half-iteration measures all three marginal errors before applying the updates, using <code>mass * |exp((old - T(old)) / epsilon) - 1|</code>. A tiled reduction sums each plan's residual separately. A one-thread node requires all three to meet tolerance at the final epsilon, applies the schedule and disarms a GPU dispatch gate, so finished iterations launch nothing; zero-mass cells are masked rather than given a log of minus infinity. The divergence is <code>sum a (f_ab - f_aa) + sum b (g_ab - g_bb)</code>, with the differences formed before the tiled sum (plain float32 inside tiles of a few hundred cells, then a fixed pairwise merge; not a compensated float sum, which the test GPU's shader compiler removes).</p>
<p><code>GPUSinkhornDivergenceOperation</code> and <code>CPUSinkhornDivergenceOperation</code> use the same three-plan stopping rule and explicit iteration budget. The CPU twin evaluates the full kernel in float64; near the tolerance, rounding can change the stopping iteration between the two forms.</p>
<h2 id="single-precision">Single precision</h2>
<p>The exponents reach <code>maximum cost / epsilon</code> and each carries a relative rounding of 6e-8, so keep <code>epsilon</code> at least about 1e-3 times the squared grid diagonal (the exponent range at most 1000; the specs run at most 200). The marginal-error test cannot resolve changes below about <code>6e-8 * maximum cost / epsilon</code>, so a tolerance below that never converges (at an exponent range of 100 and a tolerance of 1e-5, several random cases did not converge in 800 iterations even for identical inputs; they still agreed to 4e-6 of the cost).</p>
<h2 id="validation">Validation</h2>
<ul>
<li><strong>POT.</strong> POT 0.9.7.post1 (MIT) was unpacked into a scratch directory, not installed; the generator is <code>test/oracles/stm09-sinkhorn-pot.py</code> and the fixture <code>fixtures/stm09-sinkhorn-pot.json</code> (seven grids from 8 by 8 to 16 by 16, the largest exponent range 196, with zero cells, anisotropic cells, identical inputs and point masses). The cost is evaluated from POT's coupling (<code>ot.sinkhorn</code>, <code>sinkhorn_log</code>, marginal tolerance 1e-13) as <code>&#x3C;P, C> + epsilon KL(P | a x b)</code>. A converged GPU run agrees with POT to 3e-7 of the cost or better in the divergence and in each component (the spec allows 1e-6). The independent binary64 dense reference in the test directory agrees with POT to 1e-7.</li>
<li><strong>Closed form.</strong> Two point masses at squared distance <code>d2</code> give <code>S = d2</code> for any <code>epsilon</code> (one coupling cell, no entropy, <code>OT(a, a) = 0</code>); checked on three cell sizes.</li>
<li><strong>Zero for identical inputs and non-negative.</strong> On ten random sparse pairs (grids 5 to 7 cells a side, exponent range 30 to 150) with 400 iterations and the default tolerance, <code>S(a, a)</code> and the negative part of <code>S(a, b)</code> stay within <code>1.1 * 2^-24 * 150</code>, about 9.8e-6, of <code>OT(a, b)</code>; the plain <code>OT(a, a)</code> is above 0.1, so the debiasing is what makes it vanish. Converged runs are far closer: 1e-9 to 4e-7 of the cost. Distinct inputs are positive by at least 1e-3 of the cost, and <code>S(a, b)</code> and <code>S(b, a)</code> agree within the same bound.</li>
<li>Also checked: invariance to the scale of each field, invalid cells, unusable inputs, the iteration limit, the epsilon schedule, a short blur window against the exact one (and a window of three cells that is clearly wrong), growth with distance, and bitwise repeatability.</li>
<li>Not measured: GPU speed at 2048 by 2048.</li>
</ul>`};export{e as default};