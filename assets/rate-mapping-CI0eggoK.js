var e={slug:`api/gpu-spatial-analysis/rate-mapping`,title:`Rate mapping recipes`,html:`<h1 id="rate-mapping-recipes">Rate mapping recipes</h1>
<p><code>addExcessRiskRecipe(graph, props)</code> and <code>addSpatialRateRecipe(graph, props)</code> append GPU nodes
without submitting commands or reading results. Import them from <code>@luma-contributors/analysis/gpu-spatial-analysis</code>.
Both return ordinary recipe results: <code>outputs.excessRisk</code> or <code>outputs.spatialRates</code>, contributors,
intermediates and stage status. Excess risk reports <code>invalidCount</code> in an <code>excess-risk</code> stage;
spatial rates preserve the supplied
weights status in a <code>weights</code> stage. Pass <code>output</code> for a caller-owned readback-capable view;
otherwise the terminal view is graph-owned. Imported buffers remain caller-owned. Inputs are packed
<code>float32</code> or <code>uint32</code> graph views, with matching row counts; there is no implicit vector chunk packing.
For uint32 counts, GPU conversion to float32 loses integer precision above 2^24.</p>
<p>Both recipes prepare their complete declarations on a separate graph before changing the caller's
graph. Invalid shapes, aliases, foreign views and contributor expansion failures leave the caller's
nodes and resources unchanged, so a corrected configuration can reuse the same recipe ID. Preparation
uses the same device limits and preserves buffer aliases and view offsets; it never compiles or
allocates GPU storage. It repeats host declaration work once. This guarantee covers configuration
validation; existing caller node-ID collisions, a graph already frozen by compilation, and allocation
failures during final declaration are outside it. Supply unused recipe/resource IDs on a mutable graph.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> risk</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> addExcessRiskRecipe</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph, {events, populations, mask});</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> spatial</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> addSpatialRateRecipe</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph, {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  events, populations, weights, mask, scale: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">100000</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span></code></pre>
<p>Excess risk is dimensionless observed/expected:</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span>pooled = sum(events) / sum(populations)</span></span>
<span class="line"><span>risk[i] = events[i] / (populations[i] * pooled)</span></span></code></pre>
<p>The sums include exactly rows with finite events, finite positive population and nonzero optional
mask. This uses a pooled ratio, never the arithmetic mean of row rates. A nonpositive expected count
produces NaN, including the all-zero event case. Excluded rows produce NaN. Finite negative events
are accepted consistently with the existing empirical-Bayes input contract; callers needing count
validation must provide it before upload. <code>GPUEmpiricalBayesRates</code> now exposes the same optional
<code>output.excessRisk</code>, reusing its fixed-order GPU reductions; its existing outputs and defaults are
unchanged. The recipe computes the existing moments even when only risk is requested.</p>
<p>Spatial rates use the focus row and its included nonself neighbors:</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span>spatial[i] = scale * (events[i] + sum(neighbor events)) /</span></span>
<span class="line"><span>                     (populations[i] + sum(neighbor populations))</span></span></code></pre>
<p><code>scale</code> is a finite nonnegative compile-time multiplier, default 1. Excess risk has no scale option.
<code>weights</code> is a square CSR structure: numeric weights are intentionally ignored (the GeoDa binary
neighborhood convention), including nonunit or zero stored weights. Self edges are skipped to count
the focus row once. Duplicate nonself slots count repeatedly; supply unique slots for a simple
adjacency graph. Out-of-range neighbors are skipped. Masked, nonfinite or nonpositive-population
neighbors are excluded. A focus row excluded by those same rules produces NaN.</p>
<p>The default <code>islandPolicy: 'undefined'</code> produces NaN when there is no included nonself neighbor,
including islands created by the mask. Set <code>'own-rate'</code> to return the focus rate in that case.
The existing <code>GPUSpatialEmpiricalBayesRates.output.spatialRates</code> retains its own-rate island
behavior. This recipe reuses that contributor and applies the explicit island/scale policy in a
final GPU pass. It does not average rates or reinterpret CSR numeric weights as exposure weights.
At most seven storage buffers are bound by the final pass, within default WebGPU limits. The
existing spatial contributor can do extra moment calculations; there is no duplicate rate kernel.</p>
<h2 id="reference-audit-and-limits">Reference audit and limits</h2>
<p>The reference is GeoDa commit
<a href="https://github.com/GeoDaCenter/geoda/blob/37510860bd9f7b8ae145c15759f7abc5ab217cf3/ShapeOperations/RateSmoothing.cpp"><code>37510860bd9f7b8ae145c15759f7abc5ab217cf3</code></a>,
<code>RateSmoother_ExcessRisk</code>, <code>RateSmoother_SRS</code>, <code>RateSmoother_EBS</code> and <code>RateSmoother_SEBS</code>.
The audit confirmed pooled excess risk, binary neighborhood sums plus self, and undefined SRS
islands. Its undefined flag and zero placeholder are represented here by NaN. Its positive finite
input calculations agree with these formulas. This is an independent implementation using existing
luma kernels, with no upstream source copied.</p>
<p>For invalid populations, GeoDa can accumulate values before marking them undefined; these recipes
instead exclude invalid rows from both global totals and neighborhoods, preserving the existing
luma rate contract. Nonfinite inputs are likewise explicitly excluded. These are documented
extensions, not full GeoDa behavior parity for malformed input. GeoDa's EB standardized output
also differs from the existing Assuncao-Reis convention; that output is unchanged. Raw, global EB
and spatial EB remain available through the existing contributors; this addition supplies the two
missing named compositions only.</p>
<p>GPU tests use independent hand arithmetic for pooled rates, a deliberately unequal-population
example distinguishing pooling from average-of-rates, nonunit weights, self edges, mask-induced
islands, invalid rows, zero events and mutated-input replay on the same compiled graph. Existing
rate-smoothing suites retain their esda fixtures for regression coverage. All arithmetic is float32;
finite inputs whose totals or intermediate arithmetic overflow are outside the numerical parity
contract. There is no CPU fallback. Global recipes must receive the whole intended population:
executing independently per source batch changes the pooled estimator.</p>
<p>{/* lens:addExcessRiskRecipe */}</p>
<p>::lens{scenario=gpu-spatial-analysis.excess-risk-recipe rung=graph}</p>
<p>{/* lens:addSpatialRateRecipe */}</p>
<p>::lens{scenario=gpu-spatial-analysis.spatial-rate-recipe rung=graph}</p>`};export{e as default};