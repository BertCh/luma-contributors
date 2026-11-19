var e={slug:`api/gpu-spatial-analysis/space-time-hawkes`,title:`Space-time Hawkes process (STM-08)`,html:`<h1 id="space-time-hawkes-process-stm-08">Space-time Hawkes process (STM-08)</h1>
<p><code>GPUSpaceTimeHawkes</code> fits a self-exciting point process to a catalogue of events with a position and a time: how much of the activity is background, how many events each event triggers, how fast and how far the triggering reaches. It is the fixed-kernel space-time Hawkes model fitted by expectation-maximization (Veen and Schoenberg 2008, the "branching process" EM), with the parent search bounded in time and distance. ETAS (earthquake magnitudes, Omori-Utsu decay, parametric MLE steps) is later work and is not in this item.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">import</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  getGPUSpaceTimeHawkesParameterValues,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  GPUSpaceTimeHawkes</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">from</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> '@luma-contributors/analysis/gpu-spatial-analysis'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">;</span></span></code></pre>
<p>The classes are in <code>src/gpu-spatial-analysis/space-time-hawkes/</code>.</p>
<h2 id="model">Model</h2>
<p>For an event at place <code>s</code> and time <code>t</code> the conditional intensity is</p>
<pre><code>lambda(s, t) = mu + sum over earlier events i of
               omega * beta * exp(-beta (t - t_i)) / (2 pi sigma^2) * exp(-|s - s_i|^2 / (2 sigma^2))
</code></pre>
<table>
<thead>
<tr>
<th>Parameter</th>
<th>Meaning</th>
<th>Unit</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>mu</code> (<code>background</code>)</td>
<td>Background rate, constant in space and time</td>
<td>events per unit area and time unit</td>
</tr>
<tr>
<td><code>omega</code> (<code>branchingRatio</code>)</td>
<td>Expected offspring per event; below one the process is stable</td>
<td>none</td>
</tr>
<tr>
<td><code>beta</code> (<code>decayRate</code>)</td>
<td>Rate of the exponential time kernel; mean lag <code>1 / beta</code></td>
<td>per time unit</td>
</tr>
<tr>
<td><code>sigma</code> (<code>spatialStandardDeviation</code>)</td>
<td>Standard deviation of the isotropic Gaussian offspring displacement</td>
<td>position units</td>
</tr>
</tbody>
</table>
<p>Each iteration is one E-step and one M-step. The E-step gives every event <code>j</code> the probability <code>p_jj = mu / lambda_j</code> that it is background and <code>p_ij</code> that earlier event <code>i</code> triggered it. The M-step sets <code>mu = sum_j p_jj / (A T)</code>, <code>omega = sum_j (1 - p_jj) / sum_i (1 - exp(-beta (T - t_i)))</code>, <code>beta = sum p_ij / sum p_ij lag_ij</code> and <code>sigma^2 = sum p_ij d_ij^2 / (2 sum p_ij)</code>, with <code>A</code> the window area and <code>T</code> the duration. The sum over <code>i</code> in <code>omega</code> accounts for offspring that would fall after the end of the window.</p>
<p>What is not modelled, and what that costs:</p>
<ul>
<li>The spatial window edge: offspring that would fall outside the window are not accounted for, as the compensator uses the whole plane. Keep <code>sigma</code> small against the window; in the recovery spec (<code>sigma</code> 0.02 on a unit square) this biases <code>omega</code> low by about 3 percent.</li>
<li>A background that varies in space or time (the research plan names a kernel-smoothed <code>mu(s)</code>); <code>mu</code> is one number.</li>
<li>Pairs beyond <code>maximumLag</code> or <code>maximumDistance</code> are dropped and the kernel is not renormalized: choose them at several <code>1 / beta</code> and several <code>sigma</code>.</li>
<li>Simultaneous events (equal times) do not trigger each other.</li>
<li>Standard errors and any inference on the parameters. EM gives point estimates.</li>
</ul>
<h2 id="gpuspacetimehawkes"><code>GPUSpaceTimeHawkes</code></h2>
<p>On GPU, squared initialSpatialStandardDeviation must be finite normal float32 and at least 2^-126.
The CPU reference retains its wider variance domain.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> hawkes</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUSpaceTimeHawkes</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  positions,        </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// GraphDataView&#x3C;'float32x2'></span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  times,            </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// GraphDataView&#x3C;'float32'>, relative to the start of the window</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  parameters,       </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// GraphDataView&#x3C;'float32'> from getGPUSpaceTimeHawkesParameterValues</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  maximumIterations: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">100</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  maximumParents: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">256</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    estimates,      </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// 4 rows: background, branchingRatio, decayRate, spatialStandardDeviation</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    logLikelihood,  </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// optional, 1 row at the parameters entering the last update</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    fittedLogLikelihood, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// optional, 1 row at the published estimates</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    backgroundProbability, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// optional, one row per event, in input row order</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    status: {converged, iterationLimitReached, iterationCount, candidateOverflow, invalidCount, approximation}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  }</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(hawkes);</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">getGPUSpaceTimeHawkesParameterValues</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  windowArea: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  windowDuration: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">400</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  maximumLag: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  maximumDistance: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0.12</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  initialBackground: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  initialBranchingRatio: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0.3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  initialDecayRate: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  initialSpatialStandardDeviation: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0.05</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span></code></pre>
<p>Per-frame (no recompile): the contents of <code>parameters</code> (window area and duration, <code>maximumLag</code>, <code>maximumDistance</code>, <code>tolerance</code>, the four starting values) and of <code>positions</code> and <code>times</code>. Compile-time (a new graph): the event count, <code>maximumIterations</code>, <code>maximumParents</code>, and which optional outputs exist. Extra props: <code>id</code>, <code>spatialContext</code> (planar/native only) and <code>temporalContext</code> (<code>float32-relative</code>, checked and exposed, nothing converted).</p>
<p>Inputs:</p>
<ul>
<li><code>times</code> are float32 in <code>[0, windowDuration]</code>, relative to the start of the observation period, in the unit of the time kernel. A row with a non-finite position or a time outside the window is excluded and counted in <code>status.invalidCount</code>.</li>
<li>Coordinates are float32. Subtract a nearby origin in float64 before upload when they are large (UTM metres); differences of large float32 values lose their low bits, and the origin cannot be removed on the GPU after rounding.</li>
<li>Starting values must be positive. If they or the window are not, the estimates are NaN, <code>iterationCount</code> is 0 and neither <code>converged</code> nor <code>iterationLimitReached</code> is set.</li>
</ul>
<p>Outputs and status:</p>
<ul>
<li><code>estimates</code> hold the iterate after the last executed iteration. <code>fittedLogLikelihood</code> is evaluated at those estimates and is the diagnostic to use when comparing fitted models. Legacy <code>logLikelihood</code> remains the likelihood at the parameters entering the last update; even a converged fit can have a different value within its tolerance. Both compensators use the whole-plane spatial mass.</li>
<li><code>backgroundProbability</code> is <code>p_jj</code> at the published estimates; NaN for excluded rows or an invalid fit, including a catalogue consisting only of events at the observation-window endpoint. <code>1 - p_jj</code> is the probability that the event was triggered.</li>
<li><code>converged</code> is one when the largest relative change of any estimate in an iteration is at most <code>tolerance</code>; <code>iterationLimitReached</code> is one when <code>maximumIterations</code> ran out first. In that case the latest estimates are still published. <code>iterationCount</code> is the number of executed iterations.</li>
<li><code>candidateOverflow</code> is the number of events whose candidate parents inside the lag bound exceeded <code>maximumParents</code>: only the most recent <code>maximumParents</code> were examined for them, so their intensity is understated. A nonzero value means the bound should be raised (or the lag bound lowered).</li>
<li><code>approximation</code> is always one: the spatial edge and the parent bounds are intentional approximations.</li>
</ul>
<p>The background compensator multiplies bounded mantissas and combines exponents before rescaling,
so a finite three-factor result survives intermediate overflow from changing area or time units.
The Kepler fit summary pairs estimates with <code>fittedLogLikelihood</code>; the optional entering-iterate
<code>logLikelihood</code> remains a separate contributor diagnostic.</p>
<p>Limits (named constants, checked with named messages): <code>maximumIterations</code> from 1 to <code>GPU_SPACE_TIME_HAWKES_MAXIMUM_ITERATIONS</code> (1000); <code>maximumParents</code> from 1 to <code>GPU_SPACE_TIME_HAWKES_MAXIMUM_PARENTS</code> (4096; defaults <code>GPU_SPACE_TIME_HAWKES_DEFAULT_MAXIMUM_ITERATIONS</code> 100 and <code>GPU_SPACE_TIME_HAWKES_DEFAULT_MAXIMUM_PARENTS</code> 256); events from 2 to 2^24, the exact float32 integer range of the counts, and at most 8,388,608 for the 16-byte event table in a default 128 MiB binding. <code>estimatedCommandNodeCount</code> is an upper bound of the command nodes: six fixed nodes, an allowance of 64 for the published sort, and three nodes per iteration.</p>
<p>{/* lens:GPUSpaceTimeHawkes */}</p>
<h3 id="parameter-packer">Parameter packer</h3>
<p><code>getGPUSpaceTimeHawkesParameterValues</code> packs <code>[windowArea, windowDuration, maximumLag, maximumDistance, tolerance, initialBackground, initialBranchingRatio, initialDecayRate, initialSpatialStandardDeviation]</code> (<code>GPU_SPACE_TIME_HAWKES_PARAMETER_LENGTH</code> is 9; <code>tolerance</code> defaults to <code>GPU_SPACE_TIME_HAWKES_DEFAULT_TOLERANCE</code>, 1e-4) and throws if a value is not positive and finite. <code>GPU_SPACE_TIME_HAWKES_ESTIMATE_LENGTH</code> is 4. <code>GPU_SPACE_TIME_HAWKES_INTENSITY_FLOOR</code> and <code>GPU_SPACE_TIME_HAWKES_BRANCHING_RATIO_FLOOR</code> are the floors described below.</p>
<p>{/* parameters: GPUSpaceTimeHawkes generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h3 id="gpuspacetimehawkes-parameters"><code>GPUSpaceTimeHawkes</code> parameters</h3>
<ul>
<li>Packer: <code>getGPUSpaceTimeHawkesParameterValues(settings)</code>, writes <code>GPU_SPACE_TIME_HAWKES_PARAMETER_LENGTH</code> (9) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPUSpaceTimeHawkesParameterValues</code>:</p>
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
<td><code>windowArea</code></td>
<td><code>number</code></td>
<td>Area of the observation window, in squared position units. Positive and finite.</td>
</tr>
<tr>
<td><code>windowDuration</code></td>
<td><code>number</code></td>
<td>Length of the observation period in the unit of <code>times</code>. Event times lie in <code>[0, duration]</code>; an event outside it is excluded. Positive and finite.</td>
</tr>
<tr>
<td><code>maximumLag</code></td>
<td><code>number</code></td>
<td>Parents more than this many time units before an event are not considered. Positive.</td>
</tr>
<tr>
<td><code>maximumDistance</code></td>
<td><code>number</code></td>
<td>Parents farther than this many position units from an event are not considered. Positive.</td>
</tr>
<tr>
<td><code>tolerance</code> (optional)</td>
<td><code>number</code></td>
<td>Largest relative change of any estimate between two iterations that counts as converged. Defaults to <code>GPU_SPACE_TIME_HAWKES_DEFAULT_TOLERANCE</code>.</td>
</tr>
<tr>
<td><code>initialBackground</code></td>
<td><code>number</code></td>
<td>Starting background rate, events per unit area and time unit. Positive.</td>
</tr>
<tr>
<td><code>initialBranchingRatio</code></td>
<td><code>number</code></td>
<td>Starting branching ratio, expected offspring per event. Positive.</td>
</tr>
<tr>
<td><code>initialDecayRate</code></td>
<td><code>number</code></td>
<td>Starting rate of the exponential time kernel, per time unit. Positive.</td>
</tr>
<tr>
<td><code>initialSpatialStandardDeviation</code></td>
<td><code>number</code></td>
<td>Starting standard deviation of the Gaussian offspring displacement, in position units. Positive.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h2 id="how-it-runs">How it runs</h2>
<ol>
<li>Events are stably sorted by time (a radix sort of the order-preserving float key) into a table of position and time; excluded rows get a NaN time and sort last.</li>
<li>For each event one thread binary-searches the first earlier event inside the lag bound and walks the candidates up to <code>maximumParents</code> back, the ordered pair enumeration of a time-sorted list used by the Knox tests. Pairs outside the distance bound or with a non-positive lag are skipped.</li>
<li>The intensity is accumulated in the log domain in one pass with a running maximum (log-sum-exp), together with the lag- and distance-weighted sums, so no division by a tiny <code>lambda</code> occurs.</li>
<li>Event statistics are summed in tiles of consecutive events in float32 and the tile partials by a fixed pairwise (cascade) sum, so results are bitwise repeatable on a device.</li>
<li><code>maximumIterations</code> iterations are unrolled. A one-thread node tests convergence and disarms a GPU dispatch gate, after which the remaining iterations launch nothing.</li>
<li>When <code>fittedLogLikelihood</code> is requested, two final reduction nodes evaluate the log intensities and temporal mass at the published estimates. This same event evaluation writes <code>backgroundProbability</code> when requested, replacing its separate evaluation. The command-node estimate adds an allowance of two nodes. The operation form and CPU twin accept the same optional fitted-likelihood output.</li>
</ol>
<p>Cost per iteration is <code>events * min(maximumParents, candidates in the lag bound)</code> exp evaluations. No benchmark has been run on a GPU for this item.</p>
<h2 id="single-precision-what-was-measured">Single precision: what was measured</h2>
<p>The single-precision techniques come from Ko, Suchard and Holbrook (2024), who report a float32 kernel that runs reliably at one million events with log-sum-exp and a floor on tiny sums. They were measured here with <a href="../../../dev-docs/research/spatiotemporal-review-2026-11-09/experiments/stm08-hawkes-float32.mjs">stm08-hawkes-float32.mjs</a>, a CPU emulation of binary32 (<code>Math.fround</code> after every operation, denormals flushed) on a simulated catalogue of 987,262 events (unit square, <code>mu</code> 2, <code>omega</code> 0.5, <code>beta</code> 2, <code>sigma</code> 0.01, six EM iterations), against binary64 arithmetic on the same float32-rounded inputs:</p>
<table>
<thead>
<tr>
<th>Technique</th>
<th>Measured effect</th>
</tr>
</thead>
<tbody>
<tr>
<td>Log-sum-exp (running maximum) against a plain float32 sum of the terms</td>
<td>No difference at ordinary scales: both agree with binary64 to about 1e-8 per EM step. They differ only when the rates approach the float32 range. Rescaling positions so that <code>mu</code> and the offspring terms fall to about 1e-34 (position scale 1e17) leaves log-sum-exp at 1e-7 and the plain sum at 4e-6; at 1e18 it is 1e-7 against 3e-4 (the background responsibility off by 0.02); at 1e19 the plain sum gives NaN and log-sum-exp still fails because the rate itself is no longer representable. These scales are not physical, they locate where the plain sum fails.</td>
</tr>
<tr>
<td>Floor on the background rate before the logarithm</td>
<td>A floor of 1e-30 (the first choice here) ruined the fit at position scale 1e17 and beyond (relative error about 1), because the true rate was below it. The floor is therefore the smallest normal float32, <code>GPU_SPACE_TIME_HAWKES_INTENSITY_FLOOR</code>. With a start of <code>mu = 0</code> and no floor the logarithm is minus infinity; with the floor it is finite.</td>
</tr>
<tr>
<td>Tiled summation of the log-likelihood terms (about 244 events per tile, plain float32 inside)</td>
<td>One float32 accumulator over 987,262 events: relative error 5.5e-3. Tiles with the tile partials summed in float32: 1.5e-6. Neumaier compensation across tiles, on the CPU: 1.3e-6. The tiling does nearly all the work; the compensation adds little at this size, and on a GPU whose shader compiler reassociates float sums a float Neumaier term is removed altogether, so the tile partials are now merged by a fixed pairwise sum (about <code>log2(tileCount)</code> roundings of the total).</td>
</tr>
<tr>
<td>Float32 arithmetic of the E-step, all of the above in place</td>
<td>About 1e-8 relative error per EM step at one million events.</td>
</tr>
</tbody>
</table>
<p>The dominant error at one million events is not the arithmetic but the float32 time axis: rounding the inputs to float32 alone moved one EM step by 0.9 percent (relative, largest estimate) for a window of 250,000 time units with mean lag 0.5, because float32 spacing at 250,000 is 0.016. Choose the time unit and origin so that <code>windowDuration * 6e-8</code> is far below the shortest lag scale; a window longer than about 1e4 to 1e5 lag scales needs to be split in time. Position rounding is the same issue in space (subtract an origin in float64).</p>
<h2 id="validation">Validation</h2>
<ul>
<li><strong>Reference.</strong> A binary64 EM written from the definition above (<code>stm08-hawkes-reference.ts</code>), checked against a hand calculation of a three-event catalogue. The GPU matches it to 7e-7 relative in the estimates after 1 to 3 iterations and 4e-7 in the log-likelihood; at convergence the stopping iteration can differ by one and the estimates differ by up to 4e-5 (specs allow 1e-5 and 3e-4).</li>
<li><strong>External oracle: pending.</strong> The ETAS R package (0.7.2, GPL) is the oracle named for this item. R is not installed where this was built, ETAS fits a magnitude-dependent model this operator does not have, and no ETAS fixture has been generated. Nothing here claims agreement with ETAS.</li>
<li><strong>Recovery on simulated data with known parameters.</strong> Catalogues of about 1,500 events are simulated from <code>mu</code> 2, <code>omega</code> 0.5, <code>beta</code> 2, <code>sigma</code> 0.02 on a unit square over 400 time units. The sampling distribution of the estimator is measured by fitting 24 independent simulations in binary64: standard deviations of 0.059 (<code>mu</code>), 0.016 (<code>omega</code>), 0.106 (<code>beta</code>) and 0.00037 (<code>sigma</code>), and means 2.018, 0.486, 2.028 and 0.0197, the 3 percent shortfall of <code>omega</code> being the unmodelled spatial edge. A GPU fit must lie within the absolute bias plus four standard deviations of the truth, a two-sided normal probability of about 6e-5 per parameter; three fits pass. Negative controls: a catalogue simulated with <code>omega</code> 0 returns <code>omega</code> below 0.1, and the generating background and triggered labels are separated by the background probability (mean 0.85 or more for background events, 0.25 or less for triggered events).</li>
</ul>`};export{e as default};