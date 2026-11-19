var e={slug:`api/gpu-spatial-analysis/operators-dynamics`,title:`GPU Spatial Analysis: Distribution dynamics & space-time tests`,html:`<h1 id="distribution-dynamics--space-time-tests">Distribution dynamics &#x26; space-time tests</h1>
<p>Distribution dynamics contributors use period-major columns: period <code>t</code> of row <code>i</code> is at
<code>t * rowCount + i</code>, with <code>periodCount</code> periods.</p>
<p>See <a href="/luma-contributors/docs/api/gpu-spatial-analysis/space-time-statistics">Space-time statistics</a> for space-time weights, the lagged and differential
local Moran, and cross-correlation by lag. The near-repeat, local Knox and Jacquez tests are listed with
the contributors below; their times follow the <a href="/luma-contributors/docs/api/gpu-spatial-analysis/temporal-context">temporal context</a>.</p>
<h3 id="gpuclassassignment"><code>GPUClassAssignment</code></h3>
<p>Assigns each county's 2023 unemployment rate to one of five quantile classes using shared class breaks.</p>
<p>::lens{scenario=gpu-spatial-analysis.class-assignment rung=result caption="437 Midwest counties coloured by quantile class of 2023 unemployment; higher class means higher rate."}</p>
<p><code>new GPUClassAssignment({values, breaks, classCount, mask, intervalClosure, output})</code> assigns each value the number of
inner edges <code>e[1..k-1]</code> that are <code>&#x3C;=</code> the value, clamped, the same rule as <code>GPUClassBreaks.classCounts</code>.
NaN and masked rows get <code>GPU_CLASS_ASSIGNMENT_NO_CLASS</code> (<code>0xffffffff</code>).</p>
<p><code>intervalClosure</code> (compile-time, default <code>'left'</code>) decides the class of a value exactly equal to an inner edge.
<code>'left'</code> treats an edge as the lower edge of the class above it (classes <code>[e[j], e[j+1])</code>): the value goes to the
upper class. Use it for <code>'natural-breaks'</code> edges, each of which is the smallest value of its class, and for d3 scales.
<code>'right'</code> treats an edge as the upper bound of the class below it (classes <code>(e[j], e[j+1]]</code>, the minimum in the
first class): the count is of inner edges strictly <code>&#x3C;</code> the value. This is the rule of mapclassify and giddy, so pass
<code>'right'</code> with <code>'quantile'</code>, <code>'equal-interval'</code>, <code>'standard-deviation'</code>, <code>'head-tail'</code> or <code>'box-plot'</code> breaks when the
classes or transition counts must equal theirs. The two differ only for rows equal to an edge, which quantile breaks
produce whenever the data are tied at the quantile. <code>GPUSpatialMarkov</code> has the same choice for its lag classes as
<code>lagIntervalClosure</code>. Run one <code>GPUClassBreaks</code> over the
pooled <code>n * T</code> column and assign every period with its breaks to get one legend for a time slider.</p>
<h4 id="how-it-computes">How it computes</h4>
<p>{/* lens:GPUClassAssignment */}</p>
<p>::lens{scenario=gpu-spatial-analysis.class-assignment rung=datum datum=15 caption="Cook County, Illinois (row 15 of the county table): its rate and class."}</p>
<p>::lens{scenario=gpu-spatial-analysis.class-assignment rung=formula caption="A county's class is the number of inner breaks at or below its rate."}</p>
<p>::lens{scenario=gpu-spatial-analysis.class-assignment rung=kernel caption="Inspect the recorded shader and its input and output bindings."}</p>
<p>::lens{scenario=gpu-spatial-analysis.class-assignment rung=graph caption="Follow the recorded dependencies from uploaded inputs to captured outputs."}</p>
<h3 id="gpuclassificationfit"><code>GPUClassificationFit</code></h3>
<p>Scores how well five quantile classes describe 2023 county unemployment: per-class counts, medians, deviations and GADF.</p>
<p>::lens{scenario=gpu-spatial-analysis.classification-fit rung=result caption="Quantile classes of 437 counties; the table lists counts, medians and fit statistics per class."}</p>
<p>Goodness of fit of a classification (mapclassify 2.11 <code>adcm</code>, <code>get_gadf()</code>, <code>get_tss()</code>): pass <code>values</code>, the
<code>classes</code> from <code>GPUClassAssignment</code>, <code>classCount</code> (1 to 256) and an optional <code>mask</code>. Outputs are <code>counts</code>,
exact per-class <code>medians</code> (numpy convention, mean of the two middle values), <code>absoluteDeviations</code> and
<code>squaredDeviations</code> (one row per class plus a last row for all counted rows) and an optional five-row
<code>summary</code> <code>[GADF, ADCM, ADAM, GVF, TSS]</code> (<code>GPU_CLASSIFICATION_FIT_*</code>). <code>ADCM</code> is the sum of absolute
deviations from the class medians, <code>ADAM</code> the same from the overall median, <code>GADF = 1 - ADCM / ADAM</code> (1 when
<code>ADAM</code> is 0), and <code>TSS</code> is mapclassify's name for the within-class sum of squares about the class means, not a
total sum of squares. <code>GVF = 1 - TSS / total sum of squares</code> is an extension, because mapclassify 2.11 has no
<code>gvf</code>. Empty classes add 0, as in mapclassify. Two stable radix sorts (value, then class) leave each class a
sorted segment, so medians are read directly and sums run in sorted order: no atomics, bit-reproducible, no
per-class search passes.</p>
<p>The spec pins mapclassify 2.11.0 Quantiles, NaturalBreaks and EqualInterval fits (about 1e-4 relative).
Deviations: sums run in float32 and in sorted order rather than index order, non-finite values are skipped, and
at least one row is required.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUClassificationFit</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({values, classes, classCount: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, output: {counts, medians, absoluteDeviations, squaredDeviations, summary}}));</span></span></code></pre>
<h4 id="how-it-computes-1">How it computes</h4>
<p>{/* lens:GPUClassificationFit */}</p>
<p>::lens{scenario=gpu-spatial-analysis.classification-fit rung=datum datum=4 caption="Class 4, the highest class of 2023 rates: its county count and median."}</p>
<p>::lens{scenario=gpu-spatial-analysis.classification-fit rung=formula caption="GADF compares absolute deviations around class medians with those around the overall median."}</p>
<p>::lens{scenario=gpu-spatial-analysis.classification-fit rung=kernel caption="Inspect the recorded shader and its input and output bindings."}</p>
<p>::lens{scenario=gpu-spatial-analysis.classification-fit rung=graph caption="Follow the recorded dependencies from uploaded inputs to captured outputs."}</p>
<h3 id="gputransitionmatrix"><code>GPUTransitionMatrix</code></h3>
<p>Up to 65,535 classes subject to dense matrix and device binding limits; transition diagnostics
retain 64 classes.</p>
<p><strong><code>GPUTransitionMatrix</code></strong>: Counts how many counties move between unemployment classes from one year to the next, with row-normalised probabilities.</p>
<p>::lens{scenario=gpu-spatial-analysis.transition-matrix rung=result caption="Counts of 437 counties moving between five classes over 15 yearly steps, 2008 to 2023."}</p>
<p><strong><code>GPUTransitionDiagnostics</code></strong>: Tests whether five states share one transition matrix and solves each chain's steady state, flagging sparse states.</p>
<p>::lens{scenario=gpu-spatial-analysis.transition-diagnostics rung=result caption="Per-state transition counts of Illinois, Indiana, Michigan, Ohio and Wisconsin feed Q, LR, steady states and sparsity."}</p>
<p>Markov transition counts between classes of consecutive periods (giddy <code>Markov</code>).</p>
<ul>
<li>Props: <code>classes</code> (<code>rowCount * periodCount</code> rows; the row count is derived from it), <code>periodCount</code>,
<code>classCount</code> (1–65,535, subject to dense matrix/device bounds), optional <code>periodLag</code> (default 1),
<code>conditionClasses</code> with <code>conditionCount</code>, <code>mask</code>, and <code>output: {counts, probabilities?, rowTotals?, ignored?}</code>.</li>
<li><code>counts</code> holds exact <code>uint32</code> counts at <code>(c * K + from) * K + to</code> for <code>t = 0 .. T - lag - 1</code>
(integer atomics, independent of order). <code>probabilities</code> are the row-normalized floats, 0 for empty
rows. <code>rowTotals</code> and <code>ignored</code> are explain columns.</li>
<li>Optional diagnostics (also available on their own as <code>GPUTransitionDiagnostics</code>, taking the <code>counts</code>
tensor): <code>output.homogeneity</code> (needs <code>conditionCount > 1</code>; five rows <code>Q</code>, <code>Q_p_value</code>, <code>LR</code>,
<code>LR_p_value</code>, <code>dof</code>, indexed by <code>GPU_TRANSITION_HOMOGENEITY</code>), <code>output.steadyStates</code> (ergodic
distribution of the pooled chain, then one per condition, <code>chains * K</code> rows), <code>output.stateSparsity</code>
(two rows, <code>GPU_TRANSITION_STATE_SPARSITY</code>: states with 0 &#x3C; n &#x3C; <code>minimumTransitions</code>, and states with
n = 0) and <code>output.status.converged</code>. <code>minimumTransitions</code> defaults to 5. The steady-state solver keeps
its elimination matrix in a storage scratch buffer the class allocates (<code>chains * K^2</code> floats), so 64
classes fit the 8,192-byte private memory limit of core WebGPU.</li>
<li>Homogeneity follows giddy's <code>homogeneity</code> (Bickenbach and Bode 2003): Q and LR compare each conditional
matrix with the pooled one and are chi-square with <code>sum_i (b_i - 1)(A_i - 1)</code> degrees of freedom. A
from-state with no transitions adds 0 degrees of freedom; giddy adds 1 for it, so <code>dof</code> differs from
<code>dof_hom</code> by the number of empty pooled rows. The p-values are upper-tail chi-square (<code>'greater'</code>) from
an f32 incomplete gamma (relative error below 6e-5 for p above 1e-6); <code>dof = 0</code> gives NaN. The test is
asymptotic: with sparse rows (see <code>stateSparsity</code>) it is unreliable.</li>
<li>Steady states use a direct Grassmann-Taksar-Heyman solve per chain. States with no transition in or out
are dropped (probability 0, as giddy's <code>Markov</code> drops classes absent from the data); the remaining
states must have an outgoing transition and form one communicating class, otherwise the whole chain is
NaN (giddy raises or returns several distributions). <code>converged</code> is one when every chain that has
transitions got a steady state. The solver is direct, so <code>iterationLimitReached</code> is never written.
A periodic chain is solved (its stationary distribution, not a limit of powers).</li>
</ul>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// \`classCount\` is the one-row uint32 view written by GPUClassBreaks.</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUClassAssignment</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({values, breaks, classCount, output: classes}));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUTransitionMatrix</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({classes, periodCount, classCount: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, output: {counts, probabilities}}));</span></span></code></pre>
<h4 id="how-it-computes-2">How it computes</h4>
<p>{/* lens:GPUTransitionMatrix */}</p>
<p>::lens{scenario=gpu-spatial-analysis.transition-matrix rung=datum datum=0 caption="Cell 0: counties in class 0, the lowest, that are still in class 0 a year later."}</p>
<p>::lens{scenario=gpu-spatial-analysis.transition-matrix rung=formula caption="Each probability is a transition count divided by the count leaving that class."}</p>
<p>::lens{scenario=gpu-spatial-analysis.transition-matrix rung=kernel caption="Inspect the recorded shader and its input and output bindings."}</p>
<p>::lens{scenario=gpu-spatial-analysis.transition-matrix rung=graph caption="Follow the recorded dependencies from uploaded inputs to captured outputs."}</p>
<h4 id="how-it-computes-3">How it computes</h4>
<p>{/* lens:GPUTransitionDiagnostics */}</p>
<p>::lens{scenario=gpu-spatial-analysis.transition-diagnostics rung=datum datum=0 caption="Cell 0: Illinois counties in class 0 that are still in class 0 a year later."}</p>
<p>::lens{scenario=gpu-spatial-analysis.transition-diagnostics rung=formula caption="Q sums squared gaps between each state's counts and the pooled expectation."}</p>
<p>::lens{scenario=gpu-spatial-analysis.transition-diagnostics rung=kernel caption="Inspect the recorded shader and its input and output bindings."}</p>
<p>::lens{scenario=gpu-spatial-analysis.transition-diagnostics rung=graph caption="Follow the recorded dependencies from uploaded inputs to captured outputs."}</p>
<h3 id="gpuspatialmarkov-and-gpulisamarkov"><code>GPUSpatialMarkov</code> and <code>GPULISAMarkov</code></h3>
<p><strong><code>GPUSpatialMarkov</code></strong>: Splits class transitions by the quantile class of the queen-neighbour mean rate and tests whether the conditional matrices differ.</p>
<p>::lens{scenario=gpu-spatial-analysis.spatial-markov rung=result caption="Transition counts for 437 counties split by neighbour lag class: 5 lag classes by 5 by 5 cells."}</p>
<p><strong><code>GPULISAMarkov</code></strong>: Counts yearly moves of counties between local Moran quadrants HH, LH, LL and HL using queen weights.</p>
<p>::lens{scenario=gpu-spatial-analysis.lisa-markov rung=result caption="Transitions among LISA states, 2008 to 2023; without significance gating state 0 stays empty."}</p>
<ul>
<li><code>GPUSpatialMarkov</code> (giddy <code>Spatial_Markov</code>): props <code>values</code> (<code>float32</code>, <code>rowCount * periodCount</code>, period-major),
<code>classes</code> (<code>uint32</code>, same layout, for example from <code>GPUClassAssignment</code>), <code>classCount</code> (<code>K</code>, 1 to 64),
<code>weights</code> (square self-join <code>GPUSpatialWeights</code>, constant across periods), <code>periodCount</code> (at least 2),
optional <code>periodLag</code> (default 1), <code>normalize</code> (default true), <code>mask</code>, <code>maximumLagClassCount</code>,
<code>lagParameters</code> (<code>getGPUClassBreaksParameterValues</code> view), optional <code>lagMethods</code> (default
<code>['quantile']</code>) and an <code>output</code> of <code>counts</code>, <code>lagBreaks</code>, <code>lagClassCount</code> and the optional
<code>probabilities</code>, <code>rowTotals</code>, <code>ignored</code>, <code>lagValues</code> and <code>lagClasses</code>. For each period it runs <code>GPUSpatialLag</code> (row-standardized by default,
<code>normalize</code>), classifies the lag with one pooled <code>GPUClassBreaks</code>, assigns the lag classes, and builds a
transition matrix conditioned on the lag class at the start period. A lag exactly equal to a lag edge goes to
the upper lag class by default; optional <code>lagIntervalClosure: 'right'</code> keeps it in the lower one, as giddy
does (mapclassify bins are right-closed). <code>counts</code> is
<code>maximumLagClassCount x K x K</code>; slices past the produced <code>lagClassCount</code> stay zero. The <code>output</code> also
takes the <code>GPUTransitionMatrix</code> diagnostics (<code>homogeneity</code>, <code>steadyStates</code>, <code>stateSparsity</code>,
<code>status.converged</code>) and <code>minimumTransitions</code>: <code>homogeneity</code> is giddy's <code>Q</code>, <code>Q_p_value</code>, <code>LR</code>,
<code>LR_p_value</code>, <code>dof_hom</code> across the lag classes; <code>steadyStates</code> is giddy's <code>s</code> (rows <code>0..K-1</code>, pooled) and
<code>S</code> (rows <code>(1 + c) * K ..</code>), NaN where a chain is not ergodic. The pooled chain here sums the counted
transitions over the lag classes, so it excludes rows without a lag class. A small <code>Q_p_value</code> says the
spatial lag matters; a large one says the conditional matrices cannot be told apart from the pooled one.
<code>getInferenceMetadata()</code> describes the tests (analytic, <code>'greater'</code>, no correction).</li>
<li><code>GPULISAMarkov</code> (giddy <code>LISA_Markov</code>): props <code>values</code>, <code>weights</code>, <code>periodCount</code> (at least <code>periodLag + 1</code>),
optional <code>periodLag</code>, <code>parameters</code> (the <code>GPULocalMoran</code> view from
<code>getGPUSpatialAutocorrelationParameterValues</code>, which carries the significance level), optional <code>mask</code>,
<code>gating</code> and an <code>output</code> of <code>counts</code> and the optional <code>probabilities</code>, <code>rowTotals</code>, <code>ignored</code>,
<code>steadyStates</code>, <code>stateSparsity</code> and <code>quadrants</code>. It packs the values into a 64-aligned period stride, so
every period compiles the same pipelines, runs <code>GPULocalMoran</code> per period and counts moves
between the five states (<code>GPU_LISA_MARKOV_STATE_COUNT</code>: 0 not significant, 1 HH, 2 LH, 3 LL, 4 HL). The
layout is the same for both <code>gating</code> modes, so a legend never changes with the option.
<ul>
<li><code>gating: 'none'</code> (default, giddy <code>LISA_Markov</code> parity): every valid row is in its quadrant, the signs of
the centered value and of the lag; state 0 stays empty and the chain is a four-state chain. A row at
exactly the mean or with an exactly zero lag has no quadrant (esda would call it low) and is ignored.</li>
<li><code>gating: 'significance'</code>: rows whose analytic local Moran p-value is above the per-frame significance
level are state 0. The test is the per-period analytic z at an uncorrected level, not esda
permutations, and each period is gated on its own, so a transition into or out of state 0 can be a
threshold crossing rather than a change of the local pattern. Treat it as a map of significant
clusters, not as a test of dynamics. <code>getInferenceMetadata(settings)</code> takes the same
<code>GPUSpatialAutocorrelationSettings</code> packed into <code>parameters</code>, describes the test and returns
<code>undefined</code> for <code>'none'</code>.</li>
<li>Quadrant numbering is esda's default (<code>geoda_quads=False</code>): 1 HH, 2 LH, 3 LL, 4 HL. With
<code>geoda_quads=True</code> esda swaps 2 and 3 (1 HH, 2 LL, 3 LH, 4 HL).</li>
<li>Missing data: a row-period that is masked, non-finite, or has no neighbor among the finite unmasked
rows of that period is not a state. It is written as <code>0xffffffff</code> (<code>GPU_CLASS_ASSIGNMENT_NO_CLASS</code>) in
<code>quadrants</code>, and every transition with such an end is counted in <code>ignored</code>, never as state 0. Each
period excludes its invalid rows from the mean and the neighbor sums (<code>islands: 'exclude'</code>).
giddy has no missing-data handling; parity is on the complete rows. <code>quadrants</code> (<code>rowCount * periodCount</code>)
is optional.</li>
</ul>
</li>
</ul>
<p>Both emit one node chain per period, which suits periods in the tens. giddy <code>relative</code> rescaling, mean first
passage times and the permutation test <code>x2_rpvalue</code> are not built in: divide each period before
classifying.</p>
<h4 id="how-it-computes-4">How it computes</h4>
<p>{/* lens:GPUSpatialMarkov */}</p>
<p>::lens{scenario=gpu-spatial-analysis.spatial-markov rung=datum datum=0 caption="Cell 0: lag class 0, class 0 staying in class 0 a year later."}</p>
<p>::lens{scenario=gpu-spatial-analysis.spatial-markov rung=formula caption="Each conditional probability divides a count by the transitions out of its lag class and origin class."}</p>
<p>::lens{scenario=gpu-spatial-analysis.spatial-markov rung=kernel caption="Inspect the recorded shader and its input and output bindings."}</p>
<p>::lens{scenario=gpu-spatial-analysis.spatial-markov rung=graph caption="Follow the recorded dependencies from uploaded inputs to captured outputs."}</p>
<h4 id="how-it-computes-5">How it computes</h4>
<p>{/* lens:GPULISAMarkov */}</p>
<p>::lens{scenario=gpu-spatial-analysis.lisa-markov rung=datum datum=6 caption="Cell 6: a county in the high-high quadrant that is still high-high a year later."}</p>
<p>::lens{scenario=gpu-spatial-analysis.lisa-markov rung=formula caption="Probabilities divide state-to-state counts by the transitions leaving each quadrant."}</p>
<p>::lens{scenario=gpu-spatial-analysis.lisa-markov rung=kernel caption="Inspect the recorded shader and its input and output bindings."}</p>
<p>::lens{scenario=gpu-spatial-analysis.lisa-markov rung=graph caption="Follow the recorded dependencies from uploaded inputs to captured outputs."}</p>
<p>{/* parameters: GPULISAMarkov generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpulisamarkov-parameters"><code>GPULISAMarkov</code> parameters</h4>
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
<h3 id="gpuknoxtest-and-gpumanteltest"><code>GPUKnoxTest</code> and <code>GPUMantelTest</code></h3>
<p><strong><code>GPUKnoxTest</code></strong>: Counts observation pairs within 250 m and 7 days of each other and compares the count with 199 time permutations.</p>
<p>::lens{scenario=gpu-spatial-analysis.knox-test rung=result caption="2,344 observations; the strip compares the observed pair count (row 0) with 199 permuted counts."}</p>
<p><strong><code>GPUMantelTest</code></strong>: Correlates pairwise distance with pairwise day difference over 293 sampled observations and compares r with 199 permutations.</p>
<p>::lens{scenario=gpu-spatial-analysis.mantel-test rung=result caption="293 sampled observations (every 8th); row 0 of the strip is the observed correlation r."}</p>
<p>Space-time interaction tests over the pairs of a self-join <code>GPUSpatialWeights</code> (typically radius-mode
<code>GPUNeighborSearch</code>): every unordered pair listed in either direction is a spatial pair and is counted
exactly once (a symmetric list takes it from the lower row, a pair listed only from the higher row, as in a
kNN list, is taken from that row; duplicate and self entries are ignored; lists need not be sorted). Results
are only as complete as the pair list. Pass <code>GPUNeighborSearch.outputWeights</code> so both loss causes
reach inference rather than rebuilding the CSR views without their status.
Pairs carry their own <code>spatialContext</code>; Knox needs no context check because it only counts spatial pairs.
Mantel needs no context check because its correlation is unit-free and it takes no spatial distance parameter.</p>
<p>Knox, Mantel and Jacquez capture declared input/output maps, including inherited/getter-backed
fields, once at construction. Optional-port declarations and destinations stay fixed; unrelated
output properties are ignored. They read supplied <code>status.overflow</code> and <code>candidateOverflow</code>
contents on every submission. Either nonzero flag makes the float summary unavailable (all NaN);
float statistics are NaN, integer statistics and an optional permutations-used word are zero.
Clearing the loss flags with complete inputs restores the result on the same graph.</p>
<ul>
<li><code>GPUKnoxTest({pairs, times, parameters, maximumPermutations, output: {statistics, summary}})</code> counts the pairs
that are close in space and have <code>|t_i - t_j| &#x3C;= timeThreshold</code>, then runs a Monte Carlo (modified Knox)
test over <code>P</code> permuted time assignments. Counts are exact integers, and the expected value comes
from the exact count of time-close pairs. <code>output.statistics</code> is <code>uint32</code> with <code>P + 1</code> entries. Knox's
hypothesis is one-sided (an excess of close pairs), so the Knox p-value is <code>pseudoPGreater</code>.
<code>getInferenceMetadata(settings)</code> takes the same <code>GPUSpaceTimeSettings</code> that are packed into
<code>parameters</code> and returns CPU-side <code>GPUSpaceTimeInferenceMetadata</code> (alternative <code>'greater'</code>, Monte Carlo,
the clamped <code>requestedPermutationCount</code>, stopping rule and seed, no multiple-testing correction).
<code>getKnoxPoissonPValue(observed, expected)</code> gives the classic Poisson
upper tail, but it assumes Poisson variance, ignores ties and ignores the Mantel-Kulldorff variance
inflation, so it is anti-conservative; prefer the Monte Carlo p-value.</li>
<li><code>GPUMantelTest({pairs, ...})</code> (the weights need <code>distances</code>) is the Pearson correlation between pair
distances and <code>|t_i - t_j|</code>, with permuted times. By default the raw values are correlated (the
scikit-bio <code>mantel</code> definition). pointpats <code>mantel</code> instead correlates <code>(x + constant) ** power</code> of both
and its defaults are the reciprocals <code>1 / (x + 1)</code> (<code>scon = tcon = 1</code>, <code>spow = tpow = -1</code>, pointpats 2.6.0),
so a default pointpats result differs from the raw one. Pass the compile-time <code>distanceTransform</code> and
<code>timeTransform</code> (<code>{constant, power}</code>, or <code>GPU_MANTEL_POINTPATS_TRANSFORM</code> for the pointpats defaults) to
match it. The time transform acts on <code>|t_i - t_j|</code>, so it cannot be done by transforming <code>times</code> first.
The base of <code>x + constant</code> must stay positive for a non-integer power. Sums are f32 shifted partials, within about 1e-4 of a
double-precision oracle. Above about 43,000 permutations the rows are split into fewer than 256 blocks so
the <code>(permutation, block)</code> partials table stays within one storage binding (<code>maxStorageBufferBindingSize</code>). The statistic runs over the listed pairs only, so with a distance band it is a
band-truncated Mantel <code>r</code> (pairs beyond the band are absent) that differs from the full Mantel <code>r</code>;
report the band with the result. <code>getInferenceMetadata(settings)</code> states alternative <code>'greater'</code>: the interaction
hypothesis is a positive space-time correlation, the upper-tail convention of pointpats <code>mantel</code>
(<code>pseudoPTwoSided</code> is also in the summary if repulsion matters).</li>
</ul>
<p><code>timeThreshold</code>, <code>seed</code> and <code>permutationCount</code> are per-frame settings (<code>GPUSpaceTimeSettings</code>), written with
<code>getGPUSpaceTimeParameterValues({seed, permutationCount, timeThreshold})</code> (<code>uint32[4]</code>). <code>times</code> must be
finite and are float32, so raw epoch seconds (about 1.7e9) are spaced 128 s apart and epoch milliseconds
are unusable. Subtract an origin on the CPU with <code>getGPUSpaceTimeRelativeTimes(times, origin?, requiredResolution?)</code>, which subtracts a float64 origin (default the minimum), returns
<code>{times, origin, resolution}</code> and throws if the span is too coarse for <code>requiredResolution</code>; the float32
spacing at a span <code>S</code> is <code>2^(floor(log2 S) - 23)</code> (about 0.5 ms for an hour, 1 s for 4.6 months of
seconds). <code>timeThreshold</code> is in the same relative units. <code>output.summary</code> (<code>float32</code>, 12 entries, <code>GPU_SPACE_TIME_SUMMARY</code>) holds the observed statistic, pair
count, time-close pair count, expected value, permutation mean and variance, greater and lesser counts,
the pseudo p-values (greater, lesser, two-sided, <code>(1 + count) / (P + 1)</code>) and the z-score. The Kulldorff
space-time scan is <code>GPUSpatialScanStatistic</code> with <code>timeBuckets > 1</code>.</p>
<p>Both tests permute the times with the keyed swap-or-not shuffle of <code>GPUGlobalPermutationTest</code>, which is
uniform on small domains (permutation 0 is the identity, permutation <code>k</code> is slot <code>k</code>). Permuted counts and
p-values for a given seed differ from earlier versions, which used a Feistel bijection that was measurably
non-uniform on small domains. At most <code>2^24 - 1</code> events are supported.</p>
<p><strong>Sequential stopping.</strong> The compile-time <code>stopAfterExceedances</code> prop (an integer <code>h</code> in <code>[1, maximumPermutations]</code>, default omitted) enables Besag-Clifford sequential Monte Carlo p-values
(Besag and Clifford 1991) for both tests, with the greater alternative that <code>pseudoPGreater</code> reports. The
run stops at the first permutation <code>L</code> whose running exceedance count (permuted statistic <code>>=</code> observed)
equals <code>h</code>, and <code>pseudoPGreater = min(1, h / L)</code>; with fewer than <code>h</code> exceedances in all <code>P</code> it is the usual
<code>(count + 1) / (P + 1)</code>. Omitted, nothing changes. Permutations run in batches of 64 slots (the batch doubles
when <code>maximumPermutations</code> exceeds 16,383, so there are at most 256 batches; batch 0 also computes the
observed statistic). After each batch a one-thread node scans its statistics in permutation order and
disarms the remaining batches through a GPU dispatch gate, so <code>L</code> and the p-value do not depend on the batch
size, and the first <code>L</code> permutations are those of the full run (same keyed shuffle stream). Work saved is
the batches after the one holding <code>L</code>, and the batches past the per-frame <code>permutationCount</code>; the stopping
batch finishes whole, so expect savings in steps of one batch, and each batch is a few more graph nodes
(<code>batch-&#x3C;n>-partials</code>, <code>batch-&#x3C;n>-reduce</code>, <code>batch-&#x3C;n>-scan</code>, between <code>stop-reset</code> and <code>stop-finalize</code>).
<code>output.permutationCount</code> (one <code>uint32</code>, required with <code>stopAfterExceedances</code> and rejected without it)
receives <code>L</code>, or <code>P</code> when the rule did not fire. <code>output.statistics</code> rows past <code>L</code> are 0, and the summary
permutation mean, variance, greater and lesser counts and z-score cover the first <code>L</code> permutations.
<code>pseudoPLesser</code> and <code>pseudoPTwoSided</code> keep the fixed-length formulas over those <code>L</code> permutations and are not
sequentially valid; use <code>pseudoPGreater</code>. The <code>parameters</code> layout is unchanged, and <code>h</code> is baked into the kernels, so a different <code>h</code>
is a rebuild.</p>
<p>The decoder represents suppressed zero-draw inference with <code>available: false, permutationCount: 0</code>.
Positive counts still obey the fixed-count/stopping lower bounds and return <code>available: true</code>.
The Poisson helper returns NaN for nonfinite observed/expected values. Readouts must show unavailable
and omit the permutation distribution until a complete frame arrives.</p>
<p>For Knox, Mantel and Jacquez, <code>getInferenceMetadata(settings)</code> reports <code>requestedPermutationCount</code>
after clamping to <code>maximumPermutations</code>, and <code>stoppingRule</code> is <code>{method: 'fixed-count'}</code> or
<code>{method: 'besag-clifford', stopAfterExceedances: h}</code>. The settings-derived <code>permutationCount</code>
is present only with fixed-count inference; it is omitted when stopping is enabled. After reading
<code>output.permutationCount</code>, call <code>decodeGPUSpaceTimeInferenceMetadata(metadata, permutationCount)</code>
to build the final methods record. The decoder performs no readback and returns a <code>GPUInferenceMetadata</code>
whose <code>permutationCount</code> is the actual number used. For example, requesting 150 permutations with
<code>maximumPermutations: 191</code> and stopping after 10 ties gives <code>requestedPermutationCount: 150</code> and
final <code>permutationCount: 10</code>. The same decoder accepts the count from <code>computeMantelTestOnCPU</code>.
The decoder rejects counts below <code>min(requestedPermutationCount, stopAfterExceedances)</code> for sequential
inference. When the request is below the stopping threshold, all requested draws must run.
Contributors snapshot the compile-time maximum and stopping threshold at construction; later mutation
of the caller's props cannot change either the generated stopping rule or its metadata.</p>
<h4 id="how-it-computes-6">How it computes</h4>
<p>{/* lens:GPUKnoxTest */}</p>
<p>::lens{scenario=gpu-spatial-analysis.knox-test rung=datum datum=0 caption="Row 0: the observed Knox count, before any permutation of days."}</p>
<p>::lens{scenario=gpu-spatial-analysis.knox-test rung=formula caption="The p-value is one plus the permutations at or above the observed count, over P plus one."}</p>
<p>::lens{scenario=gpu-spatial-analysis.knox-test rung=kernel caption="Inspect the recorded shader and its input and output bindings."}</p>
<p>::lens{scenario=gpu-spatial-analysis.knox-test rung=graph caption="Follow the recorded dependencies from uploaded inputs to captured outputs."}</p>
<h4 id="how-it-computes-7">How it computes</h4>
<p>{/* lens:GPUMantelTest */}</p>
<p>::lens{scenario=gpu-spatial-analysis.mantel-test rung=datum datum=0 caption="Row 0: the observed correlation r, before any permutation of days."}</p>
<p>::lens{scenario=gpu-spatial-analysis.mantel-test rung=formula caption="The correlation runs over all listed pairs; p counts permutations at or above the observed r."}</p>
<p>::lens{scenario=gpu-spatial-analysis.mantel-test rung=kernel caption="Inspect the recorded shader and its input and output bindings."}</p>
<p>::lens{scenario=gpu-spatial-analysis.mantel-test rung=graph caption="Follow the recorded dependencies from uploaded inputs to captured outputs."}</p>
<p>{/* parameters: GPUKnoxTest generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpuknoxtest-parameters"><code>GPUKnoxTest</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUSpaceTimeParameterValues(settings)</code>, writes <code>GPU_SPACE_TIME_PARAMETER_LENGTH</code> (4) <code>uint32</code> values.
<ul>
<li>Input port: <code>parameters</code>.</li>
</ul>
</li>
</ul>
<p>Settings of <code>getGPUSpaceTimeParameterValues</code>:</p>
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
<td>Random seed, an integer in <code>[0, 2^53)</code>. The result is a pure function of the seed and inputs.</td>
</tr>
<tr>
<td><code>permutationCount</code></td>
<td><code>number</code></td>
<td>Number of time permutations <code>P</code>, clamped to the contributor's <code>maximumPermutations</code>.</td>
</tr>
<tr>
<td><code>timeThreshold</code> (optional)</td>
<td><code>number</code></td>
<td>Knox only: two rows are close in time when <code>|t_i - t_j| &#x3C;= timeThreshold</code> (f32), in the same units as the relative <code>times</code> (see <code>getGPUSpaceTimeRelativeTimes</code>). Ignored by <code>GPUMantelTest</code>.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<p>{/* parameters: GPUMantelTest generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpumanteltest-parameters"><code>GPUMantelTest</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUSpaceTimeParameterValues(settings)</code>, writes <code>GPU_SPACE_TIME_PARAMETER_LENGTH</code> (4) <code>uint32</code> values.
<ul>
<li>Input port: <code>parameters</code>.</li>
</ul>
</li>
</ul>
<p>Settings of <code>getGPUSpaceTimeParameterValues</code>:</p>
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
<td>Random seed, an integer in <code>[0, 2^53)</code>. The result is a pure function of the seed and inputs.</td>
</tr>
<tr>
<td><code>permutationCount</code></td>
<td><code>number</code></td>
<td>Number of time permutations <code>P</code>, clamped to the contributor's <code>maximumPermutations</code>.</td>
</tr>
<tr>
<td><code>timeThreshold</code> (optional)</td>
<td><code>number</code></td>
<td>Knox only: two rows are close in time when <code>|t_i - t_j| &#x3C;= timeThreshold</code> (f32), in the same units as the relative <code>times</code> (see <code>getGPUSpaceTimeRelativeTimes</code>). Ignored by <code>GPUMantelTest</code>.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="gpunearrepeatknox-gpulocalknox-and-gpujacqueztest"><code>GPUNearRepeatKnox</code>, <code>GPULocalKnox</code> and <code>GPUJacquezTest</code></h3>
<p>Three more space-time interaction tests on the machinery of <code>GPUKnoxTest</code>: the same swap-or-not time
permutations (a seed gives the same permutation <code>k</code> in all four), the same exact integer counts, and, for
the Jacquez test, the same Besag-Clifford stopping. None has a lens scenario yet.</p>
<ul>
<li><code>GPUNearRepeatKnox({pairs, times, parameters, distanceEdges, timeEdges, maximumPermutations, output: {counts, summary}})</code> is the near-repeat table of crime science: the number of event pairs in every cell
of distance band by time-difference band, and a Monte Carlo p-value per cell. <code>pairs</code> is a
<code>GPUSpatialWeights</code> <strong>with <code>distances</code></strong> that lists every pair up to the last distance edge (a radius-mode
<code>GPUNeighborSearch</code>), counted once per unordered pair as in <code>GPUKnoxTest</code>. Edges are compile-time,
strictly ascending, at most 16 bands per axis (256 cells). Band <code>k</code> is <code>(edges[k], edges[k + 1]]</code> and the
first band also holds <code>edges[0]</code>: the <code>&#x3C;=</code> of <code>GPUKnoxTest</code> and of pointpats, so a pair on an edge falls
in the lower band. The NearRepeat R package cuts with R's <code>cut()</code>; if its <code>right</code> and <code>include.lowest</code>
settings differ, only pairs exactly on an edge move cell (UNVERIFIED against that package, which is GPL
and used here as a definition only). <code>distanceEdges</code> are in the length unit of <code>pairs.distances</code>,
<code>timeEdges</code> in the unit of <code>times</code>. <code>output.counts</code> is <code>uint32</code>, <code>(maximumPermutations + 1) * cells</code>,
<code>counts[slot * cells + distanceBand * timeBands + timeBand]</code>, slot 0 the observed table. <code>output.summary</code>
holds <code>GPU_NEAR_REPEAT_CELL_SUMMARY_LENGTH</code> (8) words per cell, indexed by <code>GPU_NEAR_REPEAT_CELL_SUMMARY</code>:
observed, permutation mean and variance, greater and lesser counts, <code>pseudoPGreater</code> and <code>pseudoPLesser</code>,
and the Knox ratio <code>observed / mean</code> (NaN when the mean is 0). The NearRepeat ratio against the median is
not computed. A permutation is one workgroup adding into a workgroup histogram with integer atomics, so
counts are exact and deterministic, and a table with one cell equals <code>GPUKnoxTest</code>. There is no
Besag-Clifford stopping: cells stop at different permutations. The cells are separate one-sided tests with
no multiple-testing correction, which <code>getInferenceMetadata(settings)</code> states; correct on the CPU.</li>
<li><code>GPULocalKnox({pairs, times, parameters, maximumPermutations, output: {summary}})</code> is pointpats
<code>KnoxLocal</code>: for every event, the number of its spatial neighbors with <code>|t_i - t_j| &#x3C;= timeThreshold</code>, with
<strong>conditional</strong> permutation inference as in pointpats: the focal event keeps its own time and the other
times are permuted (the neighbor mapped onto the focal event takes the time of the event the focal event
is mapped to). <code>pairs</code> must list both directions (a radius-mode search); the count of event <code>i</code> runs over
its own row, ignoring self entries, out-of-range IDs and repeats. <code>output.summary</code> has
<code>GPU_LOCAL_KNOX_SUMMARY_LENGTH</code> (6) words per event, indexed by <code>GPU_LOCAL_KNOX_SUMMARY</code>: observed,
permutation mean, greater and lesser counts, <code>pseudoPGreater</code> and <code>pseudoPLesser</code>. The p-values are one
test per event with no correction. There is no stopping (events stop at different permutations). The sum
of the permuted counts per event is a <code>uint32</code>: keep <code>maximumPermutations</code> times the longest row below
2^32. The pointpats analytic hypergeometric p-values and the lead/lag orientation are not computed.</li>
<li><code>GPUJacquezTest({neighbors, times, parameters, k, maximumPermutations, stopAfterExceedances?, output: {statistics, summary, permutationCount?}})</code> is the Jacquez (1996) k-nearest-neighbor test, pointpats
<code>jacquez</code>: the number of (event, spatial neighbor) entries where the neighbor is also among the <code>k</code> nearest
events of the event in time, summed over the directed lists (a mutual pair counts twice). <code>neighbors</code> is a
k-nearest-neighbor <code>GPUSpatialWeights</code> (<code>GPUNeighborSearch</code> in k-nearest mode); use the same <code>k</code>.
Temporal neighbors come from a stable GPU sort of the times: the <code>k</code> nearest in time of an event are the
interval of <code>k + 1</code> ranks around its own rank, so membership under a permutation is one rank comparison.
Ties in the time distance take the lower rank, which can differ from another library's arbitrary choice
for tied times; without ties the set is exactly the <code>k</code> nearest. The reference distribution permutes the
times with the spatial lists fixed; the null mean is <code>sum_i |N_i| k / (n - 1)</code>. <code>output.statistics</code>,
<code>output.summary</code> (<code>GPU_SPACE_TIME_SUMMARY</code>, where <code>pairCount</code> is the number of listed neighbor entries,
<code>timeClosePairs</code> is <code>rows * k</code> and <code>expected</code> is <code>pairCount * k / (rows - 1)</code>) and Besag-Clifford
<code>stopAfterExceedances</code> work as for <code>GPUKnoxTest</code>. The test is known to react to a shift of the population
over time (Malizia and Mack 2012).</li>
</ul>
<p>All three accept <code>temporalContext</code> where the times are taken: a <code>float32-relative</code> <code>GPUTemporalContext</code>
with a declared (bigint) epoch, validated at construction and exposed as <code>.temporalContext</code>. Nothing is
converted, and <code>int64-words</code> or <code>double-single</code> is refused. Build the relative times with
<code>getGPUSpaceTimeGuardedRelativeTimes(times, narrowestTimeWidth, {origin?, temporalContext?})</code>, which is
<code>getGPUSpaceTimeRelativeTimes</code> with its resolution guard set to a quarter of the narrowest time band (or the
time threshold), so it throws rather than blur a band, and which returns the context of the float32 times
(unit kept, epoch moved to the origin; the default origin is the minimum rounded down to a whole tick).</p>
<p>References: pointpats 2.6.0 (BSD-3) <code>Knox</code>, <code>KnoxLocal</code> and <code>jacquez</code> define the statistics and made the
checked-in fixture <code>modules/analysis/test/oracles/fixtures/space-time-near-repeat.json</code> (generator
<code>space-time-near-repeat.py</code>; the table cells are inclusion-exclusion of <code>Knox</code> runs, since pointpats has no
band table). The specs match the fixture's observed counts exactly and compare pointpats' own permutation
draws statistically, because the permutations differ. They also match a double-precision CPU oracle for
every permutation exactly.</p>
<p>{/* lens:GPUNearRepeatKnox */}</p>
<p>{/* lens:GPULocalKnox */}</p>
<p>{/* lens:GPUJacquezTest */}</p>
<p>{/* parameters: GPUJacquezTest generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpujacqueztest-parameters"><code>GPUJacquezTest</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUSpaceTimeParameterValues(settings)</code>, writes <code>GPU_SPACE_TIME_PARAMETER_LENGTH</code> (4) <code>uint32</code> values.
<ul>
<li>Input port: <code>parameters</code>.</li>
</ul>
</li>
</ul>
<p>Settings of <code>getGPUSpaceTimeParameterValues</code>:</p>
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
<td>Random seed, an integer in <code>[0, 2^53)</code>. The result is a pure function of the seed and inputs.</td>
</tr>
<tr>
<td><code>permutationCount</code></td>
<td><code>number</code></td>
<td>Number of time permutations <code>P</code>, clamped to the contributor's <code>maximumPermutations</code>.</td>
</tr>
<tr>
<td><code>timeThreshold</code> (optional)</td>
<td><code>number</code></td>
<td>Knox only: two rows are close in time when <code>|t_i - t_j| &#x3C;= timeThreshold</code> (f32), in the same units as the relative <code>times</code> (see <code>getGPUSpaceTimeRelativeTimes</code>). Ignored by <code>GPUMantelTest</code>.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<p>{/* parameters: GPULocalKnox generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpulocalknox-parameters"><code>GPULocalKnox</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUSpaceTimeParameterValues(settings)</code>, writes <code>GPU_SPACE_TIME_PARAMETER_LENGTH</code> (4) <code>uint32</code> values.
<ul>
<li>Input port: <code>parameters</code>.</li>
</ul>
</li>
</ul>
<p>Settings of <code>getGPUSpaceTimeParameterValues</code>:</p>
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
<td>Random seed, an integer in <code>[0, 2^53)</code>. The result is a pure function of the seed and inputs.</td>
</tr>
<tr>
<td><code>permutationCount</code></td>
<td><code>number</code></td>
<td>Number of time permutations <code>P</code>, clamped to the contributor's <code>maximumPermutations</code>.</td>
</tr>
<tr>
<td><code>timeThreshold</code> (optional)</td>
<td><code>number</code></td>
<td>Knox only: two rows are close in time when <code>|t_i - t_j| &#x3C;= timeThreshold</code> (f32), in the same units as the relative <code>times</code> (see <code>getGPUSpaceTimeRelativeTimes</code>). Ignored by <code>GPUMantelTest</code>.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<p>{/* parameters: GPUNearRepeatKnox generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpunearrepeatknox-parameters"><code>GPUNearRepeatKnox</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUSpaceTimeParameterValues(settings)</code>, writes <code>GPU_SPACE_TIME_PARAMETER_LENGTH</code> (4) <code>uint32</code> values.
<ul>
<li>Input port: <code>parameters</code>.</li>
</ul>
</li>
</ul>
<p>Settings of <code>getGPUSpaceTimeParameterValues</code>:</p>
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
<td>Random seed, an integer in <code>[0, 2^53)</code>. The result is a pure function of the seed and inputs.</td>
</tr>
<tr>
<td><code>permutationCount</code></td>
<td><code>number</code></td>
<td>Number of time permutations <code>P</code>, clamped to the contributor's <code>maximumPermutations</code>.</td>
</tr>
<tr>
<td><code>timeThreshold</code> (optional)</td>
<td><code>number</code></td>
<td>Knox only: two rows are close in time when <code>|t_i - t_j| &#x3C;= timeThreshold</code> (f32), in the same units as the relative <code>times</code> (see <code>getGPUSpaceTimeRelativeTimes</code>). Ignored by <code>GPUMantelTest</code>.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="gpuspatialscanstatistic"><code>GPUSpatialScanStatistic</code></h3>
<p>Finds circular windows of counties with more traffic deaths than their person-years imply, with Monte Carlo p-values.</p>
<p>::lens{scenario=gpu-spatial-analysis.spatial-scan-statistic rung=result caption="437 counties coloured by the best window log-likelihood ratio centred on each; 199 replicates, seed 1."}</p>
<p>Kulldorff's scan statistic (SaTScan, CARTO <code>DETECT_SPACETIME_ANOMALIES</code>) over zones, which are cells or
points, with <code>cases</code> (<code>uint32</code>) per zone and optional time bucket (<code>zone * timeBuckets + bucket</code>). The
compile-time <code>model</code> selects the null:</p>
<ul>
<li><code>'poisson'</code> (default): needs <code>baseline</code> (population or expected counts); <code>E[cases] = baseline * C / sum(baseline)</code>. LLR <code>c ln(c/e) + (C - c) ln((C - c)/(C - e))</code>, evaluated per cell as
<code>e ((1 + d) ln(1 + d) - d)</code> with <code>d = (o - e) / e</code> (a series for small <code>d</code>; the linear parts cancel
exactly), so windows with <code>c</code> close to <code>e</code> keep f32 precision.</li>
<li><code>'bernoulli'</code>: needs <code>controls</code> (<code>uint32</code>, same layout); the population of a cell is <code>cases + controls</code>.
LLR of Kulldorff (1997), computed as the sum of <code>o ln(o/e)</code> over the inside/outside x case/control
table. Monte Carlo permutes the case labels over the fixed population (conditioning on the total
cases), by drawing individuals without replacement. Window populations are f32 (exact below 2^24).</li>
<li><code>'space-time-permutation'</code> (Kulldorff et al. 2005): no baseline or population. The expectation of a
cylinder is <code>(sum of its zones' totals over all buckets) * (sum of its buckets' totals over all zones) / C</code>, the Poisson LLR is applied to it, and Monte Carlo permutes the case times over the case locations
(both margins fixed). A trend or seasonal cycle shared by every zone is therefore not a cluster, which
the Poisson model with a static population would report as an "emerging cluster". Needs
<code>timeBuckets > 1</code> to be meaningful.</li>
</ul>
<p><code>spatialContext</code> is optional and defaults to native planar coordinates. Longitude/latitude positions
require metric <code>'great-circle'</code>: windows are ranked by great-circle distance using the shared haversine
separation key, rather than distances in degrees. Cluster <code>radius</code> uses the context's length unit
(meters by default for great-circle; kilometers with <code>units: 'kilometers'</code>), with <code>sphereRadius</code> or the
mean Earth radius in that unit. This operator measures distance, so <code>local-lambert-equal-area</code> frames throw.</p>
<p>Windows and defaults:</p>
<ul>
<li>Spatial: the <code>k</code> nearest zones of every center (the <code>'circle'</code> shape never splits equidistant zones;
<code>'nearest'</code> is plain k-nearest), kept while they hold at most <code>maximumPopulationFraction</code> of the total
population (default 0.5, as in SaTScan). The population share governs: the per-center neighbor
tables hold up to the compile-time <code>maximumWindowZones</code> zones, which defaults to every zone while
<code>zones^2 &#x3C;= 2^24</code> (4096 zones) and otherwise to the largest capacity that fits
(<code>GPU_SPATIAL_SCAN_STATISTIC_MAXIMUM_NEIGHBOR_WORDS</code>; use <code>getGPUSpatialScanStatisticLayout</code> to see it).
Neighbor discovery orders by <code>(separation key, zone row)</code>, with squared distance for planar and
the haversine term for great-circle. Short lists use exact coordinate indexes; larger lists use
exhaustive radix selection and sorting. The route and its work estimate are described below.
A window the capacity cut short while the population fraction would still allow it is never silent:
<code>output.summary</code> word <code>overflow</code> counts the affected centers and the cluster record carries
<code>truncatedByZoneCap</code> (0 or 1). The per-frame <code>maximumWindowZones</code> may lower the capacity.</li>
<li>Temporal: with <code>timeBuckets > 1</code> each window is a cylinder over every run of up to
<code>maximumTimeBuckets</code> buckets. The default is 50% of the study period, <code>max(1, floor(timeBuckets / 2))</code>
(SaTScan's default), no longer all buckets: with a global trend, pure-spatial windows over the whole
period would otherwise dominate.</li>
<li>Direction: per-frame <code>rateDirection</code> <code>'high'</code> (default, <code>c > e</code>), <code>'low'</code> (<code>c &#x3C; e</code>) or <code>'both'</code>.
Windows of the other direction score 0.</li>
<li>Defaults: <code>maximumPopulationFraction</code> 0.5, <code>maximumWindowZones</code> every zone (bounded), <code>maximumTimeBuckets</code>
50% of the study period, <code>windowShape</code> <code>'circle'</code>, <code>rateDirection</code> <code>'high'</code>, <code>maximumClusters</code> 8.</li>
</ul>
<p>Outputs: the most likely cluster plus greedy, non-overlapping secondary clusters (no shared zone, time
ignored, same maximum distribution, as SaTScan), all in the caller-owned <code>output</code> bag. The candidates
for a secondary cluster are the single best window of each center, as in SaTScan: a center whose best
window overlaps an earlier cluster is dropped, and its smaller windows are not reconsidered.
<code>output.clusterIndices</code> (<code>GPU_SPATIAL_SCAN_STATISTIC_CLUSTER_INDEX</code>) and <code>output.clusterStatistics</code>
(<code>GPU_SPATIAL_SCAN_STATISTIC_CLUSTER</code>: LLR, observed, expected, p, radius, ratio,
<code>truncatedByZoneCap</code>); <code>output.statistics</code> (the observed maximum then the maximum of every Monte Carlo
replicate), <code>output.summary</code> (<code>GPU_SPATIAL_SCAN_STATISTIC_SUMMARY</code>: cluster count, total cases,
<code>permutationCount</code>, <code>overflow</code>) and an optional per-zone best LLR <code>output.zoneStatistics</code>.</p>
<p>Inference: <code>p = (1 + #{max_rep >= LLR}) / (P + 1)</code>, a Monte Carlo maximum over every window, so it already
accounts for the search; no further correction applies. <code>getInferenceMetadata(settings)</code> takes the same
<code>GPUSpatialScanStatisticSettings</code> that are packed into <code>parameters</code> and returns the CPU-side
<code>GPUInferenceMetadata</code> (alternative <code>'greater'</code> for high rates, <code>'lesser'</code> for low and <code>'two-sided'</code> for
both, the permutation count clamped to <code>maximumPermutations</code> as the kernels do, the seed,
<code>multipleTesting: 'none'</code>, the tested-window definition).
Poisson replicates are multinomial draws from an integer cumulative table (Philox streams of the
permutation contributors); the Bernoulli and space-time permutation draws run one invocation per
replicate (about <code>C</code> sequential draws, times <code>timeBuckets</code> for the latter). The result is deterministic
(fixed-order reductions, integer atomics, ties to the lowest index). Limits, the seed and the replicate
count are per-frame settings (<code>getGPUSpatialScanStatisticParameterValues(settings)</code>), up to the compile-time
<code>maximumPermutations</code> (at most 99999, SaTScan-style runs).</p>
<p>Replicate batches: the observed data is evaluated once, then the replicates run in <code>batchCount</code> batches
of <code>batchReplicates</code> (<code>R</code>). A batch draws replicates <code>b * R + 1 .. b * R + R</code> into an <code>R</code>-slot count
table, evaluates them and folds each replicate's largest LLR into its row of <code>output.statistics</code>, which
is the running maximum, so no per-replicate table is kept. <code>R</code> is the largest count with
<code>(R + 1) * zones * timeBuckets &#x3C;= 2^25</code> words (the 128 MiB binding limit), at most 4096 and at most
<code>maximumPermutations</code>; both are readable properties of the contributor. Philox streams are keyed by the
global replicate index and every reduction is a fixed-order sum or an order-independent maximum, so the
results do not depend on <code>R</code> and equal the former single pass bit for bit. A batch whose first replicate
is past the per-frame <code>permutationCount</code> is disarmed on the GPU (indirect dispatch gate) and launches
nothing. Each unrolled batch adds four graph nodes (<code>batch-control</code>, <code>batch-clear</code>, the draw kernel and
<code>batch-evaluate</code>) and no pipelines; <code>maximumPermutations</code> may need at most 1024 batches, which is only
binding for very large <code>zones * timeBuckets</code> (above about 340,000 cells, 99999 replicates need <code>R >= 98</code>).</p>
<p>Poisson baselines are scaled by a common power of two before summation, window construction and
replicate draws. The scale is the binary exponent of the largest exposure, preserving exact
binary relationships at population-share boundaries. Maximum reduction reads nonnegative IEEE
words as integers; significands and exponents also decode positive subnormal exposures before
any float arithmetic. Thus finite
nonnegative baselines can have a raw sum above the float32 range.
Scoring and published expected counts both multiply the case total by the window's baseline
fraction, so changing baseline units does not introduce an overflowing intermediate product.
Float32 rounding still applies to baseline shares and sums; an all-zero baseline gives no cluster.
Normalization adds three GPU nodes per Poisson graph, with linear work in the cell count.</p>
<p>Neighbor discovery: compile-time <code>nearestZoneSearch: 'auto'</code> uses exact coordinate indexes when
the list is shorter than the domain and (including the extra lookahead zone) has at most
<code>GPU_SCAN_MAXIMUM_INDEXED_NEIGHBOR_COUNT</code> (65) rows. Two published radix sorts index the coordinates;
each center binary-searches its position and searches outward along the wider planar axis, or
latitude for great-circle coordinates. Conservative axis bounds prune only strictly larger keys,
preserving <code>(separation key, zone row)</code> order, ties, and the lookahead zone across the antimeridian
and at the poles. The index is rebuilt on every submission when positions may change.</p>
<p><code>'indexed'</code> explicitly selects that route and requires a list of at most 65 rows; <code>'full-scan'</code>
selects the exhaustive radix route. Auto uses full-scan for whole-domain or larger lists. The resolved route is
<code>contributor.nearestZoneSearch</code>. <code>contributor.estimatedFullScanSeparationCount</code> is an upper bound
on the exhaustive route's separation evaluations per submission: <code>5 * zones^2</code> for a truncated
list (four selection sweeps and at most one gather sweep), or <code>zones^2</code> when every zone is retained.
This excludes radix sorting. The indexed route adds two linear-size coordinate orders and uses
data-dependent pruning: coincident zones or many equal axis coordinates can still require
quadratic work. It is an exact search, with no candidate truncation or approximation flag.</p>
<p><code>estimatedCommandNodeCount</code> separately bounds discovery sorts, model preparation, all replicate
batches and cluster rounds. The constructor checks it against the shared command-node budget.</p>
<p>Limits: ordinal and normal models, elliptic windows and Gumbel p-values are not provided. The count
table needs <code>(R + 1) * zones * timeBuckets &#x3C;= 2^25</code> words (so <code>zones * timeBuckets &#x3C;= 2^24</code>), and the
cost is about <code>(P + 1) * zones * windowZones * timeBuckets^2</code>, plus neighbor discovery above.
The full-scan route has two sort scratch tables the size of the neighbor tables; the indexed route
instead uses seven <code>zones</code>-word coordinate-key and row-order buffers, plus the published sorts' scratch.
The absolute LLR error is about <code>|c - e|</code> times the relative
f32 rounding of <code>e</code>; <code>baseline</code> must be finite and non-negative. Cluster rounds after the last cluster are
GPU-gated and launch nothing. No kernel binds more than 8 storage buffers, so the scan runs on <code>'core'</code>
WebGPU devices.</p>
<h4 id="how-it-computes-8">How it computes</h4>
<p>{/* lens:GPUSpatialScanStatistic */}</p>
<p>::lens{scenario=gpu-spatial-analysis.spatial-scan-statistic rung=datum datum=15 caption="Cook County, Illinois (row 15 of the county table): the best window centred there."}</p>
<p>::lens{scenario=gpu-spatial-analysis.spatial-scan-statistic rung=formula caption="Poisson log-likelihood ratio of a window; p counts replicate maxima at or above it."}</p>
<p>::lens{scenario=gpu-spatial-analysis.spatial-scan-statistic rung=kernel caption="Inspect the recorded shader and its input and output bindings."}</p>
<p>::lens{scenario=gpu-spatial-analysis.spatial-scan-statistic rung=graph caption="Follow the recorded dependencies from uploaded inputs to captured outputs."}</p>
<p>{/* parameters: GPUSpatialScanStatistic generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpuspatialscanstatistic-parameters"><code>GPUSpatialScanStatistic</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUSpatialScanStatisticParameterValues(settings)</code>, writes <code>GPU_SPATIAL_SCAN_STATISTIC_PARAMETER_LENGTH</code> (8) <code>uint32</code> values.</li>
</ul>
<p>Settings of <code>getGPUSpatialScanStatisticParameterValues</code>:</p>
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
<td>Random seed, an integer in <code>[0, 2^53)</code>. The result is a pure function of seed and inputs.</td>
</tr>
<tr>
<td><code>permutationCount</code></td>
<td><code>number</code></td>
<td>Monte Carlo replicates <code>P</code>, at most the contributor's <code>maximumPermutations</code>.</td>
</tr>
<tr>
<td><code>maximumPopulationFraction</code> (optional)</td>
<td><code>number</code></td>
<td>Largest spatial window as a fraction of the total baseline, in <code>(0, 1]</code>. A window grows until adding the next zone would exceed it. Defaults to 0.5.</td>
</tr>
<tr>
<td><code>maximumWindowZones</code> (optional)</td>
<td><code>number</code></td>
<td>Largest number of zones per window, clamped to the contributor's compile-time capacity. Defaults to the capacity (every zone unless the neighbor tables are bounded), so the population fraction governs.</td>
</tr>
<tr>
<td><code>maximumTimeBuckets</code> (optional)</td>
<td><code>number</code></td>
<td>Longest cylinder in time buckets, clamped to <code>timeBuckets</code>. Defaults to 50% of the study period, <code>max(1, floor(timeBuckets / 2))</code>, as in SaTScan.</td>
</tr>
<tr>
<td><code>windowShape</code> (optional)</td>
<td><code>GPUSpatialScanWindowShape</code></td>
<td>Window shape. Defaults to <code>'circle'</code>.</td>
</tr>
<tr>
<td><code>rateDirection</code> (optional)</td>
<td><code>GPUSpatialScanRateDirection</code></td>
<td>Which anomalies count as clusters. Defaults to <code>'high'</code>.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>`};export{e as default};