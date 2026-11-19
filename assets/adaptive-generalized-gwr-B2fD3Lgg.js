var e={slug:`api/gpu-spatial-analysis/adaptive-generalized-gwr`,title:`Adaptive generalized geographically weighted regression`,html:`<h1 id="adaptive-generalized-geographically-weighted-regression">Adaptive generalized geographically weighted regression</h1>
<p><code>GPUGeneralizedGeographicallyWeightedRegression</code> fits local Poisson log-link or Bernoulli
logit models on GPU. <code>bandwidthMode: 'adaptive'</code> interprets <code>parameters[0]</code> as an integer
neighbor count; the default <code>'fixed'</code> continues to interpret it as a distance. In both modes
<code>parameters[1]</code> is the positive relative coefficient tolerance. <code>maximumNeighborCount</code> is a
compile-time capacity from 1 to 128 (default 128); requested counts must fit this capacity.
The existing planar spatial-context units and Gaussian/bisquare kernels apply.</p>
<p>Adaptive selection counts complete-case rows, including the focus. Masked rows and rows with
nonfinite positions, predictors, response or offsets are excluded before selection, as are
responses outside the family domain. Each focus uses <code>1.00001 * kthDistance</code> as its distance
bandwidth, matching the float32-safe Gaussian GWR convention. All equal-distance boundary ties
participate; neighbor count controls the distance scale rather than imposing an exact support
cardinality. Gaussian support remains unbounded; bisquare support is strictly inside the inflated
radius. Selection is deterministic and independent of response values beyond domain validity.</p>
<p>Too few eligible rows, fewer positive-weight rows than coefficients, singular local designs and
zero kth distance produce status 2. A zero radius is not replaced by a coordinate-unit epsilon.
Invalid/noninteger counts and numerical overflow produce status 4. Excluded focuses have status 1.
All-zero Poisson neighborhoods and single-class logistic neighborhoods have status 5
(<code>GPU_ITERATIVE_REGRESSION_STATUS.DEGENERATE_RESPONSE</code>), because their intercept has no finite
maximum-likelihood solution. This diagnostic is also available in fixed mode. Complete or quasi separation witnessed by an exact column threshold or a decided Newton
direction receives status 6 (<code>GPU_ITERATIVE_REGRESSION_STATUS.SEPARATION</code>). A logistic threshold
certificate uses exact raw-value comparisons of the two response classes' predictor intervals.
A Poisson threshold requires all positive-count rows on one predictor value, with zero-count
rows on one side and at least one strictly beyond it. These certificates avoid rounded dot
products, including for simple quasi-separation.</p>
<p>A general direction must improve at least one positive-weight row without opposing any row:
signed margins for logistic; zero projection on positive-count rows and nonpositive projection
on zero-count rows for Poisson. A zero move is established only when every term is structurally
zero. Other moves must have the required strict sign beyond a conservative float32 rounding
bound <code>(designColumns + 8) * 2^-22 * sum(abs(terms))</code>; this bound rejects undecided signs and
never permits an opposing move. Offsets do not enter the direction test. This is a sufficient
numerical certificate, not an exhaustive linear-programming feasibility test. Uncertified separation can still produce
singularity, numerical failure or iteration-limit status 3. There is no penalty or separation
correction. Status 3 publishes the last finite coefficients and fitted values; other failed/excluded
outputs remain NaN. Status 0 reports convergence. Iteration
counts are zero for selection/preflight failures and count completed IRLS steps otherwise.</p>
<p>This is a reference-tier contributor: dense neighborhood traversal and its dense local solves have
explicit shape limits. Neighbor selection runs entirely on GPU before local IRLS; there is no control readback or CPU fallback. Optional bandwidth search and inference are described below. Dense neighbor selection costs <code>O(rows² * neighborCount)</code> and the existing
fit costs <code>O(rows² * iterations * predictorCount²)</code>. Bounds are 65,536 rows, 1–24 predictors
and 128 iterations (<code>GPU_LOCAL_REGRESSION_MAXIMUM_ROW_COUNT</code>,
<code>GPU_LOCAL_REGRESSION_MAXIMUM_PREDICTOR_COUNT</code>); they were 16,384 rows and 11 predictors. These are
shape limits, not performance promises. The per-focus fit state is tiled over focal rows when it
would exceed one storage binding.</p>
<p>Two opt-in options use the shared local weighted least squares engine that GWR also uses.
<code>indexGridSize: [columns, rows]</code> builds an internal grid index over the training rows, so bisquare
fits (fixed or adaptive bandwidth) visit only the cells covering the bandwidth and the row limit
rises to 1,048,576 (<code>GPU_LOCAL_REGRESSION_MAXIMUM_INDEXED_ROW_COUNT</code>). Adaptive k-nearest selection
walks grid rings around each focus and selects the same neighbors as the dense scan. The Gaussian
kernel uses the grid only with <code>gaussianCutoff</code> (in bandwidth multiples; 5 omits at most about
3.7e-6 of a unit weight); without it the Gaussian stays exact and scans every row. Sums run in cell
order, so results differ from the dense scan in float32 rounding only. A great-circle <code>spatialContext</code>
(<code>{coordinateSpace: 'longitude-latitude', metric: 'great-circle'}</code>) takes <code>(longitude, latitude)</code>
degrees and measures bandwidths in the context units (meters on a sphere of <code>sphereRadius</code>); it is
planar-grid exclusive. The same two options, limits and the engine apply to
<code>GPUMultiscaleGeographicallyWeightedRegression</code>; its full-inference mode keeps its 128-row budget.
Both default to the previous dense, planar behavior.</p>
<p>The committed <code>test/oracles/adaptive-generalized-gwr.py</code> generator composes NumPy order-statistic
distances with statsmodels frequency-weighted GLMs, separately from the GPU Newton solver.
The fixture records package versions and float32 input rounding. Tests compare both families,
both kernels, fixed and adaptive fits on tied grid distances, masks and Poisson offsets, with
absolute tolerance 0.003. Additional cases cover invalid counts, insufficient neighborhoods,
zero-distance ties, nonfinite rows, boundary responses and iteration limits. This is independent
composition evidence, not direct mgwr generalized-regression parity.</p>
<h2 id="training-only-selection-and-model-specific-inference">Training-only selection and model-specific inference</h2>
<p><code>bandwidthSelection</code> accepts 2–32 fixed distances or adaptive neighbor counts, and caller-owned
<code>scores</code>, <code>bandwidth</code> and <code>status</code> outputs. Each candidate is scored by exact leave-one-out mean
Poisson or Bernoulli deviance. Every eligible training focus must converge; failed candidates have
NaN scores. Status is 0 when a finite candidate is selected, or 1 if none succeeds. Equal scores
choose the first supplied candidate. This is the default <code>method: 'grid'</code>, <code>criterion: 'loo-deviance'</code> behavior. For grid AICc,
set <code>criterion: 'aicc'</code>; candidates then include their own focus and use the criterion below. The ordinary <code>parameters[1]</code> coefficient tolerance applies to every inner and final fit.</p>
<p>For an outer spatial fold, pass its training mask. Selection never uses masked-out responses.
<code>predictExcluded: true</code> permits finite excluded predictor/position queries to receive the final
selected model's predictions, without requiring their response values. Adaptive selection counts
only eligible training neighbors; leave-one-out removes the focus from that count. A held-out query's
finite offset remains a caller-supplied prediction input. Rebuild this graph with each outer fold's
training mask and compute held-out metrics afterward.</p>
<p>Optional <code>inference</code> contains <code>standardErrors</code> (focus-major, intercept first) and <code>influence</code> (one
per focus). The scale-one Poisson/logistic covariance uses <code>A^-1 X' V W² X A^-1</code>, with
<code>A = X' V W X</code>, and influence is the diagonal of the local GLM smoother. These are model-specific
local-fit errors, not multiscale backfitting uncertainty. Failed fits retain NaN inference. The direct
mgwr 2.2.1 fixture verifies both families, and a separate exact-LOO fixture verifies candidate scores.</p>
<h2 id="aicc-golden-section-bandwidth-selection">AICc golden-section bandwidth selection</h2>
<p>Set <code>bandwidthSelection.method: 'golden-section'</code>. Its <code>parameters</code> view contains
<code>[lowerBandwidth, upperBandwidth, absoluteScoreTolerance]</code>. Bounds must be positive finite values
with lower strictly below upper; tolerance must be positive finite. In adaptive mode both bounds
must be integers within <code>maximumNeighborCount</code>; interior points round to nearest, ties to even.
Invalid runtime search parameters produce selection status 1, NaN bandwidth/scores, zero search
iterations and neither stopping flag. The ordinary <code>parameters[1]</code> remains the IRLS tolerance;
<code>parameters[0]</code> is replaced internally by each candidate and the final selected bandwidth.</p>
<p><code>maximumIterations</code> inside <code>bandwidthSelection</code> is a compile-time search budget, 1–64, default 24.
It is independent of the contributor's local IRLS <code>maximumIterations</code>. <code>output.scores</code> has
<code>2 * maximumIterations</code> float32 rows, one pair per executed search round. Optional
<code>output.evaluatedBandwidths</code> has the same shape and records the corresponding distances or neighbor
counts. Unused history slots are NaN, including on repeated execution of a compiled graph.
The search compares two interior points using delta 0.38197 and retains the winning interior.
Fixed-distance search stops on the absolute score difference between <strong>distinct</strong> finite candidates
or a collapsed float32 bracket. Adaptive search switches to an integer finishing pass when the
bracket spans at most five integers, a score tolerance is met, or rounding would repeat or collapse
an interior. It evaluates <strong>every integer in the remaining bracket, including its endpoints</strong> before
reporting convergence. Duplicate rounded points are never evidence of convergence. Finishing
passes use the same round budget; if it runs out first, the best evaluated candidate is published
with <code>iterationLimitReached = 1</code> and <code>converged = 0</code>.</p>
<p>Invalid scores lose to finite scores. Two invalid adaptive interiors start an integer pass over
the unchanged bracket. Two invalid fixed-distance interiors provide no score ordering, so the
bracket is preserved: the next round probes both endpoints, then remaining rounds probe the
breadth-first dyadic sequence <code>1/2, 1/4, 3/4, 1/8, ...</code> across that bracket. These feasibility probes
preserve the best finite score but do not claim convergence; budget exhaustion is reported even
when a finite candidate was found. A sufficiently narrow feasible region can remain unsampled
within the budget. This fallback avoids permanently discarding either side of an invalid pair.</p>
<p>Ties in a comparison choose the first candidate; the final result is the first best finite evaluation.
Both candidates are fitted each round (no score cache). No final decimal rounding is applied.
This is a bounded local search and does not guarantee the global optimum of a multimodal score.</p>
<p>The Poisson/Bernoulli criterion follows <a href="https://github.com/pysal/mgwr/blob/v2.2.1/mgwr/diagnostics.py">mgwr GLM diagnostics</a>:</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span>AICc = D + 2 K + 2 K (K + 1) / (n - K - 1)</span></span>
<span class="line"><span>K = sum of diagonal influence over eligible training focuses</span></span>
<span class="line"><span>D = sum of unit deviances at in-sample fitted means</span></span></code></pre>
<p>Every eligible training focus must converge with finite influence, and <code>n - K - 1</code> must be positive.
Otherwise the score is NaN. Masked/nonfinite/out-of-family rows do not count toward <code>n</code>, <code>D</code> or <code>K</code>.
Poisson offsets are log exposures. Logistic scores use the link scale to preserve confident
predictions; variance/influence use a stable logistic-tail formula even when the response-scale
mean rounds to one. AICc grid and golden searches are capped at <strong>4,096 total input slots</strong>
(<code>GENERALIZED_GWR_MAXIMUM_SEARCH_ROWS</code>, an internal reference-work limit). Candidate score
sums use bounded 256-row tiles followed by at most 16 tile totals in AICc mode; legacy LOO selection
can use the normal row limit (at most 4,096 tile totals).</p>
<p>Both methods accept optional <code>bandwidthSelection.status: {iterationCount, converged, iterationLimitReached}</code> scalar uint32 views, using the canonical iteration status port.
Grid selection executes one pass and reports convergence when a valid candidate exists.
Golden selection counts executed rounds, including integer finishing and feasibility probes. If its budget expires, it publishes the best
finite evaluated bandwidth and performs the final fit, with <code>converged = 0</code> and
<code>iterationLimitReached = 1</code>. <code>output.status</code> remains 0 for any selected finite candidate and 1
when none was valid; convergence and candidate validity are distinct. Later candidate fits and
score passes use GPU indirect gates once the search stops. All candidates and the final fit share
one training grid index, when requested. <code>estimatedCommandNodeCount</code> includes the search budget
and a conservative allowance of 1,024 nodes for published grid/sort primitives, validated at construction.
Row-proportional scratch is checked against device storage-binding limits before nodes are built.</p>
<p>The committed <code>test/oracles/generalized-gwr-aicc.py</code> uses NumPy working-response IRLS with
least-squares solves and pseudoinverse influence, independently of the GPU Newton/Cholesky code.
It follows mgwr GLM AICc. Adaptive oracle expectations enumerate every integer within the supplied
bounds, independently of the GPU search/stopping policy; fixed-distance fixtures also record the
ordinary golden-section history. The narrow integer-bracket regression compares against the
exhaustive table; wider non-unimodal intervals are checked against independently fitted evaluated
candidates, without requiring the global minimum. A separate two-site logistic regression has closed-form deviance, influence,
coefficients and fitted means, so feasible-region coverage does not rely on copied search logic.
Headless specs check fixed/adaptive, Gaussian/bisquare, both families, distance ties, masks,
Poisson offsets, empty grid slots, padded views, invalid bounds, search/local iteration budgets,
separation and score history. There is no generalized-GWR CPU operation twin in this package.</p>
<p>All final-fit, inference and selection output byte ranges must be pairwise disjoint, including
search status scalars. Construction rejects any overlap, even across different publishing stages.
Adjacent slices in one buffer may be used across stages when their ranges do not overlap.</p>`};export{e as default};