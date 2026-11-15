# Adaptive generalized geographically weighted regression

`GPUGeneralizedGeographicallyWeightedRegression` fits local Poisson log-link or Bernoulli
logit models on GPU. `bandwidthMode: 'adaptive'` interprets `parameters[0]` as an integer
neighbor count; the default `'fixed'` continues to interpret it as a distance. In both modes
`parameters[1]` is the positive relative coefficient tolerance. `maximumNeighborCount` is a
compile-time capacity from 1 to 128 (default 128); requested counts must fit this capacity.
The existing planar spatial-context units and Gaussian/bisquare kernels apply.

Adaptive selection counts complete-case rows, including the focus. Masked rows and rows with
nonfinite positions, predictors, response or offsets are excluded before selection, as are
responses outside the family domain. Each focus uses `1.00001 * kthDistance` as its distance
bandwidth, matching the float32-safe Gaussian GWR convention. All equal-distance boundary ties
participate; neighbor count controls the distance scale rather than imposing an exact support
cardinality. Gaussian support remains unbounded; bisquare support is strictly inside the inflated
radius. Selection is deterministic and independent of response values beyond domain validity.

Too few eligible rows, fewer positive-weight rows than coefficients, singular local designs and
zero kth distance produce status 2. A zero radius is not replaced by a coordinate-unit epsilon.
Invalid/noninteger counts and numerical overflow produce status 4. Excluded focuses have status 1.
All-zero Poisson neighborhoods and single-class logistic neighborhoods have status 5
(`GPU_ITERATIVE_REGRESSION_STATUS.DEGENERATE_RESPONSE`), because their intercept has no finite
maximum-likelihood solution. This diagnostic is also available in fixed mode. Complete or quasi separation witnessed by an exact column threshold or a decided Newton
direction receives status 6 (`GPU_ITERATIVE_REGRESSION_STATUS.SEPARATION`). A logistic threshold
certificate uses exact raw-value comparisons of the two response classes' predictor intervals.
A Poisson threshold requires all positive-count rows on one predictor value, with zero-count
rows on one side and at least one strictly beyond it. These certificates avoid rounded dot
products, including for simple quasi-separation.

A general direction must improve at least one positive-weight row without opposing any row:
signed margins for logistic; zero projection on positive-count rows and nonpositive projection
on zero-count rows for Poisson. A zero move is established only when every term is structurally
zero. Other moves must have the required strict sign beyond a conservative float32 rounding
bound `(designColumns + 8) * 2^-22 * sum(abs(terms))`; this bound rejects undecided signs and
never permits an opposing move. Offsets do not enter the direction test. This is a sufficient
numerical certificate, not an exhaustive linear-programming feasibility test. Uncertified separation can still produce
singularity, numerical failure or iteration-limit status 3. There is no penalty or separation
correction. Status 3 publishes the last finite coefficients and fitted values; other failed/excluded
outputs remain NaN. Status 0 reports convergence. Iteration
counts are zero for selection/preflight failures and count completed IRLS steps otherwise.

This is a reference-tier contributor: dense neighborhood traversal and its dense local solves have
explicit shape limits. Neighbor selection runs entirely on GPU before local IRLS; there is no control readback or CPU fallback. Optional bandwidth search and inference are described below. Dense neighbor selection costs `O(rows² * neighborCount)` and the existing
fit costs `O(rows² * iterations * predictorCount²)`. Bounds are 65,536 rows, 1–24 predictors
and 128 iterations (`GPU_LOCAL_REGRESSION_MAXIMUM_ROW_COUNT`,
`GPU_LOCAL_REGRESSION_MAXIMUM_PREDICTOR_COUNT`); they were 16,384 rows and 11 predictors. These are
shape limits, not performance promises. The per-focus fit state is tiled over focal rows when it
would exceed one storage binding.

Two opt-in options use the shared local weighted least squares engine that GWR also uses.
`indexGridSize: [columns, rows]` builds an internal grid index over the training rows, so bisquare
fits (fixed or adaptive bandwidth) visit only the cells covering the bandwidth and the row limit
rises to 1,048,576 (`GPU_LOCAL_REGRESSION_MAXIMUM_INDEXED_ROW_COUNT`). Adaptive k-nearest selection
walks grid rings around each focus and selects the same neighbors as the dense scan. The Gaussian
kernel uses the grid only with `gaussianCutoff` (in bandwidth multiples; 5 omits at most about
3.7e-6 of a unit weight); without it the Gaussian stays exact and scans every row. Sums run in cell
order, so results differ from the dense scan in float32 rounding only. A great-circle `spatialContext`
(`{coordinateSpace: 'longitude-latitude', metric: 'great-circle'}`) takes `(longitude, latitude)`
degrees and measures bandwidths in the context units (meters on a sphere of `sphereRadius`); it is
planar-grid exclusive. The same two options, limits and the engine apply to
`GPUMultiscaleGeographicallyWeightedRegression`; its full-inference mode keeps its 128-row budget.
Both default to the previous dense, planar behavior.

The committed `test/oracles/adaptive-generalized-gwr.py` generator composes NumPy order-statistic
distances with statsmodels frequency-weighted GLMs, separately from the GPU Newton solver.
The fixture records package versions and float32 input rounding. Tests compare both families,
both kernels, fixed and adaptive fits on tied grid distances, masks and Poisson offsets, with
absolute tolerance 0.003. Additional cases cover invalid counts, insufficient neighborhoods,
zero-distance ties, nonfinite rows, boundary responses and iteration limits. This is independent
composition evidence, not direct mgwr generalized-regression parity.

## Training-only selection and model-specific inference

`bandwidthSelection` accepts 2–32 fixed distances or adaptive neighbor counts, and caller-owned
`scores`, `bandwidth` and `status` outputs. Each candidate is scored by exact leave-one-out mean
Poisson or Bernoulli deviance. Every eligible training focus must converge; failed candidates have
NaN scores. Status is 0 when a finite candidate is selected, or 1 if none succeeds. Equal scores
choose the first supplied candidate. This is the default `method: 'grid'`, `criterion: 'loo-deviance'` behavior. For grid AICc,
set `criterion: 'aicc'`; candidates then include their own focus and use the criterion below. The ordinary `parameters[1]` coefficient tolerance applies to every inner and final fit.

For an outer spatial fold, pass its training mask. Selection never uses masked-out responses.
`predictExcluded: true` permits finite excluded predictor/position queries to receive the final
selected model's predictions, without requiring their response values. Adaptive selection counts
only eligible training neighbors; leave-one-out removes the focus from that count. A held-out query's
finite offset remains a caller-supplied prediction input. Rebuild this graph with each outer fold's
training mask and compute held-out metrics afterward.

Optional `inference` contains `standardErrors` (focus-major, intercept first) and `influence` (one
per focus). The scale-one Poisson/logistic covariance uses `A^-1 X' V W² X A^-1`, with
`A = X' V W X`, and influence is the diagonal of the local GLM smoother. These are model-specific
local-fit errors, not multiscale backfitting uncertainty. Failed fits retain NaN inference. The direct
mgwr 2.2.1 fixture verifies both families, and a separate exact-LOO fixture verifies candidate scores.

## AICc golden-section bandwidth selection

Set `bandwidthSelection.method: 'golden-section'`. Its `parameters` view contains
`[lowerBandwidth, upperBandwidth, absoluteScoreTolerance]`. Bounds must be positive finite values
with lower strictly below upper; tolerance must be positive finite. In adaptive mode both bounds
must be integers within `maximumNeighborCount`; interior points round to nearest, ties to even.
Invalid runtime search parameters produce selection status 1, NaN bandwidth/scores, zero search
iterations and neither stopping flag. The ordinary `parameters[1]` remains the IRLS tolerance;
`parameters[0]` is replaced internally by each candidate and the final selected bandwidth.

`maximumIterations` inside `bandwidthSelection` is a compile-time search budget, 1–64, default 24.
It is independent of the contributor's local IRLS `maximumIterations`. `output.scores` has
`2 * maximumIterations` float32 rows, one pair per executed search round. Optional
`output.evaluatedBandwidths` has the same shape and records the corresponding distances or neighbor
counts. Unused history slots are NaN, including on repeated execution of a compiled graph.
The search compares two interior points using delta 0.38197 and retains the winning interior.
Fixed-distance search stops on the absolute score difference between **distinct** finite candidates
or a collapsed float32 bracket. Adaptive search switches to an integer finishing pass when the
bracket spans at most five integers, a score tolerance is met, or rounding would repeat or collapse
an interior. It evaluates **every integer in the remaining bracket, including its endpoints** before
reporting convergence. Duplicate rounded points are never evidence of convergence. Finishing
passes use the same round budget; if it runs out first, the best evaluated candidate is published
with `iterationLimitReached = 1` and `converged = 0`.

Invalid scores lose to finite scores. Two invalid adaptive interiors start an integer pass over
the unchanged bracket. Two invalid fixed-distance interiors provide no score ordering, so the
bracket is preserved: the next round probes both endpoints, then remaining rounds probe the
breadth-first dyadic sequence `1/2, 1/4, 3/4, 1/8, ...` across that bracket. These feasibility probes
preserve the best finite score but do not claim convergence; budget exhaustion is reported even
when a finite candidate was found. A sufficiently narrow feasible region can remain unsampled
within the budget. This fallback avoids permanently discarding either side of an invalid pair.

Ties in a comparison choose the first candidate; the final result is the first best finite evaluation.
Both candidates are fitted each round (no score cache). No final decimal rounding is applied.
This is a bounded local search and does not guarantee the global optimum of a multimodal score.

The Poisson/Bernoulli criterion follows [mgwr GLM diagnostics](https://github.com/pysal/mgwr/blob/v2.2.1/mgwr/diagnostics.py):

```text
AICc = D + 2 K + 2 K (K + 1) / (n - K - 1)
K = sum of diagonal influence over eligible training focuses
D = sum of unit deviances at in-sample fitted means
```

Every eligible training focus must converge with finite influence, and `n - K - 1` must be positive.
Otherwise the score is NaN. Masked/nonfinite/out-of-family rows do not count toward `n`, `D` or `K`.
Poisson offsets are log exposures. Logistic scores use the link scale to preserve confident
predictions; variance/influence use a stable logistic-tail formula even when the response-scale
mean rounds to one. AICc grid and golden searches are capped at **4,096 total input slots**
(`GENERALIZED_GWR_MAXIMUM_SEARCH_ROWS`, an internal reference-work limit). Candidate score
sums use bounded 256-row tiles followed by at most 16 tile totals in AICc mode; legacy LOO selection
can use the normal row limit (at most 4,096 tile totals).

Both methods accept optional `bandwidthSelection.status: {iterationCount, converged,
iterationLimitReached}` scalar uint32 views, using the canonical iteration status port.
Grid selection executes one pass and reports convergence when a valid candidate exists.
Golden selection counts executed rounds, including integer finishing and feasibility probes. If its budget expires, it publishes the best
finite evaluated bandwidth and performs the final fit, with `converged = 0` and
`iterationLimitReached = 1`. `output.status` remains 0 for any selected finite candidate and 1
when none was valid; convergence and candidate validity are distinct. Later candidate fits and
score passes use GPU indirect gates once the search stops. All candidates and the final fit share
one training grid index, when requested. `estimatedCommandNodeCount` includes the search budget
and a conservative allowance of 1,024 nodes for published grid/sort primitives, validated at construction.
Row-proportional scratch is checked against device storage-binding limits before nodes are built.

The committed `test/oracles/generalized-gwr-aicc.py` uses NumPy working-response IRLS with
least-squares solves and pseudoinverse influence, independently of the GPU Newton/Cholesky code.
It follows mgwr GLM AICc. Adaptive oracle expectations enumerate every integer within the supplied
bounds, independently of the GPU search/stopping policy; fixed-distance fixtures also record the
ordinary golden-section history. The narrow integer-bracket regression compares against the
exhaustive table; wider non-unimodal intervals are checked against independently fitted evaluated
candidates, without requiring the global minimum. A separate two-site logistic regression has closed-form deviance, influence,
coefficients and fitted means, so feasible-region coverage does not rely on copied search logic.
Headless specs check fixed/adaptive, Gaussian/bisquare, both families, distance ties, masks,
Poisson offsets, empty grid slots, padded views, invalid bounds, search/local iteration budgets,
separation and score history. There is no generalized-GWR CPU operation twin in this package.

All final-fit, inference and selection output byte ranges must be pairwise disjoint, including
search status scalars. Construction rejects any overlap, even across different publishing stages.
Adjacent slices in one buffer may be used across stages when their ranges do not overlap.
