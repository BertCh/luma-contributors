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
maximum-likelihood solution. This diagnostic is also available in fixed mode. Other complete or
quasi separation is not exhaustively classified: it may produce singularity, numerical failure or
iteration-limit status 3. There is no penalty or separation correction. Only converged status 0
publishes finite coefficients and fitted means; failed/excluded outputs remain NaN. Iteration
counts are zero for selection/preflight failures and count completed IRLS steps otherwise.

Neighbor selection runs entirely on GPU before local IRLS; there is no control readback or CPU fallback. Optional bandwidth search and inference are described below. Dense neighbor selection costs `O(rows² * neighborCount)` and the existing
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
choose the first supplied candidate. This criterion is explicit and differs from AICc or approximate
CV selection. The ordinary `parameters[1]` coefficient tolerance applies to every inner and final fit.

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
