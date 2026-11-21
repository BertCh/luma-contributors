# Point process contributors

These contributors add bounded, GPU-resident point-process stages to `@luma-contributors/analysis/gpu-spatial-analysis`; `GPUNetworkKernelDensity` and `GPUNetworkInhomogeneousK` belong to `@luma-contributors/analysis/gpu-network`. They implement `GPUCommandNodeProducer`: add them to a `GPUCommandGraph`, compile and encode the graph, then explicitly read back only the results the application needs. Input and output buffers remain caller-owned. Inputs cannot overlap outputs. Every storage stage fits the eight-binding WebGPU minimum.

`GPUOPTICS` accepts a `times` column for ST-OPTICS; see [Space-time clustering](./space-time-clustering.md).

Continuous and balanced sampling operation forms require resolved views to match their declared
`frameRowCount`, `legacySiteCount` and `rowCount`, respectively. Mismatches are refused before
building GPU contributors, matching CPU execution. Both sampling backends capture declared counts,
input/output names and allocations at construction, including inherited getters; later caller
changes cannot change the population or remove a declared legacy-site check. Envelope contributors and both operation forms
capture their settings and output maps at construction; reusing or editing the caller's props
does not change an existing instance.

| Contributor | New behavior | Bound and status |
| --- | --- | --- |
| `GPUInhomogeneousPointPattern` | Supplied/estimated intensity; polygon border or translation K; annular or Epanechnikov pair correlation | 4096 events, 256 radii; status: 0 valid, 1 empty erosion, 2 invalid input, 3 overlap work exhausted, 4 unresolved overlap |
| `GPUPointPatternEnvelope` | Generic resident simulation-curve pointwise envelopes plus maximum-deviation, DCLF or global extreme-rank tests | 4096 simulations and 4096 ordinates; status 1 rejects invalid curves or inputs used by the selected statistic |
| `GPUNetworkKernelDensity` | Branch-length quadrature-normalized shortest-path intensity | At most 65,536 queries, 65,536 events, and 16,777,216 distances. `output.validity` is 1 for a usable query and 0 for an invalid or suppressed query. An event with no positive quadrature support invalidates every query. |
| `GPUNetworkInhomogeneousK` | Supplied-intensity network K with optional Ang geometry correction and three normalization conventions | 1024 events, 256 radii, explicit pair budget; per-radius invalidity bitmask |
| `GPUOPTICS` | Exact reachability ordering, core distances, predecessors and DBSCAN/xi extraction | 4096 events with the exhaustive reachability step, 1,048,576 with `neighborIndex: 'grid'` (the default above 4096), where xi extraction is limited to 65,536; minimumSamples up to 256, explicit maximumIterations; visited count, completion flag and excluded-invalid-row count |
| `GPUSpatialBalancedSample` | Finite-point stratified equal/unequal probability selection and design weights | 4096 frame rows, 256 strata; per-stratum selected count and status 0 success, 1 invalid design/frame, 2 random-draw exhaustion |

## Intensity and polygon corrections

When a failed preparation is retried on a fresh graph that already has a producer, call
`window.setRetryViews(freshGraph, {positions, featureOffsets, polygonOffsets, ringOffsets})`
before `freshGraph.add(window)`, using that producer's actual fresh views. If the original
preparation supplied an output summary, also provide `output: {summary}` with its equivalent
fresh view. Each range must match the original persistent buffer, format, row count, byte
offset and stride. This matters when a DynamicBuffer and its backing Buffer, or different
usage descriptors, create distinct logical handles for the same allocation. The mapping is
captured. Transient producers and their preparation handles must be recreated instead.

`GPUInhomogeneousPointPattern` consumes a `GPUStudyWindowPreparation` added to the same graph first. The window supports holes and disconnected components. Event positions use that window's planar coordinate frame. `GPUInhomogeneousPointPattern`, `GPUSpatialBalancedSample` and `GPUContinuousSpatialSample` also take an optional `spatialContext`; only the default planar context is supported and longitude-latitude is rejected with the standard message. Border K evaluates the neighbor sum of each (radius, event) pair in its own invocation, then adds the terms per radius in fixed event order, so results do not depend on scheduling. Finite events outside the polygon are excluded. Invalid or zero intensity for an included event invalidates inference; events are never removed merely because their estimated intensity is small.

Supply `suppliedIntensity` to use externally specified positive intensities. Otherwise set `bandwidth` to estimate intensity at event locations with the normalized two-dimensional Epanechnikov kernel:

```
K_h(d) = 2 / (pi * h²) * max(0, 1 - d² / h²)
lambda(x) = sum_j K_h(|x - x_j|) / integral_window K_h(|x - u|) du
```

The denominator reuses the polygon-window kernel-mass integration. `selfContribution: 'exclude'` is the default; `'include'` explicitly restores the event's own contribution. A sparse leave-one-out neighborhood can have zero intensity, which produces invalid inference rather than an arbitrary intensity floor. Estimated intensity is a plug-in procedure; it is not an unbiased estimator with known intensity.

For radius `r`, let `E(r)` be included events strictly farther than `r` from any polygon boundary (`b_i > r`, matching spatstat border erosion). The reduced-sample estimator is:

```
K(r) = sum_{i in E(r)} (1/lambda_i) * sum_{j != i, dij <= r} (1/lambda_j)
       / sum_{i in E(r)} (1/lambda_i)
```

No eligible focus gives NaN and status 1. Radii must be positive and increasing. The default `edgeCorrection: 'border'` retains this reduced-sample estimator. When the top-level `pairCorrelation` option is omitted, the output of that name remains the difference of consecutive K values divided by annulus area; erosion can make this difference negative.

Set `edgeCorrection: 'translation'` to use the overlap area `A(h) = area(W intersect (W + h))` of the prepared polygon with its translated copy:

```
K_translation(r) = sum_{i != j, dij <= r} 1 / (lambda_i * lambda_j * A(xj - xi))
```

There is no fitted-intensity renormalization, matching `Kinhom(..., renormalise=FALSE)`. Distinct coincident events contribute with `A(0)=area(W)`; only self rows are excluded. Outside and nonfinite positions remain excluded, while invalid intensity for any included event invalidates inference. Boundary membership follows the prepared window's deterministic point-in-polygon convention. Inclusive distance thresholds are tested separately from the upstream histogram's boundary-bin convention.

Select a smoothed estimator explicitly:

```typescript
import {
  GPUInhomogeneousPointPattern,
  type GPUInhomogeneousPointPatternProps
} from '@luma-contributors/analysis/gpu-spatial-analysis';

function createTranslationPattern(inputs: Pick<GPUInhomogeneousPointPatternProps,
  'positions' | 'studyWindow' | 'radii' | 'output'> &
  Required<Pick<GPUInhomogeneousPointPatternProps, 'suppliedIntensity'>>) {
  return new GPUInhomogeneousPointPattern({
    ...inputs,
    edgeCorrection: 'translation',
    pairCorrelation: {
      kernel: 'epanechnikov', bandwidth: 0.7,
      divisor: 'radius', zeroCorrection: 'none'
    },
    maximumPairCount: 100000,
    maximumOverlapEdgePairs: 65536
  });
}
```

The pair kernel is `k_h(t)=0.75/h * max(0,1-(t/h)^2)`. Its `bandwidth` is the compact support half-width `h`, corresponding to `pcfinhom(h=h)` or standard deviation `bw=h/sqrt(5)`. This bandwidth is separate from the two-dimensional intensity bandwidth. The estimate sums `k_h(r-dij)/(2*pi*r*lambda_i*lambda_j*A(xj-xi))` over ordered distinct rows. `zeroCorrection: 'reflection'` adds `k_h(r+dij)`; `'none'` applies no origin correction. `divisor: 'distance'` replaces `r` in the denominator with `dij` and rejects coincident distinct events with status 2. Upstream's distance-divisor path instead warns and omits their nonfinite terms; this contributor deliberately reports the unsupported inference. Both options require translation correction. Supplied, self-included estimated and leave-one-out estimated intensities use exactly the same downstream estimator.

Overlap is computed entirely on the GPU by signed fan composition: intersect each oriented edge triangle with each translated edge triangle, then sum signed areas. This supports concavity, holes and disconnected components without CPU triangulation or an intermediate readback. Convex clipping uses eight local slots (a triangle intersection has at most six vertices), exact integer orientation and exact determinant-magnitude fallback when floating products cancel. Constructed intersection coordinates and areas remain f32. A streaming binary reduction keeps area summation depth logarithmic. Overlaps at or below `64*f32_epsilon*max(windowArea,sumAbsoluteTriangleAreas)`, uncertain predicates, unrepresentable metrics or a clipping-capacity failure return status 4 instead of a potentially enormous reciprocal weight.

`maximumOverlapEdgePairs` defaults to 65536 and can be set from 1 through 1048576. It bounds nondegenerate triangle intersections per relevant pair; exhausted work returns status 3. Pairs beyond the largest radius plus the pcf half-width need no overlap calculation. Per-radius failures produce NaN for both K and pair correlation, so incomplete geometric work is never presented as an estimate. `maximumPairCount` optionally bounds the resident unordered-pair allocation; insufficient capacity throws during graph construction. Invalid translation radius sequences invalidate every ordinate.

With `n` events, `R` radii and `E` polygon edges, worst-case work is `O(n²E²+n²R)`. Pair geometry is evaluated once and reused by parallel event/radius reductions. Resident scratch is approximately `8*n*(n-1)/2 + 12*n*R + 4*n` bytes, plus the prepared window. The largest pair buffer at 4096 events is 67,092,480 bytes; no out-of-core or subquadratic scaling is claimed. Re-encoding recomputes pairs from current GPU inputs. Native WebGPU verification includes 4096 events on a 64-by-64 lattice, four local radii and a 0.2-unit pcf half-width: 36.4 ms first execution and 7.6/9.6 ms warm encode/submit/queue-completion measurements on the verification host, with a 67,092,480-byte pair table. These are host wall-clock smoke measurements, not isolated GPU timestamps or worst-case dense-radius throughput. Updating every resident intensity from 1 to 2 recomputes K and pcf to one quarter of their previous values. A separate 104-edge, 25-hole fixture agrees with pinned Shapely 2.1.2 polygon intersections. Existing `GPURipley` continues to supply homogeneous polygon border/isotropic corrections.

## Simulation curves and global tests

The graph-value forms are `GPUPointPatternEnvelopeOperation` and
`CPUPointPatternEnvelopeOperation`; `computePointPatternEnvelopeOnCPU` is the pure CPU twin.
All use the same statistic and exchangeability options. Direct GPU limits are 4096 ordinates
(`MAXIMUM_CURVE_ROW_COUNT`), 4096 simulations (`MAXIMUM_SIMULATION_COUNT`), and, for extreme rank,
16,777,216 ordinate/curve comparisons (`MAXIMUM_RANK_COMPARISON_COUNT`). CPU forms retain the
uint32 packed-vector limit instead of these GPU work caps. For `maximum-deviation` and
`integrated-squared-deviation`, the CPU uses exact dyadic arithmetic on the stored float32 inputs
to decide exceedances and reports the statistic as float32.
Observed, reference, bounds and simulations use the curve's units; weights supply the quadrature
measure, so the integrated statistic has squared curve units times weight units. Maximum
deviation retains curve units; ranks, p-values and all counts are dimensionless.


`GPUPointPatternEnvelope` consumes an observed curve and a replicate-major simulation table. Maximum deviation (the default) additionally requires a fixed `reference`; integrated squared deviation additionally requires positive `integrationWeights`. Extreme rank needs neither optional vector; complete legacy input bags remain accepted. Weights are ignored in maximum-deviation mode, including zero weights. `simulationCount` may be omitted: the simulation length must then be an exact multiple of the nonempty observed length, with at least two simulations. An explicit count must match that quotient. Simulation generation is an upstream graph stage. Existing `GPURipleyEnvelope` already generates conditional CSR patterns and provides pointwise/global maximum-deviation inference; this contributor generalizes inference to other resident curves and adds integrated squared deviation.

The required `nullModel` and `fittedParameters` properties preserve the simulation contract. Parameter treatment is `'none'`, `'fixed-external'`, or `'refit-each-replicate'`. For the last case, the upstream simulation graph must actually repeat its fitting procedure. Merely setting metadata cannot establish exchangeability or calibrate a fitted model. The reference curve must be fixed independently of the tested observed realization.

`lower` and `upper` are the requested pointwise order statistics. The separate global statistic is either the maximum absolute deviation or `sum(weight * (curve-reference)²)`, the DCLF form. Its conservative p-value is `(1 + number of simulated statistics >= observed)/(simulationCount + 1)`, including ties. The reported resolution is `1/(simulationCount + 1)`. Both CPU and GPU integrated comparisons use exact dyadic integer products and sums of the supplied float32 inputs: observed `[1, 2^-13]` or `[1, 2^-27]` versus nineteen `[1, 0]` simulations, zero reference and unit weights has zero exceedances and `p=1/20`, even though the displayed observed statistic rounds to `1`. The latter exact statistic is `1 + 2^-54`, whose residual would also be lost in a single binary64 total. The same comparison retains squared-product residuals (observed `[1+2^-13,1-2^-13]` versus `[1,1]`) and normal addends below `2^-103` (observed `[2^-40,2^-53]` versus `[2^-40,0]`); both have zero exceedances with nineteen identical simulations. Displayed statistics remain float32. The GPU exact comparator repeats the observed statistic per curve and uses 28 integer words per invocation; it preserves the stated 4096-ordinate/4096-simulation GPU bounds, without claiming a performance improvement. Pointwise envelopes are not simultaneous bands. Any nonfinite curve, a nonfinite reference in a deviation test, or nonpositive/nonfinite weights in the integrated test invalidates the complete test; failed simulations are not selectively dropped.

With `statistic: 'extreme-rank'`, ranks include the observed curve and all simulations. Each ordinate uses two-sided midranks, and each curve's statistic is its smallest ordinate rank. Smaller ranks are more extreme. `statistics` reports `[observedExtremeRank, conservativePValue, resolution, simulationCount]`; optional `pValueInterval` reports `[fraction strictly more extreme, fraction at least as extreme]`, retaining tie uncertainty. `significanceLevel` (default 0.05) chooses the largest attainable midrank whose strictly rejected fraction is at most the supplied level, including half-integer cutoffs and equality at an attainable fraction. For observed `0` and simulations `1…47` plus another `47`, level `1/49` gives bounds `[1,47]`; the immediately smaller level retains `0`. `lower`/`upper` span only curves whose extreme midrank meets that critical rank: entire rejected tie groups, including a rejected observation, are excluded from the bounds. This mode does not read the reference or quadrature weights; they may be omitted. Its explicit work bound is `(simulationCount+1)^2 * ordinateCount <= 16777216`; no curve or ordinate is truncated. Nonfinite curves invalidate inference. SciPy midrank fixtures include ties, half-integer cutoffs and identical curves; the fixture records separate rank-regeneration versions.

The CPU integrated statistic is rounded directly from its exact dyadic sum to float32,
with ties to even. It does not first round through binary64: for example,
`1 + 2^-24 + 2^-62` rounds above `1`, while the exact midpoint `1 + 2^-24` rounds to `1`.
The pure CPU entry also captures inherited and non-enumerable inference options before
resolving defaults, matching the named operation forms.

Use `decodeGPUPointPatternEnvelopeStatistics(readback)` and
`decodeGPUPointPatternEnvelopeStatus(readback)` for named fields from typed arrays or buffer
readback bytes. The status decoder reports `complete` and `invalidInput`.

Extreme-rank inference assigns one invocation per curve to compare its rank with the others.
Published `GPUReduction` stages sum the tail/invalid flags and select the largest admissible
midrank; the final statistics/status publication does constant work. The existing quadratic
comparison limit still applies. This path adds reduction nodes and scratch to the command graph.

## Network intensity

`GPUNetworkKernelDensity` consumes a query-major table of shortest-path distances from each query to every event. Use resident network cost-matrix products with **arc-length** costs. Time, toll and resistance costs are not length. `+Infinity` is disconnected support; NaN, negative infinity and negative distances are invalid.

Each query has a nonnegative `quadratureLengths` value representing the arc length of that part of the network. Include all branches in this quadrature. Zero-weight queries can evaluate intensity at additional event locations. For each event, the contributor integrates its triangular kernel over these lengths and divides that event's kernel contribution by the integral. Thus the length-weighted density integral equals event count to f32 error when no self exclusions are requested. Terminals, junctions and disconnected components are accounted for through the actual represented edge length. This is a normalized radial shortest-path kernel, not an equal-split diffusion/heat kernel. Accuracy depends on supplied quadrature resolution; no analytic continuous-network integration is claimed.

`excludedEventIds` enables leave-one-out evaluation: each query carries the event ID to exclude, or `0xffffffff` for none. Such exclusions change the integral and remove the event-count conservation guarantee. A source event with no positive quadrature support invalidates results explicitly. Use `output.validity` to select usable intensity rows and optional `output.status.invalidCount` to count rows whose validity is zero. Request `output.status.complete` to retain incumbent estimates with a routing-completeness certificate. Omit `output.status` when neither diagnostic is needed. Use `GPUNetworkInhomogeneousK` for the separate intensity-reweighted second-order statistic described below; density itself is not a K statistic.


## Inhomogeneous network K

`GPUNetworkInhomogeneousK` estimates `sum_{i != j, dij <= r} e(i,j)/(lambda_i*lambda_j)/D`. Its default `correction: 'ang'` uses reciprocal metric-sphere cardinality: count distinct network locations at distance `dij` from the source event. Locations within `1e-5 * dij` of each other along the network count as one, so a crossing that close to an edge end is that node; the distance table holds rounded float32 sums, and an exact comparison would count the edge into the destination a second time. `'none'` assigns unit pair correction. Default `D` is total undirected network length. `'inverse-intensity'` uses `sum(1/lambda_i)`, and `'inverse-intensity-squared'` uses `sum(1/lambda_i)^2/networkLength`, matching the two normalization powers in [spatstat's linearKinhom contract](https://spatstat.r-universe.dev/spatstat.linnet/doc/manual.html#linearKinhom). These are distinct from the existing homogeneous spaghetti `GlobalAutoK` denominator convention.

The supplied `nodeDistances` table contains complete event-to-node shortest-path distances in **arc-length units**; resident `GPUNetworkCostMatrix` results can supply this table. `+Infinity` denotes disconnected nodes. A maximum-radius-truncated routing table is not a complete distance table and must not be used. Routing must be completed before this consumer runs. `intensity` is a positive finite value per event, supplied externally or produced by an upstream intensity stage with an explicit self-contribution policy. No intensity fitting occurs inside the K contributor.

Events are represented by `eventNodeIds`. Interior events must first split their containing edge and become graph nodes; the graph must be noded at intersections, with one positive-length undirected edge per geometric segment. Contract zero-length edges upstream. The implementation sums the provided lengths exactly once for network-length normalization. It counts each vertex at the requested distance once, then finds interior metric-sphere crossings on each edge. This treats branches, cycles and disconnected components directly rather than substituting Euclidean distance or a smoothed shell derivative. Topology/metric tables must refer to the same graph; graph consistency is caller-owned. Values use f32 arithmetic, so poorly conditioned distance ties near wavefront intersections require rescaling or a higher-precision preparation.

Distinct coincident events contribute at radius zero; a row never pairs with itself. Disconnected pairs contribute zero. Radii use inclusive `<=` thresholds. `numerator` preserves the weighted ordered-pair sum, and `normalization` returns `[networkLength, inverseIntensitySum, effectiveDenominator]`. Per-radius status bits are: 1 invalid event ID/intensity, 2 edge topology/length, 4 distance table, 8 undefined geometric shell, 16 invalid radius, and 32 nonfinite arithmetic. Flagged K and numerator outputs are NaN; normalization fields remain diagnostic values. A resident intensity update recomputes all results on the next encoding.

An explicit `maximumPairCount` budget bounds the quadratic scratch allocation; insufficient budget throws instead of dropping pairs. The initial path allows at most 1024 events and 256 radii. Ang preparation costs O(events² * (nodes + edges)); the radius reduction costs O(events² * radii). Independent fixtures use NetworkX 3.5 shortest paths and enumerate quarter-length network locations whose metric spheres are exact for the selected integral graphs. They cover a three-way junction plus disconnected component, a cycle with a branch, coincident events, both correction modes and all three normalization conventions.

## OPTICS ordering

The graph-value forms are `GPUOPTICSOperation` and `CPUOPTICSOperation`;
`computeOPTICSOnCPU` is the pure CPU twin. The operation forms use the same named metric,
ordering and extraction contracts below.

For planar searches with `maximumDistance` above 2^63 in position units, the core-neighbor and
grid frontier searches scale coordinates and the radius by a power of two so the radius can be
squared in float32. Core and reachability distances are reported in the input position units.

`GPUOPTICS` defaults to planar Euclidean distances in the input coordinate units. Set
`spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle', units: 'meters'}`
for longitude/latitude degrees and spherical distances in meters; declared `kilometers` and an
explicit `sphereRadius` are also supported. The default sphere is the IUGG mean Earth radius.
Both core-distance selection and frontier updates use this same metric. The all-pairs search
handles neighbors across the antimeridian without an index seam. Geographic coordinates outside
`[-180, 180] × [-90, 90]` and nonfinite coordinates are excluded. Absolute Float32 longitude/latitude
retains roughly meter-scale quantization at Earth scale; this is spherical, not ellipsoidal,
distance. Existing planar inputs and defaults are unchanged.

`GPUOPTICS` uses self-inclusive `minimumSamples` as in scikit-learn. It computes exact core distances inside `maximumDistance`, then performs bounded resident frontier expansion. The minimum reachability wins, with input row as deterministic tie-breaker. Distances, reachability and predecessors are indexed by input row; `ordering` is visit-rank to input row. Roots/unreachable rows have infinite reachability and predecessor `0xffffffff`. Invalid coordinates are excluded and counted.

`extractionDistance` chooses DBSCAN extraction from the completed order and must not exceed `maximumDistance`. Noise is `0xffffffff`. If `maximumIterations` is exhausted, unvisited ordering slots remain sentinel and **all extracted labels remain noise**; the completion flag distinguishes this from a completed all-noise result. Set `extractionMethod: 'xi'` for steep-region hierarchy extraction, with `xi` (default 0.05), `minimumClusterSize` (default minimumSamples), and `predecessorCorrection` (default true). `extractionDistance` is then unused. Optional `clusterIntervals` has exactly `n` uint32-pair rows containing inclusive ordering-rank intervals; `clusterSummary` contains `[intervalCount, overflow]`. Nested intervals appear after contained intervals, and labels select disjoint innermost clusters. Unused intervals are sentinel pairs. Overflow invalidates labels instead of returning a truncated hierarchy. This WGSL extraction adapts scikit-learn 1.7.2 under its retained BSD 3-Clause notice; direct pinned fixtures validate the hierarchy with both predecessor settings and two xi values.

With `orderingMethod: 'sequential'` (default), frontier visits use a 256-lane cooperative minimum reduction and distributed neighbor update. `frontierBatchSize` (default 256, maximum 256) bounds work per dispatch; the graph declares every batch needed by `maximumIterations`. Thus the default completes all 4096 supported events without readback or a CPU numerical fallback. Core distances come from a `GPUNeighborSearch` k-nearest self join capped at `maximumDistance` (and the time window, with `times`), which lists the `minimumSamples - 1` nearest other rows; the row itself is the first sample. Frontier work remains O(n²), so the index does not change the total cost at this size: 4096 clustered rows took 27.4 ms against 27.9 ms before. Initialization uses a cooperative reduction, DBSCAN extraction uses a parallel prefix scan, and xi predecessor correction uses an inverse-order map for constant-time interval membership. Graph scratch is O(n · minimumSamples); no dense pair matrix is constructed. It is a small-frame correctness path, with no unmeasured performance claim over indexed CPU OPTICS.

`neighborIndex` (compile-time) chooses the reachability step. `'brute-force'` is the frontier above, at most `GPU_OPTICS_BRUTE_FORCE_MAXIMUM_ROWS` (4096) rows. `'grid'` runs a `GPUNeighborSearch` radius self join once (radius and time window widened by 1e-4, so the candidate list is a superset), relaxes only those candidates with the same distance and time tests, and selects each next row from a seed list of the unvisited rows with finite reachability, or as the lowest unvisited row when that list is empty, so the ordering, reachability, predecessors and labels equal the exhaustive step's (checked on planar, space-time, great-circle, partial and xi scenes). On the default sequential path, work per visit is the seed-list length plus the visited row's neighbor count, not the row count; visits stay sequential in one workgroup, so it remains a reference-tier operator, at most `GPU_OPTICS_GRID_MAXIMUM_ROWS` (1,048,576) rows. `'auto'` (default) uses the exhaustive step up to 4096 rows and the grid above it. `neighborCapacity` (default `rows * min(rows - 1, 64)`) holds the radius list; when it is too small, the optional `output.ports.status.candidateOverflow` is 1, the ordering is not computed, and the optional `iterationCount` and `converged` ports both report 0. Xi extraction keeps its own reference-tier limit of 65,536 rows (`GPU_OPTICS_XI_MAXIMUM_ROWS`). Initialization and overflow cleanup run per row; a sequential extraction state persists between dispatches of at most 64 ordering ranks. Each gated extraction dispatch is followed by a separate scalar gate update, keeping writable storage separate from its indirect argument usage. Steep-area extension, predecessor correction and interval labeling still perform bounded serial scans, so xi remains reference-tier and pathological hierarchies can be expensive. Independent fixtures test ordering, distances, predecessors, labels and exhaustion.

`orderingMethod: 'parallel'` requires the grid path (select `neighborIndex: 'grid'` explicitly below 4097 rows). Each visit reduces the nonnegative float32 reachability bit patterns with published `GPUReduction`, marks matching rows, reduces their input indices, commits that row, and relaxes its radius candidates across workgroups. The unsigned key is lossless, with infinity below the invalid/visited sentinel; exact ties, including disconnected roots, choose the lowest input index. Only strict reachability improvement replaces a predecessor. The two reductions preserve the sequential frontier's exact float32 ordering and predecessors. Visits still depend on previous visits and use dispatch boundaries, with GPU gates disabling finished frontier work. The scalar commit dispatch owns and updates the gate, so it runs without an indirect condition and returns immediately when the gate is disabled. This distributes frontier work; it does not make independent visits possible.

The parallel path is opt-in because it declares up to nine command nodes per visit (up to 592,128 for a full 65,536-row xi graph, including the fixed allowance). Its named command budget covers that bounded graph; construction validates it and specs check actual counts at the reduction boundaries. Reduction scratch is reused across visits. `frontierBatchSize` affects only the sequential path. No performance advantage is claimed without adapter measurements; graph compilation and encoding may dominate. For small CPU-resident collections prefer `CPUOPTICSOperation` / `computeOPTICSOnCPU`, which compute in float64. The CPU route ignores GPU dispatch and neighbor-storage capacities but honors an explicit `maximumIterations` visit budget, defaulting to the input row count.

Optional one-row uint32 ports under `output.ports.status` expose `iterationCount` (visited rows), `converged` (1 when all valid rows were visited), `invalidCount` (excluded rows with invalid positions or times), and `iterationLimitReached` (1 when the visit budget ran out). Candidate overflow leaves the ordering uncomputed, sets `candidateOverflow`, and leaves `iterationLimitReached` zero. An exhausted visit budget preserves the partial ordering and reachability. Extraction labels stay noise and xi intervals stay empty. The CPU twin returns the same named status arrays directly and suppresses both DBSCAN labels and xi extraction while incomplete. CPU candidate overflow is zero because its neighbor search is exhaustive. CPU and GPU operations retain their construction settings through deferred execution and lowering.

Lens receipts mark branch-inapplicable settings as disabled and non-sweepable: DBSCAN does not use `xi` or `minimumClusterSize`, Xi does not use `extractionDistance`, and a supplied-intensity example does not use bandwidth estimation. Deviation envelopes do not use `significanceLevel`; extreme-rank envelopes use it instead of the pointwise `rank` setting.

The independent 4,104-row fixture records its NumPy/scikit-learn versions and is reproducible with `modules/analysis/test/gpu-spatial-analysis/spatial-clustering/optics-parallel-fixtures.py`. Integer coordinates make ties exact in float32 and float64: scikit-learn chooses the lowest unprocessed input index at equal reachability and keeps the predecessor on equal candidate distances. Both predecessor-correction modes, duplicate points, disconnected components, padded workgroups and xi batch boundaries are covered. Non-integer distances near rounding ties can differ from scikit-learn's float64 values; equality to the sequential GPU path is the exact ordering contract.

## Spatial probability sampling

The graph-value forms are `GPUSpatialBalancedSampleOperation` and
`CPUSpatialBalancedSampleOperation`; `computeSpatialBalancedSampleOnCPU` is the pure CPU twin.
They share allocation, bounds and random-stream options with the direct contributor.

`GPUSpatialBalancedSample` accepts a complete finite frame with unique `frameIds`, positions, explicit enclosing bounds and optional strata. Every stratum has a positive requested sample size. Invalid/out-of-bounds positions and unknown strata invalidate results; the implementation never repairs the frame by silently omitting rows.

Sixteen quadtree levels receive independently randomized child rotations. Morton collisions break ties by input row. For equal probabilities, a uniform integer circular shift selects `n` distinct approximately equally spaced ranks from `N` rows, so **every row has exactly `n/N` inclusion probability for any spatial order**. Returned design weights are `N/n`.

For unequal inclusion probabilities, supply positive integer `inclusionTickets`. Within each stratum, total tickets must be at most `2^24`, divisible by sample size, and each row's tickets must not exceed `step = totalTickets/sampleCount`. A uniform integer start in `[0, step)` selects `start + j*step` along cumulative ticket mass in spatial order. Every row then has exact inclusion probability `tickets/step`; `tickets == step` is a certainty unit. The at-most-step constraint prevents selecting the same row twice. Outputs retain original frame IDs and reciprocal design weights. Returned f32 representations of those exact rational design quantities have ordinary rounding error.

`GPUSpatialBalancedSample` ranks rows with a stable GPU sort by (stratum, quadtree code) and a segmented prefix sum of tickets, so ties break by row index. Philox4x32-10 uses the registered `spatialBalancedSampling` purpose, seed and replicate; `provenance` records replay identity. Optional resident `parameters` words `[seed, replicate]` override constructor defaults and can change between encodings without recompilation; provenance records whether this resident source is active. Supply either resident `parameters` or a scalar `seed` (with optional `replicate`); when both are supplied, resident words are effective. `provenance.seed` remains the constructor hint (zero if omitted), inactive when `parameterSource` is `resident-view`; `effectiveSeedSource` explicitly names `resident-parameters`. Record each encoding's uploaded words with `getReplayParameters(uploadedWords)`; the returned copy remains valid after the upload array changes. This method refuses to invent resident words from an inactive scalar hint. Counter word three stays zero to preserve separation from counter-purpose streams. Bounded rejection sampling avoids modulo bias and fails explicitly after 128 rejected draws. Node tests enumerate all starts for equal and unequal designs, check uniqueness and exact inclusion frequencies, and compare spatial-rank coverage against simple random sampling. GPU tests exercise strata, repeatability, certainty units and invalid frames, then run 256 resident-parameter survey replicates to check empirical inclusion, Horvitz–Thompson total recovery, and geographic quadrant imbalance against the exact simple-random-sampling expectation. They also re-encode with changed frame-ID contents.

Use `decodeGPUSpatialBalancedSampleStatus(readback, allocationRow)` to read selected `count`,
`complete`, `invalidInput` and random-draw `iterationLimit` for one allocation row.

This is a specified hierarchical systematic probability design, **not a claim of bitwise or distributional equivalence to spsurvey GRTS**. The separate `GPUContinuousSpatialSample` below supplies continuous line/area frames, legacy exclusions, minimum spacing and independent replacement waves. Neither design supplies general design-based variance estimation.

## Bounded performance evidence

The opt-in `point-process-benchmark.spec.ts` exercises 32, 256 and 4096-row frames for OPTICS and probability sampling. It reports upload, graph construction, compilation, first execution, median warm execution, changed-input upload/execution and optional readback separately. GPU execution measurements are wall-clock encoding plus submission and queue completion, not hardware timestamps. Declared scratch-byte bounds exclude caller inputs/outputs and allocator overhead.

All 32/256/4096-row OPTICS cases now complete their ordering. Every reported run asserts the visited count, completion flag, uniqueness of the entire ordering, and all core/reachability values against the independent full CPU calculation after the input update. Dyadic coordinate perturbations avoid artificial cross-precision tie changes in this benchmark. The separate exhausted-frontier test still checks explicit partial-work behavior. Probability sampling completes every configured sample.

The following measurements predate the maximum-heap core selection and scan-based extraction; they are retained as historical evidence, not timings of the current implementation. The complete-ordering measurement record reports milliseconds on one AppleMetal WebGPU device; each warm GPU value is the median of three executions.

| Operation | Frame rows | Completed work | Warm GPU | Changed-input GPU | Full JS f64 OPTICS reference |
| --- | ---: | --- | ---: | ---: | ---: |
| OPTICS | 32 | 32 visits | 0.8 | 1.2 | 0.2 |
| Probability sampling | 32 | 8 selected | 0.6 | 0.6 | — |
| OPTICS | 256 | 256 visits | 4.0 | 4.1 | 3.6 |
| Probability sampling | 256 | 64 selected | 5.2 | 2.6 | — |
| OPTICS | 4096 | 4096 visits | 24.6 | 24.7 | 347.0 |
| Probability sampling | 4096 | 128 selected | 34.6 | 34.5 | — |

Cooperative GPU OPTICS shows a benefit over this exhaustive JavaScript reference at 256/4096 rows, while the 32-row case remains slower. This is not a comparison against indexed scikit-learn. Sampling has no optimized CPU/spsurvey timing comparator. The fixture is a deterministic grid with dyadic perturbations with minimumSamples 2; these measurements do not establish throughput for other minimumSamples settings, clustered frames or adapters. The earlier partial-frontier record is historical and is not used as complete-ordering evidence.

## Independent reference contracts

`generate-spatstat-fixtures.R` runs pinned R 4.4.3, spatstat.explore 3.8.2 and spatstat.linnet 3.5.3. Rectangle and polygon-hole fixtures compare `Kinhom(correction='border', renormalise=FALSE)` directly, including erosion-boundary equality. Epanechnikov density uses `sigma=bandwidth/sqrt(6)`, self inclusion and a 1024-pixel edge integral; the GPU analytic window mass agrees within 0.3%, reflecting reference grid integration error. Annular pair correlation remains available as the explicitly defined finite difference. `generate-translation-pcf.R` additionally executes direct `Kinhom` and `pcfinhom` translation estimates on rectangle, hole, concave and disconnected windows for supplied/self-included/leave-one-out intensity. Direct polygon references use a 1024-pixel set-covariance grid and a fine K radius grid; exact `edge.Trans(exact=TRUE)` polygon clipping plus explicit kernel sums provide a separately labelled composed reference. Radius/distance divisors and none/reflection correction are checked. Supplied-intensity exact sums use 0.03% relative tolerance; estimated-input comparisons allow 1.2% because the upstream intensity edge integral is rasterized. Direct rasterized estimates allow 2%; the compact-support formula itself is evaluated without histogram smoothing on the GPU.

DCLF fixtures call `dclf.test` on supplied binomial-CSR K curves using a common pooled reference and `leaveout=0`; the GPU receives that reference explicitly. The integration weights reproduce spatstat's interval-length times mean convention. This validates numerical inference, without treating the supplied example as a general fitted-model calibration guarantee.

Branch, cycle and disconnected network fixtures directly compare `linearKinhom` for Ang/none and all three normalization choices. Spatstat's histogram bin endpoint convention can differ at a pair distance; reference thresholds avoid those equalities, while separate exact-metric fixtures enforce this API's inclusive `<=` convention. KDE fixtures use spatstat shortest paths and independent R adaptive integration over each continuous edge, then check 64-subinterval midpoint quadrature and event-mass conservation. This radial normalization is a separately specified kernel contract; it does not claim parity with spatstat's equal-split diffusion estimator.

The survey design is a mathematically specified alternative to GRTS: exhaustive start enumeration proves exact first-order inclusion for equal and integer-unequal designs, and GPU empirical checks assess geographic balance and weighted total recovery. It intentionally does not reproduce spsurvey's traversal distribution or claim identical draws. Continuous-frame and replacement requirements use the separate random-shift design below.

## Executable point-process examples

These small synthetic runs capture the GPU graph and read back each result. The supplied curves illustrate envelope arithmetic; they do not establish a scientific null model.

### `GPUInhomogeneousPointPattern`

Estimates intensity-reweighted K and pair correlation inside a prepared planar study window. [Inputs, outputs and limits](#intensity-and-polygon-corrections).

{/* lens:GPUInhomogeneousPointPattern */}

::lens{scenario=gpu-spatial-analysis.inhomogeneous-point-pattern rung=result caption="Inspect the small synthetic result"}

::lens{scenario=gpu-spatial-analysis.inhomogeneous-point-pattern rung=graph caption="Inspect the executed computation graph"}

### `GPUPointPatternEnvelope`

Tests supplied resident curves against an explicitly declared simulation null. [Inputs, outputs and limits](#simulation-curves-and-global-tests).

{/* lens:GPUPointPatternEnvelope */}

::lens{scenario=gpu-spatial-analysis.point-pattern-envelope rung=result caption="Inspect the small synthetic result"}

::lens{scenario=gpu-spatial-analysis.point-pattern-envelope rung=graph caption="Inspect the executed computation graph"}

### `GPUSpatialBalancedSample`

Selects a finite-frame stratified probability sample and reports design weights. [Inputs, outputs and limits](#spatial-probability-sampling).

{/* lens:GPUSpatialBalancedSample */}

::lens{scenario=gpu-spatial-analysis.spatial-balanced-sample rung=result caption="Inspect the small synthetic result"}

::lens{scenario=gpu-spatial-analysis.spatial-balanced-sample rung=graph caption="Inspect the executed computation graph"}

### `GPUOPTICS`

Orders observations by reachability and extracts DBSCAN or xi clusters within declared work bounds. [Inputs, outputs and limits](#optics-ordering).

{/* lens:GPUOPTICS */}

::lens{scenario=gpu-spatial-analysis.optics rung=result caption="Inspect the small synthetic result"}

::lens{scenario=gpu-spatial-analysis.optics rung=graph caption="Inspect the executed computation graph"}


## Continuous line and area survey frames

The graph-value forms are `GPUContinuousSpatialSampleOperation` and
`CPUContinuousSpatialSampleOperation`; `computeContinuousSpatialSampleOnCPU` is the pure CPU twin.
Spacing and positions use the planar frame's coordinate units; design weights use length units
for lines and square units for areas. Seeds, replicates, capacities, provenance and status counts
are dimensionless. CPU forms enumerate all candidates and drop the GPU-only work caps.

`GPUContinuousSpatialSample` generates positions inside a continuous frame using a randomly translated lattice. It samples positions along line segments or inside triangle unions; the input vertices are not a discrete list of candidate sites. This is an explicit systematic probability design, not spsurvey GRTS distributional parity or a fixed-size design.

For `frame.type: 'line'`, `positions` defines a connected polyline. Omit `frame.bounds` for line frames. The direct and graph forms reject it during input validation. One uniform phase in `[0, spacing)` selects all distances `phase + k × spacing` below its total arc length. Segment interpolation returns continuous xy positions. The minimum separation is **arc length along this polyline**, not Euclidean distance between folded portions of the path. Repeated zero-length segments contribute no frame length. A zero-length whole frame is invalid.

For `frame.type: 'area'`, every three `positions` rows define a nondegenerate triangle. Their union is the area frame; disconnected pieces and holes are represented by the supplied triangles. Overlapping triangles do not duplicate sites, and provenance chooses the first containing triangle. Two independent phase coordinates uniformly translate a square lattice with the requested `spacing`. Only points inside the triangle union and the enclosing caller-supplied `bounds` are retained. Sites within each wave are at least `spacing` apart in Euclidean distance, to float32 coordinate precision. Input coordinates are planar and retain their units; geographic longitude/latitude must be projected first.

### Probabilities, weights and legacy sites

A continuous point has zero pointwise inclusion probability. The useful design quantity is **first-order sampling intensity**: `1 / spacing` per unit arc length for lines and `1 / spacing²` per unit area. The class exposes this as `samplingIntensity`; `designWeights` stores its reciprocal at every random site. Under ideal uniform continuous phase,

`E[sum(weight × f(site))] = integral over the eligible frame of f`.

This follows by integrating a translated lattice over one fundamental interval or square. No independence between sites within a wave is assumed. The implementation uses Philox4x32-10 with 24-bit uniform phases and float32 geometry, so the continuum identity is the design model, with finite-precision quantization in its numerical realization. Coordinates too coarse for the requested spacing (`max(abs(coordinate)) × 2^-20 > spacing`) invalidate the wave. The implementation does not claim exact real-arithmetic predicates.

`frame.legacySites` contains cumulative arc-length distances for lines or xy coordinates for areas. Sites must lie in the frame and satisfy the same mutual spacing metric. They are copied first with provenance role `2` and weight zero: they are fixed auxiliary sites, not random continuous-frame contributions. Random sites closer than `spacing` to a legacy site are excluded. Thus the eligible frame is the original frame **minus fixed legacy exclusion intervals/disks**; the intensity and integral identity hold on that eligible frame. These weights cannot recover arbitrary integrals over the excluded neighborhoods, and the fixed legacy values must not be given random-site weights.

### Replacement waves and completion

`waves` returns one primary design followed by complete, independently randomized replacement designs of the same eligible frame. Every wave retains its own minimum spacing and known weights. The combined pool need not satisfy minimum spacing across waves. A complete replacement wave can replace a whole design; averaging `R` complete waves uses each returned weight divided by `R`. Arbitrary selective substitution, informative nonresponse and taking a response-dependent prefix do not preserve these weights. Such follow-up policies require their own design analysis.

`provenance` has four uint32 components per output slot: source segment/triangle (or legacy row ID), stream replicate, lattice-candidate ordinal, and role (`0` primary random, `1` replacement random, `2` fixed legacy). Constructor `seed` and `replicate`, or resident `parameters: [seed, firstReplicate]`, replay the same phase. A scalar seed is required only without resident parameters. Both supplied sources remain accepted, with resident words effective; provenance identifies `parameterSource` and `effectiveSeedSource`. Save `getReplayParameters(uploadedWords)` with the record for replay. Each wave increments the replicate; overflow is invalid. Unwritten provenance is sentinel and unwritten positions/weights are zero.

The fixed-stride `status` row per wave is `[written, requiredSoFar, randomAcceptedSoFar, flags]`. Bit `1` means invalid geometry, legacy sites, parameters or numerical scale; bit `2` means output capacity exhaustion; bit `4` means candidate-work budget exhaustion, which only a line frame can report: an area frame's lattice of `(ceil(width / spacing) + 1) × (ceil(height / spacing) + 1)` cells is known from `bounds` and `spacing`, and construction throws, naming the required count, when `maximumCandidates` (default 65536) is smaller. When bit `4` is clear, the required count is exact even if storage overflowed. **Do not use partial waves for weighted estimates.** Re-encode with adequate capacity/work budget. `maximumCandidates` counts examined candidates, including those outside the frame or excluded by legacy sites.

Use `decodeGPUContinuousSpatialSampleStatus(readback, waveRow)` for named counts and flags,
and `decodeGPUContinuousSpatialSampleProvenance(readback, slotRow)` for source row, replicate,
candidate ordinal and role code. Decoders accept typed arrays, `Uint8Array` byte slices and
`ArrayBuffer`, and reject missing words or partial elements.

Bounds are 4096 input vertices (1365 area triangles), 256 legacy sites, 4096 waves, 65536 output slots per wave and 1048576 examined candidates per wave; combinations also obey device binding limits. Each wave runs serially in one invocation, so construction also bounds the work of one wave at 2^24 = 16,777,216 inner iterations: `(candidates + legacy sites) × (frame primitives + legacy sites × c + 1) + frame rows + capacity` must not exceed it, where candidates is the area lattice cell count or a line's `maximumCandidates`, primitives are triangles or segments, and `c` is 1 for area legacy sites and the segment count for line legacy sites (each is located along the polyline). For example, the default 65536 candidates admit a line of about 250 segments without legacy sites; a longer line needs a smaller `maximumCandidates`. The 2^24 budget is the one used by the other bounded serial reference kernels. There are no transient numerical buffers or hidden CPU sampling. The grid is a probability design, not a claim of accelerated geometry indexing.

Native GPU tests exercise 4096 independent triangle phases against analytic area, first moments and strip probabilities; 2048 legacy-excluded square phases against `1 - πr²`; overlapping triangles, continuous polyline support, replacement provenance/replay, changed inputs, capacity exhaustion and candidate-work exhaustion. These are distributional and numerical checks of the declared design, not identical draws or parity with spsurvey.

### `GPUContinuousSpatialSample`

This executable example draws a primary and an independent replacement design in a unit-square triangle frame. All eight captured random sites have weight `0.25`, and both status rows report four complete sites.

{/* lens:GPUContinuousSpatialSample */}

::lens{scenario=gpu-spatial-analysis.continuous-spatial-sample rung=result caption="Inspect continuous primary and replacement sites, weights and provenance"}

::lens{scenario=gpu-spatial-analysis.continuous-spatial-sample rung=graph caption="Inspect the actual GPU sampling graph and captures"}
