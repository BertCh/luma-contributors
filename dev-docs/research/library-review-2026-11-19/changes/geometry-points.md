# Changes for callers: Geometry, joins, search, zonal statistics, grids; point patterns, clustering, interpolation, regionalization, space-time and pair statistics

Part of [the record of caller-visible changes](README.md) from the 2026-11-19 and 2026-11-20 fix passes. Each bullet condenses what the implementers of those passes recorded; [README.md](README.md) says how the changes were tested and what that does and does not establish.

Counts in this file: results 90, rejections 73, added 58, deprecated 15, docs 20.

## Results that change

**Geometry measures and geodesics**
- `GPUGeometryMeasures` geographic group bounds join feature intervals on the first nonempty feature's longitude branch. A group that crosses the seam, or holds a duplicate endpoint already inside the first interval, no longer grows by a full turn. Multipoint bounds and mean centroids keep the longitude remainder of extreme equivalent coordinates in both serial and cooperative forms.
- `GPUGeometryMeasures` initialization writes only the scalar `invalidCount` word, so later words in a shared view are no longer overwritten.
- Geodesic inverse calculations reduce extreme finite longitudes before solving. Equivalent huge longitudes no longer leave the exact CPU inverse unresolved or produce a false coincidence in WGSL, and 0-degree and 360-degree origins give the same small positive distance in every metric.
- `GPUGeodesicPairs` with an explicit budget chooses the antipodal model from the direct small stored-coordinate gap. Boundary pairs that fell back to the sphere (about 19,887,694 m) now converge on the ellipsoid (about 19,877,222 m), which removes a CPU/GPU difference of more than 10 km. The small equatorial longitude supplement at the cutoff is kept.
- Great-circle arc endpoints keep small target longitudes from nonzero origins. In polar half-turn arcs and segmentize, interior points stay on the known endpoint's longitude branch instead of jumping 360 degrees after the pole.

**Lines and linear referencing**
- Cooperative per-row line measures keep small increments at the last moving vertex. Line chunking, segmentize, smoothing and linear referencing all build on these measures and inherit the correction.
- `GPULineChunk` and substrings resolve shared cuts the same way and keep zero-width terminal requests. Projection at fraction one reads the published endpoint measure, and location at the total length of a positive-length line returns the final coordinate.
- `GPULineLengthPerPolygon` keeps narrow clipped intersections in either segment direction, so clipped lengths, weighted lengths and contributing segment counts can rise from a wrong zero.
- Rectangle clipping checks the totals of all four clipping stages, so overflow in an intermediate stage is reported even when the final output fits.

**Offsets, predicates and equality**
- `GPUOffsetCurve` reconstruction measures numerical slack only along the relevant distance and probe directions. Covered legs no longer reappear after a translation, unobstructed offsets stay complete, coordinate collapses after publication are removed, and exact endpoint-circle tangencies no longer add cuts.
- Strict line containment (`within`) requires the interiors to intersect. Lines that touch only at a degenerate endpoint now return false; inclusive coverage still returns true.
- Euclidean `GPUCanonicalGeometryEquality` and `equalsExact` accept inclusive boundaries against the stored float32 operands (for example, distance 85 for a displacement of (13, 84)). Subnormal component differences that hardware may flush no longer decide the result. Values that round to the largest finite float32 are accepted again.
- `GPUMinimumClearance` breaks ties with the lowest original partner row whether spatial sorting is on or off.

**Validity, topology and polygonize**
- Restricted `GPUGeometryValidity` trees build without error. They report candidate loss through the aggregate `overflow` and suppress the dependent usability output.
- `GPUPolygonize` with a vertex tolerance prunes dangles only through direct endpoint contacts. A bounded face is no longer deleted by a spur or exterior endpoint touching two of its corners. The GPU pruning kernel previously failed to compile on WebGPU and now compiles.
- Host polygonization keeps the area of small translated faces; previously those faces vanished while their edges still reported closed rings.

**Triangulation, Delaunay and Voronoi**
- `GPUDelaunayTessellation` inputs with many groups use parallel insertion and flipping once the allocated scratch-triangle count passes 4,104, even below 2,049 sites. The command-budget estimate follows the route actually chosen.
- `GPUVoronoiDiagram` no longer sets capacity flags when the upstream solver is exhausted. Its optional `complete` is written on every submission and accounts for upstream, invalid-triangle and final-capacity losses.
- Earcut triangulation with `maximumWork` 4294967295 stops at the first unit over budget; previously the counter wrapped and the stopping condition was lost.

**Overlay, coverage and cleanup**
- `GPUPolygonOverlay` reports dropped open segments and continuation conflicts through `candidateOverflow`; local capacity overflow is reported separately as before.
- `GPUCoverageSimplification` returns the original coverage when topology-pair detection is incomplete, including when the optional statistics output is omitted.
- Vertex keys in `GPUGeometryCleanup`, `GPUCoverageDissolve` and `GPUCoverageSimplification` round half up without moving exact float32 grid points, and the predecessor of a half-cell is assigned to the correct cell.
- `GPUSegmentRingAssembly` given counted incomplete input suppresses its geometry when no completeness destination is supplied.
- `GPUCoverageValidity` writes only its one-word `invalidCount` and leaves trailing storage untouched.

**Labels, inscribed circles, skeletons and precision reduction**
- `GPUMinimumBounds` source envelopes enclose every finite labelled position, including points lost to hull quantization or a hull cap. Columns that depend on the hull keep their global-lattice meaning.
- `GPULabelPoint` clearance and the CPU and GPU `GPUMaximumInscribedCircle` radius and witness are computed at the published float32 center. If rounding moves the center outside the polygon, the GPU falls back to a representable boundary vertex. Search error keeps its search-only meaning.
- `GPUMaximumInscribedCircle` on the GPU projects from the nearer edge endpoint, so long split edges no longer overstate clearance (the review rectangle now gives radius 1 and witness (0, -1), not 1.414 and (1, -1)). Local search arithmetic can improve results at large origins.
- CPU label and inscribed-circle searches stop once they reach the half-shorter-envelope-side bound, including on ordinary elongated rectangles, and exactly collinear features finish at once with zero radius and zero error.
- `GPUVectorStraightSkeleton` and its CPU form keep both original support edges on the terminal two-vertex arc, including nonparallel supports such as a convex pentagon; an edge ID that was `0xffffffff` can now be the correct source row. Ridge provenance applies endpoint tolerance in position units and tests the ridge midpoint against each interval start, which corrects attribution on long split edges.
- `GPUPolygonPrecisionReduction` and `GPULinePrecisionReduction` certify cells from the original float32 coordinate and step. Positive and negative half-grid ties land correctly, and exact grid points stay put at large cell indices (polygon cells from 2^23, line cells from 2^24). Empty polygon rings no longer add origin candidates or false loss.
- `GPUNormalizePolygons` with no positions ranks the part offsets of declared empty parts (`[0,2,3]`, was `[0,0,0]`).

**Joins, search and gather**
- `GPUBufferSelection` on the CPU certifies membership at radius zero and compares positive radii robustly. False hits and zero distances that depended on endpoint direction are removed: an off-segment point is now rejected at radius zero in both directions and gets its positive distance at a positive radius.
- `GPUBufferSelection` on the CPU breaks float32 distance-rank ties by the lowest feature row, as CPU nearest does, so the winner can change where it previously went to the smaller unrounded distance.
- `GPUNearestFeatureJoin` in both planar modes keeps exact-radius matches across the distance-two rank boundary. Foot-point output keeps cancellation residuals, so symmetric distant projections publish the correct witness (for example (0, 0)) instead of displaced coordinates.
- `GPUSpatialPredicateJoin` called through its named operation without an explicit engine now uses the contributor's automatic selection, including the 1,024 average-vertex-product crossover.
- `GPUSpatialPredicateJoin` with a supplied live left count excludes allocated padding, so padding no longer becomes fabricated unmatched rows. Oversized live counts or uncertified classifications withhold all unmatched absence claims, and unused polygon tails no longer count as classification uncertainty.
- `GPUPairGather` withholds derived unmatched rows when global loss is supplied, instead of asserting that they are absent. Matched rows are kept, and complete reused frames restore the rows and clear loss.
- Prepared point-in-polygon, predicate and nearest joins (`GPUSpatialJoinPrepared`) refine against the same retained coordinates and offset hierarchy as their bounds until the requested generation runs. Source-ID mapping stays live.
- Prepared polygon indexes with edge slabs now declare successfully.
- Reverse submission of prepared generation requests can no longer apply a request encoded later to an earlier one.
- Capacity plans for predicate joins include graph-owned diagnostic words, which raises the default pair estimate by four bytes.

**Zonal statistics and areal interpolation**
- `GPUZonalStatistics` variance correction uses stable tree reductions, so constant features no longer get a false positive variance. Signed-weight variances keep negative values and explicit NaN standard deviations.
- `GPUZonalStatistics` feature-row inputs propagate geographic membership counts and clear them on clean reuse.
- `GPUArealInterpolation` rewrites the requested nested count and candidate-loss statuses on every submission, so stale loss no longer suppresses the lag. Pair overflow is always published on directly consumed weights, so truncated weights produce NaN downstream instead of a finite partial lag.

**Grids, sampling and neighbor search**
- `GPUDotDensity` keeps integer requests at float32 subtraction boundaries on the values route, which corrects an undercount and matches the integer-count route.
- `GPUDotDensity` with negative mask steps and feature normalization keeps the intended relative acceptance probabilities. Previously an inverted cell-range scan admitted positive mask weights without normalization, so output distributions can change.
- `GPUNeighborSearch` given reused inverse-distance settings from nearest-feature weights computes inverse-distance weights; it previously selected binary weights silently.
- `GPUMapColoring` uncolors every row on loss and clears the result scalars while keeping the iteration count. Zero conflicts certify a proper coloring only when the weights are symmetric and complete.

**Density**
- `GPUKernelDensity` automatic Scott and Silverman bandwidths no longer change when only the output grid pans, including for translated observations with sub-metre spread. They are also unchanged by a common scaling of weight units.
- Gaussian narrow-window overlaps and thin compact-kernel cap overlaps return small positive mass, where cancellation previously gave negative or zero values.
- Bandwidth and density populations use absolute rectangle membership, and automatic KDE crossover decisions use single-rounded products instead of reciprocal division.
- Space-time density default slice endpoints match explicit fused evaluation; for a first endpoint of 0.1, interval 0.05 and 8 slices the endpoint is 0.45000001788139343.

**Interpolation and prediction**
- `GPULocalKriging` indexes each observation in exactly the grid cell its neighbor walk reads, including at translated cell edges.
- `GPULocalKriging` and `GPULocalRBF` given producer loss report status 3 and suppress partial predictions, then recover on complete frames.
- Vecchia preparation returns distinct, correctly ordered nearest neighbors for exterior queries, keeps healthy neighborhoods when masked sites hold NaN, and works with empty, coincident and single sites.
- `GPUVecchiaLogLikelihood` and `GPUVecchiaFisherScoring` suppress dependent totals when a neighborhood loss certificate reports loss, and recover when it clears. Either loss word suppresses decoded results even when `complete` is 1.
- Short or partial Vecchia readback throws instead of inventing totals or status.
- `GPUVecchiaFisherScoring` carries sill-dependent nugget-floor derivatives even when the nugget is fixed at zero. History holds only accepted steps (empty on immediate convergence or stall), and a stalled line search keeps the last accepted fit. A coordinate with zero score and zero information is held fixed, and an unusable system stops at the last accepted fit instead of evaluating an invalid covariance.
- Space-time smooth covariance models apply the nugget floor at the exact coincident discontinuity. Smooth models with zero nugget and tiny ranges now reject instead of bypassing the floor.

**Point patterns and pair statistics**
- Extreme-rank envelopes honour equality at attainable significance levels: `alpha = 1/49` gives `[1,47]` instead of `[0,47]` in the 49-curve boundary case.
- `GPUPointPatternEnvelope` integrated inference on the GPU and the CPU compares exact dyadic products and sums and rounds once with ties to even. Near-ties can return smaller, correct Monte Carlo p-values (the pinned case changes from 1 to 1/20); displayed float32 statistics are unchanged.
- `GPURipleyEnvelope` reports a NaN probability when no simulated curve has a usable deviation, and keeps the deviation and count.
- Quadrat classification compares exact integer products at rounded ties and stays accurate for extreme finite frames whose reciprocal flushes.
- `GPUVariogram` bin edges use exact float32-by-integer comparisons on both lag axes, so interior ties no longer round into the preceding bin.
- Spatial variogram, correlogram and `GPUSpaceTimeVariogram` fixed-point units leave out rows excluded from the pair population, including outside or nonfinite positions in the robust IQR. Such rows no longer round retained semivariances to zero. `GPUSpaceTimeVariogram` also reads position views that have a byte prefix correctly.
- Pair-statistic fixed-point encoding uses direct-reciprocal rounding whenever the physical reciprocal is representable (pinned semivariance changes by one float32 ULP). Selected-scale decoding divides exact integer significands, so Metal reciprocal rounding or flushing can no longer give zero or low-rounded results with a clear overflow flag.
- `GPUSpaceTimeKFunction` status 3 also covers unsupported intensity-weight scales and positive PCF terms below half an accumulator unit. On loss every estimate, local outputs included, is NaN even when status is omitted. Small Epanechnikov products are kept through a finer fixed-point scale.

**Clustering and ranking**
- `GPUSpatialKMeans` centers coordinates by default, which corrects distances and inertia for small projected-coordinate clusters. It keeps one origin per cluster: for `[1e7, 0.125, 0.375]` with k=2 the small cluster now has center 0.25 and inertia 1/32 (previously 0 and 0). Pass `centerCoordinates: false` to reproduce the old arithmetic.
- `GPUSpatialKMeans` no longer fails WebGPU shader compilation in its relocation kernel; previously every `GPUSpatialKMeans` graph failed on hardware.
- `GPUSpatialKMeans` Lloyd convergence uses a scaled center-shift norm, so a tiny nonzero movement no longer underflows into false convergence. Direct results distinguish convergence in the last round from exhaustion.
- `GPUSimilarLocations` z-score ranking keeps standardized distances for tiny attribute units instead of tying them at zero, and stays finite for opposite large finite inputs.
- `GPUSinkhornDivergence` bounded CPU form honours an explicit `transportRadius` and keeps unreachable-mass residuals, so it can report nonconvergence where it previously certified convergence on full support. Empty or invalid frames report approximation 1 with zero iterations, and large reachable finite potentials no longer count as missing mass.
- `GPUOPTICS` keeps extreme search widening finite.

**Regionalization**
- `GPURegionPartitionEvaluation` keeps subtraction and reduction residuals in regional means (one region `[2^-84, 2^-110, -2^-84]` had mean 0 and now has 2^-110/3). Regions with identical distributions give exactly zero between-region SSD at any scale (the reported fixture gave 0.75, others 2.4e-30 or 3), and the geographical detector's q inherits this.
- Shared regionalization standardization keeps the sign and size of genuine tiny residuals, so downstream regionalization decisions can change.

**Accessibility and allocation**
- `GPUBoundedFacilityAllocation` clears its decision certificate and p-center optimum claim when costs are provisional or a finite horizon is supplied, and recovers on a later complete submission.
- `GPULocationAllocation` applies a supplied matrix `costLimit` to status `complete` even when `costsComplete` is omitted, on both GPU and CPU. A short-horizon matrix is no longer reported complete; objectives and assignments are still those of the visible matrix.
- Accessibility consumers are suppressed independently by positive weights completeness.

**Space-time scans and inference**
- `GPUSpatialScanStatistic` Poisson scans treat positive cases at zero exposure as an invalid frame and suppress inference. `invalidCount` counts the incompatible cells. Positive exposures more than 126 binary orders below the maximum count as numerical loss, and inference is suppressed even without the optional status destination. Invalid frames report zero Monte Carlo draws evaluated.
- `GPUExpectationBasedSpaceTimeScan` orders extreme finite ratios without quotient overflow, so the correct high-ratio subset is selected; exact ties keep their slot rule.
- Spatial scans accumulate fractional population totals, prefixes and complete-circle masses consistently and keep exact mass through the packed fraction multiplication. Every eligible window lost at an inclusive population or capacity boundary is reported, and rounding no longer turns a complete-study circle into a positive cluster.
- `GPUPettittTest` returns NaN for cells with fewer than two finite observations, which removes untested cells from downstream Bonferroni families. Tested constant series still return p = 1.
- `GPUEmergingHotSpots` aggregates neighborhood loss once before inference and publishes `UNDETERMINED` (17) with NaN trends on loss, even when `minimumTrendSliceCount` is 0. Results return when loss clears.
- `GPUEmergingHotSpots` Gi* keeps small coefficient deviations in dense rows regardless of CSR order. Masked or nonfinite neighbors no longer choose centering origins, and temporal sums rebase before merging. Normal-tail significance and cold/hot slice counts can change.
- A missing neighbor no longer turns a valid tiny-focal-weight Gi* score into NaN (`[1,NaN,5]` with self weight 1e-8 now gives z = -1).
- `GPUSpaceTimeHawkes`, its operation form and the CPU form fit decay against the finite-window likelihood, refit the branching ratio for each candidate decay, and undo an update that lowers the likelihood. Fitted parameters, iteration counts and convergence differ from earlier releases.
- The `GPUSpaceTimeHawkes` CPU form's default iteration budget is now 100, as on the GPU (was 1000). Calls that omit `maximumIterations` can stop at 100 with `iterationLimitReached`; pass `maximumIterations: 1000` for the former budget.
- `GPUMantelTest` and its CPU form report undefined observed correlations and inference as NaN with zero usable draws, keeping the pair count. Undefined null draws are left out of inference denominators instead of counted as zero. Constant-vector inputs that gave 0 and p = 1 now report inference as unavailable.
- `GPUMantelTest` compares and summarizes Pearson statistics with their residuals. Exact ties count as exceedances, exact zero covariance publishes 0, and null variance is zero only for identical null correlations. GPU centering follows each draw with exponent-normalized denominators, keeping valid large-gap correlations finite when the raw variance product overflows. On incomplete neighborhoods the usable count is zeroed with the attempted count, and the CPU form suppresses incomplete producer input before reading invalidated CSR rows.
- `GPUKnoxTest` Poisson tails keep representable probabilities for subnormal means whose count-to-mean quotient overflows (for example, P(X >= 1 | mean = 1e-310)).

## Input now rejected at construction

**Geometry measures and lines**
- `GPUGeometryMeasures` rejects length or area requests in a known angular planar frame, even with metric `none` and omitted result units; project to a linear frame first. Partial explicit unit metadata can no longer hide an angular CRS.
- `GPULineSimplificationOperation` checks `rowCount` and `lineCount` as nonnegative integers and rejects unsupported inherited or non-enumerable status destinations instead of dropping them. Unknown method strings are rejected; the error lists `douglas-peucker` and `visvalingam`.
- `GPULinearReferencing` rejects a radius view that is not exactly one packed float32 row (lengths 0 or 2), naming the received length. It also rejects wrong port formats before graph declaration.
- `GPUGeometryCleanup`, `GPULineSimplification` and rectangle clipping reject settings that overflow float32 or whose positive grid or tolerance underflows to zero.

**Predicates, splits and restricted trees**
- Restricted-tree wrappers (split, clip, shared paths, predicates) with a restricted `leafCapacity` require an observable loss diagnostic: `candidateOverflow` on split, `overflow` on predicates, `candidateOverflow` or aggregate `overflow` on clip and shared paths. Network noding that delegates restricted leaves to split now throws if both `pieces.status.candidateOverflow` and the network `status.overflow` are omitted.
- Split and clip reject requested `complete` and `unresolvedCount` status views, which have no writer; use the overflow and candidate diagnostics.
- `GPUSharedPaths` rejects aligned overlap among the final scalar dispatch's status words, among its four run outputs, between each provenance pass's identities and part rows, and between a provenance output and `status.count`.
- `GPUGeometryValidity` rejects adjacent polygon `status.overflow` and `status.intersectionCount` words.
- `GPUGeometryPredicates` rejects a non-object `spatialContext` or `otherSpatialContext` with a message that names the contributor.
- Precision modes accept only `valid-output` and `keep-collapsed`; other values reject with structured details.

**Triangulation, Delaunay and Voronoi**
- Named `GPUDelaunayTessellationOperation` forms reject omitted or empty `count`, `requiredCount` and `overflow` status names.
- `GPUDelaunayTessellation`, `CPUDelaunayTessellationOperation` and `GPUNeighborSearch` reject supplied `complete` and `unresolvedCount` destinations; the CPU Delaunay form previously advertised them and returned `undefined`. `GPUVoronoiDiagram` rejects `unresolvedCount`.
- `GPUVoronoiDiagram` rejects `count` or `overflow` sharing a physical buffer with `requiredCount`, and `complete` sharing one with `invalidCount` or `requiredCount`.
- Subdivision rejects same-length wrong-format inputs, offset tables without a terminal entry, invalid work or routing settings, unsupported spatial context and methods other than `convex` or `rectilinear`, all before any scratch is reserved. Null routing and null method are invalid rather than defaults.
- Named CPU polygon triangulation forms with `storage: 'compact'` reject orphan rows instead of silently using resident behaviour.

**Overlay, coverage and assembly**
- `GPUCoverageDissolve` and `GPUCoverageSimplification` reject `snapGridSize` and `snapTolerance` given together with different values.
- `GPUCoverageDissolve` input `labelsComplete` requires `output.status.labelsComplete`.
- `GPUSegmentRingAssembly` requires `count` and `segmentStatus.count` to name the same view when both are given, and requires `output.polygons` when `status.polygonCount` is requested. The assembler and dissolve reject read/write sharing of one physical buffer in completeness, conflict and polygon-count dispatches, even for disjoint aligned slices.
- `GPUPolygonOverlay` rejects non-power-of-two leaf capacity, empty nested assembly vertex or boundary capacities, and unknown `operation` or `fillRule` values (including `xor` and the empty string) before graph registration.

**Labels, circles, skeletons, precision and bounds**
- Both `GPUMinimumBounds` routes require `positions.length` to equal the declared `rowCount`, and the polygon route requires `featureOffsets.length` to equal `featureCount + 1`.
- Compact host hierarchies require an initial zero, monotonic in-range offsets and exact terminals; malformed compact terminals reject instead of being treated as padding. Empty levels `[0]` are valid. Select resident storage explicitly for noncompact layouts.
- Compact skeleton hierarchies reject terminal-only offset tables that do not cover their children (`[0]` requires zero children).
- `GPUMinimumBounds`, label and similar named outputs reject conflicting destinations for `sourceBounds`/`bounds` and `publishedCenterClearance`/`distances`.
- `GPUPolygonPrecisionReduction` rejects final-only `requiredRingCount` on intermediate segment or edge status, including through a getter-only proxy. Use the intermediate `requiredCount` or the final `output.status.requiredRingCount`.
- `GPUSegmentSnapOperation` rejects `maximumVerticesPerLine` outside `[1, 0x7fffffff]` at construction, as the CPU form already did.

**Joins, search and gather**
- `GPUBufferSelection` and both selection operation forms reject `radius` and `distance` given with different values.
- `GPUPointInPolygonJoin` rejects the new row and count outputs sharing a physical read/write dispatch buffer with source IDs, and incompatible membership-copy aliases.
- `GPUPairGather` rejects missing destinations and conflicting `status.count`/`rowCount` aliases, and a truncation view whose length differs from the query count. New loss inputs require a candidate-overflow destination.
- Named predicate join factories reject any separate unmatched diagnostic without inner `unmatchedRows`, including on anti joins, and malformed numeric CPU producer statuses.
- Named GPU segment selection rejects bound `featureStarts` or `featureEnds` lengths that differ from the captured descriptor.
- Metric `dwithin` cannot request relate output through either the planner or the constructor.
- Bounds parameter packing rejects wrapped inversions erased by float32 rounding and endpoints that encode as nonfinite, before modifying the target. Relate pattern packing rejects malformed or unrepresentable slot capacities.
- Prepared memory plans reject incomplete geometry counts: polygons need `polygonCount` and `ringCount`, multipart lines need `partCount`.
- `graph.add(prepared)` on `GPUSpatialJoinPrepared` rejects `geometry.sourceIds` from another graph. `GPUSpatialJoinCandidates` with `prepared` again rejects `left`, `right` and source-ID views from another graph.
- `prepareGeometryInput` rejects conflicting explicit field options; field encoding and metadata are authoritative.

**Zonal statistics and areal interpolation**
- `GPUZonalStatistics` feature-row `status` scalars each require the matching output status scalar; malformed, cross-graph and physically aliased views reject.
- `GPUArealInterpolation` rejects unsupported diagnostic destinations, including `complete` and `unresolvedCount`, with their full path. Nested `count` must use a different physical buffer from offsets, and a distinct nested `requiredCount` and outer `requiredPairCount` must use different buffers; disjoint bytes of one buffer do not count. Supplied nested aliases must be genuine one-row uint32 aliases.

**Grids, sampling and neighbor search**
- `GPUGridGenerator` point grids and square grids that request centers reject either axis above 8,388,608 cells; square output without centers keeps the 16,777,216 limit. `getGPUGridVerticesPerCell`, `getGPUGridCellCount` and `getGPUGridCornerCount` reject unknown grid types.
- `GPUNeighborSearch` rejects finite positive planar distance limits whose float32 square overflows, including the legacy `radius` (`Infinity` still means unbounded kNN). Mode-aware parameter packing rejects absent, nonpositive or nonfinite radius bands, and `timeBucketCount` without `times`. Unknown `weightType` names and disagreeing `weightType`/`weightKind` reject.
- `GPUDotDensity` rejects inconsistent simultaneous `maskPlacement` and `maskExtent`; polygon samplers reject conflicting `slotStarts`/`slotOffsets`.

**Density**
- `GPUKernelDensity` rejects `bandwidthBelowThreeCells` on an explicitly exact route, and on a view that is not one packed uint32 row or that overlaps another output.
- `GPULineDensity` requires the seam scalar to have exactly one row and its writable outputs not to overlap.
- Density parameter packers reject bandwidths that encode as zero or nonfinite, float32-collapsed or nonfinite grids and windows, and multi-slice controls that do not advance. Unknown kernel names and invalid `smoothing.strategy` reject with the accepted names.
- Raster line-density packing and output mapping reject a minimum whose float32 encoding moves placement by at least one cell; subtract a common local origin. Mapping also rejects a step that underflows on its own.
- `GPUSpaceTimeRelativeRisk` requires `output.values.length >= grid.width * grid.height * sliceCount` even when both intermediate density views are supplied.

**Interpolation and prediction**
- `GPUKriging` and the IDW, raster kriging and Gaussian-field packers reject collapsed float32 grid placement and unusable encoded radii, ranges, powers or total variance, with common-origin guidance. Raw IDW negative or NaN radii and negative-power nodata controls stay accepted. Rejected packs leave destination bytes unchanged.
- Co-kriging rejects coefficients or normalization scales that overflow float32. Space-time packing rejects unusable derived Gneiting coefficients and invalid model anisotropy; raster kriging rejects inherited model names; covariance packers reject coercible strings and nonnumeric fields.
- Prediction constructors reject wrong storage formats with named ports and unsupported complete-only producer polarity.
- GPU simulation operation forms reject input view lengths inconsistent with `observationCount` and `queryCount`.
- `GPUKriging` and IDW wrapped great-circle search requires index longitudes in [-180, 180].
- Vecchia setup rejects malformed XY, short mask or time columns, wrong selected-order length, duplicate or out-of-range permutations, conditioning widths outside 1..32, nonpositive, noninteger or unrepresentable prediction widths, unrepresentable allocation products, missing or nonfinite explicit `timeScale`, and nonfinite queries or ordering coordinates. Masked finite rows (including masked NaNs) and geographic conformal prediction widths up to 256 stay allowed.
- Vecchia graph forms reject wrong-format ports, explicitly empty loss or evidence names, overlapping publication bindings, and an independent positive `neighborhoodStatus.complete` certificate on all three routes, even next to a zero overflow flag. CPU loss inputs must be nonempty `Uint32Array`s.

**Point patterns and pair statistics**
- `GPUSpaceTimeKFunction` parameter packing rejects derived float32 width, height, duration, area or volume that is nonfinite, zero or below 2^-126 (for example endpoints [-2e38, 2e38]).
- `GPUVariogramModelFit` and the CPU form reject invalid model or weighting choices before checking bin eligibility, including invalid CPU weighting with too few bins.
- Point-pattern and variogram packing reject disagreeing new and legacy aliases.
- `typePair` must contain exactly two present uint32 identifiers.
- CPU coregionalization graph operations reject empty or colliding destination names.
- Integrated envelope inference rejects omission of a vector the statistic uses and a mismatched explicit simulation count.

**Clustering**
- `GPUHierarchicalDensityClustering` (DBSCAN) planar parameter packing rejects an encoded epsilon outside `[2^-63, 2^63]`; geographic angular limits are unchanged. Unknown directions and malformed bounds reject before packing.
- `GPUSinkhornDivergence` CPU form throws with migration instructions for a legacy short `blurRadius` it used to ignore, and any form rejects `blurRadius` and `transportRadius` together.
- `GPUSimilarLocations` rejects unknown ranking directions and invalid standardization names.
- GPU facility operations reject dense lengths other than `demandCount * facilityCount` and sparse offsets of length other than `demandCount + 1`.

**Regionalization**
- `GPURegionPartitionEvaluation` input `labelsComplete` requires `output.labelsComplete`.

**Accessibility and allocation**
- `GPULocationAllocation`, its named form and the matrix adapter reject unknown `operation` values, listing `p-median`, `maximum-coverage` and `set-covering`.
- `GPUCatchmentAccessibility` and `GPUHuffTradeAreas` reject `demandValidity` and `validity` aimed at different destinations.
- The labelled network allocation adapter rejects missing convergence, inconsistent direction or axis, and grouped facility starts that cannot map to one node.
- `GPULocationAllocation.estimatedCommandNodeCount` now counts hierarchical compaction passes, so configurations just under the 32,768-node gate can now exceed it and be refused.

**Space-time scans and inference**
- Spatial-scan packers reject unknown `windowShape` and `rateDirection` values (including null and prototype names) with the accepted values, and null `model` or `temporalMode` instead of selecting defaults.
- Bounded floating-population discovery needs twelve tail words per center (was two), and binding preflight can refuse a lowering that previously fitted.
- GPU change-point operations reject `sliceCount` above 512, slope `sliceCount` above 256, `maximumChangePoints` above 128, and invalid cost options or dimensions; CPU forms keep their larger shapes. Direct location-change `status.overflow` must be one uint32 row.
- `GPUSpaceTimeLocalMoran` rejects missing or wrong-format cube and parameter ports, unknown layout, variant and direction values, and invalid delegated inference options and binding sizes before reserving scratch.
- Relative-time preparation rejects nonfinite encoded times, float32 overflow, insufficient resolution at encoded magnitudes (including the subnormal quantum), rebased epochs outside signed Int64, unknown units and invalid time-zone annotations. Standalone CPU Mantel rejects incompatible temporal encodings, and unknown `summaryField` values reject.
- Malformed spatial-weights status scalars on Knox and Jacquez, which the old snapshot dropped, now reach the uint32/length validator.

## Added

**Geometry**
- `GPUCanonicalGeometryEquality` `toleranceMetric`, default per-coordinate, with a Euclidean mode that shares the inclusive comparison of `equalsExact`.
- `GPUMinimumBounds` `sourceBounds` (new spelling of `bounds`) and optional `precisionLimited`.
- `GPULabelPoint` and `GPUMaximumInscribedCircle` `publishedCenterClearance` (float32), `precisionLimited` (uint32) and `witnessOffsets` (float32x2) per feature.
- Group geometry, hull consumers and `GPUCoverageDissolve` (`labelsComplete`, `output.status.labelsComplete`) can carry solver-label completion provenance separately from geometric overflow.
- `GPUSegmentRingAssembly` `segmentStatus`, exported `GPUSegmentRingInputStatus`, optional `status.candidateOverflow`, `status.complete` and `status.polygonCount`.
- `planGPUPolygonOverlay`, `preflightGPUPolygonOverlay`, `GPUPolygonOverlay.preflight` and the types `GPUPolygonOverlayPlan`, `GPUPolygonOverlayPlanningProps` and `GPUPolygonOverlayOperandDimensions`, for checking device and scratch limits before any stage is registered.
- `GPULineLengthPerPolygon` `clippedLengths` output with companion props and output types.
- Named Delaunay operations support count-only, zero-capacity triangle output with `requiredCount`, `count` and `overflow` status. Zero-polygon triangulation accepts zero index capacity, and empty subdivision needs no scratch.
- CPU polygon and constrained triangulation `storage: 'compact'` for strict compact-offset validation.
- Line chunk and line locate parameter packers accept omitted settings.
- Zero-row `GPUMaximumInscribedCircle`, `GPUNormalizePolygons` and `GPUSegmentSnap` construction, and canonical empty batches with defined diagnostics for declared empty parents.
- `GPUGridGenerator` host admission method and an exported limit constant for the center-specific axis bound.
- `getGPUShapePolygonIdentityOffsets` for combining several generated features; polygon samplers expose `slotStarts`.
- `GPUDotDensity` named `maskPlacement` with signed cell steps.

**Joins, search and gather**
- `GPUPointInPolygonJoin` `polygonAssignmentCounts`, `polygonMembershipCounts` and `pointFeatureRows`, also on the named forms; `GPUNearestFeatureJoin` `nearestFeatureRows`.
- `GPUBufferSelection.getMatches(graph)` returns `GPUBufferSelectionMatches`, a graph-owned compact result with a candidate flag that works with `output: {}`.
- CPU polygon counting in a counts-only form that skips dense membership, exposing `candidateBoundsTestCount`; named CPU predicate and nearest result types.
- `prepareGeometryInput` adapts native GeoArrow geometry with source identity, chunk offsets, null and empty masks, frame metadata and explicit float64-to-float32 loss handling.
- `GPUSpatialPredicateJoin` typed `outputPairs` that always carry `candidateOverflow`; named inputs `leftFeatureCount` and `leftStatus.{overflow,candidateOverflow,complete}`; inner `unmatchedRows`, `unmatchedCount`, `unmatchedRequiredCount`, `unmatchedOverflow` and `unmatchedCandidateOverflow`, computed before pair truncation. The pure CPU form adds `unmatchedRows`, `unmatchedStatus` and numeric `leftStatus`.
- `GPUPairGather` optional neighbors `status: {overflow, candidateOverflow?}` and `candidateTruncated`.
- A host geometry readback decoder that trims every offset level while keeping live empty features.
- Prepared memory and predicate route planners that report padded whole-domain allocation costs and the executable route; optional `graphOwnedStatusCount`.
- `GPUSpatialJoinPrepared.getGeometry(graph)`, `GPUSpatialJoinPrepared.setRetryViews(graph, {geometry?, storage?})`, `GPUStudyWindowPreparation.setRetryViews(graph, views)` with `GPUStudyWindowPreparationRetryViews`, and `recoverFailedDeclaration` after a node-registration failure. Preparation allocation failures release owned storage so persistent inputs can be retried on a fresh graph.
- `GPUNeighborSearch` exposes its actual route and numeric grid and list plans, mode-aware distance settings, and execution-plan types and planner.
- `GPUMapColoring` and its operation form accept optional `overflow` and `candidateOverflow` in `weights.status` and bind packed scalar outputs with loss ports.

**Zonal statistics**
- `GPUZonalStatisticsFeatureRows.status` with `overflow`, `candidateOverflow`, `uncertainCount`, `antimeridianUncertainCount` and `poleUncertainCount`.

**Density and interpolation**
- `GPULineDensityOutput.status.antimeridianUncertainCount` and `GPUKernelDensityOutput.status.bandwidthBelowThreeCells`.
- Explicit bandwidth population choice with a live row count; a grid-sized, offset-preserving `outputBand` on fixed KDE; numeric route limits and allocation planners for point and fixed density.
- Temporal `bisquare` kernel name across density and relative-risk forms, and `quartic`/`bisquare` neighbor kernel names.
- A raster-output-mapping helper for line density that gives explicit row and column permutations, including north-up grids.
- `longitudeSearch` on IDW and raster kriging, default clipped, with a wrapped great-circle mode that includes antimeridian neighbors and polar caps.
- `GPULocalKriging` and `GPULocalRBF` resident `neighborhoodWeights`/`neighborhoodStatus` inputs and loss status code 3. Gaussian-field conditioning can forward CSR offsets and loss with a separate `candidateOverflow`.
- Per-model prediction status decoders and an IDW instance parameter packer that checks nearest-neighbor requests against the allocation.
- `VecchiaFisherFit.stopReason` (`converged`, `iteration-limit`, `initial-failure`, `line-search-stall`) and `VecchiaFisherFit.stallReason`.
- Vecchia neighborhood loss input, an optional output certificate, CPU evidence, named decoders and prepared-neighborhood counts with upload mapping.

**Point patterns and pair statistics**
- `GPURipleyDistanceFunctionsProps.hanischSearch`, capped by default, with an exact rectangle-only search.
- `fitVariogramModel(..., {diagnostics: true})` returns `VariogramModelFitDiagnostics`; space-time fitting reports `cycleCount` and `converged` and accepts `maximumCycles` in [1, 24].
- Point-pattern settings `analyticAlternative` and `indexCellSizeHint` with `GPUPointPatternIndicesHintSettings`; variogram setting `sectorOffsetRadians`.
- Envelope vectors that depend on the statistic, an inferred simulation count, and envelope work-limit constants.
- Named decoders for envelope statistics and status, balanced per-allocation status, continuous per-wave status and per-slot provenance.
- Spatial sampling accepts resident seed words without a scalar seed and can snapshot them for replay records.

**Clustering and regionalization**
- `GPUOPTICSOutput.ports.status.reason` with `GPU_OPTICS_REASON` (completion, candidate overflow, visit-budget exhaustion), and a reason word in the CPU result; optional `id` on pure CPU OPTICS and facility options.
- `GPUSinkhornDivergence` `transportRadius` in grid cells, where 0 means diagonal-only transport.
- Direct `GPUSpatialKMeans` and `GPUSinkhornDivergence` results expose graph-owned stopping views when destinations are omitted, plus k-means `iterationLimitReached`.
- Contributor-bound `getParameterValues` methods for DBSCAN temporal windows and Sinkhorn epsilon schedules.
- `compactRegionLabels` and `CompactRegionLabelsResult`, which keep join keys and missing labels.
- `GPURegionPartitionEvaluation` input and output `labelsComplete`.
- `GPUMaxPRegions` evidence output, CPU `result.evidence`, `GPU_MAX_P_EVIDENCE`, `decodeGPUMaxPRegionsEvidence` and `GPU_MAX_P_STATUS_WORD_COUNT`.

**Accessibility and allocation**
- `GPUBoundedFacilityAllocation` inputs `costsComplete` and `costLimit`, `output.complete`, `status.complete`, a named output type and a checked status decoder.
- A labelled network-matrix adapter for location allocation that keeps travel direction and axis identities and maps facility-row assignments back to network nodes.
- `GPUAnalysisSpacing` readback decoding of raw bytes with `invalidCount` and a detailed decoder.
- `demandValidity` on `GPUCatchmentAccessibility` and `GPUHuffTradeAreas`.

**Space-time scans and inference**
- Indexed candidate and axis work estimates and radix sort counts on both scans, and `pValueConstruction: 'upper-tail-count'`.
- `GPUExpectationBasedSpaceTimeScan` static limits and planner with `GPU_EXPECTATION_BASED_SCAN_LIMITS`.
- Optional `id` on `getGPUSpatialScanStatisticLayout`, `getGPUEmergingHotSpotsParameterValues`, `CPUPettittTestOptions` and `CPULocationChangePointsOptions`.
- `GPUMantelTest` `usablePermutationCount`, `temporalContext`, `GPU_MANTEL_TEST_SUMMARY.observedCorrelation` and `decodeGPUMantelTestSummary`. `decodeGPUSpaceTimeInferenceMetadata(metadata, attemptedCount, usableCount)`, `GPUSpaceTimePValueField` and a summary-field metadata selector.
- `GPUSpaceTimeHawkesOutput.status.maximumIterations`, the same name on `SpaceTimeHawkesOperationProps`, and `CPUSpaceTimeHawkesResult.maximumIterations`.
- `GPUSpaceTimeLocalMoran` accepts `output.permutationCounts` on fixed-budget runs; it reports the draws made per tested row and `0xffffffff` for untested rows.

## Deprecated spellings

- `featureCounts` on `GPUPointInPolygonJoin`: use `polygonAssignmentCounts` or `polygonMembershipCounts`.
- `distance` on `GPUBufferSelection` and `CPUBufferSelectionInput`: use `radius`.
- `lowest-id` nearest tie rule: use `lowest-row`.
- `bounds` on minimum-bounds outputs: use `sourceBounds`. `distances` on label outputs: use `publishedCenterClearance`.
- `snapTolerance` on `GPUCoverageDissolve` and `GPUCoverageSimplification`: use `snapGridSize`.
- `rowCount` on `GPUPairGather` output status: use `status.count`.
- `weightKind` on neighbor search: use `weightType`.
- `maskExtent` on `GPUDotDensity`: use `maskPlacement`. `slotOffsets` on polygon samplers: use `slotStarts`.
- `lengths` on `GPULineLengthPerPolygon`: use `clippedLengths`.
- `validity` on `GPUCatchmentAccessibility` and `GPUHuffTradeAreas`: use `demandValidity`.
- `zscore` on `GPUSimilarLocations`: use `z-score`.
- `blurRadius` on `GPUSinkhornDivergence`: use `transportRadius`.
- `WindowMass` density helpers: use the `WindowOverlap` helpers.
- Mantel summary `observed`: use `observedCorrelation`.
- Legacy point-pattern and variogram setting names: use `analyticAlternative`, `indexCellSizeHint` and `sectorOffsetRadians`. The boolean diagnostics-flag variogram overloads: use the explicit diagnostics helper.

## Documentation and tests

- Join guide: separates strict polygon membership from exclusive point assignment, and covers native Polygon and MultiPolygon adaptation, counts-only CPU assignment, loss-aware zonal inference, single-search left gathers and the gathered live prefix.
- Prepared-handle guidance states 1,072 fixed bytes and per-encoding replay, and separates node-construction cleanup from registration recovery and fresh-graph retry.
- Relate-pattern docs separate the 2,147,483,647-slot host limit from the device binding and upload limit.
- Geometry statistics docs explain finite but misleading float32 weighted summaries, signed-weight variance, and zero counts and weight sums for empty groups.
- Label-point docs now give the correct planar frame and position units. `triangleCount` docs include counted zero-area triangles at touching holes and exclude padding.
- Concave-hull guidance describes the shipped alpha and ratio choices; the geometry reference lists the six missing minimum-bounds and Delaunay operation forms.
- Delaunay scheduling guidance gives both site and scratch-triangle limits. The Voronoi guide separates solver exhaustion, triangle truncation and segment overflow.
- The overlay guide covers allocation-free planning, resident preflight and assembly loss despite spare capacity. Cleanup docs cover zero controls, float32 rejection and half-up snapping.
- The grid guide states the 8,388,608 center-specific axis limit and the host admission planner.
- Great-circle and spherical segmentize docs state the near-antipodal cutoff and NaN interior behaviour.
- The space-time K guide states that derived width, height, duration, area and volume must be finite and at least 2^-126.
- The emerging hot-spot guide separates incomplete-neighborhood category 17 from all-missing category 0.
- The HDBSCAN guide states the 4,194,304-row BVH limit and the separate 1,048,576-row neighbor-search limit.
- The k-means guide describes initialization, the centered default and the old arithmetic, selected-restart status and final-round convergence.
- Geographic DBSCAN and clustering guidance gives a reversible antimeridian rotation.
- The Dorling docs state the eight-storage-binding requirement of the indexed relaxation and name the public import entry.
- Accessibility and allocation guidance covers independent certificates, `demandValidity`, horizons, bounded decision certificates and row and node identities.
- The regionalization guide covers label compaction, label completeness, five-word Max-p evidence and evaluator limits.
- The Pettitt guide separates its large-sample approximation from exact conditional inference for short tied series. Space-time summary metadata separates doubled-minimum-tail probabilities (floor 0.10 at 19 draws) from scan upper-tail probabilities (floor 0.05).
- Guide examples in the geometry, join, points, interpolation and statistics families now use typed public imports and are snippet-checked. The Hawkes example writes its settings to the parameter buffer the graph uses.
