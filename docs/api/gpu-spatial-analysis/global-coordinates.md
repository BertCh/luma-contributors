# Global coordinates, spherical geometry and S2 analysis

Import from `@luma-contributors/analysis/gpu-spatial-analysis`.

These graph contributors cover bounded portions of CR01–CR05. Inputs and outputs remain GPU
views. They declare compute nodes and never submit, read back, download grids, or call a CPU GIS
engine. Use a separate contributor for each source batch; the kernels do not merge chunks or own
caller buffers. All output views must be disjoint from inputs and each other. Storage usage stays
within eight bindings per compute stage.

| Roadmap | Delivered API and scope | Remaining scope |
| --- | --- | --- |
| CR01 | Existing `GPUGeometryMeasures` now offers safeguarded inverse solving, adaptive edge integration and explicit invalid/exhaustion reporting on serial/cooperative paths. | f32 numerical envelopes are tested, not certified; singular or unresolved inverse roots and pole-winding polygon areas remain explicitly invalid. |
| CR02 | Prepared regular grids, priority/bounds/nodata, inverse residual validation and optional high/low coordinate words match direct PROJ fixtures. | Grid files, acquisition, CRS databases and epoch/subgrid selection remain caller preparation, as scoped by the original backlog. |
| CR03 | Signed vertical/combined corrections, explicit units/reference names and split height residuals; direct PROJ vertical fixture. | Named geoid/epoch acquisition and full PROJ pipeline construction are external preparation. |
| CR04 | Nonconvex containment with holes, chart and intrinsic global Boolean overlay, complement semantics, open/closed predicates, pole/antimeridian fixtures and assembled rings. | f32 tolerance bands replace exact adaptive predicates; intrinsic overlay reports unresolved separation/junctions and bounds work to 128 input vertices / 1024 output vertices. |
| CR05 | Exact hierarchy/neighbors and stable measures through level 30, adaptive polygon covers through level 15, uniform cap covers and geometric cross-family overlap weights for prepared convex boundaries. | Cover guarantees are tolerance-aware; automatic cross-family broad-phase/boundary acquisition and optimized CPU timing comparisons remain separate integration/performance work. |

## Prepared datum grids

Import `GPUDatumGridTransform`, `decodeGPUDatumGridTransformPositions` and
`decodeGPUDatumGridTransformStatus` from `@luma-contributors/analysis/geospatial-analysis`.

`GPUDatumGridTransform` takes `float32x3` positions: longitude and latitude in degrees, then
height in the declared unit. Grid samples are `(longitude shift degrees, latitude shift degrees,
height shift meters)`. Signed height samples always mean **target minus source** height. A geoid
undulation grid must therefore be sign-adjusted by its preparation adapter when converting
ellipsoidal to orthometric height. Both vertical reference names are mandatory for vertical or
combined operations. The names describe provenance; the contributor does not resolve a CRS.

Each grid supplies dimensions, sample offset, origin and nonzero spacing; negative spacing is
allowed. Coordinates use a continuous longitude branch and there is no implicit antimeridian wrap.
Grid bounds are inclusive with a `1e-5` grid-coordinate numerical margin, clamped to the outer edge. The last grid row/column uses the final interpolation cell. All four
corners must be valid and finite, even when a corner has interpolation weight zero. The first
covering grid wins. Nodata in it fails instead of silently falling through to a lower-priority grid.
An empty grid list returns no-coverage for every finite input.

Forward horizontal transforms add interpolated shifts. Inverse horizontal transforms solve
`target = source + shift(source)` with at most `maximumIterations` fixed-point steps and verify the
final residual. The inverse always takes at least one step before testing the residual (the first residual is the shift itself). Default tolerance is `5e-9` degrees (about 0.56 mm; never less than four f32 spacings of the shift) and the iteration cap is 16. A combined inverse
samples vertical correction at the recovered source longitude/latitude. Inverse lookup starts at
the target: if that location is outside every grid, it returns no coverage, even if a shifted source
might exist. This matches the target-seeded direct PROJ `hgridshift` fixtures, including the case where a forward-shifted edge point cannot be inversely looked up. Priority discontinuities can also choose a different inverse grid; inverse is not promised to recover the original source across such a discontinuity. No extrapolation or identity fallback occurs.

Status is 0 success, 1 invalid/nonfinite result or latitude outside ±90°, 2 no coverage, 4 nodata,
and 8 inverse exhaustion. Failed rows contain NaNs. Optional paired `positionsLow` input/output
views represent longitude, latitude and height as the sum of high and low f32 words. The returned
high words stay equal to the input high words; the returned low words carry displacement, and are
not a normalized double-double representation. Grid origins are split before local interpolation.
This preserves tested sub-ulp datum shifts without claiming general float64 arithmetic. With no
low views, small shifts at large coordinates can still round away. Samples remain f32, and inverse
tolerance is a residual stopping test, not a certified geodetic accuracy bound. See the [PROJ horizontal grid semantics](https://proj.org/en/stable/operations/transformations/hgridshift.html)
and [vertical grid semantics](https://proj.org/en/stable/operations/transformations/vgridshift.html)
for the broader reference contracts; this prepared-grid interface does not implement every option.

For split readbacks, reconstruct in binary64:

```ts
import {decodeGPUDatumGridTransformPositions} from '@luma-contributors/analysis/geospatial-analysis';
export function reconstructDatum(high: Float32Array, low: Float32Array, status: Uint32Array) {
  const checked = decodeGPUDatumGridTransformPositions({positions: high, positionsLow: low, status});
  const reconstructed = Float64Array.from(high, (value, i) => value + low[i]);
  return {checked, reconstructed};
}
```

The decoder checks byte windows, XYZ alignment, matching lengths and status row count; use its
status alongside the reconstructed coordinates. Failed rows remain NaN.

## Convex spherical geometry and overlap weights

Provide normalized xyz directions and counterclockwise minor great-circle edges. Every polygon
must contain 3–32 distinct vertices, exclude a repeated closing vertex, and lie strictly inside an
open hemisphere. The vertex-mean direction must lie strictly inside every oriented edge halfspace.
Degenerate, antipodal-edge, clockwise or unsupported nonconvex inputs report invalidity. Input xyz
squared lengths must differ from one by no more than `1e-4`. Default predicate tolerance is `1e-6`.
The tolerance is an f32 angular band, not exact arithmetic.

Containment uses a query-to-polygon index column. `boundary: 'closed'` includes the tolerance band;
`'open'` excludes it. `complement: true` reverses interior/exterior while retaining the chosen
boundary policy. Status 2 exposes boundary classification, and status 1 invalidity takes precedence
and must not be interpreted as a meaningful false predicate.

Intersection uses paired polygon-offset columns. It collects contained vertices and true
minor-arc crossings, deduplicates within four times tolerance, and orders the resulting boundary.
Closed-boundary contacts may have one or two vertices. Outputs have a caller-selected stride of
1–64 and `(written, required)` counts. Status 1 means invalid input, 2 capacity exhaustion, and 4
numerical uncertainty; status bits may combine. Uncertain/invalid rows write zero vertices.
Capacity-limited output is incomplete and must not be consumed as a polygon.

Optional `output.measures` supplies `(intersection steradians, source fraction, target fraction,
union steradians)`. These fractions are geometric overlap weights for **already prepared** convex
source/target cell boundaries, irrespective of cell family. They are not a relabeling of H3/S2 keys
or centroid-containment approximation. Failed or capacity-limited rows have NaN measures. No
cross-family candidate generation, H3 boundary acquisition or global conservative-remapping
recipe is implied.

## S2 identity, measurement and covers

Cell keys are always exact little-endian `(low, high)` u32 pairs. `GPUS2CellHierarchy` supports
parents at a requested level, immediate children in Hilbert order, inclusive leaf-ID ranges, and
four edge neighbors ordered left/right/down/up in the input face coordinate system. Integer cube
face transitions work through level 30, including corners, poles and mixed-level rows. Invalid
keys report status 1; asking a leaf for children or a coarser cell for a descendant parent reports
status 2. All corresponding output keys are zero.

`GPUS2CellMeasures` returns area in steradians and perimeter in radians. Multiply by a chosen
sphere's squared radius or radius explicitly for physical units. Levels 0–30 use analytic quadratic-UV increments to compute triangle determinants and edge
cross products without subtracting rounded fine-cell vertices. The exact integer key is decoded
before this local computation. This improves area/perimeter precision; it does not improve the
absolute xyz precision of arbitrary boundary vertices or polygon-cover predicates.

`GPUS2CapCovering` accepts unit xyz cap centers and angular radii in `[0, pi]`. It enumerates all
`6 * 4^level` cells, with level restricted to 0–8, in ascending key order. Exterior covers select
cells whose closest spherical point is within the cap plus a default `1e-5`-radian safety margin.
Interior covers require every point to be within the cap minus that margin, using antipodal closest
distance for the farthest point. Consequently an interior cover of a radius-pi cap conservatively
excludes cells containing the antipodal center. Capacity truncation reports status 2 and the full
required count; such output is not a complete cover. Invalid caps report status 1. This uniform
algorithm is appropriate for modest cap batches; it is not S2's adaptive priority coverer.

`refinement: 'level-synchronous'` removes the level 8 ceiling. Instead of one invocation per cap
scanning the whole sphere, it starts from the six face cells and runs `level + 1` rounds. Each
round launches one invocation per frontier cell and per cap, drops cells that cannot contain a
selected descendant (a child is never closer to the cap than its parent), splits the others in
four, and a scan numbers the survivors, so every cap's cells stay in ascending key order with no
sort. The cells, counts and status equal the uniform enumeration's (exterior and interior, any
capacity) and `level` may reach 15. Each cap holds at most `frontierCapacity` cells per round
(default `8 * capacity + 64`, at least `capacity`): a round holds the four children of every kept
parent, and a kept parent can have no selected child, so the frontier can reach about four times
the final cell count. The default holds a cover that exceeds `capacity` by a factor of about two,
so the caller still gets status 2 and the required count. A cap that outgrows the frontier reports
status bit 4, writes no cells and reports counts `(0, 0)`: a truncated frontier cannot know the
size of the full cover, so no required count is available; raise `frontierCapacity` (or
`capacity`) and run again. At level 0, a zero-radius exterior cap on a face center covers
exactly that face. With the default tolerance, at levels 1 and above it covers the four cells that touch the center in both
enumerations. The rounds are not gated on the live frontier length (the published
`GPUScan` has no GPU-written dispatch size), and `estimatedCommandNodeCount` is exact for both the
cap and the polygon level-synchronous coverings. The default remains `refinement: 'uniform'`, the reference tier.
The CPU cap twin applies this frontier bound only when `frontierCapacity` is explicit. Omission
retains its complete adaptive traversal, including level-synchronous requests. An explicit bound
clears cells/counts with status bit 4 on exhaustion. An interior cap whose packed radius is less
than or equal to the float32-encoded tolerance is empty on both forms and does not expand a
frontier (including radius and tolerance both supplied as `0.000011`). Complete CPU traversal can count
contained subtrees without storing all descendants.



## Safeguarded ellipsoidal measurements

Set `ellipsoidalInverse: 'safeguarded'` on the existing `GPUGeometryMeasures`. Ordinary converged
Vincenty rows keep their fast path. Failed rows scan 32 auxiliary-longitude intervals near the
geographic separation and bisect sign-changing residual brackets for up to 32 iterations. A root
must have residual at most `7e-7` radians; the shortest accepted positive-distance solution wins.
A singular/discontinuous or unbracketed root remains invalid. This is an independently implemented
safeguard, not the full Karney inverse algorithm. The default remains `'vincenty'`.

`geodesicAreaTolerance` requests per-edge successive-refinement quadrature error estimation in
squared output units. `geodesicAreaMaximumSegments` is a power of two from 2 through 256 (default
256). Sampling doubles until `abs(fine-coarse)/15` meets the requested estimate; exhaustion or
nonfinite results invalidates the feature and increments `invalidCount`. This estimates quadrature
convergence, not all f32, inverse/direct-series or input-quantization error. Default omitted tolerance
retains the prior bounded-sampling behavior. Pole-winding polygon areas still report invalidity;
the spherical operators below handle spherical polar regions explicitly.

Direct `geographiclib==2.1` fixtures cover 128 near-antipodal and 128 global line pairs (exact f32
input coordinates), on serial and cooperative paths. Valid results must be within 10 meters; fewer
than eight of the 256 may be explicitly unresolved. Twelve holed polygons near high latitudes,
the equator and antimeridian compare adaptive area with `Geodesic.WGS84.Polygon`: absolute error
below `max(2e6 square meters, 3e-6 * reference area)`. An intentionally impossible `1e-20` tolerance
with a two-segment cap must invalidate all twelve. These are measured acceptance envelopes, not
user-requested tolerance guarantees.

## Spatial context, limits and status codes

Host direction props (`GPUSphericalPolygonContains.hemisphereCenter`,
`GPUS2PolygonCovering.hemisphereCenter`, `GPUSphericalPolygonOverlay.chartCenter`, and
`GPUSphericalArcOverlay.referencePoint`, including the corresponding CPU helpers) use
`validateUnitVector`, which requires
finite length and **abs(length(v) - 1) < 1e-6**, a strict length bound. Buffer vertices and
queries in convex contains/intersection, polygon contains, hemisphere polygon overlay,
intrinsic arc overlay and S2 polygon covering use **abs(dot(v,v) - 1) <= 1e-4**, an inclusive
squared-length bound evaluated in f32 on the GPU. The CPU arc-overlay and S2-covering twins
use that squared-length bound in binary64. The exported constants
`GPU_SPHERICAL_UNIT_VECTOR_LENGTH_TOLERANCE` and
`GPU_SPHERICAL_UNIT_VECTOR_SQUARED_LENGTH_TOLERANCE` name those different measures.

`GPUSphericalConvexIntersection.capacity` is 1–64 output vertices **per paired polygon row**,
with that fixed row stride; each input polygon has 3–32 vertices. Intrinsic arc overlay accepts
at most 128 **total input vertices across both operands**; output positions hold 3–1024
vertices **across all result rings on the GPU**. The CPU operation has no 1024-vertex ceiling
(for example capacity 1536 is accepted). Its operation's `capacity` is that total output vertex
capacity, while `ringCapacity` counts result rings and reserves `ringCapacity + 1` offsets.
S2 polygon covering's `capacity` counts output S2 cells across all levels for either traversal.

Use `decodeGPUSphericalConvexContainsStatus` or `decodeGPUSphericalPolygonContainsStatus`
for `{valid, invalidInput, boundary}` (bits 1 invalid, 2 boundary). Use
`decodeGPUSphericalConvexIntersectionStatus` for
`{complete, invalidInput, outputVertexCapacityExceeded, numericallyAmbiguous}`, or
`decodeGPUSphericalArcOverlayStatus` for
`{complete, invalidInput, outputCapacityExceeded, numericallyUnresolved}` (bits 1/2/4).
Bit 4 can mean a short split piece, insufficient side separation, ambiguous membership,
or a ring junction/assembly failure; it cannot identify the cause.
Existing codes and ports are unchanged. `decodeGPUSphericalConvexIntersectionCounts`
names the two words `{writtenVertexCount, requiredVertexCount}`;
`decodeGPUSphericalArcOverlayCounts` names the four words
`{writtenVertexCount, writtenRingCount, requiredSegmentCount, referenceInside}`.
Hemisphere polygon overlay retains the named status views of its planar arrangement.


The five spherical classes and the two S2 coverings read unit direction vectors. Each takes an
optional `spatialContext`, resolved with `resolveGPUSpatialContext` and exposed as
`.spatialContext`. The accepted space is `longitude-latitude`, the space whose positions these
vectors embed on the sphere, with great-circle edges and the `none` or `great-circle` metric.
Omitted, it defaults to `GPU_SPHERICAL_DEFAULT_SPATIAL_CONTEXT`. A planar context throws
`<id> does not support planar coordinates`, and a longitude-latitude context with linear edges
throws `<id> needs great-circle edges`. The context declares meaning only: the vectors are never
converted, and `GPULonLatPolygonPreparation` remains the way from degrees to unit vectors.

Validation messages name the contributor, the prop and the accepted range, for example
`spherical-arc-overlay left plus right positions must be at most 128`. The limits are named
exports with their reasons:

| Constant | Value | Reason |
| --- | --- | --- |
| `GPU_SPHERICAL_ARC_OVERLAY_MAXIMUM_INPUT_VERTICES` | 128 | serialized pairwise noding and ring assembly |
| `GPU_SPHERICAL_ARC_OVERLAY_MAXIMUM_OUTPUT_VERTICES` | 1024 | bounded assembled output; segment scratch depends on input sizes |
| `GPU_SPHERICAL_CONVEX_MINIMUM_VERTICES`, `GPU_SPHERICAL_CONVEX_MAXIMUM_VERTICES` | 3, 32 | convex polygons live in fixed private arrays |
| `GPU_SPHERICAL_CONVEX_INTERSECTION_MAXIMUM_CAPACITY` | 64 | a clip has at most the sum of two 32-vertex polygons |
| `GPU_SPHERICAL_MINIMUM_TOLERANCE`, `GPU_SPHERICAL_MAXIMUM_TOLERANCE` | 1e-7, 1e-2 | below f32 noise, and the meaning of a boundary band |
| `GPU_SPHERICAL_MAXIMUM_ARRANGEMENT_TOLERANCE` | 1e-3 | arc overlay and S2 polygon covering |
| `GPU_SPHERICAL_OVERLAY_MINIMUM_COSINE` | 1e-3 | gnomonic chart scale near the horizon |
| `GPU_S2_POLYGON_COVERING_MAXIMUM_LEVEL` | 15 | f32 vertex resolution |
| `GPU_S2_POLYGON_COVERING_MAXIMUM_VISITED_CELLS` | 1,048,576 | frontier buffers fit one 128 MiB binding |
| `GPU_S2_CAP_COVERING_MAXIMUM_UNIFORM_LEVEL` | 8 | `6 * 4^level` cells scanned per cap invocation |
| `GPU_S2_CAP_COVERING_MAXIMUM_REFINED_LEVEL` | 15 | f32 vertex resolution |
| `GPU_CELL_TOPOLOGY_MAXIMUM_RADIUS` | 8 | `GPUCellTopology` disk and ring `k` (H3 disk of 217 cells, Quadbin 289) |

Status words are documented on each output type and exported as constants, with the values
unchanged: `GPU_SPHERICAL_CONVEX_CONTAINS_STATUS_*`, `GPU_SPHERICAL_POLYGON_CONTAINS_STATUS_*`,
`GPU_SPHERICAL_CONVEX_INTERSECTION_STATUS_*`, `GPU_SPHERICAL_ARC_OVERLAY_STATUS_*`,
`GPU_SPHERICAL_POLYGON_OVERLAY_COMPLEMENT_*`, `GPU_S2_POLYGON_COVERING_STATUS_*`,
`GPU_S2_CAP_COVERING_STATUS_*`, `GPU_S2_CELL_HIERARCHY_STATUS_*` and `GPU_S2_CELL_MEASURES_STATUS_*`.

## General spherical topology

`GPUSphericalPolygonContains` accepts nonconvex simple shell/hole rings in either orientation and a query-to-polygon
index. The first ring is the shell and subsequent rings are holes. Rings must be disjoint, holes
properly nested, and minor arcs nondegenerate. Boundary policy and status 0/1/2 follow the convex
predicate. Each ring must lie in the open hemisphere defined by `hemisphereCenter`, or its
normalized vertex mean when omitted. Supplying a pole avoids sensitivity to uneven vertex
sampling. Each ring denotes its hemisphere-interior region regardless of orientation;
`complement: true` explicitly selects the other polygon region. Reversing a shell or hole does
not select its antipodal image. The separate `GPUSphericalConvexContains` contract still requires
counterclockwise convex rings. The general operator checks ring self-crossings; cross-ring nesting is a prepared-input contract.
Direct S2 checks include 1,032 normal/complement classifications on concave and holed rings rotated
across the pole and antimeridian.
The general-polygon boundary band measures distance to the finite minor arc, including angular
caps around its endpoints. A short edge's antipode is outside that band. For a triangle with an
equatorial edge spanning longitudes `[-0.00075, 0.00075]` radians and a north-pole vertex, tolerance
`0.001` classifies `(-1,0,0)` as outside with status 0 and `(1,0,0)` as closed boundary with status 2.
Accepted near-unit coordinates are normalized for these distance tests: radial error does not
consume angular tolerance. For the positive octant, `(1.00002,-3e-7,-4e-7)` and its normalized
copy both have status 2; closed boundaries contain them and open boundaries exclude them.
Endpoint caps are accepted before plane-distance tests: an exact packed vertex remains boundary
even when float32 arithmetic rounds its incident plane residual just above tolerance.
The Kepler spherical-containment CPU binding uses the same finite-arc and endpoint-cap geometry.

<a id="hemisphere-polygon-overlay-contract"></a>

`GPUSphericalPolygonOverlay` maps caller-prepared polygon/multipolygon boundaries into a common
gnomonic hemisphere, runs the existing GPU snap-rounded arrangement, then lifts assembled rings
onto the sphere. All four Boolean operations and left/right complements are supported. Every
vertex must meet `dot(chartCenter, vertex) >= minimumCosine` (default `.01`, minimum `.001`).
`vertexTolerance` is in chart coordinates. Complement output is a GPU word: 1 means the sphere
minus the assembled region. Boundary provenance endpoints remain in chart coordinates. Counts,
required counts, overflow and candidate overflow retain the planar arrangement contract. Projection
or capacity failures cannot be interpreted as an empty successful region. The direct S2 oracle
checks nonconvex/holed Boolean ring counts and areas to `3e-5` steradians.

Boundary `operandIds` and representative `featureSources.operandIds` always use the caller's left
(0) and right (1), including complemented operations that swap operands internally. Representative
selection minimizes that caller `(operand, source row)` pair and marks mixed ownership explicitly.
`boundary.sourceFeatureIds` uses each operand's `sourceIds` when supplied, falling back to its
feature row when omitted; `featureSources.sourceRows` always remains a row index. Caller IDs are
published after row-based representative reduction. Every supplied `sourceIds` view must hold at
least one identity per feature. The complement scalar must not overlap any feature-source view,
geometry, boundary or status output. All actual spherical output byte ranges, including lifted XYZ
positions, must be disjoint. Separate writers may use nonoverlapping packed columns (for example,
feature-source row at byte 0 and complement at byte 256). Read-only and writable views bound in
the same dispatch must use separate physical buffers; the identity publisher stages its output
so packed boundary operand and identity columns are supported.
In particular, result `status.count`, `status.overflow` and `status.candidateOverflow` must use
physical buffers separate from `geometry.featureOffsets`, `boundary.status.overflow` and
`boundary.status.candidateOverflow`: the result publisher reads the latter while writing the
former. Disjoint slices at bytes 0 and 256 still conflict and are rejected at construction.
The assembly-completeness dispatch also requires result `status.candidateOverflow` to use
a physical buffer separate from `boundary.status.count`. Within each co-written status group
(result count/overflow/candidateOverflow, all four boundary status words, optional
ringCount/requiredRingCount), aligned binding ranges must be disjoint; adjacent uint32 words
can have overlapping aligned bindings and are rejected. Separate-dispatch complement/source-row
packing remains supported even when their aligned binding ranges overlap.

The chart arrangement uses fixed passes rather than a solver iteration budget. Result `status`
contains `count`, `overflow`, `candidateOverflow`, and optional `ringCount`, `requiredRingCount`,
`nodedSegmentRequiredCount` and `uncertainCount`. Boundary `status` contains exactly `count`,
`requiredCount`, `overflow` and `candidateOverflow`. The result's `uncertainCount` counts segment
pairs whose predicates could not be certified; it is not a boundary-level count of selected edges.
Accept geometry and provenance only when both result overflow flags are zero. The polygon output
uses implicitly closed rings, without a repeated closing vertex.
Written counts are capacity-clamped; required counts describe the unclamped requirement. For
example, a three-edge boundary with capacity two reports count 2, requiredCount 3 and overflow 1;
two disjoint rings with ring capacity one report ringCount 1, requiredRingCount 2 and overflow 1.

Neither status port exposes `iterationCount`, `converged`, `iterationLimitReached`, `invalidCount`
or `approximation`; result status also excludes `requiredCount` (use `requiredRingCount` for ring
sizing), while boundary status retains `requiredCount` and excludes `uncertainCount`. These inherited
fields had no writer and are rejected at construction, including when supplied through a cast or
object spread. Use `overflow` and `candidateOverflow` for completeness and result `uncertainCount`
for uncertified predicates. Exactness is defined on the documented chart snap lattice, controlled
by `vertexTolerance`; no solver-round or separate approximation flag is implied. Input rejection
sets incomplete noding through `candidateOverflow` rather than publishing a separate rejected-row
count. The constructor
snapshots declared descriptor fields, so unrelated getters and metadata labels are ignored, while
retired status names are refused even when inherited. Spatial-context metadata is also captured;
both public context properties retain the validated coordinate space after caller mutation.
Subsequent descriptor changes do not change the graph; buffer contents remain per-frame inputs.
The website example suppresses both result layers and reports an incomplete result when either
failure flag is set, preserving the operand outlines for context.

<a id="intrinsic-arc-overlay-contract"></a>

`GPUSphericalArcOverlay` handles boundaries that cannot share a hemisphere. It nodes great-circle
arcs intrinsically, classifies adjacent faces by crossing parity, retains oriented boundary pieces,
and assembles rings using the left-turn rule at junctions. Both operands supply the known membership
of one common `referencePoint`, separated from all boundaries; reversing that membership represents
the complement. Nested rings use parity, and an empty boundary with inside reference represents
the full sphere. The reference membership is caller-prepared semantic data, not inferred from ring
orientation. Simple disjoint boundary rings are checked for self/cross-ring intersections; touching
rings within one operand are rejected. Output rings are open xyz rings with interior on the left.

Intrinsic overlay is a bounded serial reference route accepting at most 128 total input vertices
and 3–1024 output vertex capacity. Prefer `CPUSphericalArcOverlayOperation` when GPU graph
composition is unnecessary; its CPU implementation does not have these GPU size caps.
The public operator descriptor reports tier `reference`, quadratic-source transient storage and
multi-pass dispatch: parallel offset packing, one arrangement invocation and parallel offset padding.
Operand/output view identities and compile-time configuration are captured at construction in
the contributor and both operation forms; their buffer contents remain per-frame inputs.
Tolerance ranges from `1e-7` to `1e-3` radians, default `1e-6`. Seven deterministic two-arc parity
paths avoid reference rays through vertices. Adjacent-side probes start at 16 times tolerance and
shrink according to boundary separation; inadequate separation and unresolved junctions fail
explicitly. Counts are `(written vertices, written rings, required segments, reference membership)`.
Status bits are 1 invalid input/reference/topology, 2 capacity and 4 numerical uncertainty. Failed
results have zero written geometry; required segments are only a lower bound on early failure.
Segment storage is an explicit graph transient, not a large shader-private array. For L left and R
right vertices it retains at most `max(1, L + R + 4*L*R)` seven-float records, independent of output
capacity: each edge gains at most two split points per opposing edge. This lets duplicate detection
count every retained boundary piece once even after the output fills. The union of two identical
four-edge squares reports required segments 4 at both capacities 3 and 4; capacity 3 gives counts
`[0,0,4,0]` and status 2, while capacity 4 gives `[4,1,4,0]` and status 0. Twelve direct
S2 Boolean cases cover globally distributed zigzag boundaries and complements, with area error
below `2e-5` steradians and exact ring counts. This is tolerance-based topology, not exact S2 parity.
An input offset table claiming more than `floor(vertexCount/3)` rings is invalid before its ranges
are scanned, because every ring requires at least three vertices. Oversized output offset destinations
remain accepted: a parallel pass fills every spare slot with the written vertex count. A four-edge
square in nine offset slots produces `[0,4,4,4,4,4,4,4,4]`; failure clears all offset slots to zero.
The padding dispatch reads `output.counts` while writing `output.ringOffsets`; those two views
must use separate physical buffers, even for nonoverlapping slices. Construction rejects this
mixed usage before graph mutation. Other output columns may retain safe disjoint packing.

## Adaptive polygon covers

`GPUS2PolygonCovering` accepts simple shell/hole rings in either orientation. A parallel
preparation pipeline orders each ring counterclockwise about its hemisphere pole, then the selected
traversal classifies the hemisphere-interior region; `complement` selects the other region.
Clockwise GeoJSON holes therefore subtract their interior, and reversing a shell never selects
its antipodal image. Per-vertex preparation and segmented center reductions establish each ring's
pole; per-vertex projected differences and segmented area reductions establish its orientation.
Each reduction has `ceil(log2(max(N,1)))` stages over `N` position rows. Per-ring orientation and
parallel vertex copying follow. Zero or nonfinite projected area is invalid before any cell visit.
The polygon containment predicate also uses local differences for ring
orientation, tangent-angle winding and arc normals to retain small off-axis geometry.


`GPUS2PolygonCovering` classifies the six S2 roots and their candidate descendants in parallel. It emits whole
contained cells early and subdivides boundary cells through `maximumLevel` (0–15), subject to
`minimumLevel`. Interior mode emits only contained cells; exterior mode includes intersecting
terminal cells. Inputs use the same prepared shell/hole hemisphere contract and optional pole as
the nonconvex predicate, with explicit complement. Self-crossing rings and nonadjacent touches
are rejected before traversal, producing invalid status, zero counts and zero output keys.
Cross-ring nesting remains a prepared-input contract. This is mixed-level geometric coverage, not
H3 centroid containment or S2's priority-queue `max_cells` minimization policy.

`maximumVisitedCells` defaults to 65,536 and may range from 6 to 1,048,576. Status bits are 1 invalid
input, 2 output capacity and 4 traversal exhaustion. Counts are `(written, required so far)`;
when exhausted they do not claim the complete requirement. `visitedCount` exposes traversal work.
Capacity and traversal bits can coexist: the positive-xyz octant at level 1, six visits and
capacity 1 gives serial counts `(1,4)` and status `6`. Level-synchronous exhaustion discards
that unfinished frontier, giving counts `(0,0)` and status `4`.

Both refinement routes use the same parallel preparation and bounded predicates. Each work
invocation examines at most **4,096 input edges or edge pairs**, independently of ring size;
this is a per-invocation work tile, not a maximum ring size. Larger rings use integer winding
partials. Range searches take at most **32 bisections** per invocation. Vertex reductions process
one vertex per invocation per stage. These input-work bounds are separate from the visited-cell
budget and do not promise device timing or watchdog safety.

Both routes run `maximumLevel + 1` rounds, classifying bounded candidate lists (drop, keep or
split), then scanning and expanding children. Subtree sizes and ranks recover the exact ordered
depth-first prefix for default `refinement: 'serial'`; it retains cells emitted within exactly
`maximumVisitedCells` visits on exhaustion. The `level-synchronous` route instead suppresses
the whole cover when a round cannot fit, writes counts `(0,0)` and status bit 4, and caps
`visitedCount` at the budget. Its live frontier budget includes retained and pending cells.
Complete covers and visits agree between routes and keys are in ascending cell-ID order.

Let `N=max(positionCount,1)`, `R=ringCount`, `B=maximumVisitedCells`, `L=maximumLevel+1`, and
`H0=max(0,min(R,floor((positionCount-3*R)/4094)))`. Define `H=max(1,H0)`,
`T=max(1,H0+ceil(positionCount/4096))` and `C=min(B,max(1,floor(128*1024*1024/(16*H))))`.
Scratch, excluding input/output buffers and published `GPUScan` scratch, is approximately
`48*N + 32*R + 4*H + 16*T + 16*C + 16*C*H + 40*B*L` bytes, plus eight control words and
`maximumLevel+2` count words. Winding scratch `16*C*H` is at most **128 MiB per binding**.
The tree alone costs **40 bytes per budget cell per configured level**: about **40 MiB** at
the default budget and level 15, or **640 MiB** at the maximum budget and level 15.
Every configured level still dispatches/scans its bounded arrays.

Command-node cost includes segmented preparation, ring scans and bounded predicate batches;
it is no longer two orientation nodes plus a serial walk. For valid input dimensions, define
`S(n)` as the published scan's node count (one scan per 256-row reduction level and an offset
node for every level except the last), `Q=2^32-1`, and `P=ceil(N/max(1,floor(Q/ceil(N/4096))))`.
Preparation uses `10+2*ceil(log2(N))+2*S(R)+P` nodes. A predicate batch of `c` cells uses
`2+ceil(c/max(1,floor(Q/R)))+ceil(c/max(1,floor(Q/T)))+ceil(c/max(1,floor(Q/H)))` nodes.
Sum that over batches of at most `C` cells covering `B` slots to get `D`. The total is
`preparation + L*(D+S(B)+3+isLevelSynchronous) + 1`; `estimatedCommandNodeCount` exposes it.
The existing shared command-node and device storage limits still apply.

Both contributors and named operations capture declared options, context metadata, pole tuples
and nested output selections at construction, including getter-backed fields. Resource contents
remain live. Named GPU polygon input lengths must match captured `positionCount` and
`ringOffsetCount`. Mode values are `exterior` and `interior`; polygon refinement values are
`serial` and `level-synchronous`, cap refinements `uniform` and `level-synchronous`.
The CPU polygon twin honors an explicit `maximumVisitedCells` in both refinement modes and uses
these same publication rules: a serial prefix survives exhaustion, while level-synchronous
cells/counts are cleared. Omission retains complete CPU traversal; the GPU still defaults to
65,536 visits. A frontier that is already too large fails before classification, so its cells
do not increment `visitedCount`: six roots queuing 24 children into eight slots report six visits.
Both twins retain float64 geometric predicates, so boundary rounding can differ.
The direct `s2geometry==0.14.0` oracle enumerates all 6,144 level-5 cells for each of two concave/holed
regions: emitted interior cells are S2-contained, exterior covers include every S2 `MayIntersect` cell,
and summed interior/exterior cell areas bracket the region area. This is the precise tested
cover guarantee on these two exhaustive fixtures; it does not certify every possible thin or
near-degenerate boundary. Capacity and visit-budget failures are separately exercised.

The additional reproducible generators are `difficult-geodesics.py`, `geodesic-area-adaptive.py`,
`datum-horizontal.py`, `spherical-polygon-contains.py`, `spherical-polygon-overlay.py`,
`spherical-arc-overlay.py` and `s2-polygon-covering.py` under `modules/analysis/test/oracles/`.
They pin GeographicLib 2.1, pyproj 3.7.2 / PROJ 9.5.1 and S2Geometry 0.14.0. Horizontal PROJ
fixtures write prioritized CTable2 grids and decode their actual f32-radian samples; split outputs
must agree within `1e-9` degrees, including priority discontinuities and inclusive bounds.

## Evidence and limits

`modules/analysis/test/oracles/global-coordinates.py` pins `s2sphere==0.2.5` and
`geographiclib==2.1`, `pyproj==3.7.2` and `shapely==2.1.2`. The committed JSON records direct upstream calls and a clearly labeled composed overlap oracle. PROJ samples a generated local GTX vertical grid. For spherical crossings, Shapely computes topology in a gnomonic plane (great circles become lines) and GeographicLib independently measures the lifted intersection on a unit sphere. S2 neighbor checks
include every cell at levels 0–2 and sampled levels 15/30, including face seams and poles. The tests
compare measured area/perimeter with S2 values and check six-face area sums against `4*pi`.
Additional independent tests cover affine-grid interpolation/inversion, first-grid priority, signed
height units, nodata, nonfinite arithmetic, inverse exhaustion, known octant area, overlap fractions,
open/closed/complement boundaries, capacity limits and unsupported near-antipodal geodesics.
Node tests parse generated WGSL, audit reserved symbols and check binding counts. GPU tests are
in `modules/analysis/test/gpu-spatial-analysis/global-coordinates/global-coordinates.spec.ts`.

These fixtures establish the tested numerical cases, not complete S2/PROJ/GeographicLib parity.
The bounded observations below add same-device timing and explicit-buffer evidence. Cross-device accuracy, driver memory peaks and representative geographic distributions remain unmeasured.


## Numerical limits established by the committed tests

The tested limits below are acceptance thresholds, not certified worst-case error bounds or
promises for every supported input:

| Operation | Tested threshold and scope |
| --- | --- |
| S2 hierarchy/neighbors | Exact u32 words: all 126 cells at levels 0–2, plus 20 sampled cells each at levels 15 and 30 (166 oracle rows total). |
| S2 area/perimeter | Relative error below `2e-5` across all 166 direct S2 oracle cells, including 20 level-15 and 20 level-30 samples. |
| Ellipsoidal line distance | Absolute error below 5 meters on the three pinned regular, polar and antimeridian pairs. The pinned near-antipodal pair reports invalidity in serial/cooperative modes. This is not an antipodal accuracy envelope. |
| Spherical overlap | Eight-vertex crossing topology agrees with the gnomonic oracle; area and normalized overlap fractions use five-decimal-place assertions. |
| Datum grids | Affine bilinear forward values use five-decimal-place assertions; inverse coordinates use four and recovered height uses three. Nodata, out-of-coverage, exhausted iterations, nonfinite arithmetic and pole-crossing invalidity are checked separately. |

Spherical polygon predicate tolerance can be set from `1e-7` to `1e-2` (default `1e-6`), and
intersection deduplication uses four times this tolerance. Polygons whose center-to-edge separation
is at or below tolerance are rejected as unsupported/degenerate; this avoids pretending that f32
can distinguish arbitrarily thin regions. The cap-cover margin ranges from `1e-6` to `1e-2`
radians (default `1e-5`). It deliberately expands exterior covers and shrinks interior covers.

## Opt-in bounded performance observation

The benchmark spec is
`modules/analysis/test/gpu-spatial-analysis/global-coordinates/global-coordinates-benchmark.spec.ts`.
Run `yarn test-benchmarks modules/analysis/test/gpu-spatial-analysis/global-coordinates/global-coordinates-benchmark.spec.ts`.
The launcher selects benchmark files and enables their workload flags. Set optional
`VITE_BENCHMARK_REVISION` / `VITE_BENCHMARK_SOURCE_STATE` metadata to identify the tested tree.

Eight deterministic workloads compare small and larger batches: 166/16,384 S2 neighbor rows
cycling through the exact-key corpus, 256/16,384 datum positions, 1/32 level-4 spherical caps
(1,536 candidate cells per cap), and 1/256 four-vertex convex overlap pairs. These are repeated
reference shapes, not geographic distribution or scalability benchmarks. Each workload changes
resident inputs after the repeated measurements and validates the changed output.

JSON records separate buffer allocation/setup, upload, synchronous graph compilation, first
encode/submit/completion, five warm repeated runs, update upload and three updated runs. Readback
and reference checks are timed separately. GPU timestamps are reported only when available.
Memory reports explicit input/output bytes and graph transient allocation; driver peak memory
is unavailable. The device is shared and pipeline caches are not cleared, so first execution is
not guaranteed to be a cold-driver measurement. No throughput advantage or latency threshold is
asserted. Observed timings must identify the device/browser and source state; no performance
result is implied merely by adding this opt-in spec.


## Recorded bounded observations

An Apple WebGPU adapter (`metal-3`, non-fallback, chip model unspecified) running Headless Chrome151
completed all eight benchmark workloads on October 23, 2026 at 9:07 PM EDT.
The raw phase samples and metadata
retain upload, readback, validation, first/repeated/updated execution and allocation measurements.

**Provenance limitation:** the runner recorded `sourceRevision: null` and `sourceState: null`.
The source was a shared dirty working tree. These are local observations, not a frozen release
benchmark. Device caches were shared; compilation can be deferred into first execution, and no
cold-driver guarantee applies. Timestamp queries were supported but no GPU timestamp durations
were recorded, so every execution number below includes CPU encoding, submission and queue wait.

| Workload | Host compile ms | First execution ms | Warm median ms (5 runs) | Updated median ms (3 runs) | Explicit input + output bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `s2-neighbors-166` | 2.9 | 18.6 | 0.5 | 0.8 | 7,304 |
| `s2-neighbors-16384` | 0.2 | 5.2 | 0.8 | 1.4 | 720,896 |
| `datum-grid-256` | 0.3 | 3.5 | 0.5 | 0.6 | 7,232 |
| `datum-grid-16384` | 0.2 | 2.8 | 0.6 | 0.3 | 458,816 |
| `s2-cap-cover-1` | 0.6 | 102.5 | 9.1 | 7.5 | 12,316 |
| `s2-cap-cover-32` | 0.2 | 49.4 | 7.1 | 7.0 | 394,112 |
| `spherical-overlap-1` | 0.1 | 104.6 | 0.4 | 0.9 | 332 |
| `spherical-overlap-256` | 0.2 | 110.6 | 0.5 | 0.5 | 82,952 |

All initial, repeated and updated outputs passed their pinned/analytic reference checks. Each
case reported zero graph-owned transient allocation; this excludes shader-private temporary
storage and driver allocation. The input/output byte totals include output capacity, including
unused padded slots. Sub-millisecond differences and the cap batch ordering should not be read
as scaling trends from these single shared-device runs. There is no optimized CPU timing baseline.

The level-4 cap oracle exposed a real cancellation defect during benchmark development: subtractive
projection onto a nearly perpendicular great-circle plane could leave an f32 residue that
normalized to the query direction. The corrected double-cross projection is now guarded by a
regular GPU regression against the pinned exact cell list. The successful records above come from
the corrected implementation; the earlier failed run is not represented as successful evidence.

## Executable unit-sphere examples

These lazy examples execute actual GPU graphs on synthetic analytic inputs and capture the input views, result views and status flags. They are not benchmark runs or observed geographic datasets.

### `GPUS2CellHierarchy`

Exact children of the six S2 root faces.

{/* lens:GPUS2CellHierarchy */}

::lens{scenario=gpu-spatial-analysis.s2-cell-hierarchy rung=result caption="Inspect captured input, result and status rows."}

::lens{scenario=gpu-spatial-analysis.s2-cell-hierarchy rung=graph caption="Inspect the executed GPU command graph."}

### `GPUS2CellMeasures`

Unit-sphere area and perimeter of S2 cells at levels 0–30. This compatibility API retains
steradians/radians and its per-row status (0 valid, 1 invalid with NaN measures). It shares the
stable local-increment arithmetic of [`GPUCellMeasures`](./operators-cells.md#gpucellgridpath-and-gpucellmeasures).
Use `GPUCellMeasures` with `family: 's2'` for meters/kilometers, ordered edges and the common
H3/S2/Quadbin interface; invalid rows there are zero. Both APIs accept the compile-time
`precision: 'double-single'` option for published double-single projection and local-increment arithmetic;
the default is `'float32'`. Output formats and the unit-sphere API's invalid statuses stay the same.

{/* lens:GPUS2CellMeasures */}

::lens{scenario=gpu-spatial-analysis.s2-cell-measures rung=result caption="Inspect captured input, result and status rows."}

::lens{scenario=gpu-spatial-analysis.s2-cell-measures rung=graph caption="Inspect the executed GPU command graph."}

### `GPUS2CapCovering`

Exterior root-face covering of a small north-pole cap.

{/* lens:GPUS2CapCovering */}

::lens{scenario=gpu-spatial-analysis.s2-cap-covering rung=result caption="Inspect captured input, result and status rows."}

::lens{scenario=gpu-spatial-analysis.s2-cap-covering rung=graph caption="Inspect the executed GPU command graph."}

### `GPUSphericalConvexContains`

Interior, exterior and closed-boundary queries on an octant.

{/* lens:GPUSphericalConvexContains */}

::lens{scenario=gpu-spatial-analysis.spherical-convex-contains rung=result caption="Inspect captured input, result and status rows."}

::lens{scenario=gpu-spatial-analysis.spherical-convex-contains rung=graph caption="Inspect the executed GPU command graph."}

### `GPUSphericalConvexIntersection`

Identical octants with intersection area and overlap fractions.

{/* lens:GPUSphericalConvexIntersection */}

::lens{scenario=gpu-spatial-analysis.spherical-convex-intersection rung=result caption="Inspect captured input, result and status rows."}

::lens{scenario=gpu-spatial-analysis.spherical-convex-intersection rung=graph caption="Inspect the executed GPU command graph."}

### `GPUS2PolygonCovering`

[Inputs, outputs and limits](#adaptive-polygon-covers).

{/* lens:GPUS2PolygonCovering */}

::lens{scenario=gpu-spatial-analysis.s2-polygon-covering rung=result caption="Inspect captured geometry, counts and failure flags."}

::lens{scenario=gpu-spatial-analysis.s2-polygon-covering rung=graph caption="Inspect the executed GPU stages."}

### `GPUSphericalPolygonContains`

[Inputs, outputs and limits](#general-spherical-topology).

{/* lens:GPUSphericalPolygonContains */}

::lens{scenario=gpu-spatial-analysis.spherical-polygon-contains rung=result caption="Inspect captured geometry, counts and failure flags."}

::lens{scenario=gpu-spatial-analysis.spherical-polygon-contains rung=graph caption="Inspect the executed GPU stages."}

### `GPUSphericalPolygonOverlay`

[Inputs, outputs and limits](#hemisphere-polygon-overlay-contract).

{/* lens:GPUSphericalPolygonOverlay */}

::lens{scenario=gpu-spatial-analysis.spherical-polygon-overlay rung=result caption="Inspect captured geometry, counts and failure flags."}

::lens{scenario=gpu-spatial-analysis.spherical-polygon-overlay rung=graph caption="Inspect the executed GPU stages."}

### `GPUSphericalArcOverlay`

[Inputs, outputs and limits](#intrinsic-arc-overlay-contract).

{/* lens:GPUSphericalArcOverlay */}

::lens{scenario=gpu-spatial-analysis.spherical-arc-overlay rung=result caption="Inspect captured geometry, counts and failure flags."}

::lens{scenario=gpu-spatial-analysis.spherical-arc-overlay rung=graph caption="Inspect the executed GPU stages."}

## Expanded-mode scaling observations

The expanded harness completed fourteen workloads on the same Apple `metal-3` adapter and
Headless Chrome 151 at 10:42 PM EDT on October 23, 2026. The complete phase records
include the original eight cases and six added cases below. Source state is explicitly
`shared-dirty-tree`; revision is null, so this is not a frozen-source benchmark. These are CPU
encode/submit/queue-completion measurements, with upload, validation and readback reported
separately. Five warm executions and three updated executions share caches and device state.
No optimized CPU baseline, speedup claim, cross-device result or driver peak measurement exists.

| Workload | Host compile ms | First execution ms | Warm median ms | Updated median ms | Explicit input + output bytes | Graph transient bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `s2-stable-measures-166` | 0.3 | 36.7 | 0.2 | 0.2 | 3,320 | 0 |
| `s2-stable-measures-16384` | 0.1 | 29.4 | 0.4 | 1.1 | 327,680 | 0 |
| `s2-adaptive-cover-level4` | 0.2 | 178.1 | 73.3 | 69.0 | 81,980 | 0 |
| `s2-adaptive-cover-level6` | 0.4 | 372.7 | 309.1 | 302.9 | 81,980 | 0 |
| `spherical-intrinsic-overlay-8` | 0.5 | 232.9 | 3.2 | 3.3 | 3,464 | 7,184 |
| `spherical-intrinsic-overlay-32` | 0.2 | 241.7 | 22.2 | 22.1 | 3,752 | 7,184 |

Measure batches repeat the 166 direct-S2 oracle cells, including level 30, then reverse resident
key order. Adaptive covers rotate a spherical octant and include a downstream measure stage for
all 4,096 capacity slots; checks require complete counts, sub-enumeration traversal work and an
exterior area at least the octant area. Intrinsic overlays compare the same globally distributed
S2-reference intersection with 8 versus 32 input vertices, then rotate both operands. Additional
vertices sample the original minor arcs, so the region and oracle area remain unchanged.

These observations show the cost of the bounded traversal and intrinsic arrangement as resolution
or vertex count increases; they do not establish large-region scalability. In particular the
level-6 cover costs about 309 ms in this fixture and the 32-vertex arrangement about 22 ms. Output
capacity bytes include unused slots, and graph transient bytes exclude shader-private and driver
allocation. Every initial, repeated and updated result passed its numerical checks.

### Operation forms: S2 covers and intrinsic arc overlay

`GPUS2CapCoveringOperation` / `CPUS2CapCoveringOperation` consume logical float32x4 `caps`
(unit xyz and angular radius), `capCount` and per-cap `capacity`. They publish fixed-stride
uint32x2 `cells`, `(written, required)` `counts`, and one packed status word per cap (1 invalid,
2 output capacity, 4 frontier capacity in the level-synchronous GPU route).

`GPUS2PolygonCoveringOperation` / `CPUS2PolygonCoveringOperation` consume logical unit-xyz
`positions` and uint32 `ringOffsets`, with explicit `positionCount` and `ringOffsetCount`.
They publish cells, a single `(written, required)` pair, packed `status` (1 invalid, 2 output
capacity, 4 visited-cell budget), and
optional `visitedCount`. GPU traversal selection and visit budget remain construction settings;
CPU traversal is complete when its visit budget is omitted; an explicit
`maximumVisitedCells` applies in both CPU traversal modes too. CPU cap/polygon levels may
reach 30; GPU caps stop at 8 uniform/15 refined and GPU polygons at 15. Both forms accept either ring orientation after local orientation preparation.

`GPUSphericalArcOverlayOperation` / `CPUSphericalArcOverlayOperation` take logical positions
and ring-offset names for each operand, operand reference-membership booleans, the reference
point and Boolean operation. The GPU limit is 128 combined input vertices. `capacity` counts
total result vertices and `ringCapacity` counts result rings. Outputs are positions, offsets,
four packed counts (written vertices, written rings, required segments, reference membership),
and status bits 1 invalid / 2 capacity / 4 unresolved. The CPU arrangement is not limited by
the GPU's 128-input-vertex private storage. Compile GPU forms with
`GPU_CONTRIBUTOR_OPERATION_LOWERING`.


### Work bounds and reference routes

The intrinsic GPU arrangement's 128-vertex bound is combined across both operands, not per ring
or per operand. Every edge is compared to opposing edges for splits; each piece also performs
side classification and ring assembly. Segment scratch holds at most `max(1, L + R + 4*L*R)`
records for `L` left and `R` right vertices, as described above. Increasing output capacity does
not remove that work. CPU overlay uses the same noding/classification strategy in binary64
without those GPU vertex/split/output ceilings; it has no implicit execution-time budget.
Caller vertex and ring capacities still bound publication, and overflow suppresses geometry.

For GPU S2 cap uniform traversal the bound is `6 * 4^level` cell tests per cap, with level at
most 8. Refined caps run `level + 1` rounds, level at most 15, with a per-cap frontier bound.
For serial GPU polygon refinement, `maximumVisitedCells` (default 65,536, maximum 1,048,576)
bounds the published depth-first prefix. Parallel preparation may classify up to that many
candidates **per configured level**, through level 15; total candidate classifications can reach
`maximumVisitedCells * (maximumLevel + 1)`. Level-synchronous refinement additionally bounds
the whole live frontier and suppresses an exhausted round's result. The CPU cover routes accept key levels through 30 and perform
unbudgeted traversal when explicit frontier/visit budgets are omitted, subject to the uint32
publication ceiling below. Large complete
covers can take substantial CPU work even when the stored output capacity is small; required
counts still describe the complete traversal when it finishes.


CPU traversal without an explicit budget is still subject to the uint32 publication ceiling:
`requiredCount`/visited counts cannot exceed 4294967295 (`2^32 - 1`). Output capacity controls
stored cells, not this count range. A whole-sphere exterior cap at level 16 requires
`6 * 4^16 = 25769803776` cells and throws even with output capacity one, rather than publishing
a wrapped count. CPU support through level 30 is a key-layout limit, not a guarantee that every
complete cover count is publishable.

The GPU polygon visit budget bounds the published depth-first prefix on the serial route,
while candidate classification is bounded per level. It does not bound input-ring validation
or the edge tests within each cell classification: ring self/crossing checks and polygon-edge
work scale with the supplied boundary. Serial exhaustion retains its emitted prefix; refined
whole-round exhaustion clears geometry as documented above.

Count decoders accept typed uint32 words, ArrayBuffer and Uint8Array byte windows. They reject
partial words and reads shorter than the required two/four words; the byte offset and length are
honored. These checked layouts are distinct from per-row status bits:

```ts
import {decodeGPUSphericalConvexIntersectionCounts, decodeGPUSphericalArcOverlayCounts} from '@luma-contributors/analysis/gpu-spatial-analysis';
export function readSphericalCountExamples() {
  const convex = decodeGPUSphericalConvexIntersectionCounts(new Uint8Array(Uint32Array.of(3, 3).buffer));
  const arc = decodeGPUSphericalArcOverlayCounts(new Uint8Array(Uint32Array.of(4, 1, 4, 0).buffer));
  return {convex, arc}; // 3 written/required vertices; 4 vertices, 1 ring, 4 segments, outside reference
}
```
