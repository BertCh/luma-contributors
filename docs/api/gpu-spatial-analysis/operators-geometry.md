---
title: "GPU Spatial Analysis: Geometry utilities"
sidebar_label: "Geometry utilities"
description: "Geometry utilities contributors in @luma-contributors/analysis/gpu-spatial-analysis."
---


# Geometry utilities

Import from `@luma-contributors/analysis/gpu-spatial-analysis`.

Direct GPU classes are contributors: construct, then `graph.add`. Where a class accepts a
parameter buffer, update it with the matching `getGPU...ParameterValues` helper for per-frame values.
Operation forms below declare named graph inputs and outputs; see
[operation forms and CPU twins](../operation-forms-and-cpu-twins.md) for compilation and execution.

For equality after normalization, `GPUCanonicalGeometryEquality` and `GPUCanonicalPolygonEquality`
default to `toleranceMetric: 'per-coordinate'`. Select `'euclidean'` to match the distance metric of
[`GPUGeometryPredicates.equalsExact`](./operators-validity.md#gpugeometrypredicates).
Tolerance uses planar position units: displacement `(0.75,0.75)` passes a per-coordinate tolerance
of 1 but fails Euclidean tolerance 1. Canonicalization changes ordering policy independently of this metric.

<a id="hull-construction"></a>

## Build hulls from points

For planar point sets, use `GPUGroupConvexHull` for an enclosing convex ring, or `GPUConcaveHull`
and `GPUGroupConcaveHull` for an alpha-complex area hull: retain Delaunay triangles with
`maximumCircumradius` in planar position units. Triangle and directed boundary-edge rows retain
original source point indices; `triangleValidity` and `edgeValidity` mask their fixed reservations.
The area hull may have holes and disconnected components; isolated points and collinear sets
produce no area boundary. `status.converged` must pass before consuming it. The grouped packed
route has separate triangle/edge ranges; see [group outlines and output limits](../gpu-raster/expansion.md#group-outlines-and-recipes).

For ratio-based erosion instead, use [`GPURatioConcaveHull`](../gpu-geometry-refinement.md):
`ratio` is a fraction in `[0,1]`, not a radius. It removes exposed Delaunay triangles in parallel
rounds, so it can retain different triangles from GEOS's largest-first erosion. See that guide
for hole policy, bounded status and input restrictions. For an existing valid planar polygon,
[`GPUSubdividePolygons`](../gpu-geometry-refinement.md) produces bounded convex pieces rather than
a new point-set hull. The [geometry refinement examples](./geometry-refinement-lenses.md#gpuratioconcavehull)
include ratio erosion; historical C10 hull examples remain reachable at the
[raster-expansion URL](../gpu-raster/expansion.md#operations-and-contracts).
For a polygon interior mesh, go to [constrained Delaunay triangulation](#constrained-delaunay-triangulation).

### `GPUOutlineGeometry`

Compact output layouts are opt-in; offsets and counts describe only emitted geometry. Position
addressing is at most 2,147,483,647 and candidate/fragment word counts must fit uint32.

Builds round-joined triangle outlines around points, lines or polygon rings at a fixed distance, for drawing only.

::lens{scenario=gpu-spatial-analysis.outline-geometry rung=result caption="Grey: 51 Chicago parks · blue: 100 m outline triangles, overlaps not unioned"}

Buffer-like outline geometry for drawing (turf `buffer` and `lineOffset`, PostGIS `ST_Buffer` as a
picture). Props: `positions`, `geometryType` (`'points'`, `'lines'` or `'polygons'`; lines take `lineOffsets`, polygons the four-level
`featureOffsets`, `polygonOffsets` and `ringOffsets`), `spatialContext` (omitted is planar; `{coordinateSpace: 'longitude-latitude', metric:
'great-circle'}` gives meters on a sphere of `sphereRadius`, default the mean Earth radius; this replaces
the removed `coordinateSystem` and `radius` props, and other contexts throw), `joinSegments` (default 16),
`parameters` (`getGPUOutlineGeometryParameterValues({distance})`) and `output.positions`. The default fixed-stride layout writes a
non-indexed triangle list with exactly `positions.length * getGPUOutlineGeometryVerticesPerInput(joinSegments)`
vertices: per input vertex a round-join disc fan plus a quad to the next vertex, with unused triangles
degenerate. Fixed-stride output has no capacity or overflow. With `output.layout: 'compact'`,
`vertexOffsets` and status delimit the emitted triangle vertices; `sourceRows` is optional.
Compact capacity overflow retains a bounded prefix and reports required counts. Overlaps are not unioned, so it is for drawing only;
queries use `dwithin` and `GPUBufferSelection`. Spherical mode converts meters with each vertex's local
east and north scale (equirectangular, not geodesic; wrong near the poles and across the antimeridian).
It measures distance, so `local-lambert-equal-area` frames throw. There are no negative buffers. A one-sided offset line is [`GPUOffsetCurve`](#gpuoffsetcurve-and-gpuvertexsnap).

Pass optional `spatialContext` to `getGPUOutlineGeometryParameterValues(settings)` to warn on short `distance` values; see [Precision of longitude/latitude positions](./concepts.md#precision-of-longitudelatitude-positions).

#### How it computes

{/* lens:GPUOutlineGeometry */}

::lens{scenario=gpu-spatial-analysis.outline-geometry rung=datum datum=0 caption="Park vertex 0, the first corner of Winnemac Park"}

::lens{scenario=gpu-spatial-analysis.outline-geometry rung=formula caption="Each outline vertex is the input vertex plus distance along a join direction"}

::lens{scenario=gpu-spatial-analysis.outline-geometry rung=kernel caption="One thread per park vertex writes its disc fan and quad"}

::lens{scenario=gpu-spatial-analysis.outline-geometry rung=graph caption="One outline node: positions and distance in, triangle list out"}

{/* parameters: GPUOutlineGeometry generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUOutlineGeometry` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPUOutlineGeometry`.

- Packer: `getGPUOutlineGeometryParameterValues(settings)`, writes `GPU_OUTLINE_GEOMETRY_PARAMETER_LENGTH` (4) `float32` values.

Settings of `getGPUOutlineGeometryParameterValues`:

| Field | Type | Description |
|---|---|---|
| `spatialContext` (optional) | `GPUSpatialContext` | Spatial context of the positions; when it is longitude/latitude, a distance under 10 float32 quanta (about 17 m) logs a warning (see Precision of longitude/latitude positions). Does not change the packed values. |
| `distance` | `number` | Offset distance, in position units (planar context) or in the context's `units` on its sphere (longitude/latitude context, meters by default). Must be non-negative; `0` draws nothing visible. |

{/* /parameters */}

### `GPUOffsetCurve` and `GPUVertexSnap`

`GPUOffsetCurve`: Compact output is opt-in; vertex offsets/counts bound written vertices while
line/ring slots follow the input layout. Position addressing is at most 2,147,483,647 and candidate/fragment word counts must fit
uint32.

`GPUOffsetCurveOutput` is the default fixed-stride output: `positions` is `float32x2`, with
`inputVertexCount * getGPUOffsetCurveRowsPerVertex(...)` rows, and optional `counts` is `uint32`
with one row per input vertex. Each vertex's unused stride rows repeat its last point. This
type has no compact count or overflow status; allocate the full fixed layout.

`GPUOffsetCurveCompactOutput` (`layout: 'compact'`) has `float32x2` positions and `uint32`
`lineOffsets`. It preserves one slot per input line or polygon ring, including empty or fully
truncated lines: `count` and optional `requiredCount` equal the input line/ring count.
`lineOffsets` must have exactly that count plus one rows, including the terminal offset; a shorter
or longer view fails construction. Only vertex capacity can set this route's `overflow` word.
Optional `status.vertexCount` and `status.requiredVertexCount` give written and required vertex
counts. All these status words belong to the compact `GPUOffsetCurve` route only.
Use adjacent offsets `[lineOffsets[i],lineOffsets[i+1])` to read each line's written vertices.
For two two-vertex lines and two output vertex slots, offsets are `[0,2,2]`, both line-count
words are 2 and overflow is 1: the second line slot has no written geometry.
Unused position capacity is unspecified.
Optional `sourceLineIndices` names input line/ring rows and `sourceRows` names input vertex rows.
The compact type refuses `measures` and declares `status.approximation` as optional `never`,
matching its runtime refusal even for prepared output variables.

**`GPUOffsetCurve`** — Offsets lines or polygon rings to one side by a signed distance, with round, mitre or bevel joins.

::lens{scenario=gpu-spatial-analysis.offset-curve rung=result caption="Grey: 223 Wicker Park streets · blue: the same streets offset 20 m to the left"}

**`GPUVertexSnap`** — Moves each vertex onto the nearest reference vertex closer than a tolerance, keeping the layout unchanged.

::lens{scenario=gpu-spatial-analysis.vertex-snap rung=result caption="Grey: 51 parks · blue: park vertices snapped to community-area vertices within 50 m"}

`GPUOffsetCurve` is a one-sided offset of lines (`lineOffsets`) and polygon rings (`geometryType: 'polygons'` with the four-level
layout; the ring count is preserved, so the input `featureOffsets` and `polygonOffsets` describe the output rings)
(`shapely.offset_curve`, geo `OffsetCurve`);
a positive distance is to the left of the path direction. Per vertex it writes a round arc
(`quadSegments` per quarter circle, at most `GPU_OFFSET_CURVE_MAXIMUM_QUAD_SEGMENTS` = 256), a mitre (clipped beyond `mitreLimit`, default 5) or a bevel on the
outside of a turn, the single intersection point on the inside, and a plain normal offset at path ends.
Size fixed-stride output with `getGPUOffsetCurveRowsPerVertex(joinStyle, quadSegments, insideTurns)`
rows per input vertex. Round joins use `2q + 1` rows; bevel and mitre use 2 with the default
`insideTurns: 'intersection'`, or 3 for `insideTurns: 'buffer'` (offset end, original vertex,
offset start). The optional
`output.counts` gives the distinct points per vertex and padding repeats the last point. Distance and mitre
limit are per-frame. Which side of a vertex is the inside of its turn is an exact orientation sign, so
the join is on the right side where two edges nearly reverse (the tip of a sliver thinner than one float32
step). It is render-grade: GEOS loop removal is not done, so tight inside corners can
self-cross. GEOS clamps `quad_segs` to at least 8, so round-join parity is pinned at 8 and 16; GEOS
leaves the corner at a ring's first vertex open on the outside and this closes it. Points were within 2e-4
of Shapely 2.1.2. Planar only (`spatialContext` defaults to planar; a longitude/latitude context throws,
so project first with `createGPULocalFrame` and `GPULocalFrameProjection`), rings are implicit-closed.
Both `GPUOffsetCurve` and `GPUVertexSnap` measure distance and reject `local-lambert-equal-area` frames.

`GPUVertexSnap` is the vertex-to-vertex subset of `shapely.snap`: each vertex with a reference vertex
strictly closer than the per-frame `tolerance` moves to the nearest one (lowest row on ties) and the layout
is unchanged. Optional pairwise mode (`featureOffsets` and `referenceOffsets`) and `output.referenceRows`.
Its `GPUVertexSnapOutput` contains `positions: GraphDataView<'float32x2'>`, one row per input
vertex, and optional `referenceRows: GraphDataView<'uint32'>` in that same input-vertex domain.
The latter indexes `referencePositions`, or is `GPU_VERTEX_SNAP_NO_REFERENCE` (`0xffffffff`)
when no reference wins. It has no compact status, offsets, count or overflow output.
It is `O(V * R)`. Segment insertion and snapping onto segments, which GEOS also does, are out of scope.
It matched Shapely on 60 random lines.

#### How it computes

{/* lens:GPUOffsetCurve */}

::lens{scenario=gpu-spatial-analysis.offset-curve rung=datum datum=0 caption="Street vertex 0: how many distinct offset points it wrote"}

::lens{scenario=gpu-spatial-analysis.offset-curve rung=kernel caption="One thread per street vertex emits its corner rows"}

::lens{scenario=gpu-spatial-analysis.offset-curve rung=graph caption="One offset node: positions and distance in, corner rows out"}

{/* lens:GPUVertexSnap */}

::lens{scenario=gpu-spatial-analysis.vertex-snap rung=datum datum=19 caption="Park vertex 19 in Margate Park: the area vertex it snapped to"}

::lens{scenario=gpu-spatial-analysis.vertex-snap rung=formula caption="Nearest reference vertex closer than the tolerance wins, lowest row on ties"}

::lens{scenario=gpu-spatial-analysis.vertex-snap rung=kernel caption="One thread per park vertex searches the reference vertices"}

::lens{scenario=gpu-spatial-analysis.vertex-snap rung=graph caption="A BVH over the area vertices, then the snap search"}

{/* parameters: GPUOffsetCurve generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUOffsetCurve` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPUOffsetCurve`.

- Packer: `getGPUOffsetCurveParameterValues(settings)`, writes `GPU_OFFSET_CURVE_PARAMETER_LENGTH` (4) `float32` values.

Settings of `getGPUOffsetCurveParameterValues`:

| Field | Type | Description |
|---|---|---|
| `distance` | `number` | Signed offset in position units: positive is the left of the line direction, negative the right. |
| `mitreLimit` (optional) | `number` | Mitre length over `abs(distance)` beyond which a mitre corner is clipped. Default 5, at least 1. |

{/* /parameters */}

{/* parameters: GPUVertexSnap generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUVertexSnap` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPUVertexSnap`.

- Packer: `getGPUVertexSnapParameterValues(parameters)`, writes `GPU_VERTEX_SNAP_PARAMETER_LENGTH` (4) `float32` values.

Settings of `getGPUVertexSnapParameterValues`:

| Field | Type | Description |
|---|---|---|
| `tolerance` | `number` | Snap radius in position units. A vertex snaps when a reference vertex is strictly closer. |

{/* /parameters */}

### `GPUPolygonOverlay` and `GPUBufferSurface`

`GPUPolygonOverlay`: Optional `featureSources` publishes representative retained-boundary source row,
operand and mixed flag; it does not enumerate all input contributors. Tolerance is at most 2^127 and
positive shader values must survive float32 conversion.

`GPUBufferSurface`: Optional `featureSources` publishes representative retained-boundary source row,
operand and mixed flag; it does not enumerate all input contributors. Tolerance is at most 2^127 and
positive shader values must survive float32 conversion.

**`GPUPolygonOverlay`** — Computes intersection, union, difference, symmetric difference or dissolve of planar polygon sets as compact polygons.

::lens{scenario=gpu-spatial-analysis.polygon-overlay rung=result caption="Grey: 51 parks, light grey: 17 areas · blue: parks clipped to Uptown, area 3"}

**`GPUBufferSurface`** — Buffers lines or polygons by a signed distance into dissolved polygons, with round, mitre or bevel joins.

::lens{scenario=gpu-spatial-analysis.buffer-surface rung=result caption="Grey: 51 Chicago parks · blue: dissolved 50 m buffer polygons"}

Maximum (`GPUPolygonOverlay`): both operands together have at most `getGPUSegmentIntersectionMaximumVertexCount(device)` (4194304 on a default device: the 32-byte segment-table row per vertex must fit one storage binding) `positions` rows plus one per ring. Those rows plus `capacity.intersections` must also fit `floor(device.limits.maxStorageBufferBindingSize / 128)`: snap rounding keeps four diagonal vertices per hot pixel in the segment table. This is a device-derived limit of 1,048,576 on a 128 MiB device or 8,388,608 on a 1 GiB device, not a fixed row cap. The 23-bit lattice keys encode coordinates, independently of the row count.

`GPUBufferSurface` maximum: the boundary rows (raw offset-curve rows plus one closing vertex per raw ring, `(1 + 2 * rowsPerVertex) * vertices + 3 * rings` for polygons) plus `capacity.intersections` are at most `getGPUBufferSurfaceMaximumBoundaryRowCount(device)` (1,048,576 on a default device, a quarter of the overlay's segment table); 17 rows per vertex for the default round joins.

Both classes are planar only: `spatialContext` defaults to native planar coordinates and a longitude-latitude context throws. `GPUBufferSurface` measures distance and rejects `local-lambert-equal-area` frames; its distance is in position units, so project first (`createGPULocalFrame`, `GPULocalFrameProjection`) and buffer in the frame's meters; buffering degrees gives an ellipse on the ground.

`GPUPolygonOverlay` produces polygon geometry for `intersection`, `union`, `difference`,
`symmetric-difference`, and unary `dissolve`. It combines both operand boundaries and nodes them by
snap rounding on a lattice (`vertexTolerance` is the grid: the step is the smallest power of two
that is at least `vertexTolerance`, at least 2^-22 of the input extent and at least 2^-24 of the
largest coordinate magnitude (one float32 step there, so every lattice point is a float32 value),
anchored at a multiple of the step, so a power-of-two `vertexTolerance` reproduces shapely
`grid_size` and GEOS fixed precision whenever float32 can store that grid at the coordinates of the
data; at a projected northing of 4,000,000 the step is 0.25 to 0.5 whatever the tolerance, so use
coordinates relative to a local origin for a finer result). Vertices and crossings are rounded to the lattice, every edge is rerouted through the
hot pixels it passes, and the resulting pieces meet only at shared lattice vertices; coincident
pieces are bit-identical. The faces are then labelled without testing pieces against rings: one
ray toward +x per group of coincident pieces finds the nearest group to its right (exact predicates
on the lattice, a BVH walk), and pointer jumping along those parents sums the winding number of
each polygon beside each group and then the number of covering polygons of each operand. A group
is kept where the Boolean result changes across it, oriented with the result interior on the left,
and the kept edges are assembled into shells and holes by exact vertex matching. The result is
compact polygon geometry rather than a face mask, with every vertex on the lattice. For valid
pointwise lattice-rounded operands, the result equals GEOS OverlayNG with `grid_size` applied to
those operands (shapely on `set_precision(..., mode='pointwise')` inputs). Invalid rounded rings
retain the specified winding-rule meaning on their snap-rounded linework; GEOS polygon overlay on
invalid operands is not the oracle. In the three `crowded-pixel` intersection fixtures (grid sizes
0.25, 1 and 1/64), rounding makes both star rings self-touch with retraced spikes. The even-odd
result has one ring and lattice area 2.5, while GEOS 3.14.1 returns four polygons with area 53.5.
For example, lattice point `(31.25, 29.5)` has operand windings `(0, 1)` yet lies in that GEOS
intersection. GEOS raises for union, difference and symmetric difference on these same inputs.
The fixtures retain the GEOS output, validity reasons and counterexample alongside an independent
exact rational snap-rounding and winding oracle; this difference is specific to invalid operands.
`output.boundary` retains the operand and source feature of one row per kept group (the first piece
of the group in lattice order, so a boundary shared by two features reports one of them).

Operand membership follows `fillRule`, applied to the winding number `w` of all rings of one
polygon (counter-clockwise +1, clockwise -1). The default `'even-odd'` keeps `w` odd, which is the
shell-and-holes reading for valid polygons and also defines self-touching or self-crossing rings,
holes outside their shell and overlapping holes. `'positive'`, `'negative'` and `'nonzero'` keep `w > 0`, `w < 0` and `w != 0`,
so regions a raw boundary covers twice stay inside; a point is in the operand when it is inside any
polygon. Every decision is exact: coordinates are lattice integers, crossings are rounded by exact
integer division and ordered along the ray with 96-bit integer arithmetic. Noding reports
uncertain segment pairs, non-finite vertices and candidate loss from either intersection BVH
through `candidateOverflow`, and final output capacity loss
through `overflow`. An uncertainty counter is diagnostic; omitting it does not clear incompleteness.
`uncertainCount` counts uncertifiable retained segment pairs, rather than rejected input vertices.
The current snap-rounding path classifies finite integer endpoints in `[0, 2^22 + 1]`, within the
exact predicate's supported exponent span, so its expected uncertainty count is zero. Non-finite
or unrepresentable input still sets `candidateOverflow` even when `uncertainCount` is zero and
even when the counter is omitted. After correcting the input, the next submission rewrites both
diagnostics. Always inspect `candidateOverflow` as well as `overflow` before consuming results.
The optional `output.status.requiredRingCount` is the unclamped ring (shell plus hole) requirement
of the assembly, for sizing the output ring capacity, and the optional `ringCount` is the number of
rings written. There is no `output.status.requiredCount`: contributor contract rule 8 reserves it
for features, and the unclamped feature count is not known once rings were dropped. An output
polygon is a piece of operand unions and can cover several input features, so it has no single
source row and `output.geometry` has no `sourceIds` (a view passed there throws); provenance is per
boundary edge (`output.boundary.operandIds`, `sourceFeatureIds`). **Breaking (2026-11-10):**
`status.requiredCount` (it held the ring requirement) and the deprecated `geometry.sourceIds` (it
held the output row index) are removed. A `GPULabelPoint` joined to each operand with
`GPUPointInPolygonJoin` identifies features containing that point, not every source under the output
polygon. For example, dissolving adjacent parcels `[0,1] × [0,1]` and `[1,2] × [0,1]` yields one shell,
but the interior point `(0.5,0.5)` belongs only to the first parcel. Even all matches at that point
cannot recover the second parcel. For membership, join each output polygon to each operand with an
explicit polygon relation, such as positive-area interior intersection, and record boundary-only
contacts separately. Attribute preservation requires a relation with output feature row, operand
and source row. Optional `featureSources` reports a representative retained boundary owner and
whether retained boundary ownership is mixed; it does not enumerate that membership relation.
Boundary `sourceFeatureIds` maps each retained edge through its operand's caller `sourceIds` when
supplied, falling back to its feature row otherwise. Representative `featureSources.sourceRows`
always contains row indices, selected before caller identities are published. Assembly and
representative-owner selection use source rows internally; the final row-to-caller-ID publication
maps boundary `sourceFeatureIds` through the corresponding operand's `sourceIds`, after assembly
and representative selection. Each supplied identity view must hold at least one entry per feature
and be disjoint from output buffers.

Both operand geometry maps and their view identities are captured once at construction,
including inherited accessor-backed ID columns. Deleting or replacing caller map entries
later does not change published boundary provenance. Buffer contents remain GPU-resident data.
Assembly and representative selection retain source rows; only final boundary publication
maps those rows through the captured caller-ID views.
`GPUBufferSurface` has the same output and dissolves all features into one surface.
Both classes accept only the status views they publish: top-level `count`, `overflow`,
`candidateOverflow`, optional `ringCount`, `requiredRingCount`, `uncertainCount` and
`nodedSegmentRequiredCount`; boundary status has `count`, `requiredCount`, `overflow` and
`candidateOverflow`. Other status views throw at construction, including `invalidCount`,
`iterationCount`, `converged`, `iterationLimitReached` and `approximation`, because this pipeline
does not compute those diagnostics. Omit them and inspect the documented overflow flags and
top-level uncertainty counter. Boundary uncertainty is included in `candidateOverflow`; a
boundary `uncertainCount` view is refused.
Status fields are captured once at construction, including inherited and non-enumerable known
fields. Supplying a refused view through either form still throws. Replacing a status member on
the caller's object afterwards does not redirect deferred graph construction.
Declared constructor fields, the output descriptor and its boundary descriptor are also read
once. Inherited options such as `fillRule` retain their value, and endpoint bindings use the same
boundary descriptor as the validated status views. Result-only ring counters are refused at the
boundary location, including inherited and non-enumerable views.
The buffer and overlay demos suppress drawable rings while either completeness flag is set;
clearing both flags allows retained geometry to be drawn on the next submission.
Optional `output.status.nodedSegmentRequiredCount` reports the unclamped piece count for
`capacity.nodedSegments` (exact when `candidateOverflow` is zero). Once boundary overflow is zero,
size the three output offset arrays to `requiredRingCount + 1` and positions to
`output.boundary.status.requiredCount + requiredRingCount` (one closing vertex per ring).
Ring capacity is independent of input feature count: the A/B star-grid intersection at 1,000
vertices per operand needs 521 rings, exceeding its former 512-ring allocation. Each additional
ring needs 12 bytes across the caller's offset arrays, plus assembly scratch. For the
snap noder, expanded input rows `V` and crossing capacity `I` reserve `V + I` hot pixels; their
diagonal segment table alone needs `128 * (V + I)` bytes. Incidence capacity is
`max(nodedSegments + V, 2 * (V + I))`, with twice that many candidate pairs. Size from reported
requirements before timing repeated runs; reject truncated candidate plans. Boundary groups are
already unique, so overlay assembly skips the redundant opposing-edge cancellation search.

`GPUBufferSurface` is the topology-producing counterpart of `GPUOutlineGeometry` and
`GPUOffsetCurve`. Consecutive duplicate vertices are dropped from lines and rings first, and a
polygon ring's closing vertex is dropped when it repeats the first (GeoJSON and GeoArrow rings are
closed, the offset curve needs open rings). Lines receive independent left and right offsets plus
`round`, `flat`, or `square` caps and emit their clockwise outline. Polygons are a union of simple
pieces: the polygon itself (rings normalized to counter-clockwise shells and clockwise holes, then
reversed) plus the stroke of every ring, the region between its two closed offset rings at `+|d|`
and `-|d|`; a positive distance keeps polygon plus strokes, a negative one polygon minus strokes.
Inside turns of buffer offsets meet at the offset-line intersection while it lies on both offset
segments and are otherwise routed through the input vertex (the GEOS buffer rule), so tight
switchbacks keep a consistent winding. A ring whose envelope is narrower than `2|d|` (the GEOS eroded-ring
test) drops its offset toward its own interior, so the stroke covers that whole interior. Both paths
finish with the same noding and dissolve as overlay, using the `'negative'` winding fill rule, so
regions the raw boundaries cover twice (hairpins closer than twice the distance, self-crossing lines,
slots narrower than twice the distance) stay inside, holes a positive buffer collapses disappear, holes
a negative buffer merges through a thin wall join, and necks thinner than twice the distance split, as
in `shapely.buffer`. Checked on hairpin, figure-eight, slot, thin-neck, collapsed-hole and thin-wall
shapes. A polygon ring of `n` vertices contributes `n (1 + 2 rowsPerVertex)` raw boundary rows, so
size `capacity` for about twice the segments of a plain ring offset. Joins are `round`, `mitre`, or `bevel`; round resolution is controlled by `quadSegments`.
Positive polygon distances grow shells and shrink holes, negative distances do the reverse, and
non-positive line distances produce an empty surface.

Both contributors are planar float32 operations with explicit `intersections` and `nodedSegments`
workspace capacities plus caller-owned boundary, ring, and vertex capacities. `candidateOverflow`
means arrangement construction was incomplete, including uncertifiable predicates and non-finite
input, as well as open assembly edges and continuation conflicts. Those assembly losses set
`candidateOverflow` even when final vertex, ring and boundary capacities have spare room; increasing
only final capacity does not certify the result. Geometry, boundary provenance and representative
`featureSources` are eligible only when both `output.status.overflow` and
`output.status.candidateOverflow` are zero.
`overflow` covers capacity loss in noded segments, compact boundary or
polygon capacity. Capacity, optional outputs, join/cap style, and topology are compile-time; distance
and mitre limit remain GPU parameters and can change per encoding.

Plan overlays before allocating recipe resources with the published dimensions-only planner, then
preflight existing views before registering stages. Their public signatures are:

```ts
import type {Device} from '@luma.gl/core';
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import type {
  GPUPolygonOverlayPlanningProps,
  GPUPolygonOverlayProps,
  GPUPolygonOverlayPlan
} from '@luma-contributors/analysis/gpu-spatial-analysis';

declare function planGPUPolygonOverlay(device: Device, props: GPUPolygonOverlayPlanningProps): GPUPolygonOverlayPlan;
declare function preflightGPUPolygonOverlay<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUPolygonOverlayProps
): GPUPolygonOverlayPlan;
```

Import both functions and their overlay types from
`@luma-contributors/analysis/gpu-spatial-analysis`; `Device` is from `@luma.gl/core` and
`GPUCommandGraph` from `@luma.gl/graph/gpu-core`. The planner needs only the device and dimensions:
`left: {vertexCount, ringCount}`, optional `right` with the same fields,
`output: {vertexCapacity, ringCapacity, boundaryCapacity}`, `capacity: {intersections, nodedSegments}`,
`operation` and `vertexTolerance`, plus optional `id`, `fillRule`, `leafCapacity`, `spatialContext`
and `rightSpatialContext`. Operand counts are total allocated rows, including padding, not live
counts; output capacities are vertex, ring and segment rows respectively. Dissolve accepts only
`left`; other operations require `right`. The planner validates options, dimensions and derived
device scratch-binding limits without a graph or allocation, so a recipe can plan every overlay
before its first allocation. Resident preflight takes the full constructor props, additionally
checks resident binding ranges and graph ownership, and does not mutate the graph.

Both return a frozen `GPUPolygonOverlayPlan` with `expandedVertexCount`, `inputRingCount`,
`hotPixelCapacity`, `nodedSegmentCapacity`, `boundaryCapacity`, `outputRingCapacity` and
`outputVertexCapacity`. `expandedVertexCount` is allocated operand vertex rows plus input ring
rows; `hotPixelCapacity` adds `capacity.intersections`. These allocation checks do not certify
runtime completeness: read both overflow flags after each submission.

#### How it computes

{/* lens:GPUPolygonOverlay */}

::lens{scenario=gpu-spatial-analysis.polygon-overlay rung=datum datum=0 caption="Kept boundary edge 0: its operand and source feature"}

::lens{scenario=gpu-spatial-analysis.polygon-overlay rung=kernel caption="Noding, winding labelling, edge selection and ring assembly kernels"}

::lens{scenario=gpu-spatial-analysis.polygon-overlay rung=graph caption="Combine operands, node, label faces, select edges, assemble rings"}

{/* lens:GPUBufferSurface */}

::lens{scenario=gpu-spatial-analysis.buffer-surface rung=kernel caption="Ring offsets and compaction kernels feed the overlay dissolve"}

::lens{scenario=gpu-spatial-analysis.buffer-surface rung=graph caption="Orientation and two offset curves, then the overlay dissolve stages"}

{/* parameters: GPUBufferSurface generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUBufferSurface` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPUBufferSurface`.

- Packer: `getGPUOffsetCurveParameterValues(settings)`, writes `GPU_OFFSET_CURVE_PARAMETER_LENGTH` (4) `float32` values.
  - Input port: `parameters`.

Settings of `getGPUOffsetCurveParameterValues`:

| Field | Type | Description |
|---|---|---|
| `distance` | `number` | Signed offset in position units: positive is the left of the line direction, negative the right. |
| `mitreLimit` (optional) | `number` | Mitre length over `abs(distance)` beyond which a mitre corner is clipped. Default 5, at least 1. |

{/* /parameters */}

### `GPUAffineTransform`, `GPUGeometryOrientation`, and `GPUGeometryCleanup`

**`GPUAffineTransform`** — Applies a scale, skew, rotation and translation to every vertex, about a point, center or centroid.

::lens{scenario=gpu-spatial-analysis.affine-transform rung=result caption="Grey: 51 parks · blue: each rotated 30 degrees and scaled by 0.6 about its centroid"}

**`GPUGeometryOrientation`** — Reverses rings, or orients polygon exteriors clockwise or counter-clockwise, keeping the layout.

::lens{scenario=gpu-spatial-analysis.geometry-orientation rung=result caption="Left: parks as supplied · right: rings reversed to counter-clockwise, flag 1"}

**`GPUGeometryCleanup`** — Snaps vertices to a grid and removes repeated vertices, compacting positions and republishing offsets.

::lens{scenario=gpu-spatial-analysis.geometry-cleanup rung=result caption="Grey: 51 parks · blue: snapped to a 10 m grid, vertices within 20 m of the last kept removed"}

All three classes are planar only: `spatialContext` defaults to native planar coordinates and a longitude-latitude context throws; `GPUAffineTransform` forwards the resolved context (and its frame) to its measures. `GPUGeometryOrientation` `'orient-polygons'` judges ring winding at the lexicographically smallest vertex with exact orientation, so sliver rings are not misjudged by f32 shoelace cancellation (a bow-tie ring is judged by that vertex, like `shapely.is_ccw`, not by its signed area).

Geometry edits that keep the GeoArrow layout. Polygon inputs take the four-level layout (`positions`,
`featureOffsets`, `polygonOffsets`, `ringOffsets`), lines take `lineOffsets` (plus optional
`featureOffsets` for multi-lines) and points take `positions` (plus optional `featureOffsets`); the output
keeps the input offsets, so pass them with the output `positions` to the next contributor.
`GPUAffineTransform` is `shapely.affinity` (`affine_transform`,
`translate`, `scale`, `rotate`, `skew`): either one per-frame matrix packed by
`getGPUAffineTransformParameterValues({rotate, scale, skew, translate, origin})` (applied as scale, skew,
rotate, translate about one origin) or per-feature `featureTransforms` (6 floats per feature, Shapely
order). The origin is a point, or `'center'` or `'centroid'` computed per feature on the GPU (compile-time
`origins`); it is computed once from the input, where Shapely recomputes `'center'` at each step, which
differs only for skew about the center. Agreement was within 0.03 at coordinates near 1e5 (f32 spacing
0.0078).

`GPUGeometryOrientation` is `shapely.reverse` (`mode: 'reverse'`, polygons or lines) or `orient_polygons`
(`mode: 'orient-polygons'`, which needs the four-level polygon layout and a per-frame `exteriorClockwise`
set by `getGPUGeometryOrientationParameterValues`). The layout is
unchanged and `reversedRings` flags are optional. The sign comes from the exact turn at the
lexicographically smallest vertex; self-intersecting rings still need an explicit fill convention.

`GPUGeometryCleanup` is `remove_repeated_points(tolerance)` and `set_precision(grid_size, mode='pointwise')`:
The cleanup grid origin is zero (coverage snapping instead uses its first ring vertex as origin).
Grid rounding certifies half-grid boundaries against the original float32 operands, rather than
trusting a rounded fractional quotient; this also handles reciprocal-division backends. Certification
runs only when the magnitude of the computed float32 quotient is strictly below `2^24`.
At or above that magnitude, the shader uses the rounded float32 quotient's half-up candidate
without original-operand certification, then multiplies it by the packed spacing in float32.
That fallback can differ from the nearest grid cell of the packed operands: for
`x = 167772.1875` and `gridSize = 0.01`, the nearest cell is 16777219 and its coordinate rounds
to 167772.1875, but the fallback emits 167772.203125. Accepted packer settings do not certify
this larger cell range; `overflow` diagnoses output capacity, not snapping precision.
Within the certified range, exact
ties go toward positive infinity, including negative halves: at `gridSize = 1`, `-0.5` snaps to
`0` and `-1.5` to `-1`. It preserves already integral grid coordinates instead of
adding 0.5 in float32: with `gridSize = 1`, `8388609` stays `8388609`, the immediate float32
predecessor of 0.5 (`0.4999999701976776`) rounds to 0, and 0.5 rounds to 1. At spacing 41,
20.5 selects cell 1 (coordinate 41); at spacing 7, 58720228 stays in its exact cell 8388604.
This is pointwise snapping, not polygon topology repair. Grid size and repeat tolerance use
planar position units.
`getGPUGeometryCleanupParameterValues` preserves explicit zero for `gridSize` (disable snapping)
and `tolerance` (exact-repeat removal). Both settings must be nonnegative and finite before and
after float32 conversion; a positive value that rounds to zero is rejected instead of becoming
a disable control. NaN, infinity and finite values that convert to infinity are rejected before
the target is written. `getGPULineSimplificationParameterValues` applies the same zero,
positive-underflow and finite-conversion rules to its position-unit `tolerance`.
`getGPURectangleClipParameterValues` rectangle bounds must also remain finite after float32 conversion;
all four bounds use position units, and `maxX >= minX`, `maxY >= minY` are required.
Within the certified range, `gridSize` snaps half-up on the packed operands; `tolerance` removes repeats with the exact GEOS semantics (first
and last kept, inclusive tolerance against the previous kept vertex). Even when snapping makes every
vertex coincide, a line with at least two input vertices retains both endpoints: its required vertex
count is 2, so capacity 1 publishes `[count, requiredCount, overflow] = [1, 2, 1]`.
Polygon inputs repeat the first vertex at the end: a closed triangle therefore has four coordinate
rows and requires four output rows before any snapping or repeat removal changes it.
Line and polygon output is compacted
`positions` with republished `lineOffsets` (lines) or `ringOffsets` (polygons) and `status: {count, overflow,
requiredCount, collapsedCount}` (polygon rings with fewer than 3 surviving vertices are emptied, where GEOS
raises; the input `featureOffsets` and `polygonOffsets` stay valid). GEOS `set_precision(pointwise)`
does not drop repeats, so use `removeRepeatedPoints: false` for exact parity. With `geometryType: 'points'`,
each position row is an independent Point: the direct O(n) kernel applies only precision snapping, preserves
coincident rows, and publishes `positions` and `status: {count, overflow, requiredCount?}` without offsets
outputs. The sequential per-ring walk (latency grows with the largest ring) runs when `tolerance > 0`; at
`tolerance = 0` (exact repeats after snapping, the default) a vertex-parallel path does constant work per
vertex: a vertex repeats when its snapped position equals the previous vertex's (`-0` equals `+0`; NaN and
infinite coordinates never repeat), a per-ring step applies the closing-vertex and polygon-collapse rules, and a
per-vertex step clears the affected flags. Both paths are compiled and the GPU chooses per frame from the
`tolerance` word, with output bitwise identical to the sequential walk. Line and polygon cleanup matched Shapely exactly on 5 parameter sets by 24 features; point tests
cover dynamic snapping, ties, duplicate preservation and capacity overflow. `normalize` is not provided.

#### How it computes

{/* lens:GPUAffineTransform */}

::lens{scenario=gpu-spatial-analysis.affine-transform rung=datum datum=0 caption="Vertex 0 of Winnemac Park before and after the transform"}

::lens{scenario=gpu-spatial-analysis.affine-transform rung=formula caption="Vertices move by the packed 2x2 matrix about the origin, then the translation"}

::lens{scenario=gpu-spatial-analysis.affine-transform rung=kernel caption="One thread per vertex applies the matrix about its park origin"}

::lens{scenario=gpu-spatial-analysis.affine-transform rung=graph caption="Centroid measures feed the per-vertex transform node"}

{/* lens:GPUGeometryOrientation */}

::lens{scenario=gpu-spatial-analysis.geometry-orientation rung=datum datum=0 caption="Winnemac Park: was its ring reversed"}

::lens{scenario=gpu-spatial-analysis.geometry-orientation rung=formula caption="A ring reverses when its winding at the extreme vertex mismatches"}

::lens{scenario=gpu-spatial-analysis.geometry-orientation rung=kernel caption="One kernel decides each ring, another copies vertices in order"}

::lens{scenario=gpu-spatial-analysis.geometry-orientation rung=graph caption="Per-ring orientation node, then the per-vertex reverse node"}

{/* lens:GPUGeometryCleanup */}

::lens{scenario=gpu-spatial-analysis.geometry-cleanup rung=kernel caption="Flag, scan and emit kernels compact the surviving vertices"}

::lens{scenario=gpu-spatial-analysis.geometry-cleanup rung=graph caption="Clear, flag, exclusive scan, emit and publish nodes"}

{/* parameters: GPUAffineTransform generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUAffineTransform` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPUAffineTransform`.

- Packer: `getGPUAffineTransformParameterValues(settings?)`, writes `GPU_AFFINE_TRANSFORM_PARAMETER_LENGTH` (12) `float32` values.

Settings of `getGPUAffineTransformParameterValues`:

| Field | Type | Description |
|---|---|---|
| `rotate` (optional) | `number` | Rotation angle, counter-clockwise in a y-up system. Degrees unless `useRadians`. Default 0. |
| `useRadians` (optional) | `boolean` | Interpret `rotate`, `skew` as radians. Default false (degrees, as shapely). |
| `scale` (optional) | `number \| readonly [number, number]` | Scale factors `[x, y]`, or one number for both. Default 1. |
| `skew` (optional) | `readonly [number, number]` | Shear angles `[x, y]` (`shapely.affinity.skew`: `xs`, `ys`). Default 0. |
| `translate` (optional) | `readonly [number, number]` | Translation `[x, y]` applied last. Default `[0, 0]`. |
| `origin` (optional) | `readonly [number, number] \| 'center' \| 'centroid'` | Fixed point of the scale, skew and rotation: a coordinate, the per-feature bounding box `'center'`, or the per-feature `'centroid'` of the input geometry. Default `[0, 0]` (shapely defaults to `'center'`; ask for it explicitly). `'center'` and `'centroid'` only work when the contributor was created with the matching entry in `origins`; otherwise they act as `[0, 0]`. |

{/* /parameters */}

{/* parameters: GPUGeometryCleanup generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUGeometryCleanup` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPUGeometryCleanup`.

- Packer: `getGPUGeometryCleanupParameterValues(settings?)`, writes `GPU_GEOMETRY_CLEANUP_PARAMETER_LENGTH` (4) `float32` values.

Settings of `getGPUGeometryCleanupParameterValues`:

| Field | Type | Description |
|---|---|---|
| `gridSize` (optional) | `number` | Grid size to snap every vertex to, `round(x / gridSize) * gridSize` with ties rounding up as in GEOS (`shapely.set_precision(mode='pointwise')`). `0` (default) disables snapping. |
| `tolerance` (optional) | `number` | Distance within which a vertex counts as a repeat of the previously kept vertex (`shapely.remove_repeated_points(tolerance)`). `0` (default) removes exact repeats only. Ignored for point input because every row is an independent feature. |
| `removeRepeatedPoints` (optional) | `boolean` | Whether to drop repeated vertices after snapping. Default true. `false` with a `gridSize` reproduces `set_precision(mode='pointwise')`, which snaps but keeps repeated vertices. Ignored for point input. |

{/* /parameters */}

{/* parameters: GPUGeometryOrientation generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUGeometryOrientation` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPUGeometryOrientation`.

- Packer: `getGPUGeometryOrientationParameterValues(settings?)`, writes `GPU_GEOMETRY_ORIENTATION_PARAMETER_LENGTH` (4) `float32` values.

Settings of `getGPUGeometryOrientationParameterValues`:

| Field | Type | Description |
|---|---|---|
| `exteriorClockwise` (optional) | `boolean` | Winding of polygon exteriors in a y-up coordinate system: `false` (default) is counter-clockwise exteriors with clockwise holes (GeoJSON RFC 7946, `exterior_cw=False`), `true` is clockwise exteriors with counter-clockwise holes (shapefile, `exterior_cw=True`). |

{/* /parameters */}

### `GPUMinimumBounds`

Computes each feature's minimum-area rotated rectangle, minimum bounding circle and longest line
from its hull, plus an envelope of the finite source coordinates.

::lens{scenario=gpu-spatial-analysis.minimum-bounds rung=result caption="Grey: 51 Chicago parks · blue: their minimum-area rotated rectangles"}

Per polygon feature (the four-level layout: `positions`, `featureOffsets`, `polygonOffsets`, `ringOffsets`) or
per `uint32` label group (`positions`, `labels` and `groupCount`; use it for points and lines): the minimum-area rotated rectangle (`rectangleCorners`, four counter-clockwise
corners, and `rectangleSizes` = width, height, angle, area), the minimum bounding circle (`circles` = cx, cy,
r), the longest line (`longestLines`, `diameters`) and the envelope (`sourceBounds`). Equivalents: Shapely
`oriented_envelope`, `minimum_rotated_rectangle`, `minimum_bounding_circle`, `minimum_bounding_radius`;
Sedona `ST_LongestLine`, `ST_MaxDistance`. It is built on `GPUGroupConvexHull` (exact lattice hull) with one
thread per group over the hull. Hulls over `maximumHullVertices` (default 2048) set
`output.status.overflow` to 1 and write NaN to hull-derived shape outputs; source bounds remain
independent. Optional `output.status.groupOverflow` names this hull-limit
condition and has the same 0/1 value. Both are rewritten each frame. `hullSizes` optionally reports the
true hull vertex counts. The GPU and CPU operation forms accept the optional diagnostic; the CPU twin
computes hulls without a size limit and reports both statuses as zero. Rectangle ties (areas
within 1e-6 relative) go to the lowest hull edge, and the circle is Welzl in a hash-shuffled fixed order
with the radius recomputed as the largest hull distance, so it always covers. Parity is by area and radius,
not vertex order: rectangle area within 2e-4 relative, radius 1e-4, diameter 1e-4. Cost is `O(h^2)` per
group, and near-collinear circumcenters fall back to the diametral circle.

Planar only: a longitude/latitude `spatialContext` throws at construction; project first with
`createGPULocalFrame` and `GPULocalFrameProjection`. `output.status` has no `count`: every column has one row per group.

#### Operation forms

`GPUMinimumBoundsOperation` and `CPUMinimumBoundsOperation` are public from
`@luma-contributors/analysis/gpu-spatial-analysis`, with `MinimumBoundsOperationProps`.
They take graph value names for `positions` plus `rowCount`, and either `labels`/`groupCount`
or the three polygon offset columns plus `featureCount`. Requested output names declare live
columns: `rectangleCorners` has four `float32x2` rows per group; `rectangleSizes` (width, height,
angle in radians, area), `sourceBounds` and `longestLines` are `float32x4`;
`circles` is `float32x3`; `diameters` is `float32`; `hullSizes` and optional `precisionLimited`
are `uint32`. All except corners have one row per group.
The optional input `labelsComplete` and output `labelsComplete` each name one `uint32` word for the whole partition,
independent of the number of groups. The input is a graph value name (declared internally as
`inputLabelsComplete`); `output.labelsComplete` names the forwarded completion output.
Zero means the labels are provisional, nonzero means complete; omitted input means complete.
Bounds can be finite with `status.overflow = 0` and `labelsComplete = 0`: they describe the
current partition, not its final groups. Check label completion independently of hull overflow
before treating a grouped result as final. Request the completion output or retain the input
completion value alongside the result.
Lengths use planar position units, areas their square. Required `status.overflow` and optional
`status.groupOverflow` each have one `uint32` row for the whole operation, not one per group.
Both forms capture the shape, route and output names at construction; input values remain live.
The GPU form lowers to the contributor and enforces its hull/device budgets. The CPU form uses
float64 shape arithmetic, rounds output once, and has no hull budget: `maximumHullVertices` does
not truncate it and both status words are zero. Hull quantization and zero-area residuals can
differ between forms; compare area, radius and distance rather than corner order.

`sourceBounds` reports source-position bounds `[minX,minY,maxX,maxY]`, reducing finite source
coordinates per group independently of hull quantization and hull-capacity loss.
A `[0,0.01]^2` square beside a distant group at `(10,000,000,0)` retains its source envelope
`[0,0,f32(0.01),f32(0.01)]`; it no longer collapses with the quantized hull. Optional per-group
`precisionLimited` is 1 when the shared lattice moved a finite source coordinate, warning that
hull-dependent shapes may lose detail even when both overflow words are zero. Use separate
conditioning per group when group extents differ greatly.

#### How it computes

{/* lens:GPUMinimumBounds */}

::lens{scenario=gpu-spatial-analysis.minimum-bounds rung=datum datum=0 caption="Winnemac Park: rectangle width, height, angle, area and bounding circle"}

::lens{scenario=gpu-spatial-analysis.minimum-bounds rung=formula caption="Area minimised over the hull's edge-aligned rectangles"}

::lens{scenario=gpu-spatial-analysis.minimum-bounds rung=kernel caption="One thread per park walks its hull"}

::lens{scenario=gpu-spatial-analysis.minimum-bounds rung=graph caption="Labels, convex hull, then one node per requested shape"}

### `GPUPolygonTriangulation`

Triangulates polygons with holes into vertex-index triangles using Earcut, with a monotone fallback for large polygons.

::lens{scenario=gpu-spatial-analysis.polygon-triangulation rung=result caption="Grey: 51 parks · blue: Earcut triangles, corners gathered on the GPU from the indices"}

Earcut on the GPU (`earcut`, deck.gl `PolygonLayer` tessellation, geo `TriangulateEarcut`, Sedona
`ST_TriangulatePolygon` as earcut, not constrained Delaunay). Props: the four-level layout (`positions`,
`featureOffsets`, `polygonOffsets`, `ringOffsets`), `output: {indices, valid, triangleCount?, triangleOffsets?}` and
`maximumWork` (default 4M steps, roughly rings of a few thousand vertices). Slices, `valid` and
`triangleCount` are per polygon; the polygons of feature `f` are `[featureOffsets[f], featureOffsets[f + 1])`. The first ring is the shell and the others holes, with any winding.
`GPUPolygonTriangulationOperation` and `CPUPolygonTriangulationOperation` capture their
shape, work/routing options, spatial context and graph port names at construction.
Both are public from `@luma-contributors/analysis/gpu-spatial-analysis`, with
`PolygonTriangulationOperationProps`: geometry columns are graph value names, `polygonCount`
and `indexCapacity` declare the shape, and output names declare `uint32` `indices`, per-polygon
`valid` and optional `triangleCount`/`triangleOffsets` (the latter includes a terminal entry).
There is no global compact count; read each polygon's live triangle prefix in its own slice.
Later edits to the caller's props or output map cannot change the declared outputs or
the lowered/executed operation. Input graph values still supply fresh geometry each run;
construct a new operation to change its shape, options or ports.
The CPU function and operation enforce a positive uint32 `maximumWork` per polygon (default
4,000,000 elementary steps). Budget exhaustion publishes `valid = 0`, `triangleCount = 0`
and first-vertex failure padding, discarding partial triangles. The budget resets for each
polygon. The CPU reference uses un-hashed Earcut; its work count can differ from the GPU's hashed
path. Both forms honor `largePolygonMethod` (default `'monotone'`), its vertex threshold and total
routed raw-vertex capacity. With `'none'`, both fail the committed `wavy_grid_holes` case at 1,000
steps; the monotone retry completes it. Both operation constructors reject
zero budgets; dynamic geometry contents still change on every run.
The ear search's diagonal-midpoint inside test is an exact parity (the Simulation-of-Simplicity rule of
`utils/exact-geometry-wgsl.ts`), not an f32 ray crossing.
The hole bridge is exact as well: the shell edge that the ray from a hole's leftmost vertex meets first, the
vertices inside the bridge triangle, the candidate with the smallest angle to the ray (collinear candidates
keep the nearer one, as in Earcut) and the order of holes that share their leftmost vertex are decided by
orientation tests, so a hole a few float32 steps from a shell edge, two shell edges that the ray crosses
closer than float32 resolution, and bridge candidates in almost the same direction are bridged correctly.
CPU and GPU layout preparation skip degenerate rings and scan triangle slice capacities. A ring is degenerate
when it has fewer than three vertices or every first-vertex fan triangle has exact orientation zero;
uncertified orientations are retained, matching `GPUStudyWindowPreparation`. Empty polygons and polygons
with a degenerate shell contribute zero triangles, including any holes, and complete with `valid = 1`.
Other polygons own `3 * (vertices + 2 * holes - 2)` index slots after the skipped rings are removed.
Both Earcut and monotone triangulation retain that live-ring selection. Skipped rings do not become
monotone holes, and output indices still refer to original position rows. Their raw vertices continue
to count toward `monotoneMaximumVertices` when the polygon is routed to monotone.
Optional `triangleOffsets` has `polygonCount + 1` triangle offsets, including the terminal; polygon `p`
starts at index `3 * triangleOffsets[p]`. Non-degenerate layouts retain their original slices and triangles.
`getPolygonTriangulationIndexCount(vertexCount, ringCount, polygonCount)` returns a safe capacity:
`3 * max(0, vertexCount + 2 * ringCount - 4)` when polygons exist, otherwise zero. Smaller legacy buffers
are still accepted, with insufficient slices flagged invalid. Indices refer to rows of `positions`.
Unused polygon slots hold `(first, first, first)`; spare capacity is zeroed. With no positions, the used
triangle count is zero and nothing should be drawn. `valid[p]` is 0 for untriangulable rings, a hole that
cannot be joined to the shell, non-finite coordinates, scratch
overflow or exceeded `maximumWork` (the slice is then all degenerate, unless the monotone path completes it). One
invocation per polygon, so Earcut scales with polygon count, not ring size; ear tests use Earcut's z-order hash
above 80 vertices (`useZOrderHash`). Orientation tests use the exact predicate of
`GPUSegmentIntersection`. Empty rings and polygon slots from `GPURectangleClip` are accepted directly.
Triangle area equaled Shapely `area` and the triangle count equaled the `earcut` npm
package on 14 hand-built cases (order differs), and area matched on 60 random polygons. A seeded search
(`gpu-polygon-triangulation-search.spec.ts`) triangulates random polygons with up to 12 holes, on an integer
lattice and at projected-metre magnitude down to float32 resolution, and checks in exact arithmetic that the
triangles tile the polygon (non-zero areas, every triangle inside, no two overlapping, areas summing to the
polygon area), with the triangle count of the `earcut` package (3.2.4) where that package tiles the polygon.

A result flagged `valid = 1` is a tiling of the polygon whenever the polygon is valid (simple rings, holes
inside the shell and outside each other; any winding). Rings may touch at single points, vertex to vertex or
a vertex of one ring inside an edge of another. More precisely, the positive-area triangles tile the polygon outside its holes;
the monotone route may also emit counted zero-area rows at collinear vertices or touching holes.
These counted zero-area rows retain their original source indices, including distinct input rows
with equal coordinates. They are geometry rows, not failure padding: padding is outside `triangleCount`
in each polygon's slice. In the ten-by-ten square with two two-by-two holes touching at `(4,4)`,
monotone emits 14 rows (12 positive-area and 2 zero-area), versus Earcut's 12; both cover area 92.
The zero-area source triples are `[9,6,8]` and `[11,8,6]`, with rows 6 and 8 coincident but distinct.
Here input rows are the shell `[(0,0),(10,0),(10,10),(0,10)]`, then hole
`[(2,2),(2,4),(4,4),(4,2)]`, then hole `[(4,4),(4,6),(6,6),(6,4)]`.
Do not replace these triples with `(first,first,first)` or weld their source rows when preserving provenance.
Consumers that require positive-area faces, such as an area-sampling frame, must filter these rows.
Three departures from Earcut make the Earcut tiling contract hold. Where Earcut
cures a local self-intersection of its working ring by writing a triangle across it, the port flags the
polygon: that triangle is the only step after which the triangles written and the ring that remains no
longer add up to the polygon boundary. A hole that cannot be joined to the shell flags the polygon instead of
being dropped, and so does a predicate that cannot be certified. What is left writes only counter-clockwise
ears of exact positive area whose boundaries sum to the polygon boundary, so the number of triangles over a
point is the polygon's winding number there. An invalid polygon (rings that cross, a hole outside its shell)
is flagged, or covered as often as its rings wind: **behavior change** for such input, which Earcut's cure
used to return as valid.

A hole whose leftmost vertex lies inside an edge of another ring is joined there: the vertex is inserted into
the edge (exact on-segment test) and the bridge has zero length. Two holes that share their leftmost vertex
are ordered by the edge that leaves it, also when that vertex is repeated, and of two edges that the bridge
ray meets at one point the one whose side holds the hole is taken. On 126,382 valid polygons with up to
twelve holes (90,985 on integer lattices and lattice shears with coordinates up to six million, 35,397 of
them again at projected-metre magnitude, half of those at float32 resolution; four in five with touching
rings) the Earcut port returned 126,381
tilings and flagged one, a sliver at float32 resolution. That historical run preceded the exact
diagonal-midpoint rule described above. The exact historical W2-02 fixture is unavailable:
the requested 32-vertex, six-hole input is not the committed 23-vertex, three-hole record with the
same number. Rescue of that specific input by the current default monotone retry remains unverified;
the touching-hole and default-route controls do not establish that rescue. The `earcut` package (3.2.4) returned a wrong triangulation
for 641 of them. Before these rules the port flagged 535 and returned 32 wrong triangulations as valid.
The pinned sample is `gpu-polygon-triangulation-touching-rings.spec.ts` (Shapely validity and area,
`test/oracles/polygon-triangulation-touching-rings.py`). Delaunay variants
are separate because this contributor preserves polygon constraints and holes.

Large polygons (`largePolygonMethod`, default `'monotone'`; `'none'` keeps `valid = 0` past `maximumWork`).
A polygon with at least `monotoneMinimumVertices` vertices (default 4096, all rings) skips Earcut, and every
polygon on which Earcut fails (`maximumWork` exceeded or no valid split) is retried, by a parallel y-monotone
decomposition (de Berg et al., chapter 3) followed by a linear stack triangulation of each monotone piece.
The output contract is unchanged: the same slice, `valid` and `triangleCount`, unused slots degenerate, and
polygons that Earcut triangulates are bit-identical to `'none'`. The steps are data parallel and dispatched
over the vertices of routed polygons only, sized by GPU-resident counts: clean the rings (consecutive
duplicates and the closing vertex dropped), rank the vertices in sweep order (y descending, x ascending, row
last, a total order that needs no special case for equal y), orient the rings (shell counter-clockwise, holes
clockwise) and classify every vertex as start, end, split, merge or regular with the exact orientation, find
the edge directly left of every split, merge and right-boundary vertex through the shared edge-slab index,
derive the helper diagonals from the members of each left edge sorted by (edge, rank), order the at most four
diagonals of a vertex by exact angle, label the pieces by pointer jumping on the half-edge `next` pointers
(rounds in GPU-gated pairs), and triangulate each piece on one invocation. A piece is triangulated by one
invocation, so a single huge convex piece is linear but not parallel. Triangles are counter-clockwise, and a
polygon with collinear points keeps zero-area triangles (`triangleCount` counts them), which Earcut would
have filtered. `valid` is 0 when the piece triangles do not add up to `vertices + 2 * holes - 2`, a chain is
not monotone, a predicate is uncertain, a diagonal is out of order, a walk does not close, a ray finds no
edge, or a triangle would be clockwise. The triangles of the pieces sum to the polygon boundary by
construction (every ring edge is in one piece, every diagonal in two with opposite directions, and each
triangle replaces two consecutive edges of its piece by the third side), so with no clockwise triangle a
valid result tiles a valid polygon; triangles of zero area between collinear vertices add nothing. Not
detected: rings that cross or overlapping holes that leave all of these consistent, which are then covered
as often as the rings wind. Not supported (flagged invalid by the monotone path): most polygons whose rings
touch at a vertex (87 of every 100 in the search above; the others are tiled). The clockwise test is what
flags the rest of them: without it the path alone returned 106 wrong triangulations as valid on those 126,382
polygons, and one in the default configuration on a polygon of 4,306 vertices that `'none'` tiles. A polygon that
skipped Earcut and that the monotone path flags
(or that exceeds the capacity) is then given to Earcut within `maximumWork` in a second pass, so `'monotone'`
never does worse than `'none'`: it is valid wherever `'none'` is, and a polygon the monotone path returns as
valid is a tiling. Cost: the sweep sort, the member sort and the scans are sized by the capacity
(`monotoneMaximumVertices`, default all vertices, at most `getGPUPolygonTriangulationMonotoneMaximumVertexCount(device)`,
3,355,443 on a default device), so they run even when no polygon is large; pass `largePolygonMethod: 'none'` or a
smaller capacity for inputs known to be small. Large polygons beyond the capacity (in row order) go to the
Earcut fallback pass. Inputs with more rows than that limit skip the monotone path entirely and behave like `'none'`; only an explicit `monotoneMaximumVertices` above the limit throws. Scratch is roughly 100 `uint32` words per vertex (the largest binding, the pointer-jumping arrays,
is 40 bytes per vertex). The GPU spec `gpu-polygon-triangulation-monotone.spec.ts` checks polygons of 5,000 to
200,000 vertices against the area of Shapely and of `shapely.constrained_delaunay_triangles`
(`test/oracles/geometry-triangulation-large.py`), the exact triangle count, and the cancellation of the directed
triangle edges to the polygon boundary (which proves coverage without overlap); the CPU model of the algorithm
and the WGSL kernels (in a WGSL interpreter) are checked on adversarial polygons: combs, spirals, stars, 50 holes,
collinear runs, equal heights, duplicate vertices and random polygons.

Maximum: the scratch holds `2 * vertices + 5 * (rings - polygons) + 8 * polygons` nodes (`getPolygonTriangulationNodeCount`; one node per hole more than before 2026-11-11, for the vertex inserted into a touched edge), at most `getGPUPolygonTriangulationMaximumNodeCount(device)` (4,793,490 on a default device, 28 bytes per node; 8,388,608 with `useZOrderHash: false`). `GPURandomPointsInPolygon` and `GPUDotDensity` triangulate internally and apply the same limit under their own ID.

Planar only: `spatialContext` defaults to native planar coordinates and a longitude-latitude context throws; project to a planar frame first (`createGPULocalFrame`, `GPULocalFrameProjection`). Orientation signs are exact.

#### How it computes

{/* lens:GPUPolygonTriangulation */}

::lens{scenario=gpu-spatial-analysis.polygon-triangulation rung=datum datum=0 caption="Winnemac Park: triangle count and whether Earcut finished"}

::lens{scenario=gpu-spatial-analysis.polygon-triangulation rung=kernel caption="One thread per park clips ears from its ring list"}

::lens{scenario=gpu-spatial-analysis.polygon-triangulation rung=graph caption="Layout preparation, Earcut and monotone fallback, then the display gather"}

<a id="constrained-delaunay-triangulation"></a>

### `GPUConstrainedDelaunayTriangulation`

Use `GPUConstrainedDelaunayTriangulation` for a valid planar polygon's interior mesh, retaining
shell and hole boundaries as constraints. It starts with the polygon triangulation above, then
performs bounded Lawson flips of strictly illegal interior edges; cocircular diagonals stay put.
The input is the four-level polygon layout, not an unconstrained point cloud. Output indices
refer to original position rows. `valid`, `triangleCount`, optional `triangleOffsets`, padding
and counted zero-area rows follow [`GPUPolygonTriangulation`](#gpupolygontriangulation).
The per-polygon `output.status.converged` is independent of `valid`: exhausting `maximumFlips`
(default 4096) can leave a valid mesh with `converged = 0`, which is not certified Delaunay.
The GPU's `flipMethod: 'auto' | 'serial' | 'parallel'` selects scheduling, not another constraint rule.

`GPUConstrainedDelaunayTriangulationOperation` and `CPUConstrainedDelaunayTriangulationOperation`
are public from `@luma-contributors/analysis/gpu-spatial-analysis`, with
`ConstrainedDelaunayTriangulationOperationProps`. They take graph names for `positions`,
`featureOffsets`, `polygonOffsets` and `ringOffsets`, compile-time `polygonCount`/`indexCapacity`,
and graph output names for scalar `uint32` indices, per-polygon `valid`, optional `triangleCount`,
optional terminal `triangleOffsets`, and required per-polygon `status.converged`.
Read only `triangleCount[p]` triples from polygon `p`'s slice; indices outside that prefix are padding.
Both graph forms require at least one polygon slot and one index row; empty polygon slots are
allowed. Their pure CPU function also accepts zero-length outputs.

The GPU form lowers to the bounded contributor. The CPU form uses budgeted un-hashed Earcut
and the selected monotone route for the initial mesh, then flips to completion, ignoring
`maximumFlips`. It uses arbitrary-precision predicate fallbacks for all finite float32 exponent
spans, while the GPU's incircle fallback is bounded to span 55. Initial triangulations can differ
with GPU hashing; successful CPU polygons report convergence 1. Both forms preserve slice allocation
and hole exclusion. See the [historical raster-expansion route](../gpu-raster/expansion.md#operations-and-contracts)
for capability C24; this task destination preserves that page's URL.

### `GPUDelaunayTessellation` and `GPUVoronoiDiagram`

`GPUDelaunayTessellation`: Scratch triangles are bounded by 1,073,741,823 and output triangles by
1,431,655,765, in addition to device bindings.

`GPUVoronoiDiagram`: Triangle and segment counts are bounded by 357,913,941 and 1,073,741,823
respectively, with device bindings imposing lower limits.

**`GPUDelaunayTessellation`** — Triangulates planar points into counter-clockwise triples of point rows, with exact predicates.

::lens{scenario=gpu-spatial-analysis.delaunay-tessellation rung=result caption="Grey dots: 319 Wicker Park places (every fourth) · blue: their Delaunay triangles"}

**`GPUVoronoiDiagram`** — Derives Voronoi edges from Delaunay triangles, each with the two sites it separates.

::lens{scenario=gpu-spatial-analysis.voronoi-diagram rung=result caption="Same 319 places · blue: Voronoi edges, hull rays clipped 100 m beyond the places"}

`GPUDelaunayTessellation` builds an unconstrained planar point triangulation by parallel insertion
and flipping. Exact duplicate coordinates use first-row ownership, collinear input produces no
triangles (so `GPUVoronoiDiagram`, which derives edges only from triangles, returns nothing for two
sites or for collinear sites, although those have bisectors), and cocircular ties are broken by Simulation of Simplicity on row order (a lifted
perturbation), so the result is the unique perturbed Delaunay triangulation whatever the parallel
schedule. Every output triangle is counter-clockwise, rotated so its smallest row comes first, and the
rows are sorted by their first two indices.
`output.status` uses canonical `count`, exact `requiredCount`, `overflow`, and optional
`invalidCount`; `status.duplicateCount` reports discarded site rows. Optional `status.converged`
is one only for successful convergence, `status.iterationCount` counts active steps, and
`status.iterationLimitReached` distinguishes budget exhaustion from invalid input. Use
`getDelaunayMaximumTriangleCount(pointCount)` for the general-position upper bound.
Grouped `triangleCounts` and `triangleOffsets` describe the unclamped required layout, even when
output capacity truncates the stored triangles. Intersect each group's range
`[triangleOffsets[g], triangleOffsets[g] + triangleCounts[g])` with `[0, status.count)` before
indexing the triangle output. For required counts `[2,2,0]`, offsets `[0,2,4]` and capacity one,
the readable group lengths are `[1,0,0]`.

#### Delaunay operation forms

`GPUDelaunayTessellationOperation` and `CPUDelaunayTessellationOperation` are public from
`@luma-contributors/analysis/gpu-spatial-analysis`, with `DelaunayTessellationOperationProps`.
They take a `positions` graph value name (`float32x2`, planar position units), triangle `capacity`,
and optional `groups: {ids, count}` (`uint32` labels). `output.triangles` declares `capacity`
`uint32x3` source-row triples. Required one-row `uint32` `status.count`, `requiredCount` and
`overflow` delimit the readable prefix and its unclamped requirement. Optional diagnostics
use the same status names described above. Grouped input and `output.groups` are supplied together;
group `triangleCounts`, optional `triangleOffsets`, and `status.converged` have one `uint32` row
per group. Their required ranges must still be intersected with the global readable prefix.

The GPU form uses parallel insertion and bounded Lawson rounds. The CPU form uses sequential
cavities, exact arbitrary-precision predicate fallbacks, and no GPU step/scratch/command budget;
`iterationCount` counts inserted distinct finite sites rather than GPU active rounds, and
`iterationLimitReached` is always zero. It retains first-row duplicate ownership and row-order
cocircular tie breaking. CPU padding is zeroed; GPU rows beyond `status.count` are unspecified.
Both Delaunay graph forms accept `capacity: 0`, as does the pure CPU function. This count-only
route allocates no triangle rows; read `status.requiredCount` for demand, `status.count` for the
stored prefix and `status.overflow` for truncation. Three noncollinear sites such as
`(-1,0),(1,0),(0,3)` give `count = 0`, `requiredCount = 1`, `overflow = 1` at zero capacity.
This measures the result's required triangle capacity without claiming that any triangles were stored.

#### Construction and Voronoi consumption

The first three steps insert a seed simplex: the first and last distinct sites in the coordinate-bit
sort, then the first site not collinear with that edge (exact orientation). This puts a lone apex
into the triangulation before bulk insertion of collinear sites, preventing the former linear flip
chain. Collinear input simply skips the third seed. Subsequent steps insert, in parallel, one site into every triangle that still contains one (the site of
smallest pseudo-random priority, a bijective hash of its row; a site on an edge splits both triangles of
the edge), then runs one parallel Lawson flip pass: an edge whose quad fails the exact in-circle test
flips when it holds the smallest key of both of its triangles. Insertion is valid in any triangulation
and exact Lawson flipping converges from any planar triangulation, so no repair stage is needed (gDel2D
needs star splaying only because its float flips are inexact). Steps stop on the GPU when every site is
in and no edge is illegal: the remaining steps are indirect dispatches of nothing. The initial budget
`B = 4 ceil(log2(n + 2)) + 32 + ceil(sqrt(n) / 2)` now continues on those same GPU state words up to
`T = min(4096, max(B, 2n + 32))` steps for nonempty input. Eligibility for the short parallel
schedule and serial GPU continuation requires both `pointCount ≤ 2048` sites and allocated scratch triangles
`2 * pointCount + max(groupCount, 1) + 7 ≤ 4104`, including empty groups. On that route
`P = min(B, T)` rounds are encoded in parallel. The two bounded GPU continuation kernels are added
only when `P < T`; they finish any remaining insertion
and then exact Lawson legalization with one active invocation each. They reuse the topology and
predicates, without CPU readback or
reducing the iteration limit. For 3 sites, `P = 45` and `T = 45`: no continuation kernels are
encoded. For 10 sites, `P = 50` and `T = 52`: two continuation kernels are encoded.
Empty input encodes zero rounds and no continuation kernels.
At 1,000 sites this encodes 794 iteration nodes instead of 18,288.
Exceeding either bound retains parallel execution for every allowed round, with nine nodes per
round. For 2040 sites, 17 groups allocate 4104 scratch triangles and encode 99 parallel rounds;
18 groups allocate 4105 scratch triangles and encode 4096 parallel rounds. The GPU phase counter
keeps parallel steps on the same nine pipelines; no host iteration enters shader source or overrides.
Active work is O(n) per pass, with no universal logarithmic convergence claim. The rare small-input continuation trades
parallel throughput for fewer encoded dispatches; its worst-case latency remains input-dependent.
At the documented maximum, unconverged work retains the currently accepted real triangles,
increments `invalidCount`, and publishes `converged = 0`, `iterationLimitReached = 1`,
`approximation = 1` and the actual `iterationCount`. This partial triangulation may omit sites
and need further legalization; consumers requiring a complete Delaunay triangulation must reject it.
This is a bounded
safety net, not a proof that every arbitrary Lawson chain finishes in 4096 steps. The CPU WGSL executor
checks 2000 collinear-plus-apex sites against `T = 2n - 2 - h` (1998 triangles) and exact empty circles.

The starting super triangle is symbolic: its three vertices are points at infinity (towards +x, +y and
-x, with infinitesimally tilted directions and hierarchically growing radii), never finite coordinates.
Every orientation or incircle test that touches one reduces to exact comparisons of real coordinates:
a triangle with one infinite vertex has the half-plane left of its real edge as circumcircle (a site on
the edge's line is inside only when strictly between the endpoints), and a triangle with two or three
infinite vertices is decided by one exact orientation sign. Real hull triangles are therefore never
lost to a finite super vertex, so the result is the complete Delaunay triangulation of the convex hull,
`2n - 2 - h` triangles for `n` distinct sites and `h` hull-boundary points. An earlier finite
supertriangle (corners at 32 extents) dropped hull triangles, for example 2 of about 1200 for 600
uniform points and 51 of 185 for a 1e-2-thick band.

`GPUVoronoiDiagram` consumes those triangle rows and their GPU-resident count without readback.
Optional Delaunay `candidateOverflow` and `uncertainCount` receive zero; `approximation` is one
when iteration exhaustion retains partial topology and zero after a successful solve. These
diagnostics are also available on the Delaunay GPU operation and CPU form.
When triangles come from Delaunay, forward **both** producer words as `triangleStatus.overflow`
and `triangleStatus.converged`. The fields remain optional, but each observes only its own failure:
supplied `overflow != 0` detects triangle capacity loss, and supplied `converged = 0` detects
solver exhaustion. With `triangleStatus` supplied, a requested count exceeding the triangle view
is independently detectable and also suppresses segments. Merely supplying the object does not
establish that all upstream failure evidence was forwarded.
The reasons remain distinct: solver exhaustion alone clears both Voronoi capacity flags;
triangle capacity loss sets both. Keep the original producer evidence to identify the reason.

For otherwise valid triangles, let `R` be the required clipped segment count from complete input
and `C` the final segment capacity. All counts below are segment rows. Each row assumes the
specified input evidence is supplied; optional output fields are shown when their views are supplied:

| Input/result condition | Required input evidence | `count` | `requiredCount` (demand) | Segments suppressed? | Solver evidence: `converged` | `complete` | `overflow` | `candidateOverflow` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Complete input, sufficient final capacity | Both producer words: `overflow = 0`, `converged = 1`; requested rows fit | `R` | `R` | No | `1` | `1` | `0` | `0` |
| Solver exhausted, sufficient storage | Supplied `triangleStatus.converged = 0`, with no capacity loss | `0` | `0` | Yes | `0` | `0` | `0` | `0` |
| Truncated triangle input | Supplied `triangleStatus.overflow != 0`, or oversized requested count with `triangleStatus` supplied | `0` | `0` | Yes | `0` | `0` | `1` | `1` |
| Final segment overflow | Both producer words: `overflow = 0`, `converged = 1`; requested rows fit | `C` | `R > C` | No; only `C` rows fit | `1` | `0` | `1` | `0` |

Forwarding only `overflow = 0` hides solver exhaustion; forwarding only `converged = 1` hides
triangle truncation when the producer's count is already clamped. In either case, one readable
triangle with enough segment storage can emit three segments and `complete = 1`. Output
completeness certifies only the input evidence actually supplied, not an omitted producer word.

Suppressed demand is zero, not an estimate of the full diagram's missing segments. Without
`triangleStatus`, an oversized `triangleCount` retains the legacy readable-prefix behavior:
`requiredCount` describes that prefix alone, `count` is clamped to `C`, `candidateOverflow = 1`,
`converged = 0` and `complete = 0`; `overflow` then reflects only final segment capacity loss.
A clamped count alone cannot establish triangle completeness. The Voronoi lens supplies both
Delaunay overflow and convergence.

Voronoi performs no iterative solve: optional `iterationCount`, `iterationLimitReached`,
`uncertainCount` and `approximation` receive zero, even after upstream solver exhaustion. Those
local zeros do not replace the producer's solver evidence. Optional `converged` is one only when
no input triangle was rejected (`invalidCount = 0`), the requested triangle count fits the view
and supplied upstream evidence is successful; it does not certify final capacity. Optional
`complete` additionally requires the segment demand to fit the final capacity. Ordinary float32
rounding still applies. `converged` must use a separate physical buffer from `invalidCount`. `complete` must use a
separate physical buffer from both `invalidCount` and `requiredCount`, even for disjoint byte
ranges, because its publication reads both diagnostics. A disjoint packed
`requiredCount`/`converged` layout remains supported when no completeness destination aliases
those reads. `count` and `overflow` must use a separate physical buffer from `requiredCount`.
`unresolvedCount` is unsupported and rejected at construction. Status fields are
rewritten on every submission. Output slots beyond `count` are not written.
Internal edges connect adjacent circumcenters; hull bisectors extend to a dynamic one-row
`clipBounds` view. Its compact `float32x4` segments retain the two source site IDs and canonical
bounded status. Input triples may have either winding; hull rays always point away from the
opposite site. Voronoi and alpha hull share independently power-of-two-scaled edge equations,
preserving the short-axis component when edge lengths differ greatly. The center is restored after
adding its origin in a scaled frame, so an overflowing relative offset or radius does not invalidate
a representable center. Hull directions use scaled endpoints when subtraction would overflow;
ray exits come from the clip bounds without a fixed distance cap.
Collinear triangles, an uncertifiable orientation, or an unrepresentable final center produce no
Voronoi edges and increment `status.invalidCount` when supplied. The solve remains float32;
exact orientation does not promise an exact center for nearly collinear triangles.

Circumcircle conditioning matters independently of the predicate sign. Two nearly cancelling
products may produce a zero cross or a nonzero but inaccurate cross after edge subtraction rounds.
The shared Voronoi/alpha-hull construction detects cancellation when the cross magnitude is at
most `2^-10` times the sum of the product magnitudes. It chooses an origin incident to the edge
with the smallest maximum component magnitude, preserving the nearby vertices' separation, and
recomputes the product difference with integer significands. Correcting the products alone cannot
recover an edge component already lost during subtraction. Scaled center restoration then adds
that selected origin before returning to the original coordinate scale.

The two integer controls are `(0,0),(8192,8191),(8191,8190)` and
`(0,0),(8388608,8388607),(8388607,8388606)`: both have exact determinant -1 and
exercise product cancellation. Separate symmetric dyadic probes use `(1,1),(u,0),(0,u)`,
`u = 2^-26`, with analytic center `(0.5000000037252903,0.5000000037252903)`.
The asymmetric dyadic probes use `(1,1),(0,2^-26),(2^-24,0)`, with center
`(0.2000000247359275,0.7999999871850011)` and radius `0.8246211042342516`;
an earlier construction returned `(0,1)` and radius 1 despite a nonzero cross. Both dyadic
families test all three cyclic orders. Public-consumer probes separately check clipped Voronoi
bisectors and alpha-hull retention at circumradius 0.9. The earlier thin-integer helper test log
records a device pass for its tested revision; it does not qualify the later origin-selection
change or public-consumer outputs. Only retained device readbacks qualify the specific fixture,
revision and output fields read. Current dyadic/cyclic-order and public-consumer device readbacks
remain pending; interpreter and arithmetic-model results are not device qualification.

For adjacency, require a positive-length shared Voronoi boundary. Cocircular triangles can have
equal circumcenters, so a Delaunay diagonal alone does not establish Voronoi neighbors. The square
probes check both triangulations and retain active segment rows, site IDs and counts when run.
No retained device observation here establishes whether a counted zero-length diagonal is emitted;
do not infer one from the triangle indices or present it as an observed row.
Both operations are deterministic for a fixed row order and use finite planar f32
coordinates; they do not claim constrained Delaunay, polygonal Voronoi cells, or concave construction.
Orientation and incircle tests are adaptive exact on the f32 coordinates. Both are planar only: a
longitude/latitude `spatialContext` throws at construction (project first with `createGPULocalFrame`
and `GPULocalFrameProjection`), because a lng/lat Delaunay is not a spherical one.

#### How it computes

{/* lens:GPUDelaunayTessellation */}

::lens{scenario=gpu-spatial-analysis.delaunay-tessellation rung=datum datum=0 caption="Triangle 0: three place rows, counter-clockwise, smallest row first"}

::lens{scenario=gpu-spatial-analysis.delaunay-tessellation rung=kernel caption="Exact incircle tests decide every Lawson flip"}

::lens{scenario=gpu-spatial-analysis.delaunay-tessellation rung=graph caption="Seed simplex, parallel insertion and flip rounds, then triangle publication"}

{/* lens:GPUVoronoiDiagram */}

::lens{scenario=gpu-spatial-analysis.voronoi-diagram rung=datum datum=0 caption="Segment 0: its endpoints and the two places it separates"}

::lens{scenario=gpu-spatial-analysis.voronoi-diagram rung=kernel caption="One thread per triangle edge joins circumcenters or casts a hull ray"}

::lens{scenario=gpu-spatial-analysis.voronoi-diagram rung=graph caption="Sorted edge keys give adjacency, then segments are compacted"}

### `GPULabelPoint`

featureCount is at most 67,108,863 and gridSize at most 65,535; the output and squared grid address
space must also fit the device.

Finds an interior label point for each polygon, its pole of inaccessibility, with the inscribed radius around it.

::lens{scenario=gpu-spatial-analysis.label-point rung=result caption="Grey: 51 parks · points: label points coloured by inscribed radius in metres"}

A label point inside each polygon, the pole of inaccessibility (turf `pointOnFeature`, Shapely
`representative_point`, polylabel). Props: the four-level polygon layout (`positions`, `featureOffsets`,
`polygonOffsets`, `ringOffsets`), `initialGridSize` (16), `refinementGridSize` (8), `refinementRounds` (6), `refinementCandidates` (4) and
`output: {points, publishedCenterClearance?, precisionLimited?, degenerate?}`. The rule is even-odd across rings. It starts from seeds
(scanline interval midpoints, guaranteed interior), then a coarse grid, then refines the best cells whose
bound can still win. If no part of an archipelago is hit by the three feature scanlines, they are repeated
through the bounds of the largest ring before the search falls back to the first vertex. The inside test
is an f32 even-odd crossing that only places candidates (it is not a topology claim; the exact rule is
`utils/exact-geometry-wgsl.ts`). `publishedCenterClearance` is the inscribed radius at the written point,
and `degenerate = 1` means the published point has no positive clearance (including empty or
unusable labels). It tests label usability, not polygon validity or the mathematical existence
of an interior point. It is planar and greedy: on long multi-lobed shapes it can miss the true pole (the worst ratio to
Shapely `polylabel` was above 0.97 in the random test), so raise `initialGridSize` or
`refinementCandidates`. Cost is sequential per feature, except that polygons with more than
`smallFeatureVertexLimit` vertices (default 32) use a workgroup path with bit-identical results.

Label bounds use the contiguous vertex interval from the feature's first ring offset to its final
ring offset. If the final offset is at or below the first, the feature publishes a NaN point and
clearance and sets `degenerate = 1`, even if an individual ring contains vertices.
`CPULabelPointOperation` defaults to the GPU's resident offset semantics and accepts
`storage: 'compact'` to check the full hierarchy. `computeLabelPointOnCPU` defaults to compact
validation and accepts
`storage: 'resident-capacity'` for resident snapshots.

Clearance is recomputed at the published float32 center after rounding the search candidate.
`publishedCenterClearance` has one `float32` row per feature, in planar
position units; optional `precisionLimited` is one `uint32` row per feature and is 1 when the
reported clearance is below the search clearance. Currently this includes clearance scalar rounding:
the diamond `(±3,0),(0,±3)` keeps center `(0,0)` but sets the flag when `3/sqrt(2)` rounds downward.
The flag alone does not prove the center moved. Both clearance spellings must use the same
destination if supplied together.
A square starting at
`(1048576,1048576)` with side 0.125 has true center `(1048576.0625,1048576.0625)` and radius
0.0625, but no float32 coordinate strictly between its edges. Its published point is a boundary
corner: the CPU twin now reports clearance 0, `degenerate = 1` and `precisionLimited = 1`, rather
than assigning the unrounded center's positive radius to that point. Subtract a common local origin before
float32 upload, perform the search and downstream predicates in that frame, and restore the
origin only for presentation. More search rounds cannot recover an unrepresentable coordinate.
Representability and the distinction between search error and roundoff are described for
[`GPUMaximumInscribedCircle`](../gpu-geometry-refinement.md#circle-outputs-and-coordinate-precision).

Planar only: a longitude/latitude `spatialContext` throws at construction; project first with
`createGPULocalFrame` and `GPULocalFrameProjection`. Declares `measure: 'distance'`: a Lambert
equal-area local frame throws; use an azimuthal-equidistant frame.

#### How it computes

{/* lens:GPULabelPoint */}

::lens{scenario=gpu-spatial-analysis.label-point rung=datum datum=0 caption="Winnemac Park: label point and inscribed radius"}

::lens{scenario=gpu-spatial-analysis.label-point rung=kernel caption="A thread or workgroup per park refines a grid search for the pole"}

::lens{scenario=gpu-spatial-analysis.label-point rung=graph caption="One label node: polygons in, points, radii and degenerate flags out"}

### `GPUShapeDescriptors`

Moment routing selects 256-edge tiles from 4,096 rows when scratch fits; the explicit serial route
remains. Moment tile count is uint32 bounded, and reduction scratch must fit the device.

Computes compactness, elongation, orientation, convexity and sliver flags for each polygon from its rings.

::lens{scenario=gpu-spatial-analysis.shape-descriptors rung=result caption="Left: 51 parks · right: coloured by Polsby-Popper compactness, 1 is a circle"}

Shape indices per polygon (momepy compactness, QGIS shape metrics, esda `shape`). Props: the four-level
polygon layout (`positions`, `featureOffsets`, `polygonOffsets`, `ringOffsets`; ring role by position),
`parameters` (`getGPUShapeDescriptorsParameterValues({sliverThreshold})`) and an `output` with any of `areas`,
`perimeters`, `polsbyPopper`, `schwartzberg`, `elongation`, `orientation`, `convexity`, `clockwise` and
`sliver`. It builds on `GPUGeometryMeasures` (area, perimeter, centroid) plus central second moments.
Both shape operation backends capture supported output columns by name at construction,
including prototype getters; inherited columns are declared and written like own properties.
`elongation = 1 - sqrt(lambda2 / lambda1)`. `orientation` is the major axis in radians in `(-pi/2, pi/2]`, and exactly 0 when the second moments are
isotropic (an axis-aligned square or a circle).
`convexity = area / hull area` uses a private gift-wrapping hull (`O(n * h)` per feature) for small features and `GPUGroupConvexHull` (`O(n log n)`, balanced) for large ones: `convexityMethod: 'auto'` chooses the monotone chain when features average at least 256 vertices, or when the optional `maximumVerticesPerFeature` prop declares a largest feature of at least `GPU_SHAPE_DESCRIPTORS_MONOTONE_CHAIN_LARGEST_VERTICES` (1024) vertices, so one huge feature among many small ones is not walked by a single invocation. The moments node still integrates one feature per invocation; a segmented-sum version (per-edge terms reduced with `sorted-segment-sums`) would balance it. Outputs have one row per feature and no `status.count`. `clockwise` is
read from the first ring (y up), and `sliver` is set when Polsby-Popper is below the per-frame threshold.
Coordinates are planar and degenerate features give NaN. A longitude/latitude `spatialContext` throws at
construction. Project first with `createGPULocalFrame` and `GPULocalFrameProjection`.
If `frame.unit` or a recognized `frame.crs` identifies an angular unit in a planar context, the
constructor rejects `areas`, `perimeters`, `polsbyPopper`, `schwartzberg`, `sliver` and `convexity`.
Project the positions to a planar linear frame for those outputs. For `orientation` alone, use
`coordinateSpace: 'planar'` and `metric: 'none'` with the angular frame and omit `units`. For example,
a planar `EPSG:4326` frame works in that context. Changing its metric to `'native'` throws at
construction. The resolved context, including its `frame` identity,
reaches the internal `GPUGeometryMeasures`. The gift-wrapping hull picks each next vertex with the
exact orientation predicate (`orientSign`) on the original positions.

Both hull methods use the contiguous vertex interval from the first ring offset to the final ring
offset. A final offset at or below the first gives an empty hull interval and NaN convexity. Area,
perimeter and moments are computed from the individual rings, so an earlier ring can still
contribute. If resident feature windows go out of order, the GPU uses per-feature gift wrapping
so that each window can use all its vertices, including rows shared with another window.
`CPUShapeDescriptorsOperation` defaults to these resident offset semantics and
accepts `storage: 'compact'` to check the full hierarchy.
`computeShapeDescriptorsOnCPU` checks the compact hierarchy by default and accepts
`storage: 'resident-capacity'` for resident snapshots.

#### How it computes

{/* lens:GPUShapeDescriptors */}

::lens{scenario=gpu-spatial-analysis.shape-descriptors rung=datum datum=0 caption="Winnemac Park: area, compactness, elongation and convexity"}

::lens{scenario=gpu-spatial-analysis.shape-descriptors rung=formula caption="Compactness, convexity, elongation and sliver flag from area, perimeter and moments"}

::lens{scenario=gpu-spatial-analysis.shape-descriptors rung=kernel caption="One thread per park accumulates area, moments and its hull"}

::lens{scenario=gpu-spatial-analysis.shape-descriptors rung=graph caption="Geometry measures feed the shape descriptor kernels"}

{/* parameters: GPUShapeDescriptors generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUShapeDescriptors` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPUShapeDescriptors`.

- Packer: `getGPUShapeDescriptorsParameterValues(settings?)`, writes `GPU_SHAPE_DESCRIPTORS_PARAMETER_LENGTH` (4) `float32` values.

Settings of `getGPUShapeDescriptorsParameterValues`:

| Field | Type | Description |
|---|---|---|
| `sliverThreshold` (optional) | `number` | Polsby-Popper compactness below which a polygon is flagged as a sliver. Default `GPU_SHAPE_DESCRIPTORS_DEFAULT_SLIVER_THRESHOLD`. |

{/* /parameters */}

### `GPULineDensity`

`output.status.requiredCount` counts reserved traversal records, including zero-length records
at an interior grid corner. For `(.5,.5)` to `(1.5,1.5)` on a unit `2 × 2` grid, two positive
cell spans require three records because the traversal advances both axes at the corner.
A `recordCapacity` of two therefore sets `overflow=1`; capacity three fits. Use the reserved
count, rather than the number of positive cell contributions, when sizing this buffer.

`GPU_LINE_DENSITY_PARAMETER_SCHEMA` exposes typed `GPUSpatialParameterSchema<'float32'>`
metadata, including `wordLength` and `fields`, through the public spatial-analysis entry point.
Its type is preserved for native NodeNext TypeScript consumers as well as workspace imports.

Sums street length per grid cell and divides by cell area, producing a line-density raster.

::lens{scenario=gpu-spatial-analysis.line-density rung=result caption="Grey: 223 Wicker Park streets · blue: metres of street per 100 m cell"}

Line length per grid cell and its density (length in the cell over cell area; not the QGIS or ArcGIS "Line density" tool, see Semantics). Props: `positions`, `lineOffsets`,
`columns`, `rows`, optional `spatialContext` (native planar by default, or longitude-latitude with the great-circle metric),
`recordCapacity`, `parameters`
(`getGPULineDensityParameterValues({minX, minY, cellWidth, cellHeight})`) and
`output: {lengths, densities?, status: {overflow, requiredCount?}}`. Segments are clipped to the grid and walked cell
by cell, and the lengths are summed per cell by a stable sort and a fixed-order reduction, so results are
bitwise reproducible. Spherical mode measures each clipped piece in the context's length unit using
the mean radius (`radius`); cell areas use `getGPUAuthalicSphereRadius` (the new `areaRadius` field),
in that unit squared. Lengths default to meters and cell areas to square meters; `units: 'kilometers'`
gives kilometers and square kilometers. A longitude/latitude segment is cut by the grid along the
straight line between its vertices in degrees and each piece is measured as a great circle, so the
total of a long segment depends on the grid (the segment (0, 60) to (90, 60) is 4,605 km as one piece and
about 5,004 km when cut by 1 degree cells): densify long segments first. Exceeding `recordCapacity` sets `status.overflow` and under-counts the
last segments. The intended physical boundary policy assigns an interior horizontal boundary
to the physical northern cell and an interior vertical boundary to the physical eastern cell.
Output row and column order only permutes the classified cells; see the float32 classification
limitation below. Closed rings are not special (pass ring edges as lines). Length per polygon is
[`GPULineLengthPerPolygon`](#gpulinelengthperpolygon).

For axis-aligned raster metadata, pair `getGPULineDensityParameterValues(grid)` with
`getGPULineDensityRasterOutputMapping(grid)` and matching dimensions:

```ts
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {
  getGPULineDensityParameterValues,
  getGPULineDensityRasterOutputMapping
} from '@luma-contributors/analysis/gpu-spatial-analysis';

const grid: GPURasterMetadata = {
  width: 1,
  height: 2,
  affine: [1, 0, 0, 0, -1, 2],
  pixelInterpretation: 'area'
};
const rasterSettings = {
  columns: grid.width,
  rows: grid.height,
  ...getGPULineDensityRasterOutputMapping(grid)
}; // Spread into GPULineDensity props; upload these packed parameters separately.
const parameterValues = getGPULineDensityParameterValues(grid);
```

Both parameter overloads and the raster mapping helper reject a minimum whose float32 encoding
moves it by at least one original or encoded cell size; subtract a common local origin from
positions and grid placement. The mapping helper also rejects an independently encoded zero
step even if the two corners round to different float32 values.

The packer retains `[minimumX, minimumY, positiveCellWidth, positiveCellHeight]`; the mapping
helper supplies `outputRowOrder: 'north-first' | 'south-first'` and
`outputColumnOrder: 'east-first' | 'west-first'` to match raster storage. Legacy settings default
to south-first rows and west-first columns. For a `1 × 2` north-up area grid with affine
`[1,0,0,0,-1,2]`, the packed values are `[0,0,1,1]` and mapping is north-first/west-first.
Interior segment `(.25,1.5) → (.75,1.5)` and boundary segment `(.25,1) → (.75,1)` each contribute
length `.5` to the physical northern cell: both give `[.5,0]` north-first; the boundary gives
`[0,.5]` south-first. Likewise, on a unit `2 × 1` grid, boundary segment
`(1,.25) → (1,.75)` belongs to the physical eastern cell, giving `[0,.5]` west-first and
`[.5,0]` east-first. Reversing storage order changes indices, not cell ownership or lengths.

These unit-grid examples have exactly representable grid coordinates. General float32 grids can
misclassify an exact physical boundary before permutation: for a `1 × 4` grid with
`h = Math.fround(0.11)`, `minY = -h`, `cellHeight = h`, a horizontal segment of length `.5` at
`y = 2*h` lies on boundary 3. Rounded division produces 2.999999761581421 before `floor`, so the
current traversal assigns south-first row 2 or north-first row 1, rather than the intended rows
3 and 0. The storage permutation cannot repair that earlier classification. The north/east
ownership policy is therefore not an exact-boundary guarantee for arbitrary float32 spacing/origins.

Maximum: `columns * rows` is at most `GPU_LINE_DENSITY_MAXIMUM_CELL_COUNT` (16,777,215), a device-independent constant: the per-cell sum dispatches one 256-thread workgroup per cell and the invocation index is a `u32`.

Uses `resolveGPUSpatialContext` and its shared validation errors. Declares `measure: 'area'`:
an azimuthal-equidistant local frame throws; project into a Lambert equal-area frame.

**Semantics.** `densities` is the line length inside the cell divided by the cell area, a box kernel
with the grid cell as its support, in context length units per square unit (meters per square meter
by default for longitude/latitude, kilometers per square kilometer with `units: 'kilometers'`). It is not the ArcGIS or QGIS circular-search-radius line density,
which spreads each line over a disk and is smooth; those need a smoothing step. For a smooth field,
rasterize `lengths` (or `densities`) and convolve it with `GPUConvolution` (with the same cell-index and
zero-boundary caveats as [`GPUPointDensity`](./operators-density.md#gpupointdensity) smoothing), or sample line points with
`GPURandomPointsOnLine` and use [`GPUKernelDensity`](./operators-density.md#gpukerneldensity).

#### How it computes

{/* lens:GPULineDensity */}

::lens{scenario=gpu-spatial-analysis.line-density rung=datum datum=97 caption="A central cell (column 7, row 6): road length and density"}

::lens{scenario=gpu-spatial-analysis.line-density rung=formula caption="Cell length sums clipped pieces; density divides by cell area"}

::lens{scenario=gpu-spatial-analysis.line-density rung=kernel caption="Segments walk the cells they cross, then pieces are summed per cell"}

::lens{scenario=gpu-spatial-analysis.line-density rung=graph caption="Count, scan and emit pieces, sort by cell, then a segmented sum"}

{/* parameters: GPULineDensity generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPULineDensity` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPULineDensity`.

- Packer: `getGPULineDensityParameterValues(settings)`, writes `GPU_LINE_DENSITY_PARAMETER_LENGTH` (4) `float32` values.
- Packer: `getGPULineDensityParameterValues(grid)`, writes `GPU_LINE_DENSITY_PARAMETER_LENGTH` (4) `float32` values.
  - `grid` (`GPURasterMetadata`): the frame grid, which supplies the cell size and row edges

Settings of `getGPULineDensityParameterValues(settings)` for `GPULineDensitySettings`:

| Field | Type | Description |
|---|---|---|
| `minX` | `number` | X of the grid's lower-left corner. |
| `minY` | `number` | Y of the grid's lower-left corner. |
| `cellWidth` | `number` | Cell width in position units (degrees for `'spherical'`). |
| `cellHeight` | `number` | Cell height in position units (degrees for `'spherical'`). |

{/* /parameters */}

### `GPULineLengthPerPolygon`

Sums the street length inside each polygon, with the count of segments that contribute.

Optional `featureOffsets` maps feature rows to linestring parts, while `lineOffsets` still maps
parts to vertices. Every part's clipped length contributes; empty features contribute zero.
With grouping, `lineWeights` has one value per feature and weights every segment of every part
of that feature. Without grouping it retains one value per linestring. `segmentCounts` counts
contributing segments, including separate overlapping parts; it does not count distinct features.
Candidate loss in any part sets `status.candidateOverflow` and `status.overflow` for the complete
operation. Retained lengths remain partial when either status reports capacity loss.
An edge with either endpoint nonfinite contributes nothing and consumes no candidate slot.
An all-empty multiline batch clears the polygon outputs and capacity statuses. Boundary-coincident
pieces use closed clipped-intersection semantics: outer/hole edges count, and a shared edge
contributes to both polygons. Point-only touches have zero length and do not count as segments.

::lens{scenario=gpu-spatial-analysis.line-length-per-polygon rung=result caption="Grey: sampled streets crossing three community areas · blue: total clipped street length in metres"}

The 448 sampled street centrelines span Lincoln Park (area 7), Logan Square (22), and West Town (24).
Both inputs use UTM 16N metres with the same false origin. Lengths describe this street extract,
not each community area's entire network. Regenerate only this extract with
`python3 scripts/lens-data/s3-geometry-data.py --line-length-per-polygon-only`.

Total line length inside each polygon (QGIS "Sum line lengths", PostGIS
`sum(ST_Length(ST_Intersection(line, polygon)))`). Props: `positions`, `lineOffsets`, `polygons` (a four-level
`GPUPolygonGeometryPort`), optional `lineWeights`, `spatialContext` (omitted: planar/native; or
longitude-latitude with the great-circle metric, replacing the former `coordinateSystem` and `radius`),
`pairCapacity`
(default `max(1024, 8 * positions.length)`), `maximumCrossingCount` (default 64, at most 1024: the crossings live in private memory) and
`output: {clippedLengths, weightedLengths?, segmentCounts?, status: {overflow}}`. Every vertex row that is not the last of its path
is a segment. Segments meet polygons through `GPUSpatialJoinCandidates`, are cut at every ring-edge crossing, and
each piece whose midpoint is inside (even-odd over all rings, so holes and multipolygons work, decided by the
exact point-in-ring rule of `utils/exact-geometry-wgsl.ts`; the cut parameters are f32 and only place pieces) is kept. Lengths
are summed per polygon by a stable sort and a fixed-order segmented sum, so results are bitwise reproducible
(segment counts use integer atomics). `weightedLengths` is `sum(clipped segment length * featureWeight)`: with `featureOffsets`,
one feature-row weight applies to every segment of every part owned by that feature; without it,
each linestring is its own feature. `segmentCounts` counts segments with positive length inside. A longitude-latitude context clips straight in
longitude/latitude, as `GPULineDensity` does, and measures each piece as the great-circle distance between its
endpoints. `status.overflow` is 1 when the candidate, BVH or crossing capacity was exceeded, and lengths can then be low.
Optional `status.candidateOverflow` isolates the candidate stage (pair capacity or BVH); it is 0 when only the
crossing bound was exceeded. The output is per polygon, so there is no per-row flag.

`computeLineLengthPerPolygonOnCPU` accepts the same typed-array line/feature offsets, polygon
hierarchy, optional feature weights and explicit pair/crossing budgets. It returns `clippedLengths`,
weighted lengths, contributing-segment counts and both overflow diagnostics. Float64 clipping and
accumulation round once into float32 output; near-edge fractions may differ from the GPU.
Boundary-coincident pieces use the same closed intersection on both backends. Invalid offset
hierarchies throw and valid empty arrays are accepted.
Internal `createLineLengthPerPolygonOperation` and `createCPULineLengthPerPolygonOperation` share
one packed planar contract. Geographic lengths remain available through the contributor.

Maxima: `positions` has at most `getGPULineLengthPerPolygonMaximumPositionCount(device)` (8,388,608 on a default device) rows and `pairCapacity` is at most `getGPULineLengthPerPolygonMaximumPairCapacity(device)` (16,777,216; the default is clamped to it). `maximumCrossingCount` is in [1, 1024].

Longitude/latitude edges must be linear. Prepare polygons that cross ±180 or enclose a pole with
`GPULonLatPolygonPreparation`; optional one-row `uint32` fields
`output.status.antimeridianUncertainCount` / `output.status.poleUncertainCount` report unprepared
rings. They are cleared and rewritten every encoding in lon/lat contexts. Supplying them with a
planar context is allowed and leaves them untouched.

Declares `measure: 'distance'`: a Lambert equal-area local frame throws; use an azimuthal-equidistant
frame. Longitude/latitude requires `metric: 'great-circle'`; other metrics throw the shared
`resolveGPUSpatialContext` error. Lengths use the context's length unit (meters by default for
longitude/latitude, kilometers with `units: 'kilometers'`).

Limits: lines are not unioned, so overlaps count once per line, as in QGIS; f32 precision (about
1e-6 of the segment length). The spec
checks 150 paths against 40 polygons against a float64 oracle at 5e-4 relative with exact counts, and spherical
lengths against a haversine oracle. No QGIS pin.

```ts
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  GPULineLengthPerPolygon,
  type GPULineLengthPerPolygonProps
} from '@luma-contributors/analysis/gpu-spatial-analysis';

// Fragment inputs: graph-owned geometry views and caller-owned output views.
declare const graph: GPUCommandGraph;
declare const positions: GraphDataView<'float32x2'>;
declare const lineOffsets: GraphDataView<'uint32'>;
declare const polygons: GPULineLengthPerPolygonProps['polygons'];
declare const clippedLengths: GraphDataView<'float32'>;
declare const segmentCounts: GraphDataView<'uint32'>;
declare const overflow: GraphDataView<'uint32'>;
graph.add(new GPULineLengthPerPolygon({positions, lineOffsets, polygons, output: {clippedLengths, segmentCounts, status: {overflow}}}));
```

#### How it computes

{/* lens:GPULineLengthPerPolygon */}

::lens{scenario=gpu-spatial-analysis.line-length-per-polygon rung=datum datum=2 caption="Community area 24 (West Town): metres of street and segment count"}

::lens{scenario=gpu-spatial-analysis.line-length-per-polygon rung=formula caption="Length inside a polygon sums pieces; weights and counts use the same pieces"}

::lens{scenario=gpu-spatial-analysis.line-length-per-polygon rung=kernel caption="Segments are cut at ring crossings and a midpoint test keeps inside pieces"}

::lens{scenario=gpu-spatial-analysis.line-length-per-polygon rung=graph caption="Candidate segment-polygon pairs, crossing cuts, then per-polygon sums"}

### `GPUGridGenerator`

Generates square, hexagon, triangle or point grids from an origin and cell size, flagging cells that touch a polygon.

::lens{scenario=gpu-spatial-analysis.grid-generator rung=result caption="Hexagons coloured by contact with West Town · grey: 1,275 places"}

Grid geometry (turf `squareGrid`, `hexGrid`, `triangleGrid`, `pointGrid`, QGIS create grid, PostGIS
`ST_SquareGrid`). Props: `gridType` (`'square'`, `'hex'`, `'triangle'` or `'point'`), `columns`, `rows`,
`parameters` (`getGPUGridGeneratorParameterValues({minX, minY, cellWidth, cellHeight})`) and
`output: {positions?, centers?, corners?, cornerIntersects?}`. Size cell outputs with
`getGPUGridCellCount` and `getGPUGridVerticesPerCell`; size the unique square or hex vertex lattice
with `getGPUGridCornerCount`. Hexagons are pointy-top by default (`cellWidth` is flat to flat), triangles come in
alternating strips with half-cell row shifts (not turf's exact layout), and the point grid sits at cell
centers. The cell counts are compile-time. The lattice is planar: `spatialContext` accepts only a planar
context (the default) and throws for longitude/latitude, so project first or use `GPUPointToCell`.

The bounded axes are `columns` and `rows`, not the total number of cells. Construction enforces
these shape/orientation limits so every integer lattice index converted to float32 is at most
`2^24`, where consecutive integers are exactly representable:

| Grid shape and orientation | Maximum `columns` | Maximum `rows` |
| --- | --- | --- |
| Square without `output.centers` | 16,777,216 (`2^24`) | 16,777,216 (`2^24`) |
| Point, or square with `output.centers` | 8,388,608 (`2^23`) | 8,388,608 (`2^23`) |
| Triangle | 8,388,607 (`2^23 - 1`) | 16,777,216 (`2^24`) |
| Pointy hexagon (default) | 8,388,607 | 5,592,405 |
| Flat hexagon | 5,592,405 | 8,388,607 |

Triangle columns use half-width lattice steps; hexagons use doubled and tripled indices, with
the axes swapped for flat hexagons. These are index-precision bounds, not a promise that the
allocation fits a device. Cell/output counts and storage bindings impose additional limits.
Coordinate multiplication and addition still round: a large `minX`/`minY` with small cell dimensions
can merge positions even inside these bounds. Use a local origin and representable cell dimensions.
The center-specific admission bound applies to every point grid and to square grids requesting
`output.centers`: construction rejects either axis above 8,388,608 cells so half-cell indices
stay exact. Without this bound, even with origin zero and unit width, the center expressions at
zero-based columns 8,388,609 and 8,388,610 would both round to 8,388,610. A local origin alone
cannot fix this index-magnitude limit; split the grid into smaller locally indexed tiles.
Square corner-only requests and square polygon output without centers retain the larger integer
lattice bound. Before allocating views, call
`GPUGridGenerator.getMaximumAxisCellCounts({gridType, hexOrientation, centers})`; it returns
`{columns, rows}` maxima, where `centers` indicates whether center output is requested (point
grids always use the smaller bound). Coordinate scaling and translation can still round even
within these index limits; they do not promise distinct coordinates for arbitrary origins/sizes.

- `hexOrientation: 'flat'` (GeoPandas `make_grid(flat_topped=True)`): `cellWidth` is still the
  flat-to-flat distance (GeoPandas `cell_size`), columns step `1.5 * R`, rows step `cellWidth`, odd
  columns shift up half a cell, and the ring is counter-clockwise from angle 0. The lattice is anchored at
  `(minX - R / 2, minY)`, where GeoPandas starts half a cell off the offset. Matched GeoPandas to 1e-5 on
  16 cells.
- `extent` (a nested `{kind: 'polygons', positions, featureOffsets, polygonOffsets, ringOffsets}` port) and
  `output.intersects` (GeoPandas `make_grid(intersect=True)`): a per-cell `uint32` flag
  against a polygon (even-odd rings, holes, multi-polygon) with exact orientation predicates. Boundary
  contact counts, and point grids test the center. Cost is cells times extent edges. It is a flag, not a
  compaction: compact it with `GPUCompaction` for the subset. Matched Shapely `intersects` on five grid
  types (240 cells).
- `output.corners` implements GeoPandas `make_grid(feature_type='corners')` for square and hex grids.
  It emits every shared and boundary vertex exactly once using an integer lattice, without sorting or
  floating-point hashing. `output.cornerIntersects` independently tests those points against `extent`,
  including polygon and hole boundaries. Triangle and point grids do not expose corner output.
- Cell polygons compute every vertex from the same integer-lattice expression as `output.corners`, so
  neighbouring cells share bit-identical vertices (no 1-ulp cracks or overlaps along seams).

#### How it computes

{/* lens:GPUGridGenerator */}

::lens{scenario=gpu-spatial-analysis.grid-generator rung=datum datum=84 caption="A central hexagon (column 6, row 6): corners, centre and area flag"}

::lens{scenario=gpu-spatial-analysis.grid-generator rung=formula caption="Cell centre from the lattice, or the vertex mean for hexagon and triangle cells"}

::lens{scenario=gpu-spatial-analysis.grid-generator rung=kernel caption="One thread per cell builds corners from integer lattice coordinates"}

::lens{scenario=gpu-spatial-analysis.grid-generator rung=graph caption="Cell corners and centres, then the extent mask"}

{/* parameters: GPUGridGenerator generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUGridGenerator` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPUGridGenerator`.

- Packer: `getGPUGridGeneratorParameterValues(settings)`, writes `GPU_GRID_GENERATOR_PARAMETER_LENGTH` (4) `float32` values.

Settings of `getGPUGridGeneratorParameterValues`:

| Field | Type | Description |
|---|---|---|
| `minX` | `number` | X of the lower-left corner of the grid. |
| `minY` | `number` | Y of the lower-left corner of the grid. |
| `cellWidth` | `number` | Cell width. For `'hex'` it is the distance between opposite flat sides of a hexagon (the horizontal width when pointy-top, the vertical height when flat-topped; GeoPandas `make_grid(cell_size)`); for `'triangle'` the base of one triangle. |
| `cellHeight` | `number` | Cell height. Ignored by `'hex'` (a regular hexagon's pitch follows from `cellWidth`); for `'triangle'` it is the height of one strip (use `cellWidth * sqrt(3) / 2` for equilateral). |

{/* /parameters */}

### `GPUShapeGenerator`

Generates circle, sector or ellipse rings around centres and radii, with segment count set per frame.

::lens{scenario=gpu-spatial-analysis.shape-generator rung=result caption="Blue: 60 m circles around every 12th place · grey: all 1,275 places"}

Circles, sectors and ellipses around per-feature centers and radii (turf `circle`, `sector`, `ellipse`).
Props include `shape` (`GPUShapeType`), `centers`, `radii`, optional sector `bearings` and ellipse
`rotations`, `spatialContext` (planar by default, or `{coordinateSpace: 'longitude-latitude', metric:
'great-circle'}` with spherical destination and radii in the context's length unit, meters by default on the mean Earth radius; this replaces
the removed `coordinateSystem` prop and `GPU_SHAPE_GENERATOR_EARTH_RADIUS`), compile-time `maximumSegments` and per-frame `parameters`
(`getGPUShapeGeneratorParameterValues`). The segment count and the radius scale are per-frame, up to
`maximumSegments`; size outputs with `getGPUShapeVertexCount` and `getGPUShapeMinimumSegments`. The output
is packed rings plus GeoArrow-style vertex `offsets` (feature `f` starts at `f * V`) and an optional
`vertexCount`. Use `output.offsets` as **`ringOffsets` only**. For one ring and one polygon per
feature, `polygonOffsets` and `featureOffsets` are identity offsets `[0,1,...,featureCount]`;
they index rings and polygons respectively, not vertices. For two closed four-segment diamonds,
the three-level descriptor is:

```ts
import {
  computeLabelPointOnCPU,
  getGPUShapePolygonIdentityOffsets
} from '@luma-contributors/analysis/gpu-spatial-analysis';

// Two generated rings, five stored XY vertices each (including closure).
const generatedPolygons = {
  positions: Float32Array.of(
    1, 0, 0, 1, -1, 0, 0, -1, 1, 0,
    11, 0, 10, 1, 9, 0, 10, -1, 11, 0
  ),
  ringOffsets: Uint32Array.of(0, 5, 10), // rings -> vertices
  polygonOffsets: getGPUShapePolygonIdentityOffsets(2), // [0,1,2], polygons -> rings
  featureOffsets: getGPUShapePolygonIdentityOffsets(2) // [0,1,2], features -> polygons
};
const labels = computeLabelPointOnCPU(generatedPolygons);
// labels.points = [0, 0, 10, 0]; both features survive the handoff.
```

For graph consumers, import/upload these offset columns as `uint32` views alongside the generated
`float32x2` positions. Changing the segment count changes vertex offsets, not identity offsets.
Circles and sectors match turf 7.4 exactly (pinned). Ellipses default to the original
equal-parameter sampling; `ellipseSpacing: 'arc-length'` builds a normalized 257-sample quarter-ellipse
table per feature and inverts it for approximately equal edge lengths. The table is built before rotation
and, for geodesic shapes, before spherical destination; it costs 1028 transient bytes per feature and one
additional compute pass. Coincident sector bearings keep their center spokes. Geodesic f32 coordinates
round to about 0.5 m.

Declares `measure: 'distance'`: a Lambert equal-area local frame throws; use an
azimuthal-equidistant frame.

#### How it computes

{/* lens:GPUShapeGenerator */}

::lens{scenario=gpu-spatial-analysis.shape-generator rung=datum datum=0 caption="Place 0: its 24-segment circle ring of 25 vertices"}

::lens{scenario=gpu-spatial-analysis.shape-generator rung=formula caption="Each vertex offsets the centre by the radius along its bearing"}

::lens{scenario=gpu-spatial-analysis.shape-generator rung=kernel caption="One thread per output vertex"}

::lens{scenario=gpu-spatial-analysis.shape-generator rung=graph caption="Ring vertices, then ring offsets"}

{/* parameters: GPUShapeGenerator generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUShapeGenerator` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPUShapeGenerator`.

- Packer: `getGPUShapeGeneratorParameterValues(settings)`, writes `GPU_SHAPE_GENERATOR_PARAMETER_LENGTH` (2) `float32` values.

Settings of `getGPUShapeGeneratorParameterValues`:

| Field | Type | Description |
|---|---|---|
| `segmentCount` | `number` | Segments (arc intervals) per shape this frame, clamped on the GPU to `[minimum, maximumSegments]` (minimum 3 for circles and ellipses, 1 for sectors). |
| `radiusScale` (optional) | `number` | Multiplier applied to every radius. Defaults to 1. |

{/* /parameters */}

### `GPUHilbertKeys`

Computes the Hilbert curve index of each point or box centre, optionally sorting rows into curve order.

::lens{scenario=gpu-spatial-analysis.hilbert-keys rung=result caption="Places coloured by order-6 Hilbert key: neighbours on the curve share colour"}

2D Hilbert curve index (order 1 to 16, `uint32`) of `positions`, or of feature box centers (`minima` and
`maxima`), within per-frame `bounds` `[minX, minY, maxX, maxY]` (or the GPU-reduced bounds of the valid
items when omitted; items outside are clamped to the border cells). `output.keys` holds the keys,
bit-exact against the CPU `xy2d` on exactly representable data, and optional `sortedRows` and `sortedKeys`
give the stable curve-order permutation from a radix `GPUSort`. Invalid items get key `4^order`
(`getGPUHilbertInvalidKey(order)`) and sort last. The default order is `GPU_HILBERT_DEFAULT_ORDER` (15), so the
invalid key never collides with a curve cell; at order 16 the invalid key `0xffffffff` equals the last cell. Cell borders depend on the f32
`(v - min) / extent`, and keys beyond order 16 (two words) are not provided.

`spatialContext` accepts planar coordinates or longitude/latitude with metric `'none'` (plain
coordinates for ordering; no antimeridian wrap). Metric contexts throw.

The Hilbert order clusters better than Morton: a CPU tree-cost proxy (sum of node perimeters of a complete
binary BVH over 16,000 boxes) was 27 percent lower on uniform, 21 percent on clustered and 15 percent on
road-like data. The joins' `spatialSort` now uses it by default (see
[Spatial sort default](./operators-joins.md#spatial-sort-default)).

#### How it computes

{/* lens:GPUHilbertKeys */}

::lens{scenario=gpu-spatial-analysis.hilbert-keys rung=datum datum=0 caption="Place 0: its curve cell and Hilbert key"}

::lens{scenario=gpu-spatial-analysis.hilbert-keys rung=kernel caption="Quantise to a cell, then a bitwise Hilbert transform per place"}

::lens{scenario=gpu-spatial-analysis.hilbert-keys rung=graph caption="Bounds reduction, keys, then a stable radix sort by key"}

### `GPULonLatPolygonPreparation`

Prepares lon/lat polygons for planar consumers by densifying edges, splitting at the antimeridian and capping pole rings.

::lens{scenario=gpu-spatial-analysis.lon-lat-polygon-preparation rung=result caption="Blue: 13 harbour zones densified to 0.005 degrees; none crosses the antimeridian"}

Prepares longitude/latitude polygon rings for linear-degree containment, rasterization and other
polygon consumers. All classification, densification, capping and splitting runs on the GPU;
construction validates only declared properties and view lengths. Non-pole edges take the short
longitude path. A degree-plane edge along the same pole retains its raw longitude step, so
Natural-Earth-style Antarctica needs no extra cap or split.

| Prop | Contract |
| --- | --- |
| `positions`, `featureOffsets`, `polygonOffsets`, `ringOffsets` | `GPUPolygonGeometryFields`; implicit ring closure, shell first and holes afterward. |
| `spatialContext` (required) | Declare `coordinateSpace: 'longitude-latitude'`, metric `'none'` or `'great-circle'`, and optional `edgeModel: 'linear'` (default) or `'great-circle'`. Omission has no default and throws at construction. |
| `parameters` | Four float32 words from `getGPULonLatPolygonPreparationParameterValues(edgeModel, settings?, target?)`; `GPU_LON_LAT_POLYGON_PREPARATION_PARAMETER_LENGTH = 4`. |
| `maximumPieceCount` | Integer in `[1, 65536]`, default `1024`, caps pieces per original edge. |
| `antimeridian` | `'split'` (default) or `'reject'` crossing features. |
| `poleRings` | `'cap'` (default) or `'reject'` pole-enclosing features. |
| `output` | `GPULonLatPolygonPreparationOutput`, caller-owned views and status scalars. |
| `id` | Defaults to `'lon-lat-polygon-preparation'`. |

`GPULonLatPolygonPreparationSettings.maximumSegmentDegrees` is the largest longitude or central-angle
step in degrees: default `0.1` for great-circle input and `0` for linear input. Zero disables
densification; great-circle input with zero is passed through as linear and sets `approximation`.
The helper rejects negative or NaN steps with
`getGPULonLatPolygonPreparationParameterValues maximumSegmentDegrees must be >= 0`.

The output `spatialContext` preserves the input metric, units and datum and declares
`edgeModel: 'linear'`. `output.positions` is a capacity-sized `float32x2` view with written longitudes
in `[-180, 180]`. Given F features, P polygons and R rings, allocate `featureOffsets` with F + 1
rows, `polygonOffsets` with 2P + 1 rows and `ringOffsets` with 2R + 1 rows. Each input feature keeps
its row: `featureOffsetsOut[f] = 2 * featureOffsetsIn[f]`. Within a feature, all main polygon copies
come first, then all east copies. Ring roles are preserved, including zero-vertex east copies for
noncrossing polygons. Unused rings and polygons stay in the topology. Any invalid ring empties its
whole feature; ambiguity, width greater than 360 degrees, nonfinite coordinates, latitude outside
`[-90, 90]`, or either reject option can cause this.

Optional `ringFlags` has R uint32 rows: bit 0 crossing, bit 1 pole-capped, bit 2 ambiguous edge,
bit 3 wider than 360 degrees, bit 4 nonfinite coordinate or invalid latitude, bit 5 densification
clamped. `status.overflow` is required; optional `count` reports vertices written, `requiredCount`
reports the exact final total when no intermediate stage overflowed, `invalidCount` counts rejected
features once each, and `approximation` is one when great-circle densification was clamped or disabled.
Dense, cap and both copies' clip streams each have capacity `output.positions.length`; clipping runs
only against longitude -180 and 180 because classification validates latitudes. Overflow from any
stage, including final merge, sets `overflow = 1`; all offsets remain monotone and within capacity.
Status is rewritten each encoding. If an intermediate stage overflowed, `requiredCount` instead
reports the conservative bound `8 * (D + 3 * R)`, saturated at `0xffffffff`, where D is the unclamped dense vertex count
(after `maximumPieceCount`) and R is the input ring count. Capping adds at most three vertices per
ring, each longitude boundary at most doubles a stream, and there are two copies. When representable,
this bound fits every intermediate stream and the final output on a rerun with the same input and parameters.
Size `output.positions` from `requiredCount`; transient memory scales with it. Dense scan offsets
also saturate at `0xffffffff`, preserving the written prefix after a uint32 carry. Additional source-row
scan arrays and topology metadata scale with input vertices and rings, without a `maximumPieceCount`
multiplier. An overflowed intermediate result is incomplete; use the resized rerun for analysis.

For example, a four-vertex square without seam crossings needs four output vertices. Capacity three
already overflows the dense stream and reports the conservative requirement `8 * (4 + 3) = 56`.
With sufficient capacity, disabling great-circle densification still writes the four original
vertices and sets `approximation = 1`; this does not reject the feature or set `invalidCount`.
Linear edges with no densification instead give `approximation = 0`. These status words and each
full `ringFlags` word overwrite previous caller values on resubmission.

Applications can format readback diagnostics with these exact messages (the library does not log):

- `${n} features rejected: ambiguous or over-wide rings`
- `${n} rings cross the antimeridian; prepare polygons first` (consumer `antimeridianUncertainCount`)
- `${n} rings enclose a pole; prepare polygons first` (consumer `poleUncertainCount`)
- `densification clamped; raise maximumPieceCount`

Sutherland-Hodgman clipping can leave zero-width bridges on the seam. The half-open containment
rule assigns a point at -180 to the west side only; a point at +180 belongs to neither side.
Kept canonical vertices retain their exact float32 bits. Wrapped longitude plus integer turns
persists through intermediate buffers; reconstructed longitudes exist only in shader registers.

A ring winding once around longitude is capped at the nearer pole, using the sign of its
mid-latitude `(minLat + maxLat) / 2`; exactly zero is ambiguous. The cap retains the unwrapped
closing endpoint before the two pole vertices, preserving the final edge. The heuristic ignores
winding direction and can choose the wrong cap for an intended complement. Represent that
complement as a world shell with the ring as a hole, or choose `poleRings: 'reject'`.
Great-circle minor arcs through a pole, and antipodal endpoints, are ambiguous in v1.

Edge models matter for long edges. At 45 degrees latitude, an east-west edge spanning 1 degree
bows about 120 m as a great circle; 10 degrees gives about 12 km. A 60-degree edge reaches about
49.1 N instead of 45 N, a deviation of about 455 km. Approximate H3 average-edge deviations at 45 N:

| Resolution | Average edge length | Great-circle / linear-degree deviation |
| --- | --- | --- |
| 0 | 1281 km | about 32 km |
| 1 | 483 km | about 4.6 km |
| 2 | 183 km | about 660 m |
| 3 | 69 km | about 93 m |
| 4 | 26 km | about 13 m |
| 5 | 9.9 km | about 1.9 m |

Multiply these estimates by about 1.7 at 60 degrees. The originally proposed 1.2 m bound for a
0.1-degree step is slightly low: the binary64 oracle measures 1.336 m on the 60-degree edge at
45 N. Use **about 1.3 m** for that fixture (the regression limit is 1.4 m), before float32 position
rounding; this is not a universal certified error bound. See the longitude/latitude precision
notes for the additional coordinate quantization error.


#### How it computes

{/* lens:GPULonLatPolygonPreparation */}

::lens{scenario=gpu-spatial-analysis.lon-lat-polygon-preparation rung=datum datum=0 caption="Input ring 0: its classification flags"}

::lens{scenario=gpu-spatial-analysis.lon-lat-polygon-preparation rung=kernel caption="Densify, classify, cap, normalise and clip stages"}

::lens{scenario=gpu-spatial-analysis.lon-lat-polygon-preparation rung=graph caption="Scans publish the doubled main and east topology"}

{/* parameters: GPULonLatPolygonPreparation generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPULonLatPolygonPreparation` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPULonLatPolygonPreparation`.

- Packer: `getGPULonLatPolygonPreparationParameterValues(edgeModel, settings?)`, writes `GPU_LON_LAT_POLYGON_PREPARATION_PARAMETER_LENGTH` (4) `float32` values.
  - `edgeModel` (`GPUSpatialEdgeModel`)

Settings of `getGPULonLatPolygonPreparationParameterValues`:

| Field | Type | Description |
|---|---|---|
| `maximumSegmentDegrees` (optional) | `number` | Maximum longitude or central-angle step in degrees; zero disables densification. |

{/* /parameters */}

### `GPURectangleClip`

Optional polygon output publishes feature, polygon and ring offsets for the clipped rectangles;
compact counts describe only the emitted geometry.

Clips lines or polygon rings to an axis-aligned rectangle, rejoining the surviving line pieces.

::lens{scenario=gpu-spatial-analysis.rectangle-clip rung=result caption="Grey: Wicker Park streets · blue: pieces inside the 900 m by 700 m rectangle"}

Clips lines or polygons to a rectangle (turf `bboxClip`, Shapely `clip_by_rect`, PostGIS `ST_ClipByBox2D`).
Props: `geometryType` (`'lines'` with `positions` and `lineOffsets`, or `'polygons'` with the four-level
`positions`, `featureOffsets`, `polygonOffsets` and `ringOffsets`), `parameters`
(`getGPURectangleClipParameterValues({minX, minY, maxX, maxY})`) and `output: GPURectangleClipOutput`
(`positions`, `lineOffsets`, `status` with `count` (output lines, or rings for polygons), `overflow`, optional `requiredCount` (unclamped lines or rings), `vertexCount` and `requiredVertexCount`, and, for lines,
`sourceLineIndices`). Lines use Liang-Barsky, and the pieces are re-joined into lines, so a line that leaves and
re-enters becomes several lines. Polygons use Sutherland-Hodgman in four count, scan and emit stages, with
one output ring per input ring (possibly empty; `output.lineOffsets` holds the output ring offsets, and the input
`featureOffsets` and `polygonOffsets` describe them unchanged) and zero-width bridges for concave shapes (the area is
exact). The output capacity is `output.positions.length`, shared by all stages. Single-vertex lines are
dropped, degenerate bridges are not removed, and there is no spherical mode. Polygon clip points have the
bound itself in the clipped coordinate and are interpolated from the lexicographically smaller endpoint of
the edge, so two polygons that share an edge get bit-identical clip points. Line clip points are
`a + (b - a) * t` from the segment's own start and can be one float32 step off the rectangle edge.

`output.sourceRows`, `output.measures` and `output.status.approximation` are refused and typed as
optional `never`, including when callers prepare the output in a variable. Clipped
vertices can lie between input rows, and this operation has no input measure column from which
to interpolate measures. For line provenance request `sourceLineIndices`; omit vertex-row and
measure views. No approximation diagnostic is computed; inspect `overflow` for capacity loss.
The contributor-specific output type omits these three fields as well as rejecting them at runtime.

`spatialContext` defaults to native planar coordinates. A longitude-latitude context is accepted: the rectangle is in the same degrees as the input and edges are straight in longitude and latitude (turf `bboxClip`), with no antimeridian handling; a context declaring great-circle edges throws.

```ts
import type {Buffer} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  GPURectangleClip,
  getGPURectangleClipParameterValues,
  type GPURectangleClipOutput
} from '@luma-contributors/analysis/gpu-spatial-analysis';

// parameters is the four-word float32 graph view of viewportBuffer.
declare const graph: GPUCommandGraph;
declare const positions: GraphDataView<'float32x2'>;
declare const featureOffsets: GraphDataView<'uint32'>;
declare const polygonOffsets: GraphDataView<'uint32'>;
declare const ringOffsets: GraphDataView<'uint32'>;
declare const parameters: GraphDataView<'float32'>;
declare const output: GPURectangleClipOutput;
declare const viewportBuffer: Buffer;
declare const minX: number, minY: number, maxX: number, maxY: number;
graph.add(new GPURectangleClip({positions, geometryType: 'polygons', featureOffsets, polygonOffsets, ringOffsets, parameters, output}));
viewportBuffer.write(getGPURectangleClipParameterValues({minX, minY, maxX, maxY}));
```

#### How it computes

{/* lens:GPURectangleClip */}

::lens{scenario=gpu-spatial-analysis.rectangle-clip rung=datum datum=0 caption="Clipped line 0: the road it came from"}

::lens{scenario=gpu-spatial-analysis.rectangle-clip rung=kernel caption="Liang-Barsky clips each segment against the rectangle"}

::lens{scenario=gpu-spatial-analysis.rectangle-clip rung=graph caption="Count, scan and emit surviving pieces, then rejoin lines"}

{/* parameters: GPURectangleClip generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPURectangleClip` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPURectangleClip`.

- Packer: `getGPURectangleClipParameterValues(settings)`, writes `GPU_RECTANGLE_CLIP_PARAMETER_LENGTH` (4) `float32` values.

Settings of `getGPURectangleClipParameterValues`:

| Field | Type | Description |
|---|---|---|
| `minX` | `number` | Left edge. |
| `minY` | `number` | Bottom edge. |
| `maxX` | `number` | Right edge, at least `minX`. |
| `maxY` | `number` | Top edge, at least `minY`. |

{/* /parameters */}
