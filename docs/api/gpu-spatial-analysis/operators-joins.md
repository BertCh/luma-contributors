---
title: "GPU Spatial Analysis: Spatial joins"
sidebar_label: "Spatial joins"
description: "Spatial joins contributors in @luma-contributors/analysis/gpu-spatial-analysis."
---


# Spatial joins


#### Readback destinations used below

Published results must use caller-owned storage with `COPY_SRC`, or be copied into such storage
by a graph node before scratch expires. Holding a `GraphDataView` in JavaScript does not extend a
transient allocation's compiled lifetime. This local example factory allocates persistent outputs;
it is not a library export. Independent fragments below declare its signature and use this same
implementation. Read from the buffers in `publishedBuffers` after GPU completion, then destroy them
after the compiled graph is destroyed. Caller-supplied output ports below require the same storage
and lifetime, including every status word that will be read back.

```ts
import {Buffer, type Device} from '@luma.gl/core';
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';

declare const device: Device;
const publishedBuffers = new Map<string, Buffer>();
function createReadbackView<Format extends 'uint32' | 'float32' | 'float32x2'>(
  graph: GPUCommandGraph, name: string, format: Format, length: number
) {
  const rowByteLength = format === 'float32x2' ? 8 : 4;
  const buffer = device.createBuffer({
    id: name, byteLength: Math.max(1, length) * rowByteLength,
    usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
  });
  publishedBuffers.set(name, buffer);
  const handle = graph.importBuffer({id: name, byteLength: buffer.byteLength, usage: buffer.usage}, buffer);
  return graph.createDataView(handle, {format, length});
}
```

### `GPUSpatialPredicateJoin`

Graph-owned scalar diagnostics cost four bytes each. `getGPUSpatialJoinCapacityPlan` accepts
optional `graphOwnedStatusCount` (default one pair candidate-loss word); a join instance supplies
its actual count.

candidateTileCapacity bounds paged candidate scratch independently of the unchanged global
candidateCapacity. Final pairs and weights remain globally capacity bounded; inspect candidate and
output overflow separately.

Finds feature pairs that satisfy a spatial predicate such as intersects, contains or within, and counts matches per feature.

::lens{scenario=gpu-spatial-analysis.spatial-predicate-join rung=result caption="102 parks against 199 tracts; tracts shaded by the number of parks they intersect."}

Joins two sets of features by a spatial predicate, like GeoPandas `sjoin` or PostGIS
`ST_Intersects`, `ST_Contains`, `ST_Within` and `ST_DWithin`. Coordinates are planar by default; a
longitude/latitude `spatialContext` selects great-circle `dwithin` in meters (see
[Spatial context](#spatial-context-of-gpuspatialpredicatejoin)). Either side is
`{kind: 'points', positions}`, `{kind: 'lines', positions, lineOffsets}` (one linestring per
feature) or `{kind: 'polygons', positions, featureOffsets, polygonOffsets, ringOffsets}`
(multipolygons, holes as later rings, GeoArrow layout); all nine combinations are supported.

Multipart points and lines use `GPUMultipartPointGeometryPort` and
`GPUMultipartLineGeometryPort` from `@luma-contributors/analysis/analysis-core`.
Their `featureOffsets` map feature rows to point or linestring parts; a line part's
`lineOffsets` then map to vertices. Offsets start at zero, are nondecreasing and terminate at
the child count. `sourceIds` has one row per feature. Repeated IDs do not merge features.
Measures, affine transform, prepared join indices and the following joins share that hierarchy.
The single-part port types exclude `featureOffsets` to distinguish layouts. Multipart-capable
consumers explicitly accept the grouped port types; their row-sized outputs follow logical
features, including when the input port is passed through an object spread or a geometry union.

For `GPUSpatialPredicateJoin`, multipart operands support every named predicate, `relate` patterns,
`output.relate`, the relate engine and anti joins. Both sides may be multipart. The fast engine
retains its existing predicate restrictions; multipart `coveredBy` also keeps its fast route.
`leftRows` and `rightRows` always identify FEATURE rows, with one pair per matching feature pair:

- `intersects` accepts any matching part pair, including a match in a later part.
- `within` requires all usable input parts to lie in the target's closure and at least one
  interior intersection. A feature with all parts on the target boundary is not within it.
- `coveredBy` requires all usable parts to lie in the target's closure, allowing boundary-only
  contact. Containment is against the complete target feature, including the union of line parts.

Each feature pair has one DE-9IM matrix. Its exterior cells describe points outside **every** part
of the other feature; taking maxima of part-pair exterior cells gives the wrong answer. All named
topological predicates use the same matrix masks as single-part operands. `dwithin` takes the
minimum part-pair distance. Multipart polygons already use polygon `featureOffsets` and participate
in this same whole-feature classification, including holes.

Empty features never match. Empty line parts contribute no edges; short line parts with fewer
than two vertices are empty. Multipart line boundary points follow the mod-2 endpoint rule:
an endpoint shared by two open line parts belongs to the feature interior.
Line coverage merges intervals on a supplied coordinate axis, preserving distinct endpoints
even when projecting them onto a long query would round them to the same fraction.
The BVH indexes feature bounds spanning all parts, so separate parts do not occupy separate leaf
slots. Bounds may overlap across a gap, but exact refinement never draws a line across that gap.
`candidateCapacity` budgets logical feature pairs, including bounding-box false positives.
If a feature loses a candidate reached through any of its parts, its `candidateTruncated` row
and `status.candidateOverflow` are set. A short BVH `leafCapacity` also sets candidate loss.
Final pair capacity loss stays in `overflow`; increasing final capacity cannot recover candidates.

`GPUNearestFeatureJoin` supports multipart queries and targets in both nearest-feature and
neighbors modes, with minimum distance over every usable part pair. Query and target rows
are feature rows; `neighborRows` retains the target row separately from `neighborIds`.
Containment of any later part in a polygon gives distance zero. The existing `neighborCount`, tie, radius,
attribute and exclusion rules apply to feature rows. Each topology and output binding must fit
the device's storage-binding limit; sorted multipart line queries pack their topology into one
binding to stay within the eight-buffer traversal budget.
Nearest searches on CPU and GPU skip nonfinite point parts and edges with a nonfinite endpoint.
Finite vertices separated by an unusable vertex are never joined into a replacement edge. A query
with no usable parts has no neighbor: count zero, missing-row/ID sentinel and distance -1.
Nearest-feature mode accepts `nearestFeatureRows` alone when only row indices are needed.
Request `nearestFeatureIds` when application IDs are needed, and optional `nearestDistances`, one row per query
feature. It counts query features in `featureCounts`, and compacts query `sourceIds` (or query rows)
into `matches`. `nearestFeatureRows` preserves target feature rows before caller-ID
remapping; use those rows for target attributes. Neighbors-only output ports (`neighborRows`, `neighborCounts`, `neighborDistances`,
`neighborFootPoints`, `neighborQueryPoints`, `neighborSegmentIndices`) require `neighborIds`;
mixing them into nearest-feature mode is rejected at construction.
Construction captures output maps and status destinations, including inherited properties, once;
later changes to those caller objects do not redirect the join's outputs.
Spatial-frame fields (`origin`, `areaOfUse`, `unit`) are also captured once, preserving an
explicitly undefined first value from a getter.
Multipart, line and polygon operands use the feature BVH traversal with `k = 1`;
`candidateCapacity` is ignored and `candidateCount` is zero on that route. BVH leaves index whole
features, so part count cannot consume separate leaf or candidate slots. If leaf capacity loses a
feature, every query's `candidateTruncated` and `status.candidateOverflow` are set. Neighbors,
radius limits and weights count each logical feature once regardless of how many parts are close.
The same node-table and leaf limits documented below for neighbors mode apply to geometry-aware
nearest-feature mode. Neighbors-mode `status.candidateCount` is explicitly rewritten to zero on
every submission because traversal has no intermediate candidate table.
Ranking and pruning use the same normalized squared-distance significand and biased exponent
on CPU and GPU, including squares outside float32 range. Normal small distances therefore remain
distinct from an exact hit without reducing ordinary squared-distance precision. Subnormal
coordinates or distances are outside the GPU precision contract and may flush or clamp. At the
largest finite distances, the final two squared-rank bins coalesce with their predecessor and
publish the maximum float32 distance; row order breaks those ties. For normal distances or exact hits, a radius of either positive or negative zero
admits exact hits only, including great-circle searches. Endpoint distances use the endpoint
directly, and cancellation-prone segment determinants retain integer product residuals before
distance conversion. For nearly collinear point/segment pairs, the determinant fallback uses
original coordinates so subtracting a distant endpoint cannot create a false exact hit when the
line direction is reversed. The exact fallback shares the orientation kernel's 200-bit limit on
the span of nonzero product exponents; beyond that span the scaled float32 estimate is used.
Single-point nearest-feature searches against points or explicit segments use that same rank for
candidate selection, radius filtering and distance publication. Reversing a segment cannot turn
a nearby disjoint road into a radius-zero hit. Published distances include float32 square-root
rounding and need not equal an accepted radius bit for bit; radius inclusion is decided before
that conversion. Interior foot points are constructed from the
perpendicular displacement near the query, preserving small nonzero offsets from long segments;
the published foot coordinates and distance agree within float32 rounding.
Targets whose published distances round alike may still have different ranks. Finite linear
distances are supported when their squares exceed the float32 range. Combined sorted-query
topology binding sizes are checked before graph resources are declared when query sorting actually
packs metadata (at least two query features), permitting an unsorted retry after a metadata-size
rejection. Prepared consumers require identical position and
topology views, including the presence or absence of feature grouping.

`GPUPointInPolygonJoin` accepts `pointFeatureRows` alone; an ID column is optional when gathering
by row. It also accepts multipart point queries. Each point part uses `boundaryRule`
(`half-open` by default); a feature matches a polygon when any part qualifies. `pointFeatureIds`
keeps the **lowest matching polygon row**, then applies `polygons.sourceIds`, extending the
single-point overlap/shared-boundary tie rule. `polygonAssignmentCounts` counts only those
lowest-row assignments; `polygonMembershipCounts` counts a feature once in **every** matching
polygon, so its sum can exceed the number of input features. Optional
`featurePolygonMembership` publishes all memberships in a packed feature-major matrix at
`inputFeatureRow * polygonCount + polygonRow`, including single-point queries. Empty features
have no memberships and get `GPU_SPATIAL_JOIN_NO_FEATURE`. Source IDs, assignments and truncation
flags for multipart inputs use one packed row per input feature; point positions may be chunked.
Stable compact matches list each matched feature once in input row order. Optional
`pointMatchCounts` gives the number of distinct matching polygons per input feature (including
zero for empty/unmatched features). Assignment alone reduces rows directly, without a dense
matrix or a scan over polygons. Counts deduplicate sparse `(inputFeatureRow, polygonRow)` pairs
with stable integer sorts; their scratch scales with `candidateCapacity`. Only an explicitly
requested membership matrix uses `4 * inputFeatureCount * polygonCount` bytes and must fit one
device storage binding. Membership and count outputs must use separate physical buffers.
Candidate limits apply to point parts; losing candidates for any part flags its feature
and sets whole-operation `candidateOverflow`/`overflow`. Counts and memberships are partial then.

#### Native GeoArrow input

Use the public `prepareGeometryInput` adapter before uploading native XY GeoArrow arrays. It
retains feature → polygon → ring → vertex nesting: a MultiPolygon remains one feature with
multiple polygons, and later rings remain holes. Arrow record batches and slices are normalized
without losing feature identity. Pass the actual schema `field`; its extension encoding and
metadata are authoritative, and conflicting explicit options throw. Serialized WKB/WKT and
spherical edges require conversion/projection first. The adapter packs coordinates; supplying a
`frame` records a copied description of already projected coordinates, rather than projecting them.

```ts
import * as arrow from 'apache-arrow';
import {
  prepareGeometryInput, type PrepareGeometryInputOptions
} from '@luma-contributors/analysis/gpu-spatial-analysis';

const coordinateType = new arrow.FixedSizeList(2, new arrow.Field('xy', new arrow.Float64()));
const ringType = new arrow.List(new arrow.Field('vertices', coordinateType));
const polygonType = new arrow.List(new arrow.Field('rings', ringType));
const polygon = [
  [[0, 0], [4, 0], [4, 4], [0, 4], [0, 0]],
  [[1, 1], [1, 2], [2, 2], [2, 1], [1, 1]] // hole
];
const island = [[[8, 0], [9, 0], [9, 1], [8, 1], [8, 0]]];
const frame: PrepareGeometryInputOptions['frame'] = {
  id: 'local-meters', kind: 'local-azimuthal-equidistant', origin: [10, 20],
  unit: {name: 'meters', metersPerUnit: 1}
};
for (const multi of [false, true]) {
  const type = multi ? new arrow.List(new arrow.Field('polygons', polygonType)) : polygonType;
  const feature = multi ? [polygon, island] : polygon;
  const firstBatch = arrow.vectorFromArray([null, feature], type);
  const secondBatch = arrow.vectorFromArray([feature, [], null], type);
  const column = firstBatch.concat(secondBatch).slice(1, 5);
  const field = new arrow.Field('geometry', type, true, new Map([
    ['ARROW:extension:name', multi ? 'geoarrow.multipolygon' : 'geoarrow.polygon']
  ]));
  const preparedInput = prepareGeometryInput(column, {
    field, sourceRowOffset: 20, sourceIds: Uint32Array.of(99, 1, 7, 8), frame,
    precision: 'reject-loss' // default; these integer float64 coordinates are exactly representable
  });
  console.log(preparedInput.sourceRows); // [20, 21, 22, 23], original dataset rows
  console.log(preparedInput.sourceIds); // [99, 1, 7, 8], caller IDs, one per supplied slice row
  console.log(preparedInput.batchRowOffsets); // [0, 1, 4], offsets relative to this slice
  console.log(preparedInput.validity, preparedInput.empty); // [1,1,1,0], [0,0,1,0]
  console.log(preparedInput.geometry); // positions plus all three offset levels and sourceIds
  console.log(preparedInput.roundedCoordinateCount, preparedInput.collapsedPositionCount); // 0, 0
}
```

A null row has `validity = 0, empty = 0`; a nonnull empty has `validity = 1, empty = 1`.
Both retain their feature row and caller ID, but neither matches a polygon. `sourceRowOffset` is
the original dataset row of the supplied slice's first row; it does not slice `sourceIds` for you.
Without supplied IDs, IDs default to those source rows. Join row outputs still index the supplied
feature domain, so map them through `sourceRows` or `sourceIds` explicitly.

Float64 conversion defaults to `precision: 'reject-loss'`. For example, distinct X coordinates
`2**30` and `2**30 + 1` collapse to the same float32 X; reject-loss throws. Use local coordinates
to preserve the distinction, or explicitly choose `precision: 'allow-loss'` and retain
`roundedCoordinateCount` (rounded scalar coordinates) and `collapsedPositionCount` (distinct
original XY positions beyond one per rounded XY, excluding repeated copies of the same vertex).
For Polygon/MultiPolygon, both precision policies reject nonfinite XY and finite coordinates
that overflow float32. A native Point `[NaN, NaN]` is a valid empty point under either policy,
returning `validity = [1]` and `empty = [1]`; it is distinct from a null row. Other nonfinite Point
coordinates, including a NaN paired with a finite value, are rejected.

#### CPU assignment and counts without dense membership

Use `computePointInPolygonCountsOnCPU` when assignments, compact IDs and counts are sufficient.
It returns `CPUPointInPolygonCountsResult` without allocating the query-by-polygon membership
matrix. The following point is strictly inside both rectangles because `1 < 1.5 < 2` and
`0 < 1 < 2`; the lowest polygon row wins assignment, while both distinct polygon memberships count.

```ts
import {
  computePointInPolygonCountsOnCPU, type CPUPointInPolygonCountsResult
} from '@luma-contributors/analysis/gpu-spatial-analysis';

const counts: CPUPointInPolygonCountsResult = computePointInPolygonCountsOnCPU({
  points: {kind: 'points', positions: Float32Array.of(1.5, 1), sourceIds: Uint32Array.of(77)},
  polygons: {
    kind: 'polygons',
    positions: Float32Array.of(0,0,2,0,2,2,0,2, 1,0,3,0,3,2,1,2),
    featureOffsets: Uint32Array.of(0,1,2), polygonOffsets: Uint32Array.of(0,1,2),
    ringOffsets: Uint32Array.of(0,4,8), sourceIds: Uint32Array.of(900,7)
  }
}, {candidateCapacity: 2, matchCapacity: 1, boundaryRule: 'open'});
for (const word of [counts.candidateOverflow, counts.matchOverflow, counts.uncertainCount]) {
  if (word.length !== 1 || word[0] !== 0) throw new Error('Incomplete membership; retry');
}
console.log(counts.pointFeatureRows, counts.pointFeatureIds); // [0], [900]: target row versus ID
console.log(counts.ids.slice(0, counts.count[0])); // [77]: compact query IDs, count 1
console.log(counts.pointMatchCounts); // [2]: distinct matching polygons per query feature
console.log(counts.polygonAssignmentCounts); // [1,0]: one lowest-row assignment per query
console.log(counts.polygonMembershipCounts); // [1,1]: each matching query feature once per polygon
```

Query parts, repeated vertices and repeated caller IDs do not merge feature rows. Membership
counts deduplicate part hits within each `(queryFeatureRow, polygonRow)` pair; two polygon rows
with equal caller IDs still count separately. `pointFeatureRows` is for gathering polygon
attributes; `pointFeatureIds` is for caller identity. `count` bounds the readable compact `ids`
prefix, rather than the number of membership pairs. Per-feature rows/counts are source-aligned.

`computePointInPolygonJoinOnCPU` provides dense feature membership, lowest-row assignment,
explicit `polygonAssignmentCounts` and `polygonMembershipCounts`, compact identities and candidate
diagnostics for packed planar arrays. Its dense CPU membership matrix costs
`4 × pointFeatureCount × polygonFeatureCount` bytes even when no pair matches. Choose this dense
call only when the individual membership cells are needed, at
`queryFeatureRow * polygonFeatureCount + polygonRow`. GPU dense output is
opt-in through `output.featurePolygonMembership`; assignment and counts do not require that matrix.
All three boundary
rules use exact binary64 integer orientation signs; the half-open rule uses symbolic infinitesimal
ownership without adding an epsilon to coordinates. Unsorted part-then-polygon candidate order
binds the explicit candidate budget. CPU predicates have no orientation dynamic-range uncertainty,
but nonfinite polygon candidates increment `uncertainCount` and do not match; results for valid
polygons survive. Malformed offset hierarchies throw. Compact tails are zero and
only the count prefix is readable. Internal `createPointInPolygonJoinOperation` and
`createCPUPointInPolygonJoinOperation` share a packed planar, unsorted/unprepared contract.

`GPUBufferSelection` accepts multipart point queries with packed positions and packed per-feature
outputs/source IDs. A feature selects when **any** point part is within the inclusive radius.
Distance is the minimum over all parts and nearest-feature ties keep the lowest reference row;
empty features have mask 0, distance -1 and no nearest row. Compaction retains feature row order.
This route uses the feature-level nearest traversal: `candidateCapacity` is validated but unused,
while BVH `leafCapacity` loss propagates through `candidateTruncated` and candidate/operation status.
Single-point inputs retain their existing topology, counting, capacity behavior and command graph.

`computeBufferSelectionOnCPU` takes packed typed-array points (including `featureOffsets`),
point or explicit segment references, a distance and the same candidate/leaf/compact budgets.
It returns the mask, reference rows, distances, compact IDs/counts, draw count and completeness
flags. Exact collinear segment hits retain distance zero, including radius zero. It uses float64
internally and rounds distance output once; last-ulp radius decisions may
differ from GPU float32. Invalid offset hierarchies throw; unused compact slots are zero and only
the `count` prefix is meaningful. Internal `createBufferSelectionOperation` and
`createCPUBufferSelectionOperation` factories in their adjacent operation files share one packed
planar contract and disable spatial/query sorting to make budget prefixes reproducible. The
contributor remains the route for great-circle contexts and chunked single-point inputs.

Pure `computeSpatialPredicateJoinOnCPU` and `computeNearestFeatureJoinOnCPU` take packed typed
arrays with the same hierarchy. The planar CPU predicate supports all named predicates and relate
patterns, including polygon/polygon containment, and takes explicit `candidateCapacity` and
`pairCapacity` limits. Set `relate: true` to materialize packed matrices aligned with matched
pairs; otherwise the returned `relate` array is empty. CPU `intersects`, distance and anti-intersects
joins retain short-circuit evaluation when no matrix is requested. CPU
topology uses exact rational intersections of the float32 vertices and has no orientation
uncertainty; distance thresholds compare squared endpoint/perpendicular distances, with exact
dyadic comparisons when determinant cancellation or threshold rounding makes the fast comparison
inconclusive (an exact threshold is inclusive). `how: 'anti'` writes unmatched left rows to `leftRows`
and leaves `rightRows` at `0xffffffff`; candidate-truncated rows are never definite non-matches.
The graph-operation forms accept the same predicate, pattern, distance, engine and how options;
their inner outputs are `leftRows`, `rightRows`, optional `relate` and optional `unmatchedRows`.
Inner `unmatchedRows` requires independent `unmatchedCount` and `unmatchedOverflow` destinations;
optional `unmatchedRequiredCount` and `unmatchedCandidateOverflow` describe demand and candidate
loss. The ordinary `count`, `requiredCount`, `overflow` and `candidateOverflow` describe inner
pairs. The anti output uses `unmatchedRows` with the ordinary `count`, `requiredCount`, `overflow`
and `candidateOverflow` ports, without the separate inner-only unmatched diagnostics.
Their optional `uncertainCount` destination reports provisional GPU classifications whose
orientation cannot be certified. The exact CPU operation writes zero. An uncertain GPU pair is
not a certified topological match even when capacity flags are clear. Optional destinations with
value `undefined` behave as omitted fields, including inherited getters; required names remain
validated.
Both operation factories reject `output.relate` for `dwithin` at construction, and their planar
distance limit is the largest finite float32 value, `3.4028234663852886e38`. CPU input arrays
reject non-finite coordinates. GPU predicates suppress the entire feature if any coordinate is
non-finite, independent of part order, including prepared refinement.
CPU predicate candidates use a sorted bounds index and obey the explicit candidate budget;
`requiredCount` is a lower bound after candidate loss. CPU nearest selects one unbounded planar
neighbor per query feature, breaks distance ties by lowest target row, uses a sorted bounds index
with range pruning and emits float32 distances. Reversed copies of the same segment remain tied
and select the lowest row and its caller ID. Empty queries produce empty per-query columns;
queries with no targets produce count 0, row/ID `0xffffffff` and distance -1.
The CPU predicate result's `candidateBoundsTestCount` counts index bounds tested, separately
from result status; entire ranges ending before or starting after a query are pruned together.

`predicate` is `'intersects'`, `'contains'`, `'within'`, `'dwithin'` (with an inclusive
`distance`: planar in coordinate units, or meters with a great-circle `spatialContext`), `'covers'`, `'coveredBy'`, `'touches'`, `'crosses'`, `'overlaps'`, `'equals'`,
`'containsProperly'` or `'relate'` (with a DE-9IM `pattern`). Boundaries follow OGC / DE-9IM: a point has no boundary, a linestring's boundary is its
endpoints unless it is closed, and a polygon's boundary is its rings.

- `intersects`: the closed geometries share a point, so polygons touching at an edge or corner
  intersect, and so does a point on a polygon boundary.
- `contains(a, b)`: `b` lies in the closure of `a` and the interiors intersect. A polygon does not
  contain a point or line on its boundary, and a line does not contain its own endpoints.
- `within(a, b)` is `contains(b, a)`.
- `dwithin`: the minimum distance between the closed geometries is at most `distance`.
- `covers` and `coveredBy`: `contains` and `within` that allow boundary contact.
- `touches`: the geometries intersect and their interiors do not intersect (never true for two
  points). A point on a polygon's hole edge touches the polygon even though the point has no boundary.
- `crosses`: the interiors meet and the intersection has a lower dimension than the larger input, by the
  OGC dimension rules (false for polygon/polygon and point/point).
- `overlaps`: the same dimension, the interiors meet and neither contains the other.
- `equals`: the same point set. `containsProperly`: `b` lies in the interior of `a`.
- `relate` with `pattern`: any DE-9IM pattern (characters `T F * 0 1 2`), as a string or an any-of list.

The pipeline is upstream primitives end to end: per-feature bounds into a `GPUBVH` over the right
side, a count pass, `GPUScan` and an ordered write of candidate pairs, the exact predicate per
candidate (single-point/polygon pairs use `GPUPairwisePointInPolygon`; points on an edge it reports as
uncertain are decided by the f32 kernel and counted in `uncertainCount`), then a scan and a
capacity-bounded scatter. Outputs go in `output`; `output.pairs` (a `GPUCompactPairPort`: `leftRows`,
`rightRows`, `status` with `count`, `overflow`, optional `requiredCount`) are unique and sorted by `(left, right)`, and
scalars go in `output.status` (`overflow`, `candidateCount`, `uncertainCount`). The sides are geometry
ports (`GPUPackedGeometryPort` of the point, line or four-level polygon port). On overflow the output is a
sorted subset; `join.getCapacityPlan({candidateCount, requiredCount})` turns the read-back counts into the
next capacities (see [Shared ports, status and planning](./concepts.md#shared-ports-status-and-planning)).
`candidateCapacity` defaults to that plan's estimate, `min(rightCount, 8)` per left feature.
The instance uses the same multipart-aware classifier choice for page sizing, transient bytes
and execution. A multipart point/polygon matrix candidate uses 256 lanes; it does not allocate
the singleton point/polygon classifier's scratch. Standalone planning accepts
`robustPointPolygon: false` for that route; the instance supplies this choice automatically.

Pass `output.weights` to also write the matches as a `GPUSpatialWeights` with left features as rows and
weight 1; set `excludeSameRow` for a self join. A predicate join is then a weights producer, for
example `dwithin` between polygons. Optional `output.weights.distances` contains the minimum
closed-geometry distance aligned with `neighbors`: planar coordinate units, or meters for
great-circle `dwithin`. Intersecting or contained geometries have distance zero; holes contribute
their boundaries. Planar distance output uses scaled norms, so representable distances such as
`1e20` do not saturate when their square exceeds float32. Normal gaps beside long, asymmetric
segments remain measurable even when a projection fraction would underflow. Large `dwithin`
thresholds use the same linear distance in both engines; ordinary thresholds retain squared
comparisons. Unused distance slots are NaN on every encoding, including empty results.
Distance output requires a metric (`native` or `great-circle`); longitude/latitude topology with
`metric: 'none'` cannot produce distances. The optional `weights.status` fields `count`,
`requiredCount`, `overflow`, `candidateOverflow` and `uncertainCount` are written even when no
pair output is requested. `overflow` reports final pair capacity, and `candidateOverflow` reports
an incomplete candidate or BVH stage. Distance evaluation adds one packed pair scratch buffer
(eight bytes per output slot), included by the instance's `getCapacityPlan()`.
These five fields are the complete weights-status contract. `invalidCount`, `iterationCount`,
`converged`, `iterationLimitReached` and `approximation` are rejected at construction because the
join does not publish those diagnostics. Omit them and use the supported count, capacity-loss and
classification-uncertainty fields instead. The same rule applies to every predicate engine.
Reusing a join-status map does not add CSR outputs: `candidateCount`, `candidateTruncatedCount`,
`antimeridianUncertainCount` and `poleUncertainCount` are also rejected in `weights.status`.
These refusals apply to captured bounded-result and join diagnostics. Unrelated geometry metadata,
such as `vertexCount`, is omitted from the captured status and is not an output destination.

Declared status destinations, including inherited and non-enumerable `requiredCount` getters,
are captured at construction and retained when implicit completeness destinations are added.
The pair-gather example preserves the caller's pair requirement destination in the same way.
Geometry descriptors, options and the nested pair, unmatched and weights maps are captured too.
Replacing `unmatched.ids` after construction cannot bypass the original output-alias checks.
The predicate operation requires named destinations for `count`, `requiredCount`, `overflow` and
`candidateOverflow`; absent and explicitly undefined required names both fail construction.
Optional destinations whose captured value is undefined are omitted. Choropleth recipes retain
inherited grouping and ring offsets on both operands. Passing one spatial context object for
both nearest operands reuses one snapshot, including the first values of frame getters.

For downstream statistics, pass `join.outputWeights`, which always includes both overflow and
candidateOverflow. Missing caller scalar views are allocated in the graph and written on every
submission. The caller's `output.weights` is an allocation description; rebuilding an input from
only that description can omit completeness signals.

The new predicates are masks over one DE-9IM matrix per candidate pair, computed from structure (vertex
locations, edge crossings, collinear overlaps, and edge pieces located against the other geometry, with
polygon areas decided from edge pieces) for all nine point, line and polygon combinations. Semantics
follow GEOS and Shapely. `intersects`, `contains`, `within` and `dwithin` keep their short-circuiting
kernels; the topological predicates can also use the matrix, and the specs assert both paths agree.
`output.relate` requires a topological predicate: `dwithin` classifies metric proximity, so request
DE-9IM output from a separate relate join when both results are needed.
Only bounding-box candidates are enumerated, so a pattern that admits disjoint geometries (all of II, IB,
BI and BB allow `F`) is rejected at construction. Use `how: 'anti'` with `intersects` to select left
features disjoint from **every** right feature. This does not enumerate disjoint pairs: left point
0 against right points 0 and 1 has no anti row, although the pair (0, 1) is disjoint.
Empty features produce zero join pairs, including an `equals` join of two empty features. This
candidate-based join is different from aligned topological equality, where two empty geometries
can be equal.
Lines may group several linestring parts, with the OGC mod-2 feature boundary. Union coverage
merges supplied-coordinate intervals, rescanning target edges until the covered interval stops
growing. The matrix kernel suits small features; adversarial line-part ordering can require many
coverage passes, and large polygon pairs remain expensive.

Orientation tests are exact. Edge pieces between crossings are located from an exactly located endpoint
and the exact crossings between it and the piece, not from their rounded midpoint, so edges within a few
ulps of each other (near-parallel or nearly shared) classify like Shapely. Distinct polygon crossings
that tie as float32 parameters use exact projective comparisons and symbolic locations on both
sides; narrow hole intervals are retained. Shared polygon boundaries use symbolic normal probes,
so equal polygons one coordinate ulp wide retain their interior/interior cell after translation.
Planar proximity tests include coordinate-subtraction error in their fast comparison. Ambiguous
positive-radius comparisons use exact integer dot products and original-coordinate determinants,
so inclusive radius boundaries and endpoint classification survive segment reversal. Radius zero
uses the intersection classifier, including its uncertainty diagnostic. Published nonzero GPU
distances retain float32 precision. Polygons must be valid.

```ts
import {type GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  GPUSpatialPredicateJoin,
  type GPUSpatialPredicateJoinProps
} from '@luma-contributors/analysis/gpu-spatial-analysis';

declare const graph: GPUCommandGraph;
// Local factory from the readback-destinations example above, not a library export.
declare const createReadbackView: <Format extends 'uint32' | 'float32' | 'float32x2'>(
  graph: GPUCommandGraph, name: string, format: Format, length: number
) => GraphDataView<Format>;

declare const polygons: GPUSpatialPredicateJoinProps['left'];
declare const points: GPUSpatialPredicateJoinProps['right'];
declare const pairs: NonNullable<GPUSpatialPredicateJoinProps['output']['pairs']>;
declare const weights: NonNullable<GPUSpatialPredicateJoinProps['output']['weights']>;
const candidateOverflow = createReadbackView(graph, 'strict-candidate-loss', 'uint32', 1);
const pairCandidateOverflow = createReadbackView(graph, 'strict-pair-candidate-loss', 'uint32', 1);
const uncertainCount = createReadbackView(graph, 'strict-uncertain', 'uint32', 1);
const overflow = createReadbackView(graph, 'strict-overflow', 'uint32', 1);
const join = new GPUSpatialPredicateJoin({
  left: polygons, right: points, predicate: 'contains', how: 'inner',
  candidateCapacity: 1 << 20,
  output: {
    pairs: {...pairs, status: {...pairs.status, candidateOverflow: pairCandidateOverflow}},
    weights: {...weights, status: {...weights.status, overflow, candidateOverflow}},
    status: {uncertainCount}
  }
});
graph.add(join);
// After submission, read join.outputWeights.offsets and its status, plus uncertainCount.
declare const readback: {
  offsets: Uint32Array;
  overflow: Uint32Array;
  candidateOverflow: Uint32Array;
  uncertainCount: Uint32Array;
};
// The polygon count is the number of logical left features, not their allocation capacity.
declare const polygonCount: number;
function decodeStrictPolygonCounts(
  result: typeof readback, polygonCount: number
): number[] {
  if (!Number.isSafeInteger(polygonCount) || polygonCount < 0) {
    throw new Error('polygonCount must be a nonnegative safe integer');
  }
  for (const word of [result.overflow, result.candidateOverflow, result.uncertainCount]) {
    if (word.length !== 1 || word[0] !== 0) {
      throw new Error('Strict polygon counts need exactly one zero word per diagnostic');
    }
  }
  if (result.offsets.length !== polygonCount + 1 || result.offsets[0] !== 0) {
    throw new Error('Strict polygon counts need exactly polygonCount + 1 CSR offsets starting at zero');
  }
  return Array.from({length: polygonCount}, (_, row) => {
    if (result.offsets[row + 1] < result.offsets[row]) {
      throw new Error('Strict polygon count offsets must be nondecreasing');
    }
    return result.offsets[row + 1] - result.offsets[row];
  });
}
const polygonStrictCounts = decodeStrictPolygonCounts(readback, polygonCount);

```

Here polygons are the left rows, so `offsets[row + 1] - offsets[row]` counts **every strict**
point match for that polygon, including overlaps. For rectangles `[0,0]–[2,2]` and `[1,0]–[3,2]`
and point `(1.5,1)`, the result is `[1,1]`. Exclusive PIP assignment instead gives
`polygonAssignmentCounts = [1,0]`; PIP `polygonMembershipCounts` with `boundaryRule: 'open'`
gives `[1,1]`. Moving the point to `(0,1)` gives strict counts `[0,0]`, while the PIP half-open
default includes that outer left edge and gives assignment counts `[1,0]`.

Final pair/CSR capacity and intermediate candidate capacity are independent. When requesting
both pairs and weights, their pair capacities must agree. Read only `pairs.status.count` pair slots. To certify
pair cardinality, check pair final `overflow`, `candidateOverflow` and `uncertainCount` before
printing it; for counts from weights, check `join.outputWeights.status` as above. In the overlap
fixture with pair capacity 2 and candidate capacity 1, `count = requiredCount = 1` and final
pair `overflow = 0`, but `candidateOverflow = 1`. Increasing only candidate capacity to 2 restores
both matches. Neither `count === requiredCount` nor a clear final-capacity word certifies completeness.

**`relate` output.** `output.relate` is an optional `GraphDataView<'uint32'>` as long as `pairs.leftRows`; slot `k`
holds the matrix of `(leftRows[k], rightRows[k])`. Each of the nine cells (order II IB IE BI BB BE EI EB EE,
row-major, left parts then right parts) takes 2 bits at `[2k, 2k + 1]` with value `dimension + 1` (0
empty, 1 point, 2 line, 3 area). `formatGPUSpatialRelate(value)` decodes it on the CPU to a string such as
`'FF2F11212'`. Bit 31 (`GPU_SPATIAL_RELATE_UNCERTAIN_BIT`) of a slot is set when an orientation of that pair
had no sign (see **Exactness**); `formatGPUSpatialRelate` ignores it, and `value & 0x3ffff` is the matrix alone.
It is not available for `dwithin`.

```ts
import {type GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  GPUSpatialPredicateJoin,
  type GPUSpatialPredicateJoinProps
} from '@luma-contributors/analysis/gpu-spatial-analysis';

declare const graph: GPUCommandGraph;
// Local factory from the readback-destinations example above, not a library export.
declare const createReadbackView: <Format extends 'uint32' | 'float32' | 'float32x2'>(
  graph: GPUCommandGraph, name: string, format: Format, length: number
) => GraphDataView<Format>;

declare const left: GPUSpatialPredicateJoinProps['left'];
declare const right: GPUSpatialPredicateJoinProps['right'];
declare const pairs: NonNullable<GPUSpatialPredicateJoinProps['output']['pairs']>;
declare const relate: NonNullable<GPUSpatialPredicateJoinProps['output']['relate']>;
const candidateOverflow = createReadbackView(graph, 'touch-candidate-loss', 'uint32', 1);
const uncertainCount = createReadbackView(graph, 'touch-uncertain', 'uint32', 1);
graph.add(new GPUSpatialPredicateJoin({
  left, right, predicate: 'touches', how: 'inner', candidateCapacity: 1 << 16,
  output: {
    pairs: {...pairs, status: {...pairs.status, candidateOverflow}},
    relate, status: {uncertainCount}
  }
}));
// Read only pairs.status.count slots; require pairs.status.overflow,
// candidateOverflow and uncertainCount all to be zero before treating them as a complete join.
```

**Unmatched rows and anti joins.** An inner join can request `output.unmatched` alongside pairs
or weights, so one predicate classification can feed a left attribute gather. Unmatched rows are
derived from per-left match counts before final pair truncation: a missing pair due to final output
capacity does not make its left feature unmatched. Rows with tracked per-row candidate loss
are excluded. Any nonzero `uncertainCount` suppresses the entire unmatched prefix, including
plainly disjoint rows: this pass has no per-row uncertainty attribution. Check `uncertainCount`
before interpreting zero unmatched rows as a complete result, even when all capacity flags are
clear. `how: 'anti'` with `output.unmatched: GPUCompactOutput` emits only the left rows that
have no match, in ascending order (`pairs`, `weights` and `relate` are not allowed with it). When the candidate capacity overflows,
the left rows whose candidates were dropped cannot be classified (they may have a match among the dropped
candidates), so they are kept out of `output.unmatched` and counted in `output.status.candidateTruncatedCount`;
the anti output withholds rows with known candidate loss and may omit true unmatched rows among the
truncated ones. This does not certify classifications with nonzero uncertainty. (Before this change those rows were emitted as unmatched.) A BVH overflow (more right features
than `leafCapacity`) drops right features that any row may match, so since 2026-11-10 it marks every left row and
the anti output is empty; before, it listed rows whose only matches were among the dropped features.

Supply `output.unmatched.status.candidateOverflow` as a one-row `uint32` view for an anti-only
join unless all three conditions hold. `candidateCapacity` must cover the product of the allocated
left and right feature counts, the resolved `leafCapacity` must cover the allocated right feature
count, and the left input must have neither a live count nor completeness views. This bound counts
one candidate per possible feature pair across all scratch pages. The flag is `1` when candidates
were lost, the BVH omitted right features, or the live left input reported loss. The unmatched
`status.overflow` covers final output capacity. Check both flags before treating the unmatched
prefix as complete.

Generated geometry can have more allocated feature slots than live features. Supply its resident
`status.count` as `leftFeatureCount`, and membership completeness as `leftStatus`; never infer the
live count from `featureOffsets.length - 1`. Only the live left prefix participates, including real
empty features inside it. Upstream loss becomes candidate loss and prevents claims of definite
absence. Buffer contents, including the live count, are reread on reuse.

```ts
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  GPUSpatialPredicateJoin, type GPUSpatialPredicateJoinProps
} from '@luma-contributors/analysis/gpu-spatial-analysis';

declare const graph: GPUCommandGraph;
declare const overlayGeometry: GPUSpatialPredicateJoinProps['left'];
declare const liveFeatureCount: GraphDataView<'uint32'>; // producer status.count, one scalar
declare const overlayStatus: NonNullable<GPUSpatialPredicateJoinProps['leftStatus']>;
declare const right: GPUSpatialPredicateJoinProps['right'];
declare const unmatched: NonNullable<GPUSpatialPredicateJoinProps['output']['unmatched']>;
// Supply caller-owned readback views for both diagnostics.
declare const candidateOverflow: GraphDataView<'uint32'>;
declare const uncertainCount: GraphDataView<'uint32'>;
graph.add(new GPUSpatialPredicateJoin({
  left: overlayGeometry, leftFeatureCount: liveFeatureCount, leftStatus: overlayStatus,
  right, predicate: 'intersects', how: 'anti', candidateCapacity: 256,
  output: {
    unmatched: {...unmatched, status: {...unmatched.status, candidateOverflow}},
    status: {uncertainCount}
  }
}));
// Claim definite absence only after reading one zero word from candidateOverflow and uncertainCount,
// as well as a clear unmatched.status.overflow; otherwise withhold the unmatched result.
```

For CPU readback, use the public `decodeGPUSpatialJoinGeometry` with the producer's live count.
It follows **all** terminal offsets down through polygons/rings (or line/point parts), retaining
real empty features inside that count and removing allocated tails. It accepts the element-typed
arrays below, `Uint8Array` buffer reads (respecting byte offsets) or `ArrayBuffer`; malformed or
short layouts throw. It decodes geometry only: carry producer loss separately.

A resident live-left polygon prefix may retain unused allocated vertex and ring tails.
The GPU join uses its feature-bounded exact predicate/relate kernels for that route, rather
than the compact-global pairwise classifier; this avoids false uncertainty from unused
terminal storage. The CPU form continues to decode the complete live hierarchy.

```ts
import {
  decodeGPUSpatialJoinGeometry, computeSpatialPredicateJoinOnCPU
} from '@luma-contributors/analysis/gpu-spatial-analysis';

// Readback fixture: capacity three, two live features: a square and a genuine empty.
const liveFeatureCount = Uint32Array.of(2);
const producerStatus = {overflow: Uint32Array.of(0), candidateOverflow: Uint32Array.of(0)};
const liveGeometry = decodeGPUSpatialJoinGeometry({
  kind: 'polygons',
  positions: Float32Array.of(0,0,2,0,2,2,0,2, 99,99,99,99), // unused vertex tail
  featureOffsets: Uint32Array.of(0,1,1,1),
  polygonOffsets: Uint32Array.of(0,1,1), ringOffsets: Uint32Array.of(0,4,4),
  sourceIds: Uint32Array.of(900,7,123)
}, liveFeatureCount);
console.log(liveGeometry.featureOffsets); // [0,1,1]: row 1 is live and empty; row 2 is padding
console.log(liveGeometry.polygonOffsets, liveGeometry.ringOffsets); // [0,1], [0,4]
console.log(liveGeometry.positions); // [0,0,2,0,2,2,0,2]
console.log(liveGeometry.sourceIds); // [900,7], caller IDs rather than feature rows
const decoded = {geometry: liveGeometry, producerStatus};
for (const word of [decoded.producerStatus.overflow, decoded.producerStatus.candidateOverflow]) {
  if (word.length !== 1 || word[0] !== 0) throw new Error('Decoded geometry is incomplete; retry');
}
// A clear downstream join status alone cannot certify an incomplete producer's geometry.
const joined = computeSpatialPredicateJoinOnCPU(liveGeometry, {
  kind: 'points', positions: Float32Array.of(1,1)
}, {
  predicate: 'intersects', how: 'inner', candidateCapacity: 2, pairCapacity: 2,
  leftStatus: {
    overflow: decoded.producerStatus.overflow[0],
    candidateOverflow: decoded.producerStatus.candidateOverflow[0]
  }
});
if (joined.status.overflow || joined.status.candidateOverflow ||
    joined.unmatchedStatus.overflow || joined.unmatchedStatus.candidateOverflow) {
  throw new Error('Incomplete pairs or unmatched rows; retry');
}
console.log(joined.leftRows.slice(0, joined.status.count)); // [0]: the square intersects
console.log(joined.unmatchedRows.slice(0, joined.unmatchedStatus.count)); // [1]: live empty row
// Row 1's caller ID is 7; padding row 2 never participates.
```

Trimming only feature offsets leaves padded child arrays; treating repeated offsets beyond the
live count as features creates invented unmatched rows. A real empty within the live prefix is
instead a definite non-match when producer completeness and classification certainty are clear.
`candidateTruncated` tails are zero outside `leftFeatureCount`, and `candidateTruncatedCount`
counts live rows only. Left geographic diagnostics stop at the terminal live ring. An empty live
prefix may still carry aggregate producer loss with no truncated rows. Any nonzero
`uncertainCount` conservatively withholds unmatched output; require zero to certify absence.

Both named predicate operation forms accept `leftFeatureCount: 'overlayCount'` and
`leftStatus: {overflow: 'overlayOverflow', candidateOverflow: 'overlayCandidateOverflow',
complete: 'overlayComplete'}`. Each supplied name identifies one `uint32` scalar; omitted loss
fields remain optional. Supply the producer's names rather than deriving a count from allocation.
The CPU form follows the terminal offsets exactly as the decoder does above, preserving live
empties; the GPU form reads the resident count each encoding.

For `how: 'inner'`, named outputs may include both `leftRows`/`rightRows` and `unmatchedRows`.
Request separate `unmatchedCount` and `unmatchedOverflow` when requesting `unmatchedRows`;
`unmatchedRequiredCount` and `unmatchedCandidateOverflow` are optional additional diagnostics.
`pairCapacity` allocates each compact rows column, including unmatched rows. Read only the
`unmatchedCount` prefix of `unmatchedRows`; these are left feature rows, never caller IDs. The
ordinary `count`/`requiredCount`/`overflow` describe pairs on this route. Absence is determined
before final pair truncation, so a matched row whose pair did not fit never becomes unmatched.
Producer/candidate loss or classification uncertainty withholds unknown absence claims. For
`how: 'anti'`, `count`, `requiredCount`, `overflow` and `candidateOverflow` describe `unmatchedRows`.
Both named operation forms require a graph value name in `output.candidateOverflow`. The direct
CPU twin always returns the flag in `unmatchedStatus.candidateOverflow` and, in anti mode, in
`status.candidateOverflow` as well.

**Candidate overflow.** Candidates are written in left-row order, so the slots past `candidateCapacity` are the
tail of that layout: the row whose candidates cross the capacity keeps its first ones, and every later row that
had a candidate keeps none; earlier rows are complete. A BVH leaf overflow (more right features than
`leafCapacity`, in the join's own build or in a `GPUSpatialJoinPrepared`) drops features that any row may match,
so it marks every row. `output.status.candidateOverflow` is 1 when either happened (`output.status.overflow` keeps
its combined meaning), and the optional `output.candidateTruncated` (one `uint32` per left feature) is 1 for exactly
the rows described above. `output.status.candidateTruncatedCount` counts them. `GPUPointInPolygonJoin` and every
mode of `GPUNearestFeatureJoin` (nearest-feature, planar or great-circle, and neighbors) take the same
`output.candidateTruncated` and `output.status.candidateOverflow`, one flag per query; a flagged query may show no
feature, or a farther one. Neighbors mode and the great-circle search have no candidate table, so a BVH overflow is
their only cause. `GPUSpatialJoinPrepared` holds only the right-hand BVH and has no candidate stage. The join
consumers pass the flags through: `GPUZonalStatistics` (`output.candidateTruncated` per point and
`output.status.candidateOverflow`), `GPUBufferSelection` (the same pair) and `GPULineLengthPerPolygon`
(`output.status.candidateOverflow` only; its output is per polygon).

**Exactness.** Every orientation sign is exact, in the short-circuiting kernels (`engine: 'fast'`) and in
the relate engine alike: an f32 filter decides the clear cases and the rest fall back to exact integer
arithmetic (256-bit), with one inlined copy per kernel. So touches, crosses, covers, `intersects`,
`contains` and the other predicates do not depend on f32 rounding of the orientation test, and the result
does not depend on the engine or on the feature size that `engine: 'auto'` looks at, including
shared-edge and collinear configurations. Polygon `contains` / `within` in the fast kernel decides on
which side of a shared boundary run the two interiors lie with a symbolic probe (a point just beyond an
outer vertex and just off its edge, never formed in f32), so an inner polygon that fills a hole of the
outer one, or has a hole of the outer one inscribed in it, is not contained for any vertex coordinates.
The exactness is with respect to the f32 vertices; edge piece
midpoints, break parameters and distances are computed in f32: the location of a piece shorter than about
one ulp is not exact, so a line starting one ulp outside a polygon edge can have its `crosses` piece
classified as boundary.

**Uncertain pairs.** A pair is uncertain when an orientation has no sign: a non-finite coordinate, or
product exponents spanning more than 200 bits. It is never dropped silently. The undecidable orientation is
treated as collinear (sign 0) and the predicate is evaluated with that value, so the pair can come out
wrong in either direction, and it is counted in `uncertainCount` (zero means every decision is certified).
Matched uncertain pairs are identifiable per pair through bit 31 of the `relate` output. Point/polygon
pairs use the robust double-single classifier (they do not get the exact re-decision of
`GPUPointInPolygonJoin`); pairs it cannot certify are decided by the f32 kernel and counted.
Scalar exact kernels with eight bindings carry uncertainty in candidate flags and count it in a
later pass, so adding an identity multipart grouping preserves the diagnostic. Workgroup `dwithin`
merges lane uncertainty into the same flags; each affected candidate pair contributes once,
regardless of how many lanes encounter an undecidable orientation.

**Per-frame distance and pattern.** `distance` is a `number` or a `GraphDataView<'float32'>`, and `pattern`
is a `GPUSpatialRelatePattern` or a `GraphDataView<'uint32'>`. A view is read from a parameter buffer, so
it changes without recompiling and gives bit-identical results to the compile-time value (a negative, NaN
or infinite distance selects nothing). Fill a pattern view with `packGPUSpatialRelatePattern(pattern,
slotCount)`. Its host `slotCount` must be a positive integer no greater than
`MAXIMUM_SPATIAL_RELATE_PATTERN_SLOT_COUNT = 2,147,483,647`; each slot uses
`GPU_SPATIAL_RELATE_PATTERN_WORDS = 2` `uint32` words (8 bytes). This host representability
limit is independent of upload capacity: the entire pattern view must also fit the device's
storage-binding and buffer-size limits when uploaded/bound. Unused slots are zero and match
nothing, and the helper rejects patterns that admit disjoint geometries. Size `candidateCapacity`
for the largest distance. The exact kernels dispatch by the live candidate count, but the cooperative
workgroup kernels (`dwithin` with `engine: 'relate'` and the relate engine, 64 and 256 lanes per
candidate) index their lanes with a u32, so the constructor rejects a `candidateCapacity` above
`floor((2^32 - 1) / lanes)` for them, naming the prop. `predicate` stays compile-time because it selects the kernel family; for a
per-frame predicate choice use `predicate: 'relate'` with patterns.

**Relate performance.** Per-edge break parameters are collected in one pass (up to 24 per edge) and the
covers scan runs only when a collinear edge exists. The workgroup relate kernel also builds a per-pair
y-slab index of each polygon side in workgroup memory (32 slabs, 1280 entries per side), so point location
scans the closing edges and the query's slab instead of every edge; features under 48 edges, with a
degenerate y extent or over 1280 entries use the plain scan. Results are identical. On 100 polygon pairs
the batched breaks and the slab index took relate from about 7.5 ms to 4.3 to 5 ms, against about 1 ms for
GEOS: the break collection, covers scan and vertex loops are still `O(n * m)`, and about 2 ms of the
remaining time is fixed graph and readback cost.

**`onAttribute: {left, right}`** (GeoPandas `sjoin(on_attribute=)`). Two `uint32` key views, one per left
row and one per right row. A pair survives only when the keys are equal. The filter empties candidate
slots before the exact kernels, so `candidateCount` stays the bounding-box count before filtering, and
`how: 'anti'` then reports left rows with no key-equal match. Its presence is compile-time and its
contents are per-frame. Defaults are unchanged.

#### Engine selection

`engine?: 'auto' | 'fast' | 'relate'` is compile-time and defaults to `'auto'`. It selects the kernel for
`intersects`, `contains`, `within` and `dwithin`; every other predicate always uses the relate engine.

- `'fast'` runs a short-circuiting kernel with one invocation per candidate. It is quickest for small features.
- `'relate'` runs the DE-9IM engine with one workgroup per candidate and bounding-box pruning. For `dwithin` it
  runs a workgroup distance kernel instead.
- `'auto'` picks `'relate'` when neither side is points and the product of the two sides' average vertices per
  feature is at least 1024. Existing multipart point/line fast routes retain their fast default;
  select `'relate'` explicitly or request `output.relate` to use their matrix route.

Requesting `relate` output always uses the relate engine. Every engine uses the exact orientation sign, so
shared boundaries give the same answers on all three. Measured cost is about 10 ms for 100 pairs of 600 x 175
vertices. A pair with hundreds of shared edges costs 5–10 ms. A great-circle `dwithin` always uses the
short-circuiting kernel (`'relate'` throws, `'auto'` picks `'fast'`).

#### Spatial context of `GPUSpatialPredicateJoin`

`spatialContext` (omitted: native planar) selects how positions are read. It is validated at construction and
exposed as the readonly field `spatialContext`; unsupported combinations throw.

| Context | Predicates | Meaning |
|---|---|---|
| planar (default) | all | Euclidean plane; `dwithin` `distance` in coordinate units. |
| `{coordinateSpace: 'longitude-latitude', metric: 'great-circle'}` | `dwithin` only | positions are longitude/latitude degrees; `distance` is meters on the mean Earth sphere (`sphereRadius` overrides the radius; `units: 'kilometers'` throws). |
| `{coordinateSpace: 'longitude-latitude', metric: 'none'}` | every predicate except `dwithin` | topology in the degree plane, as GEOS and PostGIS `geometry` do for coordinates in degrees; no distances. |

Great-circle `dwithin`: the candidate stage widens each left box by the angular radius in latitude and by the
longitude reach of that radius at the box's far latitude (`2 asin(sin(r/2) / cos(lat))`, every longitude when
the radius reaches a pole), and a box that crosses the antimeridian also probes its copy shifted by 360
degrees, with a 0.1% plus 1e-5 degree margin so rounding never drops a candidate. The exact kernel measures
the central angle from a vertex to an edge with `geodesicSegmentFraction` and `geodesicSegmentDistance`
(`GEODESIC_WGSL`): edges are straight in longitude/latitude with the longitude difference wrapped at +-180,
the fraction of the nearest point is found in the local equirectangular frame of the point, and the
distance is the haversine of coordinate differences, which keeps meter-scale offsets well conditioned in f32.
The same edge model is used by the intersection and containment tests (exact orientation in degrees, as the
planar kernels) and by the network snapping contributors, so there is one convention rather than two. The
foot is exact to first order for edges of road length, so densify long edges. Distances are accurate to f32
rounding across the antimeridian and over a pole, up to about 8000 km (a larger compile-time distance throws,
a larger per-frame value selects nothing). Pairs whose geometries intersect in the degree plane count as
distance 0. Prepare polygons that cross ±180 or enclose a pole with `GPULonLatPolygonPreparation`.
Longitude/latitude contexts require `edgeModel: 'linear'` (the default); `'great-circle'` edges throw
with a preparation hint. Optional `output.status.antimeridianUncertainCount` and
`output.status.poleUncertainCount` report unprepared rings, accumulated over both polygon inputs.
They are cleared each encoding in longitude/latitude contexts; planar contexts leave these fields
untouched and build no ring check. A numeric great-circle `dwithin` distance below ten float32
longitude/latitude quanta (about 17 m on the default sphere) warns to use `createGPULocalFrame`.
Oracle: numpy haversine and
`pyproj.Geod(a=R, f=0)` for point pairs, shapely distances in the azimuthal equidistant projection for
polygons (see the spec fixtures).

`getGPUSpatialPredicateJoinExecutionPlan` accepts `greatCircle: true` only with `predicate: 'dwithin'`.

The topological predicates take `metric: 'none'`; `metric: 'great-circle'` with them throws, and `dwithin`
with `metric: 'none'` throws, so the intent is explicit. `GPUSpatialJoinCandidates` is planar only;
`GPUPointInPolygonJoin` also accepts longitude/latitude with `metric: 'none'`. `GPUNearestFeatureJoin` and
`GPUBufferSelection` also take `metric: 'great-circle'` (see their section); `GPUSpatialJoinPrepared` and `GPUNearestFeatureWeights` take no positions with a metric
(a prepared tree is a bounding-box tree and may serve a great-circle join) and have no context.

#### How it computes

{/* lens:GPUSpatialPredicateJoin */}

::lens{scenario=gpu-spatial-analysis.spatial-predicate-join rung=datum datum=46 caption="Tract 17031031400 (row 46): its candidate parks and how many intersect it."}

::lens{scenario=gpu-spatial-analysis.spatial-predicate-join rung=formula caption="Intersects means some interior or boundary cell of the DE-9IM matrix is non-empty."}

::lens{scenario=gpu-spatial-analysis.spatial-predicate-join rung=kernel caption="The mask kernel reads one DE-9IM matrix per candidate pair and keeps intersecting pairs."}

::lens{scenario=gpu-spatial-analysis.spatial-predicate-join rung=graph caption="Candidate pairs from bounds, an exact relate per pair, a match mask, then scan and scatter."}

### `GPUSpatialJoinPrepared`

Retains a static geometry snapshot for one side of a join.

::lens{scenario=gpu-spatial-analysis.spatial-join-prepared rung=result caption="One retained tract snapshot serves both the candidate and the exact park join."}

A static right-hand side for repeated joins. Geometry refresh is gated on the GPU by requested
and applied generations. The applied generation changes only after the complete build executes,
so discarded encodings and abandoned execution prefixes retry automatically.
Both candidate lookup and exact predicate tests use the retained right-hand geometry. After
writing new right-hand coordinates, call `invalidate()` before encoding the submission that
should use them. A command buffer encoded earlier carries the previous generation even if you
submit it after invalidation.

Ring, bounds, sort, BVH and optional edge-slab preparation replay against the retained geometry
every encoding; consumers share that preparation within the same encoding. This
conservative implementation repeats build work because published nested nodes lack the dispatch
metadata needed for whole-build gating. Resident overhead is one copy of every geometry view plus
1,072 fixed bytes per handle: a 1,024-byte immutable lookup, 32-byte request and 16-byte state.
These fixed bytes are separate from geometry copies and other preparation storage; the snapshot
uses the same view lengths and binding limits.

`partitioning` describes input consumption; it does not partition this whole-domain BVH. With
the default `leafCapacity = nextPowerOfTwo(featureCount)`, each bounds/children binding needs
`8 × (2 × leafCapacity − 1)` bytes. At a 128 MiB binding limit, 8,388,608 features need
134,217,720 bytes for this binding and pass that particular check; 8,388,609 features round up to
16,777,216 leaves and need 268,435,448 bytes, which fails even if input partitions are smaller.
Other device and storage constraints still apply. A partition's peak input bytes cannot size the
resident index or the retained geometry copies.

Query the public whole-domain planner before declaration, using allocated counts for every
hierarchy level and the intended sort/slab settings. Polygon plans require `polygonCount` and
`ringCount`; multipart line plans require `partCount`. Counts are rows, positions are XY vertices,
and returned allocation sizes are bytes. `feasible` checks all reported bindings, not just the
BVH formula above; partition peak bytes are not a substitute for these counts.

```ts
import type {Device} from '@luma.gl/core';
import {getGPUSpatialJoinPreparedMemoryPlan} from '@luma-contributors/analysis/gpu-spatial-analysis';

declare const device: Device;
// Two allocated polygon features, one polygon/ring and four vertices per feature.
const memoryPlan = getGPUSpatialJoinPreparedMemoryPlan({
  kind: 'polygons', featureCount: 2, vertexCount: 8, polygonCount: 2, ringCount: 2,
  spatialSort: false, edgeSlabs: false
}, device.limits);
if (!memoryPlan.feasible) {
  const failures = memoryPlan.constraints.filter(constraint => !constraint.satisfied);
  throw new Error(`Prepared geometry exceeds whole-domain limits: ${JSON.stringify(failures)}`);
}
console.log(memoryPlan.leafCapacity, memoryPlan.nodeCount); // 2 leaves, 3 nodes
console.log(memoryPlan.bindingByteLengths['BVH node bounds/children']); // 24 bytes per binding
// Geometry positions add 64 snapshot bytes; other offsets/control/scratch are listed separately.
```

A failure while constructing preparation nodes or allocating their owned buffers is cleaned up
inside the producer: owned allocations are released and `isDeclaredIn(graph)` stays false.
A later failure while `graph.add(prepared)` registers the returned nodes happens after that
callback has succeeded, so the producer cannot observe it. Call
`prepared.recoverFailedDeclaration()` to release/reset that declaration, discard the partially
registered graph, and retry on a fresh graph. Never recover a successfully submitted graph;
destroy compiled graphs using the handle before destroying its resources. Even an automatically
cleaned construction failure may leave imported IDs in the failed graph, so do not reuse it.

On retry, import each persistent, caller-owned GPU buffer through the fresh graph's public
canonical import (`graph.importGPUData` with a public `GPUData` wrapper). Its view must cover the
original range, including format, byte offset, stride and length. Use those equivalent fresh views
for consumers too; a genuinely different range is rejected. When a fresh producer already imports that buffer through a different handle (a
DynamicBuffer versus its Buffer, or a narrower usage), pass its views with
`prepared.setRetryViews(freshGraph, {geometry, storage})` before `graph.add(prepared)`; each must
describe the original range. Physical equivalence alone does not make the graph order a producer's
writes before the preparation's reads. Never carry a
transient view into the fresh graph: recreate its producer, views and preparation handle there.
Caller-owned storage remains caller-owned and readable before declaration and after handle
destruction; failed cleanup frees only handle-owned buffers. These declaration failures differ
from discarded encodings, whose pending generation requests retry automatically.

The first encoding and first encoding after explicit invalidation bypass `rebuildWhen`; other
encodings call it once. Public join props preserve the handle's frame type: use
`GPUSpatialJoinCandidatesProps<Frame>` or `GPUSpatialPredicateJoinProps<Frame>` when a helper
registers `props.prepared` on a `GPUCommandGraph<Frame>`. Point-in-polygon and nearest-feature
props take the frame type after their existing row/geometry type parameters.

The props and handle accept a frame type parameter, for example
`GPUSpatialJoinPrepared<{geometryChanged: boolean}>`; use the same type on `GPUCommandGraph`.
The default is `unknown`, suitable for `invalidate()` without a frame predicate. See the
[execution guide](./concepts.md) for a typed predicate example. A recorded refresh request remains
pending until its complete preparation executes. `encodedBuildCount` counts CPU refresh-request
encodings, including discarded requests, and excludes automatic GPU retries and replayed builds.

```ts
import {type GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  GPUSpatialJoinPrepared, GPUSpatialPredicateJoin, GPUPointInPolygonJoin,
  type GPUSpatialPredicateJoinProps, type GPUPointInPolygonJoinProps
} from '@luma-contributors/analysis/gpu-spatial-analysis';

declare const graph: GPUCommandGraph;
// Local factory from the readback-destinations example above, not a library export.
declare const createReadbackView: <Format extends 'uint32' | 'float32' | 'float32x2'>(
  graph: GPUCommandGraph, name: string, format: Format, length: number
) => GraphDataView<Format>;

declare const polygons: GPUPointInPolygonJoinProps['polygons'];
declare const movingPoints: GPUPointInPolygonJoinProps['points'] & GPUSpatialPredicateJoinProps['left'];
declare const pairs: NonNullable<GPUSpatialPredicateJoinProps['output']['pairs']>;
declare const pointFeatureIds: NonNullable<GPUPointInPolygonJoinProps['output']['pointFeatureIds']>;
declare const polygonAssignmentCounts: NonNullable<GPUPointInPolygonJoinProps['output']['polygonAssignmentCounts']>;
declare const polygonMembershipCounts: NonNullable<GPUPointInPolygonJoinProps['output']['polygonMembershipCounts']>;
const candidateOverflow = createReadbackView(graph, 'prepared-candidate-loss', 'uint32', 1);
const uncertainCount = createReadbackView(graph, 'prepared-uncertain', 'uint32', 1);
const pointOverflow = createReadbackView(graph, 'assignment-loss', 'uint32', 1);
const pointUncertainCount = createReadbackView(graph, 'assignment-uncertain', 'uint32', 1);
const prepared = new GPUSpatialJoinPrepared({geometry: polygons});
graph.add(prepared); // before the joins that use it
graph.add(new GPUSpatialPredicateJoin({
  left: movingPoints, right: polygons, prepared, predicate: 'within', how: 'inner',
  candidateCapacity: 1 << 20,
  output: {
    pairs: {...pairs, status: {...pairs.status, candidateOverflow}},
    status: {uncertainCount}
  }
}));
graph.add(new GPUPointInPolygonJoin({
  points: movingPoints, polygons, prepared, candidateCapacity: 1 << 20,
  output: {
    pointFeatureIds, polygonAssignmentCounts, polygonMembershipCounts,
    status: {overflow: pointOverflow, uncertainCount: pointUncertainCount}
  }
}));
// Pair counts require clear pairs.status.overflow, candidateOverflow and uncertainCount.
// PIP counts require clear pointOverflow and pointUncertainCount.
prepared.invalidate(); // request a fresh geometry snapshot; discarded encodings retry it
```

It reuses geometry, while query/left inputs remain current each encoding. Prepared candidates,
predicate exact refinement, point-in-polygon refinement and nearest traversal/distance reads use
the same retained positions and full offset hierarchy. Moving the source coordinates without
invalidation therefore leaves geometry answers at the captured generation; call `invalidate()`
and submit the complete rebuild to use the new geometry. `prepared.getGeometry(graph)` exposes
those captured views after successful declaration in that graph, and throws before declaration,
for another graph, or after `destroy()`/`recoverFailedDeclaration()` releases them. Source-ID
contents are separate: they remain live for consumer output mapping, so changing an ID can
change the published caller ID without changing a retained feature row.

Source-ID views must belong to the same graph as the geometry; `graph.add(prepared)` rejects
them otherwise.
Storage is
persistent buffers, allocated by the handle and freed by `destroy()`, or passed as `storage`. One handle
belongs to one graph, and several joins may share it. A sorted prepared handle is accepted by
`GPUPointInPolygonJoin`, `GPUSpatialPredicateJoin`, `GPUSpatialJoinCandidates` and
`GPUNearestFeatureJoin` (points, lines and polygons, not segments). Sorting preserves logical
feature rows, including multipart grouping.

#### How it computes

{/* lens:GPUSpatialJoinPrepared */}

::lens{scenario=gpu-spatial-analysis.spatial-join-prepared rung=datum datum=46 caption="Tract 17031031400 (row 46): matched through the retained geometry snapshot."}

::lens{scenario=gpu-spatial-analysis.spatial-join-prepared rung=kernel caption="Geometry refresh is GPU-gated; bounds and edge builders replay against the snapshot."}

::lens{scenario=gpu-spatial-analysis.spatial-join-prepared rung=graph caption="Refresh the geometry, replay public builders, then commit the applied generation on the GPU."}

### `GPUSpatialJoinCandidates`

Lists feature pairs whose bounding boxes overlap: the filter step before exact predicates.

::lens{scenario=gpu-spatial-analysis.spatial-join-candidates rung=result caption="Park and tract pairs with overlapping boxes; boxes can overlap where polygons do not."}

Maximum: `output.leftRows.length` is at most `getGPUSpatialJoinCandidatesMaximumPairCapacity(device)` (2^24 on a default device).

The bounding-box candidate stage on its own: `(left, right)` pairs sorted by `(left, right)` whose bounding
boxes overlap, with the left box optionally expanded by `distance`. The `output` is a `GPUCompactPairPort`
(capacity `output.leftRows.length`, `status.count`, `status.overflow`, `status.candidateOverflow?`, `status.requiredCount`); slots past `status.count` hold `0xffffffff`.
`overflow` reports final pair-capacity loss only; `candidateOverflow` independently reports BVH
leaf loss. Both supplied words are overwritten with 0 or 1 on every encoding, including clean
reuse. When BVH leaves are lost, `requiredCount` counts only enumerated pairs and is a lower bound.
A `leafCapacity` below the right feature count requires `candidateOverflow`, including
when using a prepared index. Consumers that omit that diagnostic must use enough BVH slots;
otherwise construction rejects the configuration. `GPUPairGather` propagates the two causes separately.
The constructor captures geometry descriptors, options, pair views and nested status destinations;
editing the caller's maps later cannot retarget a declared binding. Buffer contents remain dynamic.
The separate flag passes permit adjacent `overflow` and `candidateOverflow` words in one buffer;
the words must not logically overlap. Outputs bound together must also have disjoint aligned ranges.
With both flags zero it is conservative: no intersecting pair is missed, and some candidates do not intersect. It has no
`excludeSameRow`.

`distance` is planar in coordinate units (`spatialContext` defaults to native planar coordinates and a
`longitude-latitude` context throws; use `GPUSpatialPredicateJoin` `dwithin` for great-circle candidates).
It may be a compile-time number or a one-row `GraphDataView<'float32'>`. The view is read every
encoding, so an interactive distance join can reuse one compiled graph. Size the pair capacity for the
largest distance that the application permits. Negative, NaN and infinite view values act as zero; invalid
compile-time numbers throw.

```ts
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import type {Device, CommandEncoder} from '@luma.gl/core';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
import {
  GPUSpatialJoinCandidates, type GPUSpatialJoinCandidatesProps
} from '@luma-contributors/analysis/gpu-spatial-analysis';

declare const graph: GPUCommandGraph;
declare const device: Device;
declare const commandEncoder: CommandEncoder;
declare const left: GPUSpatialJoinCandidatesProps['left'];
declare const right: GPUSpatialJoinCandidatesProps['right'];
declare const output: GPUSpatialJoinCandidatesProps['output'];
const candidateDistance = new GPUParameterBuffer(device, {
  id: 'candidate-distance', format: 'float32', length: 1
});
graph.add(new GPUSpatialJoinCandidates({
  left, right, distance: candidateDistance.importToGraph(graph), output
}));
const compiled = await graph.compileAsync();
candidateDistance.write(Float32Array.of(2_000));
compiled.encode(commandEncoder, {parameters: undefined});
```

#### How it computes

{/* lens:GPUSpatialJoinCandidates */}

::lens{scenario=gpu-spatial-analysis.spatial-join-candidates rung=datum datum=46 caption="Tract 17031031400 (row 46): every park whose box overlaps the tract's box."}

::lens{scenario=gpu-spatial-analysis.spatial-join-candidates rung=kernel caption="Count and write kernels test box overlap; the write pass fills each row's scanned slots."}

::lens{scenario=gpu-spatial-analysis.spatial-join-candidates rung=graph caption="Count candidates per row, prefix-scan to offsets, then write each row's pairs in place."}

### `GPUPointInPolygonJoin` and `GPUNearestFeatureJoin`

`GPUPointInPolygonJoin`: Chunked queries keep global source order and shared polygon geometry.
candidateTileCapacity bounds paged candidate scratch independently of the unchanged global
candidateCapacity; global overflow/status semantics remain.

**`GPUPointInPolygonJoin`**: Assigns each point the polygon that contains it, under an explicit boundary rule, and counts points per polygon.

::lens{scenario=gpu-spatial-analysis.point-in-polygon-join rung=result caption="2,344 observations (grey) assigned to the containing tract; colour is the tract."}

**`GPUNearestFeatureJoin`**: Finds each query feature's nearest feature within a radius and reports its row or caller ID and optional distance, including multipart points and lines.

::lens{scenario=gpu-spatial-analysis.nearest-feature-join rung=result caption="Observations coloured by planar distance to the nearest library."}

**`GPUNearestFeatureWeights`**: Turns each point's k nearest features into a CSR spatial-weights matrix with inverse-distance weights.

::lens{scenario=gpu-spatial-analysis.nearest-feature-weights rung=result caption="Each observation linked to its three nearest libraries, weighted by inverse distance."}

Maximum (`GPUNearestFeatureJoin`, neighbors mode and great-circle): the node table `5 * (2 * leafCapacity - 1) + 2 * queries` (plus `2 * features` with a filter) is at most `getGPUNearestFeatureJoinMaximumNodeTableWordCount(device)` (2^25 words on a default device), which allows 2^21 leaves without queries; `leafCapacity` is at most `getGPUNearestFeatureJoinMaximumLeafCapacity(device)` (2^23).

Spatial joins over a `GPUBVH` built from feature bounds every encoding. `GPUPointInPolygonJoin`
classifies each (point, feature) candidate with the shared exact orientation sign (even-odd rings, union
polygons) and assigns each point the containing feature with the smallest row (see the boundary rule
below); `GPUNearestFeatureJoin` measures candidates with
`GPUPairwisePointSegmentDistance` and keeps the nearest point or segment feature within a per-frame
radius. Both take the points as a `GPUPointGeometryPort` (`points` and `queries`; chunked positions and
`sourceIds` allowed) and the features as a geometry port whose optional `sourceIds` replace feature rows in
the outputs. Both can write, in `output`, per-point feature rows or IDs, per-feature counts, optional compact matches,
and one `output.status.overflow` flag covering BVH leaf, candidate, and match capacity. `GPUPointInPolygonJoin` takes planar
coordinates (the default) or `{coordinateSpace: 'longitude-latitude', metric: 'none'}`. Containment is
topological with straight edges in degrees, as in GeoJSON, and gives the same result as planar. A
longitude/latitude context with a metric such as `'great-circle'` throws at construction, because the join
has no geodesic edges ("project positions to a planar frame first"). `GPUNearestFeatureJoin` also takes the
great-circle context below.

Both joins require linear longitude/latitude edges (omitted `edgeModel` means `'linear'`);
`edgeModel: 'great-circle'` throws with a preparation hint. Prepare polygons that cross ±180 or enclose
a pole with `GPULonLatPolygonPreparation`. Optional `output.status.antimeridianUncertainCount` and
`output.status.poleUncertainCount` report unprepared rings; nearest joins count both polygon inputs.
These counters reset each encoding for longitude/latitude and remain untouched for planar contexts,
which build no ring check. Empty rings and polygons in the preparation's fixed 2x topology are accepted.
`getPointInPolygonContainmentContext` preserves the input edge model and datum when selecting
`metric: 'none'`, so it cannot bypass edge-model validation.

**Great-circle nearest features (`GPUNearestFeatureJoin`, `GPUBufferSelection`).** With
`spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}` positions are longitude and
latitude in degrees and every distance (`radius`, `nearestDistances`, `neighborDistances`,
`GPUBufferSelection` `radius` and `output.distances`) is in the context's length unit (meters by default;
kilometers with `units: 'kilometers'`) on the mean Earth sphere (`sphereRadius` overrides the radius in
that unit). A `longitude-latitude` context with another metric and planar contexts with `great-circle`
throw at construction. Both distance operators reject `local-lambert-equal-area` frames. Both output modes use the branch-and-bound BVH traversal of the neighbors mode (nearest-feature
mode is the `k = 1` traversal followed by the nearest-feature columns, counts and matches), so
`candidateCapacity` is not used on this path (it is accepted and ignored, `candidateCount` reads 0, and
`GPUBufferSelection` still validates it). The traversal orders and bounds by central angle. A node is pruned
with a lower bound of the great-circle distance between its box and the query box,
`hav(d) = hav(dlat) + cos^2(max |lat|) hav(dlon)`, with the latitude gap, the circular longitude gap (through
the antimeridian when that is shorter) and the cosine of the latitude farthest from the equator over both
boxes, which never overestimates; a box 180 degrees or wider in longitude (a feature across the antimeridian)
counts as covering every longitude, and a box touching a pole has cosine 0. Edges are straight in
longitude/latitude with wrapped longitude differences, so the vertex boxes bound them exactly. Point to edge
distance uses `geodesicSegmentFraction` and `geodesicSegmentDistance`; a crossing or containment is decided
in the degree plane with the exact orientation sign after the second edge is shifted by a multiple of 360
degrees to overlap the first, and counts as distance 0. Foot points and query points are on the edges with
longitude wrapped to +-180. The mixed convention of the planar-in-degrees tests is therefore one edge model;
densify edges longer than a few kilometers, because the foot is exact to first order in the local frame of the
query. Prepare crossing and pole-enclosing polygons as described above.
No warning is possible because the distance is a GPU view; see [Precision of longitude/latitude positions](./concepts.md#precision-of-longitudelatitude-positions).
The planar kernels are unchanged and compile
none of this code. Oracle: `geodesic-nearest-fixtures.ts` (numpy 2.5.4 haversine of the edge densified in
longitude/latitude, `pyproj` 3.8.0 `Geod(a=R, f=0)` cross-check), scenes at the equator, latitude 60, across
the antimeridian and next to the pole. `GPUNearestFeatureWeights` adapts the join's outputs; pass the
join's `spatialContext` so its distances and `distanceFloor` share the context's length unit and
inverse-distance weights use those distances.

**Exact orientation.** The classifier and the nearest-feature kernels use the same exact orientation sign
(an f32 filter with an exact 256-bit integer fallback), so "on the edge" means the same thing in the
point-in-polygon join, the predicate join and the nearest-feature join: a point exactly on an edge is
distance 0 in all three, and polygon containment (distance 0 inside a polygon) is decided with the exact
sign too. Distances and foot points stay f32. Only candidates that no arithmetic can decide stay
`uncertain`: non-finite input or malformed offsets. In `GPUPointInPolygonJoin` an empty polygon part
or a ring with fewer than three vertices contains nothing and is skipped, as in shapely, so the
feature's other parts still classify the point. Uncertain
pairs are never assigned. They are counted in the optional `uncertainCount`, so such a point stays
unassigned visibly rather than silently.

**Boundary rule (`GPUPointInPolygonJoin`).** `boundaryRule` decides a point exactly on a polygon
boundary:

| Rule | Boundary point | Notes |
|---|---|---|
| `'half-open'` (default) | belongs to exactly one feature of a valid coverage, independent of row order | The owner is the feature containing the point after an infinitesimal move toward +x and +y (+y dominates): left and bottom boundaries are included, right and top boundaries are excluded. A horizontal edge belongs to the feature above it, a vertical edge to the feature on its right, a vertex to the feature whose interior touches its upper right. Points on the outer right or top boundary of a coverage belong to no feature. |
| `'closed'` | contained by every feature it touches, then the smallest row wins | Nothing is lost on the outer boundary; the owner of a shared-boundary point depends on the row order. |
| `'open'` | belongs to no feature | Only strict interior counts; boundary points are lost, not half-assigned. |

The half-open decision is the crossing-number convention made exact: a
supporting-line tie is broken by the sign of dx (the epsilon term of the move), or of -dy for a vertical
edge, so the result follows from the exact orientation signs and the vertex order does not matter. Overlapping
features (an invalid coverage) still resolve to the smallest containing row. Oracle: shapely
`Polygon.contains(Point(x + 1e-9, y + 1e-6))` on integer scenes, 24 row permutations, in the specs.
`GPUZonalStatistics` and the recipes that pass no `boundaryRule` use the half-open default, so a coverage's
counts partition the points.

For exclusive counts use `output.polygonAssignmentCounts`; for all matches under the selected
boundary rule use `output.polygonMembershipCounts`. `output.pointFeatureRows` keeps polygon row
indices before `pointFeatureIds` applies caller `sourceIds`. Use rows for zonal membership and
attribute gathers; IDs may be repeated or unrelated to row offsets.

#### Zonal values and inference

Complete polygon adjacency does not make means from incomplete point membership complete.
For four adjacent polygons and point values `[1,3,2,4,100]` (the last two in polygon row 3),
candidate capacity 4 can leave means `[1,3,2,4]` with membership loss; complete membership gives
`[1,3,2,52]`. Keep the original zonal loss words alongside the means and corrected probabilities.
`candidateTruncated` has one row per **point**, so it cannot directly mask polygon-row inference.

`GPULocalMoran.createCommonMaskNodes` converts the scalar zonal status into a scalar `complete`
certificate and a common polygon-row mask. Pass that same mask to analytic and permutation
inference, and use the permutation test's tested family for subsequent multiple-testing adjustment.
The helper rereads diagnostics every encoding; incomplete membership masks the whole inference
family rather than returning finite tests on partial means. If an application has a separate scalar
`valuesComplete` certificate, the helper can consume that too. Selectively retaining polygons
requires an explicit point-to-polygon loss mapping, not a reinterpretation of point flags.
For a geographic polygon route, also include its written `antimeridianUncertainCount` and
`poleUncertainCount` in `membershipStatus`; the planar fixture below needs the three shown scalars.

```ts
import {type GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  GPUZonalStatistics, GPULocalMoran, GPULocalPermutationTest,
  type GPUZonalStatisticsProps, type GPULocalMoranProps, type GPULocalPermutationTestProps
} from '@luma-contributors/analysis/gpu-spatial-analysis';

declare const graph: GPUCommandGraph;
// Local factory from the readback-destinations example above, not a library export.
declare const createReadbackView: <Format extends 'uint32' | 'float32' | 'float32x2'>(
  graph: GPUCommandGraph, name: string, format: Format, length: number
) => GraphDataView<Format>;

declare const zonalInputs: Omit<GPUZonalStatisticsProps, 'output'>;
// Both inference configurations include complete producer weights, parameters and output views.
declare const analytic: Omit<GPULocalMoranProps, 'values' | 'mask'>;
declare const permutation: Omit<GPULocalPermutationTestProps, 'values' | 'mask'>;
const polygonCount = 4;
const means = createReadbackView(graph, 'polygon-means', 'float32', polygonCount);
const membershipStatus = {
  overflow: createReadbackView(graph, 'membership-overflow', 'uint32', 1),
  candidateOverflow: createReadbackView(graph, 'membership-candidate-loss', 'uint32', 1),
  uncertainCount: createReadbackView(graph, 'membership-uncertain', 'uint32', 1)
};
graph.add(new GPUZonalStatistics({...zonalInputs, output: {means, status: membershipStatus}}));
const common = GPULocalMoran.createCommonMaskNodes(graph, {
  rowCount: polygonCount, membershipStatus
});
for (const node of common.nodes) graph.add(node);
graph.add(new GPULocalMoran({...analytic, values: means, mask: common.mask}));
graph.add(new GPULocalPermutationTest({...permutation, values: means, mask: common.mask}));
// common.mask and common.complete are graph-local scratch consumed during this encoding.
// Retain only caller-owned published views, whose backing buffers are in publishedBuffers.
const retainedResult = {means, membershipStatus};
```

```ts
import {type GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {GPUNearestFeatureJoin} from '@luma-contributors/analysis/gpu-spatial-analysis';

declare const graph: GPUCommandGraph;
// Local factory from the readback-destinations example above, not a library export.
declare const createReadbackView: <Format extends 'uint32' | 'float32' | 'float32x2'>(
  graph: GPUCommandGraph, name: string, format: Format, length: number
) => GraphDataView<Format>;

declare const points: GraphDataView<'float32x2'>;
declare const roadStarts: GraphDataView<'float32x2'>;
declare const roadEnds: GraphDataView<'float32x2'>;
// One-row float32 radius in planar coordinate units, updated per encoding.
declare const radius: GraphDataView<'float32'>;
const nearestFeatureIds = createReadbackView(graph, 'road-ids', 'uint32', points.length);
const nearestDistances = createReadbackView(graph, 'road-distances', 'float32', points.length);
const overflow = createReadbackView(graph, 'road-loss', 'uint32', 1);
graph.add(new GPUNearestFeatureJoin({
  queries: {kind: 'points', positions: points},
  features: {kind: 'segments', starts: roadStarts, ends: roadEnds},
  radius, candidateCapacity: Math.max(1, points.length * 4),
  output: {nearestFeatureIds, nearestDistances, status: {overflow}}
}));
```

Both joins accept a [`prepared`](#gpuspatialjoinprepared) handle. Geometry refresh is generation-gated;
ring, bounds, sort, BVH and optional slab preparation replay every encoding, shared by consumers
within that encoding. The handle must be added to the graph first, and `spatialSort` and `leafCapacity` then come
from the handle (conflicting props throw).

Planar radius membership is inclusive across distance two: a feature exactly at the supplied
radius is retained, and the next smaller float32 radius excludes it. Nearest foot points retain
the projection residual for distant queries. For example, `(33554432, -33554432)` projected
onto `(-3, -3) → (3, 3)` has foot `(0, 0)` in either vertex order; the published query and
feature witnesses still describe the same minimum-distance pair.

**k nearest features.** Set `output.neighborIds` and `output.neighborCounts` on `GPUNearestFeatureJoin` to get the `k`
nearest features per query (PostGIS `a <-> b ORDER BY ... LIMIT k`, GeoPandas `sjoin_nearest`). The
original mode above is unchanged. Each query walks the BVH nearest child first and prunes boxes farther than
the current `k`-th distance, which is exact and deterministic.

Choose settings for the route actually requested:

| Route | Search settings | Result domain |
|---|---|---|
| Nearest-feature columns | one winner; required `radius` is inclusive and per encoding | one row per query, optional counts per target feature |
| Neighbors columns | `neighborCount`, `neighborCapacity`, `ties`, optional `radius` | dense `[query, slot]` lists plus one count per query |
| `GPUNearestFeatureWeights` | `slotCapacity` must match the producer; `neighborCount` and `excludeSelf` select slots; `weightType` chooses weighting | CSR rows and columns are feature rows |

`power` and `distanceFloor` configure `'inverse-distance'` weights; they do not alter binary weights
or the nearest search. The adapter does not perform a new radius search. `candidateCapacity` bounds
only the planar single-point-to-point/segment candidate route; neighbors, multipart, line/polygon
and great-circle traversal routes use the BVH without that candidate table.

- `queries` are points (possibly chunked), lines or polygons (one packed view each); features are `points`,
  `segments`, `lines` or `polygons` (multipolygons, holes, even-odd). Distance is the minimum planar
  distance between the geometries and 0 when a polygon contains the other geometry. Polygons must be valid.
- `neighborCount` and `neighborCapacity` are positive integers, bounded by the packed-index maximum
  `0x7ffffffe` (2147483646); actual usable capacity is lower when device storage bindings cannot
  hold the scratch/output views. `neighborCapacity >= neighborCount`. The 64-slot private-storage threshold
  is not the public maximum: 65 and larger capacities use storage-backed lists. The interleaved
  top-k scratch needs `3 + 2 * neighborCapacity` uint32 words per scratch query plus three shared
  control words; each output column needs `queryCount * neighborCapacity` rows in one binding.
  Foot-point scratch adds five words per slot. Each must fit `maxStorageBufferBindingSize`, and
  all packed indices must fit uint32. `ties` is `'lowest-row'` (default behavior) or `'all'`, which keeps every feature tied with the
  `k`-th.
  `neighborCapacity` can exceed `neighborCount` only with `ties: 'all'`.
  `radius` (per-frame, inclusive; omitted means unbounded) bounds the search. Negative, NaN and
  infinite radius values yield no neighbors for that encoding; a later finite nonnegative radius
  resumes matching on the same compiled graph.
  In planar mode, an ambiguous comparison between a distance rank and the radius is resolved
  against the original coordinates and radius. A match at the radius is retained, while a radius
  one float32 step smaller can exclude it. Reversed copies of one segment use the same canonical
  endpoint order and distance rank, so exact duplicate roads tie by feature row.
- Outputs are dense, `query * neighborCapacity + slot`, ordered by distance and then feature row:
  `neighborIds` (remapped by `features.sourceIds`), `neighborCounts`, optional `neighborRows`
  (the original feature rows for attribute lookup), optional `neighborDistances`,
  `neighborFootPoints` (the nearest point on the feature, or the crossing point when geometries cross)
  and `neighborSegmentIndices` (the start-vertex index of the nearest edge; `GPU_NEAREST_NO_SEGMENT` for
  points and containment). Unused slots hold `GPU_SPATIAL_JOIN_NO_FEATURE`, -1, NaN and
  `GPU_NEAREST_NO_SEGMENT`. `neighborRows` uses `GPU_SPATIAL_JOIN_NO_FEATURE` in unused slots,
  requires `neighborIds`, and has the same length. Allocate disjoint views for IDs and rows even
  when no `sourceIds` are supplied.
  Planar projected coordinates choose between an endpoint construction and a query-local
  perpendicular construction to reduce cancellation; a segment's constant coordinate is
  preserved exactly. This also preserves small perpendicular residuals near distant roads.
- `output.status.overflow` is 1 when `ties: 'all'` had more ties than `neighborCapacity` or the BVH leaf capacity was
  exceeded. Distances are f32, so near-equal distances can order differently from a double reference, and
  tie order and the foot point at exact vertex ties depend on evaluation order.
- The per-query scratch (result slots, and the foot-point scratch when a foot-point output is requested)
  is one binding while it fits `maxStorageBufferBindingSize` (128 MiB by default). Beyond that, for
  example 838k queries with `k = 8` and foot points, the scratch holds a fixed chunk of queries (a multiple
  of 64) that every chunk of queries reuses in turn, so the query count is not limited by the scratch. The
  caller-owned output views still have to fit their own bindings.

- `exclusive` skips features whose ID equals the query's ID (`features.sourceIds[row]` or the row, and optional
  `queryIds`, which require `exclusive`). Both modes support this equal-ID exclusion. It differs from
  `STRtree.query_nearest(exclusive=True)`, which excludes equal geometries: for query `(0,0)` with
  ID 0 and targets `(0,0)`, `(0,0)`, `(2,0)` with IDs 0, 1, 2, target row 1 remains eligible at
  distance zero. Distinct IDs do not make coincident geometries unequal geometrically.
- `neighborQueryPoints` (`float32x2`, the layout of `neighborIds`) is the point of the query geometry
  nearest to the feature, NaN in unused slots. `(neighborQueryPoints, neighborFootPoints)` is
  `shapely.shortest_line` or `ops.nearest_points` for every point, line and polygon pair. Ties between
  parallel edges return the first in vertex order, which can differ from GEOS. Both points are
  selected together from one minimizing endpoint/projection pair; overlapping edges publish the
  same point on both sides rather than independently choosing opposite endpoints. Both point
  outputs overwrite unused coordinates with NaN on every submission, including when a smaller
  radius empties slots that were filled on the preceding submission.
- `onAttribute: {left, right}` (as on `GPUSpatialPredicateJoin`) finds the nearest key-equal feature:
  the filter runs inside the traversal, and nearest-feature mode filters candidates in the reduce passes.

```ts
import {type GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {GPUNearestFeatureJoin} from '@luma-contributors/analysis/gpu-spatial-analysis';

declare const graph: GPUCommandGraph;
// Local factory from the readback-destinations example above, not a library export.
declare const createReadbackView: <Format extends 'uint32' | 'float32' | 'float32x2'>(
  graph: GPUCommandGraph, name: string, format: Format, length: number
) => GraphDataView<Format>;

declare const points: GraphDataView<'float32x2'>;
declare const linePositions: GraphDataView<'float32x2'>;
declare const lineOffsets: GraphDataView<'uint32'>;
declare const radius: GraphDataView<'float32'>; // one scalar, planar coordinate units
const neighborCapacity = 8;
const slots = points.length * neighborCapacity;
const nearestOutput = {
  neighborIds: createReadbackView(graph, 'line-ids', 'uint32', slots),
  neighborRows: createReadbackView(graph, 'line-rows', 'uint32', slots),
  neighborCounts: createReadbackView(graph, 'line-counts', 'uint32', points.length),
  neighborDistances: createReadbackView(graph, 'line-distances', 'float32', slots),
  neighborFootPoints: createReadbackView(graph, 'line-feet', 'float32x2', slots),
  neighborSegmentIndices: createReadbackView(graph, 'line-segments', 'uint32', slots),
  candidateTruncated: createReadbackView(graph, 'line-query-loss', 'uint32', points.length),
  status: {
    overflow: createReadbackView(graph, 'line-loss', 'uint32', 1),
    candidateOverflow: createReadbackView(graph, 'line-candidate-loss', 'uint32', 1)
  }
};
graph.add(new GPUNearestFeatureJoin({
  queries: {kind: 'points', positions: points},
  features: {kind: 'lines', positions: linePositions, lineOffsets},
  neighborCount: 3, ties: 'all', neighborCapacity, radius, spatialSort: true, output: nearestOutput
}));
```

`spatialSort: true` reorders features along a Hilbert curve of their bound centers by default before the
`GPUBVH` build. Results are identical, and traversal is 30x to 75x cheaper when features are not
spatially coherent in row order.

#### `GPUNearestFeatureWeights`

Adapter from the k-nearest output of `GPUNearestFeatureJoin` to a `GPUSpatialWeights` CSR (libpysal 4.15 `KNN`
and `Graph.build_knn`, for points, lines or polygons, using the join's exact geometry distances).
For multipart points/lines, CSR rows and columns are logical features and distances are minima
over their part pairs. Pass the
join's `neighborRows`, `neighborCounts`, optional `neighborDistances` and its `neighborCapacity` as
`slotCapacity`. Pass the join's resolved `spatialContext` to the adapter (omitted means native planar);
`distanceFloor` uses its length unit and `outputWeights.spatialContext` carries it to consumers. Per row it keeps the first `k` valid slots in the join's distance order, drops padding and
(with `excludeSelf`, for self-joins run with `k + 1`) the query's own feature row, then sorts the feature rows ascending. Short rows
stay short. The shared `weightType` subset for `GPUNearestFeatureWeights` and
`GPUNeighborSearch` is `binary` and `inverse-distance`; `kernel` additionally works on neighbor
search only. Do not copy a kernel-weight request into this nearest-feature adapter: unsupported
values throw. For this adapter, `weightType` is `'binary'` (default) or `'inverse-distance'` (`1 / max(d, distanceFloor)^power`;
`distanceFloor` defaults to `1e-6`, the same default as `GPUNeighborSearch`, so coincident or touching features weigh `1e6^power`
and dominate a row once it is standardized; pass a floor on the scale of the data when that is not wanted),
and `output.distances` is filled when present (in the join's context length unit, also used by `distanceFloor`).
The `'average-reciprocal'` transform accepts the directed weights with a separate output. It preserves
the stored links and treats a missing reverse weight as zero. For reciprocal support, use
`GPUSpatialWeightsAlgebra` to take the union of the weights and their `GPUSpatialWeightsTranspose`.
`excludeSelf` compares feature rows to
the query row. Request the join's `neighborRows` alongside its `neighborIds`; application IDs may
repeat across segment rows and must never become CSR column indices. Two winning segments with
source IDs `[7, 7]` produce columns `[0, 1]` when their distinct rows are used. The adapter's old
`neighborIds` input is rejected; precomputed input must likewise contain distinct feature rows per
query. `output` uses the CSR views of `GPUSpatialWeightsOutput`, with a narrower status map:
required `overflow` and optional `candidateOverflow`. The adapter rejects `count`, `requiredCount`,
`uncertainCount`, `invalidCount`, `iterationCount`, `converged`, `iterationLimitReached` and
`approximation` at construction because it publishes only these two completeness flags. Omit those
destinations; read the retained slot count from the final CSR offset. Pass the producing join's status as `neighborStatus` and provide
`output.status.candidateOverflow`: it carries any upstream search loss, including truncated tie
slots. `output.status.overflow` reports only the adapter's final CSR capacity loss. Both flags
are rewritten on each submission. Downstream statistics such as local Moran suppress inference
when either is nonzero. For precomputed complete slots, omit `neighborStatus`; an optional
candidate flag then receives zero.
Missing `status` or `status.overflow` is rejected immediately with a prop-specific construction
error. Join-only `candidateCount`, `candidateTruncatedCount`, `antimeridianUncertainCount` and
`poleUncertainCount` destinations are rejected too; they are not outputs of this adapter.
Unrelated geometry metadata such as `vertexCount` is omitted from the captured status rather than
treated as a destination. The refusal rule covers the captured bounded-result and join diagnostics.
The adapter likewise captures its input/output maps at construction. Its final and candidate flags
are written in separate passes and may occupy adjacent words of one buffer; using the same word
for both is rejected.

The spec pins libpysal on 24 random points with `k = 4` (identical neighbor sets). Deviations: ties at the
`k`-th distance keep the lowest feature rows (libpysal breaks them in KDTree order, and `KNN` and
`Graph.build_knn` disagree with each other on grids); inverse-distance floors the distance instead of
dividing by zero.

```ts
import {type GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  GPUNearestFeatureWeights, type GPUNearestFeatureWeightsProps
} from '@luma-contributors/analysis/gpu-spatial-analysis';

declare const graph: GPUCommandGraph;
// Local factory from the readback-destinations example above, not a library export.
declare const createReadbackView: <Format extends 'uint32' | 'float32' | 'float32x2'>(
  graph: GPUCommandGraph, name: string, format: Format, length: number
) => GraphDataView<Format>;

// Slots from a self nearest join with k = 5 and neighborCapacity = 5.
declare const ports: Pick<GPUNearestFeatureWeightsProps,
  'neighborRows' | 'neighborCounts' | 'neighborDistances'>;
declare const joinStatus: NonNullable<GPUNearestFeatureWeightsProps['neighborStatus']>;
declare const knnWeights: GPUNearestFeatureWeightsProps['output'];
const overflow = createReadbackView(graph, 'knn-final-loss', 'uint32', 1);
const candidateOverflow = createReadbackView(graph, 'knn-search-loss', 'uint32', 1);
graph.add(new GPUNearestFeatureWeights({
  ...ports, neighborStatus: joinStatus, slotCapacity: 5, neighborCount: 4, excludeSelf: true,
  weightType: 'binary', output: {...knnWeights, status: {overflow, candidateOverflow}}
}));
```

#### Spatial sort default

`spatialSort` of `GPUPointInPolygonJoin`, `GPUNearestFeatureJoin` and `GPUBufferSelection` defaults to `true`
from 256 features and `false` below; pass `false` to turn it off. With shuffled features each encoding is
50–125x faster; with spatially coherent features it is about 1 ms slower. `GPUSpatialJoinPrepared`
uses the same threshold: sorting defaults on from 256 logical feature rows and off below it
(255 logical features: `false`; 256: `true`), including multipart points and lines.

`spatialSort` orders the features along a Hilbert curve (the default) or a Morton curve
(`spatialSortCurve: 'morton'`, a prop of the three joins above, type `SpatialSortCurve`) before the BVH build.
Results are identical; in paired runs the Hilbert joins were 10 to 15 percent faster (point-in-polygon,
nearest, buffer selection). The prepared join and `GPUZoneEvents` have no curve prop and use Hilbert.

`querySort` orders queries the same way: each point chunk, or the bounds centers of line/polygon queries, is
processed along a Hilbert curve over the root bounds of the feature BVH (non-finite points and empty bounds last,
equal keys in row order), so neighboring invocations walk
neighboring subtrees. Every query still reads and writes its own row, so all outputs are identical, including which
candidate pairs a `candidateCapacity` overflow drops. It is a prop of `GPUPointInPolygonJoin`, `GPUNearestFeatureJoin`
and `GPUBufferSelection` and defaults to `true` for point queries from 4096 queries and 256 features; line and polygon queries
default to `false` because the measured sort cost exceeds the traversal saving.
Lines default off: the measured 8192-line workload took 0.80 ms without sorting and 1.20 ms with sorting;
explicit `querySort: true` remains available. Point sorting stays enabled; polygon defaults are unchanged.
Line/polygon queries use the same source-row output addressing, including
chunked scratch and foot-point outputs; polygon query topology is packed to keep every pass within eight storage
bindings. Geometry sorting adds bounds and center passes. With a planar `radius` the second candidate pass keeps
input order.

`edgeSlabs` (`GPUPointInPolygonJoin`) makes the exact refine test only the edges of the y-slab at the point's height
in each polygon part, instead of every edge of the candidate feature: O(slab) per candidate rather than O(vertices).
Results are identical (all boundary rules, the half-open owner, malformed offsets as UNCERTAIN); the one exception is
a pair that the full walk reports UNCERTAIN only because an unrelated far edge defeats the exact orientation test
(terms more than 200 binary orders apart). It defaults on when `polygons.positions.length * candidateCapacity` is at
least 4,000,000 and features average 8 vertices. `new GPUSpatialJoinPrepared({geometry, edgeSlabs: true})` builds the
index with the BVH on every encoding, and joins given that handle share it within that encoding.
Slab counts are limited by GPU-computed
total vertical edge travel and the entry budget, including ring closure. This prevents jagged long rings
from overflowing the index and reverting to a full walk through indirect edge references.

#### How it computes

{/* lens:GPUPointInPolygonJoin */}

::lens{scenario=gpu-spatial-analysis.point-in-polygon-join rung=datum datum=0 caption="Observation 0: its candidate tracts and the one whose rings contain it."}

::lens{scenario=gpu-spatial-analysis.point-in-polygon-join rung=formula caption="Crossing parity with the half-open boundary rule; rings XOR within a part, parts OR."}

::lens{scenario=gpu-spatial-analysis.point-in-polygon-join rung=kernel caption="The classify kernel walks each candidate tract's edges with exact orientation signs."}

::lens{scenario=gpu-spatial-analysis.point-in-polygon-join rung=graph caption="Bounding-box candidates first, then one exact classify pass per candidate pair."}
#### How it computes

{/* lens:GPUNearestFeatureJoin */}

::lens{scenario=gpu-spatial-analysis.nearest-feature-join rung=datum datum=0 caption="Observation 0: its nearest library row and the planar distance in metres."}

::lens{scenario=gpu-spatial-analysis.nearest-feature-join rung=formula caption="Argmin of squared distance, ties to the lower feature row; distance published as metres."}

::lens{scenario=gpu-spatial-analysis.nearest-feature-join rung=kernel caption="The traverse kernel descends the library BVH, pruning boxes farther than the best so far."}

::lens{scenario=gpu-spatial-analysis.nearest-feature-join rung=graph caption="BVH build over libraries, then one traversal per observation."}
#### How it computes

{/* lens:GPUNearestFeatureWeights */}

::lens{scenario=gpu-spatial-analysis.nearest-feature-weights rung=datum datum=0 caption="Observation 0: its CSR row of three library neighbours and their weights."}

::lens{scenario=gpu-spatial-analysis.nearest-feature-weights rung=formula caption="Weight is one over distance (floored at 1 m) to the power; no row standardisation."}

::lens{scenario=gpu-spatial-analysis.nearest-feature-weights rung=kernel caption="The emit kernel writes each row's neighbour ids in ascending order with their weights."}

::lens{scenario=gpu-spatial-analysis.nearest-feature-weights rung=graph caption="Nearest search, per-row counts, offset scan, then the CSR emit pass."}

### `GPUPairGather`, `GPUOffsetExpansion`, and `GPUBoundsFilter`

`GPUBoundsFilter`: Chunked feature bounds use chunk-local bit masks and global block-count scratch.
Compact IDs retain global source-row order; output bindings and global block counts must fit the
device.

**`GPUPairGather`**: Copies attributes from the source rows of matched pairs onto the pair rows.

The reusable pair-gather example allocates a producer `candidateOverflow` word when the caller omits
it and passes that completed pair port to both join and gather. Final `overflow` reports final output
capacity loss; `candidateOverflow` independently reports lost intermediate candidates. The pair-gather,
predicate-join and prepared-join examples mask histogram membership by `pairs.status.count` on every
submission, so unused allocation slots and stale IDs cannot contribute to tract bins after reuse.

::lens{scenario=gpu-spatial-analysis.pair-gather rung=result caption="Park area copied onto each park and tract pair, without clipping or apportioning."}

**`GPUOffsetExpansion`**: Expands CSR offsets into one row per member, with its owner index and local index.

::lens{scenario=gpu-spatial-analysis.offset-expansion rung=result caption="Observations grouped by tract receive their tract index and position within it."}

**`GPUBoundsFilter`**: Flags features whose bounds intersect, lie within, or contain an axis-aligned box: a bounds test, not exact geometry.

::lens{scenario=gpu-spatial-analysis.bounds-filter rung=result caption="Observations inside the Lincoln Park bounding box (border included) are selected."}

Dataframe joints around the joins. `GPUPairGather` consumes join row indices and columns, so it takes no
`spatialContext`.

- **`GPUPairGather`** merges join output with attribute columns (GeoPandas `sjoin` and `sjoin_nearest`
  merges, `distance_col`). The input is `pairs` (a `GPUSpatialPredicateJoin` pair output) or `neighbors`
  (the dense `[query, slot]` lists of `GPUNearestFeatureJoin`, flattened by stable compaction).
  Compact pair ports use `pairs.leftRows` and `pairs.rightRows` as input row offsets. The old
  `leftIds` and `rightIds` keys are rejected: application identities belong in attribute columns.
  With rows `[0, 1]`, attributes `[10, 20]` and caller IDs `[1, 0]`, gathering returns attributes
  `[10, 20]` and IDs `[1, 0]`; caller identity never chooses an attribute row.
  For nearest joins request `output.neighborRows` alongside `neighborIds` and wire
  `neighbors: {rows: neighborRows, counts: neighborCounts, capacity: neighborCapacity}`. Gather's
  `rows` are feature row indices, while `neighborIds` may contain application IDs. For feature
  attributes `[10, 20]`, a winning row 0 must gather 10 with either source IDs `[1, 0]` or
  `[100, 200]`. Using IDs as rows would instead gather 20 or a missing-value fill. The old
  `neighbors.ids` spelling is rejected, including in cast objects and spreads; use `neighbors.rows`
  for precomputed row-valued lists too. View types cannot detect an application-ID buffer
  deliberately assigned to `rows`, so preserve the producer's named row output when composing.
  `how: 'inner'` (the default) emits matched rows, and `'left'` appends the unmatched left rows
  (`unmatchedLeft`, from the same inner predicate join or an anti join; derived automatically for
  neighbors) with `rightRows = GPU_PAIR_GATHER_NO_ROW` and
  NaN or `0xffffffff` in the right and slot columns. `leftColumns` and `rightColumns` gather `float32` or
  `uint32` columns by row, and `slotColumns` carry slot-aligned columns such as `neighborDistances`. The
  output order is matched pairs then unmatched, which is not GeoPandas' left order (sort by
  `(left, right)` to compare). It is bounded by `output.leftRows.length`. `output.status.overflow`
  reports final row capacity loss, including input final-output flags and `overflowSources`.
  Supplied `pairs.status.candidateOverflow`, `unmatchedLeft.status.candidateOverflow` and
  `candidateOverflowSources` propagate separately to `output.status.candidateOverflow`; that
  destination is required when any candidate flag is supplied. Both flags are rewritten on every
  submission. Growing the gathered output cannot recover candidate loss.
  Gather captures the declared mode, port maps, columns and status fields once at construction,
  including inherited getters. Later caller mutations cannot change graph declarations; GPU
  buffer contents remain per-frame inputs. The recorded pair example preserves optional
  `requiredCount` and candidate status views when completing its pair port. The catalog example
  allocates separate final-overflow and candidate-overflow destinations for this composition.
  When consuming nearest lists, pass `neighbors.status` and `neighbors.candidateTruncated` as
  shown below. The nearest aggregate `overflow` includes tie truncation and BVH loss; gather maps
  it to **candidate** overflow because a larger gathered output cannot recover a lost neighbor.
  A zero neighbor count establishes absence only with sufficient producer evidence. With per-query
  `candidateTruncated`, zero-count queries with a nonzero flag are withheld from derived unmatched
  rows; unaffected zero-count queries can still be emitted. Without per-query evidence, any supplied
  global search loss conservatively withholds every derived unmatched row. Retained matched rows
  are still gathered. With no supplied diagnostics, gather trusts the lists as complete; it cannot
  discover an omitted producer flag. Legacy `candidateOverflowSources` remains supported, but
  these scalar sources have no query attribution and suppress all derived unmatched rows when set,
  even if per-query evidence is also supplied. `overflowSources` reports final row capacity loss
  and does not itself suppress known unmatched queries.
  Predicate anti-join `unmatched` ports and point-in-polygon/nearest compact `matches` ports use
  the same split: compact `overflow` is final capacity only, and `candidateOverflow` reports
  candidate/BVH loss and clears on every submission. Their separate whole-join aggregate diagnostic
  retains its documented meaning. Gather permits adjacent status words written by separate passes;
  outputs written in one pass must have disjoint aligned binding ranges.
  Chunked `GraphVectorView` inputs are unsupported.
- **`GPUOffsetExpansion`** gives the owner index and local index of every child row from an offsets view
  (`explode`, `get_coordinates(return_index=True)`, `get_parts(return_index=True)`) by binary search per row.
  Empty ranges yield no rows, `ownerMap` chains levels (vertex to ring to polygon to feature), and the
  output is bounded with `status.count`, `status.overflow` and `status.requiredCount`. Offsets monotonicity is not checked.
- **`GPUBoundsFilter`** (`spatialContext`: planar, or longitude/latitude with metric `'none'`, treated
  as plain coordinates with no antimeridian wrap, so a box needs `minX <= maxX`) is a `.cx`-style mask and compaction over `float32x4` feature bounds with a
  per-frame box (`parameters`, written with `getGPUBoundsFilterParameterValues`; the optional 0/1 mask is
  `output.outputMask`). Modes are `'intersects'`, `'within'` and
  `'contains'`, with closed comparisons and non-finite bounds rejected. It is bounds-level only, and OGC
  `envelope.within(box)` rejects bounds that only touch the border where this accepts them, so refine
  with `GPUSpatialPredicateJoin`.
  `GPUGeometryMeasures` planar bounds can be connected directly. Its geographic bounds unwrap
  longitude from each feature's first vertex, so normalize features and query to a common branch
  or test all relevant query intervals shifted by whole turns and union their masks. For example,
  bounds `[179,181]`, reversed-line bounds `[-181,-179]`, and one-turn-shifted bounds `[539,541]`
  need equivalent viewport intervals `[180.25,180.75]`, `[-179.75,-179.25]`, and `[540.25,540.75]`.
  Testing only `[-179.75,-179.25]` retains only the reversed line. `wrapX` splits a wrapped query
  into two rays; it does not move feature bounds onto the query's longitude branch. Use the
  filter's supported metric `'none'`, rather than passing a geographic measures metric context.

For wrapped interval packing, write `wrapX: true` with strictly reversed longitude endpoints
(`minX > maxX`), such as `[170,-170]`. Keep latitude endpoints ordered. An ordinary interval uses
`minX <= maxX` with `wrapX` omitted or false; a seam interval is not a broad `[-170,170]` box.
The four packed float32 values are `[minX,minY,maxX,maxY]` in the bounds' coordinate units; `wrapX`
authorizes reversed endpoints in the host packer, rather than adding a fifth parameter word.
The packer rejects endpoints that become nonfinite in float32 and a wrapped interval whose strict
inversion collapses after float32 conversion. For example, `minX: 1 + 2 ** -25, maxX: 1` cannot
encode a seam interval: both become 1. Supply endpoints that stay strictly reversed when encoded.

```ts
import {getGPUBoundsFilterParameterValues} from '@luma-contributors/analysis/gpu-spatial-analysis';

const wrappedBox = getGPUBoundsFilterParameterValues({
  minX: 170, minY: -10, maxX: -170, maxY: 10, wrapX: true
});
// Float32Array [170, -10, -170, 10]; the query covers both rays at the seam.
```

A predicate left gather needs only one spatial join. Upload query points `(1,1)`, `(0,1)`,
`(3,1)`, a square `[0,0]–[2,2]`, and polygon attribute `7` to the typed inputs below.
With strict `within`, the interior point matches, the boundary point does not, and the last point
is outside. On a complete run, gathered left rows are `[0,1,2]`, right rows are
`[0,GPU_PAIR_GATHER_NO_ROW,GPU_PAIR_GATHER_NO_ROW]`, values are `[7,NaN,NaN]`, and the live count is 3.

```ts
import {type GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  GPUSpatialPredicateJoin, GPUPairGather, type GPUSpatialPredicateJoinProps
} from '@luma-contributors/analysis/gpu-spatial-analysis';

declare const graph: GPUCommandGraph;
// Local factory from the readback-destinations example above, not a library export.
declare const createReadbackView: <Format extends 'uint32' | 'float32' | 'float32x2'>(
  graph: GPUCommandGraph, name: string, format: Format, length: number
) => GraphDataView<Format>;

declare const pointPositions: GraphDataView<'float32x2'>; // three query rows
declare const polygon: GPUSpatialPredicateJoinProps['right']; // one square
declare const polygonValue: GraphDataView<'float32'>; // one attribute row: 7
const createStatus = (name: string) => ({
  count: createReadbackView(graph, `${name}-count`, 'uint32', 1),
  overflow: createReadbackView(graph, `${name}-overflow`, 'uint32', 1),
  candidateOverflow: createReadbackView(graph, `${name}-candidate-loss`, 'uint32', 1)
});
const pairs = {
  leftRows: createReadbackView(graph, 'predicate-left', 'uint32', 3),
  rightRows: createReadbackView(graph, 'predicate-right', 'uint32', 3),
  status: createStatus('predicate-pairs')
};
const unmatched = {
  ids: createReadbackView(graph, 'predicate-unmatched', 'uint32', 3),
  status: createStatus('predicate-unmatched')
};
const uncertainCount = createReadbackView(graph, 'predicate-uncertain', 'uint32', 1);
const join = new GPUSpatialPredicateJoin({
  left: {kind: 'points', positions: pointPositions}, right: polygon,
  predicate: 'within', how: 'inner', candidateCapacity: 3,
  output: {pairs, unmatched, status: {uncertainCount}}
});
graph.add(join);
const gatheredValues = createReadbackView(graph, 'left-gather-values', 'float32', 3);
const gathered = {
  leftRows: createReadbackView(graph, 'left-gather-left', 'uint32', 3),
  rightRows: createReadbackView(graph, 'left-gather-right', 'uint32', 3),
  status: createStatus('left-gather')
};
graph.add(new GPUPairGather({
  pairs: join.outputPairs!, unmatchedLeft: unmatched, how: 'left',
  rightColumns: [{source: polygonValue, output: gatheredValues}], output: gathered
}));
// Require clear gathered.status.overflow, gathered.status.candidateOverflow and uncertainCount.
// Read only gathered.status.count rows from gatheredValues and both gathered row columns.
```

For a nearest left gather, preserve target **rows**, slot capacity, counts and both kinds of
search evidence from the same encoding. Here `nearestOutput` is the output map also supplied to
the producing `GPUNearestFeatureJoin` (`neighborCapacity: 8`). Request `neighborRows` separately
from application-valued `neighborIds`, even when IDs currently happen to equal rows.

```ts
import {type GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  GPUPairGather, type GPUPairGatherProps, type GPUNearestFeatureJoinOutput
} from '@luma-contributors/analysis/gpu-spatial-analysis';

declare const graph: GPUCommandGraph;
// Local factory from the readback-destinations example above, not a library export.
declare const createReadbackView: <Format extends 'uint32' | 'float32' | 'float32x2'>(
  graph: GPUCommandGraph, name: string, format: Format, length: number
) => GraphDataView<Format>;

declare const nearestOutput: GPUNearestFeatureJoinOutput & Required<Pick<
  GPUNearestFeatureJoinOutput, 'neighborRows' | 'neighborCounts' | 'candidateTruncated'
>>;
declare const rightColumns: GPUPairGatherProps['rightColumns'];
const neighborCapacity = 8;
// Each query emits at most eight matches OR one unmatched row, never both.
const rowCapacity = Math.max(1, nearestOutput.neighborRows.length);
const gatheredCount = createReadbackView(graph, 'gathered-count', 'uint32', 1);
const gathered = {
  leftRows: createReadbackView(graph, 'gathered-left', 'uint32', rowCapacity),
  rightRows: createReadbackView(graph, 'gathered-right', 'uint32', rowCapacity),
  status: {
    count: gatheredCount,
    overflow: createReadbackView(graph, 'gathered-final-loss', 'uint32', 1),
    candidateOverflow: createReadbackView(graph, 'gathered-search-loss', 'uint32', 1)
  }
};
graph.add(new GPUPairGather({
  how: 'left',
  neighbors: {
    rows: nearestOutput.neighborRows, counts: nearestOutput.neighborCounts,
    capacity: neighborCapacity, status: nearestOutput.status,
    candidateTruncated: nearestOutput.candidateTruncated
  },
  rightColumns, output: gathered
}));
// After submission, gatheredCount bounds the live prefix of every gathered row column.
// Final overflow means grow rowCapacity; candidate overflow means retry the nearest producer.
```

`output.status.count` is required and publishes the live gathered count, including appended
unmatched rows and clamped to output capacity. `output.status.matchedCount`, when requested, counts
matched input rows before clamping and excludes appended unmatched rows. Only `[0, gatheredCount)`
is readable. Slots beyond it are unspecified. Truncated ties can set gather candidate overflow
even when every query has a retained neighbor; do not interpret that flag as final gather capacity
loss.

The shared `how` spelling has operation-specific defaults:

| Operation | Default | Explicit modes |
|---|---|---|
| `GPUSpatialPredicateJoin` | `'inner'` | `'inner'`, `'anti'` |
| `GPUPairGather` | `'inner'` | `'inner'`, `'left'` |

`GPUKeyLookup` looks up keys and can publish each matched left row index once. Set
`compactMatchedLeftRows: true` and supply `output.rows` to receive those indices in ascending order:

```ts
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUKeyLookup, type GPUKeyLookupProps
} from '@luma-contributors/analysis/dataframe-analysis';

declare const graph: GPUCommandGraph;
declare const lookupPorts: Omit<GPUKeyLookupProps, 'compactMatchedLeftRows'>;
graph.add(new GPUKeyLookup({...lookupPorts, compactMatchedLeftRows: true}));
```

Requested lookup columns keep one slot per left row in source order.

Matched GeoPandas 1.2.0 `sjoin` (inner 11 pairs, left 32 rows) and `sjoin_nearest` with ties and
`distance_col`, and `explode` and `get_coordinates` indices, exactly.

#### How it computes

{/* lens:GPUPairGather */}

::lens{scenario=gpu-spatial-analysis.pair-gather rung=datum datum=46 caption="Tract 17031031400 (row 46): the park areas gathered onto its pairs."}

::lens{scenario=gpu-spatial-analysis.pair-gather rung=kernel caption="Each pair row reads its source row and copies the attribute into its slot column."}

::lens{scenario=gpu-spatial-analysis.pair-gather rung=graph caption="Slot assignment, one gather pass per attribute column, then publication of the gathered rows."}
#### How it computes

{/* lens:GPUOffsetExpansion */}

::lens{scenario=gpu-spatial-analysis.offset-expansion rung=datum datum=0 caption="The first observation in tract order: its tract and its local index within it."}

::lens{scenario=gpu-spatial-analysis.offset-expansion rung=kernel caption="One invocation per member binary-searches the offsets for its owner, then subtracts the start."}

::lens{scenario=gpu-spatial-analysis.offset-expansion rung=graph caption="A single expansion pass over the CSR offsets; no sort or scan is needed."}
#### How it computes

{/* lens:GPUBoundsFilter */}

::lens{scenario=gpu-spatial-analysis.bounds-filter rung=datum datum=0 caption="Observation 0: its position against the Lincoln Park box, and the mask bit."}

::lens{scenario=gpu-spatial-analysis.bounds-filter rung=formula caption="Within: every bound inside the box, borders included, compared in float32."}

::lens{scenario=gpu-spatial-analysis.bounds-filter rung=kernel caption="One invocation per feature compares its bounds with the box and writes one mask word."}

::lens{scenario=gpu-spatial-analysis.bounds-filter rung=graph caption="One mask pass, then a visibility workflow compacts and publishes the selected rows."}

{/* parameters: GPUBoundsFilter generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUBoundsFilter` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPUBoundsFilter`.

- Packer: `getGPUBoundsFilterParameterValues(settings)`, writes `GPU_BOUNDS_FILTER_PARAMETER_LENGTH` (4) `float32` values.

Settings of `getGPUBoundsFilterParameterValues`:

| Field | Type | Description |
|---|---|---|
| `minX` | `number` | Left edge. |
| `minY` | `number` | Bottom edge. |
| `maxX` | `number` | Right edge, at least `minX` unless `wrapX` is set. |
| `maxY` | `number` | Top edge, at least `minY`. |
| `wrapX` (optional) | `boolean` | Allows `minX > maxX`, meaning the box wraps through the longitude seam: it covers `x >= minX` and `x <= maxX` (for example `minX = 170, maxX = -170`). Intended for longitude/latitude contexts. Feature bounds must not wrap themselves. A wrapped box never satisfies `'contains'`, because a finite feature box cannot cover both sides of the seam. |

{/* /parameters */}

### `GPUBufferSelection`

Keep the observation and reference views distinct: `points.positions` contains rows to select;
`features.positions` contains station or bus-stop rows. For example, an observation at `(0, 0)`
is outside a 400 m buffer around a station at `(1000, 0)`. Binding the observation view to both
sides would create a self-match at zero distance. Size candidate capacity from the query view's
row count. The Near Transit snippet uses `observationPositions`, `stationPositions` or
`busPositions`, and separate `transitStarts`/`transitEnds` for line features.

Selects features within a radius of any reference feature, without building buffer polygons.

The compact `output.matches.status.overflow` reports final selection capacity loss. Optional
`output.matches.status.candidateOverflow` reports join candidate loss separately. `requiredCount`
counts matches reaching compaction and is a lower bound when candidates were lost.

::lens{scenario=gpu-spatial-analysis.buffer-selection rung=result caption="Observations within 300 m of a library are selected; distance is straight-line, not walking."}

No warning is possible because the distance is a GPU view; see [Precision of longitude/latitude positions](./concepts.md#precision-of-longitudelatitude-positions).

Use a one-row float32 `radius` view for the contributor and a numeric `radius` for the CPU function.
Both named operation forms take a descriptor such as `radius: {value: 'radius', length: 1}`. The
graph input port is named `distance`.

Radius eligibility uses an inclusive geometric distance test before winner ranking. Near the
threshold, the search certifies ambiguous planar radius comparisons against the original
coordinates; a float32 rank tie alone does not make a reference eligible. Among eligible
references, the minimum distance rank wins and tied ranks select the lowest reference row.
Published distance rounding happens afterward; caller IDs do not change the winner. For query
`(0,0)` and reference rows `(1,2**-14)` and `(1,0)`, radius 1 selects row 1, while radius 2 admits
both and selects row 0: the squared distances `1 + 2**-28` and `1` have the same float32 rank.
The named input's scalar remains in planar coordinate units.

Selects points within a per-frame distance (inclusive) of point or segment features. A
polyline is passed as consecutive segment rows. The contributor composes `GPUNearestFeatureJoin`. Unlike
the join's unordered `matches`, it writes a source-aligned 0/1 mask and stable IDs in ascending row
order (`output.outputMask`, `output.matches`), with a clamped count. `output.matches.status.overflow`
reports final output capacity loss; optional `output.matches.status.candidateOverflow` reports
join candidate loss separately. `points` is a `GPUPointGeometryPort` whose `sourceIds` become the match IDs.
Optional `output.nearestFeatureRows` contains the original feature row before application-ID
remapping; it replaces the misleading `nearestFeatureIds` spelling, which is rejected. Selection
uses those rows, so a feature ID of `0xffffffff` still selects a genuinely matched point. Feature
`sourceIds` are ignored by this selection operation; query `points.sourceIds` still identify the
selected query points in `matches.ids`.
`output.drawInstanceCount` (one `uint32` row, requires `output.matches`) receives the clamped selected count
from the publish node, so it can be bound as an indirect draw instance count. The distance is planar in
coordinate units by default; with `spatialContext: {coordinateSpace: 'longitude-latitude', metric:
'great-circle'}` it is a great-circle distance in the context's length unit (meters by default; kilometers
with `units: 'kilometers'`). The context is forwarded to `GPUNearestFeatureJoin`, whose section lists the
edge model, sphere radius and rejected combinations; `candidateCapacity` is ignored.
It measures distance, so `local-lambert-equal-area` frames throw. Optional `output.candidateTruncated` (one
packed `uint32` per point) and `output.status.candidateOverflow` pass the join's candidate status through: a
flagged point may be left out although a feature lies within the distance (see **Candidate overflow** above).

```ts
import {type GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import type {GPUCompactOutput} from '@luma-contributors/analysis/analysis-core';
import {GPUBufferSelection} from '@luma-contributors/analysis/gpu-spatial-analysis';

declare const graph: GPUCommandGraph;
// Local factory from the readback-destinations example above, not a library export.
declare const createReadbackView: <Format extends 'uint32' | 'float32' | 'float32x2'>(
  graph: GPUCommandGraph, name: string, format: Format, length: number
) => GraphDataView<Format>;

declare const points: GraphDataView<'float32x2'>;
declare const roadStarts: GraphDataView<'float32x2'>;
declare const roadEnds: GraphDataView<'float32x2'>;
declare const radius: GraphDataView<'float32'>; // one scalar, planar coordinate units
// matches is caller-owned and readback-capable; its required count/final-overflow views persist.
declare const matches: GPUCompactOutput;
const candidateOverflow = createReadbackView(graph, 'selection-candidate-loss', 'uint32', 1);
const outputMask = createReadbackView(graph, 'road-selection', 'uint32', points.length);
const selection = new GPUBufferSelection({
  points: {kind: 'points', positions: points},
  features: {kind: 'segments', starts: roadStarts, ends: roadEnds},
  radius, candidateCapacity: Math.max(1, points.length * 4), spatialSort: true,
  output: {outputMask, matches: {...matches, status: {...matches.status, candidateOverflow}}}
});
const completedMatches = selection.getMatches(graph);
graph.add(selection);
// After submission, read the caller-owned count/final-overflow words and candidateOverflow.
// Accept the mask and completedMatches.ids prefix only when BOTH overflow flags are zero.
// completedMatches supplies graph destinations for omitted views; those extra scratch views
// need a persistent destination or graph-side publication before any post-submission readback.
declare const selectionReadback: {
  count: Uint32Array; overflow: Uint32Array; candidateOverflow: Uint32Array;
  ids: Uint32Array; mask: Uint32Array;
};
for (const word of [selectionReadback.overflow, selectionReadback.candidateOverflow]) {
  if (word.length !== 1 || word[0] !== 0) throw new Error('Selection is incomplete; retry');
}
if (selectionReadback.count.length !== 1 || selectionReadback.count[0] > selectionReadback.ids.length ||
    selectionReadback.mask.length !== points.length) {
  throw new Error('Selection readback is missing its live count or row domain');
}
const selectedIds = selectionReadback.ids.slice(0, selectionReadback.count[0]);
const selectedMask = selectionReadback.mask;
```

#### How it computes

{/* lens:GPUBufferSelection */}

::lens{scenario=gpu-spatial-analysis.buffer-selection rung=datum datum=0 caption="Observation 0: distance to its nearest library and whether it is within 300 m."}

::lens{scenario=gpu-spatial-analysis.buffer-selection rung=formula caption="Selected when the nearest-library distance is between 0 and r, here 300 m."}

::lens{scenario=gpu-spatial-analysis.buffer-selection rung=kernel caption="The mask kernel turns each nearest distance into a selection bit."}

::lens{scenario=gpu-spatial-analysis.buffer-selection rung=graph caption="A nearest-library join within 300 m, a mask pass, then compaction of the selected rows."}
