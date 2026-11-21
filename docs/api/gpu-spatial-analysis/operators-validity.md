---
title: "GPU Spatial Analysis: Geometry validity & intersections"
sidebar_label: "Geometry validity & intersections"
description: "Geometry validity & intersections contributors in @luma-contributors/analysis/gpu-spatial-analysis."
---


# Geometry validity & intersections

Import from `@luma-contributors/analysis/gpu-spatial-analysis`.

### `GPUSegmentIntersection`

Finds every intersecting pair of line segments within one geometry, with the kind and point of each intersection.

::lens{scenario=gpu-spatial-analysis.segment-intersection rung=result caption="1,615 vertices of 721 street edges; points mark where segments meet or cross, coloured by kind."}

Maximum: each side has at most `getGPUSegmentIntersectionMaximumVertexCount(device)` (4194304 on a default device: the 32-byte segment-table row per vertex must fit one storage binding) `positions` rows; construction throws `segment-intersection left.positions rows must be at most 4194304` above it.

Planar only: `spatialContext` defaults to native planar coordinates and a longitude-latitude context throws; project to a planar frame first (`createGPULocalFrame`, `GPULocalFrameProjection`). Orientation signs are exact.

Finds every intersecting pair of line segments between two geometries (`left`, `right`) or within one
(`left` only), like turf `lineIntersect` and `kinks`, self-intersection tests and noding points. Each side
is a tagged line or polygon geometry port (`kind: 'lines'` or `'polygons'`, one kind per side). A segment is identified
by the index of its start vertex. Polygon rings close implicitly; rings with fewer than three vertices,
zero-length segments and segments with non-finite coordinates are skipped.
Optional line `featureOffsets` groups parts into logical features. `sameFeatureOnly` and
the feature columns use that grouping; ring columns remain line-part rows and pair rows
remain global start-vertex indices. Empty features therefore consume a feature row without
adding segment slots.

| Prop | Meaning |
| --- | --- |
| `left`, `right?` | Geometries. Omit `right` for self mode (`left < right`, adjacent segments skipped) |
| `sameFeatureOnly?` | Self mode: only same-feature pairs (per-feature self-intersection and ring crossings) |
| `output.pairs` | `{leftRows, rightRows, status: {count, overflow, candidateOverflow?, requiredCount?}}`, sorted by `(left, right)`; capacity is `leftRows.length` |
| `output.kinds?` | `GPU_SEGMENT_INTERSECTION_KIND`: proper 1, touch 2, collinearTouch 3, overlap 4, uncertain 5 |
| `output.points?`, `output.endPoints?` | The intersection point; for an overlap, the two ends of the shared span |
| `output.leftFeatures?`, `output.rightFeatures?`, `output.leftRings?`, `output.rightRings?` | Feature and ring (or line) rows of each segment |
| `output.status.uncertainCount?` | Count of uncertain pairs within capacity |
| `spatialSort?` | Morton sort before the BVH build; on by default from 1024 vertices; the output is identical |
| `leafCapacity?` | Power-of-two BVH leaves, by default the next power of two of the right vertex count |

`proper` means the interiors cross at one point. `touch` is one shared point where an endpoint of one lies
on the other (non-parallel), `collinearTouch` is collinear with one shared endpoint, and `overlap` is
collinear sharing positive length. The pipeline is a segment table, a `GPUBVH` over the right segments, a
counted and scanned probe, exact classification and a bounded sorted output; nothing is read back.
On overflow the output keeps a sorted prefix (`status.count = min(total, capacity)`, `status.overflow = 1`, `status.requiredCount`
unclamped), and the columns beyond `status.count` are unspecified. `status.uncertainCount` and the classification
columns cover pairs within capacity only. A collinear chain of `n` segments lists all `n^2` overlap pairs.
`output.pairs.status.overflow` describes final pair capacity only. BVH leaf loss is independent,
reported in `output.pairs.status.candidateOverflow`. Both supplied words are rewritten on each
encoding; `requiredCount` is a lower bound when BVH leaves were lost. On clean
reuse both flags are zero. If `leafCapacity` is below the indexed geometry's vertex-slot
count, `candidateOverflow` is required: actual segment counts are GPU data, so construction
cannot guarantee those slots suffice. Nested consumers that omit the diagnostic must use
enough BVH slots or construction refuses that configuration. `GPUPairGather` preserves both causes.
Geometry descriptors, options, pair/column maps and nested status destinations are captured at
construction. Removing or replacing a caller-map field later cannot change the validated bindings;
the contents of the retained buffers remain dynamic.

```ts
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {GPUSegmentIntersection} from '@luma-contributors/analysis/gpu-spatial-analysis';

// Fragment inputs: all views belong to graph; pair/attribute capacities agree.
declare const graph: GPUCommandGraph;
declare const positions: GraphDataView<'float32x2'>;
declare const lineOffsets: GraphDataView<'uint32'>;
declare const leftRows: GraphDataView<'uint32'>;
declare const rightRows: GraphDataView<'uint32'>;
declare const count: GraphDataView<'uint32'>;
declare const overflow: GraphDataView<'uint32'>;
declare const kinds: GraphDataView<'uint32'>;
declare const points: GraphDataView<'float32x2'>;
graph.add(new GPUSegmentIntersection({
  left: {kind: 'lines', positions, lineOffsets},
  output: {pairs: {leftRows, rightRows, status: {count, overflow}}, kinds, points}
}));
```

#### How it computes

{/* lens:GPUSegmentIntersection */}

::lens{scenario=gpu-spatial-analysis.segment-intersection rung=datum datum=0 caption="Inspect pair 0, the first intersecting segment pair in the sorted output."}

::lens{scenario=gpu-spatial-analysis.segment-intersection rung=kernel caption="Read the classification kernel built on exact orientation tests."}

::lens{scenario=gpu-spatial-analysis.segment-intersection rung=graph caption="Follow the segment table, BVH build, counted probe and classification."}

### `GPUGeometryValidity`

Point and line inputs are single-part feature ports. Multipart point/line `featureOffsets` are
rejected before output row validation; use feature-aware joins or measures when grouping is needed.

Computes per-feature diagnostic issue codes and optional nonzero-usable validity for polygons, lines or points.

::lens{scenario=gpu-spatial-analysis.geometry-validity rung=result caption="102 park polygons shaded by their issue codes; the orientation convention is ignored."}

Maximum: `polygons.positions` has at most `getGPUSegmentIntersectionMaximumVertexCount(device)` (4194304 on a default device: the 32-byte segment-table row per vertex must fit one storage binding) rows.

Planar only: `spatialContext` defaults to native planar coordinates and a longitude-latitude context throws; project to a planar frame first (`createGPULocalFrame`, `GPULocalFrameProjection`). Orientation signs are exact.

A per-feature validity bitmask for polygons and multipolygons (the four-level `GPUPolygonGeometryPort` layout), like
PostGIS `ST_IsValidDetail` and Shapely `explain_validity`. `output.issueCode` holds diagnostic bits;
`decodeGPUGeometryValidityIssueCode` returns named flags and `hasStructuralIssue`, which describes
observed evidence only. A zero issue code cannot certify validity when checks are incomplete.
Optional `output.validity` is 1 when the implemented structural checks pass, otherwise 0. Orientation
and repeated-vertex advisories leave it 1. Polygon intersection overflow suppresses every validity row
because the missing intersections cannot be assigned to particular features. This flag does not
certify the unsupported checks listed below. **Breaking (2026-11-17):** rename `output.mask` to
`output.issueCode`; feed `output.validity` to downstream input masks.
Diagnostic and status output ranges must not overlap one another or retained geometry inputs.
Supplied `status.overflow` and `status.intersectionCount` scalars are reset to zero on every
point/line submission because those kinds do not perform intersection checks. Kepler marks
zero-issue rows `incomplete` when polygon intersection overflow persists after retries. The
explorer counts published `validity` rows, including advisory-only rows, and withholds its valid
polygon count while checks are incomplete.
The website hover and showcase map use the published usability rows too: an unobserved structural
defect is not a pass. With zero issue code and usability zero they show an incomplete check;
observed structural defects remain visible. The showcase withholds “all valid” on overflow.
The website preserves issue bit 256 as `uncertain` in hover text and uses a separate “uncertain
predicate” class. It does not label an uncertifiable predicate as a repeated vertex.
The geometry-validity result view explicitly selects the `resultParks` polygon capture; full
results retain `overflow` and `intersectionCount` as separate scalar diagnostics below the map.

The bits of `GPU_GEOMETRY_VALIDITY_BIT` are `nonFinite` 1, `unclosedRing` 2 (explicit closure only),
`shortRing` 4 (fewer than 3 vertices once a closing duplicate is dropped; rows are counted, so `A, B, B, A` does not set it and fails through `selfIntersection` instead), `repeatedVertex` 8, `selfIntersection` 16, `crossingRings`
32 (a proper crossing or overlap between different rings of one feature; point touches are allowed),
`holeOutsideShell` 64, `badOrientation` 128 and `uncertain` 256. `GPU_GEOMETRY_VALIDITY_STRUCTURAL_MASK` is
every bit except `badOrientation` (a convention) and `repeatedVertex` (GEOS and OGC accept repeated
vertices, so it is reported but does not make a geometry structurally invalid).

| Prop | Meaning |
| --- | --- |
| `polygons`, `output.issueCode`, `output.validity?` | The input, diagnostic code and optional nonzero-usable flag (`featureCount` rows each) |
| `intersectionCapacity` | Capacity of the internal same-feature intersection list; valid data produces almost none |
| `output.status.overflow`, `output.status.intersectionCount?` | Aggregate intersection-list capacity or BVH candidate loss (the self-intersection and crossing bits may then be incomplete), and the unclamped count of intersections found |
| `ringClosure?` | `'implicit'` (default) or `'explicit'`, which sets `unclosedRing` |
| `orientation?` | `'counter-clockwise-shell'` (default), `'clockwise-shell'` or `'ignore'` |

`leafCapacity` may restrict the polygon segment tree below the vertex-slot count. Child candidate
loss is folded into the required public `status.overflow`; any aggregate loss suppresses every
`validity` row to zero. `intersectionCount` is only a lower bound when candidates were lost.

`lines` (`{kind: 'lines', positions, lineOffsets}`) and `points` replace `polygons` for the other kinds
(Shapely `is_valid`, `is_valid_reason` on lines and points). Lines report `nonFinite` and `tooFewPoints`,
points report `nonFinite`; neither needs an intersection list.
Both accept optional `featureOffsets`. Multipart diagnostics are the bitwise union of their
parts' issues; a feature is usable only when every part is usable. Empty features and empty
line parts are valid; duplicate points in a multipoint are valid. A nonempty line part without
two distinct vertices sets `tooFewPoints` (512). Parts are checked independently, so a second
valid part cannot hide a collapsed or nonfinite part. Both status scalars are rewritten to zero.

Not checked: interior connectivity (holes touching at two points), nested holes, nested polygons of a
multipolygon, and rings that leave the shell only through boundary vertices. Repeated vertices are reported
although OGC accepts them. Hole containment tests one off-boundary vertex per hole.

```ts
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  GPUGeometryValidity,
  type GPUGeometryValidityProps
} from '@luma-contributors/analysis/gpu-spatial-analysis';

// Fragment inputs: issueCode and validity each have one row per polygon feature.
declare const graph: GPUCommandGraph;
declare const polygons: NonNullable<GPUGeometryValidityProps['polygons']>;
declare const issueCode: GraphDataView<'uint32'>;
declare const validity: GraphDataView<'uint32'>;
declare const overflow: GraphDataView<'uint32'>;
graph.add(new GPUGeometryValidity({
  polygons,
  output: {issueCode, validity, status: {overflow}},
  intersectionCapacity: 1024
}));
// validity can be used as the next contributor's input mask.
// Inspect issueCode with decodeGPUGeometryValidityIssueCode; inspect overflow for completeness.
```

#### How it computes

{/* lens:GPUGeometryValidity */}

::lens{scenario=gpu-spatial-analysis.geometry-validity rung=datum datum=0 caption="Inspect park 0, Lincoln (Abraham) Park, and its issue code."}

::lens{scenario=gpu-spatial-analysis.geometry-validity rung=kernel caption="Read the per-ring checks that set diagnostic issue bits."}

::lens{scenario=gpu-spatial-analysis.geometry-validity rung=graph caption="Follow the segment table, BVH, same-feature intersections and issue-code assembly."}

### `GPUGeometryPredicates`

Evaluates per-feature predicates and counts such as simplicity, vertex count and hole count for points, lines or polygons.
`geometry` and `other` use `GPUGeometryPredicatesGeometry`: single points/lines, optional
point/line `featureOffsets` grouping parts into logical features, or the polygon/multipolygon
hierarchy. Every output row belongs to a logical feature. Entire coordinate columns may be empty
when offsets still describe at least one logical feature. Empty multipart features have zero
coordinates/parts, `isSimple=1`, other unary flags zero, and compare equal to identically structured
empty features; unique offsets and completeness counters are zero. Measures report zero length
and vertex count, NaN centroid/bounds and `0xffffffff` extreme rows, including wholly empty columns.
Logical emptiness does not remove the graph buffer's backing-row requirement: an imported
zero-row `float32x2` view still needs at least eight physical bytes, as required by luma.gl's
aligned storage-binding contract. Keep its logical length zero.

::lens{scenario=gpu-spatial-analysis.geometry-predicates rung=result caption="102 park polygons shaded by stored vertex count; simplicity and hole counts are captured alongside."}

Maximum: for non-point geometry with `isSimple`, `isRing` or either intersection status word, `geometry.positions` has at most `getGPUSegmentIntersectionMaximumVertexCount(device)` (4194304 on a default device: the 32-byte segment-table row per vertex must fit one storage binding) rows.

Planar only: `spatialContext` defaults to native planar coordinates and a longitude-latitude context throws; project to a planar frame first (`createGPULocalFrame`, `GPULocalFrameProjection`).

Both spatial-context descriptors, including nested frame metadata, are captured once before
validation. Shared input contexts retain a shared captured descriptor; later caller mutations do
not change the retained metric or frame. GPU buffer identities remain dynamic resources.

Per-feature columns for points, linestrings or polygons (Shapely `is_simple`, `is_ring`, `is_closed`,
`is_ccw`, `get_num_points`, `count_coordinates`, `get_num_interior_rings`, `get_num_geometries`,
`extract_unique_points`, `equals_exact`, `equals_identical`). Each output is optional and `uint32` (0 or 1
for booleans), all in the `output` bag: `isSimple`, `isRing`, `isClosed`, `isCcw`, `numPoints`,
`numCoordinates`, `numInteriorRings` and `numGeometries`.

- `isSimple` and `isRing` require `intersectionCapacity` and `output.status.overflow` at
  construction for every geometry kind, points included. For non-point geometry they use
  `GPUSegmentIntersection` in self mode; overflow suppresses both dependent outputs to 0 for every feature.
  Multipoints are simple iff no point repeats. Multiline parts may meet only at endpoints of
  two open parts; crossings, overlaps and endpoint-to-interior contacts are not simple.
  `intersectionCount` includes nonadjacent same-feature segment intersections, including allowed
  contacts between open parts. Point simplicity uses sorted unique points and writes zero status.
  Either `output.status.overflow` or `output.status.intersectionCount` may also be requested
  independently of predicate columns. Every such request requires `intersectionCapacity` (an
  integer in `[1, 4294967295]`) at construction, points included. Non-point requests run the
  segment intersection stages. Point-only requests validate the capacity but use a dedicated scalar writer that rewrites
  each supplied word to zero on every submission; they do not build a segment tree or inherit its
  vertex limit. For non-point geometry, `leafCapacity` below the vertex-slot count requires an
  observable `output.status.overflow`, even for a status-only request. This word aggregates final
  pair-capacity and intermediate BVH loss; `intersectionCount` is a lower bound after candidate loss.
  A polygon is simple when no ring meets itself
  (Shapely 2.1). `isCcw` follows the GEOS `Orientation::isCCW` rule (the JTS 1.19 flat-top algorithm) for
  linestrings and is 0 for other kinds, as in Shapely.
- `other` (same kind and feature count) enables `equalsExact` (per-frame `tolerance` through
  `getGPUGeometryPredicatesParameterValues`) and `equalsIdentical`, which compare feature `i` with
  feature `i`. Coordinates and tolerance are stored as f32. Structural comparison preserves part counts and each part's
  vertex count and order; an equal point set with a different split does not compare equal.
  Single and multipart geometry types are distinct even when a multipart has one part.
  `equalsExact` uses Euclidean distance in planar position units. Canonical equality
  (`GPUCanonicalGeometryEquality`, `GPUCanonicalPolygonEquality`) instead defaults to
  `toleranceMetric: 'per-coordinate'`; choose `'euclidean'` there to use the same distance
  neighborhood while ignoring canonicalized ordering. Euclidean comparisons certify the inclusive
  boundary against the stored coordinates and tolerance using integer products: displacement
  `(13,84)` matches tolerance `85` and fails at the preceding float32 value `84.99999237060547`.
  At tolerance 1, displacement `(0.75,0.75)`
  passes per-coordinate comparison but fails Euclidean comparison (`sqrt(1.125) > 1`).
- Optional `output.relate` evaluates aligned `(row,row)` pairs through the spatial join's
  existing DE-9IM kernel, including feature unions and mod-2 multiline boundaries. Unlike
  structural equality, `output.topological.equals` accepts different part splits. Topology-only
  requests require the same feature count but permit different kinds. The packed matrix and
  bit-31 uncertainty flag use the join format. Named `topological` columns support equals,
  intersects, contains, within, covers, coveredBy, touches, crosses, overlaps and containsProperly;
  uncertain results are suppressed. Valid empty operands retain their exterior cells: two empty
  features give `FFFFFFFF2` (packed `196608`) and topological equality 1; empty/open-line pairs
  give `FFFFFF102` (`221184`), and the reverse gives `FF1FF0FF2` (`197664`). Invalid pairs remain
  suppressed to zero. These outputs add work only when requested. Writable columns, including
  diagnostics and named predicates, must have disjoint logical byte ranges. Columns written in
  separate passes may share adjacent words (for example `topological.equals` at byte 0 and
  `topological.touches` at byte 4). Co-bound writable columns must also have disjoint aligned
  binding ranges (`equalsExact`/`equalsIdentical`, and non-point intersection status words).
  A public output read by another output's dispatch must use a separate buffer, including
  `relate` read by named topology columns and `isSimple` read by `isRing`.
- `output.uniquePositions` and `output.uniqueOffsets` give `extract_unique_points`: first-occurrence order, as Shapely,
  with `-0` equal to `0`.
- Multilines have `numPoints`, `isRing` and `isCcw` equal to 0, as Shapely. `isClosed` is true only
  for a nonempty multiline whose every part is closed. `numCoordinates` sums stored vertices
  over parts, while `numGeometries` counts parts, including zero for an empty multipart feature.
- Limits: one thread per feature in the row-wise kernels (a huge single feature is serial); the
  self-mode look-ahead is 1024; an empty polygon gives 0 geometries where Shapely gives 1. Open:
  `equals_exact(normalize=True)`.

#### How it computes

{/* lens:GPUGeometryPredicates */}

::lens{scenario=gpu-spatial-analysis.geometry-predicates rung=datum datum=0 caption="Inspect park 0, Lincoln (Abraham) Park: vertex count, simplicity and holes."}

::lens{scenario=gpu-spatial-analysis.geometry-predicates rung=kernel caption="Read the per-feature column kernels."}

::lens{scenario=gpu-spatial-analysis.geometry-predicates rung=graph caption="Follow the count columns and the same-feature segment intersection for simplicity."}

{/* parameters: GPUGeometryPredicates generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUGeometryPredicates` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPUGeometryPredicates`.

- Packer: `getGPUGeometryPredicatesParameterValues(settings?)`, writes `GPU_GEOMETRY_PREDICATES_PARAMETER_LENGTH` (4) `float32` values.

Settings of `getGPUGeometryPredicatesParameterValues`:

| Field | Type | Description |
|---|---|---|
| `tolerance` (optional) | `number` | `equalsExact` tolerance: two vertices match when their distance is at most this value, in position units. Defaults to `0` (vertices must coincide). |

{/* /parameters */}

### `GPUMinimumClearance`

Multiline `geometry.featureOffsets` groups line parts into logical output rows. Clearance
considers vertices and segments in every part of that feature, including distances between
different parts. An empty feature reports Infinity, a NaN clearance line and vertex row
`0xffffffff`. Line-part boundaries still determine adjacency; no edge bridges two parts.
Entire all-empty columns are accepted when feature offsets still declare logical rows.
Distinct vertices in collapsed line parts participate: parts collapsed at x=0 and x=9 have
clearance 9, while one distinct point retains infinite clearance. Equal vertex-candidate distances
choose the lowest original partner row, so spatial sorting cannot change the witness. For
collapsed parts at x=`0,1,-1`, the winning row is 0 and the line is `[0,0,1,0]` with either sort mode.
The diagonal tie at `(0,0)`, `(1,1)`, `(-1,-1)` likewise selects `[0,0,1,1]` with distance
`sqrt(2)` under either sort mode; conservative squared BVH bounds retain equal-distance candidates.

Measures the smallest distance a single vertex could move before each feature becomes invalid, and which vertex sets it.

::lens{scenario=gpu-spatial-analysis.minimum-clearance rung=result caption="102 park polygons shaded by minimum clearance in metres."}

Maximum: `geometry.positions` has at most `getGPUSegmentIntersectionMaximumVertexCount(device)` (4,194,304 on a default device) rows, because the 32-byte segment table per vertex must fit one 128 MiB binding.

`leafCapacity` (optional, a power of two) sets the BVH leaf slots over the segments and defaults to the next
power of two of the vertex count, which never overflows. A smaller value drops the trailing segments from the BVH
(in Morton order when `spatialSort` is on), so clearances may come out too large. To detect this, pass
`output.status.candidateOverflow`, a one-row `uint32` view that receives `1` when segments were dropped and `0`
otherwise. Without that view an undersized `leafCapacity` is silent.

Planar only: a longitude/latitude `spatialContext` throws at construction; project first with
`createGPULocalFrame` and `GPULocalFrameProjection`. It measures distance, so `local-lambert-equal-area` frames throw.

Per-feature `shapely.minimum_clearance` and `minimum_clearance_line` for linestrings and
(multi)polygons: the smallest distance one vertex could move to make the feature invalid, as the minimum
over vertices of the distance to every other vertex and to every segment that does not have that vertex as
an endpoint (GEOS rule, repeated points ignored). Props: `geometry` (lines or polygons), and an `output`
with at least one of `clearances` (`float32`, `Infinity` when none), `lines` (`float32x4`, vertex and
nearest point, NaN when none) and `vertexIds` (`uint32` row of the fragile vertex, `0xffffffff` when none). A segment
table and BVH feed one branch-and-bound probe per vertex, restricted to its own feature, then one 64-lane
workgroup per feature reduces its vertex range to the lexicographic minimum of (distance, vertex row) with a
fixed tree and no atomics, so results are deterministic and ties go to the lowest vertex row. Parity with Shapely
2.1.2 is within 2e-5 relative on 52 multipolygons and 69 linestrings; line end points can come out in the
opposite order and tied pairs can differ from GEOS (the distance never does). Coordinates are
origin-relative f32, so tiled copies shifted by thousands of units agree to about 4e-4. Not provided:
multipoints.

#### How it computes

{/* lens:GPUMinimumClearance */}

::lens{scenario=gpu-spatial-analysis.minimum-clearance rung=datum datum=0 caption="Inspect park 0, Lincoln (Abraham) Park: clearance and fragile vertex."}

::lens{scenario=gpu-spatial-analysis.minimum-clearance rung=formula caption="Shortest vertex-to-segment distance within a polygon, excluding the vertex's own edges."}

::lens{scenario=gpu-spatial-analysis.minimum-clearance rung=kernel caption="Read the BVH probe that finds each vertex's nearest other segment."}

::lens{scenario=gpu-spatial-analysis.minimum-clearance rung=graph caption="Follow the segment table, BVH, vertex probe and per-feature minimum."}

### `GPUCoverageValidity`

Checks polygons as one coverage for gaps, overlaps, crossings and unmatched shared edges, reported per ring segment.

::lens{scenario=gpu-spatial-analysis.coverage-validity rung=result caption="199 census tracts checked as one coverage with a 1 m gap width; shading counts invalid edges."}

Maximum: `positions` has at most `getGPUSegmentIntersectionMaximumVertexCount(device)` (4194304 on a default device: the 32-byte segment-table row per vertex must fit one storage binding) rows.

Polygon coverage validity (Shapely `coverage_is_valid`, `coverage_invalid_edges(gap_width)`). One BVH over
all segments writes a per-segment flag word (`unmatched`, `crossing`, `gap`, `overlap`, `uncertain` bits),
optional per-feature counts (`featureInvalidCounts`), a total (`status.invalidCount`) and `isValid`.
`status.invalidCount` rewrites one `uint32` component on every submission; any caller-provided
trailing storage is preserved. The input is the four-level polygon layout (`positions`, `featureOffsets`, `polygonOffsets`, `ringOffsets`); segments of one feature are never compared. `gapWidth` is per-frame through
`getGPUCoverageValidityParameterValues`. The output has one row per vertex and cannot overflow. Matched
edges are skipped and flagged only when both interiors are on the same side, and a T-junction touch counts.
The gap rule is the GEOS one (`InvalidSegmentDetector::isNearlyParallel`): each segment is projected onto the
other and clipped to it, and the pair is a gap when both clipped projections are longer than `gapWidth`
(strict) and their matching ends are within `gapWidth` of each other (inclusive). Staggered segments that run
side by side over only part of their length, such as `(0, 0)-(10, 0)` beside `(5, 0.0625)-(15, 0.0625)` at
`gapWidth` 0.125, are flagged (until 2026-11-11 they were not: the rule then asked for both ends of the
shorter segment within `gapWidth` of the longer). The rule is evaluated in float32 relative to the start of
the tested segment, so a distance or projected length equal to `gapWidth` to within float32 rounding can fall
on either side. With Shapely 2.2.0 (GEOS 3.14.1) the invalid segment sets are equal on the 28 curated scenes
(40 gap cases) and on the 222 cases of `test/oracles/fixtures/coverage-validity-gaps.json` (staggered
rectangles, disjoint pairs and jittered triangulations, near the origin and at projected-metre coordinates).
A matched edge lying inside a third polygon is not flagged (that polygon's own segments are). Ring winding
is the exact turn at each ring's lexicographically
smallest vertex, so a sliver whose float32 area is zero is still oriented. Every orientation sign is exact
for the f32 vertices, including the winding number of the interior probe, which is the shared symbolic side
probe (the point beside the exact midpoint of the segment, classified from the exact signs of the segment's
endpoints), so a segment of a polygon thinner than one float32 step is not read as lying inside its
neighbor. A pair with no decidable orientation (non-finite input) sets the `uncertain` bit. The operator is planar: `spatialContext` defaults to native
planar coordinates and `longitude-latitude` throws (project first).

#### How it computes

{/* lens:GPUCoverageValidity */}

::lens{scenario=gpu-spatial-analysis.coverage-validity rung=datum datum=0 caption="Inspect tract 17031010100: its invalid edge count."}

::lens{scenario=gpu-spatial-analysis.coverage-validity rung=kernel caption="Read the segment-pair kernel that sets the flag bits."}

::lens{scenario=gpu-spatial-analysis.coverage-validity rung=graph caption="Follow the segment table, BVH probe and per-feature counts."}

{/* parameters: GPUCoverageValidity generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUCoverageValidity` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPUCoverageValidity`.

- Packer: `getGPUCoverageValidityParameterValues(parameters)`, writes `GPU_COVERAGE_VALIDITY_PARAMETER_LENGTH` (4) `float32` values.

Settings of `getGPUCoverageValidityParameterValues`:

| Field | Type | Description |
|---|---|---|
| `gapWidth` | `number` | Largest gap, in position units, that counts as an error: Shapely `gap_width`. Two segments of different polygons that run nearly parallel within this distance are reported. `0` disables gap detection. Must be finite and non-negative. |

{/* /parameters */}
