# Library review: line geometry and constructive or descriptive geometry

Date: 2026-11-09. Branch `upstream-first`, working tree as is. Read-only review. Paths are relative
to `modules/analysis/src` unless they start with `dev-docs/`.

## 1. Scope and method

Scope: `gpu-spatial-analysis/{line-clip, line-merge, line-segmentize, line-simplification,
line-split, linear-referencing, rectangle-clip, minimum-bounds, minimum-clearance, label-point,
shape-descriptors, outline-geometry, offset-expansion, group-geometry, grid-generators, shared}` and
the barrel `gpu-spatial-analysis/index.ts`. 58 files, 19,066 lines. 23 exported contributor
classes, no `add*Recipe` functions.

Contract references read in full: `analysis-core/contracts/ports.ts` (250 lines),
`analysis-core/kernels/gpu-contributor-types.ts` (51), `resolveGPUSpatialContext` in
`analysis-core/contracts/spatial-context.ts:260-300`, `analysis-core/kernels/wgsl-common.ts:39-80`.

| Depth | Files |
| --- | --- |
| Read in full | `line-segmentize/{gpu-line-segmentize, line-geometry-output, line-segmentize-parameters, line-segmentize-types, balanced-search}.ts`, `line-clip/gpu-line-clip-by-polygon.ts`, `line-split/gpu-line-split.ts`, `line-merge/gpu-line-merge.ts`, `line-simplification/{gpu-line-simplification, line-simplification-parameters}.ts`, `offset-expansion/gpu-offset-expansion.ts`, `minimum-bounds/gpu-minimum-bounds.ts`, `shape-descriptors/gpu-shape-descriptors.ts`, `linear-referencing/gpu-linear-referencing.ts`, `shared/{hash-uint32, bin-index-wgsl, complementary-error-function, spatial-wgsl-point-helpers}.ts`, every `index.ts` in scope |
| Read the props, doc and dispatch code; kernels sampled | `line-segmentize/{line-segmentize-kernels (1-405), gpu-great-circle-arcs, gpu-line-smooth, gpu-line-chunk}.ts`, `linear-referencing/gpu-line-locate.ts (1-270)`, `rectangle-clip/gpu-rectangle-clip.ts`, `minimum-clearance/gpu-minimum-clearance.ts (1-420)`, `label-point/gpu-label-point.ts (1-300, 370-380)`, `outline-geometry/{gpu-outline-geometry, gpu-offset-curve}.ts (1-300 each, 380-427)`, `group-geometry/gpu-group-geometry.ts`, `group-geometry/gpu-group-convex-hull.ts (1-140)`, `grid-generators/{gpu-grid-generator (30-330, 600-690), gpu-shape-generator (1-240)}.ts`, `line-clip/gpu-shared-paths.ts (30-200, 420-660)` |
| Grep only (dispatch shapes, atomics, loops) | `line-simplification/{line-simplification-kernels, visvalingam-kernels}.ts`, `line-segmentize/{line-chunk-kernels, line-smooth-kernels}.ts`, `shape-descriptors/shape-descriptors-kernels.ts`, `segment-intersection/segment-bvh.ts:150-170` |
| Not opened | `linear-referencing/linear-referencing-kernels.ts`, `rectangle-clip/rectangle-clip-kernels.ts`, `shape-descriptors/convexity-hull.ts`, `shared/triangle-edge-adjacency.ts`, the rest of `gpu-group-convex-hull.ts` |

Out-of-scope files opened only to check overlap: `geometry-refinement/gpu-maximum-inscribed-circle.ts`
(header), `geometry-refinement/{gpu-offset-curve-components, gpu-line-loop-removal}.ts` (header),
`geospatial-analysis/gpu-point-buffer.ts` (header), `lon-lat-polygons/gpu-lon-lat-polygon-preparation.ts:250,469`.

Census check: the census numbers for these files match what I read (no bare asserts in contributor
files, `spatialContext` on 22 of 23 classes, `GPUParameterBuffer`-style parameters on 13). The
census "serialKernel" hits in line-clip and line-split are one-invocation finalize and scalar
kernels that do an `O(log n)` binary search, not serial data walks.

## 2. Contributor table

Abbreviations: P = parallelism, It = iteration. "Baked" lists numeric or structural props that
become WGSL literals (row counts and capacities are baked everywhere and are not repeated). All
contributors in scope take packed views only (no `GraphVectorView` in any file) and validate with
messaged helpers plus `throw new Error`.

| Symbol | File (lines) | Computes (reference) | P | Work | It | Hard limits | Parameters | Context | Status | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `GPULineSegmentize` | line-segmentize/gpu-line-segmentize.ts (294) | `ST_Segmentize`, turf densify | row + cooperative prefix | O(n + out), scan | none | `maximumPieceCount` 1024/segment, design, silent clamp (finding G6) | buffer: max length; baked: radius, closedLines, piece cap | planar or great-circle | count, overflow, requiredCount, lineCount | conforming |
| `GPUGreatCircleArcs` | line-segmentize/gpu-great-circle-arcs.ts (224) | turf `greatCircle` | row | O(pairs + out) | none | 256 segments/arc, design | buffer: max length, minimumSegments | great-circle only, default great-circle | as segmentize | merge into `GPULineSegmentize` |
| `GPULineSmooth` | line-segmentize/gpu-line-smooth.ts (303) | Chaikin, turf `polygonSmooth` | row per output slot | O(n·2^k) | fixed k levels | k ≤ 10 (:36), design | buffer: ratio; baked: iterationCount, closed | planar only | as segmentize | conforming |
| `GPULineChunk` | line-segmentize/gpu-line-chunk.ts (327) | turf `lineChunk`, `ST_LineSubstring` | row per piece and per slot | O(n + out), scan + binary search | none | none beyond capacity | buffer: chunk length, start/end measure; baked: mode | planar or great-circle | as segmentize | conforming |
| `GPULineSimplification` | line-simplification/gpu-line-simplification.ts (702) | Douglas-Peucker, TD-TR, Visvalingam (GEOS, simplify-js, mapshaper) | row; segmented argmax by atomics | O(n · rounds) | unrolled `maximumIterations` rounds, GPU gate | rounds ≤ 1024 (:92), finish span ≤ 128, VW radius ≤ 8 | buffer: tolerance; baked: method, metric, rounds, finishSpanLimit, radius | DP planar or great-circle; VW planar | GPUCompactOutput + converged, iterationCount | align |
| `GPULineSplit` | line-split/gpu-line-split.ts (901) | turf `lineSplit` self-noding | row; stable radix sort of events | O(n + k log k), BVH via `GPUSegmentIntersection` | none | pieces ≤ 4,194,304 (:63), inherent | none per frame | planar only | `GPUBoundedResultStatusPort` + vertex counts | conforming |
| `GPULineClipByPolygon` | line-clip/gpu-line-clip-by-polygon.ts (986) | GeoPandas `clip`, Shapely intersection/difference | row; sort; PIP join | O(n + k log k), BVH + grid | none | sub-pieces ≤ 4,194,304 (:137), inherent | baked: mode | planar only | bounded-result shape without candidateOverflow; uncertainCount | conforming |
| `GPUSharedPaths` | line-clip/gpu-shared-paths.ts (675) | Shapely `shared_paths`, turf `lineOverlap` | row, but one thread walks each run | O(n + k + Σ run length) per run head, unbalanced | none | intersections ≤ 2,796,202 (:47), inherent | none | planar only | count, vertexCount, overflow, required*, uncertain | rework (list ranking) |
| `GPULineMerge` | line-merge/gpu-line-merge.ts (678) | Shapely `line_merge` (directed), turf `lineMerge` | row; two `GPUSort`; Wyllie pointer jumping | O(L log L) | fixed `2·ceil(log2 L)` rounds | none | baked: directed | planar only (needlessly, finding C5) | count only (capacity guaranteed) | conforming |
| `GPULinearReferencing` | linear-referencing/gpu-linear-referencing.ts (492) | `ST_ClosestPoint`, `ST_LineLocatePoint`, turf `nearestPointOnLine` | row per query; BVH | O(q log n) | none | queries ≤ 3,728,270; positions ≤ 2^23 planar, 2,097,152 spherical (inherent) | radius as its own 1-row view (per frame); baked: sphere radius | planar or great-circle | overflow + `candidateCount` (not vocabulary) | align |
| `GPULineLocate` | linear-referencing/gpu-line-locate.ts (369) | turf `along`, `ST_LineInterpolatePoint`, `ST_LocateAlong` | row per event; binary search | O(n + e log n) | none | events ≤ 5,592,405, inherent | buffer: scale, offset | planar or great-circle | per-event status codes | conforming |
| `GPURectangleClip` | rectangle-clip/gpu-rectangle-clip.ts (386) | turf `bboxClip`, `clip_by_rect`, `ST_ClipByBox2D` | row; 4 Sutherland-Hodgman stages | O(n) per stage | fixed 4 stages | none | buffer: rectangle; baked: geometryType | planar only | as segmentize | align (polygon output layout) |
| `GPUMinimumBounds` | minimum-bounds/gpu-minimum-bounds.ts (720) | `oriented_envelope`, `minimum_bounding_circle`, `ST_LongestLine` | serial per group over its hull | O(n log n) hull + O(h) per group | none | rows < 2^24 (:30), hull ≤ 2048 default (:33) | baked: maximumHullVertices | planar only | overflow as a bit set | align |
| `GPUMinimumClearance` | minimum-clearance/gpu-minimum-clearance.ts (617) | `shapely.minimum_clearance(_line)` | row per vertex, BVH probe; cooperative per-feature min | O(n log n) | none | vertices ≤ 4,194,304, inherent | baked: leafCapacity, spatialSort | planar only | none (BVH overflow hidden, finding C7) | align |
| `GPULabelPoint` | label-point/gpu-label-point.ts (862) | polylabel | serial per small feature, workgroup per large | O(segments × ~1,800 samples) per feature, brute force | fixed rounds | refinementCandidates ≤ 16 | baked: grid sizes, rounds, candidates | planar only | per-feature `degenerate` flag | merge with `GPUMaximumInscribedCircle` |
| `GPUShapeDescriptors` | shape-descriptors/gpu-shape-descriptors.ts (365) | Polsby-Popper, convexity, elongation (momepy, esda) | serial per feature (moments, gift wrap) | O(n) moments, O(n·h) hull per feature | none | none | buffer: sliver threshold; baked: convexityMethod | planar only | none | rework |
| `GPUOutlineGeometry` | outline-geometry/gpu-outline-geometry.ts (427) | render-only stroke buffer | row | O(n·k) | none | none | buffer: distance; baked: joinSegments, radius | planar or lon/lat via local equirectangular scale | none | move (rendering helper) |
| `GPUOffsetCurve` | outline-geometry/gpu-offset-curve.ts (520) | `shapely.offset_curve` (raw) | row | O(n·k) | none | quadSegments ≤ 256 (8 KB private array), inherent | buffer: distance, mitre; baked: joinStyle, quadSegments, insideTurns | planar only | none | align (fixed-stride output) |
| `GPUOffsetExpansion` | offset-expansion/gpu-offset-expansion.ts (221) | Shapely `get_coordinates(return_index)`, `explode` index | row, binary search | O(n log owners) | none | none | none | none (correct: topological) | count, overflow, requiredCount | move (generic; upstream or `analysis-core`) |
| `GPUGroupGeometry` | group-geometry/gpu-group-geometry.ts (559) | cluster bounds, centers, medoid, ellipse | row; sorted segment reductions | O(n log n) + Σ g² medoid | none | medoid group ≤ 4096 (:46), design; rows < 2^24 | buffer: distribution parameters; baked: medoid cap | planar only | overflow | align |
| `GPUGroupConvexHull` | group-geometry/gpu-group-convex-hull.ts (1151) | monotone chain hull per label | row prefilter; serial chain per group | O(n log n) | fixed prefilter levels ≤ 4 | rows < 2^24, inherent (lattice) | baked: prefilter, extreme filter | planar only | overflow as a bit set | conforming |
| `GPUGridGenerator` | grid-generators/gpu-grid-generator.ts (756) | turf grids, GeoPandas `make_grid` | row per cell | O(cells); intersect O(cells × extent edges) brute force | none | cells × vertices ≤ 2^31 | buffer: origin, cell size; baked: gridType, columns, rows, extent | planar only | none (fixed size) | align |
| `GPUShapeGenerator` | grid-generators/gpu-shape-generator.ts (565) | turf `circle`, `sector`, `ellipse` | row per output vertex | O(features × segments) | none | maximumSegments ≤ 65,536 | buffer: segment count, radius scale | planar or spherical ("geodesic") | vertexCount | merge with `GPUPointBuffer` |

Verdicts: 8 conforming, 8 align, 2 rework, 3 merge, 2 move.

Counter-examples to the two-generation hypothesis: `GPULineMerge` and `GPUSharedPaths` were
written in the same style for the same kind of problem (chaining linked items), but only line merge
uses `O(log n)` pointer jumping; shared paths walks chains one thread per run. `GPUShapeDescriptors`
is a "baseline" contract-conforming contributor whose kernels are serial per feature. No file in
scope uses bare asserts, small fixture caps or one-line WGSL; this domain is all "baseline" style.

## 3. GPU quality findings

| # | What is there | Why it matters | Better design | Effort |
| --- | --- | --- | --- | --- |
| G1 | **CONFIRMED.** `GPUSharedPaths` sizes and writes every run by one thread following `next` pointers (`line-clip/gpu-shared-paths.ts:489-497`, `630-643`). | A road shared by two networks over 10,000 segments is one thread walking 10,000 spans twice; the rest of the GPU idles. | Reuse the Wyllie list ranking already in `line-merge/gpu-line-merge.ts:324-496` (fixed `log2` rounds, then one thread per span writes its slot). Extract it as a shared `analysis-core` list-ranking helper. | S-M |
| G2 | **CONFIRMED.** `GPUShapeDescriptors` integrates moments and runs gift-wrapping hulls with one invocation per feature (`shape-descriptors-kernels.ts:125-159`, `216-245`). `'auto'` switches to the balanced `GPUGroupConvexHull` only when the *average* feature has ≥ 256 vertices (`gpu-shape-descriptors.ts:227-231`). | One 200,000-vertex coastline among many small features stays on gift wrapping: O(n·h) on one thread. Moments have the same skew. | Decide on the largest feature, or always use `GPUGroupConvexHull` above a small size. Compute moment and area terms per edge and reduce with `analysis-core/kernels/load-balanced-for-each.ts` / `sorted-segment-sums.ts`. | S-M |
| G3 | **CONFIRMED.** Douglas-Peucker rounds find each interval's farthest row with `atomicMax` on f32 bits and `atomicMin` on the row (`line-simplification-kernels.ts:549, 568`). Rounds are unrolled up to `maximumIterations` (≤ 1024, five nodes each, `gpu-line-simplification.ts:606-627`). | In the first round every row of a long line hits the same address; contention serializes the round. 1024 rounds is about 5,000 encoded nodes, gated but still compiled and encoded. | Workgroup-local max then one atomic per workgroup and interval; or a segmented `GPUReduction` on a packed `(distanceBits, ~row)` key. Document the node budget in the props, or drive rounds with an indirect-dispatch loop. | M |
| G4 | **CONFIRMED.** `GPULabelPoint` scores every sample against every boundary segment of the feature (no index; doc at `gpu-label-point.ts:106-133`). About 1,800 samples per feature at defaults. | A 100,000-segment polygon costs about 1.8×10^8 distance evaluations per feature, spread over at most 64 lanes. | Use the best-first branch-and-bound of `geometry-refinement/gpu-maximum-inscribed-circle.ts` (same quantity) with a per-feature segment grid or the segment BVH from `segment-intersection/segment-bvh.ts`. Keep one class. | M |
| G5 | **CONFIRMED.** `GPUGridGenerator` intersect mask walks every ring and vertex of the extent per cell after a bounding-box early-out (`gpu-grid-generator.ts:618-690`). | A 1,000×1,000 grid against a 50,000-vertex country is 5×10^10 edge tests for the cells inside the box. | Bin extent edges with `GPUGridIndex` on the grid lattice (the grid is the index), or a per-row scanline crossing table; cells then test only edges in their bin. | S-M |
| G6 | **CONFIRMED.** The segmentize count kernel can flag clamped edges (`line-segmentize-kernels.ts:396-400`), and `lon-lat-polygons/gpu-lon-lat-polygon-preparation.ts:250` uses that flag. `GPULineSegmentize` and `GPUGreatCircleArcs` never pass it (`gpu-line-segmentize.ts:221-232`). | When an edge needs more than `maximumPieceCount` pieces, output segments are longer than `maximumSegmentLength` with no status. | Add `status.approximation` (shared vocabulary) to `GPULineGeometryOutput` and wire it. | S |
| G7 | **CONFIRMED.** `GPUOffsetCurve` writes `rowsPerVertex` rows per input vertex, padded (`gpu-offset-curve.ts:94-102`, output doc 112-121). Round joins at `quadSegments` 256 reserve 513 rows per vertex. | Memory is sized for the worst corner everywhere (17× the input at the default); downstream line contributors cannot read padded output without a compaction. | Count, `GPUScan`, emit into `GPULineGeometryOutput` (the pattern of `GPULineSegmentize`). | S |
| G8 | **CONFIRMED.** `GPUGroupGeometry` medoid costs are O(g) per row (`gpu-group-geometry.ts:498-502`), O(Σ g²) total, capped at 4096 rows per group; the select step is serial per group (`:543-550`). | Design cap; large clusters get no medoid. | Tile the pairwise sum per group in workgroup memory, select with a segmented argmin; or offer the geometric median (Weiszfeld) for large groups. | M |
| G9 | **CONFIRMED.** `GPUMinimumBounds` computes rectangle, circle, longest line and envelope with one thread per group over its hull (`gpu-minimum-bounds.ts:413-716`). | Bounded by `maximumHullVertices` (2048) and O(h) calipers; acceptable. Groups whose hull exceeds the cap become NaN. | Keep. Raise the default cap or fall back to a cooperative per-group workgroup for big hulls. | S |
| G10 | **SUSPECTED.** `GPULineMerge` sorts endpoints with two separate `GPUSort` passes plus a gather (`gpu-line-merge.ts:265-284`). | Works; duplicates `createStableLexicographicIndexSortNodes`, which line split and line clip use for the same tuple sort. | Use the shared lexicographic sort. | S |

Precision notes (all f32): Douglas-Peucker is bit-exact with a corrected division and square root
(`gpu-line-simplification.ts:263-271`); measure prefixes are Neumaier-compensated
(`line-segmentize-kernels.ts:101-114`); minimum bounds work in group-relative coordinates. These are
good. Linear-referencing and line-locate measures are plain f32 distances from the line start, so
chainage on a 1,000 km route in meters is resolved to about 0.06 m.

## 4. Consistency findings

### 4.1 Output layout for variable-length line or polygon output: seven layouts

**CONFIRMED.** There is no single layout. `status.count` means a different thing in each.

| Layout | Used by | Offsets name | `status.count` counts | Provenance | Overflow |
| --- | --- | --- | --- | --- | --- |
| `GPULineGeometryOutput` (`line-segmentize/line-segmentize-types.ts:36`) | segmentize, great-circle arcs, smooth, chunk, rectangle clip | `lineOffsets` | vertices (`lineCount` separate) | `sourceLineIndices`, `sourceRows`, `measures` | 0/1 |
| `GPULineSplitPieces` = `GPUGeneratedGeometryPort` (`line-split/gpu-line-split.ts:74`) | line split, line clip | `geometry.lineOffsets` | pieces (`vertexCount` separate) | `sourceIds`, `vertexSourceEdges` | 0/1 + candidateOverflow (split only) |
| `GPUSharedPathsRuns` (`line-clip/gpu-shared-paths.ts:61`) | shared paths | `offsets` | runs | `leftLineIds`, `rightLineIds`, `forward` | 0/1 |
| `GPULineMergeOutput` (`line-merge/gpu-line-merge.ts:42`) | line merge | `chainOffsets` | chains | inverse map `lineChains`, `lineOrders`, `lineReversed` | none |
| Row selection `GPUCompactOutput` + `lineCounts`/`lineStarts` (`gpu-line-simplification.ts:124`) | simplification | `lineStarts` | kept rows | ids are rows | 0/1 |
| Fixed stride, padded | offset curve, outline geometry, grid generator, shape generator | implicit `i·k`; shape generator writes `offsets` | none, or `vertexCount` | none | none |
| CSR hull (`gpu-group-convex-hull.ts:76`) | group convex hull | `offsets` + `counts` | none | `vertexIndices` | bit set 1 \| 2 |

Polygon output has no layout of its own. Rectangle clip reuses `GPULineGeometryOutput.lineOffsets`
as ring offsets and relies on the input `featureOffsets` and `polygonOffsets`
(`gpu-rectangle-clip.ts:118-126`); offset curve does the same by stride (`gpu-offset-curve.ts:151-158`).

**Recommendation.** Converge on `GPUGeneratedGeometryPort<GPULineGeometryPort>` (it is in
`analysis-core/contracts/ports.ts:169` and already carries `sourceIds` and
`GPUBoundedResultStatusPort`), with one documented extension block for vertex scalars
(`vertexCount`, `requiredVertexCount`) and optional per-vertex columns (`sourceRows`, `measures`).
Define `count` as the number of features (lines, pieces, runs, chains), as `GPUCompactOutput` does,
and put vertex counts in `vertexCount`. Add the polygon equivalent (`featureOffsets`,
`polygonOffsets`, `ringOffsets` written by the contributor) for rectangle clip, offset curve and
the grid and shape generators. Move the type and its validator out of `line-segmentize` into
`analysis-core/contracts`. A polygon contributor (overlay, buffer) needs exactly the same
piece-table, capacity clamp and scalar publication (see 5.1).

### 4.2 Geometry input: four shapes for the same port

**CONFIRMED.** Contributors do not all accept the ports of `analysis-core/contracts/ports.ts`.

| Input shape | Contributors |
| --- | --- |
| Nested packed port with `kind` (`lines: GPUPackedGeometryPort<GPULineGeometryPort>`) | line split, line clip, shared paths, minimum clearance (`geometry: GPUSegmentGeometry`), grid generator `extent` |
| Flat fields via `GPULineGeometryFields` / `GPUPolygonGeometryFields` | line merge, simplification, linear referencing, line locate, rectangle clip, outline, offset curve, minimum bounds, label point, shape descriptors |
| Flat fields re-declared by hand (same shape, not the type) | segmentize (`gpu-line-segmentize.ts:62-64`), smooth (`gpu-line-smooth.ts:81-84`), chunk (`gpu-line-chunk.ts:98-101`) |
| Not geometry | great-circle arcs (`sources`, `targets`), shape generator (`centers`, `radii`), group geometry and hull (`positions`, `labels`, `groupCount`), offset expansion (`offsets`) |

Multi-kind contributors use a `geometryType` discriminant (rectangle clip, outline, offset curve;
offset curve makes it optional and defaults to `'lines'`), while the ports use `kind`. Minimum
bounds discriminates by the presence of `labels` (`gpu-minimum-bounds.ts:119-123`). The `sourceIds`
of a port are ignored by every flat-field contributor; line clip and line split write input line
rows, not the input `sourceIds`, into `pieces.sourceIds` (`gpu-line-clip-by-polygon.ts:907-914`),
which contradicts the port doc at `ports.ts:79-80` ("sourceIds ... replace the feature rows in its
output").

**Recommendation.** Single-input contributors take flat fields typed with the `*Fields` types;
two-input contributors take named packed ports. Use `kind` everywhere instead of `geometryType`.
Map output provenance through the input `sourceIds` when given.

### 4.3 Longitude/latitude handling

| Behaviour | Contributors |
| --- | --- |
| True great-circle (slerp, haversine, cross-track) | segmentize, great-circle arcs, chunk, linear referencing, line locate, shape generator |
| Mixed: haversine distance to a chord that is straight in lon/lat, in a local equirectangular frame | simplification (Douglas-Peucker) (`gpu-line-simplification.ts:157-174`) |
| Approximate: local east/north scale per vertex, documented wrong near poles and antimeridian | outline geometry (`gpu-outline-geometry.ts:157-166`), declared as the `great-circle` metric |
| Rejected at construction (planar-only `resolveGPUSpatialContext` default, `spatial-context.ts:263`) | smooth, Visvalingam, line split, line clip, shared paths, line merge, rectangle clip, minimum bounds, minimum clearance, label point, shape descriptors, offset curve, group geometry, group hull, grid generator |
| No context (correct for a topological operation) | offset expansion |

None are silently planar. Two rejections are unnecessary. **C5 (CONFIRMED):** line merge compares
endpoint bits only (`gpu-line-merge.ts:241-262`) and never measures, so it should accept any
context, as offset expansion does. Rectangle clip in degree space is what turf `bboxClip` and
`ST_ClipByBox2D` do, so it could accept lon/lat with a documented linear-in-degrees rule. Outline
geometry claims the great-circle metric but implements an equirectangular approximation; declare it
as such or reject lon/lat like offset curve.

### 4.4 Units, parameters and baking

Distances and tolerances are in `spatialContext` units in every contributor that takes them
(segmentize length, chunk length and measures, simplification tolerance, locate measures and
offsets, linear-referencing radius, outline distance, shape-generator radii). Visvalingam uses
squared units, which is documented. That part is consistent.

Delivery is not:

- Packed `parameters: GraphDataView<'float32'>` written by `get*ParameterValues` (13
  contributors). Only segmentize and great-circle arcs declare a schema with
  `defineGPUSpatialParameterSchema` (`line-segmentize-parameters.ts:16-49`); the rest pack raw arrays.
  Shape generator uses a 2-element layout, all others 4.
- A dedicated one-row view: linear-referencing `radius` (`gpu-linear-referencing.ts:166`), so it
  cannot warn about short lon/lat distances.
- Compile-time props that are numeric tuning, baked as WGSL literals: label point grid sizes and
  rounds (`gpu-label-point.ts:251-259`), smooth `iterationCount`, simplification
  `maximumIterations`/`finishSpanLimit`/`neighborhoodRadius`, offset curve `quadSegments`, outline
  `joinSegments`, minimum bounds `maximumHullVertices`, group geometry `medoidMaximumGroupSize`,
  grid `columns`/`rows`, line clip `mode`, line merge `directed`. Each changes the shader. Most are
  structural (array sizes, unrolled rounds) and properly so; label point grid sizes and
  `refinementRounds` could be loop bounds read from a buffer.
- The sphere radius is baked with `getWGSLFloatLiteral` (`line-segmentize-kernels.ts:62`). It comes
  from the compile-time `spatialContext`, so this is consistent.

**Recommendation.** Every per-frame value goes through a schema declared with
`defineGPUSpatialParameterSchema` (it records units and enables the lon/lat warning). Make the
linear-referencing radius a schema field.

### 4.5 Status

- Line split uses `GPUBoundedResultStatusPort` and reports intersection-pair overflow as
  `candidateOverflow` and uncertain pairs as `overflow` (`gpu-line-split.ts:757-758, 892`). Line clip
  runs the same pipeline but folds pair overflow into `overflow` and reports uncertain pairs only in
  `uncertainCount` (`gpu-line-clip-by-polygon.ts:860-861, 897`). **CONFIRMED** inconsistency
  between two siblings.
- Linear referencing reports `candidateCount` (`gpu-linear-referencing.ts:102`), not in the shared
  vocabulary; `overflow` mixes leaf and candidate overflow.
- Minimum bounds and group convex hull use `overflow` as a bit set (1 group cap, 2 total capacity)
  (`gpu-group-convex-hull.ts:58-61`), where the contract defines 0/1.
- **C7 (CONFIRMED in this file, effect SUSPECTED):** minimum clearance accepts a `leafCapacity`
  below the vertex count (`gpu-minimum-clearance.ts:210-211`) but the BVH overflow is a transient
  (`segment-intersection/segment-bvh.ts:165`) that is never reported, so dropped segments would give
  a silently overestimated clearance.
- Line locate writes a per-event status code column; shape descriptors and label point have none.

**Recommendation.** `overflow` is 0/1; detail goes in named fields (`groupOverflow`,
`candidateOverflow`). Line clip adopts line split's scalars. Add `approximation` where caps clamp
(G6).

### 4.6 Naming

- Six names for the `0xffffffff` sentinel: `GPU_LINE_NO_SOURCE`, `GPU_LINE_SPLIT_NONE`,
  `GPU_LINE_MERGE_NONE`, `GPU_SHARED_PATHS_NONE`, `GPU_OFFSET_EXPANSION_NO_OWNER`,
  `GPU_GROUP_GEOMETRY_NO_MEDOID`. One `GPU_NO_ROW` in `analysis-core` would do.
- Closed-ring flag: `closedLines` (segmentize, `gpu-line-segmentize.ts:81`) versus `closed`
  (smooth, `gpu-line-smooth.ts:91`).
- Kernel-model types exported although documented as "not a prop": `GPULineCoordinateSystem`
  (`'planar' | 'spherical'`), `GPULinearReferencingCoordinateSystem`, `GPULineLocateCoordinateSystem`,
  `GPUShapeCoordinateSystem` (`'planar' | 'geodesic'`); `geospatial-analysis/gpu-point-buffer.ts:19`
  calls the same model `'geographic'`. Drop them from the public barrels.
- Default context: great-circle arcs defaults to great-circle (`gpu-great-circle-arcs.ts:112`);
  every other contributor defaults to planar.
- Directory placement: `GPUSharedPaths` lives in `line-clip/`; `GPUOffsetCurve` in
  `outline-geometry/` while its loop-removal and component stages live in `geometry-refinement/`;
  smooth, chunk and arcs live in `line-segmentize/`.

### 4.7 Barrel (`gpu-spatial-analysis/index.ts`)

Whole-directory `export *` (`index.ts:5-6`) makes collisions a compile error, which is good. It also
re-exports `analysis-core/contracts` and `ring-assembly` (`:8-9`) and `GPUParameterBuffer` from
`analysis-core` (`:89-92`), so the contracts are reachable from two entry points. The tail
(`:97-110`) is out of alphabetical order. `shared/` is not exported, correctly. Everything a
subdirectory index exports becomes public, including tuning constants such as
`GPU_SHAPE_DESCRIPTORS_MONOTONE_CHAIN_AVERAGE_VERTICES` and the kernel-model types above.

## 5. Duplication and shared-helper candidates

### 5.1 Helpers defined more than once

| Helper | Copies | Recommendation |
| --- | --- | --- |
| `FIND_LINE_WGSL` (identical text) | `line-segmentize/line-segmentize-kernels.ts:25`, `outline-geometry/gpu-outline-geometry.ts:390` | One copy in `analysis-core`. |
| Owner-of-row binary search (vertex → line, or ring → polygon → feature) | `lineOfVertex` in `gpu-line-clip-by-polygon.ts:358`, `gpu-line-split.ts:415` and `:703`; `findLine` above; `ownerSearch` in `gpu-minimum-bounds.ts:300-312`; `lastAtMost` in `gpu-minimum-clearance.ts:318-327`; `getRingStart` in `gpu-label-point.ts:262`; also `geometry-topology/gpu-snap-rounded-noding.ts` | One WGSL builder over 1 to 3 offset levels, or run `GPUOffsetExpansion` once with `ownerMap` chaining. |
| Line split piece machinery: event ordering key along a segment, unique flags, piece table, capacity-clamp finalize, write pieces, scalar publication | `gpu-line-split.ts:446-474, 722-897` and `gpu-line-clip-by-polygon.ts:367-375, 826-982` (near-identical finalize and scalar kernels) | Extract "noded segment pieces → bounded line output" as an internal module. Polygon overlay and make-valid node lines through a line split too (doc at `gpu-line-split.ts:58-62`); check them against this module. |
| Ordered f32 key (`bits ^ 0x80000000` / `~bits`) | inline in `gpu-line-clip-by-polygon.ts:501-502`, `gpu-line-split.ts:472`; `GET_ORDERED_KEY_WGSL` exists in `analysis-core/kernels/wgsl-common.ts:66` and grid generator uses it | Use the helper. |
| Finite test | `isFiniteBits`, `isFiniteValue`, `isFiniteFloat` (`wgsl-common.ts:46-61`), `isSpatialFinite` (`shared/spatial-wgsl-point-helpers.ts:32`) | One name. |
| Integer hash | `hashPair` in `gpu-minimum-bounds.ts:573-577` versus `HASH_UINT32_WGSL` (`analysis-core/kernels/hash-uint32.ts`, re-exported by `shared/hash-uint32.ts`) | Use the shared hash. |
| Load-balanced expansion | `line-segmentize/balanced-search.ts` versus `analysis-core/kernels/load-balanced-for-each.ts` | Merge into the `analysis-core` helper. |
| Fixed-round iteration gate | simplification imports `createRasterIteration*` from `raster-analysis/cost-distance/raster-relaxation` (`gpu-line-simplification.ts:28-34`) | Move the gate to `analysis-core`; a vector contributor should not depend on a raster module. |
| List ranking | only in `gpu-line-merge.ts`; needed by shared paths (G1) | Extract to `analysis-core`. |
| Unused | `IS_SAME_SPATIAL_POINT_WGSL` (`shared/spatial-wgsl-point-helpers.ts:45`) has no users | Delete. |

### 5.2 Overlapping contributors

| Overlap | Recommendation |
| --- | --- |
| `GPUGreatCircleArcs` is `GPULineSegmentize` on two-vertex lines plus a `minimumSegments` floor | Add `minimumPieceCount` to segmentize; keep arcs as a thin wrapper or drop it. |
| `GPULabelPoint` and `geometry-refinement/gpu-maximum-inscribed-circle.ts` compute the same pole of inaccessibility (the second with branch and bound and error bounds) | One contributor; label point becomes an output of it. |
| `GPUShapeGenerator` circles and `geospatial-analysis/gpu-point-buffer.ts` | One point-to-circle generator. |
| `GPUOffsetCurve` and `geometry-refinement/{gpu-offset-curve-components, gpu-line-loop-removal}.ts` are stages of one offset-curve pipeline in two directories | Co-locate. |
| Rectangle clip `'lines'` and `GPULineClipByPolygon` with a rectangle | Keep both (per-frame viewport clip is much cheaper), but share the output layout. |
| `GPUMinimumBounds.bounds`, `GPUGroupGeometry.bounds` and `bounds-filter/` | Minimum bounds computes the envelope from the hull; group geometry from a segmented min/max. Keep one envelope path (the segmented reduction). |
| `GPUShapeDescriptors` gift-wrapping hull versus `GPUGroupConvexHull` | Drop gift wrapping (G2). |

### 5.3 What a polygon contributor would also need

Segment tables and BVH (`segment-intersection/`), the noding-and-piece pipeline of line split
(5.1), the measure prefix (`createLinePrefixNode`, already used by
`lon-lat-polygons/gpu-lon-lat-polygon-preparation.ts`), offset joins (offset curve), and one
bounded output layout with provenance (4.1). These belong in `analysis-core` or a
`geometry-core` internal directory, not in `line-segmentize/`, whose kernels five other
directories already import (`linear-referencing`, `rectangle-clip`, `outline-geometry`,
`line-density`, `lon-lat-polygons`).

## 6. Comprehensiveness

Ranked by how often a practitioner reaches for it. "Reuses" names existing pieces.

| Rank | Missing | Reference | GPU-suitable | Reuses | Effort |
| --- | --- | --- | --- | --- | --- |
| 1 | Split lines by a second geometry (blade lines or points), not only self-noding | `ST_Split`, turf `lineSplit(line, splitter)`, Shapely `split` | yes | `GPUSegmentIntersection` two-sided mode (as line clip uses it), line split piece machinery | S |
| 2 | Topology-preserving line simplification for networks (shared vertices and crossings kept) | `ST_SimplifyPreserveTopology`, mapshaper arc simplification | yes (pin noded junctions, then per-arc DP) | line split (noding), simplification importance, `polygon-coverage-simplification` (other reviewer) | M |
| 3 | Antimeridian splitting of lines for 2D maps (cut at ±180 instead of unwrapping) | deck.gl `wrapLongitude` CPU step, `ST_WrapX`, antimeridian packages | yes | segmentize longitude shifts (`rowShifts`), line chunk emit | S |
| 4 | Points at a fixed interval along lines (as points with line id and measure) | `ST_LineInterpolatePoints`, turf `along` in a loop, QGIS "Points along geometry" | yes | line chunk ranges + line locate | S |
| 5 | Line substring and locate by fraction | `ST_LineSubstring(fraction)`, Shapely `substring(normalized=True)` | yes | chunk (distance only today) + per-line length from the prefix | S |
| 6 | Measured (M) geometry: add, interpolate and locate between measures | `ST_AddMeasure`, `ST_LocateBetween`, route calibration | yes | segmentize `measures` output, line locate | M |
| 7 | Line-to-line nearest points and shortest line | `shapely.shortest_line`, `ST_ShortestLine`, `nearest_points` | yes | linear referencing BVH, segment table | S-M |
| 8 | Bezier or Catmull-Rom spline smoothing | turf `bezierSpline` | yes | smooth level kernels | S |
| 9 | True buffer of lines with dissolve | Shapely `buffer`, turf `buffer` | partly | offset curve + polygon buffer (other reviewer) | tracked elsewhere |

Already present elsewhere, not missing: Hausdorff and Fréchet (`track-similarity`), remove repeated
points and snap to grid (`geometry-edit/gpu-geometry-cleanup.ts`,
`geometry-refinement/gpu-precision-reduction.ts`), maximum inscribed circle
(`geometry-refinement`), random points on lines (`dot-density/gpu-random-points-on-line.ts`).
Already tracked: GEOM-01 (rectangle clip polygon outputs) and GEOM-02 (smooth source rows and
measures) in `dev-docs/roadmap.md` 2.10.

Out of place in a GPU analysis library:

- `GPUOutlineGeometry` is picture-only (its own doc says it is wrong for area, overlay, selection
  and containment, `gpu-outline-geometry.ts:178-186`). deck.gl `PathLayer` already strokes on the
  GPU. Move it to the kepler or lenses package, or drop it.
- `GPUOffsetExpansion` is a generic CSR expansion with no geospatial content. Under the
  upstream-first rule it belongs in `@luma.gl/graph` (alongside `GPUSegmentOffsets` and
  `GPUSegmentedLayout`, which do not cover it); until then it is the shared helper for 5.1.

## 7. Top ten actions

| # | Action | Effort | Files |
| --- | --- | --- | --- |
| 1 | Define one bounded line and polygon output layout in `analysis-core/contracts` (`GPUGeneratedGeometryPort` plus vertex scalars and per-vertex columns, `count` means features) and migrate the seven layouts | M-L | `line-segmentize/line-segmentize-types.ts`, `line-geometry-output.ts`, `line-split/gpu-line-split.ts`, `line-clip/*`, `line-merge/gpu-line-merge.ts`, `rectangle-clip/gpu-rectangle-clip.ts`, `outline-geometry/gpu-offset-curve.ts`, `grid-generators/*`, `analysis-core/contracts/ports.ts` |
| 2 | Replace the shared-paths chain walk with list ranking extracted from line merge | S-M | `line-clip/gpu-shared-paths.ts`, `line-merge/gpu-line-merge.ts`, new `analysis-core/kernels/list-ranking.ts` |
| 3 | Load-balance shape descriptors: per-edge moments with segmented sums; hull method chosen by largest feature or always `GPUGroupConvexHull` | S-M | `shape-descriptors/*`, `analysis-core/kernels/load-balanced-for-each.ts` |
| 4 | Align status: line clip adopts line split's scalars; `overflow` 0/1 with named detail fields in hull and minimum bounds; replace `candidateCount`; report minimum-clearance BVH overflow; expose `approximation` in segmentize and arcs | S | `line-clip/gpu-line-clip-by-polygon.ts`, `group-geometry/gpu-group-convex-hull.ts`, `minimum-bounds/gpu-minimum-bounds.ts`, `linear-referencing/gpu-linear-referencing.ts`, `minimum-clearance/gpu-minimum-clearance.ts`, `line-segmentize/gpu-line-segmentize.ts`, `gpu-great-circle-arcs.ts` |
| 5 | Extract the noding-to-pieces module shared by line split and line clip, and offer it to polygon overlay | M | `line-split/gpu-line-split.ts`, `line-clip/gpu-line-clip-by-polygon.ts`, polygon overlay (other reviewer) |
| 6 | One geometry input convention: `*Fields` for single input, named ports for two inputs, `kind` not `geometryType`; honour input `sourceIds` in outputs | M | segmentize, smooth, chunk, rectangle clip, outline, offset curve, minimum bounds, line split, line clip |
| 7 | Merge duplicates: label point into maximum inscribed circle (branch and bound with a segment index); great-circle arcs into segmentize; shape-generator circles with point buffer | M | `label-point/*`, `geometry-refinement/gpu-maximum-inscribed-circle.ts`, `line-segmentize/gpu-great-circle-arcs.ts`, `grid-generators/gpu-shape-generator.ts`, `geospatial-analysis/gpu-point-buffer.ts` |
| 8 | Move shared WGSL into `analysis-core`: owner search, `FIND_LINE_WGSL`, ordered key, finite test, hash, balanced search, the iteration gate; delete `IS_SAME_SPATIAL_POINT_WGSL` | S | `line-segmentize/{line-segmentize-kernels, balanced-search}.ts`, `outline-geometry/gpu-outline-geometry.ts`, `minimum-bounds`, `minimum-clearance`, `label-point`, `shared/*`, `raster-analysis/cost-distance/raster-relaxation.ts` |
| 9 | Add split-by-blade, points-at-interval, substring-by-fraction and antimeridian line cut (ranks 1, 3, 4, 5 in section 6) | S each | `line-split/`, `line-segmentize/` |
| 10 | Context and parameter hygiene: line merge accepts any context; declare outline geometry's lon/lat model honestly; every per-frame value through `defineGPUSpatialParameterSchema`; unify `closed`/`closedLines`, sentinels and kernel-model type names; stop exporting kernel-model types | S | `line-merge/gpu-line-merge.ts`, `outline-geometry/gpu-outline-geometry.ts`, all `*-parameters` packers, `line-segmentize/gpu-line-smooth.ts`, index barrels |
