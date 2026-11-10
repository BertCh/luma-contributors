# Code review: polygon geometry, topology and spherical geometry (2026-11-09)

Reviewer domain: planar polygon geometry and topology, spherical (global) geometry, and the
`geospatial-analysis` helpers. Paths are relative to `modules/analysis/src`. Findings are tagged
**CONFIRMED** (code path read) or **SUSPECTED** (inferred from code read plus reasoning, not run).
Nothing was built or executed.

## 1. Scope and method

About 35,100 lines in the directories listed in the task (49 exported classes, no `add*Recipe`
functions). Largest files: `delaunay-tessellation/gpu-delaunay-tessellation.ts` (2,184),
`analysis-core/ring-assembly/gpu-segment-ring-assembly.ts` (1,686),
`polygon-coverage-simplification/gpu-coverage-simplification.ts` (1,610),
`polygon-overlay/gpu-polygon-overlay.ts` (1,380), `analysis-core/polygon-triangulation/*` (3,497 in
four files), `geometry-topology/gpu-snap-rounded-noding.ts` (1,153).

**Read in full:** `analysis-core/kernels/exact-orientation-wgsl.ts`, `exact-lattice-wgsl.ts`,
`exact-geometry-wgsl.ts`; `segment-intersection/segment-intersection-wgsl.ts`,
`segment-intersection/gpu-segment-intersection.ts`; `geometry-topology/gpu-line-noding.ts`;
`global-coordinates/gpu-spherical-polygon-overlay.ts`, `gpu-spherical-polygon-contains.ts`,
`gpu-spherical-arc-overlay.ts`, `assert-unit-vector.ts`;
`polygon-triangulation/gpu-constrained-delaunay-triangulation.ts`;
`geospatial-analysis/gpu-point-buffer.ts`, `gpu-datum-grid-transform.ts`;
`polygon-coverage-topology/coverage-topology-kernels.ts`.

**Read in large part:** `gpu-snap-rounded-noding.ts` (1-920 of 1,153); `gpu-segment-ring-assembly.ts`
(1-420, 760-840, 1290-1540); `gpu-polygon-overlay.ts` (1-420, 1100-1380); `gpu-make-valid.ts`
(60-365); `geometry-refinement/gpu-ratio-concave-hull.ts` (25-195), `gpu-segment-snap.ts`
(110-205), `gpu-offset-curve-components.ts` (195-258), `gpu-normalize-polygons.ts` (95-200),
`gpu-line-loop-removal.ts` (20-200); `geometry-edit/geometry-edit-kernels.ts` (53-170);
`gpu-delaunay-tessellation.ts` (540-645, 1400-1445); `global-coordinates/gpu-spherical-convex-intersection.ts` (20-100).

**Skimmed (class TSDoc, constructor validation, `invocationCount`/loop/limit greps):**
`gpu-polygonize.ts`, `gpu-geometry-validity.ts`, `gpu-geometry-predicates.ts`,
`gpu-buffer-surface.ts`, `gpu-coverage-simplification.ts`, `gpu-coverage-dissolve.ts`,
`gpu-coverage-validity.ts`, `analysis-core/polygon-triangulation/gpu-polygon-triangulation.ts`,
`polygon-triangulation-monotone.ts` (header), `gpu-voronoi-diagram.ts`, `gpu-concave-hull.ts`,
`gpu-vertex-snap.ts`, all `geometry-edit`, `geometry-measures` and remaining `geometry-refinement`
classes, `gpu-lon-lat-polygon-preparation.ts`, `analysis-core/lon-lat-ring-wgsl.ts`,
`gpu-spherical-convex-contains.ts`, `analysis-core/kernels/polygon-edge-slabs.ts` (header).

**Not opened:** `geometry-measures/geodesic-wgsl.ts`, `rhumb-wgsl.ts`, `geodesic-kernels.ts`,
`feature-measures-cooperative.ts`, `geometry-measures-kernels.ts`;
`lon-lat-polygons/lon-lat-polygon-preparation-kernels.ts`;
`polygon-overlay/overlay-face-labelling-wgsl.ts` (grep only);
`polygon-triangulation-monotone-wgsl.ts`, `polygon-triangulation-layout.ts`;
`ring-assembly/ring-vertex-quantizer.ts`, `segment-polygonization-diagnostics.ts`;
`segment-intersection/segment-bvh.ts`, `segment-table.ts`; `geometry-topology/topology-types.ts`;
`geometry-refinement/rectilinear-subdivision.ts`; `geospatial-analysis/projection-spatial-context.ts`.
Verdicts for classes whose kernels I only skimmed are based on their documented composition plus
greps of loops and dispatch sizes; they are marked as such in the table.

## 2. Contributor table

Abbreviations. Par: `row`, `coop`, `serial`, `cpu`. Iter: `none`, `fixed`, `unrolled+gate`.
Params: `buffer` (per-frame `parameters` view; none of these use `GPUParameterBuffer`) or `baked`.
Inputs: every class here takes `packed` views only. Ctx: `sc` = `spatialContext` with
`resolveGPUSpatialContext`. Status: `shared` = names from `contracts/status.ts`; `rows` = per-row
codes; `adhoc` = word arrays. Valid: `helpers` = `validate*`/`throw new Error(msg)`; `assert` = bare
`assert(`.

### 2.1 Noding, intersection, ring assembly, overlay, buffer

| Symbol | File (lines) | Computes (reference) | Par | Work | Iter | Hard limits | Params | Ctx | Status | Valid | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPUSegmentIntersection | segment-intersection/gpu-segment-intersection.ts (736) | all intersecting segment pairs, exact kinds (turf `lineIntersect`, `kinks`) | row | O(n log n + k), BVH, Morton sort | none | rows ≤ binding/32 B = 4,194,304 (inherent, `segment-intersection-wgsl.ts:20`); successor search 1,024 vertices (design, `:132`) | n/a | sc | shared | helpers | conforming |
| GPULineNoding | geometry-topology/gpu-line-noding.ts (548) | noded atomic segments, f32 crossing points (GEOS `MCIndexNoder`) | row | O(n log n + k) via GPULineSplit | none | lines + 4·capacity ≤ piece bound (inherent) | n/a | sc | shared | helpers | align |
| GPUSnapRoundedNoding (`@internal`) | geometry-topology/gpu-snap-rounded-noding.ts (1,153) | snap-rounded noding on a power-of-two lattice (GEOS `SnapRoundingNoder`) | row | O((n+k) log) BVH + radix sorts | none | lattice 2^22+1 per axis (inherent f32, `exact-lattice-wgsl.ts:11`); vertices + crossings ≤ binding/128 B = 1,048,576 (inherent, `:145-150`) | baked (`gridSize`) | sc | shared | helpers | conforming |
| GPUSegmentRingAssembly | analysis-core/ring-assembly/gpu-segment-ring-assembly.ts (1,686) | chain directed segments into rings, shells, holes | row + coop | O(n log n) match; hole→shell O(H·S) brute force (`:1404`) | fixed (pointer jumping) | none named | baked (`vertexTolerance`) | sc (planar, lon/lat turn scale) | shared | helpers | align |
| GPUPolygonize | geometry-topology/gpu-polygonize.ts (617) | all bounded faces of linework (shapely `polygonize`) | row | noding + assembly | none | inherits | baked | sc | shared | helpers | align (skimmed) |
| GPUMakeValid | geometry-topology/gpu-make-valid.ts (512) | linework make_valid by source parity (GEOS) | row | face parity O(F·S_capacity) brute force (`:349`) | none | inherits | baked | sc | shared | helpers | rework |
| GPUPolygonOverlay | polygon-overlay/gpu-polygon-overlay.ts (1,380) | Boolean overlay of operand unions on a snap lattice (OverlayNG `grid_size`) | row | O((n+k) log) BVH ray shoot, pointer jumping | fixed | features < 2^31 (`:211`); snap limit above | baked (`vertexTolerance`, `fillRule`, `operation`) | sc | shared (`GPUBoundedResultStatusPort`) | helpers | conforming (provenance gap, §4) |
| GPUBufferSurface | polygon-buffer/gpu-buffer-surface.ts (841) | offset strokes then winding dissolve (shapely `buffer`) | row | overlay cost | none | boundary rows + intersections ≤ 1,048,576 (inherent) | buffer (`distance`, mitre) | sc | shared | helpers | align (dissolves all features, §3) |
| GPUSegmentSnap | geometry-refinement/gpu-segment-snap.ts (205) | snap endpoints and insert reference vertices on segments (shapely `snap`, vertex-to-segment part) | serial per line | O(segments · R²) brute force (`:173-186`) | none | `maximumVerticesPerLine` | baked (`tolerance`, `:139`) | sc | adhoc (4 words/line) | helpers | rework → merge into GPUVertexSnap |
| GPUVertexSnap | vertex-snap/gpu-vertex-snap.ts (390) | vertex-to-vertex snap (shapely `snap` subset) | row | O(n log R) BVH | none | none | buffer | sc | none | helpers | conforming |

### 2.2 Validity, predicates, coverage, cleanup, measures

| Symbol | File (lines) | Computes (reference) | Par | Work | Iter | Hard limits | Params | Ctx | Status | Valid | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPUGeometryValidity | geometry-validity/gpu-geometry-validity.ts (876) | validity bitmask (`ST_IsValidDetail`, `explain_validity`) | row + coop (64 lanes) | O(n log n) via segment intersection | none | vertex rows 4,194,304 (inherent) | n/a | sc | rows (mask) + `overflow`, `intersectionCount` (ad hoc name) | helpers | conforming |
| GPUGeometryPredicates | geometry-predicates/gpu-geometry-predicates.ts (863) | `is_simple`, `is_ring`, `is_ccw`, `equals_exact`, counts | row (serial per feature for equality) | O(n log n) | none | successor 1,024 | n/a | sc | shared | helpers | conforming |
| GPUCoverageValidity | polygon-coverage-topology/gpu-coverage-validity.ts (561) | `coverage_invalid_edges` | row | O(n log n) BVH | none | none | buffer (`gapWidth`) | sc | rows | helpers | conforming |
| GPUCoverageDissolve | polygon-coverage-topology/gpu-coverage-dissolve.ts (572) | coverage union by label | row (serial area per ring) | O(n log n) sorts + assembly | none | none | baked (`snapTolerance`, `vertexTolerance`) | sc | shared | helpers | align |
| GPUCoverageSimplification | polygon-coverage-simplification/gpu-coverage-simplification.ts (1,610) | `coverage_simplify` / `ST_CoverageSimplify` | row (serial per arc) | O(n log n) per round | unrolled (topology rounds, default 4, each a full GPUSegmentIntersection, `:888`) | `topologyPairCapacity` | buffer + baked iteration counts | sc | adhoc (`topologyStatistics[4]`) + shared | helpers | align |
| GPUGeometryCleanup | geometry-edit/gpu-geometry-cleanup.ts (538) | `set_precision(pointwise)`, `remove_repeated_points` | row; serial per ring when tolerance > 0 | O(n) | none | none | buffer | sc | shared | helpers | conforming |
| GPUAffineTransform | geometry-edit/gpu-affine-transform.ts (399) | affine / translate / rotate / scale | row | O(n) | none | none | buffer | sc | none | helpers | conforming |
| GPUGeometryOrientation | geometry-edit/gpu-geometry-orientation.ts (243) | `reverse`, `orient_polygons` (exact extreme-vertex test) | row (serial per ring) | O(n) | none | none | buffer | sc | none | helpers | conforming |
| GPUGeometryMeasures | geometry-measures/gpu-geometry-measures.ts (668) | area, length, centroid, bounds, per group | row + coop for large rings | O(n), sort for groups | none | none | n/a | sc | shared | helpers | conforming (skimmed) |
| GPUGeodesicPairs, GPUGeodesicDestination | geometry-measures (334, 221) | distance, bearing, destination (Vincenty, haversine, rhumb) | row | O(n) | fixed cap | Vincenty iteration cap | baked | sc | rows (`converged`) | helpers | conforming (skimmed) |
| GPULocalFrameProjection | geometry-measures/gpu-local-frame-projection.ts (90) | lon/lat to local planar frame | row | O(n) | none | none | n/a | sc | none | helpers | conforming |
| GPULonLatPolygonPreparation | lon-lat-polygons/gpu-lon-lat-polygon-preparation.ts (508) | antimeridian split, pole cap, densify | row | O(n) scans | none | `maximumPieceCount` ≤ 65,536 | buffer | sc | shared | helpers | conforming (skimmed) |

### 2.3 Triangulation, tessellation, hulls, refinement

| Symbol | File (lines) | Computes (reference) | Par | Work | Iter | Hard limits | Params | Ctx | Status | Valid | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPUPolygonTriangulation | analysis-core/polygon-triangulation/gpu-polygon-triangulation.ts (1,278) + monotone (2,081) | Earcut per polygon, parallel monotone path for large polygons | serial per polygon + row (monotone) | near O(n log n) z-hash; monotone O(n log n) | fixed | `maximumWork` 4,000,000 (design, baked); scratch nodes 4,793,490 (inherent) | baked | sc | rows (`valid`) | helpers | conforming |
| GPUConstrainedDelaunayTriangulation | polygon-triangulation/gpu-constrained-delaunay-triangulation.ts (119) | Lawson flips inside each triangulated polygon | serial per polygon | O(flips · T²) per polygon (`:93-112`) | loop to `maximumFlips` 4,096 (baked) | none named; quadratic | baked | none | rows (`converged`) | assert | rework |
| GPUDelaunayTessellation | delaunay-tessellation/gpu-delaunay-tessellation.ts (2,184) | parallel insertion + flip Delaunay, exact incircle, SoS | row | O(n) per step | unrolled+gate: up to 4,096 steps × ~9 nodes (`:1415-1416`) | 4,096 steps (design, `:575-583`); coordinate span 55 binary orders (inherent, `:37`) | n/a | sc | shared | helpers | align |
| GPUVoronoiDiagram | delaunay-tessellation/gpu-voronoi-diagram.ts (514) | Voronoi edges clipped to bounds | row | O(n) + sort | none | Delaunay limits | n/a | sc | shared | helpers | conforming (edges only, §6) |
| GPUConcaveHull, GPUGroupConcaveHull | concave-hull/gpu-concave-hull.ts (447) | alpha-shape boundary edges (radius) | row | Delaunay + sort | none | Delaunay limits | override constant | sc | `converged` only | assert (6) | align |
| GPURatioConcaveHull | geometry-refinement/gpu-ratio-concave-hull.ts (319) | GEOS ratio concave hull by border erosion | coop, one workgroup of 256 for all triangles (`:129-136`) | O(T) per round, rounds ≤ T | in-kernel loop | `maximumIterations` = triangle count | override | sc | `converged` | helpers | rework → merge with GPUConcaveHull |
| GPUSubdividePolygons | geometry-refinement/gpu-subdivide-polygons.ts (390) | triangulate then merge into convex pieces (`ST_Subdivide`-like) | serial per polygon | O(v²) per polygon merge loop (`:202-234`) | none | `maximumSplits` ≤ 65,536; `maximumVertices` per piece | baked | sc | adhoc (4 words) | helpers | align |
| GPUMaximumInscribedCircle | geometry-refinement/gpu-maximum-inscribed-circle.ts (256) | branch and bound (GEOS `MaximumInscribedCircle`) | serial per feature (`:127`) | O(cells · edges) per feature | in-kernel loop | `cellCapacity` ≤ 4,096, iterations ≤ 65,536 (`:77-78`) | baked | sc | rows | helpers | align |
| GPUVectorStraightSkeleton | geometry-refinement/gpu-vector-straight-skeleton.ts (321) | event-driven straight skeleton | serial per polygon (`:130`) | O(v · events) | in-kernel loop | 1,024 vertices, 4,096 events (design, `:63-64`) — already tracked as GE08 | baked | none | adhoc | helpers | align |
| GPUConvexStraightSkeleton | geometry-refinement/gpu-convex-straight-skeleton.ts (257) | skeleton of convex polygons, self-described prototype | serial per ring | O(v²) | loop ≤ 128 | 64 vertices (`:74`) | baked | sc | adhoc | helpers | merge into GPUVectorStraightSkeleton (drop) |
| GPUOffsetCurveComponents | geometry-refinement/gpu-offset-curve-components.ts (258) | component selection for offset curves | single thread for all fragments (`:199-243`) | O(F²) to O(F³) | in-kernel loop | fragment capacity | baked (match distance) | none | adhoc (8 words) | helpers | rework |
| GPULineLoopRemoval | geometry-refinement/gpu-line-loop-removal.ts (218) | chronological loop erasure of noded curves | serial per line; one thread over lines for offsets (`:170-185`) | O(n) | none | none | n/a | sc | adhoc (4 words) | helpers | align |
| GPUPolygonPrecisionReduction, GPULinePrecisionReduction | geometry-refinement/gpu-precision-reduction.ts (341) | topology-preserving `set_precision` | row | make-valid cost | none | lattice 2^22 | baked | sc | inherits | helpers | align (inherits GPUMakeValid quadratic step) |
| GPUNormalizePolygons | geometry-refinement/gpu-normalize-polygons.ts (258) | GEOS `normalize` for polygons | serial per ring and per feature | O(v²) smallest rotation per ring (`:133-144`) | none | none | n/a | none | none | helpers | rework (S) → merge with GPUNormalizeGeometry |
| GPUNormalizeGeometry | geometry-refinement/gpu-normalize-geometry.ts (457) | normalize nested collections | row | O(n) per depth round | unrolled to depth ≤ 64 (GE01, tracked) | 64 | baked | none | none | helpers | align (skimmed) |
| GPUCanonicalPolygonEquality, GPUCanonicalGeometryEquality | geometry-refinement (132, 124) | structural equality after normalization | row (per pair/feature) | O(n) | none | none | n/a | none | rows | helpers | merge with `GPUGeometryPredicates.equalsExact` |

### 2.4 Spherical and geospatial helpers

| Symbol | File (lines) | Computes (reference) | Par | Work | Iter | Hard limits | Params | Ctx | Status | Valid | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPUSphericalConvexContains | global-coordinates/gpu-spherical-convex-contains.ts (119) | point in convex spherical polygon | row per query | O(v) per query, polygon named per query | none | none | baked tolerance | none | rows | assert | align |
| GPUSphericalConvexIntersection | global-coordinates/gpu-spherical-convex-intersection.ts (235) | paired convex spherical clip + areas | row per pair | O(v1·v2) per pair | none | 3-32 vertices, capacity ≤ 64 (design, `:26`, `:64`) | baked | none | rows | assert | align |
| GPUSphericalPolygonContains | global-coordinates/gpu-spherical-polygon-contains.ts (239) | winding containment with holes | row per query; per-vertex ring check | ring check O(v²) per ring (`:178`); O(v) per query; caller supplies `polygonIndices` (no candidate search) | none | tolerance 1e-7 to 1e-2 | baked | none | rows | assert | rework |
| GPUSphericalPolygonOverlay | global-coordinates/gpu-spherical-polygon-overlay.ts (247) | gnomonic chart + GPUPolygonOverlay + lift | row | overlay cost | none | one hemisphere (`minimumCosine` ≥ 1e-3) | baked (`chartCenter`) | none | shared (inherits) | assert | align |
| GPUSphericalArcOverlay | global-coordinates/gpu-spherical-arc-overlay.ts (252) | intrinsic arc arrangement and ring walk, non-hemisphere regions | serial: `invocationCount: 1` (`:135`) | O(n³)+ in one thread | in-kernel loops | ≤ 128 input vertices (`:59`), output ≤ 1,024 (design) | baked | none | adhoc (`counts[4]`, status bits) | assert | rework |
| GPUDatumGridTransform | geospatial-analysis/gpu-datum-grid-transform.ts (274) | bilinear datum shift, inverse iteration (PROJ grids) | row | O(grids) per row (grid list unrolled into WGSL, `:203-226`) | fixed ≤ 1,024 | grid list compiled in | baked (grid metadata) | none (should be lon/lat) | rows | assert (11) | align |
| GPUPointBuffer | geospatial-analysis/gpu-point-buffer.ts (160) | circle vertex offsets around points | row | O(n·k) | none | none | buffer | own `coordinateSystem` prop, not sc | none | assert (4) | align |

Verdict counts (49 classes): **conforming 17**, **align 21**, **rework 8** (GPUMakeValid,
GPUConstrainedDelaunayTriangulation, GPUSegmentSnap, GPURatioConcaveHull, GPUOffsetCurveComponents,
GPUNormalizePolygons, GPUSphericalPolygonContains, GPUSphericalArcOverlay), **merge 3**
(GPUConvexStraightSkeleton, both canonical-equality classes). Two of the rework items also merge
(GPUSegmentSnap into GPUVertexSnap, GPURatioConcaveHull into GPUConcaveHull).

The two-generation hypothesis holds broadly but not by file size. Counter-examples: the
`geometry-refinement` expansion classes `GPUSubdividePolygons`, `GPUMaximumInscribedCircle` and
`GPULineLoopRemoval` use `validateCapacity` with messages and scale by row; the baseline
`GPUMakeValid` (exact predicates, full contract) has a brute-force O(F·S) step; the baseline
`GPUSegmentRingAssembly` has an O(H·S) hole assignment.

## 3. GPU quality findings

**Q1. Robustness map: which paths are exact and which are float.** CONFIRMED unless noted.

| Path | Predicates | Constructions | Vertex identity |
| --- | --- | --- | --- |
| `GPUSegmentIntersection` | exact orientation (f32 filter + 256-bit integer fallback, `exact-orientation-wgsl.ts:129-143`); uncertain → `uncertainCount` | proper crossing point rounded f32 (`segment-intersection-wgsl.ts:86-95`) | input vertices |
| `GPULineNoding` / `GPUPolygonize` (default) / `GPUMakeValid` (default) | exact | f32 crossing points, no snap; node drift possible | `vertexTolerance` hash in ring assembly |
| `GPUSnapRoundedNoding`, `GPUPolygonOverlay`, `GPUBufferSurface`, `GPUPolygonPrecisionReduction`, polygonize/make-valid with `snapRounding` | exact on an integer lattice (`exact-lattice-wgsl.ts`), rounded crossing by exact binary search | lattice points; world output `origin + k·cell` | exact lattice equality (assembly tolerance 0.25 cell, `gpu-polygon-overlay.ts:1224`) |
| `GPUSegmentRingAssembly` turn choice at shared vertices | **f32** pseudo-angle (`gpu-segment-ring-assembly.ts:804`, `:823-829`) | n/a | tolerance hash, 3×3 cells (`:793`) |
| Ring orientation | exact extreme-vertex test in `geometry-edit-kernels.ts:120-160` and ring assembly (`:1293`); **f32 shoelace** in `coverage-topology-kernels.ts:63-66` | | |
| `GPUPolygonTriangulation`, `GPUDelaunayTessellation`, CDT | exact orientation, adaptive exact incircle (span ≤ 55 binary orders) | input vertices | input rows |
| Coverage family | exact pair predicates via segment intersection; grid snap key `floor(x/s + 0.5)` | input vertices | grid key |
| Spherical family | **f32 with angular tolerance** everywhere (`atan2` winding sums, `normalize(cross)`) | f32 | tolerance distance |
| `GPUSegmentSnap`, `GPUOffsetCurveComponents` | f32 distances with tolerance | f32 | tolerance |

Why it matters: the overlay TSDoc says "Results are exact on the lattice". The lattice predicates
are, but the face walk in ring assembly picks the next edge by an f32 pseudo-angle. Two outgoing
lattice edges such as (2^22, 1) and (2^22 − 1, 1) differ in angle by about 2^-44 radians, below f32
resolution of the key (about 2^-24), so the key can tie or misorder and the walk can pick the wrong
face. Tie-break by lower segment index then decides. **SUSPECTED** wrong rings on long, nearly
parallel lattice edges sharing a vertex; the f32 code is CONFIRMED. Better design: compare
candidates with `orientSign` (half-plane of each direction against the incoming edge, then
`orientSign(origin, a, b)` between candidates), which is exact on lattice integers. Effort S.

**Q2. Hole-to-shell assignment is O(holes × rings).** CONFIRMED `gpu-segment-ring-assembly.ts:1404-1426`.
Each hole thread loops over every ring, with a bounding-box reject before the exact winding walk.
This sits under overlay, polygonize, make-valid, coverage dissolve, buffer, isochrones and isobands.
An overlay result with 100k holes and 100k shells does 10^10 box tests. The "innermost" choice also
compares f32 `|area|` values (`:1407-1411`), so two nested shells of equal f32 area pick by index.
Better: a `GPUBVH` (or the existing `polygon-edge-slabs` index) over shell bounds, probe with the
hole's symbolic point, keep the smallest containing shell; for the overlay case the face labelling
already knows each face's enclosing group from the ray shoot, so parent links could be emitted
directly. Effort M.

**Q3. `GPUMakeValid` face parity is O(faces × noded-segment capacity).** CONFIRMED
`gpu-make-valid.ts:349`: every face thread loops over `SEGMENT_CAPACITY` and skips other groups.
For 100k features and 1M noded segments this is 10^11 iterations, also inherited by
`GPUPolygonPrecisionReduction`. Better: noded segments are already tagged by group; sort them by
group (one `GPUSort`) and loop only over the face's group range, or reuse the overlay's
ray-shoot-plus-pointer-jump face labelling, which is O((n+k) log n). Effort S for the group range,
M for ray shooting.

**Q4. Noding in `GPUPolygonize` / `GPUMakeValid` ignores groups.** CONFIRMED: `GPULineNoding` and
`GPUSnapRoundedNoding` take no group, and `gpu-polygonize.ts:229-244` passes all lines to one
noder. Rings of different features that cross are split at each other's crossings, which spends
`intersectionCapacity` on pairs that do not matter for per-feature repair and, **SUSPECTED**, adds
vertices to make-valid output that GEOS would not add. Better: pass `sameFeatureOnly`-style grouping
through to `GPUSegmentIntersection` (the option exists, `gpu-segment-intersection.ts:78`). Effort S.

**Q5. `GPUBufferSurface` dissolves all features into one surface.** CONFIRMED
`gpu-buffer-surface.ts:832` uses overlay `operation: 'dissolve'`, and the overlay gives every
boundary edge group 0 (`gpu-polygon-overlay.ts:1155-1158`). A buffer of N features returns their
union. `shapely.buffer`, `ST_Buffer` and turf `buffer` are per feature. This is the most visible
semantic gap in the domain (see §6 item 1). Better: carry a group ID (source feature) through
noding provenance into the overlay's winding labels and ring assembly `groups`. Effort M once Q6 is
done.

**Q6. Overlay output has no source-row provenance.** CONFIRMED `gpu-polygon-overlay.ts:1266`:
`output.geometry.sourceIds[index] = index` (the output feature index). Provenance exists only on
`output.boundary` edges (`operandIds`, `sourceFeatureIds`, `:1172-1179`). There is no way to get
the left and right source rows of each result polygon, which is what GeoPandas `overlay` and
row-wise `ST_Intersection` return. Effort L for per-pair overlay (§6 item 1); S to rename the field
or document that it is an output index.

**Q7. `GPUSphericalArcOverlay` runs in one thread.** CONFIRMED `gpu-spherical-arc-overlay.ts:135`
(`invocationCount: 1`), at most 128 input vertices (`:59`) and 1,024 output vertices. All noding,
classification (`arcContains` loops all edges up to seven times per probe) and the ring walk are
nested loops in one invocation. It is a reference implementation, not a GPU design. Better: build
it on the planar pipeline. Split the sphere into charts (for example six cube faces with gnomonic
projection, as S2 does), node and overlay per chart with `GPUPolygonOverlay`, then stitch chart
seams. Or implement S2-style exact predicates on unit vectors and reuse `GPUSegmentIntersection`
and ring assembly with a spherical turn key. Effort L (overlaps CR04).

**Q8. `GPUConstrainedDelaunayTriangulation` is quadratic per sweep in one thread per polygon.**
CONFIRMED `gpu-constrained-delaunay-triangulation.ts:93-112`: each sweep compares every triangle
pair of a polygon (O(T²)) to find one flip, then restarts, up to 4,096 times. A 10k-triangle
polygon costs about 10^8 pair tests per flip. Better: build triangle adjacency once (the shared
`createTriangleEdgeAdjacencyNodes` used by the concave hulls), then run the parallel
"flip if lowest-priority illegal edge" rounds already used by `GPUDelaunayTessellation`, with
constrained edges marked unflippable. Effort M.

**Q9. Delaunay encodes up to 4,096 unrolled steps.** CONFIRMED
`gpu-delaunay-tessellation.ts:575-583`, `:1415-1416`. Above 2,048 points every step is encoded
(about nine nodes each, gated by indirect dispatch), so a large input encodes roughly 37,000 nodes
per frame, most of them empty dispatches. Better: a fixed number of encoded rounds with a GPU loop
counter and a host-side resubmit only when `iterationLimitReached`, or a round count derived from
`log2(n)` with a convergence-driven continuation. Effort M. The ratio concave hull, concave hull
and Voronoi inherit this.

**Q10. Single-thread or single-workgroup kernels in `geometry-refinement`.** CONFIRMED:
`GPUOffsetCurveComponents` chains all fragments in one thread with O(F²) cancellation and an O(F)
search per step (`gpu-offset-curve-components.ts:199-243`); `GPURatioConcaveHull` runs all erosion
rounds in one 256-lane workgroup (`gpu-ratio-concave-hull.ts:129-136`), so it uses one compute unit
whatever the input size; `GPULineLoopRemoval` publishes offsets with a one-thread loop over lines
(`gpu-line-loop-removal.ts:170-185`) where the rest of the library uses `GPUScan`. Better: chain
fragments with `GPUSegmentRingAssembly` (it already chains directed segments with tolerance and
reports open chains); run erosion rounds as separate dispatches with indirect gating; replace the
offset loop with a scan plus a capacity cut. Effort S to M each.

**Q11. Brute-force candidate loops without an index.** CONFIRMED: `GPUSegmentSnap` scans every
reference vertex for every vertex, and for each segment loops R × R with a duplicate check over
earlier references (`gpu-segment-snap.ts:145-148`, `:173-186`): O(segments · R²) per line thread.
`GPUVertexSnap` already builds a BVH over references (`gpu-vertex-snap.ts` TSDoc). `GPUSphericalPolygonContains`
checks each ring by testing every vertex's edge against every edge of its ring (O(v²) per ring,
`:178`), and its query kernel needs the caller to pair each query with a polygon (`polygonIndices`),
so there is no candidate search at all. Better: grid or BVH indices (`GPUGridIndex`, `GPUBVH`);
for spherical containment, the planar `polygon-edge-slabs` idea on a per-ring gnomonic chart or a
cell index. Effort S (segment snap), M (spherical).

**Q12. Quadratic per-ring normalization.** CONFIRMED `gpu-normalize-polygons.ts:133-144`: the
smallest cyclic rotation is found by comparing every candidate start against the best, O(v²) per
ring in one thread; the `reverse` loop runs once (`reverse < 1u`), so it is dead code. GEOS
normalizes by starting at the minimum vertex, O(v). Effort S.

**Q13. Parameters baked into WGSL.** CONFIRMED: overlay and snap noding bake `vertexTolerance` /
`gridSize` as `GRID_CELL` (`gpu-snap-rounded-noding.ts:350`); ring assembly bakes
`vertexTolerance`; `GPUSegmentSnap` bakes `TOLERANCE` (`:139`); spherical classes bake tolerance,
`chartCenter` and `referencePoint`; CDT bakes `maximumFlips`; datum grid bakes every grid's origin,
spacing and size. None of the domain uses `GPUParameterBuffer`; the per-frame ones use a
`parameters: GraphDataView<'float32'>` instead (cleanup, affine, orientation, vertex snap, buffer
distance, coverage validity and simplification, lon/lat preparation, point buffer). For the snap
lattice, baking is defensible (the grid size changes topology capacity); for tolerances and chart
centres it is not. Effort S per class.

**Q14. Lattice resolution is capped at 2^-22 of the extent, silently.** CONFIRMED
`gpu-snap-rounded-noding.ts:366`: `cell = max(GRID_CELL, extentPower · 2^-22)`. A 5,000 km extent
in metres gives a cell of about 1.2 m whatever `vertexTolerance` the caller asks for. It is
documented (`gpu-polygon-overlay.ts:180-186`) and the actual lattice is written to `output.lattice`,
but there is no status flag when the requested grid was coarsened. Better: set
`approximation = 1` when `cell > gridCell`. Effort S. The limit itself is inherent to f32 lattice
storage; tiling by spatial partition is the way past it.

**Q15. Per-feature or per-ring serial loops (accepted pattern, note only).** `GPUGeometryOrientation`,
`GPUGeometryPredicates` equality, `GPUCoverageDissolve` ring area, `GPUCoverageSimplification` arc
copy, Earcut per polygon, `GPUMaximumInscribedCircle`, `GPUVectorStraightSkeleton`,
`GPUSubdividePolygons`. Most document it. `GPUGeometryValidity`, `GPUGeometryMeasures` and ring
assembly already have the 64-lane cooperative fallback for large rings; the others do not. A shared
"cooperative ring reduction" helper would let each adopt it (§5).

## 4. Consistency findings

**C1. Two noding paths, one ring assembly path.** Planar noding has two paths: float
(`GPULineSplit` → `GPULineNoding`, f32 crossing points) and snap rounding (`GPUSnapRoundedNoding`).
`GPUPolygonize`/`GPUMakeValid` offer both through `snapRounding`; overlay, buffer and precision
reduction use only snap rounding. Outside this domain `gpu-network/network-noding` is a third
(not reviewed). Ring assembly is one path (`GPUSegmentRingAssembly`), used by overlay, polygonize,
coverage dissolve, isochrones, cell outlines and isobands. Private chain or ring walks remain in
`GPUSphericalArcOverlay` (`:229-244`), `GPUOffsetCurveComponents` (`:216-240`) and
`GPUSubdividePolygons` (convex merge). Recommend: snap rounding as the default for every
polygon-producing operator, float noding only for line outputs, and all chaining through
`GPUSegmentRingAssembly`.

**C2. Four vertex identity rules.** Tolerance hash with a 3×3 neighbourhood (ring assembly);
lattice equality (snap rounding); grid key `floor(x / s + 0.5)` (coverage family and
`GPUContiguityWeights`, the same expression in `gpu-coverage-dissolve.ts:312`,
`gpu-coverage-simplification.ts:412`, `spatial-weights/gpu-contiguity-weights.ts:348`); plain
distance tolerance (`GPUSegmentSnap`, `GPUOffsetCurveComponents`, spherical). Grid keys split two
points straddling a cell boundary that a tolerance would join. Recommend one shared quantizer in
`analysis-core` with the two documented modes (grid key, tolerance cluster) and use lattice
coordinates wherever snap rounding ran.

**C3. Tolerance and grid prop names.** `vertexTolerance` (overlay, where it is actually a grid
size; ring assembly, where it is a match distance; polygonize `precision.vertexTolerance`),
`gridSize` (snap noding, cleanup, precision reduction), `snapTolerance` (coverage),
`tolerance` (vertex snap, segment snap, spherical classes in radians, datum grid in degrees).
Recommend `gridSize` for lattice step, `matchTolerance` for vertex matching, and units stated in
the `spatialContext`.

**C4. Status shape.** Shared vocabulary in segment intersection, noding, overlay, ring assembly,
Delaunay, coverage dissolve, lon/lat preparation. Ad hoc word arrays in `GPUCoverageSimplification`
(`topologyStatistics[4]`), `GPULineLoopRemoval` and `GPUSubdividePolygons` (4 words),
`GPUOffsetCurveComponents` (8 words), `GPUSegmentSnap` (4 words per line),
`GPUSphericalArcOverlay` (`counts[4]` plus bit codes). The five spherical classes each define their
own per-row codes (0/1/2 in contains, 0/1/2/4 in the overlays). `GPUGeometryValidity` names its
count `intersectionCount` instead of `requiredCount`. Recommend `GPUBoundedResultStatusPort` plus
`uncertainCount`/`invalidCount` for all, and one shared per-row code enum for spherical results.

**C5. Validation style.** Two styles, cleanly split by family. Helpers with messages: everything
planar except `GPUConcaveHull` (6 bare asserts), CDT (3). Bare `assert` only: all of
`global-coordinates` (15, matches the census) and `geospatial-analysis` (15). `assertUnitVector`
is a one-function file (`global-coordinates/assert-unit-vector.ts`) that belongs in the shared
validation helpers.

**C6. Spatial context.** Every planar class with geometry input except `GPUConstrainedDelaunayTriangulation`,
`GPUOffsetCurveComponents`, `GPUVectorStraightSkeleton`, `GPUNormalizePolygons`,
`GPUNormalizeGeometry` and the canonical-equality pair declares `spatialContext`. None of the five
spherical classes does; they take raw unit vectors with no datum or radius, even though
`contracts/spatial-context.ts` has a longitude-latitude space with `edgeModel`. `GPUPointBuffer`
uses its own `coordinateSystem: 'geographic' | 'planar'` (`gpu-point-buffer.ts:49`) and
`GPUDatumGridTransform` has none. Recommend a `unit-vector` (ECEF direction) coordinate space or
reuse of longitude-latitude with `edgeModel: 'great-circle'` for the spherical family, and
`spatialContext` in place of `coordinateSystem`.

**C7. Is the spherical family the same quality and scale as the planar family?** No. CONFIRMED by
reading three of five in full:

| Aspect | Planar family | Spherical family |
| --- | --- | --- |
| Predicates | exact orientation, SoS, lattice | f32 with angular tolerance 1e-7 to 1e-2 |
| Candidate search | BVH, Morton sort, edge slabs | none; caller pairs queries with polygons |
| Largest input | millions of vertices | 128 vertices (arc overlay), 32 per polygon (convex intersection) |
| Parallelism | row and cooperative | row per query; arc overlay one thread |
| Status | shared vocabulary | private codes |
| Validation | helpers with messages | bare `assert` |
| Context | `spatialContext` | none |
| WGSL style | multi-line, commented | one-line bodies, compressed identifiers (`gpu-spherical-arc-overlay.ts:146-248`) |

The exception is `GPUSphericalPolygonOverlay`, which reuses the planar overlay through a gnomonic
chart and so inherits its scale and exactness, but only within one hemisphere and with f32 chart
coordinates. `GPULonLatPolygonPreparation` and `geometry-measures` (great-circle area, geodesics)
are the scalable global pieces and follow the planar contract.

**C8. Provenance.** Kept consistently in noding (`sourceFeatureIds`, `sourceRingIds`,
`sourceEdgeIds`, start and end parameters), polygonize (`sourceIds` = group), make-valid (source
feature), coverage dissolve (label), segment snap (`sourceEdges`), offset components
(`sourceEdges`), subdivide (`[vertexCount, sourceFeature, sourcePolygon, seed]`), concave hull
(position row indices). Missing or misleading: overlay polygons (`sourceIds` is the output index,
Q6), buffer (dissolved, Q5), spherical overlays (inherit the overlay gap; arc overlay has none),
spherical convex intersection (rows are aligned pairs, which is acceptable), Voronoi (edges carry
triangle rows; no site ID per cell, **SUSPECTED** from the TSDoc only).

**C9. Two functions named `createRingOrientationNode`.** CONFIRMED
`geometry-edit/geometry-edit-kernels.ts:73` (exact extreme-vertex test) and
`polygon-coverage-topology/coverage-topology-kernels.ts:19` (f32 shoelace). The coverage version can
misjudge thin rings that the edit version gets right. Recommend deleting the coverage copy.

**C10. Hard limits named and validated.** Planar limits are derived from
`maxStorageBufferBindingSize` and exported as `get*Maximum*Count(device)` functions with messages.
Expansion limits are literal: 64 vertices (`gpu-convex-straight-skeleton.ts:74`), 1,024 and 4,096
(`gpu-vector-straight-skeleton.ts:63-64`), 4,096 cells (`gpu-maximum-inscribed-circle.ts:77`),
65,536 splits (`gpu-subdivide-polygons.ts:73`), 128 and 1,024 (`gpu-spherical-arc-overlay.ts:59-69`),
32 and 64 (`gpu-spherical-convex-intersection.ts:26`, `:64`), 4,096 flips (CDT). Recommend named
constants exported next to each class, as the planar ones are.

## 5. Duplication and shared-helper candidates

| Duplicate | Where | Recommendation |
| --- | --- | --- |
| Ring orientation node | `geometry-edit-kernels.ts:73`, `coverage-topology-kernels.ts:19` | keep the exact one; move to `analysis-core/kernels` |
| Grid snap key `floor(x/s + 0.5)` with the same clamp | `gpu-coverage-dissolve.ts:312`, `gpu-coverage-simplification.ts:412`, `spatial-weights/gpu-contiguity-weights.ts:348` | one helper in `analysis-core/ring-assembly/ring-vertex-quantizer.ts` |
| Concave hull | `concave-hull/gpu-concave-hull.ts` (radius alpha) and `geometry-refinement/gpu-ratio-concave-hull.ts` (GEOS ratio) | one class with `criterion: 'radius' \| 'ratio'`; both share Delaunay and edge adjacency already |
| Straight skeleton | convex prototype and vector skeleton | drop the convex prototype |
| Snapping | `GPUVertexSnap` (BVH) and `GPUSegmentSnap` (brute force) | one class, vertex and segment modes, BVH for both |
| Normalization | `GPUNormalizePolygons`, `GPUNormalizeGeometry` | one class; polygons is a special case of the tree |
| Equality | `GPUCanonicalPolygonEquality`, `GPUCanonicalGeometryEquality`, `GPUGeometryPredicates.equalsExact`/`equalsIdentical` | one equality contributor with a `normalize` option |
| Chain and ring walks | `GPUSegmentRingAssembly`, `GPUOffsetCurveComponents` (`:216-240`), `GPUSphericalArcOverlay` (`:229-244`) | route through ring assembly (it reports open chains) |
| Unit-sphere WGSL (`isUnit`, `onArc`, arc crossing) | each spherical file defines its own (`polygonUnit`/`polygonOnArc` in contains `:117-121`, `arcOn` in arc overlay `:159`, `sphere*` in convex intersection) | one `spherical-wgsl.ts` in `analysis-core/kernels` |
| `isFiniteBits` | redefined locally in `gpu-ratio-concave-hull.ts:163` and inline bit tests in spherical files | use `IS_FINITE_BITS_WGSL` from `wgsl-common` |
| Pointer jumping | `polygon-overlay/overlay-face-labelling-wgsl.ts:284` (`createPointerJumpNodes`) and ring assembly's own leader jumping; monotone triangulation's jump sources | candidate for `analysis-core/kernels` (S) |
| Cooperative large-ring reduction | `gpu-geometry-validity.ts` (64 lanes), ring assembly ring stats, `feature-measures-cooperative.ts` | one helper; then orientation, coverage area, predicates can use it |

`exact-orientation-wgsl.ts`, `exact-geometry-wgsl.ts`, `exact-lattice-wgsl.ts`,
`polygon-edge-slabs.ts`, `GPUSegmentIntersection` and `GPUSegmentRingAssembly` are the right shared
core. `GPUSnapRoundedNoding` is `@internal` but is the robust noder the rest should use; it should be
public. `analysis-core/lon-lat-ring-wgsl.ts` is used by about ten consumers (spatial join, region mask,
zonal statistics, measures, cell cover and others) and is correctly placed.

## 6. Comprehensiveness

Ranked by how often practitioners reach for each. "Already tracked" items are listed only by ID.

| # | Missing capability (reference) | GPU-suitable | Reuses | Effort |
| --- | --- | --- | --- | --- |
| 1 | Row-wise / pairwise overlay with attributes: GeoPandas `overlay(how=...)`, row-wise `ST_Intersection(a, b)`, `ST_Difference`, `intersection(a, b)` on aligned arrays | yes: candidate pairs from spatial join, then one overlay with `group = pair` | `GPUSpatialPredicateJoin` candidates, snap noding with groups, overlay winding per group, ring assembly `groups` | L |
| 2 | Per-feature buffer (`ST_Buffer`, `shapely.buffer` element-wise, turf `buffer`) | yes | item 1's grouping; `GPUOffsetCurve` | M after 1 |
| 3 | Dissolve by attribute for overlapping (non-coverage) polygons: `dissolve(by=...)`, `ST_Union` aggregate, `unary_union` per group | yes | overlay `dissolve` with group labels | M after 1 |
| 4 | Polygon simplification per feature preserving topology (`ST_SimplifyPreserveTopology`, `simplify(preserve_topology=True)`, Visvalingam `ST_SimplifyVW`) | yes | `GPULineSimplification` on rings, coverage simplification's repair loop | M |
| 5 | Voronoi polygons per site (`shapely.voronoi_polygons`, `ST_VoronoiPolygons`, turf `voronoi`) | yes | `GPUVoronoiDiagram` edges plus ring assembly with site as group | S |
| 6 | Polygon split by line (`shapely.ops.split`, `ST_Split`) | yes | snap noding of polygon + blade, polygonize, make-valid parity | M |
| 7 | Hausdorff and Fréchet distance between polygons/lines (`hausdorff_distance`, `frechet_distance`, `ST_HausdorffDistance`) | yes | track-similarity has a trajectory version; segment BVH | M |
| 8 | Spatial-indexed spherical point-in-polygon join at scale (S2 `S2ShapeIndex` contains) | yes | edge slabs on gnomonic charts or cell cover | M (exact predicates tracked as CR04) |
| 9 | Spherical buffer / geodesic polygon buffer (`ST_Buffer(geography)`) | yes | `GPUPointBuffer` (points only), local frame projection + planar buffer + inverse | M |
| 10 | Conforming constrained Delaunay with arbitrary constraint segments and refinement (CGAL `Constrained_Delaunay_triangulation_2`, Ruppert/Chew mesh refinement) | partly | snap noding for constraints, parallel flip of Q8 | L |
| 11 | Minimum clearance, polylabel: present elsewhere (`minimum-clearance`, `label-point`) | n/a | n/a | n/a |
| 12 | Full global (non-hemisphere) spherical overlay at scale | yes, via charts | planar overlay | L (Q7; related to CR04 / CR05) |
| 13 | Geodesic intersection of great-circle and geodesic lines, GeographicLib `Intersect` | yes | Vincenty kernels | M |

Already tracked: nesting above 64 and straight skeleton above 1,024 vertices (GE01 / GE08); exact
S2 predicates and covering (CR04 / CR05); datum epochs (CR02 / CR03); difficult geodesics (CR01);
vertical datum (DATUM-01); rectangle clip outputs (GEOM-01).

Out of place or candidates to drop or move:

- `GPUConvexStraightSkeleton` (self-described prototype, 64 vertices): drop.
- `GPUSphericalArcOverlay` in its current form: keep as a CPU-side or test oracle, or rework (Q7);
  it is not a GPU operator.
- `GPUCanonicalPolygonEquality` / `GPUCanonicalGeometryEquality`: test utilities more than analysis
  operators; fold into one equality contributor or move under `testing`.
- `GPUDatumGridTransform`: PROJ-style grid shifting fits the roadmap's projection work in luma.gl
  `gpu-project` better (§2.9 and §5 of the roadmap already point there); **SUSPECTED** move.

## 7. Top ten actions

| # | Action | Effort | Files |
| --- | --- | --- | --- |
| 1 | Add group (source feature or pair) labels through snap noding, overlay winding and ring assembly; emit real source IDs per output polygon; make `GPUBufferSurface` per feature by default | L | `gpu-snap-rounded-noding.ts`, `gpu-polygon-overlay.ts`, `overlay-face-labelling-wgsl.ts`, `gpu-buffer-surface.ts`, `gpu-segment-ring-assembly.ts` |
| 2 | Exact turn selection at shared vertices in ring assembly (replace `ringTurnKey` with `orientSign` comparisons) | S | `gpu-segment-ring-assembly.ts:780-838` |
| 3 | Indexed hole-to-shell assignment (BVH over shell bounds, or parent links from overlay ray shooting) | M | `gpu-segment-ring-assembly.ts:1368-1436` |
| 4 | Make-valid face parity over the face's own group range (sort noded segments by group), and group-aware noding in polygonize | S | `gpu-make-valid.ts:299-360`, `gpu-polygonize.ts:229-244`, `gpu-line-noding.ts`, `gpu-snap-rounded-noding.ts` |
| 5 | Replace CDT with adjacency-based parallel constrained flips | M | `gpu-constrained-delaunay-triangulation.ts`, `shared/triangle-edge-adjacency.ts` |
| 6 | Cap Delaunay encoding (fixed encoded rounds with GPU-side continuation), benefits hulls and Voronoi | M | `gpu-delaunay-tessellation.ts:1415-1560` |
| 7 | Spherical family: shared `spherical-wgsl.ts`, `spatialContext`, validation helpers, shared status codes; indexed containment; rework arc overlay on charts | L | `global-coordinates/*`, `analysis-core/contracts/spatial-context.ts` |
| 8 | Merge snapping (BVH segment mode into `GPUVertexSnap`), concave hulls (one class, two criteria), normalization and equality; drop the convex skeleton | M | `geometry-refinement/*`, `vertex-snap/*`, `concave-hull/*` |
| 9 | Route `GPUOffsetCurveComponents` chaining through ring assembly; multi-dispatch ratio-hull erosion; scan in line-loop removal; O(v) ring normalization | M | `gpu-offset-curve-components.ts`, `gpu-ratio-concave-hull.ts`, `gpu-line-loop-removal.ts`, `gpu-normalize-polygons.ts` |
| 10 | One vertex quantizer and one ring-orientation node in `analysis-core`; status `approximation` when the snap lattice is coarser than requested; make `GPUSnapRoundedNoding` public | S | `ring-vertex-quantizer.ts`, `coverage-topology-kernels.ts`, `geometry-edit-kernels.ts`, coverage files, `gpu-contiguity-weights.ts`, `gpu-snap-rounded-noding.ts:366` |
