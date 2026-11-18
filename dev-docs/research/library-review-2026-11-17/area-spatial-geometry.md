# Geometry, lines, polygons, joins and search

I reproduced one geometry-loss defect in the public CPU implementation and found two join-to-gather defects by tracing the GPU producer and consumer together. All three can produce plausible results with clear status words. I would address them before adding more geometry operators.

The library has a common polygon *input* layout. The main interoperability problem is what happens after an operation changes cardinality. A `uint32` can then mean a feature row, application ID, vertex row, ring count or diagnostic bitmask. Several public ports have matching TypeScript types and incompatible meanings.

## What I checked

I inventoried **80 GPU contributor classes in 41 source directories**, covering 244 TypeScript files and 90,846 lines. The same directories contain 19 GPU graph operation forms and 19 CPU operation forms. I counted **362 specs** under the matching `gpu-spatial-analysis` test directories, including the ring-assembly specs. The inventory below covers all 80 contributors. The operation forms are counted separately because they expose graph-value names for the underlying layouts.

I read `AGENTS.md`, the contributor contract, `overlay-feature-sources.ts`, `spatial-join-candidate-pages.ts`, `gpu-local-frame-projection.ts` and `voronoi-clip.spec.ts` in full. I sampled the largest implementation, `gpu-delaunay-tessellation.ts` at 2,364 lines, plus ring assembly, both triangulations, predicate-join paging and finalization, nearest publication, pair gather, polygonization, make-valid, bounds, normalization and offset reconstruction. Recent changes were selected from `69305eb4`, the November 16 scalability and improvement pass. I checked candidate findings against the roadmap and the earlier geometry, join, math and scalability reviews.

I used read-only source scans, a small `Math.fround` probe, and a direct call to the existing built CPU offset function. I did not build, run repository tests, dispatch GPU work or change files. **UNVERIFIED** below means the predicted device result was not measured. I did not inspect every shader branch, every spec, or other streams’ implementations.

For paths below, `S` means `modules/analysis/src/gpu-spatial-analysis`, `C` means `modules/analysis/src/analysis-core`, and `T` means `modules/analysis/test/gpu-spatial-analysis`. The following reproduces the inventory and the direct-resolver count from the repository root.

```python
from pathlib import Path
import re

s = Path("modules/analysis/src/gpu-spatial-analysis")
named = set("""segment-intersection rectangle-clip delaunay-tessellation concave-hull
minimum-bounds minimum-clearance label-point outline-geometry group-geometry vertex-snap
offset-expansion shape-descriptors lon-lat-polygons dorling-cartogram map-coloring
linear-referencing bounds-filter study-window spatial-join neighbor-search pair-gather
areal-interpolation""".split())
dirs = [d for d in s.iterdir() if d.is_dir() and
        (d.name.startswith(("geometry-", "line-", "polygon-")) or d.name in named)]
dirs += [s.parent / "analysis-core" / n for n in ("polygon-triangulation", "ring-assembly")]
files = [f for d in dirs for f in d.rglob("*.ts")]
classes = [(f, m[1], part) for f in files
           for part in f.read_text().split("export class ")[1:]
           if (m := re.match(r"(GPU\w+)", part)) and not m[1].endswith("Operation")]
tests = Path("modules/analysis/test/gpu-spatial-analysis")
print(len(dirs), len(files), sum(len(f.read_text().splitlines()) for f in files))
print(len(classes), sum("resolveGPUSpatialContext(" in part for _, _, part in classes))
print(sum(len(list((tests / n).rglob("*.spec.ts"))) for n in {d.name for d in dirs}))
```

## Findings

### GEO-01 Offset reconstruction cancels a valid straight line at small coordinate scales [P0] [correctness]

- **Evidence.** I called `computeOffsetCurveComponentsOnCPU` with original vertices `[(0,0),(s,0),(2s,0)]`, raw vertices `[(0,s),(s,s),(2s,s)]`, distance `s`, tolerance `0`, vertex capacity `16` and component capacity `8`. At `s=1`, status was `[1,3,1,3,0,0,0,0]`. At `s=2^-20`, it was `[0,0,0,0,0,0,0,0]`. The cancellation test uses `Math.max(1e-5, distance * 1e-5)`. The endpoints of the two consecutive fragments fit that threshold even when compared in reverse order, which marks both fragments deleted. The GPU implementation has the same absolute floor and opposite-endpoint test. `S/geometry-refinement/cpu-offset-curve-components-operation.ts:256`, `:265`, `:331`; `S/geometry-refinement/gpu-offset-curve-components.ts:266`, `:273`.
- **Why it matters.** Changing coordinate units can turn an ordinary offset into an empty result. The reported *required* size is also zero, so resizing or checking overflow cannot detect it.
- **Proposed change.** Use noded endpoint identity to cancel reversed fragments. Keep the distance-band tolerance separate from endpoint identity. Add the two-scale, three-vertex fixture to both the independent CPU oracle and mixed-graph comparison. **M, compatible API, corrected output.**
- **Confidence.** High for CPU, reproduced. High for the matching GPU source path, **UNVERIFIED** on device. **KNOWN(GPU-12)** covers reworking offset components, but this is a correctness defect independent of parallelization.

### GEO-02 Nearest join can feed an application ID into an attribute row lookup [P0] [correctness]

- **Evidence.** `GPUNearestFeatureJoin` documents `neighborIds` as “feature ID (or row, without `features.sourceIds`)”. Its publisher executes `featureId = featureIds[featureIdsOffset + featureId]`. `GPUPairGatherNeighbors.ids` describes the same slot as a *feature row*. Gather assigns `right = neighborIds[...]` and reads `sourceIn[sourceInOffset + row]`. The nearest gather spec supplies precomputed IDs. `S/spatial-join/gpu-nearest-feature-join.ts:129`; `S/spatial-join/nearest-nodes.ts:1057`; `S/pair-gather/gpu-pair-gather.ts:50`, `:442`, `:511`; `T/pair-gather/gpu-pair-gather.spec.ts:452`.
- **Why it matters.** With source IDs `[1,0]`, a match to right row zero gathers row one's attributes. Both indices are in range, so the result looks valid. With IDs `[100,200]`, the gather writes its missing-value fill instead.
- **Proposed change.** Add optional `neighborRows` to the nearest output and publish the pre-remap row already available in `topk`. Give gather a `rows` input, retain `ids` as a compatibility alias for row-valued inputs, and test the composed contributors with permuted IDs and IDs outside the row range. **M, additive.**
- **Confidence.** High from both sides of the public connection. Device execution **UNVERIFIED**.

### GEO-03 Pair gather loses candidate overflow from a predicate join [P0] [correctness]

- **Evidence.** Predicate join publishes `pairsOverflow` from `pairIncomplete` and `pairsCandidateOverflow` from `candidateIncomplete` separately. Gather promises overflow for “any capacity (including the inputs')”, but initializes its inherited flags with only `props.pairs.status.overflow` and caller-supplied extras. Its predicate fixture wires `count`, `requiredCount` and `overflow`. `S/spatial-join/gpu-spatial-predicate-join.ts:1870`, `:1897`; `S/pair-gather/gpu-pair-gather.ts:72`, `:393`, `:610`; `T/pair-gather/gpu-pair-gather.spec.ts:394`.
- **Why it matters.** One left feature, two matching right features, candidate capacity one and final capacity two suffice. The retained match fits. The producer can signal candidate loss while the gathered table reports a complete result. The caller already handed gather the port containing the diagnostic.
- **Proposed change.** Include `pairs.status.candidateOverflow`, when wired, in gather's inherited flags and input-binding validation. Preserve the documented aggregate `overflow` for compatibility. Add a composed regression where only the candidate stage overflows. **S, compatible, corrected status.**
- **Confidence.** High from the generated assignments. Device execution **UNVERIFIED**.

### GEO-04 Geographic bounds from measures can miss a viewport on the other side of the antimeridian [P0] [correctness]

- **Evidence.** Geometry measures publishes longitudes “unwrapped from the first vertex” and accumulates `local.x + lonLatRingStep(...)`. Bounds filter explicitly suggests using `GPUGeometryMeasures` bounds. For an ordinary query interval, it compares `fMax.x >= boxMin.x && fMin.x <= boxMax.x` with no longitude-period adjustment. `S/geometry-measures/gpu-geometry-measures.ts:91`; `S/geometry-measures/geometry-measures-kernels.ts:801`; `S/bounds-filter/gpu-bounds-filter.ts:145`, `:293`.
- **Why it matters.** A line going from `179°` to `−179°` has continuous bounds `[179,181]`. A viewport spanning `[-179.75,-179.25]` intersects it. The published bounds comparison rejects it. The query itself does not cross the seam, so the existing wrapped-query branch does not help. An early reject removes the feature before an exact predicate can see it.
- **Proposed change.** For longitude/latitude contexts, compare intervals modulo 360, including spans at least 360°. Keep the planar comparison unchanged. Test the actual measures-to-filter connection, reversing the line and shifting its longitudes by one turn. **M, compatible API, corrected geographic selection.**
- **Confidence.** High for the interval counterexample. Device execution **UNVERIFIED**.

### GEO-05 Circumcenter intermediates can overflow or underflow for well-conditioned triangles [P1] [correctness]

- **Evidence.** Both the alpha hull and Voronoi construction square an edge, multiply the result by another edge component, then divide by twice the determinant. Exact orientation protects the sign test only. For `(0,0),(s,0),(0,s)`, the exact center is `(s/2,s/2)`. A scalar `Math.fround` probe of the written formula gave `Infinity` for a center component at `s=2^45` and `0` at `s=2^-52`. In both cases the correct center and radius are representable and the triangle has the same shape as the unit triangle. `S/concave-hull/gpu-concave-hull.ts:251`; `S/delaunay-tessellation/gpu-voronoi-diagram.ts:553`.
- **Why it matters.** The hull's `radius <= MAXIMUM_CIRCUMRADIUS` can make an incorrect retention decision. Voronoi can classify a valid triangle as invalid or use an incorrect finite center. These examples have no near-collinearity or wide span among nonzero input magnitudes.
- **Proposed change.** Share a power-of-two-scaled, triangle-relative circumcenter helper across alpha hull and Voronoi. Rescale the result once and distinguish an unrepresentable final answer from an intermediate failure. Pin both exponents against an analytic oracle. **M, compatible.**
- **Confidence.** High for the scalar f32 counterexamples. Actual compiler arithmetic and device results **UNVERIFIED**.

### GEO-06 Clockwise Delaunay triples reverse Voronoi hull rays [P1] [correctness]

- **Evidence.** The public Voronoi input asks for “Delaunay triples”. `circumcenter` accepts either orientation sign. Hull emission always uses `normalize(vec2<f32>(b.y - a.y, a.x - b.x))`, the right-hand normal. The in-repo tessellator emits counter-clockwise triples; the Voronoi clipping fixture uses that winding throughout. `S/delaunay-tessellation/gpu-voronoi-diagram.ts:75`, `:447`, `:555`; `S/delaunay-tessellation/gpu-delaunay-tessellation.ts:618`; `T/delaunay-tessellation/voronoi-clip.spec.ts:55`.
- **Why it matters.** Reversing a valid triangle preserves its sites, adjacency and circumcenter. It reverses all three emitted ray directions under this formula. This affects callers supplying their own triangulation through the public views.
- **Proposed change.** Carry the certified orientation sign into hull-edge emission and orient the normal away from the opposite site. Run the same clipped-triangle oracle with both windings and compare site-pair geometry. **S, compatible.**
- **Confidence.** High from the sign calculation. Device execution **UNVERIFIED**.

### GEO-07 Polygonization publishes feature counts and ring requirements under one cardinality contract [P1] [consistency]

- **Evidence.** The shared port defines `requiredCount` as the “exact unclamped requirement”. Polygonize writes `count` from a feature offset and `requiredCount` from `ringRequiredCount`. Make-valid documents the same distinction and later replaces `count` with its parity-filtered face total. Overlay already exposes `requiredRingCount` and deliberately omits an unknowable feature requirement. `C/kernels/gpu-contributor-types.ts:21`; `S/geometry-topology/gpu-polygonize.ts:585`; `S/geometry-topology/gpu-make-valid.ts:108`, `:719`; `S/polygon-overlay/gpu-polygon-overlay.ts:155`.
- **Why it matters.** A generic consumer comparing `requiredCount` and `count` can diagnose truncation even after a successful run. A ring requirement is valuable for allocating the next run, but it cannot answer how many feature rows were lost.
- **Proposed change.** Add `requiredRingCount` to the topology result now. Deprecate the current meaning of `requiredCount`, then remove it or publish an exact feature requirement in the next breaking release. Test a holed input where the counts differ with sufficient capacity. **M, additive first, breaking cleanup later.**
- **Confidence.** High. **KNOWN(STD-08)** understates the remaining issue by focusing on other families.

### GEO-08 Geometry validity and segment snap publish failure bits under filter-compatible names [P1] [ergonomics]

- **Evidence.** `GPUGeometryValidity.output.mask` writes zero for a finite point and a failure bit for a nonfinite point. `GPUSegmentSnap.output.validity` writes bits for collapse, invalid input and incomplete output. `GPUNeighborSearch.mask` says “nonzero includes the target”. The contract reserves exactly that polarity for an input mask and `nonzero = valid` for output validity. `S/geometry-validity/gpu-geometry-validity.ts:63`, `:298`; `S/geometry-refinement/gpu-segment-snap.ts:102`, `:499`; `S/neighbor-search/gpu-neighbor-search.ts:189`; `dev-docs/contributor-contract.md:42`.
- **Why it matters.** Wiring validity's point mask directly to neighbor search excludes every ordinary finite point. Simply negating the polygon diagnostic would also exclude polygons that only have an orientation or repeated-vertex advisory. The code already distinguishes structural from advisory bits. `S/geometry-validity/geometry-validity-types.ts:53`.
- **Proposed change.** Publish the existing words as `issueBits`. Add an optional 0/1 `validity` column for geometry validity, using its structural mask and an explicit policy for incomplete checks. For segment snap, migrate its existing `validity` name through deprecation before assigning it boolean semantics. **M, additive migration with an eventual breaking rename.**
- **Confidence.** High. **KNOWN(STD-04)** covers vocabulary, but the reversed filter connection warrants higher priority than a naming cleanup.

### GEO-09 The optional context on longitude/latitude preparation always throws when omitted [P2] [ergonomics]

- **Evidence.** `GPULonLatPolygonPreparationProps` declares `spatialContext?: GPUSpatialContext`. Its constructor passes `undefined` to the shared resolver with longitude/latitude-only support. The resolver defaults to planar. A spec explicitly expects omission to throw. `GPUGreatCircleArcs` supplies its geographic default before calling the same resolver. `S/lon-lat-polygons/gpu-lon-lat-polygon-preparation.ts:93`, `:129`; `C/contracts/spatial-context.ts:265`; `T/lon-lat-polygons/lon-lat-polygon-preparation.node.spec.ts:117`; `S/line-segmentize/gpu-great-circle-arcs.ts:126`.
- **Why it matters.** A caller can satisfy the exported props type and still be guaranteed a construction error. This is especially surprising on the operation whose name already specifies longitude/latitude.
- **Proposed change.** Default omission to the existing longitude/latitude context with metric `none` and linear edges. Continue rejecting an explicitly planar context. If explicit declaration is a project requirement, make the prop required instead. **S, defaulting is backward-compatible for successful callers.**
- **Confidence.** High. The rejection is pinned by an existing spec.

### GEO-10 A label point cannot recover the provenance promised in the overlay documentation [P1] [docs]

- **Evidence.** Overlay advises callers to “find the input features under an output polygon” by joining a point inside it to each operand. The newer `featureSources` documentation accurately limits its claim to a representative boundary owner. Point-in-polygon's primary output is a single containing feature ID per point. `S/polygon-overlay/gpu-polygon-overlay.ts:148`, `:161`; `S/polygon-overlay/overlay-feature-sources.ts:20`; `S/spatial-join/gpu-point-in-polygon-join.ts:164`.
- **Why it matters.** Dissolve two adjacent parcels. One output shell covers both, while an interior point away from their shared edge lies in just one parcel. Even a point-in-polygon implementation returning *all* matches at that point could not recover the other parcel.
- **Proposed change.** Correct the documentation immediately. For source membership, describe a polygon-to-polygon join with the intended relation, such as positive-area interior intersection, and identify boundary-only contacts separately. For attribute-preserving overlay, specify a feature-to-source relation with both operand and row. **S for docs, M/L for the relation, additive.**
- **Confidence.** High. **KNOWN(STD-08)** and **KNOWN(GAP-01)** track provenance broadly. The current workaround gives an incomplete answer.

### GEO-11 Grouped alpha hull expands a packed triangulation back to `groups × all points` [P1] [performance]

- **Evidence.** Grouped Delaunay already writes packed triangles and group offsets/counts. Grouped concave hull builds that packed result internally, but requires callers to reserve `getDelaunayMaximumTriangleCount(props.positions.length) * props.groupCount` triangle slots and three boundary slots per triangle. The bound is `2*n-5`. `S/delaunay-tessellation/gpu-delaunay-tessellation.ts:629`, `:675`; `S/concave-hull/gpu-concave-hull.ts:310`, `:337`, `:374`.
- **Why it matters.** At 100,000 total sites and 100 groups, the public boundary-edge view alone requires **479,988,000 bytes** (`199995 × 100 × 3 × 8`), even for evenly sized groups. That exceeds a 128 MiB storage binding before accounting for triangles or validity words.
- **Proposed change.** Offer `layout: 'packed'` with group offsets/counts and a compact retained-edge list. Reuse the existing group-major temporary data, and retain fixed group slots for current consumers. **M, additive.**
- **Confidence.** High. Allocation arithmetic and the existing packed intermediate are explicit.

### GEO-12 Multipart feature identity works in measures but disappears at the join boundary [P1] [architecture]

- **Evidence.** Measures and affine transform accept `featureOffsets` for multi-lines and multi-points. The shared line port describes one linestring per feature. Predicate join takes that port and treats a line row as a feature. A separate postorder tree can represent multi-lines, multi-points and nested collections for normalization. `S/geometry-measures/gpu-geometry-measures.ts:162`; `S/geometry-edit/gpu-affine-transform.ts:183`; `C/contracts/ports.ts:28`, `:122`; `S/spatial-join/spatial-join-types.ts:44`; `S/geometry-refinement/gpu-normalize-geometry.ts:24`.
- **Why it matters.** A road made of two disconnected lines can produce one length row but two join rows. Joining those rows back to the road table requires an owner map and predicate-specific aggregation. For example, an intersection with any part can establish feature intersection, whereas containment needs a different reduction.
- **Proposed change.** Add shared multipart point and line port variants, with an explicit feature-to-part level and a common row-owner adapter. Start with measures, affine transform and an intersection join. Document the reduction for each additional predicate before enabling it. **L, additive.**
- **Confidence.** High.

## Cross-cutting patterns

### Public input and output layouts

The table gives the row layout of **every contributor**, including the two foundations. Class names omit the common `GPU` prefix. `XY` means `float32x2` positions. `L` means XY plus terminal `uint32 lineOffsets`. `P` means `featureOffsets → polygonOffsets → ringOffsets → XY`, with the first ring of each polygon its shell and later rings its holes. `CSR` means offsets plus a compact neighbor array and aligned value columns. These layouts are defined in `C/contracts/ports.ts:21`, `:28`, `:37`, `:112`, `:145`. Optional diagnostics are summarized where they affect interpretation.

| Contributor | Input rows | Output rows and ownership |
|---|---|---|
| ArealInterpolation | Two zone-ID rasters on one grid, optional cell weights/categories | Target-to-source CSR weights, optional overlap/source areas, transferred fractions and category shares |
| Pycnophylactic | Zone raster and one total per zone | Mass and optional density per cell, iteration and unrepresented-zone status |
| BoundsFilter | Feature boxes as `float32x4`, packed or chunked | Compact selected row IDs and optional row-aligned boolean mask |
| ConcaveHull | XY sites and radius | Dense site-index triangle triples and boundary-edge pairs, each with validity |
| GroupConcaveHull | XY, group ID per site, number of groups | Same primitives in fixed group blocks, convergence per group |
| DelaunayTessellation | XY sites, optional group IDs | Compact counter-clockwise site-index triples, optional per-group offsets/counts and duplicate diagnostics |
| VoronoiDiagram | XY sites, triangle triples, live triangle count and clip box | Compact `float32x4` segments and two site rows per segment |
| DorlingCartogram | XY centers, values, optional source IDs | Centers, radii and source IDs per input row, metrics and status |
| AffineTransform | Points, lines or polygons, optional multipart ownership and transform origins | Transformed XY in the input row order, input offsets reusable |
| GeometryCleanup | Points, L or P | Compact positions and rewritten line/ring offsets, upper hierarchy retained. `count` counts vertices |
| GeometryOrientation | L or P for reversal, P for orientation | Same-layout positions, optional reversed-ring flags |
| GeometryMeasures | Points, L or P, including multipart point/line feature offsets | Per-feature measure columns and optional grouped summaries |
| GeodesicPairs | Aligned longitude/latitude origin and destination rows | Pair distances, bearings, midpoints and numerical status |
| GeodesicDestination | Longitude/latitude origins, bearings and distances | Destination XY, arrival bearing and status per origin |
| LocalFrameProjection | Longitude/latitude XY or binary64 words, frame and bounds | Frame-local XY in meters and optional positive validity, one row per input |
| GeometryPredicates | Point, L or P port and optional corresponding second geometry | Per-feature predicates/counts, optional compact unique positions and feature offsets |
| CanonicalGeometryEquality | Two normalized trees and explicit root-ID pairs | One three-valued equality code per pair |
| CanonicalPolygonEquality | Two normalized P batches, paired by feature row | One three-valued equality code per aligned feature |
| NormalizeGeometry | XY, postorder `uint32x4` node records and child IDs | Reordered coordinates/children plus per-node status, input nodes retained |
| NormalizePolygons | P | Canonical positions, ring and polygon offsets, input feature offsets retained |
| ConvexStraightSkeleton | XY and one strictly convex ring per offset interval | Fixed per-ring arc/source-edge and offset-position blocks, packed status |
| VectorStraightSkeleton | P | Per-polygon arcs/source-edge pairs and linked offset vertices in fixed blocks, packed status |
| MaximumInscribedCircle | P | Per feature, center/radius/error `float32x4`, boundary witness XY and status |
| LineLoopRemoval | L | Positions and line offsets, per-vertex source edges, packed vertex-count status |
| OffsetCurveComponents | Original and raw offset L, aligned by line | XY plus `[firstVertex, vertexCount, sourceLine, …]` component records and per-vertex source edges |
| PolygonPrecisionReduction | P and requested grid | Make-valid topology result plus the effective lattice |
| LinePrecisionReduction | L, grid and collapse mode | Positions, line offsets and status |
| RatioConcaveHull | XY sites, ratio and hole policy | Dense triangle/edge IDs and validity, using the alpha-hull output shape |
| SegmentSnap | L and reference XY | Fixed vertex slots per line, source edges, per-line counts and diagnostic bits |
| SubdividePolygons | P | Fixed-size vertex blocks, piece records with source feature/polygon/seed triangle, packed status |
| LineNoding | L with optional source IDs | Atomic `float32x4` segments, source feature/ring/edge IDs and parameter intervals |
| SnapRoundedNoding | L and requested grid | Noded-segment port plus lattice endpoints and effective lattice |
| Polygonize | L, optional groups and precision policy | Noded and directed edge tables, P with source groups, dangle/cut/open-chain diagnostics |
| MakeValid | P and precision policy | Same topology bundle, with faces filtered by source parity |
| GeometryValidity | Exactly one of points, L or P | Diagnostic word per feature and completeness status |
| GroupGeometry | XY, labels, optional weights | Counts, bounds, centers, medoid indices, ellipse and dispersion columns by group |
| GroupConvexHull | XY and labels | Group offsets/counts into source vertex indices, optional positions and pre-cap sizes |
| LabelPoint | P | One label XY per feature, optional boundary distance and degeneracy flag |
| LineClipByPolygon | L and P | Compact line pieces with source line rows and vertex counts. Polygon membership is used as a clip mask |
| SharedPaths | Left and right L | Run offsets/XY, both source line rows and relative direction per run |
| LineDensity | L and grid settings | Length and optional density per grid cell |
| LineLengthPerPolygon | L, P and optional line weights | Length, optional weighted length and segment count per polygon feature |
| LineMerge | L | XY and `chainOffsets`, optional representative source and inverse line-to-chain/order/reversal maps |
| GreatCircleArcs | Aligned source/target longitude/latitude XY | Bounded L, one line per pair, optional feature/vertex provenance, measures and approximation |
| LineChunk | L and chunking parameters | Bounded L with source line and optional vertex provenance/measures |
| LineSegmentize | L and segment-length parameters | Bounded L with line/vertex counts, optional source columns, measures and approximation |
| LineSmooth | L and smoothing parameters | Bounded L, optional source segment rows and interpolated original chainage |
| LineSimplification | L, optional timestamps, importance/settings | Per-vertex importance and optional compact selected vertex IDs, mask, line counts/starts |
| LineSplit | L | Generated L pieces with source line per piece and optional encoded source edge per vertex |
| LineLocate | L and events `(line row, measure, optional lateral offset)` | XY per event, optional segment, tangent, angle, status and cumulative vertex measures |
| LinearReferencing | L and query points/radius | Per-query nearest line/segment, fraction, foot, distance, chainage and side |
| LonLatPolygonPreparation | Longitude/latitude P and edge/seam policy | P with retained feature rows, doubled polygon/ring slots, optional ring flags. `count` counts vertices |
| MapColoring | Spatial weights/adjacency CSR | Color per row, optional color/conflict counts and iteration status |
| MinimumBounds | P or labeled XY | Per-feature/group rectangle corners and dimensions, circle, diameter, longest line or box |
| MinimumClearance | L or P | Per-feature clearance, witness segment and vertex IDs |
| NeighborSearch | Target XY, optional query XY, masks and times | Neighbor CSR with weights/distances and optional unclamped counts per query |
| OffsetExpansion | Terminal offsets and optional owner map | Owner and optional local index per child row |
| OffsetCurve | L or polygon rings | Fixed rows per source corner, or compact L with optional source line/ring and corner rows |
| OutlineGeometry | Points, L or P | Triangle-list XY, fixed stride or compact with input-vertex offsets and optional source rows |
| PairGather | Compact join pairs or dense nearest slots, optional unmatched rows and attributes | Left/right row table, gathered columns, `rowCount` and overflow |
| BufferSurface | L or P and buffer/kernel settings | Dissolved P, boundary edges and optional representative feature provenance |
| CoverageSimplification | P coverage | Simplified XY/ring offsets, optional vertex mask, topology statistics and status |
| CoverageDissolve | P coverage, optional feature labels | Ring-assembly result, optional polygon hierarchy and dissolve status |
| CoverageValidity | P coverage | Per-segment flags, optional feature invalid counts and validity |
| PolygonOverlay | Left P, optional right P, Boolean operation and lattice tolerance | Dissolved P, boundary edges with operand/source rows, optional representative owner per output feature |
| ConstrainedDelaunayTriangulation | P | Per-polygon index slices and validity as in core triangulation, plus flip convergence |
| RectangleClip | L or P plus box | Bounded line pieces or one retained slot per input ring, optional polygon feature port reusing input upper offsets |
| SegmentIntersection | One or two L/P ports | Compact segment-row pairs, optional crossing/overlap endpoints, kinds and feature/ring owners |
| ShapeDescriptors | P | Area, perimeter, compactness, elongation, orientation, convexity and sliver columns per feature |
| BufferSelection | Query points and point or start/end-segment features | Row mask and/or compact selected IDs, optional nearest/distance columns and completeness |
| NearestFeatureJoin | Query points/L/P and target points, segments, L or P | One nearest assignment per point, or dense query-by-slot IDs, counts, distances and witnesses |
| NearestFeatureWeights | Dense nearest slots/counts, optional distances | Sorted CSR neighbors and weights |
| PointInPolygonJoin | Query points and P, optional prepared target | One containing feature assignment per point, optional per-feature counts, compact point IDs and truncation flags |
| SpatialJoinCandidates | Left/right feature geometry and optional prepared right | Compact bounding-box candidate pairs in left/right row order |
| SpatialJoinPrepared | Feature geometry, optional caller storage and rebuild condition | Reusable graph-bound BVH/feature-range handle |
| SpatialPredicateJoin | Left/right point, L or P ports, predicate, optional attributes/prepared right | Compact matched pairs, relation/weights columns, or compact unmatched left rows in anti mode |
| StudyWindowPreparation | One planar polygon/multipolygon window | Reusable edge/slab/sampling-grid handle, optional eight-float summary |
| VertexSnap | XY and reference XY, optional aligned feature ranges | Same-layout XY and optional reference row per input vertex |
| PolygonTriangulation, core | P | Flat `uint32` indices in per-polygon slices, validity, optional real triangle counts and offsets. Slices can contain degenerate padding |
| SegmentRingAssembly, core | Directed endpoint tuples, optional live count and groups | Whole compact rings, optional area/hole/shell and segment maps, optional P. Polygon `sourceIds` can be shell group IDs |

### What section 8 actually buys a caller

**Geometry output is not uniformly `GPUCompactOutput`, and source rows are not uniformly available per feature.** The literal core type is an ID vector plus status. Seven contributor class files mention that type, including consumers and optional selection outputs. Only two class files directly mention `GPUGeneratedGeometryPort`, line split and rectangle clip. Line clip inherits the former's piece type. These are declaration counts, not a conformance score. The compact line layout is genuinely shared by five named producers. `C/kernels/gpu-contributor-types.ts:42`; `C/contracts/ports.ts:180`; `S/line-split/gpu-line-split.ts:75`; `S/rectangle-clip/gpu-rectangle-clip.ts:368`; `S/line-segmentize/line-segmentize-types.ts:29`.

Three distinctions deserve a contract rule and a checker:

- **Count the declared primary row.** A line output counts lines, an outline mesh counts triangle vertices, ring assembly counts rings even when it also publishes polygons, and longitude/latitude preparation counts vertices while retaining every feature. Record the row unit of both `count` and `requiredCount`. Require both units to agree wherever the generic cardinality port is used. `S/line-segmentize/line-segmentize-types.ts:49`; `S/outline-geometry/gpu-outline-geometry.ts:155`; `C/ring-assembly/gpu-segment-ring-assembly.ts:292`; `S/lon-lat-polygons/gpu-lon-lat-polygon-preparation.ts:71`.
- **Name the owner and the output it indexes.** In the shared line output, `sourceLineIndices` is per line and `sourceRows` is per vertex. In compact outline, `sourceRows` means the owner of a triangle vertex. In overlay, `featureSources.sourceRows` is per output polygon. Prefer `sourceFeatureRows`, `sourceVertexRows` or `sourceEdgeRows` in new APIs. Preserve old names as aliases where practical. `S/line-segmentize/line-segmentize-types.ts:67`; `S/outline-geometry/gpu-outline-geometry.ts:158`; `S/polygon-overlay/overlay-feature-sources.ts:20`.
- **Describe many-to-one ownership explicitly.** A merged chain has a representative *and* an inverse membership map. A shared path has two owners. A dissolved area can have many owners, including owners with no surviving boundary. These are different mathematical relations and merit different ports. `S/line-merge/gpu-line-merge.ts:76`; `S/line-clip/gpu-shared-paths.ts:71`; `S/polygon-overlay/gpu-polygon-overlay.ts:148`.

### Coordinate models, precision and degenerate input

**64 of 80** class bodies directly call `resolveGPUSpatialContext`. Four more forward context through another contributor: ratio hull, polygon precision reduction, polygon subdivision and constrained Delaunay. Geodesic destination uses the shared geodesic resolver. That gives **69 with a context prop**, plus local-frame projection with an explicit frame. Of the remaining ten, six are geometry-refinement classes: the two canonical equalities, the two normalizers, offset-curve components and vector straight skeleton. The other four are map coloring, offset expansion, pair gather and prepared join. The most consequential missing declarations are the distance-bearing offset and skeleton operations. Add optional planar context there and pass it through their operation forms. `S/geometry-refinement/gpu-offset-curve-components.ts:35`; `S/geometry-refinement/gpu-vector-straight-skeleton.ts:28`; `S/geometry-refinement/gpu-subdivide-polygons.ts:297`; `S/geometry-measures/gpu-geodesic-destination.ts:155`; `S/geometry-measures/gpu-local-frame-projection.ts:80`.

I found **25 context-taking contributors with a longitude/latitude route** by inspecting their support declarations and the geodesic delegate. This includes coordinate-only operations and operators consuming distances or raster metadata. The number does **not** mean 25 geodesic algorithms. Rectangle clip accepts straight edges in degrees and excludes a seam-crossing box. Neighbor search accepts great-circle distance and a sphere radius. Measures distinguishes great-circle and ellipsoidal metrics. Predicate join documents a local equirectangular foot approximation for geographic distance to segments. Those differences belong in the operator catalog alongside coordinate space. `S/rectangle-clip/gpu-rectangle-clip.ts:178`; `S/neighbor-search/gpu-neighbor-search.ts:139`; `S/geometry-measures/gpu-geometry-measures.ts:65`; `S/spatial-join/gpu-spatial-predicate-join.ts:339`.

Robustness likewise has several distinct contracts:

| Operation | What the implementation promises |
|---|---|
| Segment intersection | Exact orientation classification over its supported exponent range, explicit uncertain intersections. Zero-length/nonfinite segments are skipped. Segment IDs are starting vertex rows. `S/segment-intersection/segment-intersection-types.ts:16`, `:49`, `:96` |
| Snap-rounded noding and overlay | Exact topology on the effective lattice. The step can be coarser than requested because of input extent and coordinate magnitude. Noding exposes the effective lattice, and overlay documents its fill rule. `S/geometry-topology/gpu-snap-rounded-noding.ts:100`; `S/polygon-overlay/gpu-polygon-overlay.ts:276` |
| Delaunay and constrained Delaunay | Deterministic cocircular handling. Delaunay documents its exponent-span limit. Constrained triangulation separates a valid mesh from completed Delaunay refinement. `S/delaunay-tessellation/gpu-delaunay-tessellation.ts:728`, `:743`; `S/polygon-triangulation/gpu-constrained-delaunay-triangulation.ts:63` |
| Line clipping | Robust segment classification followed by a rounded midpoint decision. The implementation documents that sub-pieces within a few f32 ulps of the boundary can be classified on the wrong side. `S/line-clip/gpu-line-clip-by-polygon.ts:155`, `:168` |
| Core triangulation | Empty or degenerate shells can succeed with zero triangles. A failed polygon gets an entirely degenerate slice and a cleared validity flag. `C/polygon-triangulation/gpu-polygon-triangulation.ts:139`, `:150` |
| Ring closure | Assembly deliberately drops the repeated closing vertex. Validity defaults to implicit closure and can require explicit closure. Cleanup documents explicitly closed input. Normalized polygon equality preserves the difference in serialization. A graph can therefore preserve shape and change the answer to structural equality. `C/ring-assembly/gpu-segment-ring-assembly.ts:266`; `S/geometry-validity/geometry-validity-types.ts:77`; `S/geometry-edit/gpu-geometry-cleanup.ts:163`; `S/geometry-refinement/gpu-normalize-polygons.ts:35` |

### GeoArrow, capability gaps and overlap

**GeoArrow-compatible hierarchy is present. A scoped Arrow-to-analysis geometry adapter was not found.** A scan of the 244 source files found zero `from '…geoarrow…'` imports. The installed published graph package exposes `./gpu-project/geoarrow`. Its adapter declarations distinguish borrowing packed Float32 point/line chunks from nested polygons that remain GeoArrow columns. This checkout's installed graph package reports `10.0.0-alpha.14`, while `AGENTS.md` names alpha.13. I used the installed declarations only to identify the available public entry point. I found no explicit geometry-adapter roadmap row. `ARROW-01` concerns analytics types. `C/contracts/ports.ts:36`; `node_modules/@luma.gl/graph/package.json:6`, `:143`; `node_modules/@luma.gl/graph/dist/gpu-project/cpu-geoarrow.d.ts:3`; `dev-docs/roadmap.md:183`.

Several roadmap rows should be narrowed before assigning new work:

| Roadmap row | What is already implemented | What remains |
|---|---|---|
| **KNOWN(GEOM-01)** | Rectangle clip exposes a polygon feature port with reusable upper offsets. `S/rectangle-clip/gpu-rectangle-clip.ts:149`, `:368` | Topologically split disconnected clip pieces if required. Current concave ring clipping can keep zero-width bridges. `S/rectangle-clip/gpu-rectangle-clip.ts:214` |
| **KNOWN(GEOM-02)** | Line smooth publishes source segment rows and original chainage. `S/line-segmentize/gpu-line-smooth.ts:121`; `S/line-segmentize/line-smooth-metadata.ts:113` | Close the row if these are its acceptance criteria. |
| **KNOWN(STD-08)** | Offset curve and outline both offer compact layouts. Overlay and buffer have representative feature provenance. `S/outline-geometry/gpu-offset-curve.ts:180`; `S/outline-geometry/gpu-outline-geometry.ts:155`; `S/polygon-buffer/gpu-buffer-surface.ts:157` | Distinguish fixed-stride compatibility, exact feature cardinality and complete many-source membership. |
| **KNOWN(GAP-01)** | Overlay dissolve handles overlapping inputs. Line split handles mutual/self crossings. Polygon preparation splits at the antimeridian. Line locate places points at supplied measures. `S/polygon-overlay/gpu-polygon-overlay.ts:283`; `S/line-split/gpu-line-split.ts:152`; `S/lon-lat-polygons/gpu-lon-lat-polygon-preparation.ts:97`; `S/linear-referencing/gpu-line-locate.ts:172` | Attribute-preserving overlay, independently buffered features, Voronoi *cells*, arbitrary splitter inputs and generation of regularly spaced events are distinct remaining requests. Current Voronoi also documents empty output for two or collinear sites. `S/delaunay-tessellation/gpu-voronoi-diagram.ts:94` |

Two further expectations should be visible to GEOS/JTS/Shapely users. `GPUGeometryValidity` explicitly omits interior connectivity, nested holes and nested multipolygon components. `GPUStudyWindowPreparation` requires valid input and refers callers to that checker, so passing its current checks is insufficient as a general precondition. Also, `GPUArealInterpolation` consumes *rasterized zone IDs*. Its overlap areas describe common cells. A vector polygon-pair intersection-area table is a separate capability. `S/geometry-validity/gpu-geometry-validity.ts:144`; `S/study-window/gpu-study-window-preparation.ts:101`; `S/areal-interpolation/gpu-areal-interpolation.ts:114`.

For **KNOWN(DUP-01)**, preserve observable distinctions before merging APIs. Label point uses greedy grid refinement, while maximum-inscribed-circle returns a boundary witness and remaining search error. Radius hull and ratio hull share a mesh layout but use different retention rules. Nearest-feature join, linear referencing and neighbor search all answer “nearest”, yet respectively publish feature assignments/dense slots, line chainage/side, and CSR neighborhoods. Shared geometry access and row-domain types would remove duplication without making these outputs interchangeable. `S/label-point/gpu-label-point.ts:118`; `S/geometry-refinement/gpu-maximum-inscribed-circle.ts:50`; `S/concave-hull/gpu-concave-hull.ts:53`; `S/geometry-refinement/gpu-ratio-concave-hull.ts:32`; `S/linear-referencing/gpu-linear-referencing.ts:113`; `S/neighbor-search/gpu-neighbor-search.ts:123`.

## Keep

- **Keep both sides of shared-path provenance and the inverse map in line merge.** One design answers “which two lines produced this run?” The other answers “which lines, in what order and direction, formed this chain?” Both support attribute reconstruction without a spatial rejoin. `S/line-clip/gpu-shared-paths.ts:71`; `S/line-merge/gpu-line-merge.ts:76`.
- **Keep the distinction between valid geometry and algorithm convergence.** A constrained-Delaunay consumer can use a valid mesh after its flip budget expires and can separately decide whether it needs Delaunay completion. Maximum-inscribed-circle similarly gives the caller a remaining search error. `S/polygon-triangulation/gpu-constrained-delaunay-triangulation.ts:63`; `S/geometry-refinement/gpu-maximum-inscribed-circle.ts:50`.
- **Keep the study-window lifecycle documentation as a model for prepared geometry.** It explains who owns the handle, when to add it to the graph, what invalidates it, what gets rebuilt and which consumer work still runs each frame. `S/study-window/gpu-study-window-preparation.ts:96`, `:118`.
- **Keep independent geometric oracles.** The hole-bridge spec checks triangle location against both shell and hole, then checks area. That kind of invariant can catch an error shared by CPU and GPU twins, as the offset cancellation example illustrates. `modules/analysis/test/analysis-core/polygon-triangulation-hole-bridge.spec.ts:133`; `S/geometry-refinement/cpu-offset-curve-components-operation.ts:256`.

## Future

1. **Publish a geometry-port catalog with row units.** Record input feature kind, multipart level, closure, coordinate/edge model, primary output row, source-row domain and completeness fields. Generate the layout table and add a checker for mismatched count units. Start with joins and generated geometry. The cost is metadata maintenance, so validate it against exported props wherever possible.
2. **Add a small suite of producer-to-consumer graphs.** Start with nearest → gather using permuted IDs, predicate → gather with candidate loss, measures → geographic bounds filter, assembly → validity, and clip → triangulation. Include a unit-rescaling oracle for metric operations. These fixtures cover failures that isolated contributor specs and CPU/GPU agreement can both miss.
3. **Build an analysis-specific GeoArrow entry path on the published APIs.** Specify how null features, empty parts, sliced offsets, multipart ownership, explicit closure and `Float64` coordinates reach an analysis port. Use a local projected frame where f32 conversion would lose necessary precision. A polygon adapter may need allocation or projection even when simpler columns can share storage. Keep generic Arrow conversion upstream.
4. **Offer packed results when the implementation already produces them.** Start with grouped alpha hull, then assess fixed-block subdivision and skeleton outputs on real workloads. Packed offsets add a scan or indirection, while fixed slots remain appropriate when each group has a tight known bound.
5. **Introduce an explicit source-membership relation for surface operations.** A `(resultFeatureRow, operand, sourceFeatureRow)` table can represent multiple owners. Offer it separately from representative boundary ownership because it costs more and requires a precise definition of membership, especially for difference and boundary-only contact.

## Top 5

1. Fix scale-dependent fragment cancellation in both offset implementations and pin the reproduced three-vertex case.
2. Publish nearest feature rows separately from stable IDs and consume those rows in pair gather.
3. Propagate candidate overflow through pair gather and test the composed graph.
4. Make geographic bounds filtering work directly with the unwrapped bounds emitted by geometry measures.
5. Share a scale-safe circumcenter implementation across hull and Voronoi, with analytic fixtures at both exponent extremes.

## Questions for the lead

- Can we reserve names ending in `Rows` for zero-based positions in a specified table, and use `Ids` for caller-defined identity? That decision would resolve both the nearest/gather bug and several provenance ambiguities.
- For binary geometry, should the public contract distinguish a pair of source rows, a representative boundary owner and a complete source-membership relation? Each answers a different consumer question.
- Is the next interoperability target a GeoArrow column entering one contributor, or an entire analysis graph retaining Arrow feature identity through geometry generation? The latter makes multipart ownership and row-domain conventions prerequisites.