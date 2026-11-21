# Geometry, predicates, joins, and overlay from a GeoPandas user’s perspective

## Summary

The four-level polygon layout works for a real hole. On the same fixture, the CPU predicate join, nearest join, clipped line length, normalization, and both triangulators agreed with Shapely on membership, distance, or area. Reversing the hole or explicitly closing both rings preserved the join result.

Three compositions deserve attention before the alpha. **Adding identity multipart offsets changes polygon counts. Passing a capacity-sized overlay into an anti join can create unmatched features that never existed. A label point can report positive clearance at a coordinate on the polygon boundary.**

There are also two direction-dependent CPU results. Reversing a segment changed a radius-zero selection, and reversing a line changed its clipped length from zero to one half. Neither call reported incomplete work.

The most valuable API change is to make **logical feature count, assignment count, and membership count explicit at the producer and directly consumable by the next operator**. This addresses ordinary joins and overlay composition while preserving caller-owned graphs.

## Census

`R` means the requested `luma-contributors-r19` checkout. `K` means the newer `luma-contributors-ki` copy of a staging-owned file. Source paths below are relative to `modules/analysis/src/gpu-spatial-analysis/` unless they start with `docs/`, `modules/`, or `../analysis-core/`. I checked candidates against the synthesis, geometry, API/catalog and direction reports, other area reports by symbol, and `NEEDS-DECISION.md`.

I enumerated the requested directory names with `fs.readdirSync`, then followed each `index.ts`’s relative value re-exports with `typescript.createSourceFile` under `node -e`, without invoking the compiler. For the second pass, each path on `staging-owned.txt` resolved to K. As an independent R check, I bundled imports of the 32 family barrels through esbuild on stdin and counted their runtime export names.

| Public surface in the scoped barrels | R | R with K-owned files substituted |
|---|---:|---:|
| Family directories | 32 | 32 |
| GPU classes excluding `*Operation` | 70 | 70 |
| GPU `*Operation` classes | 17 | 17 |
| CPU `*Operation` classes | 17 | 17 |
| `compute*OnCPU` functions | 19 | 22 |

The R runtime export counts for `^GPU[A-Z][a-zA-Z0-9]+$`, `^GPU.*Operation$`, `^CPU.*Operation$`, and `^compute.*OnCPU$` were `[87,17,17,19]`. The three additional K functions were `computePointInPolygonJoinOnCPU`, `computeBufferSelectionOnCPU`, and `computeLineLengthPerPolygonOnCPU`.

The following is a behavioral census. Each row names the entire population **tested for that comparison**.

| Test | Variant and count | Public operations |
|---|---|---|
| One real empty MultiPolygon, no parts, no rings, no vertices | Accept 5 of 9 | Predicate join, point-in-polygon join, line length per polygon, polygon triangulation, constrained triangulation |
| Same empty input | Throw 4 of 9 | Label point, maximum inscribed circle, normalize polygons, minimum bounds |
| Four actual vertices, terminal ring offset `999` | Reject 5, return invalid 2, return ordinary success 2 | Reject in both joins, line length, normalization, minimum bounds. Invalid in both triangulators. Success in label point and maximum inscribed circle |
| `1e40` passed to a finite float32 setting | Reject 2 of 5 | Grid generator and shape generator |
| Same conversion | Pack infinity 3 of 5 | Geometry cleanup, rectangle clip, line simplification |
| One point in two polygons, then the same point with identity grouping | Invariant 2 of 3 PIP result columns | `pointMatchCounts` and `featurePolygonMembership` agree. `featureCounts` changes |
| Three compact row results composed around a predicate join | `status.count` in 2, `output.rowCount` in 1 | Predicate pairs and offset expansion use `status.count`. Pair gather uses `output.rowCount` |

### Fifteen calls from the operator pages

These are the short calls I derived from the pages, with caller-owned views abbreviated. `P` is polygon geometry, `Q` is point geometry, `L` supplies line fields, `xy` is a position view, and `out` stands for that operator’s typed outputs. I checked the shapes in the declarations and ran the corresponding host function or CPU twin. For triangulation I compared the union’s expected area, independently calculated from the returned triangles. I did not require GEOS to choose the same diagonals.

The common fixture was a `4 × 4` square with a `1 × 1` hole from `(1,1)` to `(2,2)`. Query rows were `(.5,.5)`, `(1.5,1.5)`, and `(1,1.5)`. Reference was `/usr/bin/python3`, Shapely **2.0.6**, GEOS **3.11.4**.

| Documented call shape | Executed check |
|---|---|
| `new GPUSpatialPredicateJoin({left: Q, right: P, predicate: 'within', output: {pairs}})` | `within → [0]`, `coveredBy → [0,2]`, `touches → [2]`, `intersects → [0,2]`. All four matched Shapely. |
| `new GPUPointInPolygonJoin({points: Q, polygons: P, candidateCapacity: 8, output: out})` | One point in two polygon rows gave membership `[1,1]`. Counts depended on representation, UGE-01. |
| `new GPUNearestFeatureJoin({queries: Q, features: P, k: 1, output: out})` | Distances `[0,.5,0]`, including distance to the hole boundary. Shapely agreed. |
| `new GPUBufferSelection({points: Q, features: segments, distance, candidateCapacity: 4, output: out})` | A radius-zero selection changed when the reference segment was reversed, UGE-06. |
| `new GPULineLengthPerPolygon({...L, polygons: P, output: out})` | Line `(-1,1.5)→(5,1.5)` gave length `3`, matching `line.intersection(P).length`. UGE-07 gives a failing case. |
| `new GPULabelPoint({...P, output: {points, distances, degenerate}})` | Unit square gave `(.5,.5)`, radius `.5`, `degenerate=0`. The projected-coordinate case failed, UGE-05. |
| `new GPUMaximumInscribedCircle({...P, tolerance: .00001, output: out})` | The returned center and witness coincided while the returned radius was positive, UGE-05. |
| `new GPUMinimumBounds({positions: xy, labels, groupCount, output: out})` | Correct envelope for one group. Adding a distant second group collapsed the first envelope, UGE-04. |
| `new GPUNormalizePolygons({...P, output: out})` | Clockwise shell, counterclockwise hole, same cyclic sequences as Shapely normalization. Open storage stayed open. |
| `new GPUPolygonTriangulation({...P, output: out})` | Eight triangles, total area `15`, `valid=1`. |
| `new GPUConstrainedDelaunayTriangulation({...P, output: out})` | Eight triangles, total area `15`, `valid=1`. |
| `new GPUDelaunayTessellation({positions: xy, output: out})` | Triangle plus a duplicate gave `count=1`, `duplicateCount=1`. Three collinear points gave zero triangles. Shapely agreed. |
| `new GPURatioConcaveHull({positions: xy, ratio: 1, output: out})` | Triangle plus duplicate gave three boundary edges. Collinear input gave zero area edges. Shapely returned a `LineString` for the latter, consistent with the documented area-only distinction. |
| `new GPUSegmentSnap({...L, referencePositions, tolerance: .25, maximumVerticesPerLine: 4, output: out})` | `(0,0)→(2,0)` snapped through `(1,.125)`, exactly the Shapely coordinate sequence. |
| `new GPULineSimplification({...L, importance, parameters, output: out})` | At tolerance `.5`, `(0,0),(1,.5),(2,0)` retained the endpoints, matching Shapely Douglas–Peucker with topology preservation disabled. UGE-08 gives a case that loses both. |

For the hole fixture I also ran all four combinations of explicit/implicit closure and original/reversed hole winding. `within` selected only row `0` in all four. Normalization returned ring offsets `[0,4,8]` for open storage and `[0,5,10]` for closed storage. This distinction matters when testing structural equality.

**What `sjoin(points, polygons)` costs.** The K join page’s example is seven lines from `graph.add` to `));`, with five geometry input views and four pair-output views already supplied. The in-repository attributed example takes **23 lines** for the two `graph.add` calls, excluding its intervening histogram call, after allocation and before submission or readback. The join produces row indices. Gather takes an explicit source and destination for each resident attribute. A left join through those documented ports also needs unmatched rows, UGE-12. [Join example](</Users/robertchristie/Documents/GitHub/luma-contributors-ki/docs/api/gpu-spatial-analysis/operators-joins.md:302>) · [Attributed example](</Users/robertchristie/Documents/GitHub/luma-contributors-r19/modules/lenses/src/computation-lenses/scenarios/gpu-spatial-analysis/pair-gather.example.ts:170>)

For host-resident attributes the shorter route actually ran. With the packed fixture above and `join = computeSpatialPredicateJoinOnCPU`, this returned `[['A','Park']]`:

```ts
const ids = ['A', 'B', 'C'], zones = ['Park'];
const {leftRows, rightRows, status} = join(left, right, {
  predicate: 'within', candidateCapacity: 3, pairCapacity: 3
});
if (status.overflow || status.candidateOverflow) throw new Error('Incomplete join');
const rows = Array.from({length: status.count}, (_, k) => [ids[leftRows[k]], zones[rightRows[k]]]);
```

## Proposed convention

A geometry consumer should take the producer’s hierarchy, logical feature count, and completeness evidence directly. A named result should preserve its meaning under identity grouping, capacity padding, ring closure allowed by the contract, and reversal of an undirected segment. Check those transformations on the same fixture, comparing live rows, coordinates, and attributes against Shapely.

## Findings

| ID | Kind | Reach | Change | Cost | Where | One line |
|---|---|---|---|---|---|---|
| UGE-01 | C/W | ORDINARY | ADDITIVE | M | Point-in-polygon | Identity grouping changes what a polygon count measures. |
| UGE-02 | W | ORDINARY | ADDITIVE | M | Overlay → anti join | Spare feature slots can become unmatched features. |
| UGE-03 | W | ORDINARY | BREAKING results | S | Cleanup and coverage | Float32 half-up arithmetic can move a point already on its grid. |
| UGE-04 | W | UNUSUAL | BREAKING results | M | Minimum bounds | Another group can collapse this group’s envelope. |
| UGE-05 | W | UNUSUAL | BREAKING results | M | Label point and inscribed circle | A published boundary point carries positive interior clearance. |
| UGE-06 | W | UNUSUAL | BREAKING results | S | CPU buffer selection | Reversing a segment changes an exact-hit selection. |
| UGE-07 | W | EXTREME | BREAKING results | M | CPU line length per polygon | Reversing a segment recovers its entire clipped length. |
| UGE-08 | W | EXTREME | ADDITIVE validation | S | Three parameter packers | A finite JS number becomes infinity in the submitted settings. |
| UGE-09 | D/S | ORDINARY | ADDITIVE | S | Shape generator → polygon join | Its documented polygon offsets address the wrong level. |
| UGE-10 | S/C | ORDINARY | ADDITIVE | M | Nine polygon CPU functions | A canonical all-empty hierarchy works in five and throws in four. |
| UGE-11 | E | ORDINARY | ADDITIVE | S | CPU label point | A four-vertex rectangle exceeded a three-second deadline. |
| UGE-12 | E | ORDINARY | ADDITIVE | M | Predicate join → pair gather | An attributed left join asks for a second spatial classification. |
| UGE-13 | C/S | ORDINARY | ADDITIVE | S | Pair gather | Copying the producer’s cardinality layout fails late in JS. |
| UGE-14 | D | ORDINARY | ADDITIVE | S | Join reference | Three GEOS comparisons need counterexamples or qualification. |
| UGE-15 | E | ORDINARY | ADDITIVE | M | CPU point-in-polygon | Sparse queries unconditionally allocate a dense membership matrix. |
| UGE-16 | C | ORDINARY | ADDITIVE | S | Three equality contributors | The same tolerance selects two different distance neighborhoods. |

“BREAKING results” means correcting a returned value changes observable behavior. None of those fixes requires a caller syntax edit. The call counts below are named invocation matches in `modules`, `examples`, and `website`, including tests, with K-owned paths substituted. They measure the regression surface, not the number of callers known to encounter the defect.

### UGE-01: A singleton MultiPoint changes polygon counts

**Where.** K `spatial-join/cpu-point-in-polygon-join-operation.ts:123`, `gpu-point-in-polygon-join.ts:1409` and `:1441`. The K join page describes the multipart rule at `docs/api/gpu-spatial-analysis/operators-joins.md:121`.

**EXECUTED.** One point `(.5,.5)`, two identical square features, ample capacity. Ordinary point input returned `pointMatchCounts=[2]`, `featurePolygonMembership=[1,1]`, `pointFeatureIds=[0]`, **`featureCounts=[1,0]`**. Adding only `points.featureOffsets=Uint32Array.of(0,1)` changed the last column to **`[1,1]`**. Both overflow flags were zero. Shapely found both memberships for both `Point` and singleton `MultiPoint`.

**Now / wanted.** `groupby(polygon).size()` after a spatial join means memberships. Counting the lowest-row winner answers an equally legitimate assignment question. Expose `polygonMembershipCounts` and `polygonAssignmentCounts` with those meanings for both inputs. Deprecate ambiguous `featureCounts`. This extends KNOWN(GEO-12) with a reproduced reduction difference in the newer multipart implementation.

**Callers / value.** No required migrations for the additive ports. Polygon summaries become invariant to an input driver’s choice of Point versus MultiPoint. **M.**

### UGE-02: An overlay’s spare capacity can create unmatched features

**Where.** K `polygon-overlay/gpu-polygon-overlay.ts:429`, K `../analysis-core/ring-assembly/gpu-segment-ring-assembly.ts:2074`, K `spatial-join/spatial-join-geometry.ts:59`, K `spatial-join/gpu-spatial-predicate-join.ts:2397`.

**SOURCE + EXECUTED host replay.** Ring assembly pads feature offsets with `min(index, polygonCount)`. The join takes its number of polygon features from `featureOffsets.length - 1`. Its anti pass selects rows with zero matches. I replayed one square containing the right-hand point. With live features `[0,1]`, CPU anti returned `[]`. With `[0,1,1,1]` and trimmed child arrays it returned **`[1,2]`**. Giving the CPU join all three raw padded offset arrays instead threw `polygonOffsets must span all child rows`.

**Now / wanted.** `left: overlayOutput.geometry` discards `overlayOutput.status.count`. Support a resident `leftFeatureCount` or a counted geometry operand carrying that scalar and completeness. Preserve genuine empty features inside the live prefix. Provide a host decoder that slices all child levels from their terminal live offsets.

**Smallest GPU probe.** Dissolve one square into ring capacity three. Anti-join the result against one interior point. Expect zero unmatched rows, including on reuse after a larger first frame. This is distinct from boundary provenance, KNOWN(GEO-10), and stale pair tails, KNOWN(GEO-127).

**Callers / value.** None for an optional counted operand. An overlay can feed a left or anti join without readback or invented records. **M.**

### UGE-03: Pointwise grid snapping can change an exact grid coordinate

**Where.** R `geometry-edit/geometry-edit-kernels.ts:397`, `geometry-edit/gpu-geometry-cleanup.ts:232`, `polygon-coverage-topology/coverage-topology-kernels.ts:58`, `polygon-coverage-topology/gpu-coverage-dissolve.ts:436`, K `polygon-coverage-simplification/gpu-coverage-simplification.ts:492`. Compare the existing precision-reduction implementation in K `geometry-refinement/gpu-precision-reduction.ts:573`.

**SOURCE for shaders, EXECUTED for float32 model and Shapely.** With `gridSize=1`, the cleanup expression `floor(f32(x + .5))` gave:

| Stored float32 `x` | Expression | Shapely pointwise precision |
|---:|---:|---:|
| `8388609` | `8388610` | `8388609` |
| `0.4999999701976776` | `1` | `0` |
| `0.5` | `1` | `1` |

The first case is inside cleanup’s stated `|x/gridSize| < 2^24` range. The coverage helper guards the large-integer case, but still adds `.5` in the small case. Coverage uses this result as a vertex key, so the consequence can extend beyond display coordinates.

**Wanted / GPU probe.** Reuse the precision reducer’s half-grid decision. Run cleanup on those three points with repeats disabled. For coverage, anchor at zero and test whether the immediate float32 predecessor of `.5` receives the same key as zero.

**Callers / value.** Zero syntax edits. Constructor matches were cleanup **22**, dissolve **27**, coverage simplification **31**. Grid-aligned inputs remain fixed and nearby points enter the intended cell. **S.**

### UGE-04: Adding an unrelated group can shrink an envelope to a point

**Where.** R `minimum-bounds/cpu-minimum-bounds-operation.ts:93`, `:264`, `:293`, `:373`; R `minimum-bounds/gpu-minimum-bounds.ts:727`. K `docs/api/gpu-spatial-analysis/operators-geometry.md:481` calls the result the feature envelope.

**EXECUTED.** Group 0 was a square from `(0,0)` to `(.01,.01)`. Alone, it had four hull vertices and bounds `[0,0,.0099999997765,.0099999997765]`. Appending group 1 containing only `(10000000,0)` reduced group 0 to **one hull vertex, zero diameter, and bounds `[0,0,0,0]`**. Both overflow words were zero. Shapely’s bounds on the stored float32 coordinates were unchanged.

**Now / wanted.** The CPU chooses one lattice from all groups and calculates bounds from surviving hull representatives. Compute `bounds` from each group’s finite source positions. For the hull-dependent outputs, use per-group conditioning or publish a precision-loss diagnostic. KNOWN(GEO-106) concerned circle arithmetic after hull construction. This failure occurs before it.

**GPU probe.** Run those two groups, requesting only `bounds` and `hullSizes`, then remove group 1. **Callers / value.** Zero syntax edits, **12** pure CPU and **18** GPU constructor invocations to cover. Viewport filtering can rely on the envelope enclosing its feature. **M**, with the bounds-only correction smaller.

### UGE-05: Returned center, witness, and radius can contradict each other

**Where.** R `label-point/cpu-label-point-operation.ts:281`; K `geometry-refinement/cpu-maximum-inscribed-circle-operation.ts:183`.

**EXECUTED.** A valid square had lower corner `(1048576,1048576)` and side `.125`. All input vertices are exactly representable in float32. Label point returned **the lower corner**, distance `.0625`, `degenerate=0`. Inscribed circle returned that same center and witness, radius `.0625`, status `0`, remaining search error approximately `.00000632`. Shapely found that the published point was on the boundary, clearance `0`. Its interior pole was `(1048576.0625,1048576.0625)`.

**Now / wanted.** `distances[0] > 0` cannot certify the published label here. Both twins round the coordinates separately from the score computed at the unrounded center. Re-evaluate clearance at the published center. Report a precision-limited result when no representable interior center exists. Document the local-origin remedy next to the returned columns. The circle docs already exclude roundoff from the search-error bound. The separate defect is the inconsistency among the returned geometry fields.

**Callers / value.** Zero syntax edits for consistent distances. **10** label-point and **9** circle CPU invocations. A subsequent point-in-polygon test can agree with the label’s clearance. **M.**

### UGE-06: CPU radius-zero buffer selection depends on segment direction

**Where.** K `spatial-join/cpu-buffer-selection-operation.ts:105`, `:115`; compare K `spatial-join/cpu-feature-join.ts:484`.

**EXECUTED.** Query `(.5,.5+2^-24)`, segment `(-2^30,-2^30)→(1,1)`, distance `0`. `computeBufferSelectionOnCPU` selected row `0` with distance `0`. Reversing the segment returned no match. On the same inputs, predicate `dwithin(0)` returned no pair and nearest join reported `4.214684778958144e-8`. Shapely `intersects` was false. GEOS 3.11.4’s numeric distance itself lost precision in the forward direction and recovered `4.214684851089403e-8` in reverse, so `intersects` is the appropriate zero-radius oracle here.

**Wanted.** Once the exact on-segment check is false, reject at radius zero. For positive radii, reuse the predicate join’s robust threshold comparison. Test both endpoint orders.

**Callers / value.** Zero syntax edits, **2** named CPU invocations. Exact-hit selection and exact-hit spatial join agree. **S.**

### UGE-07: CPU clipped length loses a representable intersection

**Where.** K `line-density/cpu-line-length-per-polygon-operation.ts:116`, `:135`, `:146`.

**EXECUTED.** `LineString([(-2^56,.5),(1,.5)])` crossing `box(0,0,.5,1)` gave `lengths=[0]`, `segmentCounts=[0]`, both overflow flags zero. Reversed, it gave **`.5`**. Shapely returned `.5` in both directions. The two boundary crossings round to the same fraction near the distant segment’s terminal endpoint.

**Wanted.** Retain local crossing coordinates or exact ordered parameters for the CPU clipping interval. Include a direction-invariance test. For the GPU’s documented lower-precision route, the smallest probe is the same box with the distant endpoint at `-2^28`, in both orders, recording length and status. GPU behavior remains **SOURCE**.

**Callers / value.** Zero syntax edits, **2** named CPU invocations. The CPU reference remains useful for checking a narrow zone crossed by a long line. **M.**

### UGE-08: Finite host parameters can pack as infinity

**Where.** R `geometry-edit/gpu-geometry-cleanup.ts:88`, R `line-simplification/line-simplification-parameters.ts:57`, K `rectangle-clip/gpu-rectangle-clip.ts:109`. The contrasting implementations are R `grid-generators/gpu-grid-generator.ts:95` and `gpu-shape-generator.ts:95`.

**EXECUTED.** `1e40` produced `[Infinity,0,1,0]` for cleanup grid size, `[Infinity,0,0,0]` for simplification tolerance, and `[0,0,Infinity,1]` for a rectangle. The grid and shape generators rejected the same magnitude. Feeding the packed simplification tolerance to the CPU twin returned **zero vertices**, including neither endpoint. Shapely retained both endpoints. Cleanup grid size `1e-50` packed to `0`, its disable switch.

**Wanted.** Validate `Math.fround(value)` before writing. Reject a positive setting rounding to zero when zero disables the operation. Include the contributor and setting in the error. This extends the finite-conversion issue recorded for spatial clustering in the core report to three specific geometry packers.

**Callers / value.** No valid call changes. Failure moves to parameter construction, before a user consumes a plausible empty result. **S.**

### UGE-09: Shape-generator offsets cannot also be polygon offsets

**Where.** R `grid-generators/gpu-shape-generator.ts:111`, `:125`; K `docs/api/gpu-spatial-analysis/operators-geometry.md:1101`; polygon validation in K `spatial-join/cpu-feature-join.ts:140`.

**EXECUTED.** The generator’s type says its offsets can serve as both ring and polygon offsets. A three-segment circle has four stored vertices. Following that sentence gives `ringOffsets=[0,4]`, `polygonOffsets=[0,4]`, `featureOffsets=[0,1]`. The join throws `CPUFeatureJoin polygonOffsets must span all child rows`. The correct polygon offsets are **`[0,1]`**.

**Wanted.** Document the three arrays for two generated features: feature and polygon offsets `[0,1,2]`, ring offsets `[0,V,2V]`. Expose a generated polygon descriptor or an identity-offset helper. Inline the example in the generator page. `GPURectangleClip.getPolygonOutput()` is an existing API precedent.

**Callers / value.** None. A caller can feed generated circles to a join without first debugging the GeoArrow hierarchy. **S.**

### UGE-10: Canonical empty polygons fail where malformed offsets succeed

**Where.** R `shared/cpu-polygon-rows.ts:39`, `:52`; R `geometry-refinement/cpu-canonical-geometry.ts:56`; R `minimum-bounds/cpu-minimum-bounds-operation.ts:60`; K `spatial-join/cpu-feature-join.ts:140`. All nine functions are named in the census.

**EXECUTED.** One empty MultiPolygon is encoded by `positions=[]`, `ringOffsets=[0]`, `polygonOffsets=[0]`, `featureOffsets=[0,0]`. Label point, inscribed circle, normalization, and minimum bounds all threw at the first offset table. The two joins and line length accepted it. Both triangulators returned empty per-polygon arrays. Conversely, label and circle accepted a four-vertex square with `ringOffsets=[0,999]`, reporting `degenerate=0` and status `0`. Shapely recognizes the empty MultiPolygon as a geometry.

**Wanted.** Accept a one-entry terminal wherever the child count is zero and preserve the declared feature row. Give compact host input a shared hierarchy validator. Keep deliberately clamped resident-storage behavior available where the contract requires it. This is a concrete acceptance test for the GeoArrow adapter in KNOWN(DIR-102).

**Callers / value.** None for accepting empty inputs or adding strict host validation as an option. Empty filtered batches can move through the family without dummy rings. **M.**

### UGE-11: CPU label-point search spends seconds on an ordinary rectangle

**Where.** R `label-point/cpu-label-point-operation.ts:144`, `:201`, `:204`, `:231`, `:274`.

**EXECUTED.** Fresh child processes, same four vertices apart from rectangle width, height `1`. Width `1` returned in about `1.6 ms`, `1.001` in `8.8 ms`, and `1.01` in `53 ms`. Width `2` exceeded a **three-second deadline**. These are single observations, not a benchmark. The correct radius is `.5`; Shapely returned the center `(1,.5)`.

**Now / wanted.** The search already finds a point of radius `.5`, then keeps subdividing cells along a plateau to prove it. Cap a feature’s possible radius by half the shorter side of its axis-aligned envelope. Once the incumbent meets that bound, return. This costs less than changing the queue structure and directly addresses this reproduction.

**Callers / value.** None. A four-vertex parcel becomes a practical first CPU example. **S.**

### UGE-12: A left attribute join needs a second spatial classification

**Where.** K `spatial-join/gpu-spatial-predicate-join.ts:881`, `:890`; R `pair-gather/gpu-pair-gather.ts:301`; K `docs/api/gpu-spatial-analysis/operators-joins.md:328`.

**EXECUTED host construction.** `new GPUPairGather({how:'left', pairs, output})` reported `how 'left' with pairs needs unmatchedLeft (a predicate join anti output)`. A predicate join requested to write both `pairs` and `unmatched` reported `output.unmatched needs how 'anti'`. The documented path therefore asks for an inner predicate join, an anti predicate join, and gather. Nearest-list gather already derives unmatched queries.

**Wanted.** Allow `output.unmatched` alongside inner pairs. Compute it from complete per-left match counts before final pair truncation. Preserve the existing exclusion of candidate-truncated rows. Add one example gathering a right attribute for an interior point, a boundary point, and an unmatched point. This gives KNOWN(DIR-103) a narrower implementation than a new execution facade.

**Callers / value.** None. A left join can reuse one spatial search and one set of candidate diagnostics. **M.**

### UGE-13: Pair gather relocates cardinality and misses the resulting constructor error

**Where.** R `pair-gather/gpu-pair-gather.ts:75`, `:309`, `:329`; R `offset-expansion/gpu-offset-expansion.ts:42`.

**EXECUTED host construction and node declaration.** With real transient graph views and a metadata-only device stub, the documented `output.rowCount` produced two command nodes. Using the producer-style `output.status.count` instead, with no `rowCount`, passed construction and later threw **`Cannot read properties of undefined (reading 'byteOffset')`**. This affects JS callers and outputs assembled from spreads. TypeScript’s declared output requires `rowCount`.

**Wanted.** Accept `output.status.count` as the canonical spelling. Retain `rowCount` as an alias and require exactly one destination or the same view under both names. Validate at construction. Document the count to use when reading gathered columns; `matchedCount` has a different meaning for left joins.

**Callers / value.** None for the alias. **41** existing gather constructor invocations can continue using `rowCount`. A pair consumer can carry the compact result shape forward. **S.**

### UGE-14: Three GEOS comparisons lead to different queries

**Where.** K `docs/api/gpu-spatial-analysis/operators-joins.md:213`, `:285`, `:756`; R `docs/api/gpu-spatial-analysis-cross-reference.md:218`; K `spatial-join/gpu-nearest-feature-join.ts:1492`.

| Page guidance | Executed counterexample | Exact documentation change |
|---|---|---|
| Use anti-`intersects` “for disjoint” | Left point `0`, right points `0` and `1`. CPU anti returned `[]`. Shapely pairwise disjoint returned `[(0,1)]`. | Say “left features disjoint from **every** right feature.” State that enumeration of disjoint pairs is unavailable. |
| `touches` means the boundaries meet | A point on the hole edge had an empty point boundary. Shapely returned `touches=True`, matrix `F0FFFF212`; CPU join also selected it. | Say “the geometries intersect and their interiors do not intersect.” |
| `exclusive` is cross-referenced to `STRtree.query_nearest(exclusive=True)` | For targets `(0,0),(0,0),(2,0)` Shapely returned row `[2]`. The GPU filter compares IDs. With default row IDs, row 1 survives the exclusion of row 0. | Call it an equal-ID exclusion in the cross-reference, and include the duplicate-coordinate case beside the option. |

**Evidence limit.** The last GPU result is **SOURCE**. Smallest probe: one point query at `(0,0)`, those three targets, `exclusive:true`, and one neighbor. Read the winning row and distance. Also distinguish the join’s no-candidates-for-empty rule from aligned topological equality: two real empty operands yielded CPU join count `0`, whereas Shapely `equals` was true, matrix `FFFFFFFF2`.

**Callers / value.** None. A GIS developer can choose the correct operation without learning these distinctions from a production mismatch. **S.**

### UGE-15: CPU PIP materializes the Cartesian product even with no candidates

**Where.** K `spatial-join/cpu-point-in-polygon-join-operation.ts:39`, `:88`, `:108`, `:123`, `:179`; its public export at K `spatial-join/index.ts:83`.

**EXECUTED.** With all points outside every polygon, `candidateCapacity=1`, `matchCapacity=0`, a `4 × 4` query allocated **64 bytes** of membership and a `32 × 32` query allocated **4,096 bytes**. Both reported zero candidates. The public pure function always requests the dense matrix. At `50,000 × 50,000`, that array alone would require **10 GB**; I did not attempt that allocation. Even the internal counts-only path scans polygon rows per feature.

**Wanted.** Expose an explicit counts/assignments-only host call or output selection, reusing the existing internal route first. Then drive membership and counts from sparse candidate pairs. Document `4 × pointFeatureCount × polygonFeatureCount` bytes beside the **CPU** call and distinguish it from the GPU’s opt-in dense output.

**Callers / value.** None for an additive route. CPU-first users can obtain one assigned polygon per point without allocating a matrix unrelated to their result size. **M**, with the docs and public counts-only entry point **S**.

### UGE-16: Equality tolerance changes metric between siblings

**Where.** K `geometry-predicates/gpu-geometry-predicates.ts:1128`; R `geometry-refinement/gpu-canonical-geometry-equality.ts:107`, `gpu-canonical-polygon-equality.ts:135`; K `docs/api/gpu-geometry-refinement.md:12`.

**SOURCE + EXECUTED reference.** For points `(0,0)` and `(.75,.75)`, tolerance `1`, Shapely `equals_exact` returned false. Euclidean displacement is `1.06066017`; maximum coordinate displacement is `.75`. `GPUGeometryPredicates.equalsExact` uses the first metric. Both canonical equality contributors compare each coordinate independently. Switching to canonical equality to ignore ordering also changes which displacements pass.

**Wanted / GPU probe.** Name the metrics in the comparison guide. Add `toleranceMetric: 'per-coordinate' | 'euclidean'` to canonical equality while preserving the current default. Run the two points through the tree comparison and repeat with correspondingly translated normalized triangles for polygon equality. Expect the two metrics to disagree at `1`.

**Callers / value.** None for an optional metric. Users can change normalization policy independently of measurement policy. **S.**

## Departures by name

- **Logical rows and multiplicity.** `GPUPointInPolygonJoin` and `computePointInPolygonJoinOnCPU` give one polygon-count name two reductions. `GPUPolygonOverlay` publishes live feature count separately from its padded geometry. `GPUSpatialPredicateJoin` derives logical polygon rows from allocated feature offsets. `GPUPairGather` relocates the live count.
- **Empty hierarchy.** `computeLabelPointOnCPU`, `computeMaximumInscribedCircleOnCPU`, `computeNormalizePolygonsOnCPU`, and `computeMinimumBoundsOnCPU` reject the canonical all-empty input. The first two also clamp the malformed terminal in the census to an apparently successful geometry.
- **Precision and invariance.** `GPUGeometryCleanup`, `GPUCoverageDissolve`, and `GPUCoverageSimplification` use the half-add rounding sites. `GPUMinimumBounds` and `computeMinimumBoundsOnCPU` derive bounds from a shared-lattice hull. The two pole functions publish independently rounded center and clearance. `computeBufferSelectionOnCPU` and `computeLineLengthPerPolygonOnCPU` failed segment reversal.
- **Layout, units, and vocabulary.** `GPUShapeGenerator` misidentifies an offset level. `getGPUGeometryCleanupParameterValues`, `getGPURectangleClipParameterValues`, and `getGPULineSimplificationParameterValues` accept a value that packs to infinity. `GPUCanonicalGeometryEquality` and `GPUCanonicalPolygonEquality` use a different tolerance metric from `GPUGeometryPredicates.equalsExact`. `GPUNearestFeatureJoin.exclusive` compares identities.

## Quick wins

- [ ] Correct the shape generator’s two-feature offset example and test passing its layout to a polygon consumer.
- [ ] Add the disjoint-pair, boundary-point, and duplicate-nearest examples to both join references.
- [ ] Validate packed float32 settings before writing them.
- [ ] Use the exact on-segment decision for radius-zero CPU buffer selection.
- [ ] Apply the envelope inradius upper bound in CPU label-point search.
- [ ] Accept `status.count` in pair gather and diagnose a missing destination at construction.
- [ ] Document the CPU dense-membership allocation and expose the existing counts-only path.
- [ ] Put the two equality tolerance metrics beside the `equals_exact` comparison.

## Larger proposals

The **ten changes I would prioritize for a GeoPandas user**, in order, are: **(1)** separate membership counts from assignment counts, **(2)** carry live feature count from overlay into joins, **(3)** publish consistent label coordinates and clearance, **(4)** correct grid rounding, **(5)** return source-coordinate envelopes, **(6)** emit inner and unmatched rows in one spatial pass, **(7)** accept canonical empty polygon hierarchies, **(8)** make generated shapes directly consumable as polygons, **(9)** make CPU point assignment sparse, and **(10)** standardize compact count access.

I recommend keeping the low-level graph contributors and adding the counted operand and named result ports before the alpha. A general `sjoin` convenience wrapper would still have to decide all three underlying questions: which features are live, whether counts describe assignments or memberships, and how to represent an unmatched row after candidate loss. Fixing those contracts benefits direct users, recipes, and a future wrapper.

## Not read

I read the label-point CPU search, shared CPU polygon-row helpers, staged exact CPU relate implementation, staged PIP classification and reduction paths, and relevant parameter packers in full. I traced the allocation, validation, and publication paths used in the findings. I sampled the public props and docs for the other scoped families, including neighbor search, segment intersection, line merge and locate, rectangle clip, areal interpolation, zonal statistics, and lon/lat preparation. I did not audit every shader, every lens, complete overlay noding, spherical topology, or every output of the 70 direct GPU classes.

All observed numerical library results above came from host code or CPU twins. GPU predictions are marked **SOURCE** with confirming probes. No GPU, browser, build, test runner, network, or file writes were used.