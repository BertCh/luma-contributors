# Math correctness review: polygon topology (2026-11-10)

Scope id `polygon-topology`. Paths are relative to `modules/analysis/src`. Directories under
`gpu-spatial-analysis/`: `polygon-overlay`, `geometry-topology`, `delaunay-tessellation`,
`polygon-triangulation`, `segment-intersection`, `geometry-predicates`, `geometry-validity`,
`polygon-coverage-simplification`, `polygon-coverage-topology`, `rectangle-clip`, `vertex-snap`,
`polygon-buffer`, `offset-expansion`, `geometry-edit`.

Nothing was run on a GPU. No spec file was run (0 of the 5 allowed). Confirmations are hand
derivations or Node scripts that replicate the kernel arithmetic with `Math.fround`. The scripts are
in the session scratchpad, folder `polygon-topology/` (`checks1.mjs` to `checks6.mjs`). Shapely is
not installed on this machine, so no reference library was executed; GEOS and JTS behaviour is
taken from their source as I remember it and from the Shapely fixtures already in the test tree.

## 1. Coverage

**Read in full**

- `analysis-core/kernels/exact-orientation-wgsl.ts`, `exact-lattice-wgsl.ts`, `exact-geometry-wgsl.ts`
  (owned by another reviewer; read to check the calls).
- `segment-intersection/segment-intersection-wgsl.ts`, `gpu-segment-intersection.ts`, `segment-table.ts`.
- `geometry-topology/gpu-snap-rounded-noding.ts`, `gpu-make-valid.ts`, `topology-types.ts`.
- `polygon-overlay/overlay-face-labelling-wgsl.ts`.
- `delaunay-tessellation/gpu-delaunay-tessellation.ts`.
- `polygon-triangulation/gpu-constrained-delaunay-triangulation.ts` and the four re-export files.
- `polygon-coverage-topology/coverage-topology-kernels.ts`.
- `rectangle-clip/rectangle-clip-kernels.ts`.
- `offset-expansion/gpu-offset-expansion.ts`.
- `geometry-edit/geometry-edit-kernels.ts`.

**Read in part (all WGSL and parameter packers read; constructor validation and binding plumbing skimmed)**

- `polygon-overlay/gpu-polygon-overlay.ts` (60-1433 of 1,432).
- `geometry-topology/gpu-polygonize.ts` (40-617), `gpu-line-noding.ts` (60-548).
- `delaunay-tessellation/gpu-voronoi-diagram.ts` (30-514).
- `segment-intersection/segment-bvh.ts` (30-169).
- `geometry-predicates/gpu-geometry-predicates.ts` (40-863), `geometry-predicates-parameters.ts` (packer).
- `geometry-validity/gpu-geometry-validity.ts` (40-876), `geometry-validity-types.ts` (all).
- `polygon-coverage-topology/gpu-coverage-validity.ts` (30-561), `gpu-coverage-dissolve.ts` (100-572).
- `polygon-coverage-simplification/gpu-coverage-simplification.ts` (150-250, 356-1610).
- `rectangle-clip/gpu-rectangle-clip.ts` (30-180).
- `vertex-snap/gpu-vertex-snap.ts` (30-390).
- `polygon-buffer/gpu-buffer-surface.ts` (56-847).
- `geometry-edit/gpu-affine-transform.ts` (30-220), `gpu-geometry-cleanup.ts` (30-150),
  `gpu-geometry-orientation.ts` (30-110).
- Outside the scope, read because the scope depends on it: `outline-geometry/gpu-offset-curve.ts`
  (344-520, the join kernel), `analysis-core/polygon-triangulation/gpu-polygon-triangulation.ts`
  (140-160, 548-720: winding, linked list, ear test).

**Not opened**

- `segment-intersection/segment-intersection-types.ts`, `geometry-predicates/geometry-predicates-access.ts`,
  `polygon-coverage-topology/coverage-validity-parameters.ts`, every `index.ts`.
- `analysis-core/kernels/polygon-edge-slabs.ts`. No file in this scope imports it.
- `analysis-core/ring-assembly/*` (ring chaining, hole assignment), `gpu-spatial-analysis/line-split/*`,
  `line-simplification/*`, the rest of `analysis-core/polygon-triangulation/*` (the ear-clipping loop,
  hole bridging and the whole monotone path, about 3,300 lines). Overlay, polygonize, make-valid,
  coverage dissolve and the constrained Delaunay class depend on them. They are not verified here.

## 2. Findings

| ID | Severity | Status | File:line | What the kernel computes | What the reference says | Concrete input and wrong output | Suggested fix |
| --- | --- | --- | --- | --- | --- | --- | --- |
| polygon-topology-01 | wrong-edge-case | CONFIRMED (Node replica of the kernel) | `geometry-topology/gpu-make-valid.ts:346`; cause in `analysis-core/kernels/exact-geometry-wgsl.ts:124-130` | Face parity from a ray at the height of the f32-rounded midpoint of the face's first shell edge. When that rounded height equals a vertex height, the tie rule for horizontal edges is applied. | The probe is "strictly inside the face by construction" (class doc `:91-93`). | Ring `(0, 1+ulp), (0, 1), (5, 1), (5, 7), (0, 7)`, `ulp = 2^-23`, valid, area 30. Parity 0, face dropped, zero polygons out. | Decide `exactProbeAbove` for a non-horizontal side probe from the exact endpoints, not from the rounded midpoint. |
| polygon-topology-02 | wrong-edge-case | CONFIRMED (hand derivation) | `polygon-coverage-simplification/gpu-coverage-simplification.ts:1519-1576` | Topology repair only looks for simplified segments that cross, touch or overlap. | Shapely `coverage_simplify` keeps a valid coverage; a ring may not jump over another ring. | Arc `(0,0),(0,4),(3,5),(0,6),(0,10)` between A and B, an island C inside the bulge, tolerance 3.5. The arc becomes `x = 0`. C ends up inside B, and A keeps a hole outside its shell. No crossing is found. | After simplification, test one vertex of every ring that has no junction against the swept area of each dropped span, or against its old and new containing ring; restore the farthest vertex when the side changed. |
| polygon-topology-03 | wrong-edge-case | CONFIRMED (Node script) | `polygon-coverage-topology/gpu-coverage-dissolve.ts:308-314`; `polygon-coverage-simplification/gpu-coverage-simplification.ts:408-414` | Vertex key `floor(v / snapTolerance + 0.5)` on the absolute coordinate, clamped to ±2147483520. | `spatial-weights/gpu-contiguity-weights.ts:96` documents "`extent / snapTolerance` must stay below 2^24". | `y = 3,000,000` and `y = 3,000,100`, `snapTolerance = 0.001`, extent 100 (ratio 1e5, allowed by the doc). Both keys clamp to the same value, so two vertices 100 m apart on one vertical edge are one point and the edge is dropped. | Quantise relative to the data minimum (as snap rounding does), or flag `abs(v) / snapTolerance >= 2^24` in status. Correct the doc: the limit is on the coordinate magnitude. |
| polygon-topology-04 | inconsistent | CONFIRMED (hand derivation) | `geometry-topology/gpu-make-valid.ts:250-255` | Passes `polygons.ringOffsets` to the noder as open linestrings. | Repository convention: rings close implicitly (`geometry-validity/geometry-validity-types.ts:74`). Overlay adds the closing vertex itself (`polygon-overlay/gpu-polygon-overlay.ts:413-415`). | Ring `(0,0),(4,0),(4,4),(0,4)` with no repeated vertex. Three segments are noded, the closing edge is missing, no face is found, `count = 0`. | Add one closing vertex per ring as the overlay does, or document and validate explicit closure. |
| polygon-topology-05 | wrong-edge-case | SUSPECTED (kernel side confirmed by script; GEOS side from source, not run) | `polygon-coverage-topology/gpu-coverage-validity.ts:404-418`, doc `:160-161` | "Gap" when both ends of the shorter segment are within `gapWidth` of the longer segment. | GEOS `InvalidSegmentDetector::isNearlyParallel`: project each segment on the other; both projections longer than the tolerance and their matching ends within the tolerance. | `(0,0)-(10,0)` and `(5,0.0625)-(15,0.0625)`, `gapWidth = 0.125`. Kernel: no flag on either. GEOS rule: both flagged. | Port the GEOS rule. Remove the sentence that the rule is not documented. Add a staggered scene to the Shapely fixture. |
| polygon-topology-06 | precision | CONFIRMED (Node script) | `geometry-topology/gpu-snap-rounded-noding.ts:366`, `:1010`; `polygon-overlay/gpu-polygon-overlay.ts:1176-1179`, `:1300-1301` | Lattice step is `max(gridCell, extent * 2^-22)`. World output is `origin + k * cell` rounded to f32. | "The result lies on that lattice: its vertices are lattice points" (`gpu-polygon-overlay.ts:212`). | Coordinates near `2^23` (ulp 1), extent 5, tolerance `2^-10`. Crossings at `x = C+1.5` and `C+2.5` are distinct lattice points and both become `C+2`. The output triangle has two equal vertices and zero area. | Also bound the step below by the ulp of the largest coordinate magnitude, so every lattice point is an f32 value. Or state the limit and set `approximation`. |
| polygon-topology-07 | precision | CONFIRMED (Node script) | `rectangle-clip/rectangle-clip-kernels.ts:341-349` | Clip point of edge `s -> e` computed from `s`. | Two polygons that share an edge should get the same clip point. | Edge between `(0.204, 0.165)` and `(15.43, 6.349)`, bound `x = 10`. One direction gives `y = 4.1434054`, the other `4.1434059`. 27,200 of 100,000 random edges differ. | Compute from the lexicographically smaller endpoint, whatever the traversal direction. |
| polygon-topology-08 | wrong-edge-case | CONFIRMED (Node script); known as C9 of the 2026-11-09 review, still present | `polygon-coverage-topology/coverage-topology-kernels.ts:61-69` | Ring orientation from an f32 shoelace sum. | Exact orientation, as `geometry-edit/geometry-edit-kernels.ts:107-150`. | Shell `(0,0), (16777216,16777215), (16777215,16777214)`. Exact doubled area is -1 (clockwise). f32 gives 0, so `flip = 0`. Matched edges of a neighbour are then reported as `overlap`, and dissolve orients the boundary the wrong way. | Use the exact extreme-vertex node of `geometry-edit`. |
| polygon-topology-09 | inconsistent | CONFIRMED (code read; no wrong output shown) | `polygon-buffer/gpu-buffer-surface.ts:386-400` | Picks the interior side of a narrow ring from an f32 shoelace sum. | The rings were normalized two nodes earlier by the exact `GPUGeometryOrientation` (`:345-355`): shells are counter-clockwise, holes clockwise. | A narrow ring whose f32 area rounds to 0 is read as clockwise and the outer offset ring is collapsed instead of the inner one. | Use the ring role (shell or hole) instead of a second area sum. |
| polygon-topology-10 | doc-mismatch | CONFIRMED (Node script) | `rectangle-clip/rectangle-clip-kernels.ts:84-93`; doc `rectangle-clip/gpu-rectangle-clip.ts:162-163` | Line clip point is `a + (b - a) * t`. | "intersections lie on the rectangle edges exactly in the clipped coordinate". True for polygons only. | `a.x = 0.9345882535`, `b.x = 17.83302879`, `minX = 10`. Start `x = 10.00000095`. 1,142 of 100,000 random segments miss the edge by one ulp. | Write the bound itself into the limiting axis, or correct the sentence. |
| polygon-topology-11 | doc-mismatch | CONFIRMED (hand derivation) | `delaunay-tessellation/gpu-voronoi-diagram.ts:66-80` | Voronoi edges only from triangles. | Two sites have one bisector; collinear sites have parallel bisectors (`shapely.voronoi_polygons` returns them). | Two sites, or any collinear set: Delaunay gives 0 triangles, so 0 segments, `invalidCount = 0`. | Document it, or emit the bisectors of consecutive collinear sites. |
| polygon-topology-12 | doc-mismatch | CONFIRMED (code read) | `geometry-validity/gpu-geometry-validity.ts:521-526`, `:596-598`; doc `geometry-validity-types.ts:14` | `shortRing` when the vertex count without a closing duplicate is below 3. | Doc says "fewer than three distinct vertices". GEOS counts non-repeated points. | Ring `A, B, B, A`: count 3, `shortRing` not set. The ring is still invalid through `selfIntersection`. | Count non-repeated vertices, or reword the bit. |

### polygon-topology-01. The side probe breaks on an edge one f32 step tall

`exactProbeAbove` compares a vertex height with `probe.point.y`. For a side probe, `probe.point`
is `0.5 * (a + b)` rounded to f32. If `a.y` and `b.y` are adjacent f32 values the sum is a tie and
the midpoint rounds to one of them. The kernel then takes the branch written for a horizontal edge
(`probe.side * sign(dx) <= 0`). For a vertical edge that always answers "above", so the ray runs
just below the lower endpoint, outside the edge's own height range. It measures the parity of a
point beyond the end of the edge, not beside its middle.

Replica results (`checks3.mjs`, `checks5.mjs`):

- Rectangle `(0, 1+ulp), (0, 1), (5, 1), (5, 7), (0, 7)`: parity 0. Dropped.
- The same rectangle with a first edge 2 ulp tall: parity 1. Kept.
- The same rectangle started one vertex later: parity 1. Kept.
- Sliver triangle `(10, 1), (0, 1+ulp), (0, 1)`: parity 0 for two of its three possible first edges.

Fat faces whose first edge is not vertical are safe: the tie rule moves the ray to the left of the
edge, which is into the face. The failure needs a first shell edge that is one f32 step tall and
vertical (or a face one step thick). Near-duplicate vertices of f64 data become 0 or 1 ulp apart
in f32, so such edges exist in real data. With `snapRounding` the same thing happens when the
lattice step equals the f32 step of the world coordinates (coordinates in `[2^21, 2^22]`, extent
`2^20`), because the parity kernel reads world endpoints, not lattice endpoints.

The overlay is not affected: it probes on the integer lattice, where the midpoint of two integers
below `2^23` is exact (`polygon-overlay/overlay-face-labelling-wgsl.ts:219`).

The same probe is built at `analysis-core/ring-assembly/gpu-segment-ring-assembly.ts:1473` for
hole-to-shell assignment, on whatever coordinates the caller passes. That file is outside this
scope. It needs the same check.

Fix: for a side probe with `start.y != end.y`, a vertex with `v.y <= min(start.y, end.y)` is below
and one with `v.y >= max(start.y, end.y)` is above. A vertex strictly between is compared with the
exact midpoint (`v.y - low > high - v.y`). The rounded midpoint is then never used for the height.

### polygon-topology-02. Simplification can move an island to the other side of an arc

Junctions are found from a change of the edge partner (`:589`). An island and the hole it fills have
no junction, so they are tied to the arc network by nothing. A Douglas-Peucker shortcut that sweeps
across the island does not intersect it, and the repair rounds only react to intersections
(`:1566-1568`). In the example the result has C overlapping B and a hole of A outside A. The class
doc and `topologyIterationCount` say the rounds "make the simplified coverage topology-preserving".
GEOS avoids this in its coverage simplifier by refusing to remove a vertex when the removed corner
contains another vertex.

Also note for users, documented in the class: the algorithm is Douglas-Peucker per arc. Shapely
`coverage_simplify` is Visvalingam-Whyatt with a tolerance that is the square root of an area. The
same tolerance does not give the same result in the two libraries. One spec compares kept vertex
sets with Shapely on one scene at tolerance 0.5; that agreement is a property of that scene.

### polygon-topology-03. The snap key is not relative to the data

The key is formed from `v / snapTolerance` in f32. Three regimes (`checks1.mjs`):

- Ratio below `2^24`: correct.
- Ratio between `2^24` and `2^31`: the quotient has fewer bits than the coordinate. At `x` near
  500,000 with tolerance 0.001, 468 of 20,000 adjacent f32 values (0.031 apart) get the same key,
  so points 31 mm apart merge under a 1 mm tolerance.
- Ratio above `2^31`: every coordinate clamps to one key. This is the case in the table.

The contiguity-weights doc gives the limit as `extent / snapTolerance`. The code depends on the
coordinate magnitude. A UTM dataset with a small extent satisfies the doc and still fails. The same
expression is in `spatial-weights/gpu-contiguity-weights.ts` (another scope).

### polygon-topology-05. The gap rule differs from GEOS on staggered segments

The class doc says the GEOS rule "is not documented" and that the kernel rule was fitted to
`coverage_invalid_edges`. The rule is in GEOS (`coverage/InvalidSegmentDetector`, `isNearlyParallel`)
and in JTS. My transcription of it reproduces all five gap scenes of
`test/gpu-spatial-analysis/polygon-coverage-topology/coverage-topology-scenes.ts` (`checks2.mjs`),
which supports the transcription. The two rules agree when the shorter segment projects inside the
longer one. They differ when the two segments overlap only partly along their length, which is
the usual shape of a real gap between two independently digitised boundaries: the kernel reports
nothing. No fixture scene has that shape. This needs one Shapely run to become CONFIRMED:
`shapely.coverage_invalid_edges` on the rectangles `[0,10] x [-5,0]` and `[5,15] x [0.0625,5]` with
`gap_width=0.125`.

### polygon-topology-06. Lattice points that are not f32 values

The lattice is sized from the extent, not from the coordinate magnitude. When coordinates are more
than about four extents from zero, the lattice step is smaller than the f32 step of the world
coordinates. Topology is still decided exactly on the lattice. The world vertices are then rounded
again, by round-to-nearest-even, and can coincide or cross by one f32 step. Example regime: UTM
northing 4,000,000 m, extent 20 km. The lattice step is at most 8 mm; the f32 step is 0.25 to 0.5 m.

This is separate from the known coarsening at `2^-22` of the extent (Q14 of the 2026-11-09 review).

### polygon-topology-07. Clip points of a shared edge

A coverage clipped by `GPURectangleClip` no longer has identical vertices on shared edges at the
clip boundary. `GPUCoverageDissolve` and `GPUCoverageValidity` match vertices by exact f32 equality
when `snapTolerance` is 0, so the clipped coverage shows unmatched edges there.

## 3. Verified correct

- `GPUSegmentIntersection`: classification (proper, touch, collinear touch, overlap), adjacency skip,
  successor of a repeated vertex, cyclic rings, closed linestrings, pair expansion by binary search.
  Checked by derivation against the standard four-orientation test. Uses the shared exact
  `orientSign`.
- `GPUSnapRoundedNoding`: lattice frame, vertex rounding (half up, exact through the fraction),
  exact rounded crossing (`latticeRoundedCrossing` inequality and bit widths re-derived), hot pixel
  set, candidate search by pixel diagonals (complete for segments with lattice endpoints), half-open
  pixel test, ordering of pixels along a segment. Compared with Hobby / Guibas-Marimont and GEOS
  `HotPixel`. GEOS rounds half up (`java_math_round`), so a power-of-two grid matches.
- `GPULineNoding`: thin layer over `GPULineSplit` (not read); provenance by nearest source edge.
- `GPUPolygonize`: directed edge pairs and groups are right. Faces come from ring assembly (not read).
- `GPUMakeValid`: even-odd parity over the noded segments of the group is the right rule for the
  GEOS linework method. Counting coincident segments once per source ring is right. See 01 and 04.
- `GPUPolygonOverlay`: canonical edges, level 1 winding per polygon, level 2 coverage per operand,
  the ray shoot, `compareCrossings` (numerator, 96-bit product, slope tie-break), pointer jumping,
  fill rules, the four Boolean operations and the orientation of kept edges. Re-derived by hand.
- `ADAPTIVE_INCIRCLE_WGSL`: filter bound `1e-6` against Shewchuk's `(10 + 96e)e = 5.96e-7` for f32;
  permanent and determinant match Shewchuk; limb widths of the exact path (5, 11, 22 limbs) hold
  for a 55-bit span; row signs match the cofactor expansion.
- `GPUDelaunayTessellation`: symbolic super vertices (all three orientation rules and the in-circle
  rule re-derived from `d_0 = (1,-e)`, `d_1 = (e,1)`, `d_2 = (-1,-e)`), Simulation of Simplicity
  cofactor sign, the four argument permutations of `isFlipRequired` and their parities, 1-to-3 and
  2-to-4 splits, the flip, adjacency repair, relocation, independence of parallel flips, the
  convergence test and `2n - 2 - h`. First-row duplicate ownership treats `-0` as `0`.
- `GPUVoronoiDiagram`: circumcentre formula relative to a vertex (f32 error 0.008 at radius 20,000
  with UTM coordinates, `checks6.mjs`), outward hull normal, Liang-Barsky clip, each internal edge
  emitted once.
- `GPUConstrainedDelaunayTriangulation`: strict convexity test, in-circle test on a counter-clockwise
  triangle (the triangulator emits counter-clockwise triangles on both paths), flip result, bounded
  loop and `converged`. Lawson flips of unconstrained edges give the constrained Delaunay result.
- `GPUGeometryPredicates`: `isCcw` matches JTS `Orientation.isCCW` line by line; `isSimple`, `isRing`,
  `isClosed`, `equalsExact` (Euclidean distance, as GEOS), `equalsIdentical`, counts.
- `GPUGeometryValidity`: per-ring checks, exact winding at the lexicographic minimum, exact
  point-in-shell parity with boundary events, fold of intersection kinds. The unchecked cases are
  listed in the class doc. See 12.
- `GPUCoverageValidity`: matched edge and same-side test, crossing, T-junction, winding of the
  midpoint. See 05 and 08.
- `GPUCoverageDissolve`: internal edge rule and interior-left orientation. See 03 and 08.
- `GPUCoverageSimplification`: partner, arc, owner and shared keep decision; ring minimum of three;
  span owner and split choice. Junction detection by partner change is sound for a valid coverage.
  See 02 and 03.
- `GPURectangleClip`: Sutherland-Hodgman stage (inclusive inside test, no division by zero),
  Liang-Barsky signs and rejects, continuation rule. See 07 and 10.
- `GPUVertexSnap`: strict `<` tolerance, nearest reference, lowest row on ties, as GEOS
  `LineStringSnapper::findSnapForVertex`. The BVH bound is a true lower bound.
- `GPUBufferSurface`: winding of the reversed polygon and the two stroke rings for both signs of
  the distance, the `'negative'` fill rule, collapsed interior offset (the envelope test is exact,
  not a heuristic, for the question it asks), round cap step `pi / (2 * quadSegments)`, square and
  flat caps. See 09.
- `GPUOffsetCurve` join kernel (outside scope, read for the buffer): inside-turn intersection,
  GEOS fillet step count `round(angle / (pi / 2 / quadSegments))`, mitre ratio `1 / cos(half)`,
  limited mitre as in GEOS 3.12 (bevel line at `mitreLimit * d`, half length
  `d (1 - mitreLimit cos(half)) / sin(half)`).
- `GPUOffsetExpansion`: upper-bound search and local index.
- `GPUGeometryOrientation`: exact extreme-vertex winding, shell and hole roles, reversal.
- `GPUAffineTransform`: `R * K * S` with `K = [1 tan(xs); tan(ys) 1]`, as `shapely.affinity`.
- `GPUGeometryCleanup`: half-up grid snap (as GEOS), repeated-point rule and the last-vertex rule;
  the tolerance-0 parallel path is equivalent to the sequential one.

## 4. Standards consistency

**Ring orientation, six implementations.**

| Where | Method | Exact |
| --- | --- | --- |
| `geometry-edit/geometry-edit-kernels.ts:107-150` | lexicographic minimum vertex, `orientSign` | yes |
| `geometry-validity/gpu-geometry-validity.ts:559-587` | same, private copy | yes |
| `analysis-core/polygon-triangulation/gpu-polygon-triangulation.ts:610-634` | same, private copy | yes |
| `geometry-predicates/gpu-geometry-predicates.ts:832-863` | GEOS highest point (to match `is_ccw`) | yes |
| `polygon-coverage-topology/coverage-topology-kernels.ts:61-69` | f32 shoelace | no (08) |
| `polygon-buffer/gpu-buffer-surface.ts:386-400` | f32 shoelace | no (09) |

The three exact copies agree. One shared node in `analysis-core/kernels` would remove the two f32 ones.

**Point in ring, three implementations.** All use exact signs and give the same parity.

- Shared `exactProbeCrossing` (`exact-geometry-wgsl.ts:135`): a vertex at the ray height counts as above.
- `geometry-validity/gpu-geometry-validity.ts:628-669`: private, half-open `a.y > p.y`.
- `polygon-coverage-topology/gpu-coverage-validity.ts:446-463`: private, half-open `start.y <= p.y`.

**Grid rounding, three implementations.** All round half up. The numerics differ.

- `gpu-snap-rounded-noding.ts:398-403`: relative to a data origin, exact.
- `geometry-edit-kernels.ts:395-400`: `floor(v / g + 0.5) * g` on the absolute coordinate.
- `gpu-coverage-dissolve.ts:308-314` and `gpu-coverage-simplification.ts:408-414`: the same on the
  absolute coordinate, clamped (03).

**Ring closure.** Implicit closure is the stated convention. Overlay, buffer, validity, predicates
and rectangle clip accept it. `GPUMakeValid` and `GPUPolygonize` (given rings as lines) need the
repeated vertex (04). `GPUGeometryCleanup` documents the GeoArrow repeated vertex.

**Largest float literal.** `3.402823466e+38` is written again in `segment-table.ts:134`,
`segment-bvh.ts:103`, `gpu-coverage-validity.ts:373`, while `ORIENT_FLOAT_MAXIMUM` exists. `3.0e38`
(`overlay-face-labelling-wgsl.ts:189`, `gpu-polygon-overlay.ts:927`) and `3.4e38`
(`gpu-geometry-validity.ts:359`, `gpu-line-noding.ts:415`) are also used as "large".

**Offset search.** "Last row whose offset is at most `i`" is rewritten in most files of the scope
(`segment-table.ts:60`, `gpu-make-valid.ts:179`, `gpu-polygon-overlay.ts:471`,
`gpu-coverage-dissolve.ts:315`, `gpu-buffer-surface.ts:302`, `geometry-edit-kernels.ts:35`,
`gpu-offset-expansion.ts:199`). All have the same meaning.

**BVH walk.** The stackless walk is copied in `gpu-segment-intersection.ts:414-436`,
`gpu-coverage-validity.ts:325-347` and `overlay-face-labelling-wgsl.ts:228-254`.

## 5. luma.gl official implementations

- Published `@luma.gl/graph/gpu-geometry` exports only mesh normals and mesh smoothing. There is no
  official noding, overlay, Delaunay, triangulation, validity, clip or buffer to delegate to.
- The scope builds on public `GPUBVH`, `GPUScan`, `GPUSort` and `GPUCompaction` from
  `@luma.gl/graph/gpu-core`. I found no argument that changes their meaning. `GPUSort` gets
  `keyBits: 23` for lattice keys whose sentinel is `2^23 - 1`; that fits.
- Local kernels walk the `GPUBVH` arrays with implicit indices (`2n + 1`, `2n + 2`). The official
  `GPUBVHQuery` follows `nodeChildren`. The official builder writes exactly those child indices and
  its type is named a complete-binary hierarchy, so the two agree. `GPUBVHQuery` answers one query
  per graph, so it cannot replace the per-row probes.
- `@luma.gl/graph/geospatial` publishes `GPUPairwisePointInPolygon`. It uses double-single
  arithmetic with an error bound and returns `uncertain` near a tie. The local tests use an exact
  integer fallback and never need `uncertain` for finite input. Both use the same half-open crossing
  rule (`start.y <= p.y < end.y`) and both report a boundary class. The hole-in-shell test of
  `GPUGeometryValidity` is a pairwise point-in-polygon and could delegate, at the cost of an
  `uncertain` answer where the local one is exact. Better: propose the exact orientation predicate
  upstream.
- `GPUPairwisePointSegmentDistance` exists upstream. `gpu-coverage-validity.ts:385-390` has a
  private f32 point-to-segment distance inside a per-pair kernel; it cannot delegate without a
  pair list, and the formula is the standard clamped projection.

## 6. Test oracles

**Independent oracle (reference library output frozen in the test tree)**

- `GPUSegmentIntersection`: Shapely (`shapely-segment-fixture.ts`).
- `GPUSnapRoundedNoding`, `GPUPolygonOverlay`: exact rational reference plus Shapely
  (`fixtures/overlay-snap-rounding.json`, `parity-wp4-overlay-*.json`).
- `GPULineNoding`, `GPUMakeValid`: Shapely (`fixtures/line-noding.json`, `fixtures/make-valid.json`).
  `gpu-make-valid-parity.node.spec.ts` is a CPU copy of the parity formula with integer inputs; it
  cannot see finding 01.
- `GPUDelaunayTessellation`: scipy (`delaunay-reference.ts`, `fixtures/delaunay-properties.json`),
  plus the empty-circle property.
- `GPUGeometryPredicates`, `GPUGeometryValidity`: Shapely fixtures.
- `GPUCoverageValidity`, `GPUCoverageDissolve`: Shapely and GeoPandas (`coverage-topology-scenes.ts`).
  No staggered gap scene (05).
- `GPURectangleClip`: Shapely lengths and areas, and an f64 copy.
- `GPUVertexSnap`: Shapely (`outline-geometry/gpu-vertex-snap.spec.ts`).
- `GPUBufferSurface`: Shapely (`fixtures/buffer-properties.json`, mitre-limit and collapsed-hole fixtures).
- `GPUGeometryCleanup`, `GPUGeometryOrientation`, `GPUAffineTransform`: Shapely (`geometry-edit-fixtures.ts`).

**Copy of its own formula, or hand values only**

- `GPUCoverageSimplification`: `coverage-simplification-oracle.ts` is a CPU model of the same arc
  rule. One scene is compared with Shapely. No scene has an island next to a simplified arc (02).
- `GPUConstrainedDelaunayTriangulation`: two small cases with expected indices in
  `test/geospatial-expansion/raster-terrain-expansion.spec.ts`. No reference library.
- `GPUVoronoiDiagram`: hand-computed segments (`voronoi-clip.spec.ts`,
  `gpu-delaunay-tessellation.spec.ts`).
- `GPUOffsetExpansion`: hand values. The formula is an index map.

**Nothing found**

- Snap keys with a large coordinate-to-tolerance ratio (03).
- Overlay on coordinates whose f32 step exceeds the lattice step (06).

## Resolution (2026-11-10)

Paths are relative to `modules/analysis`. Every spec named below was run on the headless GPU (or
in Node for `*.node.spec.ts`) and passes. Each new GPU spec was run against the old kernel first
and failed there, except where the row says otherwise. Counts: 11 fixed (3 of them documentation
only), 1 fixed by rewording with the kernel unchanged (05), 0 not reproduced, 0 deferred; 4 extra
findings fixed.

| ID | Outcome | What changed | Spec that pins it | Behaviour change a caller sees |
| --- | --- | --- | --- | --- |
| polygon-topology-01 | fixed | `src/analysis-core/kernels/exact-geometry-wgsl.ts:136-153` (new `exactCompareToMidpoint`), `:156-163` (`exactProbeAbove`). The height of a side probe is the exact midpoint of its segment's two heights, never the rounded `probe.point.y`. Outside the two heights the answer is a comparison; between them `2 * value` is compared with the float32 sum, which is right whenever they differ (rounding is monotonic), and when they are equal the sign is that of the sum's rounding error, read off integer significands (`ADDITION_ROUNDING_ERROR_WGSL`, embedded under the name `exactAdditionRoundingError`; a float two-sum would be folded away). Point probes, the signature and the meaning of every shared function are unchanged. The same probe at `src/analysis-core/ring-assembly/gpu-segment-ring-assembly.ts:1505` (hole-to-shell assignment) runs on the caller's world coordinates, hit the same case, and is cured by the same change. CPU twin updated: `test/gpu-spatial-analysis/segment-intersection/exact-geometry-reference.ts` (`isAboveProbe` now compares with the exact midpoint in BigInt). | `test/gpu-spatial-analysis/geometry-topology/gpu-make-valid-float32-edges.spec.ts` (the report's ring, and the same shape at northing 4,000,000); `test/gpu-spatial-analysis/ring-assembly/gpu-segment-ring-assembly-hole-probe.spec.ts` (a hole whose one-step first edge starts at a notch tip of its shell: `ringShells` was `NONE`, now the shell; checked against the old rule by putting it back for one run); `test/gpu-spatial-analysis/segment-intersection/exact-geometry-helpers.spec.ts` (4,096 side probes with heights whose midpoint float32 cannot hold, against the integer reference). | `GPUMakeValid`: a valid ring whose first edge is one float32 step tall returns 1 polygon (before: 0). `GPUSegmentRingAssembly` and its users: a hole whose first edge is one step tall and touches its shell is assigned to that shell (before: unassigned and left out of `polygons`). |
| polygon-topology-02 | fixed | `src/gpu-spatial-analysis/polygon-coverage-simplification/gpu-coverage-simplification.ts:923-948` (one BVH over the input segments, built once), `:1476-1509` (`span-links`: per vertex the span that owns its input edge, the span end and the span's reach), `:1721-1843` (`sweep` and `sweep-count` kernels in every topology round). Every kept vertex walks the BVH along its horizontal line; for each input edge that belongs to a span with dropped vertices, and whose chord is within the span's reach of the vertex, the original chain plus the shortcut is tested as a ring with the exact point probe. A span that holds a kept vertex is repaired like a crossing (its farthest dropped vertex is restored). The test is exact, not conservative. Cost: one BVH build; per round, kept vertices x input edges on the vertex's horizontal line x span length in the worst case, with a four-multiplication reject for a vertex farther from the chord than the span's farthest dropped vertex. | `test/gpu-spatial-analysis/polygon-coverage-simplification/gpu-coverage-simplification.spec.ts` ("does not sweep an arc over an island": the report's scene; the island stays inside A and outside B, the arc keeps `(3, 5)` and still drops `(0, 4)` and `(0, 6)`; without the island the arc is straightened as before). | The report's scene: A's shell is `(-10,0), (0,0), (3,5), (0,10), (-10,10)` (before: the arc was `x = 0` and the island lay in B). `output.topologyStatistics[0]` and `[1]` now count swept spans as well as crossing pairs: the wavy-strip scene reports `[17, 0, 18, 0]` (before `[12, 0, 18, 0]`; same restored and kept counts). |
| polygon-topology-03 | fixed | `src/gpu-spatial-analysis/polygon-coverage-topology/coverage-topology-kernels.ts:40-63` (new `getCoverageSnapKeyWGSL`: key `floor((value - origin) / snapTolerance + 0.5)` with the first vertex of `positions` as origin; from `2^23` steps on the quotient is used as it is, because adding 0.5 there rounds to even), `:75-125` (`createCoverageSnapRangeNodes`: GPU count of the vertices `2^24` or more steps from the origin). Used by `gpu-coverage-dissolve.ts:332`, `:589-601` and `gpu-coverage-simplification.ts:451`, `:1043-1055`. The count was first ORed into `output.status.overflow`; after the fix review it is the optional `output.status.uncertainCount` (see the follow-up below). Documented limit corrected in both classes and in `docs/api/gpu-spatial-analysis/operators-lines.md`: every vertex within `2^24 * snapTolerance` of the first vertex on both axes. | `test/gpu-spatial-analysis/polygon-coverage-topology/gpu-coverage-topology-float32.spec.ts` (two 100 m squares at northing 3,000,000 with a 1 mm tolerance dissolve to one 200 x 100 ring; an extent of 20,000 m gives `uncertainCount` 2, 16,000 m gives 0, tolerance 0 always 0); `gpu-coverage-simplification.spec.ts` ("snap keys follow the extent at projected coordinates"). | Dissolve and simplification with `snapTolerance > 0` far from zero: correct point identities (before: vertices 100 m apart were one point at `y = 3,000,000`). The snap grid is anchored at the first vertex, not at zero, so which near-boundary vertices merge can differ by one cell. The optional `status.uncertainCount` counts the vertices outside the exact key range. |
| polygon-topology-04 | fixed | `src/gpu-spatial-analysis/geometry-topology/gpu-make-valid.ts:219-290` (one repeated first vertex per ring before noding, as `GPUPolygonOverlay`), `:328-354` (`sourceEdgeIds` mapped back to rows of `polygons.positions`). Convention: the repository's implicit closure. | `test/gpu-spatial-analysis/geometry-topology/gpu-make-valid-float32-edges.spec.ts` ("closes a ring given without its repeated first vertex": square, square with hole, and the closed form). | `GPUMakeValid` on rings without a repeated first vertex: the polygon (before: `count` 0). Explicitly closed rings give the same result as before. `nodedSegments.sourceEdgeIds` still index `polygons.positions`; the closing edge of an implicitly closed ring reports the ring's last row. |
| polygon-topology-05 | fixed (documentation; kernel unchanged) | The staggered pair is not a gap by the class's own definition ("both endpoints of the shorter lie within `gapWidth` of the longer": the end `(15, 0.0625)` is 5 units away), so the rule stays. `src/gpu-spatial-analysis/polygon-coverage-topology/gpu-coverage-validity.ts:164-178` now says the rule is this contributor's own, names the GEOS rule (`InvalidSegmentDetector::isNearlyParallel`), describes the staggered case that the two treat differently and says that the difference has not been checked against a GEOS run. The `gap` flag documentation (`:60-66`) and `docs/api/gpu-spatial-analysis/operators-validity.md` say the same. | `gpu-coverage-topology-float32.spec.ts` ("gap rule needs the shorter segment wholly beside the longer": a fully overlapped pair is flagged, the staggered pair is not). It pins what the kernel computes; it passed before. | None. Still open: one Shapely run of the staggered rectangles to learn what GEOS returns. |
| polygon-topology-06 | fixed | `src/gpu-spatial-analysis/geometry-topology/gpu-snap-rounded-noding.ts:372-386`: the lattice step is also at least `2^-24` of the largest coordinate magnitude (one float32 step there), so every lattice point of the data is a float32 value. `output.lattice` already reports the step in effect. Documentation: `gridSize` and `lattice` in that file, `vertexTolerance` in `polygon-overlay/gpu-polygon-overlay.ts:208-220`, `docs/api/gpu-spatial-analysis/operators-geometry.md`. CPU twin updated: `test/gpu-spatial-analysis/geometry-topology/snap-rounding-reference.ts` (`referenceLatticeFrame`), which encoded the old step (its "far offset" case expected 0.25 at `y = 20,000,000`, now 2). | `test/gpu-spatial-analysis/polygon-overlay/polygon-overlay-float32-lattice.spec.ts` (near `2^23` a strip clipped by a rectangle: four distinct vertices and doubled area 6; the same shapes at the origin keep the `2^-10` grid and their half-unit crossings); `snap-rounded-noding-executed.node.spec.ts`. | Overlay, buffer, snap-rounded polygonize and make-valid on data farther from zero than about four times its extent: the lattice is the float32 grid of the largest coordinate instead of a finer one that could not be stored (near `2^23`: step 1 instead of `2^-10`), so no two output vertices coincide. Features thinner than that step collapse, as they do at any lattice step. Data within four extents of zero is unchanged. |
| polygon-topology-07 | fixed | `src/gpu-spatial-analysis/rectangle-clip/rectangle-clip-kernels.ts:341-356`: `getIntersection` interpolates from the lexicographically smaller endpoint (x, then y). | `test/gpu-spatial-analysis/rectangle-clip/gpu-rectangle-clip.spec.ts` ("gives both polygons of a shared edge the same clip point": the report's edge and 200 random shared edges, bit for bit). | Polygon clip points can move by one float32 step for edges walked from their larger endpoint; two polygons that share an edge now get identical clip points. `GPULonLatPolygonPreparation` uses the same stage. |
| polygon-topology-08 | fixed | `src/gpu-spatial-analysis/polygon-coverage-topology/coverage-topology-kernels.ts:143-233`: ring winding is the exact turn (`orientSign`) at the lexicographically smallest vertex; the float32 shoelace sum is kept only as the fallback for a zero or uncertain turn. | `gpu-coverage-topology-float32.spec.ts` (two clockwise slivers of exact doubled area -1, one for each product a fused multiply-add can keep exact: dissolve writes a counter-clockwise shell, validity flags nothing beside a neighbour). The report's sliver alone does not fail on the test GPU, which fuses the second product; the second sliver does. | Dissolve of such a sliver: a shell (before: a ring wound as a hole, so no polygon). Validity: no `overlap` on its matched edge. |
| polygon-topology-09 | fixed | `src/gpu-spatial-analysis/polygon-buffer/gpu-buffer-surface.ts:383-416`: the collapsed offset ring is chosen from the ring's role after the exact `GPUGeometryOrientation` pass; the second float32 shoelace sum is gone. | `test/gpu-spatial-analysis/polygon-buffer/polygon-buffer.spec.ts` ("buffers a thin strip whose float32 shoelace sum is unreliable"). It passes before and after the change: I found no input where the choice changes the surface (where the sum fails, the two offset rings are mirror images to float32 precision). The sub-step slivers of 08 cannot be used here, because the offset-curve joins fail on them first (see below). | None observed. |
| polygon-topology-10 | fixed (documentation) | `src/gpu-spatial-analysis/rectangle-clip/gpu-rectangle-clip.ts:160-169`, `docs/api/gpu-spatial-analysis/operators-geometry.md`: polygon clip points hold the bound itself; line clip points are `a + (b - a) * t` and can be one float32 step off the edge. The line kernel is unchanged. | None (words only). | None. |
| polygon-topology-11 | fixed (documentation) | `src/gpu-spatial-analysis/delaunay-tessellation/gpu-voronoi-diagram.ts:77-81`, `docs/api/gpu-spatial-analysis/operators-geometry.md`: two sites or collinear sites give no triangle and so no segment. | None (words only). | None. |
| polygon-topology-12 | fixed (documentation) | `src/gpu-spatial-analysis/geometry-validity/geometry-validity-types.ts:14-20`, `docs/api/gpu-spatial-analysis/operators-validity.md`: `shortRing` counts rows after dropping a closing duplicate, not distinct points. | None (words only). | None. |

Extra findings, fixed in this pass.

| ID | What was wrong | What changed | Spec |
| --- | --- | --- | --- |
| extra-A (`wrong-edge-case`) | `GPULineNoding` wrote a zero-length row for every repeated vertex, although its port says repeated vertices are resolved. In polygonization that row is a self-loop. Found when the closing vertex of 04 made the existing "doubly covered lens" scene return 1 polygon instead of 2. | `src/gpu-spatial-analysis/geometry-topology/gpu-line-noding.ts:262-286`: a segment between equal piece vertices is not emitted. | `gpu-make-valid-float32-edges.spec.ts` ("reads a repeated vertex as one vertex"). Caller-visible: `GPULineNoding` `count` no longer includes zero-length rows. |
| extra-B (`wrong-edge-case`, ring assembly) | A zero-length segment read as "straight on" in the turn order. With a lower index than the real segment into its vertex it took that segment's continuation: a square plus one zero-length row at a corner gave 0 rings. | `src/analysis-core/ring-assembly/gpu-segment-ring-assembly.ts:800-807`, `:851`: a zero-length segment takes no continuation, is no candidate, and is flagged `DANGLING` (it counts as open). | `gpu-segment-ring-assembly-hole-probe.spec.ts` ("keeps a ring whole when a zero-length segment sits on it"). |
| extra-C (`wrong-edge-case`) | `GPUCoverageValidity` tested "segment inside another polygon" at the float32-rounded midpoint. For a polygon thinner than one float32 step that point is across the neighbour's edge, so a valid sliver was flagged `overlap` (seen on the second sliver of 08 once the orientation was right). | `src/gpu-spatial-analysis/polygon-coverage-topology/gpu-coverage-validity.ts:461-474`, `:494-505`: the interior test uses the shared exact side probe. | `gpu-coverage-topology-float32.spec.ts` ("accepts a clockwise sliver beside its neighbour"). |
| extra-D (id collision) | `GPUPolygonPrecisionReduction` passes its own id to `GPUMakeValid` and already owns transients named `<id>-closed-positions` and `<id>-closed-offsets`. | The make-valid transients are named `<id>-ring-closure-*`. | `test/analysis-core/kernel-audit-geometry.node.spec.ts`. |

Ring assembly (`analysis-core/ring-assembly/**`, not opened by the review), read in full.

- Hole assignment: the side probe was the float32 hazard (01, fixed and pinned). The bounding-box
  reject uses the rounded midpoint, which is safe (it lies between the edge's endpoints, inside the
  shell's closed box whenever the hole is). The innermost shell is chosen by float32 `ringAreas`
  ("strictly larger than the hole", "smallest containing"). I could not build a failing input: two
  nested shells or a hole and its shell would need areas within one part in 2^24 of each other.
  Not changed.
- Turn ordering: exact when every candidate starts exactly at the segment's end vertex and the
  context is planar (pinned by the existing exact-turn spec). With a tolerance match or
  longitude-latitude input the float32 pseudo-angle decides and near-parallel candidates tie to the
  lowest index. That is documented in the class and is inherent to tolerance matching (the turn at
  two different points is not one exact predicate). Not changed.
- `polygon-keys` drops rings whose float32 area about the first vertex is exactly 0, so a sliver
  face of non-zero exact area can vanish from `polygons` (the float32 sum of the 08 sliver is 0
  from one of its three starting vertices). Not fixed: telling a zero-area walk from a thin face
  exactly needs an exact area sign, which the kernel does not have. Recorded here as open.
- Zero-length segments: fixed (extra-B).

Not mine, seen while testing.

- `outline-geometry/gpu-offset-curve.ts` (owner: another stream): for a sliver thinner than one
  float32 step the turn at the tip is a float32 cross product of two long edges and reads as
  straight, so the round join there is missing. `GPUBufferSurface` of the two slivers of 08 at
  distance 1e7 lost one or both end caps (area 20% and 40% low). Not fixed.
- `test/analysis-core/doc-parameters.node.spec.ts` reports `gpu-spatial-analysis/operators-weights.md`
  as stale. That page is the spatial-weights stream's.
- `spatial-weights/gpu-contiguity-weights.ts` had the absolute snap key when this was written. The
  spatial-statistics stream has since made it relative and now imports `getCoverageSnapKeyWGSL`
  from `polygon-coverage-topology/coverage-topology-kernels.ts` (fix review fr-geometry-08).

Upstream gaps: none new. Published luma.gl has no noding, overlay, coverage, clip or buffer class to
delegate to (section 5).

### Review follow-up (2026-11-10)

After `fix-review-geometry.md`. Every spec named was run on the headless GPU and passes; the two
new GPU specs were run against the kernel as it stood and failed there.

| Item | What changed | Spec | Behaviour change a caller sees |
| --- | --- | --- | --- |
| fr-geometry-07 | The snap-range report left `status.overflow`. Contract section 9 asks for an existing name: `uncertainCount` ("classifications that could not be certified under the selected precision policy ... resolved through the documented uncertainty fallback") is the one whose meaning matches, because the vertex is not rejected (`invalidCount`) and nothing is deliberate (`approximation`): its snap cell cannot be certified in float32 and its identity comes from the inexact key. `src/gpu-spatial-analysis/polygon-coverage-topology/coverage-topology-kernels.ts:75-125` now writes an exact count (atomic add; always 0 for tolerance 0). New optional `output.status.uncertainCount` on `GPUCoverageDissolve` (`gpu-coverage-dissolve.ts:64-71`, `:589-601`; it is split off before the ring-assembly status) and `GPUCoverageSimplification` (`gpu-coverage-simplification.ts:118-125`, `:1043-1055`). `overflow` is capacity only again. `getCoverageSnapKeyWGSL` is unchanged (imported by `spatial-weights/gpu-contiguity-weights.ts`). `docs/api/gpu-spatial-analysis/operators-lines.md`. | `test/gpu-spatial-analysis/polygon-coverage-topology/gpu-coverage-topology-float32.spec.ts` ("counts vertices outside the exact snap range": 0 inside, 2 outside with `overflow` 0, 0 without a tolerance); `test/gpu-spatial-analysis/polygon-coverage-simplification/gpu-coverage-simplification.spec.ts` (same three cases). | A vertex outside the exact key range no longer sets `overflow`. It is reported only when the caller supplies `status.uncertainCount`. |
| Sliver faces in ring assembly | `src/analysis-core/ring-assembly/gpu-segment-ring-assembly.ts:1429-1443`: when a ring's float32 area is exactly 0 but it has an exact non-zero turn at a lowest vertex it passes once, `ringAreas` gets the smallest normal float (`1.17549435e-38`) with the sign of the turn. Every "area is zero, so the walk is empty" test downstream (`polygon-keys` here, the bounded-face test of `GPUPolygonize` diagnostics) then keeps the ring, with no new port. The pass count guards against a walk out and back whose lowest vertex is a bend (two passes, opposite turns); the walk over the ring runs only in the zero-area case. | `test/gpu-spatial-analysis/ring-assembly/gpu-segment-ring-assembly-hole-probe.spec.ts` ("keeps a sliver face whose float32 area is exactly zero": two counter-clockwise slivers of exact doubled area 1, one per product a fused multiply-add can keep exact; a straight and a bent slit stay empty). On the test GPU the first sliver failed before the change. | Such a ring is a polygon in `polygons` (before: left out) and its `ringAreas` is `1.17549435e-38` or its negative (before: 0). |
| Sliver tips in the offset curve (`outline-geometry`, taken over) | `src/gpu-spatial-analysis/outline-geometry/gpu-offset-curve.ts:435-441`: the side that is the inside of a turn is `orientSign(previous, vertex, next)`; the float32 cross product of the unit directions is kept only when the exact predicate returns code 2 (non-finite input). `docs/api/gpu-spatial-analysis/operators-geometry.md`. | `test/gpu-spatial-analysis/polygon-buffer/polygon-buffer.spec.ts` ("keeps the end caps of a sub-step sliver", both slivers; area within 1% of the stadium `2 L d + pi d^2`; before 20% and 40% low). | `GPUOffsetCurve`, `GPUBufferSurface` and the outline users: a join at a vertex where the edges nearly reverse is on the correct side. Where the float32 cross product had the right sign, nothing changes. |
| Stale references | Line numbers of the rows for 01, 02, 03, 08 and extra-B updated to the current files; the note on `gpu-contiguity-weights.ts` corrected (fr-geometry-08 is that stream's). No other row of the review names a file of this scope. | | |

Still open: the staggered-gap question of 05 (needs one Shapely run), the float32 innermost-shell
choice and the float32 turn key on tolerance matches in ring assembly (no failing input built).

### Closing pass (2026-11-11)

The three items left open above. Reference library: Shapely 2.2.0 on GEOS 3.14.1 (pinned CPython
3.14.8 environment). Every spec named was run on the headless GPU. Each new spec was also run
with the old rule put back for one run, and failed there. Scratch scripts and logs:
session scratchpad, folder `coverage-rings/`.

| Item | Outcome | What changed | Spec that pins it | Evidence |
| --- | --- | --- | --- | --- |
| polygon-topology-05, staggered gap | fixed (kernel) | GEOS flags the pair: `coverage_invalid_edges` of `[0,10] x [-5,0]` and `[5,15] x [0.0625,5]` returns `(10 0, 0 0)` and `(5 0.0625, 15 0.0625)` at `gap_width` 0.125 and 0.0625, nothing at 0.03125 or 0, and `coverage_is_valid` agrees. The kernel flagged nothing. `src/gpu-spatial-analysis/polygon-coverage-topology/gpu-coverage-validity.ts:401-449` is now the GEOS rule (`InvalidSegmentDetector::isNearlyParallel`): each segment is projected on the other and clipped to it; a gap when both clipped projections are longer than `gapWidth` and their matching ends are within it. It is evaluated relative to the start of the tested segment, so the float32 step of projected coordinates does not enter the distances; the projected lengths keep the division-free form the kernel already had. Class documentation, the `gap` flag, `docs/api/gpu-spatial-analysis/operators-validity.md` and the cross-reference row say so. | `test/gpu-spatial-analysis/polygon-coverage-topology/gpu-coverage-topology.spec.ts` ("matches GEOS on gaps between staggered segments": 57 scenes, 222 cases, 4,360 invalid segments of the new fixture `test/oracles/fixtures/coverage-validity-gaps.json`, generator `test/oracles/coverage-validity-gaps.py`); `gpu-coverage-topology-float32.spec.ts` (review 05: the pair is flagged on rows 2 and 4 at 0.125, nothing at 0.03125; it asserted the opposite before). | (a) My transcription of the GEOS rule equals GEOS on 15,244 cases of random disjoint polygon pairs (1,640 with flags); the old rule differs on 955 of them. (b) GPU against GEOS: the old kernel failed 85 of 342 cases of the first fixture draft (11 of 22 curated, 39 of 160 disjoint pairs, 35 of 160 jittered triangulations); the new kernel fails none, nor any of 2,022 cases generated with other seeds (507 scenes, 63,285 invalid segments, 600 of the cases at easting 500,000 / northing 4,500,000). (c) The 28 curated scenes of the existing Shapely fixture still agree. |
| Ring assembly, innermost shell of a hole | fixed (kernel) | Failing input built. A thin wedge with its apex near `(1, 1)` and a far side near `(6000, 5900)` that is 12 float32 steps tall, with a vertex at every step, and a hole one step inside it: float32 areas 6.5722 (shell) and 7.2638 (hole), so the shell was skipped by "strictly larger than the hole" and the hole had no shell (`ringShells` = `NONE`, hole missing from `polygons`). With four nested rings the inner hole went to the outer shell. `src/analysis-core/ring-assembly/gpu-segment-ring-assembly.ts:1582-1616`, `:1623-1632`: a shell is a candidate when its bounding box contains the hole's box, and of two shells that contain the probe the inner one is the one whose box lies inside the other's. Boxes are minima and maxima of stored vertices, so both tests are exact, and for a planar boundary (rings that do not cross) they are equivalent to the true area order. Float32 areas now order only shells whose boxes are equal or do not nest. The exact probe already kept a reverse walk from owning itself, so the area test was not needed for that. | `test/gpu-spatial-analysis/ring-assembly/gpu-segment-ring-assembly-float32-choices.spec.ts` (direct, outer ring first and inner ring first; and through `GPUCoverageDissolve`): 13 wedge scenes and 3 projected band scenes of `test/oracles/fixtures/ring-assembly-float32-choices.json` (generator `test/oracles/ring-assembly-float32-choices.py`). With the old rule 8 of the 13 wedge scenes fail, in both orders and through the dissolve. | Expected owners are `shapely.build_area` of the rings. Search on the GPU before the fix, 5,900 runs of generated wedge scenes: with a far side of two vertices 0 of 3,700 fail; with a vertex at every step and two rings 2 of 300 fail at 12 steps (the same 2 in the other ring order), 0 of 600 at 30, 0 of 100 at 60 and 49 of 100 at 120; with four rings 0 of 1,200 holes are wrong at 30 steps, 1 of 200 at 60 and 71 of 200 at 120. After the fix 0 of 4,200 holes are wrong. **Not reproduced at projected-metre coordinates:** nested bands one grid step apart at 500,000 / 4,500,000, 60 m to 18 km long with up to 2,000 vertices per ring, 18 runs, no failure before or after. There the ring-relative coordinates are multiples of 1/32 m and 1/2 m, every cross product is a multiple of 1/64 m^2 and exact below 262,144 m^2, so the float32 area of a ring inside a box of about 360 m is exact; beyond that the areas of rings a grid step apart differ by far more than the rounding (band area against `2^-24` of the products). |
| Ring assembly, turn key at a tolerance match | fixed (kernel) | Failing input built: the tie scene of `gpu-segment-ring-assembly-exact-turns.spec.ts` (lattice directions `(N - 1, N - 2)` and `(N, N - 1)`, `N = 2^21`, cross product 1) with the second triangle moved a quarter unit, so that its tip matches the first triangle's tip within the tolerance without being equal. The exact path was then off, the two float32 turn keys were equal, the lowest index won, and one triangle was lost (1 ring, 3 open segments), for each of five shifts. `gpu-segment-ring-assembly.ts:81-96`, `:864-869`, `:938-973`: when a candidate does not start on the end vertex, the order is the exact one on the direction vectors `end - start` of each segment (`orientSign` with the origin as first point is the exact sign of their cross product). The vectors are float32 differences: exact on lattice and projected coordinates and wherever the subtraction is exact, otherwise correct to their own rounding. The fully exact path for candidates that start on the vertex is unchanged. Longitude/latitude input still uses the float32 key. | Same spec: the scene with the tip moved left and up, interior left and right, direct; and through `GPUCoverageDissolve` with `snapTolerance` 2 anchored so that the two tips fall in neighbouring snap cells. All fail with the old rule. | `shapely.union_all` of the two triangles has two parts (they are 0.18 apart). The near u-turn scene of the same spec, moved the same way, did not fail on the test GPU (it fuses the product that keeps the sign). |

Regression runs after the changes, all passing: headless `ring-assembly`, `polygon-coverage-topology`,
`polygon-overlay`, `geometry-topology` (19 files, 265 tests); isolines, cell-set outline, polygon
buffer, urban ecology, geometry refinement, global coordinates, coverage simplification (36 files,
338 tests); node specs of the same directories with the geometry kernel audits and the
documentation checks (22 files, 758 tests). `check-upstream-first` and `check-contributor-contract`
report nothing new.

Not verified:

- The lens scenario `gpu-spatial-analysis.coverage-validity` (199 tracts, gap width 1 m) and anything
  else in `modules/lenses`, `modules/kepler` or `website` that runs `GPUCoverageValidity` with a gap
  width. Their counts can change; I did not run them.
- Gap pairs whose distance or projected length equals the gap width to within float32 rounding.
  GEOS decides those in binary64 and the kernel in float32. The fixture's ties are on dyadic
  coordinates, where both are exact, and agree.
- The innermost-shell and turn fixes through `GPUPolygonOverlay` and `GPUPolygonize`. I ran them
  directly and through `GPUCoverageDissolve`; the existing overlay and polygonize specs pass.
- A turn at a tolerance match where the direction vectors themselves round (segment ends that
  differ in magnitude by more than a factor of two, far from a lattice). No failing input was
  looked for there, and none for the longitude/latitude turn key.
- Rings that cross (not a planar boundary): there the bounding-box rule and the old area rule can
  give different owners, and neither result is defined.

### Closing pass, review fixes (2026-11-11)

The ring-assembly finding of [fix-review-closing-geometry.md](./fix-review-closing-geometry.md).
The five triangulation findings are in [core-numerics.md](./core-numerics.md). Where this
subsection and the one above disagree, this one is current: the row "turn key at a tolerance
match" above describes the order on float32 direction vectors, which is replaced, and its last
sentence on longitude/latitude input no longer holds. Reference library: Shapely 2.2.0 on GEOS
3.14.1 and exact `fractions` (pinned CPython 3.14.8). Source lines are those of
`src/analysis-core/ring-assembly/gpu-segment-ring-assembly.ts` after the fix.

| Finding | Outcome | Cause, and what changed | Spec that pins it | Evidence |
| --- | --- | --- | --- | --- |
| frc-geometry-06 (minor) | fixed, planar and longitude/latitude | Reproduced on the GPU with the kernel as it was: of the reviewer's 240 scenes 115 lose rings (the reviewer's count), and of the 64 scenes of the new fixture 47. Cause as the review states it: at a tolerance match the candidates were ordered by `orientSign` of float32 vectors `end - start`, which are not the directions when the two ends of a segment differ in magnitude. The order is now that of the directions themselves. `ringCrossSign(a, b, c, d)` (`:178`) is the exact sign of `cross(b - a, d - c)` for four float32 points: the eight products of raw coordinates are summed exactly with `multiply24` and `accumulateShifted` of `analysis-core/kernels/exact-orientation-wgsl.ts` (`ringCrossExact`, `:145`), behind a float filter. `ringSegmentTurnClass` (`:202`) classifies the turn from the incoming segment to a candidate with it, and two candidates of one half-plane are ordered with it (`:1059`, `:1071`). With a common vertex the predicate is `orientSign` of three points, and the path for candidates that all start on the end vertex keeps using that. It is a ring-assembly helper built on the kernels' integer machinery, not a new file in `analysis-core/kernels`, which this stream does not own; the terms are recomputed instead of stored in a local array (contributor contract, section 11). **Longitude/latitude.** That input used a float32 pseudo-angle of differences scaled by the cosine of the latitude, for exact matches too. The scale is a positive factor on one axis, which changes the sign of no cross product, so the order of the turns at a vertex is the same with and without it. Both exact paths now run for longitude/latitude input on the degrees as they are (`:1123`: the helpers are always in the kernel); the pseudo-angle is left for coordinates that are not finite. | `test/gpu-spatial-analysis/ring-assembly/gpu-segment-ring-assembly-rounded-directions.spec.ts` (3 tests) on `roundedTurns` of fixture `ring-assembly-float32-choices.json` (generator `test/oracles/ring-assembly-float32-choices.py`): 64 pairs of triangles that meet within the tolerance near `(1, 1)` with far vertices one to five million away and edges about 1e-7 radians apart; in 48 the float32 differences give another continuation than the exact directions | Expected continuation: exact rational directions (`fractions`); expected rings: `shapely.union_all` of each pair has two parts. Fixture scenes, kernel as it was: 47 of 64 lose a ring planar, 36 of 64 as longitude/latitude (scaled by 2^-16 to degrees); after: 0 of 64 planar with the interior on the left, 0 of 64 mirrored with the interior on the right, 0 of 64 as longitude/latitude. Reviewer's 240 scenes: 115 fail planar and 124 as longitude/latitude before; 0 and 0 after |

Regression runs after the changes of both modules, all passing, with no assertion changed in any
spec of another contributor: the 46 headless and 29 node spec files that name a user of the two
modules (polygon overlay, polygonize, make-valid, line noding, coverage dissolve and validity,
polygon buffer, cell-set outline, isoband rings and their parity specs, network isochrones and
the drive-time recipe, dot density, random points, subdivision, fractional coverage, the recipes)
ran before the first change (506 and 319 tests) and after the last (518 and 326 tests: the same
plus the new ones, and two tests another stream added to
`gpu-raster/fractional-cell-size-mode.spec.ts` in between). Kernel audits and documentation
checks: 9 node files, 819 tests. `check-contributor-contract`: no new findings. Both fixtures
regenerate `IDENTICAL` with `scripts/run-oracle-fixture.mjs`; the wedge, band and turn scenes of
the ring-assembly fixture are unchanged by the added family.

Assertions changed in existing specs: one, in an owned spec.
`gpu-polygon-triangulation-search.spec.ts` now requires `flagged` to be 0 for the Earcut path with
touching rings too (it was required only without them); nothing was loosened.

Not verified:

- Lens records: not regenerated. The kernel whose WGSL changed here is the `match` kernel of
  `GPUSegmentRingAssembly` (node `<id>-match`), for planar and for longitude/latitude input, and
  with it every contributor that assembles rings: `GPUCoverageDissolve`, `GPUPolygonOverlay`,
  `GPUPolygonize`, `GPUMakeValid`, `GPUIsobandRings`, `GPUCellSetOutline`, `GPUBufferSurface`,
  `GPUNetworkIsochrones` and the drive-time catchment recipe.
- Longitude/latitude input beyond the 64 scaled scenes and the existing specs (H3 and Quadbin
  outlines, isochrones), which pass. No real data set was compared before and after; a result
  can differ only where the float32 key rounded or tied.
- The float filter of `ringCrossSign` trusts the test GPU to round a difference and a product
  once each, as the filter of `orientSign` does; a compiler that distributed the product over the
  difference would need a wider bound. The 64 fixture scenes and the reviewer's 240 are decided
  inside the filter's band and pass.
- Coordinates whose products span more than 200 binary orders, and non-finite coordinates: the
  exact sign is not certified and the float32 pseudo-angle decides, as before.
- Other GPUs and the software adapter.

**Specs and commands** (both modules; `npx vitest run --project headless` and `--project node`
from the repository root).

- Headless, 11 files, 64 tests pass:
  `modules/analysis/test/gpu-spatial-analysis/polygon-triangulation/gpu-polygon-triangulation-touching-rings.spec.ts`
  (new, 7 tests), `.../gpu-polygon-triangulation-search.spec.ts`,
  `.../gpu-polygon-triangulation.spec.ts`, `.../gpu-polygon-triangulation-monotone.spec.ts`,
  `modules/analysis/test/analysis-core/polygon-triangulation-hole-bridge.spec.ts` and the six
  specs of `modules/analysis/test/gpu-spatial-analysis/ring-assembly/`, among them the new
  `gpu-segment-ring-assembly-rounded-directions.spec.ts` (3 tests).
- Node, 8 files, 63 tests pass: `modules/analysis/test/gpu-spatial-analysis/polygon-triangulation/`
  (with the new `polygon-triangulation-search-cases.node.spec.ts`, 7 tests) and
  `.../ring-assembly/`.
- On the kernels as they were, `gpu-polygon-triangulation-touching-rings.spec.ts` fails in its
  four GPU tests of that time (60 departures on the Earcut path, among them the two
  frc-geometry-01 inputs with `valid = 1` and wrong and the frc-geometry-03 and -04 inputs with
  `valid = 0`; 42 at float32 resolution; 7 wrong results of the monotone path; 5 of the 16
  enlarged polygons in the default configuration, 2 of them `valid = 1` and wrong), and
  `gpu-segment-ring-assembly-rounded-directions.spec.ts` fails in both GPU tests. The old
  `getTilingDefect` returns `undefined` for the review's counterexample.
- Spec typecheck: `npx tsc --noEmit -p` on a scratch tsconfig that lists the nine changed or new
  spec and helper files: no errors. Source: `node scripts/typecheck-status.mjs
  'analysis-core/(ring-assembly|polygon-triangulation)'`: 0 errors.
- Scratch (generators, the CPU model, GPU result files, logs): session scratchpad, folder
  `geometry-fixes/`.

**After the re-review (2026-11-11): frc-geometry-07.** The re-review at the end of
[fix-review-closing-geometry.md](./fix-review-closing-geometry.md) closes the six findings and
adds one, older than the fixes above.

| Finding | Outcome | Cause, and what changed | Spec that pins it | Evidence |
| --- | --- | --- | --- | --- |
| frc-geometry-07 (minor) | fixed | Reproduced on the GPU with the kernel as it was after the fixes above: the mirror image of the spike square with the interior on the right is 1 ring and 2 open segments (rows 1 and 2), where the square with the interior on the left is 2 rings; the 3 polar scenes with an exactly opposite candidate lose both rings when mirrored; 216 of 300 lattice fan scenes differ from their mirror images. Cause as the re-review states it: turns were classed right, straight, left-or-u-turn and the order reversed for `'right'`, so an exact u-turn was the first choice for `'left'` and the last for `'right'`. The orientation signs are now multiplied by `SIDE_ORDER` before they are classed or compared (`ringExactTurnClass`, `ringSegmentTurnClass`, the two comparisons of candidates of one half-plane), and the order is no longer reversed; the float32 key of the fallback takes the cross product times `SIDE` instead of being multiplied by it. The kernel for `'right'` therefore decides as the kernel for `'left'` does on the mirror image, and the u-turn is the tightest turn for both. For `'left'` nothing changes. For `'right'` the choice changes only where a candidate exactly reverses the incoming segment: by case analysis the new class and comparison equal the old ones for every other pair of candidates. | `test/gpu-spatial-analysis/ring-assembly/gpu-segment-ring-assembly-mirror-symmetry.spec.ts` (4 tests): the spike square as given, mirrored, reversed, and reversed and mirrored, each with the matching interior side; the reviewer's 240 polar scenes (`polarTurns` of fixture `ring-assembly-float32-choices.json`, the reviewer's generator ported into `ring-assembly-float32-choices.py` with seed 5, which reproduces the reviewer's file value for value) as generated with the interior left and mirrored with the interior right, planar and as longitude/latitude; 300 fans of lattice triangles at one vertex with exactly opposite and repeated directions, sharing the vertex exactly and within a tolerance, each against its mirror image for both interior sides, compared as partitions of the segment rows into rings | Spike: the rings of the face rule by hand (row 2 continues into its reversal, row 1; rows 0, 3, 4, 5 are the square), the same four times. Polar scenes: `shapely.union_all` of each pair has two parts; 240 of 240 give both rings in all four runs (237 mirrored before). Fans: 0 of 300 differ from their mirror image in each of the four comparisons |

Whether a u-turn should be the tightest turn at all is the reviewer's reading of the face rule
and the behaviour `'left'` always had. It gives a zero-width spike its own ring of two segments.
For two faces that share an edge given in both directions it sends both edges into each other
and leaves a neighbour without a continuation; that case is what `pairedOpposites` and
`cancelOpposingSegments` are for and is unchanged. (The fan scenes of the spec overlap and
repeat triangles on purpose and leave most of their segments open; they test the reflection,
not the rings.)

The `match` kernel's WGSL changed again, for both interior sides and both coordinate spaces
(the text; the decisions only for `'right'`). Lens records: not regenerated.

Regression after this change, with no assertion changed: the same 46 headless and 29 node
spec files as above (ring assembly, overlay, polygonize, make-valid, dissolve, coverage, buffer,
cell-set outline, isobands, isochrones and the triangulation users), run before (518 and 326
tests) and after (522 and 326: the same plus the four new tests). The ring-assembly folder
alone: 7 headless files, 40 tests, and 3 node files, 11 tests. The fixture regenerates
`IDENTICAL`; its earlier families are unchanged by the added one. `check-contributor-contract`:
no new findings.

Not verified: `interiorSide: 'right'` through `GPUCellSetOutline` on real Quadbin data beyond
the existing specs (a Quadbin outline has no candidate opposite to an incoming segment unless a
cell is repeated); other GPUs.

Correction to [core-numerics.md](./core-numerics.md): the polygon that stays flagged is index
2192 of the reviewer's `tri.json` (32 vertices), not 1988, which is its position among the
polygons that map to the 1/32 m by 1/2 m grid. The account there is corrected.
