# Fix review: closing pass, geometry scope (coverage gap rule, ring assembly, hole bridge)

Date: 2026-11-11. Independent review of three kernels changed in the closing pass. I wrote none of
the code. There is no commit to diff against, so I read the current tree; for the triangulation a
copy of the kernel as it was before the pass was found in the authors' scratch folder and used as
the "old kernel". Paths are relative to `modules/analysis` unless they start with `dev-docs/`,
`docs/` or `scripts/`.

Method. The authors' accounts were read as claims: the "Closing pass (2026-11-11)" subsections of
`polygon-topology.md` and `core-numerics.md` and their rows in `behaviour-changes.md`. Then:

- the three kernels, their specs, generators and `docs/api` sections were read; the gap rule was
  checked against the GEOS source rule (`InvalidSegmentDetector::isNearlyParallel`,
  `LineSegment::project`) and the hole bridge against `node_modules/earcut` 3.2.4;
- probe specs were run on the test GPU (headless Chromium, WebGPU, Apple Metal) from
  `test/review-scratch-geometry/`, which is deleted. Inputs came from Python generators with
  Shapely 2.2.0 / GEOS 3.14.1 as the oracle (the pinned `venv314`). Triangulations were judged by
  an exact checker written for this review (BigInt; pairwise overlap, no polygon edge through an
  open triangle, centroid inside, areas equal), by the authors' `getTilingDefect`, and by the
  `earcut` package;
- single changes of the triangulation pass were reverted one at a time in scratch copies of the
  kernel to attribute behaviour changes.

Generators, logs and copies of the probe specs are in
`/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-luma-contributors/ba3fe269-f86e-4664-86ec-632935ba200a/scratchpad/review-geometry/`
(specs under `probes/`).

"CONFIRMED" means run on a concrete input (on the GPU where the row says so) or derived by hand on
one.

Summary. No blocker: the three changes do what the accounts say on the inputs the accounts name,
and no wrong result found here is new. Two major findings, both in the triangulation and both
older than the pass: a `valid = 1` result can be a wrong triangulation, which the accounts say
does not happen. Four minor ones, one of them a regression to `valid = 0` on 18 polygons with
touching rings, against 549 polygons the pass repairs. The gap rule and the shell choice held on
every input tried.

## 1. Findings

### frc-geometry-01 (major, CONFIRMED on the GPU): the Earcut path returns `valid = 1` with a wrong triangulation when a hole vertex lies inside an edge of another ring

- Where: `src/analysis-core/polygon-triangulation/gpu-polygon-triangulation.ts:1137-1205`
  (`findHoleBridge`: the first walk at `:1152` looks only at downward, non-horizontal edges, and
  `touchesEdge` at `:1184` covers one direction of a horizontal edge), `:1325-1335` (`valid` is
  "not failed and not incomplete"; the triangles are not checked). Claims:
  `core-numerics.md:543-545` ("The port flags them; it does not return a wrong triangulation"),
  `docs/api/gpu-spatial-analysis/operators-geometry.md:492-495` ("none came back valid with a
  wrong triangulation").
- What is wrong. The authors' touching families move a hole so that one of its vertices equals a
  vertex of another ring. A hole can also touch another ring at a point that is not a vertex of
  that ring. Such a polygon is valid (Shapely `is_valid`), and for some of them the port returns
  `valid = 1` with triangles that overlap and cover part of a hole. The old kernel and the
  `earcut` package return the same wrong result, so this is not a regression; it is the case the
  accounts say does not occur, and "holes that touch the shell along an edge" in "Not verified"
  does not describe it (one point, not an edge).
- Concrete input (integer lattice, 11 vertices). Shell `(7,2) (8,0) (8,7) (3,12) (0,11)`, holes
  `(5,9) (3,9) (5,10)` and `(7,4) (4,9) (7,5)`. Vertex `(4,9)` of the second hole lies inside
  the edge `(5,9)-(3,9)` of the first. Polygon area 33.
  - Kernel, `largePolygonMethod: 'none'` and the default: `valid = 1`, 11 triangles (rows)
    `[6,4,0] [0,1,2] [3,4,9] [3,9,10] [8,9,4] [4,6,7] [5,6,0] [7,5,0] [8,4,0] [10,0,2] [10,2,3]`,
    areas summing to 52. Shapely on these triangles: union 34.67, of which 1.67 lies in the holes.
  - Right: any tiling of area 33. With `(4,9)` added to the first hole's ring (same polygon) the
    kernel returns one (11 triangles). Mirrored in x or in y the original input also tiles.
  - Second input, hole against shell (15 vertices): shell
    `(0,7) (1,2) (7,7) (11,1) (22,14) (12,13) (24,17) (22,22) (15,18) (11,17) (3,22) (5,15)`, hole
    `(15,16) (19,17) (15,14)`; `(15,14)` lies inside the shell edge `(12,13)-(24,17)`. `valid = 1`,
    14 triangles, area 258 against 244, 14 outside the polygon.
  - A third (three holes on shell edges, 28 vertices) gave the same wrong result on the lattice,
    at 500000 / 4500000 on the 1/32 m by 1/2 m grid and on the 2 m lattice.
- How often. 21,425 valid polygons with rings that touch (seven families, 8 to 79 vertices, 1 to
  6 holes): 7 wrong with `valid = 1`, 98 flagged. All 7 have a hole vertex inside an edge of
  another ring: 3 of 7,850 polygons built with a hole on a shell edge, 4 of 3,850 with a hole on
  a hole edge. After every such point was added to the touched ring as a vertex (11,001 polygons,
  all touches vertex to vertex): 0 wrong. So the authors' statement holds for vertex-to-vertex
  touches in this sample and fails for vertex-in-edge touches.
- Suggested fix. In the first walk of `findHoleBridge`, before the downward-edge test, when the
  hole vertex is on the edge (`orientOf == 0` and inside its box, either direction, horizontal
  included): insert a node with the hole vertex's row between the edge's ends and return it, so
  the existing coincident-vertex bridge is taken. Tried in a scratch copy of the kernel (11
  lines): on 21,950 polygons the 7 wrong results became tilings, the flagged went from 98 to 16
  and no tiling was lost. Not run against the pinned specs. It covers a touching vertex that is
  the hole's leftmost (6 of the 7) and a vertex on an earlier bridge; a touching vertex that is
  not leftmost needs the same insertion for every hole vertex, or a note in the documentation.
  Either way, pin the two inputs above and reword the two claims.

### frc-geometry-02 (major, CONFIRMED on the GPU): the monotone path returns `valid = 1` with a wrong triangulation for some polygons whose rings touch, also in the default configuration

- Where: the monotone path (`polygon-triangulation-monotone-wgsl.ts`, not changed in the pass).
  Claims: `core-numerics.md:511` ("With touching rings 5,431 of 6,337 are flagged, as documented,
  and the 906 that are valid tile"); `gpu-polygon-triangulation.ts:190-196` and `:490`,
  `docs/api/gpu-spatial-analysis/operators-geometry.md:517-520` (touching rings are "flagged
  invalid by the monotone path", "`'monotone'` never does worse than `'none'`").
- What is wrong. The monotone path does not flag every polygon whose rings touch, and some that it
  passes are wrong. Its checks (triangle count, monotone chains, closed walks) are consistent on
  them.
- Concrete input, monotone path alone (`monotoneMinimumVertices: 3`, `maximumWork: 1`), 15
  vertices, vertex-to-vertex touches only: shell
  `(0,6) (0,3) (4,3) (4,5) (6,5) (6,10) (2,10) (2,6)`, holes `(4,6) (5,6) (4,5) (3,6)` and
  `(4,6) (4,8) (5,8)`. `valid = 1`, 17 triangles (6 of zero area), areas summing to 29; the
  polygon area is 28, and 0.5 of the triangles lies in a hole. Earcut (`'none'` and the default)
  tiles it.
- Default configuration. A polygon of 4,306 vertices: the lattice polygon with shell
  `(10,12) (22,12) (22,23) (20,23) (20,24) (17,24) (17,23) (10,23)` and holes
  `(16,22) (18,20) (18,18) (13,16)`, `(18,19) (22,22) (21,19)`, `(18,17) (20,19) (18,15)`,
  scaled by 1024, with the shell edges divided evenly by collinear vertices (file
  `tri-large.json` in the scratch folder, tag `#11444`). With `largePolygonMethod: 'none'` the
  result is a tiling. With the default it is `valid = 1` and wrong: the triangle areas sum to
  122,159,104 where the polygon area is 120,061,952. The polygon skips Earcut because of its size,
  the monotone path does not flag it, and so the second Earcut pass never sees it.
- How often, monotone path alone: 3 wrong of 530 valid among 5,425 touching polygons; 13 wrong of
  1,684 valid among 12,000; 12 wrong of 965 valid among 7,001 with vertex-to-vertex touches only.
  Polygons whose rings do not touch: 0 wrong of 4,493 (all valid, tilings once zero-area
  triangles are dropped), which agrees with the authors. Of 15 polygons enlarged to about 4,300
  vertices, the monotone path flagged 14 (Earcut then tiled them) and passed the one above.
- Suggested fix. Flag in the monotone path any polygon in which two cleaned vertices of different
  rings are equal (they are neighbours in the sweep order) or a vertex lies on its left edge
  (exact side zero), and let the Earcut pass take it. Until then remove "never does worse than
  `'none'`" and "flagged invalid by the monotone path" from the class documentation and
  `docs/api`, and correct the row of `core-numerics.md`. Pin the 15-vertex input.

### frc-geometry-03 (minor, CONFIRMED on the GPU): two of the six changes turn some correct triangulations into `valid = 0`, and the accounts list only gains

- Where: `gpu-polygon-triangulation.ts:1190-1192` (core-numerics-13, the tie between collinear
  candidates) and `:1090` (core-numerics-19, `isLeftOf`). `behaviour-changes.md:324` and
  `core-numerics.md` describe changes from `valid = 0` or wrong to a triangulation only.
- Counts, old kernel correct and new kernel `valid = 0` (both methods): 2 of 5,382, 9 of 11,929
  and 7 of 6,987 polygons the old kernel tiled; every one has rings that touch. None among 4,294
  polygons without touching rings that the old kernel tiled. In the other direction the same sets
  have 416 flagged and 133 wrong results that became tilings, so the balance is strongly positive.
- Input for core-numerics-13 (17 vertices): shell
  `(3,5) (2,2) (5,3) (8,2) (9,1) (4,10) (4,9) (2,8) (2,5) (5,6)`, holes
  `(4,5) (6,6) (6,5) (5,5)` and `(5,5) (6,4) (6,3)`. Old kernel: 16 triangles, a tiling. New
  kernel: `valid = 0`. The new kernel with only the tie rule put back: a tiling. The package
  returns a wrong triangulation here, so "as Earcut does" imports its failure.
- Input for core-numerics-19 (11 vertices): shell `(8,7) (7,7) (10,12) (3,9) (5,3)`, holes
  `(6,7) (4,6) (4,7) (5,8)` and `(4,6) (4,6) (5,6) (5,5)`. The holes share their leftmost vertex
  and the second repeats it. Old kernel: 11 triangles, a tiling. New kernel: `valid = 0`. Without
  the repeated vertex, or with the old `isLeftOf` put back: a tiling. `nextOf` of the leftmost
  node is its copy, the orientation is zero both ways, and the holes keep their input order.
- Suggested fix. In `isLeftOf` step over successors at the same point before taking the
  orientation. For the tie rule, add the reverse direction to `behaviour-changes.md` and to the
  account; the insertion of frc-geometry-01 is worth trying on these polygons.

### frc-geometry-04 (minor, CONFIRMED on the GPU): a valid polygon whose rings do not touch is flagged by the Earcut path

- Where: `findHoleBridge` and the bridge order; cause not established. Claim:
  `core-numerics.md:515-516` ("holes not touching ... none flagged, none wrong").
- Input (14 vertices, no touching rings, no repeated vertex): shell
  `(21,13) (15,15) (3,0) (12,8) (14,6) (19,0) (22,2) (21,11)`, holes `(20,6) (19,6) (21,9)` and
  `(18,2) (19,4) (19,2)`. The leftmost vertex `(19,6)` of the first hole is on the row of the
  shell vertex `(14,6)`. `largePolygonMethod: 'none'`: `valid = 0`, on the old kernel too, and
  with each single change reverted. The package tiles it (15 triangles). With either hole alone
  the kernel tiles it. The default method returns a valid tiling from the monotone path (16
  triangles, one of zero area).
- How often: this polygon and its variant with collinear vertices, 2 of 4,493 polygons without
  touching rings. The other 4,491 tile.
- Suggested fix. Find the cause (two nodes at `(14,6)` after the first bridge is the likely
  place) and pin the input; until then the sentence in the account and the "one polygon whose
  holes do not touch" remark in `docs/api` should not read as "never flagged".

### frc-geometry-05 (minor, CONFIRMED in node): the tiling checker of the search spec passes a set of triangles that is not a tiling

- Where: `test/gpu-spatial-analysis/polygon-triangulation/polygon-triangulation-search-cases.ts:234-264`
  (`getTilingDefect`: edge crossings are proper crossings only, and a polygon vertex counts only
  strictly inside a triangle).
- What is wrong. A polygon edge that starts at a polygon vertex on a triangle's edge and runs
  through the triangle to its opposite corner is seen by neither test, and an uncovered piece of
  equal area hides the rest.
- Input. Polygon `A(0,0) V(1,0) C(3,3) B(4,0) E(2,-3)` (area 10.5), triangles `ABC` and `VBE`
  (rows `[0,3,2] [1,3,4]`, areas 6 and 4.5). `ABC` covers `AVC`, which is outside the polygon,
  and `AVE` is uncovered. `getTilingDefect` returns `undefined`; the review checker reports
  "polygon edge 1-2 passes through triangle 0,3,2"; `checkTriangulation(...).boundaryMatches` of
  the monotone cases is false.
- In practice: the two checkers agreed on every result of this review (about 150,000 verdicts on
  GPU and package output), and the spec also compares the triangle count with the package's.
- Suggested fix. Add the open-triangle clip of each polygon edge (three linear inequalities in
  the edge parameter, exact in BigInt), or require `boundaryMatches` on the Earcut path as well.

### frc-geometry-06 (minor, CONFIRMED on the GPU): at a tolerance match the turn order is wrong when the float32 direction vectors round, also where the exact order is available

- Where: `src/analysis-core/ring-assembly/gpu-segment-ring-assembly.ts:904` (`outgoing` is
  `end - start` in float32) and `:938-961`. The account lists this under "Not verified".
- What is wrong. `orientSign` is exact for the vectors it is given, and the vectors are rounded
  differences when the two ends of a segment differ in magnitude. The sign of a cross product of
  two rounded vectors can differ from that of the exact vectors.
- Input (`vertexTolerance` 0.5, interior left). Sliver
  `(0.6779634356498718, 0.7087751030921936) (1919326.875, 1919326.875) (4798316, 4798316.5)`,
  triangle `(0.55857914686203, 0.7131208777427673) (2878989.75, 2878990.25) (0.55857914686203, 3838655.75)`,
  segments in ring order. The tips are 0.12 apart, the polygons 0.087 apart; `shapely.union_all`
  has two parts. Cross product of the sliver's closing segment with the triangle's first edge:
  exact -306,829.6 (a right turn), with the float32 vectors +239,915.75 (a left turn). Kernel: no
  ring, 6 open segments. Right: both triangles.
- Second form: both candidates start at one point and only the incoming segment ends 0.1 away
  (first triangle closed onto `(1.4174774885177612, 1.0765413045883179)`, both rings starting at
  `(1.319739818572998, 1.0615627765655518)`). The exact orientation of the two candidates about
  their common start exists; the kernel uses the rounded vectors and loses both rings.
- Counts: 240 scenes of this shape with directions about 1e-7 radians apart and coordinates from
  1 to 4.8 million: 115 fail and 125 pass on the GPU. A CPU model of the direction rule predicts
  235 of them; the other 5 are scenes of the second form that pass because the kernel's fully
  exact path makes the second choice. Among 752 valid scenes generated, 120 (16%) are wrong with
  rounded vectors under the model.
- Suggested fix. Add an exact sign of `cross(b - a, d - c)` for four float32 points to
  `analysis-core/kernels` (the integer machinery of `orientSign` with two origins) and use it in
  `ringExactDirectionClass` and in the order of two candidates. No regression: the old float32
  key used the same rounded vectors.

## 2. Checked and found correct

Coverage validity (`gpu-coverage-validity.ts:401-449`).

- The kernel is the GEOS rule: the neighbour is projected on the tested segment and the tested
  segment on the neighbour, each clipped; both lengths strictly greater than the gap width (the
  division-free form is equivalent); the swap by the nearer end; both distances at most the gap
  width; segments in input order, the tested one first. Shared-vertex pairs reach the gap test as
  in GEOS.
- Against Shapely on the GPU: 17,524 cases in 1,012 scenes (two seeds) that the fixture does not
  hold: unquantized float32 pairs at unit, kilometre and 500000 / 4500000 scale, 1 to 50 km
  edges beside 2 to 30 m edges, wedges sharing one vertex at angles from 1e-4 to 0.2 radians,
  polygons with holes, islands, duplicates and overlapping interiors, jittered triangulations
  with repeated vertices. Away from a threshold 4,074 of 4,074 cases are equal, among them 940
  scenes at gap width 0 and 1,300 hole cases. Gap widths placed 1% and 0.1% either side of a
  GEOS threshold: 5,380 of 5,380 equal.
- Exact ties: 414 cases on dyadic coordinates with the gap width one float32 step below, at and
  above the gap and the overlap length, near the origin, at -300 / 77 and at 500000 / 4500000:
  all equal.
- Size of the float32 band the documentation mentions: with the gap width within 1e-4 of a GEOS
  threshold 62 of 2,690 cases differ, within 1e-5 195, within 3e-7 435; nearly all are long
  edges beside short ones and wedges of 1e-4 radians. Inside the band one segment of a pair can
  be flagged without the other. The band is far below the float32 step of the coordinates.
- The default path: the only code that reads the gap width is behind `gapWidth > 0.0` and the
  query box, so it cannot have changed.
- `coverage-validity-gaps.py` and `ring-assembly-float32-choices.py` regenerate IDENTICAL with
  `scripts/run-oracle-fixture.mjs`. Expected values are Shapely's, not the kernel's formula.
  `docs/api/gpu-spatial-analysis/operators-validity.md` and the cross-reference row match the code.

Ring assembly, shell of a hole (`gpu-segment-ring-assembly.ts:1582-1632`).

- By reasoning: only a shell that contains the exact probe can become the choice, so an L-shaped
  or C-shaped shell whose box contains a sibling's hole is rejected by the probe, not by the box.
  Two shells that contain the probe are nested, and a nested shell cannot have the box of its
  outer shell without touching it at two points, which a face walk splits into two rings. The
  float32 area therefore never decides between two containing shells of a planar boundary.
- On the GPU: 48 generated scenes (1,328 rings, 676 holes, shells with notches holding siblings
  with their own holes, nesting several levels deep) in three tiles, one tile again at
  500000 / 4500000, and two scenes where a hole touches its shell and an island its lake at one
  vertex: 889 holes. Every hole has the shell of `shapely.build_area` through
  `GPUSegmentRingAssembly` in both ring orders and through `GPUPolygonOverlay` (`dissolve`).
  Through `GPUPolygonize` the faces of `shapely.polygonize` come back for the tiles (885 holes)
  and for the authors' 13 wedge and 3 band scenes (27 holes), which the authors had not run.
- The 120 control scenes of frc-geometry-06 (rounded and exact order agree) all give both rings.

Triangulation (`gpu-polygon-triangulation.ts`).

- The six changes read against `earcut` 3.2.4: the hit test, the order of two hit edges, the
  triangle membership in both branches, the angle order, the collinear tie and the slope order
  are the package's decisions in exact form. The zero-length diagonal change alone causes no
  difference on the regression inputs.
- Polygons whose rings do not touch (4,493; lattice, collinear runs, repeated vertices, both
  windings of shell and holes, 1,400 slivers under lattice shears with coordinates up to 6
  million): 4,491 tilings, 2 flagged (frc-geometry-04), 0 wrong. The old kernel on the same:
  71 wrong, 128 flagged.
- Old kernel correct, new kernel correct: 5,380 of 5,382, 11,920 of 11,929, 6,980 of 6,987
  (frc-geometry-03 for the rest); every polygon without touching rings that the old kernel tiled
  is still tiled.
- Frames: 5,950 polygons give the same indices on the integer lattice and on the 2 m lattice at
  500000 / 4500000. On the 1/32 m by 1/2 m grid 5,447 map exactly; one changes from a tiling to
  flagged, none becomes wrong.
- Batches: 5,950 polygons in chunks of 3,000 and reversed in chunks of 500 give the same `valid`
  and the same triangles for every polygon.
- `getTilingDefect` is exact (float64 filter with a sufficient bound, then BigInt) and tests
  overlap, crossing, containment and area, not only the count; see frc-geometry-05 for the gap.
- The authors' specs pass on the current tree: 7 files, 34 tests (coverage topology, its float32
  spec, ring-assembly float32 choices and exact turns, triangulation search, hole bridge,
  triangulation). `check-contributor-contract` reports no new finding.

## 3. Not checked

- The lens scenario `gpu-spatial-analysis.coverage-validity`, `modules/lenses`, `modules/kepler`,
  the website, and the lens records that store the kernel hash.
- Other GPUs and the software adapter. `yarn lint`, `yarn typecheck` and the full suites were not
  run.
- The shell rule and the turn rule through `GPUMakeValid`, `GPUIsobandRings`, `GPUCellSetOutline`
  and `GPUBufferSurface`; the longitude/latitude turn key; rings that cross.
- `GPUPolygonOverlay` on the authors' wedge and band scenes: the overlay snaps them to its own
  lattice (the bands come back with other vertices), so only lattice scenes were compared.
- `GPUPolygonize` on rings that touch: it returns the face between a shell and a hole touching it
  as one ring through the shared vertex where Shapely returns a shell and a hole. Not compared
  further.
- Why the monotone path passes the polygons of frc-geometry-02; the path was not read.
- The insertion tried for frc-geometry-01 against the pinned specs and the index-order spec.
- Rings above 79 vertices on the Earcut path, apart from 15 polygons of about 4,300 vertices.
  Holes that share an edge with another ring, and self-intersecting input.
- Notes without a finding: the fixture pins two turn scenes where the account speaks of five
  shifts; `projectOntoSegment` repeats `closestPointOnSegment` of
  `gpu-spatial-analysis/spatial-join/nearest-wgsl.ts`.

## Re-review (2026-11-11)

Re-review of the fixes described in "Closing pass, review fixes (2026-11-11)" of `core-numerics.md`
and `polygon-topology.md`. Same reviewer, same method: the accounts were read as claims, the
changed kernels were read, and the probe sets of the review were rerun on the test GPU from
`test/review-scratch-geometry/`, which is deleted again. Three kernels were compared: the current
tree, the kernels before these fixes (the reviewer's copy of the Earcut kernel, and the authors'
backup of the whole triangulation module, scratch `geometry-fixes/kernels/before`), and the old
Earcut kernel. Verdicts are those of the review's own exact checker; the authors' new
`getTilingDefect` (with `getBoundaryChainDefect`) was run beside it on about 250,000 results and
agreed on every one. New inputs (capacity, large polygons, ring fans, polar turn scenes), logs
and the probe specs are in the scratch folder named at the top, the specs under `probes2/`.

Summary. All six findings are closed on the inputs of the review and on new ones. No result with
`valid = 1` was wrong in any configuration. The argument the authors give for that (every step
keeps the triangle boundaries plus the working ring equal to the polygon boundary, every ear is
counter-clockwise by an exact test) was read and holds. Nothing that triangulated before is
flagged now. One new minor finding, older than the fixes, in the ring assembly.

### Status of the findings

| Finding | Status | Evidence on the current kernels |
| --- | --- | --- |
| frc-geometry-01 | closed | The 21,425 touching-ring polygons of the review: 21,425 tilings on the Earcut path and with the default (were 7 wrong, 98 flagged). Also tilings: the 11,001 noded polygons, 2,000 new polygons with up to 12 holes, 58 capacity polygons (below), the 22 hand cases. At 1/32 m by 1/2 m: 5,446 of 5,447 tilings and 1 flagged; on the 2 m lattice 5,950 tilings with the lattice indices. Both review inputs tile (11 and 14 triangles) |
| frc-geometry-02 | closed | Monotone path alone: 13,644 valid results on the review's sets and the new ones, 0 wrong. Against the authors' backup of the module on 23,918 polygons: the 19 wrong results became flagged, and no other verdict and no triangle of a valid result changed. The 15-vertex input is flagged. The polygon of 4,306 vertices and the other 30 enlarged or small inputs are tilings with the default. Default and `'none'` give the same verdict on all 48,445 polygons run with both |
| frc-geometry-03 | closed | Both inputs tile (16 and 11 triangles). No polygon that the first-pass kernel tiled is flagged or wrong: 0 of 42,843 on lattices, 0 of 5,420 at float32 resolution. Old kernel correct and current correct: 5,382 of 5,382 on the first set |
| frc-geometry-04 | closed | The input and its variants tile on the Earcut path. Rings apart: 8,461 polygons (the 4,493 of the review and 3,968 of a new seed), all tilings with both methods; the 8,459 that the first-pass kernel tiled have bit-identical indices. Residual, stated by the authors: one touching-ring polygon is flagged at 1/32 m by 1/2 m (`middleInside`, `gpu-polygon-triangulation.ts:896`). It is index 2192 of `tri.json`; 1988 is its number among the 5,447 polygons that map to that grid |
| frc-geometry-05 | closed | The counterexample is in `polygon-triangulation-search-cases.node.spec.ts` and is reported. The boundary-chain test was read (line name with offset, position along the line, exact integers) and is complete for counter-clockwise triangles. 0 disagreements with the review checker on about 250,000 GPU and package results, wrong ones among them |
| frc-geometry-06 | closed | The 240 scenes: 0 without both rings planar with the interior left, mirrored with the interior right, and as longitude/latitude (were 115). 240 new scenes with the tips at (179.99, 89.4): 0 planar and 0 as longitude/latitude with the interior left. See frc-geometry-07 for their mirror images |

### New finding

#### frc-geometry-07 (minor, CONFIRMED on the GPU; older than the fixes): with the interior on the right an exact u-turn is the last choice, with the interior on the left the first

- Where: `src/analysis-core/ring-assembly/gpu-segment-ring-assembly.ts:82-96` and `:202-216`
  (a u-turn is class 2 for either side), `:1050`, `:1075` and `:1104` (`SIDE_ORDER` reverses the
  class order for `'right'`), `:1112` (the fallback key has the u-turn at +2 and is multiplied by
  `SIDE`). The exact path at a shared vertex had this before the fixes; the fixes carried it to
  the tolerance path and to longitude/latitude input.
- What is wrong. The face rule takes the first edge met when turning from the reversed incoming
  direction toward the interior side, so an edge that exactly reverses the incoming one comes
  first for either side. The kernel orders turns in (-pi, pi] and takes the largest for `'left'`
  and the smallest for `'right'`, which puts the u-turn first for `'left'` and last for
  `'right'`. A scene and its mirror image then give different rings.
- Input on the lattice (every candidate starts on the vertex, `vertexTolerance` 2^-8): a square
  with a zero-width spike, segments `(0,0)-(10,0)`, `(10,0)-(20,0)`, `(20,0)-(10,0)`,
  `(10,0)-(10,10)`, `(10,10)-(0,10)`, `(0,10)-(0,0)`. Interior left: 2 rings (the square, and
  the spike as a ring of two segments), 0 open segments. The mirror image (x negated) with the
  interior right: 1 ring, 2 open segments, `CONFLICT` on the segment that returns from the spike.
  The same segments reversed with the interior right: 1 ring, 2 open; their mirror image with the
  interior left: 2 rings.
- Input at a tolerance match: 3 of the 240 polar scenes have a closing segment exactly
  antiparallel to the ring's own first edge (cross product 0 from the four endpoints). As
  generated (interior left) both rings come back; mirrored with the interior right no ring comes
  back and all 6 segments are open, planar and as longitude/latitude. One of them, mirrored:
  `(-179.98995971679688, 89.40003204345703)-(-153.85028076171875, 63.26068115234375)`,
  `(-153.85028076171875, 63.26068115234375)-(-114.64082336425781, 24.051616668701172)`,
  `(-114.64082336425781, 24.051616668701172)-(-179.99002075195312, 89.39999389648438)`,
  `(-179.98995971679688, 89.40003204345703)-(-140.78050231933594, 50.19096755981445)`,
  `(-140.78050231933594, 50.19096755981445)-(-179.98995971679688, 37.12042236328125)`,
  `(-179.98995971679688, 37.12042236328125)-(-179.98995971679688, 89.40003204345703)`,
  `vertexTolerance` 2^-13.
- Reach. `'right'` is used by `GPUCellSetOutline` for Quadbin, whose outlines have no
  antiparallel candidate at a corner, and by direct callers. It needs a candidate exactly
  opposite to the incoming segment that is neither its paired twin nor cancelled.
- Suggested fix. Make the two sides mirror images: for `'right'` give the u-turn the class before
  the right turns (or classify with the cross signs negated and keep the `'left'` order), and
  the same in `ringTurnKey`. Pin the spike and its mirror image.

### New ground

- Capacity of the inserted vertices. Read: a hole takes at most three nodes now (one inserted,
  two for the bridge) against two before and the reserve grew by one per hole, so the spare for
  splits is what it was; `createNode` still fails the polygon at the limit, inside its own slice.
  Run: 58 polygons with 1 to 600 holes whose leftmost vertex lies inside an edge (300 on one
  floor edge, on the ceiling, on a wall, on a slanted edge, 600 on two edges, 300 pairs sharing
  their leftmost vertex inside an edge, a chain of 300 each touching the one before, 20 at one
  point, and the same touch points as shell vertices): 58 tilings with and without the z-order
  hash, with the default, and at float32 resolution. One batch of 66,054 such polygons (957,780
  vertices, 231,188 holes, 3,599,932 scratch nodes): 66,054 tilings with the hash and the default,
  and without the hash on the Earcut path. Not run at the limit: on the test device the binding
  size allows 153,391,689 nodes.
- Flagged sets. First-pass kernel valid and current kernel flagged: none (row frc-geometry-03).
  Six invalid polygons (crossing shell, hole outside, hole across a shell edge, NaN, infinity,
  crossing holes) are `valid = 0` with every method. Four rings with a local twist give the same
  flags before and after (Earcut's cure did not act on them), and coordinates of 1e-38 beside 1e6
  do too.
- The 88,281 polygons flagged by the monotone path are for that path forced. With the default a
  polygon below 4,096 vertices goes to Earcut first, and the flagged set of the monotone path
  grew only by the results that were wrong. Large polygons with the default, before and after
  alike: rings apart tile at 5,007, 20,007 and 60,007 vertices, and so do a hole on a shell
  vertex and a hole vertex inside an edge of another hole; two holes that share a vertex tile at
  5,007 and are `valid = 0` at 20,007 and 60,007 (the monotone path flags them and Earcut runs out
  of `maximumWork`). That limit is not new and is flagged; it deserves a sentence in `docs/api`.
- Batches. 118 polygons placed between the capacity polygons, and 11 between invalid polygons,
  have the triangles they have when run without them. 5,950 polygons in reverse order in chunks
  of 500: no difference.
- Ring assembly, several candidates. 400 scenes of 3 to 6 rings at one vertex (1,799 rings), with
  exactly opposite and exactly equal directions among them: the rings with every ring moved by a
  lattice step inside the tolerance equal the rings with the vertex shared exactly, in all four
  of planar and longitude/latitude with the interior left and right, the latter placed at
  longitudes 179.94 to 180.0 and latitude 89.5. Planar and longitude/latitude results are equal.

### Specs

- The 46 headless spec files the authors list as users of the two modules: 508 tests pass. The
  two new GPU specs: 10 tests pass. Node: the 29 listed files, 319 tests, and the two module
  folders, 8 files and 63 tests, pass. No failure in a file of another session.
- `polygon-triangulation-touching-rings.py` and `ring-assembly-float32-choices.py` regenerate
  IDENTICAL. `check-contributor-contract` and `check-upstream-first` report nothing new.

### Not checked

- A layout at the scratch limit of the device. Polygons above 60,007 vertices. Other GPUs.
- The authors' CPU model of the Earcut kernel and their 126,382 count as such; the counts above
  are on the review's sets (about 43,000 polygons, 5,447 of them again at float32 resolution, and
  the batch of 66,054) and agree with theirs where they overlap.
- Lens records, `modules/lenses`, `modules/kepler`, the website. `yarn lint` and the typechecks.
