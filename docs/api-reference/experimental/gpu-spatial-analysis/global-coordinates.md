# Global coordinates, spherical geometry and S2 analysis

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
final residual. Default tolerance is `1e-5` degrees and the iteration cap is 16. A combined inverse
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

## General spherical topology

`GPUSphericalPolygonContains` accepts nonconvex simple CCW shell/hole rings and a query-to-polygon
index. The first ring is the shell and subsequent rings are holes. Rings must be disjoint, holes
properly nested, and minor arcs nondegenerate. Boundary policy and status 0/1/2 follow the convex
predicate. Each ring must lie in the open hemisphere defined by `hemisphereCenter`, or its
normalized vertex mean when omitted. Supplying a pole avoids sensitivity to uneven vertex
sampling. The operator checks ring self-crossings; cross-ring nesting is a prepared-input contract.
Direct S2 checks include 1,032 normal/complement classifications on concave and holed rings rotated
across the pole and antimeridian.

`GPUSphericalPolygonOverlay` maps caller-prepared polygon/multipolygon boundaries into a common
gnomonic hemisphere, runs the existing GPU snap-rounded arrangement, then lifts assembled rings
onto the sphere. All four Boolean operations and left/right complements are supported. Every
vertex must meet `dot(chartCenter, vertex) >= minimumCosine` (default `.01`, minimum `.001`).
`vertexTolerance` is in chart coordinates. Complement output is a GPU word: 1 means the sphere
minus the assembled region. Boundary provenance endpoints remain in chart coordinates. Counts,
required counts, overflow and candidate overflow retain the planar arrangement contract. Projection
or capacity failures cannot be interpreted as an empty successful region. The direct S2 oracle
checks nonconvex/holed Boolean ring counts and areas to `3e-5` steradians.

`GPUSphericalArcOverlay` handles boundaries that cannot share a hemisphere. It nodes great-circle
arcs intrinsically, classifies adjacent faces by crossing parity, retains oriented boundary pieces,
and assembles rings using the left-turn rule at junctions. Both operands supply the known membership
of one common `referencePoint`, separated from all boundaries; reversing that membership represents
the complement. Nested rings use parity, and an empty boundary with inside reference represents
the full sphere. The reference membership is caller-prepared semantic data, not inferred from ring
orientation. Simple disjoint boundary rings are checked for self/cross-ring intersections; touching
rings within one operand are rejected. Output rings are open xyz rings with interior on the left.

Intrinsic overlay accepts at most 128 total input vertices and 3–1024 output vertex capacity.
Tolerance ranges from `1e-7` to `1e-3` radians, default `1e-6`. Seven deterministic two-arc parity
paths avoid reference rays through vertices. Adjacent-side probes start at 16 times tolerance and
shrink according to boundary separation; inadequate separation and unresolved junctions fail
explicitly. Counts are `(written vertices, written rings, required segments, reference membership)`.
Status bits are 1 invalid input/reference/topology, 2 capacity and 4 numerical uncertainty. Failed
results have zero written geometry; required segments are only a lower bound on early failure.
Segment storage is an explicit graph transient, not a large shader-private array. Twelve direct
S2 Boolean cases cover globally distributed zigzag boundaries and complements, with area error
below `2e-5` steradians and exact ring counts. This is tolerance-based topology, not exact S2 parity.

## Adaptive polygon covers

`GPUS2PolygonCovering` traverses the six S2 roots using a bounded depth-first stack. It emits whole
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
The direct `s2geometry==0.14.0` oracle enumerates all 6,144 level-5 cells for each of two concave/holed
regions: emitted interior cells are S2-contained, exterior covers include every S2 `MayIntersect` cell,
and summed interior/exterior cell areas bracket the region area. This is the precise tested
cover guarantee on these two exhaustive fixtures; it does not certify every possible thin or
near-degenerate boundary. Capacity and visit-budget failures are separately exercised.

The additional reproducible generators are `difficult-geodesics.py`, `geodesic-area-adaptive.py`,
`datum-horizontal.py`, `spherical-polygon-contains.py`, `spherical-polygon-overlay.py`,
`spherical-arc-overlay.py` and `s2-polygon-covering.py` under `modules/graph/test/oracles/`.
They pin GeographicLib 2.1, pyproj 3.7.2 / PROJ 9.5.1 and S2Geometry 0.14.0. Horizontal PROJ
fixtures write prioritized CTable2 grids and decode their actual f32-radian samples; split outputs
must agree within `1e-9` degrees, including priority discontinuities and inclusive bounds.

## Evidence and limits

`modules/graph/test/oracles/global-coordinates.py` pins `s2sphere==0.2.5` and
`geographiclib==2.1`, `pyproj==3.7.2` and `shapely==2.1.2`. The committed JSON records direct upstream calls and a clearly labeled composed overlap oracle. PROJ samples a generated local GTX vertical grid. For spherical crossings, Shapely computes topology in a gnomonic plane (great circles become lines) and GeographicLib independently measures the lifted intersection on a unit sphere. S2 neighbor checks
include every cell at levels 0–2 and sampled levels 15/30, including face seams and poles. The tests
compare measured area/perimeter with S2 values and check six-face area sums against `4*pi`.
Additional independent tests cover affine-grid interpolation/inversion, first-grid priority, signed
height units, nodata, nonfinite arithmetic, inverse exhaustion, known octant area, overlap fractions,
open/closed/complement boundaries, capacity limits and unsupported near-antipodal geodesics.
Node tests parse generated WGSL, audit reserved symbols and check binding counts. GPU tests are
in `modules/graph/test/gpu-spatial-analysis/global-coordinates/global-coordinates.spec.ts`.

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
`modules/graph/test/gpu-spatial-analysis/global-coordinates/global-coordinates-benchmark.spec.ts`.
Run it in the isolated browser benchmark project with
`LUMA_TEST_BROWSER_BENCHMARKS=true VITE_GLOBAL_COORDINATES_BENCHMARK=true`, and set optional
`VITE_BENCHMARK_REVISION` / `VITE_BENCHMARK_SOURCE_STATE` metadata to identify the tested tree.
The root runner selects the benchmark through the normal Vitest command.

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
The [raw phase samples and metadata](../../../../dev-docs/benchmarks/data/global-coordinates-2026-10-23.json)
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

Unit-sphere area and perimeter of the six S2 root faces.

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

{/* lens:GPUS2PolygonCovering */}

::lens{scenario=gpu-spatial-analysis.s2-polygon-covering rung=result caption="Inspect captured geometry, counts and failure flags."}

::lens{scenario=gpu-spatial-analysis.s2-polygon-covering rung=graph caption="Inspect the executed GPU stages."}

### `GPUSphericalPolygonContains`

{/* lens:GPUSphericalPolygonContains */}

::lens{scenario=gpu-spatial-analysis.spherical-polygon-contains rung=result caption="Inspect captured geometry, counts and failure flags."}

::lens{scenario=gpu-spatial-analysis.spherical-polygon-contains rung=graph caption="Inspect the executed GPU stages."}

### `GPUSphericalPolygonOverlay`

{/* lens:GPUSphericalPolygonOverlay */}

::lens{scenario=gpu-spatial-analysis.spherical-polygon-overlay rung=result caption="Inspect captured geometry, counts and failure flags."}

::lens{scenario=gpu-spatial-analysis.spherical-polygon-overlay rung=graph caption="Inspect the executed GPU stages."}

### `GPUSphericalArcOverlay`

{/* lens:GPUSphericalArcOverlay */}

::lens{scenario=gpu-spatial-analysis.spherical-arc-overlay rung=result caption="Inspect captured geometry, counts and failure flags."}

::lens{scenario=gpu-spatial-analysis.spherical-arc-overlay rung=graph caption="Inspect the executed GPU stages."}

## Expanded-mode scaling observations

The expanded harness completed fourteen workloads on the same Apple `metal-3` adapter and
Headless Chrome 151 at 10:42 PM EDT on October 23, 2026. The [complete phase records](../../../../dev-docs/benchmarks/data/global-coordinates-expanded-2026-10-23.json)
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
