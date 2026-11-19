var e={slug:`api/gpu-spatial-analysis/global-coordinates`,title:`Global coordinates, spherical geometry and S2 analysis`,html:`<h1 id="global-coordinates-spherical-geometry-and-s2-analysis">Global coordinates, spherical geometry and S2 analysis</h1>
<p>These graph contributors cover bounded portions of CR01–CR05. Inputs and outputs remain GPU
views. They declare compute nodes and never submit, read back, download grids, or call a CPU GIS
engine. Use a separate contributor for each source batch; the kernels do not merge chunks or own
caller buffers. All output views must be disjoint from inputs and each other. Storage usage stays
within eight bindings per compute stage.</p>
<table>
<thead>
<tr>
<th>Roadmap</th>
<th>Delivered API and scope</th>
<th>Remaining scope</th>
</tr>
</thead>
<tbody>
<tr>
<td>CR01</td>
<td>Existing <code>GPUGeometryMeasures</code> now offers safeguarded inverse solving, adaptive edge integration and explicit invalid/exhaustion reporting on serial/cooperative paths.</td>
<td>f32 numerical envelopes are tested, not certified; singular or unresolved inverse roots and pole-winding polygon areas remain explicitly invalid.</td>
</tr>
<tr>
<td>CR02</td>
<td>Prepared regular grids, priority/bounds/nodata, inverse residual validation and optional high/low coordinate words match direct PROJ fixtures.</td>
<td>Grid files, acquisition, CRS databases and epoch/subgrid selection remain caller preparation, as scoped by the original backlog.</td>
</tr>
<tr>
<td>CR03</td>
<td>Signed vertical/combined corrections, explicit units/reference names and split height residuals; direct PROJ vertical fixture.</td>
<td>Named geoid/epoch acquisition and full PROJ pipeline construction are external preparation.</td>
</tr>
<tr>
<td>CR04</td>
<td>Nonconvex containment with holes, chart and intrinsic global Boolean overlay, complement semantics, open/closed predicates, pole/antimeridian fixtures and assembled rings.</td>
<td>f32 tolerance bands replace exact adaptive predicates; intrinsic overlay reports unresolved separation/junctions and bounds work to 128 input vertices / 1024 output vertices.</td>
</tr>
<tr>
<td>CR05</td>
<td>Exact hierarchy/neighbors and stable measures through level 30, adaptive polygon covers through level 15, uniform cap covers and geometric cross-family overlap weights for prepared convex boundaries.</td>
<td>Cover guarantees are tolerance-aware; automatic cross-family broad-phase/boundary acquisition and optimized CPU timing comparisons remain separate integration/performance work.</td>
</tr>
</tbody>
</table>
<h2 id="prepared-datum-grids">Prepared datum grids</h2>
<p><code>GPUDatumGridTransform</code> takes <code>float32x3</code> positions: longitude and latitude in degrees, then
height in the declared unit. Grid samples are <code>(longitude shift degrees, latitude shift degrees, height shift meters)</code>. Signed height samples always mean <strong>target minus source</strong> height. A geoid
undulation grid must therefore be sign-adjusted by its preparation adapter when converting
ellipsoidal to orthometric height. Both vertical reference names are mandatory for vertical or
combined operations. The names describe provenance; the contributor does not resolve a CRS.</p>
<p>Each grid supplies dimensions, sample offset, origin and nonzero spacing; negative spacing is
allowed. Coordinates use a continuous longitude branch and there is no implicit antimeridian wrap.
Grid bounds are inclusive with a <code>1e-5</code> grid-coordinate numerical margin, clamped to the outer edge. The last grid row/column uses the final interpolation cell. All four
corners must be valid and finite, even when a corner has interpolation weight zero. The first
covering grid wins. Nodata in it fails instead of silently falling through to a lower-priority grid.
An empty grid list returns no-coverage for every finite input.</p>
<p>Forward horizontal transforms add interpolated shifts. Inverse horizontal transforms solve
<code>target = source + shift(source)</code> with at most <code>maximumIterations</code> fixed-point steps and verify the
final residual. The inverse always takes at least one step before testing the residual (the first residual is the shift itself). Default tolerance is <code>5e-9</code> degrees (about 0.56 mm; never less than four f32 spacings of the shift) and the iteration cap is 16. A combined inverse
samples vertical correction at the recovered source longitude/latitude. Inverse lookup starts at
the target: if that location is outside every grid, it returns no coverage, even if a shifted source
might exist. This matches the target-seeded direct PROJ <code>hgridshift</code> fixtures, including the case where a forward-shifted edge point cannot be inversely looked up. Priority discontinuities can also choose a different inverse grid; inverse is not promised to recover the original source across such a discontinuity. No extrapolation or identity fallback occurs.</p>
<p>Status is 0 success, 1 invalid/nonfinite result or latitude outside ±90°, 2 no coverage, 4 nodata,
and 8 inverse exhaustion. Failed rows contain NaNs. Optional paired <code>positionsLow</code> input/output
views represent longitude, latitude and height as the sum of high and low f32 words. The returned
high words stay equal to the input high words; the returned low words carry displacement, and are
not a normalized double-double representation. Grid origins are split before local interpolation.
This preserves tested sub-ulp datum shifts without claiming general float64 arithmetic. With no
low views, small shifts at large coordinates can still round away. Samples remain f32, and inverse
tolerance is a residual stopping test, not a certified geodetic accuracy bound. See the <a href="https://proj.org/en/stable/operations/transformations/hgridshift.html">PROJ horizontal grid semantics</a>
and <a href="https://proj.org/en/stable/operations/transformations/vgridshift.html">vertical grid semantics</a>
for the broader reference contracts; this prepared-grid interface does not implement every option.</p>
<h2 id="convex-spherical-geometry-and-overlap-weights">Convex spherical geometry and overlap weights</h2>
<p>Provide normalized xyz directions and counterclockwise minor great-circle edges. Every polygon
must contain 3–32 distinct vertices, exclude a repeated closing vertex, and lie strictly inside an
open hemisphere. The vertex-mean direction must lie strictly inside every oriented edge halfspace.
Degenerate, antipodal-edge, clockwise or unsupported nonconvex inputs report invalidity. Input xyz
squared lengths must differ from one by no more than <code>1e-4</code>. Default predicate tolerance is <code>1e-6</code>.
The tolerance is an f32 angular band, not exact arithmetic.</p>
<p>Containment uses a query-to-polygon index column. <code>boundary: 'closed'</code> includes the tolerance band;
<code>'open'</code> excludes it. <code>complement: true</code> reverses interior/exterior while retaining the chosen
boundary policy. Status 2 exposes boundary classification, and status 1 invalidity takes precedence
and must not be interpreted as a meaningful false predicate.</p>
<p>Intersection uses paired polygon-offset columns. It collects contained vertices and true
minor-arc crossings, deduplicates within four times tolerance, and orders the resulting boundary.
Closed-boundary contacts may have one or two vertices. Outputs have a caller-selected stride of
1–64 and <code>(written, required)</code> counts. Status 1 means invalid input, 2 capacity exhaustion, and 4
numerical uncertainty; status bits may combine. Uncertain/invalid rows write zero vertices.
Capacity-limited output is incomplete and must not be consumed as a polygon.</p>
<p>Optional <code>output.measures</code> supplies <code>(intersection steradians, source fraction, target fraction, union steradians)</code>. These fractions are geometric overlap weights for <strong>already prepared</strong> convex
source/target cell boundaries, irrespective of cell family. They are not a relabeling of H3/S2 keys
or centroid-containment approximation. Failed or capacity-limited rows have NaN measures. No
cross-family candidate generation, H3 boundary acquisition or global conservative-remapping
recipe is implied.</p>
<h2 id="s2-identity-measurement-and-covers">S2 identity, measurement and covers</h2>
<p>Cell keys are always exact little-endian <code>(low, high)</code> u32 pairs. <code>GPUS2CellHierarchy</code> supports
parents at a requested level, immediate children in Hilbert order, inclusive leaf-ID ranges, and
four edge neighbors ordered left/right/down/up in the input face coordinate system. Integer cube
face transitions work through level 30, including corners, poles and mixed-level rows. Invalid
keys report status 1; asking a leaf for children or a coarser cell for a descendant parent reports
status 2. All corresponding output keys are zero.</p>
<p><code>GPUS2CellMeasures</code> returns area in steradians and perimeter in radians. Multiply by a chosen
sphere's squared radius or radius explicitly for physical units. Levels 0–30 use analytic quadratic-UV increments to compute triangle determinants and edge
cross products without subtracting rounded fine-cell vertices. The exact integer key is decoded
before this local computation. This improves area/perimeter precision; it does not improve the
absolute xyz precision of arbitrary boundary vertices or polygon-cover predicates.</p>
<p><code>GPUS2CapCovering</code> accepts unit xyz cap centers and angular radii in <code>[0, pi]</code>. It enumerates all
<code>6 * 4^level</code> cells, with level restricted to 0–8, in ascending key order. Exterior covers select
cells whose closest spherical point is within the cap plus a default <code>1e-5</code>-radian safety margin.
Interior covers require every point to be within the cap minus that margin, using antipodal closest
distance for the farthest point. Consequently an interior cover of a radius-pi cap conservatively
excludes cells containing the antipodal center. Capacity truncation reports status 2 and the full
required count; such output is not a complete cover. Invalid caps report status 1. This uniform
algorithm is appropriate for modest cap batches; it is not S2's adaptive priority coverer.</p>
<p><code>refinement: 'level-synchronous'</code> removes the level 8 ceiling. Instead of one invocation per cap
scanning the whole sphere, it starts from the six face cells and runs <code>level + 1</code> rounds. Each
round launches one invocation per frontier cell and per cap, drops cells that cannot contain a
selected descendant (a child is never closer to the cap than its parent), splits the others in
four, and a scan numbers the survivors, so every cap's cells stay in ascending key order with no
sort. The cells, counts and status equal the uniform enumeration's (exterior and interior, any
capacity) and <code>level</code> may reach 15. Each cap holds at most <code>frontierCapacity</code> cells per round
(default <code>8 * capacity + 64</code>, at least <code>capacity</code>): a round holds the four children of every kept
parent, and a kept parent can have no selected child, so the frontier can reach about four times
the final cell count. The default holds a cover that exceeds <code>capacity</code> by a factor of about two,
so the caller still gets status 2 and the required count. A cap that outgrows the frontier reports
status bit 4, writes no cells and reports counts <code>(0, 0)</code>: a truncated frontier cannot know the
size of the full cover, so no required count is available; raise <code>frontierCapacity</code> (or
<code>capacity</code>) and run again. A zero-radius cap on a face center covers the four cells that touch
it in both enumerations. The rounds are not gated on the live frontier length (the published
<code>GPUScan</code> has no GPU-written dispatch size), and <code>estimatedCommandNodeCount</code> is exact for both the
cap and the polygon level-synchronous coverings. The default remains <code>refinement: 'uniform'</code>, the reference tier.</p>
<h2 id="safeguarded-ellipsoidal-measurements">Safeguarded ellipsoidal measurements</h2>
<p>Set <code>ellipsoidalInverse: 'safeguarded'</code> on the existing <code>GPUGeometryMeasures</code>. Ordinary converged
Vincenty rows keep their fast path. Failed rows scan 32 auxiliary-longitude intervals near the
geographic separation and bisect sign-changing residual brackets for up to 32 iterations. A root
must have residual at most <code>7e-7</code> radians; the shortest accepted positive-distance solution wins.
A singular/discontinuous or unbracketed root remains invalid. This is an independently implemented
safeguard, not the full Karney inverse algorithm. The default remains <code>'vincenty'</code>.</p>
<p><code>geodesicAreaTolerance</code> requests per-edge successive-refinement quadrature error estimation in
squared output units. <code>geodesicAreaMaximumSegments</code> is a power of two from 2 through 256 (default
256). Sampling doubles until <code>abs(fine-coarse)/15</code> meets the requested estimate; exhaustion or
nonfinite results invalidates the feature and increments <code>invalidCount</code>. This estimates quadrature
convergence, not all f32, inverse/direct-series or input-quantization error. Default omitted tolerance
retains the prior bounded-sampling behavior. Pole-winding polygon areas still report invalidity;
the spherical operators below handle spherical polar regions explicitly.</p>
<p>Direct <code>geographiclib==2.1</code> fixtures cover 128 near-antipodal and 128 global line pairs (exact f32
input coordinates), on serial and cooperative paths. Valid results must be within 10 meters; fewer
than eight of the 256 may be explicitly unresolved. Twelve holed polygons near high latitudes,
the equator and antimeridian compare adaptive area with <code>Geodesic.WGS84.Polygon</code>: absolute error
below <code>max(2e6 square meters, 3e-6 * reference area)</code>. An intentionally impossible <code>1e-20</code> tolerance
with a two-segment cap must invalidate all twelve. These are measured acceptance envelopes, not
user-requested tolerance guarantees.</p>
<h2 id="spatial-context-limits-and-status-codes">Spatial context, limits and status codes</h2>
<p>The five spherical classes and the two S2 coverings read unit direction vectors. Each takes an
optional <code>spatialContext</code>, resolved with <code>resolveGPUSpatialContext</code> and exposed as
<code>.spatialContext</code>. The accepted space is <code>longitude-latitude</code>, the space whose positions these
vectors embed on the sphere, with great-circle edges and the <code>none</code> or <code>great-circle</code> metric.
Omitted, it defaults to <code>GPU_SPHERICAL_DEFAULT_SPATIAL_CONTEXT</code>. A planar context throws
<code>&#x3C;id> does not support planar coordinates</code>, and a longitude-latitude context with linear edges
throws <code>&#x3C;id> needs great-circle edges</code>. The context declares meaning only: the vectors are never
converted, and <code>GPULonLatPolygonPreparation</code> remains the way from degrees to unit vectors.</p>
<p>Validation messages name the contributor, the prop and the accepted range, for example
<code>spherical-arc-overlay left plus right positions must be at most 128</code>. The limits are named
exports with their reasons:</p>
<table>
<thead>
<tr>
<th>Constant</th>
<th>Value</th>
<th>Reason</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>GPU_SPHERICAL_ARC_OVERLAY_MAXIMUM_INPUT_VERTICES</code></td>
<td>128</td>
<td>O(n^3) arrangement in one invocation with private scratch</td>
</tr>
<tr>
<td><code>GPU_SPHERICAL_ARC_OVERLAY_MAXIMUM_OUTPUT_VERTICES</code></td>
<td>1024</td>
<td>seven floats of scratch per output vertex</td>
</tr>
<tr>
<td><code>GPU_SPHERICAL_CONVEX_MINIMUM_VERTICES</code>, <code>GPU_SPHERICAL_CONVEX_MAXIMUM_VERTICES</code></td>
<td>3, 32</td>
<td>convex polygons live in fixed private arrays</td>
</tr>
<tr>
<td><code>GPU_SPHERICAL_CONVEX_INTERSECTION_MAXIMUM_CAPACITY</code></td>
<td>64</td>
<td>a clip has at most the sum of two 32-vertex polygons</td>
</tr>
<tr>
<td><code>GPU_SPHERICAL_MINIMUM_TOLERANCE</code>, <code>GPU_SPHERICAL_MAXIMUM_TOLERANCE</code></td>
<td>1e-7, 1e-2</td>
<td>below f32 noise, and the meaning of a boundary band</td>
</tr>
<tr>
<td><code>GPU_SPHERICAL_MAXIMUM_ARRANGEMENT_TOLERANCE</code></td>
<td>1e-3</td>
<td>arc overlay and S2 polygon covering</td>
</tr>
<tr>
<td><code>GPU_SPHERICAL_OVERLAY_MINIMUM_COSINE</code></td>
<td>1e-3</td>
<td>gnomonic chart scale near the horizon</td>
</tr>
<tr>
<td><code>GPU_S2_POLYGON_COVERING_MAXIMUM_LEVEL</code></td>
<td>15</td>
<td>f32 vertex resolution</td>
</tr>
<tr>
<td><code>GPU_S2_POLYGON_COVERING_MAXIMUM_VISITED_CELLS</code></td>
<td>1,048,576</td>
<td>frontier buffers fit one 128 MiB binding</td>
</tr>
<tr>
<td><code>GPU_S2_CAP_COVERING_MAXIMUM_UNIFORM_LEVEL</code></td>
<td>8</td>
<td><code>6 * 4^level</code> cells scanned per cap invocation</td>
</tr>
<tr>
<td><code>GPU_S2_CAP_COVERING_MAXIMUM_REFINED_LEVEL</code></td>
<td>15</td>
<td>f32 vertex resolution</td>
</tr>
<tr>
<td><code>GPU_CELL_TOPOLOGY_MAXIMUM_RADIUS</code></td>
<td>8</td>
<td><code>GPUCellTopology</code> disk and ring <code>k</code> (H3 disk of 217 cells, Quadbin 289)</td>
</tr>
</tbody>
</table>
<p>Status words are documented on each output type and exported as constants, with the values
unchanged: <code>GPU_SPHERICAL_CONVEX_CONTAINS_STATUS_*</code>, <code>GPU_SPHERICAL_POLYGON_CONTAINS_STATUS_*</code>,
<code>GPU_SPHERICAL_CONVEX_INTERSECTION_STATUS_*</code>, <code>GPU_SPHERICAL_ARC_OVERLAY_STATUS_*</code>,
<code>GPU_SPHERICAL_POLYGON_OVERLAY_COMPLEMENT_*</code>, <code>GPU_S2_POLYGON_COVERING_STATUS_*</code>,
<code>GPU_S2_CAP_COVERING_STATUS_*</code>, <code>GPU_S2_CELL_HIERARCHY_STATUS_*</code> and <code>GPU_S2_CELL_MEASURES_STATUS_*</code>.</p>
<h2 id="general-spherical-topology">General spherical topology</h2>
<p><code>GPUSphericalPolygonContains</code> accepts nonconvex simple CCW shell/hole rings and a query-to-polygon
index. The first ring is the shell and subsequent rings are holes. Rings must be disjoint, holes
properly nested, and minor arcs nondegenerate. Boundary policy and status 0/1/2 follow the convex
predicate. Each ring must lie in the open hemisphere defined by <code>hemisphereCenter</code>, or its
normalized vertex mean when omitted. Supplying a pole avoids sensitivity to uneven vertex
sampling. The operator checks ring self-crossings; cross-ring nesting is a prepared-input contract.
Direct S2 checks include 1,032 normal/complement classifications on concave and holed rings rotated
across the pole and antimeridian.</p>
<p><code>GPUSphericalPolygonOverlay</code> maps caller-prepared polygon/multipolygon boundaries into a common
gnomonic hemisphere, runs the existing GPU snap-rounded arrangement, then lifts assembled rings
onto the sphere. All four Boolean operations and left/right complements are supported. Every
vertex must meet <code>dot(chartCenter, vertex) >= minimumCosine</code> (default <code>.01</code>, minimum <code>.001</code>).
<code>vertexTolerance</code> is in chart coordinates. Complement output is a GPU word: 1 means the sphere
minus the assembled region. Boundary provenance endpoints remain in chart coordinates. Counts,
required counts, overflow and candidate overflow retain the planar arrangement contract. Projection
or capacity failures cannot be interpreted as an empty successful region. The direct S2 oracle
checks nonconvex/holed Boolean ring counts and areas to <code>3e-5</code> steradians.</p>
<p><code>GPUSphericalArcOverlay</code> handles boundaries that cannot share a hemisphere. It nodes great-circle
arcs intrinsically, classifies adjacent faces by crossing parity, retains oriented boundary pieces,
and assembles rings using the left-turn rule at junctions. Both operands supply the known membership
of one common <code>referencePoint</code>, separated from all boundaries; reversing that membership represents
the complement. Nested rings use parity, and an empty boundary with inside reference represents
the full sphere. The reference membership is caller-prepared semantic data, not inferred from ring
orientation. Simple disjoint boundary rings are checked for self/cross-ring intersections; touching
rings within one operand are rejected. Output rings are open xyz rings with interior on the left.</p>
<p>Intrinsic overlay accepts at most 128 total input vertices and 3–1024 output vertex capacity.
Tolerance ranges from <code>1e-7</code> to <code>1e-3</code> radians, default <code>1e-6</code>. Seven deterministic two-arc parity
paths avoid reference rays through vertices. Adjacent-side probes start at 16 times tolerance and
shrink according to boundary separation; inadequate separation and unresolved junctions fail
explicitly. Counts are <code>(written vertices, written rings, required segments, reference membership)</code>.
Status bits are 1 invalid input/reference/topology, 2 capacity and 4 numerical uncertainty. Failed
results have zero written geometry; required segments are only a lower bound on early failure.
Segment storage is an explicit graph transient, not a large shader-private array. Twelve direct
S2 Boolean cases cover globally distributed zigzag boundaries and complements, with area error
below <code>2e-5</code> steradians and exact ring counts. This is tolerance-based topology, not exact S2 parity.</p>
<h2 id="adaptive-polygon-covers">Adaptive polygon covers</h2>
<p><code>GPUS2PolygonCovering</code> traverses the six S2 roots using a bounded depth-first stack. It emits whole
contained cells early and subdivides boundary cells through <code>maximumLevel</code> (0–15), subject to
<code>minimumLevel</code>. Interior mode emits only contained cells; exterior mode includes intersecting
terminal cells. Inputs use the same prepared shell/hole hemisphere contract and optional pole as
the nonconvex predicate, with explicit complement. Self-crossing rings and nonadjacent touches
are rejected before traversal, producing invalid status, zero counts and zero output keys.
Cross-ring nesting remains a prepared-input contract. This is mixed-level geometric coverage, not
H3 centroid containment or S2's priority-queue <code>max_cells</code> minimization policy.</p>
<p><code>maximumVisitedCells</code> defaults to 65,536 and may range from 6 to 1,048,576. Status bits are 1 invalid
input, 2 output capacity and 4 traversal exhaustion. Counts are <code>(written, required so far)</code>;
when exhausted they do not claim the complete requirement. <code>visitedCount</code> exposes traversal work.</p>
<p>The default <code>refinement: 'serial'</code> runs the ring check and the depth-first walk in one invocation,
which is exact but cannot use the GPU's width. <code>refinement: 'level-synchronous'</code> distributes both.
Ring validation becomes one invocation per ring (shape, unit vertices, hemisphere center) and one
per edge (zero length, hemisphere, crossings and touches against the ring's later edges). The
traversal then runs <code>maximumLevel + 1</code> rounds over one ordered frontier list: each round
classifies every cell of the round's level in parallel (drop, keep, or split into four children),
a scan numbers the survivors and a write builds the next list, so the cover is already in
ascending cell-ID order and needs no sort. For every input that completes, the cells, counts,
status bits 1 and 2 and <code>visitedCount</code> equal the serial traversal's, which the recorded fixtures
check on more than 100 cases. The difference is exhaustion: the serial traversal stops after
exactly <code>maximumVisitedCells</code> pops and keeps the cells emitted so far, while the level-synchronous
traversal fails the whole round, writes no cells and reports counts <code>(0, 0)</code> with status bit 4
(<code>visitedCount</code> is capped at <code>maximumVisitedCells</code>). <code>maximumVisitedCells</code> also sizes the two
ping-pong frontier buffers (8 bytes per cell). The serial traversal stays the default because its
partial results are pinned.
The direct <code>s2geometry==0.14.0</code> oracle enumerates all 6,144 level-5 cells for each of two concave/holed
regions: emitted interior cells are S2-contained, exterior covers include every S2 <code>MayIntersect</code> cell,
and summed interior/exterior cell areas bracket the region area. This is the precise tested
cover guarantee on these two exhaustive fixtures; it does not certify every possible thin or
near-degenerate boundary. Capacity and visit-budget failures are separately exercised.</p>
<p>The additional reproducible generators are <code>difficult-geodesics.py</code>, <code>geodesic-area-adaptive.py</code>,
<code>datum-horizontal.py</code>, <code>spherical-polygon-contains.py</code>, <code>spherical-polygon-overlay.py</code>,
<code>spherical-arc-overlay.py</code> and <code>s2-polygon-covering.py</code> under <code>modules/analysis/test/oracles/</code>.
They pin GeographicLib 2.1, pyproj 3.7.2 / PROJ 9.5.1 and S2Geometry 0.14.0. Horizontal PROJ
fixtures write prioritized CTable2 grids and decode their actual f32-radian samples; split outputs
must agree within <code>1e-9</code> degrees, including priority discontinuities and inclusive bounds.</p>
<h2 id="evidence-and-limits">Evidence and limits</h2>
<p><code>modules/analysis/test/oracles/global-coordinates.py</code> pins <code>s2sphere==0.2.5</code> and
<code>geographiclib==2.1</code>, <code>pyproj==3.7.2</code> and <code>shapely==2.1.2</code>. The committed JSON records direct upstream calls and a clearly labeled composed overlap oracle. PROJ samples a generated local GTX vertical grid. For spherical crossings, Shapely computes topology in a gnomonic plane (great circles become lines) and GeographicLib independently measures the lifted intersection on a unit sphere. S2 neighbor checks
include every cell at levels 0–2 and sampled levels 15/30, including face seams and poles. The tests
compare measured area/perimeter with S2 values and check six-face area sums against <code>4*pi</code>.
Additional independent tests cover affine-grid interpolation/inversion, first-grid priority, signed
height units, nodata, nonfinite arithmetic, inverse exhaustion, known octant area, overlap fractions,
open/closed/complement boundaries, capacity limits and unsupported near-antipodal geodesics.
Node tests parse generated WGSL, audit reserved symbols and check binding counts. GPU tests are
in <code>modules/analysis/test/gpu-spatial-analysis/global-coordinates/global-coordinates.spec.ts</code>.</p>
<p>These fixtures establish the tested numerical cases, not complete S2/PROJ/GeographicLib parity.
The bounded observations below add same-device timing and explicit-buffer evidence. Cross-device accuracy, driver memory peaks and representative geographic distributions remain unmeasured.</p>
<h2 id="numerical-limits-established-by-the-committed-tests">Numerical limits established by the committed tests</h2>
<p>The tested limits below are acceptance thresholds, not certified worst-case error bounds or
promises for every supported input:</p>
<table>
<thead>
<tr>
<th>Operation</th>
<th>Tested threshold and scope</th>
</tr>
</thead>
<tbody>
<tr>
<td>S2 hierarchy/neighbors</td>
<td>Exact u32 words: all 126 cells at levels 0–2, plus 20 sampled cells each at levels 15 and 30 (166 oracle rows total).</td>
</tr>
<tr>
<td>S2 area/perimeter</td>
<td>Relative error below <code>2e-5</code> across all 166 direct S2 oracle cells, including 20 level-15 and 20 level-30 samples.</td>
</tr>
<tr>
<td>Ellipsoidal line distance</td>
<td>Absolute error below 5 meters on the three pinned regular, polar and antimeridian pairs. The pinned near-antipodal pair reports invalidity in serial/cooperative modes. This is not an antipodal accuracy envelope.</td>
</tr>
<tr>
<td>Spherical overlap</td>
<td>Eight-vertex crossing topology agrees with the gnomonic oracle; area and normalized overlap fractions use five-decimal-place assertions.</td>
</tr>
<tr>
<td>Datum grids</td>
<td>Affine bilinear forward values use five-decimal-place assertions; inverse coordinates use four and recovered height uses three. Nodata, out-of-coverage, exhausted iterations, nonfinite arithmetic and pole-crossing invalidity are checked separately.</td>
</tr>
</tbody>
</table>
<p>Spherical polygon predicate tolerance can be set from <code>1e-7</code> to <code>1e-2</code> (default <code>1e-6</code>), and
intersection deduplication uses four times this tolerance. Polygons whose center-to-edge separation
is at or below tolerance are rejected as unsupported/degenerate; this avoids pretending that f32
can distinguish arbitrarily thin regions. The cap-cover margin ranges from <code>1e-6</code> to <code>1e-2</code>
radians (default <code>1e-5</code>). It deliberately expands exterior covers and shrinks interior covers.</p>
<h2 id="opt-in-bounded-performance-observation">Opt-in bounded performance observation</h2>
<p>The benchmark spec is
<code>modules/analysis/test/gpu-spatial-analysis/global-coordinates/global-coordinates-benchmark.spec.ts</code>.
Run <code>yarn test-benchmarks modules/analysis/test/gpu-spatial-analysis/global-coordinates/global-coordinates-benchmark.spec.ts</code>.
The launcher selects benchmark files and enables their workload flags. Set optional
<code>VITE_BENCHMARK_REVISION</code> / <code>VITE_BENCHMARK_SOURCE_STATE</code> metadata to identify the tested tree.</p>
<p>Eight deterministic workloads compare small and larger batches: 166/16,384 S2 neighbor rows
cycling through the exact-key corpus, 256/16,384 datum positions, 1/32 level-4 spherical caps
(1,536 candidate cells per cap), and 1/256 four-vertex convex overlap pairs. These are repeated
reference shapes, not geographic distribution or scalability benchmarks. Each workload changes
resident inputs after the repeated measurements and validates the changed output.</p>
<p>JSON records separate buffer allocation/setup, upload, synchronous graph compilation, first
encode/submit/completion, five warm repeated runs, update upload and three updated runs. Readback
and reference checks are timed separately. GPU timestamps are reported only when available.
Memory reports explicit input/output bytes and graph transient allocation; driver peak memory
is unavailable. The device is shared and pipeline caches are not cleared, so first execution is
not guaranteed to be a cold-driver measurement. No throughput advantage or latency threshold is
asserted. Observed timings must identify the device/browser and source state; no performance
result is implied merely by adding this opt-in spec.</p>
<h2 id="recorded-bounded-observations">Recorded bounded observations</h2>
<p>An Apple WebGPU adapter (<code>metal-3</code>, non-fallback, chip model unspecified) running Headless Chrome151
completed all eight benchmark workloads on October 23, 2026 at 9:07 PM EDT.
The raw phase samples and metadata
retain upload, readback, validation, first/repeated/updated execution and allocation measurements.</p>
<p><strong>Provenance limitation:</strong> the runner recorded <code>sourceRevision: null</code> and <code>sourceState: null</code>.
The source was a shared dirty working tree. These are local observations, not a frozen release
benchmark. Device caches were shared; compilation can be deferred into first execution, and no
cold-driver guarantee applies. Timestamp queries were supported but no GPU timestamp durations
were recorded, so every execution number below includes CPU encoding, submission and queue wait.</p>
<table>
<thead>
<tr>
<th>Workload</th>
<th align="right">Host compile ms</th>
<th align="right">First execution ms</th>
<th align="right">Warm median ms (5 runs)</th>
<th align="right">Updated median ms (3 runs)</th>
<th align="right">Explicit input + output bytes</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>s2-neighbors-166</code></td>
<td align="right">2.9</td>
<td align="right">18.6</td>
<td align="right">0.5</td>
<td align="right">0.8</td>
<td align="right">7,304</td>
</tr>
<tr>
<td><code>s2-neighbors-16384</code></td>
<td align="right">0.2</td>
<td align="right">5.2</td>
<td align="right">0.8</td>
<td align="right">1.4</td>
<td align="right">720,896</td>
</tr>
<tr>
<td><code>datum-grid-256</code></td>
<td align="right">0.3</td>
<td align="right">3.5</td>
<td align="right">0.5</td>
<td align="right">0.6</td>
<td align="right">7,232</td>
</tr>
<tr>
<td><code>datum-grid-16384</code></td>
<td align="right">0.2</td>
<td align="right">2.8</td>
<td align="right">0.6</td>
<td align="right">0.3</td>
<td align="right">458,816</td>
</tr>
<tr>
<td><code>s2-cap-cover-1</code></td>
<td align="right">0.6</td>
<td align="right">102.5</td>
<td align="right">9.1</td>
<td align="right">7.5</td>
<td align="right">12,316</td>
</tr>
<tr>
<td><code>s2-cap-cover-32</code></td>
<td align="right">0.2</td>
<td align="right">49.4</td>
<td align="right">7.1</td>
<td align="right">7.0</td>
<td align="right">394,112</td>
</tr>
<tr>
<td><code>spherical-overlap-1</code></td>
<td align="right">0.1</td>
<td align="right">104.6</td>
<td align="right">0.4</td>
<td align="right">0.9</td>
<td align="right">332</td>
</tr>
<tr>
<td><code>spherical-overlap-256</code></td>
<td align="right">0.2</td>
<td align="right">110.6</td>
<td align="right">0.5</td>
<td align="right">0.5</td>
<td align="right">82,952</td>
</tr>
</tbody>
</table>
<p>All initial, repeated and updated outputs passed their pinned/analytic reference checks. Each
case reported zero graph-owned transient allocation; this excludes shader-private temporary
storage and driver allocation. The input/output byte totals include output capacity, including
unused padded slots. Sub-millisecond differences and the cap batch ordering should not be read
as scaling trends from these single shared-device runs. There is no optimized CPU timing baseline.</p>
<p>The level-4 cap oracle exposed a real cancellation defect during benchmark development: subtractive
projection onto a nearly perpendicular great-circle plane could leave an f32 residue that
normalized to the query direction. The corrected double-cross projection is now guarded by a
regular GPU regression against the pinned exact cell list. The successful records above come from
the corrected implementation; the earlier failed run is not represented as successful evidence.</p>
<h2 id="executable-unit-sphere-examples">Executable unit-sphere examples</h2>
<p>These lazy examples execute actual GPU graphs on synthetic analytic inputs and capture the input views, result views and status flags. They are not benchmark runs or observed geographic datasets.</p>
<h3 id="gpus2cellhierarchy"><code>GPUS2CellHierarchy</code></h3>
<p>Exact children of the six S2 root faces.</p>
<p>{/* lens:GPUS2CellHierarchy */}</p>
<p>::lens{scenario=gpu-spatial-analysis.s2-cell-hierarchy rung=result caption="Inspect captured input, result and status rows."}</p>
<p>::lens{scenario=gpu-spatial-analysis.s2-cell-hierarchy rung=graph caption="Inspect the executed GPU command graph."}</p>
<h3 id="gpus2cellmeasures"><code>GPUS2CellMeasures</code></h3>
<p>Unit-sphere area and perimeter of S2 cells at levels 0–30. This compatibility API retains
steradians/radians and its per-row status (0 valid, 1 invalid with NaN measures). It shares the
stable local-increment arithmetic of <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-cells#gpucellgridpath-and-gpucellmeasures"><code>GPUCellMeasures</code></a>.
Use <code>GPUCellMeasures</code> with <code>family: 's2'</code> for meters/kilometers, ordered edges and the common
H3/S2/Quadbin interface; invalid rows there are zero. Both APIs accept the compile-time
<code>precision: 'fp64'</code> option for published double-single projection and local-increment arithmetic;
the default is <code>'float32'</code>. Output formats and the unit-sphere API's invalid statuses stay the same.</p>
<p>{/* lens:GPUS2CellMeasures */}</p>
<p>::lens{scenario=gpu-spatial-analysis.s2-cell-measures rung=result caption="Inspect captured input, result and status rows."}</p>
<p>::lens{scenario=gpu-spatial-analysis.s2-cell-measures rung=graph caption="Inspect the executed GPU command graph."}</p>
<h3 id="gpus2capcovering"><code>GPUS2CapCovering</code></h3>
<p>Exterior root-face covering of a small north-pole cap.</p>
<p>{/* lens:GPUS2CapCovering */}</p>
<p>::lens{scenario=gpu-spatial-analysis.s2-cap-covering rung=result caption="Inspect captured input, result and status rows."}</p>
<p>::lens{scenario=gpu-spatial-analysis.s2-cap-covering rung=graph caption="Inspect the executed GPU command graph."}</p>
<h3 id="gpusphericalconvexcontains"><code>GPUSphericalConvexContains</code></h3>
<p>Interior, exterior and closed-boundary queries on an octant.</p>
<p>{/* lens:GPUSphericalConvexContains */}</p>
<p>::lens{scenario=gpu-spatial-analysis.spherical-convex-contains rung=result caption="Inspect captured input, result and status rows."}</p>
<p>::lens{scenario=gpu-spatial-analysis.spherical-convex-contains rung=graph caption="Inspect the executed GPU command graph."}</p>
<h3 id="gpusphericalconvexintersection"><code>GPUSphericalConvexIntersection</code></h3>
<p>Identical octants with intersection area and overlap fractions.</p>
<p>{/* lens:GPUSphericalConvexIntersection */}</p>
<p>::lens{scenario=gpu-spatial-analysis.spherical-convex-intersection rung=result caption="Inspect captured input, result and status rows."}</p>
<p>::lens{scenario=gpu-spatial-analysis.spherical-convex-intersection rung=graph caption="Inspect the executed GPU command graph."}</p>
<h3 id="gpus2polygoncovering"><code>GPUS2PolygonCovering</code></h3>
<p>{/* lens:GPUS2PolygonCovering */}</p>
<p>::lens{scenario=gpu-spatial-analysis.s2-polygon-covering rung=result caption="Inspect captured geometry, counts and failure flags."}</p>
<p>::lens{scenario=gpu-spatial-analysis.s2-polygon-covering rung=graph caption="Inspect the executed GPU stages."}</p>
<h3 id="gpusphericalpolygoncontains"><code>GPUSphericalPolygonContains</code></h3>
<p>{/* lens:GPUSphericalPolygonContains */}</p>
<p>::lens{scenario=gpu-spatial-analysis.spherical-polygon-contains rung=result caption="Inspect captured geometry, counts and failure flags."}</p>
<p>::lens{scenario=gpu-spatial-analysis.spherical-polygon-contains rung=graph caption="Inspect the executed GPU stages."}</p>
<h3 id="gpusphericalpolygonoverlay"><code>GPUSphericalPolygonOverlay</code></h3>
<p>{/* lens:GPUSphericalPolygonOverlay */}</p>
<p>::lens{scenario=gpu-spatial-analysis.spherical-polygon-overlay rung=result caption="Inspect captured geometry, counts and failure flags."}</p>
<p>::lens{scenario=gpu-spatial-analysis.spherical-polygon-overlay rung=graph caption="Inspect the executed GPU stages."}</p>
<h3 id="gpusphericalarcoverlay"><code>GPUSphericalArcOverlay</code></h3>
<p>{/* lens:GPUSphericalArcOverlay */}</p>
<p>::lens{scenario=gpu-spatial-analysis.spherical-arc-overlay rung=result caption="Inspect captured geometry, counts and failure flags."}</p>
<p>::lens{scenario=gpu-spatial-analysis.spherical-arc-overlay rung=graph caption="Inspect the executed GPU stages."}</p>
<h2 id="expanded-mode-scaling-observations">Expanded-mode scaling observations</h2>
<p>The expanded harness completed fourteen workloads on the same Apple <code>metal-3</code> adapter and
Headless Chrome 151 at 10:42 PM EDT on October 23, 2026. The complete phase records
include the original eight cases and six added cases below. Source state is explicitly
<code>shared-dirty-tree</code>; revision is null, so this is not a frozen-source benchmark. These are CPU
encode/submit/queue-completion measurements, with upload, validation and readback reported
separately. Five warm executions and three updated executions share caches and device state.
No optimized CPU baseline, speedup claim, cross-device result or driver peak measurement exists.</p>
<table>
<thead>
<tr>
<th>Workload</th>
<th align="right">Host compile ms</th>
<th align="right">First execution ms</th>
<th align="right">Warm median ms</th>
<th align="right">Updated median ms</th>
<th align="right">Explicit input + output bytes</th>
<th align="right">Graph transient bytes</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>s2-stable-measures-166</code></td>
<td align="right">0.3</td>
<td align="right">36.7</td>
<td align="right">0.2</td>
<td align="right">0.2</td>
<td align="right">3,320</td>
<td align="right">0</td>
</tr>
<tr>
<td><code>s2-stable-measures-16384</code></td>
<td align="right">0.1</td>
<td align="right">29.4</td>
<td align="right">0.4</td>
<td align="right">1.1</td>
<td align="right">327,680</td>
<td align="right">0</td>
</tr>
<tr>
<td><code>s2-adaptive-cover-level4</code></td>
<td align="right">0.2</td>
<td align="right">178.1</td>
<td align="right">73.3</td>
<td align="right">69.0</td>
<td align="right">81,980</td>
<td align="right">0</td>
</tr>
<tr>
<td><code>s2-adaptive-cover-level6</code></td>
<td align="right">0.4</td>
<td align="right">372.7</td>
<td align="right">309.1</td>
<td align="right">302.9</td>
<td align="right">81,980</td>
<td align="right">0</td>
</tr>
<tr>
<td><code>spherical-intrinsic-overlay-8</code></td>
<td align="right">0.5</td>
<td align="right">232.9</td>
<td align="right">3.2</td>
<td align="right">3.3</td>
<td align="right">3,464</td>
<td align="right">7,184</td>
</tr>
<tr>
<td><code>spherical-intrinsic-overlay-32</code></td>
<td align="right">0.2</td>
<td align="right">241.7</td>
<td align="right">22.2</td>
<td align="right">22.1</td>
<td align="right">3,752</td>
<td align="right">7,184</td>
</tr>
</tbody>
</table>
<p>Measure batches repeat the 166 direct-S2 oracle cells, including level 30, then reverse resident
key order. Adaptive covers rotate a spherical octant and include a downstream measure stage for
all 4,096 capacity slots; checks require complete counts, sub-enumeration traversal work and an
exterior area at least the octant area. Intrinsic overlays compare the same globally distributed
S2-reference intersection with 8 versus 32 input vertices, then rotate both operands. Additional
vertices sample the original minor arcs, so the region and oracle area remain unchanged.</p>
<p>These observations show the cost of the bounded traversal and intrinsic arrangement as resolution
or vertex count increases; they do not establish large-region scalability. In particular the
level-6 cover costs about 309 ms in this fixture and the 32-vertex arrangement about 22 ms. Output
capacity bytes include unused slots, and graph transient bytes exclude shader-private and driver
allocation. Every initial, repeated and updated result passed its numerical checks.</p>`};export{e as default};