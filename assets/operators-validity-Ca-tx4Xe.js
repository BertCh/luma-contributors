var e={slug:`api/gpu-spatial-analysis/operators-validity`,title:`GPU Spatial Analysis: Geometry validity & intersections`,html:`<h1 id="geometry-validity--intersections">Geometry validity &#x26; intersections</h1>
<h3 id="gpusegmentintersection"><code>GPUSegmentIntersection</code></h3>
<p>Finds every intersecting pair of line segments within one geometry, with the kind and point of each intersection.</p>
<p>::lens{scenario=gpu-spatial-analysis.segment-intersection rung=result caption="1,615 vertices of 721 street edges; points mark where segments meet or cross, coloured by kind."}</p>
<p>Maximum: each side has at most <code>getGPUSegmentIntersectionMaximumVertexCount(device)</code> (4194304 on a default device: the 32-byte segment-table row per vertex must fit one storage binding) <code>positions</code> rows; construction throws <code>segment-intersection left.positions rows must be at most 4194304</code> above it.</p>
<p>Planar only: <code>spatialContext</code> defaults to native planar coordinates and a longitude-latitude context throws; project to a planar frame first (<code>createGPULocalFrame</code>, <code>GPULocalFrameProjection</code>). Orientation signs are exact.</p>
<p>Finds every intersecting pair of line segments between two geometries (<code>left</code>, <code>right</code>) or within one
(<code>left</code> only), like turf <code>lineIntersect</code> and <code>kinks</code>, self-intersection tests and noding points. Each side
is a tagged line or polygon geometry port (<code>kind: 'lines'</code> or <code>'polygons'</code>, one kind per side). A segment is identified
by the index of its start vertex. Polygon rings close implicitly; rings with fewer than three vertices,
zero-length segments and segments with non-finite coordinates are skipped.</p>
<table>
<thead>
<tr>
<th>Prop</th>
<th>Meaning</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>left</code>, <code>right?</code></td>
<td>Geometries. Omit <code>right</code> for self mode (<code>left &#x3C; right</code>, adjacent segments skipped)</td>
</tr>
<tr>
<td><code>sameFeatureOnly?</code></td>
<td>Self mode: only same-feature pairs (per-feature self-intersection and ring crossings)</td>
</tr>
<tr>
<td><code>output.pairs</code></td>
<td><code>{leftRows, rightRows, status: {count, overflow, candidateOverflow?, requiredCount?}}</code>, sorted by <code>(left, right)</code>; capacity is <code>leftRows.length</code></td>
</tr>
<tr>
<td><code>output.kinds?</code></td>
<td><code>GPU_SEGMENT_INTERSECTION_KIND</code>: proper 1, touch 2, collinearTouch 3, overlap 4, uncertain 5</td>
</tr>
<tr>
<td><code>output.points?</code>, <code>output.endPoints?</code></td>
<td>The intersection point; for an overlap, the two ends of the shared span</td>
</tr>
<tr>
<td><code>output.leftFeatures?</code>, <code>output.rightFeatures?</code>, <code>output.leftRings?</code>, <code>output.rightRings?</code></td>
<td>Feature and ring (or line) rows of each segment</td>
</tr>
<tr>
<td><code>output.status.uncertainCount?</code></td>
<td>Count of uncertain pairs within capacity</td>
</tr>
<tr>
<td><code>spatialSort?</code></td>
<td>Morton sort before the BVH build; on by default from 1024 vertices; the output is identical</td>
</tr>
<tr>
<td><code>leafCapacity?</code></td>
<td>Power-of-two BVH leaves, by default the next power of two of the right vertex count</td>
</tr>
</tbody>
</table>
<p><code>proper</code> means the interiors cross at one point. <code>touch</code> is one shared point where an endpoint of one lies
on the other (non-parallel), <code>collinearTouch</code> is collinear with one shared endpoint, and <code>overlap</code> is
collinear sharing positive length. The pipeline is a segment table, a <code>GPUBVH</code> over the right segments, a
counted and scanned probe, exact classification and a bounded sorted output; nothing is read back.
On overflow the output keeps a sorted prefix (<code>status.count = min(total, capacity)</code>, <code>status.overflow = 1</code>, <code>status.requiredCount</code>
unclamped), and the columns beyond <code>status.count</code> are unspecified. <code>status.uncertainCount</code> and the classification
columns cover pairs within capacity only. A collinear chain of <code>n</code> segments lists all <code>n^2</code> overlap pairs.
<code>output.pairs.status.overflow</code> describes final pair capacity only. BVH leaf loss is independent,
reported in <code>output.pairs.status.candidateOverflow</code>. Both supplied words are rewritten on each
encoding; <code>requiredCount</code> is a lower bound when BVH leaves were lost. On clean
reuse both flags are zero. If <code>leafCapacity</code> is below the indexed geometry's vertex-slot
count, <code>candidateOverflow</code> is required: actual segment counts are GPU data, so construction
cannot guarantee those slots suffice. Nested consumers that omit the diagnostic must use
enough BVH slots or construction refuses that configuration. <code>GPUPairGather</code> preserves both causes.
Geometry descriptors, options, pair/column maps and nested status destinations are captured at
construction. Removing or replacing a caller-map field later cannot change the validated bindings;
the contents of the retained buffers remain dynamic.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUSegmentIntersection</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  left: {kind: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'lines'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, positions, lineOffsets},</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {pairs: {leftRows, rightRows, status: {count, overflow}}, kinds, points}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span></code></pre>
<h4 id="how-it-computes">How it computes</h4>
<p>{/* lens:GPUSegmentIntersection */}</p>
<p>::lens{scenario=gpu-spatial-analysis.segment-intersection rung=datum datum=0 caption="Inspect pair 0, the first intersecting segment pair in the sorted output."}</p>
<p>::lens{scenario=gpu-spatial-analysis.segment-intersection rung=kernel caption="Read the classification kernel built on exact orientation tests."}</p>
<p>::lens{scenario=gpu-spatial-analysis.segment-intersection rung=graph caption="Follow the segment table, BVH build, counted probe and classification."}</p>
<h3 id="gpugeometryvalidity"><code>GPUGeometryValidity</code></h3>
<p>Point and line inputs are single-part feature ports. Multipart point/line <code>featureOffsets</code> are
rejected before output row validation; use feature-aware joins or measures when grouping is needed.</p>
<p>Computes per-feature diagnostic issue codes and optional nonzero-usable validity for polygons, lines or points.</p>
<p>::lens{scenario=gpu-spatial-analysis.geometry-validity rung=result caption="102 park polygons shaded by their issue codes; the orientation convention is ignored."}</p>
<p>Maximum: <code>polygons.positions</code> has at most <code>getGPUSegmentIntersectionMaximumVertexCount(device)</code> (4194304 on a default device: the 32-byte segment-table row per vertex must fit one storage binding) rows.</p>
<p>Planar only: <code>spatialContext</code> defaults to native planar coordinates and a longitude-latitude context throws; project to a planar frame first (<code>createGPULocalFrame</code>, <code>GPULocalFrameProjection</code>). Orientation signs are exact.</p>
<p>A per-feature validity bitmask for polygons and multipolygons (the four-level <code>GPUPolygonGeometryPort</code> layout), like
PostGIS <code>ST_IsValidDetail</code> and Shapely <code>explain_validity</code>. <code>output.issueCode</code> holds diagnostic bits;
<code>decodeGPUGeometryValidityIssueCode</code> returns named flags and <code>hasStructuralIssue</code>, which describes
observed evidence only. A zero issue code cannot certify validity when checks are incomplete.
Optional <code>output.validity</code> is 1 when the implemented structural checks pass, otherwise 0. Orientation
and repeated-vertex advisories leave it 1. Polygon intersection overflow suppresses every validity row
because the missing intersections cannot be assigned to particular features. This flag does not
certify the unsupported checks listed below. <strong>Breaking (2026-11-17):</strong> rename <code>output.mask</code> to
<code>output.issueCode</code>; feed <code>output.validity</code> to downstream input masks.
Diagnostic and status output ranges must not overlap one another or retained geometry inputs.
Supplied <code>status.overflow</code> and <code>status.intersectionCount</code> scalars are reset to zero on every
point/line submission because those kinds do not perform intersection checks. Kepler marks
zero-issue rows <code>incomplete</code> when polygon intersection overflow persists after retries. The
explorer counts published <code>validity</code> rows, including advisory-only rows, and withholds its valid
polygon count while checks are incomplete.
The website hover and showcase map use the published usability rows too: an unobserved structural
defect is not a pass. With zero issue code and usability zero they show an incomplete check;
observed structural defects remain visible. The showcase withholds “all valid” on overflow.
The website preserves issue bit 256 as <code>uncertain</code> in hover text and uses a separate “uncertain
predicate” class. It does not label an uncertifiable predicate as a repeated vertex.
The geometry-validity result view explicitly selects the <code>resultParks</code> polygon capture; full
results retain <code>overflow</code> and <code>intersectionCount</code> as separate scalar diagnostics below the map.</p>
<p>The bits of <code>GPU_GEOMETRY_VALIDITY_BIT</code> are <code>nonFinite</code> 1, <code>unclosedRing</code> 2 (explicit closure only),
<code>shortRing</code> 4 (fewer than 3 vertices once a closing duplicate is dropped; rows are counted, so <code>A, B, B, A</code> does not set it and fails through <code>selfIntersection</code> instead), <code>repeatedVertex</code> 8, <code>selfIntersection</code> 16, <code>crossingRings</code>
32 (a proper crossing or overlap between different rings of one feature; point touches are allowed),
<code>holeOutsideShell</code> 64, <code>badOrientation</code> 128 and <code>uncertain</code> 256. <code>GPU_GEOMETRY_VALIDITY_STRUCTURAL_MASK</code> is
every bit except <code>badOrientation</code> (a convention) and <code>repeatedVertex</code> (GEOS and OGC accept repeated
vertices, so it is reported but does not make a geometry structurally invalid).</p>
<table>
<thead>
<tr>
<th>Prop</th>
<th>Meaning</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>polygons</code>, <code>output.issueCode</code>, <code>output.validity?</code></td>
<td>The input, diagnostic code and optional nonzero-usable flag (<code>featureCount</code> rows each)</td>
</tr>
<tr>
<td><code>intersectionCapacity</code></td>
<td>Capacity of the internal same-feature intersection list; valid data produces almost none</td>
</tr>
<tr>
<td><code>output.status.overflow</code>, <code>output.status.intersectionCount?</code></td>
<td>The overflow flag (the self-intersection and crossing bits may then be incomplete) and the unclamped count</td>
</tr>
<tr>
<td><code>ringClosure?</code></td>
<td><code>'implicit'</code> (default) or <code>'explicit'</code>, which sets <code>unclosedRing</code></td>
</tr>
<tr>
<td><code>orientation?</code></td>
<td><code>'counter-clockwise-shell'</code> (default), <code>'clockwise-shell'</code> or <code>'ignore'</code></td>
</tr>
</tbody>
</table>
<p><code>lines</code> (<code>{kind: 'lines', positions, lineOffsets}</code>) and <code>points</code> replace <code>polygons</code> for the other kinds
(Shapely <code>is_valid</code>, <code>is_valid_reason</code> on lines and points). Lines report <code>nonFinite</code> and <code>tooFewPoints</code>,
points report <code>nonFinite</code>; neither needs an intersection list.</p>
<p>Not checked: interior connectivity (holes touching at two points), nested holes, nested polygons of a
multipolygon, and rings that leave the shell only through boundary vertices. Repeated vertices are reported
although OGC accepts them. Hole containment tests one off-boundary vertex per hole.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUGeometryValidity</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  polygons,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {issueCode, validity, status: {overflow}},</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  intersectionCapacity: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">1024</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// validity can be used as the next contributor's input mask.</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// Inspect issueCode with decodeGPUGeometryValidityIssueCode; inspect overflow for completeness.</span></span></code></pre>
<h4 id="how-it-computes-1">How it computes</h4>
<p>{/* lens:GPUGeometryValidity */}</p>
<p>::lens{scenario=gpu-spatial-analysis.geometry-validity rung=datum datum=0 caption="Inspect park 0, Lincoln (Abraham) Park, and its issue code."}</p>
<p>::lens{scenario=gpu-spatial-analysis.geometry-validity rung=kernel caption="Read the per-ring checks that set diagnostic issue bits."}</p>
<p>::lens{scenario=gpu-spatial-analysis.geometry-validity rung=graph caption="Follow the segment table, BVH, same-feature intersections and issue-code assembly."}</p>
<h3 id="gpugeometrypredicates"><code>GPUGeometryPredicates</code></h3>
<p>Evaluates per-feature predicates and counts such as simplicity, vertex count and hole count for points, lines or polygons.
<code>geometry</code> and <code>other</code> use <code>GPUGeometryPredicatesGeometry</code>: one point or one linestring per row,
or the polygon/multipolygon hierarchy. Multipart point/line <code>featureOffsets</code> are rejected at
construction, including ports supplied through spreads or casts; this contributor does not yet
reduce those parts into feature-level predicate/count columns.</p>
<p>::lens{scenario=gpu-spatial-analysis.geometry-predicates rung=result caption="102 park polygons shaded by stored vertex count; simplicity and hole counts are captured alongside."}</p>
<p>Maximum: with <code>isSimple</code> or <code>isRing</code>, <code>geometry.positions</code> has at most <code>getGPUSegmentIntersectionMaximumVertexCount(device)</code> (4194304 on a default device: the 32-byte segment-table row per vertex must fit one storage binding) rows.</p>
<p>Planar only: <code>spatialContext</code> defaults to native planar coordinates and a longitude-latitude context throws; project to a planar frame first (<code>createGPULocalFrame</code>, <code>GPULocalFrameProjection</code>).</p>
<p>Per-feature columns for points, linestrings or polygons (Shapely <code>is_simple</code>, <code>is_ring</code>, <code>is_closed</code>,
<code>is_ccw</code>, <code>get_num_points</code>, <code>count_coordinates</code>, <code>get_num_interior_rings</code>, <code>get_num_geometries</code>,
<code>extract_unique_points</code>, <code>equals_exact</code>, <code>equals_identical</code>). Each output is optional and <code>uint32</code> (0 or 1
for booleans), all in the <code>output</code> bag: <code>isSimple</code>, <code>isRing</code>, <code>isClosed</code>, <code>isCcw</code>, <code>numPoints</code>,
<code>numCoordinates</code>, <code>numInteriorRings</code> and <code>numGeometries</code>.</p>
<ul>
<li><code>isSimple</code> and <code>isRing</code> use <code>GPUSegmentIntersection</code> in self mode, so they need <code>intersectionCapacity</code>
and <code>output.status.overflow</code>; features after an overflow can be wrong. A polygon is simple when no ring meets itself
(Shapely 2.1). <code>isCcw</code> follows the GEOS <code>Orientation::isCCW</code> rule (the JTS 1.19 flat-top algorithm) for
linestrings and is 0 for other kinds, as in Shapely.</li>
<li><code>other</code> (same kind and feature count) enables <code>equalsExact</code> (per-frame <code>tolerance</code> through
<code>getGPUGeometryPredicatesParameterValues</code>) and <code>equalsIdentical</code>, which compare feature <code>i</code> with
feature <code>i</code>. The distance is f32.</li>
<li><code>output.uniquePositions</code> and <code>output.uniqueOffsets</code> give <code>extract_unique_points</code>: first-occurrence order, as Shapely,
with <code>-0</code> equal to <code>0</code>.</li>
<li>Limits: one thread per feature in the row-wise kernels (a huge single feature is serial); the
self-mode look-ahead is 1024; an empty polygon gives 0 geometries where Shapely gives 1. Open:
<code>equals_exact(normalize=True)</code>, multilinestring and multipoint layouts.</li>
</ul>
<h4 id="how-it-computes-2">How it computes</h4>
<p>{/* lens:GPUGeometryPredicates */}</p>
<p>::lens{scenario=gpu-spatial-analysis.geometry-predicates rung=datum datum=0 caption="Inspect park 0, Lincoln (Abraham) Park: vertex count, simplicity and holes."}</p>
<p>::lens{scenario=gpu-spatial-analysis.geometry-predicates rung=kernel caption="Read the per-feature column kernels."}</p>
<p>::lens{scenario=gpu-spatial-analysis.geometry-predicates rung=graph caption="Follow the count columns and the same-feature segment intersection for simplicity."}</p>
<p>{/* parameters: GPUGeometryPredicates generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpugeometrypredicates-parameters"><code>GPUGeometryPredicates</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUGeometryPredicatesParameterValues(settings?)</code>, writes <code>GPU_GEOMETRY_PREDICATES_PARAMETER_LENGTH</code> (4) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPUGeometryPredicatesParameterValues</code>:</p>
<table>
<thead>
<tr>
<th>Field</th>
<th>Type</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>tolerance</code> (optional)</td>
<td><code>number</code></td>
<td><code>equalsExact</code> tolerance: two vertices match when their distance is at most this value, in position units. Defaults to <code>0</code> (vertices must coincide).</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="gpuminimumclearance"><code>GPUMinimumClearance</code></h3>
<p>Measures the smallest distance a single vertex could move before each feature becomes invalid, and which vertex sets it.</p>
<p>::lens{scenario=gpu-spatial-analysis.minimum-clearance rung=result caption="102 park polygons shaded by minimum clearance in metres."}</p>
<p>Maximum: <code>geometry.positions</code> has at most <code>getGPUSegmentIntersectionMaximumVertexCount(device)</code> (4,194,304 on a default device) rows, because the 32-byte segment table per vertex must fit one 128 MiB binding.</p>
<p><code>leafCapacity</code> (optional, a power of two) sets the BVH leaf slots over the segments and defaults to the next
power of two of the vertex count, which never overflows. A smaller value drops the trailing segments from the BVH
(in Morton order when <code>spatialSort</code> is on), so clearances may come out too large. To detect this, pass
<code>output.status.candidateOverflow</code>, a one-row <code>uint32</code> view that receives <code>1</code> when segments were dropped and <code>0</code>
otherwise. Without that view an undersized <code>leafCapacity</code> is silent.</p>
<p>Planar only: a longitude/latitude <code>spatialContext</code> throws at construction; project first with
<code>createGPULocalFrame</code> and <code>GPULocalFrameProjection</code>. It measures distance, so <code>local-lambert-equal-area</code> frames throw.</p>
<p>Per-feature <code>shapely.minimum_clearance</code> and <code>minimum_clearance_line</code> for linestrings and
(multi)polygons: the smallest distance one vertex could move to make the feature invalid, as the minimum
over vertices of the distance to every other vertex and to every segment that does not have that vertex as
an endpoint (GEOS rule, repeated points ignored). Props: <code>geometry</code> (lines or polygons), and an <code>output</code>
with at least one of <code>clearances</code> (<code>float32</code>, <code>Infinity</code> when none), <code>lines</code> (<code>float32x4</code>, vertex and
nearest point, NaN when none) and <code>vertexIds</code> (<code>uint32</code> row of the fragile vertex, <code>0xffffffff</code> when none). A segment
table and BVH feed one branch-and-bound probe per vertex, restricted to its own feature, then one 64-lane
workgroup per feature reduces its vertex range to the lexicographic minimum of (distance, vertex row) with a
fixed tree and no atomics, so results are deterministic and ties go to the lowest vertex row. Parity with Shapely
2.1.2 is within 2e-5 relative on 52 multipolygons and 69 linestrings; line end points can come out in the
opposite order and tied pairs can differ from GEOS (the distance never does). Coordinates are
origin-relative f32, so tiled copies shifted by thousands of units agree to about 4e-4. Not provided:
multipoints.</p>
<h4 id="how-it-computes-3">How it computes</h4>
<p>{/* lens:GPUMinimumClearance */}</p>
<p>::lens{scenario=gpu-spatial-analysis.minimum-clearance rung=datum datum=0 caption="Inspect park 0, Lincoln (Abraham) Park: clearance and fragile vertex."}</p>
<p>::lens{scenario=gpu-spatial-analysis.minimum-clearance rung=formula caption="Shortest vertex-to-segment distance within a polygon, excluding the vertex's own edges."}</p>
<p>::lens{scenario=gpu-spatial-analysis.minimum-clearance rung=kernel caption="Read the BVH probe that finds each vertex's nearest other segment."}</p>
<p>::lens{scenario=gpu-spatial-analysis.minimum-clearance rung=graph caption="Follow the segment table, BVH, vertex probe and per-feature minimum."}</p>
<h3 id="gpucoveragevalidity"><code>GPUCoverageValidity</code></h3>
<p>Checks polygons as one coverage for gaps, overlaps, crossings and unmatched shared edges, reported per ring segment.</p>
<p>::lens{scenario=gpu-spatial-analysis.coverage-validity rung=result caption="199 census tracts checked as one coverage with a 1 m gap width; shading counts invalid edges."}</p>
<p>Maximum: <code>positions</code> has at most <code>getGPUSegmentIntersectionMaximumVertexCount(device)</code> (4194304 on a default device: the 32-byte segment-table row per vertex must fit one storage binding) rows.</p>
<p>Polygon coverage validity (Shapely <code>coverage_is_valid</code>, <code>coverage_invalid_edges(gap_width)</code>). One BVH over
all segments writes a per-segment flag word (<code>unmatched</code>, <code>crossing</code>, <code>gap</code>, <code>overlap</code>, <code>uncertain</code> bits),
optional per-feature counts (<code>featureInvalidCounts</code>), a total (<code>status.invalidCount</code>) and <code>isValid</code>. The input is the four-level polygon layout (<code>positions</code>, <code>featureOffsets</code>, <code>polygonOffsets</code>, <code>ringOffsets</code>); segments of one feature are never compared. <code>gapWidth</code> is per-frame through
<code>getGPUCoverageValidityParameterValues</code>. The output has one row per vertex and cannot overflow. Matched
edges are skipped and flagged only when both interiors are on the same side, and a T-junction touch counts.
The gap rule is the GEOS one (<code>InvalidSegmentDetector::isNearlyParallel</code>): each segment is projected onto the
other and clipped to it, and the pair is a gap when both clipped projections are longer than <code>gapWidth</code>
(strict) and their matching ends are within <code>gapWidth</code> of each other (inclusive). Staggered segments that run
side by side over only part of their length, such as <code>(0, 0)-(10, 0)</code> beside <code>(5, 0.0625)-(15, 0.0625)</code> at
<code>gapWidth</code> 0.125, are flagged (until 2026-11-11 they were not: the rule then asked for both ends of the
shorter segment within <code>gapWidth</code> of the longer). The rule is evaluated in float32 relative to the start of
the tested segment, so a distance or projected length equal to <code>gapWidth</code> to within float32 rounding can fall
on either side. With Shapely 2.2.0 (GEOS 3.14.1) the invalid segment sets are equal on the 28 curated scenes
(40 gap cases) and on the 222 cases of <code>test/oracles/fixtures/coverage-validity-gaps.json</code> (staggered
rectangles, disjoint pairs and jittered triangulations, near the origin and at projected-metre coordinates).
A matched edge lying inside a third polygon is not flagged (that polygon's own segments are). Ring winding
is the exact turn at each ring's lexicographically
smallest vertex, so a sliver whose float32 area is zero is still oriented. Every orientation sign is exact
for the f32 vertices, including the winding number of the interior probe, which is the shared symbolic side
probe (the point beside the exact midpoint of the segment, classified from the exact signs of the segment's
endpoints), so a segment of a polygon thinner than one float32 step is not read as lying inside its
neighbor. A pair with no decidable orientation (non-finite input) sets the <code>uncertain</code> bit. The operator is planar: <code>spatialContext</code> defaults to native
planar coordinates and <code>longitude-latitude</code> throws (project first).</p>
<h4 id="how-it-computes-4">How it computes</h4>
<p>{/* lens:GPUCoverageValidity */}</p>
<p>::lens{scenario=gpu-spatial-analysis.coverage-validity rung=datum datum=0 caption="Inspect tract 17031010100: its invalid edge count."}</p>
<p>::lens{scenario=gpu-spatial-analysis.coverage-validity rung=kernel caption="Read the segment-pair kernel that sets the flag bits."}</p>
<p>::lens{scenario=gpu-spatial-analysis.coverage-validity rung=graph caption="Follow the segment table, BVH probe and per-feature counts."}</p>
<p>{/* parameters: GPUCoverageValidity generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpucoveragevalidity-parameters"><code>GPUCoverageValidity</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUCoverageValidityParameterValues(parameters)</code>, writes <code>GPU_COVERAGE_VALIDITY_PARAMETER_LENGTH</code> (4) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPUCoverageValidityParameterValues</code>:</p>
<table>
<thead>
<tr>
<th>Field</th>
<th>Type</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>gapWidth</code></td>
<td><code>number</code></td>
<td>Largest gap, in position units, that counts as an error: Shapely <code>gap_width</code>. Two segments of different polygons that run nearly parallel within this distance are reported. <code>0</code> disables gap detection. Must be finite and non-negative.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>`};export{e as default};