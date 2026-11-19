var e={slug:`api/gpu-spatial-analysis/urban-morphology`,title:`Urban morphology recipes`,html:`<h1 id="urban-morphology-recipes">Urban morphology recipes</h1>
<p>These graph recipes operate on projected planar coordinates. They append GPU work to the supplied
command graph and retain caller-owned inputs. Returned graph-owned outputs stay on the GPU; callers
that need readback must add a copy into a caller-owned readback buffer, or pass an <code>output</code> prop
(see below) so the results are written into views the caller owns. Horizontal coordinates and
building heights must use compatible physical units. No recipe submits work or reads data on the CPU.</p>
<p>The implementations retain explicit capacities and approximation controls. Direct pinned momepy
0.10.0 fixtures cover street width and nonconvex/holed tessellation areas; independent Shapely,
NetworkX and analytical tests cover additional geometry and graph invariants. This is evidence for
the documented modes, not complete momepy or depthmapX library equivalence.</p>
<h2 id="caller-owned-outputs-limits-and-obstacle-index">Caller-owned outputs, limits and obstacle index</h2>
<p>Each recipe takes an optional <code>output</code> prop. Every entry is a view the caller created in the same
graph, with the format and length of the recipe's transient view of that name; an absent entry is a
graph-owned transient view, as before. The <code>outputs</code> of the result are the views actually used.</p>
<table>
<thead>
<tr>
<th>Recipe</th>
<th><code>output</code> entries</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>addBuildingTessellationRecipe</code></td>
<td><code>overlays</code> (one <code>GPUPolygonOverlay</code> output per building, the final polygons), <code>samples</code>, <code>owners</code>, <code>fragments</code>, <code>fragmentCounts</code>, <code>samplingStatus</code>, <code>complete</code></td>
</tr>
<tr>
<td><code>addStreetProfilesRecipe</code></td>
<td><code>profiles</code>, <code>buildings</code>, <code>validity</code>, <code>canyon</code>, <code>frontage</code>, and with <code>streetSegment</code> <code>facadeLengths</code>, <code>facadeFrontage</code></td>
</tr>
<tr>
<td><code>addBuildingAdjacencyRecipe</code></td>
<td><code>pairs</code>, <code>lengths</code>, <code>morphology</code>, <code>components</code>, <code>complete</code>, and with <code>structures</code> <code>structurePerimeters</code></td>
</tr>
<tr>
<td><code>addIsovistsRecipe</code></td>
<td><code>endpoints</code>, <code>distances</code>, <code>segments</code>, <code>validity</code>, <code>measures</code>, <code>angles</code>, <code>viewpointClassifications</code></td>
</tr>
<tr>
<td><code>addVisibilityGraphRecipe</code></td>
<td><code>adjacency</code>, <code>edges</code>, <code>metrics</code>, <code>choice</code>, <code>integration</code>, <code>normalizedChoice</code>, <code>viewpointClassifications</code>, <code>extendedIntegration</code>, <code>status</code></td>
</tr>
</tbody>
</table>
<p>Limits are named constants and are rejected with a message that starts with the recipe ID and names
the prop. Tessellation: 64 buildings (one overlay per building), 32 enclosure vertices and 128 boundary
samples (thread-private polygons and a samples-squared fragment table). Adjacency: 1,024 buildings
(buildings-squared pair table), and 64 buildings with <code>structures</code>. Visibility graph: 256 viewpoints.
Isovists: 4,096 rays per viewpoint with <code>sampling: 'vertices'</code> and 65,536 otherwise. Street profiles:
1,024 obstacle segments with <code>streetSegment</code>. Every recipe accepts at most 16,777,216 obstacle segments.</p>
<p><code>addIsovistsRecipe</code> and <code>addStreetProfilesRecipe</code> take <code>obstacleIndex</code>: <code>'brute-force'</code> tests every
segment for every ray, <code>'bvh'</code> builds a Morton-sorted segment BVH (<code>GPUBVH</code>) and traverses it, and the
default <code>'auto'</code> uses brute force up to 4,096 segments and the BVH above that. Both give identical
hits: the nearest segment wins and the lowest segment row breaks ties. Segment finiteness is checked
once per segment, not once per ray. <code>addVisibilityGraphRecipe</code> takes the same <code>obstacleIndex</code> and
applies it to its sightlines. In the isovists and the visibility graph the free-space classification
uses the same index: with the BVH each viewpoint visits only the segments that meet the horizontal ray to its right
(the ones that change its winding number or lie within the tolerance), and one reduction finds
non-finite segments. The classification is identical either way.</p>
<h2 id="addbuildingtessellationrecipe--ur01"><code>addBuildingTessellationRecipe</code> — UR01</h2>
<p><code>addBuildingTessellationRecipe</code>: Local mode uses an adaptive candidate index and publishes
convex-piece triangle unions for bounded simple nonconvex or holed enclosures (at most 32 rings).
Reference mode retains dissolved rings, 64 buildings and 128 samples. Device bindings and
certification/overflow status still apply.</p>
<p>Supply closed building boundary <code>segments</code>, their <code>segmentBuildingIds</code>, a <code>buildingCount</code>, and one
<code>enclosure</code>. Convex counterclockwise rings use the default fast path. Set
<code>enclosureMode: 'simple'</code> for a nonconvex ring; provide <code>enclosureRingOffsets</code> to describe an
exterior and its holes in the same position view. Segment endpoints are sampled at intervals no larger than
<code>sampleSpacing</code>; repeated segment end vertices are avoided. The recipe clips every boundary-sample
Voronoi cell to the enclosure, groups cells by building, and uses the existing <code>GPUPolygonOverlay</code>
dissolve contributor to remove internal fragment boundaries. <code>outputs[buildingId]</code> is an actual
packed polygon result; the building ID is its index in <code>outputs</code> (the overlay result has no
source-ID column, so the former per-ring copy of the building ID is gone).</p>
<p><code>sampleCapacity</code> is explicit and at most 128; enclosures have 3–32 vertices, and a recipe supports
at most 64 buildings. <code>capacity.intersections</code>, <code>nodedSegments</code>, <code>vertices</code> and <code>rings</code> separately
bound dissolve work and output. Sampling status is <code>[written, required, overflow, invalidSegments]</code>.
<code>status.complete[buildingId]</code> combines sampling, fragment validity and overlay capacity evidence.
An overflow is never a complete tessellation; increase the reported capacity and rerun. The returned
sample locations, building owners, cell fragments and counts preserve the approximation provenance.
Fragment-count <code>0xffffffff</code> denotes invalid enclosure geometry, incompatible duplicate sites or a
fragment-capacity failure. Zero is an empty/duplicate cell.</p>
<p>In simple mode, cells are clipped to the enclosure bounding box, dissolved by building, then
intersected with the actual enclosure through <code>GPUPolygonOverlay</code>. Both dissolve and intersection
capacity evidence gates completeness; holes and disconnected clipped pieces retain the same building
ID. Ring offsets must describe one valid polygon (exterior followed by holes); the input is not a
request for automatic repair. Run separate enclosures independently. Street-network enclosure
construction and automatic building shrinking remain explicit upstream geometry operations. Duplicate
sites owned by different buildings invalidate the result. Sample spacing determines the building
boundary approximation; <code>vertexTolerance</code> independently controls the common overlay snap grid and
sliver collapse. Halving spacing is the appropriate convergence check. A building may legitimately
have an empty cell if no sampled support owns area within the enclosure.</p>
<h2 id="addstreetprofilesrecipe--ur02"><code>addStreetProfilesRecipe</code> — UR02</h2>
<p>Pass centerline sample <code>positions</code>, left-pointing <code>normals</code>, building <code>segments</code>,
<code>segmentBuildingIds</code>, and <code>buildingHeights</code>. <code>sampleSpacing</code> records caller sampling provenance;
<code>maximumDistance</code> bounds each perpendicular sightline. Normals are normalized on the GPU.</p>
<ul>
<li><code>profiles</code>: <code>[leftDistance, rightDistance, leftHeight, rightHeight]</code> per sample.</li>
<li><code>buildings</code>: left and right stable building IDs, or <code>0xffffffff</code> for an absent side.</li>
<li><code>validity</code>: bit 0 left hit, bit 1 right hit, bit 2 viewpoint on a boundary, bit 3 invalid input.</li>
<li><code>canyon</code>: <code>[observedWidth, meanHeight, heightToWidthRatio, rangeCensoredWidth]</code>.</li>
<li><code>frontage</code>: <code>[leftFrontage, rightFrontage]</code> per building, summing <code>sampleSpacing</code> for each valid hit.</li>
</ul>
<p>An absent side has the finite range limit as its censored distance and NaN height. Observed canyon
width and ratio are NaN unless both sides hit and the viewpoint is valid. Missing/negative heights
preserve widths and invalidate height-dependent measures. Shared segment endpoints are closed
obstacles; collinear rays stop at the closest forward overlap. Stable segment order resolves ties.
Frontage is a sampled centerline measure: each sample represents an interval of <code>sampleSpacing</code>.
Use interval-midpoint samples to avoid double-counting endpoints; it is not exact facade arc length.
Invoke the recipe per street when separate street-level frontage is required.</p>
<p>For exact facade arc length along one straight street, provide <code>streetSegment</code> (one <code>float32x4</code>
start/end row). <code>facades.lengths</code> and <code>facades.frontage</code> then contain left/right visible facade length
per segment and per building. The GPU splits facade segments at corridor endpoints, range/side
crossings, obstacle endpoint projections and segment intersections, and tests visibility inside each
resulting interval. An occluded wall contributes no facade length. This is exact for the declared
straight rectangular corridor up to the shared geometric tolerance, rather than a sample-count
estimate. It supports at most 1,024 obstacle segments; curved streets require explicit subdivision
and ownership at joins so facade intervals are not counted twice.</p>
<p>A stretch of wall is visible when no other segment crosses the perpendicular from the street axis
before it. The facade kernel works relative to the street start, so the same geometry gives the same
lengths at the origin and at projected-metre coordinates (the inputs are still float32: near
4,500,000 a coordinate resolves 0.5). Two rules use a tolerance that scales with the geometry
(<code>tolerance</code> plus 1e-5 of a length):</p>
<ul>
<li>A wall perpendicular to the street, within that slack of its own length, is edge-on and has no
visible length.</li>
<li>Where segments coincide, the lowest segment row owns the stretch and the others report 0 there.
Segments coincide when their distances along the sightline are within that slack (of the larger
of the street length and <code>maximumDistance</code>) of the nearest one; a segment farther than that
behind the nearest one is hidden. Ownership is decided against the one nearest hit, so any number
of coincident walls has exactly one owner. Two walls that meet at a corner are each counted once.</li>
<li>A segment within <code>tolerance</code> of the point where a sightline leaves the street axis blocks that
sightline at distance 0, on either side of the axis.</li>
</ul>
<h2 id="addbuildingadjacencyrecipe--ur03"><code>addBuildingAdjacencyRecipe</code> — UR03</h2>
<p>Compose the existing <code>GPUSharedPaths</code> result with a common <code>lineBuildingIds</code> mapping and axial
<code>orientations</code> in radians, normally from the existing shape-descriptor contributor. Supply each
unordered building-boundary pairing once; repeated runs would double-count wall length. Each dense
<code>pairs[first * buildingCount + second]</code> row contains
<code>[sharedWallLength, axialAngleDifference, hasSharedWall, valid]</code>. Axial differences wrap modulo π
into <code>[0, π/2]</code>. A point touch contributes zero wall length and is not wall adjacency. Missing
orientation affects the angle only. Numerical outputs are NaN when supplied upstream overflow or
uncertain-intersection evidence is nonzero. <code>lengths</code> retains each measured shared run.</p>
<p>The recipe does not recompute a shape orientation or replace the shared-path topology algorithm.
The line mapping must use the same row identities for both operands, and all upstream capacity
status should be supplied when completeness is required.</p>
<p><code>morphology</code> contains <code>[totalSharedWalls, cardinalOrientationDegrees, meanCardinalAlignmentDegrees, structureExteriorPerimeter]</code>. Cardinal orientation folds the supplied axial shape direction into
0–45 degrees; alignment averages absolute cardinal-deviation differences over neighbors, with zero
for an isolate (momepy convention). This reuses the caller's existing minimum-rectangle/shape
orientation rather than fitting a second rectangle. <code>contiguity</code> optionally supplies undirected
Queen/custom building pairs; otherwise positive shared walls define neighbors. Point contacts can
therefore connect a structure without becoming shared walls. Out-of-range neighborhood IDs produce
sentinel components and undefined alignment/perimeter.</p>
<p>Optional <code>structures</code> accepts one packed polygon feature per building, a snap tolerance and explicit
intersection/noded-segment/vertex/ring capacities. GPU connected components drive a positive-winding
polygon union for each building's component; only exterior rings contribute to perimeter, so
courtyards are excluded. Input shells must be counterclockwise and holes clockwise. This matches the
bounded <code>momepy.perimeter_wall(..., buffer=0)</code> workflow, including isolated footprints. Desired buffer
or repair operations are explicit upstream geometry preparation. At most 64 buildings are supported
in this optional branch (1,024 for adjacency alone). <code>structures.overlays</code> returns every union's
geometry and independent capacity evidence; incomplete unions publish NaN perimeter. Returned
<code>components</code> preserves the canonical smallest building ID.</p>
<h2 id="addisovistsrecipe-and-addvisibilitygraphrecipe--ur05"><code>addIsovistsRecipe</code> and <code>addVisibilityGraphRecipe</code> — UR05</h2>
<p><code>addVisibilityGraphRecipe</code>: The <code>gpu-network</code> recipe of the same name supports budgeted exact CSR all-source metrics or
deterministic sampled estimates via metricsMode auto/exact. Auto samples when the exact command
budget cannot fit; sampled maximum depth is a lower bound. Approximation, overflow and convergence
are separate statuses; explicit exact rejects an insufficient budget.</p>
<p>Isovists cast <code>rayCount</code> uniformly spaced rays counterclockwise from +X, clipped by closed segment
obstacles and a finite circular <code>maximumDistance</code>. <code>endpoints</code>, <code>distances</code>, hit <code>segments</code>, and
per-ray <code>validity</code> preserve every sampled ray. <code>measures</code> is
<code>[polygonArea, polygonPerimeter, maximumRayDistance, valid]</code> per viewpoint. The polygon joins adjacent
ray endpoints. With <code>sampling: 'uniform'</code> it can miss an obstacle between rays. With
<code>sampling: 'vertices'</code>, the GPU adds before/at/after rays at every segment endpoint and sorts all
angles; <code>angularEpsilon</code> defaults to <code>1e-5</code> radians. This resolves narrow endpoint events while
retaining uniform rays for the circular range boundary. Total rays are <code>rayCount + 6*segmentCount</code>
and at most 4,096 in this mode. Returned <code>angles</code> records the actual sampling. It remains a
controlled angular approximation, not an exact circular-arc polygon. Viewpoints on
an obstacle are invalid. Without <code>region</code>, segment input checks boundary contact only. Set
<code>region: {fillRule: 'even-odd' | 'nonzero', freeSpace: 'inside' | 'outside'}</code> to classify closed
segment rings explicitly. Outside mode models filled obstacles; inside mode models an enclosure.
Even-odd supports nested holes, while nonzero uses directed shell/hole winding. Rings must be closed
and topologically valid. <code>viewpointClassifications</code> reports 0 free, 1 blocked, 2 boundary, or 3
nonfinite input. Both isovists and visibility graphs consume this shared GPU classification.</p>
<p>Size each per-ray output with
<code>positions.length * getGPUIsovistRayCount(rayCount, segments.length, sampling)</code> (import the helper
from <code>@luma-contributors/analysis/gpu-spatial-analysis</code>). This applies to <code>endpoints</code>, <code>distances</code>,
hit <code>segments</code>, <code>validity</code> and <code>angles</code>; <code>measures</code> and <code>viewpointClassifications</code> each have
<code>positions.length</code> rows. The helper and recipe require a base <code>rayCount</code> integer in [3, 65,536]
and a segment count integer in [1, 16,777,216]. With vertex sampling, the expanded count must
also be at most 4,096. All caller output lengths are checked before classification or BVH nodes
are added. For one viewpoint, <code>rayCount: 3</code> and one segment, vertex sampling needs nine rows in
each per-ray output. Provenance records <code>baseRayCount</code> for the uniform rays and <code>rayCount</code> for
the total; <code>angularSpacing</code> is <code>2π / baseRayCount</code>, the spacing of the uniform subset. Use the
returned <code>angles</code> for the actual, nonuniform vertex-sampled angles.</p>
<p>Every output must have disjoint logical byte ranges from inputs and other outputs; for example,
<code>angles</code> and <code>measures</code> cannot alias the same 16 bytes even when both view lengths are correct.
Views bound together in a dispatch also need disjoint aligned binding ranges, and a physical buffer
cannot be both a read-only and a writable binding in that dispatch. When disjoint outputs share an
allocation, create their subviews on one graph buffer handle; separate imports of a physical buffer
are rejected when either handle is written. All supplied output layouts, graph ownership and alias
rules are checked before reserving transient resource IDs or adding commands. Correcting a rejected
override permits retrying with the same recipe ID on the same graph. Inputs must be single packed
views: <code>positions</code> uses <code>float32x2</code> and <code>segments</code> uses <code>float32x4</code>. Both formats and packed layouts
are checked before any output is resolved, so correcting an invalid input also permits a retry
with the same recipe ID on the same graph.</p>
<p>The visibility graph evaluates all pairs among at most 256 supplied free-space viewpoints. An
obstacle contact, including a tangent or endpoint contact, blocks a pair. Coincident viewpoints do
not create an edge. <code>adjacency</code> is the complete dense visibility matrix; <code>edges</code> contains the bounded
upper-triangle edge list. <code>status</code> is <code>[writtenEdges, requiredEdges, overflow, invalidViewpoints]</code>.
Dense analysis remains valid when only the edge-list capacity overflows, but an incomplete edge list
must not be consumed as a complete network. Invalid viewpoints invalidate all graph summary metrics: when any viewpoint is invalid (<code>status</code>
word 3 is not 0), <code>metrics</code>, <code>choice</code>, <code>integration</code>, <code>extendedIntegration</code> and <code>normalizedChoice</code> are NaN
for every viewpoint, and <code>normalizedChoice</code> is also NaN for every viewpoint when fewer than 3
viewpoints are given. In a valid graph a viewpoint that sees nothing scores 0.</p>
<p><code>metrics</code> contains <code>[degree, reachableOtherVertices, sumHopDistances, reachableHopCloseness]</code>.
Closeness is reachable count divided by total hop distance, and is NaN for an isolated point.
<code>choice</code> is unnormalized undirected Brandes betweenness (excluding source and target endpoints).
<code>normalizedChoice</code> divides undirected choice by <code>(N-1)(N-2)/2</code>, using the supplied graph's full
vertex count. <code>integration</code> is <code>[meanDepth, relativeAsymmetry, realRelativeAsymmetry, HHIntegration]</code>
on each reachable component. For component size <code>n</code>, mean depth is <code>totalHopDepth/(n-1)</code>, relative
asymmetry is <code>2*(meanDepth-1)/(n-2)</code>, and the Hillier–Hanson diamond normalization is
<code>2*(n*(log2((n+2)/3)-1)+1)/((n-1)*(n-2))</code>. HH integration is its reciprocal normalized asymmetry.
Components smaller than three return NaN for normalized asymmetry/integration; a complete component
has zero asymmetry and infinite HH integration. These are topological VGA measures; angular segment
NACh/NAIn are different segment-graph definitions and are not relabeled as VGA.
<code>extendedIntegration</code> is <code>[pyramidIntegration, TeklenburgIntegration, PennNormalization, maximumShortestPathDepth]</code>. Pyramid integration divides
<code>2*(n-log2(n)-1)/((n-1)*(n-2))</code> by relative asymmetry. Teklenburg is
<code>log(0.5*(n-2))/log(totalDepth-n+1)</code>. Penn is <code>(Dmax-totalDepth)/(Dmax-(n-1))</code>, with
<code>Dmax=r*(n-(r+1)/2)</code> and the actual maximum BFS depth <code>r</code>. These additional normalizations use
NaN where depthmapX marks the value unavailable (small/complete components or a zero denominator);
HH retains its existing infinite complete-component result. The Penn formula is the named axial
normalization applied to the supplied topological graph, not metric Penn distance.
The distinction follows the <a href="https://github.com/SpaceGroupUCL/depthmapX/blob/master/docs/howdoi.md">depthmapX metric documentation</a>.</p>
<h2 id="reference-and-execution-evidence">Reference and execution evidence</h2>
<p><code>modules/analysis/test/oracles/urban-ecology-advanced.py</code> executes momepy 0.10.0 directly with
<code>shrink=0</code>, <code>segment=1</code>, and explicit enclosure polygons, including a hole. It also directly compares street-profile width and strict shared-wall totals, including a
point-only contact, and executes the pinned segregation package described in the ecology documentation. The native advanced suite tests
nonconvex/holed areas, occluded facade intervals, narrow angular events, and HH normalization.
Capacity overflow and boundary-viewpoint tests remain in the original urban/ecology suite.
<code>urban-depthmap.py</code> downloads the pinned v0.9.1 mathematical header, records its SHA-256, compiles
and directly executes its HH/P-value/Teklenburg/palm-tree helpers for a known visibility path graph.
This is direct upstream normalization evidence, not a full GUI/CLI graph-import comparison. The
momepy fixture also executes zero-buffer joined-structure perimeter, cardinal orientation and
alignment, including an isolated courtyard footprint. No city-wide tessellation throughput is claimed.</p>`};export{e as default};