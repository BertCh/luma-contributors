var e={slug:`api/gpu-geometry-refinement`,title:`GPU geometry refinement`,html:`<h1 id="gpu-geometry-refinement">GPU geometry refinement</h1>
<p>The contributors in <code>@luma-contributors/analysis/gpu-spatial-analysis</code> add bounded geometry
processing to caller-owned <code>GPUCommandGraph</code> buffers. They return command nodes;
they never submit, read back or run a CPU geometry fallback. All geometric
measurements use planar coordinates. Use a local coordinate frame for numerical
stability. Output capacities and compile-time budgets belong to the caller.</p>
<table>
<thead>
<tr>
<th>Contributor</th>
<th>Input and output contract</th>
<th>Limits and status</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>GPUNormalizePolygons</code></td>
<td>Four-level polygon layout → canonical coordinates, ring offsets and polygon offsets. Reuse input feature offsets. Shells become clockwise, holes counterclockwise; rings start at their smallest vertex; holes and parts sort descending.</td>
<td>Finite inputs, valid offsets. Feature order, empty parts, repeated vertices and closure conventions are preserved. Use <code>GPUNormalizeGeometry</code> for mixed collections and line/point normalization.</td>
</tr>
<tr>
<td><code>GPUCanonicalPolygonEquality</code></td>
<td>Pair corresponding normalized features; compare structure and coordinates without hashes. <code>tolerance</code> is absolute per-coordinate tolerance.</td>
<td>Output 1 equal, 0 unequal, 2 nonfinite. Canonical structure and closure must agree.</td>
</tr>
<tr>
<td><code>GPUSubdividePolygons</code></td>
<td>Triangulate polygon interiors, then merge adjacent triangles into convex pieces of at most <code>maximumVertices</code> (3–128). A piece occupies a fixed block of that many position slots.</td>
<td><code>pieces</code> rows are <code>[vertexCount, sourceFeature, sourcePolygon, sourceSeedTriangle]</code>; <code>status</code> is <code>[written, required, capacityOverflow, invalidPolygonCount]</code>. Holes remain excluded from the piece union. Rectilinear leaf offsets and the required piece total saturate at <code>0xffffffff</code>; capacity overflow preserves the output prefix. <code>method: 'rectilinear'</code> uses a bounded GPU queue to split convex seeds by axis-aligned clips (minimum limit 4). Bit 31 of the final status word reports incomplete splitting. The seed decomposition and piece ordering differ from PostGIS; the union and per-piece vertex bound are the comparison contracts.</td>
</tr>
<tr>
<td><code>GPULinePrecisionReduction</code></td>
<td>Grid-round lines, remove successive repeats, preserve source line slots.</td>
<td><code>valid-output</code> drops collapsed lines; <code>keep-collapsed</code> retains two equal endpoints for collapsed input lines. Per-line status 0 regular/empty, 1 dropped collapse, 2 kept collapse, 3 nonfinite. Output position capacity must cover the input length.</td>
</tr>
<tr>
<td><code>GPUPolygonPrecisionReduction</code></td>
<td>Materialize ring closures, hot-pixel node boundaries and reconstruct even-odd polygon faces. Reuses bounded <code>GPUPolygonizeOutput</code>, including provenance and remnants.</td>
<td>Any positive finite float32 <code>gridSize</code>; <code>lattice = [originX, originY, actualStep, 0]</code> may report a larger step for large extents. Both collapse modes drop collapsed polygons. The topology operates in grid-normalized units and converts coordinates, provenance endpoints and the actual lattice back to input units.</td>
</tr>
<tr>
<td><code>GPUSegmentSnap</code></td>
<td>Snap endpoints to nearest references, then insert nearby unique references in segment projection order. Equal-distance ties choose the first reference row.</td>
<td>Fixed <code>maximumVerticesPerLine</code> blocks; <code>counts</code> are <code>[written, required, overflow, nonfinite]</code>. Optional <code>issueCode</code> bits report collapse (1), nonfinite input (2) and incomplete capacity or candidates (4); decode with <code>decodeGPUSegmentSnapIssueCode</code> or <code>GPU_SEGMENT_SNAP_ISSUE</code>. Optional <code>validity</code> is 1 only when no issue is present. Empty lines are usable; self-crossing lines remain usable. Inclusive planar tolerance. GEOS safety heuristics are not implemented and topology validity is not certified.</td>
</tr>
<tr>
<td><code>GPULineLoopRemoval</code></td>
<td>Exact-predicate noding followed by chronological loop erasure for open curves, including prepared raw offsets. Direction, source line IDs through the offset layout, and original edge IDs are retained.</td>
<td><code>status</code> is <code>[writtenVertices, requiredVertices, outputOverflow, incompleteNoding]</code>. Whole lines that exceed remaining output capacity are omitted. Use <code>GPUOffsetCurveComponents</code> for distance-band clipping and multipart component selection.</td>
</tr>
<tr>
<td><code>GPURatioConcaveHull</code></td>
<td>Delaunay border erosion using <code>minimumEdge + ratio * (maximumEdge - minimumEdge)</code>, independent of alpha-complex circumradius. Erosion runs in parallel rounds (each round removes every removable border triangle that is the largest around its three vertices), so the hull is a valid erosion hull for the ratio but not the largest-first sequence of GEOS <code>ConcaveHull</code>: it can keep different triangles than <code>shapely.concave_hull</code>. Optional interior erosion seeds holes.</td>
<td>Uses the same masked triangle/boundary-edge shape as <code>GPUConcaveHull</code>. <code>converged</code> covers tessellation and erosion budget. Ratio 1 yields the convex area hull. Point/collinear sets have no area boundary.</td>
</tr>
<tr>
<td><code>GPUMaximumInscribedCircle</code></td>
<td>Per-feature best-first search returns <code>(centerX, centerY, radius, remainingSearchError)</code>, nearest boundary witness, and status. Holes and multipart boundaries use even-odd fill.</td>
<td>0 tolerance success, 1 cell/iteration exhaustion, 2 empty, 3 nonfinite. The remaining bound includes all unresolved cells. It bounds geometric search discretization, <strong>not floating-point roundoff</strong>; this is not an interval-arithmetic certificate.</td>
</tr>
<tr>
<td><code>GPUConvexStraightSkeleton</code></td>
<td>Parallel edge wavefronts return skeleton arcs, original bounding edge IDs, and an inward polygon offset at <code>offsetDistance</code>. Simultaneous rectangle collapses produce a terminal ridge.</td>
<td>Strictly convex simple polygons, up to 64 vertices. Status <code>[writtenArcs, requiredArcs, offsetVertices, flags]</code>: unsupported 1, capacity 2, budget 4, numerical 8. Holes, nonconvex split events and collinear corners remain unsupported.</td>
</tr>
</tbody>
</table>
<p>| <code>GPUNormalizeGeometry</code> | Postorder tree with point, line, ring, polygon, multi and collection nodes; stable source IDs remain in the input node table. Normalize leaf direction/winding and sort children by GEOS type, coordinate count and coordinates. | Nesting at most 64; empty nodes remain explicit. Status per node: 0 success, 1 nonfinite leaf, 2 invalid structure/range/depth, uncertain orientation or invalid child. Leaf position ranges must be disjoint. |
| <code>GPUCanonicalGeometryEquality</code> | Compare pairs of successful normalized tree roots with explicit absolute coordinate tolerance. | 1 equal, 0 unequal, 2 invalid structure/nonfinite. Geometry structure matters; source IDs do not participate. |
| <code>GPUOffsetCurveComponents</code> | Node prepared raw offsets, analytically split against source segment distance bands, reject covered and collapsed fragments, then assemble directed multipart curves. | <code>original</code> and <code>raw</code> have aligned source line order/direction. <code>rowsPerVertex</code> maps join padding to original source edges. Status rows: <code>[writtenComponents,writtenVertices,requiredComponents,requiredVertices]</code>, <code>[overflow,incompleteNoding,0,0]</code>. Only complete components are published. |
| <code>GPUVectorStraightSkeleton</code> | Linked wavefronts process edge collapse, reflex split, hole merge, collinear motion and simultaneous terminal ridges. Return skeleton arcs and original bounding edge IDs, plus linked inward-offset loops. | Valid planar polygons with holes; maximum 1024 original vertices and 4096 events per polygon. Status <code>[writtenArcs,requiredArcs,activeOffsetVertices,flags]</code>: input 1, capacity 2, work 4, numerical 8. <code>eventTolerance</code> controls float32 event grouping. |</p>
<h2 id="composition">Composition</h2>
<p><strong>Breaking (2026-11-17):</strong> segment snap's former diagnostic <code>output.validity</code> is now
<code>output.issueCode</code>; <code>output.validity</code> has nonzero-usable semantics. The contributor, GPU operation
and CPU twin share both ports. <code>computeSegmentSnapOnCPU</code> returns both arrays.
The third <code>counts</code> word is final vertex-capacity overflow only (<code>required > maximumVerticesPerLine</code>).
Candidate exhaustion alone leaves it zero; <code>status.candidateOverflow</code> reports the candidate loss,
while <code>issueCode</code> bit 4 and <code>validity = 0</code> identify suppressed incomplete lines. Growing the vertex
output cannot repair a candidate budget. Conversely, final vertex truncation sets the third word
even when every candidate fits.
When grid suppression is possible, construction requires at least one per-line signal:
<code>output.validity</code>, <code>output.issueCode</code>, or <code>output.status.candidateOverflow</code> (the legacy alias is
also accepted). This applies to the contributor and both operation forms. Suppression is possible
when input positions are nonempty, reference count exceeds <code>candidateCapacity</code>, and either policy
is <code>incomplete</code> or the exhaustive work bound is exceeded. Required outputs alone remain sufficient
for empty input, a list that can hold all references, guaranteed bounded fallback, or explicit
brute force. Policy, input references and nested output/status maps are captured at construction;
later caller mutations do not change the declared operation.
Nonfinite query lines have no candidate search: their <code>candidateOverflow</code> is zero, their
<code>issueCode</code> includes nonfinite (2), and their usability is zero. All output views are retained at
construction; replacing the caller's output map does not redirect later stages. Disjoint aligned
counts and diagnostic views may share a buffer; the publication pass uses compatible read-write
storage bindings for that layout.</p>
<p>Normalize each operand before canonical equality; structural equality requires the
same representation and precision. To compare geometry after precision reduction,
check topology status, normalize the reconstructed outputs and then compare them.
Tree ring winding uses the shared exact extreme-vertex orientation predicate,
avoiding float32 signed-area cancellation after large translations. An uncertain
predicate result propagates invalid status to the containing geometry.
A zero polygon count with overflow set is incomplete work, not an empty geometry.</p>
<p>Subdivision metadata preserves both feature and polygon identity. Join each piece
against the query set, map results back to <code>sourceFeature</code>, and deduplicate feature
pairs. Unused fixed position slots are padding and must never be interpreted as
vertices. Successful piece unions reconstruct successful source triangulations;
failed polygons are explicitly counted.</p>
<p><code>GPULineLoopRemoval</code> consumes explicit open curves. To use it after <code>GPUOffsetCurve</code>,
scale the input line offsets by <code>getGPUOffsetCurveRowsPerVertex</code> and pass the raw
output positions. Repeated join padding is removed during noding/traversal. For complete offset selection, pass the raw curve and original line to
<code>GPUOffsetCurveComponents</code>, including the signed distance and an absolute tolerance
covering round-join chord error. Mitre, bevel and round joins have direct reference
fixtures for both signs, inner corners, self-crossings and multipart results.</p>
<p>Offset components use shared endpoint coordinates for duplicate/reversed-fragment
cancellation and component chaining in both the GPU contributor and CPU twin. Both
coordinates must compare equal; proximity alone never establishes connectivity. Clipping
copies stored noded endpoints at cut parameters <code>0</code> and <code>1</code> instead of recomputing them.
The GPU materializes each interior cut coordinate once and shares it between adjacent
fragments. Accepted nonzero intervals are retained even below <code>1e-7</code> of a segment's length;
discarding those intervals could leave numerical gaps in a continuous component.
Circle cuts store their geometric intersection directly, rather than reconstructing it
from a rounded parameter. In particular, this keeps tangent endpoints identical on
reversed traversals. Equal cut parameters share one stored coordinate.
This preserves tiny shared endpoints of long edges, keeps narrow hairpins distinct,
and prevents chaining across intervals removed by clipping. Endpoint matching uses no
distance tolerance or coordinate-unit floor; the distance-band <code>tolerance</code> applies only
to fragment selection. Exactly reversed spurs still cancel. Float32 representation and
clipping roundoff still limit numerical precision.
Circle intersections use along-segment and perpendicular distances to avoid subtracting
large nearly equal squared quantities at tangencies. Outer joins omit cuts from the two
source edges exempted by their distance test. GPU outward probes also account for float32
coordinate spacing, so a probe can move at projected-coordinate magnitudes; this probe
step does not affect endpoint matching.
An interval that passes the distance-band test and is no longer than the outward probe
is retained: that probe cannot resolve the interval's local two-sided coverage. This
preserves the actual tiny connecting segment instead of joining distinct endpoints by
proximity. Reversed coincident fragments are still removed by endpoint identity.
Near a distance-band tangency, float32 clipping can insert extra collinear subdivision
vertices compared with the float64 CPU twin. Those vertices preserve the directed curve
and source-edge provenance; a minimal vertex count is not guaranteed.</p>
<p><code>GPUMaximumInscribedCircle</code> stores its live cells in graph-owned scratch storage,
using <code>featureCount * cellCapacity * 16</code> bytes. The search supports up to 4096 live
cells per feature and 65536 splits; insufficient capacity or work reports the
unresolved geometric error.</p>
<h2 id="representation-and-bounded-work">Representation and bounded work</h2>
<p>A <code>GPUCanonicalGeometryTree</code> stores <code>nodes: uint32x4</code> rows as
<code>[type, firstVertexOrChildSlot, vertexOrChildCount, stableSourceId]</code> and a separate
<code>children: uint32</code> adjacency list. <code>GPU_CANONICAL_GEOMETRY_TYPE</code> gives the GEOS
comparison order. Child IDs precede their parent, and polygon children are shell
first followed by holes. Reuse the original node buffer with normalized positions
and child IDs. Normalization preserves the source IDs and explicit empty nodes.</p>
<p>Rectilinear subdivision first constructs a convex decomposition of the polygon
interior, including holes, then clips oversized seeds at the midpoint of their
longest bounding-box axis. A depth-first storage queue has 128 slots;
<code>maximumSplits</code> (default 4096, maximum 65536) bounds operations per seed. Collinear
clip vertices are removed. Every successful leaf has at most <code>maximumVertices</code>
vertices and carries feature/polygon provenance. Incomplete status invalidates
any claim that the output union reconstructs the entire input.</p>
<p>Offset component scratch is explicit: with <code>S = segmentCapacity</code>,
<code>C = 6 * original.positions.length + 2</code>, cut parameters and shared coordinates occupy <code>12SC</code> bytes and fragments
plus provenance occupy <code>32S(C - 1)</code> bytes, in addition to noding storage. Selection
and component assembly scan this bounded storage; callers should use capacities
appropriate to the workload. The signed distance belongs to raw-offset generation;
the selector assumes those directed raw offsets were generated from the aligned
source lines. It does not repair arbitrary unrelated linework.</p>
<p><code>CPUOffsetCurveComponentsOperation</code> honours the same explicit noding budgets as its GPU
form. <code>segmentCapacity</code> retains at most that many noded segments in traversal order;
loss sets status word 5 (<code>incompleteNoding</code>) while final-output <code>overflow</code> remains zero
unless output capacity also drops rows. When the intersection-pair count exceeds
<code>intersectionCapacity</code>, the CPU twin suppresses all geometry and sets <code>incompleteNoding=1</code>.
The GPU may retain a spatial-index-dependent pair prefix; the CPU does not replay that
partial ordering. Component/vertex requirements after noding loss describe the retained
result, not the missing complete geometry. Adjacent segments meeting only at their shared
vertex consume no pair capacity; adjacent overlaps do. The standalone
<code>computeOffsetCurveComponentsOnCPU</code> accepts both capacities optionally and remains
uncapped when they are omitted. Its exact rational noding does not impose the GPU's
storage-binding limits or successor-scan bound.</p>
<p>The vector skeleton uses 76 bytes per internal wavefront slot and one 4-byte count
per polygon. When source-position ranges use fewer slots than maximum-vertex
blocks, internal initial-vertex storage follows those ranges (including closing
vertices); each polygon additionally reserves <code>2 * maximumEvents</code> event slots.
Otherwise initial-vertex storage reserves <code>maximumVertices</code> slots per polygon.
Slots are initialized when vertices are created, so unused event capacity does
not add a full clearing pass. Caller-owned offset outputs retain fixed
<code>maximumVertices + 2 * maximumEvents</code> blocks per polygon.</p>
<p>An ascending active-vertex list skips retired vertices during event searches.
Stable nondegenerate components are measured once per event. A conservative
rounding envelope keeps the original ordered checks for components close enough
to the zero-area tolerance that rotating the sum could change its classification.
Edge and split candidate comparisons retain their original ordered tolerance
rules. Polygons execute independently in parallel; each polygon's event chain
remains sequential, and reflex/edge search can still require quadratic work per
event. These changes reduce storage and redundant work without promising scalable
latency for one arbitrarily complex polygon.</p>
<p><code>offsetLinks</code> contains previous/next slot IDs local to that polygon's block;
<code>0xffffffff</code> marks inactive slots. Traverse active cycles to recover multiple
inward offset components, including holes. Invalid/self-intersecting polygon
repair is a separate operation. Event ordering is float32 and tolerance-based,
not an exact-arithmetic skeleton guarantee.
The convex helper compacts surviving edges in place and reuses event-position
storage for new trace origins, avoiding three additional per-invocation arrays.
Its simultaneous-event grouping and source-edge attribution are unchanged.</p>
<h2 id="independent-references-and-scaling-evidence">Independent references and scaling evidence</h2>
<p><code>geometry-refinement.py</code> and <code>geometry-advanced.py</code> directly execute pinned
Shapely 2.1.2; fixtures record GEOS and NumPy versions. Tests cover nested and empty
geometry trees, normalization idempotence, arbitrary grids 0.1/0.3/2.5, polygon
collapse policies, 36 signed offset/join cases, 42 ratio hull endpoint/degenerate
cases, and circle success/exhaustion bounds. Offset assertions include component
count, length, bidirectional coordinate distance and open-curve orientation.
Subdivision compares area and 841 independent Shapely membership queries per
fixture, including holes and concavities. Direct PostgreSQL 16.6/PostGIS 3.5.3
fixtures are generated by <code>geometry-subdivision-postgis.py</code> against GEOS 3.14.1.
The twelve <code>ST_Subdivide</code> cases use closed-ring limits 5/9/17, validate <code>ST_Equals</code>
union reconstruction, <code>ST_IsValid</code>, vertex limits and 841 <code>ST_Covers</code> queries each.
The GPU tests combine all four source geometries in one batch at each limit and
verify per-feature provenance, area and union membership. Engine pieces omit the
closing duplicate, so their corresponding limits are 4/8/16. Different partitions
and piece ordering are permitted; the direct-reference union contract is verified.</p>
<p><code>geometry-straight-skeleton.cpp</code> calls CGAL 6.0.1 directly using
<code>Exact_predicates_inexact_constructions_kernel</code>, with Boost 1.87.0 as the isolated
number backend. <code>geometry-straight-skeleton.py</code> records ten complete arc and inward
offset fixtures: rectangle, triangle, reflex notch, concave L/C, one or two holes,
collinear corners and asymmetric events. The GPU tests match the full arc sets and
traverse reciprocal offset links to compare complete closed rings. Reference dependencies remain external; no CGAL implementation
has been copied into shipped source. Reproduction commands are in the oracle file.
<code>straight-skeleton-storage.node.spec.ts</code> also executes these kernels against the
CGAL fixtures on the CPU interpreter, checks packed mixed-polygon storage, and
reuses the same graph after shrinking source ranges to detect stale snapshot links.</p>
<p><code>geometry-scaling.spec.ts</code> executes subdivision, inscribed circles and concave
straight skeletons on batches of 16, 64 and 256 polygons (96–1536 vertices), checks
every completion status, and times upload, construction, compilation, first/warm
execution, changed inputs and readback separately. The selective GPU bounding-box
join has 50% fewer candidates after subdivision at all three scales. On the
recorded Apple/Metal WebGPU run, warm combined geometry times were 6.2, 5.4 and
12.9 ms. Warm unsplit/subdivided join times were 0.6/0.3, 0.5/0.3 and 0.2/1.3 ms:
candidate reduction does not guarantee faster joins when piece scanning dominates.
First-use execution includes shader compilation and took 2.87–3.20 seconds.
See <code>geometry-scaling-evidence.json</code> beside the oracle fixtures for the complete
measurement record. These are observations on one device, not a universal envelope.</p>
<p>The circle's remaining error bounds geometric search discretization, excluding
float32 rounding. Ratio hulls return area boundaries, so point and collinear
results have no area edges. Segment snapping deliberately uses the documented
nearest/tie/insertion rule, without GEOS's optional safety heuristics or a universal
topology-preservation guarantee. These are explicit contracts rather than silent
CPU fallbacks.</p>`};export{e as default};