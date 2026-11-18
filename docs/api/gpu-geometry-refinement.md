# GPU geometry refinement

The contributors in `@luma-contributors/analysis/gpu-spatial-analysis` add bounded geometry
processing to caller-owned `GPUCommandGraph` buffers. They return command nodes;
they never submit, read back or run a CPU geometry fallback. All geometric
measurements use planar coordinates. Use a local coordinate frame for numerical
stability. Output capacities and compile-time budgets belong to the caller.

| Contributor | Input and output contract | Limits and status |
| --- | --- | --- |
| `GPUNormalizePolygons` | Four-level polygon layout → canonical coordinates, ring offsets and polygon offsets. Reuse input feature offsets. Shells become clockwise, holes counterclockwise; rings start at their smallest vertex; holes and parts sort descending. | Finite inputs, valid offsets. Feature order, empty parts, repeated vertices and closure conventions are preserved. Use `GPUNormalizeGeometry` for mixed collections and line/point normalization. |
| `GPUCanonicalPolygonEquality` | Pair corresponding normalized features; compare structure and coordinates without hashes. `tolerance` is absolute per-coordinate tolerance. | Output 1 equal, 0 unequal, 2 nonfinite. Canonical structure and closure must agree. |
| `GPUSubdividePolygons` | Triangulate polygon interiors, then merge adjacent triangles into convex pieces of at most `maximumVertices` (3–128). A piece occupies a fixed block of that many position slots. | `pieces` rows are `[vertexCount, sourceFeature, sourcePolygon, sourceSeedTriangle]`; `status` is `[written, required, capacityOverflow, invalidPolygonCount]`. Holes remain excluded from the piece union. Rectilinear leaf offsets and the required piece total saturate at `0xffffffff`; capacity overflow preserves the output prefix. `method: 'rectilinear'` uses a bounded GPU queue to split convex seeds by axis-aligned clips (minimum limit 4). Bit 31 of the final status word reports incomplete splitting. The seed decomposition and piece ordering differ from PostGIS; the union and per-piece vertex bound are the comparison contracts. |
| `GPULinePrecisionReduction` | Grid-round lines, remove successive repeats, preserve source line slots. | `valid-output` drops collapsed lines; `keep-collapsed` retains two equal endpoints for collapsed input lines. Per-line status 0 regular/empty, 1 dropped collapse, 2 kept collapse, 3 nonfinite. Output position capacity must cover the input length. |
| `GPUPolygonPrecisionReduction` | Materialize ring closures, hot-pixel node boundaries and reconstruct even-odd polygon faces. Reuses bounded `GPUPolygonizeOutput`, including provenance and remnants. | Any positive finite float32 `gridSize`; `lattice = [originX, originY, actualStep, 0]` may report a larger step for large extents. Both collapse modes drop collapsed polygons. The topology operates in grid-normalized units and converts coordinates, provenance endpoints and the actual lattice back to input units. |
| `GPUSegmentSnap` | Snap endpoints to nearest references, then insert nearby unique references in segment projection order. Equal-distance ties choose the first reference row. | Fixed `maximumVerticesPerLine` blocks; `counts` are `[written, required, overflow, nonfinite]`. Optional `issueCode` bits report collapse (1), nonfinite input (2) and incomplete capacity or candidates (4); decode with `decodeGPUSegmentSnapIssueCode` or `GPU_SEGMENT_SNAP_ISSUE`. Optional `validity` is 1 only when no issue is present. Empty lines are usable; self-crossing lines remain usable. Inclusive planar tolerance. GEOS safety heuristics are not implemented and topology validity is not certified. |
| `GPULineLoopRemoval` | Exact-predicate noding followed by chronological loop erasure for open curves, including prepared raw offsets. Direction, source line IDs through the offset layout, and original edge IDs are retained. | `status` is `[writtenVertices, requiredVertices, outputOverflow, incompleteNoding]`. Whole lines that exceed remaining output capacity are omitted. Use `GPUOffsetCurveComponents` for distance-band clipping and multipart component selection. |
| `GPURatioConcaveHull` | Delaunay border erosion using `minimumEdge + ratio * (maximumEdge - minimumEdge)`, independent of alpha-complex circumradius. Erosion runs in parallel rounds (each round removes every removable border triangle that is the largest around its three vertices), so the hull is a valid erosion hull for the ratio but not the largest-first sequence of GEOS `ConcaveHull`: it can keep different triangles than `shapely.concave_hull`. Optional interior erosion seeds holes. | Uses the same masked triangle/boundary-edge shape as `GPUConcaveHull`. `converged` covers tessellation and erosion budget. Ratio 1 yields the convex area hull. Point/collinear sets have no area boundary. |
| `GPUMaximumInscribedCircle` | Per-feature best-first search returns `(centerX, centerY, radius, remainingSearchError)`, nearest boundary witness, and status. Holes and multipart boundaries use even-odd fill. | 0 tolerance success, 1 cell/iteration exhaustion, 2 empty, 3 nonfinite. The remaining bound includes all unresolved cells. It bounds geometric search discretization, **not floating-point roundoff**; this is not an interval-arithmetic certificate. |
| `GPUConvexStraightSkeleton` | Parallel edge wavefronts return skeleton arcs, original bounding edge IDs, and an inward polygon offset at `offsetDistance`. Simultaneous rectangle collapses produce a terminal ridge. | Strictly convex simple polygons, up to 64 vertices. Status `[writtenArcs, requiredArcs, offsetVertices, flags]`: unsupported 1, capacity 2, budget 4, numerical 8. Holes, nonconvex split events and collinear corners remain unsupported. |


| `GPUNormalizeGeometry` | Postorder tree with point, line, ring, polygon, multi and collection nodes; stable source IDs remain in the input node table. Normalize leaf direction/winding and sort children by GEOS type, coordinate count and coordinates. | Nesting at most 64; empty nodes remain explicit. Status per node: 0 success, 1 nonfinite leaf, 2 invalid structure/range/depth, uncertain orientation or invalid child. Leaf position ranges must be disjoint. |
| `GPUCanonicalGeometryEquality` | Compare pairs of successful normalized tree roots with explicit absolute coordinate tolerance. | 1 equal, 0 unequal, 2 invalid structure/nonfinite. Geometry structure matters; source IDs do not participate. |
| `GPUOffsetCurveComponents` | Node prepared raw offsets, analytically split against source segment distance bands, reject covered and collapsed fragments, then assemble directed multipart curves. | `original` and `raw` have aligned source line order/direction. `rowsPerVertex` maps join padding to original source edges. Status rows: `[writtenComponents,writtenVertices,requiredComponents,requiredVertices]`, `[overflow,incompleteNoding,0,0]`. Only complete components are published. |
| `GPUVectorStraightSkeleton` | Linked wavefronts process edge collapse, reflex split, hole merge, collinear motion and simultaneous terminal ridges. Return skeleton arcs and original bounding edge IDs, plus linked inward-offset loops. | Valid planar polygons with holes; maximum 1024 original vertices and 4096 events per polygon. Status `[writtenArcs,requiredArcs,activeOffsetVertices,flags]`: input 1, capacity 2, work 4, numerical 8. `eventTolerance` controls float32 event grouping. |

## Composition

**Breaking (2026-11-17):** segment snap's former diagnostic `output.validity` is now
`output.issueCode`; `output.validity` has nonzero-usable semantics. The contributor, GPU operation
and CPU twin share both ports. `computeSegmentSnapOnCPU` returns both arrays.
The third `counts` word is final vertex-capacity overflow only (`required > maximumVerticesPerLine`).
Candidate exhaustion alone leaves it zero; `status.candidateOverflow` reports the candidate loss,
while `issueCode` bit 4 and `validity = 0` identify suppressed incomplete lines. Growing the vertex
output cannot repair a candidate budget. Conversely, final vertex truncation sets the third word
even when every candidate fits.
When grid suppression is possible, construction requires at least one per-line signal:
`output.validity`, `output.issueCode`, or `output.status.candidateOverflow` (the legacy alias is
also accepted). This applies to the contributor and both operation forms. Suppression is possible
when input positions are nonempty, reference count exceeds `candidateCapacity`, and either policy
is `incomplete` or the exhaustive work bound is exceeded. Required outputs alone remain sufficient
for empty input, a list that can hold all references, guaranteed bounded fallback, or explicit
brute force. Policy, input references and nested output/status maps are captured at construction;
later caller mutations do not change the declared operation.
Nonfinite query lines have no candidate search: their `candidateOverflow` is zero, their
`issueCode` includes nonfinite (2), and their usability is zero. All output views are retained at
construction; replacing the caller's output map does not redirect later stages. Disjoint aligned
counts and diagnostic views may share a buffer; the publication pass uses compatible read-write
storage bindings for that layout.

Normalize each operand before canonical equality; structural equality requires the
same representation and precision. To compare geometry after precision reduction,
check topology status, normalize the reconstructed outputs and then compare them.
Tree ring winding uses the shared exact extreme-vertex orientation predicate,
avoiding float32 signed-area cancellation after large translations. An uncertain
predicate result propagates invalid status to the containing geometry.
A zero polygon count with overflow set is incomplete work, not an empty geometry.

Subdivision metadata preserves both feature and polygon identity. Join each piece
against the query set, map results back to `sourceFeature`, and deduplicate feature
pairs. Unused fixed position slots are padding and must never be interpreted as
vertices. Successful piece unions reconstruct successful source triangulations;
failed polygons are explicitly counted.

`GPULineLoopRemoval` consumes explicit open curves. To use it after `GPUOffsetCurve`,
scale the input line offsets by `getGPUOffsetCurveRowsPerVertex` and pass the raw
output positions. Repeated join padding is removed during noding/traversal. For complete offset selection, pass the raw curve and original line to
`GPUOffsetCurveComponents`, including the signed distance and an absolute tolerance
covering round-join chord error. Mitre, bevel and round joins have direct reference
fixtures for both signs, inner corners, self-crossings and multipart results.

Offset components use shared endpoint coordinates for duplicate/reversed-fragment
cancellation and component chaining in both the GPU contributor and CPU twin. Both
coordinates must compare equal; proximity alone never establishes connectivity. Clipping
copies stored noded endpoints at cut parameters `0` and `1` instead of recomputing them.
The GPU materializes each interior cut coordinate once and shares it between adjacent
fragments. Accepted nonzero intervals are retained even below `1e-7` of a segment's length;
discarding those intervals could leave numerical gaps in a continuous component.
Circle cuts store their geometric intersection directly, rather than reconstructing it
from a rounded parameter. In particular, this keeps tangent endpoints identical on
reversed traversals. Equal cut parameters share one stored coordinate.
This preserves tiny shared endpoints of long edges, keeps narrow hairpins distinct,
and prevents chaining across intervals removed by clipping. Endpoint matching uses no
distance tolerance or coordinate-unit floor; the distance-band `tolerance` applies only
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
and source-edge provenance; a minimal vertex count is not guaranteed.

`GPUMaximumInscribedCircle` stores its live cells in graph-owned scratch storage,
using `featureCount * cellCapacity * 16` bytes. The search supports up to 4096 live
cells per feature and 65536 splits; insufficient capacity or work reports the
unresolved geometric error.

## Representation and bounded work

A `GPUCanonicalGeometryTree` stores `nodes: uint32x4` rows as
`[type, firstVertexOrChildSlot, vertexOrChildCount, stableSourceId]` and a separate
`children: uint32` adjacency list. `GPU_CANONICAL_GEOMETRY_TYPE` gives the GEOS
comparison order. Child IDs precede their parent, and polygon children are shell
first followed by holes. Reuse the original node buffer with normalized positions
and child IDs. Normalization preserves the source IDs and explicit empty nodes.

Rectilinear subdivision first constructs a convex decomposition of the polygon
interior, including holes, then clips oversized seeds at the midpoint of their
longest bounding-box axis. A depth-first storage queue has 128 slots;
`maximumSplits` (default 4096, maximum 65536) bounds operations per seed. Collinear
clip vertices are removed. Every successful leaf has at most `maximumVertices`
vertices and carries feature/polygon provenance. Incomplete status invalidates
any claim that the output union reconstructs the entire input.

Offset component scratch is explicit: with `S = segmentCapacity`,
`C = 6 * original.positions.length + 2`, cut parameters and shared coordinates occupy `12SC` bytes and fragments
plus provenance occupy `32S(C - 1)` bytes, in addition to noding storage. Selection
and component assembly scan this bounded storage; callers should use capacities
appropriate to the workload. The signed distance belongs to raw-offset generation;
the selector assumes those directed raw offsets were generated from the aligned
source lines. It does not repair arbitrary unrelated linework.

`CPUOffsetCurveComponentsOperation` honours the same explicit noding budgets as its GPU
form. `segmentCapacity` retains at most that many noded segments in traversal order;
loss sets status word 5 (`incompleteNoding`) while final-output `overflow` remains zero
unless output capacity also drops rows. When the intersection-pair count exceeds
`intersectionCapacity`, the CPU twin suppresses all geometry and sets `incompleteNoding=1`.
The GPU may retain a spatial-index-dependent pair prefix; the CPU does not replay that
partial ordering. Component/vertex requirements after noding loss describe the retained
result, not the missing complete geometry. Adjacent segments meeting only at their shared
vertex consume no pair capacity; adjacent overlaps do. The standalone
`computeOffsetCurveComponentsOnCPU` accepts both capacities optionally and remains
uncapped when they are omitted. Its exact rational noding does not impose the GPU's
storage-binding limits or successor-scan bound.

The vector skeleton uses 76 bytes per internal wavefront slot and one 4-byte count
per polygon. When source-position ranges use fewer slots than maximum-vertex
blocks, internal initial-vertex storage follows those ranges (including closing
vertices); each polygon additionally reserves `2 * maximumEvents` event slots.
Otherwise initial-vertex storage reserves `maximumVertices` slots per polygon.
Slots are initialized when vertices are created, so unused event capacity does
not add a full clearing pass. Caller-owned offset outputs retain fixed
`maximumVertices + 2 * maximumEvents` blocks per polygon.

An ascending active-vertex list skips retired vertices during event searches.
Stable nondegenerate components are measured once per event. A conservative
rounding envelope keeps the original ordered checks for components close enough
to the zero-area tolerance that rotating the sum could change its classification.
Edge and split candidate comparisons retain their original ordered tolerance
rules. Polygons execute independently in parallel; each polygon's event chain
remains sequential, and reflex/edge search can still require quadratic work per
event. These changes reduce storage and redundant work without promising scalable
latency for one arbitrarily complex polygon.

`offsetLinks` contains previous/next slot IDs local to that polygon's block;
`0xffffffff` marks inactive slots. Traverse active cycles to recover multiple
inward offset components, including holes. Invalid/self-intersecting polygon
repair is a separate operation. Event ordering is float32 and tolerance-based,
not an exact-arithmetic skeleton guarantee.
The convex helper compacts surviving edges in place and reuses event-position
storage for new trace origins, avoiding three additional per-invocation arrays.
Its simultaneous-event grouping and source-edge attribution are unchanged.

## Independent references and scaling evidence

`geometry-refinement.py` and `geometry-advanced.py` directly execute pinned
Shapely 2.1.2; fixtures record GEOS and NumPy versions. Tests cover nested and empty
geometry trees, normalization idempotence, arbitrary grids 0.1/0.3/2.5, polygon
collapse policies, 36 signed offset/join cases, 42 ratio hull endpoint/degenerate
cases, and circle success/exhaustion bounds. Offset assertions include component
count, length, bidirectional coordinate distance and open-curve orientation.
Subdivision compares area and 841 independent Shapely membership queries per
fixture, including holes and concavities. Direct PostgreSQL 16.6/PostGIS 3.5.3
fixtures are generated by `geometry-subdivision-postgis.py` against GEOS 3.14.1.
The twelve `ST_Subdivide` cases use closed-ring limits 5/9/17, validate `ST_Equals`
union reconstruction, `ST_IsValid`, vertex limits and 841 `ST_Covers` queries each.
The GPU tests combine all four source geometries in one batch at each limit and
verify per-feature provenance, area and union membership. Engine pieces omit the
closing duplicate, so their corresponding limits are 4/8/16. Different partitions
and piece ordering are permitted; the direct-reference union contract is verified.

`geometry-straight-skeleton.cpp` calls CGAL 6.0.1 directly using
`Exact_predicates_inexact_constructions_kernel`, with Boost 1.87.0 as the isolated
number backend. `geometry-straight-skeleton.py` records ten complete arc and inward
offset fixtures: rectangle, triangle, reflex notch, concave L/C, one or two holes,
collinear corners and asymmetric events. The GPU tests match the full arc sets and
traverse reciprocal offset links to compare complete closed rings. Reference dependencies remain external; no CGAL implementation
has been copied into shipped source. Reproduction commands are in the oracle file.
`straight-skeleton-storage.node.spec.ts` also executes these kernels against the
CGAL fixtures on the CPU interpreter, checks packed mixed-polygon storage, and
reuses the same graph after shrinking source ranges to detect stale snapshot links.

`geometry-scaling.spec.ts` executes subdivision, inscribed circles and concave
straight skeletons on batches of 16, 64 and 256 polygons (96–1536 vertices), checks
every completion status, and times upload, construction, compilation, first/warm
execution, changed inputs and readback separately. The selective GPU bounding-box
join has 50% fewer candidates after subdivision at all three scales. On the
recorded Apple/Metal WebGPU run, warm combined geometry times were 6.2, 5.4 and
12.9 ms. Warm unsplit/subdivided join times were 0.6/0.3, 0.5/0.3 and 0.2/1.3 ms:
candidate reduction does not guarantee faster joins when piece scanning dominates.
First-use execution includes shader compilation and took 2.87–3.20 seconds.
See `geometry-scaling-evidence.json` beside the oracle fixtures for the complete
measurement record. These are observations on one device, not a universal envelope.

The circle's remaining error bounds geometric search discretization, excluding
float32 rounding. Ratio hulls return area boundaries, so point and collinear
results have no area edges. Segment snapping deliberately uses the documented
nearest/tie/insertion rule, without GEOS's optional safety heuristics or a universal
topology-preservation guarantee. These are explicit contracts rather than silent
CPU fallbacks.
