# GPU geometry refinement

The contributors in `@luma-contributors/analysis/gpu-spatial-analysis` add bounded geometry
processing to caller-owned `GPUCommandGraph` buffers. They return command nodes;
they never submit, read back or run a CPU geometry fallback. All geometric
measurements use planar coordinates. Use a local coordinate frame for numerical
stability. Output capacities and compile-time budgets belong to the caller.

| Contributor | Input and output contract | Limits and status |
| --- | --- | --- |
| `GPUNormalizePolygons` | Four-level polygon layout → canonical coordinates, ring offsets and polygon offsets. Reuse input feature offsets. Shells become clockwise, holes counterclockwise; rings start at their smallest vertex; holes and parts sort descending. | Finite inputs, valid offsets. Feature order, empty parts, repeated vertices and closure conventions are preserved. Use `GPUNormalizeGeometry` for mixed collections and line/point normalization. |
| `GPUCanonicalPolygonEquality` | Pair corresponding normalized features; compare structure and coordinates without hashes. Absolute `tolerance` uses `toleranceMetric: 'per-coordinate'` by default; `'euclidean'` is also supported. | Output 1 equal, 0 unequal, 2 nonfinite. Canonical structure and closure must agree. |
| `GPUSubdividePolygons` | Triangulate polygon interiors, then merge adjacent triangles into convex pieces of at most `maximumVertices` (3–128). A piece occupies a fixed block of that many position slots. | `pieces` rows are `[vertexCount, sourceFeature, sourcePolygon, sourceSeedTriangle]`; `status` is `[written, required, capacityOverflow, invalidPolygonCount]`. Holes remain excluded from the piece union. Rectilinear leaf offsets and the required piece total saturate at `0xffffffff`; capacity overflow preserves the output prefix. `method: 'rectilinear'` uses a bounded GPU queue to split convex seeds by axis-aligned clips (minimum limit 4). Bit 31 of the final status word reports incomplete splitting. The seed decomposition and piece ordering differ from PostGIS; the union and per-piece vertex bound are the comparison contracts. |
| `GPULinePrecisionReduction` | Grid-round lines, remove successive repeats, preserve source line slots. | `valid-output` drops collapsed lines; `keep-collapsed` retains two equal endpoints for collapsed input lines. Per-line status 0 regular/empty, 1 dropped collapse, 2 kept collapse, 3 nonfinite. Output position capacity must cover the input length. |
| `GPUPolygonPrecisionReduction` | Materialize ring closures, hot-pixel node boundaries and reconstruct even-odd polygon faces. Reuses bounded polygonization geometry, provenance and remnant diagnostics. | Any positive finite float32 `gridSize`; `lattice = [originX, originY, actualStep, 0]` may report a larger step for large extents. Both collapse modes drop collapsed polygons. The topology operates in grid-normalized units and converts coordinates, provenance endpoints and the actual lattice back to input units. At all three status levels, `uncertainCount` is the optional predicate counter; `invalidCount`, `iterationCount`, `converged`, `iterationLimitReached` and `approximation` are refused because the topology pipeline does not publish them. Omit those fields and inspect `candidateOverflow` and `uncertainCount` for incomplete input. Intermediate `nodedSegments.status` and `directedEdges.status` accept `requiredCount` for segment/edge demand and reject `requiredRingCount`, including inherited destinations. Only `output.status.requiredRingCount` reports final ring demand. Empty rings do not contribute lattice candidates or loss. Both precision routes certify half-up cells from original float32 coordinates and the grid step, including negative half ties; already divided coordinates are not used to decide a tie. Polygon cells from 2^23 publish the certified integer when the half boundary is not representable; line precision returns an input that is already an exact grid multiple unchanged from 2^24 cells (other inputs there keep the uncertified quotient candidate). |
| `GPUSegmentSnap` | Snap endpoints to nearest references, then insert nearby unique references in segment projection order. Equal-distance ties choose the first reference row. | Fixed `maximumVerticesPerLine` blocks; `counts` are `[written, required, overflow, nonfinite]`. Optional `issueCode` bits report collapse (1), nonfinite input (2) and incomplete capacity or candidates (4); decode with `decodeGPUSegmentSnapIssueCode` or `GPU_SEGMENT_SNAP_ISSUE`. Optional `validity` is 1 only when no issue is present. Empty lines are usable; self-crossing lines remain usable. Inclusive planar tolerance. GEOS safety heuristics are not implemented and topology validity is not certified. |
| `GPULineLoopRemoval` | Exact-predicate noding followed by chronological loop erasure for open curves, including prepared raw offsets. Direction, source line IDs through the offset layout, and original edge IDs are retained. | `status` is `[writtenVertices, requiredVertices, outputOverflow, incompleteNoding]`. Whole lines that exceed remaining output capacity are omitted. Use `GPUOffsetCurveComponents` for distance-band clipping and multipart component selection. |
| `GPURatioConcaveHull` | Delaunay border erosion using `minimumEdge + ratio * (maximumEdge - minimumEdge)`, independent of alpha-complex circumradius. Erosion runs in parallel rounds (each round removes every removable border triangle that is the largest around its three vertices), so the hull is a valid erosion hull for the ratio but not the largest-first sequence of GEOS `ConcaveHull`: it can keep different triangles than `shapely.concave_hull`. Optional interior erosion seeds holes. | Uses the same masked triangle/boundary-edge shape as `GPUConcaveHull`. `converged` covers tessellation and erosion budget. Ratio 1 yields the convex area hull. Point/collinear sets have no area boundary. |
| `GPUMaximumInscribedCircle` | Per-feature best-first search returns `(centerX, centerY, radius, remainingSearchError)`, nearest boundary witness, and status. Holes and multipart boundaries use even-odd fill. | 0 tolerance success, 1 cell/iteration exhaustion, 2 empty, 3 nonfinite. The remaining bound includes all unresolved cells. It bounds geometric search discretization, **not floating-point roundoff**; this is not an interval-arithmetic certificate. |
| `GPUConvexStraightSkeleton` | Parallel edge wavefronts return skeleton arcs, original bounding edge IDs, and an inward polygon offset at `offsetDistance`. Simultaneous rectangle collapses produce a terminal ridge. | Strictly convex simple polygons, up to 64 vertices. Status `[writtenArcs, requiredArcs, offsetVertices, flags]`: unsupported 1, capacity 2, budget 4, numerical 8. Holes, nonconvex split events and collinear corners remain unsupported. |
| `GPUNormalizeGeometry` | Postorder tree with point, line, ring, polygon, multi and collection nodes; stable source IDs remain in the input node table. Normalize leaf direction/winding and sort children by GEOS type, coordinate count and coordinates. | Nesting at most 64; empty nodes remain explicit. Status per node: 0 success, 1 nonfinite leaf, 2 invalid structure/range/depth, uncertain orientation or invalid child. Leaf position ranges must be disjoint. |
| `GPUCanonicalGeometryEquality` | Compare pairs of successful normalized tree roots with absolute `tolerance` and `toleranceMetric` set to `'per-coordinate'` or `'euclidean'` (default `'per-coordinate'`). | 1 equal, 0 unequal, 2 invalid structure/nonfinite. Geometry structure matters; source IDs do not participate. |
| `GPUOffsetCurveComponents` | Node prepared raw offsets, analytically split against source segment distance bands, reject covered and collapsed fragments, then assemble directed multipart curves. | `original` and `raw` have aligned source line order/direction. `rowsPerVertex` maps join padding to original source edges. Status rows: `[writtenComponents,writtenVertices,requiredComponents,requiredVertices]`, `[overflow,incompleteNoding,iterationLimit,0]`. Only complete components are published. |
| `GPUVectorStraightSkeleton` | Linked wavefronts process edge collapse, reflex split, hole merge, collinear motion and simultaneous terminal ridges. Return skeleton arcs and original bounding edge IDs, plus linked inward-offset loops. | Valid planar polygons with holes; maximum 1024 original vertices and 4096 events per polygon. Status `[writtenArcs,requiredArcs,activeOffsetVertices,flags]`: input 1, capacity 2, work 4, numerical 8. `eventTolerance` controls float32 event grouping. |

For polygon precision, `output.nodedSegments.sourceEdgeIds` identifies the original input
position row at the start of each edge, including the last distinct vertex of an implicitly
closed ring. Materializing closures and removing unused input padding preserve those identities.
For a nonbinary precision step, normalization compares the stored float32 coordinate against
the exact half-grid boundary before rounding, so division roundoff cannot collapse a surviving
face by moving an original vertex across that boundary. Lattice sizing uses unperturbed bounds;
the half-grid correction applies only to snapping, after the actual lattice step is selected.
Precision props and nested input/output records are captured at construction. Replacing a member
on the caller's records later does not redirect the validated graph views. Frame metadata getters
are read once and nested origin, area-of-use and unit values are copied. Final polygon status
refuses `requiredCount`; use `requiredRingCount` for ring capacity.
Precision polygonization recursively peels degree-one fringes as dangles. Internal edges exposed
by peeling are dangles too; cut edges are the bridges that remain after that pruning.
Diagnostics count noded source-segment occurrences, including repeated traversals: a three-edge chain
traversed out and back contributes six dangles and zero cuts. A bridge between two square loops
traversed in both directions contributes two cut occurrences. `closedRingCount` likewise counts
source segments belonging to bounded polygon faces, rather than the number of rings.
For one isolated square, `closedRingCount` is four source edges, final `count` is one polygon
feature, and `requiredRingCount` is two directed walks (bounded interior and exterior).

CPU skeletons honor explicitly supplied iteration/event budgets after offset capture and before
arc emission: zero work can retain an early offset snapshot while reporting `[0,0,offsetCount,4]`.
The CPU circle search honors explicit iteration and cell budgets before subdivision, retains its
last-vertex fallback if the bounds-center seed has not improved it, and reports status 1 with the
remaining search bound at exhaustion. The CPU ratio hull counts triangle removals, including a
whole simultaneous round or none; exhausted work leaves the current mesh with `converged=0`.
An explicit zero removal budget can still report `converged=1` when no eligible removal exists:
a three-site simplex must retain its sole triangle, and a collinear set has no triangle to erode.
For sites `(0,0),(6,0),(0,4),(1,1)` at ratio zero, the unique longest exterior edge is
between sites 1 and 2. Budget zero keeps all three Delaunay triangles with `converged=0`;
budget one invalidates triangle `{1,2,3}`, retains `{0,1,3}` and `{0,2,3}`, and converges.
Circle cell capacities 1–3 can hold the root but cannot subdivide it; when the root remains
unresolved they report status 1 and retain its error bound. CPU execution and GPU operation
lowering capture work options at construction; mutating the caller's budget later has no effect.
Pure CPU functions remain uncapped when their corresponding optional work limits are omitted.

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
Both canonical equality contributors accept `toleranceMetric: 'per-coordinate' | 'euclidean'`;
unknown strings reject at construction. The existing default, `'per-coordinate'`, requires each
absolute XY difference to be at most `tolerance`. `'euclidean'` requires the planar distance of
each corresponding XY pair to be at most `tolerance`, matching the metric of
`GPUGeometryPredicates`'s `equalsExact` comparison. Tolerance has the same unit as the planar
positions. At tolerance 1, displacement `(0.75, 0.75)` passes per-coordinate comparison but
fails Euclidean comparison because `sqrt(0.75² + 0.75²) ≈ 1.06066`. Choose the metric explicitly
when switching between normalization-based equality and `equalsExact`; normalization and the
distance metric are independent choices. This option leaves the existing output codes unchanged.
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

Loop removal accepts `lines.featureOffsets` and optional `output.featureOffsets`; logical
features retain the same part rows, including erased or capacity-dropped empty parts. Loop
erasure restarts at each part and never connects disconnected parts. Both operation forms accept
the same ports and require `featureCount` with input feature offsets. With ungrouped input they
infer an optional output identity hierarchy from `lineCount`; the direct CPU function requests
the same hierarchy with `includeFeatureOffsets: true`. The CPU function accepts optional
`featureOffsets` and an optional `intersectionCapacity`: an explicit pair budget binds exact CPU
noding too. Exhausting it suppresses CPU geometry and sets status word 4 (incomplete noding) to 1;
the GPU can retain an incomplete prefix with the same flag. Omitting the CPU budget is uncapped.

Terminal-only ungrouped input `lineOffsets=[0]` with zero positions is also accepted by the
direct CPU loop-removal function, returning offsets `[0]` and all-zero status. Track similarity
still requires at least one ungrouped part. With grouping, both offset arrays must start at zero;
the terminal feature offset selects the active part prefix, and capacity padding must repeat its
terminal vertex offset. Both CPU consumers reject nonzero grouped starts or nonconstant padding.

`GPUBufferSurface` accepts multiline `geometry.featureOffsets`. Each part generates its own
two-sided stroke and caps, then the existing dissolve unions all strokes. Grouping preserves
logical feature ownership in boundary provenance and optional representative `featureSources`;
part rows and stable caller IDs are not substituted for feature rows. An empty input feature
contributes no boundary. Empty parts and parts stored with only one vertex reserve no logical
raw-boundary rows, so they cannot change the snap lattice of another part. A stored line with at
least two finite vertices that deduplicates to one point retains its square or round cap at a
positive distance; a flat cap has zero area. Nonfinite ring-owned or line-owned vertices set both candidate
overflow flags before compaction, even if deduplication removes them or their part has no stroke.
Unused leading/trailing position padding is not input and does not set this flag. Correcting the
owned coordinates clears the flag on the next submission. Dissolved output polygons still follow
the existing union semantics and can combine several input features; inspect the mixed-provenance flag when needed.
Buffer `status.ringCount` and `requiredRingCount` count assembled shells plus holes. Two
disconnected rectangle strokes without holes therefore write and require two rings, even though
a polygonize face-cycle inventory can also include exterior cycles.

For buffer boundaries, `sourceFeatureIds` publishes caller feature IDs (logical rows when no
IDs are supplied); optional `sourceFeatureRows` retains the logical row. Representative
`featureSources.sourceRows` remains a row, independent of caller IDs. The buffer adapts the
overlay's internal row provenance without changing the overlay contributor's public port.

`GPUOffsetCurveComponents`, its GPU operation and CPU twin accept optional `featureOffsets`
on both `original` and `raw`. Both arrays must encode identical feature-to-line-part grouping,
including empty features; raw and original parts stay aligned. As in GEOS, each part is offset
independently. Component metadata is `[firstVertex, vertexCount, sourceLine, sourceFeature]`:
`sourceLine` is the input part row and `sourceFeature` is its logical feature row. Without
`featureOffsets` the fourth word remains 0. Output capacities and completeness status continue
to count components and vertices, so multiple components may belong to one feature.

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

### Circle outputs and coordinate precision

`circles` has one float32 row `[centerX, centerY, radius, remainingSearchError]` per source
feature; `witnesses` has one XY boundary point and `status` one code in the same feature domain.
Coordinates, radius and error use the input planar unit. The fourth component bounds unresolved
geometric search work, not the precision of the published center, witness or radius. Increasing
cell capacity or decreasing tolerance cannot create a float32 coordinate between adjacent values.

The CPU twin recomputes boundary clearance and its nearest witness at the published float32
center, using that clearance as the radius. For a square with corner `(1048576,1048576)` and side
`0.125`, the true center is `(1048576.0625,1048576.0625)` and true radius `0.0625`. No float32
center lies strictly inside that square: coordinate spacing is `0.125`. At tolerance `0.000002`,
the CPU result is `circles = [1048576, 1048576, 0, 0]`, `publishedCenterClearance = [0]`,
`precisionLimited = [1]` and legacy `status = [0]`. The published center is a boundary corner,
so its radius is zero. The fourth component is zero because the geometric search has no unresolved
work; publication loss does not become search error. This corrects the earlier positive radius
at a rounded boundary center and can reduce a caller's returned radius.

If rounding places a searched interior center outside the polygon, the CPU twin falls back to
a representable source boundary vertex and recomputes clearance there, yielding radius zero.
Its additive per-feature `precisionLimited` word reports reduced clearance from center publication
or loss of absolute-witness precision. The legacy codes remain 0 (search completed), 1 (search
budget or resolution limit), 2 (empty) and 3 (nonfinite input); status 0 alone does not certify
that the published center remains strictly interior.

The CPU result returns `witnessOffsets`, one float32 XY row per feature: nearest boundary witness minus the published center, in the input position unit. Raw GPU and named CPU/GPU circle forms accept optional `publishedCenterClearance` (float32), `precisionLimited` (uint32) and `witnessOffsets` (float32x2), all in the feature row domain. Clearance and witness are evaluated at the actual written float32 center; an exterior rounded center falls back to a representable source boundary vertex. Empty/nonfinite rows have NaN clearance/relative witness and precision flag 0. Use relative offsets when the absolute witness rounds onto the center; subtracting the already rounded absolute witness cannot recover that displacement. GPU search runs in a local frame and retains its cell-search bound in component four; CPU additionally caps bounds by half the shorter envelope side, so the two search-error values need not agree.

GPU `precisionLimited` is also 1 when the stored witness on an oblique edge is not certified,
by exact integer arithmetic, as that edge's closest point to the stored center (its float32
coordinates cannot represent the foot, or the exponents span more than 384 bits). Vertex and
axis-aligned witnesses are exact. Both routes project from the nearer edge endpoint.

Canonical zero-row circle, polygon-normalization and segment-snap operation forms are
supported. Offset tables retain one terminal zero at every empty level; zero parents and one
declared empty parent remain distinct. An empty circle feature writes NaN geometry, status 2
and precision flag 0. A declared empty snap line with empty references writes counts `[0,0,0,0]`,
validity 1, issue code 0 and `candidateOverflow` 0; zero lines produce zero per-line rows. Compact
skeleton `[0]` is valid only over zero children at that level, and provenance interval slack
uses the position-unit `eventTolerance`. A terminal two-vertex arc retains both surviving source
supports, including oblique pairs; it does not require parallel opposing edges.

Subtract a local origin in binary64 **before** float32 conversion, run refinement in that local
frame, and restore the origin in binary64 after readback:

```ts
import {computeMaximumInscribedCircleOnCPU} from '@luma-contributors/analysis/gpu-spatial-analysis';

const origin = [1048576, 1048576] as const;
const sourceXY = Float64Array.of(
  1048576, 1048576, 1048576.125, 1048576,
  1048576.125, 1048576.125, 1048576, 1048576.125
);
const localXY = Float32Array.from(sourceXY, (value, index) => value - origin[index % 2]);
const localCircle = computeMaximumInscribedCircleOnCPU(
  {
    positions: localXY,
    ringOffsets: Uint32Array.of(0, 4),
    polygonOffsets: Uint32Array.of(0, 1),
    featureOffsets: Uint32Array.of(0, 1)
  },
  0.000002
);
// localCircle.circles is [0.0625, 0.0625, 0.0625, 0],
// precisionLimited [0], status [0]: the local interior center and radius are representable.
const publishedLocalCenter = localCircle.circles.subarray(0, 2);
const worldCenter = Float64Array.from(publishedLocalCenter, (value, index) => value + origin[index]);
// worldCenter is [1048576.0625, 1048576.0625]; keep it in float64.
```

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
appropriate to the workload. Reservations of 8,192 through 65,536 fragments use up to 32 cancellation or traversal rounds per workgroup dispatch. If clipping or traversal cannot finish within its bounds, word 6 (`iterationLimit`) is set and the readable component and vertex counts are zero. Use `decodeGPUOffsetCurveComponentsStatus` to read the status rows. The signed distance belongs to raw-offset generation;
the selector assumes those directed raw offsets were generated from the aligned
source lines. It does not repair arbitrary unrelated linework.
CPU and GPU selection retain each noded fragment's recorded raw edge when mapping it through
`rowsPerVertex` to the original source edge. Coincident or retraced raw traversals keep their
own identity: if raw edges 0 and 1 cancel and edge 2 survives, its vertices report source edge 2
when `rowsPerVertex` is one.

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
`0xffffffff` marks inactive slots. Coordinates in inactive slots are unspecified;
read `offsetPositions` only for slots with active links. The CPU twin zeroes unused
coordinates, while caller-owned GPU buffers can retain their previous contents there.
Traverse active cycles to recover multiple
inward offset components, including holes. Invalid/self-intersecting polygon
repair is a separate operation. Event ordering is float32 and tolerance-based,
not an exact-arithmetic skeleton guarantee.
Vector skeleton source-edge pairs identify the caller's original outgoing edges regardless of
shell/hole traversal reversal. A terminal ridge reports the opposing supporting edges that meet
along each live wavefront interval, rather than either endpoint's corner pair. Adjacent collinear
caller edges retain separate source identities on their respective ridge intervals.
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
