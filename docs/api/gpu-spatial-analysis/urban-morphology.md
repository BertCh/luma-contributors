# Urban morphology recipes

Related tasks: [landscape metrics and connectivity](./landscape-ecology.md) · [workflow recipes](./recipes.md).

These graph recipes operate on projected planar coordinates. They append GPU work to the supplied
command graph and retain caller-owned inputs. Returned graph-owned outputs stay on the GPU; callers
that need readback must add a copy into a caller-owned readback buffer, or pass an `output` prop
(see below) so the results are written into views the caller owns. Horizontal coordinates and
building heights must use compatible physical units. No recipe submits work or reads data on the CPU.

The implementations retain explicit capacities and approximation controls. Direct pinned momepy
0.10.0 fixtures cover street width and nonconvex/holed tessellation areas; independent Shapely,
NetworkX and analytical tests cover additional geometry and graph invariants. This is evidence for
the documented modes, not complete momepy or depthmapX library equivalence.

## Caller-owned outputs, limits and obstacle index

Each recipe takes an optional `output` prop. Every entry is a view the caller created in the same
graph, with the format and length of the recipe's transient view of that name; an absent entry is a
graph-owned transient view, as before. The `outputs` of the result are the views actually used.

| Recipe | `output` entries |
| --- | --- |
| `addBuildingTessellationReferenceRecipe` (or dispatcher's reference route) | `overlays` (one final `GPUPolygonOverlay` output per building), `samples`, `owners`, `fragments`, `fragmentCounts`, `samplingStatus`, `complete` |
| `addLocalBuildingTessellationRecipe` (or dispatcher's local route) | `positions`, `cells`, `buildingStatus`, `samplingStatus`, and optional `status` scalars; fixed convex-piece slots, with [local certification and loss evidence](#local-certification-and-loss-evidence) |
| `addStreetProfilesRecipe` | `profiles`, `buildings`, `validity`, `canyon`, `frontage`, and with `streetSegment` `facadeLengths`, `facadeFrontage` |
| `addBuildingAdjacencyRecipe` | `pairs`, `lengths`, `morphology`, `components`, `complete`, and with `structures` `structurePerimeters` |
| `addIsovistsRecipe` | `endpoints`, `distances`, `segments`, `validity`, `localMeasures`, `angles`, `viewpointClassifications` |
| `addVisibilityGraphRecipe` reference route | `adjacency`, `edges`, `metrics`, `choice`, `integration`, `normalizedChoice`, `viewpointClassifications`, `extendedIntegration`, `status` |
| `addVisibilityGraphRecipe` spatial `mode: 'csr'` | `offsets`, `neighbors`, `weights`, `degree`, `gridVisits`, `viewpointClassifications`, named `status` scalars |

Limits are named constants and are rejected with a message that starts with the recipe ID and names
the prop. Reference tessellation: 64 buildings (one overlay per building), 32 total enclosure vertices
and 128 total boundary samples per call (thread-private polygons and a samples-squared fragment
table). The local route has the separate limits listed below. Adjacency: 1,024 buildings
(buildings-squared pair table), and 64 buildings with `structures`. Visibility graph: 256 viewpoints
on the dense reference route; CSR adjacency allows at most 1,000,000 viewpoints and 16,777,216
directed edge slots, with at most 65,536 candidate grid visits per viewpoint query. Device binding
limits may lower these route limits. Isovists: 4,096 rays per viewpoint with `sampling: 'vertices'` and 65,536 otherwise. Street profiles:
1,024 obstacle segments with `streetSegment`. Every recipe accepts at most 16,777,216 obstacle segments.

`addIsovistsRecipe` and `addStreetProfilesRecipe` take `obstacleIndex`: `'brute-force'` tests every
segment for every ray, `'bvh'` builds a Morton-sorted segment BVH (`GPUBVH`) and traverses it, and the
default `'auto'` uses brute force up to 4,096 segments and the BVH above that. Both give identical
hits: the nearest segment wins and the lowest segment row breaks ties. Segment finiteness is checked
once per segment, not once per ray. `addVisibilityGraphRecipe` takes the same `obstacleIndex` and
applies it to its sightlines. In the isovists and the visibility graph the free-space classification
uses the same index: with the BVH each viewpoint visits only the segments that meet the horizontal ray to its right
(the ones that change its winding number or lie within the tolerance), and one reduction finds
non-finite segments. The classification is identical either way.

## `addBuildingTessellationRecipe` — UR01

Import `addBuildingTessellationReferenceRecipe` for the reference route or
`addLocalBuildingTessellationRecipe` for the local route from
`@luma-contributors/analysis/gpu-spatial-analysis`. Both have separate prop and result types.
`addBuildingTessellationRecipe` also retains its reference and local overloads. Reference results
contain dissolved polygon rings; local results contain fixed-stride convex polygon pieces with
variable vertex counts. Explicit
`mode: 'reference'` selects reference even when `candidateRadius` is present; explicit
`mode: 'local'` selects local; otherwise the presence of `candidateRadius` selects local, and its
absence selects reference.

The reference route accepts at most 64 buildings, 32 total enclosure vertices and 128 total boundary
samples per call. Local mode accepts at most 1,000,000 buildings and 16,777,216 total samples per
call; its enclosure has at most 32 total vertices and 32 rings. Each building may use at most 32
own samples and 256 candidate samples; each convex piece has at most 128 vertices. Each indexed
query attempt visits at most 8,192 rows. Local mode uses an adaptive candidate index and supports
bounded simple nonconvex or holed enclosures. Device storage limits may lower these bounds;
certification and overflow status still apply.

Both routes take closed building boundary `segments`, their `segmentBuildingIds`, a
`buildingCount`, and one `enclosure`. Segment endpoints are sampled at intervals no larger than
`sampleSpacing`; repeated segment end vertices are avoided. Sample spacing determines the boundary
approximation: halve it for a convergence check. Run separate enclosures independently.
Street-network enclosure construction and automatic building shrinking remain upstream geometry
operations.

### `addBuildingTessellationReferenceRecipe`

The reference route returns `outputs[buildingId]`, a packed polygon result from
`GPUPolygonOverlay`; its array index identifies the building. It clips each boundary-sample Voronoi
cell to the enclosure, groups fragments by building, and dissolves their internal boundaries.
Use this route when downstream consumers need dissolved shell/hole rings.

Convex counterclockwise enclosures use the default fast path. Set `enclosureMode: 'simple'` for a
nonconvex ring; `enclosureRingOffsets` describes the exterior followed by holes. Simple mode clips
cells to the bounding box, dissolves by building, then intersects with the actual enclosure through
`GPUPolygonOverlay`. Ring offsets must describe one valid polygon; the recipe does not repair it.
Holes and disconnected pieces retain the same building array index.

`sampleCapacity` is explicit and at most 128 total samples per call; the enclosure has 3–32 total
vertices and a call supports at most 64 buildings. `capacity.intersections`, `nodedSegments`,
`vertices` and `rings` separately bound overlay work and output. The recipe-generated default
`outputs[i].status` exposes `count`, `overflow`, `candidateOverflow` and `requiredRingCount`.
For caller-owned overlays, `requiredRingCount` is optional: supply a one-row `uint32` view at
`output.overlays[i].status.requiredRingCount` when the unclamped ring requirement is needed.
The supplied overlay object is returned as supplied; the recipe does not add that optional view.
Caller-owned geometry status can instead retain optional `ringCount` (written rings),
`uncertainCount` (uncertified intersection pairs) and `nodedSegmentRequiredCount` (unclamped noded
pieces, exact only with `candidateOverflow == 0`). These provide separate evidence; `ringCount`
is a written count and cannot replace the missing unclamped `requiredRingCount` for sizing.
Independently inspect `outputs[i].boundary.status.requiredCount`: it counts selected boundary
edges needed before ring assembly. For generated default buffers this fits
`capacity.nodedSegments`; for a caller-owned overlay it must fit `boundary.endpoints.length`.
The supplied boundary's `count`, `requiredCount`, `overflow` and `candidateOverflow` are required fields,
not optional alternatives. For example, four selected edges and three rows in a supplied boundary
still overflow even when `capacity.nodedSegments = 64`. Grow the supplied boundary storage and its
matching provenance views when rebuilding. This is distinct from
feature counts: `outputs[i].status.count` counts written polygon features, and this overlay has
no separate unclamped feature requirement. The boundary requirement is also distinct from the
geometry ring requirement `outputs[i].status.requiredRingCount`. Generated default ring capacity
is `capacity.rings`; supplied geometry instead uses `geometry.ringOffsets.length - 1`, with its
position and other offset views also required to fit the assembled geometry. An adequate ring or
feature capacity cannot certify an incomplete boundary-edge list. Retain all required boundary and
geometry loss words and the available requirement words; one capacity's zero-overflow flag cannot
certify the others.
The recipe-level `status` instead contains `sampling` (the packed
`[written, required, overflow, invalidSegments]` row) and `complete` (one word per building).
`status.complete[i]` combines sampling, fragment validity, dissolve and final intersection capacity
evidence; treat a polygon as complete only when that word is one.

The returned intermediates retain sample positions, owners, fragments and fragment counts.
Fragment-count `0xffffffff` denotes invalid enclosure geometry, incompatible duplicate sites or a
fragment-capacity failure; zero denotes an empty/duplicate cell. Duplicate sites owned by different
buildings invalidate the result. `vertexTolerance` controls the reference overlay snap grid and
sliver collapse independently of sample spacing. A building may have an empty cell when no sampled
support owns area within the enclosure.

See the [eight-sample reference lens](urban-ecology-lenses.md#addbuildingtessellationreferencerecipe).

### `addLocalBuildingTessellationRecipe`

The local route returns fixed `outputs.positions`, `outputs.cells` and `outputs.buildingStatus`.
A building's region is the union of its convex pieces; shared internal edges remain. These pieces
are neither dissolved polygons nor a shell/hole hierarchy. Convex enclosures use one piece slot per
own sample. In `enclosureMode: 'simple'`, the recipe triangulates the enclosure once and intersects
each sample cell with its triangles; `enclosureRingOffsets` can describe a shell and holes in either
winding. `provenance.piecesPerBuilding` gives the fixed stride.

Each `cells` row is `[vertexCount, requiredVertexCount, sourceBuildingId, sourceSampleRow]`.
The first word is the live vertex count, not a fixed triangle count. Piece slot `piece` starts at
`piece * cellVertexCapacity` in `positions`; read exactly `vertexCount` rows from that position
slice, ignoring its unused capacity tail. Unused slots have zero
counts and source sample `0xffffffff`. Read the [certification and status contract](#local-certification-and-loss-evidence)
before consuming any building's pieces; the [local lens](urban-ecology-lenses.md#addlocalbuildingtessellationrecipe)
exercises this representation.

#### Local certification and loss evidence

`candidateRadius` is measured around the midpoint of a building's sample bounding box. The adaptive
index gathers nearby sample sites; an overfull rank-grid search retries the shorter sorted
coordinate interval. A triangle-inequality / twice-cell-radius certificate must prove that every
excluded site cannot clip the published cell. Increase `candidateRadius` when that certificate
fails, and `candidateCapacity` when the larger search needs more sites. `maximumGridVisits` bounds
each indexed search attempt separately (at most 8,192 row visits), rather than all work for a
building. `vertexTolerance` is the coordinate slack for this certificate, not the reference snap
lattice.

`buildingStatus` is `[validity, overflow, candidateOverflow, invalid]`, four boolean words per
building. Consume pieces only when validity is one. Failed buildings publish zero vertex counts
for every piece and use required-vertex sentinel `0xffffffff` when uncertified. The recipe-level
scalar statuses keep causes separate:

| Status | Meaning |
| --- | --- |
| `count` | Usable building rows, whose fixed positions are identified by `buildingStatus`; not the piece count |
| `requiredCount` | Requested `buildingCount`, including uncertified rows |
| `overflow` | `cellVertexCapacity` discarded geometry |
| `candidateOverflow` | Sampling, own-sample capacity, candidate capacity, index visits or radius certification was insufficient |
| `invalidCount` | Building rows invalidated by input or non-finite clipping arithmetic |

`status.sampling` retains the packed sampling row. Increasing output vertex capacity alone cannot
resolve candidate loss. The per-call and per-building limits above still apply, and device binding
sizes can lower them. Compare the two routes geometrically at the declared coordinate tolerance,
allowing reference snap-grid quantization; raw ring order and vertex counts differ by design.

## `addStreetProfilesRecipe` — UR02

Pass centerline sample `positions`, left-pointing `normals`, building `segments`,
`segmentBuildingIds`, and `buildingHeights`. `sampleSpacing` records caller sampling provenance;
`maximumDistance` bounds each perpendicular sightline. Normals are normalized on the GPU.

- `profiles`: `[leftDistance, rightDistance, leftHeight, rightHeight]` per sample.
- `buildings`: left and right stable building IDs, or `0xffffffff` for an absent side.
- `validity`: bit 0 left hit, bit 1 right hit, bit 2 viewpoint on a boundary, bit 3 invalid input.
- `canyon`: `[observedWidth, meanHeight, heightToWidthRatio, rangeCensoredWidth]`.
- `frontage`: `[leftFrontage, rightFrontage]` per building, summing `sampleSpacing` for each valid hit.

An absent side has the finite range limit as its censored distance and NaN height. Observed canyon
width and ratio are NaN unless both sides hit and the viewpoint is valid. Missing/negative heights
preserve widths and invalidate height-dependent measures. Shared segment endpoints are closed
obstacles; collinear rays stop at the closest forward overlap. Stable segment order resolves ties.
Frontage is a sampled centerline measure: each sample represents an interval of `sampleSpacing`.
Use interval-midpoint samples to avoid double-counting endpoints; it is not exact facade arc length.
Invoke the recipe per street when separate street-level frontage is required.

For exact facade arc length along one straight street, provide `streetSegment` (one `float32x4`
start/end row). `facades.lengths` and `facades.frontage` then contain left/right visible facade length
per segment and per building. The GPU splits facade segments at corridor endpoints, range/side
crossings, obstacle endpoint projections and segment intersections, and tests visibility inside each
resulting interval. If any segment owner is outside `buildingHeights.length`, all exact facade
lengths and all building facade summaries are NaN, matching the profile input-invalid flag. An
absent side still reports `0xffffffff` as its building ID. An occluded wall contributes no facade
length. This is exact for the declared straight rectangular corridor up to the shared geometric
tolerance, rather than a sample-count
estimate. It supports at most 1,024 obstacle segments; curved streets require explicit subdivision
and ownership at joins so facade intervals are not counted twice.

A stretch of wall is visible when no other segment crosses the perpendicular from the street axis
before it. The facade kernel works relative to the street start, so the same geometry gives the same
lengths at the origin and at projected-metre coordinates (the inputs are still float32: near
4,500,000 a coordinate resolves 0.5). Two rules use a tolerance that scales with the geometry
(`tolerance` plus 1e-5 of a length):

- A wall perpendicular to the street, within that slack of its own length, is edge-on and has no
  visible length.
- Where segments coincide, the lowest segment row owns the stretch and the others report 0 there.
  Segments coincide when their distances along the sightline are within that slack (of the larger
  of the street length and `maximumDistance`) of the nearest one; a segment farther than that
  behind the nearest one is hidden. Ownership is decided against the one nearest hit, so any number
  of coincident walls has exactly one owner. Two walls that meet at a corner are each counted once.
- A segment within `tolerance` of the point where a sightline leaves the street axis blocks that
  sightline at distance 0, on either side of the axis.

## `addBuildingAdjacencyRecipe` — UR03

Compose the existing `GPUSharedPaths` result with a common `lineBuildingIds` mapping and axial
`orientations` in radians, normally from the existing shape-descriptor contributor. Supply each
unordered building-boundary pairing once; repeated runs would double-count wall length. Each dense
`pairs[first * buildingCount + second]` row contains
`[sharedWallLength, axialAngleDifference, hasSharedWall, valid]`. Axial differences wrap modulo π
into `[0, π/2]`. A point touch contributes zero wall length and is not wall adjacency. Missing
orientation affects the angle only. Any supplied upstream `overflow`, `candidateOverflow` or
`uncertainCount` evidence makes `complete` zero and suppresses the adjacency metrics as NaN.
Supplied `requiredCount` must not exceed the written run `count`; supplied `vertexCount` must fit
the run-position capacity, and `requiredVertexCount` must not exceed `vertexCount` (or the
run-position capacity when no vertex count is supplied).

Certification requires either the producer's aggregate loss word `sharedPaths.status.overflow`
equal to zero, or the full alternative set of
`requiredCount`, `vertexCount`, `requiredVertexCount`, `candidateOverflow` and `uncertainCount`.
For that alternative, the run and vertex requirements must fit the written counts, the vertex
count must fit the run-position capacity, and both candidate and uncertainty words must be zero.
All supplied checks must pass even when the aggregate word is present, and `count` must fit the
run capacity. Count alone, including zero, leaves `complete=0` and adjacency metrics suppressed
as NaN. These status views remain optional at construction; available loss evidence is always
consumed. `lengths` retains each measured shared run.

The recipe does not recompute a shape orientation or replace the shared-path topology algorithm.
The line mapping must use the same row identities for both operands, and all upstream capacity
status should be supplied when completeness is required.

`morphology` contains `[totalSharedWalls, cardinalOrientationDegrees, meanCardinalAlignmentDegrees,
structureExteriorPerimeter]`. Cardinal orientation folds the supplied axial shape direction into
0–45 degrees; alignment averages absolute cardinal-deviation differences over neighbors, with zero
for an isolate (momepy convention). This reuses the caller's existing minimum-rectangle/shape
orientation rather than fitting a second rectangle. `contiguity` optionally supplies undirected
Queen/custom building pairs; otherwise positive shared walls define neighbors. Point contacts can
therefore connect a structure without becoming shared walls. Out-of-range neighborhood IDs produce
sentinel components and undefined alignment/perimeter.

Optional `structures` accepts one packed polygon feature per building, a snap tolerance and explicit
intersection/noded-segment/vertex/ring capacities. GPU connected components drive a positive-winding
polygon union for each building's component; only exterior rings contribute to perimeter, so
courtyards are excluded. Input shells must be counterclockwise and holes clockwise. This matches the
bounded `momepy.perimeter_wall(..., buffer=0)` workflow, including isolated footprints. Desired buffer
or repair operations are explicit upstream geometry preparation. At most 64 buildings are supported
in this optional branch (1,024 for adjacency alone). `structures.overlays` returns every union's
geometry and independent capacity evidence; incomplete unions publish NaN perimeter. Returned
`components` preserves the canonical smallest building ID.

## `addIsovistsRecipe` and `addVisibilityGraphRecipe` — UR05

`addVisibilityGraphRecipe`: The `gpu-network` recipe of the same name supports budgeted exact CSR all-source metrics or
deterministic sampled estimates via metricsMode auto/exact. Auto samples when the exact command
budget cannot fit; sampled maximum depth is a lower bound. Approximation, overflow and convergence
are separate statuses; explicit exact rejects an insufficient budget.

Isovists cast `rayCount` uniformly spaced rays counterclockwise from +X, clipped by closed segment
obstacles and a finite circular `maximumDistance`. `endpoints`, `distances`, hit `segments`, and
per-ray `validity` preserve every sampled ray. `localMeasures` is
`[polygonArea, polygonPerimeter, maximumRayDistance, valid]` per viewpoint. The polygon joins adjacent
ray endpoints, but measures are computed from local `angles` and `distances` before adding the
viewpoint origin. Absolute `endpoints` are for drawing and can round away short offsets at large
projected coordinates. Four range-limited rays with radius 0.1 have area 0.02, perimeter
`0.4 * sqrt(2)` and maximum distance 0.1 even at origin `(5_000_000, 5_000_000)`.
With `sampling: 'uniform'` it can miss an obstacle between rays. With
`sampling: 'vertices'`, the GPU adds before/at/after rays at every segment endpoint and sorts all
angles; `angularEpsilon` defaults to `1e-5` radians. This resolves narrow endpoint events while
retaining uniform rays for the circular range boundary. Total rays are `rayCount + 6*segmentCount`
and at most 4,096 in this mode. Returned `angles` records the actual sampling. It remains a
controlled angular approximation, not an exact circular-arc polygon. Viewpoints on
an obstacle are invalid. Without `region`, segment input checks boundary contact only. Set
`region: {fillRule: 'even-odd' | 'nonzero', freeSpace: 'inside' | 'outside'}` to classify closed
segment rings explicitly. Outside mode models filled obstacles; inside mode models an enclosure.
Even-odd supports nested holes, while nonzero uses directed shell/hole winding. Rings must be closed
and topologically valid. `viewpointClassifications` reports 0 free, 1 blocked, 2 boundary, or 3
nonfinite input. Both isovists and visibility graphs consume this shared GPU classification.

Size each per-ray output with
`positions.length * getGPUIsovistRayCount(rayCount, segments.length, sampling)` (import the helper
from `@luma-contributors/analysis/gpu-spatial-analysis`). This applies to `endpoints`, `distances`,
hit `segments`, `validity` and `angles`; `localMeasures` and `viewpointClassifications` each have
`positions.length` rows. The helper and recipe require a base `rayCount` integer in [3, 65,536]
and a segment count integer in [1, 16,777,216]. With vertex sampling, the expanded count must
also be at most 4,096. All caller output lengths are checked before classification or BVH nodes
are added. For one viewpoint, `rayCount: 3` and one segment, vertex sampling needs nine rows in
each per-ray output. Provenance records `baseRayCount` for the uniform rays and `rayCount` for
the total; `angularSpacing` is `2π / baseRayCount`, the spacing of the uniform subset. Use the
returned `angles` for the actual, nonuniform vertex-sampled angles. `localMeasures` contains area
in squared coordinate units, perimeter and range in coordinate units, and a dimensionless validity word.

Every output must have disjoint logical byte ranges from inputs and other outputs; for example,
`angles` and `localMeasures` cannot alias the same 16 bytes even when both view lengths are correct.
Views bound together in a dispatch also need disjoint aligned binding ranges, and a physical buffer
cannot be both a read-only and a writable binding in that dispatch. When disjoint outputs share an
allocation, create their subviews on one graph buffer handle; separate imports of a physical buffer
are rejected when either handle is written. All supplied output layouts, graph ownership and alias
rules are checked before reserving transient resource IDs or adding commands. Correcting a rejected
override permits retrying with the same recipe ID on the same graph. Inputs must be single packed
views: `positions` uses `float32x2` and `segments` uses `float32x4`. Both formats and packed layouts
are checked before any output is resolved, so correcting an invalid input also permits a retry
with the same recipe ID on the same graph.
The reference tessellation and building adjacency recipes prepare the complete declaration on a
separate graph using the caller's device before adding caller commands. Before creating recipe
resources, every dissolve and optional enclosure clip is planned from its padded operand and
actual destination dimensions with `planGPUPolygonOverlay`; resident preflight then checks the
created views and their bindings. This includes every polygon overlay's scratch-byte and
vertex-count limits, for both convex/simple tessellation and
adjacency with `structures`. A device-limit rejection leaves caller nodes and resources unchanged;
correcting capacities permits retrying the same recipe ID. Reference results return each supplied
`output.overlays` record itself, preserving its identity and all leaf views. Foreign input errors
name the recipe ID and input field before preparation begins.
Overlay storage checks use actual supplied destination lengths. Unused default vertex/ring
capacities do not require buffers when a convex reference building has its own overlay output;
simple-enclosure mode still allocates the default dissolve intermediate before the final clip.

The spatial `addVisibilityGraphRecipe` keeps its reference overload and also accepts
`mode: 'csr'` for graph construction without reference centrality. CSR outputs use directed
`offsets`, `neighbors`, unit `weights`, `degree` and named status, not the reference packed status.
A safe retry sequence is: inspect `candidateOverflow` first; if nonzero, increase
`maximumGridVisits` (up to 65,536 per viewpoint query) and rebuild, since `requiredCount` is then
unknown (`0xffffffff`). Decode the five scalar words in `[count, requiredCount, overflow,
candidateOverflow, invalidCount]` order. The public `decodeGPUVisibilityGraphAdjacencyCSRStatus`
returns a shared requirement observation: nonzero `candidateOverflow` means `'unknown'`; otherwise a
`requiredCount` of `0xffffffff` means `'saturated'`; other counts are `'exact'`.
Unknown requirements need candidate recovery first; saturated requirements are only a lower bound,
never an allocation request; exact requirements permit sizing checks. Before increasing
`edgeCapacity`, check both the route cap of 16,777,216 directed slots and device limits (each
neighbor/weight buffer needs four bytes per slot, and all offset/scratch bindings must fit too).
A complete visible graph of 4,097 viewpoints needs `4097 * 4096 = 16,781,312` directed slots,
which cannot fit this route despite being exact. Report the limit or reduce the requested graph;
never retry an unsupported allocation. Only when an exact requirement fits all limits, grow the
producer allocation, rebuild/rebind affected consumers and rerun. Accept the graph only when both
loss flags are zero and `invalidCount` is zero. CSR loss suppresses publication (`count = 0`); it is not a usable prefix.

Import `GPUVisibilityGraphAdjacencyCSRProps`, `GPUVisibilityGraphAdjacencyCSROutput` and
`GPUVisibilityGraphAdjacencyCSRStatus` from `@luma-contributors/analysis/gpu-spatial-analysis`.

Feed the decoded observation directly to the shared planner. This example plans final directed
slots; reconstruction must still pass the recipe's full scratch/binding preflight:

```ts
import type {Device} from '@luma.gl/core';
import {
  decodeGPUVisibilityGraphAdjacencyCSRStatus,
  getGPUSpatialWeightsCapacityPlan,
  VISIBILITY_GRAPH_CSR_MAXIMUM_EDGE_CAPACITY
} from '@luma-contributors/analysis/gpu-spatial-analysis';

declare const device: Device;
declare const pointCount: number;
declare const currentEdgeCapacity: number;
// Pack reads from the five separate scalar ports, in the order documented above.
declare const scalarReads: Uint32Array;
const decoded = decodeGPUVisibilityGraphAdjacencyCSRStatus(scalarReads);
const plan = getGPUSpatialWeightsCapacityPlan({
  rowCount: pointCount,
  capacity: currentEdgeCapacity,
  requiredCount: decoded.requirement,
  headroom: 0
});
if (decoded.requirement.kind === 'saturated') {
  throw new Error('Reduce the graph: the saturated requirement exceeds this CSR route.');
}
if (plan.retryStage === 'candidates') {
  throw new Error('Restore candidate coverage: increase maximumGridVisits or reduce the graph.');
}
if (decoded.invalidCount !== 0) {
  throw new Error('Correct invalid viewpoints or parameters before accepting the graph.');
}
const bindingByteLimit = Math.min(
  device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize
);
const slotLimit = Math.min(
  VISIBILITY_GRAPH_CSR_MAXIMUM_EDGE_CAPACITY, Math.floor(bindingByteLimit / 4)
);
if (plan.capacity > slotLimit || plan.offsetsByteLength > bindingByteLimit) {
  throw new Error('Reduce the graph: exact directed slots or offsets exceed route/device limits.');
}
const nextEdgeCapacity = plan.capacity;
// If allocation changes, rebuild the producer and rebind consumers in a new graph using
// nextEdgeCapacity; rerun and accept only when both loss flags and invalidCount are zero.
```

Radius may change through the live parameter view. Tolerance must match the obstacle-index build;
a different runtime tolerance invalidates the run until reconstructed with that tolerance.
Use the distinct network-owned recipe for exact/sampled centrality on CSR; the reference route
below retains exact dense measures. The spatial dispatcher's existing reference spelling (`addVisibilityGraphRecipe` with omitted
mode or `mode: 'reference'`) remains available.

The reference visibility graph evaluates all pairs among at most 256 supplied free-space viewpoints. An
obstacle contact, including a tangent or endpoint contact, blocks a pair. Coincident viewpoints do
not create an edge. `adjacency` is the complete dense visibility matrix; `edges` contains the bounded
upper-triangle edge list. `status` is `[writtenEdges, requiredEdges, overflow, invalidViewpoints]`.
Dense analysis remains valid when only the edge-list capacity overflows, but an incomplete edge list
must not be consumed as a complete network. Invalid viewpoints invalidate all graph summary metrics: when any viewpoint is invalid (`status`
word 3 is not 0), `metrics`, `choice`, `integration`, `extendedIntegration` and `normalizedChoice` are NaN
for every viewpoint, and `normalizedChoice` is also NaN for every viewpoint when fewer than 3
viewpoints are given. In a valid graph a viewpoint that sees nothing scores 0. Only the invalid
viewpoints' adjacency rows and columns are `0xffffffff`; the edge list keeps the valid pairs. In
`mode: 'csr'` an invalid viewpoint likewise gets an empty row and appears in no other row, the
rest of the CSR graph is published, and `status.invalidCount` counts the invalid viewpoints.

`metrics` contains `[degree, reachableOtherVertices, sumHopDistances, reachableHopCloseness]`.
Closeness is reachable count divided by total hop distance, and is NaN for an isolated point.
`choice` is unnormalized undirected Brandes betweenness (excluding source and target endpoints).
`normalizedChoice` divides undirected choice by `(N-1)(N-2)/2`, using the supplied graph's full
vertex count. `integration` is `[meanDepth, relativeAsymmetry, realRelativeAsymmetry, HHIntegration]`
on each reachable component. For component size `n`, mean depth is `totalHopDepth/(n-1)`, relative
asymmetry is `2*(meanDepth-1)/(n-2)`, and the Hillier–Hanson diamond normalization is
`2*(n*(log2((n+2)/3)-1)+1)/((n-1)*(n-2))`. HH integration is its reciprocal normalized asymmetry.
Components smaller than three return NaN for normalized asymmetry/integration; a complete component
has zero asymmetry and infinite HH integration. These are topological VGA measures; angular segment
NACh/NAIn are different segment-graph definitions and are not relabeled as VGA.
`extendedIntegration` is `[pyramidIntegration, TeklenburgIntegration, PennNormalization,
maximumShortestPathDepth]`. Pyramid integration divides
`2*(n-log2(n)-1)/((n-1)*(n-2))` by relative asymmetry. Teklenburg is
`log(0.5*(n-2))/log(totalDepth-n+1)`. Penn is `(Dmax-totalDepth)/(Dmax-(n-1))`, with
`Dmax=r*(n-(r+1)/2)` and the actual maximum BFS depth `r`. These additional normalizations use
NaN where depthmapX marks the value unavailable (small/complete components or a zero denominator);
HH retains its existing infinite complete-component result. The Penn formula is the named axial
normalization applied to the supplied topological graph, not metric Penn distance.
The distinction follows the [depthmapX metric documentation](https://github.com/SpaceGroupUCL/depthmapX/blob/master/docs/howdoi.md).

## Reference and execution evidence

`modules/analysis/test/oracles/urban-ecology-advanced.py` executes momepy 0.10.0 directly with
`shrink=0`, `segment=1`, and explicit enclosure polygons, including a hole. It also directly compares street-profile width and strict shared-wall totals, including a
point-only contact, and executes the pinned segregation package described in the ecology documentation. The native advanced suite tests
nonconvex/holed areas, occluded facade intervals, narrow angular events, and HH normalization.
Capacity overflow and boundary-viewpoint tests remain in the original urban/ecology suite.
`urban-depthmap.py` downloads the pinned v0.9.1 mathematical header, records its SHA-256, compiles
and directly executes its HH/P-value/Teklenburg/palm-tree helpers for a known visibility path graph.
This is direct upstream normalization evidence, not a full GUI/CLI graph-import comparison. The
momepy fixture also executes zero-buffer joined-structure perimeter, cardinal orientation and
alignment, including an isolated courtyard footprint. No city-wide tessellation throughput is claimed.
