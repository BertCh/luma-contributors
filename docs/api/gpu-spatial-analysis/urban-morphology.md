# Urban morphology recipes

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
| `addBuildingTessellationRecipe` | `overlays` (one `GPUPolygonOverlay` output per building, the final polygons), `samples`, `owners`, `fragments`, `fragmentCounts`, `samplingStatus`, `complete` |
| `addStreetProfilesRecipe` | `profiles`, `buildings`, `validity`, `canyon`, `frontage`, and with `streetSegment` `facadeLengths`, `facadeFrontage` |
| `addBuildingAdjacencyRecipe` | `pairs`, `lengths`, `morphology`, `components`, `complete`, and with `structures` `structurePerimeters` |
| `addIsovistsRecipe` | `endpoints`, `distances`, `segments`, `validity`, `measures`, `angles`, `viewpointClassifications` |
| `addVisibilityGraphRecipe` | `adjacency`, `edges`, `metrics`, `choice`, `integration`, `normalizedChoice`, `viewpointClassifications`, `extendedIntegration`, `status` |

Limits are named constants and are rejected with a message that starts with the recipe ID and names
the prop. Tessellation: 64 buildings (one overlay per building), 32 enclosure vertices and 128 boundary
samples (thread-private polygons and a samples-squared fragment table). Adjacency: 1,024 buildings
(buildings-squared pair table), and 64 buildings with `structures`. Visibility graph: 256 viewpoints.
Isovists: 4,096 rays per viewpoint with `sampling: 'vertices'` and 65,536 otherwise. Street profiles:
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

Supply closed building boundary `segments`, their `segmentBuildingIds`, a `buildingCount`, and one
`enclosure`. Convex counterclockwise rings use the default fast path. Set
`enclosureMode: 'simple'` for a nonconvex ring; provide `enclosureRingOffsets` to describe an
exterior and its holes in the same position view. Segment endpoints are sampled at intervals no larger than
`sampleSpacing`; repeated segment end vertices are avoided. The recipe clips every boundary-sample
Voronoi cell to the enclosure, groups cells by building, and uses the existing `GPUPolygonOverlay`
dissolve contributor to remove internal fragment boundaries. `outputs[buildingId]` is an actual
packed polygon result; the building ID is its index in `outputs` (the overlay result has no
source-ID column, so the former per-ring copy of the building ID is gone).

`sampleCapacity` is explicit and at most 128; enclosures have 3–32 vertices, and a recipe supports
at most 64 buildings. `capacity.intersections`, `nodedSegments`, `vertices` and `rings` separately
bound dissolve work and output. Sampling status is `[written, required, overflow, invalidSegments]`.
`status.complete[buildingId]` combines sampling, fragment validity and overlay capacity evidence.
An overflow is never a complete tessellation; increase the reported capacity and rerun. The returned
sample locations, building owners, cell fragments and counts preserve the approximation provenance.
Fragment-count `0xffffffff` denotes invalid enclosure geometry, incompatible duplicate sites or a
fragment-capacity failure. Zero is an empty/duplicate cell.

In simple mode, cells are clipped to the enclosure bounding box, dissolved by building, then
intersected with the actual enclosure through `GPUPolygonOverlay`. Both dissolve and intersection
capacity evidence gates completeness; holes and disconnected clipped pieces retain the same building
ID. Ring offsets must describe one valid polygon (exterior followed by holes); the input is not a
request for automatic repair. Run separate enclosures independently. Street-network enclosure
construction and automatic building shrinking remain explicit upstream geometry operations. Duplicate
sites owned by different buildings invalidate the result. Sample spacing determines the building
boundary approximation; `vertexTolerance` independently controls the common overlay snap grid and
sliver collapse. Halving spacing is the appropriate convergence check. A building may legitimately
have an empty cell if no sampled support owns area within the enclosure.

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
resulting interval. An occluded wall contributes no facade length. This is exact for the declared
straight rectangular corridor up to the shared geometric tolerance, rather than a sample-count
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
orientation affects the angle only. Numerical outputs are NaN when supplied upstream overflow or
uncertain-intersection evidence is nonzero. `lengths` retains each measured shared run.

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

Isovists cast `rayCount` uniformly spaced rays counterclockwise from +X, clipped by closed segment
obstacles and a finite circular `maximumDistance`. `endpoints`, `distances`, hit `segments`, and
per-ray `validity` preserve every sampled ray. `measures` is
`[polygonArea, polygonPerimeter, maximumRayDistance, valid]` per viewpoint. The polygon joins adjacent
ray endpoints. With `sampling: 'uniform'` it can miss an obstacle between rays. With
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

The visibility graph evaluates all pairs among at most 256 supplied free-space viewpoints. An
obstacle contact, including a tangent or endpoint contact, blocks a pair. Coincident viewpoints do
not create an edge. `adjacency` is the complete dense visibility matrix; `edges` contains the bounded
upper-triangle edge list. `status` is `[writtenEdges, requiredEdges, overflow, invalidViewpoints]`.
Dense analysis remains valid when only the edge-list capacity overflows, but an incomplete edge list
must not be consumed as a complete network. Invalid viewpoints invalidate all graph summary metrics: when any viewpoint is invalid (`status`
word 3 is not 0), `metrics`, `choice`, `integration`, `extendedIntegration` and `normalizedChoice` are NaN
for every viewpoint, and `normalizedChoice` is also NaN for every viewpoint when fewer than 3
viewpoints are given. In a valid graph a viewpoint that sees nothing scores 0.

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
