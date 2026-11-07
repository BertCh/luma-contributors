# Footprint shape lab implementation record

## Question and provenance

The scene asks how interior geometry responds when an archived Chicago building boundary moves.
It reads `chicago-buildings` and selects source feature IDs **354712** and **814558** by
`properties.bldg_id`. Feature 354712 is the courtyard case: one polygon, two stored rings, 24 stored
positions including ring closure and 22 unique positions. Feature 814558 is the strictly convex
four-corner comparison. Both are projected in a `LocalMetricProjection` centered at the source
shell centroid. The resulting coordinates are local Web Mercator-scaled planar meters for
footprint-scale display and computation, not cadastral precision or code-compliant setbacks.

Ring coordinate order and the stored closing coordinate are retained in the input buffers. The
vector skeleton normalizes shell and hole orientation internally; the convex solver receives the
single convex ring. No roof, structural, access or setback-compliance conclusion is made.

## GPU contract and output mapping

Each selected question compiles its own graph on demand. The cache is keyed by source, question,
signed offset controls, join, resolution, work budget and tolerance, and holds at most eight
variants. Switching questions clears existing marks and readouts and deactivates the previous
graph before compiling. A completed source graph is activated only after compilation. A generation
and active-variant guard prevents superseded summary reads from replacing the current scene,
including cache revisits. The graph encodes once when activated; `SummaryReader` copies its stale
result once after that encode. `GraphDataView` values are mapped through scene-owned buffers;
graph handles are never treated as device buffers. Cached readers and resources are released on
eviction or destroy, and failed builds destroy partial resources.

| Contributor | Configuration and output shown | Status and contract kept visible |
| --- | --- | --- |
| `GPUVectorStraightSkeleton` | `maximumVertices: 32`, event budget 2 or 32, `segmentCapacity: 128`; arcs from `segments`, source pairs from `sourceEdges`, active count from `status`, requested inward contour is available in `offsetPositions`/`offsetLinks`. | `[writtenArcs, requiredArcs, activeOffsetVertices, flags]`; flags or count/capacity mismatches withhold geometry. The algorithm supports reflex splits and holes; whether either event occurs depends on GPU output. It is not a medial axis. |
| `GPUConvexStraightSkeleton` | `maximumVertices: 16`, iteration budget 1 or 16, capacity 64; convex source-edge arcs and the requested offset ring. | `[writtenArcs, requiredArcs, offsetVertexCount, flags]`; only the strictly convex, simple, non-collinear one-ring input is selected. |
| `GPUMaximumInscribedCircle` | Tolerance slider; quick/resolved limits are 32/512 cells and 32/2048 iterations. Reads center/radius/error, closest boundary witness and status. | `SUCCESS` and `BUDGET_EXHAUSTED` are distinct. A budget-exhausted bounded circle is shown with its remaining search error. The search error bounds geometric search, not f32 rounding. |
| `GPUMinimumClearance` | One polygon feature; reads scalar minimum displacement, `minimum_clearance_line` witness and fragile vertex ID. | `0xffffffff` means no finite witness. The measure is smallest vertex displacement to invalid geometry, not courtyard width, sidewalk width or free-space radius. |
| `GPUBufferSurface` | Convex footprint polygon; signed distance from an inward/outward selector, round/mitre/bevel join, round resolution and mitre limit. Reads feature and required-ring counts, candidate/uncertain statuses, boundary counts and offsets before outlining the topology-resolved surface or reporting area. | Any overflow, incomplete candidate set, uncertainty or capacity/count mismatch withholds geometry and area. Positive polygon distance expands and negative contracts the filled polygon. |
| `GPUOffsetCurve` | Courtyard's first exterior ring is treated as an open directed boundary chain after dropping its stored closing coordinate, so the closing edge is omitted. Writes raw signed offset rows and per-source-vertex counts. | Positive is left of this chain's stored direction; negative is right. Raw rows are not presented as a final polygon. |
| `GPUOffsetCurveComponents` | Nodes the raw chain, removes the source-distance band and selects components at the same signed distance; reads written/required component and vertex counts plus source edge IDs. | Overflow or incomplete noding withholds component geometry. The output is a boundary chain, not a path. |
| `GPUOffsetExpansion` | Three uniquely named producers expand polygon offsets to rings, ring offsets to vertex-to-ring/local-index rows, and ring offsets composed through the polygon owner map to vertex-to-polygon rows. | Count, required count and overflow are distinct from capacity. Producer order makes the owner-map dependency explicit. |

The offset branches expose join style. Round resolution (`quadSegments`) and mitre limit are
controls because they change the output boundary. The directed-chain branch has an explicit
left/right selector, mapped to positive/negative `GPUOffsetCurve` distance. The convex polygon
buffer exposes an inward/outward sign selector and reports the area of its topology-resolved GPU
offset surface. The skeleton `eventTolerance` is held at
`1e-5` local meters; no reader control claims it is a positional accuracy estimate. The convex
skeleton's low/high iteration budget and the circle's low/high cell/iteration budget are
meaningful work controls with status retained.

## Numeric reference checks

The runtime reports GEOS/Shapely reference deltas for both largest-inscribed-circle radii and
minimum-clearance values. Circle acceptance is withheld when its search budget is exhausted; its
allowed radius delta is the larger of the requested search tolerance and 0.25 m. Clearance uses a
0.05 m tolerance. For the convex source at a 4 m offset, the runtime compares complete polygon
buffer area against GEOS/Shapely references for each join, with a 15 m² tolerance. These checks
appear alongside the measured GPU result and do not replace GPU status/capacity gates. References
came from Shapely 2.1.2 / GEOS 3.13.1 on the float32-projected source fixture. The scene's local
frame now uses the source shell centroid, which can slightly change Web Mercator scale from the
reference frame; the listed tolerances include that small frame-scale difference. The scene was not
GPU-executed in this review, so these runtime checks have not yet been observed to pass on a device.

| Source | Circle radius (m) | Minimum clearance (m) | 4 m outward buffer area (round / mitre / bevel, m²) | 4 m inward buffer area (all joins, m²) |
| --- | ---: | ---: | ---: | ---: |
| Courtyard 354712 | 18.3556437407 | 0.1519976729 | — | — |
| Convex 814558 | 28.6538888484 | 56.7707254306 | 6875.38742554 / 6889.44752088 / 6857.44275212 | 4411.04924439 |

## Conceptual references

- [CGAL Straight Skeleton and Polygon Offsetting](https://doc.cgal.org/6.2/Straight_skeleton_2/index.html): inward wavefront traces, offset components that can split, and opposite shell/hole orientation.
- [Shapely maximum inscribed circle](https://shapely.readthedocs.io/en/stable/reference/shapely.maximum_inscribed_circle.html): largest interior circle, nearest-boundary witness, and use as a label anchor.
- [Shapely minimum clearance](https://shapely.readthedocs.io/en/stable/reference/shapely.minimum_clearance.html): minimum vertex displacement that produces invalid geometry.
- [Shapely offset curve](https://shapely.readthedocs.io/en/stable/reference/shapely.offset_curve.html): positive-left and negative-right convention, join style and mitre limit.

These sources support the conceptual vocabulary. luma.gl's checked source and tests define exact
buffer layouts, capacity behavior, status values, option limits and topology contracts.

## Delivered and verification limits

Delivered: the new `footprint-shape-lab-data.ts`, `.compute.ts`, `.scene.ts` and this record. The
scene passes its chapter story check, WGSL identifier check, direct Biome check and an isolated
strict TypeScript check. The chapter cartography check reported existing findings in
polygon-measures and sinusoidal-projection; none named this scene. The showcase-wide typecheck
reports unrelated concurrent diagnostics in point-patterns and habitat-connectivity; no
diagnostics name this scene. No GPU execution, browser rendering or observed numerical output
verification was performed for this scene.
