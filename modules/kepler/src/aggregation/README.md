# Resident cell aggregation

`GPUKeplerAggregation` reuses `GPUPointDensity` for selected grid and hexagon working sets. It
borrows packed float32 XY source chunks, optional float32 weights, and uint32 selection/validity
masks. Logical lengths must agree; physical chunk boundaries may differ. Inputs must already
be projected into a common planar coordinate system. Rebase coordinates before upload when
float32 cannot distinguish nearby cell boundaries.

```ts
const aggregation = new GPUKeplerAggregation({
  device,
  positions,
  weights,
  selectionMask: explorer.selectionMask,
  validity: validWeightRows,
  statistic: 'sum',
  topology: {
    kind: 'grid',
    origin: [0, 0],
    cellSize: [100, 100],
    gridSize: [100, 100]
  }
});
const encoder = device.createCommandEncoder();
explorer.encode(encoder);
aggregation.encode(encoder);
device.submit(encoder.finish());

// Exact planar footprints; counts hide empty cells without a result download.
const geometry = aggregation.getPolygonGeometry();
const layer = new GPUSolidPolygonLayer({id: 'cells', ...geometry});
// Centers and values may also feed GPUColumnLayer getPosition/getElevation directly.
```

The adapter never submits commands or reads buffers. `prepare()` compiles and allocates scratch
without encoding, so a host can include it in memory admission before rendering. Changing borrowed mask, validity, position,
or weight contents and encoding again reuses the graph and output buffers. Replacing input buffers
or changing topology/statistic requires a new adapter. Destroy the adapter after its GPU work and
dependent layers have finished; borrowed source buffers remain owned by the caller.

Grid origin is its lower-left boundary. Bounds are origin plus cell size times grid size; both
maximum edges are included in the last cell. Hexagons are pointy-top, with odd rows shifted right
by half a hexagon width. Hexagon `(0, 0)` is centered at the lower-left clipping bound. Its radius
is the circumradius in source units. Set hexagon topology `assignment: 'd3'` for d3-hexbin's native
normalized-distance assignment and positive-infinity half-tie rule. The compatibility default
`'cube'` uses nearest-center cube rounding; these rules differ for some interior points, not only
ties. The d3 CPU helper and GPU implementation are checked against an independent d3-hexbin 0.2.2
fixture, including a known interior disagreement and an exactly representable GPU half tie.
Float32 division may still move a point across a boundary compared with a float64 CPU reference.
Aggregate ID is `row * columns + column`, including empty
cells. Polygon row indices and picking IDs refer to these aggregates, not original source rows.

Counts include selected finite in-bounds positions. Zero and negative weights are included.
NaN/infinite weights contribute zero and still count; pass a zero validity mask for missing rows
that should be excluded from count and the mean denominator. Means are `sums / counts`, or zero
for an empty cell. The deterministic existing sum implementation uses fixed-point quantization
and float32 outputs; it is not an exact float64 sum. `counts` are uint32; count-valued `values`
are float32 and lose integer precision above 2^24. `overflow` is a one-word resident flag for
points falling outside the hex lattice. All outputs and optional polygon geometry are owned by
the adapter. The explicit composed mask and the density primitive's key/masked-position scratch
must fit storage bindings; oversized working sets are rejected rather than silently truncated.

`memberLookup` is explicitly `not-materialized`: this adapter does not create aggregate member
lists. Geometry is created once from the fixed lattice and does not rebuild during filtering.
The current column/grid renderers use pixel radii, so their radius props are not equivalent to
world cell dimensions. Use polygon footprints for the latter. Polygons currently have zero
elevation and constant/caller-supplied colors; the scalar outputs remain resident for subsequent
color/elevation computation. The example runtime supplies resident aggregate color classification;
extrusion/elevation channels are explicitly unsupported there. Heatmap smoothing/KDE, arbitrary
aggregation operations and source-member lookup remain separate integration work. Host projection
and origin conventions must match the intended reference; the core operates in the supplied planar
frame. Tests pin origins, clipping, counts, weight policies, d3 assignment, stable identities and
ownership; they do not establish parity for every Kepler layer control or float64 boundary.
