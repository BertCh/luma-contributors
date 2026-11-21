---
title: "GPU Spatial Analysis concepts"
description: "Ontology, shared ports, conventions, coordinate precision and how analysis contributors run."
---


# GPU Spatial Analysis concepts


## Ontology and capability catalogue

The entry point exports a small ontology over the flat contributor API. It gives applications,
documentation tools and future planners stable names for the data flowing between contributors:
points, lines, polygons, trajectories, DGGS cells, spatial weights, feature pairs, regions, sampled
surfaces, summaries and render geometry. Capabilities then group public exports by domain and
pipeline stage and state their logical inputs, outputs, coordinate assumptions and output-bounding
contract.

The catalogue stores export names, not constructor references, so using it for discovery does not
eagerly retain every implementation in a bundle. Every public `GPU...` contributor and `add...`
recipe belongs to exactly one available capability. Deliberate gaps are `status: 'opportunity'` and
have no operators, which makes coverage and missing connective tissue queryable with the same API.

```ts
import {
  getGPUSpatialAnalysisCapabilityForOperator,
  queryGPUSpatialAnalysisCapabilities
} from '@luma-contributors/analysis/gpu-spatial-analysis/ontology';

const producers = queryGPUSpatialAnalysisCapabilities({
  status: 'available',
  output: 'spatial-weights'
});
const gaps = queryGPUSpatialAnalysisCapabilities({status: 'opportunity'});
const localMoranFamily = getGPUSpatialAnalysisCapabilityForOperator('GPULocalMoran');
```

The exported vocabulary is:

- `GPU_SPATIAL_ANALYSIS_DATA_ONTOLOGY` and `GPU_SPATIAL_ANALYSIS_DATA_KINDS`: logical data nouns.
- `GPU_SPATIAL_ANALYSIS_DOMAINS` and `GPU_SPATIAL_ANALYSIS_STAGES`: subject and pipeline roles.
- `GPU_SPATIAL_ANALYSIS_CAPABILITIES`: available capability families and named opportunities.
- `queryGPUSpatialAnalysisCapabilities(query)`: filters by status, domain, stage, input, output,
  coordinate model or search text.
- `getGPUSpatialAnalysisCapability(id)` and
  `getGPUSpatialAnalysisCapabilityForOperator(name)`: direct lookup from a concept or public export.
- `getGPUSpatialAnalysisCapabilityConnections(id)`: resolves the curated adjacent capabilities
  used to assemble workflows.

Operator-level discovery is available through `GPU_SPATIAL_ANALYSIS_OPERATORS`,
`getGPUSpatialAnalysisOperator(name)` and `queryGPUSpatialAnalysisOperators(query)`. Descriptors
contain export names rather than constructors and distinguish primary capability, secondary tags,
logical ports, spatial support, topology, cardinality, numerical semantics, determinism, status and
explicit readback. Unknown details are reported as `unspecified`, never inferred as a stronger
guarantee. `GPU_SPATIAL_ANALYSIS_AUDIT` and `formatGPUSpatialAnalysisAuditMarkdown()` expose the
same inventory used by package verification.

## Contributor operation contracts

`GPUContributorOperation` from `@luma-contributors/analysis/analysis-core` adapts a contributor
into a published luma.gl `GPUGraph` operation. Compile with `GPU_CONTRIBUTOR_OPERATION_LOWERING`.
Input and output graph value names must be non-empty; output formats must be `float32`, `sint32`
or `uint32` vectors with one to four components. Output lengths are validated at construction
and must be positive unless the port is listed in `emptyOutputPorts`. Errors name the operation
and port before luma.gl plans the contract. An operation cannot write the same graph value twice
or read a value it also writes. Packed single chunks are checked when views are resolved.

## Shared ports, status and planning

The entry point exports non-owning structural ports for point, line, polygon and trajectory
geometry, compact pairs, generated geometry, spatial weights, cell tables, sampled surfaces and
classified values. Ports only group existing graph views; they do not allocate, pack, submit or
read back buffers. Polygons always use the four-level layout of `GPUPolygonGeometryPort`
(`positions`, `featureOffsets`, `polygonOffsets`, `ringOffsets`; rows are features, the first ring of
each polygon is its shell), lines use `positions` and `lineOffsets`, and tracks `trackOffsets`.
`GPUPointGeometryPort` and `GPULineGeometryPort` exclude `featureOffsets`; named multipart ports
are distinct types and cannot be passed to single-part consumers. Those consumers also reject
grouping at construction, including object-spread inputs, before unpacking positions or offsets.
This includes geometry validity, line topology/clipping/refinement, track similarity and the
points-in-polygons, cluster-outline and straight-line catchment recipes. Both choropleth backends
reject a grouped point feature instead of counting its parts separately.

`GPUMultipartPointGeometryPort` adds feature-to-point-part `featureOffsets` to the point port;
`GPUMultipartLineGeometryPort` adds feature-to-linestring-part `featureOffsets` above `lineOffsets`.
Both use one `sourceIds` row per feature. Offsets start at zero, are nondecreasing and terminate
at the child count. Equal adjacent offsets describe an empty feature or part, rather than a missing
owner. `validateGPUMultipartGeometryPort` checks static formats, terminal entries and identity
lengths; it does not read offset contents from the GPU. Measures and affine transform share this
validation and retain their existing flat props. For multipart input, both require at least one
feature and one position.

`createGPUMultipartRowOwnersNode(graph, id, geometry, featureRows)` writes one feature row per
point part or linestring part using the current offsets on each submission. The caller owns the
packed `uint32` output, whose length must equal the part count. Empty features own no rows; empty
line parts still have an owner. Uncovered parts receive `GPU_MULTIPART_NO_FEATURE_ROW`
(`0xffffffff`), never an application ID. The adapter requires disjoint buffers for its packed
`featureOffsets` and output views, and each aligned binding must fit the device's
`maxStorageBufferBindingSize`. It does not validate offset contents or make a join feature-aware;
each consumer must explicitly support multipart input.
Predicate joins support every feature-level predicate, DE-9IM output and both engines; nearest joins support
multipart queries and targets in neighbors mode, and prepared indices use feature bounds.
See [multipart join rules and operation forms](./operators-joins.md) for supported routes,
candidate budgets and CPU twins.

A zero nearest-neighbor count establishes that a query is unmatched only with sufficient
producer completeness evidence. Forward the same encoding's `neighborRows`, live `neighborCounts`,
`neighborCapacity`, producer `status` and per-query `candidateTruncated` into `GPUPairGather`, and
request gather `output.status.candidateOverflow` as well as final `overflow`. Per-query evidence
withholds affected zero-count queries from derived unmatched rows. Without per-query attribution,
supplied unattributed global loss withholds **all** derived unmatched rows; the legacy
`candidateOverflowSources` input also represents unattributed global loss. A clear final overflow
alone does not certify the producer's search. Gather `output.status.count` bounds the readable
result prefix, including emitted unmatched rows; tails are unspecified. See the checked
[predicate-to-gather and nearest-to-gather examples](./operators-joins.md#gpupairgather-gpuoffsetexpansion-and-gpuboundsfilter)
for the live-count, capacity and loss wiring.

Contributors with one geometry take these fields flat (`GPUPolygonGeometryFields`,
`GPULineGeometryFields`, `GPUTrajectoryGeometryFields`); contributors with two geometries, such as
the joins, take whole tagged ports (`GPUPackedGeometryPort<...>`, one packed `positions` view), so a
generated geometry output feeds them directly. `GPUSpatialPredicateJoin` `output.pairs` is a
`GPUCompactPairPort` and feeds `GPUPairGather` directly. Every GPU-written status scalar lives in the
`status` of the bag that owns it: capacity-bounded outputs publish `status.count`, `status.requiredCount`
when exact, and the applicable overflow, convergence, invalid or uncertainty fields of the shared status
ports (`GPUCardinalityPort`, `GPUOverflowStatusPort`, `GPUIterationStatusPort`, ...). A nested list owns
its own `status` (`output.pieces.status.requiredVertexCount`), stage-prefixed words keep their prefix
(`output.status.pairOverflow`), and model fits report their solver `status.code`.

`GPUSpatialContext` separates coordinate space, metric, units and sphere/ellipsoid parameters; omitted context means native planar coordinates.
Planar native contexts may declare `units: 'meters'` (local frames do); `'kilometers'` is only for geographic metrics.
Planar `'native'` units are undeclared and compatible with planar `'meters'`; meters and kilometers are incompatible.
The `frame` identity has kind `'local-azimuthal-equidistant'`, `'local-lambert-equal-area'` or `'crs'`;
`assertCompatibleGPUSpatialContexts` rejects contexts with different frame ids.
`datum` declares the horizontal datum of longitude/latitude positions or of a planar frame's source (for example `'EPSG:4326'`, `'EPSG:4269'`, `'GCJ-02'`); `createGPULocalFrame({datum})` copies it to the frame's context.
Two declared datums must match by `getGPUDatumKey` (trimmed, upper-cased; only WGS84 spellings alias to `'EPSG:4326'`), and an undeclared datum matches any; the library never converts datums.
The check runs where two contexts meet; `GPUProjection` CRS ids already imply their datum.
Two-input operators (joins, zonal and region statistics, raster join, pairwise point-in-polygon, line clip, overlay, zone events and the choropleth, change-of-support and hot-spot recipes) take an optional context for their second input, named after that input (`polygonsSpatialContext`, `rightSpatialContext`, `featuresSpatialContext`, `regionSpatialContext`, `zonesSpatialContext`, ...).
When given, `resolveGPUSecondarySpatialContext` validates it, applies the operator's edge-model requirement to it and checks it against `spatialContext`, so a datum, frame, units or metric mismatch between the two sides throws `<id> <side>SpatialContext incompatible spatial contexts: <reason>`; omitted, `spatialContext` describes both inputs and nothing is checked between them.
`edgeModel` (longitude/latitude only) says how polygon and line edges run between vertices: `'linear'` (default, straight in degrees as in RFC 7946, GEOS and turf) or `'great-circle'` (minor arcs, as in PostGIS `geography`, BigQuery `GEOGRAPHY` and S2); ellipsoidal contexts use `ellipsoidalEdgeModel` instead.
Polygon consumers declare `edgeModels: GPU_LINEAR_EDGE_MODELS` and reject great-circle input with a pointer to `GPULonLatPolygonPreparation`, which splits rings at the antimeridian, caps pole rings and densifies great-circle edges; their optional `status: GPULonLatPolygonStatusPort` counts unprepared rings (`antimeridianUncertainCount`, `poleUncertainCount`).
Point-pattern statistics and kernel density take a `studyWindow`: `'rectangle'` (default, per-frame bounds) or a `GPUPolygonStudyWindow` built by `GPUStudyWindowPreparation` in the same planar frame, checked by `resolveGPUStudyWindow`.
Support declarations use `measure: 'distance' | 'area'`: distance operators reject Lambert equal-area frames,
while area/density operators reject azimuthal-equidistant frames, with an error naming the frame to project into.
This measure check accepts `'crs'` frames and frameless contexts.
Spherical distances use `getGPUSphereRadius` (mean radius 6371008.8 m); spherical areas use `getGPUAuthalicSphereRadius` (WGS84 authalic radius 6371007.1809 m).
Both honour `sphereRadius` in the context's units; `units: 'kilometers'` divides default radii by 1000, giving areas in km².
Ellipsoidal metrics use `getGPUEllipsoid` (`context.ellipsoid` or `GPU_WGS84_ELLIPSOID`).
Terrain heights declare a vertical datum the same way (`GPUTerrainHeightReference`, `GPU_ELLIPSOIDAL_VERTICAL_DATUM`, `assertCompatibleGPUVerticalDatums`); see [GPU Terrain](../gpu-terrain/README.md).
`getGPULengthUnits` gives the length unit reported by `GPUMeasureDescription`.
Compatibility helpers reject invalid workflow connections during graph construction and contain no
CRS parsing or reprojection; that remains the responsibility of GPU Project.

Both spatial-context resolvers capture declared metadata once before validation. Supplied contexts and nested frame, unit and ellipsoid metadata are copied and frozen. Known resolved snapshots are reused between stages; later mutation of a caller descriptor cannot change them. An omitted secondary returns the resolved primary object.

`GPUSpatialParameterSchema` describes packed layouts without changing their WGSL representation.
Schemas are published for geometry predicates, neighbor search, line segmentizing, great-circle
arcs, spatial autocorrelation and line density. The generic packer validates defaults, offsets,
ranges and units; contributor-specific packers retain any documented normalization that is not yet
expressed by the generic schema.

Capacity plans (`GPUCapacityPlan` per stage, `GPUContributorCapacityPlan` per contributor) report
stage capacities, work items, caller-owned output bytes and transient bytes. They are computed by
the contributor that allocates, from the same constants as its allocation, and stay pure:
applications own allocations. The weights grow loop is `getGPUSpatialWeightsCapacityPlan` plus the
producers' `getCapacityPlan` (see [Spatial weights](./operators-weights.md)). Candidate and output stages
grow independently: a candidate overflow (for example `output.status.candidateOverflow` of
`GPUContiguityWeights`) means the final `requiredCount` may describe only the candidates that fit,
so growing only the output is invalid.

```ts
import type {GPUContiguityWeights} from '@luma-contributors/analysis/gpu-spatial-analysis';

declare const contiguity: GPUContiguityWeights;
declare const requiredCount: number;
declare const candidateOverflow: number;
const plan = contiguity.getCapacityPlan({requiredCount, candidateOverflow: candidateOverflow === 1});
// Reallocate plan.slots.capacity slots and rebuild with pairCapacity: plan.pairs.capacity.
```

`GPUPartitionDescriptor` retains contiguous source-batch cores, read halos and deterministic seam
ownership without repacking `GPUVector` chunks. `GPUTilePartitionDescriptor` carries spatial core
and halo bounds for tiled consumers. `GPUSpatialJoinPrepared` validates its `partitioning` against the
indexed features. The Arrow
analytics adapter publishes its record batches through this contract, including empty batches.
`iterateGPUAnalyticsPartitionsFromArrowTable()` lazily allocates one source batch at a time while
retaining its global batch ID and row offset; destroying each yielded result before advancing bounds
GPU residency to one partition. The tile-LOD explorer declares explicit core, halo and cross-tile
ownership rules with the tile contract. Execution remains consumer-owned.

Partitioning sizes input consumption; the prepared BVH still covers the whole indexed domain.
Its default `leafCapacity` is `nextPowerOfTwo(featureCount)`, and each whole-domain bounds/children
binding needs `8 * (2 * leafCapacity - 1)` bytes. At a **128 MiB per-binding limit**, 8,388,608
features need 134,217,720 bytes per bounds/children binding and pass this particular check.
8,388,609 features round to 16,777,216 leaves and need 268,435,448 bytes per binding, so partitioning
the input cannot make that index fit. Use independent prepared handles over separate domains if
necessary, and check the binding byte sizes evaluated by the public memory planner against
`maxStorageBufferBindingSize` and `maxBufferSize` before allocation. Graph declaration separately
checks kernel support, including the requirement for eight storage bindings:

```ts
import {getGPUSpatialJoinPreparedMemoryPlan} from '@luma-contributors/analysis/gpu-spatial-analysis';

const limits = {maxStorageBufferBindingSize: 128 * 1024 ** 2, maxBufferSize: 256 * 1024 ** 2};
for (const featureCount of [8_388_608, 8_388_609]) {
  const plan = getGPUSpatialJoinPreparedMemoryPlan({
    kind: 'points', featureCount, vertexCount: featureCount
  }, limits);
  console.log(plan.leafCapacity, plan.bindingByteLengths['BVH node bounds/children'], plan.feasible);
}
// 8388608, 134217720, true
// 16777216, 268435448, false
```

The handle's fixed generation-control storage is **1,072 bytes**: a 1,024-byte lookup table,
32 requested-generation bytes and 16 state bytes. This excludes copies of geometry positions and
offsets, the BVH, and other preparation scratch; the planner accounts for them separately.

Recipe results use a required common envelope with `outputs`, `intermediates` and stage-owned
`status` in addition to `contributors`. Result-level status words sit in `result.status` next to
`result.status.stages` (for example `result.status.weightsOverflow`), and sub-results own a `status`
(`result.permutation.status.overflow`); the `outputs` overrides mirror those paths.

## Conventions

- **Graph views in, graph views out.** Inputs and outputs are `GraphDataView`, `GraphVectorView`, or
  `GraphTextureView` objects created on the target graph. Outputs are always caller-owned; scratch
  storage normally uses graph transients that die with `compiled.destroy()`. Some contributors
  additionally own persistent buffers or preparation state; retain and destroy those owners as
  described in [Running contributors](#running-contributors). Contributors never compile, encode,
  submit, or read back.
- **Per-frame values never recompile.** Viewports, thresholds, radii, observer positions, budgets,
  and region shapes are read from storage views, usually a `GPUParameterBuffer` that the
  application rewrites with `write()`. Lengths, capacities, grid sizes, and which optional views
  exist are compile-time topology; each prop's TSDoc says which category it belongs to.
  `GPUParameterValues<Format>` maps `float32`, `uint32`, and `sint32` to `Float32Array`,
  `Uint32Array`, and `Int32Array`. Initial values and updates must match the declared format;
  matching arrays from other JavaScript realms are accepted. Mismatched arrays and supplied
  non-array initializers (including `null`) are rejected without reinterpreting their bytes.
  Only `undefined` omits initialization. `length` is an integer in
  `[0, 2^32 - 1]`. Initial values must fit that logical length, including when it is zero.
  Updates start at an integer `firstElement` in `[0, length]` and must fit the remaining elements.
- **Bounded results report overflow on the GPU.** Compact ID lists use `GPUCompactOutput`
  (`ids` and `status: {count, overflow, candidateOverflow?, requiredCount?}`). `status.count` is clamped
  to `ids.length` and can be an indirect draw instance count. `overflow` means final output capacity
  loss; `candidateOverflow` means earlier loss that growing the output cannot recover. Both supplied
  flags are rewritten every encoding. `requiredCount` counts accepted rows reaching the final stage,
  before output truncation; it is a lower bound after candidate loss or saturation at `0xffffffff`.
  A consumer must propagate supplied incompleteness, suppress dependent results, or refuse/fall back.
  The shared `createPublishNode` refuses nonempty `candidateOverflowSources` without a candidate
  destination. Producers publish their own optional domain diagnostics separately; omitting such
  diagnostics is not evidence of completeness.
- **Stable IDs.** Result IDs are the caller's `sourceIds[row]` (or tile IDs) when given and zero-based
  rows otherwise. Node and transient IDs are `${id}-<step>`, so two instances in one graph need
  different `id` props.
- **One import per buffer.** Import a buffer once (for example with `graph.importGPUData()` or
  `GPUParameterBuffer.importToGraph()`) and pass the returned view to every contributor that reads
  it, so the graph tracks hazards on one logical handle.
- **Shared contracts.** Four shapes mean the same on every analysis page:
  - **Status.** Every GPU-written status word sits in the `status` member of the bag that owns it:
    `output.status.overflow`, `output.status.converged`, and `output.status.code` for model fits.
  - **Grid.** Raster and terrain contributors take `grid: GPURasterMetadata`, and their packers read
    the cell size from it (see [GPU Terrain conventions](../gpu-terrain/concepts.md#conventions)).
  - **Network.** Network contributors take `topology: GPUNetworkTopology` (`offsets`, `neighbors`,
    `weights`, optional `nodePositions`). Its CSR part can be passed as `GPUSpatialWeights` (see
    [GPU Network conventions](../gpu-network/concepts.md#conventions)).
  - **Polygons.** Polygons use the four-level `GPUPolygonGeometryPort`: `kind: 'polygons'`,
    `positions`, `featureOffsets`, `polygonOffsets` and `ringOffsets`. Contributors with a single
    geometry take the same fields flat, without `kind` (`GPUPolygonGeometryFields`). See
    [Shared ports](#shared-ports-status-and-planning).

`defineGPUSpatialParameterSchema` checks defaults and `packGPUSpatialParameterValues` checks
overrides in both their input and encoded domains. Float32 fields are checked after `Math.fround`:
a finite-only field rejects `1e39`, which would encode as infinity. Bounds apply to the stored
value too: a default or override of `0.1` with maximum `0.1` is rejected because float32 stores
`0.10000000149011612`. Use representable bounds when that is the intended endpoint. Integer
fields require whole values within their signed or unsigned 32-bit range; they never wrap.
`allowNonFinite` explicitly permits NaN and infinity for fields whose kernel defines them.

Contributor authors can import `validateSinglePackedView(view, formats, name)` from
`@luma-contributors/analysis/analysis-core` for the common packed-format, alignment and stride
check. It rejects `GraphVectorView` even when it has one chunk.
`validateGraphOutputsDisjointFromInputs(id, outputs, inputs)` accepts records keyed by port name
and reports both conflicting names, including aliases of imported physical buffers. Array
arguments remain supported; their errors identify the two array indices.

Public props describe supported analysis choices. Benchmark batch sizes, forced cleanup passes,
kriging test chunk sizes, chunk-recursion row bases and extra cell-count destinations belong to
internal composition factories. Use `GPUCellPyramid` for per-level counts and pass the complete
chunked weights object to `GPUSpatialLag`; it handles row bases itself. Generalized GWR publishes
fitted means on the response scale; its bandwidth-selection candidates use link-scale fits internally.
Cell-pyramid and host-scheduled centrality compositions also share their binding preflights through
internal helpers. Their standalone contributors run these checks from `getCommandNodes` before
declaring scratch; callers do not need a separate validation call. The former `validateBindings`
and `validateGraphBindings` composition methods are private implementation details.

For contributor authors, `getWGSLKernelBinding` infers scalar component arrays only for packed,
32-bit-aligned `float32`, `uint32`, or `sint32` scalar/vector views (two to four components).
Atomic access requires an integer view. Other formats and strided layouts are rejected; an
explicit `WGSLKernelBinding` record is required for deliberate raw-word reinterpretation.
`createFillNode` fills every scalar component of these layouts by default. Its `componentCount`
is an integer prefix length from zero to `view.length * componentsPerRow`, measured in components.
Its `type` must match the view's scalar format (`f32`, `u32`, or `i32`). A zero-component fill
binds no target and records no commands, including for an empty view at the backing-buffer end.
`sliceGraphChunkGroupView` borrows a packed parent range with nonnegative integer row bounds
inside `view.length`; an empty slice at the end is valid. It rejects strided parents and never
expands into spare backing-buffer capacity. This descriptor helper accepts packed byte formats
at unaligned offsets and permits nested slicing; storage-word alignment is checked when binding.

## Precision of longitude/latitude positions

Longitude/latitude positions are float32 degrees, never split into high/low parts. At
`|longitude| >= 128°`, the worst-case ground quantum is `Q = 2⁻¹⁶° · π/180 · R ≈ 1.70 m`
for the mean Earth radius `R = 6371008.8 m` (the equatorial worst case). Representative combined
east-west/north-south quanta are:

| Location | Ground quantum |
| --- | --- |
| Chicago | 0.76 m |
| NYC | 0.77 m |
| San Francisco | 0.79 m |
| London | 0.42 m |
| Sydney | 1.47 m |
| Equator near ±180° | 1.70 m |

Great-circle distance kernels use haversine, so short-distance arithmetic is not the limit;
the rounded inputs are. Use `createGPULocalFrame` for sub-17 m work, projecting before rounding
positions to float32 degrees. When a host-known distance is finite, greater than zero and less
than `10·Q` (16.97 m; 0.01697 km with the default sphere), operators warn once per operator class:
`<Operation> distance is too short for float32 lon/lat; use createGPULocalFrame`.
This precision check never throws; zero, NaN and non-finite distances do not warn.

The exported helpers are `getGPULongitudeLatitudeQuantum(context)`,
`isGPUShortLongitudeLatitudeDistance(context, distance)` and
`warnGPUShortLongitudeLatitudeDistance(operation, context, distance)`, with
`GPU_SHORT_LONGITUDE_LATITUDE_DISTANCE_FACTOR = 10`. The quantum uses the context's sphere radius
(or ellipsoid semi-major axis) and length unit; it is zero for planar contexts or metric `'none'`.

- **Tier 1 — construction.** `GPUSpatialPredicateJoin` checks numeric `dwithin` distances;
  `GPUTrajectoryEncounters` checks its derived `maximumDistance`.
- **Tier 2 — per-frame writers.** Pass optional `spatialContext` in the settings passed to the
  parameter writer to check neighbor-search `searchRadius`, DBSCAN `epsilon`, IDW `searchRadius`,
  kriging `searchRadius` and `range`, Douglas-Peucker `tolerance`, line segmentize / great-circle
  arcs `maximumSegmentLength`, line chunk `chunkLength`, outline `distance`, map matching
  `sigma` and `searchRadius`, and GWR fixed `bandwidths`. Packed parameter layouts are unchanged;
  omitting the settings context disables this warning.
- **Tier 3 — GPU views.** No warning is possible for `GPUBufferSelection.radius`,
  `GPUNearestFeatureJoin.radius`, a view-valued `GPUSpatialPredicateJoin.distance`,
  `GPUTrajectoryEncounters.distance`, `GPUTrackSimilarity.frechetEpsilon`,
  `GPULinearReferencing.radius`, `GPUNetworkSnapping.maximumSnapDistance`,
  `GPUNetworkKFunction` distance views, the drive-time catchment recipe's `maximumSnapDistance`, or
  `GPUGeodesicDestination.distances`: the host does not read these GPU values. Trajectory-metrics
  stop detection also cannot warn: the affected distance is speed × sampling interval, so float32
  position noise means about ±0.8 m/s at 1 Hz in Chicago; use a local frame.

## Running contributors

This applies to every contributor page (spatial analysis, network, terrain, raster, dataframe
analysis and geospatial). Applications compile with `await graph.compileAsync()`. It creates every
GPU pipeline asynchronously and concurrently, so the page keeps responding while a graph with many
kernels compiles. `graph.compile()` gives the same result but blocks the main thread until every
pipeline exists; keep it for tests and scripts. Either call freezes the graph, and a graph compiles
once, including when `compileAsync()` rejects.

Use `graph.add(producer)` to declare and add a contributor in one call. For inspection, declare
once and add the returned nodes:

```ts
import type {GPUCommandGraph, GPUCommandNode, GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';

declare const graph: GPUCommandGraph<void>;
declare const producer: GPUCommandNodeProducer<void>;
declare function inspect(nodes: readonly GPUCommandNode<void>[]): void;
const nodes = producer.getCommandNodes(graph);
inspect(nodes);
graph.add(nodes);
```

Declaration can allocate graph resources and contributor-owned buffers. Calling
`getCommandNodes(graph)` for inspection and then `graph.add(producer)` declares twice and can
collide on resource IDs or violate an owner's one-declaration guard. Repeatable declaration is an
optional implementation detail. Replay the compiled work with `compiled.encode(...)`.

Keep these seven cleanup owners when using them: `GPUConditionalGaussianSimulation`,
`GPULocalFrameProjection`, `GPUSpatialJoinPrepared`, `GPUStudyWindowPreparation`,
`GPUTerrainSpikeRepair`, `GPURasterReprojection` and `GPURasterTileReprojection`. Their owned
resources are distinct from graph-transient storage. Destroy the compiled graph first, then call
each owner's `destroy()`, then release caller-owned imported buffers. Retained preparation belongs
to its handle and graph; rebuild it with a new owner for a new topology.

Compile once per topology and rewrite parameter buffers every frame. A compile-time option (a grid
size, a method that adds nodes, an optional output) needs a new graph. Build the new graph and await
its compile while the old compiled graph keeps encoding. Then swap the new graph in, and only then
destroy the old one. If a later request arrives before the compile finishes, destroy the stale
result instead of swapping it in.

```ts
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';

type Options = {gridSize: [number, number]};
// The application builder retains every contributor needing destroy(). Its destroy method
// releases those contributors, including after a failed compile. It also rolls back if building throws.
type GraphOwner = {graph: GPUCommandGraph<void>; destroy(): void};
type CompiledOwner = {
  compiled: Awaited<ReturnType<GPUCommandGraph<void>['compileAsync']>>;
  owner: GraphOwner;
};
declare function buildGraph(options: Options): GraphOwner;
const options: Options = {gridSize: [64, 64]};

async function compileOwner(owner: GraphOwner): Promise<CompiledOwner> {
  try {
    return {compiled: await owner.graph.compileAsync(), owner};
  } catch (error) {
    owner.destroy();
    throw error;
  }
}

function release(pair: CompiledOwner): void {
  try {
    pair.compiled.destroy();
  } finally {
    pair.owner.destroy();
  }
}

let current = await compileOwner(buildGraph(options));
let latestRequest = 0;

async function rebuild(nextOptions: Options): Promise<void> {
  const request = ++latestRequest;
  const next = await compileOwner(buildGraph(nextOptions)); // the old graph keeps encoding
  if (request !== latestRequest) {
    release(next); // graph first, then its contributors, when a later request supersedes this one
    return;
  }
  const previous = current;
  current = next; // swap first
  release(previous); // then release the previous graph and its contributors
}

// every frame: record into the device's default encoder, then submit it
current.compiled.encode(device.commandEncoder, {parameters: undefined});
device.submit();
// On application teardown, invalidate pending rebuilds before releasing the active pair.
function destroy(): void {
  ++latestRequest;
  release(current);
}
```

`encode()` only records commands; the application submits (`device.submit()` for the device's
default encoder, or `device.submit(encoder.finish())` for its own encoder). The second argument's
`parameters` is the graph's typed CPU channel, which node conditions and encode callbacks receive.
Numeric per-frame inputs are packed into GPU views (a `GPUParameterBuffer` written by a
`get*ParameterValues` packer, see each class's "Parameters" subsection). For those contributors,
use `GPUCommandGraph<void>` and pass `{parameters: undefined}`.

`GPUSpatialJoinPrepared` and `GPUStudyWindowPreparation` also accept a CPU `rebuildWhen` predicate.
Give the handle and graph the same frame type. The first encoding and first encoding after
explicit invalidation bypass the predicate; otherwise it receives that frame once per encoding.
Requested geometry generations remain pending on the GPU until complete preparation executes,
including when an encoding is discarded. The current public builders replay against a retained
geometry snapshot each encoding; geometry refresh alone is GPU-gated. Use one active resumable
execution per handle/graph.

Prepared joins replay ring mapping, bounds, sorting, BVH and optional edge-slab construction from
that retained snapshot on every encoding; consumers share the preparation within that encoding.
For predicate joins, retained bounds, feature-ring mappings and exact refinement all read
**the same retained right positions/offsets** through `prepared.getGeometry(graph)`.
Source coordinate and topology edits become visible after `invalidate()` (or a true
`rebuildWhen`) and a submitted rebuild refresh the snapshot.
Source-ID contents remain live for output mapping; changing an ID does not change a retained
feature row or its geometry. Changing view lengths requires a new handle and graph;
invalidation alone does not resize them.

```ts
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUSpatialJoinPrepared, type GPUSpatialJoinPreparedProps} from '@luma-contributors/analysis/gpu-spatial-analysis';

type Frame = {geometryChanged: boolean};
// Import or create every polygon view on the graph that will declare its consumers.
declare function importPolygons(graph: GPUCommandGraph<Frame>): GPUSpatialJoinPreparedProps<Frame>['geometry'];
const graph = new GPUCommandGraph<Frame>(device);
const polygons = importPolygons(graph);
const prepared = new GPUSpatialJoinPrepared<Frame>({
  geometry: polygons,
  rebuildWhen: frame => frame.geometryChanged
});
let compiled: Awaited<ReturnType<typeof graph.compileAsync>> | undefined;
try {
  graph.add(prepared);
  // Add the joins that consume prepared before compiling.
  compiled = await graph.compileAsync();
  compiled.encode(device.commandEncoder, {parameters: {geometryChanged: true}});
  device.submit();
} finally {
  compiled?.destroy();
  prepared.destroy();
}
```

Both handles default their frame type to `unknown`; an annotated callback can infer it. When
using `invalidate()` alone, they also work with a `GPUCommandGraph<void>`.

`GPUAdamStep`, `GPUFittingCoordinateEncoder`, `GPUFittingGradientSums`,
`GPUFittingMinibatchOrder`, `GPUPolygonTriangulation` and `GPUSegmentRingAssembly` snapshot
construction options, spatial contexts and port maps. Mutating the caller's props or replacing a
port afterward does not change their validated shapes or bindings. GPU views and buffers retain
their identities: update their contents for per-frame inputs, and construct a new contributor to
change topology.

Snapshots read declared fields through property access, so class getters and non-enumerable
ports are supported. Exposed spatial context descriptors and their nested metadata are frozen.

The gpu-raster base operators (`GPURasterStatistics`, the filters, morphology and region operators)
predate `GPUCommandNodeProducer`: they declare their work with `operator.addToGraph(graph)` instead
of `graph.add(operator)`. Compiling, encoding and submitting are the same.

Destroy a compiled graph before the caller-owned buffers it imports. Keep compilation and blocking
readback outside the frame loop; encode and submit each frame without waiting for GPU completion.


## GPU execution model

How spatial analysis contributors run on the device: the command graph they declare, the kernels it
compiles to, and the resources each pass reads and writes, drawn from a recorded real run.

{/* lens:execution-model */}

Each contributor adds nodes to a `GPUCommandGraph`. `compile()` orders them by the buffers they read and write,
`describeGPUCommandGraph(compiled, options)` from `@luma-contributors/analysis/computation-record` records that structure
without running anything, and the lenses below draw it from recorded runs.

::lens{scenario=gpu-spatial-analysis.kernel-density rung=graph caption="Kernel density: grid index over the points, in-cell sort for determinism, then one gather pass."}

::lens{scenario=gpu-spatial-analysis.kernel-density rung=kernel caption="The gather kernel: one invocation per output cell; the binding table names every buffer it reads."}

::lens{scenario=gpu-spatial-analysis.point-in-polygon-join rung=graph caption="A join: bounding-box candidate pairs first, then one exact containment test per candidate pair."}
