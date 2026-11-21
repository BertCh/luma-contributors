# Upstream proposals

Ready-to-file bodies for the first four filings are in the
[draft index](./upstream-drafts/README.md), ranked 02, 03, 01, 04:
[histogram edges and domains](./upstream-drafts/02-histogram-edges-and-domains.md),
[caller CSR with masks](./upstream-drafts/03-caller-csr-and-masks.md),
[packed CPU/GPU transfers](./upstream-drafts/01-packed-cpu-gpu-transfer.md), then
[typed buffer imports and CPU vector construction](./upstream-drafts/04-typed-buffer-and-cpu-vector.md).
They distinguish installed alpha.16 source checks from historical GPU measurements. Nothing has been filed.

Nothing has been filed upstream: no issue and no pull request. Historical measurements cover one adapter: headless Chromium WebGPU on Apple Metal. Repository pins now target `10.0.0-alpha.16` (2026-11-18); deck.gl remains 10.0.0-alpha.1 and math.gl 5.0.0-beta.8. The offline audit below supersedes old claims of API absence; it does not remeasure GPU behavior. No alpha.14 declaration snapshot is available for an exhaustive release diff.

## Alpha.16 request disposition (2026-11-18)

Public means exported from an installed package entry point, not merely declared under dist.
The five grid-coordinate copies were replaced as part of this bump to keep readers aligned
with the published builder. Other adoption remains separate. See [the register](./upstream-workarounds.md#alpha16-public-api-audit-2026-11-18)
for replacement contracts and call-site risks. Detailed reproductions below retain their original
measurement scope; satisfied requests should not be filed as missing APIs.

| Request | Alpha.16 status and public entry point |
| --- | --- |
| Kernel authoring factory (summary 1) | **Partly:** `Kernel` (`@luma.gl/graph`), `createGPUComputeCommandNode`/bounded-dispatch helpers (`@luma.gl/graph/gpu-core`) and `getGPUVectorWGSLLayout` (`@luma.gl/graph/gpu-data`) supply building blocks. No typed packed accessor/node factory, `getBoundedDispatchOverrides` or `getSpatialOperationNodes`. |
| 1. One grid formula and WGSL export | **Satisfied and adopted in alpha.16:** `GPU_GRID_COORDINATE_WGSL` (`@luma.gl/graph/gpu-core`); index/query/binning/aggregation and all five local readers use it. Generated-source pins pass; the headless builder/reader edge/overflow regression passes 30 of 30 cases. |
| 2. Caller CSR/masks/self-loops | **Not provided:** `@luma.gl/graph/gpu-graph-algorithms` still requires physical topology/output vectors; no requested view/mask contract. Solver CSR does not widen those algorithms. |
| 3. Haversine | **Partly:** `GPUHaversineDistance` (`@luma.gl/graph/geospatial`) documents the 6371-kilometre radius and accepts raw Float64 deltas. No named default-radius constant or established float32 seam replacement. |
| 4. Finite differences; 5. morphology | **Missing contracts:** constant spacing/no validity in `GPUFiniteDifference2D` (`@luma.gl/graph/gpu-core`); at most matching finite-difference stencils could delegate, while selectable estimators (default `'evans-young'`), Hessian and curvature formulas remain local. Morphology in `@luma.gl/graph/gpu-raster` still caps radius at 8 without the requested fill/count/reconstruction contract. |
| 6. Histogram | **Partly:** `GPUHistogram` (`@luma.gl/graph/gpu-core`) has GPU/auto domains, offsets, masks and explicit edges. Equal-width float quotient remains; NaN-as-finite-extent policy is absent. Historical alpha.12 failures need rerunning. |
| 7. K-means; 8. contours; 9. wide/sparse joins/groups; 10. exact polygon fallback | **Not provided:** no requested seeding/tolerance/restarts, multi-level world/band contour, wide-key/deterministic sum or exact indexed-candidate polygon contract in the public vector-search/raster/gpu-core/geospatial entry points. |
| 11. Standalone uint64/H3 data; 12. count; 13. compiler remedies | **Not provided:** no standalone multiply/divide or H3 table data export, universal requiredCount/overflow contract or compiler-safe exact arithmetic guarantee. Historical shader observations require adapter/browser requalification. |
| 14. Smaller gaps | **Separate adoption, contracts and measurements:** caller-owned grid offsets/IDs/count/overflow already exist; fixed-cell lattice, half-open bounds, stable ordering and per-row keys remain contract requests. Opt-in `kernelMetadata`/`preflight.nodes[].kernel` is available for primitives declaring metadata. Section 14 separates other missing contracts from pipeline/dispatch/lifetime performance investigations; existing `dependsOn` already permits dependencies. |
| Segmented reduction/scan and scatter (export summary) | **Available adoption candidates:** `GPUSegmentedReduction`, `GPUSegmentedScan`, `GPUScatter` (`@luma.gl/graph/gpu-core`). Adopt eligible ungated sums after gather, empty-segment, nonzero-prefix, output-tail and accumulation-order parity. Gated sums still need primitive dispatch geometry/output adaptation; exact accumulation stays local. Scatter needs unique destinations for deterministic results. |
| Jacobi PCG (export summary) | **Available export; incompatible schedule for direct EC03 replacement:** `GPUJacobiPCG` (`@luma.gl/graph/gpu-core`) accepts caller CSR/chunked views, residual/breakdown outputs and `getCommandNodes(graph)`. The reported two-row diagonal system declares 51/85/153 nodes at budgets 1/2/4 (`17 + 34I`), 34,833 at 1,024, exceeding the local 32,768-node budget before setup; 139,281 at 4,096. Local PCG uses five nodes per iteration and independently certifies true residual and actual iteration count. Require compatible compile budgets, stopping/residual/count parity, grounding, normalization, disconnected-input semantics and continuation before migration. A warm start does not resume Krylov state. |
| Other requested exports | **Not provided:** `getGPUComputeDispatchWorkgroups`, `gateGPUCommandNodes`, `getPlatformFP64Arithmetic`, `getClassicFP64Defines`, raster-utils validators and `GPUScalar` are not public exports. Public `GPUScalarFormat` is a different type. |
| Attachment/binding cache/module wiring | **Not provided:** producers and addToGraph domain classes still differ; Kernel requires per-dispatch bindings; no automatic shader-module resource wiring. |
| Adapter limits/subgroups/buffer_view (section 4) | **Partly existing tools:** numeric `requiredLimits`, subgroup reduction, unsegmented-scan subgroup selection and the sort local-bitonic subgroup path are public; segmented scan stays portable. Seek limits-only adapter-max policy and evidence for any additional strategy; preserve fallbacks. No portable buffer_view guarantee. |
| Dataframe lifecycle/nullability/crossfilter/descriptions | **Not provided:** no vector-construction nullable fields, query compileAsync/stats/preflight/mask-only mode, external mask/invalidate or source-expression descriptions. |
| Arrow normalization/exact time | **Partly:** `planArrowGPUConversion`, `convertArrowToGPUVector`, `convertArrowTemporalToGPUVector(s)` (`@luma.gl/graph/arrow`) cover numeric/relative temporal routes, not the local string/dictionary policy or exact Int64 two-word upload. |
| Null device/resource counters/packaging | **Not provided:** no null-WebGPU graph replacement; the texture-counter bug remains. Registry/tag observations are historical and unverified offline; crosswalk/authoring documentation remains work. |
| 15.1 Glyph/vector helpers | **Not provided:** no `walkTextGlyphs` in `@luma.gl/engine/text` or `alignGraphVectorViews`/`getGraphVectorData` in `@luma.gl/graph/gpu-core`. |
| 15.2 OOM; 15.3 pre-allocation planning; 15.4 atlas | **Not provided:** no typed class in `@luma.gl/core`, `GPUCommandGraph.planResources`/query `compilationStats`, or mipmaps/cache text-resource options in `@luma.gl/engine/text`. |
| 15.5 Geometry/trips | **Partly:** `projectCPUVector`, `tessellateCPUVectorPolygons`, `tessellateGeoArrowColumnToCPUVectors`, `projectArrowCoordinates` (`@luma.gl/graph/arrow`) do not cover globe cutting, polygon contiguity or timed trips; math.gl dimension handling is separately scoped. |
| 15.6 Deck patch; 15.7 spawn region; 15.8 globe paths | **Not provided:** deck is unchanged; `FlowParticleSimulation` (`@luma.gl/graph/experimental`) still has no spawn-region setter/option. |
| Continuation (section 3) | **Not provided:** progressive `CompiledGPUCommandGraph.createExecution` schedules a finite graph; `GPULoopOperation` still unrolls its budget. |
| Other fork candidates (section 5) | **Not established as supplied:** allocation-free plans, staging/upload pools, failed-map replacement, cache, caller-CSR SCC, core layout/resource fixes, query choose/coalesce/COUNT/USING/UNION, candidate-frontier queries and universally bounded Arrow converter dispatch require separate implementation evidence/tests. The historical fork inventory is not proof all fork changes are still absent. |

`makeGPUExpressionFromPortableAST`/`makeGPUPredicateFromPortableAST` are public through
`@luma.gl/graph/gpu-dataframe`; they do not expose private
`makeGPUQueryExpressionShaderPlan`/`addGPUQueryPredicatePass`. `GPUExpressionOperation`
(`@luma.gl/graph/gpu-core`) supports vectors and integer rounding (default truncate; nearest
ties away from zero); no direct consumer was found under `modules/*/src`.

Public `GPUCommandGraphInspector`/`GPUCommandGraphEncoding.readResource` from
`@luma.gl/graph/gpu-core` partly supply resource inspection after submission, but read current
allocations and reject reused transients. They do not satisfy intermediate retention/capture;
the local snapshot tap remains necessary for that contract.

## 1. Summary for the maintainer

Original survey basis: about 570 kernel files (2,523 compute-node constructions) built only on the public entry points of @luma.gl/graph, core and shaders. Availability rechecked against 10.0.0-alpha.16 above; these counts are historical.

For most items below a first implementation with tests already exists: the luma.gl fork these contributors came from (commit `e0ed33791`, 72 commits ahead of upstream at `704183953`) changed about 360 luma.gl-owned files in graph, core and engine and added 304 exported names. The API-availability comparison used an alpha.13 baseline; none of the inventory's measurements compared against alpha.13, and they do not establish alpha.16 performance. Some requests are now satisfied as marked above. Section 5 maps the original items to that code.

**Top five**

1. Ship a kernel-node authoring factory. Local createWGSLKernelNode is called 2,523 times and new Kernel 12 times. One binding list produces the WGSL declarations, offsets, resource uses, bounded dispatch and limit checks that name the kernel and binding.
2. Let published graph algorithms take a CSR that already lives in the caller's GPUCommandGraph. The affected degree/components/modularity/BFS/SSSP algorithms require a GPUGraphAlgorithmsTopology rebuilt from physical GPUVectors and physical outputs. Accept CSR as GraphDataViews plus node and edge masks. Public `GPUJacobiPCG` separately accepts caller CSR already; its schedule and EC03 adaptation are qualified above.
3. 64-bit and multi-key keys throughout: GPUSort (keyBits at most 32), GPUHashJoin, GPUGroupAggregation (dense uint32 keys) and GPURunLengthEncode cannot take H3, Quadbin or Int64 ids.
4. A deterministic sum mode. GPUGroupAggregation, GPUGridAggregation, GPURasterRegionMeasurements and modularity use float compare-exchange sums that depend on thread order. The local fixed-point accumulator can be contributed.
5. A GPU-gated loop that is not unrolled. GPULoopOperation unrolls maximumIterations copies, so centrality is capped at 32,768 command nodes.

**Original correctness findings (current disposition above)**

- Grid cell rule: the historical -177 formula split is closed by the adopted alpha.16 `GPU_GRID_COORDINATE_WGSL` export.
- GPUGridIndexQuery: the separate -76 reconstructed-cell-bound lead remains unverified on alpha.16; the public bounds-query reproduction below includes an interior control (CPU float32 evidence, no device readback).
- GPUHistogram: historical alpha.12 observations put exact edges one bin low (3 bins over [0, 123], values 0, 41, 82, 123 give 2, 1, 1) and a wide GPU domain puts everything in bin 0. Alpha.16 device outcomes remain unverified; proposed boundary comparisons need GPU validation.
- count semantics: unclamped in GPUHashJoin and GPUBVHQuery, clamped in GPUPointSpatialQuery and GPURasterContours. Clamped count plus requiredCount and overflow everywhere.
- getResourceCounts reads 'Texture2Ds Active', which luma.gl 10 never increments, so the texture count is always 0.
- GPUReadbackRing: the constructor does not free earlier slots if a later createBuffer throws.
- GPUTranspose: the invocation count overflows for degenerate shapes such as 268,435,456 x 1.
- GPUHaversineDistance: the raw longitude subtraction loses about 1.7 m across the antimeridian. The default radius of 6371 gives kilometres; the declaration says so, the class name and output prop do not.
- Self-loops: Degree counts a self-loop once and Modularity counts it twice.
- Metal fast math deletes float compensation terms (Kahan, Neumaier, two-sum), so a compensated sum is the plain float32 sum bit for bit; any future compensated reduction needs an integer path, and this deserves a WGSL authoring page. A reduced reproduction of an Apple back-end miscompile of small integer tables exists.

**Export requests (current disposition above)**

- Satisfied in alpha.16: `GPUSegmentedReduction`, `GPUSegmentedScan`, `GPUScatter`, `GPUJacobiPCG` from `@luma.gl/graph/gpu-core`.
- getGPUComputeDispatchWorkgroups and gateGPUCommandNodes; without them a gated loop cannot gate a published primitive.
- getPlatformFP64Arithmetic and getClassicFP64Defines; the assembler picks the safe integer variant only on Apple.
- getSpatialOperationNodes, validateRasterBand, and the GPUScalar type (public lowerers return it).
- The two consumed H3 face tables as supported data (centres and axis azimuths; draft 05), and a standalone uint64 shader module with multiply and divide (pulling in dggs costs about 70 KB per kernel).

**Ergonomics**

- One attachment convention: core primitives return nodes from getCommandNodes; domain classes only have addToGraph(): void. The local bridge temporarily replaces three graph methods (21 files).
- GPU-resident parameters, consistently: GPUGridBinning takes GPU bounds while GPUGridIndex takes literals only; raster neighbourhood, morphology and blur are capped at radius 8, which keeps five terrain and point-cloud kernels local.
- Shader modules for compute: the assembler returns a module's bindings, but nothing wires them into a command-graph node's resource declarations, so the local factory accepts only binding-free modules and kernels are composed by string concatenation with hand-kept ordering rules.
- Errors: uncaptured WebGPU errors are re-wrapped as plain Error, so out-of-memory is undetectable, and error scopes do nothing unless debug is set.
- Limits: feature level 'max' couples adapter limits to all features; the limits alone are wanted.
- Dataframe validity: new GPUTable({vectors}) sets nullable: false and the expression shader then ignores validity; .nullable is patched after construction in 12 files.
- Dataframe lifecycle: compiled queries have no compileAsync, no statistics or preflight, and no mask-only filter.
- Arrow: the analytics adapter refuses Bool, 8- and 16-bit integers, Float64, plain Utf8 and exact Int64 time.
- GPUCrossfilter needs external-mask dimensions and invalidate; all dimensions feed one GPUMask, which probably exceeds 8 storage bindings at 8 dimensions (inferred, not run).
- Test kit: GPUCommandGraph rejects the null device, so 467 node specs go through a shim that overwrites its type and limits. A recording device with WebGPU type and real limits would replace it.
- Packaging: npm i @luma.gl/graph installs alpha.5 (the latest tag) next to core 9.4.2; gpu-core has 184 value exports and no authoring guide; a luma.gl 9 to 10 entry-point crosswalk would replace the 1,875-line deck.gl 9.4 patch.

**Can be contributed from here (all with GPU specs)**

- The kernel-node factory and named-limit validators.
- Fixed-point accumulator and sorted-segment sums (57 importing files).
- Philox 4x32-10 with a CPU mirror (47 files); the published random is fract(sin(dot)).
- Exact orientation predicate in WGSL (36 files), which would resolve uncertain in GPUPairwisePointInPolygon.
- Stable multi-key sort, sorted-segment offsets and load-balanced CSR traversal.
- Dense Cholesky/LU and Jacobi eigen WGSL.
- k-means++ seeding and a tolerance stop for GPUKMeans.
- Tests: the histogram edge spec and a 33.5M-row correctness spec for scan, sort, compaction and reduction (they pass).

## 2. Details and reproductions

Proposal numbers 1–14 below identify the math proposals; 15 collects the Kepler port items (luma.gl and deck.gl). Local source paths are under `modules/analysis/src`; published graph paths are under `node_modules/@luma.gl/graph/dist`. `spec` means committed GPU or Node evidence; `replication` means CPU arithmetic from published source, not a GPU run; `reading` means source/declarations only; `measured, not committed` means a reported test-GPU run without a committed reproducing spec. [upstream-workarounds.md](./upstream-workarounds.md) owns the retirement register. Evidence remains limited to the stated adapter and inputs.

### 1. One grid cell-coordinate formula, public as WGSL

**Satisfied in alpha.16:** `GPU_GRID_COORDINATE_WGSL` from `@luma.gl/graph/gpu-core`.
The following problem/evidence describes the pre-bump implementation. All five registered copies
were replaced during the bump, with generated-source pins retained; no missing-export filing is needed.

Problem: `GPUGridBinning` (`gpu-core/gpu-grid-binning.js`, `getCoordinate`) and `GPUGridAggregation` (`gpu-core/gpu-grid-aggregation.js`, `getCoordinate`) use `min(u32((value - minimum) / (maximum - minimum) * f32(size)), size - 1u)`. `GPUGridIndex` (`gpu-core/gpu-grid-index-internals.js`, `getCoordinate`), query, point-spatial-query and spatial force layout rescale zero-straddling bounds by `max(abs(minimum), abs(maximum))` first. For `[-180, 180]`, 360 columns, x = -177, float32 gives `3 / 360 * 360 = 3.0000002` (column 3) versus `(-0.98333335 + 1) / 2 * 360 = 2.9999971` (column 2). Differences occur at 45 of 360 integer degrees and 967 of 12,407 near-edge values over seven bounds. Correctly rounded CPU division is a model; WGSL permits 2.5 ULP.

Adopted change: one rule, exported as `GPU_GRID_COORDINATE_WGSL` from `@luma.gl/graph/gpu-core`, defining `getCoordinate(value, minimum, maximum, size) -> u32`; `maximum` maps to the last cell. The five registered readers now interpolate it: `gpu-spatial-analysis/trajectory-encounters/encounter-kernels.ts`, `gpu-spatial-analysis/region-statistics/region-statistics-grid.ts`, `gpu-spatial-analysis/spatial-interpolation/gpu-inverse-distance-weighting.ts`, `gpu-spatial-analysis/point-density/point-density-kernels.ts`, `gpu-network/flow-aggregation/flow-aggregation-kernels.ts`. Retain their generated-source pins for verbatim public text and exactly one `getCoordinate`.

Evidence: reading/replication; pins `modules/analysis/test/gpu-spatial-analysis/point-density/point-density-grid-coordinate-pin.node.spec.ts`, `modules/analysis/test/gpu-spatial-analysis/trajectory-encounters/encounter-grid-coordinate-pin.node.spec.ts`, `modules/analysis/test/gpu-spatial-analysis/region-statistics/region-statistics-grid-coordinate-pin.node.spec.ts`, `modules/analysis/test/gpu-spatial-analysis/spatial-interpolation/grid-coordinate-pin.node.spec.ts`, `modules/analysis/test/gpu-network/flow-aggregation/flow-aggregation-grid-coordinate-pin.node.spec.ts`. Consequence spec: `modules/analysis/test/gpu-spatial-analysis/point-density/gpu-point-density-cell-mapping.spec.ts` covers counts and weights landing in different cells. As of 2026-11-19, no spec compared the two published classes on the same point.

As of 2026-11-20, the six-case comparison in
[`grid-coordinate-builder-agreement.spec.ts`](../modules/analysis/test/utils/grid-coordinate-builder-agreement.spec.ts)
compares published `GPUGridBinning` counts and `GPUGridAggregation` weighted sums with density
and flow cell assignments on the same uploaded points. Its
[`grid-coordinate-builder-agreement-fixture.ts`](../modules/analysis/test/utils/grid-coordinate-builder-agreement-fixture.ts)
uses one shared resident `float32x4` bounds view for the overflowing-domain case, preserving
the independent expected cell 180 and weights 7/13. This reaches the overflow arithmetic but
does not establish that either published builder accepts large literal bounds. Device
qualification of this six-case comparison remains pending; the shared-coordinate export request
stays closed.

#### Separate open correction: large literal bounds formatting (LUMA-112)

Both published builders' `getFloatLiteral` append `.0` to integer-valued numbers even when
JavaScript stringifies them in exponent notation, emitting invalid WGSL such as `-3e+38.0`.
Correct the formatter upstream to produce valid exponent-aware float literals. Acceptance must
compile **both** `GPUGridBinning` and `GPUGridAggregation` with raw literal bounds at `±2e38`
and `±3e38`, without a resident bounds view or preformatted strings, and reject any malformed
tokens such as `-3e+38.0`. Then run their count/sum readbacks against independent literal cells
and weights; compilation alone does not establish arithmetic parity. Keep the existing resident
overflow comparison as a separate arithmetic control. The local
[public density literal formatting workaround](./upstream-workarounds.md#public-density-literal-formatting)
passes a supported bounds view to both builders until a published formatter fix is qualified.

#### Separate open investigation: reconstructed bounds-query edge (LUMA-103)

Closing the shared-formula request does not close conservative query selection. In installed
alpha.16, `gpu-grid-index-query.js` reconstructs cell bounds with
`minimum * (1 - coordinate / size) + maximum * (coordinate / size)`. A CPU float32 model places
`(-76, 0.5)` in column 104 of a 360×1 grid over `[-180, 0, 180, 1]`, but reconstructs that column's
minimum as `-75.99998474121094`, above the point. The closed point-sized bounds query must return
the point's cell candidate. Expected count is one, ID zero, overflow zero. The local
`region-statistics-grid.ts` reader already expands its search by one cell; this lead establishes
no local reader miss. Alpha.16 GPU behavior remains **unverified**.

This public-only reproduction rebuilds a single-point index for the edge and an interior control
at `(-75.5, 0.5)`. A third query at `(100.5, 0.5)` against the point at `(-76, 0.5)` addresses a
distant unoccupied cell: expect count zero, overflow zero and the poisoned ID slot untouched.
This distinguishes conservative boundary selection from returning the sole point for every query.
Each result slot starts at `0xffffffff`, so a stale zero cannot masquerade as a returned ID.
Supply the actual package version, adapter/backend and browser/version and retain all three
complete records, including the index count/overflow and full ID vectors, before interpretation.

```typescript
import {Buffer, type Device} from '@luma.gl/core';
import {GPUCommandGraph, GPUGridIndex, GPUGridIndexQuery} from '@luma.gl/graph/gpu-core';

export async function reproduceGridBounds(
  device: Device,
  longitude: number,
  queryLongitude = longitude
) {
  const graph = new GPUCommandGraph<undefined>(device);
  const buffers: Buffer[] = [];
  const usage = Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST;
  const upload = (id: string, data: Float32Array | Uint32Array) => {
    const buffer = device.createBuffer({data, usage});
    buffers.push(buffer);
    return {
      buffer,
      handle: graph.importBuffer({id, byteLength: buffer.byteLength, usage}, buffer)
    };
  };
  const uint32 = (id: string, data: Uint32Array) => {
    const resource = upload(id, data);
    return {
      buffer: resource.buffer,
      view: graph.createDataView(resource.handle, {format: 'uint32', length: data.length})
    };
  };
  const read = async (buffer: Buffer) => {
    const bytes = await buffer.readAsync(0, buffer.byteLength);
    return Array.from(new Uint32Array(bytes.buffer, bytes.byteOffset, buffer.byteLength / 4));
  };
  try {
    const positions = upload('positions', Float32Array.of(longitude, 0.5));
    const query = upload('bounds', Float32Array.of(queryLongitude, 0.5, queryLongitude, 0.5));
    const cellOffsets = uint32('cell-offsets', new Uint32Array(361));
    const objectIds = uint32('object-ids', Uint32Array.of(0xffffffff));
    const indexCount = uint32('index-count', Uint32Array.of(0xffffffff));
    const indexOverflow = uint32('index-overflow', Uint32Array.of(0xffffffff));
    const output = uint32('candidate-ids', Uint32Array.of(0xffffffff));
    const count = uint32('candidate-count', Uint32Array.of(0xffffffff));
    const overflow = uint32('candidate-overflow', Uint32Array.of(0xffffffff));
    const index = new GPUGridIndex({
      positions: graph.createDataView(positions.handle, {format: 'float32x2', length: 1}),
      gridSize: [360, 1], bounds: [-180, 0, 180, 1],
      cellOffsets: cellOffsets.view, objectIds: objectIds.view,
      count: indexCount.view, overflow: indexOverflow.view
    });
    graph.add(index);
    graph.add(new GPUGridIndexQuery({
      index, kind: 'bounds',
      query: graph.createDataView(query.handle, {format: 'float32', length: 4}),
      output: output.view, count: count.view, overflow: overflow.view
    }));
    const compiled = graph.compile();
    try {
      const encoder = device.createCommandEncoder();
      try {
        compiled.encode(encoder, {parameters: undefined});
        device.submit(encoder.finish());
        return {
          point: [longitude, 0.5], query: [queryLongitude, 0.5, queryLongitude, 0.5],
          indexCount: (await read(indexCount.buffer))[0],
          indexOverflow: (await read(indexOverflow.buffer))[0],
          count: (await read(count.buffer))[0], ids: await read(output.buffer),
          overflow: (await read(overflow.buffer))[0]
        };
      } finally {
        encoder.destroy();
      }
    } finally {
      compiled.destroy();
    }
  } finally {
    for (const buffer of buffers) buffer.destroy();
  }
}

export async function collectGridBoundsEvidence(
  device: Device,
  evidence: {packageVersion: string; adapter: string; browser: string}
) {
  const fixtures = [
    {longitude: -76, queryLongitude: -76, expectedCount: 1, expectedIds: [0]},
    {longitude: -75.5, queryLongitude: -75.5, expectedCount: 1, expectedIds: [0]},
    {longitude: -76, queryLongitude: 100.5, expectedCount: 0, expectedIds: [0xffffffff]}
  ];
  const records = [];
  for (const fixture of fixtures) {
    const result = await reproduceGridBounds(device, fixture.longitude, fixture.queryLongitude);
    const record = {
      ...evidence, ...result, expectedCount: fixture.expectedCount, expectedIds: fixture.expectedIds
    };
    records.push(record);
    console.log(JSON.stringify(record));
  }
  if (records.some(record => record.indexCount !== 1 || record.indexOverflow !== 0 ||
      record.count !== record.expectedCount || record.overflow !== 0 ||
      record.ids.length !== record.expectedIds.length ||
      record.ids.some((id, index) => id !== record.expectedIds[index]))) {
    throw new Error('Grid bounds mismatch; retain all three version-labelled records above');
  }
  return records;
}
```

This is a queued device probe, not a confirmed alpha.16 failure. A confirmed miss warrants
conservative upstream candidate selection consistent with the builder before exact point
filtering. Preserve legitimate candidates near boundaries rather than imposing an unqualified
epsilon. Keep this investigation separate from filing the already satisfied formula export.

### 2. Graph algorithms on caller CSR with masks; one self-loop rule

Problem: `GPUGraphAlgorithmsTopology` (`gpu-graph-algorithms/gpu-graph-topology.d.ts`) rebuilds CSR from physical `GPUVector<'uint32'>` edge columns into separate physical outputs. A caller's `GraphDataView` CSR or per-frame closure mask cannot be supplied. For a masked road network, subtracting offsets counts closed slots too. Degree documents one contribution per undirected self-loop (`gpu-graph-degree.d.ts`); Modularity documents two (`gpu-graph-modularity.d.ts`). One vertex with a self-loop and an edge to a second has degree 2 but modularity volume 3.

Change: accept `GPUGraphAlgorithmsTopologyView` beside `GPUGraphAlgorithmsTopology`: CSR `GraphDataView`s (`offsets`, `neighbors`, optional `edgeIds`, `edgeWeights`), `nodeMask` and `slotMask` or a documented `neighbor >= nodeCount` sentinel; outputs as `GraphDataView`s on the caller's graph. State one self-loop rule, or expose `selfLoops: 'once' | 'twice'`. Use deterministic modularity accumulation (integer atomics unweighted; fixed-order tree weighted). This also supports REDCAP/AZP cut edges.

Retires `createMaskedWeakComponentNodes` in `analysis-core/kernels/masked-weak-components.ts` (five callers), degree/modularity in `gpu-network/network-statistics/gpu-network-statistics.ts` (`GPUNetworkStatistics`), degree in `gpu-network/network-analysis/gpu-network-analytics-columns.ts` (`GPUNetworkAnalyticsColumns`), and potentially hop traversal in `gpu-network/network-analysis/gpu-network-neighborhood.ts` (`GPUNetworkNeighborhood`) and CSR construction in `gpu-network/network-noding/gpu-network-noding.ts` (`GPUNetworkNoding`). Enables PageRank, core number and label propagation on network views.

Evidence: reading; local-rule specs `modules/analysis/test/gpu-network/network-analysis/gpu-network-analytics-columns.spec.ts` and `modules/analysis/test/gpu-terrain/terrain-decode/spike-repair-component-labels.spec.ts`. No same-input comparison with a published algorithm.

### 3. `GPUHaversineDistance`: units and seam-safe subtraction

Problem: default radius 6371 means kilometres on a rounded radius (`geospatial/gpu-haversine-distance.js`, `GPUHaversineDistance`); the declaration states the unit, but the output prop does not. New York (-73.98, 40.75) to Paris (2.35, 48.85): 5833.150 by default versus 5,833,158.6 on the IUGG mean radius in metres; 8.1 m is radius error, 1000 is the unit factor. Raw `rightDegrees - leftDegrees` for (179.94722, 0) to (-179.99805, 0), 6.1 km apart, loses 1.53e-5 degrees or 1.7 m at the equator.

Change: retain radius 6371, document kilometres at the output and export `GPU_HAVERSINE_DEFAULT_RADIUS_KILOMETERS = 6371`; compute small seam-side distances before adding, guarded against reassociation. Retires the sphere path of `GPUGeodesicPairs` in `gpu-spatial-analysis/geometry-measures/gpu-geodesic-pairs.ts`; the model is `geodesicLongitudeDelta` in `gpu-spatial-analysis/geometry-measures/geodesic-wgsl.ts`. Bearings, midpoints, Vincenty and rhumb lines remain.

These are three separate contracts: radius units scale the result; longitude wrapping chooses an
equivalent angular separation; accurate float32 seam subtraction preserves the small separation
before rounding. Wrapping a previously rounded `rightDegrees - leftDegrees` cannot recover lost
bits, and changing the radius does not fix that subtraction. Any sphere-path delegation above
requires same-radius device parity for `(179.94722, 0)` → `(-179.99805, 0)`, the reverse pair and
nearby seam controls against an independent binary64 reference of the uploaded float32 endpoints.
Metal may reassociate floating expressions, so source spelling alone is insufficient evidence.

The fork at `e0ed33791` was checked locally via `git show` in
`vis.gl-build/luma-contributors`: `geospatial/gpu-haversine-distance.ts` adds `units` and an IUGG
`sphereRadius` default, but still generates `let deltaDegrees = rightDegrees - leftDegrees;`
for float32 inputs and exposes no wrapping option. The older alpha.13 comparison's wrapping
description is unsupported by that implementation; it is not a seam-safe replacement. Raw
Float64 subtraction in published alpha.16 is a different input route and does not certify the
float32 acceptance pair.

For those uploaded float32 endpoints, binary64 subtraction and wrapping gives
`0.0547332763671875°`; wrapping the already rounded float32 subtraction gives
`0.05474853515625°`. Use the same radius for both results when qualifying replacement.
Device parity remains outstanding.

Evidence: float64 replication agrees within 3e-16 on six pairs; source reading. Local seam specs `modules/analysis/test/gpu-spatial-analysis/geometry-measures/gpu-geodesic.spec.ts` (1.70 m error) and `modules/analysis/test/gpu-spatial-analysis/geometry-measures/parity-wp2-geodesic-seam.parity.spec.ts`; published seam case not run.

### 4. `GPUFiniteDifference2D`: row spacing and no-data

Problem: constant `spacing` pair and no validity ports (`gpu-core/gpu-finite-difference-2d.d.ts`). One arc-second cells are 30.9 m wide at the equator and 15.4 m at latitude 60: a factor of two. NaN/sentinel samples yield unflagged invalid derivatives. Near 8,000 m, float32 steps are about 5e-4 m; `f(x+h) - 2 f(x) + f(x-h)` cancels terms near 16,000.

Change: `spacing` as a numeric pair, a pair with per-row `GraphDataView<'float32'>` east-west spacing, or `GPURasterMetadata` for derived row widths; optional `validity`/`outputValidity` with `'propagate'`; centre-relative zero-sum stencils `(f(x+h) - f(x)) - (f(x) - f(x-h))`, also useful in `GPURasterNeighborhood`. Retires at most the finite-difference stencils of `gpu-terrain/terrain-curvature/gpu-terrain-curvature.ts` (`GPUTerrainCurvature`), with a matching first-derivative stencil at one-sided borders only. Second-derivative border parity is not established: on a unit grid with rows `[0,1,8,27]`, local quadratic extrapolation gives second derivative 6 at the left edge, while the published four-sample formula gives `2·0 − 5·1 + 4·8 − 27 = 0`. Its selectable partial-derivative estimators (default `'evans-young'`), second-difference Hessian and curvature formulas remain local. Evidence: reading only; no parity spec.

### 5. `GPURasterMorphology`: radius and valid count

Problem: radius is an integer 0–8 (`gpu-raster/gpu-raster-morphology.js`, `GPURasterMorphology`); invalid centres stay invalid under `'ignore'`, so composition fails across holes. In a 17-cell row, cell 0 = 1, cells 1–15 invalid, cell 16 = 9: radius-16 erosion gives 1 at cell 16, but two radius-8 passes give 9 because cell 8 stays invalid. Valid-sample counts are absent.

Change: larger radii (e.g. van Herk/Gil-Werman for squares), or internal composition that propagates across invalid centres; optional `output.validCount` uint32 or completeness flag. Retires square windows in `pointcloud-analysis/gpu-pointcloud-ground.ts` (`GPUPointCloudGround`, radius up to 32, `GPU_POINT_CLOUD_GROUND_STATUS_INCOMPLETE_WINDOW`); enables delegation of quantiles 0/1 in `raster-analysis/gpu-raster-rank-filter.ts` (`GPURasterRankFilter`). Evidence: reading; the 17-cell case is not run. `modules/analysis/test/gpu-pointcloud/pointcloud-diamond-morphology.spec.ts` pins working `GPUPointCloudSMRF` composition through published erosion/dilation, radius at most 8 per pass, bit-identical for radii 1–37.

### 6. `GPUHistogram`: domain views and bin edges

Problem: literal, `'auto'` and two-row GPU domains use `u32((value - minimum) / (maximum - minimum) * f32(BIN_COUNT))` (`gpu-core/gpu-histogram.js`, `binCalculation`). Historically on alpha.12, for 16 bins over `[-1e38, 1e38]`, values 0, 1, -5e37, 5e37 gave bins 8, 8, 4, 12 with literals and all bin 0 with a view. Alpha.16 retains the quotient but has no new device readback here. Reciprocal multiplication plus subnormal flushing is an inference: `1 / 2e38 = 5e-39`, range above 2^126; generated code was not inspected for this case.

Exact float32 edges also land low. Inputs are `n + 1` edges of `n` bins, width `w`, from `minimum`; NumPy 2.5.4 `numpy.histogram` with float32 inputs gives one per bin and two in the last. Observed on Apple M3 Pro/headless Chromium with alpha.12:

| Bins | Width, minimum | Low edge indices: literal | Low edge indices: GPU view |
| --- | --- | --- | --- |
| 49 | 1, 0 | none | 7, 14, 27, 28, 35, 41, 47 |
| 49 | 8, 1024 | none | 7, 14, 27, 28, 35, 41, 47 |
| 3 | 41, 0 | 1, 2 | 1, 2 |
| 6 | 41, 0 | 1, 2, 4 | 1, 2, 3, 4 |
| 12 | 41, 0 | 1, 2, 4, 7, 8 | 1, 2, 3, 4, 6, 8 |
| 7 | 83, 0 | 1, 2, 4 | 1, 2, 4 |

Both forms are correct for 7, 255, 1000 bins of width 1 from 0; 12 of width 0.25 from -16; 30 of width 8 from 1024. Smallest case: 3 bins `[0, 123]`, values 0, 41, 82, 123: expected 1, 1, 2; observed 2, 1, 1. At 49 bins `[0, 49]`, values 0–49, only the view puts 7 in bin 6, 14 in bin 13, etc. Literal/view lists match constant `BIN_COUNT / range` multiplication/runtime reciprocal multiplication respectively in these rows, not every case. Correctly rounded CPU division/multiplication instead misses edge 27 of 49 and edges 251, 253, 502, 506, 511 of 1000 width-1 bins, although the GPU gets the latter case right. All three evaluations have boundary errors.

Change: decide bins using boundary comparisons, potentially `k * range <= (value - minimum) * BIN_COUNT < (k + 1) * range` when products are finite and exact, with scaling before dangerous subtraction/products and an explicit representable-edge rule. The remedy is not GPU-verified. Add wide-domain, every-edge and adjacent-edge tests for both forms; [draft 02](./upstream-drafts/02-histogram-edges-and-domains.md) supplies a public runner. Preserve existing domain/output offsets, masks and explicit edges; NaN-as-finite-extent policy remains a separate request. Only after parity could this retire `BIN_WGSL` in `dataframe-analysis/column-profile/column-profile-kernels.ts` (`GPUColumnProfile` histogram).

Evidence: `modules/analysis/test/gpu-core/gpu-histogram-domain-view.spec.ts` has 12 tests and runs published literal/view forms twice per case and pins observed counts, including failures of the desired convention; flip assertions when fixed. `modules/analysis/test/gpu-dataframe/column-profile/gpu-column-profile.spec.ts` documents the boundary reason but the reproduction does not run its local kernel. Confirmation needed on a second adapter; domains bracketing 2^126, narrow domains far from zero and bin-centre inputs remain untested.

### 7. `GPUKMeans`: seeding and tolerance

Problem: `'evenly-spaced'` valid-row seeds only, stop on unchanged labels, empty cluster retains centroid (`gpu-vector-search/gpu-k-means.js`, `GPUKMeans`). For 80 blob-A rows then 10 B then 10 C, `clusterCount: 3`, `floor(index * rowCount / clusterCount)` selects 0, 33, 66, all A. Recovery depends on geometry; case not run.

Change: `seed: 'evenly-spaced' | 'k-means++'`, `seedValue: number`, GPU distance pass and counter-based weighted draw per centroid; `tolerance?: number` on largest centroid shift with status. Optional restarts, inertia, per-row squared distance and empty-cluster relocation to the farthest row. Retires matching portions of `GPUSpatialKMeans` in `gpu-spatial-analysis/spatial-clustering/gpu-spatial-kmeans.ts` (2D points). Evidence: reading; no same-input numeric comparison.

### 8. `GPURasterContours`: levels and world coordinates

Problem: one `level` per instance and local pixel-centre output (`gpu-raster/gpu-raster-contours.d.ts`). Ten levels on 4,096 × 4,096 cells means ten scans of 16.8 million cells, ten guessed segment capacities and a world-transform pass per output, with a local origin needed for precision. `gpu-raster/gpu-raster-contours.js` scales all four corners and the level by the largest magnitude before the saddle determinant, so division rounding can lose an integer-raster tie that should connect the high diagonal.

Change: `levels: number[] | GraphDataView<'float32'>`, shared segment output with level indices/offsets and `requiredSegmentCount`; `coordinates: 'pixel' | 'world'` relative to a stated origin. Take unscaled differences and scale only when products risk overflow (local threshold 1e18). Later options: global edge IDs for stitching and two-level bands. Retires segment output in `raster-analysis/isolines/gpu-isolines.ts` (`GPUIsolines`); `gpu-isobands.ts` (`GPUIsobands`) needs bands. Local classes use the bilinear saddle rule; this alone does not establish numeric parity.

Evidence: declaration reading; `modules/analysis/test/gpu-raster/isolines/gpu-isolines.spec.ts` supplies the random-field tie regression, and `modules/analysis/test/gpu-raster/isolines/isolines-saddle-rule.spec.ts` covers saddle topology and integer ties. The latter checks corners 7, 4, 7, 1 at level 5 (differences 2, -1, 2, -4; determinant 0) and the mirrored 4, 7, 1, 7 locally; it does not assert published tie behavior. `GPUTerrainContours` in `gpu-terrain/terrain-analysis/gpu-terrain-contours.ts` already delegates one level.

### 9. `GPUHashJoin` and `GPUGroupAggregation`: wide and sparse keys

Problem: packed uint32 join keys and dense uint32 aggregation keys below `output.length` (`gpu-core/gpu-hash-join.d.ts`, `gpu-group-aggregation.d.ts`). H3 `0x8928308280fffff`, Quadbin and Arrow Int64 do not fit. Keys 7 and 4,000,000,000 require four billion output rows. Float compare-exchange sums depend on thread order; join has no row mask.

Change: `uint32x2` keys, low word first in storage, for hash index/join/grouping; compact sparse `(key, statistic)` output with count/overflow; left mask and 1:n aggregates over repeated right keys. Retires `GPUKeyLookup` in `dataframe-analysis/key-join/gpu-key-join.ts`, count/sum/mean/min/max of `GPUGroupStatistics` in `dataframe-analysis/group-statistics/gpu-group-statistics.ts` (moments/order statistics remain), and table build of `GPUCellAggregation` in `gpu-spatial-analysis/cell-aggregation/gpu-cell-aggregation.ts`. Evidence: declaration reading.

### 10. Exact point-in-ring fallback

Problem: `GPUPairwisePointInPolygon` returns `uncertain` (3) for arithmetic-envelope overflow or degeneracy (`geospatial/gpu-pairwise-point-in-polygon.d.ts`). A square `(10, 0)-(14, 4)` with a second ring `[(11, 1), (12, 2)]` has an area-free ring: published classification is uncertain; exact-orientation even/odd classification skips it.

Change: integer exact orientation when the fast filter fails; `EXACT_ORIENTATION_WGSL` in `analysis-core/kernels/exact-orientation-wgsl.ts` computes six products exact in 48 bits, aligns exponents and compares two wide unsigned accumulators. Keep uncertain only for malformed offsets; expose `boundaryRule: 'closed' | 'open' | 'half-open'`. Retires inline classification in `gpu-spatial-analysis/spatial-join/gpu-point-in-polygon-join.ts` (`GPUPointInPolygonJoin`) and fallback in `gpu-spatial-predicate-join.ts` (`GPUSpatialPredicateJoin`), with indexed candidate pairs as a separate request.

Evidence: `modules/analysis/test/gpu-spatial-analysis/spatial-join/gpu-point-in-polygon-classifier-agreement.spec.ts` compares local and published classification on four scenes: shared edges/vertices, projected metres, one float32 step either side of an oblique edge, and the degenerate ring (the disagreement).

### 11. Standalone uint64 and H3 tables as data

Problem: WGSL has no int64; `dggs_u64_*` and `fp64_u64_*` use `vec2u(high, low)` but omit multiply/divide and require whole shader modules. Epoch-millisecond division, accumulators and key comparison need local words. H3 centres/azimuths exist only in `dggs_h3_get_face_center_geo` and `dggs_h3_get_face_axis_azimuths`; downstream parsing now validates exactly 20 finite rows and fails at import if shader text changes. That guard prevents silent empty arrays but does not make shader formatting a public data contract.

Change: standalone `uint64` add/subtract/negate/compare, `shiftLeft`, `shiftRight`, `multiply32` (64 by 32), `divide32`, float32 conversion and storage-order converters; dggs/fp64arithmetic depend on it. Export the two consumed H3 centre/azimuth tables from `@luma.gl/shaders` as frozen 20-row arrays and generate WGSL from them; [draft 05](./upstream-drafts/05-h3-face-tables.md) names the exact data contract (DGGS-115), separately from uint64 arithmetic. Retires `UINT64_WGSL` in `analysis-core/kernels/uint64-wgsl.ts`, word helpers in `analysis-core/kernels/cell-key-wgsl.ts` (`getFixedPointWGSL`) and `gpu-spatial-analysis/cell-aggregation/cell-keys.ts` (`CELL_KEY_WGSL`), `TERRAIN_SUMMED_AREA_WGSL_HELPERS` in `gpu-terrain/topographic-position/terrain-summed-area-table.ts`, `add64`/`subtract64` in `dataframe-analysis/pair-statistics/gpu-ripley.ts`, `divideUnsigned64`/`negateInt64` in `dataframe-analysis/calendar-buckets/gpu-calendar-buckets.ts`, plus `parseDggsFaceTable` in `gpu-spatial-analysis/cell-indexing/h3-index-extended-wgsl.ts` after the table export passes parity.

Evidence: `modules/analysis/test/utils/wgsl-words-search.spec.ts` compares words with BigInt; `modules/analysis/test/gpu-spatial-analysis/cell-indexing/h3-index-extended.node.spec.ts` requires 20 parser rows. Published side: reading.

### 12. One meaning of count

| API | Current count | Unclamped value |
| --- | --- | --- |
| `GPUHashJoin` (`gpu-core/gpu-hash-join.d.ts`) | Required matches before truncation | `count`; `overflow` flags excess |
| `GPUBVHQuery` (`gpu-core/gpu-bvh-query.d.ts`) | Full match count | `count` |
| `GPUPointSpatialQuery` (`geospatial/gpu-point-spatial-query.d.ts`) | Clamped to `output.ids.length`, safe for indirect draw | `totalCount` |
| `GPURasterContours` (`gpu-raster/gpu-raster-contours.d.ts`) | `segmentCount = min(required, capacity)` | `requiredSegmentCount` |

Problem: 1,000 matches into capacity 600 means `i < count` reads 400 rows past join output but safely consumes point-query output. Change: clamped `count`, unclamped `requiredCount`, `overflow = requiredCount > capacity`; add requiredCount to hash join, BVH query and grid-index query before a documented count change. Retires nothing: `GPUCompactOutput` in `analysis-core/kernels/gpu-contributor-types.ts` already follows this rule through `status.requiredCount`. Evidence: four declarations, reading only.

### 13. Four shader-compiler facts

`research/math-correctness-review-2026-11-10/shader-compiler-findings.md` includes generated Metal and reductions. Facts 13.1, 13.2 and 13.4 are Metal fast math: the generated Metal gives the IEEE result with fast math off; Chromium 151 exposes no switch. Fact 13.3 is an Apple GPU back-end miscompile of correct Metal, independent of fast math; browser/Apple reports are drafted and unfiled. Reassociation and inexact division are permitted by WGSL. The table failure is not. All measurements cover one machine, OS and browser version.

**13.1 Compensation disappears.** Kahan `(next - sum) - term`, Neumaier `(sum - next) + term` and float two-sum are optimized to zero. A million terms, mean 1234.567, one quarter negative: true sum 1,234,041,587.7; correction 0, plain/compensated GPU total 1,234,025,472 (16,115 off); IEEE serial CPU 1,234,033,152. Integer-significand rounding errors give a total 0.44 off before final rounding; fixed point gives nearest float32. The local H3 guarded double-float path has 0 h3-js mismatches where float32 misses more than 0.5%.

Cure/change: integer fixed point with known range, per-addition error from integer significands, or guarded fp64arithmetic two-sum; publish the device-appropriate fp64 defines/runtime-zero guard (`LUMA_FP64_CODE_ELIMINATION_WORKAROUND`). A float `x + ONE * 0.0` barrier folds away. Test GPU sums at a million terms, not just CPU twins. Retires applicable `getAdditionRoundingError` in `analysis-core/kernels/addition-rounding-error-wgsl.ts` and fixed-point paths in `analysis-core/kernels/fixed-point-accumulator.ts` when published reductions/scans meet the needed accuracy. No installed published graph kernel contains Kahan/Neumaier/two-sum by text search; future compensated sums need this protection.

Evidence: `modules/analysis/test/utils/addition-rounding-error-wgsl.spec.ts` pins the totals and 4,022 signed pairs bit for bit; `modules/analysis/test/gpu-spatial-analysis/linear-referencing/gpu-linear-referencing.spec.ts` covers 86,400 equal segments, a prefix near 1.2 million off by 16.43; `modules/analysis/test/gpu-spatial-analysis/geometry-measures/gpu-geometry-measures-summation.spec.ts`, `modules/analysis/test/gpu-spatial-analysis/shape-descriptors/shape-descriptors-moments-summation.spec.ts`, `modules/analysis/test/gpu-spatial-analysis/cell-indexing/h3-index-extended.spec.ts`.

**13.2 Differences regroup.** `(y - mean) - intercept - slope * (x - xmean)` becomes `y - (mean + intercept + ...)`, even across statements. For response near 10,000 and residuals about 0.002, residuals become multiples of the 10,000 float32 step; Chow F is 12% off and spatial-error first-stage `s2` 5% off. Values 1e7, 1e7 + 1, +2, +3 have mean 1e7 + 1.5, stored rounded with correction -0.5; first z-score becomes -1.549 rather than -1.162.

Cure/change: XOR the first difference's bits with runtime-zero buffer data (a literal folds away). Publish a rounded-difference/kept-intermediate helper; retires `getRoundedDifference` in `gpu-spatial-analysis/spatial-regression/rounded-difference-wgsl.ts` and the same guard in `dataframe-analysis/group-statistics/group-statistics-moments.ts`. Evidence: `modules/analysis/test/gpu-spatial-analysis/spatial-regression/math-review-restricted-regimes.spec.ts` ("response mean far above its residuals"), `modules/analysis/test/gpu-spatial-analysis/spatial-regression/math-review-error-ml-high-fit.spec.ts` ("response mean of 10000"), `modules/analysis/test/gpu-dataframe/group-statistics/gpu-group-statistics.spec.ts` (1e7). Lag-ML, GM and 2SLS residual chains are untested; tested `GPUOrdinaryLeastSquares` does not regroup. No published kernels were tested for this shape.

**13.3 Integer tables lose stores.** Function-local or `var<private>` `array<u32, N>`/`array<i32, N>`, filled only by constant-trip loops then read at unresolved indices, returns zeros for any stored value (index, constant, sentinel, buffer copy). Reduced WebGPU: lost at 33 to about 250 entries with 3–96 loop iterations; right through 32. Louvain: lost from 12 through every size tried to 48; from 33 when a helper with two float arrays is removed. No size is generally safe. Tint's Metal and Metal front-end intermediate are correct; the Apple back end loses stores after zero initialization when unrolling. Metal `#pragma clang loop unroll(disable)` cures it; fast math on/off does not.

Cures: literal initializer/module `const` for known tables; runtime-bound fill such as `N * min(arrayLength(&binding), 1u)` otherwise; remove identity tables. `var<private>` alone is not a cure. Float/vector tables, later data-dependent stores, simultaneous storage writes and conditional stores also passed, for unknown reasons. Current audited kernels pass their permitted shapes; this proves only those compiled inputs. Retires no luma.gl replacement: keep the authoring rule and send the reduced bug to browser/Apple. Published identity fills in `gpu-physics/sphere-world-shaders.js`, `gpu-core/gpu-text-selection.js`, `gpu-core/gpu-sort-chunks.js` write storage, not local arrays; no published fault is known from that limited text search.

Evidence: findings report above; `modules/analysis/test/analysis-core/shader-compiler-table-fill.spec.ts`, `modules/analysis/test/analysis-core/shader-compiler-table-shapes.spec.ts`, `modules/analysis/test/analysis-core/table-audit/*.spec.ts`, `modules/analysis/test/utils/function-scope-table-readback.spec.ts` (one-binding reduction, dense LU pivot table, natural neighbour, phenology), `modules/analysis/test/gpu-network/network-optimization/gpu-network-louvain-visit-order.spec.ts` (13 sizes, 2–512, five seeds against TypeScript Fisher-Yates).

**13.4 Division is inexact.** A region mean computes `6.0 / 3.0 = 1.9999999`; WGSL permits 2.5 ULP. CPU twins cannot demand exact quotients or serial loop sums. Cure/change: products and comparisons at bin/cell boundaries (`k * range <= (value - minimum) * count < (k + 1) * range`), with product-rounding limits respected; no local code retirement by this observation alone. Evidence is an observation, not an exact-result assertion: `modules/analysis/test/gpu-spatial-analysis/spatial-regionalization/gpu-region-partition-evaluation-nonfinite.spec.ts` uses relative 1e-6; recorded in `research/math-correctness-review-2026-11-10/point-process-clustering.md`.

Request for all four: one WGSL authoring page with cures and CPU-twin limits, public runtime-zero helpers, and GPU tests large enough to expose summation, centring and local-table failures. The table miscompile belongs with browser/Apple, not a luma.gl implementation fix.

### 14. Smaller gaps

These are source/declaration readings unless a measurement is explicitly labelled. Available
primitives, missing contracts and performance requests have different acceptance work; none of
the following promises wholesale downstream deletion.

**Available adoption candidates**

| Published surface | Remaining acceptance work |
| --- | --- |
| `GPUSegmentedReduction`, `GPUSegmentedScan`, `GPUScatter` from `@luma.gl/graph/gpu-core` | Eligible ungated sums need gather, empty-segment, prefix, tail and accumulation-order parity. Gated sums need dispatch/output adaptation; fixed-point exact sums remain local. Scatter requires unique destinations for deterministic results. |
| `GPUJacobiPCG` from `@luma.gl/graph/gpu-core` | Export and caller-CSR/chunked inputs exist. On the reported two-row diagonal fixture, budgets 1/2/4 declare 51/85/153 nodes (`17 + 34I`); 1,024 declares 34,833, above the local 32,768-node budget before setup. Local PCG uses five nodes per iteration. Require compatible compilation/continuation plus independent true-residual, actual iteration count, stopping, grounding, normalization and disconnected-input parity. A warm start does not resume Krylov state. |
| `GPUGridIndex` caller-owned `cellOffsets`, `objectIds`, `count`, `overflow` views | Already supplied by alpha.16. Remaining point-cloud lattice semantics are listed below; the separate bounds-query correctness lead is LUMA-103 above. |
| `GPUCommandGraphCompileOptions.kernelMetadata` and `preflight.nodes[].kernel` | Available opt-in for nodes declaring metadata. Preserve coverage per primitive and local annotation precedence; typed resource links, constants and semantic formula/provenance are separate contracts. |

**Missing contracts**

Local paths below identify code that could delegate or simplify only after the stated contract
and same-input parity. No new export request is made for the available primitives above.

| Published API | Request | Local code it would retire or simplify |
| --- | --- | --- |
| `GPUGridIndex` | Cell-size mode `origin + k * cellSize`, half-open domain, ascending IDs within cells, Z-ignoring raster mode and named limits; caller-owned offsets/ID views already exist | `createPointCloudGridIndex`, `pointcloud-analysis/operations/pointcloud-grid-index.ts` |
| `GPUGridBinning` | Per-frame active grid size | `GPUFlowAggregation`, `gpu-network/flow-aggregation/gpu-flow-aggregation.ts` grid zones |
| `GPUGridBinning`, `GPUGridAggregation` | Per-row cell key or deterministic fixed-point sum instead of float compare-exchange | `GPUPointDensity`, `gpu-spatial-analysis/point-density/point-density-kernels.ts` (`createPointDensityGridKeysNode`) |
| `GPURasterRegionMeasurements` | Exact deterministic sums and per-row cell area | `GPUTerrainCatchmentSummary`, `gpu-terrain/terrain-processing/gpu-terrain-catchment-summary.ts` area/mean/extrema |
| `GPURasterDilation`, `GPURasterErosion` | Mask and GPU-gated fixpoint reconstruction | `GPURasterReconstruction`, `raster-analysis/gpu-raster-reconstruction.ts` |
| `GPURasterContrast`, `GPURasterGlobalPercentile` | Statistics window/region mask, per-frame mode, `[0, 1]` output, caller histogram and per-frame percentile | `GPURasterStretch`, `raster-analysis/raster-stretch/gpu-raster-stretch.ts` |
| `GPUFiniteDifference2D` | Per-row spacing and no-data (proposal 4) | `GPUTerrainFlowField`, `gpu-terrain/terrain-flow-field/gpu-terrain-flow-field.ts` gradient |
| `GPUGallopingSearch` | Convenience for an implicit query ramp `0..segmentCount` without caller-owned segment and validation views; 64-bit keys are a separate missing format contract | `createSortedSegmentOffsetsNode`, `analysis-core/kernels/sorted-segment-offsets.ts` |
| `GPUDGGSCellProjection` | Invalid-cell value option; boundary/vertex-count outputs | `GPUCellGeometry`, `gpu-spatial-analysis/cell-indexing/gpu-cell-geometry.ts` |
| `GPURasterGradient`, `GPURasterNeighborhood` | One-sided/extrapolating border | `GPUTerrainDerivatives`, `gpu-terrain/terrain-analysis/gpu-terrain-derivatives.ts`; `GPUReliefShading`, `gpu-terrain/terrain-illumination/gpu-relief-shading.ts` padded bands |
| `GPURasterBandMath` | Per-frame operation; unary, power, minimum, maximum | `GPURasterArithmetic`, `raster-analysis/raster-algebra/gpu-raster-arithmetic.ts` |
| `GPURasterNeighborhood` | Minimum, maximum, range, standard deviation, valid count | `GPUFocalStatistics`, `gpu-spatial-analysis/spatial-interpolation/gpu-focal-statistics.ts` |
| `GPUCompaction` | Bounded capacity and required count | `GPUNetworkSubgraphFilter`, `gpu-network/network-subgraph-filter/gpu-network-subgraph-filter.ts` scan/scatter |
| `GPUReduction` | Two-word 64-bit integers beside `GPUScanUint64` | Network K edge-draw prefix, `gpu-network/network-k-function/gpu-network-k-function.ts` (`GPUNetworkKFunction.getCommandNodes`, `chunk-prefix` variant) |
| `GPUPointCloudVoxelDownsample` | Per-voxel centroids/attribute means | `GPUPointCloudVoxelMerge`, `pointcloud-analysis/gpu-pointcloud-voxel-merge.ts` input |
| `compileProjectionPlan` | `destinationOrigin` so error-budget validation uses the actual relative origin | `GPULocalFrameProjection`, `gpu-spatial-analysis/geometry-measures/gpu-local-frame-projection.ts` override |
| `GPUGraphAlgorithmsSingleSourceShortestPath` | Multiple sources/start costs, per-frame views, compact frontier | `GPUNetworkReachability`, `gpu-network/network-reachability/gpu-network-reachability.ts` |
| `GPUGraphAlgorithmsModularityOptimization` | Multilevel coarsening | `GPUNetworkLouvain`, `gpu-network/network-optimization/gpu-network-louvain.ts` (32 vertices) |
| `random` shader module | Counter-based Philox and open-interval unit float | `PHILOX_WGSL`, `getPhiloxOpenUnitFloat`, `analysis-core/kernels/philox-random.ts` |
| `GPUCommandGraph` | Constants, typed resource/view links and semantic formula/provenance beyond existing opt-in kernel metadata | Per-graph kernel record table, [upstream-first-recording.md](./upstream-first-recording.md) |

The explicit-input lower-bound contract already exists in alpha.16: sorted keys `[0,2]`, queries
`[0,1,2,3]`, segment `[0,2,0,4]`, `maximumQueryCount: 4` and an initialized zero validation word
with `preserveValidationErrors: true` declare one search node. The independent lower-bound
expectation is `[0,1,1,2]`, counting keys strictly below each query. Host node declaration was
checked offline; device output was not read. The convenience request is to generate the query
ramp implicitly and dispense with caller-owned segment and validation views, not to add basic
dense-key search or an already available single search dispatch. Measure the avoided buffers,
uploads and scheduling separately from adding 64-bit keys (current scalar formats are uint32 and
float32).

**Performance and scale investigations**

The observations below retain their original measurement scope. Qualify the proposed benefit on
alpha.16 with named input sizes and adapters before adopting or crediting a retirement. A public
building block or a historical fork timing alone does not establish a current speedup.

| Published API | Request or measurement | Local consumer |
| --- | --- | --- |
| `GPUGallopingSearch` | Measure the allocation/upload and scheduling saving from an implicit query ramp without caller-owned segment and validation views; one search node already exists for explicit inputs. Measure separately from 64-bit key support | `createSortedSegmentOffsetsNode`, `analysis-core/kernels/sorted-segment-offsets.ts` |
| `GPUScan` (and sort/reduction recursion) | Size-independent recursion levels driven by runtime parameters, so pipeline counts do not grow with input size (36 to 38 variants between 1M and 2M rows, scalability pass SC) | Chunked contributors' constant-pipeline guarantees, for example `GPUGeographicallyTemporallyWeightedRegression` and `GPUKernelDensity` |
| `GPUBVH` | Two-dimensional or bounded refit dispatch; the one-dimensional refit exceeds `maxComputeWorkgroupsPerDimension` above 16,777,216 leaves | `GPUFuzzyContiguity` dispatch pre-check (`gpu-spatial-analysis/spatial-weights/gpu-fuzzy-contiguity.ts`) |
| `GPUReduction` and sort kernels | Runtime chunk, recursion-level and input-size bounds instead of specializing shader source for each size; stable pipelines across unequal chunks | Profile, statistics and tiled aggregation contributors still inherit published reduction/sort shader variants (IMP F-DF/F-STATS) |
| `GPUCommandGraph` | Measure scratch lifetime with existing `dependsOn` before requesting a memory-aware scheduling policy for independently seeded batches | Historical CSR visibility planned 35.7 GB while independent centrality batches stayed live simultaneously; explicit dependencies bounded their lifetimes (IMP SC3-VIS), not an alpha.16 measurement |

**Suggested filing order**

1. Draft **02** (proposal 6): correctness investigation for histogram arithmetic. Keep alpha.12 observations separate from queued alpha.16 literal/view and adjacent-edge probes; record every bin, version and adapter. Second-adapter confirmation remains acceptance work.
2. Draft **03** (proposal 2): caller-CSR/mask semantics, live-mask resubmission, sentinel/completeness rules and two BFS hops; downstream retirement is conditional on parity.
3. Draft **01**: packed transfer performance request. Row-copy source evidence exists; measure changed snapshots, offsets and strides on alpha.16. Current benefit is unmeasured and no downstream source removal is credited.
4. Draft **04**: typed import/CPU construction ergonomics. Preserve borrowed ownership and alias semantics; public two-call buffer import already works.
5. Separately prioritize count/compiler reports (12/13; send 13.3 to the browser first) and qualified feature requests 3–5, 7–11 and 14. Proposal **1 is closed** for the adopted grid formula; LUMA-103 remains a separate unverified bounds-query investigation, not a missing-export filing.
6. Port proposal 15: helper exports, typed OOM, resource planning and atlas controls remain actionable. Split geometry/trips from math.gl dimensions. Send 15.6/15.8 to deck.gl with patch-preimage/render evidence; freeze viewport spawning pending measurements.

Draft numbers and the alpha.16 target remain unchanged. Nothing has been filed.

### 15. Kepler port items (rechecked on luma.gl alpha.16; deck.gl 10.0.0-alpha.1)

Found while porting `examples/deck/kepler-compute` onto published packages (KE-PORT in [roadmap.md](./roadmap.md)). Nothing is filed. Paths: luma.gl under `node_modules/@luma.gl/<package>/dist`, deck.gl under `node_modules/@deck.gl/core`. Local code that each item would retire is listed in [upstream-workarounds.md](./upstream-workarounds.md).

The fork API-availability comparison used an **alpha.13 baseline**; none of its recorded timings compared against alpha.13. The alpha.16 source/declaration checks here
are separate evidence; the old comparison cannot establish that nothing landed or that a fork
implementation is faster on alpha.16.

**15.1 Export three helpers that remain unexported in alpha.16.** Evidence: installed public declarations; the original report also records a failed import.

- `walkTextGlyphs`, with `TextGlyphWalkOptions`, `TextGlyphVisit` and `TextRowMetrics`, is declared in `engine/dist/text/fonts/atlas/text-glyph-walker.d.ts` and documented as the canonical CPU atlas layout walk. `engine/dist/text/index.d.ts`, `experimental.d.ts` and `text-2d/index.d.ts` do not export it.
- `alignGraphVectorViews` and `getGraphVectorData` are declared in `graph/dist/gpu-core/graph-vector-view-utils.d.ts`; `graph/dist/gpu-core/index.d.ts` does not export that module.
- Reproduction (`reading`: the entry-point declarations were read, the import was not compiled separately): `import {walkTextGlyphs} from '@luma.gl/engine/text'` has no exported member to resolve. Likewise `import {alignGraphVectorViews} from '@luma.gl/graph/gpu-core'`. An `_internal` or deep `dist` import is forbidden here ([upstream-first-plan.md](./upstream-first-plan.md)).
- Request: export the three names from `@luma.gl/engine/text` and `@luma.gl/graph/gpu-core` with their types. Retires `walkKeplerLabelGlyphs` and `alignKeplerGraphVectorViews`; the local copies match the declared behaviour and are used by the label and resident-chart paths.

**15.2 A typed out-of-memory error.** Published `@luma.gl/core` exports no `OutOfMemoryError`, and an asynchronous WebGPU out-of-memory is not thrown. Reproduction (`reading`, plus the example's scope helper): allocate buffers until the device reports out-of-memory; the error arrives through the uncaptured-error path as a plain `Error`, and `Device.pushErrorScope` does nothing outside debug mode. Request: the fork's typed error and scope balancing (`modules/core/src/api/utils/out-of-memory-error.ts` at `e0ed33791`; section 5). Kepler works around it with `KeplerDeviceOutOfMemoryError` and a raw-handle scope in `binding5-device-oom-scope.ts`.

**15.3 Resource planning before compile.** `GPUCommandGraph.planResources()` and `compilationStats` are absent, so Kepler admits after compile (`MeteredCommandGraph`). Request and fork reference are in section 5 ("Resource planning without allocation").

**15.4 Text atlas options the port needed.** Evidence: `spec` for mipmaps (`modules/kepler/test/gpu-layers/gpu-point-label-layer.spec.ts`), `reading` for the cache.

- `GPUTextResourcesProps` has `fontAtlas` and `id` only; the atlas texture is a one-level `2d-array` with `{minFilter: 'linear', magFilter: 'linear'}` and no `mipmapFilter` (`engine/dist/text/text-2d/gpu-text-resources.js`). The fork accepted `mipmaps` and set `mipmapFilter: 'linear'` with `mipLevels: 'auto'`.
- Reproduction: build a bitmap atlas at the default 64 px, draw labels at 15 to 24 px with `GPUTextResources`, and compare with the deck.gl 10 `TextLayer` on the same text and font. The label paints one pixel narrower (about 290 differing pixels of about 360 painted, maximum channel error 167). Swapping in a `DynamicTexture` with `mipmaps: true`, `mipLevels: 'auto'` and `mipmapFilter: 'linear'` brings the same comparison under the original tolerance (12 differing pixels, error at most 12), in cartesian and geographic views, for three anchor and alignment settings.
- `BrowserFontAtlasSettings` has no `cache`. `buildBitmapFontAtlas` always caches per font settings (three entries) and extends the shared canvas and mapping in place; the fork's `cache: false` returned a private atlas for explicit ownership (`modules/engine/src/text/fonts/build-bitmap-font-atlas.ts`).
- Request: `mipmaps` on `GPUTextResourcesProps` (and the mip filter on the sampler when set), and `cache?: boolean` on the bitmap atlas settings.

**15.5 Geometry, polygon-contiguity and trip preparation.** The fork's engine arrow module (`modules/engine/src/arrow/geometry/` at `e0ed33791`) prepared native GeoArrow, WKB and JSON geometry for GPU rendering, globe cutting, timed trips and fp64 residuals. The port re-homed it as 22 Kepler-prefixed symbols in `examples/deck/kepler-compute/src/geometry-preparation/` on public APIs. Request: a public geometry-preparation module in engine or graph/arrow. One finding for `@math.gl/arrow` (evidence: `spec`, `modules/kepler/test/resident-select-geometry.node.spec.ts`; the failing route was probed in the geometry stream): a `FixedSizeList(3, Field('value', Float64))` point vector throws `Ambiguous three-component coordinates` on the default, generic and explicit-dimension routes; unnamed three-wide lists should default to `xyz` or accept a dimension. The local helper `nameKeplerCoordinateDimension` renames the child field and shares the buffers.

**15.6 Three hunks against deck.gl 10.0.0-alpha.1.** Carried as a yarn patch of `@deck.gl/core` (`.yarn/patches/@deck.gl-core-npm-10.0.0-alpha.1-*.patch`; the hunks are in `src/lib/deck-renderer.ts`, `src/lib/deck-picker.ts` and `src/lib/deck.ts`). Evidence: `reading` of deck.gl source and confirmation that the patch is installed; none was exercised in a browser during the port. Reproductions below are the sequences the code paths imply.

1. Depth attachment on the effect input. The patch preimage creates its two render buffers with `colorAttachments` only (`deck-renderer.ts`, around `deck-renderbuffer-${i}`); this checkout has the patch installed already. Use a color-preserving post-processing effect (`PostProcessEffect`) with a `postRender` hook in `effects`: `DeckRenderer` selects `renderBuffers[0]` only for such a hook. Render two overlapping opaque, differently colored layers with depth testing, drawing the nearer opaque layer first and the farther layer second. Check an interior overlap pixel away from antialiasing edges on both the patch preimage and patched build: expect the farther layer's color on the preimage without depth, and the nearer layer's color with the patch. A lighting-only effect (`LightingEffect`, no `postRender`) is a negative control: it bypasses that framebuffer and must keep the nearer color on both builds. Fix: `depthStencilAttachment: 'depth24plus'` on the first buffer; the second (swap) buffer only receives fullscreen effects. These pixel checks are queued, not measured here.
2. Hover clear on canvas leave. `DeckPicker.pickObject` and `pickObjects` return `{result: [], emptyInfo}` early when there are no viewports, without `processPickInfo`, so `lastPickedInfo` keeps the previous object. Reproduction: hover a pickable layer with `onHover` and a highlight, then move the pointer off the canvas; `onHover` is not called with a cleared info and the highlight stays. Fix: run `processPickInfo` with `pickedColor: null, pickedObjectIndex: -1` in that branch and return the infos that carry a layer.
3. `isHovering` from `info.picked`. `Deck._onPointerMove` sets `cursorState.isHovering = result.length > 0` (`deck.ts`). A result array can hold infos with `picked: false` (a cleared layer), so the cursor keeps its picking style over empty space. Fix: `result.some(info => info.picked)`.

Request: three small pull requests, or one, with these reproductions turned into tests in deck.gl's picking and effects suites.

**15.7 Flow particle spawn region — frozen pending measurement.** `FlowParticleSimulation` in alpha.16 (`graph/dist/experimental/simulation/flow-particle-simulation.d.ts`) has `setField`, `reset` and `step` only. The fork's `setSpawnRegion(region)` (`modules/graph/src/experimental/simulation/flow-particle-simulation.ts`, `flow-particle-shaders.ts` at `e0ed33791`) added a `spawn` rectangle uniform in normalized field coordinates: particles spawn inside it and respawn when they leave it. Candidate reproduction: zoom a Flow Field layer to a tenth of the field; a uniform distribution would put roughly a hundredth of particles on screen (area argument, unmeasured). Measure visible particles, stepping cost and behavior with changing viewport/field before proposing a spawn-region API or crediting performance. The alpha.13 fork comparison alone does not establish alpha.16 benefit.

**15.8 deck.gl 10 WGSL `PathLayer` on a globe, non-billboard paths.** In `core/src/shaderlib/path/path.wgsl.ts` (lines 149 to 152) the join code writes `deltaA3 = rotationResult.transform * deltaA3` and the same for `deltaB3`. The GLSL path layer (`layers/dist/path-layer/path-layer-vertex.glsl.js`, `deltaA3 = deltaA3 * rotationMatrix`) multiplies from the left, which is the product with the transposed matrix. WGSL `M * v` is the untransposed product, so the tangent-frame rotation of the segment deltas is inverted. Reproduction (measured in the port's headless run by the gpu-layers stream; not committed as a spec): render the same path on a `GlobeView` with `billboard: false` through deck's native WGSL `PathLayer` and through the resident path layer, which applies the transpose; the native layer paints 38 pixels where the resident one paints 254. With `billboard: true` the two agree, because the rotation branch is skipped. Request: use `transpose(rotationResult.transform) * delta` (or multiply from the left) in `path.wgsl.ts`, with a `GlobeView` render test for non-billboard paths.

## 3. GPU-resident iteration (PLAT-04)

Public compute nodes already accept `condition: {source: 'gpu', mode: 'indirect', buffer, byteOffset}` (`GPUCommandGraphGPUIndirectCondition` in published `gpu-core/gpu-command-graph-types.d.ts`). Local `createWGSLActiveCountDispatch` and `createWGSLDispatchGate` in `analysis-core/kernels/wgsl-kernel-nodes.ts` use it. The missing request is a GPU-gated continuation that does not unroll the entire worst-case iteration budget, plus one convergence readback per batch of rounds. Published primitives also need public dispatch metadata or `gateGPUCommandNodes` so a caller can gate their exact dispatch geometry.

**Prepared-build reproduction and acceptance (GEO-115).** `GPUSpatialJoinPrepared` and
`GPUStudyWindowPreparation` retain a geometry snapshot and a requested/applied generation pair.
Only the geometry refresh is gated today: rings, bounds, nested BVH/sort and slab nodes replay on
every encoding, even for a clean applied generation. The request is public gating of the complete
nested build, preserving each primitive's actual dispatch shape and all resource/dependency edges.
`prepared-geometry-snapshot.ts` commits the applied generation on the GPU only after every refresh
and build node; an encode-time latch cannot replace that complete-build dependency.

Use the one-feature square `[(0,0),(1,0),(1,1),(0,1)]`, then translate it by +10 in x and call
`invalidate()`. After the submitted rebuild, the join must return pair `(0,0)` for `(10.5,0.5)`
and none for `(0.5,0.5)`; the study window must retain area one and move its bounds to
`[10,0,11,1]`. Submit once, then repeat a clean encoding: build dispatches should be zero while
consumer results agree. Encode an invalidated build and discard it, then retry; also discard a
clean prefix and submit a prefix that stops before commit. None may mark the generation applied
before the full build executes. Run both prepared consumers, initial/invalidated paths and
`rebuildWhen` paths. Existing `modules/analysis/test/analysis-core/prepared-rebuild-discard.spec.ts`
is the downstream discard/retry fixture, not evidence that upstream nested gating exists. Also
define queued/out-of-order encoding ownership so one encoding cannot apply another's generation.
No device execution, upstream implementation or build-skipping speedup is claimed here.

`GPULoopOperation` unrolls `maximumIterations` copies. `GPUNetworkCentrality` in `gpu-network/network-centrality/gpu-network-centrality.ts` unrolls reachability/sigma/delta rounds, `maximumIterations` 1–1,024, default 64, and rejects over 32,768 command nodes (NW-CENT). Exact betweenness around 15,000–20,000 nodes needs sampled pivots under this budget. A continuation would replace this expansion and allow reconsidering the guard; local iteration gates remain useful until then.

WGLog (arXiv 2607.17571, 2026-07-20), described in [research-gpu-landscape.md](./research/spatiotemporal-review-2026-11-09/research-gpu-landscape.md), reports approximately 2.5 ms for a 4-byte mapAsync readback (285 times a CUDA 4-byte copy) and 0.18 ms for an empty fence. Its GPU writes indirect arguments, dispatches zero after convergence, batches K = 30 rounds (adaptively halved), double buffers state and reads one fixpoint word per batch. Indirect dispatch alone gives 4.7 times, batching 15.4 times over readback-per-round on recursive joins; it runs in Chrome on M5 Pro. These are the paper's numbers, not measurements of this library.

Proposed `GPUIterationBatch` (name open): record caller-controlled K copies of a subgraph, with a device-side active flag, a subgraph-written convergence word and one readback per batch; the host chooses the next K. Specify zero-workgroup behavior, double buffering to avoid read/write aliasing within a pass, and known-iteration/early-convergence tests.

## 4. Adapter limits, subgroups and buffer_view (PLAT-05)

`createDevice({featureLevel: 'max'})` already requests adapter limits and every supported feature: see `getRequiredWebGPUFeatures` and `WebGPUAdapter.create` in `node_modules/@luma.gl/core/dist/webgpu/adapter/webgpu-adapter.js`. The request is adapter limits without enabling all features, for example a limits-only policy or selected adapter-limit names; explicit numeric `requiredLimits` already works. Spec defaults are 8 storage buffers per stage, 256 MiB buffers and 128 MiB bindings. The recorded headless Apple device reports binding/buffer limits 4,294,967,292, 10 storage buffers and 1,024 invocations. `getAnalysisDeviceLimits` in `modules/lenses/src/computation-views/hosts/device-profile.ts` already requests up to 16 storage buffers and adapter buffer/binding maxima. PLAT-02 storage-binding guards remain a separate repository item.

The cited web3dsurvey reports (collection date unspecified) show 98% with ≥10 storage buffers, 80% with ≥16, 77% with ≥2 GiB `maxBufferSize` and 93% with ≥512 workgroup invocations; the cited browser coverage gives Chrome 146 up to 16 storage buffers and subgroups from Chrome 134, with none found through Safari 27.2/Firefox 162. Proposed `createDevice({requiredLimits: 'adapter-max'})` or selected limit names would request adapter values and report grants in `device.limits`.

Published `GPUReduction` already selects `subgroups` or `portable` via `getGPUReductionStrategy`, unsegmented `GPUScan` does the same when the device and WGSL language support it, and `GPUSort` uses a subgroup local bitonic stage under that condition (segmented scan stays portable); each keeps a workgroup-memory fallback. The landscape evidence reports about 68% subgroup availability, 11–28 times slower emulation on M4, and radix-sort rates near 1 G keys/s portable on M1 Max versus 3 G/s with real Metal subgroups; these are external reports, not local measurements. `bufferView<T>()`, `bufferArrayView<T>()` and `bufferLength()` pack arrays into fewer bindings and could relieve geographically weighted regression binding pressure, but the landscape places them in Chrome 153–154 only. Wait for a second browser before API work and preserve a portable fallback.

## 5. Implementations that already exist in the fork

The fork is luma.gl commit `e0ed33791` (branch `backup/map-contributors-pre-split-2026-11-07`, local only). Its code is public at the same paths on [`BertCh/luma.gl`, branch `reference/map-contributors-pre-split-code`](https://github.com/BertCh/luma.gl/tree/reference/map-contributors-pre-split-code) (commit `244ddc5ba`, datasets and fixtures left out). [research/fork-comparison-2026-11-13](./research/fork-comparison-2026-11-13/README.md) has the full table with fork paths, tests, the **alpha.13 baseline for API availability** and the fork's own recorded measurements, none compared against alpha.13. The fork code was read for that comparison, not run; it is not an alpha.16 absence or performance audit. Paths below are under the fork's `modules/graph/src` unless they name another module.

| Item above | Fork code | Note |
| --- | --- | --- |
| Kernel-node factory (top five 1) | `gpu-core/gpu-kernel-node.ts`: `createGPUKernelNode`; `utils/gpu-spatial-operation.ts` | Every fork primitive was written with it |
| Algorithms on a caller CSR (top five 2, 2.2) | `gpu-graph-algorithms/gpu-graph-topology-view.ts` and view-accepting degree, components, PageRank, core number, label propagation | No node or slot masks yet |
| Deterministic sums (top five 4) | 96-bit fixed-point sums in `gpu-vector-search/gpu-k-means.ts`, `gpu-raster/gpu-raster-cross-tile-components.ts`, `gpu-raster-global-statistics.ts` | Chunk-order reproducibility is tested for k-means |
| GPU-gated loops (top five 5, section 3) | `gpu-graph-algorithms/gpu-graph-iteration-gate.ts`, compact frontiers in BFS and SSSP, outputs invalidated on non-convergence | The gates still unroll; a continuation is new work |
| One grid cell formula, GPU bounds (2.1) | `gpu-core/gpu-spatial-utils.ts`, `gpu-grid-index.ts` (`boundsBuffer`) | |
| Haversine units and seam (2.3) | `geospatial/gpu-haversine-distance.ts` (`units`, IUGG radius, metres by default) | Checked at `e0ed33791`: float32 still subtracts raw longitudes; no wrapping option or accurate seam subtraction. Keep the 179.94722 → -179.99805 acceptance pair. |
| Histogram edges (2.6) | `gpu-core/gpu-histogram.ts` edge correction with a parity spec | Not checked against the cases of 2.6 |
| Exact point-in-ring (2.10) | `geospatial/gpu-pairwise-point-in-polygon.ts` (exact float32 containment) | |
| Smaller gaps (2.14) | `compileProjectionPlan` at an explicit origin (`gpu-project/projection-plan.ts`); NaN for invalid DGGS cells (`gpu-dggs/gpu-dggs-cell-projection.ts`); voxel centroids and attribute means (`gpu-pointcloud/gpu-pointcloud-voxel-downsample.ts`); raster local origin (`gpu-raster/raster-utils.ts`) | |
| Exports and sort or scan speed | Onesweep sort, look-back scan, block compaction, cooperative segmented scan (`gpu-core/`) | The fork's own 8M-row Apple benchmark records compaction 1.4 ms against 3.6 ms; neither timing is an alpha.13 comparison, and no alpha.16 benefit is measured |
| External crossfilter masks | `gpu-crossfilter/gpu-crossfilter.ts`, `gpu-selection.ts`, `histogram-cube.ts` | |
| Dataframe statistics and description | `gpu-dataframe/gpu-query-compiler.ts` (`compilationStats`), `gpu-query-describe.ts` | |
| Exact Int64 time | `arrow/vectors/arrow-temporal-gpu-vector.ts`: `makeArrowTemporalWordGPUVector` | |
| Typed out-of-memory errors | `modules/core/src/api/utils/out-of-memory-error.ts`, scope balancing in the WebGPU device | |
| Per-node recording | `gpu-core/gpu-command-graph-describe.ts`, `gpu-command-node-record.ts`, the `recording` option | |

The comparison also found fork changes that are not in the summary above and are worth offering:

- **Resource planning without allocation.** `GPUCommandGraph.planResources()` returns a frozen plan equal to the compiled allocation, without allocating or compiling, and picks the smallest fitting allocation for reuse. An application can refuse a query before it allocates. Kepler measures after compiling today.
- **Readback and transfer.** Failed-map replacement and cancellation in `GPUReadbackRing`; a staging-buffer pool and an upload ring in core, so `readAsync` need not create one staging buffer per call.
- **Bind-group cache in `Kernel`**, keyed by buffer, offset and size and excluding destroyed resources, with a spec.
- **Strongly connected components** with O(V) scratch, checked against Tarjan to 5,000 vertices.
- **Core fixes:** std140 matrix alignment in the shader block layout, 16-byte rounding in place of 1 KiB uniform allocations, `destroyTestDevices`, borrowed-buffer and resize races in the dynamic resources, leaks in `Computation`.
- **Query language:** `choose` and `coalesce`, nullable `COUNT(column)`, quoted identifiers, `USING`, join-error readback.
- **Candidate-driven queries:** `GPUBVHQuery` on indirect frontiers and a `GPUGridIndexQuery` that gathers only the selected slots; the published versions scan every node or slot.
- **Arrow converters** on bounded two-dimensional dispatch; the published converters dispatch in one dimension.

## 6. Additional upstream correctness and capture proposals (third wave)

These are local proposals, not filed changes or claims of published fixes. Installed-source
inspection concerns `@luma.gl/graph@10.0.0-alpha.16`.

### 16. Origin-preserving integer raster calibration (TWN-08)

`gpu-raster/gpu-raster-neighborhood.ts` converts `rawSample` to `f32` before applying scale and
offset. For `sint32` cells `[16777216,16777217]`, scale `1`, offset `-16777216`, the mathematical
heights are `[0,1]`; cast-first arithmetic gives `[0,0]`. The reported CPU prominence result is
`parentPeak=[1,NONE]`. Plateau tie handling predicts `[NONE,0]` on GPU, but that prediction
still requires device readback.

Propose a public `createOriginPreservingIntegerRasterCalibration` helper and an additive
`integerCalibration` declaration with integer `origin`, declared raw integer range, `scale`
and calibrated `offset`. Define calibrated samples as `(raw - origin) * scale + offset`, with
integer subtraction before conversion; check subtraction range before emitting a signed or
unsigned operation. Host code must validate that the declared shifted range is exactly
representable in float32 (conservatively within ±16,777,216), and that coefficients and the
calibrated range fit the supported finite route. Those checks alone are insufficient: the
calibration must also preserve cancellation, independent of multiply-add fusion. For the direct
multiply-then-add route, require a construction-time proof that the products are exact across
the declared integer range and that addition gives the correctly rounded calibrated values.
An exact-integer certificate with products and sums within ±16,777,216 is one conservative
sufficient route; exact representability of the inputs alone is not a certificate. Alternatively,
use a separately qualified precision-preserving route. Reject uncertifiable declarations rather
than silently converting or rounding large intermediates, and name a safe origin in the error
when one is established from the declared range. Do not promise exact arbitrary real-valued products.

Retain legacy `scale`/`offset` for safe declarations and existing float32 bands. Mark their integer
calibration spelling `@deprecated` in favor of `integerCalibration` in upstream doc comments.
If both forms are supplied, resolve legacy offset to `origin * scale + offset` and reject
disagreement. Reject ambiguous, conflicting or precision-losing integer declarations at
construction with an instance-prefixed message naming `integerCalibration` and the declared
range/limit; request the origin and raw range when safety cannot be established. Never infer
the range by reading GPU data. Removing the old spelling remains a maintainer decision.

Acceptance must upload the original `Int32Array` samples as a `sint32` band and apply the
qualified public helper on the native integer-conversion route. Uploading pre-rebased floats
bypasses the defect and qualifies only the application workaround. Require both `2×1` controls:

- Positive cells `[16777216,16777217]`, origin `16777216`, scale `1`, calibrated offset `0`
  (legacy offset `-16777216`): samples `[0,1]`, validity `[1,1]`,
  `parentPeak=[1,0xffffffff]` (`NONE`).
- Negative signed cells `[-33554433,-33554432]`, origin `-33554432`, scale `2`, calibrated
  offset `0` (legacy offset `67108864`): samples `[-2,0]`, validity `[1,1]`,
  `parentPeak=[1,0xffffffff]`. Exact integer subtraction yields `[-1,0]` before conversion;
  cast-first arithmetic collapses these distinct cells to `[0,0]`.

Also cover safe legacy declarations, equal dual declarations, conflicting dual declarations,
missing range and an unsafe shifted range. Qualify every raster conversion route before adoption.
Cancellation control: shifted integers `[16777210,16777211]`, scale `3`, calibrated offset
`-50331632` have exact products `[50331630,50331633]` and heights `[-2,1]`. Separate float32
product rounding gives `[0,0]` even though every input operand is representable and the final
range is finite. Reject that direct-route declaration at construction, naming an additional
origin `16777210`: shifted integers become `[0,1]`, adjusted calibrated offset becomes `-2`.
The safe declaration must read back samples `[-2,1]` and `parentPeak=[1,NONE]` on a `2×1`
grid; a separately qualified exact route accepting the original declaration must do likewise.
Do not assume fused multiply-add or compiler reassociation will repair it.

Native three-cell nodata acceptance: upload the original
`Int32Array.of(20000001,20000000,20000002)` as `sint32`, with origin and raw sentinel `20000000`,
scale `1` and calibrated offset `0`; omit the host mask. Compare the raw sentinel against the
original integer cells before origin subtraction or calibration. Require validity `[1,0,1]`,
valid endpoint samples `1` and `2`, and `parentPeak=[NONE,NONE,NONE]`: the invalid middle cell
separates the two peaks. Comparing shifted samples `[1,0,2]` with raw sentinel `20000000`
incorrectly produces validity `[1,1,1]` and connects those peaks. This native control must pass
through the qualified public helper and integer-conversion route without host rebasing or a
host mask that could hide an incorrect sentinel comparison.

Until published support, subtract the origin in exact host integer arithmetic before upload:
`rawPrime = raw - origin`, `offsetPrime = oldOffset + origin * scale`. Check integer subtraction
and shifted format bounds; do not subtract after a Float32Array conversion. Preserve raw nodata:
also set `noDataValuePrime = noDataValue - origin`, checked for representability in the shifted
format, or compute validity from the original raw values, upload that mask and remove the stale
sentinel. Apply the same rule to proposed origin-aware conversions before testing calibrated values.
Three-cell workaround coverage only: raw `[20000001,20000000,20000002]`, origin and raw nodata `20000000`,
scale `1`, old offset `-20000000`. Upload `[1,0,2]`, adjusted offset `0`, nodata `0` (or mask
`[1,0,1]` with no sentinel). The middle cell must remain invalid and separate two peaks:
`parentPeak=[NONE,NONE,NONE]`. Keeping stale nodata `20000000` incorrectly makes zero valid
and predicts `parentPeak=[2,0,NONE]`. Qualify sample validity as well as parent IDs.

### 17. Own-property Arrow analytics metadata lookup (EGR-08)

The dictionary lookups in `arrow/gpu/arrow-gpu-analytics-readback.ts` at field-type selection
and dictionary materialization, and validity lookups in validation and batch materialization,
all read inherited properties. A legal explicitly non-null Int32 field `constructor=[42]`
uploads but ordinary-map table readback throws `Cannot read properties of undefined (reading
'length')`. The table-only call succeeds when both optional maps are absent; null-prototype
copies of actual metadata entries also succeed. The initial scratch fixture used
`tableFromArrays`, which declares a nullable field even for `[42]`; its different error
`source is not iterable` was not the report's non-nullable fixture. Ordinary nullable
`score=[5,null,7]` round-trips with both ordinary and null-prototype maps.

The upload side also needs correction: `arrow/gpu/arrow-gpu-analytics-adapters.ts` builds
ordinary validity, dictionary, null-count and per-batch GPU-data maps by field name. For the
legal nullable Int32 field `__proto__=[18,null,5]`, assignment changes the map's prototype
instead of creating an own entry. Copying validity and dictionary maps after upload cannot
recover that lost validity: readback still throws `Arrow analytics nullable column "__proto__"
requires GPU validity`. Use null-prototype maps or explicit own-property creation for these
upload maps and all derived field-name maps; preserve entries through table/vector derivation.

Retain own-property guards at **all four** readback sites: read a field only when
`map != null && Object.hasOwn(map, field.name)` is true. Guard optional map presence at each
site before the own-property test; absent maps must remain supported. Preserve actual dictionary
metadata and validity vectors, including supplied labels/order and batch alignment. Do not
drop the maps: nullable and dictionary columns need them.

Upstream NullDevice acceptance must explicitly construct `Field('constructor', new Int32(), false)`
with `[42]` and assert source and uploaded field nullability are false. Assert the
table-only, ordinary-map and null-prototype-map cases separately: all must return `[42]`
after the fix; only ordinary-map readback fails on the installed baseline. Retain nullable Int32
`score=[5,null,7]`, and a dictionary column with explicitly supplied dictionary metadata;
assert decoded labels, null positions and dictionary order survive. Include inherited fake
entries alongside real own entries to verify only the latter are consumed. Add nullable
`__proto__=[18,null,5]` uploaded under its original field name: assert own validity, null-count
and per-batch GPU-data entries (and own dictionary entries for a dictionary variant), then
require full readback `[18,null,5]` and compact selection `[null,5]` for live source rows
`[1,2]`, count `2`. Check the derived field-name maps as well as the uploader. These are
NullDevice controls; no GPU execution is needed.

Application workaround: use the [reversible pre-upload alias](../docs/api/README.md#fields-named-proto)
for `__proto__`, choosing an unused safe name such as `__analysis_proto__`; retain the inverse
map and restore the original name after materialization. Preserve field/schema metadata,
nullability, batch topology and source-aligned validity, and use the alias in GPU references.
Also copy actual own validity and dictionary entries into separate `Object.create(null)` maps
before calling `makeArrowTableFromGPUAnalyticsTable`, preserving the other resident-table
properties. Those post-upload copies alone remain sufficient for the `constructor` control,
but cannot repair entries already lost by the uploader.

### 18. Value-based nested CRS comparison (URT-07)

`GPURaster.isCompatibleWith` delegates CRS equality to `raster-utils.ts`, where declared
`authority` and `wellKnownText` compare by value but `projectionJson` uses `JSON.stringify`.
Top-level CRS key reordering already passes; nested JSON key order still rejects equivalent
grids. Propose recursive comparison of JSON object own members independent of insertion order,
preserving array order and scalar values. Preserve affine, dimensions, pixel interpretation,
level and level-zero-origin checks.

Also expose a public CRS-only comparison of declared `authority`, `wellKnownText` and nested
`projectionJson` values for reprojection constructors and the metadata packer. Those consumers
allow different grids in the same CRS: full `GPURaster.isCompatibleWith` is unsuitable because
its required dimension/affine checks reject legitimate warps. Retain an adoption control that
warps from width 3 to width 2 with a quarter-cell translation in the same CRS. The CRS-only
comparison must accept it, and the affine packer must still return 20 finite parameters;
the full-grid comparator must continue to reject those grids as incompatible.

Acceptance uses otherwise identical grids with authority `EPSG:4326`, the same well-known
text, and `projectionJson` values
`{projection:{name:'longlat',parameters:{datum:'WGS84',units:'degree'}},axes:['x','y']}` and
`{axes:['x','y'],projection:{parameters:{units:'degree',datum:'WGS84'},name:'longlat'}}`.
They must compare equal. Changing `datum` to another value must compare unequal, as must
reversing `axes` to `['y','x']`; changing affine or any other existing grid check must still
compare unequal.

Local reprojection validation has landed: `validateRasterReprojectionCRS` in
`modules/analysis/src/raster-analysis/raster-band-utils.ts` compares nested JSON object members
by value while retaining scalar values and array order. The direct `GPURasterReprojection`
and tiled `GPURasterTileReprojection` constructors use it through
`getRasterReprojectionShader` before graph allocation, and the public affine packer
`getGPURasterReprojectionParameterValues` calls it directly. An explicit inverse projection
permits distinct source/destination CRS. This analysis-domain validation does not correct the
public `GPURaster.isCompatibleWith` comparator. Retire the local comparison only after a
public CRS-only comparison usable by both constructors and the metadata packer passes
reordered nested-object equality, changed-scalar inequality, reordered-array inequality and
the different-grid, same-CRS warp control. Correcting the full-grid comparator alone is
insufficient; preserve all of its other grid checks listed above.

### 19. Public resolved graph build identity for captures (SEAM-113)

Propose a concrete `GPU_GRAPH_BUILD_REVISION` export on the public graph entry used by captures,
generated by that graph package's build from its version or immutable revision. It must identify
the loaded artifact, including distinct revisions when version alone is reused. The installed
package version exists in package.json, but package.json is outside the public export map.
Do not import that private path or substitute core's revision.

Both `createGPUComputationRecord` and `getGPUComputationRunId` currently default to the fixed
integration target `@luma.gl/graph@10.0.0-alpha.16`; an independently resolved graph peer can
change without changing that identity. Acceptance resolves two different graph revisions with
otherwise identical capture inputs and no caller override: each default record identifies the
actual loaded graph revision and the run IDs differ. Wire the export into both default paths
when published; retain explicit caller override precedence.

Available workaround: the capturing application resolves its actual graph artifact at build
time and passes the existing optional `dependencyRevision` to both capture and run-ID helpers.
An analysis-package build-time constant cannot identify a later independently resolved peer.
Requiring an override on every existing capture needs a maintainer decision; this proposal
does not impose that requirement.
