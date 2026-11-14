# Fork comparison: what the luma.gl fork changed in luma.gl itself

Date: 2026-11-13. Read-only. Nothing in either repository was changed and nothing was run.

Before 2026-11-07 the contributors lived inside a fork of luma.gl, in `modules/graph/src` beside
luma.gl's own code, and the fork changed luma.gl-owned code to suit them. The
[split review](../split-review-2026-11-08/README.md) compared the contributor files of the fork with
this repository. This comparison covers the other half: the changes the fork made to luma.gl-owned
code, whether published luma.gl has them, what this repository does without them, and where each is
tracked.

## Scope and method

- **Fork:** luma.gl commit `e0ed33791` (branch `backup/map-contributors-pre-split-2026-11-07`), 72
  commits ahead of upstream luma.gl at `704183953` (2026-11-06). Local clone:
  `~/Documents/GitHub/vis.gl-build/luma.gl`. Read with `git diff 704183953 e0ed33791 -- <path>` and
  `git show e0ed33791:<path>`. That commit is local only.
- **Public copy of the code:** branch
  [`reference/map-contributors-pre-split-code`](https://github.com/BertCh/luma.gl/tree/reference/map-contributors-pre-split-code)
  of `BertCh/luma.gl`, commit `244ddc5ba`. It is one commit on top of `704183953` with the fork's
  `modules/`, `docs/` and `dev-docs/` code at the same paths, so every source, test and document
  path below opens there. It leaves out datasets and what derives from them (examples, lens
  records, lens data, JSON fixtures), `dev-docs/research` and the website; a spec that reads a
  fixture will not run from it. It is a reference, not a branch to build or merge.
- **Published:** `@luma.gl/{core,shaders,graph,engine}@10.0.0-alpha.13` as installed. Alpha.14 does
  not touch any file named here.
- **Fork-carried change:** a file the fork added or modified since `704183953` that is not a
  contributor file. A contributor file is one under the fork's analysis-owned directories or one
  whose basename exists under `modules/*/src` here.
- **By script:** 275 luma.gl-owned files changed in `modules/graph/src` (about 24,000 added lines),
  31 in `modules/core/src` (815), 51 in `modules/engine/src` outside splats (about 5,000) and 51 in
  the two deck layer modules (about 10,300). Those files add 304 exported names (194 graph, 101
  engine, 9 core). None of the 304 is declared anywhere in the alpha.13 `dist`.
- **By reading:** six reviewers each read the fork diff, the alpha.13 `dist` and this repository for
  one area. They ran nothing. Fork tests named below were read, not run. Sizes are approximate.
- **Not covered:** the splat, ANARI and three.js changes on the same branch. No contributor file
  imports them.

Fork paths below are relative to the fork root. `G` is `modules/graph/src`, `GT` is
`modules/graph/test`. "Register" is [upstream-workarounds.md](../../upstream-workarounds.md);
"proposals" is [upstream-proposals.md](../../upstream-proposals.md).

## What this means

1. **Most of what the proposals ask luma.gl for already exists as code with tests.** Each row below
   that says "absent" is a candidate pull request whose first draft is in the fork. That changes
   the cost of the feedback: the work is porting and review, not design.
2. **Nothing the fork carried has landed upstream.** The published classes are the fork's base
   versions. Where the register says "accepted regression", the fork holds the faster or more exact
   version.
3. **Several fork changes turned out not to be needed.** This repository runs on the published API
   for completion waiting, bounded dispatch above 4.2 million rows, device limits and benchmark
   runners (last table). That is evidence the public API is adequate there.
4. **Some fork capabilities could return here as analysis code** without waiting for luma.gl,
   because they are domain algorithms and not luma.gl APIs: the O(V)-scratch strongly connected
   components and the voxel centroid producer (roadmap NW-SCC and GAP-05).

## Command graph, kernels and primitives

| Capability | Fork code and tests | Alpha.13 | Here today | Tracked |
| --- | --- | --- | --- | --- |
| Kernel-node factory | `G/gpu-core/gpu-kernel-node.ts` (`createGPUKernelNode`, binding and override declarations), `G/utils/gpu-spatial-operation.ts`; every fork primitive used it. `GT/gpu-core/gpu-kernel-reuse.node.spec.ts`, `gpu-dispatch-utils.node.spec.ts` | Factory absent; bounded layout public; `getSpatialOperationNodes` in dist, not exported | `createWGSLKernelNode` (`analysis-core/kernels/wgsl-kernel-nodes.ts`) | Proposals summary, top five 1 |
| Sort and scan strategies | `G/gpu-core/gpu-sort-onesweep.ts` (532 lines), `gpu-scan-lookback.ts` (497), `gpu-sort-chunks.ts` (balanced merge tree in place of ranking all run pairs). `GT/gpu-core/gpu-sort-onesweep.spec.ts`, `gpu-scan-lookback.spec.ts`, `gpu-sort-chunks.spec.ts` pin the recount fallback, check-word collisions, repeated encoding and stable ties | Absent; `GPUSort` and `GPUScan` have no algorithm option | Published defaults | Register, "Specialized sort/scan and benchmark harness" |
| Block compaction | `G/gpu-core/gpu-compaction.ts`, `gpu-flag-offsets.ts`: 1,024-row blocks ranked with subgroup prefix sums and a workgroup-scan fallback, count folded into the scan. `GT/gpu-core/gpu-scan-compaction.spec.ts` | Absent | Published `GPUCompaction` | Register, "Block compaction" |
| Reduction, segmented operations, scatter | `G/gpu-core/gpu-reduction.ts` (`compensated` option), `gpu-double-single.ts`, `gpu-segmented-reduction.ts`, `gpu-segmented-scan.ts` (cooperative), `gpu-scatter.ts`. The compensation spec allows equality with the plain float32 sum, so it does not prove the term survives the compiler | Options absent; segmented classes and scatter in dist, not exported; published segmented scan is serial in lane 0 | `getSortedSegmentSumNodes`, fixed-point accumulator, local scatter in `GPUNetworkSubgraphFilter` | Register, "Segmented sum" and "Exact/compensated reduction and scan sums"; proposals 2.13 |
| GPU-resident grid bounds, one cell formula | `G/gpu-core/gpu-grid-index.ts` (`boundsBuffer`), `gpu-grid-index-internals.ts`, `gpu-grid-binning.ts`, `gpu-grid-aggregation.ts`, `gpu-spatial-utils.ts` (shared coordinate WGSL). `GT/gpu-core/gpu-grid-index.spec.ts` | Absent | Unit-domain normalisation pass and five pinned WGSL copies | Register, "Grid cell-coordinate WGSL" and "Grid bounds and cell order"; proposals 2.1 |
| Histogram edge correction | `G/gpu-core/gpu-histogram.ts` (+175): corrects the estimated bin by comparing against the interval edges. `GT/gpu-core/parity-wp6-histogram.parity.spec.ts`. Not checked against the edge and wide-domain cases of proposals 6 | Absent | Published class, plus local `BIN_WGSL` in the column profile | Proposals 2.6 |
| Candidate-driven spatial queries | `G/gpu-core/gpu-bvh-query.ts`, `gpu-grid-index-query.ts` (+927): indirect BVH frontiers; grid gather over selected slots only. Specs of the same names | Different design: both scan every active node or object slot | Neither query class is imported; `getGridCandidateNodes` in region statistics | Register, `GPUGridIndexQuery` row. The BVH cost was not recorded before this note |
| Recording and description | `G/gpu-core/gpu-command-graph.ts` (`recording` option), `gpu-command-graph-describe.ts`, `gpu-command-node-record.ts`, `gpu-graph-describe.ts`, annotations in every primitive and in `G/gpu-graph-algorithms/gpu-graph-computation-record.ts`. `GT/gpu-core/gpu-command-graph-recording.node.spec.ts`, `-describe`, `-timings` | Absent; public `preflight`, statistics and timings only | `computation-record/` rebuilds records from public data | [upstream-first-recording.md](../../upstream-first-recording.md) |
| Resource planning without allocation | `GPUCommandGraph.planResources()` (`planGPUCommandGraphResources` in `G/gpu-core/gpu-command-graph-compiler.ts`: smallest fitting reuse, else grow the largest allocation), `gpu-planning-ordered-set.ts`. `GT/gpu-core/gpu-command-graph-resource-plan.node.spec.ts` checks that a frozen dry-run plan equals the compiled allocation and that planning allocates and compiles nothing | Absent; the compiler picks the smallest available allocation | Kepler measures after compiling (`MeteredCommandGraph`) and checks sort reservations after allocation | Register, "`MeteredCommandGraph`"; proposals summary, dataframe lifecycle |
| Bind-group cache and readback recovery | `modules/graph/src/execution/compute/kernel.ts` (+140: cache keyed by buffer, offset and size; destroyed resources excluded), `G/gpu-core/gpu-readback-ring.ts` (failed-map replacement, cancellation of encoded reads, release after late compilation). `GT/execution/compute/kernel-bind-group-cache.node.spec.ts`, `GT/gpu-core/gpu-readback-ring.node.spec.ts` | Absent | Bindings passed on every dispatch; one-slot rings in the explorer | Register, "Bind-group reuse" and "Readback-ring constructor failure" |
| Sparse solve on skewed rows | `G/gpu-core/gpu-adaptive-spmv.ts`, `gpu-spmv-strategy.ts`, `gpu-jacobi-preconditioner.ts`, `gpu-pcg.ts` (+300: hybrid traversal, cooperative diagonal extraction). `GT/gpu-core/gpu-hybrid-spmv.spec.ts`, `gpu-jacobi-preconditioner.spec.ts` | Base classes in dist, not exported; the changes absent | Local Jacobi PCG in `landscape-ecology/sparse-resistance-solver.ts` | Register, "Sparse grounded-Laplacian solve (EC03)" |

## Graph algorithms, vector search, physics

| Capability | Fork code and tests | Alpha.13 | Here today | Tracked |
| --- | --- | --- | --- | --- |
| Algorithms on command-graph views | `G/gpu-graph-algorithms/gpu-graph-topology-view.ts` (+280) and view-accepting degree, components, PageRank, core number and label propagation. `GT/gpu-graph/gpu-graph-topology-view*.spec.ts`. The views have no node or slot mask, and degree still subtracts offsets | Absent | Local degree, weak components and modularity; PageRank, core number and label propagation not offered | Register, "Transient CSR and column views"; proposals 2.2 (masks remain new work) |
| Strongly connected components at scale | `G/gpu-graph-algorithms/gpu-graph-strongly-connected-components.ts` and `-internals.ts` (+480): trim, colour and backward mark on the forward CSR, O(V) scratch, 256 to 4,096 steps, output invalidated when the budget runs out. `GT/gpu-graph/gpu-graph-strongly-connected-components.spec.ts` compares with Tarjan at 4,096 to 5,000 vertices | Absent | Exact reachability only: 1,024 vertices, O(V²) scratch | Roadmap NW-SCC; register, "Large-graph SCC" |
| Frontiers and convergence | `G/gpu-graph-algorithms/gpu-graph-iteration-gate.ts`; compact frontiers in BFS and SSSP, zero dispatch after convergence, outputs invalidated on non-convergence, PageRank tolerance and optional renormalisation (about 1,900 lines). The gates still unroll | Changes absent; the indirect condition itself is public | `createWGSLDispatchGate`, local `GPUNetworkReachability` | Proposals section 3 (PLAT-04) |
| Clustering coefficient and Louvain on hubs | `G/gpu-graph-algorithms/gpu-graph-local-clustering-coefficient*.ts` (sorted-unique scratch), `gpu-graph-modularity-optimization-internals.ts` (+451, community hash with a binding-limit fallback). Specs cover duplicate-heavy cliques and weighted hubs | Absent; published versions scan | Not used; `GPUNetworkLouvain` is a 32-vertex reference | Register, `GPUGraphAlgorithmsModularityOptimization` row |
| Force layout stability | `gpu-graph-force-layout-internals.ts`, `gpu-graph-spatial-force-layout-internals.ts`: attraction gain capped at 2 so hubs do not oscillate | Absent | Not used | This note only |
| k-means and similarity search | `G/gpu-vector-search/gpu-k-means.ts` (+225: 96-bit fixed-point scatter, staged centroids), `gpu-similarity-search.ts` (+280: cooperative scoring, `rowSplit`). `GT/gpu-vector-search/*.spec.ts` check chunk-order reproducibility. The fork has no k-means++ or tolerance stop either | Classes public; changes absent | Published classes for embeddings; local `GPUSpatialKMeans` for points | Proposals 2.7 (seeding only; the fixed-point accumulation is new here) |
| Simple-graph batching, physics, parser | `G/gpu-simple-graph/operations/webgpu/` (one submit per evaluation scope, buffer pool capped at 256 MiB), `G/gpu-physics/` (compact warm start above 256 bodies, cell-range table), `G/gpu-parse/` (stream cleanup) | Absent | Not used | This note only |

## Raster, geospatial, projection, point cloud

| Capability | Fork code and tests | Alpha.13 | Here today | Tracked |
| --- | --- | --- | --- | --- |
| Raster kernels: finite tests and limits | `G/gpu-raster/gpu-raster-{band-math,contours,morphology,neighborhood,statistics}.ts` (about 600 lines): finite test on exponent bits in place of arithmetic, reused binding records, dispatch and capacity checks, `getGPURaster*Maximum*` limit getters | Operators public; changes absent | `IS_FINITE_VALUE_WGSL` (`analysis-core/kernels/wgsl-common.ts`) in local kernels | Register, "Raster validators and constructors" |
| Cross-tile sums and convergence | `G/gpu-raster/gpu-raster-cross-tile-components.ts`, `gpu-raster-global-statistics.ts` (+383): 96-bit fixed-point intensity and moment sums, indirect convergence gating, capacity limits. The `sumCompensation` test is a CPU float32 simulation, not GPU evidence | Classes public; float compare-exchange sums, ungated rounds | No consumer here | Proposals summary, top five 4 |
| Raster local origin | `G/gpu-raster/raster-utils.ts` (`getRasterLocalOrigin`, `validateRasterExtent`): subtract the origin before rounding to float32. `GT/gpu-raster/local-origin-precision.node.spec.ts` | Absent; band validators in dist, not exported | `readRasterParameterOrigin`, `validateRasterParameterExtent` (`raster-analysis/raster-band-utils.ts`) | Register, "Raster validators and constructors" |
| Pairwise predicates and spatial query | `G/geospatial/gpu-pairwise-point-in-polygon.ts`, `gpu-pairwise-point-linestring-nearest.ts`, `gpu-point-spatial-query.ts` and two distance classes (about 1,200 lines): exact float32 containment, optional nearest outputs, spatial-context checks, nodes returned to the caller | Different design: older props, `addToGraph`, conservative classification with `uncertain` | Published classifier plus local exact kernels; `captureGraphCommandNodes` | Proposals 2.10; register, `GPUPairwisePointInPolygon` and "`addToGraph` bridge" rows |
| Haversine units and seam | `G/geospatial/gpu-haversine-distance.ts`, `gpu-sinusoidal-projection.ts` (+121): metres on the IUGG radius by default, explicit `units`, optional longitude wrapping, the equirectangular name | Radius 6371 (kilometres), raw longitude subtraction | `GPUGeodesicPairs` | Proposals 2.3 |
| Projection fitted at an explicit origin | `G/gpu-project/projection-plan.ts` and neighbours (about 350 lines): fit at a caller origin, frame tags, strided coordinate reads (`readFloat32Coordinates`). `GT/gpu-project/projection-frame.node.spec.ts`, `projection-spatial-context.node.spec.ts` | `compileProjectionPlan` has no `destinationOrigin` | Origin replaced after compiling, in `GPULocalFrameProjection` | Register, "Projection-plan origin" |
| Invalid DGGS cell centres | `G/gpu-dggs/gpu-dggs-cell-projection.ts`: NaN when no validity output is bound, zero with one. `GT/gpu-h3/gpu-h3-cell-projection.spec.ts` | Always zero | `GPUCellGeometry` writes NaN | Proposals 2.14 |
| Voxel centroids and attribute means | `G/gpu-pointcloud/gpu-pointcloud-voxel-downsample.ts` (+91). `GT/gpu-pointcloud/gpu-pointcloud-analysis.spec.ts`, `-advanced.spec.ts` | Representative sampling only | `GPUPointCloudVoxelMerge` takes centroids nothing produces | Roadmap GAP-05; register, `GPUPointCloudVoxelDownsample` row |

## Tables, dataframe, crossfilter, Arrow

| Capability | Fork code and tests | Alpha.13 | Here today | Tracked |
| --- | --- | --- | --- | --- |
| Residency arena and tile LOD | `G/gpu-tables/residency-arena/`, `tile-lod-selection/` (about 2,400 lines): paging and live-row compaction with a fixed graph topology, budgeted LOD requests. `GT/gpu-tables/residency-arena/*.spec.ts`, `tile-lod-selection/gpu-tile-lod-selection.spec.ts` | Absent | Dropped; Kepler has its own `resident-*` code | Register, "Residency arena, resident selection and tile LOD" |
| Crossfilter cubes and external masks | `G/gpu-crossfilter/gpu-crossfilter.ts`, `gpu-selection.ts`, `histogram-cube.ts` (about 1,400 lines): dependency skipping, histogram cubes, external masks with `invalidate`, count views. `GT/gpu-crossfilter/gpu-crossfilter-skip-cube*.spec.ts`, `-external-mask.spec.ts`. Lessons recorded in the code: cube brushes snap outward to bin edges; histogram-only cube mode can leave masks stale; a skipped encoding that was never submitted needs an invalidate | Base class public; all of this absent | Masks turned into range dimensions in `GPUResidentExplorer`; the lens backend rejects key sets | Register, "Histogram-cube / skip-cube crossfilter" and "Crossfilter extensions" |
| Query description and statistics | `G/gpu-dataframe/gpu-query-describe.ts`, `gpu-query-compiler.ts` (`compilationStats`), `gpu-expression-shader.ts` (expression to WGSL to command lineage); statement descriptions in `G/gpu-sql/gpu-sql.ts` | Absent; the basic shader planner is in dist, not exported | Dropped (8 lens scenarios); `MeteredCommandGraph` recovers transient bytes | Roadmap DF-DESC and LN-DQ; register, "Dataframe query descriptions and 8 lens scenarios" |
| Query language additions | `G/gpu-dataframe/gpu-expression.ts` and query classes, `G/gpu-sql/gpu-sql.ts`: `choose` and `coalesce`, nullable `COUNT(column)`, quoted identifiers, `USING`, group counts, join-error readback. `GT/gpu-dataframe/gpu-expression.node.spec.ts`, `GT/gpu-sql/gpu-sql.node.spec.ts` | Absent; published SQL rejects `COUNT(column)` | No consumer here | This note only (roadmap KG06 lists other SQL gaps) |
| Exact temporal words | `G/arrow/vectors/arrow-temporal-gpu-vector.ts` (`makeArrowTemporalWordGPUVector`). `GT/arrow/arrow-temporal-words.spec.ts` | Absent; published conversion is Float32 relative to an origin | `makeGPUTimeVectorFromPreparedArrowColumn` (`analysis-core/graph-support/`) | Register, "Arrow analytics normalization" |
| Arrow batch streams and partitions | `G/arrow/gpu/arrow-batch-stream.ts`, `arrow-gpu-analytics-adapters.ts` (about 400 lines): append across Arrow versions, lazy partitions with core and halo metadata | Absent | `KeplerArrowStreamPublisher` (`modules/kepler/src/session/arrow-stream-publisher.ts`) for Kepler; nothing generic | Register, "Arrow batch streaming" and "Arrow partition descriptors" |
| Arrow converters above the 1D dispatch limit | `G/arrow/gpu/arrow-compute-kernel.ts` (`getArrowComputeDispatchLayout`) and three converters use bounded 2D dispatch. No over-limit test found | Converters dispatch in one dimension | Converters not used here | This note only |
| Geometry, trips and polygon rendering | `modules/engine/src/arrow/geometry/`, `trips/` (about 3,500 lines; `makeGeometryGPUVectors`, native GeoArrow, WKB and JSON, globe cutting, timed trips), `renderers/polygon/`, `modules/engine/src/graph/models/polygon/` (source vertices, fp64 residual streams). `modules/engine/test/arrow/geometry/*.spec.ts`, `trips/*.spec.ts`. Topology preparation is capped at 4,096 rows and 65,536 vertices | Absent | One Kepler spec excluded; the parked example depends on it | Register, "Geometry-vector helper" (which named only the one helper) |

## Core, engine and the deck layers

| Capability | Fork code and tests | Alpha.13 | Here today | Tracked |
| --- | --- | --- | --- | --- |
| Typed out-of-memory errors, balanced scopes | `modules/core/src/api/utils/out-of-memory-error.ts`; buffer, texture and device changes that keep the GPU error name and balance submission scopes when a call throws. `modules/core/test/api/out-of-memory-error.node.spec.ts` covers the class only | Absent; error scopes work only with `debug` | `isOutOfMemoryError` matches names nothing produces. The parked example imports the missing class (`examples/deck/kepler-compute/src/binding5-device-oom-scope.ts`) | Register, "Out-of-memory detection"; proposals summary, errors |
| Submission epochs and deferred destruction | `modules/core/src/api/adapter/device.ts`, `modules/engine/src/utils/device-submit-ring.ts`, `model/batch-uniform-buffer.ts` (about 220 lines) | Absent | Fences replace the waiting; nothing replaces epoch-based retirement | Register, "`waitForSubmittedWork`" |
| Staging pool and upload ring | `modules/core/src/webgpu/adapter/helpers/webgpu-staging-buffer-pool.ts`, `webgpu-upload-ring.ts` (about 300 lines): pooled staging buffers to 32 MiB, 4 MiB upload blocks, uploads above 1 MiB bypass the ring. `modules/core/test/webgpu/adapter/webgpu-staging-pools.spec.ts` | Absent; `readAsync` makes one staging buffer per call | Published `GPUReadbackRing` where reads repeat | Roadmap LUM-35 |
| Uniform packing and factory keys | `modules/core/src/api/portable/uniform-store.ts`, `factories/bind-group-factory.ts`, `pipeline-factory.ts`, `modules/shaders/src/types/shader-block-layout.ts` (about 80 lines): std140 matrix alignment fix, 16-byte rounding in place of 1 KiB allocations, digested long cache keys. `modules/core/test/portable/uniform-buffer-layout.spec.ts` | Absent | Not worked around | This note only |
| Resource lifetime fixes | `modules/core/src/dynamic/` (borrowed buffers, resize and loading races), `modules/engine/src/compute/` (leaks in `Computation` and the transforms). `modules/core/test/dynamic/*.spec.ts` | Absent | Not worked around | This note only |
| Test devices | `modules/core/src/test-utils/create-test-device.ts` (`destroyTestDevices`). `modules/core/test/test-utils/destroy-test-devices.node.spec.ts` | Absent | No equivalent | Proposals summary, test kit |
| Text atlases | `modules/engine/src/text/` (private bitmap atlases, `clearFontAtlasCache`, glyph traversal) | Changes absent; `walkTextGlyphs` in dist, not exported | The parked example's label files need it | Roadmap KE-EX |
| GPU layers for deck.gl | `modules/deck-gpu-layers/src/` (about 9,500 lines: masks, fp64, globe, joins and extrusion, labels, flow, trip markers, shadows), with its own tests. `dev-docs/roadmaps/deck-core-patch-upstreaming.md` splits the deck.gl core patch into behaviour fixes and luma.gl 10 migration | Not a luma.gl entry point | Parked example imports the unpublished `@deck.gl-community/gpu-layers` | Register, "Kepler-compute example" and "deck.gl 9.4.0 patch" |

## Measurements recorded in the fork

These are the fork's own records, on the machines named there. They were not reproduced, and none
compares against alpha.13. Roadmap UF-03 still asks for fresh numbers.

| What | Number | Fork source |
| --- | --- | --- |
| Block compaction, 8M rows, Apple | 1.4 ms against 3.6 ms | `dev-docs/reviews/comprehensive-gpu-review-2026-11-07.md` |
| Onesweep sort | about 6 dispatches against 24 to 40; on Apple it equals 8-bit radix in time | same |
| Look-back scan | no gain on Apple | same |
| Residency arena, 1,048,576 rows, M3 Pro | 10 graph nodes against 1,249 for compaction over a 32-chunk view; 0.8 ms against 120 ms per queued encoding. The gap between queued and synchronised timing is unexplained | `docs/api-reference/experimental/gpu-tables/gpu-residency.mdx`; `dev-docs/roadmaps/gpugraph-roadmap.md` |
| Point-in-polygon, y-slab refine | 5 polygons of 50,000 vertices by 1M points: 1,833 ms to 115 ms | `dev-docs/roadmaps/gpu-analysis-algorithms-roadmap.md` |
| Point-in-polygon, sorted cells, M3 Pro | 14,400 cells by 250,000 points: 218 ms to 6.7 ms. This one survives here in `GPUPointInPolygonJoin` | `dev-docs/roadmaps/gpugraph-roadmap.md` |
| Projection of 4,194,304 points, UTM local | CPU 1,477.9 ms, GPU resident 443.9 ms, GPU round trip 562.3 ms; the GPU first wins at 65,536 points | `dev-docs/benchmarks/gpu-project-table-performance.md` |
| `Computation` to `Kernel`, SwiftShader | warm compile 43.75 to 3.13 µs per node; CPU encode 7.81 to 9.37 µs per dispatch. Overhead only, not throughput | `dev-docs/roadmaps/arisia-kernel-migration.md` |
| Kepler acceptance, one Apple machine | 1M rows: filter, summary and frame p95 of 22.8, 18.8 and 12.7 ms; 10M rows: 880 MB accounted | `dev-docs/roadmaps/kepler-compute-integration-implementation.md` |
| Sphere physics, Apple | 64 bodies 0.59 to 0.88 ms; 1,024 sparse 2.24 ms; 10,000 sparse 24.43 ms (before the compact and cell-table changes) | `dev-docs/benchmarks/gpu-physics-performance.md` |

## Fork changes this repository turned out not to need

| Fork change | Why it is not needed |
| --- | --- |
| `Device.waitForSubmittedWork` | Public `device.createFence()` does the waiting |
| Bounded dispatch patches in sort, scan, compaction and reduction | The published primitives already apply `getBoundedDispatchLayout`; correct to 33,554,432 rows (`gpu-primitive-large-input-plat01.spec.ts`) |
| `G/gpu-core/benchmarks.ts` barrel | The benchmark runners are public in alpha.13 |
| `GPUVector` offset shortcut in `G/gpu-data/` | Public `GPUData` takes `valueOffsets` |
| `G/gpu-trace` dispatch delegation | The published trace code already bounds its dispatch |
| Device limit plumbing | `requiredLimits` is public (`modules/lenses/src/computation-views/hosts/device-profile.ts`) |
