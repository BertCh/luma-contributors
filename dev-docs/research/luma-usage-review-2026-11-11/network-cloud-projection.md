# Stream network-cloud-projection (NC): luma.gl usage review, 2026-11-11

Installed `@luma.gl/graph@10.0.0-alpha.13` (checked `node_modules/@luma.gl/graph/package.json`).
Read-only. One Node script was run (`review/network-cloud-projection/plan-origin.mjs`, CPU only,
published `compileProjectionPlan` and `planProjection`). No GPU, no spec runs.

## 1. Coverage

Read (published, `.d.ts` and `.js`): `gpu-graph-algorithms/index.d.ts`, `gpu-graph.d.ts`,
`gpu-graph-topology.{d.ts,js}` (constructor and validation in full, `importGPUVector` calls of the
internals), `gpu-graph-{degree,page-rank,core-number,label-propagation,local-clustering-coefficient,
connected-components}.d.ts`, `gpu-graph-page-rank-internals.js` (imports only);
`gpu-core/gpu-coo-to-csr.{d.ts,js}` (whole), `gpu-graph-traversal.d.ts`, `gpu-semantic-spmv.d.ts`,
`gpu-adaptive-spmv.d.ts` (not exported from the index), `gpu-run-length-encode.d.ts`;
`gpu-pointcloud/*.d.ts` (all three); `gpu-geometry/gpu-mesh-normals.d.ts`;
`gpu-project/index.d.ts`, `types.d.ts`, `projection-plan.{d.ts,js}` (compiler, patch fit, f32
error model), `projection-planning.{d.ts,js}` (whole), `projection-program.js` (`CompiledProjection`,
`compileProgram` in full), `gpu-projection.d.ts`, `gpu-projection-program.d.ts`,
`cpu-projection-operation.{d.ts,js}`, `cpu-geometry-pipelines.d.ts`, `crs.d.ts`,
`projection-table-transform.d.ts`, `web-mercator.{d.ts,js}`; `gpu-dggs/gpu-dggs-cell-projection.d.ts`,
`gpu-h3/*.d.ts`; `geospatial/gpu-haversine-distance.{d.ts,js}` (kernel text);
`arrow/dggs/dggs-gpu-polygons.d.ts` and the point-count constants of its `.js`; the `dggs` shader
module source (function list, `dggs_web_mercator_tile_y_to_latitude`, `dggs_h3_neighbor`, constants).

Read (local): A: `analysis-core/network-topology.ts:24-42`; `gpu-network-analytics-columns.ts:50-135`;
`gpu-network-statistics.ts:230-300`; `gpu-network-neighborhood.ts:20-130`; `gpu-network-noding.ts`
`:95-115`, `:600-720`; `gpu-network-line-graph.ts` (grep of passes); `gpu-network-coarsening.ts:300-330`,
`:1100-1150`; `analysis-core/kernels/sorted-segment-offsets.ts` (whole) and its 7 callers (grep);
`spatial-weights/slot-grouping.ts:14-90`; `landscape-ecology/sparse-resistance-system.ts:180-235`.
Counted atomics/scan/sort per CSR-building file (11 files). B: `gpu-pointcloud-statistical-filter.ts:15-60`,
`gpu-pointcloud-neighborhoods.ts:24-110`, `gpu-pointcloud-voxel-merge.ts:1-80`, `:130-272` (grep),
grep for octree/Morton/normals/radius filters over `modules/*/src`. C: `geospatial-analysis/*` (index,
`projection-spatial-context.ts` whole, datum-grid header), `gpu-local-frame-projection.ts` (whole),
`cpu-local-frame-projection-operation.ts:1-140`, `gpu-local-frame-projection-operation.ts:25-55`,
`gpu-raster-reprojection.ts:1-200`, `gpu-raster-tile-reprojection.ts` (destroy), `analysis-core/contracts/
earth.ts`, `local-frame.ts` (imports), `raster-grid.ts` (Mercator lines), the Mercator grep over
`modules/{analysis,lenses,kepler}/src`. D: `cell-indexing/gpu-cell-geometry.ts` (whole),
`cell-cover/cell-cover-kernels.ts:290-310`, `:455-480`, `cell-cover-scanline.ts:175-195`,
`cell-aggregation/cell-keys.ts:100-130`, `cell-topology/h3-neighbor-wgsl.ts:1-80`,
`gpu-geodesic-pairs.ts:96-112`, the spec list of `gpu-cell-geometry.spec.ts`, call sites of
`GPUCellGeometry` (10 outside the class).

Not done (a later pass should know): the 2,149-line `gpu-map-matching.ts` and `network-k-function`,
`network-isochrones`, `edge-bundling`, `flow-aggregation` internals were only grepped, not read;
their uses of `GPUGroupAggregation` (6 constructions under `gpu-network`) were not checked for the
float-atomics order dependence the register describes. `cell-table-compare`, `healpix-cells` and
`hilbert-keys` were checked only for a published counterpart (none: `dggs` has A5 Hilbert digits
only, no HEALPix, no planar Hilbert). `pointcloud-change`, `-icp`, `-surfaces`, `-features` were not
read beyond headers. The projection error comparison is a CPU model, not a GPU run. Website, kepler
and lenses hits are listed as leads only.

## 2. Findings

### NC-01 `GPULocalFrameProjection` can use the published `planProjection` + `GPUProjectionProgram` with `destinationOrigin`

- Class: `stale-claim` (register row "Projection plan origin") and `adopt-with-cost`.
- Local: `modules/analysis/src/gpu-spatial-analysis/geometry-measures/gpu-local-frame-projection.ts:81-95`
  (compiles a `local-f32` plan, then overwrites `destinationOrigin` with `[0, 0]`).
- Published: `node_modules/@luma.gl/graph/dist/gpu-project/projection-planning.d.ts:37` (`destinationOrigin`
  in `ProjectionOutputOptions`), `projection-planning.js:14-51` (adaptive plan compiled in
  `double-single`, program origin taken from the option), `projection-program.js:333-339` (the
  program subtracts the origin in double-single and rounds once to f32), `gpu-projection-program.d.ts`
  (`positions` may be `float32x2` or `uint32x4`, `validity` output).
- Evidence: read both; plus a CPU run of both compilers on a spherical AEQD frame
  (`review/network-cloud-projection/plan-origin.mjs`, origin (-73.98, 40.75), square bounds offset
  by one degree, tolerance 0.01 m):

  | bounds half-width | local `compileProjectionPlan` | `planProjection` (`local-f32`, origin `[0,0]`) |
  | --- | --- | --- |
  | 0.25 deg | 1 patch, f32 fit 0.0026 m | 1 patch, fit 0.0001 m |
  | 1 deg | 7 patches, f32 fit 0.0094 m | 4 patches, fit 0.0016 m |
  | 2 deg | throws "exceeds its tolerance at the maximum subdivision depth" | 16 patches, fit 0.0016 m |
  | 4 deg | throws | 64 patches, fit 0.0017 m |

  So `compileProjectionPlan` still has no origin option (the register is right about that
  function), but the public planning entry point has had one since at least this release and is the
  luma.gl way to get origin-relative f32 rows. The published route validates the fit for the rows
  it writes, removes the "tolerance does not describe the rows" caveat of the class documentation
  (`:55-62`), accepts regional bounds the local route refuses, and rounds exactly once at the output,
  which is what the CPU twin does (`cpu-local-frame-projection-operation.ts:36-44`), so GPU/CPU
  parity should tighten.
- Change: construct `planProjection({projection: frame.project, bounds, tolerance, precision:
  'local-f32', destinationOrigin: [0, 0], inputFormat: <positions format>, onUnsupported: 'throw'})`
  and a `GPUProjectionProgram({projection: result.compiled, positions, output, validity})`; keep
  `getCommandNodes` through `captureGraphCommandNodes` and `destroy`. One class, one call site of
  the compiler; the public `plan: ProjectionPlan` field becomes the program (no reader of `.plan`
  in `modules`, `examples`, `website`: grep found none).
- Cost: double-single arithmetic per row instead of f32 polynomials (speed not measured; needs a
  GPU run), and the float32 rounding of the absolute frame coordinate stays (it is inherent in
  origin `[0, 0]` f32 output).
- Verify: `modules/analysis/test/gpu-spatial-analysis/geometry-measures/gpu-local-frame-projection.spec.ts`
  (0.02 m bound), `parity-wp2-local-frame.parity.spec.ts`, `parity-wp2-round8-local-frame.parity.spec.ts`,
  `local-frame-projection-operation.spec.ts` (GPU against the CPU twin). Add a case with 4-degree
  bounds, which throws today.
- Risk: patch layout and per-row results change at the centimetre level; rows on a patch boundary
  may pick a different patch; `inputFormat` must be fixed at construction from the positions view.

### NC-02 `createSortedSegmentOffsetsNode` is the offsets kernel of published `GPUCOOToCSR`; the register compares it with the wrong class

- Class: register correction, `adopt-with-cost`.
- Local: `modules/analysis/src/analysis-core/kernels/sorted-segment-offsets.ts:26-53` (one lower
  bound per segment `0..segmentCount` over ascending keys); 7 call sites:
  `gpu-network/network-noding/gpu-network-noding.ts:675`, `spatial-regression/regime-moments.ts:137`,
  `location-allocation/location-allocation-sparse.ts:223`, `spatial-weights/slot-grouping.ts:64`,
  `trajectory-zones/gpu-zone-events.ts:796`, `geometry-measures/gpu-geometry-measures.ts:515`,
  `raster-analysis/gpu-raster-fractional-zonal-distribution.ts:404`.
- Published: `node_modules/@luma.gl/graph/dist/gpu-core/gpu-coo-to-csr.js:59-95`: for each row
  `index` in `0..rows` (rowOffsets has `rows + 1` entries) it writes the binary-search lower bound
  of `index` in the row-sorted `rowIndices`; keys `>= rows` sort last and are excluded exactly as
  the local sentinel `segmentCount` is. Same results by construction (read both).
- The register row (`GPUGallopingSearch`, `GPUSegmentOffsets`; reason at `sorted-segment-offsets.ts:20-22`)
  says delegation needs a query ramp, a segment record and a validation word. `GPUCOOToCSR` needs
  none of those; what it costs instead is two `nnz` copies (`columnIndices` uint32, `values`
  float32, `gpu-coo-to-csr.js:45-58`) and a float32 values column even when the caller has none.
- The repository already pays that cost once: `sparse-resistance-system.ts:177-215` writes
  `entryColumns` and `entryConductances` and then `GPUCOOToCSR` copies them into `csrColumns`
  and `csrConductances` (two extra passes and two `nnz` transients). Noding (`:604-720`) does the
  opposite: local offsets and a gather that writes the CSR arrays directly.
- Change: rename the published counterpart in the register row to `GPUCOOToCSR` and state the
  real cost (two copies). What luma.gl needs: an offsets-only mode (optional `columnIndices` and
  `values`) on `GPUCOOToCSR`; then the 7 calls delegate with no loss. No code change today.
- Verify: a node spec cannot run it; a headless parity spec (random sorted keys with sentinel rows,
  compare `rowOffsets` with the local node) would prove bit equality.
- Risk: none for results (integer lower bounds); only passes and memory.

### NC-03 `CompiledProjection.getShader()` would let `GPURasterReprojection` project inside its sampler

- Class: `unused-feature`.
- Local: `modules/analysis/src/raster-analysis/gpu-raster-reprojection.ts:120-197`: three passes
  (grid generation into a `float32x2` transient, `GPUProjectionProgram` into a second `float32x2`
  transient plus a validity transient, sampling). `:83-89` refuses `double-single` programs
  because the intermediate rows are f32.
- Published: `node_modules/@luma.gl/graph/dist/gpu-project/projection-program.js:36-63`
  (`getShader({namespace, parameterOffset})` returns the callable WGSL, its fp64 module, defines,
  entry point and parameter binding name; documented "Namespace permits several independent
  transformations in one consumer shader").
- Evidence: read one (published source read; not tried with the local kernel builder).
- Change: call `projection_<ns>_project(position, 1u)` inside the sample kernel. Removes two
  transients of `width * height` rows and one pass per reprojection, and lets a `double-single`
  program be differenced against the source origin in double-single (lifting the restriction at
  `:83-89`). One class (the tile variant composes it). The kernel must be assembled with the
  returned module (public `WGSLShaderAssembler` of `@luma.gl/shaders`) and bind the parameter words,
  which `createWGSLKernelNode` does not do today.
- Verify: the existing reprojection specs under `modules/analysis/test/gpu-raster` plus a
  `double-single` UTM case.
- Risk: medium effort; results identical for `local-f32` programs (same WGSL), better for
  `double-single`.

### NC-04 Two private copies of `dggs_web_mercator_tile_y_to_latitude` in cell cover

- Class: `adopt-with-cost` (or pin).
- Local: `modules/analysis/src/gpu-spatial-analysis/cell-cover/cell-cover-kernels.ts:469-472` and
  `cell-cover-scanline.ts:187-190` (`coverTileLatitude`).
- Published: `dggs` module of `@luma.gl/shaders`, `fn dggs_web_mercator_tile_y_to_latitude`, already
  called by `cell-indexing/gpu-cell-geometry.ts:331-340` and `cell-set-outline/gpu-cell-set-outline.ts:579`.
- Evidence: read both. Same operations in the same order; `Math.fround(3.14159265358979) ===
  Math.fround(3.141592653589793)`; `row * 2^(1-z)` and `2 * row / 2^z` are exact for integer and
  half-integer rows, so the results are bit-identical. The copy spec
  (`modules/analysis/test/analysis-core/official-wgsl-copies.node.spec.ts`) does not see them because
  it matches identical text, not identical arithmetic.
- Change: either concatenate `dggs.source` in the two Quadbin branches (cost: the about 2.3k-line
  module in two kernels that do not include it today, the same reason `cell-keys.ts:110-111` gives
  for `dggs_u64_*`), or list them as pinned semantic copies. Cell cover must keep agreeing with the
  tile edges `GPUCellGeometry` draws, which is why a pin matters.
- Risk: none today.

### NC-05 Web Mercator arithmetic uses the WGS84 ellipsoid constant or a literal; `WEB_MERCATOR_EARTH_RADIUS` is never imported

- Class: `adopt-now` (naming; no numeric change).
- Local: `analysis-core/raster-grid.ts:134`, `:168` (EPSG:3857 `y / a`), `gpu-terrain/point-horizon/
  point-horizon-parameters.ts:186-187` (world size and Mercator ordinate), `raster-analysis/
  raster-patches/gpu-raster-patch-metrics.ts:587` (literal `6378137`). 5 sites in 3 files.
- Published: `node_modules/@luma.gl/graph/dist/gpu-project/web-mercator.js:6` (`WEB_MERCATOR_EARTH_RADIUS =
  6_378_137`), `:8` (`WEB_MERCATOR_MAX_LATITUDE`). Grep: no file in `modules`, `examples` or `website`
  imports either.
- Keep `GPU_WGS84_SEMI_MAJOR_AXIS_METERS` where it means the ellipsoid (`raster-grid.ts:63`,
  `point-horizon-march.ts:531`, `spatial-context.ts:431`, `geodesic-wgsl.ts:18`);
  `trajectory-estimation/prepare-trajectory-observations.ts:17` repeats the ellipsoid value as a
  literal and should import the local constant. Keep the three `85.051129` (Quadbin `clip_latitude`
  of quadbin-py: `cell-cover-kernels.ts:309`, `cell-keys.ts:201`; f32-equal to the published value).
- Leads outside this stream (CPU inverse Mercator by hand, candidates for
  `createWebMercatorProjection().unproject`): `modules/kepler/src/cluster/viewport-bounds.ts:29`,
  `modules/kepler/src/cluster/kepler-compact-cluster.ts:88`,
  `modules/kepler/src/analysis/catalog-platform4-dggs-cells.ts:36-45`, `:270` (its own
  `85.051129` at `:16` differs from the published float64 value by 2.2e-7 degrees),
  `modules/lenses/src/computation-lenses/scenarios/gpu-spatial-analysis/cell-display.ts:129`.

### NC-06 Dead projection glue in `geospatial-analysis`

- Class: `misuse` (unused local code next to a published path nothing uses).
- Local: `modules/analysis/src/geospatial-analysis/projection-spatial-context.ts:24-58`
  (`getGPUProjectionSpatialContext`, `getGPUProjectionFrameOrigin`). Not exported by
  `geospatial-analysis/index.ts`; no caller in `modules/*/src`; used only by
  `modules/analysis/test/gpu-project/projection-spatial-context.node.spec.ts` and
  `projection-frame.node.spec.ts`, the latter feeding it to published `prepareCRSProjection`
  (`@luma.gl/graph/gpu-project/crs`, which no source file imports).
- Change: delete the file and the two specs, or give it a caller (a contributor that prepares a CRS
  projection with the frame's `destinationOrigin`). 2 functions, 0 source callers.

### NC-07 The website run leaks the plan buffer of `GPULocalFrameProjection`

- Class: `misuse` (lead; website is outside this stream).
- `website/src/gpu/runs/entries/geometry-processing/local-frame-projection.ts:60-76` adds
  `new GPULocalFrameProjection(...)` to the graph without `builder.own(...)`
  (`website/src/gpu/runs/define-run.ts:110`), so the published `GPUProjection` plan buffer is never
  destroyed on rebuild. The showcase (`hot-spots.compute.ts:1192` via `resources.track`) and the
  operation form (`gpu-contributor-operation.ts:140-160`) release it.

## 3. Checked and fine

- Register claim on graph algorithms (A) holds at alpha.13. Every published algorithm takes a
  `GPUGraphAlgorithmsTopology`; its only constructor validates single-chunk physical `GPUVector`s with
  distinct allocations (`gpu-graph-topology.js:28-53`, `:86-142`) and `addToGraph` rebuilds the CSR
  from source edge columns via `importGPUVector` (`gpu-graph-topology-internals.js:15-54`). The
  algorithms do have `addToGraph(commandGraph)` and so can join a caller's graph (through
  `captureGraphCommandNodes`), but there is no public way to build a topology from views or from an
  existing CSR. Running an algorithm on a topology whose `addToGraph` was skipped would rely on
  internals, like the retired cast.
- No local contributor in `gpu-network` takes a caller-owned physical edge list: the family
  contract is a CSR of `GraphDataView`s (`analysis-core/network-topology.ts:24-42`). The two examples
  that need PageRank, label propagation and communities already construct the published classes
  directly (`examples/deck/spatial-analysis-showcase/src/scenes/flows/airline-network.compute.ts`,
  `bixi-communities.compute.ts`); those classes own no GPU resources, so no `destroy` is missing.
- No local re-implementation of PageRank, core number, label propagation, local clustering
  coefficient, force layout or spatial force layout in `modules/*/src` (grep over all three
  packages; the hits are minimum-label component propagation, already a register row). The analytics
  columns state the reason at `gpu-network-analytics-columns.ts:112-114`.
- `GPUGraphTraversal` accepts views (`gpu-graph-traversal.d.ts`) but publishes only a mask, so the
  register reason for `GPUNetworkNeighborhood` (hop distances) holds.
- `GPUGraphCSRMatrix` / `GPUGraphSpMV` are semantic `GPUGraph` operations; `GPUAdaptiveSpMV` is not
  exported from the `gpu-core` index, so nothing can use it.
- Count-scan-fill CSR builds (`network-line-graph`, `network-subgraph-filter`, `neighbor-search`,
  `contiguity-weights`, `lattice-weights`, `network-cost-weights`, `map-matching`) know each row's
  length from the data, with no COO to sort; `GPUCOOToCSR` would need a sort first.
- `GPUNetworkCoarsening` superedges (sort, segment flags, scan, fixed-point segment sums):
  `GPURunLengthEncode` takes the same single `uint32` key but outputs at input capacity with no
  segment offsets or weight sums, so it would add a copy and still need the local sums (lead only).
- `GPUCOOToCSR` in the sparse resistance system is constructed and added correctly
  (`sparse-resistance-system.ts:204-215`); see NC-02 for its copy cost.
- B: no local octree, mesh-normal or radius-outlier code exists. `GPUMeshNormals` work on triangle
  meshes; the point-cloud normals are PCA over neighbourhoods. `GPUPointCloudRadiusFilter` is O(N²)
  all-pairs; `GPUPointCloudNeighborhoods` already returns the true radius support in `counts` with a
  grid, so a radius filter is one threshold away and the published class would be slower.
  `GPUPointCloudStatisticalFilter` uses published `GPUReduction`. The voxel-merge reason (no
  centroids from `GPUPointCloudVoxelDownsample`) still holds at alpha.13
  (`gpu-pointcloud-voxel-downsample.d.ts`: representatives, counts, representative positions).
- C: `CPUProjectionOperation` / `projectCPUVector` would not serve the local-frame CPU twin: they take
  float32 rows only (no raw binary64 `uint32x4`), have no bounds rejection or validity output, run
  asynchronously and use bare `assert` (`cpu-projection-operation.js:9-33`). The twin's direct
  `frame.projectArray` is the better oracle.
- C: `GPURasterReprojection` and the tile variant construct `GPUProjectionProgram` once per
  contributor, add it through `captureGraphCommandNodes`, and destroy it
  (`gpu-raster-reprojection.ts:99-101`, `gpu-raster-tile-reprojection.ts:83-85`).
- C: local frames delegate to math.gl (`analysis-core/contracts/local-frame.ts:5`); the datum grid
  transform has no published counterpart (no grid-shift support found in `gpu-project`).
- C: `GPU_SINUSOIDAL_KILOMETERS_PER_DEGREE` (`earth.ts:28`) is used only by a test; its removal is
  already deferred (luma-official-08, roadmap 12.4). `GPUSinusoidalProjection` has no local user.
- D, item 12 (H3/A5 centres through `GPUDGGSCellProjection`): the reason is good. The local centre
  path is two WGSL lines (`gpu-cell-geometry.ts:434-437`, `:457-459`) inside the one kernel that also
  writes boundaries and vertex counts, both of which callers use (kepler `catalog-platform4-dggs.ts:305`,
  lenses `cell-display.ts:58`, `:93`). Delegating would add the published pass, a NaN fix-up pass
  for invalid rows (it writes 0), and keep the local kernel for boundaries: three passes for
  identical numbers, nothing deleted. Only a centres-only H3 caller (the hot-spot recipe,
  `hot-spot-analysis-recipe.ts:1024`) could use `GPUH3CellProjection` alone, at the cost of a
  family branch and zero instead of NaN. Parity stays proven by `gpu-cell-geometry.spec.ts:507`.
- D, item 13 (`GPUGeodesicPairs` sphere mode onto `GPUHaversineDistance`): the reason still holds at
  alpha.13: both published paths take a raw longitude difference (`gpu-haversine-distance.js:57-68`,
  f32 `right - left` and raw binary64 `sub_fp64u32_to_f32`), with no seam handling; the local kernel
  also writes bearings and midpoints in the same pass.
- D: `convertDggsCellKeysToGPUPaths` (`@luma.gl/graph/arrow`) is not a replacement for
  `GPUCellGeometry` boundaries: it submits on its own outside a command graph, has no Quadbin, and
  uses 7 points for H3 (`dggs-gpu-polygons.js:14`), dropping pentagon and distortion vertices
  (local: up to 10). `dggs_h3_neighbor` is an IJK step, not a cell-index neighbour; local
  `h3-neighbor-wgsl.ts` already reuses 11 published `dggs_h3_*`/`dggs_u64_*` helpers. `dggs` has no
  geographic-to-cell functions, so `GPUPointToCell` has nothing to delegate to.

## 4. Register corrections (`dev-docs/upstream-workarounds.md`)

- Section 2, "Projection plan origin" (line 66): "What luma.gl needs: a `destinationOrigin` option
  on `compileProjectionPlan`" is met by public `planProjection` (`destinationOrigin`) with
  `GPUProjectionProgram`; status should become "replaceable" (NC-01).
- Section 4, `GPUGallopingSearch`, `GPUSegmentOffsets` row (line 100): the closest published
  counterpart of `createSortedSegmentOffsetsNode` is `GPUCOOToCSR`; delegation costs two `nnz` copies,
  not a query ramp; luma.gl need: an offsets-only `GPUCOOToCSR` (NC-02).
- Section 1, "Command-graph-transient CSR" (line 27): accurate, but add that the algorithms can be
  added to a caller graph (`addToGraph`); what is missing is a topology built from views or from an
  existing CSR, not graph attachment.
- Line drift: `GPUNetworkNoding` class is at `gpu-network-noding.ts:211` (row says `:213`);
  `GPUNetworkLouvain` at `gpu-network-louvain.ts:114` (row says `:111`); `createPointCloudGridIndex`
  at `pointcloud-grid-index.ts:92` (row says `:85`; the gap list is now around `:69-90`).
