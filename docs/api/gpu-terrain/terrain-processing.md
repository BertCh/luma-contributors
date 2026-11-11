# Terrain processing

These contributors emit WebGPU command-graph nodes. All numeric inputs and outputs remain on the GPU; there is no host convergence read or CPU algorithm fallback. Callers own outputs. Every invocation rewrites output statuses. Inputs must be packed graph views, and outputs must not alias inputs or one another. The native GPU tests are in `modules/analysis/test/gpu-terrain/terrain-processing/terrain-processing.spec.ts`.

## Breaching and explicit burning

`GPUTerrainBreach` consumes explicit pit cell IDs and a canonical float32 elevation surface. The cut is vertical, so every output is in elevation units and no horizontal unit or `cellSizeMode` applies. It minimizes **summed excavation depth** on a D8 graph layered by path step. A candidate at step `s` is lowered to at most `pitElevation - s * elevationDecrement`; paths may use at most `maximumSteps` and each proposed cut at most `maximumCutDepth`. Endpoints must already lie at or below that descending level. Equal costs select the earliest step, then cell order, then predecessor ID. This is a named bounded objective, not a claim of Whitebox bitwise equivalence or global minimum earthwork volume.

The decrement must survive float32 subtraction at each pit. Nodata is a barrier: NaN, or any cell whose optional `validity` word (uint32, nonzero valid) is zero, which is treated as NaN and leaves NaN in the output elevation. A pit with invalid elevation or an unrepresentable first decrement is invalid. Failure to find a bounded path remains unresolved. Each pit search uses the original surface; overlapping proposals merge by minimum elevation. A post-edit pass checks every proposed path for strict descent and marks conflicts unresolved. Edits remain visible even for an unresolved overlapping path; callers may use the returned path/provenance to choose a different batch. The operator does not implicitly fill sinks.

`burnDepth`, when supplied, explicitly burns after breaching. Negative/nonfinite burn depths produce NaN and modification-mask bit 4. Mask bits 1 and 2 record breach and burn respectively. Burn depth is separate from the breach depth constraint. The returned cut depth includes both operations.

A path of at most `S = maximumSteps` D8 steps cannot leave its Chebyshev radius-S window. Search scratch is therefore `8 * pitCount * (S + 1) * min(width, 2*S + 1) * min(height, 2*S + 1)` bytes, independent of the rest of a large DEM. It lives in one storage binding, so `getCommandNodes` rejects a request above the device's `maxStorageBufferBindingSize` (128 MiB by default, about 15 pits at `maximumSteps` 64) with `<id> search scratch needs N bytes in one binding (limit M); reduce pitCells.length or maximumSteps`. The constructor rejects a request whose scratch word count would overflow a 32-bit index. Deterministic ordered-float integer atomic minima combine path proposals in a separate pass; applying edits scans the DEM once, rather than scanning every pit path for every cell. An additional four bytes per domain cell holds proposed edits. Searches remain bounded by the explicitly selected pits and step limit; they are not an unbounded priority-queue breacher.

## Branching consumers

`GPUTerrainFlowFractions` materializes the existing D-infinity, Freeman MFD and Quinn MFD routing code from the same surface, parameters and D8 fallback directions as `GPUTerrainFlow`. A valid cell whose direction is unresolved keeps NaN fractions: its receiver is unknown, so everything downstream of it stays unresolved. A cell without data (NaN elevation) has a zero row: it donates nothing, as in the accumulation of `GPUTerrainFlow`, so a coastline or a mask does not block the cells around it. The zero row does not by itself make the no-data cell invalid in a consumer: give `GPUTerrainBranchingFlow` the same `validity` view (or NaN `values`), and give `GPUTerrainTiledFlow` the `flowDirections` of `GPUTerrainFlow`. It also supplies row-dependent ground edge lengths, in the units of the `parameters` cell size (ground meters under `cellSizeMode`). An optional `validity` view (uint32, nonzero valid) marks `elevation` NaN where zero.

`GPUTerrainBranchingFlow` consumes materialized eight-neighbor fractions, east first clockwise, matching D8 direction order. Its inputs can represent existing D-infinity/MFD routing. Fractions are not silently renormalized: each positive row must sum to one within `1e-5`; a zero row is a terminal. Positive fractions leaving the supplied domain are invalid.

Modes have different recurrences:

- `transport`: local nonnegative injection plus the sum of each donor's transported mass times its outgoing fraction.
- `expected-distance`: sum of each receiver's distance plus edge length, weighted by its fraction.
- `minimum-distance` / `maximum-distance`: extrema over positive-fraction receivers; zero fractions do not participate.
- `fractional-catchment`: terminal membership in `[0,1]` propagated upstream as outlet probability. Execute separate instances for independently selected outlets.

Distances use explicit `edgeLengths` (in whatever unit the caller chose) or lengths measured from the grid: affine units by default, or ground meters under `cellSizeMode`. `transport` and `fractional-catchment` carry no length unit. An optional `validity` view (uint32, nonzero valid) marks `values` NaN where zero. Values are finite/nonnegative, including the otherwise unused distance-mode source view. Each cell uses one atomic uint32 word that publishes the final float32 value and readiness together. Nodes finalize only after all required neighbors finalize; fixed neighbor-order pulls avoid floating-point atomic accumulation. A successful publisher follows ready successors for up to `maximumWalkLength` steps (default 32, maximum 1024). GPU-gated sweeps stop after no progress or the explicit sweep budget. The default `maximumIterations` clamps `width + height` to [128, 1024]; explicit budgets must be in [1, 1024]. A sweep can resolve multiple dependency levels, so a short-budget partial result can depend on GPU scheduling. Finalized values remain deterministic. Cycles, invalid dependencies (a donor or receiver with a positive fraction whose own input is invalid, or a donor whose fraction is NaN) and insufficient budgets remain unresolved with NaN values. A neighbour with a zero fraction toward the cell is not a dependency, whatever its own state. Dimensions alone do not guarantee convergence for an arbitrary winding DAG. Scratch readiness/value storage is four bytes per cell, plus small loop-control buffers. Float32 mass conservation is approximate.

## Depression hierarchy and stage curves

`GPUTerrainDepressionHierarchy` creates a sublevel-set merge tree over finite D8-connected cells. It activates cells by `(elevation, cellID)`, creates nodes at minima and merges, and preserves each cell's single activation owner. Parents connect child components at saddles; parent queries include descendants exactly once. Equal-height saddles produce a deterministic binary/multifurcating refinement with zero-persistence branches; topology-equivalent collapsed trees are the appropriate external comparison.

`nodes` stores parent, minimum cell, lowest-ID explicit outlet in the component, and validity. `levels` stores birth and parent-merge elevation; roots have infinite merge elevation. The outlet field distinguishes externally draining roots from real closed components. Root infinity is a topological sentinel, not an infinite physical storage claim. Nodata barriers are never converted to ocean. Curves count cells at or below each query stage and sum positive stage-minus-elevation depth times cell area: squared affine units by default, or square meters under `cellSizeMode` (per-cell ground area at its row). An optional `validity` view (uint32, nonzero valid) marks `elevation` NaN where zero, so the cell is a barrier. They are geometric stage-area-volume curves for a node's owned region; they do not simulate draining through outlets or dynamically fill/spill/merge.

Internal capacity is at most one node per cell. Caller capacity may be smaller: exact required count and overflow are returned (in `output.status` and as the optional named one-row ports `output.ports.status.requiredCount` and `output.ports.status.overflow`; `output.status` is the packed view, so the named ports sit under `ports.status`), all published nodes are invalid, and all curves are NaN on overflow. Cell ownership is inspectable internal node identity even on overflow and must not be consumed then. Stable GPU radix/bitonic sorting establishes activation order; a serial GPU activation pass uses compressed disjoint-set roots without repeatedly scanning the raster or rewriting every component label. Historical parent links remain separate from compressed connectivity links. The activation pass remains serial, which is the remaining tree-construction throughput limit. Stage queries first reduce exclusive cell contributions, then propagate each child once in creation order: `O(stageCount * (cellCount + nodeCount))` work rather than a scan for every ancestor query. Stage areas and volumes are fixed-point integer sums (three uint32 words per curve value, in a unit set by the largest term of the stage), so they are the float32 nearest the exact sum of the float32 terms whatever the number of cells, and shader fast-math has no float identity to fold. The curve sum scratch is two views of the output curve size (the middle and high words; the low words use the output itself until a value is decoded), so no scratch binding is larger than the caller's curves; sorting and tree scratch are linear in cell count. Native scale coverage includes 16,384 cells.

**Size limit.** The merge-tree build is inherently serial (a single GPU invocation walks the sorted cells with union-find), so its run time grows linearly with cell count and very large grids can approach GPU watchdog limits. Tile very large domains. The unresolved counts of `GPUTerrainBreach` and `GPUTerrainBranchingFlow` are parallel atomic reductions and are not limited this way.

## Basin summaries and profiles

`GPUTerrainCatchmentSummary` computes valid area, mean/minimum/maximum elevation, stream length, drainage density, cumulative normalized-elevation bins and directed stream profiles. Finite elevation is the validity mask; out-of-range basin IDs are excluded. A stream cell is any nonzero `streams` word other than the unresolved sentinels (0xffffffff, 0xfffffffe), the same rule as stream order, stream network, flow distance and height above drainage, so a mask of Strahler orders counts every stream edge. `flowDirections` are ESRI codes unless `directionEncoding` names `'whitebox'` or `'taudem'`. Each stream edge is counted once in its source basin, including an edge crossing a basin boundary. Nonstream receivers end a stream profile. Cycles and invalid directions remain unresolved. Empty basins have zero area/length and NaN mean/extrema/density.

All physical quantities use the supplied planar affine coordinate units by default: lengths in coordinate units and areas in squared units, so geographic degrees are not converted to meters. With `cellSizeMode: 'web-mercator'` or `'geographic'` (axis-aligned grids only) areas are square meters at each cell's row, and stream lengths, profile distances and gradients use the ground length of each D8 move at its midpoint row, the same rule as `getD8Distance`. Elevation statistics keep the elevation's units. An optional `validity` view (uint32, nonzero valid) marks `elevation` NaN where zero. A rotated affine is supported. The executed reference is explicitly composed pysheds catchment membership plus independent NumPy reductions, not a claim that one upstream toolbox exposes this exact API. Basin IDs are stably sorted once and converted to contiguous ranges: statistics and each histogram bin visit only their basin members. The per-basin sums (elevation for the mean, ground areas and hypsometry areas under `cellSizeMode`, stream length) are fixed-point integer sums in a unit set by the basin's largest term, so they do not lose accuracy with basin size: a serial float32 sum reported a mean of 2015.58 for a million cells at a constant 2000, and an area 0.6 % off. This removes basin-count times raster-size scanning. Directed profiles use pointer jumping in `ceil(log2(cellCount))` rounds; their edge-count sidecar preserves the exact `maximumSteps` contract, including the terminal inspection step. Cycles remain unresolved. Native scale coverage includes 65,536 cells, 256 basins and all per-cell profiles; each ping-pong profile state uses 16 bytes per cell.

## DEM comparison

`GPUTerrainDEMCoregistration` implements the median vertical-shift objective of xDEM `VerticalShift`: fit `median(moving - reference)` on stable finite cells, then subtract it. Even-sized medians average the two middle order statistics; ties are deterministic. NMAD uses `1.4826022 * median(abs(residual - median))`. There is no horizontal resampling, Nuth–Kääb model, tilt or ICP. Fewer than two stable samples leave uncertainty unresolved and output validity zero.

The caller supplies per-cell difference variance and an optional exponential spatial-correlation length in the grid's horizontal units: affine units by default, ground meters under `cellSizeMode`. The volume is in vertical units times the area units above; under `cellSizeMode` its uncertainty scales the correlated term by the mean valid cell area, an approximation of the per-cell weights. An optional `validity` view (uint32, nonzero valid) marks both DEMs NaN where zero. Zero length means independent measurement errors. `correlationLength` is the e-folding length `L` of the correlation `exp(-d / L)`: 0.37 at `L`, 0.05 at `3 L`. It is not the practical (effective) range of an exponential variogram, which is `3 L`; if your variogram tool reports that range, pass a third of it, otherwise the volume uncertainty is overstated. The bias standard error uses the large-sample Gaussian median approximation `sqrt(pi/2) * NMAD * sqrt(sum(correlation)) / stableCount`. This is an explicit model approximation; NMAD zero can underestimate uncertainty on quantized/flat data. Integrated volume variance includes pairwise measurement covariance and a common fitted-bias variance term. The model treats fitted-bias error as independent of measurement errors; it omits calibration/evaluation cross covariance, so the reported uncertainty is not an exact confidence interval for reused calibration samples. Inspect this assumption when integrating stable-ground pixels. Residual uncertainty is not automatically measurement uncertainty.

Median and NMAD use two stable GPU sorts followed by a constant-time selection; the stable count, covariance sums, valid count and volume sums are `GPUReduction` sums, so no kernel loops over cells in one invocation (float32 sums are tree-ordered, not sequential). Independent measurement covariance takes linear work. Nonzero exponential correlation retains the full dense covariance model: one GPU invocation computes each covariance row, followed by a linear reduction. This is exact for the stated discrete covariance model up to float32 arithmetic and uses linear scratch, but requires quadratic arithmetic for nonzero correlation. It does not silently truncate long-range covariance. Native scale coverage includes a 16,384-cell independent-error fit; correlated coverage checks a full independently constructed 1,024-cell covariance matrix with heterogeneous variances and a sheared affine.

## Cross-tile D8 and branching accumulation and reach IDs

Accumulation is in runoff units (a transport sum), so `GPUTerrainTiledFlow` has no length or area unit and no `cellSizeMode`. NaN runoff marks a cell invalid.

`GPUTerrainTiledFlow` consumes globally prepared D8 directions by default. With `routing: 'branching'`, each tile also supplies eight-neighbor `fractions`, such as materialized D-infinity/MFD output. In branching mode the routing comes from `fractions`; the only word read from `flowDirections` is `GPU_TERRAIN_FLOW_NONE` (0xffffffff, a cell without data), which makes the cell invalid (status 0, NaN accumulation) as in D8 mode, so the view may contain zeros when every cell has data. A stream cell is any nonzero `streams` word other than the unresolved sentinels (0xffffffff, 0xfffffffe), so a mask of Strahler orders is a valid stream mask. Positive fraction rows must sum to one within `1e-5`; zero rows are outlets. Negative/nonfinite fractions or positive edges leaving the global domain invalidate a row. Fractions are never rerouted or renormalized at seams. The required `grid` describes the complete domain. Tiles specify input-buffer origin, shape and an owned rectangle; rectangles must cover the domain exactly once without overlaps. Halo data never contributes runoff, and owner values are used at seams. In D8 mode only global zero directions represent outlets; a positive direction leaving the global domain is invalid.

Owned-cell exchange dispatches copy routing, runoff and stream flags into one resident global DAG. Atomic dependency sweeps then resolve transport across all seams, including repeated exits/reentries and diagonal corners, without assuming the tile dependency graph itself is acyclic. Per-cell cycles remain unresolved. Once transport readiness is known, each stream cell identifies its chain predecessor; pointer jumping resolves global reach IDs in `ceil(log2(cellCount))` dispatches. Headwaters, confluences and receivers of a branching donor start new IDs. Branch degree counts every positive routing edge, including nonstream receivers; zero-weight edges do not participate. Halo outputs are invalid.

The resident exchange and solver scratch use approximately 60 bytes per global cell plus small loop-control buffers, independent of halo size, in addition to caller-owned tile inputs and outputs. The largest mandatory single binding is the eight-fraction array at 32 bytes per cell, including D8 mode. Graph construction rejects bindings exceeding device limits. This is an explicit resident full-domain strategy; it is not out-of-core hydrology. The default/explicit sweep budget is the same as `GPUTerrainBranchingFlow`.

For preprocessing, compose `GPUTerrainFlow` once over the complete resident elevation domain with explicit `fillDepressions` and/or `resolveFlats`. Pass its directions and streams to tiled consumers; for branching, materialize `GPUTerrainFlowFractions` from its **filled** surface and matching parameters. Tile input views can borrow those whole-domain buffers with `column: 0`, `row: 0`, whole-domain width/height and disjoint owned rectangles. This performs no CPU readback, rerouting, or tile-edge filling. The integration test exercises a depression and flat spanning the ownership seam for D8 and Freeman MFD, and compares every owned accumulation with the existing whole-domain operator. Independently preprocessing each tile is not equivalent and is not the provided strategy.

## Hydrology path classes

`GPUTerrainFlowDistance` and `GPUTerrainStreamNetwork` (in `gpu-terrain/hydrology`) resolve every cell's downstream path by pointer jumping with accumulated distance, not one thread per whole path: `ceil(log2(min(maximumSteps, cellCount)))` gated rounds, each doubling the jump, ping-ponging between two state buffers, and stopping once a round changes nothing. Both expose `estimatedCommandNodeCount` and validate it against the command node budget. Distances are sums in tree order, so they agree with a serial walk to float32 rounding; terminal cells, validity, reach cells and counts are exact. `GPUTerrainStreamNetwork` compacts reaches with `GPUCompaction`, converts per-reach vertex counts to offsets with an inclusive `GPUScan` and counts unresolved cells with `GPUReduction`. When `sourceCells` or `positions` are too small, the reaches that fit in order are kept (a prefix) and `status.overflow` is set. Lengths and distances are in affine units by default or ground meters under `cellSizeMode`. `positions` are cell centers in world coordinates minus the compile-time `localOrigin` of `GPUTerrainStreamNetwork` (default `[0, 0]`, so absolute). Absolute float32 positions are only as fine as float32 at that magnitude: the float32 spacing is 0.5 m at a UTM northing of 4.5e6, so a vertex is up to 0.25 cell off for 1 m cells, and 7.6e-6 degree at longitude 122, about a third of a cell for 1e-5 degree cells. Pass the grid origin `[grid.affine[2], grid.affine[5]]` as `localOrigin` and add it back in float64 (or use it as the layer's coordinate origin) to get exact cell centers; the contributor's `localOrigin` field holds the value used. Both take `directionEncoding` for direction grids that are not ESRI-coded, and read any nonzero `streams` word as a stream cell except the unresolved sentinels. Both treat only the direction and stream sentinels as no-data, not a `validity` view.

## Executed references and scale coverage

`modules/analysis/test/oracles/terrain-upstream.py` executes pinned pysheds 0.5, xDEM 0.1.6, WhiteboxTools 2.4.0 (Python wrapper 2.3.6), NumPy 2.2.6 and SciPy 1.15.3. The adjacent requirements file and checked-in JSON preserve the inputs, reference versions and outputs. It checks supplied D8/MFD accumulation, xDEM vertical shift and NMAD, and a forced Whitebox escape corridor. Catchment statistics are explicitly composed pysheds membership plus independent NumPy reductions. Hierarchy comparisons use independently labeled SciPy sublevel components and their area/volume, so equal-height zero-persistence tree refinements do not produce false topology mismatches.

Whitebox lowers the starting pit by its flat increment in the corridor fixture, while this contributor preserves the pit elevation and minimizes its explicitly stated cut objective. Tests retain and compare both outputs and the common drainage corridor; they do not claim general Whitebox elevation or path-cost identity.

`terrain-upstream-scale.spec.ts` compares D8 and branching accumulation over four owned tiles against the whole-domain operator at 65,536 cells, with deliberately poisoned halo samples. It also covers 16,384-cell forward/reverse chains, 16,384-cell hierarchy/DEM processing, selected-pit breaching and 256-basin/profile reductions on a 65,536-cell domain, and the resident preprocessing composition. Timing messages include compilation, submission, readback and test assertions and are smoke-test evidence, not isolated GPU throughput benchmarks. One headless Chromium run measured 625 ms for whole-domain plus tiled D8, 486 ms for the same branching case, 268 ms for combined 16,384-cell hierarchy/DEM processing, 98 ms for 64-pit breaching, and 361 ms for 256-basin/profile processing. Pipeline caches and CPU assertion costs affect these numbers. Inspect unresolved status on larger or deeper problems; none of these contributors falls back to a CPU algorithm.

## Separate phase measurements

The opt-in `terrain-processing-benchmark.spec.ts` covers all six terrain roadmap rows at 64 and
16,384 cells. Its raw phase record
includes source hashes, device limits, initial allocation/preparation/upload, graph declaration,
synchronous compilation, first submit/fence, three repeated in-place upload/submit samples, full
output readback, native GPU timestamps and actual compiled transient allocation. Run it with
`VITE_TERRAIN_PROCESSING_BENCHMARK=true yarn test-browser-benchmarks --maxWorkers=1 --silent=false --reporter=verbose modules/analysis/test/gpu-terrain/terrain-processing/terrain-processing-benchmark.spec.ts`.

The Apple WebGPU/Metal device measured the following at 16,384 cells. Both timing columns are
medians of three repeated submissions, and scratch excludes caller inputs/outputs and driver
allocations. GPU timestamps separate compute passes and therefore describe the instrumented run.

| Workload | Submit/fence wall time (ms) | GPU node time (ms) | Resident scratch (bytes) |
| --- | ---: | ---: | ---: |
| Bounded breaching, 64 pits | 0.9 | 0.324 | 65,536 |
| Directed-chain transport | 4.4 | 2.959 | 66,844 |
| Nested depression hierarchy, one query stage | 52.2 | 50.610 | 655,872 |
| 128 basin summaries and stream profiles | 2.6 | 0.512 | 853,504 |
| Vertical DEM fit, independent errors | 8.7 | 6.580 | 525,312 |
| D8 accumulation across two owned regions | 2.6 | 0.815 | 786,460 |

At 64 cells, submit/fence medians range from 0.7 to 2.1 ms; the directed-chain CPU reference is
below the browser timer resolution. At 16,384 cells that CPU recurrence takes 0.3 ms, less than
the measured GPU transport submission: these cases establish no GPU speed advantage. The
nested hierarchy exposes the serial activation cost that a monotone single-basin surface can
hide. Repeated samples reupload identical input bytes and reuse the compiled graph; they measure
the update path without changing topology. Full output assertions pass, while upstream and
adversarial tests above establish numerical semantics. These observations supplement the larger
65,536-cell correctness workloads; they do not establish a general device or throughput limit.

## Executable computation lenses

These examples execute the public contributors on small synthetic grids and capture actual GPU results, kernels, and command graphs.

### GPUTerrainBreach

{/* lens:GPUTerrainBreach */}

::lens{scenario=gpu-terrain.terrain-breach rung=result}

::lens{scenario=gpu-terrain.terrain-breach rung=graph}

### GPUTerrainBranchingFlow

{/* lens:GPUTerrainBranchingFlow */}

::lens{scenario=gpu-terrain.terrain-branching-flow rung=result}

::lens{scenario=gpu-terrain.terrain-branching-flow rung=graph}

### GPUTerrainFlowFractions

{/* lens:GPUTerrainFlowFractions */}

::lens{scenario=gpu-terrain.terrain-flow-fractions rung=result}

::lens{scenario=gpu-terrain.terrain-flow-fractions rung=graph}

### GPUTerrainDepressionHierarchy

{/* lens:GPUTerrainDepressionHierarchy */}

::lens{scenario=gpu-terrain.terrain-depression-hierarchy rung=result}

::lens{scenario=gpu-terrain.terrain-depression-hierarchy rung=graph}

### GPUTerrainCatchmentSummary

{/* lens:GPUTerrainCatchmentSummary */}

::lens{scenario=gpu-terrain.terrain-catchment-summary rung=result}

::lens{scenario=gpu-terrain.terrain-catchment-summary rung=graph}

### GPUTerrainDEMCoregistration

{/* lens:GPUTerrainDEMCoregistration */}

::lens{scenario=gpu-terrain.terrain-dem-coregistration rung=result}

::lens{scenario=gpu-terrain.terrain-dem-coregistration rung=graph}

### GPUTerrainTiledFlow

{/* lens:GPUTerrainTiledFlow */}

::lens{scenario=gpu-terrain.terrain-tiled-flow rung=result}

::lens{scenario=gpu-terrain.terrain-tiled-flow rung=graph}
