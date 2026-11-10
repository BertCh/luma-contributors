# Library review: terrain, hydrology and point clouds (2026-11-09)

Scope: `modules/analysis/src/gpu-terrain` (81 files, 29,021 lines) and
`modules/analysis/src/pointcloud-analysis` (15 files, 2,435 lines). Paths below are relative to
`modules/analysis/src` unless they start with `node_modules/` or `dev-docs/`. Read-only review; no
code, tests or builds were run.

## 1. Scope and method

**Read in full** (every line): all 15 `pointcloud-analysis` files; all 10 `terrain-processing` files;
`hydrology/gpu-terrain-stream-network.ts`, `hydrology/gpu-terrain-flow-distance.ts`,
`terrain-flow-field/gpu-terrain-flow-field.ts`; `hydrology/gpu-terrain-flow.ts` lines 1 to 560
(all prose, props and constructor); `hydrology/terrain-flow-tracing.ts` 1 to 140;
`hydrology/terrain-flow-spill-fill.ts` 1 to 130; `terrain-features/gpu-terrain-peak-prominence.ts`
55 to 228; published `node_modules/@luma.gl/graph/src/gpu-pointcloud/*` (3 files) and the head of
`gpu-core/gpu-grid-index.ts`; `raster-analysis/cost-distance/raster-grid-utils.ts` 95 to 215;
`raster-analysis/gpu-raster-watershed-segmentation.ts` 1 to 120; the point-cloud benchmark specs
(`test/gpu-pointcloud/gpu-pointcloud-benchmark.spec.ts`, `...-advanced-benchmark.spec.ts`).

**Skimmed** (class TSDoc, props, `invocationCount`, limits and validation found by grep):
`hydrology/gpu-terrain-watersheds.ts`, `-height-above-drainage.ts`, `-stream-order.ts`,
`-hydrologic-indices.ts`, `terrain-flow-passes.ts`; every class in `terrain-illumination`,
`point-horizon`, `relief-visualization`, `terrain-analysis`, `terrain-curvature`,
`terrain-features`, `topographic-position`, `terrain-decode`, `geomorphons`.

**Not opened**: the WGSL bodies of `terrain-flow-euler-tour.ts`, `terrain-flow-flats.ts`,
`terrain-flow-routing.ts`, `terrain-flow-spill-tree.ts`, `terrain-flow-spill-utils.ts`,
`terrain-flow-unresolved.ts`, `point-horizon-march.ts`, `terrain-horizon-sweep.ts`,
`relief-visualization-utils.ts`, `terrain-summed-area-table.ts`, and most kernels of the
illumination, relief and feature families. Findings about those are from their TSDoc and are
tagged SUSPECTED where they matter.

Census numbers checked: terrain-processing bare asserts 4+8+7+4+4+2+9 = 38 (CONFIRMED); point cloud
9 of 9 class files use bare asserts and none declares `implements GPUCommandNodeProducer`
(CONFIRMED); terrain uses `spatialContext` in 0 files (CONFIRMED).

## 2. Contributor table

Abbreviations. Par: parallelism. Params: `buffer` = per-frame `parameters` view, `baked` = WGSL
literals. Inputs: `band` = `GPURasterBand` (validity, nodata, scale, offset, texture or buffer);
`view` = raw packed `GraphDataView` with NaN as nodata. Ctx: `rg` = raster grid contract
(`GPURasterMetadata` + `cellSizeMode`), `rg-planar` = raster grid but affine units only (no
`cellSizeMode`). Val: `msg` = `throw new Error(message)` / validate helpers, `assert` = bare assert.

### 2a. Hydrology (older generation)

| Symbol | File (lines) | Computes | Par | Work | Iteration | Hard limits | Params | Inputs | Ctx | Status | Val | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `GPUTerrainFlow` | hydrology/gpu-terrain-flow.ts (746) + 8 helper files | fill (Planchon-Darboux fixpoint), BLM flats, D8 dir, D8/D∞/Freeman/Quinn accumulation, streams | row + cooperative (32×32 tile fill) | fill: tiled spill graph, Borůvka MST, Euler tour, pointer jumping, O(n log n); D8 cells: Euler tour + Wyllie ranking O(n log n); weighted/MFD: readiness pull with 1,024-step walks | gated GPU loops; fill relaxation ≤1,024 (default 512); flats ≤1,024 or continuation across frames; pull ≤1,024 (default 64) | spill path needs 8(n+1) B ≤ binding (terrain-flow-spill-fill.ts:130, inherent); Euler tour needs 8n B (terrain-flow-passes.ts:250, inherent); else bounded relaxation (design) | buffer | band | rg | one-row `*Converged`, `*IterationCount`, sentinel words | msg | conforming |
| `GPUTerrainWatersheds`, `GPUTerrainHeightAboveDrainage` | hydrology (267, 310) | basins / nested pour-point watersheds; HAND | row | pointer jumping O(n log L) | gated, default 32 doubling rounds | none beyond binding | none needed | band (HAND) / view (dirs) | none needed (index only) | `converged`, `iterationCount`, sentinel labels | msg | conforming |
| `GPUTerrainStreamOrder` | hydrology/gpu-terrain-stream-order.ts (639) | Strahler order | row + `GPUScan` | Euler-tour levels + scans; pull fallback | gated ≤ log2 n levels | Euler path while 8n B ≤ binding (≈4096², :178), else pull ≤1,024 rounds (design) | none | view | none needed | converged/iterations | msg | conforming |
| `GPUTerrainHydrologicIndices` | hydrology (279) | SCA, TWI, SPI | row | O(n) | none | none | buffer | band | rg | none | msg | conforming |
| `GPUTerrainFlowDistance` | hydrology/gpu-terrain-flow-distance.ts (132) | downstream D8 path length | row, but each cell walks its whole path | O(n·L), divergent | per-cell loop to `maximumSteps` (default n, :60) | none named | baked (affine) | view | rg-planar: affine length only (:111-112) | per-cell validity + terminal sentinel | assert | rework: pointer jumping with distance sums (as in catchment summary) |
| `GPUTerrainStreamNetwork` | hydrology/gpu-terrain-stream-network.ts (185) | reach polylines | row trace + **serial** compaction (one invocation loops all cells, :140-150) | O(n·L) trace + O(n) serial | none | output capacity, overflow flag | baked | view | rg-planar | `overflow`, `unresolvedCount` | assert | rework: scan/`GPUCompaction` for compaction; pointer jumping for reach ends |

### 2b. Terrain processing (newer generation)

| Symbol | File (lines) | Computes | Par | Work | Iteration | Hard limits | Params | Inputs | Ctx | Status | Val | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `GPUTerrainBreach` | terrain-processing/gpu-terrain-breach.ts (266) | least summed-cut breach paths for listed pits, optional burn | one invocation per pit; DP over a (2S+1)² window per step | O(pits·S·(2S+1)²·8), serial per pit | none | scratch pits·(S+1)·(2S+1)²·2 words ≤ 2^32 (:85-88); not checked against the binding (≈15 pits at S=64 with 128 MiB) (design) | baked (decrement, cut depth) | view | none (cut is vertical) | per-pit status, `unresolvedCount` | assert | rework: parallel least-cost relaxation (reuse raster cost-distance) from all pits at once |
| `GPUTerrainFlowFractions` | terrain-processing/gpu-terrain-flow-fractions.ts (138) | 8 fractions + edge lengths per cell from `GPUTerrainFlow` routing | row | O(n) | none | 32 B/cell outputs (inherent) | buffer (reuses flow parameters) | view | rg | NaN rows | assert | align |
| `GPUTerrainBranchingFlow` | terrain-processing/gpu-terrain-branching-flow.ts (256) | transport, expected/min/max distance, fractional catchment over MFD/D∞ fractions | row, readiness DAG pull + downstream walk | O(n·sweeps) | gated sweeps ≤1,024 (default clamp(w+h,128,1024), :78), walk ≤1,024 (default 32) | dependency depth > sweeps·walk stays unresolved (design) | baked (affine edge lengths) | view | rg-planar default (:154-155) | per-cell 0/1/2, `unresolvedCount` | assert | align |
| `GPUTerrainDepressionHierarchy` | terrain-processing/gpu-terrain-depression-hierarchy.ts (288) | sublevel merge tree, ownership, stage area/volume curves | **serial**: `GPUSort`, then one invocation walks all n cells with union-find (:133-185); one invocation per stage scans all cells (:251-282) | O(n α(n)) in one thread; O(stages·n) | none | watchdog (TSDoc :52-54); 16 B/cell tree transient | baked (cell area) | view | rg-planar area (:262) | `[count, requiredCount, overflow]` | assert | rework / merge with `GPUTerrainPeakProminence` (parallel join tree on sorted edges); tracked as TE03 |
| `GPUTerrainCatchmentSummary` | terrain-processing/gpu-terrain-catchment-summary.ts (267) | per-basin area, mean, min, max, stream length, density, hypsometry; stream profiles | one invocation per basin (serial inside basin); profiles row + pointer jumping | statistics O(n_b) per basin; hypsometry O(bins·n_b) per basin (:185-194) | unrolled ⌈log2 n⌉ jump rounds, ungated | one large basin serialises onto one thread (design) | baked (affine) | view | rg-planar ("degrees are not ground meters", :30) | per-cell profile status | assert | rework: segmented reductions (`sorted-segment-sums`) and sort by (basin, bin) |
| `GPUTerrainDEMCoregistration` | terrain-processing/gpu-terrain-dem-coregistration.ts (273) | vertical-shift median, NMAD, bias error, volume change | row + `GPUSort`; reductions are **serial** single invocations over n (:190-212, :246-268) | O(n log n); O(n²) when `correlationLength > 0` (:151-156) | none | O(n²) makes >~10^5 cells impractical (design) | baked | view | rg-planar | fit words | assert | rework: `GPUReduction` / scan totals; covariance by FFT or a range-limited stencil. Horizontal models tracked as TE05 |
| `GPUTerrainTiledFlow` | terrain-processing/gpu-terrain-tiled-flow.ts (138) + terrain-tiled-resident.ts (144) | owned-tile scatter into a resident global DAG | row | as branching flow | as branching flow | allocates full-domain fractions, 32 B/cell (terrain-tiled-resident.ts:23): not out-of-core | baked | view | rg-planar | per-tile status | assert | align (name overstates; TE06 tracked) |

### 2c. Surface analysis, visibility, illumination, features (older generation)

All take a `GPURasterBand`, honour `cellSizeMode` where cell size matters, read per-frame
`parameters`, validate with messages, and declare the producer interface. Status is per-pixel
validity unless noted. Verdict `conforming` unless noted.

| Symbol(s) | Files | Computes | Par | Notable limits / notes |
|---|---|---|---|---|
| `GPUTerrainDerivatives`, `GPUTerrainCurvature` | terrain-analysis, terrain-curvature | Horn slope/aspect/hillshade; Evans-Young, Zevenbergen-Thorne curvatures, ring curvature | row; derivatives build on published `GPURasterGradient` | ring count ≤4 (gpu-terrain-curvature.ts:96) |
| `GPUTerrainRuggedness`, `GPUTerrainVectorRuggedness`, `GPUTerrainTopographicPosition`, `GPUTerrainWeissLandforms`, `GPUGeomorphons` | topographic-position, geomorphons | gdaldem TPI/TRI/roughness, VRM, multiscale TPI/DEV (exact 64-bit summed-area table on `GPUScan`), Weiss classes, GRASS geomorphons | row | TPI scales ≤64 (gpu-terrain-topographic-position.ts:52) |
| `GPUSimpleLocalRelief`, `GPUMultiScaleRelief`, `GPULocalDominance`, `GPUReliefBlend` | relief-visualization | RVT SLRM, MSRM, local dominance, VAT blending | row | blend ≤5 layers (gpu-relief-blend.ts:30); `GPUReliefBlend` is image compositing: `move` candidate |
| `GPUTerrainViewshed`, `GPUTerrainCumulativeViewshed`, `GPUTerrainLineOfSight` | terrain-analysis | R3-style per-target march; cumulative over (observer, pixel) pairs with `atomicAdd`; pair LOS | row | O(n·max(w,h)) per observer (gpu-terrain-viewshed.ts:309); default `traversal: 'march'` (:278): `align` (see G6) |
| `GPUTerrainContours` | terrain-analysis | multi-level contours | wraps published `GPURasterContours` per level | conforming (builds on published) |
| `GPUTerrainHorizon`, `GPUTerrainCastShadow`, `GPUSolarShadowMask`, `GPUSolarIrradiance`, `GPUSolarPosition` | terrain-illumination | horizon angles, SVF, openness, anisotropic SVF; single-sun shadow; insolation | row march or exact hull sweep | ≤64 sectors (gpu-terrain-horizon.ts:159); sweep extent ≤32,767 (terrain-horizon-sweep.ts:27); default `algorithm: 'march'` (:679) although its TSDoc measures the sweep 4 to 36× faster: `align`. `GPUSolarPosition` is per-row point data, not terrain: `move` to geospatial |
| `GPUReliefShading`, `GPUTextureShading` | terrain-illumination | multidirectional hillshade + colour/tint/exposure blend; Brown texture shading | row, separable pyramid | ≤8 lights, ≤8 stops, ≤8 levels; colour output is cartographic rendering: `move` the colour stage |
| `GPUPointHorizonProfile`, `GPUPointHorizonVisibility` | point-horizon | skyline profiles and target visibility from many observers (mt-image ports) | one invocation per ray, min-max pyramid skip | lattice ≤2^24 samples (point-horizon-march.ts:47) |
| `GPUTerrainSummits`, `GPUTerrainCriticalPoints`, `GPUTerrainPeakSnap`, `GPUTerrainPeakProminence`, `GPUProfilePeaks` | terrain-features | disc summits, PL critical points, candidate snap, elder-rule prominence (parallel join tree, `GPUSort`), 1-D profile peaks (`GPUCompaction`) | row; prominence parallel merge tree | radius ≤64 px (terrain-features-utils.ts:9); prominence n ≤ binding/40 (gpu-terrain-peak-prominence.ts:56); `GPUProfilePeaks` is a generic 1-D signal tool: `move` to dataframe/timeseries |
| `GPUTerrainRGBDecode`, `GPUTerrainSpikeRepair` | terrain-decode | Terrarium/Mapbox decode; ±256 m spike repair (published connected components) | row | decode is ingest, borderline `move` to an adapter |
| `GPUTerrainFlowField` | terrain-flow-field (233) | terrain-deflected wind field for particle advection | row | not hydrology despite the name; `move` next to particle advection or rename |

### 2d. Point clouds

| Symbol | File (lines) | Computes (PDAL) | Par | Neighbour search | Iteration | Hard limits | Params | Inputs | Ctx | Status | Val | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `GPUPointCloudNeighborhoods` | gpu-pointcloud-neighborhoods.ts (247) | radius-bounded k nearest (filters.normal support) | row (`serial`, default) or cooperative per query | **default all-pairs O(N²)** (:104); optional private sorted grid (:106-134) | none | capacity ≤64 (:52) | baked (radius, grid) | packed | none | per-row bits 1/2/4 | assert | rework: default to published `GPUGridIndex` + `GPUGridIndexQuery`, require a grid |
| `GPUPointCloudFeatures` | gpu-pointcloud-features.ts (112) | covariance, Jacobi eigen, normals, shape (filters.covariancefeatures) | row | consumes neighbour lists | 24 Jacobi sweeps unrolled | capacity ≤64 (:33) | baked | packed | none | per-row bits | assert | align |
| `GPUPointCloudStatisticalFilter` | gpu-pointcloud-statistical-filter.ts (95) | mean neighbour distance, global μ+kσ mask (filters.outlier statistical) | row + 256-row blocks + **one invocation** combining N/256 blocks (:66-82) | consumes radius-capped neighbours (not true kNN) | none | capacity ≤64 | baked (k) | packed | none | mask only | assert | align (use `GPUReduction`) |
| `GPUPointCloudGround` | gpu-pointcloud-ground.ts (189) | progressive morphological filter (Zhang 2003) | row per raster cell / point | square window O(r²) per cell (:130-139) | radii unrolled ≤32 (:38) | radius ≤32 cells | baked | packed + caller `minimumSurface` | private grid | per-point bits | assert | merge morphology with published `GPURasterMorphology` / SMRF stages |
| `GPUPointCloudSMRF` | gpu-pointcloud-smrf.ts (355) | SMRF (filters.smrf) | row, except **serial** valid count (:172) | private sorted grid; inpaint ring search up to max(w,h) rings per empty cell (:185) | unrolled: Σ(1+r) for r ≤ R kernels (:282-294): ≈190 nodes at the default R = 18, ≈2,100 at R = 64 | window ≤64 cells, cut ≤32 (:66-69) | baked | packed | private grid | per-point bits | assert | rework: van Herk/Gil-Werman or log-step morphology (O(1) or O(log r) per cell), jump-flooding inpaint, `GPUReduction` count |
| `GPUPointCloudICP` | gpu-pointcloud-icp.ts (333) + pointcloud-plane-icp.ts (59) | point-to-point (Horn) and point-to-plane ICP, Huber/Tukey (filters.icp) | row matching; 256-row block moments; one-invocation 4×4/6×6 solve | **default brute force O(N·M)** (:158), capped N·M ≤ 2^32 (:117-118); optional grid | iterations unrolled ≤100 (:65), GPU state gate, but matching still runs after convergence | 2^32 pair tests (design) | baked | packed | local origin helper only | `status` word 0 to 4, history | assert | align once a grid is required |
| `GPUPointCloudChange` | gpu-pointcloud-change.ts (218) | M3C2 cylinders with registration error (filters.m3c2) | one invocation per core | **every core scans every point**, no index (:165) | none | cores·(N1+N2) ≤ 2^32 (:118-121); sampling covariance O(N²) per core, N ≤ 32,768 (:123-124) | baked | packed | none | per-core code | assert | rework: cylinder query through a grid index |
| `GPUPointCloudSurfaces`, `GPUPointCloudHeight` | gpu-pointcloud-surfaces.ts (222) | cell-min DTM, cell-max DSM, CHM; HAG (cell, bilinear, IDW) (filters.hag_dem/hag_nn) | row per cell / per point | private sorted 2-D grid | none | IDW reach ≤64 cells (:141) | baked | packed | private `PointCloudRasterGrid` | per-point code | assert | align |
| `GPUPointCloudVoxelMerge` | gpu-pointcloud-voxel-merge.ts (242) | associative merge of per-part voxel statistics | row per voxel per part | n/a | parts unrolled | none | baked (row offsets) | packed | none | per-voxel bits | assert | align; but no contributor produces its `centroids` input (see C7) |

**Verdict counts** (56 symbols; 46 terrain classes, 10 point-cloud classes): conforming 29,
align 12, rework 9 (flow distance, stream network, breach, depression hierarchy, catchment summary,
DEM coregistration, neighbourhoods, SMRF, M3C2 change), merge 1 (PMF ground into SMRF/published
morphology; depression hierarchy is also a merge candidate but counted under rework), move 5
(relief blend, relief-shading colour stage, solar position, profile peaks, flow field; RGB decode
and spike repair borderline, counted conforming). By generation: the older terrain generation (37
classes) is 29 conforming, 3 align, 5 move, 0 rework; the newer terrain generation
(terrain-processing plus `GPUTerrainFlowDistance` and `GPUTerrainStreamNetwork`, which sit in
`hydrology/` but use the expansion helpers and bare asserts) is 0 conforming, 3 align, 6 rework;
point cloud is 0 conforming, 6 align, 3 rework, 1 merge.

Counter-examples to the two-generation hypothesis: newer `GPUTerrainBranchingFlow` and
`GPUTerrainFlowFractions` are real GPU-scale designs (gated DAG pull; row kernels), and the point
cloud has genuine indexed and cooperative paths, GPU-resident ICP solves and careful float32
handling; its problems are defaults and contract, not the absence of a scalable path. On the
older side, `GPUTerrainHorizon` and `GPUTerrainViewshed` scale but default to their slower
algorithms (G6).

## 3. GPU quality findings

**G1. Serial merge tree in `GPUTerrainDepressionHierarchy`** (CONFIRMED,
terrain-processing/gpu-terrain-depression-hierarchy.ts:130-185). One invocation walks all n
cells with union-find after a GPU sort; stage curves are one invocation per stage over all cells
(:251-282). A 4096² DEM is 16.7M serial iterations on one GPU thread: well past watchdog limits
(SUSPECTED, not measured). `GPUTerrainPeakProminence` already builds the same structure (an
elder-rule join tree) in parallel: radix-sorted spill edges, atomic-min parent splicing, gated
pointer doubling (gpu-terrain-peak-prominence.ts:59-79). A depression hierarchy is the join tree of
the negated surface with explicit outlets. Better design: one shared merge-tree builder used by
both, plus segmented reductions for curves. Effort M. Tracked in part as TE03.

**G2. Path walking instead of pointer jumping** (CONFIRMED). `GPUTerrainFlowDistance`
(hydrology/gpu-terrain-flow-distance.ts:108-118) and the trace in `GPUTerrainStreamNetwork`
(:112-131) walk each cell's downstream path to its end: O(n·L) work with heavy divergence, default
bound `maximumSteps = n`. The library already does this right three times: watersheds and HAND
(`createTerrainFlowPointerNodes`, terrain-flow-tracing.ts:110), and catchment profiles
(gpu-terrain-catchment-summary.ts:217-239, pointer jumping that carries a distance sum). Better
design: one `pointer jump with sum` helper; O(n log L). Effort S.

**G3. Serial reductions and compaction** (CONFIRMED). Single-invocation loops over all cells or
points: stream-network compaction (gpu-terrain-stream-network.ts:140-150), DEM coregistration
counts and volume (gpu-terrain-dem-coregistration.ts:199, :259), SMRF valid count
(gpu-pointcloud-smrf.ts:172), statistical-filter global moments over N/256 blocks
(gpu-pointcloud-statistical-filter.ts:66-82), ICP solves over N/256 blocks (acceptable: small).
Published `GPUReduction`, `GPUScan`, `GPUCompaction` cover all of these. Effort S.

**G4. Per-basin serial statistics** (CONFIRMED, gpu-terrain-catchment-summary.ts:159-195). One
invocation per basin loops over that basin's cells, and hypsometry repeats the loop once per bin:
O(bins·n_b). Real DEMs have one or two dominant basins, so the work lands on one thread. Better:
sort by (basin, bin) and use `sorted-segment-sums`; area and stream length by segmented sums.
Effort S.

**G5. Brute-force defaults in point clouds** (CONFIRMED). Neighbourhoods default to all-pairs
(gpu-pointcloud-neighborhoods.ts:100-104), ICP to N·M matching (gpu-pointcloud-icp.ts:158), M3C2
scans every point for every core with no index option at all (gpu-pointcloud-change.ts:165). The
coverage matrix's one-million-point measurements (dev-docs/coverage-matrix.md:105, :109, :110) were
taken on the non-default indexed paths: the neighbourhood benchmark passes `grid`
(test/gpu-pointcloud/gpu-pointcloud-benchmark.spec.ts:76), ICP passes `targetGrid`
(...-advanced-benchmark.spec.ts:97), and SMRF runs 1M points on a grid capped at 256² with
`window: 4` (:41, :69-70) instead of the default 18-cell window. The SMRF cost driver is the raster
and the window, not the point count. All three benchmarks are opt-in
(`VITE_POINTCLOUD_BENCHMARK`, default maximum 100,000 rows). So: the default path at one million
points is not the measured path. Better: make the grid index mandatory (or auto-sized) and build on
published `GPUGridIndex` + `GPUGridIndexQuery` (node_modules/@luma.gl/graph/src/gpu-core/gpu-grid-index.ts),
which also accepts chunked `GraphVectorView` inputs. Effort M.

**G6. Default algorithms are the slow ones** (CONFIRMED from code and TSDoc). `GPUTerrainHorizon`
defaults to `'march'` (gpu-terrain-horizon.ts:679) while its own TSDoc reports the exact sweep at
98.6 ms against 411 ms (radius 256) and 40.6 ms against 1,474 ms (radius 1,023) on 1024². The
viewshed defaults to a per-target march, O(n·max(w,h)) (gpu-terrain-viewshed.ts:278, :309); an
R2-style ray per perimeter cell (or a radial hull sweep) is O(n) per observer for square tiles.
Effort S for the horizon default, M for an R2 viewshed.

**G7. SMRF morphology unrolled to O(R²) nodes** (CONFIRMED, gpu-pointcloud-smrf.ts:216-294). The
diamond opening at radius r is r repeated cross passes, and the object loop runs openings for every
r ≤ R, each a fresh transient grid. Node count ≈ R²/2 (≈190 at the default, ≈2,100 at the 64-cell
cap) and compile time grows with it. The PMF does square windows directly at O(r²) per cell
(gpu-pointcloud-ground.ts:130-139). Better: erosion of radius r from radius r-1 (already done for
erosion, not for dilation), or van Herk/Gil-Werman for squares; published `GPURasterMorphology`
covers radius ≤8 only, so an upstream large-radius morphology would serve both. Effort M.

**G8. SMRF inpaint ring search** (CONFIRMED, gpu-pointcloud-smrf.ts:185-207). Each empty cell
searches rings out to max(w,h) until eight supported cells are found: worst case O(cells) per
cell on sparse coverage (water, buildings). Jump flooding or the published
`raster-analysis/gpu-raster-fill-nodata.ts` (other domain) is the scalable alternative. Effort S.

**G9. Breach scratch grows with pits·S³** (CONFIRMED, gpu-terrain-breach.ts:82-88, :111-116). The
2^32-word assert ignores the storage binding limit; at S = 64 one pit needs 2.2M words, so a
128 MiB binding holds about 15 pits. The search per pit is serial. Better: one multi-source
least-cost relaxation over the whole DEM (the raster cost-distance relaxation in
`raster-analysis/cost-distance/raster-relaxation.ts` is already used by `GPUTerrainFlow`), as in
Lindsay's whole-DEM least-cost breaching. Effort M.

**G10. O(n²) covariance in DEM coregistration** (CONFIRMED,
gpu-terrain-dem-coregistration.ts:151-156). With `correlationLength > 0` every cell sums over every
other cell. Better: the exponential kernel is separable enough for a truncated stencil, or use FFT
convolution (tracked upstream in section 5 of the roadmap). Effort M.

**G11. Private grid index** (CONFIRMED, pointcloud/pointcloud-grid-index.ts:18-70). Sort by cell key
with published `GPUSort`, then one invocation per cell binary-searches the sorted keys for offsets.
Published `GPUGridIndex` does the same job with capacity, overflow and chunked inputs. Effort S.

**G12. Recompiles** (CONFIRMED). All of terrain-processing and point cloud bake numeric props into
WGSL (radius, thresholds, affine, grid origin, transforms, iteration counts). The older terrain
generation reads a per-frame `parameters` view (for example gpu-terrain-flow.ts:139-158). Changing
an ICP distance or an SMRF slope recompiles. Effort S per contributor.

Precision notes (CONFIRMED, positive): point-cloud local-origin helpers
(pointcloud-local-origin.ts:5-13), compensated stage curves (depression-hierarchy.ts:215-241), exact
64-bit summed-area tables for TPI and relief filters. No float atomics in scope: transport uses a
readiness word and fixed-order pulls (gpu-terrain-branching-flow.ts:135-174).

## 4. Consistency findings

**C1. Two input contracts** (CONFIRMED). 29 terrain classes take `elevation: GPURasterBand` and
canonicalise it with `getRasterBufferBandNodes` (validity, `noDataValue`, scale, offset, textures).
All seven terrain-processing classes plus `GPUTerrainFlowDistance` and `GPUTerrainStreamNetwork`
take raw `GraphDataView<'float32'>` with NaN as the only nodata (for example
gpu-terrain-breach.ts:31, gpu-terrain-depression-hierarchy.ts:31). A DEM with a `noDataValue` of
-9999 works in `GPUTerrainFlow` and silently becomes terrain in `GPUTerrainBreach`. Converge on
`GPURasterBand` everywhere.

**C2. DEM georeferencing** (CONFIRMED). The DEM is `GPURasterMetadata` (published type: affine,
`pixelInterpretation`, optional opaque CRS). Ground cell size comes from a separate
`cellSizeMode: 'uniform' | 'web-mercator' | 'geographic'` prop (analysis-core/raster-grid.ts:24),
evaluated per row in WGSL (raster-grid-utils.ts:142-200). This is the same contract
`raster-analysis` uses (same helpers), so terrain is consistent with raster; it is not
`spatialContext`, and that is acceptable for rasters per brief item 5. Gaps:
- `coordinateReferenceSystem` is never read in `gpu-terrain`; nothing ties the CRS to
  `cellSizeMode`, which defaults to `'uniform'`. A geographic DEM with the default gives slopes in
  metres per degree. A `resolveRasterCellSizeMode(grid)` check (or requiring the prop when a
  geographic CRS is present) would close it.
- Nine classes ignore `cellSizeMode` and measure in affine units: flow distance (:111-112), stream
  network (:162-170), branching flow defaults (:154-155), catchment summary (:30, :101-104),
  depression hierarchy area (:262), DEM coregistration (:116-117, :238-242), breach (vertical only,
  fine), tiled flow (via branching). `GPUTerrainFlow`, `GPUTerrainFlowFractions` and the hydrologic
  indices honour it. Mixed pipelines give metres in one output and degrees in the next.
- Vertical units: no metadata. Older contributors take `zFactor` (66 uses) or
  `verticalExaggeration` (24 uses, relief and flow field); terrain-processing assumes elevation
  units equal horizontal units. Pick `zFactor` everywhere.
- Row direction: only illumination and derivatives take `rowDirection`
  (`'south'` default). Hydrology assumes rows run south when it names D8 codes. The point-cloud
  `PointCloudRasterGrid` puts row 0 at the minimum y, so rows run north
  (gpu-pointcloud-surfaces.ts:31-35). A point-cloud DTM fed to `GPUTerrainFlow` routes correctly,
  but code 4 ("south") then points north. Point-cloud rasters should emit `GPURasterMetadata`.

**C3. Flow encodings** (CONFIRMED). D8 codes agree everywhere they are used: ESRI `1 << d`, east
first, clockwise with rows south; 0 terminal; `0xffffffff` invalid; `0xfffffffe` unresolved
(gpu-terrain-flow.ts:269-275; terrain-flow-tracing.ts:43-54). Two independent D8 tables exist
(array form in terrain-flow-tracing.ts:45-46, bit-packed in raster-grid-utils.ts:164-169); I
checked by hand that they give the same offsets. MFD and D∞ are exchanged as eight fractions per
cell, east first clockwise (gpu-terrain-flow-fractions.ts, gpu-terrain-branching-flow.ts:38); no
D∞ angle raster is produced or accepted anywhere (TauDEM and Whitebox users exchange angles).
`flowDirections` output is always D8 even under D∞ routing (gpu-terrain-flow.ts:351-353).
`terrain-flow-field` has no direction encoding (it is a wind field). The raster watershed files
(`raster-analysis/gpu-raster-watershed-segmentation.ts`, `raster-watershed-forest.ts`) are marker
image segmentation with their own forward-edge order (right, down-left, down, down-right,
raster-watershed-forest.ts:82-88) and share nothing with hydrology; the shared word "watershed" is
the only overlap. Neighbour ring orders also differ: D8 clockwise from east; critical points
counter-clockwise from east; prominence edges E, SE, S, SW. Fine internally, worth one table in
docs.

**C4. No-data and edge rules disagree** (CONFIRMED).
- `GPUTerrainFlow`: grid border cells and cells next to nodata are outlets (open boundary).
- `GPUTerrainDepressionHierarchy`: nodata is a barrier and nothing is an outlet unless listed
  (gpu-terrain-depression-hierarchy.ts:32).
- `GPUTerrainPeakProminence`: closed boundary, nodata barrier.
- `GPUTerrainBranchingFlow`: a positive fraction toward outside the grid invalidates the cell
  (:121); zero rows are outlets.
- Stream masks: producer writes 1 / 0 / `0xffffffff` unresolved. Catchment summary and tiled flow
  test `== 1` and treat `> 1` as error (catchment-summary.ts:255; terrain-tiled-resident.ts:55);
  stream network tests `!= 0` in `incomingCount` (gpu-terrain-stream-network.ts:79-86), so an
  unresolved neighbour counts as a donor (SUSPECTED effect: a missing junction).
Recommendation: one documented boundary policy prop (`'open' | 'closed' | 'explicit-outlets'`) shared
by fill, hierarchy, breach and prominence, and `== 1u` as the only stream test.

**C5. Status shapes** (CONFIRMED). Older hydrology: one-row `converged` and `iterationCount` words
plus sentinel values in data (close to the shared vocabulary, without the named port types).
Terrain-processing: per-cell 0/1/2 codes and an `unresolvedCount`; depression hierarchy
`[stored, required, overflow]` (matches `count/requiredCount/overflow` in meaning, not in type).
Point cloud: per-row bit flags with different meanings per class (neighbourhoods bit 1 invalid, 2
truncated, 4 unsupported; features 1 degenerate, 2 truncated, 4 not converged; SMRF 1, 2, 4, 8;
ICP a scalar 0 to 4). Converge on the named status ports for one-row status and a documented
per-row bit set for point clouds.

**C6. Validation style** (CONFIRMED). Older terrain: `throw new Error(message)` and validate helpers
(e.g. gpu-terrain-flow.ts:466-560). Terrain-processing and point cloud: bare `assert` with the
reason in a comment (AGENTS.md style). Point cloud copies the published luma.gl `gpu-pointcloud`
style exactly (bare asserts, no `implements`, baked literals:
node_modules/@luma.gl/graph/src/gpu-pointcloud/gpu-pointcloud-voxel-downsample.ts:30-41). Bare
asserts cannot name the limit that failed (tracked as AUD-02 for point cloud; terrain-processing is
not tracked).

**C7. Voxel merge consumes data no contributor produces** (CONFIRMED by grep). `centroids` and
attribute means per voxel appear only in `gpu-pointcloud-voxel-merge.ts`; the published
`GPUPointCloudVoxelDownsample` emits representatives, counts and representative positions, not
centroids. Tests build the parts by hand (test/gpu-pointcloud/gpu-pointcloud-advanced.spec.ts:378-400).
The coverage-matrix row PC02.voxel ("indexed centroids", dev-docs/coverage-matrix.md:106) describes a
producer that does not exist in either package.

**C8. PMF and SMRF disagree on inputs** (CONFIRMED). `GPUPointCloudGround` needs a caller-built
`minimumSurface` and `coverage` (gpu-pointcloud-ground.ts:18-19); `GPUPointCloudSMRF` builds its own
minimum from points (:137-159). `GPUPointCloudSurfaces.terrain` is the minimum of *ground-class*
points, not of all points, so it is not the PMF input either.

**C9. Naming** (CONFIRMED). `GPUTerrainFlowField` is a wind field, not hydrology;
`GPUTerrainTiledFlow` is resident, not tiled processing; `GPUTerrainDEMCoregistration` is vertical
shift only; `GPUPointCloudChange` is M3C2. Validity names: `validity`, `outputValidity`,
`validityOutput`, `gradientValidity`, `noData`, `stableMask`, `modificationMask` (grep counts:
10, 21, 2, 4, 3, 1, 1). Recommend `validity` in, `outputValidity` out.

## 5. Duplication and shared-helper candidates

| Duplicate | Files | Recommendation |
|---|---|---|
| Merge tree (join tree) of a scalar field | gpu-terrain-depression-hierarchy.ts (serial), gpu-terrain-peak-prominence.ts + terrain-peak-prominence-tree.ts (parallel) | one parallel builder in `gpu-terrain/terrain-merge-tree.ts`, used by both (G1) |
| Downstream pointer jumping | terrain-flow-tracing.ts:110 (labels), gpu-terrain-catchment-summary.ts:217-239 (sum), terrain-tiled-resident.ts:102-115 (labels), flow distance and stream network (walks) | extend `createTerrainFlowPointerNodes` with an optional additive payload; delete the copies |
| D8 neighbour table | terrain-flow-tracing.ts:43-54, raster-grid-utils.ts:164-176 | keep one (raster-grid-utils) |
| Affine edge length WGSL | branching-flow.ts:154-155, catchment-summary.ts:101-104, flow-distance.ts:111-112, stream-network.ts:169, dem-coregistration.ts:116-117 | replace with `getD8Distance` (honours `cellSizeMode`) |
| `finite` / `missing` / `infinity` WGSL | terrain-processing-utils.ts:40-42, `IS_FINITE_VALUE_WGSL` and `getQuietNaN` in raster-grid-utils | use the analysis-core ones |
| Order-preserving float-to-uint key (`bits ^ 0x80000000` / `~bits`) | gpu-terrain-breach.ts:190, :208; gpu-terrain-depression-hierarchy.ts:120; gpu-terrain-dem-coregistration.ts:177; spill key in prominence | one `getFloatSortKeyWGSL` in analysis-core |
| Sorted uniform grid index | pointcloud-grid-index.ts | published `GPUGridIndex` / `GPUGridIndexQuery` |
| Raster min/max morphology | gpu-pointcloud-ground.ts:115-143, gpu-pointcloud-smrf.ts:216-243 | published `GPURasterMorphology` (radius ≤8 today); a large-radius version belongs upstream |
| Horn normals / slope | gpu-terrain-derivatives.ts (via published `GPURasterGradient`), gpu-relief-shading.ts, gpu-terrain-vector-ruggedness.ts, gpu-terrain-weiss-landforms.ts, SMRF central differences (smrf.ts:309-315), flow-field central differences | derivatives as the single slope source; point-cloud SMRF should reuse it |
| Block moments + single-thread combine | statistical-filter.ts:49-82, icp.ts:203-307, plane-icp.ts | `GPUReduction` for scalars; keep the ICP solve |
| 4×4 / 3×3 Jacobi eigen solve | gpu-pointcloud-features.ts:80-95, gpu-pointcloud-icp.ts:272-286 | one WGSL `symmetricJacobi` helper in analysis-core (other domains have PCA too, SUSPECTED) |

Published pieces in `@luma.gl/graph` and how local code relates (CONFIRMED from
`node_modules/@luma.gl/graph/package.json` exports and `dist/*/index.d.ts`):
- `gpu-pointcloud`: `GPUPointCloudVoxelDownsample` (min-ID representative), `GPUPointCloudRadiusFilter`
  (O(N²) all pairs), `GPUPointCloudOctree`. Local `GPUPointCloudVoxelMerge` builds on the voxel
  layout (but needs a centroid producer, C7); no local contributor duplicates radius filter or
  octree. Nothing local uses the octree, which could serve as the LOD/neighbour structure.
  Upstream candidates: a voxel centroid/attribute-mean reduction, an indexed radius filter (the
  published one is O(N²)), and `GPUPointCloudNeighborhoods` itself (a generic 3-D kNN that
  luma.gl's own pointcloud module lacks).
- `gpu-raster`: terrain builds on `GPURasterContours` (contours), `GPURasterGradient` (derivatives,
  relief shading), `GPURasterConnectedComponents` (spike repair), `GPURasterMetadata`. Terrain does
  not use `GPURasterTileHaloAssembler`/`GPURasterTileCache`, which is the natural base for real
  tiled hydrology (TE06). `GPURasterMorphology` should back PMF/SMRF.
- No terrain or hydrology algorithm is published upstream. Per AGENTS.md they belong here.

## 6. Comprehensiveness

Excludes roadmap items already tracked: TE03 (parallel hierarchy), TE05 (horizontal/Nuth-Kääb/tilt
coregistration), TE06 (out-of-core tiles), HAND-01 (D∞ HAND), X05 (flood simulation), PC01
(global normal orientation), PC02 (distributed thresholds), PC03 (Delaunay/nearest ground
interpolation, vertical datum), PC05 (global alignment, trimmed ICP), PC06 (M3C2 scale selection),
AUD-02.

### Terrain and hydrology (ranked by how often practitioners use them)

| Missing | Reference | GPU-suitable | Reuses | Effort |
|---|---|---|---|---|
| Snap pour points to the stream/accumulation maximum | pysheds `snap_to_mask`, Whitebox `JensonSnapPourPoints`, TauDEM `MoveOutletsToStreams` | yes, per point disc search | `GPUTerrainPeakSnap` disc logic on accumulation | S |
| Automatic whole-DEM breaching (detect pits, breach all, fall back to fill) | Whitebox `BreachDepressionsLeastCost`, RichDEM | yes, multi-source relaxation | `GPUTerrainFlow` `cellClasses` pits + `GPUCompaction` + cost-distance relaxation (G9) | M |
| D∞ angle output / input | TauDEM `DinfFlowDir`, Whitebox `DInfPointer`, RichDEM | yes, row | flow-routing facet code | S |
| Shreve, Horton, Hack stream orders; stream link IDs; subbasins at junctions | Whitebox `ShreveStreamMagnitude`, `HortonStreamOrder`, `StreamLinkIdentifier`, `Subbasins`, GRASS `r.stream.order` | yes | Euler tour / pull of `GPUTerrainStreamOrder`; reach IDs exist in tiled flow | S |
| Upslope flow-path length (longest flow path), upslope mean elevation | Whitebox `MaxUpslopeFlowpathLength`, `AverageUpslopeFlowpathLength`, GRASS `r.flow` | yes | weighted accumulation pull with max/mean combiners (branching flow has max-distance downstream only) | S |
| LS factor / RUSLE terms, MRVBF valley bottoms, downslope index | SAGA `LS Factor`, `MRVBF`, Whitebox `SedimentTransportIndex`, `DownslopeIndex` | yes, row / multiscale | accumulation, slope, summed-area tables | S each |
| Depth in sink, sink labels, impoundment size | Whitebox `DepthInSink`, `Sink`, `ImpoundmentSizeIndex` | yes | filled − original (trivial); labels via watersheds on pits | S |
| Zevenbergen-Thorne slope/aspect option in derivatives | `gdaldem -alg ZevenbergenThorne`, xarray-spatial | yes | curvature already has the stencil | S |
| Visibility index (viewshed from every cell) | Whitebox `VisibilityIndex`, GRASS `r.viewshed` cumulative | yes but O(n²): needs sampled observers | cumulative viewshed + R2 viewshed (G6) | M |
| Hypsometric integral and elevation-relief ratio per window | Whitebox, SAGA | yes | summed-area tables | S |

### Point clouds (ranked)

| Missing | Reference | GPU-suitable | Reuses | Effort |
|---|---|---|---|---|
| Voxel centroid / attribute-mean downsample (the producer `GPUPointCloudVoxelMerge` expects) | PDAL `filters.voxelcentroidnearestneighbor`, Open3D `voxel_down_sample`, PCL `VoxelGrid` | yes | published voxel layout + sort by voxel + `sorted-segment-sums` | S |
| True k-nearest neighbours without a radius cap; k > 64 | PDAL `filters.outlier` (statistical), PCL `KdTreeFLANN`, Open3D | yes, grid with expanding rings | `GPUGridIndexQuery` | M |
| Cloth simulation ground filter | PDAL `filters.csf`, CloudCompare CSF, lidR `csf` | yes, grid particles | point-cloud raster minimum, gated iterations | M |
| Euclidean clustering / DBSCAN on 3-D points | PDAL `filters.cluster`, Open3D `cluster_dbscan`, PCL | yes | `gpu-spatial-analysis/spatial-clustering` (2-D, SUSPECTED) + grid index | S to M |
| RANSAC plane segmentation | Open3D `segment_plane`, PCL `SACSegmentation`, PDAL `filters.planefit` | yes, parallel hypotheses | Philox stream, features normals | M |
| Gridding with mean/count/IDW/density (writers.gdal output types) | PDAL `writers.gdal`, lidR `grid_metrics`, `rasterize_density` | yes | sorted 2-D grid in surfaces | S |
| Farthest-point / Poisson-disk sampling | PDAL `filters.fps`, `filters.sample` | yes (iterative) | grid index | M |
| Tree-top detection and crown segmentation on CHM | lidR `locate_trees`, `segment_trees` | yes | `GPUTerrainSummits` on CHM, watershed segmentation | S (composition recipe) |
| FPFH descriptors for global registration | PCL, Open3D | yes | features + neighbourhoods | M (feeds PC05) |
| Extended local minimum (low-noise) filter | PDAL `filters.elm` | yes | SMRF cell minima | S |

### Out of place in a GPU analysis library

- `GPUReliefShading` colour stage (ramp, tint, exposure, RGB output) and `GPUReliefBlend`: image
  compositing. Keep scalar hillshade/SVF outputs here; move colour composition to the deck/kepler
  adapter or luma.gl `gpu-effects`.
- `GPUTerrainFlowField`: wind field for particle advection; belongs with `raster-analysis` flow
  texture, or rename.
- `GPUSolarPosition`: per-row point attribute, belongs in `geospatial-analysis`.
- `GPUProfilePeaks`: generic 1-D signal peaks; belongs in dataframe/timeseries.
- `GPUTerrainRGBDecode`, `GPUTerrainSpikeRepair`: tile ingest. Borderline: keep, but they are
  adapter work by the X01 rule.

**Question 5 (illumination).** Horizon angles, sky-view factor, openness, anisotropic SVF, cast
shadows and insolation are terrain analysis: GRASS `r.horizon`, `r.sun`, `r.skyview`, SAGA, RVT and
Whitebox all ship them, and their outputs are physical rasters (degrees, Wh/m²). They belong here,
not upstream (luma.gl carries no domain algorithms). Texture shading and the RVT visualisations
(SLRM, MSRM, local dominance) are visualisation products but are used as analysis rasters in
archaeology; keep them. Only the colour composition stages are rendering (above). Their engineering
is good (exact hull sweep, mip-max skipping, f32 precision ladders); the one fix is the slow
default (G6).

## 7. Top ten actions

| # | Action | Effort | Files |
|---|---|---|---|
| 1 | Make point-cloud neighbourhoods, ICP and M3C2 indexed by default on published `GPUGridIndex` + `GPUGridIndexQuery`; delete the private grid index; re-run the one-million benchmarks on the default path and correct the coverage-matrix wording | M | pointcloud-grid-index.ts, gpu-pointcloud-neighborhoods.ts, -icp.ts, -change.ts, -surfaces.ts, -smrf.ts, dev-docs/coverage-matrix.md:105-111 |
| 2 | Parallel depression hierarchy by sharing the prominence join tree (TE03) | M | gpu-terrain-depression-hierarchy.ts, gpu-terrain-peak-prominence.ts, terrain-peak-prominence-tree.ts |
| 3 | Replace path walks with pointer jumping carrying sums; replace serial compaction/reductions with `GPUScan`/`GPUCompaction`/`GPUReduction` | S | gpu-terrain-flow-distance.ts, gpu-terrain-stream-network.ts, gpu-terrain-dem-coregistration.ts, gpu-terrain-catchment-summary.ts, gpu-pointcloud-statistical-filter.ts, gpu-pointcloud-smrf.ts |
| 4 | One input contract for terrain-processing: `GPURasterBand` elevation, `cellSizeMode`, `getD8Distance`, per-frame `parameters`, validate helpers with messages instead of 38 bare asserts | M | all of terrain-processing, flow distance, stream network |
| 5 | Shared boundary policy and stream test (`== 1u`) across fill, hierarchy, breach, prominence, stream network | S | gpu-terrain-flow.ts docs, depression-hierarchy.ts, breach.ts, peak-prominence.ts, stream-network.ts |
| 6 | Add the voxel centroid/attribute-mean producer (or remove the centroid claim) so voxel merge has an input; propose it upstream next to `GPUPointCloudVoxelDownsample` | S | new pointcloud file or luma.gl PR; gpu-pointcloud-voxel-merge.ts |
| 7 | Rework SMRF/PMF morphology (incremental dilation, large-radius morphology upstream), jump-flooding inpaint, take `GPURasterMetadata` and emit north-up rasters | M | gpu-pointcloud-smrf.ts, gpu-pointcloud-ground.ts, gpu-pointcloud-surfaces.ts |
| 8 | Whole-DEM least-cost breaching on the raster relaxation instead of per-pit serial windows; check scratch against the binding size | M | gpu-terrain-breach.ts, raster-analysis/cost-distance/raster-relaxation.ts |
| 9 | Point-cloud contract: `implements GPUCommandNodeProducer`, validate helpers (AUD-02), parameter buffers for thresholds/transforms, named status words; decide with upstream whether luma.gl `gpu-pointcloud` adopts the same contract | M | all 9 point-cloud class files |
| 10 | Defaults and names: horizon `algorithm: 'sweep'`, an R2 viewshed traversal, tie `cellSizeMode` to `coordinateReferenceSystem`, rename `GPUTerrainFlowField`/`GPUTerrainTiledFlow`, move colour/solar-position/profile-peaks | S to M | gpu-terrain-horizon.ts:679, gpu-terrain-viewshed.ts, analysis-core/raster-grid.ts, terrain-flow-field, terrain-processing/gpu-terrain-tiled-flow.ts |

### Answer summary for question 1 (largest DEM accepted)

Every per-cell buffer must fit one storage binding (`maxStorageBufferBindingSize`, 128 MiB by
default, often 1 to 4 GiB on desktop). Older hydrology checks this and falls back to bounded
relaxations (terrain-flow-spill-fill.ts:130, terrain-flow-passes.ts:250,
gpu-terrain-stream-order.ts:178): about 16.7M cells (4096²) on the exact paths at 128 MiB.
Terrain-processing does not check bindings at all; its largest transient is 32 B per cell
(fractions in flow fractions and tiled flow, terrain-tiled-resident.ts:23), so about 4.2M cells
(2048²) at 128 MiB before WebGPU rejects the binding, and depression hierarchy and DEM coregistration
hit serial-time limits far earlier (SUSPECTED, not measured). Tested sizes per the coverage matrix
are 65,536 cells for breach, branching, catchment and tiled flow, and 1,024 cells for the
correlated coregistration (dev-docs/coverage-matrix.md:94-99).
