# Library review: cells, indexing, joins, zones and density (2026-11-09)

Domain: discrete global grid cells, spatial indexing and joins, aggregation to zones, density
surfaces. Paths are relative to `modules/analysis/src`. Builds on
`dev-docs/research/spatiotemporal-review-2026-11-09/code-architecture.md` section B (index table)
and does not repeat it. Read-only review; nothing was built or run.

## 1. Scope and method

116 files in 19 directories of `gpu-spatial-analysis` plus `analysis-core/cell-key-wgsl.ts` (69)
and `analysis-core/spherical-cell-area.ts` (163): about 40,600 lines, 44 exported contributor
classes, one non-producer class (`GPURegionStatisticsReadback`), no `add*Recipe` in scope.

| Directory | Lines | Read in full | Skimmed (docs, props, kernels by grep) |
| --- | --- | --- | --- |
| cell-topology | 3,157 | all four `gpu-s2-*.ts`, `s2-analysis-wgsl.ts`, `gpu-h3-cell-identities.ts` (constructor, kernel head) | `gpu-cell-topology.ts`, `gpu-cell-compaction.ts`, `gpu-cell-grid-path.ts` (docs, limits) |
| cell-aggregation | 2,032 | `gpu-cell-aggregation.ts` | `cell-table.ts` (types, node list), rollup, pyramid, level selection |
| cell-indexing, cell-cover | 4,793 | | `gpu-point-to-cell.ts`, `gpu-cell-measures.ts`, `gpu-cell-geometry.ts`, `cell-index-families.ts`, `h3-boundary-wgsl.ts` header, `gpu-cell-cover.ts` doc and limits |
| cell-set-outline, cell-table-compare | 1,327 | | single-thread nodes and limits only |
| hilbert-keys | 539 | | `gpu-hilbert-keys.ts` doc and WGSL, `bounds-reduction.ts` doc |
| spatial-join | 11,323 | `spatial-join-passes.ts:705-977` (probe, scan, write) | PIP, predicate, nearest join docs and overflow paths; candidates, buffer selection, nearest weights, capacity plan |
| neighbor-search, neighborhood-summary, pair-gather, bounds-filter | 3,887 | | docs, limits, primitive use |
| zonal-statistics, region-statistics | 3,809 | | docs, sum strategy, status |
| areal-interpolation | 2,118 | | docs; pycnophylactic iteration and one-thread nodes |
| point-density, line-density, dot-density | 6,703 | | KDE doc, rank-in-cell kernel `gpu-kernel-density.ts:758-795`, point density options, line density doc and parameters, dot density doc |
| similar-locations | 718 | | doc, one-thread nodes |
| analysis-core files | 232 | `cell-key-wgsl.ts` | `spherical-cell-area.ts` exports |

Not opened: `cell-cover-scanline.ts` body, `cell-cover-kernels.ts` body, `h3-index-*.ts` bodies,
`h3-local-ij-wgsl.ts`, `h3-neighbor-*`, `nearest-wgsl.ts` body, `spatial-predicate-wgsl.ts`,
`spatial-relate-wgsl.ts`, `zonal-statistics-passes.ts` body, `region-statistics-grid.ts` body,
`kernel-density-binned.ts` body, `dot-density-sampling.ts` body. Also read the props of
`raster-analysis/raster-zonal-statistics/gpu-raster-zonal-statistics.ts:60-338`, the upstream
`@luma.gl/shaders` `dggs` function list, and `census.json` rows.

Census correction: the census counts 1 named status-port file in scope. In fact 19 files use the
shared vocabulary (`GPUCardinalityPort`, `GPUOverflowStatusPort`, `GPUCompactOutput`,
`GPUCompactPairPort`, `GPUCellTablePort`), which the census regex missed because they are
re-exported from `gpu-contributor-types`. Most census `serialKernel` hits in scope are one-thread
finalize nodes (scan totals, cursors), not serial work; only the two S2 coverings are real serial
kernels. `bounds-filter` and `pair-gather` use published `GPUVisibilityWorkflow` for compaction,
which the census list does not include.

## 2. Contributor table

Abbreviations: ctx = `spatialContext`; DG = discrete-grid keys (no context needed); helpers =
`validate*` / `throw new Error` with messages; shared = shared status vocabulary.

| Symbol | File (lines) | Computes (reference) | Par. | Work | Iter. | Hard limits | Params | Inputs | Context | Status | Validation | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPUPointToCell | cell-indexing/gpu-point-to-cell.ts (256) | lon/lat to Quadbin, H3, quadkey, geohash, S2 key (h3 `latLngToCell`) | row | O(n) | none | f32 input; H3 exact only with `precision:'extended'`; S2 exact to level ~20 (doc :101-107) inherent | baked: family, resolution | packed | ctx | per-row `validity` | helpers | conforming |
| GPUCellGeometry | cell-indexing/gpu-cell-geometry.ts (452) | key to center, boundary (h3 `cellToBoundary`) for 6 families incl. A5 | row | O(n) | none | ≤16 vertices (:55) | baked | packed | DG | none (zero rows) | helpers | conforming |
| GPUCellMeasures | cell-indexing/gpu-cell-measures.ts (275) | H3 area, perimeter, edge length, pentagon (h3 `cellAreaM2`) | row | O(n) | none | H3 only (:131); pentagons unreliable > res 5 (doc) | baked units | packed | DG | none | helpers | merge (with GPUS2CellMeasures) |
| GPUS2CellMeasures | cell-topology/gpu-s2-cell-measures.ts (72) | S2 area/perimeter in steradians/radians | row | O(n) | none | none | none | packed | DG | per-row code 0/1 | bare assert (:38) | merge (into GPUCellMeasures) |
| GPUCellAggregation | cell-aggregation/gpu-cell-aggregation.ts (309) + cell-table.ts (678) | points or keys to sorted cell table, count/sum/min/max (CARTO `QUADBIN_FROMLONGLAT` + GROUP BY) | row | O(n) radix sort | none | positions Quadbin only (:147); 64-bit fixed sum, scale 2^16 (cell-key-wgsl.ts:9) | baked: resolution, sumScale | packed | ctx | shared (count, overflow, requiredCount) | helpers | align (fixed-point scheme, see 4.3) |
| GPUCellRollup, GPUCellPyramid, GPUCellLevelSelection | cell-aggregation (200, 215, 161) | coarser levels of a cell table; per-frame level pick | row | O(capacity) | none | Quadbin/H3 | level index buffer | packed | ctx | shared | helpers | conforming |
| GPUCellCover | cell-cover/gpu-cell-cover.ts (597) + kernels (1,611) | polygon to Quadbin/H3 cells, center/full/intersects (h3 `polygonToCells`) | row per candidate | O(candidates), scanline row intervals, edge slabs | none | H3 lattice clamped 89° (:225-243); slab entries ≤2^24 | baked: family, resolution | packed | ctx | shared incl. `candidateOverflow`, `uncertainCount`, `invalidCount` | helpers | conforming |
| GPUS2PolygonCovering | cell-topology/gpu-s2-polygon-covering.ts (204) | adaptive S2 cover of ONE polygon (S2RegionCoverer) | serial: `invocationCount: 1` (:107) | O(visited × V) in one thread, plus O(V²) ring check (:177-189) | DFS loop | maxLevel ≤15 (:63), visited ≤1,048,576 (:80), stack 52 (:191) design | baked: tolerance, levels, budget, hemisphere center | packed | none (unit xyz) | one word bitmask 1/2/4 | bare assert ×7 | rework |
| GPUS2CapCovering | cell-topology/gpu-s2-cap-covering.ts (103) | uniform-level S2 cells within caps | row per cap, each loops all 6·4^level cells (:87) | O(caps × 6·4^L) brute force | none | level ≤8 (:47) design | baked: level, capacity, tolerance | packed | none | per-cap code 0/1/2 | bare assert ×4 | rework |
| GPUS2CellHierarchy | cell-topology/gpu-s2-cell-hierarchy.ts (148) | S2 parent, 4 immediate children, leaf range, 4 edge neighbors | row | O(n) | none | children: one level only | baked: operation, level | packed | DG | per-row code | bare assert ×2 | merge (into GPUCellTopology) |
| GPUCellTopology | cell-topology/gpu-cell-topology.ts (336) | gridDisk, gridRing, parent, children (Quadbin, H3) | row (children: row per output slot) | O(n·stride), private per-thread sort | none | k ≤8 (:30), children stride ≤4096 (:33) design | baked: k, resolution | packed | DG | per-row counts | helpers | align (k limit) |
| GPUCellGridPath | cell-topology/gpu-cell-grid-path.ts (346) | H3 gridDistance, gridPathCells | row | O(n·L) | none | H3 only (:128); path ≤256 (:32) | baked | packed | DG | per-row true length, no flag | helpers | align |
| GPUCellCompaction | cell-topology/gpu-cell-compaction.ts (813) | compactCells / uncompactCells (Quadbin, H3) | row | O(n log n) sort + binary search | none | uncompact depth default 8 (:34), H3 subtree 11 (:108) | baked | packed | DG | shared + `droppedCount` (:83) | helpers | align (`droppedCount` name) |
| GPUH3CellIdentities | cell-topology/gpu-h3-cell-identities.ts (143) | H3 directed edges, vertices of a cell | row | O(n) | none | 6 slots | baked | packed | DG | none (zeros) | one throw without id prefix (:41-42) | align |
| GPUCellSetOutline | cell-set-outline (612) | outline of a cell set (h3 `cellsToMultiPolygon`) | row + ring assembly | O(n·edges) | none | H3/Quadbin | baked | packed | ctx | shared | helpers | conforming |
| GPUCellTableCompare | cell-table-compare (685) | two-period cell table diff, z-scores | row | O(capacity) | none | partial groups ≤256 (:39) | baked | packed | DG | shared | helpers | conforming |
| GPUHilbertKeys | hilbert-keys/gpu-hilbert-keys.ts (361) | 2D Hilbert keys and order | row | O(n) + radix sort | none | order ≤16 (:41) inherent to u32 | baked order | packed | ctx | none | helpers | merge (joins carry a second Hilbert, 5.1) |
| GPUSpatialJoinPrepared | spatial-join/gpu-spatial-join-prepared.ts (442) | cached right-hand BVH and edge slabs | row | O(n log n) | none | power-of-two leaves | n/a | packed | n/a | BVH overflow | helpers | conforming |
| GPUSpatialJoinCandidates | spatial-join/gpu-spatial-join-candidates.ts (390) | bbox candidate pairs (PostGIS `&&`) | row | BVH | none | capacity ≤ binding | baked capacity | chunked | ctx | `GPUCompactPairPort` | helpers | conforming |
| GPUPointInPolygonJoin | spatial-join/gpu-point-in-polygon-join.ts (913) | points to containing polygon (`sjoin within`, cuSpatial PIP) | row; count-scan-write | BVH + exact refine; edge slabs | none | `candidateCapacity` ≤ 2^31-2 (:369) | baked capacity, rules | chunked points | ctx | `overflow` folds BVH+candidate+matches (:136) | helpers | align (status split) |
| GPUSpatialPredicateJoin | spatial-join/gpu-spatial-predicate-join.ts (1,850) | DE-9IM predicate join, inner and anti (`sjoin`, `ST_Relate`) | row per candidate | BVH + exact | none | candidate and pair capacities | per-frame distance/pattern as views | chunked | ctx | `overflow`, `candidateCount`, `uncertainCount` | helpers | align (status split; uncertainCount not tracked in two kernels, doc :231-233) |
| GPUNearestFeatureJoin | spatial-join/gpu-nearest-feature-join.ts (1,265) | nearest and k-nearest features (`sjoin_nearest`, PostGIS `<->`) | row; stack BVH branch-and-bound | O(n log m) | none | `neighborCapacity` ≤64 (:534) design | baked k | chunked | ctx | `overflow` | helpers | align (second KNN, 5.2) |
| GPUNearestFeatureWeights | spatial-join/gpu-nearest-feature-weights.ts (358) | KNN join to CSR weights (libpysal `KNN`) | row | O(n·k) | none | k | baked | packed | ctx | shared | helpers | conforming |
| GPUBufferSelection | spatial-join/gpu-buffer-selection.ts (368) | features within distance of a selection | wraps predicate join | BVH | none | capacities | per-frame distance | packed | ctx | shared | helpers | conforming |
| GPUPairGather | pair-gather/gpu-pair-gather.ts (574) | attribute side of `sjoin` / `sjoin_nearest`, how inner/left | row + `GPUVisibilityWorkflow` | O(pairs) | none | output capacity | n/a | packed | none (justified) | shared | helpers | conforming |
| GPUBoundsFilter | bounds-filter/gpu-bounds-filter.ts (278) | features whose bbox relates to a box (`gdf.cx`) | row | O(n) per box (optimal for one box) | none | capacity | `GPUParameterBuffer` | packed | ctx | `GPUCompactOutput` | helpers | conforming |
| GPUNeighborSearch | neighbor-search/gpu-neighbor-search.ts (1,252) | KNN and distance band to CSR (libpysal `KNN`, `DistanceBand`) | row | grid (default) or BVH; radius rows insertion-sorted (:260) | none | k ≤32 (:87); BVH planar only (:297); grid ≤2^25-1 cells | per-frame parameters view | packed | ctx | shared (overflow, requiredCount, invalidCount) | helpers | align (PLAT-03) |
| GPUNeighborhoodSummary | neighborhood-summary (659) | lag family over weights (momepy `describe`) | row | O(Σk), categories O(k²) per row | none | median ≤64 members (:36) | baked: statistics, focalWeight | packed | none (weights input) | `GPUBoundedResultStatusPort.overflow` | helpers | conforming |
| GPUZonalStatistics | zonal-statistics (1,642) | points-in-polygons count, sum, mean, min, max, variance, density (choropleth) | row | PIP + sort + segmented sum | none | via PIP capacity | baked | chunked | ctx | `overflow`, `uncertainCount` | helpers | align (naming vs raster zonal, 4.2) |
| GPURegionStatistics | region-statistics/gpu-region-statistics.ts (701) | one interactive region: count, sum, min, max, histogram, selected IDs | row | O(n) or grid gather | none | histogram ≤65,536 bins (:65) | per-frame region | chunked values | ctx | packed `summary` words, `candidatesTruncated` flag | helpers | align (ad hoc summary status) |
| GPURegionMask, GPUPickRegionMask | region-statistics (405, 124) | lasso/rectangle and pick-buffer masks | row | O(n·V) / O(picks) | none | lasso capacity | per-frame region | packed | ctx / none | `overflow` | helpers | conforming |
| GPURegionStatisticsReadback | region-statistics/region-statistics-readback.ts (110) | maps the summary to JS | cpu | readback | n/a | n/a | n/a | n/a | n/a | n/a | n/a | move (application helper, not a contributor) |
| GPUArealInterpolation | areal-interpolation/gpu-areal-interpolation.ts (1,031) | tobler `area_interpolate` on a raster | row | O(cells) radix sorts | none | areas exact to 2^24 cells per zone | baked | packed | ctx | `overflow`, `invalidCount` | helpers | align (raster approximation only, 6) |
| GPUPycnophylactic | areal-interpolation/gpu-pycnophylactic.ts (878) | Tobler 1979 pycnophylactic surface | row | O(cells × iterations) | unrolled ≤512 (:58) with GPU convergence gate | 512 iterations design | per-frame tolerance | packed | ctx | `converged`, `iterationCount`, `invalidCount` | helpers | conforming |
| GPUKernelDensity | point-density/gpu-kernel-density.ts (1,137) | planar KDE with edge correction (spatstat `density.ppp`, ArcGIS Kernel Density) | row per output cell | grid gather O((5h/cell)² per point) or binned blur | none | binned cells limit | per-frame parameters view | packed | ctx | none for index overflow (see 3.4) | helpers | align (rank-in-cell, 3.3) |
| GPUPointDensity | point-density/gpu-point-density.ts (812) | grid or hexagon binning count/sum/mean + convolution (deck.gl `GridLayer`, d3-hexbin) | row | O(n) | none | fixed-point table ≤ binding/12 | `GPUParameterBuffer` | packed | ctx | ad hoc overflow flag | helpers | conforming |
| GPULineDensity | line-density/gpu-line-density.ts (393) | line length per grid cell / cell area | row per piece | O(pieces) DDA + sort | none | `recordCapacity` | `GPUParameterBuffer` | packed | ctx | `overflow` | helpers | align (no search radius, 6) |
| GPULineLengthPerPolygon | line-density/gpu-line-length-per-polygon.ts (688) | line length per polygon (QGIS Sum line lengths) | row per (segment, polygon) | BVH candidates, O(V) per pair | none | crossings ≤1,024 per pair (:143) | baked | packed | ctx | `overflow` | helpers | conforming |
| GPUDotDensity, GPURandomPointsInPolygon, GPURandomPointsOnLine | dot-density (157, 119, 371) | dasymetric dots; uniform points in polygons and on lines | row | O(dots) + triangulation | none | capacity ≤8,388,608 | per-frame parameters | packed | ctx (planar only) | `failedCount`, shared | helpers | conforming |
| GPUSimilarLocations | similar-locations (631) | attribute-space ranking (CARTO `FIND_SIMILAR_LOCATIONS`) | row | O(n·c) + sort | none | values ≤2^25 elements | per-frame parameters | packed | none (attribute space) | `GPUCompactOutput` | helpers | conforming |

Verdict counts (45 classes, grouped siblings counted per class): conforming 24, align 14,
merge 4, rework 2, move 1.

Counter-examples to the two-generation hypothesis:

- The S2 files are short, terse and use bare asserts, and two of them are serial (expansion
  pattern). `GPUS2CellHierarchy` and `GPUS2CellMeasures` are short but row-parallel and exact; they
  scale. Their problem is duplication of the H3/Quadbin API, not scale.
- `GPUH3CellIdentities` (143 lines, one-line kernels) is row-parallel and scales.
- Baseline `GPUCellTopology` has a design limit of k ≤ 8 and per-thread enumerate-and-sort.
- Baseline `GPUKernelDensity` and IDW carry an O(m²)-per-cell rank pass (3.3).

## 3. GPU quality findings

**3.1 `GPUS2PolygonCovering` runs on one GPU thread. CONFIRMED.** `invocationCount: 1`
(`cell-topology/gpu-s2-polygon-covering.ts:107`). One invocation validates every ring with an
O(V²) pairwise crossing test (`:177-189`), then walks a depth-first stack of at most 52 entries
(`:191-197`) over up to 1,048,576 cells (`:80`); each visited cell runs `coverClassify`, which loops
over every polygon vertex five times (`:151-168`). One polygon per contributor. A 1,000-vertex
polygon at level 12 is on the order of 10^9 serial operations; the WebGPU watchdog will stop it
first. Better: level-synchronous frontier expansion (per level, one invocation per (frontier cell,
polygon), count, scan, write the children of cells that straddle the boundary), as `GPUCellCover`
already does for H3/Quadbin with count, scan and write; per-polygon validation as a separate
row-parallel pass over edges with the edge-slab index. Effort M.

**3.2 `GPUS2CapCovering` enumerates the whole sphere per cap. CONFIRMED.** One thread per cap
loops over all `6 * 4^level` cells (`gpu-s2-cap-covering.ts:87-95`), so level is capped at 8
(`:47`): 393,216 cells per thread, and level 8 cells are still about 40 km across. Better: compute the cap's face-uv
bounding box, enumerate only those `ij` ranges with one invocation per (cap, candidate), count,
scan, write. Same pipeline as `GPUCellCover`. Effort S.

**3.3 Rank-in-cell pass after `GPUGridIndex` is O(m²) per occupied cell and is copied three
times. CONFIRMED.** `gpu-kernel-density.ts:758-795`: each point finds its cell by binary search,
then counts the IDs smaller than itself by a loop over the whole cell. The same kernel, with the
same comment, is in `spatial-interpolation/gpu-inverse-distance-weighting.ts:290` and
`spatial-regression/gwr-grid-index.ts`. On clustered data (a city of 10^5 points in one index cell)
this is 10^10 comparisons. Better: one stable radix sort of `(cell, row)` with published `GPUSort`
(or `GPUSegmentedSort`, public and unused per review B4), or ask upstream for a stable
`GPUGridIndex` option. Effort S (shared helper), touches three files outside this domain.

**3.4 Join candidate overflow is single-shot and silent at the row level. CONFIRMED.** Probe,
scan and write in `spatial-join-passes.ts:714-977` write candidates in point-row order; slots past
`candidateCapacity` are dropped (`:962-967`), so the highest point rows lose candidates and get
`GPU_SPATIAL_JOIN_NO_FEATURE`. Only a one-row `overflow` flag says so. This is deterministic and
the right GPU shape, but no join exposes which rows were affected. Per join:

| Join | Capacity stages | On overflow | Recovery |
| --- | --- | --- | --- |
| `GPUPointInPolygonJoin` | BVH leaves, candidates, `matches` | tail rows unassigned; `featureCounts` undercount; one `overflow` flag (`gpu-point-in-polygon-join.ts:136`) | optional `candidateCount`; caller rebuilds |
| `GPUSpatialPredicateJoin` | BVH, candidates, pairs | inner: missing pairs; anti: false unmatched rows, "unsafe direction" (`:357`) | `getCapacityPlan` host grow loop (`:484`) |
| `GPUSpatialJoinCandidates` | BVH, pairs | `GPUCompactPairPort` count clamp, `requiredCount` | caller |
| `GPUNearestFeatureJoin` | nearest mode: candidates in radius box; neighbors mode: `neighborCapacity` ≤64 | nearest mode as PIP; neighbors mode only tie truncation | none needed in neighbors mode (exact traversal) |
| `GPUBufferSelection`, `GPULineLengthPerPolygon`, `GPUZonalStatistics` | inherit predicate join / candidates / PIP | flags ORed into own `overflow` | caller |
| `GPUNeighborSearch` | CSR slots | one row cut, offsets clamped, `requiredCount` exact | caller |

Better: emit a per-row "truncated" bit or a first-truncated-row scalar, split `candidateOverflow`
from `overflow` (the vocabulary already has it, and `GPUCellCover` uses it), and for PIP prefer a
count-then-allocate pass whose capacity is the scanned total (a two-phase allocation with a GPU
check, as `GPUCellCompaction` uncompact does with `requiredCount`). Effort S for status, M for
self-sizing.

**3.5 `GPUCellTopology` disk and ring are per-thread enumerate-and-sort, k ≤ 8. CONFIRMED (limit),
SUSPECTED (cost).** `gpu-cell-topology.ts:30`; stride `3k(k+1)+1` (217 at k = 8) in a private
array per thread. H3 analysis commonly uses k of 10 to 50 (catchments, smoothing). Better: one
invocation per output slot with closed-form ring addressing (H3 rings are walked by direction and
step, so slot `s` of ring `r` is computable), then a segmented sort if (distance, key) order is
required. Effort M.

**3.6 `GPUNeighborSearch` radius mode insertion-sorts each row by ID. CONFIRMED (doc :260),
SUSPECTED (cost).** Dense bands (thousands of neighbors per row) make this O(k²) per row. Better:
the rows are already scanned into CSR slots, so a segmented radix sort by ID over the slots. PLAT-03
already tracks the index unification; this is the sort inside it. Effort S.

**3.7 Two fixed-point accumulators. CONFIRMED.** Cell tables sum into signed 64-bit integers with
a caller-fixed `sumScale` (default 2^16, saturating at 2^62; `analysis-core/cell-key-wgsl.ts:9-66`).
The shared `fixed-point-accumulator.ts` is 96-bit with a data-derived power-of-two unit and is
used by `GPUPointDensity` and the binned KDE. With the default scale a cell table rounds values
below 7.6e-6 to zero and saturates sums of large values (for example population times area in
square metres); the shared accumulator does neither. Better: cell tables adopt the shared
accumulator (keep the exact roll-up property: integer sums of the same unit are still
associative). Effort M.

**3.8 Single-thread finalize nodes duplicate `scan-total`. CONFIRMED.** `gpu-pycnophylactic.ts:399`,
`gpu-cell-set-outline.ts:418`, `gpu-cell-table-compare.ts:658`, `gpu-line-length-per-polygon.ts:682`
each write `offsets[last] + counts[last]` by hand. `createSaturatedScanTotalNodes`
(`analysis-core/kernels/scan-total.ts:87`) does this with 2^32 saturation; the joins use it. Cheap
but these copies wrap silently at 2^32. Effort S.

**3.9 Recompiles on numeric props. CONFIRMED.** S2 coverings bake `tolerance`, `level`, capacity
and `hemisphereCenter` into WGSL (`gpu-s2-cap-covering.ts:87-98`, `gpu-s2-polygon-covering.ts:126,175`).
`GPUNeighborhoodSummary` bakes `focalWeight`. Elsewhere in the domain per-frame values go through
a packed `parameters` view or `GPUParameterBuffer`. Effort S.

Not a finding: `GPUZonalStatistics`, `GPURasterZonalStatistics`, `GPUPointDensity` and
`GPULineDensity` default to deterministic sorted or fixed-point sums; float atomics are opt-in.

## 4. Consistency findings

**4.1 The three cell systems do not offer the same operations. CONFIRMED.** Matrix of what exists
(contributor, file):

| Operation | H3 | Quadbin | S2 | quadkey / geohash | A5 |
| --- | --- | --- | --- | --- | --- |
| Point to cell | `GPUPointToCell`; extended precision exact | `GPUPointToCell`, also inside `GPUCellAggregation` | `GPUPointToCell`, f32 only, exact to ~level 20 | `GPUPointToCell` | none |
| Cell to center / boundary | `GPUCellGeometry` (6 to 10 vertices) | `GPUCellGeometry` | `GPUCellGeometry` | `GPUCellGeometry` | `GPUCellGeometry` |
| Parent | `GPUCellTopology` | `GPUCellTopology` | `GPUS2CellHierarchy` | none | none |
| Children | `GPUCellTopology`, any depth to stride 4096 | same | `GPUS2CellHierarchy`, one level only; `range` for leaf bounds | none | none |
| k-ring / disk | `GPUCellTopology`, k ≤ 8 | `GPUCellTopology`, k ≤ 8 (Chebyshev) | 4 edge neighbors only (`gpu-s2-cell-hierarchy.ts:24,130-147`) | none | none |
| Grid distance / path | `GPUCellGridPath` | none | none | none | none |
| Compact / uncompact | `GPUCellCompaction` | `GPUCellCompaction` | none | none | none |
| Polyfill / cover | `GPUCellCover` (scales) | `GPUCellCover` (scales) | `GPUS2PolygonCovering` (one polygon, serial), `GPUS2CapCovering` (level ≤ 8) | none | none |
| Aggregation, rollup, pyramid | `GPUCellAggregation` (pre-keyed only), rollup, pyramid | all, including points | none | none | none |
| Cell set outline | `GPUCellSetOutline` | `GPUCellSetOutline` | none | none | none |
| Cell area, perimeter | `GPUCellMeasures`, metres or km | none | `GPUS2CellMeasures`, steradians and radians | none | none |
| Edge / vertex identities | `GPUH3CellIdentities` | n/a | none | n/a | n/a |

H3 is near-complete, Quadbin lacks distance and area, S2 is a separate partial API with its own
classes, status codes and units. The type `GPUCellFamily` is `'quadbin' | 'h3'`
(`cell-aggregation/cell-keys.ts:13`) while `GPUCellIndexFamily` adds quadkey, geohash and S2
(`cell-indexing/cell-index-families.ts:17`) and `GPUCellGeometryFamily` adds A5
(`gpu-cell-geometry.ts:34`). Recommended convention: one `family` prop on each operation class,
typed as the widest family the operation supports, and S2 folded into `GPUCellTopology`,
`GPUCellMeasures`, `GPUCellCover`, `GPUCellCompaction` and `GPUCellAggregation` rather than parallel
`GPUS2*` classes.

**4.2 Three zonal contributors overlap and name things differently. CONFIRMED.**

| | `GPUZonalStatistics` | `GPURegionStatistics` | `GPURasterZonalStatistics` |
| --- | --- | --- | --- |
| Zones | polygons via PIP, or caller feature rows | one per-frame region (rect, circle, lasso, pick, mask) | dense raster zone IDs |
| Values | per-point values, weights | per-point values | raster band with nodata, calibration |
| Statistics | counts, valueCounts, sums, weightSums, means, minima, maxima, variances, standardDeviations, densities | count, sum, min, max, histogram, selected IDs | cellCounts, valueCounts, sums, means, minimums, maximums, weightSums, variances, standardDeviations |
| Sum order | `sumOrder: 'sorted' \| 'atomic'` | `GPUReduction` | `sumOrder: 'sorted' \| 'atomic'` |
| Status | `overflow`, `uncertainCount`, lon/lat counts | packed `summary` header words | `overflow` (out-of-range zone ID) |

Overlap is real but the inputs differ, so a single class is not the answer. Align: `minima/maxima`
vs `minimums/maximums` (cell tables also use `minimums`), `counts` vs `cellCounts`, and a shared
"zonal statistics output" type. `GPURegionStatistics` is a one-zone, interactive case of
`GPUZonalStatistics` with a histogram; keep it for its per-frame region but publish its status as
named scalars instead of header words. `GPUCellAggregation` and dataframe `GPUGroupStatistics` are a
fourth and fifth "aggregate to keys" path.

**4.3 Sum-accumulation option names differ.** `sumOrder` (zonal, raster zonal), `sumAccumulation:
'auto' | 'workgroup' | 'atomic'` (`gpu-point-density.ts:80`), fixed `sumScale`
(`GPUCellAggregation`), sequential f32 (`GPUNeighborhoodSummary`). Converge on `sumOrder` with
values `'deterministic'` (sorted or fixed-point, implementation's choice) and `'atomic'`.

**4.4 KDE, IDW, neighbor weights and line density do not share a kernel vocabulary. CONFIRMED.**

| Contributor | Kernel names | Profile | Edge correction |
| --- | --- | --- | --- |
| `GPUKernelDensity` | gaussian, quartic, epanechnikov (`gpu-kernel-density.ts:425`) | 2D-normalised, `3/π(1-t)²` for quartic | none, uniform, diggle (`:84`); polygon window |
| `GPUNeighborSearch` | gaussian, triangular, epanechnikov, bisquare, uniform (`neighbor-search-parameters.ts:69-80`) | 1D PySAL, `15/16(1-z²)²` for bisquare | none |
| `GPUSpatialWeightsTransform` | same five as neighbor search (`spatial-weights/gpu-spatial-weights-transform.ts:37`) | | |
| IDW | power, optional taper (`gpu-inverse-distance-weighting.ts:442-453`) | | none |
| `GPUPointDensity` | caller-supplied convolution weights, in cells | | zero boundary |
| `GPULineDensity` | none: length / cell area (`gpu-line-density.ts:182-200`) | | none |

"quartic" in KDE is "bisquare" in weights; both are correct for their references but a user moving
between them will not see that. Recommend one `GPUKernelProfile` type in `analysis-core` naming the
profile (`'quartic'` with `'bisquare'` as alias), with a separate normalisation flag (2D density vs
1D weight), and one edge-correction type reused by KDE and future line or space-time KDE.

**4.5 Status shapes vary inside the domain.** Per-row codes 0/1/2 (`GPUS2CellMeasures`,
`GPUS2CellHierarchy`, `GPUS2CapCovering` with different meanings), a one-word bitmask
(`GPUS2PolygonCovering:41`), `(written, required)` pairs as `uint32x2` (S2 coverings) instead of
`count` / `requiredCount`, `droppedCount` (`gpu-cell-compaction.ts:83`) instead of `invalidCount`,
per-row lengths with overflow implied by `counts > maximumPathLength` (`GPUCellGridPath`), header
words (`GPURegionStatistics`), and `failedCount` (dot density). The non-S2 cell, join and cover
contributors use the shared vocabulary. Converge on `GPUBoundedResultStatusPort` scalars plus an
optional per-row validity view.

**4.6 Validation style.** All 14 bare asserts of the cell-topology family are in the four S2 files
(`gpu-s2-*`); everything else in the domain uses `validate*` helpers and `throw new Error(id ...)`.
`GPUH3CellIdentities` throws without the id prefix (`:41-42`). The two asserts in joins
(`gpu-spatial-join-prepared.ts:435`, `gpu-spatial-predicate-join.ts:1251`) are internal invariants,
which is the right use.

**4.7 Units.** `GPUCellMeasures` returns metres by default (`units`), `GPUS2CellMeasures` returns
steradians and radians with "multiply externally" (`gpu-s2-cell-measures.ts:23`). Recommend metres
on the authalic sphere with the same `units` prop.

**4.8 Word order.** Most cell contributors accept `wordOrder: 'little-endian' | 'high-low'`; the S2
and H3-identity classes accept little-endian only and take no `mask`.

**4.9 Context.** All position-taking contributors take `spatialContext`. Cell-key contributors
correctly take none and say so. S2 coverings take unit xyz with no context and no lon/lat entry
point, unlike every other cover.

## 5. Duplication and shared-helper candidates

**5.1 Two Hilbert implementations. CONFIRMED.** `hilbert-keys/gpu-hilbert-keys.ts:300` (parallel
prefix, order ≤ 16) and `spatial-join/spatial-join-passes.ts:173` (`HILBERT_INDEX_WGSL`, classic
loop, order 16, "same curve"). `GPUHilbertKeys` is not used by any contributor (only its own
directory and `ontology/spatial-analysis-catalog.ts:346`); the joins, PIP query order,
`GPUSpatialJoinPrepared`, linear referencing, HDBSCAN and zone events all use the join copy. Move
one WGSL function to `analysis-core` and have both use it. The earlier review's "Users: BVH
packing, joins" for `GPUHilbertKeys` is not borne out.

**5.2 How many point or feature index implementations exist.** Building on review B:

| # | Structure | Built by | Traversal | Used by (in scope) |
| --- | --- | --- | --- | --- |
| 1 | Uniform grid | published `GPUGridIndex` | per-consumer gather loops, plus a private rank-in-cell pass (3.3) | `GPUNeighborSearch` grid path, `GPUKernelDensity`, region-statistics grid (`region-statistics-grid.ts`); outside: IDW, GWR, clustering, Dorling, encounters |
| 2 | Feature-box BVH | published `GPUBVH` via `getFeatureBVHNodes` / `getSortedFeatureBVHNodes` (`spatial-join-passes.ts:125,236`) | stackless count/write probe (`:714`) | PIP, predicate join, candidates, prepared, line length per polygon, zonal, buffer selection |
| 3 | Same BVH, branch-and-bound | `getNearestBVHNodes` (`nearest-nodes.ts:274`) | explicit stack, near child first (`nearest-wgsl.ts:521`) | nearest join, `GPUNeighborSearch` `'bvh'` |
| 4 | Sorted 64-bit cell keys | `cell-table.ts` (two `GPUSort`) | binary search | cell aggregation, rollup, pyramid, compaction, table compare |
| 5 | Polygon edge slabs | `analysis-core/kernels/polygon-edge-slabs.ts` | slab lookup | PIP refine, cell cover |
| 6 | Cell-cover scanline intervals | `cell-cover-scanline.ts` | row ranges | cell cover |
| 7 | Rasterized zones | `GPUPolygonRasterization` | none (dense IDs) | areal interpolation, pycnophylactic |
| 8 | Brute force | none | O(n) per query | bounds filter, region mask, S2 cap covering (O(sphere)) |

Two built structures (grid, BVH) from published primitives, with two BVH traversal styles and three
separate KNN front ends: `GPUNeighborSearch` (k ≤ 32), `GPUNearestFeatureJoin` neighbors mode
(k ≤ 64) and IDW's `grid-neighbor-scan.ts`. PLAT-03 tracks unifying the point index for
clustering, LOF and KDE; it should also name `GPUNearestFeatureJoin` as the k > 32 path.

**5.3 Other duplicates. CONFIRMED.**
- Rank-in-cell kernel: `gpu-kernel-density.ts:758`, `gpu-inverse-distance-weighting.ts:290`,
  `gwr-grid-index.ts` (3.3).
- Scan-total one-thread nodes (3.8): four copies.
- Bounds reduction: `hilbert-keys/bounds-reduction.ts` (used by hilbert keys, segment BVH, snap
  rounding) and `createSpatialJoinBoundsNode` (`spatial-join-passes.ts:533`) and `GPUReduction` in
  KDE. One `getBoundsReductionNodes` in `analysis-core` would do.
- Cell measures: `GPUCellMeasures` and `GPUS2CellMeasures`; spherical polygon area helpers in
  `h3-measures-wgsl.ts` and `s2-analysis-wgsl.ts:13-15` (both compute spherical triangle excess;
  same formula SUSPECTED, `h3-measures-wgsl.ts` body not read).
- `gwr-grid-index.ts` carries the same rank-in-cell comment as KDE and IDW (found by grep, body
  not read).
- Binning: `GPUPointDensity` (grid, hexagon), `GPUKernelDensity` binned path (linear binning),
  `GPUCellAggregation` (Quadbin). Three binning kernels with three sum schemes.
- `cell-key-wgsl.ts` is misnamed: it holds the cell-table fixed-point WGSL and ordered-key
  helpers, not cell keys. Fold into `fixed-point-accumulator.ts` with 3.7.

**5.4 Upstream candidates (move).** `cell-indexing/h3-boundary-wgsl.ts` extends upstream
`dggs_h3_get_boundary_point` with distortion vertices (6 to 10 vertices); H3 forward indexing
(`h3-index-wgsl.ts`), local IJ, neighbor and parent/children helpers and the S2 neighbor, encode and
measure WGSL are generic DGGS primitives. Upstream `dggs` has decoders, `dggs_h3_get_parent` and
`dggs_h3_neighbor` but no encoders. These belong in luma.gl `dggs` as ordinary pull requests; this
library would keep only the analysis contributors. Effort M, upstream.

## 6. Comprehensiveness

Ranked by how often a practitioner reaches for them. Items already in `dev-docs/roadmap.md` are
marked.

| Rank | Missing | Reference | GPU-suitable | Reuses | Effort |
| --- | --- | --- | --- | --- | --- |
| 1 | Fractional-coverage zonal statistics (coverage fraction per cell, weighted mean, sum, majority) | exactextract, rasterstats `all_touched` | yes: per polygon edge, per row, analytic coverage | cell-cover scanline, edge slabs, raster zonal sorted sums | M |
| 2 | Zonal median, percentiles, majority, minority, variety, unique | rasterstats, exactextract, `GPUZonalStatistics` non-goal | yes | dataframe `GPUGroupStatistics` percentiles via sorted segments | S |
| 3 | Exact polygon areal interpolation (overlay-based weights) | tobler `area_interpolate`, GeoPandas `overlay` | yes, through candidates and clipping | `GPUSpatialJoinCandidates`, `GPUPolygonOverlay`, `GPUArealInterpolation` CSR | M |
| 4 | S2 at parity with H3: k-ring and vertex neighbors, multi-level children, compaction, aggregation, cover at scale, extended-precision point to cell | S2 `CellUnion`, `RegionCoverer`, BigQuery `S2_*` | yes | fold into existing cell classes (4.1) | M |
| 5 | H3 disk and ring with k > 8 | h3 `gridDisk` | yes (3.5) | `GPUCellTopology` | M |
| 6 | Line density with a search radius and kernel (and line KDE) | ArcGIS Line Density, QGIS Line density | yes: per output cell, BVH or grid over segments | `GPULineDensity` DDA, KDE gather, nearest-segment distance | M |
| 7 | KNN join with k > 64 and radius-limited KNN as one API | PostGIS `<->`, GeoPandas `sjoin_nearest(max_distance)`, scikit-learn `kneighbors` | yes | already tracked as PLAT-03 (k above 32) | — |
| 8 | Quadbin cell area, grid distance, sibling, ZXY and quadkey conversion | CARTO `QUADBIN_*` | yes, closed form | `GPUCellMeasures`, `GPUCellGridPath` | S |
| 9 | Dissolve (union by attribute) for polygons | GeoPandas `dissolve`, PostGIS `ST_Union` GROUP BY | partly; cell sets already have `GPUCellSetOutline` | `GPUPolygonOverlay` (geometry reviewer's domain) | M to L |
| 10 | Per-pixel categorical and dispersion aggregates (`count_cat`, `by`, `std`, `var`, `min`, `max`, `first`, `last`) | datashader | yes | `GPUPointDensity` keys, `GPUGroupStatistics` | S |
| 11 | KDE on the sphere and network KDE | scikit-learn `haversine` KDE, spatstat `density.lpp` | yes | KDE gather with great-circle distance; network distances | M |
| 12 | Adaptive (variable) bandwidth KDE | spatstat `adaptive.density`, scikit-learn none | yes | KNN k-th distance from `GPUNeighborSearch` | S |
| 13 | H3 inverse identities: edge to cells and boundary, vertex to lat/lng, `cellToChildPos` / `childPosToCell`, `getIcosahedronFaces` | h3 v4 | yes | `GPUH3CellIdentities`, H3 boundary WGSL | S |
| 14 | Geodesic point-in-polygon and predicates on lon/lat without projection | PostGIS `geography`, DuckDB spatial spherical | yes | `GPULonLatPolygonPreparation`, S2 edge tests | L |
| 15 | HEALPix, A5 beyond geometry, DGGS-JSON | healpy, A5 | yes | already tracked as RTS-08 | — |
| 16 | (cell, period) cubes on DGGS keys | — | yes | already tracked as TC-04 | — |
| 17 | Distance output on predicate joins | `sjoin(distance_col)` | yes | already tracked as JOIN-01 | — |

cuSpatial baseline: point-in-polygon, quadtree point-in-polygon, point-to-nearest-linestring,
haversine and trajectory functions are all covered at least as well here (BVH plus exact refine,
deterministic). No gap against cuSpatial.

Out of place or candidates to drop or move:
- `GPURegionStatisticsReadback`: a host readback helper; move to an examples or app-utilities module.
- `GPUS2CapCovering`: in its present form (level ≤ 8, full-sphere scan) it should be rewritten (3.2)
  or dropped in favour of S2 in `GPUCellCover`.
- The DGGS WGSL ports (5.4): move upstream.
- `GPUHilbertKeys` as a public contributor is fine, but with no internal user it only earns its
  place once the joins call it.

## 7. Top ten actions

| # | Action | Effort | Files |
| --- | --- | --- | --- |
| 1 | Replace the O(m²) rank-in-cell pass with a stable sort of `(cell, row)` as one `analysis-core` helper | S | `point-density/gpu-kernel-density.ts:758`, `spatial-interpolation/gpu-inverse-distance-weighting.ts:290`, `spatial-regression/gwr-grid-index.ts` |
| 2 | Rewrite `GPUS2PolygonCovering` and `GPUS2CapCovering` as count-scan-write frontier passes, or add `family: 's2'` to `GPUCellCover` and drop them | M | `cell-topology/gpu-s2-*-covering.ts`, `cell-cover/*` |
| 3 | Fold S2 into the shared cell classes (`GPUCellTopology`, `GPUCellMeasures`, `GPUCellCompaction`, `GPUCellAggregation`); one `family` type; metres; shared status; helper validation | M | `cell-topology/gpu-s2-*.ts`, `cell-aggregation/cell-keys.ts`, `cell-indexing/gpu-cell-measures.ts` |
| 4 | Split `candidateOverflow` from `overflow` in all joins and add a first-truncated-row scalar | S | `spatial-join/gpu-point-in-polygon-join.ts`, `gpu-spatial-predicate-join.ts`, `gpu-nearest-feature-join.ts`, `spatial-join-passes.ts`, zonal and line-length consumers |
| 5 | Move cell tables to the shared 96-bit fixed-point accumulator; retire `cell-key-wgsl.ts` | M | `analysis-core/cell-key-wgsl.ts`, `cell-aggregation/cell-table.ts`, `gpu-cell-rollup.ts`, `cell-table-compare` |
| 6 | One kernel-profile and edge-correction vocabulary in `analysis-core` (quartic = bisquare) | S | `point-density/gpu-kernel-density.ts`, `neighbor-search/neighbor-search-parameters.ts`, `spatial-weights/gpu-spatial-weights-transform.ts`, IDW |
| 7 | Align zonal outputs (`minimums`/`maximums`, `counts`/`cellCounts`), shared output type, `sumOrder` everywhere; `GPURegionStatistics` status as named scalars | S | `zonal-statistics/*`, `region-statistics/*`, `raster-analysis/raster-zonal-statistics/*`, `point-density/gpu-point-density.ts` |
| 8 | Zonal percentiles and majority/variety via sorted segments; then fractional-coverage zonal statistics | S then M | `zonal-statistics/*`, `raster-analysis/raster-zonal-statistics/*`, `cell-cover/cell-cover-scanline.ts` |
| 9 | One Hilbert WGSL, one bounds reduction, `createSaturatedScanTotalNodes` in the four hand-written totals | S | `hilbert-keys/*`, `spatial-join/spatial-join-passes.ts:173,533`, `areal-interpolation/gpu-pycnophylactic.ts:399`, `cell-set-outline:418`, `cell-table-compare:658`, `line-length-per-polygon:682` |
| 10 | H3/Quadbin disk and ring past k = 8 with per-slot addressing; Quadbin area and distance | M | `cell-topology/gpu-cell-topology.ts`, `cell-topology/cell-topology-wgsl.ts`, `cell-indexing/gpu-cell-measures.ts`, `gpu-cell-grid-path.ts` |

Separately, upstream: offer the H3 encoder, H3 distortion-vertex boundary and S2 encode/neighbor
WGSL to luma.gl `dggs` (5.4).
