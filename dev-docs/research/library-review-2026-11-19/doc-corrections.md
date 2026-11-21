# Documentation corrections (2026-11-19)

Scope: Markdown outside `dev-docs/roadmap.md`, `dev-docs/research/` and the files the sibling clone `luma-contributors-ki` changes. Every applied row was checked against the source named in the table. Generated blocks are the MDX comment pairs `{/* parameters: GPUClass ... */}` to `{/* /parameters */}` written by `test/analysis-core/doc-parameters.ts`; no edit touched one.

## Applied

| Document:line | Said | Says now | Source confirming | Finding |
| --- | --- | --- | --- | --- |
| docs/api/gpu-spatial-analysis/point-process.md:13 | `GPUOPTICS` grid route handles 65,536 rows (including xi) | 1,048,576 rows on the grid route; xi extraction limited to 65,536 | `spatial-clustering/gpu-optics.ts:66,69,350-362,421` | PTS-122 |
| docs/api/gpu-spatial-analysis/point-process.md:117 | `frontierBatchSize` default 64 | default 256 (maximum 256) | `gpu-optics.ts:88,124,424` | PTS-122 |
| docs/api/gpu-spatial-analysis/point-process.md:119 | grid route at most `GPU_OPTICS_GRID_MAXIMUM_ROWS` (65,536); "Xi extraction now accepts the indexed limit of 65,536 rows" | constant is 1,048,576; xi keeps its own reference-tier limit of 65,536 (`GPU_OPTICS_XI_MAXIMUM_ROWS`) | `gpu-optics.ts:66,69,421` | PTS-122 |
| docs/api/gpu-spatial-analysis/space-time-clustering.md:67 | ST-OPTICS grid route at most 65,536 rows (including xi) | at most 1,048,576 rows (xi extraction at most 65,536) | `gpu-optics.ts:66,69,350-362` | PTS-122 |
| docs/api/gpu-spatial-analysis/space-time-clustering.md:122 | ST-HDBSCAN limits stay at 1,048,576 rows | 4,194,304 rows (`GPU_HIERARCHICAL_DENSITY_CLUSTERING_MAXIMUM_ROW_COUNT`); 32 minimum samples unchanged. The 2^20 cap applies only to the `'neighbor-search'` reference route, which rejects `times` | `spatial-clustering/gpu-hierarchical-density-clustering.ts:96-105,150-158,280-285` | PTS-123 |
| docs/api/gpu-spatial-analysis-cross-reference.md:99 | Dorling "at most 1024 circles/256 passes" | 1,073,741,822 rows on the neighbor-grid route (default from 32,768 rows), 65,536 on the all-pairs route, 256 passes | `dorling-cartogram/gpu-dorling-cartogram.ts:47,56-61,84-110,153-165` | PTS-127 |
| docs/api/gpu-spatial-analysis/gpu-dorling-cartogram.md:12-13 | each pass sums forces over "all other circles" | shared neighbor grid from 32,768 rows by default, all other circles with `neighborGrid: false` | `gpu-dorling-cartogram.ts:91-98,153-157` | PTS-127 |
| docs/api/gpu-spatial-analysis/gpu-dorling-cartogram.md:19 | "at most 1024 input rows" | 1,073,741,822 rows on the grid route, 65,536 with `neighborGrid: false` | `gpu-dorling-cartogram.ts:59-61,160-165` | PTS-127 |
| docs/api/gpu-spatial-analysis/gpu-dorling-cartogram.md:42-44 | O(n^2) pair queries per pass; "Larger interacting layouts require a future indexed implementation" | O(n^2) is the all-pairs route; the grid route visits every overlapping pair but can approach O(n^2) for clustered layouts or a dominant circle; the "future indexed implementation" sentence removed because the grid route ships | `gpu-dorling-cartogram.ts:91-98,107-110` | PTS-127 |

## Deferred to the staging copy

All rows were verified against source first. "Staging copy" is `/Users/robertchristie/Documents/GitHub/luma-contributors-ki`.

| Document | Correction | Staging copy |
| --- | --- | --- |
| docs/api/gpu-raster/operations-analysis.md (`output.settledAt` row, `:220`) | Delete "Not supported with `continuation`"; it is supported with cumulative rounds (`raster-analysis/cost-distance/gpu-cost-distance.ts:183-187`). RT-103 | Fixed in the staging copy (`:251` now says "With `continuation`, import it from a caller-owned buffer; stamps are cumulative") |
| docs/api/gpu-raster/operations-analysis.md (settledAt row) and the class page | State that `iterationCount`, `settledAt` and budget-edge `converged` depend on tile scheduling while costs do not. RT-104 | Still wrong: the clone adds "stamps depend on scheduling" for `settledAt` only; `iterationCount` and `converged` are not mentioned |
| docs/api/gpu-raster/raster-embeddings.md:22 (`:16`, `:34` in the working tree) | "1-128 for the reference covariance family" is incomplete: k-means, zone mean, nearest pixels and pyramid level also call `validateRasterEmbeddingShape` with the default 128 (`raster-embedding-common.ts:15-20,31-37`; `gpu-raster-embedding-kmeans.ts:69`, `-zone-mean.ts:119`, `-nearest-pixels.ts:67`, `-pyramid-level.ts:174`, `-covariance.ts:84`). List the five classes. RT-107 | Still wrong: the clone keeps the "covariance family" wording at `:22,40` |
| docs/api/gpu-raster/operations-analysis.md:316 and docs/api/gpu-raster/expansion.md:230 | Name the Earth model per output: whole-cell geographic zonal uses the authalic sphere, fractional siblings, terrain, walking cost and flow use WGS84 meters per degree at the row-centre latitude. RT-106 | Still wrong: the clone keeps "WGS84 authalic sphere" at `:350` with no contrast |
| docs/api/gpu-spatial-analysis/operators-regression.md:1086 | GTWR "at most 32,768 rows" is the brute-force reference route (`GPU_GTWR_MAXIMUM_ROW_COUNT`); the grid route (`searchStructure: 'grid'`, compact kernel or `gaussianCutoff`, no `bandwidthSearch`) allows 16,777,216 rows (`gpu-geographically-temporally-weighted-regression.ts:62-66,205-216,297-304`). SSR-122 | Still wrong: the clone keeps the 32,768 sentence at `:1283` |
| docs/api/gpu-spatial-analysis/operators-regression.md, panel-fixed-effects.md, coverage-statistics.md | Rose `positive-y-clockwise` origin and degrees are undocumented (`gpu-rose-statistic.ts:90,165,399`); panel regression and panel error limits differ; `GPUSpatialPanelLagMaximumLikelihood` has no `###` heading (only `coverage-statistics.md:103,533,537` mention it). SSR-122 | Not checked per line; the clone changes all three files but the report's items were not found in the sampled passages. Treat as still wrong |
| docs/api/gpu-spatial-analysis/operators-statistics.md:1656,1660 | HDBSCAN `positions` limit is 2^22 rows (`GPU_HIERARCHICAL_DENSITY_CLUSTERING_MAXIMUM_ROW_COUNT`), core distances use the BVH route by default; the `GPUNeighborSearch` route is the opt-in `'neighbor-search'` reference limited to 2^20 and no `times`. PTS-123 | Still wrong: the clone keeps "2^20 rows" at `:1710` and the `GPUNeighborSearch` pipeline at `:1715` |
| docs/api/gpu-spatial-analysis/operators-statistics.md:1578 | `'kmeans++'` "matches scikit-learn": scikit-learn uses greedy k-means++, the GPU does one D^2 draw per round (`gpu-spatial-kmeans.ts:256-262`). PTS-136 | Still wrong: the clone keeps it at `:1636` |
| docs/api/gpu-spatial-analysis/operators-statistics.md:1194,1420,1434,1448 | AZP, Max-p, Ward and location-allocation have only `####` parameters headings, no own section. PTS-128 | Still wrong: the clone has the same `####` headings at `:1478,1492,1506` |
| docs/api/gpu-spatial-analysis/operators-statistics.md:~745 | State that `pZSimulated` is always one-sided (`0.5 * two-sided`), whatever `alternative` is (`gpu-global-permutation-test.ts:1160`). SSR-128 | Still silent: the clone lists `p_z_sim` at `:746` without the convention |
| docs/api/gpu-network/operators.md:683-697 | The documented cost-matrix plus accessibility example requests no `status.converged`; `costs` is `+Infinity` for beyond-limit, unreachable and round-budget-truncated entries (`gpu-network-cost-matrix.ts:54-66`). Add `status: {converged}` to the example prose or note the three causes. NET-101 | Still wrong: the clone's example (`:696-698`) is unchanged and does not request status |
| docs/api/gpu-spatial-analysis/operators-lines.md:94-97 | "Near-antipodal interpolation can be numerically unstable": the code writes NaN interior vertices when `pi - angle < 1e-6` (`gpu-great-circle-arcs.ts:107-109`, `geodesic-wgsl.ts:259-261`). GEO-120 (the report marks the code behaviour as read, no probe run) | Not compared against the clone's copy; treat as still wrong |

## Generated blocks and source comments

No correction falls inside a `{/* parameters: ... */}` block. The following belong in source.

| Where | Fix | Finding |
| --- | --- | --- |
| `gpu-spatial-analysis/spatial-regression/gpu-seemingly-unrelated-regression.ts:78-84` TSDoc | "not spatial SUR" is wrong; the `spatial` option exists (`:57`) | SSR-122 |
| `gpu-spatial-analysis/spatial-interpolation/vecchia-fisher-fit.ts:24` | "a zero nugget stays zero and fixed" is false for Matern 3/2, which uses a floor (PTS-115). The docs page (`vecchia-likelihood.md:75`) already says the nugget is pinned at the floor | PTS-115 |
| `gpu-spatial-analysis/emerging-hot-spots/gpu-emerging-hot-spots.ts:222` | comment says 1024-bin blocks; `BIN_BLOCK` is 4096 (`:66`, per the report; not re-read) | PTS-135 |
| `gpu-spatial-analysis/label-point/gpu-label-point.ts:54` | `points` doc says "longitude/latitude or planar"; class is planar only | GEO-119 |
| `raster-analysis/.../raster-relaxation.ts:762` | "never depends on scheduling beyond float rounding" is not true of `iterationCount`, `settledAt` and budget-edge `converged` | RT-104 |
| Antimeridian note for great-circle kriging and IDW (`separation-wgsl.ts:99-111`) | Documented in the WGSL comment and in `operators-density.md:398,514`, so no page is silent; no change needed beyond the staged page | PTS-117 |
| `gpu-spatial-analysis/permutation-inference/gpu-global-permutation-test.ts` and local test | inference metadata still reports the requested `permutationCount` after early stopping | SSR-114 |
| `gpu-spatial-analysis/line-segmentize/gpu-great-circle-arcs.ts:107-109`, `polygon-triangulation/gpu-polygon-triangulation.ts:215-226`, `geometry-measures/geometry-measures-kernels.ts:609-624` | NaN near antipodes, zero-area triangles from the monotone route versus "a tiling", unwrapped feature centroids: TSDoc to state them | GEO-120, GEO-124, GEO-117 |

## Not applied

| Item | Reason |
| --- | --- |
| PTS-125: "Partition does not depend on unit" (`change-points-and-expectation-based-scan.md:11`, `gpu-location-change-points.ts:146-148`) | Report wrong. The kernel divides `penalty` by `scale` for `'mean'` and slope (`gpu-location-change-points.ts:446,498`), so the partition stays unit-free. Not executed on a GPU here |
| PTS-117: docs silent on the antimeridian cut for kriging and IDW | Report wrong about the docs: `operators-density.md:398,514` already state that the query does not wrap the antimeridian |
| PTS-124: HDBSCAN `minimumSamples` default | Report wrong about the docs: the source comment (`gpu-hierarchical-density-clustering.ts:150-152`) and the docs page both state `min(minimumClusterSize, 32)`; the divergence from scikit-learn is the real limit and is already stated |
| PTS-138: Hanisch G bias | Report wrong about the docs: `gpu-dataframe-analysis.md:1069-1077` already states the 0.44 versus 1 example and the `maximumDistance` advice |
| SSR-124: geodetector p-value above non-centrality 10,000 | Documented behaviour, report agrees; no edit |
| NET-103: threshold must not exceed `costLimit` | Already stated at `operators.md:676-678` |
| NET-104, NW02, NW05, LRN-01, LRN-02, GAP rows | Roadmap content; owned by the lead reviewer |
| PTS-119, PTS-129, PTS-131, PTS-132, PTS-134, GEO-123, GEO-127, GEO-128 and other C findings that state a behaviour no current page claims | Not verifiable as a wrong statement: the pages are silent or the page that would carry the note is in the staged list (`operators-density.md`, `recipes.md`, `operators-geometry.md`, `operators-joins.md`). Add the notes when the staging copy lands |
| area-direction.md factual statements (counts of 633 classes and 548 operators, `visgl/luma-contributors` naming) | No README or CHANGELOG line states these counts; the README links use the `visgl` path as the report says, and the owner decision is open (DIR-112), so nothing is wrong to fix |

## Batch 2 (2026-11-19)

Inputs: the cells, dataframe/point-cloud/record, tooling and roadmap/developer-document area reports. Every applied row was checked against the source, script or build output named in it (the working tree is the truth, including today's urban and landscape recipe fixes). The staged list is `git -C /Users/robertchristie/Documents/GitHub/luma-contributors-ki diff --name-only` plus untracked files; no file on it was edited. `dev-docs/roadmap.md` was not edited.

### Applied

| Document:line | Said | Says now | Source confirming | Finding |
| --- | --- | --- | --- | --- |
| AGENTS.md:10 | `yarn verify-package-exports` "verify every package entry point loads" | imports and requires every JavaScript entry; non-JavaScript targets such as assets are only resolved | `scripts/verify-package-exports.mjs:112-150` (`import(specifier)` and `require(specifier)` for `.mjs/.cjs/.js` targets, `import.meta.resolve` otherwise) | ROAD-107 |
| AGENTS.md:25 | `yarn typecheck` "checks all source roots separately" | one combined project, `tsconfig.check-src.json` over `modules/*/src` | `package.json:18`; `tsconfig.check-src.json` | ROAD-107 |
| AGENTS.md:58-62 | command-line heap options take precedence over `NODE_OPTIONS`; wrappers resolve the precedence at entry | adds that verify and release validate every heap option on both sides first, so a `NODE_OPTIONS` heap above 8192 MiB is rejected even when a lower command-line heap would override it | `scripts/verification-policy.mjs:98-124`; run: `getChildNodeOptions('--max-old-space-size=16384', ['--max-old-space-size=4096'])` throws "verify allows a V8 heap of at most 8192 MiB" | TOOL-103 |
| AGENTS.md:123 | engine allowed "for examples and tests only"; checker scope unstated | states that `yarn check-upstream-first` covers the three packages, their tests and `examples/deck`, not `website`, where ten files import `@luma.gl/engine` | `scripts/check-upstream-first.mjs:24-30,96-97`; `grep -rl "@luma.gl/engine" website` finds 10 non-JSON files | ROAD-107 |
| AGENTS.md:160 | contributor rules without the worked example | links `dev-docs/authoring-walkthrough.md` next to the contract | `dev-docs/authoring-walkthrough.md` exists; no entry link existed in AGENTS.md or the contract | ROAD-109 |
| dev-docs/contributor-contract.md:3 | no link to the worked guide | one sentence linking the authoring walkthrough | as above | ROAD-109 |
| README.md:26 and CHANGELOG.md:15 | "carries no copies of luma.gl code" | no copies of luma.gl APIs; a few small WGSL helpers are pinned copies, registered in `dev-docs/upstream-workarounds.md` and compared with upstream text | `modules/analysis/test/analysis-core/official-wgsl-copies.node.spec.ts` pins `safeNormal`, `encodeOrderedFloat`, `coverTileLatitude` and the grid-coordinate import; register row "Other carried WGSL" | ROAD-107 |
| README.md:45 | "kepler-compute is parked" | registered workspace whose deck.gl 10 port is still being qualified | `package.json:11` workspace entry; `examples/deck/kepler-compute/package.json` has `start`, `build`, `typecheck` scripts; roadmap KE-EX/KE-PORT rows state the gates are open | ROAD-107 |
| dev-docs/upstream-first-plan.md:15 | exceptional WGSL copies only for grid kernels | grid kernels plus the three named small helpers, each still needing a register row, a spec entry and a pin | same spec; register "Other carried WGSL" row | ROAD-107 |
| dev-docs/upstream-first-recording.md:12 and docs/api/computation-record.md:215 | timings are "start, end and duration in milliseconds"; metrics are "nanosecond metrics" | a duration-only report (`gpuTimeMilliseconds`) yields `cpuEncodeMs` alone; metrics carry `timebase: 'ms'` when any node has `gpuStartTimeMilliseconds` and `'ns'` otherwise | `computation-record/computation-metrics.ts:49-70,90`; `computation-record-types.ts:593`; `grep gpuTimeMilliseconds modules/analysis/src` has no reader | ROAD-111 |
| dev-docs/upstream-first-recording.md:17 | unrecorded ownership "conservatively says `imported`" | `unknown` ownership with the summary's byte count (`lower-bound` when only observed bytes exist) | `describe-command-graph.ts:174,244,262,270` | ROAD-106 |
| dev-docs/upstream-first-recording.md:57 and dev-docs/upstream-workarounds.md:34 | per-node kernel metadata presented as wholly missing / not stating local adoption | published `compile({kernelMetadata})` supplies `preflight.nodes[].kernel` for nodes that declare it (`GPUScan` yes, `GPUSort` and `GPUReduction` no); the local describer does not read the field | `grep kernelMetadata modules/analysis/src/computation-record` has no match; register row already stated the upstream side | ROAD-106 |
| dev-docs/coverage-matrix.md:30 | "Published" rows "use `@luma.gl/graph@10.0.0-alpha.10`" | recorded against alpha.10, not walked again for the alpha.16 target; links the release guide | `package.json` target alpha.16; `dev-docs/release-package.md` | ROAD-108 |
| dev-docs/script-gates.md:202-206 | three regression merges "have no enforced algorithmic tile-count cap"; "records each serial loop's actual bound and owner" | each constructor enforces `tileCount <= 4096` (including explicit tile widths); only some allowances record an `owner` and many reasons stay generic | `gpu-spatial-error-gm.ts:356-361`, `gpu-spatial-two-stage-least-squares.ts:87-89,395-397`, `gpu-spatial-weights-traces.ts:72-73,345`; baseline JSON has 166 allowances of which 19 carry `owner`, 63 start "Pre-review legacy departure" | TOOL-106 |
| dev-docs/release-package.md:57-58 | the `.mts` sentinel pins declaration resolution | adds that it does not resolve the relative references of other declarations; `line-segmentize-parameters.d.ts` still imports `import("../../analysis-core")` | `modules/analysis/dist/gpu-spatial-analysis/line-segmentize/line-segmentize-parameters.d.ts:5` read today | TOOL-102 |
| modules/analysis/test/oracles/GENERATORS.md:1091-1092, :21 | 748 fixtures, 242 generator scripts | 790 fixtures on disk, 290 scripts; `scripts/oracle-fixtures.json` configures 290 generators with 348 fixture outputs; the 542/204/2 mapping split is marked as derived for the old tree | `ls fixtures \| wc -l` = 790; 290 `.py/.R/.mjs/.cpp` files; manifest output count 348 (the report says 337 paths; the manifest in the tree has 348 under `oracles/fixtures/`) | TOOL-107 |
| modules/analysis/test/gpu-spatial-analysis/spatial-interpolation/interpolation-performance-README.md:42 | command targets `modules/graph/test/...` | `modules/analysis/test/...` (the spec exists there) | file listing | IN01 (tooling report) |
| docs/api/gpu-dataframe-analysis.md:534 | classification inputs "must be single packed views; chunked vectors are rejected" | `GPUColumnQuantiles`, `GPUClassBreaks`, `GPUColorScale` and `GPUBivariateClassification` accept packed or chunked row inputs and outputs with identical chunk lengths; controls, edges and counts stay single packed views | `gpu-column-quantiles.ts:63,149`, `gpu-class-breaks.ts:116,222`, `gpu-color-scale.ts:75,148-150`, `gpu-bivariate-classification.ts:71-75` | DPC-116 |
| docs/api/gpu-dataframe-analysis.md:933-943 | every pair-statistics class publishes an `overflow` word | only `GPUVariogram` and `GPUSpatialCorrelogram` do; Ripley, distance functions, envelope and point-pattern indices have none (`pairCounts` of Ripley saturates) | `pair-statistics/gpu-variogram.ts`, `gpu-spatial-correlogram.ts:145` have `overflow`; `gpu-ripley.ts:68-88`, `gpu-point-pattern-indices.ts:219-` have no status/overflow | DPC-116 |
| docs/api/gpu-dataframe-analysis.md:469 | natural-breaks bin count described only as an accuracy setting | a requested class count above the bin count is clamped to it | `column-classification/natural-breaks-nodes.ts:572-574` | DPC-120 |
| docs/api/gpu-spatial-analysis/landscape-ecology.md:107-113 | boundary mode: inspect `status.candidateOverflow` and capacity overflow, nothing on `status.overflow` | `status.overflow` is always 0 and `status.requiredCount` is the patch count; candidate count and capacity-overflow flag are the two words of `intermediates.candidateRequirement` | `landscape-ecology/patch-proximity-boundary.ts:407-418,638-651` | URB-102 |
| docs/api/gpu-spatial-analysis/urban-morphology.md:207 | validation before adding nodes, retry on the same graph (stated without exception) | states the exception: overlay device-limit checks of the tessellation and adjacency-with-structures recipes can reject after nodes were added | `/private/tmp/.../review19/fixes/fix-urban-recipes.md`, "Residual" paragraph; the rest holds after today's `validateProvidedRecipeViews` move | RCP-102 |
| CHANGELOG.md:408-409 | "Recipes left resources registered on the caller graph after a failure at a later stage." | scoped to the spatial and network recipe indexes; adds that the 13 urban and landscape recipes validate options, packed layouts and caller outputs before the first `graph.add`, with the overlay device-limit residual | same status file; CHANGELOG.md:184 was already scoped to the two indexes | RCP-102 |
| docs/api/gpu-spatial-analysis/expansion-recipes.md:219-223 | cell-flow totals equal the valid-pair sum; "bounded direct-scan cost" | equal up to float32 rounding; cost is O(cells x pairs) with no enforced pair-count limit | `recipes/cell-flow-topology-recipe.ts:238,280` (sequential f32 sums over all pairs); `docs/api/gpu-network/expansion.md:155` | RCP-105 |
| docs/api/gpu-spatial-analysis/operators.md:19 and docs/api/gpu-spatial-analysis-cross-reference.md:126 | fragment `#gpukmeans` | `#gpuspatialkmeans` | heading `### \`GPUSpatialKMeans\`` at `operators-statistics.md:1562` | ROAD-109 |

### Deferred to the staging copy

| Document | Correction | Staging copy |
| --- | --- | --- |
| dev-docs/regression-loop-inventory.md:15-17 | "their old baseline entries can be removed by the lead" is stale: `scripts/contributor-contract-baseline.json` has no `restricted-regime` entry and the sites carry `contract-allow` comments (`gpu-restricted-regime-regression.ts:428,518`). Date the 21-site/15-file count as the historical baseline (ROAD-108) | Still wrong in the clone (same sentence at `:15-17`) |
| docs/api/gpu-spatial-analysis/recipes.md:53-54 | After the 13 urban and landscape recipes moved validation ahead of `graph.add`, say so and name the overlay device-limit residual instead of "their own family contracts" (RCP-102) | The clone has the same sentence at `:53-54`; not updated |
| docs/api/gpu-spatial-analysis/operators-cells.md:36 | "exact to about level 20" for S2 is wrong: measured mismatch 6.6 percent at level 20, exact only through level 10; Quadbin differs from float64 in 1.9e-4 of points at zoom 16 and 1.3 percent at zoom 26 (DGGS-109) | Still wrong in the clone (`:36` identical) |
| docs/api/gpu-spatial-analysis/operators-cells.md | document the `GPUCellTable.sums` headroom at the default `sumScale` 65536 (a row above 7.04e13 clamps, a cell sum above 2^63/65536 wraps) (DGGS-102) and the candidate-table product rule `(featureCount + 1) * (candidateCapacity + 1) < 2^32` of `GPUCellCover` (DGGS-103) | Not checked per line in the clone; treat as still missing |
| docs/api/gpu-spatial-analysis/global-coordinates.md:186 | state that ring orientation is not validated: a clockwise hole is ignored and a clockwise shell contains nothing, status 0 (DGGS-105) | Not checked per line in the clone |
| docs/api/gpu-network/expansion.md:~155 | cell-flow topology cost and the missing limit are stated; add the matching float32 tolerance for the flow totals (RCP-105) | Not checked per line in the clone |
| dev-docs/roadmap.md | all `ROAD-1xx` rows, including the LR-10/ROSE-01 obsolete premises, the pending `library-review-2026-11-19/README.md` target, split IDs and the KE-WF rows | Owned by another agent; not touched |

### Generated blocks and source comments

No edit touched a `{/* parameters: ... */}` block. These belong in source.

| Where | Fix | Finding |
| --- | --- | --- |
| `gpu-spatial-analysis/cell-indexing/h3-index-wgsl.ts:57` TSDoc | "longitude may be any finite value below 3e38" is false above about 1e8 degrees (f32 quadrant reduction); document `\|longitude\| < 1e6` or reduce exactly | DGGS-106 |
| `healpix-cells/gpu-healpix-point-to-cell.ts:106` comment | "a mismatch is always an adjacent cell" is false near the belt edge | DGGS-104 |
| `dataframe-analysis/pair-statistics/gpu-ripley.ts`, `gpu-ripley-distance-functions.ts`, `gpu-ripley-envelope.ts`, `gpu-point-pattern-indices.ts` | TSDoc should state that these classes publish no overflow word | DPC-116 |
| `dataframe-analysis/column-classification/gpu-class-breaks.ts` `naturalBreaksBinCount` TSDoc (`:143`) | state the class-count clamp to the bin count | DPC-120 |
| `gpu-spatial-analysis/landscape-ecology/patch-proximity-boundary.ts:92` TSDoc | name `intermediates.candidateRequirement` as the capacity-overflow carrier | URB-102 |

### Not applied

| Item | Reason |
| --- | --- |
| research/README.md:26 (upgrade scripts direction), parity report absent from the index (ROAD-108) | Outside the permitted edits under `dev-docs/research/`; the file has no dead relative link |
| AGENTS.md:128 "never copy luma.gl source" versus the pinned WGSL helpers (ROAD-107) | Policy decision (see Proposals); the rule text was kept. The README, CHANGELOG and upstream-first plan now state the exception factually |
| AGENTS.md ten website engine imports: allow or forbid | Same: only the checker's scope was stated, not a policy |
| ROAD-102, 103, 104, 105, 110, 112-115 | Roadmap rows, test findings or maintainability notes, not statements in user docs |
| CHANGELOG "125 contributors" (`CHANGELOG.md:62`) | Count not re-derived; the census unit is ambiguous (see Proposals) |
| DGGS-108, 112, 113, 114; TRJ-101 to TRJ-107; URB-104; RCP-103, RCP-104 | Source or test defects; the docs are silent on each point, so nothing a page says is wrong |
| TOOL-101, 104, 105, 108 to 121 | Source, script or test findings; no document makes the contradicted statement |
| DPC-117 to DPC-119, DPC-121 | Source defects; the guide's relevant sentences were not shown to contradict the code |
| ROAD-109 `dev-docs/roadmap.md:24` pending review index target | Roadmap link; the review index does not exist yet |
| Seven historical `#L` link fragments and the 441 historical missing targets under `dev-docs/research/library-review-2026-11-17/` etc. | Research is not editable here; revision permalinks are the proposed fix |
| `docs/api/pointcloud-analysis.md:544` "alpha.10" | Describes when the primitive was published; not a claim about the installed target |

### Proposals not applied

- **AGENTS.md shortening** (ROAD-115): about 100 lines of gate and build mechanics (lines 20-119) precede the architecture rules. The roadmap report proposes moving grouping, Vite ambient types, manifests, caches, report membership, packed-worker isolation, incremental-build repair and snapshot/promotion/rollback detail to `dev-docs/script-gates.md`, `dev-docs/testing-gates.md` and `dev-docs/release-package.md` first, then replacing AGENTS.md with a version of about 60 lines (sections `Work locally`, `Relationship to luma.gl`, `Plans and docs`; text in `area-roadmap-devdocs.md`, section "AGENTS adherence and proposed replacement"). Constraints from the same report: `release-package.node.spec.ts:43` extracts an AGENTS.md section by heading and `documentation-facts.node.spec.ts:110` expects the runner and manifest paths, so keep `## Relationship to luma.gl` and `## Plans and docs`, or change those specs in the same patch. Move the release-recovery detail into `release-package.md` before shortening. Not applied because the instruction was to keep every rule and not to restructure.
- **Pinned-copy policy**: decide whether AGENTS.md rule "never ... copied luma.gl source" should name the registered pinned WGSL helpers, and whether the ten website `@luma.gl/engine` imports are allowed (extend `check-upstream-first` to `website`, or state the exception).
- **Changelog restructuring** (ROAD-114): one Unreleased section ordered Compatibility and installation, Required migrations by family, Changed result semantics, Added capabilities, Fixes, Qualification and known limits; move private consumer changes behind links; replace "125 contributors" with a dated, defined census (CPU operation class, pure function and GPU contributor are different units); the report's migration table covers network, spatial joins, alignment, raster and classification.
- **Document moves or merges** (ROAD-114, ROAD-115): move consumer stages, parked options, pass histories and caller migrations out of the roadmap by owner; keep gate machinery in the script and release guides; give `research/README.md` the parity report and the November 19 index and point upgrade execution at `scripts/upstream/`.
- **Operation-forms subsections** (DGGS-113): 18 of 19 operation-form classes in the cells scope are named in no docs file; one subsection per family page.
- **Generated test-census prose** (TOOL-107): generate the GENERATORS.md counts (disk, configured outputs, provenance, completed regeneration) instead of hand-written numbers.

## Batch 3 (2026-11-19)

Inputs: the network, raster/terrain, API-docs/catalog, core-contract, luma-upstream and consumer-seams area reports (DOC-101 and DOC-102 left to the other agent), and the DOCS sections of the `fix-*.md` records present when this batch ran (23 files). Each applied row was checked against the working-tree source named in it. `dev-docs/roadmap.md` was not touched. The staging-owned list is `git -C /Users/robertchristie/Documents/GitHub/luma-contributors-ki diff --name-only 9e93cbc7 -- docs dev-docs README.md CHANGELOG.md AGENTS.md modules/analysis/README.md` plus its four untracked pages (`gpu-network/status-oracles.md`, `spatial-weights-audit.md`, `gpu-terrain/output-oracles.md`, `gpu-terrain/status-and-count-evidence.md`). "Clone" below is that copy, compared by grep for the key phrase only.

### Applied

| Document:line | Said | Says now | Source confirming | Finding |
| --- | --- | --- | --- | --- |
| docs/api/gpu-raster/remote-sensing.md:29 | Savitzky-Golay: "There is no irregular-time interpretation" | `dateAxis` fits the polynomial on real elapsed time over the same window; without it the fit uses slot indices; composites ignore dates; no implicit resampling | `raster-time-series/gpu-raster-temporal.ts:44-52,84` (`dateAxis`, `resolveGPURasterIrregularDateAxis`) | RT-138 |
| docs/api/gpu-raster/remote-sensing.md:49,97 | region graph: 256 regions enforced on every route, "up to 256 region IDs" | 256 on the dense and adaptive routes; fixed-linkage Boruvka up to 65,535 (`MAXIMUM_BORUVKA_REGION_GRAPH_REGION_COUNT`, `4 * regionCount^2`-byte matrices subject to device bindings); `maximumIterations` at most 256 on every route | `remote-sensing/gpu-raster-region-graph.ts:46-50,112-148` | RT-138 |
| docs/api/gpu-network/flow-models-mob10.md:26 | finite-size normalization makes "every row sum to `T_i`" | only when no two destinations share a distance from the origin; the tie case (zones 0, -1, +1, unit populations, `T_0` = 10 gives 7.5 and 7.5) is stated | `flow-aggregation/gpu-radiation-flow-model.ts:49-52,91-100` (comment now carries the condition) | NET-119 |
| docs/api/gpu-spatial-analysis/interpolation-expansion.md:88 | space-time kriging: "A query scans every sample... no resident neighbor index input yet" | `sampleIndex` (default `'auto'`): scan to 4,096 samples, then a per-frame metric grid with identical neighbors; `'brute-force'` limited to 1,048,576; `'grid'` to 1,073,741,808; whole-index walk cases named | `spatial-interpolation/gpu-space-time-kriging.ts:52-57,133-136,143-166,217-229` | DOC-105, LUMA-110 |
| docs/api/gpu-spatial-analysis/operators-density.md:627 | "Reference tier: every query scans every sample... limited to 1,048,576 rows" | the same `sampleIndex` rule; samples one packed view, queries chunked | same | LUMA-110 |
| docs/api/gpu-spatial-analysis/urban-morphology.md:48,62 | tessellation limits "64 buildings and 128 samples", "at most 64 buildings" stated for the recipe | those are the `mode: 'reference'` limits; the local route (`candidateRadius`) accepts 1,000,000 buildings, 16,777,216 samples, 32 samples per building, candidate capacity 256, cell vertex capacity 128, 8,192 grid visits, enclosure 32 vertices and 32 rings | `urban-morphology/building-tessellation.ts:49-58,118-134`; `building-tessellation-local.ts:44-70,187-255` | DOC-105 |
| docs/api/gpu-spatial-analysis-cross-reference.md:32 | point-in-polygon join: "one overflow flag" | `status.overflow` covers BVH leaf, candidate and `matches` capacity; optional `status.candidateOverflow` isolates the intermediate capacities; `ids` has its own overflow word | `spatial-join/gpu-point-in-polygon-join.ts:170-205` | DOC-103 |
| docs/api/gpu-dataframe-analysis.md:175 | `output.status.code` "for model fits" in a shared status convention | `status.code` belongs to regression-style solver fits (`GPUSolverStatusPort`); stage loss is a separate word; per-model fits such as `GPUVariogramModelFit` write a per-model `output.status` vector | `analysis-core/contracts/inference.ts:81-85`; `pair-statistics/gpu-variogram-model-fit.ts:44-52` | DOC-103 |
| docs/api/gpu-spatial-analysis-cross-reference.md (24 links) | dataframe contributors linked to `https://luma.gl/docs/api-reference/experimental/gpu-dataframe-analysis#...` (upstream pages for locally implemented operators) | `/docs/api/gpu-dataframe-analysis#...`; every fragment checked against the local heading slugs (8 of 8 exist). The luma.gl raster and SQL links are unchanged: those operators are upstream | local heading list of `docs/api/gpu-dataframe-analysis.md` | DOC-107 |
| docs/api/gpu-spatial-analysis-cross-reference.md:87 and docs/api/gpu-dataframe-analysis.md:1182 | `GPURipley` "in a rectangular window" only | rectangular (default) or a valid polygon window prepared with `GPUStudyWindowPreparation` and `bounds: 'study-window'` | `pair-statistics/gpu-ripley.ts:104-115`; the page's own `:1032,1117` | DOC-107 |
| docs/api/gpu-dataframe-analysis.md:389 | rational-rank quantiles: "up to one storage binding, 2^25 rows at the default limit" | at most 2^31 - 1 rows in total; one packed `values` view is bound by one storage binding, chunked `values` span several; 2^24 still applies with a filter output | `column-classification/gpu-column-quantiles.ts:190-206` (`MAXIMUM_RATIONAL_ROW_COUNT`) | DOC-104 |
| docs/api/pointcloud-analysis.md:125,133 | "Initial operations accept one resident packed view"; all eleven "take non-owning packed views" | names the classes with aligned chunk support: Height (queries and aligned outputs), Features (all but `positions`), Ground (point rows and outputs; raster packed), VoxelMerge (every voxel-aligned field) | `operations/gpu-pointcloud-height.ts:10-26`, `-features.ts:96-102`, `-ground.ts:153`, `-voxel-merge.ts:82-95`; the other seven use `validateSinglePackedView` only | DOC-104 |
| dev-docs/coverage-matrix.md:254 (IN01) | "Resident variogram fitting is slower than CPU on small inputs" | the serial fit was slower; the cooperative fit measured 0.7 ms against 2.3-2.5 ms; small fits show no consistent winner; links the route and transfer boundary | `docs/api/gpu-spatial-analysis/interpolation-expansion.md:160` (the measurements) | DOC-109 |
| docs/api/analysis-core/fitting-kernels.md:14 | fitting limits listed without a row ceiling | `GPUFittingGradientSums`, `GPUFittingMinibatchOrder` and `GPUFittingCoordinateEncoder` accept at most 2^24 rows (`GPU_FITTING_MAXIMUM_ROW_COUNT`) | `kernels/fitting-utils.ts:27`; `gpu-fitting-gradient-sums.ts:77`, `gpu-fitting-minibatch-order.ts:74`, `gpu-fitting-coordinate-encoder.ts:154` | CORE-111 |
| dev-docs/upstream-workarounds.md (EC03 row, status word `Partly` kept) | "solver core replaceable since alpha.16" | published `GPUJacobiPCG` unrolls every iteration: 51, 85, 153 nodes at budgets 1, 2, 4 (`17 + 34 * iterations`), 34,833 at 1,024, over the local 32,768 budget; local loop uses five nodes per iteration with a true-residual certificate; warm start is not Krylov continuation; budgets must be specified before migrating | report probe (lead-repeated node count); `node_modules/@luma.gl/graph/dist/gpu-core/gpu-pcg.js:120-175` unrolls per iteration | LUMA-108 |
| dev-docs/upstream-workarounds.md (`GPUHistogram` row, :260) | gap "Domain/output offsets within larger views and NaN domain meaning finite extent" | upstream already supports offsets, masks and explicit edges; gap is equal-width edge arithmetic and the NaN-domain meaning (as the summary at `:62` already said) | `gpu-histogram.d.ts:24-58`; the register's own `:62` | LUMA-109 |
| dev-docs/upstream-workarounds.md (Louvain row, :268) | `GPUNetworkLouvain` "(at most 32 vertices)" | dense route 32 vertices; sparse route up to 1,073,741,823 nodes and edges | `network-optimization/gpu-network-louvain.ts:43`; `network-louvain-sparse.ts:50-53` | LUMA-109 |
| dev-docs/upstream-workarounds.md (similarity-search row, :284) | classifier "at most 4,096 training rows"; "unused slots wrongly vote for sample zero"; extra `pixelCount * k * 8` bytes "(64 MiB ...)" | tiled default accepts 4,294,967,295 training elements, 4,096 is the explicit brute-force limit; only filled neighbours vote; scratch is two `pixelCount * k * 4`-byte bindings (128 MiB in total at 1,048,576 pixels and k=16, the old arithmetic gave half) | `raster-embeddings/gpu-raster-embedding-knn-classifier.ts:42-49,100-112,153-163` | LUMA-109 |
| dev-docs/upstream-workarounds.md (:317, :323) | SMRF "radius-at-most-8 cross passes"; scan-statistic "inclusive `GPUScan` plus a local total node" | repeated radius-one passes to erosion/dilation; saturating uint64-backed count scans (`createSaturatedCountScanNodes` over `GPUScanUint64`) with inclusive conversion | `pointcloud-analysis/operations/pointcloud-diamond-morphology.ts:11,22`; `analysis-core/kernels/` saturated scan helper; `scan-statistics/gpu-spatial-scan-statistic.ts:1090-1115` | LUMA-109 |
| dev-docs/upstream-proposals.md:347 | request subgroup paths for `GPUScan`/`GPUSort` | unsegmented `GPUScan` and `GPUSort` already select subgroup paths when device and WGSL support them (segmented scan stays portable) | `gpu-scan.js:11-16,403,508`; `gpu-sort.js:12,221-252` | LUMA-111 |
| dev-docs/upstream-proposals.md:155 | proposal "retires stencils in `GPUTerrainCurvature`" | at most the finite-difference stencils; selectable estimators (default `'evans-young'`), Hessian and curvature formulas stay local | `terrain-curvature/gpu-terrain-curvature.ts:221,238-246,410-425` | LUMA-111 |
| docs/api/gpu-terrain/height-above-drainage.md:46 | cycles named only for fractional sweeps | D8 path entering a receiver cycle (caller directions) is `UNRESOLVED` with `converged` 0 | `hydrology/gpu-terrain-height-above-drainage.ts:58-64,168-175` | fix-terrain-hydrology record |
| dev-docs/release-package.md:56-60 (rewrites the batch 2 TOOL-102 sentence) and :85 | the checker "does not resolve the relative references of the other declarations" | it resolves every relative reference in every packed declaration under NodeNext and type-checks `.mts` and `.cts` consumers; release records carry the tarball's SHA-256 and byte length and promotion refuses a different archive | `scripts/check-packed-consumer.mjs:203-275,298-316`; `scripts/build-release.mjs:95-122,283` | fix-package-build record |
| AGENTS.md:11 | `yarn check-packed-consumer` described as "isolated packed consumer" only | adds the NodeNext reference resolution and the ES module and CommonJS consumers | same | fix-package-build record |

Count: 24 rows.

### Edited although the staging pass owns it

`docs/api/gpu-network/operators.md` (the one permitted exception; hand-merge into the clone). Text from `fix-network.md`, `fix-network-geometry.md` and `fix-network-completeness.md`, each verified against the working-tree source (`gpu-network-accessibility.ts:47,130-149,244-256`, `gpu-network-cost-weights.ts:128-161`, `gpu-network-k-function.ts:106,380`, `network-isochrones-parameters.ts:18-22,74-91`). Six hunks, no generated block touched: the cost-matrix plus accessibility example now requests `status.converged` and `status.complete` and ends with the three-cause comment (`:683-690`); a sentence under `GPUNetworkCostMatrix` naming the three causes of `+Infinity`; `costsComplete`/`costLimit` text under `GPUNetworkAccessibility` and `GPUNetworkCostWeights`; the 92,682-event limit under `GPUNetworkKFunction`; the finite positive-extent rule in the isochrones "Parameters" bullet. A copy of the page before editing is in the scratchpad (`operators.md.before`).

**The spec still fails.** `doc-parameters.node.spec.ts` fails because the generated blocks of this page are out of date with the new props, not because of prose. Prose cannot fix it and the blocks must not be hand-edited: run `UPDATE_DOC_PARAMETERS=1 npx vitest run --project node modules/analysis/test/analysis-core/doc-parameters.node.spec.ts` once (not run here, rule 7). That will also regenerate any other page the new props touch.

### Deferred to the staging copy

All verified against source first. "Clone" is a phrase grep of the staging copy.

| Document | Correction | Staging copy |
| --- | --- | --- |
| docs/api/gpu-network/expansion.md:39-42 | Defect in both checkouts: "work, for at most 1024 nodes and 65536 slots. For larger graphs, use" is followed by a second copy of the line ("...For larger directed graphs, use"). Delete the first. NET-119 also: the guide misstates tight zero-cost paths (`:15`) and the structure route limits; the report names `S/flow-aggregation/gpu-flow-aggregation.ts:321` (the `output.ids` row/key switch above 65,535 zones: lowest accepted source row; use `flowOriginZoneIds`/`flowDestinationZoneIds`) for the same decode sentence in `operators.md:390` | Clone has the same duplicated line at `:39-42` |
| docs/api/gpu-network/network-time-geography-mob09.md:35-38 | Percentile rank: ranks are exact for the percentile's shortest decimal (`String(p)`), 99.9% of 1,000 selects 999; replacement text in `fix-network.md` under NET-102 | Clone: no `String(p)`; still wrong |
| docs/api/gpu-network/operators.md:390 | The components/strong-components implementation and structure route limits (NET-119) | Not checked per line |
| docs/api/gpu-raster/operations-analysis.md | `parameterLayoutVersion` defaults to 2; default view must hold exactly seven words; version 1 is explicit and padding-blind; length never selects a layout; packer target shorter than seven words writes the legacy prefix (`fix-raster-terrain.md`). Also RT-138: v1 still described as the zonal default (`:130,312`), legacy-infinity blur handling (`:525`) | Clone: none of the wording present |
| docs/api/gpu-raster/expansion.md:47 | RT-138: stale reprojection precision description in the raster expansion table (the C23 row says "Float32 projection path only" and then that same-CRS origins are differenced in float64); the report gives no replacement and the row was not re-derived | Not checked per line |
| docs/api/gpu-terrain/operators.md (~:177 and SLRM ~:1626) | 15 constructors reject a sheared `grid.affine`; rotated orthogonal grids stay supported in `'uniform'` mode (`fix-raster-terrain.md`). SLRM anchor subtracted exactly in 64-bit integers; invalid if `\|z / quantum\|` reaches 2^62 (`fix-terrain-hydrology.md`) | Clone: "quantized value exactly" absent; "sheared" occurs twice, not compared |
| docs/api/gpu-spatial-analysis/operators-cells.md | `sumOverflow`, saturation constants, neighbour rule by squared chord length, `candidateCount` saturation, longitude reduced modulo 360 (`fix-cells-a.md`, `fix-cells-b.md`) | Clone: `sumOverflow`, "squared chord", "modulo 360" absent; `candidateCount` mentioned 3 times, unchecked |
| docs/api/gpu-spatial-analysis/global-coordinates.md:186 | rings in either orientation (`fix-cells-b.md`) | Clone: absent |
| docs/api/gpu-spatial-analysis/operators-weights.md (`GPUMapColoring`) | incomplete adjacency leaves every row `GPU_MAP_COLORING_UNCOLORED`, counts 0 (`fix-geometry.md`) | Clone mentions the constant only for unfinished rows, not the loss rule |
| docs/api/gpu-spatial-analysis/operators-geometry.md | grid generator index limits (2^24, 2^23 - 1, 8,388,607 by 5,592,405) and compact-only status of `GPUOffsetCurve`/`GPUVertexSnap` (`fix-geometry.md`, `fix-api-catalog.md`) | Clone has one `layout: 'compact'` mention; limits not checked |
| docs/api/gpu-spatial-analysis/operators-accessibility.md | `GPUCatchmentAccessibility` `costsComplete` (`fix-network-completeness.md`) | Clone: `costsComplete` absent |
| docs/api/gpu-spatial-analysis/operators-statistics.md | location-allocation `costsComplete`/`costLimit`; neighbourhood summary, emerging hot spots and local outlier factor loss wording (`fix-network-completeness.md`, `fix-points-weights-loss.md`, `fix-stats-weights.md`) | Clone: `costsComplete` absent; loss wording not compared |
| docs/api/gpu-raster/raster-time-series.md | temporal resample: exact observation returns status 0, fractions clamped to [0, 1] (`fix-raster-timeseries.md`) | Clone: absent |
| docs/api/gpu-raster/raster-embeddings.md, operations-analysis.md `:316`, expansion.md `:230` | still the batch 1 rows (RT-107, RT-104, RT-106); unchanged | as in batch 1 |

### Generated blocks and source comments

No edit touched a `{/* parameters: ... */}` block. The `operators.md` blocks need regeneration (see above). These belong in source.

| Where | Fix | Finding |
| --- | --- | --- |
| `raster-analysis/particle-advection/gpu-particle-advection.ts:135` TSDoc | names `@luma-contributors/analysis/experimental/simulation`, which does not exist; the standalone class is `FlowParticleSimulation` from `@luma.gl/graph/experimental/simulation` | LUMA-106 |
| `analysis-core/contracts/status.ts:42,108` | a count-of-rejected-rows field also carries a solver code in `spatial-regression-recipe.ts:383,706`; expose a typed `code` | SEAM-110 |
| `gpu-network/network-subgraph-filter/gpu-network-subgraph-filter.ts:56`, `adjacency-matrix/gpu-adjacency-matrix.ts:479` | state that offsets describe the required adjacency beyond stored neighbours after overflow | NET-116 |
| `network-reachability-passes.ts:416`, `network-service-areas-passes.ts:270`, `alternative-routes/gpu-alternative-routes.ts:305` | per contributor: unit of each budget, which phases a completion flag certifies, usable part of a stopped result | NET-104 |
| `network-accessibility/cpu-network-cost-matrix-operation.ts:39`, `network-centrality/cpu-network-centrality-operation.ts:55` | name the exact-only CPU operations as exceptions to explicit budgets | NET-115 |
| `raster-analysis/raster-relaxation.ts:762` and zonal TSDoc | batch 1 items still open | RT-104, RT-106 |

### Not applied

| Item | Reason |
| --- | --- |
| CORE-112 class comment | Already correct in the working tree (`gpu-contributor-operation.ts:81-85` assigns the one-chunk rule to the local lowering); `contributor-contract.md:174` already separates the layers |
| CORE-110 | Policy (which bounded-loss producers must publish completeness when status is omitted), not a false statement |
| CORE-111 | Applied. The report also names the shared guide's absence of a subject/condition/limit table: not built |
| DOC-106 | The CHANGELOG migration omits outputs (`output.trips.status.candidateOverflow`, conditional `output.triplegs.status.candidateOverflow`); an omission, not a false statement (rule 3). The constructor checks are at `gpu-trajectory-segments.ts:233-248`. The claim about mobility indicators suppression was not verified |
| DOC-108 | No canonical anchor for 289 classes: a structural proposal |
| NET-102 in CHANGELOG ("exact nearest rank") | Still true of the decimal reading; the guide page is deferred |
| NET-104 (units and flag scope), NET-105, NET-115 | Source comments or roadmap text (see source table) |
| RT-109, RT-135, RT-137, RT-133 | No page states the contradicted fact, or the report requires a GPU probe first (RT-133 wording at `raster-eof-analysis.md:93` not changed) |
| LUMA-106 grid formulas, LUMA-107 | Plan sentence already states the helper (`upstream-first-plan.md:15`); LUMA-107 text was applied in batch 2 |
| fix-stats-regression wanted text (probit tolerance, GWR translation invariance, LM serial sums) | Each is conditional on a fix that has not landed ("once SSR-107/110 lands"), or its page was not identified; not applied |
| fix-dataframe-numerics "-0 sorts below +0" | Waits on a decision (DPC-110) |
| consumer-seams catalog entries (SEAM-111 to SEAM-115, LENS-103, LENS-104, LENS-108) | Website, lenses and kepler text, outside the allowed files |
| area-direction, area-spatial-* reports | Not part of this batch |

### Proposals not applied

- **Haversine proposal (3)**: the report says the fork candidate lacks the claimed seam-safe subtraction; the fork is not in this tree and `geodesicLongitudeDelta` in this repository does it with integer units, so the proposal text was left until the fork copy is read.
- **Upstream-workarounds register**: the report's remaining row dispositions (rows 173-289 keep/narrow/retire) were not applied; only rows the report called wrong were rewritten. The retired list at `:315-325` and `:29` summary should inherit the PCG correction when the lead reconciles them.
- **CHANGELOG DOC-106**: add an old/new output map for the trajectory segment outputs when the lead aggregates today's entries.
- **Report-level proposals** (NET-104 budget glossary, DOC-108 class anchors, CORE-110 completeness policy): leave to the roadmap owner.
