# Library review: recipes and applied-analysis families (2026-11-09)

Domain reviewer report. Scope: `gpu-spatial-analysis/recipes`, `gpu-network/recipes`, and the
`gpu-spatial-analysis` families `urban-morphology`, `landscape-ecology`, `location-allocation`,
`catchment-accessibility`, `map-coloring`, `dorling-cartogram`, `segregation`, `rate-smoothing`,
`outlier-detection`, `geographic-distribution`, `study-window`, `point-process`,
`spatial-validation`, `spatial-interaction`, and `ontology`. Paths are relative to
`modules/analysis/src`. Read-only review; nothing was built or run except a read-only Node script
that evaluates the ontology and the exported symbols of each entry point (TypeScript checker).

## 1. Scope and method

About 31,000 lines in 104 files (recipes 9,402; network recipes 1,458; urban 1,313; landscape
1,914; location-allocation 2,206; point-process 1,777; spatial-validation 1,050; ontology 3,141;
the remaining families 8,400).

**Read in full:** `recipes/recipe-utils.ts`, `recipes/expansion-recipe-utils.ts`,
`recipes/index.ts`, `recipes/surface-preparation-recipe.ts`; every file in `urban-morphology` and
`landscape-ecology`; `point-process/gpu-point-pattern-envelope.ts`,
`gpu-point-pattern-rank-envelope.ts`, `translation-point-pattern.ts`, `point-process-utils.ts`,
`polygon-overlap-metric.ts`, `polygon-translation-overlap.ts`;
`spatial-validation/spatial-validation-design.ts`, `gpu-spatial-cross-validation.ts`.

**Read in part (named line ranges):** `point-process/gpu-inhomogeneous-point-pattern.ts` (80-279),
`gpu-spatial-balanced-sample.ts` (220-384), `gpu-continuous-spatial-sample.ts` (160-287),
`spatial-validation/gpu-nested-interpolation-validation.ts` (100-339),
`recipes/quantile-lisa-recipe.ts` (60-200), `recipes/dasymetric-redistribution-recipe.ts` (84-116,
337-400), `recipes/rate-mapping-recipes.ts` (68-88), `location-allocation/gpu-location-allocation.ts`
(84-134, 286-330, plus grep of the loop structure), `gpu-bounded-facility-allocation.ts` (1-80),
`catchment-accessibility/gpu-catchment-accessibility.ts` (20-80, 225-300),
`outlier-detection/gpu-local-outlier-factor.ts` (60-110, 290-330),
`spatial-interaction/gpu-spatial-interaction.ts` (20-78, 240-347),
`study-window/gpu-study-window-preparation.ts` (1-80, 415-535), `segregation/gpu-segregation.ts`
(1-186), `ontology/spatial-analysis-ontology.ts`, `ontology/spatial-analysis-operators.ts` (1-80,
864-935).

**Skimmed by grep only (dispatch shapes, limits, status, imports):** the other 20 recipe files, the
three network recipe files, `gpu-huff-trade-areas.ts`, `gpu-map-coloring.ts`,
`gpu-dorling-cartogram.ts`, both rate-smoothing classes, `gpu-geographic-distribution.ts`,
`gpu-rose-statistic.ts`, `gpu-spatial-cross-validated-ols.ts`, `location-allocation-sparse.ts`,
`facility-score-reduction.ts`, `ontology/spatial-analysis-catalog.ts`,
`coverage-expansion-catalog.ts`, `kepler-analysis-catalog.ts`.

**Not opened:** `outlier-detection-parameters.ts`, `geographic-distribution-kernels.ts`,
`geographic-distribution-parameters.ts`, `grouped-sum-reduction.ts`, `segregation-layout.ts`,
`spatial-validation-descriptors.ts`, the bodies of `hot-spot-analysis-recipe.ts` and
`space-time-hot-spots-recipe.ts`. Verdicts for grep-only files are marked (s).

Ontology numbers come from evaluating `GPU_SPATIAL_ANALYSIS_OPERATORS` and
`GPU_SPATIAL_ANALYSIS_CAPABILITIES` directly, and from `checker.getExportsOfModule` on each of the
nine entry `index.ts` files (script: scratchpad `ontology-audit.cjs`).

## 2. Contributor table

61 symbols: 39 `add*Recipe` functions (all 39 in the library are in this scope) and 22 classes.
Abbreviations: P = parallelism, It = iteration, Par = parameters, Ctx = context, Val = validation.
"builder" means the recipe uses `RecipeBuilder` and returns `GPURecipeResult` with stage status.

### 2.1 Builder recipes (`recipes/`, `gpu-network/recipes/`)

These compose published or library contributors. P, Work and limits are those of the composed
contributors; the recipe adds glue kernels. Inputs packed; Status `GPURecipeStatusPort` stages;
Val `assertRecipe` / `throw` with messages unless noted.

| Symbol | File (lines) | Computes | Glue kernels | Par | Ctx | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| addHotSpotAnalysisRecipe (s) | hot-spot-analysis-recipe.ts (1420) | ArcGIS Optimized Hot Spot | row | buffer (`GPUParameterBuffer`) | spatialContext | conforming |
| addRateClusterMapRecipe (s) | rate-cluster-map-recipe.ts (594) | EB rate + LISA map | row | buffer | none, weights only | conforming |
| addSpatialRegressionRecipe (s) | spatial-regression-recipe.ts (608) | OLS + LM diagnostics + decision | one scalar node `:158` | buffer | spatialContext | conforming |
| addSpaceTimeHotSpotsRecipe (s) | space-time-hot-spots-recipe.ts (806) | ArcGIS emerging hot spots | row | buffer | spatialContext | conforming (TC-03 tracked) |
| addPeriodComparisonRecipe (s) | period-comparison-recipe.ts (359) | before/after zonal change | none | inherited | none | conforming |
| addPointsInPolygonsChoroplethRecipe (s) | points-in-polygons-choropleth-recipe.ts (499) | count/sum per polygon + colour | none | inherited | spatialContext | conforming |
| addClusterAndOutlineRecipe (s) | cluster-and-outline-recipe.ts (481) | DBSCAN + hull + measures | serial group loop `:417-421` | inherited | spatialContext | align |
| addChangeOfSupportRecipe (s) | change-of-support-recipe.ts (539) | areal weighting | none | inherited | spatialContext | conforming |
| addStraightLineCatchmentsRecipe (s) | straight-line-catchments-recipe.ts (225) | buffer catchment stats | none | inherited | spatialContext | conforming |
| addFleetDwellRecipe, addFleetDwellZoneEventsRecipe (s) | fleet-dwell-recipe.ts (714) | stops, zone dwell | none | inherited | spatialContext | conforming |
| addZonalDistributionRecipe (s) | zonal-distribution-recipe.ts (192) | zone histograms | row | inherited | spatialContext | conforming |
| addRasterZonalDistributionRecipe, addExposureByZoneRecipe, addRegionalRasterChangeRecipe (s) | raster-summary-recipes.ts (354) | raster zonal summaries | row | inherited | raster grid | conforming |
| addCellFlowTopologyRecipe (s) | cell-flow-topology-recipe.ts (294) | DGGS flow topology | row | inherited | cell family | conforming |
| addSpatialRateRecipe (s) | rate-mapping-recipes.ts (178) | spatial rate | row | inherited | weights | conforming |
| addExcessRiskRecipe | rate-mapping-recipes.ts:68-88 | wraps one `GPUEmpiricalBayesRates`, `stages: []` | none | inherited | none | merge (into `GPUEmpiricalBayesRates`) |
| addInterpolationDiagnosticsRecipe (s) | interpolation-diagnostics-recipe.ts (210) | kriging + CV summary | serial summary `:157-161` | baked | spatialContext | align |
| addSurfacePreparationRecipe | surface-preparation-recipe.ts (150) | fill no-data + rank filter | serial count over every cell `:120-130` | inherited | raster grid | align |
| addDrainageNetworkRecipe (s) | drainage-network-recipe.ts (194) | flow, streams, catchments | serial count over every cell `:95-100` | inherited | raster grid | align |
| addLocalVariabilityRecipe (s) | local-variability-recipe.ts (169) | LOSH + local Moran | serial count `:128-132` | inherited | weights | align |
| addDasymetricRedistributionRecipe | dasymetric-redistribution-recipe.ts (502) | dasymetric mapping | three serial evidence nodes `:337-400`; bare `assert` `:93-103` | baked | raster grid | align |
| addDriveTimeCatchmentRecipe (s) | drive-time-catchment-recipe.ts (808) | network service areas + bands | row | inherited | spatialContext | conforming |
| addNetworkDisruptionRecipe (s) | network-scenarios-recipe.ts:45 | baseline vs scenario routing | row | inherited | none | conforming |
| addFacilityScenarioRecipe (s) | network-scenarios-recipe.ts:279 | facility scenario summary | serial demand loop `:383-387` | inherited | none | align |
| addTrackQualityRecipe (s) | track-quality-recipe.ts (207) | trajectory QA | row | inherited | spatialContext | align (FX-08 tracked) |
| GPUQuantileLISA | recipes/quantile-lisa-recipe.ts (332) | GeoDa quantile LISA | composes `GPUColumnQuantiles` + permutation; scalar nodes `:150`, `:177` | buffer (resident params) | weights | move (class in `recipes/`; belongs in `spatial-autocorrelation`) |

### 2.2 Expansion-generation recipes (`urban-morphology`, `landscape-ecology`)

All twelve: no `RecipeBuilder`, no `contributors`, no stage status, outputs created with
`getRecipeView(graph, id, format, length)` and no caller override, so every output is a graph-owned
transient view (CONFIRMED, e.g. `landscape-metrics.ts:53-59`, `isovists.ts:57-111`; the test file
adds a copy kernel to read them). Par = baked for every numeric prop. Inputs packed only. Ctx none.
Val bare `assert`.

| Symbol | File (lines) | Computes (reference) | P | Work | It | Hard limits | Status | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| addBuildingTessellationRecipe | building-tessellation.ts (324) | momepy morphological tessellation | serial sampling `:81-103`; row clip `:104-149`; serial pack per building `:161-178` | O(s²·v) clip, brute force; one `GPUPolygonOverlay` dissolve per building | none | 64 buildings `:46`, enclosure 32 vertices `:48`, 128 samples `:49` (design) | ad hoc `uint32x4` + `complete` | rework |
| addBuildingAdjacencyRecipe | building-adjacency.ts (200) + building-structures.ts (146) | momepy shared walls, alignment, structures | row over n² pairs, each scanning all runs `:88-112`; serial union-find `:139-150`; per-building overlay `building-structures.ts:48-144` | O(n²·runs) | none | 1,024 buildings `:34`; 64 with structures `building-structures.ts:28` (design) | `complete` scalar | rework |
| addIsovistsRecipe | isovists.ts (170) | depthmapX / momepy isovists | row per ray `:112-137` | O(V·R·S), no index; insertion sort per viewpoint `:80` | none | 65,536 rays `:43`; 4,096 with vertex sampling `:37` (design) | per-ray validity | rework |
| addVisibilityGraphRecipe | visibility-graph.ts (191) | depthmapX VGA (integration, choice) | row per pair `:68-87`; serial edge compaction `:88-102`; per-source Brandes in private arrays `:103-131` | O(n²·S) + O(n³) | none | 256 points `:32` (design) | ad hoc `uint32x4` | rework |
| addStreetProfilesRecipe | street-profiles.ts (188) | momepy street profile | row per sample `:56-86`; frontage O(B·N) `:104-116`; facades O(S²) per thread + insertion sort `:135-158` | O(N·S) | none | facades 1,024 segments `:121` (design) | per-row flags | rework |
| addLandscapeMetricsRecipe | landscape-metrics.ts (240) + reconcile-patches.ts (57) | landscapemetrics area, perimeter, cohesion, AI, contagion, mesh | row per patch, each scanning every cell `:70-100`; class pairs scanning every cell `:102-119`; serial landscape `:152-214`; serial union-find over raster `reconcile-patches.ts:26-41` | O(P·N + C²·N) | none | 256 classes `:45` (design) | none | rework + merge (`raster-analysis/raster-patches/gpu-raster-patch-metrics.ts`) |
| addPatchProximityRecipe | patch-proximity.ts (188) | FRAGSTATS core area, ENN, proximity | separable EDT per patch `:98-123`; O(P²·N) pair distances `:124-137`; O(P²·N) area recount in summary `:155-163` | O(P·N·(W+H) + P²·N) memory P·N | none | none stated; P·N floats `:59-66` hit the binding limit | none | rework |
| addResistanceConnectivityRecipe | resistance-connectivity.ts (137) | Circuitscape pairwise, dense | serial: whole CG in one invocation `:46-100` | O(it·E) | in-kernel loop | 256 vertices `:34`, 65,536 edges `:37` | ad hoc `uint32x4` | merge (into sparse; delete) |
| addSparseResistanceConnectivityRecipe | sparse-resistance-connectivity.ts (340) + system (267) + solver (354) | Circuitscape pairwise effective resistance | row; deterministic tree reductions | O(it·E); CSR by published `GPUSort` + `GPUCOOToCSR` | unrolled, 5 nodes/iteration, GPU dispatch gate | 1,048,576 vertices, 2,097,152 edges `:24-25` (inherent-ish), 4,096 iterations `:26` (design: no continuation) | ad hoc `uint32x4` `[converged, iterations, reachable, invalid]` | align |
| addSegregationDecompositionRecipe | segregation-decomposition.ts (161) | Shapley decomposition of D (PySAL) | serial: four one-thread index nodes `:40-63`; row per replicate `:96-123` | O(n) serial, O(R·n) | none | 65,536 replicates `:90` | validity words | merge (D from `GPUSegregation`) |
| addSegregationCounterfactualRecipe | segregation-counterfactual.ts:21 | PySAL counterfactual composition | serial: one thread does insertion sort + O(n²) rank `:35-69` | O(n²) | none | 2,048 units `:27` (design) | inherited | rework |
| addPopulationSupportAlignmentRecipe | segregation-counterfactual.ts:97 | exact ID join | row O(n·m) `:115-128`; serial O(n·m) certificate `:130-142` | O(n·m) brute force | none | none | ad hoc `uint32x4` | rework (use `GPUKeyJoin`) |

### 2.3 Classes

Status column: "words" means a positional `uint32` status view with documented meaning but not the
named shared ports.

| Symbol | File (lines) | Computes | P | Work | It | Hard limits | Par | Ctx | Status | Val | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPUSegregation | segregation/gpu-segregation.ts (791) | D, H, D_g, Atkinson, isolation, interaction, local terms, multiscale | cooperative, fused CSR pass + workgroup trees | O(nnz·K + n·K²) | none | 16 groups `:50` (design, fine) | baked (`atkinsonInequalityAversion`, `selfWeight`) | weights | none | helpers | conforming |
| GPUCatchmentAccessibility | gpu-catchment-accessibility.ts (358) | 2SFCA, 3SFCA | row over CSR | O(nnz) | none | none | n/a | context compatibility check | none | helpers | conforming |
| GPUHuffTradeAreas (s) | gpu-huff-trade-areas.ts (372) | Huff model | row over CSR | O(nnz) | none | none | (s) | (s) | none | helpers | conforming |
| GPUMapColoring (s) | gpu-map-coloring.ts (481) | Jones-Plassmann colouring | row | O(it·nnz) | unrolled | iterations | buffer | weights | `GPUIterationStatusPort` | helpers | conforming |
| GPUDorlingCartogram (s) | gpu-dorling-cartogram.ts (662) | Dorling | row, `GPUGridIndex` | O(it·n·k) | unrolled ≤256 `:127` | 65,536 rows `:40` | (s) | spatialContext | words | helpers | align (KG02 tracked) |
| GPUEmpiricalBayesRates, GPUSpatialEmpiricalBayesRates (s) | rate-smoothing/ (437, 291) | global and spatial EB | cooperative / row | O(n), O(nnz) | none | none | (s) | weights | none | helpers | conforming |
| GPULocalOutlierFactor | gpu-local-outlier-factor.ts (345) | LOF over kNN CSR | row | O(n·k) | none | none | buffer | spatialContext | count (atomic) | helpers | conforming (PLAT-03 tracked) |
| GPUGeographicDistribution, GPURoseStatistic (s) | geographic-distribution/ (1414, 398) | mean/median centre, SDE, standard distance, rose | segmented sorted trees, `GPUSort`, `GPUScan` | O(n log n) | Weiszfeld unrolled ≤256 | 360 bins | (s) | spatialContext | words | helpers | conforming |
| GPUStudyWindowPreparation | study-window/ (722) | polygon window, sampler raster | row; serial ring summary `:418-440` (ring count only) | O(V + cells) | none | 1,048,576 sampler cells | baked | spatialContext | flags | helpers + 2 bare asserts | conforming |
| GPUInhomogeneousPointPattern | gpu-inhomogeneous-point-pattern.ts (281) + translation (228) | spatstat Kinhom, pcfinhom | row intensity O(n) each `:198`; border K: one thread per radius doing O(n²) `:220-260`; translation: one thread per pair with O(E²) overlap `translation-point-pattern.ts:52-89` | O(R·n²) border; O(n²·E²) translation | none | 4,096 events, 256 radii `:88`; 1,048,576 edge pairs `:124` (design) | baked (bandwidth) | studyWindow, planar check `:89-92` | per-radius code | bare assert | rework |
| GPUPointPatternEnvelope | gpu-point-pattern-envelope.ts (235) + rank (108) | spatstat envelope, global MAD/DCLF, extreme rank | row per curve; serial global test `:182-207`; rank: O(S·count·S) per curve `rank:38-47`, O(S²) per ordinate `rank:96-105` | O(S·count), O(S²·count) | none | 4,096 ordinates, 4,096 curves `:90-95`; (S+1)²·count ≤ 2^24 `:119` (design) | baked | none needed | word | bare assert | align (overlaps `GPURipleyEnvelope`) |
| GPUContinuousSpatialSample | gpu-continuous-spatial-sample.ts (288) | spsurvey-style systematic sample on a line or area | serial: one invocation per wave loops every candidate and every triangle `:183`, `:266-278` | O(candidates·triangles) per thread | none | 4,096 frame vertices `:91`, 65,536 capacity, 1,048,576 candidates `:102-106` (design) | buffer (seed, replicate) | none | words | bare assert | rework |
| GPUSpatialBalancedSample | gpu-spatial-balanced-sample.ts (386) | GRTS (spsurvey) | row quadtree codes `:231-263`; O(n²) rank by counting `:264-283`; serial selection per stratum `:288-351` | O(n²) | none | 4,096 rows `:84`, 256 strata `:85` (design) | baked | none | words | bare assert | rework |
| GPUSpatialCrossValidation | gpu-spatial-cross-validation.ts (200) + design (273) | blockCV folds, buffer exclusion, NNDM, kNNDM, environmental | row assign; mask O(F·n²) brute force `:137-175`; every design is one serial invocation `design:262-270` (k-means, NNDM, kNNDM) | O(F·n²); designs O(n²·…) serial | serial in-kernel | 65,536 rows `:69`, 64 folds `:70`; designs 256 rows `design:75`, NNDM 64 `design:89` (design) | baked | none (planar assumed) | per-fold bits | helpers | rework |
| GPUSpatialCrossValidatedOLS (s) | gpu-spatial-cross-validated-ols.ts (188) | fold-wise OLS | composes CV + `GPUOrdinaryLeastSquares` | per fold | none | 15 predictors `:60` | n/a | none | words | helpers | conforming |
| GPUNestedInterpolationValidation | gpu-nested-interpolation-validation.ts (342) | nested CV model selection (kriging, RBF) | composes; serial checks `:137-147`, `:291-305`, `:322-336` | F·C·(I+1) fits | unrolled in TS: up to 16·16·9 = 2,304 fit instantiations `:130-285` | 256 rows, 16 folds, 8 inner, 16 candidates `:71-75` | baked | none | per-fold bits | helpers | align |
| GPUSpatialInteraction | gpu-spatial-interaction.ts (348) | Poisson gravity, singly/doubly constrained (spint) | serial: one thread per coefficient candidate runs the whole IPF `:180`; serial select `:282` | O(it·pairs) per thread | in-kernel + golden-section unrolled `:252-276` | 4,096 pairs, 128 origins, 128 destinations, 128 candidates `:98-101` (design) | baked | none | word | helpers | rework |
| GPULocationAllocation | gpu-location-allocation.ts (1540) | p-median, MCLP, LSCP; Teitz-Bart swap | row + cooperative reductions | O(F·D) per greedy step and per swap | unrolled with dispatch gate: greedy `:1156`, swaps `:672` | storage-binding limits `:84-131` (inherent); ≤1,000 swaps `:71`; no node-count guard | buffer for costs | none | words incl. converged / iterationLimitReached | helpers | conforming |
| GPUBoundedFacilityAllocation | gpu-bounded-facility-allocation.ts (310) | exact p-center, capacitated p-median | one workgroup per subset; capacitated assignment serial per subset | O(C(F,p)·D) exhaustive | none | 32 facilities, 8 selected, 65,536 subsets, 4,096 demand `:61-79` (design, stated) | baked | none | words | helpers | align (exact small oracle; say so in the name or docs) |

**Verdict counts (61 rows, grouped rows counted per symbol):** conforming 30, align 13, rework 14,
merge 3, move 1.

## 3. GPU quality findings

Each item: what is there, why it matters, the better design, effort.

**Q1. Brute-force ray casting with an unrolled loop over every obstacle segment. CONFIRMED.**
`urban-morphology/urban-ray-utils.ts:17` loops `${segmentCount}` segments per ray; isovists cast
V·R rays (`isovists.ts:112-137`), visibility tests n² pairs (`visibility-graph.ts:68-87`), street
profiles cast 2 rays per sample (`street-profiles.ts:56-86`), and each kernel also re-checks every
segment for finiteness per ray (`isovists.ts:132`, `visibility-graph.ts:83`). The segment cap is
2^24 (`isovists.ts:41`), so the asserts allow work of V·R·2^24. The library already builds a
Morton-sorted segment BVH on published `GPUBVH`
(`segment-intersection/segment-bvh.ts`). Better: one finiteness pass, a segment BVH or uniform-grid
DDA, then traversal per ray. Effort M.

**Q2. Serial single-thread kernels doing the core algorithm. CONFIRMED.** One invocation loops over
all data in: tessellation boundary sampling (`building-tessellation.ts:81-103`) and per-building pack
(`:161-178`); adjacency union-find over n² (`building-adjacency.ts:139-150`); visibility-graph edge
compaction (`visibility-graph.ts:88-102`); landscape reconcile over the whole raster
(`reconcile-patches.ts:26-41`); landscape summary (`landscape-metrics.ts:152-214`); the complete
dense CG (`resistance-connectivity.ts:46-100`); dissimilarity (`segregation-decomposition.ts:40-63`);
counterfactual rank (`segregation-counterfactual.ts:35-69`); every cross-validation design
(`spatial-validation-design.ts:262-270`, including a serial k-means `:212-226`); continuous sample per
wave (`gpu-continuous-spatial-sample.ts:183`); spatial interaction per candidate
(`gpu-spatial-interaction.ts:180`). Better designs: `createMaskedWeakComponentNodes`
(`analysis-core/kernels/masked-weak-components.ts:34`) or published `GPURasterConnectedComponents`
for union-find; `GPUCompaction` for edge lists; `GPUReduction` / tree sums for totals; `GPUKMeans`
for the environmental design; `GPUSort` for ranks; IPF as row-parallel margin passes (one
dispatch per origin margin and one per destination margin per iteration, gated). Effort M overall.

**Q3. Patch-major raster scans in landscape ecology. CONFIRMED.** `landscape-metrics.ts:70-100`
gives each patch a thread that scans every cell: O(P·N). Class adjacency is O(C²·N)
(`:102-119`). Patch proximity allocates two P·N float buffers (`patch-proximity.ts:59-66`) and runs
an O(P²·N) pair scan (`:124-137`) and an O(P²·N) area recount inside the summary (`:155-163`). At
1,000 patches on a 1,000 × 1,000 raster, the P·N buffers are 4 GB each. Better: one cell-parallel
pass with integer atomics into per-patch and C×C accumulators (exactly what
`raster-analysis/raster-patches/gpu-raster-patch-metrics.ts` does for area, perimeter and extent),
and a multi-label distance transform (jump flooding, or `GPUDistanceField` with a label channel)
whose boundary pixels emit (patch, nearest other patch, distance) candidates reduced by sort and
segmented minimum. Effort M.

**Q4. Single-thread status counts in builder recipes. CONFIRMED.** Recipes count invalid rows with
one invocation over every row or raster cell: `surface-preparation-recipe.ts:120-130`,
`drainage-network-recipe.ts:95-100`, `local-variability-recipe.ts:128-132`,
`interpolation-diagnostics-recipe.ts:157-161`, `dasymetric-redistribution-recipe.ts:337-400` (three
nodes), `network-scenarios-recipe.ts:383-387`, `cluster-and-outline-recipe.ts:417-421`. On a
10-million-cell raster this is one thread doing 10^7 reads, a likely device-lost timeout. Better: one
shared `addRecipeCountNodes` helper in `recipe-utils.ts` (workgroup tree + atomic add, or published
`GPUReduction`). Effort S.

**Q5. O(n²) pair loops without an index in point process and validation. CONFIRMED.** Inhomogeneous
border K uses one thread per radius doing O(n²) (`gpu-inhomogeneous-point-pattern.ts:220-260`), so
256 threads do all the work. Translation K enumerates n(n−1)/2 pairs and then O(n) per (row,
radius) (`translation-point-pattern.ts:56`, `:114`). Balanced sampling ranks by counting
(`gpu-spatial-balanced-sample.ts:264-283`). Cross-validation buffer masks are O(F·n²) at up to
65,536 rows (`gpu-spatial-cross-validation.ts:69`, `:137-175`): 2.7·10^11 distance tests at the cap.
Better: `GPUGridIndex` radius queries per row (radii are bounded by the largest radius or the buffer
distance), `GPUSort` of quadtree codes plus `GPUScan` of tickets for GRTS, row-parallel border K
with a per-(row, radius) partial and the existing 256-lane reduction. Effort M.

**Q6. Compile-time unrolling with no node budget. CONFIRMED for the counts, SUSPECTED for the
impact.** Sparse resistance: 5 nodes per iteration (`sparse-resistance-solver.ts:132-351`), up to
4,096 iterations: 20,480 nodes; default 1,024 is 5,120. `GPULocationAllocation` unrolls greedy
steps (`:1156`) and up to 1,000 swap iterations of 6 nodes (`:672`) with no node guard, unlike
`GPUNetworkCentrality`, which rejects more than 32,768 nodes. Nested validation instantiates up to
2,304 kriging or RBF fits in TypeScript (`gpu-nested-interpolation-validation.ts:130-285`).
Building tessellation and structures add one or two `GPUPolygonOverlay` instances per building
(`building-tessellation.ts:152-309`, `building-structures.ts:48-144`). Better: an exact
`estimatedCommandNodeCount` and a shared limit, and batching (one overlay with source IDs for all
buildings). Effort S for guards, L for batching overlays.

**Q7. Baked parameters force recompiles. CONFIRMED.** Every urban and landscape recipe bakes all
numeric props as WGSL literals (examples: `isovists.ts:133` `maximumDistance`,
`landscape-metrics.ts:95` `cellSize`, `patch-proximity.ts:160` `searchRadius`,
`street-profiles.ts:114` `sampleSpacing`). So do the point-process classes (bandwidth,
significance level), `GPUSpatialCrossValidation` (`bufferDistance`, `:169`) and
`GPUSpatialInteraction` (coefficients, tolerances). Interactive sliders on any of these rebuild
pipelines. Better: `GPUParameterBuffer` or a resident parameter view, as
`GPULocalOutlierFactor` and `GPUMapColoring` do. Effort S per contributor.

**Q8. Private re-implementations of published primitives. CONFIRMED.** WGSL insertion sorts inside
one thread: `recipe-utils.ts:395-408`, `isovists.ts:80`, `street-profiles.ts:154`,
`segregation-counterfactual.ts:62`, `spatial-validation-design.ts:205`. Counting ranks instead of a
sort: `gpu-spatial-balanced-sample.ts:276-282`. The first is bounded by the disk stride and is
acceptable; the others sort data-sized arrays and should use `GPUSort` (or a segmented sort keyed by
row). Effort S to M.

**Q9. Precision. CONFIRMED where cited.** The sparse solver is float32 throughout and its docs ask
callers to normalise conductances (`sparse-resistance-connectivity.ts:41`); the stopping test uses
both the recursive residual and an independent true residual (`:136-246`), which is good. Dense
`resistance-connectivity.ts` uses plain float sums in one thread. Landscape and segregation sums are
plain float loops; `GPUSegregation` documents its 2^24 limit (`gpu-segregation.ts:174-177`); the
expansion files do not.

## 4. Consistency findings

**C1. Two shapes of composition, and no rule that separates them. CONFIRMED.** Three forms exist:

1. Classes implementing `GPUCommandNodeProducer`. Some own kernels (`GPUSegregation`); some only
   compose other classes (`GPUSpatialCrossValidatedOLS`, `GPUNestedInterpolationValidation`,
   `GPUQuantileLISA`).
2. Builder recipes (27): `RecipeBuilder`, `GPURecipeOverrides` for caller-owned inputs, outputs and
   scratch, and `GPURecipeResult` with `status.stages`.
3. Expansion recipes (12, urban and landscape): named `add*Recipe` but they compose almost nothing.
   10 of 12 contain only raw kernels. Tessellation composes `GPUPolygonOverlay`; sparse resistance
   composes `GPUSort` and `GPUCOOToCSR`. They return `{outputs, status?, provenance}`.

So the name does not tell the reader whether a symbol is a composition. Recommended rule:
- **Class**: owns kernels for one named algorithm, validates at construction, takes caller-owned
  outputs, and reports status with the shared port names.
- **Recipe**: only wires public contributors plus thin adapter kernels, and returns
  `GPURecipeResult`.

Under that rule the twelve expansion recipes become classes (`GPUIsovists`, `GPULandscapeMetrics`,
`GPUSparseResistanceConnectivity`, ...). `GPUQuantileLISA` and the two validation compositions stay
classes, but should move to their families' directories.

**C2. `RecipeBuilder` and stage status are not applied uniformly. CONFIRMED.** 27 of 39 recipes use
`RecipeBuilder`; none of the 12 expansion recipes do. Even in builder recipes,
`RecipeBuilder.contributors` lists only `builder.add` calls (`recipe-utils.ts:66-72`). Raw
`graph.add(createWGSLKernelNode(...))` glue is not listed. Examples:
`surface-preparation-recipe.ts:81`, `:120`, and seven raw adds in the hot-spot recipe. So
`contributors` is not "every node the recipe added", as `GPURecipeResult` documents (`:45-46`).
`addExcessRiskRecipe` returns `status: {stages: []}` (`rate-mapping-recipes.ts:86`), although its
contributor has invalid-input conditions.

**C3. Recipes do not do CPU work or readback (census: 0 readbacks). CONFIRMED by grep.** The
exception is `GPUStudyWindowPreparation`, which takes an optional CPU `rebuildWhen` predicate
(`study-window/gpu-study-window-preparation.ts:47-56`). It is explicit and documented.

**C4. Two recipe utility modules.** `recipe-utils.ts` (`getOrCreateView`, `assertRecipe`, which
throws with a message) and `expansion-recipe-utils.ts` (`getRecipeView`, which uses bare `assert`,
`createRecipeKernelNode`, and `validateRecipeBoundary`). Nine builder recipe files import both. Converge
on one module: keep `validateRecipeBoundary` and per-binding validation, and use messages in the
`assertRecipe` style.

**C5. Output ownership.** Builder recipes accept caller-owned outputs; expansion recipes cannot
(C1). Contract item 3 is broken for 12 public symbols.

**C6. Status vocabulary.** Expansion recipes and point-process, validation and interaction classes
publish positional words: `uint32x4` `[count, required, overflow, invalid]` in
`visibility-graph.ts:100`; `[converged, iterations, reachable, invalid]` in
`sparse-resistance-connectivity.ts:63`; per-radius codes 0 to 5 in translation K; codes 0 to 4 in
`gpu-spatial-interaction.ts:51`. The meanings line up with `count`, `requiredCount`, `overflow`,
`invalidCount`, `converged`, `iterationCount`, but none use the named ports. Recommend named
one-word views, or a shared `GPUStatusWords` layout constant per port, which recipes can then put
in stages.

**C7. Validation style split (AGENTS.md versus the contract).** Bare `assert` without a message:
urban (23), landscape (29), point-process (44), `GPUQuantileLISA` (6),
`dasymetric-redistribution-recipe.ts` (10), and `expansion-recipe-utils.ts:29`, `:65`. Helpers with
messages: the baseline classes, spatial validation and location allocation. Builder recipes use
`assertRecipe` with messages. A caller who passes 65 buildings gets an unlabeled assertion failure
from `building-tessellation.ts:43-47`. Recommend `validateCapacity` with named constants
(`GPU_BUILDING_TESSELLATION_MAXIMUM_BUILDINGS`, ...), which also makes the limits discoverable.

**C8. Space and units.** None of the 12 expansion recipes takes `spatialContext`. Landscape uses
`width`, `height` and `cellSize` rather than the raster grid contract (`analysis-core/raster-grid.ts`).
`GPUSpatialCrossValidation` and `GPUSpatialBalancedSample` take positions with no context. The
inhomogeneous point pattern checks planar context through its study window (`:89-92`); that is the
pattern to follow.

**C9. No-data and missing values.** Expansion code writes NaN with
`bitcast<f32>(0x7fc00000u | (index & 0u))` 99 times library-wide (40 in scope), instead of
`GET_NAN_WGSL` from `analysis-core/kernels/wgsl-common`, which the sparse solver and validation use.
Validity outputs are spelt four ways in this scope: `validity`, `complete`, `status`, per-row
`flags`.

**C10. File placement.** Segregation counterfactual, decomposition and population alignment live in
`landscape-ecology/` but are PySAL segregation methods. `GPUQuantileLISA` lives in `recipes/`.
`GPUBoundedFacilityAllocation` and `GPULocationAllocation` sit in different ontology capabilities
(see C11).

**C11. Ontology completeness and vocabulary. CONFIRMED by evaluating the module.**
- Size: 49 capabilities (48 available, 1 opportunity) and 261 operators. 260 operators carry package
  `gpu-spatial-analysis`; 1 carries `gpu-terrain`.
- Count per entry point (exported classes and recipes, and how many have an ontology entry):

| Entry point | Exported classes | In ontology | Exported `add*Recipe` | In ontology |
| --- | --- | --- | --- | --- |
| gpu-spatial-analysis | 224 | 224 | 35 | 35 |
| gpu-network | 34 | 1 (`GPUParameterBuffer`) | 4 | 0 |
| gpu-terrain | 47 | 2 (`GPUParameterBuffer`, `GPUTerrainPeakProminence`) | 0 | 0 |
| raster-analysis | 50 | 0 | 0 | 0 |
| dataframe-analysis | 24 | 0 | 0 | 0 |
| pointcloud-analysis | 10 | 0 | 0 | 0 |
| computation-record | 4 | 0 | 0 | 0 |
| geospatial-analysis | 2 | 0 | 0 | 0 |
| analysis-core | 0 | 0 | 0 | 0 |

- Missing, by package (168 classes and 4 recipes):
  - gpu-network: 33 classes (every network contributor, from `GPUNetworkFlow` to
    `GPUNetworkInhomogeneousK`) and all four network recipes.
  - gpu-terrain: 45 classes.
  - raster-analysis: all 50.
  - dataframe-analysis: all 24, including `GPURipley`, `GPUVariogram` and `GPUInequality`, which
    analysts will look for under spatial statistics.
  - pointcloud-analysis: all 10.
  - geospatial-analysis: `GPUDatumGridTransform`, `GPUPointBuffer`.
  - computation-record: 4 infrastructure classes. These are arguably not operators.
  - Full lists: scratchpad `ontology-audit.json`.
- The one non-class, non-recipe operator is `addClockEncounters`, a third naming pattern for an
  `add*` function.
- `GPUParameterBuffer` is listed as an operator. It is a parameter helper.
- Vocabulary:
  - 13 of 49 capability IDs are named after the development wave rather than the analysis:
    `expansion-*` (`spatial-analysis-catalog.ts:27-119`) and `coverage-*`
    (`coverage-expansion-catalog.ts:23-261`).
  - One family is split across wave and domain capabilities. Allocation:
    `expansion-facility-allocation` (`:113`) versus `region-optimization` (`:776`). Segregation:
    `accessibility-and-segregation` (`:618`) versus `coverage-landscape-ecology`. Workflows:
    `workflow-recipes` versus `expansion-workflows`.
  - 200 of 261 operators have `semantics: 'unspecified'`; about 200 are unspecified for
    dimensionality, topology, cardinality and determinism.
  - 238 of 261 have evidence `partial`.
  - Only 1 operator is `2d+time` (TC-08 tracks this).
  - The `package` field defaults to `gpu-spatial-analysis` (`spatial-analysis-operators.ts:838`)
    rather than being derived from where the symbol is exported.

## 5. Duplication and shared-helper candidates

| What | Copies | Recommendation |
| --- | --- | --- |
| Serial union-find | `building-adjacency.ts:149`, `reconcile-patches.ts:37-40` | `analysis-core/kernels/masked-weak-components.ts` (already used by `sparse-resistance-system.ts:244`) or published `GPURasterConnectedComponents` (`@luma.gl/graph/gpu-raster`) |
| Patch area and perimeter | `landscape-metrics.ts:70-100` | `raster-analysis/raster-patches/gpu-raster-patch-metrics.ts` (integer atomics, exact) |
| k-means | `spatial-validation-design.ts:212-226` | `gpu-spatial-analysis/spatial-clustering/gpu-kmeans.ts` (has k-means++) |
| Simulation envelopes | `point-process/gpu-point-pattern-envelope.ts` | overlaps `dataframe-analysis/pair-statistics/gpu-ripley-envelope.ts` (pointwise envelopes with its own simulation); share the order-statistic and global-test kernels |
| Dissimilarity index | `segregation-decomposition.ts:56-60` | `GPUSegregation` D per scenario, then a 1-thread Shapley node |
| Exact ID join | `segregation-counterfactual.ts:115-142` | `dataframe-analysis/key-join/gpu-key-join.ts` |
| Laplacian CG | `resistance-connectivity.ts` (dense, serial) and `sparse-resistance-solver.ts` | keep the sparse one only |
| Segment ray cast | `urban-ray-utils.ts` (shared by 3 urban files, good) | back it with the BVH technique in `segment-intersection/segment-bvh.ts`; that file already calls itself "a private copy of the technique used by the spatial join contributors", so promote one copy to `analysis-core` |
| NaN constant idiom | 99 sites | `GET_NAN_WGSL` |
| `readOutput` / `writeOutput` vec4 helpers | `landscape-metrics.ts:78-79`, `:128-129`, `:159-160`, `patch-proximity.ts:145-146`, `resistance-connectivity.ts:55-59`, `segregation-decomposition.ts:70-71` | generated accessors in `createWGSLKernelNode`, or drop (several are unused) |
| Recipe helper modules | `recipe-utils.ts`, `expansion-recipe-utils.ts` | merge (C4) |
| Status count glue | 7 recipe sites (Q4) | `addRecipeCountNodes` in `recipe-utils.ts` |

**Sparse resistance solver as the library's only sparse linear solver (question 3).**
- What it does well, CONFIRMED:
  - CSR assembly on published `GPUSort` and `GPUCOOToCSR`, with a stable sort, so the order is
    deterministic (`sparse-resistance-system.ts:163-216`).
  - Jacobi preconditioner (`:218-242`) and component masking (`:244-256`).
  - Deterministic two-level pairwise dot products, a GPU dispatch gate that disarms on convergence,
    breakdown or budget, and override constants for source and sink, so they are not WGSL literals
    (`sparse-resistance-solver.ts:62-351`).
  - An independent true residual and a reachability certificate
    (`sparse-resistance-connectivity.ts:136-246`).
- Limits:
  1. One right-hand side per graph (`source`, `sink` are construction props), so Circuitscape
     pairwise, one-to-all, all-to-one and cumulative-current modes need one graph per pair.
  2. No continuation: at most 4,096 iterations in one submission. For a grid Laplacian, Jacobi-CG
     needs on the order of the grid side length in iterations (SUSPECTED, standard theory), so a
     2,048 × 2,048 raster is at or beyond the cap.
  3. Jacobi only, where Circuitscape uses algebraic multigrid.
  4. It is wired to the resistance problem: grounded sink, unit injection, conductance edge list.
  5. It lives in `landscape-ecology/` as `@internal`.
- Who else would call a shared symmetric positive-definite CSR solver:
  - Spatial lag and error models: prediction and the `(I − ρW)^{-1}` solves and impacts. The
    current exact log-determinant is capped at 1,024 rows by its dense eigenvalue method
    (`spatial-regression/spatial-exact-log-determinant.ts:14`). Lanczos on the same CSR would
    replace it. SUSPECTED.
  - CAR and SAR smoothing, and Gaussian Markov random field fields.
  - Harmonic (Laplace) no-data fill and raster inpainting. `GPURasterFillNoData` has only nearest,
    IDW and the GDAL four-quadrant method (`raster-analysis/gpu-raster-fill-nodata.ts:43`).
  - Current-flow betweenness and closeness, and resistance distance, in network centrality
    (absent; grep finds no "current flow").
  - A raster resistance adapter (EC03-R).
  - Spectral clustering and embedding.
- Recommendation: move it to `analysis-core/kernels/sparse-spd-solve.ts` with a generic CSR plus
  diagonal input, and support a block of right-hand sides (batched CG with per-column scalars).
  Keep the resistance recipe as a thin adapter. The upstream request (`GPUJacobiPCG` as command
  nodes) stays as recorded in `dev-docs/upstream-workarounds.md:55`. Effort M.

## 6. Comprehensiveness

Ranked by how often practitioners use them. Items already in the roadmap are named by ID only.

| Rank | Missing (reference) | GPU-suitable | Reuses | Effort |
| --- | --- | --- | --- | --- |
| 1 | Core FRAGSTATS / landscapemetrics set at raster scale: PLAND, NP, LPI, edge density, shape index, FRAC/PAFRAC, Shannon and Simpson diversity and evenness, IJI, ENN statistics, division / splitting / effective mesh, moving-window landscape metrics. The current recipe has area, perimeter, cohesion, AI, contagion and mesh only, and does not scale (Q3) | yes: per-cell atomics, segmented reductions, focal windows | published `GPURasterConnectedComponents`, `GPURasterPatchMetrics`, `GPUGroupStatistics`, `GPUDistanceField` | M |
| 2 | Circuitscape modes: pairwise matrix, one-to-all, all-to-one, cumulative current map, focal-node sets (EC03-R tracks only the raster adapter) | yes, with multi-RHS CG | sparse solver (section 5) | M to L |
| 3 | Least-cost corridors and Linkage Mapper style normalised corridors (ArcGIS Corridor) | yes | `GPUCostDistance` twice plus a raster add | S |
| 4 | Space-syntax segment angular analysis (depthmapX angular choice and integration with metric radius, NACH, NAIN), axial lines; VGA beyond 256 points | yes | `GPUNetworkLineGraph`, `GPUNetworkCentrality` with angular costs; raster visibility for grid VGA | M |
| 5 | momepy street-network and block metrics: meshedness, gamma, node density, cul-de-sac length, COINS strokes, enclosures and blocks from street plus barrier lines; tessellation at city scale | yes (tessellation by jump flooding on a raster, or GPU Delaunay) | network statistics, `GPUPolygonOverlay`, line merge | M to L |
| 6 | spatstat simulation of null patterns: CSR (`rpoispp`), binomial, Thomas and Matérn cluster, inhomogeneous Poisson by thinning. `GPUPointPatternEnvelope` requires caller-supplied curves (`gpu-point-pattern-envelope.ts:38`) | yes | Philox stream, `GPUCompaction` | S |
| 7 | spatstat model fitting: `ppm` (Berman-Turner quadrature Poisson GLM), `kppm` minimum contrast; quadrat count test (PP01/PP02 cover intensity and corrections, not model fitting) | yes (IRLS on GPU exists in GWR code) | spatial-regression IRLS, KDE | M |
| 8 | ArcGIS location-allocation problem types: Minimize Facilities, Maximize Attendance (distance-decay demand), Maximize and Target Market Share; p-center beyond 32 candidates; p-dispersion (spopt) | yes (greedy plus swap generalises) | `GPULocationAllocation` swap engine, `GPUHuffTradeAreas` | M |
| 9 | spopt SKATER and spectral regionalization | yes | `GPUSpatialWeightsMinimumSpanningTree`, `GPUKMeans` | S to M |
| 10 | Hansen gravity accessibility and E2SFCA convenience (PySAL access) | yes | catchment weights (SUSPECTED partly covered by weight kernels) | S |
| 11 | Segregation: other indices and nulls | already tracked as EC04 | | |
| 12 | blockCV hierarchical and balanced blocks | already tracked as ST10 | | |
| 13 | spatial interaction with several covariates | already tracked as ST11 | | |
| 14 | GRTS design variance and replacement sites | already tracked as PP05 | | |

**Out of place, or candidates to drop or move.**
- `addResistanceConnectivityRecipe`: superseded by the sparse recipe. Drop.
- `addExcessRiskRecipe`: a wrapper of one contributor. Merge.
- The three segregation recipes in `landscape-ecology/`: move to `segregation/`.
- `GPUQuantileLISA`: move to `spatial-autocorrelation/`.
- `GPUParameterBuffer` as an ontology operator: drop the entry.
- `GPUBoundedFacilityAllocation`: keep, but document it as an exhaustive exact oracle for small
  problems, not a production allocator.

## 7. Top ten actions

| # | Action | Effort | Files |
| --- | --- | --- | --- |
| 1 | Apply one composition rule (C1): turn the 12 expansion recipes into classes with caller-owned outputs, named limits through `validateCapacity`, named status ports and `spatialContext` or raster grid. Merge `expansion-recipe-utils.ts` into `recipe-utils.ts` | M | `urban-morphology/*`, `landscape-ecology/*`, `recipes/recipe-utils.ts`, `recipes/expansion-recipe-utils.ts` |
| 2 | Replace the 7 single-thread status counts with a shared parallel count helper; make `RecipeBuilder` record raw glue nodes, or document that `contributors` lists only composed contributors | S | `recipes/recipe-utils.ts`, the 7 recipe files in Q4, `gpu-network/recipes/network-scenarios-recipe.ts` |
| 3 | Rebuild landscape metrics on connected components plus `GPURasterPatchMetrics` plus atomic class-adjacency histograms; delete the serial reconcile; add the FRAGSTATS core set | M | `landscape-ecology/landscape-metrics.ts`, `reconcile-patches.ts`, `raster-analysis/raster-patches/*` |
| 4 | Rebuild patch proximity on a multi-label distance transform with boundary-pair reduction; remove the P·N buffers | M | `landscape-ecology/patch-proximity.ts`, `raster-analysis/distance-field/*` |
| 5 | Put urban ray casting on a segment BVH (promote `segment-bvh.ts` to `analysis-core`); compute visibility-graph metrics with `GPUCompaction` plus `GPUNetworkCentrality` on the CSR; lift the 256-point cap | M | `urban-morphology/urban-ray-utils.ts`, `isovists.ts`, `visibility-graph.ts`, `street-profiles.ts`, `segment-intersection/segment-bvh.ts` |
| 6 | Generalise the sparse solver into `analysis-core` (CSR SPD, multi-RHS, gated continuation); delete the dense CG; then add the Circuitscape modes and the raster adapter (EC03-R) | M | `landscape-ecology/sparse-resistance-*.ts`, `resistance-connectivity.ts`, new `analysis-core/kernels/sparse-spd-solve.ts` |
| 7 | Index the pair loops: grid-index radius queries for inhomogeneous K (border and translation) and the CV buffer masks; GRTS on `GPUSort` plus `GPUScan`; continuous sampling as candidate-parallel plus compaction; validation designs on `GPUKMeans` | M | `point-process/*`, `spatial-validation/*` |
| 8 | Rework `GPUSpatialInteraction` as row-parallel margin passes with gated iterations; raise the 4,096-pair and 128-zone caps | M | `spatial-interaction/gpu-spatial-interaction.ts` |
| 9 | Ontology: generate operator entries from all nine entry points (168 classes and 4 recipes missing); rename the 13 wave-named capabilities to domain names; derive `package` from the export site; fill the `unspecified` fields for this domain; remove `GPUParameterBuffer` | S to M | `ontology/*`, `coverage-expansion-catalog.ts`, `kepler-analysis-catalog.ts` |
| 10 | Merge duplicates and fix placement: segregation decomposition onto `GPUSegregation`, the ID join onto `GPUKeyJoin`, the shared envelope kernels with `GPURipleyEnvelope`, `GPUQuantileLISA` and the segregation recipes to their families; add node-count guards to `GPULocationAllocation` and tessellation; move numeric props to parameter buffers (Q7) | S each | `landscape-ecology/segregation-*.ts`, `recipes/quantile-lisa-recipe.ts`, `point-process/gpu-point-pattern-envelope.ts`, `location-allocation/gpu-location-allocation.ts` |
