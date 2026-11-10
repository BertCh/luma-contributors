# Library review: spatial statistics, spatial models and movement

Domain reviewer report, 2026-11-09, branch `upstream-first`, working tree as is. Read-only.
Paths are relative to `modules/analysis/src` unless they start with `dev-docs/` or `node_modules/`.
`gsa/` = `gpu-spatial-analysis/`.

Tags: **CONFIRMED** = I read the code path (or the cited report did, and I cite it). **SUSPECTED** =
inferred from names, comments or grep without reading the kernel.

## 1. Scope and method

Scope: 20 directories under `gsa/` (`spatial-autocorrelation`, `permutation-inference`,
`multiple-testing`, `spatial-weights`, `spatial-regression`, `spatial-clustering`,
`spatial-regionalization`, `spatial-interpolation`, `scan-statistics`, `global-spatial-statistics`,
`distribution-dynamics`, `emerging-hot-spots`, `space-time-tests`, `track-similarity` and the six
`trajectory-*` directories). That is **191 files, 79,876 lines**, of which 87 are contributor files
in the census (53,087 lines). 91 exported symbols (90 classes and `addClockEncounters`); three of
them (`PhiloxStream`, `WardActiveGraph`, `GPULocalPermutationReplay`) are not re-exported by any
`index.ts`.

Method:
- Correctness was reviewed earlier today. I treat the CONFIRMED findings of four reports as given
  and cite them as **[SS]** `code-spatial-statistics.md`, **[SM]** `code-spatial-models.md`,
  **[ST]** `code-spacetime-statistics.md`, **[MV]** `code-movement.md` (all in
  `dev-docs/research/spatiotemporal-review-2026-11-09/`). I did not repeat their defect hunt.
- I filled the columns the reports are silent on (parallelism, baked parameters, packed versus
  chunked input, status shape, validation style, hard limits) from `census.json` **[C]** and from
  targeted greps and reads **[O]**.
- Read in full: the four reports; `gsa/spatial-regression/spatial-regression-solve.ts`;
  `gsa/spatial-interpolation/equilibrated-solve-wgsl.ts`; `gsa/spatial-clustering/gpu-optics-xi.ts`;
  `raster-analysis/remote-sensing/raster-kernel-utils.ts:55-100`;
  `node_modules/@luma.gl/graph/dist/gpu-vector-search/gpu-k-means.d.ts`.
- Read in part (the cited line ranges): `gpu-probit-regression.ts:75-240`,
  `gpu-multiscale-geographically-weighted-regression.ts:270-505`, `gpu-local-variability.ts:125-140`,
  `gpu-spatial-scan-statistic.ts:990-1006`, `gpu-optics.ts:60-130`, `gpu-kriging.ts:625-660`,
  `gpu-spatial-panel-lag-maximum-likelihood.ts:200-310`, `gpu-spatial-error-gm.ts:940-1055`,
  `gpu-spatial-regression-diagnostics.ts:700-730`, `gpu-spatial-two-stage-least-squares.ts:845-875`,
  `gpu-generalized-geographically-weighted-regression.ts:300-315`, `gpu-natural-neighbor.ts:20-62`,
  `gpu-class-assignment.ts:40-63`, `gpu-classification-fit.ts:95-114`, `gpu-kmeans.ts:232-238`,
  `separation-wgsl.ts:1-40`, `grid-neighbor-scan.ts:1-40`, `gwr-grid-index.ts:1-30`.
- Grep-only (every file in scope): `GraphVectorView`, `GPUChunkedSpatialWeights`,
  `GPUParameterBuffer`, `*StatusPort`, `contracts/status`, bare `assert(`, `getWGSLFloatLiteral(`,
  single-invocation dispatches (104 sites; I printed the loops around 32 of them), `validateCapacity`
  limits, point-index imports, and dense-solver keywords across all of `modules/analysis/src`.
- Not opened beyond grep: the remaining ~160 files, including all trajectory kernels (covered by
  [MV]) and the spatial-weights, autocorrelation and permutation kernels (covered by [SS]).

## 2. Contributor table

Abbreviations. Par: `row`, `coop` (cooperative), `serial`, `cpu`. Iter: `none`, `fixed`, `unroll`
(unrolled with GPU gate), `cpu`. Params: `buf` = per-frame `parameters` view or `GPUParameterBuffer`;
`baked(x)` = prop `x` pasted into WGSL as a literal. Inputs: `packed` or `chunked W` (only the CSR
weights have a chunked twin; **no file in scope accepts `GraphVectorView`**, CONFIRMED by grep).
Status: `shared` (types from `contracts/status`), `rowbits`, `adhoc`, `none`. Val: `helpers`,
`assert` (bare), `mixed`. Src: where the non-trivial cells come from.

### 2.1 Spatial weights (14 symbols)

| Symbol | File (lines) | Computes (ref) | Par | Work | Iter | Hard limits | Params | Inputs | Ctx | Status | Val | Verdict | Src |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `GPUContiguityWeights` | spatial-weights/gpu-contiguity-weights.ts (757) | queen/rook (libpysal) | row | sort-based; O(Σm²) per shared vertex | fixed | pair capacity (design) | baked(snapTolerance) `:340` | packed | none; planar (ST09 tracked) | adhoc `candidateOverflow` | helpers | align | SS, O |
| `GPULatticeWeights` | gpu-lattice-weights.ts (336) | lat2W | row | O(cells·r²) direct | none | radius ≤ 32 `:42` (design) | baked(cellWidth, cellHeight) `:210` | — | none; should take raster grid | none | helpers | align | SS, O |
| `GPUFuzzyContiguity` | gpu-fuzzy-contiguity.ts (237) | fuzzy_contiguity | row | O(n²) dense distance table | none | ≤ 512 features `:62`, ≤ 65,536 vertices (design) | baked(rows) inline loops `:217,232` | packed | none | adhoc | helpers | **rework**: BVH segment-distance join with tolerance → CSR | SS, O |
| `GPUIslandAttachment` | gpu-island-attachment.ts (221) | attach_islands | row | O(islands·n) BF | none | none | — | packed | none | adhoc | helpers | **rework**: nearest-feature join / `GPUNeighborSearch` k=1 | SS |
| `GPUSpatialWeightsTransform`, `GPUChunkedSpatialWeightsTransform` | gpu-spatial-weights-transform.ts (550), gpu-chunked-…-transform.ts (366) | R/B/D/V/kernel | row | O(slots) | none | binding/4 slots (inherent) | baked(bandwidth) `:489` | chunked W | yes (chunked) | adhoc | helpers | align | SS, O |
| `GPUSpatialWeightsAlgebra`, `GPUChunkedSpatialWeightsAlgebra` | gpu-spatial-weights-algebra.ts (767), gpu-chunked-…-algebra.ts (318) | set ops, higher order | row | merge per row; order ∝ frontier² | fixed | order ≤ 32 `:50` | — | chunked W | yes | adhoc | helpers | conforming | SS |
| `GPUSpatialWeightsSummary`, `GPUSpatialWeightsValidation`, `GPUChunkedSpatialWeightsValidation`, `GPUSpatialWeightsTranspose`, `GPUChunkedSpatialWeightsTranspose`, `GPUSpatialLag` | gpu-spatial-weights-summary.ts (773) and five siblings (44–247) | S0/S1/S2, invariants, Wᵀ, lag | row | O(slots) | none | binding/4 slots (inherent) | baked(symmetryTolerance) summary `:165` | chunked W | n/a (topology) | adhoc | helpers | conforming (6) | SS, C |

### 2.2 Autocorrelation, permutation, multiple testing (16 symbols)

| Symbol | File (lines) | Computes (ref) | Par | Work | Iter | Hard limits | Params | Inputs | Ctx | Status | Val | Verdict | Src |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `GPUGlobalSpatialStatistics` | global-spatial-statistics/gpu-global-spatial-statistics.ts (1586) | Moran, Geary, G, BV Moran, join counts (esda) | row + fixed-order reduce | O(slots) | fixed | rows < 2^24 `:119` (f32 counts, design) | buf | chunked W | n/a | adhoc `overflow` | helpers | conforming | SS |
| `GPULocalMoran` | spatial-autocorrelation/gpu-local-moran.ts (696) | LISA analytic z | row | O(slots) | none | rows < 2^24 | buf | chunked W | n/a | adhoc | helpers | conforming | SS |
| `GPUHotSpotAnalysis` | gpu-hot-spot-analysis.ts (555) | Gi/Gi* + FDR | row | O(slots) | none | rows < 2^24 | baked(selfWeight) `:443` | chunked W | n/a | adhoc | helpers | align | SS, O |
| `GPULocalGeary`, `GPUSpatialPearson`, `GPUGammaStatistic` | gpu-additional-spatial-statistics.ts (654) | local Geary, edge Pearson, Γ | row | O(slots) | none | ≤ 256 groups per chunk `:54` | buf | chunked W | n/a | adhoc | helpers | conforming (3) | SS |
| `GPULocalSpatialHeteroscedasticity` | gpu-local-variability.ts (440) | LOSH | row + **serial mean** | O(slots) + O(n) on one thread `:136` | none | — | buf | chunked W | n/a | adhoc | helpers | align (mean → `GPUReduction`) | O |
| `GPUMultivariateLocalGeary` | gpu-local-variability.ts | MV local Geary | row | O(slots·c) | none | — | buf | chunked W | n/a | adhoc | helpers | conforming | SS |
| `GPULOSHResidualPermutationTest`, `GPUMultivariateLocalGearyPermutationTest` | gpu-losh-residual-permutation-test.ts (110), gpu-multivariate-local-geary-permutation-test.ts (412) | randomisation | row×perm | O(P·Σk) | none | exact ≤ 256 rows; tiled ≤ 2^24−1 rows, ≤ 9,999 perms, ≤ 128 neighbours (design) | baked(significance) `:248,405` | chunked W | n/a | adhoc | helpers | **merge** into `GPULocalPermutationTest` adapters (own Fisher–Yates) | SS, O |
| `GPUAssuncaoReisStandardization` | gpu-rate-standardization.ts (495) | EB rate (esda) | row | O(n) | none | rows < 2^24 `:39` | buf | packed | n/a | shared-ish `invalidCount` | helpers | conforming | SS, O |
| `GPULocalPermutationTest` | permutation-inference/gpu-local-permutation-test.ts (1455) | conditional permutation, 9 statistics, BH, Besag–Clifford | row×perm | O(P·Σk²) (linear swap lookup) | none | ≤ 64 neighbours (512 opt-in) `:79,86`; perms ≤ 2^20 | buf (Philox seed, count, level) | chunked W | n/a | adhoc | mixed (1 bare) | align (swap lookup; FX-05 tracked) | SS, C |
| `GPUGlobalPermutationTest` | gpu-global-permutation-test.ts (1438) | total permutation, swap-or-not | row×perm | O(P·slots) | none | (P+1)·⌈n/2048⌉ ≤ 16,777,215 `:127-138` (inherent dispatch) | buf | chunked W | n/a | adhoc | helpers | conforming | SS |
| `GPUMultipleTestingAdjustment` | multiple-testing/gpu-multiple-testing-adjustment.ts (570) | Bonferroni/Holm/BH/BY | row | `GPUSort` + scans | none | rows < 2^24 `:43` | buf | packed | n/a | none | helpers | conforming | SS |
| `GPULocalPermutationReplay` (internal) | gpu-local-permutation-replay.ts (151) | replay one row's draws | serial | O(P·k) | none | one row | buf | packed | n/a | none | helpers | conforming | C |
| `PhiloxStream` (internal, CPU) | permutation-random.ts (146) | CPU twin of Philox draws | cpu | — | — | — | — | — | — | — | none | **move** to `analysis-core/kernels` beside `philox-random.ts` | O |

### 2.3 Spatial regression (21 symbols)

| Symbol | File (lines) | Computes (ref) | Par | Work | Iter | Hard limits | Params | Inputs | Ctx | Status | Val | Verdict | Src |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `GPUOrdinaryLeastSquares` | spatial-regression/gpu-ordinary-least-squares.ts (1064) | OLS + spreg diagnostics | row tiles, serial p×p solve | O(n·p²) | none | P ≤ 15 (design, registers) | buf | packed | n/a | shared `GPUSolverStatusPort.code` | helpers | conforming | SM, O |
| `GPUSpatialLagMaximumLikelihood`, `GPUSpatialErrorMaximumLikelihood` | gpu-spatial-lag-…ts (1177), gpu-spatial-error-…ts (1118) | spreg ML_Lag / ML_Error | row (SpMV) + grid search | O(Q·slots·probes) | fixed (1025 grid + 32 bisection); Q−1 Horner nodes | P ≤ 14/15; exact traces ≤ 4,096 (16,384); eigen log-det ≤ 1,024 **on CPU** | buf | chunked W | n/a | shared solver port | helpers | align | SM |
| `GPUSpatialErrorGM`, `GPUSpatialTwoStageLeastSquares` | gpu-spatial-error-gm.ts (1388), gpu-spatial-two-stage-least-squares.ts (1124) | GM_Error, GM_Lag | row moments, serial solve | O(n·p²) | fixed | P ≤ 15 / 8 | buf | chunked W | n/a | shared solver port | helpers | conforming | SM, O |
| `GPUInstrumentalVariablesCovariance` | gpu-instrumental-variables-covariance.ts (313) | IV/HAC covariance | **one thread per matrix entry, loops all rows** `:158,179,280` | O(n) serial per entry | none | rows ≤ 65,536 `:63` (design) | buf | packed | n/a | adhoc | helpers | **rework**: row tiles as in OLS | O |
| `GPUSpatialWeightsTraces`, `GPUSpatialLagImpacts` | gpu-spatial-weights-traces.ts (886), gpu-spatial-lag-impacts.ts (239) | tr(Wq), impacts | row SpMV | O(Q·slots·probes) | fixed | q ≤ 256; exact ≤ 16,384 | buf | chunked W | n/a | shared `approximation` | helpers | conforming (2) | SM, C |
| `GPUSpatialRegressionDiagnostics` | gpu-spatial-regression-diagnostics.ts (905) | Moran on residuals, LM, robust LM, SARMA | row | O(slots) | none | — | buf | packed | n/a | shared solver port | helpers | conforming | SM, O |
| `GPUSpatialPanelRegression`, `GPUSpatialPanelErrorMaximumLikelihood` | gpu-spatial-panel-regression.ts (267), gpu-spatial-panel-error-…ts (208) | panel lag 2SLS, Panel_FE_Error | row + serial within-finish `:221` | I_T⊗W materialised (T-fold work) | fixed | T ≤ 1,024; N·T ≤ 65,536 (error) `:57` | buf | packed | n/a | adhoc | helpers | align | SM, O |
| `GPUSpatialPanelLagMaximumLikelihood` | gpu-spatial-panel-lag-…ts (309) | Panel_RE_Lag | **serial** dense determinant `:212-214` and Gauss–Jordan inverse `:306-307` | O(N³) per evaluation | fixed | N ≤ 32, T ≤ 32, P ≤ 7 `:66` (design) | baked(tolerance) `:232` | packed | n/a | adhoc | helpers | **rework**: T·ln\|I−ρW\| from `GPUSpatialWeightsTraces` | SM, O |
| `GPUSeeminglyUnrelatedRegression` | gpu-seemingly-unrelated-regression.ts (144) | SUR | row moments + serial step (`spatial-sur-nodes.ts:97`, SUSPECTED) | O(n·p²) | fixed | 2–4 eq, P ≤ 7, ≤ 16 coefficients, n ≤ 65,536 `:103-106` | buf | packed | n/a | adhoc | helpers | align | SM, O |
| `GPUProbitRegression` | gpu-probit-regression.ts (339) | probit IRLS | **serial**: one invocation loops all rows per iteration and per line-search step `:177`, doc `:81-82` | O(iter·24·n·p²) on one thread | serial loop in WGSL | n ≤ 65,536 `:114`, P ≤ 11 | baked(tolerance, maximumIterations) `:227` | packed | n/a | adhoc code | helpers | **rework**: row-tile score/information + unrolled IRLS | O |
| `GPUSpatialRegimeRegression` | gpu-spatial-regime-regression.ts (172) | OLS_Regimes | **one thread per regime loops all rows** `:113,127` | O(R·n) | none | ≤ 4,096 regimes, n ≤ 65,536 `:58` | — | packed | n/a | adhoc | helpers | **rework**: sort by regime + segmented sums | O |
| `GPURestrictedRegimeRegression` | gpu-restricted-regime-regression.ts (256) | restricted regimes, Chow | **serial** fit `:161` | O(n·p²) one thread | none | P ≤ 7, ≤ 8 regimes, n ≤ 65,536 `:67-76` | — | packed | n/a | adhoc | helpers | **rework** (same as above) | O |
| `GPUGeographicallyWeightedRegression`, `GPUGeneralizedGeographicallyWeightedRegression`, `GPUGeographicallyWeightedRegressionNonstationarityTest` | gpu-geographically-weighted-regression.ts (1659), gpu-generalized-…ts (723), …-nonstationarity-test.ts (1007) | GWR, Poisson/logit GWR, MC test (mgwr) | row per focus | O(n²) BF or private grid (`gwr-grid-index.ts`) | unroll (golden section, IRLS) | n ≤ 65,536 BF / 1,048,576 indexed; adaptive k ≤ 128; P ≤ 11 / 24 / 7 | buf; GGWR baked(bandwidth) in selection `generalized-gwr-selection.ts:55` | packed | spatialContext | rowbits | helpers | align (FX-09 tracked; own grid index) | SM, O |
| `GPUMultiscaleGeographicallyWeightedRegression` | gpu-multiscale-…ts (828) | MGWR backfit | row per (focus, candidate) + serial global fit `:285` | **O(iter·P·candidates·n²)**, no index `:486-492` | unroll | full inference n ≤ 128; ≤ 32 candidates | buf | packed | spatialContext | adhoc | helpers | **rework**: bounded kernels on the GWR grid index | SM, O |
| `GPULocalCollinearity` | gpu-local-collinearity.ts (334) | VIF, VDP, condition (mgwr) | row | O(n·k·p²) | fixed (Jacobi) | P ≤ 12 `:74` | — | chunked W | n/a | adhoc | helpers | conforming | SM, O |

### 2.4 Clustering and regionalization (11 symbols)

| Symbol | File (lines) | Computes (ref) | Par | Work | Iter | Hard limits | Params | Inputs | Ctx | Status | Val | Verdict | Src |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `GPUSpatialClustering` | spatial-clustering/gpu-spatial-clustering.ts (713) | DBSCAN (sklearn) | row | own `GPUGridIndex` 3×3 walk; → O(n²) if grid coarse | GPU loop (CAS union-find) | minPts ≤ 2^24 | buf | packed | spatialContext | `GPUCompactOutput`; `gridOverflow` unread | helpers | **merge** index into shared neighbour search (PLAT-03) | SS |
| `GPUHierarchicalDensityClustering` | gpu-hierarchical-density-clustering.ts (567) | HDBSCAN EOM | row | kNN via `GPUNearestFeatureJoin` + BVH; Borůvka | unroll | n ≤ 2^20 `:68`, minSamples ≤ 32 `:63` | buf | packed | spatialContext (planar only) | adhoc; `neighborOverflow` unread | helpers | **merge** (same) | SS |
| `GPUOPTICS` (+ Xi node) | gpu-optics.ts (383), gpu-optics-xi.ts | OPTICS, DBSCAN/Xi extraction (sklearn 1.7.2) | **serial** frontier; Xi one invocation `gpu-optics-xi.ts:29` | O(n²), ~n/64 sequential dispatches | unroll batches | n ≤ 4,096 `:103` (design) | baked (sphere radius only) | packed | spatialContext | adhoc u32×3 | **assert** (7) `:103-120` | **rework**: OPTICS order from HDBSCAN mutual-reachability MST (PP04 tracked) | SS, O |
| `GPUKMeans` | gpu-kmeans.ts (1397) | Lloyd + k-means++ + restarts | row; deterministic chunked means | O(n·k) BF (fine) | unroll ≤ 64 | k ≤ 256, restarts ≤ 32 `:54-60` | buf | packed, **2D only** | spatialContext | shared (`contracts/status`) | helpers | **move**: same name as published `@luma.gl/graph` `GPUKMeans`; upstream the 2D improvements | SS, O |
| `GPUAZPRegions` | spatial-regionalization/gpu-azp-regions.ts (1165) | batched AZP (spopt) | row | O(iter·slots) | unroll ≤ 256 | columns ≤ 8 `:41`; articulation ≤ 128 rows/degree 64 | buf (seed) | packed W | n/a | adhoc positional words | helpers | align | SS |
| `GPUMaxPRegions` | gpu-max-p-regions.ts (1291) | parallel max-p | row × starts | O(rounds·starts·slots) | unroll ≤ 4,096 | columns ≤ 8; starts ≤ 64; scratch starts·n·(c+3)·4 B in one binding | buf | packed W | n/a | adhoc (6 words) | helpers | align | SS |
| `GPUSpatialWeightsMinimumSpanningTree`, `GPUSkaterRegions` | gpu-spatial-weights-minimum-spanning-tree.ts (620), gpu-skater-regions.ts (1023) | Borůvka MST, SKATER | row | MST O(slots log n); SKATER tour stats O(Σ subtree) (quadratic on paths) | unroll regions−1 | columns ≤ 32 / 16 | buf | packed W | n/a | adhoc | helpers | align (2) (FX-12 tracked) | SS |
| `GPUWardRegions` (+ internal `WardActiveGraph`) | gpu-ward-regions.ts (1217), ward-active-graph.ts (358) | constrained Ward | row; **sequential mode = ~15 nodes per merge** | O(n−k) graph nodes (sequential); mutual-nearest scales | unroll | columns ≤ 8 | buf | packed W | n/a | adhoc | helpers | align (drop or cap sequential mode); internal helper conforming | SS |
| `GPURegionPartitionEvaluation` | gpu-region-partition-evaluation.ts (683) | within/between SS, pseudo-F | row | `GPUSort` + segments | none | ≤ 32 columns `:38` | — | chunked W | n/a | none | helpers | conforming | SS, O |

### 2.5 Interpolation and simulation (7 symbols)

| Symbol | File (lines) | Computes (ref) | Par | Work | Iter | Hard limits | Params | Inputs | Ctx | Status | Val | Verdict | Src |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `GPUInverseDistanceWeighting` | spatial-interpolation/gpu-inverse-distance-weighting.ts (605) | IDW (gstat/GDAL) | row per cell | grid ring walk (`grid-neighbor-scan.ts`) | none | k ≤ 64 `:45` | buf; index bounds baked `:390` (topology) | packed | spatialContext | none | helpers | conforming | O |
| `GPUKriging` | gpu-kriging.ts (821) | raster ordinary kriging | row per cell | grid ring walk; own Gaussian elimination `:628` | none | k ≤ 16 | buf (variogram) | packed | spatialContext | shared (Invalid, Uncertainty ports) | helpers | **merge** with `GPULocalKriging` | SM, O |
| `GPULocalKriging` | gpu-local-kriging.ts (480) | SK/OK/UK/KED/block/co-kriging | row per query | **O(q·n) BF** unless caller CSR `neighborhoods`; Ruiz LU | none | k ≤ 32, ≤ 8 drift, ≤ 64 support, 2–4 variables | **baked(model sill, range, nugget, searchRadius)** `:349,392,476-477` | packed | **none** (planar, silent) | rowbits | helpers | **merge** (one kriging: index + buffer params + modes) | SM, O |
| `GPULocalRBF` | gpu-local-rbf.ts (300) | local RBFInterpolator (SciPy) | row per query | grid candidates | none | k ≤ 32, degree ≤ 2 | baked(epsilon, searchRadius, smoothing) `:240,262,277` | packed | **none** | none | helpers | align | SM, O |
| `GPUNaturalNeighbor` | gpu-natural-neighbor.ts (309) | Sibson (CGAL) | row per query | bounded cavity walk on caller Delaunay | none | ≤ 64 neighbours, 256 triangles, 2^24 sites | — | packed | **none** (planar) | rowbits | helpers | align | SM, O |
| `GPUConditionalGaussianSimulation` | gpu-conditional-gaussian-simulation.ts (582) | exact conditional Gaussian simulation | coop / blocked; 5 single-invocation steps (`conditional-simulation-blocked.ts:120-323`) | dense (n+q)³ f32 Cholesky | fixed | n+q ≤ 4,096, 64 MiB factor (design) | baked(mean, tolerance) `:300,438` | packed | **none** | adhoc | **assert** (9) | **rework**: Vecchia/sequential simulation on kNN, or FFT/turning-bands field + kriged residual | SM, O |
| `GPUFocalStatistics` | gpu-focal-statistics.ts (338) | raster focal mean/SD/min/max | row per cell | O(cells·r²) | none | radius ≤ 64 `:29` | buf; baked(noDataValue) `:294` | — | raster grid | none | helpers | **move** to `raster-analysis` (no interpolation role) | O |

### 2.6 Space-time statistics and distribution dynamics (11 symbols)

| Symbol | File (lines) | Computes (ref) | Par | Work | Iter | Hard limits | Params | Inputs | Ctx | Status | Val | Verdict | Src |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `GPUEmergingHotSpots` | emerging-hot-spots/gpu-emerging-hot-spots.ts (1363) | ESRI EHS | row per bin | O(bins·(2r+1)²) lattice or CSR | none | slices ≤ 256, r ≤ 32, bins < 2^24 with FDR `:333-339` | buf; baked(selfWeight) `:508` | chunked W | n/a | none | helpers | align (FX-03 tracked) | ST, O |
| `GPUSpatialScanStatistic` | scan-statistics/gpu-spatial-scan-statistic.ts (1887) | Kulldorff scan (SaTScan) | row per (centre, replicate); serial margin prefix sums `:990-1004` | ~(P+1)·zones·window·B² | unroll ≤ 1,024 batches | timeBuckets ≤ 32 (private arrays); full windows ≤ 4,096 zones | buf | packed | spatialContext | adhoc | helpers | align (STX-07 tracked) | ST, O |
| `GPUKnoxTest` | space-time-tests/gpu-knox-test.ts (538) | Knox (pointpats) | row per pair×perm | O(P·pairs), `GPUSort` | none | perms ≤ 65,536 | buf | packed | caller pairs | adhoc | helpers | conforming | ST |
| `GPUMantelTest` | gpu-mantel-test.ts (558) | Mantel (skbio) | row per pair×perm | O(P·pairs) | none | full Mantel ≈ n ≤ 8,190 (binding) | buf | packed | caller pairs | adhoc | helpers | align (FX-04 tracked) | ST |
| `GPUTransitionMatrix`, `GPUTransitionDiagnostics`, `GPUSpatialMarkov`, `GPULISAMarkov` | distribution-dynamics/*.ts (470, 492, 313, 327) | giddy Markov family | row; tiled u32 atomics | O(n·T) | fixed (GTH) | classes ≤ 64 | buf | packed | n/a | diagnostics shared `GPUIterationStatusPort`; rest adhoc | helpers | conforming (4) | ST, C |
| `GPURankMarkov` | gpu-rank-markov.ts (185) | full/geographic rank Markov | row | dense n×n states | none | **rows ≤ 64** `:54` | — | packed | n/a | none | helpers | **rework** (ST08 tracked) | ST |
| `GPUClassificationFit`, `GPUClassAssignment` | gpu-classification-fit.ts (411), gpu-class-assignment.ts (147) | mapclassify fit stats; class by breaks | row; `GPUSort` | O(n log n) | none | classes ≤ 256 `:25` | — | packed | n/a | none | helpers | **move** (2) to `dataframe-analysis/column-classification` | O |

### 2.7 Movement (11 symbols)

| Symbol | File (lines) | Computes (ref) | Par | Work | Iter | Hard limits | Params | Inputs | Ctx | Status | Val | Verdict | Src |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `GPUTrajectoryMetrics` | trajectory-analysis/gpu-trajectory-metrics.ts (671) | length, speed, stops | row + segmented reduce | centroids O(rows) workgroups | none | stop capacity (compile time) | `GPUParameterBuffer` | packed | spatialContext | `GPUCompactOutput` | helpers | align | MV, C |
| `GPUTrajectoryResample` | trajectory-interpolation/gpu-trajectory-resample.ts (284) | resample by time/arc/clock | row; **serial arc length per track** | O(rows) per track serial | none | tracks·samples < 2^32 | `GPUParameterBuffer` | packed | spatialContext | none | helpers | align (FX-06 tracked) | MV |
| `GPUTrajectoryPlayhead` | gpu-trajectory-playhead.ts (400) | position at time | row per track (binary search) | O(T log n) | none | — | `GPUParameterBuffer` | packed | spatialContext | `GPUCompactOutput` + gap | helpers | conforming | MV |
| `GPUTrajectoryEncounters`, `addClockEncounters` | trajectory-encounters/gpu-trajectory-encounters.ts (637), clock-encounters.ts | same-bucket proximity | row | own space-time grid | none | lattice·buckets ≤ 2^25−1 | baked(cellSize) `encounter-kernels.ts:49` | packed | spatialContext | `GPUCompactOutput` | helpers | align (2) (FX-06 tracked) | MV, O |
| `GPUTrajectoryClosestApproach` | gpu-trajectory-closest-approach.ts (491) | CPA | row | hashed space-time grid | none | ≤ 4,096 cells per segment | baked(cellSize, timeCellSize) `closest-approach-kernels.ts:42-43` | packed | spatialContext (planar) | `GPUCompactOutput` | helpers | align | MV, O |
| `GPUZoneEvents` | trajectory-zones/gpu-zone-events.ts (1036) | enter/exit, dwell | row; `GPUBVH` | dense track×zone scratch | none | tracks·zones in one binding | buf | packed | spatialContext (planar) | `GPUCompactOutput` | helpers | align | MV |
| `GPUTrajectoryKalman` | trajectory-estimation/gpu-trajectory-kalman.ts (276) | CV Kalman + RTS | serial per track (inherent) | O(rows) per track | none | — | buf | packed, own port (`offsets`, f32 times) | **none** | rowbits | **assert** (2–4) | align (FX-07 tracked) | MV, C |
| `GPUTrajectoryRestriction` | trajectory-restriction/gpu-trajectory-restriction.ts (278) | atTime/atGeometry, tdwithin | serial per pair | **O(nA·nB) per thread**, pairs baked | none | ≤ 4,096 pairs **in WGSL**; ≤ 256 intervals | baked(pairs) | packed, own port | **none** | rowbits | **assert** (12) | **rework** (MOB-04 tracked) | MV, C |
| `GPUTrackSimilarity` | track-similarity/gpu-track-similarity.ts (924) | Hausdorff, Fréchet, DTW, LCSS | **coop** anti-diagonal DP (`track-similarity-dynamic-programs.ts`) | O(n·m) per pair, parallel lanes | none | shorter track ≤ 2,048 | buf | packed | spatialContext | adhoc | helpers | conforming | MV, O |
| `GPUTrackSimilaritySearch` | gpu-track-similarity-search.ts (349) | threshold + top-k | serial per pair / per track | top-k O(T·P·K²) | none | topK ≤ 1,024 | — | packed | **none** (lon/lat silently planar) | adhoc | **assert** (11) | **rework** (MOB-04 tracked) | MV, C |

**Verdict counts (91 symbols):** conforming 35, align 32, rework 13, merge 6, move 5.

Counter-examples to the two-generation hypothesis:
- Baseline-style contributors that do not scale: `GPUMultiscaleGeographicallyWeightedRegression`
  (O(n²) per candidate), `GPUWardRegions` sequential mode, `GPUSkaterRegions` tour statistics,
  the LOSH serial mean.
- Expansion-style contributors that do scale: `GPUTrackSimilarity` (cooperative anti-diagonal DP),
  `GPUClassificationFit` (two radix sorts, fixed-order reductions, no atomics), `GPULocalCollinearity`.
- The regression family is the most internally consistent part of the domain: 15 files share one
  Cholesky and one Jacobi module (section 5). Its weak points are the six classes that loop over
  all rows on one thread, all capped at 65,536 rows.

## 3. GPU quality findings

**G1. Six regression classes do their main work on one GPU thread. CONFIRMED.**
- `GPUProbitRegression`: one invocation runs every IRLS iteration and every line-search step over
  all rows (`gpu-probit-regression.ts:177`; the doc admits it at `:81-82`). Worst case
  65 × 24 × 65,536 row visits in one thread.
- `GPURestrictedRegimeRegression` fit `:161`; `GPUSpatialRegimeRegression` one thread per regime
  `:113,127`; `GPUSpatialPanelRegression` within-finish `:221`; MGWR global fit `:285`;
  `GPUInstrumentalVariablesCovariance` one thread per matrix entry `:158,179,280`.
- Why it matters: the 65,536-row caps in these classes are a consequence of the serial design, not
  of WebGPU. A large dispatch on one thread also risks the device watchdog.
- Better design: the OLS pattern already in the family (row tiles with Kahan merges into p×p
  moments, then a one-thread p×p solve). For probit, accumulate score and information per tile,
  unroll IRLS iterations with a GPU convergence gate as GWR does, and evaluate the line search
  likelihood as a tiled reduction. Regimes: sort by regime, segmented sums.
- Effort: M for the family (one shared "moments by tile" helper), then S per class.

**G2. MGWR is quadratic without an index. CONFIRMED.** Candidate scoring loops every row for every
(focus, candidate) (`gpu-multiscale-geographically-weighted-regression.ts:486-492`), per backfit
iteration and per variable. The GWR classes already have a grid index (`gwr-grid-index.ts`) and
bounded kernels. Better: reuse it for bounded candidates; keep brute force only for the
`n ≤ 128` full-inference mode. Effort M.

**G3. Two kriging solvers and two candidate strategies. CONFIRMED.**
- `GPUKriging`: grid ring walk, per-frame variogram, private Gaussian elimination
  (`gpu-kriging.ts:628-660`), pivot ratio 1e-4, k ≤ 16, ordinary kriging only.
- `GPULocalKriging`: brute force O(q·n) unless the caller supplies a CSR, the model baked into WGSL
  (`gpu-local-kriging.ts:349,476-477`), Ruiz-equilibrated LU (`equilibrated-solve-wgsl.ts:29`) at
  pivot ratio 1e-5, no `spatialContext`.
- Better: one kriging contributor with `GPUKriging`'s index and parameter path, `GPULocalKriging`'s
  modes, and one equilibrated solver at the measured 1e-4 tolerance. Effort M. The tolerance part
  is FX-10 and the parameter part STM-01; the merge itself is not tracked.

**G4. Private point indices. CONFIRMED by imports.** In scope there are five candidate-generation
mechanisms for points, plus brute force:
1. DBSCAN's own `GPUGridIndex` 3×3 walk (`spatial-clustering-kernels.ts:100-190`, [SS]);
2. HDBSCAN through `GPUNearestFeatureJoin` and the spatial-join BVH ([SS]);
3. `spatial-interpolation/grid-neighbor-scan.ts` ring walk (IDW, `GPUKriging`);
4. `spatial-regression/gwr-grid-index.ts` (GWR, generalized GWR, nonstationarity test, and also
   `point-density/gpu-kernel-density.ts` outside this domain);
5. the trajectory encounter grid and the hashed space-time grid of closest approach ([MV]).

Brute force: OPTICS, local kriging (default), conditional simulation, island attachment, fuzzy
contiguity, MGWR. `GPUNeighborSearch` (out of scope) is the documented producer and none of the
above uses it. Published `@luma.gl/graph` also exposes `GPUPointSpatialQuery` with a
`GPUGridIndexView` radius query. Better: one CSR neighbour producer with k > 32, plus one
analysis-core WGSL ring-walk helper for kernels that must walk inline. Tracked as PLAT-03;
this review adds items 3 and 4 to its scope. Effort M–L.

**G5. OPTICS is serial and quadratic by design. CONFIRMED.** The frontier is sequential with about
n/64 dispatches and O(n²) work; Xi extraction is one invocation (`gpu-optics-xi.ts:29`); the class
says so itself (`gpu-optics.ts:82-88`). Better: OPTICS reachability ordering is the Prim order of
the mutual-reachability minimum spanning tree, which HDBSCAN already builds by Borůvka. A tree walk
of n−1 edges replaces O(n²). Tracked as PP04; the HDBSCAN route is new. Effort M.

**G6. Dense conditional simulation. CONFIRMED.** (n+q)² f32 factor in one binding, n+q ≤ 4,096,
blocked Cholesky with five single-invocation steps (`conditional-simulation-blocked.ts:120-323`).
f32 failure arrives earlier ([SM] N4). Better: sequential Gaussian simulation on ordered kNN
(Vecchia, reusing local kriging; STM-07 builds the likelihood side) or an unconditional field by
FFT/turning bands plus a kriged residual. Effort L.

**G7. Panel lag ML is a dense toy. CONFIRMED.** One-thread Gaussian elimination determinant and a
Gauss–Jordan inverse (`gpu-spatial-panel-lag-maximum-likelihood.ts:212-214,306-307`), N ≤ 32.
Better: ln|I_NT − ρ(I_T⊗W)| = T·ln|I_N − ρW|, so the trace series of `GPUSpatialWeightsTraces` on
the N-row W is enough, as in the cross-section lag ML. The same identity removes the T-fold
materialised I_T⊗W in panel error ML ([SM] §4). Effort S–M.

**G8. Recompiling numeric props. CONFIRMED by grep.** `GPUParameterBuffer` appears in only three
files in scope, all trajectory parameter files. Literals pasted into WGSL from props:
`GPUHotSpotAnalysis` and `GPUEmergingHotSpots` selfWeight, multivariate Geary significance,
lattice cell size, transform bandwidth, contiguity snap tolerance, summary symmetry tolerance,
focal noDataValue, local RBF epsilon/searchRadius/smoothing, local kriging model and searchRadius,
conditional simulation mean and tolerance, encounter and closest-approach cell sizes, generalized
GWR selection bandwidth, probit and panel-lag tolerances (file:line in section 2). Most other
contributors use a per-frame `parameters` view. Effort S each.

**G9. Small serial steps that should use published primitives. CONFIRMED.**
- LOSH mean: one thread sums all rows (`gpu-local-variability.ts:136`) → `GPUReduction`.
- Scan margins: "two serial prefix sums" over zones (`gpu-spatial-scan-statistic.ts:990-1004`) →
  `GPUScan`.
- Resample arc length serial per track ([MV]) → segmented `GPUScan`.
- AZP's private Shiloach–Vishkin components (`gpu-azp-regions.ts:383-552`, [SS]) overlap published
  `GPUGraphAlgorithmsConnectedComponents`. SUSPECTED fit: the published class takes its own topology
  type, not `GPUSpatialWeights`.
Effort S each.

**G10. `GPUKMeans` shadows a published class. CONFIRMED.** `@luma.gl/graph/gpu-vector-search`
exports `GPUKMeans` (N-D, evenly spaced seeds, means by one invocation per component). The local
class has the same name, is 2D only, and is the better GPU design (k-means++ race, restarts,
fixed-order chunked means). Its doc points at `@luma-contributors/analysis/gpu-vector-search`
(`gpu-kmeans.ts:234`), a path that does not exist. Upstream-first says improvements go to luma.gl.
Better: upstream k-means++, restarts and chunked means; rename or delete the local class. Effort M.

**G11. Already ranked by the earlier reports (cited, not re-derived).** Restriction pairwise O(nA·nB)
on one thread and similarity top-k O(T·P·K²) (MOB-04); local permutation O(P·Σk²) swap lookup
([SS] rec 8); Ward sequential O(n) graph nodes and SKATER unrolled steps ([SS] §4, root cause
PLAT-04); f32 counts capping statistics at 2^24 rows ([SS] §4); `GPUSpaceTimeVariogram` brute force
(FX-11).

## 4. Consistency findings

1. **Status.** Shared port types are used by OLS, the ML and GM estimators, 2SLS and diagnostics
   (`GPUSolverStatusPort`), `GPUKriging` (`GPUInvalidStatusPort`, `GPUUncertaintyStatusPort`),
   `GPUTransitionDiagnostics` (`GPUIterationStatusPort`), and `GPUKMeans` and the traces
   (`contracts/status`). That is 10 of 87 files. The rest use one-word `overflow` views
   (statistics), positional status words in two layouts (regionalization), per-row codes (GWR,
   local kriging, Kalman, restriction) or nothing ([SS] §5.5). Converge on: shared ports for summary
   status, plus a per-row code view only where rows fail independently.
2. **Inputs.** No contributor in scope accepts chunked vector input (`GraphVectorView`, zero
   matches). Only CSR weights have a chunked twin (33 files), and regionalization, GWR, clustering
   and all trajectory classes take one packed binding. Converge on: chunked CSR everywhere W is read
   (regionalization and `GPUNeighborhoodSummary` are the gaps, [SS] §4), packed positions elsewhere
   with an explicit binding check.
3. **Spatial context.** Missing on contributors that measure distance: `GPULocalKriging`,
   `GPULocalRBF`, `GPUNaturalNeighbor`, `GPUConditionalGaussianSimulation`, `GPUTrajectoryKalman`,
   `GPUTrajectoryRestriction`, `GPUTrackSimilaritySearch`, and the polygon-weights producers
   (contiguity, fuzzy, island; ST09). Interpolation has a resolver (`separation-wgsl.ts:35`) that
   only IDW and `GPUKriging` call. CSR statistics correctly have none.
4. **Validation.** Bare asserts concentrate in OPTICS (`gpu-optics.ts:103-120`), conditional
   simulation (9), similarity search (11), restriction (4), Kalman (2) and
   `prepare-trajectory-observations.ts` (4). Everything else uses `validateCapacity` or
   `throw new Error`. Converge on the helpers with named limits.
5. **Row caps.** Three regimes: 2^24 (f32 counts, statistics), 65,536 (nine regression classes, from
   serial loops or O(n²) scans, G1–G2), and small reference caps (4,096 OPTICS and conditional
   simulation; 64 rank Markov; 32 panel lag; 512 fuzzy contiguity; 128 MGWR inference). Only the
   first is a numeric property; the others are design choices and should be named as such in the
   coverage matrix.
6. **Parameters.** Per-frame values travel four ways: a uint32 `parameters` view, a float32
   `parameters` view, `GPUParameterBuffer` (trajectories only), or baked literals (G8).
   [SS] §5.2 lists the packing layouts. Converge on `GPUParameterBuffer` for new code and keep
   `parameters` views where they exist.
7. **Names and layouts** ([SS] §5, [MV] §2, cited): `minimumPoints` versus `minimumSamples`;
   `epsilon` versus `maximumDistance`; four region-label conventions; trajectory track delimiters
   `trackOffsets` versus `offsets` versus `lineOffsets`; f32 versus word timestamps; noise label
   `0xffffffff` versus NaN versus 0 for "not tested".
8. **Placement.** Raster focal statistics in `spatial-interpolation`; mapclassify fit and class
   assignment in `distribution-dynamics`; a CPU Philox twin in `permutation-inference`. See §6.

## 5. Duplication and shared-helper candidates

### 5.1 Small dense solvers across `modules/analysis/src` (grep, then read at each site)

| # | Kind | Location | Users |
|---|---|---|---|
| 1 | Cholesky factor / solve / inverse diagonal | `gsa/spatial-regression/spatial-regression-solve.ts:36,59,77` | OLS, lag/error ML, GM, 2SLS, diagnostics, SUR, probit, regimes, IV covariance, panel lag, MGWR (15 files) |
| 2 | Equilibrated Cholesky (wraps 1) | `gsa/spatial-regression/local-weighted-least-squares-wgsl.ts:71-98` | GWR, generalized GWR, nonstationarity test |
| 3 | Dense Cholesky, cooperative | `gsa/spatial-interpolation/gpu-conditional-gaussian-simulation.ts:438-445` | conditional simulation |
| 4 | Blocked Cholesky | `gsa/spatial-interpolation/conditional-simulation-blocked.ts:125-139` | conditional simulation |
| 5 | Gaussian elimination, partial pivot | `gsa/spatial-interpolation/gpu-kriging.ts:628-660` | `GPUKriging` |
| 6 | Ruiz-equilibrated LU, partial pivot | `gsa/spatial-interpolation/equilibrated-solve-wgsl.ts:29-90` | local kriging, local RBF |
| 7 | Gaussian elimination log-determinant | `gsa/spatial-regression/gpu-spatial-panel-lag-maximum-likelihood.ts:212-214` | panel lag ML |
| 8 | Gauss–Jordan inverse | same file `:306-307` | panel lag ML |
| 9 | Gauss–Jordan 4×4 inverse | `gsa/trajectory-estimation/trajectory-kalman-full-covariance.ts:44-51` | Kalman full covariance |
| 10 | Gaussian elimination 6×6 | `pointcloud-analysis/pointcloud-plane-icp.ts:41-43` | plane ICP |
| 11 | Cyclic Jacobi symmetric eigen (P×P) | `gsa/spatial-regression/spatial-regression-solve.ts:111` | OLS, ML, GM, 2SLS, GWR condition numbers |
| 12 | One-sided Jacobi (Belsley) | `spatial-regression-solve.ts:182` | OLS, ML, GM, 2SLS |
| 13 | Givens QR + one-sided Jacobi SVD | `gsa/spatial-regression/local-collinearity-wgsl.ts:20-122` | local collinearity |
| 14 | Jacobi 3×3 eigen | `pointcloud-analysis/gpu-pointcloud-features.ts:82-94` | point-cloud features |
| 15 | Jacobi 4×4 eigen (Horn quaternion) | `pointcloud-analysis/gpu-pointcloud-icp.ts:274-293` | point-to-point ICP |
| 16 | Modified Gram–Schmidt QR least squares | `raster-analysis/remote-sensing/raster-kernel-utils.ts:60-100` | seasonal break (+extended), unmixing, raster temporal |
| 17 | Power iteration (first eigenvector) | `dataframe-analysis/composite-indicators/gpu-composite-score.ts:~715-744` | composite score PCA |
| 18 | CPU Householder + QL eigenvalues | `gsa/spatial-regression/spatial-exact-log-determinant.ts` | exact log-det (≤ 1,024 rows, CPU) |

So **17 distinct GPU implementations plus one CPU one**: three Cholesky (1, 3, 4; 2 is a wrapper),
six elimination or LU variants (5–10), five Jacobi variants (11–15), one MGS QR (16), one power
iteration (17). Sparse solvers (Jacobi-preconditioned CG in `landscape-ecology/sparse-resistance-solver.ts`)
are excluded.

One shared WGSL module (`analysis-core/kernels/small-dense-solve-wgsl.ts`) could hold:
- Cholesky with optional diagonal equilibration and a **relative** pivot tolerance (absorbs 1 and 2,
  and closes FX-09);
- Ruiz-equilibrated partial-pivot LU with log-determinant and inverse outputs (absorbs 5–10; the
  4×4 and 6×6 cases are size parameters);
- cyclic Jacobi symmetric eigen and one-sided Jacobi SVD (absorbs 11–15; fixed-size 3×3 and 4×4 are
  size parameters);
- MGS least squares (16, and the QR half of 13).

Keep separate: the blocked Cholesky (3/4 are a large-matrix workgroup algorithm, not a small solve;
merge 3 into 4 inside the family), power iteration, and the CPU eigen path. Effort M, mostly tests.

### 5.2 Other duplication

| Duplicate | Where | Recommendation |
|---|---|---|
| Kriging ×2 | `gpu-kriging.ts`, `gpu-local-kriging.ts` | one class (G3) |
| Point candidate generation ×5 | G4 list | PLAT-03 plus a ring-walk helper in analysis-core |
| Fisher–Yates ×2 | `gpu-multivariate-local-geary-tiled.ts:~290-310` vs `getLocalPermutationKernelWGSL` ([SS] §2.2) | Geary test becomes a local permutation adapter |
| Column standardisation ×2 | `regionalization-inputs.ts:86-158` vs `gpu-spatial-weights-minimum-spanning-tree.ts:265-285` ([SS]) | one `createStandardizeColumnsNode` (FX-12) |
| Connected components | `gpu-azp-regions.ts:383-552` vs published `GPUGraphAlgorithmsConnectedComponents` | adapter, or lift into `GPUSpatialWeightsSummary` |
| k-means | local `GPUKMeans` vs published `GPUKMeans` | upstream (G10) |
| Focal statistics | `gpu-focal-statistics.ts` vs `raster-analysis` focal family | move |
| Class assignment | `gpu-class-assignment.ts` vs `GPUClassBreaks` class rule | move next to `GPUClassBreaks` |
| Fixed-order 256-lane reductions | many ([SS] §2.2) | analysis-core helper |
| Hash RNG vs Philox | k-means, AZP, Ward ([SS] §2.2) | Philox for sampling, hash only for ties |
| Trajectory ports ×4 | [MV] §2 | TC-06 (tracked) |

## 6. Comprehensiveness (purely spatial)

Already tracked, not repeated: geographical detector q, REDCAP, contiguity-order correlogram,
spatial entropy, colocation quotient (STX-08); Matérn and per-frame covariance (STM-01); Vecchia
(STM-07); GTWR (STM-06); OPTICS at scale (PP04); intensity models (PP01/02); global RBF and indexed
global kriging (IN05/IN02); LMC fitting (IN04); spreg HC/HAC, regimes inference, panels, probit,
SUR, MGWR AICc (ST01–ST06b); one point index (PLAT-03). All space-time gaps (ST-DBSCAN, space-time
K, Jacquez, prospective scan, space-time weights, staypoints, trip segmentation, co-movement inputs)
are tracked in roadmap §2.15.

Missing, ranked by how often practitioners use them (absence CONFIRMED by grep for each name):

| # | Missing | Reference | GPU-suitable | Reuses | Effort |
|---|---|---|---|---|---|
| 1 | Spatial Durbin (SDM), SLX, SDEM | spreg, spatialreg, GeoDa | yes | `GPUSpatialLag` for WX columns, then OLS / lag ML / error ML; recipe plus impacts for SDM | S |
| 2 | GWR prediction at unsampled locations | mgwr `predict`, GWmodel | yes (row per query) | `fitLocation` with query positions and the GWR grid index | S |
| 3 | Multivariate attribute clustering (k-means on c columns, k-medoids, Gaussian mixture) | sklearn, GeoDa | yes | upstream N-D `GPUKMeans`; GMM is EM over row-parallel responsibilities | S (k-means), M (GMM, k-medoids) |
| 4 | SARAR / GM_Combo, lag and error with regimes, GM with endogenous regressors | spreg | yes | `GPUSpatialErrorGM` + `GPUSpatialTwoStageLeastSquares` moments | M |
| 5 | Rate smoothers: spatial median, headbanging, disk/kernel, age-standardised rates | esda.smoothing, GeoDa | yes | CSR neighbour loop; median via `GPUNeighborhoodSummary` | S each |
| 6 | Mean-shift and spectral clustering | sklearn | mean-shift yes; spectral needs Lanczos | radius CSR per iteration; DBSCAN union-find for modes | M / L |
| 7 | Triangulation weights: Delaunay, Gabriel, relative neighbourhood, Voronoi | libpysal `Graph.build_triangulation` | yes | `delaunay-tessellation` output → CSR with edge filters | M |
| 8 | Weights connected components, `n_components` | libpysal | yes | published connected components or AZP kernel | S |
| 9 | Spatial Gini (Rey–Smith) | inequality | yes | `GPUInequality` pairs split by W | S |
| 10 | Partial / multivariate local Moran, spatial silhouettes, local neighbour match test | esda (2024), GeoDa | yes | permutation adapters; neighbour search on attribute kNN | S–M each |
| 11 | Point process simulation: Thomas, Matérn cluster, SSI, Strauss | spatstat, pointpats | cluster processes row-parallel; Strauss/SSI need MCMC | Philox streams, study window | S / M |
| 12 | Anisotropy, nested structures, indicator and regression kriging, 3D kriging | gstat, GSTools, PyKrige | yes | merged kriging (G3), covariance WGSL | M |
| 13 | Mark correlation function | spatstat `markcorr` | yes | pair histogram with mark products | S–M |
| 14 | Unconditional random fields at grid scale (FFT, randomisation) | GSTools SRF | yes | FFT is not in the published primitives; turning bands is row-parallel | M |

Out of place (drop or move): `GPUFocalStatistics` (raster), `GPUClassificationFit` and
`GPUClassAssignment` (column classification), local `GPUKMeans` (upstream), `PhiloxStream`
(analysis-core). Reference-sized classes that should be reworked or labelled "reference" in the
public docs: `GPUSpatialPanelLagMaximumLikelihood` (32 × 32 × 7), `GPURankMarkov` (64 rows),
`GPUFuzzyContiguity` (512 features), `GPUOPTICS` (4,096 rows).

## 7. Top ten actions

| # | Action | Effort | Files |
|---|---|---|---|
| 1 | Replace the one-thread fits with tiled moments plus a one-thread solve; lift the 65,536 caps (G1) | M | `gpu-probit-regression.ts`, `gpu-restricted-regime-regression.ts`, `gpu-spatial-regime-regression.ts`, `gpu-spatial-panel-regression.ts`, `gpu-instrumental-variables-covariance.ts`, `spatial-sur-nodes.ts`, new tile helper in `spatial-regression/` |
| 2 | One small dense solver WGSL module with relative pivot tolerance and condition estimate; migrate the 17 sites (§5.1; closes FX-09, FX-10) | M | `analysis-core/kernels/`, `spatial-regression-solve.ts`, `equilibrated-solve-wgsl.ts`, `gpu-kriging.ts`, panel lag, Kalman, point cloud, raster utils |
| 3 | Merge `GPUKriging` and `GPULocalKriging`: index, per-frame model, one solver, `spatialContext` (G3, STM-01) | M | `spatial-interpolation/*kriging*`, `separation-wgsl.ts`, `grid-neighbor-scan.ts` |
| 4 | One point-neighbour producer used by DBSCAN, HDBSCAN, OPTICS, IDW/kriging, GWR (G4, PLAT-03) | M–L | `neighbor-search/`, `spatial-clustering/`, `spatial-interpolation/grid-neighbor-scan.ts`, `spatial-regression/gwr-grid-index.ts` |
| 5 | MGWR on the GWR grid index for bounded kernels (G2) | M | `gpu-multiscale-geographically-weighted-regression.ts`, `gwr-grid-index.ts` |
| 6 | OPTICS from the HDBSCAN mutual-reachability MST (G5, PP04) | M | `gpu-optics.ts`, `gpu-optics-xi.ts`, `gpu-hierarchical-density-clustering.ts` |
| 7 | Upstream k-means improvements to `@luma.gl/graph` `GPUKMeans`; remove the name clash and the wrong doc path (G10) | M | `gpu-kmeans.ts`, luma.gl PR |
| 8 | Move baked numeric props to `parameters` views or `GPUParameterBuffer` (G8) | S each | files in G8 |
| 9 | Panel ML via T·ln\|I−ρW\| and N-row traces; drop the materialised I_T⊗W (G7) | S–M | `gpu-spatial-panel-*-maximum-likelihood.ts` |
| 10 | Contract clean-up: shared status ports, `spatialContext` on the seven distance contributors, helpers instead of bare asserts, move the four misplaced classes (§4, §6) | M | files named in §4.1–4.4 and §6 |

Not checked: runtime behaviour or performance (nothing was run); whether the published
`GPUGraphAlgorithmsConnectedComponents` and `GPUPointSpatialQuery` accept this library's CSR and
position layouts; the serial step in `spatial-sur-nodes.ts:97` (not read); trajectory kernels
beyond [MV].
