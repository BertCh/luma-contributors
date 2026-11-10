# Code review: exploratory spatial statistics foundation

Scope: `modules/analysis/src/gpu-spatial-analysis/{spatial-weights, neighbor-search, neighborhood-summary, pair-gather, spatial-autocorrelation, global-spatial-statistics, permutation-inference, multiple-testing, rate-smoothing, spatial-clustering, outlier-detection, similar-locations, spatial-regionalization, geographic-distribution, point-density, line-density, shared}`. That is about 45k lines.

Method:
- I read the main class, index and documentation block of every directory.
- I traced the weights, neighbour-search, permutation, multiple-testing and global/local autocorrelation kernels myself.
- Three read-only sub-reviews covered clustering/LOF/similar-locations, regionalization, and density/distribution/EB. I spot-checked their load-bearing claims (SKATER SSD, MST standardisation, max-p rounds, k-means centring, EB guard) against the code.
- Nothing was built or run.

Path prefix below: `gsa/` = `modules/analysis/src/gpu-spatial-analysis/`.

Tags:
- **CONFIRMED** means I traced the code path.
- **SUSPECTED** means plausible from the code, with no failing input constructed or run.

Roadmap rows already covering an item are not repeated: WEIGHTS7 (chunked transpose), PERM-01 (batching with stopping), PP04 (OPTICS 4,096 rows), KG01f, ST07.

Overall verdict: the statistics core is unusually careful and well documented. It has:
- one shared CSR contract;
- one neighbour definition, as a WGSL snippet;
- fixed-order reductions with no float atomics;
- two-pass centred moments;
- an exact conditional-permutation null with Philox streams;
- statsmodels-exact FDR.

Most real risk sits in three places:
- the clustering, regionalization and density modules, which do not share the weights or neighbour-search layer;
- a few sign or standardisation edge cases;
- hard 2D-only and per-row neighbour caps.

---

## 1. Inventory

"Idx" = indexed (grid/BVH/CSR) vs brute force (BF). Oracle = pinned reference used by specs (fixtures under `modules/analysis/test/oracles/fixtures/`; versions are recorded inside each fixture by its generator).

| Symbol | File | Operation | Capacity limits (where) | Idx/BF | Reference oracle | Spec(s) |
|---|---|---|---|---|---|---|
| `GPUSpatialWeights` (type), `getGPUSpatialWeightsNeighborLoopWGSL` | `gsa/spatial-weights/spatial-weights.ts:38-59,156-179` | Shared CSR (`offsets`, `neighbors` u32, `weights` f32, optional `distances`) and the single "active slot" definition | One storage binding per view (u32 slots ≤ `maxStorageBufferBindingSize/4`, 33.5M on 128 MiB default) unless chunked | n/a | libpysal (`statistics-spatial-weights.json`) | `test/gpu-spatial-analysis/spatial-weights/*.spec.ts` |
| `GPUChunkedSpatialWeights` + `GPUChunked*` | `gsa/spatial-weights/gpu-chunked-spatial-weights*.ts` | Row-chunked CSR; consumers dispatch once per chunk | Each chunk fits a binding; one row > limit is refused | n/a | same | `permutation-inference/chunked-spatial-weights.node.spec.ts`, `weights7-*` |
| `GPUNeighborSearch` | `gsa/neighbor-search/gpu-neighbor-search.ts` | Exact kNN / distance band to CSR; binary, inverse-distance, 5 kernels, adaptive kNN bandwidth (libpysal eps) | k ≤ 32 (`:87`); rows < 2^31 (`:303`); grid cells ≤ 2^25-1 (`:435-440`); radius rows insertion-sorted O(m²) per row (`:1096-1111`) | Grid (`GPUGridIndex`) or Hilbert BVH | libpysal (`statistics-neighbor-search.json`, clustered fixture) | `neighbor-search/*.spec.ts` (BVH A/B, row-standardise, benchmark) |
| `GPUContiguityWeights` | `gsa/spatial-weights/gpu-contiguity-weights.ts` | Queen/rook by snapped-vertex / edge radix grouping | Pair list capacity; O(m²) pairs per shared vertex (`:157-162`) | Sort-based | libpysal | `gpu-spatial-weights*.spec.ts`, `spatial-weights-libpysal.node.spec.ts` |
| `GPULatticeWeights` | `.../gpu-lattice-weights.ts` | Raster queen/rook/radius | radius ≤ 32 (`:42`) | Direct | libpysal `lat2W` | same |
| `GPUFuzzyContiguity` | `.../gpu-fuzzy-contiguity.ts` | libpysal `fuzzy_contiguity` | **≤ 512 features, ≤ 65,536 vertices, dense distance table** (`:73-74`) | BF | libpysal | ST09 specs |
| `GPUIslandAttachment` | `.../gpu-island-attachment.ts` | libpysal `attach_islands` | O(n) scan per island (`:158`) | BF | libpysal | same |
| `GPUSpatialWeightsTransform` (+chunked) | `.../gpu-spatial-weights-transform.ts` | R, B, D, V, kernel, symmetrise | – | CSR | libpysal | `gpu-spatial-weights.spec.ts` |
| `GPUSpatialWeightsAlgebra` (+chunked) | `.../gpu-spatial-weights-algebra.ts` | union/intersection/difference, higherOrder (≤ 32, `:50`), selfWeight, subgraph, block | Higher order: work ∝ frontier² | CSR | libpysal `Graph` | `gpu-spatial-weights-algebra*.spec.ts`, parity-wp1 |
| `GPUSpatialWeightsSummary`, `…Validation`, `…Transpose`, `GPUSpatialLag` | `.../` | S0/S1/S2, cardinalities, asymmetry, islands; invariant checker; Wᵀ; lag | – | CSR | libpysal | same |
| `GPUNeighborhoodSummary` | `gsa/neighborhood-summary/gpu-neighborhood-summary.ts` | Weighted lag, mean, SD, min/max, median, mode, local entropy | median ≤ 64 members (`:36`); categorical O(k²) | CSR | momepy/libpysal-style CPU oracle | `neighborhood-summary/*.spec.ts` |
| `GPUPairGather` | `gsa/pair-gather/gpu-pair-gather.ts` | sjoin attribute gather | output capacity | n/a | GeoPandas semantics, CPU | `gpu-pair-gather.spec.ts` |
| `GPUGlobalSpatialStatistics` | `gsa/global-spatial-statistics/gpu-global-spatial-statistics.ts` | Moran I, Geary C, General G, bivariate Moran, join counts (BB/BW/WW) with analytic moments | rows < 2^24 (`:119`) | CSR (+chunked, transpose or pairwise symmetry lookup) | esda (`statistics-global-autocorrelation.json`) | `global-spatial-statistics/*.spec.ts` |
| `GPULocalMoran` | `gsa/spatial-autocorrelation/gpu-local-moran.ts` | LISA with exact conditional-randomisation z | rows < 2^24 | CSR (+chunked) | esda (`statistics-local-autocorrelation.json`) | `gpu-local-moran.spec.ts`, `spatial-autocorrelation.node.spec.ts` |
| `GPUHotSpotAnalysis` | `.../gpu-hot-spot-analysis.ts` | Gi*/Gi analytic z, bins, FDR | rows < 2^24 | CSR (+chunked) | esda, ArcGIS formula | `gpu-hot-spot-analysis.spec.ts` (has 1e6-offset test at `:602`) |
| `GPULocalGeary`, `GPUSpatialPearson`, `GPUGammaStatistic` | `.../gpu-additional-spatial-statistics.ts` | Local Geary (descriptive), edge Pearson, Hubert Γ | edge association ≤ 256 groups per chunk (`:54`) | CSR | esda | `additional-spatial-statistics.spec.ts` |
| `GPULocalSpatialHeteroscedasticity`, `GPUMultivariateLocalGeary`, `GPULOSHResidualPermutationTest`, `GPUMultivariateLocalGearyPermutationTest` | `.../gpu-local-variability.ts`, `gpu-multivariate-local-geary-*.ts` | LOSH, multivariate local Geary + randomisation | exact path ≤ 256 rows; tiled ≤ 2^24-1 rows, ≤ 9,999 perms, ≤ 128 neighbours (`gpu-multivariate-local-geary-permutation-test.ts:47-51`, `-tiled.ts:33-34`) | CSR | esda `Geary_Local_MV` | `multivariate-geary-tiled.node.spec.ts` |
| `GPUAssuncaoReisStandardization` | `.../gpu-rate-standardization.ts` | EB-standardised rate for Moran | rows < 2^24 | – | esda `assuncao_rate` | `statistics-rate-standardization.json` |
| `GPULocalPermutationTest` | `gsa/permutation-inference/gpu-local-permutation-test.ts` | Conditional permutation for local Moran, Geary, G, G*, bivariate Moran, Lee's L, (bivariate) local join counts; FDR; Besag–Clifford stopping | ≤ 64 active neighbours per tested row (512 with `extendedNeighborScratch`) (`:79,86`); perms ≤ 2^20; per-draw swap lookup linear in k → O(P·Σk²) | CSR (+chunked) | esda (`statistics-local-autocorrelation.json`, `statistics-local-join-counts-and-bivariate.json`, `statistics-permutation-alternative.json`) | `gpu-local-permutation-test.spec.ts`, `new-statistics*.spec.ts`, `permutation-*.spec.ts` |
| `GPUGlobalPermutationTest` | `.../gpu-global-permutation-test.ts` | Total permutation (swap-or-not bijection) for Moran, Geary, G, BV Moran, Lee's L; reference distribution and histogram | `(P+1)·ceil(n/2048) ≤ 16,777,215` (`:127-138`): 2^20 perms gives 30,720 rows, 999 perms gives about 34M rows | CSR | esda `p_sim`/`z_sim` | `gpu-global-permutation-test.spec.ts`, `permutation-shuffle.node.spec.ts` |
| `GPUMultipleTestingAdjustment` | `gsa/multiple-testing/gpu-multiple-testing-adjustment.ts` | Bonferroni, Holm, BH, BY | rows < 2^24 (`:43`) | GPU sort + scans | statsmodels `multipletests` (`statistics-multiple-testing.json`) | `gpu-multiple-testing-adjustment*.spec.ts` |
| `GPUEmpiricalBayesRates`, `GPUSpatialEmpiricalBayesRates` | `gsa/rate-smoothing/*.ts` | Global EB (Marshall), spatial EB | u32 inputs exact < 2^24 | CSR | esda 2.10 (`statistics-rate-smoothing.json`) | `rate-smoothing/*.spec.ts` |
| `GPUSpatialClustering` (DBSCAN) | `gsa/spatial-clustering/gpu-spatial-clustering.ts` | DBSCAN with CAS union-find | minPts ≤ 2^24; cluster capacity; no row cap | **Own** `GPUGridIndex` 3×3 walk, not `GPUNeighborSearch` | sklearn 1.9.1 (`parity-wp1-spatial-clustering.json`, `parity-wp1-clustering-modes.json`) | `gpu-spatial-clustering*.spec.ts`, parity specs |
| `GPUHierarchicalDensityClustering` (HDBSCAN) | `.../gpu-hierarchical-density-clustering.ts` | Core-distance kNN, Borůvka MST, condensed tree, EOM | rows ≤ 2^20 (`:68`), minSamples ≤ 32 (`:63`) | `GPUNearestFeatureJoin` kNN + spatial-join BVH (third index family) | sklearn 1.9.1 (`spatial-clustering-hdbscan.json`) | `gpu-hierarchical-density-clustering.spec.ts`, AB spec |
| `GPUOPTICS`, `GPUOPTICSXi` | `.../gpu-optics.ts`, `gpu-optics-xi.ts` | OPTICS + DBSCAN/Xi extraction | rows ≤ 4,096 (`:103`, roadmap PP04); `maximumDistance` must be finite | BF O(n²) | sklearn 1.7.2 | `point-process/optics-selection.spec.ts`, point-process advanced |
| `GPUKMeans` | `.../gpu-kmeans.ts` | Lloyd with exponential-race k-means++ and restarts | k ≤ 256, iterations ≤ 64, restarts ≤ 32 (`:54-60`) | BF O(nk) | sklearn 1.9.1 (relocation fixture only) | `gpu-kmeans*.node.spec.ts` |
| `GPULocalOutlierFactor` | `gsa/outlier-detection/gpu-local-outlier-factor.ts` | LOF from a kNN CSR | inherits k ≤ 32 from `GPUNeighborSearch` | CSR | sklearn (24+2 point fixture) | `gpu-local-outlier-factor*.spec.ts` |
| `GPUSimilarLocations` | `gsa/similar-locations/gpu-similar-locations.ts` | Attribute similarity (z or rank) to reference mean | rows·cols ≤ binding limit (`:50-52`) | BF O(nc); rank O(n²) per column below 4,096 rows | NumPy/SciPy float64 | `gpu-similar-locations*.spec.ts` |
| `GPUAZPRegions` | `gsa/spatial-regionalization/gpu-azp-regions.ts` | Greedy batched AZP (no tabu/SA) | columns ≤ 8, iterations ≤ 256 (`:41-42`); local articulation test ≤ 128 rows/degree 64 (`:60,65`) | CSR | spopt AZP (quality only) | `gpu-azp-regions.spec.ts`, `azp-batched-moves.node.spec.ts` |
| `GPUMaxPRegions` | `.../gpu-max-p-regions.ts` | Parallel multi-seed max-p + optional AZP | columns ≤ 8, starts ≤ 64, growth rounds ≤ 4,096 (default 256) (`:128-136`) | CSR | spopt 0.8.0 `MaxPHeuristic` (quality ratio) | `gpu-max-p-regions*.spec.ts` |
| `GPUSpatialWeightsMinimumSpanningTree`, `GPUSkaterRegions` | `.../gpu-spatial-weights-minimum-spanning-tree.ts`, `gpu-skater-regions.ts` | Borůvka MST + SKATER cuts | columns ≤ 32 / 16; regions-1 unrolled steps; tour stats O(Σ subtree sizes) | CSR | `skater-reference.json` (spopt) | `parity-wp1-skater*.spec.ts`, MST parity |
| `GPUWardRegions` | `.../gpu-ward-regions.ts` | Spatially constrained Ward (sequential or mutual-nearest) | columns ≤ 8; **sequential = one unrolled iteration (~15 nodes) per merge, n−k merges** | CSR | scipy/sklearn (unconstrained), `statistics-ward.json` | `gpu-ward-regions.spec.ts`, mutual-nearest specs |
| `GPURegionPartitionEvaluation` | `.../gpu-region-partition-evaluation.ts` | Within/between SS, pseudo-F, contiguity check | ≤ 32 columns | CSR | float64 CPU | evaluation parity specs |
| `GPUGeographicDistribution` | `gsa/geographic-distribution/gpu-geographic-distribution.ts` | Mean/median (Weiszfeld + Vardi–Zhang) centre, standard distance, SDE, linear directional mean; grouped | rows < 2^24; chunked ≤ 256 groups, sorted unlimited | BF reductions | scipy (geometric median, circmean) | `gpu-geographic-distribution*.spec.ts`, parity-wp1 |
| `GPURoseStatistic` | `.../gpu-rose-statistic.ts` | Direction rose, mean angle, Rayleigh z | bins 2..360 | BF | hand values | `gpu-rose-statistic*.spec.ts` |
| `GPUPointDensity` | `gsa/point-density/gpu-point-density.ts` | Count/sum/mean per square or hex cell | deterministic mode ≤ about 11.2M cells (`:89`) | Binning | CPU oracle | `gpu-point-density*.spec.ts` |
| `GPUKernelDensity` | `.../gpu-kernel-density.ts`, `kernel-density-binned.ts` | Exact (gaussian/quartic/epanechnikov) or binned Gaussian KDE; Scott/Silverman; edge correction | cells ≤ 2^25-1 (exact), ≤ about 11.2M (binned) | Grid gather / linear binning + blur | sklearn/scipy/dblquad (`geometry-kernel-density*.json`) | `kernel-density*.spec.ts` |
| `GPULineDensity`, `GPULineLengthPerPolygon` | `gsa/line-density/*.ts` | Clipped length per cell / per polygon | cells ≤ 2^24-1; positions ≤ 8,388,608 | DDA walk / BVH | `parity-wp2-line-density.json` | `line-density*.spec.ts` |

---

## 2. Shared infrastructure and duplication

### 2.1 What is genuinely shared
1. **Weights representation.** There is one CSR type, `GPUSpatialWeights` (`spatial-weights.ts:38-59`), with written invariants:
   - sorted unique neighbours;
   - no self slot;
   - finite non-negative weights;
   - optional `information` (transform, symmetric, provenance).

   `GPUChunkedSpatialWeights` is its row-chunked twin, and consumers loop over `getGPUSpatialWeightsChunkDispatches`. There is no fixed-slot (ELL) representation. Fixed per-row private arrays appear only inside consumers: `maximumNeighbors` scratch in the local permutation test, neighbourhood summary and multivariate Geary.
2. **One neighbour definition.** `getGPUSpatialWeightsNeighborLoopWGSL` (`:156-179`) defines an active slot as positive weight, valid ID, not self, and included. It is inlined by global and local statistics, permutation tests, the summary and LOSH, so "island" and "row re-standardisation over remaining neighbours" mean the same thing everywhere.
3. **Neighbour index.** `GPUNeighborSearch` (`GPUGridIndex` from published `@luma.gl/graph` plus an optional Hilbert BVH from `spatial-join`) is the producer of distance-based weights. LOF consumes its output.
4. **RNG.** Philox 4x32-10 (`analysis-core/kernels/philox-random.ts`, with `PHILOX_PURPOSES` tags) and the stream/Lemire bounded draw in `permutation-inference/permutation-random.ts:61-76` are bit-identical between CPU and WGSL. They are used by:
   - the local test (partial Fisher–Yates with a sparse swap list, `gpu-local-permutation-test.ts:410-495`);
   - the global test (swap-or-not bijection, `permutation-shuffle.ts`);
   - multivariate Geary;
   - max-p seeding.
5. **Permutation machinery.** `permutation-inference-kernels.ts` compacts rows, centres them with a second-pass mean correction (`:466-474`), and builds fixed-order column sums. `permutation-alternative.ts` has a single p-value convention, and statistics plug in through "adapters" (`GPU_LOCAL_GEARY_PERMUTATION_ADAPTER` and others).
6. **Multiple testing.** One node, consumed by the local permutation test, hot spots, local Moran, emerging hot spots and multivariate Geary.
7. **Pair enumeration.** There is no generic one. Distance-pair statistics (correlogram, Ripley, variogram, Clark–Evans) live in `dataframe-analysis/pair-statistics` with their own `pair-statistics-grid.ts`.

### 2.2 Duplication (CONFIRMED by imports)
- **Three neighbour indices for points.**
  - `GPUNeighborSearch` (grid/BVH);
  - DBSCAN's own `GPUGridIndex` 3×3 walk (`spatial-clustering-kernels.ts:100-190`);
  - HDBSCAN's `GPUNearestFeatureJoin` kNN plus the spatial-join BVH.

  On top of these, `dataframe-analysis/pair-statistics/pair-statistics-grid.ts` is a fourth for the correlogram and Ripley, and OPTICS, k-means and island attachment are brute force. None of clustering, OPTICS, k-means or the distance correlogram consumes `GPUSpatialWeights`. So a DBSCAN-on-contiguity, or a correlogram on arbitrary W, cannot be built by composition today.
- **Two Fisher–Yates loops.** `GPUMultivariateLocalGearyPermutationTest` (tiled path, `gpu-multivariate-local-geary-tiled.ts:~290-310`) re-implements the conditional draw instead of using `getLocalPermutationKernelWGSL`. It does share the Philox stream.
- **Two column-standardisation kernels in regionalization.** `regionalization-inputs.ts:86-158` skips non-finite values; the MST copy (`gpu-spatial-weights-minimum-spanning-tree.ts:265-285`) does not. See finding F5.
- **Many local fixed-order tree reductions.** Each contributor carries its own 256-lane `reduceSum`. They are deterministic, but there are several copies, for example `gpu-max-p-regions.ts` finalize, the global statistics column sums and `grouped-sum-reduction.ts`.
- **RNG families.** k-means uses a seeded integer hash/murmur mix, and AZP and Ward use `hashUint32(row ^ seed)` (`shared/hash-uint32.ts`) for tie-breaking, not Philox. That is acceptable for ties, but k-means++ sampling is not reproducible against the Philox CPU twin.

### 2.3 Time-readiness of each building block
| Block | Hard-coded 2D? | Change needed for a time dimension | Effort |
|---|---|---|---|
| `GPUSpatialWeights` CSR | No: topology-only, row IDs are opaque | Index rows as `slice·nCells + cell`. Every consumer (Moran, Gi*, permutations, FDR) then works unchanged on space-time weights | none |
| `GPUSpatialWeightsAlgebra` | No | Add a `kronecker` / `temporalLag` operation: `W ⊗ T` (contemporaneous neighbours plus lagged self/neighbours, as in emerging hot spots). The per-row two-pointer merge and block-weights code are the template | S–M |
| `GPUNeighborSearch` | **Yes**: `float32x2` positions, 2D lattice, `d² = dx²+dy²` | Space-time cylinder search (`d ≤ r` and `|t_i−t_j| ≤ τ`): add an optional `times` view and a temporal predicate in the radius/kNN loops (S). Pruning by time needs time-sorted targets or a 3D lattice, because the published `GPUGridIndex` is 2D (M) | S/M |
| Philox streams / local permutation | No | Counter word `a` is the row and `b` the permutation. Stratified (within-slice) conditional permutation needs a per-row stratum offset/size instead of "all other n−1". Knox and Mantel already exist in `space-time-tests` | M |
| `GPUGlobalSpatialStatistics` / permutation tests | No | Works on space-time CSR as-is. Lagged bivariate Moran (`x_t` vs `W y_{t−1}`) is just `secondValues` | none |
| `GPUHotSpotAnalysis` | No (CSR) | Already generalised in `GPUEmergingHotSpots` (weights mode) | – |
| DBSCAN / HDBSCAN / OPTICS / LOF | **Yes** (2D grid, f32x2) | ST-DBSCAN needs a temporal predicate in the 3×3 walk (S) but no temporal pruning (M) | S–M |
| KDE / point density / geographic distribution / rose | **Yes**, planar | Time via `groupIds` (distribution, rose) or per-slice weights (KDE) works today. A native space-time KDE cube is M | S–M |
| Regionalization | No (CSR) | Space-time regionalization works on space-time CSR. Contiguity proofs assume symmetric W | none |

---

## 3. Correctness and numerical findings

### F1. Local G / G* permutation tails reverse when the denominator is negative — CONFIRMED
- **Where.**
  - `gsa/permutation-inference/gpu-local-permutation-test.ts:159-162` drops the denominator. The spec is `drawnA: 'dx', term: 'lagA'` (`'lagA + fx'` for G*), and the doc comment says "the statistic up to a positive per-row constant".
  - The dropped constant is `1/(Σx − x_i)` for G and `1/Σx` for G* (`:1108-1109`).
  - The tail counts `simulated >= observed` (`:505`) compare the raw lags.
  - Nothing rejects negative values. There are no `negative` checks in `permutation-inference/`. By contrast, `GPUGlobalSpatialStatistics` NaNs the whole G block on any negative value.
- **Failing input.** `values` all negative, for example temperature anomalies in [−5, −1], with `statistic: 'localG'` and `alternative: 'greater'`.
  - `Σx − x_i < 0`, so a large G_i corresponds to a small raw lag.
  - The reported "greater" p-value is the lower-tail p-value, and `significant` marks cold spots as hot.
  - For G* this happens whenever `Σx < 0`.
  - Two-sided, directed and folded are unaffected because they are symmetric.
- **Same pattern, global test.** The `getisOrdG` scale `1/(n(n−1)·mean² − SS)` (`gpu-global-permutation-test.ts:823`) can be negative for mixed-sign data.
- **Fix.** Either reject (or NaN) negative values for G/G*, as the global statistics do, or multiply the term by `sign(denominator)`.

### F2. Pseudo p-value floor makes FDR / Bonferroni nearly powerless at default permutation counts — CONFIRMED (arithmetic on the documented formula)
- **Where.** p = `min(m(M+1), P+1)/(P+1)` (`permutation-alternative.ts:122-125`). The default is two-sided, so `m = 2` and the minimum attainable p is `2/(P+1)`.
- **Scenario.** n = 10,000 tested rows, P = 999, BH at α = 0.05.
  - The smallest p is 0.002.
  - BH needs `p_(k) ≤ kα/m`, so at least 400 rows must sit at the floor before any discovery is possible.
  - Bonferroni can never reject once m > 25.
  - BY, which is recommended in the docs for local statistics, is about 9.8× stricter still.
- **Status.** This is correct behaviour, but neither the API nor the docs warn about it. Recommendation: expose `getMinimumDetectablePermutations(m, α, method)`, or document that P must exceed roughly `2m/α`. Besag–Clifford stopping (`stopAfterExceedances`) makes large P affordable.

### F3. SKATER cut gain uses variance by subtraction in f32 — CONFIRMED pattern, impact SUSPECTED
- **Where.**
  - `gpu-skater-regions.ts:396-398` has `getSsd = Σx² − (Σx)²/n`.
  - It is evaluated at `:767-769` on region sums that are themselves updated by subtraction (`:850-855`).
  - The MST step centres the columns globally, but after the first cuts sub-region means move away from 0.
- **Scenario.** A region of 10^5 rows whose standardised mean is about 3 with within-SD about 0.05.
  - `Σx² ≈ 9·10^5`, while SSD is about 250.
  - The f32 relative ulp of 6e-8 on 9e5 is about 0.05 per term, plus accumulated subtraction drift.
  - Candidate gains differ by less than the noise, so the cut choice becomes arbitrary.
- `max(gain, 0)` (`:770`) hides negative noise.
- **Fix.** Carry per-region centred sums (Welford/Chan merge), or recompute the SSD from a per-region pivot.

### F4. Max-p construction ranking and AZP incremental sums drift — SUSPECTED
- AZP gains use incrementally updated f32 region sums across up to 256 rounds (`gpu-azp-regions.ts:943-953`). They are never recomputed during the search; only the final objective is recomputed.
- Accepted moves may therefore not truly improve the objective on large high-offset regions. Labels stay valid and contiguous, so this costs only optimality.

### F5. One non-finite value poisons the whole MST/SKATER column — CONFIRMED
- **Where.** `gpu-spatial-weights-minimum-spanning-tree.ts:265-285` sums `values − origin` over all rows with no finite check, computes `mean` and `deviation`, then writes `(x − origin − mean)·inverse` for every row.
- **Failing input.** A single NaN attribute in any row.
  - The column mean becomes NaN, so every scaled value and every edge cost is NaN.
  - The MST and SKATER then run on NaN keys. Bitcast-ordered keys make the tree arbitrary rather than throwing.
- **Contrast.** AZP, Ward and max-p use `regionalization-inputs.ts`, which skips non-finite entries and reports `invalidCount`.
- **Fix.** Reuse `createStandardizeColumnsNode`.

### F6. Max-p enclave stage can leave a disconnected failed-region fragment — SUSPECTED
- **Where.** `gpu-max-p-regions.ts:969-1040`, with enclave rounds `min(maximumGrowthSteps, n+1)` (`:128-136`).
- **Mechanism.**
  - Rows of failed regions and unreached rows are absorbed one hop per round from complete regions.
  - If the round cap is hit mid-absorption, the rows that remain keep the failed label even though absorption may have split them.
  - `finalize-starts` (`:1090-1100`) then counts that label once, as a single region.
- **Plausible input.** A failed region separated from every complete region by a chain of more than about 256 unreached rows, using the default 256 rounds.
- Not constructed. The status reports it as infeasible, but "region" contiguity is not guaranteed for infeasible regions.

### F7. Island handling: one pass, no cascade, and `exclude` is the default — CONFIRMED, documented, a parity trap
- **Default.** `GPU_DEFAULT_ISLAND_POLICY = 'exclude'` (`spatial-weights.ts:115`).
- **Effect.** esda parity needs `islands: 'keep'`, so with default props Moran's I differs from esda whenever islands exist: `n` differs.
- **Cascade.** A row whose only neighbours are islands stays in `n` with zero contribution (`gpu-global-spatial-statistics.ts:157-163`).
- Documented, but every quick-start comparison against PySAL will show a mismatch on real polygon data with islands. Consider defaulting to `'keep'`, or warning when `islandCount > 0`.

### F8. Two-sided p-value is the doubled smaller tail, not esda's — CONFIRMED, documented
- **Where.** `permutation-alternative.ts:14-35`.
- **Detail.** The default `'two-sided'` (`analysis-core/contracts/inference.ts:16`) differs from esda's interpolated-percentile rule. The code itself cites one fixture where esda gives 3/21 and this gives 2/21; `'folded'` matches esda exactly.
- esda's own legacy default is reproducible as `'directed'`, except under ties, where the code returns p = 1 and esda returns 1/(R+1). That deviation is deliberate and correct.
- No defect. Flagged because "matches esda" claims elsewhere need the alternative named.

### F9. Conditional permutation and moment conventions — CONFIRMED correct
- **Local test.** Draws k_i distinct rows from the other n−1 analysis rows, assigned to slots in slot order (`gpu-local-permutation-test.ts:479-492`). This is esda's conditional randomisation.
  - Observed and simulated values use the same f32 expression and summation order (`:1004-1011` against `:492-495`), so tail counts are exact.
  - Rows with `k_i > n−1` are not tested (`:455-458`).
- **Global test.** Total permutation via a keyed swap-or-not bijection with 6⌈log₂n⌉+6 rounds.
- **`GPULocalMoran` analytic variance.** `σ²(N·S1_i − W_i²)/(N−1)` (`gpu-local-moran.ts:190-195`) is the exact without-replacement variance, so it is correct.
- **Moments.** Centred, with a second-pass mean correction (`permutation-inference-kernels.ts:466-474`; `gpu-global-spatial-statistics.ts:226-229`). General G variance is in centred form, which avoids `E[G²]−E[G]²`.
- **Large offsets.** The 1e6-offset test exists only for hot spots (`test/.../gpu-hot-spot-analysis.spec.ts:602`). There is no equivalent for global Moran or the permutation tests (untested, see §5).

### F10. Row standardisation is re-applied after masking — CONFIRMED, intentional divergence
- With `information.transform === 'row'`, every consumer renormalises over the active neighbours.
- PySAL leaves masked or missing neighbours as zeros, which shrinks the lag. Results equal PySAL only when nothing is masked.
- Producers that do not set `information` (uploads) are treated as raw. So an uploaded row-standardised W with masked rows behaves differently from the same W produced by `GPUNeighborSearch`. Documented, but the asymmetry deserves a validation warning.

### F11. Kernel weights exclude the focal unit — CONFIRMED, documented
- **Where.** `GPUNeighborSearch` never lists self (`gpu-neighbor-search.ts:236-240`).
- **Difference from libpysal.** libpysal `Kernel` includes `K(0)` on the diagonal, so Gi* or Moran on kernel weights differs from libpysal unless the caller passes `selfWeight = K(0)`.
- The helper `getGPUSpatialWeightsKernelValue(kernel, 0)` exists; only `GPUHotSpotAnalysis` takes `selfWeight`.

### F12. MST/SKATER and region labels trust symmetry metadata — SUSPECTED
- **Where.** `regionalization-inputs.ts:177-179` skips symmetrisation when `information.symmetric === true` without checking.
- **Requirement.** SKATER's Euler-tour twins require every edge to be listed in both directions, and the MST docs state the requirement but nothing enforces it.
- **Risk.** Weights uploaded with a wrong `symmetric: true` flag, or with duplicate slots, produce undefined trees. `GPUSpatialWeightsValidation` exists but is not invoked.

### F13. k-means: raw f32 accumulation by default — CONFIRMED, documented opt-in
- **Where.** `gpu-kmeans.ts:200-208,293`: `centerCoordinates` defaults to false.
- **Effect.** At UTM coordinates (~4e6 m, ulp 0.5 m) and 1e5 or more points per cluster, centroid sums lose metres, and `tolerance: 0` (the default) may never converge, which wastes iterations up to 64.
- **Other differences.** Convergence is an absolute centre shift, not sklearn's `tol·mean variance`. k-means++ is a single exponential-race D² draw, not greedy `n_local_trials`, so centres will not reproduce sklearn.
- DBSCAN centroids are also raw f32 sums.

### F14. HDBSCAN and DBSCAN discard index-overflow status — SUSPECTED
- **HDBSCAN.** `neighborOverflow` is written but never read (`gpu-hierarchical-density-clustering.ts`, per the sub-review). A truncated kNN becomes infinite core distances, so the affected points are silently labelled noise.
- **DBSCAN.** `gridOverflow` is allocated (`gpu-spatial-clustering.ts:491`) but never inspected.
- **Coincident points.** In HDBSCAN they give lambda = FLT_MAX, which overflows to inf in the stability sums. The comment at `hierarchical-density-clustering-kernels.ts:594` says otherwise, but EOM tie behaviour probably still matches sklearn.

### F15. DBSCAN border points — CONFIRMED deterministic and sklearn-equivalent
- **Rule.** A border point joins the smallest adjacent root, and the root is the minimum core row (`spatial-clustering-kernels.ts:576-618`). Labels rank roots in ascending order.
- **Equivalence.** sklearn expands clusters in ascending first-core order and labels only unlabelled points, so the result is the same up to f32/f64 distance ties.
- **Other conventions.** Noise is `0xffffffff`, not −1. Self counts toward `minimumPoints`, and the epsilon test is inclusive. The CAS union-find has no iteration cap and terminates.

### F16. HDBSCAN versus scikit-learn — CONFIRMED subset
- **Matches sklearn:**
  - self-inclusive core distance;
  - mutual reachability;
  - exact Borůvka MST ordered by `(weight, low, high)`;
  - EOM selection, where equality picks the parent;
  - root excluded, which is `allow_single_cluster=False`.
- **Missing:** `cluster_selection_epsilon`, leaf selection, `allow_single_cluster`, probabilities and outlier scores. These are documented non-goals.
- **Default differs:** `minimumSamples` defaults to `min(minimumClusterSize, 32)`.

### F17. LOF k-distance with ties and self-inclusion — SUSPECTED
- The k-distance is the maximum distance in the supplied row. A radius or tie-extended table (more than k entries) or an uploaded table that includes self diverges from sklearn silently.
- Validation only checks `provenance.criterion` when it is present.

### F18. Geographic distribution numerics — CONFIRMED sound
- Standard distance and the SDE are two-pass about the mean, not `E[x²]−E[x]²`. At 1e6 m with 100 m spread the result is fine to sub-metre.
- Weiszfeld coincidence and convergence tolerances are absolute: 1e-6 and 1e-3 coordinate units. In degrees, a spread below about 1e-3 mis-converges. Documented.
- The SDE angle is counter-clockwise from +x of the minor axis, not ArcGIS's clockwise-from-north. Documented.

### F19. KDE default edge correction — CONFIRMED API trap
- `GPUKernelDensity` defaults to `edgeCorrection: 'uniform'`, with the window equal to the output grid.
- With a viewport-sized grid smaller than the data:
  - edge cells inflate;
  - out-of-window cells become NaN;
  - points outside the window are dropped (counted in `invalidCount`).
- ArcGIS and sklearn have no edge correction, so naïve parity comparisons fail.
- The bandwidth rule is isotropic (the trace of the covariance). "Scott" and "Silverman" are identical in 2D, and the ArcGIS default rule is absent.

### F20. Line density is per-cell length, not ArcGIS Line Density — CONFIRMED naming gap
- `GPULineDensity` computes clipped length per cell divided by cell area.
- There is no search radius and no population field, so it is not ArcGIS's circular-neighbourhood density despite the shared name.

### F21. EB edge behaviour — CONFIRMED, parity rather than bug
- **Clamping.** `GPUEmpiricalBayesRates` clamps the prior variance by default (`clampPriorVariance ?? true`, `gpu-empirical-bayes-rates.ts:228`), which is Marshall/GeoDa. esda does not clamp.
- **All-zero events.** The standardised rate is `0/0 = NaN` (`:421-422`), the same as esda.
- **Spatial EB.** The window includes self; weights are binarised.

### F22. Doc/code mismatches — CONFIRMED
- `GPUHotSpotAnalysis` class doc lists "chunked inputs" as a non-goal (`gpu-hot-spot-analysis.ts:201-202`), but the prop accepts `GPUChunkedSpatialWeights` (`:107`) and dispatches per chunk (`:307`).
- The `GPUMaxPRegions` doc mentions an "O(rowCount) connectivity check per improving candidate" (`gpu-max-p-regions.ts:289`). That is stale: AZP now uses the local bounded articulation test.
- The SKATER cost description in `docs/api/gpu-spatial-analysis/operators-statistics.md:1321-1325` disagrees with the code: steps are O(edges), and the one-time Euler-tour statistics are the quadratic part.

---

## 4. Scaling walls

| Contributor | Wall | Kind |
|---|---|---|
| Every statistic with f32 counts (global/local stats, multiple testing, EB, distribution) | n < 2^24 (16.7M rows) | **Implementation**: counts and ranks are held as f32. A u32 count with f64-emulated or split division would lift it |
| Any whole CSR | slots ≤ `maxStorageBufferBindingSize/4` (33.5M on the 128 MiB default) | **WebGPU-inherent** per binding; already solved by chunked weights for permutation, global/local stats, lag and EB, but **not** for `GPUNeighborhoodSummary` (`weights: GPUSpatialWeights` only), the regionalization ops (16.7M / 8.4M slots), LOF or the producers (`GPUNeighborSearch` writes one CSR) |
| `GPUNeighborSearch` kNN | k ≤ 32 | **Implementation**: a private sorted top-k array per thread. This caps LOF and HDBSCAN `minSamples`, and libpysal-style adaptive kernels with k > 32 |
| `GPUNeighborSearch` radius | O(m²) insertion sort per row; dense bands (m about 10^3 to 10^4) are quadratic per thread | Implementation (a segmented sort exists upstream) |
| `GPULocalPermutationTest` | ≤ 64 active neighbours (512 opt-in); per-draw swap lookup is linear, so cost is O(P·Σk_i²) | Implementation (private memory). Rows above the cap are silently untested, with `overflow` set |
| `GPUGlobalPermutationTest` | (P+1)·⌈n/2048⌉ ≤ 16.7M; 9,999 perms gives about 3.4M rows | **WebGPU-inherent** dispatch size (2^32 invocations), compounded by the choice not to chunk permutations |
| `GPUNeighborhoodSummary` | median ≤ 64 members | Implementation |
| `GPUMultivariateLocalGearyPermutationTest` | exact path ≤ 256 rows; tiled ≤ 9,999 perms, ≤ 128 neighbours | Implementation |
| DBSCAN | Grid; degrades toward O(n²) when cells are much larger than ε (`gridSize` too coarse) or on dense clusters | Implementation |
| HDBSCAN | n ≤ 2^20, minSamples ≤ 32; at the maxima the neighbour buffer is exactly 128 MiB | Mixed (binding size is inherent) |
| OPTICS | n ≤ 4,096, brute force, about n/64 sequential dispatches (roadmap PP04) | Implementation |
| k-means | k ≤ 256, ≤ 64 iterations, BF O(nk) | Implementation (fine for k ≤ 256) |
| Similar locations | rows·cols ≤ 33.5M; rank standardisation O(n²) below 4,096 rows | Binding is inherent |
| `GPUFuzzyContiguity` | **512 features**, 65,536 vertices, dense | Implementation. The coverage matrix row ST09.fuzzy says "Implemented" with no limit noted |
| `GPUIslandAttachment` | O(islands·n) | Implementation |
| AZP / max-p / Ward | columns ≤ 8; max-p scratch `starts·n·(c+3)·4` B in one binding (16 starts × 1M rows × 11 × 4 is about 704 MB, which exceeds the binding) | Max-p scratch is binding-inherent; the column cap is an implementation choice |
| Ward `'sequential'` | One unrolled iteration (about 15 nodes) per merge, so the command graph is O(n−k) nodes and impractical above about 10^4 rows | Implementation; there is no GPU-side loop in the published compiler (same root cause as NW-CENT) |
| SKATER | `regions−1` unrolled steps; Euler-tour statistics O(Σ subtree sizes), quadratic on path-like MSTs | Implementation |
| KDE exact | cost about (5h/cell)² per point; binned Gaussian only | Implementation |

---

## 5. API inconsistencies

1. **Same concept, different names:**
   - minimum neighbourhood count: `minimumPoints` (DBSCAN), `minimumSamples` (HDBSCAN, OPTICS);
   - radius: `epsilon` (DBSCAN), `maximumDistance` / `extractionDistance` (OPTICS), `radius` (neighbour search packed parameters);
   - threshold: `extensiveThreshold` (AZP), `threshold` (max-p);
   - region count: `targetRegionCount` (Ward), `regionCount` (SKATER packed parameters).
2. **Parameter packing differs:**
   - float32 views: neighbour search, DBSCAN, LOF, similar locations, KDE;
   - uint32 views: permutation (`[seedLow, seedHigh, count, f32 bits of level]`), k-means (`[seed, f32 tolerance bits]`), AZP/max-p/Ward `[seed]`, SKATER `[regionCount, minimumRegionSize]`.

   The significance level lives in packed parameters for permutations, but the hot spot / local Moran level comes via parameters while the critical z is a constant.
3. **Weights input:**
   - The statistics, regionalization, EB and LOF consume `GPUSpatialWeights`.
   - DBSCAN, HDBSCAN, OPTICS, k-means and the distance correlogram take raw positions and build private indices.
   - `GPUNeighborhoodSummary` and the regionalization ops accept only whole CSR; the statistics accept chunked CSR too.
4. **Label layout:**
   - Clustering uses dense u32 labels with noise `0xffffffff`.
   - Regionalization labels follow three conventions: AZP keeps the caller's initial IDs, Ward uses the smallest row of the region, SKATER uses the top-of-subtree row, and only max-p is dense `[0, regionCount)`.

   Consumers cannot colour regions without a remap pass.
5. **Status:**
   - Statistics use `output.status.overflow`, a one-u32 view.
   - Neighbour search adds `requiredCount`, `invalidCount` and `candidateOverflow`.
   - Clustering uses a mix of `clusters.status`, `status.count`, OPTICS `status` u32x3 `[visited, complete, invalid]`, and k-means `status.iterationCount`/`converged`.
   - Regionalization uses positional status words in two different layouts (AZP/Ward `[iterationCount, converged, iterationLimitReached, invalidCount]`; max-p `[regionCount, feasible, infeasibleRegionCount, invalidCount, selectedStart, improvementMoveCount]`).

   There is no common `GPUStatusPort` shape across families.
6. **Untested and NaN markers:**
   - Local permutation `exceedances` use `0xffffffff` for not tested, while p-values use NaN.
   - Local Geary classes use 0 for undefined.
   - Bivariate quadrants use 0 for not analysed.
   - The neighbourhood-summary mode uses `GPU_NEIGHBORHOOD_SUMMARY_NO_MODE`.
7. **Units:**
   - Neighbour search and DBSCAN accept longitude/latitude great-circle input.
   - HDBSCAN, k-means, KDE, point density and geographic distribution throw on longitude/latitude.
   - Line density accepts it.

   A user switching DBSCAN to HDBSCAN on lon/lat data hits a hard error.
8. **Island default:** `'exclude'` everywhere (F7), but `GPULocalGeary` returns 0 for kept islands while the permutation test leaves them untested (NaN).
9. **`standardize` defaults to true in regionalization** and `false` / not applicable in similar locations, which uses an explicit `standardization` enum.

---

## 6. Gaps and extension points

Present (do not re-add):
- global join counts BB/BW/WW;
- local join counts (univariate and bivariate BJC/CLC);
- bivariate Moran (global and local);
- Lee's L (global and local);
- local Geary (univariate and multivariate);
- Getis-Ord General G;
- distance-band correlogram (`dataframe-analysis/pair-statistics/gpu-spatial-correlogram.ts`);
- AZP, max-p, SKATER, constrained Ward;
- standard deviational ellipse;
- local entropy (neighbourhood summary);
- Clark–Evans (`gpu-point-pattern-indices.ts`).

| Missing method | Reuses | Effort |
|---|---|---|
| Correlogram on **arbitrary weights / contiguity orders** (`esda` correlogram over `higher_order`) | `GPUSpatialWeightsAlgebra` `higherOrder` (≤ 32) feeding `GPUGlobalSpatialStatistics` per order; recipe only | S |
| Colocation quotient (global CLQ, Leslie–Kronenfeld) and local CLQ (Wang et al.) | `GPUNeighborSearch` kNN or kernel weights, `GPUNeighborhoodSummary` categorical weighted frequencies, `GPULocalPermutationTest` (needs a categorical draw term) | M |
| LICD (Boots local indicators for categorical data) | Local join count path plus a multi-category composition term; permutation adapter | M |
| Moran eigenvector maps / ESF | Needs a symmetric eigen solver on `MCM` (dense: L; Lanczos on CSR: L). The spatial-regression log-determinant machinery is the closest | L |
| Spatially constrained hierarchical clustering, other linkages (single, complete, average) | `WardActiveGraph` contraction and mutual-nearest selection; needs Lance–Williams cost merge on duplicate edges | M |
| REDCAP (first-order single linkage) | `GPUSpatialWeightsMinimumSpanningTree` + SKATER step/gate/cut machinery | S |
| REDCAP full-order / average / complete | Ward graph + per-region MST re-evaluation | M |
| Spectral clustering (spatially weighted) | CSR Laplacian, a new Lanczos/LOBPCG, `GPUKMeans` on the embedding (2D-only today, so it needs N-D), enclave repair from max-p | L |
| Mean-shift | `GPUNeighborSearch` radius CSR per iteration, or the KDE grid gather; mode merging via DBSCAN union-find | M |
| Adaptive-bandwidth KDE (Abramson) | Exact KDE gather with per-point h (index range from max h); pilot from the fixed KDE | M |
| Voronoi-based density | `delaunay-tessellation` directory (out of scope here) for Voronoi areas, then 1/area per site; raster approximation via nearest-feature join | M |
| Nearest-neighbour hierarchical clustering (CrimeStat NNH) | `GPUNeighborSearch` k=1 with a random-NN threshold from Clark–Evans expectation, DBSCAN-style union-find, recursion on cluster centroids | M |
| Geographical detector / q-statistic | Grouped SS via `GPURegionPartitionEvaluation` (within/between SS by label) is almost exactly q = 1 − SSW/SST; add the non-central F p-value | S |
| Global spatial entropy (Batty, Leibovici co-occurrence, Altieri) | `GPUNeighborhoodSummary` categorical frequencies plus a reduction; Leibovici needs pair co-occurrence over CSR | S–M |
| Getis-Ord G **local permutation on negative data** | Fix F1 | S |
| Weights connected-component count (libpysal `n_components`) | AZP's Shiloach–Vishkin component kernel (`gpu-azp-regions.ts:383-552`) lifted into `GPUSpatialWeightsSummary` | S |
| ArcGIS Line Density (radius, population) | Segment walker plus a radial gather | M |
| Space-time weights (`W ⊗ T`, lags) | New `GPUSpatialWeightsAlgebra` operation; every CSR consumer then works unchanged | S–M |
| Space-time neighbour search (temporal radius) | `GPUNeighborSearch` loops + optional `times` view; pruning needs time sort or 3D lattice | S (filter) / M (indexed) |
| ST-DBSCAN | DBSCAN 3×3 walk plus a temporal predicate | S (unpruned) / M |
| Stratified (within-time-slice) conditional permutation | Philox stream unchanged; draw range per stratum in `getLocalPermutationKernelWGSL` | M |

---

## 7. Top recommendations (ranked)

1. **Fix F1** (local and global G permutation tails with non-positive denominators). Reject negative values for G/G* as `GPUGlobalSpatialStatistics` does, or carry the sign. This is a silent hot/cold inversion under `'greater'`/`'lesser'`. S.
2. **Unify point neighbour search.** Make DBSCAN, HDBSCAN (core distances), LOF, mean-shift and the correlogram consume `GPUNeighborSearch` / `GPUSpatialWeights`. Raise k beyond 32, for example with a two-pass or segmented top-k. This removes three private indices and gives clustering lon/lat support and chunking. M–L.
3. **Make the CSR the space-time carrier.** Add a `W ⊗ T` / temporal-lag operation to `GPUSpatialWeightsAlgebra`, and an optional `times` + temporal radius to `GPUNeighborSearch`. Every statistic, permutation test and regionalization op then works on space-time data without change. S–M.
4. **Fix the regionalization numerics.** Reuse `createStandardizeColumnsNode` in the MST (F5). Replace SKATER's `Σx² − (Σx)²/n` with pivoted or centred region sums (F3). Recompute region sums periodically in AZP (F4). S–M.
5. **Document and guard the permutation-count floor for FDR** (F2). Add a helper that computes the minimum P for the chosen method, m and α, and make Besag–Clifford stopping the recommended path for large P. S.
6. **Surface the silent drops.** Read HDBSCAN `neighborOverflow` and DBSCAN `gridOverflow` into status (F14). Validate symmetry before SKATER and MST (F12). Warn when `islandCount > 0` under the default `'exclude'` policy (F7). S.
7. **Normalise region label output** to dense `[0, regionCount)` across AZP, Ward and SKATER, and standardise status-word layouts and clustering prop names (§5). S–M.
8. **Lift the per-row neighbour caps in the local permutation test.** Replace the linear swap lookup with a small hash or bitset, or a sort-based sample, so distance-band W with hundreds of neighbours can be tested without O(k²) draws. M.
9. **Add missing tests:**
   - global Moran and the permutation tests at 1e6 value offsets;
   - negative-value G permutation;
   - NaN in MST input;
   - LOF with tied k-distances;
   - k-means at UTM coordinates with `centerCoordinates` on and off;
   - HDBSCAN with coincident points;
   - a max-p enclave-cap fragmentation case;
   - SKATER with duplicate or asymmetric slots.

   S each.
10. **Cheap new methods.** Geographical detector q (on `GPURegionPartitionEvaluation`), REDCAP single-linkage (on SKATER and the MST), a correlogram-over-orders recipe, the weights component count, and global CLQ. Each is S, and together they close the most-requested ESDA gaps.

Also fix the coverage matrix:
- ST09.fuzzy should record the 512-feature dense limit.
- The doc mismatches in F22 should be corrected.
