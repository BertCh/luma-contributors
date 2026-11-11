# Math correctness review: spatial statistics

Scope id: `spatial-statistics`. Date: 2026-11-10. Reviewer: one of 18.

Scope: `gpu-spatial-analysis/{spatial-weights, spatial-autocorrelation, global-spatial-statistics, permutation-inference, multiple-testing, rate-smoothing, emerging-hot-spots, scan-statistics, geographical-detector, spatial-entropy, segregation, categorical-association, outlier-detection}`. Paths below are relative to `modules/analysis/src/gpu-spatial-analysis` unless they start with `docs/` or `test/`.

Method. I read the kernels and compared each formula with the reference method. Three scratch scripts replicate kernel arithmetic with `Math.fround`. They are in the session scratchpad under `spatial-statistics/` (`g-overflow.mjs`, `local-centering.mjs`, `gamma-adapter.mjs`, `moment-overflow.mjs`). I ran no spec. I fetched two public pages to settle reference questions: the spdep source `R/licd_boots.R` (r-spatial/spdep, main branch) and the ArcGIS Pro page on emerging hot spot analysis.

`neighbor-search` is not in this scope. k-nearest-neighbour ties, distance-band inclusivity and the kernel weights written by `GPUNeighborSearch` are not reviewed here. The kernel transform that re-weights an existing CSR is.

No file looked half-edited.

## 1. Coverage

Read in full (every kernel body and the class documentation):

- `global-spatial-statistics/gpu-global-spatial-statistics.ts`
- `spatial-autocorrelation/spatial-autocorrelation-kernels.ts`
- `spatial-weights/spatial-weights.ts`
- `permutation-inference/permutation-alternative.ts`
- `multiple-testing/permutation-p-value-floor.ts`
- `scan-statistics/expectation-based-scan-poisson-wgsl.ts`
- `geographical-detector/gpu-geographical-detector.ts`, `geographical-detector-p-value.ts`
- `spatial-entropy/gpu-spatial-entropy.ts`

Read all math-bearing lines, skipped only imports and the first prop types:

- `spatial-autocorrelation/gpu-local-moran.ts` (60 to end), `gpu-hot-spot-analysis.ts` (100 to end), `gpu-additional-spatial-statistics.ts` (40 to end), `gpu-local-variability.ts` (30 to end), `gpu-multivariate-local-geary-permutation-test.ts` (40 to end), `gpu-multivariate-local-geary-tiled.ts` (60 to end), `gpu-losh-residual-permutation-test.ts` (20 to end), `gpu-rate-standardization.ts` (documentation and 350 to end)
- `permutation-inference/gpu-local-permutation-test.ts` (70 to end), `permutation-inference-kernels.ts` (150 to end)
- `multiple-testing/gpu-multiple-testing-adjustment.ts` (28 to end)
- `rate-smoothing/gpu-empirical-bayes-rates.ts`, `gpu-spatial-empirical-bayes-rates.ts` (30 to end)
- `scan-statistics/scan-statistic-nearest-zones.ts`
- `categorical-association/gpu-colocation-quotient.ts`, `gpu-local-category-dependence.ts` (28 to end)
- `outlier-detection/gpu-local-outlier-factor.ts` (30 to end)
- `spatial-weights/gpu-lattice-weights.ts` (36 to end), `spatial-weights-space-time-stages.ts`

Read in part:

- `permutation-inference/gpu-global-permutation-test.ts`: 300 to 370 and 440 to 1090 (statistic terms, scales, finalize). Not read: the materialized and sequential batch node builders (1090 to end).
- `emerging-hot-spots/gpu-emerging-hot-spots.ts`: 60 to 390 and 520 to end. `emerging-hot-spot-parameters.ts`: the constants and the critical-z table only.
- `scan-statistics/gpu-spatial-scan-statistic.ts`: 60 to 140, 225 to 315, 660 to end. `gpu-expectation-based-space-time-scan.ts`: 150 to 240, 420 to end.
- `segregation/gpu-segregation.ts`: 30 to 210, 330 to end.
- `spatial-weights/gpu-contiguity-weights.ts`: 60 to 608 (keys, rook edges, pair emission). Not read: the final dedup and CSR compaction.
- `spatial-weights/gpu-spatial-weights-transform.ts`: documentation, kernel table, a line-level skim of the bodies.
- `spatial-weights/gpu-spatial-weights-summary.ts`: 84 to 110, 150 to 320, the finalize lines. Not read: the chunked path.
- `spatial-weights/gpu-spatial-lag.ts`: 60 to end.
- `spatial-weights/gpu-spatial-weights-algebra.ts`: documentation only (52 to 175, 344 to 356).
- `spatial-weights/gpu-island-attachment.ts`: documentation and the nearest-row kernel.
- `spatial-weights/gpu-fuzzy-contiguity.ts`: documentation and helper signatures.
- `spatial-entropy/gpu-category-pair-counts.ts`: 25 to 75, 120 to end.
- `multiple-testing/gpu-raster-false-discovery-rate-mask.ts`: 30 to 110 and a skim.
- `spatial-autocorrelation/spatial-autocorrelation-parameters.ts`: 55 to 75.

Not opened: `spatial-weights/{spatial-weights-algebra-wgsl, spatial-weights-row-local-stages, spatial-weights-transpose-pipeline, gpu-spatial-weights-transpose, gpu-spatial-weights-validation, gpu-chunked-spatial-weights, gpu-chunked-spatial-weights-algebra, gpu-chunked-spatial-weights-transform, gpu-chunked-spatial-weights-transpose, gpu-chunked-spatial-weights-validation, key-pair-grouping, slot-grouping, float32-values}.ts`, `permutation-inference/{permutation-parameters, gpu-local-permutation-replay}.ts`, `scan-statistics/scan-statistic-parameters.ts`, `segregation/segregation-layout.ts`, `global-spatial-statistics/global-spatial-statistics-layout.ts`, `outlier-detection/outlier-detection-parameters.ts`, every `index.ts`.

So the set algebra of weights (union, intersection, difference, higher order, block, subgraph), the transposes, the validation kernel and all chunked weights variants are unverified.

## 2. Findings

| ID | Severity | Status | File:line | Kernel computes | Reference says | Input that goes wrong | Suggested fix |
|---|---|---|---|---|---|---|---|
| spatial-statistics-01 | wrong-edge-case | CONFIRMED (script) | `global-spatial-statistics/gpu-global-spatial-statistics.ts:1517-1522` | General G variance as `quadraticForm / (denominator * denominator)` in f32, `denominator = n(n-1) mean^2 - SS` | `esda.G.VG`, finite for any scale | 40,000 values near 300,000 (sum 1.2e10): `denominator^2` is infinity, variance field reads 0, z and p are NaN. float64 gives variance 1.32e-16, z 1.131 | Divide twice (`q / d / d`), or form `sqrt(q) / d`, or scale the values by a power of two first |
| spatial-statistics-02 | precision | CONFIRMED (hand arithmetic, script) | same file `:1476`, `:1399`, `:1375`, `:1387` | Kurtosis as `n m4 / (SS * SS)`; randomization variances divide by `(n-1)(n-2)(n-3) S0^2` | Scale-invariant in exact arithmetic | 10,000 rows with standard deviation 5e7: `SS^2` overflows, kurtosis is NaN, Moran and Geary randomization variance, z and p are NaN (normality fields stay valid). 1.2e7 rows with 60 binary neighbours: the `S0^2` product overflows, variance is `-E^2`, z NaN | Standardize the centred values by a power-of-two scale before the moment columns; compute `m4 / SS / SS`; group the large products |
| spatial-statistics-03 | doc-mismatch | CONFIRMED (spdep source, hand value) | `categorical-association/gpu-local-category-dependence.ts:83-89`, `:212-227`, `:236-244`; `docs/api/gpu-spatial-analysis/colocation-category-dependence-adaptive-density.md:20` | Like and neighbour counts without the focal row; local proportion; upper tail `P(X >= like)` on `Binomial(k, p)` | spdep `licd_multi`: counts include the focal row (`k + 1`), `prop_i` is the global share, `pbinom_like_BW` is the lower tail `P(X <= like)` | 4 neighbours, 3 like, `p = 0.5`: kernel 0.3125; spdep `pbinom_like_BW` 0.96875, `pbinom_unlike_BW_alt` 0.1875, counts 4 of 5 | Either match spdep (add the focal row, report the tails under spdep names) or stop naming spdep columns |
| spatial-statistics-04 | doc-mismatch | CONFIRMED (script) | `spatial-autocorrelation/gpu-additional-spatial-statistics.ts:335-348`, `:297-303`, `:458-460`; `docs/api/gpu-spatial-analysis-cross-reference.md:86` | Gamma `cross-product` centres on edge-weighted means and divides by the edge weight; the adapter says the `'moran'` permutation has "the same order-preserving pair sum" | `esda.Gamma` is the uncentred sum `sum w_ij x_i x_j`; the adapter claim needs equal orderings under permutation | 5-node graph with unequal degrees, values 1, 2, 4, 7, 11: two permutations are ordered one way by the kernel Gamma and the other way by the Moran pair sum | Say that the adapter tests Moran's I, not this Gamma; drop the `esda.Gamma` and `esda.Spatial_Pearson` mapping for these two classes |
| spatial-statistics-05 | inconsistent | CONFIRMED (code trace, hand value) | `spatial-entropy/gpu-category-pair-counts.ts:176-185`; `segregation/gpu-segregation.ts:601-607`; `categorical-association/gpu-local-category-dependence.ts:236-244`; `rate-smoothing/gpu-spatial-empirical-bayes-rates.ts:244-250` | Four neighbour definitions that differ from the shared active-slot rule | `spatial-weights/spatial-weights.ts:156-179`: positive weight, another row | Weights with self slots (`GPUSpatialWeightsAlgebra` `'selfWeight'`), 2 rows, categories 0 and 1: pair counts are 1, 1, 1, 1 and mutual information is 0; without the self links it is ln 2 | Use `getGPUSpatialWeightsNeighborLoopWGSL` in all four, or reject `hasSelfWeights` at construction |
| spatial-statistics-06 | doc-mismatch | CONFIRMED (code trace) | `scan-statistics/gpu-spatial-scan-statistic.ts:256-258` against `:1458-1465`, `:1621-1630` | Secondary clusters are chosen among each centre's single best window | The class doc and `docs/api/gpu-spatial-analysis/operators-dynamics.md:556` say "the best window sharing no zone with any earlier cluster" | Centre D's best window overlaps the first cluster (LLR 8); its smaller window does not (LLR 7); centre E's best is 5. The doc promises 7, the kernel returns 5 | Reword: candidates are the best window of each centre (this is what SaTScan does) |
| spatial-statistics-07 | inconsistent | CONFIRMED (code), impact measured small | `spatial-autocorrelation/spatial-autocorrelation-kernels.ts:533-552`, `:575-583`; `emerging-hot-spots/gpu-emerging-hot-spots.ts:861-866`, `:897-899` | Three ways to centre values: residual kept apart (global, permutation), corrected mean rounded to one f32 (hot spot, local Moran, local Geary), no second pass (emerging hot spots) | One second-pass rule; the global kernel's own comment says the rounded form "defeats the second pass" | Values near 1e6, standard deviation 1: z differs from float64 by up to 0.003 in my sample; the bound is half an ulp of the mean times `sqrt(k)` over the standard deviation | Store the first-pass deviations and subtract the residual mean, as the global kernel does |
| spatial-statistics-08 | inconsistent | CONFIRMED (code) | `spatial-autocorrelation/gpu-hot-spot-analysis.ts:410`, `:493`; `gpu-local-moran.ts:450`, `:581`; against `global-spatial-statistics/...:1447-1450` and `emerging-hot-spots/...:748` | Analytic p-values are floored at 1.17549435e-38 in two contributors and not in two others | One rule | `z = 15` in `GPUHotSpotAnalysis`: `erfc(10.6)` is about 1e-49, below the smallest f32, so p is exactly 0; the same z in `GPUEmergingHotSpots` gives 1.18e-38 | Put the floor in `getTwoSidedPValue` |
| spatial-statistics-09 | doc-mismatch | still present (prior F22) | `spatial-autocorrelation/gpu-hot-spot-analysis.ts:201-202` | Chunked weights are accepted (`:107`) and dispatched per chunk (`:307`) | The class doc lists "chunked inputs" as a non-goal | Not a numeric defect | Delete the two words |

### Notes on the findings

**01 and 02.** The global moment formulas mix sums whose magnitudes grow with the square and fourth power of the data. float32 tops out at 3.4e38. `denominator` for General G is about `(sum x)^2`, so squaring it overflows once `sum x` passes about 4.3e9. House prices, populations and incomes reach that with a few thousand to a few tens of thousands of rows. The G statistic itself and its expectation stay correct; only the variance, z and p are lost. The variance field reads exactly 0, which the class doc reserves for "positive constant data". The global permutation test is not affected (its G scale is not squared, `permutation-inference/gpu-global-permutation-test.ts:860`). No spec covers large-magnitude values for the global statistics. The Moran and Geary cases in 02 need a standard deviation above about `4.3e9 / sqrt(n)`, which is rarer.

**03.** I confirmed the spdep side from the package source, not from a pinned release, and R is not installed here. The quoted lines are `x_nb_i <- c(xi, x[nb_i])`, `c1_comp_obs_i <- sum(x_nb_i_xi)`, `c3_crdip1 <- crdi + 1`, `c2_prop_level_i <- tifx[xi]`, `pbinom(c1_comp_obs_i, c3_crdip1, c2_prop_level_i, lower.tail=TRUE)` for `pbinom_like_BW`, and `pbinom(c1_comp_obs_i-1, ..., lower.tail=FALSE)` for `pbinom_unlike_BW_alt`. The kernel's own definition is self-consistent and its binomial recurrence is right. The user doc already says the spdep formulas "were not available beyond the output names". They are available now, and they differ in all four named columns.

**04.** `GPUSpatialPearson` is an edge-weighted Pearson coefficient of `x_i` against `y_j`. esda's `Spatial_Pearson` is Lee's L, which this library exposes as `'leesL'`. The user doc says so at `docs/api/gpu-spatial-analysis/operators-statistics.md:621`, but the cross-reference table still maps `esda.Spatial_Pearson` and `esda.Gamma` to these classes. For row-standardized weights the Gamma and Moran orderings do agree (I derived this by hand); they differ for binary weights with unequal degrees. The `'squared-difference'` adapter to `'geary'` is exact.

**05.** Per contributor: `GPUCategoryPairCounts` (and so `GPUSpatialEntropy`) counts every listed slot, self and zero-weight included. `GPULocalCategoryDependence` skips self, counts zero-weight slots. `GPUColocationQuotient` uses the shared rule. `GPUSegregation` adds every slot's weight with no self check and then adds `selfWeight`, so a self slot is counted twice. `GPUSpatialEmpiricalBayesRates` skips self and counts zero-weight slots (esda's binary transform does the same, and the class says so). Each class documents its own rule, so this is a consistency finding. `GPUSpatialWeightsAlgebra` lists which consumers accept self weights (`spatial-weights/gpu-spatial-weights-algebra.ts:348-353`); these four are not in that list.

**07.** The difference is real but small. The `GPULocalMoran` doc already warns that the sign of z can flip for values within rounding of the mean; in my sample at offset 1e6 that was 70 of 2,500 rows. The existing large-offset spec (`test/gpu-spatial-analysis/spatial-autocorrelation/gpu-hot-spot-analysis.spec.ts:598-620`) checks the moments only, not the z-scores.

### Prior findings rechecked

- F1 (local and global G permutation tails reverse on negative data): fixed. Negative values now leave every row untested and set `overflow` and `invalidCount` (`permutation-inference/gpu-local-permutation-test.ts:1163-1233`, `gpu-global-permutation-test.ts:842-914`).
- F2 (permutation p-value floor against FDR): addressed by `getPermutationRejectionFloor`.
- F17 (LOF k-distance with ties): now documented on the `neighbors` prop; behaviour unchanged and equal to scikit-learn.
- F22 (hot spot doc): still present, see 09.

## 3. Verified correct

Each line names what I compared the kernel against.

- `GPUGlobalSpatialStatistics` Moran's I, `E`, `VI_norm`, `VI_rand`: esda `Moran` moment formulas, term by term.
- Same, Geary's C, `VC_norm`, `VC_rand`: esda `Geary`.
- Same, General G statistic and expectation: esda `G`. The centred variance identity `mean^2 Var(d'z) + 2 mean Cov(d'z, z'Wz) + Var(z'Wz)`: I re-derived the covariance term `-m3 Vd / ((n-1)(n-2))` by hand from permutation moments. (Overflow aside, see 01.)
- Same, bivariate Moran: esda `Moran_BV` (standardization with `ddof=1`, denominator `n - 1`). Randomization variance derived by hand.
- Same, join counts BB and BW, expectations and variances: Cliff-Ord nonfree sampling, through the indicator moments `npq`, `npq(q-p)`, `npq(p^3+q^3)`.
- Same, `S1` and binary `S1` with a missing reverse slot, `S2`: libpysal definitions.
- `GPULocalMoran`: esda `Moran_Local` scaling `(n-1) c_i L_i / M2`; conditional-randomization variance `sigma^2 (N S1 - W^2) / (N - 1)` derived by hand.
- `GPUHotSpotAnalysis` Gi*: Ord-Getis and ArcGIS formula. Gi (`selfWeight: 0`): esda `G_Local(star=False)` `Zs`. Critical z values are exact (1.6448536, 1.9599640, 2.5758293).
- `GPULocalGeary`: esda `Geary_Local` (population-standardized).
- `GPULocalSpatialHeteroscedasticity`: esda `LOSH` (residual about the neighbour mean, lag of residuals over their mean).
- `GPUMultivariateLocalGeary` and its permutation test (exact and tiled): esda `Geary_Local_MV`, mean over variables. The tiled virtual candidate list equals the exact path's array. Chan merge of the moments is right.
- `GPULocalPermutationTest`: partial Fisher-Yates with a sparse swap list is a uniform ordered sample without replacement; the focal row is skipped correctly. Scalings of local Moran, Geary, G, G*, bivariate Moran, Lee's L against esda. Quadrant codes against esda `Moran_Local_BV.q`. Geary classes against esda `Geary_Local` labels. Join counts test only rows with a nonzero count, as esda.
- Pseudo p-values: `(M + 1) / (P + 1)`, doubled smaller tail for two-sided, Besag-Clifford `h / L` when stopped.
- `GPUGlobalPermutationTest`: statistic terms and scales for Moran, Geary, G, bivariate Moran, Lee's L (`z'W'Wz / 1'W'W1`) against esda. `z_sim` uses the population deviation and `p_z_sim` is one-tailed, as esda.
- `GPUMultipleTestingAdjustment`: Bonferroni, Holm, Benjamini-Hochberg, Benjamini-Yekutieli against statsmodels `multipletests` definitions. Tie handling, the harmonic number series and the 30-bit sort key bound.
- `GPUEmpiricalBayesRates` and `GPUAssuncaoReisStandardization`: esda `Empirical_Bayes` and `assuncao_rate`.
- `GPUSpatialEmpiricalBayesRates`: esda `Spatial_Rate` and `Spatial_Empirical_Bayes`, including the clamp of the local variance.
- `GPUEmergingHotSpots`: Gi* with a temporal window (lattice and weights forms agree for binary weights). Mann-Kendall `S`, tie-corrected variance, continuity correction; both the serial and the cooperative tie counters. Hamed-Rao and Yue-Wang factors against the pymannkendall formulas. The 16 category rules against the ArcGIS Pro definitions (page fetched). ESRI does apply FDR here, as the prop doc says.
- `GPUSpatialScanStatistic`: Kulldorff Poisson LLR, Bernoulli LLR as four `o ln(o/e)` terms, space-time permutation expectation. The `e((1+d)ln(1+d) - d)` form and its series. Multinomial, hypergeometric (accept-reject) and time-permutation draws. p-value `(1 + count) / (P + 1)`. Radix select and LSD sort of the nearest-zone table.
- `GPUExpectationBasedSpaceTimeScan`: Neill's `C ln(C/B) + B - C`, the prefix property of the `C_i / B_i` ordering, the Knuth merge-exchange network bounds, the PTRS Poisson sampler constants (Hormann 1993, as in NumPy), Stirling `ln(n!)`.
- `GPUGeographicalDetector`: `q`, `F`, `lambda` with population variance against the Wang formula as in R `GD`. `getNonCentralFUpperTail`: Poisson mixture of incomplete betas, Lentz continued fraction against Numerical Recipes.
- `GPUSpatialEntropy`: Shannon, co-occurrence (Leibovici), mutual information (Altieri) from the count table.
- `GPUColocationQuotient`: Leslie-Kronenfeld global CLQ with the `N_b - 1` same-category correction; local form.
- `GPUSegregation`: entropy H, multigroup D, `D_g`, isolation, interaction, Atkinson against the PySAL `segregation` formulas.
- `GPULocalOutlierFactor`: Breunig and scikit-learn (`lrd = 1 / (mean reach + 1e-10)`).
- `GPULatticeWeights`: rook as Manhattan and queen as Chebyshev distance, ascending IDs.
- `GPUContiguityWeights`: queen by shared snapped vertex, rook by shared edge in either direction, closing edge included, degenerate edges dropped.
- `GPUSpatialWeightsTransform`: row, binary, double and variance-stabilizing transforms against libpysal `R`, `B`, `D`, `V`; kernel profiles and constants against libpysal `Kernel`; adaptive bandwidth with `eps = 1.0000001`.
- `GPUSpatialWeightsSummary` (whole path): `S0`, `S1`, `S2`.
- `GPUSpatialLag`: `sum w_ij y_j`, optional normalization.
- Space-time weights (`T (x) W`): ascending neighbour IDs in both layouts, self insertion in order.

## 4. Standards consistency

- **Neighbour definition.** See finding 05. The shared loop is used by the global, local and permutation statistics, the summary and LOSH. The categorical, segregation and spatial rate contributors each carry their own loop. `GPUEmergingHotSpots` weights mode and the edge-association kernels also use private loops; they skip self and give zero weights no effect, so they agree numerically.
- **Island definition.** `GPUIslandAttachment` treats only empty rows as islands; every statistic treats a row with no active slot as an island. Documented.
- **Mean centring.** Three schemes, finding 07.
- **p-value floor.** Two conventions, finding 08.
- **Permutation count.** The permutation tests never clamp an invalid count (NaN and `overflow`). `GPUSpatialScanStatistic` clamps to `[1, maximumPermutations]` (`scan-statistics/gpu-spatial-scan-statistic.ts:688-690`).
- **One transform, two implementations.** `GPUAssuncaoReisStandardization` and `GPUEmpiricalBayesRates.standardizedRates` both compute esda `assuncao_rate`. Same formula. They differ at the edges: the first requires `events >= 0`, corrects the pooled rate with a second pass and writes NaN when the variance is not positive; the second accepts any finite event count, has no second pass and divides by the square root as is.
- **Two tables of the same critical values.** `GPU_HOT_SPOT_CRITICAL_Z_SCORES` and `GPU_EMERGING_HOT_SPOTS_CRITICAL_Z_SCORES` hold the same three numbers.
- **No-data.** The statistics exclude non-finite rows. `GPUGeographicalDetector` delegates to `GPURegionPartitionEvaluation`, where I found no finiteness test; a NaN value appears to make `q` NaN. SUSPECTED: that file is outside this scope and I did not read its kernels.
- **Private copies of shared numerics.**
  - Binary searches: `findWeight` in the global statistics (three variants) and in the weights summary, the cumulative-table searches in the scan (four), the ring, polygon and feature searches in contiguity. All are plain lower or upper bounds and look right. `analysis-core` has a shared one.
  - Log-gamma: a Stirling series in the scan sampler (WGSL) and a Lanczos form in the geodetector p-value (TypeScript). `analysis-core` has the incomplete gamma; I did not check whether it exposes a log-gamma or an incomplete beta.
  - Orientation and segment distance: `GPUFuzzyContiguity` uses plain f32 `cross` and segment distance, not the exact predicates. The class says near-degenerate input is not repaired.
  - Binomial tail in `GPULocalCategoryDependence`, harmonic number in the multiple-testing node: local, small, correct.
  - Random numbers: every draw I read uses Philox with a registered purpose. No private hash.
  - Great-circle distance: the scan uses the shared geodesic WGSL through `spatial-interpolation/separation-wgsl.ts` and the context's sphere radius.
- **Literals.** `1.17549435e-38`, `0.70710678118654752` and `0.7071067811865476` are written out at each use.

## 5. luma.gl official implementations

Published `@luma.gl/graph` 10.0.0-alpha.10 has no spatial weights, autocorrelation, permutation inference, multiple testing, rate smoothing, scan statistic, segregation or outlier contributor. I searched the 14 entry-point declaration files for those names. Nothing in this scope re-implements a published statistic.

Published primitives this scope does use: `GPUScan`, `GPUSort`, `GPUCompaction`, `GPUReduction` (`gpu-core`). The arguments I saw do not change meaning (exclusive scans for offsets, 30-bit keys for p-values in `[0, 1]`).

Two places to look at, neither a defect:

- `GPUGlobalPermutationTest` bins its reference distribution with its own kernel (`permutation-inference/gpu-global-permutation-test.ts:1045-1060`: `floor((v - min) / range * bins)`, clamped). `gpu-core` publishes `GPUHistogram`. I did not compare the two binning rules.
- `GPUSpatialWeightsAlgebra` `'higherOrder'` is breadth-first levels from every row. `gpu-graph-algorithms` publishes `GPUGraphAlgorithmsBreadthFirstSearch` for a single source; it is not a replacement. I did not read the local kernel.

The many private fixed-order tree sums are already tracked by the library review and are not repeated here.

## 6. Test oracles

Independent oracle (a fixture generated by a reference library):

- Global Moran, Geary, G, bivariate Moran, join counts: esda 2.10.0 (`statistics-global-autocorrelation.json`).
- Local Moran, hot spots, local permutation statistics, alternatives: esda 2.10.0 (`statistics-local-autocorrelation.json`, `statistics-local-join-counts-and-bivariate.json`, `statistics-permutation-alternative.json`).
- Multiple testing: statsmodels 0.15.0.
- Rate smoothing and standardization: esda 2.10.0.
- Weights (contiguity, lattice, transforms, kernels, summary): libpysal 4.15.0.
- Segregation: PySAL `segregation` 2.6.0.
- Emerging hot spots: esda and scipy fixtures for Gi* and the trend; pymannkendall for the plain and modified Mann-Kendall.
- Spatial scan: R `SpatialEpi` 1.2.8 (`parity-wp1-scan.json`).
- Geographical detector: `geodetector` 0.2.0, and scipy `ncf.sf` for the tail.
- Local outlier factor: a scikit-learn fixture (per the prior review; I saw the fixture loader, not the generator).

Own formula only (a CPU or binary64 copy of the same definition):

- `GPUExpectationBasedSpaceTimeScan`: `test/gpu-spatial-analysis/scan-statistics/stx07-expectation-based-scan-reference.ts` says the external fixture is pending.
- `GPUColocationQuotient`, `GPULocalCategoryDependence`: a binary64 evaluation of the class's own definition. For the second this is why finding 03 was not caught.
- `GPUSpatialEntropy`: a numpy evaluation of the same formulas; no package.
- `GPUSpatialPearson`, `GPUGammaStatistic`: own definitions; no package computes them.

Not determined (I did not open the specs): `GPULocalSpatialHeteroscedasticity`, `GPUMultivariateLocalGeary` and its permutation test, `GPUIslandAttachment`, `GPUFuzzyContiguity`, the weights algebra.

Gaps worth a test:

- General G, Moran and Geary analytic inference on large-magnitude values (findings 01, 02).
- Gi* and local Moran z-scores at a large offset (finding 07); the existing spec checks moments only.
- Categorical contributors on weights with self slots (finding 05).

## Resolution (2026-11-10)

Paths are relative to `modules/analysis/src/gpu-spatial-analysis` (source) and `modules/analysis/test/gpu-spatial-analysis` (specs) unless they start with `docs/`. Every spec named here ran on the GPU (headless project) and passed after the fix. The spec for the geographical detector was written first and failed on the old kernel; the others were written with or after the fix, and the old failure is the arithmetic given in the finding.

| ID | Outcome | What changed (`file:line`) | Spec that pins it | Behaviour change a caller sees |
|---|---|---|---|---|
| spatial-statistics-01 | fixed | `global-spatial-statistics/gpu-global-spatial-statistics.ts:1157` (new `value-scales` kernel: a power of two near the mean magnitude of `values` and of `secondValues`, from fixed-order sums of `abs(x)`; stored in the counter view), the `deviate`, `pairs` and `finalize` kernels work on the scaled values, `:1625` (`q / d / d`). Only `summary.mean` and `summary.variance` are scaled back | `global-spatial-statistics/gpu-global-spatial-statistics.spec.ts:1011` (40,000 values near 300,000 on a 200 by 200 rook grid against float64 `esda.G`: `G`, `E`, `VG`, z, p) | General G variance, z and p were 0, NaN, NaN on such data; now finite and equal to float64 (z within 0.02, one f32 ulp of G is 5e-4 z there). Results that were finite are unchanged: the scale is an exact power of two |
| spatial-statistics-02 | fixed | same file: `:1463` and `:1477` (Moran and Geary moments with `r1 = S1 / S0^2`, `r2 = S2 / S0^2`, so `n^3 S0^2` is never formed), `:1497` and `:1579` (kurtosis as `n (m4 / SS) / SS`), plus the value scale of 01 | same spec `:1044` (10,000 rows, standard deviation above 4.5e7, against float64 `esda.Moran` and `esda.Geary`), `:1076` (1,024 rows with weights 1e13, where `n^3 S0^2 = 1.7e42`; the 1.2e7-row case of the report is not affordable in a test, and a float32 replication in the scratchpad gives 2.7777638e-9 against 2.7777640e-9 in float64, where the old form gave `-E^2`) | Moran and Geary randomization variance, z and p were NaN (or `-E^2`) on such data; now finite. Rounding of the variances changes in the last bits elsewhere |
| spatial-statistics-03 | fixed (words; formula unchanged) | `categorical-association/gpu-local-category-dependence.ts:86` (class documentation states the definition and no longer claims spdep columns), `docs/api/gpu-spatial-analysis/colocation-category-dependence-adaptive-density.md:20-24`, `ontology/spatiotemporal-review-catalog.ts:720` (summary no longer names `licd_multi`) | `stx08-spatial-methods/stx08-local-category-dependence.spec.ts:117` (hand value 0.3125 for 4 neighbours, 3 like, share 0.5) | None in the numbers. The spdep parity claim is gone |
| spatial-statistics-04 | fixed (words) | `spatial-autocorrelation/gpu-additional-spatial-statistics.ts:305` and `:361` (both adapters now say which test they run and when it has the same ordering), class documentation of `GPUGammaStatistic` and `GPUSpatialPearson`, `docs/api/gpu-spatial-analysis-cross-reference.md:86`, `docs/api/gpu-spatial-analysis/operators-statistics.md` (adapter paragraph) | `spatial-autocorrelation/additional-spatial-statistics.spec.ts:330` (the two permutations of the report: Gamma -10 and -11.0625, Moran's I -0.757576 and -0.606061, by hand; opposite orders on the GPU) | None in the numbers. The adapters return the same descriptors. No source file uses an adapter to compute a p-value, so there was no code path to change |
| spatial-statistics-05 | fixed | `spatial-entropy/gpu-category-pair-counts.ts:187`, `categorical-association/gpu-local-category-dependence.ts:248`, `segregation/gpu-segregation.ts:613`, `rate-smoothing/gpu-spatial-empirical-bayes-rates.ts:222`: all four now use `getGPUSpatialWeightsNeighborLoopWGSL` (converged on the shared active-slot rule, the convention most siblings and the library's own weights contract use) | `stx08-spatial-methods/stx08-spatial-entropy.spec.ts:63`, `stx08-local-category-dependence.spec.ts:117`, `segregation/gpu-segregation.spec.ts:323`, `rate-smoothing/gpu-spatial-empirical-bayes-rates.spec.ts:194` (all hand values) | See the list below |
| spatial-statistics-06 | fixed (words) | `scan-statistics/gpu-spatial-scan-statistic.ts:256-262`, `docs/api/gpu-spatial-analysis/operators-dynamics.md:568-571` | none (wording) | None |
| spatial-statistics-07 | fixed | `spatial-autocorrelation/spatial-autocorrelation-kernels.ts:100` and `:613` (the shared front end of Gi*, local Moran and local Geary stores the first-pass deviations and subtracts their mean, as the global and permutation kernels do; fixed moments go through the same stored deviations), `emerging-hot-spots/gpu-emerging-hot-spots.ts:926` (two new nodes: block deviations and mean correction) and `:760` (the residual is subtracted once per Gi* sum) | `spatial-autocorrelation/gpu-hot-spot-analysis.spec.ts:628` (Gi* and local Moran z at whole numbers near 1e7 against binary64), `emerging-hot-spots/gpu-emerging-hot-spots.spec.ts:1030` | z-scores of data with a large offset move toward the float64 value. A row whose value equals the reported f32 mean now has a finite local Moran z (it was NaN). `GPUEmergingHotSpots` has two more nodes |
| spatial-statistics-08 | fixed | `spatial-autocorrelation/spatial-autocorrelation-kernels.ts:183` (`ANALYTIC_P_VALUE_FLOOR` inside `getTwoSidedPValue`), used by `global-spatial-statistics/...:1546`, `emerging-hot-spots/...:800`, `permutation-inference/gpu-global-permutation-test.ts:1080` | `spatial-autocorrelation/gpu-hot-spot-analysis.spec.ts:673` (Gi* and local Moran, z above 15: p is exactly 1.17549435e-38) | See the list below |
| spatial-statistics-09 | fixed (words) | `spatial-autocorrelation/gpu-hot-spot-analysis.ts:205-207` | none (wording) | None |

### Items from other reports and from the lead, in this stream's files

| Item | Outcome | What changed | Spec |
|---|---|---|---|
| core-numerics-05, the part in `scan-statistics/expectation-based-scan-poisson-wgsl.ts` | fixed | `:19` uses `philoxOpenUnitFloat`; the CPU reference `scan-statistics/stx07-expectation-based-scan-reference.ts` uses `getPhiloxOpenUnitFloat` | `scan-statistics/stx07-expectation-based-scan.spec.ts:168`, and the existing replicate-reproduction and Poisson-law cases of that spec |
| polygon-topology-03 and geometry-measures-04, the part in `spatial-weights/gpu-contiguity-weights.ts` | fixed | `:383` (the snap grid is anchored at the first vertex of the buffer, read in the same kernel; the kernel already binds eight storage buffers, so a reduced minimum could not be bound; since the review follow-up the key is the imported `getCoverageSnapKeyWGSL`), `:64` (`GPU_CONTIGUITY_WEIGHTS_MAXIMUM_SNAP_CELLS = 2^24`, exported), new optional prop `coordinateExtent` validated against it with `validateMaximumCount` | `spatial-weights/gpu-spatial-weights.spec.ts:353` (three 100 m squares at x = 5e6, y = 4.5e6, tolerance 1 mm: A-B and B-C by inspection), `spatial-weights/gpu-spatial-weights.node.spec.ts` (the limit) |
| Standards note: one NaN value makes the geographical detector NaN | fixed (reproduced first: q was NaN) | `geographical-detector/gpu-geographical-detector.ts:179` (a row with a non-finite value is given no stratum before the partition evaluation) | `stx08-spatial-methods/stx08-geographical-detector.spec.ts:69` |
| Hand-computed cases for contributors pinned only by their own formula | added | none | `scan-statistics/stx07-expectation-based-scan.spec.ts:138` (four zones, best subset 7.538682), `stx08-spatial-methods/stx08-colocation-quotient.spec.ts:108` (five points, weighted), `stx08-local-category-dependence.spec.ts:117`, `stx08-spatial-entropy.spec.ts:63` |
| Compensated float sums (lead's note during the pass) | nothing to do | `grep -rliE "neumaier\|kahan\|compensat"` finds no file in this stream's directories | none |

### Behaviour changes callers will see

- `GPUCategoryPairCounts`, `GPUSpatialEntropy`: a self slot and a zero-weight slot are no longer links. Two rows of categories A and B that each list themselves and the other: table was 1, 1, 1, 1 and mutual information 0; now 0, 1, 1, 0 and ln 2.
- `GPULocalCategoryDependence`: a zero-weight slot is no longer a neighbour (self slots were already skipped).
- `GPUSegregation`: a self slot in the weights is no longer added on top of `selfWeight`. Units (10, 0) and (0, 30) that list themselves and each other: environments were (20, 30) and (10, 60); now both (10, 30). Negative and NaN weights are skipped instead of entering the sum.
- `GPUSpatialEmpiricalBayesRates`: a slot stored with weight 0 is no longer pooled (esda's binary transform would count it; the class documentation says so). With every output and no mask the contributor now emits two kernels instead of one, because the kernel binds `weights`.
- Analytic p-values of `GPUHotSpotAnalysis`, `GPULocalMoran` and the Mann-Kendall trend of `GPUEmergingHotSpots`: exactly 0 becomes 1.17549435e-38. `GPUGlobalPermutationTest` `pZSimulated` has the same floor.
- `GPUGlobalSpatialStatistics`: NaN or zero inference on large-magnitude data becomes the correct finite value (01, 02). The row maximum is unchanged (the magnitudes are in a separate two-column view: 8 more bytes per row).
- `GPULocalMoran`, `GPUHotSpotAnalysis`, `GPULocalGeary`, `GPUEmergingHotSpots`: centered values no longer sit on the f32 grid of the raw values. Visible only for data with a large offset.
- `GPUContiguityWeights` with `snapTolerance > 0`: the grid is anchored at the first vertex, not at the coordinate origin. Vertices that coincide exactly still coincide; which near-coincident vertices share a cell can differ. At projected coordinates the result changes from wrong (every feature a queen neighbour of every other, no rook neighbour) to right.
- `GPUExpectationBasedSpaceTimeScan`: the Monte Carlo replicates use a different uniform (23 bits), so replicate maxima and p-values of a given seed change within Monte Carlo noise.
- `GPUGeographicalDetector`: rows with a NaN or infinite value are left out; `observationCount` no longer counts them.

### Existing oracles that encoded the old behaviour, changed with the fix

- `spatial-autocorrelation/gpu-local-moran.spec.ts`: the helper expected NaN z and p for a row whose value equals the f32 mean. That branch is removed; such a row is now compared with the float64 oracle like any other.
- `spatial-weights/spatial-weights-oracle.ts`: the contiguity oracle anchors its snap grid at the first vertex.
- `scan-statistics/stx07-expectation-based-scan-reference.ts`: the replicate reference uses the open-interval uniform.
- `emerging-hot-spots/emerging-hot-spots.node.spec.ts`: node lists and counts include the two new nodes.

### Notes for the lead

- 07, a side effect found and removed. The first version subtracted the residual mean per bin in `GPUEmergingHotSpots`. `gpu-dataframe/value-cube/spatiotemporal-wave-two-composition.spec.ts` (another stream's spec) then failed: two windows whose values add up to the same total stopped having bitwise equal z, so the Mann-Kendall `S` lost a tie (-10 against -11). The residual is now subtracted once per sum, which keeps such ties exact; that spec passes unchanged. The shared front end of the local statistics subtracts per row, as the global and permutation kernels do; nothing there ranks z-scores.
- 05 for `GPUSpatialEmpiricalBayesRates` is a judgment call: the lead's note asked for the shared rule; the named reference (esda) counts an explicit zero-weight slot. The departure is documented in the class.
- `coordinateExtent` on `GPUContiguityWeights` is a new optional prop. The vertex data is on the GPU and the port carries no bounds, so this is the only place a named limit can be checked at construction. Drop it if a new prop is not wanted; the kernel fix does not depend on it.
- The Pearson adapter has the same defect as the Gamma one (not in the report): on the same 5-node graph two permutations of `y` are ordered one way by `GPUSpatialPearson` and the other way by the bivariate Moran pair sum (scratch script `fix-spatial-statistics/gamma-adapter.mjs`). Documented with the Gamma adapter.
- 08, one copy left outside this stream: `dataframe-analysis/pair-statistics/gpu-spatial-correlogram.ts:741` has a private `getTwoSidedPValue` without the floor. It is the `dataframe` stream's file and was not touched.
- `analysis-core/doc-parameters.node.spec.ts` reports `gpu-spatial-analysis/operators-weights.md` as stale. The stale section is the generated `GPUNeighborSearch` parameters (`neighbor-search-parameters.ts` was edited by another stream), not the contiguity paragraph changed here.
- Not finished: a permutation test of the actual cross-product Gamma or of the edge Pearson coefficient. It needs a new statistic in `GPUGlobalPermutationTest`, which is a feature, not a fix.

### Upstream gaps

None. Nothing in this scope has an official luma.gl implementation.

### Review follow-up

After the gates and the independent review (`fix-review-geometry.md`, `fix-review-statistics.md`).

| Item | Outcome | What changed (`file:line`) | Spec |
|---|---|---|---|
| fr-geometry-04: the byte-for-byte pin of the permutation WGSL failed in all 162 configurations | fixed (baseline regenerated after checking the text) | `test/gpu-spatial-analysis/computation-pack/local-permutation-kernels.node.spec.ts:14-27` (how the fixture is produced, the regeneration command and a baseline history that says what changed and why), `:73` (`UPDATE_LOCAL_PERMUTATION_KERNELS=1` rewrites the fixture), `computation-pack/fixtures/local-permutation-kernels.json` | the same spec, 1 test, passes without the update flag |
| fr-geometry-08: the contiguity snap key rounded an odd step in [2^23, 2^24) to the next even one | fixed | `spatial-weights/gpu-contiguity-weights.ts:383` imports `getCoverageSnapKeyWGSL` from `polygon-coverage-topology/coverage-topology-kernels.ts` in place of the private copy of the key (same origin rule, and the quotient is used as it is from 2^23 steps) | `spatial-weights/gpu-spatial-weights.spec.ts:392` (corners at steps 8388609 and 8388610, expected A-C only by inspection; written first and failed on the old key: every feature was a neighbour of every other) |

How the baseline was checked. The fixture is `JSON.stringify(kernels, null, 2)` plus a newline (the stored file round-trips byte for byte), so it was regenerated with the new flag and compared with the old file for all 162 configurations, not a sample (scratch script `fix-spatial-statistics/diff-kernels.mjs`): there is one change set, identical in every configuration. It is the comment and constant `ANALYTIC_P_VALUE_FLOOR` with the `clamp` line replacing the `min` line of `getTwoSidedPValue`, and the comment and body of `philoxOpenUnitFloat`. Removing those two blocks and restoring the `min` line reproduces the old text byte for byte in 162 of 162 configurations. No configuration calls either function, so no permutation result changes.

`fix-review-statistics.md`, "Needs a second look": none of its four items is in this stream's directories (group statistics and composite score are `dataframe-analysis`; the restricted regimes and the pairwise cascade are `spatial-regression`). The pattern behind the first two, a correction subtracted in the same expression as the value it corrects, was checked in this stream's kernels and is not present: the local statistics and the emerging hot spots read the stored deviation back from memory before subtracting the residual (`spatial-autocorrelation-kernels.ts:613`, `gpu-emerging-hot-spots.ts`, `block-squares` and the Gi* sum), and the global kernel's one-expression difference is `x * s - mean * s` with `s` a power of two, which is the same number however it is associated.

What that review did not cover in this scope:

- The new and changed specs of this stream were not read. The ones whose old failure was not certain by arithmetic were checked with a float64 replica of the old centring on the exact spec scenes (scratch `fix-spatial-statistics/old-centering-replica.ts`). Rounding the mean to the f32 grid moves the Gi* z of the hot-spot scene by up to 1.33 relative to `1 + |z|` (tolerance 2e-3); 590 rows of the local Moran scene equal the f32 mean, where the old kernel wrote NaN; the emerging scene moves by 0.18 (tolerance 2e-3 + 1e-3 |z|). So the three large-offset specs of 07 fail on the old kernels. The other fix specs fail on the old kernels by the arithmetic stated in each (an overflow, a clamp, or an exact count). The cross-product Gamma spec and the hand cases for the scan and the colocation quotient pin values that were already right; they are checks, not fix pins.
- 03, 04, 06, 09 and the scan uniform were read as a diff only, and the `docs/api` pages were searched, not read. Nothing further was done for these beyond the specs already listed.
- Its docs note (fr-statistics-06, "Kahan compensation" on `operators-regression.md` and `coverage-statistics.md`) is about `spatial-regression` classes, outside this stream. The pages this stream edited have no such wording.

Also aligned: `spatial-weights/spatial-weights-oracle.ts` (the contiguity oracle uses the quotient as it is from 2^23 steps) and `docs/api/gpu-spatial-analysis/operators-weights.md` (the `snapTolerance` bullet).
