# Math correctness review: point process, space-time tests, clustering, regionalization

Scope id: `point-process-clustering`. Date: 2026-11-10. Branch `upstream-first`, working tree.
Paths are relative to `modules/analysis/src`. `gsa/` is short for `gpu-spatial-analysis/`.

Scope directories (all under `gsa/`): `point-process`, `study-window`, `space-time-tests`,
`space-time-hawkes`, `space-time-statistics`, `change-points`, `distribution-dynamics`,
`spatial-clustering`, `spatial-regionalization`, `redcap-regions`.

Scratch scripts are in
`/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-luma-contributors/9972295f-91de-4373-ba95-d2f799611a84/scratchpad/point-process-clustering/`:
`change-point-variance-floor.mjs`, `skater-gain.mjs`, `standardize-constant.mjs`,
`redcap-vs-single-linkage.mjs`, `weak-components-iterations.mjs`. No spec was run.

Summary: 8 findings. 1 wrong-edge-case, 1 precision, 3 inconsistent, 3 doc-mismatch (one of them
is a method named for something it does not compute). No wrong result on ordinary input was found
in the kernels I read. Most estimators match their references.

---

## 1. Coverage

"Body" means everything except the import header.

**Read in full (body):**

- `gsa/point-process/`: `gpu-inhomogeneous-point-pattern.ts`, `translation-point-pattern.ts`,
  `polygon-translation-overlap.ts`, `polygon-overlap-metric.ts`, `point-process-utils.ts`,
  `gpu-point-pattern-envelope.ts`, `gpu-point-pattern-rank-envelope.ts`,
  `gpu-continuous-spatial-sample.ts`, `gpu-spatial-balanced-sample.ts`.
- `gsa/study-window/gpu-study-window-preparation.ts`.
- `gsa/space-time-tests/`: `space-time-kernels.ts`, `space-time-parameters.ts`,
  `space-time-stopping.ts`, `space-time-bands.ts`, `gpu-knox-test.ts`, `gpu-mantel-test.ts`,
  `gpu-jacquez-test.ts`, `gpu-near-repeat-knox.ts`, `gpu-local-knox.ts`.
- `gsa/space-time-hawkes/`: all three source files.
- `gsa/space-time-statistics/`: both source files.
- `gsa/change-points/`: all three source files.
- `gsa/distribution-dynamics/`: `gpu-transition-diagnostics.ts`, `gpu-spatial-markov.ts`,
  `gpu-lisa-markov.ts`, `gpu-rank-markov.ts`, `gpu-classification-fit.ts`,
  `gpu-class-assignment.ts`.
- `gsa/spatial-clustering/`: `spatial-clustering-kernels.ts`, `gpu-kmeans.ts`,
  `gpu-optics-xi.ts`, `hierarchical-density-tree.ts`.
- `gsa/spatial-regionalization/`: `regionalization-wgsl.ts`,
  `gpu-spatial-weights-minimum-spanning-tree.ts`.
- `gsa/redcap-regions/gpu-redcap-single-linkage.ts`.

**Read in part (named ranges):**

- `gsa/distribution-dynamics/gpu-transition-matrix.ts` 60-469 (types above not read).
- `gsa/spatial-clustering/gpu-optics.ts` 60-309 and 330-926 (both frontier paths and the DBSCAN
  extraction; the grid-frontier status node at the end only skimmed).
- `gsa/spatial-clustering/hierarchical-density-clustering-kernels.ts` 40-584 and 600-832.
- `gsa/spatial-clustering/gpu-hierarchical-density-clustering.ts` 50-299 (node wiring 300-583 not
  read).
- `gsa/spatial-clustering/gpu-spatial-clustering.ts` 60-260 (docs and props; node wiring 261-777
  not read).
- `gsa/spatial-regionalization/regionalization-inputs.ts` 60-309.
- `gsa/spatial-regionalization/gpu-skater-regions.ts` 60-200 and 744-973. **Not read: 201-743**
  (Euler tour, list ranking, subtree statistics) and 974-1052.
- `gsa/spatial-regionalization/gpu-azp-regions.ts` 60-71, 100-200, 380-714, 728-1140.
- `gsa/spatial-regionalization/gpu-max-p-regions.ts` 160-235 and 755-1174. **Not read: 236-754**
  (setup, seed-wave control) and 1175-1290.
- `gsa/spatial-regionalization/gpu-ward-regions.ts` 95-215 plus a search for the cost and
  objective lines. **The merge kernels were not read.**
- `gsa/spatial-regionalization/ward-active-graph.ts` 285-318 (the cost kernel only).
- `gsa/spatial-regionalization/gpu-region-partition-evaluation.ts` 60-130, 398-452, 655-712.

**Not opened:** every `index.ts`; `gsa/distribution-dynamics/period-views.ts`;
`gsa/spatial-clustering/spatial-clustering-parameters.ts`; `optics-xi-license.txt`;
`gsa/spatial-regionalization/ward-mutual-selection.ts` (searched, not read).

**Read outside the scope, to check a dependency:** `analysis-core/kernels/study-window-wgsl.ts`
205-389 (Epanechnikov window mass), `analysis-core/kernels/wgsl-poisson-observed-term.ts`,
`analysis-core/kernels/masked-weak-components.ts` 34-112, and the published
`@luma.gl/graph` declarations `gpu-vector-search/gpu-k-means.d.ts`,
`gpu-graph-algorithms/gpu-graph-connected-components.d.ts`, `gpu-core/gpu-scan.d.ts`, plus a
keyword scan of `gpu-vector-search/gpu-k-means.js`.

**Reference methods named in my brief but not in my directories.** Ripley's K, L, G, F and J,
Clark-Evans, the quadrat test and the space-time K function live in
`dataframe-analysis/pair-statistics/` (`gpu-ripley.ts`, `gpu-ripley-distance-functions.ts`,
`gpu-point-pattern-indices.ts`, `gpu-space-time-k-function.ts`). That directory is not in my
scope, so I did not review it. I only spot-checked the Clark-Evans constants in
`gpu-point-pattern-indices.ts:686-693` (0.5 sqrt(A/n), 0.26136 sqrt(A)/n, Donnelly 0.0514, 0.041,
0.0703, 0.037): they are the textbook values. The lead should confirm another reviewer owns that
directory.

---

## 2. Findings

| ID | Severity | Status | File:line | What the kernel computes | What the reference says | Concrete input and wrong output | Suggested fix |
| --- | --- | --- | --- | --- | --- | --- | --- |
| point-process-clustering-01 | wrong-edge-case | CONFIRMED (script) | `gsa/change-points/gpu-location-change-points.ts:409`, `:414`, constant `:385-387`, flat cost `:599` | `'variance'` and `'mean-variance'` costs clamp every segment variance to at least `1e-11` in data units (`max(variance, VARIANCE_FLOOR / scale)`) | R `changepoint` replaces the variance only when it is not positive. The cost is otherwise scale-free | 60 slices, alternating +-1 for 30 slices then +-5, values in units of 1e-6 (series variance 1.3e-11): kernel finds no change point; the same series in units of 1 finds slice 30 | Apply the floor in standardized units (`max(variance, VARIANCE_FLOOR)`, or replace only non-positive values) and write the flat cost the same way |
| point-process-clustering-02 | precision | CONFIRMED (script) | `gsa/spatial-regionalization/gpu-skater-regions.ts:415-416`, `:786-800` | SKATER cut gain as `SSD(R) - SSD(A) - SSD(B)` with each SSD as `sum x^2 - (sum x)^2 / n` in f32 | The same quantity is `n_A n_B / (n_A + n_B) * |mean_A - mean_B|^2`, which has no cancellation | Region of 1,000 rows, 500 at z = 2.000 and 500 at z = 2.001. Cut inside the first plateau: true gain 8.3e-5, kernel 0. Cut inside the second: true 8.3e-5, kernel 1.8e-4 | Compute the gain with the direct form, as `ward-active-graph.ts:309-313` already does. The `sum x^2` statistic is then not needed |
| point-process-clustering-03 | inconsistent | CONFIRMED (hand) | `gsa/distribution-dynamics/gpu-class-assignment.ts:135-139` | Class = number of inner edges `<=` value, so a value equal to a break goes to the upper class (left-closed classes) | mapclassify and giddy use right-closed bins (`x <= upper bound`): a value equal to a break stays in the lower class | Pooled values `[1, 2, 3, 4, 5]`, two quantile classes, break 3. mapclassify `yb = [0,0,0,1,1]`; kernel `[0,0,1,1,1]` | State the difference where giddy parity is claimed (`gpu-spatial-markov.ts:96-105`, `docs/api/gpu-spatial-analysis/operators-dynamics.md`), or add a `closed: 'right'` option shared with `GPUClassBreaks` |
| point-process-clustering-04 | inconsistent | CONFIRMED (script) | `gsa/spatial-regionalization/regionalization-inputs.ts:147-169` against `gsa/spatial-regionalization/gpu-spatial-weights-minimum-spanning-tree.ts:298-330` | Two column standardization kernels. The shared one (AZP, Ward, max-p) subtracts a rounded mean; the tree one first subtracts the first finite entry | Both documents say "a constant column becomes zero" | Constant column 37.3 over 791 rows: shared kernel gives 1.0 in every row (deviation 3.8e-6 from mean rounding); the tree kernel gives 0 | Keep one kernel, the tree's origin-centred version |
| point-process-clustering-05 | inconsistent | CONFIRMED (both documented) | `gsa/spatial-clustering/spatial-clustering-kernels.ts:49-50`, `:193-212` against `gsa/spatial-clustering/gpu-optics.ts:116` | Great-circle DBSCAN walks its own lattice and does not wrap at +-180. OPTICS takes neighbours from `GPUNeighborSearch`, which wraps | One library, one answer for "which points are within epsilon on the sphere" | Points at longitude 179.99 and -179.99, same latitude, epsilon 5 km: one DBSCAN cluster is split in two; OPTICS with DBSCAN extraction at the same epsilon keeps them together | Move DBSCAN to the shared neighbour search (already tracked as PLAT-03 for other reasons) |
| point-process-clustering-06 | doc-mismatch | CONFIRMED (arithmetic); reference recalled, see note | `gsa/redcap-regions/gpu-redcap-single-linkage.ts:116-126`, `:291-311`; `docs/api/gpu-spatial-analysis/geographical-detector-redcap-entropy.md:32` | Cuts the `k - c` heaviest spanning-tree edges: the contiguity-constrained single-linkage dendrogram cut at `k` | Guo (2008) REDCAP partitions the tree by removing, at each step, the edge with the largest SSD reduction. First-order single-linkage REDCAP is the SKATER partition | Path of six rows, values `[0, 1, 2.1, 3.3, 4.6, 6.0]`, `k = 2`. Kernel: `{0..4}, {5}` (within SS 13.26). REDCAP: `{0,1,2}, {3,4,5}` (within SS 5.85) | Rename and document the class as constrained single-linkage clustering, and say that first-order SLK REDCAP is `GPUSpatialWeightsMinimumSpanningTree` plus `GPUSkaterRegions` |
| point-process-clustering-07 | doc-mismatch | CONFIRMED (trace) | `gsa/spatial-clustering/gpu-kmeans.ts:134-136` against `:743-745`, `:766-769` | When every remaining k-means++ weight is 0 the race key is `0x7f800000` for every row, so the lowest valid row becomes the next centre (a duplicate, finite) | Class doc: a centre is NaN when "`'kmeans++'` [is] running out of weight" | Three identical points, `k = 2`, `'kmeans++'`: centre 1 is the point, not NaN | Correct the doc. The behaviour is the better one: the duplicate is then handled by the empty-cluster relocation |
| point-process-clustering-08 | doc-mismatch | still present (prior G10, S8) | `gsa/spatial-clustering/gpu-kmeans.ts:234`; `gsa/distribution-dynamics/gpu-rank-markov.ts:40` | Doc points to `@luma-contributors/analysis/gpu-vector-search`, which does not exist. Rank Markov says giddy 2.3.9; `dev-docs/coverage-matrix.md:150` says 2.3.6 | The published class is in `@luma.gl/graph/gpu-vector-search`. The fixture was generated with giddy 2.3.9, so the coverage matrix is the stale one | n/a | Fix both texts |

### Notes on the findings

**01. Variance floor.** The kernel standardizes each series by its own mean and standard deviation
so that the costs do not depend on units. It then divides the floor by `scale` (the series
variance), which turns the floor back into an absolute `1e-11` in data units. The comment at
`:385-386` says the floor is "relative to the series variance" and that "the changepoint package
floors the variance at 1e-11 likewise". Neither is what the code or the package does: `changepoint`
substitutes `1e-11` only when the variance is `<= 0`. Consequences by series variance: at 1e-9 the
floor is 1 percent of the series variance, so quiet segments are under-rewarded; at 1e-11 and
below every segment gets the same cost and no change point is ever returned. Data in small units
reach this easily (mole fractions near 1e-6, strain rates, values in metres with millimetre
noise are fine, micrometre noise is not). `change-point-variance-floor.mjs` replicates the dynamic
program in f32 with `Math.fround`: units 1, 1e-3 and 1e-5 return `[30]`; units 1e-6 and 1e-7
return `[]`, while the reference rule returns `[30]` at every unit. No test can see it: the
`'variance'` cost is pinned only by a binary64 copy of the same formula (see section 6).

**02. SKATER gain.** The class doc already warns about raw attributes and asks for the
standardized table. Standardizing fixes the global mean, not a region's mean. A homogeneous region
whose mean is a few standard deviations from zero still has `sum x^2` and `(sum x)^2 / n` equal
to seven digits. The error is about one ulp of the region's `sum x^2`, so gains below roughly
`1e-6 * sum x^2` are noise. It matters late in the greedy sequence, inside quiet regions, where
the cut choice then follows rounding. It also affects the optional `cutGains` output. My script
uses row-order sums, which is the best case; the kernel's subtree subtraction adds more rounding
(its own doc says so). Top-level cuts on ordinary data are not affected.

**03. Class interval side.** The rule is stated in the class doc, so the kernel does what it says.
The issue is parity: `GPUSpatialMarkov` and the dynamics docs cite giddy, and giddy classifies
with mapclassify. Quantile breaks land exactly on data values whenever the data have ties at the
break (rounded rates) or `(n - 1) j / k` is an integer. Every tied row at a break then sits one
class higher than in giddy, which moves transition counts. The giddy fixture
(`test/oracles/statistics-markov.py`) feeds precomputed class arrays, so it does not exercise the
rule. The same rule lives in `GPUClassBreaks` (`dataframe-analysis/column-classification`, not my
scope); whoever reviews that should own the decision.

**04. Two standardizers.** Prior review FX-12 tracks the duplication. New here: the two copies
differ numerically. With the shared kernel a constant column becomes a constant 1 (or -1). Region
differences are still zero, so SKATER, Ward and max-p are unaffected. AZP is slightly affected:
the column adds `2 (n_A + n_B)` to the noise scale at `gpu-azp-regions.ts:952`, and the move test
at `:961` refuses gains below `4e-6` times that scale. For two regions of 5,000 rows the refused
band is 0.08 in squared z units.

**06. REDCAP.** The code is a correct constrained single-linkage clustering, and its "Reference"
paragraph says so honestly. The name, the first doc sentence and the user doc call it REDCAP.
REDCAP's second step is a tree partition that optimizes within-region SSD; it is what makes
REDCAP differ from a plain dendrogram cut, and it is why GeoDa documents first-order
single-linkage REDCAP as identical to SKATER. I recalled the definition from Guo (2008) and the
GeoDa documentation; I could not re-read either here. Please get a second opinion on the reference
before renaming. Single linkage also has the well-known chaining behaviour (it peels off single
outlier rows), so users who expect REDCAP-quality regions will be surprised on ordinary data.

---

## 3. Verified correct

Each line: contributor, what I compared it against.

- `GPUInhomogeneousPointPattern`, border K: spatstat `Kinhom` reduced-sample ratio
  `sum_i 1{b_i > r} w_i sum_j w_j 1{d_ij <= r} / sum_i 1{b_i > r} w_i`, `w = 1 / lambda`. Matches.
- Same, estimated intensity: 2D Epanechnikov `2 / (pi h^2) (1 - u^2)`, leave-one-out, divided by
  the kernel mass inside the window at the evaluation point (spatstat uniform edge correction).
  I re-derived the polygon mass integral in `study-window-wgsl.ts:327-343` by hand: deficit per
  fan is `atan(x) + (q^2 - 2q) x + q^2 x^3 / 3` with `q = (p/h)^2`; the code is the same expression
  in the along-edge variable.
- Same, translation K: `sum over ordered pairs 1 / (lambda_i lambda_j |W ∩ W_h|)`, spatstat
  `edge.Trans` with `renormalise = FALSE`. Matches. Pair index inversion by `sqrt` is corrected by
  one step each way and is exact below 2^24 pairs.
- Same, kernel pcf: `sum w k_h(r - d) / (2 pi r)` or `/ (2 pi d)`, 1D Epanechnikov
  `0.75 / h (1 - t^2)`, reflection term `k_h(r + d)`. Matches spatstat `pcfinhom` with half-width
  `h`. The doc states that `bandwidth` is the half-width, not the standard deviation.
- `polygonTranslationOverlap`: signed fan triangle identity and Sutherland-Hodgman clipping of two
  counter-clockwise triangles. Orientation handling and the refusal threshold are sound.
- `GPUPointPatternEnvelope`: pointwise rank-th order statistics; MAD and DCLF global tests with
  `(1 + #{sim >= obs}) / (S + 1)`. Matches spatstat `envelope`, `mad.test`, `dclf.test`.
- Extreme-rank envelope: midranks over the `S + 1` curves, `p- = #{R_i < R_obs} / (S + 1)`,
  `p+ = #{R_i <= R_obs} / (S + 1)`, critical rank and envelope from the k-th smallest and largest
  of all curves. Matches Myllymäki et al. (2017).
- `GPUSpatialBalancedSample`: inclusion probability is exactly `n / N` (or `ticket / step`).
  Rejection sampling threshold `2^32 mod L` is right. Hierarchy and selection Philox counters do
  not collide (word 2 high bit separates them).
- `GPUContinuousSpatialSample`: random-shift lattice covers every frame point with intensity
  `1 / spacing` (line) or `1 / spacing^2` (area); weights are the inverse.
- `GPUStudyWindowPreparation`: shoelace area about the first vertex with a pairwise tree, role by
  ring position, interior-left edge orientation, perimeter. Correct.
- `GPUKnoxTest`: count of pairs close in both; `E = S T / (n (n - 1) / 2)`; `T` by sort and binary
  search with the same f32 predicate as the pair kernel (f32 subtraction is monotone, so the
  search is valid). Matches pointpats `Knox`.
- Shared summary: pseudo p `(r + 1) / (m + 1)`, sample variance with `P - 1`, Besag-Clifford
  `h / L` when stopped. Correct.
- `getKnoxPoissonPValue`: `P(X >= observed)` by a log-space lower-tail sum. Correct.
- `GPUMantelTest`: Pearson r of transformed distance and `|dt|`, centred sums; pointpats
  transform `(x + c)^p` now available (prior S5 is fixed, `gpu-mantel-test.ts:146-170`).
- `GPUJacquezTest`: directed sum over spatial k-NN entries whose partner is in the temporal k-NN
  interval; `E = sum |N_i| k / (n - 1)`. Matches pointpats `jacquez` (which folds the p-value; the
  class reports the upper tail and says so).
- `GPUNearRepeatKnox`: band rule `(e_k, e_k+1]` with band 0 closed at `e_0`, per-cell p-values.
  Consistent with the `<=` of Knox.
- `GPULocalKnox`: the swap `pi(j) = i -> pi(i)` yields a uniform permutation of the other `n - 1`
  times (each conditional permutation has exactly `n` preimages). Matches pointpats `KnoxLocal`.
- `GPUSpaceTimeHawkes`: intensity, log-likelihood `sum log lambda - (mu A T + omega sum (1 -
  exp(-beta (T - t_i))))`, E-step by log-sum-exp, M-step `mu = sum p_jj / (A T)`,
  `omega = sum (1 - p_jj) / sum (1 - exp(-beta (T - t_i)))`, `beta = sum p / sum p lag`,
  `sigma^2 = sum p d^2 / (2 sum p)`. Matches Veen and Schoenberg (2008) for this kernel. The
  small-argument series for `1 - exp(-x)` is right.
- `GPUSpaceTimeCrossCorrelation`: overlap `[max(0, -l), min(P, P - l))`, two-pass Pearson, Fisher
  z `erfc(|atanh r| sqrt(n - 3) / sqrt 2)`, Numerical Recipes `erfcc` coefficients. Correct.
- `GPUSpaceTimeLocalMoran`: slices the cube and delegates to `GPULocalPermutationTest` (not my
  scope). The slice addresses are right for both layouts.
- `GPUPettittTest`: `U_t = U_(t-1) + sum_j sign(x_t - x_j)`, `K = max |U_t|`,
  `p = 2 exp(-6 K^2 / (n^3 + n^2))`. Correct.
- `GPULocationChangePoints`: optimal partitioning `F(t) = min_s F(s) + C + beta`, `F(0) = -beta`;
  `'mean'`, `'slope'` (exact u32 time sums, residual `Szz - Stz^2 / Stt`), `'poisson'` (deviance
  form whose linear part telescopes over a partition), penalty scaling, backtracking. Correct,
  apart from finding 01.
- `GPUTransitionMatrix`: exact counts, `p_ab = n_ab / n_a`, tile coverage. Correct.
- `GPUTransitionDiagnostics`: `Q = sum (O - E)^2 / E` with `E = n_im p_ij`, `LR = 2 sum O ln(O /
  E)`, degrees of freedom (documented departure from giddy for empty rows), GTH elimination.
  Matches giddy `homogeneity` and `steady_state`.
- `GPURankMarkov`: descending ordinal ranks with giddy's reversed tie order; geographic rank
  occupants. Using descending instead of giddy's ascending ranks relabels the rank index only, so
  the pooled counts are the same.
- `GPUClassificationFit`: numpy medians, ADCM, ADAM, GADF, mapclassify `tss`. Correct.
- `GPUSpatialClustering` (DBSCAN): self-inclusive core count, closed radius, lock-free union-find
  that only hooks a larger root under a smaller one (no cycle, root is the smallest core row),
  border points to the smallest adjacent root. I checked that this reproduces scikit-learn's
  border rule: clusters are grown in order of their first core row, so a contested border point
  gets the cluster with the smallest minimum core row. Label order also matches.
- Great-circle cell width `2 asin(sin(eps / 2R) / cos(max |lat|))`: I re-derived it from the
  haversine identity for two points inside the latitude bounds. It is an exact bound.
- `GPUOPTICS`: selection by (reachability, row), relaxation `max(core, d)` within `max_eps`, core
  distance infinite when fewer than `minimumSamples` rows are within `max_eps`. DBSCAN extraction
  `cumsum(far_reach & near_core) - 1` with noise for `far_reach & !near_core`. Matches
  scikit-learn. Lanes own disjoint rows, so the cooperative relaxation has no race.
- `createOPTICSXiNode`: compared line by line with scikit-learn 1.7.2 `_xi_cluster`,
  `_update_filter_sdas`, `_extend_region`, `_correct_predecessor`, `_extract_xi_labels`. Same
  logic, including the infinity and zero cases of the steepness ratio.
- `GPUHierarchicalDensityClustering`: core distance, mutual reachability, Borůvka with the total
  order (weight, low, high) and mutual-edge deduplication, condensed-tree rows and stability
  `sum (lambda_row - birth) * size`, excess-of-mass selection with equality favouring the parent,
  root excluded, labels from the topmost selected ancestor. The max-plus heavy-path composition is
  algebraically right. Matches scikit-learn `HDBSCAN` defaults. The BVH stack cannot overflow
  (depth-first, at most one net push per level, capacity `2 * levels`).
- `GPUKMeans`: exponential race `-ln(u) / w` samples proportional to `w`; uniform first centre;
  strict nearest with lowest ID; mean update; relocation follows scikit-learn
  `_relocate_empty_clusters_dense` including the "all distances zero" return; final assignment
  against the final centres; best-of-restarts by inertia.
- `GPUSpatialWeightsMinimumSpanningTree`: Borůvka with (cost, edge slot) order, mutual pairs
  rooted at the lower label, enough pointer jumps for a chain of the active trees. Unique forest.
- `GPUSkaterRegions`: greedy choice of the best cut over all regions, minimum size on both sides,
  prefix property. Matches Assunção et al. (2006). The oracle correctly overrides spopt's Manhattan
  default. Gain numerics: finding 02. Tree statistics (201-743): not read.
- `GPUWardRegions`, cost only: `n_A n_B / (n_A + n_B) |mean_A - mean_B|^2`. The claim that the
  objective is the sum of the merge increases is true. Merge kernels not read.
- `GPUAZPRegions`: move delta `S_A^2/n_A + S_B^2/n_B - (S_A - v)^2/(n_A - 1) - (S_B + v)^2/(n_B +
  1)`; two-stage claim gives each region at most one move (keys are a bijection), the best move
  always wins, so gains add and contiguity arguments hold. The initial-contiguity
  Shiloach-Vishkin pass hooks roots under smaller roots only and has a final check that fails
  closed if the round bound is hit. The local articulation proof in the code is valid.
- `GPUMaxPRegions`, growth and enclave kernels: one row per region per round, the incremental
  SSD `n / (n + 1) d^2` uses the pre-update mean, a row joins only a region one of its neighbours
  is in, completed regions never lose rows. Start selection as documented.
- `GPURegionPartitionEvaluation`: two-pass within SS, Calinski-Harabasz
  `(B / (k - 1)) / (W / (n - k))` with scikit-learn's value 1 for `W = 0`.
- `GPURedcapSingleLinkage` as constrained single linkage: correct. Label propagation converged in
  at most 25 synchronous iterations on paths of 262,144 rows in four row orders
  (`weak-components-iterations.mjs`), so the default of 32 is ample; it is still silent when the
  optional `status` is not bound.

---

## 4. Standards consistency

- **Standardization, two kernels** (finding 04). Numerically different on constant columns.
- **Between-group sum of squares, two formulas.** SKATER uses SSD differences
  (`gpu-skater-regions.ts:797-799`); Ward uses the direct form (`ward-active-graph.ts:309-313`).
  Same quantity, different conditioning (finding 02).
- **Class interval side** (finding 03): left-closed here, right-closed in the cited library.
- **Antimeridian** (finding 05): DBSCAN does not wrap, OPTICS does.
- **Random streams.** Knox, near-repeat Knox, local Knox and Jacquez all use
  `PHILOX_PURPOSES.knoxTestShuffle` (`gpu-knox-test.ts:316`, `gpu-near-repeat-knox.ts:345`,
  `gpu-local-knox.ts:295`, `gpu-jacquez-test.ts:317`); Mantel has its own. Each doc says so and
  each test stays valid, but four tests run with one seed share their permutations, so their
  p-values are not independent. A purpose per contributor would match the contract.
- **Non-Philox randomness.** k-means++ draws from `hashUint32(row ^ roundSeed)`
  (`gpu-kmeans.ts:727-745`); AZP and Ward tie keys use the same hash. Known from the prior review;
  still present. For k-means it is a sampling use, not a tie-break.
- **Pseudo p-value rule.** `(r + 1) / (m + 1)` everywhere in scope. Consistent.
- **Variance divisor.** Population (`n`) for standardization and for the change-point scale;
  sample (`P - 1`) for permutation variances. Each is documented.
- **Private copies of shared helpers, all numerically identical to the shared ones:**
  `sampleFinite` (`gpu-continuous-spatial-sample.ts:266`), `isFiniteNumber`
  (`hierarchical-density-clustering-kernels.ts:63-65`), `getNaNBits` and `getNaNDeclaration`
  (`gpu-space-time-hawkes.ts:423`, `:715-717`), the ordered-float key in
  `gpu-point-pattern-envelope.ts:64-67` and `gpu-classification-fit.ts:226`, and two copies of
  the path-halving `findRoot` and `unite` (`spatial-clustering-kernels.ts:530-563`,
  `hierarchical-density-clustering-kernels.ts:76-90`, `:533-549`). The binary searches in
  `gpu-knox-test.ts:458-467`, `space-time-hawkes-wgsl.ts:88-97`, `regionalization-wgsl.ts:13-44`
  and `gpu-classification-fit.ts:312-324` are local lower-bound loops with correct bounds.
- **Literals.** `3.141592653589793` and `6.283185307179586` are written out in the point-process
  kernels; `6.2831853` in `space-time-hawkes-wgsl.ts:82`. Harmless in f32.
- **Sphere distance.** DBSCAN and OPTICS both use the shared `GEODESIC_WGSL` and the context's
  sphere radius. No private haversine.

---

## 5. luma.gl official implementations

**`GPUKMeans` (local) against `GPUKMeans` in `@luma.gl/graph/gpu-vector-search`.**

| Aspect | Published | Local |
| --- | --- | --- |
| Data | N-dimensional embedding rows | 2D points only |
| Seeding | Evenly spaced valid rows, no randomness | k-means++ by exponential race (default) or first valid rows |
| Assignment | Nearest, strict `<`, lowest cluster wins | Same |
| Update | Mean, serial fixed order | Mean, chunked fixed order |
| Empty cluster | Keeps its previous centroid | Relocated to the farthest points (scikit-learn rule) |
| Convergence | No label changed | Largest centre shift `<= tolerance` |
| Final labels | From the assignment before the last update (can lag the centroids by one update at the iteration limit) | Extra assignment against the final centres |
| Extras | status `[iterations, changed, converged]` | restarts, inertia, per-point squared distance, origin centring |

The two agree on the Lloyd step itself. They differ in seeding, empty-cluster policy, convergence
test and the final assignment, so they return different clusterings on the same data. Neither is
wrong; the local one is closer to scikit-learn. The local class cannot delegate today without
changing its results. Recommendation, same direction as prior G10: propose k-means++ seeding,
restarts, relocation, a tolerance and an inertia output upstream as options of the published
class; then make the local class a 2D adapter over it. Until then rename the local class so the
two public names do not collide, and fix the doc path (finding 08).

**Connected components.** `@luma.gl/graph/gpu-graph-algorithms` publishes
`GPUGraphAlgorithmsConnectedComponents` (weak components, lowest vertex ID, 32 iterations by
default, optional `converged`). Two local re-implementations in scope give the same labels:
AZP's contiguity check (`gpu-azp-regions.ts:383-552`) and REDCAP's region labels through
`analysis-core/kernels/masked-weak-components.ts`. Both need an edge filter (same label, kept
edge) that the published class does not take: it reads a `GPUGraphAlgorithmsTopology`. Options:
build a filtered topology and delegate, or propose an edge-mask input upstream. The union-find in
DBSCAN and HDBSCAN works on implicit geometric edges and has no published equivalent.

**Delegations that are correct.** `GPUSort` (stable, used for time order, class segments, tree
edges), `GPUScan` (default exclusive, checked in the published declaration; OPTICS passes
`'inclusive'` explicitly and subtracts one), `GPUCompaction`, `GPUFlagOffsets`,
`GPUGroupAggregation`. I found no argument that changes a meaning.

**Nothing else in scope has a published counterpart.** `gpu-timeseries` publishes only
`GPUTimeSeries` and an envelope; no change-point or cross-correlation class.

---

## 6. Test oracles

**Independent oracle (reference library or pinned numbers):**

- Inhomogeneous K (border, polygon with hole, supplied and estimated intensity), translation pcf,
  DCLF: spatstat.explore 3.8.2 fixtures (`test/gpu-spatial-analysis/point-process/
  generate-spatstat-fixtures.R`, `generate-translation-pcf.R`).
- Extreme-rank envelope: scipy midranks. OPTICS ordering, DBSCAN and xi extraction: scikit-learn
  1.7.2.
- Study window: `parity-wp1-study-window.py`.
- Mantel: scikit-bio and scipy. Near-repeat Knox, local Knox, Jacquez: pointpats 2.6.0
  (`space-time-near-repeat.py`). Knox itself: hard-coded pointpats numbers for one example plus
  the in-repo swap-or-not oracle (prior S6; the pointpats fixture now covers the Knox count
  through the near-repeat cells).
- Space-time local Moran and cross-correlation: esda, numpy, scipy (`space-time-statistics.py`).
- Change points: ruptures Pelt for `'mean'`, `'slope'`, `'mean-variance'`; pyhomogeneity for
  Pettitt (`stx06-change-points.py`).
- Markov family: giddy 2.3.9 and esda (`statistics-markov.py`), with precomputed classes.
  Classification fit: mapclassify (`distribution-mapclassify-fit.py`).
- DBSCAN: scikit-learn, planar and haversine (`parity-wp1-spatial-clustering.py`,
  `parity-wp1-clustering-modes.py`). ST-DBSCAN and ST-OPTICS: `st_dbscan` 0.2.3.
- HDBSCAN: scikit-learn (`spatial-clustering-hdbscan.py`).
- k-means relocation and inertia: frozen scikit-learn runs (`kmeans-oracle.ts:190-202`).
- MST: scipy. SKATER: spopt on the same tree with squared Euclidean cost
  (`parity-wp1-skater-lag.py`). Ward: scipy and scikit-learn (`statistics-ward.py`).
- AZP and max-p: spopt, quality bounds only (labels are not comparable).

**Only a copy of the same formula:**

- `GPUSpaceTimeHawkes`: a binary64 EM written from the class doc, plus parameter recovery. The
  reference file says "EXTERNAL FIXTURE PENDING". I checked the formulas by hand.
- `GPULocationChangePoints` `'variance'` and `'poisson'` costs: binary64 reference in the spec;
  the oracle header says an R `changepoint` fixture is pending. This is why finding 01 is not
  caught.
- `GPUKMeans` `'first-valid'` Lloyd iterations: an f64 oracle with the same rules.
- `GPURedcapSingleLinkage`: scipy MST plus a cut (`stx08-spatial-methods.py`). Independent
  arithmetic, but it pins the single-linkage definition, not REDCAP.
- Knox permutation null and Besag-Clifford stopping: in-repo oracle with the same permutations.

**Nothing independent that I could find:**

- k-means++ sampling: only "deterministic per seed" and "recovers separated blobs". No test that
  the draw is proportional to squared distance.
- `GPUSpatialBalancedSample` and `GPUContinuousSpatialSample`: empirical inclusion frequencies
  and analytic moments (good property tests), no reference package.
- `GPUClassAssignment` edge rule against mapclassify: not exercised (finding 03).
- `GPURankMarkov` mobility output: no reference.
- The great-circle DBSCAN fixture states "no antimeridian crossing", so finding 05 is not
  exercised either.

---

## Resolution (2026-11-10)

Source paths are relative to `modules/analysis/src/gpu-spatial-analysis` (`gsa/` above), spec paths
to `modules/analysis/test/gpu-spatial-analysis`. Every spec named below was run on the test GPU
(headless Chromium, Metal). The new cases for 01, 02, 04, 05 and for the extra findings E1, E2 and
E5 were run against the old kernel first and failed there. 03 is a new option, 06 to 08 are
documentation. mapclassify, giddy, libpysal and ruptures are not importable on this machine, so no
reference fixture of those libraries was regenerated; new expected values are hand-computed,
closed forms, or float64 evaluations of the definition. (Corrected in the review follow-up: an
earlier version of this paragraph said NumPy was not importable either. The system
`/usr/bin/python3`, CPython 3.9.6, has NumPy 2.0.2 and SciPy 1.13.1 in the user site-packages; the
`python3` first on the path, 3.15, has neither. A NumPy reference for 03 was added, see below.)
Closing pass (2026-11-11): that reference was rerun on the pinned set and mapclassify 2.11.0 and giddy 2.3.9 give the same right-closed classes and transition counts; see [fixtures-pinned-rerun.md](./fixtures-pinned-rerun.md).

| ID | Outcome | What changed | Spec that pins it | Behaviour change a caller sees |
| --- | --- | --- | --- | --- |
| point-process-clustering-01 | fixed | `change-points/gpu-location-change-points.ts:413`, `:416`: the floor is `max(variance, VARIANCE_FLOOR)` in standardized units, that is `1e-11` times the series variance (`:388-392`). The flat cost (`:613-617`) keeps the `changepoint` substitute `1e-11` in data units, because a constant series has no variance to be relative to; the comment now says so. Class documentation and `docs/api/gpu-spatial-analysis/change-points-and-expectation-based-scan.md` | `change-points/stx06-location-change-points.spec.ts`, "variance costs do not depend on the unit of the values" (two cases: `'variance'` and `'mean-variance'` at units 1, 1e-3, 1e-6, 1e-7; change point and `optimalCost` by hand) | A series in small units (variance near or below `1e-11`) now gets the same change points as the same series in units of 1; before it got none. For other series the only difference is the value given to an exactly constant segment: `1e-11 * seriesVariance` instead of `1e-11` |
| point-process-clustering-02 | fixed | `spatial-regionalization/gpu-skater-regions.ts:797-814`: the gain is `nA nB / (nA + nB) * |meanA - meanB|^2` (division last, so small integer inputs stay exact). The sum-of-squares statistic is gone (word 1 of the statistics tables is unused; the stride and the documented capacity are unchanged). The subtree and tree sums (`:680-700`, `:735-752`) accumulate the exact rounding error of every addition (`getAdditionRoundingError`). Remaining error, stated in the class documentation: the last place of the region's attribute sum, spread over the smaller side, per cut | `spatial-regionalization/gpu-skater-regions-gain.spec.ts` (new, three cases: the report's two plateaus, four quiet levels at a mean of 3, a subtree of 32,767 rows at 3.0001); expected cuts and gains are float64 `SSD(R) - SSD(A) - SSD(B)` about float64 means, or a closed form | Cuts inside quiet regions whose mean is far from zero move to where the true gain is largest, and `cutGains` is right there (the report's case: gain 2.5e-4 at the plateau boundary, before a cut 354 rows away). Gains on ordinary data change in the last places |
| point-process-clustering-03 | fixed by an explicit option; default unchanged; roadmap 12.3 default reversed by the lead | `distribution-dynamics/gpu-class-assignment.ts`: new exported type `GPUClassIntervalClosure` and optional compile-time prop `intervalClosure: 'left' | 'right'` (default `'left'`, the existing rule); `'right'` counts inner edges strictly below the value (`:152-180`), the rule of mapclassify and giddy. `distribution-dynamics/gpu-spatial-markov.ts:91`, `:296`: optional `lagIntervalClosure` passed through. Documentation: which kind of edge needs which value, on both classes, in `docs/api/gpu-spatial-analysis/operators-dynamics.md`, and per break method on `GPUClassBreaks` (`dataframe-analysis/column-classification/gpu-class-breaks.ts:178-190`, documentation only) | `distribution-dynamics/gpu-distribution-dynamics.spec.ts`: "intervalClosure 'right' keeps a value equal to an edge in the lower class" (hand-written classes for both closures), "pooled quantile breaks with rows tied at a break count giddy's transitions with 'right'" (two rows tied at each tercile; classes and transition counts by hand from the mapclassify rule, not from a giddy run), "natural-breaks lower edges with 'left' keep the smallest value of each class in its class" | None by default. With `'right'`, rows equal to an inner edge are one class lower than before |
| point-process-clustering-04 | fixed (converged on 0 for a constant column) | `spatial-regionalization/regionalization-inputs.ts:153-181`: the shared standardizer now takes each column about its first finite entry, the tree's rule. `spatial-regionalization/gpu-spatial-weights-minimum-spanning-tree.ts:262-272` calls it instead of holding a copy, so there is one kernel | `spatial-regionalization/standardize-columns.spec.ts` (new): constant column 37.3 over 791 rows is exactly 0 in both, other columns against float64 z-scores, and the two tables are equal bit for bit | AZP, Ward and max-p: a constant column is 0 instead of 1 (or -1) in the standardized table. A non-finite entry is copied to the table as it is (before, an infinity became an infinity or NaN). AZP's refused-gain band no longer includes the constant column. Standardized values of other columns change in the last places |
| point-process-clustering-05 | fixed | `spatial-clustering/spatial-clustering-kernels.ts:218-247`: in great-circle mode the neighbour walk also scans the columns within one cell width of the point's images at `x - 360` and `x + 360`, skipping columns already scanned. Documentation in the kernel, the `spatialContext` prop and `docs/api/gpu-spatial-analysis/operators-statistics.md` | `spatial-clustering/gpu-spatial-clustering.spec.ts`, "joins great-circle neighbours across the antimeridian": four points by hand, and blobs centred on +-180 at three latitudes and four epsilons against the brute-force haversine oracle | A cluster straddling +-180 is one cluster; before it was split (or, with `minimumPoints` above the half counts, noise). Reproduces only when the latitude bounds stay away from the poles: bounds reaching +-90 already used one 360-degree column and so already joined them. `clusterCentroids` of a straddling cluster are still arithmetic means of the longitudes (documented) |
| point-process-clustering-06 | fixed (documentation only; no rename, no algorithm change) | `redcap-regions/gpu-redcap-single-linkage.ts:115-145`: the class is described as contiguity-constrained single-linkage clustering, with a "What this is not" paragraph (REDCAP's SSD tree partition, the six-row example, pointer to `GPUSpatialWeightsMinimumSpanningTree` plus `GPUSkaterRegions`). `docs/api/gpu-spatial-analysis/geographical-detector-redcap-entropy.md` and the one-line summary in `ontology/spatiotemporal-review-catalog.ts:673`. The text says the description of Guo's second step was not re-read against the paper | none (no formula change); the example's two within-region sums of squares (13.26 and 5.85) were recomputed | None |
| point-process-clustering-07 | fixed (documentation only) | `spatial-clustering/gpu-kmeans.ts:133-140`: a centre is NaN only when it could not be initialized; when k-means++ runs out of weight the lowest valid row is repeated | `spatial-clustering/gpu-clustering-extras.spec.ts`, "'kmeans++' repeats a point, not NaN, when no distance is left to draw from" | None |
| point-process-clustering-08 | fixed (documentation only); the giddy half is the other way round | `spatial-clustering/gpu-kmeans.ts:238`: the published class is in `@luma.gl/graph/gpu-vector-search`. `distribution-dynamics/gpu-rank-markov.ts:40-42`: the rank Markov reference (`test/oracles/fixtures/statistics-upstream.json`) was generated with giddy 2.3.6, so `dev-docs/coverage-matrix.md` was right and the class documentation (2.3.9, the version of the other Markov fixture) was stale | none | None |

### Extra findings fixed in this pass

| ID | Outcome | What changed | Spec that pins it | Behaviour change a caller sees |
| --- | --- | --- | --- | --- |
| E1: dead compensated prefix sums in `GPULocationChangePoints` | fixed | Two things were wrong on the GPU. The float Neumaier term was folded away. And after replacing it with `getAdditionRoundingError`, the difference of two compensated prefixes, `(end.x - start.x) + (end.y - start.y)`, was still reassociated by the shader compiler (the case below still failed, and passed once the high difference was made opaque). The dynamic program is now in forward form (`change-points/gpu-location-change-points.ts:497-548`): each segment's moments are running sums from its own start, about its first value, so no prefix difference is formed. Slope time sums are exact `u32` | `change-points/stx06-location-change-points.spec.ts`, "keeps a quiet half that a plain float32 prefix sum would lose" (512 slices, +-1000 then +-1; change point and cost by hand; the old kernel was off by 6,484 in cost) | A quiet segment after a loud one is costed correctly. `optimalCost` and, where two partitions were within rounding, change points differ from before. The ruptures and binary64 parity cases pass unchanged |
| E2: dead Kahan sum in `GPUStudyWindowPreparation` | fixed | `study-window/gpu-study-window-preparation.ts:426-452`: ring totals (area, perimeter, edge count) accumulate the exact rounding error | `study-window/gpu-study-window-preparation.spec.ts`, "totals a large ring and a thousand small ones exactly" (old result 33,554,432, right result 33,555,432) | Window area and perimeter of a multipolygon whose first ring dwarfs the rest are right; they feed every edge-corrected point-pattern statistic |
| E3: dead Kahan sum in the `GPUKMeans` inertia | fixed, no visible change | `spatial-clustering/gpu-kmeans.ts:1183-1196`: the lane sums use `getAdditionRoundingError`. The plain sum was already right to about 1e-6 because lanes hold at most 256 terms and merge in fixed binary trees | `spatial-clustering/gpu-clustering-extras.spec.ts`, "sums the inertia of a million equal terms to float32 accuracy" (passes before and after) | None on the test GPU |
| E4: `GPUSpaceTimeHawkes` documentation | fixed (documentation only) | The Neumaier merge it described lives in `spatial-regression/regression-row-tiles.ts`, which the regression stream already replaced with a pairwise sum. `space-time-hawkes/gpu-space-time-hawkes.ts:196-201` and `docs/api/gpu-spatial-analysis/space-time-hawkes.md` now say so | `space-time-hawkes/stm08-space-time-hawkes.spec.ts` (unchanged, passes) | None |
| E5: one non-finite value turned every `GPURegionPartitionEvaluation` statistic into NaN | fixed | `spatial-regionalization/gpu-region-partition-evaluation.ts:350-380`: a labelled row with a NaN or infinite value in any column is treated as unlabelled in every kernel, counted in `ignoredRows`, and counted by the new optional output `invalidRowCount`. `GPUGeographicalDetector` already filtered such rows before calling it, so `q` was not affected there | `spatial-regionalization/gpu-region-partition-evaluation-nonfinite.spec.ts` (new; hand-computed sums for NaN and for an infinity) | Summary, sizes and means are finite and exclude the row; `ignoredRows` includes it. Also reaches the evaluation inside `GPUMaxPRegions` |
| E6: k-means++ random numbers | fixed | `spatial-clustering/gpu-kmeans.ts:733-751`: one Philox 4x32-10 block per `(row, round)`, key `(restart seed, PHILOX_PURPOSES.kMeansPlusPlusSeeding)` (new purpose `0x4b4d5050` in `analysis-core/kernels/philox-random.ts:92-96`), uniform from `philoxOpenUnitFloat`. `getKMeansRestartSeed` is unchanged (it derives a seed, it draws nothing) | `spatial-clustering/gpu-clustering-extras.spec.ts`, "'kmeans++' draws each next center in proportion to its squared distance" (new: 1,500 seeds, closed-form probabilities of the six draw orders, chi-square); `test/utils/philox-random.node.spec.ts` | Every seeded `'kmeans++'` result changes: the centres a given `seed` selects are different from earlier builds. The distribution is the same |
| E7: a NaN SKATER gain could win a cut | fixed | `spatial-regionalization/gpu-skater-regions.ts:809-813`: a non-finite gain is not a candidate | `spatial-regionalization/gpu-skater-regions-nonfinite.spec.ts`, "does not let a non-finite gain win a cut when no tree status is connected" | Only with non-finite attributes and no `treeStatus`: the region holding them is not cut |

### Not done

- Renaming `GPUKMeans`: deferred (roadmap 12.4).
- A second opinion on Guo (2008) for 06 was not obtained; the documentation states what the kernel
  computes and marks the description of REDCAP's partition step as recalled.
- Not read for math defects: the Ward merge kernels, `ward-mutual-selection.ts`,
  `gpu-max-p-regions.ts:236-754`. `gpu-skater-regions.ts:201-743` was read in full: the Euler tour,
  list ranking (`ceil(log2(2 rows))` jumps), orientation, interval tests for relabelling and
  ancestor updates are right; the only defects were the serial float sums (fixed under 02) and E7.
- Seen and at first left alone, then fixed in the review follow-up below:
  `GPURegionPartitionEvaluation` reports `betweenSumOfSquares` as
  `total - within`, a difference of two float sums merged serially over row tiles. When the
  partition explains a small share of the variance the between value and `pseudoF` lose digits in
  proportion. The direct form `sum n_r |mean_r - mean|^2` would not; it needs a per-label pass.
- `modules/analysis/test/analysis-core/doc-parameters.node.spec.ts` reports
  `gpu-spatial-analysis/operators-weights.md` as stale. That page is not in this scope and was not
  regenerated here; the pages this stream edited are in sync.

### Observations for the other streams

- A difference of two compensated (sum, error) prefixes is reassociated by the shader compiler
  just as the compensation term is. Keeping the error term alive is not enough when the result is
  `(end.sum - start.sum) + (end.error - start.error)`: use running sums from the range start, or
  remove the common part before summing.
- Float division on the test GPU is not correctly rounded: `6.0 / 3.0` came back as
  `1.9999999` in a region mean. Do not assert exact quotients in specs.
- Great-circle lattices whose latitude bounds reach a pole use one 360-degree column, which hides
  antimeridian defects in a test. Use bounds short of the poles.

### Upstream gaps

- `GPUKMeans` (`@luma.gl/graph/gpu-vector-search`): k-means++ seeding from a seed, restarts with an
  inertia output, a centre-shift tolerance and empty-cluster relocation, before the local 2D class
  can delegate.
- `GPUGraphAlgorithmsConnectedComponents`: an edge mask input, so REDCAP's and AZP's weak
  components can delegate.

### Review follow-up

From `fix-review-models.md` (fr-models-04, fr-models-09) and the lead's third item. All specs
below were run on the test GPU after the changes.

| Item | Outcome | What changed | Spec that pins it | Behaviour change a caller sees |
| --- | --- | --- | --- | --- |
| fr-models-04: `lagIntervalClosure` validated late and untested | fixed | `distribution-dynamics/gpu-spatial-markov.ts:152-157`: validated in the constructor with `validateRequirement`, so a wrong value throws under the contributor's own ID | `distribution-dynamics/gpu-distribution-dynamics.spec.ts`, "GPUSpatialMarkov lagIntervalClosure 'right' keeps a lag equal to a lag edge in the lower lag class": six rows in three pairs, the lag of a row is exactly its partner's value, the pooled median lag is 8 with two lags tied at it; lag values, lag breaks, lag classes and the `2 x 2 x 2` transition counts are by hand for `'right'`, `'left'` and the default, and the constructor refuses another value | A wrong `lagIntervalClosure` now throws at construction instead of in `getCommandNodes` |
| fr-models-09: wrong statement about Python; misstated probability | fixed | The Resolution paragraph above. `test/gpu-spatial-analysis/spatial-validation/spatial-cross-validation-seeds.spec.ts:142-144`: the chance of a repeated design among eight seeds is about `28 / 5040 = 0.0056`, not below 1e-3 (a comment in another stream's spec, edited at the lead's request; the spec is deterministic). The comment in the hand-computed quantile case no longer says no Python package is installed | n/a | None |
| fr-models-09: independent reference for the right-closed rule | added as a second check | New generator `modules/analysis/test/oracles/distribution-class-assignment-numpy.py` and fixture `fixtures/distribution-class-assignment-numpy.json` (NumPy 2.0.2, system CPython 3.9.6; listed in `GENERATORS.md`): `numpy.digitize(values, inner, right=True)` and `right=False` for 400 values on fixed edges (38 of them differ between the closures), and for 40 rows over 3 periods of integer rates with pooled `numpy.percentile` quartile edges `[0, 3, 5, 9, 11]` (26 rows tied at a break), with the transition counts of both closures | `distribution-dynamics/gpu-distribution-dynamics.spec.ts`, "GPUClassAssignment matches numpy.digitize for both closures, on fixed edges and pooled quantiles": classes, quantile edges and `GPUTransitionMatrix` counts equal the fixture exactly for `'right'` and `'left'`. The hand-computed cases stay | None |
| `betweenSumOfSquares` as `total - within` | fixed (it fits the existing per-label pass; no new pass) | `spatial-regionalization/gpu-region-partition-evaluation.ts:660-708`: the label-tile kernel also reads the region sums and the overall column means and accumulates `n_k |mean_k - mean|^2`; the means kernel now runs before it. `:792`: the summary writes that sum, and `pseudoF` uses it. Layout documentation (`:82-89`) and `docs/api/gpu-spatial-analysis/operators-statistics.md` | `spatial-regionalization/gpu-region-partition-evaluation-between.spec.ts` (new): 65,536 rows, two labels whose means differ by 0.0014 against values of +-0.7, so the partition explains 1e-6 of the total; between, within, total and pseudo-F against float64 evaluations of the definitions. The old kernel returned 0 for the between value (true value 0.0321). `spatial-regionalization/gpu-spatial-regionalization.node.spec.ts` pins the new node order | `betweenSumOfSquares` and `pseudoF` are right when the partition explains little; elsewhere they change in the last places. `between` is no longer exactly `total - within` in float32. Also reaches `GPUMaxPRegions`' evaluation. Remaining limit, documented: the accuracy is that of `mean_k - mean`, so centre a table whose overall mean is large against the differences between region means |
