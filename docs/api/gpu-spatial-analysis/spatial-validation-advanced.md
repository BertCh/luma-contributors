# Spatial fold designs and nested interpolation validation

`GPUSpatialCrossValidation` deals blocks to folds in a seeded random order (rank of a per-block Philox draw, modulo the fold count) when `design` is omitted. The optional design consumes resident covariates and prediction locations; outcomes are never an input to fold construction. All coordinates and exclusion distances use native planar units; an optional `spatialContext` accepts only the default planar context. Reference tier: fold designs are one serial invocation (256 rows; 64 for NNDM) and buffered masks cost O(folds * rows^2) at up to 65,536 rows and 64 folds. Fold-major training/testing masks remain the estimator boundary.

| Design | Executed contract | Bounds and distinctions |
| --- | --- | --- |
| `groups` | Sort distinct valid numeric group IDs, assign rank modulo fold count, preserve every group | At most 256 rows. Matches blockCV 4.0.1 `cv_group(balance=FALSE)` including leave-group-out. Seed does not change fixed group ranks. |
| `environmental` | Complete-row feature scaling, deterministic farthest-first seeds, Lloyd assignment/centroid iterations | At most 256 rows, eight feature columns and 16 clusters; 1–1024 iterations. Standardization uses sample mean/SD of supplied complete rows, not raster-wide moments. Constant columns contribute zero; insufficient distinct centers/empty clusters fail explicitly. blockCV partition comparison uses `cv_cluster(scale=FALSE)` on extracted covariates and compares memberships up to label permutation. R's Hartigan-Wong/random initialization is not replicated. |
| `nndm` | Leave-one-out nearest-neighbor distance-distribution matching with the prediction domain, bounded maximum distance and minimum retained training proportion | At most 64 complete sample rows and 256 prediction locations; fold count equals sample count. Direct blockCV `cv_nndm` radii and masks agree, including strict training distance **greater than** the final exclusion radius. The upstream minimum-proportion rule operates during neighbor deletion; final strict-radius exclusion can retain fewer rows when distances tie. |
| `knndm` | Score every valid resident candidate partition by empirical 1-Wasserstein distance between test-to-training and prediction-to-training nearest distances; choose the lowest score subject to nonempty folds and maximum fold fraction | At most 256 sample/prediction rows and 64 candidates. Omitted candidates generate deterministic seeded farthest-first/Lloyd spatial partitions (up to 16 folds). Caller-supplied partitions can instead come from any resident clustering graph. Direct blockCV hierarchical candidate partitions and `cv_knndm` selected partition/objective are verified. Automatic generation is a bounded Lloyd restart search, not blockCV's hierarchical/overcluster/merge ladder or its KS-test random fallback. |

`design.diagnostics`, when supplied, contains `[status, iterations, objective, selectedCandidate]`; NNDM appends one exclusion radius per source row. Status is 0 success, 1 invalid/incomplete required inputs or nonfinite distance/objective arithmetic, 2 infeasible clustering/partition, or 3 exhausted Lloyd iterations. Unused objective/candidate fields are NaN. Failed designs publish `0xffffffff` folds and empty masks, so existing per-fold status reports empty training/testing. NNDM and kNNDM require all sample and prediction positions to be finite and unmasked; environmental/group designs explicitly omit masked/incomplete rows. Deterministic initialization, seed, candidate order and returned folds provide replay provenance. Candidate ties choose the lowest index. No CPU clustering, distance matching, readback or fitted-outcome reuse occurs.

`GPUSpatialCrossValidatedOLS` evaluates each held-out fold with training-only coefficients.
Its metric status reports bit1 for no scored rows, bit2 for missing outcomes/predictions,
bit4 for undefined R² and bit8 for nonfinite metric arithmetic. Arithmetic failure preserves
scored/missing counts and returns NaN for all four aggregate metrics. Every fold-design
input and diagnostic is included in the recipe's input/output alias checks.

## Nested variogram and bandwidth selection

`GPUNestedInterpolationValidation` takes explicit outer `training`/`testing` masks, inner block IDs and a finite candidate ladder. Each candidate is either an ordinary-kriging covariance/variogram specification (`model`, effective `range`, `sill`, `nugget`) or an RBF kernel with inverse bandwidth `epsilon` and smoothing. This is discrete predictive hyperparameter selection: it does not run an unbounded continuous optimizer or silently fit an empirical variogram on the complete dataset.

For every outer fold the graph:

1. Rejects intersecting outer training/testing masks and excludes nonfinite training responses.
2. Constructs inner block folds and optional buffers **inside the outer training mask**.
3. Fits every candidate separately on each inner training mask. Only inner held-out responses enter its score.
4. Selects the lowest complete inner MSE. Any missing inner prediction invalidates that candidate, so missing coverage cannot improve its score.
5. Fits candidates using only outer training observations, publishes the selected model's outer test predictions, and computes held-out coverage, MSE, MAE, signed prediction-minus-observation bias and R².

Changing outer held-out responses changes the outer metrics but leaves that fold's candidate scores, selected index and predictions unchanged. Changed-buffer replay verifies this invariant. Kriging observations are packed with per-fit validity; RBF excluded responses are NaN. Existing local solvers retain rank, capacity and missing-neighborhood failure semantics. An empty inner test fold contributes no observations; all-empty inner validation cannot select a model. No complete candidate yields selected index `0xffffffff` and missing predictions.

Bounds are 256 observations, 2–16 outer folds, 2–8 inner folds, 1–16 candidates and 2–32 local neighbors. Work scales with the product of these counts; a supported bound is not a latency guarantee. Outputs are caller-owned. Prediction rows outside the outer test mask are NaN. Status bit1 marks overlapping outer masks, bit2 no selected candidate, bit4 missing held-out predictions bit8 an empty held-out set and bit16 nonfinite held-out metric arithmetic (the four aggregate metrics become NaN). Undefined R² is NaN when the held-out outcome variance is zero.

The independent reference generator is `modules/analysis/test/oracles/spatial-validation-advanced.R`, pinned to blockCV 4.0.1. `spatial-validation-advanced.spec.ts` compares real native GPU results against that fixture, tests complete test coverage, disjoint masks, invalid/empty/exhausted designs, automatic kNNDM generation and nested changed-input leakage invariance. Independent NumPy 2.2.6 bordered covariance solves and SciPy 1.15.3 RBFInterpolator verify every nested candidate score, chosen model, held-out prediction and metric. Maximal shader options have their own kernel-audit cases.

{/* lens:GPUNestedInterpolationValidation */}

::lens{scenario=gpu-spatial-analysis.nested-interpolation-validation rung=result caption="Two outer folds select between supplied kriging and RBF models using training-only inner folds."}

::lens{scenario=gpu-spatial-analysis.nested-interpolation-validation rung=graph caption="Inspect the separate inner masks, fits, candidate scores, outer refits and held-out metrics."}

## Conformal prediction intervals

`GPUConformalPredictionIntervals` and `GPUGeoConformalPredictionIntervals` turn held-out predictions into prediction intervals with a coverage guarantee, without a distributional assumption. The calibration scores are `R_i = |y_i - prediction_i| / scale_i`, where `scale_i` is optional (for example the kriging standard deviation). The held-out predictions come from the fold-major `predictions` of `GPUSpatialCrossValidatedOLS`, or from one holdout model. Pass the fold of each row (`GPUSpatialCrossValidation` `output.folds`) as `calibration.folds`; a row with an unassigned fold, a non-finite value or a false mask is excluded.

| Contributor | Interval | Guarantee under exchangeability |
| --- | --- | --- |
| `GPUConformalPredictionIntervals`, `foldCount` 1 | Split conformal: `p -+ s R_(r)` with `r = ceil((1 - alpha)(n + 1))` | Coverage at least `1 - alpha` and at most `1 - alpha + 1/(n + 1)` |
| `GPUConformalPredictionIntervals`, `foldCount` above 1 | CV+: lower is the `floor(alpha (n + 1))`-th smallest of `p_k(i) - s R_i`, upper the `ceil((1 - alpha)(n + 1))`-th smallest of `p_k(i) + s R_i` (Barber et al. 2021) | Coverage at least `1 - 2 alpha`; in practice close to `1 - alpha` |
| `GPUGeoConformalPredictionIntervals` | Weighted quantile of the neighbouring scores with the query's own weight as a point at infinity (Lou, Luo and Meng 2024, on Tibshirani et al. 2019) | Approximate; local when the weights reflect the exchangeability structure |

### `GPUConformalPredictionIntervals`

`alpha` is a per-frame parameter: write `getGPUConformalPredictionParameterValues({miscoverage, selfWeight})` into a `GPUParameterBuffer` between encodings to change it without recompiling. Calibration scores are sorted once by (fold, score) with the shared stable radix sort. Each query then finds its exact order statistic by a 32-step bisection over the float32 keys of the candidate values, with one binary search per fold, so no loop runs over the calibration rows and a query costs `32 x folds x log n` comparisons. A rank outside `[1, n]` gives an unbounded side (`-Infinity` or `Infinity`) with a status bit instead of a clamped value.

The ranks use a relative slack of 2^-20 so that a product that is an integer in exact arithmetic rounds to that integer: without it, float32 `(1 - 0.4) x 25` evaluates to 15.000001 and the rank becomes 16. Products within 2^-20 relative of an integer are treated as that integer. Interval arithmetic is float32 (`p + s R`), so with a non-power-of-two `scale` the bound can differ from a binary64 evaluation by one ulp.

`status` per query has bit1 for an unbounded lower side, bit2 for an unbounded upper side, bit4 for a query prediction or scale that is not a finite positive number (the interval is NaN), bit8 for no usable calibration row, and bit16 (localized only) when a query has more than 256 neighbour entries and the tail was dropped. `output.calibrationCount` holds the number of usable calibration rows.

{/* parameters: GPUConformalPredictionIntervals generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUConformalPredictionIntervals` parameters

- Packer: `getGPUConformalPredictionParameterValues(settings)`, writes `GPU_CONFORMAL_PREDICTION_PARAMETER_LENGTH` (4) `float32` values.

Settings of `getGPUConformalPredictionParameterValues`:

| Field | Type | Description |
|---|---|---|
| `miscoverage` | `number` | Target miscoverage `alpha` in the open interval (0, 1): intervals cover at least `1 - alpha`. |
| `selfWeight` (optional) | `number` | Localized intervals only: weight of the query location itself in the weighted quantile, the kernel value at distance zero. Default 1, right for kernels scaled so that `K(0) = 1`. |

{/* /parameters */}

{/* lens:GPUConformalPredictionIntervals */}

### `GPUGeoConformalPredictionIntervals`

The localized contributor takes a compressed-sparse-row neighbour list per query (`offsets`, `indices`, `weights`; at most 256 entries are used per query) with kernel weights from any neighbour search, and uses one model (`foldCount` 1). An entry that is out of range, excluded or has a zero, negative or non-finite weight is skipped. With equal weights equal to `selfWeight` the result is exactly the global split interval, which the specs use as an independent equivalence check. Weights are summed in float32 in list order, so with non-uniform weights the quantile level is accurate to float32 rounding only.

{/* lens:GPUGeoConformalPredictionIntervals */}

### Conformal verification

`conformal-prediction-intervals.spec.ts` checks real GPU results against a brute-force reference that sorts every candidate value and takes the rank from exact rational arithmetic (split, CV+ with 64 folds, scales, masks, non-finite inputs, unbounded sides, the exact-integer rank cases, and the localized quantile with dyadic weights). Coverage is verified by simulation with seeded replicates of one compiled graph. For split conformal the calibration coverage is Beta(r, n + 1 - r), so a replicate covering `m` test points has mean `r/(n + 1)` and variance `p(1 - p)/m + (1 - 1/m) Var(C)`; the spec requires the mean over 1000 replicates to lie within 4.5 standard errors of that mean for `n` 99 with `alpha` 0.1 and `n` 19 with `alpha` 0.05. A heteroscedastic case (noise standard deviation 0.2 on the left half and 2 on the right) shows global intervals covering the right half only about 80 percent while the localized intervals reach 90 percent in both halves.

Reference: the external oracle (the authors' GeoConformal code) is pending; the reference used is the binary64 order-statistic implementation above. Lou, Luo and Meng (2024) is cited for the definition only.

## Area of applicability

`GPUAreaOfApplicability` computes the dissimilarity index (DI) and the area of applicability (AOA) of Meyer and Pebesma (2021), following the definition of the R package CAST 1.2.0 (GPL: used as a definition and test oracle only, no code is copied). Predictors are standardized with the sample mean and standard deviation (`n - 1`) of the usable training rows and multiplied by optional variable-importance weights. The index of a point is its Euclidean distance to the nearest training point divided by `meanDistance`, the mean over training rows of their mean distance to all other training rows. With `crossValidation` (the `folds` and fold-major `training` masks of `GPUSpatialCrossValidation`) a training point's nearest neighbour is searched among the training rows of the fold in which it is held out, so a spatial buffer is honored; without it every other row is a candidate. The threshold is `min(Q3 + 1.5 IQR, max)` of the training indices, with type 7 quartiles as in R `quantile`, and a query row is inside the area when its index is at most the threshold. `summary` holds `[meanDistance, threshold, usableTrainingRows, scoredTrainingRows, Q1, Q3]`; `status` has bit1 for fewer than two usable rows, bit2 for a predictor without variance and bit4 for no scored training row.

Differences from CAST: Euclidean distance only; categorical predictors must be supplied as dummy columns; a training row held out in no fold or without a neighbour is left out of the threshold (CAST would give it an infinite index); rows with non-finite predictors are unusable. The distances are exhaustive and in float32: the training pass costs `O(rows^2 x predictors)` (at most 65,536 rows and 16 predictors) and each query row `O(rows x predictors)`. `area-of-applicability.spec.ts` compares real GPU results against a binary64 implementation of the CAST definition written independently of the kernels (with and without folds, buffered masks, weights, a row mask and non-finite entries), and checks two hand-computed examples, one of which exercises the maximum cap of the threshold. The R oracle is pending: R and CAST are not installed in the test environment.

{/* lens:GPUAreaOfApplicability */}

## Measured execution envelopes

The native Chromium WebGPU run recorded ten correctness-checked workloads on the Apple Metal-3 adapter. Timings include CPU encode, submit and queue completion. Warm medians use five repetitions after the first execution; source hashes, upload, compile, first-run, readback, three changed-input repetitions and caller/transient bytes are recorded in `dev-docs/research/evidence/spatial-validation-2026-10-23.json`. Shared device caches were not cleared; these observations are not latency guarantees or CPU speedup claims. GPU timestamps were unavailable for these runs.

| Workload | Warm median ms | First execution ms | Physical transient bytes |
| --- | ---: | ---: | ---: |
| validation-groups-64 | 2.600 | 53.900 | 0 |
| validation-groups-256 | 5.200 | 46.400 | 0 |
| validation-environmental-64 | 0.400 | 3.800 | 0 |
| validation-environmental-256 | 0.700 | 4.100 | 0 |
| validation-nndm-16 | 0.500 | 6.300 | 1024 |
| validation-nndm-64 | 9.200 | 23.600 | 16384 |
| validation-knndm-64 | 3.400 | 50.600 | 1040 |
| validation-knndm-256 | 42.400 | 148.600 | 4112 |
| nested-interpolation-12 | 0.600 | 28.100 | 1584 |
| nested-interpolation-256 | 2.100 | 270.900 | 33792 |

The 256-row nested measurement uses two outer folds, two inner folds, two candidates and four neighbors. Increasing these axes multiplies the fit count; maximal per-kernel capacities are separately audited, and the combined maximum graph is not a measured interactive workload. Automatic kNNDM measurements use four deterministic clustering candidates. Every workload validates disjoint masks or constant-function prediction and repeats after changing resident input values.
