# Spatial fold designs and nested interpolation validation

`GPUSpatialCrossValidation` retains its existing seeded block-hash assignment when `design` is omitted. The optional design consumes resident covariates and prediction locations; outcomes are never an input to fold construction. All coordinates and exclusion distances use native planar units. Fold-major training/testing masks remain the estimator boundary.

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

The independent reference generator is `modules/graph/test/oracles/spatial-validation-advanced.R`, pinned to blockCV 4.0.1. `spatial-validation-advanced.spec.ts` compares real native GPU results against that fixture, tests complete test coverage, disjoint masks, invalid/empty/exhausted designs, automatic kNNDM generation and nested changed-input leakage invariance. Independent NumPy 2.2.6 bordered covariance solves and SciPy 1.15.3 RBFInterpolator verify every nested candidate score, chosen model, held-out prediction and metric. Maximal shader options have their own kernel-audit cases.

{/* lens:GPUNestedInterpolationValidation */}

::lens{scenario=gpu-spatial-analysis.nested-interpolation-validation rung=result caption="Two outer folds select between supplied kriging and RBF models using training-only inner folds."}

::lens{scenario=gpu-spatial-analysis.nested-interpolation-validation rung=graph caption="Inspect the separate inner masks, fits, candidate scores, outer refits and held-out metrics."}

## Measured execution envelopes

The native Chromium WebGPU run recorded ten correctness-checked workloads on the Apple Metal-3 adapter. Timings include CPU encode, submit and queue completion. Warm medians use five repetitions after the first execution; source hashes, upload, compile, first-run, readback, three changed-input repetitions and caller/transient bytes are recorded in `dev-docs/perf/spatial-validation-2026-10-23.json`. Shared device caches were not cleared; these observations are not latency guarantees or CPU speedup claims. GPU timestamps were unavailable for these runs.

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
