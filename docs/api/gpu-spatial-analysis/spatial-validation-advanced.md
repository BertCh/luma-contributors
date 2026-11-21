# Spatial fold designs and nested interpolation validation

Import from `@luma-contributors/analysis/gpu-spatial-analysis`.

`GPUSpatialCrossValidation` deals blocks to folds in a seeded random order (rank of a per-block Philox draw, modulo the fold count) when `design` is omitted. The optional design consumes resident covariates and prediction locations; outcomes are never an input to fold construction. All coordinates and exclusion distances use native planar units; an optional `spatialContext` accepts only the default planar context. Reference tier: fold designs are one serial invocation (256 rows; 64 for NNDM) and buffered masks cost O(folds * rows^2) at up to 65,536 rows and 64 folds. `bufferDistance` is finite and in [0, 1e18] native planar units on every GPU route. Fold-major training/testing masks remain the estimator boundary. Supplied `design.mode` accepts only `groups`, `environmental`, `nndm` or `knndm`; omit `design` for seeded block folds.

| Design | Executed contract | Bounds and distinctions |
| --- | --- | --- |
| `groups` | Sort distinct valid numeric group IDs, assign rank modulo fold count, preserve every group | At most 256 rows. Matches blockCV 4.0.1 `cv_group(balance=FALSE)` including leave-group-out. Seed does not change fixed group ranks. |
| `environmental` | Complete-row feature scaling, deterministic farthest-first seeds, Lloyd assignment/centroid iterations | At most 256 rows, eight feature columns and 16 clusters; 1–1024 iterations. Standardization uses sample mean/SD of supplied complete rows, not raster-wide moments. Constant columns contribute zero; insufficient distinct centers/empty clusters fail explicitly. blockCV partition comparison uses `cv_cluster(scale=FALSE)` on extracted covariates and compares memberships up to label permutation. R's Hartigan-Wong/random initialization is not replicated. |
| `nndm` | Leave-one-out nearest-neighbor distance-distribution matching with the prediction domain, bounded maximum distance and minimum retained training proportion | At most 64 complete sample rows and 256 prediction locations; fold count equals sample count. `minimumTrainingFraction` must be in `(0, 1)`. Direct blockCV `cv_nndm` radii and masks agree, including strict training distance **greater than** the final boundary defined by the retained neighbor. The upstream minimum-proportion rule operates during neighbor deletion; final boundary exclusion can retain fewer rows when distances tie. |
| `knndm` | Score every valid resident candidate partition by empirical 1-Wasserstein distance between test-to-training and prediction-to-training nearest distances; choose the lowest score subject to nonempty folds and maximum fold fraction | At most 256 sample/prediction rows and 64 candidates. `maximumFoldFraction` must be in `(1 / foldCount, 1)`. Omitted candidates generate deterministic seeded farthest-first/Lloyd spatial partitions (up to 16 folds). Caller-supplied partitions can instead come from any resident clustering graph. Direct blockCV hierarchical candidate partitions and `cv_knndm` selected partition/objective are verified. Automatic generation is a bounded Lloyd restart search, not blockCV's hierarchical/overcluster/merge ladder or its KS-test random fallback. |

`design.diagnostics`, when supplied, contains `[status, iterations, objective, selectedCandidate]`; NNDM appends one exclusion radius per source row. Status is 0 success, 1 invalid/incomplete required inputs or nonfinite distance/objective arithmetic, 2 infeasible clustering/partition, or 3 exhausted Lloyd iterations. Unused objective/candidate fields are NaN. Failed designs publish `0xffffffff` folds and empty masks, so existing per-fold status reports empty training/testing. NNDM and kNNDM require all sample and prediction positions to be finite and unmasked; environmental/group designs explicitly omit masked/incomplete rows. Deterministic initialization, seed, candidate order and returned folds provide replay provenance. Candidate ties choose the lowest index. No CPU clustering, distance matching, readback or fitted-outcome reuse occurs.

The CPU twin honors an explicit `design.maximumIterations`. An exhausted environmental
budget publishes design status 3, missing folds, empty masks and per-fold status 3; it does not
continue to convergence. When omitted, CPU environmental clustering and generated kNNDM
restarts retain their original convergence default. GPU clustering defaults to 100 steps.
The CPU kNNDM route accepts at most 16,777,216 candidates. CPU fold masks, fold counts,
environmental feature tables and kNNDM candidate tables each cap at 4,294,967,295 entries.
The GPU NNDM route accepts `maximumDistance` in [0, 1e18] native planar units.

Buffer membership on both CPU and GPU is inclusive Euclidean distance on the stored float32
coordinates: a training row at distance at most `bufferDistance` from a testing row is excluded.
Clear comparisons use a fast path; ambiguous boundaries use a residual-preserving exact
comparison of squared coordinate differences. For `(0,0)` and `(1,2^-13)`, the exact squared
distance is `1 + 2^-26`, strictly outside buffer 1, even though the displayed float32 distance
is 1. With one row in each fold, training is `[0,1,1,0]` and fold status is `[0,0]`.

NNDM independently compares each candidate training row against the neighbor defining its
matched boundary, using the stored coordinates rather than the rounded diagnostic radius.
Initial neighbor ordering, replacement after deletion and final neighbor recovery also resolve
ambiguous rounded-distance ties from the stored coordinates. For `(0,0)`, `(1,2^-13)`, `(1,0)`,
the last row defines the first row's boundary even though both displayed distances are 1;
with zero maximum matching distance and buffer, fold 0 retains training `[0,1,0]`.
It also applies the configured buffer comparison. A row is excluded if either inclusive
comparison holds; retained training rows are strictly beyond both boundaries. The explicit
buffer remains a minimum separation when matching stops earlier. Diagnostic radii report the
matched distances for inspection; using those displayed radii to rebuild masks can change
boundary membership. The defining neighbor is excluded even if its displayed radius rounds down.

Environmental GPU clustering materializes a lossless feature shift in a separate GPU pass
before moment and centroid accumulation, including when `standardize: false`. It tries the
first usable value and then the minimum as origins, checking a bit-exact rounded round trip
for every usable value. If neither shift preserves the packed inputs, it copies raw coordinates
instead; centering must not merge distinct groups. The Lloyd kernel reads only this materialized
matrix, keeping any removed translation outside its centroid arithmetic. Centroid updates sum
rounded deviations from the current centroid, including on the raw-coordinate fallback.
Each centroid retains a fixed cluster-local origin and a separate residual: seeding and
assignment subtract the origin, then the residual, before dividing by the sample SD. Updates
accumulate unscaled deviations into the residual rather than rounding a fractional update into
the origin. Distances are compared as scaled Euclidean norms before they are squared for the
objective, so a representable nonzero separation does not vanish merely because its square
underflows. Thus standardization
does not merge the distinct constant groups `[3,a,b,3,3,b,b,b]`, with `a = f32(0.1)` and
`b = a + 2^-25`: seed 0 yields folds `[0,1,2,0,0,2,2,2]`, design status 0 and zero objective.
For `u=2^-23`, `[0,1,1+u,1+2u,1+2u,1+2u,1+4u]` retains folds `[0,2,2,1,1,1,1]`;
the two nonzero centroids are `1+0.5u` and `1+2.5u`, with unstandardized SSE `3.5u²`.
For `h=2^-78`, `[0,h,1,0,h,1]` retains `[0,2,1,0,2,1]` and zero objective.
Constant groups therefore have exactly zero updates even when float32 division uses an
approximate reciprocal; repeated raw-coordinate sums could otherwise erase a distinct group.
Disabling standardization
removes scale normalization, rather than the numerical centering. Each design reserves
`rows * featureCount` float32 scratch values (two columns per generated kNNDM restart).
Originally unusable rows remain excluded. The raw fallback retains finite rows whose subtraction
would overflow; nonfinite fitting distances still report arithmetic failure rather than dropping rows.

The fold contributor, both CPU/GPU named operations and the pure CPU function capture their declared configuration
before validation. Each inherited getter is read once. Design options, output maps and spatial
context/frame metadata retain their construction values through later lowering and execution;
changing the caller's objects does not change the folds or redirect outputs. Input GPU buffers
and CPU input rows remain per-execution data.

`GPUSpatialCrossValidatedOLS` evaluates each held-out fold with training-only centers, slopes and
the correction for rounded means. Its fold-major predictions use the same centered primitive as
OLS fitted values, preserving response-level detail with large coordinate-like predictors. The
published fold coefficients retain the raw intercept representation.
Each fit accepts 1–15 predictors and inherits its fold design's row and fold limits.
The recipe captures fold counts, design/context metadata and nested input/output view maps at
construction. Editing the caller's configuration later does not change the validated graph;
write input buffers to change per-frame data, or construct a new recipe to change topology.
Its metric status reports bit1 for no scored rows, bit2 for missing outcomes/predictions,
bit4 for undefined R² and bit8 for nonfinite metric arithmetic. Arithmetic failure preserves
scored/missing counts and returns NaN for all four aggregate metrics. Every fold-design
input and diagnostic is included in the recipe's input/output alias checks.
Held-out R² uses the variance of scored observations, accumulated as deviations from the first
scored observation. Translating representable outcomes and predictions together therefore retains
the same centered variance and metrics, including when the raw mean loses a half-step in float32.

## Nested variogram and bandwidth selection

`GPUNestedInterpolationValidation`: omitted mode and `mode: 'tiled'` support at most 4,096
observations total; `mode: 'reference'` supports at most 256 observations total. Both routes
support 2–32 local neighbors per fit and retain the fold, candidate and device limits below.

`GPUNestedInterpolationValidation` takes explicit outer `training`/`testing` masks, inner block IDs and a finite candidate ladder. Each candidate is either an ordinary-kriging covariance/variogram specification (`model`, effective `range`, `sill`, `nugget`) or an RBF kernel with inverse bandwidth `epsilon` and smoothing. This is discrete predictive hyperparameter selection: it does not run an unbounded continuous optimizer or silently fit an empirical variogram on the complete dataset.

The contributor, named CPU/GPU operations and pure CPU function capture declared settings before
validation. Candidate arrays, candidate options, covariance maps and output maps are copied and
frozen; inherited getters are read once. Later edits to the caller's settings cannot change the
selected model, its topology or its output destinations. Views retain their identities and input
row/buffer contents remain live for subsequent executions.

For every outer fold the graph:

1. Rejects intersecting outer training/testing masks and excludes nonfinite training responses.
2. Constructs inner block folds and optional buffers **inside the outer training mask**.
3. Fits every candidate separately on each inner training mask. Only inner held-out responses enter its score.
4. Selects the lowest complete inner MSE. Any missing inner prediction invalidates that candidate, so missing coverage cannot improve its score.
5. Fits candidates using only outer training observations, publishes the selected model's outer test predictions, and computes held-out coverage, MSE, MAE, signed prediction-minus-observation bias and R².

Changing outer held-out responses changes the outer metrics but leaves that fold's candidate scores, selected index and predictions unchanged. Changed-buffer replay verifies this invariant. Kriging observations are packed with per-fit validity; RBF excluded responses are NaN. Existing local solvers retain rank, capacity and missing-neighborhood failure semantics. An empty inner test fold contributes no observations; all-empty inner validation cannot select a model. No complete candidate yields selected index `0xffffffff` and missing predictions.

Held-out outcome variance is accumulated around an observed origin in each tile and merged in
centered coordinates. This retains representable response differences when the outcome offset
is large: observations `[10000000, 10000001]` with predictions `[10000001, 10000002]` have
MSE 1 and R² −3, just as their copies shifted down by 10000000 do.

The single-invocation selection scans at most 16 candidate scores per outer fold. The metric merge
scans at most 64 tile partials per outer fold: the 4,096-observation total limit and tile width of
at least 64 rows on multi-tile default/tiled routes enforce that bound (the final tile can have
fewer live rows). Omitted mode retains one tile for at most 256 observations; explicit tiled mode
uses the tile layout even for small inputs. Reference mode merges one tile of at most 256 rows. These bounded loops
retain their reviewed contract allowances at their source call sites.

Bounds are 4,096 observations total by default/tiled (256 total with `mode: 'reference'`), 2–16 outer folds, 2–8 inner folds per outer fold, 1–16 candidates per ladder and 2–32 local neighbors per fit. `innerBufferDistance` is finite and in [0, 1e18] native planar units on both routes. Work scales with the product of these counts; a supported bound is not a latency guarantee. Outputs are caller-owned. Prediction rows outside the outer test mask are NaN. Status bit1 marks overlapping outer masks, bit2 no selected candidate, bit4 missing held-out predictions bit8 an empty held-out set and bit16 nonfinite held-out metric arithmetic (the four aggregate metrics become NaN). Undefined R² is NaN when the held-out outcome variance is zero.

The independent reference generator is `modules/analysis/test/oracles/spatial-validation-advanced.R`, pinned to blockCV 4.0.1. `spatial-validation-advanced.spec.ts` compares real native GPU results against that fixture, tests complete test coverage, disjoint masks, invalid/empty/exhausted designs, automatic kNNDM generation and nested changed-input leakage invariance. Independent NumPy 2.2.6 bordered covariance solves and SciPy 1.15.3 RBFInterpolator verify every nested candidate score, chosen model, held-out prediction and metric. Maximal shader options have their own kernel-audit cases.

The resident graph is limited to 32,768 estimated command nodes. The CPU nested-search route
accepts at most 16,777,216 observations so count outputs stay exact. CPU prediction,
inner-mask, candidate-score and metric array lengths each cap at 4,294,967,295.

{/* lens:GPUNestedInterpolationValidation */}

::lens{scenario=gpu-spatial-analysis.nested-interpolation-validation rung=result caption="Two outer folds select between supplied kriging and RBF models using training-only inner folds."}

::lens{scenario=gpu-spatial-analysis.nested-interpolation-validation rung=graph caption="Inspect the separate inner masks, fits, candidate scores, outer refits and held-out metrics."}

## Conformal prediction intervals

Both conformal contributors read declared construction fields once, including inherited getters,
and capture the calibration, query, neighbor and output maps. Replacing fields on the caller's
configuration after construction does not redirect the graph. GPU views retain their identity:
write their buffers and the parameter buffer to update each frame's scores and intervals.

`GPUConformalPredictionIntervals` and `GPUGeoConformalPredictionIntervals` compute intervals from held-out predictions. Coverage claims require the statistical assumptions below; the contributors cannot verify them from numerical views. The calibration scores are `R_i = |y_i - prediction_i| / scale_i`, where `scale_i` is optional (for example the kriging standard deviation). The held-out predictions come from the fold-major `predictions` of `GPUSpatialCrossValidatedOLS`, or from one holdout model. Pass the fold of each row (`GPUSpatialCrossValidation` `output.folds`) as `calibration.folds`; a row with an unassigned fold, a non-finite value or a false mask is excluded.

| Contributor | Interval | Coverage assumptions and limits |
| --- | --- | --- |
| `GPUConformalPredictionIntervals`, `foldCount` 1 | Split conformal: `p -+ s R_(r)` with `r = ceil((1 - alpha)(n + 1))` | A common scoring rule fitted independently of calibration/test rows, and exchangeable scores: at least `1 - alpha`; at most `1 - alpha + 1/(n + 1)` with almost surely distinct scores |
| `GPUConformalPredictionIntervals`, `foldCount` above 1 | CV+: lower is the `floor(alpha (n + 1))`-th smallest of `p_k(i) - s R_i`, upper the `ceil((1 - alpha)(n + 1))`-th smallest of `p_k(i) + s R_i` (Barber et al. 2021) | The CV+ theorem requires balanced, data-independent folds and permutation-invariant fitting on each training complement; the finite-sample bound depends on calibration size and fold count. No universal bound for arbitrary group/spatial folds. |
| `GPUGeoConformalPredictionIntervals` | Weighted quantile of the neighbouring scores with the query's own weight as a point at infinity (Lou, Luo and Meng 2024, on Tibshirani et al. 2019) | Approximate; local when the weights reflect the exchangeability structure |

### `GPUConformalPredictionIntervals`

The global route accepts 1–64 folds, subject to the device binding and sort limits.

The CV+ finite-sample result is Barber et al. (2021), Theorem 4; an unqualified `1 - 2 alpha`
bound omits both its assumptions and fold-count correction. Spatial/group membership derived
from observations generally violates the data-independent fold design. Such callers can report
the computed intervals and measured held-out coverage; they need a separate justified theorem
or evaluation design to claim coverage for new observations. Exchangeable row order alone is
insufficient. For example, take 32 zero outcomes in eight groups of four, hold out one query,
and fit leave-group-out models predicting 1 for a group represented in training, 0 otherwise.
Every calibration residual is zero, while each query has three zero and 28 unit candidates.
At alpha 1/4, ranks 8 and 24 give [1,1], covering 0 of the 32 possible zero outcomes.
The exhaustive enumeration is in conformal-validation-audit.node.spec.ts; the device spec
checks the same endpoint calculation. Endpoints and ranks are unchanged by this clarification.

Ties can increase split-conformal coverage above the stated distinct-score upper bound. For example, seven calibration scores equal to one, prediction zero and `alpha = 0.25` give `[-1, 1]` with status 0. An exchangeable population always equal to one has coverage 1, exceeding `1 - alpha + 1/(n + 1) = 0.875`; the lower coverage guarantee still holds.

`alpha` is a per-frame parameter: write `getGPUConformalPredictionParameterValues({miscoverage, selfWeight})` into a `GPUParameterBuffer` between encodings to change it without recompiling. Calibration scores are sorted once by (fold, score) with the shared stable radix sort. Each query then finds its exact order statistic by a 32-step bisection over the float32 keys of the candidate values, with one binary search per fold, so no loop runs over the calibration rows and a query costs `32 x folds x log n` comparisons. A rank outside `[1, n]` gives an unbounded side (`-Infinity` or `Infinity`) with a status bit instead of a clamped value.

The ranks are exact integers for the packed float32 `alpha`: the shader multiplies its significand by the integer `n + 1` using 64-bit integer words, then shifts to obtain `lowerRank = floor(alpha (n + 1))` and uses `upperRank = n + 1 - lowerRank`. There is no rank slack or snapping to a nearby decimal. For example, `alpha = 0.1`, `n = 99` gives ranks 10 / 90; `alpha = 0.7`, `n = 9` gives 6 / 4 because float32 0.7 is slightly below 7/10. The coverage statements refer to the packed alpha. At `n = 8,388,607`, `alpha = 0.25`, the exact ranks are 2,097,152 / 6,291,456. Use `getConformalPredictionRanks(n, alpha)` to inspect ranks on the CPU without allocating calibration rows. The global contributor accepts at most `GPU_CONFORMAL_MAXIMUM_CALIBRATION_COUNT` (4,294,967,294) calibration rows so that `n + 1` fits uint32; storage and sort limits can impose smaller bounds. Localized intervals retain a 2^-20 relative tolerance on their approximate float32 weighted-quantile threshold. Interval arithmetic is float32 (`p + s R`), so with a non-power-of-two `scale` the bound can differ from a binary64 evaluation by one ulp.

CV+ selects among computed candidates using ordered integer float32 keys, with both signed zeros
treated as the same tie and published as positive zero. Integer comparisons keep the search exact
at zero even on devices that flush subnormal float comparisons to zero.

Localized weighted quantiles preserve positive finite weight bits during entry filtering,
including positive subnormals. All neighbor and self masses are scaled by 2^-64 when their
maximum is at least 2^64, or by 2^128 when that maximum is positive and below 2^-64.
Subnormal inputs are decoded from their integer significands before arithmetic, so uniformly
tiny weights become normal values without a device-dependent flush to zero. These common
power-of-two factors prevent overflow and preserve uniformly small masses. Equal-weight ranks
still use the exact integer path; ordinary masses retain their arithmetic. Extremely small
relative masses can still underflow in float32, and the weighted-threshold tolerance still applies.

`status` per query has bit1 for an unbounded lower side, bit2 for an unbounded upper side, bit4 for a nonfinite query prediction or a scale that is not finite and positive (the interval is NaN), bit8 for no usable calibration row, and bit16 (localized only) when a query has more than 256 neighbour entries and the tail was dropped. `output.calibrationCount` holds the number of usable calibration rows.

For localized intervals, bit16 (`GPU_CONFORMAL_STATUS_NEIGHBORS_TRUNCATED`) means the supplied
CSR row span itself exceeded the internal 256-entry cap; only that span's first 256 entries
were considered. A nonzero `neighbors.status.overflow` or `neighbors.status.candidateOverflow`,
or a supplied zero `neighbors.status.complete` certificate, sets bit32
(`GPU_CONFORMAL_STATUS_INPUT_INCOMPLETE`) and suppresses both endpoints to NaN with the lower/upper
unbounded bits cleared, even for a span
shorter than 256. Both bits can be set. Clearing the loss flags and restoring any supplied
certificate permits recomputation on a later submission. Carry the
search's loss views with its neighbor buffers. `neighbors.status.requiredCount` is refused by
this contract: first verify it fits the neighbor slot capacity, then omit that port and supply
the two loss flags. For `converged`, `iterationLimitReached` or `unresolvedCount`, verify
the neighbor lists are complete before construction and omit those status views. Decode
independent bits with `decodeConformalStatus(numberOrReadback)`,
which accepts a numeric word, `Uint32Array`, `Uint8Array` or `ArrayBuffer`, validates whole-word
length and preserves byte slices. It returns the six independent boolean fields `lowerUnbounded`,
`upperUnbounded`, `invalidQuery`, `noCalibration`, `neighborsTruncated` and `inputIncomplete`.
Neither bit16 nor bit32 establishes complete local coverage.

{/* lens:GPUConformalPredictionIntervals */}

{/* parameters: GPUConformalPredictionIntervals generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUConformalPredictionIntervals` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPUConformalPredictionIntervals`.

- Packer: `getGPUConformalPredictionParameterValues(settings)`, writes `GPU_CONFORMAL_PREDICTION_PARAMETER_LENGTH` (4) `float32` values.
  - Input port: `parameters`.

Settings of `getGPUConformalPredictionParameterValues`:

| Field | Type | Description |
|---|---|---|
| `miscoverage` | `number` | Target miscoverage `alpha` in the open interval (0, 1); coverage depends on the fold, fitting and exchangeability assumptions documented by the contributor. |
| `selfWeight` (optional) | `number` | Localized intervals only: weight of the query location itself in the weighted quantile, the kernel value at distance zero. Default 1, right for kernels scaled so that `K(0) = 1`. |

{/* /parameters */}

### `GPUGeoConformalPredictionIntervals`

The localized contributor takes a compressed-sparse-row neighbour list per query (`offsets`, `indices`, `weights`; at most 256 entries are used per query) with kernel weights from any neighbour search, and uses one model (`foldCount` 1). An entry that is out of range, excluded or has a zero, negative or non-finite weight is skipped. When every usable weight equals the positive `selfWeight`, the quantile uses the global contributor's exact packed-alpha rank over those same usable entries. Both bounds and the unbounded status bits therefore agree, including at decimal boundaries: 49 scores `1..49`, prediction zero and alpha 0.02 produce `[-Infinity, Infinity]` with status 3 for both contributors. This equivalence is independent of the common weight's magnitude; a truncated neighbour list still sets its truncation bit. Weights are summed in float32 in list order, so otherwise the weighted quantile retains its approximate float32 threshold.

Carry the producer's final-output and candidate-loss views with the CSR, as this helper does.
Pass resident inputs/parameters and caller-owned interval outputs in `searchProps` and
`intervalProps`. Search target rows must be the calibration rows in the same order; search query
rows must be the interval query rows. CSR offsets/indices are slot/calibration-row indices,
not caller IDs; weights and `selfWeight` use the same dimensionless kernel scale.

```ts
import {createTransientView, type GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUNeighborSearch,
  GPUGeoConformalPredictionIntervals,
  type GPUNeighborSearchProps,
  type GPUGeoConformalPredictionIntervalsProps
} from '@luma-contributors/analysis/gpu-spatial-analysis';

export function addLocalizedIntervals(
  graph: GPUCommandGraph,
  id: string,
  searchProps: Omit<GPUNeighborSearchProps, 'output'>,
  intervalProps: Omit<GPUGeoConformalPredictionIntervalsProps, 'neighbors'>,
  neighborCapacity: number
) {
  const queryCount = (searchProps.queryPositions ?? searchProps.positions).length;
  const search = {
    offsets: createTransientView(graph, `${id}-offsets`, 'uint32', queryCount + 1),
    neighbors: createTransientView(graph, `${id}-neighbors`, 'uint32', neighborCapacity),
    weights: createTransientView(graph, `${id}-weights`, 'float32', neighborCapacity),
    status: {
      overflow: createTransientView(graph, `${id}-overflow`, 'uint32', 1),
      candidateOverflow: createTransientView(graph, `${id}-candidate-overflow`, 'uint32', 1)
    }
  };
  graph.add(new GPUNeighborSearch({...searchProps, id: `${id}-search`, output: search}));
  graph.add(new GPUGeoConformalPredictionIntervals({
    ...intervalProps,
    id: `${id}-intervals`,
    neighbors: {
      offsets: search.offsets,
      indices: search.neighbors,
      weights: search.weights,
      status: {
        overflow: search.status.overflow,
        candidateOverflow: search.status.candidateOverflow
      }
    }
  }));
  return search;
}
```

This exact search writes a requested candidate-loss flag as zero; passing it preserves the
same wiring contract when using a producer with bounded candidates. The helper does not request
`requiredCount`. For an existing producer that supplies it, complete that producer submission
and resolve any candidate loss first: retry its candidate stage until the requirement is exact,
rather than treating a retained-candidate lower bound as the full requirement. Read the exact
requirement and verify that it fits both index and weight slot capacities **before
constructing** this consumer. Grow the buffers and rerun the producer if needed. Then build a
new neighbors descriptor containing the three CSR views and both loss views, omitting
`requiredCount`; passing even a zero-valued requirement view is rejected. Keep the loss views
attached for later changed-input submissions. A fitting total slot requirement does not remove
the consumer's separate 256-entry limit per query (bit16). Upstream loss (bit32) suppresses the
whole interval. Clearing both producer loss flags and setting any supplied `status.complete`
certificate to one restores ordinary computation on the next submission. Bit16 still reports
any internal row-span truncation.

{/* lens:GPUGeoConformalPredictionIntervals */}

### Conformal verification

`conformal-prediction-intervals.spec.ts` checks real GPU results against a brute-force reference that sorts every candidate value and derives ranks independently by multiplying packed float32 alpha by the small fixture count exactly in binary64 (split, CV+ with 64 folds, scales, masks, non-finite inputs, unbounded sides, the exact-integer rank cases, tied scores, and the localized quantile with dyadic weights). Coverage is verified by simulation with seeded replicates of one compiled graph. For split conformal with continuous scores, calibration coverage is Beta(r, n + 1 - r), so a replicate covering `m` test points has mean `r/(n + 1)` and variance `p(1 - p)/m + (1 - 1/m) Var(C)`; the spec requires the mean over 1000 replicates to lie within 4.5 standard errors of that mean for `n` 99 with `alpha` 0.1 and `n` 19 with `alpha` 0.05. A heteroscedastic case (noise standard deviation 0.2 on the left half and 2 on the right) shows global intervals covering the right half only about 80 percent while the localized intervals reach 90 percent in both halves.

Reference: the external oracle (the authors' GeoConformal code) is pending; the reference used is the binary64 order-statistic implementation above. Lou, Luo and Meng (2024) is cited for the definition only.

## Area of applicability

Accepts at most 65,536 training rows and 16 predictors. The optional cross-validation input
accepts 1–64 folds.

The moment pass centers each predictor around a usable training value before computing
sample variance. Identical packed decimal predictors therefore have exact zero deviation and
publish the constant-predictor/no-threshold status instead of a spurious positive variance.

`GPUAreaOfApplicability` captures declared construction fields once, including inherited getters,
and copies its cross-validation and output maps. Replacing predictor counts, fold metadata or
output views in the caller's objects after construction cannot alter the validated graph.
Training/query/weight/mask views remain live GPU inputs with their original identities.

`GPUAreaOfApplicability` computes the dissimilarity index (DI) and the area of applicability (AOA) of Meyer and Pebesma (2021), following the definition of the R package CAST 1.2.0 (GPL: used as a definition and test oracle only, no code is copied). Predictors are standardized with the sample mean and standard deviation (`n - 1`) of the usable training rows and multiplied by optional variable-importance weights. The index of a point is its Euclidean distance to the nearest training point divided by `meanDistance`, the mean over training rows of their mean distance to all other training rows. With `crossValidation` (the `folds` and fold-major `training` masks of `GPUSpatialCrossValidation`) a training point's nearest neighbour is searched among the training rows of the fold in which it is held out, so a spatial buffer is honored; without it every other row is a candidate. The threshold is `min(Q3 + 1.5 IQR, max)` of the training indices, with type 7 quartiles as in R `quantile`, and a query row is inside the area when its index is at most the threshold. `summary` holds `[meanDistance, threshold, usableTrainingRows, scoredTrainingRows, Q1, Q3]`; `status` has bit1 for fewer than two usable rows, bit2 for a predictor without variance and bit4 for no scored training row.

Differences from CAST: Euclidean distance only; categorical predictors must be supplied as dummy columns; a training row held out in no fold or without a neighbour is left out of the threshold (CAST would give it an infinite index); rows with non-finite predictors are unusable. A query with no finite computed squared distance publishes NaN dissimilarity and applicability 0, including when every squared distance overflows float32. The distances are exhaustive and in float32: the training pass costs `O(rows^2 x predictors)` (at most 65,536 rows and 16 predictors) and each query row `O(rows x predictors)`. `area-of-applicability.spec.ts` compares real GPU results against a binary64 implementation of the CAST definition written independently of the kernels (with and without folds, buffered masks, weights, a row mask and non-finite entries), and checks two hand-computed examples, one of which exercises the maximum cap of the threshold. The R oracle is pending: R and CAST are not installed in the test environment.

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
