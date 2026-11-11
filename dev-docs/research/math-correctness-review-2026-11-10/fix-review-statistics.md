# Fix review: statistics scope (2026-11-10)

Independent review of the math-fix pass against the snapshot `f5dcdc41`, for
`gpu-spatial-analysis/{spatial-regression, spatial-weights, spatial-autocorrelation,
global-spatial-statistics, permutation-inference, multiple-testing, rate-smoothing,
emerging-hot-spots, scan-statistics, geographical-detector, spatial-entropy, segregation,
categorical-association, outlier-detection}` and all of `dataframe-analysis`, checked against the
Resolution sections of `regression.md`, `spatial-statistics.md` and `dataframe.md`.

Paths are relative to `modules/analysis` unless they start with `docs/` or `dev-docs/`.

Result: no blocker, 1 major, 5 minor. The three precision fixes that were never run against the
old kernels (regression-02, -03, -04) do fail on the old arithmetic: float32 replicas of the old
and new kernels on the spec scenes are in section 3.

Evidence used:

- Reading the diff and each changed kernel in context.
- Pure-Node replicas with `Math.fround` at every operation, in
  `/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-luma-contributors/9972295f-91de-4373-ba95-d2f799611a84/scratchpad/review-statistics/`:
  `restricted.mjs`, `error-ml.mjs`, `panel-lag.mjs`, `probit-overshoot.mjs`, `composite.mjs`,
  `fixture-diff.mjs`.
- Two spec runs on the headless GPU project (both pass):
  `math-review-restricted-regimes.spec.ts` (7 tests) and `statistics-upstream.spec.ts` (33 tests).

## 1. Coverage

Read in full (changed kernel in context, not only the hunk):

- `src/gpu-spatial-analysis/spatial-regression/`: `pairwise-sum-wgsl.ts`,
  `gpu-restricted-regime-regression.ts`, `spatial-regression-solve.ts` (Cholesky and pivot limit),
  `regression-row-tiles.ts` (merge node), `gpu-spatial-panel-lag-maximum-likelihood.ts`
  (`getCommandNodes`), the changed parts of `gpu-probit-regression.ts` with the iteration and
  finish kernels, `gpu-spatial-error-maximum-likelihood.ts` (row function, Gram, first-stage OLS,
  search gate, filtered solve, residual pass).
- `src/gpu-spatial-analysis/global-spatial-statistics/gpu-global-spatial-statistics.ts`: classify,
  islands, value scales, deviate, center, pairs, column lag, moment functions, finalize body.
- `src/gpu-spatial-analysis/spatial-autocorrelation/spatial-autocorrelation-kernels.ts` (moment
  passes), `emerging-hot-spots/gpu-emerging-hot-spots.ts` (diff plus every reader of values and
  deviations), `geographical-detector/gpu-geographical-detector.ts`,
  `spatial-weights/spatial-weights.ts` (neighbour loop).
- `src/dataframe-analysis/pair-statistics/pair-statistics-cell-neighbors.ts` (new),
  `src/analysis-core/kernels/fixed-point-accumulator.ts` (not changed; read because group
  statistics and key join now depend on it).
- Specs: `math-review-restricted-regimes`, `math-review-error-ml-high-fit`,
  `math-review-panel-lag-centering`, `math-review-probit-bound`, `math-review-tile-merge-sums`,
  `math-review-regression-test-utils.ts`.

Read as a diff, with the surrounding lines where a question came up:

- The other eight users of the pairwise sum, `ordinary-sur-nodes.ts`,
  `gpu-spatial-regime-regression.ts`, `gwr-fit-wgsl.ts`, the GTWR and generalized GWR files,
  `gpu-spatial-regression-diagnostics.ts`.
- `spatial-entropy`, `segregation`, `categorical-association`, `rate-smoothing`,
  `spatial-weights/gpu-contiguity-weights.ts`, `spatial-weights/gpu-fuzzy-contiguity.ts`,
  `scan-statistics`, `permutation-inference`.
- All of `dataframe-analysis`: composite score, group statistics, key join, column profile,
  variogram, variogram model fit, marked point statistics, cross variogram, the small
  pair-statistics edits.
- Changed existing specs of `spatial-regression` and `gpu-dataframe` (removed lines only, plus
  the new small-magnitude case of `gpu-group-statistics.spec.ts`), the two pandas generators and
  the regenerated fixture.

Not read:

- New and changed specs of the spatial-statistics stream (only the diff statistics), and
  `math-review-gtwr-aicc`, `math-review-logistic-selection`, `math-review-local-condition-number`,
  `math-review-pivot-limit` beyond its scenes and assertions.
- `local-collinearity-wgsl.ts` (`getDesignSingularValueWGSL`), `segment-bvh.ts`, the cooperative
  variogram refinement, `GPUReduction`.
- `docs/api` pages: searched for stale statements, not read.
- `multiple-testing` and `outlier-detection` have no change. `gpu-additional-spatial-statistics.ts`
  changed in words only.

## 2. Findings

| ID | Severity | Status | File:line | What is wrong | Input or reasoning | Fix I would make |
| --- | --- | --- | --- | --- | --- | --- |
| fr-statistics-01 | major | CONFIRMED (float32 replica, derivation) | `src/gpu-spatial-analysis/spatial-regression/gpu-restricted-regime-regression.ts:228`, `:237`; `dev-docs/research/math-correctness-review-2026-11-10/regression.md:403-406` | regression-02 is not fixed for a common intercept with a varying slope. That predictor gets shift 0, so its moments are rebuilt as `C + n mean^2` in float32, the old arithmetic. The Resolution declares this and says "no shift removes it". That reason is wrong. | 600 rows, 2 regimes, `common: [true, false]`. Elevation 1500 +- 3: status 0, both slopes 3.4% off. Year 2010 +- 10: status 0, 0.28% off. Northing 5.2e6 +- 5000: status 2, although the model is identified. | Solve in the basis `{x - c, x 1[g]}` with `c` the pooled mean. It spans the same model, because `1 = (sum_g x 1[g] - (x - c)) / c`. Its conditioning depends on the spread between regime means, not on `c`. Slopes are `beta_w + beta_g`, the intercept is `-c beta_w` plus the response shift. In float32 from the same regime moments this gives the three scenes to 2e-7 (`restricted.mjs`, last three lines). |
| fr-statistics-02 | minor | CONFIRMED (float32 replica) | `src/dataframe-analysis/composite-indicators/gpu-composite-score.ts:714-748` | dataframe-01 is fixed for clear negative correlation only. For two indicators with a weak negative correlation the kept start is an axis start that has not converged in 64 steps, so the output is neither eigenvector. The class documentation says near-equal eigenvalues are not separated; the Resolution says "fixed". | d = 2. r = -0.05: loadings (0.7083, -0.7059). r = -0.02: (0.760, -0.651). r = -0.01: (0.871, -0.492). r = -0.001: (0.998, -0.064), so the score is almost the first indicator alone. r = +0.01: (0.7071, 0.7071) exactly. The result jumps at r = 0 and depends on the column order. | For d = 2 write the closed form `(1, sign(r)) / sqrt(2)`. In general use the new shared `analysis-core/kernels/symmetric-eigen-wgsl.ts`, which the original finding suggested. |
| fr-statistics-03 | minor | SUSPECTED (evaluation order; not run) | `src/dataframe-analysis/group-statistics/group-statistics-moments.ts:634` | The z-score is `(v - mean) - delta` in one expression. This is the form that `gpu-global-spatial-statistics.ts:1230-1233` and `spatial-autocorrelation-kernels.ts:610-612` avoid by reading a stored deviation back, because the shader compiler of the test GPU may reassociate it to `v - (mean + delta)`. `delta` is at most half a float32 step of `mean`, so the reassociated sum rounds back to `mean` and the correction is lost. | Values 1e7, 1e7 + 1, 1e7 + 2, 1e7 + 3: the float32 mean is 1e7 + 2, `delta` is -0.5, standard deviation 1.29. Without `delta` every z-score is 0.39 off. No spec has such a column (section 4). | Store `v - mean` in the moment pass (it is already formed there) and subtract `delta` from the stored value, as the two spatial kernels do. Add a large-offset scene. |
| fr-statistics-04 | minor | CONFIRMED (float64 replica of the control flow) | `test/gpu-spatial-analysis/spatial-regression/coverage-estimators.spec.ts:611-615` | The replaced probit scene no longer covers what the test was written for. With a slope of 3.0 no step is ever shortened by the bound, so "an overshoot is damped, then the fit converges with status 0" has no test. The assertion itself was not weakened (tolerance 0.3 became 2e-2). | `probit-overshoot.mjs`: slope 3.0 takes nine full steps, largest linear predictor 9.96. Slope 3.5 is shortened from iteration 5 and ends in status 4 at iteration 12, as the Resolution says. | Add a scene where one early step crosses 12 but the maximum does not, for example a start that overshoots through a few high-leverage rows, and assert status 0. |
| fr-statistics-05 | minor | CONFIRMED (arithmetic) | `test/gpu-spatial-analysis/spatial-regression/math-review-pivot-limit.spec.ts:23-24`, `:186`, `:191`; the same scene in `math-review-restricted-regimes.spec.ts` (`createCollinearScene`) | The specs do not pin the limit of 1e-6 and can pass on the old kernels. The singular scene has a squared cosine of 1 - 1e-10, far below float32 rounding. On the unit-diagonal matrix the off-diagonal rounds to 1 or to 1 - 6e-8, so the old pivot is 0 (status 2 already) or 1.2e-7 (status 0) by rounding. The control is at 1 - 1e-2. Any limit between 1e-9 and 1e-2 passes both. | The Resolution itself reports that only one of two regimes returned status 0 on the old kernel. Probit, SUR and panel lag were not seen to fail. | Use spreads of 7e-4 (1 - cos^2 = 5e-7: status 0 on the old kernel for certain, status 2 now) and 2e-3 (4e-6: must fit). |
| fr-statistics-06 | minor | CONFIRMED (read) | `docs/api/gpu-spatial-analysis/operators-regression.md:74`, `:429`, `:552`, `:702`; `docs/api/gpu-spatial-analysis/coverage-statistics.md:37` | The user documentation of `GPUOrdinaryLeastSquares`, `GPUSpatialRegressionDiagnostics`, `GPUSpatialWeightsTraces`, `GPUSpatialErrorGM` and the instrumental-variables covariance still says the tile partials are merged "with Kahan compensation" or "with compensation". The class documentation was updated; these pages were not. | Five lines found by a search for `Kahan` and `compensat`. | Reword to "a fixed pairwise merge". |

### fr-statistics-01 in more detail

- The spec case "restricted regimes with a common intercept" uses an order-1 predictor for the
  varying slope and a large-mean predictor for the common slope. It does not see this.
- The kernel result for the configuration is the old one. The numbers above come from the replica
  of the new kernel (`restricted.mjs`), which agrees with float64 to under 1% of the spec
  tolerance on all five spec scenes and with the GPU run of that spec.
- The limit of `1 - 1e-6` turns the northing case into status 2. Before the pass it was status 0
  with a wrong answer. The Resolution lists this change.
- The Wald tests of that configuration use the same solve.

## 3. Fixes checked and found right

Regression:

- regression-01: decide kernel read. `limitedByBound` is per iteration; status 4 needs a shortened
  step below `tolerance` whose full Fisher step is not. Float64 replica of the control flow on
  both scenes: slope 3.5 ends in status 4, slope 3.0 in status 0. Coefficients are NaN on status 4
  (finish kernel returns before writing them).
- regression-02 (varying intercept, and common intercept with common slopes): shifted normal
  matrix, intercept map, `T V T'`, residuals and contrasts derived by hand.
  - Hand case, 6 rows, 2 regimes, common slope: Chow F = 0.6 (df 1, 2), intercept Wald = 54 / 13 =
    4.153846. The replica of the kernel gives 0.6000001 and 4.153844.
  - Spec scenes, error as a multiple of the spec tolerance on the coefficients: old 2.8 (year),
    1018 (elevation), 3344 (northing), 73 (varying slopes), 2.2 (common intercept); new below 0.01
    on all five. Old Chow F: 0, 66.2 and 0 against 0.0199. So every regression-02 spec case fails
    on the old arithmetic.
  - `statistics-upstream.spec.ts` (spreg covariance and Chow stability tails) passes on the GPU.
- regression-03: replacing the response by its first-stage residual leaves the minimum over the
  coefficients unchanged. Float32 replica of `ee(lambda)` at the maximum, relative error: old
  6.4e-3, 4.7e-2 and 5.0e-3 on the three high-fit scenes (spec tolerance 5e-4), new at most
  1.1e-6. The specs fail on the old arithmetic. The residual pass and R^2 use the raw response.
- regression-04: shift algebra checked: lag packed as `W y - mean(y)`, intercept
  `+ (1 - rho) mean(y) - slopes' mean(x)`, covariance row `[1, -mean(x), -mean(y)]`, fitted
  `+ phi mean(y)`. Float32 replica of the slope at the float64 `(rho, phi)`: old -3.6e-2, +2.4e-1
  and +5.2e-3 (spec tolerance 2e-3), new below 1e-6. The specs fail on the old arithmetic. The
  spreg panel covariance oracle passes on the GPU.
- regression-05, -06: formula read; NaN unless `n - tr - 2 > 0`.
- regression-07: `2 (max(t, 0) + log(1 + exp(-abs(t))))` with `t = -eta` for a response of 1 and
  `eta` for 0 is the Bernoulli deviance and cannot overflow. `fittedScale` is used at the one
  place the fitted value is written.
- regression-10: the limit is the pivot of the unit-diagonal matrix in every caller.
  - On an already scaled matrix: restricted regimes, the two SUR normal systems.
  - On the unscaled matrix against `sqrt(A_ii)`, which is the same pivot: fully varying regimes,
    probit information, SUR error covariance.
  - Through `factorEquilibrated`: panel lag, three sites.
  - Spatial SUR composes OLS, error GM and the ordinary SUR nodes, so it inherits the limit.
- regression-11: weighted rows `w [1, x]`, streaming QR, unit column norms, ratio of singular
  values. The singular-value routine itself was not read.
- regression-15: read.
- regression-16: every call site sums what it summed, with the same base, stride and count.
  - `gpu-ordinary-least-squares.ts` (three merges), `gpu-spatial-error-gm.ts` (seven),
    `gpu-spatial-regression-diagnostics.ts` (two), `gpu-spatial-two-stage-least-squares.ts` (four),
    `gpu-spatial-weights-traces.ts` (three plus the two probe sums),
    `spatial-exact-log-determinant.ts` (four), `spatial-maximum-likelihood-nodes.ts` (three),
    `ordinary-sur-nodes.ts`, `regression-row-tiles.ts`, and `getKahanSumWGSL` (five uses in lag
    ML, four in error ML).
  - The probe sums changed from a batch loop to `item / batchProbeCount`, `item % batchProbeCount`
    over `probeTotal`: the same terms in the same order.
  - Depth: the number of live blocks after `i` terms is the number of one bits of `i`, at most
    32 for any `u32` count, so `array<f32, 32>` cannot overflow and `Depth--` cannot wrap.
  - No two sums of the same name share a scope.
- regression-08, -09, -12, -13, -14, -17: words or spec text; read as a diff only.

Spatial statistics:

- spatial-statistics-01, -02: every statistic of the block is scale free or scaled back.
  - General G: cross term, mean and sum of squares are all in scaled units; variance is divided
    by the denominator twice.
  - Moran and Geary: statistic is a ratio of scaled squares; the moment functions use
    `S1 / S0^2` and `S2 / S0^2`, and I checked each of the six formulas against the old form.
  - Bivariate Moran: the x and y scales cancel; the column lag reads the scaled centered values.
  - Join counts: the black test reads the raw value; the variances use counts only.
  - Kurtosis: `n (m4 / SS) / SS` in both places and for the black moments.
  - Only `summary.mean` and `summary.variance` are multiplied back. The negative-value counter
    reads the raw sign.
  - Binding counts stay at 7 or below in the kernels that gained `magnitudes`.
- spatial-statistics-05: the four loops use the shared rule. Segregation passes `CAPACITY` as
  `0xffffffff` so one pipeline serves every scale. Empirical Bayes passes the chunk row start and
  now needs two kernels for four outputs (5 inputs, a mask, 2 or 3 outputs per kernel).
- spatial-statistics-07: deviations are stored and read back; fixed moments go through the same
  path; emerging hot spots subtracts the residual once per sum.
- spatial-statistics-08: one floor inside `getTwoSidedPValue`; the one-sided permutation value
  takes the maximum with the floor.
- Contiguity snap: at x = 5e6 float32 coordinates are on a 0.5 grid, so `value - origin` is exact
  and the key no longer saturates. One kernel quantises, so one origin is used.
- Geographical detector: later kernels read only the partition outputs, so the unlabelled rows
  cannot return.
- spatial-statistics-03, -04, -06, -09 and the scan uniform: read as a diff only.

Dataframe:

- dataframe-01: replica gives the major component for r = -0.958 and -0.1 at d = 2, for the
  three-indicator matrix whose uniform start is the exact minor eigenvector, and for a block case
  at d = 4. The zero-sum sign rule picks the first of the tied loadings. Weak negative
  correlation: fr-statistics-02.
- dataframe-03: scale pass and accumulate pass use the same predicate; privatised tables are at
  most 8 KiB; the key-join carry chain is the shared one with the high word in its own view; at
  most 7 storage buffers per kernel. Neither `sumValues`, `means` nor the key-join sum reads
  `sumScale`.
- Fixture `parity-wp6-group-statistics.json`: structural diff shows the four `versions` entries
  and 8 values, all in case `unsigned64-carries`, groups 5 and 6, sample and population.
  - Each group holds one row: `0.5 / 65536` and `1.5 / 65536`.
  - `sumValues` and `means` go from 0 and `2 / 65536` (half to even of the words) to the values
    themselves. That is what the new kernel should give.
  - The `sums` words are unchanged (0 and 2).
- dataframe-04: the report's case by hand: difference 2.625, range 15.75, scaled 78.75 = 5 x
  15.75, bin 5. One correction step each way is enough. An infinite range is rejected by the
  domain mode.
- dataframe-05: constant read.
- dataframe-06: divisor `n - 1` only for the same set without intensities; one point fails
  `n - 1 > 0`, so NaN and status 1. The changed assertion in `geospatial-expansion.spec.ts`
  replaces "homogeneous equals inhomogeneous" by two hand values; it is not weaker.
- dataframe-09: weights normalised before the normal equations; the status-2 guard is written in
  the prepare pass and every later kernel runs only on status 0.
- `VARIOGRAM_MODEL_SHAPE_WGSL`: the three series (exponential, Matern 3/2, Matern 5/2) checked
  term by term by hand.
- dataframe-07, -08, -10: words only.

## 4. Tests that do not pin their fix

- dataframe-02 (`gpu-group-statistics.spec.ts`, small-magnitude case): passes with the `- delta`
  line removed. After dataframe-03 the mean is the float32 mean, so `delta` is below 4e-12 for
  1.0e-4, 1.2e-4, 1.4e-4 and changes z by 2e-7 against a tolerance of 1e-4. No scene in the spec
  has a column whose mean is large against its spread. See fr-statistics-03.
- regression-10: fr-statistics-05.
- regression-01, the status-0 path after a shortened step: fr-statistics-04.
- regression-04 covariance: the congruence test compares two runs of the kernel with the same
  response, so the `-mean(y)` entry of the transform cancels. It is pinned only by the spreg
  fixture in `statistics-upstream.spec.ts`.
- regression-02 with a common intercept: the spec scene has the large mean on the common slope
  only. See fr-statistics-01.

## 5. Undeclared behaviour changes

None that changes a number without a line in a Resolution table. Items a caller or the lead
could miss:

- `GPUSpatialErrorMaximumLikelihood`: when the first-stage centered system fails the pivot limit
  of 1e-3 the kernel uses `b0 = 0`, which is the old arithmetic. The filtered solve has no pivot
  limit (bare `choleskyFactor`), so near-collinear predictors can still end in status 0 with the
  old precision. Only a code comment mentions the fallback
  (`gpu-spatial-error-maximum-likelihood.ts:668-669`).
- `GPURestrictedRegimeRegression` `fitted` is still evaluated in the raw basis in float32
  (`:419`). For a northing the intercept is near 2.6e6, so fitted values carry about 0.25 of
  rounding while the residual sums are now exact to the shifted basis. Not mentioned.
- `dataframe.md` says there is no edit inside `analysis-core/cell-key-wgsl.ts`. The file has 66
  new lines with a new export `getGPUCellSumScale`. It comes from another stream
  (`core-numerics.md`); the sentence is true only of the dataframe stream.
- `gpu-fuzzy-contiguity.ts` and the grid paths of `gpu-marked-point-statistics.ts` and
  `gpu-cross-variogram.ts` were restructured by the parallel pass
  (`dev-docs/research/parallel-pass-2026-11-10/`), not by the three streams checked here. I read
  them as diffs and found nothing wrong.
- The regenerated fixture records Python 3.9.6 and pandas 2.2.2 while its sibling fixtures from
  the same generators keep the pinned versions. Declared in `dataframe.md`.
  Closing pass (2026-11-11): regenerated on the pinned set, every value equal; see [fixtures-pinned-rerun.md](./fixtures-pinned-rerun.md).

## Needs a second look

- fr-statistics-03 on the GPU: one scene with values near 1e7 settles it.
- `gpu-restricted-regime-regression.ts:352`: `(observed - shift) - intercept` is the same
  one-expression form. With a varying intercept the shifted intercept is near 0 and nothing can be
  lost. With a common intercept and a response whose mean is large against its residuals, a
  reassociated sum would round by half a step of the response mean.
- The pairwise cascade with a small compile-time count could be unrolled and linearised by the
  compiler. The result would then be the serial sum, which is what the old code gave, never
  worse. The GPU spec covers 1,000 and 4,001 terms only.
- Composite score with more than two indicators: a later start replaces the kept one only above a
  margin of 1e-5, so a first start that is slightly less converged is kept. The loading error is
  bounded by `sqrt(1e-5 lambda1 / (lambda1 - lambda2))`. Not reproduced.
