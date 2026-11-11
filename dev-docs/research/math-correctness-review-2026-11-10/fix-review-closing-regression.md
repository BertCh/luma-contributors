# Fix review: closing pass, spatial regression (2026-11-11)

Independent review of the "Closing pass (2026-11-11)" of `regression.md`: the changes to
`GPUOrdinaryLeastSquares`, `GPUSpatialLagMaximumLikelihood`, `GPUSpatialTwoStageLeastSquares`,
`GPUSpatialErrorGM` and `GPUProbitRegression`, their three specs, two node-spec edits and two
oracle generators. The reviewer wrote none of that code. The tree is uncommitted, so the current
files were read as they stand.

Paths are relative to `modules/analysis` unless they start with `docs/` or `dev-docs/`.

Result: no blocker, 3 major, 5 minor. The centring algebra, the intercept mapping, the standard
errors and the Anselin-Kelejian and Kelejian-Prucha formulas are right, and the author's numbers
reproduce. The three major findings are inputs outside the author's scenes: two publish wrong
numbers with status 0, one reports separated probit data as status 2 or 3.

Evidence used:

- Reading the five kernels in full, with `rounded-difference-wgsl.ts`, `pairwise-sum-wgsl.ts`,
  the parameter search of `spatial-maximum-likelihood-wgsl.ts`, the pivot-limited Cholesky of
  `spatial-regression-solve.ts` and `ERFC_VALUE_WGSL`.
- Both generators rerun with `node scripts/run-oracle-fixture.mjs <generator> --python <venv314>`:
  both fixtures IDENTICAL.
- Existing specs on the test GPU (headless Chromium, WebGPU, Apple Metal): the three closing
  specs (20 tests), eleven further spec files of the five classes and their composers (140
  tests), six node spec files (40 tests). All pass.
- 134 GPU probe runs in a scratch folder under `test/` (deleted), against float64 references
  from spreg 1.9.1, libpysal 4.15.0, statsmodels 0.15.0, NumPy 2.5.4, SciPy 1.16.3 on the same
  float32 inputs. Probe specs, reference scripts and their JSON are kept in
  `/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-luma-contributors/ba3fe269-f86e-4664-86ec-632935ba200a/scratchpad/review-regression/`
  (`probe_data.py` to `probe_data5.py`, `kp_claim.py`, `gpu-probe-specs/`).

Common scene of the spatial probes unless stated: 600 rows, `x` uniform on `[-2, 2)` from the
author's stream 20261111, noise `e` from the author's stream 77003, reference spreg in float64
on the float32 inputs with the response centre subtracted (the weights of every such scene have
equal row sums, so the shift moves only the intercept).

## Findings

### frc-regression-01. Lag ML: the `rho` search differences three float32 sums, and a high-fit dataset gets a wrong fit with status 0 or status 3

- Severity: major. CONFIRMED on the test GPU.
- Where: `src/gpu-spatial-analysis/spatial-regression/gpu-spatial-lag-maximum-likelihood.ts`
  lines 724 to 734 (`getSumOfSquares`, `getObjective`, `getDerivative`) and 755 to 767; the
  search is `spatial-maximum-likelihood-wgsl.ts` lines 106 to 162.
- Not introduced today. The author's account lists it under "Not verified, or left" ("`rho` is
  within 6e-8, so it was left; a scene with a smaller ratio was not built"). The scenes below
  are that smaller ratio, and the statement does not hold there.
- What is wrong. `s2(rho)` in the search is `e0'e0 - 2 rho e0'e1 + rho^2 e1'e1` in float32. When
  `u'u` is below about 1e-7 of `e0'e0`, the rounding of the three sums is larger than the
  minimum itself, so the quadratic is negative or zero near its vertex. The derivative
  `-n (rho C - B) / s2(rho)` then has a pole where the quadratic crosses zero; the bisection
  converges on the pole instead of the stationary point, or the objective is `log` of a negative
  number and no candidate survives. A NumPy float32 emulation of the three sums (tiles of 64,
  pairwise merge) gives a quadratic of -0.5 to +1.5 at its vertex on the eight scenes below,
  where the float64 minimum is 0.0083 to 0.0106. The published `sigma2` is recomputed from the
  per-row residuals `e0 - rho e1`, so it is right exactly when `rho` is.
- Inputs and values (ring of two neighbours with weights 0.5,
  `y = (I - 0.4 W)^-1 (0.6 c + s x + 0.004 e)` rounded to `2^-9`, noise streams 1001 to 1008;
  reference spreg `ML_Lag(method='full')`):

  | Scene | `u'u / e0'e0` | Status 3 (NaN) | Status 0 and wrong | Right |
  | --- | --- | --- | --- | --- |
  | `s = 200`, `c = 10000` | 2.4e-9 to 3.0e-9 | streams 1001, 1002, 1005 | 1004, 1008 | 1003, 1006, 1007 |
  | `s = 200`, `c = 0` | the same | 1001, 1002, 1004, 1007 | 1008 | 1003, 1005, 1006 |
  | `s = 60`, `c = 10000` | 2.6e-8 to 3.4e-8 | seven of eight | none | 1006 |
  | `s = 20`, `c = 10000`, stream 77003 | 2.8e-7 | | | right (`sigma2` 1.3e-5) |

  Stream 1008, `c = 10000`: `rho` 0.399862 against 0.4000002 (155 standard errors), `sigma2`
  41 times spreg's (7.2e-4 against 1.76e-5), log-likelihood 1114 units low, standard errors 6.4
  times too large; status 0, not at the boundary, variance defined. Stream 1008, `c = 0`:
  `sigma2` 19.9 times spreg's. Stream 1004, `c = 10000`: `rho` 4.5 standard errors off, `sigma2`
  3.3% high, log-likelihood 9.8 low. The response mean plays no part (`c = 0` fails as often).

  With islands: rows 100, 250 and 477 without neighbours, the other rows row-standardised,
  `y = (I - 0.4 W)^-1 (6000 + 2 x + 0.004 e)` (an island then sits at 6,000 and the rest at
  10,000, `u'u / e0'e0` 2.0e-10), reference spreg on the unshifted response: `rho` 0.3996977
  against 0.39999995 (1296 standard errors), `sigma2` 0.04536 against 1.6195e-5 (2,801 times),
  log-likelihood 2,381 units low, standard errors 53 times too large, slope 0.42% off; status 0,
  converged. The same weights with an ordinary response are right: no spatial process, mean
  10,000, noise 0.5 (`sigma2` 7e-8, `rho` -5.6e-6 as spreg); mean 0.5, noise 0.5 (1e-8).
- Fix. Use the direct form. Add one tile and merge pass after the three sums:
  `rho0 = e0'e1 / e1'e1` and `S0 = sum (e0 - rho0 e1)^2` summed per row. Then
  `s2(rho) n = S0 + e1'e1 (rho - rho0)^2`, a sum of two non-negative terms, and the derivative
  numerator is `e1'e1 (rho - rho0)`. In the float32 emulation `S0` is within 1.2e-4 of float64
  on all eight `s = 200` scenes. Also treat `s2(rho) <= 0` at a grid point as status 3, so a
  remaining case cannot converge on a pole. Pin with the `s = 200` scenes (several streams, both
  centres) and the island scene. `GPUSpatialErrorMaximumLikelihood` and the two panel ML classes
  were not checked for the same form.

### frc-regression-02. Lag ML and 2SLS: the float32 lag of a large-mean response costs up to 4.2% of `sigma2` and 45% of the Anselin-Kelejian statistic; the stated 0.35% and the 1% pin are one scene, not a limit, and the closing specs cannot see it

- Severity: major. CONFIRMED on the test GPU.
- Where: `gpu-spatial-lag-maximum-likelihood.ts` lines 607 to 625 (`lagged-response-tiles`) and
  the class documentation lines 274 to 277; `gpu-spatial-two-stage-least-squares.ts` `getLags`
  lines 784 to 819 and the documentation lines 155 to 157;
  `docs/api/gpu-spatial-analysis/operators-regression.md` lines 501 to 506 and 787 to 791;
  `test/gpu-spatial-analysis/spatial-regression/math-review-closing-residuals.spec.ts` lines
  329 to 357.
- What is wrong. `W y` is summed from the raw response, so each lagged value carries the
  rounding of partial sums near the response mean. The author's 0.35% reproduces exactly on
  their scene, but it is one scene: the error has either sign and reaches twelve times that:

  | Weights, `rho`, residual deviation | Lag ML `sigma2` | 2SLS `sigma2` | Lag ML log-likelihood | Anselin-Kelejian (GPU against spreg) |
  | --- | --- | --- | --- | --- |
  | ring 0.5, 0.4, 0.004 (the author's `lag-unrounded`) | +0.345% | +0.345% | -1.03 | 0.975 against 1.084 (-10.0%) |
  | ring 0.5, 0.8, 0.004 | +0.573% | +0.573% | -1.71 | -1.5% |
  | ring 0.5, 0.8, 0.002 | +0.563% | +0.559% | -1.68 | 0.264 against 0.476 (-44.6%) |
  | six neighbours, float32 1/6, 0.4, 0.004 | -0.027% | -0.027% | +0.08 | 0.217 against 0.288 (-24.9%) |
  | six neighbours, 0.7, 0.004 | -0.302% | -0.297% | +0.91 | 0.249 against 0.336 (-25.8%) |
  | six neighbours, 0.7, 0.002 | +4.24% | +4.25% | -12.5 | -2.5% |

  The last row has the residual scale of the author's own `plain` scene (0.002 around 10,000),
  the case the pass reports as fixed to 6e-6: with six neighbours and `rho = 0.7` the standard
  errors are 2.1% high and the log-likelihood is 12.5 units low, with status 0.
- Why the closing specs pass. Their two scenes per estimator are `plain`, where `rho` is 9.7e-5
  so the lag error is multiplied by nothing, and `lag`, built so that the float32 lag is exact.
  The Anselin-Kelejian pin of 1e-3 holds only there. The only test of an inexact lag asserts 1%
  on the one scene that measures 0.345%.
- The author's reason for leaving it ("lagging the centered response does not cure it in
  general") is too strong. `W y = W (y - c) + c s_i` with `s_i` the row sum. For weights whose
  rows have equal sums (the ring, the six-neighbour lattice, any fixed-count neighbour weights)
  `c s_i` is a constant and goes to the intercept, so lagging the centred response is an exact
  cure. `GPUSpatialErrorGM` already lags centred columns: on the six-neighbour weights with
  `lambda = 0.7` and a residual deviation of 0.002 around 10,000 its `sigma2` is within 2.0e-6
  (homoskedastic) and 5.8e-6 (Kelejian-Prucha) of the reference. For rows with unequal float32
  sums the remainder `c (s_i - 1)` is a second small column, and the row sum is a bounded serial
  walk, the case `getAdditionRoundingError` of contract section 11 is for.
- Fix. Form the lag as `sum w_ij (y_j - ybar)` with each difference through
  `getRoundedDifference`, as `getFilteredRow` of the GM kernel does, and carry `ybar s_i`
  separately (constant when the row sums are equal, otherwise with the exact row-sum remainder).
  Lag ML needs the same shift in the intercept of its second OLS fit and in `reduced`. If it is
  left, state the limit as a rule, not as one scene (added residual variance of the order of
  `rho^2` times half a float32 step of the response, squared, and up to twice that in the table;
  Moran's I of the residuals moved by 0.002 to 0.007), and pin a six-neighbour scene.

### frc-regression-03. Probit: quasi-separation by a small group is status 2 or status 3 at the default settings, not status 4

- Severity: major. CONFIRMED on the test GPU.
- Where: `gpu-probit-regression.ts` line 473 (`select(2.0, 4.0, stateAt(S_LARGEST) > 8.0)`),
  line 539, the class documentation lines 119 to 128;
  `docs/api/gpu-spatial-analysis/coverage-statistics.md` lines 152 to 157.
- What is wrong. Both separation exits test the largest |linear predictor| over all rows against
  8. That works when the separating predictor is continuous, because some row is far from the
  boundary. When a few rows are separated by an indicator, their linear predictor is the largest
  and grows by about `1 / eta` per step. Either the information matrix loses the indicator
  direction first (pivot below 1e-6 of its diagonal; by derivation at a linear predictor near
  6.5 for three rows of 600), so the fit is reported as a collinear design, or the 64 iterations
  end with the indicator coefficient at 7.2 to 7.6 and the fit is status 3 with coefficients
  written out.
- Inputs and values. `x1` and `y1` from the generator's `create_sample(600, 0.2, 1.0, 31337)`;
  an indicator column that is 1 on the listed rows, where the response is set to 1. statsmodels
  `Probit` does not converge on any of them (ConvergenceWarning, indicator coefficient about 8).

  | Predictors | Settings | Status | Right |
  | --- | --- | --- | --- |
  | `x1`, indicator of rows 17, 301, 555 | defaults | 2 | 4 |
  | the same | `maximumIterations` 128 or 256 | 2 | 4 |
  | `x1`, indicator of row 17 | defaults | 3, coefficients (0.1686, 1.0282, 7.2247) written | 4 |
  | `x1`, nine noise columns, indicator of rows 17, 301, 555 | defaults | 3, indicator coefficient 7.57 written | 4 |
  | `x1`, indicator of rows 17, 301, 555 | `tolerance` 1e-2 | 0, indicator 4.77 with standard error 17.0 | 4 (the documented tolerance caveat) |
  | `x1`, indicator of every 20th row; of every 2nd row; rows 17, 301, 555 as the middle of three predictors | defaults | 4 | 4 |

  The documentation says to keep the defaults when separation is possible and that only "very
  few iterations" give status 3. The defaults give status 2 or 3 here.
- Not reproduced: separated data reported as status 0 at the default tolerance. By derivation it
  needs the scaled step of the indicator, about `sqrt(m / n) / eta`, below
  `1e-5 (1 + largest scaled coefficient)`: more than 1e7 rows for ordinary coefficients, or about
  2e5 rows next to a scaled slope near 40. Not run.
- Fix. Decide separation by what defines it, not by a threshold on |eta|: at every exit that is
  not convergence (singular information, exhausted iterations, failed line search) test whether
  the last Fisher direction `d` moves no row towards the wrong side,
  `min_i s_i x_i' d >= -epsilon max_i |x_i' d|`, while the likelihood still rises. The
  line-search tile pass already forms `x_i' d` per row and has a maximum slot per step, so this
  is one more reduced value and no extra dispatch. Add the three failing scenes to
  `math-review-probit-bound.spec.ts`.

### frc-regression-04. 2SLS accepts a constant predictor with status 0

- Severity: minor (degenerate input). CONFIRMED on the test GPU.
- Where: `gpu-spatial-two-stage-least-squares.ts` `solveSystem` lines 895 to 908 (the
  `diagonal > 0` tests) on the moments of lines 484 to 509.
- What is wrong (by reading; the mechanism was not isolated on the GPU). The moments are about
  the stored float32 means and, unlike the OLS `momentAt` of today's change, are not corrected
  by the deviation sums. A constant column whose float32 mean is not exactly the constant leaves
  a "centred" column of one tiny value, a sum of squares above zero, and a unit-scaled column
  that nothing compares with the intercept.
- Inputs and values. The lag scene with the predictor near 4,096 (of the checked list below)
  plus a second predictor equal to 12345.678 in every row: status 0, constant-column coefficient -0.0112
  with standard error 0.028, intercept -2054.58 with standard error 346.6 (without the column:
  -2193.04 and 0.947); slope, `rho` and `sigma2` unchanged. The island weights with a small mean
  plus a constant 0.1: status 0, constant-column coefficient 0.58 with standard error 339,750.
  The other constants tried (1/3 and 0.1 on the first scene, 7 on the second) give status 1.
  Lag ML and both GM estimators give status 1 on all five inputs, and
  `GPUOrdinaryLeastSquares` on its own constant columns (0.1, 12345.678, 1/3).
- Fix. Correct the 2SLS moments as OLS does (subtract `n` times the product of the two mean
  remainders, which the moment pass already sums for the regressors and needs for the
  instruments), so the diagonal of a constant column is zero; or test each diagonal against
  `n` times the squared float32 step of the column mean.

### frc-regression-05. Probit: a legitimate fit that runs out of iterations is status 4 with NaN coefficients

- Severity: minor. CONFIRMED on the test GPU.
- Where: `gpu-probit-regression.ts` line 539; output documentation line 79 ("Nonconverged finite
  iterates are available only with status 3"); `coverage-statistics.md` line 131.
- Inputs and values. The author's `correctOutlier` scene (2,001 rows, converges in 7 steps with
  the defaults) with `maximumIterations: 3`: status 4, coefficients NaN. The `control` scene
  with the same budget: status 3 with the iterate (0.2475, 1.8517). 2,000 rows from a latent
  index `0.2 + 8 x` (statsmodels converges, slope 7.70) with `maximumIterations: 4`: status 4.
  Neither dataset is separated; the only difference from `control` is a correctly classified
  row beyond a linear predictor of 8.
- Fix. The test of frc-regression-03 removes this too. Otherwise say in both documents that
  status 4 also covers an unfinished fit with a linear predictor above 8.

### frc-regression-06. The specs are built where the remaining defects cannot show

- Severity: minor. CONFIRMED by reading and by the probes above.
- `math-review-closing-residuals.spec.ts`: one predictor with mean near zero, one weights matrix
  (ring, 0.5), and for the lag models `rho` near zero or an exact lag (frc-regression-02). No
  scene has a high fit (frc-regression-01), islands, several predictors or a predictor with a
  large mean. The last was probed and is right (see the list below), so it only needs a pin.
- Line 221 names every test "matches spreg in float64". For the Kelejian-Prucha estimator the
  reference is the generator's own minimiser of spreg's moment function, with `sigma2`
  eliminated by the same clamped least squares as the kernel's `getMomentResidual`. That is the
  right definition and I confirmed the author's reason for it (list below), but spreg's own
  `lambda` on the `error` scene (0.53285, stored as `spregOptimizerLambda`) is 1.8e-4 from the
  reference, outside the 1e-4 the test allows (line 257), and nothing asserts it. Rename the
  test, and assert the stored spreg value on `error` at about 5e-4 so the library still bounds
  the kernel.
- `math-review-probit-bound.spec.ts` lines 214 to 240: the three separation tests all separate
  on continuous predictors, where some row is always far beyond 8.
- `probit-regression.node.spec.ts` lines 72 to 90 assert on kernel source text
  (`select(2.0, 4.0, stateAt(S_LARGEST) > 8.0)`, `limitedByBound`). They pin the wording of the
  rule that frc-regression-03 shows to be wrong, and will fail on its fix for no numerical
  reason.
- The probit standard errors of the fixture are the kernel's formula
  `X' diag(lambda(eta) lambda(-eta)) X` in float64 with SciPy, not a library output. They are
  right: statsmodels `GLM(Binomial(Probit))` (IRLS, expected information) gives the `control`
  values to 1e-8. Putting that call in the generator would make the fixture what its name says.

### frc-regression-07. Documentation that disagrees with the code or with the measurements

- Severity: minor. CONFIRMED by reading.
- `docs/api/gpu-spatial-analysis/operators-regression.md` line 510: "heteroskedasticity-robust
  and HAC standard errors are not provided (spreg `robust`)". The class has a `robust` prop
  (HC0, HC1, spatial HAC; `gpu-spatial-two-stage-least-squares.ts` lines 123 to 126 and 718 to
  772). The prop list at the top of that section omits it too.
- The same file, lines 501 to 506 and 787 to 791: `sigma2` and the Anselin-Kelejian statistic
  "match spreg in float64 ... (1e-4 and 1e-3)" and "0.35% of `sigma2`". True on the two spec
  scenes only (frc-regression-02).
- `docs/api/gpu-spatial-analysis/coverage-statistics.md` lines 152 to 157 and
  `gpu-probit-regression.ts` lines 119 to 128: the separation paragraph (frc-regression-03), and
  line 131 with source line 79 (frc-regression-05).
- `gpu-spatial-lag-maximum-likelihood.ts`: `getKahanSumWGSL` is still the name of the four
  merges (lines 674, 755 to 757, 909, 969), one of them new today. It is a pairwise sum; the
  helper's comment says so, the call sites do not.

### frc-regression-08. Differences from a mean that are still unguarded

- Severity: minor. SUSPECTED: no wrong value was produced on the inputs tried.
- Where: `gpu-ordinary-least-squares.ts` line 382 (`slope * (design - mean)` in the residual
  pass, while the moment pass guards the same difference);
  `gpu-spatial-lag-maximum-likelihood.ts` lines 935, 937 and 950;
  `gpu-spatial-two-stage-least-squares.ts` lines 682 and 685; `gpu-probit-regression.ts` line
  302 (`getDesign`).
- These are the shape contract section 11 warns about: a sum of `b_j (x_j - m_j)` can be
  regrouped as `sum b_j x_j - sum b_j m_j`, which for a predictor near 4,096 with a slope of 2
  rounds each fitted value to a float32 step of 8,192 (0.001), half the residual deviation of
  the `plain` scene. On the test GPU it is not regrouped: OLS with one predictor on a grid
  around 4,096 and with three predictors around 4,096, -8,192 and 300 has a residual sum of
  squares within 2.8e-6 and 5.0e-6 of float64 and a largest residual error of 2.0e-4 and 4.6e-4
  of the residual deviation; probit with the predictor shifted by 10,000 is within 2e-6 of a
  standard error. The author found the same kind of chain compiled as written with two terms and
  regrouped with three. Guard line 382 of OLS as the moment pass is guarded, or pin the two OLS
  scenes.

## Checked and found correct

- Fixtures: both generators rerun, both fixtures byte-identical. The specs read every reference
  number from the fixtures and check an input sum against the generator's.
- OLS algebra: `momentAt` (moment about the stored means less `n` times the product of the two
  mean remainders) is the moment about the true means; the residual offset
  `delta_y - sum b_j delta_j` makes the residual `(y - mu_y) - sum b_j (x_j - mu_j)`; the
  intercept and the fitted value add the same offset; degrees of freedom `n - k - 1`, with
  `sigma2 = RSS / n` only in the log-likelihood. The workspace and partial layouts agree
  (moments then deviation sums, stride `momentCount + k + 1`).
- OLS on the GPU against float64 least squares of the float32 inputs: predictor near 4,096
  (residual sum of squares 2.8e-6, standard errors 1.3e-6); three predictors with large means
  (5.0e-6, 2.5e-6); response near -50,000 with a row mask (1.3e-6); 200,000 rows near 10,000
  (2.5e-8); response near 1e6 with `tileRowCount: 1` (1.3e-7); ordinary data, two predictors
  (2.4e-8, slopes 6e-8), so a well-conditioned dataset is not moved; three rows and one predictor
  (status 0, 1.2e-5); two rows (status 2); constant predictor 0.1, 12345.678, and 1/3 as the
  second of two (status 1); constant response 0.1 and 10000.1 (status 0, slope 0, residuals 0,
  `R2` NaN, log-likelihood at its documented floor).
- Lag ML information matrix: equals spreg's `v` with the `sigma2` row eliminated
  (`-2 tr1^2 / n`); centring the predictors and `W A X b` is a change of basis that leaves the
  `rho` diagonal term alone, and the shift vector `(1, -mean(x), -mean(W A X b))` gives the
  intercept variance. Against spreg with the intercept mapped back from its covariance:
  predictor near 4,096, intercept standard error 0.935078 against 0.935049, standard errors of
  the slope and of `rho` within 5e-6 and 2.5e-5, `sigma2` 2.3e-6, intercept -2192.912 against
  -2192.907 (the float32 slope error times 4,096).
- 2SLS: the residual offset makes the residuals sum to zero as 2SLS with a constant among the
  instruments requires; `sigma2 = u'u / n` is spreg's default; the intercept variance
  `sigma2 (1 / n + m' (Zc' P Zc)^-1 m)` and the Anselin-Kelejian quadratic form
  `(u'W1)^2 / n + (u'W Zc) (Zc' P Zc)^-1 (u'W Zc)'` are spreg's `(u'WZ) (Z'PZ)^-1 (u'WZ)'` in
  centred coordinates (derived). Predictor near 4,096: `sigma2` 1.7e-6, standard errors 1e-6,
  Anselin-Kelejian 2.7e-5, intercept standard error 0.946978 against 0.946977. Three islands,
  response near 10,000 with noise 0.5 and no spatial process, against spreg on the unshifted
  data: `sigma2` 6e-5, Anselin-Kelejian 1.3e-6. (With noise 0.004 the three island residuals
  `rho * 10000` are nearly all of `u'u`, in spreg too, and `sigma2` is 0.16% from spreg: `rho`
  would have to agree to 3e-8.)
- GM: centring before filtering is exact for any centring constant because the filtered constant
  `1 - lambda s_i` is a column, so the float32 means cost nothing; the reported residual is
  formed in the same centred model. Predictor near 4,096: `sigma2` 5e-6 (homoskedastic) and
  1.9e-5 (Kelejian-Prucha), intercept standard error 0.539957 against 0.539955. Three islands,
  response near 10,000: 4.9e-6 and 1.2e-5, `lambda` standard error 5e-6.
- The Kelejian-Prucha claim. On the shifted `plain` scene spreg `GM_Error` returns
  `lambda = -0.98999...` while a 19,801-point grid of its own objective (with `sigma2`
  eliminated) has one minimum, at 0.0597; on the unshifted `plain` scene, whose moments agree to
  nine digits, spreg returns 0.05982. On `error` it returns 0.53285 against a grid minimum of
  0.5327. spreg's optimiser is unreliable there, and the generator's grid plus golden section is
  the global minimum of the profiled objective, which is the same problem as spreg's joint
  problem in `(lambda, sigma2 >= 0)`.
- Guards: `getRoundedDifference` forces the difference to be materialised, and each runtime zero
  is written by an earlier node and read from a buffer (as a `0.0` float through `bitcast` in the
  three float workspaces). No float compensated sum or error-free transformation remains in the
  five files; every merge is the pairwise sum.
- Probit numerics: `erfcValue` is the Numerical Recipes `erfcc` (fractional error 1.2e-7); with
  the floor at the smallest normal f32 no likelihood, score or weight is NaN or infinite for a
  finite linear predictor (right side beyond 13: score 0, weight 0; wrong side to 12: Mills
  ratio 12.08). A response of 0 added to `control` at `x = 5, 6, 7, 7.5` (wrong side at 9.12,
  10.37, 11.45, 11.94): status 0, coefficients within 2.6e-5 of a standard error, standard
  errors within 1e-6, log-likelihood within 8e-8. At `x = 8` (12.40): status 4, as designed.
  Steep fits that are not separated, latent slopes 8, 20, 20 and 40 with largest linear
  predictors of 30, 64, 104 and 176: status 0 in 11 to 17 steps, within 1e-5 of a standard
  error. Predictor shifted by 1,000 and by 10,000: status 0 and within 2e-6 of a standard
  error, where statsmodels itself does not converge.
- Rows close to the predictor count: three rows on a three-row ring are status 2 for all four
  spatial estimators; four rows are status 0.
- Contract: no bare `assert`; `check-contributor-contract` reports no new finding; the one
  abbreviated name on a changed line (`wp`) predates the pass.

## Not checked

- The old kernels were not run, so "before" values are the author's.
- `GPUSpatialErrorMaximumLikelihood` and the panel ML classes for the cancellation of
  frc-regression-01.
- The four-row fits were not compared with spreg.
- Chunked weights, the stochastic trace mode, `tailExtrapolation`, the exact log-determinant
  mode, `instrumentOrder: 2`, the `robust` covariance and ridge OLS with a large-mean response.
- Several predictors with large means in the spatial estimators (OLS only).
- Probit `spatialDiagnostics`, masks, and more than eleven predictors; separation reported as
  status 0 at the default tolerance (derived only, frc-regression-03).
- Weights with unequal float32 row sums for frc-regression-02 (all probe weights have equal row
  sums, the case in which the shifted reference is exact).
- Any GPU other than the test GPU.
- `docs/lens-records/**`, the operator catalog entries and the generated parameter tables.
- The full suites, `yarn build` and the benchmark specs were not run.

## Re-review (2026-11-11)

Re-review of "Closing pass, review fixes (2026-11-11)" of `regression.md`, by the same reviewer,
who again wrote none of the code. Result: frc-regression-01, -04, -05, -06, -07 and -08 closed;
-02 closed for weights whose rows share a row sum, with a leftover that is not the float32
limit it is described as (frc-regression-11); -03 partly closed (frc-regression-10). The new
ground gives one more major finding in a kernel the pass did not touch (frc-regression-09).
No blocker, 2 major, 2 minor.

Evidence used:

- All 134 probes of the first review rerun unchanged on the current kernels, plus 65 new GPU
  probe runs (error ML, panel lag ML, the author's leftover scenes rebuilt with the author's
  generator, probit on small and large samples), in a scratch folder under `test/` (deleted).
  Sources, reference scripts and raw output are in
  `/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-luma-contributors/ba3fe269-f86e-4664-86ec-632935ba200a/scratchpad/review-regression/`
  (`probe_data6.py`, `probe_data7.py`, `leftover_2sls.py`, `gpu-probe-specs/rereview-*.spec.ts`,
  `rerun/`).
- Reading the changed parts of the lag ML, 2SLS and probit kernels, and
  `gpu-spatial-error-maximum-likelihood.ts` and `gpu-spatial-panel-lag-maximum-likelihood.ts`
  in full.
- The regression spec folder on the test GPU: headless 45 files, 470 tests; node 34 files, 202
  tests; all pass. `statistics-regression-lag-conditioning.py` and `statistics-probit-tails.py`
  rerun in the fixture sandbox: IDENTICAL.

### The eight findings of the first review

| Finding | State | Rerun of my probes |
| --- | --- | --- |
| frc-regression-01 | closed | The 24 high-fit scenes: all status 0, interior; `rho` within 0.035 standard errors, `sigma2` within 3.4e-4, log-likelihood within 0.10 (before: 12 status 3, 3 wrong with status 0). Islands with the lag process: `rho` 0.092 standard errors, `sigma2` -4.0e-5, log-likelihood 0.012 (before: `sigma2` 2,801 times too high). |
| frc-regression-02 | closed where the rows share a row sum; see frc-regression-11 for the rest | The six rows of my table: lag ML `sigma2` within 1.1e-5 and log-likelihood within 0.0033; 2SLS `sigma2` within 6.7e-6 and Anselin-Kelejian within 3.8e-4. Six neighbours, `rho` 0.7, deviation 0.002: -1.4e-6 and -3.6e-6 (before: +4.2%). |
| frc-regression-03 | partly closed; see frc-regression-10 | All my 600-row scenes are status 4 at the defaults, also with 128 and 256 iterations and with `tolerance` 1e-3 and 1e-2 on the three-row indicator. |
| frc-regression-04 | closed | All 20 constant-column fits (five inputs, four estimators) are status 1. |
| frc-regression-05 | closed | Correct-outlier scene with 2 and 3 iterations, latent slope 8 with 4: status 3 with the iterate. |
| frc-regression-06 | closed | The Kelejian-Prucha test is named for its reference and asserts spreg's `lambda` at 5e-4; GLM standard errors are in the fixture; the `> 8` text assertions are gone. What the new specs still lack is in frc-regression-09 and -10. |
| frc-regression-07 | closed | `robust` is in the prop list and the standard-error paragraph; the four lag ML merges call a pairwise helper by that name. One new statement is inaccurate (frc-regression-11). |
| frc-regression-08 | closed | Guards present in the OLS residual pass, lag ML final moments, 2SLS Anselin-Kelejian products and probit `getDesign`. |

### frc-regression-09. Panel lag ML publishes `sigma2` from `first - 2 rho cross + rho^2 last`: wrong from a moderately high fit on, with status 0

- Severity: major. CONFIRMED on the test GPU. Not introduced by either pass; this is the new
  ground asked for.
- Where: `src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-lag-maximum-likelihood.ts`
  line 249 (`objective`, the golden section over `rho`) and line 281 (the published variance,
  which also scales the log-likelihood and the information matrix); the three sums are formed
  serially at line 267.
- What is wrong. It is the form frc-regression-01 removed from the cross-section kernel, and
  worse here: the published `sigma2` itself is the difference of the three float32 sums, not a
  sum over the per-row residuals. The search treats a non-positive variance as the worst value,
  so it settles where the rounding of the three sums happens to leave the smallest positive
  number.
- Inputs and values. 16 entities on a ring (weights 0.5), 16 periods, one predictor uniform on
  `[-2, 2)`, unit effects of deviation 0.7,
  `y_t = (I - 0.4 W)^-1 (0.6 c + s x_t + mu + d e_t)`. Reference: spreg 1.9.1 `Panel_FE_Lag`
  and `Panel_RE_Lag` (`epsilon=1e-9`); for unit effects also a float64 concentrated likelihood
  (within transformation, grid and golden section), which agrees with spreg to 1e-8.

  | `s`, `d` | `u'u / e0'e0` | Unit effects | Random effects |
  | --- | --- | --- | --- |
  | 0.5, 1 | 0.70 | `sigma2` -2.7e-5, log-likelihood 5e-5 | `sigma2` +9.2e-5 |
  | 2, 0.004 | 3.0e-5 | `sigma2` +1.1%, log-likelihood -1.0 | `sigma2` -2.0%, log-likelihood +2.6 |
  | 20, 0.004 | 2.6e-7 | `sigma2` -36%, log-likelihood +190, `rho` 1.7e-4 off | `sigma2` +96%, log-likelihood -81 |
  | 200, 0.004, four streams, `c` 0 and 10,000 | 2.6e-9 to 3.0e-9 | 7 of 8 status 0 with `sigma2` 13 to 238 times spreg's; 1 status 2 with `sigma2 = -8.7e-5` written | 2 of 8 status 0 with `sigma2` 9.6 and 101 times spreg's; 6 status 3 |

  The per-row `residuals` output is much better than the published variance: at `s = 20`,
  random effects, its mean square is within 9.5e-4 of spreg's `sig2` while the published
  `sigma2` is 96% high. The covariance is scaled by the published variance, so the standard
  errors are wrong by the square root of the same factor.
- Fix. As frc-regression-01: `rho0 = cross / last`, `S0 = sum (residual - rho0 lagResidual)^2`
  summed per row in the same loop that forms the three sums, `variance(rho) = (S0 + last
  (rho - rho*)^2) / rows`; publish `sigma2` from that. Pin with the `s = 2`, 20 and 200 scenes
  for both effects.

### frc-regression-10. Probit: the separation certificate fails when the separated group is below about 2e-5 of the rows, and separated data are again status 0, 2 or 3 at the default settings

- Severity: major. CONFIRMED on the test GPU. This is the residue of frc-regression-03 and the
  case the first review derived but could not run.
- Where: `gpu-probit-regression.ts` `SEPARATION_TOLERANCE` (lines 55 to 65), the move test in
  the line-search pass (lines 356 to 366) and its use in `decide` (lines 552 to 572).
- What is wrong. The move of a row is measured against the sum of the magnitudes of its own
  terms. For a row outside the separated group the true move is zero, and the only terms that
  give the ratio a scale are the indicator term and the intercept term that cancels it, both
  proportional to the fraction of separated rows. Once that fraction is small enough the
  rounding of the settled coefficients' direction is no longer 1e-3 of those terms, no direction
  is certified, and the fit ends as it did before the fix. The author measured a rounding floor
  "just under 1e-4" on one row of 600 and set the tolerance at 1e-3; the floor grows as the
  fraction falls.
- Inputs and values. `create_sample(n, 0.2, 1.0, 31337)` of the author's generator with an
  indicator column on the listed number of rows, whose responses are set to 1 (quasi-separated
  by construction). Default `maximumIterations` and `tolerance`.

  | Rows | Separated rows | Status | Written |
  | --- | --- | --- | --- |
  | 600, 2,000, 5,000, 20,000, 50,000 | 1 | 4 | |
  | 20,000 | 3; 20 | 4 | |
  | 100,000 | 2; 3; 5 | 4 | |
  | 100,000 | 1 | 0 (also with 128 iterations) | indicator 7.589, standard error 17,060 |
  | 200,000 | 3 | 0 | indicator 7.449, standard error 10,947 |
  | 400,000 | 5 | 2 | |
  | 1,000,000 | 5 | 3 | indicator 7.465 |
  | 1,000,000 | 20 | 0 | indicator 8.008, standard error 3,644 |

- No false certificate was found: 60 samples of 25 to 69 rows with one to four predictors that
  a linear program and statsmodels agree are not separated are all status 0 (within 7e-6 of a
  standard error), and 12 such samples that are separated are all status 4; an indicator group
  with both outcomes (600 and 20,000 rows) and two rare-event fits (33 and 343 events in 20,000
  rows) are status 0 within 6e-6 of a standard error.
- Fix (suggested, not tested). Take the rounding out of the direction before testing it: express
  the direction in uncentred coordinates, drop every component whose largest contribution to
  any row is below about 1e-4 of the largest contribution of any component, and test the sign of
  what is left. For an indicator that leaves exactly the indicator term, zero outside the group.
  Whatever the test, pin it at 100,000 and 1,000,000 rows; the 22 tests of
  `math-review-probit-bound.spec.ts` all have 400 to 2,001 rows.

### frc-regression-11. The 2SLS `sigma2` leftover (3% and 0.8%) is solve error, not the float32 resolution of `rho`, and one refinement step removes it

- Severity: minor (the scenes are extreme). CONFIRMED: the values on the test GPU, the
  decomposition in float64 from the GPU's coefficients (`leftover_2sls.py`).
- Where: `gpu-spatial-two-stage-least-squares.ts` class documentation lines 175 to 181;
  `docs/api/gpu-spatial-analysis/operators-regression.md` lines 518 to 521; the pins of
  `math-review-lag-conditioning.spec.ts` lines 426 and 450 (2.5% and 10%).
- The author's measurements reproduce: `lag-unequal-rho-0.15-deviation-0.004` `sigma2` +2.96%,
  standard errors +1.5%; `islands-lag` +0.77%; both status 0 with the precision warning 0.
- The attribution does not hold.

  | Scene | `rho` error | `sigma2` at the GPU's coefficients, in float64 | At the float32 nearest to the float64 estimate | After one refinement step |
  | --- | --- | --- | --- | --- |
  | row sums 2 and 3 | 4.6e-8, 3.1 float32 steps of 0.15 | +2.95% | +0.07% | -1e-10 |
  | three islands | 1.55e-7, 5.2 float32 steps of 0.4 | +0.85% | +0.03% | 4e-11 |

  A float32 `rho` of 0.15 is spaced 1.5e-8, not "known to about 5e-8". The representation
  costs 0.07% and 0.03%; the rest is the float32 solve landing three and five steps from the
  estimate.
- A cheap exact route exists, and it is not the row-sum term (that is already carried exactly).
  One step of iterative refinement: the residual pass also sums `H'u` (`m` more sums), the solve
  kernel forms `d = (Z'PZ)^-1 Z'P u` with the factor it already holds, and the residuals are
  corrected per row by `-z d`. The correction is kept beside the float32 coefficient, not added
  to it, so `sigma2`, the standard errors and the Anselin-Kelejian statistic are those of the
  estimate. In float32 the correction (4.6e-8 in `rho`) needs two digits; by estimate `H'u`
  gives it about three (a signal near 1e-4 against a rounding of the sum near 1e-7).
- Status 0 is defensible for a `rho` that is right to 0.015 standard errors, but not silently.
  The statement that "the kernel has no float32 quantity that says the fit is in this regime" is
  wrong: the relative change of `sigma2` per float32 step of `rho` is
  `2 |u'z_rho| ulp(rho) / u'u`, two sums of the residual pass. It is 0.97% on the first scene and
  0.16% on the second (computed in float64 from the GPU's coefficients), and by estimate of the
  order of 1e-8 for row-standardized weights. It belongs in the existing precision warning if
  the refinement is not done.
- The same sensitivity moved one of my scenes between the two reviews (ordinary response near
  10,000 with residuals of 0.004 on the island weights): 2SLS `sigma2` 0.16% from spreg before,
  0.41% now.

### frc-regression-12. The lag ML consistency check cannot catch a wrong `rho`

- Severity: minor. By derivation from `gpu-spatial-lag-maximum-likelihood.ts` lines 116 to 123
  and 1384 to 1392; it did not fire in any lag ML probe of either review.
- The two quantities compared are `S + e1'e1 (rho - rho*)^2` and `sum (e0 - rho e1)^2`: the same
  quadratic in the same `rho`, from the same two residual columns. They agree for every `rho`,
  whether or not it maximises the likelihood. The check therefore catches only a disagreement
  between the two summation routes (a compiler that regroups one of them). A `rho` that is
  wrong because of the search grid, the log-determinant series, stochastic traces or wrong
  `e0`, `e1` passes it. The class documentation ("a larger gap means the search did not see the
  likelihood it reports") says more than the check does; the first sentence of that comment
  ("the same quantity formed two ways") is the accurate one.
- A correct fit failing it was not seen: the largest gap needs `rho e1` rounded at a size far
  above the residual, which is past the resolution of the inputs.

### The other leftovers

- Probit `tolerance: 1e-2` on one separated row of 600: status 0 after 4 steps (indicator 3.94,
  standard error 5.39). `maximumIterations: 6` on the same scene: status 3; 12: status 4. Both
  reproduce and neither is reachable with the default props. The hazard at the default settings
  is frc-regression-10.
- Lag ML with row sums 2 and 3: `sigma2` -2.5e-4, `rho` 0.065 standard errors. Accepted: here
  the fitted values of the first-stage residuals are at thousands with residuals of 0.004.

### Error ML

`gpu-spatial-error-maximum-likelihood.ts` has a three-term form in `lambda`
(`G0 - lambda (G1 + G1') + lambda^2 G2`, lines 985 to 989), but its response column is the
first-stage OLS residual, so the filtered sum of squares is never a small fraction of its terms:
0.61 to 0.75 of `e0'e0` on the high-fit, island and six-neighbour scenes below, and 0.15, 0.09
and 0.068 at `lambda` 0.85, 0.9 and 0.92. Against a float64
concentrated likelihood (and spreg `ML_Error(method='full')`, whose own `lambda` is up to 1e-3
from it): slope 200 with residuals of 0.004, centres 0 and 10,000, three streams: status 0,
`lambda` within 0.006 standard errors, `sigma2` within 2.9e-4, log-likelihood within 0.06
(the same float32 rounding of the fitted values as lag ML); three islands: `sigma2` -2.0e-6;
six neighbours, `lambda` 0.7, deviation 0.002: -1.2e-6; ring, `lambda` 0.85, 0.9 and 0.92:
within 1.9e-6, log-likelihood within 4e-4. Not affected.

### Regressions

None found. Of the 134 first-review probes, every ordinary scene agrees with its first-review
value to float32 noise (OLS on ordinary data 2.4e-8 both times; probit control, GM and the
island scenes with an ordinary response identical to the digits recorded). Two values moved by
more than that, neither a regression: four rows on a ring with 2SLS went from intercept 1156.9
and `rho` 0.8838 to 1174 and 0.8822, and the float64 solve gives 1173.71 and 0.88216 (the old
value carried the float32 lag error); and the island scene of frc-regression-11.

### Not checked in the re-review

- `GPUSpatialPanelErrorMaximumLikelihood` (a text search finds no three-sum form; not read in
  full, not probed).
- Negative weights, chunked weights, the stochastic trace mode, `tailExtrapolation`, the exact
  log-determinant mode, `instrumentOrder: 2` and the `robust` covariance with the new lag column.
- The refinement of frc-regression-11 and the test of frc-regression-10 are proposals: neither
  was run on the GPU.
- Any GPU other than the test GPU; the full suites and `yarn build`.

## Re-review of round three (2026-11-11)

Bounded re-review of "Third round (2026-11-11): the second reviewer" of `regression.md`, in the
three areas the lead named. Same reviewer, same GPU and reference libraries as above. Result: the
probit rules and the lag ML first-order check hold on everything tried; the restructured panel
kernel holds on every output except one pre-existing case. No blocker, 1 major, 2 minor.

Evidence used:

- 1,939 GPU fits in a scratch folder under `test/` (deleted): 58 panel fits; 1,026 lag ML fits;
  197 probit scenes at the defaults, 649 probit fits under other settings, 9 with spatial
  diagnostics. Sources, reference scripts and raw output:
  `/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-luma-contributors/ba3fe269-f86e-4664-86ec-632935ba200a/scratchpad/review-regression/`
  (`probe_data8.py` to `probe_data11.py`, `pair_check.py`, `gpu-probe-specs/round3-*.spec.ts`,
  `rerun/round3-*`).
- The regression spec folder, once: headless 46 files, 513 tests; node 34 files, 204 tests; all
  pass. (The author's 47 files and 515 tests include the helper spec in `test/utils`.)

### 1. Panel lag ML

29 scenes, unit and random effects (58 fits), against spreg 1.9.1 `Panel_FE_Lag` and
`Panel_RE_Lag` (`epsilon=1e-9`). Compared per fit: every coefficient in units of its spreg
standard error, every standard error, the whole covariance matrix (each entry over the product
of the two standard errors), `phi`, `sigma2`, the log-likelihood, and the `residuals` and
`fitted` outputs row by row over the residual deviation. All 58 are status 0.

| Group (fits) | Coefficients | Standard errors | Covariance | `sigma2` | Log-likelihood | Residuals | Fitted |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Ordinary: 8 x 6 to 32 x 32, 1 to 7 predictors, predictor means of 2,024 and 30,000, no effects (16) | 0.0042 | 2.1e-4 | 4.1e-4 | 9.3e-5 | 5e-4 | 1.0e-3 | 0.017 |
| Unbalanced weights: path, hub, directed three nearest, float32 `1 / degree` (8) | 0.0054 | 2.1e-4 | 4.2e-4 | 9.4e-5 | 1.3e-3 | 1.0e-3 | 0.070 |
| Near-collinear: two predictors 0.02 apart, a predictor nearly constant in time, effects collinear with the predictor's entity means, `phi` 0.004 (8) | 0.0072 | 3.9e-4 | 7.8e-4 | 1.6e-4 | 6e-4 | 9.4e-4 | 9.4e-4 |
| High fit: slopes 2, 20, 200 against residuals of 0.004, centres 0 and 10,000, ring and path (14) | 0.20 | 4.8e-4 | 9.6e-4 | 8.4e-4 | 0.11 | 0.048 | 1.2 |
| Entity effects of +3000 and -3000 (and 4097), ordinary and small within-entity variation, unit effects (6) | 0.026 | 2.1e-3 | 4.2e-3 | 4.2e-3 | 0.51 | 0.15 | 0.062 |
| The same with residuals of 1, random effects (`phi` 9e-5 and 7e-5) (2) | 0.0076 | 3.2e-3 | 6.4e-3 | 5.9e-4 | 0.054 | 2.0e-3 | 1.2e-3 |

Ordinary panels did not move beyond float32 noise (the first re-review had 2.7e-5 and 9.2e-5 in
`sigma2` on its ordinary scene). The high-fit and opposing-effect rows are inside what the
author states (1.7e-3 and 0.6%). The four remaining fits are frc-regression-13; the `fitted`
column is frc-regression-15. `hausman` was not checked.

### frc-regression-13. Panel lag ML, random effects: `phi` is clamped at 1e-5, and a panel whose `phi` is smaller gets the constrained fit with status 0

- Severity: major. CONFIRMED on the test GPU. Not introduced by round three: the author lists
  the case as not run.
- Where: `src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-lag-maximum-likelihood.ts`
  line 340 (`clamp(..., 1.0e-5, 1.0)`); `docs/api/gpu-spatial-analysis/coverage-statistics.md`
  lines 549 to 550 state the interval and the boundary bit.
- Inputs and values. 16 entities on a ring, 16 periods, entity effects alternating +3000 and
  -3000 (or 4097), `rho` 0.4, slope 2, residual deviation 0.004; spreg's `phi` is 2.2e-7 to
  3.1e-7.

  | Scene | GPU | spreg `Panel_RE_Lag` |
  | --- | --- | --- |
  | 3000, ordinary predictor | `sigma2` 64 times, `rho` 2.3 standard errors off, log-likelihood 533 low, standard errors 8 times | |
  | 3000, predictor varying by 0.01 within an entity | `sigma2` 28.6 times, `rho` -0.338, log-likelihood 424 low | `rho` 0.4225 |
  | 4097, ordinary predictor | `sigma2` 119 times, `rho` 4.3 standard errors off | |
  | 4097, predictor varying by 0.01 within an entity | `sigma2` 35.5 times, `rho` -0.498 | `rho` 0.4225 |

  All four are status 0 with boundary bit 2. A float64 fit with `phi` fixed at 1e-5 reproduces
  the GPU values (`sigma2` ratios 64.4, 28.4, 119.0, 35.4; `rho` -0.3384 and -0.4985), so the
  kernel is right for the problem it solves and the clamp is the whole error. Bit 2 is also set
  when `phi` ends at 1 (two of my ordinary scenes, both right to 1e-4), so it does not tell the
  two apart. With residuals of 1 on the same effects (`phi` 9e-5) the fit is right.
- Fix. Either lower the bound (the moment polynomial in `phi` has no term that fails at 1e-9;
  the limit `phi = 0` is the unit-effects estimator), or report status 3 when `phi` ends at
  the lower bound, and give the two ends different bits. Pin the four scenes.

### frc-regression-15. Panel lag ML, random effects: `fitted` moves by the convergence error of `phi` times the response mean

- Severity: minor. CONFIRMED on the test GPU.
- Where: the same file, `residuals` node (line 429, `predicted + phi * centers`), with the
  `tolerance` of the alternation (1e-5 in `phi`).
- The quasi-demeaned fitted value contains `phi * mean(y)`. `phi` agrees with spreg to 4.5e-7
  on the slope-200 scene with centre 10,000 and to 1.1e-5 on the ordinary scene with large
  predictor means, inside the tolerance; times the response mean that is 1.14 and 0.017
  residual deviations, which is the whole `fitted` error measured (1.18 and 0.017). The
  `residuals` output is not affected (0.048 and 2.4e-4).
- Fix. Document it, or stop the alternation on `phi * |mean(y)|` against the residual
  deviation as well.

### 2. Probit

(a) False status 4. None in 176 scenes that are not separated at the defaults, nor in 518 fits
of 74 such scenes under `tolerance` 1e-2 and 1e-3 and `maximumIterations` 1, 2, 8, 128 and 256.
The scenes: random designs with 1 to 11 predictors (the class limit) at 100, 400, 2,000 and
20,000 rows; rare outcomes to 200,000 rows; an indicator group of 3 to 50,000 rows that is all
ones but for one contrary row, at 600 to 1,000,000 rows; a threshold with one or two flipped
rows; predictor offsets of 1e3, 1e5, 1e7 and -3e6; 1,000,000 rows with 2 and 6 predictors and
300,000 with 11; and, for the exact tests, a separated sample plus a pair of rows of opposite
response `2^-1` to `2^-20` apart across the boundary. "Not separated" is the generator's linear
program up to 30,000 rows and a converged float64 Newton solve above. Duplicated columns,
a column that is the sum of two others, complementary indicators, and a column that is a
combination of two others to float32 rounding are status 2 (15 of 15).

(b) Status 0 that is not the maximum. None. The 153 status-0 fits at the defaults are within
7.8e-4 of a standard error of the tail-exact float64 maximum (statsmodels `Probit` agrees with
that solve to 1e-5 of a standard error on the 124 scenes where its own raw design is
conditioned; it is 5 to 48 standard errors off at offsets of 1e7 and -3e6, where the GPU is
right), standard errors within 2.2e-4, log-likelihood within 1.4e-6. The near-separated pairs
converge to maxima with slopes of 7 to 45,720 and linear predictors up to 148,000, within
2.3e-5 of a standard error. No separated scene is status 0 under any setting: the 26 of the
first batch are status 4 at the defaults; of six scenes with tied rows of both responses on
the boundary, the two with one predictor are status 4 and the four with two predictors status
2, the limit the author states; and 105 fits of 15 separated scenes under the other settings
are status 4, 3 (below the iteration budget) or 2.

(c) Ordinary fits at the defaults are never status 3. The only status 3 among the 156 are the
three whose float64 maximum has a misclassified row beyond a linear predictor of 12 (13.8,
14.8, 22.3), as documented. Two more near-separated pairs with such a row (16.0, 15.3) are
status 3 as well, and six pairs whose slopes are 550 to 102,000 with standard errors larger
than the slopes are status 2.

Outputs of the centred design, 8 scenes (two and five predictors, offsets of 1e4, 1e6 and
1.5e7, rare outcome, a steep fit, column scales of 1e-4 and 1e4): `fitted` within 4.3e-7 of
statsmodels `predict`, marginal effects within 6.3e-7 of `get_margeff(at='overall')`, the
Pinkse, Kelejian-Prucha and Pinkse-Slade statistics and p-values within 1.2e-5 of spreg
`Probit(spat_diag=True)` on a ring. In the 153 status-0 scenes the first 64 fitted values are
within 2.4e-4 and the mean fitted value within 1.1e-7; the marginal effects are within 1.3e-3
except one coefficient of 0.0045 with a standard error of 0.93 (16% of itself, 7.8e-4 standard
errors).

By reading: the three certificates are sound. The column-extreme and threshold tests compare
raw float32 values, and the directions they certify (`x_j - value` with an intercept) move no
row the comparison does not put on the right side. The candidate test decides a sign only
beyond a bound that holds for any order of the additions, and `getDesign` fixes `x - mean`
before the products, so a regrouping compiler cannot break the bound.

### 3. Lag ML first-order check

No false status 3. My 41 lag ML scenes (the 24 high-fit ones, islands, unequal and mixed row
sums, six neighbours, large predictor mean) were each run 25 times: as they are, and with the
response, the predictor, or both multiplied by `2^-33`, `2^-17`, `2^17`, `2^33`, 1e-10, 1e-5,
1e5 and 1e10. All 1,025 fits are status 0. With the powers of two `rho` and `sigma2 / scale^2`
are identical to the unscaled fit; with the powers of ten (which round the inputs anew) `rho`
stays within one standard error of it.

The allowance. The test accepts `rho` within half a standard error plus four float32 steps of a
stationary point. Half a standard error costs at most `0.25 / n` of `sigma2` and an eighth of a
log-likelihood unit, so on ordinary data nothing material passes. The four steps matter only
where the standard error is a few steps: in 38 of my scenes it is 13 steps or more (the
allowance adds under a third of a standard error); in the two island scenes and the scene with
row sums 2 and 3 it is 3 to 3.9 steps, so a `rho` up to 1.8 standard errors off (about 8e-8)
would pass, worth 0.6% of `sigma2` and 1.7 log-likelihood units. The measured errors there are
0.06 to 0.12 standard errors. It cannot be made tighter without rejecting a float32 `rho` that
is as near as float32 allows.

### frc-regression-14. The 64-row near-exact fit: `sigma2` 33 times too large is one float32 step of the slope, not the float32 floor

- Severity: minor (an exact fit). CONFIRMED: GPU values, and the decomposition in float64 from
  the GPU's outputs.
- Where: `gpu-spatial-lag-maximum-likelihood.ts` class documentation lines 369 to 373;
  `docs/api/gpu-spatial-analysis/operators-regression.md` lines 830 to 833.
- The measurement reproduces: status 0, `sigma2` 7.78e-13 against 2.37e-14 (32.8 times), standard
  errors 5.7 times, log-likelihood 111 low, precision warning 0. Both documents say so in plain
  words, and say that nothing flags it.
- The stated cause ("residuals that are float32 rounding of `e0 - rho e1`") is not the floor.

  | What is rounded to float32 | `sigma2` over the float64 value |
  | --- | --- |
  | `rho` to the nearest float32 | 1.006 |
  | `e0` and `e1` (exact `rho`) | 1.04 |
  | `e0`, `e1`, the GPU's `rho` (one step off) and the product | 1.65 |
  | The GPU's residuals with their component in the span of `[1, x]` removed | 2.9 |
  | The GPU's residuals as published | 32.8 |

  91% of the published residual sum of squares lies in the span of `[1, x]`: the published slope
  is 1.00000012, one float32 step above 1, where the float64 slope is 1.0000000014 and the
  nearest float32 is 1 exactly; the intercept is -1.2e-7 against 0. The slope is `b0 - rho b1`
  of two float32 OLS fits, each rounded on its own. The residuals of a model with an intercept
  and `X` should be orthogonal to both; these are not.
- Fix. One regression of the published residuals on `[1, X]` (the OLS kernel the class already
  composes twice), with the correction applied to `b` and to the residuals as in the 2SLS
  refinement, brings this scene to about 3 times; or flag it: the regime is `c step^2 >= 1`
  with the curvature `c` and the step the first-order check already forms. At the least the
  two documents should name the slope, not the residual arithmetic.

### Not checked in this round

- The 2SLS changes of round three (`finish`, zero total weight) and the multiplication helper.
- Panel `hausman`; panels with more than one predictor in the high-fit and opposing-effect
  groups; `GPUSpatialPanelErrorMaximumLikelihood`.
- Probit with a `mask` or non-finite rows; separation certificates on a GPU other than the test
  GPU; whether a compiler other than this one keeps the error bound of the candidate test.
- Lag ML scales with the stochastic trace mode, tail extrapolation, the exact log-determinant
  mode and chunked weights.
- The full suites and `yarn build`.
