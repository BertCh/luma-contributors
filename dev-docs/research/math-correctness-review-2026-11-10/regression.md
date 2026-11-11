# Math correctness review: spatial regression (scope `regression`)

Date: 2026-11-10. Scope: `gpu-spatial-analysis/spatial-regression/` (57 files, 23,101 lines).
Paths are relative to `modules/analysis/src`. Read-only review. Nothing was built.
No spec was run. Five pure-Node scripts replicate kernel arithmetic; they are in
`/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-luma-contributors/9972295f-91de-4373-ba95-d2f799611a84/scratchpad/regression/`.
Python reference libraries (spreg, mgwr, numpy) are not installed on this machine, so every
statement about what a reference library does comes from its published source as I know it, or
from the fixtures in `modules/analysis/test/oracles`.

Two files were modified in the working tree by another session while I read
(`spatial-maximum-likelihood-nodes.ts`, `spatial-maximum-likelihood-wgsl.ts`). Neither looked half-edited.

## 1. Coverage

**Read in full (17):** `spatial-regression-solve.ts`, `ordinary-least-squares-statistics.ts`,
`incomplete-beta.ts`, `spatial-maximum-likelihood-wgsl.ts`, `spatial-maximum-likelihood-nodes.ts`,
`spatial-maximum-likelihood-parameters.ts`, `spatial-exact-log-determinant.ts`,
`spatial-weights-traces-parameters.ts`, `spatial-series-tail.ts`, `gwr-fit-wgsl.ts`,
`local-weighted-least-squares-wgsl.ts`, `geographically-weighted-regression-parameters.ts`,
`geographically-weighted-regression-nonstationarity-parameters.ts`, `gwr-student-t.ts`,
`generalized-gwr-selection.ts`, `multiscale-inference.ts`, `local-collinearity-wgsl.ts`.
Outside the scope, in full: `analysis-core/kernels/dense-solve-wgsl.ts`.

**Read in part (28).** Every WGSL kernel body and the class documentation were read; constructors,
validation and binding plumbing were skipped. Line ranges read:

| File | Lines read |
| --- | --- |
| `gpu-ordinary-least-squares.ts` | 120–1063 |
| `gpu-spatial-regression-diagnostics.ts` | 40–904 |
| `gpu-spatial-lag-maximum-likelihood.ts` | 150–1176 |
| `gpu-spatial-error-maximum-likelihood.ts` | 196–265, 400–1117 |
| `gpu-spatial-weights-traces.ts` | 60–200, 330–885 |
| `gpu-spatial-error-gm.ts` | 150–300, 370–1387 |
| `gpu-spatial-two-stage-least-squares.ts` | 175–262, 420–1123 |
| `gpu-instrumental-variables-covariance.ts` | 30–443 |
| `gpu-geographically-weighted-regression.ts` | 100–114, 270–399, 1040–1339 |
| `gpu-generalized-geographically-weighted-regression.ts` | 60–209, 228–722 |
| `gpu-multiscale-geographically-weighted-regression.ts` | 60–139, 270–827 |
| `gpu-geographically-weighted-regression-nonstationarity-test.ts` | 150–260, 736–1005 |
| `gpu-geographically-temporally-weighted-regression.ts` | 60–140, 236–300, 335–392 |
| `gpu-local-collinearity.ts` | 30–75, 200–333 |
| `gwr-grid-index.ts` | 20–312 |
| `gpu-probit-regression.ts` | 56–125, 236–672 |
| `gpu-spatial-panel-regression.ts` | 30–266 |
| `gpu-spatial-panel-error-maximum-likelihood.ts` | 20–207 |
| `gpu-spatial-panel-lag-maximum-likelihood.ts` | 28–308 |
| `panel-diagnostics.ts` | 18–146 |
| `gpu-spatial-regime-regression.ts` | 24–200 |
| `gpu-restricted-regime-regression.ts` | 28–383 |
| `regime-moments.ts` | 20–238 |
| `ordinary-sur-nodes.ts` | 30–373 |
| `spatial-sur-nodes.ts` | 20–226 |
| `gpu-seemingly-unrelated-regression.ts` | 20–100 |
| `gpu-spatial-lag-impacts.ts` | 60–110, 170–238 |
| `geographically-weighted-regression-model-result.ts` | 25–97 |

**Searched with grep only (6):** `gpu-multiscale-geographically-temporally-weighted-regression.ts`
(the update rule at 306–333 and the score at 370), `gwr-golden-section-nodes.ts` (interior points,
rounding, tolerance at 263–268, 329–373), `ordinary-least-squares-parameters.ts`,
`spatial-regression-diagnostics-parameters.ts`, `spatial-error-gm-parameters.ts`,
`spatial-two-stage-least-squares-parameters.ts`.

**Not opened (6):** `chunked-weights-kernels.ts`, `regression-row-tiles.ts`,
`spatial-error-gm-model-result.ts`, `spatial-two-stage-least-squares-model-result.ts`,
`spatial-maximum-likelihood-model-result.ts`, `index.ts`. These are not verified.

## 2. Findings

| ID | Severity | Status | File:line | What the kernel computes | What the reference says | Concrete input and wrong output | Suggested fix |
| --- | --- | --- | --- | --- | --- | --- | --- |
| regression-01 | wrong-edge-case | CONFIRMED (script) | `gpu-probit-regression.ts:462`, `:467`, `:491` | A line-search step is accepted only if the largest `abs(x'b)` over all rows stays at or below 12. Steps shrink toward that bound and the step-size test then reports convergence. | Probit maximum likelihood has no bound on the linear predictor (spreg `Probit`). The class documentation (`:98-100`) says a fit pinned on the bound is status 4. | 4000 rows, `x ~ N(0,1)`, `y ~ Bernoulli(Phi(0.2 + 2x))`, plus one row `x = 9, y = 1`. Result: status 0, slope 1.32, log-likelihood −1330. The maximum is slope 2.05, log-likelihood −1232. | Drop the bound from the acceptance test (the 1e-7 probability floor already keeps the likelihood finite), or report status 4 when the accepted step was limited by the bound. |
| regression-02 | precision | CONFIRMED (script) | `gpu-restricted-regime-regression.ts:176-197` | Rebuilds raw cross-products `C + n·mean·mean'` from the centred regime moments, then solves the raw normal equations in float32 with norm scaling only. | Centred least squares (what `GPUOrdinaryLeastSquares` and the sibling `GPUSpatialRegimeRegression` do). | Two regimes, 200 rows each, common slope 0.5. Predictor "year" 2000–2020: slope error 0.8%. Elevation 1500 ± 3: 11%. UTM northing 5.2e6 ± 5 km: 31%. Status 0 in all three. | Solve in a centred basis (grand-mean centring keeps the restricted structure). Apply the OLS pivot limit. |
| regression-03 | precision | CONFIRMED (script) | `gpu-spatial-error-maximum-likelihood.ts:676-688`, `:956-964` | `ee(lambda) = yy − 2b'r + b'Ab` from float32 Gram matrices. The same value gives `sigma2`, the likelihood and the derivative whose sign changes select `lambda`. | spreg `ML_Error` computes the filtered residuals row by row in float64. The sibling lag estimator uses per-row OLS residuals. | n = 2000, one predictor. At R² = 0.9975 the relative error of `ee` is 1e-4 (log-likelihood noise 0.03 to 0.1). At R² = 0.9999 it is 0.2% (noise 2 log-likelihood units). At 1 − R² = 6e-6 it is 4% (noise 40 units). | Accumulate the Gram matrices with `y` replaced by its first-stage OLS residual (the minimum over `b` is unchanged), or at least take the final `sigma2` from a per-row filtered-residual pass as `GPUSpatialErrorGM` does. |
| regression-04 | precision | CONFIRMED (script) | `gpu-spatial-panel-lag-maximum-likelihood.ts:220-221`, `:226-227` | Raw, uncentred, unscaled Gram of the quasi-demeaned design, bare `choleskyFactor`. | spreg `Panel_RE_Lag` in float64. | 16 entities × 16 periods, phi 0.3, slope 0.5. Predictor "year" 2000–2020: slope error −1.6%. Elevation 1500 ± 3: −6.7%. Income 50000 ± 500: −0.6%. Status 0. | Centre and equilibrate as OLS does, or factor with the shared equilibrated Cholesky and a pivot ratio. |
| regression-05 | wrong-edge-case | CONFIRMED (by hand) | `gpu-geographically-temporally-weighted-regression.ts:369-377` | Publishes `AICc = aic + 2·tr·(tr+1)/(n − tr − 1)` whenever `n > 0`. | The correction is defined only for `n − tr − 1 > 0`. GWR voids the candidate when `n − 2 − tr <= 0` (`gpu-geographically-weighted-regression.ts:1137-1139`). | n = 100, tr(S) = 99.5 (a very small Gaussian bandwidth): the correction is −39,999. A caller that profiles `(bandwidth, timeScale)` by smallest AICc, which is the documented use (`:143-144`), selects the degenerate bandwidth. | Write NaN when `count − trace − 1 <= 0`. |
| regression-06 | inconsistent | CONFIRMED (by hand) | `gpu-geographically-temporally-weighted-regression.ts:369-371` against `gpu-geographically-weighted-regression.ts:1141` | Two AICc formulas in the same summary slot. GWR: `−2llf + 2n(tr+1)/(n−tr−2)`. GTWR: `−2llf + 2(tr+1) + 2tr(tr+1)/(n−tr−1)`. | The first is mgwr `aicc`; the second is mgtwr `aicc`. Each is pinned to its own library. | n = 200, tr = 60: penalties 176.8 and 174.7. With `timeScale = 0` the two contributors fit the same model and report AICc values 2.15 apart. GWR against GTWR by AICc is the standard comparison, and it is biased toward GTWR. | Use one formula in the shared summary slot (mgwr's). Keep the mgtwr value under its own name if parity is wanted. |
| regression-07 | wrong-edge-case | CONFIRMED (script) | `generalized-gwr-selection.ts:84` with `gpu-generalized-geographically-weighted-regression.ts:312` | Logistic bandwidth selection takes the leave-one-out deviance from the float32 fitted mean and rejects `mean >= 1`. One rejected row voids the candidate bandwidth. | The Bernoulli deviance is finite for every finite linear predictor: `2·log(1 + exp(∓eta))`. | The float32 mean is exactly 1 for `eta >= 17`. One confident leave-one-out prediction voids the candidate. `eta = −80` is accepted. The fit itself allows `abs(eta) <= 80`. | Compute the deviance from the linear predictor, not from the rounded mean. |
| regression-08 | inconsistent | CONFIRMED (read) | `gpu-ordinary-least-squares.ts:889`, `gpu-spatial-lag-maximum-likelihood.ts:1130`, `gpu-spatial-error-maximum-likelihood.ts:1078`, `geographically-weighted-regression-model-result.ts:83` | Four parameter counts for AIC: OLS `p`; lag ML `p + 1` (counts rho); error ML `p` (does not count lambda); GWR `tr(S) + 1` (counts the variance). | Each matches its reference (spreg, mgwr) and the class documentation says so. | Lag against error on the same data: the error model gains 2 AIC units (and `ln n` Schwarz units) from the counting rule alone. | Keep the reference values. Add a common-convention count to the shared model result, or state the rule in `docs/api`. |
| regression-09 | inconsistent | CONFIRMED (by hand) | `panel-diagnostics.ts:142` against `gpu-spatial-regression-diagnostics.ts:845` | The panel LM tests use `sigma2 = RSS/(n − p)` (OLS summary slot 3). The cross-section LM tests use `RSS/n`. | Both follow spreg (`panel_LMerror` uses `ols.sig2`; `LMtests` uses `sig2n`). | Same residuals and block-diagonal weights, n = 48, p = 3: the panel LM-error is 0.879 times the cross-section one. | Document the divisor in `docs/api/gpu-spatial-analysis/coverage-statistics.md:82`, or use one divisor. |
| regression-10 | inconsistent | CONFIRMED (read) | 20 call sites of `getCholeskyWGSL`; `gpu-spatial-panel-lag-maximum-likelihood.ts:212-214`, `:306-307` | No file uses the shared `analysis-core/kernels/dense-solve-wgsl.ts`. Pivot policy differs by contributor. See section 4. | Contributor contract rule 11. | Near-collinear predictors: OLS and the GWR family report singular; regimes, SUR, probit and panel lag return status 0. | See section 4. |
| regression-11 | inconsistent | CONFIRMED as two code paths; numeric gap SUSPECTED | `gwr-fit-wgsl.ts:273-337` against `gpu-local-collinearity.ts:283-325` | The same mgwr local condition number by two methods: Gram matrix plus Jacobi eigenvalues in GWR, streaming QR plus one-sided Jacobi in `GPULocalCollinearity`. | `local-collinearity-wgsl.ts:16-17` states that the Gram route resolves only a few thousand in float32. | Not run. Expect disagreement above a condition number of about 1e3. mgwr's warning level is 30, so the practical effect is small. | Have GWR call `getDesignSingularValueWGSL`. |
| regression-12 | doc-mismatch | SUSPECTED | `gwr-fit-wgsl.ts:233-255` | Local R² uses the residuals of local model `i` at every neighbour `j`. | mgwr `localR2` uses the GWR residuals `y_j − yhat_j` of each row's own fit. The form here is the spgwr one. | Any dataset: the two definitions differ. | Say which definition in the output documentation (`gpu-geographically-weighted-regression.ts:110`). Needs an mgwr run to confirm. |
| regression-13 | doc-mismatch | SUSPECTED | `gpu-multiscale-geographically-weighted-regression.ts:273-274` | Backfitting starts every surface at the global OLS coefficients. The comment says "as mgwr does". | mgwr `multi_bw` starts from a single-bandwidth GWR fit. | Same fixed point if it is unique. Iteration counts and results at the iteration limit differ. | Correct the comment. |
| regression-14 | doc-mismatch | CONFIRMED (read) | `gpu-spatial-regression-diagnostics.ts:772` | Rejects `n <= predictorCount + 2`. | The status documentation (`spatial-regression-diagnostics-parameters.ts:65`, class doc `:192`) says `n <= predictorCount + 1`. OLS accepts `n = predictorCount + 2`. | n = predictorCount + 2: OLS fits, diagnostics report `TOO_FEW_ROWS`. | Change the test to `n > P` or the documentation. |

### regression-01. Probit returns status 0 with an estimate pinned on the linear-predictor bound

- The decide kernel accepts the first halved step whose likelihood did not drop **and** whose largest
  `abs(eta)` is at most 12 (`:462`).
- When the maximum-likelihood estimate has a row beyond 12, each iteration accepts a smaller step, and
  the gap to the bound at least halves each time.
- The convergence test is on the step size: `largest <= tolerance * (1 + magnitude)` (`:467`). It fires
  after about ten iterations, well inside the 24 halvings and 64 iterations.
- The finish kernel flags separation only when the code is not 0 (`:491`). So the fit is published
  as converged, with standard errors, fitted values and marginal effects from the pinned estimate.
- One correctly classified, high-leverage row is enough. Such a row has almost no effect on the true
  likelihood, but here it caps every coefficient.
- Script `probit-bound.mjs` replicates the control flow in float64 with the kernel's `erfcValue` and
  probability floor. It is not a GPU run. The step logic does not depend on float32.
- No test covers this. The spreg oracle scenes (`test/oracles/statistics-upstream.py`) have moderate
  linear predictors.

### regression-02. Restricted regimes undo the centring of the regime moments

- `regime-moments.ts` computes means and centred cross-products per regime. That is the accurate part.
- `rawMoment` (`:176-185`) then adds `count * mean_a * mean_b` back in float32, and `fit_*`
  (`:188-197`) solves the raw system `[regime intercepts, predictors]`.
- The Cholesky pivot of a predictor after the intercepts is about `(spread/mean)^2`. Float32 holds
  it to `6e-8 / pivot`.
- The class documentation of the sibling (`gpu-spatial-regime-regression.ts:50-51`) says centring
  "avoids cancellation from large coordinate-like predictor means". The restricted class has the
  cancellation.
- The covariance, the Chow F statistic and the Wald stability tests are computed from the same
  solve, so they inherit the error.
- The oracle scenes use standard-normal predictors, so no test sees this.

### regression-03. Error ML residual sum of squares by cancellation

- Columns are centred, so the cancellation is driven by R² alone: the measured relative error of
  `ee` is about `2e-7 / (1 − R²)` (script `error-ml-gram.mjs`, ring lattice, lambda 0.5).
- The search (`getParameterSearchWGSL`) picks `lambda` from sign changes of the derivative of the
  same noisy quantity, then compares candidates by the noisy objective.
- This applies to `GPUSpatialPanelErrorMaximumLikelihood` too, which delegates.
- R² above 0.997 is unusual for socio-economic data and common for physical or engineered
  relationships. I rate it `precision`, not `wrong-result`.

### regression-05 and regression-06. GTWR AICc

- The two formulas differ by `2n(k+1)/(n−k−2) − 2(k+1)(n−1)/(n−k−1)` with `k = tr(S)`. The gap grows
  with `k/n`.
- The mgtwr oracle (`test/oracles/gtwr-mgtwr-oracle.py:91`) pins the GTWR value, so a change needs
  a decision, not just an edit.

## 3. Verified correct

Each line names what I compared the kernel against. "spreg" and "mgwr" mean the library's formulas.

- **`GPUOrdinaryLeastSquares`**: centred, correlation-scaled solve; slopes, intercept and its
  variance `sigma2 (1/n + xbar' C^-1 xbar)`; `sigma2 = RSS/(n − p)`; R², adjusted R²; Gaussian
  log-likelihood with `RSS/n`; AIC and Schwarz with `p`; Jarque–Bera `n/6 (S² + (K−3)²/4)` with
  p `exp(−JB/2)`; Koenker–Bassett `n R²_aux`; Breusch–Pagan `0.5 R²_aux n (K − 1)` (I re-derived it
  from spreg's `0.5 · ESS` of `e²/s² − 1`); packed-moment index; ridge on slopes only. Against spreg
  `OLS` and `diagnostics`.
- **Chi-square, Student t, incomplete beta helpers** (`ordinary-least-squares-statistics.ts`,
  `incomplete-beta.ts`, `gwr-student-t.ts`): half-integer log-gamma recurrence, series and Lentz
  fraction, `I_x(df/2, 1/2)` with `x = df/(df + t²)`, the `Gamma(a + 1/2)/Gamma(a)` asymptotic
  series to the sixth term. Against Numerical Recipes and the standard expansions.
- **`GPUSpatialRegressionDiagnostics`**: LM-lag, LM-error, both robust forms, LM-SARMA (the identity
  `LM-lag + RLM-error = RLM-lag + LM-error` holds); `T = tr(W'W + WW)` with one-way links;
  `J = (WXb)'M(WXb)/sigma2 + T`; `sigma2 = e'e/n`; Moran's I of residuals with the Cliff–Ord mean
  and variance (I expanded `tr(MWMW')`, `tr(MWMW)`, `tr(MW)` term by term against the Gram traces);
  two-sided normal p; row re-standardization on the fitted subset. Against spreg `LMtests`,
  `MoranRes`.
- **`GPUSpatialLagMaximumLikelihood`**: concentrated likelihood and its derivative; `b = b0 − rho b1`;
  `sigma2 = u'u/n`; full log-likelihood; `pr2`, `pr2_e`; information matrix with the `sigma2` row
  eliminated (`tr2 + tr3 + wpy'wpy/sigma2 − 2 tr1²/n`); intercept variance mapped back from the
  centred model; Horner index parity of the `(I − rho W)^-1 Xb` chain. Against spreg `ML_Lag`.
- **Trace series** (`spatial-maximum-likelihood-wgsl.ts`, `spatial-weights-traces-parameters.ts`):
  `ln|I − rho W| = −sum t^q c_q/q`, `tr1 = s sum t^(q−1) c_q`, `tr2 = s² sum (m+1) t^m c_(m+2)`, the
  scaled parameter `t = rho·s`, the truncation bound and the series limit. By hand.
- **Tail extrapolation** (`spatial-series-tail.ts`): the three closed-form tails match the stated
  geometric model. It is an approximation and off by default.
- **Exact log-determinant** (`spatial-exact-log-determinant.ts`): the symmetrizing similarity
  `S = D^(1/2) W D^(−1/2)`; `sum log(1 − rho·lambda)`; `tr1`, `tr2` from eigenvalues.
- **`GPUSpatialWeightsTraces`**: exact unit probes and Hutchinson probes; power sums; standard
  errors; Gram traces `tr(W'WW)` (scale `s³`) and `tr((W'W)²)` (scale `s²`). By hand.
- **Probe pass for `tr3`** (`spatial-maximum-likelihood-nodes.ts`): `||W A z||²`, buffer parity.
- **`GPUSpatialErrorMaximumLikelihood`**: the Gram identity `G0 − lambda(G1 + G1') + lambda² G2`; the
  envelope-theorem derivative; `Var(lambda) = 1/(tr2 + tr3 − 2 tr1²/n)`; coefficient covariance and
  intercept mapping. Against spreg `ML_Error`. The arithmetic precision is regression-03.
- **`GPUSpatialErrorGM`**: Kelejian–Prucha moments `G`, `g` and the projection of `sigma2 >= 0`; the
  `GM_Error_Hom` moments with `A1`, `A2`; the four trace terms of `psi` (each re-derived); the
  weighted second step; `Var(lambda) = (j' psi^-1 j)^-1 / n`; filtered regression and its intercept
  variance. Against spreg `GM_Error`, `GM_Error_Hom`.
- **`GPUSpatialTwoStageLeastSquares`**: `delta = (Z'PZ)^-1 Z'Py` on centred columns; `sigma2 = RSS/n`;
  intercept variance; first-stage F and partial R² from sequential Cholesky sums of squares;
  Anselin–Kelejian test (I re-derived the centred form of `u'WZ (Z'PZ)^-1 Z'W'u`). Against spreg
  `GM_Lag`, `akTest`.
- **`GPUInstrumentalVariablesCovariance`**: HC0 and HC1 sandwich, intercept influence
  `u_i (1/n − mean' A h_i)`, HAC as `s'Ks`. By hand.
- **`GPUSpatialLagImpacts`**: direct `beta tr((I − rho W)^-1)/n`, total `beta 1'(I − rho W)^-1 1/n`.
  Caveat that the documentation states: the series is truncated with no flag. At `rho·s = 0.98` and
  `Q = 128` the total is 7% low (`0.98^129/0.02 = 3.7` of 50); at 0.99 it is 27% low.
- **`GPUGeographicallyWeightedRegression`**: Gaussian `exp(−0.5 (d/h)²)` and bisquare
  `(1 − (d/h)²)²` kernels; adaptive bandwidth as the k-th nearest distance counting the row itself;
  focal-centred design so `S_ii = (A^-1)_00`; `CCT = A^-1 X'W²X A^-1` including the uncentred
  intercept direction; AICc; `sigma2 = RSS/(n − tr S)`; adjusted alpha `alpha·k/tr(S)`; critical t
  with `n − 1` degrees; log-likelihood. Against mgwr `GWR`, `GWRResults`. The pivot limit reported
  missing as V1 in the 2026-11-09 review is now present (`local-weighted-least-squares-wgsl.ts:89-94`).
- **Golden-section search**: interior points, round-half-even, comparison rule, as mgwr
  `golden_section` (grep only).
- **`GPUGeneralizedGeographicallyWeightedRegression`**: Newton step for the canonical Poisson and
  logit links with offset; likelihoods; step halving; coefficient mapping from the centred, scaled
  basis; sandwich covariance `I^-1 (sum w² v x x') I^-1` and influence `x_i' I^-1 x_i v_i w_ii`
  (equal to mgwr's `CCT` and `influ` for GLMs); Poisson and Bernoulli deviance. The selection
  criterion is leave-one-out deviance, not mgwr's AICc; the documentation says so.
- **`GPUMultiscaleGeographicallyWeightedRegression`**: Gauss–Seidel order across variables;
  partial residual; univariate smoother; exact leave-one-out score; conditional standard error
  `sqrt(sigma2 sum w²x²)/sum w x²`; full inference by propagating `d beta/d y` (standard errors,
  per-variable effective parameters, `sigma2 = RSS/(n − tr S)`). Against mgwr `multi_bw` and the
  MGWR inference of Yu et al.
- **`GPUGeographicallyWeightedRegressionNonstationarityTest`**: population standard deviation;
  pseudo p `(g + 1)/(P + 1)` with `>=`; permuting `(X, y)` is equivalent to permuting coordinates.
  Differences from mgwr are documented.
- **`GPULocalCollinearity`**: VIF as the inverse diagonal of the weighted correlation matrix;
  mgwr's design condition number with weights `w` (not `sqrt(w)`); Belsley variance proportions.
  Against mgwr `local_collinearity`.
- **GTWR**: space-time distance `sqrt(ds² + tau dt²)`, kernels, hat diagonal. Against mgtwr. The
  AICc is regression-05 and regression-06.
- **`GPUProbitRegression`**: Fisher score and expected information; coefficient and standard-error
  mapping from the scaled basis; average marginal effects (spreg `scalem = 'phimean'`); Pinkse,
  Kelejian–Prucha and Pinkse–Slade statistics term by term. Against spreg `Probit`, `sp_tests`.
  Standard errors use expected information; spreg uses the observed Hessian. The documentation
  says so. The line search is regression-01.
- **`GPUSpatialPanelRegression`**: block weights, within transform, degrees `n − N − k − 1`,
  first-stage F rescale.
- **`GPUSpatialPanelErrorMaximumLikelihood`**: I checked that the implicit intercept column of the
  delegated estimator is exactly orthogonal to within-demeaned columns for any `W`, so the result
  equals spreg's no-constant `Panel_FE_Error`; `Var(lambda)` on `I_T ⊗ W` equals spreg's.
- **`GPUSpatialPanelLagMaximumLikelihood`**: analytic `phi² = within/((T − 1)·between)` (I derived
  it from the concentrated likelihood); rho objective `NT/2 log(sigma2) − T log|I − rho W|`;
  information `T(tr2 + tr3) + wpy'wpy/sigma2`. Arithmetic precision is regression-04.
- **Panel LM tests** (`panel-diagnostics.ts`): formulas match spreg `panel_LMerror`, `panel_rLMlag`,
  `panel_rLMerror`. The divisor is regression-09.
- **`GPUSpatialRegimeRegression`**: per-regime centred OLS.
- **`GPURestrictedRegimeRegression`**: Chow F, covariance `SSE/(n − parameters)`, Wald contrasts
  against regime 0. Formulas are right; the solve is regression-02.
- **`GPUSeeminglyUnrelatedRegression`**: one-step FGLS with `Sigma = E'E/n`; the back-transform of
  the covariance to raw intercepts; spatial mode: filter, intercept rescale by `1/(1 − lambda)`.
  Against spreg `SUR`, `SURerrorGM`.

## 4. Standards consistency

**Dense solver.** Contract rule 11 names `analysis-core/kernels/dense-solve-wgsl.ts`. Nothing in
`modules/analysis/src` imports it. The regression family uses its own
`spatial-regression-solve.ts` `getCholeskyWGSL` at 20 call sites.

- The two are numerically the same factorization. `factorDenseCholesky_N(matrix, size, true, 1e-6)`
  does what `factorEquilibrated_P` plus the `>= 1e-3` pivot test does
  (`local-weighted-least-squares-wgsl.ts:73-96`). The shared one also separates `NONFINITE`,
  `SINGULAR` and `ILL_CONDITIONED`; the local one returns one boolean.
- Pivot policy is not uniform:
  - relative pivot limit 1e-3: OLS (`gpu-ordinary-least-squares.ts:786-791`), GWR, generalized GWR,
    the nonstationarity test, GTWR;
  - positivity only: diagnostics, lag ML information, error ML, GM, 2SLS, IV covariance, probit,
    both regime regressions, SUR, MGWR global fit, panel lag ML. Several of these report a
    condition number and a precision warning instead (ML, GM, 2SLS). Regimes, SUR, probit and
    panel lag report nothing.
- `gpu-spatial-panel-lag-maximum-likelihood.ts` carries a private Gauss–Jordan inverse (`:306-307`)
  and a private LU log-determinant (`:212-214`), both with an absolute pivot threshold of 1e-10.
  The shared `factorDenseLU_N` gives both with a relative test.
- Suggested direction: move `getCholeskyWGSL` callers to `getDenseCholeskyWGSL` with
  `equilibrate = true` and one `minimumPivotRatio`, and delete the private inverse.

**Centring.** OLS, diagnostics, both ML estimators, GM, 2SLS, IV covariance, SUR, probit and the
GWR family centre before forming normal equations. `GPURestrictedRegimeRegression` (regression-02)
and `GPUSpatialPanelLagMaximumLikelihood` (regression-04) do not.

**Other private copies.**

- Eigenvalues: `getSymmetricEigenvalueRangeWGSL` (`spatial-regression-solve.ts:107-160`, 16 sweeps,
  threshold 1e-12) beside the shared `analysis-core/kernels/symmetric-eigen-wgsl.ts`. Not compared
  numerically.
- One-sided Jacobi twice: `getBelsleyConditionNumberWGSL` (16 sweeps, threshold 1e-7) and
  `getDesignSingularValues_n` (64 sweeps, threshold 1e-6).
- Chi-square survival: a private series and fraction in OLS (`gpu-ordinary-least-squares.ts:944-995`),
  closed forms in diagnostics, and the shared `incomplete-gamma-wgsl.ts` in restricted regimes and
  panel lag. Same function, three routes.
- Row-standardization tolerance `1e-5` is written as a literal in two places
  (`gpu-spatial-panel-lag-maximum-likelihood.ts:173`, `spatial-sur-nodes.ts:115`).
- Adaptive bandwidth factor: the named constant `ADAPTIVE_FACTOR` in GWR, the literal `1.00001` in
  generalized GWR (`gpu-generalized-geographically-weighted-regression.ts:357`, `:372`).
- `ln(2 pi)` is a literal in three kernels and a named constant in a fourth.

**Conventions that differ between siblings.**

- AIC parameter count: regression-08. AICc formula: regression-06.
- LM-test variance divisor: regression-09.
- Local R²: regression-12.
- Local condition number: regression-11.
- Default kernel: bisquare in GWR, Gaussian in GTWR and in the generalized and multiscale classes.
- Gaussian cutoff: GWR and generalized GWR accept `gaussianCutoff`. The nonstationarity test refits
  with the untruncated Gaussian (`gpu-geographically-weighted-regression-nonstationarity-test.ts:740`).
  The omitted weights are below `exp(−12.5)` at the recommended cutoff, so the effect is negligible.
- Search bound: ML uses `min(0.99, seriesLimit)/||W||_inf`; GM uses `0.99/||W||_inf`; panel lag
  uses a fixed `±0.999` and requires row-standardized weights.

**Shared numerics that are used.** Philox with registered purposes (trace probes, the
nonstationarity shuffle), `ERFC_VALUE_WGSL`, Kahan tile merges, `GPUSpatialWeightsTranspose`,
`getGPUSphereRadius` from the spatial context. Great-circle distance comes from
`geometry-measures/geodesic-wgsl.ts` (`geodesicHaversineDistance`), not from
`analysis-core/kernels/sphere-distance-wgsl.ts`.

## 5. luma.gl official implementations

- Published `@luma.gl/graph` 10.0.0-alpha.10 has no regression, least-squares or dense-solver entry
  point (I searched every `dist/<entry>/index.d.ts`). There is nothing official to delegate the
  estimators to.
- Delegations that exist and look right:
  - `GPUSort` from `@luma.gl/graph/gpu-core` for the regime sort (`regime-moments.ts:121-128`), with
    `keyBits` sized for `regimeCount + 1` keys.
  - `GPUGridIndex` from `@luma.gl/graph/gpu-core` for the GWR family (`gwr-grid-index.ts:156-165`).
    Positions are normalized to `[0, 1]` because the published class takes literal bounds.
- One convention to watch: the query side recomputes the cell with the local `getBinIndex` on raw
  coordinates (`gwr-grid-index.ts:204-217`), while the index bins the normalized positions. The two
  can differ by one unit of rounding at a cell edge. The range is padded by `1.0001·bandwidth` plus
  `2.4e-7·abs(origin)` and the ring walk keeps one cell of slack, which covers it. I did not test
  the edge case numerically.
- The in-cell order of the published index depends on atomics, so the local code re-sorts rows
  inside each cell to keep float sums reproducible. That is a local need, not a disagreement.
- No private path or copied luma.gl source is used in this directory.

## 6. Test oracles

**Pinned by an independent reference library (fixtures under `test/oracles/fixtures`):**

- spreg 1.9.1 (`statistics-spatial-regression.py`): OLS and its diagnostics, LM tests and Moran,
  `ML_Lag`, `ML_Error`, `GM_Lag`, `GM_Error`, `GM_Error_Hom`, lag impacts.
- spreg 1.8.3 (`statistics-upstream.py`, `statistics-advanced.py`): `Probit`, `OLS_Regimes`, `SUR`,
  `SURerrorGM`, `Panel_FE_Error` and the three panel LM tests, `Panel_RE_Lag`, `Panel_FE_Lag`,
  `panel_Hausman`, robust `TSLS`.
- mgwr 2.2.1: GWR (`statistics-gwr.py`, `statistics-upstream.py`), `local_collinearity`, MGWR full
  inference (`statistics-advanced.py`), the nonstationarity test (`gwr-nonstationarity-mgwr.py`).
- mgtwr: GTWR (`gtwr-mgtwr-oracle.py`), including its AICc.
- statsmodels: generalized GWR local fits as weighted GLMs (`adaptive-generalized-gwr.py`).
- `GPUSpatialWeightsTraces`: `statistics-spatial-weights-traces.py`.

**Pinned only by a copy of the same formula:**

- GWR local R² (`geographically-weighted-regression-oracle.ts:649-676` is the kernel formula in
  TypeScript; the mgwr reference file has no local R²).
- GWR critical t and Student t (float64 mirror of the same algorithm).
- Tail extrapolation (CPU mirror).
- Generalized GWR bandwidth selection score (I found no independent reference for the
  leave-one-out deviance).

**What the oracles cannot see:**

- Every oracle scene I looked at uses predictors and responses of order 1. None has a large mean
  relative to its spread, none has R² near 1, and none has a linear predictor beyond 12. That is
  why regression-01 to regression-04 pass the test suite.
- The default large-n stochastic-trace regime of the ML estimators is still unpinned (N2 of the
  2026-11-09 review; not re-checked).

## Needs a second opinion

- **regression-01**: confirm on a GPU. My replica is float64.
- **Hausman direction** (`gpu-spatial-panel-lag-maximum-likelihood.ts:300`): the statistic uses
  `Var_RE − Var_FE`, as spreg `panel_Hausman` and Elhorst do, and rejects a negative value. The
  classical form uses `Var_FE − Var_RE`. The documentation states the spreg convention, so I did
  not list it as a finding. Someone with spreg at hand should decide whether to follow it.
- **Panel fixed-effects log-likelihood** carries spreg's extra `−NT/2·ln(NT)` term
  (`:236`). Documented. It is not a likelihood that can be compared with any other model.
- **regression-12 and regression-13** rest on my memory of mgwr source. Needs a run.

## Resolution (2026-11-10)

Paths are relative to `modules/analysis/src/gpu-spatial-analysis/spatial-regression` unless they
start with `docs/` or `test/`. Specs are in
`modules/analysis/test/gpu-spatial-analysis/spatial-regression`. Line numbers are those of the
working tree after the pass. Expected values in the new specs are float64 computations of the
reference formula from the raw float32 inputs (least squares by normal equations with partial
pivoting, likelihood maxima by golden section, closed-form ring log-determinants), never a copy
of the kernel arithmetic.

| ID | Outcome | What changed | Spec that pins it | Behaviour change a caller sees |
| --- | --- | --- | --- | --- |
| regression-01 | fixed | `gpu-probit-regression.ts:467-483` (the decide kernel records a step shortened by the bound; a shortened step below `tolerance` whose full Fisher step is not is status 4), class documentation `:98-106`, `docs/api/gpu-spatial-analysis/coverage-statistics.md` (probit statuses) | `math-review-probit-bound.spec.ts` (failed on the old kernel on the GPU: status 0, slope 1.32 against 2.07) | A fit whose maximum puts a row beyond a linear predictor of 12 was status 0 with pinned coefficients and standard errors; it is now status 4 with NaN coefficients. The existing scene `coverage-estimators.spec.ts` "early overshoot" (slope 3.5, maximum at max linear predictor 12.06) was such a fit accepted with a tolerance of 0.3; it now uses a slope of 3.0 and a tolerance of 2e-2 |
| regression-02 | fixed | `gpu-restricted-regime-regression.ts:177-275` (both nested models solved about shifts: regime means with a varying intercept, pooled means of the response and of common-slope predictors with a common intercept; coefficients and covariance mapped back with `T V T'`), `:336-371` (residuals in the shifted basis; third row sum), `:406` (Chow numerator is the direct sum of squared fit differences), `:457-481` (Wald contrasts formed in the shifted basis; joint test recombines intercept and slope contrasts of a regime) | `math-review-restricted-regimes.spec.ts` (year, elevation, UTM northing, varying slopes, common intercept) | Coefficients, covariance, Chow F and Wald tests of a model with a large-mean predictor change by the former error (0.8% to 31% in the slope on the report's scenes); order-1 scenes agree to float32 rounding. A varying slope under a common intercept was left in the raw basis in the first pass; that was wrong and is fixed in "Review follow-up" below |
| regression-03 | fixed | `gpu-spatial-error-maximum-likelihood.ts:514-521` (the response column of the Gram passes is the first-stage OLS residual), `:623-698` (new `ols-tiles`, `ols-merge`, `ols-solve` nodes), `:1064-1070` (first-stage slopes added back) | `math-review-error-ml-high-fit.spec.ts` (R^2 0.9, 0.9999, `1 - R^2 = 6e-6`, and 0.9999 with a predictor mean of 1500) | `sigma2`, the log-likelihood and the selected `lambda` change at high R^2 by the former noise; three more command nodes. `GPUSpatialPanelErrorMaximumLikelihood` inherits it |
| regression-04 | fixed | `gpu-spatial-panel-lag-maximum-likelihood.ts:156-181` (pooled means node), `:217-226` (panel packed minus the pooled means, lag shifted consistently), `:259`, `:265`, `:287` (equilibrated Cholesky with the pivot limit), `:274-279` and `:288-300` (intercept and its covariance row mapped back), `:341` (fitted values) | `math-review-panel-lag-centering.spec.ts` (year, elevation, income, and the covariance congruence under a shift) | Slopes, `rho`, `phi`, `sigma2` of a model with a large-mean predictor change by the former error (0.6% to 6.7% in the slope); a near-singular system is status 2 |
| regression-05 | fixed | `gpu-geographically-temporally-weighted-regression.ts:380-391` (AICc is NaN unless `n - tr(S) - 2 > 0`) | `math-review-gtwr-aicc.spec.ts` | A degenerate bandwidth published a finite, very negative AICc; it is now NaN |
| regression-06 | fixed, converged on the GWR (mgwr) formula, the sibling in this library | same lines; class documentation `:148-157`; `docs/api/gpu-spatial-analysis/operators-regression.md` (GTWR summary) | `math-review-gtwr-aicc.spec.ts` (float64 Gaussian GWR at `timeScale = 0`), `gtwr-mgtwr.spec.ts` (mgwr formula on mgtwr's `llf` and trace, and the gap to mgtwr's `aicc`) | The GTWR summary AICc is larger than before by `2 n (k + 1) / (n - k - 2) - 2 (k + 1) (n - 1) / (n - k - 1)`, `k = tr(S)` (2.15 at `n = 200`, `k = 60`). It no longer equals mgtwr's `aicc`; the oracle fixture is unchanged and the spec states the gap |
| regression-07 | fixed, from the linear predictor (the report's fix) instead of a clamped mean, because a clamp caps the deviance of a confident miss at about 32 | `generalized-gwr-selection.ts:66-67`, `:86-94`; `gpu-generalized-geographically-weighted-regression.ts:116-123` (optional internal prop `fittedScale`), `:434` | `math-review-logistic-selection.spec.ts` (float64 leave-one-out local logistic fits) | A candidate bandwidth with a leave-one-out linear predictor of 17 or more had a NaN score and could not be selected; it now has its exact deviance |
| regression-08 | fixed (documentation, formulas unchanged) | `gpu-ordinary-least-squares.ts:149-154`, `spatial-maximum-likelihood-parameters.ts:59-64`, `geographically-weighted-regression-model-result.ts:51-55`, `docs/api/gpu-spatial-analysis/operators-regression.md` (OLS criteria) | none (words) | none |
| regression-09 | fixed (documentation, formulas unchanged) | `gpu-spatial-regression-diagnostics.ts:157-162`, `gpu-spatial-panel-error-maximum-likelihood.ts:50-54`, `panel-diagnostics.ts:32-37`, `docs/api/gpu-spatial-analysis/coverage-statistics.md` (panel diagnostics) | none (words) | none |
| regression-10 | pivot policy fixed; adoption of the shared `dense-solve-wgsl.ts` deferred (roadmap 12.4) | `spatial-regression-solve.ts:90-125` (`getPivotLimitedCholeskyWGSL`, the OLS limit relative to the diagonal), used at `gpu-restricted-regime-regression.ts:247`, `gpu-spatial-regime-regression.ts:177`, `gpu-probit-regression.ts:440`, `ordinary-sur-nodes.ts:255`, `:293`, `:343`; `gpu-spatial-panel-lag-maximum-likelihood.ts:259`, `:265`, `:287` use the equilibrated factorization of the GWR family | `math-review-pivot-limit.spec.ts` (probit, SUR, panel lag), `math-review-restricted-regimes.spec.ts` (both regime classes; the fully varying class returned status 0 for one regime on the old kernel) | Near-collinear predictors (squared cosine above `1 - 1e-6`) are status 2 (SUR error covariance: 3) instead of status 0 with an arbitrary solution. The private Gauss-Jordan inverse and LU log-determinant of the panel lag class are unchanged |
| regression-11 | fixed (confirmed on the GPU: 17% off for 1500 +- 3, `1.4e15` for a northing) | `gwr-fit-wgsl.ts:262-323` (streaming QR and one-sided Jacobi of `local-collinearity-wgsl.ts` instead of the Gram matrix and its eigenvalues) | `math-review-local-condition-number.spec.ts` (float64 closed form for one predictor); the mgwr fixture test in `geographically-weighted-regression-nonstationarity.spec.ts` still passes | `localConditionNumber` of a design with a large-mean predictor changes to the correct value |
| regression-12 | fixed (documentation reworded to what the kernel computes; formula unchanged; not checked against mgwr, which is not installed) | `gpu-geographically-weighted-regression.ts:110-119`, `gwr-fit-wgsl.ts:233-234`, `docs/api/gpu-spatial-analysis/operators-regression.md` (GWR outputs) | none (words) | none |
| regression-13 | fixed (comment and documentation; no parity claim) | `gpu-multiscale-geographically-weighted-regression.ts:273-277`, `docs/api/gpu-spatial-analysis/statistical-expansion.md` (which said the surfaces start at zero) | none (words) | none |
| regression-14 | fixed (documentation: the kernel's limit is `n <= predictorCount + 2`) | `gpu-spatial-regression-diagnostics.ts:198-201`, `spatial-regression-diagnostics-parameters.ts:65`, `docs/api/gpu-spatial-analysis/operators-regression.md` (diagnostics status) | none (words) | none |

### Extra findings of this pass

| ID | Outcome | What changed | Spec that pins it | Behaviour change a caller sees |
| --- | --- | --- | --- | --- |
| regression-15 (chi-square survival, lead's note) | fixed | `gpu-spatial-regression-diagnostics.ts:702-713`: the private helper returned `exp(-x / 2)` for every degree of freedom other than 1; it now returns that only for 2 and NaN otherwise. Its callers pass 1 or 2 | none: no caller can reach the other branch | none |
| regression-16 (compensated merges do nothing on the GPU) | fixed | The Neumaier merge of `regression-row-tiles.ts` and the Kahan merge of `getKahanSumWGSL` both returned the plain serial sum on the test GPU (`[2^24, 1 x 999]` summed to `2^24`; `2^20 + 4000 x 0.1` was 100 off). New `pairwise-sum-wgsl.ts` (`getPairwiseSumWGSL`, a fixed cascade sum, error about `log2(count)` roundings). It replaces every float compensation in the directory: `regression-row-tiles.ts:165-176`, `spatial-maximum-likelihood-wgsl.ts:166-186` (`getKahanSumWGSL` keeps its name and delegates), `gpu-ordinary-least-squares.ts` (two merges), `gpu-spatial-error-gm.ts`, `gpu-spatial-two-stage-least-squares.ts`, `gpu-spatial-regression-diagnostics.ts`, `gpu-spatial-weights-traces.ts`, `spatial-exact-log-determinant.ts`, `spatial-maximum-likelihood-nodes.ts` (three sums), `ordinary-sur-nodes.ts` | `math-review-tile-merge-sums.spec.ts` (failed on the old helpers on the GPU) | Sums over tiles differ in the last bits and are closer to exact. `GPUSpaceTimeHawkes` and `GPUSinkhornDivergence` use the row-tile merge and get the same change. The sum inside one tile is still a plain float32 loop, as documented |
| regression-17 (oracle that encoded old behaviour) | fixed | `test/.../probit-regression.node.spec.ts` asserted that the kernel text contains `compensation`; `spatial-maximum-likelihood-nodes.node.spec.ts` pinned the node list without the first-stage OLS nodes; `gtwr-mgtwr.spec.ts` compared the summary AICc with mgtwr's | the three specs | none |

### Not reproduced

None.

### Left as it is, and why

- **regression-02, -03, -04 were not run on the GPU against the old kernels.** The specs were
  written after the kernels were changed, so their failure on the old arithmetic rests on the
  reviewer's scripts and on reading the code. regression-01, -10 (fully varying regimes), -11 and
  -16 were seen to fail on the GPU before the fix.
- **A varying slope under a common intercept**: the first pass left it in the raw basis and said
  here that no shift removes its conditioning. No shift does, but a change of basis does
  (fr-statistics-01). Corrected in "Review follow-up" below.
- **The probit bound itself is kept** (lead's decision): a dataset whose maximum has a linear
  predictor beyond 12 gets status 4, not an estimate. Removing the bound would return the
  maximum for a correctly classified outlier, at the cost of redesigning separation detection.
- **Hausman direction**: documented spreg convention, unchanged.
- **Deferred (roadmap 12.4)**: the 20 call sites of the private `getCholeskyWGSL`, the private
  Gauss-Jordan inverse and LU log-determinant of the panel lag class, the private Jacobi eigenvalue
  and Belsley helpers, the three chi-square routes, the literals of section 4 (row-standardization
  tolerance, adaptive factor, `ln(2 pi)`), and the rename of `getKahanSumWGSL`.
- **Benchmark specs** (`*benchmark*.spec.ts`) were not run.
- `docs/lens-records/**` twins of the changed estimators were not regenerated (out of bounds).

### Upstream gaps

- None. Published `@luma.gl/graph` 10.0.0-alpha.10 has no regression, least-squares or dense-solver
  entry point, so there is nothing to delegate to or to converge on.

### Review follow-up (2026-11-10, after `fix-review-statistics.md`)

Line numbers are those of the working tree after this follow-up.

| Review row | Outcome | What changed | Spec that pins it | Behaviour change a caller sees |
| --- | --- | --- | --- | --- |
| fr-statistics-01 (regression-02 not fixed for a common intercept with a varying slope) | fixed. The reviewer is right and the first Resolution was wrong: the raw basis `{1, x 1[g]}` is replaced by `{x - mean, x 1[g]}` | `gpu-restricted-regime-regression.ts:254-330` (`getCommonInterceptFitWGSL`): every column about its pooled mean, a design form `constant + deviation + pivotTerm * pivotDeviation` per source column, the pivot is the varying-slope predictor with the largest `mean^2 / variance` and is used when that ratio is above 1 (below 1 the raw basis is the better conditioned one), other varying columns enter as `x_c - (mean_c / mean_p) x_p`, coefficients and covariance mapped back with `T V T'`; `:476-496` (residuals from the design forms and the solution in its basis, so Chow and Wald use it too) | `math-review-restricted-regimes.spec.ts`: the reviewer's three scenes (year, elevation, northing, 600 rows, `common: [true, false]`) to 2e-4 against float64, and a three-regime scene with two varying slopes and a common slope, all with large means | Elevation 1500 +- 3 was status 0 with both slopes 3.4% off, year 0.28% off, northing status 2: all three now match float64 with status 0 |
| second look: `(observed - shift) - intercept` | confirmed harmful on the GPU and fixed, in three kernels | The shader compiler of the test GPU regroups the residual chain as `y - (shift + small terms)`: the sum of the residuals of a response near 10,000 came out as an exact multiple of the float32 step of 10,000, and the Chow F was 12% off. Writing the difference on its own line does not help. New `rounded-difference-wgsl.ts` (`getRoundedDifference`: the bits of the difference pass through an exclusive-or with a zero read from a buffer, so the subtraction is carried out as written). Used at `gpu-restricted-regime-regression.ts:482-496`, `regime-moments.ts:223-228`, `gpu-spatial-error-maximum-likelihood.ts:526-532` (the first-stage residual of regression-03 had the same regrouping: `s2` 5% off for a response near 10,000) | `math-review-restricted-regimes.spec.ts` ("response mean far above its residuals", failed before), `math-review-error-ml-high-fit.spec.ts` ("response mean of 10000", failed before) | Chow F, `s2`, log-likelihood and `lambda` of a response whose mean is far above its residuals change to the float64 values |
| found with it: float32 regime means | fixed | A float32 regime mean misses the true mean by up to half a float32 step, which biased the levels of both nested models differently. `regime-moments.ts:24`, `:219-254` now also stores the sums of the deviations from the float32 means (the stride grows by `predictorCount + 1`), and `gpu-restricted-regime-regression.ts:233-246` (`momentAboutShifts`) carries them through every moment, so the moments are exact for the stored means | same "response mean far above its residuals" scene | as above. `GPUSpatialRegimeRegression` reads the same moments view and is unchanged apart from the stride |
| found with it: `GPUOrdinaryLeastSquares` | documented, not changed | Its residual pass is not regrouped on the test GPU (a guard changed no bit of the result and was removed again). Its float32 response mean shifts every residual by one common offset of at most half a float32 step of that mean, so the residual sum of squares is up to `n` times its square too large: 4% for 600 rows with residuals of 0.002 around 10,000. Stated in the class documentation `gpu-ordinary-least-squares.ts:144-148` | `math-review-large-response-mean.spec.ts` (the offset is common to all rows, at most half a step, and accounts for the residual sum of squares) | none |
| second look: error ML fallback and filtered solve | fixed | `gpu-spatial-error-maximum-likelihood.ts:1042-1045`, `:1068-1075`: the filtered solve applies the OLS pivot limit to its unit-diagonal system and reports status 1; the classification of a failed search uses the same limit | `math-review-pivot-limit.spec.ts` (spatial error ML) | Near-collinear predictors (squared cosine above `1 - 1e-6`) are status 1 instead of status 0 |
| fr-statistics-05 (pivot scenes do not pin 1e-6) | fixed | Scenes with predictor spreads of 7e-4 (squared cosine `1 - 4.9e-7`, must be singular) and 2e-3 (`1 - 4e-6`, must fit) added for probit, SUR, panel lag, error ML and both regime classes, beside the far scenes | `math-review-pivot-limit.spec.ts`, `math-review-restricted-regimes.spec.ts` | none |
| fr-statistics-06 (user documentation still says Kahan) | fixed | `docs/api/gpu-spatial-analysis/operators-regression.md` (OLS, diagnostics, weights traces, error GM) and `coverage-statistics.md` (IV covariance) now say a fixed pairwise merge; the regime paragraph of `coverage-statistics.md` describes the basis change | none (words) | none |
| fr-statistics-04 (no test of "overshoot damped, then status 0") | not added: no such scene could be constructed | The first Fisher step from the zero start is 2.5 times the linear-probability fit, which for a row on the model is `1 / sqrt(1 + slope^2 variance)` of its own linear predictor at the maximum, so it cannot pass a bound that the maximum respects. A float64 replica of the control flow found a step shortened by the bound followed by status 0 in 0 of 16,608 scenes (uniform, normal, cubed-normal and mixture predictors, slopes 0.5 to 16, 60 to 2,000 rows, leverage rows at 3 to 40 spreads with and without misclassified ones, and the strong-slope family of the old scene for slopes 2.0 to 6.0). Whenever the bound shortened a step the maximum lay beyond it. Steps halved by the likelihood test do occur (19 of 5,040 leverage scenes) but only within the noise of the likelihood next to convergence, where the float32 kernel need not take the same branch, so they do not make a deterministic test either | the status 4 path stays pinned by `math-review-probit-bound.spec.ts`; the status 0 path by its control scene and by `coverage-estimators.spec.ts` | none |

Still open after the follow-up:

- The second-look note on the pairwise cascade (a small compile-time count could be unrolled and
  linearised) was not tested beyond the 1,000 and 4,001 terms of the spec.
- `fitted` of `GPURestrictedRegimeRegression` is still evaluated from the raw coefficients and
  carries the float32 rounding of the response magnitude (0.25 for a response near 2.6e6); stated
  in the class documentation.
- The same regrouping can affect any other kernel of the library that forms
  `(value - mean) - small terms` in one chain. In this directory the regime, error-ML and OLS
  residual passes were tested on the GPU; the lag-ML, GM and 2SLS residual passes were not.

### Closing pass (2026-11-11)

Paths as above. Everything below was run on the test GPU (headless Chromium, WebGPU, Apple
Metal). References: spreg 1.9.1, libpysal 4.15.0, statsmodels 0.15.0, NumPy 2.5.4, SciPy 1.16.3
on CPython 3.14 (the pinned environment; spreg and statsmodels were installed into it as binary
wheels at the versions of `test/oracles/requirements.txt`), always in float64 on the float32
inputs the GPU receives. The scenes are built from a 32-bit linear congruential stream and IEEE
double arithmetic only, so Python and the specs build the same bits; each spec checks an input
sum against the generator's.

The test scenes: 600 rows on a ring (two neighbours, weights 0.5), `x` uniform on `[-2, 2)`.
`plain`: `y = 10000 + 2 x + 0.002 e`. `lag`: `y = (I - 0.4 W)^-1 (6000 + 2 x + 0.004 e)`, rounded
to multiples of `2^-9` so that the float32 lag is exact. `error`:
`y = 10000 + 2 x + (I - 0.5 W)^-1 0.004 e`.

| Item | Outcome | What was wrong (measured on the GPU before the change) | What changed | Spec that pins it | Evidence |
| --- | --- | --- | --- | --- | --- |
| 1a. Lag ML residuals | fixed (through item 2) | Its residual kernel has no `(value - shift) - small` chain: `u = e0 - rho e1` of two `GPUOrdinaryLeastSquares` fits. It inherited their float32-mean offset: `plain` residuals summed to -0.534 (a common offset of 0.9 of a float32 step), `s2` 19.6% high, log-likelihood 53 units low; `lag` 0.23% high | `gpu-ordinary-least-squares.ts` (item 2) | `math-review-closing-residuals.spec.ts` (lag ML, `plain` and `lag`) | spreg `ML_Lag(method='full')` on `y - 10000` (the shift changes only the intercept on a row-standardized `W`; spreg's unshifted value agrees to 1e-8; a NumPy golden-section maximum of the concentrated likelihood with ring eigenvalues agrees to 1e-10). After: `s2` 6e-6 and 1e-6 relative, log-likelihood within 0.002, `rho` within 1e-3 of its standard error, residuals row by row within 1% of their deviation (the bound the spec asserts for all four estimators; the largest error was not recorded) |
| 1b. Lag ML variance (found with 1a) | fixed, the centering cure of regression-04 | The information matrix used the raw `W A X b` column against the intercept: `sum wp^2 - (sum wp)^2 / n` cancels in float32 for a response near 10,000. Standard errors of `rho` and the intercept were 0.52 (`plain`) and 0.32 (`lag`) of spreg's; once the residuals were right the matrix was not positive definite and they were NaN | `gpu-spatial-lag-maximum-likelihood.ts`: new nodes `weighted-mean-tiles`, `weighted-mean-merge`; `final-moments-tiles` accumulates `W A X b` about its mean (through `getRoundedDifference`); the `variance` kernel maps the intercept back with the shift vector `(1, -mean(x), -mean(W A X b))` | same spec (standard errors); `spatial-maximum-likelihood-nodes.node.spec.ts` (node list) | spreg `ML_Lag` `std_err`: after, within 1.4e-4 relative on both scenes |
| 1c. 2SLS residuals | fixed, the regime cure (exact deviation sums and `getRoundedDifference`) | `residual = centeredResponse - sum z delta` with float32 means: the same common offset. `plain`: `sigma2` 19.6% high, every standard error 9.4% high, Moran's I of the residuals 0.212, Anselin-Kelejian 16.25 against 1.98; `lag`: `sigma2` 0.23% high, Anselin-Kelejian 13% high | `gpu-spatial-two-stage-least-squares.ts`: the moment pass also sums the deviations of the regressors and the response from their stored means; the solve stores `(sum (y - ybar) - sum (Z - mean) delta) / n`; the residual pass subtracts it; every `value - mean` of `getCenteredRow` goes through `getRoundedDifference` (runtime zero written by the means merge); the intercept adds the same remainder | same spec (2SLS, `plain` and `lag`) | spreg `GM_Lag(w_lags=1, spat_diag=True)` on `y - 10000`. After: `sigma2` 1.5e-6 relative, Anselin-Kelejian 4.5e-5, standard errors 3e-5. spreg on the unshifted response is itself 1e-4 off in `sigma2` (its raw normal equations leave a residual mean of 2e-5), which is why the shifted fit is the reference |
| 1d. GM residuals | fixed, the regime cure | Three faults. (i) The first-stage OLS offset read as spatial dependence: Kelejian-Prucha `lambda` 0.192 against 0.060 (`plain`). (ii) The filtered residual `(y - ybar) - lambda W (y - ybar) - f b` was regrouped by the shader compiler: after (i) was fixed `sigma2` was still 6.1% high (`plain`) and 0.4% (`error`). (iii) The reported `u = y - (intercept + x b)` is rounded per row to a float32 step of the response and carries the rounding of the float32 intercept: `u'u` 1.5% to 2% high, which also feeds the second step of the homoskedastic estimator (`sigma2` 1.5% and 0.8% high) | `gpu-spatial-error-gm.ts`: `getFilteredRow` forms every `value - mean` with `getRoundedDifference`; the residual pass forms `u = (y - ybar) - a0 - (x - xbar) b` with the same guard; runtime zero written by the means merge | same spec (both estimators, `plain` and `error`) | Homoskedastic: spreg `GM_Error_Hom(A1='hom_sc', max_iter=1)` on `y - 10000`: `lambda` within 2e-5, its standard error 1e-5 relative, `sigma2` 7e-6. Kelejian-Prucha: the exact minimum of spreg's own moment objective (`_momentsGM_Error`, golden section in NumPy) followed by spreg's filtered regression, because spreg's L-BFGS-B search stops within 2e-4 of the minimum and ran to -0.99 on the shifted `plain` scene: `lambda` within 2.6e-5, `sigma2` 1e-5 |
| 2. OLS float32 response mean | fixed | A float32 mean misses the true mean by up to a float32 step (more than the half step the first pass stated: the float32 sum of 600 values near 10,000 loses most of a step). Residual sum of squares 4% high on the first-pass scene and 19.6% on `plain` | `gpu-ordinary-least-squares.ts`: the moments pass also sums the deviations of each column from its stored mean (stride `+ k + 1`; `getRoundedDifference` on each deviation); `momentAt` subtracts `n` times the product of two mean remainders; the solve stores the residual offset; the residual pass subtracts it from `(y - ybar) - fit`, with `y - ybar` through `getRoundedDifference`; the intercept adds it. No output or prop changed. A fourth residual sum was not needed | `math-review-large-response-mean.spec.ts`, rewritten: the first-pass scene against float64 least squares (the spec's own normal equations), and `plain` against spreg `OLS` | After: residual sum of squares 3.7e-6 relative, the common offset 1e-8, standard errors, log-likelihood and Jarque-Bera as spreg; Koenker-Bassett and Breusch-Pagan against NumPy least squares of their definitions on `[1, x]` (spreg regresses on the squared regressors, a different test) |
| 3. Probit near the bound | fixed by a redesign of the likelihood and of what the bound applies to | The review's kernel returned status 4 for any maximum with a row beyond a linear predictor of 12 | `gpu-probit-regression.ts`: `tailProbability` (`Phi` through `erfc`, floored at the smallest normal f32); the likelihood is `log Phi(s eta)`, the score `s phi / Phi(s eta)`, the weight `(phi / Phi(eta)) (phi / Phi(-eta))`; the line search reports the largest \|eta\| among rows on the wrong side only; a singular information matrix after a linear predictor has passed 8 is status 4 | `math-review-probit-bound.spec.ts`, rewritten (8 tests); `probit-regression.node.spec.ts` (kernel text) | statsmodels `Probit` (Newton) and a Newton solve of the tail-exact likelihood (`scipy.special.log_ndtr`), which agree to 1e-13 on every scene but the gross outlier: control; a correct row at a linear predictor of 18.9; the just-beyond scene (largest 12.06); a misclassified row at 7.7. All status 0 within a hundredth of a standard error, standard errors (expected information) within 2e-3, log-likelihood 2e-5. Complete separation, quasi-complete separation and separation by two predictors: status 4 (statsmodels diverges with a convergence warning on the first two). A maximum with a misclassified row at 13.2: status 4 |

What the GPU does that the source does not say, found in this pass:

- `((y - mean) - fit) - offset` with `mean` and `offset` both read from a buffer is compiled as
  `(y - (mean + offset)) - fit`: the offset (below a float32 step of the mean) is lost. With the
  guard on `y - mean` the common offset of the OLS residuals fell from 8.6e-5 to 1e-8. This is
  the regrouping of the follow-up above in a second form, and it means the first pass's note that
  the OLS residual pass "is not regrouped" held only while the chain had two terms.
- The sum of deviations `sum (y - mean)` was not regrouped on the test GPU (the guard changed no
  bit), but the guard is kept there, as in `regime-moments.ts`.

Why item 3 needed more than the bound. A bound on every row cannot tell a fit pinned by a
harmless row from separated data, and removing it alone is wrong: with probabilities floored at
1e-7 a row on the wrong side beyond a linear predictor of 5.2 has a flat likelihood and a score of
about `phi 1e7` where the true ones are `-eta^2 / 2` and `-eta`, so such a row was ignored (by
derivation from the old kernel text; the old kernel was not run on the misclassified scene). The
redesign therefore has three parts: tail-accurate likelihood, score and weight; a bound of 12 for
rows on the wrong side only (where float32 `Phi` ends); separation detected by what it does to
the fit (iterations exhausted with a linear predictor above 8, or the information matrix
singular after one has passed 8).

Not verified, or left:

- **Float32 lag** (`left`, documented and pinned): `W y` is summed from the raw response in the
  lag ML and 2SLS kernels, so a lagged value is rounded to a float32 step of the response. On the
  `lag` scene without the rounding to `2^-9` both estimators are 0.35% high in `sigma2` against
  spreg (`math-review-closing-residuals.spec.ts`, last two tests, bound 1%). Lagging the centered
  response does not cure it in general, because `W y = W (y - c) + c (row sum)` and the row sum of
  float32 weights such as 1/3 is not exact either. GM lags the centered columns already.
- **Probit, misclassified row beyond 12**: status 4 by design. A fit of that kind needs `log Phi`
  beyond the float32 range (an asymptotic series for `log Phi` and for the Mills ratio below -12);
  not done.
- **Probit separation** needs the default iteration budget: with a `maximumIterations` too small
  for any linear predictor to reach 8, separated data are status 3. Stated in the class
  documentation. A `tolerance` above about `1 / eta^2` can report a converged step on separated
  data; that was also true before.
- **Probit `fitted` and the spatial diagnostics** still use probabilities floored at 1e-7.
- The old kernels were run on the GPU for every row of the table except the misclassified probit
  scene (see above).
- The lag ML rho search still forms `e0'e0 - 2 rho e0'e1 + rho^2 e1'e1` in float32. On the `lag`
  scene (`u'u` 2e-5 of `e0'e0`) `rho` is within 6e-8, so it was left; a scene with a smaller
  ratio was not built.
- Heavier scenes (thousands of rows, several predictors, non-ring weights) were not added for the
  large-mean case; the existing fixtures of the four classes still pass.
- `docs/lens-records/**` twins were not regenerated (out of bounds); the lag ML record gains two
  nodes.
- No reference number is pasted into a spec. The five scratch scripts of the pass were merged
  into two committed generators, each self-contained and listed in `test/oracles/GENERATORS.md`:
  - `test/oracles/statistics-regression-large-mean.py` writes
    `test/oracles/fixtures/statistics-regression-large-mean.json` (items 1 and 2), read by
    `math-review-closing-residuals.spec.ts` and `math-review-large-response-mean.spec.ts`;
  - `test/oracles/statistics-probit-tails.py` writes
    `test/oracles/fixtures/statistics-probit-tails.json` (item 3), read by
    `math-review-probit-bound.spec.ts`.
  Both were run from the repository with `venv314/bin/python -I <generator>` (CPython 3.14.8,
  NumPy 2.5.4, SciPy 1.16.3, libpysal 4.15.0, spreg 1.9.1, statsmodels 0.15.0; the fixtures
  record the versions), twice, with byte-identical output. `spreg==1.9.1` and
  `statsmodels==0.15.0` were already pinned in `test/oracles/requirements.txt`. The fixtures
  equal the numbers the specs carried inline before, with three kinds of difference, all far
  inside the tolerances: the standard error of the unshifted intercept of the lag models is now
  derived from the covariance of the shifted fit instead of read from spreg's unshifted fit
  (7e-9 to 5e-5 relative); the Kelejian-Prucha `error` scene differs in the eleventh digit
  (the golden section starts from a different bracket on a flat objective); the probit
  coefficients are those of the tail-exact solve instead of statsmodels' (1e-13). The
  generators assert the cross-checks named in the table (golden-section `rho` against
  `ML_Lag` to 1e-7; statsmodels against the tail-exact solve to 1e-9, and more than 0.1 apart
  on the gross outlier).

Gates run for this pass (2026-11-11): source typecheck watcher clean for `spatial-regression`;
a scoped `tsc` over the six changed specs clean; `biome check` clean on the eleven changed
files; `check-contributor-contract` no new findings; `check-upstream-first` no violations;
headless: 28 spec files, 283 tests (the changed classes, their composers and the new specs);
node: 13 spec files, 538 tests (the regression node specs and the statistics kernel audit). The
full suites, `yarn build` and the benchmark specs were not run (the brief forbids the first two).
After the specs were switched to the committed fixtures: the scoped `tsc` clean again; headless
`math-review-large-response-mean`, `math-review-closing-residuals`, `math-review-probit-bound`
and `coverage-estimators` (4 files, 37 tests); node `spatial-maximum-likelihood-nodes`,
`probit-regression` and `oracles/oracle-fixture-runner-plat06` (3 files, 13 tests).

### Closing pass, review fixes (2026-11-11)

The fixes for `fix-review-closing-regression.md` (three major and five minor findings on the
closing pass above). Paths as above. Everything was run on the test GPU (headless Chromium,
WebGPU, Apple Metal). References: spreg 1.9.1, libpysal 4.15.0, statsmodels 0.15.0, NumPy
2.5.4, SciPy 1.16.3 on CPython 3.14, in float64 on the float32 inputs the GPU receives. Every
finding was reproduced on the kernel as it stood before it was changed; "before" below is that
measurement, on the scenes of the new fixtures (the reviewer's scenes rebuilt from a power
series instead of a linear solve, so that the specs rebuild the same bits).

| Finding | Outcome | Before (measured on the GPU) | What changed | Spec that pins it | Evidence after |
| --- | --- | --- | --- | --- | --- |
| frc-regression-01. Lag ML `rho` search | fixed | 24 high-fit scenes (`u'u` 2.4e-9 to 3.4e-8 of `e0'e0`): 14 status 3, 3 wrong with status 0 (`sigma2` 20 and 41 times the reference, and 3.3% high), 7 right. Three islands (`u'u` 2e-10 of `e0'e0`): `sigma2` 2,801 times the reference, log-likelihood 2,381 low, status 0 | `gpu-spatial-lag-maximum-likelihood.ts`: new nodes `moments-merge` (`rho0 = e0'e1 / e1'e1`) and `direct-moments-tiles` (`r = e0 - rho0 e1` per row through `getRoundedDifference`; sums `r'r` and `r'e1`); `rho-search` uses `u'u = S + e1'e1 (rho - rho*)^2` with `rho* = rho0 + r'e1 / e1'e1`, `S = r'r - (r'e1)^2 / e1'e1`, and searches only if `S > 0`; `finish` compares the search's `u'u` at the estimate with the one summed from the published residuals and reports status 3 above a relative difference of 1e-2 | `math-review-lag-conditioning.spec.ts` (`high-fit-*`, `islands-lag`, `islands-ordinary`) | All 25 status 0, interior, variance defined. High fit: `rho` within 0.035 standard errors, `sigma2` within 3.4e-4, log-likelihood within 0.10, standard errors within 1.7e-4 (what remains is the float32 rounding of the fitted values, 1% to 3% of a residual). Islands: `rho` within 0.12 standard errors (one float32 step of `rho`), `sigma2` within 3.9e-5, log-likelihood within 0.012. Reference: the float64 maximum of the concentrated likelihood (centered `lstsq`, `eigvals`, grid and golden section), which the generator holds to spreg `ML_Lag(method='full')` (`rho` within 5e-7, a log-likelihood not above it and within 0.05); the spec also holds the GPU log-likelihood to spreg's own |
| frc-regression-02. Float32 lag of the raw response | fixed for weights whose rows share a row sum (exactly or up to the rounding of the weights) and for rows without neighbours; a float32 limit is left for row sums that differ, see below | Lag ML `sigma2`: 0.34%, 0.57%, 0.56%, 0.027%, 0.30%, 4.2% on the six rows of the review's table; 2.5% on mixed row sums (1 and 1.0000000298); log-likelihood up to 12.5 low. 2SLS the same in `sigma2`; Anselin-Kelejian 10%, 1.5%, 45%, 25%, 26%, 2.5% off, 10% on mixed row sums | Both kernels form the lag column `v = W (y - c) + c (s_i - s)`: `c` the float32 response mean, each `y - c` through `getRoundedDifference`, `s_i` the exact row sum of the float32 weights (`getAdditionRoundingError` on every addition), `s` the largest row sum; `W y = v + c s`, and `c s` goes to the intercept (lag ML: `solve-coefficients` and `reduced`; 2SLS: `regressorMean`, and the raw lag of the `robust` design). Lag ML: `means-tiles`/`means-merge` moved first, `lagged-response-tiles` changed, new `row-sum-merge` and `lagged-response-shift`. 2SLS: `weights-norm-tiles`/`-merge` moved first and also give `c` and `s`; `getLags` changed | `math-review-lag-conditioning.spec.ts` (`lag-*`, `two-predictors-*`, bounds per group); `math-review-closing-residuals.spec.ts` (the 1% pin replaced by 4e-5) | Lag ML: `sigma2` within 1.1e-5, log-likelihood within 0.0033, residuals within 7e-4 of their deviation on the nine scenes (ring 0.5, six neighbours at float32 1/6, binary row sum 2, mixed row sums, two predictors; `rho` 0.2 to 0.8; residual deviation 0.002 and 0.004). 2SLS: `sigma2` within 6.7e-6, Anselin-Kelejian within 5.5e-4 (1.2e-3 with a second predictor near 4,096). Bounds in the spec: 4e-5 and 0.01 (lag ML), 2.5e-5 and 4e-3 (2SLS). Reference for 2SLS: a centered float64 solve that the generator holds to spreg `GM_Lag` at 1e-8 wherever a shift of the response is an intercept shift |
| frc-regression-03. Probit quasi-separation by a small group | fixed | Indicator of rows 17, 301, 555: status 2 (also with 128 iterations). Indicator of row 17: status 3 with (0.1686, 1.0282, 7.2247) written. Eleven predictors: status 3, indicator coefficient 7.57 | `gpu-probit-regression.ts`: the line-search tile pass also reduces, per direction, the largest move of a row towards the wrong side and towards the right side, each over the sum of the magnitudes of the row's own terms; `decide` records a separating direction (right-side move at least 0.5, wrong-side move at most 1e-3) and keeps it; a recorded direction makes every exit status 4 (singular information, iterations exhausted, failed line search, and a step below `tolerance`); the `> 8` threshold is gone | `math-review-probit-bound.spec.ts` (22 tests: six indicator scenes, 128 and 10 iterations, tolerances 1e-3 and 1e-4, complete separation with 8 iterations) | All status 4. Oracle for "separated": the optimum of a linear program over directions (SciPy `linprog`, HiGHS), positive on every separated scene and 0 on every finite one (asserted in the generator), and statsmodels `Probit`, which does not converge on them. The five fits that return the statsmodels maximum still do (control, correct outlier at 18.9, just beyond 12, misclassified row at 7.7, and a new steep fit with a largest linear predictor of 29.5), within a hundredth of a standard error |
| frc-regression-04. 2SLS and a constant predictor | fixed | With the old moments: status 0 for the constants 0.1, 1/3, 12345.678 and -4096.3 on the ring and island scenes (8 constants tried on 3 scenes); OLS, lag ML and both GM estimators status 1 on all 24 | `gpu-spatial-two-stage-least-squares.ts`: the moment pass also sums the deviations of the lagged instruments; `getInstrumentMoment`, `getCrossMoment`, `getResponseMoment`, `getRegressorSquares` subtract `n` times the product of two mean remainders (moments about the true means, as OLS); a column whose corrected sum of squares is below 1e-5 of the uncorrected one is constant: status 1 | `math-review-lag-conditioning.spec.ts` ("with a constant predictor is status 1", 5 estimators, 3 scenes, 8 constants) | 15 tests, 120 fits, all status 1. The "before" was measured by switching the correction off in the new kernel, not on the old file |
| frc-regression-05. Unfinished probit fit above 8 | fixed, by the test of frc-regression-03 | Correct-outlier scene with `maximumIterations: 3`: status 4, NaN. Latent `0.2 + 8 x` with 4 iterations: status 4, NaN | The exit for exhausted iterations no longer reads the largest linear predictor: status 3 with the iterate unless a separating direction was recorded | `math-review-probit-bound.spec.ts` (three "runs out of iterations" tests) | Status 3, finite coefficients, NaN standard errors, the iteration count, a log-likelihood between `n ln 0.5` and the fixture maximum, coefficients nearer to the maximum than the start. The iterate itself has no library value |
| frc-regression-06. Spec scenes and names | fixed | | New fixture and spec with 36 lag scenes (high fit, islands, `rho` 0.15 to 0.8, five kinds of weights, inexact lags, two predictors with one near 4,096); the Kelejian-Prucha test is named for its reference and asserts spreg's own `lambda` on `error` at 5e-4; the probit kernel-text assertions that pinned the `> 8` rule are replaced; the probit fixture stores statsmodels `GLM` standard errors, which the spec asserts | `math-review-lag-conditioning.spec.ts`, `math-review-closing-residuals.spec.ts`, `probit-regression.node.spec.ts`, `math-review-probit-bound.spec.ts` | Generator asserts: GLM `bse` against the expected-information standard errors within 1e-6 (2e-5 on the misclassified-outlier scene, where the library's link clips) |
| frc-regression-07. Documentation | fixed | | `docs/api/gpu-spatial-analysis/operators-regression.md` (2SLS: the `robust` prop in the prop list and in the standard-error paragraph, the status line, the measured limits; lag ML: steps, status 3, the measured limits), `coverage-statistics.md` (status 3, separation), the three class documents. The four merges of the lag ML kernel call `getPairwiseSumWGSL` through a local `getMergeSumWGSL`; `getKahanSumWGSL` stays exported for the other classes | | Every number in those paragraphs is a measurement of this pass |
| frc-regression-08. Unguarded differences from a mean | guarded where a sum could be regrouped; see the audit below | No wrong value was produced before or after | OLS residual pass, lag ML final moments (three), 2SLS Anselin-Kelejian products (two), probit `getDesign` | the existing specs; no new failure could be produced to pin | The whole regression folder passes unchanged |

frc-regression-02, what is left. (The 2SLS part of this paragraph was wrong in its attribution
and is superseded by frc-regression-11 in the second round below: the 3% and 0.77% were solve
error, and a refinement step removes them.) For weights whose row sums differ by more than
rounding (row sums 2 and 3) with a response near 10,000 the lag itself differs between rows by
the response
mean. Lag ML: `sigma2` within 2.5e-4, `rho` within 0.065 standard errors (its standard error is
4.5e-8, three float32 steps). 2SLS: `sigma2` 3.0% off with `rho` within 0.015 standard errors
(4.6e-8), and 0.77% off with three islands. This is not the lag any more: the two-stage `rho` is
not at the minimum of `u'u`, so `sigma2` has a slope of `2 (rho - rho_ML) var(W y)` in `rho`,
and a float32 `rho` cannot be closer than a few 1e-8. The published `sigma2` is that of the
published float32 coefficients. Both are pinned at their measured size
(`lag-unequal-rho-0.15-deviation-0.004`: 1e-3 and 10%; `islands-lag`: 1.5e-4 and 2.5%) and
stated in the class documentation and `docs/api`. No status is raised: the kernel has no
float32 quantity that says the fit is in this regime. The scene with row sums 2 and 3 was not
run on the kernel before the fix. `W X` (the instruments) is still summed from the raw
predictors: the Anselin-Kelejian statistic is 1.2e-3 off with a predictor near 4,096.

frc-regression-01, the consistency check. (Replaced in the second round below,
frc-regression-12: it compared one quadratic with itself and could not catch a wrong `rho`.
`GPUSpatialPanelLagMaximumLikelihood` had the three-sum form: frc-regression-09.) The search
evaluates `u'u` from `S`, `e1'e1` and
`rho*`; the published `sigma2` is summed from `e0 - rho e1` per row. A relative difference above
1e-2 is status 3. On the 36 scenes the two agree far inside that (no scene is status 3). It
would catch a compiler that regroups the direct sums; it was not seen to fire.
`GPUSpatialErrorMaximumLikelihood` and the two panel ML classes were not checked for the
three-sum form.

frc-regression-03, how the tolerance was set and what is not caught.

- The first form of the test (largest wrong-side move over the largest move of any row, last
  direction only) failed on the GPU for one separated row of 600: near a linear predictor of 6
  the score of that row is below the float32 noise of the score sums, the direction of the
  indicator is rounding (the last one was negative), and the fit ended as status 3 at 7.22. The
  certificate is therefore kept once found.
- That form is also wrong for a correctly classified row with high leverage (a float64 run of
  the same iteration with one row at 1e5: ratio near 1e-5 on data that are not separated). The
  move of a row is now measured against the sum of the magnitudes of its own terms.
- Tolerance. On the GPU the rounding floor of the wrong-side ratio on separated data is just
  under 1e-4: with 1e-5 the one-row and eleven-predictor scenes are not certified, with 1e-4
  the one-row scene needs 17 to 24 iterations, with 1e-3 every indicator scene is certified
  within 8 to 10 iterations and complete separation within 4. On data that are not separated
  the smallest ratio in any iteration of a float64 run was 1.1e-2 (the row at 1e5), 2.8e-2
  (latent slope 40), 3.9e-2 (twenty samples of 40 rows, two of which were separated and
  certified), and near 1 for ordinary fits. The tolerance is 1e-3.
- A certified fit that takes a step below `tolerance` is status 4, not 0. Before, the one-row
  scene was status 0 at tolerances of 1e-3 and 1e-4 (coefficient 7.30, standard error 5,511):
  a rounding step passed for convergence.
- Not caught: a `maximumIterations` below 8 to 10 on the indicator scenes (status 3), and a
  `tolerance` of 1e-2 on the one-row scene (status 0, coefficient 3.94, standard error 5.39:
  the step passes before the other coefficients have settled). Both are in the class
  documentation and `docs/api`.
- Data that are not separated but whose misclassified rows all move by less than 1e-3 of their
  own terms would be reported as separated. None was found; the float64 run is not the GPU.

frc-regression-08, the audit of every difference of a value and a mean in the five kernels.

| Kernel | Difference | State |
| --- | --- | --- |
| OLS `moments-tiles` | `x - xbar`, `y - ybar` | guarded before this pass |
| OLS `residuals` | `y - ybar` | guarded before this pass |
| OLS `residuals` | `x - xbar` inside `sum b (x - xbar)` | guarded now: the sum could be regrouped into `sum(b x) - sum(b xbar)` |
| OLS `finish` | `ybar - sum b xbar` | safe: it is the intercept, a value at the scale of the response; no small quantity is formed |
| Lag ML `lagged-response-tiles` | `y_j - ybar`, row sum minus `s` | new, guarded |
| Lag ML `direct-moments-tiles`, `rho-search`, `finish` | `e0 - rho0 e1`, `rho - rho0` | new, guarded |
| Lag ML `final-moments-tiles` | `y - ybar`, `prediction - ybar`, `x - xbar` | guarded now: each feeds tile sums that could be regrouped |
| Lag ML `final-moments-tiles` | `W A X b` minus its mean | guarded before this pass |
| Lag ML `residuals` | `(y - e0) - rho (W y - e1)` | safe: fitted values at the scale of the response |
| 2SLS `getCenteredRow` | every `value - mean` | guarded before this pass |
| 2SLS `getLags` | `y_j - c`, row sum minus `s` | new, guarded |
| 2SLS `anselin-kelejian-products` | `x - xbar`, lag minus its mean | guarded now: tile sums of products |
| GM `getFilteredRow`, residual pass | every `value - mean` | guarded before this pass; no other difference from a mean in the file |
| Probit `getDesign` | `(x - mean) * scale` | guarded now (runtime zero written by `initialize`): the linear predictor is a sum over predictors of products with it |
| Probit `variance` tiles | `x - sum / n`, squared at once | left: it runs before any node has written a runtime zero, and a sum of squares of differences cannot be regrouped without expanding the square. Measured by the reviewer: a predictor shifted by 10,000 is within 2e-6 of a standard error |

Whether the shader compiler would have regrouped any of the newly guarded sums is not known:
none was observed to. The guards cost two bit casts each.

Results on ordinary data. The regression spec folder was run before and after:

- before: headless 44 files, 369 tests; node 34 files, 202 tests, all passing;
- after: headless 45 files, 470 tests (the 369, 87 in the new spec, 14 added to the probit
  spec); node 34 files, 202 tests, all passing.

No numerical assertion of an existing spec was loosened. Assertions changed:
`spatial-maximum-likelihood-nodes.node.spec.ts` (node list: order and four new nodes),
`spatial-two-stage-least-squares.node.spec.ts` (node order), `probit-regression.node.spec.ts`
(two kernel-text assertions on the `> 8` rule replaced by its absence and the separation
flag), `math-review-closing-residuals.spec.ts` (test names; the 1% bound of the inexact-lag
scene tightened to 4e-5; spreg's own Kelejian-Prucha `lambda` asserted).

For the session that regenerates lens records (`docs/lens-records/**` was not touched):

- `GPUOrdinaryLeastSquares`: the `residuals` kernel text changed; node count unchanged. This
  reaches every composer: `GPUSpatialLagMaximumLikelihood` (two fits), `GPUSpatialErrorGM`,
  `GPUSpatialPanelRegression`, the panel diagnostics, the spatial SUR nodes,
  `GPUSpatialCrossValidatedOLS`, the spatial regression recipe, the kepler OLS adapters.
- `GPUSpatialLagMaximumLikelihood`: four more nodes (`row-sum-merge`,
  `lagged-response-shift`, `moments-merge`, `direct-moments-tiles`); `means-tiles` and
  `means-merge` now come first; the workspace has nine more slots, so the constants block of
  every kernel of the class changed; the bodies of `means-merge`, `lagged-response-tiles`,
  `rho-search`, `solve-coefficients`, `residuals`, `final-moments-tiles` and `finish` changed.
  New transients `weights-row-sums` and `partial-direct-moments`; `partial-weights-norm` is
  twice as long. `GPUSpatialLagImpacts` scenes that run the fit inherit this.
- `GPUSpatialTwoStageLeastSquares`: node count unchanged; `weights-norm-tiles` and
  `weights-norm-merge` now come first; the constants block of every kernel changed (two
  workspace slots, a longer moment stride); the bodies of `weights-norm-tiles`,
  `weights-norm-merge`, `moments-tiles`, `anselin-kelejian-products`, `robust-inputs`, and the
  lag and solve declarations (`getLags`, `regressorMean`, the moment accessors) changed;
  `means-tiles` and `robust-inputs` bind `workspace`.
- `GPUProbitRegression`: node count unchanged; the state layout has six more words, so the
  constants of every kernel that reads the state changed; `initial-likelihood` has 4 terms
  (was 2) and each `line-search` 50 (was 48); the bodies of the likelihood tiles, `initialize`,
  each `solve`, each `decide` and `finish` changed, and `getDesign` in every row pass.
- `GPUSpatialErrorGM`: its own kernels are unchanged; its first-stage OLS `residuals` kernel
  changed.

Fixtures and generators. `test/oracles/statistics-regression-lag-conditioning.py` (new) writes
`fixtures/statistics-regression-lag-conditioning.json`; `statistics-probit-tails.py` gained
eight scenes, the linear-program oracle and the GLM standard errors (no existing value of its
fixture changed); `statistics-regression-large-mean.py` is unchanged. All three were rerun with
`node scripts/run-oracle-fixture.mjs <generator> --python <venv314>`: IDENTICAL.
`GENERATORS.md` lists the new generator.

Not verified:

- The kernels before the fix were run on the new scenes for frc-regression-01, -02, -03 and
  -05; for -04 the old behaviour was reproduced by switching the moment correction off.
- Chunked weights, the stochastic trace mode, `tailExtrapolation`, the exact log-determinant
  mode and `instrumentOrder: 2` with the new lag column beyond what the existing specs cover
  (they pass). The `robust` covariance reads the raw lag (the lag column plus its constant),
  as before.
- Negative weights: the row sum the lag is formed about is the largest row sum, which is not
  typical when weights have both signs; the result is still exact in the row sums, the lag
  column is then rounded at the size of the response mean.
- Any GPU other than the test GPU. The full suites, `yarn build` and the benchmark specs were
  not run.

Gates (2026-11-11): source typecheck watcher clean for `spatial-regression`; a scoped `tsc` over
the six changed specs clean; `biome check` on the eleven changed TypeScript files;
`check-contributor-contract` no new findings; `check-upstream-first` no violations; headless
`modules/analysis/test/gpu-spatial-analysis/spatial-regression` (45 files, 470 tests) and the
recipe and value-cube composition specs that use these classes (2 files, 5 tests); node the same
folder (34 files, 202 tests), and the kernel audits, the recipes, the oracle fixture runner and
the two lens node specs that build these classes (69 files, 2,916 tests).

#### Second round (2026-11-11): the re-review

The fixes for the four findings of "Re-review (2026-11-11)" in
`fix-review-closing-regression.md`. Same GPU, same references. Each finding was reproduced on
the kernel before it was changed, on scenes of the committed fixtures.

| Finding | Outcome | Before (measured on the GPU) | What changed | Spec that pins it | Evidence after |
| --- | --- | --- | --- | --- | --- |
| frc-regression-09. Panel lag ML publishes `sigma2` from three float32 sums | fixed | 16 entities, 16 periods, unit and random effects. `u'u` 3e-5 of `e0'e0`: `sigma2` 1.1% and 2.0% off. 3e-7: 36% low and 96% high, log-likelihood 190 high and 81 low. 3e-9 (8 scenes, 16 fits): 9 status 0 with `sigma2` 10 to 241 times spreg's, 1 status 2 with `sigma2 = -8.7e-5` written, 6 status 3 | `gpu-spatial-panel-lag-maximum-likelihood.ts`: after the three sums a second row loop sums `r = e0 - rho0 e1` (`getRoundedDifference`) and `r'e1`; `getVariance` is `(S + e1'e1 (rho - rho*)^2) / n`, used by the golden section and for the published `sigma2`, log-likelihood and covariance. The pack node guards every difference from a pooled mean and carries the row sum exactly (`getAdditionRoundingError`); `pooled-means` writes the runtime zero (one more word in `centers`). No node added | `math-review-panel-lag-conditioning.spec.ts` (22 tests: 11 scenes, both effects) | All 22 status 0. `sigma2` within 7.8e-6 (3e-5), 6.3e-5 (3e-7), 1.7e-3 (3e-9); `rho` within 0.043 standard errors; log-likelihood within 0.22; standard errors within 8.5e-4; the mean square of the published residuals within the same bounds. Reference: spreg 1.9.1 `Panel_FE_Lag` and `Panel_RE_Lag` (`epsilon=1e-9`), which the generator holds to float64 maxima of both likelihoods found another way (`rho` 2e-8, `sig2` 1e-7 and 1e-5). The ordinary scene moved from 2.7e-5 and 9.2e-5 to 6.9e-5 and 6.6e-5 in `sigma2`: rounding of a serial float32 fit of 256 rows either way |
| frc-regression-10. Probit certificate fails below 2e-5 of the rows | fixed | With the new step switched off (the round-one kernel): one row of 100,000 status 0 (indicator 7.589, standard error 17,060); three of 200,000 status 0 (7.449, 10,947); five of 400,000 status 2; five of 1,000,000 status 3 (7.465 written); twenty of 1,000,000 status 0 (8.008, 3,644) | `gpu-probit-regression.ts`: the solve node stores a candidate direction, the Fisher direction with its rounding taken out (columns whose largest contribution to any row is below 1e-4 of the largest contribution of any column are dropped; when the rows at the low or the high end of every kept column would move by less than that, the intercept is set so that they do not move); the line-search pass tests the candidate, still per row and reduced with `max`. The initial likelihood pass also reduces the extremes of every scaled column | `math-review-probit-bound.spec.ts` (28 tests; six new: the five scenes and a twin of 100,000 rows whose group has both outcomes) | The five are status 4; the twin is status 0 within a hundredth of a standard error of the statsmodels maximum. Oracle: the linear program of the generator and statsmodels, as before. All five are still certified with a noise fraction of 1e-6 (the constant is 1e-4). The reviewer's probes on this kernel: 22 large-sample probes status 4 at the defaults; 72 small samples, 60 not separated all status 0 (within 6.9e-6 of a standard error) and 12 separated all status 4; an indicator group with both outcomes at 600 and 20,000 rows and two rare-event fits status 0 |
| frc-regression-11. The 2SLS leftover is solve error | fixed | `lag-unequal-rho-0.15-deviation-0.004`: `sigma2` +2.96%, `rho` 0.015 standard errors (3 float32 steps). `islands-lag`: +0.77% | `gpu-spatial-two-stage-least-squares.ts`: three nodes (`refinement-tiles`, `refinement-merge`, `refine`): the residuals at the float32 `delta` formed without the rounding of their products (`getStructuralResidual`), `H'u`, and the correction `(Z'PZ)^-1 Z'P u` with the factors of the solve, kept beside `delta`; the residual pass applies it; `table` publishes the float32 nearest to `delta + correction`. New `analysis-core/kernels/multiplication-rounding-error-wgsl.ts` (moved there by the lead, beside the addition helper) (`getMultiplicationRoundingError`, integer arithmetic on the significands) | `math-review-lag-conditioning.spec.ts` (2SLS bounds tightened: `sigma2` 2.5e-5 to 1e-5, the 10% and 2.5% pins to 1e-5 and 3e-5); `test/utils/multiplication-rounding-error-wgsl.spec.ts` (4,014 products against float64) | Row sums 2 and 3: `sigma2` within 9.5e-7, `rho` within 3.7e-4 standard errors, standard errors 6.2e-7, Anselin-Kelejian 4.9e-7. Three islands: 9.2e-6, 4.8e-5 standard errors, 9.1e-6. The other scenes improved too: `lag-*` `sigma2` within 2.8e-6 (was 6.7e-6) and the statistic within 2.4e-4 (5.5e-4; two predictors 3.3e-5 from 1.2e-3); high fit 1.6e-4 (2.5e-4); slopes within 5.3e-8 (1.2e-6) |
| frc-regression-12. The lag ML check could not catch a wrong `rho` | replaced by a first-order condition | The check compared `S + e1'e1 (rho - rho*)^2` with `sum (e0 - rho e1)^2`, the same quadratic at the same `rho` | `gpu-spatial-lag-maximum-likelihood.ts`: `final-moments-tiles` also sums `u'e1` from the published residuals and the second OLS residual; `finish` forms `g = n u'e1 / u'u - tr(W A)` and its curvature `c` and reports status 3 for an interior `rho` unless `g^2 <= 0.25 c`; the four `DIRECT` workspace slots and the old comparison are gone | not pinned by a spec: the kernel has no input that misleads the search | Mutation run: with the search moved 1e-4 off on purpose, all 36 scenes of `math-review-lag-conditioning.spec.ts` are status 3 (the shift is 0.8 to 2,200 standard errors of `rho`); unmutated, all 36 are status 0, as are the other lag ML specs (stochastic traces, tail extrapolation, exact log-determinant, chunked weights) |

frc-regression-11, what the refinement step uses and why. The correction is
`(Z'PZ)^-1 Z'P u`, and `u` has to be good to a few digits of the correction (4.6e-8 in `rho`).
With row sums 2 and 3 a regressor times its coefficient is near 750 and the residual is 0.01:
each product is rounded to 2e-5, and the lag column `W (y - c) + c (s_i - s)` is rounded to a
float32 step of 10,000 (1e-3) on the rows whose row sum is not the largest. By estimate that
rounding, random over rows, is worth 1e-8 in the correction, a quarter of it (a refinement on
plainly formed residuals was not run). So the residual of the refinement is formed in two
float32 parts: every product keeps its error
(`getMultiplicationRoundingError`), every subtraction keeps its own
(`getAdditionRoundingError`), each product goes through an integer exclusive-or so that the
compiler cannot fuse it with the subtraction, and the lag enters as its two parts multiplied
separately. The published residuals are these, less the correction. The contract asks for such
a helper in `analysis-core`; it is in `spatial-regression/` because that is what this stream
owns (see the final message).

One number of the reviewer's reruns moved the wrong way against its reference and is not a
regression: the Anselin-Kelejian statistic of `islands-plain-large-mean` (three islands, an
ordinary response near 10,000 with residuals of 0.004) went from 2.6e-5 to 1.0e-3 against spreg
on the unshifted response. The centered float64 solve gives 0.0102025, spreg on the unshifted
response 0.0101923 (its raw normal equations), the GPU 0.0102026: within 7e-6 of the former.
`sigma2` of the same scene went from 0.41% to 9.3e-6.

frc-regression-10, what the certificate still needs, measured on this kernel.

| Scene | Smallest `maximumIterations` that gives status 4 | `tolerance` 1e-2 | `tolerance` 1e-3 |
| --- | --- | --- | --- |
| one row of 600; three rows of 600 | 6 | status 0 (3.94, standard error 5.39); status 4 | status 4 |
| every second row of 600 | 10 | status 4 | status 4 |
| one row of 100,000 | 6 | status 0 | status 0 (4.19, standard error 8.26, after 5 steps) |
| twenty rows of 1,000,000 | 5 | status 0 | status 4 |
| complete separation by one predictor, 400 rows | 4 | | |

Below the budget the fit is status 3. The tolerance column is the old caveat with its rule now
stated: the step of the separating coefficient in the scaled coordinates of the fit is about
`sqrt(m / n) / eta`, so a tolerance not well below `0.2 sqrt(m / n)` passes as converged before
the certificate forms. The default 1e-5 is below that for every row count the kernel accepts
(by this estimate; the largest sample run is 1,000,000 rows). Both are in the class
documentation and `docs/api`.

The whole regression folder after the second round: headless 47 files, 499 tests (the 470, 22
in the panel spec, 6 added to the probit spec, 1 for the multiplication helper); node 34 files,
202 tests; all passing. No numerical assertion was loosened. Assertions changed:
`spatial-two-stage-least-squares.node.spec.ts` (three more nodes),
`math-review-lag-conditioning.spec.ts` (the 2SLS bounds tightened as above).

The reviewer's probe set (`review-regression/gpu-probe-specs/`, 10 files, 199 probes with the
data of `probe-data*.json`) was copied into a scratch folder under `modules/analysis/test`, run
on the final kernels and deleted. Against the reviewer's own output: 188 probes in common; the
status changes are the panel fits (7, to status 0), one row of 100,000 (0 to 4) and one row of
600 with 6 iterations (3 to 4); the fields that moved by more than 1e-3 are the 2SLS, panel and
probit ones above; the one error that grew is the Anselin-Kelejian value explained above.

For the session that regenerates lens records, on top of the list of the first round:

- `GPUSpatialPanelLagMaximumLikelihood`: node count unchanged; the text of `pooled-means`,
  `pack` and of each fit kernel (`random-fit`, `unit-fit`) changed; the `pooled-means`
  transient is one word longer.
- `GPUSpatialTwoStageLeastSquares`: three more nodes (`refinement-tiles`, `refinement-merge`,
  `refine`, between `solve` and `residuals-tiles`); the workspace has `predictorCount + 2` more
  words, so the constants block of every kernel changed again; the lag declarations (three
  private variables) and so every row kernel, `residuals-tiles` and `finish` changed. New
  transients `partial-refinement` and `refinement-sums`.
- `GPUSpatialLagMaximumLikelihood`: node count as after the first round; the four `DIRECT`
  slots are gone and one final sum is added, so the constants block of every kernel and the
  length of `final-sums` changed; the bodies of `rho-search`, `final-moments-tiles` (one more
  binding), `variance` and `finish` changed.
- `GPUProbitRegression`: node count unchanged; the state has `3 predictorCount + 1` more words
  (the column extremes and the candidate direction), so the constants of every kernel that
  reads the state changed; `initial-likelihood` has `4 + 2 predictorCount` terms; the bodies of
  the likelihood tiles and of each `solve` changed.
- `GPUOrdinaryLeastSquares` and `GPUSpatialErrorGM`: unchanged in this round.

Fixtures and generators. New `test/oracles/statistics-regression-panel-lag-conditioning.py`
and its fixture; `statistics-probit-tails.py` gained six large-sample scenes (no existing value
of its fixture changed; it now takes about a minute, most of it the linear programs and
statsmodels on 1,000,000 rows). All four generators of this stream rerun with
`node scripts/run-oracle-fixture.mjs`: IDENTICAL. `GENERATORS.md` lists the new one.

Not verified in the second round:

- frc-regression-12 has no spec; its evidence is the mutation run.
- The panel kernel was not given a first-order check, and its golden section (32 steps on
  [-0.999, 0.999]) resolves `rho` to 4e-7, a third of its standard error on the 3e-9 scenes.
- The refinement with `instrumentOrder: 2` and with the `robust` covariance beyond the
  existing specs (they pass); more than one predictor in the scenes with unequal row sums.
- Probit: separation by a column that is not at its lowest or highest value on the rows it
  does not separate (a three-valued column) with a group below 2e-5 of the rows; the default
  tolerance above 1,000,000 rows (derived only).
- Lag ML with row sums 2 and 3 is unchanged (`sigma2` within 2.5e-4), which the re-review
  accepted. `GPUSpatialPanelErrorMaximumLikelihood` was not read.
- Any GPU other than the test GPU; the full suites, `yarn build`, the benchmark specs.

Gates (second round): source typecheck watcher clean for `spatial-regression`; the scoped `tsc`
over the eight changed specs clean; `biome check` on the ten changed TypeScript files;
`check-contributor-contract` no new findings; `check-upstream-first` no violations; headless
the regression folder (47 files, 499 tests) and the recipe and value-cube composition specs (2
files, 5 tests); node the same folder (34 files, 202 tests) and the kernel audits, recipes,
oracle fixture runner and the two lens node specs (69 files, 2,927 tests).

#### Third round (2026-11-11): the second reviewer

The fixes for the eight findings of
[fix-review-closing-regression-codex.md](./fix-review-closing-regression-codex.md), a float64
review that ran nothing on a GPU. Same GPU and references as above (headless Chromium, WebGPU,
Apple Metal; spreg 1.9.1, libpysal 4.15.0, statsmodels 0.15.0, NumPy 2.5.4, SciPy 1.16.3 on
CPython 3.14, float64 on the float32 inputs). Each finding's input was first run on the kernels
as they stood ("before" below); seven of the eight reproduced on the GPU, and frc2-regression-08
is a documentation defect that the GPU confirms.

Design rules of this round. Probit: status 0 only for a maximum; status 4 only with a
certificate of separation in which the sign of every row's move is known (exact comparisons, or
a float32 value beyond a proven evaluation error bound), with no tolerance; everything else is
status 3 with the iterate. A check that sets status 3 is scale-free and allows for the float32
resolution of what it tests. The three probit invariants are stated in the class documentation
and in `docs/api`.

| Finding | Outcome | Before (measured on the GPU) | What changed | Spec that pins it | Evidence after |
| --- | --- | --- | --- | --- | --- |
| frc2-regression-01. Lag ML rejects `S = 0` outside the interval and `e1'e1 = 0` | fixed | Both of the reviewer's inputs status 3, every output NaN | `gpu-spatial-lag-maximum-likelihood.ts`: `moments-merge` writes `rho0 = 0` when `e1'e1 = 0`; `rho-search` takes `e1'e1 = 0` as a constant sum of squares, and searches with `S = 0` (or a rounding-sized negative `S`, above `-1e-6 r'r`) when the minimizer `rho*` is outside the search interval. `S = 0` with `rho*` inside the interval is still status 3 (a pole) | `math-review-lag-conditioning.spec.ts` ("degenerate", two tests) | Six-row ring: `rho` 0.78077638 against spreg 0.78077641 (1.6e-7 standard errors), `sigma2` 3.7e-8, log-likelihood 3e-8, standard errors 2e-8. Eight rows with `W y = x`: `rho` -2.1e-13 (reference 0), `sigma2` 0.5, standard errors 0.25, 0.3535534, 0.25. Reference: spreg `ML_Lag(method='full', epsilon=1e-11)` |
| frc2-regression-02. The first-order check overflows and cannot pass below the float32 spacing of `rho` | fixed | Response times 1e10: status 3 (times 1 and 1e-10: status 0). The 64-row scene (standard error of `rho` 4.6e-9): status 3 | `finish` of the same kernel forms `t = u'e1 / u'u` and `c = n (e1'e1 / u'u - 2 t^2) + tr2` and tests `\|g\| <= sqrt(0.25 c) + 4 c step`, with `step` one float32 step of `rho` (`STATIONARITY_PARAMETER_STEPS`, derived in its comment: bisection one step, the product rounding of the direct sums and of the residuals one each). The two pseudo R2 are formed from ratios too | Same spec: "the same status and rho for a response multiplied by 1e-10, 1 and 1e10", "accepts the float32 nearest", "rejects a rho displaced from the stationary point" (the mutation spec: a node inserted after `rho-search` offsets the stored `rho`); `spatial-maximum-likelihood-nodes.node.spec.ts` pins the workspace slot and the ratio form | Scales: status 0 three times, `rho` 0.40891266, 0.40891272, 0.40891266 (reference 0.40891268 to 0.40891269; 2e-6 standard errors), `sigma2` within 3e-8, pseudo R2 equal to 1e-5. 64 rows: status 0, `rho` 0.40000004 (the float32 next to the nearest; reference 0.4000000032 in exact rational arithmetic). Mutation on `high-fit-slope-200-center-10000-stream-1001`: unmodified and zero offset status 0 with the same table, offsets of +1e-4 and -1e-4 status 3 with NaN outputs; on the scale scene a displacement of 0.2 standard errors is status 0 and of 1 standard error status 3 |
| frc2-regression-03. The probit certificate accepts negative moves up to 1e-3 | fixed, by the design rule | The 3,002-row input: status 4 at the defaults and at budgets of 1, 2, 4, 8, 16, 32 iterations | `gpu-probit-regression.ts`: the tolerance test is gone. A certificate is one of: a column that separates at its lowest or highest value (four exact flags per column from the initial pass); a single-column threshold from the Fisher direction, and the two data values next to it one pass later (exact comparisons); the candidate direction when every row's move is beyond the evaluation error bound `(k + 5) 2^-23` of the sum of the magnitudes of its terms or exactly zero (`getSeparationErrorBound`). A pinned fit and an overflowing linear predictor are status 3, not 4 | `math-review-probit-bound.spec.ts` (two "within 1.5e-5 of separation" tests; the pinned-fit test); `probit-regression.node.spec.ts` (no ratio test in the kernel text) | Status 0 after 23 steps: coefficients (4.167e-6, 6.021896) against (4.178e-6, 6.022116), 1e-8 and 2e-6 standard errors; standard errors within 6.3e-4; log-likelihood within 2e-6 relative. Budgets 1 to 16: status 3 with the iterate. Reference: statsmodels `Probit` and the tail-exact Newton solve (1e-9 apart), separation objective 0 |
| frc2-regression-04. Probit outputs rebuilt from the raw float32 coefficients | fixed | Predictor near 2^24: fitted probabilities 0.1848, 0.4213, 0.6915, 0.8847, 0.9711 against 0.1362, 0.3449, 0.6178, 0.8409, 0.9551; marginal effect 0.08876 against 0.09119; Pinkse, Kelejian-Prucha, Pinkse-Slade 2998.6, 49.79, 2432.0 against 2621.7, 48.18, 2070.5. (The reviewer's float64 emulation predicted 0.5 for the second group; the GPU fuses the product) | `predict`, the `effects` tiles and the diagnostic residuals compute the linear predictor from `getDesign` and the centered estimate; `predict` binds the state and the status instead of the coefficients. The raw coefficients are published as before | Same spec ("for a predictor near 2^24"); node spec (the three kernels read `getDesign`, not `coefficients`) | Fitted within 1e-8 of statsmodels `predict`, marginal effect 7e-8 of `get_margeff`, the three statistics 7e-7 of spreg `Probit(spat_diag=True)`; slope and its standard error as statsmodels; raw intercept -5860769.5 against -5860770.6 (a float32 step there is 0.5) |
| frc2-regression-05. Panel lag ML quasi-demeaning regrouped | fixed (suspected by the reviewer, reproduced) | Entity effects of +3000 and -3000, 16 x 16, unit effects, residual deviation 0.004: all 256 published residuals multiples of 2^-12 (the float32 step of 3000); `rho` 0.026 standard errors off, slope 1.0e-6, `sigma2` +1.1%, within-entity residual error 7.7e-5 root mean square. At +4097 and -4097: 202 of 256 residuals multiples of 2^-11, `sigma2` +3% | `gpu-spatial-panel-lag-maximum-likelihood.ts`: `pack` stores each column as its within deviation (`getRoundedDifference` of the centered value and its entity mean) and its entity mean; `datum` is `deviation + phi mean`; the normal equations of each alternation step are a polynomial in `phi` of three moment sums taken once (`getQuasiMoment`); the random-effects `phi` update and the `residuals` node use the deviations | `math-review-panel-lag-conditioning.spec.ts` ("entity effects are large and of opposing sign"); run once against the kernel before the fix: fails with 256 of 256 | 0 of 256 residuals on the step; `rho` 0.0022 standard errors, slope 5e-8, within-entity residual error 3.3e-5, `sigma2` +0.58%. Reference: spreg `Panel_FE_Lag(epsilon=1e-9)` |
| frc2-regression-06. Three-valued predictor with mixed rows at the middle value | fixed (suspected, reproduced) | 100,002 rows: status 3 at the defaults (slope 11.4 written), status 2 with 128 iterations, status 3 with 8, status 0 with a tolerance of 1e-3 (slope 3.93, standard error 21.9) | The single-column threshold and its neighbouring data values (frc2-regression-03) | `math-review-probit-bound.spec.ts` (four option sets; "at any iteration budget and tolerance" for an indicator) | Status 4 at the defaults, 128 and 8 iterations, tolerance 1e-3. Oracle: linear program optimum 2, statsmodels does not converge |
| frc2-regression-07. Anselin-Kelejian NaN for a zero total weight | fixed | Status 0, statistic NaN, Moran's I NaN | `gpu-spatial-two-stage-least-squares.ts` `finish`: the statistic is `(u'Wu / sigma2)^2 / (T + 4 a / sigma2)`; Moran's I is NaN when the total weight is zero | `math-review-lag-conditioning.spec.ts` ("signed weights whose total is zero") | Statistic 0.95056707 against 0.95056706 (the cancelled formula on spreg `GM_Lag`'s residuals, `z` and `varb`; the generator holds that formula to spreg's `ak_test` at 1e-10 on the same pairs with positive weights); coefficients within 1e-7 of a standard error, `sigma2` 1e-7 |
| frc2-regression-08. Multiplication helper: cutoff and exactness claim | documentation fixed, helper unchanged | Exponent sum 173: returns 0 where the error is -4.93e-32. Product one step off: returns 2^-22 where the distance is 2^-22 + 2^-46 | `multiplication-rounding-error-wgsl.ts`: the stated domain is the exponent sum (every product below 2^-80 unreported, every product of 2^-79 or more reported, between them by the significands); exactness is claimed for a correctly rounded product only, and the rounding to float32 of a 25-bit distance is stated | `test/utils/multiplication-rounding-error-wgsl.spec.ts` (second test: five cases) | Both counterexamples return what the documentation now says; the same significands at an exponent sum of 174 and the correctly rounded product of the second pair return the exact error |

frc2-regression-03 and -06, what the probit kernel now certifies, measured.

| Scene | Smallest `maximumIterations` for status 4 | Before this round |
| --- | --- | --- |
| Indicator of 1, 3, every twentieth, every second row of 600; 1 row of 100,000 to 20 of 1,000,000 | any (certified before the first iteration), any tolerance | 6 to 10; status 0 for one row of 600 at a tolerance of 1e-2 |
| Complete separation by one predictor, 400 rows | 2 | 4 |
| Quasi-complete separation at an interior value (six tied rows) | 5 | not measured below the defaults |
| Complete separation by two predictors, 500 rows | 7 | not measured below the defaults |
| Three-valued predictor, mixed rows at the middle (100,002 rows) | 8 or fewer (8 and the defaults run) | none |
| Quasi-complete separation by two columns with tied rows on the boundary (606 rows) | none: status 2 at the defaults, status 3 with 8 and 16 iterations | not run |

The last row is the honest limit: a combination of columns whose tied rows have moves that
cancel to rounding has undecided rows, so it is never a certificate. It is not status 0.

frc2-regression-05, a side effect and how it was avoided. Guarding `datum` alone (the reviewer's
suggestion) made the alternation of the random-effects fit jitter: with per-row values rounded
afresh at each `phi`, the solve of two predictors with a squared cosine of 1 - 4e-6
(`math-review-pivot-limit.spec.ts`) did not settle below 1e-4 in `phi` and was status 3 after
128 iterations, where the unguarded kernel converged in 8 (the compiler's regrouping made the
sums smooth in `phi`). The moment polynomial restores that smoothness without the regrouping:
the scene converges in 4 iterations. Its estimate in the collinear direction is float32 noise
before and after (second slope -36.5 before, -51.8 now, spreg -30.6 with a standard error of
24.5); `phi` 0.3763 and 0.3808 against 0.3773.

Results on ordinary data. The regression spec folder with the helper spec was run before and
after:

- before: headless 47 files, 499 tests; node 34 files, 202 tests, all passing;
- after: headless 47 files, 515 tests (the 499, 6 added to the lag spec, 8 to the probit spec,
  1 to the panel spec, 1 to the helper spec); node 34 files, 204 tests (one added to each of
  the two node specs); all passing.

No numerical assertion was loosened. One assertion changed:
`math-review-probit-bound.spec.ts`, "probit whose maximum puts a misclassified row beyond a
linear predictor of 12" expected status 4 with NaN coefficients and now expects status 3 with
the pinned iterate (the data are not separated; the design rule keeps status 4 for a
certificate). `runEstimator` of the lag spec takes its row count from the response.

The first reviewer's probe set (`review-regression/gpu-probe-specs/`, 10 files, 199 probes) was
copied into a scratch folder under `modules/analysis/test`, run on the final kernels and
deleted. Against the second-round run (197 probes parsed in both): two status changes, the
pinned probit fit `misclassified-x8.0` (4 to 3, iterate within 0.8 standard errors of the
maximum) and one row of 600 at a tolerance of 1e-2 (0 to 4). The 60 small samples that are not
separated are all status 0 and the 12 separated ones status 4, as before. The 2SLS statistic
moved by 1e-7 or less. The panel fits moved inside their bands (the largest `sigma2` error on
the `slope-200` scenes is 1.5e-3; it was 1.7e-3): the mean square of the published residuals of
two random-effects fits moved from 8e-6 and 3e-4 to 1.3e-3, the level of the other scenes.

For the session that regenerates lens records, on top of the lists above. No node count
changed in this round.

- `GPUSpatialLagMaximumLikelihood`: the bodies of `moments-merge`, `rho-search` and `finish`.
- `GPUSpatialTwoStageLeastSquares`: the body of `finish`; and the WGSL text of
  `getMultiplicationRoundingError` (comments only), which is part of the refinement kernels.
- `GPUSpatialPanelLagMaximumLikelihood`: the text of `pack`, of each fit kernel (`random-fit`,
  `unit-fit`) and of `residuals`.
- `GPUProbitRegression`: the state layout grew (`6 predictorCount` words for the variance pass
  instead of `predictorCount`, `2 predictorCount` raw extremes, `16 + 6 predictorCount` for the
  initial pass instead of `4 + 2 predictorCount`, one undecided flag, nine threshold words, 62
  line-search words instead of 50), so the constants of every kernel that reads the state
  changed; the `variance` tiles have `5 predictorCount` maxima; `initial-likelihood` has
  `16 + 6 predictorCount` terms and each `line-search` 62; the bodies of the `variance` tiles,
  `initialize`, the likelihood tiles, each `solve`, each `decide`, `finish`, `predict`, the
  `effects` tiles and `diagnostic-residuals` changed; `predict` and `diagnostic-residuals` bind
  the state (and `predict` the status) instead of the coefficients.
- `GPUOrdinaryLeastSquares`, `GPUSpatialErrorGM` and the error ML kernels: unchanged.

Fixtures and generators. No new generator. `statistics-regression-lag-conditioning.py` gained
`edgeCases` (five scenes with their inputs stored in full), `statistics-probit-tails.py` three
scenes (`nearlySeparated`, `largeOffsetPredictor`, `threeValuedMiddle`; it now imports spreg
and libpysal and records their versions), `statistics-regression-panel-lag-conditioning.py`
`opposingEffectScene`. No existing value of the three fixtures changed (compared before and
after). All four generators of this stream rerun with `node scripts/run-oracle-fixture.mjs`:
IDENTICAL. `GENERATORS.md` was not edited (outside this stream): its three descriptions do not
mention the new scenes.

Not verified, or left, in the third round:

- frc2-regression-02: where the standard error of `rho` is below its float32 step the kernel
  returns `rho` and status 0, but `sigma2`, the log-likelihood and the standard errors are those
  of residuals that are float32 rounding (`sigma2` 33 times the float64 value on the 64-row
  scene, the standard error of `rho` 5.7 times). No output flags it; stated in the class
  documentation and `docs/api`. The `residuals` output of a fit rejected by the first-order
  check holds the residuals of the rejected `rho`, not NaN; documented, not changed.
- frc2-regression-03: the guard against a rounding step on data with an undecided direction
  (`S_UNDECIDED`) is exercised by no scene. Separated data that are not certified end as
  status 3 or as status 2 (singular information of the runaway fit); status 2 was not renamed.
  An overflowing linear predictor is status 3 by the code; no scene overflows.
- frc2-regression-05: the entity means of the pack node are sequential float32 sums (off by up
  to 6e-4 for an effect of 3000, `sigma2` 0.58% high) and the lag of the centered response is
  rounded to a float32 step of the effect; neither was changed. Random effects with large
  opposing entity effects were not run against spreg (the `phi` of such data is below the
  kernel's lower limit of 1e-5). `GPUSpatialPanelErrorMaximumLikelihood` was not read.
- The reviewer's "could not check" list beyond the eight findings (refinement with several
  large-mean predictors, second-order instruments, chunked weights, stochastic traces, tail
  extrapolation, exact-determinant mode, candidate cleanup in many-predictor designs) was not
  probed; the existing specs for those modes pass.
- Any GPU other than the test GPU; the full suites, `yarn build`, the benchmark specs.

Gates (third round): source typecheck watcher clean for `spatial-regression` and the helper; a
scoped `tsc` over the six changed specs clean; `biome check` clean on the twelve changed
TypeScript files; `check-contributor-contract` no new findings; `check-upstream-first` no
violations; headless the regression folder and the helper spec (47 files, 515 tests) and the
recipe, value-cube composition and device-limit specs (3 files, 7 tests); node the same folder
(34 files, 204 tests) and the kernel audits, recipes, oracle specs and short-distance warnings
(70 files, 2,916 tests).

#### Fourth round (2026-11-11): the re-review of round three

The three findings of "Re-review of round three (2026-11-11)" in
[fix-review-closing-regression.md](./fix-review-closing-regression.md). Same GPU and references.
Changes were kept to one kernel (the panel fit) and to documentation.

| Finding | Outcome | Before (measured on the GPU) | What changed | Spec that pins it | Evidence after |
| --- | --- | --- | --- | --- | --- |
| frc-regression-13. Panel random effects hold `phi` at 1e-5 with status 0 | fixed: `phi` is followed to 1e-8, and an estimate below that is reported | Entity effects of +3000 and -3000, residuals of 0.004 (the stream scene of the fixture; spreg `phi` 2.45e-7): status 0, boundary bit 2, `phi` 1e-5, `sigma2` 105 times spreg's, `rho` 3.5 standard errors off, log-likelihood 595 low. The reviewer's four scenes: `sigma2` 28 to 119 times | `gpu-spatial-panel-lag-maximum-likelihood.ts`, the random-effects fit kernel only: the lower limit is `RANDOM_EFFECTS_MINIMUM_PHI = 1e-8`; an estimate below it is held there, status 3, boundary bits 2 and 4 (a `phi` of 1 sets bit 2 alone, as before); the alternation stops on the change of `phi` against `tolerance` times `min(1, phi / 0.01)`, because an absolute 1e-5 is no test for a `phi` of 3e-7 | `math-review-panel-lag-conditioning.spec.ts`: "follows a phi of 2.5e-7 and matches spreg Panel_RE_Lag", "reports a phi below its lowest value as status 3 with its own boundary bit". Both fail on the round-three kernel | Fixture scene: status 0, no boundary bit, `phi` 2.459e-7 against 2.454e-7 (0.19%), `rho` 0.009 standard errors, intercept 0.20, slope 3e-7, `sigma2` +1.1%, log-likelihood 1.5 low, standard errors 5.7e-3. The reviewer's four scenes (its probe spec, rerun): status 0, coefficients within 0.058, 0.0065, 0.32 and 0.073 standard errors, `sigma2` within 0.7%, standard errors within 3.5e-3, log-likelihood within 0.9; its two fits with a `phi` of 7e-5 and 9e-5 moved nearer to spreg (`sigma2` 6e-4 to 6e-5 and 8e-5); its other 52 fits unchanged (same status, every error within 20% of its earlier value). Effects of 16,000,000 with residuals of 0.5 (spreg `phi` 9.1e-9): status 3, bits 6, `phi` 1e-8, no covariance. Reference: spreg `Panel_RE_Lag(epsilon=1e-9)`, held by the generator to the first-order condition in `phi` |
| frc-regression-15. `fitted` moves with the convergence error of `phi` | verified, documented, pinned; no code change | Reviewer: 1.2 residual deviations on slope 200 with a response mean of 10,000, `phi` within 4.5e-7 of spreg; residuals within 0.05 | Nothing in the kernel. `fitted` already is the quasi-demeaned response at the published `phi` less the published residual (`fitted + residuals = y - (1 - phi) mean_i(y)`); the difference from spreg is the difference in `phi` times the entity mean, which is in the quasi-demeaned response itself. Stated in the class documentation, the output's prop documentation and `docs/api` | Same spec: "publishes a fitted value that is the quasi-demeaned response at the published phi less the residual" | The identity holds to 2.1e-5 on `slope-200-center-10000-stream-21` (bound 1e-4) |
| frc-regression-14. Lag ML near-exact fit: the stated cause was wrong | documentation corrected; no refinement step | 64-row scene: `sigma2` 32.8 times the float64 value, status 0 | The class documentation and `docs/api` name the cause with the reviewer's numbers: the residuals are `e0 - rho e1` of two float32 OLS fits and are not orthogonal to `[1, x]`; 91% of their sum of squares lies in that span; the float32 floor of the residual arithmetic is 1.04 to 1.65 times | `math-review-lag-conditioning.spec.ts`, end of the "float32 nearest" test: `sigma2` below 45 times, and below 6 times with the share in the span of `[1, x]` taken out | Measured in the spec: ratio 32.78, share in the span 0.9126, remainder 2.87 (the reviewer's 32.8, 91% and 2.9) |

frc-regression-14, why no refinement. A regression of the published residuals on `[1, X]` is a
third `GPUOrdinaryLeastSquares` composition and a correction of `b`, of the residuals and of
every sum taken from them: more nodes and a changed record for every lag ML fit, to improve a
fit that is exact to float32. It is described in the class documentation as what would remove
the limit.

frc-regression-13, what was tried and not kept. The float32 entity means of the pack node are
what is left on these scenes (`sigma2` 1.1%, the intercept 0.2 to 0.3 standard errors). Two
forms of a more accurate entity mean were run: deviations from the first period's value, and a
correction of the mean by the mean of the deviations. Both brought the reviewer's opposing-effect
fits to `sigma2` within 0.2% and coefficients within 0.03 standard errors. The first moved an
ordinary high-fit scene past a bound of the existing spec (`slope-20`, `sigma2` 2.2e-4 against
2e-4); the second made a time-invariant predictor under unit effects status 0 instead of 2
(`statistics-advanced.spec.ts`, "within-singular": its deviations were no longer exactly zero).
Neither was kept; the pack node is unchanged in this round.

Results. The regression folder now also holds the other session's `*-operation` specs. Headless,
folder and helper spec: 54 files, 569 tests, all passing; of these the 47 files of the earlier
rounds have 518 tests (the 515, and 3 added to the panel spec). Node: 41 files, 258 tests, all
passing; the 34 files of the earlier rounds have 204, unchanged. No assertion of an existing
test changed in this round; the "float32 nearest" test of the lag spec gained three assertions.

Kernels whose text changed in the fourth round: `GPUSpatialPanelLagMaximumLikelihood`
`random-fit` only (`unit-fit`, `pack` and `residuals` are as after round three). Node counts
unchanged. `GPUSpatialLagMaximumLikelihood`: comments in the TypeScript documentation only.

Fixtures and generators. `statistics-regression-panel-lag-conditioning.py` gained the `random`
reference of `opposingEffectScene` and `phiBelowLimitScene` (no existing value changed; rerun
with `node scripts/run-oracle-fixture.mjs`: IDENTICAL). The three regression rows of
`GENERATORS.md` (probit tails, lag conditioning, panel lag conditioning) now describe the
scenes of rounds three and four.

Not verified in the fourth round:

- Panels with more than one predictor at a `phi` below 1e-5; the Hausman comparison with such a
  `phi`; a `phi` between 1e-8 and 2e-7 (float32 data with 16 periods do not reach it: the
  below-limit scene sits at 9e-9 with residuals that are mostly the float32 rounding of the
  response).
- The relative stopping rule changes when a fit with a `phi` below 0.01 stops (the reviewer's
  two fits at 7e-5 and 9e-5 take 5 sweeps instead of 3 and are nearer to spreg); no fit with
  such a `phi` was seen to stop later than the iteration budget.
- Any GPU other than the test GPU; the full suites, `yarn build`.

Gates (fourth round): source typecheck watcher clean for the two kernels; the scoped `tsc` over
the changed specs clean; `biome check` clean on the four changed TypeScript files;
`check-contributor-contract` no new findings; `check-upstream-first` no violations; headless
the regression folder and the helper spec (54 files, 569 tests); node the folder (41 files, 258
tests) and the kernel audits and oracle specs (53 files, 2,718 tests).
