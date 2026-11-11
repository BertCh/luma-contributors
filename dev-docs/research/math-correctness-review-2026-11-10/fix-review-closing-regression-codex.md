# Review of the second-round regression fixes (2026-11-11)

An independent read-only review by a Codex agent (`gpt-6.1-sol`) of the regression kernels after the two fix rounds recorded in [regression.md](./regression.md) and [fix-review-closing-regression.md](./fix-review-closing-regression.md). The reviewer could read the code and run float64 probes (NumPy 2.5.4, SciPy 1.16.3, spreg 1.9.1, statsmodels 0.15.0); it could not run a kernel on a GPU. Its text follows unedited. What was done about each finding is in the third-round block of [regression.md](./regression.md).

CONFIRMED below means demonstrated by algebra, Python integers, or CPU numerical probes on the stated inputs. No finding was verified by running the GPU.

**frc2-regression-01 — major — CONFIRMED: lag ML rejects valid likelihoods when `S = 0` or `e1'e1 = 0`.**

File: `modules/analysis/src/gpu-spatial-analysis/spatial-regression/gpu-spatial-lag-maximum-likelihood.ts`, lines 848, 950–955.

The search requires both `laggedSquares > 0` and `residualSquares > 0`. Neither condition is necessary for a finite, well-defined concentrated likelihood over the search interval.

Two concrete counterexamples:

- **`S = 0`, vertex outside the interval.** Six-row ring, two neighbours of weight `0.5`; `y = [2, 1, -1, -2, -1, 1]`; predictor `x = [1, -1, 1, -1, 1, -1]`. The OLS residuals are exactly `e0 = y`, `e1 = 0.5 y`, giving `C = 3`, `rho* = 2`, `S = 0`. Nevertheless, `RSS(rho) = 12(1 - rho/2)^2` is strictly positive throughout the admissible interval. The code rejects the search and produces status 3. `spreg.ML_Lag(method='full', epsilon=1e-11)` gives `rho = 0.78077640735`, `sigma2 = 0.74325308444`, log likelihood `-8.89478870858`.
- **`C = 0`.** Eight rows, with unit-weight reciprocal pairs `(0,4), (1,5), (2,6), (3,7)`; `y = [-1,1,-1,1,0,0,0,0]`; `x = Wy = [0,0,0,0,-1,1,-1,1]`. Both means are zero, `e0 = y`, and `e1 = 0` exactly. RSS is constantly 4, while `logdet = 4 log(1-rho²)` has its unique maximum at zero. The code divides `0/0` and rejects the search. The correct result is `rho = 0`, slopes/intercept zero, `sigma2 = 0.5`; spreg returns these, with log likelihood `-8.57891954340` and finite standard errors `[0.25, 0.35355339, 0.25]`.

Suggested fix: handle `C = 0` with a constant RSS, as the panel kernel already does. Permit `S = 0` when the vertex lies outside the admissible interval. Validate RSS at evaluated parameters; distinguish an admissible zero-RSS pole from a harmless zero minimum outside the interval.

**frc2-regression-02 — major — CONFIRMED: the stationarity check can reject a correct fit through overflow or float32 parameter resolution.**

File: `gpu-spatial-lag-maximum-likelihood.ts`, lines 1396–1403.

The derivative is correct, but its numerical evaluation and acceptance threshold are defective.

**Overflow on an ordinary fit.** Reproducible input:

```python
rng = np.random.default_rng(20261111)
x = rng.uniform(-2, 2, 600).astype(np.float32).astype(float)
e = rng.normal(size=600).astype(np.float32).astype(float)
# W: 600-row ring, two neighbours of weight 0.5
y = (np.linalg.solve(np.eye(600) - 0.4*W, 1 + 2*x + e) * 1e10).astype(np.float32)
```

At the float64 concentrated maximum:

- `rho = 0.408912693532`
- `R = u'u = 5.78103578033e22`
- `D = u'e1 = 1.35441032559e22`
- `C = e1'e1 = 2.28049263684e23`
- Correct curvature: `2746.72591139`.

The written float32 expression evaluates its second curvature term through overflowing numerator and denominator, producing `NaN`; the check consequently rejects it. All three moments themselves fit comfortably in float32.

Compute the dimensionless ratios first:

```text
t = D/R
c = n*(C/R - 2*t*t) + traceTwo
```

Also avoid overflow in `score*score` by testing a normalized score.

**The nearest float32 rho need not satisfy the threshold.** A second concrete input:

```python
rng = np.random.default_rng(73)
y = rng.integers(-10, 11, 64).astype(float)
y[32:] = -y[:32]
# W: 64-row ring, weights 0.5
x = (y - 0.4*(W@y)).astype(np.float32).astype(float)
```

Float64 profiling on these float32 inputs gives:

- RSS vertex `0.40000000317695344`
- Nearest float32 rho `0.4000000059604645`
- Minimum RSS `1.51796396846e-12`
- Profile standard error approximately `4.62680e-9`
- Float32 spacing of rho `2.98023e-8`
- At that nearest rho, `g²/c = 0.363988`, exceeding `0.25`.

Thus even the correctly rounded reference parameter fails the check when evaluated with accurate residual moments. This is a check-level demonstration; GPU OLS rounding can change the actual moments.

Suggested fix: include parameter quantization and residual-moment error in the stationarity allowance, or retain a low part of rho through residual evaluation.

The missing spec is concrete: insert a test-only node immediately after `rho-search` that offsets its stored rho, then run every downstream node normally. On `high-fit-slope-200-center-10000-stream-1001`, require the unmodified fit to be status 0 and shifts of both signs by `1e-4` to produce status 3 and NaN outputs. Separately pin response-scale invariance and the nearest-float32 case above. Ordinary successful fits alone do not pin rejection.

**frc2-regression-03 — major — CONFIRMED: the probit “certificate” accepts genuinely negative signed moves on non-separated data.**

File: `gpu-probit-regression.ts`, lines 393–396, 570–579, 602–603.

Removing components from a candidate direction is mathematically safe **only if the resulting direction is verified to satisfy the separation inequalities**. This verification instead permits negative moves up to `1e-3` of each row’s sum of absolute terms. That is a geometric tolerance, not a rounding guarantee.

Concrete float32 input:

- 1,000 rows: `x = -1`, `y = 0`.
- 2,000 rows: `x = +1`, `y = 1`.
- One row: `x = float32(-1e-5)`, `y = 1`.
- One row: `x = float32(+1e-5)`, `y = 0`.

These data have no nonzero separating direction. The endpoint observations exclude a negative slope; the two central observations exclude a positive slope; zero slope forces zero intercept.

At the first Fisher step, the centered/scaled candidate is approximately `[0.41749305, 1.18129139]`. Neither cleanup operation changes it. The two central signed moves are:

```text
-1.25300099e-5
-1.25362720e-5
```

Both are genuinely negative. Nevertheless:

```text
largest wrong-side ratio = 1.50139747e-5
largest right-side ratio = 1
```

The code records separation immediately and keeps it permanently. A float64 replay converges in 23 steps to a finite maximum: raw intercept approximately `4.17755e-6`, slope `6.02211527`, log likelihood `-1.38639304475`. The retained flag converts that convergence into status 4.

Suggested fix: certify nonnegative signed dot products using rigorous sign evaluation. Ambiguous rounding intervals must remain uncertified; exact zeros need an exact tie calculation. Reducing a fixed tolerance does not make this a valid certificate.

**frc2-regression-04 — major — CONFIRMED: probit predictions, marginal effects and diagnostics discard the centered fit’s accuracy.**

File: `gpu-probit-regression.ts`, lines 665, 674–677, 731–734.

These passes reconstruct linear predictors from the published raw float32 intercept and slopes. The iteration and reported likelihood use the centered state. Consequently the outputs can describe substantially different fits.

Concrete input: five groups of 1,000 rows, with `z = -2,-1,0,1,2`, predictor `x = float32(2^24 + 2z)`, and respectively `[136,345,618,841,955]` positive responses. Every group contains both outcomes, so separation is impossible.

The float64 probit maximum in z coordinates is approximately:

```text
intercept = 0.29961284
slope     = 0.69865819
```

Applying the kernel’s float32 coefficient publication and prediction arithmetic gives:

| z | Centered linear predictor | Raw reconstruction |
|---:|---:|---:|
| -2 | -1.097704 | -1 |
| -1 | -0.399045 | 0 |
| 0 | 0.299613 | 0.5 |
| 1 | 0.998271 | 1 |
| 2 | 1.696929 | 2 |

At `z = -1`, fitted probability becomes `0.5` instead of `0.3449299`. At `z = 0`, it becomes `0.6914625` instead of `0.6177638`. Average marginal effect becomes approximately `0.09005285` instead of `0.09118714`. The reconstructed predictions imply log likelihood `-2400.22156`, versus the fitted maximum `-2328.49603`.

The central-row error survives fused multiplication: multiplication by `2^24` is exact for the published slope.

Suggested fix: calculate predictions, densities, marginal effects and diagnostic residuals from `getDesign(row)` and `S_ESTIMATE`. Keep raw coefficients as a presentation output.

**frc2-regression-05 — major — SUSPECTED: panel quasi-demeaning remains exposed to the documented reassociation failure.**

File: `gpu-spatial-panel-lag-maximum-likelihood.ts`, lines 275–279 and 394.

`datum` returns an unguarded quasi-demeaned difference. `getRowResiduals` subsequently subtracts fitted terms from it. The pooled-mean guards in `pack` do not materialize this later subtraction of the entity mean.

For unit effects, the vulnerable expression is exactly the documented pattern:

```text
(centeredResponse - entityMean) - fittedTerm
```

Concrete float32 operands:

```text
centeredResponse = 4097.00048828125
entityMean       = 4097
fittedTerm       = 0.0003662109375
```

Evaluating the first difference first gives `0.0001220703125`. Regrouping as `centeredResponse - (entityMean + fittedTerm)` gives zero.

The same issue exists in the published residual expression. Large entity effects remain large after pooled centering; this is independent of the correctness of the centering algebra.

Suggested fix: materialize the quasi-demeaned difference with `getRoundedDifference` inside `datum`, and use the same protected operation in publication. Pin large opposing entity effects with small within-entity variation. Whether Metal currently performs this reassociation here needs GPU execution.

**frc2-regression-06 — major — SUSPECTED: default probit separation detection still misses a three-valued predictor whose non-separated rows occupy its middle value.**

File: `gpu-probit-regression.ts`, lines 575–578, 603, 617–622.

The intercept cleanup only snaps a corner at the minimum or maximum predictor values. It cannot remove an intercept direction on rows at an interior value.

Concrete quasi-separated input, 100,002 rows:

- `x = 0`: 40,000 responses zero and 60,000 responses one.
- `x = -1`: one response zero.
- `x = +1`: one response one.

The direction `(intercept, slope) = (0,1)` is an exact separation certificate. Both predictor extremes and its mean are represented exactly.

A float64 grouped replay of the current scoring and candidate rules, at the defaults, records no certificate in 64 steps. The mixed-outcome middle group has `rowScale = |candidateIntercept|`; whenever that intercept direction is nonzero, one outcome therefore has wrong-side ratio 1, however tiny the direction is. Neither endpoint corner is near zero, so cleanup does nothing.

After 64 steps the linear predictors are approximately:

```text
[-11.14607508, 0.25334710, 0.25334710, 11.65276929]
```

The last scaled slope step is `4.0119e-4`, above the convergence threshold `1.25335e-5`; this replay exhausts with status 3.

Suggested fix: include actual interior support values when constructing a candidate, or use a more general certificate search. GPU rounding could accidentally make the intercept direction exactly zero and recover this example; that is why the GPU outcome remains SUSPECTED. Add this scene to the default-settings specs.

**frc2-regression-07 — minor — CONFIRMED: accepted signed weights with zero total weight make the Anselin–Kelejian statistic NaN unnecessarily.**

File: `gpu-spatial-two-stage-least-squares.ts`, lines 1442–1446.

Moran’s I is undefined when `S0 = sum(W) = 0`. But `S0` cancels completely from the AK statistic:

```text
AK = (u'Wu)^2 / [sigma2^2 * (T + 4a/sigma2)].
```

The implementation computes infinite intermediate normalizations and then `inf/inf`.

Concrete eight-row input:

- Reciprocal pairs `(0,1), (2,3), (4,5), (6,7)`, with respective weights `+1,+1,-1,-1` in both directions.
- `x = [-2,-1,0,1,2,4,-3,5]`.
- `y = [1,3,-1,2,7,0,-2,8]`.

Instruments and regressors both have full rank. Float64 2SLS gives approximately:

```text
delta = [1.67414722, 0.78296429, 0.01137044]
RSS   = 56.31016998
u'Wu  = 47.43964865
T     = 16
a     = 55.93471640
AK    = 0.9505670553
```

The current normalized expression produces NaN AK while leaving solver status 0.

Suggested fix: calculate AK in its cancelled form. Moran’s I can remain explicitly undefined for this input.

**frc2-regression-08 — minor — CONFIRMED: the multiplication helper’s exactness guarantee exceeds what its cutoff and single-float return support.**

File: `modules/analysis/src/analysis-core/kernels/multiplication-rounding-error-wgsl.ts`, lines 21–25, 49–50, 78–79.

Two counterexamples:

1. **Cutoff suppresses products above `2^-80`.** Let:

   ```text
   a = (1.5 + 2^-23) * 2^-41
   b = (1.5 + 2^-23) * 2^-40
   ```

   Their exact product is approximately `9.30578337034e-25`, above `2^-80 ≈ 8.27180612553e-25`. The true rounding error is `-4.93038006988e-32`, a normal float32 value. The helper returns zero because the biased exponent sum is 173.

2. **An adjacent, inexact product can require 25 significant error bits.** Set `a = b = 2 - 2^-23`, and provide product `p = 4 - 3*2^-22`, one float32 step below the correctly rounded product. Then:

   ```text
   exact a*b - p = 2^-22 + 2^-46
   helper return = 2^-22
   ```

   Integer subtraction obtains `errorUnits = 2^24 + 1` correctly, but converting that integer to float32 loses the final bit. The statement that any product within two ULPs has an exactly representable error is false.

Suggested fix: document the actual exponent-based reporting domain, or handle exponent sum 173 with an adjusted scaling. Restrict the exactness guarantee to correctly rounded products, or return an error expansion when accounting exactly for inexact multiplication.

**Checked and found correct**

**Direct-form algebra and centering.** Let `C = e1'e1`, choose any `rho0`, and define:

```text
r = e0 - rho0 e1
D = r'e1
rho* = rho0 + D/C
S = r'r - D²/C
```

Then, in real arithmetic,

```text
||e0 - rho e1||² = S + C(rho - rho*)².
```

This is the same concentrated RSS used by spreg. I checked the identity numerically with signed, unequal-sum weights and an island. It agrees to float64 rounding.

It is not unconditionally a sum of two nonnegative **computed float32** terms: `S` still subtracts two quantities. When `r` is almost parallel to `e1`, that subtraction can lose the remaining RSS. Rounded `rho0*e1` also means the direct moments need not describe precisely the same residual polynomial as the publication pass. Small `C`, division overflow/underflow and zero `S` require explicit handling.

For cross-sectional ML,

```text
v = W(y-c) + c(s_i-s) = Wy - cs.
```

The removed `cs` is a constant and is correctly restored through the lag intercept. This identity holds for islands, unequal row sums and negative weights. Those cases can nevertheless leave a large lag column and restore cancellation: exact row-sum bookkeeping does not make subsequent products and sums exact. In addition, `rowSumError += error` is itself a float32 accumulation, so “exact row sums” is not a universal guarantee for arbitrarily scaled signed weights.

The determinant remains that of the original `I-rho W`; centering does not justify changing it. The code keeps it consistently. Cross-sectional default series mode approximates `ML_Lag(method='full')`; exact eigenvalue mode removes that approximation. Bounds also differ from spreg’s full interval.

**Panel transformation.** With `Q_phi = I-(1-phi)P`, `Q_phi 1 = phi 1`. The kernel centers first and quasi-demeans second. For a balanced panel with time-invariant weights, `Q_phi` commutes with `I_T ⊗ W`. Packing `Wy-c`, including `c(s_i-1)`, preserves the required transformation even when accepted float32 row sums are not exactly one. The intercept mapping

```text
b_raw = b_centered + (1-rho)c_y - c_x' beta
```

is correct. Here `phi = 1-theta` in the alternative theta notation.

The analytic update `phi² = within / ((T-1)*between)` is correct with the code’s repeated-row definition of `between`. The rho objective uses `T log|I-rho W|`, matching spreg. Its reported RE log likelihood also matches spreg’s convention of omitting `N log(phi)`; the FE extra RSS/variance constant is handled consistently. These equivalences do not eliminate the different bounds, finite searches or finding 05.

**Stationarity algebra.** For `R = u'u`, `D = u'e1`, the code tests the correct score and negative second derivative:

```text
g = nD/R - tr(WA)
c = nC/R - 2nD²/R² + tr((WA)²).
```

The constant threshold is `0.25`, independent of the tolerance prop. Locally, it permits approximately half an **observed profile-likelihood** standard error, not necessarily half the table’s expected-information standard error.

A displaced rho within that allowance can pass by design. Boundary fits skip the test. A wrong shared trace approximation or a stationary local maximum can also pass; this is not an independent check against the full spreg likelihood.

**2SLS refinement.** With centered instruments `H`, regressors `Z`, `G = H'H` and `B = H'Z`, the implemented correction is:

```text
A = B'G^-1 B
d = A^-1 B'G^-1 H'u.
```

That is the correct projected normal-equation refinement, rather than an OLS correction. The deviation-sum subtraction correctly removes the residual mean from `H'u`. `instrumentOrder: 2` is included in refinement; omitting second-order instruments from the later residual-only pass is harmless.

`subtractProduct` has the correct signs: subtract the exact product error and add the subtraction error. The split lag terms and residual-offset corrections have the correct algebra. A float64 probe with signed weights and an island reduced coefficient error from `2.75e-5` to `1.82e-12` after one refinement.

Published RSS, sigma², classical standard errors and AK all consume the refined residuals. Their covariance matrix depends on the design, so reusing it does not use stale coefficients. Sigma² divides by `n`, matching spreg’s default. The robust covariance also receives the refined residual buffer.

**Multiplication helper, within its narrower domain.** The 12-bit limb construction is correct. `middle` and `lowSum` stay below `2^25`; the two 24-bit limbs reconstruct the 48-bit product. The rounded significand is placed correctly for shifts 23, 24 and 25. For correctly rounded, reported normal products, the signed difference fits exactly in float32. Operand-sign XOR and exponent `exponentSum-300` are correct. The zero/subnormal/nonfinite early returns match their explicit exclusions.

Hand/Python-integer checks:

| Operands | Product supplied | WGSL return | True error |
|---|---:|---:|---:|
| `3`, `5` | `15` | `0` | `0` |
| `4097`, `4097` — rounds down | `16785408` | `+1` | `+1` |
| `4097`, `4099` — rounds up | `16793604` | `-1` | `-1` |
| `-4097`, `4097` | `-16785408` | `-1` | `-1` |
| `1.0099999904632568`, `1.9801980257034302` — crosses into binade at 2 | `2` | `-1.29241755076e-8` | same |
| Both `4097*2^-52` — near `2^-80` | `8.27584509337e-25` | `2^-104` | `2^-104` |
| Cutoff pair from finding 08 | `9.30578386337e-25` | `0` | `-4.93038006988e-32` |
| Adjacent-product pair from finding 08 | `4-3*2^-22` | `2^-22` | `2^-22+2^-46` |

**OLS and GM paths examined.** OLS uses `n-k-1` for inferential variance and `RSS/n` for Gaussian likelihood. Its residual centering and mean-remainder offset have the correct signs. GM’s centered filtering includes the filtered constant, so the centering algebra works with unequal row sums. Its homoskedastic one-iteration path intentionally publishes the first FGLS betas while evaluating their final filtered variance at the updated lambda, matching the stated spreg procedure; the separately solved final betas are not accidentally substituted.

**Could not check**

- Actual GPU statuses and numerical outputs for any new input above; findings 05 and 06 particularly depend on compiled behaviour or rounding.
- Whether Metal reassociates the panel expressions identified in finding 05, or rescales the overflowing stationarity expression during optimization.
- Native WGSL `ldexp`, multiplication accuracy, flushing, and the helper’s boundary behaviour on the test GPU.
- GPU refinement accuracy with several large-mean predictors, second-order instruments, robust covariance, chunked weights, stochastic traces, tail extrapolation and exact-determinant mode.
- Whether float32 candidate cleanup finds additional separating directions in many-predictor or nearly collinear designs. It is a heuristic search; the finite collection of passing indicator scenes establishes no completeness guarantee.
- The proposed mutation and regression specs, existing GPU suites, and cross-adapter behaviour.