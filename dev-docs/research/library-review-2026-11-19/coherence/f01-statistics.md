# Spatial statistics, weights, and regression from a PySAL user’s seat

## Summary

I found 69 public GPU contributor classes in the thirteen requested directories. Twenty have a public `compute<Name>OnCPU` twin. I exercised 16 of those twins and used repository test oracles for prominent statistics without twins. Ordinary OLS, spatial 2SLS, both spatial ML fits, spatial lag, local Moran, Gi*, and empirical Bayes agreed with independent calculations on well-conditioned examples.

Three findings deserve attention before the alpha. The weights tutorial teaches a handle-reuse pattern that can **reverse the sign of masked Moran’s I**. Two empirical-Bayes smoothers accept negative event counts and can publish negative disease rates. The global General G formula can lose the entire denominator on a valid four-row count column. The first two have executable host evidence. The General G shader finding still needs a device probe.

The most valuable first-use change is a worked **PySAL `W` and indexed attribute column → validated CSR → transformed weights → named result** example. It should preserve feature order, pass the transformed handle, distinguish observed and tested populations, and show which p-value is being reported. All four details affect the answer.

## Census

`S` below means `modules/analysis/src/gpu-spatial-analysis/`, `T` means `modules/analysis/test/gpu-spatial-analysis/`, and `D` means `docs/api/gpu-spatial-analysis/`. `KI` identifies the newer sibling checkout. I checked the staging ownership list before selecting source and documentation for each finding. Unqualified paths refer to r19.

**COUNT.** `rg --files` found 234 TypeScript files in the thirteen directories. For the public count I bundled the spatial-analysis barrel, enumerated exported class declarations in each directory, intersected their names with the runtime exports, excluded `*Operation` wrappers, and checked for the corresponding public CPU function. This avoids counting the internally declared `GPULocalPermutationReplay` as a public class.

```js
const names = [...source.matchAll(/export class (GPU\w+)\b/g)].map(m => m[1]);
const contributors = names.filter(n => n in publicBarrel && !n.endsWith('Operation'));
const twins = contributors.filter(n => `compute${n.slice(3)}OnCPU` in publicBarrel);
```

| Directory | Public GPU classes | With public CPU twin | Without |
|---|---:|---:|---:|
| spatial-weights | 15 | 3 | 12 |
| spatial-autocorrelation | 11 | 0 | 11 |
| permutation-inference | 2 | 0 | 2 |
| global-spatial-statistics | 1 | 0 | 1 |
| spatial-regression | 23 | 13 | 10 |
| multiple-testing | 2 | 1 | 1 |
| rate-smoothing | 2 | 0 | 2 |
| segregation | 1 | 0 | 1 |
| spatial-entropy | 2 | 0 | 2 |
| geographical-detector | 1 | 0 | 1 |
| distribution-dynamics | 7 | 2 | 5 |
| outlier-detection | 1 | 0 | 1 |
| spatial-interaction | 1 | 1 | 0 |
| **Total** | **69** | **20** | **49** |

### Documented calls and numerical checks

I read the operator pages before tracing their implementations. In the table, `run(C, props)` abbreviates `graph.add(new C(props))`. `output` stands for the caller-owned output bag for that class, `design` for `{predictors, response, predictorCount: 1}`, and `controls` for the packed parameter view. The GPU calls were checked against source declarations. The numbers came from executed CPU functions or explicitly identified test oracles. No GPU result is claimed.

The first fixture was four adjacent unit squares, `x = [1,2,4,8]`, binary rook adjacency, and row-standardized weights `R`. The regression fixture was an eight-row ring with the two neighbors weighted `0.5`, `x = [0,1,4,2,5,7,3,6]`, and `y = [1,2,6,2,4,8,5,9]`. Python references used Shapely boundary intersections, NumPy matrix arithmetic, SciPy likelihood optimization and distribution tails. Neither `esda` nor `libpysal` was installed.

| Call from the page | Executed comparison |
|---|---|
| `run(GPUContiguityWeights, {criterion:'rook', polygons, output})` | Public twin and Shapely gave offsets `[0,1,3,5,6]`, neighbors `[1,0,2,1,3,2]`. |
| `run(GPUSpatialWeightsTransform, {operation:'row', weights})` | Test oracle and `A / A.sum(1)[:,None]` gave `[1,.5,.5,.5,.5,1]` in slot order. |
| `run(GPUSpatialLag, {weights:R, values, output:lag})` | Test oracle and `R @ x` gave `[2,2.5,5,4]`. |
| `run(GPUGlobalSpatialStatistics, {weights:R, values, statistics:['moran','geary','getisOrdG'], output})` | Test oracle and NumPy gave `I=.29130435`, `EI=-1/3`, `C=.38478261`, weighted `G=.42142857`. |
| `run(GPULocalMoran, {weights:R, values, parameters:controls, islands:'keep', output})` | Test oracle and `(n-1) z * (R@z) / (z@z)` gave `Is=[.502174,.228261,.032609,.110870]`, with `z=x−mean(x)`. |
| `run(GPUHotSpotAnalysis, {weights:binary, values, parameters:controls, selfWeight:1, output})` | Test oracle and the Gi* normal-score formula gave `[-1.453631,-1.585258,1.025755,1.453631]`. |
| `run(GPUGlobalPermutationTest, {weights:R, values, statistic:'moran', parameters:controls, maximumPermutations:9, alternative:'greater', output})` | Seed 7 test oracle gave 0 exceedances and `.1`. Python rescored the nine simulated values and obtained `.1`. Exhaustive enumeration of the 24 permutations gave `2/24`. |
| `run(GPULocalPermutationTest, {weights:R, values, statistic:'localMoran', parameters:controls, maximumPermutations:9, alternative:'greater', output})` | Same seed and count gave exceedances `[2,1,3,2]`, p-values `[.3,.2,.4,.3]`, and the four observed local I values above. Python checked the plus-one conversion. I did not independently implement Philox. |
| `run(GPUMultipleTestingAdjustment, {pValues, method:'benjamini-hochberg', output})` | Public twin on `[.01,.04,NaN,.03]` gave `[.03,.04,NaN,.04]` to float32 precision, matching adjustment of the three finite values. |
| `run(GPUEmpiricalBayesRates, {events, populations, output})` | Events `[1,3,7,1]`, populations `[10,10,10,10]`. Clamped test oracle and Python gave `[.2,.3,.5,.2]`. |
| `run(GPUSpatialEmpiricalBayesRates, {events, populations, weights:binary, output})` | Local test oracle and independent neighborhood fits gave `[.2,.339286,.503571,.233333]`. |
| `run(GPUOrdinaryLeastSquares, {...design, output})` | For `x=[0,1,2,3]`, `y=[1,3,4,8]`, public twin and `lstsq` gave `[.7,2.2]`, RSS `1.8`. |
| `run(GPUSpatialTwoStageLeastSquares, {...design, weights:R, output})` | Ring fixture. Public twin gave `[2.228875,1.033100,-.263724]`, matching NumPy 2SLS using instruments `[1,X,WX]`. |
| `run(GPUSpatialLagMaximumLikelihood, {...design, weights:R, logDeterminant:'exact', eigenvalues, output})` | Public twin gave `[1.259116,1.068007,-.080463]`. SciPy optimization with a dense log determinant gave the same estimates. |
| `run(GPUSpatialErrorMaximumLikelihood, {...design, weights:R, logDeterminant:'exact', eigenvalues, output})` | Public twin gave `[.365175,1.217093,.639358]`, matching independent optimization with both `X` and `y` filtered. |

I also executed these **public twins** to extend coverage beyond the operations with public twins in the table:

| Contributor | Input and independent result |
|---|---|
| `GPUSpatialWeightsValidation`, `GPUSpatialWeightsAlgebra` | Valid ring CSR had zero invalid rows. Exact second-order neighbors were `i±2 mod 8`, in ascending row order. Both matched NumPy support checks. |
| `GPUSpatialWeightsTraces`, `GPUSpatialLagImpacts` | Ring powers 1–3 had traces `[0,4,0]`. With slope `2`, rho `.25` and power 3, direct, indirect and total impacts were `[2.0625,.59375,2.65625]`, matching the finite matrix series. |
| `GPUProbitRegression` | Binary response `[0,1,1,0,0,1,0,1]`. Twin `[-.929112,.269505]` matched minimization of the negative log normal CDF. |
| `GPURestrictedRegimeRegression` | First four versus last four rows, separate intercepts and common slope. Twin and restricted design-matrix solve gave `[.825,.725,1.1]`. |
| `GPUSeeminglyUnrelatedRegression` | Two equations with identical predictors. Second response `[3,1,1,4,5,4,2,6]`. Twin gave `[.833333,1.083333,1.833333,.404762]`, matching the two OLS fits as required in this case. |
| `GPURankMarkov`, `GPUTransitionDiagnostics` | Periods `[1,3,2]`, `[2,1,3]` gave zero-based descending ranks `[2,0,1,1,2,0]`. Counts `[[2,1],[2,4]]` gave stationary probabilities `[.5,.5]`. |
| `GPUSpatialInteraction` | Origin-constrained flows `[[1,3],[2,6]]` with a constant covariate within each origin gave `[2,2,4,4]` and Poisson deviance `3.139489`. See UST-07 for the coefficient. |

Two further reference checks passed. The segregation test oracle gave dissimilarity `.6` for `[[8,2],[2,8]]`. The public geographical-detector tail helper returned `.3658138856602241` for `F=3, df=(1,2), λ=1`, against SciPy `.3658138856602239`.

## Proposed convention

A PySAL cross-reference should specify the feature order, weights transform and diagonal policy, tested population, tail, correction, and exact result field wherever those choices affect the number. A result field should retain its meaning when an optional method changes. Check both rules with one asymmetric, masked fixture per linked operation, one tied permutation fixture, and a numerical assertion on the value a reader would actually display.

## Findings

| ID | Kind | Reach | Change | Cost | Where | One line |
|---|---|---|---|---|---|---|
| UST-01 | W | ORDINARY | ADDITIVE | M | Global statistics and permutation | Sparse nonnegative counts can lose General G’s entire denominator. |
| UST-02 | D/W | ORDINARY | ADDITIVE | S | Weights tutorial | Reusing the pre-transform handle can reverse masked Moran’s I. |
| UST-03 | W | UNUSUAL | ADDITIVE | S | Rate smoothing | Negative events enter two Poisson smoothers and contaminate valid neighbors. |
| UST-04 | W/C | ORDINARY | ADDITIVE | S | Multivariate Geary | Reusing `significanceLevel` silently changes a 1% test into a 5% test. |
| UST-05 | W | UNUSUAL | ADDITIVE | M | Global permutation | Bivariate Moran and Lee’s L can lose a finite normalization under a change of units. |
| UST-06 | W | ORDINARY | ADDITIVE | S | OLS result reader | A penalized fit is presented as ordinary least squares with ordinary Student t inference. |
| UST-07 | W | ORDINARY | ADDITIVE | M | Spatial interaction | A completely unidentified gravity coefficient is reported as a successful estimate. |
| UST-08 | W | UNUSUAL | ADDITIVE | S | OLS packer | A positive finite ridge penalty can become zero regularization. |
| UST-09 | W | UNUSUAL | ADDITIVE | M | OLS likelihood | A fixed variance floor makes different residual scales produce identical likelihoods. |
| UST-10 | W/D | ORDINARY | ADDITIVE | S | Segregation | A missing demographic count becomes an observed zero without a result diagnostic. |
| UST-11 | S/C | ORDINARY | ADDITIVE | S | Weights validation | The validator rejects matrices the library constructs or explicitly supports. |
| UST-12 | C | ORDINARY | ADDITIVE | S | Multivariate Geary | One p-value port changes from raw to adjusted, with two Bonferroni denominators. |
| UST-13 | E/D | ORDINARY | ADDITIVE | S | PySAL cross-reference | Import instructions omit the identity check that keeps attributes aligned with `W`. |
| UST-14 | E | ORDINARY | ADDITIVE | S | Permutation planning | The power-planning helper can recommend an impossible GPU run. |
| UST-15 | C | ORDINARY | BREAKING | S | Weights transform | `symmetrize` can return a directed matrix. |
| UST-16 | C | ORDINARY | ADDITIVE | S | Multiple testing | `testedCount` can exceed even the number of input rows. |
| UST-17 | E/S | UNUSUAL | ADDITIVE | S | Regression readers | Six model readers can return `undefined` for a required numerical status code. |
| UST-18 | M | ORDINARY | ADDITIVE | M | Spatial-interaction CPU twin | IPF retains every previous full OD state, including 50,000 states on three cells. |

For numerical repairs, ADDITIVE means supporting a valid input for the existing documented estimator. The affected numerical outputs should be called out in the alpha release notes.

### UST-01 — General G loses the pair mass it needs

**Where.** KI `S/global-spatial-statistics/gpu-global-spatial-statistics.ts:1819` and `S/permutation-inference/gpu-global-permutation-test.ts:966`.

**Evidence.** SOURCE plus EXECUTED float32 emulation. On the binary four-cell path, use `x=[M,1,0,0]`. The only positive off-diagonal products are the two directions of the first edge. Both numerator and denominator are exactly `2M`, so **G must be 1**. Both repository double-precision oracles returned 1 at `M=10,000,000` and `M=2^24`. Both shaders obtain the denominator by subtracting centered sums of order `M²`. NumPy float32 evaluation of the scaled analytic expression gave `G=.7947286` at ten million and a zero denominator at `2^24`. Reduction order may change the first erroneous value.

**Now.** `statistics:['getisOrdG']` on valid, exactly representable counts can lose a result that has no statistical degeneracy. This is distinct from KNOWN(SSR-115), which concerns negative values.

**Wanted.** Accumulate off-diagonal mass directly. A block can maintain `(sum, orderedPairMass)` and add `2*left.sum*right.sum` when merging. Use scaled units and the same primitive in analytic and permutation modes. **GPU probe:** the four-cell fixture above, both values of `M`, both analytic modes and the permutation observed field. Expect 1 in all six readbacks.

**In-repository callers that change:** none. **Value and cost:** preserves General G for sparse tract counts, M.

### UST-02 — The tutorial passes the wrong handle after in-place row standardization

**Where.** KI `D/operators-weights.md:336`, `S/spatial-weights/gpu-contiguity-weights.ts:268`, `S/spatial-weights/gpu-spatial-weights-transform.ts:374`, `S/global-spatial-statistics/gpu-global-spatial-statistics.ts:256`.

**Evidence.** The documented lag example discards the transform object and passes `contiguity.outputWeights` onward. Its shared buffer will contain normalized weights, but its metadata still says `binary`. The lag shown happens to work. A reader reusing that handle for masked Moran gets a different estimator. EXECUTED with the global-statistics test oracle on the four-cell fixture, mask `[1,1,1,0]`. The old handle’s policy gave `I=+.02857143`. Re-standardizing the surviving neighbors gave `I=−.03571429`.

**Wanted.** Use this idiom in the example:

```ts
const transform = new GPUSpatialWeightsTransform({
  operation: 'row',
  weights: contiguity.outputWeights
});
graph.add(transform);
const rowWeights = transform.outputWeights;
graph.add(new GPUSpatialLag({weights: rowWeights, values, output: lag}));
```

Explain that an in-place transform shares storage and publishes a **new description of those bytes**. **GPU probe:** run the masked global statistic once with each handle and compare the two I values above.

**In-repository callers that change:** one documentation example. **Value and cost:** prevents a plausible signed result from the wrong neighborhood definition, S.

### UST-03 — Negative event counts participate in both empirical-Bayes fits

**Where.** `S/rate-smoothing/gpu-empirical-bayes-rates.ts:293`, KI `S/rate-smoothing/gpu-spatial-empirical-bayes-rates.ts:216`. Compare KI `S/spatial-autocorrelation/gpu-rate-standardization.ts:377` and `D/operators-statistics.md:549`.

**Evidence.** EXECUTED test oracles on events `[-10,2,2,2]`, populations `[10,10,10,10]`. The global smoother gave pooled rate `−.1` and smoothed rates `[-1.033333,.211111,.211111,.211111]`. The spatial smoother gave `[-1.066667,.225,.2,.2]`. Both shader predicates admit a finite negative event. `GPUAssuncaoReisStandardization` explicitly excludes it. With that common eligibility rule, the remaining observations all have rate `.2`.

**Wanted.** Require `events >= 0` in both focal and donor eligibility, return NaN for that focus, and count the invalid input wherever an invalid-count output exists. **GPU probe:** the same four rows. Expect `[NaN,.2,.2,.2]` in both smoothed outputs.

**In-repository callers that change:** none for valid counts. **Value and cost:** prevents a negative missing-data sentinel from changing an entire rate map, S.

### UST-04 — The shared significance setting silently falls back to 0.05

**Where.** KI `S/spatial-autocorrelation/multivariate-geary-parameters.ts:16` and `:58`, `S/spatial-autocorrelation/gpu-multivariate-local-geary-permutation-test.ts:84`, `S/permutation-inference/permutation-parameters.ts:24`.

**Evidence.** EXECUTED the public packers. This is an ordinary TypeScript settings variable with a recognized `seed` field:

```ts
const settings = {seed: [7, 0] as const, significanceLevel: 0.01};
getGPUMultivariateLocalGearyPermutationTestParameterValues(settings);
```

The decoded threshold was `.05000000074505806`. Replacing `significanceLevel` with `significance` gave `.009999999776482582`. The local/global permutation packer accepted `significanceLevel` and packed the latter value. Seed `2^32+7` packed as words `[7,1]` there, whereas multivariate Geary takes `[7,1]` directly. Its `maximumPermutations` is also the actual run count.

**Wanted.** Accept `significanceLevel` in the multivariate settings, reject conflicting aliases, and expose a common seed-to-words helper while retaining the full two-word input. Document explicitly that this contributor runs its compile-time count. Publish the effective packed threshold in inference metadata.

**In-repository callers that change:** none with the alias. **Value and cost:** preserves the requested false-positive threshold when changing statistics, S.

### UST-05 — Global bivariate permutation statistics can depend on the measurement unit

**Where.** KI `S/permutation-inference/gpu-global-permutation-test.ts:967`, `:969`, `:1105`. The shared preparation squares centered values at `S/permutation-inference/permutation-inference-kernels.ts`, in the compact-values pass. KI `S/permutation-inference/gpu-local-permutation-test.ts:80` already explains and handles the geometric-mean overflow issue for local statistics.

**Evidence.** SOURCE plus EXECUTED oracle and NumPy checks. Use the four-cell path, `y=x`, first in units `[1,2,4,8]`, then multiply both variables by `2^30`. The double-precision oracle returned observed bivariate Moran `.29130435` and Lee’s L `.21739130` at **both** scales. Each large-scale sum of squares is a finite float32 `3.3146493e19`; their float32 product is infinity. The global kernel forms that product inside `sqrt`, then requires a strictly positive finite scale.

**Wanted.** Compute the normalization with bounded mantissas and exponents, or normalize the two input columns before accumulating permutation pair sums. **GPU probe:** seed 7, nine permutations, identical inputs at both scales. Compare observed statistics, all nine reference values and exceedance counts separately for `bivariateMoran` and `leesL`.

**In-repository callers that change:** none. **Value and cost:** restores scale invariance for valid float32 data, M.

### UST-06 — The OLS reader drops the evidence that a fit was penalized

**Where.** KI `S/spatial-regression/ordinary-least-squares-statistics.ts:315` and `:342`, `S/spatial-regression/cpu-ordinary-least-squares-operation.ts:225`, `D/operators-regression.md:87`. The packed summary already has a ridge-lambda slot.

**Evidence.** EXECUTED `getGPUOrdinaryLeastSquaresModelResult(fit)` on the four-row regression fixture with `ridgeLambda:5`. It reported kind `ordinary-least-squares`, slope `1.1`, SE `.6264982`, Student t `p=.2212084`, status 0. The unpenalized slope is `2.2`. The page describes the penalized inverse-matrix uncertainty convention. That distinction disappears in the named model result. Holding the same estimated error variance fixed, the frequentist ridge sandwich gives SE `.4430011`; the smoothing-matrix trace is `1.5` for this fit.

**Wanted.** Add `regularization: {kind:'ridge', lambda}` and an `inferenceMethod` that explicitly identifies penalized approximate inference. Carry those fields into the displayed coefficient table. Keep the original summary available for callers already using that uncertainty convention.

**In-repository callers that change:** none for additive fields. **Value and cost:** prevents a named report from implying ordinary spreg OLS inference for a different fit, S.

### UST-07 — A gravity coefficient can be completely unidentified and still have status 0

**Where.** KI `S/spatial-interaction/cpu-spatial-interaction-operation.ts:195`, `:360`, `:377`, and `D/coverage-statistics.md:690`.

**Evidence.** EXECUTED with `constraint:'origin'`, flows `[[1,3],[2,6]]`, covariate `[[1,1],[7,7]]`. Grids `[0,1]`, `[1,0]`, and `[-3,-1]` returned fitted coefficients **0, 1, and −3**, respectively. All had status 0, predictions `[2,2,4,4]`, and deviance `3.13948894`. Every coefficient has the same likelihood because each origin’s normalization cancels its constant exponent. This is stronger than missing coefficient uncertainty.

**Wanted.** Add `coefficientIdentified` or `coefficientStatus` independently of the prediction status. Detect a covariate absorbed by the declared margins. A flat profile across sampled grid points alone should be labeled as sampled evidence. Preserve the fitted flows and deviance. **GPU probe:** reverse the two grid values on this four-cell fixture and compare the coefficient status and predictions.

**In-repository callers that change:** none. **Value and cost:** lets transport analysts distinguish a fitted decay coefficient from an arbitrary tie winner, M.

### UST-08 — A finite positive ridge penalty can turn regularization off

**Where.** `S/spatial-regression/ordinary-least-squares-parameters.ts:121`, KI `S/spatial-regression/cpu-ordinary-least-squares-operation.ts:156`.

**Evidence.** EXECUTED. `pack({ridgeLambda:1e30})` produced a fitted slope of `1.1e−29`. Increasing the requested penalty to `1e40` packed infinity, recorded applied lambda **0**, and returned the unpenalized slope **2.2** with status 0. The input to the public packer was positive and finite.

**Wanted.** Validate a positive finite request after float32 conversion and before writing the target. Reject overflow and positive-to-zero underflow with an error naming `ridgeLambda`. The GPU needs no per-row check for values passed through the packer.

**In-repository callers that change:** none for representable penalties. **Value and cost:** prevents a unit conversion from selecting the opposite model, S.

### UST-09 — Small residuals produce a fictitious likelihood plateau

**Where.** KI `S/spatial-regression/cpu-ordinary-least-squares-operation.ts:245` and `S/spatial-regression/gpu-ordinary-least-squares.ts:1267`.

**Evidence.** EXECUTED. For `x=[0,1,2,3]`, multiply `y=[1,3,4,8]` by `1e−20` and `1e−21`. The CPU twin reported log likelihood **168.99733 for both**, although it returned distinct nonzero RSS values. Independent Gaussian likelihoods were **180.12807** and **189.33841**. A zero response also returned 168.99733. The literal floor is applied before taking the logarithm. AIC and BIC inherit it.

**Wanted.** Use the binary64 RSS directly on CPU. On GPU, accumulate residual squares in scaled units and compute the logarithm using the residual scale. Give an exact zero RSS an explicit documented likelihood convention. **GPU probe:** the two four-row fits above should differ in log likelihood by `4 log(10)` within float32 tolerance.

**In-repository callers that change:** none. **Value and cost:** restores the likelihood’s documented response-unit behavior and makes model comparisons interpretable, M.

### UST-10 — Suppressed demographic counts silently become observed zeros

**Where.** KI `S/segregation/gpu-segregation.ts:458`, `D/operators-accessibility.md:318`, and `T/segregation/segregation-oracle.ts:24`.

**Evidence.** EXECUTED the test oracle. Group-0 dissimilarity for `[[8,NaN],[2,8]]` was `.8`, identical to `[[8,0],[2,8]]`. With the missing value observed as 2, dissimilarity was `.6`. SOURCE shows that the shader converts negative, NaN and infinite counts to zero before computing totals. A GIS user cannot distinguish suppression from a reported zero in the result.

**Wanted.** Add an optional `invalidUnitCount` and an explicit missing-count policy, including `exclude-unit`. Explain which units enter the numerator and population totals. **GPU probe:** use the three tables above and read the index and invalid-unit count. The NaN case alone should report one invalid unit.

**In-repository callers that change:** none with the additive policy. **Value and cost:** makes demographic suppression visible before a segregation index is published, S.

### UST-11 — “Validate the weights” assumes a square, diagonal-free neighbor graph

**Where.** KI `S/spatial-weights/cpu-spatial-weights-validation-operation.ts:58`, `S/spatial-weights/gpu-spatial-weights-validation.ts:103`, `S/spatial-weights/gpu-spatial-weights-algebra.ts:313`, `D/operators-weights.md:54`. Compare `S/spatial-weights/gpu-spatial-lag.ts:59` and KI `D/instrumental-variables-covariance.md:64`.

**Evidence.** EXECUTED. I built a two-row identity with the public `selfWeight` algebra twin, then passed its three arrays directly to the validation twin. It returned `flags=[8,8], invalidCount=2`. For a rectangular two-target, three-source matrix, `offsets=[0,1,2], neighbors=[2,0], weights=[1,1]` returned `flags=[2,0]`. Both are legitimate matrix layouts for library consumers. The upload page instructs readers to drop self slots on a violation. Doing that to a unit-diagonal HAC kernel destroys a requirement of its consumer.

**Wanted.** Extend whole, chunked and CPU validation with `columnCount` and `diagonal:'forbid'|'allow'|'require-unit'`. Keep existing defaults. Explain the square-neighbor assumption at the upload example. **GPU probe:** the same identity and rectangular fixtures should pass under their respective declarations and retain current flags under default validation.

**In-repository callers that change:** none. **Value and cost:** makes one validator usable for neighborhood weights, transfer matrices and kernel diagonals, S.

### UST-12 — Multivariate Geary overloads the p-value output and Bonferroni population

**Where.** KI `S/spatial-autocorrelation/gpu-multivariate-local-geary-permutation-test.ts:96`, `:375`, `:589`, `S/spatial-autocorrelation/gpu-multivariate-local-geary-tiled.ts:398`, and `D/operators-statistics.md:463`.

**Evidence.** SOURCE. `GPULocalMoran` and `GPUHotSpotAnalysis` separate `pValues` and `adjustedPValues`. `GPULocalPermutationTest` and `GPULOSHResidualPermutationTest` separate `pseudoPValues` and `adjustedPValues`. In multivariate Geary, `output.pValues` receives the adjusted result when a correction is enabled. Further, `correction:'bonferroni'` counts all complete rows, including islands, whereas `multipleTesting:'bonferroni'` counts finite tests. Both the port meaning and denominator matter to downstream adjustment.

**Wanted.** Offer `output.pseudoPValues` and `output.adjustedPValues` with invariant meanings. Express the two Bonferroni populations as `family:'tested'|'complete'` and document the legacy mapping. **GPU probe:** one included island and four tested rows with a raw p-value below `.2`. Read all three settings. The uncapped adjusted values should be `5p` and `4p`, while the raw port stays `p`.

**In-repository callers that change:** none if `pValues` remains the legacy alias. **Value and cost:** prevents double correction and makes differing denominators reproducible, S.

### UST-13 — A correct CSR can be paired with the wrong tract order

**Where.** `D/README.md:241`, `S/spatial-weights/spatial-weights.ts:93`, KI `D/operators-weights.md:24` and `:739`.

**Evidence.** COUNT: searching the spatial API pages for `id_order`, `idOrder`, or `sort_indices` found zero occurrences in r19. EXECUTED on the same valid four-row weights: `[1,2,4,8]` gave global I `+.29130435`. Interchanging just the middle two attribute rows gave `−.43913043`. Both inputs satisfy the shape and CSR validation rules. This can happen after sorting a GeoDataFrame or joining a separately ordered attribute table.

**Wanted.** Put this at the PySAL entry point, before the first upload:

```python
assert frame.index.is_unique
assert set(frame.index) == set(w.id_order)
values = frame.loc[w.id_order, "value"].to_numpy(dtype="float32")
csr = w.sparse.tocsr(copy=True)
csr.sum_duplicates()
csr.sort_indices()
row_ids = list(w.id_order)  # use the same IDs when joining results back
```

Show the `indptr`, `indices`, `data` mapping, dimensional checks before casting to uint32, and the corresponding transform declaration. Also distinguish `W.islands` (feature IDs) from the summary’s isolate count. To recover IDs from this API, request per-row cardinality and map zero-cardinality rows through `row_ids`, checking completeness first.

**In-repository callers that change:** none. **Value and cost:** prevents a plausible, sign-reversed result from a routine attribute join, S.

### UST-14 — Permutation power planning can prescribe a run the contributors reject

**Where.** KI `S/multiple-testing/permutation-p-value-floor.ts:61`, `S/permutation-inference/permutation-parameters.ts:57`, `S/permutation-inference/permutation-inference-kernels.ts:173` in r19, KI `D/operators-statistics.md:642` and `:768`.

**Evidence.** EXECUTED. For 50,000 rows, two-sided Bonferroni at `.05`, `getPermutationRejectionFloor` returned `canReject:true, pValueFloor:.000001` at `P=1,999,999`. The shared packer accepted that count. The constructor’s shared validation caps both local and global tests at `2^20`. At that actual maximum the helper gave `canReject:false, maximumRowCount:26214`. The formula is correct. Raising the count as the page advises cannot solve this user’s problem on these contributors.

**Wanted.** Export the supported count limit and allow `getPermutationRejectionFloor` to accept a contributor budget. Return both `requiredPermutations` and `attainableWithinBudget`. Include the 50,000-row example. Global planning should also take account of its row-block limit.

**In-repository callers that change:** none. **Value and cost:** tells users before allocating buffers that their intended correction cannot reject, S.

### UST-15 — `symmetrize` preserves a one-way edge

**Where.** `docs/api/gpu-spatial-analysis-cross-reference.md:69`, `D/README.md:248`, KI `D/operators-weights.md:248`, `S/spatial-weights/gpu-spatial-weights-transform.ts:649` and its chunked counterpart `gpu-chunked-spatial-weights-transform.ts`.

**Evidence.** EXECUTED test oracle on `offsets=[0,1,1], neighbors=[1], weights=[1]`. `symmetrize` returned the single weight `[.5]`; the reverse slot still does not exist. The detailed page explains the averaging rule. The cross-reference explicitly maps `W.symmetrize` to it, and the resulting transform name is `symmetric`. A kNN user following that mapping still has a directed graph.

**Wanted.** Rename the operation `averageReciprocalWeights` before alpha. Map `W.symmetrize` to transpose plus support union, with an explicit weight-combination rule. Keep structural symmetry and numeric equality as separate information. **GPU probe:** one directed edge, separate output, assert the old operation retains one slot and the support-union operation has two.

**In-repository callers that change:** `rg` found **3 direct literal call sites in 2 files** and **27 quoted literal occurrences in 15 consumer/test files**, including data-driven operation lists, the lenses pack, explorer, showcase and website controls. Including implementation and scripts gave 45 occurrences in 18 files. **Value and cost:** makes the method name predict the output topology, S.

### UST-16 — `testedCount` reports padded family size

**Where.** `S/multiple-testing/cpu-multiple-testing-adjustment-operation.ts:44`, KI `S/multiple-testing/gpu-multiple-testing-adjustment.ts:444`, `S/multiple-testing/gpu-raster-false-discovery-rate-mask.ts:307`, `D/operators-statistics.md:642`.

**Evidence.** EXECUTED `computeMultipleTestingAdjustmentOnCPU([.02,NaN], 'bonferroni', 10)` using a float32 input. It returned adjusted values `[.19999999,NaN]` and `testedCount=[10]`. There are two supplied rows, one tested hypothesis, and ten hypotheses in the effective family. The adjustment is correct. The output name is what makes “10 tracts tested” an easy reporting error.

**Wanted.** Publish `finiteTestCount=1` and `effectiveFamilySize=10` as independently requestable outputs. Keep `testedCount` as a documented alias for the latter during migration. Apply the same names to the raster wrapper.

**In-repository callers that change:** none with aliases. **Value and cost:** lets a legend report actual tests and the correction denominator independently, S.

### UST-17 — Empty model status buffers become an untyped failure

**Where.** KI `S/spatial-regression/ordinary-least-squares-statistics.ts:347`, `spatial-two-stage-least-squares-model-result.ts:168`, `spatial-maximum-likelihood-model-result.ts:167`, `spatial-error-gm-model-result.ts:103`, and r19 `spatial-panel-error-maximum-likelihood-model-result.ts:38`.

**Evidence.** EXECUTED five direct readers on otherwise sufficiently sized arrays with `status:new Uint32Array()`, and the panel-error reader with eight observations and two entities. All six returned `status.code === undefined`. JSON serialization removed the `code` key. The same 2SLS reader already rejects an empty `robust.status`. KNOWN(SSR-112) addresses the meanings of valid status codes. Here there is no code to decode.

**Wanted.** Validate the primary status array before reading coefficients: `"<model reader> status must hold at least one uint32 code"`. Validate symmetrically for point and covariance status and use the same guard in the shared ML reader.

**In-repository callers that change:** none with valid readbacks. **Value and cost:** turns an incomplete readback into a specific error instead of an apparent model failure, S.

### UST-18 — CPU gravity balancing stores every historical iterate

**Where.** KI `S/spatial-interaction/cpu-spatial-interaction-operation.ts:211` and `:246`.

**Evidence.** EXECUTED a doubly constrained three-cell table: `(0,0)=1, (0,1)=0, (1,1)=1`, constant covariate, coefficient 0. Budgets of 10, 100 and 1,000 gave relative margin errors `.047619`, `.0049751`, `.00049975`. With no explicit budget it took **50,000 iterations** to meet the default tolerance. Every continuing iteration serializes all active log means and inserts the string into `seenStates`. The example finished; the storage growth is visible in source and scales with both iterations and active OD cells.

**Wanted.** Use bounded-memory stagnation or cycle detection. Keep the explicit iteration budget and convergence evidence. Add this boundary-support case to the CPU fixtures and verify that increasing the number of iterations does not retain one full OD vector per iteration.

**In-repository callers that change:** none. **Value and cost:** lets sparse OD tables approach a boundary solution without retaining their entire fitting history, M.

## Departures by name

| Convention being checked | Public departures |
|---|---|
| Count-data eligibility | `GPUEmpiricalBayesRates` and `GPUSpatialEmpiricalBayesRates` accept negative events. `GPUAssuncaoReisStandardization` excludes them. `GPUSegregation` replaces invalid group counts with zero. |
| Raw and adjusted p-values have stable ports | `GPUMultivariateLocalGearyPermutationTest` changes the meaning of `pValues`. `GPULocalMoran`, `GPUHotSpotAnalysis`, `GPULocalPermutationTest` and `GPULOSHResidualPermutationTest` expose separate destinations. |
| Shared local-permutation controls | `GPUMultivariateLocalGearyPermutationTest` uses `significance`, a two-word seed and a compile-time actual count. `GPULocalPermutationTest` and `GPUGlobalPermutationTest` use `significanceLevel`, a safe-integer seed and runtime `permutationCount`. `GPUQuantileLISA` and `GPULOSHResidualPermutationTest` reuse the shared parameter layout, with their own documented tail restrictions. |
| Transform name predicts structural symmetry | `GPUSpatialWeightsTransform`, `GPUChunkedSpatialWeightsTransform`. |
| Validator can describe the matrix its consumer accepts | `GPUSpatialWeightsValidation`, `GPUChunkedSpatialWeightsValidation`, `CPUSpatialWeightsValidationOperation`, `computeSpatialWeightsValidationOnCPU`. |
| Actual tested population is separately readable | `GPUMultipleTestingAdjustment`, `CPUMultipleTestingAdjustmentOperation`, `computeMultipleTestingAdjustmentOnCPU`, `GPURasterFalseDiscoveryRateMask`. |
| Primary model status is validated before decoding | `getGPUOrdinaryLeastSquaresModelResult`, `getGPUSpatialTwoStageLeastSquaresModelResult`, `getGPUSpatialLagMaximumLikelihoodModelResult`, `getGPUSpatialErrorMaximumLikelihoodModelResult`, `getGPUSpatialErrorGMModelResult`, `getGPUSpatialPanelErrorMaximumLikelihoodModelResult`. |

## Quick wins

- [ ] Replace `contiguity.outputWeights` after transformation with `transform.outputWeights` in the lag tutorial.
- [ ] Add the `w.id_order` import and result-join example, including the distinction between island IDs and isolate count.
- [ ] Put a result-field crosswalk beside the PySAL entry table: local `Is → localI`, global `EI → expected` in the Moran block, `z_sim → zSimulated`, and `p_sim → pseudoPValue` **with the tail and tie rule stated**.
- [ ] Accept `significanceLevel` in the multivariate Geary packer and reject conflicting threshold aliases.
- [ ] Expose separate raw and adjusted multivariate Geary outputs and label the population used for Bonferroni.
- [ ] Exclude negative events in both rate smoothers and pin the disease-rate sentinel example.
- [ ] Expose the local/global permutation ceiling in the planning helper and show a family too large for Bonferroni.
- [ ] Preserve applied ridge lambda and its uncertainty convention in the OLS model result.
- [ ] Reject a positive finite ridge request that packs to zero or infinity.
- [ ] Validate the primary status word in all six regression readers before returning a result.

## Larger proposals

**Make weights identity and meaning part of the first successful run.** The cheapest implementation is the worked import plus the corrected transform example. A further additive helper could accept row IDs and aligned attribute IDs at the CPU upload boundary and report the first duplicate, missing ID or order mismatch. No per-frame validation is needed. I recommend shipping the example first and using the same fixtures for the helper.

**Rename the reciprocal-weight average before alpha.** An alias preserves existing examples at low cost, but leaves `symmetrize` discoverable as the first answer to a PySAL search. I recommend the breaking rename and a separately documented structural operation. The migration is bounded by the counts in UST-15.

**Use descriptive result readers for flagship spatial statistics.** Regression already demonstrates the benefit. A host reader over the existing buffers could return `moran.statistic`, `moran.expected`, `moran.pRandomization`, or `permutation.zSimulated`, together with the chosen alternative, correction and counts. Keep packed layouts for graph composition. Start with `GPUGlobalSpatialStatistics` and `GPUGlobalPermutationTest`; the important acceptance test is that a user can extract an observed I and the intended p-value from the page alone.

## Not read

I read the named operator-page sections, PySAL cross-references, permutation and multivariate-Geary parameter packers, rejection-floor helper, and panel-error result adapter. I traced the specific input predicates, transform metadata, summary writes, CPU fits and reader branches cited above. For long WGSL sources I sampled the preparation and finalization paths that determine the reported values. I checked candidates against the first-round synthesis, spatial-statistics and API findings, direction report, and decision queue.

I did not read every regression kernel, every ontology entry, every lens or Kepler adapter, or every test in these directories. Entropy and outlier detection are included in the census, without an independent end-to-end numerical result here. The numerical probes used public CPU twins where available, private test oracles where identified, and independent Python arithmetic. **UST-01 and UST-05 are shader leads awaiting the specified GPU readbacks.** No GPU, build, test runner, network service or file write was used.