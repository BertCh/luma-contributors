# Statistical formulas, null distributions, and numerical edge cases

## Summary

I found 15 departures across scan statistics, space-time tests, point processes, climate indices, forecast scores, mobility, inequality, density and entropy.  
The Poisson scan can give its ordinary no-cluster answer when the observed data have probability zero under the supplied baseline.  
A shifted, zero-inflated climate model can assign a **lower percentile to a higher observation**. A separate gamma approximation can turn a value at the fitted mean into a negative anomaly.  
Two inference APIs publish plausible numbers when inference is undefined. Mantel returns `r = 0, p = 1` for constant distances. The Ripley envelope returns `p = 1` after zero usable simulations. Both were reproduced through CPU twins.  
The most valuable pre-alpha change is to make the statistic’s domain and p-value construction travel with the result. A caller should be able to distinguish an unavailable test, a doubled-tail test and a maximum-statistic test without reading WGSL.  
I rechecked two candidates after the staging checkout changed during review. Its Mantel tie recount and half-integer extreme-rank envelope now pass the discriminating examples. Neither is filed below.

## Census

`S` denotes `modules/analysis/src/gpu-spatial-analysis`, `D` denotes `modules/analysis/src/dataframe-analysis`, and `R` denotes `modules/analysis/src/raster-analysis/raster-time-series`. **KI** identifies the newer copy under `/Users/robertchristie/Documents/GitHub/luma-contributors-ki`. Other paths refer to the requested checkout.

I enumerated direct TypeScript class declarations in 20 target directories, substituting existing KI files listed in `.typecheck/r19/staging-owned.txt`. The declaration query was `^export class ((?:CPU|GPU)\w+)`, with `Operation$` identifying operation forms. This counted **77 direct GPU classes, 21 GPU operation classes and 21 CPU operation classes**. It is a scope count, not a claim that every kernel was audited.

| Directories | Direct GPU classes | GPU operation classes | CPU operation classes |
|---|---:|---:|---:|
| Space-time tests, space-time statistics, scan statistics, point process | 13 | 4 | 4 |
| Change points, emerging hot spots, scoring rules, multiple testing, geographical detector | 8 | 4 | 4 |
| Segregation, spatial entropy, outlier detection, trajectory mobility, trajectory cleaning, point density | 16 | 4 | 4 |
| Dataframe column classification, composite indicators, group statistics, pair statistics | 21 | 7 | 7 |
| Raster time series | 19 | 2 | 2 |

The narrower formula census identified several conventions worth preserving.

| Measured variant | Count | Names and result |
|---|---:|---|
| Constant series treated as untestable in the three sampled Pearson contributors | 2 | `GPUSpaceTimeCrossCorrelation` and `GPURasterLaggedCorrelation` identify zero spread as untestable. |
| Constant series assigned a finite Pearson coefficient | 1 | `GPUMantelTest`, including `computeMantelTestOnCPU`, assigns zero. |
| Published kernel-window integrals tested on narrow windows | 2 | `getGPUKernelDensityWindowMass` and `getGPUSpaceTimeTemporalWindowMass` both returned negative Gaussian mass. |
| Quantile interpolation modes compared on `[1,2,3,8]`, `p=.5` | 5 | `lower`, `higher`, `nearest`, `linear`, `midpoint` returned `2, 3, 3, 2.5, 2.5`, respectively. All matched NumPy, including the nearest-even tie. |
| FDR procedures compared on `[.004,.01,.01,.9]` | 2 | BH gave `[.01333333,.01333333,.01333333,.9]`; BY gave `[.02777778,.02777778,.02777778,1]`. Both matched `scipy.stats.false_discovery_control`. |

I ran the TypeScript host functions and CPU twins through the requested esbuild-to-Node pipeline. Independent references used NumPy 2.0.2 and SciPy 1.13.1. The staged geographical-detector survival function matched `scipy.stats.ncf.sf`, including `F=100, df=(6,50), λ=3` at `1.9230185072884408e-23` versus `1.923018507288415e-23`. This extends the earlier central-F check without reopening `KNOWN(SSR-04)` or the `KNOWN(SSR-124)` range decision. The neighborhood twin also gave the expected population standard deviation `1`, median `2`, and categorical entropy `ln(2)` for two equally weighted neighbors with values `[1,3]` and distinct labels.

For migration counts I used `rg` over construction and `compute…OnCPU(` expressions in `modules`, `examples` and `website`, excluding function declarations. “Production” below counts direct expressions in lenses, Kepler, examples and website source. Numerical changes can require fixture or interpretation changes even when zero call expressions need editing.

## Proposed convention

Each public statistical result should specify its admissible sample, estimator, tail construction, and undefined-result policy. Check each implementation against an independent reference on an ordinary sample and a boundary that distinguishes it from a plausible alternative, such as a constant column, zero usable draws, a point mass, a tied rank, or a change of coordinate origin.

## Findings

| ID | Kind | Reach | Change | Cost | Where | One line |
|---|---|---|---|---|---|---|
| NUM-01 | W | ORDINARY | ADDITIVE | S | Poisson spatial scan | A case at zero exposure can disappear into a no-cluster result. |
| NUM-02 | W | UNUSUAL | BREAKING | S | Raster standardized index | The shifted zero-inflated CDF can decrease across zero. |
| NUM-03 | W | ORDINARY | BREAKING | S | Mantel | Undefined Pearson correlation becomes `r=0, p=1`. |
| NUM-04 | W | UNUSUAL | ADDITIVE | S | Ripley envelope | Zero usable null replicates produce `p=1`. |
| NUM-05 | C/W | ORDINARY | ADDITIVE | M | Permutation planning and metadata | “Two-sided” is insufficient to determine the attainable p-value. |
| NUM-06 | W | UNUSUAL | BREAKING | M | Raster gamma index | A 256-term cap biases the gamma CDF around its mean. |
| NUM-07 | W | UNUSUAL | ADDITIVE | S | Ensemble variogram score | A negative pair weight rewards a forecast error. |
| NUM-08 | C/W | UNUSUAL | BREAKING | M | Mobility indicators | Rotating a two-stop track changes its radius by a factor of 179. |
| NUM-09 | C | ORDINARY | BREAKING | S | Atkinson index | The default returns NaN for a zero income where the limiting index is one. |
| NUM-10 | W | UNUSUAL | BREAKING | S | Gaussian kernel-window mass | Both spatial and temporal host integrals can be negative. |
| NUM-11 | D | ORDINARY | ADDITIVE | S | Pettitt | The page omits that its p-value is an approximation that can reverse a 5% decision. |
| NUM-12 | W | EXTREME | BREAKING | M | Fair ensemble scores | A forecast one unit away can receive exactly the same score as a perfect boundary hit. |
| NUM-13 | W | EXTREME | BREAKING | M | Compact kernel-window mass | Subtracting nearly complete integrals loses a positive support cap. |
| NUM-14 | T/W | ORDINARY | ADDITIVE | S | Spatial entropy | The published error estimate misses a uniformly populated category-pair table. |
| NUM-15 | W | EXTREME | BREAKING | S | Weighted Gini | Subtraction from one erases a representable inequality signal. |

### NUM-01: A case at zero exposure can yield no cluster

**Where.** KI `S/scan-statistics/gpu-spatial-scan-statistic.ts:183,886,1100,1121,1692,1949`. The Poisson model admits finite nonnegative baselines. All cases enter the total, but a window with zero expected cases scores zero.

**Evidence — SOURCE.** Use two distinct zones, one time bucket, `cases=[1,0]`, `baseline=[0,1]`, high-rate direction and the default 50% population cap. The zero-exposure zone fits the cap and its likelihood branch returns `0`. The populated zone exceeds the cap. No positive winner is available. Independently, `scipy.stats.binom.pmf(1,1,0)` returned `0`: under this conditional null, the observed allocation is impossible. The smallest GPU probe is this two-zone frame. Read observed maximum, cluster count and `invalidCount`. The source predicts `0,0,0`.

**Now → wanted.** A caller can read `clusterCount === 0` as evidence of no elevated zone. Detect `cases[cell] > 0 && baseline[cell] === 0` during the totals pass, suppress inference, and publish the number of incompatible cells through `invalidCount`. Keep a genuine zero-case, zero-exposure cell admissible.

**Callers, value and cost.** Zero syntax edits. Six production scan constructions gain a way to distinguish incompatible denominator data from a negative search result. **S.**

### NUM-02: A shifted zero-inflated climate CDF is nonmonotone

**Where.** `R/gpu-raster-standardized-index.ts:215–235` and `R/gpu-raster-standardized-index-calibration.ts:245–251`. The evaluation packer accepted `{activeSliceCount:2, location:-1}` and returned `[2,-1]`.

**Evidence — SOURCE with executed reference.** Use `distribution:'gamma'`, `zeroInflated:true`, `fit=[1,1,.25]`, `location=-1`, values `[-.5,0]`. The source adds the zero atom to the negative observation, then gives exactly `.25` to zero. Evaluating those branches gives CDFs `[.5451020052,.25]`, hence scores `[+.11329587,-.67448975]`. The mixture CDF is  
`0.75 * GammaCDF(x; shape=1, scale=1, loc=-1) + 0.25 * I(x>=0)`.  
SciPy gives `[.2951020052,.7240904191]`, hence `[-.53854042,+.59503637]`. Run one cell with these two slices and inspect both index and status words.

**Now → wanted.** `index[0] > index[1]` can hold for increasing observations. Evaluate the continuous component at both values, then add the atom exactly where `x >= 0`. Compute the survival side directly with the complementary atom. If only nonnegative precipitation is supported for zero inflation, reject incompatible settings explicitly.

**Callers, value and cost.** Two direct production evaluations, in `raster-timeseries.ts` and `standardized-index.ts`. Zero syntax edits for the mixture fix. Users of shifted climate variables regain ordered percentiles. **S.**

### NUM-03: Mantel publishes a correlation for a constant distance vector

**Where.** KI `S/space-time-tests/cpu-mantel-test-operation.ts:207–220` and `gpu-mantel-test.ts:191,520`. Compare `S/space-time-statistics/gpu-space-time-cross-correlation.ts:102,306` and `R/gpu-raster-lagged-correlation.ts:70,220`.

**Evidence — EXECUTED.** Three unordered pairs, spatial distances `[1,1,1]`, times `[0,1,3]`, 19 permutations gave `observed=0`, both exceedance counts `19`, all three p-values `1`, and `zScore=NaN`. `scipy.stats.pearsonr([1,1,1],[1,3,2])` gave NaN with `ConstantInputWarning`. Empty CSR rows also returned `r=0, p=1`. Rechecking the newer KI copy with ten equal distances and 99 permutations gave the same result.

**Now → wanted.** `summary[GPU_SPACE_TIME_SUMMARY.observed] === 0` conflates a measured zero association with an undefined coefficient. Publish NaN for observed correlation and inferential fields when fewer than two usable pairs or either zero norm makes the observed coefficient undefined. Preserve pair count and publish zero usable draws. Define the policy for degenerate null draws separately.

**Callers, value and cost.** Six direct production uses, including `mantel-test.ts` and the space-time cluster examples. Zero syntax edits. The staged CPU spec explicitly pins `p=1` at `cpu-mantel-test-operation.node.spec.ts:207`, and the band/full parity fixtures carry constant-null cases. GIS users can distinguish untestable geometry from a measured absence of association. **S.**

### NUM-04: A Ripley envelope with no usable simulations reports p equal to one

**Where.** `D/pair-statistics/cpu-ripley-envelope-operation.ts:209–217` and KI `D/pair-statistics/gpu-ripley-envelope.ts:645,675–691`.

**Evidence — EXECUTED.** Unit-square bounds, points `(.5,.5),(.51,.5)`, border correction, one radius `.45`, `observedK=[.5]`, two simulations, seed zero. The CPU twin returned:

```text
statistics = [2, 0, 0.0510577075, NaN, 1, 1]
lower = upper = globalLower = globalUpper = [NaN]
```

The second word is the valid simulation count. Neither simulated pattern supplied an estimate at that radius. The observed points do supply one. The pseudo-count then turns an empty comparison into `(0+1)/(0+1)`.

**Now → wanted.** Reading `statistics[4]` alone says the fit is wholly unsurprising. Keep the observed deviation and valid count, and publish NaN for the probability when `validSimulationCount === 0`. The earlier points report identified differing support among usable curves. This case has **no** usable curve.

**Callers, value and cost.** Zero syntax edits. Four production constructions, including the Ripley lens, can display “no usable simulations.” The GPU probe is the same two-point fixture. **S.**

### NUM-05: Two public planners infer different floors from the same tail label

**Where.** `S/ontology/analysis-experience/analysis-interpretation.ts:816,833,1401,1426,1441`, KI `S/multiple-testing/permutation-p-value-floor.ts` in `getPermutationRejectionFloor`, and KI `S/scan-statistics/gpu-spatial-scan-statistic.ts:606,1978`. The doubled-tail space-time summary is in KI `S/space-time-tests/space-time-kernels.ts:264–266`.

**Evidence — EXECUTED and SOURCE.** This is the natural composition of the documented host APIs:

```ts
const metadata = scan.getInferenceMetadata({
  seed: 0, permutationCount: 19, rateDirection: 'both'
});
const plan = getPermutationRejectionFloor({
  alternative: metadata.alternative,
  permutationCount: metadata.permutationCount,
  rowCount: 1,
  alpha: 0.05
});
```

I exercised the metadata method with its two host state fields and passed the result to the staged planner. It gave `'two-sided'`, `pValueFloor=.1`, `canReject=false`. The scan compares **one maximum statistic** against 19 replicate maxima. Zero exceedances give `1/20=.05`. Conversely, `getGPUAnalysisSignificanceAttainability` returned `.05, attainable=true` for 19 draws, including when used to describe a doubled-minimum-tail test whose actual floor is `.1`.

**Now → wanted.** Add `pValueConstruction: 'upper-tail-count' | 'doubled-minimum-tail'` or equivalent metadata. Pass it to both planners and the human-readable resolution helpers. Keep scientific sidedness separate from the counting formula. Pin the two 19-draw examples side by side.

**Callers, value and cost.** Additive. Existing call expressions remain valid. I found 30 calls to `getGPUAnalysisSignificanceAttainability` and nine to `getPermutationRejectionFloor` across source and tests. Users choosing a permutation budget get the same answer as the operation they will run. **M.**

### NUM-06: The raster gamma approximation can label its fitted mean unusually dry

**Where.** `R/raster-climate-distribution-wgsl.ts:9,25–35`, `R/gpu-raster-standardized-index.ts:175,232`, and KI `docs/api/gpu-raster/raster-time-series.md:372`. The shared chi-square evaluator already has a large-shape branch in `modules/analysis/src/analysis-core/kernels/incomplete-gamma-wgsl.ts:78`.

**Evidence — SOURCE with executed numerical replay.** Use one cell, `fit=[65536,1,0]`, gamma distribution, `value=65536`, location zero. SciPy gives `P=.5005194561`, index `+.00130208`. The source’s 256 terms, even evaluated in binary64 with an accurate log-gamma, give only `.34233588`. Explicit float32 replay gave `.3419124`, index `−.40724945`. The omitted mass persists independently of device-specific rounding. This shape corresponds to a coefficient of variation of about 0.39%.

**Now → wanted.** A successful evaluation can publish a negative anomaly at the mean. Use a large-shape approximation with an error bound near `x=a`, or detect nonconvergence and return unavailable. Pin `x=a` and `a±sqrt(a)` against SciPy. The smallest device probe uses one value and the fit above.

**Callers, value and cost.** Two production evaluations, zero syntax edits. Climate users get meaningful scores for low-variability slots. **M.**

### NUM-07: A negative variogram weight rewards forecast error

**Where.** `S/scoring-rules/cpu-ensemble-scores-operation.ts:73–80,161` and `gpu-ensemble-scores.ts:99,417`. The GPU prop describes nonnegative weights. KI `docs/api/gpu-spatial-analysis/scoring-rules.md:60–74` presents them as one weight per pair.

**Evidence — EXECUTED.** One ensemble member `[0,0]`, observations `[0,1]`, pair `[[0,1]]`, order `.5`. The squared discrepancy is one. Weights `1,0,-1` produced scores `1,0,-1`. A perfect prediction scores zero, so a negative input gives a better score to the worse prediction. It lies outside the nonnegative-weight domain of the Scheuerer–Hamill score.

**Now → wanted.** `variogramScore[0] === -1` looks usable. Reject a negative or nonfinite pair weight on the host, or return a NaN score consistently on both backends. Check resident weights as part of the existing GPU pair kernel. Document the domain next to the example. The one-pair fixture is sufficient for GPU confirmation.

**Callers, value and cost.** Zero changes for valid callers. Three production constructions. Forecast evaluators can trust the score ordering. **S.**

### NUM-08: The geographic radius of gyration depends on the longitude seam

**Where.** `S/trajectory-mobility/cpu-mobility-indicators-operation.ts:218,246–256`, `gpu-mobility-indicators.ts:154,461`, and KI `docs/api/gpu-spatial-analysis/movement-semantics.md:209`.

**Evidence — EXECUTED.** Two equally weighted equatorial staypoints at longitudes `[-1,1]` gave radius `111195.078125 m`. Rotating the same track to `[179,-179]` gave **19,903,920 m**. `jumpLengths[1]` was `222390.15625 m` in both cases. For a two-point shortest arc, the geographic midpoint gives radius half that jump, `111195.078125 m`.

**Now → wanted.** The arithmetic-longitude convention is documented and deliberately reproduces the reference implementation. It is a poor geographic default. Use a spherical center for great-circle contexts, retain an explicit arithmetic-coordinate mode for reference parity, and define the result for an indeterminate antipodal center.

**Callers, value and cost.** Three production uses, in `trajectory-mobility.ts`, `catalog-trajectory-cpu.ts` and `trajectory-mobility-indicators.ts`. Only callers needing historical parity need to select the old convention. The Kepler adapter directly publishes this number. **M.**

### NUM-09: Zero income makes the default Atkinson index unavailable

**Where.** `D/composite-indicators/cpu-inequality-operation.ts:123–129`, KI `D/composite-indicators/inequality-wgsl.ts:233–246`, KI `inequality-parallel-statistics.ts:206–212`, and KI `docs/api/gpu-dataframe-analysis.md:1467`.

**Evidence — EXECUTED.** Equal weights, incomes `[0,2]`, positive mean. At `epsilon=1` and `2`, the CPU result was NaN. SciPy gave `1-gmean([0,2])/mean([0,2]) = 1` and `1-hmean([0,2])/mean([0,2]) = 1`. Both are also the limits obtained as the first income decreases to zero. At `epsilon=1-2^-24` the library itself returned `1`.

**Now → wanted.** This is an explicit policy difference. Adopt the continuous extension for a positive-mean population with a positive-weight zero income: `atkinson=1` for `epsilon>=1`. Keep an all-zero population undefined because the normalizing mean is zero. State the distinction in the formula table.

**Callers, value and cost.** Six production constructions, zero syntax edits. At least two explicit assertions change in `gpu-inequality.node.spec.ts:271–272`, alongside the shared oracle and GPU expectation. Income analysts retain the maximum-inequality boundary at the default setting. **S.**

### NUM-10: A Gaussian window can have negative mass

**Where.** `S/shared/complementary-error-function.ts:11–19`, `S/point-density/kernel-density-math.ts:85–95,166`, and `space-time-kernel-density-math.ts:143–168`.

**Evidence — EXECUTED.** With bandwidth one, query zero and time window `[-1e-8,1e-8]`, `getGPUSpaceTimeTemporalWindowMass` returned `−2.2021104273e-8`. SciPy integrated the positive density to `+7.9788456080e-9`. For rectangle `[-1e-8,-1,1e-8,1]`, the spatial helper returned `−1.5033576164e-8` against `+5.4470740560e-9`. Both queries are **inside** their windows. The erfc fit’s offset at zero is larger than the requested mass.

**Now → wanted.** A caller using the host reference gets a negative normalization factor. Use a centered erf or short-interval density integral near zero, direct survival differences on one-sided intervals, and one shared host interval-mass function. Clamping to zero would still lose a representable positive answer.

**Callers, value and cost.** Zero syntax edits. `rg` found two direct spatial-helper calls and one temporal-helper call. Density users gain valid edge factors. For shader parity, evaluate the generated mass functions on the same windows before the contributor’s separate minimum-mass cut. **S.**

### NUM-11: Pettitt’s displayed p-value needs an approximation label

**Where.** KI `S/change-points/cpu-pettitt-test-operation.ts:93–100` and KI `docs/api/gpu-spatial-analysis/change-points-and-expectation-based-scan.md:83–89`.

**Evidence — EXECUTED.** For `[0,0,0,0,1,1,1,1]`, the twin returned `K=16, changeSlice=4, p=.13896690`. Exact conditional enumeration using SciPy midranks found two equally or more extreme orderings among 70, `p=.02857143`. For five zeros followed by five ones the results were `K=25, p=.06614250` versus `2/252=.00793651`. Both reverse a 5% decision. The implementation evaluates the printed exponential formula correctly.

**Now → wanted.** A newcomer can reasonably write `changed = pValues[cell] <= .05` after reading that the statistic uses exact integers and was checked against pyhomogeneity. Add: “`pValues` uses Pettitt’s large-sample approximation. Short or heavily tied series can give a different decision from an exact conditional permutation test.” Include the eight-slice example. An exact host option can be considered independently.

**Callers, value and cost.** Documentation only. Four production constructions and no migration. Users can choose an appropriate inference method before interpreting a short raster series. **S.**

### NUM-12: Fair scoring loses the entire one-dimensional forecast error

**Where.** `S/scoring-rules/cpu-ensemble-scores-operation.ts:95,114,141`. The GPU also subtracts observation and member-pair terms in `gpu-ensemble-scores.ts:313–315,363–365`.

**Evidence — EXECUTED.** `memberCount=2`, `dimension=1`, `fair=true`, members `[0,2^53]`, observation `−1`. Both CPU `crps` and `energyScore` returned `0`. In one dimension, the two-member fair score is exactly the observation’s distance from the interval between the members. Both reference scores are therefore **1**. All inputs and the answer are representable in float32. Replacing `2^53` with `2^24` gave CPU control values `1,1`.

**Now → wanted.** A binary64 subtraction loses the small score before output conversion. Accumulate fair CRPS from nonnegative sorted intervals, and reuse the scalar result for one-dimensional energy. Pin the two-member identity. For GPU confirmation run both magnitudes and read both scores.

**Callers, value and cost.** Three production score constructions, zero syntax edits. The affected inputs have extreme dynamic range, but the exact identity makes a cheap, permanent regression test. **M** for both execution paths.

### NUM-13: Compact-kernel integrals lose a thin cap at the support boundary

**Where.** `S/point-density/kernel-density-math.ts:98–105,130–135,170–174` and `space-time-kernel-density-math.ts:148–168`.

**Evidence — EXECUTED.** Bandwidth one, query `(0,0)`, rectangle `[.999999,-1,1,1]`. The Epanechnikov helper returned `−1.55431223e-15` against a SciPy chord integral of `+9.60336883e-16`. Quartic returned `−2.60902411e-15` against `+1.64629109e-21`. Temporal quartic on `[.999999,1]` returned `0` against `1.24999906e-18`. This is cancellation in the compact polynomial formulas, independently of the Gaussian fit in NUM-10.

**Now → wanted.** The public helpers can be used outside the window to measure its overlap with kernel support. Integrate a cap in terms of its remaining width, `1-s`, using factored endpoint polynomials or a stable chord integral. Pin positivity and reflection symmetry. The corresponding generated mass function on one rectangle or interval is the smallest GPU probe.

**Callers, value and cost.** Same three direct helper calls as NUM-10, zero syntax edits. Benefit is concentrated in standalone overlap calculations near the support limit. **M.**

### NUM-14: Entropy accuracy degrades on an ordinary independent category table

**Where.** KI `S/spatial-entropy/gpu-spatial-entropy.ts:248–266` and KI `docs/api/gpu-spatial-analysis/geographical-detector-redcap-entropy.md:176`. The page gives about `2e-6` agreement with binary64.

**Evidence — SOURCE with executed float32 replay.** Give each of 26 origin categories one link to each of 26 destination categories. The 676 ordered pairs are equiprobable. Exact joint entropy is `ln(676)=6.5161930760` nats, and endpoint mutual information is exactly zero. Replaying the source’s serial float32 addition gave joint entropy `6.5161619186` and mutual information `3.0040741e-5`. With 16 categories, the corresponding spurious mutual information was `9.5367432e-6`. Exact integer pair counts do not protect the subsequent sum.

**Now → wanted.** Pairwise-reduce the entropy terms or use compensated accumulation. Test joint entropy and mutual information against an independent uniform table, and publish an absolute tolerance for information near zero. A direct GPU fixture needs only 32 vertices for the 16-category case: one origin and one destination vertex per category, with all 256 cross-links and no self-slots.

**Callers, value and cost.** No API or call-site changes. Categorical-map users gain an accuracy statement tested across the supported category range. **S.**

### NUM-15: The weighted Gini can discard all inequality before output rounding

**Where.** `D/composite-indicators/cpu-inequality-operation.ts:89–95,121,230,245`.

**Evidence — EXECUTED.** Incomes `[0,1]`, weights `[1,2^58]`, one zone. Both zone and pooled Gini returned `0`. The weighted pair-difference definition gives `1/(2^58+1) = 3.46944695195e-18`, readily representable in float32. As a second control, twelve equal unit-weight incomes returned `1.11022302463e-16` instead of exact zero. The implementation subtracts a rounded Lorenz area from one.

**Now → wanted.** Use the already sorted income gaps: sum each adjacent gap times the weight to its left and the weight to its right, then divide by `totalWeight * totalIncome`. Accumulate suffix weights directly so a small tail is preserved. A constant column then has an exactly zero numerator.

**Callers, value and cost.** Six production inequality constructions, zero syntax edits. The principal gain is for highly unequal survey weights and reproducible equality checks. **S** for the host implementation and oracle.

## Departures by name

- **Undefined support presented as a statistic:** `GPUSpatialScanStatistic`, `GPUMantelTest`, `GPUMantelTestOperation`, `CPUMantelTestOperation`, `computeMantelTestOnCPU`, `GPURipleyEnvelope`, `GPURipleyEnvelopeOperation`, `CPURipleyEnvelopeOperation`, `computeRipleyEnvelopeOnCPU`.
- **Tail construction lost in interpretation:** `getPermutationRejectionFloor`, `getGPUAnalysisSignificanceAttainability`, `getGPUAnalysisSmallestPseudoPValue`, `canResolveGPUAnalysisSignificanceLevel`, `describeGPUAnalysisPermutationResolution`, and `GPUSpatialScanStatistic.getInferenceMetadata`.
- **Distribution, score or index boundaries:** `GPURasterStandardizedIndex`, `GPUEnsembleScores`, `GPUEnsembleScoresOperation`, `CPUEnsembleScoresOperation`, `computeEnsembleScoresOnCPU`, `GPUInequality`, `GPUInequalityOperation`, `CPUInequalityOperation`, `computeInequalityOnCPU`.
- **Geographic center and numerical integration:** `GPUMobilityIndicators`, its GPU and CPU operation forms, `computeMobilityIndicatorsOnCPU`, `getGPUKernelDensityWindowMass`, `getGPUSpaceTimeTemporalWindowMass`, `GPUSpatialEntropy`.
- **Approximation insufficiently identified on the user page:** `GPUPettittTest`, its GPU and CPU operation forms, and `computePettittTestOnCPU`.

## Quick wins

- [ ] Suppress Poisson scan inference on a positive case at zero exposure. Pin the two-zone case.
- [ ] Emit an unavailable Ripley probability when there are zero usable replicate deviations.
- [ ] Reject negative and nonfinite variogram weights during the existing pair calculation.
- [ ] Add the eight-slice exact-versus-approximate Pettitt example to the API page.
- [ ] Add uniform 16- and 26-category entropy fixtures and measure absolute error at zero mutual information.
- [ ] Pin the two-member fair-score interval identity and the two-value weighted-Gini identity as independent CPU tests.

## Larger proposals

1. **Unify inference metadata before alpha.** Add the p-value construction and usable null count, then make the planners and reader-facing summaries consume them. Preserve sidedness as a separate scientific description. This addresses both a false promise of significance and a false claim that significance is unattainable.
2. **Adopt explicit undefined-result policies.** Change degenerate Mantel inference to unavailable and adopt the positive-mean continuous extension for Atkinson. Both current conventions have pinned tests. Updating those tests and the public formula tables together is part of the change.
3. **Choose geographic center semantics explicitly.** Make the spherical center the great-circle default and offer arithmetic-coordinate parity by name. The same two-stop fixture should check both a Greenwich-centered track and its antimeridian rotation.
4. **Qualify climate probabilities across support and shape.** The shifted atom needs a formula correction. The large-shape gamma needs a bounded approximation or an explicit convergence failure. Test monotonicity at zero and accuracy near the mean independently.

## Not read

I read the ensemble CPU score implementation, kernel-window formulas, raster gamma-tail routine, and staged permutation-floor implementation in full. I followed the numerical paths of both envelope twins, Mantel, Pettitt, inequality, mobility, spatial entropy, the Poisson scan and the raster index through their result writes. I sampled the remaining census directories by public class, formula terms, docs, and prior finding IDs. I checked staging ownership and used the newer copy wherever marked KI.

I did not audit regression, likelihood, GWR or autocorrelation, reread every scan sampler, exhaust the tie patterns in corrected emerging-hot-spot inference, or independently certify every segregation, outlier, bootstrap or trajectory statistic. No GPU, browser, network, build or test suite was used. All shader-result predictions are marked `SOURCE`, with a device probe that distinguishes the predicted behavior from the reference.