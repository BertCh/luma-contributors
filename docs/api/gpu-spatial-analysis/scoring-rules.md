# Proper scoring rules

Scores of probabilistic forecasts against observed values: the continuous ranked probability score
(CRPS), the energy score and the variogram score, for ensembles such as conditional Gaussian
simulations, and the closed-form Gaussian CRPS for kriging predictions. Lower is better for all of
them, and each is a proper score: in expectation it is smallest for the true distribution.

```ts
import {GPUEnsembleScores, GPUGaussianPredictiveScores} from '@luma-contributors/analysis/gpu-spatial-analysis';
```

The classes are in the directory `src/gpu-spatial-analysis/scoring-rules/`. The re-export line
`export * from './scoring-rules/index';` in `src/gpu-spatial-analysis/index.ts` is what makes the
import above resolve; until it is added, import from `./scoring-rules/index`.

## Definitions

Definitions are those of scoringRules 1.1.3 (Jordan, Krueger and Lerch, Journal of Statistical
Software 90(12), 2019). With members `x_1 ... x_m` and observation `y`:

| Score | scoringRules function | Formula |
| --- | --- | --- |
| CRPS | `crps_sample(method = "edf")` | `(1/m) sum_i abs(x_i - y) - 1/(2 m^2) sum_i sum_j abs(x_i - x_j)` |
| Energy score | `es_sample` (Gneiting et al. 2008) | the same with Euclidean norms of the `d`-vectors |
| Variogram score of order `p` | `vs_sample(p = ...)` (Scheuerer and Hamill 2015) | `sum_(i,j) w_ij (abs(y_i - y_j)^p - (1/m) sum_k abs(x_ki - x_kj)^p)^2` |
| Gaussian CRPS | `crps_norm` (Gneiting et al. 2005) | `sd (z (2 Phi(z) - 1) + 2 phi(z) - 1/sqrt(pi))`, `z = (y - mean) / sd` |

### Fair and non-fair ensemble variants

scoringRules 1.1.3 defines only the non-fair forms above, the empirical-distribution scores whose
member-pair term is divided by `2 m^2`. They are the default (`fair` false). They are biased
upward for a finite ensemble: even a calibrated ensemble of `m` draws scores worse, by
`E|X - X'| / (2m)`, than its distribution would.

With `fair: true` the pair term of the CRPS and the energy score is divided by `2 m (m - 1)`
instead (Ferro 2014; Zamo and Naveau 2018). That removes the bias and lets scores of ensembles with
different member counts be compared. The fair variant is an extension of this package: scoringRules
1.1.3 does not define it, so no scoringRules value exists for it. It needs `memberCount >= 2`. The
variogram score has no fair variant here.

## GPUEnsembleScores

```ts
const scores = new GPUEnsembleScores({
  ensemble,        // GraphDataView<'float32'>, memberCount * dimension rows, member-major
  observations,    // GraphDataView<'float32'>, dimension rows
  memberCount: 100,
  dimension: 4096,
  fair: false,
  variogramOrder: 0.5,
  variogramPairs,  // GraphDataView<'uint32x2'>: ordered site pairs [i, j]
  variogramWeights, // optional GraphDataView<'float32'>, one weight per pair
  output: {crps, energyScore, variogramScore}
});
graph.add(scores);
```

- `ensemble` is member-major, row `member * dimension + site`. That is the layout of
  `GPUConditionalGaussianSimulation` `output.values`, so a simulation batch is scored in place.
- `output.crps` has one row per site. `output.energyScore` and `output.variogramScore` have one row
  each and score the whole field. At least one output is required.
- `variogramPairs` lists ordered pairs; every listed pair contributes one term. Listing every
  `[i, j]` with `i != j` reproduces `vs_sample` with its default unit weights, which sums the full
  symmetric matrix. A list of neighbouring pairs restricts the score to them: the all-pairs form
  costs `O(d^2 m)`, so for large fields use neighbour pairs, a documented choice that matches the
  `w_vs` weights argument of `vs_sample`.
- Members, observations or pairs that are NaN give NaN scores; a pair index at or beyond `dimension`
  gives a NaN variogram score.

### How it is computed

The CRPS runs one 64-lane workgroup per site. Lane `l` adds the terms of members `l`, `l + 64`, ...
in order, then a fixed binary tree adds the lanes, so the f32 sum does not depend on scheduling.
The energy and variogram scores write one term per member or per pair and the published
`GPUReduction` sums them. There are no float atomics. The result is repeatable on one device; the
published reduction may use subgroup collectives where the device offers them, and their order is
the device's.

Cost: `O(m^2 d)` for the CRPS and energy score, `O(P m)` for the variogram score over `P` pairs.
`memberCount` is at most 65535, `dimension` at most 67108863 (`floor((2^32 - 1) / 64)`), and the
ensemble must fit one storage binding. Tile larger fields by calling the class on subsets of sites
for the CRPS; the energy and variogram scores are scores of the whole field.

Accuracy: f32. The CRPS and variogram scores agree with a binary64 evaluation to about
`2e-5` relative on the test fixtures; the CRPS is a difference of two means of similar size when
the ensemble is wide relative to the error, so its absolute error scales with the spread.

{/* lens:GPUEnsembleScores */}

## GPUGaussianPredictiveScores

```ts
const gaussianScores = new GPUGaussianPredictiveScores({
  means: prediction,      // kriging prediction
  variances: variance,    // kriging variance, squared value units
  observations,
  output: {crps}
});
```

The closed-form CRPS of `N(mean, variance)` per site, for `GPUKriging`, `GPULocalKriging` and
`GPUConditionalGaussianSimulation` `output.moments` (split into means and variances). Use the
leave-one-out prediction and variance of `GPUKriging` with the sample values to score kriging by
cross-validation. A zero variance is a point forecast and scores the absolute error; a negative or
NaN variance scores NaN.

`2 Phi(z) - 1 = erf(z / sqrt(2))` has no WGSL built-in. The kernel sums the all-positive series
`erf(x) = 2/sqrt(pi) exp(-x^2) sum_n 2^n x^(2n+1) / (2n + 1)!!` for `x <= 4` (no cancellation in
f32) and takes 1 beyond, which truncates `erf` by less than `1.6e-8` (`|z|` above 5.7). The
agreement with `math.erf` on the fixture is within `2e-6`.

A Gaussian ensemble scored with `GPUEnsembleScores` converges to this score as `m` grows: the test
suite checks a 200-member quantile ensemble of `N(0, 1)` against `crps_norm(1; 0, 1)` within
`2e-3`.

{/* lens:GPUGaussianPredictiveScores */}

## Verification

- Oracle `test/oracles/scoring-rules.py` writes `fixtures/scoring-rules.json`: binary64
  evaluations of the definitions above, with the CRPS cross-checked against the exact integral of
  `(F_m - 1{y <= x})^2` and the Gaussian CRPS against numerical integration.
- Oracle `test/oracles/scoring-rules-fair.py` (NumPy and SciPy) writes
  `fixtures/scoring-rules-fair.json` for the fair variants, which scoringRules does not define: the
  fair CRPS by order statistics, the fair energy score as a mean over distinct pairs
  (`scipy.spatial.distance.pdist`), and the closed-form CRPS and energy score of the standard normal
  distribution, which the mean of the fair scores over many small ensembles must equal (and the
  non-fair mean must exceed). Spec `gpu-ensemble-scores-fair.spec.ts`.
- **The external scoringRules fixture is pending.** scoringRules is an R package and R was not
  installed where the fixture was made. A sibling script that calls `crps_sample`, `es_sample`,
  `vs_sample` and `crps_norm` on the same inputs should replace the self-computed values.
- Specs in `test/gpu-spatial-analysis/scoring-rules/`: GPU values against the fixture, hand values
  (members 0, 1, 2 with `y = 0` score `5/9`, fair `1/3`), bitwise repeatability, propriety of the
  CRPS on simulated data, kernel limits and prop validation.
