# Proper scoring rules

Scores of probabilistic forecasts against observed values: the continuous ranked probability score
(CRPS), the energy score and the variogram score, for ensembles such as conditional Gaussian
simulations, and Gaussian CRPS, negative log, threshold Brier and central interval scores for
kriging predictions. Lower is better for all of them. Each is proper for the predictive
distribution, event probability or interval quantiles that it evaluates.

```ts
import {GPUEnsembleScores, GPUGaussianPredictiveScores} from '@luma-contributors/analysis/gpu-spatial-analysis';
```

The classes are in the directory `src/gpu-spatial-analysis/scoring-rules/`. The re-export line
`export * from './scoring-rules/index';` in `src/gpu-spatial-analysis/index.ts` is what makes the
import above resolve. Gaussian CPU helpers and CPU/GPU operation forms are exported from the
same public entry point.

## Definitions

Definitions are those of scoringRules 1.1.3 (Jordan, Krueger and Lerch, Journal of Statistical
Software 90(12), 2019). With members `x_1 ... x_m` and observation `y`:

| Score | scoringRules function | Formula |
| --- | --- | --- |
| CRPS | `crps_sample(method = "edf")` | `(1/m) sum_i abs(x_i - y) - 1/(2 m^2) sum_i sum_j abs(x_i - x_j)` |
| Energy score | `es_sample` (Gneiting et al. 2008) | the same with Euclidean norms of the `d`-vectors |
| Variogram score of order `p` | `vs_sample(p = ...)` (Scheuerer and Hamill 2015) | `sum_(i,j) w_ij (abs(y_i - y_j)^p - (1/m) sum_k abs(x_ki - x_kj)^p)^2` |
| Gaussian CRPS | `crps_norm` (Gneiting et al. 2005) | `sd (z (2 Phi(z) - 1) + 2 phi(z) - 1/sqrt(pi))`, `z = (y - mean) / sd` |
| Gaussian negative log score | negative normal log density | `log(sd) + log(2 pi)/2 + (y - mean)^2/(2 sd^2)` |
| Threshold Brier score | binary event probability score | `(P(Y > threshold) - 1{y > threshold})^2` |
| Central interval score | proper interval score at noncoverage `alpha` | `(upper - lower) + 2/alpha (lower - y) 1{y < lower} + 2/alpha (y - upper) 1{y > upper}` |

### Fair and non-fair ensemble variants

scoringRules 1.1.3 defines only the non-fair forms above, the empirical-distribution scores whose
member-pair term is divided by `2 m^2`. They are the default (`estimator: 'empirical'`). They are biased
upward for a finite ensemble: even a calibrated ensemble of `m` draws scores worse, by
`E|X - X'| / (2m)`, than its distribution would.

With `estimator: 'fair'` the pair term of the CRPS and the energy score is divided by `2 m (m - 1)`
instead (Ferro 2014; Zamo and Naveau 2018). That removes the bias and lets scores of ensembles with
different member counts be compared. The fair variant is an extension of this package: scoringRules
1.1.3 does not define it, so no scoringRules value exists for it. It needs `memberCount >= 2`. The
variogram score has no fair variant here. The deprecated `fair` boolean remains the old spelling:
`true` means `'fair'`, `false` means `'empirical'`. Supplying both spellings with different meanings
throws at construction (or on entry to the CPU helper); write `estimator` alone. Unknown estimator
strings are rejected with the accepted values. These rules apply to both operation forms too.

## GPUEnsembleScores

Variogram pair count is at most 2,147,483,647 to prevent flattened-index overflow; all score inputs
and outputs remain device-binding bounded. The estimated GPU graph is limited to 32,768 command
nodes.

```ts
import {GPUEnsembleScores} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';

declare const graph: GPUCommandGraph;
declare const ensemble: GraphDataView<'float32'>;
declare const observations: GraphDataView<'float32'>;
declare const variogramPairs: GraphDataView<'uint32x2'>;
declare const variogramWeights: GraphDataView<'float32'>;
declare const crps: GraphDataView<'float32'>;
declare const energyScore: GraphDataView<'float32'>;
declare const variogramScore: GraphDataView<'float32'>;

const scores = new GPUEnsembleScores({
  ensemble,        // GraphDataView<'float32'>, memberCount * dimension rows, member-major
  observations,    // GraphDataView<'float32'>, dimension rows
  memberCount: 100,
  dimension: 4096,
  estimator: 'empirical',
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
- Variogram weights must be finite and nonnegative. Zero is a valid weight. A negative or nonfinite
  weight (including a negative subnormal; signed zero is valid), a nonfinite pair endpoint or a pair index at or beyond `dimension` gives a NaN variogram
  score. Other independently requested scores remain available. Weights are checked during the
  existing pair evaluation, including when they reside on the GPU.
- Nonfinite ensemble members give NaN CRPS/energy scores; NaN observations propagate to the score
  that uses them. Only requested scores are evaluated.

All value inputs use the same caller-chosen value unit `U`. `memberCount` and `dimension` are integer
counts; `dimension` defines the site row domain of observations and CRPS. `variogramPairs` contains
zero-based site indices into observations and each member's site rows, not caller IDs.
`variogramOrder`, weights and `estimator` are dimensionless; `variogramOrder` defaults to 0.5 and
must be finite and positive after float32 rounding on the GPU. `id` is an optional node-name prefix.
CRPS and energy have units `U`; variogram has units `U^(2 * variogramOrder)` for dimensionless
weights. Only the first `dimension` CRPS rows are written; any extra allocated tail is unspecified.
Scalar score outputs must have exactly one row. Numbers and the output map are captured at
construction while GPU resource identities are preserved.

### Operation forms and CPU results

`GPUEnsembleScoresOperation` and `CPUEnsembleScoresOperation` take the same settings, graph input
names and output selection, for example `output: {crps: 'siteScore'}`. The accepted output keys are
`crps`, `energyScore` and `variogramScore`; a typo such as `energyScroe` is rejected before declaring
ports. Both twins require variogram output and pair input together, reject weights without pairs,
and reject fair scoring with fewer than two members at construction. Empty output selections also reject at construction. Both twins capture
names and settings through property access, including prototype/non-enumerable getters, so editing a
caller's configuration later does not relabel old results. The contributor's public `props` retains
the caller's estimator choice. Copy a contributor's props with `{...scores.props, estimator: 'fair'}`
to choose the fair estimator. Resolved defaults are stored separately for execution.
GPU lowering checks every port's element format. `operation.getOutput('crps')` on the GPU wrapper
preserves a literal selection and destination name; an unselected port returns `undefined`.
Keys covered only by a string index signature remain possibly absent, even when their value type
is `string`. Declared required keys such as `crps` remain required.

`computeEnsembleScoresOnCPU` takes typed arrays and returns only the selected keys. A literal
`scores` tuple makes those result keys required in TypeScript. A dynamic array retains optional
keys because the selected set is only known at runtime. A tuple element that is a union of score
names guarantees neither score; a union of tuples requires only keys present in every alternative. Without `scores`, CRPS and energy are
computed, plus variogram when pairs are supplied. Requesting variogram without pairs throws.

```ts
import {computeEnsembleScoresOnCPU} from '@luma-contributors/analysis/gpu-spatial-analysis';

const result = computeEnsembleScoresOnCPU({
  ensemble: Float32Array.of(0, 2 ** 53), observations: Float32Array.of(-1)
}, {memberCount: 2, dimension: 1, estimator: 'fair', scores: ['crps']});
console.log(result.crps[0]); // 1; energyScore is absent because it was not selected.

const invalidWeight = computeEnsembleScoresOnCPU({
  ensemble: Float32Array.of(0, 1), observations: Float32Array.of(0, 0),
  variogramPairs: Uint32Array.of(0, 1), variogramWeights: Float32Array.of(-1)
}, {memberCount: 1, dimension: 2, scores: ['energyScore', 'variogramScore']});
console.log(invalidWeight.variogramScore[0], invalidWeight.energyScore[0]); // NaN, 1
```

The CPU helper uses binary64 arithmetic with one float32 rounding at output. It drops the GPU's
member/workgroup limits and evaluates the numeric variogram order in binary64; GPU order is rounded
to float32. For well-conditioned scores, relative tolerance `3e-5` is appropriate between backends;
multidimensional energy and empirical CRPS can still need an absolute error bound near cancellation.

### How it is computed

The CRPS runs one 64-lane workgroup per site. Lane `l` adds the terms of members `l`, `l + 64`, ...
in order, then a fixed binary tree adds the lanes, so the f32 sum does not depend on scheduling.
The fair CRPS integrates nonnegative sorted intervals: each member finds its rank and successor,
with ties ordered by member index. One-dimensional fair energy uses the same interval identity.
Multidimensional energy and variogram scores write one term per member or per pair and the published
`GPUReduction` sums them. There are no float atomics. The result is repeatable on one device; the
published reduction may use subgroup collectives where the device offers them, and their order is
the device's.

Cost: `O(m^2 d)` for the CRPS and energy score, `O(P m)` for the variogram score over `P` pairs.
`memberCount` is at most 65,535, `dimension` at most 67,108,863 (`floor((2^32 - 1) / 64)`), and
`memberCount * dimension` at most 4,294,967,294. The ensemble must fit one storage binding.
Tile larger fields by calling the class on subsets of sites
for the CRPS; the energy and variogram scores are scores of the whole field.

Accuracy: f32. The CRPS and variogram scores agree with a binary64 evaluation to about
`2e-5` relative on the test fixtures. Empirical CRPS is a difference of two means of similar size
when the ensemble is wide relative to the error, so its absolute error scales with the spread.
Fair CRPS and scalar fair energy use nonnegative intervals to avoid that cancellation: with two
members they equal the observation's distance from the interval between the members. Variogram
pair endpoints are decoded into bounded mantissas and integer exponents before subtraction.
This preserves subnormal gaps without unconditionally halving them, including on devices that flush
subnormal arithmetic. Order 1 uses a common power-of-two frame to retain small exact differences
between large gaps; order 0.5 rationalizes close square roots. Other orders use a frame based on the
largest pair magnitude. Bounded member discrepancies are summed before dividing once by member count.
Only the final weighted squared discrepancy is restored to value units, on both CPU and GPU. A
perfect finite forecast therefore stays zero even when the individual powers would overflow. Identical finite fields
therefore retain zero even when their endpoint difference exceeds float32 range.

{/* lens:GPUEnsembleScores */}

## GPUGaussianPredictiveScores

```ts
import {GPUGaussianPredictiveScores} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {GraphDataView} from '@luma.gl/graph/gpu-core';

declare const prediction: GraphDataView<'float32'>;
declare const variance: GraphDataView<'float32'>;
declare const observations: GraphDataView<'float32'>;
declare const crps: GraphDataView<'float32'>;
declare const logScore: GraphDataView<'float32'>;
declare const brierScore: GraphDataView<'float32'>;
declare const intervalScore: GraphDataView<'float32'>;

const gaussianScores = new GPUGaussianPredictiveScores({
  means: prediction,      // kriging prediction
  variances: variance,    // kriging variance, squared value units
  observations,
  brierThreshold: 1,      // defines the strict event y > 1; required with brierScore
  intervalAlpha: 0.1,     // central 90% interval; optional, default 0.05
  output: {crps, logScore, brierScore, intervalScore}
});
```

The closed-form CRPS of `N(mean, variance)` per site, for `GPUKriging`, `GPULocalKriging` and
`GPUConditionalGaussianSimulation` `output.moments` (split into means and variances). Use the
leave-one-out prediction and variance of `GPUKriging` with the sample values to score kriging by
cross-validation. A zero variance is a point forecast and scores the absolute error; a negative or
NaN variance scores NaN.

`crps` remains required. An omitted output map or null/missing `output.crps` is rejected at
construction with the contributor ID and `output.crps` in the diagnostic.
The three additional outputs are optional, each with one row per mean:

- `logScore` is the **negative** log density, so lower is better and values can be negative.
  A zero-variance forecast has a point mass and no continuous Gaussian density: this output is
  NaN; use CRPS, Brier or interval scores for point forecasts.
- `brierScore` scores `P(Y > brierThreshold) = Phi((mean - brierThreshold)/sd)`. An observation
  exactly at the threshold is a non-event. At zero variance the forecast probability is 1
  precisely when `mean > brierThreshold`, otherwise 0. The small normal tail is evaluated directly
  with complementary erf; when the event occurs, its survival residual is used directly rather
  than subtracting a probability rounded to one. Scores at six and eight standard deviations
  retain their representable values (about 9.73e-19 and 3.87e-31).
  Beyond ten standard deviations the squared small tail is below float32 range. That branch
  saturates before division, keeping Brier scores finite for extreme standardized thresholds.
- `intervalScore` scores the central Gaussian interval with half-width
  `-Phi^-1(intervalAlpha/2) * sd`. At zero variance its width is zero and the score is
  `2 * abs(y - mean) / intervalAlpha`. The centered calculation avoids cancellation in
  `mean +/- halfWidth` for predictions with a large offset.

Additional scores require finite means, observations and nonnegative finite variances; invalid
rows produce NaN on those ports. Settings are captured at construction. `brierThreshold` must
be finite after float32 rounding. `intervalAlpha` is rounded to float32, defaults to 0.05, and
must lie in `[2^-125, 1)` after rounding; its half-tail must remain a normal float32 value.
This keeps both the tail parameter and penalty coefficient representable. Use a representable
level in that range when a requested level is refused.

Packed and chunked contributors support every score. Chunked ports must share ordered chunk
lengths, and all four scores fit the core limit of eight storage bindings, including chunk
parameters. Omitting the optional outputs retains the original CRPS calculation and graph shape.
Both routes accept at most 4,294,967,295 means in total. Each chunk must fit the device's
storage binding limit and at most 16,777,216 addressable 256-lane workgroups per dispatch.
The chunk plan and parameter upload must fit the 32,768-command-node budget.

The CPU helper `computeGaussianPredictiveScoresOnCPU(input, {scores, brierThreshold,
intervalAlpha})` always returns CRPS; `scores` selects any of `logScore`, `brierScore` and
`intervalScore`. It evaluates in binary64 and rounds outputs once to float32. Gaussian score
options are exposed on the existing contributor; no new Gaussian operation class is exported.

For a common transformation `y' = c*y + d`, `mean' = c*mean + d`, `variance' = c^2*variance`,
with `c > 0` and the threshold transformed likewise, CRPS and interval score multiply by `c`,
log score increases by `log(c)`, and Brier score stays unchanged.

`2 Phi(z) - 1 = erf(z / sqrt(2))` has no WGSL built-in. The kernel sums the all-positive series
`erf(x) = 2/sqrt(pi) exp(-x^2) sum_n 2^n x^(2n+1) / (2n + 1)!!` for `x <= 4` (no cancellation in
f32) and takes 1 beyond, which truncates `erf` by less than `1.6e-8` (`|z|` above 5.7). The
agreement with `math.erf` on the fixture is within `2e-6`.
The leading term is evaluated as `|error| * erf(...)` and only the bounded correction is scaled
by the deviation. For `|error| > 4 sqrt(2) * deviation`, the kernel uses
`|error| - deviation / sqrt(pi)` before dividing, so an overflowing standardized error does not
turn a finite representable score into infinity.

A Gaussian ensemble scored with `GPUEnsembleScores` converges to this score as `m` grows: the test
suite checks a 200-member quantile ensemble of `N(0, 1)` against `crps_norm(1; 0, 1)` within
`2e-3`.

{/* lens:GPUGaussianPredictiveScores */}

## Verification

- `gaussian-proper-scores.oracle.py` beside the Gaussian score specs uses SciPy normal
  `logpdf`, survival probabilities and quantiles to generate independent literals for all four
  ports. Its 12-row fixture includes threshold ties, point forecasts, invalid moments, and a
  translated/rescaled copy. Node specs check the CPU and numerical helpers;
  headless specs check packed/chunked contributors against those same literals. Small-tail
  Brier cases at both signs of six/eight deviations use relative tolerances that reject zero.
  Their GPU relative tolerance is `2e-5`: near eight deviations, one float32 ULP in the
  exponential's argument contributes about `7.63e-6` after squaring the tail. The budget allows
  two exponent ULPs plus argument and polynomial rounding. The binary64 CPU evaluation retains
  its tighter `2e-6` relative check.
  GPU tolerances allow float32 transcendental and interval-penalty rounding.

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
