---
title: "GPU Spatial Analysis: Distribution dynamics & space-time tests"
sidebar_label: "Distribution dynamics & space-time tests"
description: "Distribution dynamics & space-time tests contributors in @luma-contributors/analysis/gpu-spatial-analysis."
---


# Distribution dynamics & space-time tests


Distribution dynamics contributors use period-major columns: period `t` of row `i` is at
`t * rowCount + i`, with `periodCount` periods.

Contributor constructors and GPU/CPU operation forms snapshot their topology options, output
routes, weight descriptors and method lists. Mutating the caller's options later does not change
the declared computation; construct a new contributor or operation for a different topology.
The contents of the existing input and parameter views remain per-frame data.

See [Space-time statistics](./space-time-statistics.md) for space-time weights, the lagged and differential
local Moran, and cross-correlation by lag. The near-repeat, local Knox and Jacquez tests are listed with
the contributors below; their times follow the [temporal context](./temporal-context.md).

### `GPUClassAssignment`

Assigns each county's 2023 unemployment rate to one of five quantile classes using shared class breaks.

::lens{scenario=gpu-spatial-analysis.class-assignment rung=result caption="437 Midwest counties coloured by quantile class of 2023 unemployment; higher class means higher rate."}

`new GPUClassAssignment({values, breaks, classCount, mask, intervalClosure, output})` assigns each value the number of
inner edges `e[1..k-1]` that are `<=` the value, clamped, the same rule as `GPUClassBreaks.classCounts`.
NaN and masked rows get `GPU_CLASS_ASSIGNMENT_NO_CLASS` (`0xffffffff`).

`intervalClosure` (compile-time, default `'left'`) decides the class of a value exactly equal to an inner edge.
`'left'` treats an edge as the lower edge of the class above it (classes `[e[j], e[j+1])`): the value goes to the
upper class. Use it for `'natural-breaks'` edges, each of which is the smallest value of its class, and for d3 scales.
`'right'` treats an edge as the upper bound of the class below it (classes `(e[j], e[j+1]]`, the minimum in the
first class): the count is of inner edges strictly `<` the value. This is the rule of mapclassify and giddy, so pass
`'right'` with `'quantile'`, `'equal-interval'`, `'standard-deviation'`, `'head-tail'` or `'box-plot'` breaks when the
classes or transition counts must equal theirs. The two differ only for rows equal to an edge, which quantile breaks
produce whenever the data are tied at the quantile. `GPUSpatialMarkov` has the same choice for its lag classes as
`lagIntervalClosure`. Run one `GPUClassBreaks` over the
pooled `n * T` column and assign every period with its breaks to get one legend for a time slider.

Check a stored option object with `satisfies GPUClassAssignmentProps` before construction:
a misspelled key or an `intervalClosure` other than `'left'` or `'right'` is then a type
error. The constructor also rejects both at run time (for example `class-assignment unknown
prop intervalClosue; accepted keys: id, values, breaks, classCount, mask, intervalClosure,
output`), so a typo never falls back to `'left'`. For the value `1` on breaks `[0, 1, 2]`
(one inner edge, `1`), `'left'` gives class `1` and `'right'` gives class `0`:

```ts
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  GPUClassAssignment,
  type GPUClassAssignmentProps
} from '@luma-contributors/analysis/gpu-spatial-analysis';
declare const graph: GPUCommandGraph;
declare const values: GraphDataView<'float32'>; // [1]
declare const breaks: GraphDataView<'float32'>; // [0, 1, 2]
declare const classCount: GraphDataView<'uint32'>; // [2]
declare const leftClasses: GraphDataView<'uint32'>;
declare const rightClasses: GraphDataView<'uint32'>;
const left = {
  id: 'left-closed',
  values,
  breaks,
  classCount,
  intervalClosure: 'left',
  output: leftClasses
} satisfies GPUClassAssignmentProps; // leftClasses = [1]
const right = {
  id: 'right-closed',
  values,
  breaks,
  classCount,
  intervalClosure: 'right',
  output: rightClasses
} satisfies GPUClassAssignmentProps; // rightClasses = [0]
graph.add(new GPUClassAssignment(left));
graph.add(new GPUClassAssignment(right));
```

#### How it computes

{/* lens:GPUClassAssignment */}

::lens{scenario=gpu-spatial-analysis.class-assignment rung=datum datum=15 caption="Cook County, Illinois (row 15 of the county table): its rate and class."}

::lens{scenario=gpu-spatial-analysis.class-assignment rung=formula caption="A county's class is the number of inner breaks at or below its rate."}

::lens{scenario=gpu-spatial-analysis.class-assignment rung=kernel caption="Inspect the recorded shader and its input and output bindings."}

::lens{scenario=gpu-spatial-analysis.class-assignment rung=graph caption="Follow the recorded dependencies from uploaded inputs to captured outputs."}

### `GPUClassificationFit`

Scores how well five quantile classes describe 2023 county unemployment: per-class counts, medians, deviations and GADF.

::lens{scenario=gpu-spatial-analysis.classification-fit rung=result caption="Quantile classes of 437 counties; the table lists counts, medians and fit statistics per class."}

Goodness of fit of a classification (mapclassify 2.11 `adcm`, `get_gadf()`, `get_tss()`): pass `values`, the
`classes` from `GPUClassAssignment`, `classCount` (1 to 256) and an optional `mask`. Outputs are `counts`,
exact per-class `medians` (numpy convention, mean of the two middle values), `absoluteDeviations` and
`squaredDeviations` (one row per class plus a last row for all counted rows) and an optional five-row
`summary` `[GADF, ADCM, ADAM, GVF, TSS]` (`GPU_CLASSIFICATION_FIT_*`). `ADCM` is the sum of absolute
deviations from the class medians, `ADAM` the same from the overall median, `GADF = 1 - ADCM / ADAM` (1 when
`ADAM` is 0), and `TSS` is mapclassify's name for the within-class sum of squares about the class means, not a
total sum of squares. `GVF = 1 - TSS / total sum of squares` is an extension, because mapclassify 2.11 has no
`gvf`. Empty classes add 0, as in mapclassify. Two stable radix sorts (value, then class) leave each class a
sorted segment, so medians are read directly and sums run in sorted order: no atomics, bit-reproducible, no
per-class search passes.

Squared deviations reduce offsets from the already computed median of each class (and of the
overall slot), then center those offsets. This also avoids summing millions of identical large
offsets when almost every value equals the median. The absolute mean is never rounded back onto the
input scale. For classes `[a,a,b,b]`, `[b,b,b,b]` with `a = 1000000` and `b = a + 0.0625`,
the class/overall squared deviations are `[0.00390625, 0, 0.005859375]` and GVF is `1/3`,
just as after subtracting `a` from every value.

Check a stored option object with `satisfies GPUClassAssignmentProps` before construction:
a misspelled key or an `intervalClosure` other than `'left'` or `'right'` is then a type
error. The constructor also rejects both at run time (for example `class-assignment unknown
prop intervalClosue; accepted keys: id, values, breaks, classCount, mask, intervalClosure,
output`), so a typo never falls back to `'left'`. For the value `1` on breaks `[0, 1, 2]`
(one inner edge, `1`), `'left'` gives class `1` and `'right'` gives class `0`:

The spec pins mapclassify 2.11.0 Quantiles, NaturalBreaks and EqualInterval fits (about 1e-4 relative).
Deviations: sums run in float32 and in sorted order rather than index order, non-finite values are skipped, and
at least one row is required.

```ts
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {GPUClassificationFit} from '@luma-contributors/analysis/gpu-spatial-analysis';
declare const graph: GPUCommandGraph;
declare const values: GraphDataView<'float32'>;
declare const classes: GraphDataView<'uint32'>;
declare const counts: GraphDataView<'uint32'>;
declare const medians: GraphDataView<'float32'>;
declare const absoluteDeviations: GraphDataView<'float32'>;
declare const squaredDeviations: GraphDataView<'float32'>;
declare const summary: GraphDataView<'float32'>;
graph.add(new GPUClassificationFit({values, classes, classCount: 5, output: {counts, medians, absoluteDeviations, squaredDeviations, summary}}));
```

#### How it computes

{/* lens:GPUClassificationFit */}

::lens{scenario=gpu-spatial-analysis.classification-fit rung=datum datum=4 caption="Class 4, the highest class of 2023 rates: its county count and median."}

::lens{scenario=gpu-spatial-analysis.classification-fit rung=formula caption="GADF compares absolute deviations around class medians with those around the overall median."}

::lens{scenario=gpu-spatial-analysis.classification-fit rung=kernel caption="Inspect the recorded shader and its input and output bindings."}

::lens{scenario=gpu-spatial-analysis.classification-fit rung=graph caption="Follow the recorded dependencies from uploaded inputs to captured outputs."}

### `GPUTransitionMatrix`

Up to 65,535 classes subject to dense matrix and device binding limits; transition diagnostics
retain 64 classes.

**`GPUTransitionMatrix`**: Counts how many counties move between unemployment classes from one year to the next, with row-normalised probabilities.

::lens{scenario=gpu-spatial-analysis.transition-matrix rung=result caption="Counts of 437 counties moving between five classes over 15 yearly steps, 2008 to 2023."}

**`GPUTransitionDiagnostics`**: Tests whether five states share one transition matrix and solves each chain's steady state, flagging sparse states.

::lens{scenario=gpu-spatial-analysis.transition-diagnostics rung=result caption="Per-state transition counts of Illinois, Indiana, Michigan, Ohio and Wisconsin feed Q, LR, steady states and sparsity."}

Markov transition counts between classes of consecutive periods (giddy `Markov`).

- Props: `classes` (`rowCount * periodCount` rows; the row count is derived from it), `periodCount`,
  `classCount` (1–65,535, subject to dense matrix/device bounds), optional `periodLag` (default 1),
  `conditionClasses` with `conditionCount`, `mask`, and `output: {counts, probabilities?, rowTotals?, ignored?}`.
- `counts` holds exact `uint32` counts at `(c * K + from) * K + to` for `t = 0 .. T - lag - 1`
  (integer atomics, independent of order). `probabilities` are the row-normalized floats, 0 for empty
  rows. `rowTotals` and `ignored` are explain columns.
- Optional diagnostics (also available on their own as `GPUTransitionDiagnostics`, taking the `counts`
  tensor): `output.homogeneity` (needs `conditionCount > 1`; five rows `Q`, `Q_p_value`, `LR`,
  `LR_p_value`, `dof`, indexed by `GPU_TRANSITION_HOMOGENEITY`), `output.steadyStates` (ergodic
  distribution of the pooled chain, then one per condition, `chains * K` rows), `output.stateSparsity`
  (two rows, `GPU_TRANSITION_STATE_SPARSITY`: states with 0 < n < `minimumTransitions`, and states with
  n = 0) and `output.status.converged`. `minimumTransitions` defaults to 5. The steady-state solver keeps
  its elimination matrix in a storage scratch buffer the class allocates (`chains * K^2` floats), so 64
  classes fit the 8,192-byte private memory limit of core WebGPU.
- Homogeneity follows giddy's `homogeneity` (Bickenbach and Bode 2003): Q and LR compare each conditional
  matrix with the pooled one and are chi-square with `sum_i (b_i - 1)(A_i - 1)` degrees of freedom. A
  from-state with no transitions adds 0 degrees of freedom; giddy adds 1 for it, so `dof` differs from
  `dof_hom` by the number of empty pooled rows. The p-values are upper-tail chi-square (`'greater'`) from
  an f32 incomplete gamma (relative error below 6e-5 for p above 1e-6); `dof = 0` gives NaN. The test is
  asymptotic: with sparse rows (see `stateSparsity`) it is unreliable.
  LR accumulates the centered deviance `2 * (O * log(O/E) - O + E)` for every positive
  expected cell, including `O = 0`. This is algebraically the same likelihood ratio, but
  avoids cancellation between signed terms for nearly identical conditional matrices.
  The GPU forms `O-E` from the difference of count products, retaining their multiplication
  rounding errors before division; an expected count rounded onto its observation must not
  erase the evidence. The CPU also uses centered deviance rather than signed log terms.
- Steady states use a direct Grassmann-Taksar-Heyman solve per chain. States with no transition in or out
  are dropped (probability 0, as giddy's `Markov` drops classes absent from the data); the remaining
  states must have an outgoing transition and form one communicating class, otherwise the whole chain is
  NaN (giddy raises or returns several distributions). `converged` is one when every chain that has
  transitions got a steady state. The solver is direct, so `iterationLimitReached` is never written.
  A periodic chain is solved (its stationary distribution, not a limit of powers).
- The connectivity check uses both directions of the transition support, so rejection is independent
  of state labels: counts `[[1, 0], [1, 0]]` and its relabeling both give NaN and `converged = 0`.
  Completely empty chains give NaN without clearing convergence. If float32 stationary weights
  overflow, the GPU also writes NaN and clears convergence; the CPU twin rescales its float64 weights.
  Diagnostic aggregate counts use float32 instead of wrapping at uint32; sparsity counts retain exact
  integer comparisons. The GPU's `minimumTransitions` must be between 1 and 4,294,967,295.
  Both CPU entry points accept integer thresholds through `Number.MAX_SAFE_INTEGER`.
  The GPU operation checks its uint32 threshold limit at construction; the CPU operation
  shares graph-port validation without inheriting that backend limit.
- Condition-axis aggregation and global diagnostic totals use published `GPUSegmentedReduction`
  and `GPUReduction` stages. Per-row kernels visit at most 64 destination classes; the scalar
  publication folds at most 64 class rows after the condition reductions. No serial summary scans
  the complete condition tensor. The graph uses float32 reduction trees, so the last rounding bits
  of Q/LR and pooled stationary probabilities can differ from serial sums; definitions and output
  layouts are unchanged. Summary
  scratch includes one float32 transpose of the count tensor and scalar fields per condition/state;
  every individual buffer must fit the device binding limit. `estimatedCommandNodeCount` is a
  conservative 64-node bound including public reduction stages; no new condition-count cap applies.
  The stationary solver reuses these pooled counts, including when homogeneity and sparsity are
  omitted; it visits only the bounded state system rather than rescanning conditions per matrix cell.

```ts
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {GPUClassAssignment, GPUTransitionMatrix} from '@luma-contributors/analysis/gpu-spatial-analysis';
declare const graph: GPUCommandGraph;
declare const values: GraphDataView<'float32'>;
declare const breaks: GraphDataView<'float32'>;
declare const classCount: GraphDataView<'uint32'>;
declare const classes: GraphDataView<'uint32'>;
declare const counts: GraphDataView<'uint32'>;
declare const probabilities: GraphDataView<'float32'>;
declare const periodCount: number;
// `classCount` is the one-row uint32 view written by GPUClassBreaks.
graph.add(new GPUClassAssignment({values, breaks, classCount, output: classes}));
graph.add(new GPUTransitionMatrix({classes, periodCount, classCount: 5, output: {counts, probabilities}}));
```

#### How it computes

{/* lens:GPUTransitionMatrix */}

::lens{scenario=gpu-spatial-analysis.transition-matrix rung=datum datum=0 caption="Cell 0: counties in class 0, the lowest, that are still in class 0 a year later."}

::lens{scenario=gpu-spatial-analysis.transition-matrix rung=formula caption="Each probability is a transition count divided by the count leaving that class."}

::lens{scenario=gpu-spatial-analysis.transition-matrix rung=kernel caption="Inspect the recorded shader and its input and output bindings."}

::lens{scenario=gpu-spatial-analysis.transition-matrix rung=graph caption="Follow the recorded dependencies from uploaded inputs to captured outputs."}

#### How it computes

{/* lens:GPUTransitionDiagnostics */}

::lens{scenario=gpu-spatial-analysis.transition-diagnostics rung=datum datum=0 caption="Cell 0: Illinois counties in class 0 that are still in class 0 a year later."}

::lens{scenario=gpu-spatial-analysis.transition-diagnostics rung=formula caption="Q sums squared gaps between each state's counts and the pooled expectation."}

::lens{scenario=gpu-spatial-analysis.transition-diagnostics rung=kernel caption="Inspect the recorded shader and its input and output bindings."}

::lens{scenario=gpu-spatial-analysis.transition-diagnostics rung=graph caption="Follow the recorded dependencies from uploaded inputs to captured outputs."}

### `GPUSpatialMarkov` and `GPULISAMarkov`

**`GPUSpatialMarkov`**: Splits class transitions by the quantile class of the queen-neighbour mean rate and tests whether the conditional matrices differ.

::lens{scenario=gpu-spatial-analysis.spatial-markov rung=result caption="Transition counts for 437 counties split by neighbour lag class: 5 lag classes by 5 by 5 cells."}

**`GPULISAMarkov`**: Counts yearly moves of counties between local Moran quadrants HH, LH, LL and HL using queen weights.

::lens{scenario=gpu-spatial-analysis.lisa-markov rung=result caption="Transitions among LISA states, 2008 to 2023; without significance gating state 0 stays empty."}

- `GPUSpatialMarkov` (giddy `Spatial_Markov`): props `values` (`float32`, `rowCount * periodCount`, period-major),
  `classes` (`uint32`, same layout, for example from `GPUClassAssignment`), `classCount` (`K`, 1 to 64),
  `weights` (square self-join `GPUSpatialWeights`, constant across periods), `periodCount` (at least 2),
  optional `periodLag` (default 1), `normalize` (default true), `mask`, `maximumLagClassCount`,
  `lagParameters` (`getGPUClassBreaksParameterValues` view), optional `lagMethods` (default
  `['quantile']`) and an `output` of `counts`, `lagBreaks`, `lagClassCount` and the optional
  `probabilities`, `rowTotals`, `ignored`, `lagValues` and `lagClasses`. For each period it runs `GPUSpatialLag` (row-standardized by default,
  `normalize`), classifies the lag with one pooled `GPUClassBreaks`, assigns the lag classes, and builds a
  transition matrix conditioned on the lag class at the start period. A lag exactly equal to a lag edge goes to
  the upper lag class by default; optional `lagIntervalClosure: 'right'` keeps it in the lower one, as giddy
  does (mapclassify bins are right-closed). `counts` is
  `maximumLagClassCount x K x K`; slices past the produced `lagClassCount` stay zero. The `output` also
  takes the `GPUTransitionMatrix` diagnostics (`homogeneity`, `steadyStates`, `stateSparsity`,
  `status.converged`) and `minimumTransitions`: `homogeneity` is giddy's `Q`, `Q_p_value`, `LR`,
  `LR_p_value`, `dof_hom` across the lag classes; `steadyStates` is giddy's `s` (rows `0..K-1`, pooled) and
  `S` (rows `(1 + c) * K ..`), NaN where a chain is not ergodic. The pooled chain here sums the counted
  transitions over the lag classes, so it excludes rows without a lag class. A small `Q_p_value` says the
  spatial lag matters; a large one says the conditional matrices cannot be told apart from the pooled one.
  `getInferenceMetadata()` describes the tests (analytic, `'greater'`, no correction). Incomplete
  weights (nonzero `weights.status.overflow` or `weights.status.candidateOverflow`) suppress the
  result: every period's `lagValues` and the pooled `lagBreaks` are NaN, lag classes are absent,
  and every transition is ignored. `counts`, `rowTotals`, `lagClassCount` and
  `status.converged` are 0, and `probabilities`, `homogeneity` and `steadyStates` are NaN.
  Supplying a complete neighborhood and clearing the producer's loss flags restores ordinary
  results on the next submission.
  Both spatial and LISA Markov validate all transition/diagnostic output shapes in their
  constructors, before declaring graph stages. Their writable outputs must be disjoint from
  every input, including both neighborhood loss flags. Transition outputs also require
  separate backing buffers even when byte ranges do not overlap; this is checked at construction.
- `GPULISAMarkov` (giddy `LISA_Markov`): props `values`, `weights`, `periodCount` (at least `periodLag + 1`),
  optional `periodLag`, `parameters` (the `GPULocalMoran` view from
  `getGPUSpatialAutocorrelationParameterValues`, which carries the significance level), optional `mask`,
  `gating` and an `output` of `counts` and the optional `probabilities`, `rowTotals`, `ignored`,
  `steadyStates`, `stateSparsity` and `quadrants`. It packs the values into a 64-aligned period stride, so
  every period compiles the same pipelines, runs `GPULocalMoran` per period and counts moves
  between the five states (`GPU_LISA_MARKOV_STATE_COUNT`: 0 not significant, 1 HH, 2 LH, 3 LL, 4 HL). The
  layout is the same for both `gating` modes, so a legend never changes with the option.
  - `gating: 'none'` (default, giddy `LISA_Markov` parity): every valid row is in its quadrant, the signs of
    the centered value and of the lag; state 0 stays empty and the chain is a four-state chain. A row at
    exactly the mean or with an exactly zero lag has no quadrant (esda would call it low) and is ignored.
  - `gating: 'significance'`: rows whose analytic local Moran p-value is above the per-frame significance
    level are state 0. The test is the per-period analytic z at an uncorrected level, not esda
    permutations, and each period is gated on its own, so a transition into or out of state 0 can be a
    threshold crossing rather than a change of the local pattern. Treat it as a map of significant
    clusters, not as a test of dynamics. `getInferenceMetadata(settings)` takes the same
    `GPUSpatialAutocorrelationSettings` packed into `parameters`, describes the test and returns
    `undefined` for `'none'`.
  - Quadrant numbering is esda's default (`geoda_quads=False`): 1 HH, 2 LH, 3 LL, 4 HL. With
    `geoda_quads=True` esda swaps 2 and 3 (1 HH, 2 LL, 3 LH, 4 HL).
  - Missing data: a row-period that is masked, non-finite, or has no neighbor among the finite unmasked
    rows of that period is not a state. It is written as `0xffffffff` (`GPU_CLASS_ASSIGNMENT_NO_CLASS`) in
    `quadrants`, and every transition with such an end is counted in `ignored`, never as state 0. Each
    period excludes its invalid rows from the mean and the neighbor sums (`islands: 'exclude'`).
    giddy has no missing-data handling; parity is on the complete rows. `quadrants` (`rowCount * periodCount`)
    is optional.

Both emit one node chain per period, which suits periods in the tens. giddy `relative` rescaling, mean first
passage times and the permutation test `x2_rpvalue` are not built in: divide each period before
classifying.

#### How it computes

{/* lens:GPUSpatialMarkov */}

::lens{scenario=gpu-spatial-analysis.spatial-markov rung=datum datum=0 caption="Cell 0: lag class 0, class 0 staying in class 0 a year later."}

::lens{scenario=gpu-spatial-analysis.spatial-markov rung=formula caption="Each conditional probability divides a count by the transitions out of its lag class and origin class."}

::lens{scenario=gpu-spatial-analysis.spatial-markov rung=kernel caption="Inspect the recorded shader and its input and output bindings."}

::lens{scenario=gpu-spatial-analysis.spatial-markov rung=graph caption="Follow the recorded dependencies from uploaded inputs to captured outputs."}

#### How it computes

{/* lens:GPULISAMarkov */}

::lens{scenario=gpu-spatial-analysis.lisa-markov rung=datum datum=6 caption="Cell 6: a county in the high-high quadrant that is still high-high a year later."}

::lens{scenario=gpu-spatial-analysis.lisa-markov rung=formula caption="Probabilities divide state-to-state counts by the transitions leaving each quadrant."}

::lens{scenario=gpu-spatial-analysis.lisa-markov rung=kernel caption="Inspect the recorded shader and its input and output bindings."}

::lens{scenario=gpu-spatial-analysis.lisa-markov rung=graph caption="Follow the recorded dependencies from uploaded inputs to captured outputs."}

{/* parameters: GPULISAMarkov generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPULISAMarkov` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPULISAMarkov`.

- Packer: `getGPUSpatialAutocorrelationParameterValues(settings?)`, writes `GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH` (8) `float32` values.
  - Input port: `parameters`.

Settings of `getGPUSpatialAutocorrelationParameterValues`:

| Field | Type | Description |
|---|---|---|
| `significanceLevel` (optional) | `number` | Two-sided significance level of `GPULocalMoran` quadrants, in `(0, 1)`, applied to the adjusted p-value when `multipleTesting` is set. Defaults to `0.05`. `GPUHotSpotAnalysis` ignores it and always reports 90/95/99% bins. |
| `fixedMoments` (optional) | `GPUSpatialAutocorrelationFixedMoments` | Optional moments that replace the moments of the current rows. |

{/* /parameters */}

### `GPUKnoxTest` and `GPUMantelTest`

**`GPUKnoxTest`**: Counts observation pairs within 250 m and 7 days of each other and compares the count with 199 time permutations.

::lens{scenario=gpu-spatial-analysis.knox-test rung=result caption="2,344 observations; the strip compares the observed pair count (row 0) with 199 permuted counts."}

**`GPUMantelTest`**: Correlates pairwise distance with pairwise day difference over 293 sampled observations and compares r with 199 permutations.

::lens{scenario=gpu-spatial-analysis.mantel-test rung=result caption="293 sampled observations (every 8th); row 0 of the strip is the observed correlation r."}

Space-time interaction tests over the pairs of a self-join `GPUSpatialWeights` (typically radius-mode
`GPUNeighborSearch`): every unordered pair listed in either direction is a spatial pair and is counted
exactly once (a symmetric list takes it from the lower row, a pair listed only from the higher row, as in a
kNN list, is taken from that row; duplicate and self entries are ignored; lists need not be sorted). Results
are only as complete as the pair list. Pass `GPUNeighborSearch.outputWeights` so both loss causes
reach inference rather than rebuilding the CSR views without their status.
Pairs carry their own `spatialContext`; Knox needs no context check because it only counts spatial pairs.
Mantel's correlation is unit-free and it takes no spatial distance parameter. Its optional `temporalContext`
is captured and validated as float32-relative ticks; it performs no unit conversion.

Knox, Mantel and Jacquez capture declared input/output maps, including inherited/getter-backed
fields, once at construction. Optional-port declarations and destinations stay fixed; unrelated
output properties are ignored. They read supplied `status.overflow` and `candidateOverflow`
contents on every submission. Either nonzero flag makes the float summary unavailable (all NaN);
float statistics are NaN, integer statistics and an optional permutations-used word are zero.
Clearing the loss flags with complete inputs restores the result on the same graph.

- `GPUKnoxTest({pairs, times, parameters, maximumPermutations, output: {statistics, summary}})` counts the pairs
  that are close in space and have `|t_i - t_j| <= timeThreshold`, then runs a Monte Carlo (modified Knox)
  test over `P` permuted time assignments. Counts are exact integers, and the expected value comes
  from the exact count of time-close pairs. `output.statistics` is `uint32` with `P + 1` entries. Knox's
  hypothesis is one-sided (an excess of close pairs), so the Knox p-value is `pseudoPGreater`.
  `getInferenceMetadata(settings)` takes the same `GPUSpaceTimeSettings` that are packed into
  `parameters` and returns CPU-side `GPUSpaceTimeInferenceMetadata` (alternative `'greater'`, Monte Carlo,
  the clamped `requestedPermutationCount`, stopping rule and seed, no multiple-testing correction).
  `getKnoxPoissonPValue(observed, expected)` gives the classic Poisson
  upper tail, but it assumes Poisson variance, ignores ties and ignores the Mantel-Kulldorff variance
  inflation, so it is anti-conservative; prefer the Monte Carlo p-value.
- `GPUMantelTest({pairs, ...})` (the weights need `distances`) is the Pearson correlation between pair
  distances and `|t_i - t_j|`, with permuted times. By default the raw values are correlated (the
  scikit-bio `mantel` definition). pointpats `mantel` instead correlates `(x + constant) ** power` of both
  and its defaults are the reciprocals `1 / (x + 1)` (`scon = tcon = 1`, `spow = tpow = -1`, pointpats 2.6.0),
  so a default pointpats result differs from the raw one. Pass the compile-time `distanceTransform` and
  `timeTransform` (`{constant, power}`, or `GPU_MANTEL_POINTPATS_TRANSFORM` for the pointpats defaults) to
  match it. The time transform acts on `|t_i - t_j|`, so it cannot be done by transforming `times` first.
  The base of `x + constant` must stay positive for a non-integer power. Sums are f32 shifted partials, within about 1e-4 of a
  double-precision oracle. With compile-time `maximumPermutations = M` and storage binding limit
  `B` bytes, row blocks are at most `min(256, rowCount, floor(B / (4 * (9 * (M + 1) + 2))))`.
  Each (draw, block) stores nine float32 words and each block has two spatial moment words.
  Both routes choose this block count; stopping allocates only one draw batch in the partials table. The statistic runs over the listed pairs only, so with a distance band it is a
  band-truncated Mantel `r` (pairs beyond the band are absent) that differs from the full Mantel `r`;
  report the band with the result. `getInferenceMetadata(settings)` states alternative `'greater'`: the interaction
  hypothesis is a positive space-time correlation, the upper-tail convention of pointpats `mantel`
  (`pseudoPTwoSided` is also in the summary if repulsion matters).

The Mantel operation forms are `GPUMantelTestOperation` and `CPUMantelTestOperation`, sharing
`MantelTestOperationProps`; the standalone CPU twin is `computeMantelTestOnCPU`. Their pair list,
transforms, temporal metadata and output names are captured at construction, while buffer contents
remain live. `GPU_MANTEL_TEST_SUMMARY.observedCorrelation` names the observed coefficient.
The shared `GPU_SPACE_TIME_SUMMARY` indices are unchanged. `decodeGPUMantelTestSummary` accepts float32 words,
`Uint8Array` readback bytes (including subviews) or an `ArrayBuffer` and returns named fields.

With fewer than two usable pairs, or a zero spatial or temporal norm, Mantel preserves `pairCount`
and publishes NaN observed/inferential statistics and zero usable permutations. This applies to
full and band-restricted tests. When the observation is defined, a degenerate null draw remains NaN
in its statistics slot and is excluded from both tails, moments and the denominator; the null
distribution is conditional on a defined correlation. If no usable
null draw remains, the probabilities and moments are NaN. `output.permutationCount` is a one-word
uint32 attempted-prefix length, including NaN null slots; it is optional for fixed-count Mantel
and required for stopping. The additive optional one-word uint32 `output.usablePermutationCount`
reports the defined null correlations used by inference. CPU results expose both arrays. Both
counts are zero for an undefined observation; a defined observation with no usable null draws
retains its positive attempted-prefix count and reports zero usable draws.
Slots beyond the attempted/stopping prefix remain zero; an undefined observation makes all owned
statistics slots NaN. For stopping with degenerate draws, `h / usableDraws` replaces `h / L`.

`getInferenceMetadata(settings, 'pseudoPTwoSided')` describes the summary's doubled minimum tail as
`alternative: 'two-sided', pValueConstruction: 'doubled-minimum-tail'`. Its attainable floor at
19 usable draws is `2 / 20 = .10`. This differs from a scan's maximum-statistic upper-tail
count, whose floor is `1 / 20 = .05`. The default field is `'pseudoPGreater'` with
`pValueConstruction: 'upper-tail-count'`; `'pseudoPLesser'` is also accepted. The numerical
summary formulas are unchanged.

`timeThreshold`, `seed` and `permutationCount` are per-frame settings (`GPUSpaceTimeSettings`), written with
`getGPUSpaceTimeParameterValues({seed, permutationCount, timeThreshold})` (`uint32[4]`). `times` must be
finite and are float32, so raw epoch seconds (about 1.7e9) are spaced 128 s apart and epoch milliseconds
are unusable. Subtract an origin on the CPU with `getGPUSpaceTimeRelativeTimes(times, origin?,
requiredResolution?)`, which subtracts a float64 origin (default the minimum), returns
`{times, origin, resolution}` and throws if the span is too coarse for `requiredResolution`; the float32
spacing at the encoded magnitude `S` is `max(2^-149, 2^(floor(log2 S) - 23))` (about 0.5 ms for an hour, 1 s for 4.6 months of
seconds). Encoding rejects infinity, the guard detects subnormal collapse, and guarded context rebasing
rejects an epoch outside signed Int64, an unknown temporal unit or an empty/non-string timeZone. The default origin still maps `[2²⁴, 2²⁴+1]` to `[0,1]`.
`timeThreshold` is in the same relative units. `output.summary` (`float32`, 12 entries, `GPU_SPACE_TIME_SUMMARY`) holds the observed statistic, pair
count, time-close pair count, expected value, permutation mean and variance, greater and lesser counts,
the pseudo p-values (greater, lesser, two-sided, `(1 + count) / (P + 1)`) and the z-score. The Kulldorff
space-time scan is `GPUSpatialScanStatistic` with `timeBuckets > 1`.

Both tests permute the times with the keyed swap-or-not shuffle of `GPUGlobalPermutationTest`, which is
uniform on small domains (permutation 0 is the identity, permutation `k` is slot `k`). Permuted counts and
p-values for a given seed differ from earlier versions, which used a Feistel bijection that was measurably
non-uniform on small domains. At most `2^24 - 1` events are supported.

**Sequential stopping.** The compile-time `stopAfterExceedances` prop (an integer `h` in `[1,
maximumPermutations]`, default omitted) enables Besag-Clifford sequential Monte Carlo p-values
(Besag and Clifford 1991) for both tests, with the greater alternative that `pseudoPGreater` reports. The
run stops at the first permutation `L` whose running exceedance count (permuted statistic `>=` observed)
equals `h`. Knox uses `pseudoPGreater = min(1, h / L)`; Mantel uses `min(1, h / U)`, where
`U` is the usable count inside prefix `L`. Without a stop, the add-one denominator is `P + 1`
for Knox and `U + 1` for Mantel (NaN when U is zero). Omitted, nothing changes. Permutations run in batches of 64 slots (the batch doubles
when `maximumPermutations` exceeds 16,383, so there are at most 256 batches; batch 0 also computes the
observed statistic). After each batch a one-thread node scans its statistics in permutation order and
disarms the remaining batches through a GPU dispatch gate, so `L` and the p-value do not depend on the batch
size, and the first `L` permutations are those of the full run (same keyed shuffle stream). Work saved is
the batches after the one holding `L`, and the batches past the per-frame `permutationCount`; the stopping
batch finishes whole, so expect savings in steps of one batch, and each batch is a few more graph nodes
(`batch-<n>-partials`, `batch-<n>-reduce`, a parallel Mantel finite-comparison pass, and
`batch-<n>-scan`, between `stop-reset` and `stop-finalize`). `output.permutationCount` (one uint32,
required with stopping) receives attempted prefix `L`, or `P` when the rule did not fire. Without
stopping it is accepted for Mantel and rejected for Knox. Mantel preserves NaN draws in that prefix
and exposes the separate optional `output.usablePermutationCount` for the inference denominator. `output.statistics` rows past `L` are 0, and the summary
permutation mean, variance, greater and lesser counts and z-score cover the first `L` permutations.
`pseudoPLesser` and `pseudoPTwoSided` keep the fixed-length formulas over those `L` permutations and are not
sequentially valid; use `pseudoPGreater`. The `parameters` layout is unchanged, and `h` is baked into the kernels, so a different `h`
is a rebuild.

Settings metadata always reports `requestedPermutationCount` after clamping to
`maximumPermutations`, with the fixed-count or Besag-Clifford stopping rule. For Knox and Jacquez,
`permutationCount` is present only with a fixed budget; stopping omits it. Their decoder remains
`decodeGPUSpaceTimeInferenceMetadata(metadata, attemptedCount)`. Positive attempted counts must
meet the fixed-count or stopping lower bound.

Mantel returns `GPUMantelTestInferenceMetadata` with
`inferenceCountDefinition: 'usable-correlations'`. It omits `permutationCount` even with a fixed
budget because settings cannot know which correlations will be defined. Read both count outputs,
then call `decodeGPUSpaceTimeInferenceMetadata(metadata, attemptedCount, usableCount)`. The final
record preserves attempted `permutationCount` and separate `usablePermutationCount`; use its
nested `inference` metadata for significance-resolution helpers because
`decoded.inference.permutationCount` is the actual usable denominator. For example, requesting
19 draws can produce 19 attempted and 11 usable draws: the doubled-tail floor is `2 / 12`, not
`2 / 20`. A stopped six-attempt prefix with four usable correlations retains count 6, so a reader
can inspect every stopping draw. The same decoder accepts both arrays from `computeMantelTestOnCPU`.

No decoder performs readback. Zero usable draws return `available: false`; an undefined observation
has both counts zero. The Poisson helper returns NaN for nonfinite observed/expected values.
Contributors capture the maximum and stopping rule at construction; later caller mutation cannot
change their kernels or metadata. The standalone CPU twin also validates optional temporal metadata.

#### How it computes

{/* lens:GPUKnoxTest */}

::lens{scenario=gpu-spatial-analysis.knox-test rung=datum datum=0 caption="Row 0: the observed Knox count, before any permutation of days."}

::lens{scenario=gpu-spatial-analysis.knox-test rung=formula caption="The p-value is one plus the permutations at or above the observed count, over P plus one."}

::lens{scenario=gpu-spatial-analysis.knox-test rung=kernel caption="Inspect the recorded shader and its input and output bindings."}

::lens{scenario=gpu-spatial-analysis.knox-test rung=graph caption="Follow the recorded dependencies from uploaded inputs to captured outputs."}

#### How it computes

{/* lens:GPUMantelTest */}

::lens{scenario=gpu-spatial-analysis.mantel-test rung=datum datum=0 caption="Row 0: the observed correlation r, before any permutation of days."}

::lens{scenario=gpu-spatial-analysis.mantel-test rung=formula caption="The correlation runs over all listed pairs; p counts permutations at or above the observed r."}

::lens{scenario=gpu-spatial-analysis.mantel-test rung=kernel caption="Inspect the recorded shader and its input and output bindings."}

::lens{scenario=gpu-spatial-analysis.mantel-test rung=graph caption="Follow the recorded dependencies from uploaded inputs to captured outputs."}

{/* parameters: GPUKnoxTest generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUKnoxTest` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPUKnoxTest`.

- Packer: `getGPUSpaceTimeParameterValues(settings)`, writes `GPU_SPACE_TIME_PARAMETER_LENGTH` (4) `uint32` values.
  - Input port: `parameters`.

Settings of `getGPUSpaceTimeParameterValues`:

| Field | Type | Description |
|---|---|---|
| `seed` | `number` | Random seed, an integer in `[0, 2^53)`. The result is a pure function of the seed and inputs. |
| `permutationCount` | `number` | Number of time permutations `P`, clamped to the contributor's `maximumPermutations`. |
| `timeThreshold` (optional) | `number` | Knox only: two rows are close in time when `\|t_i - t_j\| <= timeThreshold` (f32), in the same units as the relative `times` (see `getGPUSpaceTimeRelativeTimes`). Ignored by `GPUMantelTest`. |

{/* /parameters */}

{/* parameters: GPUMantelTest generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUMantelTest` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPUMantelTest`.

- Packer: `getGPUSpaceTimeParameterValues(settings)`, writes `GPU_SPACE_TIME_PARAMETER_LENGTH` (4) `uint32` values.
  - Input port: `parameters`.

Settings of `getGPUSpaceTimeParameterValues`:

| Field | Type | Description |
|---|---|---|
| `seed` | `number` | Random seed, an integer in `[0, 2^53)`. The result is a pure function of the seed and inputs. |
| `permutationCount` | `number` | Number of time permutations `P`, clamped to the contributor's `maximumPermutations`. |
| `timeThreshold` (optional) | `number` | Knox only: two rows are close in time when `\|t_i - t_j\| <= timeThreshold` (f32), in the same units as the relative `times` (see `getGPUSpaceTimeRelativeTimes`). Ignored by `GPUMantelTest`. |

{/* /parameters */}

### `GPUNearRepeatKnox`, `GPULocalKnox` and `GPUJacquezTest`

Three more space-time interaction tests on the machinery of `GPUKnoxTest`: the same swap-or-not time
permutations (a seed gives the same permutation `k` in all four), the same exact integer counts, and, for
the Jacquez test, the same Besag-Clifford stopping. None has a lens scenario yet.

- `GPUNearRepeatKnox({pairs, times, parameters, distanceEdges, timeEdges, maximumPermutations, output:
  {counts, summary}})` is the near-repeat table of crime science: the number of event pairs in every cell
  of distance band by time-difference band, and a Monte Carlo p-value per cell. `pairs` is a
  `GPUSpatialWeights` **with `distances`** that lists every pair up to the last distance edge (a radius-mode
  `GPUNeighborSearch`), counted once per unordered pair as in `GPUKnoxTest`. Edges are compile-time,
  strictly ascending, at most 16 bands per axis (256 cells). Band `k` is `(edges[k], edges[k + 1]]` and the
  first band also holds `edges[0]`: the `<=` of `GPUKnoxTest` and of pointpats, so a pair on an edge falls
  in the lower band. The NearRepeat R package cuts with R's `cut()`; if its `right` and `include.lowest`
  settings differ, only pairs exactly on an edge move cell (UNVERIFIED against that package, which is GPL
  and used here as a definition only). `distanceEdges` are in the length unit of `pairs.distances`,
  `timeEdges` in the unit of `times`. `output.counts` is `uint32`, `(maximumPermutations + 1) * cells`,
  `counts[slot * cells + distanceBand * timeBands + timeBand]`, slot 0 the observed table. `output.summary`
  holds `GPU_NEAR_REPEAT_CELL_SUMMARY_LENGTH` (8) words per cell, indexed by `GPU_NEAR_REPEAT_CELL_SUMMARY`:
  observed, permutation mean and variance, greater and lesser counts, `pseudoPGreater` and `pseudoPLesser`,
  and the Knox ratio `observed / mean` (NaN when the mean is 0). The NearRepeat ratio against the median is
  not computed. A permutation is one workgroup adding into a workgroup histogram with integer atomics, so
  counts are exact and deterministic, and a table with one cell equals `GPUKnoxTest`. There is no
  Besag-Clifford stopping: cells stop at different permutations. The cells are separate one-sided tests with
  no multiple-testing correction, which `getInferenceMetadata(settings)` states; correct on the CPU.
  If `pairs.status.overflow` or `pairs.status.candidateOverflow` is nonzero, the contributor clears
  its `(maximumPermutations + 1) * cells` count words to zero and its `8 * cells` summary words to
  NaN. Extra caller-owned words are preserved. Complete input on a later submission restores the
  normal tables and summaries.
  The packed pair wrapper, its declared status views and the band edges are captured at construction;
  changing buffer contents affects replay, but deleting or replacing wrapper fields does not change
  the validated bindings. Undeclared fields such as `chunks` do not select chunked semantics.
  `output.counts` and `output.summary` must use separate buffers from all inputs, including
  `pairs.status.overflow` and `pairs.status.candidateOverflow`; use dedicated output buffers when
  those inputs are present. Alias errors name the conflicting public ports.
- `GPULocalKnox({pairs, times, parameters, maximumPermutations, output: {summary}})` is pointpats
  `KnoxLocal`: for every event, the number of its spatial neighbors with `|t_i - t_j| <= timeThreshold`, with
  **conditional** permutation inference as in pointpats: the focal event keeps its own time and the other
  times are permuted (the neighbor mapped onto the focal event takes the time of the event the focal event
  is mapped to). `pairs` must list both directions (a radius-mode search); the count of event `i` runs over
  its own row, ignoring self entries, out-of-range IDs and repeats. `output.summary` has
  `GPU_LOCAL_KNOX_SUMMARY_LENGTH` (6) words per event, indexed by `GPU_LOCAL_KNOX_SUMMARY`: observed,
  permutation mean, greater and lesser counts, `pseudoPGreater` and `pseudoPLesser`. The p-values are one
  test per event with no correction. There is no stopping (events stop at different permutations). The sum
  of the permuted counts per event is a `uint32`: keep `maximumPermutations` times the longest row below
  2^32. The pointpats analytic hypergeometric p-values and the lead/lag orientation are not computed.
- Incomplete pairs: `GPUNearRepeatKnox` and `GPULocalKnox` read `pairs.status.overflow` and
  `pairs.status.candidateOverflow` like `GPUKnoxTest`. Either nonzero means pairs were dropped, so that
  frame's owned `summary` words are all quiet NaN and the near-repeat owned `counts` words are 0,
  as detailed above. Clearing the flags restores ordinary results on the next frame without recompiling.
- `GPUJacquezTest({neighbors, times, parameters, neighborCount, maximumPermutations, stopAfterExceedances?, output:
  {statistics, summary, permutationCount?}})` is the Jacquez (1996) k-nearest-neighbor test, pointpats
  `jacquez`: the number of (event, spatial neighbor) entries where the neighbor is also among the `k` nearest
  events of the event in time, summed over the directed lists (a mutual pair counts twice). `neighbors` is a
  k-nearest-neighbor `GPUSpatialWeights` (`GPUNeighborSearch` in k-nearest mode); use the same `neighborCount`.
  Temporal neighbors come from a stable GPU sort of the times: the `k` nearest in time of an event are the
  interval of `k + 1` ranks around its own rank, so membership under a permutation is one rank comparison.
  Ties in the time distance take the lower rank, which can differ from another library's arbitrary choice
  for tied times; without ties the set is exactly the `k` nearest. The reference distribution permutes the
  times with the spatial lists fixed; the null mean is `sum_i |N_i| k / (n - 1)`. `output.statistics`,
  `output.summary` (`GPU_SPACE_TIME_SUMMARY`, where `pairCount` is the number of listed neighbor entries,
  `timeClosePairs` is `rows * k` and `expected` is `pairCount * k / (rows - 1)`) and Besag-Clifford
  `stopAfterExceedances` work as for `GPUKnoxTest`. The test is known to react to a shift of the population
  over time (Malizia and Mack 2012).

All three accept `temporalContext` where the times are taken: a `float32-relative` `GPUTemporalContext`
with a declared (bigint) epoch, validated at construction and exposed as `.temporalContext`. Nothing is
converted, and `int64-words` or `double-single` is refused. Build the relative times with
`getGPUSpaceTimeGuardedRelativeTimes(times, narrowestTimeWidth, {origin?, temporalContext?})`, which is
`getGPUSpaceTimeRelativeTimes` with its resolution guard set to a quarter of the narrowest time band (or the
time threshold), so it throws rather than blur a band, and which returns the context of the float32 times
(unit kept, epoch moved to the origin; the default origin is the minimum rounded down to a whole tick).

References: pointpats 2.6.0 (BSD-3) `Knox`, `KnoxLocal` and `jacquez` define the statistics and made the
checked-in fixture `modules/analysis/test/oracles/fixtures/space-time-near-repeat.json` (generator
`space-time-near-repeat.py`; the table cells are inclusion-exclusion of `Knox` runs, since pointpats has no
band table). The specs match the fixture's observed counts exactly and compare pointpats' own permutation
draws statistically, because the permutations differ. They also match a double-precision CPU oracle for
every permutation exactly.

{/* lens:GPUNearRepeatKnox */}

{/* lens:GPULocalKnox */}

{/* lens:GPUJacquezTest */}

{/* parameters: GPUJacquezTest generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUJacquezTest` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPUJacquezTest`.

- Packer: `getGPUSpaceTimeParameterValues(settings)`, writes `GPU_SPACE_TIME_PARAMETER_LENGTH` (4) `uint32` values.
  - Input port: `parameters`.

Settings of `getGPUSpaceTimeParameterValues`:

| Field | Type | Description |
|---|---|---|
| `seed` | `number` | Random seed, an integer in `[0, 2^53)`. The result is a pure function of the seed and inputs. |
| `permutationCount` | `number` | Number of time permutations `P`, clamped to the contributor's `maximumPermutations`. |
| `timeThreshold` (optional) | `number` | Knox only: two rows are close in time when `\|t_i - t_j\| <= timeThreshold` (f32), in the same units as the relative `times` (see `getGPUSpaceTimeRelativeTimes`). Ignored by `GPUMantelTest`. |

{/* /parameters */}

{/* parameters: GPULocalKnox generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPULocalKnox` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPULocalKnox`.

- Packer: `getGPUSpaceTimeParameterValues(settings)`, writes `GPU_SPACE_TIME_PARAMETER_LENGTH` (4) `uint32` values.
  - Input port: `parameters`.

Settings of `getGPUSpaceTimeParameterValues`:

| Field | Type | Description |
|---|---|---|
| `seed` | `number` | Random seed, an integer in `[0, 2^53)`. The result is a pure function of the seed and inputs. |
| `permutationCount` | `number` | Number of time permutations `P`, clamped to the contributor's `maximumPermutations`. |
| `timeThreshold` (optional) | `number` | Knox only: two rows are close in time when `\|t_i - t_j\| <= timeThreshold` (f32), in the same units as the relative `times` (see `getGPUSpaceTimeRelativeTimes`). Ignored by `GPUMantelTest`. |

{/* /parameters */}

{/* parameters: GPUNearRepeatKnox generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUNearRepeatKnox` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPUNearRepeatKnox`.

- Packer: `getGPUSpaceTimeParameterValues(settings)`, writes `GPU_SPACE_TIME_PARAMETER_LENGTH` (4) `uint32` values.
  - Input port: `parameters`.

Settings of `getGPUSpaceTimeParameterValues`:

| Field | Type | Description |
|---|---|---|
| `seed` | `number` | Random seed, an integer in `[0, 2^53)`. The result is a pure function of the seed and inputs. |
| `permutationCount` | `number` | Number of time permutations `P`, clamped to the contributor's `maximumPermutations`. |
| `timeThreshold` (optional) | `number` | Knox only: two rows are close in time when `\|t_i - t_j\| <= timeThreshold` (f32), in the same units as the relative `times` (see `getGPUSpaceTimeRelativeTimes`). Ignored by `GPUMantelTest`. |

{/* /parameters */}

### `GPUSpatialScanStatistic`

Finds circular windows of counties with more traffic deaths than their person-years imply, with Monte Carlo p-values.

::lens{scenario=gpu-spatial-analysis.spatial-scan-statistic rung=result caption="437 counties coloured by the best window log-likelihood ratio centred on each; 199 replicates, seed 1."}

Kulldorff's scan statistic (SaTScan, CARTO `DETECT_SPACETIME_ANOMALIES`) over zones, which are cells or
points, with `cases` (`uint32`) per zone and optional time bucket (`zone * timeBuckets + bucket`). The
compile-time `model` selects the null:

- `'poisson'` (default): needs `baseline` (population or expected counts); `E[cases] = baseline * C /
  sum(baseline)`. LLR `c ln(c/e) + (C - c) ln((C - c)/(C - e))`, evaluated per cell as
  `e ((1 + d) ln(1 + d) - d)` with `d = (o - e) / e` (a series for small `d`; the linear parts cancel
  exactly), so windows with `c` close to `e` keep f32 precision.
- `'bernoulli'`: needs `controls` (`uint32`, same layout); the population of a cell is `cases + controls`.
  LLR of Kulldorff (1997), computed as the sum of `o ln(o/e)` over the inside/outside x case/control
  table. Monte Carlo permutes the case labels over the fixed population (conditioning on the total
  cases), by drawing individuals without replacement. Window populations are f32 (exact below 2^24).
- `'space-time-permutation'` (Kulldorff et al. 2005): no baseline or population. The expectation of a
  cylinder is `(sum of its zones' totals over all buckets) * (sum of its buckets' totals over all zones) /
  C`, the Poisson LLR is applied to it, and Monte Carlo permutes the case times over the case locations
  (both margins fixed). A trend or seasonal cycle shared by every zone is therefore not a cluster, which
  the Poisson model with a static population would report as an "emerging cluster". Needs
  `timeBuckets > 1` to be meaningful.

`spatialContext` is optional and defaults to native planar coordinates. Longitude/latitude positions
require metric `'great-circle'`: windows are ranked by great-circle distance using the shared haversine
separation key, rather than distances in degrees. Cluster `radius` uses the context's length unit
(meters by default for great-circle; kilometers with `units: 'kilometers'`), with `sphereRadius` or the
mean Earth radius in that unit. This operator measures distance, so `local-lambert-equal-area` frames throw.

Windows and defaults:

- Spatial: the `k` nearest zones of every center (the `'circle'` shape never splits equidistant zones;
  `'nearest'` is plain k-nearest), kept while they hold at most `maximumPopulationFraction` of the total
  population (default 0.5, as in SaTScan). The population share governs: the per-center neighbor
  tables hold up to the compile-time `maximumWindowZones` zones, which defaults to every zone while
  `zones^2 <= 2^24` (4096 zones) and otherwise to the largest capacity that fits
  (`GPU_SPATIAL_SCAN_STATISTIC_MAXIMUM_NEIGHBOR_WORDS`; use `getGPUSpatialScanStatisticLayout` to see it).
  Neighbor discovery orders by `(separation key, zone row)`, with squared distance for planar and
  the haversine term for great-circle. Short lists use exact coordinate indexes; larger lists use
  exhaustive radix selection and sorting. The route and its work estimate are described below.
  A window the capacity cut short while the population fraction would still allow it is never silent:
  `output.summary` word `overflow` counts the affected centers and the cluster record carries
  `truncatedByZoneCap` (0 or 1). The per-frame `maximumWindowZones` may lower the capacity.
  For circles, capacity loss requires the entire next tied distance group to fit the population cap;
  a partial tie never qualifies. Discovery accumulates that boundary population during its existing
  search, including tied rows beyond the stored list, and checks the whole circle's row count against
  the caller's requested per-frame zone limit. Reaching that explicit limit is not capacity loss.
  Omission leaves the compiled capacity as a storage constraint: population-admissible windows
  beyond it still count as capacity loss. An explicit limit also excludes larger windows from
  that diagnostic, even when the limit equals capacity. With capacity one, the three-zone oracle
  reports overflow `0` for an explicit limit of one and `3` when omitted.
  Positive float32 boundary masses are accumulated exactly with integer words. Eligibility compares
  the exact mass with the exact study mass times the packed float32 fraction, independently of
  discovery order; rounding is deferred to published/scoring masses. A 100% cap admits every subset.
  Space-time permutation uses the integer zone
  case margins and grand case count for the spatial cap, independently of rounded cell expectations.
  Its threshold is `floor(C * packedFloat32Fraction)`: for example, packed `0.7` admits at most
  6,999 of 10,000 cases, not the raw JavaScript threshold of 7,000. The CPU test reference uses the
  same packed fraction and an exact integer comparison.
  Window-count requests above the uint32 range saturate during parameter packing, so the GPU
  still clamps them to capacity rather than interpreting wrapped counts as smaller windows.
  Explicit window limits must be finite and positive; they round down with a minimum of one.
  `maximumPopulationFraction` must remain in `(0,1]` after float32 conversion.
- Temporal: with `timeBuckets > 1` each window is a cylinder over every run of up to
  `maximumTimeBuckets` buckets. The default is 50% of the study period, `max(1, floor(timeBuckets / 2))`
  (SaTScan's default), no longer all buckets: with a global trend, pure-spatial windows over the whole
  period would otherwise dominate.
- Direction: per-frame `rateDirection` `'high'` (default, `c > e`), `'low'` (`c < e`) or `'both'`.
  Windows of the other direction score 0.
- Defaults: `maximumPopulationFraction` 0.5, compile-time `maximumWindowZones` every zone (bounded),
  per-frame zone limit omitted, `maximumTimeBuckets` 50% of the study period, `windowShape` `'circle'`,
  `rateDirection` `'high'`, `maximumClusters` 8.

Outputs: the most likely cluster plus greedy, non-overlapping secondary clusters (no shared zone, time
ignored, same maximum distribution, as SaTScan), all in the caller-owned `output` bag. The candidates
for a secondary cluster are the single best window of each center, as in SaTScan: a center whose best
window overlaps an earlier cluster is dropped, and its smaller windows are not reconsidered.
`output.clusterIndices` (`GPU_SPATIAL_SCAN_STATISTIC_CLUSTER_INDEX`) and `output.clusterStatistics`
(`GPU_SPATIAL_SCAN_STATISTIC_CLUSTER`: LLR, observed, expected, p, radius, ratio,
`truncatedByZoneCap`); `output.statistics` (the observed maximum then the maximum of every Monte Carlo
replicate), `output.summary` (`GPU_SPATIAL_SCAN_STATISTIC_SUMMARY`: cluster count, total cases,
`permutationCount`, `overflow`) and an optional per-zone best LLR `output.zoneStatistics`.

Inference: `p = (1 + #{max_rep >= LLR}) / (P + 1)`, one upper-tail count of replicate maxima
over every searched window. Searching both rate directions uses that same count, with no doubled
tail probability. This already accounts for the search; no further correction applies.
`getInferenceMetadata(settings)` takes the same
`GPUSpatialScanStatisticSettings` that are packed into `parameters` and returns the CPU-side
`GPUInferenceMetadata` (alternative `'greater'` for high rates, `'lesser'` for low and `'two-sided'` for
both, the permutation count clamped to `maximumPermutations` as the kernels do, the seed,
`multipleTesting: 'none'`, the tested-window definition).
Poisson replicates are multinomial draws from an integer cumulative table (Philox streams of the
permutation contributors); the Bernoulli and space-time permutation draws run one invocation per
replicate (about `C` sequential draws, times `timeBuckets` for the latter). The result is deterministic
(fixed-order reductions, integer atomics, ties to the lowest index). Limits, the seed and the replicate
count are per-frame settings (`getGPUSpatialScanStatisticParameterValues(settings)`), up to the compile-time
`maximumPermutations` (at most 99999, SaTScan-style runs).

Construction captures capacities, contexts and nested output-port maps; later changes to the caller's
props do not alter the validated graph. Buffer contents and parameter views remain per-frame inputs.
Independent eight-zone, four-bucket fixtures cover two-bucket retrospective and prospective optima
for all three models, including every output and replicate maximum. Separate combinatorial fixtures
enumerate the conditional maximum distributions for multinomial, hypergeometric and fixed-margin
draws; fixed-seed CDF tests cover both serial draws and the parallel swap-or-not route on small domains.

Replicate batches: the observed data is evaluated once, then the replicates run in `batchCount` batches
of `batchReplicates` (`R`). A batch draws replicates `b * R + 1 .. b * R + R` into an `R`-slot count
table, evaluates them and folds each replicate's largest LLR into its row of `output.statistics`, which
is the running maximum, so no per-replicate table is kept. `R` is the largest count with
`(R + 1) * zones * timeBuckets <= 2^25` words (the 128 MiB binding limit), at most 4096 and at most
`maximumPermutations`; both are readable properties of the contributor. Philox streams are keyed by the
global replicate index and every reduction is a fixed-order sum or an order-independent maximum, so the
results do not depend on `R` and equal the former single pass bit for bit. A batch whose first replicate
is past the per-frame `permutationCount` is disarmed on the GPU (indirect dispatch gate) and launches
nothing. Each unrolled batch adds four graph nodes (`batch-control`, `batch-clear`, the draw kernel and
`batch-evaluate`) and no pipelines; `maximumPermutations` may need at most 1024 batches, which is only
binding for very large `zones * timeBuckets` (above about 340,000 cells, 99999 replicates need `R >= 98`).

Poisson baselines are scaled by a common power of two before summation, window construction and
replicate draws. The scale is the binary exponent of the largest exposure, preserving exact
binary relationships at population-share boundaries. Maximum reduction reads nonnegative IEEE
words as integers; significands and exponents also decode positive subnormal exposures before
any float arithmetic. Thus finite
nonnegative baselines can have a raw sum above the float32 range.
Scoring and published expected counts both multiply the case total by the window's baseline
fraction, so changing baseline units does not introduce an overflowing intermediate product.
Baseline shares use float32. Study totals, eligible prefixes and complete-circle masses accumulate
stored float32 exposures exactly before rounding once for scoring. Eligibility retains the exact
mass through multiplication by the packed float32 population fraction, including inclusive
half- and three-quarter-population boundaries. A complete-study window has expected cases equal
to the study case total and scores zero when it contains all cases.
All-zero cases at zero exposure are a valid no-cluster control. Positive cases at zero exposure
invalidate inference, as do negative, NaN or infinite baseline cells (`-0` counts as zero).
`status.invalidCount` counts incompatible or numerically lost cells: positive exposures more than
126 binary exponents below the maximum cannot survive the normalized float32 route and are counted.
Count/population overflow reports at least one. Invalid frames publish NaN inference, no clusters
and zero evaluated replicates in the summary, even when the optional status destination is omitted.
Baseline normalization and exact population reduction take linear work in the cell count.

Neighbor discovery: compile-time `nearestZoneSearch: 'auto'` uses exact coordinate indexes when
the list is shorter than the domain and (including the extra lookahead zone) has at most
`GPU_SCAN_MAXIMUM_INDEXED_NEIGHBOR_COUNT` (65) rows. Two published radix sorts index the coordinates;
each center binary-searches its position and searches outward along the wider planar axis, or
latitude for great-circle coordinates. Conservative axis bounds prune only strictly larger keys,
preserving `(separation key, zone row)` order, ties, and the lookahead zone across the antimeridian
and at the poles. The index is rebuilt on every submission when positions may change.

`'indexed'` explicitly selects that route and requires a list of at most 65 rows; `'full-scan'`
selects the exhaustive radix route. Auto uses full-scan for whole-domain or larger lists. The resolved route is
`contributor.nearestZoneSearch`. `contributor.estimatedFullScanSeparationCount` is an upper bound
on the exhaustive route's separation evaluations per submission: `5 * zones^2` for a truncated
list (four selection sweeps and at most one gather sweep), or `zones^2` when every zone is retained.
This excludes radix sorting. For `N = zones`, both scan classes expose
`estimatedIndexedCandidateSeparationCount = N²`,
`estimatedIndexedAxisSeparationCount = 2N(N+1)`, their sum as
`estimatedIndexedSeparationCount`, and `indexedRadixSortCount = 2` per indexed frame.
These conservative separation-call bounds exclude the two coordinate radix sorts. Pruning is
data-dependent: equal-latitude great-circle inputs can require all `N²` candidate separations,
because latitude cannot narrow the search. No longitude pruning is assumed across wraps or poles.
Discovery remains exact, with no candidate truncation or approximation flag.

`estimatedCommandNodeCount` separately bounds discovery sorts, model preparation, all replicate
batches and cluster rounds. The constructor checks it against the shared command-node budget.

Limits: ordinal and normal models, elliptic windows and Gumbel p-values are not provided. The count
table needs `(R + 1) * zones * timeBuckets <= 2^25` words (so `zones * timeBuckets <= 2^24`), and the
cost is about `(P + 1) * zones * windowZones * timeBuckets^2`, plus neighbor discovery above.
The full-scan route has two sort scratch tables the size of the neighbor tables; the indexed route
instead uses seven `zones`-word coordinate-key and row-order buffers, plus the published sorts' scratch.
Bounded lists append twelve words per center for Poisson and Bernoulli: rounded boundary population,
complete-circle row count and ten exact mass words. Space-time permutation needs only the first
two words per center, using its integer margins. The entire distance length is preflighted before
allocating scratch. The neighbor
table budget still counts `zones * listLength` entries. Bounded discovery runs after model population
preparation so it can include all members of the lookahead circle in that record.
The absolute LLR error is about `|c - e|` times the relative
f32 rounding of `e`; `baseline` must be finite and non-negative. Cluster rounds after the last cluster are
GPU-gated and launch nothing. No kernel binds more than 8 storage buffers, so the scan runs on `'core'`
WebGPU devices.

#### How it computes

{/* lens:GPUSpatialScanStatistic */}

::lens{scenario=gpu-spatial-analysis.spatial-scan-statistic rung=datum datum=15 caption="Cook County, Illinois (row 15 of the county table): the best window centred there."}

::lens{scenario=gpu-spatial-analysis.spatial-scan-statistic rung=formula caption="Poisson log-likelihood ratio of a window; p counts replicate maxima at or above it."}

::lens{scenario=gpu-spatial-analysis.spatial-scan-statistic rung=kernel caption="Inspect the recorded shader and its input and output bindings."}

::lens{scenario=gpu-spatial-analysis.spatial-scan-statistic rung=graph caption="Follow the recorded dependencies from uploaded inputs to captured outputs."}

{/* parameters: GPUSpatialScanStatistic generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUSpatialScanStatistic` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPUSpatialScanStatistic`.

- Packer: `getGPUSpatialScanStatisticParameterValues(settings)`, writes `GPU_SPATIAL_SCAN_STATISTIC_PARAMETER_LENGTH` (8) `uint32` values.

Settings of `getGPUSpatialScanStatisticParameterValues`:

| Field | Type | Description |
|---|---|---|
| `seed` | `number` | Random seed, an integer in `[0, 2^53)`. The result is a pure function of seed and inputs. |
| `permutationCount` | `number` | Monte Carlo replicates `P`, at most the contributor's `maximumPermutations`. |
| `maximumPopulationFraction` (optional) | `number` | Largest spatial window as a fraction of the total population, remaining in `(0, 1]` after float32 conversion. A window grows until adding the next zone would exceed it. Defaults to 0.5. |
| `maximumWindowZones` (optional) | `number` | Finite positive requested zone limit, rounded down (minimum 1); evaluated windows are also bounded by compile-time capacity. Omission leaves capacity as a storage constraint, with the population fraction governing eligible windows. An explicit limit also excludes larger windows from capacity loss diagnostics, even when it equals capacity. |
| `maximumTimeBuckets` (optional) | `number` | Finite positive cylinder length in buckets, rounded down (minimum 1) and clamped to `timeBuckets`. Defaults to 50% of the study period, `max(1, floor(timeBuckets / 2))`, as in SaTScan. |
| `windowShape` (optional) | `'circle' \| 'nearest'` | Window shape. Defaults to `'circle'`. |
| `rateDirection` (optional) | `'high' \| 'low' \| 'both'` | Which anomalies count as clusters. Defaults to `'high'`. |

{/* /parameters */}
