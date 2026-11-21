---
title: "GPU Spatial Analysis: Spatial statistics"
sidebar_label: "Spatial statistics"
description: "Spatial statistics contributors in @luma-contributors/analysis/gpu-spatial-analysis."
---


# Spatial statistics

Space-time clustering (ST-DBSCAN and ST-OPTICS through the optional `times` of `GPUDBSCAN` and
`GPUOPTICS`) is described in [Space-time clustering](./space-time-clustering.md). OPTICS exposes optional
one-word `output.ports.status.iterationCount`, `converged` and `invalidCount` views for visited rows,
completed ordering and excluded rows. Space-time statistics on
cubes are in [Space-time statistics](./space-time-statistics.md).


### `GPUGlobalSpatialStatistics`

Scalable rows are bounded by 4,294,963,199 and actual device bindings. Additive uint32 count outputs
preserve exact populations; float32 statistics remain rounded. Tiled reductions are the default;
mode reference retains its previous limit.

Summarise a tract column with global Moran’s I, Geary’s C and General G from queen weights, each with an analytic z-score and p-value.

::lens{scenario=gpu-spatial-analysis.global-spatial-statistics rung=result caption="Global Moran I = 0.60 (z 29.8), Geary and G for 791 Chicago tracts, with p-values."}

Global spatial autocorrelation from a `GPUSpatialWeights` CSR and a value column, each statistic with
its analytic expectation, variance, z-score and normal p-value (`analyticAlternative`, default `'two-sided'`).
Pick the statistics with `statistics` (compile-time). The `output.summary` layout (`GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT`) is fixed,
and blocks that were not requested hold NaN. `values` and `secondValues` may be `float32` or
`uint32` (for example the `uint32` counts of `addPointsInPolygonsChoroplethRecipe`); counts are
converted to f32 by one node, exact below 2^24, so the results equal those of an f32 copy bit for bit.

- `moran`: Moran's I with esda's `VI_norm` and `VI_rand` (sample kurtosis).
- `geary`: Geary's C with `VC_norm` and `VC_rand`.
- `getisOrdG`: Getis-Ord General G, which needs non-negative ratio-scale data: when any analysis
  value is negative the whole G block is NaN. Its randomization variance
  uses the Getis–Ord analytical moments, evaluated in centered form to avoid f32 cancellation in
  `E[G^2] - E[G]^2`. Both `varianceNormality` and `varianceRandomization` report this variance;
  their z and p fields use the same normal approximation. These are analytical results, not
  sampled permutations. [esda.G](https://pysal.org/esda/_modules/esda/getisord.html) calls these
  `VG` and `z_norm` and binarizes its weights. Match it with binary weights and `islands: 'keep'`.
  This card retains row-standardized weights for all three statistics, so its G is the weighted
  generalization. The former NaN normality fields were unpopulated slots, not failed arithmetic.
  Zeros are allowed when the pair-product denominator is positive; a zero denominator makes G
  and its variance undefined. Positive constant values give G = E[G], variance zero and NaN z/p.
- `bivariateMoran`: `(n / S0) sum w_ij zx_i zy_j / sqrt(sum zx^2 sum zy^2)`, which equals esda's
  `Moran_BV` when row-standardized weights have `S0 = n`. Under randomization of `y`, `E = 0`;
  variance is `(n/S0)^2 sum_j (b_j - mean(b))^2 / ((n-1) sum zx^2)`, where
  `b_j = sum_i w_ij zx_i`. Both variables must have positive variance; constant `y` makes the
  statistic, variance, z and p undefined. esda itself offers only permutation inference.
- `joinCount`: BB, BW and WW joins of a binary column (nonzero is black) over binarized weights.
  Ordered-pair counts in `output.joinCounts` are exact integers; summary counts are half those
  numbers and can be half-integers for asymmetric weights. Cliff-Ord randomization
  moments are given for BB and BW.

```ts
import type {Device} from '@luma.gl/core';
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUGlobalSpatialStatistics, type GPUGlobalSpatialStatisticsProps
} from '@luma-contributors/analysis/gpu-spatial-analysis';
declare const graph: GPUCommandGraph<void>;
declare const device: Device;
declare const weights: NonNullable<NonNullable<GPUGlobalSpatialStatisticsProps>['weights']>;
declare const values: NonNullable<NonNullable<GPUGlobalSpatialStatisticsProps>['values']>;
declare const summary: NonNullable<NonNullable<NonNullable<GPUGlobalSpatialStatisticsProps>['output']>['summary']>;
declare const joinCounts: NonNullable<NonNullable<NonNullable<GPUGlobalSpatialStatisticsProps>['output']>['joinCounts']>;

graph.add(new GPUGlobalSpatialStatistics({
  weights, values, statistics: ['moran', 'geary', 'getisOrdG', 'joinCount'],
  output: {summary, joinCounts}
}));
// summary[GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.moran + GPU_GLOBAL_SPATIAL_STATISTIC_FIELD.zRandomization]
```

Neighbors, islands and re-standardization follow the shared rules of the local statistics (see
`GPUHotSpotAnalysis` and `GPULocalMoran`):

- **Neighbors.** A neighbor is a slot with a finite positive weight to another analysis row.
  Nonfinite/nonpositive weights, the row's own slot and out-of-range IDs are not neighbors,
  consistently in tiled and reference modes.
- **Re-standardization.** Weights are used as given, except when `weights.information.transform` is `'row'`:
  each row is then divided by the sum of its active weights everywhere (`S0`, `S1`, `S2`, the reverse
  weights `w'_ji` and the cross products), so masked, missing and island neighbors drop out and the
  row still sums to 1 (esda on the subset `W`, re-row-standardized).
- **Islands.** An island is an included row (mask nonzero, every value finite) with no active slot.
  The rule looks at out-links only; for symmetric weights it equals the old in-or-out rule.
  `islands: 'exclude'` (default, compile-time) removes islands from `n`, the moments, `S0`, `S1`,
  `S2` and from the neighbors of other rows, in one pass with no cascade (a row whose only neighbors
  were islands keeps its place in `n` and adds no weight). `islands: 'keep'` is esda's convention.
  The summary `islandCount` reports the islands under either policy. esda parity needs `'keep'`.
- **Incomplete weights.** If either `weights.status.overflow` or `weights.status.candidateOverflow`
  is nonzero, the weight sums, mean, variance and every statistic block (including analytic
  p-values) are NaN. Population, island and black counts remain available; `output.joinCounts`
  holds 0 as an unavailable value, not an observed zero. Display consumers must check both loss
  flags: unavailable means or lags have no quadrant, and join counts are unavailable until recovery.
  This applies to packed and chunked weights in both `mode: 'tiled'` and
  `mode: 'reference'`, even when no output status is requested. Pass the producer's `outputWeights`
  directly so its loss flags travel with the CSR buffers.
  Observed joins use up to one fractional digit: asymmetric weights can produce half joins.
  The showcase tooltip reads the published join class (the no-class sentinel means no data),
  and the quadrant legend counts unavailable rows separately from its reported island total.
- **Loss diagnostics.** Optional `output.status.overflow` and `output.status.candidateOverflow`
  are separate uint32 scalars. They forward the producer's corresponding flags as 0 or 1,
  and write 0 for an absent input flag. Overflow reports final capacity loss; candidateOverflow
  reports intermediate loss. Contiguity pair-only loss reports `[0, 1]`, even if the final slots
  are ample. The statistic forwards these causes independently.

`GPUGlobalSpatialStatistics` rejects a supplied `weights.status.complete` certificate at either
matrix or child-chunk scope during construction, even alongside clear overflow flags. It does
not interpret that inverse-polarity certificate or label its loss as either output cause.
  Each is overwritten on every submission, so
  inference resumes when the producer clears both flags. Increasing final output capacity cannot
  recover dropped candidates. The summary layout, scalar allocation ownership and baseline limit
  of eight storage bindings per kernel are unchanged.
  Bind `weights.status.overflow` and `candidateOverflow` as one-row uint32 views, for packed weights
  and at the root of chunked weights. A view can select a word in a larger buffer. A supplied
  `requiredCount` needs the producer's final overflow flag covering those slots. For chunked
  weights, supply the producer's aggregate final overflow at the root. Verify completeness before
  omitting `converged`, `iterationLimitReached` or `unresolvedCount` and passing the CSR.

  For `GPULocalMoran`, a `requiredCount` above retained CSR slots (the terminal offset), a supplied
  `count`, or capacity also makes spatial statistics unavailable. Child terminal offsets are
  summed for a chunk-root requirement. Its output `overflow` includes this count-proven final
  truncation; `candidateOverflow` keeps the input candidate-loss meaning. Count views are captured,
  validated and checked for graph ownership before nodes are allocated.
- **Chunk status.** Loss flags must be aggregated on outer `weights.status`. Local Moran,
  hot-spot, local Geary and both global modes reject loss flags on individual CSR chunks at
  construction, even if the outer port is also present, rather than silently ignore them.
- `analyticAlternative: 'two-sided' | 'greater' | 'lesser'` selects the normal tail.
  Analytic tests accept those three tails.
  `getInferenceMetadata()` returns a `GPUInferenceMetadata` for a legend. Under non-Gaussian
  data prefer the randomization fields.

When every off-diagonal symmetric pair weight `(w_ij + w_ji)/2` has the same positive value,
Geary's C is identically 1 under every permutation. Its normality and randomization variances
are exactly 0; z-scores and p-values are undefined (NaN). Both modes determine this from the
effective pair weights, retaining addition-rounding errors, rather than rounded moment equalities.
On a complete symmetric support, variance uses deviations from a common pair weight, divided by
`S0` before squaring. This preserves tiny positive variances near invariance even when all weights
are rescaled: nine rows `0..8`, unit off-diagonal weights except `w_01 = 1 + 2^-23`, have
randomization variance `3.0702463789201337e-18` and z `-0.8819171037`, including when every weight is
multiplied by `2^-60`. Inference always uses centered `C-1` arithmetic. The published C is `1 + (C-1)`
when `|C-1| < 1/2`, so uniform complete weights give exactly 1, and otherwise its direct nonnegative
quadratic numerator: labels `[0,0,1,1]`, `w_01 = w_10 = 1` and every other off-diagonal weight
`2^-60` give positive C approximately `5.204170427930421e-18`. This pass does not depend on the
ordinary float32 ratio: labels `[1,1,1,1+2^-23]` with complete weights `2^-112` (`w_01` doubled)
give C `12/13`, randomization variance `1/169` and z `-1`.
These pair passes add per-chunk scratch arrays and a quadratic chunk-pair dispatch term;
the constructor's command-node budget includes them.

The summary block reports `n`, `S0`, `S1`, `S2`, the mean, the variance, the black count and the
island count. `n` is counted in `u32` and the mean gets a second-pass correction
(`mean += sum(x - mean) / n`), so a large offset does not bias the variance; the row count must be
at most 4,294,963,199 for scalable routes, with device binding limits and exact uint32 counts.
The statistics do not change when the values are multiplied by a constant, so the values are divided
by a power of two near their mean magnitude before the moment sums (an exact operation; only the
reported mean and variance are scaled back) and the weights enter the variances as `S1 / S0^2` and
`S2 / S0^2`. Large data therefore keeps its inference: 40,000 values near 300,000, a standard
deviation of 5e7, or 1.2e7 rows with 60 neighbors no longer overflow float32 in the General G,
Moran or Geary variances.
`w_ji` comes from a binary search of row `j`. Column sums come from a deterministic transpose: a
stable sort of slots by neighbor. Per-row partials are reduced by fixed-shape workgroup trees, and
join counts by u32 atomics. Results are bitwise reproducible, and values, weights and the mask
change per frame without a rebuild. Inference is NaN when `n < 4`, `S0 = 0`, or the values are
constant (G reports zero variance for positive constant values, with NaN z/p). Constant values give NaN Moran's I and Geary's C: the centering corrects the f32 mean
quotient with one `fma` remainder step, so constant data centers to exactly 0 on every adapter.
`E[I]` and `E[C]` are NaN below two rows (join counts of an empty input stay 0).

#### How it computes

{/* lens:GPUGlobalSpatialStatistics */}

::lens{scenario=gpu-spatial-analysis.global-spatial-statistics rung=datum bind=page datum=420 caption="Inspect tract 420: its poverty value is one of 791 rows entering the global sums."}
::lens{scenario=gpu-spatial-analysis.global-spatial-statistics rung=formula caption="Follow 4 equations: Moran’s I, its expectation, the randomisation z-score and the two-sided p-value."}
::lens{scenario=gpu-spatial-analysis.global-spatial-statistics rung=kernel caption="Inspect the row-gather, column-sum and finalize kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.global-spatial-statistics rung=graph caption="Trace queen weights, row standardisation and the fixed-order reductions into 52 summary values."}

### `GPUHotSpotAnalysis` and `GPULocalMoran`

`GPULocalMoran`: Scalable rows are bounded by 4,294,963,199 and actual device bindings. Additive
uint32 count outputs preserve exact populations; float32 statistics remain rounded.

`GPUHotSpotAnalysis`: Scalable rows are bounded by 4,294,963,199 and actual device bindings.
Additive uint32 count outputs preserve exact populations; float32 statistics remain rounded.

`GPUHotSpotAnalysis`, `GPULocalMoran` and `GPULocalGeary` accept optional uint32 scalars
`output.status.invalidCount`, `output.status.overflow` and `output.status.candidateOverflow`.
`invalidCount` counts selected input rows with nonfinite values; masked rows, islands and
undefined analytic tests do not count. `overflow` and `candidateOverflow` forward the producer's
corresponding final-capacity and intermediate-loss flags.
Each is 0 or 1,
and is 0 when its input port is absent. All requested status scalars are overwritten on every
encoding, independently of multiple-testing correction. Candidate loss is kept separate: increasing
the final capacity cannot recover candidates already dropped upstream.

`GPUHotSpotAnalysis.selfWeight` accepts a numeric default or a packed float32 view from
`getGPUHotSpotAnalysisParameterValues` hosted in a `GPUParameterBuffer`. Updating that buffer
between encodings switches between Gi (`0`) and Gi* (positive), or changes the focal weight,
without rebuilding the graph. Numeric defaults and the packer reject positive weights that
round below the smallest normal float32 (`2^-126`), preventing underflow or GPU flushing from
silently changing Gi* to Gi. Exactly zero remains valid for Gi.
The existing `parameters` view still controls fixed moments;
its layout and the fixed confidence bins are unchanged.

`minimumNeighbors` defaults to **1** for both hot spots and local Moran. An island never gets a
hot-spot z-score solely from its own value, including with a positive `selfWeight`.

Score tract values against neighbouring values with local Moran’s I, analytic conditional-randomisation z-scores, and corrected significance.

::lens{scenario=gpu-spatial-analysis.local-moran rung=result caption="Compare poverty percentages and local I across 791 Chicago tracts; significance is reported separately."}

`GPUHotSpotAnalysis`: Score each tract with Getis-Ord Gi* from binary queen weights, returning z-scores, 90, 95 and 99% confidence bins and corrected p-values.

::lens{scenario=gpu-spatial-analysis.hot-spot-analysis rung=result caption="Gi* z-scores for 791 Chicago tracts; 265 are significant at 95% after Benjamini-Hochberg correction."}

Local spatial autocorrelation on caller-supplied square `weights`. Rows count when their `mask` is
nonzero and their value is finite; excluded rows are neither foci nor neighbors. `n`, the mean `X`
and the population standard deviation `S` are taken over the analysis rows (see islands below), or
from per-frame `fixedMoments`. The mean gets a second-pass correction (`mean += sum(x - mean) / n`),
and the centered values are the stored first-pass deviations minus the mean of those deviations (one
scheme with `GPUGlobalSpatialStatistics`, the permutation tests and `GPUEmergingHotSpots`), so a large
offset biases neither the variance nor the z-scores; `n` is counted in `u32`. Sums run in slot order
with fixed-order reductions, so outputs are bitwise reproducible. Analytic p-values are never exactly
0: they are floored at the smallest normal float32 (1.17549435e-38), the same rule in the global,
local, hot-spot and emerging-hot-spot statistics.

Fixed moments pin the population count, mean and variance. Changing the mask can still change
active neighbors, re-standardized weights and local z-scores. Counts through 2^24 use the legacy
count word; larger counts through uint32 maximum use the packer's exact high/low 16-bit words.
`globalCounts` preserves that exact count; the float32 `globalStatistics` count is rounded.
Significance must remain strictly between zero and one after float32 packing, in both the
declarative parameter schema and the public packer.

For computed moments, Moran and Gi also detect a constant leave-one-out population using exact
extrema multiplicities. Such a conditional null has undefined z and p even when subtracting its
variance from the global moments leaves positive rounding noise. The local Moran statistic stays
defined. This donor check is disabled for external fixed moments, whose full population is unavailable.
Conditional inference is also undefined when equal effective weights cover every donor: its lag
is invariant even when the donors differ. CPU references count selected extrema and their
multiplicities once, so leave-one-out constant checks preserve linear sparse-graph traversal.

These contributors capture construction settings and port maps once. To change resident inputs
between submissions, write their buffer contents; replacing a caller's props, weights metadata or
output map does not change an existing contributor. Build a new contributor for those changes.

Shared semantics (also `GPULocalGeary`):

- **Inputs and outputs.** `values` is `float32` or `uint32` (counts are converted to f32 by one
  node, exact below 2^24), so the `uint32` counts of a points-in-polygons aggregation chain without a
  cast. Every caller-owned result view lives in one `output` bag (`GPUHotSpotAnalysisOutput`,
  `GPULocalMoranOutput`, `GPULocalGearyOutput`); the CPU settings type is
  `GPUSpatialAutocorrelationSettings`.
- **Neighbors.** A neighbor is a slot with a positive weight that points at another included row.
  Zero-weight slots, the focal row's own slot and out-of-range IDs are not neighbors: they are not
  counted in `neighborCounts` or `minimumNeighbors`. If either `weights.status.overflow` or
  `weights.status.candidateOverflow` is nonzero, the weights are incomplete. Floating statistic
  and inference outputs (including raw and adjusted p-values) are NaN, and significance classes
  (`quadrants` or `bins`) are 0. This also suppresses ungated Moran quadrants. The statistic's
  own optional status forwards both producer flags. `neighborCounts` is 0 for every row during
  loss, meaning unavailable, not that the retained CSR has no neighbors. Population/island counts
  retain their meanings; `invalidCount` still counts nonfinite input values rather than
  incomplete neighborhoods. When loss clears, active-neighbor counts are recomputed normally.
- **Re-standardization.** Weights are used as given, except when `weights.information.transform` is
  `'row'`: then each row's weights are divided by their sum over the active neighbors, so masked or
  missing neighbors drop out and the row still sums to 1 (esda on the subset `W`, re-row-standardized,
  not a lag that shrinks toward 0). Without masking this changes nothing beyond rounding. Weights
  with no declared transform are never rescaled.
- **Islands.** An island is an included row with no neighbor among the included rows.
  `islands: 'exclude'` (default, compile-time) removes islands from `n`, the mean and the variance and
  writes NaN for every output. `islands: 'keep'` is the esda convention: islands stay in the moments;
  their local statistic is the formula value (local I and the spatial lag are 0, local Geary is 0),
  and z and p are NaN because they are not tested. The decision is one pass with no cascade: a row
  whose only neighbors were islands has no neighbor among the analysis rows, gets NaN, and stays in
  `n`. `globalStatistics` is `[n, mean, variance, sd, islandCount]` under either policy. esda
  parity specs use `'keep'`.
- **`minimumNeighbors`** (default 1, compile-time integer). Rows with fewer neighbors get NaN z and p
  and bin or quadrant 0; the moments are unchanged. The default only rejects rows without a
  neighbor, since no row can have a z-score from its own value. Esri recommends at least 8
  neighbors for Gi* (pass `minimumNeighbors: 8`); the normal approximation of local Moran is poor
  below about 4.
- **`multipleTesting`** (`'none'` default, `'bonferroni'`, `'holm'`, `'benjamini-hochberg'`,
  `'benjamini-yekutieli'`, compile-time) adjusts the two-sided p-values through
  `GPUMultipleTestingAdjustment`; the family is every row with a defined z-score, independent of any
  viewport. The optional `output.adjustedPValues` holds the adjusted values (equal to `pValues` for
  `'none'`).
  `getInferenceMetadata(settings)` takes the same `GPUSpatialAutocorrelationSettings` that are packed
  into `parameters` (validated identically) and returns a `GPUInferenceMetadata` (two-sided,
  analytic, the method, the significance level, the tested-rows definition) for building a legend.
- **Legend contract.** Bins and quadrants mean the same thing for every correction method. A Gi*
  bin of level `L` means the adjusted p-value (the plain p-value for `'none'`) is at most `0.10`,
  `0.05` or `0.01` for `L = 1, 2, 3`, and the sign is the sign of z. A local Moran quadrant is the
  sign pair of the centered value and the lag, reported only where the adjusted p-value is at most
  `significanceLevel`. With `'none'`, Gi* bins use the critical values `1.645, 1.960, 2.576`,
  which is the same rule.

- `GPUHotSpotAnalysis` writes the Getis-Ord Gi* z-score per row for arbitrary weights:
  `z_i = sum_j w_ij (x_j - X) / (S sqrt((n S1_i - W_i^2) / (n - 1)))`, with `W_i = sum_j w_ij` and
  `S1_i = sum_j w_ij^2`. The focal row enters through `selfWeight` (default 1,
  classic Gi*), never through the matrix; a row's own value alone never produces a z-score. The
  z-score is unchanged when every value is shifted by a constant. It equals esda's
  `G_Local(star=True)` analytic `Zs` for positive data (esda normalizes by the sum of values, which
  flips the sign when the mean is negative). `selfWeight: 0` is the exact Ord-Getis Gi (esda
  `G_Local(star=False)`), which standardizes with the mean and variance of the `n - 1` other
  observations: `z_i = (sum_j w_ij x_j - W_i mu_i) / (s_i sqrt(((n - 1) S1_i - W_i^2) / (n - 2)))`.
  With row-standardized weights `selfWeight: 1` weighs the focal row as much as all its neighbors,
  so prefer `0` there. Optional outputs: the two-sided p-value, the adjusted p-value, the neighbor
  count (the focal row not included) and a `sint32` confidence bin in `-3..3` (99/95/90% hot or cold).
- `GPULocalMoran` writes the z-score of local Moran's I with the analytic moments of conditional
  randomization, the null esda's permutation test samples: `E[L_i] = W_i mu`,
  `Var[L_i] = sigma^2 (N S1_i - W_i^2) / (N - 1)` over the other `N = n - 1` rows. Optional outputs
  are `localI` (`(n - 1) c_i L_i / sum c^2`, esda's `Is`; NaN when the values are constant, 0 for
  kept islands), the spatial lag `L_i = sum_j w_ij c_j`,
  the p-value, the adjusted p-value, the neighbor count and the esda quadrant code (1 HH, 2 LH, 3 LL,
  4 HL; 0 when not significant at the per-frame `significanceLevel`).
  `quadrantGating` (`'analytic'` default or `'none'`, compile-time) chooses what `quadrants` holds. With
  `'analytic'` it is nonzero only where the analytic p-value (adjusted when `multipleTesting` is set) is
  significant. With `'none'` it
  is the ungated quadrant from the signs of the centered value and the spatial lag, for every included
  row with a nonzero centered value and lag, and 0 otherwise, so another test can gate it, for example the
  `significant` mask of `GPULocalPermutationTest` (esda's `q` with `p_sim`). `zScores` and `pValues` do
  not change. `'none'` cannot be combined with `multipleTesting` (the constructor throws).
  Local Moran significance does not depend on `|c_i|`: a row near the mean can be significant.

Checked against esda 2.10 on 200 points with 6-nearest-neighbor weights: `localI` matches `Is` to
3e-7, the z-score tracks esda's 99,999-permutation `z_sim` (correlation 0.999997, largest gap 0.03),
and Gi* matches `Zs` to 1e-7. A 60-row fixture with an island and masked rows pins `G_Local` (star
and non-star), `Moran_Local`, `Geary_Local` and the statsmodels adjustments for `islands: 'keep'`,
`'exclude'` and the re-standardized masked case.

```ts
import type {Buffer} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
import {
  GPUNeighborSearch, GPUHotSpotAnalysis, GPULocalMoran,
  getGPUNeighborSearchParameterValues,
  type GPUNeighborSearchProps, type GPUHotSpotAnalysisProps, type GPULocalMoranProps
} from '@luma-contributors/analysis/gpu-spatial-analysis';
declare const graph: GPUCommandGraph<void>;
declare const searchInputs: Pick<GPUNeighborSearchProps, 'positions' | 'output'>;
declare const searchParameters: GPUParameterBuffer<'float32'>;
declare const bounds: readonly [number, number, number, number];
declare const values: GPUHotSpotAnalysisProps['values'];
declare const parameters: NonNullable<GPULocalMoranProps['parameters']>;
declare const giZ: GPUHotSpotAnalysisProps['output']['zScores'];
declare const giP: NonNullable<GPUHotSpotAnalysisProps['output']['pValues']>;
declare const bins: NonNullable<GPUHotSpotAnalysisProps['output']['bins']>;
declare const moranI: NonNullable<GPULocalMoranProps['output']['localI']>;
declare const moranZ: GPULocalMoranProps['output']['zScores'];
declare const moranP: NonNullable<GPULocalMoranProps['output']['pValues']>;
declare const quadrants: NonNullable<GPULocalMoranProps['output']['quadrants']>;
declare const readbackBuffers: {giZ: Buffer; giP: Buffer; moranI: Buffer; moranZ: Buffer; moranP: Buffer};
declare const submitAndRequireComplete: () => Promise<void>;

const search = new GPUNeighborSearch({
  ...searchInputs, mode: 'radius', gridSize: [256, 256],
  parameters: searchParameters.importToGraph(graph)
});
graph.add(search);
graph.add(new GPUHotSpotAnalysis({
  weights: search.outputWeights, values, parameters,
  output: {zScores: giZ, bins, pValues: giP}
}));
graph.add(new GPULocalMoran({
  weights: search.outputWeights, values, parameters,
  output: {localI: moranI, zScores: moranZ, pValues: moranP, quadrants}
}));
searchParameters.write(getGPUNeighborSearchParameterValues({mode: 'radius', bounds, searchRadius: 500}));
await submitAndRequireComplete(); // Application submits and checks both search loss flags.
async function readFloat32(buffer: Buffer, view: GraphDataView<'float32'>): Promise<Float32Array> {
  const bytes = await buffer.readAsync(view.byteOffset, view.length * 4);
  return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
}
const gi = {z: await readFloat32(readbackBuffers.giZ, giZ), p: await readFloat32(readbackBuffers.giP, giP)};
const moran = {
  I: await readFloat32(readbackBuffers.moranI, moranI),
  z: await readFloat32(readbackBuffers.moranZ, moranZ),
  p: await readFloat32(readbackBuffers.moranP, moranP)
};
console.log({giZ: gi.z[0], giP: gi.p[0], moranI: moran.I[0], moranZ: moran.z[0], moranP: moran.p[0]});
```

Pass the producer's `outputWeights` directly. Rebuilding a weights object from only its CSR buffers
drops the search's completeness signals. For a line of four points spaced one unit apart, radius
1.1 requires six directed neighbor slots: capacity one reports overflow and suppresses Moran
inference; capacity six permits the defined tests. After candidate loss, even ample final capacity
must leave inference unavailable until the producer reports a complete neighborhood.

#### How it computes

{/* lens:GPUHotSpotAnalysis */}
{/* lens:GPULocalMoran */}

`GPULocalMoran`:

::lens{scenario=gpu-spatial-analysis.local-moran rung=datum bind=page datum=420 caption="Inspect tract 420: its value, neighbour lag, local I, and corrected p-value."}
::lens{scenario=gpu-spatial-analysis.local-moran rung=formula caption="Follow 2 equations: local I and its conditional-randomisation z-score."}
::lens{scenario=gpu-spatial-analysis.local-moran rung=kernel caption="Inspect 791 focal invocations and the retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.local-moran rung=graph caption="Trace queen weights, row standardisation, local moments, and FDR across 791 tracts."}

`GPUHotSpotAnalysis`:

::lens{scenario=gpu-spatial-analysis.hot-spot-analysis rung=datum bind=page datum=420 view=neighbourhood-view caption="Inspect tract 420: its queen neighbours, Gi* z-score, corrected p-value and confidence bin."}
::lens{scenario=gpu-spatial-analysis.hot-spot-analysis rung=formula caption="Follow the Gi* z-score: a weighted sum of centred values over its neighbour-count spread."}
::lens{scenario=gpu-spatial-analysis.hot-spot-analysis rung=kernel caption="Inspect 791 focal invocations, the moment passes, FDR adjustment and classify kernel."}
::lens{scenario=gpu-spatial-analysis.hot-spot-analysis rung=graph caption="Trace queen weights, moments, Gi* gather, Benjamini-Hochberg adjustment and confidence bins."}

{/* parameters: GPUHotSpotAnalysis generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUHotSpotAnalysis` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPUHotSpotAnalysis`.

- Packer: `getGPUHotSpotAnalysisParameterValues(settings?)`, writes `GPU_HOT_SPOT_ANALYSIS_PARAMETER_LENGTH` (1) `float32` values.
- Packer: `getGPUSpatialAutocorrelationParameterValues(settings?)`, writes `GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH` (8) `float32` values.
  - Input port: `parameters`.

Settings of `getGPUHotSpotAnalysisParameterValues`:

| Field | Type | Description |
|---|---|---|
| `selfWeight` (optional) | `number` | Non-negative focal weight. Zero selects Gi, a positive value selects Gi*. Positive weights must round to a normal float32 (at least 2^-126); smaller values are rejected rather than silently switching statistics. Defaults to 1. |

Settings of `getGPUSpatialAutocorrelationParameterValues`:

| Field | Type | Description |
|---|---|---|
| `significanceLevel` (optional) | `number` | Two-sided significance level of `GPULocalMoran` quadrants, in `(0, 1)`, applied to the adjusted p-value when `multipleTesting` is set. Defaults to `0.05`. `GPUHotSpotAnalysis` ignores it and always reports 90/95/99% bins. |
| `fixedMoments` (optional) | `GPUSpatialAutocorrelationFixedMoments` | Optional moments that replace the moments of the current rows. |

{/* /parameters */}

{/* parameters: GPULocalMoran generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPULocalMoran` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPULocalMoran`.

- Packer: `getGPUSpatialAutocorrelationParameterValues(settings?)`, writes `GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH` (8) `float32` values.
  - Input port: `parameters`.

Settings of `getGPUSpatialAutocorrelationParameterValues`:

| Field | Type | Description |
|---|---|---|
| `significanceLevel` (optional) | `number` | Two-sided significance level of `GPULocalMoran` quadrants, in `(0, 1)`, applied to the adjusted p-value when `multipleTesting` is set. Defaults to `0.05`. `GPUHotSpotAnalysis` ignores it and always reports 90/95/99% bins. |
| `fixedMoments` (optional) | `GPUSpatialAutocorrelationFixedMoments` | Optional moments that replace the moments of the current rows. |

{/* /parameters */}

### `GPULocalGeary`

Scalable rows are bounded by 4,294,963,199 and actual device bindings. Additive uint32 count outputs
preserve exact populations; float32 statistics remain rounded.

Measure how much each tract differs from its neighbours with local Geary c, where small values mean similar surroundings.

::lens{scenario=gpu-spatial-analysis.local-geary rung=result caption="Local Geary c across 791 Chicago tracts, mean 0.81; small values mark tracts like their neighbours."}

Neighbor, row re-standardization and `islands` semantics follow `GPULocalMoran`. The output is
`output.statistic`; the parameters and the permutation adapter are described under
Additional spatial statistics below.

#### How it computes

{/* lens:GPULocalGeary */}

::lens{scenario=gpu-spatial-analysis.local-geary rung=datum bind=page datum=420 caption="Inspect tract 420: its value, its queen neighbours’ values and the resulting c."}
::lens{scenario=gpu-spatial-analysis.local-geary rung=formula caption="Follow the rule: squared differences to each neighbour, weighted, divided by the variance."}
::lens{scenario=gpu-spatial-analysis.local-geary rung=kernel caption="Inspect 791 focal invocations over the retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.local-geary rung=graph caption="Trace queen weights, row standardisation, moments and the local Geary kernel."}

### `GPULocalSpatialHeteroscedasticity`

Scalable rows are bounded by 4,294,963,199 and actual device bindings. Additive uint32 count outputs
preserve exact populations; float32 statistics remain rounded.

Descriptive LOSH divides each row's weighted neighbor residual power by the study-wide mean
residual power. Positive usable neighbors are normalized within each row. Masked rows, nonfinite
values and islands have no residual; an all-zero or absent residual field leaves the statistic
undefined. The per-row `output.status` codes remain `0` defined, `1` excluded, `2` undefined and
`3` incomplete weights.

When the focus and every eligible neighbor have the same finite value, their residual is exactly
zero for either power. This avoids turning GPU weighted-mean division error into a normalized
LOSH pattern. Such zero residuals still count in `residualCount`; an entirely zero residual field
has undefined statistics and normalized residuals (NaN), with per-row status 2 for eligible rows.
Residuals accumulate weighted differences from the focus before division, so a focus equal to
the mean of unequal neighbors also retains zero when balanced differences cancel. Addition and
product rounding errors are retained using integer-significand helpers, so the four-neighbor
sum `1 + float32(.1) - 1 - float32(.1)` does not leave a false residual. Subnormal rounding-error
terms remain subject to the helpers' documented float32 limits.

Either upstream overflow or candidateOverflow (including a flag on an individual chunk) makes
all LOSH statistics and normalized residuals NaN, every per-row status INCOMPLETE, and residualCount
zero. The same compiled graph recovers after both input loss causes clear. The local-variability
recipe therefore suppresses LOSH consistently with its Moran stage.

The global mean uses parallel maximum, count and sum reductions. Residuals are divided by their
maximum before summation, so a finite mean does not overflow merely because its unscaled total
would overflow. Exponent-scaled quotients avoid underflow of a large divisor's reciprocal;
the mean is reconstructed from a bounded mantissa and two finite power-of-two scales.
No invocation scans all rows to form the mean. Float32 results may differ from
the former running mean through reduction order.

Per-frame `residualPower` accepts a packed float32 view from
`getGPULocalSpatialHeteroscedasticityParameterValues` hosted in a `GPUParameterBuffer`; write
`1` or `2` between encodings to switch absolute deviations and squared deviations. Numeric
powers remain supported as uploaded defaults. View shapes, optional outputs and chunk topology
are compile-time. The [conditional residual inference](./losh-residual-permutation.md) wrapper
accepts the same parameter view.

{/* parameters: GPULocalSpatialHeteroscedasticity generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPULocalSpatialHeteroscedasticity` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPULocalSpatialHeteroscedasticity`.

- Packer: `getGPULocalSpatialHeteroscedasticityParameterValues(settings?)`, writes `GPU_LOCAL_SPATIAL_HETEROSCEDASTICITY_PARAMETER_LENGTH` (1) `float32` values.

Settings of `getGPULocalSpatialHeteroscedasticityParameterValues`:

| Field | Type | Description |
|---|---|---|
| `residualPower` (optional) | `1 \| 2` | Absolute-deviation power: 1 for absolute deviation, 2 for variance. Defaults to 2. |

{/* /parameters */}

### `GPUMultivariateLocalGearyPermutationTest`

Conditional randomization tests multivariate local Geary statistics over complete, unmasked
rows. Islands and incomplete rows remain untested; incomplete weights invalidate inference.
Both outer and chunk-local `overflow` and `candidateOverflow` flags suppress the statistic,
p-values and retained simulations with NaN and classes with `0xffffffff`; clearing the flags
restores the same graph. Included islands remain in the population moments and donor pool.
`correction: 'bonferroni'` counts all complete rows, including islands, while `multipleTesting`
uses only rows with a finite p-value. Equal-tail inference includes ties in both tails; the
legacy `esda-min-tail` uses the strict complement `P - upper`, so an all-tied conditional
distribution has p-value `1 / (P + 1)`. A significant value equal to the simulated mean receives
class 2 under the existing direction convention.
Direction uses mean simulated-minus-observed differences, with tied draws contributing exact zero;
ten draws all equal to 4 therefore retain class 2 and legacy p-value 1/11.

Both descriptive modes and both inference strategies center columns on a finite eligible reference
observation in a separate stored pass before their running moment updates. This preserves population
variance for representable translations close to a float32 spacing. Observed and simulated
distances use differences of original values divided by the column deviation; they never subtract
two stored standardized donors, which could merge distinct small observations far from the reference.
The inference constructor exposes `estimatedCommandNodeCount`, including each chunk's observed
row tiles and row-by-permutation simulation tiles, setup/reductions, loss folding and correction, and rejects an
estimate above 32,768 before scratch allocation. Smaller tiles can increase this graph cost.

Numeric significance must remain strictly between zero and one after float32 packing. Tiled
inference does not retain the simulation matrix: omit `output.simulations`, or use `strategy:
'exact'` within its 256-row and 4096-permutation limits to retain every draw.

Both strategies accept `seed` as a nonnegative safe integer or exactly two uint32 Philox key
words. Defaults are seed `[0,0]`, 999 fixed draws, significance level .05 and inclusive equal
tails. Set `permutationCount` and `significanceLevel` to change the defaults. The optional
resident uint32 `parameters` view overrides numeric seed and
threshold defaults; its three words are `[seedLow, seedHigh, float32 bits of significanceLevel]`.
It contains no active draw count: draws are fixed at construction. Use the packer for each
frame's settings. On the fixed route `getInferenceMetadata()` decodes captured words and an
optional argument cannot override inference. With resident parameters, pass actual current
settings or packed Uint32Array words to `getInferenceMetadata(...)`; omission throws.

`pseudoPValues` contains the raw values. `adjustedPValues` contains the values after correction
and equals the raw output when correction is disabled. `family: 'tested'|'complete'` controls
whether an included island contributes to Bonferroni's denominator. An explicit family requires
Bonferroni; other corrections or none reject it. Omitted family defaults to complete for
`correction:'bonferroni'`, tested for `multipleTesting:'bonferroni'`. Untested observation rows
have NaN p-values.

Values, mask, CSR contents and parameter contents are per-frame. View shapes, variable count,
maximum permutations, strategy, tile sizes, maximum neighbors, tail convention, correction choice
and optional outputs stay compile-time. The tiled simulation stages the parameter words beside
the valid-row count to keep each kernel within eight storage-buffer bindings. The conditional
randomization, tie handling and multiple-testing formulas are unchanged.

{/* parameters: GPUMultivariateLocalGearyPermutationTest generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUMultivariateLocalGearyPermutationTest` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPUMultivariateLocalGearyPermutationTest`.

- Packer: `getGPUMultivariateLocalGearyPermutationTestParameterValues(settings?)`, writes `GPU_MULTIVARIATE_LOCAL_GEARY_PERMUTATION_TEST_PARAMETER_LENGTH` (3) `uint32` values.

Settings of `getGPUMultivariateLocalGearyPermutationTestParameterValues`:

| Field | Type | Description |
|---|---|---|
| `seed` (optional) | `number \| readonly [number, number]` | Nonnegative safe-integer seed or two uint32 Philox words, preserving every high bit. Defaults to [0, 0]. |
| `significanceLevel` (optional) | `number` | Significance threshold in (0, 1), compared as float32. Defaults to 0.05. |
| `permutationCount` (optional) | `number` | Fixed number of draws, resolved at construction, not a per-frame parameter. Defaults to 999. |

{/* /parameters */}

### `GPUAssuncaoReisStandardization`

Standardize each tract's rate against the pooled rate and Poisson noise, turning event counts and populations into one z-score per tract.

::lens{scenario=gpu-spatial-analysis.assuncao-reis-standardization rung=result caption="Standardized poverty rates of 791 tracts range from -1.5 to 3.3; the 3 zero-population tracts stay missing."}

Empirical-Bayes standardized rates of Assuncao and Reis (1999), the transformation behind
`esda.Moran_Rate` and `esda.Moran_Local_Rate`. It writes one standardized value per row and no
spatial statistic; Moran statistics of rate data come from composing it with the existing nodes.
Lives in `spatial-autocorrelation/gpu-rate-standardization`.

```ts
import type {Device} from '@luma.gl/core';
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUAssuncaoReisStandardization, type GPUAssuncaoReisStandardizationProps
} from '@luma-contributors/analysis/gpu-spatial-analysis';
declare const graph: GPUCommandGraph<void>;
declare const device: Device;
declare const events: NonNullable<NonNullable<GPUAssuncaoReisStandardizationProps>['events']>;
declare const populations: NonNullable<NonNullable<GPUAssuncaoReisStandardizationProps>['populations']>;
declare const mask: NonNullable<NonNullable<GPUAssuncaoReisStandardizationProps>['mask']>;
declare const standardizedRates: NonNullable<NonNullable<NonNullable<GPUAssuncaoReisStandardizationProps>['output']>['standardizedRates']>;
declare const statistics: NonNullable<NonNullable<NonNullable<GPUAssuncaoReisStandardizationProps>['output']>['statistics']>;
declare const invalidCount: NonNullable<NonNullable<NonNullable<NonNullable<GPUAssuncaoReisStandardizationProps>['output']>['status']>['invalidCount']>;

const standardization = new GPUAssuncaoReisStandardization({
  events, // float32 or uint32 view (counts are converted to f32, exact below 2^24)
  populations, // float32 or uint32 view, population at risk
  mask, // optional uint32
  output: {
    standardizedRates, // float32 view, NaN for rows outside the analysis
    statistics, // optional, [n, b, s2, a, totalPopulation]
    status: {invalidCount} // optional uint32 scalar
  }
});
```

Definition (identical to `esda.smoothing.assuncao_rate`), over the `m` valid rows with rates
`p_i = e_i / n_i`: `b = sum e / sum n`, `s2 = sum n_i (p_i - b)^2 / sum n`,
`a = s2 - b / (sum n / m)`, `v_i = a + b / n_i` (replaced by `b / n_i` where `v_i < 0`), and
`z_i = (p_i - b) / sqrt(v_i)`.

- **Negative prior variance.** `a` is negative when the rates vary less than Poisson noise predicts.
  esda does not clamp `a`: it keeps the negative `a` wherever `v_i` stays non-negative and falls
  back to the pure Poisson variance `b / n_i` only for rows where `v_i < 0` (large populations,
  where `b / n_i` is smallest). This node does the
  same, for parity, and `statistics[3]` reports the unclamped `a`. The EB smoothing in
  `rate-smoothing` clamps the prior variance at zero instead (a different, also standard
  convention for shrinkage); the two are not meant to agree when `a < 0`.
- **Missing rows.** A row is valid when it is unmasked, its events are finite and non-negative, and
  its population is finite and positive. Other rows write NaN, are excluded from `b`, `s2`, `m`
  and the mean population, and are excluded by the downstream statistic. `output.status.invalidCount`
  counts the unmasked rows that were rejected. Rows where `v_i` is not positive (for example `b = 0`)
  write NaN, as esda's 0 / 0 does.
- **Precision.** `m` is a u32; sums are fixed-order two-level workgroup trees (no float atomics,
  bitwise reproducible); `b` gets a second-pass correction and `s2` is formed from centered
  deviations. Scalable count/index bounds and device limits apply.

Composition (no statistic code is duplicated):

```ts
import {createTransientView} from '@luma.gl/graph/gpu-core';
import type {Device} from '@luma.gl/core';
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUAssuncaoReisStandardization, type GPUAssuncaoReisStandardizationProps, GPUGlobalSpatialStatistics, type GPUGlobalSpatialStatisticsProps, GPULocalMoran, type GPULocalMoranProps
} from '@luma-contributors/analysis/gpu-spatial-analysis';
declare const graph: GPUCommandGraph<void>;
declare const device: Device;
declare const rows: number;
declare const events: NonNullable<NonNullable<GPUAssuncaoReisStandardizationProps>['events']>;
declare const populations: NonNullable<NonNullable<GPUAssuncaoReisStandardizationProps>['populations']>;
declare const weights: NonNullable<NonNullable<GPULocalMoranProps>['weights']>;
declare const parameters: NonNullable<NonNullable<GPULocalMoranProps>['parameters']>;
declare const zScores: NonNullable<NonNullable<NonNullable<GPULocalMoranProps>['output']>['zScores']>;
declare const localI: NonNullable<NonNullable<NonNullable<GPULocalMoranProps>['output']>['localI']>;
declare const summary: NonNullable<NonNullable<NonNullable<GPUGlobalSpatialStatisticsProps>['output']>['summary']>;

const standardizedRates = createTransientView(graph, 'rates', 'float32', rows);
graph.add(new GPUAssuncaoReisStandardization({events, populations, output: {standardizedRates}}));

// EB Moran's I (esda.Moran_Rate, transformation 'r'): row-standardized weights.
graph.add(
  new GPUGlobalSpatialStatistics({
    weights, // weights.information.transform === 'row'
    values: standardizedRates,
    statistics: ['moran'],
    islands: 'keep', // esda's convention
    output: {summary}
  })
);
// EB local Moran (esda.Moran_Local_Rate): GPULocalMoran, or GPULocalPermutationTest('localMoran').
graph.add(
  new GPULocalMoran({
    weights,
    values: standardizedRates,
    parameters,
    islands: 'keep',
    output: {zScores, localI}
  })
);
```

`summary[moran + statistic]`, `summary[moran + expected]` and `summary[moran + zRandomization]`
equal `Moran_Rate.I`, `.EI` and `.z_rand` (esda recomputes the Moran of the standardized values and
ignores the EB sampling variance); `localI` equals `Moran_Local_Rate.Is`. Row-standardized weights
are re-standardized over the active neighbors, so NaN rows drop out of the neighborhoods of the
rows that remain.

#### How it computes

{/* lens:GPUAssuncaoReisStandardization */}

::lens{scenario=gpu-spatial-analysis.assuncao-reis-standardization rung=datum bind=page datum=420 caption="Inspect tract 420: its events, population, crude rate and standardized z."}
::lens{scenario=gpu-spatial-analysis.assuncao-reis-standardization rung=formula caption="Follow z: the tract rate minus the pooled rate, over its Poisson-plus-prior spread."}
::lens{scenario=gpu-spatial-analysis.assuncao-reis-standardization rung=kernel caption="Inspect the three block-sum passes, the mean correction and the standardize kernel."}
::lens{scenario=gpu-spatial-analysis.assuncao-reis-standardization rung=graph caption="Trace the totals, mean correction, variance and per-tract standardization passes."}

### `GPUMultipleTestingAdjustment`

Exact integer ranks support at most 2,147,483,648 rows, subject to published sort and device
bindings.

Adjust a column of per-tract p-values for multiple testing, returning corrected values that are significant when at most alpha.

::lens{scenario=gpu-spatial-analysis.multiple-testing-adjustment rung=result caption="Hot spot p-values of 754 tested tracts before and after Benjamini-Hochberg: 369 pass 0.05 unadjusted, 265 adjusted."}

Adjusts a column of per-row p-values for multiple testing on the GPU. `method` is `'none'`,
`'bonferroni'`, `'holm'`, `'benjamini-hochberg'` or `'benjamini-yekutieli'`, and
`output.adjustedPValues` holds the same numbers as statsmodels `multipletests` (`bonferroni`, `holm`, `fdr_bh`, `fdr_by`), in
`[0, 1]`. `getMultipleTestingAdjustedPValues` is the float64 CPU reference with the same semantics.

- **Permutation p-value floor.** A permutation p-value is never below `m / (P + 1)` (`m` is 2 for the
  `'two-sided'` default, else 1), so with `P = 999` the two-sided floor is 0.002. Then
  Bonferroni and Holm cannot reject in a family of more than 25 rows at `alpha = 0.05`, and with
  10,000 rows Benjamini-Hochberg can reject with 400 rows at the floor and all others at one. Nothing warns at run
  time. `getPermutationRejectionFloor({permutationCount, alternative, rowCount, alpha, method})`
  is the CPU helper that answers it before you run: it returns `pValueFloor`, `canReject`,
  `minimumRowsAtFloor` (Infinity when `canReject` is false) and, for Bonferroni and Holm,
  `maximumRowCount`. The minimum counts rows sufficient **when all other p-values are one**;
  it is not necessary for a general family. For example, with 99 permutations and eight rows,
  the two-sided floor is .02 and the helper returns four at alpha=.05 under BH. Eight attainable
  p-values of .04 all adjust to .04 and reject, with zero rows at the floor.
  At alpha=1 every tested row rejects because adjustments cap at one; minimumRowsAtFloor is zero
  for a nonempty family (even all-one p-values reject), and maximumRowCount is then
  Infinity for Bonferroni and Holm. The helper uses the supplied binary64 alpha without tolerance
  slack or float32 rounding. For Benjamini-Yekutieli the floor scenario is multiplied by the harmonic sum of
  `rowCount`. Near a decision boundary the budget helper resolves this comparison with rational arithmetic
  or certified high-precision bounds before rounding the harmonic factor. The same predicate determines
  required draws and sufficient floor occupancy. For 73 rows, 255 two-sided draws and
  alpha=0.03808209987270469, it reports canReject=false, minimumRowsAtFloor=Infinity and 256 required draws;
  the next representable host cutoff permits rejection with 255 draws. Raise
  `permutationCount` (the floor needs `P + 1 >= m rowCount / alpha` for Bonferroni
  and Holm) when `canReject` is false. The helper requires an integer `permutationCount` in
  `[1, 2^31 - 1]`, an integer `rowCount` in `[0, 2^32 - 1]`, `alpha` in `(0, 1]`, and a supported
  `alternative`; invalid settings throw. Sequential stopping does not change the floor.

- **Family.** The family is every tested row: a row whose p-value is finite and in `[0, 1]`. NaN
  rows are not tested and get NaN. There is no mask prop by design: run the statistic and this node
  on the full analysis set and apply a viewport or display mask only when rendering, otherwise
  panning changes the family size `m` and makes the visible rows look more or less significant.
- **`minimumFamilySize`.** `m = max(testedRows, minimumFamilySize)`; rows without a p-value count as
  `p = 1`. Use it to pin `m` when the tested set can shrink (for example an analysis mask that
  follows the viewport). `output.finiteTestCount` reports the number of finite probabilities in `[0,1]`;
`output.effectiveFamilySize` reports `m = max(finiteTestCount, minimumFamilySize)`, the denominator
used for correction. The two quantities are independently selectable. Each count view holds
exactly one uint32 word.
  The GPU contributor accepts `minimumFamilySize` up to 2^31. The host helper
  `getMultipleTestingAdjustedPValues` and CPU twin `computeMultipleTestingAdjustmentOnCPU` accept
  up to 2^32 - 1. Their allocation and sorting depend only on the supplied p-value rows, including
  when the logical family is larger. For Benjamini-Yekutieli the host harmonic factor uses at most
  64 terms, then an Euler-Maclaurin expansion with remainder below binary64 rounding.

The exported bound is `MAXIMUM_MULTIPLE_TESTING_EXACT_HARMONIC_TERMS = 64`.
- **Which method.** Local statistics of neighboring rows share observations, so their p-values are
  dependent. Benjamini-Hochberg only controls the false discovery rate under independence or
  positive regression dependence; use `'benjamini-yekutieli'` (Benjamini-Hochberg times
  `c(m) = sum 1/i`, a cost of about `ln m` in power) for controlled error under arbitrary
  dependence. Holm controls the family-wise error rate and is uniformly more powerful than
  Bonferroni.
- **Significance.** For every method a row is significant at level `alpha` exactly when
  `adjustedPValue <= alpha`, so 0.10 / 0.05 / 0.01 bins are consistent by construction. Tied
  p-values get identical adjusted values.
- **Cost.** `none` and Bonferroni are O(n). The step procedures add a 30-bit `GPUSort` plus a
  deterministic cumulative maximum (Holm) or minimum (Benjamini) scan in workgroup memory, a few
  linear kernels in all. At most 2,147,483,648 rows (ranks are exact uint32 integers), within device limits; results are
  independent of scheduling.

```ts
import type {Device} from '@luma.gl/core';
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUMultipleTestingAdjustment, type GPUMultipleTestingAdjustmentProps
} from '@luma-contributors/analysis/gpu-spatial-analysis';
declare const graph: GPUCommandGraph<void>;
declare const device: Device;
declare const pValues: NonNullable<NonNullable<GPUMultipleTestingAdjustmentProps>['pValues']>;
declare const adjustedPValues: NonNullable<NonNullable<NonNullable<GPUMultipleTestingAdjustmentProps>['output']>['adjustedPValues']>;

graph.add(new GPUMultipleTestingAdjustment({
  pValues, method: 'benjamini-yekutieli', output: {adjustedPValues}
}));
// significant at 5%: adjustedPValues[i] <= 0.05
```

#### How it computes

{/* lens:GPUMultipleTestingAdjustment */}

::lens{scenario=gpu-spatial-analysis.multiple-testing-adjustment rung=datum bind=page datum=420 view=fdr-staircase caption="Inspect tract 420: its p-value, rank in the sorted family and adjusted p-value."}
::lens{scenario=gpu-spatial-analysis.multiple-testing-adjustment rung=formula caption="Follow the step-up rule: a running minimum of p times m over rank."}
::lens{scenario=gpu-spatial-analysis.multiple-testing-adjustment rung=kernel caption="Inspect the count, 30-bit sort, scan and scatter kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.multiple-testing-adjustment rung=graph caption="Trace the hot spot p-values through counting, sorting, the cumulative minimum and scatter."}

```ts
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUMultipleTestingAdjustment, type GPUMultipleTestingAdjustmentProps
} from '@luma-contributors/analysis/gpu-spatial-analysis';
declare const graph: GPUCommandGraph<void>;
declare const pValues: GPUMultipleTestingAdjustmentProps['pValues']; // Upload [.02, NaN].
declare const output: Required<Pick<GPUMultipleTestingAdjustmentProps['output'],
  'adjustedPValues' | 'finiteTestCount' | 'effectiveFamilySize'>>;
graph.add(new GPUMultipleTestingAdjustment({
  pValues, method: 'bonferroni', minimumFamilySize: 10, output
}));
// After submission/readback: adjusted≈[.20,NaN], finiteTestCount=1, effectiveFamilySize=10.
// Thresholding at alpha=.05 produces mask [0,0] and significantCount=0.
// method:'none' preserves [.02,NaN], counts remain 1 and 10, mask [1,0], significantCount=1.
```

### `GPURasterFalseDiscoveryRateMask`

Exact integer ranks support at most 2,147,483,648 cells, subject to sort and device binding limits.

Field significance of a p-value raster: adjust every tested cell for multiple testing and mask the cells that survive.

Run it on the p-value output of a Mann-Kendall, Welch or correlation operator so a trend map shows only
cells that remain significant after correction (Wilks 2016: testing every grid point without a
multiplicity adjustment overstates the result). It composes `GPUMultipleTestingAdjustment` on the cells
whose p-value is finite and in `[0, 1]` and whose optional `validity` is nonzero, so the family is exactly
the valid cells and is not changed by a display mask. `method` is `'benjamini-hochberg'` (default),
`'benjamini-yekutieli'` (for dependent local statistics), `'holm'` or `'bonferroni'`.

- `output.significant` is 1 where `adjusted <= alpha`, which is statsmodels' `multipletests` `reject`.
  `alpha` is a per-frame parameter (`getGPURasterFalseDiscoveryRateMaskParameterValues`).

  A target must contain at least one float32 word;
  short targets reject before mutation and larger targets retain their trailing words.
- Optional outputs include `adjustedPValues`, `finiteTestCount` (after the `validity` mask),
  `effectiveFamilySize` (after minimum-family padding), and `significantCount` (the number of nonzero
  significant-mask cells after adjustment and alpha thresholding). All raster count views are
  exactly one uint32 word. A zero significance count does not imply an empty tested family. Other outputs include
  `criticalPValue` (the largest raw p-value that is significant, the Benjamini-Hochberg step-up
  threshold; NaN when none) and `histogram`, the counts of tested p-values over equal bins of `[0, 1]`
  by the published `GPURasterHistogram`. A histogram that is flat means no signal; a spike near zero
  means there is some.
- **Reference.** The adjusted p-values reuse the statsmodels fixtures of `GPUMultipleTestingAdjustment`;
  the mask is checked against those, against the 15 p-values of Benjamini and Hochberg (1995, four
  rejections at q = 0.05) and against the textbook step-up and step-down rules written independently.
  statsmodels itself was not run for the mask.
- **float32 limit.** Adjusted p-values are float32, so a cell whose adjusted value is within a few ulp of
  `alpha` may fall on either side of statsmodels' float64 decision.
- At most 2,147,483,648 cells, subject to sort and device limits. This is a field operation on a linear cell array, so it needs no raster grid.

```ts
import type {Device} from '@luma.gl/core';
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPURasterFalseDiscoveryRateMask, type GPURasterFalseDiscoveryRateMaskProps
} from '@luma-contributors/analysis/gpu-spatial-analysis';
declare const graph: GPUCommandGraph<void>;
declare const device: Device;
declare const trendPValues: NonNullable<NonNullable<GPURasterFalseDiscoveryRateMaskProps>['pValues']>;
declare const trendValidity: NonNullable<NonNullable<GPURasterFalseDiscoveryRateMaskProps>['validity']>;
declare const alphaParameters: NonNullable<NonNullable<GPURasterFalseDiscoveryRateMaskProps>['parameters']>;
declare const significant: NonNullable<NonNullable<NonNullable<GPURasterFalseDiscoveryRateMaskProps>['output']>['significant']>;
declare const criticalPValue: NonNullable<NonNullable<NonNullable<GPURasterFalseDiscoveryRateMaskProps>['output']>['criticalPValue']>;
declare const width: number;
declare const height: number;

graph.add(new GPURasterFalseDiscoveryRateMask({
  cellCount: width * height,
  pValues: trendPValues,
  validity: trendValidity,
  method: 'benjamini-hochberg',
  parameters: alphaParameters, // getGPURasterFalseDiscoveryRateMaskParameterValues({alpha: 0.05})
  output: {significant, criticalPValue}
}));
```

{/* lens:GPURasterFalseDiscoveryRateMask */}

```ts
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPULocalMoran, GPULocalPermutationTest, GPUMultipleTestingAdjustment,
  type GPULocalMoranProps, type GPULocalPermutationTestProps,
  type GPUMultipleTestingAdjustmentProps
} from '@luma-contributors/analysis/gpu-spatial-analysis';
declare const graph: GPUCommandGraph<void>;
declare const commonInputs: Parameters<typeof GPULocalMoran.createCommonMaskNodes>[1];
declare const moranInputs: GPULocalMoranProps;
declare const permutationInputs: GPULocalPermutationTestProps;
declare const adjustedPValues: GPUMultipleTestingAdjustmentProps['output']['adjustedPValues'];
const common = GPULocalMoran.createCommonMaskNodes(graph, commonInputs);
for (const node of common.nodes) graph.add(node);
graph.add(new GPULocalMoran({...moranInputs, mask: common.mask}));
graph.add(new GPULocalPermutationTest({...permutationInputs, mask: common.mask}));
graph.add(new GPUMultipleTestingAdjustment({
  pValues: permutationInputs.output.pseudoPValues,
  method: 'benjamini-hochberg', output: {adjustedPValues}
}));
const diagnostics = {membershipStatus: common.membershipStatus, complete: common.complete};
```

Use the same observation values/row domain in both consumers. `commonInputs.mask` is an
observation/polygon-row membership mask, never a per-point candidateTruncated mask. Supplied
valuesComplete and membershipStatus.complete must be nonzero, while supplied membership
overflow, candidateOverflow, uncertainCount, antimeridianUncertainCount and poleUncertainCount
must be zero. Any failure suppresses the whole family. Retain diagnostics beside corrected
probabilities; an empty tested family from incomplete values cannot certify absence of signal.

{/* parameters: GPURasterFalseDiscoveryRateMask generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPURasterFalseDiscoveryRateMask` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPURasterFalseDiscoveryRateMask`.

- Packer: `getGPURasterFalseDiscoveryRateMaskParameterValues(settings)`, writes `GPU_RASTER_FALSE_DISCOVERY_RATE_MASK_PARAMETER_LENGTH` (1) `float32` values.

Settings of `getGPURasterFalseDiscoveryRateMaskParameterValues`:

| Field | Type | Description |
|---|---|---|
| `alpha` | `number` | Significance level (false discovery rate for the Benjamini procedures), in `(0, 1]`. |

{/* /parameters */}

### `GPULocalPermutationTest` and `GPUGlobalPermutationTest`

For recording one row's raw conditional draws, import `createLocalPermutationReplay` and
`LocalPermutationReplayProps` from `@luma-contributors/analysis/gpu-spatial-analysis`.
Add the test to the same recording graph first, then add the returned command-node producer.
Replay requires whole-CSR weights: exactly one chunk covering all test rows. Multi-chunk
weights are rejected when constructing the replay, even after the test has been lowered;
repeating the lowering step cannot enable that unsupported form. A supported whole-CSR test
that has not been lowered instead reports that the test must be lowered first.
The helper borrows the test's prepared inputs and replays its full permutation budget, including
when sequential testing stopped early. Use `output.permutationCounts[row]` from the test to select
the prefix used by its decision. A zero replay-info count means the row is unavailable or invalid;
suppress its distribution. Captures use `gpu-replay` provenance and name the pinned row.
The helper snapshots the test, row and both output view references before validating them;
later edits to the caller's props or nested output map cannot replace a validated binding.
Upload new values to the captured row view to change the pin. Row-domain metadata belongs to
the capture; the producer has no `keySpace` option. The scenario reads `replayInfo` alongside
`nullDistribution`, omits a zero-count replay and publishes exactly the reported prefix (for
example five values when five draws were requested into a 99-value allocation). Missing or
out-of-capacity replay counts are rejected before sidecar publication.

`GPUGlobalPermutationTest`: Scalable rows are bounded by 4,294,963,199 and actual device bindings.
Additive uint32 count outputs preserve exact populations; float32 statistics remain rounded.

`GPULocalPermutationTest`: Scalable rows are bounded by 4,294,963,199 and actual device bindings.
Additive uint32 count outputs preserve exact populations; float32 statistics remain rounded.

`GPULocalPermutationTest`: Test each tract’s local Moran I against 99 conditional permutations, returning pseudo p-values, adjusted values and a significance mask.

::lens{scenario=gpu-spatial-analysis.local-permutation-test rung=result caption="Two-sided pseudo p-values of 791 tracts from 99 permutations; 303 are at most 0.05, none after adjustment."}

`GPUGlobalPermutationTest`: Test global Moran’s I against 99 relabelings of the tracts, returning the pseudo p-value and the simulated reference distribution.

::lens{scenario=gpu-spatial-analysis.global-permutation-test rung=result caption="Observed Moran I = 0.60 against 99 simulated values near 0; pseudo p = 0.02, the smallest possible."}

Both contributors support at most `MAXIMUM_PERMUTATION_COUNT = 1,048,576` requested draws per
submission, subject to tighter device limits. Global inference additionally requires
`(maximumPermutations+1) * sum(ceil(chunkRows/2048)) <= 16,777,215`; each partial CSR chunk contributes
a row block. At 2^20 draws only 15 blocks fit. The general packer's representable integer range
does not enlarge this contributor budget. A 50,000-row Bonferroni family with doubled tails at
.05 needs 1,999,999 draws; at the supported ceiling at most 26,214 family rows fit that floor.

```ts
import {getPermutationRejectionFloor, MAXIMUM_PERMUTATION_COUNT}
  from '@luma-contributors/analysis/gpu-spatial-analysis';
declare const globalRowBlockCount: number; // Sum ceil(chunkRows/2048), not ceil(totalRows/2048).
const plan = getPermutationRejectionFloor({
  permutationCount: 999, rowCount: 50000, alpha: .05, method: 'bonferroni',
  alternative: 'two-sided', pValueConstruction: 'doubled-minimum-tail',
  contributorBudget: {maximumPermutations: MAXIMUM_PERMUTATION_COUNT, globalRowBlockCount}
});
console.log(plan.requiredPermutations, plan.maximumSupportedPermutations, plan.attainableWithinBudget);
```

Permutation inference (pseudo p-values) over a `GPUSpatialWeights` CSR, following PySAL esda's
`p_sim` with the corrections below. Both contributors read a uint32 parameter view written with
`getGPUPermutationParameterValues` from a `GPUPermutationSettings` object
`{seed, permutationCount, significanceLevel}`. Changing the seed, the permutation count (up to the
compile-time `maximumPermutations`), the values or the weights never rebuilds the graph. `values`
and `secondValues` are `float32` or `uint32` (converted to f32, exact below 2^24), and every result
view lives in the `output` bag (`GPULocalPermutationTestOutput`, `GPUGlobalPermutationTestOutput`).

- `GPULocalPermutationTest` does conditional randomization for `localMoran`, `localGeary`, `localG`,
  `localGStar` (a self weight of 1), `localBivariateMoran`, `localLeeL`, `localJoinCount` and
  `localBivariateJoinCount` (see **Bivariate, join-count and Lee statistics** below). For each tested row and permutation it draws an ordered sample
  of `k_i` distinct other analysis rows (a partial Fisher-Yates over their compacted positions) and
  assigns them to the row's weight slots. It writes `exceedances`, `pseudoPValues`, an optional
  `observed` statistic, an optional `adjustedPValues`, and a `significant` mask. Rows with more than
  `maximumNeighbors` (64 at most) neighbors are not tested and set `output.status.overflow`.
  **Negative data.** `localG` and `localGStar` are defined for non-negative data: the ranking
  drops the denominator as a positive constant, which fails for negative values (with
  `alternative: 'greater'` the p-value would be the lower tail). Values are only known on the GPU, so
  when any analysis row is negative no row is tested: `exceedances` is
  `GPU_LOCAL_PERMUTATION_NOT_TESTED`, `pseudoPValues`, `adjustedPValues` and `observed` are NaN,
  `significant` is 0, `output.status.overflow` is 1 and the optional `output.status.invalidCount`
  (one uint32, 0 for every other statistic) holds the number of negative analysis rows. Valid
  non-negative data is unchanged. Zero is allowed.
- `GPUGlobalPermutationTest` handles `moran`, `geary`, `getisOrdG`, `bivariateMoran` (`y`
  permuted) and `leesL` (Lee's spatial Pearson, `x` and `y` permuted together).
  `output.summary` (`GPU_GLOBAL_PERMUTATION_SUMMARY`) holds the observed statistic, `p_sim`, the
  exceedance count, the permutation count used, the simulated mean and standard deviation, `z_sim`,
  `p_z_sim`, the minimum and maximum, and the island and analysis row counts (`islandCount`,
  `analysisCount`). Optional outputs are the `referenceDistribution` and its `histogram` (GeoDa's
  permutation plot). **Negative data.** `getisOrdG` is defined for non-negative data, like the
  analytical G of `GPUGlobalSpatialStatistics` (which also makes its G block NaN when any analysis
  value is negative). When any analysis row is negative the observed statistic, every p-value,
  `zSimulated`, the simulated summaries and the `referenceDistribution` are NaN, the `histogram` is
  all zero, and the optional `output.status.invalidCount` (one uint32, 0 for every other statistic)
  holds the number of negative analysis rows.

`pZSimulated` is the one-tailed normal probability of `abs(zSimulated)`, floored at
`1.17549435e-38`: `max(.5 * getTwoSidedPValue(zSimulated), 1.17549435e-38)`. At |z|=2 the
unfloored mathematical value is about .02275013. This convention is independent of permutation
`alternative`; `pseudoPValue` instead uses inclusive permutation counts, the chosen tail,
finite-draw correction and any stopping rule.

**Sequential stopping (`stopAfterExceedances`, opt-in).**

`getInferenceMetadata(settings)` describes the requested budget. Its `requestedPermutationCount`
names that budget. Decode actual readback with the matching helper. Fixed runs must record the requested count. Besag–Clifford
runs record L at the first inclusive-tail exceedance count h, or P when never stopped; reaching
h exactly at P uses the stopped formula. Decoders reject counts inconsistent with the metadata.

```ts
import {
  GPUGlobalPermutationTest, GPULocalPermutationTest, type GPUPermutationSettings,
  decodeGPUGlobalPermutationTestInferenceMetadata,
  decodeGPULocalPermutationTestInferenceMetadata
} from '@luma-contributors/analysis/gpu-spatial-analysis';
declare const globalTest: GPUGlobalPermutationTest;
declare const localTest: GPULocalPermutationTest;
declare const settings: GPUPermutationSettings; // Actual settings uploaded for this submission.
declare const globalSummaryReadback: Uint8Array;
declare const localCountsReadback: Uint8Array;
declare const rowCount: number; // Source observation rows; counts are in this row domain.
const globalInference = decodeGPUGlobalPermutationTestInferenceMetadata(
  globalTest.getInferenceMetadata(settings), globalSummaryReadback
);
const localInference = decodeGPULocalPermutationTestInferenceMetadata(
  localTest.getInferenceMetadata(settings), localCountsReadback, rowCount
);
console.log(globalInference.actualPermutationCount, globalInference.stoppingRule);
console.log(localInference.actualPermutationCounts, localInference.stoppingRule);
```

Besag and Clifford (1991, "Sequential Monte
Carlo p-values", Biometrika 78(2)): set the compile-time integer `h` in `[1, maximumPermutations]`
and each test consumes its permutations in index order, stopping at the first `L` whose exceedance
count (the count the p-value uses for the chosen `alternative`: `g`, `l` or `min(g, l)`) equals `h`.
Then `p = min(1, m h / L)` (`m` is 2 for `'two-sided'`, else 1), `exceedances` is `h` and `L` is
reported: `GPULocalPermutationTest` writes it per row to the optional `output.permutationCounts`
(uint32; `P` for a row that never reached `h`, `GPU_LOCAL_PERMUTATION_NOT_TESTED` for an untested
row; the output is supported on fixed runs too: each tested row records the requested fixed count,
and each untested row retains `GPU_LOCAL_PERMUTATION_NOT_TESTED`), and `GPUGlobalPermutationTest` writes it to the
`permutationCount` field of `output.summary`. A row or test that never reaches `h` keeps the
standard `min(m (M + 1), P + 1) / (P + 1)`, so with `h > P` the results equal the unstopped run bit
for bit; reaching `h` exactly at `L = P` counts as a stop (`p = m h / P`). The Philox stream of
permutation `p` is unchanged, so the first `L` permutations are those of the full run and `L` does not
depend on the dispatch shape. The `'folded'` alternative is rejected (its statistic needs the mean
of the whole run). Omitting the prop keeps today's behavior and kernels unchanged.

- Local: each row loops over permutations in its own invocation and `break`s at `L`, so a clearly
  non-extreme row costs about `h / (tail probability)` permutations instead of `P`. `significant`,
  `adjustedPValues` and the Geary `classes` use the stopped p-value.
- Global: the permutations run in fixed batches of `permutationBatchSize` when provided, otherwise
  `getGPUGlobalPermutationBatchSize(rows, maximumPermutations)` (about 128 row-block workgroups per
  batch so the GPU stays occupied, at most 64 batches). The batch is also the most permutations
  computed past `L`. `permutationBatchSize` materializes each permuted variable once per row and
  permutation before the pair kernel gathers its values. It accepts an integer in
  `[1, maximumPermutations + 1]`, at most `2^25` elements per materialized variable and at most
  4096 unrolled batches; `estimatedCommandNodeCount` also enforces the shared command node budget.
  With or without early stopping, the shuffle sequence and sum order are unchanged, so the results
  are bit-identical to the inline shuffle for the same seed. The observed statistic runs separately
  before the first sequential batch. After each batch a one-workgroup
  node scans that batch's tail flags in permutation order, keeps the running counts in a GPU state
  word and zeroes the indirect dispatches of the remaining batches, so the stopping index does not
  depend on the batch size. `referenceDistribution`, `histogram`, `simulatedMean`,
  `simulatedStandardDeviation`, `zSimulated`, `minimum` and `maximum` use the first `L` permutations
  only (`referenceDistribution` is NaN past `L`).
- Typical use is `h` between 10 and 20 with large `P` (a LISA map with `P = 999`, `h = 10`): rows
  with `p` far above 0.05 stop after tens of permutations while the significant rows run to `P`.

**Tail (`alternative`).** The type is the shared `GPUInferenceAlternative` and the default is now `'two-sided'`: with `g = #{sim >= observed}`
and `l = #{sim <= observed}`, `p = min(2 (min(g, l) + 1), P + 1) / (P + 1)`. A row (or a global
statistic) whose simulations all tie the observed value has `g = l = P` and `p = 1`. `'directed'`
(`p = (min(g, l) + 1) / (P + 1)`) is esda's legacy default and is anti-conservative: the tail is
chosen from the data and not doubled, so under the null `p` is about twice too small. Its tie
handling deviates from esda on purpose: esda uses `min(g, P - g)`, and `P - g` counts only
simulations strictly below the observed value, so when simulations tie the observed value (discrete
data, zero-inflated counts, a constant neighborhood) esda reports the smallest attainable
`p = 1/(P+1)` (for 20 equal simulations: 0.0476 for a tied observed value), while `min(g, l)` reports
1. The two agree whenever no simulation equals the observed value (pinned against esda 2.10.0).
`'folded'` matches esda's undirected test exactly; `'greater'` and `'lesser'`
match esda's one-sided tests. Do not combine `'directed'` with `multipleTesting`.

**Neighbors, islands and re-standardization.** One rule, shared with the other Stream A statistics:
a neighbor slot is *active* when its neighbor is a valid row other than the row itself, its weight is
positive, and the neighbor is an included row (mask nonzero, finite value). Zero-weight slots are not
neighbors: they are not counted in `k_i` and are never drawn. An *island* is an included row with no
active slot. `islands: 'exclude'` (default) drops islands from the analysis: NaN outputs, and they
are not counted in `n`, the mean, the variance or `S0`, and not neighbors of other rows. There is no
cascade: a row whose only neighbors were islands stays in `n` with no neighbor, so it is not tested
(global tests add 0 to `S0` for it). `islands: 'keep'` reproduces esda: islands stay in `n` and the
moments, their local statistic is the formula value with a zero lag (`observed`, for example local
I = 0), but they are never tested (`exceedances` is `GPU_LOCAL_PERMUTATION_NOT_TESTED`, p-value NaN).
Parity specs against esda use `islands: 'keep'`. Island counts are reported in the optional
one-row `output.islandCount` (local) or `summary[GPU_GLOBAL_PERMUTATION_SUMMARY.islandCount]` (global).
When `weights.information.transform` is `'row'`, each analysis row's active weights are re-standardized to
sum to 1, in the lags and in `S0`, so a masked or missing neighbor drops out instead of shrinking
the row sum below 1 (numerically identical when nothing is dropped); other transforms are used as
given. Root or child nonzero `weights.status.overflow` / `candidateOverflow`, or a supplied zero
`weights.status.complete`, makes the weights incomplete: no row is analyzed, statistics/p-values are NaN,
analysisCount is zero, and local `output.status.overflow` is 1. A complete certificate of 1 permits
inference again on the next submission. Final and candidate cause outputs retain their separate meanings;
an incomplete certificate alone sets neither capacity-loss cause.

Every supplied `overflow`, `candidateOverflow` or `complete` view, at outer matrix scope or
inside an individual CSR chunk, must be packed uint32, have byte offset divisible by four,
and integer length in `[1,0xffffffff]`. The first logical word carries the flag; spare words
are accepted. Nonzero `overflow`/`candidateOverflow` or zero `complete` suppresses analysis.
Current local and global inference consume positive completeness certificates directly at both
scopes; `complete` is not ignored and does not require a redundant failure-polarity alias.
All supplied conditions are combined: a true complete certificate cannot override a loss flag.
Nonzero offsets are allowed when the full device-aligned binding range, including its alignment
prefix, fits both the backing allocation and device storage-binding/allocation limits.

```ts
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
declare const graph: GPUCommandGraph<void>;
declare const statusBuffer: Parameters<GPUCommandGraph<void>['createDataView']>[0];
// statusBuffer has at least 20 bytes and a fitting aligned device binding range.
const candidateOverflow = graph.createDataView(statusBuffer, {
  format: 'uint32', byteOffset: 4, length: 4
});
// The logical flag is byte 4; the remaining three uint32 words are spare capacity.
```

Optional output causes `weightsOverflow`, `candidateOverflow`, `invalidInput`,
and local `neighborOverflow` distinguish failures; aggregate local `overflow` remains available.
Negative-data `invalidCount` is separate. Causes reset each submission; an untested row is not
evidence of nonsignificance. The mean is a fixed-order tree sum followed by a
second-pass correction of the deviations, and `n` is an exact uint32, so large-offset data keep
`sum(z) = 0` to f32 accuracy.

Local permutation construction rejects an estimated command graph above 32,768 nodes before
scratch allocation. Its conservative bound is `256 + 3*C + T + 2*L + F + N`. C is the CSR chunk
count (one for whole weights), T the sum of row-tile counts, and L the polarity-aware root/chunk
loss-check count: distinct negative `overflow`/`candidateOverflow` views count once together,
and distinct positive `complete` views count separately. The same view with both polarities counts twice.
F is the number of requested scalar cause-fold passes: for `weightsOverflow`,
`max(1,ceil(distinctOverflowViews/7))`; for `candidateOverflow`,
`max(1,ceil(distinctCandidateOverflowViews/7))`; plus one if `invalidInput` is requested.
Absent cause outputs add zero. N is `C+1` if `neighborOverflow` is requested, otherwise zero.
Each polarity-aware loss check contributes preparation and aggregate-status work; shared storage
does not collapse opposite meanings.

**Permutation count.** The count is never clamped. A per-frame `permutationCount` outside
`[1, maximumPermutations]` writes NaN p-values: the local test leaves every row untested and sets
`output.status.overflow`; the global test writes NaN for the `permutationCount` field of the summary,
the p-values and the simulated summaries (the observed statistic is still reported). Validate on the
CPU with `getGPUPermutationParameterValues(settings, {maximumPermutations})` or the contributor's
`getParameterValues(settings)`, both of which throw.

**Multiple testing (local).** At 999 permutations the smallest two-sided p-value is 0.002, so a large
family may be unable to reject after correction: check `getPermutationRejectionFloor` (see
`GPUMultipleTestingAdjustment`). `multipleTesting` is `'none'` (default), `'bonferroni'`, `'holm'`,
`'benjamini-hochberg'` or `'benjamini-yekutieli'`, run through `GPUMultipleTestingAdjustment` over the
pseudo p-values of all tested rows (never a display mask). `output.adjustedPValues` receives the
adjusted values, and `output.significant` is `adjustedPValue <= level`; with `'none'` it is the exact
integer-ratio comparison against the exact dyadic value of the stored float32 level. Requesting `output.adjustedPValues` with
`multipleTesting: 'none'` copies the p-values and preserves that integer decision. At a float32
boundary the copied p-value can equal `level` even when the integer comparison rejects
significance: for example, correctly rounded `13 / 22` is `0.5909090638160706`, while
the exact product `level * 22` is less than 13. At `level = float32(9/22)`, the exact product
is 8.999999940395355, so 9/22 is also rejected even though a float32 product rounds to 9.
Shader division may instead produce the neighboring
float `0.5909091234207153` through reciprocal multiplication. The numeric p-value columns
have float32 accuracy; the uncorrected significance decision uses the counts directly and
does not depend on which quotient rounding the device produces.
Zero or nonfinite normalization denominators make a local statistic unavailable before testing
and replay: observed/p-values are NaN, exceedances are NOT_TESTED, significant is zero and replay
count is zero. This covers constant centered statistics and zero-total local G/G*.
Validation and observation reuse the same stored denominator. Bivariate denominators split each
sum of squares into mantissa and integer exponent with `frexp`. Their scaled mantissa product
stays in `[0.25, 2)`; its square root is rescaled with `ldexp`, handling odd exponents of either
sign. This prevents intermediate overflow/underflow even if the compiler regroups floating
arithmetic; simply writing separate square roots does not provide that guarantee.
For local G a dominant focal value uses a direct sum of the other values with integer-recovered
addition-error compensation; subtracting it from the rounded total or repeatedly adding small terms
to an uncompensated large sum can erase the complement. On a unit-weight ten-row cycle with values
`[2^25, 2^24, 1, 1, 1, 1, 1, 1, 1, 1]`, row zero's denominator is 16777224 and its observed G is
0.9999995231628418. Draw/replay comparisons retain their public raw-term units, with the same
validated observation and positive per-row normalization.
Undefined global reference members are NaN and do not enter histograms; an empty finite reference
has NaN minimum and maximum, and incomplete references cannot publish inference.
General G needs at least two strictly positive analyzed values; the contributor counts them
before interpreting its scale, so cancellation in the moment identity cannot turn a zero
pair-product denominator into a tested statistic. With no permutations the CPU twin preserves a
defined observation but reports NaN inference and an empty reference.
`getInferenceMetadata(settings)` on either contributor (the same `GPUPermutationSettings` that are
packed, validated identically)
returns a `GPUInferenceMetadata` (alternative, permutation count, seed, multiple-testing method,
significance level decoded from the same packed words and a `testedRowsDefinition` sentence) for
legends. Settings getters are captured once; metadata cannot reread a different seed, count or level.

**Bivariate, join-count and Lee statistics.** Definitions below use `c = x - mean` over the analysis
rows, `SS = sum c^2`, `n` analysis rows and the weights as given (re-standardized for `'row'`
weights). Parity: statistics against esda 2.10 at f32 accuracy, p-values against the twin of the
Philox draws, plus a Monte Carlo bound against esda `p_sim` (R = 9999).

- `localBivariateMoran` (esda `Moran_Local_BV`, needs `secondValues`): `I_i = (n - 1) c^x_i sum_j
  w_ij c^y_j / sqrt(SS_x SS_y)`; the focus keeps `x`, the drawn neighbors supply `y`. The optional
  `quadrants` output is esda's `q` in the PySAL scheme (1 high-high, 2 low-high, 3 low-low, 4
  high-low in `x_i` and the lag of `y`); it is not gated by significance.
- `localLeeL` (esda `Spatial_Pearson_Local`, needs `secondValues`): `L_i = n (sum_j w_ij c^x_j)(sum_j
  w_ij c^y_j) / sqrt(SS_x SS_y)`. A drawn row supplies both its `x` and its `y` (the pair stays
  together), the focus value does not enter. For row-standardized weights the mean of the local
  values is the global `L`. esda's own p-value is the `'directed'` one (anti-conservative); the
  default here is `'two-sided'`.
- `leesL` in `GPUGlobalPermutationTest` (esda `Spatial_Pearson`, element `[0, 1]`): `L = z'W'Wz /
  1'W'W1 = n sum_i (sum_j w_ij c^x_j)(sum_j w_ij c^y_j) / (sqrt(SS_x SS_y) sum_i (sum_j w_ij)^2)`,
  with the rows of `(x, y)` permuted together by the swap-or-not shuffle (esda permutes the rows of
  `Z`). Note that esda's global version does not row-standardize `W`; here `'row'` weights are
  re-standardized over active slots, others are used as given. The smoothing factors esda reports
  at `[0, 0]` and `[1, 1]` are not exposed. (`GPUSpatialPearson` is a different statistic: the
  weighted correlation across edges.)
- `localJoinCount` (esda `Join_Counts_Local`, binary `values`): `LJC_i = x_i sum_j x_j` over the active
  neighbors with binary weights (the weight values are ignored), where a value is a join only when
  it is exactly 1. Only rows with `LJC_i > 0` are tested; the others get NaN p-values and
  `GPU_LOCAL_PERMUTATION_NOT_TESTED`, as esda sets their `p_sim` to NaN. The tail that detects
  clustering is `'greater'`. With ties (discrete statistics) esda's `'two-sided'` p-value
  (percentile based) is not `2 min(g, l)`; the parity specs therefore use `'greater'`.
- `localBivariateJoinCount` (esda `Join_Counts_Local_BV`, `joinCase`): `'CLC'` (co-location) counts
  neighbors with `x_j = y_j = 1` for a focus with `x_i = y_i = 1`; `'BJC'` (no co-location) counts
  neighbors with `x_j = 0, y_j = 1` for a focus with `x_i = 1, y_i = 0`. A drawn row supplies its `x`
  and `y` together. **Deviation from esda for `'BJC'`:** esda observes the count above but
  simulates `x_i sum_j y_j`, which ignores `y_i = 0` and `x_j = 0`, so its p-values do not test the
  statistic it reports (on the specs' fixture they differ by up to 0.09 at R = 9999). This
  implementation simulates the observed statistic; `'CLC'` is identical to esda.
- `localGeary` classification (`classes` output, `GPU_LOCAL_GEARY_CLASS`; Anselin 2019, esda
  `Geary_Local(labels=True)`): with `m` the mean of the local Geary values over the analysis rows, a
  significant row (`p <= level`, or `adjustedPValue <= level` with `multipleTesting`) is 1
  (high-high, positive association: `c_i < m` and `x_i` above the mean), 2 (low-low: `c_i < m`, `x_i`
  below the mean) or 3 (negative association: `c_i > m`); an insignificant row is 4 and a row that
  is not tested (NaN p-value), or exactly at a mean, is 0 (esda: NaN). esda's own docstring calls
  codes 1 and 2 "outlier" and "cluster"; the numeric rule is ported, not the words. Small `c_i` means
  similar neighbors, so the tail of clustering is `'lesser'` (also the tail declared by
  `GPU_LOCAL_GEARY_PERMUTATION_ADAPTER`). The p-value compares a row with its conditional reference
  distribution; the association label compares its local Geary value with the mean across rows.
  A one-sided alternative does not guarantee an empty association class. For example, the
  independent eight-row audit fixture gives a high-high class with `'greater'` at level 0.5.
  The classification is not available analytically.

**Global shuffle.** Each permutation relabels the analysis values through a keyed swap-or-not
shuffle (Hoang, Morris and Rogaway 2012, `6 ceil(log2 n) + 6` rounds, round keys and salts from
Philox 4x32-10 on the seed and the permutation index), so no sort runs per permutation and nothing
of size `P x n` is stored. The earlier four-round Feistel network is no longer used by the global
test because it is not uniform on small domains: with 400000 draws the marginal of `pi(0)` and the
joint placement of fixed pairs reject uniformity far below the 0.1 percent level for domains of
10 to 50 elements (joint chi-square p-values below 1e-100; marginal p-values from 4e-4 at `n = 50`
down to 1e-139 at `n = 10`), while swap-or-not passes (all 150 chi-square p-values above 0.0023,
none below 1e-3). At 20000 draws on rook lattices and random kNN graphs with `n` = 30 to 5000
both shuffles reproduce the analytic randomization mean and variance of Moran's I (`E = -1/(n-1)`,
esda `VI_rand`) to within about 2.7 standard errors, so the Feistel bias is a small-domain,
joint-placement effect that the variance check alone does not see; it is removed rather than
tolerated. The Knox and Mantel tests and the GWR non-stationarity test use the same swap-or-not
shuffle.

```ts
import type {Device} from '@luma.gl/core';
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
import {
  GPULocalPermutationTest, type GPULocalPermutationTestProps, GPU_PERMUTATION_PARAMETER_LENGTH, getGPUPermutationParameterValues
} from '@luma-contributors/analysis/gpu-spatial-analysis';
declare const graph: GPUCommandGraph<void>;
declare const device: Device;
declare const weights: NonNullable<NonNullable<GPULocalPermutationTestProps>['weights']>;
declare const values: NonNullable<NonNullable<GPULocalPermutationTestProps>['values']>;
declare const exceedances: NonNullable<NonNullable<NonNullable<GPULocalPermutationTestProps>['output']>['exceedances']>;
declare const pseudoPValues: NonNullable<NonNullable<NonNullable<GPULocalPermutationTestProps>['output']>['pseudoPValues']>;
declare const significant: NonNullable<NonNullable<NonNullable<GPULocalPermutationTestProps>['output']>['significant']>;
declare const adjustedPValues: NonNullable<NonNullable<NonNullable<GPULocalPermutationTestProps>['output']>['adjustedPValues']>;
declare const islandCount: NonNullable<NonNullable<NonNullable<GPULocalPermutationTestProps>['output']>['islandCount']>;
declare const overflow: NonNullable<NonNullable<NonNullable<NonNullable<GPULocalPermutationTestProps>['output']>['status']>['overflow']>;

const parameters = new GPUParameterBuffer(device, {
  id: 'permutations', format: 'uint32', length: GPU_PERMUTATION_PARAMETER_LENGTH,
  values: getGPUPermutationParameterValues(
    {seed: 1, permutationCount: 99},
    {maximumPermutations: 999}
  )
});
const test = new GPULocalPermutationTest({
  weights, values, statistic: 'localMoran', parameters: parameters.importToGraph(graph),
  maximumPermutations: 999, multipleTesting: 'benjamini-yekutieli', islands: 'exclude',
  output: {
    exceedances, pseudoPValues, significant, adjustedPValues, islandCount, status: {overflow}
  }
});
graph.add(test);
// Preview with 99 permutations, then refine on idle:
const settings = {seed: 1, permutationCount: 999};
parameters.write(test.getParameterValues(settings));
const metadata = test.getInferenceMetadata(settings);
```

#### How it computes

{/* lens:GPULocalPermutationTest */}
{/* lens:GPUGlobalPermutationTest */}

`GPULocalPermutationTest`:

::lens{scenario=gpu-spatial-analysis.local-permutation-test rung=datum bind=page datum=420 view=fdr-staircase caption="Inspect tract 420: its observed I, exceedance count, pseudo p-value and 99-draw null distribution."}
::lens{scenario=gpu-spatial-analysis.local-permutation-test rung=formula caption="Follow the pseudo p-value: doubled smaller tail count plus one, over permutations plus one."}
::lens{scenario=gpu-spatial-analysis.local-permutation-test rung=kernel caption="Inspect the per-tract Philox draw loop and tail counting with retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.local-permutation-test rung=graph caption="Trace weights, moments, 99 permutation draws, classification and Benjamini-Hochberg adjustment."}

`GPUGlobalPermutationTest`:

::lens{scenario=gpu-spatial-analysis.global-permutation-test rung=datum bind=page datum=420 caption="Inspect tract 420: its poverty value, one of 791 relabeled across 99 permutations."}
::lens{scenario=gpu-spatial-analysis.global-permutation-test rung=formula caption="Follow the pseudo p-value and the simulated z-score from the 99 pair sums."}
::lens{scenario=gpu-spatial-analysis.global-permutation-test rung=kernel caption="Inspect the pair-sum, shuffle and finalize kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.global-permutation-test rung=graph caption="Trace weights, 100 pair sums, reduction over row blocks and the finalize summary."}

{/* parameters: GPUGlobalPermutationTest generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUGlobalPermutationTest` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPUGlobalPermutationTest`.

- Packer: `getGPUPermutationParameterValues(settings, options?)`, writes `GPU_PERMUTATION_PARAMETER_LENGTH` (4) `uint32` values.
  - Input port: `parameters`.
  - `options` (`{maximumPermutations?: number}`)
- Packer: `instance.getParameterValues(settings)`, writes the `uint32` parameter view.

Settings of `getGPUPermutationParameterValues`:

| Field | Type | Description |
|---|---|---|
| `seed` | `number` | Random seed, an integer in `[0, 2^53)`. Results are a pure function of the seed, the inputs and `permutationCount`, independent of the dispatch shape. Per frame. |
| `permutationCount` | `number` | Number of permutations `P`, in `[1, maximumPermutations]` of the contributor. The count is never clamped: a value outside that range makes the kernels write NaN p-values (local: rows are not tested and `output.status.overflow` is set; global: the `permutationCount` field of `output.summary` is NaN). Validate on the CPU with the contributor's `getParameterValues` or the `maximumPermutations` option of `getGPUPermutationParameterValues`. |
| `significanceLevel` (optional) | `number` | Significance level for the local mask, strictly in (0, 1) after float32 rounding. Defaults to 0.05. Ignored by the global test. |

{/* /parameters */}

{/* parameters: GPULocalPermutationTest generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPULocalPermutationTest` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPULocalPermutationTest`.

- Packer: `getGPUPermutationParameterValues(settings, options?)`, writes `GPU_PERMUTATION_PARAMETER_LENGTH` (4) `uint32` values.
  - Input port: `parameters`.
  - `options` (`{maximumPermutations?: number}`)
- Packer: `instance.getParameterValues(settings)`, writes the `uint32` parameter view.

Settings of `getGPUPermutationParameterValues`:

| Field | Type | Description |
|---|---|---|
| `seed` | `number` | Random seed, an integer in `[0, 2^53)`. Results are a pure function of the seed, the inputs and `permutationCount`, independent of the dispatch shape. Per frame. |
| `permutationCount` | `number` | Number of permutations `P`, in `[1, maximumPermutations]` of the contributor. The count is never clamped: a value outside that range makes the kernels write NaN p-values (local: rows are not tested and `output.status.overflow` is set; global: the `permutationCount` field of `output.summary` is NaN). Validate on the CPU with the contributor's `getParameterValues` or the `maximumPermutations` option of `getGPUPermutationParameterValues`. |
| `significanceLevel` (optional) | `number` | Significance level for the local mask, strictly in (0, 1) after float32 rounding. Defaults to 0.05. Ignored by the global test. |

{/* /parameters */}

### `GPUSpatialPearson` and `GPUGammaStatistic`

`GPUSpatialPearson`: Scalable rows are bounded by 4,294,963,199 and actual device bindings. Additive
uint32 count outputs preserve exact populations; float32 statistics remain rounded.

`GPUGammaStatistic`: Scalable rows are bounded by 4,294,963,199 and actual device bindings. Additive
uint32 count outputs preserve exact populations; float32 statistics remain rounded.

`GPUSpatialPearson`: Correlate a tract variable with a second variable at its neighbours, returning one edge-weighted Pearson coefficient.

::lens{scenario=gpu-spatial-analysis.spatial-pearson rung=result caption="Edge-weighted Pearson r = 0.30 between poverty and neighbouring uninsured rates over 791 tracts."}

`GPUGammaStatistic`: Reduce neighbour pairs to one Gamma value: the weighted mean cross-product, squared difference or absolute difference.

::lens{scenario=gpu-spatial-analysis.gamma-statistic rung=result caption="Weighted mean squared poverty difference across the directed queen pairs of 791 tracts is 214 points squared."}

Both statistics sum over directed edges with positive weight to another included row, in three
fixed-order stages. Neither reports an analytic p-value; their permutation adapters and the other
details are described under Additional spatial statistics below.

#### How it computes

{/* lens:GPUSpatialPearson */}
{/* lens:GPUGammaStatistic */}

`GPUSpatialPearson`:

::lens{scenario=gpu-spatial-analysis.spatial-pearson rung=datum bind=page datum=420 caption="Inspect tract 420: its poverty value is one of 791 foci whose neighbours supply y."}
::lens{scenario=gpu-spatial-analysis.spatial-pearson rung=formula caption="Follow the edge-weighted correlation: covariance over the product of weighted standard deviations."}
::lens{scenario=gpu-spatial-analysis.spatial-pearson rung=kernel caption="Inspect the moments, centered and statistic kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.spatial-pearson rung=graph caption="Trace queen weights, row standardisation and the three-stage edge reduction."}

`GPUGammaStatistic`:

::lens{scenario=gpu-spatial-analysis.gamma-statistic rung=datum bind=page datum=420 caption="Inspect tract 420: its poverty value enters every directed edge to its neighbours."}
::lens{scenario=gpu-spatial-analysis.gamma-statistic rung=formula caption="Follow the Gamma ratio: weighted pair terms over total edge weight."}
::lens{scenario=gpu-spatial-analysis.gamma-statistic rung=kernel caption="Inspect the moments, centered and statistic kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.gamma-statistic rung=graph caption="Trace queen weights, row standardisation and the three-stage edge reduction."}

### Additional spatial statistics and bounded optimization

Solver contracts have dedicated sections: [`GPUAZPRegions`](#gpuazpregions),
[`GPUWardRegions`](#gpuwardregions), [`GPUMaxPRegions`](#gpumaxpregions),
[`GPULocationAllocation`](#gpulocationallocation) and
[`GPUBoundedFacilityAllocation`](#gpuboundedfacilityallocation). These links preserve this
section as an entry point for existing solver references; the Rose and edge-statistics contracts
remain below.

Rose rows retain the uint32 ceiling 4,294,967,295. There is no separate 2^24 group cap:
`groupCount * summaryStride` and `groupCount * circular.binCount` must each fit uint32 and
device bindings. Empirical-Bayes contributors allow 4,294,967,039 rows. Scan accepted count totals
are bounded by 4,294,967,294, reserving `0xffffffff` for overflow.

`GPURoseStatistic`: Supports radians/degrees and positive-x counterclockwise or positive-y clockwise
conventions. Row, group and flattened output addressing are bounded before construction.

`GPULocalGeary` writes source-aligned `sum_j w_ij (z_i - z_j)^2` values to `output.statistic`
(`GPULocalGearyOutput`, with the optional `globalStatistics`), with the neighbor, re-standardization and `islands` semantics of
`GPULocalMoran` (an island is NaN under `'exclude'`, the default, and 0 under `'keep'`, esda's
convention; a constant variable gives NaN). It has no analytic classification: use permutation
inference (`GPULocalPermutationTest` with `statistic: 'localGeary'` and `classes` gives the Anselin
2019 positive / negative association classes, see above). `GPU_LOCAL_GEARY_PERMUTATION_ADAPTER` declares the `'lesser'` tail, because small `c_i`
means similar neighbors (a cluster). `GPUSpatialPearson` is the weighted Pearson correlation across
directed spatial edges `(x_i, y_j)`; `secondValues` is required and checked at construction.
`GPUGammaStatistic` reports the weighted mean centered
cross-product, squared difference, or absolute difference over those edges, into the one-row
`output.statistic` (`GPUEdgeAssociationOutput`; `float32` or `uint32` values). Both skip the focal
row's own slot and zero-weight slots, and reduce the edges in three fixed-order stages (per-workgroup
moments, per-workgroup centered sums around the combined means, one combining workgroup), so they use
the whole device and stay deterministic. The local Geary, Pearson, and two Gamma forms expose
`GPUPermutationStatisticAdapter` descriptors that name a statistic of the existing permutation
contributors; they do not duplicate permutation loops. Only two of them test the same statistic:
local Geary, and squared-difference Gamma (Geary's numerator over a constant), provided both use
identical effective weights and donor populations, and selected labels have positive variance.
Constant labels give squared-difference Gamma 0 while Geary is undefined, so the mapping does not
apply to them. Gamma retains supplied weights after masking; the permutation engine re-normalizes
surviving rows tagged `transform: 'row'`. To obtain the exact
mapping, materialize the selected weights, apply any intended normalization once, and pass the same
result tagged `transform: 'raw'` to both calculations with the same explicit mask. If isolated
rows remain in that population, select `islands: 'keep'` in the permutation engine; otherwise mask
them out of both calculations before materializing the weights.
For example, masking the .75 edge in a [.25,.75] row leaves .25 in Gamma but 1 in the engine;
the audit labels give Gamma 85/13 > 72/13 while re-normalized Geary gives 33/35 < 9/7.
The cross-product
Gamma adapter runs the `'moran'` test and the Pearson adapter the `'bivariateMoran'` test. These
order permutations like the edge statistic only on regular weights: every row or every column
with the same total weight for Gamma (for example row-standardized weights), every column with the
same total for Pearson. On other graphs they are tests of Moran's I and of bivariate Moran's I.
The Gamma mappings require a univariate input (`secondValues` omitted or identical to `values`);
with a distinct second variable they do not test the bivariate Gamma edge statistic.
All row/column-total conditions apply to the effective weights after selection. For Pearson with
missing data, supply an explicit joint complete-case mask: the bivariate permutation engine drops
an entire incomplete pair, while the descriptive edge coefficient can use a finite value at
either end independently. The absolute-difference Gamma operation has no matching shared test;
use its descriptive result or a separate absolute-difference randomization test. Unknown operation
names are rejected instead of selecting Moran implicitly.
Example: on the star 0-1, 0-2, 0-3 with the edge 3-4 and binary weights, the values
(7, 2, 4, 1, 11) and (11, 2, 1, 7, 4) have cross-product Gamma -10 and -11.0625 but Moran pair sums
-80 and -64. `GPUGammaStatistic` is not `esda.Gamma` (a plain sum, uncentered by default, not
divided by the edge weight), and `GPUSpatialPearson` is not `esda.Spatial_Pearson` (Lee's L, see
`leesL`).

Both edge contributors return NaN when either outer `weights.status.overflow` or
`weights.status.candidateOverflow` is nonzero. For chunked weights, aggregate these flags on the
outer status port; chunk-local loss flags are rejected at construction. Pearson also handles
overflow or underflow of the product of its two finite sums of squares with bounded significands
and integer exponents for the division by their square roots. This prevents GPU arithmetic
regrouping from restoring the unrepresentable product; individual moment sums must still fit float32.
Each eligible edge marginal's exact range is checked independently of its rounded weighted mean;
if either marginal is constant, its variance and the Pearson coefficient are undefined (NaN).

`GPURoseStatistic` consumes planar line starts and ends. Its required
`GPUCircularStatisticsContract` makes directional versus axial interpretation, radians or degrees, angular
origin, and equal-width bin count explicit. It reports count, mean angle, resultant length,
circular variance, Rayleigh z, and the weighted Rose bins for every group, in
`output: {summary, bins}`. The descriptive mean, resultant length, variance and bins use the
positive finite `weights`; invalid lines or weights are excluded. The descriptive resultant is
clamped to `[0,1]` before publication and circular variance `1 - resultant`, so rounding identical
non-cardinal directions cannot publish a negative variance. Rayleigh z uses **each valid
segment once**, with unweighted resultant $\bar R_0$ and valid segment count $n$:

$$z = n\bar R_0^2 = \frac\{(\sum_i\cos\phi_i)^2 + (\sum_i\sin\phi_i)^2\}\{n\}.$$

Here $\phi_i=\theta_i$ for directional data and $2\theta_i$ for axial data. This corrects the
previous use of total descriptive weight as sample size: metres of street are not independent
observations, and changing their units must not change z. Weighted descriptive `resultantLength`
is distinct from $\bar R_0$. No p-value is returned. Under independent uniform directions the
usual [Rayleigh approximation](https://github.com/circstat/circstat-matlab/blob/master/circ_rtest.m)
is $p\approx e^\{-z\}[1+(2z-z^2)/(4n)-(24z-132z^2+76z^3-9z^4)/(288n^2)]$;
spatially dependent street segments do not establish that null assumption. This planar rose is
separate from [giddy Rose](https://pysal.org/giddy/generated/giddy.directional.Rose.html), which
forms directional LISA vectors and uses permutations of sector counts for inference.

### `GPUAZPRegions`

`GPUAZPRegionsOperation`, `CPUAZPRegionsOperation` and `computeAZPRegionsOnCPU` are its GPU graph, CPU graph and pure CPU forms.

`GPUAZPRegions` is a greedy AZP with batched moves: a bounded, deterministic local search from
caller-supplied contiguous labels with no tabu list, annealing or reactive escape, so it stops at the
first local optimum and the result depends on `initialLabels` (for several starts, run several
instances from different initial partitions and keep the smallest `objective`). Each round every row
finds its best improving move (the largest reduction of the within-region sum of squares from leaving
its region for an adjacent one) and the round applies an independent set of them at once: every region
takes part in at most one selected move, chosen by gain and then by a stable seed-keyed row order
(two atomic-max claim passes over the regions), so the selected gains add up exactly and the objective
falls strictly every round. Each destination must first pass its own relative gain cutoff
(`4e-6` times its four squared-moment terms); the row then ranks only those admissible destinations.
A lower-gain destination remains eligible when a higher-gain destination fails its own cutoff.
The best admissible move of a round is always selected, so a round without a move is
the local optimum. Contiguity is kept by a **local articulation test** instead of a search of the whole
source region: a row may leave region A only when its A-neighbors stay connected to each other through
rows of A within two steps of it (at most 128 rows, scanning neighbor lists of at most 64 entries;
a larger neighborhood or a hub refuses the
move). The test is sufficient, never necessary, so a region never becomes disconnected, but a safe
move whose reconnecting path is longer is refused: the result is a local optimum with respect to the
moves that pass the test, not necessarily to every contiguity-preserving move (on random lattice
partitions the test refuses about 1% of the safe moves). A row only moves to the region of one of its
neighbors, which keeps the destination connected. The initial labels are checked on the GPU as well: a
region that is not connected adds its extra components to `invalidCount` and the search does not run.
The operator publishes `output: {labels, objective, status}`: the final within-region sum of squares
(the same quantity as `withinSumOfSquares` of `GPURegionPartitionEvaluation` on the same table,
recomputed from the returned labels) plus packed iteration, convergence, iteration-limit, and
invalid-count status. `iterationCount` counts the rounds that applied at least one move, not single
moves; `maximumIterations` is a hard compile-time bound on rounds (a round applies many moves, so far
fewer are needed than the single moves of earlier versions). The seed is per frame: the one-word
`parameters` view written with `getGPUAZPRegionsParameterValues({seed})` fixes tie order, so a new seed
is a buffer write. Optional `extensiveValues` and `extensiveThreshold` forbid a move that would leave
the source region below the threshold (the max-p constraint). A multi-start AZP is deferred: one
instance already costs `maximumIterations` rounds of row-parallel work, so parallel starts would
multiply the dispatch count rather than fill idle lanes. AZP accepts 1–8 attribute columns, 1–256
move rounds, and a region capacity of 2 through the input row count. At a 128 MiB storage binding
limit, the attribute and region tables each hold at most 33,554,432 words and the input CSR has at
most 16,777,216 neighbor slots. After the local optimum the one-thread round-finish kernel zeroes a GPU-written indirect
dispatch, so the remaining rounds launch only that kernel.

The AZP CPU twin obeys the same explicit `maximumIterations` bound. Its pure function defaults to
256 rounds when the budget is omitted. A successful batch on the last permitted round reports
`converged = 0` and `iterationLimitReached = 1`; a no-move round must execute to prove convergence.

`GPUAZPRegions`, `GPUMaxPRegions` and `GPUWardRegions` share two input rules. **Attributes are
standardized by default** (`standardize`, default true): each column becomes a z-score over its finite
rows with the population standard deviation (divisor `n`, the `sklearn` `StandardScaler` convention),
a constant column becomes zero, and `objective` is in z-score units. Pass `standardize: false` for the
raw-unit behavior of earlier versions (a default change). The shared standardizer also used by the
minimum spanning forest rescales extreme finite columns by a power of two before centering, so
squaring their deviations does not overflow or underflow float32. A nonconstant column remains
informative after a change of units; float32 input quantization and device subnormal flushing still
apply. Mean terms retain input-subtraction, summation and division residuals in two float32 lanes,
including through the fixed workgroup reduction. Variance and final scores subtract that split
mean directly, preserving genuine small positive and negative residuals. Variance terms use
pairwise lane sums followed by the fixed workgroup tree.
The AZP, Max-P and Ward CPU twins likewise treat only zero population deviation as a
constant column; a small nonzero deviation retains the column's information.
**Adjacency is undirected**: row `i` is
adjacent to `j` when either row lists the other, so directed (for example k-nearest-neighbor) weights
are symmetrized inside the operator with `GPUSpatialWeightsTranspose`; weights whose
`information.symmetric` is `true` skip that pass. All three take a per-frame `uint32` `parameters`
view holding `[seed]` (`getGPUAZPRegionsParameterValues`, `getGPUWardRegionsParameterValues`,
`getGPUMaxPRegionsParameterValues`) and write `output: {labels, objective, status}`.

AZP and Max-P suppress their partitions when either supplied `weights.status.overflow` or
`weights.status.candidateOverflow` is nonzero: labels are `0xffffffff`, the objective is NaN,
and every status word is zero (including `converged`/`feasible`). Their CPU functions accept
the same scalar arrays in `input.status`, and both operation forms accept optional graph value
names in `weights.status`. Loss is unavailable adjacency, not an invalid feature. Clearing the
flags on a later submission restores ordinary computation, including repeated `1 → 0 → 1` changes
on the same compiled graph and flag buffer. Declared props and weights/status/output fields are
captured once, including inherited or non-enumerable getters. The captured maps keep their
construction-time identities while the buffers remain per-frame inputs.

{/* parameters: GPUAZPRegions generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUAZPRegions` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPUAZPRegions`.

- Packer: `getGPUAZPRegionsParameterValues(settings?)`, writes `GPU_AZP_REGIONS_PARAMETER_LENGTH` (1) `uint32` values.

Settings of `getGPUAZPRegionsParameterValues`:

| Field | Type | Description |
|---|---|---|
| `seed` (optional) | `number` | Reproducible tie-order seed, an integer in `[0, 2^32 - 1]`. Defaults to 0. Record the seed used to create `initialLabels` next to it. |

{/* /parameters */}

### `GPUWardRegions`

`GPUWardRegionsOperation`, `CPUWardRegionsOperation` and `computeWardRegionsOnCPU` expose the same merge schedules and round budgets.

`GPUWardRegions` begins with one region per row and repeatedly merges the adjacent pair with the
smallest Ward increase in within-region squared error. `targetRegionCount` and `maximumIterations`
bound the search; disconnected weights can stop it before the target, which is reported through
`output.status` (`GPU_WARD_STATUS_LAYOUT`).

Both schedules accept 1–8 attribute columns and a target of 1 through the input row count. The
iteration budget is at most the input row count minus the target. At a 128 MiB storage binding
limit, the input CSR is capped at 16,777,216 neighbor slots and the region table at 33,554,432 words.

The Ward CPU twin obeys the same explicit round budget, including zero. With zero rounds and valid
input it returns the singleton partition and reports the iteration limit unless the target is already
met. Invalid input sets `invalidCount` and leaves both `converged` and `iterationLimitReached` zero,
including at a zero-round budget, for both merge schedules and operation forms. The pure CPU function
defaults to `rowCount - targetRegionCount` rounds when the budget is omitted.

Ward captures declared options, output maps and adjacency status fields by property read at
construction, including inherited/non-enumerable getters. Later caller-map changes cannot remove
a declared loss flag; buffer contents remain per-frame inputs.
Both Ward operation forms require each supplied `weights.status.overflow` or `candidateOverflow`
value name to be a nonempty string. Omit an unused flag; an empty string is rejected rather than
silently disconnecting loss propagation.
With `standardize: false`, both GPU and CPU Ward keep each region's feature offsets relative to
its own representative row. A merge rebases the absorbed offsets to the retained representative;
costs combine the two representative values and offset means. The objective stays in raw squared
units. A distant first row does not erase differences among other rows, and representable
translations retain the earlier small-difference behavior. The raw GPU region table holds
`rows * (2 * columnCount + 2)` float32 words; the standardized table holds
`rows * (columnCount + 1)` words. Each must fit `getGPUWardMaximumRegionTableWordCount(device)`.

Ward suppresses partition outputs if either `weights.status.overflow` or
`weights.status.candidateOverflow` is nonzero: every label is `0xffffffff`, `objective` is NaN,
and iteration/convergence fields are zero. The flags retain their separate final-output and
candidate-loss meanings; neither contributes to `invalidCount`. Both operation forms accept the
same optional `weights.status` graph values; the pure CPU function accepts corresponding uint32
arrays in `input.status`. Clearing both flags on a later submission permits merging again.

`GPUWardRegions` has a compile-time `mergeSchedule`. The default `'sequential'` applies one merge per
unrolled iteration (always the globally cheapest adjacent pair), so reaching `k` regions from `n` rows
needs `n - k` iterations. `'mutual-nearest'` makes every iteration a round: each region finds its cheapest
adjacent region (Ward increase, ties by the same seed-keyed pair order), and every pair that chose each
other merges at once. Many graphs need far fewer rounds than sequential merges, but a star graph
needs `n - k` rounds because only the center and its chosen leaf are reciprocal. `maximumIterations`
is an upper bound on rounds, not a logarithmic convergence guarantee. When a round would pass
`targetRegionCount`, exact radix selection keeps only the cheapest pairs in the total order. Above
4,096 rows, counting and byte-wise radix histograms run across multiple workgroups. `status[iterationCount]`
still counts merges, `objective` is still the sum of the merges' Ward increases (the within-region sum of
squares of `labels`), and a `status` view of five words receives the number of rounds in word
`GPU_WARD_ROUND_COUNT_STATUS_WORD`. It is opt-in because the result can differ from `'sequential'`. Without
contiguity constraints (complete adjacency) Ward's linkage is reducible, so both schedules build the same
dendrogram and agree whenever the target is a natural cut (every merge below it is cheaper than every
merge above it, as for well separated clusters), which the `scipy.cluster.hierarchy.ward` fixture checks.
At other targets a round can merge a costly reciprocal pair before a cheaper pair that exists only after
other merges of the same round, so the partitions differ; with contiguity constraints Ward can also have
inversions (a merged region can be cheaper to join than its parts), which makes them differ further.
Either result is a valid contiguous partition with an exact objective.

Both schedules contract a GPU-resident graph of unique undirected region pairs after each merge or
round. Internal edges and duplicate boundary pairs are discarded, edge dispatches shrink with the
active edge count, and one Ward cost is shared by both endpoints. Costs are recomputed only when an
endpoint's region moments changed. Merges update parent links; dense row labels are published once
at the end with race-free pointer doubling. These changes retain each schedule's seeded tie order
and moment arithmetic. The sequential schedule still has `n - k` dependent merge decisions, and
contraction adds scratch and dispatch overhead, so small inputs do not imply a speedup.

{/* parameters: GPUWardRegions generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUWardRegions` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPUWardRegions`.

- Packer: `getGPUWardRegionsParameterValues(settings?)`, writes `GPU_WARD_REGIONS_PARAMETER_LENGTH` (1) `uint32` values.

Settings of `getGPUWardRegionsParameterValues`:

| Field | Type | Description |
|---|---|---|
| `seed` (optional) | `number` | Reproducible tie-order seed, an integer in `[0, 2^32 - 1]`. Defaults to 0. |

{/* /parameters */}

### `GPUMaxPRegions`

`GPUMaxPRegionsOperation`, `CPUMaxPRegionsOperation` and `computeMaxPRegionsOnCPU` expose the GPU graph, CPU graph and pure CPU routes.

`GPUMaxPRegions` is a parallel multi-seed Max-P: `startCount` (compile time, default 16, at most 64)
independent constructions share every dispatch. Each start plants many region seeds at once and grows all
regions together in gated rounds, so there is no row cap. A *seed wave* plants a seed at every
unassigned row that is a local minimum of its Philox 4x32-10 key of `(row, start; seed)` (seed read from
`parameters`) among the unassigned rows next to it, is not next to a growing region and is below the
wave's key cut; the first wave cuts to about `0.4 / s` of the rows, with `s = ceil(threshold /
mean(extensiveValues))` the expected region size, later waves cut 1.3 times as many, and a wave starts
when no region is growing or `ceil(s / 2)` rounds after the previous one. Each growing region then claims
one frontier row per round: rows offer themselves with a key made of their number of unassigned
neighbors (fewest first, which keeps regions compact) and their squared distance to the region mean, a
region keeps its smallest key and a row offered by several regions joins the smallest. Region sums are
updated by their single owner, so no float atomics are involved. A region that reaches the threshold
stops, and a region with no frontier left under the threshold is marked failed. The rows of failed
regions and rows that no region reached then join the most similar adjacent completed region in
label-propagation rounds; a failed region with no completed neighbor stays and is reported in the
status. The returned start has the fewest under-threshold regions, then the most regions, then the
smallest within-region sum of squares of all grown regions (frozen before enclave rows join,
including regions that are subsequently absorbed in full), then the lowest
start index; the status reports the region count, any infeasible remainder, the `selectedStart` and the
`improvementIterationCount` (successful rounds applying at least one move, rather than reassigned
rows; formerly named `improvementMoveCount`), and `objective` is the exact within-region sum of squares of the returned
labels. `maximumIterations` bounds refinement rounds on both CPU and GPU; a round can move several
independent rows. With `maximumIterations > 0` (default 0) the selected construction is refined by greedy AZP
moves that leave each donor at or above the threshold, which lowers `objective` at the same region
count (the AZP phase reads the same `parameters` view); spopt's `MaxPHeuristic` anneals instead.
After refinement, `feasible` and `infeasibleRegionCount` are recomputed from the returned
memberships; a receiving region can repair a deficit and become feasible.

The original six `GPU_MAX_P_STATUS_LAYOUT` words remain unchanged: `regionCount`, `feasible`,
`infeasibleRegionCount`, `invalidCount`, `selectedStart`, `improvementIterationCount`.
Optional `output.evidence` adds five
`uint32` words (`GPU_MAX_P_EVIDENCE`): `constructionLimitReached`, `constructionSettled`,
`improvementEnabled`, `improvementConverged`, `iterationLimit`. Read them with
`decodeGPUMaxPRegionsEvidence`, which accepts typed words or packed bytes and rejects missing/partial
words. CPU results expose the same evidence. Settled construction is not a proof of global
infeasibility; unfinished growth or possible enclave absorption sets the construction limit word.
Disabled improvement has both improvement convergence and limit words zero, unlike exhausted
improvement. Invalid input or adjacency loss clears all evidence, with recovery after loss clears.
Requesting evidence adds probe/reduction/publication and loss-suppression nodes to the construction
budget; omission preserves the no-evidence admission contract.

Both CPU and GPU limit growth to `min(maximumGrowthSteps, 2 * rowCount + 2)` rounds and enclave
assignment to `min(maximumGrowthSteps, rowCount + 1)` rounds. At the limit they publish the current
partition: unreached rows become singleton regions, unfinished regions contribute to
`infeasibleRegionCount`, and `feasible` is zero while any under-threshold region remains.
An unreached singleton whose own extensive value meets the threshold is feasible, including
when a short seed-wave budget left it unassigned.
After enclave propagation, each unfinished remainder is split only across disconnected
components. A connected remainder stays together even if its seed was absorbed. A retained
component keeps the original ID (the seed's component when the seed remains, otherwise the
component with the smallest row); extra components use their smallest row before dense label
publication. Completed regions keep their memberships. Raising `maximumGrowthSteps` permits
more enclave propagation before this component check.

The rounds are unrolled to `maximumGrowthSteps` (compile time, default 256, at most 4096) and gated on
the GPU, so rounds after the construction has settled launch nothing and every dispatch is
`O(startCount * (rowCount + neighborCount))`. Max-P accepts 1–8 attribute columns, 1–64 starts,
1–4,096 growth steps, and 0–256 improvement rounds. Its seeding estimate is capped at 1,048,576
rows. At a 128 MiB storage binding limit, input CSR capacity is at most 8,388,608 neighbor slots.
A region claims one row per round, so a region needing more
rows than `maximumGrowthSteps + 1` cannot complete during growth (one seed plus at most one added row
per round) and is reported under the threshold unless enclave assignment absorbs it; seed waves are spaced
half an expected region apart, so raise `maximumGrowthSteps` (nine command nodes per round) when the
threshold needs regions of more than about `maximumGrowthSteps / 4` rows. The seven per-start row
buffers use `startCount * rows * (2 * columnCount + 12) * 4` bytes in total; each must fit one
storage binding. This includes the two component planes for unfinished remainders. Standardization,
adjacency, scans and final evaluation/refinement need additional scratch, and the unrolled
construction obeys the command-node budget. Against `spopt`'s
`MaxPHeuristic` (test/oracles/statistics-max-p.py, spopt 0.8.0, lattices and Delaunay graphs of 144 to
900 rows) the construction reaches 0.975 to 1.04 of spopt's best region count with every region at the
threshold and contiguous; its objective before the optional AZP improvement is up to 1.55 times spopt's
annealed objective.

{/* parameters: GPUMaxPRegions generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUMaxPRegions` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPUMaxPRegions`.

- Packer: `getGPUMaxPRegionsParameterValues(settings?)`, writes `GPU_MAX_P_REGIONS_PARAMETER_LENGTH` (1) `uint32` values.

Settings of `getGPUMaxPRegionsParameterValues`:

| Field | Type | Description |
|---|---|---|
| `seed` (optional) | `number` | Reproducible seed of the random seed-row order of every start (and of the improvement phase's tie order), an integer in `[0, 2^32 - 1]`. Defaults to 0. |

{/* /parameters */}

### `GPULocationAllocation`

The direct contributor selects facilities; its named graph and CPU forms below retain the same cost and status contracts.

`GPULocationAllocation` consumes a row-major demand-by-facility cost matrix and implements bounded
greedy p-median, maximum-coverage, and set-covering selection. Pair it with
`GPUNetworkCostMatrix` for network costs, or provide any dense cost matrix (`costs`) or a sparse one
(`sparseCosts`, below). Outputs include selected facility flags, nearest assignments, the objective, uncovered demand count, completion, and invalid
input count. The contributor keeps selection on the GPU, but it does not claim the integer-programming
optimum or application-specific facility constraints of a packaged solver.

Construction captures numeric options, nested output maps, weights and evidence bindings, and the
`sparseCosts.offsets`, `sparseCosts.facilities` and `sparseCosts.costs` views. Reassigning a field such
as `sparseCosts.costs` after construction leaves the original view bound. The captured buffers’
contents remain live per submission. `GPULocationAllocationGraphOperation`,
`CPULocationAllocationOperation` and `computeLocationAllocationOnCPU` provide the named GPU graph,
CPU graph and pure CPU forms. `addNetworkLocationAllocationAdapter` consumes a labelled network
matrix with explicit facility axis and trip direction, forwarding finality and horizon independently.
It transposes a facility-row matrix for the demand-major consumer. Existing `assignments` are
facility row indices; the adapter's facility-node map and optional `assignmentNodes` translate those
rows back to network nodes. Only facility rows require one individual start node per row, without
`startRows`. The adapter accepts grouped demand starts with `facilityAxis: 'columns'`.
For facilities in columns, assignments map through `targetNodes`, or directly to the column index
when no target-node map is supplied. For facilities in rows, assignments map through `startNodes`.

Read the full seven-word status in order: selected facility count, uncovered demand count,
completion, invalid input count, improving swaps applied, swap convergence, swap iteration limit.
Supplied `costsComplete` (one `uint32`) and `costLimit` (one `float32`, in matrix cost units) are
independent evidence: a zero certificate or an insufficient/NaN horizon clears completion.
A supplied coverage horizon must reach `coverageDistance` for maximum coverage and set covering;
unrestricted p-median requires `+Infinity`. The horizon is checked even without `costsComplete`.
Omitting evidence keeps the supplied-matrix assumption; it does not certify an unfinished search.
For true costs `[0, 2]`, a visible matrix `[0, Infinity]` at horizon 1 cannot certify coverage at
cutoff 4, even if the network search converged within its horizon. At horizon 2 the full matrix
covers both unit demands and the numerical objective recovers to 2, but completion remains zero:
the supplied horizon still does not certify every cost up to the requested cutoff. The three
submissions give:

| Matrix horizon | Objective | Complete |
| --- | --- | --- |
| 1 | 1 | 0 |
| 2 | 2 | 0 |
| 4 | 2 | 1 |

This assumes valid input and a nonzero supplied `costsComplete`; the horizon is independent evidence.
Certified input, completed selection, swap convergence and global optimality are distinct claims.
Completion of greedy selection does not certify swap convergence. For example, a one-swap budget
can yield `[2, 0, 1, 0, 1, 0, 1]` while a second improving swap remains; increasing the budget enough
to apply that swap and check its neighbors can yield `[2, 0, 1, 0, 2, 1, 0]`. Dense and sparse costs,
the contributor, and both graph operation forms use the same word meanings.

#### `GPUBoundedFacilityAllocation`

`GPUBoundedFacilityAllocation` has a separate three-word status: unassigned demand rows, valid
input, global-optimum certificate. A valid feasible p-center result gives `[0, 1, 1]`; greedy
capacitated p-median gives `[0, 1, 0]`. One demand row exceeding every available selected capacity
gives `[1, 1, 0]`. A zero certificate in capacitated mode reflects its demand-order greedy rule.
Its separate one-word `contributor.status.complete` certifies
valid enumeration on complete supplied costs. Supplied `costsComplete` must be nonzero and a
supplied `costLimit` must be `+Infinity` for either unbounded objective. Bind a caller-owned
one-row `uint32` certificate destination through `output.ports.status.complete`. Incomplete evidence clears
this word and the p-center optimum word while preserving local diagnostics; clearing loss and
restoring the horizon restores certification on resubmission. Read the original three diagnostic
words and the separate word with `decodeGPUBoundedFacilityAllocationStatus(statusRead, completeRead)`.
It accepts `Uint32Array`, `Uint8Array` or `ArrayBuffer` reads and rejects short/partial layouts;
local valid-input or unassigned counts alone do not certify a complete decision.
Negative demand is invalid: one p-center demand row of -1 gives `[1, 0, 0]`. General allocation's
`invalidCount` counts invalid entries, not just their presence or the number of affected rows.
Two costs of -1 and NaN in one row of a three-demand matrix give full status
`[0, 3, 0, 2, 0, 0, 0]` in both dense and sparse forms, before any selection or swaps run.

Set covering scores a facility by the number of uncovered demand rows it would cover, not by their
weight: as in `spopt`'s `LSCP`, every reachable row must be covered, including a row whose
`demandWeights` entry is 0. `demandWeights` enters the p-median and maximum-coverage objectives only.

A `+Infinity` cost marks an unreachable demand-facility pair. Unreachable demand is never summed
into the p-median objective: the objective is the weighted cost of the reachable rows, the
unreachable rows are counted in `uncoveredDemandCount` (not `invalidCount`, which marks invalid input
and stops the run), their assignment is `0xffffffff` and their `assignedCosts` entry is `+Infinity`.
Greedy and swap steps compare p-median candidates lexicographically, first by the demand weight left
unreachable and then by the reachable cost, so one unreachable row cannot swamp the float32 cost
differences of the others.

After the greedy construction, p-median and maximum coverage run a Teitz-Bart vertex-substitution
(swap) phase, bounded by `maximumSwapIterations` (compile-time, integer 0 to 1000, default 100; 0 is
pure greedy; set covering has no swap phase). Each iteration scores every (selected facility, unselected
candidate) swap in parallel, applies the single best strictly improving swap (ties go to the lowest
selected facility, then the lowest candidate) and stops when none improves. A swap must gain more than
1e-6 of the objective scale, which keeps f32 summation noise from cycling. It needs
`maximumFacilityCount * facilityCount <= 2^24` (scores are kept per pair). The `status` output now has seven words: after the four earlier ones,
`swapIterationCount` (swaps applied), `converged` (1 when no improving swap remains, so the result is a
swap-local optimum) and `iterationLimitReached` (1 when the budget was used up, so local optimality is
not verified); all three are 0 when the phase is off. Behavior change: swaps are on by default, so
p-median and coverage results can differ from (and are never worse than) the old greedy result, and
`output.status` must hold seven words. Greedy maximum coverage alone has the `(1 - 1/e)` guarantee
relative to the optimum (Nemhauser, Wolsey and Fisher), which the swap phase preserves because it never
lowers the covered weight; greedy p-median has no constant-factor guarantee. With certified costs,
a permitted round finding no improvement above the implemented threshold certifies a swap-local
optimum for p-median or maximum coverage. Greedy completion, a disabled swap phase or budget
exhaustion alone does not certify that optimum. For example, facilities covering `{0,1,3,4}`,
`{0,1,2}` and `{3,4,5}` with six unit demands and two selections give greedy coverage 5; one swap
can raise coverage to 6. A swap-local certificate still does not establish the integer-programming
optimum. It is checked against a brute-force optimum on small
instances, and p-median against a randomized weighted fixture with `+Infinity` entries whose exact optimum
comes from the integer program solved by `scipy.optimize.milp` (generator and package versions in
`test/oracles/p-median-oracle.py`).

Swap scoring uses the Whitaker fast-interchange gain decomposition in the form of Resende and Werneck
("A fast swap-based local search procedure for location problems", Annals of Operations Research,
2007). With `d1` and `d2` the cost of a demand row to its nearest and second-nearest selected facility,
`w` its weight and `c(i, j)` the cost of candidate `i` to row `j`, swapping out `r` for `i` changes the
p-median cost by `gain(i) - loss(r) + extra(i, r)`, where `gain(i)` sums `w * max(0, d1 - c(i, j))` over
all rows, `loss(r)` sums `w * (d2 - d1)` over the rows served by `r`, and `extra(i, r)` sums
`w * (d2 - max(c(i, j), d1))` over the rows served by `r` with `c(i, j) < d2`. Unreachable rows keep the
lexicographic rule: a separate reachable-weight term (which `d2 = +Infinity` and a newly reached row
feed) takes precedence over the cost term. Maximum coverage has the same shape with `gain(i)` the weight
of the uncovered rows `i` covers, `loss(r)` the weight covered only by `r`, and `extra(i, r)` the part of
it that `i` also covers. One thread scores all `maximumFacilityCount` swaps of one candidate in a single
ascending pass over that candidate's costs, writing only its own pair entries (no atomics), and a
fixed workgroup tree sums the objective scale (sparse input: `loss(r)` too), so an iteration costs `O(facilityCount *
demandCount)` work (dense) instead of `O(maximumFacilityCount * facilityCount * demandCount)` and the
sums do not depend on scheduling. The earlier per-pair kernels summed per-row differences; the
decomposition sums the same terms grouped differently, so the applied swap is the same except where
float32 summation order changes a near-tie, which the 1e-6 threshold guards (the small integer
fixtures in the specs sum exactly, and the specs require identical swaps). The dense kernel visits every row of a
candidate, so each row served by `r` adds its own net term `w * (min(max(c, d1), d2) - d1)` (and the
reachable weight it loses if neither `d2` nor `c` reaches it) instead of a separate `loss(r)` and
`extra(i, r)`; no large `d2` terms cancel, which a spec with `d2` about `1e7 * d1` checks in float32.
The sparse kernel cannot visit unlisted pairs, so it sums `loss(r)` per selected facility (one
workgroup each) and adds `extra(i, r)`. For a fully reachable solution the comparison cap described
below bounds the individual terms; float32 summation order can still change near-ties.

Every finite nonnegative cost is reachable, including values at or above `1e30`; only `+Infinity`
marks an unreachable pair. This rule applies to greedy scores, swaps and final assignments in both
representations. Assignment ties choose the lowest facility row. A coverage threshold of `+Infinity`
covers all reachable pairs and never covers unreachable pairs.

P-median greedy scores and swap cost totals, gains and tolerance comparisons use a common binary
scale when the complete weighted demand sum risks overflow. The cost exponent uses costs and
weights from the same demand row and excludes zero products, so unrelated maxima on
noncontributing rows do not erase small improvements. For p-median swaps, when every positive-weight
demand row is already reachable, the exponent instead bounds the weighted nearest costs for that
iteration, with headroom for the sparse sum of capped terms. Weighted comparison costs are capped
at twice that scaled objective before differences are formed: a larger cost cannot improve the
current solution. This prevents unused, enormous
fallback costs from erasing a small gain through scaling or sparse loss/extra cancellation. A swap
that can reach new positive-weight demand retains the full input product scale. These bounds affect
comparisons only; assigned costs and the reported objective remain in caller units.
Swap reach weights and maximum-coverage swap gains, losses, totals and tolerance likewise share a
weight exponent. The tolerance remains relative to the total demand weight, including uncovered
rows, even when its raw sum overflows.
This allows a representable improvement even when the previous objective overflows float32.
Assigned costs and the final objective remain in caller units;
score arithmetic retains the usual float32 rounding.
The CPU twin honors the same explicit or default swap budget. It reports a limit after the last
permitted applied swap, and convergence only when a permitted iteration finds no improvement.

`sparseCosts` replaces the dense matrix with CSR rows: `offsets` (`demandCount + 1` non-decreasing
`uint32`, from 0 to the entry count), `facilities` (strictly ascending and below `facilityCount` within a
row) and `costs` (`float32`), with at least one entry; a pair that is not listed is unreachable
(`+Infinity`), and a coverage distance that is `+Infinity` does not cover unlisted pairs. Exactly one of
`costs` and `sparseCosts` is given. Every operation (greedy p-median, maximum coverage, set covering and
the swap phase) supports it and gives the result of the equivalent dense matrix. Violations of the CSR
structure, NaN or negative costs count in `invalidCount`. The GPU builds the facility-major transpose
once (a stable radix sort of the entries by facility and a binary search of the segment offsets), which
the facility-side loops read, so greedy scores and swap candidates scan only the listed pairs and the
work scales with the entry count; hub rows are validated with the shared load-balanced loop, and the
demand loops scan the shorter of the row and the selected-facility list. The dense `demand rows x
facilityCount` limit no longer applies: the entry count is at most
`getGPULocationAllocationMaximumSparseEntryCount(device)` (16,777,216 on a default device, the 8-byte
transpose entry in one 128 MiB binding), the demand row count `offsets.length - 1` at most
`getGPULocationAllocationMaximumDemandCount(device)`, and the facility count and swap pair limits are
unchanged.

Selected-facility lists are compacted deterministically in ascending facility-row order with
256-lane scan/scatter passes. Each compiled swap round contributes `6 + 2 * L` command nodes,
where `L` counts the facility scan level and repeated ceil-divisions by 256 until at most 256
entries remain. `estimatedCommandNodeCount` includes these passes and the independent final
completeness check when either evidence port is supplied; construction must fit the shared
32,768-node budget (`GPU_MAXIMUM_COMMAND_NODE_COUNT`). See the
[allocation command-node budget](./operators-accessibility.md#gpulocationallocation-command-node-budget)
for the formula and examples. Once no swap improves, two one-thread launches remain per compiled swap
round: the apply kernel and the list-dispatch publisher, which disarms the cooperative compaction
passes. The other gated passes do not launch. Maximum coverage and set covering also gate
greedy scoring after a step chooses nothing.

`alternative` (compile-time; a change recompiles) selects the tail, with esda's definitions. With `R`
permutations, `g` is the number of simulated values `>= observed` and `l` the number `<= observed`:

| `alternative` | pseudo p-value |
| --- | --- |
| `'directed'` (default) | `(min(g, R - g) + 1) / (R + 1)`, the previous behavior |
| `'greater'` | `(g + 1) / (R + 1)` |
| `'lesser'` | `(l + 1) / (R + 1)` |
| `'two-sided'` | `min(2 (min(g, l) + 1) / (R + 1), 1)` |
| `'folded'` | `(f + 1) / (R + 1)`, `f` = number of simulated values with `abs(sim - mean) >= abs(observed - mean)` |

`exceedances` holds the numerator count of the chosen tail, and `significant` and the Benjamini-Hochberg
path use the same p-value. `'two-sided'` doubles the p-value against `'directed'` and cannot go below
`2 / (R + 1)`, so raise `permutations` when testing a small level.

`'folded'` matches esda 2.10 exactly: `mean` is the mean of the `R` simulated statistics (the observed
value is excluded, as esda computes it from the reference distribution), ties count, and the test is
symmetric about that mean. The kernels run two sweeps over the same deterministic stream (a sum, then
the count); the other tails keep one. `'two-sided'` is not esda's definition: esda 2.10 places the
observed value at a percentile of the simulated statistics and counts simulations beyond the
interpolated percentiles, which would need every simulated value of a row stored and sorted. The two
can differ by a few counts (a 20-permutation fixture gives 3/21 in esda and 2/21 here). Use `'folded'`
when an undirected test must agree with esda.

On skewed data `'folded'` can flag far fewer rows than `'two-sided'`. With local Moran's I over a
right-skewed variable (hot spots on a low background), a low-value row's conditional distribution is
left-skewed, so its observed value sits in the short tail and the long tail holds simulations just as
far from the mean. Low-low rows that `'greater'` flags then mostly fail `'folded'`, while high-high rows
pass. esda 2.10 behaves the same: on a 1,500-point hot-spot field it flags 211 rows with `'folded'` and
551 with `'two-sided'`.

Random numbers come from Philox 4x32-10 counter streams, keyed by the 64-bit seed with the counter
`(block, row, permutation, tag)`. Results are therefore a pure function of the seed and inputs,
independent of the dispatch shape. The observed and simulated statistics run through the same f32
code, positive constant factors are dropped from the comparisons, and exceedances are integers. The
significance and BH tests compare correctly rounded f32 products, never quotients. Global pair sums
use fixed-shape workgroup trees. Every output is bitwise reproducible for a seed.

Maximum: all limits derive from the device's `maxStorageBufferBindingSize`; the values below are for a default device (128 MiB). `GPUAZPRegions` and `GPUWardRegions` take at most `getGPURegionalizationMaximumSlotCount(device)` (`2^24`) `weights.neighbors` slots (AZP symmetrized adjacency and Ward contracted edge records hold 8 bytes per slot in one binding; Ward also reserves a hash table at exactly twice the input slot capacity) and `GPUMaxPRegions` at most `getGPUMaxPMaximumSlotCount(device)` (`2^23`, its AZP refinement symmetrizes again); `GPUAZPRegions` tables are at most `getGPUAZPMaximumTableWordCount(device)` (`2^25`) words. Ward's region table is at most `getGPUWardMaximumRegionTableWordCount(device)` (`2^25`) words: `rows x (columnCount + 1)` when standardized, `rows x (2 * columnCount + 2)` when raw. `GPULocationAllocation` takes at most `getGPULocationAllocationMaximumFacilityCount(device)` (`2^24`) facilities, `getGPULocationAllocationMaximumDemandCount(device)` (`11184810`) demand rows and `getGPULocationAllocationMaximumCostCount(device)` (`2^25`) cost entries.

`GPUAZPRegions`: Improve contiguous starting regions by moving border tracts, returning region labels, the within-region sum of squares and status words.

::lens{scenario=gpu-spatial-analysis.azp-regions rung=result caption="Eight community areas regrouped by poverty; a 64-round budget allows longer local searches."}

The former 16-round budget stopped while improving moves remained; 32 rounds were also insufficient.
The example now allows 64 rounds, including a no-move round to establish convergence. A round
can move only one tract per participating region; the budget is not a count of full boundary sweeps.
Check `status[GPU_AZP_STATUS_LAYOUT.converged]` before interpreting the result as a local optimum under the bounded
connectivity test.

`GPUMaxPRegions`: Grow as many contiguous regions as possible that each reach a minimum extensive total, returning labels, the within-region sum of squares and status words.

::lens{scenario=gpu-spatial-analysis.max-p-regions rung=result caption="11 regions of at least 40,000 residents grown over 136 tracts; compare poverty with region labels."}

`GPUWardRegions`: Merge adjacent tracts by smallest Ward cost into a target region count, returning labels and the within-region sum of squares.

::lens{scenario=gpu-spatial-analysis.ward-regions rung=result caption="Ward merges 136 tracts into five contiguous regions in 131 merges; the largest holds 91 tracts."}

`GPULocationAllocation`: Choose p facilities from candidate tracts to minimize population-weighted cost, returning selected flags, nearest-facility assignments and the objective.

::lens{scenario=gpu-spatial-analysis.location-allocation rung=result caption="Five facilities chosen from 136 tracts, each tract coloured by its nearest; weighted mean distance 1.24 km."}

`GPURoseStatistic`: Summarize planar line directions as a weighted rose, returning count, mean angle, resultant length, circular variance, Rayleigh z and sector bins.

::lens{scenario=gpu-spatial-analysis.rose-statistic rung=result caption="Axial rose of 894 street segments in 18 sectors; east carries 42.5 km, north 41.3 km."}

The regionalization and location-allocation cards use community areas 1 to 8 (136 contiguous north-side
tracts) of the Chicago tract data, so the unrolled algorithms stay small. Their label values are
nominal identifiers, not scores. The location-allocation cost matrix is straight-line kilometres between
tract vertex-mean centres, prepared on the CPU as the input; the figure does not draw it.

#### How it computes

{/* lens:GPUAZPRegions */}
{/* lens:GPULocationAllocation */}
{/* lens:GPUMaxPRegions */}

`GPUAZPRegions`:

::lens{scenario=gpu-spatial-analysis.azp-regions rung=datum bind=page datum=40 caption="Inspect tract 40: its poverty value, starting community area and AZP region."}
::lens{scenario=gpu-spatial-analysis.azp-regions rung=formula caption="Follow the AZP objective and the gain of moving one tract between regions."}
::lens{scenario=gpu-spatial-analysis.azp-regions rung=kernel caption="Inspect the standardize, candidate-move, claim and apply kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.azp-regions rung=graph caption="Trace queen weights, standardization, up to 64 rounds of batched moves and the final objective."}

`GPUMaxPRegions`:

::lens{scenario=gpu-spatial-analysis.max-p-regions rung=datum bind=page datum=40 caption="Inspect tract 40: its poverty, population and the max-p region that claimed it."}
::lens{scenario=gpu-spatial-analysis.max-p-regions rung=formula caption="Follow the max-p rule: regions meet the population threshold, then the sum of squares decides."}
::lens{scenario=gpu-spatial-analysis.max-p-regions rung=kernel caption="Inspect the growth, enclave and selection kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.max-p-regions rung=graph caption="Trace queen weights, eight parallel starts, growth rounds and the selection of the best start."}

`GPULocationAllocation`:

::lens{scenario=gpu-spatial-analysis.location-allocation rung=datum bind=page datum=40 caption="Inspect tract 40: its population weight, assigned facility and straight-line distance to it."}
::lens{scenario=gpu-spatial-analysis.location-allocation rung=formula caption="Follow the p-median objective: weight times distance to the nearest selected facility."}
::lens{scenario=gpu-spatial-analysis.location-allocation rung=kernel caption="Inspect the greedy choice, swap scoring and finalize kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.location-allocation rung=graph caption="Trace the greedy construction, up to eight swap iterations and the final objective reduction."}

{/* lens:GPURoseStatistic */}
{/* lens:GPUWardRegions */}

`GPUWardRegions`:

::lens{scenario=gpu-spatial-analysis.ward-regions rung=datum bind=page datum=40 caption="Inspect tract 40: its poverty value and the Ward region it merged into."}
::lens{scenario=gpu-spatial-analysis.ward-regions rung=formula caption="Follow the Ward merge cost: size-weighted squared distance between adjacent region means."}
::lens{scenario=gpu-spatial-analysis.ward-regions rung=kernel caption="Inspect the candidate, merge and relabel kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.ward-regions rung=graph caption="Trace queen weights, standardization and up to 32 mutual-nearest merge rounds (23 ran) toward five regions."}

`GPURoseStatistic`:

::lens{scenario=gpu-spatial-analysis.rose-statistic rung=datum bind=page datum=100 caption="Inspect segment 100: its endpoints and length weight, one of 894 rows entering the rose."}
::lens{scenario=gpu-spatial-analysis.rose-statistic rung=formula caption="Follow the circular moments: weighted moments describe street length; unweighted directions give standard Rayleigh z."}
::lens{scenario=gpu-spatial-analysis.rose-statistic rung=kernel caption="Inspect the tiled bin and moment reduction kernel with its retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.rose-statistic rung=graph caption="Trace the single-pass reduction from 894 segments into five summary words and 18 sector bins."}

### `GPUDBSCAN`

Cluster planar points by density (DBSCAN), returning compact cluster labels, core flags and the cluster count.

::lens{scenario=gpu-spatial-analysis.spatial-clustering rung=result caption="DBSCAN with a 250 m radius and six-point cores finds 34 clusters among 2,344 observations; noise stays unclustered."}

Maximum: `gridSize` has at most `getGPUGridIndexMaximumCellCount(device)` (`2^25 - 1` on a default device: `cells + 1` uint32 cell offsets in one 128 MiB binding) cells; with `output.clusterCentroids` and sorted sums the cluster capacity is at most `GPU_DBSCAN_MAXIMUM_SORTED_CLUSTER_COUNT` (`2^24 - 1`, one 256-thread workgroup per cluster).

DBSCAN density clustering of planar points. Bounds, `epsilon`, and `minimumSamples` are per-frame
parameters (`getGPUDBSCANParameterValues`). A point is core when at least `minimumSamples`
valid points lie within its `epsilon` neighborhood, counting the point itself.
`gridSize` is a compile-time maximum
lattice. Each encoding derives an active lattice whose cells are at least `epsilon` wide, so the
3×3 neighborhood search is always exact and `epsilon` changes never recompile. The neighbor scans
read positions and core flags from copies in grid-cell order, so the points of one cell are read
from consecutive memory whatever the input order. Clusters are joined
by a lock-free union-find that hooks larger roots under smaller ones. Labels are therefore
canonical: compact cluster IDs follow the smallest core row of each cluster, and a border point
joins the adjacent cluster with the smallest root. Outputs are per-point labels, roots, and core
flags, the cluster count, a compact list of cluster representatives, and cluster sizes and
centroids, all in the `output` bag. The settings type is `GPUDBSCANSettings`.

With zero positions, only cluster counts, sizes, centroids and compact status scalars are written.
The zero-length per-point views may share their unused input buffer; no dummy storage row is
required. Writable cluster outputs still need valid storage bindings and disjoint logical ranges;
sizes and centroids may share an aligned binding prefix because separate passes fill them.
The optional `clusters.status.candidateOverflow` is a writable publication output too: it must
not overlap labels or other outputs, share an input buffer, or alias another publish binding.
Such aliases are rejected at construction instead of allowing a zero flag to overwrite a label.
The output map, compact cluster map and nested status map are captured before validation.
Replacing a caller-owned candidate destination afterward cannot redirect publication into labels.

```ts
import type {Device} from '@luma.gl/core';
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
import {
  GPUDBSCAN, type GPUDBSCANProps,
  GPU_DBSCAN_PARAMETER_LENGTH, getGPUDBSCANParameterValues
} from '@luma-contributors/analysis/gpu-spatial-analysis';

export function addDBSCANExample(
  graph: GPUCommandGraph, device: Device,
  inputs: Omit<GPUDBSCANProps,
    'parameters' | 'gridSize' | 'spatialContext' | 'times' | 'temporalContext'>
) {
  // Planar example without times: positions/labels: N rows.
  // Compact cluster columns share their declared capacity.
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'dbscan', format: 'float32', length: GPU_DBSCAN_PARAMETER_LENGTH,
    values: getGPUDBSCANParameterValues({
      bounds: [0, 0, 1000, 1000], epsilon: 25, minimumSamples: 8
    })
  });
  graph.add(new GPUDBSCAN({
    ...inputs, parameters: parameterBuffer.importToGraph(graph), gridSize: [128, 128]
  }));
  return {parameterBuffer}; // write this owner between submissions; keep output owners for reads
}
```

These typed examples take caller-imported input/output views. Keep their buffer owners for
readback; returned parameter owners are the objects to write between submissions. Positions and
per-point outputs have N readable rows. Compact cluster IDs, sizes and centroids are readable
only through `clusters.status.count` (clamped to their shared capacity). The contributor writes
unused centroid rows as `(0, 0)`, preserving its documented fill guarantee; this fill does not
make them live clusters.

With `minimumSamples = 1` every point is a core point, so clusters are the connected components of the
graph that joins points within `epsilon`, with no noise. That is PostGIS `ST_ClusterWithin(geom,
distance)` (single-linkage clustering by distance, with the same transitive chaining); larger values
are `ST_ClusterDBSCAN(geom, eps, minpoints)`. Pass `GPUDBSCAN` labels to
[`GPUGroupGeometry`](#gpugroupgeometry) and [`GPUGroupConvexHull`](#gpugroupconvexhull) for cluster
bounds, centers and outlines.

`denseBoxShortcut: true` enables the FDBSCAN dense-box shortcut. Points are binned into boxes of side
`0.7 * epsilon`, and a box with at least `minimumSamples` points makes all of its points core without a
neighbor scan. Labels, roots and core flags are identical to the plain algorithm. The union kernel
still scans neighbors, so the measured gain on 200,000 clustered points is only 3 to 9 percent, and
the option is off by default. Points whose box index exceeds 2^16 skip the shortcut.

For indirect rendering, import an indirect draw record's instance count with
`graph.importGPUData(id, drawCommands.getInstanceCountData(0))` and pass that one-row `uint32` view
as `output.clusters.status.count`.

`sumOrder` defaults to `'sorted'` (stable sort of members by cluster, then a fixed-order segmented sum):
because labels are canonical, `clusterCentroids` is bitwise reproducible across encodings on one device,
and cluster sizes are exact either way. `'atomic'` skips the sorts but uses float atomics, so last bits vary
between runs. Behavior change: last-bit differences versus the previous atomic default.

`spatialContext` (optional) declares the coordinate space of `positions`. Omitted or planar, the search is
Euclidean in position units. It measures distance, so `local-lambert-equal-area` frames throw. `{coordinateSpace: 'longitude-latitude', metric: 'great-circle'}` clusters
`[longitude, latitude]` degrees with `epsilon` a great-circle distance in meters (kilometers with
`units: 'kilometers'`, or any `sphereRadius`; the mean Earth radius by default). Then `bounds` are
`[minLon, minLat, maxLon, maxLat]` degrees inside `[-180, 180] x [-90, 90]`. The neighbor test is the
haversine term against `sin^2(epsilon / (2 R))`, and each encoding sizes its cells to
`epsilon / R` of latitude by `2 asin(sin(epsilon / (2 R)) / cos(maxAbsLatitude))` of longitude (the whole
extent in one column near a pole), so the 3x3 search stays exact. The search wraps at the
antimeridian: points on the two sides of +-180 are neighbours when their great-circle distance is within
`epsilon`, so a cluster straddling +-180 is one cluster, as in `GPUOPTICS`. Cluster centroids remain arithmetic
means of the coordinates. For seam-safe centroids, rotate about a central meridian `c`:
wrap each `longitude - c` into `[-180, 180]`, rotate the bounds consistently, cluster, then inverse-rotate
and wrap each centroid's longitude. At `c = 180` degrees, `[179.9, -179.9]` becomes approximately
`[-0.1, 0.1]`; their centroid rotates back to the antimeridian. Naively unwrapping to `180.1`
violates the supported bounds. Keep latitude and geographic bounds within `[-90, 90]` and
`[-180, 180]` respectively. `denseBoxShortcut`
is planar only, and other metrics (rhumb, ellipsoidal) throw. Geographic results match scikit-learn
`DBSCAN(metric='haversine')` on the test scene.
Planar positions are float32, so the representable spacing near a coordinate of magnitude `M` is about
`M * 1.2e-7`: UTM northings near 5e6 are only 0.5 m apart, and an `epsilon` below a few such spacings
(a few metres) is unreliable. Subtract a local origin from the coordinates (and from `bounds`) before
upload so the spacing stays far below `epsilon`.
The encoded planar epsilon must lie in the inclusive interval
`[SPATIAL_CLUSTERING_MINIMUM_PLANAR_EPSILON, SPATIAL_CLUSTERING_MAXIMUM_PLANAR_EPSILON]` =
`[2^-63, 2^63]` in position units, so squared-distance arithmetic remains usable. The geographic
route instead requires positive finite epsilon in the sphere radius's length unit and
`0.5 * epsilon / sphereRadius < 1.5707963` radians. Both routes require finite representable
lattice extents, cell sizes and indices; a mathematically positive radius alone does not certify
a usable float32 lattice. Use the contributor's `getParameterValues(settings)` to include its
captured context in validation.
Pass optional `spatialContext` to `getGPUDBSCANParameterValues(settings)` to warn on short `epsilon` values; see [Precision of longitude/latitude positions](./concepts.md#precision-of-longitudelatitude-positions).

#### How it computes

{/* lens:GPUDBSCAN */}

::lens{scenario=gpu-spatial-analysis.spatial-clustering rung=datum bind=page datum=1000 caption="Inspect observation 1000: its position, core flag and cluster label."}
::lens{scenario=gpu-spatial-analysis.spatial-clustering rung=formula caption="Follow the core test: observations within epsilon, counted against the minimum points."}
::lens{scenario=gpu-spatial-analysis.spatial-clustering rung=kernel caption="Inspect the grid-binned core, union and label kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.spatial-clustering rung=graph caption="Trace the cell grid, core flags, lock-free union-find and compact labels over 2,344 observations."}

{/* parameters: GPUDBSCAN generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUDBSCAN` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPUDBSCAN`.

- Packer: `getGPUDBSCANParameterValues(settings)`, writes `GPU_DBSCAN_PARAMETER_LENGTH` (8) `float32` values.
- Packer: `instance.getParameterValues(settings)`, writes the `float32` parameter view.

Settings of `getGPUDBSCANParameterValues`:

| Field | Type | Description |
|---|---|---|
| `spatialContext` (optional) | `GPUSpatialContext` | Spatial context of the positions; when it is longitude/latitude, a distance under 10 float32 quanta (about 17 m) logs a warning (see Precision of longitude/latitude positions). Does not change the packed values. |
| `bounds` | `readonly [number, number, number, number]` | Inclusive `[minX, minY, maxX, maxY]` extent of the data. Points outside it, or with a non-finite coordinate, are excluded and labeled noise. Bounds should cover the data: they size the active cell lattice, and a tighter extent makes the neighbor search cheaper. Bounds with `maxX < minX` or `maxY < minY` exclude every point. |
| `epsilon` | `number` | Neighbor radius. Positive; on the planar route its float32 value must lie in [2^-63, 2^63]. |
| `minimumSamples` | `number` | Integer `>= 1`. A point is a core point when its epsilon-neighborhood, including the point itself, holds at least this many points. |
| `timeEpsilon` (optional) | `number` | Space-time clustering (ST-DBSCAN): two points are neighbors only when, besides the spatial test, their relative times differ by at most this many ticks of the contributor's `temporalContext` unit. Positive. Required when the contributor has `times`, and ignored otherwise. Stored at index 6. |

{/* /parameters */}

### `GPUSpatialKMeans`

Partition planar points into k clusters by Lloyd iterations, returning labels, centers, sizes, squared distances, inertia and status.

::lens{scenario=gpu-spatial-analysis.spatial-k-means rung=result caption="k-means assigns 2,344 observations to six clusters; inertia 3.1e9 m² after 14 iterations, converged."}

Deterministic k-means over planar points (scikit-learn `KMeans`, PostGIS `ST_ClusterKMeans`). Props:
`positions`, optional planar `spatialContext` (longitude/latitude throws; project first; it measures
distance, so `local-lambert-equal-area` frames also throw), `clusterCount`
(1 to 256), `maximumIterations` (1 to 64), `initialization` (`'kmeans++'` by default, or `'first-valid'`),
`restartCount` (1 to 32, default 1), the per-frame `parameters` view
(`getGPUSpatialKMeansParameterValues({seed, tolerance})`, `GPU_SPATIAL_KMEANS_PARAMETER_LENGTH` `uint32` words) and the
`output` bag: `labels` (cluster or noise), `centers` (`k` rows), optional `sizes`, `squaredDistances` (the
explain column), `inertia` (a one-row `float32`: the total inertia, the sum of `squaredDistances`),
`status.iterationCount` and `status.converged`.
Behavior change: the default `initialization` was `'first-valid'`, which is order dependent and can place
centers next to each other. Here `'kmeans++'` makes one seeded D-squared draw per round;
scikit-learn uses greedy multiple-trial initialization. Identical seeds across those implementations
do not imply identical centers, partitions or inertia. Each point goes to the nearest center with
lowest-ID ties; means are fixed-order chunked reductions (no sort). Reproducibility applies to this
implementation's seed and input order on one device. Reordering rows can change draws and tie outcomes;
`'kmeans++'` uses `log`, whose rounding WGSL does not pin down across devices.

With `centerCoordinates: true` (the default), k-means accumulates each cluster’s offsets from the
point used to seed or relocate it. It adds back that cluster’s origin when publishing the center,
which rounds to float32. The example also sets `centerCoordinates: true`. Use explicit
`centerCoordinates: false` for historical raw-coordinate arithmetic, which can lose local offsets
and distort inertia far from the origin.
Behavior change (2026-11-10): the `'kmeans++'` draws are Philox 4x32-10 blocks per `(row, round)` under the
registered purpose `kMeansPlusPlusSeeding`, mapped to the open interval `(0, 1)`; they were a private integer hash
whose largest value mapped to exactly 1. The centers a given `seed` selects therefore differ from earlier builds.
When every remaining point coincides with a chosen center, the next center repeats the lowest valid row (finite, not
NaN) and its empty cluster is relocated; a center is NaN only when it could not be initialized at all.

Behavior change (empty clusters): a center that receives no point is no longer left in place. After every
assignment step the empty clusters (lowest ID first) are relocated like scikit-learn's
`_relocate_empty_clusters_dense`: they take the points with the largest squared distance to their assigned
center, farthest first, and the donor clusters' counts and sums lose those points before the means are
taken. Ties in distance go to the lowest row. scikit-learn's `argpartition` leaves the order among the
relocated points unspecified; here it is always descending distance. If every point sits on its center
nothing is relocated, and a cluster that is still empty (or whose only member was taken) moves to the mean of
the largest cluster (lowest ID on ties). A center that is NaN because it could not be initialized (fewer valid
points than `k`) is never relocated. The selection is a single-workgroup iterated argmax with no atomics and
no readback; it costs `O(n E)` for `E` empty clusters and nothing when none is empty.

`restartCount` runs that many independent seeded initializations, each with its own Lloyd iterations, and
returns the one with the lowest total inertia (a fixed-order compensated sum; a tie goes to the lowest
restart index). Restart `r` uses the seed `getGPUSpatialKMeansRestartSeed(seed, r)` (restart 0 uses `seed`), so any
single restart can be reproduced with `restartCount: 1` and that seed. The default of 1 matches
scikit-learn's `n_init='auto'` for k-means++. Restarts are separate node chains in the graph, so graph size
and GPU time grow linearly with `restartCount`; `status.iterationCount`, `status.converged`,
`status.iterationLimitReached` and `inertia` describe the selected lowest-inertia restart,
rather than the last restart executed. Record the input seed, input order, restart count and derived
restart seed when reproducing a particular run.
`restartCount` above 1 needs `'kmeans++'` (`'first-valid'` has no randomness).

```ts
import type {Device} from '@luma.gl/core';
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
import {
  GPUSpatialKMeans, type GPUSpatialKMeansProps,
  GPU_SPATIAL_KMEANS_PARAMETER_LENGTH, getGPUSpatialKMeansParameterValues
} from '@luma-contributors/analysis/gpu-spatial-analysis';

export function addKMeansExample(
  graph: GPUCommandGraph, device: Device,
  inputs: Pick<GPUSpatialKMeansProps, 'positions' | 'output'>
) {
  // N >= 8 positions/labels; centers: 8 float32x2 rows; sizes: 8 uint32 words.
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'kmeans', format: 'uint32', length: GPU_SPATIAL_KMEANS_PARAMETER_LENGTH,
    values: getGPUSpatialKMeansParameterValues({seed: 3})
  }); // two uint32 words; tolerance occupies its packed float32 bit pattern
  const kmeans = new GPUSpatialKMeans({
    ...inputs, clusterCount: 8, maximumIterations: 20, initialization: 'kmeans++',
    centerCoordinates: true, parameters: parameterBuffer.importToGraph(graph)
  });
  graph.add(kmeans);
  parameterBuffer.write(getGPUSpatialKMeansParameterValues({seed: 3}));
  return {kmeans, parameterBuffer, status: kmeans.output.status};
}
```

After `graph.add(kmeans)`, omitted stopping destinations are published at `kmeans.output.status`:
`iterationCount`, `converged` and `iterationLimitReached`, each one `uint32`. Supplied destinations
are retained. `maximumIterations` is a cap, and `tolerance` (per frame, default 0) is the largest
Euclidean center displacement in position units, not its square or an inertia change. A zero
tolerance requires an exact fixed point, including tiny representable nonzero movements. A final
permitted round that meets tolerance reports `[converged, iterationLimitReached] = [1, 0]`;
exhaustion before meeting it reports `[0, 1]` and retains the current iterate. After an
iteration whose shift is at most `tolerance`, the one-thread convergence check zeroes a GPU-written
indirect dispatch, so the later iterations' assign, reduction and relocation kernels do not launch; the
result equals running every iteration when `tolerance` is 0 (an exact fixed point). The optional one-row
`status.iterationCount` and `status.converged` outputs report the iterations used and the flag. The seed and tolerance
are `parameters` words and the k-means++ round and the restart index come from a GPU cursor word, so the graph compiles one pipeline per kernel at any `k` and
`restartCount`, and restart scoring is a two-stage (per-chunk, then fixed-order) reduction.

#### How it computes

{/* lens:GPUSpatialKMeans */}

::lens{scenario=gpu-spatial-analysis.spatial-k-means rung=datum bind=page datum=1000 caption="Inspect observation 1000: its cluster label and squared distance to that center."}
::lens{scenario=gpu-spatial-analysis.spatial-k-means rung=formula caption="Follow the inertia and the center update: squared distance to the nearest center, then member means."}
::lens{scenario=gpu-spatial-analysis.spatial-k-means rung=kernel caption="Inspect the assignment, chunked reduction and center-update kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.spatial-k-means rung=graph caption="Trace k-means++ seeding and 20 unrolled Lloyd iterations down to the inertia reduction."}

{/* parameters: GPUSpatialKMeans generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUSpatialKMeans` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPUSpatialKMeans`.

- Packer: `getGPUSpatialKMeansParameterValues(settings?)`, writes `GPU_SPATIAL_KMEANS_PARAMETER_LENGTH` (2) `uint32` values.

Settings of `getGPUSpatialKMeansParameterValues`:

| Field | Type | Description |
|---|---|---|
| `seed` (optional) | `number` | Seed of the `'kmeans++'` draws, an integer in `[0, 2^32 - 1]`. Default 0. Restart `r` is seeded with `getGPUSpatialKMeansRestartSeed(seed, r)`. |
| `tolerance` (optional) | `number` | Convergence tolerance: an iteration converges when the largest center shift (Euclidean, in position units) is at most this value. Finite and `>= 0`; default `0`, which stops only at an exact fixed point and so gives the same result as running every iteration. A positive value trades exactness for fewer iterations. After convergence the remaining iterations' assign, reduction and relocation kernels are skipped through a GPU-written indirect dispatch, so each one costs only a one-thread convergence check. |

{/* /parameters */}

### `GPUHDBSCAN`

Cluster planar points by density at every scale (HDBSCAN), returning compact cluster labels, core distances and the cluster count. Optional relative times and a per-frame time scale enable [scaled ST-HDBSCAN](./space-time-clustering.md#st-hdbscan) through the same contributor.

::lens{scenario=gpu-spatial-analysis.hierarchical-density-clustering rung=result caption="HDBSCAN finds 19 clusters of different density among 782 observations without a radius; the rest is noise."}

HDBSCAN density clustering of planar points, fully on the GPU, matching scikit-learn's `HDBSCAN` with its
defaults (`cluster_selection_method='eom'`, `allow_single_cluster=False`, `cluster_selection_epsilon=0`,
Euclidean) up to a relabeling of the clusters. Props: `positions` (at most
`GPU_HDBSCAN_MAXIMUM_ROW_COUNT`, 4,194,304 rows on the default
`coreDistanceSearch: 'bvh'` route), the compile-time `minimumClusterSize`
(`min_cluster_size`, at least 2) and `minimumSamples` (`min_samples`, the point itself counted, 1 to 32,
default `min(minimumClusterSize, 32)`), and `output: {labels, coreDistances?, status?: {count}}`. Labels are
compact cluster IDs or `0xffffffff` for noise; non-finite points are noise.

The default core-distance search walks a BVH with a private per-point top-k list (the point
itself is the first sample). Opt-in `coreDistanceSearch: 'neighbor-search'` uses a
`GPUNeighborSearch` k-nearest self join as a reference route, limited to
`GPU_HDBSCAN_REFERENCE_MAXIMUM_ROW_COUNT` = 1,048,576 rows and rejecting
`times`. Both routes retain the downstream pipeline: the exact minimum spanning tree over the mutual
reachability distance by Boruvka (unrolled `ceil(log2 rows) + 1` rounds of the same kernels, gated on the GPU
component count; each point walks a Hilbert-sorted BVH and skips nodes that hold only its own component);
the single-linkage dendrogram by parallel parent splicing over the sorted tree, with Euler-tour pointer
doubling for subtree sizes; condensing by pointer jumping; stability as a fixed-order segmented sum; and
excess-of-mass selection by heavy-path tree contraction. The tree passes use GPU convergence flags and
indirect dispatch gates, including on deep chains; the row maximum bounds transient storage.
Where mutual reachability distances tie, several minimum spanning trees exist and the GPU breaks ties by point
rows, which can condense a few points differently from scikit-learn: on the oracle scenes
(`test/oracles/spatial-clustering-hdbscan.py`) the spanning-tree weights are identical and the adjusted Rand
index is at least 0.98. Membership probabilities, outlier scores, `allow_single_cluster` and `'leaf'`
selection are not provided.

```ts
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUHDBSCAN, type GPUHDBSCANProps
} from '@luma-contributors/analysis/gpu-spatial-analysis';

export function addHDBSCANExample(
  graph: GPUCommandGraph,
  inputs: Pick<GPUHDBSCANProps, 'positions' | 'output'>
) {
  // positions, labels and optional coreDistances: N rows; output.status.count: one word.
  graph.add(new GPUHDBSCAN({
    ...inputs, minimumClusterSize: 15, minimumSamples: 5
  }));
}
```

#### How it computes

{/* lens:GPUHDBSCAN */}

::lens{scenario=gpu-spatial-analysis.hierarchical-density-clustering rung=datum bind=page datum=300 caption="Inspect observation 300: its core distance and cluster label."}
::lens{scenario=gpu-spatial-analysis.hierarchical-density-clustering rung=formula caption="Follow core distance, mutual reachability and cluster stability, which decide the surviving clusters."}
::lens{scenario=gpu-spatial-analysis.hierarchical-density-clustering rung=kernel caption="Inspect the k-nearest, Boruvka, dendrogram, condensing and selection kernels with retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.hierarchical-density-clustering rung=graph caption="Trace core distances, the spanning tree, the dendrogram and excess-of-mass selection over 782 observations."}

{/* parameters: GPUHDBSCAN generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUHDBSCAN` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPUHDBSCAN`.

- Packer: `getGPUHDBSCANParameterValues(settings)`, writes `GPU_HDBSCAN_PARAMETER_LENGTH` (4) `float32` values.

Settings of `getGPUHDBSCANParameterValues`:

| Field | Type | Description |
|---|---|---|
| `timeScale` | `number` | Spatial position units per temporal-context tick. Finite and non-negative; zero ignores time distances. |

{/* /parameters */}

### `GPUSpatialWeightsMinimumSpanningTree`, `GPUSkaterRegions`, and `GPURegionPartitionEvaluation`

`GPUSpatialWeightsMinimumSpanningTree`: Find the minimum spanning forest of a contiguity graph by attribute dissimilarity, returning tree-edge flags, component labels and edge costs.

Either weights loss flag suppresses the forest: tree flags are zero, component labels are
`0xffffffff` (unavailable), and compact count/requiredCount are zero. The producer's `output`
port carries compact candidateOverflow for either upstream cause; compact overflow only reports
compact destination capacity. SKATER on the same weights suppresses labels (`0xffffffff`),
region count (zero/unavailable), cut IDs (no cut) and gains (NaN). Clearing loss restores normal
results on resubmission. Showcase/explorer region and coloring readouts withhold definite results
while either producer flag is set; a sentinel is never a region, color or observed count.
`GPUSkaterRegionsOperation` and `CPUSkaterRegionsOperation` accept the two graph value names under
`weights.status`. The pure `computeSkaterRegionsOnCPU` accepts the corresponding uint32 arrays
under `input.status`, alongside its flat CSR arrays, and uses the same suppression sentinels.
Both operation forms read the supplied loss values on every run of a reused compiled graph;
clearing a loss word restores normal outputs on the next run.
Pass the complete producer weights port to every forest and partition stage, including REDCAP
and Ward; a clean supplied status has the same result as omitted status. MST's packed
`output.status` requires `GPU_MINIMUM_SPANNING_TREE_STATUS_LAYOUT.length` words (now two):
`invalidCount` stays at word zero and `candidateOverflow` at word one reports either upstream
adjacency loss cause as 0/1, even without compact edges. Increase old one-word allocations to
the named length. Compact `output.edges.status.candidateOverflow` reports the same adjacency loss.
SKATER consumes the forest's loss word through `treeStatus`, in both operation forms and its CPU
twin. On clean and suppressed frames it writes one region-count word and
`maximumRegionCount - 1` cut entries; spare output capacity stays untouched.
Forest, SKATER, LOSH and multivariate Geary capture their declared props and nested port maps at
construction, including inherited getters and masks. Change GPU buffer contents to resubmit;
construct a new contributor to change a view or destination.
Both named SKATER operations accept `weights.status.overflow` and `candidateOverflow` value names
and apply this suppression before CPU forest traversal or GPU cuts. The pure CPU twin accepts the
same scalar arrays in `input.status`. A missing forest label during loss is unavailable, not an
invalid-tree exception. Only one `regionCount` word and `maximumRegionCount - 1` cut-log words are
written, including during suppression; spare destination words are preserved. A supplied SKATER
`regionCount` must be a packed `uint32` view holding at least one word; empty views are rejected
at construction.
SKATER retries a nonfinite intermediate gain with an exponent-separated weighted squared norm.
A finite gain remains eligible even when squaring a raw mean difference alone would overflow;
genuinely nonfinite gains and nonfinite side means remain ineligible.
Finite columns constant within an initial tree are removed from that tree's sums before they
can overflow; they contribute exactly zero to every cut. Constant detection is per tree, so a
different feature value on an island does not affect another tree's cuts. Nonfinite columns
are not removed this way and retain the documented invalid-input behavior.

::lens{scenario=gpu-spatial-analysis.spatial-weights-minimum-spanning-tree rung=result caption="The 135 tree edges joining 136 tracts, drawn between centroids over poverty and coloured by dissimilarity; total 29.6."}

`GPUSkaterRegions`: Cut a spanning tree greedily where the within-region sum of squares falls most, returning region labels, cut edges and cut gains.

::lens{scenario=gpu-spatial-analysis.skater-regions rung=result caption="Target five regions, with at least five tracts per cut side to exclude singleton regions."}

This example sets `minimumRegionSize: 5` (spopt's `floor`) because the default of one permits
singleton regions. Each cut maximizes SSD reduction among cuts satisfying that floor; it does
not balance region sizes and may stop below the target if no feasible cut remains.

`GPURegionPartitionEvaluation`: Score any label column by within-region, between-region and total sum of squares, region sizes, boundary share and pseudo-F.

A nonzero supplied weights loss flag (`overflow` or `candidateOverflow`, including a flag on any
CSR chunk) makes `crossLinkFraction` NaN. The remaining statistics depend only on labels and
values and remain available.
Outer chunk-level loss flags must be nonempty packed `uint32` views, just like whole-CSR flags.
Loss views must also have packed `uint32` layout and a byte offset divisible by four; invalid
outer, whole-CSR and chunk-local flags are rejected at construction with the public field name.
Regional and global moments use member origins for same-sign columns, preserving constant
stored float32 columns exactly. Mixed-sign columns use zero origins. Parallel regional sum
stages retain subtraction and summation residuals in two float32 lanes, including residuals
below 2^-103; grand sums and bounded summary merges retain the same residuals. Between SSD
forms each regional contrast as (N S_r - n_r S) / (n_r N) from split sums in the global frame
and exact products before one division, so regions with identical distributions give exactly
zero. A constant table has exactly zero within,
between and total SSD; the documented zero-within pseudo-F is 1 when it is defined.
Pseudo-F combines normal mantissas, degrees of freedom and exponent differences, avoiding both
an overflowing product and an underflowing mean square when its SSDs and ratio are representable.
The final summary combines row and label tiles with published parallel scalar reductions, then
publishes their totals without a serial tile scan. Float32 SSD rounding can differ from the
previous sequential tile merge and between devices with different reduction strategies.

::lens{scenario=gpu-spatial-analysis.region-partition-evaluation rung=result caption="Nine summary words for the eight community areas as a partition of poverty; pseudo-F is 13.7."}

These cards use community areas 1 to 8 (136 contiguous north-side Chicago tracts). The spanning-tree card
draws tree edges as segments between GPU-computed tract centroids.

SKATER accepts 1–16 attribute columns. At a 128 MiB storage binding limit, its arc and row statistic
tables each hold at most 33,554,432 words, with two words plus one per attribute per slot or row.
The minimum spanning forest accepts 1–32 attribute columns. Its attribute, row and slot scratch
bindings each hold at most 33,554,432 32-bit entries. Partition evaluation accepts 1–32 attribute
columns, up to 16,777,216 label rows, label capacity up to 33,554,431, and at most 33,554,432
entries in `labelCapacity * columnCount` at that binding size. The minimum spanning forest
shares the 32,768 estimated command-node limit.

Spatially constrained regionalization over a contiguity graph (PySAL spopt `SpanningForest` and SKATER,
Assuncao et al. 2006).

- `GPUSpatialWeightsMinimumSpanningTree`: Boruvka minimum spanning forest over a symmetric
  `GPUSpatialWeights` (every link in both rows; for directed weights build the union with
  `GPUSpatialWeightsTranspose` first, unlike the AZP, Max-P and Ward operators, which symmetrize
  themselves because the tree's edge IDs are the caller's CSR slots). The edge cost is the squared Euclidean distance between (optionally standardized)
  attribute rows. A total order of `(cost, lower-row CSR slot)` makes the forest unique and equal to
  Kruskal under that order, with integer atomics and no float races. `output` holds `treeEdgeFlags`
  (per CSR slot), `componentLabels` (the minimum row of each tree) and, optionally, a compact `edges`
  list (`ids` plus `status: {count, overflow, requiredCount?, candidateOverflow?}`) with `edgeEndpoints`, `edgeDissimilarities` and `standardizedValues`. It unrolls `ceil(log2 rows)` rounds. Tied costs
  can pick different edges than scipy's traversal order, with the same total cost.
  Non-finite attribute entries are skipped as in `GPUAZPRegions`, `GPUWardRegions` and `GPUMaxPRegions`:
  the standardization mean and deviation use the finite entries of each column, so one NaN no longer
  makes every edge cost NaN, and the entry stays non-finite in `standardizedValues`. Every link that
  touches a row with a non-finite entry costs `+Infinity`, so it joins the forest last and the forest
  still spans each connected component. The optional `output.status` stores the skipped-entry count,
  before standardization, at `GPU_MINIMUM_SPANNING_TREE_STATUS_LAYOUT.invalidCount`; its second word,
  `GPU_MINIMUM_SPANNING_TREE_STATUS_LAYOUT.candidateOverflow`, independently reports adjacency loss.
  Unlike the three solvers above, the tree is still built when the count is nonzero. Pass the status
  view to `GPUSkaterRegions` as `treeStatus`: like those solvers, SKATER then makes no cut while the
  count is nonzero (each tree stays one region and every `cutEdges` entry is `GPU_SKATER_NO_CUT`).
  A nonzero adjacency-loss word suppresses labels/count/cuts/gains instead of publishing a partition.
- `GPUSkaterRegions`: greedy SKATER cuts of that tree. Each step evaluates the sum-of-squared-deviation
  reduction of every tree edge in parallel (Euler-tour side sums) and applies the best one, ties to the
  lowest slot, while honoring `minimumRegionSize`. The reduction is computed in its direct form
  `nA nB / (nA + nB) |mean A - mean B|^2` from the two sides' sizes and attribute sums (the Ward increase), not as
  a difference of three sums of squares, so a small gain inside a quiet region whose mean is far from zero is not
  lost to float32 rounding; the side sums carry the exact rounding error of each addition. What remains is the
  last place of the region's attribute sum spread over the smaller side, which centred attributes keep small. The target region count and the minimum size are
  per-frame `parameters` words written with
  `getGPUSkaterRegionsParameterValues({regionCount, minimumRegionSize})`
  (`GPU_SKATER_REGIONS_PARAMETER_LENGTH` words); at most `min(maximumRegionCount - 1, rows - 1)` steps
  are compiled, so the
  partition with k regions is a prefix of the one with k + 1. `output` holds `labels` (the row index of
  each region's top), `regionCount`, `cutEdges` and `cutGains`. The partitions equal spopt `SpanningForest` (squared-Euclidean metric) exactly, including
  floors and islands; spopt's `islands='increase'` adds the tree count to `n_clusters`, while here the
  target counts every region. SKATER does not center attributes itself: the tree's `standardize`
  (default true) does, and `standardizedValues` should be passed as the SKATER `values` so the cuts and
  the tree share a metric and the f32 subtree-sum subtraction stays well conditioned.
  The optional cut logs still contain `maximumRegionCount - 1` entries; entries beyond the attainable
  cuts are `GPU_SKATER_NO_CUT` and zero gain.
- `GPURegionPartitionEvaluation`: scores any label column (SKATER, `GPUDBSCAN`, `GPUSpatialKMeans`):
  region count, within, between and total sum of squared deviations, smallest and largest region, the
  fraction of weights-graph links that cross regions (a compactness proxy) and the number of ignored
  rows, plus per-label sizes and within-region deviations. The summary also holds `pseudoF`, the
  Calinski-Harabasz statistic `(between / (k - 1)) / (within / (n - k))`, pinned against
  `sklearn.metrics.calinski_harabasz_score` (NaN where undefined, 1 when the within-region dispersion
  is zero, as sklearn). `output` holds `summary` (`GPU_REGION_PARTITION_EVALUATION_LAYOUT`: `withinSumOfSquares`,
  `betweenSumOfSquares`, `totalSumOfSquares`, ...) and the optional `regionSizes` and
  `regionWithinSumOfSquares`. GPU evaluation stages row and label tiles through bounded
  256-item reductions; `GPU_REGION_PARTITION_EVALUATION_MAXIMUM_COLUMNS = 32` limits attribute
  columns, not row or label tiles. Row, label-offset and `labelCapacity * columnCount` binding
  limits above still apply.
  `betweenSumOfSquares` is computed directly as `sum over regions of n_r |mean_r - mean|^2`, not as
  `total - within`, so it and `pseudoF` keep their digits when the partition explains a small share of the total
  (it equals `total - within` up to float32 rounding).
  A labelled row with a NaN or infinite value in any column is left out of every statistic, as an unlabelled row
  is: it is counted in `ignoredRows`, and the optional one-row `output.invalidRowCount` says how many such rows
  there were. One missing value removes one row; it no longer turns every statistic into NaN.

`labelCapacity` is an exclusive source-label bound, not a region count. On path 0–1–2–3 with
attributes `[0, 0, 10, 10]`, SKATER can return `[0, 0, 2, 2]` and region count 2. Evaluate these
labels with capacity 3 or the default 4; capacity 2 silently excludes rows labelled 2. Alternatively,
compact first and retain the join keys:

```ts
import {compactRegionLabels} from '@luma-contributors/analysis/gpu-spatial-analysis';

const compact = compactRegionLabels(Uint32Array.of(0, 0, 2, 2));
// labels = [0, 0, 1, 1], sourceLabels = [0, 2], count = 2
const labelCapacity = Math.max(1, compact.count); // 2; size per-label outputs to this capacity
// Upload compact.labels for evaluation; sourceLabels[compact.labels[row]] recovers the source key.
```

For all-missing labels, true count is zero but capacity and per-label allocations are at least 1;
all rows are ignored, region sizes are zero and pseudo-F is undefined. Optional scalar
`labelsComplete` (`uint32`, nonzero means complete) carries solver finality independently of
adjacency loss. Supplying it requires `output.labelsComplete`, which publishes normalized 0/1;
output-only publishes 1 and omission assumes ordinary complete labels. Scores remain readable
for provisional partitions, so inspect this evidence before presenting them as final.

```ts
import type {Device} from '@luma.gl/core';
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
import {
  GPUSpatialWeightsMinimumSpanningTree, type GPUSpatialWeightsMinimumSpanningTreeProps,
  GPUSkaterRegions, type GPUSkaterRegionsProps,
  GPU_SKATER_REGIONS_PARAMETER_LENGTH, getGPUSkaterRegionsParameterValues
} from '@luma-contributors/analysis/gpu-spatial-analysis';

export function addSkaterExample(
  graph: GPUCommandGraph, device: Device,
  inputs: Pick<GPUSpatialWeightsMinimumSpanningTreeProps, 'weights' | 'values'>,
  tree: Required<Pick<GPUSpatialWeightsMinimumSpanningTreeProps['output'],
    'treeEdgeFlags' | 'componentLabels' | 'standardizedValues' | 'status'>>,
  output: GPUSkaterRegionsProps['output']
) {
  // values/standardizedValues: N*3 floats; treeEdgeFlags: CSR slots; componentLabels/labels: N.
  // tree.status: two words; output.regionCount: one; optional cut logs: 11 entries.
  graph.add(new GPUSpatialWeightsMinimumSpanningTree({
    ...inputs, columnCount: 3, output: tree
  }));
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'skater', format: 'uint32', length: GPU_SKATER_REGIONS_PARAMETER_LENGTH,
    values: getGPUSkaterRegionsParameterValues({regionCount: 8})
  });
  graph.add(new GPUSkaterRegions({
    weights: inputs.weights, treeEdgeFlags: tree.treeEdgeFlags,
    componentLabels: tree.componentLabels, treeStatus: tree.status,
    values: tree.standardizedValues, columnCount: 3,
    parameters: parameterBuffer.importToGraph(graph), maximumRegionCount: 12, output
  }));
  return {parameterBuffer};
}
```

SKATER, Ward, AZP and Max-p expose `estimatedCommandNodeCount` and reject construction when this
upper bound exceeds the shared 32,768-node limit. The estimate includes transitive adjacency,
sort/scan, contiguity and nested refinement nodes. Published primitives have named conservative
allowances, checked against emitted graphs in node tests. A GPU dispatch gate saves execution
after convergence; the construction budget still counts every unrolled round.

Include the producer's final overflow view with any `requiredCount`. Verify completeness and
convergence before supplying a CSR that carries `complete`, `converged`, `iterationLimitReached`
or `unresolvedCount`, then omit those views from the regionalization input. This rule also
applies to each chunk supplied for partition evaluation.

SKATER's bound includes loss aggregation and each requested suppression pass. With loss ports
and all four outputs, 4,094 rows and `maximumRegionCount: 4094` require 32,773 nodes and are refused.
Ward and AZP snapshot their iteration budgets (and Ward's schedule) before validation. Ward's
mutual-nearest estimate uses 18 nodes per round through 4,096 input rows and 45 per round above
that threshold, matching the cooperative and radix selection routes respectively.

Cost: the publish and apply kernels are single-threaded and each SKATER step is
`O(sum of side sizes)`, which suits thousands of units and is quadratic on path-like trees of 100,000
rows or more. Once a step finds no cut or the target region count is reached, the one-thread kernels
zero a GPU-written indirect dispatch, so the remaining steps launch only their one-thread kernels. The
step index comes from a GPU cursor word, so every step shares one pipeline per kernel.

#### How it computes

{/* lens:GPUSpatialWeightsMinimumSpanningTree */}
{/* lens:GPUSkaterRegions */}
{/* lens:GPURegionPartitionEvaluation */}

`GPUSpatialWeightsMinimumSpanningTree`:

::lens{scenario=gpu-spatial-analysis.spatial-weights-minimum-spanning-tree rung=datum bind=page datum=40 caption="Inspect tract 40: its poverty value, standardized value and component label in the tree."}
::lens{scenario=gpu-spatial-analysis.spatial-weights-minimum-spanning-tree rung=formula caption="Follow the edge cost: squared difference of standardized attribute rows across each link."}
::lens{scenario=gpu-spatial-analysis.spatial-weights-minimum-spanning-tree rung=kernel caption="Inspect the standardize, edge-cost, cheapest-edge and relabel kernels with retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.spatial-weights-minimum-spanning-tree rung=graph caption="Trace queen weights, edge costs and eight Boruvka rounds that merge trees."}

`GPUSkaterRegions`:

::lens{scenario=gpu-spatial-analysis.skater-regions rung=datum bind=page datum=40 caption="Inspect tract 40: its poverty value, tree component and SKATER region."}
::lens{scenario=gpu-spatial-analysis.skater-regions rung=formula caption="Follow the cut gain: region sum of squares minus the sums of its two sides."}
::lens{scenario=gpu-spatial-analysis.skater-regions rung=kernel caption="Inspect the Euler-tour, cut-gain, best-cut and apply kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.skater-regions rung=graph caption="Trace queen weights, the spanning tree and up to seven greedy cut steps."}

`GPURegionPartitionEvaluation`:

::lens{scenario=gpu-spatial-analysis.region-partition-evaluation rung=datum bind=page datum=40 caption="Inspect tract 40: its poverty value and community-area label in the evaluated partition."}
::lens{scenario=gpu-spatial-analysis.region-partition-evaluation rung=formula caption="Follow within, between and total sums of squares into the Calinski-Harabasz pseudo-F."}
::lens{scenario=gpu-spatial-analysis.region-partition-evaluation rung=kernel caption="Inspect the sort, region-sum, deviation and summary kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.region-partition-evaluation rung=graph caption="Trace queen weights, the label sort, per-region sums and the nine-word summary."}

{/* parameters: GPUSkaterRegions generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUSkaterRegions` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPUSkaterRegions`.

- Packer: `getGPUSkaterRegionsParameterValues(settings)`, writes `GPU_SKATER_REGIONS_PARAMETER_LENGTH` (2) `uint32` values.

Settings of `getGPUSkaterRegionsParameterValues`:

| Field | Type | Description |
|---|---|---|
| `regionCount` | `number` | Target region count, clamped to `maximumRegionCount` on the GPU. A value at or below the number of forest trees cuts nothing. |
| `minimumRegionSize` (optional) | `number` | Minimum region size in rows. Defaults to 1. |

{/* /parameters */}

### `GPUGroupGeometry`

Summarize the geometry of each label group, returning counts, bounds, centres, medoids, standard distances and ellipses.

::lens{scenario=gpu-spatial-analysis.group-geometry rung=result caption="Mean centres of the 34 DBSCAN clusters of 2,344 observations, sized by member count; the largest has 553."}

Planar only: `spatialContext` defaults to native planar coordinates and a longitude/latitude context
throws at construction (project first with `createGPULocalFrame` and `GPULocalFrameProjection`).
Geographic group centroids live in `GPUGeometryMeasures`.
Declares `measure: 'distance'`: a Lambert equal-area local frame throws; use an
azimuthal-equidistant frame.

Per-label summaries over a `uint32` label column, for example the output of `GPUDBSCAN`:
count, bounds `[minX, minY, maxX, maxY]`, mean center, weighted center, weight sum, medoid, standard
deviational ellipse and standard distance.

- Props: `positions` (`float32x2`), `labels`, `groupCount`, optional `noiseLabel` and `weights`,
  `parameters` (the `getGPUGeographicDistributionParameterValues` layout: origin, standard deviations,
  ellipse convention), `medoidMaximumGroupSize` (default 4096) and `output`.
- Outputs (all optional, at least one): `counts`, `bounds` (`float32x4`), `meanCenters`,
  `weightedCenters` and `weightSums` (need `weights`), `medoidIndices`, `ellipses`
  (`[angle, sigmaX, sigmaY]`, weighted when `weights` is given), `standardDistances` and `status.overflow`.
- Labels at or above `groupCount` (for example DBSCAN noise, `0xffffffff`), a label equal to
  `noiseLabel`, and rows with a non-finite coordinate are excluded. For empty groups, requested
  `counts` and `weightSums` are zero, every other requested float summary is NaN, and requested
  `medoidIndices` is `GPU_GROUP_GEOMETRY_NO_MEDOID`.
- The medoid is the included row with the smallest summed Euclidean distance to the group, lowest row
  on ties, unweighted. It is quadratic per group, so a group larger than `medoidMaximumGroupSize` gets
  no medoid and sets `output.status.overflow`.
- Sums are fixed-order, so results are bitwise reproducible.

```ts
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUGroupGeometry, type GPUGroupGeometryProps
} from '@luma-contributors/analysis/gpu-spatial-analysis';

declare const graph: GPUCommandGraph;
declare const positions: GPUGroupGeometryProps['positions'];
declare const clusterLabels: GPUGroupGeometryProps['labels'];
declare const parameters: GPUGroupGeometryProps['parameters'];
declare const output: Required<Pick<GPUGroupGeometryProps['output'],
  'counts' | 'bounds' | 'meanCenters' | 'medoidIndices' | 'ellipses'
>>;
graph.add(new GPUGroupGeometry({
  positions, labels: clusterLabels, groupCount: 64,
  parameters, output
}));
```

#### How it computes

{/* lens:GPUGroupGeometry */}

::lens{scenario=gpu-spatial-analysis.group-geometry rung=datum bind=page datum=1000 caption="Inspect observation 1000: its cluster label, the group it adds to."}
::lens{scenario=gpu-spatial-analysis.group-geometry rung=formula caption="Follow the group mean centre and the medoid, the member with the smallest summed distance."}
::lens{scenario=gpu-spatial-analysis.group-geometry rung=kernel caption="Inspect the sort, segmented bounds, centre, ellipse and medoid kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.group-geometry rung=graph caption="Trace DBSCAN labels, the sort by group, segmented reductions and the medoid search."}

{/* parameters: GPUGroupGeometry generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUGroupGeometry` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPUGroupGeometry`.

- Packer: `getGPUGeographicDistributionParameterValues(settings?)`, writes `GPU_GEOGRAPHIC_DISTRIBUTION_PARAMETER_LENGTH` (8) `float32` values.
  - Input port: `parameters`.

Settings of `getGPUGeographicDistributionParameterValues`:

| Field | Type | Description |
|---|---|---|
| `origin` (optional) | `readonly [number, number]` | Local origin `[x, y]` subtracted from every position before any sum, so float32 sums keep precision when the planar coordinates are large (for example Web Mercator meters). Choose a point near the data. Defaults to `[0, 0]`. |
| `standardDeviations` (optional) | `number` | Number of standard deviations that scale `standardDistances`, `ellipses` and the vertex rings (ArcGIS offers 1, 2 and 3). Any positive number is accepted. Defaults to `1`. |
| `ellipseConvention` (optional) | `GPUGeographicDistributionEllipseConvention` | Axis scaling of the ellipse: `'root-two'` scales the axes by `sqrt(2)` as the ArcGIS Directional Distribution tool does, `'standard'` uses the plain standard deviation along each principal axis. Defaults to `'root-two'`. |
| `orientationOnly` (optional) | `boolean` | Treat lines as undirected (angles are doubled before averaging, so a line and its reverse agree). Only affects `directionalMeans`. Defaults to `false`. |
| `medianTolerance` (optional) | `number` | Absolute tolerance, in position units, under which the last Weiszfeld step counts as converged for `medianConverged`. The iteration stops once every non-empty group's last step is at most this value. Non-negative. Defaults to `GPU_GEOGRAPHIC_DISTRIBUTION_DEFAULT_MEDIAN_TOLERANCE`. |

{/* /parameters */}

### `GPUGroupConvexHull`

Outline each label group by its convex hull, returning hull vertices in CSR layout with emitted counts and true sizes.

::lens{scenario=gpu-spatial-analysis.group-convex-hull rung=result caption="Convex hulls of the 34 DBSCAN clusters, one outline per cluster; the largest has 15 vertices."}

Planar only: a longitude/latitude `spatialContext` throws at construction; project first with
`createGPULocalFrame` and `GPULocalFrameProjection`.

Convex hull per label (Andrew monotone chain), for cluster and hot-region outlines and trajectory
footprints (PostGIS `ST_ConvexHull` per group, `GeoSeries.convex_hull`). Props: `positions`, `labels`,
`groupCount`, optional `noiseLabel`, `maximumVerticesPerGroup` (at least 3), `vertexCapacity` and
`output`: `vertexIndices`, optional `vertexPositions`, `offsets` (`groupCount + 1`), `counts`, optional
`sizes` (true hull sizes) and `status.overflow` (0 or 1). Optional one-row diagnostics
`status.groupOverflow` and `status.vertexOverflow` identify the per-group and total vertex limits,
respectively. Every supplied status is rewritten each frame, including empty groups.

`status.overflow` previously packed bits 1 and 2; code testing those bits must use the corresponding
named diagnostic. The exported `GPU_GROUP_CONVEX_HULL_GROUP_OVERFLOW` and
`GPU_GROUP_CONVEX_HULL_TOTAL_OVERFLOW` constants remain available but are deprecated; they no longer
apply to `status.overflow`.

- Hulls are counter-clockwise, not closed, and start at the smallest `(x, y)` vertex. Collinear points
  are excluded, equal points collapse to the lowest row, one distinct point gives one vertex, and
  all-collinear points give the two extremes.
- Orientation is exact: coordinates are snapped to a 2^29 lattice with one global power-of-two scale,
  and the sign is computed with 64-bit integer arithmetic. Output positions are the original floats.
- The lattice is global, derived from the extent of all rows rather than per group. The hull is exact
  for the snapped points, not for the raw floats, and adding a distant group (or any row that widens
  the extent) can change the lattice scale and so the hull of an unrelated small group.
- A hull over the per-group cap is dropped whole. Groups are admitted in group order until the total
  capacity runs out, and the first group that does not fit and all later ones are dropped (`counts` 0,
  `sizes` still true).
- `extremePointFilter` (default `true` up to 65536 groups) is an Akl-Toussaint filter before the sorts:
  per group it finds the eight extreme lattice points (min and max of x, y, x + y and x - y) and gives
  every row strictly inside their octagon (exact test) the excluded-row key. Boundary and collinear
  rows are kept, so the hull is bit for bit the same with `true` or `false`. `GPUSort` has a fixed
  length and cannot be skipped from the GPU, so the three sorts still cover every row; the saving is in
  the prefilter and chain, which skip the discarded run. The extreme tables hold 16 words per group, so
  forcing it on above 65536 groups throws.
- `prefilterLevels` (0 to 4, `GPU_GROUP_CONVEX_HULL_MAXIMUM_PREFILTER_LEVELS`; default 2 from 8192 rows)
  discards points inside nested octagon-like filters before the chain. For one 500k-point disk the median
  went from 551 ms (level 0) to 36, 8.8 and 7.6 ms (levels 1 to 3); 64 groups of 500k: 33 to 7.3 ms. With
  4096 small groups the levels cost 2 to 3 ms (3.0 ms at level 0), so set 0 there.
- The chain is one invocation per group, so a group whose points are all on the hull stays serial. For a
  concave outline, close a raster `GPUDistanceField`; exact alpha shapes stay on the CPU.

#### How it computes

{/* lens:GPUGroupConvexHull */}

::lens{scenario=gpu-spatial-analysis.group-convex-hull rung=datum bind=page datum=1000 caption="Inspect observation 1000: its cluster label, the group whose hull it may bound."}
::lens{scenario=gpu-spatial-analysis.group-convex-hull rung=formula caption="Follow the exact lattice orientation test that keeps only counter-clockwise hull turns."}
::lens{scenario=gpu-spatial-analysis.group-convex-hull rung=kernel caption="Inspect the lattice, extreme-point filter, sort and monotone-chain kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.group-convex-hull rung=graph caption="Trace DBSCAN labels, three sort passes, the per-group chain and the hull vertex emission."}

### `GPUNeighborhoodSummary`

Summarise each tract's neighbourhood from a spatial weights table, returning a row-standardised mean of values and the entropy of neighbour labels.

::lens{scenario=gpu-spatial-analysis.neighborhood-summary rung=result caption="Mean poverty of each tract's queen neighbours across 791 tracts, from row-standardised weights; range 4.2 to 57.5%."}

Per-row summaries of a neighborhood over a `GPUSpatialWeights` CSR (momepy `describe`, generalized and
categorical spatial lag). The members of row `i`, in fixed order, are the focal row (with
`includeFocal`, weight `focalWeight`, default 1) and then the active CSR slots in slot order. A slot
is active when its weight is positive; zero-weight slots, a listed self slot, out-of-range slots,
zero-mask neighbors and, for numeric statistics, non-finite values are skipped. Zero-weight slots are
therefore not members (`count`, `min`, `max`, `median` and the categorical outputs ignore them),
which differs from libpysal `lag_spatial` and momepy when zero weights are listed. When
`weights.information.transform` is `'row'` the slot weights are divided by the sum of the active slot
weights, so `sum` (the lag) stays a weighted average when neighbors are masked or non-finite (the
focal weight is unchanged). Numeric and categorical members use separate normalization denominators:
a neighbor with a missing numeric value can still contribute its category. Category `0xffffffff`
is reserved for missing category/no mode and is excluded from categorical membership, including
the focal row; its numeric value can still contribute. Category masses are summed before the
common normalization, with focal mass accounted for separately. Category comparisons retain a
renormalized sum and its addition residual, so differently partitioned equal masses reach the
lowest-category tie rule instead of being ordered by one rounding error.
Both forms suppress every included row's numeric summaries, including `count`, and entropy to
NaN and its mode to `GPU_NEIGHBORHOOD_SUMMARY_NO_MODE` when supported upstream evidence reports loss.
A masked-out focal row keeps `count` 0, other numeric statistics and entropy NaN, and the same
no-mode sentinel. Clearing loss and restoring supplied convergence restores results on the next run.

The pure CPU helper `computeNeighborhoodSummaryOnCPU` reads the producer’s supported evidence.
At matrix or chunk scope, nonzero `status.overflow` or `status.candidateOverflow`, zero
`status.complete`, or a `status.requiredCount` that exceeds the retained extent suppresses included
rows. The helper checks the required count against available terminal offsets, `status.count`, and
neighbor capacity. At matrix scope, nonzero `status.iterationLimitReached` or zero `status.converged` also
suppresses included rows. The top-level `weightsIterationLimitReached` and `weightsConverged`
inputs use the same polarity, and loss reported through either input takes effect. The helper
rejects chunk-local stopping fields and rejects `status.unresolvedCount`, `status.invalidCount`
and `status.approximation` at either scope.

The direct `GPUNeighborhoodSummary` reads `overflow` and `candidateOverflow` at matrix or chunk
scope and `converged` and `iterationLimitReached` from the outer `weights.status`. Nonzero overflow
or iteration-limit flags or zero convergence suppress included rows. Construction rejects
`complete`, `requiredCount`, `unresolvedCount`, `invalidCount` and `approximation` at either scope,
and rejects chunk-local `converged` and `iterationLimitReached`.

The named graph forms, `GPUNeighborhoodSummaryOperation` and `CPUNeighborhoodSummaryOperation`, bind
`overflow`, `candidateOverflow`, `converged` and `iterationLimitReached` at matrix scope. Both reject
those bindings on child chunks, so pass aggregate flags covering the whole matrix in `weights.status`.
They reject the same five unsupported evidence fields as the direct GPU form at either scope.
For these fields, all three graph forms check own properties and inherited getters. Use
`computeNeighborhoodSummaryOnCPU` with the intact CPU result when it carries `status.complete`.

For two adjacent rows with values `[10, 20]` and no focal inclusion, complete neighborhoods have
count/mean rows `[1, 20]` and `[1, 10]`. Candidate loss makes both pairs NaN; masking the second
focal row instead keeps its count zero and its mean NaN. Restoring complete neighborhoods and
clearing loss recovers the two original rows. The separate GPU median-overflow word is not a
certificate that the upstream neighborhoods are complete.
With `k` members, weights `w` and values `x`:

- `count` is `k`, `weightSum` is `W`, `sum` is `sum w x` (the lag) and `mean` is `sum / W` (NaN when
  `W = 0`). The public mean uses these public sums; relative centering is retained separately
  for variance and does not reconstruct the published mean.
- `min` and `max` are unweighted. `standardDeviation` is the weighted population deviation about the
  mean, NaN when `W = 0`. Variance retains the mean in relative coordinates around a valid
  member, so rounding a large absolute mean does not change the spread.
- `median` is unweighted (the mean of the two middle values for even `k`) for rows with at most
  `maximumNeighbors` members (default 32, at most 64). Larger rows give a NaN median and set
  `output.status.overflow`, which is required when `median` is requested; their other summaries
  remain available. Optional overflow output is rewritten as zero even without a median request.
- `output.modes` (`uint32`) is the category with the largest weighted frequency, ties to the lowest category
  (`GPU_NEIGHBORHOOD_SUMMARY_NO_MODE` for none). `output.entropy` is the Shannon entropy (natural log) of the
  weighted category shares. Categorical cost is `O(k^2)` per row.

`statistics` fixes the order of the `output.table` columns (no repeats): column `c` of row `i` is at
`i * statistics.length + c`. `values` may be `float32` or `uint32` counts (one conversion node,
exact below 2^24). The numeric and categorical parts are two kernels, each below the eight
storage buffer limit. Sums run in member order in f32, so results are deterministic.
The CPU twin rounds `focalWeight` to the validated float32 prop before float64 arithmetic; a
zero-weight focal row is a member for counts/extrema/median but cannot define a weighted mean
by itself. Finite extrema and even medians cover the full float32 value range. Weighted sums,
weighted central squares and row-normalization sums must remain representable in float32 on
the GPU; rescale values or weights if these intermediates would overflow. CPU medians remain
uncapped; GPU median overflow reports only that member limit, while upstream completeness
flags explain suppressed input-dependent results.

```ts
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  GPUNeighborhoodSummary, type GPUNeighborhoodSummaryProps,
  computeNeighborhoodSummaryOnCPU, type CPUNeighborhoodSummaryInput
} from '@luma-contributors/analysis/gpu-spatial-analysis';

export function addNeighborhoodExample(
  graph: GPUCommandGraph,
  inputs: Pick<GPUNeighborhoodSummaryProps, 'weights'> &
    Required<Pick<GPUNeighborhoodSummaryProps, 'values' | 'categories'>>,
  output: Required<Pick<GPUNeighborhoodSummaryProps['output'], 'table' | 'modes' | 'entropy'>> &
    {status: {overflow: GraphDataView<'uint32'>}}
) {
  // values/categories: N; table: N*4 floats; modes/entropy: N; median overflow: one word.
  graph.add(new GPUNeighborhoodSummary({
    ...inputs, output, includeFocal: true,
    statistics: ['count', 'mean', 'standardDeviation', 'median']
  }));
}

export function summarizeProducerOnCPU(
  producer: CPUNeighborhoodSummaryInput, values: Float32Array
) {
  return computeNeighborhoodSummaryOnCPU(
    {...producer, values}, // preserve the intact producer's flat/nested completeness evidence
    {statistics: ['count', 'mean', 'median']}
  );
}
```

#### How it computes

{/* lens:GPUNeighborhoodSummary */}

::lens{scenario=gpu-spatial-analysis.neighborhood-summary rung=datum bind=page datum=420 caption="Inspect tract 420: its neighbours, their weights and the weighted mean."}
::lens{scenario=gpu-spatial-analysis.neighborhood-summary rung=formula caption="Follow the neighbour mean and label entropy: weighted sums over the active slots."}
::lens{scenario=gpu-spatial-analysis.neighborhood-summary rung=kernel caption="Inspect the numeric and categorical kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.neighborhood-summary rung=graph caption="Trace queen weights, row standardisation and the neighbourhood summary kernels."}

### `GPUEmpiricalBayesRates`

Shrink each tract's crude rate toward the pooled rate according to its population, returning smoothed rates, raw rates and moments.

::lens{scenario=gpu-spatial-analysis.empirical-bayes-rates rung=result caption="Crude and empirical-Bayes poverty rates of 791 tracts around a pooled 25.3%; small populations shrink the most."}

Global empirical-Bayes standardization and smoothing of rates (esda `assuncao_rate`, as used by
`Moran_Rate` and `Moran_Local_Rate` with `adjusted=True`, and `Empirical_Bayes`). Over the included rows
(nonzero optional `mask`, finite nonnegative event count `e`, finite population `b > 0`; esda gives NaN for `b = 0`, here the
row is excluded): `y = e / b`, `m = sum e / sum b`, `s2 = sum b (y - m)^2 / sum b`, `a = s2 - m / mean(b)`.

Rows require finite nonnegative events, finite positive populations and a nonzero supplied
mask. Excluded rows output NaN. Optional `output.status.invalidCount` counts unmasked invalid
event/population rows, including negative events. Events `[-10,2,2,2]`, populations `[10,10,10,10]`
give `[NaN,.2,.2,.2]`.

`events` and `populations` are `float32` or `uint32` (for example the `uint32` `counts` of
`addPointsInPolygonsChoroplethRecipe`; one conversion node, exact below 2^24). Outputs go in `output`:

- `output.standardizedRates`: `z = (y - m) / sqrt(v)` with `v = a + m / b`, replaced by `m / b` when negative
  (esda's guard).
- `output.smoothedRates`: `r = w y + (1 - w) m` with `w = a / (a + m / b)`. By default (`clampPriorVariance: true`)
  `a` is clamped at 0 (Marshall 1991, GeoDa), so `w` stays in `[0, 1]` and `a <= 0` (counts no more dispersed
  than Poisson, common for small `n` or low counts) shrinks every rate fully to `m`. esda `Empirical_Bayes`
  does not clamp, so with `a < 0` its `w` leaves `[0, 1]` and rates are pushed away from `m`, possibly below
  zero. Pass `clampPriorVariance: false` for exact esda parity. This is a default change.
- `standardizedRates` never clamps `a`; it keeps esda's `assuncao_rate` guard (`v = m / b` when `a + m / b < 0`).
- Optional `output.rawRates` and `output.summary` (`GPU_EMPIRICAL_BAYES_SUMMARY`, 8 floats: count, event sum,
  population sum, pooled rate, weighted rate variance, prior variance, clamped prior variance). The
  `priorVariance` slot stays unclamped (esda `ebi_a`); `clampedPriorVariance` is `max(a, 0)`.

Sums are fixed-order two-level trees, so results are bitwise reproducible. Excluded rows are NaN, which
`GPULocalMoran`, `GPUGlobalSpatialStatistics` and the permutation tests exclude. Write
`standardizedRates` to a transient view and pass it as `values`. For the neighborhood-pooled variant see
[`GPUSpatialEmpiricalBayesRates`](#gpuspatialempiricalbayesrates).

```ts
import type {Device} from '@luma.gl/core';
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUEmpiricalBayesRates, type GPUEmpiricalBayesRatesProps, GPUGlobalSpatialStatistics, type GPUGlobalSpatialStatisticsProps
} from '@luma-contributors/analysis/gpu-spatial-analysis';
declare const graph: GPUCommandGraph<void>;
declare const device: Device;
declare const events: NonNullable<NonNullable<GPUEmpiricalBayesRatesProps>['events']>;
declare const populations: NonNullable<NonNullable<GPUEmpiricalBayesRatesProps>['populations']>;
declare const z: NonNullable<NonNullable<NonNullable<GPUEmpiricalBayesRatesProps>['output']>['standardizedRates']>;
declare const weights: GPUGlobalSpatialStatisticsProps['weights'];
declare const summary: NonNullable<GPUGlobalSpatialStatisticsProps['output']['summary']>;

graph.add(new GPUEmpiricalBayesRates({events, populations, output: {standardizedRates: z}}));
graph.add(
  new GPUGlobalSpatialStatistics({weights, values: z, statistics: ['moran'], output: {summary}})
);
```

#### How it computes

{/* lens:GPUEmpiricalBayesRates */}

::lens{scenario=gpu-spatial-analysis.empirical-bayes-rates rung=datum bind=page datum=420 caption="Inspect tract 420: its crude rate, shrinkage weight and smoothed rate."}
::lens{scenario=gpu-spatial-analysis.empirical-bayes-rates rung=formula caption="Follow the shrinkage: a weighted mix of the tract rate and the pooled rate."}
::lens{scenario=gpu-spatial-analysis.empirical-bayes-rates rung=kernel caption="Inspect the partial sums, totals, variance and per-tract rates kernels."}
::lens{scenario=gpu-spatial-analysis.empirical-bayes-rates rung=graph caption="Trace the two-pass reduction into the summary and the per-tract rates."}

### `GPUSpatialEmpiricalBayesRates`

Invalid focal rows output NaN; invalid donors are excluded. The spatial smoother exposes no
invalidCount port; use its eligibility rules and NaN focal outputs. Nonfinite or negative events
and nonfinite/nonpositive populations are invalid.

Pool each tract with its queen neighbours into a local prior, returning spatial rates and shrunken rates per tract.

::lens{scenario=gpu-spatial-analysis.spatial-empirical-bayes-rates rung=result caption="Spatial empirical-Bayes poverty rates of 791 tracts pooled over queen neighbourhoods, spanning 2.8% to 75.9%."}

Neighborhood-pooled rate smoothing (esda 2.10 `Spatial_Rate` and `Spatial_Empirical_Bayes`, libpysal 4.15
weights). The neighborhood of an included row `i` is `i` plus its active neighbors (slots with positive
weight to another row, the shared neighbor rule) that are included
(nonzero optional `mask`, finite nonnegative `events`, finite positive `populations`), at both focus and donor rows; only the weights' structure counts (esda sets
`transform = 'b'`; a slot stored with weight 0 is not a neighbor here). With `E = sum e`, `B = sum b` and `n` members: the spatial rate is `E / B`
(`Spatial_Rate.r`), the local prior mean is `m = E / B`, the local variance is
`s2 = sum b (e / b - m)^2 / B`, `a = max(s2 - m / (B / n), 0)`, and the smoothed rate is
`w e_i / b_i + (1 - w) m` with `w = a / (a + m / b_i)`. `events` and `populations` are `float32` or
`uint32`. Outputs are the optional `output.spatialRates`, `output.smoothedRates`, `output.priorMeans` and
`output.priorVariances` (NaN for excluded rows). One invocation reduces each row
over its ascending slots, so results are bitwise reproducible. An eligible isolate retains its own
rate; an invalid focus is NaN. Zero event counts remain eligible. For events `[6,-2]`, populations
`[30,10]` and reciprocal unit edges, the spatial rates are `[.2,NaN]`: the negative-event row is
excluded both as a focus and as a donor. The spatial-rate recipe's default `islandPolicy: 'undefined'`
instead returns `[NaN,NaN]`; its `'own-rate'` policy retains `[.2,NaN]`.

If `weights.status.overflow` or `weights.status.candidateOverflow` is nonzero on the outer
weights or any CSR chunk, the neighborhoods are incomplete. Every requested rate and prior
output is NaN until all supplied loss flags are zero. The weight structure and loss views are
captured at construction; replacing the caller's status object cannot remove that guard.
The retained GPU buffer contents remain inputs to each submission.

The rate-smoothing showcase also withholds the selected county's neighborhood-pooled readout
and clears its neighbor outline while either weights loss flag is set. The county's own births,
exposure and raw rate remain available; the neighborhood readout and outline recover when both
flags clear.

esda 2.10.0 `Spatial_Empirical_Bayes` itself sets `r_var[r_var < 0] = 0`, so the clamp matches esda and needs
no option (unlike the global `Empirical_Bayes`, which does not clamp; see
[`GPUEmpiricalBayesRates`](#gpuempiricalbayesrates)). The spec pins esda 2.10.0 on a 6x6 queen lattice (spatial
rate within 1e-5 relative, smoothed within 2e-4 in f32) and on low-dispersion counts where the local `a`
clamps. Deviations: a neighborhood with no events gives `0 / 0 = NaN` in esda and 0 here; rows excluded by the
mask or a non-positive population are skipped as neighbors instead of raising; a neighbor ID equal to the
focus row is skipped, where esda would count it twice.

```ts
import type {Device} from '@luma.gl/core';
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUSpatialEmpiricalBayesRates, type GPUSpatialEmpiricalBayesRatesProps
} from '@luma-contributors/analysis/gpu-spatial-analysis';
declare const graph: GPUCommandGraph<void>;
declare const device: Device;
declare const events: NonNullable<NonNullable<GPUSpatialEmpiricalBayesRatesProps>['events']>;
declare const populations: NonNullable<NonNullable<GPUSpatialEmpiricalBayesRatesProps>['populations']>;
declare const weights: NonNullable<NonNullable<GPUSpatialEmpiricalBayesRatesProps>['weights']>;
declare const smoothedRates: NonNullable<NonNullable<NonNullable<GPUSpatialEmpiricalBayesRatesProps>['output']>['smoothedRates']>;

graph.add(new GPUSpatialEmpiricalBayesRates({events, populations, weights, output: {smoothedRates}}));
```

#### How it computes

{/* lens:GPUSpatialEmpiricalBayesRates */}

::lens{scenario=gpu-spatial-analysis.spatial-empirical-bayes-rates rung=datum bind=page datum=420 caption="Inspect tract 420: its neighbourhood totals, local prior mean and smoothed rate."}
::lens{scenario=gpu-spatial-analysis.spatial-empirical-bayes-rates rung=formula caption="Follow the local shrinkage: the tract rate mixed with its neighbourhood prior mean."}
::lens{scenario=gpu-spatial-analysis.spatial-empirical-bayes-rates rung=kernel caption="Inspect 791 invocations, each summing its own and its neighbours' events and populations."}
::lens{scenario=gpu-spatial-analysis.spatial-empirical-bayes-rates rung=graph caption="Trace queen weights into the neighbourhood-pooled smoothing kernel."}

### `GPUEmergingHotSpots`

Classify a space-time cube into ArcGIS emerging hot and cold spot categories, returning Gi* z-scores, Mann-Kendall trends and categories.

::lens{scenario=gpu-spatial-analysis.emerging-hot-spots rung=result caption="Emerging hot spot category of 224 cells over twelve months; all 224 are no pattern: the 35 cells that would be sporadic hot under the older rule have no significant final month."}

ArcGIS-style emerging hot spot analysis over a dense space-time cube, on a regular lattice or on any
spatial weights.
`values` is indexed `cell * sliceCount + slice` (the `GPUTemporalReduction` layout, `cell = row *
gridWidth + column`); a `float32` view treats NaN as a missing bin, a `uint32` view (for example
`GPUTemporalReduction` `counts`) is read as `f32(count)`. An optional per-cell `mask` excludes cells.
`cellCount * sliceCount` must fit one storage binding (2^25 bins at the default 128 MiB
`maxStorageBufferBindingSize`); `getCommandNodes` throws above it.

Keep cell identities and period order fixed across the entire cube and the weights rows. Joining
independently sorted occupied-cell tables is insufficient: use one cell dictionary and one period
axis, then build the dense cube in that order. For dictionary `[20, 10]`, events in cells
`[10, 20, 20]` at periods `[0, 1, 1]` give counts `[0, 2, 1, 0]`. An observed empty bin is zero;
an unknown observation is float NaN and is excluded from the population. When gathering sparse
unsigned counts, set `GPUKeyLookupGather.unmatchedValue: 0` for observed-empty bins; its default
unsigned no-match sentinel is not a missing observation to this contributor. For `[0, 2, 1, 0]`,
the independent population count is 4, mean is 0.75 and variance is 0.6875.

Pass exactly one neighborhood: `gridWidth` and `gridHeight` for the lattice, with a per-frame
`radius` in cells, or `weights` over the cells (rows are cells) plus a compile-time `selfWeight`,
which works on H3 cells, polygons or points. With binary weights equal to the lattice disc, weights
mode reproduces lattice mode.

```ts
import type {Device} from '@luma.gl/core';
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
import {
  GPUEmergingHotSpots, type GPUEmergingHotSpotsProps,
  GPU_EMERGING_HOT_SPOTS_PARAMETER_LENGTH
} from '@luma-contributors/analysis/gpu-spatial-analysis';

export function addEmergingExample(
  graph: GPUCommandGraph, device: Device,
  inputs: Pick<GPUEmergingHotSpotsProps, 'values' | 'output'>
) {
  // values/giZScores/p-values: 64*64*24; trends/category/counts/lastValidSlice: 64*64.
  // Optional globalStatistics: four float32 words; includedCount: one uint32 word.
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'emerging', format: 'float32', length: GPU_EMERGING_HOT_SPOTS_PARAMETER_LENGTH
  });
  const hotSpots = new GPUEmergingHotSpots({
    ...inputs, gridWidth: 64, gridHeight: 64, sliceCount: 24, maximumRadius: 4,
    parameters: parameterBuffer.importToGraph(graph)
  });
  graph.add(hotSpots);
  const settings = {radius: 2, temporalWindow: 3, confidenceLevel: 0.95};
  parameterBuffer.write(hotSpots.getParameterValues(settings));
  return {hotSpots, parameterBuffer, metadata: hotSpots.getInferenceMetadata(settings)};
}
```

Defaults changed: `multipleTesting` is `'benjamini-hochberg'` (it was no correction), `confidenceLevel` is 0.95
(it was 0.9), the trend level is `1 - confidenceLevel`, `minimumTrendSliceCount` is 10, and
`finalSlice` is `'require'`. Pass `multipleTesting: 'none'` for the previous rule.
Use `getGPUEmergingHotSpotsCriticalZ(confidenceLevel)` to look up the two-sided normal critical z
for 0.9, 0.95 or 0.99. The lookup rejects other levels.

Passes (deterministic, no float atomics):

1. Global `n`, mean and population standard deviation of valid bins: fixed workgroup trees over
   4096-bin blocks, followed by a workgroup tree over block partials. A second-pass mean correction
   and stored centered deviations keep an additive offset from destroying the variance and Gi*.
2. Space-time Gi* per bin: the spatial neighbors (lattice cells with `dx^2 + dy^2 <= radius^2`, or
   the cell's weights row plus the cell itself at `selfWeight`) over the current and `temporalWindow`
   previous slices, `z = sum w (x - X) / (S sqrt((n S1 - W^2) / (n - 1)))` over valid bins.
   The weighted route scales each slice's coefficients by a power of two. Its centered coefficient
   moments use a verified strict majority among usable coefficients in each slice when comparable
   to the positive focal weight. Otherwise they use the smallest comparable usable coefficient,
   with the focal coefficient as fallback. Masked and non-finite neighbor bins cannot choose the
   scale or origin. The temporal merge converts each slice's weighted sum and coefficient moments
   to the largest scale in the window and a common origin before combining them. The weighted route
   uses compensated sums for the centered coefficient squares and population residual mean.
   It carries the rounding corrections for both weighted and unweighted deviations through the
   temporal merge. Zero coefficients for valid bins outside
   the neighborhood/window are included, avoiding subtraction of large nearly equal squared sums.
   Exactly uniform weights over the full population have zero spread and undefined Gi*;
   nearly uniform weights retain their small positive spread and valid scores.
   The numerator removes the same common coefficient before summation. Over the complete
   population that coefficient cancels exactly, so long unit-weight rows sum only the exceptional
   coefficients. Power-of-two rescaling of the near-uniform weights preserves these scores.
3. Two-sided normal p-values `erfc(|z| / sqrt(2))` of every valid Gi* bin, adjusted by
   `GPUMultipleTestingAdjustment` over the whole family of valid bins of the cube (never a display
   mask). A bin is hot or cold when its adjusted p-value is at most `1 - confidenceLevel`, with the
   sign of z. With `multipleTesting: 'none'` the rule is `|z| >= criticalZ` with the same two-sided
   critical values, and the p-value passes are skipped unless `output.pValues` or
   `output.adjustedPValues` is requested. Bins within a temporal window overlap and neighboring cells share observations, so the
   tests are dependent: Benjamini-Hochberg is only guaranteed under positive dependence, and
   `'benjamini-yekutieli'` controls the false discovery rate under arbitrary dependence at a cost in
   power of about `ln m`. The cube needs fewer than 2^31 bins and every binding must also fit the
   device's `maxStorageBufferBindingSize`; the device limit is normally tighter.
4. Per-cell Mann-Kendall on the finite z series: exact integer `S`, tie-corrected variance
   `[n(n-1)(2n+5) - sum t(t-1)(2t+5)] / 18` (ties by exact f32 equality), continuity-corrected z, and a
   two-sided p-value from the Numerical Recipes `erfcc` fit. Consecutive Gi* values share
   `temporalWindow` slices, so the series is serially correlated and this p-value is anti-conservative
   (ESRI has the same property): treat the trend level as a screening threshold. Cells with fewer
   than `minimumTrendSliceCount` finite slices (default 10, ESRI guidance) get NaN `trendZ` and
   `trendP`.
5. Classification into the 17 `GPU_EMERGING_HOT_SPOTS_CATEGORIES` (0 = no pattern, hot 1..8, cold 9..16:
   new, consecutive, intensifying, persistent, diminishing, sporadic, oscillating, historical) plus
   `UNDETERMINED` (17).

Sporadic follows current ArcGIS Pro: a sporadic hot (cold) spot has a statistically significant hot
(cold) final slice, with fewer than 90% hot (cold) slices and no cold (hot) slice. A cell with only
hot (or only cold) history and a non-significant final slice is therefore `NO_PATTERN`, not sporadic.
Earlier versions labelled such a cell sporadic, and the default changed. Pass
`sporadicRequiresFinalSlice: false` to the packer (a per-frame setting, no rebuild) to reproduce results
computed under the older rule; it changes only that neutral-final case.

`UNDETERMINED` marks a cell whose category cannot be decided: with `finalSlice: 'require'` (the
default) a cell with data whose last slice is missing, and a cell that is persistent in one direction
(Intensifying, Persistent or Diminishing, hot or cold) but has fewer finite slices than
`minimumTrendSliceCount`. It is never silently reported as Persistent. `finalSlice: 'last-valid'`
keeps the old behavior, where the last finite slice stands in for the final slice, so a stale cell can
be labelled New or Persistent for a period it has no data in. Cells without any finite slice stay
`NO_PATTERN`. The optional `output.lastValidSlice` gives each cell's last finite slice index
(`GPU_EMERGING_HOT_SPOTS_NO_SLICE`, `0xffffffff`, for none).
A missing final slice alone does not suppress the trend: `trendZ` and `trendP` remain defined
when there are enough finite slices, even when the category is `UNDETERMINED`.

Topology options, output routes and neighborhood descriptors (including chunk and status handles)
are snapshotted at construction. Buffer contents remain per-frame inputs. If any supplied
`weights.status.overflow` or `candidateOverflow` flag is nonzero, including aggregate or nested
chunk flags, every Gi* bin, raw/adjusted p-value, `trendZ` and `trendP` is NaN,
even when `minimumTrendSliceCount` is zero. Every cell publishes category 17 (`UNDETERMINED`),
zero `trendS`, zero hot/cold slice counts and `GPU_EMERGING_HOT_SPOTS_NO_SLICE` for
`lastValidSlice`. On complete neighborhoods, all-missing cells retain `NO_PATTERN` (category 0)
alongside finite cells with no qualifying pattern. Category 0 is not a missing-data test:
a finite, nonsignificant Gi* series such as `[0.3, 0.2, 0.1]` can also produce it. Clearing all loss flags restores scores,
trends, counts and categories on the next submission of the same graph. Independent input
`globalStatistics` and exact `includedCount` (or its `count` alias) remain available because
they do not depend on neighborhood edges.

Legend contract: category codes never change meaning with `multipleTesting`, `confidenceLevel` or
`finalSlice`; those options only change which bins are hot or cold and which cells are undetermined.
`getInferenceMetadata(settings)` returns a plain `GPUInferenceMetadata` for the legend, derived from the
same `GPUEmergingHotSpotsSettings` that `getParameterValues(settings)` packs. With an explicit
`criticalZ` and no correction, `significanceLevel` is the two-sided normal tail of the encoded
float32 critical value (`criticalZ: 1` gives `0.3173105079`); otherwise it is
`1 - confidenceLevel`. The separately configured trend level is unchanged. The space-time
recipe snapshots each settings update once before packing and publishing metadata, including
accessor-backed settings. `getParameterValues` takes `multipleTesting` from the node and rejects a
different `settings.multipleTesting`, so the packed values and the legend cannot disagree.

`criticalZ` is only accepted with `multipleTesting: 'none'` (the parameter packer throws otherwise).
Without a correction `confidenceLevel` must be 0.9, 0.95 or 0.99 or `criticalZ` must be given; with a
correction any level in `(0, 1)` works.

The parameter packer also checks the encoded float32 domain. Radii, windows and minimum trend
lengths must stay finite; `criticalZ` must stay positive; trend and bin significance levels must
stay strictly between 0 and 1; and `persistentFraction` must stay above 0.5 and at most 1. Values
that overflow, underflow to zero, or round onto a forbidden boundary throw before writing the target.
Unsupported `multipleTesting` names are rejected by both the packer and the contributor.

Per-frame parameters (radius, temporal window, confidence level, critical z, trend significance,
persistence fraction, minimum trend length, the Sporadic rule) live in a parameter buffer and never rebuild the graph.
Compile-time: `gridWidth`, `gridHeight`, `sliceCount` (at most 256), `maximumRadius` (at most 32),
`multipleTesting`, `finalSlice`, value format, optional views. Outputs are caller-owned fields of
`output` (`GPUEmergingHotSpotsOutput`): `giZScores` (per bin), `trendZ`, `trendP`, `trendS`, `category`,
`hotSliceCount`, `coldSliceCount` (per cell), optional `pValues` and `adjustedPValues` (per bin),
optional `lastValidSlice` (per cell) and optional `globalStatistics`
`[n, mean, variance, standardDeviation]`. Optional `includedCount` is the exact uint32 count of
finite bins in unmasked cells, including bins whose Gi* is undefined; `count` is its legacy alias.
Empty input publishes four zero moments and count zero. Constant input has zero variance and
undefined Gi*, rather than a zero z-score.

#### How it computes

{/* lens:GPUEmergingHotSpots */}

::lens{scenario=gpu-spatial-analysis.emerging-hot-spots rung=datum bind=page datum=100 caption="Inspect cell 100: its trend statistics, hot and cold month counts and category."}
::lens{scenario=gpu-spatial-analysis.emerging-hot-spots rung=formula caption="Follow space-time Gi*, the Mann-Kendall trend and the adjusted p-value rule for hot and cold months."}
::lens{scenario=gpu-spatial-analysis.emerging-hot-spots rung=kernel caption="Inspect the moment, Gi*, adjustment, Mann-Kendall and classification kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.emerging-hot-spots rung=graph caption="Trace the global moments, Gi* per bin, Benjamini-Hochberg adjustment and per-cell classification."}

{/* parameters: GPUEmergingHotSpots generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUEmergingHotSpots` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPUEmergingHotSpots`.

- Packer: `getGPUEmergingHotSpotsParameterValues(settings, target?, id?)`, writes `GPU_EMERGING_HOT_SPOTS_PARAMETER_LENGTH` (8) `float32` values.
  - `target` (`Float32Array`): Optional destination of at least 8 elements. A new array is returned when omitted.
  - `id` (`unknown`): Error-message identity, supplied by the contributor for instance-specific diagnostics.
- Packer: `instance.getParameterValues(settings)`, writes the `float32` parameter view.

Settings of `getGPUEmergingHotSpotsParameterValues`:

| Field | Type | Description |
|---|---|---|
| `radius` (optional) | `number` | Lattice mode: spatial neighborhood radius in lattice cells. Cell `(dx, dy)` offsets with `dx * dx + dy * dy <= radius * radius` are neighbors (the focal cell included, so `0` is the focal cell alone). Clamped to the compile-time `maximumRadius`. Ignored in weights mode. Defaults to `0`. |
| `temporalWindow` | `number` | Temporal window `k >= 0`: neighbors span the current slice and the `k` previous slices. |
| `confidenceLevel` (optional) | `number` | Confidence level of hot and cold bins, in `(0, 1)`. A bin is hot or cold when its adjusted two-sided p-value is at most `1 - confidenceLevel`. With `multipleTesting: 'none'` only `0.9`, `0.95` and `0.99` (the tabulated critical z values) or an explicit `criticalZ` are accepted. Defaults to `0.95` (it was `0.9`). |
| `criticalZ` (optional) | `number` | Explicit critical z that overrides `confidenceLevel` for hot and cold bins. Only meaningful without a correction: it requires `multipleTesting: 'none'` and throws otherwise, because a fixed critical z cannot express a family-wise or false discovery rate threshold. |
| `multipleTesting` (optional) | `GPUMultipleTestingMethod` | The compile-time `multipleTesting` of the node these settings feed. It does not change the packed values; it makes the packer reject a `criticalZ` combined with a correction and decide whether `confidenceLevel` must be tabulated. Defaults to `'benjamini-hochberg'`, like the node. `GPUEmergingHotSpots.getParameterValues` fills it from the node and rejects a different value. |
| `trendSignificanceLevel` (optional) | `number` | Two-sided p-value at or below which the Mann-Kendall trend is significant, in `(0, 1)`. Defaults to `1 - confidenceLevel`. Consecutive Gi* z values share `temporalWindow` slices, so the series is serially correlated and this p-value is anti-conservative (ESRI has the same property); treat it as a screening threshold. |
| `minimumTrendSliceCount` (optional) | `number` | Cells with fewer finite slices than this get NaN `trendZ` and `trendP`, and a persistent cell that would need the trend decision gets `UNDETERMINED`. A non-negative integer; `0` disables the guard. Defaults to 10, ESRI's guidance for Mann-Kendall. |
| `persistentFraction` (optional) | `number` | Fraction of slices that makes a spot persistent or historical, in `(0.5, 1]`. Default `0.9`. |
| `sporadicRequiresFinalSlice` (optional) | `boolean` | Whether Sporadic hot and cold need a statistically significant hot (cold) final slice, as current ArcGIS Pro defines them. Defaults to `true`. `false` restores the older rule that also labels a cell with a non-significant final slice Sporadic when it has only hot (cold) history, so results computed before the change can be reproduced. Only that neutral-final case changes; a location that fails the new test is `NO_PATTERN`. |

{/* /parameters */}

### `addSpaceTimeHotSpotsRecipe`

Turn timestamped events into an emerging hot spot map, returning the month-by-cell cube, Gi* z-scores, trends and categories.

::lens{scenario=gpu-spatial-analysis.space-time-hot-spots-recipe rung=result caption="Hot spot categories of 224 cells from 2,344 timestamped observations over twelve months; none is sporadic hot because no cell has a significant hot final month."}

Timestamped events to an emerging hot spot map: `GPUCalendarBuckets` decodes the event times, an
adapter bins every event into a dense `cell * sliceCount + slice` cube, `GPUEmergingHotSpots`
classifies it and an ordinal `GPUColorScale` colors the categories (18 palette entries, 0 to 17).

For `cells.kind: 'lattice'`, optional `cells.spatialContext` describes positions and bounds, defaults to
native planar, and rejects longitude/latitude. `result.spatialContext` returns this resolved context;
with caller-supplied cell IDs it carries `cells.weights.spatialContext` when declared.

Slices are absolute time steps, the ArcGIS "time step interval", so the Mann-Kendall trend runs over
a monotone axis:

```ts
import type {Device} from '@luma.gl/core';
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
import {
  addSpaceTimeHotSpotsRecipe, type GPUSpaceTimeHotSpotsRecipeProps,
  GPU_EMERGING_HOT_SPOTS_PARAMETER_LENGTH
} from '@luma-contributors/analysis/gpu-spatial-analysis';

export function addSpaceTimeExample(
  graph: GPUCommandGraph, device: Device,
  inputs: Omit<GPUSpaceTimeHotSpotsRecipeProps,
    'parameters' | 'settings' | 'slices' | 'multipleTesting'>
) {
  // One encoded timestamp per event; inputs include the typed calendar and cell metadata.
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'space-time-hot-spots', format: 'float32', length: GPU_EMERGING_HOT_SPOTS_PARAMETER_LENGTH
  });
  const spaceTime = addSpaceTimeHotSpotsRecipe(graph, {
    ...inputs, parameters: parameterBuffer,
    settings: {temporalWindow: 1, radius: 1.5},
    slices: {step: 'month', interval: 3, origin: {year: 2022, month: 11}, count: 12},
    multipleTesting: 'benjamini-yekutieli'
  });
  spaceTime.setSettings({temporalWindow: 2, radius: 1.5});
  return {spaceTime, parameterBuffer};
}
```

Slice `floor((stepIndex(t) - stepIndex(origin)) / (interval * stepsPerUnit))` holds the events at
local time `t` (the same UTC-offset semantics as `GPUCalendarBuckets`). `stepIndex` counts whole
steps: `year`, `year * 4 + quarter - 1`, `year * 12 + month - 1`, days since the epoch for `'day'` and
`'week'` (a week is 7 days anchored to the origin day), hours since the epoch for `'hour'`. It is
integer arithmetic on the GPU (Hinnant's `days_from_civil`) and on the CPU
(`getSpaceTimeSliceIndex`). Events before the origin or past the last slice are skipped and counted
in the `slice-range` status stage (`invalidCount`, or the `output.status.skippedEventCount` override). The sorted path's
`cube-counts` stage also reports `requiredCount` (the `output.status.groupRequiredCount` destination).

Optional `timeZone: GPUCalendarBucketsTimeZone` forwards a transition table from
`getGPUTimeZoneTransitionTable` to `GPUCalendarBuckets`, which searches it per row on the GPU.
It is mutually exclusive with `utcOffsets`; setting both throws. With an `'hour'` step, local
slices on transition days are 0 h or 2 h long, so use UTC for hourly cubes.

Cyclic fields (month of year, weekday, hour of day, day of year) are no longer accepted: `month = 1`
would pool every January of a multi-year range, and a trend over hour or weekday is meaningless.
Cyclic profiles belong in a non-trend summary such as `GPUCalendarBuckets` `hourWeekdayCounts`.

`multipleTesting` (default `'benjamini-hochberg'`) and `finalSlice` (default `'require'`) pass through
to `GPUEmergingHotSpots`. The per-frame `settings` (`GPUSpaceTimeHotSpotsRecipeSettings`: the emerging
hot spot settings without `multipleTesting`) are packed by the recipe into its `parameters`
`GPUParameterBuffer`, and `result.inference` (a plain `GPUInferenceMetadata` object) is
`GPUEmergingHotSpots.getInferenceMetadata` of the same object; `result.setSettings` repacks both. The
result also has optional `pValues`, `adjustedPValues` and `lastValidSlice` outputs.

#### How it computes

{/* lens:addSpaceTimeHotSpotsRecipe */}

::lens{scenario=gpu-spatial-analysis.space-time-hot-spots-recipe rung=formula caption="Follow the binning: event time to month slice, position to lattice cell, counts per cell and slice."}
::lens{scenario=gpu-spatial-analysis.space-time-hot-spots-recipe rung=kernel caption="Inspect the calendar decode, cube binning and emerging hot spot kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.space-time-hot-spots-recipe rung=graph caption="Trace calendar decoding, the atomic cube count and the emerging hot spot chain."}

### `GPUGeographicDistribution`

Select at least one actual terminal view: counts, weightSums, meanCenters, medianCenters,
medianConverged, standardDistances, ellipses, directionalMeans (with lineEnds), ellipseVertices
or circleVertices. An empty output is invalid. Counts alone is a genuinely minimal request.

```ts
import type {Device} from '@luma.gl/core';
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUGeographicDistribution, type GPUGeographicDistributionProps}
  from '@luma-contributors/analysis/gpu-spatial-analysis';
declare const graph: GPUCommandGraph<void>;
declare const device: Device;
declare const positions: GPUGeographicDistributionProps['positions'];
declare const parameters: GPUGeographicDistributionProps['parameters'];
declare const counts: NonNullable<GPUGeographicDistributionProps['output']['counts']>; // One uint32 for one group.
const props: GPUGeographicDistributionProps = {positions, parameters, output: {counts}};
const plan = GPUGeographicDistribution.getExecutionPlan(props, device.limits);
if (!plan.feasible) throw new Error(JSON.stringify(plan.constraints.filter(check => !check.satisfied)));
graph.add(new GPUGeographicDistribution(props));
```

`reduction:'auto'` selects chunked through 64 groups, sorted above 64; this is a crossover,
not a maximum group count. Forced chunked accepts at most 256 groups over the whole domain.
Preflight with actual device limits checks selected-route bindings and allocation sizes,
including grouped partial-sum scratch, vertex products and median command budget. These can
fail before allocation even when the group count fits. A topology-only plan without limits
does not certify bindings. Both routes are deterministic; float32 sum association can differ.

Measure the weighted centre, spread and ellipse of tract centroids, returning mean centre, standard distance and ellipse per group.

::lens{scenario=gpu-spatial-analysis.geographic-distribution rung=result caption="Population-weighted standard distance of 791 Chicago tract centroids: 11,343 m in the local frame."}

`GPUGeographicDistribution` computes the ArcGIS "Measuring Geographic Distributions" statistics
over planar points, per group or per masked selection: counts, weight sums, mean centre, Weiszfeld
median centre, standard distance, standard deviational ellipse, linear directional mean, and
polygon rings for the ellipse and the standard-distance circle. Every statistic is Euclidean:
`spatialContext` (default native planar) rejects `longitude-latitude` at construction with a
"project positions to a planar frame first" error, and `GPURoseStatistic` does the same. A
spherical mean centre (normalised mean of 3D unit vectors) and a great-circle standard distance are
deferred, because they need a second implementation of every reduction pass in both the chunked and
the sorted path.

Declares `measure: 'distance'`: a Lambert equal-area local frame throws; use an
azimuthal-equidistant frame.

```ts
import type {Device} from '@luma.gl/core';
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
import {
  GPUGeographicDistribution, type GPUGeographicDistributionProps, GPU_GEOGRAPHIC_DISTRIBUTION_PARAMETER_LENGTH, getGPUGeographicDistributionParameterValues
} from '@luma-contributors/analysis/gpu-spatial-analysis';
declare const graph: GPUCommandGraph<void>;
declare const device: Device;
declare const positions: NonNullable<NonNullable<GPUGeographicDistributionProps>['positions']>;
declare const weights: NonNullable<NonNullable<GPUGeographicDistributionProps>['weights']>;
declare const groupIds: NonNullable<NonNullable<GPUGeographicDistributionProps>['groupIds']>;
declare const mask: NonNullable<NonNullable<GPUGeographicDistributionProps>['mask']>;
declare const meanCenters: NonNullable<NonNullable<NonNullable<GPUGeographicDistributionProps>['output']>['meanCenters']>;
declare const ellipses: NonNullable<NonNullable<NonNullable<GPUGeographicDistributionProps>['output']>['ellipses']>;
declare const ellipseVertices: NonNullable<NonNullable<NonNullable<GPUGeographicDistributionProps>['output']>['ellipseVertices']>;
declare const x0: number;
declare const y0: number;

const parameters = new GPUParameterBuffer(device, {
  id: 'distribution-parameters',
  format: 'float32',
  length: GPU_GEOGRAPHIC_DISTRIBUTION_PARAMETER_LENGTH
});
graph.add(
  new GPUGeographicDistribution({
    positions, weights, groupIds, groupCount: 8, mask,
    parameters: parameters.importToGraph(graph),
    output: {meanCenters, ellipses, ellipseVertices}
  })
);
parameters.write(
  getGPUGeographicDistributionParameterValues({origin: [x0, y0], standardDeviations: 2})
);
```

- Inputs: `positions` (`float32x2`, fewer than 2^24 rows), optional `weights` (a non-finite,
  negative or zero weight excludes the row), `groupIds` with compile-time `groupCount` (ids at or
  above it are excluded), `mask`, and `lineEnds` for the directional mean.
- Per-frame parameters (no recompile): `origin` (local origin subtracted before every sum for
  float32 stability), `standardDeviations`, `ellipseConvention` (`'root-two'` scales the axes by
  `sqrt(2)`, `'standard'` does not), `orientationOnly`, `medianTolerance`.
- Median centre: Weiszfeld steps from the mean centre, all groups together, for at most the
  compile-time `maximumMedianIterations` (default 24). Behavior change: the steps stop after the first one
  in which every non-empty group moved at most `medianTolerance` (a GPU-written indirect dispatch skips the
  rest), instead of always running `maximumMedianIterations` steps; `medianConverged` is then 1 for every
  non-empty group.
- Divisors are the weight sum, as in ArcGIS (not `n - 1`). Second moments are central (a mean
  pass, then a pass about the mean), never `E[x^2] - E[x]^2`. No small-sample degrees-of-freedom
  correction is applied, so for small `n` the spread is biased low by `sqrt((n - 1) / n)`; there is
  no confidence interval or significance test.
- `ellipses` holds `[angle, sigmaX, sigmaY]` per group. As in ArcGIS the "y" axis is the long one
  (`sigmaY >= sigmaX`). `angle` is counter-clockwise from +x of the short x axis, so the long axis
  points at `angle + pi / 2`; a ring vertex is `M + R(angle) (sigmaX cos t, sigmaY sin t)`.
  The larger eigenvalue uses a Jacobi correction. The smaller is the determinant divided by the
larger eigenvalue, with the determinant taken from a second, centered row pass: `p` is the
coordinate of larger variance, `r = s - a p` the compensated transverse residual with
`a = cov / var(p)`, and `det = var(p) * (var(r) - cov(r, p)^2 / var(p))`. Variance below the
float32 ulp of the rounded covariance entries is retained: points `±(100, 99) ± (1/64, 1/64)`
give a short axis of `0.000111039`, and the exactly collinear pair `±(1, float32(0.1))` gives
exactly 0, on both reduction routes. The pass adds one `groupCount * 20`-byte scratch binding.
- `directionalMeans` holds `[meanAngle, circularVariance, meanLength]` (angles counter-clockwise
  from +x). With `orientationOnly`, angles are doubled before averaging and the mean angle lies in
  `(-pi/2, pi/2]`.
- Nonfinite and zero-length lines are excluded from every directional sum, including mean length,
  while their valid start positions still contribute to the point statistics. Both reduction
  routes use the same inclusion rule. Weighted medians use the Vardi-Zhang correction at a
  coincident point; a coincident weight at least as large as the other points' pull fixes the iterate.
- Settings must remain finite after float32 packing, and `standardDeviations` must remain positive.
  Input coordinate/weight products, line squared lengths, moment sums and the covariance trace
  must remain finite in float32; rescale units when these intermediates would overflow. The
  ellipse discriminant scales its norm to avoid squaring an already large variance.
- `ellipseVertices` and `circleVertices` are counter-clockwise rings of `polygonVertexCount`
  (default 64) vertices per group, not closed, ready for a deck.gl `PolygonLayer` or `PathLayer`.
- The median centre runs at most `maximumMedianIterations` (default 24) Weiszfeld steps from the mean
  centre; `medianConverged` is 1 when the last step moved at most `medianTolerance`.
- Empty groups report 0 counts, weight sums and `medianConverged`, and NaN in the other outputs.
- Determinism: rows are sorted by group once and every sum, including each Weiszfeld step, is a
  fixed-order segmented tree sum. No float atomics; repeated encodings are bitwise identical.

#### How it computes

{/* lens:GPUGeographicDistribution */}

::lens{scenario=gpu-spatial-analysis.geographic-distribution rung=datum bind=page datum=420 caption="Inspect tract 420: its centroid and population weight enter every group sum."}
::lens{scenario=gpu-spatial-analysis.geographic-distribution rung=formula caption="Follow the standard distance and ellipse: weighted second moments about the mean centre."}
::lens{scenario=gpu-spatial-analysis.geographic-distribution rung=kernel caption="Inspect the moment passes, the spread kernel and the ellipse ring kernel."}
::lens{scenario=gpu-spatial-analysis.geographic-distribution rung=graph caption="Trace the weighted sums into the mean centre, spread and ellipse vertices."}

{/* parameters: GPUGeographicDistribution generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUGeographicDistribution` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPUGeographicDistribution`.

- Packer: `getGPUGeographicDistributionParameterValues(settings?)`, writes `GPU_GEOGRAPHIC_DISTRIBUTION_PARAMETER_LENGTH` (8) `float32` values.

Settings of `getGPUGeographicDistributionParameterValues`:

| Field | Type | Description |
|---|---|---|
| `origin` (optional) | `readonly [number, number]` | Local origin `[x, y]` subtracted from every position before any sum, so float32 sums keep precision when the planar coordinates are large (for example Web Mercator meters). Choose a point near the data. Defaults to `[0, 0]`. |
| `standardDeviations` (optional) | `number` | Number of standard deviations that scale `standardDistances`, `ellipses` and the vertex rings (ArcGIS offers 1, 2 and 3). Any positive number is accepted. Defaults to `1`. |
| `ellipseConvention` (optional) | `GPUGeographicDistributionEllipseConvention` | Axis scaling of the ellipse: `'root-two'` scales the axes by `sqrt(2)` as the ArcGIS Directional Distribution tool does, `'standard'` uses the plain standard deviation along each principal axis. Defaults to `'root-two'`. |
| `orientationOnly` (optional) | `boolean` | Treat lines as undirected (angles are doubled before averaging, so a line and its reverse agree). Only affects `directionalMeans`. Defaults to `false`. |
| `medianTolerance` (optional) | `number` | Absolute tolerance, in position units, under which the last Weiszfeld step counts as converged for `medianConverged`. The iteration stops once every non-empty group's last step is at most this value. Non-negative. Defaults to `GPU_GEOGRAPHIC_DISTRIBUTION_DEFAULT_MEDIAN_TOLERANCE`. |

{/* /parameters */}

### `GPURegionStatistics`

`GPURegionStatistics`, `GPURegionMask` and `GPUPickRegionMask` capture construction settings and
nested port maps. Shape view identities, histogram domain pairs, grid metadata, pick filters and
output/status views stay as validated even when the caller later mutates its props objects.
Update the contents of shape, value, mask, pick and other GPU views for per-frame changes.
Rectangle bounds (four floats), polygon storage (at least three vertices), vertex-count storage,
screen transforms (20 floats), every packed positions chunk used by a shape, and pick results
(at least four words) are validated at construction, including on the grid path.
A caller mask must use packed `uint32` rows (byte stride 4); aligned nonzero byte offsets work.
Compose a mask producer as `region: {kind: 'mask', ...producerOutput}` to preserve its optional
`status.overflow`. A nonzero flag sets `regionTruncated`, just as the direct pick-region route does;
the flag is read again on each submission, and omitted status means complete input.
For a `GPURegionStatistics` producer, pass its complete output, including `summary`. The consumer
reads the producer's `regionTruncated` and `candidatesTruncated` bits as mask membership loss,
preserving both reasons and ignoring `selectionTruncated`: compact-ID capacity does not truncate
the source-aligned mask. This summary takes precedence over aggregate `status.overflow`.
For a standalone mask producer without a summary, `status.overflow` must describe mask membership.

The compact `output.selectedRows.status.overflow` reports final selection capacity loss;
optional `candidateOverflow` reports truncated region input or grid candidate loss. Its `requiredCount` counts
selected rows reaching compaction and is a lower bound when candidates were lost.

`GPURegionStatistics`: Select tract centroids inside a lasso and reduce them to counts, sum, mean, extremes and a histogram.

::lens{scenario=gpu-spatial-analysis.region-statistics rung=result caption="Tracts whose centroids fall inside a lasso: 240 of 791 Chicago tracts."}

`GPUPickRegionMask`: Convert a region-pick result into a deduplicated 0/1 mask over the source rows.

::lens{scenario=gpu-spatial-analysis.pick-region-mask rung=result caption="20 tracts of community area 8, picked by a supplied region-pick result."}

`GPURegionMask`: Test every point against a rectangle or lasso, writing one 0/1 mask value per row.

::lens{scenario=gpu-spatial-analysis.region-mask rung=result caption="Tracts whose centroids fall inside a rectangle: 180 of 791 Chicago tracts."}

Selects points inside a per-frame rectangle, circle, lasso polygon (world space or screen pixels
through a view-projection transform), pick region (`GPUIndexPickingTarget.addRegionPass` result),
or caller mask, and reduces them into selected count, value count, sum, mean, minimum, maximum, and
a fixed-bin histogram packed into one small `output.summary` buffer. Optional stable selected IDs
(`output.selectedRows`, a `GPUCompactOutput`) and a 0/1 `output.mask` stay on the GPU for
highlighting. `GPURegionStatisticsReadback` reads the summary through a
`GPUReadbackRing` without stalling. `GPURegionMask` and `GPUPickRegionMask` are exported as
standalone building blocks.
Readback `binCount` must be an integer from 0 through 65,536; zero still copies the 32-byte header.
A failed copy releases its reserved ring slot before rethrowing, so an immediate valid retry can
use that slot.

```ts
import type {Device} from '@luma.gl/core';
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
import {
  GPURegionStatistics, type GPURegionStatisticsProps
} from '@luma-contributors/analysis/gpu-spatial-analysis';
declare const graph: GPUCommandGraph<void>;
declare const device: Device;
declare const lasso: import('@luma.gl/graph/gpu-core').GraphDataView<'float32x2'>;
declare const lassoCount: GPUParameterBuffer<'uint32'>;
declare const transform: GPUParameterBuffer<'float32'>;
declare const positions: NonNullable<NonNullable<GPURegionStatisticsProps>['positions']>;
declare const values: NonNullable<NonNullable<GPURegionStatisticsProps>['values']>;
declare const summary: NonNullable<NonNullable<NonNullable<GPURegionStatisticsProps>['output']>['summary']>;
declare const selectedRows: NonNullable<NonNullable<NonNullable<GPURegionStatisticsProps>['output']>['selectedRows']>;

graph.add(new GPURegionStatistics({
  region: {kind: 'polygon', vertices: lasso, vertexCount: lassoCount.importToGraph(graph),
    screenTransform: transform.importToGraph(graph)},
  positions, values, histogram: {binCount: 16},
  output: {summary: summary, selectedRows}
}));
```

Rectangle and lasso regions are inclusive, as luma.gl `GPUPointSpatialQuery` is for bounds and
polygons: a point exactly on the boundary is selected. Lasso membership is even-odd with an exact
side test. A point is on an edge when it lies in the edge's bounding box and the exact orientation
sign is zero; otherwise a straddling edge counts when the exact orientation sign puts the point
strictly on its inner side, so no f32 crossing abscissa decides the result.

`GPURegionMask` writes a `GPURegionMaskOutput`. `GPUPickRegionMask` writes a
`GPUPickRegionMaskOutput` with a source-aligned `mask` and `status.overflow` set to 0 or 1.
Omit `status.antimeridianUncertainCount` and `status.poleUncertainCount` when constructing a
pick mask, since the pick route has no geographic polygon diagnostics. Both `GPUPickRegionMask`
and a `{kind: 'pick-region', picks}` region read the packed pick result from `picks`.

Coordinates: `spatialContext` defaults to native planar. Rectangle, lasso and pick regions are
topological tests and also accept `longitude-latitude` (metric `'none'`): edges are straight lines
in degree space and require linear edges. Prepare polygons that cross ±180 or enclose a pole with
`GPULonLatPolygonPreparation`. Optional `output.status.antimeridianUncertainCount` and
`output.status.poleUncertainCount` report unprepared rings in `GPURegionMask` and
`GPURegionStatistics`. These one-row `uint32` counters are cleared and rewritten every encoding
for world-space lon/lat lassos; planar contexts, screen-space lassos and other region kinds leave
them untouched. A lasso consumes one ring at a time, including an empty prepared copy through a
zero active vertex count. A `'radius'` region is a planar distance in
position units, so it throws at construction in `longitude-latitude` (the region kind is
compile-time); project positions to a planar frame first, or use a rectangle or lasso. Geographic
metrics are rejected. Radius selections measure distance and reject `local-lambert-equal-area` frames. `GPURegionMask` takes the same `spatialContext`; `GPUPickRegionMask` reads no
positions and has none.

An optional `spatialIndex: {kind: 'grid', index, candidateCapacity}` takes a caller-built
`GPUGridIndex` over the same positions and reduces only candidates from the cells a world-space
rectangle, circle, or lasso touches. Results match the full scan except for float reassociation in
`sum` and `mean`. Candidate queries include a one-cell halo around the touched cell rectangle
to cover float32 binning differences. `candidateCapacity` bounds these candidates before the exact
predicate: narrowing a region can still leave candidate loss when the halo includes neighboring rows.
The `candidatesTruncated` flag reports capacity overflow or rows missing from the
index.

**Summary words.** `output.summary` holds `8 + binCount` `uint32` words (`GPU_REGION_STATISTICS_SUMMARY_LAYOUT`):
0 `selectedCount`, 1 `valueCount`, 2 `sum` (f32 bits), 3 `mean` (f32 bits), 4 `minimum` (f32 bits), 5 `maximum`
(f32 bits), 6 `histogramOutsideCount`, 7 `flags`, then the bins. `flags` bits (`GPU_REGION_STATISTICS_FLAGS`):
1 `selectionTruncated` (`output.selectedRows.ids` too small), 2 `regionTruncated` (lasso vertex count above
capacity, or pick-region overflow), 4 `candidatesTruncated` (grid candidate stage incomplete); other bits are 0.
`decodeGPURegionStatistics` and `GPURegionStatisticsReadback` decode the words. The same flags are also available
as named one-row status scalars beside the header, which keeps its words: optional `output.status.overflow` is 1
when bit 1 or 2 is set and `output.status.candidateOverflow` is 1 when bit 4 is set (it is not folded into
`overflow`). Optional `output.selectedRows.status.candidateOverflow` reports bits 2 or 4 separately
from `output.selectedRows.status.overflow`; `selectedRows.status.requiredCount` counts selected rows
reaching final compaction and is a lower bound when the region input or grid candidates were truncated.

An optional `output.drawInstanceCount` (one-row `uint32` view, for example the instance-count word
of an indirect draw record) receives the same capacity-clamped selected count as
`output.selectedRows.status.count` on every encoding, so selected rows can be drawn instanced through
`output.selectedRows.ids` without a readback. It requires `output.selectedRows`; statistics-only
graphs read the count from `output.summary`.

#### How it computes

{/* lens:GPURegionStatistics */}
{/* lens:GPUPickRegionMask */}
{/* lens:GPURegionMask */}

`GPURegionStatistics`:

::lens{scenario=gpu-spatial-analysis.region-statistics rung=datum bind=page datum=420 caption="Inspect tract 420: its centroid and value, and whether the lasso selects it."}
::lens{scenario=gpu-spatial-analysis.region-statistics rung=formula caption="Follow the summary: selected finite values give the count, sum and mean."}
::lens{scenario=gpu-spatial-analysis.region-statistics rung=kernel caption="Inspect the lasso mask, count, sum, extent and summary kernels."}
::lens{scenario=gpu-spatial-analysis.region-statistics rung=graph caption="Trace the mask, count and reduction nodes into the summary buffer."}

`GPUPickRegionMask`:

::lens{scenario=gpu-spatial-analysis.pick-region-mask rung=datum bind=page datum=420 caption="Inspect tract 420: which pick pairs name it and the mask value they set."}
::lens{scenario=gpu-spatial-analysis.pick-region-mask rung=formula caption="Follow the rule: a row is 1 when a pick pair of the chosen batch names it."}
::lens{scenario=gpu-spatial-analysis.pick-region-mask rung=kernel caption="Inspect the clear, scatter and overflow kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.pick-region-mask rung=graph caption="Trace the mask clear, the pick scatter and the overflow copy."}

`GPURegionMask`:

::lens{scenario=gpu-spatial-analysis.region-mask rung=datum bind=page datum=420 caption="Inspect tract 420: its centroid against the rectangle bounds."}
::lens{scenario=gpu-spatial-analysis.region-mask rung=formula caption="Follow the rule: a point is 1 inside the inclusive, valid rectangle on both axes."}
::lens{scenario=gpu-spatial-analysis.region-mask rung=kernel caption="Inspect 791 point tests with the retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.region-mask rung=graph caption="Trace the overflow flag and the single region mask kernel."}

### `GPUZonalStatistics`

Reduce tract values to one count, sum, mean and deviation per community area, using per-row zone membership.

::lens{scenario=gpu-spatial-analysis.zonal-statistics rung=result caption="People below the poverty line summed over 77 community areas from 791 tracts; total 688,314."}

Choropleth aggregation. Each point is joined to the polygon feature that contains it (an internal
`GPUPointInPolygonJoin`), or the caller supplies per-point feature rows. The points are then
reduced per feature to count, valid-value count, sum, weighted sum and weight sum, mean, minimum,
maximum, density (count divided by a caller area or a GPU shoelace area), and a feature-value
`[min, max]` extent for color scales.

These are exclusive assignment summaries: one point contributes to at most one polygon row.
For PIP, request `polygonAssignmentCounts` for this count and `polygonMembershipCounts` for every
matching polygon.
For one point strictly inside each of two overlapping polygons, assignment counts are `[1,0]`
(lowest polygon row wins), whereas membership counts are `[1,1]`. For every **strict** match, put
polygons on the left of `GPUSpatialPredicateJoin` with `predicate: 'contains'`, request weights,
and difference adjacent CSR offsets. A weights-only join also requires top-level
`output.status.overflow`; request `uncertainCount` and publish candidate loss before using offsets.
The caller-owned views below follow the [readback storage example](./operators-joins.md#readback-destinations-used-below).

```ts
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  GPUSpatialPredicateJoin, type GPUSpatialPredicateJoinProps
} from '@luma-contributors/analysis/gpu-spatial-analysis';

declare const graph: GPUCommandGraph;
declare const polygons: GPUSpatialPredicateJoinProps['left'];
declare const points: GPUSpatialPredicateJoinProps['right'];
// All destinations, including the status scalars, are caller-owned and readback-capable.
declare const weights: NonNullable<GPUSpatialPredicateJoinProps['output']['weights']>;
declare const overflow: GraphDataView<'uint32'>;
declare const candidateOverflow: GraphDataView<'uint32'>;
declare const uncertainCount: GraphDataView<'uint32'>;
graph.add(new GPUSpatialPredicateJoin({
  left: polygons, right: points, predicate: 'contains', how: 'inner', candidateCapacity: 256,
  output: {
    weights,
    status: {overflow, candidateOverflow, uncertainCount}
  }
}));
```

Read exactly `polygonCount + 1` CSR offsets, starting at zero and nondecreasing, and exactly one
word each from the top-level `output.status` for `overflow`, `candidateOverflow` and `uncertainCount`. All three words must explicitly
equal zero before differencing; missing words are not success. The [join guide decoder](./operators-joins.md)
checks these conditions.

Boundary points follow `boundaryRule` (passed to the join). The default `'half-open'` counts a point
on an edge or vertex shared by coverage features exactly once, whatever the feature row order; points
on the outer right or top edge of the coverage are counted by no feature. `'closed'` counts every
boundary point (smallest containing row wins) and `'open'` only strict interiors.

Polygon features use the four-level `GPUPolygonGeometryPort` layout (`positions`, `featureOffsets`,
`polygonOffsets`, `ringOffsets`; one output row per feature, multipart features included) with the
join options; their `sourceIds` are not read. Point positions are the top-level `positions`.

The output views you pass select the statistics at compile time. Empty features give a sum of 0 and
NaN for the mean, minimum and maximum. `sumOrder: 'sorted'` stably sorts points by feature and
reduces each segment in a fixed tree order, so sums are bitwise reproducible at a bounded cost. It is
the default when every point-rate input (positions, values, weights, `pointFeatureRows`) is packed.
Chunked inputs fall back to `sumOrder: 'atomic'`, whose float atomics can vary in the last bits
between runs and serialize when many points share a feature.

```ts
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  GPUZonalStatistics, type GPUZonalStatisticsProps
} from '@luma-contributors/analysis/gpu-spatial-analysis';

declare const graph: GPUCommandGraph;
declare const features: Extract<GPUZonalStatisticsProps['features'], {kind: 'polygons'}>;
declare const positions: NonNullable<GPUZonalStatisticsProps['positions']>;
declare const incomes: NonNullable<GPUZonalStatisticsProps['values']>;
declare const households: NonNullable<GPUZonalStatisticsProps['weights']>;
// One exclusive assignment count per polygon, matching PIP's polygonAssignmentCounts.
declare const polygonAssignmentCounts: GraphDataView<'uint32'>;
declare const means: GraphDataView<'float32'>;
declare const densities: GraphDataView<'float32'>;
declare const extent: GraphDataView<'float32'>; // two rows: [minimum mean, maximum mean]
// Caller-owned STORAGE | COPY_SRC | COPY_DST buffers, as in the linked readback factory.
declare const overflow: GraphDataView<'uint32'>;
declare const candidateOverflow: GraphDataView<'uint32'>;
declare const uncertainCount: GraphDataView<'uint32'>;
graph.add(new GPUZonalStatistics({
  features, positions, values: incomes, weights: households,
  output: {
    counts: polygonAssignmentCounts, means, densities, extent, extentStatistic: 'mean',
    status: {overflow, candidateOverflow, uncertainCount}
  }
}));
```

`output.status.uncertainCount` (one `uint32` row) reports the join's uncertain candidate pairs. A nonzero
value means some points could not be classified and the per-feature statistics may be low.
In `features: {kind: 'feature-rows', pointFeatureRows, featureCount, status}`, callers can supply
`overflow`, `candidateOverflow`, `uncertainCount`, `antimeridianUncertainCount` and
`poleUncertainCount`. Every supplied scalar requires a matching `output.status` destination in a
separate physical buffer. Diagnostics copy on every encoding; absent inputs clear their matching
destinations. Uncertainty counters preserve their numeric counts rather than becoming booleans.
Retain these diagnostics alongside the summaries when passing means to inference; polygon weights
being complete does not certify point membership. The [join guide](./operators-joins.md#zonal-values-and-inference)
shows the public common-mask helper.

Optional `output.status.candidateOverflow` is 1 when the join's candidate capacity or BVH leaf capacity was too
small, and optional `output.candidateTruncated` (polygon features only; one packed `uint32` per point) flags the
points whose membership may be incomplete: the points whose candidates were dropped, or every point after a BVH
leaf overflow. Both are the `GPUPointInPolygonJoin` outputs passed through. For caller-supplied
feature rows, `candidateOverflow` copies `features.status.candidateOverflow`, or clears to zero
when absent. `candidateTruncated` stays in the point-row domain; it is not a polygon inference mask.

- Variance. `output.variances` and `output.standardDeviations` are the unbiased estimates in two
  passes around a refined mean, so there is no `E[v^2] - E[v]^2` cancellation. Unweighted it
  is `ddof = 1` (`numpy.var(v, ddof=1)`). With `weights`, which act as reliability weights, it is
  the corrected numerator `sum w d^2 - (sum w d)^2 / V1`, divided by `V1 - V2 / V1`,
  with deviations `d` around the refined mean, `V1 = sum w`, `V2 = sum w^2`. For
  nonnegative reliability weights this is equivalent to
  `numpy.cov(v, aweights=w, ddof=1)`. A feature with fewer than two valid rows, or a non-positive
  denominator, gives NaN, never 0. Variances need packed point views (not chunked vectors) and follow
  `sumOrder` for the initial sums; mean-refinement and deviation reductions use stable fixed trees
  in both sum orders. Checked against numpy in `zonal_variance.py`.
- Float32 product overflow. In **both** `'sorted'` and `'atomic'` sum orders, a finite value and
  finite weight remain value-valid when their product overflows float32. The product is omitted
  from `sums`, while the row remains in `counts`, `valueCounts`, `weightSums`, `minima` and `maxima`.
  For values `[2^127, 2]`, weights `[4, 1]`, and both rows assigned to one polygon: `counts = 2`,
  `valueCounts = 2`, `sums = 2`, `weightSums = 5`, `means ≈ 0.4`, `minima = 2`, `maxima = 2^127`.
  Overflowing deviation terms are also omitted: the reported variance is about `0.8192` and
  standard deviation about `0.9051`, despite the enormous true spread. Finite summaries do not
  certify that every arithmetic term survived. Signed weights remain accepted: values `[0,0,0,5]`
  with weights `[2,2,2,-1]` give mean `-1`, corrected variance `-12.5` and NaN standard deviation.
  Negative corrected variance is retained; it is not clamped to zero.
- Weighted density. `output.weightedDensities` is `weightSums / area` (needs `weights`); the area rules
  of `densities` apply (finite and positive, else NaN).
- Coordinates. `spatialContext` accepts planar (default) or `longitude-latitude` with metric `'none'`.
  Membership runs `GPUPointInPolygonJoin` with a containment-only context derived from `spatialContext`
  (metric `'none'`), so in degrees it is topological with straight edges (not geodesics) and polygons
  require linear edges. Prepare polygons that cross ±180 or enclose a pole with
  `GPULonLatPolygonPreparation`; optional one-row `uint32`
  `output.status.antimeridianUncertainCount` / `output.status.poleUncertainCount` report unprepared
  rings and are cleared every encoding on the geographic polygon route. Planar polygon contexts
  do not run those ring checks. The feature-row route copies supplied geographic counters and
  clears matching destinations when absent, as above. The shoelace area of degrees is
  wrong, so a longitude-latitude context rejects GPU areas (`densities`, `weightedDensities`,
  `featureAreas` without `areas`) at construction: pass spherical `areas` in square meters, for example
  from `GPUGeometryMeasures`. GPU-computed planar areas and densities measure area and reject
  `local-azimuthal-equidistant` frames; supplied `areas` and means-only configurations do not trigger
  this area-frame restriction. Counts, sums, means, extrema and variances need no area.
- Not provided: median and quantiles (they need a per-feature sort of the values), weighted minima and
  maxima, fractional polygon coverage (a point belongs to a feature or it does not).

#### How it computes

{/* lens:GPUZonalStatistics */}

::lens{scenario=gpu-spatial-analysis.zonal-statistics rung=datum bind=page datum=420 caption="Inspect tract 420: its zone row, value and share of the zone sum."}
::lens{scenario=gpu-spatial-analysis.zonal-statistics rung=formula caption="Follow the zone sum, mean and deviation: finite values grouped by zone row."}
::lens{scenario=gpu-spatial-analysis.zonal-statistics rung=kernel caption="Inspect the sorted sums and the mean and deviation kernels."}
::lens{scenario=gpu-spatial-analysis.zonal-statistics rung=graph caption="Trace the stable sort by zone, segmented sums, means and deviations."}

### `GPUFocalStatistics`

Summarise each raster cell's moving window, writing the window mean, deviation and valid-cell count per cell.

::lens{scenario=gpu-spatial-analysis.focal-statistics rung=result caption="Window mean elevation (radius 3) on a 120 m terrain lattice; means span 1,083 to 1,531 m."}

Moving-window statistics over a float32 raster, like ArcGIS `FocalStatistics` and GRASS
`r.neighbors`: mean, sum, min, max, range, population standard deviation, and the count of valid
cells. The radius, window shape, minimum count, and center nodata policy are per-frame.
A cell is nodata when it is not finite (NaN or an infinity), equals the optional `noDataValue`, or
has a zero `validity` word; nodata cells are left out of every window. (Before 2026-11-10 an
infinity was kept as a value, which gave an infinite mean and a NaN standard deviation.)
The operator is raster-only: it reads a lattice of values and never positions, so it takes no
`spatialContext`, its `grid` (`GPURasterMetadata`) only sizes the raster, and its radius is in
cells, not ground units. Cell size and shape (for example
lon/lat cells that shrink with latitude) are the caller's concern.

```ts
import type {Device} from '@luma.gl/core';
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
import {
  GPUFocalStatistics, type GPUFocalStatisticsProps, getGPUFocalStatisticsParameterValues
} from '@luma-contributors/analysis/gpu-spatial-analysis';
declare const graph: GPUCommandGraph<void>;
declare const device: Device;
declare const focalParameters: GPUParameterBuffer<'float32'>;
declare const raster: NonNullable<NonNullable<GPUFocalStatisticsProps>['values']>;
declare const validity: NonNullable<NonNullable<GPUFocalStatisticsProps>['validity']>;
declare const grid: NonNullable<NonNullable<GPUFocalStatisticsProps>['grid']>;
declare const mean: NonNullable<NonNullable<NonNullable<GPUFocalStatisticsProps>['output']>['mean']>;
declare const standardDeviation: NonNullable<NonNullable<NonNullable<GPUFocalStatisticsProps>['output']>['standardDeviation']>;
declare const min: NonNullable<NonNullable<NonNullable<GPUFocalStatisticsProps>['output']>['min']>;
declare const max: NonNullable<NonNullable<NonNullable<GPUFocalStatisticsProps>['output']>['max']>;
declare const counts: NonNullable<NonNullable<NonNullable<GPUFocalStatisticsProps>['output']>['counts']>;

graph.add(
  new GPUFocalStatistics({
    values: raster, // row-major float32, NaN = nodata
    validity, // optional uint32, 0 = nodata
    noDataValue: -9999, // optional sentinel
    grid, // GPURasterMetadata: width * height cells, any affine
    maximumRadius: 16, // compile-time cap, at most 64
    parameters: focalParameters.importToGraph(graph), // 4 float32
    output: {mean, standardDeviation, min, max, counts} // any non-empty subset
  })
);
focalParameters.write(
  getGPUFocalStatisticsParameterValues({radius: 3, shape: 'circle', minimumCount: 5})
); // no recompile
```

- Square windows cover `|dx|, |dy| <= floor(radius)`; circles keep offsets with
  `dx^2 + dy^2 <= radius^2`. The radius is clamped to `maximumRadius`.
- Window cells outside the raster and nodata cells are skipped, so edge windows are clipped.
  A cell is nodata (NaN) with fewer than `max(minimumCount, 1)` valid cells, or when its own value
  is nodata and `propagateCenterNoData` is set. `count` is written for every cell.
- Direct gather in fixed row-major window order: deterministic, no atomics. Standard deviation
  subtracts the window mean in a second pass. Moments and extremes are separate kernels, each
  scheduled only when one of its outputs is requested.
- Cost is `(2r + 1)^2` reads per cell; see Status and limitations for the still-open separable and
  summed-area variants.

#### How it computes

{/* lens:GPUFocalStatistics */}

::lens{scenario=gpu-spatial-analysis.focal-statistics rung=datum bind=page datum=528 caption="Inspect cell 528: its window, mean, deviation and valid-cell count."}
::lens{scenario=gpu-spatial-analysis.focal-statistics rung=formula caption="Follow the window mean and population deviation over the valid cells of each window."}
::lens{scenario=gpu-spatial-analysis.focal-statistics rung=kernel caption="Inspect 1,024 gather invocations reading each window in row-major order."}
::lens{scenario=gpu-spatial-analysis.focal-statistics rung=graph caption="Trace the single moments kernel from the elevation raster to the window statistics."}

{/* parameters: GPUFocalStatistics generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUFocalStatistics` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPUFocalStatistics`.

- Packer: `getGPUFocalStatisticsParameterValues(settings)`, writes `GPU_FOCAL_STATISTICS_PARAMETER_LENGTH` (4) `float32` values.

Settings of `getGPUFocalStatisticsParameterValues`:

| Field | Type | Description |
|---|---|---|
| `radius` | `number` | Window radius in cells. A square window covers `\|dx\|, \|dy\| <= floor(radius)`; a circle covers offsets with `dx^2 + dy^2 <= radius^2` inside that square. Clamped to the contributor's compile-time `maximumRadius`. |
| `shape` (optional) | `GPUFocalStatisticsShape` | Window shape. Defaults to `'square'`. |
| `minimumCount` (optional) | `number` | Cells whose window has fewer valid cells are nodata. Defaults to 1. |
| `propagateCenterNoData` (optional) | `boolean` | When true, a nodata center cell yields nodata statistics. Defaults to false. |

{/* /parameters */}

### `GPUSimilarLocations`

Rank rows by weighted distance in standardized attribute space to a reference, returning ranks, distances and the best row ids.

::lens{scenario=gpu-spatial-analysis.similar-locations rung=result caption="Distance of 136 north-side tracts to one reference tract in five indicators; nearest 0.44, reference blank."}

Maximum: `columnCount` is at most `GPU_SIMILAR_LOCATIONS_MAXIMUM_COLUMN_COUNT` (one 256-lane reduction per column within a uint32 dispatch) and the value table at most `getGPUSimilarLocationsMaximumValueCount(device)` (`2^25` on a default device) float32 values.

Ranks rows by weighted Euclidean distance, in standardized attribute space, to a reference row or the mean
of several selected rows (the ArcGIS Similarity Search task; checked against NumPy, SciPy and scikit-learn distances). Props: `values` (row-major `rowCount * columnCount`), `columnCount`,
`referenceMask` (a `uint32` mask of the reference rows), optional `mask`, per-frame `parameters`
(`getGPUSimilarLocationsParameterValues`: result count, most or least similar, reference exclusion and
attribute weights), `standardization` (`'z-score'`, or `'rank'` for percentiles; compile-time) and
`resultCapacity` (compile-time size of `output.topIds`).

- Outputs: `ranks` (0 is the most similar; unranked rows hold `GPU_SIMILAR_LOCATIONS_NO_RANK`),
  `distances` (NaN when unranked), `topIds` (the best rows in rank order) and `status.count`.
  The entire supplied `topIds` view is rewritten: unused words, including capacity beyond
  `resultCapacity`, hold `GPU_SIMILAR_LOCATIONS_NO_RANK`.
- Rows with a non-finite attribute or a zero mask are never ranked. A stable `GPUSort` on the distance bits
  keeps ties on the lowest ID.
- `'z-score'` uses population standard-deviation units (divisor `n`). Apply the same normalization to reference rows,
  candidates and displayed attribute readouts; percentile units belong only to `'rank'`.
- `'rank'` standardization uses average-tie percentile ranks `(rank - 1) / (validCount - 1)`.
  It scans pairs below 4096 rows and sorts each column from 4096 rows. Z-scores use population
  deviation over all valid rows, including references; a constant column contributes zero.
  Variance is accumulated relative to a valid row. For z-score distances, the common column
  mean cancels: reference accumulation and candidate subtraction occur in a common binary
  frame before scaling by the column deviation, retaining opposite large finite endpoints. The centroid retains a high word and a rounding residual,
  including when multiple references have a mean between representable coordinates.
- Distances are weighted Euclidean distances to the mean standardized reference vector. Cosine
  similarity is not an available metric. With no valid reference every distance is NaN, every
  rank/top ID is `GPU_SIMILAR_LOCATIONS_NO_RANK` and `status.count` is zero.
- Attribute weights must remain finite after float32 packing. `resultCount` must be a nonnegative
  integer exactly representable in float32 (all integers through `2^24` are exact). For z-score
  standardization, relative coordinate sums, central squared deviations and weighted distance sums must
  remain representable in float32; shift/rescale columns when these intermediate quantities would overflow.

#### How it computes

{/* lens:GPUSimilarLocations */}

::lens{scenario=gpu-spatial-analysis.similar-locations rung=datum bind=page datum=60 caption="Inspect tract 60: its distance to the reference tract and its similarity rank."}
::lens{scenario=gpu-spatial-analysis.similar-locations rung=formula caption="Follow the z-scores, the reference vector and the weighted Euclidean distance to it."}
::lens{scenario=gpu-spatial-analysis.similar-locations rung=kernel caption="Inspect the column statistics, standardize, reference, distance, sort and rank kernels with retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.similar-locations rung=graph caption="Trace validation, z-scores, the reference mean, distances and the stable sort into ranks."}

{/* parameters: GPUSimilarLocations generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUSimilarLocations` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPUSimilarLocations`.

- Packer: `getGPUSimilarLocationsParameterValues(settings, columnCount)`, writes `getGPUSimilarLocationsParameterLength(...)` `float32` values.
  - `columnCount` (`number`): Number of attributes of the contributor.

Settings of `getGPUSimilarLocationsParameterValues`:

| Field | Type | Description |
|---|---|---|
| `resultCount` | `number` | Number of top-ranked rows reported in `output.topIds`, clamped to the compile-time capacity. |
| `direction` (optional) | `'most' \| 'least'` | `'most'` ranks the nearest rows first; `'least'` ranks the farthest first. Defaults to `'most'`. |
| `excludeReference` (optional) | `boolean` | Excludes the reference (selected) rows from the ranking. Defaults to true. |
| `weights` (optional) | `ArrayLike<number>` | Non-negative weight per attribute. Defaults to 1 for every attribute. |

{/* /parameters */}

### `GPULocalOutlierFactor`

Score each point's local density against its neighbours' over a kNN table, returning k-distance, density, LOF and an outlier mask.

::lens{scenario=gpu-spatial-analysis.local-outlier-factor rung=result caption="Local outlier factor of 2,344 observations from 10 neighbours; 75 exceed 3 and mark sparse points."}

Local Outlier Factor (Breunig 2000; geo `OutlierDetection`, scikit-learn `LocalOutlierFactor`, Sedona
`ST_LocalOutlierFactor`) over a kNN table. `neighbors` is a `GPUSpatialWeights` with `distances`, typically
`GPUNeighborSearch` in `'knn'` mode as a self join; the weight values are unused. Per row it writes
`kDistance` (the largest neighbor distance), `localReachabilityDensity = 1 / (mean max(kDistance_j, d_ij) +
densityFloor)` and `lof = mean(lrd_j) / lrd_i`, which equals `-negative_outlier_factor_`, all in the `output` bag. Optional
outputs are an `outlier` mask (`lof > threshold`) and a one-row `outlierCount` (integer atomics, deterministic).
`parameters` (`getGPULocalOutlierFactorParameterValues({threshold, densityFloor = 1e-10})`) change per
frame without recompiling. There is no default `threshold`: LOF has no null distribution, so no cut is
calibrated (the former hidden 1.5 is gone). Omit it and the mask and count are all 0 while `lof` is
unaffected; otherwise pick a cut from the score distribution. LOF is exploratory, not inferential: no
p-value, nothing "significant". Choose `k` (the neighbor count of the table) of at least 10 and larger than
the smallest group that should count as a cluster rather than as outliers (Breunig 2000), and compare
scores only at the same `k`. The input must be a true kNN list; with ties at the k-th distance a list cut
at exactly `k` entries changes the k-distance neighborhood (as in scikit-learn). The class reads only
distances; `neighbors.spatialContext` describes their length unit, which also applies to `densityFloor`.
Optional `spatialContext` declares the floor's context and is checked against `neighbors.spatialContext`
using metric-results compatibility when both are supplied: effective distance units and metric
parameters must match. Omitted great-circle units mean meters and cannot be paired with an
explicit kilometer floor; convert the numeric floor and declare matching units. LOF itself is a unit-free ratio. The floor keeps coincident clusters finite at density `1 / densityFloor`, and
points whose neighbors are such duplicates score enormous values, as in scikit-learn, so use a `k` above
the duplicate count or a larger floor. A row with no neighbors gets kDistance 0, density 0 and lof 1 and is
never flagged, including when the threshold is below one. Neighbor IDs outside the row range are
excluded from k-distance, reachability density and LOF alike. A declared non-kNN provenance is
refused: provide a k-nearest-neighbor table; a radius list has density-dependent neighborhoods.
The packed density floor and its reciprocal must both be finite in float32; rescale distances
and the floor together if needed. Reachability and density sums must remain representable.
When `neighbors.status` reports nonzero `overflow`, `candidateOverflow` or `iterationLimitReached`,
supplies a zero `converged` or `complete` flag, or reports `requiredCount` above the physical
neighbor-slot capacity, all three numerical outputs are NaN and the outlier mask and count are zero.
Clearing the loss evidence restores the ordinary scores on the next submission. Inspect the upstream status to distinguish incomplete support from a valid fit.
A disjoint density or score view may share the classification mask/count buffer; classification
uses private copies of density or score inputs that share the classification mask/count buffer,
avoiding conflicting read/write storage usages.
Against
a pinned scikit-learn run (24 points with `k = 5`, and a duplicate case) kDistance was within 2e-6 relative
and lof within 1e-4, with the mask and count exact.

#### How it computes

{/* lens:GPULocalOutlierFactor */}

::lens{scenario=gpu-spatial-analysis.local-outlier-factor rung=datum bind=page datum=1000 caption="Inspect observation 1000: its k-distance, density, LOF and outlier flag."}
::lens{scenario=gpu-spatial-analysis.local-outlier-factor rung=formula caption="Follow the reachability distance, local density and the ratio of neighbour density to its own."}
::lens{scenario=gpu-spatial-analysis.local-outlier-factor rung=kernel caption="Inspect the k-distance, density, score and outlier-mask kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.local-outlier-factor rung=graph caption="Trace the ten-nearest-neighbour search into k-distance, density, LOF and the thresholded mask."}

See [local variability and regression extensions](./statistical-expansion.md) for the expanded
statistics, fixed, finite-grid or golden-section-selected multiscale fits, generalized local models, pooled panels and regimes.

{/* parameters: GPULocalOutlierFactor generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPULocalOutlierFactor` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPULocalOutlierFactor`.

- Packer: `getGPULocalOutlierFactorParameterValues(settings?)`, writes `GPU_LOCAL_OUTLIER_FACTOR_PARAMETER_LENGTH` (4) `float32` values.

Settings of `getGPULocalOutlierFactorParameterValues`:

| Field | Type | Description |
|---|---|---|
| `threshold` (optional) | `number` | Rows with `lof > threshold` are flagged in the `outlier` mask and counted. There is no default: LOF has no null distribution, so no cut is calibrated (Breunig et al. 2000 give none; values near 1 are as dense as their neighbors, well above 1 locally sparse, and the scale depends on `k` and the data). Choose it from the score distribution, for example a top fraction. When omitted the packed threshold is NaN, no row is flagged, the `outlier` mask is all 0 and `outlierCount` is 0; the `lof` scores are unaffected. The flag is exploratory, not a significance statement. |
| `densityFloor` (optional) | `number` | Added to each row's mean reachability distance before inverting it, so a cluster of coincident points (mean reachability 0) gets a large finite density `1 / densityFloor` instead of infinity. Must be positive and finite. Defaults to scikit-learn's `1e-10`. Raise it (for example to the coordinate resolution) to make duplicate-heavy data less extreme. Its unit is the length unit of `neighbors.spatialContext`; declare the context of this value through `GPULocalOutlierFactorProps.spatialContext` to check compatibility. |

{/* /parameters */}
