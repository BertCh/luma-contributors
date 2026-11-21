# Space-time weights, lagged and differential local Moran, cross-correlation

STX-02. This section belongs under [operators-dynamics.md](./operators-dynamics.md); it lives in its own
file until that file is free of other edits. Import from `@luma-contributors/analysis/gpu-spatial-analysis`.
Cube layouts are those of `GPUCubeLayoutTranspose`: `'cell-major'` is `cell * periodCount + period`,
`'period-major'` is `period * cellCount + cell`.

## Space-time weights: `GPUSpatialWeightsAlgebra` operation `'spaceTime'`

`T (x) W`: a spatial weights matrix `W` crossed with a temporal lag matrix `T`. The result is the ordinary
sparse `GPUSpatialWeights` CSR over `cellCount * periodCount` rows (ascending neighbor IDs per row), so the
permutation tests, `GPUSpatialLag`, global statistics and multiple testing read it unchanged.

```ts
import type {GPUSpatialWeights, GPUSpatialWeightsAlgebraProps} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {GPUSpatialWeightsAlgebra} from '@luma-contributors/analysis/gpu-spatial-analysis';

declare const weights: GPUSpatialWeights;
declare const periodCount: number;
declare const output: GPUSpatialWeightsAlgebraProps['output'];
new GPUSpatialWeightsAlgebra({
  operation: 'spaceTime',
  weights, // W over cellCount cells
  periodCount,
  layout: 'period-major', // or 'cell-major'
  lags: [{lag: 0}, {lag: 1, weight: 0.5, includeSelf: true}],
  output // rows = cellCount * periodCount, with status.overflow
});
```

Row (cell `i`, period `t`) lists, for each lag `l`, the cells `j` of row `i` of `W` at period `t - l` with
weight `T * w_ij`. Lag 0 is the contemporaneous neighborhood, lags above 0 look into the past, negative lags
look ahead. Entries outside `[0, periodCount)` are dropped, so the first `max(lag)` periods of a pure lag
matrix are islands. `includeSelf` adds the same cell at the lagged period with spatial weight 1 (`I + W`
for that lag) and replaces a diagonal slot of `W`; it is refused at lag 0. Lags must be distinct (at most
16) and temporal weights positive. `output.distances` is not supported. The result keeps the `'raw'` or
`'binary'` transform of `W` only for unit temporal weights without `includeSelf`; otherwise it is
`'unknown'`, so row-standardize afterwards with `GPUSpatialWeightsTransform` when the permutation test
should see rows that sum to 1.

Reference: a dense Kronecker sum built independently in the spec (both layouts, negative lags, self
entries, overflow).

## Lagged and differential local Moran: `GPUSpaceTimeLocalMoran`

Construction rejects a missing `cube`, a non-float32 cube, non-uint32 `parameters`, and unknown
`layout`, `variant` or `direction` values. Delegated permutation options are validated before
scratch allocation, so a corrected setting can reuse the same graph and ID.

Like analytic local Moran, permutation scoring is unavailable when either supplied neighborhood
loss word (`weights.status.overflow` or `candidateOverflow`) is nonzero. Every p-value and
observed statistic is NaN, and the local permutation completeness word is 1 for either loss stage.
Inspect the input flags separately to identify the loss stage. Clearing
the flags with complete weights restores inference on the same graph.

The Knox, Mantel and Jacquez contributors also suppress inference on either pair/neighborhood
loss word: their entire float summary is NaN, float statistics are NaN, integer count statistics
are zero, and an optional permutations-used output is zero. The Mantel operation and CPU twin
accept `pairs.status: {overflow?, candidateOverflow?}` graph-value names; the pure CPU function
accepts the corresponding Uint32Array words in `input.status`. These conditions are read afresh
on each submission; the two input loss meanings remain distinct.
Suppression writes only the published layout: `GPU_SPACE_TIME_SUMMARY_LENGTH` summary words,
`maximumPermutations + 1` statistic words and one permutations-used word. Larger caller views
retain their spare words on complete, suppressed and recovered submissions.
Only `overflow` and `candidateOverflow` suppress CPU Mantel; capacity metadata such as
`requiredCount` is ignored. The decoder represents a suppressed zero-draw frame with
`permutationCount: 0, available: false`; positive draw counts retain their existing lower bounds.

Mantel correlations for exactly two nonconstant listed pairs use the ordering of their transformed
distances and time gaps before any variance guard. They are exactly +1 or -1, so
permutations with the same sign count as ties in the upper tail and sequential stopping.
Null moments are centered on the first permuted correlation: identical nonzero null statistics
have zero sample variance and a NaN z-score, just like an all-zero null distribution.
CPU inference compares correlations at their published float32 precision, so hidden float64
differences cannot split published ties. GPU Pearson moments use a selected pair's data-valued
anchor rather than a rounded mean to preserve equal integral moments across reordered gaps.
Published `GPUReduction` stages sum the observed pair counts and select the first nonempty
block's anchor. The shift publication reads that anchor directly, without a serial scan of
block summaries. This adds two hierarchical reductions to the Mantel command graph.
Hawkes and Mantel operation forms refuse resolved positions/times/pair offsets that disagree
with their declared `rowCount`, before building a GPU contributor, matching CPU execution.
Both Hawkes backends capture nested spatial/temporal metadata, counts and named optional diagnostics at construction, including
prototype getters; caller mutation cannot change the population or drop a declared status binding.

GeoDa definitions (the GeoDa workbook lab 6c (multivariate local statistics) and the GeoDa 1.22 differential
Moran):

- `variant: 'lagged'`: bivariate local Moran of one variable at two times,
  `I_i = z_{i,t} sum_j w_ij z_{j,t-lag}` (`direction: 'inward'`, default); `'outward'` swaps the roles
  (`z_{i,t-lag}` with the neighbors at `t`). It runs `GPULocalPermutationTest` `'localBivariateMoran'`
  with the permuted variable being the second one.
- `variant: 'differential'`: the univariate local Moran of the change `y_t - y_{t-lag}` (GeoDa documents the
  differential Moran as the Moran statistic of the difference variable). It runs `'localMoran'`.

The recipe slices period `period` and `period - lag` out of the cube (one kernel), then forwards every other
prop (`parameters`, `maximumPermutations`, `alternative`, `islands`, `multipleTesting`, `output`, ...) to the
permutation test. Scaling is esda's: `I = (n - 1) z_x lag(z_y) / sum z_x^2`, which is `(n - 1) / n` times the
GeoDa z product. Permutations are conditional on the focal row and drawn over the cells of the period.

`output.analysisCount` counts the finite analysis rows after the island policy; `islandCount`
counts finite included rows with no included neighbor. Missing cube values are excluded, and
untested rows have exceedances/permutation counts `0xffffffff`, NaN p-values and zero significance.
`permutationCounts` reports actual draws for tested rows in both fixed-budget and stopped runs;
unavailable rows contain `GPU_LOCAL_PERMUTATION_NOT_TESTED` (`0xffffffff`). `quadrants` is available for the lagged
bivariate variant (1 HH, 2 LH, 3 LL, 4 HL; zero for excluded rows), independently of significance.
The inherited `classes` output is refused at construction, before graph mutation, because it
applies only to local Geary. Use lagged Moran `quadrants`, or a local Geary permutation test for classes.
Differential `quadrants` are also refused at construction, before scratch registration; use lagged Moran.
`status.invalidCount` is the shared Getis-Ord negative-value diagnostic and is always zero for
these Moran variants; it does not count missing cube values.

GeoDa's caveat applies: the lagged bivariate form ignores in-situ correlation between the two times, so
High-High can reflect temporal inertia rather than diffusion.

The whole cube at once is a composition, not a class: build `'spaceTime'` weights with `lags: [{lag: 1}]`
and run `localBivariateMoran` with `values = secondValues = cube`. The permutation then draws over the
pooled cube (a different null from the per-period recipe), the first period is an island, and the
standardization is pooled. Use `islands: 'keep'`.

References: esda 2.10.0 `Moran_Local_BV` and `Moran_Local` (fixture
`modules/analysis/test/oracles/fixtures/space-time-statistics.json`, p-values within five binomial standard
errors), an independent double-precision CPU twin, and the Philox twin for exceedance counts.

{/* lens:GPUSpaceTimeLocalMoran */}

## Cross-correlation per location: `GPUSpaceTimeCrossCorrelation`

`GPUSpaceTimeCrossCorrelation`: periodCount is at most 2,147,483,647; cube and correlation address
products must fit uint32.

For each cell and lag `l` (distinct integers, `|l| <= periodCount - 4`, at most 64) the Pearson correlation of
the overlapping pairs `(first[t], second[t + l])`, so a positive lag asks whether `second` follows `first`.
The p-value is Fisher's z test, `erfc(|atanh r| sqrt(n - 3) / sqrt 2)`. It assumes serially independent
observations: autocorrelated series give p-values that are too small, so difference or prewhiten first.
Pairs with a non-finite value, fewer than 4 overlapping periods or a constant series are NaN and outside the
family. `multipleTesting` (default `'benjamini-hochberg'`; use `'benjamini-yekutieli'` for arbitrary
dependence, which neighboring cells and adjacent lags are) adjusts across every tested (cell, lag) pair
through `GPUMultipleTestingAdjustment`. Outputs are indexed `cell * lags.length + lagIndex`: `correlations`,
`pValues`, optional `adjustedPValues`, `pairCounts`, `testedCount`. Both layouts are read in place.
The optional outputs are independent: `testedCount` is a one-row uint32 count of finite raw p-values
even without `adjustedPValues`. It is rewritten on every submission, including zero when no pair
can be tested. `pairCounts` records overlap lengths even for untestable constant or missing series.
`testedCount` may share a buffer with `pValues` in disjoint ranges, with or without adjusted p-values;
counting uses private storage and a separate publication pass.

Reference: numpy `corrcoef`, scipy `norm.sf` for the Fisher p-values and `false_discovery_control` for BH
(same fixture), plus an independent double-precision implementation with an exact `erfc`. f32 sums, so
correlations agree to 1e-4 and Fisher p-values to 1e-5 absolute plus 1e-3 relative.

{/* lens:GPUSpaceTimeCrossCorrelation */}
