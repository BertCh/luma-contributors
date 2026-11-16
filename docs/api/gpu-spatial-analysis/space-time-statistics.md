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

Reference: numpy `corrcoef`, scipy `norm.sf` for the Fisher p-values and `false_discovery_control` for BH
(same fixture), plus an independent double-precision implementation with an exact `erfc`. f32 sums, so
correlations agree to 1e-4 and Fisher p-values to 1e-5 absolute plus 1e-3 relative.

{/* lens:GPUSpaceTimeCrossCorrelation */}
