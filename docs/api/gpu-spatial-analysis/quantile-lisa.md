# GPUQuantileLISA

{/* lens:GPUQuantileLISA */}

::lens{scenario=gpu-spatial-analysis.quantile-lisa rung=graph}

`GPUQuantileLISA` is a spatial-autocorrelation contributor exported from
`@luma-contributors/analysis/gpu-spatial-analysis`; the recipes barrel retains its compatibility
export. It composes exact resident column quantiles, a selected-class indicator and
conditional local join-count permutations for one scalar variable. Add it to a `GPUCommandGraph`
and submit the compiled graph explicitly. It never submits work, reads buffers or transfers
ownership of inputs.

Supply packed `float32` `values`, optional packed `uint32` `mask`, square CSR spatial `weights`,
a two-word resident `uint32` `quantileParameters` view `[k, q]`, the existing resident permutation
`parameters` view (packed with `getGPUPermutationParameterValues`), and caller-owned outputs.
`k` is the number of quantiles, `q` the selected class (one-based); `2 <= k < values.length` and
`1 <= q <= k`. Values, weights, masks, quantile settings, seed, permutation count and significance
level can change between submissions without rebuilding. `maximumPermutations`,
`maximumNeighbors` (default 32, at most 64) and view layouts are graph topology.

## Quantiles and ties

For `n` finite, unmasked values sorted as `x`, a boundary at probability `p` interpolates at
`h = clamp(n*p - 0.5, 0, n-1)` (Hyndman–Fan type 5). A row belongs when
`lower <= value < upper`. Exterior bounds are negative and positive infinity. A tie exactly at
an interior boundary belongs to the higher class. Ties can make a selected class empty; a
constant column belongs entirely to the final class. These rules follow the pinned
[libgeoda gda_quantilelisa](https://github.com/GeoDaCenter/libgeoda/blob/71238cb58979032048081434ce6e36a53e254280/gda_sa.cpp#L233)
and [QuantileBreaks and percentile](https://github.com/GeoDaCenter/libgeoda/blob/71238cb58979032048081434ce6e36a53e254280/GenUtils.cpp#L2080).
The desktop GeoDa dialog has a different upper-inclusive tie rule; this API follows libgeoda.

The resident selector uses type-7 quantiles after GPU conversion of type-5 ranks to probabilities.
Arithmetic and outputs are float32, so interpolated boundaries can differ from GeoDa's float64
arithmetic by rounding. Values arbitrarily close to a rounded boundary can change membership.
At most 2^24 packed source rows are supported. Chunked inputs require explicit caller packing.

## Missing values, islands and inference

NaN, infinities and zero-masked rows are omitted before quantiling and inference. The pinned
upstream `QuantileBreaks` implementation ignores its undefined mask when computing breaks;
this API deliberately corrects that defect. `validCount` counts finite unmasked rows before
island exclusion. The quantile distribution includes islands; inference excludes them using the
shared noncascading island policy. Only finite positive-weight, nonself slots count as neighbors;
join counts treat each such slot as a binary edge, regardless of the weight magnitude.

The observed statistic is `membership[i] * sum(membership[j])`. Only rows with a nonzero
observed count are tested. Conditional permutations draw distinct analysis rows without the
focus row, preserving its neighbor count. The shared Philox stream replays exactly for the same
seed and inputs. It draws uniformly without replacement; pinned GeoDa's complete mode rounds
`hash * (n-1)` to an integer, giving endpoints half the sampling mass of interior rows. The GPU
deliberately uses the shared uniform sampler and does not reproduce that biased Thomas Wang stream.

The pseudo p-value follows pinned GeoDa's legacy directed rule:
`(min(upperExceedances, P - upperExceedances) + 1) / (P + 1)`, including equality in the upper tail.
This legacy rule is anti-conservative; it does not provide a two-sided error-control guarantee.
A constant final class illustrates the legacy defect: all draws tie, producing `1/(P+1)` rather than 1. No multiple-testing adjustment is applied. Untested rows have NaN p-values and the shared
`GPU_LOCAL_PERMUTATION_NOT_TESTED` exceedance sentinel. Empty selected classes are valid results.

## Outputs and completeness

- `membership`: selected-class binary indicator, zero for missing rows.
- `boundaries`: two selected-class bounds, including exterior infinities.
- `validCount`: one-word source count.
- `observed`, `significant`: optional shared inference outputs.
- `exceedances`, `pseudoPValues`: required shared inference outputs.
- `classes`: `GPU_QUANTILE_LISA_CLASS`: 0 not significant, 1 significant, 2 undefined/masked,
  3 isolated. Unlike upstream `GetClusterIndicators`, which collapses exceptional rows to zero,
  these retain the label distinctions declared by its join-count class.
- `invalidQuantile`: one when resident k/q is invalid; membership is zero and no row is tested.
- `status.overflow`: shared inference invalid-count, neighbor-capacity or upstream weights-overflow
  flag. Read both flags before presenting results. A capacity failure is not an isolated row.

Every terminal output must be a distinct packed caller-owned view to permit readback. Internal
scratch is graph-owned. Inputs and outputs must not alias. Destruction follows normal graph and
buffer ownership; the contributor does not destroy borrowed inputs.

## Scope and evidence

This is the single-variable selected-quantile recipe. Multivariate joint membership is separate:
pinned `gda_multiquantilelisa` quantiles each column then invokes `MultiJoinCount`, which multiplies memberships when joint membership exists but switches to a bivariate
no-colocation statistic when the joint class is empty. Missing-mask merging also needs a separate
reference audit. This API does not claim that
multiplying independently computed memberships establishes multivariate parity.

Numerical GPU tests cover type-5 boundaries versus type 7, negative values, constant and tied
columns, empty classes, missing values, islands, invalid controls, neighbor capacity and replay.
The sorted test oracle independently computes boundaries; the established Philox oracle checks
exact exceedances, pseudo p-values and significance. The reproducible `modules/analysis/test/oracles/quantile-lisa-geoda.py` fixture runs actual
PyGeoDa 0.1.3 with 9,999 permutations. GPU observed counts match exactly; p-values are checked
with a broad 0.2 absolute tolerance because both the random streams and endpoint sampling differ.
Exact inference is asserted against the independent uniform Philox oracle, including the directed
tail and its all-ties degeneracy.
