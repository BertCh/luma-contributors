# GPUQuantileLISA

{/* lens:GPUQuantileLISA */}

::lens{scenario=gpu-spatial-analysis.quantile-lisa rung=graph}

`GPUQuantileLISA` composes exact resident column quantiles, a selected-class indicator and
conditional local join-count permutations for one or more scalar variables. Add it to a `GPUCommandGraph`
and submit the compiled graph explicitly. It never submits work, reads buffers or transfers
ownership of inputs.

Supply packed `float32` `values`, optional packed `uint32` `mask`, square CSR spatial `weights`,
a two-word resident `uint32` `quantileParameters` view `[k, q]`, the existing resident permutation
`parameters` view (packed with `getGPUPermutationParameterValues`), and caller-owned outputs.
For multivariate analysis, add `additionalVariables: [{values, quantileParameters}, ...]`.
The primary pair comes first; every variable has the same row count and its own `[k, q]` controls.
The common `mask` applies to every variable. Allocate `boundaries` with `2 * variableCount` words
and `validCount` with `variableCount` words; all other outputs remain source-sized or scalar.
For example, `additionalVariables: [{values: income, quantileParameters: incomeQuantiles}]`
intersects the primary variable's selected class with income's independently selected class.

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
At most 2^24 - 1 packed source rows and 64 variables are supported. The shared eight-column inference matrix needs `32 * values.length` bytes in one storage binding (at most 4,194,304 rows on a device with a 128 MiB binding limit). `estimatedCommandNodeCount` bounds graph compilation cost and is checked against the shared command-node budget at construction. Chunked inputs require explicit caller packing.

## Missing values, islands and inference

NaN, infinities and zero-masked rows are omitted independently from each variable before quantiling. Joint membership is the intersection of all selected-class indicators. Inference includes only rows finite in every variable and retained by the common mask. This does not quantile the complete-case subset: a finite value remains in its own marginal quantile distribution even when another variable is missing in that row. The pinned
upstream `QuantileBreaks` implementation ignores its undefined mask when computing breaks;
this API deliberately corrects that defect. `validCount` counts finite unmasked rows per variable before
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

- `membership`: joint selected-class binary indicator, zero when any variable is missing.
- `boundaries`: `[lower, upper]` pairs in primary-then-additional variable order, including exterior infinities.
- `validCount`: source count per variable in the same order.
- `observed`, `significant`: optional shared inference outputs.
- `exceedances`, `pseudoPValues`: required shared inference outputs.
- `classes`: `GPU_QUANTILE_LISA_CLASS`: 0 not significant, 1 significant, 2 undefined/masked,
  3 isolated. Unlike upstream `GetClusterIndicators`, which collapses exceptional rows to zero,
  these retain the label distinctions declared by its join-count class.
- `invalidQuantile`: one when any variable's resident k/q is invalid; membership is zero and no row is tested.
- `status.overflow`: shared inference invalid-count, neighbor-capacity or upstream weights-overflow
  flag. Optional `status.invalidCount` is forwarded by shared inference (zero for these binary inputs). Read `invalidQuantile` and `status.overflow` before presenting results. A capacity failure is not an isolated row.

Every terminal output must be a distinct packed caller-owned view to permit readback. Internal
scratch is graph-owned. Inputs and outputs must not alias. Destruction follows normal graph and
buffer ownership; the contributor does not destroy borrowed inputs.

## Scope and evidence

The contributor lives in `spatial-autocorrelation/gpu-quantile-lisa.ts` and is exported by the
spatial-autocorrelation barrel. The recipes barrel preserves its existing export for callers.
There is no existing CPU twin.

Multivariate output implements the selected-quantile intersection and joint local join count.
An empty joint class yields zero membership and observed counts on analysis rows, untested
p-values and no significant classes. Pinned libgeoda's `MultiJoinCount` has an additional
bivariate no-colocation fallback when the intersection is empty; this API keeps the explicit
intersection definition and does not switch statistics. Missing-mask handling follows the
per-variable omission and joint validity rule above.

The independent NumPy fixture generator
`modules/analysis/test/oracles/multivariate-quantile-lisa-numpy.py` uses `numpy.quantile` with
`method='hazen'` and exhaustively enumerates conditional combinations. Its fixtures cover two
and three variables, ties, empty intersections, empty marginal classes, different missing rows,
common masking and islands. GPU tests compare joint observed counts with those fixtures and
exact sampled exceedances with the established independent uniform Philox oracle; resident
replay, invalid settings in the first and last variable, invalid permutation counts, padding slots
and neighbor overflow are covered.

Numerical GPU tests cover type-5 boundaries versus type 7, negative values, constant and tied
columns, empty classes, missing values, islands, invalid controls, neighbor capacity and replay.
The sorted test oracle independently computes boundaries; the established Philox oracle checks
exact exceedances, pseudo p-values and significance. The reproducible `modules/analysis/test/oracles/quantile-lisa-geoda.py` fixture runs actual
PyGeoDa 0.1.3 with 9,999 permutations. GPU observed counts match exactly; p-values are checked
with a broad 0.2 absolute tolerance because both the random streams and endpoint sampling differ.
Exact inference is asserted against the independent uniform Philox oracle, including the directed
tail and its all-ties degeneracy.
