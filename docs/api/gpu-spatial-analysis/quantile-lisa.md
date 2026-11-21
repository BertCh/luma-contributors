# GPUQuantileLISA

{/* lens:GPUQuantileLISA */}

::lens{scenario=gpu-spatial-analysis.quantile-lisa rung=graph}

`GPUQuantileLISA` is a spatial-autocorrelation contributor exported from
`@luma-contributors/analysis/gpu-spatial-analysis`; the recipes barrel retains its compatibility
export. It composes exact resident column quantiles, a selected-class indicator and
conditional local join-count permutations for one or more scalar variables. Add it to a `GPUCommandGraph`
and submit the compiled graph explicitly. It never submits work, reads buffers or transfers
ownership of inputs.

Supply packed `float32` `values`, optional packed `uint32` `mask`, square CSR spatial `weights`,
a two-word resident `uint32` `quantileParameters` view `[k, q]`, the existing resident permutation
`parameters` view (packed with `getGPUPermutationParameterValues`), and caller-owned outputs.
Choose `alternative: 'greater'`, `'two-sided'` or `'geoda-legacy'` explicitly at construction.
For multivariate analysis, add `additionalVariables: [{values, quantileParameters}, ...]`.
The primary pair comes first; every variable has the same row count and its own `[k, q]` controls.
The common `mask` applies to every variable. Allocate `boundaries` with `2 * variableCount` words
and `validCount` with `variableCount` words; all other outputs remain source-sized or scalar.
For example, `additionalVariables: [{values: income, quantileParameters: incomeQuantiles}]`
intersects the primary variable's selected class with income's independently selected class.

`k` is the number of quantiles, `q` the selected class (one-based); `2 <= k < values.length` and
`1 <= q <= k`. Values, weights, masks, quantile settings, seed, permutation count and significance
level can change between submissions without rebuilding. `maximumPermutations`,
`maximumNeighbors` (default 32, at most 64), `alternative` and view layouts are graph topology.
The local inference route accepts at most `MAXIMUM_PERMUTATION_COUNT = 1,048,576` draws per
submission; `maximumPermutations` is an allocation ceiling and the resident `permutationCount` is
the requested live draw count. Counts and neighbor limits are integers; masks and membership
are unitless indicators, weights are unitless, and values and boundaries retain each input
variable’s units. `significanceLevel` is a fraction strictly in (0, 1); p-values are fractions in [0, 1].

## Quantiles and ties

For `n` finite, unmasked values sorted as `x`, a boundary at probability `p` interpolates at
`h = clamp(n*p - 0.5, 0, n-1)` (Hyndman–Fan type 5). A row belongs when
`lower <= value < upper`. Exterior bounds are negative and positive infinity. A tie exactly at
an interior boundary belongs to the higher class. Ties can make a selected class empty; a
constant column belongs entirely to the final class. These rules follow the pinned
[libgeoda gda_quantilelisa](https://github.com/GeoDaCenter/libgeoda/blob/71238cb58979032048081434ce6e36a53e254280/gda_sa.cpp#L233)
and [QuantileBreaks and percentile](https://github.com/GeoDaCenter/libgeoda/blob/71238cb58979032048081434ce6e36a53e254280/GenUtils.cpp#L2080).
The desktop GeoDa dialog has a different upper-inclusive tie rule; this API follows libgeoda.

The resident selector computes type-5 ranks with integer quotient/remainder arithmetic, selects
the adjacent order statistics using exact rational ranks, then interpolates in float32. Integer
ranks are preserved: values `0..24`, ten quantiles and class four have bounds `[7, 9.5]` and members
`7, 8, 9`. The recipe reference also forms the integer rank numerator before division: values
`0..20`, fourteen quantiles and class ten have bounds `[13, 14.5]` and members `13, 14`.
Interpolated boundaries can still differ from GeoDa's float64 arithmetic by rounding;
values arbitrarily close to a rounded noninteger boundary can change membership.
At most 2^24 - 1 (16,777,215) packed source rows and 64 variables are supported. The inference
front end has eight separate row-sized columns, each `4 * values.length` bytes. Its largest
row-proportional binding is the uint32 row-position view, `4 * (values.length + 1)` bytes.
At the source-row ceiling this is 64 MiB, so a 128 MiB binding limit does not lower that ceiling.
Each supplied CSR and output binding must also fit its device limit; total allocation feasibility
is separate from the per-binding limit. `estimatedCommandNodeCount` bounds graph compilation
cost and is checked against the shared command-node budget at construction. Chunked inputs
require explicit caller packing.

## Missing values, islands and inference

NaN, infinities and zero-masked rows are omitted independently from each variable before
quantiling. Joint membership is the intersection of all selected-class indicators. Inference
includes only rows finite in every variable and retained by the common mask. This does not
quantile the complete-case subset: a finite value remains in its own marginal quantile
distribution even when another variable is missing in that row. The pinned
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

The inclusive `'greater'` rule uses `(upperExceedances + 1) / (P + 1)`. The inclusive
`'two-sided'` rule uses `min(1, 2 * (min(upperExceedances, lowerExceedances) + 1) / (P + 1))`.
Both tails include draws equal to the observation. On a constant final class every draw ties,
so both choices return 1 at nine draws and at 99 draws.

The explicitly selected `'geoda-legacy'` rule preserves the former implicit choice:
`(min(upperExceedances, P - upperExceedances) + 1) / (P + 1)`. It includes equality only in the
upper tail and uses a strict lower complement. It is anti-conservative: the same all-tied class
returns .1 at nine draws and .01 at 99. Existing callers wanting those historical numbers must
write `alternative: 'geoda-legacy'`; omission and unknown spellings now throw at construction.
No multiple-testing adjustment is applied. Untested rows have NaN p-values and the shared
`GPU_LOCAL_PERMUTATION_NOT_TESTED` exceedance sentinel. Empty selected classes are valid results.

For every alternative, significance compares the exact count ratio with the resident packed
float32 `significanceLevel`, including equality. The displayed p-value is rounded separately.
For example, legacy `1/25` is not significant at packed `.04` (which is smaller than `1/25`),
while `1/32` is significant at the exactly representable `.03125`. The same decision writes
`significant` and class 0 or 1; the legacy tail/count construction is unchanged.

## Outputs and completeness

- `membership`: joint selected-class binary indicator, zero when any variable is missing.
- `boundaries`: `[lower, upper]` pairs in primary-then-additional variable order, including exterior infinities.
- `validCount`: source count per variable in the same order.
- `observed`, `significant`: optional source-row shared inference outputs (join count and binary indicator).
- `tested`: optional source-row uint32 indicator, 1 only when that row was actually tested. It is
  zero for missing, isolated, zero-observed and suppressed rows, and is overwritten every submission.
- `exceedances`, `pseudoPValues`: required shared inference outputs.
- `classes`: `GPU_QUANTILE_LISA_CLASS`: 0 not significant, 1 significant, 2 undefined/masked,
  3 isolated. Unlike upstream `GetClusterIndicators`, which collapses exceptional rows to zero,
  these retain the label distinctions declared by its join-count class. Class 0 can also be an
  untested row; request `tested` to distinguish that case without changing the existing codes.
- `invalidQuantile`: one when any variable's resident k/q is invalid; membership is zero and no row is tested.
- `status.overflow`: shared inference invalid-count, neighbor-capacity or upstream weights-overflow
  flag. Optional `status.invalidCount` is forwarded by shared inference (zero for these binary
  inputs). Read `invalidQuantile` and `status.overflow` before presenting results. A capacity
  failure is not an isolated row. Optional one-word uint32 cause ports are independent:
  `status.weightsOverflow` forwards final input loss, `status.candidateOverflow` forwards
  intermediate input loss, `status.neighborOverflow` records a local neighbor-cap failure and
  `status.invalidInput` records an invalid resident permutation count or local significance cutoff
  outside float32 (0, 1). Every supplied cause is
  overwritten on every submission, including zero after a corrected resubmission.

A supplied root or chunk `weights.status.complete` is a one-word packed uint32 positive certificate: zero suppresses
all inference, sets `status.overflow` and writes `analysisCount = 0`, even with both input overflow
flags clear. One permits inference again on the next submission. The optional final and candidate
cause outputs retain their separate meanings; an incomplete certificate alone sets neither cause.

Every terminal output must be a distinct packed caller-owned view to permit readback. Internal
scratch is graph-owned. Inputs and outputs must not alias. Destruction follows normal graph and
buffer ownership; the contributor does not destroy borrowed inputs.

The optional uint32 `output.analysisCount` reports the joint finite, unmasked population after
the inference stage's one-pass island removal. `output.validCount` counts each variable's finite,
unmasked rows before that removal and before intersecting the variables. For example, an
eleven-row source with one masked row, one nonfinite row and one island can report validCount 9
and analysisCount 8. The count view is overwritten on each encoding, including invalid quantile
settings that leave the analyzed population empty.

## Scope and evidence

There is no existing CPU twin. Multivariate output implements the selected-quantile intersection
and joint local join count. An empty joint class yields zero membership and observed counts on
analysis rows, untested p-values and no significant classes. Pinned libgeoda's `MultiJoinCount`
has an additional bivariate no-colocation fallback when the intersection is empty; this API keeps
the explicit intersection definition and does not switch statistics. Missing-mask handling follows
the per-variable omission and joint validity rule above.

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
Exact inference is asserted against the independent uniform Philox oracle, including the explicitly selected legacy
tail and its all-ties degeneracy. The inclusive-tail GPU regression is queued for device execution.

{/* parameters: GPUQuantileLISA generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

## `GPUQuantileLISA` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPUQuantileLISA`.

- Packer: `getGPUPermutationParameterValues(settings, options?)`, writes `GPU_PERMUTATION_PARAMETER_LENGTH` (4) `uint32` values.
  - Input port: `parameters`.
  - `options` (`{maximumPermutations?: number}`)

Settings of `getGPUPermutationParameterValues`:

| Field | Type | Description |
|---|---|---|
| `seed` | `number` | Random seed, an integer in `[0, 2^53)`. Results are a pure function of the seed, the inputs and `permutationCount`, independent of the dispatch shape. Per frame. |
| `permutationCount` | `number` | Number of permutations `P`, in `[1, maximumPermutations]` of the contributor. The count is never clamped: a value outside that range makes the kernels write NaN p-values (local: rows are not tested and `output.status.overflow` is set; global: the `permutationCount` field of `output.summary` is NaN). Validate on the CPU with the contributor's `getParameterValues` or the `maximumPermutations` option of `getGPUPermutationParameterValues`. |
| `significanceLevel` (optional) | `number` | Significance level for the local mask, strictly in (0, 1) after float32 rounding. Defaults to 0.05. Ignored by the global test. |

{/* /parameters */}
