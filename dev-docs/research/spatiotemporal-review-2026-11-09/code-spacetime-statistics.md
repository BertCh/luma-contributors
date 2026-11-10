# Space-time statistics code review (read-only)

Scope: `@luma-contributors/analysis` space-time statistics: emerging hot spots, space-time tests
(Knox, Mantel), scan statistics, distribution dynamics (Markov family), the three recipes, the
dataframe time contributors (time-window filter, temporal reduction, calendar buckets), and raster
change detection / temporal composites / seasonal break. Trajectories are out of scope.

Paths below are relative to `modules/analysis/src/` unless they start with `modules/`, `docs/` or
`dev-docs/`. Line numbers are from the working tree on branch `upstream-first` (2026-11-09).
Tags: **CONFIRMED** means I traced the code path; **SUSPECTED** means it depends on an external
reference I could not check here, or on a plausibility argument.

Roadmap rows I do not repeat: ST08 (rank Markov ties, missing panels, more than 64 states), RS06
(irregular dates in raster composites and filters), RS11 (multi-break BFAST), LN-V5 (space-time cube
lens), KE-UI (micro/nanosecond admission in the kepler host).

---

## 1. Inventory

| Symbol | File | Operation | Time model | Capacity limits | Reference oracle | Spec files |
| --- | --- | --- | --- | --- | --- | --- |
| `GPUTimeWindowFilter` | `dataframe-analysis/time-window-filter/gpu-time-window-filter.ts` | Per-frame window or interval filter, fades, compaction, sorted binary-search path | Three modes: f32 relative; f32 double-single (`timestamps` + `timestampsLow`, `splitTimestamps`); Int64 words `uint32x2` in any unit (`time-words.ts`). Window edges carry an f32 fraction | Rows unbounded except buffer size; sorted mode only for instants, no track counts | `parity-wp6-time-window-*.json` (`parity-wp6-round5.py`) | `test/gpu-dataframe/time-window-filter/*` (incl. `-int64`, `-sorted`, `time-words.node.spec.ts`) |
| `GPUTemporalReduction` | `dataframe-analysis/temporal-reduction/gpu-temporal-reduction.ts` | Per (cell, fixed-width bucket): count, min, max, first, last; occupied-slot list | f32 relative (`origin`, `width` f32) or Int64 words with integer `width` in `[1, 2^32 - 1]` units | f32 mode at most 2^20 buckets (`:53`); `cellCount * bucketCount` below `0xfffffffe`; dense outputs must fit one binding | `parity-wp6-temporal-reduction.json` (`parity-wp6-dataframe-reductions.py`) | `test/gpu-dataframe/temporal-reduction/*` |
| `GPUCalendarBuckets` | `dataframe-analysis/calendar-buckets/gpu-calendar-buckets.ts` | Civil fields (year..quarter, ISO week), hour-by-weekday matrix | Int64 words, **epoch milliseconds only** (`MILLISECONDS_PER_DAY` at `:372`); fixed offset in minutes, per-row offset, or GPU-searched transition table (seconds) | Days within +-1e9; offsets within +-24 h | Python `zoneinfo` (`dataframe-time-zones.py`); JS `Date` oracle | `test/gpu-dataframe/calendar-buckets/*` incl. `time-zone-reference.node.spec.ts` |
| `getGPUTimeZoneTransitionTable` | `dataframe-analysis/calendar-buckets/time-zone-transitions.ts` | CPU table from `Intl` (daily sampling and bisection) | Epoch ms | Misses transitions closer than one day | zoneinfo fixture | same |
| `GPUEmergingHotSpots` | `gpu-spatial-analysis/emerging-hot-spots/gpu-emerging-hot-spots.ts` | Space-time Gi* per bin (lattice or CSR weights), FDR, Mann-Kendall on Gi* series, 17 + 1 categories | Integer slice index (cube column), no time values | `sliceCount <= 256`; `maximumRadius <= 32`; cells < 2^24; bins <= binding/4 (2^25 at 128 MiB, `:472-475`); **bins < 2^24 whenever FDR or p-value outputs are used (the default)** (`:333-339`) | esda `G_Local` on the expanded cube, scipy `kendalltau`, statsmodels `multipletests` (`parity-wp1-emerging-*.py`, `statistics-emerging-hot-spots.py`) | `test/gpu-spatial-analysis/emerging-hot-spots/*` |
| `addSpaceTimeHotSpotsRecipe` | `gpu-spatial-analysis/recipes/space-time-hot-spots-recipe.ts` | Events -> calendar -> dense count cube -> `GPUEmergingHotSpots` -> colors | Int64 epoch ms -> local civil fields -> step index (`year`/`quarter`/`month`/`week`/`day`/`hour`) | Same as EHS; counts only (no value cube) | Python `datetime` slice indices | `test/gpu-spatial-analysis/recipes/space-time-hot-spots-recipe*.ts` |
| `GPUKnoxTest` | `gpu-spatial-analysis/space-time-tests/gpu-knox-test.ts` | Knox count over caller pair list, exact T via sort, swap-or-not permutations, Besag-Clifford stop | f32 times relative to a CPU-subtracted origin (`getGPUSpaceTimeRelativeTimes`, `space-time-parameters.ts:138`) | rows < 2^24; permutations <= 65,536; pair list size | In-repo TS oracle; pointpats numbers hard-coded for one small example (`gpu-space-time-tests.spec.ts:245`), no generator | `test/gpu-spatial-analysis/space-time-tests/*` |
| `GPUMantelTest` | `gpu-spatial-analysis/space-time-tests/gpu-mantel-test.ts` | Pearson r of pair distance and abs time difference over listed pairs | Same f32 relative times | Same; full Mantel needs an O(n^2) pair list | skbio `mantel` + scipy (`parity-wp1-mantel*.py`), DKW distribution bounds | same |
| `GPUSpatialScanStatistic` | `gpu-spatial-analysis/scan-statistics/gpu-spatial-scan-statistic.ts` | Kulldorff circular / nearest windows, cylinders, Poisson / Bernoulli / space-time permutation, greedy non-overlapping secondaries | Integer time bucket; cases in `zone * timeBuckets + bucket` | `timeBuckets <= 32` (`scan-statistic-parameters.ts:18`); `zones * timeBuckets <= 2^24` (`:389`); full windows only to 4,096 zones; permutations <= 99,999 and <= 1,024 batches | R SpatialEpi likelihoods per candidate window with injected null tables (`parity-wp1-scan.py`) | `test/gpu-spatial-analysis/scan-statistics/*` |
| `GPUTransitionMatrix`, `GPUTransitionDiagnostics` | `gpu-spatial-analysis/distribution-dynamics/` | Classic and conditional Markov counts, Q / LR homogeneity, GTH steady state, sparsity | Integer period index, **period-major** `t * rows + i` | classes <= 64 | giddy (`statistics-markov.py`) | `test/gpu-spatial-analysis/distribution-dynamics/*` |
| `GPUSpatialMarkov` | `.../gpu-spatial-markov.ts` | Rey (2001) spatial Markov, pooled lag classes (giddy `fixed=True`) | Period-major | classes and lag classes <= 64 | giddy | same |
| `GPULISAMarkov` | `.../gpu-lisa-markov.ts` | Quadrant transitions, optional significance gating | Period-major | 5 states | giddy `LISA_Markov`, esda | same |
| `GPURankMarkov` | `.../gpu-rank-markov.ts` | Full / geographic rank Markov | Period-major, complete cases only | **rows <= 64** (`:54`) | giddy (coverage matrix says 2.3.6, code doc says 2.3.9, `:40`) | same |
| `GPUChangeDetection` | `raster-analysis/change-detection/gpu-change-detection.ts` | Two-slice diff, Welch t, Theil-Sen, Mann-Kendall, change vector | Ordinal slice index, regular | slices <= 256; Theil-Sen <= 64 | SciPy (`parity-wp4-change-detection.py`, `-change-modes.py`) | `test/gpu-raster/change-detection/*` |
| `GPURasterTemporal` | `raster-analysis/remote-sensing/gpu-raster-temporal.ts` | Masked composites, Savitzky-Golay with gaps | Regular slots | slices <= 256; window 3..15 | terra, SciPy (`raster-remote-sensing.py`) | `test/gpu-raster/remote-sensing/*` |
| `GPURasterSeasonalBreak` (+ extended) | `raster-analysis/remote-sensing/gpu-raster-seasonal-break.ts`, `seasonal-break-extended.ts` | Harmonic or dummy season + trend + one **level** jump by BIC; monitoring | Regular slots; `period` in slices | Private QR budget: harmonic about 256 slices; dummy period 12 about **129 slices** (assert at `:65-77`) | NumPy (`raster-seasonal-extended.py`) | `seasonal-extended.spec.ts`, `remote-sensing*.spec.ts` |
| `addPeriodComparisonRecipe` | `gpu-spatial-analysis/recipes/period-comparison-recipe.ts` | Two cell tables, outer join, delta/ratio/z, class breaks | No time: the caller supplies two row sets (or two masks) | table capacities | none temporal | `period-comparison-recipe.spec.ts` |
| `addHotSpotAnalysisRecipe` | `.../hot-spot-analysis-recipe.ts` | Purely spatial Gi* | none | n/a | esda fixture | `hot-spot-analysis-recipe*.ts` |

---

## 2. Time-model findings

There is **no single time contract**. Five incompatible models coexist, and nothing carries the unit
or slice width alongside a cube.

| Model | Used by | Unit / epoch | Precision loss |
| --- | --- | --- | --- |
| A. Int64 words `uint32x2` (`time-words.ts`) | Time-window filter, temporal reduction, trajectories | Any unit, implicit | Comparisons exact; differences rounded to f32 (exact below 2^24 units) |
| B. Int64 words, epoch **ms** fixed | Calendar buckets, space-time hot spots recipe | ms since 1970, UTC + offset | None (integer) |
| C. f32 relative (optionally double-single) | Time-window filter, temporal reduction (f32 mode) | Caller origin | 24-bit mantissa |
| D. f32 relative, origin subtracted on CPU | Knox, Mantel | Caller | `span * 2^-24`; no GPU path from A |
| E. Integer slice or period index | EHS, scan, Markov, change detection, raster temporal, seasonal break | Regular, implicit width | Irregular sampling not representable except as NaN slots |

**T1. Arrow timestamp units are not normalized, but the calendar assumes milliseconds. CONFIRMED.**
`prepareTimeColumn` (`analysis-core/graph-support/arrow-analytics-normalization.ts:343-379`) copies
raw Int64 values in the column's own unit (`second` / `millisecond` / `microsecond` /
`nanosecond`, `:321-331`); only `Date32` days are converted to ms (`:357-360`). The unit is CPU
metadata only. `GPUCalendarBuckets` takes a bare `GraphDataView<'uint32x2'>` with no unit prop and
divides by 86,400,000 (`gpu-calendar-buckets.ts:372`, `:464`). Scenario: a pandas-default
`timestamp[ns]` column uploaded with `makeGPUTimeVectorFromPreparedArrowColumn` and fed to the
calendar or to `addSpaceTimeHotSpotsRecipe` decodes 2024-01-01 as year ~ 55,970. There is no
error: values are within the +-1e9-day validity range, so rows are not flagged invalid. KE-UI
covers admission in the kepler host only, not this library contract.

**T2. Word-mode temporal reduction cannot express common bucket widths. CONFIRMED.**
`getGPUTemporalReductionWordParameterValues` requires `width <= 2^32 - 1` units
(`temporal-reduction-parameters.ts:68`), and the kernel divides by a single u32
(`gpu-temporal-reduction.ts:619-646`). Largest bucket: ms gives 49.7 days (no quarters or years,
and calendar months are irregular anyway), us gives 71.6 minutes, ns gives **4.29 seconds**. So an
hourly cube of `timestamp[ns]` data is impossible in exact mode. The fallback is f32 mode, which
needs a CPU-rebased origin (model C).

**T3. Knox and Mantel have no GPU path from Int64 time words. CONFIRMED.** Their `times` prop is
f32 only, and the helper `getGPUSpaceTimeRelativeTimes` is CPU code
(`space-time-parameters.ts:138-168`). Data already resident as words (from the time-window filter or
Arrow upload) needs a readback or a CPU copy. The WGSL to do it on the GPU exists
(`timeWordsDifferenceToF32`, `time-words.ts:214`) but no contributor exposes it as a column operation.

**T4. Regular-slice assumption everywhere downstream. CONFIRMED.** EHS, scan, Markov, change
detection (Theil-Sen divides by `f32(j - i)`, `gpu-change-detection.ts:561`; the oracle states "Time
is the ORIGINAL ordinal index ... no claim of slope/year"), seasonal break (harmonic angle uses the
slice index) and raster temporal all treat time as a slice index. Missing periods are handled
correctly when encoded as NaN slots (Theil-Sen keeps the true index gap; MK skips NaN; EHS skips NaN
bins). Irregular acquisitions cannot be expressed. No cube carries `(origin, step, unit)`, so a Sen
slope or MK trend has no physical time unit unless the caller tracks it.

**T5. Two panel layouts. CONFIRMED.** Cell-major `cell * T + t` (temporal reduction, EHS, scan,
change detection, raster temporal and seasonal break) versus period-major `t * rows + i` (all
Markov contributors, `period-views.ts`, spatial panel regression). No public transpose contributor
exists. `createPeriodPackNode` only pads, and the spatial-weights transpose is for CSR. See section 5.

**T6. Ties.** Mann-Kendall (EHS and change detection) uses exact f32 equality tie groups with
tie-corrected variance (exact integers in u32; the largest term at n = 256 is about 3.4e7, which
fits). Knox: `<=` threshold, `-0 == +0`. Temporal reduction first/last: lowest row on equal times
(exact in word mode). Rank Markov: giddy reverse-ordinal ties. Raster composites: earliest
observation on ties. These are consistent and documented.

**T7. Time zones.** Only calendar buckets and the space-time recipe know about time zones. The
transition table is exact to the second and checked against zoneinfo. The recipe's `'hour'` step
documents 0 h / 2 h slices on DST days (`space-time-hot-spots-recipe.ts:242`). The `'day'` step has
the same issue in milder form (23 h / 25 h local days, so about 4% count bias on two bins a year).
That is SUSPECTED to be minor and is undocumented. Knox, Mantel, temporal reduction and the time
window are zone-free (absolute instants), which is correct.

**T8. Duplicate Int64 helpers.** Calendar has its own `lessOrEqualInt64` (`gpu-calendar-buckets.ts:500`)
instead of `timeWordsCompare`. It is equivalent but duplicated.

---

## 3. Correctness and statistical validity

**S1. Sporadic hot / cold differs from the current ArcGIS definition, and the code comment claims
the opposite. CONFIRMED (code). The ArcGIS text is quoted from the repo's own oracle.**
`gpu-emerging-hot-spots.ts:1273-1276` labels a cell Sporadic Hot when its final slice is *not*
significant, `H > 0` and `C = 0`. The comment at `:216-217` says "as ArcGIS defines it without
requiring a hot final slice". The project's own oracle `test/oracles/parity-wp1-emerging-hot-spots.py`
(docstring) states: "a neutral final bin with only hot/cold history is Sporadic (the current ESRI
page instead requires a significant final bin)". Scenario: slices `H,_,H,_,_,_,_,_,_,_` with a
non-significant final bin. GPU gives `SPORADIC_HOT (6)`; current ArcGIS Pro gives No Pattern
Detected. This is the dominant category in the documented showcase ("35 are sporadic hot",
`docs/api/gpu-spatial-analysis/operators-statistics.md:1666`, `:1788`), so the headline output
depends on the deviation. Fix: an option or the default `sporadicRequiresFinal`, and correct the
comment and the docs.

**S2. EHS on a lattice count cube counts never-occupied cells as valid zeros. CONFIRMED (code);
ESRI comparison SUSPECTED.** In the recipe's lattice mode, every `width * height` cell enters the
global mean, the standard deviation and the FDR family (`uint32` values are always valid,
`gpu-emerging-hot-spots.ts:507-514`). For a city lattice with water or parks, structural zeros
lower the mean, raise Gi* for every occupied cell, and enlarge the FDR family. I believe ArcGIS
space-time cubes keep only locations with at least one point. Verify that before acting. A cheap
"ever occupied" auto `cellMask` (one per-cell OR over slices) would match it.

**S3. The EHS trend p-value is anti-conservative. Known and documented.** Mann-Kendall is run on
overlapping-window Gi* series without an autocorrelation correction. Documented as a screening
threshold. A Hamed-Rao or Yue-Wang modified variance would turn it into a test. Listed in section 6.

**S4. EHS Gi* and classification formulas are correct. CONFIRMED.** Lattice
`spread = W(n - W)/(n - 1)` equals `(n S1 - W^2)/(n - 1)` for binary weights (`:654-655`). The
weights mode uses the general form. Population SD, two-pass. Missing bins are excluded from
numerator and weight sums. The window is current plus `k` previous slices, as in ESRI. The
New / Consecutive / Persistent / Intensifying / Diminishing / Oscillating / Historical rules match
the ESRI table except S1, and the documented precedence of New over Oscillating. Mann-Kendall S,
tie term and continuity correction are exact. The Theil-Sen median selection in change detection is
also correct (heap of the `floor(N/2) + 1` smallest slopes, `gpu-change-detection.ts:547-597`).

**S5. Mantel cannot reproduce the pointpats default, and the doc advice is wrong for time.
CONFIRMED (code); pointpats signature from memory.** pointpats
`mantel(..., scon=1, spow=-1, tcon=1, tpow=-1)` correlates `(d + 1)^-1` with `(|dt| + 1)^-1`. The
GPU uses raw `d` and `|dt|` (`gpu-mantel-test.ts:358-359`). Its doc says "transform `distances` and
`times` first if wanted" (`:142-144`). That works for distances, but no transform of `times`
yields `(|t_i - t_j| + c)^p`. The cross-reference table (`docs/api/gpu-spatial-analysis-cross-reference.md:84`)
lists `pointpats.Mantel` as the counterpart. Parity tests use skbio with untransformed matrices, so
they pass. Scenario: pointpats default on any dataset gives a different r and p than `GPUMantelTest`
with no way to match. Fix (S): add `distanceTransform` / `timeTransform` `{constant, power}`
applied in the partials kernel.

**S6. Knox has weak external evidence. CONFIRMED.** No Python generator mentions Knox. The only
external check is hard-coded pointpats numbers for one small example (observed 29, expected
31.76, `gpu-space-time-tests.spec.ts:245-273`). The permutation null is checked only against the
in-repo swap-or-not oracle, which uses the same permutations. The Poisson helper uses `P(X >= obs)`
while pointpats reports `P(X > obs)` (spec comment `:269`). The library's choice is the correct tail,
but this is not stated in the API doc.

**S7. Scan statistic versus SaTScan. CONFIRMED (code).**
- Retrospective only. There is no prospective mode (windows forced to end at the last bucket, plus
  optional adjustment for earlier analyses). `grep prospective` finds nothing.
- Circular and nearest windows only; no elliptic windows.
- Secondary clusters: one candidate per center (its best window); a center whose best window
  overlaps an earlier cluster is dropped entirely (`round-mask`, `:1756-1766`). Smaller
  non-overlapping windows at that center are not considered. I believe this matches SaTScan's
  "most likely cluster per centroid" (SUSPECTED), but it is untested against SaTScan. The parity
  spec runs `maximumClusters: 1` (`parity-wp1-scan.parity.spec.ts:86`), and secondaries are
  checked only against the in-repo oracle (`gpu-spatial-scan-statistic.spec.ts:207`).
- Ranking `(1 + #{max >= LLR}) / (P + 1)` and secondaries ranked against the same maxima match
  SaTScan's standard p-value. There is no Gumbel approximation option.
- References are SpatialEpi likelihoods with injected null tables. The Philox sampling itself is not
  validated against an independent sampler. That is acknowledged in the oracle docstring.

**S8. Markov family.** It matches giddy on the tested scenes (counts exact; Q, LR and steady state
compared to 1e-9). The degrees of freedom intentionally differ from giddy by the number of empty
pooled rows (documented). LISA Markov drops rows exactly at the mean or with zero lag; esda would
call them low (documented). Mean first passage times, `x2_rpvalue` and the `relative` option are
absent (documented). Doc drift: rank Markov says giddy 2.3.9 (`gpu-rank-markov.ts:40`), while the
coverage matrix says 2.3.6 (`dev-docs/coverage-matrix.md:148`).

**S9. Seasonal break is level-shift only. CONFIRMED.** The break regressor is a step dummy
(`gpu-raster-seasonal-break.ts`, `design[4u * slices + slice] = 1.0` after the split). BFAST and
`bfastmonitor` breaks change trend slope and level per segment. The doc says "BFAST-inspired", so
this is not a mislabel, but a slope change (for example degradation onset) is detected only through
its level component. Not in RS11, which is about multiple breaks.

**S10. The recipe's hour step can overflow i32. CONFIRMED but low impact.** The origin year is
accepted up to +-1,000,000 (`space-time-hot-spots-recipe.ts:385`). The `'hour'` origin step
`daysFromCivil * 24` then reaches about 8.8e9 and is pasted into WGSL as an i32 literal (`:475`),
which is a shader compile error rather than a validation error. Events beyond about +-245,000 years
wrap silently in `daysFromCivil(...) * 24` (`:467`). Clamp the origin year for `'hour'`.

---

## 4. Scaling walls

| Contributor | Wall | Where | Consequence |
| --- | --- | --- | --- |
| EHS | bins < 2^24 with FDR (default) or p-value outputs | `gpu-emerging-hot-spots.ts:333-339` | 65k cells x 256 slices, or 16.7M bins, is the ceiling at defaults. Above it only `'none'` correction works |
| EHS | bins <= `maxStorageBufferBindingSize / 4` | `:472-475` | 2^25 at the 128 MiB default |
| EHS | slices <= 256, radius <= 32 | parameters `:14-17` | Ten years of weekly slices does not fit (520) |
| Recipe | counts only; one u32 atomic per event | recipe `:572-590` | No mean/sum cubes (ESRI summary fields) |
| Temporal reduction | word width < 2^32 units; f32 at most 2^20 buckets | T2 | See T2 |
| Scan | `timeBuckets <= 32` | parameters `:18` | Monthly 3-year maximum; daily impossible |
| Scan | `zones * timeBuckets <= 2^24`; full windows only to 4,096 zones; work about `(P+1) * zones * windowZones * B^2` with one thread per (center, replicate) looping all windows | `:389`, layout | 4,096 zones x 4,096 window x 32^2 x 999 is about 1.7e13 window evaluations. Long single dispatches risk device watchdog resets (SUSPECTED; not measured) |
| Scan | <= 1,024 unrolled batches (4 nodes each) | `:52-55` constants | Graph size grows with `maximumPermutations` |
| Knox / Mantel | rows < 2^24; permutations <= 65,536; work O(P x pairs) | `space-time-kernels.ts:16-22` | Fine for banded pairs |
| Mantel (full) | needs n(n-1)/2 pair slots in one binding | pair list | About n <= 8,190 at 128 MiB. Above that only band-truncated r |
| Markov | classes <= 64; rank Markov rows <= 64 (n x n states) | `gpu-transition-matrix.ts:25`, `gpu-rank-markov.ts:54` | Rank Markov is a toy size (ST08 known) |
| Change detection | slices <= 256; Theil-Sen <= 64 with a private heap of up to 1,009 f32 per thread | `gpu-change-detection.ts:27-28`, `:549` | Register spill; correctness unaffected |
| Seasonal break | private QR budget 2,048 words per thread | `gpu-raster-seasonal-break.ts:65-77` | Monthly dummy seasons: about 129 slices (about 10.7 years); harmonic about 256 |
| Calendar | none material | | O(n), all integer |

Brute force versus indexed: Knox and Mantel are pair-list driven (indexed through
`GPUNeighborSearch`). The Knox T count is O(n log n) via `GPUSort` plus binary search. The scan
neighbor tables use a radix select per center, O(zones^2). The EHS lattice is O(bins x (2r+1)^2) per
instant, with temporal factoring.

---

## 5. Composability map

GPU-resident edges (no readback):

```
Arrow Int64 -> makeGPUTimeVectorFromPreparedArrowColumn -> uint32x2 words   [unit not checked, T1]
words -> GPUTimeWindowFilter.outputMask (u32/row) -> GPUTemporalReduction.mask
                                                   -> GPUCalendarBuckets.mask / recipe.mask
                                                   -> addPeriodComparisonRecipe before.mask / after.mask
words + cellIds -> GPUTemporalReduction.counts (cell-major u32) -> GPUEmergingHotSpots.values
                                                                -> GPUSpatialScanStatistic.cases (if buckets <= 32)
GPUTemporalReduction.minimums/maximums/first/last (f32, NaN = missing) -> EHS, GPUChangeDetection,
                                                                         GPURasterTemporal, GPURasterSeasonalBreak
words -> GPUCalendarBuckets -> (recipe adapter kernel) -> count cube -> EHS -> GPUColorScale
EHS.giZScores (cell-major f32) -> GPUChangeDetection (Theil-Sen or MK of Gi*)
GPUNeighborSearch (radius) -> GPUKnoxTest.pairs / GPUMantelTest.pairs
```

Forced CPU work or missing glue:

1. **Int64 words to f32 relative times** for Knox and Mantel. The origin is subtracted on the CPU
   (T3).
2. **Cell-major to period-major transpose.** None exists, so a temporal-reduction cube or an EHS Gi*
   cube cannot feed `GPUTransitionMatrix`, `GPUSpatialMarkov`, `GPULISAMarkov` or the panel
   regressions without a hand-written kernel (T5).
3. **Calendar (irregular) buckets to a cube** exist only inside the recipe's private adapter kernel.
   `GPUTemporalReduction` takes timestamps, not bucket indices, so a monthly min/max/first/last cube
   needs either the recipe (counts only) or passing calendar step indices as fake f32 "times" with
   width 1.
4. **Value cubes (sum or mean per cell-slice).** Neither `GPUTemporalReduction` nor the recipe
   provides them. `GPUGroupStatistics` can sum per key, and the recipe already has a key-and-scatter
   path for counts (`:592-685`), so extending the scatter to sums is the cheapest route.
5. **Dense cell IDs.** Lattice binning lives inline in the recipe (`:453-458`). H3 and Quadbin keys
   from `GPUPointToCell` need a key-to-row lookup (the recipe comment says so).
6. **u32 counts to f32 slices.** Change detection and the raster contributors require float32, so
   count cubes need a one-line conversion kernel.
7. **Period comparison has no time input**: two time-window masks are the composition route, which
   works but is not a recipe option.

Conclusion: there *is* a space-time cube path (count cube via recipe or `GPUTemporalReduction`), and
`filter -> cube -> Gi* -> Mann-Kendall -> Theil-Sen` runs fully on the GPU. Markov chains and the
Knox / Mantel tests are islands.

---

## 6. Gaps and extension points

| Gap | Reuses | Size |
| --- | --- | --- |
| Jacquez k-NN test (pointpats `jacquez`), absent everywhere including the roadmap | `GPUNeighborSearch` k-NN (space), `GPUSort` of times (the temporal k-NN of an event is a contiguous rank interval `[lo_i, hi_i]`, so membership under a permutation is the O(1) check `rank(pi(j)) in [lo_pi(i), hi_pi(i)]`), Knox partials, swap-or-not shuffle and stopping machinery | S |
| Prospective space-time scan | Evaluate kernel: restrict `last == BUCKETS - 1` behind a parameter word; a recurrence-interval helper on the CPU | S (basic), M (with adjustment for earlier analyses) |
| Mantel pointpats transforms `(x + c)^p` | Mantel partials and pair-stats kernels | S |
| Unit-aware calendar (s / us / ns) | `divideUnsigned64` (divisor below 2^28 covers 1e3 and 1e6; ns needs two steps) | S |
| Word-mode bucket width >= 2^32 units | 64-by-64 division in `WORD_BUCKET_BODY` | S |
| Words to relative f32 column node (unblocks T3) | `timeWordsDifferenceToF32` | S |
| Cell-major and period-major transpose node | `createPeriodPackNode` pattern | S |
| Sum / mean value cube | `GPUGroupStatistics` sums plus the recipe scatter | S-M |
| Pettitt / CUSUM change point on vector or raster panels | Mann-Kendall S loops in `gpu-change-detection.ts` (Pettitt's U_t is a running sign sum) | S |
| Change point on vector panels (BFAST-like) | `GPURasterSeasonalBreak` already takes generic cell-major `cellCount x sliceCount` f32. Feed a temporal-reduction cube; mostly documentation and a count-to-f32 conversion | S |
| Bivariate / differential Moran over time | `GPUSpatialLag`, `getPeriodView`, existing bivariate statistics (`gpu-additional-spatial-statistics.ts`); differential Moran needs a period-difference column | S |
| Modified Mann-Kendall (Hamed-Rao / Yue-Wang) for the EHS trend | EHS Mann-Kendall kernels plus a lag-autocorrelation pass | M |
| Space-time kernel density (cube of per-slice KDE) | `GPUKernelDensity` with per-event `weights` = temporal kernel per slice (T unrolled nodes or a slice loop) | M |
| Space-time K function (Diggle) | `pair-statistics` histogram / `GPURipley` pair loops with a second (time-lag) bin axis; `GPUInhomogeneousPointPattern` for weighting; envelope via `GPURipleyEnvelope` pattern | M |
| ST-DBSCAN | `GPUSpatialClustering` grid DBSCAN with an added `abs(dt) <= epsTime` predicate and a time input | M |
| Space-time weights `W_s (x) W_t` for space-time LISA / Moran on cubes | `GPUSpatialWeightsAlgebra` (`block`, `union`) plus a new Kronecker op | M |
| Elliptic scan windows | New neighbor tables per (center, shape, angle) | L |
| Scan with more than 32 buckets | Private per-thread `caseSums[BUCKETS]` arrays need a cooperative redesign | M-L |
| Irregular-time Theil-Sen / Mann-Kendall / seasonal (true dates) | Add an optional f32 or words time column to the change-detection and seasonal kernels | M (overlaps RS06 for composites only) |

---

## 7. Top recommendations, ranked

1. **Fix or flag the time-unit contract (T1, T2).** Add a `unit` prop to `GPUCalendarBuckets` and
   the recipe, or reject non-ms prepared columns, and lift the word-width cap. Today a `timestamp[ns]`
   column silently produces garbage calendars and cannot be bucketed hourly. Small change, high blast
   radius.
2. **Resolve the Sporadic definition (S1).** Add an option to require a significant final slice
   (current ArcGIS), decide the default, and correct the comment at
   `gpu-emerging-hot-spots.ts:216-217` and the showcase captions.
3. **Add the three glue nodes**: words to relative f32, cell-major to and from period-major transpose,
   and a sum/mean value cube. Together they connect Knox / Mantel and the Markov family to the cube
   pipeline with no readback (section 5).
4. **Add pointpats-compatible Mantel transforms (S5)** and an external Knox generator with a
   null-distribution check (S6). Both are cheap and close the pointpats claim in the
   cross-reference.
5. **Add the Jacquez k-NN test.** It is the missing third pointpats space-time test, and it reuses
   the Knox machinery almost entirely.
6. **Add a prospective scan mode** and a SaTScan (or SpatialEpi `kulldorff`) secondary-cluster
   fixture with `maximumClusters > 1`.
7. **Add an ever-occupied auto mask** for lattice cubes in the space-time recipe (S2, verify the ESRI
   behavior first).
8. **Add a modified Mann-Kendall variance option** for the EHS trend so Intensifying and Diminishing
   become a test rather than a screen (S3).
9. **Doc fixes**: rank Markov giddy version (2.3.6 vs 2.3.9), the `getKnoxPoissonPValue` tail versus
   pointpats, DST day-length bias for the `'day'` step, and the seasonal break being level-shift only.
