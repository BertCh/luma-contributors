# Change points and the expectation-based scan

STX-06 and STX-07. Import from `@luma-contributors/analysis/gpu-spatial-analysis`. Cubes use the layouts of
`GPUCubeLayoutTranspose`: `'cell-major'` (`cell * sliceCount + slice`) or `'period-major'`.
Both contributors, both GPU/CPU operation families and the exported CPU helpers reject any
other layout at construction or helper entry; unknown strings never select a fallback layout.

Contributors and both operation forms snapshot layout, dimensions, cost and output routing at
construction. To change these options, construct a new contributor or operation. Updating the
contents of the existing value and parameter views remains a per-frame operation.

## Per-location change points: `GPULocationChangePoints`

`GPULocationChangePoints` runs unpruned optimal partitioning per cell: `F(t) = min_s F(s) + C(s + 1..t) + beta`, the same optimum PELT
finds, with `sliceCount^2 / 2` bounded segment costs and no pruning. Missing (NaN) slices are skipped;
`minimumSegmentLength` counts finite slices. The mean cost retains raw observations and forms
segment-local offsets in data units. The other Normal costs standardize per location using the
series' own mean and standard deviation (the Poisson cost uses a deviance form). Changing units preserves the partition
when the penalty is scaled with its cost: by the square of the value scale for mean/slope, by the
value scale for Poisson, and unchanged for variance/mean-variance. Additive shifts preserve the
Normal costs' partitions; Poisson counts must retain their nonnegative domain.
For standardized Normal costs the center is the mean of offsets from the first finite observation, and
centering remains in those offset coordinates. This preserves small representable differences
under large translations; rounding an absolute mean must not invent a variance change point.
The mean cost does not subtract that global origin: doing so can erase small real changes
after large mixed-sign observations. Its required change count and overflow describe the full
optimal partition even when only the earliest changes fit the output capacity.
The moments of each segment are running sums that begin at the segment's own start, about its first value; no
difference of two long prefix sums is formed, so a quiet segment after a loud one keeps f32 relative accuracy at
`sliceCount = 512`.

Variance floor (`'variance'` and `'mean-variance'`): a segment variance below `1e-11` times the variance of the whole
series is raised to that value, so a constant segment has a finite cost whatever the unit. (Until 2026-11-10 the floor
was `1e-11` in data units, which clamped every segment of a series in small units, for example values near `1e-6`,
and returned no change point.) The R `changepoint` package substitutes `1e-11` in data units only when a segment
variance is not positive; these rules can affect different small-variance segments. A constant series has no change point, and its `optimalCost` uses `1e-11`
in data units.

| `cost` | Segment cost | Smallest segment |
| --- | --- | --- |
| `'mean'` | `sum (y - mean)^2`, a unit-variance Normal cost: standardize `values` or scale `penalty` | 1 |
| `'variance'` | `n (ln(2 pi) + ln(s^2) + 1)`, `s^2 = sum (y - m)^2 / n`, `m` the mean of the whole series (the `changepoint` package's `cpt.var`) | 2 |
| `'mean-variance'` | As `'variance'` with each segment's own mean | 2 |
| `'poisson'` | `2 x (ln n - ln x)`, `x` the segment total | 1 |
| `'slope'` | Residual sum of squares of a line per segment, true slice positions | 3 |

`penalty` and `minimumSegmentLength` are per-frame; pack them with `getParameterValues`. The R `changepoint` package uses the `penalty` SIC
`(p + 1) ln n`, with `p` the parameters added per segment. With the unit-variance `'mean'` and `'slope'` costs, scale the
penalty by the variance.

Outputs per cell: `changePointCount` (clamped to `maximumChangePoints`, the earliest kept), `changePoints` at `cell * maximumChangePoints + index`, the first slice of each later
segment, unused rows `GPU_LOCATION_CHANGE_POINTS_NONE`. Optional: `requiredChangePointCounts` (unclamped), `optimalCost` (the sum of the
segment costs plus `penalty` per change point), and `status.overflow` (`1` when any cell needed more than the capacity).
A series with fewer finite slices than `minimumSegmentLength` has no feasible partition, including
a flat series: counts are zero, change-point slots are unused, and `optimalCost` is NaN. Negative
Poisson observations produce the same suppressed result. A lone allowed Poisson observation `x > 0`
has cost `-2 x ln x`. The first segment carries no penalty, which also preserves its residual when
the change penalty is much larger than that residual.

Limits: `sliceCount <= 512` (`<= 256` for `'slope'`, whose time sums are exact unsigned integers up to that length), `maximumChangePoints <= 128`. In the slope cost, when one line explains almost all the variance the cancellation of
standardized moments limits the absolute residual to about `2e-6 * sliceCount * variance`.

Validated against ruptures 1.1.10 Pelt for `'mean'`, `'slope'` and `'mean-variance'`. The variance and Poisson costs are checked only against a binary64 reference of the published `changepoint`
definitions in the spec; the R package is GPL and R-only and was not run, so the external R fixture is pending.

{/* lens:GPULocationChangePoints */}

The operation forms are `GPULocationChangePointsOperation` and
`CPULocationChangePointsOperation`; both take the same cost, layout and dimension names.
The CPU operation is the host reference form.

{/* parameters: GPULocationChangePoints generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPULocationChangePoints` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPULocationChangePoints`.

- Packer: `instance.getParameterValues(settings)`, writes the `float32` parameter view.

Settings of `instance.getParameterValues`:

| Field | Type | Description |
|---|---|---|
| `penalty` | `number` | Penalty `beta` per added change point, finite and non-negative, on the scale of the cost: see `GPULocationChangePoints`. For the `changepoint` package's `SIC`: `(p + 1) ln n` with `p` the parameters added per segment (1 for mean, variance and Poisson, 2 for mean-variance and slope, the Poisson and Normal likelihoods assume unit-variance scaling for mean and slope). |
| `minimumSegmentLength` (optional) | `number` | Smallest number of finite slices per segment. Defaults to the smallest length the cost is defined on: 1 for mean and Poisson, 2 for variance and mean-variance, 3 for slope. |

{/* /parameters */}

## Pettitt test: `GPUPettittTest`

`K = max_t |U_t|`, `U_t = sum_{i<=t} sum_{j>t} sign(x_i - x_j)`, over the finite series in exact integers. Outputs one row per cell: `statistic` (uint32), `changeSlice` (the first slice after the
change, `GPU_PETTITT_TEST_NO_CHANGE_SLICE` under 2 finite slices) and `pValues` (`min(1, 2 exp(-6 K^2 / (n^3 + n^2)))`).
`pValues` uses Pettitt's large-sample approximation. Short or heavily tied series can give a
different decision from an exact conditional permutation test. For `[0,0,0,0,1,1,1,1]`,
`K = 16` and `changeSlice = 4`: the approximation gives `p = .13896690`, while exact conditional
enumeration of the 70 distinct arrangements of four zeros and four ones gives
`2/70 = .02857143`. The statistic's integer arithmetic does not make its p-value exact.

`sliceCount <= 512`. Checked against pyhomogeneity 1.1.
The operation forms are `GPUPettittTestOperation` and `CPUPettittTestOperation`; both return the
same approximate p-value and use the same layout and dimension names.
The first maximum wins; when all finite values are tied and there are at least two, `changeSlice`
is the second finite slice even though `statistic = 0` and `pValues = 1`.
With fewer than two finite observations the cell is untested: `statistic = 0`,
`changeSlice = GPU_PETTITT_TEST_NO_CHANGE_SLICE`, and `pValues = NaN`. Multiple-testing
adjustment excludes those cells from its family; tested constant cells retain `pValues = 1`.

{/* lens:GPUPettittTest */}

## Corrected emerging hot spot trend

`GPUEmergingHotSpots` and `addSpaceTimeHotSpotsRecipe` take `trendAutocorrelationCorrection`:

- `'none'` (default): the unmodified Mann-Kendall test, as before.
- `'hamed-rao'` (Hamed and Rao 1998): ranks of the Sen-detrended series, lags whose autocorrelation exceeds `trendCorrectionCriticalZ / sqrt(n)` (default 1.96).
- `'yue-wang'` (Yue and Wang 2004): autocorrelation of the Sen-detrended values over all lags.

`trendCorrectionMaximumLag` limits the lags (default every lag). An explicit value must be an integer
from 1 through 4,294,967,295; only lags through `n - 1` of the finite series are used. Missing slices
are removed before detrending, so the slope and autocorrelation use consecutive finite-series
positions. Both methods divide each lag's covariance sum by the full centered sum of squares,
without a lag-specific divisor. Hamed-Rao uses the absolute autocorrelation for its significance
filter. A nonpositive correction factor falls back to the unmodified variance; constant detrended
series also leave that variance unchanged. Exact comparisons of the detrended values (or their
midranks for Hamed-Rao) identify this constant case before using the centered variance, so rounding
in mean division cannot invent autocorrelation. No tolerance is used to discard small variation.
The correction is compile-time and needs
`sliceCount <= GPU_EMERGING_HOT_SPOTS_MAXIMUM_CORRECTED_TREND_SLICE_COUNT` (63), where each cell can select the Sen slope in one
invocation. `trendS` and the category codes keep their meaning. Checked against pymannkendall 1.4.3.

## Expectation-based scan: `GPUExpectationBasedSpaceTimeScan`

The space-time permutation model of `GPUSpatialScanStatistic` conditions on the zone and bucket margins, so a rise shared by every zone is
explained by the bucket margin and is not a cluster (Tango et al. 2011). This scan compares the counts with a supplied `baseline` instead:
`F = C ln(C / B) + B - C` for `C > B`, Neill et al. (2005). It scans windows ending at the last bucket.

For each center and window length, the neighborhood is its nearest zones (the kNN table the scan builds,
`maximumNeighbors <= 64`). Fast Subset Scan (Neill 2012): the best subset is a prefix of the zones sorted by
`C_i / B_i`, so an exact search over the subsets of the neighborhood is a fixed sort and one prefix scan.
Ordering compares exact cross-products of the finite float32 zone sums, so distinct ratios
remain ordered even when both quotients would overflow. Exact ratio ties retain neighbor-slot order.
Cells with non-positive or non-finite baselines contribute neither observed nor expected counts.
The remaining cells are summed per zone; a zone with no valid cell in the window is excluded.
The most likely cluster is reported with its member zones. For example, counts `[100,6,8]` with
baselines `[0,2,2]` contribute 14 cases and expectation 4 to the final two/three-bucket window,
not 114 cases. A negative or missing baseline cannot cancel a valid cell's expectation.

Construction captures settings, contexts and output-port maps. Mutating the caller's props later
does not change the validated graph; buffer contents and parameter views remain per-frame inputs.

Neighbor discovery defaults to `nearestZoneSearch: 'auto'`, which uses the same exact coordinate
indexes as the spatial scan for truncated lists (at most 64 rows). A whole-domain list uses
full-scan. Neighbors are ordered by `(separation key, zone row)`; a zero-distance tie with lower
rows can displace the center itself. For three coincident zones and `maximumNeighbors: 1`, every
neighborhood contains only row 0, so cases on rows 1 and 2 are outside those neighborhoods.
`'indexed'` selects it explicitly; `'full-scan'` selects the exhaustive route for comparison.
`contributor.nearestZoneSearch` reports the resolved route, and
`contributor.estimatedFullScanSeparationCount` bounds exhaustive separation evaluations per submission
by `5 * zones^2` for a shorter-than-domain list, or `zones^2` for a whole-domain list, excluding sorting.
Both scan classes also expose `estimatedIndexedCandidateSeparationCount = N²`,
`estimatedIndexedAxisSeparationCount = 2N(N+1)`, their sum as
`estimatedIndexedSeparationCount`, and `indexedRadixSortCount = 2` for each indexed frame,
where `N` is the zone count. These are conservative separation-call bounds and exclude sorting
work. Equal-latitude great-circle inputs can require all `N²` candidate calls: latitude pruning
cannot narrow them. No longitude pruning is assumed across wraps or poles.
Both routes order by `(separation key, zone row)` in planar and great-circle contexts. The index is
rebuilt each submission; axis-degenerate data can still require quadratic search work. See the
[spatial scan neighbor discovery details](./operators-dynamics.md#gpuspatialscanstatistic).

`GPUExpectationBasedSpaceTimeScan.limits` describes route conditions and limits;
`GPUExpectationBasedSpaceTimeScan.getExecutionPlan(shape, deviceLimits)` checks the numeric
shape and reports the selected route, batch sizes, per-binding bytes, constraints and `feasible`
before allocating buffers. Count-table bindings use the smaller of the producer's 128 MiB cap
and the device limit. Indexed plans also include the published radix sort's key/value and
histogram/offset bindings (at least 64 bytes for a multirow radix sort); full-scan plans omit those
radix-only requirements. Supply `zones`, `timeBuckets`, `maximumNeighbors`,
`maximumWindowLength`, `maximumReplicates`, optional `nearestZoneSearch`, and the device's
`maxBufferSize` and `maxStorageBufferBindingSize` in bytes. This covers numeric admission;
actual views still undergo graph ownership, format, usage and alias checks.

Inference: `maximumReplicates` Monte Carlo replicates of independent Poisson counts from the baseline, each searched like the data.
The p-value is `(1 + #{replicate maxima >= observed}) / (P + 1)`.
Both scans report `pValueConstruction: 'upper-tail-count'` in inference metadata, including the
conditional scan with `rateDirection: 'both'`: windows of both directions compete in one maximum.
At 19 draws the minimum attainable p-value is `.05`, not the `.10` of a doubled minimum tail. Secondary clusters and repeated-analysis adjustment are not
implemented; convert with `getGPUScanRecurrenceInterval(pValue, analysisInterval)`. Checked against an exhaustive binary64 subset enumeration and the same Philox streams; the `scanstatistics` R package (GPL, R-only) was not run, so its fixture is pending.

Replicate sampling uses a product-of-uniforms Poisson sampler below mean 30 and PTRS rejection
sampling from mean 30 to below `1e7` (at most 64 proposals, then the rounded mean). At means
`>= 1e7`, it uses a rounded normal approximation because float32 cannot resolve the PTRS
acceptance calculation. The finite proposal limit and large-mean route are approximations.
`scan-sampling.spec.ts` checks sums, variance and CDF counts against independent SciPy Poisson
bounds at means 0, 0.5, 2, 29, 30, 100, `1e6` and `1e7`; these checks do not certify every mean.

Inputs: `cases` and `baseline`, `zones * timeBuckets` rows (`zone * timeBuckets + bucket`). The baseline is the caller's: for example a moving average of earlier buckets. Parameters (seed, replicate count,
per-frame window limit) are packed by `getGPUExpectationBasedScanParameterValues`. Limits on both routes: at most 64 neighbors per center, at most 32 buckets per window,
and 99,999 replicates total. The count table holds at most `2^25` words per batch,
`(batchReplicates + 1) * zones * maximumWindowLength`, with at most 4,096 replicates per batch
and 1,024 batches; each binding must also fit the device's byte limit.
Per-frame window-length requests above the uint32 range saturate before packing and are then
clamped to the compile-time capacity; they do not wrap to smaller windows.
Explicit per-frame window lengths must be finite and positive; they round down with a minimum of one.
The parameter packer captures each supplied setting once before validation and conversion, including
accessor-backed settings. The conditional scan packer follows the same capture rule.
The exhaustive test reference refuses neighborhoods above 16 zones because its subset count grows
exponentially. Larger GPU neighborhoods use independent closed-form cases, including a 31-zone
fixture whose only elevated singleton scores `2 ln 2 - 1`. The production limit remains 64 neighbors.
Outputs: `clusterZones`, `clusterStatistics` (`GPU_EXPECTATION_BASED_SCAN_CLUSTER`), `statistics` (maxima per replicate, observed first),
`summary` (`GPU_EXPECTATION_BASED_SCAN_SUMMARY`) and optional `zoneStatistics`.

{/* lens:GPUExpectationBasedSpaceTimeScan */}

## Conditional scan input diagnostics

For the Poisson model of `GPUSpatialScanStatistic`, an included cell with positive `cases`
and zero `baseline` is incompatible with the conditional null. Negative or non-finite baselines
are also invalid. A positive exposure whose binary exponent is more than 126 below the
largest positive exposure cannot survive this route's normal-float32 scaling: the frame is
reported as numerically unavailable, rather than certifying a no-cluster result. Optional
`output.status.invalidCount` counts incompatible or numerically lost cells; a count/population
overflow reports at least one. Invalid frames publish no clusters and fill inference statistics
with NaN even when the status destination is omitted. The summary's replicate-count word is
zero for an invalid frame because its batch dispatches are suppressed; valid frames report the
number of replicates actually evaluated. Zero cases at zero exposure are admissible.
The count and flags are reset each encoding. Space-time permutation margins use ordered parallel
integer prefixes; observed counts remain separate from replicate slots. Center overflow flags
are reduced afresh for the summary on every encoding.

The conditional scan's parameter packer accepts only `windowShape: 'circle' | 'nearest'` and
`rateDirection: 'high' | 'low' | 'both'`. Unknown strings are rejected before packing.

## Prospective mode of `GPUSpatialScanStatistic`

`temporalMode: 'prospective'` keeps only windows that end at the last bucket, in the observed data and in every replicate. The default
`'retrospective'` is unchanged. No adjustment for earlier analyses is applied: report recurrence intervals with
`getGPUScanRecurrenceInterval(pValue, analysisInterval)`, in the unit of `analysisInterval`, and treat alarms from repeated analyses
accordingly (Correa et al. 2015; Kulldorff and Kleinman 2015).
