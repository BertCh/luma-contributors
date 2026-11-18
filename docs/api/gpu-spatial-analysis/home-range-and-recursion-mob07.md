# Home ranges and recursion

Brownian bridge utilization distributions with volume isopleths, and recursion analysis (revisits,
residence time, first passage time) of trajectories. Planar only. Out of scope: continuous-time
movement models and autocorrelated kernel density estimation (ctmm).

None of this was checked against the R packages `move` or `recurse`: R is not installed here, the
licence of `move` is not verified, and both are used only as definitions. Tests use binary64 CPU
references written from the formulas below, and cases with hand-computable answers.

## `GPUBrownianBridgeUtilization`

The utilization distribution of Horne et al. (Ecology 2007). Between fixes `a` and `b`, `T` apart
in time, the position `alpha T` after `a` is bivariate normal with mean `a + alpha (b - a)` and
variance `T alpha (1 - alpha) sigma^2 + (1 - alpha)^2 delta_a^2 + alpha^2 delta_b^2`, with `sigma^2`
the Brownian motion variance (length squared per time unit) and `delta` the location errors. The
density of a cell is the duration-weighted average, over all segments, of the time average of that
normal density at the cell centre. The time average over a segment is the midpoint rule with
`subStepCount` steps (default 15).

The dynamic Brownian bridge (Kranstauber et al. 2012) differs only in giving a variance per
segment: `motionVariances` has one row per fix, the variance of the segment that starts there. The
CPU helpers `estimateBrownianMotionVariance` and `estimateWindowedBrownianMotionVariance` produce
it (binary64, a leave-one-out maximum likelihood in the manner of Horne et al.). The likelihood
predicts every second fix from its two neighbours with variance
`T alpha (1 - alpha) sigma^2 + (1 - alpha)^2 delta_a^2 + alpha^2 delta_b^2 + delta_i^2`: the last
term is the location error of the predicted fix itself, which the three-term bridge variance of
`BBMM` and `move` leaves out. With a location error above 0 the estimate is therefore lower than
theirs (15.6 against 28.5 on a simulated walk of true variance 15 and location error 20); with no
location error the two are the same. The windowed estimate is a sliding-window variance; the
behavioural change point step of the dynamic Brownian bridge is not implemented.

Both estimators include the exact lower bound of zero motion variance. For diagnostics,
`estimateBrownianMotionVarianceWithStatus` returns `{motionVariance, status: {lowerBoundReached}}`;
the flag is true when the estimate is zero. `estimateWindowedBrownianMotionVarianceWithStatus`
returns `{motionVariances, segmentStatus}`, with the same diagnostic for each segment's window.
Windows with fewer than three fixes, unusable terms (including non-finite coordinates, times or
location errors), or no finite likelihood candidate return NaN and `lowerBoundReached: false`.
Boundary comparisons use stable likelihood differences, including when coordinates and errors
are rescaled or the optimum is positive but very near zero. The likelihood difference
keeps a small-change series near zero and uses log/reciprocal-variance differences for large ratios,
so heterogeneous location errors do not erase the boundary likelihood. The scan includes the
individual terms' error-transition and positive-optimum scales and refines each detected local
peak; it does not assume the mean residual scale brackets the best peak. Zero-residual terms skip
the reciprocal penalty, including for subnormal positive error variances. The estimator uses the
closed-form mean residual scale when all location errors are zero. For nonzero errors,
overlapping logarithmic scan intervals are merged before scoring; the interval from zero to the
first positive sample is refined even when that sample has a lower likelihood than zero. Thus a
positive optimum below the scan spacing is not labelled as a boundary optimum. The numeric-only helper
does not retain a per-segment diagnostic array. The original
helpers still return a number or `Float64Array`. The CPU and GPU utilization operations both
accept zero motion variance; positive location error still supplies the bridge's uncertainty.

The windowed helpers return one variance per segment (`fixCount - 1`), while utilization requires
one row per fix. Allocate the per-fix column and copy each track's segment variances at its first
fix row, leaving that track's unused final row zero. For one three-fix track:

```ts
import {estimateWindowedBrownianMotionVariance} from '@luma-contributors/analysis/gpu-spatial-analysis';

const track = {x: [0, 1, 0], y: [0, 0, 0], times: [0, 1, 2]};
const segmentVariances = estimateWindowedBrownianMotionVariance(track, {locationError: 1});
const motionVariances = new Float32Array(track.times.length);
motionVariances.set(segmentVariances); // [0, 0, 0], including the unused final row
```

For packed tracks, use `motionVariances.set(segmentVariances, trackOffsets[trackRow])` for each
separately estimated track. Do not concatenate the segment columns without the final rows.

| Prop | Meaning |
| --- | --- |
| `positions`, `timestamps`, `trackOffsets` | The trajectory port |
| `motionVariances`, `locationErrors` | One float32 per fix |
| `gridSize`, `parameters` | Grid size (compile-time) and extent from `getGPUBrownianBridgeUtilizationParameterValues` |
| `output.density` | Density per unit area at each cell centre, row 0 at the smallest y |

Segments farther than 7 standard deviations from a cell are skipped for that cell. The first and
last segments are used (`move` omits them by default). The tier is reference: every cell loops over
all fixes, so the fix count is limited to 16384.

{/* lens:GPUBrownianBridgeUtilization */}

{/* parameters: GPUBrownianBridgeUtilization generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPUBrownianBridgeUtilization` parameters

- Packer: `getGPUBrownianBridgeUtilizationParameterValues(settings)`, writes `GPU_BROWNIAN_BRIDGE_UTILIZATION_PARAMETER_LENGTH` (4) `float32` values.

Settings of `getGPUBrownianBridgeUtilizationParameterValues`:

| Field | Type | Description |
|---|---|---|
| `extent` | `readonly [number, number, number, number]` | Grid extent `[minX, minY, maxX, maxY]`; the cells are `gridSize` equal rectangles, row 0 at the smallest y. |

{/* /parameters */}

## `GPUUtilizationVolumeIsopleths`

Sorts the cells by density and takes a cumulative sum, giving each cell the share of total mass
held by the cells at least as dense as it, its own mass included (`output.volume`), and a mask
`output.inside` of the smallest set of densest cells whose mass reaches `level`. A cell of zero
density has volume 1, so `volume <= level` never selects an empty cell. Cells of equal density are
ordered by cell index. Sums are float32 in a fixed order. The volume raster of `move`
(`getVolumeUD`) serves the same purpose; whether it counts a cell's own mass was not checked, so
values can differ from it by one cell's share.

{/* lens:GPUUtilizationVolumeIsopleths */}

{/* parameters: GPUUtilizationVolumeIsopleths generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPUUtilizationVolumeIsopleths` parameters

- Packer: `getGPUUtilizationVolumeIsoplethsParameterValues(settings)`, writes `GPU_UTILIZATION_VOLUME_ISOPLETHS_PARAMETER_LENGTH` (4) `float32` values.

Settings of `getGPUUtilizationVolumeIsoplethsParameterValues`:

| Field | Type | Description |
|---|---|---|
| `level` | `number` | Isopleth level in `(0, 1]`, for example 0.95 for the 95% home range. |

{/* /parameters */}

## `GPUTrajectoryRecursions`

For a circle of radius `r` around each fix (several radii in one pass), after Bracis et al.
(`recurse::getRecursions`, Ecography 2018). The path between fixes is the straight segment at
constant speed, so circle crossings are exact and times are linearly interpolated.

| Output | Definition |
| --- | --- |
| `revisits` | Entries into the closed disc, not counting the visit that contains the fix |
| `residenceTimes` | Total time inside the disc over the whole track |
| `firstPassageTimes` | Time from the fix until the track first leaves the disc; if the track ends inside, the time to the end |
| `firstPassageCensored` | 1 where the previous value is the time to the end |

`firstPassageTimes` is the forward leg only: from the fix at the centre to the first exit. The
first passage time of Fauchald and Tveraa (2003) and of `adehabitatLT::fpt` is the time to cross the
whole circle, backward leg plus forward leg; on a straight track at constant speed `v` that is
`2 r / v` against `r / v` here. Add the value of the reversed track to get the two-sided time.

Only the fix's own track is examined, and each fix examines at most `maximumTrackSegments` segments
(default 4096); a longer track is examined in a window around the fix and sets
`status.candidateOverflow`. Not provided: polygon locations, a minimum time outside to end a visit,
and interval residence time.

{/* lens:GPUTrajectoryRecursions */}
