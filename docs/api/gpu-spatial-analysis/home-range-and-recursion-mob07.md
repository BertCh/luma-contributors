# Home ranges and recursion

The canonical task guide is [Home ranges and recursions](./home-ranges-and-recursions.md).
This historical URL and every contributor/parameter fragment below remain supported through
explicit anchors and links. Generated parameters and lens slots have one owner: the canonical guide.

Brownian bridge utilization distributions with volume isopleths, and recursion analysis (revisits,
residence time, first passage time) of trajectories. Planar only. Out of scope: continuous-time
movement models and autocorrelated kernel density estimation (ctmm).

None of this was checked against the R packages `move` or `recurse`: R is not installed here, the
licence of `move` is not verified, and both are used only as definitions. Tests use binary64 CPU
references written from the formulas below, and cases with hand-computable answers.

<a id="gpubrownianbridgeutilization"></a>

## Brownian bridge utilization

See the [canonical contributor contract](./home-ranges-and-recursions.md#gpubrownianbridgeutilization).

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
If at least one bridge has zero error variance and all zero-error bridges have zero residual,
the likelihood grows without bound as motion variance approaches zero. Both APIs return exactly
zero with `lowerBoundReached: true` for that limit, even when other bridges have positive error.
Any nonzero residual on a zero-error bridge rules out this singular boundary.
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

`density` is cell-centre density in inverse square coordinate units, not probability per cell.
The area-weighted raster sum approximates captured mass and can exceed one on coarse grids.
For two stationary fixes at `(0, 0)`, times `[0, 1]`, location errors `[1, 1]`, zero motion variance
and one midpoint substep, the midpoint variance is `1/2`; the centre density is `1 / π`.
One cell spanning `[-5, -5, 5, 5]` estimates mass `100 / π ≈ 31.8309886`, despite the analytic
Gaussian total mass being 1. Refine the grid to reduce spatial quadrature error; expand the extent
to capture more tails, then check stability of both mass and isopleths under further refinement.
If extent endpoints collapse after float32 packing, translate coordinates to a local planar origin
and pack the translated extent; increasing `gridSize` cannot recover collapsed coordinates.

`tailPolicy: 'truncate-at-seven-sigma'` skips segments outside a conservative seven-sigma
support box for that cell; `tailPolicy: 'full'` retains the full Gaussian tail. An explicit policy
applies to the direct GPU class and both GPU/CPU operation forms. Historical omitted defaults remain
GPU `'truncate-at-seven-sigma'` and CPU `'full'`; supply the same policy to compare backends.
The first and last segments are used (`move` omits them by default). The tier is reference: every cell loops over
all fixes, so the GPU fix count is limited to 16,384 total fixes across all tracks.
The CPU utilization reference's default retains all positive-variance midpoint Gaussian tails.
An explicit `subStepCount` applies to both forms (default 15), as does an explicit `tailPolicy`.
Neither form renormalizes bridge density to the chosen extent.

<a id="gpubrownianbridgeutilization-parameters"></a>

[Generated parameter reference](./home-ranges-and-recursions.md#gpubrownianbridgeutilization-parameters).

<a id="gpuutilizationvolumeisopleths"></a>

## Utilization volume isopleths

See the [canonical contributor contract](./home-ranges-and-recursions.md#gpuutilizationvolumeisopleths).

Normalizes over the supplied raster's positive finite density, not over the unobserved plane.
With equal-area cells this is the area-weighted raster normalization; it does not repair coarse-grid
quadrature or restore mass outside the extent. Negative/NaN/zero-density rows have volume 1 and
inside 0; an all-empty raster selects no cells. All grid rows are written; there is no compact tail.

Sorts the cells by density and takes a cumulative sum, giving each cell the share of total mass
held by the cells at least as dense as it, its own mass included (`output.volume`), and a mask
`output.inside` of the smallest set of densest cells whose mass reaches `level`. A cell of zero
density has volume 1, so `volume <= level` includes empty cells at `level: 1`.
Use `output.inside` for membership. For densities `[7, 3, 0]`, volumes are `[0.7, 1, 1]` and
`inside = [1, 1, 0]` at level 1; the volume threshold alone would select all three cells.
Cells of equal density are
ordered by cell index. Sums are float32 in a fixed order. The volume raster of `move`
(`getVolumeUD`) serves the same purpose; whether it counts a cell's own mass was not checked, so
values can differ from it by one cell's share.

<a id="gpuutilizationvolumeisopleths-parameters"></a>

[Generated parameter reference](./home-ranges-and-recursions.md#gpuutilizationvolumeisopleths-parameters).

<a id="gputrajectoryrecursions"></a>

## Trajectory recursions

See the [canonical contributor contract](./home-ranges-and-recursions.md#gputrajectoryrecursions).

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
(GPU default 4,096; CPU omission examines the whole track); a longer track is examined in a
window around the fix and sets
`status.candidateOverflow`. Not provided: polygon locations, a minimum time outside to end a visit,
and interval residence time.

`CPUTrajectoryRecursionsOperation` honours an explicit `maximumTrackSegments` with the same
centred windows and `candidateOverflow` meaning. `computeTrajectoryRecursionsOnCPU` accepts this
cap, a positive safe integer, in its optional second argument. When omitted, both CPU forms
examine whole tracks. An explicit window censors first passage at its last segment endpoint if
no exit was found.

Both graph operations capture their window budget, dimensions and timestamp format at
construction. Each declared caller field is read once, including inherited and non-enumerable
output/status getters. Spatial metadata and the temporal descriptor, including its axis, are
copied before validation and lowering. Changing the original props or context objects later
does not change an existing operation; construct a new operation to change these settings.
Input buffer contents remain per-frame. Recursion result rows are fix-major/radius-minor, with
exactly `rowCount * radiusCount` entries; every requested row is written, so there is no compact
tail. The radius predicate is `radius > 0`: nonpositive or NaN radii produce zero times/counts and
censored 1. A radius of positive infinity passes that predicate and includes the whole examined track/window;
it is not rejected as an invalid radius. Rows outside tracks also produce zero times/counts and
censored 1.
For bridge density, every grid cell is written; allocation capacity beyond these logical views
is outside the output contract.

## Operation forms

`GPUBrownianBridgeUtilizationOperation` and `CPUBrownianBridgeUtilizationOperation` accept the
same logical input names for planar positions, timestamps, track offsets, per-fix motion variances
and location errors, and the extent parameter vector. Set `timestampFormat` to `float32` (relative
time, the default) or `uint32x2` (signed Int64 words) with the matching `temporalContext`.
`rowCount`, `gridSize` and `subStepCount` are construction settings. The sole output is `density`:
there is no packed status output. For example, `output: {density: 'bridgeDensity'}` names the graph
value consumed by a later isopleth stage. The GPU reference route accepts at most 16,384 total
fixes across all tracks; this is an input-row limit, not a per-track or per-grid-cell allowance.

`GPUTrajectoryRecursionsOperation` and `CPUTrajectoryRecursionsOperation` take logical names for
planar positions, timestamps in either encoding, track offsets and radii. Supply `rowCount` and
`radiusCount`; each selected result has `rowCount * radiusCount` entries. Outputs are optional
`revisits`, `residenceTimes`, `firstPassageTimes`, `firstPassageCensored`, and the scalar
`output.status.candidateOverflow`. The GPU's `maximumTrackSegments` bounds the window examined
for each fix, and that flag marks an incomplete search. An explicit cap bounds both backends.
With the cap omitted, the CPU examines the whole track and the GPU defaults to 4,096 segments
per fix. A truncated window sets `candidateOverflow` and censors an unobserved first passage
at the window's last segment endpoint.

GPU operation stages compile with `GPU_CONTRIBUTOR_OPERATION_LOWERING`. These forms preserve
the contributor's first-passage and censoring definitions above.
