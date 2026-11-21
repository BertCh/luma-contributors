# Movement semantics

Import from `@luma-contributors/analysis/gpu-spatial-analysis`.

Staypoints, locations, triplegs and trips, mobility indicators on staypoints, and trajectory
cleaning. Everything here reads the shared [`GPUTrajectoryGeometryPort`](./concepts.md) layout:
`positions` (`float32x2`), `trackOffsets` and `timestamps` (`float32` relative or `uint32x2` Int64
words, with an optional [`temporalContext`](./temporal-context.md)). Durations and the shared gap
rule, `maximumGap`, are in the timestamps' unit: a step is a gap only if it is strictly longer than
the limit, and 0 or absent means no limit.

The families follow, in order, trackintel (MIT), scikit-mobility 1.3.1 (BSD) and MovingPandas 0.23
(BSD-3); fixtures from those packages check them. Parameters are per-frame packed float32
buffers unless a section says otherwise.

Out of scope (as the 2026-11 spatiotemporal review concluded): exact
flocks and convoys, TRACLUS, T-LoCoH, and continuous-time movement model fitting (CTMM, OU/OUF).

## Staypoints, locations and triplegs

The contributors compose, each reading the columns the previous one writes:

```ts
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {getGPUTrajectoryStaypointsParameterValues, GPUTrajectoryStaypoints} from '@luma-contributors/analysis/gpu-spatial-analysis';

export function addMovementSemanticsExample0(graph: GPUCommandGraph, inputs: {
  positions: NonNullable<ConstructorParameters<typeof GPUTrajectoryStaypoints>[0]['positions']>;
  timestamps: NonNullable<ConstructorParameters<typeof GPUTrajectoryStaypoints>[0]['timestamps']>;
  trackOffsets: NonNullable<ConstructorParameters<typeof GPUTrajectoryStaypoints>[0]['trackOffsets']>;
  parameterBuffer: GPUParameterBuffer<'float32'>;
  ids: NonNullable<NonNullable<ConstructorParameters<typeof GPUTrajectoryStaypoints>[0]['output']>['ids']>;
  count: NonNullable<NonNullable<NonNullable<ConstructorParameters<typeof GPUTrajectoryStaypoints>[0]['output']>['status']>['count']>;
  overflow: NonNullable<NonNullable<NonNullable<ConstructorParameters<typeof GPUTrajectoryStaypoints>[0]['output']>['status']>['overflow']>;
  candidateOverflow: NonNullable<NonNullable<NonNullable<ConstructorParameters<typeof GPUTrajectoryStaypoints>[0]['output']>['status']>['candidateOverflow']>;
  startRows: NonNullable<NonNullable<ConstructorParameters<typeof GPUTrajectoryStaypoints>[0]['output']>['startRows']>;
  finishRows: NonNullable<NonNullable<ConstructorParameters<typeof GPUTrajectoryStaypoints>[0]['output']>['finishRows']>;
  centroids: NonNullable<NonNullable<ConstructorParameters<typeof GPUTrajectoryStaypoints>[0]['output']>['centroids']>;
  durations: NonNullable<NonNullable<ConstructorParameters<typeof GPUTrajectoryStaypoints>[0]['output']>['durations']>;
  staypointIds: NonNullable<NonNullable<ConstructorParameters<typeof GPUTrajectoryStaypoints>[0]['output']>['staypointIds']>;
}) {
  const {positions, timestamps, trackOffsets, parameterBuffer, ids, count, overflow, candidateOverflow, startRows, finishRows, centroids, durations, staypointIds} = inputs;
  parameterBuffer.write(getGPUTrajectoryStaypointsParameterValues({
      distanceThreshold: 100,
      timeThreshold: 300_000,
      maximumGap: 900_000,
      includeLast: true
    }));
  graph.add(
    new GPUTrajectoryStaypoints({
      spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'},
      positions,
      timestamps,
      trackOffsets,
      parameters: parameterBuffer.importToGraph(graph),
      output: {ids, status: {count, overflow, candidateOverflow}, startRows, finishRows, centroids, durations, staypointIds}
    })
  );
}
```

For three coincident fixes at timestamps `[0, 300000, 600000]` milliseconds, these settings
emit one stay of `600000 ms`. Without `includeLast: true`, the final open window is not emitted.

| Definition | Spatial test and equality | Time and end rows | Final window |
| --- | --- | --- | --- |
| MovingPandas maximum-diameter stop | Minimum rotated rectangle diagonal is strictly below `max_diameter` (metres for geographic coordinates, CRS units for planar); equality breaks the stop; no exact counterpart here | Duration is at least `min_duration` (a timedelta); start/end timestamps select included observations | Detector evaluates the final candidate |
| `GPUTrajectoryMetrics` stop | Consecutive steps with zero distance, or distance strictly below `stopSpeedThreshold * deltaTime`, with nonnegative delta time | `stopMinimumDuration` and durations use timestamp units; start row is the fix before the first slow step, end row is the last slow step's destination (inclusive) | Final slow run is evaluated |
| `GPUTrajectoryStaypoints` | Distance from window anchor is strictly below `distanceThreshold`; equality closes the window | Duration at least `timeThreshold`, in timestamp units; `endRows` exclusive, `finishRows` selects finish time (the closing fix on a distance break) | Omitted unless `includeLast: true` |

### `GPUTrajectoryStaypoints`

Detects staypoints with trackintel's `sliding` rule (Li et al. 2008) in a bounded number of passes:
every row searches forward independently for its first break (a gap, or a fix at or beyond
`distanceThreshold`), at most `maximumSearchRows` rows ahead (default 4,096, at most 16,777,216 per anchor). Pointer doubling in
`ceil(log2(rowCount))` fixed passes then keeps the rows reachable from each track's first row. That
reproduces the sequential chain exactly as long as no search hits the cap. Supply a one-row
uint32 `output.status.candidateOverflow` to detect a cut search. Every staypoint on a track
whose chain was cut has NaN `durations` and `centroids`.

`output.status.overflow` reports final staypoint-table capacity only. A capped search can set
`candidateOverflow` while all retained staypoints fit and final overflow stays zero. Both words
are rewritten each submission, so a frame whose searches fit the bound clears candidate loss.

`status.count` is the retained staypoint count, `requiredCount` is the count before final
capacity clamping, and `drawInstanceCount` equals the retained count. Two independent stationary
tracks each meeting the duration threshold require two stays: capacity one gives
`[count,requiredCount,overflow]=[1,2,1]`, while capacity two gives `[2,2,0]`.

Compact candidate-status words on trajectory metrics stops and trajectory segments are outputs,
too. They must be separate from input `trackOffsets` and the other output words; publication
must never alter the trajectory used by the next submission.

Unlike a per-step speed test, one parked stop stays one stop under GPS jitter: the window restarts
only when a fix leaves `distanceThreshold` of its anchor. The cost is `N * min(maximumSearchRows,
L)` distance tests for stays of `L` rows, so 1 Hz data parked for a long time needs a cap of at least
that many rows, or thinning first.

| Prop | Meaning |
| --- | --- |
| `parameters` | Packed `getGPUTrajectoryStaypointsParameterValues`: distance, time, gap, include-last |
| `maximumSearchRows` | Forward search cap per row, compile-time, default 4096 |
| `output.ids` | Track of each staypoint, ordered by track then time |
| `output.startRows`, `output.endRows`, `output.finishRows` | First row, exclusive end row and the row whose time is `finished_at` |
| `output.centroids` | Mean of the staypoint's distinct coordinates (trackintel `union_all`) |
| `output.durations` | `finished_at - started_at` |
| `output.staypointIds` | Per-row `staypoint_id`, `0xffffffff` outside staypoints |

Distinct votes use raw coordinates in planar contexts. In great-circle contexts, voting and
anchor-relative center offsets use one canonical geographic representative: longitude wraps to
`[-180,180)`, so `180`, `-180` and `540` identify the same point, and longitude is zero for voting
at an exact north or south pole. Original coordinates and row IDs are preserved. Thus the three
equatorial tracks `[180,180,177]`, `[180,-180,177]` and `[180,540,177]` each center at `178.5°`.
Longitude is reduced modulo 360 before testing interval endpoints, preserving equivalent
spellings immediately beside the antimeridian, including `180-2^-16` and `-180-2^-16` degrees.
Nearest-even half-turn offsets are applied after canonicalization: `[0,180]` and `[0,-180]`
both center at `-90°`. A pole-only center has longitude zero.
The CPU staypoint output also corrects the endpoint after float32 rounding, so a rounded `180°`
is published as `-180°`, matching CPU locations.

{/* lens:GPUTrajectoryStaypoints */}

{/* parameters: GPUTrajectoryStaypoints generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUTrajectoryStaypoints` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPUTrajectoryStaypoints`.

- Packer: `getGPUTrajectoryStaypointsParameterValues(settings)`, writes `GPU_TRAJECTORY_STAYPOINTS_PARAMETER_LENGTH` (4) `float32` values.

Settings of `getGPUTrajectoryStaypointsParameterValues`:

| Field | Type | Description |
|---|---|---|
| `spatialContext` (optional) | `GPUSpatialContext` | Spatial context of the positions, for the short-distance warning only. |
| `distanceThreshold` | `number` | Distance at or beyond which a fix leaves the current staypoint (trackintel `dist_threshold`), in position units, or the context length unit for great-circle. Finite and positive. |
| `timeThreshold` | `number` | Smallest staypoint duration (`finished - started`, trackintel `time_threshold`) in the timestamps' unit. Finite and non-negative. |
| `maximumGap` (optional) | `number` | Shared gap rule (`GPUTrajectoryGeometryPort`): a step is a gap only when strictly longer than this limit, in the timestamps' unit; 0 or absent means no limit. A gap restarts the search at the fix after it without emitting a staypoint (trackintel `gap_threshold`). |
| `includeLast` (optional) | `boolean` | Emit a staypoint for the window still open at the end of a track (trackintel `include_last`). |

{/* /parameters */}

### `GPUTrajectoryLocations`

Staypoint locations per track with trackintel's `generate_locations` (`dbscan`, `num_samples=1`,
`agg_level='user'`): the connected components of staypoints of one track within `epsilon`
(inclusive), labelled by bounded min-label hooking with path compression (`maximumIterations`
rounds, `status.converged`). Locations are numbered by first visit within a track, tracks in order.
When the labels do not settle within `maximumIterations` (default 32, at most 256; a chain in
random row order can need many more rounds than its length suggests), no partial labels are
published: every `locationIds` row is `0xffffffff`, `status.count` is 0, GPU `centers` are not
written and the required one-row uint32 `output.status.iterationLimitReached` is 1. Raise
`maximumIterations` to allow more rounds.
Optional `status.iterationCount` reports the hook/compress rounds actually executed, equal to
`maximumIterations`, even if the labels stabilize earlier; the final convergence check is not a
round. Centers average distinct coordinates within each location, using the same canonical
geographic voting and offset rule as staypoint centroids (raw equality in planar contexts).

The CPU locations operation honors an explicit `maximumIterations` with the same bounded
hook/compress rounds and stopping flags. Without an explicit budget or requested iteration count
its component calculation remains exact. Bounded CPU calls also suppress unsettled labels:
IDs are `0xffffffff`, location count is 0, and the returned center capacity is zero-filled.
`maximumIterations` must be an integer from 0 through 256 in every form. Zero executes no
hook/compress rounds, then checks the initial labels; they are published only if already settled.
Only an omitted or `undefined` budget selects the default; an explicit `null` is rejected.
The pure CPU function captures the budget once on entry, so an accessor cannot change the
validated round limit or the reported iteration count during execution.
Both graph operations accept optional `output.status.iterationCount`. A CPU operation requesting
it runs the explicit budget, or 32 rounds by default, so it reports work actually executed.
The pure CPU function reports zero hook/compress rounds on its exact union-find path.
Bounded CPU neighbor scans and the final convergence check stay within each contiguous track
range; one round costs `O(sum m²)` for per-track staypoint counts `m`.
The GPU finds sorted track ranges with one lower/upper binary-search pair per input row.
Per-label atomic minima select the first member of each location, then published stable
`GPUCompaction` assigns IDs in first-visit order. Neither preparation nor relabeling scans the
entire staypoint table in one invocation. IDs and centers are recomputed on every complete
submission, so no location borrows another track's ID.
Construction snapshots the output and status maps, retaining the original view identities;
later caller edits to those maps do not redirect writes. Graph operations likewise snapshot
their graph value names and iteration settings.
Dataset-level locations and `num_samples > 1` are `GPUDBSCAN` jobs; the convex-hull
`extent` is not produced.

`GPUTrajectoryLocationsOperation` / `CPUTrajectoryLocationsOperation` accept graph value names
for stay `centroids`, `tracks`, `count` and `parameters`, with an explicit table `capacity`.
Their outputs are location IDs, optional centers, `output.status.count` and the required
`output.status.iterationLimitReached`. The `iterationCount` and `converged` outputs are optional. Compile the GPU form with
`GPU_CONTRIBUTOR_OPERATION_LOWERING`.

{/* lens:GPUTrajectoryLocations */}

{/* parameters: GPUTrajectoryLocations generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUTrajectoryLocations` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPUTrajectoryLocations`.

- Packer: `getGPUTrajectoryLocationsParameterValues(settings)`, writes `GPU_TRAJECTORY_LOCATIONS_PARAMETER_LENGTH` (4) `float32` values.

Settings of `getGPUTrajectoryLocationsParameterValues`:

| Field | Type | Description |
|---|---|---|
| `spatialContext` (optional) | `GPUSpatialContext` | Spatial context of the positions, for the short-distance warning only. |
| `epsilon` | `number` | Clustering radius (trackintel `epsilon`, inclusive), in position or context length units. |

{/* /parameters */}

### `GPUTrajectorySegments`

Triplegs (`between_staypoints`) and trips (`generate_trips`) from the per-row staypoint IDs and the
staypoint table. A tripleg is a maximal run of rows outside every staypoint, broken at a gap, with
at least two rows that are not all the same coordinate. A trip aggregates triplegs and non-activity
staypoints between activities (staypoints strictly longer than `activityMinimumDuration`), breaks at
a gap between events, and starts at the last activity of an activity run. Origin and destination
staypoints are the activities directly around a trip, unknown across a gap or at a track end, as
`origin_staypoint_id` and `destination_staypoint_id` are in trackintel. `triplegTripIds` is the
`trip_id` of each tripleg. The `overlap_staypoints` method is not implemented.

Supply available staypoint `status.overflow` and `status.candidateOverflow` beside `status.count`.
Their loss is propagated into `triplegs.status.candidateOverflow` and `trips.status.candidateOverflow`.
Trips consume only the stored tripleg prefix, so tripleg capacity loss also sets trip candidate
overflow. Each table's `status.overflow` still means only that table's final output capacity was
exceeded. A trip count/required count derived from an incomplete input describes the partial result;
the candidate flag must be checked before treating it as a complete trip table.
Trip construction requires a caller-visible `trips.status.candidateOverflow` destination. Supplying
either staypoint loss input also requires `triplegs.status.candidateOverflow`. Both operation forms
and the direct contributor reject omitted destinations at construction; private scratch never
substitutes for a public loss diagnostic.

Pass both tripleg `overflow` and `candidateOverflow` to modal split and tracking quality. If either
supplied flag is nonzero, modal split suppresses every aggregate to NaN and every predicted mode to
`0xffffffff`; tracking quality suppresses every track's value to NaN. The flags describe whole-table
loss, so these consumers cannot identify unaffected tracks. Omitting both flags asserts that the
input table is complete. Complete empty tracks still produce zero. Website views hide incomplete
modal bars/classes and label tracking quality unavailable; Kepler exports unavailable numeric values
as null. Never interpret NaN or the missing-mode sentinel as a count or a speed class.

The thinning and staypoint CPU twins honor an explicit `maximumSearchRows`, including the GPU's
anchor jumps/window splits and candidate-overflow flag. Omitted CPU search budgets remain uncapped.
Thinning count and candidate-overflow words have separate writers, so adjacent non-overlapping
scalar subviews of one status allocation are supported.

{/* lens:GPUTrajectorySegments */}

{/* parameters: GPUTrajectorySegments generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUTrajectorySegments` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPUTrajectorySegments`.

- Packer: `getGPUTrajectorySegmentsParameterValues(settings)`, writes `GPU_TRAJECTORY_SEGMENTS_PARAMETER_LENGTH` (4) `float32` values.

Settings of `getGPUTrajectorySegmentsParameterValues`:

| Field | Type | Description |
|---|---|---|
| `maximumGap` (optional) | `number` | Shared gap rule (`GPUTrajectoryGeometryPort`): a step strictly longer than this limit breaks a tripleg and a trip (trackintel `gap_threshold`); 0 or absent means no limit. |
| `activityMinimumDuration` | `number` | A staypoint is an activity when its duration is strictly longer than this, in the timestamps' unit (trackintel `create_activity_flag`). Finite and non-negative. |

{/* /parameters */}

## Mobility indicators

### `GPUMobilityIndicators`

Per-track indicators on the staypoint table (and, for tracking quality, the tripleg table),
following scikit-mobility 1.3.1 `individual` measures and trackintel:

| Output | Definition |
| --- | --- |
| `radiusOfGyration` | RMS Euclidean distance from the coordinate mean in planar contexts; great-circle distance from the spherical mean in great-circle contexts by default. Explicit `gyrationCenter: 'coordinate-mean'` retains arithmetic centering. |
| `jumpLengths` | Distance between consecutive staypoints (0 for a track's first) |
| `waitingTimes` | Difference between consecutive staypoint start times |
| `randomEntropy` | `log2(distinct locations)` |
| `uncorrelatedEntropy` | Shannon entropy in bits of location shares |
| `realEntropy` | Lempel-Ziv estimator of the location sequence, `n * log2(n) / sum(lambda)` as in scikit-mobility |
| `homeLocations`, `workLocations` | Locations with the longest and second longest summed duration (trackintel FREQ, no pre-filter) |
| `trackingQuality` | Sum of positive durations over staypoints and triplegs, divided by the extent from first start to last finish (trackintel `temporal_tracking_quality`, granularity `all`) |

After a linear pass over staypoints, location statistics are `O(m^2)` per track and real entropy
is `O(m^3)` in the worst case for `m` usable location keys. The GPU
`maximumEntropySequenceRows` defaults to 1,024 and accepts at most 16,384 keys per track. An
over-cap track gets NaN `realEntropy` and sets `status.candidateOverflow`. The CPU function applies the cap
only when its `maximumEntropySequenceRows` option is given, and the CPU operation passes the prop
through, so an omitted prop leaves the CPU twin uncapped.
Pass available staypoint `status.overflow` and `status.candidateOverflow` alongside `status.count`,
including when the table comes directly from `GPUTrajectoryStaypoints`. Either nonzero input
suppresses every numeric indicator to NaN and both location outputs to `0xffffffff`, across all
tracks and all capacity rows. The optional output `status.candidateOverflow` is the OR of supplied
staypoint loss (`overflow` or `candidateOverflow`), entropy sequence loss, and supplied
`locationIterationLimitReached`; it resets on each frame, including when real entropy is not
requested. Suppression remains observable when that optional status destination is omitted.
The CPU function takes the same loss arrays beside `staypoints.count`. Omitting both loss inputs
asserts completeness. Website views hide these indicators and label them unavailable for incomplete
staypoints; Kepler exports missing numeric and location values as null.

Bind Locations' `status.iterationLimitReached` as
`staypoints.status.locationIterationLimitReached`. A nonzero value applies to the whole table.
Every active key is excluded, every track's three entropies are NaN, and home/work IDs are
`0xffffffff`. The optional `status.candidateOverflow` is 1. Radius of gyration, jump lengths,
waiting times and tracking quality still use the full staypoint sequence. The pure CPU
function takes the flag as `staypoints.locationIterationLimitReached`, beside `staypoints.count`.

With a clear or omitted flag, sentinel IDs (`0xffffffff`) are excluded from all location
indicators. Omitting `staypoints.locationIds` makes every active key missing. The retained keys
keep their visit order for real entropy, and only their durations contribute to home/work.
A track with stays but zero usable keys has NaN entropies and sentinel home/work IDs. A track
with no stays has zero entropies and sentinel home/work IDs unless the global loss flag is set.
Supply the optional one-row uint32 `output.status.excludedLocationCount` to read the number of
active rows with missing or unsettled keys, summed over all tracks and rewritten each frame.
The pure CPU result always returns this count.

{/* lens:GPUMobilityIndicators */}

### `GPUModalSplit`

Per-(track, mode) accumulation of tripleg count, distance or duration (trackintel
`calculate_modal_split`). Modes come from `modes`, or are predicted from average tripleg speed when
`modeSpeedUpperBoundsFloat32` is given (trackintel `simple-coarse`, left-closed classes, for example
`[15 / 3.6, 100 / 3.6, Infinity]` in metres per second). The last class has no upper bound, so a
speed at or above a finite last break stays in the last class. A tripleg with no speed (no time
between its first and last row, or a non-finite length) gets the missing key `0xffffffff` in
`output.predictedModes` and is left out of the split; the same key in a caller's `modes` is left
out too. Prediction rounds each step length to binary32, then accumulates those lengths in row
order with binary32 rounding after every addition. It compares `length < duration * encodedBound`
strictly, using the exact product of the encoded duration and bound, without a rounded speed
quotient or product. A requested `output.predictedModes` on a supplied-`modes` run receives
`0xffffffff` in every capacity row; the supplied IDs still determine the split.

{/* lens:GPUModalSplit */}

## Cleaning

### `GPUTrajectoryOutlierFlags`

Point-centred speed and acceleration flags. The speed bit flags a fix when both adjacent steps fail,
so the valid fix after a spike is not. The acceleration bit flags a fix when the step speed rises by
more than the limit into it and falls by more than the limit after the step out of it (the two
accelerations on either side of the two fast steps; the one between them is near zero), so a
sustained jump flags no row. At a track end the missing side is waived. A single pass; two
consecutive displaced fixes are not caught (the step between them is slow), unlike
`movingpandas.OutlierCleaner`, which compares each fix to the last kept one. `addTrackQualityRecipe` applies the same rule with `speedOutlierMode: 'point-centred'`;
its default remains `'step'`.

{/* lens:GPUTrajectoryOutlierFlags */}

{/* parameters: GPUTrajectoryOutlierFlags generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUTrajectoryOutlierFlags` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPUTrajectoryOutlierFlags`.

- Packer: `getGPUTrajectoryOutlierFlagsParameterValues(settings)`, writes `GPU_TRAJECTORY_OUTLIER_FLAGS_PARAMETER_LENGTH` (4) `float32` values.

Settings of `getGPUTrajectoryOutlierFlagsParameterValues`:

| Field | Type | Description |
|---|---|---|
| `maximumSpeed` | `number` | Speed limit, length unit per timestamp unit. Finite and non-negative; 0 disables the test. |
| `maximumAcceleration` (optional) | `number` | Acceleration limit, length per timestamp unit squared. 0 disables the test. |

{/* /parameters */}

### `GPUTrajectoryHampel`

Hampel filter per track (the `hampel` package used by PTRAIL): a value is an outlier when
`|value - median| > numberOfSigmas * 1.4826 * MAD` over a window centred on it; outliers become the
window median, and rows within half a window of a track end pass through unchanged. Compile-time
`windowSize`, at most 129.

{/* lens:GPUTrajectoryHampel */}

{/* parameters: GPUTrajectoryHampel generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUTrajectoryHampel` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPUTrajectoryHampel`.

- Packer: `getGPUTrajectoryHampelParameterValues(settings)`, writes `GPU_TRAJECTORY_HAMPEL_PARAMETER_LENGTH` (4) `float32` values.

Settings of `getGPUTrajectoryHampelParameterValues`:

| Field | Type | Description |
|---|---|---|
| `numberOfSigmas` | `number` | Number of scaled median absolute deviations above which a value is an outlier. |

{/* /parameters */}

### `GPUTrajectorySplit`

Splits tracks and rewrites offsets: `'gap'` (MovingPandas `ObservationGapSplitter`), `'stop'`
(maximal runs of rows outside every staypoint, the trackintel tripleg rule: the rows of a staypoint
are in no segment, unlike the MovingPandas `StopSplitter`, which keeps the boundary fixes of each
stop), and `'value-change'` (MovingPandas `ValueChangeSplitter`, where each segment but the last
repeats the first row of the next segment, so `rowIds` can reference a row twice). Segments need at
least two rows. The output is `segmentOffsets`, `segmentTracks` and `rowIds`
with optional copied `positions` and `timestamps`, so a split feeds the next contributor without
readback.

{/* lens:GPUTrajectorySplit */}

{/* parameters: GPUTrajectorySplit generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUTrajectorySplit` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPUTrajectorySplit`.

- Packer: `getGPUTrajectorySplitParameterValues(settings)`, writes `GPU_TRAJECTORY_SPLIT_PARAMETER_LENGTH` (4) `float32` values.

Settings of `getGPUTrajectorySplitParameterValues`:

| Field | Type | Description |
|---|---|---|
| `maximumGap` | `number` | Mode `'gap'`: the shared gap rule; a step strictly longer than this splits. 0 means no limit. |

{/* /parameters */}

### `GPUTrajectoryThinning`

Minimum-distance and minimum-time generalization (`MinDistanceGeneralizer`,
`MinTimeDeltaGeneralizer`): the first and last row of a track are kept, and each kept row is followed
by the first later row at or beyond the tolerance. As with staypoints, the sequential rule is a
bounded forward search plus pointer doubling. The GPU checks up to `maximumSearchRows` rows per
anchor, default 4,096 and at most 16,777,216. Supply the required one-row uint32
`output.status.candidateOverflow` to detect searches that hit the cap.

{/* lens:GPUTrajectoryThinning */}

{/* parameters: GPUTrajectoryThinning generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUTrajectoryThinning` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPUTrajectoryThinning`.

- Packer: `getGPUTrajectoryThinningParameterValues(settings)`, writes `GPU_TRAJECTORY_THINNING_PARAMETER_LENGTH` (4) `float32` values.

Settings of `getGPUTrajectoryThinningParameterValues`:

| Field | Type | Description |
|---|---|---|
| `tolerance` | `number` | Minimum distance (context length unit) or time (timestamps' unit) between consecutive kept rows. |

{/* /parameters */}

### `GPUTrajectoryTrackRelativeTimes`

Int64 word timestamps to float32 ticks after each track's first row, exact while the track spans
fewer than 2^24 ticks. Use its output as the float32 time column of the time-aware simplification
(`GPULineSimplification`, `metric: 'time-ratio'`) to run TD-TR on word timestamps, and of any other
float32 trajectory path.

{/* lens:GPUTrajectoryTrackRelativeTimes */}

### Operation forms: cleaning

`GPUTrajectorySplitOperation` / `CPUTrajectorySplitOperation` accept logical positions,
timestamps, track offsets, optional parameters, staypoint IDs and value columns according to
`mode`. Declare `rowCount`, `segmentCapacity` and `rowCapacity`. Outputs include segment offsets,
segment tracks, row IDs, optional compact positions/timestamps, and `status.count`,
`status.overflow`, and optional `status.requiredCount`. In stop mode, forward both available
staypoint loss flags via `staypointStatus` and provide `output.status.candidateOverflow` for
their combined value.

`GPUTrajectoryThinningOperation` / `CPUTrajectoryThinningOperation` accept logical positions,
timestamps, track offsets and parameters; declare row/track counts and output-row `capacity`.
Outputs include `sourceRows`, track offsets, optional
`sourceSelectionMask`, compact columns, required count/overflow/candidateOverflow status values and optional
requiredCount. The GPU's `maximumSearchRows` bounds sequential
search. Its CPU reference walk is uncapped only when `maximumSearchRows` is omitted; an
explicit cap can split the retained chain and sets `candidateOverflow`. Both operation pairs support relative float32 or
signed Int64 word timestamps through `timestampFormat` and matching temporal metadata. GPU
forms compile with `GPU_CONTRIBUTOR_OPERATION_LOWERING`.

### Operation forms: stays, triplegs and locations

`GPUTrajectoryStaypointsOperation` / `CPUTrajectoryStaypointsOperation` take logical positions,
timestamps, track offsets and parameters with row count and stay capacity. Outputs are stay
IDs, optional start/end/finish rows, centroids, durations, row-to-stay IDs and draw count;
count, overflow and candidateOverflow are required, and requiredCount is optional.

`GPUTrajectorySegmentsOperation` / `CPUTrajectorySegmentsOperation` take logical positions,
timestamps, track offsets, row-to-stay IDs and parameters, with optional counted stay table.
Declare row count, tripleg capacity and optional trip capacity. Triplegs and optional trips
each retain their own count/overflow and optional requiredCount/candidateOverflow views.
Forward supplied stay-table loss views when creating trips. These two trajectory operations
support relative float32 or signed Int64 word timestamps through `timestampFormat`.

`GPUTrajectoryLocationsOperation` / `CPUTrajectoryLocationsOperation` consume logical stay
centroids, tracks, count and parameters with explicit table `capacity`. Outputs are location
IDs, optional centers, location count, a required `iterationLimitReached` word and optional
`converged`.
Pass `staypointStatus.overflow` and `staypointStatus.candidateOverflow` from the source Staypoints
contributor. Independently supplied tables can omit `staypointStatus`. To distinguish source loss
from solver exhaustion,
request `output.status.complete` to determine whether the location result is whole. Source loss
suppresses all location labels/counts even when `converged=1` and `iterationLimitReached=0`:
convergence alone does not certify complete input. Keep this separate from the location-iteration
handoff to mobility indicators described above.
Explicit CPU iteration budgets use bounded hook/compress rounds too. An omitted CPU budget uses
complete connected components only when `output.status.iterationCount` is not requested; requesting
iterationCount selects bounded rounds with default budget 32. Use completeness and solver evidence together. GPU forms compile with
`GPU_CONTRIBUTOR_OPERATION_LOWERING`.

### Row domains and gathers

Thinning `sourceRows` maps the stored compact prefix to original fix rows. Optional
`sourceSelectionMask` selects rows in the original input-row domain. The mask is computed before compact capacity clamping, so it
can select more rows than fit in `[0, output.status.count)`. Gather positions and timestamps using
only the stored `sourceRows` prefix; Int64 timestamps require both words. Do not count selected
source rows as successfully stored output rows.
Split row IDs may repeat at value-change boundaries. Do not use an input-row mask as a compact
output mask. Construct the compact stop-table mask as `stopRow < stops.status.count`; use stop start/end
row gathers to recover times from the original fix column, not from stop-table row numbers.
Staypoint inputs support a per-fix `mask` and `missingFixPolicy: 'break'`; a zero mask or
nonfinite position breaks the stay chain, including at the first or final fix. Missing fixes do
not extend duration or add a centroid coordinate. `output.status.missingFixCount` counts the
missing fixes owned by live tracks, excluding unowned resident capacity rows; it is separate
from capacity loss and search truncation.
Missing/nonfinite fixes break usable movement evidence; a missing mode or location key is a
separate categorical absence. Forward producer loss flags before interpreting downstream counts:
each `overflow` and `candidateOverflow` input is a one-word `uint32` scalar, not a per-fix mask.
Locations accept them as `staypointStatus.overflow` / `staypointStatus.candidateOverflow`;
mobility accepts `staypoints.status` / `triplegs.status` flags (flattened beside each table's
`count` in the pure CPU input).

`gyrationCenter` defaults to `'spherical-mean'` for great-circle `radiusOfGyration` in every form:
it normalizes the mean unit direction. Planar contexts default to `'coordinate-mean'`.
Explicit `'coordinate-mean'` retains the scikit-mobility arithmetic longitude/latitude convention;
explicit `'spherical-mean'` requires a great-circle spatial context.
Both choices return RMS distance in context length units. If the mean direction has magnitude
at most `1e-6` per row (for example balanced antipodal fixes), the spherical center is undefined
and gyration is NaN. Two equally weighted equatorial stays at `[-1,1]` or `[179,-179]` both have
default gyration `R * pi/180`, half their two-degree equatorial jump length. Explicit arithmetic
centering on `[179,-179]` instead gives `R * 179*pi/180`.
The GPU spherical center uses compensated unit-vector sums so repeated directions do not drift
with track length or hide cancellation. Identical staypoint positions retain their stored center
and give exactly zero gyration in every form.

`modeSpeedUpperBoundsFloat32` gives exclusive class upper bounds in length per timestamp unit.
Bounds must remain strictly ascending after float32 encoding. The existing convention remains: internal bounds are exclusive,
and the last class is unbounded even with a finite last entry. Use `Infinity` for that last
entry to state the intent. Convert every cutoff to length per timestamp unit before passing it
(e.g. metres per millisecond for epoch-ms input).

Each encoded step length is added in trajectory row order with binary32 rounding at every step.
Classification uses the strict comparison `length < duration * encodedBound`, with the exact
product of the binary32 operands. For `length=5`, `duration=25` and
`encodedBound=f32(0.2)=0.20000000298023224`, the product is `5.000000074505806`, so the result is
bin zero. Dividing and rounding `5/25` first would lose that distinction. As a multistep control,
fifteen steps of `0.277777761220932` accumulated in binary32 give `4.166666030883789` in one time
unit, below the encoded 15 km/h bound `f32(15/3.6)=4.166666507720947` metres per second: bin zero.

Use `modes` for caller-assigned classes, or speed bounds for predicted classes; omit `modes`
when predicting from speed. `unknownModePolicy: 'last-bin'` retains the legacy treatment of
out-of-range IDs; `'exclude'` drops them. The missing key `0xffffffff` is excluded under both
policies. If `modes` is supplied, a requested `predictedModes` receives `0xffffffff` in every
capacity row, including the inactive tail; those sentinels describe the absence of a prediction.

<a id="stops-to-a-cell-period-cube"></a>

### `addStopValueCubeRecipe`

`addStopValueCubeRecipe` gathers signed Int64 timestamps through compact stop `startRows` or
`finishRows` (`time: 'start' | 'finish'`), then masks rows beyond `stops.status.count`. The original
timestamp unit and `temporalContext` must agree with the fixed periods. Cell keys join to one
fixed dictionary; `GPUKeyLookup.output.rightRows` feeds `GPUValueCube.cells`, and values of one
produce `GPUValueCube.output.counts` in `cell * periodCount + period` order. Dictionary order
must also match the spatial weights' row order. Allocate at least one row each for
`stops.centroids`, `stops.startRows` and `stops.finishRows`, even when `stops.status.count`
is zero: zero-capacity stop inputs are currently rejected during declaration. A zero live
count with those allocated rows represents no observed stops. Empty observed bins are zero. Forward stop
`overflow`/`candidateOverflow`, per-slice loss and weights loss: incomplete cubes suppress to
zero with status evidence, so a zero alone cannot establish an observed empty bin.

For pre-existing per-slice tables, align every slice to that dictionary, zero-fill absent keys,
and transpose slice-major rows into cell-major cube storage before downstream analysis.
