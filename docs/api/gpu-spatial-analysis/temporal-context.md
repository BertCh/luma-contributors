# Temporal context

A `GPUTemporalContext` declares what a time column means, the way
[`GPUSpatialContext`](./concepts.md) declares coordinates. It is a description only: nothing here
converts data. Import it from `@luma-contributors/analysis/analysis-core` or
`@luma-contributors/analysis/gpu-spatial-analysis`.

```ts
import type {GPUTemporalCalendarUnit} from '@luma-contributors/analysis/analysis-core';

type GPUTemporalContext = {
  unit: 'second' | 'millisecond' | 'microsecond' | 'nanosecond'; // ArrowAnalyticsTimeUnit
  epoch: 'unix' | bigint; // bigint: origin as signed Int64 ticks of `unit` after the Unix epoch
  encoding: 'float32-relative' | 'double-single' | 'int64-words';
  timeZone?: string; // as declared by the source; never parsed
  axis:
    | {kind: 'regular'; step: number | bigint | {calendarUnit: GPUTemporalCalendarUnit; interval?: number}}
    | {kind: 'irregular'};
};
```

`unit` is also the unit of durations given with the times, such as `maximumGap`.

## Encodings

| Encoding | Rows | Reads | Epoch |
| --- | --- | --- | --- |
| `int64-words` | One `uint32x2` Int64 `(low, high)` row, two's complement | Exact. Take differences on the words with `timeWordsDifferenceToF32`, then use f32 | `'unix'` or an origin |
| `double-single` | A `float32` high and a `float32` low part; the time is their sum | About 48 bits | `'unix'` or an origin |
| `float32-relative` | One `float32`, ticks after the origin | Exact to 2^24 ticks from the origin | An origin; `'unix'` is rejected because float32 cannot hold absolute time |

`0n` is the same instant as `'unix'`; declare it for times that really are near 1970 or have no
calendar meaning.

## Resolving and checking

`resolveGPUTemporalContext(id, context, support?)` validates a context and checks it against what an
operation accepts (`encodings`, `units`, `axisKinds`, `unixEpoch`). There is no default context,
because a unit or epoch must be declared. It rejects an unknown unit or encoding, an origin outside
Int64, `float32-relative` with `'unix'`, an empty `timeZone`, a non-positive or non-finite step, a
fractional step on `int64-words`, and a bad calendar step.

`getGPUTemporalContextCompatibility(source, target)` and
`assertCompatibleGPUTemporalContexts(id, source, target)` compare two columns on unit, epoch and
encoding. Time zone and axis are not compared: stored instants stay comparable across both. A
contributor that needs matching axes checks them itself.

## Arrow uploads

`makeGPUTimeVectorFromPreparedArrowColumn` returns the upload with a `temporalContext`, and
`getGPUTemporalContextFromPreparedArrowColumn(column)` gives the same context without a device:
`int64-words`, Unix epoch, the column's own unit and declared time zone, and an irregular axis.
A signed Int64 column prepared with a declared unit is taken to count from the Unix epoch; spread
another `epoch` over the result if it does not.

## What each family accepts today

These contributors take a context. The rest still document their conventions per prop, and the table
states them in terms of this contract. A context is always optional where the time view alone fixes the
encoding, and it never converts anything: it is checked against the bound views and the operation's
support at construction.

| Family | Contributors | Encoding | Unit and epoch |
| --- | --- | --- | --- |
| Calendar | `GPUCalendarBuckets` | `int64-words` | Any unit (`unit` or `temporalContext`), Unix epoch required |
| Period index and relative words | `GPUPeriodIndex`, `GPUTimeWordsRelative` | `int64-words` in; `GPUTimeWordsRelative` writes `float32-relative` and exposes the context of its output as `outputTemporalContext` | Any unit; the context is required |
| Time filters and flows | `GPUTimeWindowFilter`, `GPUFlowAggregation` | `int64-words`, `float32-relative`, or `double-single` through `timestampsLow` | Any unit, caller's epoch |
| Temporal reduction | `GPUTemporalReduction` | `int64-words` or `float32-relative` | Any unit, caller's epoch |
| Trajectory metrics, resample, playhead, zone events, encounters, closest approach | `GPUTrajectoryGeometryPort.temporalContext` | `int64-words` for `uint32x2` rows, `float32-relative` for `float32` rows; metrics also take `double-single` with `timestampsLow`. The context must match the bound view | Any unit; irregular axis only; `maximumGap` in the context unit |
| Kalman and restriction | `GPUTrajectoryKalman`, trajectory restriction (`temporalContext` beside the `times` or `timestamps` fields) | `int64-words` or `float32-relative` | Kalman derives `timeUnitsPerSecond` from the context unit (1, 1e3, 1e6 or 1e9) for either encoding, and a `timeUnitsPerSecond` given with it must agree; without a context it is for words only, and float32 times are seconds |
| Knox pair test | `GPUKnoxTest` | `float32-relative`, origin subtracted on the CPU | Caller's unit; no context prop |
| Mantel pair test | `GPUMantelTest` (`temporalContext`, optional) | `float32-relative` only | Caller's unit and origin; direct, named-operation and standalone CPU forms capture and validate the descriptor without converting values. The caller supplies timestamps in that unit |
| Near-repeat, local Knox and Jacquez | `GPUNearRepeatKnox`, `GPULocalKnox`, `GPUJacquezTest` (`temporalContext`) | `float32-relative` only; `int64-words` and `double-single` are refused | Any unit, any axis, and the epoch must be a declared bigint origin (`'unix'` is refused for float32). `getGPUSpaceTimeGuardedRelativeTimes` builds the times and returns the context; `GPUTimeWordsRelative.outputTemporalContext` is the context of times produced on the GPU |
| Space-time kriging | `GPUSpaceTimeKriging` (`temporalContext`, optional `queryTemporalContext`) | `float32-relative` only; the query context must agree with the sample context on unit, epoch and encoding | Any unit and axis. `GPUTimeWordsRelative.outputTemporalContext` provides the context |
| Space-time clustering | `GPUDBSCAN`, `GPUOPTICS` (`temporalContext` with `times`; see [Space-time clustering](./space-time-clustering.md)) | `float32-relative` only; the context is required with `times` and rejected without | Any unit, time zone and axis; `timeEpsilon` and `maximumTimeDistance` are in the context unit |
| Neighbor search | `GPUNeighborSearch` (`temporalContext` with `times`; see [`GPUNeighborSearch`](./operators-weights.md)) | `float32-relative` only; the context is required with `times` and rejected without | Any unit, time zone and axis; `timeRadius` and `timeBounds` are in the context unit |
| Variogram | `GPUSpaceTimeVariogram` | `float32-relative` | Caller's unit and origin |
| Cube statistics | `GPUEmergingHotSpots`, `addSpaceTimeHotSpotsRecipe`, `GPUValueCube`, `GPUCellPeriodKeys` | Integer slice index, cell-major | The recipe's event words are Int64 milliseconds from the Unix epoch |
| Dynamics and panels | Markov chains, spatial panels | Integer period index, period-major | Not a time column |
| Raster stacks | `GPUChangeDetection`, `GPURasterTemporal`, `GPURasterSeasonalBreak` (`dateAxis`) | Integer slice index on regular slots, or `int64-words` with `{kind: 'irregular'}` through `dateAxis` | Any unit and epoch for the irregular axis, read as centred fractional years of 365.2425 days |
| Raster time series | `GPURasterWhittakerSmoother`, `GPURasterHANTS`, `GPURasterTemporalResample` (`dateAxis`); `GPURasterStreamingAccumulator` and the RTS-06 climate contributors take chunks with slot indices | `int64-words` with `{kind: 'irregular'}` for dates; slot indices otherwise | See [Raster time series](../gpu-raster/raster-time-series.md) |
