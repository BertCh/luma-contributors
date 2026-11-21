# Raster time series

`GPURasterHarmonicRegression`, `GPURasterClimateRunAccumulator`, `GPURasterGeometricMedianComposite`,
`GPURasterBestAvailablePixel`, `GPURasterBFASTMonitor`, `GPURasterClimateAnomaly`, `GPURasterHANTS`,
`GPURasterEOFProjection` and `GPURasterCOLDMonitor` accept aligned chunked cell
records. Keep each complete time series, mask and output record within a chunk; slice limits still
apply per cell. Harmonic regression defaults to storage-backed QR; the other reference slice caps remain.

Both time-series execution routes use the same cell topology: packed fields can accompany a
single-chunk vector when each field holds exactly `cellCount * entriesPerCell` entries. Once a
field has multiple chunks, every cell field must have the same number of chunks and the same
cell counts per chunk, with no split cell records. Shared date and parameter tables stay packed.
Calls using only packed fields retain their documented spare-capacity support.
HANTS and temporal resampling type their output views independently of the source view: a packed
source can feed matching one-chunk outputs, and a one-chunk source can feed packed outputs.

Writable logical ranges must be disjoint across fields and chunks. Co-bound writable ranges
must also have disjoint storage-aligned binding ranges. HANTS and temporal resampling can use
one pooled output buffer under these rules. A dispatch's read-only inputs and writable outputs
must use separate physical buffers, even for disjoint ranges. A physical buffer participating in
writes must use one imported handle; a buffer used only by read-only inputs may have multiple
imports. Pooling output fields requires an explicit per-dispatch access declaration in the
contributor; callers without one retain separate output buffers because a later dispatch may read
one output while writing another. HANTS and temporal resampling declare their accesses explicitly.

Contributors for stacks of raster time slices beyond regular composites: an irregular acquisition-date
axis, streaming accumulators for bounded chunks and declared total stream lengths, penalised and harmonic smoothers, gap filling onto a
regular grid, and climate indices. Import from `@luma-contributors/analysis/raster-analysis`. All are
`GPUCommandNodeProducer` classes. Every contributor is cell-major (`cell * sliceCount + slice`) and takes
`mask` per observation with zero rejecting a sample; non-finite values are missing. Each class
documents its per-frame and compile-time props in its source.

## Irregular date axis

`GPURasterIrregularDateAxis` declares a stack's acquisition dates: one `uint32x2` Int64 `(low, high)` row
per slice, with a [`GPUTemporalContext`](../gpu-spatial-analysis/temporal-context.md) of encoding
`int64-words` and `{kind: 'irregular'}`, any unit and epoch. Dates must be strictly increasing. They become
**centred fractional years** in float32: years of 365.2425 days after the midpoint of the first and last
acquisition, so the trend column is O(1) and harmonics stay in `[-1, 1]`. Elapsed years come from the Int64
words directly, so nearby dates keep full float32 precision. `getGPURasterCentredFractionalYears` gives the
same axis on the CPU, for example to choose a regular output grid. It rounds the word difference
to float32, then multiplies by a pinned float32 reciprocal of the rounded ticks per year:
`f32(f32(wordDifference) * f32(1 / f32(ticksPerYear)))`. The shader uses that same reciprocal literal
and multiplication sequence. This avoids backend division-versus-reciprocal endpoint drift, including
a 419-day millisecond span whose endpoints are `±0.5735914707183838` years.

`GPURasterTemporal`, `GPUChangeDetection` and `GPURasterSeasonalBreak` take this axis as `dateAxis`.
Omitted, slot behaviour and results are unchanged.

- `GPURasterTemporal`: the Savitzky-Golay polynomial is fitted on real elapsed time over the same window of
  neighbouring acquisitions. Composites do not depend on time.
- `GPUChangeDetection`: `senSlope` is per year of real elapsed time. Other outputs depend on order only.
- `GPURasterSeasonalBreak`: `period` is in fractional years, the slope is per year, and `missingPolicy:
  'interpolate'` uses elapsed time. Dummy seasons are defined on regular slots and are rejected.

## `GPURasterStreamingAccumulator`

Accumulates counts, sums, squared deviations and extrema chunk by chunk.
Each dispatch consumes at most 256 slices (`MAXIMUM_RASTER_STREAM_CHUNK_SLICE_COUNT`); memory is
O(cells x calendar slots), with at most 366 slots (`MAXIMUM_RASTER_STREAM_SLOT_COUNT`). The caller
declares the total stream bound with `maximumStreamSliceCount` (default 16,777,216 slices) for fixed-point
headroom and must reset or start a new stream before exceeding it; this is not an unlimited-total guarantee. Slot indices assign a
calendar slot (for example a month) to each chunk slice.

Set `reset: true` on the first chunk, then `reset: false` on every continuing dispatch; leaving it true
clears prior state on every dispatch. `activeSliceCount` allows a short last chunk, but the cell-major
input pitch remains the compile-time `chunkSliceCount`: pad the last chunk or use a matching constructor.
Keep the magnitude and total-stream bounds (hence fixed-point exponents) unchanged until reset. Declare
`maximumValueMagnitude`: a sample above it is treated as missing. Sums and squared deviations are **96-bit
fixed-point integers** (three `uint32` words per slot, decoded with `getGPURasterFixedPointValues`), because
float32 compensated summation was optimized away by the shader compiler on the tested GPU (the low word stayed
zero). Fixed-point sums are exact to the declared unit, about `maximumValueMagnitude * 2^-70` at the default
stream length.

Squared deviations center in fixed-point units before float conversion and scale before multiplication.
The first valid observation contributes zero; `10000000` and `10000001` contribute a total of `0.5`
in either order, within one chunk or across two one-slice submissions. Moments keep three words per slot
in squared observation units. General moment updates still round to float32 precision before quantization.

Moment-unit selection retains the wide squared magnitude bound instead of narrowing it to float32.
For `[2^69, 2^70]` with `maximumStreamSliceCount: 2`, both producer plans select moment exponent `50`
and retain M2 `2^137`. Reset all retained state when adopting corrected units for an existing
large-magnitude stream; repack its parameters with a newly created plan rather than reinterpreting old words.

The three independent limits are **256 slices per update**, **2^24 declared slices by default**
(`DEFAULT_MAXIMUM_RASTER_STREAM_SLICE_COUNT`) for fixed-point headroom over the whole stream, and
**4,294,967,295 (2^32−1) valid observations per cell/slot** (`MAXIMUM_RASTER_STREAM_OBSERVATION_COUNT`).
`GPURasterStreamingAccumulator` saturates its count at that last limit and retains the prior statistics
when another valid observation arrives. Supply optional `state.status` (one `uint32` per cell/slot):
0 certifies no count overflow, and 1 records overflow, sticky until reset. Check every affected slot with
`decodeGPURasterStreamingAccumulatorStatus` and publish derived means/variances only when `complete` is
true. Counts alone cannot certify completeness. This certificate does not check cumulative declared
headroom: the caller must track total submitted slices and reset before exceeding its promise.
Count saturation and this certificate describe this producer, not calibration or climate-run counters.
Zero declared headroom is accepted only for `reset: true, activeSliceCount: 0`, an empty reset.

The Kepler `runRasterTimeSeries` wrapper accepts rolling `windowLength` from 1 to 128 slices and
optional `activeSliceCount` from 0 to the allocated `sliceCount`; short chunks preserve
`cell * sliceCount + slice` pitch and inactive outputs are NaN. Fractional or out-of-range active
counts reject, and magnitude limits apply to live observations only. Omitted `calibrationSliceCount`
is `min(12, sliceCount)`, while explicitly oversized counts reject. Its `zeroAtom` boolean controls
calibration and evaluation together. Wrapper defaults are gamma on and Fisk off. Streaming results expose raw `columns.streamingStatus` and
decoded `streamingCountStatus`, in cell-major slot order. These certify observation-count saturation
only, not moment arithmetic. The status read must contain exactly `cellCount * slotCount` words;
malformed lengths reject the result. Statistics for a slot with count loss are NaN, using the
submitted slot indices even if caller dates change during readback. Streaming standard deviation
uses sample normalization (`n - 1`); Fisk APP calibration uses population variance (`n`).

This complete two-cell example retains one immutable encoding plan for writes and readback. Each update
has one active slice in a physical width of two; values and masks both keep that width. It assumes a
WebGPU `Device` supplied by the application; all buffers and the graph are created here.

```ts
import type {Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
import {
  GPURasterStreamingAccumulator,
  createGPURasterStreamingAccumulatorEncodingPlan,
  getGPURasterStreamingAccumulatorParameterValues,
  getGPURasterFixedPointValues,
  decodeGPURasterStreamingAccumulatorStatus
} from '@luma-contributors/analysis/raster-analysis';

async function accumulateTwoUpdates(device: Device) {
  const graph = new GPUCommandGraph(device);
  const plan = createGPURasterStreamingAccumulatorEncodingPlan({
    maximumValueMagnitude: 32, maximumStreamSliceCount: 1_000_000
  }, 2);
  const values = new GPUParameterBuffer(device, {id: 'values', format: 'float32', length: 4});
  const mask = new GPUParameterBuffer(device, {id: 'mask', format: 'uint32', length: 4});
  const parameters = new GPUParameterBuffer(device, {id: 'parameters', format: 'float32', length: 5});
  const counts = new GPUParameterBuffer(device, {id: 'counts', format: 'uint32', length: 2});
  const sums = new GPUParameterBuffer(device, {id: 'sums', format: 'uint32', length: 6});
  const moments = new GPUParameterBuffer(device, {id: 'moments', format: 'uint32', length: 6});
  const status = new GPUParameterBuffer(device, {id: 'status', format: 'uint32', length: 2});
  graph.add(new GPURasterStreamingAccumulator({
    cellCount: 2, chunkSliceCount: 2,
    values: values.importToGraph(graph), mask: mask.importToGraph(graph),
    parameters: parameters.importToGraph(graph),
    state: {counts: counts.importToGraph(graph), sums: sums.importToGraph(graph),
      moments: moments.importToGraph(graph), status: status.importToGraph(graph)}
  }));
  const compiled = await graph.compileAsync();
  const packet = new Float32Array(5);
  try {
    const updates = [Float32Array.of(12, 0, 22, 0), Float32Array.of(14, 0, 24, 0)];
    for (let update = 0; update < updates.length; update++) {
      // Validate/pack before any GPU write. On rejection, packet keeps its last valid bytes;
      // do not submit or advance the stream, and retry with corrected frame settings.
      getGPURasterStreamingAccumulatorParameterValues(plan, {
        reset: update === 0, activeSliceCount: 1
      }, packet);
      parameters.write(packet);
      values.write(updates[update]);
      mask.write(Uint32Array.of(1, 0, 1, 0));
      const encoder = device.createCommandEncoder();
      compiled.encode(encoder, {parameters: undefined});
      device.submit(encoder.finish()); // submit before overwriting queue-visible inputs
      const certificates = await status.buffer.readAsync();
      for (let cell = 0; cell < 2; cell++) {
        if (!decodeGPURasterStreamingAccumulatorStatus(certificates.subarray(4 * cell, 4 * cell + 4)).complete) {
          throw new Error('Partial statistics: reset before publishing derived values');
        }
      }
    }
    return {
      sums: getGPURasterFixedPointValues(await sums.buffer.readAsync(), plan.sumUnitExponent),
      moments: getGPURasterFixedPointValues(await moments.buffer.readAsync(), plan.momentUnitExponent)
    }; // sums [26, 46], squared-deviation sums [2, 2]
  } finally {
    compiled.destroy();
    for (const storage of [values, mask, parameters, counts, sums, moments, status]) storage.destroy();
  }
}
```

Analogous retained-state plans are
`createGPURasterClimateRunAccumulatorEncodingPlan({threshold, maximumIncrement, maximumStreamSliceCount}, chunkSliceCount)`
with `getGPURasterClimateRunAccumulatorParameterValues(plan, {reset, activeSliceCount})` and readback
`getGPURasterFixedPointValues(bytes, plan.degreeDayUnitExponent)`; and
`createGPURasterStandardizedIndexCalibrationEncodingPlan({location, maximumValueMagnitude, maximumStreamSliceCount}, chunkSliceCount)`
with `getGPURasterStandardizedIndexCalibrationParameterValues(plan, {reset, activeSliceCount})`.
Decode each calibration slot's three-word sums, logarithm sums and squared deviations using
`plan.sumUnitExponent`, `plan.logarithmUnitExponent` and `plan.momentUnitExponent`, respectively.
Changing magnitude, total-stream headroom, an encoding exponent, climate threshold/increment bound,
or calibration location requires a new plan and reset of all retained state. Changes to slot meaning,
mask policy, climate comparison condition, distribution or zero-atom policy likewise require a fresh model/reset rather than mixing
incompatible observations. Legacy settings signatures remain supported, with encoding settings fixed
until reset. Every packer validates the entire packet before writing a supplied target; preserve the
last valid GPU packet by packing first and skipping GPU writes/submission when validation rejects.

Supplied frame-packet targets need at least `GPU_RASTER_STREAMING_ACCUMULATOR_PARAMETER_LENGTH = 5`,
`GPU_RASTER_CLIMATE_RUN_ACCUMULATOR_PARAMETER_LENGTH = 4`,
`GPU_RASTER_STANDARDIZED_CALIBRATION_PARAMETER_LENGTH = 7`, or
`GPU_RASTER_STANDARDIZED_INDEX_PARAMETER_LENGTH = 2` float32 elements, respectively. Short targets
reject before any write; oversized subview tails and surrounding backing elements survive.

{/* parameters: GPURasterStreamingAccumulator generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPURasterStreamingAccumulator` parameters

Import from `@luma-contributors/analysis/raster-analysis`: `GPURasterStreamingAccumulator`.

- Packer: `getGPURasterStreamingAccumulatorParameterValues(settings, chunkSliceCount)`, writes `GPU_RASTER_STREAMING_ACCUMULATOR_PARAMETER_LENGTH` (5) `float32` values.
  - `chunkSliceCount` (`number`)
- Packer: `getGPURasterStreamingAccumulatorParameterValues(plan, frame)`, writes `GPU_RASTER_STREAMING_ACCUMULATOR_PARAMETER_LENGTH` (5) `float32` values.
  - `frame` (`GPURasterStreamFrameSettings`)

Settings of `getGPURasterStreamingAccumulatorParameterValues(settings, chunkSliceCount)` for `GPURasterStreamingAccumulatorSettings`:

| Field | Type | Description |
|---|---|---|
| `reset` | `boolean` | True on the first chunk of a stream, which clears the state first. |
| `activeSliceCount` (optional) | `number` | Slices of this chunk that hold data; defaults to the whole chunk. |
| `maximumValueMagnitude` | `number` | Largest absolute sample the stream can hold. It sets the fixed-point unit of the sums: declare it tightly, because a sample above it is treated as missing, and the unit is about `maximumValueMagnitude * 2^-70` at the default stream length. |
| `maximumStreamSliceCount` (optional) | `number` | Longest stream in slices, for the fixed-point headroom. Defaults to 2^24. |

Settings of `getGPURasterStreamingAccumulatorParameterValues(plan, frame)` for `Readonly<{ kind: "raster-streaming-accumulator"; chunkSliceCount: number; maximumValueMagnitude: number; maximumStreamSliceCount: number; sumUnitExponent: number; momentUnitExponent: number; }>`:

| Field | Type | Description |
|---|---|---|
| `kind` | `'raster-streaming-accumulator'` | - |
| `chunkSliceCount` | `number` | - |
| `maximumValueMagnitude` | `number` | - |
| `maximumStreamSliceCount` | `number` | - |
| `sumUnitExponent` | `number` | - |
| `momentUnitExponent` | `number` | - |

{/* /parameters */}

## `GPURasterRollingMean`

Trailing rolling mean over a chunked stream, the aggregation SPI and SPEI apply before fitting. A missing
sample in the window makes that output missing, as xclim's `rolling(...).mean(skipna=False)`. Window state
lives in caller-owned buffers between chunks.

`windowLength` accepts **1–128 observations**. On a short update, both `values` and the optional mask
retain the physical `chunkSliceCount` pitch, not `activeSliceCount`. For two cells, physical width 2,
active count 1 and `windowLength: 1`, write values `[12, 0, 22, 0]` with mask `[1, 0, 1, 0]`.
The active outputs at indices 0 and 2 are 12 and 22: both cells survive the one-sample window.
Indices 1 and 3 are padding, not samples of the second cell.

{/* parameters: GPURasterRollingMean generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPURasterRollingMean` parameters

Import from `@luma-contributors/analysis/raster-analysis`: `GPURasterRollingMean`.

- Packer: `getGPURasterStreamingAccumulatorParameterValues(settings, chunkSliceCount)`, writes `GPU_RASTER_STREAMING_ACCUMULATOR_PARAMETER_LENGTH` (5) `float32` values.
  - Input port: `parameters`.
  - `chunkSliceCount` (`number`)
- Packer: `getGPURasterStreamingAccumulatorParameterValues(plan, frame)`, writes `GPU_RASTER_STREAMING_ACCUMULATOR_PARAMETER_LENGTH` (5) `float32` values.
  - Input port: `parameters`.
  - `frame` (`GPURasterStreamFrameSettings`)

Settings of `getGPURasterStreamingAccumulatorParameterValues(settings, chunkSliceCount)` for `GPURasterStreamingAccumulatorSettings`:

| Field | Type | Description |
|---|---|---|
| `reset` | `boolean` | True on the first chunk of a stream, which clears the state first. |
| `activeSliceCount` (optional) | `number` | Slices of this chunk that hold data; defaults to the whole chunk. |
| `maximumValueMagnitude` | `number` | Largest absolute sample the stream can hold. It sets the fixed-point unit of the sums: declare it tightly, because a sample above it is treated as missing, and the unit is about `maximumValueMagnitude * 2^-70` at the default stream length. |
| `maximumStreamSliceCount` (optional) | `number` | Longest stream in slices, for the fixed-point headroom. Defaults to 2^24. |

Settings of `getGPURasterStreamingAccumulatorParameterValues(plan, frame)` for `Readonly<{ kind: "raster-streaming-accumulator"; chunkSliceCount: number; maximumValueMagnitude: number; maximumStreamSliceCount: number; sumUnitExponent: number; momentUnitExponent: number; }>`:

| Field | Type | Description |
|---|---|---|
| `kind` | `'raster-streaming-accumulator'` | - |
| `chunkSliceCount` | `number` | - |
| `maximumValueMagnitude` | `number` | - |
| `maximumStreamSliceCount` | `number` | - |
| `sumUnitExponent` | `number` | - |
| `momentUnitExponent` | `number` | - |

{/* /parameters */}

## `GPURasterWhittakerSmoother`

A supplied packer target must hold at least `GPU_RASTER_WHITTAKER_SMOOTHER_PARAMETER_LENGTH = 2`
float32 elements. Short targets throw before any write. Oversized targets, including nonzero-offset
subviews, keep every tail element after the two-word record and every surrounding backing element.

Weighted Whittaker smoother per cell: minimises `sum w (y - z)^2 + lambda |D z|^2` by a banded Givens QR of
`[sqrt(W); sqrt(lambda) D]` on the values minus their per-cell mean (added back afterwards; the smoother is
exact on constants), with zero weight on missing observations so gaps are filled. `D` is the order `d` difference (1 to 3). With
`dateAxis` it is the divided difference on acquisition dates, scaled by the mean spacing, so a regular axis
reduces to ordinary differences and `lambda` keeps its usual meaning. `lambda` is limited to `1e7`.

Fixed robust reweighting passes: Tukey `bisquare` on the residual over 6 MAD, or `upper-envelope` for
cloud-depressed series. Reference: the phenofit `smooth_wWHIT` definition (Eilers 2003, divided differences);
R is not installed, so the fixtures use a binary64 SciPy `solve_banded` implementation of the same algorithm
(`test/oracles/rts02-whittaker-hants-scipy.py`). phenofit itself remains a pending oracle.

float32 limits: the result agrees with a binary64 solve to about one float32 step of the published value
for every difference order and every `lambda` up to the cap of `1e7` (256 slices, series at 0.5, 300 and
3000 with one gap in five), and to within `2e-5` on the irregular fixture series. The error does not grow
with the offset of the data (kelvin, scaled reflectance). Status 2 means the triangular factor has a zero
or non-finite diagonal or the solution is not finite, for example `lambda = 0` with a gap; it is not raised
for a system that is only badly conditioned. Very sparse support is solved, with status 0: 2, 3 or 5 valid
observations out of 256 slices, at either end, in the middle or spread out, agree with an exact rational
solve (`test/oracles/rts02-whittaker-sparse-exact.py`) for `lambda` from 1 to `1e7`. There, orders 2 and 3
continue the curve as a line or a parabola across the gap and the float32 error grows with the gap: over
250 slices beyond 3 or 5 adjacent observations it stays within `6e-4` (order 2) and `1.3e-2` (order 3) of
the spread of the curve, and within one float32 step of the values for order 1 or for observations spread
over the series (order 3 spread: `1e-3` of the spread).

{/* parameters: GPURasterWhittakerSmoother generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPURasterWhittakerSmoother` parameters

Import from `@luma-contributors/analysis/raster-analysis`: `GPURasterWhittakerSmoother`.

- Packer: `getGPURasterWhittakerSmootherParameterValues(settings)`, writes `GPU_RASTER_WHITTAKER_SMOOTHER_PARAMETER_LENGTH` (2) `float32` values.

Settings of `getGPURasterWhittakerSmootherParameterValues`:

| Field | Type | Description |
|---|---|---|
| `smoothingParameter` | `number` | Penalty weight lambda of the Whittaker smoother; larger is smoother. |
| `minimumWeight` (optional) | `number` | Robust-weight floor, so a point is never removed outright. Defaults to `0.01`. |

{/* /parameters */}

## `GPURasterHANTS`

A supplied packer target must hold at least `GPU_RASTER_HANTS_PARAMETER_LENGTH = 3` float32 elements.
Short targets throw before any write; oversized subview tails and surrounding backing elements survive.

Harmonic ANalysis of Time Series (Roerink, Menenti and Verhoef 2000; GRASS `r.hants`): a least-squares
harmonic fit (`getRasterLeastSquaresWGSL`) with a bounded reject-and-refit loop that drops the worst
observation beyond the fit-error tolerance on the chosen side. A cell out of rejection passes publishes its
latest fit and reports status 8 (`converged = 0`). `2 * harmonicCount + 1 + degreeOfOverdetermination`
observations are always kept. Coefficients are centred on the axis midpoint. The GRASS module is a definition
reference only; the fixture is a binary64 NumPy implementation, and GRASS itself remains a pending oracle.

Working memory is checked at construction: 256 slices allow two harmonics; 128 slices allow six.

{/* parameters: GPURasterHANTS generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPURasterHANTS` parameters

Import from `@luma-contributors/analysis/raster-analysis`: `GPURasterHANTS`.

- Packer: `getGPURasterHANTSParameterValues(settings)`, writes `GPU_RASTER_HANTS_PARAMETER_LENGTH` (3) `float32` values.

Settings of `getGPURasterHANTSParameterValues`:

| Field | Type | Description |
|---|---|---|
| `period` | `number` | Base period of the harmonics, in fractional years with a date axis, otherwise in slices. |
| `fitErrorTolerance` | `number` | Fit-error tolerance: observations further than this from the curve are rejected. |
| `degreeOfOverdetermination` (optional) | `number` | Observations kept above the `2 * harmonicCount + 1` minimum; defaults to 1. |

{/* /parameters */}

## `GPURasterTemporalResample`

Fills a regular time grid (`gridStart`, `gridStep` in centred fractional years, up to 4096 points) by linear
interpolation between bracketing valid observations. Grid points outside the observed range get status 1;
points inside a gap larger than `maximumGap` get status 2. An exact match to a retained observation
keeps that observation with status 0 even when an adjacent gap exceeds `maximumGap`.
For an endpoint-inclusive grid, set optional compile-time `gridEnd` to the final centred coordinate
from `getGPURasterCentredFractionalYears`. It replaces the final step-derived coordinate, avoiding
float32 step-rounding overshoot; it must not precede the previous grid point. Kepler supplies the
final input date this way for its default 24-point grid. Without `gridEnd`, every coordinate remains
`gridStart + point * gridStep`, including the last.
The source route accepts 2–256 acquisitions per cell (`MAXIMUM_RESAMPLE_SLICE_COUNT`), and the
output route accepts 1–4096 target points per cell (`MAXIMUM_RESAMPLE_GRID_COUNT`). Fractional-year
coordinates are float32: distinct acquisitions can round to the same coordinate, so that route cannot
preserve exact acquisition identity. Interpolation uses those same rounded coordinates in both the
numerator and denominator, guards nonpositive spans and clamps its fraction to `[0,1]`. `maximumGap`
always uses elapsed years converted from the exact Int64 difference.

For exact identity, pass optional `targetTimes`: one packed `uint32x2` Int64 `(low, high)` row per
output point, strictly increasing in the acquisition axis's unit and epoch. It overrides the fractional
grid and cannot be combined with `gridEnd`. Word comparisons select exact hits and brackets; only
elapsed numerator/denominator differences are converted to float32 for interpolation. The three-word
`parameters` view remains required; its `maximumGap` still applies, while `gridStart`/`gridStep` are ignored.

Nothing is extrapolated. A Whittaker or HANTS output
is finite in its gaps, so it feeds this node directly, which then feeds regular-time operations.

{/* parameters: GPURasterTemporalResample generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPURasterTemporalResample` parameters

Import from `@luma-contributors/analysis/raster-analysis`: `GPURasterTemporalResample`.

- Packer: `getGPURasterTemporalResampleParameterValues(settings)`, writes `GPU_RASTER_TEMPORAL_RESAMPLE_PARAMETER_LENGTH` (3) `float32` values.

Settings of `getGPURasterTemporalResampleParameterValues`:

| Field | Type | Description |
|---|---|---|
| `gridStart` | `number` | - |
| `gridStep` | `number` | - |
| `maximumGap` (optional) | `number` | Largest allowed span between the bracketing observations; defaults to no limit. |

{/* /parameters */}

## Compositing and phenology

Per-cell compositing across the slices of a stack, and phenology metrics on a smoothed curve. The multivariate
contributors use the layout `(cell * sliceCount + slice) * bandCount + band` and treat an observation with any
non-finite band as missing as a whole; `mask` is per observation (`cell * sliceCount + slice`) and zero
rejects it. A cell without a valid observation publishes NaN and `0xffffffff` as its slice.

Reference: hdmedians (Apache-2.0) defines `nanmedoid` and the geometric median. hdmedians is not installed
here, so the fixture `test/oracles/fixtures/rts05-compositing-numpy.json` comes from binary64 NumPy
implementations of those definitions (`test/oracles/rts05-compositing-numpy.py`); the geometric median is
also checked against the true minimiser found by SciPy. TIMESAT parity and double-logistic fits are out of
scope: the latter are ill-conditioned in float32.

### Preparing acquisition planes

`prepareRasterStackInput` copies host arrays into the composite layout and returns `values`, optional
`mask`, `cellCount`, `sliceCount` and `bandCount`. Pass `sourceLayout: 'acquisition-major'` for one
row-major cells-times-bands plane per acquisition, or `'cell-major'` for one slices-times-bands history
per cell. Bands are innermost in both layouts. All acquisitions must already share the grid, band order,
calibration and chronological order; packing does not resample or calibrate them. If a mask is supplied,
`maskLayout` is required independently: one uint32 per cell/acquisition, zero rejects it.

```ts
import {prepareRasterStackInput} from '@luma-contributors/analysis/raster-analysis';

const prepared = prepareRasterStackInput({
  grid: {width: 2, height: 1, affine: [1, 0, 0, 0, -1, 0], pixelInterpretation: 'area'},
  bandCount: 1,
  sourceLayout: 'acquisition-major',
  slices: [Float32Array.of(0, 10), Float32Array.of(100, 20), Float32Array.of(200, 30)],
  maskLayout: 'acquisition-major',
  mask: [Uint32Array.of(1, 1), Uint32Array.of(1, 1), Uint32Array.of(1, 1)]
});
// prepared.values = [0,100,200, 10,20,30]; the per-cell medoids are [100,20].
// Upload prepared.values/mask and pass prepared's counts to either composite constructor.
```

For resident single-band data, use `GPUCubeLayoutTranspose` from
`@luma-contributors/analysis/dataframe-analysis`: `from: 'period-major'`, `periodCount: sliceCount`,
`cellCount`, `values: acquisitionMajorValues`, `output: cellMajorValues`. Transpose a uint32 QA mask
the same way. For multiband resident data, transpose each band's scalar cube separately before
interleaving bands in the composite layout.

### From acquisition NDVI to a temporal mean

On buffer bands, import `GPURasterNDVI` from `@luma.gl/graph/gpu-raster` and schedule each acquisition
with a distinct ID for every acquisition (where `slice` is its acquisition index):

```ts
import {GPURasterNDVI} from '@luma.gl/graph/gpu-raster';

declare const ndviProps: ConstructorParameters<typeof GPURasterNDVI>[0];
declare const slice: number;

new GPURasterNDVI({...ndviProps, id: `ndvi-${slice}`}).addToGraph(graph);
```

Store each date's row-major output in its own
`cellCount` prefix of a period-major buffer, at element offset `slice * cellCount`. Store externally
supplied QA or per-date validity in the same observation layout; a nonzero mask accepts the sample.
Transpose values and mask into cell-major views before reducing:

```ts
import {GPUCubeLayoutTranspose} from '@luma-contributors/analysis/dataframe-analysis';
import {GPURasterTemporal, type GPURasterTemporalProps}
  from '@luma-contributors/analysis/raster-analysis';

type TemporalProps = GPURasterTemporalProps;
declare const cellCount: TemporalProps['cellCount'];
declare const sliceCount: TemporalProps['sliceCount'];
declare const dateMajorNDVI: TemporalProps['values'];
declare const dateMajorQA: NonNullable<TemporalProps['mask']>;
declare const cellMajorNDVI: TemporalProps['values'];
declare const cellMajorQA: NonNullable<TemporalProps['mask']>;
declare const meanNDVI: TemporalProps['output']['values'];
declare const counts: TemporalProps['output']['counts'];
declare const chosenSlice: TemporalProps['output']['chosenSlice'];
declare const status: TemporalProps['output']['status'];

// dateMajorNDVI and dateMajorQA hold consecutive acquisition planes; destination views have
// cellCount * sliceCount scalar elements. GraphDataView offsets use bytes (element offset * 4).
graph.add(new GPUCubeLayoutTranspose({
  id: 'ndvi-values-to-cell-major',
  values: dateMajorNDVI, from: 'period-major', cellCount, periodCount: sliceCount,
  output: cellMajorNDVI
}));
graph.add(new GPUCubeLayoutTranspose({
  id: 'ndvi-qa-to-cell-major',
  values: dateMajorQA, from: 'period-major', cellCount, periodCount: sliceCount,
  output: cellMajorQA
}));
graph.add(new GPURasterTemporal({
  cellCount, sliceCount, values: cellMajorNDVI, mask: cellMajorQA, operation: 'mean',
  output: {values: meanNDVI, counts, chosenSlice, status}
}));
// Date planes [0,.2] and [.6,1] become [0,.6,.2,1], yielding means [.3,.6].
```

For texture inputs, use the buffer/texture-capable local normalized-difference route described in
[spectral indices](spectral-indices.md), or gather a published texture band into a buffer before the
NDVI step. Keep masks in the same named observation layout throughout.

### `GPURasterMedoidComposite`

The valid observation with the smallest sum of Euclidean distances (over all bands) to the other valid
observations. The result is a real acquisition, so `output.sliceIndex` is the provenance of every band. Ties go
to the earliest slice. Work is `O(slices^2 * bands)` per cell on one invocation, so `sliceCount` is limited to 64
and `bandCount` to 16. Float32 distance sums that differ by less than rounding can pick a different slice than
binary64 when two or more bands vary. Multiband distances are compared after a common power-of-two
scaling chosen from inter-observation differences per cell, so
finite spectra at scales such as `1e-25` and `1e20` do not lose their ordering through squared-distance
underflow or overflow. Constant bands contribute zero distance, even with a large absolute value
such as `1e24`, and cannot change provenance. With at most one varying band, both CPU and GPU
select the earliest observation in its closed median interval,
including even sample counts and a permuted acquisition order.

### `GPURasterGeometricMedianComposite`

The point minimising the sum of Euclidean distances to the valid observations over all bands together (Roberts,
Mueller and McIntyre 2017). Bands are fitted jointly and the result is a synthetic spectrum. Compute
NDVI from that spectrum when you want the index of the composite. To obtain the temporal median of
acquisition-level NDVI, compute NDVI per valid acquisition, then take that scalar's temporal median.
These are different scientific quantities; NDVI does not commute with joint fitting. With band order
`[nearInfrared, red]` and `NDVI = (nearInfrared - red) / (nearInfrared + red)`:

- `[[1,1],[3,1],[1,2]]` gives composite NDVI approximately `-0.0195127`, while the acquisition NDVIs
  `[0,0.5,-1/3]` have median `0`.
- `[[0.5,0.25],[1,0.25],[0.5,0.75]]` gives composite NDVI approximately `0.2600577`, while
  `[1/3,0.6,-0.2]` have median `0.3333333`.

A singleton is copied exactly with `lastStep = 0`; identical nonzero observations stay unchanged.
Updates accumulate weighted displacements from the current estimate to avoid rounding an absolute
weighted coordinate. With at least two valid observations it runs Weiszfeld's recursion from the mean
for exactly `iterationCount` iterations (at most 64), on at most 256 slices and 16 bands per cell,
with each distance floored at
`distanceEpsilon`. `output.lastStep` is the length of the last step, the convergence indicator to read when the
observations are few or skewed. The fixture cells agree with binary64 to within `2e-5` and sit within `1e-5`
relative of the SciPy minimum after 32 iterations.

Call `getGPURasterGeometricMedianCompositeOutputLayout({cellCount, bandCount, lastStep})` before
allocating outputs. It returns scalar formats and element counts without allocating storage.
`lastStep` defaults to false: its descriptor is absent unless requested. The contributor's
validation, `GPURasterGeometricMedianCompositeOperation`, the CPU operation's graph outputs and
the pure CPU result's allocation all use this layout. The pure CPU function always returns
`lastStep`; the operation publishes it only when `output.lastStep` names a graph value.

{/* geometric-median-output-layout:start */}

| Output field | Scalar format | Element count | Condition and meaning |
| --- | --- | --- | --- |
| `composite` | `float32` | `cellCount * bandCount` | Always: `cell * bandCount + band`; NaN when the cell has no valid observation. |
| `status` | `uint32` | `cellCount` | Always: per-cell code 0 for a composite, 1 for no valid observation. |
| `lastStep` | `float32` | `cellCount` | When `lastStep: true`: final step length per cell; NaN when there is no valid observation. |

{/* geometric-median-output-layout:end */}

The table is checked against the public layout in the node tests. Counts and the
`cellCount * bandCount` product must be positive integers no greater than 4,294,967,295.
The layout covers both execution forms; GPU execution additionally limits each cell to 256 slices,
16 bands and 64 iterations. The CPU twin retains the iteration budget and uint32 indexing limits,
and allows larger slice, band and iteration counts. The GPU operation uses exact-sized packed
graph values; a direct contributor also accepts packed output views with spare capacity and
nonzero byte offsets, writing only the used prefix. Chunked cell fields must follow the shared
time-series topology rules above. Output formats must match the table, including `uint32` status.

For example, two cells with three bands need six `float32` composite entries and two `uint32`
status entries. Requesting `lastStep: true` adds two `float32` entries. Input masks and output
buffers remain caller-owned.

{/* parameters: GPURasterGeometricMedianComposite generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPURasterGeometricMedianComposite` parameters

Import from `@luma-contributors/analysis/raster-analysis`: `GPURasterGeometricMedianComposite`.

- Packer: `getGPURasterGeometricMedianCompositeParameterValues(settings?)`, writes `GPU_RASTER_GEOMETRIC_MEDIAN_COMPOSITE_PARAMETER_LENGTH` (1) `float32` values.

Settings of `getGPURasterGeometricMedianCompositeParameterValues`:

| Field | Type | Description |
|---|---|---|
| `distanceEpsilon` (optional) | `number` | Smallest distance used as a Weiszfeld weight denominator, in value units: an estimate that lands on an observation stays there instead of dividing by zero. Defaults to `1e-5`. |

{/* /parameters */}

### `GPURasterBestAvailablePixel`

Best-available-pixel scoring (Griffiths et al. 2013): `score = wDay * exp(-0.5 (dDay / sigma)^2) +
wYear * max(0, 1 - |year - targetYear| / maximumYearDistance) + wCloud * clamp((cloudDistance - minimum) /
(maximum - minimum), 0, 1)`, with `dDay` the circular distance in days to the target day of year. The cell takes
the highest score, the earliest slice on ties, and optionally its band values. Per-slice day of year and year come
in `sliceAttributes`. `cloudDistance` is per observation, in pixels to the nearest cloud or shadow of its slice:
run `GPUDistanceField` on each slice's cloud mask and pass the result (positive infinity means no cloud in
reach; without the view the cloud score is zero).

Cloud-distance limits are in pixels and must satisfy `maximumCloudDistance > minimumCloudDistance`
both as requested and after float32 encoding. For example `[10, 10.0000001]` is rejected because both
encode as `10`; a rejected pack leaves the destination unchanged.

{/* parameters: GPURasterBestAvailablePixel generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPURasterBestAvailablePixel` parameters

Import from `@luma-contributors/analysis/raster-analysis`: `GPURasterBestAvailablePixel`.

- Packer: `getGPURasterBestAvailablePixelParameterValues(settings)`, writes `GPU_RASTER_BEST_AVAILABLE_PIXEL_PARAMETER_LENGTH` (9) `float32` values.

Settings of `getGPURasterBestAvailablePixelParameterValues`:

| Field | Type | Description |
|---|---|---|
| `targetDayOfYear` | `number` | Target day of year in `[0, 366]`. |
| `dayOfYearSigma` | `number` | Standard deviation of the Gaussian day-of-year score, in days. |
| `dayOfYearWeight` | `number` | Weight of the day-of-year score; Griffiths et al. 2013 example `0.5`. |
| `targetYear` | `number` | Target year in the unit of the `sliceAttributes` year column. |
| `maximumYearDistance` | `number` | Year distance at which the linear year score reaches zero. |
| `yearWeight` | `number` | Weight of the year score; example `0.2`. |
| `minimumCloudDistance` | `number` | Cloud distance at or below which the cloud score is zero (pixels), example `10`. |
| `maximumCloudDistance` | `number` | Cloud distance at or above which the cloud score is one (pixels), example `100`. |
| `cloudWeight` | `number` | Weight of the cloud-distance score; example `0.3`. |

{/* /parameters */}

### `GPURasterPhenologyMetrics`

Up to two seasons per cell on 2–256 slices, from amplitude thresholds on the smoothed curve, for example `output.smoothed` of
`GPURasterWhittakerSmoother` (same `cell * sliceCount + slice` layout; pass the same `dateAxis` and times are
centred fractional years, otherwise slice indices). Base is the curve minimum and amplitude the maximum minus the
minimum. A season starts where the curve last falls below `startFraction` of the amplitude before its peak and ends
where it first falls below `endFraction` after it, both by linear interpolation (the Copernicus MR-VPP
definitions use 0.25 and 0.15). The first season is the highest sample; the second is the highest sample outside
the first season's span that reaches `secondSeasonMinimumFraction`. Seasons are ordered by peak time. A crossing
that is not found cuts the season at the series end or at the other season and sets a bit of
`GPU_RASTER_PHENOLOGY_FLAG`. `output.metrics` holds the ten fields of `GPU_RASTER_PHENOLOGY_METRIC` per season:
start, end, peak time, length, peak value, base value, amplitude, left and right slope, and the integral between
the curve and the base from start to end. Cell status is 1 for a non-finite curve and 2 for an amplitude below
`minimumAmplitude`. Start/end thresholds and the second-season threshold are fractions (0–1);
`minimumAmplitude` has the curve's value unit. When omitted, `secondSeasonMinimumFraction` is the
larger encoded start/end fraction, so ordinary decimal fractions such as start `0.7`, end `0.2` work
without rounding-dependent rejection. Explicit thresholds must respect the requested and encoded order.

{/* parameters: GPURasterPhenologyMetrics generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPURasterPhenologyMetrics` parameters

Import from `@luma-contributors/analysis/raster-analysis`: `GPURasterPhenologyMetrics`.

- Packer: `getGPURasterPhenologyMetricsParameterValues(settings)`, writes `GPU_RASTER_PHENOLOGY_METRICS_PARAMETER_LENGTH` (4) `float32` values.

Settings of `getGPURasterPhenologyMetricsParameterValues`:

| Field | Type | Description |
|---|---|---|
| `startFraction` | `number` | Fraction of the curve amplitude above the base where a season starts, in `(0, 1)`; MR-VPP uses `0.25`. |
| `endFraction` | `number` | Fraction of the curve amplitude above the base where a season ends, in `(0, 1)`; MR-VPP uses `0.15`. |
| `minimumAmplitude` | `number` | Cells whose amplitude (maximum minus minimum) is below this get status 2 and no season. |
| `secondSeasonMinimumFraction` (optional) | `number` | A second season needs a peak at least this fraction of the amplitude above the base. It must be at least both start and end fractions, so the shoulders of the first season are not a second one. Defaults to the larger of the two. |

{/* /parameters */}

## SPI and SPEI

`GPURasterStandardizedIndex` application evaluates up to 16,777,216 (2^24) slices per cell/chunk
(`MAXIMUM_STANDARDIZED_INDEX_CHUNK_SLICE_COUNT`), subject to device bindings. Calibration independently
accepts at most 256 slices per update (`MAXIMUM_RASTER_STREAM_CHUNK_SLICE_COUNT`); its declared total
headroom defaults to 2^24 slices and must cover the whole calibration stream. Cell records and aligned outputs
accept chunked input without changing the fitted calendar-slot model.

`GPURasterStandardizedIndexCalibration` streams calibration chunks (for example 30 years of months) into
per-slot statistics and writes a fit per calendar slot. Positive location-adjusted values enter the
continuous fit; other finite observations are counted but excluded, subject to the magnitude bound.
With `zeroAtom: true`, original zeros instead enter the atom count and never the continuous fit. `GPURasterStandardizedIndex` then evaluates any chunk against that fit. Pass the rolling
mean output for aggregations such as SPI-3. Both match xclim's `method='APP'` (Thom's gamma estimator for
`gamma` with `floc`, the method-of-moments estimator for `fisk` with `floc`) and xclim's defaults for zero
atom handling (`prob_zero_interpolation='upper'`, `plotting_position_zero='ecdf'`) when enabled.
`location` is the fixed `floc` used for water-balance series that may be negative.

Set the construction option `zeroAtom` consistently on calibration and application; the library default
is **false**. Parameter packers cannot enable the atom. The atom is always at **original zero**, independently
of the continuous distribution's `location`: both +0 and -0 take its jump. If `q` is the fitted zero
probability and `G(x)` the location-shifted continuous CDF, the mixture CDF is `(1-q)G(x)` for `x < 0`
and `(1-q)G(x) + q` for `x >= 0`. Its survival probability is `(1-q)(1-G(x)) + q` below zero and
`(1-q)(1-G(x))` at/above zero; the implementation evaluates a small continuous tail directly.

Reference: xclim 1.0.0, Apache-2.0, run for `test/oracles/fixtures/rts06-climate-indices-xclim.json` (numba and
dask imports stubbed; see the fixture). SPI-1, SPI-3 and SPEI agree with xclim to within 0.01 over 420 months in
float32.

float32 limits and what the code does about them:

- **APP, not iterative ML.** Newton iterations on the shape are noisy in float32; APP is closed form. Only
  sums over the slot enter it.
- **Fit sums use fixed-point storage.** The gamma estimator is driven by `ln(mean) - mean(ln x)`, a
  difference of nearly equal numbers. Sums of `x`, `ln x` and squared deviations use 96-bit integers.
  Squared-deviation updates center in integer units before float conversion and scale before multiplication;
  the first observation contributes zero. Updates still round to float32 precision before quantization,
  and fit evaluation retains float32 limits. Calibration's squared deviations occupy words 6–8 of each
  nine-word slot, in squared units of the positive, location-adjusted observation.
  Calibration retains low bits of `value - location` in both the sum and moment update: shifting
  `[16, 16 + 2^-19]` by location `-64` preserves M2 `2^-39`. Fisk fitting uses a dimensionless
  standard-deviation-to-mean ratio, avoiding overflow from cubing a large mean or decoding a whole
  large moment. These arithmetic corrections preserve existing state units; reset/replay an older
  calibration to recover observations whose low bits or moments were already discarded.
- **Tails are computed directly.** The gamma CDF and quantile near 1 cannot be represented as `1 - p` in
  float32, so whichever tail is small is computed directly (series below, continued fraction above) and the
  quantile is taken of that tail. Bounded gamma evaluation must converge: exhausting its evaluation
  budget produces NaN with status 3 (`iterationLimit`), never an unfinished approximation marked usable.
- **Clipped to `+-8.21`**, xclim's bound. Tails flush to zero near `1e-38` in float32, well beyond that.
- **The fit needs at least two positive values per slot**; a slot without a usable fit marks each active, finite, unmasked observation NaN
  with status 2 (`calibrationUnavailable`), distinct from evaluation exhaustion. Invalid or masked
  observations are checked first and return NaN with status 1 (`invalidInput`), even in an unusable slot.

Check each observation's status before publishing its index: code 0 is usable, 1 is invalid input,
2 is unusable calibration, and 3 is bounded-evaluation exhaustion. The decoder returns named fields:

```ts
import {decodeGPURasterStandardizedIndexStatus} from '@luma-contributors/analysis/raster-analysis';
// A bounded byte view may start inside a larger GPU readback. Include the COMPLETE uint32 word.
const allocation = Uint32Array.of(99, 3, 88);
const oneStatus = new Uint8Array(allocation.buffer, 4, 4);
const status = decodeGPURasterStandardizedIndexStatus(oneStatus);
// {complete: false, invalidInput: false, calibrationUnavailable: false, iterationLimit: true}
```

The decoder accepts `Uint32Array`, bounded `Uint8Array` or `ArrayBuffer` readbacks. Byte offset and
length are respected; a partial uint32 word (for example three bytes) or a missing status word rejects.
Similarly, `getGPURasterFixedPointValues` requires complete uint32 elements AND complete three-word
96-bit values, including when reading calibration words 6–8 as a bounded 12-byte view.

{/* parameters: GPURasterStandardizedIndex generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPURasterStandardizedIndex` parameters

Import from `@luma-contributors/analysis/raster-analysis`: `GPURasterStandardizedIndex`.

- Packer: `getGPURasterStandardizedIndexParameterValues(settings, chunkSliceCount)`, writes `GPU_RASTER_STANDARDIZED_INDEX_PARAMETER_LENGTH` (2) `float32` values.
  - `chunkSliceCount` (`number`)

Settings of `getGPURasterStandardizedIndexParameterValues`:

| Field | Type | Description |
|---|---|---|
| `activeSliceCount` (optional) | `number` | - |
| `location` (optional) | `number` | - |

{/* /parameters */}

{/* parameters: GPURasterStandardizedIndexCalibration generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPURasterStandardizedIndexCalibration` parameters

Import from `@luma-contributors/analysis/raster-analysis`: `GPURasterStandardizedIndexCalibration`.

- Packer: `getGPURasterStandardizedIndexCalibrationParameterValues(settings, chunkSliceCount)`, writes `GPU_RASTER_STANDARDIZED_CALIBRATION_PARAMETER_LENGTH` (7) `float32` values.
  - `chunkSliceCount` (`number`)
- Packer: `getGPURasterStandardizedIndexCalibrationParameterValues(plan, frame)`, writes `GPU_RASTER_STANDARDIZED_CALIBRATION_PARAMETER_LENGTH` (7) `float32` values.
  - `frame` (`GPURasterStreamFrameSettings`)

Settings of `getGPURasterStandardizedIndexCalibrationParameterValues(settings, chunkSliceCount)` for `GPURasterStandardizedIndexCalibrationSettings`:

| Field | Type | Description |
|---|---|---|
| `reset` | `boolean` | - |
| `activeSliceCount` (optional) | `number` | - |
| `location` (optional) | `number` | - |
| `maximumValueMagnitude` | `number` | - |
| `maximumStreamSliceCount` (optional) | `number` | - |

Settings of `getGPURasterStandardizedIndexCalibrationParameterValues(plan, frame)` for `Readonly<{ kind: "raster-standardized-index-calibration"; chunkSliceCount: number; maximumStreamSliceCount: number; location: number; maximumValueMagnitude: number; sumUnitExponent: number; logarithmUnitExponent: number; momentUnitExponent: number; }>`:

| Field | Type | Description |
|---|---|---|
| `kind` | `'raster-standardized-index-calibration'` | - |
| `chunkSliceCount` | `number` | - |
| `maximumStreamSliceCount` | `number` | - |
| `location` | `number` | - |
| `maximumValueMagnitude` | `number` | - |
| `sumUnitExponent` | `number` | - |
| `logarithmUnitExponent` | `number` | - |
| `momentUnitExponent` | `number` | - |

{/* /parameters */}

## `GPURasterClimateRunAccumulator`

Streaming spell lengths, exceedance counts and degree days per cell against a threshold. The condition
(`above`, `at-or-above`, `below`, `at-or-below`) is fixed at construction and matches xclim's comparison
operators; spell length, wet/dry spell and degree-day definitions match xclim 1.0.0 for reset-per-year chunks.
A missing sample breaks the run and is counted. `minimumSpellLength` counts spells when they reach the length.

{/* parameters: GPURasterClimateRunAccumulator generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPURasterClimateRunAccumulator` parameters

Import from `@luma-contributors/analysis/raster-analysis`: `GPURasterClimateRunAccumulator`.

- Packer: `getGPURasterClimateRunAccumulatorParameterValues(settings, chunkSliceCount)`, writes `GPU_RASTER_CLIMATE_RUN_ACCUMULATOR_PARAMETER_LENGTH` (4) `float32` values.
  - `chunkSliceCount` (`number`)
- Packer: `getGPURasterClimateRunAccumulatorParameterValues(plan, frame)`, writes `GPU_RASTER_CLIMATE_RUN_ACCUMULATOR_PARAMETER_LENGTH` (4) `float32` values.
  - `frame` (`GPURasterStreamFrameSettings`)

Settings of `getGPURasterClimateRunAccumulatorParameterValues(settings, chunkSliceCount)` for `GPURasterClimateRunAccumulatorSettings`:

| Field | Type | Description |
|---|---|---|
| `reset` | `boolean` | - |
| `activeSliceCount` (optional) | `number` | - |
| `threshold` | `number` | - |
| `maximumIncrement` | `number` | - |
| `maximumStreamSliceCount` (optional) | `number` | - |

Settings of `getGPURasterClimateRunAccumulatorParameterValues(plan, frame)` for `Readonly<{ kind: "raster-climate-run-accumulator"; chunkSliceCount: number; maximumStreamSliceCount: number; threshold: number; maximumIncrement: number; degreeDayUnitExponent: number; }>`:

| Field | Type | Description |
|---|---|---|
| `kind` | `'raster-climate-run-accumulator'` | - |
| `chunkSliceCount` | `number` | - |
| `maximumStreamSliceCount` | `number` | - |
| `threshold` | `number` | - |
| `maximumIncrement` | `number` | - |
| `degreeDayUnitExponent` | `number` | - |

{/* /parameters */}

## `GPURasterClimateAnomaly`

`GPURasterClimateAnomaly` reads the retained state of a `GPURasterStreamingAccumulator`. Use the
accumulator's `parameters` view for `accumulatorParameters` and pass the same caller-owned state
as `state: retainedState`. To detect observation-count overflow, allocate `state.status` on the
accumulator with one `uint32` word per cell and slot and pass that view in the retained state.
The status is sticky until reset, and a nonzero word makes the corresponding slot's mean, sample
standard deviation, and anomalies NaN.

An anomaly is the query value minus its slot mean. When the query fits the fixed-point arithmetic,
the reader subtracts the retained sum from `count × encodedQuery` in integer units, then divides
by `count` and applies the encoding scale. For a wider query, it scales the query and mean to a
common power of two before subtracting. Centering this way preserves differences that can be lost
when subtracting the rounded `climatologyMean` output. With `standardized: true`, supply
`state.moments` to divide by the slot's sample standard deviation (`ddof = 1`). The reader divides
the scaled residual by the scaled deviation, then multiplies by two raised to the difference of
their exponents. A standardized anomaly is NaN if the slot has fewer than two observations or
its standard deviation is zero.

{/* parameters: GPURasterClimateAnomaly generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPURasterClimateAnomaly` parameters

Import from `@luma-contributors/analysis/raster-analysis`: `GPURasterClimateAnomaly`.

- Packer: `getGPURasterStreamingAccumulatorParameterValues(settings, chunkSliceCount)`, writes `GPU_RASTER_STREAMING_ACCUMULATOR_PARAMETER_LENGTH` (5) `float32` values.
  - Input port: `accumulatorParameters`.
  - `chunkSliceCount` (`number`)
- Packer: `getGPURasterStreamingAccumulatorParameterValues(plan, frame)`, writes `GPU_RASTER_STREAMING_ACCUMULATOR_PARAMETER_LENGTH` (5) `float32` values.
  - Input port: `accumulatorParameters`.
  - `frame` (`GPURasterStreamFrameSettings`)

Settings of `getGPURasterStreamingAccumulatorParameterValues(settings, chunkSliceCount)` for `GPURasterStreamingAccumulatorSettings`:

| Field | Type | Description |
|---|---|---|
| `reset` | `boolean` | True on the first chunk of a stream, which clears the state first. |
| `activeSliceCount` (optional) | `number` | Slices of this chunk that hold data; defaults to the whole chunk. |
| `maximumValueMagnitude` | `number` | Largest absolute sample the stream can hold. It sets the fixed-point unit of the sums: declare it tightly, because a sample above it is treated as missing, and the unit is about `maximumValueMagnitude * 2^-70` at the default stream length. |
| `maximumStreamSliceCount` (optional) | `number` | Longest stream in slices, for the fixed-point headroom. Defaults to 2^24. |

Settings of `getGPURasterStreamingAccumulatorParameterValues(plan, frame)` for `Readonly<{ kind: "raster-streaming-accumulator"; chunkSliceCount: number; maximumValueMagnitude: number; maximumStreamSliceCount: number; sumUnitExponent: number; momentUnitExponent: number; }>`:

| Field | Type | Description |
|---|---|---|
| `kind` | `'raster-streaming-accumulator'` | - |
| `chunkSliceCount` | `number` | - |
| `maximumValueMagnitude` | `number` | - |
| `maximumStreamSliceCount` | `number` | - |
| `sumUnitExponent` | `number` | - |
| `momentUnitExponent` | `number` | - |

{/* /parameters */}
