---
title: GPU Dataframe analysis contributors
description: Column profiles, classification, group statistics, key lookups, composite indicators, time windows, and pair statistics.
---


# GPU Dataframe analysis contributors


## Overview

The `@luma-contributors/analysis/dataframe-analysis` entry point exports analysis contributors for GPU-resident
columns: column profiles, quantiles and class breaks, color scales, group statistics, key lookups,
composite scores, inequality measures, calendar buckets, period indices, cube layout transposes, relative times,
time-window filters, temporal reductions,
and pair-statistics tools (variograms, correlograms, Ripley's K, point-pattern indices). Each is an
algorithm or workflow that declares resources and command nodes into a caller-owned
`GPUCommandGraph` through `getCommandNodes(graph)` (`GPUCommandNodeProducer`). A contributor never compiles,
submits, encodes, or reads back.

### Execution and ownership of wide numeric operations

The entry point also exports one-shot functions. These create a separate graph on every call and
await a summary on the CPU; use the contributor when the result must compose into an existing graph.

| API | Execution and transfer cost | Ownership |
| --- | --- | --- |
| `GPUWideNumericScan` | Declares nodes in the caller's graph over resident word vectors; no implicit upload, submission or readback. | Caller owns input vectors, initialized output words, compilation, encoding, submission and optional readback. |
| `runWideNumericKernel` | Allocates and initializes a result buffer, compiles a new graph, encodes, submits, and awaits readback of `max(16, resultWordCount * 4)` bytes; returns `resultWordCount` words. | Borrows resident inputs; destroys its compiled graph and result buffer in `finally`. |
| `aggregateInteger64OnGpu`, `aggregateInteger64Range` | Resident-vector one-shot aggregation through `runWideNumericKernel`; reads 44 bytes and returns a CPU summary. | Borrows input vectors and validity; releases its graph and result buffer. |
| `scanWideNumericColumnOnGpu` | Resident-vector one-shot narrowing diagnostics; reads 24 bytes and returns a CPU report. | Borrows input vectors and validity; releases its graph and result buffer. |
| `aggregateDecimal128OnGpu` | Resident-vector one-shot decimal aggregation; reads 76 bytes and returns a CPU summary. | Borrows word vectors and validity; releases its graph and result buffer. |
| `makeDecimal128WordVectors` | Uploads an Arrow Decimal128 column's words and, when needed, validity. Does not submit a scan or read back. | Caller must invoke the returned `destroy()` to release the uploaded vectors. |
| `aggregateDecimal128Column` | Uploads an Arrow Decimal128 column, performs the one-shot decimal aggregation, and awaits its summary readback. | Releases the uploaded vectors in `finally` as well as the scan's graph and result buffer. |

A graph-resident count of present nonzero values can feed another node without a CPU wait:

```ts
import {Buffer, type Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import type {GPUVector} from '@luma.gl/graph/gpu-data';
import {GPUWideNumericScan} from '@luma-contributors/analysis/dataframe-analysis';

declare const device: Device;
declare const words: GPUVector<'uint32x2'>;
const resultBuffer = device.createBuffer({
  data: new Uint32Array(4),
  usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
});
const graph = new GPUCommandGraph(device);
const output = graph.createDataView(
  graph.importBuffer({id: 'present-count', byteLength: 16, usage: resultBuffer.usage}, resultBuffer),
  {format: 'uint32', length: 4}
);
graph.add(new GPUWideNumericScan({
  id: 'present-count', words, resultWordCount: 1, output,
  phaseBodies: ['if (present && any(pair != vec2u(0u))) { atomicAdd(&result[0], 1u); }']
}));
// Add downstream nodes reading output before compiling.
const compiled = graph.compile();
const encoder = device.createCommandEncoder();
compiled.encode(encoder, {parameters: undefined});
device.submit(encoder.finish());
// Before reusing this graph, reset resultBuffer to zero, then encode and submit again.
// Once all consumers have finished: compiled.destroy(); resultBuffer.destroy();
```

For an explicit CPU summary, use the one-shot path and await its readback:

```ts
import type {Device} from '@luma.gl/core';
import type {GPUVector} from '@luma.gl/graph/gpu-data';
import {aggregateInteger64OnGpu} from '@luma-contributors/analysis/dataframe-analysis';

declare const device: Device;
declare const words: GPUVector<'uint32x2'>;
const summary = await aggregateInteger64OnGpu({device, words, signed: true});
// summary.sum is a bigint on the CPU. The caller still owns words.
```

## When to use it

Use these contributors for the fixed analyses that map and dashboard applications repeat every
frame, where inputs are packed graph views and results must stay on the GPU. Use the
[`GPUDataFrame` query API](https://luma.gl/docs/api-reference/experimental/gpu-dataframe-operations) when the question is a general expression,
filter, or aggregation plan.

- `GPUGroupStatistics` groups by a 32- or 64-bit key and computes many statistics per group (moments,
  median, percentiles, mode, unique count). Use `GPUDataFrameGroupedAggregationQuery` for dense
  categorical keys and standard aggregations.
- `GPUKeyLookup` looks up each left row's key in a right table and supports 1:n aggregates and 64-bit
  keys. Use `GPUDataFrameJoinQuery` for a unique right-side key lookup between dataframes.
- `GPUColumnProfile` fuses count, nulls, moments, histogram, distinct estimate, and top categories
  into one statistics panel. Use `GPUDataFrameHistogramQuery` when you only need an explicit
  histogram.
- `GPUTimeWindowFilter` publishes compacted source rows, optional caller IDs, masks, and fade weights for a window that moves
  every frame, including exact `Int64` time words. Express a fixed time predicate as a dataframe
  filter expression instead.

## Quick start

Use the [complete GPU walkthrough](./operation-forms-and-cpu-twins.md#point-density-from-device-to-readback)
for device creation, buffer imports, submission, typed readback and cleanup. The
[complete first-run time-window filter](./README.md#first-run-time-window-filter) supplies the
family-specific runnable example. A time-window filter
uses a compact output object rather than a single output view:

Import `Float32Array.of(0, 10, 20)` as a three-row `float32` timestamps view. Times and
window bounds share an epoch and units. Allocate the output views below, then add the filter
with this checked function.

```ts
import type {Device} from '@luma.gl/core';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
import {
  GPUTimeWindowFilter, getGPUTimeWindowFilterParameterValues,
  GPU_TIME_WINDOW_FILTER_PARAMETER_LENGTH
} from '@luma-contributors/analysis/dataframe-analysis';

export function addTimeWindowExample(
  device: Device, graph: GPUCommandGraph, timestamps: GraphDataView<'float32'>,
  rows: GraphDataView<'uint32'>, count: GraphDataView<'uint32'>,
  overflow: GraphDataView<'uint32'>, requiredCount: GraphDataView<'uint32'>
) {
  const window = new GPUParameterBuffer(device, {
    id: 'time-window-example-parameters',
    format: 'float32',
    length: GPU_TIME_WINDOW_FILTER_PARAMETER_LENGTH,
    values: getGPUTimeWindowFilterParameterValues({start: 5, end: 15})
  });
  graph.add(new GPUTimeWindowFilter({
    timestamps, window: window.importToGraph(graph),
    output: {rows, status: {count, overflow, requiredCount}}
  }));
  return window;
}
```

Keep the returned window buffer until the compiled graph has been destroyed.

| Output port | Format and rows | Expected readback |
| --- | --- | --- |
| `rows` | `uint32`, capacity 3 | First `count[0]` source-row indices only: `[1]` |
| `status.count` | `uint32`, 1 | `[1]`, clamped to capacity |
| `status.requiredCount` | `uint32`, 1 | `[1]`, accepted count before capacity truncation |
| `status.overflow` | `uint32`, 1 | `[0]`; 1 means accepted rows were dropped |

The inclusive window accepts only timestamp 10. Spare row slots have no result meaning; use the
count rather than reading the entire capacity as accepted rows. Submit and read the buffers with
the walkthrough's lifecycle, reading overflow alongside IDs. When overflow is nonzero, resize and
rerun or propagate incompleteness. Rewrite the window buffer with the packer to move the window
without recompiling. Destroy the compiled graph before the window and other caller-owned buffers.

## Conventions

- **Graph views in, graph views out.** Inputs and outputs are `GraphDataView`, `GraphVectorView`, or
  `GraphTextureView` objects created on the target graph. Outputs are always caller-owned; scratch
  storage is graph transients that die with `compiled.destroy()`. Contributors never compile, encode,
  submit, or read back.
- **Per-frame values never recompile.** Viewports, thresholds, radii, observer positions, budgets,
  and region shapes are read from storage views, usually a `GPUParameterBuffer` that the
  application rewrites with `write()`. Lengths, capacities, grid sizes, and which optional views
  exist are compile-time topology; each prop's TSDoc says which category it belongs to.
- **Bounded results report overflow on the GPU.** Compact ID lists use `GPUCompactOutput`
  (`ids` and a `status` with `count`, `overflow` and optional `requiredCount`). `status.count` is clamped
  to `ids.length` and can be an indirect draw instance count; `status.overflow` is rewritten every
  encoding.
- **Rows and IDs.** `GPUTimeWindowFilter.output.rows` always contains zero-based source rows;
  optional `output.ids` contains caller identities from `sourceIds`. Node and transient IDs are `${id}-<step>`, so two instances in one graph need
  different `id` props.
- **One import per buffer.** Import a buffer once (for example with `graph.importGPUData()` or
  `GPUParameterBuffer.importToGraph()`) and pass the returned view to every contributor that reads
  it, so the graph tracks hazards on one logical handle.
- **Shared contracts.** Four shapes mean the same on every analysis page:
  - **Status.** Status words follow their owning output contract. Shared status bags use
    `output.status.overflow`, `output.status.converged`, and `output.status.code` for regression-style fits. Stage-specific loss (for example `status.candidateOverflow`) is a separate word from the final-output `status.overflow`, and per-model fits such as `GPUVariogramModelFit` write one status per model in a plain `output.status` vector, not a scalar bag.
    Domain-specific numeric-loss flags such as `columns[i].output.status.sumOverflow` use their documented output locations.
  - **Grid.** Raster and terrain contributors take `grid: GPURasterMetadata`, and their packers read
    the cell size from it (see [GPU Terrain conventions](./gpu-terrain/concepts.md#conventions)).
  - **Network.** Network contributors take `topology: GPUNetworkTopology` (`offsets`, `neighbors`,
    `weights`, optional `nodePositions`). Its CSR part can be passed as `GPUSpatialWeights` (see
    [GPU Network conventions](./gpu-network/concepts.md#conventions)).
  - **Polygons.** Polygons use the four-level `GPUPolygonGeometryPort`: `kind: 'polygons'`,
    `positions`, `featureOffsets`, `polygonOffsets` and `ringOffsets`. Contributors with a single
    geometry take the same fields flat, without `kind` (`GPUPolygonGeometryFields`). See
    [Shared ports](./gpu-spatial-analysis/concepts.md#shared-ports-status-and-planning).

## `GPUTimeWindowFilter`

Filter timestamped rows against an inclusive time window, producing a row mask and compact accepted source-row indices in `output.rows`. Use `rows` as gather indices. Optional `sourceIds` and `output.ids` must be supplied together: the latter maps each retained row to its opaque caller identity. Both destinations have the same capacity and must use separate buffers. For times `[0, 10]`, source IDs `[1, 0]`, and window `[0, 0]`, the result is `rows = [0]`, `ids = [1]`; gathering attributes `[10, 20]` by `rows` returns `[10]`.

Optional `output.status.candidateOverflow` is rewritten to zero: this exact filter has no
candidate stage. It is a writable output and must be disjoint from the compact rows and IDs, all other
outputs and every input, including application `sourceIds`.

The time filter, key lookup and temporal reduction capture declared input, output and nested status
bindings once at construction, including inherited getters. Later edits to those JavaScript maps
do not redirect writes. Buffer contents remain per-frame values.

::lens{scenario=gpu-dataframe.time-window-filter rung=result caption="588 of 1,500 observations fall inside 2023-06-01 to 2023-09-01 (rows 513 to 1100, CPU check)."}

The lens tags the summary scalar with `count` and the accepted source-row array with `compact`.
The first retained source row is 513; the result cardinality and compact panel length are 588.

Filters timestamped rows (instants) or time intervals (trail segments) against a per-frame window
and publishes stable compacted rows, optional caller IDs, a clamped count, and optional mask, fade weights, segment clip
fractions, per-track visible counts, and an indirect `instanceCount`. All outputs live in one
`output` prop (`GPUTimeWindowFilterOutput`: compact `rows` and `status`, optional mapped `ids`, plus `outputMask`, `fadeWeights`,
`clipFractions`, `trackVisibleCounts`, `drawInstanceCount`); `trackIds` is an input. The per-frame settings object is `GPUTimeWindowFilterSettings` (`GPUTimeWindowFilterWordSettings` for Int64 words), packed by `getGPUTimeWindowFilterParameterValues` or `getGPUTimeWindowFilterWordParameterValues`. Timestamps come in three
forms, chosen by the view format:

- `float32` relative to an application epoch, the cheapest path;
- `float32` plus `timestampsLow` (double-single, from `splitTimestamps`) for inputs that arrive as
  float64;
- exact `Int64` words: a `uint32x2` view holding each value's little-endian `(low, high)` words,
  for example Arrow `Int64` or `Timestamp` epoch milliseconds uploaded without a CPU pass
  (`getInt64TimeWords`). The window is
  then a `uint32` parameter buffer written with `getGPUTimeWindowFilterWordParameterValues`, whose start
  and end are a `number` (with a sub-unit fraction for a smooth playhead) or a `bigint`. Kernels
  order times lexicographically on the words and subtract with a borrow before converting to
  float32, so acceptance is exact for every `Int64` value, including an `INT64_MIN`/`INT64_MAX`
  "unbounded" window over pre-1970 rows, with no epoch to choose; fade and clip differences have the
  right sign for any two values and are exact below 2^24 units.
- The window is inclusive. A reversed window (`end < start`) accepts nothing, interval rows that
  span it included, and NaN timestamps are never accepted (edge-policy table
  `TIME_WINDOW_EDGE_POLICY` in the tests).

Arrow null validity is separate from the timestamp words: null can have physical words `[0,0]`,
just like a valid epoch-zero time. For nullable time use the unsorted route with
`additionalPredicates: [{kind: 'selection', mask: timeValidity}]`. With `[null,0n,10n]` and
window `[0n,0n]`, only source row 1 is eligible. Sorted mode is suitable only for an all-valid
view or a separately established contiguous valid slice with its source-row offset retained;
order validation alone cannot detect nulls.

Double-single inclusion is exact for the **encoded pairs**, which may differ from original
doubles. Request `splitTimestamps(values, {requireExact: true})` when membership must preserve
original doubles; it rejects the first row not exactly represented by finite high/low lanes.
The default retains its rounding policy. Window endpoints must be exactly encodable regardless
of the row-splitting policy. Use Int64 words for integer timestamps.

```ts
import {splitTimestamps} from '@luma-contributors/analysis/dataframe-analysis';
const exactRows = splitTimestamps([1 + 2 ** -25], {requireExact: true});
// splitTimestamps([1 + 2 ** -25 + Number.EPSILON], {requireExact: true}) throws at row 0.
```

Time-window packers validate all returned lanes before writing reusable targets. Float endpoints
require finite high/low lanes with no low-lane rounding; word endpoints require exact Int64 words
and an exactly representable float32 fraction in `[0,1)`. Positive fades must remain positive and
finite in float32; explicit zero disables them. Rejections preserve caller-supplied targets.

Word splitting takes `whole = floor(endpoint)` and rounds `endpoint - whole` to float32. Negative
fractions use that floor convention: `-0.125` is integer `-1` plus fraction `0.875`. If the rounded
fraction reaches 1, carry one into the integer and store fraction 0. The numeric integer part,
including that carry, must be a safe integer. Use `9007199254740992n` for the wide integer
endpoint; numeric `2**53` is rejected even though it is itself representable as a double.
The integer must also fit signed Int64. Exactly representable equal windows remain valid;
exactly representable reversed windows remain valid and accept no rows.

Exact reconstruction means comparing `Number(joinTimeWords(low, high)) + packedFraction` with
the **original numeric endpoint**, after fraction rounding and carry. Comparing only with
`endpoint - floor(endpoint)` is insufficient: that subtraction can lose a negative residual.
The word-window packer checks full reconstruction before writing: it rejects
`-0.125 + 2**-56`, which would otherwise encode as `-0.125`. Use `bigint` for integer endpoints
outside the safe-integer range. Both time-window packers leave a supplied target unchanged on
rejection, including failed reconstruction, invalid fades and endpoint-order loss.

### Sorted range mode

When the instants are sorted by time (the usual layout of a recorded track or event log), set
`sorted: true`. The accepted rows are then one contiguous range, found by two binary searches.
Float/high-low mode uses two parallel boundary invocations, each bounded to 32 midpoint steps,
then a constant-work count publication; Int64 mode retains its single-thread word search.
Finding the range takes O(log n), with no mask pass, scan or compaction. Publishing row IDs
dispatches one invocation per `output.rows` capacity slot, returning early past the live count;
a full-size row output therefore adds O(n) dispatch work even without masks or order validation.
The range is written to
`output.rowRange` (`[firstRow, rowCount]`, `rowCount` unclamped),
`output.drawFirstInstance` (the draw record's `firstInstance`, so the instance index is the row
index) and the usual status words and `output.drawInstanceCount`; `output.rows` receives only its
first `min(count, rows.length)` source-row indices from a kernel that returns early past the count.
An optional gather maps these rows through `sourceIds` into `output.ids`. A caller that reads only the range passes a small `rows` view. Acceptance
uses the same comparison helpers as the mask path (double-single `timestampsLow` and Int64 words
included), so the rows are exactly those the mask path accepts.

- Precondition: `timestamps` (with `timestampsLow`) is non-decreasing in row order and has no NaN. The
  default path does not scan for this. Set `validateOrder: true` with `output.orderViolations` (one
  `uint32`) to count adjacent descents on every encoding; it is an O(n) pass, as expensive as the mask
  path, so use it while developing or after an upload, not per frame.
- Instants only: `endTimestamps`, `additionalPredicates`, `trackIds`, `output.trackVisibleCounts` and
  `output.clipFractions` are refused, and views must be single packed (unchunked).
- `output.outputMask` and `output.fadeWeights` are still allowed and still cost O(n), so leave them
  out when scrubbing very large data.
- A reversed window (`end < start`) gives an empty range.

```ts
import {GPUTimeWindowFilter} from '@luma-contributors/analysis/dataframe-analysis';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
declare const compactOutput: import('@luma-contributors/analysis/dataframe-analysis').GPUTimeWindowFilterOutput;
declare const sortedInstants: import('@luma.gl/graph').GraphDataView<'float32'>;
declare const rowRange: import('@luma.gl/graph').GraphDataView<'uint32'>;
declare const drawInstanceCount: import('@luma.gl/graph').GraphDataView<'uint32'>;
declare const window: GPUParameterBuffer<'float32'>;
graph.add(new GPUTimeWindowFilter({
  sorted: true,
  timestamps: sortedInstants,
  window: window.importToGraph(graph),
  output: {...compactOutput, rowRange, drawInstanceCount}
}));
```

```ts
import {GPUTimeWindowFilter, getGPUTimeWindowFilterParameterValues} from '@luma-contributors/analysis/dataframe-analysis';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
declare const compactOutput: import('@luma-contributors/analysis/dataframe-analysis').GPUTimeWindowFilterOutput;
declare const drawCountView: import('@luma.gl/graph').GraphDataView<'uint32'>;
declare const segmentStarts: import('@luma.gl/graph').GraphDataView<'float32'>;
declare const segmentEnds: import('@luma.gl/graph').GraphDataView<'float32'>;
declare const fadeWeights: import('@luma.gl/graph').GraphDataView<'float32'>;
declare const clipFractions: import('@luma.gl/graph').GraphDataView<'float32x2'>;
declare const now: number;
declare const trailLength: number;
declare const window: GPUParameterBuffer<'float32'>;
graph.add(new GPUTimeWindowFilter({
  timestamps: segmentStarts, endTimestamps: segmentEnds,
  window: window.importToGraph(graph),
  // `output` is a `GPUTimeWindowFilterOutput`: the compact list plus the optional views.
  output: {
    ...compactOutput, // rows, status: {count, overflow, optional requiredCount}
    fadeWeights, clipFractions,
    drawInstanceCount: drawCountView
  }
}));
window.write(getGPUTimeWindowFilterParameterValues({start: now - trailLength, end: now, startFadeDuration: trailLength}));
```

```ts
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
import {type GraphDataView} from '@luma.gl/graph';
declare const departureValues: BigInt64Array;
declare const uploadedDepartureWords: import('@luma.gl/graph/gpu-data').GPUData<'uint32x2'>;
declare const output: import('@luma-contributors/analysis/dataframe-analysis').GPUTimeWindowFilterOutput;
declare const playheadMs: number;
import type {GPUData} from '@luma.gl/graph/gpu-data';
import {
  GPUTimeWindowFilter,
  GPU_TIME_WINDOW_FILTER_WORD_PARAMETER_LENGTH,
  getGPUTimeWindowFilterWordParameterValues,
  getInt64TimeWords
} from '@luma-contributors/analysis/dataframe-analysis';

const window = new GPUParameterBuffer(device, {
  id: 'window', format: 'uint32', length: GPU_TIME_WINDOW_FILTER_WORD_PARAMETER_LENGTH
});
// `departureValues` is the BigInt64Array of an Arrow Int64 or Timestamp column. Upload its words
// as a `uint32x2` GPUData (one row per timestamp) with the luma.gl Arrow or GPUData APIs.
const departureWords: Uint32Array = getInt64TimeWords(departureValues);
const departures: GPUData<'uint32x2'> = uploadedDepartureWords; // from departureWords
graph.add(new GPUTimeWindowFilter({
  timestamps: graph.importGPUData('departures', departures),
  window: window.importToGraph(graph), output
}));
window.write(getGPUTimeWindowFilterWordParameterValues({start: playheadMs - 3_600_000, end: playheadMs}));
```

#### How it computes

{/* lens:GPUTimeWindowFilter */}

::lens{scenario=gpu-dataframe.time-window-filter rung=datum bind=page datum=513 caption="Observation 513, 2023-06-01 10:57, is the first row inside the window."}
::lens{scenario=gpu-dataframe.time-window-filter rung=formula bind=page datum=513}
::lens{scenario=gpu-dataframe.time-window-filter rung=kernel caption="Look at the classify kernel: the window start and end are read from the parameter buffer, not baked into WGSL."}
::lens{scenario=gpu-dataframe.time-window-filter rung=graph caption="One classify node, the visibility workflow that compacts source rows, and a publish node that clamps the count."}

{/* parameters: GPUTimeWindowFilter generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPUTimeWindowFilter` parameters

Import from `@luma-contributors/analysis/dataframe-analysis`: `GPUTimeWindowFilter`.

- Packer: `getGPUTimeWindowFilterParameterValues(settings)`, writes `GPU_TIME_WINDOW_FILTER_PARAMETER_LENGTH` (8) `float32` values.
- Packer: `getGPUTimeWindowFilterWordParameterValues(settings)`, writes `GPU_TIME_WINDOW_FILTER_WORD_PARAMETER_LENGTH` (8) `uint32` values.

Settings of `getGPUTimeWindowFilterParameterValues`:

| Field | Type | Description |
|---|---|---|
| `start` | `number` | Inclusive window start. |
| `end` | `number` | Inclusive window end. A window with `end < start` accepts nothing. |
| `startFadeDuration` (optional) | `number` | Fade weight ramps 0 to 1 over `[start, start + startFadeDuration]`. 0 (default) disables the ramp. |
| `endFadeDuration` (optional) | `number` | Fade weight ramps 1 to 0 over `[end - endFadeDuration, end]`. 0 (default) disables the ramp. |

Settings of `getGPUTimeWindowFilterWordParameterValues`:

| Field | Type | Description |
|---|---|---|
| `start` | `number \| bigint` | Inclusive window start. |
| `end` | `number \| bigint` | Inclusive window end. A window with `end < start` accepts nothing. |
| `startFadeDuration` (optional) | `number` | Fade weight ramps 0 to 1 over `[start, start + startFadeDuration]`. 0 (default) disables the ramp. |
| `endFadeDuration` (optional) | `number` | Fade weight ramps 1 to 0 over `[end - endFadeDuration, end]`. 0 (default) disables the ramp. |

{/* /parameters */}

## `GPUColumnQuantiles`, `GPUClassBreaks`, `GPUColorScale`, and `GPUBivariateClassification`

Choropleth classification on the GPU, every frame, with no readback: exact quantiles and
percentile filters, class breaks, and colour scales that write a packed `rgba8` column a deck.gl
layer can bind as an attribute. Each output can feed the next contributor as a graph view, so a brush
or filter change recolours the map without a CPU round trip.

```ts
import {GPUColumnQuantiles, GPUClassBreaks, GPUColorScale, getGPUColumnQuantilesParameterValues, getGPUClassBreaksParameterValues, getGPUColorScaleParameterValues} from '@luma-contributors/analysis/dataframe-analysis';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
declare const values: import('@luma.gl/graph').GraphDataView<'float32'>;
declare const mask: import('@luma.gl/graph').GraphDataView<'uint32'>;
declare const quantileParameters: GPUParameterBuffer<'float32'>;
declare const quantiles: import('@luma.gl/graph').GraphDataView<'float32'>;
declare const validCount: import('@luma.gl/graph').GraphDataView<'uint32'>;
declare const outputMask: import('@luma.gl/graph').GraphDataView<'uint32'>;
declare const filterBounds: import('@luma.gl/graph').GraphDataView<'float32'>;
declare const breakParameters: GPUParameterBuffer<'float32'>;
declare const breaks: import('@luma.gl/graph').GraphDataView<'float32'>;
declare const classCount: import('@luma.gl/graph').GraphDataView<'uint32'>;
declare const classCounts: import('@luma.gl/graph').GraphDataView<'uint32'>;
declare const palette: import('@luma.gl/graph').GraphDataView<'uint32'>;
declare const scaleParameters: GPUParameterBuffer<'float32'>;
declare const colors: import('@luma.gl/graph').GraphDataView<'uint32'>;
declare const classIndices: import('@luma.gl/graph').GraphDataView<'uint32'>;
graph.add(new GPUColumnQuantiles({
  values, mask, quantileCount: 3,
  parameters: quantileParameters.importToGraph(graph),
  output: {quantiles, validCount, outputMask, filterBounds} // outputMask may be a transient
}));
graph.add(new GPUClassBreaks({
  values, mask: outputMask, maximumClassCount: 9,
  methods: ['quantile', 'natural-breaks', 'head-tail'], // omitted: capacity and layout compatible methods
  parameters: breakParameters.importToGraph(graph),
  output: {breaks, classCount, classCounts}
}));
graph.add(new GPUColorScale({
  values, mask: outputMask, domain: breaks, classCount, palette,
  parameters: scaleParameters.importToGraph(graph),
  maximumDomainCount: 10, maximumPaletteCount: 9,
  output: {colors, classIndices, classCounts}
}));
quantileParameters.write(getGPUColumnQuantilesParameterValues({quantiles: [0.25, 0.5, 0.75], filterRange: [0.05, 0.95]}));
breakParameters.write(getGPUClassBreaksParameterValues({method: 'natural-breaks', classCount: 7}, 9));
scaleParameters.write(getGPUColorScaleParameterValues({scale: 'threshold', paletteCount: 9}));
// Any of these writes takes effect on the next encoding, with no recompile.
```

Threshold settings can omit the numeric `domainCount` when the color scale reads its class count from
the resident `classCount` view. With a fixed domain, pass the number of active domain entries as
`domainCount`.

When classifying a density raster, use its grid-sized live prefix rather than the allocation's
spare rows. `makeGPURasterBand(...).storage.values` borrows that prefix and preserves the source
byte offset; it does not copy or reset the view to offset zero.

```ts
import {GPUClassBreaks} from '@luma-contributors/analysis/dataframe-analysis';
import {makeGPURasterBand, makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';
import {getViewBindingRange, type GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import type {Device} from '@luma.gl/core';
declare const graph: GPUCommandGraph;
declare const device: Device;
// Five readable allocation rows [2,4,6,8,1000], possibly at a nonzero byteOffset.
declare const densityValues: GraphDataView<'float32'>;
declare const parameters: GraphDataView<'float32'>;
declare const output: import('@luma-contributors/analysis/dataframe-analysis').GPUClassBreaksOutput;
const grid = makeGPURasterMetadata({width: 2, height: 2});
const values = makeGPURasterBand(densityValues, grid).storage.values; // length 4
const plan = GPUClassBreaks.getExecutionPlan({
  rowCount: values.length, maximumClassCount: 2, methods: ['equal-interval'],
  bindings: [{name: 'values.0', byteLength: getViewBindingRange(values).size,
    bufferByteLength: values.buffer.byteLength}]
}, device.limits);
if (!plan.feasible) throw new Error('Class-break route exceeds device or command limits');
graph.add(new GPUClassBreaks({
  values, maximumClassCount: 2, methods: ['equal-interval'], parameters, output
}));
// Pack parameters with {method: 'equal-interval', classCount: 2} and capacity 2.
// Four live values: edges [2,5,8], counts [2,2]. Explicitly using five rows instead:
// edges [2,501,1000], counts [4,1]. Inner edges are (minimum+maximum)/2.
```

`rowCount` is the live population. Physical `bindings` overrides describe actual aligned ranges
and backing sizes, merged with required inferred bindings: names are `values.0`, `mask.0`,
`parameters`, `output.breaks`, `output.classCount`, `output.classCounts`. Alignment prefixes count
in `byteLength`; backing size is `bufferByteLength`. Supplying one override never disables checks
on the other bindings. Inspect `plan.constraints` for failed admission, including the estimated
command-node budget; raw construction and encoding also preflight the actual views. For example,
with 18 classes and compiled approximate natural breaks (`methods: ['natural-breaks']`,
`naturalBreaksMode: 'approximate'`), 659 nonempty chunks estimate 32,725 command nodes;
660 estimate 32,771 and reject
against the 32,768-node budget. A logical row limit alone does not certify an executable graph.
For equal-interval-only methods, the same 660 one-row chunks estimate 5,384 nodes and are feasible.

- Class rule shared by all four contributors: breaks are edges `e[0..k]` (`e[0]` the smallest and
  `e[k]` the largest finite value). A value is in class `i` when `i` inner edges `e[1..k-1]` are at
  or below it (d3 `scaleThreshold`/`bisectRight`), compared on numeric order-preserving `u32` keys
  that treat `-0` and `+0` as equal. Both zeros enter the upper class at a zero inner edge;
  raw keys still preserve their bits for extrema. Values outside `[e[0], e[k]]` clamp to the end classes. NaN and masked
  rows are skipped (class `0xffffffff`, the no-data colour). Colours are `r | g << 8 | b << 16 | a << 24`.
- `GPUColumnQuantiles`: exact order statistics by four 8-bit radix-select passes over the ordered
  keys (no sort): up to 64 per-frame probabilities, `'lower' | 'higher' | 'nearest' | 'linear' |
  'midpoint'` (numpy names; `'linear'` is d3 and R-7). The rank is `fround(fround(n - 1) * p)` and
  every interpolation step is rounded separately (no fused multiply-add), so the GPU rank policy is reproducible; CPU interpolation uses wider arithmetic. `filterRange` gives deck.gl/kepler
  lower/upper percentile filtering: bounds `x[floor(n * lower)]` and `x[ceil(n * upper) - 1]`, and a
  `outputMask` of the rows inside them. Float ranks or any filter output admit at most 2^24 total
  rows; rational probabilities without filter outputs admit at most 2^31 - 1 total rows, subject
  to per-binding limits below. About 1.8 ms at 1M rows and 3.5 ms at
  4M rows for 9 quantiles plus the filter (Apple M-series, headless Chromium).
  - Which rank is exact. The default rank `fround(fround(n - 1) * p)` is a float32 product and is
    not exact: its fraction is lost from rank 2^23 (16,777,216 rows at `p = 0.95`: `linear`
    returns `x[lo]`, numpy interpolates by 0.25), and the float32 `p` can put `floor(h)` one rank
    from numpy's at any size (101 rows at `p = 0.59`). The optional `rationalProbabilities` view
    (`getGPUColumnQuantilesRationalProbabilityValues([[19, 20], [1, 2]])`, uint32 pairs
    `numerator, denominator`) replaces the float32 probabilities: the rank
    `floor((n - 1) numerator / denominator)` and its remainder come from a 64-bit integer product
    and division, exact for every row count, and the `linear` fraction is the correctly rounded
    float32 of `remainder / denominator`. It agrees with `numpy.quantile` 2.5.4 for `linear`
    everywhere tested, and for the other methods except where numpy's own float64 product
    `(n - 1) * q` lands just below or just above an integer the exact product equals: just below,
    `lower` and `midpoint` are one rank off (101 rows, `q = 0.29`: numpy `lower` returns `x[28]`,
    the exact rank is 29); just above, `higher` and `midpoint` are (101 rows, `q = 0.55`: numpy
    `higher` returns `x[56]`, the exact rank is 55). With it the column may
    exceed 2^24 rows (at most 2^31 - 1 rows in total; one packed `values` view is limited by one storage binding, 2^25 rows at the default limit, and chunked `values` span several bindings) when no filter
    output is requested; the filter ranks stay float32. `GPUClassBreaks` and `GPUGroupStatistics`
    use the float32 rank only. Fixture and generator:
    `modules/analysis/test/oracles/dataframe-quantile-rational-rank.py`.
  - Row admission is independent of chunk capacity: rational unfiltered total <=2,147,483,647;
    float-rank or any filter output total <=16,777,216. A 128 MiB float32 binding holds
    33,554,432 rows. Aligned 20M/empty/20M chunks demonstrate 40M rational unfiltered rows.
    CPU packed-float ranks use the same rounded products through 16,777,216 valid rows and
    wider products above that GPU limit. For `p=.7`, lower interpolation on rows 0..10 is 7;
    filtering rows 0..9 by `[.7,1]` gives bounds `[7,9]` and excludes row 6.
    Invalid original probabilities such as `-1e-50` and `1+Number.EPSILON` remain invalid and
    yield NaN instead of rounding to valid endpoints.

- CPU slot counts above 64 use the public backend-neutral `getColumnQuantilesParameterValues`
  and `getColumnQuantilesRationalProbabilityValues`. The existing `getGPU...` packers retain
  the 64-slot limit. Do not hand-pack a numeric interpolation header to bypass it.

The CPU packers and operation are available from the dataframe entry:

```ts
import {
  CPUColumnQuantilesOperation, getColumnQuantilesParameterValues,
  getColumnQuantilesRationalProbabilityValues
} from '@luma-contributors/analysis/dataframe-analysis';
const floatParameters = getColumnQuantilesParameterValues({quantiles: Array(65).fill(0.5)});
const ratios = getColumnQuantilesRationalProbabilityValues(
  Array.from({length: 65}, () => [1, 2] as const)
);
const rows = (values: Float32Array | Uint32Array) => ({
  format: values instanceof Float32Array ? 'float32' as const : 'uint32' as const,
  length: values.length, chunks: [{values, length: values.length}], pack: () => values
});
for (const rationalProbabilities of [undefined, ratios]) {
  // NaN float slots ensure the rational route cannot pass through float fallback.
  const parameters = rationalProbabilities
    ? getColumnQuantilesParameterValues({quantiles: Array(65).fill(NaN)})
    : floatParameters;
  const operation = new CPUColumnQuantilesOperation({
    values: 'values', parameters: 'parameters', rowCount: 2, quantileCount: 65,
    rationalProbabilities: rationalProbabilities ? 'ratios' : undefined,
    output: {quantiles: 'quantiles', validCount: 'validCount'}
  });
  const result = await operation.execute({inputs: {
    values: rows(Float32Array.of(0, 2)), parameters: rows(parameters),
    ...(rationalProbabilities ? {rationalProbabilities: rows(rationalProbabilities)} : {})
  }});
  // result['quantiles'] is 65 ones: each median is (0+2)/2=1; validCount is [2].
}
```

- One order-statistic policy for `GPUColumnQuantiles`, the `GPUClassBreaks` quantile and box-plot
  methods, and the `GPUGroupStatistics` medians and percentiles: the sample is the finite unmasked
  values (NaN and ±Infinity are missing), `-0` equals `+0`, and a probability that is NaN or outside
  `[0, 1]`, or an empty sample, gives NaN. The result is `numpy.quantile(x[numpy.isfinite(x)], p)`;
  numpy itself keeps infinities and raises on an invalid `p`. When `x[high] - x[low]` overflows f32
  the convex form `x[low] (1 - t) + x[high] t` keeps the result finite.
- Custom class edges must remain nondecreasing after float32 packing; repeated edges are allowed.
  Signed zeros compare equal for class membership and maximum-gap distinctness. Ordered opacity
  endpoints that collapse to one float32 value are rejected; deliberately equal endpoints retain
  the full-opacity policy. Natural approximation uses per-range scales and publishes input-unit
  edges even across tiny/extreme ranges.
- Omitted `methods` resolves capacity and layout compatible methods. Box plot is excluded below
  six classes; approximate natural breaks is excluded if maximumClassCount exceeds its bin count.
  Explicit incompatible selections reject. The raw chunked contributor omits maximum-breaks and
  packed-only exact/monotone natural breaks from its defaults. CPU operations pack their source
  chunks and support maximum-breaks with either omitted `methods` or explicit
  `methods: ['maximum-breaks']`; select `method: 'maximum-breaks'` in the per-frame parameters too.
  For chunks `[1,2]`, `[7,8]` and two classes, the largest gap is 5, so the edges are
  `[1,4.5,8]`, class count `[2]` and class counts `[2,2]`. GPU operation lowering still
  requires one packed values chunk; the raw GPU contributor supports chunk-compatible methods.
- Natural row limits apply only when natural breaks is enabled. Equal-interval-only declarations
  ignore unused natural route row/storage limits; option values and storage/mode compatibility
  are still validated. Approximate natural breaks requires maximumClassCount <= naturalBreaksBinCount.
  The CPU reference computes exact float64 Fisher–Jenks, distinct from the default GPU histogram
  refinement: for `[174,15,79,177,157,114,36,56,179,166,104,100000]`, four classes give CPU inner
  edges `[79,157,100000]` versus host approximation mirror `[104,157,100000]` (not a GPU measurement).
  `provenance.naturalBreaksMode`, `provenance.naturalBreaksAlgorithm` and `provenance.methods`
  record the route: CPU algorithm `'exact-fisher-jenks-float64'`; GPU approximation
  `'histogram-refinement'`, exact `'exact-fisher-jenks-normalized-float32'`, or monotone
  `'monotone-quantized-dynamic-programming'`.
- The pure function and CPUClassBreaksOperation both accept empty input. A fitted two-class
  method returns NaN edges, classCount 0 and zero counts; custom edges `[-5,0,7]` retain those
  edges, classCount 2 and counts `[0,0]`. GPU forms reject a resolved zero-row view, so guard
  only the GPU route. A nonempty view with no eligible finite sample remains admissible.
- `GPUClassBreaks`: per-frame `method` and `classCount` among the compiled `methods`:
  `'equal-interval'`, `'quantile'` (exact linear quantiles at `i / k`), `'standard-deviation'`
  (classes of `interval` sample standard deviations centred on the mean, as mapclassify `StdMean`
  and R classInt `'sd'`), `'head-tail'` (Jiang's
  breaks with a per-frame head ratio, default 0.4; 1 gives mapclassify's rule), `'box-plot'`
  (six classes, per-frame hinge), `'maximum-breaks'` (float32 midpoints of the `k - 1` largest gaps
  between distinct sorted values; if a midpoint rounds to the lower observation, use the upper
  observation so lower-edge membership separates them), `'natural-breaks'` (Fisher-Jenks, see below), and
  `'custom'`. Head/tail and maximum breaks may return fewer classes; `classCount` reports `k`
  (0 with no finite values, 1 when they are all equal). Infinities are missing for every edge, so
  `e[0]` and `e[k]` bound the inner edges. Kernels of unselected methods return at
  once, except the two radix sorts of `'maximum-breaks'`, which run every frame when compiled.
  Natural-breaks layers and refinement rounds and head/tail rounds read their index from a GPU
  state word and are dispatched indirectly, so the pipeline count does not grow with
  `maximumClassCount` and finished rounds launch no workgroups. Moments use a fixed-order Chan merge, so every output is bitwise
  reproducible. About 3 to 4 ms per frame at 1M rows with six methods compiled. Every closed-form
  method is checked against mapclassify 2.11 and natural breaks against the exact Fisher-Jenks
  partition (an f64 dynamic program, agreeing with jenkspy 0.4.1) by the committed generator
  `test/oracles/dataframe-class-breaks.py`.

  For adjacent float32 observations `[1,1+2^-23]`, maximum breaks publishes edges
  `[1,1.0000001192092896,1.0000001192092896]` and counts `[1,1]`. Use lower-edge closure
  (`'left'`) when applying these edges; closure can matter when a stored edge equals an
  observation. Class-break comparisons treat -0 and +0 as equal.
  - `naturalBreaksMode: 'exact'` explicitly selects a full Fisher–Jenks dynamic program for
    `'natural-breaks'`; the default `'approximate'` retains the histogram/refinement path below.
    Exact mode accepts at most **8192 source rows** (including masked/missing rows), with the
    existing capacity of 64 classes. Construction rejects larger columns; it never falls back
    to an approximate method. `naturalBreaksExactStorage` defaults to `'auto'`: up to 1024 rows
    it materializes interval costs; larger inputs recompute the same backward Welford recurrence
    during each DP layer. Explicit `'materialized'` retains the 1024-row cap; `'recompute'`
    can also be selected for small inputs to minimize storage.
    A radix sort places selected finite values first. Equal numeric values (including signed
    zeros) cannot be split; the produced count is `min(requested, distinct finite values)`.
    Each inner edge is the first value of the upper class, included in that class. This matches
    GeoDa's `< break` assignment contract, pinned to
    [GeoDa 37510860, CatClassification.cpp](https://github.com/GeoDaCenter/geoda/blob/37510860bd9f7b8ae145c15759f7abc5ab217cf3/Explore/CatClassification.cpp#L806).
    GeoDa's randomized natural-break search itself is not the exact reference; the independent
    reference is [jenkspy 0.4.1](https://github.com/mthh/jenkspy/blob/3ff493f8b18af5c0e165d9963ba35ad7dc8f5043/jenkspy/src/_jenks.c).
    The objective is total within-class squared deviation. Exact means the global optimum of
    the documented GPU arithmetic recurrence: divide values by the largest power of two at or
    below `max(abs(min), abs(max))`
    (the magnitude itself for subnormal scales, one for zero), subtract each interval's scaled last value, then
    compute each interval's centered sum of squares by backward Welford updates in f32,
    then enumerate every eligible split using f32 score addition. Division decomposes both operands with `frexp`,
    divides bounded mantissas, then applies their exponent difference with `ldexp`; no extreme reciprocal is formed. Arithmetic ties retain the
    lowest last split, recursively within the prefix. This is not an f64 or real-arithmetic
    guarantee: normalization can underflow tiny values and close objectives can round together;
    device multiply-add contraction can also affect close partitions. Work is `O(k n²)` and
    scratch is `O(n² + k n)` when materialized, or `O(k n)` when recomputed (two score
    rows and the partition backpointers). Recomputation eliminates the quadratic allocation
    but repeats the Welford arithmetic; it does not promise lower latency. The 8192-row cap
    is a bounded-work contract, not a measured interactive-performance guarantee. Scratch
    bindings are checked against the actual device limit. There is no CPU solve, submission or readback. NaNs/masked rows are omitted; infinities are
    omitted from the objective and counted in the end classes, as in the approximate mode.
  - `naturalBreaksMode: 'monotone'` selects a separate **approximate** full-data dynamic program,
    for up to **262144 source rows** and 64 classes. It sorts the finite selected values and
    uses full distinct source-value boundaries with monotone pruning, avoiding the histogram mode's restricted
    initial boundaries and subsequent single-cut local search. It is useful when a full-column
    partition search is wanted and exhaustive f32 recurrence optimization is too expensive.
    Normalized positions are quantized onto `[0, 1048575]` (20 bits); exact 64-bit integer
    prefix moments avoid raw floating-point prefix cancellation. Interval costs and DP score
    additions are still f32, and divide-and-conquer pruning assumes mathematical monotonicity.
    **Neither real-arithmetic optimality nor parity with exact mode is guaranteed.** Widely
    separated outliers can collapse small distinctions under quantization; this is not a universal
    accuracy improvement over the histogram/refinement mode. Edges remain actual source values,
    equal numeric values (including signed zero) are indivisible, and arithmetic ties prefer the
    lowest split. Each breadth-first search level distributes endpoints and candidate scans across
    workgroups; pipelines are reused and inactive levels dispatch no work. Normal search work is
    `O(k n log n)` with `O(k n)` storage. If rounding leaves a feasible endpoint without a valid
    candidate in its pruned interval, that endpoint falls back to exhaustive search to preserve a
    valid partition, so worst-case work remains `O(k n²)`. This fallback does not certify
    optimality. Device binding limits are checked before scratch allocation. No GPU-to-CPU readback
    or CPU solve is performed.
  - With the default approximate mode, `'natural-breaks'` seeks the partition into `k` classes of
    least total within-class squared
    deviation. A coarse dynamic program over `naturalBreaksBinCount` equal-width bins of
    `[min, max]` (2 to 2048, default 2048; when this method is compiled, the bins also cap the class count, so `maximumClassCount` above `naturalBreaksBinCount` is rejected at construction in this mode) sums count, position and squared position of each bin as
    exact 64-bit integers, so the class cost is evaluated without the cancellation of `S2 - S1^2 / n`
    in f32 and without a bin-by-bin cost table. A layer of the dynamic program is one workgroup that
    exploits the monotone split points of the cost (the quadrangle inequality) over `log2(bins)`
    barrier-separated levels: `O(k B log B)` work in all, against `O(k B^2)` before. The bins cannot
    see structure finer than `range / bins`, so the breaks are then refined on the rows, and no
    longer snap to bin edges: red-black coordinate descent in which each break, with its two
    neighbours fixed, takes the best split of the values in its two adjacent classes (odd breaks and
    then even breaks, so a pass never moves two interacting breaks and the cost only falls). The
    first window of a break covers the data extent of its two classes, so a heavy outlier cannot hide
    the structure, and the window then shrinks around the chosen cell boundary until a cell is
    below an f32 ulp. If empty classes show that the bins collapsed the most populated class (one
    extreme outlier puts the whole bulk into one bin), the coarse stage runs again over the range of
    that class with the classes the empty ones would have held. Rounds end once no cut moved with
    every break zoomed twice, at most eight; a column usually settles in two rounds, that is five
    passes over the rows (one for the class extents, two per round). Every inner edge `e[i]` is an
    actual data value, the smallest value of class `i`; mapclassify and jenkspy report the
    largest value of each class (`b[i-1]`), so `e[i]` is the smallest data value above `b[i-1]`
    and the classes are the same. A class count above the number of distinct values (which
    mapclassify and jenkspy refuse) yields one class per distinct value, like `'maximum-breaks'`;
    the produced count is `classCount`.
  - Accuracy: on the 34 reference cases (10 columns from lognormal and cluster mixtures to heavy
    ties, an extreme outlier of 1e6 times the spread, an outlier pair and a Pareto tail, `k` from 2
    to 10) 32 equal the exact partition, bit for bit in the edges. The other two stop in a
    neighbouring local optimum of the coordinate descent (a goodness of variance fit within 2e-6
    of the optimum; one of them is the partition mapclassify 2.11 itself returns, which is not
    optimal). On random well-scaled columns about 1 in 300 does. On 180 random heavy-tailed
    columns (Pareto tails, outliers, clusters beside an outlier) 179 were exact and the other was
    within a fit of 4e-8. Structure narrower than a bin that is not the most populated class (for
    example several clusters in one bin beside an outlier, when a lighter class also collapses)
    keeps the coarse allocation and is refined locally only.
    The previous implementation snapped breaks to bin edges (an error up to `range / 1024`) and
    cost `O(k B^2)` for the cost table and the layers.
- `GPUColorScale`: `linear`, `sqrt`, `pow`, `log` (kepler's `logFloor` for `v <= 0`), `symlog`,
  `quantize`, `threshold`/`quantile` (binary search over the active edges), and `ordinal` (a
  `uint32` code column). Scale, domain, palette, `clamp`, and `'step'`/`'linear'` interpolation are
  per frame. A multi-stop domain with as many stops as palette entries is piecewise linear, like a
  d3 multi-stop scale. A `classCount` view (for example `GPUClassBreaks.classCount`) holds a
  class count `k`, and the domain then has `k + 1` edges. `'step'` colours are exact; `'linear'`
  blends round half up (within 1 per channel of a double-precision reference).
- `GPUBivariateClassification`: two columns and two edge views give `classId = yClass *
  classCountX + xClass` (up to 16 per axis), a colour from an `n x n` palette, per-class counts,
  and optional value-by-alpha (`alphaValues` mapped through a per-frame `[low, high]` to
  `[minimumAlpha, 1]` times the palette alpha).
- Optional outputs are independent: `GPUColorScale` accepts any nonempty subset of colors,
  class indices and class counts; `GPUBivariateClassification` accepts any nonempty subset of
  colors, joint class IDs and joint counts. Counts exclude masked/NaN rows. Every encoding clears
  the first `maximumPaletteCount` words for `GPUColorScale` and the first `maximumClassCount ** 2`
  words for `GPUBivariateClassification`, including unused class slots within those prefixes.
  Larger count views preserve their trailing caller-owned words. Omitting colors still produces
  requested IDs/counts.
  Both constructors snapshot the selected outputs and other declared props; later mutation of the
  caller's objects does not remove count writers or change the validated topology. GPU buffer
  contents remain per-frame.
  `GPUScalarScale` always writes mapped values and may also write validity. `GPUClassBreaks`
  always writes breaks/classCount and may also write classCounts. `GPUColumnQuantiles` always
  writes quantiles/validCount; outputMask and filterBounds can each be requested independently.
- Integer columns: `GPUClassBreaks.values` accepts `uint32` and `sint32` in addition to `float32`. One
  internal pass converts the column to `float32` (exact to 2^24) and every method behaves as for that
  float column. `GPUColorScale.values` accepts `sint32` (always numeric) and `uint32` with
  `integerValues: 'numeric'`, which reads the integers as numbers and applies the `scale` parameter; the
  default `'ordinal'` keeps `uint32` as category codes. The value format is topology (fixed at compile
  time). This is what lets a count column (for example dense `GPUGroupStatistics` `counts`) feed a
  choropleth without a float copy.
- `GPUClassBreaksOperation` and `CPUClassBreaksOperation` infer the values format and row count
  from the input vector. Optional `valuesFormat` and `rowCount` assert that inferred schema;
  both forms reject a mismatch. CPUClassBreaksOperation accepts genuinely empty input; GPU
  lowering requires at least one input row, including when `rowCount` is omitted. The same props object selects either execution form. Constructors snapshot the
  schema, compiled methods and output port names: mutating that caller object later does not
  change an existing operation. Declared fields include inherited and non-enumerable getters,
  including `mask` and output port names, and are captured once at construction.
  `naturalBreaksExactStorage` requires `naturalBreaksMode: 'exact'` on both forms even when
  `methods` excludes natural breaks; omit the storage option otherwise. This dependency is
  independent of device limits. Input vector contents and packed parameter values remain per-frame.
- Per-frame settings are the CPU objects `GPUColumnQuantilesSettings`, `GPUClassBreaksSettings`,
  `GPUColorScaleSettings`, and `GPUBivariateClassificationSettings`, packed by the matching
  `getGPU<Class>ParameterValues(settings, target?)`.
- `GPUColumnQuantilesOperation` and `CPUColumnQuantilesOperation` require a positive integer
  `rowCount` equal to the resolved values length, even when `outputMask` is omitted. Both forms
  snapshot the caller's props and output names at construction; later mutation does not change
  the declared population or optional output selection.
- `GPUColumnQuantiles`, `GPUClassBreaks`, `GPUColorScale` and `GPUBivariateClassification` accept
  packed or chunked row inputs and outputs with identical ordered chunk lengths. Controls,
  domains, palettes, quantiles, breaks and per-class counts remain packed; reductions span all chunks.
  `GPUClassBreaks` requires packed values and mask when its compiled `methods` includes
  `maximum-breaks`, or `natural-breaks` with `naturalBreaksMode: 'exact'` or `'monotone'`, because
  those methods sort all rows. The raw contributor's chunked defaults omit these incompatible
  methods; packed defaults include them when capacity and natural settings permit. Explicit
  incompatible methods reject. GPU operation lowering still requires packed values.

### Chunking and device limits by route

A logical total is distinct from a physical binding size. “Chunkable” below means ordered packed chunks, not strided/interleaved rows. Unless an operator explicitly aligns different partitions, row inputs/outputs share the source chunk lengths. Controls, compact results, grouped results and status words remain single packed bindings where stated. Every actual aligned binding must fit maxStorageBufferBindingSize and its backing allocation must fit maxBufferSize; offset/alignment prefixes reduce capacity. At 128 MiB and zero alignment prefix: a 4-byte row fits 33,554,432 entries, an 8-byte row 16,777,216, a 12-byte row 11,184,810, and a 16-byte row 8,388,608. These are per-binding figures, not universal logical-total caps. Shared graph-chunk total is 4,294,967,295 rows; the command-node budget can reject excessive chunk counts sooner.

| Producer and route | Chunkable source-row ports | Single-binding controls and outputs | Global/order scratch and logical-versus-binding limit |
|---|---|---|---|
| GPUTimeWindowFilter, unsorted float/high-low or Int64 | timestamps, timestampsLow, endTimestamps, endTimestampsLow; selection predicates/sourceIds/trackIds; outputMask/fadeWeights/clipFractions | window, compact output.rows/output.ids, status, trackVisibleCounts and draw counts | Compaction/visibility scratch is max(totalRows,2) uint32 words in one binding; default binding therefore bounds total to 33,554,432 before other bindings/outputs. Logical rows <=4,294,967,295. Null words require an explicit selection validity mask. |
| GPUTimeWindowFilter, sorted float/high-low or Int64 | None: timestamps/low/sourceIds must be packed | All row/compact outputs, window, range, order diagnostics, draw controls | Range scratch has two words, not all rows. Float mode searches lower/upper bounds in two parallel invocations with at most 32 midpoint steps each, then publishes the count in a separate constant-work pass; word mode retains its original serial search. Input binding caps total packed float rows at 33,554,432 or word rows at 16,777,216 by default. Only all-valid instants; no predicates, trackIds or intervals. A proved valid slice must retain its source offset explicitly. |
| GPUTimeWordsRelative | timestamps, optional mask, output | Origin/context are captured host options; generated parameter words are packed | No full-column order scratch. Logical total <=4,294,967,295; each uint32x2 timestamp chunk <=16,777,216 at default binding. Zero mask writes NaN. |
| GPUCalendarBuckets | timestamps, mask, utcOffsets and every row-calendar output | parameters; timeZone.transitions/offsets; hourWeekdayCounts (168 uint32 cells) | Per-chunk row decode; matrix aggregates across chunks. No full-column order scratch. Logical total <=4,294,967,295; timestamp chunks have the default 16,777,216-row 8-byte limit. Padded packed outputs use their source-row prefix. |
| GPUForestInference, row-major / feature-major | features, mask, predictions/validity; scores use outputCount scalars per row and leafIndices use treeCount scalars per row, with identical source-row boundaries | nodes, trees, leafValues, parameters; model shape host options | Work <=5,000,000,000 per chunk: chunkRows*treeCount*(maximumDepth+outputCount+1). Packed mode applies that bound to the entire input. Total rowCount<=4,294,967,295; featureCount<=256, trees<=1,024, depth<=64, outputs<=32, nodes<=1,048,576. Each full scalar chunk and reusable per-batch scratch independently fits a binding. No global row sort. |
| GPUForestInference, raster-stack | mask and row outputs as above; packed raster band values/validity are borrowed in slices at those output row boundaries | Original raster band descriptors stay packed; model buffers/parameters stay packed | Same per-chunk work limit and total rowCount. At most 16 raster bands; complete feature vectors are built in reusable batch scratch. Bound band slices and rowCount*featureCount batch elements must fit the device; output chunking does not make the supplied raster band itself a GraphVectorView. |
| GPUPeriodIndex, fixed | timestamps, mask, output.periods | status.invalidCount; fixed origin/width/count are captured host options | No global row sort. Logical total <=4,294,967,295, each word chunk <=16,777,216 at default binding. Period count <=4,294,967,293. |
| GPUPeriodIndex, calendar | columns.year/month/dayOfMonth/hour as required by unit, mask, output.periods | status.invalidCount; calendar origin/step/count host options | No global row sort. Logical total <=4,294,967,295; scalar chunk binding limits apply. Period count <=4,294,967,293. Origins use valid Gregorian civil dates. |
| GPUCellPeriodKeys | cells, periods, mask, output.keys | status.invalidCount; fill/layout/periodCount host options | No global order scratch. Logical total <=4,294,967,295; cell/key chunks are 8 bytes per row (16,777,216 at default binding). periodCount <=4,294,967,293. `layout` is required. For one period, `'zero-period-bit-identity'` uses zero period bits. Select `'legacy-one-bit'` explicitly for multiple periods or to match legacy keys. |
| GPUColumnQuantiles, ordinary or filter output | values, mask, output.outputMask | parameters, rationalProbabilities if present, quantile outputs, filterBounds and count/status fields | Radix-selection reductions span chunks without gathering the entire column. Logical total <=16,777,216 for float ranks or any requested filter output, even if values are chunked. 64 quantiles maximum. |
| GPUColumnQuantiles, rational unfiltered | values and mask | Exact rational probability words, parameters, quantile outputs/status | Logical total <=2,147,483,647 across chunks. Each float chunk separately fits a binding (33,554,432 at default); there is no full-column values binding. Requesting outputMask/filterBounds restores the 16,777,216 total cap. |
| GPUClassBreaks, chunk-compatible methods | values and mask for approximate natural breaks, extrema, quantiles, class counts, standard deviation/head-tail | parameters, breaks, classCount, classCounts | Total <=16,777,216; maximumClassCount <=64. Standard-deviation/head-tail preserve global moment order with packed row scratch; integer conversion also uses packed f32 scratch. Approximate histogram/refinement reductions span chunks. Node budget for compiled approximate natural breaks (methods ['natural-breaks'], naturalBreaksMode 'approximate') bounds chunk count: 18 classes and 659 nonempty chunks estimate 32,725; 660 estimate 32,771 and reject. Equal-interval-only at 660 one-row chunks estimates 5,384 nodes and is feasible. |
| GPUClassBreaks, maximum-breaks and exact/monotone natural breaks | None for the compiled global-sort routes, even if inactive this frame | All inputs, controls and outputs packed | Global row sorts remain packed. Exact natural breaks: <=8,192 total rows, materialized storage <=1,024; monotone <=262,144. General class-break total remains <=16,777,216 when those smaller route limits do not apply. |
| GPUColorScale | values, mask, output.colors/classIndices | domain, classCount, palette, parameters, classCounts | Independent row chunks; optional class counts aggregate globally as uint32. Counted total <=4,294,967,295; each scalar chunk separately fits a binding. Controls/class counts are never chunked. |
| GPUScalarScale | values, mask (retained validity spelling), output.values/validity | domain, range, parameters | Independent row chunks, no global order scratch. Shared logical chunk total <=4,294,967,295; domain capacity 2..256; scalar bindings separately fit the device. |
| GPUBivariateClassification | valuesX, valuesY, alphaValues, mask, output.classIds/colors | breaksX/breaksY, palette, parameters, classCounts | Per-chunk row scratch, global class counts. Counted total <=4,294,967,295; each axis maximumClassCount <=16. Every chunk and its scratch separately fits one binding. |
| GPUColumnProfile | None | All numeric/category columns, mask, parameters and all profile outputs | Packed inputs; logical total <=4,294,967,295 but every source and moments/category/histogram/HLL scratch binding fits the device. Up to 16 columns, 1,048,576 histogram bins, HLL precision 4..16, top category count 1..64; category counts <=16,777,216 per category column, plus device total-category limit. Chunked Arrow ingress does not imply this fused panel accepts vectors. |
| GPUWideNumericScan | Resident GPUVector words (uint32x2 or uint32x4), matching resident validity chunks | output result words; captured phase/selection program | Scans every physical word chunk in phases, no packed full-column row sort. Logical total <=4,294,967,295; 64-bit chunk <=16,777,216 rows, 128-bit chunk <=8,388,608 at default binding. Unlike the aligned-view checks elsewhere, this producer checks the entire backing buffer byteLength for every words chunk, every validity chunk and output, not only its live view. A one-row slice backed by 160 MiB still fails a 128 MiB device limit: use smaller backing allocations. This resident GPUVector contract differs from GraphVectorView props. |
| GPUGroupStatistics, counts/sums/means/extremes/moments | keys and key-aligned mask; each value column may have its own boundaries | All group-table outputs/status, percentile probabilities | Auto tiled scratch above the global packed-row threshold for these statistics; each fixed-width tile and moment summary still fits a binding. Total <=2,147,483,648. Global group table remains capacity bounded: 16-byte/group shifted-power binding gives <=8,388,608 groups by default. Float moments may differ in last bits between tiled/packed routes; integer products keep their documented identity. |
| GPUGroupStatistics, median/percentile/mode/unique-count/z-score or scratchMode packed | Input chunks are accepted as above | All grouped outputs, row-domain zScores, controls/status packed | Exact order routes gather values into one global packed sort binding. Z-scores also retain global row scratch. Per-binding/global scratch admission can therefore bound total rows despite chunked input; scalar binding default is 33,554,432 words. Probability-result total fits a separate packed binding (<=33,554,432 float entries by default). |
| GPUKeyLookup, every lookup route | None | All left/right keys, masks/values, controls and outputs | Full right sort/scan and distinct-value order scratch stay packed. Logical left <=4,294,967,295, right <=2,147,483,648; every caller/scratch binding separately fits the device. Every lookup requires rightRows+1 scalar uniqueStart scratch words: rightRows<=33,554,431 at 128 MiB even with scalar keys. Requesting sum or mean additionally requires 2*rightRows scalar sumWords: rightRows<=16,777,216 for that scratch binding. A packed uint32x2 key source independently has the same default 16,777,216-row source-binding ceiling. |
| GPUTemporalReduction, float or Int64; sorted or unsorted | None | All cellIds/timestamps/values/mask/parameters, dense slot outputs and compact occupiedSlots/status | Logical total rows <=4,294,967,295; logical cellCount*bucketCount <=4,294,967,294. These are not binding promises. Word route needs 8 bytes per time-key slot: <=16,777,216 total slots at 128 MiB, plus row scratch. Float route permits <=1,048,576 buckets; scalar scratch default <=33,554,432 words. Sorted scratch includes a two-word range header: rows+2 must fit separately. |
| GPUValueCube | None | All cells/periods/values/mask and dense outputs | Logical row total <=4,294,967,295; logical cellCount*periodCount <=4,294,967,294. Exact accumulation scratch uses 12 bytes/bin, so <=11,184,810 total bins at default binding, independent of source-row binding limits. |
| GPUCubeLayoutTranspose | None | Entire source and destination cubes | No global order scratch. cellCount*periodCount <=4,294,967,295 logically and both scalar bindings fit one device binding: <=33,554,432 total cells by default. |
| GPUCompositeScore, ordinary / rank / PCA | None | Full row-major indicator matrix, mask, parameters and all outputs | indicatorCount <=16; rowCount*indicatorCount <=4,294,967,295 logically. Every matrix/output/scratch plane is packed. Rank additionally sorts per column and requires rowCount<=2,147,483,648, with rowCount*indicatorCount rank words in one binding. Ordinary/PCA scratch planes also receive independent device checks. |
| GPUInequality, parallel / reference | None | Full values/zoneIds/weights/mask, parameters and all zone/Lorenz outputs | Global row sort/scratch remains packed in both modes; total rows<=2,147,483,648 plus binding caps. Zone statistic binding caps zones at 3,050,402 by default; knot count 2..1,025 and total zoneCount*knotCount<=33,554,432 by default. Parallel changes zone traversal, not input chunkability. |
| GPUVariableStandardization, every method | values, mask and output with matching chunk boundaries | Generated reduction controls and global summaries | Global row scratch remains one scalar binding: max(totalRows,2)*4 bytes, so default total <=33,554,432 despite chunkable row ports. Logical total <=4,294,967,295. |
| All thirteen pair-statistics producers (listed below), including simulation/fitting variants | None | All input rows or empirical bins, parameters, random controls, public outputs and statuses packed | Empirical gridded point routes use getGPUPairStatisticsMaximumRowCount: min(floor(bindingBytes/12),16,777,215), default 11,184,810 points. Fit routes consume packed bin/model tables and apply their own shape/iteration limits, not this point-row limit. Simulation/permutation batches and all search/sort/fit scratch must fit separately. These producers require packed inputs. |

The thirteen packed pair producers are GPUVariogram, GPUSpatialCorrelogram, GPURipley, GPURipleyDistanceFunctions, GPUPointPatternIndices, GPUCrossVariogram, GPUSpaceTimeVariogram, GPUMarkedPointStatistics, GPUSpaceTimeKFunction, GPUSpaceTimeDiggleTest, GPURipleyEnvelope, GPUVariogramModelFit and GPUCoregionalizationFit. Each producer retains its own overflow/status contract; there is no universal overflow word. The smaller marked-statistics brute-force route is capped at 16,384 rows; Ripley-envelope simulation count <=199 and Diggle permutation count <=999 do not relax packed point/scratch admission.

### `GPUBivariateClassification`

Classify two columns into an n by n grid and colour each row from a bivariate palette.

::lens{scenario=gpu-dataframe.bivariate-classification rung=result caption="The 3 by 3 grid puts 218 tracts in the low-poverty, low-uninsured class and 64 in the opposite corner (CPU check)."}

#### How it computes

{/* lens:GPUBivariateClassification */}

::lens{scenario=gpu-dataframe.bivariate-classification rung=datum bind=page datum=16 caption="Tract 16 falls in joint class 8: high poverty and high uninsured, shared by 64 tracts (CPU check)."}
::lens{scenario=gpu-dataframe.bivariate-classification rung=formula bind=page datum=16}
::lens{scenario=gpu-dataframe.bivariate-classification rung=kernel caption="Look at classY * classCountX + classX: each axis counts its inner edges at or below the value."}
::lens{scenario=gpu-dataframe.bivariate-classification rung=graph caption="A single bivariate kernel reading two value columns, two edge lists, a palette and parameters."}

{/* parameters: GPUBivariateClassification generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUBivariateClassification` parameters

Import from `@luma-contributors/analysis/dataframe-analysis`: `GPUBivariateClassification`.

- Packer: `getGPUBivariateClassificationParameterValues(settings)`, writes `GPU_BIVARIATE_CLASSIFICATION_PARAMETER_LENGTH` (8) `float32` values.

Settings of `getGPUBivariateClassificationParameterValues`:

| Field | Type | Description |
|---|---|---|
| `classCountX` | `number` | Active classes along X, `1..maximumClassCount`; `breaksX` holds `classCountX + 1` edges. |
| `classCountY` | `number` | Active classes along Y, `1..maximumClassCount`; `breaksY` holds `classCountY + 1` edges. |
| `noDataColor` (optional) | `number` | Packed `rgba8` colour (`r \| g << 8 \| b << 16 \| a << 24`) for no-data rows. Default `0`. |
| `valueByAlpha` (optional) | `GPUBivariateValueByAlpha` | Enables value-by-alpha. Omit to leave the palette alpha unchanged. |

{/* /parameters */}

### `GPUClassBreaks`

Compute choropleth class edges and per-class counts from a float column, with a switchable method.

::lens{scenario=gpu-dataframe.class-breaks rung=result caption="Equal-interval breaks split 788 valid tracts into classes of 257, 250, 168, 85 and 28 (CPU check)."}

#### How it computes

{/* lens:GPUClassBreaks */}

::lens{scenario=gpu-dataframe.class-breaks rung=datum bind=page datum=2 caption="Inner edge e2 is 32.1 percent, the lower bound of the third class (CPU check)."}
::lens{scenario=gpu-dataframe.class-breaks rung=formula bind=page datum=2}
::lens{scenario=gpu-dataframe.class-breaks rung=kernel caption="Look at the equal-interval branch: edges step from the sample minimum by (max - min) / k."}
::lens{scenario=gpu-dataframe.class-breaks rung=graph caption="Moment, quantile and method nodes compiled for the chosen methods, publishing breaks and counts."}

{/* parameters: GPUClassBreaks generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUClassBreaks` parameters

Import from `@luma-contributors/analysis/dataframe-analysis`: `GPUClassBreaks`.

- Packer: `getGPUClassBreaksParameterValues(settings, maximumClassCount)`, writes `getGPUClassBreaksParameterLength(...)` `float32` values.
  - `maximumClassCount` (`number`): Compile-time class capacity of the contributor.

Settings of `getGPUClassBreaksParameterValues`:

| Field | Type | Description |
|---|---|---|
| `method` | `GPUClassBreaksMethod` | Classification method. It must be one of the contributor's compiled `methods`. |
| `classCount` (optional) | `number` | Requested class count `k`. Ignored by `'box-plot'` (always 6) and `'custom'` (one less than the number of edges). For `'head-tail'` and `'maximum-breaks'` it is an upper bound: fewer classes are produced when the data runs out of heads or distinct gaps. |
| `standardDeviationInterval` (optional) | `number` | `'standard-deviation'`: width of one class in standard deviations. Default 1. |
| `headTailRatio` (optional) | `number` | `'head-tail'`: largest head fraction that keeps splitting. Default 0.4 (Jiang's 40% rule). Use 1 for mapclassify's rule, which splits until a head has a single distinct value. |
| `boxPlotHinge` (optional) | `number` | `'box-plot'`: whisker length in interquartile ranges. Default 1.5. |
| `customEdges` (optional) | `ArrayLike<number>` | `'custom'`: ascending class edges `e[0..k]`, at most `maximumClassCount + 1` of them. |

{/* /parameters */}

### `GPUScalarScale`

[`GPUScalarScale`](#gpuscalarscale) maps prepared numeric columns to scalar rendering
attributes using replayable linear, signed square-root, same-sign logarithmic, native fixed,
and categorical point scales. It preserves source chunks and validity without source readback.

The optional input row filter is `mask` (`uint32`, one entry per source row; nonzero keeps the row, zero maps the row to `nullValue` with output validity 0). `validity` is accepted as a `@deprecated` alias with the same meaning; `mask` wins when both are set. Output `validity` is unchanged. Construction errors start with the contributor ID and name the failed limit, for example `scalar-scale mask must hold exactly one entry per source row`.

Prepare scalar controls with matching formats and separate live counts from allocation:

```ts
import {prepareGPUScalarScaleControls} from '@luma-contributors/analysis/dataframe-analysis';
for (const domain of [[], [7]]) {
  const controls = prepareGPUScalarScaleControls({scale: 'point', domain});
  const domainAllocation = new Float32Array(controls.minimumDomainCapacity); // capacity 2
  domainAllocation.set(controls.domain); // upload only this live prefix
  const maximumDomainCount = controls.minimumDomainCapacity; // GPUScalarScale allocation prop
  const domainCount = controls.counts.domainCount; // 0 or 1, already packed in parameters
  // Domain view: format controls.formats.domain, length maximumDomainCount.
  // Range view: format controls.formats.range, length controls.counts.rangeCount (2).
  // Parameters: format controls.formats.parameters, length controls.counts.parameterCount (4).
  // Upload controls.range and controls.parameters; all three formats are 'float32'.
}
```

`minimumDomainCapacity` is at least two even for empty and singleton point domains. Use it as
`maximumDomainCount` for a minimal instance, retain `counts.domainCount` as the live count and
write the returned parameters unchanged. Allocating capacity two does not invent two categories.

{/* parameters: GPUScalarScale generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUScalarScale` parameters

Import from `@luma-contributors/analysis/dataframe-analysis`: `GPUScalarScale`.

- Packer: `getGPUScalarScaleParameterValues(settings)`, writes `GPU_SCALAR_SCALE_PARAMETER_LENGTH` (4) `float32` values.

Settings of `getGPUScalarScaleParameterValues`:

| Field | Type | Description |
|---|---|---|
| `scale` | `'linear' \| 'sqrt' \| 'log' \| 'identity' \| 'point'` | - |
| `domainCount` | `number` | Numeric modes require exactly two entries. Point accepts zero through the capacity. |
| `nullValue` (optional) | `number` | A finite value written for nulls, unknown categories and invalid numerical results. |
| `invert` (optional) | `boolean` | Reverse the GPU domain as the output range of the native fixed (identity) scale only. |

{/* /parameters */}

### `GPUColorScale`

Map a float column to packed rgba8 colours and palette classes with a threshold, step or continuous scale.

Continuous domains may ascend or descend. `extrapolate: true` extends the first and last
palette segments, matching native RGB interpolation with saturated output bytes; class indices
remain clamped. `clamp: true` takes precedence. The existing default (`clamp: false`, no
extrapolation) continues to mark out-of-domain rows as no-data. Signed square-root and power
transforms preserve negative values.

`logMode: 'domain-sign'` selects `log(x)` for a positive domain or `-log(-x)` for a negative
domain. Use finite, nonzero, same-sign domain stops. Zero or wrong-sign inputs are no-data,
except that a constant valid domain maps every finite input to its midpoint. Native log and
extrapolation reject nonfinite input or normalized values; masks still represent nulls, with
transparent no-data by default. Omitting `logMode` preserves the existing positive `logFloor`
behavior. These options reuse the nine-word parameter buffer, so changing them needs no
recompilation. Float32 RGB interpolation can differ from a double-precision reference by one
byte per channel; subnormal arithmetic is outside that parity contract.

::lens{scenario=gpu-dataframe.color-scale rung=result caption="Five poverty bands hold 117, 181, 171, 148 and 171 tracts; darker blue means higher poverty (CPU check)."}

#### How it computes

{/* lens:GPUColorScale */}

::lens{scenario=gpu-dataframe.color-scale rung=datum bind=page datum=1 caption="Tract 1 has 43 percent poverty: above the 40 percent edge, class 4, the darkest blue."}
::lens{scenario=gpu-dataframe.color-scale rung=formula bind=page datum=1}
::lens{scenario=gpu-dataframe.color-scale rung=kernel caption="Look at the binary search over inner domain edges that yields the class."}
::lens{scenario=gpu-dataframe.color-scale rung=graph caption="A single colour-scale kernel reading values, domain, palette and parameters."}

{/* parameters: GPUColorScale generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUColorScale` parameters

Import from `@luma-contributors/analysis/dataframe-analysis`: `GPUColorScale`.

- Packer: `getGPUColorScaleParameterValues(settings)`, writes `GPU_COLOR_SCALE_PARAMETER_LENGTH` (9) `float32` values.
- Packer: `getGPUColorScaleParameterValues(settings)`, writes `GPU_COLOR_SCALE_PARAMETER_LENGTH` (9) `float32` values.

Settings of `getGPUColorScaleParameterValues(settings)` for `GPUColorScaleSettings`:

| Field | Type | Description |
|---|---|---|
| `scale` | `'linear' \| 'quantile' \| 'sqrt' \| 'pow' \| 'log' \| 'symlog' \| 'quantize' \| 'threshold' \| 'ordinal'` | Scale type. |
| `domainCount` | `number` | Number of active domain entries, `1..maximumDomainCount`. A resident `classCount` view supplies a class count `k` and takes precedence with `k + 1` edges. |
| `paletteCount` | `number` | Number of active palette entries, `1..maximumPaletteCount`. |
| `interpolation` (optional) | `GPUColorScaleInterpolation` | `'step'` picks `palette[class]` exactly; `'linear'` blends adjacent entries per channel. Default `'step'`. |
| `clamp` (optional) | `boolean` | Clamp out-of-domain values to the first or last colour instead of no-data. Default `false`. |
| `extrapolate` (optional) | `boolean` | Extrapolate continuous scales along the first/last palette segment, saturating color bytes. Default false; clamp true takes precedence. Ignored by discrete scales. |
| `logMode` (optional) | `'positive-floor' \| 'domain-sign'` | Log transform: legacy positive floor (default), or native same-sign positive/negative domain. Native mode ignores logFloor and requires finite nonzero same-sign domain stops. |
| `noDataColor` (optional) | `number` | Packed `rgba8` colour (`r \| g << 8 \| b << 16 \| a << 24`) for no-data rows. Default `0`. |
| `logFloor` (optional) | `number` | Value that replaces `v <= 0` (and non-positive domain entries) on the `log` scale, as kepler does with a small positive floor. `NaN` treats such rows as no-data. Default `1e-5`. |
| `exponent` (optional) | `number` | Exponent of the `pow` scale. Default `1`. |

Settings of `getGPUColorScaleParameterValues(settings)` for `GPUColorScaleResidentThresholdSettings`:

| Field | Type | Description |
|---|---|---|
| `paletteCount` | `number` | Number of active palette entries, `1..maximumPaletteCount`. |
| `interpolation` (optional) | `GPUColorScaleInterpolation` | `'step'` picks `palette[class]` exactly; `'linear'` blends adjacent entries per channel. Default `'step'`. |
| `clamp` (optional) | `boolean` | Clamp out-of-domain values to the first or last colour instead of no-data. Default `false`. |
| `extrapolate` (optional) | `boolean` | Extrapolate continuous scales along the first/last palette segment, saturating color bytes. Default false; clamp true takes precedence. Ignored by discrete scales. |
| `logMode` (optional) | `'positive-floor' \| 'domain-sign'` | Log transform: legacy positive floor (default), or native same-sign positive/negative domain. Native mode ignores logFloor and requires finite nonzero same-sign domain stops. |
| `noDataColor` (optional) | `number` | Packed `rgba8` colour (`r \| g << 8 \| b << 16 \| a << 24`) for no-data rows. Default `0`. |
| `logFloor` (optional) | `number` | Value that replaces `v <= 0` (and non-positive domain entries) on the `log` scale, as kepler does with a small positive floor. `NaN` treats such rows as no-data. Default `1e-5`. |
| `exponent` (optional) | `number` | Exponent of the `pow` scale. Default `1`. |
| `scale` | `'threshold'` | - |
| `domainCount` (optional) | `never` | Forbidden for this settings branch. |

{/* /parameters */}

### `GPUScalarScale`

Map a prepared scalar column to numeric rendering values (radius, width, height) with a native-style `linear`, `sqrt`, `log`, `identity` or `point` scale. See [`GPUScalarScale`](#gpuscalarscale) for the full contract.

Each mode normalizes the transformed value between the transformed domain endpoints and interpolates the two range endpoints, extrapolating without a clamp. Rows with null or nonfinite input, unknown categories, or unsupported log inputs write the fallback value with output validity zero.

::lens{scenario=gpu-dataframe.scalar-scale rung=result caption="Poverty share mapped to a square-root radius between 1 and 20 pixels; suppressed tracts get the fallback and validity zero."}

#### How it computes

{/* lens:GPUScalarScale */}

::lens{scenario=gpu-dataframe.scalar-scale rung=datum bind=page datum=1 caption="Tract 1 maps its poverty share through the signed square root to a radius."}
::lens{scenario=gpu-dataframe.scalar-scale rung=formula bind=page datum=1}
::lens{scenario=gpu-dataframe.scalar-scale rung=kernel caption="Look at the normalization: the transformed value is placed between the transformed domain endpoints, then the range endpoints are interpolated."}
::lens{scenario=gpu-dataframe.scalar-scale rung=graph caption="A single scalar-scale kernel reading values, domain, range and parameters."}

### `GPUColumnQuantiles`

Select exact quantiles of a float column and a percentile-range row mask, from column to quantile values.

::lens{scenario=gpu-dataframe.column-quantiles rung=result caption="Poverty quartiles are 14.1, 25.6 and 38.1 percent over 788 valid tracts; 711 tracts sit inside the 5-95 filter (CPU check)."}

#### How it computes

{/* lens:GPUColumnQuantiles */}

::lens{scenario=gpu-dataframe.column-quantiles rung=datum bind=page datum=0 caption="First quartile (p = 0.25): 14.1 percent of residents below 150% of poverty."}
::lens{scenario=gpu-dataframe.column-quantiles rung=formula bind=page datum=0}
::lens{scenario=gpu-dataframe.column-quantiles rung=kernel caption="Look at the interpolation: rank h, its fractional part, then the two neighbouring order statistics."}
::lens{scenario=gpu-dataframe.column-quantiles rung=graph caption="Four radix passes, each a histogram plus a select node, then a finish node and the filter mask."}

{/* parameters: GPUColumnQuantiles generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUColumnQuantiles` parameters

Import from `@luma-contributors/analysis/dataframe-analysis`: `GPUColumnQuantiles`.

- Packer: `getGPUColumnQuantilesParameterValues(settings)`, writes `getGPUColumnQuantilesParameterLength(...)` `float32` values.

Settings of `getGPUColumnQuantilesParameterValues`:

| Field | Type | Description |
|---|---|---|
| `quantiles` | `ArrayLike<number>` | Probabilities in `[0, 1]`; others (and NaN) produce a NaN quantile. |
| `interpolation` (optional) | `GPUColumnQuantileInterpolation` | Defaults to `'linear'`. |
| `filterRange` (optional) | `[number, number]` | Percentile filter as fractions `[lower, upper]` in `[0, 1]`. Defaults to `[0, 1]`. |
| `quantileCount` (optional) | `number` | Number of probability slots. Defaults to `quantiles.length`. Unused slots become NaN. |

{/* /parameters */}

## `GPUColumnProfile`

Columns and the optional mask must be packed views. The logical row limit is 4,294,967,295,
further limited by each storage binding. Counts are uint32; float32 statistical fields remain rounded.

Profile several columns at once: counts, nulls, extent, moments, histogram and distinct estimate.

::lens{scenario=gpu-dataframe.column-profile rung=result caption="Median household income has 19 suppressed tracts; the three rate columns have 3 each (CPU check)."}

A dataset statistics panel (kepler.gl / Foursquare Studio field summaries) for several columns of
one table, recomputed on the GPU every frame so it follows filters and brushes without readback
of the rows.

```ts
import {GPUColumnProfile, getGPUColumnProfileParameterValues} from '@luma-contributors/analysis/dataframe-analysis';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
declare const fare: import('@luma.gl/graph').GraphDataView<'float32'>;
declare const vendorCodes: import('@luma.gl/graph').GraphDataView<'uint32'>;
declare const mask: import('@luma.gl/graph').GraphDataView<'uint32'>;
declare const domainParameters: GPUParameterBuffer<'float32'>;
declare const statistics: import('@luma.gl/graph').GraphDataView<'float32'>;
declare const counts: import('@luma.gl/graph').GraphDataView<'uint32'>;
declare const histograms: import('@luma.gl/graph').GraphDataView<'uint32'>;
declare const hyperLogLogRegisters: import('@luma.gl/graph').GraphDataView<'uint32'>;
declare const topCategories: import('@luma.gl/graph').GraphDataView<'uint32'>;
declare const topCategoryCounts: import('@luma.gl/graph').GraphDataView<'uint32'>;
graph.add(new GPUColumnProfile({
  columns: [
    {values: fare}, // float32, numeric
    {values: vendorCodes, kind: 'category', categoryCount: 500} // uint32 dictionary codes
  ],
  mask, // optional uint32, shared by every column
  parameters: domainParameters.importToGraph(graph), // optional histogram domains
  histogramBinCount: 64,
  hyperLogLogPrecision: 12, // 2^12 registers per column
  topCategoryCount: 10,
  output: {statistics, counts, histograms, hyperLogLogRegisters, topCategories, topCategoryCounts}
}));
domainParameters.write(getGPUColumnProfileParameterValues([[0, 100], [NaN, NaN]])); // NaN = auto
```

- `statistics` holds `GPU_COLUMN_PROFILE_STATISTIC_COUNT` (11) floats per column, indexed by
  `GPU_COLUMN_PROFILE_STATISTIC`: count, nullCount, minimum, maximum, sum, mean, variance
  (population), sampleVariance, standardDeviation, distinctEstimate, overflowCount. `counts`
  holds the exact `uint32` count and null count per column.
- NaN (numeric) or `0xffffffff` (category) rows are nulls. ±Infinity counts toward count, minimum
  and maximum but is excluded from the moments. With no rows, the float statistics are NaN and the
  distinct estimate is 0. Category columns report their smallest and largest code and NaN moments.
- Moments are a fixed-order two-level Chan/Welford reduction (workgroup trees, then one
  fixed-order merge), with no float atomics: bitwise identical across runs and stable for large
  offsets (values near 1e6 with a spread of 100).
- Histograms: equal-width bins over a per-frame `[minimum, maximum]` per column (a
  `GPUColumnProfileDomain`: a pair or `{minimum, maximum}`), or the frame's finite minimum and
  maximum when a bound is NaN; the last bin includes the maximum and values outside are not counted.
  Bin assignment has the convention of the official `GPUHistogram`,
  `min(floor((v - minimum) / (maximum - minimum) * bins), bins - 1)`, so a value on a bin edge is
  in the upper bin for every bin count (-0.875 is in bin 5 of 30 over `[-3.5, 12.25]`). The bin
  is decided by correctly rounded float32 products
  (`k * range <= (v - minimum) * bins < (k + 1) * range`), not by one division, so it is identical
  on every adapter. Category columns get per-code counts in their first bins.
- Distinct count: HyperLogLog over a murmur3 `fmix32` hash of the value bits (`-0` folded to
  `+0`) with `atomicMax` registers and linear-counting small-range correction. Registers are
  exact, and the estimate is within about `1.04 / sqrt(2^p)` relative error.
- Top categories: exact per-code counts with `uint32` atomics, then the top `topCategoryCount`
  ordered by count (descending) and code (ascending). Codes at or above `categoryCount` are
  counted in `overflowCount`.
- At most 16 columns and 4,294,967,295 logical rows (within device limits), and at most `getGPUColumnProfileMaximumCategoryTotal(device)` (2^25 on a default device)
  `categoryCount` summed over the category columns (one 4-byte slot per category in a 128 MiB binding).
  Inputs must be single packed views. The per-column kernels read
  the current column from a GPU cursor word, so the pipeline count does not grow with the column
  count.

#### How it computes

{/* lens:GPUColumnProfile */}

::lens{scenario=gpu-dataframe.column-profile rung=datum bind=page datum=3 caption="Column 3, median household income: 772 non-null tracts, 19 nulls."}
::lens{scenario=gpu-dataframe.column-profile rung=formula bind=page datum=3}
::lens{scenario=gpu-dataframe.column-profile rung=kernel caption="Look at the Welford and Chan merges: moments never come from raw squared sums."}
::lens{scenario=gpu-dataframe.column-profile rung=graph caption="Per-column moment tiles, a merge, domain, histogram and HyperLogLog nodes, then a finish node."}

{/* parameters: GPUColumnProfile generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPUColumnProfile` parameters

Import from `@luma-contributors/analysis/dataframe-analysis`: `GPUColumnProfile`.

- Packer: `getGPUColumnProfileParameterValues(domains, columnCount?)`, writes `getGPUColumnProfileParameterLength(...)` `float32` values.
  - `domains` (`readonly (GPUColumnProfileDomain | undefined)[]`): One entry per column, `undefined` for fully automatic.
  - `columnCount` (`number`): Number of columns. Defaults to `domains.length`.

{/* /parameters */}

## `GPUGroupStatistics`

Reducible statistics use tiled scratch controlled by scratchTileSize. Compact tiled padded rows are
at most 4,294,967,295; packed output and exact order-statistic scratch remain device-binding
bounded.

Groups rows by a key and computes count, mean, variance and median of a value column per group, ascending by key.

::lens{scenario=gpu-dataframe.group-statistics rung=result caption="Mean tract poverty runs from 7.4 percent in community area 12 to 70.2 percent in area 37 across 77 areas (CPU check)."}

Groups rows by a 32- or 64-bit key and computes kepler-style statistics per group and value column on the GPU: count, sum, mean, minimum, maximum, variance, standard deviation, skewness, kurtosis, median, percentiles, mode, unique count and a per-row group z-score. Use it for "aggregate this column by that key" panels, choropleth values keyed by a region id or an H3 or Quadbin cell, and per-group outlier styling.

```ts
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
import {GPUGroupStatistics} from '@luma-contributors/analysis/dataframe-analysis';
declare const keys: import('@luma.gl/graph').GraphDataView<"uint32x2" | "uint32">;
declare const mask: import('@luma.gl/graph').GraphDataView<'uint32'>;
declare const values: import('@luma.gl/graph').GraphDataView<'float32'>;
declare const counts: import('@luma.gl/graph').GraphDataView<'uint32'>;
declare const means: import('@luma.gl/graph').GraphDataView<'float32'>;
declare const variances: import('@luma.gl/graph').GraphDataView<'float32'>;
declare const medians: import('@luma.gl/graph').GraphDataView<'float32'>;
declare const quantiles: import('@luma.gl/graph').GraphDataView<'float32'>;
declare const modes: import('@luma.gl/graph').GraphDataView<'float32'>;
declare const zScores: import('@luma.gl/graph').GraphDataView<'float32'>;
declare const groupKeys: import('@luma.gl/graph').GraphDataView<"uint32x2" | "uint32">;
declare const groupCounts: import('@luma.gl/graph').GraphDataView<'uint32'>;
declare const count: import('@luma.gl/graph').GraphDataView<'uint32'>;
declare const overflow: import('@luma.gl/graph').GraphDataView<'uint32'>;
declare const requiredCount: import('@luma.gl/graph').GraphDataView<'uint32'>;
const fractions = new GPUParameterBuffer(device, {id: 'fractions', format: 'float32', length: 3});
graph.add(
  new GPUGroupStatistics({
    keys, // uint32, or uint32x2 (low, high) words; all-ones is "no key"
    mask, // optional uint32
    columns: [
      {
        values, // float32 per row; non-finite rows are excluded for this column only
        statistics: ['count', 'mean', 'variance', 'median', 'percentiles', 'mode', 'zScore'],
        output: {counts, means, variances, medians, quantiles, modes, zScores}
      }
    ],
    variance: 'sample', // or 'population'
    probabilities: fractions.importToGraph(graph),
    output: {keys: groupKeys, counts: groupCounts, status: {count, overflow, requiredCount}} // GPUGroupStatisticsOutput
  })
);
fractions.write(Float32Array.of(0.1, 0.5, 0.9)); // no recompile
```

Canonical grouped input `probabilities` is in **fractions [0,1]** and output `quantiles` holds
values in the measure's units. Request the `'percentiles'` statistic and bind `output.quantiles`.
Network travel-time percentiles use **percent (0,100]**; raster percentiles use **percent [0,100]**: `.95 ↔ 95`.
For travel times `[5,90]`, the network nearest-rank 95th percentile is 90;
accidentally requesting `.95` percent yields 5, not the 95th percentile.

Keep **key eligibility** independent of each measure's validity. For one key repeated three times,
use a key mask `[1,1,1]`, x validity `[0,1,1]` and y validity `[1,0,1]`:

```ts
import * as arrow from 'apache-arrow';
import {prepareArrowMeasureColumnsForGPUAnalytics} from '@luma-contributors/analysis/analysis-core';
const table = new arrow.Table({
  x: arrow.vectorFromArray([null, 0, 2], new arrow.Float32()),
  y: arrow.vectorFromArray([12, null, 4], new arrow.Float32())
});
const measures = prepareArrowMeasureColumnsForGPUAnalytics(table, ['x', 'y']);
// measures['x'].chunks contains [NaN, 0, 2]; measures['y'].chunks contains [12, NaN, 4].
// Each measure also returns its own source-aligned validity chunks.
// COUNT(*) = 3; COUNT(x) = 2; AVG(x) = (0+2)/2 = 1; AVG(y) = (12+4)/2 = 8.
```

Upload these owned float chunks once and pass them as separate `columns[].values`; nulls become
NaN while real zeros remain zero. Do not intersect the validity masks: that would leave only the
last row. Float measures inherit source units; Float64 needs an explicit precision policy.

`prepareArrowMeasureColumnsForGPUAnalytics` accepts Float16, Float32 and Float64 source fields
only; integer fields are rejected even if a precision policy would narrow them.
For lookup, use the same prepared chunks as aggregate columns: `count` without a column counts
three matching rows, `count` with x counts two, and separate means yield 1 and 8. A first-row
gather of x is still missing. Preserve that NaN (or gather its validity alongside it), and combine
`matched` with per-measure finite/valid flags before a downstream fit; `matched` alone says only
that the key exists.

- Output groups are ascending by key, capacity is `output.keys.length`, at most `getGPUGroupStatisticsMaximumGroupCount(device)` (2^23 on a default device; the 16-byte moment sums per group fit one 128 MiB binding) and, with `probabilities`, `capacity * P` at most `getGPUGroupStatisticsMaximumPercentileTotal(device)` (2^25 on a default device). When more groups exist, the smallest keys are kept, `status.overflow` is 1 and `status.requiredCount` is the unclamped number. Tail rows hold the all-ones key, count 0, sums 0 and NaN for every other float statistic.
- `output.counts` counts valid rows (mask set, key valid); each column's `counts` counts finite values. Zero to four columns; each column lists the statistics it needs (a compile-time set) and supplies exactly the matching output views: `counts`, `sums` (signed 64-bit fixed point, `uint32x2`, the sum of `roundHalfEven(value * sumScale)`, default scale 65536: each value is rounded to `1 / sumScale` first) and/or `sumValues` (the sum of the values themselves as float32, from a 96-bit fixed-point accumulator whose unit follows the largest value of the column; it does not depend on `sumScale`) for `sum`, `means`, `minimums`, `maximums`, `variances`, `standardDeviations`, `skewness`, `kurtosis`, `medians`, `quantiles` (`capacity * P`, group-major), `modes`, `uniqueCounts`, and `zScores` (one row per source row).
- Optional `columns[i].output.status.sumOverflow` is one `uint32` word and requires `sums`. It reports
  caller-scale row clamping or a final signed-64-bit total outside its range for retained groups,
  independently of compact capacity overflow. The `sums` words retain their clamp/wrap semantics;
  float `sumValues` and `means` are independent. Requesting the flag adds parallel guarded reductions.
- Exactness: counts, the fixed-point `sums` words, minimum, maximum, median, percentiles, mode and unique counts are bit-identical to a CPU reference. `sumValues` rounds the accumulated quantized integer total once to float32; shared-scale quantization can lose small values across keys (see below). `means` divides the fixed-point total by the exact integer finite count before rounding to float32 (a few float32 units in the last place), independently of the published `sumValues`: values 1.0e-4, 1.2e-4, 1.4e-4 have mean 1.2e-4. When the sum is past the float32 range, `sumValues` is infinite but `means` is taken from the exact accumulator and stays finite (four values of 1e38 have mean 1e38). A value keeps all its bits in the sum while its magnitude is at least `max * 2^(h - 71)`, where `max` is the largest value of the column in a kept group and `h` the bit length of the row count. Percentiles use the numpy `linear` rule (`h = (n - 1) p`, `lo + (hi - lo) * (h - floor h)` in f32) and the order-statistic policy shared with `GPUColumnQuantiles`, with `p` read per frame; a fraction that is NaN or outside [0, 1] gives NaN (it is no longer clamped). Mode is the longest run of equal values (ties: smallest value). `-0` is treated as `+0`; NaN and infinities are excluded.
- Shared-scale precision (also used by [key lookups](#gpukeylookup) and [value cubes](#gpuvaluecube)):
  with row-count bit length `h` and maximum binary exponent `E`, the shared unit is
  `2^(E + 1 + h - 95)`. Each term rounds to the nearest unit; the total error is at most
  `n / 2` units before float32 publication. Values at least `maxMagnitude * 2^(h - 71)`
  retain all 24 significand bits. This is a cross-key bound, not an exact-real-sum guarantee.
  For left keys `[1]`, right keys `[1,2]`, values `[1,2**100]`, `h=2` gives unit 256:
  the matched singleton 1 rounds to zero, even though the huge right key never matches.
  Group and cube bins can suffer the same cross-key loss. When independent small-group accuracy
  is required, set `scaleMode: 'per-group'` for group statistics or key lookups, or
  `scaleMode: 'per-bin'` for value cubes. The default is `'shared'`. Local scales isolate
  unrelated keys' maxima and derive headroom from each group's or bin's **local finite count**,
  rather than the full column count; each route retains its quantization bound.
  Means divide the accumulated integer total by a separate exact integer count, independently
  of rounded `sumValues`; identical finite maxima have finite means even if `sumValues`
  overflows. Empty means are NaN. Quantization loss still affects the mean.

- Moments are corrected two-pass: `d = v - mean` is summed in a fixed order (a binary tree over sorted row blocks), so a second encode is bitwise identical. The signed first and third powers are scaled by a power of two chosen from each group's extremes before reduction and restored afterwards, preventing same-sign branches from overflowing before opposite signs cancel. Values with binary exponent at most 30 use no scaling. Even powers retain their float32 range limits. Each level merges only the group crossing each block midpoint; the work is O(rows), the depth is O(log rows), and a dominant group spreads across workgroups. The four power sums use separate scalar scratch bindings (16 bytes per input row in total); each binding must fit the device storage limit. The row count is also bounded by the published sort maximum of 2^31. `estimatedCommandNodeCount` includes the merge levels and conservative allowances for published sorts, scans and gathers, checked against the shared command-node budget. Central moments about the true mean come from the shifted power sums. Oracle specs require relative tolerances of 1e-6 for means, 2e-5 for variance, and 1e-4 for skewness, kurtosis and z-scores, with small absolute tolerances near zero. Skewness is Fisher-Pearson `g1`, kurtosis is excess `g2`; both are NaN for constant groups, a group whose variance is under 2^-19 of its shifted sum of squares is treated as constant (the shift mean is rounded to float32). Sample variance of one value is NaN; the z-score is `(v - mean) / sd` with the same corrected mean the variance uses, and is 0 when the deviation is 0 or a sample group has one value, NaN for masked, invalid-key, non-finite or capacity-dropped rows.
- Dense mode: `keyCount` (compile-time) requires `uint32` keys in `[0, keyCount)`. `output.keys.length` and every per-group output hold exactly `keyCount` rows, and row `k` is key `k` (`output.keys[k] = k`, `output.status.count` and `status.requiredCount` are `keyCount`, `status.overflow` is 0). Keys with no rows keep a row: `counts` 0, `sums` and `uniqueCounts` 0, and NaN for mean, minimum, maximum, variance, standard deviation, skewness, kurtosis, median, percentiles and mode. Keys outside the range and the reserved all-ones key are skipped like masked rows and are not reported as overflow. It costs one binary search per key instead of a group scan, and suits region IDs or other small dense key spaces (the choropleth recipe uses it so empty polygons keep a row). The default compact table of occupied keys is unchanged.
- Cost: one stable key sort (two 32-bit radix sorts for 64-bit keys); each column that asks for median, percentiles, mode or unique count adds a value sort plus a key re-sort. Chunked reducible inputs use fixed tile scratch; exact order statistics still require device-bound packed scratch. Approximate distinct counts (HyperLogLog) are not included.

### Reading signed sums

```ts
import {decodeGPUGroupSum} from '@luma-contributors/analysis/dataframe-analysis';
const words = Uint32Array.of(0xffff0000, 0xffffffff);
const {scaledInteger, value} = decodeGPUGroupSum(words, 0, 65536);
// scaledInteger === -65536n; value === -1 (the source measure's units).
```

Each `sums` row is a signed two's-complement 64-bit integer in low/high word order.
Pass the producer's `sumScale`; the decoder rounds the **exact integer/scale quotient once**
to binary64, ties to even. This can differ from rounding the bigint to Number before dividing:
`scaledInteger = 36028797018963969n`, scale 3 gives `12009599006321324`, while
`Number(scaledInteger) / 3` gives `12009599006321322`. The bigint retains every integer bit.
The decoder accepts `Uint32Array`, byte-offset `Uint8Array` and `ArrayBuffer` readbacks and rejects
partial pairs or missing rows. Inspect the optional one-word `sumOverflow` companion before
using a clamped/wrapped total: the per-row scaled limit is `±(2**62 - 2**38)`
(`4611685743549480960n` in magnitude), separately from the final signed-64 total wrap.
For example, one `2**46` at scale 65536 clamps to `4611685743549480960n`, even though
its mathematical scaled value `2**62` fits signed int64. The final total wraps to signed 64 bits. This numeric loss is independent of compact capacity overflow.
Floating `sumValues` uses a separate adaptive accumulator and is not decoded with `sumScale`.
[Key lookup sums](#gpukeylookup) and [cell-table sums](./gpu-spatial-analysis/operators-cells.md)
use the same signed low/high representation.

#### How it computes

{/* lens:GPUGroupStatistics */}

::lens{scenario=gpu-dataframe.group-statistics rung=datum bind=page datum=34 caption="Community area 35 has 10 tracts, mean poverty 39.3 percent and the widest sample variance, 357.6 (CPU check)."}
::lens{scenario=gpu-dataframe.group-statistics rung=formula bind=page datum=34}
::lens{scenario=gpu-dataframe.group-statistics rung=kernel caption="Look at the corrected two-pass moments: deviations from the fixed-point mean are summed before the variance divides by n - 1."}
::lens{scenario=gpu-dataframe.group-statistics rung=graph caption="A radix sort of rows by key, group ranges from head flags and a scan, then the moment and order-statistic nodes."}

## `GPUKeyLookup`

Each left row looks up its key in the right table. When right keys repeat, gather takes the value
at the smallest matching right row index and aggregates use all matching right rows.

```ts
import {GPUKeyLookup} from '@luma-contributors/analysis/dataframe-analysis';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
declare const graph: GPUCommandGraph;
declare const leftKeys: GraphDataView<'uint32'>; // [2, 9, 1, 2]
declare const rightKeys: GraphDataView<'uint32'>; // [2, 2, 1]
declare const rightValues: GraphDataView<'float32'>; // [10, 20, 30]
declare const gathered: GraphDataView<'float32'>; // four left rows
declare const sums: GraphDataView<'float32'>; // four left rows
declare const rightRows: GraphDataView<'uint32'>; // four left rows
declare const matchCounts: GraphDataView<'uint32'>; // four left rows
declare const matchedLeftRows: GraphDataView<'uint32'>; // capacity four
declare const count: GraphDataView<'uint32'>; // one word
declare const overflow: GraphDataView<'uint32'>; // one word
graph.add(new GPUKeyLookup({
  leftKeys, rightKeys,
  compactMatchedLeftRows: true,
  gather: [{column: rightValues, output: gathered}],
  aggregates: [{operation: 'sum', column: rightValues, output: sums}],
  output: {rightRows, matchCounts, rows: {ids: matchedLeftRows, status: {count, overflow}}}
}));
// After submission, key 2 gathers from row 0 (10) and sums rows 0 and 1 (10 + 20 = 30).
// Key 9 has no match. Key 1 gathers and sums the single value 30 at right row 2.
// rightRows = [0, 0xffffffff, 2, 0], matchCounts = [2, 0, 1, 2]
// gathered = [10, NaN, 30, 10], sums = [30, NaN, 30, 30]
// matchedLeftRows = [0, 2, 3, 0xffffffff], count = 3, overflow = 0
```

Prepare cross-table categorical keys with one shared dictionary **before uploading**. Independent
encoders can assign code 0 to different labels. This lookup uses differently named fields:

```ts
import * as arrow from 'apache-arrow';
import {prepareArrowKeyPairForGPUAnalytics} from '@luma-contributors/analysis/analysis-core';
const leftTable = arrow.tableFromArrays({label: ['A', 'B']});
const rightTable = arrow.tableFromArrays({name: ['B', 'A'], value: Float32Array.of(20, 10)});
const pair = prepareArrowKeyPairForGPUAnalytics(leftTable, 'label', rightTable, 'name');
// pair.left.chunks: [Uint32Array.of(0, 1)]
// pair.right.chunks: [Uint32Array.of(1, 0)]
// Upload chunks as uint32 keys and validity as leftMask/rightMask.
// Gathering right value by these keys yields left-aligned [10, 20].
// Decode category code k with pair.dictionary?.values[k].
```

Keep record-batch chunk boundaries and source-aligned validity. Integer keys instead use exact
`uint32x2` low/high words, with `pair.signed` recording signedness; both sides must agree.
The reserved 64-bit all-ones identifier is rejected: remap it before preparation. Do not narrow
identifiers to float32. Cross-table grouped keys likewise require compatible encodings; equal
integer codes prove equal labels only under the same dictionary.

Paired categorical equality uses an unordered shared codebook, even when left and right
dictionaries declare different ordinal orders. Each ordered source must still preserve its
labels and order across its own batches. For unordered source dictionaries, a valid index
decoding to a null label has eligibility zero. An observed ordered definition requires unique
string labels and rejects null labels before returning any mask. Schema-only ordered
dictionaries remain unobserved, whereas an observed empty dictionary is a real ordinal definition.
See [Arrow preparation](./reused-analysis-preparation.md#preparearrowtableforgpuanalytics).

Compact `rows.status` destinations, including optional `candidateOverflow`, are writable outputs:
they must not overlap other outputs or share buffers with input keys. A standalone exact lookup has
no candidate loss and writes zero there; final overflow reports only compact capacity. When the
right table may be incomplete, supply `rightTable: {keys: rightKeys, status: {overflow,
candidateOverflow}}`, retaining both available producer loss views. This requires
`output.status.complete`: it is 1 only when neither supplied loss word nor local compact overflow
is set. Requested table/compact `candidateOverflow` forwards either upstream loss word. A retained
table can have `overflow = 0` and `candidateOverflow = 1`; exact lookup cannot recover its missing keys.

`count-distinct` counts distinct finite values among matching right rows; missing and nonfinite
values do not contribute. Right tables are bounded by 2,147,483,648 rows, subject to sort and device
limits. Left-row and output ordering remains unchanged.

::lens{scenario=gpu-dataframe.key-join rung=result caption="1,499 of 1,500 observations pick up their tract's poverty; the one in tract 789 has no estimate and stays unmatched (CPU check)."}

Keys are `uint32` or 64-bit `uint32x2` little-endian `(low, high)` words (the same
format on both sides). The all-ones key (`0xffffffff`, or both words all-ones) means "no key" and
never matches; masked rows (`mask == 0`) never match.

Valid right rows are sorted by key with a stable radix sort (one pass per key word), so equal keys
stay in right-row order. Segment heads and a scan produce the unique right keys with row ranges;
each left row binary-searches them on the unsigned `(high, low)` order.

- `gather` copies a right column (`float32` or `uint32`, bit for bit) from the smallest matching
  right row. Unmatched rows get NaN or `0xffffffff`.
- `aggregates` run over all matching right rows: `count`, `count-distinct`, `sum`, `mean`, `minimum`, `maximum`.
  `sum` is the shared-scale quantized sum of the finite values in a 96-bit fixed-point accumulator whose unit
  follows the largest value among the valid right rows, rounded once to float32. `mean` divides the
  fixed-point total by the exact integer finite count before rounding to float32; it does not divide
  the published rounded `sum`. A column of 1e-4 and a column of 1e6 are both right to float32. The
  optional `sums` words are a separate 64-bit sum of `roundHalfEven(value * sumScale)` (default
  65536): each value is rounded to `1 / sumScale` first, so a value below half a unit adds
  nothing there. Extremes use ordered-key atomics. Non-finite values are excluded per aggregate.
  Unmatched rows get 0 for `count` and `count-distinct`, and NaN otherwise.
  Optional `aggregates[i].status.sumOverflow` is one `uint32` word, requires `sums`, and reports row clamping
  or signed-64-bit total loss only for right keys contributing to valid, unmasked left matches.
  Unmatched right keys cannot set it. It is independent of compact capacity overflow and the
  float aggregate output; existing `sums` retain clamp/wrap semantics.
- `output.rightRows`, `matchCounts`, `matched` (usable as a crossfilter live mask) are left-aligned;
  `rightMatched` is right-aligned (1 when some valid left row has the same key).
- `compactMatchedLeftRows` defaults to `false` and keeps one slot per left row in the aligned
  outputs. With `true`, the same aligned outputs are available and the required `output.rows`
  additionally lists each matched left row index once, in ascending order. `status.count` is
  clamped to capacity, `status.overflow` flags insufficient capacity, and optional
  `status.requiredCount` reports the full match count in left rows. Entries past `status.count`
  are `0xffffffff`. Supply `output.rows` only when compaction is enabled.

The shared-scale bound and unmatched `2**100` example under
[GPUGroupStatistics](#gpugroupstatistics) apply to this lookup too. The numeric-loss flag
`sumOverflow` checks caller-scale signed sums only; it does not detect adaptive-scale rounding.
Use [decodeGPUGroupSum](#reading-signed-sums) for aggregate `sums`, with the same `sumScale`.

For sparse event counts, a missing key means zero only if the period was fully observed **and**
the grouped right table is complete. Pass its keys and loss together to the lookup; a truncated
table can omit a real event key. In this example both gather routes share the same certification:

```ts
import {GPUKeyLookup} from '@luma-contributors/analysis/dataframe-analysis';
import {GPUGather} from '@luma.gl/graph/gpu-core';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph';
declare const graph: GPUCommandGraph;
declare const table: {
  keys: GraphDataView<'uint32'>; counts: GraphDataView<'uint32'>;
  status: {overflow: GraphDataView<'uint32'>; candidateOverflow?: GraphDataView<'uint32'>};
};
declare const leftKeys: GraphDataView<'uint32'>;
declare const rightRows: GraphDataView<'uint32'>;
declare const complete: GraphDataView<'uint32'>; // one word
// Use distinct planes if comparing the two equivalent zero-fill routes in one graph.
declare const plane: GraphDataView<'uint32'>;
declare const directPlane: GraphDataView<'uint32'>;
graph.add(new GPUKeyLookup({
  id: 'event-count-lookup', leftKeys, rightKeys: table.keys, rightTable: table,
  gather: [{column: table.counts, output: directPlane, unmatchedValue: 0}],
  output: {rightRows, status: {complete}}
}));
graph.add(new GPUGather({id: 'event-count-gather', source: table.counts, indices: rightRows, output: plane}));
// After submit/readback, require complete[0] === 1 before interpreting either plane's misses as zero.
// Observed keys [4,9], capacity 1 => retained [4], table overflow 1 => complete 0.
// The zero for key 9 is unusable in this control: key 9 actually had an event.
// Inherited-loss-only control: overflow 0, candidateOverflow 1 also gives complete 0.
// Only overflow 0 AND every supplied candidateOverflow 0 can certify these missing-key zeros.
```

Missing observation coverage remains missing even when table completeness is 1. A zero-fill
value alone never certifies observation coverage or table completeness.

The lookup's gather defaults to `0xffffffff` for missing unsigned values. Set
`gather: [{column: table.counts, output: plane, unmatchedValue: 0}]` to fill them with zero.

### Grouped predictors into OLS

The function below gathers one predictor and two column planes in separate dispatches, then transposes
the planes into `predictors[row * 2 + column]`. All rows follow the left observations.
`zeroRows` is an immutable uint32 view filled with zeros, one per observation; gathering the
lookup's one-word completeness certificate through it broadcasts admission without a CPU readback. Allocate all
mask outputs in separate buffers. `measureValidity` is a left-row mask requiring both predictors
and the response to be usable; OLS also excludes nonfinite inputs.

```ts
import {GPUKeyLookup, GPUCubeLayoutTranspose} from '@luma-contributors/analysis/dataframe-analysis';
import {GPUGather, GPUMask} from '@luma.gl/graph/gpu-core';
import {GPUOrdinaryLeastSquares, type GPUOrdinaryLeastSquaresOutput} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph';
export function addGroupedFits(
  graph: GPUCommandGraph, leftKeys: GraphDataView<'uint32'>,
  table: {keys: GraphDataView<'uint32'>; meansX: GraphDataView<'float32'>;
    meansY: GraphDataView<'float32'>;
    status: {overflow: GraphDataView<'uint32'>; candidateOverflow?: GraphDataView<'uint32'>}},
  response: GraphDataView<'float32'>,
  views: {onePredictor: GraphDataView<'float32'>; columnPlanes: GraphDataView<'float32'>;
    predictors: GraphDataView<'float32'>; rightRows: GraphDataView<'uint32'>;
    matched: GraphDataView<'uint32'>; complete: GraphDataView<'uint32'>;
    zeroRows: GraphDataView<'uint32'>; broadcastComplete: GraphDataView<'uint32'>;
    eligible: GraphDataView<'uint32'>; measureValidity: GraphDataView<'uint32'>;
    fitMask: GraphDataView<'uint32'>},
  output: {one: GPUOrdinaryLeastSquaresOutput; two: GPUOrdinaryLeastSquaresOutput}
) {
  const rowCount = response.length;
  // columnPlanes is packed float32 with capacity 2 * rowCount, one contiguous plane per column.
  const firstPlane = graph.createDataView(views.columnPlanes.buffer, {
    format: 'float32', length: rowCount, byteOffset: views.columnPlanes.byteOffset
  });
  const secondPlane = graph.createDataView(views.columnPlanes.buffer, {
    format: 'float32', length: rowCount, byteOffset: views.columnPlanes.byteOffset + rowCount * 4
  });
  graph.add(new GPUKeyLookup({
    id: 'predictor-lookup', leftKeys, rightKeys: table.keys, rightTable: table,
    output: {rightRows: views.rightRows, matched: views.matched, status: {complete: views.complete}}
  }));
  // Adjacent planes can have overlapping aligned binding ranges. Separate dispatches preserve
  // the nonoverlapping logical slices, including an odd observation count such as five.
  graph.add(new GPUGather({
    id: 'first-predictor-plane', source: table.meansX, indices: views.rightRows, output: firstPlane
  }));
  graph.add(new GPUGather({
    id: 'second-predictor-plane', source: table.meansY, indices: views.rightRows, output: secondPlane
  }));
  graph.add(new GPUGather({
    id: 'one-predictor', source: table.meansX, indices: views.rightRows, output: views.onePredictor
  }));
  graph.add(new GPUCubeLayoutTranspose({
    values: views.columnPlanes, from: 'period-major', cellCount: rowCount,
    periodCount: 2, output: views.predictors
  }));
  graph.add(new GPUGather({
    id: 'broadcast-join-complete', source: views.complete,
    indices: views.zeroRows, output: views.broadcastComplete
  }));
  graph.add(new GPUMask({
    id: 'complete-matches', inputs: [views.matched, views.broadcastComplete],
    output: views.eligible
  }));
  graph.add(new GPUMask({
    id: 'fit-validity', inputs: [views.eligible, views.measureValidity], output: views.fitMask
  }));
  graph.add(new GPUOrdinaryLeastSquares({
    id: 'one-predictor-fit', predictors: views.onePredictor, predictorCount: 1,
    response, mask: views.fitMask, output: output.one
  }));
  graph.add(new GPUOrdinaryLeastSquares({
    id: 'two-predictor-fit', predictors: views.predictors, predictorCount: 2,
    response, mask: views.fitMask, output: output.two
  }));
  return {groupOverflow: table.status.overflow,
    groupCandidateOverflow: table.status.candidateOverflow, complete: views.complete,
    one: output.one, two: output.two};
}
```

The mask implements `matched AND broadcast(complete)` plus measure validity.
Retain `groupOverflow`, `groupCandidateOverflow` and `complete` beside both fit statuses. In the truncated-table control
with source keys `[4,9]` but only key 4 retained, **every** fit-mask row is zero; a fit on the
retained subset must not be mistaken for a complete group-derived model. The same all-zero fit
mask is required with final overflow 0 and inherited candidate overflow 1, even if a retained key matches.

Pass the whole producer status in both workflows. Ordinary grouped tables supply final `overflow`
without `candidateOverflow`; producers carrying inherited loss may additionally supply that word.

All results are integer or fixed-point, independent of thread order, and bitwise reproducible.
Input contents (keys, masks, values) are per-frame and need no rebuild; view lengths, key format,
`compactMatchedLeftRows`, `sumScale` and the set of outputs are topology.

#### How it computes

{/* lens:GPUKeyLookup */}

::lens{scenario=gpu-dataframe.key-join rung=datum bind=page datum=0 caption="Observation 0 is in tract 46, which matches right row 46 and gathers poverty 9.7 percent (CPU check)."}
::lens{scenario=gpu-dataframe.key-join rung=formula bind=page datum=0}
::lens{scenario=gpu-dataframe.key-join rung=kernel caption="Look at the binary search of each left key over the unique right keys, in unsigned (high, low) order."}
::lens{scenario=gpu-dataframe.key-join rung=graph caption="A radix sort of the valid right keys, segment heads and a scan, then the per-left-row lookup and gather."}

## `GPUVariogram`, `GPUSpatialCorrelogram`, `GPURipley`, `GPURipleyDistanceFunctions`, and `GPUPointPatternIndices`

Every contributor here accepts at most `getGPUPairStatisticsMaximumRowCount(device)` (11,184,810 on a default device, never above 2^24 - 1) `positions`
rows: the cell-sorted points table holds 12 bytes per row and must fit one 128 MiB storage binding.

Pair statistics over planar points share one deterministic pair-histogram pass: rows are sorted
into a per-frame lattice whose cells are at least the per-frame `maximumDistance` wide, one thread
per included row visits the 3x3 cell neighbourhood, and each pair adds integer amounts into
per-workgroup shared-memory 64-bit accumulators that are then added into global 64-bit
accumulators. Float terms are fixed point (a bounded term such as a distance maps its maximum to
2^24; the variogram and correlogram unit comes from the typical term, see below, never from the
maximum), so every result is exact or has a documented fixed-point bound, independent of
scheduling, and bitwise reproducible. In `GPUVariogram` and `GPUSpatialCorrelogram` a 64-bit sum that
would wrap raises the optional `output.status.overflow` word (`readPairOverflow`) instead of wrapping
silently. Pair counts saturate at `0xffffffff`. `GPUVariogram` and `GPUSpatialCorrelogram` raise
`output.status.overflow` for a count at or above `0xffffffff`, including the exactly representable
boundary: their scalar writers inspect the already-clamped counts.
`GPUCrossVariogram`, `GPUSpaceTimeVariogram` and
`GPUMarkedPointStatistics` instead set status 3 in their per-bin `status` view. `GPURipley`,
`GPURipleyEnvelope`, `GPURipleyDistanceFunctions` and `GPUPointPatternIndices` publish no overflow
or saturation word. Every pair-statistics input and output is a single packed view; chunked vectors
are rejected.
When `maximumDistance` covers the extent the pass is a plain all-pairs loop. A row is included when
its mask is nonzero, its value (if any) finite and its position finite and inside the per-frame
`bounds`. Parameters (`[minX, minY, maxX, maxY, maximumDistance, ...]`) are per-frame, packed from a
`GPU<Class>Settings` object (`GPUVariogramSettings`, `GPUSpatialCorrelogramSettings`,
`GPURipleySettings`, `GPURipleyDistanceFunctionsSettings`, `GPUPointPatternIndicesSettings`) by
`getGPU<Class>ParameterValues(settings)`; bin counts and the set of outputs are compile-time. The
packers check each value after rounding to float32 (`Math.fround`), because that is what the
shaders read. They throw, with the class name first, when a finite value would overflow to
infinity or a positive distance, lag, bandwidth or guard width would round to 0. The space-time K
packer also throws when strictly ordered bounds or time-window edges would round to the same value:
bounds `[1e7, 0, 1e7 + 0.25, 1]` pack to a zero X extent. Recenter such coordinates on a local
origin. Every caller-owned output view goes in one `output` prop (`GPUVariogramOutput`,
`GPUSpatialCorrelogramOutput`, `GPURipleyOutput`, `GPURipleyEnvelopeOutput`,
`GPURipleyDistanceFunctionsOutput`, `GPUPointPatternIndicesOutput`); only `GPUVariogramOutput`
and `GPUSpatialCorrelogramOutput` carry the optional `overflow` status word.

All five classes (and `GPURipleyEnvelope`) take an optional `spatialContext` and are planar only:
distances and window areas are Euclidean in the units of `positions`. Omitting it means native
planar coordinates; a `longitude-latitude` context throws at construction ("project positions to a
planar frame first"), because degrees give latitude-dependent distances, wrong areas and biased
indices. The resolved context is exposed as `spatialContext`. By default the window is the `bounds`
rectangle. `GPURipley`, `GPUVariogram` and `GPUSpatialCorrelogram` also accept a compile-time
`studyWindow` prepared by `GPUStudyWindowPreparation` from
`@luma-contributors/analysis/gpu-spatial-analysis`. Add the preparation to the same graph before the consumer and use
`bounds: 'study-window'` (`GPUPairStatisticsBounds`) in the per-frame settings. The GPU copies the
parameters and replaces the bounds with the prepared polygon's bbox; the original buffer is unchanged.
The window and points must share a planar spatial context. Windows may contain holes and disjoint
components, but must be valid, without self-intersections or overlapping polygons; consider
`GPUGeometryValidity` before preparation. `studyWindow` exposes the resolved handle or `'rectangle'`.
Passing `bounds: 'study-window'` to a rectangle-mode operator produces NaN statistics; unusable polygon
windows exclude every row. For an irregular study area, using its bbox as the observation window
instead of the polygon overstates the area and can make a pattern look clustered.
`GPUVariogram` and `GPUSpatialCorrelogram` measure distance and reject `local-lambert-equal-area` frames.
Ripley statistics (including distance functions and envelopes) and point-pattern indices combine distance
and window area, so neither local frame is exact: use an azimuthal-equidistant frame only where its area
distortion is negligible, or use a projected CRS.

```ts
import {GPUVariogram, getGPUVariogramParameterValues, fitVariogramModelWithDiagnostics} from '@luma-contributors/analysis/dataframe-analysis';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
declare const positions: import('@luma.gl/graph').GraphDataView<'float32x2'>;
declare const values: import('@luma.gl/graph').GraphDataView<'float32'>;
declare const parameters: GPUParameterBuffer<'float32'>;
declare const semivariances: import('@luma.gl/graph').GraphDataView<'float32'>;
declare const pairCounts: import('@luma.gl/graph').GraphDataView<'uint32'>;
declare const meanDistances: import('@luma.gl/graph').GraphDataView<'float32'>;
declare const pairOverflow: import('@luma.gl/graph').GraphDataView<'uint32'>; // one word
declare const bounds: readonly [number, number, number, number];
declare const distances: Float32Array;
declare const semivarianceReadback: Float32Array;
declare const pairCountReadback: Uint32Array;
declare const overflowReadback: Uint32Array;
graph.add(
  new GPUVariogram({
    positions, values, parameters: parameters.importToGraph(graph), gridSize: [64, 64], lagCount: 24, directionCount: 4,
    output: {semivariances, pairCounts, meanDistances, status: {overflow: pairOverflow}}
  })
);
parameters.write(getGPUVariogramParameterValues({bounds, maximumDistance: 500, sectorOffsetRadians: 0}));
// After submission, read pairOverflow alongside the three bin arrays before fitting.
export function fitCompleteVariogram(
  distances: Float32Array, semivariances: Float32Array, pairCounts: Uint32Array,
  overflowReadback: Uint32Array
) {
  if (overflowReadback.length !== 1 || overflowReadback[0] !== 0) {
    throw new Error('Empirical variogram is incomplete or its overflow readback is missing');
  }
  if (distances.length !== semivariances.length || distances.length !== pairCounts.length ||
      distances.length !== 24 * 4) {
    throw new Error('Read exactly all 96 empirical bins before fitting');
  }
  for (let bin = 0; bin < pairCounts.length; bin++) {
    if (pairCounts[bin] > 0 &&
        (!Number.isFinite(distances[bin]) || !Number.isFinite(semivariances[bin]))) {
      throw new Error(`Empirical bin ${bin} has no finite distance or semivariance`);
    }
  }
  return fitVariogramModelWithDiagnostics({distances, semivariances, pairCounts}, {model: 'spherical'});
}
const fit = fitCompleteVariogram(distances, semivarianceReadback, pairCountReadback, overflowReadback);
const model = fit.model; // nugget, sill, range
const {atBoundary, logRangeBracket, converged} = fit;
```

`maximumDistance` is a distance cutoff for variograms. `GPUPointPatternIndices` instead uses
`indexCellSizeHint`, a positive hint in planar position units that changes search work, never
nearest distances. With points `(0,0)`, `(3,0)` inside bounds `[-1,-1,4,1]`, a hint of 1 still
gives nearest distances `[3,3]`; a variogram `maximumDistance: 1` excludes that pair entirely.
`sectorOffsetRadians` is an undirected sector **edge**, radians counterclockwise from +X modulo
pi. A 90-degree offset is `Math.PI / 2`. `analyticAlternative` accepts `'two-sided'`,
`'greater'` or `'lesser'`.

For ordinary `GPUVariogram` and `GPUSpatialCorrelogram`, IQR and moments now use the same
complete-case spatial population: caller mask nonzero, finite value, finite X/Y inside the
inclusive bounds and, when supplied, inside the polygon study window. Spatially excluded
outliers cannot change precision. An admitted isolated point still enters global scales even
when it contributes no lag pair. Space-time additionally requires both values and time to be
finite, as detailed below. Variogram edge ties compare float32/integer products consistently on
both lag axes. Fit models and weighting modes are validated before execution.

The public dataframe `fitVariogramModel` returns coefficients directly;
`fitVariogramModelWithDiagnostics` returns `.model` plus `atBoundary`, `logRangeBracket` and
`converged`. Read the fitted coefficients on `.model` when requesting diagnostics.

- `GPUVariogram`: polygon windows filter rows only; no area or edge correction is applied.
  Matheron semivariance, pair count, mean lag distance and Cressie-Hawkins robust
  semivariance per lag and optional azimuth sector (`directionCount` equal sectors over `[0, pi)`
  from a per-frame `sectorOffsetRadians`; a pair of coincident points has no direction and is counted
  in sector 0). The robust estimator is
  `(mean |z_i - z_j|^(1/2))^4 / (2 (0.457 + 0.494 / N + 0.045 / N^2))`, the Cressie and Hawkins
  (1980) correction that scikit-gstat evaluates; the `0.045 / N^2` term lowers a one-pair bin by
  4.7 % against the two-term form. Matheron precision: a term `(z_i - z_j)^2 / 2` is scaled by
  `2^24 / u` and rounded to an integer, where `u` is the robust variance `(IQR / 1.349)^2` of the
  included values (exact quartiles), clamped to at most the population variance, so the absolute
  error of a bin mean is at most `u * 2^-25` (never more than `variance * 2^-25`, itself below the
  former `(max - min)^2 * 2^-26`) plus f32 rounding of the term. The former unit
  `2^24 / (0.5 * (max - min)^2)` rounded every typical term to 0 once one outlier or sentinel
  stretched the range (-83% at an isolated outlier of 2e4 standard deviations on 4000 points,
  -100% at 1e5) with no status. Terms above 256 `u` (a difference beyond 22 standard deviations)
  are added exactly in a second 64-bit channel at `2^12` units, with relative error at most
  `2^-21`; the worst-case relative error of `gamma` per bin is `max(u * 2^-25 / gamma, 5e-7 +
  2^-24)`. Capacity is `2^40 u` for the first channel and `2^52 u` for the second. The optional
  `overflow` output is 1 when a sum or clamped term (`2^51 u`) left its range, or a pair count
  reaches `0xffffffff` (including that exact count). Larger counts saturate there; previously the
  count wrapped modulo 2^32 and became the `N_h` weight of the fit.
  The Cressie-Hawkins channel scales with `sqrt(max - min)`. The pass uses 4
  channels per bin (5 with `robustSemivariances`), so `lagCount * directionCount` is at most 512
  (409 with the robust channel). Model fitting is CPU post-processing: `fitVariogramModel` runs on
  the CPU over the read-back bins (the `distances`, `semivariances` and `pairCounts` arrays must
  have equal lengths, as must the four arrays of `fitSpaceTimeVariogramModel`; unequal lengths
  throw, where they used to fit only the shortest prefix), fits spherical, exponential or Gaussian models
  (practical-range convention) with Cressie, pair-count or uniform weights, and should be skipped
  when `output.status.overflow` is 1. The Matheron and Cressie-Hawkins bins, pair counts and mean distances are
  checked against scikit-gstat 1.0.24 (including an embedded 2e4 sentinel) by the committed
  generator `test/oracles/dataframe-variogram.py`, with tolerances derived from the bounds above.
- `GPUSpatialCorrelogram`: polygon windows filter rows only; no area or edge correction is applied.
  Global Moran's I per distance band (`'cumulative'` bands as PySAL
  `DistanceBand` / ArcGIS Incremental Spatial Autocorrelation, or `'annulus'`), with `E[I]`,
  normality or randomization variance (per-frame), z, p, unordered pair counts, and
  `[firstPeakBand, maximumBand]`. Variances lost to f32 cancellation (for example a band holding
  every pair) are NaN. Pair products use the same typical-term unit as the variogram, `2^24 / u`
  with `u` the robust variance (the former `2^24 / max|c|^2` rounded every product to 0, so `I` was
  0, once one outlier or sentinel dominated), so `|I|` errs by at most `2^-25 * u / variance` plus
  `2^-21` of the contribution of products above 256 `u`, which are summed exactly in a second
  channel. The optional `overflow` output is 1 when a product sum or clamped product left its
  range, or a band pair count reaches `0xffffffff` (including that exact count; larger counts saturate).
  Both this scalar and the variogram overflow scalar are rewritten on every encoding, including
  writing zero when no arithmetic loss occurred and every count is below `0xffffffff`;
  callers must inspect the flag independently of whether the neighboring numerical outputs are finite.
  Tiny positive units use exponent-separated term scaling and matching decoding when `2^24 / u`
  is not representable in float32. A bounded term therefore does not raise overflow merely because
  an intermediate reciprocal would overflow; the existing variance floor and channel capacities
  still apply. Adaptive units below the normal float32 range are stored as a scaled value and
  binary exponent, so flushing subnormal storage cannot turn a usable column into a constant.
  Randomization kurtosis scales centered values and their second moment by consistent binary
  powers before forming the dimensionless fourth-moment ratio. This also keeps the reported
  kurtosis, randomization variance and z-score invariant under small amplitude rescaling.
- `GPURipley`: K, L, L - r and an annulus pair-correlation estimate over `radiusCount` radii in a
  rectangle or prepared polygon window, with `'none'`, `'border'` (reduced sample, spatstat `Kest`
  border estimator
  with `lambda = n / A`, so `K = A * numerator / (n * borderCount)`) or `'isotropic'` (Ripley 1977
  weights capped at 100 like spatstat `edge.Ripley` `maxweight`) edge correction
  switchable per frame. Border `K` changed from `lambda = (n - 1) / A` to `n / A` to match spatstat,
  which lowers it by the factor `(n - 1) / n`. The isotropic rectangle formula is exact at every
  radius (checked against brute-force circle sampling), so there is no radius guard; the practical
  limits are variance and the weight cap, which biases `K` low once less than 1% of a circle is
  inside, so keep `maximumDistance` well below the shorter window side (spatstat's default `rmax` is
  a quarter of it). Polygon mode uses the polygon area, the distance to its boundary for border
  correction, and an oriented-edge fan sum for the isotropic circle fraction. Interior circles use
  weight 1 without an edge scan. The same weight cap of 100 applies. The pair correlation is an
  unsmoothed annulus difference and is noisy.
- `GPURipleyEnvelope`: `studyWindow` defaults to `'rectangle'`; pass a valid, non-overlapping
  polygon prepared by `GPUStudyWindowPreparation` in the same planar frame and pack
  `bounds: 'study-window'` to use its bbox, inclusion rule and area. The observed pass and every
  simulation must share this window. Monte Carlo CSR envelopes of K, L or `L - r` (`statistic`, default
  `'l-minus-r'`), with `simulationCount` (at most 199) binomial CSR patterns of the observed
  included count `n`, drawn with Philox from the per-frame `seed` (`getGPURipleyParameterValues({..,
  seed})`, a uint32 stored as a word) and measured by a full `GPURipley` pass each, with the same window,
  radii and edge correction, without CPU readback. Outputs (`output`): `lower` / `upper` (pointwise: the
  `rank`-th smallest / largest simulated value, spatstat `envelope(nrank)`, not simultaneous),
  `globalLower` / `globalUpper` (Ripley 1981; Baddeley et al. 2014: `L(r) = r` plus or minus the
  `rank`-th largest `max_r |L_j(r) - r|`, a simultaneous test of level `rank / (simulationCount + 1)`),
  `theoretical`, and `statistics` `[n, validSimulationCount, observedMaximumDeviation,
  criticalMaximumDeviation, pValue, rank]` with the maximum absolute deviation (MAD) Monte Carlo
  p-value `(1 + #{T_j >= T_obs}) / (s + 1)` when `observedK` (an input, the observed `GPURipley` `output.k`) is given.
  Cost is `simulationCount + 1` times one K pass and about 25 graph nodes per simulation; the graph
  aliases transient buffers across simulations. The simulation index is a GPU counter word, not an
  override constant, so every simulation shares the same pipelines (an override still compiles one
  pipeline per value). Radii where an estimate is NaN are skipped per
  curve, so prefer `'isotropic'` or `'none'` for exact Monte Carlo validity. Cross-type K and
  inhomogeneous K (Kinhom) are not implemented. Polygon simulations uniformly choose an equal-area
  sampler cell, draw a point within it, and reject points outside boundary cells; each rejection
  restarts from the cell draw. Philox uses `(row, simulationIndex, attempt, 0)`. Compile-time
  `maximumAttempts` defaults to 32 and must be an integer in `[1, 256]`. Failed draws are masked;
  optional `output.unplacedCount` is one uint32 word totaling failures across simulations, cleared
  each encoding. An empty sampler or unusable window cannot place points. Undefined curves remain
  NaN. Rectangle draws retain their original attempt-0 sequence. Using `'study-window'` bounds
  without a polygon produces NaN statistics.
  For a valid finite rectangle, uniform placement has no rejection and `unplacedCount` is zero,
  including in the rectangle-only GPU and CPU operation forms. The GPU still writes that zero
  to the caller-owned output on every encoding.
  Both operation forms require the resolved position-row count to equal the declared `rowCount`;
  a mismatch is rejected before computation. Both forms snapshot caller props at construction;
  later mutation does not alter that row contract or deferred execution. Oversized `unplacedCount` views retain their trailing
  caller data: only the first uint32 word is cleared and written.
- `GPURipleyDistanceFunctions`: the distance functions next to `GPURipley` (spatstat `Gest`, `Fest`,
  `Jest`). `G` is the nearest-neighbour distance function of the events, `F` the empty-space function
  over a `referenceGrid` lattice of reference locations, and `J = (1 - G) / (1 - F)` (below 1 is
  clustered, above 1 regular). `parameters` (`getGPURipleyDistanceFunctionsParameterValues`, from a `GPURipleyDistanceFunctionsSettings`) hold `bounds`,
  `maximumDistance` and `edgeCorrection` (`'border'`, the default, `'kaplan-meier'`, `'hanisch'` or
  `'none'`, switchable per frame without recompiling); `gridSize`, `referenceGrid` and `radiusCount` are
  compile-time. Outputs (`output.g`, `output.f`, `output.j`, `output.radii`; at least one of `g`, `f`, `j`) hold NaN where a value is undefined. Integer atomics
  keep it bitwise reproducible; the F pass sums each workgroup's reference locations in workgroup
  memory and adds one exact 64-bit partial per bin and workgroup. `'border'` is spatstat `rs`. `'kaplan-meier'` is spatstat `km`, with hazards
  binned on the radius grid, exact including the censoring of points that have no neighbor within
  `maximumDistance`. `'hanisch'` is spatstat `han` for G and Chiu-Stoyan style weights for F, with 14-bit
  fixed-point weights (relative error about 1e-4); a point with no neighbor within `maximumDistance` but a
  border distance of at least `maximumDistance` enters the denominator at `d = maximumDistance`. That is
  too little weight when the point is uncensored (G too high) and too much when it is censored (G too low:
  two such points beside one close pair give 0.44 where the Hanisch estimator is 1). The estimate is exact
  when every point has a neighbor in range, so set `maximumDistance` to at least the largest
  nearest-neighbor distance when it matters. The corrections are checked against an f64
  oracle written from the spatstat definitions, not against spatstat itself. Alternatively, compile
  `hanischSearch: 'exact'` for a rectangle to include uncensored nearest distances beyond the last
  reported radius in the denominator, retaining the same 14-bit weights and area-ratio cap of 1024.
  Its ring search can visit every included event per unresolved event or reference location;
  polygon windows retain the capped search. K and L (`GPURipley`) have
  no Kaplan-Meier or Hanisch option. The pass uses 8 channels per radius (a `radiusCount` of 256 still fits
  the default 16 KB of workgroup memory exactly). F depends on the reference lattice, so keep it coprime with the
  window so no reference location sits exactly on a radius. `studyWindow` accepts a valid polygon
  prepared by `GPUStudyWindowPreparation` in the same planar frame (default `'rectangle'`). Pack
  `bounds: 'study-window'`: the effective bbox sizes the search and reference lattices, while event
  inclusion and border/Kaplan-Meier censoring use the polygon boundary, including holes. F counts
  only reference cell centres inside the polygon. Polygon Hanisch estimates eroded area as
  `A * insideReferencesWithBorderDistanceAtLeast(r) / insideReferenceCount`, using an integer
  histogram with `4 * radiusCount` sub-slots and linear interpolation at each nearest distance.
  This pass also runs for G-only output. Refine `referenceGrid` to improve the area estimate; the
  weight cap remains 1024. An empty inside reference set gives NaN F and J. `'study-window'` bounds
  in rectangle mode give NaN statistics; the existing rectangle shaders and draws are unchanged.
- Rectangle Hanisch worked fixture: in `[0,10]²`, events (3,3),(3.5,3),(1.5,8),(8.5,8),
  maximumDistance 1, the close pair has nearest distance .5 and weight 100/81 each. The other
  two are truly censored, but default capped search assigns weight 100/64 each. Thus capped
  G(1) = (2*100/81)/(2*100/81+2*100/64) = **64/145 ≈ .4413793103448276**; exact G(1) = **1**.
  Use the following parameter array for both construction strategies, upload it to their
  parameter buffer, and choose `hanischSearch: 'capped'` or rectangle-only `'exact'` on the
  contributor. The default edge correction is border; changing the search alone is insufficient.

```ts
import {getGPURipleyDistanceFunctionsParameterValues} from '@luma-contributors/analysis/dataframe-analysis';
const hanischParameters = getGPURipleyDistanceFunctionsParameterValues({
  bounds: [0, 0, 10, 10], maximumDistance: 1, edgeCorrection: 'hanisch'
});
```

- `GPUPointPatternIndices`: exact nearest neighbour per row (ring search, ties to the smallest row),
  Clark-Evans `[n, observed, expected, R, standardError, z, pValue]`, and quadrat counts with
  `[m, mean, variance, varianceToMeanRatio, chiSquare, degreesOfFreedom, pValue]`. The per-frame
  `clarkEvansCorrection` defaults to `'donnelly'` (Donnelly 1978 edge correction for the rectangular
  window: `expected = 0.5 sqrt(A / n) + (0.0514 + 0.041 / sqrt(n)) P / n`, variance
  `0.0703 A / n^2 + 0.037 P sqrt(A / n^5)`, `P` the perimeter; spatstat uses 0.0412 for the second
  constant). The plain infinite-plane index (`'none'`, `0.5 sqrt(A / n)`, `0.26136 sqrt(A) / n`,
  pointpats / ArcGIS) overstates clustering in small or elongated windows and is no longer the
  default. The per-frame `analyticAlternative` (default `'two-sided'`; `'greater'` tests for regularity in
  Clark-Evans and for clustering in the quadrat chi-square, `'lesser'` the opposite) sets the tail
  of both p-values. The quadrat p-value is the regularized incomplete gamma function at every
  degree of freedom (float32 series or continued fraction; no normal approximation); assume
  expected counts of about 5 or more per quadrat. `GPU_CLARK_EVANS_LENGTH` and
  `GPU_QUADRAT_STATISTICS_LENGTH` are now 7.
  Set compile-time `studyWindow` to a `GPUStudyWindowPreparation` added before this consumer in the
  same graph and planar frame. Polygon windows must be valid (non-overlapping, non-self-intersecting);
  holes and disjoint components are supported. Pack `bounds: 'study-window'` and explicitly choose
  `clarkEvansCorrection: 'guard'` or `'none'`; `'donnelly'` and an omitted correction are rejected.
  A polygon kernel given code 1 directly writes NaN expected distance, standard error, ratio, z and p.
  Packing `'study-window'` without a polygon window leaves invalid bounds and undefined statistics.
  For `'guard'` (also valid for rectangles), set a positive finite per-frame `guardDistance` (slot 7).
  Only included points at least that far from every boundary contribute to the observed mean;
  their neighbors still come from all included points. With `n_focus` focus points, the expectation
  is `0.5 sqrt(A / n)` and standard error is `0.26136 sqrt(A / (n * n_focus))`. Row 0 still reports
  all `n` included points; zero focus points give NaN statistics. Recommend
  `guardDistance >= 2 / sqrt(lambda)`, where `lambda = n / A`.
  Polygon quadrats use a compile-time `quadratSamples` by `quadratSamples` sub-cell-centre lattice
  (default 16 per axis, positive integer, `quadratCount * quadratSamples² <= 16777216`). Sampled
  areas are rescaled to sum to the window area `A`. Zero-area quadrats are dropped; with
  `e_q = n * a_q / A` and `df = m_eff - 1`, rows are
  `[m_eff, n/m_eff, sum((c_q-e_q)²)/df, chiSquare/df, chiSquare, df, pValue]`, where
  `chiSquare = sum((c_q-e_q)²/e_q)`. Equal areas preserve the existing rectangle formula exactly.
  Optional `output.quadratAreas` has one float32 area per row-major quadrat (plain cell areas for
  rectangles) and requires `quadratGrid`, just like counts and statistics. Lattice accuracy depends
  on `quadratSamples`; increase it for narrow boundary features. If a quadrat with no area sample
  holds points (a strip narrower than the lattice), rows 1 to 4 and 6 are NaN, since spatstat
  `quadrat.test` counts every window point and that quadrat has no expected count; previously its
  points stayed in `n` while the quadrat was dropped, so a strip holding 10 of 100 points read as
  chi-square 1.0 (df 1) instead of about 182 (df 2). A point on an interior quadrat edge belongs to
  the upper quadrat on every adapter (the index is decided by products, not by a quotient that can
  compile as `a * (1 / b)`). Counts and fixed-order sums are deterministic.

Measured on an Apple M3 Pro (headless Chromium), `GPUVariogram` with 32 lags including the cell
sort: 100k points with 54M pairs in range in 10 ms; 20k points all-pairs (200M pairs) in 12 ms.

### `GPUPointPatternIndices`

Tests whether points cluster or spread out, from nearest-neighbour distances and a quadrat chi-square over a fixed window.

::lens{scenario=gpu-dataframe.point-pattern-indices rung=result caption="Mean nearest-neighbour distance is 122 m against 497 m expected: ratio 0.25 for 1,500 observations, strongly clustered."}

#### How it computes

{/* lens:GPUPointPatternIndices */}

::lens{scenario=gpu-dataframe.point-pattern-indices rung=datum bind=page datum=0 caption="The single Clark-Evans row: 1,500 points, observed 122.4 m, expected 497.3 m (Donnelly), ratio 0.246."}
::lens{scenario=gpu-dataframe.point-pattern-indices rung=formula bind=page datum=0}
::lens{scenario=gpu-dataframe.point-pattern-indices rung=kernel caption="Look at the Donnelly edge term added to 0.5 sqrt(A / n), and the guard and none alternatives beside it."}
::lens{scenario=gpu-dataframe.point-pattern-indices rung=graph caption="A cell sort, nearest-neighbour and guard reductions, then Clark-Evans and quadrat statistics nodes."}

{/* parameters: GPUPointPatternIndices generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUPointPatternIndices` parameters

Import from `@luma-contributors/analysis/dataframe-analysis`: `GPUPointPatternIndices`.

- Packer: `getGPUPointPatternIndicesParameterValues(settings)`, writes `GPU_POINT_PATTERN_INDICES_PARAMETER_LENGTH` (8) `float32` values.

Settings of `getGPUPointPatternIndicesParameterValues`:

| Field | Type | Description |
|---|---|---|
| `bounds` | `GPUPairStatisticsBounds` | Inclusive `[minX, minY, maxX, maxY]` window. Rows outside it, or with a non-finite coordinate, are excluded. Its area is the study area of the Clark-Evans index and the quadrats tile it. Use 'study-window' for a prepared polygon; writes NaN bounds until GPU window resolution. |
| `indexCellSizeHint` | `number` | Size hint in planar position units of the neighbor-search cells (cells are at least this wide). It changes search work, never distances. A value near the expected nearest-neighbor distance is fastest. |
| `clarkEvansCorrection` (optional) | `'none' \| 'donnelly' \| 'guard'` | Clark-Evans edge correction. Defaults to `'donnelly'` (Donnelly 1978: expected mean distance `0.5 sqrt(A / n) + (0.0514 + 0.041 / sqrt(n)) P / n`, variance `0.0703 A / n^2 + 0.037 P sqrt(A / n^5)`, `P` the window perimeter), because the plain expectation is biased toward apparent clustering for windows with a large perimeter. `'none'` gives the classic `0.5 sqrt(A / n)` with standard error `0.26136 sqrt(A) / n` (pointpats, ArcGIS). Polygon windows require explicit 'guard' or 'none'. 'guard' uses all included neighbors but averages only the interior focus points. |
| `guardDistance` (optional) | `number` | Positive finite guard width (slot 7); required for 'guard'. Recommend >= 2 / sqrt(n / A). |
| `analyticAlternative` (optional) | `GPUAnalyticAlternative` | Tail of the Clark-Evans z-test and the quadrat chi-square test. Defaults to `'two-sided'`. `'greater'` tests for a statistic above its complete-spatial-randomness expectation (Clark-Evans: regular; quadrat chi-square: clustered), `'lesser'` for one below it. The permutation-only tails `'folded'` and `'directed'` throw. |

{/* /parameters */}

### `GPURipley`

Estimates Ripley's K and L over radii from planar points with an edge correction, in a rectangular window (default) or a valid polygon window prepared with `GPUStudyWindowPreparation` and `bounds: 'study-window'`.

::lens{scenario=gpu-dataframe.ripley rung=result caption="L - r is 2,558 m at 200 m and 3,699 m at 2 km: clustering at every radius."}

#### How it computes

{/* lens:GPURipley */}

::lens{scenario=gpu-dataframe.ripley rung=datum bind=page datum=4 caption="Radius 1,000 m: K is 55.7 million m2, so L is 4,210 m and L - r is 3,210 m."}
::lens{scenario=gpu-dataframe.ripley rung=formula bind=page datum=4}
::lens{scenario=gpu-dataframe.ripley rung=kernel caption="Look at the isotropic weight 1 / max(circle fraction, 1/100): pairs near the window edge count more."}
::lens{scenario=gpu-dataframe.ripley rung=graph caption="A cell sort, one pair histogram, then finish and derive nodes for K, L and L - r."}

{/* parameters: GPURipley generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPURipley` parameters

Import from `@luma-contributors/analysis/dataframe-analysis`: `GPURipley`.

- Packer: `getGPURipleyParameterValues(settings)`, writes `GPU_RIPLEY_PARAMETER_LENGTH` (8) `float32` values.

Settings of `getGPURipleyParameterValues`:

| Field | Type | Description |
|---|---|---|
| `bounds` | `GPUPairStatisticsBounds` | Inclusive `[minX, minY, maxX, maxY]` observation window. Rows outside it, or with a non-finite coordinate, are excluded. Its area is the study area of the estimators. Use `'study-window'` with a compile-time polygon window; it writes NaN bounds slots. |
| `maximumDistance` | `number` | Largest radius: radius `b` is `maximumDistance * (b + 1) / radiusCount`. Keep it well below the shorter window side (spatstat's default is a quarter): K is unbiased at any radius, but its variance grows and the isotropic weight cap biases it low once a circle is mostly outside. |
| `edgeCorrection` (optional) | `'none' \| 'border' \| 'isotropic'` | Edge-correction mode. Defaults to `'isotropic'`. |
| `seed` (optional) | `number` | Monte Carlo seed of `GPURipleyEnvelope`, a uint32. `GPURipley` ignores it. Defaults to 0. The same seed reproduces the same simulated patterns exactly. |

{/* /parameters */}

### `GPURipleyDistanceFunctions`

Estimates the nearest-neighbour G, empty-space F and J functions over radii from planar points, border-corrected.

::lens{scenario=gpu-dataframe.ripley-distance-functions rung=result caption="G is 0.82 at 200 m while F is 0.05: events are far closer to each other than random locations are to events."}

#### How it computes

{/* lens:GPURipleyDistanceFunctions */}

::lens{scenario=gpu-dataframe.ripley-distance-functions rung=datum bind=page datum=0 caption="Radius 200 m: G 0.818, F 0.050, so J = (1 - G) / (1 - F) is 0.19."}
::lens{scenario=gpu-dataframe.ripley-distance-functions rung=formula bind=page datum=0}
::lens{scenario=gpu-dataframe.ripley-distance-functions rung=kernel caption="Look at the border test: only events at least r from the window edge enter the G denominator."}
::lens{scenario=gpu-dataframe.ripley-distance-functions rung=graph caption="A cell sort, separate G and F accumulations (event pairs and reference lattice), then a finish node."}

{/* parameters: GPURipleyDistanceFunctions generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPURipleyDistanceFunctions` parameters

Import from `@luma-contributors/analysis/dataframe-analysis`: `GPURipleyDistanceFunctions`.

- Packer: `getGPURipleyDistanceFunctionsParameterValues(settings)`, writes `GPU_RIPLEY_DISTANCE_FUNCTIONS_PARAMETER_LENGTH` (8) `float32` values.

Settings of `getGPURipleyDistanceFunctionsParameterValues`:

| Field | Type | Description |
|---|---|---|
| `bounds` | `GPUPairStatisticsBounds` | Inclusive `[minX, minY, maxX, maxY]` observation window. Events outside it, or with a non-finite coordinate, are excluded. Use 'study-window' for a prepared polygon; its bbox supplies the reference lattice. Only reference centres inside the window count. Rectangle mode with these NaN bounds is undefined. |
| `maximumDistance` | `number` | Largest radius: radius `b` is `maximumDistance * (b + 1) / radiusCount`. |
| `edgeCorrection` (optional) | `GPURipleyDistanceEdgeCorrection` | `'border'` (default) divides by the points farther from the window edge than `r`; `'kaplan-meier'` and `'hanisch'` use every point (see `GPURipleyDistanceFunctions`). |

{/* /parameters */}

### `GPURipleyEnvelope`

Simulates complete spatial randomness with a fixed seed and returns pointwise and global envelopes of L - r with a Monte Carlo p-value.

If no simulated curve supplies a usable deviation, the valid simulation count is zero and
inference (`pValue` and unavailable critical/envelope values) is NaN. The observed deviation and
valid count remain readable; zero usable simulations do not establish a probability of 1.

::lens{scenario=gpu-dataframe.ripley-envelope rung=result caption="Observed L - r reaches 3,699 m at 2 km; 39 seeded simulations never leave plus or minus 41 m."}

#### How it computes

{/* lens:GPURipleyEnvelope */}

::lens{scenario=gpu-dataframe.ripley-envelope rung=datum bind=page datum=9 caption="Radius 2,000 m: simulations span -19.7 to 30.2 m, the observed L - r is 3,698.8 m, p is 0.025."}
::lens{scenario=gpu-dataframe.ripley-envelope rung=formula bind=page datum=9}
::lens{scenario=gpu-dataframe.ripley-envelope rung=kernel caption="Look at the lower and upper reductions over finite simulations and the rank-based p-value."}
::lens{scenario=gpu-dataframe.ripley-envelope rung=graph caption="39 seeded simulations, each with its own cell keys and sort, feeding the envelope reduction."}

{/* parameters: GPURipleyEnvelope generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPURipleyEnvelope` parameters

Import from `@luma-contributors/analysis/dataframe-analysis`: `GPURipleyEnvelope`.

- Packer: `getGPURipleyParameterValues(settings)`, writes `GPU_RIPLEY_PARAMETER_LENGTH` (8) `float32` values.
  - Input port: `parameters`.

Settings of `getGPURipleyParameterValues`:

| Field | Type | Description |
|---|---|---|
| `bounds` | `GPUPairStatisticsBounds` | Inclusive `[minX, minY, maxX, maxY]` observation window. Rows outside it, or with a non-finite coordinate, are excluded. Its area is the study area of the estimators. Use `'study-window'` with a compile-time polygon window; it writes NaN bounds slots. |
| `maximumDistance` | `number` | Largest radius: radius `b` is `maximumDistance * (b + 1) / radiusCount`. Keep it well below the shorter window side (spatstat's default is a quarter): K is unbiased at any radius, but its variance grows and the isotropic weight cap biases it low once a circle is mostly outside. |
| `edgeCorrection` (optional) | `'none' \| 'border' \| 'isotropic'` | Edge-correction mode. Defaults to `'isotropic'`. |
| `seed` (optional) | `number` | Monte Carlo seed of `GPURipleyEnvelope`, a uint32. `GPURipley` ignores it. Defaults to 0. The same seed reproduces the same simulated patterns exactly. |

{/* /parameters */}

### `GPUSpatialCorrelogram`

Computes Moran's I per cumulative distance band for a station value, with z-scores and p-values under randomization.

::lens{scenario=gpu-dataframe.spatial-correlogram rung=result caption="Moran's I is 0.97 in the first 25 km band and 0.82 by 200 km: rainfall is strongly autocorrelated."}

#### How it computes

{/* lens:GPUSpatialCorrelogram */}

::lens{scenario=gpu-dataframe.spatial-correlogram rung=datum bind=page datum=0 caption="First band (up to 25 km): I is 0.971 against E[I] = -0.0008, z is 40.9, from 1,752 pairs."}
::lens{scenario=gpu-dataframe.spatial-correlogram rung=formula bind=page datum=0}
::lens{scenario=gpu-dataframe.spatial-correlogram rung=kernel caption="Look at the band test on pair distance and the product (x_i - mean)(x_j - mean) summed per band."}
::lens{scenario=gpu-dataframe.spatial-correlogram rung=graph caption="Moments, a cell sort, the band pair histogram and a fourth-moment sum, then finish and publish nodes."}

{/* parameters: GPUSpatialCorrelogram generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUSpatialCorrelogram` parameters

Import from `@luma-contributors/analysis/dataframe-analysis`: `GPUSpatialCorrelogram`.

- Packer: `getGPUSpatialCorrelogramParameterValues(settings)`, writes `GPU_SPATIAL_CORRELOGRAM_PARAMETER_LENGTH` (8) `float32` values.

Settings of `getGPUSpatialCorrelogramParameterValues`:

| Field | Type | Description |
|---|---|---|
| `bounds` | `GPUPairStatisticsBounds` | Inclusive `[minX, minY, maxX, maxY]` extent. Rows outside it, or with a non-finite coordinate or value, are excluded. The extent also sizes the neighbor-search lattice. Use `'study-window'` with a compile-time polygon window; it writes NaN bounds slots. |
| `maximumDistance` | `number` | Upper distance of the last band. Band `b` ends at `maximumDistance * (b + 1) / bandCount`. |
| `varianceAssumption` (optional) | `GPUSpatialCorrelogramVarianceAssumption` | Variance assumption of the z-scores and p-values. Defaults to `'randomization'`. |

{/* /parameters */}

### `GPUVariogram`

Computes the empirical semivariogram of a station value over distance lags, with pair counts and mean pair distances.

::lens{scenario=gpu-dataframe.variogram rung=result caption="Semivariance rises from 546 at the first 25 km lag to 2,567 at 300 km: nearby stations see similar rainfall."}

#### How it computes

{/* lens:GPUVariogram */}

::lens{scenario=gpu-dataframe.variogram rung=datum bind=page datum=3 caption="Lag 3 (75 to 100 km): semivariance 962.6 from 6,216 pairs at a mean separation of 87.9 km."}
::lens{scenario=gpu-dataframe.variogram rung=formula bind=page datum=3}
::lens{scenario=gpu-dataframe.variogram rung=kernel caption="Look at the lag index min(floor(d / D * H), H - 1) and the squared value difference."}
::lens{scenario=gpu-dataframe.variogram rung=graph caption="Moments and a unit-variance estimate, a cell sort, the pair-histogram pass and a finish node."}

{/* parameters: GPUVariogram generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUVariogram` parameters

Import from `@luma-contributors/analysis/dataframe-analysis`: `GPUVariogram`.

- Packer: `getGPUVariogramParameterValues(settings)`, writes `GPU_VARIOGRAM_PARAMETER_LENGTH` (8) `float32` values.

Settings of `getGPUVariogramParameterValues`:

| Field | Type | Description |
|---|---|---|
| `bounds` | `GPUPairStatisticsBounds` | Inclusive `[minX, minY, maxX, maxY]` extent. Rows outside it, or with a non-finite coordinate or value, are excluded. The extent also sizes the neighbor-search lattice. Use `'study-window'` with a compile-time polygon window; it writes NaN bounds slots. |
| `maximumDistance` | `number` | Largest lag: pairs farther apart are ignored. Lag bins split `[0, maximumDistance]`. |
| `sectorOffsetRadians` (optional) | `number` | Start of the first direction sector, in radians counterclockwise from +x. Sector `s` holds pairs whose undirected direction (taken modulo pi) minus this offset, again modulo pi, lies in `[s * pi / directionCount, (s + 1) * pi / directionCount)`. Ignored when `directionCount` is 1. Defaults to 0. |

{/* /parameters */}

## `GPUCompositeScore`

Scales indicator columns to [0, 1], flips higher-is-worse directions and combines them into one weighted score per row.

::lens{scenario=gpu-dataframe.composite-score rung=result caption="788 of 791 tracts are scored; scores span 0.03 in tract 657 to 0.79 in tract 603 (CPU check)."}

`GPUCompositeScore` builds a composite indicator per row from up to 16 indicator columns
(row-major `row * indicatorCount + column`), as CARTO's composite-indicator procedures and ArcGIS
index builders do: scale each column, flip "higher is worse" indicators, and combine them. Weights,
directions, the scaler and the aggregation live in a parameter buffer, so weight sliders recolor
every row without rebuilding the graph.

Weights must remain finite in float32; geometric-mean weights are nonnegative. Epsilon must
remain finite and strictly positive after packing. Rejection leaves reusable targets unchanged.

```ts
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
import {GPU_COMPOSITE_SCORE_PARAMETER_LENGTH, GPUCompositeScore, getGPUCompositeScoreParameterValues} from '@luma-contributors/analysis/dataframe-analysis';
declare const indicators: import('@luma.gl/graph').GraphDataView<'float32'>;
declare const score: import('@luma.gl/graph').GraphDataView<'float32'>;
declare const columnStatistics: import('@luma.gl/graph').GraphDataView<'float32'>;
declare const loadings: import('@luma.gl/graph').GraphDataView<'float32'>;
declare const principalComponentSummary: import('@luma.gl/graph').GraphDataView<'float32'>;
const parameters = new GPUParameterBuffer(device, {
  id: 'composite-parameters',
  format: 'float32',
  length: GPU_COMPOSITE_SCORE_PARAMETER_LENGTH
});
graph.add(
  new GPUCompositeScore({
    indicators, // rowCount * 4 float32
    indicatorCount: 4,
    parameters: parameters.importToGraph(graph),
    enableRank: true,
    enablePrincipalComponent: true,
    output: {score, columnStatistics, loadings, principalComponentSummary}
  })
);
parameters.write(
  getGPUCompositeScoreParameterValues({
    scaler: 'rank',
    aggregation: 'weighted-sum',
    weights: [1, 2, 0.5, 1],
    directions: [1, -1, 1, 1]
  })
);
```

- A row is included when its mask is non-zero and every indicator is finite; excluded rows write
  NaN and do not enter the statistics.
- Scalers: `'min-max'` (`(x - min) / (max - min)`, 0 for a constant column), `'z-score'`
  (population standard deviation, 0 for a constant column), `'rank'` (percentile rank
  `r / (n - 1)` with tied ranks averaged, exact; needs `enableRank`). A negative direction flips the
  scaled value (`1 - s`, or `-s` for z-scores).
- Aggregations: `'weighted-sum'` (`sum(w * s) / sum(|w|)`), `'weighted-geometric-mean'`
  (`exp(sum(w * ln(s + epsilon)) / sum(w))`; pair it with min-max or rank; the packer refuses a
  negative weight here, use a direction to flip an indicator),
  `'principal-component'` (projection of direction-adjusted z-scores on the first eigenvector of
  their correlation matrix; needs `enablePrincipalComponent`; weights are ignored).
- The first component of two indicators is the closed form: `(1, 1) / sqrt(2)` with eigenvalue
  `1 + r` for a correlation `r >= 0`, `(1, -1) / sqrt(2)` with `1 - r` for `r < 0` (at `r = 0`
  the two tie and `(1, 1)` is taken). For more indicators it is the last eigenvector of a cyclic
  Jacobi decomposition in one thread, which separates close eigenvalues; when the largest
  eigenvalue is repeated, any unit vector of its eigenspace is a first component and the one
  returned is fixed by the rotation order. No power iteration is used.
  The loadings are signed so that they sum to a positive value; when they sum to zero the first
  loading that ties the largest magnitude is positive. `principalComponentSummary` holds
  `[eigenvalue, explainedVarianceRatio, residual]`; the ratio divides by the number of
  non-constant columns.
- Outputs: `score` (required), `scaled`, `columnStatistics` (`[minimum, maximum, mean, standardDeviation]` per column),
  `loadings`, `principalComponentSummary`.
- Centering chooses a close same-sign endpoint as reference (endpoints within a factor of two;
  minimum for close positive values,
  maximum for close negative values), zero for a wide or sparse column, and the endpoint for a
  constant column. It accumulates centered totals in **96-bit fixed-point integers**, divides by
  the exact count, and retains the mean offset separately. Scores use the
  separate center rather than the rounded public mean: `[10000000,10000001]` has population
  deviation 0.5 and z-scores `[-1,1]`, although its public float32 mean rounds to 10000000.
  For 65,535 zeros followed by 10,000,000, the mean is 152.587890625.

- Determinism: min and max use u32 order-key atomics; means and centered (co)variances are
  two-pass fixed-order sums over 256-row tiles, merged by one 256-lane workgroup per output
  (strided tiles, then a fixed binary tree); ranks use one stable radix
  sort per column plus a binary search per tie run. No float atomics.
- `compile-time`: row count, `indicatorCount`, `enableRank`, `enablePrincipalComponent`, the mask
  and which outputs exist. `per frame`: scaler, aggregation, weights, directions, epsilon.

#### How it computes

{/* lens:GPUCompositeScore */}

::lens{scenario=gpu-dataframe.composite-score rung=datum bind=page datum=603 caption="Tract 603 scores highest, 0.79: poverty 67.8, unemployment 37.1, uninsured 24.5 and 62.7 percent without a vehicle (CPU check)."}
::lens{scenario=gpu-dataframe.composite-score rung=formula bind=page datum=603}
::lens{scenario=gpu-dataframe.composite-score rung=kernel caption="Look at the min-max step (x - min) / (max - min), the direction flip and the division by the sum of absolute weights."}
::lens{scenario=gpu-dataframe.composite-score rung=graph caption="Per-column extrema and moment reductions merged in tile order, then a score node reading the parameter buffer."}

{/* parameters: GPUCompositeScore generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPUCompositeScore` parameters

Import from `@luma-contributors/analysis/dataframe-analysis`: `GPUCompositeScore`.

- Packer: `getGPUCompositeScoreParameterValues(settings)`, writes `GPU_COMPOSITE_SCORE_PARAMETER_LENGTH` `float32` values.

Settings of `getGPUCompositeScoreParameterValues`:

| Field | Type | Description |
|---|---|---|
| `scaler` (optional) | `'min-max' \| 'z-score' \| 'rank'` | Per-indicator scaler. Defaults to `'min-max'`. |
| `aggregation` (optional) | `'weighted-sum' \| 'weighted-geometric-mean' \| 'principal-component'` | How scaled indicators combine into the score. Defaults to `'weighted-sum'`. |
| `weights` | `ArrayLike<number>` | One weight per indicator (at most 16). Missing trailing weights are 0. A negative weight penalizes an indicator in `'weighted-sum'`; `'weighted-geometric-mean'` refuses negative weights (a weighted geometric mean needs `w >= 0`): use `directions` to flip an indicator. |
| `directions` (optional) | `ArrayLike<number>` | One direction per indicator: a negative value means "higher is worse" and flips the scaled value (`1 - s` for min-max and rank, `-s` for z-scores). Defaults to all `+1`. |
| `epsilon` (optional) | `number` | Positive shift added before the logarithm of the geometric mean. Defaults to `1e-6`. |

{/* /parameters */}

## `GPUInequality`

Parallel and reference routes support at most 2,147,483,648 rows, subject to sort and device
limits. `mode: 'reference'` preserves the serial oracle.

Parallel reductions are the default; mode reference retains the serial per-zone oracle. Lorenz knots
and output bindings retain their existing limits.

Computes weighted Gini and other inequality indices per zone from a value column, with Lorenz knots and a pooled summary.

::lens{scenario=gpu-dataframe.inequality rung=result caption="Within community areas the income Gini runs from 0.01 (area 37, two tracts) to 0.31 (area 28); pooled it is 0.33 (CPU check)."}

`GPUInequality` computes per-zone income inequality indices from a column of non-negative values: Gini, Theil T and L, Atkinson, Hoover, Palma, Lorenz curve knots, plus the between/within-zone decomposition of Theil T and pooled statistics.

Both execution modes preserve the Atkinson signal near `epsilon = 1`: for `0 < abs(1 - epsilon) <= 0.01`, centered positive-income terms and stable logarithmic divided differences avoid rounding the power mean to one before exponentiation. For `epsilon < 1`, zero incomes contribute a separately accumulated population-weight correction. If a positive income-to-mean quotient underflows, the centered Atkinson term uses `log(income) - log(mean)` instead. Exactly one keeps the geometric-mean formula. In a positive-mean population, a positive-weight zero income gives Atkinson **1** at `epsilon >= 1`: the equally distributed equivalent income is zero. An all-zero population remains undefined. A zero-weight row is excluded and cannot trigger this boundary rule. Float32 Atkinson comparisons allow absolute error `5e-4`.
Both GPU modes use weight fractions in the centered and geometric-mean reductions for zones with
finite total weight above `2^112`. This avoids overflow in the weighted terms. The zone weight
and weighted-income totals still use float32, whose largest finite value is approximately
`3.4028235e38`.

Inputs: `values` (float32), `zoneIds` (uint32; `0xffffffff` or any id `>= zoneCount` skips the row), `zoneCount` (at most `getGPUInequalityMaximumZoneCount(device)`, 3,050,402 on a default device) and `lorenzKnotCount` (compile-time; `zoneCount * lorenzKnotCount` at most `getGPUInequalityMaximumLorenzKnotTotal(device)`, 2^25 on a default device), optional population `weights` (float32) and `mask` (uint32), and a `parameters` view written with `getGPUInequalityParameterValues({epsilon, palmaTopShare, palmaBottomShare})`. Changing the parameters never rebuilds or recompiles the graph. A row is included when its mask is non-zero, its zone is valid, its value is finite and non-negative and, with weights, its weight is finite and positive. With `w` the weight (1 without weights), `W = sum w`, `S = sum w x` and `mu = S / W`:

| Output | Definition |
| --- | --- |
| `gini` | `1 - sum (P_i - P_(i-1)) (L_i + L_(i-1))`, trapezoid area under the Lorenz curve of the value-sorted rows; equals `2 sum i x_(i) / (n S) - (n + 1) / n` for unit weights, `(n - 1) / n` for one holder of everything |
| `theilT` | `(1 / W) sum w r ln r`, `r = x / mu`, `0 ln 0 = 0` |
| `theilL` | `-(1 / W) sum w ln r`; NaN when the zone contains a zero (zeros are never silently dropped) |
| `atkinson` | `1 - ((1 / W) sum w r^(1 - e))^(1 / (1 - e))`, `1 - exp((1 / W) sum w ln r)` at `e = 1`; positive-weight zeros contribute normally for `e < 1`, give 1 for `e >= 1` when `mu > 0`; all-zero populations give NaN; `e` in `[0, 32]` per frame |
| `hoover` | `sum w abs(x - mu) / (2 S)` |
| `palma` | `(1 - L(1 - top)) / L(bottom)`, cuts per frame (0.1 and 0.4 by default); NaN when `L(bottom) = 0` |
| `mean`, `counts` | weighted mean, included row count per zone (`counts` is `uint32`) |
| `lorenzKnots` | `L(k / (K - 1))`, row-major `zone * K + k`, `L(0) = 0`, `L(1) = 1` |
| `globalSummary` | slots `GPU_INEQUALITY_GLOBAL_SUMMARY`: `TOTAL_THEIL_T`, `BETWEEN_THEIL_T`, `WITHIN_THEIL_T`, `GINI`, `COUNT`, `MEAN`, `TOTAL_WEIGHT`, `TOTAL_INCOME` |

The Lorenz curve is linear inside each row (a row's income spreads over its population share), so a Palma cut or knot inside a row takes a fractional share of that row. Empty zones, and zones with zero total income, report NaN for every index (zero-income zones keep `mean = 0`). The Theil decomposition is `between = sum_g s_g ln(s_g / p_g)` and `within = sum_g s_g T_g` with income share `s_g` and population share `p_g`; the pooled Theil T is computed independently from all included rows, so `total = between + within` is a real check.

The host `computeInequalityOnCPU` result exposes `weightedGini`; its deprecated `gini` property
is the same array. For sorted incomes it sums each adjacent gap times the population weight
on its left times the population weight on its right, then divides by `W * S`. The two population
sides are accumulated directly, avoiding subtraction of nearly equal Lorenz areas or a rounded
total-minus-prefix weight. This preserves tiny positive coefficients and yields exactly zero for
constant positive incomes. For `[0,1]` with weights `[1,2**58]`, Gini is `1/(2**58+1)`, published
as float32 `3.469446951953614e-18`. Both GPU modes also publish zone and pooled Gini through
adjacent-gap reductions with populations accumulated directly on each side. The reference route
replaces its intermediate area-complement value before publication. Read the CPU array as
`weightedGini`, the graph port as `output.gini`, and the pooled GPU slot at
`GPU_INEQUALITY_GLOBAL_SUMMARY.GINI`.

Both GPU modes scale adjacent incomes to a common binary exponent before subtracting, and sum
weighted-income products at a common exponent before forming the Gini quotient. Thus normal
inputs can retain a normal Gini even when their raw difference or weighted product would
underflow: unit weights with incomes `[2**-110, 2**-110*(1+2**-23)]` give approximately
`2.9802320611338473e-8`; incomes `[0,2**-80]` with weights `[2**-80,2**-80]` give `0.5`.
At epsilon 3 the latter positive-mean population gives Atkinson 1; an all-zero population
remains undefined.

Raw inequality, color-scale and bivariate output selections require at least one actual graph
view. `{}`, undefined-only selections and nested empty objects are not output views. Color
terminals are `colors`, `classIndices`, `classCounts`; bivariate terminals are `classIds`,
`colors`, `classCounts`. Named inequality operations use existing logical ports such as
`gini`, not the host-only `weightedGini`. Select at least one destination; every selected name
must be nonempty, unique and different from every input name, including optional weights/mask.
Both CPU and GPU inequality operation constructors reject empty selections, unsupported logical
ports (including `weightedGini`), empty optional input names, duplicate destinations and input/output
collisions before execution. They use the shared graph-name contract for selected destinations.

Algorithm and cost: rows are sorted by value, re-keyed by zone and stably sorted, so each zone
is a value-ordered segment. The default parallel route builds bounded 256-row Lorenz prefixes,
computes row-parallel terms and reduces them by segment. `mode: 'reference'` preserves the two
serial per-zone walks for indices other than Gini; a dominant zone serializes those walks.
Both modes publish zone and pooled Gini through parallel adjacent-gap reductions. The
parallel summation order can change the last float32 bits; sums and published statistics remain
float32. Gini uses bounded 256-row forward/reverse scan tiles and fixed-order segment reductions;
the other global fields use cooperative workgroup reductions.

Atkinson epsilon is in `[0,32]` before and after packing: explicit zero is legal, but a positive
value rounded to zero is rejected. Each Palma share remains in `(0,1]` after float32 packing.
Invalid settings leave reusable targets unchanged.

#### How it computes

{/* lens:GPUInequality */}

::lens{scenario=gpu-dataframe.inequality rung=datum bind=page datum=27 caption="Community area 28 has the highest Gini, 0.307, over 20 tracts with a population-weighted mean income of 85,684 dollars (CPU check)."}
::lens{scenario=gpu-dataframe.inequality rung=formula bind=page datum=27}
::lens{scenario=gpu-dataframe.inequality rung=kernel caption="Look at adjacent sorted income gaps weighted by the directly accumulated populations on each side, divided by total population times total income."}
::lens{scenario=gpu-dataframe.inequality rung=graph caption="Value and zone sorts feed zone statistics and bidirectional Gini gap reductions; the pooled summary has its own gap reduction."}

{/* parameters: GPUInequality generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPUInequality` parameters

Import from `@luma-contributors/analysis/dataframe-analysis`: `GPUInequality`.

- Packer: `getGPUInequalityParameterValues(settings?)`, writes `GPU_INEQUALITY_PARAMETER_LENGTH` (4) `float32` values.

Settings of `getGPUInequalityParameterValues`:

| Field | Type | Description |
|---|---|---|
| `epsilon` (optional) | `number` | Atkinson inequality aversion, `0` to `GPU_INEQUALITY_MAXIMUM_EPSILON`. `1` selects the geometric-mean form. Defaults to 1. |
| `palmaTopShare` (optional) | `number` | Population share of the Palma numerator, in `(0, 1]`. Defaults to 0.1. |
| `palmaBottomShare` (optional) | `number` | Population share of the Palma denominator, in `(0, 1]`. Defaults to 0.4. |

{/* /parameters */}

## `GPUTemporalReduction`

All `occupiedSlots.status` destinations, including optional `candidateOverflow`, are writable
outputs and must not overlap other outputs or share buffers with inputs such as `cellIds`.

Reduce timestamped rows to per cell and time-bucket counts, extremes, first and last values.

::lens{scenario=gpu-dataframe.temporal-reduction rung=result caption="251 of 530 taxon-week slots are occupied; the busiest holds 36 bird observations in week 18 (CPU check)."}

Reduces timestamped rows into one record per (cell, coarse time bucket) so a scrubbing view draws one feature per occupied slot instead of one per row. Bucket `b` covers `[origin + b * width, origin + (b + 1) * width)`; the slot is `cell * bucketCount + bucket`.

```ts
import {GPUTemporalReduction, getGPUTemporalReductionParameterValues} from '@luma-contributors/analysis/dataframe-analysis';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
declare const cellIds: import('@luma.gl/graph').GraphDataView<'uint32'>;
declare const timestamps: import('@luma.gl/graph').GraphDataView<'float32'>;
declare const values: import('@luma.gl/graph').GraphDataView<'float32'>;
declare const mask: import('@luma.gl/graph').GraphDataView<'uint32'>;
declare const bucketParameters: GPUParameterBuffer<'float32'>;
declare const cellCount: number;
declare const bucketCount: number;
declare const counts: import('@luma.gl/graph').GraphDataView<'uint32'>;
declare const minimums: import('@luma.gl/graph').GraphDataView<'float32'>;
declare const maximums: import('@luma.gl/graph').GraphDataView<'float32'>;
declare const first: import('@luma.gl/graph').GraphDataView<'float32'>;
declare const last: import('@luma.gl/graph').GraphDataView<'float32'>;
declare const ids: import('@luma.gl/graph').GraphDataView<'uint32'>;
declare const count: import('@luma.gl/graph').GraphDataView<'uint32'>;
declare const overflow: import('@luma.gl/graph').GraphDataView<'uint32'>;
declare const width: number;
declare const origin: number;
graph.add(
  new GPUTemporalReduction({
    cellIds, // uint32 per row, 0xffffffff skips
    timestamps, // float32 relative times, or uint32x2 Int64 words
    values, // float32 per row
    mask, // optional uint32
    parameters: bucketParameters.importToGraph(graph), // [origin, width]
    cellCount,
    bucketCount,
    output: {counts, minimums, maximums, first, last, occupiedSlots: {ids, status: {count, overflow}}}
  })
);
bucketParameters.write(getGPUTemporalReductionParameterValues(origin, width)); // no recompile
```

- Outputs are dense columns of `cellCount * bucketCount` rows. Empty slots have count 0 and NaN minimums, maximums, first and last. `occupiedSlots.ids` lists occupied slots in ascending order (`cell = Math.floor(id / bucketCount)`, `bucket = id % bucketCount`), clamped to its length with `overflow` set when more are occupied.
Decode a nontrivial occupied slot with integer division:

```ts
const slot = 5;
const bucketCount = 3;
const cell = Math.floor(slot / bucketCount); // 1
const bucket = slot % bucketCount; // 2
```

`GPUValueCube` uses the same rule with `periodCount`: `cell = Math.floor(slot / periodCount)`,
`period = slot % periodCount`. Ordinary division would produce a fractional array index.

- `first` and `last` are the values at the earliest and latest timestamp in the slot; ties are broken by the lower row index.
- Rows with a bucket outside `[0, bucketCount)`, a skipped cell, a zero mask, or a NaN value or time are dropped, not clamped.
- Exact Int64 times: pass `uint32x2` words and write `getGPUTemporalReductionWordParameterValues(origin, width)` (integer origin, integer width in `[1, 2^64 - 1]`, so hourly buckets of nanosecond data use `3_600_000_000_000n`). Bucket assignment and the first and last time inside a bucket are exact integer arithmetic; the bucket index must still be below `bucketCount`.
- Determinism: u32 atomics only, so results do not depend on thread order. In float mode bucket edges are defined by correctly rounded products (`fround(b * width) <= fround(t - origin) < fround((b + 1) * width)`), so assignment is identical on every adapter; at most 2^20 buckets.
- Logical admission: `cellCount * bucketCount <= 4,294,967,294` at construction. Separately, each slot binding must fit the device: `cellCount * bucketCount` is at most 33554432 slots with float32 times and 16777216 slots with `uint32x2` times on a default device (128 MiB bindings). Word mode keeps the in-bucket time of each slot as two 32-bit words (8 bytes per slot instead of 4), so its maximum is half the float one. Larger slot counts are refused at declaration.
- Inputs must be single packed views; chunked vectors are not supported.
- `sorted: true` (default false, compile-time) declares `timestamps` non-decreasing and non-NaN. The rows with bucket in `[0, bucketCount)` are then one contiguous range found by two binary searches (same bucket rules as above: the correctly rounded `fround(t - origin) < fround(bucketCount * width)` test in float mode, the exact quotient in word mode), and the classify, extreme and tie kernels are dispatched indirectly over just that range, with the first row read from a GPU word. A scrub costs O(range), not O(rows), and the result is bit-identical to the unsorted path on sorted input. It adds one `uint32` scratch column of `rows` entries. The order is not checked; unsorted input silently reduces the wrong rows.

#### How it computes

{/* lens:GPUTemporalReduction */}

::lens{scenario=gpu-dataframe.temporal-reduction rung=datum bind=page datum=133 caption="Slot 133 is insects in week 27: 32 observations, research grade mixed (minimum 0, maximum 1; CPU check)."}
::lens{scenario=gpu-dataframe.temporal-reduction rung=formula bind=page datum=133}
::lens{scenario=gpu-dataframe.temporal-reduction rung=kernel caption="Look at the bucket test: fround(b * width) <= fround(t - origin) < fround((b + 1) * width)."}
::lens{scenario=gpu-dataframe.temporal-reduction rung=graph caption="Init, classify, extreme, row-tie and finish nodes, then compaction and publish of occupied slots."}

{/* parameters: GPUTemporalReduction generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPUTemporalReduction` parameters

Import from `@luma-contributors/analysis/dataframe-analysis`: `GPUTemporalReduction`.

- Packer: `getGPUTemporalReductionParameterValues(origin, width)`, writes `GPU_TEMPORAL_REDUCTION_PARAMETER_LENGTH` (2) `float32` values.
  - `origin` (`number`): Start of bucket 0, in the unit of the timestamps.
  - `width` (`number`): Bucket width, in the unit of the timestamps.
- Packer: `getGPUTemporalReductionWordParameterValues(origin, width)`, writes `GPU_TEMPORAL_REDUCTION_WORD_PARAMETER_LENGTH` (4) `uint32` values.
  - `origin` (`number | bigint`): Integer start of bucket 0, any signed 64-bit value.
  - `width` (`number | bigint`): Integer bucket width in `[1, 2^64 - 1]`. A `number` must be a safe integer to be exact; pass a `bigint` beyond 2^53.

{/* /parameters */}

## `GPUCalendarBuckets`

Decode epoch timestamps (second, millisecond, microsecond or nanosecond) into calendar columns and a weekday-by-hour count matrix.

::lens{scenario=gpu-dataframe.calendar-buckets rung=result caption="Saturday 11:00 is the busiest weekday-hour bucket with 48 of 1,500 observations (CPU check)."}

Decodes epoch timestamps (milliseconds by default, see `unit`) into calendar columns (year, month, day of month, hour, minute,
weekday, day of year, ISO week, ISO week-year, quarter) and an hour by weekday count matrix, so
time-of-day and weekday charts and calendar bucket keys need no per-row `Date` on the CPU. The
columns are plain `uint32` and `sint32` outputs for heatmaps and charts. They do not feed
`GPUTemporalReduction` (which buckets by fixed width) or `GPUCellAggregation` keys (which accept
only Quadbin or H3 cells).

```ts
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
import {GPUCalendarBuckets, getGPUCalendarBucketsParameterValues} from '@luma-contributors/analysis/dataframe-analysis';
declare const timestamps: import('@luma.gl/graph').GraphDataView<'uint32x2'>;
declare const hour: import('@luma.gl/graph').GraphDataView<'uint32'>;
declare const weekday: import('@luma.gl/graph').GraphDataView<'uint32'>;
declare const hourWeekdayCounts: import('@luma.gl/graph').GraphDataView<'uint32'>;
const parameters = new GPUParameterBuffer(device, {id: 'calendar', format: 'sint32', length: 2});
graph.add(
  new GPUCalendarBuckets({
    timestamps, // GraphDataView<'uint32x2'>, Int64 epoch words, e.g. from getInt64TimeWords
    unit: 'nanosecond', // the unit of the column; defaults to 'millisecond'
    parameters: parameters.importToGraph(graph),
    output: {hour, weekday, hourWeekdayCounts}
  })
);
parameters.write(getGPUCalendarBucketsParameterValues(-300 /* UTC-5 */, 0 /* Monday first */));
```

- `timestamps`: signed 64-bit two's complement epoch times as `uint32x2` `(low, high)` rows;
  times before 1970 are fine. Optional `mask` (`uint32`, zero makes the row invalid).
- `unit` (optional, compile-time): `'second'`, `'millisecond'` (default), `'microsecond'` or
  `'nanosecond'`, the `ArrowAnalyticsTimeUnit` of the column. Set it to the unit the Arrow upload kept
  (a pandas `timestamp[ns]` column needs `'nanosecond'`); the wrong unit decodes to wrong dates
  without an error. Rows are floored to whole milliseconds first (a negative sub-millisecond time
  lands in the earlier millisecond), then decoded as before. Seconds beyond +-2^52 are invalid. The
  `timeZone` transitions stay epoch milliseconds whatever the unit.
- Host calendar display: `getTimestampCalendarMilliseconds(ticks, arrowUnit)` floors exact ticks to
  the containing millisecond before conversion to Number, matching this decode and the day rendered by
  `formatTimestampTicks`. Arrow units are `0` seconds, `1` milliseconds, `2` microseconds and `3`
  nanoseconds: `-1n` in microseconds or nanoseconds returns `-1` millisecond (1969-12-31), while zero
  and positive sub-millisecond ticks return `0`. `getTimestampPresentationMilliseconds` retains
  truncation toward zero for numeric callers. Neither helper changes source ticks or exact export text.
  The calendar helper accepts source instants only within Date's inclusive
  `[-8640000000000000, 8640000000000000]` millisecond range and throws `RangeError` outside it,
  before conversion to Number. Every millisecond in that range is exactly representable. The GPU
  decoder supports a wider range; group those ticks into days in `bigint` first, using floor
  division, rather than narrowing milliseconds. For microseconds, for example:
  `day = ticks >= 0n ? ticks / 86400000000n : -((-ticks + 86399999999n) / 86400000000n)`.
- `parameters` (`sint32`, 2 elements, per frame, no rebuild): fixed UTC offset in minutes and the first
  day of the week for `weekday` and the matrix (0 = Monday, the ISO default, to 6 = Sunday). ISO week
  numbers always start on Monday.
- Time zones: the per-frame offset is a fixed offset. Daylight saving time is supported by either
  the optional per-row `utcOffsets` column (`sint32` minutes, within +-1440, precomputed per row on
  the CPU) or the optional `timeZone` transition table, which needs no per-row CPU work. They are
  mutually exclusive and both override the fixed offset.
- `timeZone: {transitions, offsets}`: `transitions` is a `uint32x2` view of strictly increasing Int64
  epoch-millisecond instants (the format of `timestamps`), `offsets` a `sint32` view of UTC offsets in
  seconds with `transitions.length + 1` elements. `offsets[0]` applies before the first transition
  and `offsets[i + 1]` from transition `i` on. Each row binary-searches its instant (signed 64-bit
  compare, about `log2(n)` steps) and uses that offset; seconds keep pre-standardization mean time
  such as +00:19:32 exact. A table offset beyond +-86400 s makes the row invalid. The table length
  is topology (a table of another length needs a new graph); its contents can change per frame.
  `getGPUTimeZoneTransitionTable(zone, {start, end})` builds the table on the CPU once per zone
  and range from `Intl.DateTimeFormat` (offset sampled daily and bisected to the millisecond; a
  zone without a transition in the range gets one sentinel transition at the Int64 minimum;
  an unknown zone throws). Instants outside the range use the offset at its start or end. The
  helper is as exact as the ICU data in the runtime: Node 22 reproduces Python `zoneinfo` (tzdata
  2026d) transition for transition for New_York, London, Lord_Howe, Kathmandu, Amsterdam and
  Dublin over 1900 to 2040, including seconds offsets; it differs only where the databases are
  different releases (Africa/Casablanca after 2026-09).
- Outputs, each optional but at least one required (compile-time): `year` and `isoWeekYear`
  (`sint32`), `month` (1-12), `dayOfMonth` (1-31), `hour`, `minute`, `weekday` (0-6),
  `dayOfYear` (1-366), `isoWeek` (1-53), `quarter` (1-4) as `uint32`, and `hourWeekdayCounts`
  (168 `uint32`, `weekday * 24 + hour`, cleared every encoding).
- Invalid rows (masked, local day beyond +-1e9, Int64 overflow with the offset, per-row offset beyond
  +-1440) write `0xffffffff` to `uint32` outputs and `-2^31` to `year` and `isoWeekYear`, and are not
  counted in the matrix.
- Exact integer arithmetic only (floor conversion of the unit to milliseconds, nibble-wise 64-bit division, Hinnant `civil_from_days`), identical on
  every adapter. Outputs are split over several decode kernels to respect the 8 storage binding limit.

#### How it computes

{/* lens:GPUCalendarBuckets */}

::lens{scenario=gpu-dataframe.calendar-buckets rung=datum bind=page datum=0 caption="Observation 0 is Sunday 2023-01-01 08:05: weekday 6, hour 8, month 1."}
::lens{scenario=gpu-dataframe.calendar-buckets rung=formula bind=page datum=0}
::lens{scenario=gpu-dataframe.calendar-buckets rung=kernel caption="Look at the floor division by 86,400,000 and the Hinnant civil-date arithmetic after it."}
::lens{scenario=gpu-dataframe.calendar-buckets rung=graph caption="Decode kernels split over the storage-binding limit, plus a cleared matrix and its count kernel."}

{/* parameters: GPUCalendarBuckets generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPUCalendarBuckets` parameters

Import from `@luma-contributors/analysis/dataframe-analysis`: `GPUCalendarBuckets`.

- Packer: `getGPUCalendarBucketsParameterValues(utcOffsetMinutes, firstDayOfWeek?)`, writes `GPU_CALENDAR_BUCKETS_PARAMETER_LENGTH` (2) `sint32` values.
  - `utcOffsetMinutes` (`number`): Fixed offset east of UTC in minutes, an integer in `[-1440, 1440]`.
  - `firstDayOfWeek` (`number`): First day of the week for the `weekday` output and the hour by weekday matrix: 0 = Monday (ISO, default) to 6 = Sunday. ISO week numbers always use Monday.

{/* /parameters */}

## `GPUPeriodIndex`

Turn calendar columns or Int64 word times into a `uint32` period per row: whole calendar steps from an origin, or a fixed width.

The period index is the adapter between `GPUCalendarBuckets` (or raw Int64 words) and a (cell, period) cube. It is the code that `addSpaceTimeHotSpotsRecipe` runs between its calendar decode and its cube, extracted so that other space-time pipelines share it and the recipe's slices are exactly these periods. It reads the temporal context of the times, never converts units, and writes one `uint32` per row with no readback.

```ts
import {GPUCalendarBuckets, GPUPeriodIndex} from '@luma-contributors/analysis/dataframe-analysis';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
declare const temporalContext: import('@luma-contributors/analysis/analysis-core').GPUTemporalContext;
declare const timestamps: import('@luma.gl/graph').GraphDataView<'uint32x2'>;
declare const parameters: import('@luma.gl/graph').GraphDataView<'sint32'>;
declare const year: import('@luma.gl/graph').GraphDataView<'sint32'>;
declare const month: import('@luma.gl/graph').GraphDataView<'uint32'>;
declare const dayOfMonth: import('@luma.gl/graph').GraphDataView<'uint32'>;
declare const periods: import('@luma.gl/graph').GraphDataView<'uint32'>;
declare const invalidCount: import('@luma.gl/graph').GraphDataView<'uint32'>;
declare const fixedPeriods: import('@luma.gl/graph').GraphDataView<'uint32'>;
declare const nanosecondContext: import('@luma-contributors/analysis/analysis-core').GPUTemporalContext;
declare const origin: number;
// Calendar axis: 52 weeks from Monday 2024-01-01, from the calendar columns of the same times.
graph.add(
  new GPUCalendarBuckets({temporalContext, timestamps, parameters, output: {year, month, dayOfMonth}})
);
graph.add(
  new GPUPeriodIndex({
    id: 'calendar-periods',
    temporalContext, // 'int64-words' counted from the Unix epoch
    columns: {year, month, dayOfMonth},
    periods: {
      kind: 'calendar',
      step: {calendarUnit: 'week'},
      origin: {year: 2024, month: 1, day: 1},
      count: 52
    },
    output: {periods, status: {invalidCount}}
  })
);

// Fixed axis: 24 one-hour bins of nanosecond words from 2024-01-01T00:00:00Z.
graph.add(
  new GPUPeriodIndex({
    id: 'fixed-periods',
    temporalContext: nanosecondContext,
    timestamps,
    periods: {kind: 'fixed', width: 3_600_000_000_000n, origin: 1_704_067_200_000_000_000n, count: 24},
    output: {periods: fixedPeriods}
  })
);
```

- `temporalContext`: resolved with `{encodings: ['int64-words'], unixEpoch: true}`, so Arrow uploads pass their own context. An unsupported encoding or an epoch other than the Unix epoch throws at construction. The `unit` is the unit of a fixed `width` and `origin`; a calendar axis only reads the decoded columns, whose unit `GPUCalendarBuckets` already handled.
- Calendar axis: `step` is `{calendarUnit, interval?}` with `calendarUnit` one of `'year'`, `'quarter'`, `'month'`, `'week'`, `'day'` or `'hour'` and a positive integer `interval` (default 1). Period `floor((stepIndex(time) - stepIndex(origin)) / stepsPerPeriod)` holds a time, where `stepIndex` counts whole steps (`year`, `year * 4 + quarter - 1`, `year * 12 + month - 1`, days since 1970 for `'day'` and `'week'`, hours since 1970 for `'hour'`) and a week is 7 days anchored to the origin day. `origin` is `{year, month?, day?, hour?}` in the local calendar of `columns`, with the unset fields at the start of the step. `columns.year` is always read; `columns.month` for every step but `'year'`; `columns.dayOfMonth` for `'week'`, `'day'` and `'hour'`; `columns.hour` for `'hour'`. Days are numbered with Hinnant's `days_from_civil` in `i32` arithmetic, so leap days, quarter starts and years before 1970 are exact. The CPU twins `getGPUPeriodIndexCalendarPeriod`, `getGPUPeriodIndexStepIndex` and `getGPUPeriodIndexDaysFromCivil` return the same numbers.
- Fixed axis: period `floor((time - origin) / width)` with `width` a positive integer below 2^63 (a `number` must be a safe integer; pass a `bigint` beyond 2^53) and `origin` signed Int64 ticks after the Unix epoch. The difference and the division run on the Int64 words, so they are exact for every time, including times before 1970 and beyond 2^53 ticks, and for widths past 32 bits.
- Output `periods` holds one `uint32` per row: the period, `GPU_PERIOD_INDEX_OUT_OF_RANGE` (`0xfffffffe`) for a valid time before the origin or past the last period, or `GPU_PERIOD_INDEX_INVALID` (`0xffffffff`) for a masked row or a time the calendar decode rejected. Rows are skipped, never clamped.
- Status: `output.status.invalidCount` (optional) is the number of skipped rows, those with `GPU_PERIOD_INDEX_OUT_OF_RANGE`. Masked and undecodable rows are not counted. It is cleared on every encoding and counted with an integer `atomicAdd`.
  Only its first word is a result. `GPUPeriodIndex`, `GPUCellPeriodKeys` and `GPUValueCube` preserve
  trailing caller data when an oversized invalid-count view is supplied.
- Limits: `periods.count` is an integer in `[1, GPU_PERIOD_INDEX_MAXIMUM_COUNT]` (4294967293, the two sentinels are not periods). Each bound range must fit `device.limits.maxStorageBufferBindingSize` **per chunk**, including any prefix required for an aligned binding offset. At a 128 MiB binding limit with no prefix, a `uint32` output or calendar column holds up to 33,554,432 rows per chunk, and Int64 `timestamps` hold up to 16,777,216 rows per chunk. Larger totals are supported by chunking, up to 4,294,967,295 total rows and subject to the command-node budget.
- Row inputs and `output.periods` accept packed views or chunked `GraphVectorView`s with matching ordered chunk lengths across every active column, the optional `mask` and the output. Each chunk is packed; mixing mismatched row partitions is rejected. Status views remain packed. An optional `mask` (`uint32`, zero writes the invalid sentinel) must have one row per input row.

For example, the fixed-axis chunk fixture uses three rows partitioned as `[1, 2]`. Given caller-owned
GPU vectors `timestampVector`, `maskVector` and `periodVector` with those same ordered chunk lengths:

```ts
import {GPUPeriodIndex} from '@luma-contributors/analysis/dataframe-analysis';
declare const temporalContext: import('@luma-contributors/analysis/analysis-core').GPUTemporalContext;
declare const timestampVector: import('@luma.gl/graph/gpu-data').GPUVector<'uint32x2'>;
declare const maskVector: import('@luma.gl/graph/gpu-data').GPUVector<'uint32'>;
declare const periodVector: import('@luma.gl/graph/gpu-data').GPUVector<'uint32'>;
declare const invalidCount: import('@luma.gl/graph').GraphDataView<'uint32'>;
declare const origin: number;
graph.add(
  new GPUPeriodIndex({
    id: 'periods',
    temporalContext, // millisecond, Unix epoch, int64-words
    periods: {kind: 'fixed', origin: 0n, width: 1000, count: 12},
    timestamps: graph.importGPUVector('chunked-times', timestampVector),
    mask: graph.importGPUVector('chunked-mask', maskVector),
    output: {
      periods: graph.importGPUVector('chunked-periods', periodVector),
      status: {invalidCount} // one packed uint32 word
    }
  })
);
```

This declares `periods-clear-invalid-count`, `periods-periods-chunk-0` and
`periods-periods-chunk-1`. A combined timestamp length above one binding's row limit is accepted
when each bound chunk range fits the device limit.

Calendar hour multiplication, step differences and division use exact 64-bit words, so indices at
`2^31 - 1`, `2^31` and above remain usable up to `periods.count - 1`. Calendar origins must have a
whole-step index in `[-2^31, 2^31 - 1]`; steps per period (`interval`, times seven for weeks) must
be in `[1, 2^31 - 1]`. These construction-time origin and divisor limits do not cap row indices or
the unsigned period count. Row dates follow the supported `GPUCalendarBuckets` decoder range.

### `GPUPeriodIndex` parameters

- Per frame (no recompile): the contents of `columns`, `timestamps` and `mask`.
- Compile-time (needs a new graph): the row count, `periods` (`kind`, `step`, `origin`, `width`, `count`), which calendar columns are present, and whether `mask` and `status.invalidCount` are present. The axis is compile-time because it fixes the size of the cube the periods index.
- Packer: none.

## `GPUCubeLayoutTranspose`

Move a dense (cell, period) cube between the cell-major and period-major layouts.

The space-time hot spot cube is cell-major (`cell * periodCount + period`); Markov chains, spatial panels and per-period maps are period-major (`period * cellCount + cell`). The transpose converts one into the other on the GPU for `uint32` counts or `float32` values.

```ts
import {GPUCubeLayoutTranspose} from '@luma-contributors/analysis/dataframe-analysis';
declare const cube: import('@luma.gl/graph').GraphDataView<"float32" | "uint32">;
declare const cellCount: number;
declare const periodCount: number;
declare const periodMajorCube: import('@luma.gl/graph').GraphDataView<"float32" | "uint32">;
graph.add(
  new GPUCubeLayoutTranspose({
    values: cube, // GraphDataView<'uint32'> or <'float32'>, cellCount * periodCount values
    from: 'cell-major', // the layout of `values`; the output holds the other one
    cellCount,
    periodCount,
    output: periodMajorCube // same format and length
  })
);
```

- Every value is copied bit for bit (counts, NaN payloads and signed zeros alike), and transposing back returns the original cube.
- Limits: `cellCount * periodCount` is an integer at most 4294967295 and the cube must fit one storage binding, 33,554,432 values on a default device. Larger cubes are refused at declaration with the binding limit in the message.
- The transpose is a scatter with one thread per element. `cellCount` and `periodCount` are overrides of the shader, so cubes of different shapes share one compiled module.

### `GPUCubeLayoutTranspose` parameters

- Per frame (no recompile): the contents of `values` and `output`.
- Compile-time (needs a new graph): `from`, `cellCount`, `periodCount` and the value format.
- Packer: none.

## `GPUTimeWordsRelative`

Convert Int64 word times to `float32` ticks after an origin on the GPU.

Several contributors read relative `float32` times (`GPUKnoxTest`, `GPUMantelTest`, `GPUSpaceTimeVariogram`, float-mode `GPUTemporalReduction`) while uploads keep exact Int64 words. This node connects them without a CPU pass.

```ts
import {GPUTimeWordsRelative} from '@luma-contributors/analysis/dataframe-analysis';
declare const timestamps: import('@luma.gl/graph').GraphDataView<'uint32x2'>;
declare const temporalContext: import('@luma-contributors/analysis/analysis-core').GPUTemporalContext;
declare const timeValidity: import('@luma.gl/graph').GraphDataView<'uint32'>;
declare const relativeTimes: import('@luma.gl/graph').GraphDataView<'float32'>;
declare const origin: number;
const relative = new GPUTimeWordsRelative({
  mask: timeValidity, // Arrow validity: 0 writes NaN
  timestamps, // GraphDataView<'uint32x2'>, Int64 words
  temporalContext, // 'int64-words', any unit and epoch
  origin: 1_700_000_000_000n, // ticks after the Unix epoch, in the context unit
  output: relativeTimes // GraphDataView<'float32'>, one row per timestamp
});
graph.add(relative);
relative.outputTemporalContext; // {...temporalContext, encoding: 'float32-relative', epoch: 1_700_000_000_000n}
```

- The difference to the origin is taken on the words with the exact `timeWordsDifferenceToF32` kernel and only then rounded, so it is exact below 2^24 ticks and has a relative error below 2^-23 beyond, with the right sign for every pair of Int64 values.
- `origin` is an instant (signed Int64 ticks after the Unix epoch). When the input context counts from another epoch, the node subtracts that epoch first, so the result is always ticks after `origin`.
- `outputTemporalContext` is `{...temporalContext, encoding: 'float32-relative', epoch: origin}`: the unit, time zone and axis are the input's. Nothing is converted between units.
- Limits: corresponding timestamp and output chunks must have aligned row lengths and each binding must fit. A 128 MiB binding holds 16,777,216 uint32x2 rows per chunk, not per column. Aligned chunks `[10_000_000, 9_000_000, 11_000_000]` therefore support 30 million total rows. Small status views, where present in downstream contributors, stay packed.

Pass Arrow time validity through the converter's optional `mask`, aligned with timestamp and
output chunks. A zero mask writes NaN: `[0n,null,1n]` with mask `[1,0,1]` and origin 0 produces
`[0,NaN,1]`. Without this mask physical null words can look like valid epoch zero. Retain validity
for downstream consumers and compact missing rows (preserving source-row mapping) when a consumer
cannot accept them. Units are ticks of `temporalContext.unit`, with the stated epoch;
conversion changes the encoding, not the unit.

### `GPUTimeWordsRelative` parameters

- Per frame (no recompile): the contents of `timestamps`, `mask` and `output`.
- Compile-time (needs a new graph): the row count, mask presence, `origin` and `temporalContext` (the origin words are shader overrides, so graphs that differ only in origin share a module).
- Packer: none.

## `GPUValueCube`

Dense (cell, period) cube of count, sum and mean of a value column.

`GPUTemporalReduction` and the hot spot recipe bin counts, minima, maxima, first and last values but no sum or mean. `GPUValueCube` adds them in the cell-major layout `cell * periodCount + period` that `GPUEmergingHotSpots` reads, so `GPUTimeWindowFilter` (the `mask`), `GPUPeriodIndex` (the `periods`) and the cube chain to Gi\*, trend, Markov chains and panels with no readback.

```ts
import {GPUValueCube} from '@luma-contributors/analysis/dataframe-analysis';
declare const cellCount: number;
declare const periodCount: number;
declare const cells: import('@luma.gl/graph').GraphDataView<'uint32'>;
declare const periodIndexOutput: import('@luma.gl/graph').GraphDataView<'uint32'>;
declare const values: import('@luma.gl/graph').GraphDataView<'float32'>;
declare const filterOutputMask: import('@luma.gl/graph').GraphDataView<'uint32'>;
declare const counts: import('@luma.gl/graph').GraphDataView<'uint32'>;
declare const sums: import('@luma.gl/graph').GraphDataView<'float32'>;
declare const means: import('@luma.gl/graph').GraphDataView<'float32'>;
declare const invalidCount: import('@luma.gl/graph').GraphDataView<'uint32'>;
graph.add(
  new GPUValueCube({
    cellCount,
    periodCount,
    cells, // GraphDataView<'uint32'>, cell index per row
    periods: periodIndexOutput, // GraphDataView<'uint32'> from GPUPeriodIndex
    values, // GraphDataView<'float32'>
    mask: filterOutputMask, // optional, for example outputMask of GPUTimeWindowFilter
    output: {
      counts, // GraphDataView<'uint32'>, cellCount * periodCount rows
      sums, // GraphDataView<'float32'>
      means, // GraphDataView<'float32'>, NaN where a bin is empty
      status: {invalidCount} // optional
    }
  })
);
```

- Sums are deterministic and independent of thread order. Each value is quantized to the shared 96-bit fixed-point accumulator (`analysis-core/kernels/fixed-point-accumulator.ts`) and added with integer atomics; the exact integer is rounded once to float32. There are no float atomics. For `1e8 + 1 - 1e8` the shared unit retains the `1`; a float32 running sum loses it. This does not guarantee exact real sums over arbitrary dynamic ranges.
- The unit is a power of two set by the largest magnitude among the rows that are binned and by the bit length of the row count. A value keeps all 24 significand bits while it is at least `max * 2^(h - 71)` (`h` the bit length of the row count; about 1e15 of dynamic range at a million rows); smaller values round to the nearest unit, so a sum of `n` terms is off by at most `n / 2` units before the final rounding. Because the unit follows the binned rows, the rounding of a bin can change when another set of rows is binned, for example another time window.
- `means` is computed separately from the accumulated integer total and integer count, not
  by dividing published `sums`. Empty means are NaN. The shared-scale bound and cross-bin
  small-value loss described under [GPUGroupStatistics](#gpugroupstatistics) apply here too;
  set `scaleMode: 'per-bin'` when independent bin precision is required. Identical finite maxima retain finite means even when the independent sum overflows.
- Not binned: masked rows (silent), rows whose period is `0xffffffff` (`GPU_PERIOD_INDEX_INVALID`, excluded upstream, silent), and rows with a cell or period out of range (including `GPU_PERIOD_INDEX_OUT_OF_RANGE`) or a non-finite value. The last group is counted in `status.invalidCount`.
- Limits: `cellCount`, `periodCount` and their bin product at most 4,294,967,294; the logical input-row count is at most 4,294,967,295; the fixed-point words need 12 bytes per bin in one storage binding, so `cellCount * periodCount` is at most `getGPUValueCubeMaximumBinCount(device)`, 11,184,810 on a default device. Larger cubes are refused at declaration with the binding limit in the message. Inputs and outputs are single packed views.

### `GPUValueCube` parameters

- Per frame (no recompile): the contents of `cells`, `periods`, `values` and `mask`.
- Compile-time (needs a new graph): the row count, `cellCount`, `periodCount`, and which outputs and `mask` are present.
- Packer: none.

## `GPUCellPeriodKeys`

Packs a 64-bit H3, S2 or Quadbin cell key and a period into one group key for `GPUGroupStatistics`.

A dense cube needs a cell index below `cellCount`; a DGGS has billions of possible cells. `GPUGroupStatistics` already groups by `uint32x2` keys and computes count, sum, mean, minimum, maximum, variance and percentiles in a capacity-bounded table sorted by key, so a sparse DGGS by period cube is one packing kernel away.

```ts
import {GPUCellPeriodKeys, GPUGroupStatistics} from '@luma-contributors/analysis/dataframe-analysis';
declare const cellKeys: import('@luma.gl/graph').GraphDataView<'uint32x2'>;
declare const periodIndexOutput: import('@luma.gl/graph').GraphDataView<'uint32'>;
declare const mask: import('@luma.gl/graph').GraphDataView<'uint32'>;
declare const packedKeys: import('@luma.gl/graph').GraphDataView<'uint32x2'>;
declare const invalidCount: import('@luma.gl/graph').GraphDataView<'uint32'>;
declare const values: import('@luma.gl/graph').GraphDataView<'float32'>;
declare const counts: import('@luma.gl/graph').GraphDataView<'uint32'>;
declare const sumValues: import('@luma.gl/graph').GraphDataView<'float32'>;
declare const means: import('@luma.gl/graph').GraphDataView<'float32'>;
declare const groupKeys: import('@luma.gl/graph').GraphDataView<"uint32x2" | "uint32">;
declare const groupCounts: import('@luma.gl/graph').GraphDataView<'uint32'>;
declare const count: import('@luma.gl/graph').GraphDataView<'uint32'>;
declare const overflow: import('@luma.gl/graph').GraphDataView<'uint32'>;
graph.add(
  new GPUCellPeriodKeys({
    cells: cellKeys, // GraphDataView<'uint32x2'>, little-endian (low, high) words
    periods: periodIndexOutput, // GraphDataView<'uint32'> from GPUPeriodIndex
    periodCount: 365,
    layout: 'legacy-one-bit', // preserve persisted keys; see explicit identity below
    fill: 'ones', // 'ones' for H3 and Quadbin, 'zeros' for S2
    mask, // optional
    output: {keys: packedKeys, status: {invalidCount}}
  })
);
graph.add(
  new GPUGroupStatistics({
    keys: packedKeys,
    columns: [{values, statistics: ['count', 'sum', 'mean'], output: {counts, sumValues, means}}],
    output: {keys: groupKeys, counts: groupCounts, status: {count, overflow}}
  })
);
```

Which route builds a DGGS (cell, period) cube without readback? Two compose from the existing pieces: (a) this packing kernel and `GPUGroupStatistics` on the composite key, or (b) `GPUCellAggregation` once to get the distinct cells, `GPUKeyLookup` to map every row to its cell row, then `row * periodCount + period` as a key into `GPUValueCube` or `GPUGroupStatistics` in dense mode. Use (a) for DGGS cubes: it needs one sort and no cell-row lookup, its memory follows the number of occupied (cell, period) groups instead of `cellCount * periodCount`, and cells never need a table. Use (b), or `GPUValueCube` with a cell index you already have (a lattice, or a weights table), only when the consumer needs the dense cell-major cube, as `GPUEmergingHotSpots` and the layout transpose do.

- Packing: with legacy `b = max(1, ceil(log2(periodCount)))` period bits, the key is `(cellKey & ~(2^b - 1)) | period`. The period replaces the low `b` bits of the cell key, so the packed key sorts by cell and then period. A packed key is not a valid cell id: unpack group-table keys with `unpackGPUCellPeriodKeyChecked(key, periodCount, fill, layout)`.
- Select the same explicit layout when packing and unpacking persisted keys.
  `layout: 'legacy-one-bit'` preserves old keys, including `'ones'` cell `3n`, period 0,
  count 1 stored as `2n` (cell `7n` is stored as `6n`).
  `layout: 'zero-period-bit-identity'` is allowed only for one period and stores cell `3n`
  unchanged as `3n`. Use the legacy layout to read old `2n` keys. The producer, bit-count helper,
  packer and checked unpacker all require an explicit layout.

```ts
import {packGPUCellPeriodKey, unpackGPUCellPeriodKeyChecked} from '@luma-contributors/analysis/dataframe-analysis';
const legacyKey = packGPUCellPeriodKey(3n, 0, 1, 'ones', 'legacy-one-bit'); // 2n
const identityKey = packGPUCellPeriodKey(3n, 0, 1, 'ones', 'zero-period-bit-identity'); // 3n
const decoded = unpackGPUCellPeriodKeyChecked(
  0xffffffffffffffffn, 3, 'zeros', 'legacy-one-bit'
); // undefined: excluded row, never a group
if (decoded !== undefined) {
  const {cellKey, period} = decoded;
  // Consume a valid group only here.
}
// unpackGPUCellPeriodKeyChecked(3n, 3, 'zeros', 'legacy-one-bit') throws:
// non-sentinel period 3 is outside 0..2; malformed data is different from an excluded row.
```

- What happens to the key bits: the cell key stays 64 bits, nothing is truncated to 32. The `b` replaced bits must hold the `fill` pattern. At the resolution of the cells they do: unused H3 digits are 7 (`3 * (15 - resolution)` low bits, all ones), the Quadbin footer is all ones (`52 - 2 * zoom` bits), an S2 id has a lone sentinel one above `60 - 2 * level` zero bits. Within that many free bits the packing is lossless and injective. A cell with any other bits where the period goes (a finer cell) is not packed: its key becomes the reserved all-ones key, which `GPUGroupStatistics` skips, and it is counted in `status.invalidCount`; cells are never merged silently. A finer cell whose extra bits happen to equal the fill pattern is packed and unpacks to itself. `getGPUCellKeyFreeBits('h3' | 'quadbin' | 's2', resolution)` gives the free bits. 365 daily periods need 9 bits: H3 resolution 12 or coarser, Quadbin zoom 21 or coarser, S2 level 25 or coarser. Hourly periods over a year (8,760) need 14 bits: H3 resolution 10, Quadbin zoom 19, S2 level 23.
- Rows are skipped silently when masked, when the cell is the all-ones "no cell" key, or when the period is `0xffffffff`. A period at or above `periodCount` is skipped and counted.
- Limits: `periodCount` at most 4294967293 (32 period bits). Corresponding `cells`, `periods`, mask and output key chunks must have aligned row lengths, and each binding must fit the device. At 128 MiB a uint32x2 chunk holds 16,777,216 rows; this is not a total-column cap. Aligned chunks `[10_000_000, 9_000_000, 11_000_000]` admit 30 million rows. Keep optional `status.invalidCount` as one packed word. Downstream, `GPUGroupStatistics` keeps at most `getGPUGroupStatisticsMaximumGroupCount` (8,388,608 on a default device) distinct (cell, period) groups and keeps the smallest keys on overflow, and its `sums` words are fixed point with a resolution of `1 / sumScale` (65536 by default); its `sumValues` and `means` use a unit that follows the data, as `GPUValueCube` does.

### `GPUCellPeriodKeys` parameters

- Per frame (no recompile): the contents of `cells`, `periods` and `mask`.
- Compile-time (needs a new graph): the row count, `periodCount`, `fill`, `layout`, and whether `mask` and `status.invalidCount` are present.
- Packer: none.

## Related pages

- [GPU Dataframe](https://luma.gl/docs/api-reference/experimental/gpu-dataframe)
- [Indexes and joins](https://luma.gl/docs/api-reference/experimental/gpu-dataframe-indexes-joins)
- [GPU Spatial Analysis contributors](/docs/api/gpu-spatial-analysis)

## `GPUCrossVariogram`

{/* lens:GPUCrossVariogram */}

::lens{scenario=gpu-dataframe.cross-variogram rung=graph}

This executable four-point projected example captures pair counts, estimator values and validity status. See [the estimator contracts](./gpu-spatial-analysis/statistical-expansion.md) for bin edges, complete-case masks, normalization and traversal bounds.

The default `[256, 256]` index accepts the same row counts as an explicit `gridSize`;
`rows * lagCount * temporalLagCount` must fit the `2^24` partial-entry budget and the device's
storage binding limit. The former 16,384-row cap on the default has been removed.
If a bin's pair count exceeds `0xffffffff`, its count saturates at that value, status is 3,
and semivariance is NaN. An exact count of `0xffffffff` remains valid.

## `GPUSpaceTimeVariogram`

{/* lens:GPUSpaceTimeVariogram */}

::lens{scenario=gpu-dataframe.space-time-variogram rung=graph}

Empirical space-time semivariogram on the indexed pair histogram, the same traversal as `GPUVariogram`. Each unordered pair within `maximumDistance` and `maximumTimeLag` is counted once, in a spatial class and a temporal class. Class 0 of each axis is the exact zero lag, so the output holds `(lagCount + 1) * (temporalLagCount + 1)` bins, spatial-major (`spatialClass * (temporalLagCount + 1) + temporalClass`): the first row is the purely temporal variogram, the first column the purely spatial one, and bin 0 is coincident points at the same time. Positive lags use equal-width classes over `(0, maximum]`.

After readback, import `getGPUSpaceTimeVariogramBinLags`, `fitSpaceTimeVariogramModel` and
`evaluateSpaceTimeVariogramModel` alongside the producer from
`@luma-contributors/analysis/dataframe-analysis`. The spatial entry's existing aliases remain
available. Distances use position units, time lags use the
relative-time unit, semivariances use squared measure units and pair counts are dimensionless.
Keep the supplied space-time props, options and port mappings unchanged through graph declaration;
the current producer reads them in `getCommandNodes`. Buffer contents stay live.
Status is one word per bin; these codes are separate from the optional scalar overflow ports on
`GPUVariogram` and `GPUSpatialCorrelogram`.

For this space-time producer, both columns' IQRs and moments use the pair index's complete-case
population: caller mask nonzero, finite coordinates inside the inclusive bounds, both values
finite (secondValues defaults to values), and finite time. An isolated but admitted point still
contributes to global scales even without a pair within the lag limits. An outside point affects
neither scale nor moments. Ordinary variogram/correlogram also apply positional and polygon
complete-case filtering; this space-time route additionally requires both values and time.

Fit only after complete readbacks, using the same maximum lags as the parameter buffer:

```ts
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUSpaceTimeVariogram,
  getGPUSpaceTimeVariogramBinLags,
  fitSpaceTimeVariogramModel,
  evaluateSpaceTimeVariogramModel,
  type GPUSpaceTimeVariogramProps
} from '@luma-contributors/analysis/dataframe-analysis';

export async function fitEmpiricalSpaceTime(
  graph: GPUCommandGraph,
  props: Omit<GPUSpaceTimeVariogramProps, 'secondValues'> & {secondValues?: never},
  limits: {maximumDistance: number; maximumTimeLag: number},
  submitAndRead: () => Promise<{
    semivariances: Float32Array;
    pairCounts: Uint32Array;
    status: Uint32Array;
  }>
) {
  // Fit a single-column semivariogram: signed cross-semivariances are unsupported.
  if (props.secondValues !== undefined) {
    throw new Error('Fit a single-column semivariogram: omit secondValues');
  }
  // The parameter buffer must contain these same maximum lag limits.
  graph.add(new GPUSpaceTimeVariogram(props));
  const bins = await submitAndRead();
  const binCount = (props.lagCount + 1) * (props.temporalLagCount + 1);
  for (const name of ['semivariances', 'pairCounts', 'status'] as const) {
    if (bins[name].length !== binCount) {
      throw new Error(`${name} length must equal binCount ${binCount}, received ${bins[name].length}`);
    }
  }
  if (bins.status.some(code => code !== 0 && code !== 1)) {
    throw new Error('Empirical space-time bins contain arithmetic failure or loss');
  }
  for (let bin = 0; bin < binCount; bin++) {
    if (bins.pairCounts[bin] > 0 &&
        (bins.status[bin] !== 0 || !Number.isFinite(bins.semivariances[bin]))) {
      throw new Error(`Empirical bin ${bin} has no usable semivariance`);
    }
  }
  const lags = getGPUSpaceTimeVariogramBinLags({
    lagCount: props.lagCount,
    temporalLagCount: props.temporalLagCount,
    maximumDistance: limits.maximumDistance,
    maximumTimeLag: limits.maximumTimeLag
  });
  const fit = fitSpaceTimeVariogramModel(
    {...lags, semivariances: bins.semivariances, pairCounts: bins.pairCounts},
    {type: 'metric', weighting: 'pairs'}
  );
  return {fit, atMaximumSpatialLag: evaluateSpaceTimeVariogramModel(
    fit.model, limits.maximumDistance, 0
  )};
}
```

The callback submits the graph and reads exactly the live bin count from each output, excluding
allocation tails. Status 1 bins may be empty and are skipped when their count is zero; a
positive-count bin must have status 0 and finite semivariance. Reject failed or unknown statuses
before fitting. The fitter uses nonnegative nugget/sill and this workflow therefore omits
`secondValues`; signed cross-semivariances are not admitted. Inspect `fit.cycleCount`,
`fit.converged`, `fit.atBoundary` and `fit.residual` separately. A boundary solution can converge.

Props: `positions`, `values`, optional `secondValues` (signed cross-semivariance, default the Matheron semivariogram), `times`, optional `mask`, `parameters` written with `getGPUSpaceTimeVariogramParameterValues({bounds, maximumDistance, maximumTimeLag})`, `gridSize`, `lagCount`, `temporalLagCount` and `output: {semivariances, pairCounts, status}`. Status per bin is 0 defined, 1 empty or invalid parameters, 2 nonfinite arithmetic, 3 accumulation overflow or more than `0xffffffff` pairs in the bin. With status 3 for a saturated count, `pairCounts` holds `0xffffffff` and the semivariance is still written. An exact count of `0xffffffff` keeps status 0. Check for status 3 before using `pairCounts` as fitting weights. Both columns are centered on the complete-case rows only (mask set, both values and the time finite, position inside `bounds`). Rows excluded by that rule cannot move either center. Before this, one excluded row of 1e9 could round the retained float32 differences to 0.

Fixed-point terms use the geometric mean of the two adaptive column units (the single column's
unit when `secondValues` is absent). Scaled unit storage, reciprocal-free term encoding and
matching decoding preserve bounded estimates even when that unit is subnormal or its reciprocal
would overflow float32. The same variance floor and integer-channel limits as `GPUVariogram` apply.

`times` must be relative to a CPU float64 origin. Raw Unix epoch seconds in float32 are spaced 128 s apart (milliseconds about 2 minutes), which merges every short lag into one. Subtract the origin with `getGPUSpaceTimeRelativeTimes` from `@luma-contributors/analysis/gpu-spatial-analysis` before upload; it also returns the remaining float32 resolution. Times closer than that resolution share the zero-lag class. See [the estimator contracts](./gpu-spatial-analysis/statistical-expansion.md) for bin edges, complete-case masks and the validation status: the results match an in-repo brute-force reference, and a gstat `variogramST` fixture is still pending.

Breaking change: the parameter buffer is now eight values (bounds, then the two maximum lags), `gridSize` is required, the output has the extra zero-lag classes, and the limit is 409 bins instead of 16,384 rows by 128 bins.

{/* parameters: GPUSpaceTimeVariogram generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPUSpaceTimeVariogram` parameters

Import from `@luma-contributors/analysis/dataframe-analysis`: `GPUSpaceTimeVariogram`.

- Packer: `getGPUSpaceTimeVariogramParameterValues(settings)`, writes `GPU_SPACE_TIME_VARIOGRAM_PARAMETER_LENGTH` (8) `float32` values.

Settings of `getGPUSpaceTimeVariogramParameterValues`:

| Field | Type | Description |
|---|---|---|
| `bounds` | `readonly [number, number, number, number]` | Inclusive `[minX, minY, maxX, maxY]` extent; rows outside it are excluded. |
| `maximumDistance` | `number` | Largest spatial lag, in the units of the positions. |
| `maximumTimeLag` | `number` | Largest time lag, in the unit of the relative `times`. |

{/* /parameters */}

## `GPUMarkedPointStatistics`

{/* lens:GPUMarkedPointStatistics */}

::lens{scenario=gpu-dataframe.marked-point-statistics rung=graph}

This executable four-point projected example captures pair counts, estimator values and validity status. See [the estimator contracts](./gpu-spatial-analysis/statistical-expansion.md) for bin edges, complete-case masks, normalization and traversal bounds.

Candidate search defaults to `searchStructure: 'grid'` with `gridSize: [256, 256]` maximum
dimensions. It accepts `rows * radiusCount * 3 <= 2^24` float32 partials, subject to the device's
binding size. The explicit `searchStructure: 'brute-force'` reference is limited to 16,384 rows
and cannot take `gridSize`. Both use the same estimators and radius edges; indexed sums may differ
within float32 rounding tolerance. Finite radii whose square overflows use a single-cell fallback
with the finite-distance predicate. Broad neighborhoods can still require quadratic pair work.
Counts exceeding `0xffffffff` saturate, with status 3 and all four statistics NaN for that radius.
An exact count of `0xffffffff` remains valid.
Category means use a separate scaled sum for large marks, averaging before restoring the scale;
the unnormalized mark sum need not fit float32. Small marks remain unscaled.

## `GPUVariogramModelFit`

The direct contributor and both variogram fit operations snapshot model order and options at
construction. The operations also capture input and output graph names. Caller edits to model
arrays or output maps cannot change the declared row counts, result keys or later GPU lowering.
The contributor preserves view/buffer identities and dynamic contents. Construct a new fit to
change its model list, options or bindings.

Both operation forms require three distinct output graph names, each different from every input
graph name. Collisions throw at construction with the operation ID and conflicting output port;
shared read-only input names are allowed. This prevents an output from erasing another result or
overwriting an input in the declared graph contract.

{/* lens:GPUVariogramModelFit */}

::lens{scenario=gpu-dataframe.variogram-model-fit rung=result caption="Resident spherical, exponential and Gaussian fits to 20 explicitly synthetic empirical lag bins."}

::lens{scenario=gpu-dataframe.variogram-model-fit rung=graph caption="The bounded GPU range search produces fitted parameters, residual/AIC diagnostics and convergence status."}

This executable example uploads bins generated from an exponential model with nugget `0.2`, partial sill `1.5` and effective range `2.2`, then fits all three candidates on the GPU with pair weights. Model rows are spherical, exponential and Gaussian. Captured parameter rows contain nugget, partial sill, effective range and weighted residual; diagnostic rows contain usable bins, unweighted residual, residual AIC and final log-range bracket. Status 0 is an interior converged fit, 1 insufficient bins, 2 numerical failure and 3 a boundary/budget result. See [the fitting contract](./gpu-spatial-analysis/interpolation-expansion.md) for identifiability, weighting and performance limitations.

The fitter also supports `matern-3/2` and `matern-5/2`. With `a = factor * h / range`, their normalized semivariogram shapes are `1 - (1 + a) exp(-a)` and `1 - (1 + a + a² / 3) exp(-a)`. The factors are `4.749031386012701` and `5.924462618951797`, respectively, giving correlation `exp(-3)` at the effective range. These definitions also apply to coregionalization below. CPU evaluation uses a positive exponential-remainder series through order 18 for `a < 0.5`; GPU evaluation uses a bounded Taylor series for `a < 0.1`. These evaluations preserve the quadratic leading term at tiny lag/range ratios. CPU exponential and Gaussian shapes use `expm1` for the same cancellation concern. The mathematical model and fitting objective are unchanged by these numerical evaluations.

## `GPUCoregionalizationFit`

{/* lens:GPUCoregionalizationFit */}

::lens{scenario=gpu-dataframe.coregionalization-fit rung=result caption="Fit a PSD two-variable structure matrix with a negative cross coefficient; exclude one zero-pair bin."}

::lens{scenario=gpu-dataframe.coregionalization-fit rung=graph caption="Gated residual fits and PSD projections publish matrices, square loading factors and named convergence status."}

Fits a positive semidefinite linear model of coregionalization from complete direct/cross
semivariogram matrices on shared lag supports. The [coregionalization guide](dataframe-analysis/coregionalization-fit.md)
is the canonical reference for input layout, per-frame parameters, GPU bounds, output sizing,
budget/validity status, independent verification and supplying the fitted model to local kriging.

It covers `GPUCoregionalizationFit`, `GPUCoregionalizationFitOperation`,
`CPUCoregionalizationFitOperation`, `computeCoregionalizationFitOnCPU` and
`fitLinearModelOfCoregionalization`. Prefer the float64 CPU twin for small host-resident fits;
the GPU operation uses `GPU_CONTRIBUTOR_OPERATION_LOWERING` with the same graph properties.

The lens example uses three hand-derived lag matrices from `B = [[2, -1], [-1, 3]]` and a unit-sill spherical structure of range 2. A fourth, unrelated matrix has zero pairs and is excluded. When run, the scenario uses an eight-sweep budget and records GPU matrices, factors, residual diagnostics and every named status port.

{/* parameters: GPUCoregionalizationFit generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPUCoregionalizationFit` parameters

Import from `@luma-contributors/analysis/dataframe-analysis`: `GPUCoregionalizationFit`.

- Packer: `getGPUCoregionalizationFitParameterValues(structures, tolerance?)`, writes the `float32` parameter view.
  - `structures` (`readonly CoregionalizationStructure[]`)
  - `tolerance` (`number`)

{/* /parameters */}


## `GPUForestInference`

{/* lens:GPUForestInference */}

::lens{scenario=gpu-dataframe.forest-inference rung=formula}

Apply an imported random forest or boosted model to resident feature rows or co-registered raster
bands. The synthetic raster-stack lens shows a tied classification leaf, the missing-feature branch,
and a masked padding row. See [Forest inference](./forest-inference.md) for model import formats,
parameters, output scores and leaf indices, limits and validation.
## `GPUWideNumericScan`

Low-level word-phase execution shared by exact Int64/Decimal128 aggregates and Float32 narrowing
diagnostics. All rows run in parallel; validity chunks must match the packed word chunks. Its
caller initializes the result and owns the bounded readback. See
[the public preparation and profiling contracts](reused-analysis-preparation.md) for precision,
row limits, CPU references and the exact bigint integer-error representation. This execution
helper has no standalone lens; consumers use the domain aggregate and diagnostic functions.
