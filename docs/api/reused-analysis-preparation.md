# Source topology and wide numeric profiling

### `prepareArrowTableForGPUAnalytics`

Import from `@luma-contributors/analysis/analysis-core`. Preparation is a host step before upload;
it retains source record-batch order, row offsets, null validity and field metadata. Use
`sourceBatches` to map a batch-local row to `sourceRowIndexOffset + localRow`; retain
`sourceRowCount` even for time-only tables, whose scalar `table` has no columns.

| Source Arrow type | Prepared representation | Precision and validity |
| --- | --- | --- |
| Float16 | Float32 | Exact widening of each stored half-precision value; nulls remain null. No narrowing policy required. |
| Float32 | Float32 | Existing portable scalar data is borrowed. |
| Boolean / integers through 32 bits | Uint32 / Int32 | Exact widening; nulls remain null. |
| Utf8 / LargeUtf8 | Dictionary with Uint32 codes and Utf8 labels | Lossless label preparation; nulls remain null, not category zero. LargeUtf8's 64-bit source offsets are not numeric values to narrow. |
| Float64 / wide integers / supported wide Decimal | Float32 | Explicit precision policy required, as described below. |
| Timestamp | Separate low/high Uint32 words and Uint32 validity | Exact ticks in the Timestamp type's unit. Retain words, validity and chunk boundaries. |
| DateDay / DateMillisecond | Separate low/high Uint32 words and Uint32 validity | Both normalize to milliseconds; a DateDay value of 2 becomes 172800000 millisecond ticks. |
| Explicitly declared Int64 time | Separate low/high Uint32 words and Uint32 validity | Exact ticks in the unit declared through `timeColumns`. |

Category preparation reconciles labels across selected batches. Reuse returned `dictionaries`
for stable unordered codes in a later call; independently prepared tables do not automatically
share a codebook. The resulting dictionary enforces
`MAXIMUM_ARROW_ANALYTICS_CATEGORY_COUNT = 4294967295` categories: codes 0 through 4294967294;
`0xffffffff` is reserved for missing keys. Capacity is checked before adding a new label.
Independently, each rebuilt physical Utf8 dictionary permits at most
`MAXIMUM_ARROW_ANALYTICS_DICTIONARY_BYTE_LENGTH = 2147483647` encoded bytes across all labels,
including when the source is LargeUtf8. Split larger string sets into separately prepared
dictionaries. These are preparation limits; downstream device and producer limits still apply.
Installed and reused labels must be unique strings. An ordered dictionary must preserve its exact
ordinal definition (labels and order) across batches **within one source**. For cross-source
equality, `prepareArrowKeyPairForGPUAnalytics` reconciles the two sources by decoded label into
one unordered equality codebook; the left and right sources may have different ordinal orders.
The per-source consistency check still applies. For unordered source dictionaries, a valid index
that decodes to a null label has eligibility zero; index validity alone does not establish a usable
key. An observed ordered definition requires unique string labels: a null label is rejected before
any eligibility mask is returned.

A schema-only ordered dictionary has no observed ordinal definition and does not install an empty
definition in returned `dictionaries`. The first actual dictionary establishes that definition.
A physically observed empty dictionary, or an explicitly reused empty definition, is a real
definition and must agree with later batches. See the
[paired-key example](./gpu-dataframe-analysis.md#gpukeylookup) for shared codes, chunk boundaries,
source-aligned eligibility and exact integer keys.

`prepareArrowMeasureColumnsForGPUAnalytics` admits Float16, Float32 and Float64 source fields only;
integer fields are rejected even if a precision policy could narrow them. Float16 widens exactly,
Float32 retains its stored values, and Float64 needs the explicit precision policy below. Measures
retain the source field's units and record-batch boundaries. Each measure has independent validity:
null lanes become NaN and real zeros remain zero. Keep these masks separate from key eligibility
and from each other. The [nullable-measure example](./gpu-dataframe-analysis.md#gpugroupstatistics)
shows COUNT(*) = 3, COUNT(x) = 2 and separate means 1 and 8 without intersecting the masks.

```ts
import * as arrow from 'apache-arrow';
import {prepareArrowTableForGPUAnalytics} from '@luma-contributors/analysis/analysis-core';

const source = new arrow.Table({
  measurement: arrow.vectorFromArray([1.5, null, -3.25], new arrow.Float16()),
  label: arrow.vectorFromArray(['a', null, 'b'], new arrow.LargeUtf8())
});
const prepared = prepareArrowTableForGPUAnalytics(source);
const measurement = prepared.table.getChild('measurement'); // Float32 [1.5, null, -3.25]
const label = prepared.table.getChild('label'); // labels ['a', null, 'b'], not ['a', 'a', 'b']
const codebook = prepared.dictionaries['label']; // values ['a', 'b']
// Preserve Arrow validity when uploading: a null's physical numeric lane can contain zero.
```

These two routes are lossless preparation, separate from wide-number narrowing.
Float64, Int64, Uint64 and wide Decimal
columns (`bitWidth >= 64`, including Decimal64, Decimal128 and Decimal256) require an explicit
`float64` precision policy before narrowing to Float32. Decimal32 is unsupported. For wide
integers and decimals, preparation rounds the original signed coefficient and decimal scale
directly to nearest-even Float32. `mode: 'exact'` rejects any changed value, including
`9007199254740993` and Decimal128 `1.0000000000000001`.

Round mode compares the error of a finite candidate with
`maximumAbsoluteError + maximumRelativeError * abs(original)`. Int64, Uint64 and supported Decimal
columns compare budgets with exact rational arithmetic. Float64 columns compute the error and
budget expression with JavaScript binary64 arithmetic. For example, with relative budget
`1 / 16777281` and absolute budget `0`, Float64 `16777281` accepts candidate `16777280`, while
Decimal128 rejects it because the binary64 relative budget is slightly too small.

Wide integer and Decimal error reports round each exact rational error once to nearest-even
binary64, including subnormals. An Int64 value of `9007199254740993` accepts an absolute budget
of `1` and reports an error of `1`. Extreme Decimal scales do not require constructing an enormous
power of ten: zero remains exact, definite overflow rejects, and definite underflow respects the
budget and preserves the sign of zero. An absolute error below half `Number.MIN_VALUE` reports
zero while its relative error is still `1`. Nulls and record-batch topology are preserved.
The standalone `convertWideNumericValue` / `convertWideNumericVector` helpers convert through JavaScript numbers;
they do not establish an exact Float32 precision policy for Decimal values.

### Nullable Arrow time: keep words and validity together

Declare the unit for an Int64 time column with `timeColumns`. Timestamp retains its Arrow unit;
both DateDay and DateMillisecond normalize to milliseconds. Use the prepared column's `unit` for window endpoints and relative-time origins,
rather than assuming the physical source unit: two DateDay days become `172800000n` milliseconds,
so an exact window for that date is `[172800000n,172800000n]`, not `[2n,2n]`.
`prepared.timeColumns[name]` contains independent low/high word chunks and
one Uint32 validity value per source row (nonzero means valid), in the same record-batch layout.
`makeGPUTimeVectorFromPreparedArrowColumn(device, column)` uploads both and returns their
temporal context. Import both vectors into the same graph, keep their ordered chunks aligned,
and call the upload's `destroy()` after the compiled graph is destroyed.

```ts
import * as arrow from 'apache-arrow';
import {prepareArrowTableForGPUAnalytics} from '@luma-contributors/analysis/analysis-core';

const source = new arrow.Table({
  time: arrow.vectorFromArray([null, 0n, 10n], new arrow.Int64())
});
const prepared = prepareArrowTableForGPUAnalytics(source, {timeColumns: {time: 'millisecond'}});
const time = prepared.timeColumns['time'];
// time.chunks[0]: [0, 0, 0, 0, 10, 0]; time.validity[0]: [0, 1, 1].
// A zero word pair alone cannot distinguish the null at row 0 from epoch zero at row 1.
```

For `GPUTimeWindowFilter`, use the unsorted route with
`additionalPredicates: [{kind: 'selection', mask: timeValidity}]`, where `timeValidity` is the
uploaded validity vector aligned with `timestamps`. Window `[0n, 0n]` then selects source row
1 only, with count 1. `sorted: true` requires all-valid, nondecreasing instants in a single packed view.
It rejects additional predicates and a `GraphVectorView`, even one containing a single chunk.
Select an established all-valid, nondecreasing packed chunk or contiguous slice; an all-valid
slice alone does not establish order. Retain its original source-row offset. For the example,
the valid slice starts at source row 1; add 1 to each returned slice-local row. An order check
alone cannot distinguish the two zero pairs. Use `validateOrder: true` with an
`output.orderViolations` word during development to check order; also establish validity separately.

```ts
import type {GPUCommandGraph, GraphVectorView} from '@luma.gl/graph/gpu-core';

declare const graph: GPUCommandGraph;
declare const timestamps: GraphVectorView<'uint32x2'>; // uploaded [null, 0n, 10n], one chunk
const chunkIndex = 0;
const chunk = timestamps.data[chunkIndex]; // packed GraphDataView, not the vector wrapper
const chunkSourceRowOffset = 0; // prepared.sourceBatches[chunkIndex].sourceRowIndexOffset in general
const firstValidRow = 1;
const validRowCount = 2; // caller established that this entire slice is valid AND nondecreasing
const sortedTimestamps = graph.createDataView(chunk.buffer, {
  format: 'uint32x2',
  byteOffset: chunk.byteOffset + firstValidRow * chunk.byteStride,
  byteStride: chunk.byteStride,
  length: validRowCount
});
// Use sortedTimestamps with sorted: true; omit additionalPredicates on that route.
// For a whole verified chunk use chunk itself and firstValidRow = 0 instead.
const sourceRowOffset = chunkSourceRowOffset + firstValidRow;
const sliceLocalRow = 0; // zero-window result from this slice
const originalRow = sourceRowOffset + sliceLocalRow; // original row 1
```

```ts
import {Buffer, type Device} from '@luma.gl/core';
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUParameterBuffer,
  readPackedBuffer,
  makeGPUTimeVectorFromPreparedArrowColumn,
  type PreparedArrowAnalyticsTimeColumn
} from '@luma-contributors/analysis/analysis-core';
import {
  GPUTimeWindowFilter,
  GPUTimeWordsRelative,
  GPU_TIME_WINDOW_FILTER_WORD_PARAMETER_LENGTH,
  getGPUTimeWindowFilterWordParameterValues
} from '@luma-contributors/analysis/dataframe-analysis';

declare const device: Device;
declare const graph: GPUCommandGraph;
declare const time: PreparedArrowAnalyticsTimeColumn; // prepared millisecond ticks, including validity
const upload = makeGPUTimeVectorFromPreparedArrowColumn(device, time);
const timestamps = graph.importGPUVector('arrow-times', upload.timestamps);
const timeValidity = graph.importGPUVector('arrow-time-validity', upload.validity);
const windowParameters = new GPUParameterBuffer(device, {
  id: 'time-window', format: 'uint32', length: GPU_TIME_WINDOW_FILTER_WORD_PARAMETER_LENGTH,
  values: getGPUTimeWindowFilterWordParameterValues({start: 0n, end: 0n})
});
const outputUsage = Buffer.STORAGE | Buffer.COPY_SRC;
const rowsBuffer = device.createBuffer({
  id: 'selected-time-rows', byteLength: Math.max(1, timestamps.length) * 4, usage: outputUsage
});
const countBuffer = device.createBuffer({id: 'selected-time-count', byteLength: 4, usage: outputUsage});
const overflowBuffer = device.createBuffer({id: 'selected-time-overflow', byteLength: 4, usage: outputUsage});
const rows = graph.createDataView(
  graph.importBuffer({id: 'selected-time-rows', byteLength: rowsBuffer.byteLength, usage: outputUsage}, rowsBuffer),
  {format: 'uint32', length: timestamps.length}
);
const count = graph.createDataView(
  graph.importBuffer({id: 'selected-time-count', byteLength: countBuffer.byteLength, usage: outputUsage}, countBuffer),
  {format: 'uint32', length: 1}
);
const overflow = graph.createDataView(
  graph.importBuffer({id: 'selected-time-overflow', byteLength: overflowBuffer.byteLength, usage: outputUsage}, overflowBuffer),
  {format: 'uint32', length: 1}
);
graph.add(new GPUTimeWindowFilter({
  timestamps,
  window: windowParameters.importToGraph(graph),
  additionalPredicates: [{kind: 'selection', mask: timeValidity}],
  output: {rows, status: {count, overflow}}
}));

// Keep the same ordered timestamp/validity chunks for relative-time conversion.
const relativeColumns = timestamps.data.map((chunk, chunkIndex) => {
  const id = `relative-time-${chunkIndex}`;
  const buffer = device.createBuffer({
    id, byteLength: Math.max(1, chunk.length) * 4, usage: outputUsage
  });
  const output = graph.createDataView(
    graph.importBuffer({id, byteLength: buffer.byteLength, usage: outputUsage}, buffer),
    {format: 'float32', length: chunk.length}
  );
  if (chunk.length > 0) {
    graph.add(new GPUTimeWordsRelative({
      id,
      timestamps: chunk,
      mask: timeValidity.data[chunkIndex],
      temporalContext: upload.temporalContext,
      origin: 0n, // milliseconds after Unix epoch here; use the prepared column's unit
      output
    }));
  }
  return {buffer, output};
});
// Call only after encoding and submitting the graph; do not rewrite inputs during these reads.
async function readSelectedRows() {
  const [countWords, overflowWords] = await Promise.all([
    readPackedBuffer(countBuffer, 0, 'uint32', 1),
    readPackedBuffer(overflowBuffer, 0, 'uint32', 1)
  ]);
  const selectedRows = await readPackedBuffer(rowsBuffer, 0, 'uint32', countWords[0]);
  return {rows: selectedRows, count: countWords[0], overflow: overflowWords[0]};
}
async function readRelativeChunks() {
  return Promise.all(relativeColumns.map(({buffer, output}) =>
    readPackedBuffer(buffer, 0, 'float32', output.length)
  ));
}
// Only the compact prefix is read; any tail beyond status.count is unspecified.
// After reads and compiled-graph destruction: destroy rowsBuffer, countBuffer, overflowBuffer,
// each relativeColumns buffer, windowParameters and upload. These resources are caller-owned.
```

`GPUTimeWordsRelative` accepts the optional chunk-aligned `mask` shown above: a zero mask writes
NaN, while a valid timestamp equal to the origin writes 0. With origin `0n`, `[null,0n,10n]`
therefore produces `[NaN, 0, 10]` milliseconds, with the same source-row layout. The input word
pair `[0,0]` alone is insufficient; omit the mask only for established all-valid input.
Consumers such as `GPUKnoxTest`, `GPUMantelTest`,
`GPUSpaceTimeVariogram`, float-mode `GPUTemporalReduction` and float trajectory paths need
valid-row compaction when their particular input contract has no aligned missing-time mask.
Do not pass a null's stored zero as a real observation.

### `prepareGeometryInput`

Import `prepareGeometryInput` and its types from
`@luma-contributors/analysis/gpu-spatial-analysis`. The native Arrow adapter takes
`(vector, options)`, accepts Polygon and MultiPolygon nesting directly, and produces compact
metric XY geometry. This typed example checks null, valid empty and nonempty rows across two
Arrow batches. Coordinates are already local planar metres stored relative to the declared UTM
origin `[500000, 0]`: the adapter neither projects coordinates nor subtracts `frame.origin`.

```ts
import * as arrow from 'apache-arrow';
import {
  prepareGeometryInput,
  type PrepareGeometryInputOptions,
  type PreparedGeometryInput
} from '@luma-contributors/analysis/gpu-spatial-analysis';

const coordinateType = new arrow.FixedSizeList(2, new arrow.Field('xy', new arrow.Float64()));
const ringType = new arrow.List(new arrow.Field('vertices', coordinateType));
const polygonType = new arrow.List(new arrow.Field('rings', ringType));
const multiPolygonType = new arrow.List(new arrow.Field('polygons', polygonType));
const shell: [number, number][] = [[10, 20], [30, 20], [10, 35], [10, 20]];
const polygon = [shell];
const polygonColumn = arrow.vectorFromArray([null, []], polygonType).concat(
  arrow.vectorFromArray([polygon], polygonType)
);
const multiPolygonColumn = arrow.vectorFromArray([null, []], multiPolygonType).concat(
  arrow.vectorFromArray([[polygon]], multiPolygonType)
);
const options: PrepareGeometryInputOptions = {
  sourceRowOffset: 40,
  sourceIds: Uint32Array.of(700, 701, 702),
  frame: {
    id: 'local-utm-33n', kind: 'crs', crs: 'EPSG:32633', origin: [500000, 0],
    anchor: 'relative-origin', precision: 'float32-origin-relative',
    unit: {name: 'meters', metersPerUnit: 1}
  }
};
const preparedPolygon: PreparedGeometryInput = prepareGeometryInput(polygonColumn, {
  ...options, encoding: 'geoarrow.polygon'
});
const preparedMultiPolygon: PreparedGeometryInput = prepareGeometryInput(multiPolygonColumn, {
  ...options, encoding: 'geoarrow.multipolygon'
});
// Both: validity [0, 1, 1], empty [0, 1, 0], batchRowOffsets [0, 2, 3].
// Both: sourceRows [40, 41, 42], sourceIds [700, 701, 702].
// Both: geometry.positions [10, 20, 30, 20, 10, 35, 10, 20] (metres).
// Both: geometry.ringOffsets [0, 4]; roundedCoordinateCount = collapsedPositionCount = 0.
// Polygon: featureOffsets [0, 0, 1, 2], polygonOffsets [0, 0, 1].
// MultiPolygon: featureOffsets [0, 0, 0, 1], polygonOffsets [0, 1].
```

`validity = 0` identifies a source null; `validity = 1, empty = 1` identifies a valid empty
geometry. Neither becomes a nonempty feature. Polygon's empty row retains an empty polygon part;
MultiPolygon's empty row has zero parts. Both preserve source feature slots. Malformed schemas or
coordinates reject rather than being silently skipped. Only native XY is supported; project
spherical edges before preparation. Supply authoritative Arrow extension metadata through `field`
when available; conflicting explicit encoding or metadata is rejected.

`batchRowOffsets` starts each batch in the supplied vector's row domain and ends with its length;
it remains vector-relative after slicing. `sourceRows` adds `sourceRowOffset`, so set that offset
to the slice's actual source-row start. Optional `sourceIds` must contain one caller ID per supplied
row, including null and empty rows; otherwise IDs default to `sourceRows`. IDs and frame metadata
are copied. Geometry offsets are zero-based vertex, ring and polygon-part indices, with terminal
offsets, rather than caller IDs.
The frame's `anchor` and `precision` explicitly identify origin-relative storage. Coordinate
compatibility rejects an otherwise identical relative frame with origin `[600000, 0]`; equal
frame IDs alone do not make coordinates stored against different origins interchangeable.

The default `precision: 'reject-loss'` rejects any coordinate changed by Float32 conversion.
To accept rounding explicitly, write
`prepareGeometryInput(polygonColumn, {...options, encoding: 'geoarrow.polygon', precision: 'allow-loss'})`.
Inspect `roundedCoordinateCount` (changed scalar coordinates) and `collapsedPositionCount`
(distinct original XY positions beyond one per rounded XY; repeated vertices do not add loss).
For example, `(1073741824, 1)` and `(1073741825, 1)` collapse to the same Float32 XY under
`allow-loss`: one rounded coordinate and one collapsed position. Prefer subtracting an appropriate
local origin in binary64 before preparation when that loss is unacceptable.
For Polygon/MultiPolygon, both policies reject nonfinite XY or finite values that overflow Float32.
The helper also accepts other native geometry
types: a native Point `[NaN, NaN]` is a valid empty point under either policy, returning
`validity = [1]` and `empty = [1]` rather than a rejection.

### `preparePolygonContiguityTopology`

Import from `@luma-contributors/analysis/gpu-spatial-analysis`. Accepts source-aligned Arrow
polygon lists, retaining empty and invalid rows, batches and diagnostics. Original binary64 XY
coordinates are interned into Float32 integer identity tokens. They are equality-only values:
never use them for rendering, metric distances or positive snap tolerance. Z is ignored; signed
zero compares equal. No snapping, noding, repair or Float32 coordinate rounding changes adjacency.
Limits are 4096 source rows, 65536 inspected vertex occurrences (including closures and rejected
rows), and 1 MiB of batch metadata. Temporary preparation reserves a conservative 24 MiB.

This preparer requires a `PreparedPolygonTopologySource` wrapper whose `batches[].polygons`
contains each original native Arrow Polygon or MultiPolygon vector. Native Polygon nesting is
`source row → ring → XY[Z…]`; native MultiPolygon adds a polygon-part level. Pass either vector
directly in the batch, retaining source validity and batch offsets. The preparer reads original
XY and **produces** equality tokens; do not replace its input coordinates with tokens or metric
Float32 positions. For metric coordinates use
[`prepareGeometryInput(vector, options)`](#preparegeometryinput) above.

Both native forms below retain `[null, [], triangle]` without manually adapting individual rows:

```ts
import * as arrow from 'apache-arrow';
import {
  preparePolygonContiguityTopology,
  type PreparedPolygonTopologySource
} from '@luma-contributors/analysis/gpu-spatial-analysis';

const coordinateType = new arrow.FixedSizeList(2, new arrow.Field('xy', new arrow.Float64()));
const ringType = new arrow.List(new arrow.Field('vertices', coordinateType));
const polygonType = new arrow.List(new arrow.Field('rings', ringType));
const multiPolygonType = new arrow.List(new arrow.Field('polygons', polygonType));
const triangle: [number, number][] = [[10, 20], [30, 20], [10, 35], [10, 20]];
const nativePolygon = arrow.vectorFromArray([null, [], [triangle]], polygonType);
const nativeMultiPolygon = arrow.vectorFromArray([null, [], [[triangle]]], multiPolygonType);

function prepareNativeTopology(polygons: arrow.Vector) {
  const topologySource: PreparedPolygonTopologySource = {
    metadata: {rowCount: 3, invalidRows: []},
    batches: [{id: 'native', rowCount: 3, sourceRowOffset: 0, polygons}]
  };
  return preparePolygonContiguityTopology(topologySource);
}
const polygonTopology = prepareNativeTopology(nativePolygon);
const multiPolygonTopology = prepareNativeTopology(nativeMultiPolygon);
// Both: geometryMask [0, 0, 1], invalidRows [], positions [0, 0, 1, 0, 2, 0].
// The three distinct original XY vertices produce three equality tokens; closure is implicit.
// Both: featureOffsets [0, 0, 0, 1], polygonOffsets [0, 1], ringOffsets [0, 3].
```

The returned zero-based offsets are:

| Array | Index space |
| --- | --- |
| `ringOffsets` | Position rows delimiting each ring. |
| `polygonOffsets` | Ring rows delimiting each polygon part. |
| `featureOffsets` | Polygon-part rows delimiting each original source feature. |
| `batches[].sourceRowOffset` | Original source-row start of each retained batch. |

Each offsets array includes its terminal offset. Feature slots are not compacted: empty,
source-invalid and malformed rows retain their source row with empty ranges and `geometryMask = 0`.
Retain original Arrow validity to distinguish null from a valid empty geometry, and
`invalidRows` for source-invalid or malformed geometry; the mask alone cannot make that distinction.
Caller feature IDs remain separate from these row indices. This preparer returns equality tokens,
not a metric coordinate buffer or a spatial frame declaration.

For example, source vertices `(10,20)` and `(30,20)` can become tokens `(0,0)` and `(1,0)`.
Their planar distance is 20 source units; the token distance is 1 and has no distance unit.
Keep original XY separately for distance, area, refinement or rendering, project to an appropriate
planar frame when necessary, and subtract a local origin before converting metric XY to Float32.
Never reuse token positions for a metric port, even though their array type is Float32.

### `computePolygonContiguityOnCPU`

Optional CPU route on the prepared tokens, for queen or whole-edge rook adjacency. GPU consumers
use `GPUContiguityWeights` with zero snap tolerance. The CPU result reports final
capacity loss in `status.overflow` and the exact slot requirement in `status.requiredCount`. This
route has no intermediate candidate overflow. geometryMask still controls
which source rows participate in downstream statistics.

### `GPUWideNumericScan`

Import from `@luma-contributors/analysis/dataframe-analysis`. Chunked packed word phases run in
parallel over rows. The caller initializes result words. Props and chunk layout are compile-time;
no per-frame CPU row validation occurs. Command-node budget and device binding limits are checked.
Fewer than 2^32 total rows keep counters and carry sums exact. No readback occurs in the contributor.

### Exact aggregates and CPU references

`aggregateInteger64OnGpu` / `aggregateInteger64Range` preserve signed or unsigned 64-bit extrema,
null counts and bigint sums (44-byte summaries). `aggregateDecimal128OnGpu` /
`aggregateDecimal128Column` preserve 128-bit unscaled bigint values, decimal scale and exact
scale-normalized comparisons (76-byte summaries). Optional selection skips unselected nulls.
`summarizeInteger64Reference` and `summarizeDecimal128Reference` are pure CPU reference routes.
Integer reference inputs are normalized to the declared word width, as resident storage is.

`makeDecimal128WordVectors(device, column)` accepts an Arrow `Vector<Decimal>` only when its
physical `bitWidth` is 128. Every chunk must also be Decimal128 and have the column's scale;
mixed-width or mixed-scale concatenations reject before any device allocation. The upload rejects
other types and widths, including Decimal256. It retains each source chunk, including empty chunks.
When any chunk has nulls, `validity` has the same topology as `words`; otherwise it is absent and consumes no
device memory. Call the returned `destroy()` to free all uploaded vectors.
`aggregateDecimal128Column` owns and frees its upload.

### Narrowing diagnostics

`diagnoseFloat32Narrowing` and `scanWideNumericColumnOnGpu` share `NumericNarrowingReport`.
`maximumAbsoluteError` is an exact bigint for Int64/Uint64, including zero; subtraction happens
before conversion to Number. Integer rounding is nearest-even and avoids binary64 double rounding.
For other formats it is a number: CPU JS absolute difference, or a GPU Float64 lower bound within
relative 2^-31. Overflow and nonfinite counts are separate from the maximum finite error. The GPU
reads six words (24 bytes) over two phases. `mergeNumericNarrowingReports` keeps the maximum across
chunks. `summarizeNarrowingReference` independently rounds integer candidates for qualification.
Arithmetic, WGSL-expression and upload helpers are public building blocks, not catalog operators.
