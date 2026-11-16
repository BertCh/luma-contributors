---
title: GPU Dataframe analysis contributors
description: Column profiles, classification, group statistics, key joins, composite indicators, time windows, and pair statistics.
---


# GPU Dataframe analysis contributors


## Overview

The `@luma-contributors/analysis/dataframe-analysis` entry point exports analysis contributors for GPU-resident
columns: column profiles, quantiles and class breaks, color scales, group statistics, key joins,
composite scores, inequality measures, calendar buckets, period indices, cube layout transposes, relative times,
time-window filters, temporal reductions,
and pair-statistics tools (variograms, correlograms, Ripley's K, point-pattern indices). Each is an
algorithm or workflow that declares resources and command nodes into a caller-owned
`GPUCommandGraph` through `getCommandNodes(graph)` (`GPUCommandNodeProducer`). It never compiles,
submits, encodes, or reads back.

## When to use it

Use these contributors for the fixed analyses that map and dashboard applications repeat every
frame, where inputs are packed graph views and results must stay on the GPU. Use the
[`GPUDataFrame` query API](https://luma.gl/docs/api-reference/experimental/gpu-dataframe-operations) when the question is a general expression,
filter, or aggregation plan.

- `GPUGroupStatistics` groups by a 32- or 64-bit key and computes many statistics per group (moments,
  median, percentiles, mode, unique count). Use `GPUDataFrameGroupedAggregationQuery` for dense
  categorical keys and standard aggregations.
- `GPUKeyJoin` joins a left table to a right table by key and supports 1:n aggregates and 64-bit
  keys. Use `GPUDataFrameJoinQuery` for a unique right-side key lookup between dataframes.
- `GPUColumnProfile` fuses count, nulls, moments, histogram, distinct estimate, and top categories
  into one statistics panel. Use `GPUDataFrameHistogramQuery` when you only need an explicit
  histogram.
- `GPUTimeWindowFilter` publishes compacted IDs, masks, and fade weights for a window that moves
  every frame, including exact `Int64` time words. Express a fixed time predicate as a dataframe
  filter expression instead.

## Quick start

```ts
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
import {
  GPUTimeWindowFilter,
  getGPUTimeWindowFilterParameterValues,
  GPU_TIME_WINDOW_FILTER_PARAMETER_LENGTH
} from '@luma-contributors/analysis/dataframe-analysis';

const graph = new GPUCommandGraph(device, {id: 'time'});
const window = new GPUParameterBuffer(device, {
  id: 'time-window',
  format: 'float32',
  length: GPU_TIME_WINDOW_FILTER_PARAMETER_LENGTH
});

// `timestamps` and `output` are graph views created on `graph`.
graph.add(new GPUTimeWindowFilter({timestamps, window: window.importToGraph(graph), output}));
const compiled = await graph.compileAsync(); // once

// every frame: rewrite parameters, encode the same compiled graph
window.write(getGPUTimeWindowFilterParameterValues({start: now - 60, end: now}));
compiled.encode(device.commandEncoder, {parameters: undefined});
```

`encode()` only records; submit with `device.submit()`. [Running contributors](./gpu-spatial-analysis/concepts.md#running-contributors)
covers `compileAsync()`, rebuilding without a stall, submission and why `parameters` is `undefined`.

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
- **Stable IDs.** Result IDs are the caller's `sourceIds[row]` (or tile IDs) when given and zero-based
  rows otherwise. Node and transient IDs are `${id}-<step>`, so two instances in one graph need
  different `id` props.
- **One import per buffer.** Import a buffer once (for example with `graph.importGPUData()` or
  `GPUParameterBuffer.importToGraph()`) and pass the returned view to every contributor that reads
  it, so the graph tracks hazards on one logical handle.
- **Shared contracts.** Four shapes mean the same on every analysis page:
  - **Status.** Every GPU-written status word sits in the `status` member of the bag that owns it:
    `output.status.overflow`, `output.status.converged`, and `output.status.code` for model fits.
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

Filter timestamped rows against an inclusive time window, producing a row mask and compact accepted ids.

::lens{scenario=gpu-dataframe.time-window-filter rung=result caption="588 of 1,500 observations fall inside 2023-06-01 to 2023-09-01 (rows 513 to 1100, CPU check)."}

Filters timestamped rows (instants) or time intervals (trail segments) against a per-frame window
and publishes stable compacted IDs, a clamped count, and optional mask, fade weights, segment clip
fractions, per-track visible counts, and an indirect `instanceCount`. All outputs live in one
`output` prop (`GPUTimeWindowFilterOutput`: `GPUCompactOutput` plus `outputMask`, `fadeWeights`,
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

### Sorted range mode

When the instants are sorted by time (the usual layout of a recorded track or event log), set
`sorted: true`. The accepted rows are then one contiguous range, found by two binary searches in a
single one-thread kernel: O(log n) per frame for any row count, with no mask pass, scan or
compaction. The range is written to `output.rowRange` (`[firstRow, rowCount]`, `rowCount` unclamped),
`output.drawFirstInstance` (the draw record's `firstInstance`, so the instance index is the row
index) and the usual status words and `output.drawInstanceCount`; `output.ids` receives only its
first `min(count, ids.length)` entries (`sourceIds` if given, else row indices) from a kernel that
returns early past the count. A caller that reads only the range passes a small `ids` view. Acceptance
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
graph.add(new GPUTimeWindowFilter({
  sorted: true,
  timestamps: sortedInstants,
  window: window.importToGraph(graph),
  output: {...compactOutput, rowRange, drawInstanceCount}
}));
```

```ts
graph.add(new GPUTimeWindowFilter({
  timestamps: segmentStarts, endTimestamps: segmentEnds,
  window: window.importToGraph(graph),
  // `output` is a `GPUTimeWindowFilterOutput`: the compact list plus the optional views.
  output: {
    ...compactOutput, // ids, status: {count, overflow, optional requiredCount}
    fadeWeights, clipFractions,
    drawInstanceCount: graph.importGPUData('trail-count', drawCommands.getInstanceCountData(0))
  }
}));
window.write(getGPUTimeWindowFilterParameterValues({start: now - trailLength, end: now, startFadeDuration: trailLength}));
```

```ts
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

{/* parameters: GPUTimeWindowFilter generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUTimeWindowFilter` parameters

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

#### How it computes

{/* lens:GPUTimeWindowFilter */}

::lens{scenario=gpu-dataframe.time-window-filter rung=datum bind=page datum=513 caption="Observation 513, 2023-06-01 10:57, is the first row inside the window."}
::lens{scenario=gpu-dataframe.time-window-filter rung=formula bind=page datum=513}
::lens{scenario=gpu-dataframe.time-window-filter rung=kernel caption="Look at the classify kernel: the window start and end are read from the parameter buffer, not baked into WGSL."}
::lens{scenario=gpu-dataframe.time-window-filter rung=graph caption="One classify node, the visibility workflow that compacts ids, and a publish node that clamps the count."}

## `GPUColumnQuantiles`, `GPUClassBreaks`, `GPUColorScale`, and `GPUBivariateClassification`

Choropleth classification on the GPU, every frame, with no readback: exact quantiles and
percentile filters, class breaks, and colour scales that write a packed `rgba8` column a deck.gl
layer can bind as an attribute. Each output can feed the next contributor as a graph view, so a brush
or filter change recolours the map without a CPU round trip.

```ts
graph.add(new GPUColumnQuantiles({
  values, mask, quantileCount: 3,
  parameters: quantileParameters.importToGraph(graph),
  output: {quantiles, validCount, outputMask, filterBounds} // outputMask may be a transient
}));
graph.add(new GPUClassBreaks({
  values, mask: outputMask, maximumClassCount: 9,
  methods: ['quantile', 'natural-breaks', 'head-tail'], // default: every method
  parameters: breakParameters.importToGraph(graph),
  output: {breaks, classCount, classCounts}
}));
graph.add(new GPUColorScale({
  values, mask: outputMask, domain: breaks, domainCount: classCount, palette,
  parameters: scaleParameters.importToGraph(graph),
  maximumDomainCount: 10, maximumPaletteCount: 9,
  output: {colors, classIndices, classCounts}
}));
quantileParameters.write(getGPUColumnQuantilesParameterValues({quantiles: [0.25, 0.5, 0.75], filterRange: [0.05, 0.95]}));
breakParameters.write(getGPUClassBreaksParameterValues({method: 'natural-breaks', classCount: 7}, 9));
scaleParameters.write(getGPUColorScaleParameterValues({scale: 'threshold', domainCount: 8, paletteCount: 9}));
// Any of these writes takes effect on the next encoding, with no recompile.
```

- Class rule shared by all four contributors: breaks are edges `e[0..k]` (`e[0]` the smallest and
  `e[k]` the largest finite value). A value is in class `i` when `i` inner edges `e[1..k-1]` are at
  or below it (d3 `scaleThreshold`/`bisectRight`), compared on order-preserving `u32` keys, so
  `-0` sorts below `+0`. Values outside `[e[0], e[k]]` clamp to the end classes. NaN and masked
  rows are skipped (class `0xffffffff`, the no-data colour). Colours are `r | g << 8 | b << 16 | a << 24`.
- `GPUColumnQuantiles`: exact order statistics by four 8-bit radix-select passes over the ordered
  keys (no sort): up to 64 per-frame probabilities, `'lower' | 'higher' | 'nearest' | 'linear' |
  'midpoint'` (numpy names; `'linear'` is d3 and R-7). The rank is `fround(fround(n - 1) * p)` and
  every interpolation step is rounded separately (no fused multiply-add), so results match a
  sorted CPU column bit for bit. `filterRange` gives deck.gl/kepler
  lower/upper percentile filtering: bounds `x[floor(n * lower)]` and `x[ceil(n * upper) - 1]`, and a
  `outputMask` of the rows inside them. At most 2^24 rows. About 1.8 ms at 1M rows and 3.5 ms at
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
    exceed 2^24 rows (up to one storage binding, 2^25 rows at the default limit) when no filter
    output is requested; the filter ranks stay float32. `GPUClassBreaks` and `GPUGroupStatistics`
    use the float32 rank only. Fixture and generator:
    `modules/analysis/test/oracles/dataframe-quantile-rational-rank.py`.
- One order-statistic policy for `GPUColumnQuantiles`, the `GPUClassBreaks` quantile and box-plot
  methods, and the `GPUGroupStatistics` medians and percentiles: the sample is the finite unmasked
  values (NaN and ±Infinity are missing), `-0` equals `+0`, and a probability that is NaN or outside
  `[0, 1]`, or an empty sample, gives NaN. The result is `numpy.quantile(x[numpy.isfinite(x)], p)`;
  numpy itself keeps infinities and raises on an invalid `p`. When `x[high] - x[low]` overflows f32
  the convex form `x[low] (1 - t) + x[high] t` keeps the result finite.
- `GPUClassBreaks`: per-frame `method` and `classCount` among the compiled `methods`:
  `'equal-interval'`, `'quantile'` (exact linear quantiles at `i / k`), `'standard-deviation'`
  (classes of `interval` sample standard deviations centred on the mean, as mapclassify `StdMean`
  and R classInt `'sd'`), `'head-tail'` (Jiang's
  breaks with a per-frame head ratio, default 0.4; 1 gives mapclassify's rule), `'box-plot'`
  (six classes, per-frame hinge), `'maximum-breaks'` (midpoints of the `k - 1` largest gaps
  between distinct sorted values), `'natural-breaks'` (Fisher-Jenks, see below), and
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
    `[min, max]` (2 to 2048, default 2048) sums count, position and squared position of each bin as
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
  d3 multi-stop scale. A `domainCount` view (for example `GPUClassBreaks.classCount`) holds a
  class count `k`, and the domain then has `k + 1` edges. `'step'` colours are exact; `'linear'`
  blends round half up (within 1 per channel of a double-precision reference).
- `GPUBivariateClassification`: two columns and two edge views give `classId = yClass *
  classCountX + xClass` (up to 16 per axis), a colour from an `n x n` palette, per-class counts,
  and optional value-by-alpha (`alphaValues` mapped through a per-frame `[low, high]` to
  `[minimumAlpha, 1]` times the palette alpha).
- Integer columns: `GPUClassBreaks.values` accepts `uint32` and `sint32` in addition to `float32`. One
  internal pass converts the column to `float32` (exact to 2^24) and every method behaves as for that
  float column. `GPUColorScale.values` accepts `sint32` (always numeric) and `uint32` with
  `integerValues: 'numeric'`, which reads the integers as numbers and applies the `scale` parameter; the
  default `'ordinal'` keeps `uint32` as category codes. The value format is topology (fixed at compile
  time). This is what lets a count column (for example dense `GPUGroupStatistics` `counts`) feed a
  choropleth without a float copy.
- Per-frame settings are the CPU objects `GPUColumnQuantilesSettings`, `GPUClassBreaksSettings`,
  `GPUColorScaleSettings`, and `GPUBivariateClassificationSettings`, packed by the matching
  `getGPU<Class>ParameterValues(settings, target?)`.
- Inputs must be single packed views; chunked vectors are rejected.

### `GPUBivariateClassification`

Classify two columns into an n by n grid and colour each row from a bivariate palette.

::lens{scenario=gpu-dataframe.bivariate-classification rung=result caption="The 3 by 3 grid puts 218 tracts in the low-poverty, low-uninsured class and 64 in the opposite corner (CPU check)."}

{/* parameters: GPUBivariateClassification generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUBivariateClassification` parameters

- Packer: `getGPUBivariateClassificationParameterValues(settings)`, writes `GPU_BIVARIATE_CLASSIFICATION_PARAMETER_LENGTH` (8) `float32` values.

Settings of `getGPUBivariateClassificationParameterValues`:

| Field | Type | Description |
|---|---|---|
| `classCountX` | `number` | Active classes along X, `1..maximumClassCount`; `breaksX` holds `classCountX + 1` edges. |
| `classCountY` | `number` | Active classes along Y, `1..maximumClassCount`; `breaksY` holds `classCountY + 1` edges. |
| `noDataColor` (optional) | `number` | Packed `rgba8` colour (`r \| g << 8 \| b << 16 \| a << 24`) for no-data rows. Default `0`. |
| `valueByAlpha` (optional) | `GPUBivariateValueByAlpha` | Enables value-by-alpha. Omit to leave the palette alpha unchanged. |

{/* /parameters */}

#### How it computes

{/* lens:GPUBivariateClassification */}

::lens{scenario=gpu-dataframe.bivariate-classification rung=datum bind=page datum=16 caption="Tract 16 falls in joint class 8: high poverty and high uninsured, shared by 64 tracts (CPU check)."}
::lens{scenario=gpu-dataframe.bivariate-classification rung=formula bind=page datum=16}
::lens{scenario=gpu-dataframe.bivariate-classification rung=kernel caption="Look at classY * classCountX + classX: each axis counts its inner edges at or below the value."}
::lens{scenario=gpu-dataframe.bivariate-classification rung=graph caption="A single bivariate kernel reading two value columns, two edge lists, a palette and parameters."}

### `GPUClassBreaks`

Compute choropleth class edges and per-class counts from a float column, with a switchable method.

::lens{scenario=gpu-dataframe.class-breaks rung=result caption="Equal-interval breaks split 788 valid tracts into classes of 257, 250, 168, 85 and 28 (CPU check)."}

{/* parameters: GPUClassBreaks generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUClassBreaks` parameters

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

#### How it computes

{/* lens:GPUClassBreaks */}

::lens{scenario=gpu-dataframe.class-breaks rung=datum bind=page datum=2 caption="Inner edge e2 is 32.1 percent, the lower bound of the third class (CPU check)."}
::lens{scenario=gpu-dataframe.class-breaks rung=formula bind=page datum=2}
::lens{scenario=gpu-dataframe.class-breaks rung=kernel caption="Look at the equal-interval branch: edges step from the sample minimum by (max - min) / k."}
::lens{scenario=gpu-dataframe.class-breaks rung=graph caption="Moment, quantile and method nodes compiled for the chosen methods, publishing breaks and counts."}

### `GPUScalarScale`

[`GPUScalarScale`](#gpuscalarscale) maps prepared numeric columns to scalar rendering
attributes using replayable linear, signed square-root, same-sign logarithmic, native fixed,
and categorical point scales. It preserves source chunks and validity without source readback.

The optional input row filter is `mask` (`uint32`, one entry per source row; nonzero keeps the row, zero maps the row to `nullValue` with output validity 0). `validity` is accepted as a `@deprecated` alias with the same meaning; `mask` wins when both are set. Output `validity` is unchanged. Construction errors start with the contributor ID and name the failed limit, for example `scalar-scale mask must hold exactly one entry per source row`.

{/* parameters: GPUScalarScale generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUScalarScale` parameters

- Packer: `getGPUScalarScaleParameterValues(settings)`, writes `GPU_SCALAR_SCALE_PARAMETER_LENGTH` (4) `float32` values.

Settings of `getGPUScalarScaleParameterValues`:

| Field | Type | Description |
|---|---|---|
| `scale` | `GPUScalarScaleType` | - |
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

{/* parameters: GPUColorScale generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUColorScale` parameters

- Packer: `getGPUColorScaleParameterValues(settings)`, writes `GPU_COLOR_SCALE_PARAMETER_LENGTH` (9) `float32` values.

Settings of `getGPUColorScaleParameterValues`:

| Field | Type | Description |
|---|---|---|
| `scale` | `GPUColorScaleType` | Scale type. |
| `domainCount` | `number` | Number of active domain entries, `1..maximumDomainCount`. Ignored when the contributor was given a `domainCount` view, which then supplies a CLASS count `k` and the domain has `k + 1` edges. |
| `paletteCount` | `number` | Number of active palette entries, `1..maximumPaletteCount`. |
| `interpolation` (optional) | `GPUColorScaleInterpolation` | `'step'` picks `palette[class]` exactly; `'linear'` blends adjacent entries per channel. Default `'step'`. |
| `clamp` (optional) | `boolean` | Clamp out-of-domain values to the first or last colour instead of no-data. Default `false`. |
| `extrapolate` (optional) | `boolean` | Extrapolate continuous scales along the first/last palette segment, saturating color bytes. Default false; clamp true takes precedence. Ignored by discrete scales. |
| `logMode` (optional) | `'positive-floor' \| 'domain-sign'` | Log transform: legacy positive floor (default), or native same-sign positive/negative domain. Native mode ignores logFloor and requires finite nonzero same-sign domain stops. |
| `noDataColor` (optional) | `number` | Packed `rgba8` colour (`r \| g << 8 \| b << 16 \| a << 24`) for no-data rows. Default `0`. |
| `logFloor` (optional) | `number` | Value that replaces `v <= 0` (and non-positive domain entries) on the `log` scale, as kepler does with a small positive floor. `NaN` treats such rows as no-data. Default `1e-5`. |
| `exponent` (optional) | `number` | Exponent of the `pow` scale. Default `1`. |

{/* /parameters */}

#### How it computes

{/* lens:GPUColorScale */}

::lens{scenario=gpu-dataframe.color-scale rung=datum bind=page datum=1 caption="Tract 1 has 43 percent poverty: above the 40 percent edge, class 4, the darkest blue."}
::lens{scenario=gpu-dataframe.color-scale rung=formula bind=page datum=1}
::lens{scenario=gpu-dataframe.color-scale rung=kernel caption="Look at the binary search over inner domain edges that yields the class."}
::lens{scenario=gpu-dataframe.color-scale rung=graph caption="A single colour-scale kernel reading values, domain, palette and parameters."}

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

{/* parameters: GPUColumnQuantiles generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUColumnQuantiles` parameters

- Packer: `getGPUColumnQuantilesParameterValues(settings)`, writes `getGPUColumnQuantilesParameterLength(...)` `float32` values.

Settings of `getGPUColumnQuantilesParameterValues`:

| Field | Type | Description |
|---|---|---|
| `quantiles` | `ArrayLike<number>` | Probabilities in `[0, 1]`; others (and NaN) produce a NaN quantile. |
| `interpolation` (optional) | `GPUColumnQuantileInterpolation` | Defaults to `'linear'`. |
| `filterRange` (optional) | `[number, number]` | Percentile filter as fractions `[lower, upper]` in `[0, 1]`. Defaults to `[0, 1]`. |
| `quantileCount` (optional) | `number` | Number of probability slots. Defaults to `quantiles.length`. Unused slots become NaN. |

{/* /parameters */}

#### How it computes

{/* lens:GPUColumnQuantiles */}

::lens{scenario=gpu-dataframe.column-quantiles rung=datum bind=page datum=0 caption="First quartile (p = 0.25): 14.1 percent of residents below 150% of poverty."}
::lens{scenario=gpu-dataframe.column-quantiles rung=formula bind=page datum=0}
::lens{scenario=gpu-dataframe.column-quantiles rung=kernel caption="Look at the interpolation: rank h, its fractional part, then the two neighbouring order statistics."}
::lens{scenario=gpu-dataframe.column-quantiles rung=graph caption="Four radix passes, each a histogram plus a select node, then a finish node and the filter mask."}

## `GPUColumnProfile`

Chunked GraphVectorView columns support up to 4,294,967,295 logical rows, subject to device
bindings. Counts are uint32; float32 statistical fields remain rounded.

Profile several columns at once: counts, nulls, extent, moments, histogram and distinct estimate.

::lens{scenario=gpu-dataframe.column-profile rung=result caption="Median household income has 19 suppressed tracts; the three rate columns have 3 each (CPU check)."}

A dataset statistics panel (kepler.gl / Foursquare Studio field summaries) for several columns of
one table, recomputed on the GPU every frame so it follows filters and brushes without readback
of the rows.

```ts
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

{/* parameters: GPUColumnProfile generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPUColumnProfile` parameters

- Packer: `getGPUColumnProfileParameterValues(domains, columnCount?)`, writes `getGPUColumnProfileParameterLength(...)` `float32` values.
  - `domains` (`readonly (GPUColumnProfileDomain | undefined)[]`): One entry per column, `undefined` for fully automatic.
  - `columnCount` (`number`): Number of columns. Defaults to `domains.length`.

{/* /parameters */}

#### How it computes

{/* lens:GPUColumnProfile */}

::lens{scenario=gpu-dataframe.column-profile rung=datum bind=page datum=3 caption="Column 3, median household income: 772 non-null tracts, 19 nulls."}
::lens{scenario=gpu-dataframe.column-profile rung=formula bind=page datum=3}
::lens{scenario=gpu-dataframe.column-profile rung=kernel caption="Look at the Welford and Chan merges: moments never come from raw squared sums."}
::lens{scenario=gpu-dataframe.column-profile rung=graph caption="Per-column moment tiles, a merge, domain, histogram and HyperLogLog nodes, then a finish node."}

## `GPUGroupStatistics`

Reducible statistics use tiled scratch controlled by scratchTileSize. Compact tiled padded rows are
at most 4,294,967,295; packed output and exact order-statistic scratch remain device-binding
bounded.

Groups rows by a key and computes count, mean, variance and median of a value column per group, ascending by key.

::lens{scenario=gpu-dataframe.group-statistics rung=result caption="Mean tract poverty runs from 7.4 percent in community area 12 to 70.2 percent in area 37 across 77 areas (CPU check)."}

Groups rows by a 32- or 64-bit key and computes kepler-style statistics per group and value column on the GPU: count, sum, mean, minimum, maximum, variance, standard deviation, skewness, kurtosis, median, percentiles, mode, unique count and a per-row group z-score. Use it for "aggregate this column by that key" panels, choropleth values keyed by a region id or an H3 or Quadbin cell, and per-group outlier styling.

```ts
const fractions = new GPUParameterBuffer(device, {id: 'fractions', format: 'float32', length: 3});
graph.add(
  new GPUGroupStatistics({
    keys, // uint32, or uint32x2 (low, high) words; all-ones is "no key"
    mask, // optional uint32
    columns: [
      {
        values, // float32 per row; non-finite rows are excluded for this column only
        statistics: ['count', 'mean', 'variance', 'median', 'percentiles', 'mode', 'zScore'],
        output: {counts, means, variances, medians, percentiles, modes, zScores}
      }
    ],
    variance: 'sample', // or 'population'
    percentiles: fractions.importToGraph(graph),
    output: {keys: groupKeys, counts: groupCounts, status: {count, overflow, requiredCount}} // GPUGroupStatisticsOutput
  })
);
fractions.write(Float32Array.of(0.1, 0.5, 0.9)); // no recompile
```

- Output groups are ascending by key, capacity is `output.keys.length`, at most `getGPUGroupStatisticsMaximumGroupCount(device)` (2^23 on a default device; the 16-byte moment sums per group fit one 128 MiB binding) and, with `percentiles`, `capacity * P` at most `getGPUGroupStatisticsMaximumPercentileTotal(device)` (2^25 on a default device). When more groups exist, the smallest keys are kept, `status.overflow` is 1 and `status.requiredCount` is the unclamped number. Tail rows hold the all-ones key, count 0, sums 0 and NaN for every other float statistic.
- `output.counts` counts valid rows (mask set, key valid); each column's `counts` counts finite values. Zero to four columns; each column lists the statistics it needs (a compile-time set) and supplies exactly the matching output views: `counts`, `sums` (signed 64-bit fixed point, `uint32x2`, the sum of `roundHalfEven(value * sumScale)`, default scale 65536: each value is rounded to `1 / sumScale` first) and/or `sumValues` (the sum of the values themselves as float32, from a 96-bit fixed-point accumulator whose unit follows the largest value of the column; it does not depend on `sumScale`) for `sum`, `means`, `minimums`, `maximums`, `variances`, `standardDeviations`, `skewness`, `kurtosis`, `medians`, `percentiles` (`capacity * P`, group-major), `modes`, `uniqueCounts`, and `zScores` (one row per source row).
- Exactness: counts, the fixed-point `sums` words, minimum, maximum, median, percentiles, mode and unique counts are bit-identical to a CPU reference. `sumValues` is the exact sum rounded once to float32 and `means` is that divided by the count (a few float32 units in the last place), at any magnitude: values 1.0e-4, 1.2e-4, 1.4e-4 have mean 1.2e-4. A value keeps all its bits in the sum while its magnitude is at least `max * 2^(h - 71)`, where `max` is the largest value of the column in a kept group and `h` the bit length of the row count. Percentiles use the numpy `linear` rule (`h = (n - 1) p`, `lo + (hi - lo) * (h - floor h)` in f32) and the order-statistic policy shared with `GPUColumnQuantiles`, with `p` read per frame; a fraction that is NaN or outside [0, 1] gives NaN (it is no longer clamped). Mode is the longest run of equal values (ties: smallest value). `-0` is treated as `+0`; NaN and infinities are excluded.
- Moments are corrected two-pass: `d = v - mean` is summed in a fixed order (a binary tree over sorted row blocks), so a second encode is bitwise identical. The signed first and third powers are scaled by a power of two chosen from each group's extremes before reduction and restored afterwards, preventing same-sign branches from overflowing before opposite signs cancel. Values with binary exponent at most 30 use no scaling. Even powers retain their float32 range limits. Each level merges only the group crossing each block midpoint; the work is O(rows), the depth is O(log rows), and a dominant group spreads across workgroups. The four power sums use separate scalar scratch bindings (16 bytes per input row in total); each binding must fit the device storage limit. The row count is also bounded by the published sort maximum of 2^31. `estimatedCommandNodeCount` includes the merge levels and conservative allowances for published sorts, scans and gathers, checked against the shared command-node budget. Central moments about the true mean come from the shifted power sums. Oracle specs require relative tolerances of 1e-6 for means, 2e-5 for variance, and 1e-4 for skewness, kurtosis and z-scores, with small absolute tolerances near zero. Skewness is Fisher-Pearson `g1`, kurtosis is excess `g2`; both are NaN for constant groups, a group whose variance is under 2^-19 of its shifted sum of squares is treated as constant (the shift mean is rounded to float32). Sample variance of one value is NaN; the z-score is `(v - mean) / sd` with the same corrected mean the variance uses, and is 0 when the deviation is 0 or a sample group has one value, NaN for masked, invalid-key, non-finite or capacity-dropped rows.
- Dense mode: `keyCount` (compile-time) requires `uint32` keys in `[0, keyCount)`. `output.keys.length` and every per-group output hold exactly `keyCount` rows, and row `k` is key `k` (`output.keys[k] = k`, `output.status.count` and `status.requiredCount` are `keyCount`, `status.overflow` is 0). Keys with no rows keep a row: `counts` 0, `sums` and `uniqueCounts` 0, and NaN for mean, minimum, maximum, variance, standard deviation, skewness, kurtosis, median, percentiles and mode. Keys outside the range and the reserved all-ones key are skipped like masked rows and are not reported as overflow. It costs one binary search per key instead of a group scan, and suits region IDs or other small dense key spaces (the choropleth recipe uses it so empty polygons keep a row). The default compact table of occupied keys is unchanged.
- Cost: one stable key sort (two 32-bit radix sorts for 64-bit keys); each column that asks for median, percentiles, mode or unique count adds a value sort plus a key re-sort. Chunked reducible inputs use fixed tile scratch; exact order statistics still require device-bound packed scratch. Approximate distinct counts (HyperLogLog) are not included.

#### How it computes

{/* lens:GPUGroupStatistics */}

::lens{scenario=gpu-dataframe.group-statistics rung=datum bind=page datum=34 caption="Community area 35 has 10 tracts, mean poverty 39.3 percent and the widest sample variance, 357.6 (CPU check)."}
::lens{scenario=gpu-dataframe.group-statistics rung=formula bind=page datum=34}
::lens{scenario=gpu-dataframe.group-statistics rung=kernel caption="Look at the corrected two-pass moments: deviations from the fixed-point mean are summed before the variance divides by n - 1."}
::lens{scenario=gpu-dataframe.group-statistics rung=graph caption="A radix sort of rows by key, group ranges from head flags and a scan, then the moment and order-statistic nodes."}

## `GPUKeyJoin`

`count-distinct` counts distinct finite values among matching right rows; missing and nonfinite
values do not contribute. Right tables are bounded by 2,147,483,648 rows, subject to sort and device
limits. Left-row and output ordering remains unchanged.

Looks up each left row's key among unique right keys and gathers the matching right row's value, entirely on the GPU.

::lens{scenario=gpu-dataframe.key-join rung=result caption="1,499 of 1,500 observations pick up their tract's poverty; the one in tract 789 has no estimate and stays unmatched (CPU check)."}

Attribute join of a left table onto a right table by key, for example "join CSV statistics onto
boundaries". Keys are `uint32` or 64-bit `uint32x2` little-endian `(low, high)` words (the same
format on both sides). The all-ones key (`0xffffffff`, or both words all-ones) means "no key" and
never matches; masked rows (`mask == 0`) never match.

```ts
const join = new GPUKeyJoin({
  leftKeys, rightKeys,
  kind: 'left', // or 'inner'
  gather: [{column: rightPopulation, output: leftPopulation}],
  aggregates: [{operation: 'sum', column: rightIncome, output: leftIncomeSum}],
  output: {matched, rightRows, matchCounts, rightMatched}
});
graph.add(join);
```

Valid right rows are sorted by key with a stable radix sort (one pass per key word), so equal keys
stay in right-row order. Segment heads and a scan produce the unique right keys with row ranges;
each left row binary-searches them on the unsigned `(high, low)` order.

- `gather` copies a right column (`float32` or `uint32`, bit for bit) from the smallest matching
  right row. Unmatched rows get NaN or `0xffffffff`.
- `aggregates` run over all matching right rows: `count`, `count-distinct`, `sum`, `mean`, `minimum`, `maximum`.
  `sum` is the exact sum of the finite values in a 96-bit fixed-point accumulator whose unit
  follows the largest value among the valid right rows, rounded once to float32; `mean` divides
  it by the finite count. A column of 1e-4 and a column of 1e6 are both right to float32. The
  optional `sums` words are a separate 64-bit sum of `roundHalfEven(value * sumScale)` (default
  65536): each value is rounded to `1 / sumScale` first, so a value below half a unit adds
  nothing there. Extremes use ordered-key atomics. Non-finite values are excluded per aggregate.
  Unmatched rows get 0 for `count` and `count-distinct`, and NaN otherwise.
- `output.rightRows`, `matchCounts`, `matched` (usable as a crossfilter live mask) are left-aligned;
  `rightMatched` is right-aligned (1 when some valid left row has the same key).
- `kind: 'inner'` additionally compacts matched left row IDs, ascending, into `output.rows`
  (`status.count` clamped, `status.overflow`, optional `status.requiredCount`; ids past `status.count` are `0xffffffff`). Other
  outputs stay left-aligned. `'left'` rejects `output.rows`.

All results are integer or fixed-point, independent of thread order, and bitwise reproducible.
Input contents (keys, masks, values) are per-frame and need no rebuild; view lengths, key format,
`kind`, `sumScale` and the set of outputs are topology.

#### How it computes

{/* lens:GPUKeyJoin */}

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
scheduling, and bitwise reproducible. A 64-bit sum that would wrap raises an `overflow` word
(`readPairOverflow`) instead of wrapping silently, and pair counts saturate at `0xffffffff`.
When `maximumDistance` covers the extent the pass is a plain all-pairs loop. A row is included when
its mask is nonzero, its value (if any) finite and its position finite and inside the per-frame
`bounds`. Parameters (`[minX, minY, maxX, maxY, maximumDistance, ...]`) are per-frame, packed from a
`GPU<Class>Settings` object (`GPUVariogramSettings`, `GPUSpatialCorrelogramSettings`,
`GPURipleySettings`, `GPURipleyDistanceFunctionsSettings`, `GPUPointPatternIndicesSettings`) by
`getGPU<Class>ParameterValues(settings)`; bin counts and the set of outputs are compile-time. Every
caller-owned output view goes in one `output` prop (`GPUVariogramOutput`,
`GPUSpatialCorrelogramOutput`, `GPURipleyOutput`, `GPURipleyEnvelopeOutput`,
`GPURipleyDistanceFunctionsOutput`, `GPUPointPatternIndicesOutput`), including the `overflow`
status word.

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
graph.add(
  new GPUVariogram({
    positions, values, parameters, gridSize: [64, 64], lagCount: 24, directionCount: 4,
    output: {semivariances, pairCounts, meanDistances}
  })
);
parameters.write(getGPUVariogramParameterValues({bounds, maximumDistance: 500, azimuthOffset: 0}));
// After readback: CPU weighted least squares.
const model = fitVariogramModel({distances, semivariances, pairCounts}, {model: 'spherical'});
```

- `GPUVariogram`: polygon windows filter rows only; no area or edge correction is applied.
  Matheron semivariance, pair count, mean lag distance and Cressie-Hawkins robust
  semivariance per lag and optional azimuth sector (`directionCount` equal sectors over `[0, pi)`
  from a per-frame `azimuthOffset`; a pair of coincident points has no direction and is counted
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
  `overflow` output is 1 when a sum, a clamped term (`2^51 u`) or a pair count (saturated at
  `0xffffffff`; previously the count wrapped modulo 2^32 and became the `N_h` weight of the fit)
  left its range. The Cressie-Hawkins channel scales with `sqrt(max - min)`. The pass uses 4
  channels per bin (5 with `robustSemivariances`), so `lagCount * directionCount` is at most 512
  (409 with the robust channel). Model fitting is CPU post-processing: `fitVariogramModel` runs on
  the CPU over the read-back bins, fits spherical, exponential or Gaussian models
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
  channel. The optional `overflow` output is 1 when a product sum, a clamped product or a band pair
  count (saturated at `0xffffffff`) left its range.
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
  oracle written from the spatstat definitions, not against spatstat itself, and K and L (`GPURipley`) have
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
- `GPUPointPatternIndices`: exact nearest neighbour per row (ring search, ties to the smallest row),
  Clark-Evans `[n, observed, expected, R, standardError, z, pValue]`, and quadrat counts with
  `[m, mean, variance, varianceToMeanRatio, chiSquare, degreesOfFreedom, pValue]`. The per-frame
  `clarkEvansCorrection` defaults to `'donnelly'` (Donnelly 1978 edge correction for the rectangular
  window: `expected = 0.5 sqrt(A / n) + (0.0514 + 0.041 / sqrt(n)) P / n`, variance
  `0.0703 A / n^2 + 0.037 P sqrt(A / n^5)`, `P` the perimeter; spatstat uses 0.0412 for the second
  constant). The plain infinite-plane index (`'none'`, `0.5 sqrt(A / n)`, `0.26136 sqrt(A) / n`,
  pointpats / ArcGIS) overstates clustering in small or elongated windows and is no longer the
  default. The per-frame `alternative` (default `'two-sided'`; `'greater'` tests for regularity in
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
  on `quadratSamples`; increase it for narrow boundary features. Counts and fixed-order sums are
  deterministic.

Measured on an Apple M3 Pro (headless Chromium), `GPUVariogram` with 32 lags including the cell
sort: 100k points with 54M pairs in range in 10 ms; 20k points all-pairs (200M pairs) in 12 ms.

### `GPUPointPatternIndices`

Tests whether points cluster or spread out, from nearest-neighbour distances and a quadrat chi-square over a fixed window.

::lens{scenario=gpu-dataframe.point-pattern-indices rung=result caption="Mean nearest-neighbour distance is 122 m against 497 m expected: ratio 0.25 for 1,500 observations, strongly clustered."}

{/* parameters: GPUPointPatternIndices generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUPointPatternIndices` parameters

- Packer: `getGPUPointPatternIndicesParameterValues(settings)`, writes `GPU_POINT_PATTERN_INDICES_PARAMETER_LENGTH` (8) `float32` values.

Settings of `getGPUPointPatternIndicesParameterValues`:

| Field | Type | Description |
|---|---|---|
| `bounds` | `GPUPairStatisticsBounds` | Inclusive `[minX, minY, maxX, maxY]` window. Rows outside it, or with a non-finite coordinate, are excluded. Its area is the study area of the Clark-Evans index and the quadrats tile it. Use 'study-window' for a prepared polygon; writes NaN bounds until GPU window resolution. |
| `maximumDistance` | `number` | Size hint of the neighbor-search cells (cells are at least this wide). Results never depend on it, only speed: a value near the expected nearest-neighbor distance is fastest. |
| `clarkEvansCorrection` (optional) | `GPUClarkEvansCorrection` | Clark-Evans edge correction. Defaults to `'donnelly'` (Donnelly 1978: expected mean distance `0.5 sqrt(A / n) + (0.0514 + 0.041 / sqrt(n)) P / n`, variance `0.0703 A / n^2 + 0.037 P sqrt(A / n^5)`, `P` the window perimeter), because the plain expectation is biased toward apparent clustering for windows with a large perimeter. `'none'` gives the classic `0.5 sqrt(A / n)` with standard error `0.26136 sqrt(A) / n` (pointpats, ArcGIS). Polygon windows require explicit 'guard' or 'none'. 'guard' uses all included neighbors but averages only the interior focus points. |
| `guardDistance` (optional) | `number` | Positive finite guard width (slot 7); required for 'guard'. Recommend >= 2 / sqrt(n / A). |
| `alternative` (optional) | `GPUInferenceAlternative` | Tail of the Clark-Evans z-test and the quadrat chi-square test. Defaults to `'two-sided'`. `'greater'` tests for a statistic above its complete-spatial-randomness expectation (Clark-Evans: regular; quadrat chi-square: clustered), `'lesser'` for one below it. The permutation-only tails `'folded'` and `'directed'` throw. |

{/* /parameters */}

#### How it computes

{/* lens:GPUPointPatternIndices */}

::lens{scenario=gpu-dataframe.point-pattern-indices rung=datum bind=page datum=0 caption="The single Clark-Evans row: 1,500 points, observed 122.4 m, expected 497.3 m (Donnelly), ratio 0.246."}
::lens{scenario=gpu-dataframe.point-pattern-indices rung=formula bind=page datum=0}
::lens{scenario=gpu-dataframe.point-pattern-indices rung=kernel caption="Look at the Donnelly edge term added to 0.5 sqrt(A / n), and the guard and none alternatives beside it."}
::lens{scenario=gpu-dataframe.point-pattern-indices rung=graph caption="A cell sort, nearest-neighbour and guard reductions, then Clark-Evans and quadrat statistics nodes."}

### `GPURipley`

Estimates Ripley's K and L over radii from planar points with an edge correction, in a rectangular window.

::lens{scenario=gpu-dataframe.ripley rung=result caption="L - r is 2,558 m at 200 m and 3,699 m at 2 km: clustering at every radius."}

{/* parameters: GPURipley generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPURipley` parameters

- Packer: `getGPURipleyParameterValues(settings)`, writes `GPU_RIPLEY_PARAMETER_LENGTH` (8) `float32` values.

Settings of `getGPURipleyParameterValues`:

| Field | Type | Description |
|---|---|---|
| `bounds` | `GPUPairStatisticsBounds` | Inclusive `[minX, minY, maxX, maxY]` observation window. Rows outside it, or with a non-finite coordinate, are excluded. Its area is the study area of the estimators. Use `'study-window'` with a compile-time polygon window; it writes NaN bounds slots. |
| `maximumDistance` | `number` | Largest radius: radius `b` is `maximumDistance * (b + 1) / radiusCount`. Keep it well below the shorter window side (spatstat's default is a quarter): K is unbiased at any radius, but its variance grows and the isotropic weight cap biases it low once a circle is mostly outside. |
| `edgeCorrection` (optional) | `GPURipleyEdgeCorrection` | Edge-correction mode. Defaults to `'isotropic'`. |
| `seed` (optional) | `number` | Monte Carlo seed of `GPURipleyEnvelope`, a uint32. `GPURipley` ignores it. Defaults to 0. The same seed reproduces the same simulated patterns exactly. |

{/* /parameters */}

#### How it computes

{/* lens:GPURipley */}

::lens{scenario=gpu-dataframe.ripley rung=datum bind=page datum=4 caption="Radius 1,000 m: K is 55.7 million m2, so L is 4,210 m and L - r is 3,210 m."}
::lens{scenario=gpu-dataframe.ripley rung=formula bind=page datum=4}
::lens{scenario=gpu-dataframe.ripley rung=kernel caption="Look at the isotropic weight 1 / max(circle fraction, 1/100): pairs near the window edge count more."}
::lens{scenario=gpu-dataframe.ripley rung=graph caption="A cell sort, one pair histogram, then finish and derive nodes for K, L and L - r."}

### `GPURipleyDistanceFunctions`

Estimates the nearest-neighbour G, empty-space F and J functions over radii from planar points, border-corrected.

::lens{scenario=gpu-dataframe.ripley-distance-functions rung=result caption="G is 0.82 at 200 m while F is 0.05: events are far closer to each other than random locations are to events."}

{/* parameters: GPURipleyDistanceFunctions generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPURipleyDistanceFunctions` parameters

- Packer: `getGPURipleyDistanceFunctionsParameterValues(settings)`, writes `GPU_RIPLEY_DISTANCE_FUNCTIONS_PARAMETER_LENGTH` (8) `float32` values.

Settings of `getGPURipleyDistanceFunctionsParameterValues`:

| Field | Type | Description |
|---|---|---|
| `bounds` | `GPUPairStatisticsBounds` | Inclusive `[minX, minY, maxX, maxY]` observation window. Events outside it, or with a non-finite coordinate, are excluded. Use 'study-window' for a prepared polygon; its bbox supplies the reference lattice. Only reference centres inside the window count. Rectangle mode with these NaN bounds is undefined. |
| `maximumDistance` | `number` | Largest radius: radius `b` is `maximumDistance * (b + 1) / radiusCount`. |
| `edgeCorrection` (optional) | `GPURipleyDistanceEdgeCorrection` | `'border'` (default) divides by the points farther from the window edge than `r`; `'kaplan-meier'` and `'hanisch'` use every point (see `GPURipleyDistanceFunctions`). |

{/* /parameters */}

#### How it computes

{/* lens:GPURipleyDistanceFunctions */}

::lens{scenario=gpu-dataframe.ripley-distance-functions rung=datum bind=page datum=0 caption="Radius 200 m: G 0.818, F 0.050, so J = (1 - G) / (1 - F) is 0.19."}
::lens{scenario=gpu-dataframe.ripley-distance-functions rung=formula bind=page datum=0}
::lens{scenario=gpu-dataframe.ripley-distance-functions rung=kernel caption="Look at the border test: only events at least r from the window edge enter the G denominator."}
::lens{scenario=gpu-dataframe.ripley-distance-functions rung=graph caption="A cell sort, separate G and F accumulations (event pairs and reference lattice), then a finish node."}

### `GPURipleyEnvelope`

Simulates complete spatial randomness with a fixed seed and returns pointwise and global envelopes of L - r with a Monte Carlo p-value.

::lens{scenario=gpu-dataframe.ripley-envelope rung=result caption="Observed L - r reaches 3,699 m at 2 km; 39 seeded simulations never leave plus or minus 41 m."}

{/* parameters: GPURipleyEnvelope generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPURipleyEnvelope` parameters

- Packer: `getGPURipleyParameterValues(settings)`, writes `GPU_RIPLEY_PARAMETER_LENGTH` (8) `float32` values.

Settings of `getGPURipleyParameterValues`:

| Field | Type | Description |
|---|---|---|
| `bounds` | `GPUPairStatisticsBounds` | Inclusive `[minX, minY, maxX, maxY]` observation window. Rows outside it, or with a non-finite coordinate, are excluded. Its area is the study area of the estimators. Use `'study-window'` with a compile-time polygon window; it writes NaN bounds slots. |
| `maximumDistance` | `number` | Largest radius: radius `b` is `maximumDistance * (b + 1) / radiusCount`. Keep it well below the shorter window side (spatstat's default is a quarter): K is unbiased at any radius, but its variance grows and the isotropic weight cap biases it low once a circle is mostly outside. |
| `edgeCorrection` (optional) | `GPURipleyEdgeCorrection` | Edge-correction mode. Defaults to `'isotropic'`. |
| `seed` (optional) | `number` | Monte Carlo seed of `GPURipleyEnvelope`, a uint32. `GPURipley` ignores it. Defaults to 0. The same seed reproduces the same simulated patterns exactly. |

{/* /parameters */}

#### How it computes

{/* lens:GPURipleyEnvelope */}

::lens{scenario=gpu-dataframe.ripley-envelope rung=datum bind=page datum=9 caption="Radius 2,000 m: simulations span -19.7 to 30.2 m, the observed L - r is 3,698.8 m, p is 0.025."}
::lens{scenario=gpu-dataframe.ripley-envelope rung=formula bind=page datum=9}
::lens{scenario=gpu-dataframe.ripley-envelope rung=kernel caption="Look at the lower and upper reductions over finite simulations and the rank-based p-value."}
::lens{scenario=gpu-dataframe.ripley-envelope rung=graph caption="39 seeded simulations, each with its own cell keys and sort, feeding the envelope reduction."}

### `GPUSpatialCorrelogram`

Computes Moran's I per cumulative distance band for a station value, with z-scores and p-values under randomization.

::lens{scenario=gpu-dataframe.spatial-correlogram rung=result caption="Moran's I is 0.97 in the first 25 km band and 0.82 by 200 km: rainfall is strongly autocorrelated."}

{/* parameters: GPUSpatialCorrelogram generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUSpatialCorrelogram` parameters

- Packer: `getGPUSpatialCorrelogramParameterValues(settings)`, writes `GPU_SPATIAL_CORRELOGRAM_PARAMETER_LENGTH` (8) `float32` values.

Settings of `getGPUSpatialCorrelogramParameterValues`:

| Field | Type | Description |
|---|---|---|
| `bounds` | `GPUPairStatisticsBounds` | Inclusive `[minX, minY, maxX, maxY]` extent. Rows outside it, or with a non-finite coordinate or value, are excluded. The extent also sizes the neighbor-search lattice. Use `'study-window'` with a compile-time polygon window; it writes NaN bounds slots. |
| `maximumDistance` | `number` | Upper distance of the last band. Band `b` ends at `maximumDistance * (b + 1) / bandCount`. |
| `varianceAssumption` (optional) | `GPUSpatialCorrelogramVarianceAssumption` | Variance assumption of the z-scores and p-values. Defaults to `'randomization'`. |

{/* /parameters */}

#### How it computes

{/* lens:GPUSpatialCorrelogram */}

::lens{scenario=gpu-dataframe.spatial-correlogram rung=datum bind=page datum=0 caption="First band (up to 25 km): I is 0.971 against E[I] = -0.0008, z is 40.9, from 1,752 pairs."}
::lens{scenario=gpu-dataframe.spatial-correlogram rung=formula bind=page datum=0}
::lens{scenario=gpu-dataframe.spatial-correlogram rung=kernel caption="Look at the band test on pair distance and the product (x_i - mean)(x_j - mean) summed per band."}
::lens{scenario=gpu-dataframe.spatial-correlogram rung=graph caption="Moments, a cell sort, the band pair histogram and a fourth-moment sum, then finish and publish nodes."}

### `GPUVariogram`

Computes the empirical semivariogram of a station value over distance lags, with pair counts and mean pair distances.

::lens{scenario=gpu-dataframe.variogram rung=result caption="Semivariance rises from 546 at the first 25 km lag to 2,567 at 300 km: nearby stations see similar rainfall."}

{/* parameters: GPUVariogram generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUVariogram` parameters

- Packer: `getGPUVariogramParameterValues(settings)`, writes `GPU_VARIOGRAM_PARAMETER_LENGTH` (8) `float32` values.

Settings of `getGPUVariogramParameterValues`:

| Field | Type | Description |
|---|---|---|
| `bounds` | `GPUPairStatisticsBounds` | Inclusive `[minX, minY, maxX, maxY]` extent. Rows outside it, or with a non-finite coordinate or value, are excluded. The extent also sizes the neighbor-search lattice. Use `'study-window'` with a compile-time polygon window; it writes NaN bounds slots. |
| `maximumDistance` | `number` | Largest lag: pairs farther apart are ignored. Lag bins split `[0, maximumDistance]`. |
| `azimuthOffset` (optional) | `number` | Start of the first direction sector, in radians counterclockwise from +x. Sector `s` holds pairs whose undirected direction (taken modulo pi) minus this offset, again modulo pi, lies in `[s * pi / directionCount, (s + 1) * pi / directionCount)`. Ignored when `directionCount` is 1. Defaults to 0. |

{/* /parameters */}

#### How it computes

{/* lens:GPUVariogram */}

::lens{scenario=gpu-dataframe.variogram rung=datum bind=page datum=3 caption="Lag 3 (75 to 100 km): semivariance 962.6 from 6,216 pairs at a mean separation of 87.9 km."}
::lens{scenario=gpu-dataframe.variogram rung=formula bind=page datum=3}
::lens{scenario=gpu-dataframe.variogram rung=kernel caption="Look at the lag index min(floor(d / D * H), H - 1) and the squared value difference."}
::lens{scenario=gpu-dataframe.variogram rung=graph caption="Moments and a unit-variance estimate, a cell sort, the pair-histogram pass and a finish node."}

## `GPUCompositeScore`

Scales indicator columns to [0, 1], flips higher-is-worse directions and combines them into one weighted score per row.

::lens{scenario=gpu-dataframe.composite-score rung=result caption="788 of 791 tracts are scored; scores span 0.03 in tract 657 to 0.79 in tract 603 (CPU check)."}

`GPUCompositeScore` builds a composite indicator per row from up to 16 indicator columns
(row-major `row * indicatorCount + column`), as CARTO's composite-indicator procedures and ArcGIS
index builders do: scale each column, flip "higher is worse" indicators, and combine them. Weights,
directions, the scaler and the aggregation live in a parameter buffer, so weight sliders recolor
every row without rebuilding the graph.

```ts
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
- Determinism: min and max use u32 order-key atomics; means and centered (co)variances are
  two-pass fixed-order sums over 256-row tiles, merged by one 256-lane workgroup per output
  (strided tiles, then a fixed binary tree); ranks use one stable radix
  sort per column plus a binary search per tie run. No float atomics.
- `compile-time`: row count, `indicatorCount`, `enableRank`, `enablePrincipalComponent`, the mask
  and which outputs exist. `per frame`: scaler, aggregation, weights, directions, epsilon.

{/* parameters: GPUCompositeScore generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPUCompositeScore` parameters

- Packer: `getGPUCompositeScoreParameterValues(settings)`, writes `GPU_COMPOSITE_SCORE_PARAMETER_LENGTH` `float32` values.

Settings of `getGPUCompositeScoreParameterValues`:

| Field | Type | Description |
|---|---|---|
| `scaler` (optional) | `GPUCompositeScoreScaler` | Per-indicator scaler. Defaults to `'min-max'`. |
| `aggregation` (optional) | `GPUCompositeScoreAggregation` | How scaled indicators combine into the score. Defaults to `'weighted-sum'`. |
| `weights` | `ArrayLike<number>` | One weight per indicator (at most 16). Missing trailing weights are 0. A negative weight penalizes an indicator in `'weighted-sum'`; `'weighted-geometric-mean'` refuses negative weights (a weighted geometric mean needs `w >= 0`): use `directions` to flip an indicator. |
| `directions` (optional) | `ArrayLike<number>` | One direction per indicator: a negative value means "higher is worse" and flips the scaled value (`1 - s` for min-max and rank, `-s` for z-scores). Defaults to all `+1`. |
| `epsilon` (optional) | `number` | Positive shift added before the logarithm of the geometric mean. Defaults to `1e-6`. |

{/* /parameters */}

#### How it computes

{/* lens:GPUCompositeScore */}

::lens{scenario=gpu-dataframe.composite-score rung=datum bind=page datum=603 caption="Tract 603 scores highest, 0.79: poverty 67.8, unemployment 37.1, uninsured 24.5 and 62.7 percent without a vehicle (CPU check)."}
::lens{scenario=gpu-dataframe.composite-score rung=formula bind=page datum=603}
::lens{scenario=gpu-dataframe.composite-score rung=kernel caption="Look at the min-max step (x - min) / (max - min), the direction flip and the division by the sum of absolute weights."}
::lens{scenario=gpu-dataframe.composite-score rung=graph caption="Per-column extrema and moment reductions merged in tile order, then a score node reading the parameter buffer."}

## `GPUInequality`

Parallel and reference routes support at most 2,147,483,648 rows, subject to sort and device
limits. `mode: 'reference'` preserves the serial oracle.

Parallel reductions are the default; mode reference retains the serial per-zone oracle. Lorenz knots
and output bindings retain their existing limits.

Computes weighted Gini and other inequality indices per zone from a value column, with Lorenz knots and a pooled summary.

::lens{scenario=gpu-dataframe.inequality rung=result caption="Within community areas the income Gini runs from 0.01 (area 37, two tracts) to 0.31 (area 28); pooled it is 0.33 (CPU check)."}

`GPUInequality` computes per-zone income inequality indices from a column of non-negative values: Gini, Theil T and L, Atkinson, Hoover, Palma, Lorenz curve knots, plus the between/within-zone decomposition of Theil T and pooled statistics.

Inputs: `values` (float32), `zoneIds` (uint32; `0xffffffff` or any id `>= zoneCount` skips the row), `zoneCount` (at most `getGPUInequalityMaximumZoneCount(device)`, 3,050,402 on a default device) and `lorenzKnotCount` (compile-time; `zoneCount * lorenzKnotCount` at most `getGPUInequalityMaximumLorenzKnotTotal(device)`, 2^25 on a default device), optional population `weights` (float32) and `mask` (uint32), and a `parameters` view written with `getGPUInequalityParameterValues({epsilon, palmaTopShare, palmaBottomShare})`. Changing the parameters never rebuilds or recompiles the graph. A row is included when its mask is non-zero, its zone is valid, its value is finite and non-negative and, with weights, its weight is finite and positive. With `w` the weight (1 without weights), `W = sum w`, `S = sum w x` and `mu = S / W`:

| Output | Definition |
| --- | --- |
| `gini` | `1 - sum (P_i - P_(i-1)) (L_i + L_(i-1))`, trapezoid area under the Lorenz curve of the value-sorted rows; equals `2 sum i x_(i) / (n S) - (n + 1) / n` for unit weights, `(n - 1) / n` for one holder of everything |
| `theilT` | `(1 / W) sum w r ln r`, `r = x / mu`, `0 ln 0 = 0` |
| `theilL` | `-(1 / W) sum w ln r`; NaN when the zone contains a zero (zeros are never silently dropped) |
| `atkinson` | `1 - ((1 / W) sum w r^(1 - e))^(1 / (1 - e))`, `1 - exp(mean ln r)` at `e = 1`; zeros are fine for `e < 1`, NaN for `e >= 1`; `e` in `[0, 32]` per frame |
| `hoover` | `sum w abs(x - mu) / (2 S)` |
| `palma` | `(1 - L(1 - top)) / L(bottom)`, cuts per frame (0.1 and 0.4 by default); NaN when `L(bottom) = 0` |
| `mean`, `counts` | weighted mean, included row count per zone (`counts` is `uint32`) |
| `lorenzKnots` | `L(k / (K - 1))`, row-major `zone * K + k`, `L(0) = 0`, `L(1) = 1` |
| `globalSummary` | slots `GPU_INEQUALITY_GLOBAL_SUMMARY`: `TOTAL_THEIL_T`, `BETWEEN_THEIL_T`, `WITHIN_THEIL_T`, `GINI`, `COUNT`, `MEAN`, `TOTAL_WEIGHT`, `TOTAL_INCOME` |

The Lorenz curve is linear inside each row (a row's income spreads over its population share), so a Palma cut or knot inside a row takes a fractional share of that row. Empty zones, and zones with zero total income, report NaN for every index (zero-income zones keep `mean = 0`). The Theil decomposition is `between = sum_g s_g ln(s_g / p_g)` and `within = sum_g s_g T_g` with income share `s_g` and population share `p_g`; the pooled Theil T is computed independently from all included rows, so `total = between + within` is a real check.

Algorithm and cost: rows are sorted by value, re-keyed by zone and stably sorted, so each zone
is a value-ordered segment. The default parallel route builds bounded 256-row Lorenz prefixes,
computes row-parallel terms and reduces them by segment. `mode: 'reference'` preserves the two
serial per-zone walks and their rounding order; a dominant zone serializes that route. The
parallel summation order can change the last float32 bits; sums and published statistics remain
float32. Pooled and global reductions use fixed-order tiles.

{/* parameters: GPUInequality generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPUInequality` parameters

- Packer: `getGPUInequalityParameterValues(settings?)`, writes `GPU_INEQUALITY_PARAMETER_LENGTH` (4) `float32` values.

Settings of `getGPUInequalityParameterValues`:

| Field | Type | Description |
|---|---|---|
| `epsilon` (optional) | `number` | Atkinson inequality aversion, `0` to `GPU_INEQUALITY_MAXIMUM_EPSILON`. `1` selects the geometric-mean form. Defaults to 1. |
| `palmaTopShare` (optional) | `number` | Population share of the Palma numerator, in `(0, 1]`. Defaults to 0.1. |
| `palmaBottomShare` (optional) | `number` | Population share of the Palma denominator, in `(0, 1]`. Defaults to 0.4. |

{/* /parameters */}

#### How it computes

{/* lens:GPUInequality */}

::lens{scenario=gpu-dataframe.inequality rung=datum bind=page datum=27 caption="Community area 28 has the highest Gini, 0.307, over 20 tracts with a population-weighted mean income of 85,684 dollars (CPU check)."}
::lens{scenario=gpu-dataframe.inequality rung=formula bind=page datum=27}
::lens{scenario=gpu-dataframe.inequality rung=kernel caption="Look at the trapezoid sum 1 - sum (P_i - P_(i-1)) (L_i + L_(i-1)) walked over each zone's value-sorted rows."}
::lens{scenario=gpu-dataframe.inequality rung=graph caption="Two sorts, by value and then by zone, a per-zone walk, and a finish node for the pooled summary."}

## `GPUTemporalReduction`

Reduce timestamped rows to per cell and time-bucket counts, extremes, first and last values.

::lens{scenario=gpu-dataframe.temporal-reduction rung=result caption="251 of 530 taxon-week slots are occupied; the busiest holds 36 bird observations in week 18 (CPU check)."}

Reduces timestamped rows into one record per (cell, coarse time bucket) so a scrubbing view draws one feature per occupied slot instead of one per row. Bucket `b` covers `[origin + b * width, origin + (b + 1) * width)`; the slot is `cell * bucketCount + bucket`.

```ts
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

- Outputs are dense columns of `cellCount * bucketCount` rows. Empty slots have count 0 and NaN minimums, maximums, first and last. `occupiedSlots.ids` lists occupied slots in ascending order (`cell = id / bucketCount`, `bucket = id % bucketCount`), clamped to its length with `overflow` set when more are occupied.
- `first` and `last` are the values at the earliest and latest timestamp in the slot; ties are broken by the lower row index.
- Rows with a bucket outside `[0, bucketCount)`, a skipped cell, a zero mask, or a NaN value or time are dropped, not clamped.
- Exact Int64 times: pass `uint32x2` words and write `getGPUTemporalReductionWordParameterValues(origin, width)` (integer origin, integer width in `[1, 2^64 - 1]`, so hourly buckets of nanosecond data use `3_600_000_000_000n`). Bucket assignment and the first and last time inside a bucket are exact integer arithmetic; the bucket index must still be below `bucketCount`.
- Determinism: u32 atomics only, so results do not depend on thread order. In float mode bucket edges are defined by correctly rounded products (`fround(b * width) <= fround(t - origin) < fround((b + 1) * width)`), so assignment is identical on every adapter; at most 2^20 buckets.
- Slot limit: `cellCount * bucketCount` is at most 33554432 slots with float32 times and 16777216 slots with `uint32x2` times on a default device (128 MiB bindings). Word mode keeps the in-bucket time of each slot as two 32-bit words (8 bytes per slot instead of 4), so its maximum is half the float one. Larger slot counts are refused at declaration.
- Inputs must be single packed views; chunked vectors are not supported.
- `sorted: true` (default false, compile-time) declares `timestamps` non-decreasing and non-NaN. The rows with bucket in `[0, bucketCount)` are then one contiguous range found by two binary searches (same bucket rules as above: the correctly rounded `fround(t - origin) < fround(bucketCount * width)` test in float mode, the exact quotient in word mode), and the classify, extreme and tie kernels are dispatched indirectly over just that range, with the first row read from a GPU word. A scrub costs O(range), not O(rows), and the result is bit-identical to the unsorted path on sorted input. It adds one `uint32` scratch column of `rows` entries. The order is not checked; unsorted input silently reduces the wrong rows.

{/* parameters: GPUTemporalReduction generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPUTemporalReduction` parameters

- Packer: `getGPUTemporalReductionParameterValues(origin, width)`, writes `GPU_TEMPORAL_REDUCTION_PARAMETER_LENGTH` (2) `float32` values.
  - `origin` (`number`): Start of bucket 0, in the unit of the timestamps.
  - `width` (`number`): Bucket width, in the unit of the timestamps.
- Packer: `getGPUTemporalReductionWordParameterValues(origin, width)`, writes `GPU_TEMPORAL_REDUCTION_WORD_PARAMETER_LENGTH` (4) `uint32` values.
  - `origin` (`number | bigint`): Integer start of bucket 0, any signed 64-bit value.
  - `width` (`number | bigint`): Integer bucket width in `[1, 2^64 - 1]`. A `number` must be a safe integer to be exact; pass a `bigint` beyond 2^53.

{/* /parameters */}

#### How it computes

{/* lens:GPUTemporalReduction */}

::lens{scenario=gpu-dataframe.temporal-reduction rung=datum bind=page datum=133 caption="Slot 133 is insects in week 27: 32 observations, research grade mixed (minimum 0, maximum 1; CPU check)."}
::lens{scenario=gpu-dataframe.temporal-reduction rung=formula bind=page datum=133}
::lens{scenario=gpu-dataframe.temporal-reduction rung=kernel caption="Look at the bucket test: fround(b * width) <= fround(t - origin) < fround((b + 1) * width)."}
::lens{scenario=gpu-dataframe.temporal-reduction rung=graph caption="Init, classify, extreme, row-tie and finish nodes, then compaction and publish of occupied slots."}

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

{/* parameters: GPUCalendarBuckets generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPUCalendarBuckets` parameters

- Packer: `getGPUCalendarBucketsParameterValues(utcOffsetMinutes, firstDayOfWeek?, target?)`, writes `GPU_CALENDAR_BUCKETS_PARAMETER_LENGTH` (2) values.
  - `utcOffsetMinutes` (`number`): Fixed offset east of UTC in minutes, an integer in `[-1440, 1440]`.
  - `firstDayOfWeek` (`number`): First day of the week for the `weekday` output and the hour by weekday matrix: 0 = Monday (ISO, default) to 6 = Sunday. ISO week numbers always use Monday.
  - `target` (`Int32Array`): Optional destination of at least 2 elements.

{/* /parameters */}

#### How it computes

{/* lens:GPUCalendarBuckets */}

::lens{scenario=gpu-dataframe.calendar-buckets rung=datum bind=page datum=0 caption="Observation 0 is Sunday 2023-01-01 08:05: weekday 6, hour 8, month 1."}
::lens{scenario=gpu-dataframe.calendar-buckets rung=formula bind=page datum=0}
::lens{scenario=gpu-dataframe.calendar-buckets rung=kernel caption="Look at the floor division by 86,400,000 and the Hinnant civil-date arithmetic after it."}
::lens{scenario=gpu-dataframe.calendar-buckets rung=graph caption="Decode kernels split over the storage-binding limit, plus a cleared matrix and its count kernel."}

## `GPUPeriodIndex`

Turn calendar columns or Int64 word times into a `uint32` period per row: whole calendar steps from an origin, or a fixed width.

The period index is the adapter between `GPUCalendarBuckets` (or raw Int64 words) and a (cell, period) cube. It is the code that `addSpaceTimeHotSpotsRecipe` runs between its calendar decode and its cube, extracted so that other space-time pipelines share it and the recipe's slices are exactly these periods. It reads the temporal context of the times, never converts units, and writes one `uint32` per row with no readback.

```ts
// Calendar axis: 52 weeks from Monday 2024-01-01, from the calendar columns of the same times.
graph.add(
  new GPUCalendarBuckets({temporalContext, timestamps, parameters, output: {year, month, dayOfMonth}})
);
graph.add(
  new GPUPeriodIndex({
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
    temporalContext: nanosecondContext,
    timestamps,
    periods: {kind: 'fixed', width: 3_600_000_000_000n, origin: 1_704_067_200_000_000_000n, count: 24},
    output: {periods}
  })
);
```

- `temporalContext`: resolved with `{encodings: ['int64-words'], unixEpoch: true}`, so Arrow uploads pass their own context. An unsupported encoding or an epoch other than the Unix epoch throws at construction. The `unit` is the unit of a fixed `width` and `origin`; a calendar axis only reads the decoded columns, whose unit `GPUCalendarBuckets` already handled.
- Calendar axis: `step` is `{calendarUnit, interval?}` with `calendarUnit` one of `'year'`, `'quarter'`, `'month'`, `'week'`, `'day'` or `'hour'` and a positive integer `interval` (default 1). Period `floor((stepIndex(time) - stepIndex(origin)) / stepsPerPeriod)` holds a time, where `stepIndex` counts whole steps (`year`, `year * 4 + quarter - 1`, `year * 12 + month - 1`, days since 1970 for `'day'` and `'week'`, hours since 1970 for `'hour'`) and a week is 7 days anchored to the origin day. `origin` is `{year, month?, day?, hour?}` in the local calendar of `columns`, with the unset fields at the start of the step. `columns.year` is always read; `columns.month` for every step but `'year'`; `columns.dayOfMonth` for `'week'`, `'day'` and `'hour'`; `columns.hour` for `'hour'`. Days are numbered with Hinnant's `days_from_civil` in `i32` arithmetic, so leap days, quarter starts and years before 1970 are exact. The CPU twins `getGPUPeriodIndexCalendarPeriod`, `getGPUPeriodIndexStepIndex` and `getGPUPeriodIndexDaysFromCivil` return the same numbers.
- Fixed axis: period `floor((time - origin) / width)` with `width` a positive integer below 2^63 (a `number` must be a safe integer; pass a `bigint` beyond 2^53) and `origin` signed Int64 ticks after the Unix epoch. The difference and the division run on the Int64 words, so they are exact for every time, including times before 1970 and beyond 2^53 ticks, and for widths past 32 bits.
- Output `periods` holds one `uint32` per row: the period, `GPU_PERIOD_INDEX_OUT_OF_RANGE` (`0xfffffffe`) for a valid time before the origin or past the last period, or `GPU_PERIOD_INDEX_INVALID` (`0xffffffff`) for a masked row or a time the calendar decode rejected. Rows are skipped, never clamped.
- Status: `output.status.invalidCount` (optional) is the number of skipped rows, those with `GPU_PERIOD_INDEX_OUT_OF_RANGE`. Masked and undecodable rows are not counted. It is cleared on every encoding and counted with an integer `atomicAdd`.
- Limits: `periods.count` is an integer in `[1, GPU_PERIOD_INDEX_MAXIMUM_COUNT]` (4294967293, the two sentinels are not periods). `output.periods` and the Int64 `timestamps` must each fit one storage binding: 33,554,432 rows and 16,777,216 rows on a default device; larger inputs are refused at declaration.
- Inputs must be single packed views; an optional `mask` (`uint32`, zero writes the invalid sentinel) must have one row per input row.

### `GPUPeriodIndex` parameters

- Per frame (no recompile): the contents of `columns`, `timestamps` and `mask`.
- Compile-time (needs a new graph): the row count, `periods` (`kind`, `step`, `origin`, `width`, `count`), which calendar columns are present, and whether `mask` and `status.invalidCount` are present. The axis is compile-time because it fixes the size of the cube the periods index.
- Packer: none.

## `GPUCubeLayoutTranspose`

Move a dense (cell, period) cube between the cell-major and period-major layouts.

The space-time hot spot cube is cell-major (`cell * periodCount + period`); Markov chains, spatial panels and per-period maps are period-major (`period * cellCount + cell`). The transpose converts one into the other on the GPU for `uint32` counts or `float32` values.

```ts
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
const relative = new GPUTimeWordsRelative({
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
- Limit: the `uint32x2` timestamps must fit one storage binding, 16,777,216 rows on a default device; larger inputs are refused at declaration.

### `GPUTimeWordsRelative` parameters

- Per frame (no recompile): the contents of `timestamps` and `output`.
- Compile-time (needs a new graph): the row count, `origin` and `temporalContext` (the origin words are shader overrides, so graphs that differ only in origin share a module).
- Packer: none.

## `GPUValueCube`

Dense (cell, period) cube of count, sum and mean of a value column.

`GPUTemporalReduction` and the hot spot recipe bin counts, minima, maxima, first and last values but no sum or mean. `GPUValueCube` adds them in the cell-major layout `cell * periodCount + period` that `GPUEmergingHotSpots` reads, so `GPUTimeWindowFilter` (the `mask`), `GPUPeriodIndex` (the `periods`) and the cube chain to Gi\*, trend, Markov chains and panels with no readback.

```ts
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

- Sums are deterministic and independent of thread order. Each value is quantized to the shared 96-bit fixed-point accumulator (`analysis-core/kernels/fixed-point-accumulator.ts`) and added with integer atomics; the exact integer is rounded once to float32. There are no float atomics. A float32 running sum that cancels `1e8 + 1 - 1e8` loses the `1`; the cube returns it exactly.
- The unit is a power of two set by the largest magnitude among the rows that are binned and by the bit length of the row count. A value keeps all 24 significand bits while it is at least `max * 2^(h - 71)` (`h` the bit length of the row count; about 1e15 of dynamic range at a million rows); smaller values round to the nearest unit, so a sum of `n` terms is off by at most `n / 2` units before the final rounding. Because the unit follows the binned rows, the rounding of a bin can change when another set of rows is binned, for example another time window.
- `means` is the rounded sum divided by the count in float32. WGSL division is not correctly rounded (up to 2.5 ULP), so a mean is within a few units in the last place of the true mean; an empty bin is NaN, which `GPUEmergingHotSpots` treats as missing.
- Not binned: masked rows (silent), rows whose period is `0xffffffff` (`GPU_PERIOD_INDEX_INVALID`, excluded upstream, silent), and rows with a cell or period out of range (including `GPU_PERIOD_INDEX_OUT_OF_RANGE`) or a non-finite value. The last group is counted in `status.invalidCount`.
- Limits: `cellCount`, `periodCount`, `cellCount * periodCount` and the row count at most 4294967294; the fixed-point words need 12 bytes per bin in one storage binding, so `cellCount * periodCount` is at most `getGPUValueCubeMaximumBinCount(device)`, 11,184,810 on a default device. Larger cubes are refused at declaration with the binding limit in the message. Inputs and outputs are single packed views.

### `GPUValueCube` parameters

- Per frame (no recompile): the contents of `cells`, `periods`, `values` and `mask`.
- Compile-time (needs a new graph): the row count, `cellCount`, `periodCount`, and which outputs and `mask` are present.
- Packer: none.

## `GPUCellPeriodKeys`

Packs a 64-bit H3, S2 or Quadbin cell key and a period into one group key for `GPUGroupStatistics`.

A dense cube needs a cell index below `cellCount`; a DGGS has billions of possible cells. `GPUGroupStatistics` already groups by `uint32x2` keys and computes count, sum, mean, minimum, maximum, variance and percentiles in a capacity-bounded table sorted by key, so a sparse DGGS by period cube is one packing kernel away.

```ts
graph.add(
  new GPUCellPeriodKeys({
    cells: cellKeys, // GraphDataView<'uint32x2'>, little-endian (low, high) words
    periods: periodIndexOutput, // GraphDataView<'uint32'> from GPUPeriodIndex
    periodCount: 365,
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

Which route builds a DGGS (cell, period) cube without readback? Two compose from the existing pieces: (a) this packing kernel and `GPUGroupStatistics` on the composite key, or (b) `GPUCellAggregation` once to get the distinct cells, `GPUKeyJoin` to map every row to its cell row, then `row * periodCount + period` as a key into `GPUValueCube` or `GPUGroupStatistics` in dense mode. Use (a) for DGGS cubes: it needs one sort and no cell-row lookup, its memory follows the number of occupied (cell, period) groups instead of `cellCount * periodCount`, and cells never need a table. Use (b), or `GPUValueCube` with a cell index you already have (a lattice, or a weights table), only when the consumer needs the dense cell-major cube, as `GPUEmergingHotSpots` and the layout transpose do.

- Packing: with `b = ceil(log2(periodCount))` period bits, the key is `(cellKey & ~(2^b - 1)) | period`. The period replaces the low `b` bits of the cell key, so the packed key sorts by cell and then period. A packed key is not a valid cell id: unpack the group table keys with `unpackGPUCellPeriodKey(key, periodCount, fill)`.
- What happens to the key bits: the cell key stays 64 bits, nothing is truncated to 32. The `b` replaced bits must hold the `fill` pattern. At the resolution of the cells they do: unused H3 digits are 7 (`3 * (15 - resolution)` low bits, all ones), the Quadbin footer is all ones (`52 - 2 * zoom` bits), an S2 id has a lone sentinel one above `60 - 2 * level` zero bits. Within that many free bits the packing is lossless and injective. A cell with any other bits where the period goes (a finer cell) is not packed: its key becomes the reserved all-ones key, which `GPUGroupStatistics` skips, and it is counted in `status.invalidCount`; cells are never merged silently. A finer cell whose extra bits happen to equal the fill pattern is packed and unpacks to itself. `getGPUCellKeyFreeBits('h3' | 'quadbin' | 's2', resolution)` gives the free bits. 365 daily periods need 9 bits: H3 resolution 12 or coarser, Quadbin zoom 21 or coarser, S2 level 25 or coarser. Hourly periods over a year (8,760) need 14 bits: H3 resolution 10, Quadbin zoom 19, S2 level 23.
- Rows are skipped silently when masked, when the cell is the all-ones "no cell" key, or when the period is `0xffffffff`. A period at or above `periodCount` is skipped and counted.
- Limits: `periodCount` at most 4294967293 (32 period bits); `cells` fit one storage binding, 16,777,216 rows on a default device. Downstream, `GPUGroupStatistics` keeps at most `getGPUGroupStatisticsMaximumGroupCount` (8,388,608 on a default device) distinct (cell, period) groups and keeps the smallest keys on overflow, and its `sums` words are fixed point with a resolution of `1 / sumScale` (65536 by default); its `sumValues` and `means` use a unit that follows the data, as `GPUValueCube` does.

### `GPUCellPeriodKeys` parameters

- Per frame (no recompile): the contents of `cells`, `periods` and `mask`.
- Compile-time (needs a new graph): the row count, `periodCount`, `fill`, and whether `mask` and `status.invalidCount` are present.
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

Props: `positions`, `values`, optional `secondValues` (signed cross-semivariance, default the Matheron semivariogram), `times`, optional `mask`, `parameters` written with `getGPUSpaceTimeVariogramParameterValues({bounds, maximumDistance, maximumTimeLag})`, `gridSize`, `lagCount`, `temporalLagCount` and `output: {semivariances, pairCounts, status}`. Status per bin is 0 defined, 1 empty or invalid parameters, 2 nonfinite arithmetic, 3 accumulation overflow.

`times` must be relative to a CPU float64 origin. Raw Unix epoch seconds in float32 are spaced 128 s apart (milliseconds about 2 minutes), which merges every short lag into one. Subtract the origin with `getGPUSpaceTimeRelativeTimes` from `@luma-contributors/analysis/gpu-spatial-analysis` before upload; it also returns the remaining float32 resolution. Times closer than that resolution share the zero-lag class. See [the estimator contracts](./gpu-spatial-analysis/statistical-expansion.md) for bin edges, complete-case masks and the validation status: the results match an in-repo brute-force reference, and a gstat `variogramST` fixture is still pending.

Breaking change: the parameter buffer is now eight values (bounds, then the two maximum lags), `gridSize` is required, the output has the extra zero-lag classes, and the limit is 409 bins instead of 16,384 rows by 128 bins.

{/* parameters: GPUSpaceTimeVariogram generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPUSpaceTimeVariogram` parameters

- Packer: `getGPUSpaceTimeVariogramParameterValues(settings)`, writes `GPU_SPACE_TIME_VARIOGRAM_PARAMETER_LENGTH` (8) `float32` values.

Settings of `getGPUSpaceTimeVariogramParameterValues`:

| Field | Type | Description |
|---|---|---|
| `bounds` | `Exclude<GPUPairStatisticsBounds, 'study-window'>` | Inclusive `[minX, minY, maxX, maxY]` extent; rows outside it are excluded. |
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

{/* lens:GPUVariogramModelFit */}

::lens{scenario=gpu-dataframe.variogram-model-fit rung=result caption="Resident spherical, exponential and Gaussian fits to 20 explicitly synthetic empirical lag bins."}

::lens{scenario=gpu-dataframe.variogram-model-fit rung=graph caption="The bounded GPU range search produces fitted parameters, residual/AIC diagnostics and convergence status."}

This executable example uploads bins generated from an exponential model with nugget `0.2`, partial sill `1.5` and effective range `2.2`, then fits all three candidates on the GPU with pair weights. Model rows are spherical, exponential and Gaussian. Captured parameter rows contain nugget, partial sill, effective range and weighted residual; diagnostic rows contain usable bins, unweighted residual, residual AIC and final log-range bracket. Status 0 is an interior converged fit, 1 insufficient bins, 2 numerical failure and 3 a boundary/budget result. See [the fitting contract](./gpu-spatial-analysis/interpolation-expansion.md) for identifiability, weighting and performance limitations.

The fitter also supports `matern-3/2` and `matern-5/2`. With `a = factor * h / range`, their normalized semivariogram shapes are `1 - (1 + a) exp(-a)` and `1 - (1 + a + a² / 3) exp(-a)`. The factors are `4.749031386012701` and `5.924462618951797`, respectively, giving correlation `exp(-3)` at the effective range. These definitions also apply to coregionalization below. CPU evaluation uses a positive exponential-remainder series through order 18 for `a < 0.5`; GPU evaluation uses a bounded Taylor series for `a < 0.1`. These evaluations preserve the quadratic leading term at tiny lag/range ratios. CPU exponential and Gaussian shapes use `expm1` for the same cancellation concern. The mathematical model and fitting objective are unchanged by these numerical evaluations.

## `GPUCoregionalizationFit`

{/* lens:GPUCoregionalizationFit */}

::lens{scenario=gpu-dataframe.coregionalization-fit rung=result caption="Fit a PSD two-variable structure matrix with a negative cross coefficient; exclude one zero-pair bin."}

::lens{scenario=gpu-dataframe.coregionalization-fit rung=graph caption="Gated residual fits and PSD projections publish matrices, square loading factors and named convergence status."}

Fits a linear model of coregionalization from complete direct/cross semivariogram matrices on common lag supports. For fixed basic semivariograms `g_s(h) = nugget_s + sill_s * shape_s(h / range_s)`, it minimizes `Σ_h N_h ||Γ_h - Σ_s B_s g_s(h)||_F² / Σ_h N_h` over positive semidefinite matrices `B_s`. Off-diagonal residuals count twice. Each sweep updates structures sequentially by fitting the residual from the other structures and clipping negative eigenvalues to zero, starting from zero matrices ([Goulard–Voltz, 1992](https://doi.org/10.1007/BF00893750)). It fits coefficients for supplied models and ranges; dependent structures can have nonunique coefficients.

`distances: float32` and `pairCounts: uint32` have one row per lag. `semivariances: float32` holds one bin-major upper triangle `(0,0), (0,1), …, (1,1), …` per lag. Assemble direct and cross variograms using the same supports and complete-case sample pairs. Negative cross semivariances are retained. A nonpositive/nonfinite lag, zero pair count or any nonfinite matrix entry excludes the whole bin; missing cross entries are not imputed. No coordinates are read, and lags and ranges must share units.

`getGPUCoregionalizationFitParameterValues(structures, tolerance = 1e-5)` supplies the per-frame buffer `[tolerance, modelCode, nugget, sill, range, …]`. Structures support spherical, exponential, Gaussian and Matérn 3/2 and 5/2, with the effective-range convention above. Use `sill: 1, nugget: 0` for a unit-sill basis, or `sill: 0, nugget: 1` for a pure nugget. Compile-time props are `variableCount` (2–4), `structureCount` (1–4), `maximumIterations` (0–1024, default 128) and view shapes. All view contents, including tolerance and model parameters, can change per frame. This reference-tier GPU fit accepts at most 256 lag matrices and estimates `2 * maximumIterations + 3` command nodes.

Outputs `matrices` and `loadings` are `float32`, each with a result prefix of `structureCount * variableCount²` elements in structure-major, row-major order. `B_s = L_s L_sᵀ`; loading factors are square spectral factors and need not be triangular. Compare matrices or reconstructed factors across implementations because eigenvector signs and repeated-eigenvalue bases are arbitrary. `diagnostics: float32x4` holds `[weighted objective, relative coefficient change, usable bins, total pair weight]`. Relative change is the largest coefficient change divided by the largest absolute old/new coefficient over the sweep, or zero when both are zero. Views must be packed; outputs use distinct buffers disjoint from inputs. Trailing output capacity is left untouched.

Required uint32 scalar status ports are `validity`, `iterationCount`, `converged`, `iterationLimitReached` and `invalidCount` (excluded lag matrices). A budget exit publishes the current PSD fit with validity 1, converged 0 and iterationLimitReached 1; zero budget publishes zero matrices and factors. No usable bins, invalid parameters, a zero-energy structure or nonfinite arithmetic gives validity 0, convergence/limit flags 0 and NaN coefficients, objective and change. Usable-bin and total-weight diagnostics remain available.

Prefer `CPUCoregionalizationFitOperation` for small fits whose bins are already on the CPU. It uses the same ports as `GPUCoregionalizationFitOperation`, computes in float64 and rounds once at output, and drops the GPU bin and sweep caps. Both operations take the same graph properties; the GPU form requires `GPU_CONTRIBUTOR_OPERATION_LOWERING`. Float32 accumulation can change convergence counts and loading bases. The pure `computeCoregionalizationFitOnCPU` accepts typed arrays; `fitLinearModelOfCoregionalization` additionally returns `coregionalizationModels` ready for `GPULocalKriging`:

```typescript
import {fitLinearModelOfCoregionalization} from '@luma-contributors/analysis/dataframe-analysis';

const fit = fitLinearModelOfCoregionalization(
  {distances, semivariances, pairCounts},
  {variableCount: 2, structures: [{model: 'spherical', range: 2, sill: 1}]}
);
if (!fit.validity[0]) throw new Error('Coregionalization fit is invalid');
// Inspect fit.converged[0] and fit.iterationLimitReached[0] before using a budget result.
const krigingModel = {
  mode: 'co-kriging',
  variableCount: 2,
  covariance: fit.coregionalizationModels[0].covariance,
  coregionalizationModels: fit.coregionalizationModels
};
```

Invalid fits return an empty `coregionalizationModels` list. Local co-kriging requires `covariance` when no parameter buffer is supplied; use the first fitted structure after checking validity. GPU factors use the same layout: read back each square structure slice to supply its `loadings`.

The lens example uses three hand-derived lag matrices from `B = [[2, -1], [-1, 3]]` and a unit-sill spherical structure of range 2. A fourth, unrelated matrix has zero pairs and is excluded. When run, the scenario uses an eight-sweep budget and records GPU matrices, factors, residual diagnostics and every named status port.

{/* parameters: GPUCoregionalizationFit generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPUCoregionalizationFit` parameters

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
