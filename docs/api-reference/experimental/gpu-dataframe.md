import {ExperimentalDocsTabs} from '@site/src/components/docs/experimental-docs-tabs';
import {GPUExampleCard} from '@site/src/components/docs/gpu-example-card';
import {GPUDataAnalysisExample} from '@site/src/examples';

# GPU Dataframe

<ExperimentalDocsTabs active="gpu-dataframe" />

## Overview

`@luma-contributors/graph/gpu-dataframe` compiles dataframe-style expressions, filters, aggregations, ordering,
and joins into a caller-owned `GPUCommandGraph`. It consumes Arrow-independent `GPUTable` and
`GPUVector` storage while preserving source chunks and GPU-resident outputs.

## When to use it

Use GPU Dataframe when columnar data is already on the GPU or when its results will feed GPU rendering or
additional analysis. It is designed for explicit, repeatable plans rather than ad hoc CPU queries or
implicit downloads.

## Quick start

```ts
import {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {GPUDataFrame} from '@luma-contributors/graph/gpu-dataframe';

const graph = new GPUCommandGraph({device, id: 'dataframe-analysis'});
const frame = new GPUDataFrame({device, graph, table});
const filtered = frame.filter({column: 'duration', greaterThan: 10});

filtered.groupBy('service').mean('duration');
const compiledGraph = graph.compile();
```

## Core concepts and data model

- A dataframe borrows or owns explicit GPU table storage.
- Expressions describe work without encoding or submitting it.
- Compilation fixes formats, chunk topology, capacities, and execution shape.
- Filters and derived columns remain GPU-resident and can be shared with renderers or GPU Crossfilter.
- Global ordering and bounded joins require explicit policies rather than hidden repacking.

### Design note: two validity representations (deferred)

Nullable columns currently travel with a separate per-row `uint32` validity sidecar
(`GPUDataFrame.validity`, one word per row, `0` means null). The underlying `GPUData` primitive
also supports a packed bit-level `nullBitmap` (the Arrow representation). The sidecar is simple to
read and write from WGSL compute passes and lets derived columns publish validity without bit
packing, at the cost of 32x the memory of a bitmap and a second binding per nullable column.
The bitmap is compact and Arrow-compatible but needs word-level atomics or an extra packing pass
when a shader writes validity.

Unifying the two (for example deriving sidecars from `nullBitmap` at the Arrow boundary, or
storing bitmaps and unpacking in shaders) is a larger redesign of every compiler in this module
and is intentionally not part of the current work.

TODO: decide on one canonical representation, provide explicit conversion helpers between the
sidecar and `nullBitmap`, and migrate compilers behind a single validity-binding helper.

### Problem sizes baked into shaders

Row counts, batch lengths, capacities, and similar problem sizes are currently emitted into the
generated WGSL as constants. A compiled query is therefore specific to its source batch topology
and recompiles when sizes change. Moving sizes into uniform buffers would let one pipeline serve
changing inputs, but each shader needs its own review (bounds checks, dispatch layout, and
specialization benefits), so this is tracked as follow-up work rather than swapped mechanically.

## Try the interactive example

<GPUExampleCard
  demonstrates={['derived columns', 'filtering', 'grouped aggregation', 'sorting']}
  input="Chunked Arrow-independent GPU table columns"
  gpuOutput="Masks, derived columns, grouped summaries, and ordered row IDs"
  cpuReadback="Bounded table and chart results only"
  execution="Compiled queries re-encode against changed controls"
  compatibility="WebGPU"
  fullPageHref="/examples/experimental/gpu-data-analysis"
  sourceHref="https://github.com/visgl/luma.gl/tree/master/examples/experimental/gpu-data-analysis"
  inspectorHref="/examples/experimental/gpu-data-analysis?panel=graph"
/>

<GPUDataAnalysisExample embedded />

## Operations and API index

| Family | Operations |
| --- | --- |
| Data and expressions | `GPUDataFrame`, `GPUDataFrameQuery`, `GPUExpression`, `CompiledGPUDataFrameQuery`, filters, nullable derived columns |
| Aggregation | `GPUDataFrameGroupByQuery`, `GPUDataFrameGroupedAggregationQuery`, `GPUDataFrameAggregationQuery`, `GPUDataFrameHistogramQuery`, `CompiledGPUDataFrameGroupedAggregation`, `CompiledGPUDataFrameAggregation`, `CompiledGPUDataFrameHistogram` |
| Ordering | `GPUDataFrameSortQuery`, `GPUDataFrameGlobalSortQuery`, `CompiledGPUDataFrameSort`, `CompiledGPUDataFrameGlobalSort` |
| Indexes and joins | `GPUDataFrameLookupQuery`, `GPUDataFrameJoinQuery`, `CompiledGPUDataFrameLookup`, `CompiledGPUDataFrameJoin` |
| Integration | retained outputs for rendering, GPU Crossfilter, telemetry, and bounded readback |

See the [operations reference](./gpu-dataframe-operations) for detailed execution and ownership contracts.

## Catalog

One figure per contributor, each drawn from a recorded GPU run on real data; click through to its card.

::lens-catalog{entry=gpu-dataframe}

## Limits and compatibility

- GPU Dataframe is experimental and WebGPU-only.
- Source batch boundaries are preserved unless global behavior is requested explicitly.
- Join uniqueness, output capacity, null behavior, and overflow are explicit contracts.
- Submission, synchronization, readback, and fallback remain application-owned.

## Related modules

- [GPU Core](/docs/api-reference/experimental/gpu-core) supplies scheduling and primitives.
- [GPU Crossfilter](./gpu-crossfilter) composes linked interactive selections.
- [`@luma-contributors/graph/gpu-data`](/docs/api-reference/graph/gpu-data) defines primitive GPU storage,
  while [`@luma-contributors/graph/gpu-tables`](/docs/api-reference/experimental/gpu-tables) defines
  record batches and tables.
