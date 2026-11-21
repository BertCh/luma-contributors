---
title: Analysis modules
description: Geospatial and analysis operations in @luma-contributors/analysis, built on published luma.gl.
---

# Analysis modules

`@luma-contributors/analysis` is an extension of [luma.gl](https://luma.gl). It holds only the
geospatial and analysis operations; the GPU infrastructure they run on (`GPUCommandGraph`, GPU Core
operations, `GPUData`/`GPUVector`/`GPUTable`, GPU Dataframe, GPU Graph Algorithms, GPU Project, GPU
DGGS, GPU Trace, physics and simulation, renderers) is owned and documented by luma.gl. Operations
here are built from the public `@luma.gl/core`, `@luma.gl/shaders` and `@luma.gl/graph` APIs.

## Run your first computation

The [CPU variogram walkthrough](./operation-forms-and-cpu-twins.md#variogram-fitting-without-a-graph)
fits generated data without a GPU. Follow it with the complete
[point-density GPU walkthrough](./operation-forms-and-cpu-twins.md#point-density-from-device-to-readback)
to create a device, allocate views, submit work, read typed results and release resources.

## Find an operation

Choose a task before allocating buffers. These destinations also appear in the sidebar:

| Task | Start here |
| --- | --- |
| Browse spatial tasks, weights, joins or statistics | [Spatial task cross-reference](./gpu-spatial-analysis/README.md#cross-reference) and [operator families](./gpu-spatial-analysis/operators.md) |
| Traverse or analyse a network | [Network operators](./gpu-network/operators.md) |
| Analyse elevation | [Terrain operators](./gpu-terrain/operators.md) |
| Transform or summarise a raster | [Raster operations chooser](./gpu-raster/operations.md) |
| Filter or aggregate table columns | [Dataframe analysis](./gpu-dataframe-analysis.md) |
| Filter or register point clouds | [Point cloud analysis](./pointcloud-analysis.md) |
| Transform coordinates | [Geospatial analysis](./geospatial.md) |
| Refine geometry | [Geometry refinement](./gpu-geometry-refinement.md) |
| Inspect a computation visually | [Computation views](./computation-views.md) |
| Browse every analysis entry | [Analysis catalog](./analysis-catalog.md) |

The four examples below use the same lifecycle with fully specified inputs and typed outputs.
In a browser TypeScript entry point served on localhost or HTTPS with WebGPU enabled, paste
the **Shared first-run lifecycle** block followed by any one example block. Each example imports
its own analysis contributors and runs without an existing device, graph, buffers or dataset.
For a workspace Vite page, use the setup in the linked point-density walkthrough; package imports
require a current package build. The examples return copied arrays that survive cleanup.
The slope, allocation IDs and overflow flag start at sentinels different from their expected
results, so an omitted GPU write cannot look like a successful zero result.

### Shared first-run lifecycle

Allocate each raw buffer once, import it once, then give it a typed graph view. The parameter
packers below produce ordinary `Float32Array` inputs, allocated with the same helper. All buffers
have storage, upload and readback usage. The callback constructs the graph and returns a readback
function, which runs only after compilation, encoding and submission.

```ts
import {Buffer, type Device, luma} from '@luma.gl/core';
import {webgpuAdapter} from '@luma.gl/core/webgpu';
import {GPUCommandGraph} from '@luma.gl/graph';
import {readPackedBuffer} from '@luma-contributors/analysis/analysis-core';

function createFirstRunContext(device: Device, graph: GPUCommandGraph, buffers: Buffer[]) {
  const allocate = (id: string, data: Float32Array | Uint32Array) => {
    const buffer = device.createBuffer({
      id, data, usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
    });
    buffers.push(buffer);
    const handle = graph.importBuffer(
      {id, byteLength: buffer.byteLength, usage: buffer.usage}, buffer
    );
    return {buffer, handle};
  };
  const float32 = (id: string, data: Float32Array) => {
    const {buffer, handle} = allocate(id, data);
    return {
      handle,
      view: graph.createDataView(handle, {format: 'float32', length: data.length}),
      read: (elementCount = data.length) => readPackedBuffer(buffer, 0, 'float32', elementCount)
    };
  };
  const uint32 = (id: string, data: Uint32Array) => {
    const {buffer, handle} = allocate(id, data);
    return {
      view: graph.createDataView(handle, {format: 'uint32', length: data.length}),
      read: (elementCount = data.length) => readPackedBuffer(buffer, 0, 'uint32', elementCount)
    };
  };
  return {graph, float32, uint32};
}

export async function runFirstComputationOnDevice<Result>(
  device: Device,
  id: string,
  construct: (context: ReturnType<typeof createFirstRunContext>) => () => Promise<Result>
) {
  const buffers: Buffer[] = [];
  let compiled: Awaited<ReturnType<GPUCommandGraph['compileAsync']>> | undefined;
  let encoder: ReturnType<Device['createCommandEncoder']> | undefined;
  try {
    const graph = new GPUCommandGraph(device, {id});
    const readResult = construct(createFirstRunContext(device, graph, buffers));
    compiled = await graph.compileAsync();
    encoder = device.createCommandEncoder();
    compiled.encode(encoder, {parameters: undefined});
    const commandBuffer = encoder.finish();
    encoder = undefined; // Successful finish releases the encoder and transfers its resources.
    device.submit(commandBuffer);
    return await readResult();
  } finally {
    encoder?.destroy(); // Encoding or finishing may throw before releasing transient resources.
    compiled?.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
}

export async function runFirstComputation<Result>(
  id: string,
  construct: (context: ReturnType<typeof createFirstRunContext>) => () => Promise<Result>
) {
  const device = await luma.createDevice({type: 'webgpu', adapters: [webgpuAdapter]});
  try {
    return await runFirstComputationOnDevice(device, id, construct);
  } finally {
    device.destroy();
  }
}
```

`readPackedBuffer(buffer, byteOffset, format, elementCount)` reads a packed logical range,
checks both allocation bounds and returned bytes, and returns an owned typed array. `byteOffset`
is in bytes from the buffer start; `format` is `'float32'` or `'uint32'`; `elementCount` is the
live scalar count, rather than allocation capacity. The helpers above use offset `0`; a packed
suballocation supplies its own offset, for example
`await readPackedBuffer(buffer, 16, 'uint32', liveCount)` for IDs starting at byte 16.
A zero count returns an empty array without a buffer read.
`readAsync()` waits for submitted results. For strided or chunked `GPUVector` inputs use
[`makeCPUVectorFromGPUVector`](https://luma.gl/docs/api-reference/graph/gpu-data) from
`@luma.gl/graph/gpu-data`, which carries the vector layout through transfer.

### Using a Deck-owned device

Use Deck's public `onDeviceInitialized(device)` callback to borrow its device. Configure
WebGPU explicitly because these contributors require it. Paste this block after the shared
lifecycle, then call `createAnalysisDeck` with your graph-construction callback:

```ts
import {Deck} from '@deck.gl/core';

export function createAnalysisDeck(
  construct: Parameters<typeof runFirstComputationOnDevice>[2]
) {
  return new Deck({
    deviceProps: {type: 'webgpu', adapters: [webgpuAdapter]},
    onDeviceInitialized(device) {
      void runFirstComputationOnDevice(device, 'deck-analysis', construct)
        .then(result => console.log(result))
        .catch(error => console.error(error));
    }
  });
}
```

Inside `construct`, import an already resident column with `graph.importGPUData(id, data)`;
it returns a typed graph view of that `GPUData`, preserving its logical offset and format.
Do not add borrowed buffers to the example's allocation list. The function releases its compiled
graph and allocated buffers on success or failure, including an unfinished encoder when
encoding or finishing throws. The owning wrapper also destroys the device
it created; the borrowed-device function leaves device ownership with Deck.

### First-run terrain derivatives

Each row of this north-up 3 × 3 DEM is `[0, 30, 60]` meters, with 30 m cells: elevation
rises eastward by one meter per horizontal meter. The center slope is `atan(1) = 45` degrees.
Light from the west (azimuth 270°) at altitude 45° aligns with the center surface normal,
so center hillshade is approximately `1`; both results must be finite.
Only the center is used for this independent expectation; consult the terrain reference for edges.

```ts
import {
  GPUTerrainDerivatives, getGPUTerrainDerivativesParameterValues
} from '@luma-contributors/analysis/gpu-terrain';
import {makeGPURasterBand, makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';

export async function runTerrainDerivativesExample() {
  return runFirstComputation('first-terrain', ({graph, float32}) => {
    const grid = makeGPURasterMetadata({width: 3, height: 3, cellSize: 30, origin: [0, 90]});
    const elevations = float32('elevations', Float32Array.of(0, 30, 60, 0, 30, 60, 0, 30, 60));
    const elevation = makeGPURasterBand(elevations.view, grid, {id: 'elevation'});
    const parameters = float32('parameters', getGPUTerrainDerivativesParameterValues({
      azimuthDegrees: 270, altitudeDegrees: 45
    }, grid));
    const slope = float32('slope', new Float32Array(9).fill(-1));
    const hillshade = float32('hillshade', new Float32Array(9));
    graph.add(new GPUTerrainDerivatives({
      grid, elevation, parameters: parameters.view,
      output: {slope: slope.view, hillshade: hillshade.view}
    }));
    return async () => ({slope: await slope.read(), hillshade: await hillshade.read()});
  });
}

const terrain = await runTerrainDerivativesExample();
console.log(terrain.slope[4], terrain.hillshade[4]);
// 45, approximately 1 (finite float32 results)
```

### First-run exact distance field

The sole seed `(1.5, 1.5)` is at the center of a north-up 3 × 3 unit grid. The exact route
measures cell-center distances: the four corners are `sqrt(2)` away, the four edge centers
are `1` away, and the seed cell is `0` away. Every cell is within radius `5` and belongs to
seed ID `0`. Rows are in raster order, with X varying fastest.

```ts
import {
  GPUDistanceField, makeGPURasterMetadata, getGPUDistanceFieldParameterValues
} from '@luma-contributors/analysis/raster-analysis';

export async function runExactDistanceFieldExample() {
  return runFirstComputation('first-distance', ({graph, float32, uint32}) => {
    const grid = makeGPURasterMetadata({width: 3, height: 3, cellSize: 1, origin: [0, 3]});
    const seeds = float32('seeds', Float32Array.of(1.5, 1.5));
    const seedPositions = graph.createDataView(seeds.handle, {format: 'float32x2', length: 1});
    const parameters = float32('parameters', getGPUDistanceFieldParameterValues({
      maximumDistance: 5
    }, grid));
    const distances = float32('distances', new Float32Array(9));
    const allocation = uint32('allocation', new Uint32Array(9).fill(0xffffffff));
    const withinDistance = uint32('within-distance', new Uint32Array(9));
    graph.add(new GPUDistanceField({
      grid, seedPositions, parameters: parameters.view, mode: 'exact',
      output: {distances: distances.view, allocation: allocation.view, withinDistance: withinDistance.view}
    }));
    return async () => ({
      distances: await distances.read(), allocation: await allocation.read(),
      withinDistance: await withinDistance.read()
    });
  });
}

const distanceField = await runExactDistanceFieldExample();
console.log(Array.from(distanceField.distances));
// [sqrt(2), 1, sqrt(2), 1, 0, 1, sqrt(2), 1, sqrt(2)] within float32 tolerance
console.log(Array.from(distanceField.allocation), Array.from(distanceField.withinDistance));
// [0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1]
```

### First-run time-window filter

Three events one second apart use Unix milliseconds in the original JavaScript doubles.
Subtract the epoch `t` **before** constructing a `Float32Array`, and subtract it from both
window endpoints. Uploading absolute Unix milliseconds directly collapses these events to
one float32 value. The inclusive middle-event window accepts only source row `1`.
To retain absolute times, use the [double-single `splitTimestamps` / `timestampsLow` route](./gpu-dataframe-analysis.md#gputimewindowfilter). Read compact rows through `count`; the unused tail of the
three-row capacity is outside the result. Read `requiredCount` and `overflow` with the rows.

```ts
import {
  GPUTimeWindowFilter, getGPUTimeWindowFilterParameterValues
} from '@luma-contributors/analysis/dataframe-analysis';

export async function runTimeWindowFilterExample() {
  return runFirstComputation('first-time-window', ({graph, float32, uint32}) => {
    const t = Date.UTC(2026, 10, 19);
    const events = [t, t + 1000, t + 2000];
    const timestamps = float32('timestamps', Float32Array.from(events, time => time - t));
    const window = float32('window', getGPUTimeWindowFilterParameterValues({
      start: (t + 1000) - t, end: (t + 1000) - t
    }));
    const rows = uint32('rows', new Uint32Array(3));
    const count = uint32('count', new Uint32Array(1));
    const requiredCount = uint32('required-count', new Uint32Array(1));
    const overflow = uint32('overflow', new Uint32Array(1).fill(0xffffffff));
    graph.add(new GPUTimeWindowFilter({
      timestamps: timestamps.view, window: window.view,
      output: {rows: rows.view, status: {
        count: count.view, requiredCount: requiredCount.view, overflow: overflow.view
      }}
    }));
    return async () => {
      const acceptedCount = (await count.read())[0];
      const required = (await requiredCount.read())[0];
      const lost = (await overflow.read())[0];
      if (lost !== 0) throw new Error('Time-window output is incomplete; resize and rerun');
      return {rows: await rows.read(acceptedCount), count: acceptedCount, requiredCount: required, overflow: lost};
    };
  });
}

const timeWindow = await runTimeWindowFilterExample();
console.log(Array.from(timeWindow.rows), timeWindow.count, timeWindow.requiredCount, timeWindow.overflow);
// [1], 1, 1, 0
```

### First-run point-cloud statistical filter

This filter-only fixture supplies packed squared neighbor distances and counts directly; no
positions or neighborhood producer are implicit. Capacity `1` means one retained slot per row.
Square roots give supported means `[1, 1, 10]`; the unsupported fourth row receives `-1` and is
excluded from global moments. The supported mean is `(1 + 1 + 10) / 3 = 4`; the population
variance is `((1 - 4)^2 + (1 - 4)^2 + (10 - 4)^2) / 3 = 18`.

The inlier threshold is `4 + multiplier * sqrt(18)`. At multipliers `2` and `1.5`, `10` is
retained; at `0`, it is rejected. Including the unsupported row as zero would instead give
mean `3`, variance `16.5` and threshold about `9.093` at `1.5`, incorrectly rejecting `10`.

```ts
import {GPUPointCloudStatisticalFilter} from '@luma-contributors/analysis/pointcloud-analysis';

export async function runPointCloudStatisticalFilterExample(standardDeviations: number) {
  return runFirstComputation('first-point-cloud-filter', ({graph, float32, uint32}) => {
    const squaredDistances = float32('squared-distances', Float32Array.of(1, 1, 100, 0));
    const counts = uint32('counts', Uint32Array.of(1, 1, 1, 0));
    const meanDistances = float32('mean-distances', new Float32Array(4));
    const validity = uint32('validity', new Uint32Array(4));
    graph.add(new GPUPointCloudStatisticalFilter({
      squaredDistances: squaredDistances.view, counts: counts.view,
      capacity: 1, standardDeviations,
      output: {meanDistances: meanDistances.view, validity: validity.view}
    }));
    return async () => ({meanDistances: await meanDistances.read(), validity: await validity.read()});
  });
}

for (const multiplier of [2, 0, 1.5]) {
  const filtered = await runPointCloudStatisticalFilterExample(multiplier);
  console.log(multiplier, Array.from(filtered.meanDistances), Array.from(filtered.validity));
}
// 2:   [1, 1, 10, -1], [1, 1, 1, 0]
// 0:   [1, 1, 10, -1], [1, 1, 0, 0]
// 1.5: [1, 1, 10, -1], [1, 1, 1, 0]
```

When supplying neighborhoods from another producer, handle its candidate-loss and overflow status
before treating these slots as a complete spatial-neighborhood analysis.

For documentation fixtures, the stable headings above identify exactly one TypeScript fence each.
Concatenate the shared fence with one example fence, preserving imports and executable bodies.
The functions return the displayed typed results for independent assertions; fixtures should check
finite numeric values before comparing them with the expectations above.

## Arrow tables back to the CPU

A graph buffer handle alone carries no table schema. Use the upstream Arrow adapters to retain
field names, schema metadata, nullability and dictionary labels alongside resident columns.
This complete round trip borrows a device, uploads three source rows, and materializes compact
source rows `[1, 2]`. The selection vectors contain source-row indices (not caller IDs) and one
live count per source batch. Here there is one batch; for multiple batches preserve one index
chunk and count chunk per batch. Check the producing operation's overflow and candidate-loss
status before materializing a selection.

```ts
import type {Device} from '@luma.gl/core';
import * as arrow from 'apache-arrow';
import {
  makeGPUAnalyticsTableFromArrowTable, makeArrowTableFromGPUAnalyticsTable,
  makeGPUVectorFromArrow
} from '@luma.gl/graph/arrow';

export async function roundTripSelectedArrowRows(device: Device) {
  const source = new arrow.Table({
    constructor: arrow.vectorFromArray([42, 43, 44], new arrow.Int32()),
    score: arrow.vectorFromArray([5, null, 7], new arrow.Int32()),
    category: arrow.vectorFromArray(
      ['west', 'east', 'west'], new arrow.Dictionary(new arrow.Utf8(), new arrow.Int32())
    )
  });
  const resident = makeGPUAnalyticsTableFromArrowTable(device, source);
  let rowIndices: ReturnType<typeof makeGPUVectorFromArrow> | undefined;
  let selectedCounts: ReturnType<typeof makeGPUVectorFromArrow> | undefined;
  try {
    // Keep only own entries: legal column names such as constructor must not resolve
    // inherited Object.prototype values in metadata lookups.
    const validity: typeof resident.validity = Object.assign(Object.create(null), resident.validity);
    const dictionaries: typeof resident.dictionaries = Object.assign(Object.create(null), resident.dictionaries);
    const selectedRows = makeGPUVectorFromArrow(
      device, arrow.vectorFromArray([1, 2, 0], new arrow.Uint32()), {format: 'uint32'}
    );
    rowIndices = selectedRows;
    const liveCounts = makeGPUVectorFromArrow(
      device, arrow.vectorFromArray([2], new arrow.Uint32()), {format: 'uint32'}
    );
    selectedCounts = liveCounts;
    const result = await makeArrowTableFromGPUAnalyticsTable({
      table: resident.table, validity, dictionaries,
      rowIndices: selectedRows, selectedCounts: liveCounts
    });
    console.log(Array.from(result.getChild('constructor')!)); // [43, 44]
    console.log(Array.from(result.getChild('score')!)); // [null, 7]
    console.log(Array.from(result.getChild('category')!)); // ['east', 'west']
    return result;
  } finally {
    rowIndices?.destroy();
    selectedCounts?.destroy();
    for (const validity of Object.values(resident.validity)) validity?.destroy();
    resident.table.destroy();
  }
}
```

The materializer gathers source-aligned validity with the selected source rows and reconstructs
the dictionary from its retained labels. Keep those maps with the table through GPU stages;
a stage that changes values or nullability must provide matching schema and validity for its
result. Copying own metadata entries into null-prototype maps also handles absent metadata for
fields named `constructor`. The returned Arrow table is CPU-owned and survives GPU cleanup.

### Fields named __proto__

The own-key readback maps above cover `constructor`, but the installed uploader also builds
ordinary objects. A nullable field named `__proto__` loses its validity entry during upload,
before the copies can repair it. Rename that field to an unused safe name before upload and
restore it after materialization. Keep a reversible name map with the schema; use the safe
name in GPU column references and metadata maps. This helper preserves field types, nullability,
metadata and source batch boundaries and rejects name collisions:

```ts
import type {Device} from '@luma.gl/core';
import * as arrow from 'apache-arrow';
import {
  makeGPUAnalyticsTableFromArrowTable, makeArrowTableFromGPUAnalyticsTable
} from '@luma.gl/graph/arrow';

export function renameArrowFields(source: arrow.Table, names: ReadonlyMap<string, string>) {
  const renameSchema = (schema: arrow.Schema) => {
    const fields = schema.fields.map(field => field.clone({name: names.get(field.name) ?? field.name}));
    if (new Set(fields.map(field => field.name)).size !== fields.length) {
      throw new Error('Choose unused Arrow field aliases; renaming must be reversible');
    }
    return new arrow.Schema(fields, new Map(schema.metadata));
  };
  return new arrow.Table(
    renameSchema(source.schema),
    source.batches.map(batch => new arrow.RecordBatch(renameSchema(batch.schema), batch.data))
  );
}

export async function roundTripPrototypeField(device: Device) {
  const source = new arrow.Table({
    ['__proto__']: arrow.vectorFromArray([18, null, 5], new arrow.Int32())
  });
  const uploadNames = new Map([['__proto__', '__analysis_proto__']]);
  const restoreNames = new Map(Array.from(uploadNames, ([original, alias]) => [alias, original]));
  const resident = makeGPUAnalyticsTableFromArrowTable(device, renameArrowFields(source, uploadNames));
  try {
    const result = await makeArrowTableFromGPUAnalyticsTable({
      table: resident.table,
      validity: Object.assign(Object.create(null), resident.validity),
      dictionaries: Object.assign(Object.create(null), resident.dictionaries)
    });
    return renameArrowFields(result, restoreNames); // __proto__: [18, null, 5]
  } finally {
    for (const validity of Object.values(resident.validity)) validity?.destroy();
    resident.table.destroy();
  }
}
```

### Single numeric vectors

For a single numeric `GPUVector`, use `makeCPUVectorFromGPUVector` from
`@luma.gl/graph/gpu-data`, then `makeArrowVectorFromCPUVector` from `@luma.gl/graph/arrow`.
These public transfers honor vector chunking and strides. Preserve the vector's validity metadata;
external analytical table validity maps are not implicitly attached to a bare numeric vector.

## Entries

| Entry | Use it for |
| --- | --- |
| [`@luma-contributors/analysis`](/docs/api/computation-record) | Root (`.`): alias of `computation-record`; recording APIs. |
| [`@luma-contributors/analysis/gpu-spatial-analysis/ontology`](/docs/api/analysis-catalog) | Catalog import: operator records and analysis ontology. |
| [`@luma-contributors/analysis/gpu-spatial-analysis`](/docs/api/gpu-spatial-analysis) | Spatial weights and statistics, joins, density, cells, lines, geometry, trajectories, and regression. |
| [`@luma-contributors/analysis/gpu-network`](/docs/api/gpu-network) | CSR network analysis: reachability, service areas, accessibility, flow aggregation, and edge bundling. |
| [`@luma-contributors/analysis/gpu-terrain`](/docs/api/gpu-terrain) | Elevation-tile analysis: slope, contours, viewshed, hydrology, solar shadows, and relief shading. |
| [`@luma-contributors/analysis/raster-analysis`](/docs/api/gpu-raster) | Raster analysis: fill nodata, tile reprojection, fractional coverage, focal pairs, spectral indices, regions, contours, and statistics. |
| [`@luma-contributors/analysis/dataframe-analysis`](/docs/api/gpu-dataframe-analysis) | Analysis contributors over luma.gl GPU Dataframe columns: profiles, group statistics, joins, and standardization. |
| [`@luma-contributors/analysis/geospatial-analysis`](/docs/api/geospatial) | Geospatial contexts, datum grid transforms and point buffers layered on luma.gl GPU Project. |
| [`@luma-contributors/analysis/pointcloud-analysis`](/docs/api/pointcloud-analysis) | Point cloud neighborhoods, features, ground classification, registration, and change. |
| [`@luma-contributors/analysis/computation-record`](/docs/api/computation-record) | Recorded executions of a computation as plain data, for the lens views. |
| [`@luma-contributors/analysis/analysis-core`](/docs/api/gpu-spatial-analysis/concepts) | Shared kernels and contracts used by the analysis entries. |

See the [analysis catalog](/docs/api/analysis-catalog) for every contributor
at a glance and the [cross-reference](/docs/api/gpu-spatial-analysis-cross-reference)
for a map from familiar tool names to contributors.

Read [operation forms and CPU twins](/docs/api/operation-forms-and-cpu-twins) to choose between
caller-owned GPU views, named graph operations and CPU execution, with a complete variogram fit.

## What luma.gl owns

Link to the luma.gl documentation rather than to copies here:

- [GPU Core and `GPUCommandGraph`](https://luma.gl/docs/api-reference/experimental/gpu-core)
- [GPU Data and GPU Tables](https://luma.gl/docs/api-reference/graph/gpu-data)
- [GPU Dataframe and GPU SQL](https://luma.gl/docs/api-reference/experimental/gpu-dataframe)
- [GPU Graph Algorithms](https://luma.gl/docs/api-reference/experimental/gpu-graph)
- [GPU Project and GPU DGGS](https://luma.gl/docs/api-reference/experimental/gpu-project)
- [Glossary](https://luma.gl/docs/glossary)

## Conventions

Contributors declare work on a caller-owned `GPUCommandGraph` and return command nodes; callers own
the frame loop, command submission and readback. Entries are WebGPU-first; individual pages state
backend, capacity and precision limits.
