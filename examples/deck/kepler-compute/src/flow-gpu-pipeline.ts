// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * GPU-resident Flow aggregation: location dedupe and origin-destination aggregation.
 *
 * These kernels cannot run in Node. `scripts/check-flow-gpu.mjs` compares them with the CPU
 * reference in `flow-gpu-aggregation.ts` (`deduplicateFlowEndpoints`, `aggregateFlowPairs`).
 *
 * Pipeline (all sorts are stable `GPUSort`, composite 64-bit keys use two LSD passes):
 * 1. One streaming pass uploads float32 source/target longitude and latitude and counts per
 *    Arrow chunk (chunk boundaries preserved; counts stay resident, coordinates are released after
 *    the location phase).
 * 2. Location phase over `2 * rowCount` endpoints (`2 * row` source, `2 * row + 1` target): key =
 *    float32 `(latitude bits, longitude bits)`; sort by key, flag run starts, scan to group
 *    indices, keep each run's smallest endpoint (first seen), sort runs by that endpoint to assign
 *    ids from 1 in first-seen order, and scatter the id to every endpoint (0 for non-finite).
 * 3. Pair phase per selection revision over `rowCount` rows: key = `(sourceId, targetId)` of
 *    selected rows with both ids (others get the invalid key), two-pass sort, run starts, scan, and
 *    `GPUGroupAggregation` sum of counts per run. Run starts scatter source id, target id and the
 *    smallest row (stable sort) to pair arrays.
 * Readbacks are O(unique locations) once and O(unique pairs) per revision.
 */

import {KeplerAllocationError} from '@luma-contributors/kepler';
import {Buffer, type Device} from '@luma.gl/core';
import {Kernel} from '@luma.gl/graph';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import {
  createTransientView,
  getBoundedDispatchLayout,
  getBoundedInvocationIndexSource,
  getViewBinding,
  getViewElementOffset,
  GPUCommandGraph,
  GPUGather,
  GPUGroupAggregation,
  GPUScan,
  GPUSort,
  type CompiledGPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {alignKeplerGraphVectorViews} from './luma-gaps/graph-vector-chunks';
import type * as arrow from 'apache-arrow';
import type {KeplerDatasetRegistry, KeplerMemoryReservation} from '@luma-contributors/kepler';
import {
  FLOW_INVALID_WORD,
  FLOW_LOCATION_WORKING_BYTES_PER_ROW,
  FLOW_PAIR_WORKING_BYTES_PER_ROW,
  FLOW_RETAINED_BYTES_PER_ROW,
  FlowColumnReader,
  MAXIMUM_FLOW_AGGREGATED_PAIRS,
  MAXIMUM_FLOW_LOCATIONS,
  getFlowBounds,
  sanitizeFlowCounts,
  type FlowAggregateSource,
  type FlowPairs,
  type FlowSelectionMask
} from './flow-gpu-aggregation';
import type {FlowRoles} from './flow-source';

const WORKGROUP_SIZE = 256;
const INVALID = `${FLOW_INVALID_WORD}u`;

type KernelBinding = {name: string; view: GraphDataView; write?: boolean};

/** Adds one per-index compute pass; `index` and `<name>Offset` constants are visible to `body`. */
function addIndexKernel(
  graph: GPUCommandGraph<void>,
  device: Device,
  id: string,
  length: number,
  bindings: readonly KernelBinding[],
  body: string
): void {
  if (!length) return;
  const dispatch = getBoundedDispatchLayout(
    id,
    length,
    WORKGROUP_SIZE,
    device.limits.maxComputeWorkgroupsPerDimension
  );
  graph.addComputePass({
    id,
    resources: bindings.map(binding => ({
      buffer: binding.view,
      usage: binding.write ? 'storage-read-write' : 'storage-read'
    })),
    compile: ({device: compileDevice}) => {
      const declarations = bindings
        .map(
          (binding, location) =>
            `@group(0) @binding(${location}) var<storage, ${binding.write ? 'read_write' : 'read'}> ${binding.name}: array<${binding.view.format === 'float32' ? 'f32' : 'u32'}>;\nconst ${binding.name}Offset: u32 = ${getViewElementOffset(binding.view)}u;`
        )
        .join('\n');
      const kernel = new Kernel(compileDevice, {
        id,
        source: `${declarations}
fn isFiniteValue(value: f32) -> bool { return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u; }
fn getKeyBits(value: f32) -> u32 { let bits = bitcast<u32>(value); return select(bits, 0u, (bits & 0x7fffffffu) == 0u); }
@compute @workgroup_size(${WORKGROUP_SIZE}) fn main(@builtin(local_invocation_index) localInvocationIndex: u32, @builtin(workgroup_id) workgroupId: vec3<u32>) {
${getBoundedInvocationIndexSource(dispatch, WORKGROUP_SIZE)}
if (index >= ${length}u) { return; }
${body}
}`,
        shaderLayout: {
          bindings: bindings.map((binding, location) => ({
            name: binding.name,
            type: 'storage' as const,
            group: 0,
            location
          }))
        }
      });
      return {
        encode: ({computePass, getBuffer}) =>
          kernel.dispatch(computePass, {
            bindings: Object.fromEntries(
              bindings.map(binding => [binding.name, getViewBinding(binding.view, getBuffer)])
            ),
            ...dispatch
          }),
        destroy: () => kernel.destroy()
      };
    }
  });
}

/** Imported, readable buffer with a full-length view. */
type ImportedView<Format extends 'uint32' | 'float32'> = {
  buffer: Buffer;
  view: GraphDataView<Format>;
};

function importView<Format extends 'uint32' | 'float32'>(
  graph: GPUCommandGraph<void>,
  device: Device,
  buffers: Buffer[],
  id: string,
  format: Format,
  length: number,
  existing?: Buffer
): ImportedView<Format> {
  const buffer =
    existing ??
    device.createBuffer({
      id: `flow-${id}`,
      byteLength: Math.max(16, length * 4),
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
    });
  if (!existing) buffers.push(buffer);
  return {
    buffer,
    view: graph.createDataView(
      graph.importBuffer({id, byteLength: buffer.byteLength, usage: buffer.usage}, buffer),
      {format, length}
    )
  };
}

/** Sorts `(keys, values)` stably into new transient views and returns the outputs. */
function addSort(
  graph: GPUCommandGraph<void>,
  id: string,
  keys: GraphDataView<'uint32'>,
  values: GraphDataView<'uint32'>,
  outputs?: {keys?: GraphDataView<'uint32'>; values?: GraphDataView<'uint32'>}
): {keys: GraphDataView<'uint32'>; values: GraphDataView<'uint32'>} {
  const outputKeys =
    outputs?.keys ?? createTransientView(graph, `${id}-keys`, 'uint32', keys.length);
  const outputValues =
    outputs?.values ?? createTransientView(graph, `${id}-values`, 'uint32', keys.length);
  graph.add(new GPUSort({id, keys, values, outputKeys, outputValues, algorithm: 'auto'}));
  return {keys: outputKeys, values: outputValues};
}

function addGather(
  graph: GPUCommandGraph<void>,
  id: string,
  source: GraphDataView<'uint32'> | GraphDataView<'float32'>,
  indices: GraphDataView<'uint32'>
): GraphDataView<'uint32'> | GraphDataView<'float32'> {
  const output = createTransientView(graph, id, source.format, indices.length);
  graph.add(new GPUGather({id, source, indices, output} as never));
  return output;
}

async function readWords(buffer: Buffer, wordCount: number): Promise<ArrayBuffer> {
  const bytes = await buffer.readAsync(0, wordCount * 4);
  return bytes.slice().buffer as ArrayBuffer;
}

type FlowGpuColumns = {
  sourceLongitude: GPUVector<'float32'>;
  sourceLatitude: GPUVector<'float32'>;
  targetLongitude: GPUVector<'float32'>;
  targetLatitude: GPUVector<'float32'>;
  count: GPUVector<'float32'>;
};

/**
 * Creates the resident aggregation: streams the table to the GPU, deduplicates locations and
 * reads back the unique first endpoints. Rejects tables with more than `MAXIMUM_FLOW_LOCATIONS`
 * unique locations.
 */
export async function createFlowGpuAggregator(
  device: Device,
  table: arrow.Table,
  props: {
    coordinates: {longitude: string; latitude: string};
    roles: FlowRoles;
    memoryRegistry?: KeplerDatasetRegistry;
  }
): Promise<FlowAggregateSource> {
  const aggregator = new FlowGpuAggregator(device, table.numRows, props.memoryRegistry);
  try {
    await aggregator.initialize(table, props.coordinates, props.roles);
    return aggregator;
  } catch (error) {
    aggregator.destroy();
    throw error;
  }
}

class FlowGpuAggregator implements FlowAggregateSource {
  firstEndpoints = new Uint32Array(0);
  bounds: [number, number, number, number] | null = null;
  readbackCount = 0;
  readbackBytes = 0;
  private endpointIds?: Buffer;
  private countVector?: GPUVector<'float32'>;
  private readonly buffers: Buffer[] = [];
  private readonly reservations: KeplerMemoryReservation[] = [];
  private queue: Promise<unknown> = Promise.resolve();
  private destroyed = false;

  constructor(
    private readonly device: Device,
    readonly rowCount: number,
    private readonly registry?: KeplerDatasetRegistry
  ) {}

  get gpuByteLength(): number {
    return this.rowCount * FLOW_RETAINED_BYTES_PER_ROW;
  }

  async initialize(
    table: arrow.Table,
    coordinates: {longitude: string; latitude: string},
    roles: FlowRoles
  ): Promise<void> {
    const {rowCount, device} = this;
    if (!rowCount) return;
    // Assertion: every endpoint-length buffer must fit one storage binding.
    if (rowCount * 8 > device.limits.maxStorageBufferBindingSize)
      throw new KeplerAllocationError('capacity', {
        message: 'Flow table exceeds the device storage buffer size'
      });
    const retained = this.reserve(rowCount * FLOW_RETAINED_BYTES_PER_ROW);
    if (retained) this.reservations.push(retained);
    const working = this.reserve(rowCount * FLOW_LOCATION_WORKING_BYTES_PER_ROW);
    const coordinateBuffers: Buffer[] = [];
    const scratchBuffers: Buffer[] = [];
    let compiled: CompiledGPUCommandGraph | undefined;
    try {
      const columns = this.uploadColumns(table, coordinates, roles, coordinateBuffers);
      const endpointCount = rowCount * 2;
      const graph = new GPUCommandGraph<void>(device, {id: 'flow-locations'});
      const meta = importView(graph, device, scratchBuffers, 'location-meta', 'uint32', 4);
      const endpointIds = importView(
        graph,
        device,
        this.buffers,
        'endpoint-ids',
        'uint32',
        endpointCount
      );
      this.endpointIds = endpointIds.buffer;
      const firstEndpoints = importView(
        graph,
        device,
        scratchBuffers,
        'first-endpoints',
        'uint32',
        endpointCount
      );
      const longitudeBits = createTransientView(graph, 'longitude-bits', 'uint32', endpointCount);
      const latitudeBits = createTransientView(graph, 'latitude-bits', 'uint32', endpointCount);
      const iota = createTransientView(graph, 'endpoint-iota', 'uint32', endpointCount);
      const imported = {
        sourceLongitude: graph.importGPUVector('source-longitude', columns.sourceLongitude),
        sourceLatitude: graph.importGPUVector('source-latitude', columns.sourceLatitude),
        targetLongitude: graph.importGPUVector('target-longitude', columns.targetLongitude),
        targetLatitude: graph.importGPUVector('target-latitude', columns.targetLatitude)
      };
      let rowBase = 0;
      for (const [
        spanIndex,
        [sourceLongitude, sourceLatitude, targetLongitude, targetLatitude]
      ] of alignKeplerGraphVectorViews(graph, [
        imported.sourceLongitude,
        imported.sourceLatitude,
        imported.targetLongitude,
        imported.targetLatitude
      ]).entries()) {
        const base = rowBase;
        rowBase += sourceLongitude.length;
        addIndexKernel(
          graph,
          device,
          `flow-pack-endpoints-${spanIndex}`,
          sourceLongitude.length,
          [
            {name: 'sourceLongitude', view: sourceLongitude},
            {name: 'sourceLatitude', view: sourceLatitude},
            {name: 'targetLongitude', view: targetLongitude},
            {name: 'targetLatitude', view: targetLatitude},
            {name: 'longitudeBits', view: longitudeBits, write: true},
            {name: 'latitudeBits', view: latitudeBits, write: true},
            {name: 'iota', view: iota, write: true}
          ],
          // Endpoint 2 * row is the source and 2 * row + 1 the target; invalid keys sort last.
          `let row = ${base}u + index;
let sourceLongitudeValue = sourceLongitude[sourceLongitudeOffset + index];
let sourceLatitudeValue = sourceLatitude[sourceLatitudeOffset + index];
let targetLongitudeValue = targetLongitude[targetLongitudeOffset + index];
let targetLatitudeValue = targetLatitude[targetLatitudeOffset + index];
let sourceValid = isFiniteValue(sourceLongitudeValue) && isFiniteValue(sourceLatitudeValue);
let targetValid = isFiniteValue(targetLongitudeValue) && isFiniteValue(targetLatitudeValue);
let sourceEndpoint = 2u * row;
longitudeBits[longitudeBitsOffset + sourceEndpoint] = select(${INVALID}, getKeyBits(sourceLongitudeValue), sourceValid);
latitudeBits[latitudeBitsOffset + sourceEndpoint] = select(${INVALID}, getKeyBits(sourceLatitudeValue), sourceValid);
longitudeBits[longitudeBitsOffset + sourceEndpoint + 1u] = select(${INVALID}, getKeyBits(targetLongitudeValue), targetValid);
latitudeBits[latitudeBitsOffset + sourceEndpoint + 1u] = select(${INVALID}, getKeyBits(targetLatitudeValue), targetValid);
iota[iotaOffset + sourceEndpoint] = sourceEndpoint;
iota[iotaOffset + sourceEndpoint + 1u] = sourceEndpoint + 1u;`
        );
      }
      // Stable LSD passes: longitude, then latitude => ordered by (latitude, longitude, endpoint).
      const byLongitude = addSort(graph, 'flow-sort-longitude', longitudeBits, iota);
      const latitudeGathered = addGather(
        graph,
        'flow-gather-latitude',
        latitudeBits,
        byLongitude.values
      ) as GraphDataView<'uint32'>;
      const byLatitude = addSort(graph, 'flow-sort-latitude', latitudeGathered, byLongitude.values);
      const longitudeGathered = addGather(
        graph,
        'flow-gather-longitude',
        longitudeBits,
        byLatitude.values
      ) as GraphDataView<'uint32'>;
      const runStarts = createTransientView(graph, 'run-starts', 'uint32', endpointCount);
      addIndexKernel(
        graph,
        device,
        'flow-location-run-starts',
        endpointCount,
        [
          {name: 'latitudeSorted', view: byLatitude.keys},
          {name: 'longitudeSorted', view: longitudeGathered},
          {name: 'runStarts', view: runStarts, write: true}
        ],
        `let isStart = index == 0u || latitudeSorted[latitudeSortedOffset + index] != latitudeSorted[latitudeSortedOffset + index - 1u] || longitudeSorted[longitudeSortedOffset + index] != longitudeSorted[longitudeSortedOffset + index - 1u];
runStarts[runStartsOffset + index] = select(0u, 1u, isStart);`
      );
      const runScan = createTransientView(graph, 'run-scan', 'uint32', endpointCount);
      graph.add(
        new GPUScan({
          id: 'flow-location-scan',
          input: runStarts,
          output: runScan,
          mode: 'inclusive'
        })
      );
      const firstKeys = createTransientView(graph, 'first-keys', 'uint32', endpointCount);
      const runIndices = createTransientView(graph, 'run-indices', 'uint32', endpointCount);
      addIndexKernel(
        graph,
        device,
        'flow-location-first-keys',
        endpointCount,
        [
          {name: 'runStarts', view: runStarts},
          {name: 'latitudeSorted', view: byLatitude.keys},
          {name: 'sortedEndpoints', view: byLatitude.values},
          {name: 'runScan', view: runScan},
          {name: 'firstKeys', view: firstKeys, write: true},
          {name: 'runIndices', view: runIndices, write: true}
        ],
        // Stability makes a run's first element its smallest endpoint, i.e. first seen.
        `let isRealStart = runStarts[runStartsOffset + index] == 1u && latitudeSorted[latitudeSortedOffset + index] != ${INVALID};
firstKeys[firstKeysOffset + index] = select(${INVALID}, sortedEndpoints[sortedEndpointsOffset + index], isRealStart);
runIndices[runIndicesOffset + index] = runScan[runScanOffset + index] - 1u;`
      );
      // Sorting runs by first endpoint ranks them in first-seen order: rank k has id k + 1.
      const byFirst = addSort(graph, 'flow-sort-first', firstKeys, runIndices, {
        keys: firstEndpoints.view
      });
      const idOfRun = createTransientView(graph, 'id-of-run', 'uint32', endpointCount);
      addIndexKernel(
        graph,
        device,
        'flow-location-clear-ids',
        endpointCount,
        [{name: 'idOfRun', view: idOfRun, write: true}],
        'idOfRun[idOfRunOffset + index] = 0u;'
      );
      addIndexKernel(
        graph,
        device,
        'flow-location-assign-ids',
        endpointCount,
        [
          {name: 'firstEndpoints', view: firstEndpoints.view},
          {name: 'rankedRuns', view: byFirst.values},
          {name: 'idOfRun', view: idOfRun, write: true}
        ],
        `if (firstEndpoints[firstEndpointsOffset + index] != ${INVALID}) { idOfRun[idOfRunOffset + rankedRuns[rankedRunsOffset + index]] = index + 1u; }`
      );
      addIndexKernel(
        graph,
        device,
        'flow-location-scatter-ids',
        endpointCount,
        [
          {name: 'sortedEndpoints', view: byLatitude.values},
          {name: 'runScan', view: runScan},
          {name: 'idOfRun', view: idOfRun},
          {name: 'endpointIds', view: endpointIds.view, write: true},
          {name: 'runSummary', view: meta.view, write: true},
          {name: 'latitudeSorted', view: byLatitude.keys}
        ],
        // The invalid run (if any) is last; runSummary = [run count, valid location count].
        `endpointIds[endpointIdsOffset + sortedEndpoints[sortedEndpointsOffset + index]] = idOfRun[idOfRunOffset + runScan[runScanOffset + index] - 1u];
if (index == ${endpointCount - 1}u) {
  let total = runScan[runScanOffset + index];
  runSummary[runSummaryOffset] = total;
  runSummary[runSummaryOffset + 1u] = select(total, total - 1u, latitudeSorted[latitudeSortedOffset + index] == ${INVALID});
}`
      );
      compiled = graph.compile();
      const encoder = device.createCommandEncoder({id: 'flow-locations'});
      compiled.encode(encoder, {parameters: undefined});
      device.submit(encoder.finish());
      const locationCount = new Uint32Array(await readWords(meta.buffer, 2))[1];
      this.readbackBytes += 8;
      // Assertion: too many unique locations for clustering and display; rejects before readback.
      if (locationCount > MAXIMUM_FLOW_LOCATIONS)
        throw new Error(`Flow layers support at most ${MAXIMUM_FLOW_LOCATIONS} unique locations`);
      this.firstEndpoints = new Uint32Array(await readWords(firstEndpoints.buffer, locationCount));
      this.readbackBytes += locationCount * 4;
      this.readbackCount++;
    } finally {
      compiled?.destroy();
      for (const buffer of scratchBuffers) buffer.destroy();
      for (const buffer of coordinateBuffers) buffer.destroy();
      working?.release();
    }
  }

  /** One explicit streaming upload; spans are the union of the columns' own chunk boundaries. */
  private uploadColumns(
    table: arrow.Table,
    coordinates: {longitude: string; latitude: string},
    roles: FlowRoles,
    coordinateBuffers: Buffer[]
  ): FlowGpuColumns {
    const getColumn = (name: string | undefined) => {
      const column = name ? table.getChild(name) : null;
      // Assertion: the retained table must contain every configured Flow column.
      if (name && !column) throw new Error('Flow column is missing from the retained source');
      return column;
    };
    const names = {
      sourceLongitude: coordinates.longitude,
      sourceLatitude: coordinates.latitude,
      targetLongitude: roles.targetLongitude,
      targetLatitude: roles.targetLatitude,
      count: roles.count
    };
    const vectors = Object.fromEntries(
      Object.entries(names).map(([key, name]) => [key, getColumn(name)])
    ) as Record<keyof typeof names, arrow.Vector | null>;
    const readers = Object.fromEntries(
      Object.entries(vectors).map(([key, vector]) => [key, new FlowColumnReader(vector)])
    ) as Record<keyof typeof names, FlowColumnReader>;
    const boundaries = new Set<number>();
    for (const vector of Object.values(vectors)) {
      let end = 0;
      for (const chunk of vector?.data ?? []) {
        end += chunk.length;
        boundaries.add(end);
      }
    }
    boundaries.add(this.rowCount);
    const chunks: Record<keyof typeof names, GPUData<'float32'>[]> = {
      sourceLongitude: [],
      sourceLatitude: [],
      targetLongitude: [],
      targetLatitude: [],
      count: []
    };
    const bounds: [number, number, number, number] = [Infinity, Infinity, -Infinity, -Infinity];
    let start = 0;
    for (const end of Array.from(boundaries).sort((first, second) => first - second)) {
      const length = end - start;
      if (length <= 0) continue;
      start = end;
      const spans = {
        sourceLongitude: readers.sourceLongitude.readFloat32(length),
        sourceLatitude: readers.sourceLatitude.readFloat32(length),
        targetLongitude: readers.targetLongitude.readFloat32(length),
        targetLatitude: readers.targetLatitude.readFloat32(length),
        // Null and non-finite counts contribute zero; a missing count column counts one.
        count: sanitizeFlowCounts(readers.count.readFloat32(length, 1))
      };
      const spanBounds = getFlowBounds(spans);
      if (spanBounds) {
        bounds[0] = Math.min(bounds[0], spanBounds[0]);
        bounds[1] = Math.min(bounds[1], spanBounds[1]);
        bounds[2] = Math.max(bounds[2], spanBounds[2]);
        bounds[3] = Math.max(bounds[3], spanBounds[3]);
      }
      for (const key of Object.keys(spans) as (keyof typeof names)[]) {
        const buffer = this.device.createBuffer({
          id: `flow-${key}`,
          data: spans[key],
          usage: Buffer.STORAGE | Buffer.COPY_DST
        });
        coordinateBuffers.push(buffer);
        chunks[key].push(new GPUData({buffer, format: 'float32', length, ownsBuffer: false}));
      }
    }
    this.bounds = bounds.every(Number.isFinite) ? bounds : null;
    const makeVector = (key: keyof typeof names): GPUVector<'float32'> =>
      new GPUVector({
        type: 'data',
        name: `flow-${key}`,
        format: 'float32',
        data: chunks[key],
        ownsData: false
      });
    // Counts stay resident across revisions, so they get their own buffers.
    const count = makeVector('count');
    this.buffers.push(...chunks.count.map(chunk => chunk.buffer as Buffer));
    this.countVector = count;
    for (const chunk of chunks.count) {
      const index = coordinateBuffers.indexOf(chunk.buffer as Buffer);
      if (index >= 0) coordinateBuffers.splice(index, 1);
    }
    return {
      sourceLongitude: makeVector('sourceLongitude'),
      sourceLatitude: makeVector('sourceLatitude'),
      targetLongitude: makeVector('targetLongitude'),
      targetLatitude: makeVector('targetLatitude'),
      count
    };
  }

  /** A registry-less aggregator reserves nothing. */
  private reserve(byteLength: number): KeplerMemoryReservation | undefined {
    return this.registry?.reserveMemory(byteLength, 'gpu', 'scratch');
  }

  aggregatePairs(mask?: FlowSelectionMask): Promise<FlowPairs> {
    // Serialized so concurrent revisions never hold two working sets at once.
    const result = this.queue.then(() => this.runPairs(mask as GPUVector<'uint32'> | undefined));
    this.queue = result.catch(() => undefined);
    return result;
  }

  private async runPairs(mask?: GPUVector<'uint32'>): Promise<FlowPairs> {
    const {rowCount, device} = this;
    if (this.destroyed) throw new Error('Flow aggregator was destroyed');
    if (!rowCount || !this.endpointIds || !this.countVector)
      return {
        sourceIds: new Uint32Array(0),
        targetIds: new Uint32Array(0),
        counts: new Float64Array(0),
        firstRows: new Uint32Array(0)
      };
    const working = this.reserve(rowCount * FLOW_PAIR_WORKING_BYTES_PER_ROW);
    const buffers: Buffer[] = [];
    let compiled: CompiledGPUCommandGraph | undefined;
    try {
      const graph = new GPUCommandGraph<void>(device, {id: 'flow-pairs'});
      const endpointIds = importView(
        graph,
        device,
        buffers,
        'endpoint-ids',
        'uint32',
        rowCount * 2,
        this.endpointIds
      );
      const meta = importView(graph, device, buffers, 'pair-meta', 'uint32', 4);
      const sums = importView(graph, device, buffers, 'pair-sums', 'float32', rowCount);
      const pairSources = importView(graph, device, buffers, 'pair-sources', 'uint32', rowCount);
      const pairTargets = importView(graph, device, buffers, 'pair-targets', 'uint32', rowCount);
      const pairRows = importView(graph, device, buffers, 'pair-rows', 'uint32', rowCount);
      const countsPacked = createTransientView(graph, 'counts-packed', 'float32', rowCount);
      const keyHigh = createTransientView(graph, 'key-high', 'uint32', rowCount);
      const keyLow = createTransientView(graph, 'key-low', 'uint32', rowCount);
      const iota = createTransientView(graph, 'row-iota', 'uint32', rowCount);
      const counts = graph.importGPUVector('counts', this.countVector);
      const selection = mask ? graph.importGPUVector('selection', mask) : undefined;
      let rowBase = 0;
      // Without a selection the counts align with themselves, so one span set serves both cases.
      for (const [spanIndex, [countSpan, selectionSpan]] of alignKeplerGraphVectorViews(graph, [
        counts,
        selection ?? counts
      ] as const).entries()) {
        const base = rowBase;
        rowBase += countSpan.length;
        addIndexKernel(
          graph,
          device,
          `flow-pack-pairs-${spanIndex}`,
          countSpan.length,
          [
            {name: 'counts', view: countSpan},
            ...(selection ? [{name: 'selection', view: selectionSpan as GraphDataView}] : []),
            {name: 'endpointIds', view: endpointIds.view},
            {name: 'countsPacked', view: countsPacked, write: true},
            {name: 'keyHigh', view: keyHigh, write: true},
            {name: 'keyLow', view: keyLow, write: true},
            {name: 'iota', view: iota, write: true}
          ],
          // Rows with an unresolved endpoint or a zero selection get the invalid key.
          `let row = ${base}u + index;
let sourceId = endpointIds[endpointIdsOffset + 2u * row];
let targetId = endpointIds[endpointIdsOffset + 2u * row + 1u];
let isSelected = ${selection ? 'selection[selectionOffset + index] != 0u' : 'true'};
let isValid = sourceId != 0u && targetId != 0u && isSelected;
countsPacked[countsPackedOffset + row] = counts[countsOffset + index];
keyHigh[keyHighOffset + row] = select(${INVALID}, sourceId, isValid);
keyLow[keyLowOffset + row] = select(${INVALID}, targetId, isValid);
iota[iotaOffset + row] = row;`
        );
      }
      // Stable LSD passes: target, then source => ordered by (source, target, row).
      const byTarget = addSort(graph, 'flow-sort-target', keyLow, iota);
      const sourceGathered = addGather(
        graph,
        'flow-gather-source',
        keyHigh,
        byTarget.values
      ) as GraphDataView<'uint32'>;
      const bySource = addSort(graph, 'flow-sort-source', sourceGathered, byTarget.values);
      const targetGathered = addGather(
        graph,
        'flow-gather-target',
        keyLow,
        bySource.values
      ) as GraphDataView<'uint32'>;
      const countsSorted = addGather(
        graph,
        'flow-gather-counts',
        countsPacked,
        bySource.values
      ) as GraphDataView<'float32'>;
      const runStarts = createTransientView(graph, 'pair-run-starts', 'uint32', rowCount);
      addIndexKernel(
        graph,
        device,
        'flow-pair-run-starts',
        rowCount,
        [
          {name: 'sourceSorted', view: bySource.keys},
          {name: 'targetSorted', view: targetGathered},
          {name: 'runStarts', view: runStarts, write: true}
        ],
        `let isStart = index == 0u || sourceSorted[sourceSortedOffset + index] != sourceSorted[sourceSortedOffset + index - 1u] || targetSorted[targetSortedOffset + index] != targetSorted[targetSortedOffset + index - 1u];
runStarts[runStartsOffset + index] = select(0u, 1u, isStart);`
      );
      const runScan = createTransientView(graph, 'pair-run-scan', 'uint32', rowCount);
      graph.add(
        new GPUScan({id: 'flow-pair-scan', input: runStarts, output: runScan, mode: 'inclusive'})
      );
      const runKeys = createTransientView(graph, 'pair-run-keys', 'uint32', rowCount);
      addIndexKernel(
        graph,
        device,
        'flow-pair-run-keys',
        rowCount,
        [
          {name: 'sourceSorted', view: bySource.keys},
          {name: 'runScan', view: runScan},
          {name: 'runKeys', view: runKeys, write: true}
        ],
        `runKeys[runKeysOffset + index] = select(runScan[runScanOffset + index] - 1u, ${INVALID}, sourceSorted[sourceSortedOffset + index] == ${INVALID});`
      );
      graph.add(
        new GPUGroupAggregation({
          id: 'flow-pair-sums',
          keys: runKeys,
          values: countsSorted,
          output: sums.view,
          operation: 'sum'
        })
      );
      addIndexKernel(
        graph,
        device,
        'flow-pair-emit',
        rowCount,
        [
          {name: 'runScan', view: runScan},
          {name: 'sourceSorted', view: bySource.keys},
          {name: 'targetSorted', view: targetGathered},
          {name: 'sortedRows', view: bySource.values},
          {name: 'pairSources', view: pairSources.view, write: true},
          {name: 'pairTargets', view: pairTargets.view, write: true},
          {name: 'pairRows', view: pairRows.view, write: true},
          {name: 'runSummary', view: meta.view, write: true}
        ],
        // Stability makes a run's first row its smallest selected row; runSummary = [runs, valid pairs].
        `let isValid = sourceSorted[sourceSortedOffset + index] != ${INVALID};
// A run starts where the inclusive scan increments (keeps the pass within 8 storage buffers).
let isRunStart = index == 0u || runScan[runScanOffset + index] != runScan[runScanOffset + index - 1u];
if (isRunStart && isValid) {
  let pair = runScan[runScanOffset + index] - 1u;
  pairSources[pairSourcesOffset + pair] = sourceSorted[sourceSortedOffset + index];
  pairTargets[pairTargetsOffset + pair] = targetSorted[targetSortedOffset + index];
  pairRows[pairRowsOffset + pair] = sortedRows[sortedRowsOffset + index];
}
if (index == ${rowCount - 1}u) {
  let total = runScan[runScanOffset + index];
  runSummary[runSummaryOffset] = total;
  runSummary[runSummaryOffset + 1u] = select(total, total - 1u, !isValid);
}`
      );
      compiled = graph.compile();
      const encoder = device.createCommandEncoder({id: 'flow-pairs'});
      compiled.encode(encoder, {parameters: undefined});
      device.submit(encoder.finish());
      const pairCount = new Uint32Array(await readWords(meta.buffer, 2))[1];
      this.readbackBytes += 8;
      this.readbackCount++;
      // Assertion: too many unique origin-destination pairs to draw; the previous flows stay.
      if (pairCount > MAXIMUM_FLOW_AGGREGATED_PAIRS)
        throw new Error(
          `Flow layers support at most ${MAXIMUM_FLOW_AGGREGATED_PAIRS} aggregated origin-destination pairs`
        );
      const [sumWords, sourceWords, targetWords, rowWords] = await Promise.all([
        readWords(sums.buffer, pairCount),
        readWords(pairSources.buffer, pairCount),
        readWords(pairTargets.buffer, pairCount),
        readWords(pairRows.buffer, pairCount)
      ]);
      this.readbackBytes += pairCount * 16;
      return {
        sourceIds: new Uint32Array(sourceWords),
        targetIds: new Uint32Array(targetWords),
        counts: Float64Array.from(new Float32Array(sumWords)),
        firstRows: new Uint32Array(rowWords)
      };
    } finally {
      compiled?.destroy();
      for (const buffer of buffers) if (buffer !== this.endpointIds) buffer.destroy();
      working?.release();
    }
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    // Queued work holds its own buffers; wait for it before releasing the resident ones.
    const release = () => {
      for (const buffer of this.buffers) buffer.destroy();
      for (const reservation of this.reservations) reservation.release();
    };
    void this.queue.then(release, release);
  }
}
