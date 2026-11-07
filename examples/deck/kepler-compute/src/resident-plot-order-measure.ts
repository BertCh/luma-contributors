// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type CommandEncoder} from '@luma.gl/core';
import {Kernel} from '@luma.gl/graph';
import type {GPUVector} from '@luma.gl/graph/gpu-data';
import {
  GPUCommandGraph,
  getBoundedDispatchLayout,
  getBoundedInvocationIndexSource,
  type CompiledGPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {GPUGroupStatistics} from '@luma-contributors/analysis/dataframe-analysis';
import type {
  KeplerArrowDataset,
  KeplerComputeSession,
  KeplerDatasetLease,
  KeplerMemoryReservation
} from '@luma-contributors/kepler';
import {KeplerAllocationError} from '@luma-contributors/kepler';
import {
  ResidentPlotMeasureWidening,
  isWidenableResidentPlotMeasure
} from './resident-plot-measure-widening';
import type {ResidentPlotMeasureSummary, ResidentPlotOrderOperation} from './resident-plot-measure';

/** Declared upper bound on rows whose y values are sorted for exact per-bin statistics. */
export const MAXIMUM_RESIDENT_ORDER_MEASURE_ROWS = 4_194_304;
/** Packed key, mask and value copies plus the radix-sort scratch of `GPUGroupStatistics`. */
const ORDER_MEASURE_BYTES_PER_ROW = 96;

/**
 * Exact per-key median or sample standard deviation of one Float32 column. Keys are the plot's
 * own bin classification (or a composite group-and-bin key), so the statistic and the histogram
 * cannot disagree about membership.
 *
 * `GPUGroupStatistics` stable-sorts rows by key and re-sorts the values inside each key: the median
 * is the linear-interpolated 0.5 quantile of d3 `median`, and the deviation is the square root of
 * the corrected two-pass sample variance of d3 `deviation`. Null and non-finite y values are
 * skipped like d3 skips null and NaN; a key with no value has no median, and one with fewer than
 * two values has no deviation (d3 returns undefined for both). Sources are packed into one
 * buffer per input, a declared working-memory cost of `ORDER_MEASURE_BYTES_PER_ROW` per row.
 */
export class ResidentPlotOrderMeasure {
  private readonly compiled: CompiledGPUCommandGraph;
  private readonly reservation: KeplerMemoryReservation;
  private readonly buffers: Buffer[] = [];
  private readonly valuesBuffer: Buffer;
  private readonly countsBuffer: Buffer;
  private widening?: ResidentPlotMeasureWidening;
  private destroyed = false;

  constructor(
    private readonly props: {
      session: KeplerComputeSession;
      lease: KeplerDatasetLease<KeplerArrowDataset>;
      /** Per-row dense keys; omitted for a single global statistic over key zero. */
      keys?: GPUVector<'uint32'>;
      /** Rows with a zero mask are excluded; omitted when every row counts. */
      mask?: GPUVector<'uint32'>;
      /** Rows with zero key validity receive an out-of-range key and are excluded. */
      keyValidity?: GPUVector<'uint32'>;
      /** Exact Float32 replacement for the source column, such as a widened integer measure. */
      values?: GPUVector<'float32'>;
      keyCount: number;
      column: string;
      operation: ResidentPlotOrderOperation;
      workingMemoryBudgetBytes: number;
    }
  ) {
    const {session, keyCount} = props;
    const source = props.lease.dataset.value;
    const sourceColumn = props.values ?? source.columns[props.column];
    // Float32 is exact; Int32 and Uint32 widen on the GPU behind a +/-2^24 range check.
    const needsWidening =
      !props.values &&
      isWidenableResidentPlotMeasure(
        sourceColumn,
        Boolean(source.prepared.dictionaries[props.column])
      );
    if (!sourceColumn || (sourceColumn.format !== 'float32' && !needsWidening))
      throw new Error('Resident line-chart measures require a numeric field');
    let measured = sourceColumn as GPUVector<'float32'>;
    const rowCount = measured.length;
    if (rowCount < 1 || rowCount > MAXIMUM_RESIDENT_ORDER_MEASURE_ROWS)
      throw new Error(
        `Exact median and standard deviation are limited to ${MAXIMUM_RESIDENT_ORDER_MEASURE_ROWS} rows`
      );
    const estimatedBytes =
      65536 +
      rowCount * (ORDER_MEASURE_BYTES_PER_ROW + (needsWidening ? 4 : 0)) +
      keyCount * 128 +
      measured.data.length * 8192;
    if (estimatedBytes > props.workingMemoryBudgetBytes)
      throw new KeplerAllocationError('working-memory', {
        message: 'Line-chart measure exceeds its working memory budget'
      });
    this.reservation = session.registry.reserveMemory(estimatedBytes, 'gpu', 'scratch');
    let compiled: CompiledGPUCommandGraph | undefined;
    try {
      const device = session.device;
      if (needsWidening && isWidenableResidentPlotMeasure(sourceColumn, false)) {
        this.widening = new ResidentPlotMeasureWidening(device, sourceColumn);
        measured = this.widening.vector;
      }
      const allocate = (byteLength: number, label: string) => {
        const buffer = device.createBuffer({
          id: `plot-order-${label}`,
          byteLength: Math.max(4, byteLength),
          usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
        });
        this.buffers.push(buffer);
        return buffer;
      };
      const graph = new GPUCommandGraph(device, {id: 'kepler-plot-order-measure'});
      const view = <Format extends 'uint32' | 'float32'>(
        label: string,
        format: Format,
        length: number
      ): {buffer: Buffer; view: GraphDataView<Format>} => {
        const buffer = allocate(length * 4, label);
        return {
          buffer,
          view: graph.createDataView(
            graph.importBuffer(
              {id: label, byteLength: buffer.byteLength, usage: buffer.usage},
              buffer
            ),
            {format, length}
          )
        };
      };
      const packedKeys = view('keys', 'uint32', rowCount);
      const packedMask = view('mask', 'uint32', rowCount);
      const packedValues = view('values', 'float32', rowCount);
      const keys = props.keys ? graph.importGPUVector('source-keys', props.keys) : undefined;
      const mask = props.mask ? graph.importGPUVector('source-mask', props.mask) : undefined;
      const keyValidity = props.keyValidity
        ? graph.importGPUVector('source-key-validity', props.keyValidity)
        : undefined;
      const values = graph.importGPUVector('source-values', measured);
      const validityVector = source.analytics.validity[props.column];
      const validity = validityVector
        ? graph.importGPUVector('source-validity', validityVector)
        : undefined;
      let rowBase = 0;
      for (const [chunkIndex, valueChunk] of values.data.entries()) {
        const length = valueChunk.length;
        if (!length) continue;
        const keyChunk = keys?.data[chunkIndex];
        const maskChunk = mask?.data[chunkIndex];
        const keyValidChunk = keyValidity?.data[chunkIndex];
        const validChunk = validity?.data[chunkIndex];
        const base = rowBase;
        rowBase += length;
        const dispatch = getBoundedDispatchLayout(
          'PlotOrderMeasurePack',
          length,
          256,
          device.limits.maxComputeWorkgroupsPerDimension
        );
        graph.addComputePass({
          id: `pack-${chunkIndex}`,
          resources: [
            ...(keyChunk ? [{buffer: keyChunk, usage: 'storage-read' as const}] : []),
            ...(maskChunk ? [{buffer: maskChunk, usage: 'storage-read' as const}] : []),
            ...(keyValidChunk ? [{buffer: keyValidChunk, usage: 'storage-read' as const}] : []),
            {buffer: valueChunk, usage: 'storage-read'},
            ...(validChunk ? [{buffer: validChunk, usage: 'storage-read' as const}] : []),
            {buffer: packedKeys.view, usage: 'storage-write'},
            {buffer: packedMask.view, usage: 'storage-write'},
            {buffer: packedValues.view, usage: 'storage-write'}
          ],
          compile: ({device: compileDevice}) => {
            // Optional inputs are bound only when present; names double as WGSL identifiers.
            const inputNames = [
              ...(keyChunk ? ['keys'] : []),
              ...(maskChunk ? ['selected'] : []),
              ...(keyValidChunk ? ['keyValidity'] : []),
              'values',
              ...(validChunk ? ['validity'] : []),
              'packedKeys',
              'packedMask',
              'packedValues'
            ];
            const declarations = inputNames
              .map(
                (name, location) =>
                  `@group(0) @binding(${location}) var<storage,${name.startsWith('packed') ? 'read_write' : 'read'}> ${name}:array<u32>;`
              )
              .join('\n');
            const keySource = keyChunk ? `keys[${keyChunk.byteOffset / 4}u + index]` : '0u';
            const kernel = new Kernel(compileDevice, {
              source: `
${declarations}
@compute @workgroup_size(256) fn main(@builtin(local_invocation_index) localInvocationIndex:u32,@builtin(workgroup_id) workgroupId:vec3<u32>) {
${getBoundedInvocationIndexSource(dispatch, 256)}
if (index >= ${length}u) {return;}
let row = ${base}u + index;
${
  keyValidChunk
    ? `packedKeys[row] = select(0xffffffffu, ${keySource}, keyValidity[${keyValidChunk.byteOffset / 4}u + index] != 0u);`
    : `packedKeys[row] = ${keySource};`
}
packedMask[row] = ${maskChunk ? `selected[${maskChunk.byteOffset / 4}u + index]` : '1u'};
// A null y value is skipped like d3 skips null: it becomes a NaN, which the statistics exclude.
${
  validChunk
    ? `packedValues[row] = select(0x7fc00000u, values[${valueChunk.byteOffset / 4}u + index], validity[${validChunk.byteOffset / 4}u + index] != 0u);`
    : `packedValues[row] = values[${valueChunk.byteOffset / 4}u + index];`
}
}`,
              shaderLayout: {
                bindings: inputNames.map((name, location) => ({
                  name,
                  location,
                  group: 0,
                  type: 'storage' as const
                }))
              }
            });
            return {
              encode: ({computePass, getBuffer}) =>
                kernel.dispatch(computePass, {
                  bindings: {
                    ...(keyChunk ? {keys: getBuffer(keyChunk.buffer)} : {}),
                    ...(maskChunk ? {selected: getBuffer(maskChunk.buffer)} : {}),
                    ...(keyValidChunk ? {keyValidity: getBuffer(keyValidChunk.buffer)} : {}),
                    values: getBuffer(valueChunk.buffer),
                    packedKeys: getBuffer(packedKeys.view.buffer),
                    packedMask: getBuffer(packedMask.view.buffer),
                    packedValues: getBuffer(packedValues.view.buffer),
                    ...(validChunk ? {validity: getBuffer(validChunk.buffer)} : {})
                  },
                  ...dispatch
                }),
              destroy: () => kernel.destroy()
            };
          }
        });
      }
      const outputKeys = view('output-keys', 'uint32', keyCount);
      const outputCounts = view('output-counts', 'uint32', keyCount);
      const columnCounts = view('column-counts', 'uint32', keyCount);
      const statistic = view('statistic', 'float32', keyCount);
      const statusCount = view('status-count', 'uint32', 1);
      const statusOverflow = view('status-overflow', 'uint32', 1);
      this.valuesBuffer = statistic.buffer;
      this.countsBuffer = columnCounts.buffer;
      graph.add(
        new GPUGroupStatistics({
          keys: packedKeys.view,
          mask: packedMask.view,
          keyCount,
          variance: 'sample',
          columns: [
            {
              values: packedValues.view,
              statistics: ['count', props.operation === 'median' ? 'median' : 'standardDeviation'],
              output: {
                counts: columnCounts.view,
                ...(props.operation === 'median'
                  ? {medians: statistic.view}
                  : {standardDeviations: statistic.view})
              }
            }
          ],
          output: {
            keys: outputKeys.view,
            counts: outputCounts.view,
            status: {count: statusCount.view, overflow: statusOverflow.view}
          }
        })
      );
      compiled = graph.compile();
      this.compiled = compiled;
      const accountedBytes =
        (this.widening?.byteLength ?? 0) +
        this.buffers.reduce((total, buffer) => total + buffer.byteLength, 0) +
        compiled.stats.physicalTransientResourceBytes;
      if (accountedBytes > estimatedBytes)
        throw new KeplerAllocationError('working-memory', {
          message: 'Line-chart measure exceeds its reserved working memory'
        });
    } catch (error) {
      compiled?.destroy();
      this.widening?.destroy();
      this.buffers.forEach(buffer => buffer.destroy());
      this.reservation.release();
      throw error;
    }
  }

  /** Encode after the key classification pass on the same command encoder. */
  encode(encoder: CommandEncoder): void {
    this.widening?.encode(encoder);
    this.compiled.encode(encoder, {parameters: undefined});
  }

  get readbackByteLength(): number {
    return this.props.keyCount * 8;
  }

  async readSummary(): Promise<ResidentPlotMeasureSummary> {
    const {keyCount} = this.props;
    await this.widening?.assertExactRange();
    const byteLength = keyCount * 4;
    const [valueBytes, countBytes] = await Promise.all([
      this.valuesBuffer.readAsync(0, byteLength),
      this.countsBuffer.readAsync(0, byteLength)
    ]);
    const values = new Float32Array(keyCount);
    const counts = new Uint32Array(keyCount);
    const valid = new Uint8Array(keyCount);
    const valueView = new DataView(valueBytes.buffer, valueBytes.byteOffset, byteLength);
    const countView = new DataView(countBytes.buffer, countBytes.byteOffset, byteLength);
    const minimumCount = this.props.operation === 'median' ? 1 : 2;
    for (let key = 0; key < keyCount; key++) {
      counts[key] = countView.getUint32(key * 4, true);
      values[key] = valueView.getFloat32(key * 4, true);
      valid[key] = counts[key] >= minimumCount && Number.isFinite(values[key]) ? 1 : 0;
    }
    return {values, valid, counts};
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.compiled.destroy();
    this.widening?.destroy();
    this.buffers.forEach(buffer => buffer.destroy());
    this.reservation.release();
  }
}
