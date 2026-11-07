// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {GPUVector, type GPUData} from '@luma.gl/graph/gpu-data';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUDataFrame,
  column,
  literal,
  type CompiledGPUDataFrameGroupedAggregation,
  type GPUDataFrameAggregationValue,
  type GPUDataFrameQueryParameters
} from '@luma.gl/graph/gpu-dataframe';
import {GPUTable} from '@luma.gl/graph/gpu-tables';
import type {
  KeplerArrowDataset,
  KeplerComputeSession,
  KeplerDatasetLease,
  KeplerMemoryReservation
} from '@luma-contributors/kepler';
import type {CommandEncoder} from '@luma.gl/core';
import {ResidentPlotOrderMeasure} from './resident-plot-order-measure';
import {
  ResidentPlotMeasureWidening,
  isWidenableResidentPlotMeasure
} from './resident-plot-measure-widening';

/** Native y-axis aggregations that a grouped GPU reduction represents exactly. */
export const RESIDENT_PLOT_MEASURE_OPERATIONS = {
  average: 'mean',
  sum: 'sum',
  minimum: 'min',
  maximum: 'max',
  median: 'median',
  // Native `stdev` is d3.deviation: the square root of the sample (n - 1) variance.
  stdev: 'standardDeviation'
} as const;
export type ResidentPlotMeasureOperation =
  (typeof RESIDENT_PLOT_MEASURE_OPERATIONS)[keyof typeof RESIDENT_PLOT_MEASURE_OPERATIONS];
/** Operations that need ordered or centered per-bin statistics rather than a streaming reduction. */
export type ResidentPlotOrderOperation = Extract<
  ResidentPlotMeasureOperation,
  'median' | 'standardDeviation'
>;
export type ResidentPlotStreamingOperation = Exclude<
  ResidentPlotMeasureOperation,
  ResidentPlotOrderOperation
>;

type MeasureColumns = {category: 'uint32'; selected: 'uint32'; value: 'float32'};
type MeasureQuery = CompiledGPUDataFrameGroupedAggregation<{
  category: 'uint32';
  result: 'float32';
  count: 'uint32';
}>;

/** Per-bin aggregate. `valid` is false when no selected row had a valid measured value. */
export type ResidentPlotMeasureSummary = {
  values: Float32Array;
  valid: Uint8Array;
  counts: Uint32Array;
};

/**
 * Grouped y-axis reduction over a plot population. Keys are the plot's own bin classification
 * (one dense key per source row, out-of-range for rows outside the plot domain), so the measure
 * and the histogram can never disagree about bin membership. Borrowed vectors stay owned by the
 * caller; this object owns only its query resources.
 */
export class ResidentPlotMeasure {
  private readonly query: MeasureQuery;
  private readonly reservation: KeplerMemoryReservation;
  private readonly widening?: ResidentPlotMeasureWidening;

  constructor(
    private readonly props: {
      session: KeplerComputeSession;
      lease: KeplerDatasetLease<KeplerArrowDataset>;
      keys: GPUVector<'uint32'>;
      mask: GPUVector<'uint32'>;
      binCount: number;
      column: string;
      operation: ResidentPlotStreamingOperation;
      workingMemoryBudgetBytes: number;
    }
  ) {
    const source = props.lease.dataset.value;
    const measured = source.columns[props.column];
    // Float32 is exact; Int32 and Uint32 widen on the GPU behind a +/-2^24 range check.
    const needsWidening = isWidenableResidentPlotMeasure(
      measured,
      Boolean(source.prepared.dictionaries[props.column])
    );
    if (!measured || (measured.format !== 'float32' && !needsWidening))
      throw new Error('Resident line-chart measures require a numeric field');
    const estimatedBytes =
      65536 +
      props.keys.length * 160 +
      props.keys.data.length * 8192 +
      props.binCount * 128 +
      (needsWidening ? measured.length * 4 + measured.data.length * 8192 : 0);
    if (estimatedBytes > props.workingMemoryBudgetBytes)
      throw new Error('Line-chart measure exceeds its working memory budget');
    const reservation = props.session.registry.reserveMemory(estimatedBytes, 'gpu', 'scratch');
    let frame: GPUDataFrame<MeasureColumns> | undefined;
    let query: MeasureQuery | undefined;
    let widening: ResidentPlotMeasureWidening | undefined;
    try {
      if (needsWidening) widening = new ResidentPlotMeasureWidening(props.session.device, measured);
      const validity: {value?: GPUVector<'uint32'>} = {};
      const measureValidity = source.analytics.validity[props.column];
      if (measureValidity) validity.value = measureValidity;
      const table = new GPUTable<MeasureColumns>({
        vectors: {
          category: new GPUVector({
            name: 'category',
            type: 'data',
            format: 'uint32',
            data: props.keys.data,
            ownsData: false
          }),
          selected: new GPUVector({
            name: 'selected',
            type: 'data',
            format: 'uint32',
            data: props.mask.data,
            ownsData: false
          }),
          value: new GPUVector({
            name: 'value',
            type: 'data',
            format: 'float32',
            data: (widening?.vector ?? (measured as GPUVector<'float32'>)).data,
            ownsData: false
          })
        }
      });
      for (const schema of [table.schema, ...table.batches.map(batch => batch.schema)])
        for (const field of schema.fields)
          field.nullable = field.name === 'value' && Boolean(validity.value);
      frame = new GPUDataFrame({table, validity});
      const graph = new GPUCommandGraph<GPUDataFrameQueryParameters>(props.session.device, {
        id: 'kepler-plot-measure'
      });
      const result: GPUDataFrameAggregationValue<MeasureColumns, keyof MeasureColumns> =
        props.operation === 'sum'
          ? {sum: 'value'}
          : props.operation === 'mean'
            ? {mean: 'value'}
            : props.operation === 'min'
              ? {min: 'value'}
              : {max: 'value'};
      query = frame
        // Rows outside the plot domain carry an out-of-range key and never reach a bin.
        .filter(
          column('selected')
            .notEqual(literal(0))
            .and(column('category').lessThan(literal(props.binCount)))
        )
        .groupBy('category', {groupCount: props.binCount})
        .aggregate({result, count: 'count'})
        .compile(graph);
      const buffers = new Set<Buffer>();
      for (const vector of [
        ...Object.values(query.table.gpuVectors),
        ...Object.values(query.validity),
        query.selectionMask,
        query.rowIndices,
        query.selectedCounts
      ])
        if (vector) for (const data of vector.data) buffers.add(getBuffer(data));
      const accountedBytes =
        (widening?.byteLength ?? 0) +
        query.compilationStats.physicalTransientResourceBytes +
        [...buffers].reduce((total, buffer) => total + buffer.byteLength, 0);
      if (accountedBytes > estimatedBytes)
        throw new Error('Line-chart measure exceeds its reserved working memory');
      this.query = query;
      this.widening = widening;
      this.reservation = reservation;
    } catch (error) {
      widening?.destroy();
      query?.destroy();
      reservation.release();
      throw error;
    } finally {
      frame?.destroy();
    }
  }

  /** Encode after the key classification pass on the same command encoder. */
  encode(encoder: CommandEncoder): void {
    this.widening?.encode(encoder);
    this.query.encode(encoder);
  }

  async readSummary(): Promise<ResidentPlotMeasureSummary> {
    const binCount = this.props.binCount;
    await this.widening?.assertExactRange();
    const result = this.query.table.gpuVectors['result'];
    const count = this.query.table.gpuVectors['count'];
    const validity = this.query.validity.result;
    const byteLength = binCount * 4;
    const [values, counts, valid] = await Promise.all([
      getBuffer(result.data[0]).readAsync(result.data[0].byteOffset, byteLength),
      getBuffer(count.data[0]).readAsync(count.data[0].byteOffset, byteLength),
      validity
        ? getBuffer(validity.data[0]).readAsync(validity.data[0].byteOffset, byteLength)
        : undefined
    ]);
    const valueView = new DataView(values.buffer, values.byteOffset, byteLength);
    const countView = new DataView(counts.buffer, counts.byteOffset, byteLength);
    const validView = valid ? new DataView(valid.buffer, valid.byteOffset, byteLength) : undefined;
    return {
      values: Float32Array.from({length: binCount}, (_, index) =>
        valueView.getFloat32(index * 4, true)
      ),
      counts: Uint32Array.from({length: binCount}, (_, index) =>
        countView.getUint32(index * 4, true)
      ),
      valid: Uint8Array.from({length: binCount}, (_, index) =>
        validView ? (validView.getUint32(index * 4, true) ? 1 : 0) : 1
      )
    };
  }

  get readbackByteLength(): number {
    return this.props.binCount * 12;
  }

  destroy(): void {
    this.widening?.destroy();
    this.query.destroy();
    this.reservation.release();
  }
}

/** Working memory a y-axis measure may reserve for a source: the larger streaming bound. */
export function getResidentPlotMeasureBudgetBytes(rowCount: number, chunkCount: number): number {
  return 1048576 + rowCount * 160 + chunkCount * 8192;
}

export type ResidentPlotMeasureRunner = {
  encode(encoder: CommandEncoder): void;
  readSummary(): Promise<ResidentPlotMeasureSummary>;
  readonly readbackByteLength: number;
  destroy(): void;
};

/** Streaming reductions use the grouped dataframe; median and deviation use exact ordered statistics. */
export function makeResidentPlotMeasure(props: {
  session: KeplerComputeSession;
  lease: KeplerDatasetLease<KeplerArrowDataset>;
  keys: GPUVector<'uint32'>;
  mask: GPUVector<'uint32'>;
  binCount: number;
  column: string;
  operation: ResidentPlotMeasureOperation;
  workingMemoryBudgetBytes: number;
}): ResidentPlotMeasureRunner {
  if (props.operation === 'median' || props.operation === 'standardDeviation')
    return new ResidentPlotOrderMeasure({
      ...props,
      keyCount: props.binCount,
      operation: props.operation
    });
  return new ResidentPlotMeasure({...props, operation: props.operation});
}

function getBuffer(data: GPUData): Buffer {
  return data.buffer instanceof Buffer ? data.buffer : data.buffer.buffer;
}
