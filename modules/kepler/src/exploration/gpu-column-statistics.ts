// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type CommandEncoder, type Device} from '@luma.gl/core';
import {GPUData, type GPUVector} from '@luma-contributors/graph/gpu-data';
import {
  GPUCommandGraph,
  GPUMask,
  type CompiledGPUCommandGraph
} from '@luma-contributors/graph/gpu-core';
import {
  GPUColumnProfile,
  GPU_COLUMN_PROFILE_STATISTIC,
  GPU_COLUMN_PROFILE_STATISTIC_COUNT
} from '@luma-contributors/graph/gpu-dataframe';

/** Largest column the fused GPU profile accepts: row counts are exact in float32 up to 2^24. */
export const MAXIMUM_GPU_COLUMN_STATISTICS_ROWS = 2 ** 24;
export const DEFAULT_GPU_COLUMN_STATISTICS_BINS = 20;

/** Exact moments plus an equal-width histogram of one Float32 column, reduced on the GPU. */
export type GPUColumnStatistics = {
  /** Rows in the population (masked-in before null removal). */
  populationRows: number;
  /** Non-null rows. */
  count: number;
  nullCount: number;
  minimum: number;
  maximum: number;
  sum: number;
  mean: number;
  standardDeviation: number;
  sampleVariance: number;
  /** HyperLogLog estimate (about 1.6% standard error at the default precision). */
  distinctEstimate: number;
  /** Equal-width counts between minimum and maximum of the finite values; the last bin is closed. */
  histogram: Uint32Array;
};

export type GPUColumnStatisticsRoute = {supported: true} | {supported: false; reason: string};

/** Decides, without touching the GPU, whether the fused profile can reduce this column. */
export function planGPUColumnStatistics(props: {
  format: string | undefined;
  rowCount: number;
}): GPUColumnStatisticsRoute {
  if (props.format !== 'float32')
    return {
      supported: false,
      reason: 'GPU statistics reduce Float32 columns; this column format uses the bounded CPU path.'
    };
  if (props.rowCount < 1) return {supported: false, reason: 'The table has no rows.'};
  if (props.rowCount > MAXIMUM_GPU_COLUMN_STATISTICS_ROWS)
    return {
      supported: false,
      reason: 'GPU statistics support at most 16,777,216 rows; narrow the selection.'
    };
  return {supported: true};
}

/** Bytes the profile needs: packed values, combined mask, and small outputs. */
export function getGPUColumnStatisticsByteLength(rowCount: number, histogramBinCount: number) {
  return (
    rowCount * 8 +
    (GPU_COLUMN_PROFILE_STATISTIC_COUNT + 2 + histogramBinCount) * 4 +
    // hyperloglog registers at the default precision of 12, plus slack for transient buffers
    4096 * 4 +
    1024 * 1024
  );
}

/** Copies chunked vector bytes into one packed buffer (explicit packing, owned by the caller). */
function packVector(
  device: Device,
  encoder: CommandEncoder,
  vector: GPUVector,
  id: string
): Buffer {
  const destination = device.createBuffer({
    id,
    byteLength: Math.max(
      4,
      vector.data.reduce((total, data) => total + data.length * data.rowByteLength, 0)
    ),
    usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
  });
  let offset = 0;
  for (const data of vector.data) {
    if (!(data.length * data.rowByteLength)) continue;
    encoder.copyBufferToBuffer({
      sourceBuffer: data.buffer instanceof Buffer ? data.buffer : data.buffer.buffer,
      sourceOffset: data.byteOffset,
      destinationBuffer: destination,
      destinationOffset: offset,
      size: data.length * data.rowByteLength
    });
    offset += data.length * data.rowByteLength;
  }
  return destination;
}

/**
 * One fused GPU profile of a Float32 column. `selection` limits the population; `validity` (zero is
 * null) is folded into the mask, so nulls are counted as population rows minus counted rows.
 * Only the small statistics, count and histogram buffers return to the CPU. The values are packed
 * into one temporary buffer explicitly; source buffers are borrowed and never modified.
 */
export async function readGPUColumnStatistics(props: {
  device: Device;
  values: GPUVector<'float32'>;
  validity?: GPUVector<'uint32'>;
  /** Selected-row mask; omit for the whole table. */
  selection?: GPUVector<'uint32'>;
  /** Population row count (selected count or table rows); used to derive nulls from validity. */
  populationRows: number;
  histogramBinCount?: number;
  isCurrent(): boolean;
}): Promise<GPUColumnStatistics | null> {
  const {device, values} = props;
  const histogramBinCount = props.histogramBinCount ?? DEFAULT_GPU_COLUMN_STATISTICS_BINS;
  const rowCount = values.data.reduce((total, data) => total + data.length, 0);
  const route = planGPUColumnStatistics({format: values.format, rowCount});
  if (!route.supported) throw new Error(route.reason);
  const buffers: Buffer[] = [];
  let compiled: CompiledGPUCommandGraph | undefined;
  try {
    const encoder = device.createCommandEncoder();
    const packed = packVector(device, encoder, values, 'column-statistics-values');
    buffers.push(packed);
    const makeOutput = (id: string, length: number) => {
      const buffer = device.createBuffer({
        id,
        byteLength: length * 4,
        usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
      });
      buffers.push(buffer);
      return buffer;
    };
    const graph = new GPUCommandGraph(device, {id: 'kepler-column-statistics'});
    const valueView = graph.importGPUData(
      'values',
      new GPUData({buffer: packed, format: 'float32', length: rowCount, ownsBuffer: false})
    );
    const maskInputs = [
      props.selection && graph.importGPUVector('selection', props.selection),
      props.validity && graph.importGPUVector('validity', props.validity)
    ].filter(input => input !== undefined);
    let mask;
    if (maskInputs.length) {
      const maskBuffer = makeOutput('column-statistics-mask', rowCount);
      mask = graph.importGPUData(
        'mask',
        new GPUData({buffer: maskBuffer, format: 'uint32', length: rowCount, ownsBuffer: false})
      );
      graph.add(new GPUMask({id: 'column-statistics-mask', inputs: maskInputs, output: mask}));
    }
    const statisticsBuffer = makeOutput('column-statistics', GPU_COLUMN_PROFILE_STATISTIC_COUNT);
    const countsBuffer = makeOutput('column-statistics-counts', 2);
    const histogramBuffer = makeOutput('column-statistics-histogram', histogramBinCount);
    const importOutput = <Format extends 'float32' | 'uint32'>(
      id: string,
      buffer: Buffer,
      format: Format,
      length: number
    ) => graph.importGPUData(id, new GPUData({buffer, format, length, ownsBuffer: false}));
    graph.add(
      new GPUColumnProfile({
        id: 'column-statistics',
        columns: [{values: valueView}],
        ...(mask ? {mask} : {}),
        histogramBinCount,
        output: {
          statistics: importOutput(
            'statistics',
            statisticsBuffer,
            'float32',
            GPU_COLUMN_PROFILE_STATISTIC_COUNT
          ),
          counts: importOutput('counts', countsBuffer, 'uint32', 2),
          histograms: importOutput('histograms', histogramBuffer, 'uint32', histogramBinCount)
        }
      })
    );
    compiled = graph.compile();
    compiled.encode(encoder, {parameters: undefined});
    device.submit(encoder.finish());
    const [statisticsBytes, countsBytes, histogramBytes] = await Promise.all([
      statisticsBuffer.readAsync(0, GPU_COLUMN_PROFILE_STATISTIC_COUNT * 4),
      countsBuffer.readAsync(0, 8),
      histogramBuffer.readAsync(0, histogramBinCount * 4)
    ]);
    if (!props.isCurrent()) return null;
    const statistics = new Float32Array(
      statisticsBytes.buffer,
      statisticsBytes.byteOffset,
      GPU_COLUMN_PROFILE_STATISTIC_COUNT
    );
    const counts = new Uint32Array(countsBytes.buffer, countsBytes.byteOffset, 2);
    const count = counts[0];
    return {
      populationRows: props.populationRows,
      count,
      nullCount: Math.max(0, props.populationRows - count),
      minimum: statistics[GPU_COLUMN_PROFILE_STATISTIC.minimum],
      maximum: statistics[GPU_COLUMN_PROFILE_STATISTIC.maximum],
      sum: statistics[GPU_COLUMN_PROFILE_STATISTIC.sum],
      mean: statistics[GPU_COLUMN_PROFILE_STATISTIC.mean],
      standardDeviation: statistics[GPU_COLUMN_PROFILE_STATISTIC.standardDeviation],
      sampleVariance: statistics[GPU_COLUMN_PROFILE_STATISTIC.sampleVariance],
      distinctEstimate: statistics[GPU_COLUMN_PROFILE_STATISTIC.distinctEstimate],
      histogram: new Uint32Array(histogramBytes.slice().buffer)
    };
  } finally {
    compiled?.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
}
