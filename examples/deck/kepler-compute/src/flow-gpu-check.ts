// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {Buffer, type Device} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma-contributors/graph/gpu-data';
import {FlowReferenceAggregator, type FlowPairs} from './flow-gpu-aggregation';

export type FlowGpuCheckColumns = {
  sourceLongitude: number[];
  sourceLatitude: number[];
  targetLongitude: number[];
  targetLatitude: number[];
  count: number[];
  /** Selection mask words (1 selected, 0 not); omitted checks only the unfiltered pass. */
  selected?: number[];
  /** Number of Arrow record batches the table is split into. */
  batchCount?: number;
};

type FlowPairsSummary = {
  pairCount: number;
  sourceIds: number[];
  targetIds: number[];
  counts: number[];
  firstRows: number[];
};

function summarizePairs(pairs: FlowPairs): FlowPairsSummary {
  return {
    pairCount: pairs.sourceIds.length,
    sourceIds: [...pairs.sourceIds],
    targetIds: [...pairs.targetIds],
    counts: [...pairs.counts],
    firstRows: [...pairs.firstRows]
  };
}

/** Test-only: runs the GPU Flow aggregator and the CPU reference on one table and returns both. */
export async function runFlowGpuCheck(device: Device, columns: FlowGpuCheckColumns) {
  const rowCount = columns.count.length;
  const names = [
    'sourceLongitude',
    'sourceLatitude',
    'targetLongitude',
    'targetLatitude',
    'count'
  ] as const;
  const batchCount = Math.max(1, columns.batchCount ?? 1);
  const batchSize = Math.ceil(rowCount / batchCount);
  const batches: arrow.RecordBatch[] = [];
  for (let start = 0; start < rowCount; start += batchSize) {
    const batch = new arrow.Table(
      Object.fromEntries(
        names.map(name => [
          name,
          arrow.vectorFromArray(columns[name].slice(start, start + batchSize), new arrow.Float64())
        ])
      )
    );
    batches.push(...batch.batches);
  }
  const table = new arrow.Table(batches);
  const roles = {
    targetLongitude: 'targetLongitude',
    targetLatitude: 'targetLatitude',
    count: 'count'
  };
  const coordinates = {longitude: 'sourceLongitude', latitude: 'sourceLatitude'};
  const {createFlowGpuAggregator} = await import('./flow-gpu-pipeline');
  const selected = columns.selected;
  const reference = new FlowReferenceAggregator(
    {
      sourceLongitude: table.getChild('sourceLongitude'),
      sourceLatitude: table.getChild('sourceLatitude'),
      targetLongitude: table.getChild('targetLongitude'),
      targetLatitude: table.getChild('targetLatitude'),
      count: table.getChild('count')
    },
    rowCount,
    () => Uint32Array.from(selected!)
  );
  const gpu = await createFlowGpuAggregator(device, table, {coordinates, roles});
  let maskBuffer: Buffer | undefined;
  try {
    const result: Record<string, unknown> & {
      gpuSelected?: FlowPairsSummary;
      referenceSelected?: FlowPairsSummary;
      gpuSelectedRepeat?: FlowPairsSummary;
    } = {
      rowCount,
      gpuFirstEndpoints: [...gpu.firstEndpoints],
      referenceFirstEndpoints: [...reference.firstEndpoints],
      gpuBounds: gpu.bounds,
      referenceBounds: reference.bounds,
      gpuByteLength: gpu.gpuByteLength,
      gpuUnfiltered: summarizePairs(await gpu.aggregatePairs()),
      referenceUnfiltered: summarizePairs(await reference.aggregatePairs())
    };
    if (selected) {
      const words = Uint32Array.from(selected);
      maskBuffer = device.createBuffer({
        data: words,
        usage: Buffer.STORAGE | Buffer.COPY_DST | Buffer.COPY_SRC
      });
      const mask = new GPUVector({
        type: 'data',
        name: 'flow-check-mask',
        format: 'uint32',
        data: [
          new GPUData({buffer: maskBuffer, format: 'uint32', length: rowCount, ownsBuffer: false})
        ],
        ownsData: false
      });
      result.gpuSelected = summarizePairs(await gpu.aggregatePairs(mask));
      result.referenceSelected = summarizePairs(await reference.aggregatePairs(mask));
      // Run twice to catch stale or destroyed resources after a second aggregation.
      result.gpuSelectedRepeat = summarizePairs(await gpu.aggregatePairs(mask));
    }
    return result;
  } finally {
    gpu.destroy();
    maskBuffer?.destroy();
  }
}
