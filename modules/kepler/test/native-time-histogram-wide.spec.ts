// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import * as arrow from 'apache-arrow';
import {Buffer} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma-contributors/graph/gpu-data';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {requireTestDevice} from '../../graph/test/utils/require-test-device';
import {KeplerComputeSession} from '../src/session/compute-session';
import {
  ResidentTimeHistogram,
  makeResidentTimeHistogramPlan
} from '../../../examples/deck/kepler-compute/src/resident-time-histogram';

const BIN_COUNT = 1000;
const ROWS_PER_BIN = 5;

it('native time histogram classifies more than 256 intervals exactly on the GPU', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 64 * 1024 * 1024});
  const origin = Date.UTC(2024, 0, 1);
  const rowCount = BIN_COUNT * ROWS_PER_BIN;
  // Row i sits at i milliseconds so every 1-millisecond bin receives exactly ROWS_PER_BIN... of the
  // 5-millisecond grid: use a 5-millisecond-spaced thresholds list below.
  const plan = makeResidentTimeHistogramPlan([origin, origin + rowCount - 1], '1-millisecond');
  expect(plan.thresholds.length).toBeGreaterThan(256);
  const thresholds = Array.from(
    {length: BIN_COUNT + 1},
    (_, index) => origin + index * ROWS_PER_BIN
  );
  const table = new arrow.Table({
    time: arrow.vectorFromArray(
      Array.from({length: rowCount}, (_, row) => new Date(origin + row)),
      new arrow.TimestampMillisecond()
    )
  });
  const dataset = session.importArrowDataset({id: 'wide-time-histogram', table});
  const source = dataset.value.timeColumns.time;
  const selected = new GPUVector({
    name: 'selected',
    type: 'data',
    format: 'uint32',
    ownsData: true,
    data: source.timestamps.data.map(
      chunk =>
        new GPUData({
          format: 'uint32',
          length: chunk.length,
          ownsBuffer: true,
          buffer: device.createBuffer({
            data: new Uint32Array(Math.max(1, chunk.length)).fill(1),
            usage: Buffer.STORAGE | Buffer.COPY_DST
          })
        })
    )
  });
  const histogram = new ResidentTimeHistogram({
    session,
    datasetId: dataset.id,
    field: 'time',
    mask: selected,
    plan: {interval: '5-millisecond', thresholds},
    workingMemoryBudgetBytes: 4 * 1024 * 1024
  });
  try {
    const encoder = device.createCommandEncoder();
    histogram.encode(encoder);
    device.submit(encoder.finish());
    const counts = Array.from((await histogram.readSummary())!.histograms['native-plot']);
    expect(counts.length).toBe(BIN_COUNT + 1);
    expect(counts.slice(0, BIN_COUNT).every(count => count === ROWS_PER_BIN)).toBe(true);
    expect(counts[BIN_COUNT]).toBe(0);
  } finally {
    const fence = device.createFence();
    await histogram.close(fence.signaled);
    fence.destroy();
    selected.destroy();
    await session.destroy();
  }
});
