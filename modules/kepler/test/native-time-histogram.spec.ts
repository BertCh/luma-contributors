// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it, vi} from 'vitest';
import * as arrow from 'apache-arrow';
import {Buffer} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {requireTestDevice} from '../../analysis/test/utils/require-test-device';
import {KeplerComputeSession} from '../src/session/compute-session';
import {ResidentTimeHistogram} from '../../../examples/deck/kepler-compute/src/resident-time-histogram';

it('native time histogram preserves exact submillisecond boundaries, chunks, selection, final edge and bounded stale reads', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 32 * 1024 * 1024});
  const origin = Date.UTC(2300, 0, 1);
  const ticks = BigInt(origin) * 1000n;
  const parts = [
    [ticks - 1n, ticks, ticks + 1n],
    [],
    [ticks + 999999n, ticks + 1000000n, ticks + 2000000n, ticks, 0x7fffffffffffffffn]
  ];
  const table = new arrow.Table(
    parts.flatMap(
      (values, index) =>
        new arrow.Table({
          time: arrow.makeVector(
            arrow.makeData({
              type: new arrow.TimestampMicrosecond(),
              data: BigInt64Array.from(values),
              length: values.length,
              nullBitmap: new Uint8Array([index === 2 ? 0b10111 : 0xff]),
              nullCount: index === 2 ? 1 : 0
            })
          )
        }).batches
    )
  );
  const dataset = session.importArrowDataset({id: 'time-histogram-source', table});
  const source = dataset.value.timeColumns.time;
  const selected = new GPUVector({
    name: 'selected',
    type: 'data',
    format: 'uint32',
    ownsData: true,
    data: source.timestamps.data.map(
      (chunk, index) =>
        new GPUData({
          format: 'uint32',
          length: chunk.length,
          ownsBuffer: true,
          buffer: device.createBuffer({
            data: Uint32Array.from({length: Math.max(1, chunk.length)}, (_, row) =>
              index === 0 && row === 2 ? 0 : 1
            ),
            usage: Buffer.STORAGE | Buffer.COPY_DST
          })
        })
    )
  });
  const spies = [...source.timestamps.data, ...source.validity.data, ...selected.data].map(chunk =>
    vi.spyOn(chunk.buffer, 'readAsync')
  );
  const histogram = new ResidentTimeHistogram({
    session,
    datasetId: dataset.id,
    field: 'time',
    mask: selected,
    plan: {interval: '1-second', thresholds: [origin, origin + 1000, origin + 2000]},
    workingMemoryBudgetBytes: 1024 * 1024
  });
  const submit = () => {
    const encoder = device.createCommandEncoder();
    histogram.encode(encoder);
    device.submit(encoder.finish());
  };
  try {
    expect(await histogram.readSummary()).toBeNull();
    submit();
    expect(Array.from((await histogram.readSummary())!.histograms['native-plot'])).toEqual([
      2, 1, 1
    ]);
    expect(histogram.readbackByteLength).toBe(12);
    expect(spies.every(spy => spy.mock.calls.length === 0)).toBe(true);
    const pending = histogram.readSummary();
    histogram.invalidate();
    expect(await pending).toBeNull();
    expect(await histogram.readSummary()).toBeNull();
    for (const chunk of selected.data)
      chunk.buffer.write(new Uint32Array(Math.max(1, chunk.length)));
    histogram.invalidate();
    submit();
    expect(Array.from((await histogram.readSummary())!.histograms['native-plot'])).toEqual([
      0, 0, 0
    ]);
    expect(source.timestamps.data.map(chunk => chunk.length)).toEqual(
      parts.map(part => part.length)
    );
    expect(
      () =>
        new ResidentTimeHistogram({
          session,
          datasetId: dataset.id,
          field: 'time',
          mask: selected,
          plan: {interval: '1-second', thresholds: [origin, origin + 1000]},
          workingMemoryBudgetBytes: 1
        })
    ).toThrow('budget');
    session.importArrowDataset({id: dataset.id, table});
    expect(await histogram.readSummary()).toBeNull();
    const staleEncoder = device.createCommandEncoder();
    expect(() => histogram.encode(staleEncoder)).toThrow('stale');
    staleEncoder.destroy();
  } finally {
    const fence = device.createFence();
    await histogram.close(fence.signaled);
    fence.destroy();
    expect(session.registry.memoryUsage.reservedBytes.gpu).toBe(0);
    spies.forEach(spy => spy.mockRestore());
    selected.destroy();
    await session.destroy();
  }
});
