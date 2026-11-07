// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import * as arrow from 'apache-arrow';
import {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {KeplerComputeSession} from '../src/session/compute-session';
import {AnalysisRuntime} from '../../../examples/deck/kepler-compute/src/analysis-runtime';
import {requireTestDevice} from '../../graph/test/utils/require-test-device';

test('excess risk pools selected source batches and excludes missing or nonpositive populations', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 64 * 1024 * 1024});
  const first = new arrow.Table({
    events: arrow.vectorFromArray([10, 20, null], new arrow.Float32()),
    population: arrow.vectorFromArray([100, null, 300], new arrow.Float32()),
    row: arrow.makeVector(new Float32Array([0, 1, 2]))
  });
  const second = new arrow.Table({
    events: arrow.vectorFromArray([40, 5, 0], new arrow.Float32()),
    population: arrow.vectorFromArray([200, 0, 100], new arrow.Float32()),
    row: arrow.makeVector(new Float32Array([3, 4, 5]))
  });
  session.importArrowDataset({id: 'rates', table: first.concat(second)});
  const explorer = session.openExplorer('rates', {
    workingMemoryBudgetBytes: 8 * 1024 * 1024,
    dimensions: [{id: 'row', column: 'row', kind: 'range'}]
  });
  const analysis = new AnalysisRuntime({session, explorer, requestUpdate: () => {}});
  const render = async () => {
    const encoder = device.createCommandEncoder();
    explorer.encode(encoder);
    analysis.encode(encoder);
    device.submit(encoder.finish());
    await analysis.readSummary();
  };
  try {
    analysis.setMode('excess-risk');
    await render();
    expect(analysis.summary?.count).toBe(3);
    expect(analysis.summary?.denominatorColumn).toBe('population');
    expect((await analysis.readRow(0))?.value).toBeCloseTo(0.8, 5);
    expect((await analysis.readRow(3))?.value).toBeCloseTo(1.6, 5);
    expect((await analysis.readRow(5))?.value).toBe(0);
    for (const row of [1, 2, 4]) expect((await analysis.readRow(row))?.value).toBeNaN();
    expect((await analysis.readRow(0))?.kind).toBe('excess-risk');
    expect(analysis.colors.data.map(chunk => chunk.length)).toEqual([3, 3]);
    const colors = analysis.colors.data[0].buffer;
    if (!(colors instanceof Buffer)) throw new Error('Expected analysis-owned buffer');
    const bytes = await colors.readAsync(0, 24);
    for (const row of [1, 2, 4]) expect(bytes[row * 4 + 3]).toBe(0);
    analysis.setPopulation('selected');
    explorer.setRange('row', 3, 5);
    await render();
    expect(analysis.summary?.count).toBe(2);
    expect((await analysis.readRow(0))?.value).toBeNaN();
    expect((await analysis.readRow(3))?.value).toBeCloseTo(1.5, 5);
    expect((await analysis.readRow(5))?.value).toBe(0);
    explorer.setRange('row', 5, 5);
    await render();
    expect(analysis.summary?.count).toBe(0);
    expect((await analysis.readRow(5))?.value).toBeNaN();
  } finally {
    const fence = device.createFence();
    try {
      await analysis.destroy(fence.signaled);
      await session.destroy(fence.signaled);
    } finally {
      fence.destroy();
    }
  }
  expect(session.registry.memoryUsage.totalBytes).toBe(0);
});
