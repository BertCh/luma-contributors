// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import * as arrow from 'apache-arrow';
import {Buffer, type Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {KeplerComputeSession} from '../src/session/compute-session';
import {AnalysisRuntime} from '../../../examples/deck/kepler-compute/src/analysis-runtime';
import {requireTestDevice} from '../../analysis/test/utils/require-test-device';

function createFixture(
  device: Device,
  table = arrow.tableFromArrays({value: new Float32Array([1, 2, 3, 4])})
) {
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 64 * 1024 * 1024});
  session.importArrowDataset({
    id: 'observations',
    table
  });
  const explorer = session.openExplorer('observations', {
    workingMemoryBudgetBytes: 8 * 1024 * 1024,
    dimensions: [{id: 'value', kind: 'range', column: 'value'}]
  });
  let updates = 0;
  const analysis = new AnalysisRuntime({
    session,
    explorer,
    requestUpdate: () => {
      updates++;
    }
  });
  return {
    session,
    explorer,
    analysis,
    get updates() {
      return updates;
    },
    async render() {
      const encoder = device.createCommandEncoder();
      explorer.encode(encoder);
      analysis.encode(encoder);
      device.submit(encoder.finish());
      await analysis.readSummary();
    },
    async destroy() {
      const fence = device.createFence();
      try {
        await analysis.destroy(fence.signaled);
        await session.destroy(fence.signaled);
      } finally {
        fence.destroy();
      }
    }
  };
}

it('the host analysis renders resident z-score colors and recomputes only the chosen population', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createFixture(device);
  try {
    fixture.analysis.setMode('z-score');
    await fixture.render();
    expect(fixture.analysis.summary?.count).toBe(4);
    expect(fixture.analysis.summary?.mean).toBeCloseTo(0, 5);
    const last = await fixture.analysis.readRow(3);
    expect(last?.value).toBeCloseTo(1.5 / Math.sqrt(5 / 3), 5);
    expect(last?.sourceRevision).toBe(1);
    const colors = fixture.analysis.colors.data[0].buffer;
    expect(colors).toBeInstanceOf(Buffer);
    if (!(colors instanceof Buffer)) throw new Error('Expected resident color buffer');
    const colorBytes = await colors.readAsync(0, 16);
    expect(Array.from(colorBytes.slice(0, 4))).not.toEqual(Array.from(colorBytes.slice(12, 16)));
    expect(colorBytes[3]).toBe(255);
    const fullRecipe = fixture.analysis.serializeRecipe();
    const transferred = fixture.analysis.transferredReadbackBytes;
    fixture.explorer.setRange('value', 3, 4);
    await fixture.render();
    expect(fixture.analysis.summary?.count).toBe(4);
    expect(fixture.analysis.transferredReadbackBytes).toBe(transferred);
    fixture.analysis.setPopulation('selected');
    await fixture.render();
    expect(fixture.analysis.summary?.count).toBe(2);
    expect((await fixture.analysis.readRow(0))?.value).toBeNaN();
    expect((await fixture.analysis.readRow(3))?.value).toBeCloseTo(Math.SQRT1_2, 5);
    const selectedRecipe = fixture.analysis.serializeRecipe();
    fixture.explorer.setRange('value', 1, 2);
    expect(await fixture.analysis.readRow(3)).toBeNull();
    expect(() => fixture.analysis.restoreRecipe(selectedRecipe)).toThrow('selection');
    await fixture.render();
    expect((await fixture.analysis.readRow(1))?.value).toBeCloseTo(Math.SQRT1_2, 5);
    fixture.analysis.restoreRecipe(fullRecipe);
    await fixture.render();
    expect(fixture.analysis.summary?.count).toBe(4);
    expect(fixture.updates).toBe(3);
    expect(fixture.analysis.transferredReadbackBytes).toBeLessThan(512);
  } finally {
    await fixture.destroy();
  }
  expect(fixture.session.registry.memoryUsage.totalBytes).toBe(0);
});

it('the host classification uses GPU breaks for colors, bounded diagnostics and saved recipe restoration', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createFixture(device);
  try {
    fixture.analysis.setMode('quantile');
    fixture.analysis.setClassCount(2);
    await fixture.render();
    expect(fixture.analysis.summary?.breaks).toEqual([1, 2.5, 4]);
    expect(fixture.analysis.summary?.classCounts).toEqual([2, 2]);
    expect((await fixture.analysis.readRow(0))?.value).toBe(0);
    expect((await fixture.analysis.readRow(3))?.value).toBe(1);
    const saved = fixture.analysis.serializeRecipe();
    fixture.analysis.setMode('equal-interval');
    fixture.analysis.setClassCount(3);
    await fixture.render();
    expect(fixture.analysis.summary?.breaks).toEqual([1, 2, 3, 4]);
    fixture.analysis.restoreRecipe(saved);
    await fixture.render();
    expect(fixture.analysis.settings.mode).toBe('quantile');
    expect(fixture.analysis.settings.classCount).toBe(2);
    fixture.session.importArrowDataset({
      id: 'observations',
      table: arrow.tableFromArrays({value: new Float32Array([5, 6, 7, 8])})
    });
    expect(fixture.analysis.active).toBe(false);
    expect(await fixture.analysis.readRow(0)).toBeNull();
    expect(() => fixture.analysis.restoreRecipe(saved)).toThrow('stale');
  } finally {
    await fixture.destroy();
  }
});

it('selected analysis intersects source validity even when the nullable field dimension is inactive', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const table = new arrow.Table({
    value: arrow.vectorFromArray([null, 10, 20, null], new arrow.Float32())
  });
  const fixture = createFixture(device, table);
  try {
    fixture.analysis.setMode('z-score');
    await fixture.render();
    expect(fixture.analysis.summary?.count).toBe(2);
    expect((await fixture.analysis.readRow(1))?.value).toBeCloseTo(-Math.SQRT1_2, 5);
    fixture.analysis.setPopulation('selected');
    await fixture.render();
    expect(fixture.analysis.summary?.count).toBe(2);
    expect((await fixture.analysis.readRow(0))?.value).toBeNaN();
    expect((await fixture.analysis.readRow(2))?.value).toBeCloseTo(Math.SQRT1_2, 5);
    fixture.analysis.setMode('quantile');
    fixture.analysis.setClassCount(2);
    await fixture.render();
    expect(fixture.analysis.summary?.breaks).toEqual([10, 15, 20]);
    expect(fixture.analysis.summary?.classCounts).toEqual([1, 1]);
    expect(fixture.analysis.summary?.count).toBe(2);
    const colors = fixture.analysis.colors.data[0].buffer;
    if (!(colors instanceof Buffer)) throw new Error('Expected resident color buffer');
    expect((await colors.readAsync(0, 4))[3]).toBe(0);
  } finally {
    await fixture.destroy();
  }
});
