// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import * as arrow from 'apache-arrow';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {KeplerComputeSession, type KeplerIDWSettings} from '../src';
import {AnalysisRuntime} from '../../../examples/deck/kepler-compute/src/analysis-runtime';
import {requireTestDevice} from '../../graph/test/utils/require-test-device';

const settings: KeplerIDWSettings = {
  longitudeColumn: 'longitude',
  latitudeColumn: 'latitude',
  coordinateSpace: 'longitude-latitude',
  metric: 'great-circle',
  units: 'meters',
  grid: {bounds: [0, 0, 2, 2], width: 2, height: 2},
  searchRadius: 100000,
  power: 2,
  neighborCount: 4,
  minimumNeighborCount: 3,
  taper: 'none'
};

test('interpolation publishes a separate north-up grid with exact-hit, source validity, selection and replay semantics', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 64 * 1024 * 1024});
  const first = arrow.tableFromArrays({
    longitude: new Float32Array([0.5, 0.5]),
    latitude: new Float32Array([1.5, 1.5]),
    value: new Float32Array([Infinity, 10]),
    row: new Float32Array([0, 1])
  });
  const second = arrow.tableFromArrays({
    longitude: new Float32Array([1.5, 0.5, 1.5, 181]),
    latitude: new Float32Array([1.5, 0.5, 0.5, 1.5]),
    value: new Float32Array([20, 30, 40, 999]),
    row: new Float32Array([2, 3, 4, 5])
  });
  const dataset = session.importArrowDataset({id: 'idw-source', table: first.concat(second)});
  const sourceChunks = [...dataset.value.columns.value.data];
  const explorer = session.openExplorer(dataset.id, {
    workingMemoryBudgetBytes: 8 * 1024 * 1024,
    dimensions: [{id: 'row', column: 'row', kind: 'range'}]
  });
  const analysis = new AnalysisRuntime({session, explorer, requestUpdate() {}});
  const exportsToDestroy: Array<{destroy(): void}> = [];
  const render = async () => {
    const encoder = device.createCommandEncoder();
    explorer.encode(encoder);
    analysis.encode(encoder);
    device.submit(encoder.finish());
    await analysis.readSummary();
    expect(analysis.error).toBe('');
  };
  try {
    analysis.configureInterpolation(settings);
    const recipe = analysis.serializeRecipe();
    await render();
    expect(analysis.active).toBe(false);
    expect(analysis.getTableColumns()).toBeUndefined();
    expect(analysis.summary).toMatchObject({
      operation: 'idw-points',
      count: 4,
      nullCount: 0,
      minimum: 10,
      maximum: 40,
      mean: 25,
      interpolation: {width: 2, height: 2, validSourceCount: 4, cellCount: 4}
    });
    expect(await analysis.readInterpolationPage(0, 4)).toEqual([
      {cellIndex: 0, longitude: 0.5, latitude: 1.5, value: 10, neighbors: 1},
      {cellIndex: 1, longitude: 1.5, latitude: 1.5, value: 20, neighbors: 1},
      {cellIndex: 2, longitude: 0.5, latitude: 0.5, value: 30, neighbors: 1},
      {cellIndex: 3, longitude: 1.5, latitude: 0.5, value: 40, neighbors: 1}
    ]);
    expect(analysis.getInterpolationLayer()?.props.bounds).toEqual([0, 0, 2, 2]);
    await expect(analysis.readInterpolationPage(0, 129)).rejects.toThrow('128');
    const cpuReservationBeforeExport = session.registry.memoryUsage.reservedBytes.cpu;
    const exportedGrid = await analysis.exportInterpolationGeoTIFF();
    exportsToDestroy.push(exportedGrid);
    expect(exportedGrid.readbackBytes).toBe(2 * 2 * Float32Array.BYTES_PER_ELEMENT);
    expect(Array.from(exportedGrid.data.subarray(0, 2))).toEqual([0x49, 0x49]);
    expect(exportedGrid.filename).toMatch(/\.tiff?$/i);
    expect(exportedGrid.isCurrent()).toBe(true);
    const retainedCpuReservation = session.registry.memoryUsage.reservedBytes.cpu;
    expect(retainedCpuReservation).toBeGreaterThan(cpuReservationBeforeExport);
    analysis.setPopulation('selected');
    explorer.setRange('row', 1, 2);
    expect(analysis.getInterpolationLayer()).toBeNull();
    expect(exportedGrid.isCurrent()).toBe(false);
    // A stale completed export still owns its CPU bytes until the caller releases the snapshot.
    expect(session.registry.memoryUsage.reservedBytes.cpu).toBe(retainedCpuReservation);
    exportedGrid.destroy();
    expect(session.registry.memoryUsage.reservedBytes.cpu).toBe(cpuReservationBeforeExport);
    exportedGrid.destroy();
    expect(session.registry.memoryUsage.reservedBytes.cpu).toBe(cpuReservationBeforeExport);
    await render();
    expect(analysis.summary).toMatchObject({
      count: 2,
      nullCount: 2,
      interpolation: {validSourceCount: 2}
    });
    expect(
      (await analysis.readInterpolationPage(2, 2)).map(cell => [cell.value, cell.neighbors])
    ).toEqual([
      [null, 0],
      [null, 0]
    ]);
    analysis.setMode('off');
    expect(analysis.getInterpolationLayer()).toBeNull();
    analysis.restoreRecipe(recipe);
    await render();
    expect((await analysis.readInterpolationPage(0, 4)).map(cell => cell.value)).toEqual([
      10, 20, 30, 40
    ]);
    expect(dataset.value.columns.value.data).toEqual(sourceChunks);
    expect(sourceChunks.map(chunk => chunk.length)).toEqual([2, 4]);
    const cpuReservationBeforeCancellation = session.registry.memoryUsage.reservedBytes.cpu;
    const pendingExport = analysis.exportInterpolationGeoTIFF();
    const cancelledExport = expect(pendingExport).rejects.toThrow();
    // Invalidate synchronously while the GPU buffer read is still asynchronous.
    analysis.setMode('off');
    await cancelledExport;
    expect(session.registry.memoryUsage.reservedBytes.cpu).toBe(cpuReservationBeforeCancellation);
  } finally {
    for (const exported of exportsToDestroy) exported.destroy();
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
