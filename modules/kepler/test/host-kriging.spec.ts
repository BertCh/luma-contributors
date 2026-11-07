// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test, vi} from 'vitest';
import * as arrow from 'apache-arrow';
import {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {KeplerComputeSession, type KeplerKrigingSettings} from '../src';
import {AnalysisRuntime} from '../../../examples/deck/kepler-compute/src/analysis-runtime';
import {requireTestDevice} from '../../analysis/test/utils/require-test-device';

const settings: KeplerKrigingSettings = {
  longitudeColumn: 'longitude',
  latitudeColumn: 'latitude',
  coordinateSpace: 'longitude-latitude',
  metric: 'great-circle',
  units: 'meters',
  grid: {bounds: [-0.5, -0.5, 3.5, 0.5], width: 4, height: 1},
  searchRadiusMeters: 150000,
  neighborCount: 2,
  minimumNeighbors: 2,
  variogram: {model: 'spherical', nugget: 0.25, sill: 4, rangeMeters: 500000}
};

test('kriging publishes separate prediction and variance cells with exact hits, source validity, selected samples and replay', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 64 * 1024 * 1024});
  const first = new arrow.Table({
    longitude: arrow.vectorFromArray([0, 2], new arrow.Float32()),
    latitude: arrow.makeVector(new Float32Array(2)),
    value: arrow.vectorFromArray([10, 10], new arrow.Float32()),
    row: arrow.makeVector(new Float32Array([0, 1]))
  });
  const second = new arrow.Table({
    longitude: arrow.vectorFromArray([1, 1, null], new arrow.Float32()),
    latitude: arrow.makeVector(new Float32Array(3)),
    value: arrow.vectorFromArray([null, Infinity, 1000], new arrow.Float32()),
    row: arrow.makeVector(new Float32Array([2, 3, 4]))
  });
  const dataset = session.importArrowDataset({id: 'kriging-source', table: first.concat(second)});
  const sourceChunks = ['longitude', 'latitude', 'value'].flatMap(
    name => dataset.value.columns[name].data
  );
  const sourceBuffers = [
    ...new Set(
      sourceChunks.map(chunk =>
        chunk.buffer instanceof Buffer ? chunk.buffer : chunk.buffer.buffer
      )
    )
  ];
  const sourceWrites = sourceBuffers.map(buffer => vi.spyOn(buffer, 'write'));
  const explorer = session.openExplorer(dataset.id, {
    workingMemoryBudgetBytes: 8 * 1024 * 1024,
    dimensions: [{id: 'row', column: 'row', kind: 'range'}]
  });
  const analysis = new AnalysisRuntime({session, explorer, requestUpdate: () => {}});
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
    analysis.setColumn('value');
    analysis.configureKriging(settings);
    const savedRecipe = analysis.serializeRecipe();
    expect(JSON.parse(savedRecipe).operation).toEqual({
      kind: 'kriging-points',
      column: 'value',
      ...settings
    });
    await render();
    expect(analysis.getTableColumns()).toBeUndefined();
    expect(await analysis.readRow(0)).toBeNull();
    expect(analysis.summary).toMatchObject({
      operation: 'kriging-points',
      sourceRevision: dataset.revision,
      count: 3,
      nullCount: 1,
      interpolation: {width: 4, height: 1, validSourceCount: 2, cellCount: 4},
      kriging: {invalidCount: 1, uncertainCount: 0, variance: {count: 3, nullCount: 1, minimum: 0}}
    });
    const cells = await analysis.readInterpolationPage(0, 4);
    for (const cell of cells) {
      expect(cell.operation).toBe('kriging-points');
      expect(cell).not.toHaveProperty('neighbors');
      expect(cell.longitude).toBe(cell.cellIndex);
      expect(cell.latitude).toBe(0);
    }
    for (const cellIndex of [0, 2])
      expect(cells[cellIndex]).toMatchObject({cellIndex, value: 10, variance: 0});
    expect(cells[3]).toMatchObject({cellIndex: 3, value: null, variance: null});

    // Independent ordinary-kriging reference: symmetric samples imply weights (1/2,1/2).
    // On the equator one degree spans R*pi/180; mu=gamma(d)-gamma(2d)/2,
    // giving variance=sum(weight*gamma(d))+mu=2*gamma(d)-gamma(2d)/2.
    const distance = (6371008.8 * Math.PI) / 180;
    const sphericalSemivariance = (meters: number) => {
      const fraction = Math.min(meters / settings.variogram.rangeMeters, 1);
      return (
        settings.variogram.nugget + settings.variogram.sill * (1.5 * fraction - 0.5 * fraction ** 3)
      );
    };
    const expectedVariance =
      2 * sphericalSemivariance(distance) - sphericalSemivariance(2 * distance) / 2;
    if (cells[1].operation !== 'kriging-points') throw new Error('Expected kriging cell');
    expect(cells[1].value).toBeCloseTo(10, 5);
    expect(cells[1].variance).toBeCloseTo(expectedVariance, 4);
    expect(analysis.summary?.kriging?.variance.maximum).toBeCloseTo(expectedVariance, 4);
    expect(analysis.summary?.kriging?.variance.mean).toBeCloseTo(expectedVariance / 3, 4);
    expect(analysis.getInterpolationLayer()?.props.bounds).toEqual(settings.grid.bounds);
    const varianceExport = await analysis.exportInterpolationGeoTIFF('variance');
    exportsToDestroy.push(varianceExport);
    expect(varianceExport.isCurrent()).toBe(true);
    expect(varianceExport.readbackBytes).toBe(16);

    analysis.setPopulation('selected');
    explorer.setRange('row', 0, 0);
    expect(analysis.getInterpolationLayer()).toBeNull();
    expect(varianceExport.isCurrent()).toBe(false);
    varianceExport.destroy();
    await render();
    expect(analysis.summary).toMatchObject({
      count: 1,
      nullCount: 3,
      interpolation: {validSourceCount: 1},
      kriging: {
        invalidCount: 3,
        uncertainCount: 0,
        variance: {count: 1, nullCount: 3, minimum: 0, maximum: 0}
      }
    });
    // Exact hits remain valid below minimumNeighbors; other cells have insufficient samples.
    const selectedCells = await analysis.readInterpolationPage(0, 4);
    expect(selectedCells[0]).toMatchObject({value: 10, variance: 0});
    for (const cell of selectedCells.slice(1))
      expect(cell).toMatchObject({value: null, variance: null});

    analysis.setMode('off');
    analysis.restoreRecipe(savedRecipe);
    await render();
    expect(analysis.settings.population).toBe('all');
    expect(analysis.settings.kriging).toEqual(settings);
    expect(await analysis.readInterpolationPage(0, 4)).toEqual(cells);

    // This nearly flat Gaussian model loses its nonzero pair semivariances in Float32.
    // The midpoint solve is uncertain; the distant cell still lacks enough neighbors.
    analysis.configureKriging({
      ...settings,
      variogram: {model: 'gaussian', nugget: 0, sill: 4, rangeMeters: 1e12}
    });
    await render();
    expect(analysis.summary).toMatchObject({
      count: 2,
      nullCount: 2,
      kriging: {invalidCount: 1, uncertainCount: 1}
    });
    const uncertainCells = await analysis.readInterpolationPage(0, 4);
    for (const cellIndex of [0, 2])
      expect(uncertainCells[cellIndex]).toMatchObject({value: 10, variance: 0});
    for (const cellIndex of [1, 3])
      expect(uncertainCells[cellIndex]).toMatchObject({value: null, variance: null});
    analysis.restoreRecipe(savedRecipe);
    await render();
    expect(await analysis.readInterpolationPage(0, 4)).toEqual(cells);
    for (const [index, chunk] of ['longitude', 'latitude', 'value']
      .flatMap(name => dataset.value.columns[name].data)
      .entries())
      expect(chunk).toBe(sourceChunks[index]);
    expect(dataset.value.columns.value.data.map(chunk => chunk.length)).toEqual([2, 3]);
    for (const write of sourceWrites) expect(write).not.toHaveBeenCalled();
  } finally {
    for (const exported of exportsToDestroy) exported.destroy();
    for (const write of sourceWrites) write.mockRestore();
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
