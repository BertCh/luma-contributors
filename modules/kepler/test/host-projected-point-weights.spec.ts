// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import * as arrow from 'apache-arrow';
import {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {KeplerComputeSession} from '../src/session/compute-session';
import {AnalysisRuntime} from '../../../examples/deck/kepler-compute/src/analysis-runtime';
import {requireTestDevice} from '../../analysis/test/utils/require-test-device';

test('projected point weights preserve planar roles, null validity and source chunks through capacity recovery, selection and inference', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 64 * 1024 * 1024});
  const first = new arrow.Table({
    x: arrow.vectorFromArray([180, 181], new arrow.Float32()),
    y: arrow.makeVector(new Float32Array(2)),
    events: arrow.makeVector(new Float32Array([10, 20])),
    population: arrow.makeVector(new Float32Array([100, 100])),
    row: arrow.makeVector(new Float32Array([0, 1]))
  });
  const second = new arrow.Table({
    x: arrow.vectorFromArray([182, 200, null], new arrow.Float32()),
    y: arrow.makeVector(new Float32Array(3)),
    events: arrow.makeVector(new Float32Array([60, 90, 1000])),
    population: arrow.makeVector(new Float32Array([100, 100, 100])),
    row: arrow.makeVector(new Float32Array([2, 3, 4]))
  });
  const dataset = session.importArrowDataset({id: 'projected-points', table: first.concat(second)});
  const explorer = session.openExplorer(dataset.id, {
    workingMemoryBudgetBytes: 8 * 1024 * 1024,
    dimensions: [{id: 'row', column: 'row', kind: 'range'}]
  });
  const analysis = new AnalysisRuntime({session, explorer, requestUpdate: () => {}});
  const coordinates = {
    coordinateSpace: 'planar' as const,
    xColumn: 'x',
    yColumn: 'y',
    crs: 'EPSG:3857',
    bounds: [179, -1, 201, 1] as const
  };
  const topology = {
    ...coordinates,
    kind: 'distance-band' as const,
    radiusMeters: 1.1,
    neighborCapacity: 4
  };
  const rates = {...topology, scale: 100, islandPolicy: 'undefined' as const};
  const expectedWeights = {
    ...topology,
    radiusMeters: Math.fround(1.1),
    metric: 'native',
    units: 'meters'
  };
  const render = async () => {
    const encoder = device.createCommandEncoder();
    explorer.encode(encoder);
    analysis.encode(encoder);
    device.submit(encoder.finish());
    await analysis.readSummary();
  };
  const expectRates = async (values: readonly number[]) => {
    for (const [sourceRowIndex, value] of values.entries()) {
      const result = await analysis.readRow(sourceRowIndex);
      expect(result?.sourceRowIndex).toBe(sourceRowIndex);
      expect(result?.sourceRevision).toBe(dataset.revision);
      if (Number.isNaN(value)) expect(result?.value).toBeNaN();
      else expect(result?.value).toBeCloseTo(value, 5);
    }
  };
  try {
    analysis.setColumn('events');
    analysis.setDenominatorColumn('population');
    analysis.configurePointWeights({...rates, neighborCapacity: 1});
    await render();
    expect(analysis.error).toMatch(/overflowed; 4 slots are required/);
    expect(analysis.active).toBe(false);
    expect(analysis.summary).toBeNull();
    expect(analysis.getTableColumns()).toBeUndefined();
    expect(await analysis.readRow(0)).toBeNull();

    analysis.configurePointWeights(rates);
    const rateRecipe = analysis.serializeRecipe();
    expect(JSON.parse(rateRecipe).operation.weights).toEqual(expectedWeights);
    await render();
    expect(analysis.error).toBe('');
    expect(analysis.active).toBe(true);
    expect(analysis.summary?.pointWeights).toEqual({
      weightsOverflow: 0,
      weightsRequiredCount: 4,
      invalidCount: 1
    });
    // x=181 is a valid projected coordinate, even though it was a geographic null sentinel.
    await expectRates([15, 30, 40, NaN, NaN]);
    expect(analysis.summary?.count).toBe(3);
    expect(
      analysis.getTableColumns()?.columns['analysis_spatial_rate'].data.map(chunk => chunk.length)
    ).toEqual([2, 3]);
    expect(dataset.value.columns['x'].data.map(chunk => chunk.length)).toEqual([2, 3]);
    const {buildCount, encodeCount} = analysis.pointWeightsCacheStats;

    analysis.setPopulation('selected');
    explorer.setRange('row', 0, 1);
    await render();
    expect(analysis.error).toBe('');
    await expectRates([15, 15, NaN, NaN, NaN]);
    expect(analysis.summary?.pointWeights?.weightsRequiredCount).toBe(4);
    expect(analysis.pointWeightsCacheStats).toMatchObject({buildCount, encodeCount});
    analysis.setMode('off');
    analysis.restoreRecipe(rateRecipe);
    await render();
    expect(analysis.error).toBe('');
    expect(analysis.settings.population).toBe('all');
    expect(analysis.settings.pointWeights).toEqual({...rates, radiusMeters: Math.fround(1.1)});
    expect(JSON.parse(analysis.serializeRecipe()).operation.weights).toEqual(expectedWeights);
    await expectRates([15, 30, 40, NaN, NaN]);
    expect(analysis.pointWeightsCacheStats).toMatchObject({buildCount, encodeCount});

    analysis.configureLocalMoran({
      ...topology,
      minimumNeighbors: 1,
      islands: 'exclude',
      multipleTesting: 'none',
      significanceLevel: 0.9999,
      inference: 'analytic-conditional-normal-two-sided'
    });
    await render();
    expect(analysis.error).toBe('');
    expect(analysis.summary?.operation).toBe('local-moran-points');
    expect(JSON.parse(analysis.serializeRecipe()).operation.weights).toEqual(expectedWeights);
    for (const [sourceRowIndex, expected] of [2 / 7, -1 / 7, -3 / 7].entries())
      expect((await analysis.readRow(sourceRowIndex))?.value).toBeCloseTo(expected, 5);
    for (const vector of Object.values(analysis.getTableColumns()!.columns))
      expect(vector.data.map(chunk => chunk.length)).toEqual([2, 3]);
    expect(analysis.pointWeightsCacheStats).toMatchObject({buildCount, encodeCount});

    analysis.configureQuantileLisa({
      ...topology,
      quantileCount: 2,
      selectedQuantile: 1,
      permutationCount: 99,
      seed: 7,
      significanceLevel: 0.9999
    });
    await render();
    expect(analysis.error).toBe('');
    expect(analysis.summary?.operation).toBe('quantile-lisa-points');
    expect(JSON.parse(analysis.serializeRecipe()).operation.weights).toEqual(expectedWeights);
    // The isolated valid point contributes 90; the null coordinate's 1000 is excluded.
    expect(analysis.summary?.quantileLisa).toEqual({
      boundaries: [-Infinity, 40],
      validCount: 4,
      islandCount: 1,
      invalidQuantile: 0,
      overflow: 0
    });
    const quantileTable = analysis.getTableColumns()!;
    for (const vector of Object.values(quantileTable.columns))
      expect(vector.data.map(chunk => chunk.length)).toEqual([2, 3]);
    const membershipBuffer =
      quantileTable.columns['analysis_quantile_lisa_membership'].data[0].buffer;
    if (!(membershipBuffer instanceof Buffer)) throw new Error('Expected owned membership buffer');
    expect(Array.from(new Uint32Array((await membershipBuffer.readAsync(0, 20)).buffer))).toEqual([
      1, 1, 0, 0, 0
    ]);
    expect(analysis.pointWeightsCacheStats).toMatchObject({buildCount, encodeCount});

    analysis.configurePointWeights({
      ...coordinates,
      kind: 'knn',
      k: 1,
      scale: 100,
      islandPolicy: 'undefined'
    });
    await render();
    expect(analysis.error).toBe('');
    expect(JSON.parse(analysis.serializeRecipe()).operation.weights).toEqual({
      ...coordinates,
      kind: 'knn',
      k: 1,
      metric: 'native',
      units: 'meters'
    });
    expect(analysis.summary?.pointWeights?.invalidCount).toBe(1);
    // The formerly isolated D now has C as its nearest neighbor.
    expect((await analysis.readRow(3))?.value).toBeCloseTo(75, 5);
    expect((await analysis.readRow(4))?.value).toBeNaN();
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
