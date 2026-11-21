// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import * as arrow from 'apache-arrow';
import {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {KeplerComputeSession} from '../src/session/compute-session';
import {
  AnalysisRuntime,
  type PointWeightsSettings
} from '../../../examples/deck/kepler-compute/src/analysis-runtime';
import {requireTestDevice} from '../../analysis/src/testing/require-test-device';

const pointSettings: PointWeightsSettings = {
  longitudeColumn: 'longitude',
  latitudeColumn: 'latitude',
  k: 1,
  scale: 100,
  islandPolicy: 'undefined'
};

test('point spatial rates preserve source nearest neighbors through filtering and diagnose invalid coordinates', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 64 * 1024 * 1024});
  const first = new arrow.Table({
    events: arrow.makeVector(new Float32Array([10, 20, 90])),
    population: arrow.makeVector(new Float32Array([100, 100, 100])),
    longitude: arrow.vectorFromArray([0, 1, 3], new arrow.Float32()),
    latitude: arrow.makeVector(new Float32Array([0, 0, 0])),
    row: arrow.makeVector(new Float32Array([0, 1, 2]))
  });
  const second = new arrow.Table({
    events: arrow.makeVector(new Float32Array([80, 1000, 1000, 1000])),
    population: arrow.makeVector(new Float32Array([100, 100, 100, 100])),
    longitude: arrow.vectorFromArray([10, null, 181, 0], new arrow.Float32()),
    latitude: arrow.makeVector(new Float32Array([0, 0, 0, 91])),
    row: arrow.makeVector(new Float32Array([3, 4, 5, 6]))
  });
  const dataset = session.importArrowDataset({id: 'points', table: first.concat(second)});
  const explorer = session.openExplorer('points', {
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
    expect(analysis.error).toBe('');
    expect(analysis.summary?.pointWeights).toEqual({
      weightsOverflow: 0,
      weightsRequiredCount: 4,
      invalidCount: 3
    });
  };
  try {
    analysis.configurePointWeights(pointSettings);
    const saved = analysis.serializeRecipe();
    await render();
    expect(analysis.pointWeightsCacheStats).toMatchObject({
      buildCount: 1,
      encodeCount: 1,
      entryCount: 1
    });
    // Along the equator the unique nearest source neighbors are 0→1, 1→0, 2→1, 3→2.
    for (const [row, expected] of [15, 15, 55, 85].entries())
      expect((await analysis.readRow(row))?.value).toBeCloseTo(expected, 5);
    for (const row of [4, 5, 6]) expect((await analysis.readRow(row))?.value).toBeNaN();
    expect((await analysis.readRow(0))?.kind).toBe('spatial-rate-points');
    expect(analysis.summary?.count).toBe(4);
    expect(
      analysis.getTableColumns()?.columns['analysis_spatial_rate'].data.map(chunk => chunk.length)
    ).toEqual([3, 4]);
    analysis.setPopulation('selected');
    explorer.setRange('row', 2, 3);
    await render();
    // Row 2's original neighbor 1 is excluded; row 3 must not refill its neighborhood.
    expect((await analysis.readRow(2))?.value).toBeNaN();
    expect((await analysis.readRow(3))?.value).toBeCloseTo(85, 5);
    expect(analysis.summary?.count).toBe(1);
    expect(analysis.pointWeightsCacheStats).toMatchObject({buildCount: 1, encodeCount: 1});
    analysis.configurePointWeights({...pointSettings, islandPolicy: 'own-rate'});
    await render();
    expect((await analysis.readRow(2))?.value).toBeCloseTo(90, 5);
    expect((await analysis.readRow(3))?.value).toBeCloseTo(85, 5);
    expect(dataset.value.columns['longitude'].data.map(chunk => chunk.length)).toEqual([3, 4]);
    analysis.restoreRecipe(saved);
    expect(analysis.settings.population).toBe('all');
    expect(analysis.settings.pointWeights?.islandPolicy).toBe('undefined');
    await render();
    analysis.configurePointHotspots({
      longitudeColumn: 'longitude',
      latitudeColumn: 'latitude',
      k: 1,
      minimumNeighbors: 1,
      selfWeight: 1,
      multipleTesting: 'none',
      islands: 'exclude'
    });
    await render();
    expect(analysis.summary?.operation).toBe('hot-spots-points');
    expect(analysis.summary?.count).toBe(4);
    expect(analysis.summary?.hotspot?.invalidCount).toBe(0);
    expect(
      analysis
        .getTableColumns()
        ?.columns['analysis_hot_spot_p_value'].data.map(chunk => chunk.length)
    ).toEqual([3, 4]);
    expect(analysis.pointWeightsCacheStats).toMatchObject({
      buildCount: 1,
      encodeCount: 1,
      entryCount: 1
    });
    analysis.configureLocalMoran({
      longitudeColumn: 'longitude',
      latitudeColumn: 'latitude',
      k: 1,
      minimumNeighbors: 1,
      significanceLevel: 0.9999,
      multipleTesting: 'none',
      islands: 'exclude'
    });
    const localMoranRecipe = analysis.serializeRecipe();
    await render();
    expect(analysis.summary?.operation).toBe('local-moran-points');
    expect(analysis.summary?.localMoran?.invalidCount).toBe(0);
    // n=4, mean=50, sum of squares=5000; directed neighbors 0→1,1→0,2→1,3→2.
    for (const [row, expected] of [0.72, 0.72, -0.72, 0.72].entries())
      expect((await analysis.readRow(row))?.value).toBeCloseTo(expected, 5);
    const snapshot = analysis.getTableColumns()!;
    expect(Object.keys(snapshot.columns)).toHaveLength(7);
    for (const vector of Object.values(snapshot.columns))
      expect(vector.data.map(chunk => chunk.length)).toEqual([3, 4]);
    const colors = analysis.colors.data[0].buffer;
    if (!(colors instanceof Buffer)) throw new Error('Expected owned color buffer');
    expect(Array.from(await colors.readAsync(0, 16))).toEqual([
      90, 150, 205, 255, 90, 150, 205, 255, 150, 60, 175, 255, 240, 125, 90, 255
    ]);
    expect(analysis.pointWeightsCacheStats).toMatchObject({buildCount: 1, encodeCount: 1});
    analysis.setMode('off');
    analysis.restoreRecipe(localMoranRecipe);
    await render();
    expect(analysis.settings.localMoran?.significanceLevel).toBe(0.9999);
    expect(analysis.pointWeightsCacheStats).toMatchObject({buildCount: 1, encodeCount: 1});
    analysis.configureLocalMoran({
      ...analysis.settings.localMoran!,
      inference: 'permutation-conditional-two-sided',
      permutationCount: 99,
      seed: 314159
    });
    expect(analysis.active).toBe(false);
    expect(analysis.getTableColumns()).toBeUndefined();
    const permutationRecipe = analysis.serializeRecipe();
    await render();
    expect(analysis.active).toBe(true);
    const readPermutationValues = async () => {
      const vector = analysis.getTableColumns()!.columns['analysis_local_moran_p_value'];
      const buffer = vector.data[0].buffer;
      if (!(buffer instanceof Buffer)) throw new Error('Expected owned p-value buffer');
      return new Float32Array((await buffer.readAsync(0, 28)).buffer);
    };
    const permutationValues = await readPermutationValues();
    for (const probability of permutationValues.slice(0, 4)) {
      expect(probability).toBeGreaterThanOrEqual(Math.fround(2 / 100));
      expect(probability).toBeLessThanOrEqual(1);
    }
    for (const probability of permutationValues.slice(4)) expect(probability).toBeNaN();
    analysis.setPopulation('selected');
    explorer.setRange('row', 2, 3);
    expect(analysis.active).toBe(false);
    await render();
    // Fixed neighbors: row 2 loses row 1; excluding that island leaves row 3 untested too.
    const selectedProbabilities = await readPermutationValues();
    expect(selectedProbabilities[2]).toBeNaN();
    expect(selectedProbabilities[3]).toBeNaN();
    expect(analysis.summary?.localMoran?.invalidCount).toBe(2);
    analysis.restoreRecipe(permutationRecipe);
    await render();
    expect(Array.from(await readPermutationValues())).toEqual(Array.from(permutationValues));
    expect(analysis.settings.localMoran).toMatchObject({
      inference: 'permutation-conditional-two-sided',
      permutationCount: 99,
      seed: 314159
    });
    expect(analysis.pointWeightsCacheStats).toMatchObject({buildCount: 1, encodeCount: 1});
    analysis.configureQuantileLisa({
      longitudeColumn: 'longitude',
      latitudeColumn: 'latitude',
      k: 1,
      quantileCount: 2,
      selectedQuantile: 2,
      seed: 314159,
      permutationCount: 19,
      significanceLevel: 0.9999
    });
    const quantileLisaRecipe = analysis.serializeRecipe();
    await render();
    expect(analysis.summary?.operation).toBe('quantile-lisa-points');
    expect(analysis.summary?.quantileLisa).toEqual({
      alternative: 'geoda-legacy',
      boundaries: [50, Infinity],
      validCount: 4,
      islandCount: 0,
      invalidQuantile: 0,
      overflow: 0
    });
    expect(analysis.summary?.count).toBe(1);
    for (const row of [0, 1, 2]) expect((await analysis.readRow(row))?.value).toBeNaN();
    const quantileProbability = (await analysis.readRow(3))!.value;
    expect(quantileProbability).toBeGreaterThanOrEqual(0.05);
    expect(quantileProbability).toBeLessThanOrEqual(0.55);
    expect(Array.from(await colors.readAsync(0, 16))).toEqual([
      150, 150, 150, 255, 150, 150, 150, 255, 150, 150, 150, 255, 215, 25, 28, 255
    ]);
    const quantileSnapshot = analysis.getTableColumns()!;
    expect(Object.keys(quantileSnapshot.columns)).toHaveLength(6);
    for (const vector of Object.values(quantileSnapshot.columns))
      expect(vector.data.map(chunk => chunk.length)).toEqual([3, 4]);
    expect(analysis.pointWeightsCacheStats).toMatchObject({buildCount: 1, encodeCount: 1});
    analysis.setMode('off');
    analysis.restoreRecipe(quantileLisaRecipe);
    await render();
    expect((await analysis.readRow(3))?.value).toBe(quantileProbability);
    const invalidQuantileRecipe = JSON.parse(quantileLisaRecipe);
    invalidQuantileRecipe.operation.multipleTesting = 'benjamini-hochberg';
    expect(() => analysis.restoreRecipe(JSON.stringify(invalidQuantileRecipe))).toThrow();
    expect(analysis.serializeRecipe()).toBe(quantileLisaRecipe);
    analysis.setPopulation('selected');
    explorer.setRange('row', 2, 3);
    await render();
    expect(analysis.summary?.quantileLisa).toMatchObject({
      boundaries: [85, Infinity],
      validCount: 2,
      islandCount: 1
    });
    expect(analysis.summary?.count).toBe(0);
    // Source row2 loses its original neighbor1, so it stays isolated instead of adopting row3.
    expect(Array.from(await colors.readAsync(8, 8))).toEqual([
      253, 174, 97, 255, 150, 150, 150, 255
    ]);
    expect(analysis.pointWeightsCacheStats).toMatchObject({buildCount: 1, encodeCount: 1});
    analysis.restoreRecipe(quantileLisaRecipe);
    await render();
    expect((await analysis.readRow(3))?.value).toBe(quantileProbability);
    analysis.setMode('off');
    analysis.configurePointWeights(pointSettings);
    await render();
    expect(analysis.pointWeightsCacheStats).toMatchObject({
      buildCount: 1,
      encodeCount: 1,
      entryCount: 1
    });
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

test('one point supports oversized k and explicit island policy without allocating zero-size CSR buffers', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 64 * 1024 * 1024});
  session.importArrowDataset({
    id: 'one',
    table: arrow.tableFromArrays({
      events: new Float32Array([10]),
      population: new Float32Array([100]),
      longitude: new Float32Array([0]),
      latitude: new Float32Array([0])
    })
  });
  const explorer = session.openExplorer('one', {
    workingMemoryBudgetBytes: 8 * 1024 * 1024,
    dimensions: []
  });
  const analysis = new AnalysisRuntime({session, explorer, requestUpdate: () => {}});
  const render = async () => {
    const encoder = device.createCommandEncoder();
    explorer.encode(encoder);
    analysis.encode(encoder);
    device.submit(encoder.finish());
    await analysis.readSummary();
    expect(analysis.summary?.pointWeights).toEqual({
      weightsOverflow: 0,
      weightsRequiredCount: 0,
      invalidCount: 0
    });
  };
  try {
    analysis.configurePointWeights({...pointSettings, k: 32, islandPolicy: 'own-rate'});
    const abandoned = device.createCommandEncoder();
    analysis.encode(abandoned);
    expect(analysis.pointWeightsCacheStats).toMatchObject({buildCount: 0, encodeCount: 1});
    abandoned.destroy();
    await render();
    expect(analysis.pointWeightsCacheStats).toMatchObject({buildCount: 1, encodeCount: 2});
    expect((await analysis.readRow(0))?.value).toBeCloseTo(10, 5);
    analysis.configurePointWeights({...pointSettings, k: 32});
    await render();
    expect((await analysis.readRow(0))?.value).toBeNaN();
    expect(analysis.summary?.count).toBe(0);
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
