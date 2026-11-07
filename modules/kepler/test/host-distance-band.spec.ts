// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import * as arrow from 'apache-arrow';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {KeplerComputeSession} from '../src/session/compute-session';
import {AnalysisRuntime} from '../../../examples/deck/kepler-compute/src/analysis-runtime';
import {requireTestDevice} from '../../graph/test/utils/require-test-device';

test('distance-band rates reject truncated antimeridian topology and reuse complete source weights across selection and replay', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 64 * 1024 * 1024});
  const first = new arrow.Table({
    events: arrow.makeVector(new Float32Array([10, 20])),
    population: arrow.makeVector(new Float32Array([100, 100])),
    longitude: arrow.vectorFromArray([179.99, -179.99], new arrow.Float32()),
    latitude: arrow.makeVector(new Float32Array(2)),
    row: arrow.makeVector(new Float32Array([0, 1]))
  });
  const second = new arrow.Table({
    events: arrow.makeVector(new Float32Array([60, 90, 1000])),
    population: arrow.makeVector(new Float32Array([100, 100, 100])),
    longitude: arrow.vectorFromArray([179.98, 0, null], new arrow.Float32()),
    latitude: arrow.makeVector(new Float32Array(3)),
    row: arrow.makeVector(new Float32Array([2, 3, 4]))
  });
  const dataset = session.importArrowDataset({id: 'distance-band', table: first.concat(second)});
  const explorer = session.openExplorer('distance-band', {
    workingMemoryBudgetBytes: 8 * 1024 * 1024,
    dimensions: [{id: 'row', column: 'row', kind: 'range'}]
  });
  const analysis = new AnalysisRuntime({session, explorer, requestUpdate: () => {}});
  const settings = {
    kind: 'distance-band' as const,
    longitudeColumn: 'longitude',
    latitudeColumn: 'latitude',
    radiusMeters: 4000,
    neighborCapacity: 1,
    scale: 100,
    islandPolicy: 'undefined' as const
  };
  const render = async () => {
    const encoder = device.createCommandEncoder();
    explorer.encode(encoder);
    analysis.encode(encoder);
    device.submit(encoder.finish());
    await analysis.readSummary();
  };
  try {
    analysis.setColumn('events');
    analysis.setDenominatorColumn('population');
    analysis.configurePointWeights(settings);
    expect(analysis.active).toBe(false);
    await render();
    expect(analysis.error).toMatch(/overflowed; 6 slots are required/);
    expect(analysis.active).toBe(false);
    expect(analysis.summary).toBeNull();
    expect(analysis.getTableColumns()).toBeUndefined();
    expect(await analysis.readRow(0)).toBeNull();

    analysis.configurePointWeights({...settings, neighborCapacity: 6});
    const saved = analysis.serializeRecipe();
    expect(JSON.parse(saved).operation.weights).toEqual({
      kind: 'distance-band',
      longitudeColumn: 'longitude',
      latitudeColumn: 'latitude',
      coordinateSpace: 'longitude-latitude',
      metric: 'great-circle',
      units: 'meters',
      radiusMeters: 4000,
      neighborCapacity: 6
    });
    await render();
    expect(analysis.error).toBe('');
    expect(analysis.active).toBe(true);
    expect(analysis.summary?.pointWeights).toEqual({
      weightsOverflow: 0,
      weightsRequiredCount: 6,
      invalidCount: 1
    });
    expect(analysis.summary?.count).toBe(3);
    for (const row of [0, 1, 2]) expect((await analysis.readRow(row))?.value).toBeCloseTo(30, 5);
    for (const row of [3, 4]) expect((await analysis.readRow(row))?.value).toBeNaN();
    expect(
      analysis.getTableColumns()?.columns['analysis_spatial_rate'].data.map(chunk => chunk.length)
    ).toEqual([2, 3]);
    expect(dataset.value.columns['longitude'].data.map(chunk => chunk.length)).toEqual([2, 3]);
    const {buildCount, encodeCount} = analysis.pointWeightsCacheStats;

    analysis.setPopulation('selected');
    explorer.setRange('row', 0, 1);
    expect(analysis.active).toBe(false);
    await render();
    expect(analysis.error).toBe('');
    expect(analysis.summary?.pointWeights?.weightsRequiredCount).toBe(6);
    expect(analysis.summary?.count).toBe(2);
    for (const row of [0, 1]) expect((await analysis.readRow(row))?.value).toBeCloseTo(15, 5);
    expect(analysis.pointWeightsCacheStats).toMatchObject({buildCount, encodeCount});

    analysis.setMode('off');
    analysis.restoreRecipe(saved);
    await render();
    expect(analysis.error).toBe('');
    expect(analysis.settings.population).toBe('all');
    expect(analysis.settings.pointWeights).toMatchObject({
      kind: 'distance-band',
      radiusMeters: 4000,
      neighborCapacity: 6
    });
    for (const row of [0, 1, 2]) expect((await analysis.readRow(row))?.value).toBeCloseTo(30, 5);
    expect(analysis.pointWeightsCacheStats).toMatchObject({buildCount, encodeCount});
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

test('distance-band Local Moran refuses permutation publication when actual degree exceeds 64', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 64 * 1024 * 1024});
  session.importArrowDataset({
    id: 'dense-band',
    table: arrow.tableFromArrays({
      events: Float32Array.from({length: 66}, (_, row) => row + 1),
      longitude: Float32Array.from({length: 66}, (_, row) => row * 0.00001),
      latitude: new Float32Array(66)
    })
  });
  const explorer = session.openExplorer('dense-band', {
    workingMemoryBudgetBytes: 8 * 1024 * 1024,
    dimensions: []
  });
  const analysis = new AnalysisRuntime({session, explorer, requestUpdate: () => {}});
  try {
    analysis.setColumn('events');
    analysis.configureLocalMoran({
      kind: 'distance-band',
      longitudeColumn: 'longitude',
      latitudeColumn: 'latitude',
      radiusMeters: 1000,
      neighborCapacity: 66 * 65,
      minimumNeighbors: 1,
      islands: 'exclude',
      multipleTesting: 'none',
      significanceLevel: 0.05,
      inference: 'permutation-conditional-two-sided',
      permutationCount: 99,
      seed: 314159
    });
    expect(analysis.active).toBe(false);
    const encoder = device.createCommandEncoder();
    explorer.encode(encoder);
    analysis.encode(encoder);
    device.submit(encoder.finish());
    await analysis.readSummary();
    // Complete CSR fits 4290 slots. Each row has 65 neighbors, exceeding only inference scratch.
    expect(analysis.error).toBe(
      'Local Moran permutation capacity was exceeded; results are unavailable'
    );
    expect(analysis.active).toBe(false);
    expect(analysis.summary).toBeNull();
    expect(analysis.getTableColumns()).toBeUndefined();
    expect(await analysis.readRow(0)).toBeNull();
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
