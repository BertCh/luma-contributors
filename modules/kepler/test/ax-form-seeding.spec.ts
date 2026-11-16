// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import * as arrow from 'apache-arrow';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {KeplerComputeSession} from '../src/session/compute-session';
import {AnalysisRuntime} from '../../../examples/deck/kepler-compute/src/analysis-runtime';
import {requireTestDevice} from '../../analysis/test/utils/require-test-device';

/** Both regressions need execution evidence in addition to constructor/recipe Node coverage. */
test('coordinate-only DBSCAN executes across original source batches without a selected value', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 64 * 1024 * 1024});
  const first = arrow.tableFromArrays({
    longitude: new Float32Array([0, 0.001]),
    latitude: new Float32Array([0, 0])
  });
  const second = arrow.tableFromArrays({
    longitude: new Float32Array([1, 1.001]),
    latitude: new Float32Array([0, 0])
  });
  const dataset = session.importArrowDataset({
    id: 'coordinate-only-dbscan',
    table: first.concat(second)
  });
  const explorer = session.openExplorer(dataset.id, {
    workingMemoryBudgetBytes: 8 * 1024 * 1024,
    dimensions: [{id: 'longitude', column: 'longitude', kind: 'range'}]
  });
  const analysis = new AnalysisRuntime({session, explorer, requestUpdate: () => {}});
  try {
    expect(analysis.settings.column).toBe('');
    analysis.configureDBSCAN({
      longitudeColumn: 'longitude',
      latitudeColumn: 'latitude',
      coordinateSpace: 'longitude-latitude',
      metric: 'great-circle',
      units: 'meters',
      epsilon: 200,
      minimumSamples: 2
    });
    const encoder = device.createCommandEncoder();
    explorer.encode(encoder);
    analysis.encode(encoder);
    device.submit(encoder.finish());
    await analysis.readSummary();
    const labels = await Promise.all(
      [0, 1, 2, 3].map(async sourceRow => (await analysis.readRow(sourceRow))?.value)
    );
    if (analysis.error || JSON.stringify(labels) !== '[0,0,1,1]')
      throw new Error(
        `Coordinate-only DBSCAN mismatch: ${JSON.stringify({labels, expected: [0, 0, 1, 1], error: analysis.error, summary: analysis.summary})}`
      );
    expect(analysis.settings.column).toBe('');
    expect(analysis.summary?.dbscan).toMatchObject({clusterCount: 2, validCount: 4, noiseCount: 0});
    expect(analysis.colors.data.map(chunk => chunk.length)).toEqual([2, 2]);
    analysis.setMode('off');
    expect(() => analysis.setMode('z-score')).toThrow('Float32 analysis column');
  } finally {
    const fence = device.createFence();
    try {
      await analysis.destroy(fence.signaled);
      await session.destroy(fence.signaled);
    } finally {
      fence.destroy();
    }
  }
});

test('effective cases override uses population as its denominator and computes hand-calculated risks', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 64 * 1024 * 1024});
  const dataset = session.importArrowDataset({
    id: 'effective-rate-column',
    table: arrow.tableFromArrays({
      population: new Float32Array([100, 200]),
      cases: new Float32Array([10, 40])
    })
  });
  const explorer = session.openExplorer(dataset.id, {
    workingMemoryBudgetBytes: 8 * 1024 * 1024,
    dimensions: [{id: 'population', column: 'population', kind: 'range'}]
  });
  const analysis = new AnalysisRuntime({
    session,
    explorer,
    column: 'cases',
    requestUpdate: () => {}
  });
  try {
    analysis.setMode('excess-risk');
    const encoder = device.createCommandEncoder();
    explorer.encode(encoder);
    analysis.encode(encoder);
    device.submit(encoder.finish());
    await analysis.readSummary();
    const actual = await Promise.all(
      [0, 1].map(async sourceRow => (await analysis.readRow(sourceRow))?.value)
    );
    // Population rate = 50 / 300; risks = (10 / 100) / rate and (40 / 200) / rate.
    const expected = [0.6, 1.2];
    if (
      analysis.error ||
      actual.some(
        (value, index) =>
          value === undefined || !Number.isFinite(value) || Math.abs(value - expected[index]) > 1e-5
      )
    )
      throw new Error(
        `Effective rate denominator mismatch (tolerance 1e-5): ${JSON.stringify({actual, expected, error: analysis.error, summary: analysis.summary})}`
      );
    expect(analysis.summary?.denominatorColumn).toBe('population');
    expect(JSON.parse(analysis.serializeRecipe()).operation).toEqual({
      kind: 'excess-risk',
      events: 'cases',
      populations: 'population'
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
});
