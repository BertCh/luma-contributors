// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import * as arrow from 'apache-arrow';
import {NullDevice} from '@luma.gl/core/test-utils';
import {KeplerComputeSession, type KeplerSessionExplorer} from '../src/session/compute-session';
import {
  AnalysisRuntime,
  type PointWeightsSettings
} from '../../../examples/deck/kepler-compute/src/analysis-runtime';

test('point-weight configuration validates coordinate roles before mutation and detaches nested settings', async () => {
  const device = new NullDevice({});
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 64 * 1024 * 1024});
  const dataset = session.importArrowDataset({
    id: 'points',
    table: arrow.tableFromArrays({
      events: new Float32Array([10, 20]),
      population: new Float32Array([100, 100]),
      longitude: new Float32Array([0, 1]),
      latitude: new Float32Array([0, 0]),
      category: ['a', 'b']
    })
  });
  const explorer = {
    dataset,
    isCurrent: true,
    currentRevision: 0
  } as unknown as KeplerSessionExplorer;
  const analysis = new AnalysisRuntime({session, explorer, requestUpdate: () => {}});
  const settings: PointWeightsSettings = {
    longitudeColumn: 'longitude',
    latitudeColumn: 'latitude',
    k: 1,
    scale: 100,
    islandPolicy: 'undefined'
  };
  try {
    const before = analysis.settings;
    expect(() => analysis.setMode('spatial-rate-points')).toThrow('Configure explicit');
    expect(() => analysis.configurePointWeights({...settings, k: 33})).toThrow('1 to 32');
    expect(() => analysis.configurePointWeights({...settings, latitudeColumn: 'category'})).toThrow(
      'Float32'
    );
    expect(() => analysis.configurePointWeights({...settings, scale: Infinity})).toThrow('finite');
    expect(analysis.settings).toEqual(before);
    const bytes = session.registry.memoryUsage.totalBytes;
    analysis.configurePointWeights(settings);
    settings.k = 32;
    analysis.settings.pointWeights!.longitudeColumn = 'missing';
    expect(analysis.settings.pointWeights).toMatchObject({k: 1});
    expect(analysis.settings.pointWeights?.longitudeColumn).toBe('longitude');
    expect(analysis.resultColumn).toBe('analysis_spatial_rate');
    expect(session.registry.memoryUsage.totalBytes).toBe(bytes);
    const saved = analysis.serializeRecipe();
    expect(JSON.parse(saved).operation).toEqual({
      kind: 'spatial-rate-points',
      events: 'events',
      populations: 'population',
      weights: {
        kind: 'knn',
        longitudeColumn: 'longitude',
        latitudeColumn: 'latitude',
        k: 1,
        coordinateSpace: 'longitude-latitude',
        metric: 'great-circle',
        units: 'meters'
      },
      scale: 100,
      islandPolicy: 'undefined'
    });
    analysis.setMode('off');
    analysis.restoreRecipe(saved);
    expect(analysis.settings.mode).toBe('spatial-rate-points');
    const invalid = JSON.parse(saved);
    invalid.operation.weights.longitudeColumn = 'missing';
    expect(() => analysis.restoreRecipe(JSON.stringify(invalid))).toThrow('Float32');
    expect(analysis.settings.pointWeights?.longitudeColumn).toBe('longitude');
  } finally {
    await analysis.destroy(Promise.resolve());
    await session.destroy(Promise.resolve());
    device.destroy();
  }
});
