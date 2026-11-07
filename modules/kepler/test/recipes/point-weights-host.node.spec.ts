// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {NullDevice} from '@luma.gl/core/test-utils';
import {expect, test, vi} from 'vitest';
import {KeplerComputeSession} from '../../src/session/compute-session';
import {getAnalysisRecipeColumns} from '../../../../examples/deck/kepler-compute/src/analysis-recipe-settings';
import {
  saveSandbox,
  replaySandbox
} from '../../../../examples/deck/kepler-compute/src/sandbox-state';
import type {ComputeRuntime} from '../../../../examples/deck/kepler-compute/src/compute-runtime';
import type {
  KeplerAnalysisRecipe,
  KeplerPointWeightsSettings
} from '../../src/recipes/analysis-recipe';

test.each([
  'spatial-rate-points',
  'hot-spots-points'
] as const)('%s workspace replay preflights coordinate fields and preserves topology when rebinding selection', async operationKind => {
  const device = new NullDevice({});
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 1024 * 1024});
  const dataset = session.importArrowDataset({
    id: 'resident-observations',
    table: arrow.tableFromArrays({
      events: new Float32Array([1, 2, 3]),
      population: new Float32Array([10, 20, 30]),
      longitude: new Float32Array([0, 1, 2]),
      latitude: new Float32Array([0, 0, 0])
    })
  });
  const explorer = {dataset, isCurrent: true, currentRevision: 5};
  const weights: KeplerPointWeightsSettings = {
    kind: 'knn',
    longitudeColumn: 'longitude',
    latitudeColumn: 'latitude',
    k: 2,
    coordinateSpace: 'longitude-latitude',
    metric: 'great-circle',
    units: 'meters'
  };
  const recipe: KeplerAnalysisRecipe = {
    schemaVersion: 1,
    algorithmVersion: 1,
    id: 'point-rates',
    source: {datasetId: dataset.id, revision: dataset.revision},
    population: 'selected',
    selectionRevision: 6,
    backendPolicy: 'require-gpu',
    operation:
      operationKind === 'spatial-rate-points'
        ? {
            kind: 'spatial-rate-points',
            events: 'events',
            populations: 'population',
            weights,
            scale: 1000,
            islandPolicy: 'undefined'
          }
        : {
            kind: 'hot-spots-points',
            column: 'events',
            weights,
            minimumNeighbors: 1,
            selfWeight: 1,
            multipleTesting: 'holm',
            islands: 'keep',
            inference: 'analytic-two-sided'
          }
  };
  const restoreRecipe = vi.fn();
  const runtime = {
    explorer,
    session,
    publications: new Map(),
    analysis: {
      settings: {mode: operationKind},
      serializeRecipe: () => JSON.stringify(recipe),
      restoreRecipe,
      setMode: vi.fn()
    },
    requestUpdate: vi.fn()
  } as unknown as ComputeRuntime;
  const applyNativeConfig = vi.fn(() => {
    explorer.currentRevision++;
  });
  const callbacks = {
    runtime,
    getNativeConfig: () => ({version: 'v1', config: {visState: {filters: []}}}),
    applyNativeConfig
  };
  try {
    expect(getAnalysisRecipeColumns(recipe)).toEqual([
      'events',
      ...(operationKind === 'spatial-rate-points' ? ['population'] : []),
      'longitude',
      'latitude'
    ]);
    const serialized = saveSandbox(callbacks);
    const invalid = JSON.parse(serialized);
    invalid.analyses[0].operation.weights.latitudeColumn = 'missing-latitude';
    await expect(replaySandbox(JSON.stringify(invalid), callbacks)).rejects.toThrow();
    expect(applyNativeConfig).not.toHaveBeenCalled();
    expect(restoreRecipe).not.toHaveBeenCalled();
    explorer.currentRevision = 10;
    await replaySandbox(serialized, callbacks);
    expect(applyNativeConfig).toHaveBeenCalledOnce();
    const restored = JSON.parse(restoreRecipe.mock.calls[0][0]);
    expect(restored.operation).toEqual(recipe.operation);
    expect(restored.selectionRevision).toBe(12);
    expect(restored.source).toEqual(recipe.source);
  } finally {
    await session.destroy(Promise.resolve());
    device.destroy();
  }
});
