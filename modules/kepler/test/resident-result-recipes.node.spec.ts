// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {KeplerDatasetRegistry} from '../src/runtime/dataset-registry';
import {ResidentEvictionHost} from '../../../examples/deck/kepler-compute/src/resident-eviction-host';
import {
  ResidentResultRecipes,
  type ResidentResultRecipe
} from '../../../examples/deck/kepler-compute/src/resident-result-recipes';

function createRecipe(resultId: string, inputIds: string[]): ResidentResultRecipe {
  return {
    resultId,
    rowCount: 300000,
    inputs: inputIds.map(datasetId => ({datasetId, revision: 1})),
    operation: {
      kind: 'suitability',
      request: {
        id: `operation-${resultId}`,
        dataId: inputIds[0],
        weights: {cost: 1},
        weightStandardization: 'normalize',
        dataStandardization: 'range',
        outputFieldName: 'score',
        resultId,
        resultLabel: resultId
      }
    }
  };
}

describe('ResidentResultRecipes', () => {
  it('orders recipes after the resident results they consume', () => {
    const recipes = new ResidentResultRecipes();
    recipes.record(createRecipe('second', ['first', 'other']));
    recipes.record(createRecipe('first', ['other']));
    expect(recipes.collect(['second', 'first']).map(recipe => recipe.resultId)).toEqual([
      'first',
      'second'
    ]);
  });

  it('fails when a resident result has no lineage or the lineage is cyclic', () => {
    const recipes = new ResidentResultRecipes();
    expect(() => recipes.collect(['missing'])).toThrow();
    recipes.record(createRecipe('a', ['b']));
    recipes.record(createRecipe('b', ['a']));
    expect(() => recipes.collect(['a', 'b'])).toThrow();
  });

  it('detaches recorded requests from the caller and replaces by result ID', () => {
    const recipes = new ResidentResultRecipes();
    const recipe = createRecipe('result', ['input']);
    recipes.record(recipe);
    if (recipe.operation.kind === 'suitability') recipe.operation.request.weights['cost'] = 99;
    const stored = recipes.get('result')?.operation;
    expect(stored?.kind === 'suitability' && stored.request.weights['cost']).toBe(1);
    recipes.record({...createRecipe('result', ['input']), rowCount: 5});
    expect(recipes.get('result')?.rowCount).toBe(5);
    recipes.forget('result');
    expect(recipes.get('result')).toBeUndefined();
  });
});

describe('ResidentEvictionHost.ensureResident', () => {
  it('shares one rebuild between concurrent callers', async () => {
    const registry = new KeplerDatasetRegistry({memoryBudgetBytes: 100});
    registry.registerDataset({
      id: 'input',
      value: {},
      batches: [{id: 'input-batch', rowCount: 1}],
      resources: [
        {
          id: 'input-resource',
          value: {},
          byteLength: 80,
          memoryKind: 'gpu',
          ownership: 'owned',
          destroy() {}
        }
      ]
    });
    const rebuilt: string[] = [];
    const host = new ResidentEvictionHost({
      enabled: true,
      getVisibleDatasetIds: () => new Set<string>(),
      canRebuild: () => true,
      evict: id => registry.removeDataset(id),
      async rebuild(id) {
        rebuilt.push(id);
      }
    });
    host.attach(registry);
    host.touch('input');
    registry.reserveMemory(50);
    expect(host.isEvicted('input')).toBe(true);
    await Promise.all([host.ensureResident('input'), host.ensureResident('input')]);
    expect(rebuilt).toEqual(['input']);
    expect(host.isEvicted('input')).toBe(false);
    await host.ensureResident('input');
  });
});
