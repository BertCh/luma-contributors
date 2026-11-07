// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it, vi} from 'vitest';
import * as arrow from 'apache-arrow';
import {
  parseKeplerAnalysisRecipe,
  restoreKeplerAnalysisRecipe,
  serializeKeplerAnalysisRecipe,
  type KeplerAnalysisRecipe
} from '../src/recipes/analysis-recipe';
import {
  executeKeplerRequest,
  selectKeplerExecutionRoute,
  type KeplerExecutionRequest
} from '../src/execution/execution-router';
import {createKeplerDuckDBExecutor} from '../src/execution/duckdb-executor';

const recipe: KeplerAnalysisRecipe = {
  schemaVersion: 1,
  algorithmVersion: 1,
  id: 'z-scores',
  source: {datasetId: 'observations', revision: 2},
  population: 'all',
  backendPolicy: 'prefer-gpu',
  operation: {kind: 'standardize', column: 'income', method: 'z-score', variance: 'sample'}
};
const capabilities = {webgpu: true, availableBytes: 1024, duckdb: true, analysisBackend: true};
function analysisRequest(value = recipe): Extract<KeplerExecutionRequest, {kind: 'analysis'}> {
  return {
    kind: 'analysis',
    recipe: value,
    input: {rowCount: 10, chunkCount: 2, formats: {income: 'float32'}},
    estimatedBytes: 128
  };
}

it('saved recipes preserve choices, detach mutable input and reject unavailable versions', () => {
  const parsed = parseKeplerAnalysisRecipe(recipe);
  expect(parsed).not.toBe(recipe);
  expect(Object.isFrozen(parsed.operation)).toBe(true);
  const restored = restoreKeplerAnalysisRecipe(serializeKeplerAnalysisRecipe(recipe), {
    resolveDatasetRevision: () => 2
  });
  expect(restored).toEqual(recipe);
  expect(() =>
    restoreKeplerAnalysisRecipe(serializeKeplerAnalysisRecipe(recipe), {
      resolveDatasetRevision: () => 3
    })
  ).toThrow('stale');
  expect(() => parseKeplerAnalysisRecipe({...recipe, schemaVersion: 2})).toThrow('version');
  expect(() =>
    parseKeplerAnalysisRecipe({...recipe, operation: {...recipe.operation, hiddenMask: true}})
  ).toThrow('Unknown');
  expect(() => parseKeplerAnalysisRecipe({...recipe, population: 'selected'})).toThrow('selection');
});

it('selected and spatial-rate recipes require the exact saved mask and weights revisions', () => {
  const selected: KeplerAnalysisRecipe = {
    ...recipe,
    population: 'selected',
    selectionRevision: 4,
    operation: {
      kind: 'spatial-rate',
      events: 'events',
      populations: 'population',
      weightsId: 'neighbors',
      weightsRevision: 3,
      scale: 100000,
      islandPolicy: 'undefined'
    }
  };
  const saved = serializeKeplerAnalysisRecipe(selected);
  expect(() => restoreKeplerAnalysisRecipe(saved, {resolveDatasetRevision: () => 2})).toThrow(
    'selection'
  );
  expect(() =>
    restoreKeplerAnalysisRecipe(saved, {
      resolveDatasetRevision: () => 2,
      resolveSelectionRevision: () => 4,
      resolveWeightsRevision: () => 2
    })
  ).toThrow('weights');
  expect(
    restoreKeplerAnalysisRecipe(saved, {
      resolveDatasetRevision: () => 2,
      resolveSelectionRevision: () => 4,
      resolveWeightsRevision: () => 3
    })
  ).toEqual(selected);
});

it('routing preserves precision, capacity and explicit complementary choices', () => {
  const request = analysisRequest();
  expect(selectKeplerExecutionRoute(request, capabilities)).toEqual({
    backend: 'luma',
    reason: 'supported'
  });
  expect(selectKeplerExecutionRoute({...request, estimatedBytes: 2048}, capabilities)).toEqual({
    backend: 'analysis-backend',
    reason: 'memory-budget'
  });
  expect(
    selectKeplerExecutionRoute(
      {...request, input: {...request.input, formats: {income: 'float64'}}},
      capabilities
    )
  ).toEqual({backend: 'analysis-backend', reason: 'precision'});
  expect(
    selectKeplerExecutionRoute(analysisRequest({...recipe, backendPolicy: 'require-gpu'}), {
      ...capabilities,
      webgpu: false
    }).backend
  ).toBe('unavailable');
  expect(
    selectKeplerExecutionRoute(
      analysisRequest({...recipe, backendPolicy: 'complementary'}),
      capabilities
    ).backend
  ).toBe('analysis-backend');
  expect(
    selectKeplerExecutionRoute({kind: 'sql', sql: 'SELECT * FROM observations'}, capabilities)
      .backend
  ).toBe('duckdb');
  expect(
    selectKeplerExecutionRoute(
      {kind: 'sql', sql: 'SELECT 1', backendPolicy: 'require-gpu'},
      capabilities
    ).backend
  ).toBe('unavailable');
  const classified: KeplerAnalysisRecipe = {
    ...recipe,
    operation: {
      kind: 'classify',
      column: 'income',
      method: 'natural-breaks',
      naturalBreaksMode: 'exact',
      classCount: 5
    }
  };
  expect(
    selectKeplerExecutionRoute(
      {
        ...analysisRequest(classified),
        input: {rowCount: 1025, chunkCount: 1, formats: {income: 'float32'}}
      },
      capabilities
    ).reason
  ).toBe('capacity');
  expect(selectKeplerExecutionRoute(analysisRequest(classified), capabilities).reason).toBe(
    'packing-required'
  );
});

it('execution never retries a failed GPU operation on another backend and disposes cancelled results', async () => {
  const fallback = vi.fn();
  await expect(
    executeKeplerRequest(analysisRequest(), capabilities, {
      luma: async () => {
        throw new Error('GPU failure');
      },
      analysisBackend: fallback,
      discardResult: () => {}
    })
  ).rejects.toThrow('GPU failure');
  expect(fallback).not.toHaveBeenCalled();
  const controller = new AbortController();
  const result = {buffer: 'owned'};
  const dispose = vi.fn();
  await expect(
    executeKeplerRequest(
      analysisRequest(),
      capabilities,
      {
        luma: async () => {
          controller.abort();
          return result;
        },
        discardResult: dispose
      },
      controller.signal
    )
  ).rejects.toThrow();
  expect(dispose).toHaveBeenCalledWith(result);
});

it('DuckDB exchanges are explicit, preserve Arrow results and record committed cancelled imports', async () => {
  const table = arrow.tableFromArrays({value: new Float32Array([1, 2, 3])});
  const query = vi.fn(async () => table);
  const insertArrowTable = vi.fn(async () => {});
  const executor = createKeplerDuckDBExecutor({query, insertArrowTable});
  const result = await executor.execute({kind: 'sql', sql: 'SELECT value FROM observations'});
  expect(result.table).toBe(table);
  expect(result.residency).toBe('cpu-arrow');
  await executor.registerArrowResult('result_v2', table, recipe.source);
  expect(insertArrowTable).toHaveBeenCalledWith(table, {name: 'result_v2'});
  expect(executor.getRelationSource('result_v2')).toEqual(recipe.source);
  await expect(executor.registerArrowResult('result_v2', table, recipe.source)).rejects.toThrow(
    'already'
  );
  expect(executor.statistics).toEqual({
    queries: 1,
    importedTables: 1,
    importedRows: 3,
    resultRows: 3
  });
  const controller = new AbortController();
  const cancelled = createKeplerDuckDBExecutor({
    query,
    insertArrowTable: async () => {
      controller.abort();
    }
  });
  await expect(
    cancelled.registerArrowResult('cancelled', table, recipe.source, controller.signal)
  ).rejects.toThrow();
  expect(cancelled.getRelationSource('cancelled')).toEqual(recipe.source);
});

it('DuckDB relation provenance is published only after import completes', async () => {
  const table = arrow.tableFromArrays({value: new Float32Array([1])});
  let finishImport: (() => void) | undefined;
  const imported = new Promise<void>(resolve => {
    finishImport = resolve;
  });
  const executor = createKeplerDuckDBExecutor({
    query: async () => table,
    insertArrowTable: async () => imported
  });
  const pending = executor.registerArrowResult('pending', table, recipe.source);
  expect(executor.getRelationSource('pending')).toBeUndefined();
  await expect(executor.registerArrowResult('pending', table, recipe.source)).rejects.toThrow(
    'already'
  );
  finishImport!();
  await pending;
  expect(executor.getRelationSource('pending')).toEqual(recipe.source);
});
