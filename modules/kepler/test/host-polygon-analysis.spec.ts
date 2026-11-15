// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import * as arrow from 'apache-arrow';
import {Buffer} from '@luma.gl/core';
import {prepareKeplerGeometrySource} from '../../../examples/deck/kepler-compute/src/geometry-preparation';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {KeplerComputeSession} from '../src/session/compute-session';
import type {
  KeplerPolygonAnalysisOperation,
  KeplerPolygonWeightsSettings
} from '../src/recipes/analysis-recipe';
import {AnalysisRuntime} from '../../../examples/deck/kepler-compute/src/analysis-runtime';
import {requireTestDevice} from '../../analysis/test/utils/require-test-device';

function square(left: number, bottom: number) {
  return {
    type: 'Polygon',
    coordinates: [
      [
        [left, bottom],
        [left + 1, bottom],
        [left + 1, bottom + 1],
        [left, bottom + 1],
        [left, bottom]
      ]
    ]
  };
}

test('polygon analysis keeps original rows, rejects incomplete topology and reuses queen/rook weights through inference and replay', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 64 * 1024 * 1024});
  const geometries = [square(0, 0), square(1, 0), square(2, 1), square(10, 10), null];
  const first = new arrow.Table({
    events: arrow.makeVector(new Float32Array([10, 20])),
    population: arrow.makeVector(new Float32Array([100, 100])),
    row: arrow.makeVector(new Float32Array([0, 1])),
    geometry: arrow.vectorFromArray(
      geometries.slice(0, 2).map(value => JSON.stringify(value)),
      new arrow.Utf8()
    )
  });
  const second = new arrow.Table({
    events: arrow.makeVector(new Float32Array([60, 90, 1000])),
    population: arrow.makeVector(new Float32Array([100, 100, 100])),
    row: arrow.makeVector(new Float32Array([2, 3, 4])),
    geometry: arrow.vectorFromArray(
      geometries.slice(2).map(value => (value === null ? null : JSON.stringify(value))),
      new arrow.Utf8()
    )
  });
  const dataset = session.importArrowDataset({id: 'polygon-analysis', table: first.concat(second)});
  const explorer = session.openExplorer(dataset.id, {
    workingMemoryBudgetBytes: 8 * 1024 * 1024,
    dimensions: [{id: 'row', column: 'row', kind: 'range'}]
  });
  const prepared = prepareKeplerGeometrySource([
    {id: 'first', rows: geometries.slice(0, 2)},
    {id: 'second', rows: geometries.slice(2)}
  ]);
  const analysis = new AnalysisRuntime({
    session,
    explorer,
    requestUpdate: () => {},
    polygonSource: {column: 'geometry', source: prepared}
  });
  const weights: KeplerPolygonWeightsSettings = {
    kind: 'contiguity',
    geometryColumn: 'geometry',
    criterion: 'queen',
    equality: 'exact-source-xy',
    neighborCapacity: 4,
    pairCapacity: 64
  };
  const rates: KeplerPolygonAnalysisOperation = {
    kind: 'spatial-rate-polygons',
    events: 'events',
    populations: 'population',
    weights,
    scale: 100,
    islandPolicy: 'undefined'
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
    analysis.configurePolygon({...rates, weights: {...weights, pairCapacity: 1}});
    expect(analysis.active).toBe(false);
    await render();
    expect(analysis.error).toMatch(/candidate.*lower bound/i);
    expect(analysis.active).toBe(false);
    expect(analysis.summary).toBeNull();
    expect(analysis.getTableColumns()).toBeUndefined();
    expect(await analysis.readRow(0)).toBeNull();

    analysis.configurePolygon({...rates, weights: {...weights, neighborCapacity: 1}});
    await render();
    expect(analysis.error).toMatch(/neighbor storage overflowed; 4 slots are required/);
    expect(analysis.active).toBe(false);
    expect(analysis.getTableColumns()).toBeUndefined();

    analysis.configurePolygon(rates);
    const queenRecipe = analysis.serializeRecipe();
    await render();
    expect(analysis.error).toBe('');
    expect(analysis.active).toBe(true);
    expect(analysis.summary?.polygonWeights).toEqual({
      weightsOverflow: 0,
      weightsRequiredCount: 4,
      candidateOverflow: 0,
      invalidCount: 1
    });
    // Queen: A—B—C; B/C share only one corner. D is isolated and E has no polygon.
    await expectRates([15, 30, 40, NaN, NaN]);
    expect(
      analysis.getTableColumns()?.columns['analysis_spatial_rate'].data.map(chunk => chunk.length)
    ).toEqual([2, 3]);
    expect(dataset.value.columns['events'].data.map(chunk => chunk.length)).toEqual([2, 3]);
    const queenCache = analysis.polygonWeightsCacheStats;

    analysis.setPopulation('selected');
    explorer.setRange('row', 0, 1);
    expect(analysis.active).toBe(false);
    await render();
    expect(analysis.error).toBe('');
    await expectRates([15, 15, NaN, NaN, NaN]);
    expect(analysis.summary?.polygonWeights?.weightsRequiredCount).toBe(4);
    expect(analysis.polygonWeightsCacheStats).toMatchObject({
      buildCount: queenCache.buildCount,
      encodeCount: queenCache.encodeCount
    });
    analysis.restoreRecipe(queenRecipe);
    await render();
    expect(analysis.settings.population).toBe('all');
    await expectRates([15, 30, 40, NaN, NaN]);
    expect(analysis.polygonWeightsCacheStats).toMatchObject({
      buildCount: queenCache.buildCount,
      encodeCount: queenCache.encodeCount
    });

    analysis.configurePolygon({...rates, weights: {...weights, criterion: 'rook'}});
    const rookRecipe = analysis.serializeRecipe();
    await render();
    expect(analysis.error).toBe('');
    expect(analysis.summary?.polygonWeights?.weightsRequiredCount).toBe(2);
    await expectRates([15, 15, NaN, NaN, NaN]);
    analysis.setMode('off');
    analysis.restoreRecipe(rookRecipe);
    await render();
    expect(analysis.settings.polygon?.weights).toEqual({...weights, criterion: 'rook'});
    await expectRates([15, 15, NaN, NaN, NaN]);

    const moran: KeplerPolygonAnalysisOperation = {
      kind: 'local-moran-polygons',
      column: 'events',
      weights,
      minimumNeighbors: 1,
      islands: 'exclude',
      multipleTesting: 'none',
      significanceLevel: 0.9999,
      inference: 'analytic-conditional-normal-two-sided'
    };
    analysis.configurePolygon(moran);
    await render();
    expect(analysis.error).toBe('');
    expect(analysis.summary?.operation).toBe('local-moran-polygons');
    // n=3, centered values [-20,-10,30], sumSquares=1400, binary lags [-10,10,-10].
    for (const [sourceRowIndex, expected] of [2 / 7, -1 / 7, -3 / 7].entries())
      expect((await analysis.readRow(sourceRowIndex))?.value).toBeCloseTo(expected, 5);
    expect((await analysis.readRow(3))?.value).toBeNaN();
    expect((await analysis.readRow(4))?.value).toBeNaN();
    const table = analysis.getTableColumns()!;
    expect(Object.keys(table.columns)).toHaveLength(7);
    for (const vector of Object.values(table.columns))
      expect(vector.data.map(chunk => chunk.length)).toEqual([2, 3]);
    const moranCache = analysis.polygonWeightsCacheStats;

    analysis.configurePolygon({
      ...moran,
      inference: 'permutation-conditional-two-sided',
      permutationCount: 99,
      seed: 7
    });
    const permutationRecipe = analysis.serializeRecipe();
    await render();
    expect(analysis.error).toBe('');
    const readProbabilities = async () => {
      const vector = analysis.getTableColumns()!.columns['analysis_local_moran_p_value'];
      const buffer = vector.data[0].buffer;
      if (!(buffer instanceof Buffer)) throw new Error('Expected owned probability buffer');
      return new Float32Array((await buffer.readAsync(0, 20)).buffer);
    };
    const probabilities = await readProbabilities();
    for (const probability of probabilities.slice(0, 3)) {
      expect(probability).toBeGreaterThanOrEqual(Math.fround(2 / 100));
      expect(probability).toBeLessThanOrEqual(1);
    }
    expect(probabilities[3]).toBeNaN();
    expect(probabilities[4]).toBeNaN();
    analysis.setMode('off');
    analysis.restoreRecipe(permutationRecipe);
    await render();
    expect(Array.from(await readProbabilities())).toEqual(Array.from(probabilities));
    expect(analysis.polygonWeightsCacheStats).toMatchObject({
      buildCount: moranCache.buildCount,
      encodeCount: moranCache.encodeCount
    });

    analysis.configurePolygon({
      kind: 'hot-spots-polygons',
      column: 'events',
      weights,
      minimumNeighbors: 1,
      selfWeight: 1,
      multipleTesting: 'none',
      islands: 'exclude',
      inference: 'analytic-two-sided'
    });
    await render();
    expect(analysis.error).toBe('');
    expect(analysis.summary?.operation).toBe('hot-spots-polygons');
    expect(analysis.summary?.polygonWeights?.weightsRequiredCount).toBe(4);
    expect(
      analysis
        .getTableColumns()
        ?.columns['analysis_hot_spot_p_value'].data.map(chunk => chunk.length)
    ).toEqual([2, 3]);
    expect(analysis.polygonWeightsCacheStats).toMatchObject({
      buildCount: moranCache.buildCount,
      encodeCount: moranCache.encodeCount
    });
    analysis.configurePolygon({
      kind: 'quantile-lisa-polygons',
      column: 'events',
      weights,
      quantileCount: 2,
      selectedQuantile: 1,
      seed: 7,
      permutationCount: 99,
      significanceLevel: 0.9999,
      inference: 'legacy-directed-pseudo',
      multipleTesting: 'none',
      islands: 'exclude',
      quantileMethod: 'type-5'
    });
    const quantileRecipe = analysis.serializeRecipe();
    await render();
    expect(analysis.error).toBe('');
    expect(analysis.summary?.operation).toBe('quantile-lisa-polygons');
    // D remains in the four-value type-5 distribution; geometryless E=1000 is excluded first.
    // The lowest quantile has an exterior -Infinity bound, with median (20 + 60) / 2 = 40.
    expect(analysis.summary?.quantileLisa).toEqual({
      boundaries: [-Infinity, 40],
      validCount: 4,
      islandCount: 1,
      invalidQuantile: 0,
      overflow: 0
    });
    expect(analysis.summary?.count).toBe(2);
    const quantileTable = analysis.getTableColumns()!;
    expect(Object.keys(quantileTable.columns)).toHaveLength(6);
    for (const vector of Object.values(quantileTable.columns))
      expect(vector.data.map(chunk => chunk.length)).toEqual([2, 3]);
    const readQuantileColumn = async (name: string, format: 'float32' | 'uint32') => {
      const buffer = analysis.getTableColumns()!.columns[name].data[0].buffer;
      if (!(buffer instanceof Buffer)) throw new Error('Expected owned quantile output buffer');
      const bytes = await buffer.readAsync(0, 20);
      return format === 'float32' ? new Float32Array(bytes.buffer) : new Uint32Array(bytes.buffer);
    };
    expect(
      Array.from(await readQuantileColumn('analysis_quantile_lisa_membership', 'uint32'))
    ).toEqual([1, 1, 0, 0, 0]);
    const joins = await readQuantileColumn('analysis_quantile_lisa_join_count', 'float32');
    expect(Array.from(joins.slice(0, 2))).toEqual([1, 1]);
    const quantileProbabilities = await readQuantileColumn(
      'analysis_quantile_lisa_pseudo_p_value',
      'float32'
    );
    for (const probability of quantileProbabilities.slice(0, 2)) {
      expect(probability).toBeGreaterThanOrEqual(Math.fround(1 / 100));
      expect(probability).toBeLessThanOrEqual(0.5);
    }
    for (const probability of quantileProbabilities.slice(2)) expect(probability).toBeNaN();
    expect(analysis.polygonWeightsCacheStats).toMatchObject({
      buildCount: moranCache.buildCount,
      encodeCount: moranCache.encodeCount
    });
    analysis.setPopulation('selected');
    explorer.setRange('row', 0, 1);
    await render();
    expect(analysis.error).toBe('');
    expect(analysis.summary?.quantileLisa).toMatchObject({
      boundaries: [-Infinity, 15],
      validCount: 2,
      islandCount: 0
    });
    expect(analysis.summary?.polygonWeights?.weightsRequiredCount).toBe(4);
    expect(analysis.summary?.count).toBe(0);
    analysis.restoreRecipe(quantileRecipe);
    await render();
    expect(analysis.error).toBe('');
    expect(
      Array.from(await readQuantileColumn('analysis_quantile_lisa_pseudo_p_value', 'float32'))
    ).toEqual(Array.from(quantileProbabilities));
    expect(analysis.polygonWeightsCacheStats).toMatchObject({
      buildCount: moranCache.buildCount,
      encodeCount: moranCache.encodeCount
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
