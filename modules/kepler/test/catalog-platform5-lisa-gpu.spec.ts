// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, test} from 'vitest';
import {getCatalogBinding, runCatalogOperationAsync} from '../src/analysis/catalog-bindings';
import type {CatalogFeatureCollection} from '../src/analysis/catalog-geojson';
import {requireTestDevice} from '../../analysis/src/testing/require-test-device';

// These bindings have no CPU engine: independent CPU formulas in this spec are the oracle.

async function getDevice(): Promise<Device> {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  return device;
}

const COLUMNS = 10;
const ROWS = 8;

const getSignedValue = (column: number, row: number) =>
  Math.sin(column / 2) * 3 + row * 0.7 + ((column * 13 + row * 5) % 7) * 0.2;

function makeGrid(): CatalogFeatureCollection {
  const features: CatalogFeatureCollection['features'] = [];
  for (let row = 0; row < ROWS; row++)
    for (let column = 0; column < COLUMNS; column++)
      features.push({
        geometry: {
          type: 'Polygon',
          coordinates: [
            [
              [column * 10, row * 10],
              [column * 10 + 10, row * 10],
              [column * 10 + 10, row * 10 + 10],
              [column * 10, row * 10 + 10],
              [column * 10, row * 10]
            ]
          ]
        },
        properties: {
          // Clustered black block plus noise.
          b: column < 5 && row < 4 ? 1 : (column * 7 + row * 3) % 11 === 0 ? 1 : 0,
          x: getSignedValue(column, row),
          // The same surface shifted by a constant above its minimum (about -3): Getis-Ord G is
          // defined for non-negative data only.
          xShifted: getSignedValue(column, row) + 4,
          y: Math.cos(row / 2) * 2 + column * 0.4 + ((column * 3 + row * 11) % 5) * 0.3
        }
      });
  return {type: 'FeatureCollection', features};
}

/** Queen neighbors (shared vertex) of the unit grid. */
function getQueenNeighbors(): number[][] {
  return Array.from({length: ROWS * COLUMNS}, (_, index) => {
    const row = Math.floor(index / COLUMNS);
    const column = index % COLUMNS;
    const list: number[] = [];
    for (let dr = -1; dr <= 1; dr++)
      for (let dc = -1; dc <= 1; dc++) {
        const r = row + dr;
        const c = column + dc;
        if ((dr || dc) && r >= 0 && r < ROWS && c >= 0 && c < COLUMNS) list.push(r * COLUMNS + c);
      }
    return list;
  });
}

const properties = (collection: CatalogFeatureCollection, name: string) =>
  collection.features.map(feature => (feature.properties as Record<string, number>)[name]);

test('global-join-counts-getis-ord matches independent join counts', async () => {
  const device = await getDevice();
  const features = makeGrid();
  const binding = getCatalogBinding('global-join-counts-getis-ord')!;
  expect(binding.engine).toBe('gpu');
  const result = await runCatalogOperationAsync(
    binding,
    {valueProperty: 'b', neighborhood: 'queen', coordinateSystem: 'planar'},
    {features},
    {device}
  );
  expect(result.engine).toBe('gpu');
  expect(result.dispatchedContributors).toEqual(['GPUGlobalSpatialStatistics']);
  const black = properties(features, 'b');
  const neighbors = getQueenNeighbors();
  let blackBlack = 0;
  let blackWhite = 0;
  let whiteWhite = 0;
  neighbors.forEach((list, a) =>
    list.forEach(b => {
      if (b <= a) return;
      if (black[a] && black[b]) blackBlack++;
      else if (black[a] || black[b]) blackWhite++;
      else whiteWhite++;
    })
  );
  const table = result.tables[0];
  const statistics = table.columns['statistic'] as string[];
  const values = table.columns['value'] as number[];
  expect(values[statistics.indexOf('join-count-black-black')]).toBe(blackBlack);
  expect(values[statistics.indexOf('join-count-black-white')]).toBe(blackWhite);
  expect(values[statistics.indexOf('join-count-white-white')]).toBe(whiteWhite);
  // Clustered black cells must show more black-black joins than expected.
  const z = table.columns['z_score'] as number[];
  expect(z[statistics.indexOf('join-count-black-black')]).toBeGreaterThan(2);
  expect(Number.isFinite(values[statistics.indexOf('getis-ord-g')])).toBe(true);
});

test('local-lisa-variants matches independent local statistics', async () => {
  const device = await getDevice();
  const features = makeGrid();
  const binding = getCatalogBinding('local-lisa-variants')!;
  const base = {
    neighborhood: 'queen',
    coordinateSystem: 'planar',
    permutations: 99,
    seed: 4,
    significance: 0.05,
    valueProperty: 'x'
  };
  const neighbors = getQueenNeighbors();
  const run = (extra: Record<string, unknown>) =>
    runCatalogOperationAsync(binding, {...base, ...extra}, {features}, {device});

  const join = await run({variant: 'local-join-count', valueProperty: 'b'});
  expect(join.engine, join.fallbackReason).toBe('gpu');
  const black = properties(features, 'b');
  const joinStatistic = join.tables[0].columns['statistic'] as number[];
  neighbors.forEach((list, row) =>
    expect(joinStatistic[row], `join ${row}`).toBe(
      black[row] * list.reduce((sum, other) => sum + black[other], 0)
    )
  );

  const x = properties(features, 'x');
  const xShifted = properties(features, 'xShifted');
  expect(Math.min(...x)).toBeLessThan(0);
  expect(Math.min(...xShifted)).toBeGreaterThan(0);
  const total = xShifted.reduce((sum, value) => sum + value, 0);
  const g = await run({variant: 'local-getis-ord-g', valueProperty: 'xShifted'});
  expect(g.engine, g.fallbackReason).toBe('gpu');
  const gStatistic = g.tables[0].columns['statistic'] as number[];
  for (const name of ['statistic', 'p_value', 'significant'])
    expect(g.tables[0].columns[name]).toHaveLength(ROWS * COLUMNS);
  neighbors.forEach((list, row) => {
    const expected =
      list.reduce((sum, other) => sum + xShifted[other] / list.length, 0) / (total - xShifted[row]);
    expect(Math.abs(gStatistic[row] - expected), `G ${row}`).toBeLessThan(1e-4);
  });
  // Independent Python Philox / Fisher-Yates enumeration:
  // .typecheck/r19/scratch/consumer-seams-lenses-w5-quantile-lisa/independent-counts.py
  // source row 10 has tails (93, 6), hence p=2*(6+1)/100=.14, and is nonsignificant.
  expect(Number.isFinite(gStatistic[10])).toBe(true);
  expect((g.tables[0].columns['p_value'] as number[])[10]).toBeCloseTo(0.14, 6);
  expect((g.tables[0].columns['significant'] as boolean[])[10]).toBe(false);

  // Source row 51 (canonical row 13) has tails (0,39) or (0,49), independently
  // enumerated with Python full-width Philox multiplication. Thus p=2/40=1/20 or 2/50=1/25.
  const boundary = await run({
    variant: 'local-getis-ord-g',
    valueProperty: 'xShifted',
    permutations: 39
  });
  expect(boundary.engine, boundary.fallbackReason).toBe('gpu');
  for (const name of ['statistic', 'p_value', 'significant'])
    expect(boundary.tables[0].columns[name]).toHaveLength(ROWS * COLUMNS);
  expect(Number.isFinite((boundary.tables[0].columns['statistic'] as number[])[51])).toBe(true);
  expect((boundary.tables[0].columns['p_value'] as number[])[51]).toBe(Math.fround(0.05));
  expect((boundary.tables[0].columns['significant'] as boolean[])[51]).toBe(true);

  // The packed .04 cutoff
  // is 0.03999999910593033, strictly smaller than 1/25, so it must not reject.
  const smallerCutoff = await run({
    variant: 'local-getis-ord-g',
    valueProperty: 'xShifted',
    permutations: 49,
    significance: 0.04
  });
  expect(smallerCutoff.engine, smallerCutoff.fallbackReason).toBe('gpu');
  for (const name of ['statistic', 'p_value', 'significant'])
    expect(smallerCutoff.tables[0].columns[name]).toHaveLength(ROWS * COLUMNS);
  expect(Number.isFinite((smallerCutoff.tables[0].columns['statistic'] as number[])[51])).toBe(
    true
  );
  expect((smallerCutoff.tables[0].columns['p_value'] as number[])[51]).toBe(Math.fround(0.04));
  expect((smallerCutoff.tables[0].columns['significant'] as boolean[])[51]).toBe(false);

  const star = await run({variant: 'local-getis-ord-g-star', valueProperty: 'xShifted'});
  expect(star.engine, star.fallbackReason).toBe('gpu');
  const starStatistic = star.tables[0].columns['statistic'] as number[];
  neighbors.forEach((list, row) => {
    // A self weight of 1 plus the row-standardized neighbor weights of the binding (1 / k each),
    // over the total, as in the documented G*_i = (x_i + sum_s w_s x_s) / sum x.
    const lag = list.reduce((sum, other) => sum + xShifted[other] / list.length, 0);
    const expected = (xShifted[row] + lag) / total;
    expect(Math.abs(starStatistic[row] - expected) / expected, `G* ${row}`).toBeLessThan(1e-3);
  });
  const starP = star.tables[0].columns['p_value'] as number[];
  expect(starP.every(value => value > 0 && value <= 1)).toBe(true);

  // Negative data is rejected by G and G*: the binding fails (it has no CPU engine to fall back
  // to) and the error names the cause.
  for (const variant of ['local-getis-ord-g', 'local-getis-ord-g-star']) {
    await expect(run({variant})).rejects.toThrow(/Getis-Ord local G requires non-negative values/);
  }

  const y = properties(features, 'y');
  const mean = (values: number[]) => values.reduce((a, b) => a + b, 0) / values.length;
  const mx = mean(x);
  const my = mean(y);
  const ssx = x.reduce((sum, value) => sum + (value - mx) ** 2, 0);
  const ssy = y.reduce((sum, value) => sum + (value - my) ** 2, 0);
  const n = x.length;
  const bivariate = await run({variant: 'local-bivariate-moran', secondValueProperty: 'y'});
  expect(bivariate.engine, bivariate.fallbackReason).toBe('gpu');
  const bivariateStatistic = bivariate.tables[0].columns['statistic'] as number[];
  neighbors.forEach((list, row) => {
    const lag = list.reduce((sum, other) => sum + (y[other] - my) / list.length, 0);
    const expected = ((n - 1) * (x[row] - mx) * lag) / Math.sqrt(ssx * ssy);
    expect(Math.abs(bivariateStatistic[row] - expected), `BV ${row}`).toBeLessThan(2e-3);
  });
});
