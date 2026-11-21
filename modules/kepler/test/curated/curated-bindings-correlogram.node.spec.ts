// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {computeContiguityRings} from '../../src/curated/bindings/curated-contiguity-correlogram-binding';
import {CORRELOGRAM_LATTICE} from './curated-bindings-oracle-data';
import {
  getNumbers,
  getResultTable,
  makeLattice,
  recordBindingMargin,
  runCuratedBinding
} from './curated-bindings-test-support';

// Oracle: /usr/bin/python3 .typecheck/b1-oracles.py (CORRELOGRAM_LATTICE): an 8 by 8 lattice with a
// smooth surface and noise. Rings are the shortest-path distance classes of the queen graph
// (scipy.sparse.csgraph.shortest_path), each row-standardised with islands of the order left out;
// Moran's I, its expectation -1 / (n - 1) and the randomization variance of Cliff and Ord (1981)
// are computed directly, and the p-value is 2 * scipy.stats.norm.sf(|z|).

const ID = 'curated-contiguity-correlogram';
const PLANAR = {coordinateSystem: 'planar'};

function runCorrelogram(
  values: readonly number[],
  parameters: Record<string, unknown> = {},
  columns = 8
) {
  const rows = values.length / columns;
  const areas = makeLattice(columns, rows, index => ({v: values[index]}));
  return runCuratedBinding(
    ID,
    {valueProperty: 'v', maximumOrder: 5, permutations: 99, seed: 2, ...PLANAR, ...parameters},
    {areas}
  );
}

describe('computeContiguityRings', () => {
  it('gives the exact-order rings of a path graph, excluding the area itself and nearer areas', () => {
    const path = [[1], [0, 2], [1, 3], [2, 4], [3]];
    const rings = computeContiguityRings(path, 3);
    expect(rings[0]).toEqual(path);
    expect(rings[1]).toEqual([[2], [3], [0, 4], [1], [2]]);
    expect(rings[2]).toEqual([[3], [4], [], [0], [1]]);
  });

  it('leaves rows of a separate component with empty rings and stops when the graph runs out', () => {
    const rings = computeContiguityRings([[1], [0], []], 2);
    expect(rings[0]).toEqual([[1], [0], []]);
    expect(rings[1]).toEqual([[], [], []]);
  });
});

describe('curated-contiguity-correlogram against NumPy and SciPy (8 by 8 lattice)', () => {
  const result = runCorrelogram(CORRELOGRAM_LATTICE.values);
  const table = getResultTable(result, 'correlogram');

  it("reproduces Moran's I of each order to 1e-12, with the same sample and link counts", () => {
    expect(table.rowCount).toBe(5);
    CORRELOGRAM_LATTICE.orders.forEach((expected, index) => {
      const error = Math.abs(Number(table.columns['morans_i'][index]) - expected.moransI);
      recordBindingMargin(`correlogram order ${expected.order} Moran I abs error`, error);
      expect(error, `order ${expected.order}`).toBeLessThan(1e-12);
      expect(table.columns['order'][index]).toBe(expected.order);
      expect(table.columns['area_count'][index]).toBe(expected.count);
      expect(table.columns['link_count'][index]).toBe(expected.links);
      expect(Math.abs(Number(table.columns['expected_i'][index]) - expected.expected)).toBeLessThan(
        1e-15
      );
    });
    // At order 5 the four central areas are within four steps of every area, so they have no ring
    // and 60 areas take part.
    expect(table.columns['area_count'][4]).toBe(60);
    expect(table.columns['island_count'][4]).toBe(4);
  });

  it('reproduces the randomization variance and z-score to 1e-10, and the normal p-value to 1e-6', () => {
    CORRELOGRAM_LATTICE.orders.forEach((expected, index) => {
      const varianceError = Math.abs(
        Number(table.columns['variance'][index]) / expected.variance - 1
      );
      const zError = Math.abs(Number(table.columns['z_score'][index]) / expected.zScore - 1);
      recordBindingMargin(
        `correlogram order ${expected.order} variance relative error`,
        varianceError
      );
      recordBindingMargin(`correlogram order ${expected.order} z relative error`, zError);
      expect(varianceError).toBeLessThan(1e-10);
      expect(zError).toBeLessThan(1e-10);
      // The binding's normal tail is the Abramowitz and Stegun 26.2.17 approximation, good to 7.5e-8.
      const pError = Math.abs(Number(table.columns['p_value'][index]) - expected.pValue);
      recordBindingMargin(`correlogram order ${expected.order} p abs error`, pError);
      expect(pError).toBeLessThan(1e-6);
    });
  });

  it('decays with order on the smooth surface: strongly positive first, then falling, ending negative', () => {
    const moran = getNumbers(table, 'morans_i');
    expect(moran[0]).toBeGreaterThan(0.7);
    for (let index = 1; index < moran.length; index++)
      expect(moran[index]).toBeLessThan(moran[index - 1]);
    expect(moran[4]).toBeLessThan(0);
    recordBindingMargin('correlogram order 1 Moran I', moran[0]);
  });

  it('gives permutation pseudo p-values that are small where the association is strong, repeat for a seed and never fall under 1 / 100', () => {
    const pseudo = getNumbers(table, 'pseudo_p_value');
    expect(pseudo[0]).toBe(1 / 100);
    expect(table.columns['significant'][0]).toBe(true);
    for (const p of pseudo) {
      expect(p).toBeGreaterThanOrEqual(1 / 100);
      expect(Math.abs(p * 100 - Math.round(p * 100))).toBeLessThan(1e-9);
    }
    expect(getResultTable(runCorrelogram(CORRELOGRAM_LATTICE.values), 'correlogram')).toEqual(
      table
    );
  });
});

describe('curated-contiguity-correlogram declared behaviour', () => {
  it('leaves an island out of every order, counted, without changing the statistics of the rest', () => {
    const areas = makeLattice(8, 8, index => ({v: CORRELOGRAM_LATTICE.values[index]}));
    areas.features.push({
      id: 'island',
      geometry: {
        type: 'Polygon',
        coordinates: [
          [
            [50, 50],
            [51, 50],
            [51, 51],
            [50, 51],
            [50, 50]
          ]
        ]
      },
      properties: {v: 123}
    });
    const result = runCuratedBinding(
      ID,
      {valueProperty: 'v', maximumOrder: 5, permutations: 99, ...PLANAR},
      {areas}
    );
    const table = getResultTable(result, 'correlogram');
    CORRELOGRAM_LATTICE.orders.forEach((expected, index) => {
      expect(Math.abs(Number(table.columns['morans_i'][index]) - expected.moransI)).toBeLessThan(
        1e-12
      );
      expect(table.columns['island_count'][index]).toBe(index === 4 ? 5 : 1);
    });
    expect(result.diagnostics.find(item => item.code === 'island-features')?.count).toBe(1);
  });

  it('leaves orders beyond the reach of the graph empty and counts them in a diagnostic', () => {
    const values = Array.from({length: 16}, (_, index) => Math.sin(index));
    const result = runCorrelogram(values, {maximumOrder: 8}, 4);
    const table = getResultTable(result, 'correlogram');
    // On a 4 by 4 lattice the queen graph has diameter 3: orders 4 to 8 have no ring.
    expect(Number(table.columns['area_count'][2])).toBeGreaterThan(0);
    expect(table.columns['area_count'].slice(3)).toEqual([0, 0, 0, 0, 0]);
    expect(table.columns['morans_i'].slice(4)).toEqual([null, null, null, null]);
    expect(
      result.diagnostics.find(item => item.code === 'order-not-computed')?.count
    ).toBeGreaterThanOrEqual(4);
    expect(result.summary['computedOrderCount']).toBeLessThanOrEqual(4);
  });

  it('lists skipped rows by reason, and throws named errors for constant values, too few areas and unnamed columns', () => {
    const areas = makeLattice(8, 8, index => ({v: CORRELOGRAM_LATTICE.values[index]}));
    areas.features[0].properties = {v: 'x'};
    areas.features[1].properties = {v: Number.NaN};
    areas.features.push({id: 'none', geometry: null, properties: {v: 1}});
    const result = runCuratedBinding(
      ID,
      {valueProperty: 'v', permutations: 99, ...PLANAR},
      {areas}
    );
    expect(result.diagnostics.find(item => item.code === 'missing-value')?.count).toBe(2);
    expect(result.diagnostics.find(item => item.code === 'missing-geometry')?.count).toBe(1);
    expect(result.summary['areaCount']).toBe(62);
    expect(() => runCorrelogram(new Array(16).fill(2), {}, 4)).toThrow(
      /the values of v are all equal/
    );
    expect(() => runCorrelogram([1, 2, 3, 4], {}, 2)).toThrow(/4 usable areas; at least 5/);
    expect(() => runCorrelogram(CORRELOGRAM_LATTICE.values, {valueProperty: ''})).toThrow(
      /parameter valueProperty must name a column/
    );
    expect(() => runCorrelogram(CORRELOGRAM_LATTICE.values, {maximumOrder: 13})).toThrow(
      /maximumOrder must be at most 12; received 13/
    );
    expect(() => runCorrelogram(CORRELOGRAM_LATTICE.values, {maximumOrder: 0})).toThrow(
      /maximumOrder must be at least 1; received 0/
    );
  });

  it('throws a capacity error when the permutation work is too large', () => {
    const values = Array.from({length: 2500}, (_, index) => Math.sin(index / 7));
    expect(() => runCorrelogram(values, {maximumOrder: 12, permutations: 9999}, 50)).toThrow(
      /exceed the 20000000/
    );
  });

  it('works with rook contiguity, which links fewer areas', () => {
    const queen = getResultTable(
      runCorrelogram(CORRELOGRAM_LATTICE.values, {maximumOrder: 1}),
      'correlogram'
    );
    const rook = getResultTable(
      runCorrelogram(CORRELOGRAM_LATTICE.values, {maximumOrder: 1, criterion: 'rook'}),
      'correlogram'
    );
    expect(Number(rook.columns['link_count'][0])).toBe(224);
    expect(Number(queen.columns['link_count'][0])).toBe(420);
  });
});
