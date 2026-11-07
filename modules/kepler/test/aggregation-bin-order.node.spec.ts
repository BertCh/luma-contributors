// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import {
  getCategoricalOrdinalLookup,
  getFirstAppearanceIndices,
  getOrdinalLookup
} from '../../../examples/deck/kepler-compute/src/aggregation-scale-lookup';
import {
  getH3HighlightProps,
  getH3HoveredOutline,
  getH3OutlineWidthScale
} from '../../../examples/deck/kepler-compute/src/h3-highlight';

describe('native bin order (first source row per cell)', () => {
  test('labels are indexed by the first appearance of their earliest cell', () => {
    // Cells in id order; first source rows decide the native bin order, not the cell id.
    const firstRows = [5, 0, 9, 2, 7];
    const codes = [1, 2, 0, 1, 2];
    const indices = getFirstAppearanceIndices(firstRows, codes);
    // Bin order by first row: cell1(code2), cell3(code1), cell0(code1), cell4(code2), cell2(code0).
    expect(Array.from(indices)).toEqual([
      [2, 0],
      [1, 1],
      [0, 2]
    ]);
  });
  test('cells without a first row or a finite code are skipped', () => {
    const indices = getFirstAppearanceIndices([3, 4294967295, 1, 2], [7, 8, NaN, 9]);
    expect(Array.from(indices)).toEqual([
      [9, 0],
      [7, 1]
    ]);
  });
  test('ranking first-appearance indices equals the categorical oracle for ascending first rows', () => {
    const codes = [4, 1, 4, 3, 1, 2];
    const firstRows = codes.map((_, cell) => cell);
    const indices = getFirstAppearanceIndices(firstRows, codes);
    const mapped = codes.map(code => indices.get(code)!);
    const gpuStyle = getOrdinalLookup(mapped);
    const oracle = getCategoricalOrdinalLookup(codes);
    const rankOf = (code: number) => oracle.ranks[oracle.values.indexOf(code)];
    for (let cell = 0; cell < codes.length; cell++)
      expect(gpuStyle.ranks[gpuStyle.values.indexOf(mapped[cell])]).toBe(rankOf(codes[cell]));
  });
});

describe('H3 hovered outline', () => {
  const source = {
    batches: [
      {sourceRowOffset: 0, rowCount: 2},
      {sourceRowOffset: 2, rowCount: 3}
    ]
  };
  const getBoundary = (batchIndex: number, rowIndex: number): [number, number][] =>
    rowIndex === 2 ? [] : [[batchIndex, rowIndex]];
  test('resolves the batch of a source row and skips invalid rows', () => {
    expect(getH3HoveredOutline(source, 3, false, getBoundary)).toEqual([[1, 1]]);
    expect(getH3HoveredOutline(source, 4, false, getBoundary)).toBeUndefined();
    expect(getH3HoveredOutline(source, 9, false, getBoundary)).toBeUndefined();
    expect(getH3HoveredOutline(source, -1, false, getBoundary)).toBeUndefined();
  });
  test('native draws no outline when a size field is set', () => {
    expect(getH3HoveredOutline(source, 0, true, getBoundary)).toBeUndefined();
  });
  test('flat cells are not auto-highlighted and width follows native zoom scaling', () => {
    expect(getH3HighlightProps(false, false).autoHighlight).toBe(false);
    expect(getH3HighlightProps(true, false).autoHighlight).toBe(true);
    expect(getH3OutlineWidthScale(0.5)).toBe(4);
  });
});

describe('aggregation count labels', () => {
  test('labels nonempty visible cells at their geographic centers', async () => {
    const {getAggregateCountLabels} = await import(
      '../../../examples/deck/kepler-compute/src/aggregation-count-labels'
    );
    const layout = {
      kind: 'grid' as const,
      bounds: [0, 0, 20, 10] as const,
      gridSize: [2, 1] as const,
      cellCount: 2,
      cellSize: [10, 10] as const
    };
    const unitsPerMeter = 1e-4;
    const origin = [256, 256, 0] as const;
    const all = getAggregateCountLabels(layout, [3, 0], undefined, origin, unitsPerMeter);
    expect(all).toHaveLength(1);
    expect(all[0].count).toBe(3);
    expect(Math.abs(all[0].position[0])).toBeLessThan(1);
    expect(getAggregateCountLabels(layout, [3, 4], [0, 1], origin, unitsPerMeter)).toHaveLength(1);
  });
});
