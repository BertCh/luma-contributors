// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import {
  AGGREGATION_EXTRUDED_HIGHLIGHT_COLOR,
  getAggregateCellOutline,
  getAggregationHighlightProps,
  getAggregationOutlineWidth
} from '../../../examples/deck/kepler-compute/src/aggregation-hover';
import {
  getCategoricalOrdinalLookup,
  getJenksBreaks,
  getOrdinalLookup,
  getReferenceJenksIndex,
  getReferenceOrdinalIndex,
  getScaleLookupLength,
  packScaleLookup,
  SCALE_LOOKUP_WGSL
} from '../../../examples/deck/kepler-compute/src/aggregation-scale-lookup';
import {makeAggregateFootprints} from '../../../examples/deck/kepler-compute/src/aggregation-runtime';

describe('jenks breaks', () => {
  test('returns k - 1 sorted thresholds that separate obvious clusters deterministically', () => {
    const values = [1, 1.1, 1.2, 10, 10.1, 10.2, 50, 50.5, 51];
    const breaks = getJenksBreaks(values, 3);
    expect(breaks).toHaveLength(2);
    expect(breaks).toEqual(getJenksBreaks(values, 3));
    expect(breaks[0]).toBe(10);
    expect(breaks[1]).toBe(50);
    expect(values.map(value => getReferenceJenksIndex(value, breaks))).toEqual([
      0, 0, 0, 1, 1, 1, 2, 2, 2
    ]);
  });
  test('pads when there are fewer distinct values than classes and handles empty input', () => {
    expect(getJenksBreaks([5, 5, 7], 4)).toEqual([7, 7, 7]);
    expect(getJenksBreaks([], 4)).toEqual([]);
    expect(getJenksBreaks([1, 2, 3], 1)).toEqual([]);
    expect(getReferenceJenksIndex(NaN, [1])).toBeNaN();
    expect(getReferenceJenksIndex(100, [1, 2])).toBe(2);
  });
});

describe('ordinal lookup', () => {
  test('ranks distinct finite values by deck default (string) sort order', () => {
    const lookup = getOrdinalLookup([10, 9, NaN, 10, 2, Infinity]);
    // Numeric order is [2, 9, 10]; deck's comparator-less sort orders them "10" < "2" < "9".
    expect(lookup.values).toEqual([2, 9, 10]);
    expect(lookup.ranks).toEqual([1, 2, 0]);
    expect(getReferenceOrdinalIndex(10, lookup)).toBe(0);
    expect(getReferenceOrdinalIndex(7, lookup)).toBeNaN();
  });
  test('packs jenks and ordinal tables behind a ready header', () => {
    const length = getScaleLookupLength(4);
    expect(length).toBe(65);
    const jenks = packScaleLookup('jenks', [3, 8], length);
    expect(Array.from(jenks.subarray(0, 4))).toEqual([1, 2, 3, 8]);
    const ordinal = packScaleLookup('ordinal', getOrdinalLookup([2, 9, 10]), length);
    expect(Array.from(ordinal.subarray(0, 8))).toEqual([1, 3, 2, 9, 10, 1, 2, 0]);
    expect(SCALE_LOOKUP_WGSL).toContain('fn getJenksIndex');
    expect(SCALE_LOOKUP_WGSL).toContain('fn getOrdinalIndex');
  });
});

describe('aggregate hover styling', () => {
  test('extruded cells auto-highlight in translucent white; flat cells use an outline instead', () => {
    expect(getAggregationHighlightProps(true)).toEqual({
      autoHighlight: true,
      highlightColor: AGGREGATION_EXTRUDED_HIGHLIGHT_COLOR
    });
    expect(getAggregationHighlightProps(false)).toEqual({autoHighlight: false});
    expect(getAggregationOutlineWidth(4)).toBe(32);
  });
  test('cell outlines are the closed full-coverage footprints used for rendering', () => {
    const grid = {
      kind: 'grid',
      gridSize: [3, 2],
      bounds: [0, 0, 30, 20],
      cellSize: [10, 10],
      cellCount: 6
    } as never;
    const hexagon = {
      kind: 'hexagon',
      gridSize: [3, 2],
      bounds: [0, 0, 40, 30],
      radius: 6,
      cellCount: 6
    } as never;
    for (const layout of [grid, hexagon] as const) {
      const vertexCount = (layout as {kind: string}).kind === 'grid' ? 4 : 6;
      const footprints = makeAggregateFootprints(layout, 1);
      for (const cell of [0, 4]) {
        const outline = getAggregateCellOutline(layout, cell);
        expect(outline).toHaveLength(vertexCount + 1);
        expect(outline[vertexCount]).toEqual(outline[0]);
        for (let vertex = 0; vertex < vertexCount; vertex++) {
          expect(outline[vertex][0]).toBeCloseTo(footprints[(cell * vertexCount + vertex) * 4], 4);
          expect(outline[vertex][1]).toBeCloseTo(
            footprints[(cell * vertexCount + vertex) * 4 + 1],
            4
          );
        }
      }
    }
  });
});

describe('categorical ordinal lookup', () => {
  test('indexes codes by first appearance in cell order and ranks the indices in string order', () => {
    // Codes 7, 3, 7, 9 appear as native indices 0, 1, 0, 2.
    const lookup = getCategoricalOrdinalLookup([7, NaN, 3, 7, 9]);
    expect(lookup.values).toEqual([3, 7, 9]);
    expect(lookup.ranks).toEqual([1, 0, 2]);
  });
});
