// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  makeColumnClipboardText,
  makeColumnStatisticRows
} from '../../../examples/deck/kepler-compute/src/resident-table-column-tools';

test('numeric statistics list nulls, mean, deviation and quantiles with a median', () => {
  const rows = makeColumnStatisticRows({
    column: 'x',
    population: 'selected',
    fieldType: 'real',
    rowCount: 4,
    nullCount: 1,
    stats: {
      type: 'numeric',
      bins: [],
      mean: 2.5,
      std: 1,
      percentNulls: 0.25,
      quantiles: [
        {label: 'Min', value: 1},
        {label: '50%', value: 2},
        {label: 'Max', value: 4}
      ]
    } as never
  });
  expect(rows).toContainEqual(['Nulls', '1 (25%)']);
  expect(rows).toContainEqual(['Median (50%)', '2']);
  expect(rows).toContainEqual(['Mean', '2.5']);
});

test('clipboard text keeps one value per line with empty missing cells', () => {
  expect(makeColumnClipboardText(['a', null, 2n, 3])).toBe('a\n\n2\n3');
});
