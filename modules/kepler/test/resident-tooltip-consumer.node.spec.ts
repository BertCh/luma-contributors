// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  readConsumerState,
  reconcileConsumerState,
  makeReadyState
} from '../../../examples/deck/kepler-compute/src/resident-consumer-state';
import {
  formatTooltipValue,
  resolveTooltipFields
} from '../../../examples/deck/kepler-compute/src/resident-tooltip-fields';
import {
  compareTooltipValues,
  formatTooltipDifference
} from '../../../examples/deck/kepler-compute/src/resident-tooltip-compare';
import {
  ResidentBrushMask,
  isPointBrushed
} from '../../../examples/deck/kepler-compute/src/resident-brush-mask';

test('consumer state drops values when the revision moves', async () => {
  let revision = 'a';
  const state = await readConsumerState(
    'tooltip',
    () => revision,
    async () => {
      revision = 'b';
      return {x: 1};
    }
  );
  expect(state.status).toBe('stale');
  expect(reconcileConsumerState(makeReadyState('table', 'a', 1), 'b').status).toBe('stale');
  expect(
    (
      await readConsumerState(
        'chart',
        () => 'a',
        async () => null
      )
    ).status
  ).toBe('empty');
  expect(
    (
      await readConsumerState(
        'chart',
        () => 'a',
        async () => Promise.reject(new Error('x'))
      )
    ).status
  ).toBe('error');
});

test('tooltip fields, formats and comparison', () => {
  expect(resolveTooltipFields([{name: 'a'}, {name: 'a'}, {name: 'b'}], 1).truncated).toBe(true);
  expect(formatTooltipValue(1234.5, ',')).toBe('1,234.5');
  expect(formatTooltipValue(0.256, '.1%')).toBe('25.6%');
  expect(formatTooltipValue(3, '+,.1f')).toBe('+3.0');
  expect(compareTooltipValues({v: 2}, {v: 3, w: 'x'}, 'absolute')[0].difference).toBe(1);
  expect(compareTooltipValues({v: 2}, {v: 3}, 'relative')[0].difference).toBe(0.5);
  // Native relative delta is value / pinned - 1, so a negative pinned value is not folded to |pinned|.
  expect(compareTooltipValues({v: -2}, {v: -1}, 'relative')[0].difference).toBe(-0.5);
  expect(compareTooltipValues({v: 0}, {v: 1}, 'relative')[0].difference).toBeNull();
  expect(formatTooltipDifference(0.5, 'relative')).toBe('+50.00%');
  expect(formatTooltipDifference(1.23456, 'absolute')).toBe('+1.235');
  expect(formatTooltipDifference(-2, 'absolute')).toBe('-2');
  expect(formatTooltipDifference(null, 'absolute')).toBe('\u2212');
});

test('brush mask is independent and revisioned', () => {
  const mask = new ResidentBrushMask();
  const state = mask.setBrush('d', [0, 0], 1000);
  expect(isPointBrushed(state, 0.001, 0)).toBe(true);
  expect(isPointBrushed(state, 1, 0)).toBe(false);
  mask.clearBrush('d');
  expect(mask.isCurrent(state)).toBe(false);
});
