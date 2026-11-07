// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  getCartogramPairEvaluations,
  normalizeCartogramSettings
} from '../../../examples/deck/kepler-compute/src/cartogram-settings';

const settings = {
  coordinateSpace: 'planar',
  metric: 'native',
  units: 'meters',
  crs: 'EPSG:32618',
  bounds: [0, 0, 1000, 1000],
  xColumn: 'east',
  yColumn: 'north',
  valueColumn: 'area',
  areaScale: 100,
  maximumIterations: 256,
  overlapTolerance: 1,
  attraction: 0
} as never;

test('all-pairs cartogram work is bounded to one GPU submission', () => {
  expect(getCartogramPairEvaluations(1024, 256)).toBe(1024 * 1024 * 256);
  expect(() => normalizeCartogramSettings(settings, 1024)).not.toThrow();
  expect(() => normalizeCartogramSettings(settings, 16384)).not.toThrow();
  expect(() => normalizeCartogramSettings(settings, 65536)).toThrow(/reduce passes/);
  expect(() =>
    normalizeCartogramSettings({...(settings as object), maximumIterations: 90} as never, 65536)
  ).not.toThrow();
});
