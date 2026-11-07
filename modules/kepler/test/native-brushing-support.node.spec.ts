// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  getBrushingRadiusMeters,
  getNativeBrushingEntry,
  NATIVE_BRUSHING_AUDIT
} from '../../../examples/deck/kepler-compute/src/native-brushing-support';

test('native brushing audit covers exactly the layers that attach the extension', () => {
  expect(NATIVE_BRUSHING_AUDIT.map(entry => entry.layerType).sort()).toEqual([
    'arc',
    'hexagonId',
    'icon',
    'line',
    'point'
  ]);
  expect(getNativeBrushingEntry('geojson')).toBeUndefined();
  expect(getNativeBrushingEntry('arc')?.target).toBe('source_target');
  expect(getNativeBrushingEntry('point')?.target).toBe('source');
  for (const entry of NATIVE_BRUSHING_AUDIT) expect(entry.residentLayers.length).toBeGreaterThan(0);
});

test('brush radius converts kilometers to meters and ignores invalid sizes', () => {
  expect(getBrushingRadiusMeters({enabled: true, config: {size: 0.5}})).toBe(500);
  expect(getBrushingRadiusMeters({enabled: true, config: {size: Number.NaN}})).toBe(0);
  expect(getBrushingRadiusMeters(undefined)).toBe(0);
});
