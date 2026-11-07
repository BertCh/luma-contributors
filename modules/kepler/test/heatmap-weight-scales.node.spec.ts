// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  isNativeHeatmapWeightScale,
  NATIVE_HEATMAP_WEIGHT_SCALES
} from '../../../examples/deck/kepler-compute/src/heatmap-weight-scales';

test('heatmap weight scales match the native size-channel scale list', () => {
  expect(NATIVE_HEATMAP_WEIGHT_SCALES).toEqual(['linear', 'sqrt', 'log']);
  for (const scale of ['linear', 'sqrt', 'log'])
    expect(isNativeHeatmapWeightScale(scale)).toBe(true);
  for (const scale of ['quantile', 'quantize', 'ordinal', 'point', undefined, 1])
    expect(isNativeHeatmapWeightScale(scale)).toBe(false);
});
