// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  applyGPUPolygonLow,
  GPU_POLYGON_LOW_MARKER
} from '../../../../examples/deck/kepler-compute/gpu-layers/layers/gpu-polygon-low';

const SOURCE = `head\n${GPU_POLYGON_LOW_MARKER}\ntail`;

test('default polygon shaders get zero-valued low stubs and no extra bindings', () => {
  const shader = applyGPUPolygonLow(SOURCE, false);
  expect(shader).not.toContain(GPU_POLYGON_LOW_MARKER);
  expect(shader).toContain('fn getPolygonTopLow');
  expect(shader).toContain('fn getPolygonRingLow');
  expect(shader).not.toContain('@binding');
});

test('fp64 polygon shaders bind borrowed low streams for top faces and extrusion rings', () => {
  const shader = applyGPUPolygonLow(SOURCE, true);
  expect(shader).toContain('var<storage, read> polygonLowPositions');
  expect(shader).toContain('var<storage, read> polygonRingLowPositions');
  expect(shader).toContain('var<uniform> polygonLowConfig');
  expect(shader).toContain('fn getPolygonTopLow');
  expect(shader).toContain('fn getPolygonRingLow');
});
