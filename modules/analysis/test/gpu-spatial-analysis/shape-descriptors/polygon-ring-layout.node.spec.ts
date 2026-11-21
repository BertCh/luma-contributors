// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {WgslExec, WgslParser} from 'wgsl_reflect';
import {
  POLYGON_FEATURE_RING_RANGES_WGSL,
  POLYGON_SHELL_RING_WGSL
} from '../../../src/analysis-core/ring-assembly/polygon-ring-ranges-wgsl';

it('GEO-131 polygon ring helpers preserve offsets, empty polygons, shell roles and fallback', () => {
  // Three features: empty, multipolygon, empty. The multipolygon has shell/hole rings 0/1,
  // an empty middle polygon and a second shell at ring 2. All view offsets are nonzero.
  const featureOffsets = new Uint32Array([999, 0, 0, 3, 3, 999]);
  const polygonOffsets = new Uint32Array([999, 999, 0, 2, 2, 3, 999]);
  const result = new Uint32Array(11).fill(999);
  const source = `
@group(0) @binding(0) var<storage, read> featureOffsets: array<u32>;
@group(0) @binding(1) var<storage, read> polygonOffsets: array<u32>;
@group(0) @binding(2) var<storage, read_write> result: array<u32>;
const featureOffsetsOffset: u32 = 1u;
const polygonOffsetsOffset: u32 = 2u;
const POLYGON_COUNT: u32 = 3u;
const RING_COUNT: u32 = 3u;
${POLYGON_FEATURE_RING_RANGES_WGSL}
${POLYGON_SHELL_RING_WGSL}
@compute @workgroup_size(1, 1, 1) fn main() {
  for (var feature = 0u; feature < 3u; feature++) {
    result[feature * 2u] = getRingStart(feature);
    result[feature * 2u + 1u] = getRingEnd(feature);
  }
  result[6] = select(0u, 1u, isShellRing(0u, true));
  result[7] = select(0u, 1u, isShellRing(1u, true));
  result[8] = select(0u, 1u, isShellRing(2u, false));
  result[9] = select(0u, 1u, isShellRing(3u, true));
  result[10] = select(0u, 1u, isShellRing(3u, false));
}`;
  new WgslExec(new WgslParser().parse(source)).dispatchWorkgroups('main', [1, 1, 1], {
    0: {0: featureOffsets, 1: polygonOffsets, 2: result}
  });
  // Literal layout expectation: the second shell is not the first ring of the feature.
  expect(Array.from(result)).toEqual([0, 0, 0, 3, 3, 3, 1, 0, 1, 1, 0]);
});
