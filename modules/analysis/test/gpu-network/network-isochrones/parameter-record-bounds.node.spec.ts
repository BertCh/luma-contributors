// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getGPUNetworkIsochronesParameterValues} from '../../../src/gpu-network/network-isochrones/index';

it('packs exactly twelve words into an interior arena view (PCK-02)', () => {
  const arena = Float32Array.of(11, 22, ...new Array(12).fill(99), 33, 44);
  const target = arena.subarray(2, 15); // Word 12 is in the view but outside the parameter record.
  expect(
    getGPUNetworkIsochronesParameterValues({breakCount: 1, extent: [0, 0, 2, 2]}, target) === target
  ).toBe(true);
  expect(Array.from(arena.slice(0, 2))).toEqual([11, 22]);
  expect(Array.from(arena.slice(14))).toEqual([33, 44]);
  // The 12-word layout reserves its final two words as zero.
  expect(Array.from(arena.slice(12, 14))).toEqual([0, 0]);
});
