// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {getGPUNeighborSearchParameterValues} from '../../../src/gpu-spatial-analysis/neighbor-search/index';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createNeighborSearchHarness} from './neighbor-search-harness';

it('NAM-12 radius 1.1 finds the two unit-separated self-join rows with no invalid input', async () => {
  const bounds = [-1, -1, 2, 1] as const;
  for (const searchRadius of [undefined, 0, -1, Infinity, NaN]) {
    expect(() =>
      getGPUNeighborSearchParameterValues({mode: 'radius', bounds, searchRadius} as never)
    ).toThrow(/searchRadius/);
  }
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const legacy = createNeighborSearchHarness(device, {
    mode: 'radius',
    positions: Float32Array.of(0, 0, 1, 0),
    capacity: 2,
    parameters: {bounds},
    gridSize: [2, 2]
  });
  try {
    // The retained old overload packs Infinity: radius mode has an invalid lattice, but both
    // coordinates are valid, so invalidCount does not diagnose the missing radius.
    const result = await legacy.run();
    expect(result.offsets).toEqual([0, 0, 0]);
    expect(result.invalidCount).toBe(0);
  } finally {
    legacy.destroy();
  }
  const harness = createNeighborSearchHarness(device, {
    mode: 'radius',
    positions: Float32Array.of(0, 0, 1, 0),
    capacity: 2,
    parameters: {mode: 'radius', bounds, searchRadius: 1.1},
    gridSize: [2, 2]
  });
  try {
    const result = await harness.run();
    // Euclidean distance is exactly one; exclude self and retain the other row in each row.
    expect(result.offsets).toEqual([0, 1, 2]);
    expect(result.neighbors).toEqual([1, 0]);
    expect(result.distances).toEqual([1, 1]);
    expect(result.invalidCount).toBe(0);
    expect(result.overflow).toBe(0);
  } finally {
    harness.destroy();
  }
});
