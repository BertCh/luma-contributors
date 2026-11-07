// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {createLocalPermutationHarness} from './permutation-harness';
import {createKnnWeights, createSeededRandom} from './permutation-oracle';
import {requireTestDevice} from '../../utils/require-test-device';

it('tiled row dispatch gives identical local permutation results for every tile size', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const random = createSeededRandom(5);
  const count = 531;
  const positions = Float64Array.from({length: count * 2}, () => random() * 100);
  const values = Float32Array.from({length: count}, (_, row) => Math.sin(row / 11) + random());
  const weights = createKnnWeights(positions, 5, true);
  const parameters = {seed: 9, permutationCount: 99, significanceLevel: 0.05};
  const readings = [];
  for (const rowTileSize of [undefined, 200, 64, 1]) {
    const harness = createLocalPermutationHarness(device, {
      scene: {weights, values, rowStandardized: true},
      statistic: 'localMoran',
      parameters,
      maximumPermutations: 99,
      rowTileSize
    });
    readings.push(await harness.run());
    harness.destroy();
  }
  const [reference, ...tiled] = readings;
  for (const result of tiled) {
    expect(Array.from(result.exceedances)).toEqual(Array.from(reference.exceedances));
    expect(Array.from(result.pseudoPValues)).toEqual(Array.from(reference.pseudoPValues));
    expect(Array.from(result.significant)).toEqual(Array.from(reference.significant));
  }
});
