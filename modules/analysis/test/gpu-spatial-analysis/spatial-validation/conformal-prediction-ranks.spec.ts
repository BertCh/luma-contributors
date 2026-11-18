// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CONFORMAL_PREDICTION_RANKS_WGSL} from '../../../src/gpu-spatial-analysis/spatial-validation/conformal-prediction-ranks';
import {requireTestDevice} from '../../utils/require-test-device';
import {runWGSLModuleKernel} from '../../utils/wgsl-module-harness';

it('computes exact conformal ranks on the GPU using only a small table of counts', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // [calibration count, alpha, expected lower rank, expected upper rank].
  // Large row counts are scalars: no calibration dataset is allocated.
  const cases = [
    [8_388_607, 0.25, 2_097_152, 6_291_456],
    [8_388_606, 0.25, 2_097_151, 6_291_456],
    [8_388_608, 0.25, 2_097_152, 6_291_457],
    [99, 0.1, 10, 90],
    [19, 0.05, 1, 19],
    [24, 0.4, 10, 15],
    [44, 0.4, 18, 27],
    [9, 0.7, 6, 4],
    [9, 0.9, 8, 2],
    [5, 0.1, 0, 6],
    [0, 0.25, 0, 1],
    [0xfffffffe, 0.5, 2_147_483_647, 2_147_483_648],
    [0xfffffffe, 1 - 2 ** -24, 4_294_967_039, 256],
    [0xfffffffe, 2 ** -149, 0, 0xffffffff],
    [0xfffffffe, 2 ** -40, 0, 0xffffffff]
  ];
  const actual = await runWGSLModuleKernel(device, {
    id: 'conformal-exact-ranks',
    inputs: {
      counts: Uint32Array.from(cases, entry => entry[0]),
      miscoverages: Float32Array.from(cases, entry => entry[1])
    },
    outputLength: 2 * cases.length,
    invocationCount: cases.length,
    declarations: CONFORMAL_PREDICTION_RANKS_WGSL,
    body: `let ranks = getConformalPredictionRanks(counts[countsOffset + index], miscoverages[miscoveragesOffset + index]);
  output[outputOffset + 2u * index] = ranks.x;
  output[outputOffset + 2u * index + 1u] = ranks.y;`
  });
  expect(Array.from(actual)).toEqual(cases.flatMap(entry => entry.slice(2)));
});
