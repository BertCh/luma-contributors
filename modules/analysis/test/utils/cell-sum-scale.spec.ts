// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  getFixedPointWGSL,
  getGPUCellSumScale,
  GPU_CELL_DEFAULT_SUM_SCALE
} from '../../src/analysis-core/kernels/cell-key-wgsl';
import {requireTestDevice} from './require-test-device';
import {getFloatWords, runWGSLModuleKernel} from './wgsl-module-harness';

// The 64-bit fixed-point sum of analysis-core/kernels/cell-key-wgsl.ts on a column of small values (math
// review finding core-numerics-03). With the default scale every row is rounded to a multiple of
// 1 / 65536, which the documentation used to call exact; with a scale from getGPUCellSumScale the
// rows are added without rounding. Expected values are hand arithmetic on the documented rule.

const ROW_COUNT = 1000;

async function sumConstantColumn(value: number, sumScale: number): Promise<number> {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const words = await runWGSLModuleKernel(device, {
    id: 'cell-sum-scale',
    inputs: {values: new Float32Array(ROW_COUNT).fill(value)},
    outputLength: 1,
    invocationCount: 1,
    declarations: getFixedPointWGSL(sumScale),
    body: `var total = vec2u(0u, 0u);
  for (var row = 0u; row < ${ROW_COUNT}u; row++) {
    total = cellAddI64(total, cellScaleValue(values[valuesOffset + row]));
  }
  output[outputOffset] = bitcast<u32>(cellI64ToF32(total) / SUM_SCALE);`
  });
  return getFloatWords(words)[0];
}

it('the default scale rounds each row of a 1e-4 column to 7 / 65536', async () => {
  // 1e-4 * 65536 = 6.5536 rounds to 7, so 1000 rows sum to 7000 / 65536, 6.8% above 0.1.
  const total = await sumConstantColumn(1e-4, GPU_CELL_DEFAULT_SUM_SCALE);
  expect(total).toBe(Math.fround(7000 / 65536));
  // A value below 2^-17 contributes nothing.
  expect(await sumConstantColumn(7e-6, GPU_CELL_DEFAULT_SUM_SCALE)).toBe(0);
});

it('a scale from getGPUCellSumScale adds the same column without rounding', async () => {
  // 1000 rows of float32(1e-4) = 9.999999747378752e-5 sum to 0.09999999747378752.
  const small = await sumConstantColumn(1e-4, getGPUCellSumScale(1e-4, ROW_COUNT));
  expect(Math.abs(small - 0.09999999747378752) / 0.1).toBeLessThan(1e-7);
  // 1000 rows of float32(7e-6) = 7.000000096013537e-6 sum to 0.007000000096013537.
  const tiny = await sumConstantColumn(7e-6, getGPUCellSumScale(7e-6, ROW_COUNT));
  expect(Math.abs(tiny - 0.007000000096013537) / 7e-3).toBeLessThan(1e-7);
  // A large column keeps its sum too: 1000 rows of 3e9 (float32 exactly 3,000,000,000).
  const large = await sumConstantColumn(3e9, getGPUCellSumScale(3e9, ROW_COUNT));
  expect(Math.abs(large - 3e12) / 3e12).toBeLessThan(1e-7);
});
