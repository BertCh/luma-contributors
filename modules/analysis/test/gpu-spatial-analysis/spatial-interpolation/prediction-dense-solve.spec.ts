// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {getEquilibratedSolveWGSL} from '../../../src/gpu-spatial-analysis/spatial-interpolation/equilibrated-solve-wgsl';
import {IS_FINITE_VALUE_WGSL} from '../../../src/analysis-core/kernels/wgsl-common';
import {requireTestDevice} from '../../utils/require-test-device';
import {getFloatWords, runWGSLModuleKernel} from '../../utils/wgsl-module-harness';

it('the prediction adapter solves padded, pivoted and scaled systems and rejects singular ones', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Hand-derived solutions. The last row/column of capacity 4 is deliberately NaN padding.
  const cases = [
    {matrix: [0, 2, 1, 2, 0, 1, 1, 1, 0], right: [1, 3, 3], solution: [2, 1, -1]},
    {matrix: [2e-4, 1, 1, 0], right: [3.0004, 2], solution: [2, 3]},
    {matrix: [2e6, 1e6, 1e6, 3e6], right: [7e6, 11e6], solution: [2, 3]},
    {matrix: [1, 1, 1, 1], right: [1, 1]},
    {matrix: [1, 0, 0, 1e-7], right: [1, 1e-7], solution: [1, 1]}
  ];
  const stride = 21; // size, capacity-square matrix, capacity-vector
  const input = new Float32Array(cases.length * stride).fill(Number.NaN);
  cases.forEach((sample, index) => {
    const size = sample.right.length;
    input[index * stride] = size;
    sample.right.forEach((value, row) => {
      input[index * stride + 17 + row] = value;
      for (let column = 0; column < size; column++) {
        input[index * stride + 1 + row * 4 + column] = sample.matrix[row * size + column];
      }
    });
  });
  const words = await runWGSLModuleKernel(device, {
    id: 'prediction-dense-solve',
    inputs: {input},
    invocationCount: cases.length,
    outputLength: cases.length * 6,
    declarations: `${IS_FINITE_VALUE_WGSL}\n${getEquilibratedSolveWGSL(4)}`,
    body: `let base = index * 21u;
let size = u32(input[inputOffset + base]);
for (var entry = 0u; entry < 16u; entry++) { matrix[entry] = input[inputOffset + base + 1u + entry]; }
for (var row = 0u; row < 4u; row++) { right[row] = input[inputOffset + base + 17u + row]; }
let success = solveSystem(size);
output[outputOffset + index * 6u] = u32(success);
output[outputOffset + index * 6u + 1u] = bitcast<u32>(pivotRatio);
for (var row = 0u; row < size; row++) { output[outputOffset + index * 6u + 2u + row] = bitcast<u32>(solution[row]); }`
  });
  const values = getFloatWords(words);
  cases.forEach((sample, index) => {
    expect(words[index * 6]).toBe(sample.solution ? 1 : 0);
    sample.solution?.forEach((expected, row) => {
      expect(Math.abs(values[index * 6 + 2 + row] - expected)).toBeLessThan(2e-5);
    });
  });
});
