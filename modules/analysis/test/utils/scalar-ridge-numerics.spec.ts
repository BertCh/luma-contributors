// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {SCALED_QUOTIENT_WGSL} from '../../src/analysis-core/kernels/scaled-quotient-wgsl';
import {SMALLEST_SYMMETRIC_EIGEN_2_WGSL} from '../../src/analysis-core/kernels/symmetric-eigen-wgsl';
import {requireTestDevice} from './require-test-device';
import {getFloatWords, runWGSLModuleKernel} from './wgsl-module-harness';

it('spacing quotients preserve normal results when denominator products overflow or underflow', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const cases = [
    [-2e10, 1e20, 1e20, 0],
    [-2e-30, 1e-20, 1e-20, 0],
    [-4, 1e20, 1e-20, -2],
    [1e10, 1e20, 1, -1],
    [0, 1e20, 1e20, 0]
  ];
  const inputs = Float32Array.from(cases.flat());
  const actual = getFloatWords(
    await runWGSLModuleKernel(device, {
      id: 'scalar-ridge-scaled-quotients',
      inputs: {inputs},
      invocationCount: cases.length,
      outputLength: cases.length,
      declarations: SCALED_QUOTIENT_WGSL,
      body: `let base = 4u * index;
      output[index] = bitcast<u32>(getScaledQuotient(inputs[base], inputs[base + 1u],
        inputs[base + 2u], i32(inputs[base + 3u])));`
    })
  );
  for (let index = 0; index < cases.length; index++) {
    const base = 4 * index;
    // Binary64 can form these products without overflow or underflow; inputs are the uploaded f32.
    const expected = (inputs[base] * 2 ** inputs[base + 3]) / (inputs[base + 1] * inputs[base + 2]);
    if (expected === 0) {
      expect(actual[index]).toBe(0);
    } else {
      expect(Math.abs(actual[index] / expected - 1)).toBeLessThan(2e-5);
    }
  }
});

it('smallest eigenpairs preserve weak diagonals, tiny coupling corrections and their normal directions', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // (first diagonal, cross, second diagonal); closed forms avoid a second eigensolver.
  const inputs = Float32Array.from([
    1e30, 0, -1e-20, -1e-20, 0, 1e30, 1e30, 1e5, -1e-20, 1e-25, 2e-25, 1e-25, 1e21, -2e21, 1e21, 3,
    0, 3
  ]);
  const count = inputs.length / 3;
  const actual = getFloatWords(
    await runWGSLModuleKernel(device, {
      id: 'scalar-ridge-smallest-eigenpairs',
      inputs: {inputs},
      invocationCount: count,
      outputLength: 3 * count,
      declarations: SMALLEST_SYMMETRIC_EIGEN_2_WGSL,
      body: `let base = 3u * index;
      let pair = getSmallestSymmetricEigen2(inputs[base], inputs[base + 1u], inputs[base + 2u]);
      output[base] = bitcast<u32>(pair.value);
      output[base + 1u] = bitcast<u32>(pair.vector.x);
      output[base + 2u] = bitcast<u32>(pair.vector.y);`
    })
  );
  const expectedValues = [
    inputs[2],
    inputs[3],
    inputs[8] - inputs[7] ** 2 / inputs[6],
    inputs[9] - Math.abs(inputs[10]),
    inputs[12] - Math.abs(inputs[13]),
    3
  ];
  const expectedDirections = [
    [0, 1],
    [1, 0],
    [inputs[7] / inputs[6], -1],
    [Math.SQRT1_2, -Math.SQRT1_2],
    [Math.SQRT1_2, Math.SQRT1_2],
    [1, 0]
  ];
  for (let index = 0; index < count; index++) {
    expect(Math.abs(actual[3 * index] / expectedValues[index] - 1)).toBeLessThan(2e-5);
    for (let component = 0; component < 2; component++) {
      const expected = expectedDirections[index][component];
      const value = actual[3 * index + component + 1];
      if (expected === 0) {
        expect(value).toBe(0);
      } else {
        expect(Math.abs(value / expected - 1)).toBeLessThan(2e-5);
      }
    }
  }
});
