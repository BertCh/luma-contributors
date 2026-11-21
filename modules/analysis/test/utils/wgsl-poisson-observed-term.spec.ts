// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {POISSON_OBSERVED_TERM_WGSL} from '../../src/analysis-core/kernels/wgsl-poisson-observed-term';
import {requireTestDevice} from './require-test-device';
import {createSeededRandom, getFloatWords, runWGSLModuleKernel} from './wgsl-module-harness';

// getObservedTerm against the Poisson term it stands for, o ln(o / e) - o + e, in float64 (math
// review finding core-numerics-11: no spec named this kernel). The reference is written as
// o log1p(d) - (o - e) with d = (o - e) / e, which is the same quantity and keeps float64 accuracy
// near o = e; it is not the kernel's Taylor series.

function getReferenceTerm(observed: number, expected: number): number {
  if (!(expected > 0)) {
    return 0;
  }
  if (!(observed > 0)) {
    return expected;
  }
  return observed * Math.log1p((observed - expected) / expected) - (observed - expected);
}

it('the Poisson observed term matches o ln(o / e) - o + e in float64 on both sides of the series switch', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const random = createSeededRandom(1107);
  const pairs: [number, number][] = [
    // Hand cases: o = e gives 0; o = 0 gives e; e <= 0 gives 0.
    [5, 5],
    [0, 3.5],
    [4, 0],
    [4, -1],
    // o = 2e: e (2 ln 2 - 1); o = e / 2: e (1 - ln 2) / 2.
    [20, 10],
    [5, 10],
    // Either side of |d| = 1/4, where the kernel switches from the series to the direct form.
    [12.4999, 10],
    [12.5, 10],
    [12.5001, 10],
    [7.4999, 10],
    [7.5, 10],
    [7.5001, 10],
    // Nearly equal counts, where the direct form cancels.
    [1000.001, 1000],
    [999.999, 1000],
    [1000001, 1000000],
    // A tiny observed count against a large expectation, and the reverse. In float32
    // (1e-6 - 50) / 50 and (1 - 5e7) / 5e7 are exactly -1, where (1 + d) ln(1 + d) is 0 * log(0):
    // the term must be the expectation, not NaN.
    [1e-6, 50],
    [1, 5e7],
    [5000, 0.01]
  ];
  for (let index = 0; index < 600; index++) {
    const expected = 10 ** (6 * random() - 2);
    const kind = index % 3;
    const ratio =
      kind === 0
        ? 1 + 0.5 * (random() - 0.5)
        : kind === 1
          ? 1 + 1e-3 * (random() - 0.5)
          : 10 ** (4 * random() - 2);
    pairs.push([expected * ratio, expected]);
  }
  const inputs = Float32Array.from(pairs.flat());
  const words = await runWGSLModuleKernel(device, {
    id: 'poisson-observed-term',
    inputs: {inputs},
    outputLength: pairs.length,
    invocationCount: pairs.length,
    declarations: POISSON_OBSERVED_TERM_WGSL,
    body: 'output[outputOffset + index] = bitcast<u32>(getObservedTerm(inputs[inputsOffset + 2u * index], inputs[inputsOffset + 2u * index + 1u]));'
  });
  const values = getFloatWords(words);
  // Exact hand cases.
  expect(values[0]).toBe(0);
  expect(values[1]).toBe(3.5);
  expect(values[2]).toBe(0);
  expect(values[3]).toBe(0);
  expect(Math.abs(values[4] - 10 * (2 * Math.LN2 - 1))).toBeLessThan(1e-5);
  expect(Math.abs(values[5] - (10 * (1 - Math.LN2)) / 2)).toBeLessThan(1e-5);
  const mismatches: string[] = [];
  pairs.forEach((_, index) => {
    // The kernel sees the float32 inputs, so the reference does too.
    const observed = inputs[2 * index];
    const expected = inputs[2 * index + 1];
    const reference = getReferenceTerm(observed, expected);
    // The float32 difference o - e is exact for nearby counts, so d carries about one float32
    // rounding and the term two; the direct form adds the cancellation of (1 + d) ln(1 + d) - d
    // near |d| = 1/4 (a factor of about 10). 2e-5 relative covers both.
    const tolerance = 2e-5 * Math.abs(reference) + 1e-30;
    if (!(Math.abs(values[index] - reference) <= tolerance) || values[index] < 0) {
      mismatches.push(`o = ${observed}, e = ${expected}: ${values[index]} against ${reference}`);
    }
  });
  expect(mismatches.slice(0, 10)).toEqual([]);
});

// CORE-104: a large count over a tiny but normal float32 baseline overflowed d = (o - e) / e or
// (1 + d) ln(1 + d), and the term was NaN or +inf. References: float64 o (ln o - ln e) - o + e of
// the float32 inputs (/usr/bin/python3, NumPy float32 rounding of the inputs).
it('the Poisson observed term stays finite when its ratio overflows (CORE-104)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const pairs: [number, number][] = [
    [1e9, 1e-30],
    [1e8, 1e-30],
    [1e6, 1e-36],
    [5e37, 1e-3]
  ];
  const inputs = Float32Array.from(pairs.flat());
  const words = await runWGSLModuleKernel(device, {
    id: 'poisson-observed-term-overflow',
    inputs: {inputs},
    outputLength: pairs.length,
    invocationCount: pairs.length,
    declarations: POISSON_OBSERVED_TERM_WGSL,
    body: 'output[outputOffset + index] = bitcast<u32>(getObservedTerm(inputs[inputsOffset + 2u * index], inputs[inputsOffset + 2u * index + 1u]));'
  });
  const values = getFloatWords(words);
  const references = [88800818623.5967, 8649823353.060265, 95708573.86981079];
  references.forEach((reference, index) => {
    expect(Number.isFinite(values[index]), `pair ${index}`).toBe(true);
    expect(Math.abs(values[index] - reference)).toBeLessThanOrEqual(1e-5 * reference);
  });
  // 4.6e39 exceeds float32: +inf, not NaN.
  expect(values[3]).toBe(Infinity);
});
