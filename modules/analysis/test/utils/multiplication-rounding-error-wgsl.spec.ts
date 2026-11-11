// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {MULTIPLICATION_ROUNDING_ERROR_WGSL} from '../../src/analysis-core/kernels/multiplication-rounding-error-wgsl';
import {requireTestDevice} from './require-test-device';
import {createSeededRandom, getFloatWords, runWGSLModuleKernel} from './wgsl-module-harness';

// getMultiplicationRoundingError against float64. The product of two float32 values has at most
// 48 significant bits, so it is exact in float64, and so is its difference from the float32
// product: nothing here repeats the kernel's integer arithmetic.

it('getMultiplicationRoundingError is the exact error of a float32 multiplication for operands of either sign', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const random = createSeededRandom(2611);
  const pairs: [number, number][] = [
    // Exact products: the error is 0.
    [1, 2],
    [0.5, -0.25],
    [3, 5],
    [0, 7.25],
    [-7.25, 0],
    [4097, 4095],
    // (2^12 + 1)^2 = 2^24 + 2^13 + 1 needs 25 bits: it rounds to 2^24 + 2^13, the error is +1.
    [4097, 4097],
    [-4097, 4097],
    // (2^24 - 1)^2 = 2^48 - 2^25 + 1 rounds to 2^48 - 2^25, the error is +1.
    [16777215, 16777215],
    // The case of the fit: a centered lag of thousands times a coefficient near 0.15.
    [-3333.33349609375, 0.15000273287296295],
    [6666.66650390625, 0.4000356495380402],
    // Not finite, or below the range the helper reports.
    [Infinity, 1],
    [3e38, 3e38],
    [1e-20, 1e-20]
  ];
  for (let index = 0; index < 4000; index++) {
    // Exponents that keep the product between 2^-60 and 2^60.
    const first = (0.5 + 0.5 * random()) * 2 ** (Math.floor(60 * random()) - 30);
    const second = (0.5 + 0.5 * random()) * 2 ** (Math.floor(60 * random()) - 30);
    pairs.push([random() < 0.5 ? -first : first, random() < 0.5 ? -second : second]);
  }
  const inputs = Float32Array.from(pairs.flat());
  const words = await runWGSLModuleKernel(device, {
    id: 'multiplication-rounding-error',
    inputs: {inputs},
    outputLength: pairs.length * 2,
    invocationCount: pairs.length,
    declarations: MULTIPLICATION_ROUNDING_ERROR_WGSL,
    body: `let first = inputs[inputsOffset + 2u * index];
  let second = inputs[inputsOffset + 2u * index + 1u];
  let product = first * second;
  output[outputOffset + 2u * index] = bitcast<u32>(product);
  output[outputOffset + 2u * index + 1u] = bitcast<u32>(getMultiplicationRoundingError(first, second, product));`
  });
  const values = getFloatWords(words);
  const mismatches: string[] = [];
  let inexactCount = 0;
  pairs.forEach((_, index) => {
    const first = inputs[2 * index];
    const second = inputs[2 * index + 1];
    // The GPU's own product: the helper reports its distance from the exact product, whether or
    // not the multiplication was correctly rounded. It must be within one unit of the last place.
    const product = values[2 * index];
    const rounded = Math.fround(first * second);
    // Every product here is above 2^-64 except the one case far below 2^-80, where the helper
    // reports no error by design (and the GPU may flush the product to zero).
    const isReported =
      Number.isFinite(first) &&
      Number.isFinite(second) &&
      Number.isFinite(rounded) &&
      Math.abs(first * second) >= 2 ** -70;
    const expected = isReported ? first * second - product : 0;
    if (expected !== 0) inexactCount++;
    const isProductClose =
      !isReported ||
      product === rounded ||
      Math.abs(product - rounded) <= Math.abs(rounded) * 2 ** -22;
    if (!isProductClose || values[2 * index + 1] !== expected) {
      mismatches.push(
        `${first} * ${second}: product ${product} (${rounded}), error ${values[2 * index + 1]} (${expected})`
      );
    }
  });
  expect(mismatches.slice(0, 10)).toEqual([]);
  // Most random products are inexact: the test is not vacuous.
  expect(inexactCount).toBeGreaterThan(3000);
  // Hand values of the cases above.
  expect(values[2 * 5 + 1]).toBe(0);
  expect(values[2 * 6 + 1]).toBe(1);
  expect(values[2 * 7 + 1]).toBe(-1);
  expect(values[2 * 8 + 1]).toBe(1);
});

it('getMultiplicationRoundingError reports no error below an exponent sum of -80 and rounds the distance of a product that is not correctly rounded', async () => {
  // The two counterexamples of the second review of the closing pass (frc2-regression-08 of
  // fix-review-closing-regression-codex.md): the documentation claimed a cutoff at an exact
  // product of 2^-80 and an exact distance for any product within two units in the last
  // place. Both claims were wrong; the helper is unchanged and its documentation now states
  // what these cases pin. Expected values are float64 arithmetic on the float32 operands (a
  // product of two float32 values has 48 bits) and powers of two by hand.
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const significand = 1.5 + 2 ** -23;
  const cases: [first: number, second: number, product: number | undefined][] = [
    // Binary exponents -41 and -40 sum to -81: the exact product, 1.125 * 2^-80, is above
    // 2^-80, yet no error is reported (the true error is -2^-104 to within 1e-38).
    [significand * 2 ** -41, significand * 2 ** -40, undefined],
    // The same significands with exponents that sum to -80: reported exactly.
    [significand * 2 ** -40, significand * 2 ** -40, undefined],
    // Exponents that sum to -81 with significands whose product is below 2 (a product below
    // 2^-80): not reported either.
    [1.25 * 2 ** -41, 1.25 * 2 ** -40, undefined],
    // A product supplied one step below the correctly rounded one: the distance is
    // 2^-22 + 2^-46, which needs 25 bits; the helper returns its float32 rounding, 2^-22.
    [2 - 2 ** -23, 2 - 2 ** -23, 4 - 3 * 2 ** -22],
    // The correctly rounded product of the same operands, 4 - 2^-21: the exact error 2^-46.
    [2 - 2 ** -23, 2 - 2 ** -23, 4 - 2 ** -21]
  ];
  const inputs = Float32Array.from(
    cases.flatMap(([first, second, product]) => [
      first,
      second,
      product ?? 0,
      product === undefined ? 0 : 1
    ])
  );
  const words = await runWGSLModuleKernel(device, {
    id: 'multiplication-rounding-error-domain',
    inputs: {inputs},
    outputLength: cases.length * 2,
    invocationCount: cases.length,
    declarations: MULTIPLICATION_ROUNDING_ERROR_WGSL,
    body: `let first = inputs[inputsOffset + 4u * index];
  let second = inputs[inputsOffset + 4u * index + 1u];
  var product = first * second;
  if (inputs[inputsOffset + 4u * index + 3u] != 0.0) { product = inputs[inputsOffset + 4u * index + 2u]; }
  output[outputOffset + 2u * index] = bitcast<u32>(product);
  output[outputOffset + 2u * index + 1u] = bitcast<u32>(getMultiplicationRoundingError(first, second, product));`
  });
  const values = getFloatWords(words);
  const getExact = (index: number) => inputs[4 * index] * inputs[4 * index + 1] - values[2 * index];
  // Exponent sum -81, product above 2^-80: unreported, although the error is a normal float32.
  expect(inputs[0] * inputs[1]).toBeGreaterThan(2 ** -80);
  expect(values[0]).toBe(Math.fround(inputs[0] * inputs[1]));
  expect(Math.abs(getExact(0) + 2 ** -104)).toBeLessThan(1e-38);
  expect(values[1]).toBe(0);
  // Exponent sum -80: the exact error.
  expect(getExact(1)).not.toBe(0);
  expect(values[3]).toBe(getExact(1));
  // Exponent sum -81, product below 2^-80: unreported.
  expect(inputs[8] * inputs[9]).toBeLessThan(2 ** -80);
  expect(values[5]).toBe(0);
  // One step below the correctly rounded product: 25 significant bits, rounded to float32.
  expect(values[6]).toBe(4 - 3 * 2 ** -22);
  expect(getExact(3)).toBe(2 ** -22 + 2 ** -46);
  expect(values[7]).toBe(2 ** -22);
  expect(values[7]).toBe(Math.fround(getExact(3)));
  // The correctly rounded product: exact.
  expect(values[8]).toBe(4 - 2 ** -21);
  expect(values[9]).toBe(2 ** -46);
});
