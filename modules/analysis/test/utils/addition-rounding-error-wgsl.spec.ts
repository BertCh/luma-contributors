// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {ADDITION_ROUNDING_ERROR_WGSL} from '../../src/analysis-core/kernels/addition-rounding-error-wgsl';
import {
  FIXED_POINT_WGSL,
  getFixedPointHeadroomBits
} from '../../src/analysis-core/kernels/fixed-point-accumulator';
import {requireTestDevice} from './require-test-device';
import {createSeededRandom, getFloatWords, runWGSLModuleKernel} from './wgsl-module-harness';

// getAdditionRoundingError against float64, and the reason it exists: on the test GPU the float
// form of a compensated sum is removed by the shader compiler. Expected values are float64
// arithmetic on float32 inputs chosen so that float64 is exact; nothing here repeats the kernel's
// integer arithmetic.

const toFloat32 = Math.fround;

it('getAdditionRoundingError is the exact error of a float32 addition for operands of either sign', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const random = createSeededRandom(2311);
  const pairs: [number, number][] = [
    // Exact additions: the error is 0.
    [1, 2],
    [0.5, -0.5],
    [3, -1],
    [0, 7.25],
    [-7.25, 0],
    // 2^24 + 1 is a tie that rounds to even: the error is +1. With 3 it rounds up: the error is -1.
    [16777216, 1],
    [16777216, 3],
    [-16777216, -1],
    [-16777216, -3],
    // Subtraction that borrows into the binade below: 2^24 - 0.75 rounds to 2^24 - 1 (error +0.25).
    [16777216, -0.75],
    [-16777216, 0.75],
    // Opposite signs one binade apart with an odd result of 25 bits: (2^25 - 2) - (2^23 + 1).
    [33554430, -8388609],
    [-33554430, 8388609],
    // The smaller operand is far below the last place of the larger: it is the whole error.
    [1073741824, 0.001],
    [1073741824, -0.001],
    [-3e20, 5],
    // Exactly half and a quarter of the last place of a power of two.
    [8388608, 0.5],
    [8388608, -0.25],
    [8388608, -0.125],
    // Not finite: no error is reported.
    [Infinity, 1],
    [3e38, 3e38],
    // Below 2^-103 the helper reports no error by design.
    [1, 1e-35]
  ];
  for (let index = 0; index < 4000; index++) {
    // Significands of 24 bits, exponents at most 27 binades apart, so the float64 sum is exact.
    const exponent = Math.floor(60 * random()) - 30;
    const gap = Math.floor(28 * random());
    const first = (0.5 + 0.5 * random()) * 2 ** exponent * (random() < 0.5 ? -1 : 1);
    const second = (0.5 + 0.5 * random()) * 2 ** (exponent - gap) * (random() < 0.5 ? -1 : 1);
    pairs.push(index % 2 === 0 ? [first, second] : [second, first]);
  }
  const inputs = Float32Array.from(pairs.flat());
  const words = await runWGSLModuleKernel(device, {
    id: 'addition-rounding-error',
    inputs: {inputs},
    outputLength: pairs.length * 2,
    invocationCount: pairs.length,
    declarations: ADDITION_ROUNDING_ERROR_WGSL,
    body: `let first = inputs[inputsOffset + 2u * index];
  let second = inputs[inputsOffset + 2u * index + 1u];
  let sum = first + second;
  output[outputOffset + 2u * index] = bitcast<u32>(sum);
  output[outputOffset + 2u * index + 1u] = bitcast<u32>(getAdditionRoundingError(first, second, sum));`
  });
  const values = getFloatWords(words);
  const mismatches: string[] = [];
  pairs.forEach((_, index) => {
    const first = inputs[2 * index];
    const second = inputs[2 * index + 1];
    const sum = toFloat32(first + second);
    let expected = 0;
    if (Number.isFinite(sum) && Number.isFinite(first) && Number.isFinite(second)) {
      const larger = Math.max(Math.abs(first), Math.abs(second));
      const smaller = Math.min(Math.abs(first), Math.abs(second));
      if (smaller > 0 && smaller < 2 ** -103) {
        expected = 0;
      } else if (larger / smaller >= 2 ** 28) {
        // Float64 cannot hold first + second exactly; the sum is the larger operand.
        expected = Math.abs(first) < Math.abs(second) ? first : second;
        expect(Math.abs(sum)).toBe(larger);
      } else {
        // first + second is exact in float64 here, and so is the difference from the float32 sum.
        expected = first + second - sum;
      }
    }
    if (values[2 * index] !== sum || values[2 * index + 1] !== expected) {
      mismatches.push(
        `${first} + ${second}: sum ${values[2 * index]} (${sum}), error ${values[2 * index + 1]} (${expected})`
      );
    }
  });
  expect(mismatches.slice(0, 10)).toEqual([]);
  // Hand values of the cases above.
  expect(values[2 * 5 + 1]).toBe(1);
  expect(values[2 * 6 + 1]).toBe(-1);
  expect(values[2 * 7 + 1]).toBe(-1);
  expect(values[2 * 8 + 1]).toBe(1);
  expect(values[2 * 9 + 1]).toBe(0.25);
  expect(values[2 * 10 + 1]).toBe(-0.25);
  expect(values[2 * 13 + 1]).toBe(toFloat32(0.001));
});

const TERM_COUNT = 1_000_000;
/** Terms are multiples of 1 / 1024 below 4096 in magnitude, so their exact sum is an integer sum. */
const TERM_SCALE = 1024;

type SerialSums = {
  /** Plain float32 running sum. */
  plain: number;
  /** Neumaier sum with the float correction `(sum - next) + term`: running sum and correction. */
  floatSum: number;
  floatCompensation: number;
  /** Neumaier sum with getAdditionRoundingError: running sum, correction and their float32 total. */
  integerSum: number;
  integerCompensation: number;
  integerTotal: number;
  /** The 96-bit fixed-point accumulator, decoded once. */
  fixedPoint: number;
};

async function sumOnGPU(terms: Float32Array): Promise<SerialSums[]> {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const datasetCount = terms.length / TERM_COUNT;
  const words = await runWGSLModuleKernel(device, {
    id: 'million-term-sums',
    inputs: {terms},
    outputLength: datasetCount * 7,
    invocationCount: datasetCount,
    declarations: `${ADDITION_ROUNDING_ERROR_WGSL}
${FIXED_POINT_WGSL}`,
    // A test kernel: one invocation walks a million terms on purpose, as a serial prefix does.
    body: `let base = termsOffset + index * ${TERM_COUNT}u;
  // Largest term below 4096 = 2^12: biased exponent 127 + 11.
  let unitExponent = getFixedPointUnitExponent(138u, ${getFixedPointHeadroomBits(TERM_COUNT)}u);
  var plain = 0.0;
  var floatSum = 0.0;
  var floatCompensation = 0.0;
  var integerSum = 0.0;
  var integerCompensation = 0.0;
  var fixedPoint = vec3<u32>(0u, 0u, 0u);
  for (var row = 0u; row < ${TERM_COUNT}u; row++) {
    let term = terms[base + row];
    plain = plain + term;
    let floatNext = floatSum + term;
    if (abs(floatSum) >= abs(term)) {
      floatCompensation = floatCompensation + ((floatSum - floatNext) + term);
    } else {
      floatCompensation = floatCompensation + ((term - floatNext) + floatSum);
    }
    floatSum = floatNext;
    let integerNext = integerSum + term;
    integerCompensation = integerCompensation + getAdditionRoundingError(integerSum, term, integerNext);
    integerSum = integerNext;
    fixedPoint = addFixedPoint(fixedPoint, quantizeFixedPoint(term, unitExponent));
  }
  let outputBase = outputOffset + index * 7u;
  output[outputBase] = bitcast<u32>(plain);
  output[outputBase + 1u] = bitcast<u32>(floatSum);
  output[outputBase + 2u] = bitcast<u32>(floatCompensation);
  output[outputBase + 3u] = bitcast<u32>(integerSum);
  output[outputBase + 4u] = bitcast<u32>(integerCompensation);
  output[outputBase + 5u] = bitcast<u32>(integerSum + integerCompensation);
  output[outputBase + 6u] = bitcast<u32>(getFixedPointFloat(fixedPoint, unitExponent));`
  });
  const values = getFloatWords(words);
  return Array.from({length: datasetCount}, (_, dataset) => {
    const [
      plain,
      floatSum,
      floatCompensation,
      integerSum,
      integerCompensation,
      integerTotal,
      fixedPoint
    ] = values.slice(dataset * 7, dataset * 7 + 7);
    return {
      plain,
      floatSum,
      floatCompensation,
      integerSum,
      integerCompensation,
      integerTotal,
      fixedPoint
    };
  });
}

/** A million terms with mean 1234.567 and a spread of 5000, a quarter of them negative; then their negation. */
function createTerms(): {terms: Float32Array; exactSums: number[]; serialSums: number[]} {
  const random = createSeededRandom(907);
  const terms = new Float32Array(2 * TERM_COUNT);
  let scaledSum = 0;
  let serial = 0;
  for (let row = 0; row < TERM_COUNT; row++) {
    const scaled = Math.round((1234.567 + 5000 * (random() - 0.5)) * TERM_SCALE);
    terms[row] = scaled / TERM_SCALE;
    terms[TERM_COUNT + row] = -scaled / TERM_SCALE;
    // Integers below 2^22 each and 2^42 in total: this float64 sum is exact.
    scaledSum += scaled;
    // The float32 serial sum, one correctly rounded addition per term.
    serial = toFloat32(serial + terms[row]);
  }
  return {
    terms,
    exactSums: [scaledSum / TERM_SCALE, -scaledSum / TERM_SCALE],
    serialSums: [serial, -serial]
  };
}

it('a million signed terms: the integer correction recovers the float64 sum', async () => {
  const {terms, exactSums, serialSums} = createTerms();
  // Every term is an exact float32, so the reference sums the values the kernel reads.
  expect(terms[0] * TERM_SCALE).toBe(Math.round(terms[0] * TERM_SCALE));
  const results = await sumOnGPU(terms);
  for (const [dataset, result] of results.entries()) {
    const exact = exactSums[dataset];
    // The plain float32 sum is far from the true sum (8,436 for the serial order; the compiler
    // is free to reorder a plain reduction, so only the size of the error is checked).
    expect(Math.abs(result.plain - exact)).toBeGreaterThan(1000);
    // The running sum that feeds the helper is the IEEE serial sum, one rounded addition per term.
    expect(result.integerSum).toBe(serialSums[dataset]);
    // Running sum plus correction, added in float64: within 4 of the true sum of about 1.2e9
    // (the correction itself is a float32 sum of a million errors of up to 64).
    expect(Math.abs(result.integerSum + result.integerCompensation - exact)).toBeLessThan(4);
    // The float32 total is the nearest float32 or its neighbour (spacing 128 at 1.2e9).
    expect(Math.abs(result.integerTotal - exact)).toBeLessThanOrEqual(64 + 4);
    // The fixed-point accumulator holds the exact sum and rounds once.
    expect(result.fixedPoint).toBe(toFloat32(exact));
  }
});

it('the float form of the same correction is removed by the shader compiler of the test GPU', async () => {
  const {terms, exactSums} = createTerms();
  const results = await sumOnGPU(terms);
  for (const [dataset, result] of results.entries()) {
    // (sum - next) + term is algebraically zero. The compiler folds it, so the "compensated"
    // sum is the plain sum of the same kernel bit for bit, 16,115 from the true sum here. If a
    // device ever keeps the expression this expectation fails; the integer form is right on both.
    expect(result.floatCompensation).toBe(0);
    expect(result.floatSum).toBe(result.plain);
    expect(
      Math.abs(result.floatSum + result.floatCompensation - exactSums[dataset])
    ).toBeGreaterThan(1000);
    // The integer form in the same kernel and loop is not affected.
    expect(
      Math.abs(result.integerSum + result.integerCompensation - exactSums[dataset])
    ).toBeLessThan(4);
  }
});
