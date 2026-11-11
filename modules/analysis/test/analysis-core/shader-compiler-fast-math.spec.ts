// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getAdapterLabel, getRawTestDevice, runRawComputeShader} from './shader-compiler-raw-webgpu';

// What Metal fast math does to float32 WGSL, in one plain WebGPU shader with no library code
// (contributor contract section 11; dev-docs/research/math-correctness-review-2026-11-10/
// shader-compiler-findings.md). On the test GPU the browser compiles its Metal source with
// `fastMathEnabled`, and three things follow:
//   (a) the float compensation of Kahan, Neumaier and the two-sum is removed;
//   (b) `(value - shift) - small` is computed as `value - (shift + small)`;
//   (d) a quotient can be one unit in the last place off the correctly rounded one.
// The same Metal source compiled outside the browser with fast math off gives the IEEE result for
// all three, so they are one cause. The browser offers no switch for it.
//
// Expected values are IEEE float32 arithmetic done here with Math.fround. Each test pins what
// the test GPU does and skips on an adapter that computes the expression as written.

const TERM_COUNT = 1_000_000;
const RESIDUAL_COUNT = 16;
const DIVISION_COUNT = 2048;
const PAIR_OFFSET = TERM_COUNT;
const CHAIN_OFFSET = PAIR_OFFSET + 2;
const VALUES_OFFSET = CHAIN_OFFSET + 2;
const DIVISION_OFFSET = VALUES_OFFSET + RESIDUAL_COUNT;
const INPUT_LENGTH = DIVISION_OFFSET + 2 * DIVISION_COUNT;
const SUM_WORD_COUNT = 6;
const OUTPUT_LENGTH = SUM_WORD_COUNT + 2 * RESIDUAL_COUNT + DIVISION_COUNT;
/** The float32 step between 8192 and 16384. */
const STEP_AT_SHIFT = 2 ** -10;

const SHADER = `@group(0) @binding(0) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<storage, read> terms: array<f32>;
@compute @workgroup_size(1)
fn main() {
  // (a) Plain, Kahan and Neumaier sums of the same terms, and the two-sum of one pair.
  var plain = 0.0;
  var kahanSum = 0.0;
  var kahanCompensation = 0.0;
  var neumaierSum = 0.0;
  var neumaierCompensation = 0.0;
  for (var row = 0u; row < ${TERM_COUNT}u; row++) {
    let term = terms[row];
    plain = plain + term;
    let corrected = term - kahanCompensation;
    let kahanNext = kahanSum + corrected;
    kahanCompensation = (kahanNext - kahanSum) - corrected;
    kahanSum = kahanNext;
    let neumaierNext = neumaierSum + term;
    if (abs(neumaierSum) >= abs(term)) {
      neumaierCompensation = neumaierCompensation + ((neumaierSum - neumaierNext) + term);
    } else {
      neumaierCompensation = neumaierCompensation + ((term - neumaierNext) + neumaierSum);
    }
    neumaierSum = neumaierNext;
  }
  let first = terms[${PAIR_OFFSET}u];
  let second = terms[${PAIR_OFFSET + 1}u];
  let sum = first + second;
  let virtualSecond = sum - first;
  let virtualFirst = sum - virtualSecond;
  let twoSumError = (first - virtualFirst) + (second - virtualSecond);
  output[0] = bitcast<u32>(plain);
  output[1] = bitcast<u32>(kahanSum);
  output[2] = bitcast<u32>(kahanCompensation);
  output[3] = bitcast<u32>(neumaierSum);
  output[4] = bitcast<u32>(neumaierCompensation);
  output[5] = bitcast<u32>(twoSumError);
  // (b) (value - shift) - small with a shift and a small term that do not change in the loop,
  // as one expression and as two statements.
  let shift = terms[${CHAIN_OFFSET}u];
  let small = terms[${CHAIN_OFFSET + 1}u];
  for (var row = 0u; row < ${RESIDUAL_COUNT}u; row++) {
    let value = terms[${VALUES_OFFSET}u + row];
    output[${SUM_WORD_COUNT}u + row] = bitcast<u32>((value - shift) - small);
    let difference = value - shift;
    let residual = difference - small;
    output[${SUM_WORD_COUNT + RESIDUAL_COUNT}u + row] = bitcast<u32>(residual);
  }
  // (d) Quotients of operands read from a buffer.
  for (var pair = 0u; pair < ${DIVISION_COUNT}u; pair++) {
    let numerator = terms[${DIVISION_OFFSET}u + 2u * pair];
    let denominator = terms[${DIVISION_OFFSET}u + 2u * pair + 1u];
    output[${SUM_WORD_COUNT + 2 * RESIDUAL_COUNT}u + pair] = bitcast<u32>(numerator / denominator);
  }
}`;

/** Deterministic uniform random numbers in `[0, 1)` (mulberry32). */
function createSeededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function createInputs(): Float32Array {
  const random = createSeededRandom(2311);
  const inputs = new Float32Array(INPUT_LENGTH);
  for (let row = 0; row < TERM_COUNT; row++) {
    // Multiples of 1/1024 below 8192, a quarter of them negative: every term is an exact
    // float32 and the exact sum is an integer number of 1/1024.
    const magnitude = Math.round(1234.567 * 4 * random() * 1024) / 1024;
    inputs[row] = random() < 0.25 ? -magnitude : magnitude;
  }
  // 2^24 + 1 is a tie that rounds to 2^24: the rounding error of the addition is exactly 1.
  inputs[PAIR_OFFSET] = 16777216;
  inputs[PAIR_OFFSET + 1] = 1;
  // Values a few float32 steps above 10000, and a small term of three quarters of a step:
  // value - shift is exact, and shift + small rounds up by a whole step.
  inputs[CHAIN_OFFSET] = 10000;
  inputs[CHAIN_OFFSET + 1] = 0.75 * STEP_AT_SHIFT;
  for (let row = 0; row < RESIDUAL_COUNT; row++) {
    inputs[VALUES_OFFSET + row] = 10000 + (row + 3) * STEP_AT_SHIFT;
  }
  inputs[DIVISION_OFFSET] = 6;
  inputs[DIVISION_OFFSET + 1] = 3;
  for (let pair = 1; pair < DIVISION_COUNT; pair++) {
    inputs[DIVISION_OFFSET + 2 * pair] = pair < 64 ? pair : 1 + 999 * random();
    inputs[DIVISION_OFFSET + 2 * pair + 1] = pair < 64 ? 3 : 1 + 99 * random();
  }
  return inputs;
}

type Measurement = {
  adapterLabel: string;
  inputs: Float32Array;
  words: Uint32Array;
  values: Float32Array;
};

let measurement: Promise<Measurement> | undefined;

/** Runs the shader once for the three tests. */
function getMeasurement(): Promise<Measurement> {
  measurement ??= (async () => {
    const device = await getRawTestDevice();
    const inputs = createInputs();
    const words = await runRawComputeShader(device, {
      code: SHADER,
      outputLength: OUTPUT_LENGTH,
      inputValues: inputs
    });
    return {
      adapterLabel: getAdapterLabel(device),
      inputs,
      words,
      values: new Float32Array(words.buffer, words.byteOffset, words.length)
    };
  })();
  return measurement;
}

it('(a) Kahan, Neumaier and two-sum compensation terms are removed on the test GPU', async context => {
  const {adapterLabel, inputs, values} = await getMeasurement();
  const [plain, kahanSum, kahanCompensation, neumaierSum, neumaierCompensation, twoSumError] =
    values;
  // IEEE float32, one rounded operation at a time.
  let exactScaledSum = 0;
  let serialSum = 0;
  let serialCompensation = 0;
  for (let row = 0; row < TERM_COUNT; row++) {
    const term = inputs[row];
    // Integers below 2^23 each and below 2^43 in total: exact in float64.
    exactScaledSum += term * 1024;
    const next = Math.fround(serialSum + term);
    serialCompensation = Math.fround(
      serialCompensation +
        (Math.abs(serialSum) >= Math.abs(term)
          ? Math.fround(Math.fround(serialSum - next) + term)
          : Math.fround(Math.fround(term - next) + serialSum))
    );
    serialSum = next;
  }
  const exactSum = exactScaledSum / 1024;
  // The inputs are such that a plain float32 sum is visibly wrong and Neumaier's is not.
  expect(Math.abs(serialSum - exactSum)).toBeGreaterThan(1000);
  expect(Math.abs(serialSum + serialCompensation - exactSum)).toBeLessThan(1);

  if (twoSumError === 1) {
    // The adapter computes the expressions as written: the sums are then the IEEE ones.
    expect(neumaierSum).toBe(serialSum);
    expect(neumaierCompensation).toBe(serialCompensation);
    context.skip(`${adapterLabel} keeps float compensation terms`);
  }
  // Measured on the test GPU: every compensation is exactly zero and both "compensated" sums
  // are the plain sum of the same loop bit for bit, more than 1000 from the true sum.
  expect(twoSumError).toBe(0);
  expect(kahanCompensation).toBe(0);
  expect(neumaierCompensation).toBe(0);
  expect(kahanSum).toBe(plain);
  expect(neumaierSum).toBe(plain);
  expect(Math.abs(plain - exactSum)).toBeGreaterThan(1000);
});

it('(b) (value - shift) - small is computed as value - (shift + small) on the test GPU', async context => {
  const {adapterLabel, inputs, values} = await getMeasurement();
  const shift = inputs[CHAIN_OFFSET];
  const small = inputs[CHAIN_OFFSET + 1];
  const asWritten: number[] = [];
  const regrouped: number[] = [];
  for (let row = 0; row < RESIDUAL_COUNT; row++) {
    const value = inputs[VALUES_OFFSET + row];
    asWritten.push(Math.fround(Math.fround(value - shift) - small));
    regrouped.push(Math.fround(value - Math.fround(shift + small)));
  }
  // Hand values of the first row: 3 steps - 0.75 step = 2.25 steps as written; shift + small
  // rounds to shift + 1 step, so the regrouped form gives 2 steps.
  expect(asWritten[0]).toBe(2.25 * STEP_AT_SHIFT);
  expect(regrouped[0]).toBe(2 * STEP_AT_SHIFT);
  const expression = Array.from(values.slice(SUM_WORD_COUNT, SUM_WORD_COUNT + RESIDUAL_COUNT));
  const statements = Array.from(
    values.slice(SUM_WORD_COUNT + RESIDUAL_COUNT, SUM_WORD_COUNT + 2 * RESIDUAL_COUNT)
  );
  if (expression.every((value, row) => value === asWritten[row])) {
    expect(statements).toEqual(asWritten);
    context.skip(`${adapterLabel} subtracts in the order written`);
  }
  // Measured on the test GPU: all 16 rows regrouped, and a statement boundary does not help.
  expect(expression).toEqual(regrouped);
  expect(statements).toEqual(regrouped);
});

it('(d) float division is up to one unit in the last place off on the test GPU', async context => {
  const {adapterLabel, inputs, words, values} = await getMeasurement();
  const expectedWords = new Uint32Array(1);
  const expectedValues = new Float32Array(expectedWords.buffer);
  let mismatchCount = 0;
  let largestDifference = 0;
  for (let pair = 0; pair < DIVISION_COUNT; pair++) {
    // The float64 quotient of two float32 values rounds to the correctly rounded float32 one.
    expectedValues[0] = inputs[DIVISION_OFFSET + 2 * pair] / inputs[DIVISION_OFFSET + 2 * pair + 1];
    const word = words[SUM_WORD_COUNT + 2 * RESIDUAL_COUNT + pair];
    if (word !== expectedWords[0]) {
      mismatchCount++;
      largestDifference = Math.max(largestDifference, Math.abs(word - expectedWords[0]));
    }
  }
  // 6 / 3 is exact in this shader on every adapter tried.
  expect(values[SUM_WORD_COUNT + 2 * RESIDUAL_COUNT]).toBe(2);
  if (mismatchCount === 0) {
    context.skip(`${adapterLabel} rounds division correctly`);
  }
  // Measured on the test GPU: 551 of the 2048 quotients, each by one unit in the last place
  // (WGSL allows 2.5). The count is not pinned exactly; the size of the error is.
  expect(mismatchCount).toBeGreaterThan(100);
  expect(largestDifference).toBe(1);
});
