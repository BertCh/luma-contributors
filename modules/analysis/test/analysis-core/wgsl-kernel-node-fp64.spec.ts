// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// createWGSLKernelNode with the published double-single modules (`fp64`, or `fp64arithmetic`
// alone), assembled by the published WGSLShaderAssembler and run through a compiled
// GPUCommandGraph, keeps double-single accuracy for add, multiply, divide, sin and cos against
// float64 computed in JavaScript. The node always selects the integer-controlled variant, so the
// module uniform is never bound.
//
// Measured 2026-11-11 on apple metal-3 (headless Chromium): worst add 7.4e-16, multiply 3.5e-15,
// divide 2.7e-15 relative; sin 7.1e-15 and cos 2.4e-15 absolute over angles in [-1.2, 6]. The
// node construction (assembly) takes 6 to 15 ms; compile plus the first run 100 to 200 ms with a
// cold shader cache and about 10 ms with a warm one; a later run 1 to 4 ms including readback.

import type {Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {fp64, fp64arithmetic, fp64ify} from '@luma.gl/shaders';
import {expect, it} from 'vitest';
import {
  createWGSLKernelNode,
  type WGSLKernelProps
} from '../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {importGraphBuffer} from '../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  submitGraph
} from '../../src/testing/index';
import {requireTestDevice} from '../utils/require-test-device';

const ARITHMETIC_CASES: readonly [number, number][] = [
  [1 / 3, 3],
  [Math.PI, Math.E],
  [1e10 + 1e-3, 1e-7],
  [123456.789012345, 0.000123456789],
  [0.1, 0.2],
  [40075016.68557849, 6378137],
  [1 + 2 ** -40, 1 - 2 ** -41],
  [2.5e-9, 7.25e11]
];
/** Angles in radians for sin and cos, inside the range the double-single reduction serves. */
const ANGLES: readonly number[] = [
  1e-3,
  0.1,
  0.5,
  1,
  Math.PI / 3 + 1e-9,
  2.5,
  -1.2,
  3,
  6,
  -0.7853981633974483
];
const CASE_COUNT = Math.max(ARITHMETIC_CASES.length, ANGLES.length);
/** Five double-single results per row: sum, product, quotient, sine, cosine. */
const RESULTS_PER_ROW = 5;

/** Double-single value of `value` as the shader sees it. */
function getDoubleSingle(value: number): number {
  const [high, low] = fp64ify(value);
  return high + Math.fround(low);
}

it('models double-single input after float32 storage rounds the low word', () => {
  const uploaded = new Float32Array(fp64ify(Math.PI));
  expect(getDoubleSingle(Math.PI)).toBe(uploaded[0]! + uploaded[1]!);
  expect(getDoubleSingle(Math.PI)).not.toBe(Math.PI);
});

async function runKernel(
  device: Device,
  modules: NonNullable<WGSLKernelProps<unknown>['modules']>,
  withTrigonometry: boolean
): Promise<{
  values: number[];
  constructionMilliseconds: number;
  firstRunMilliseconds: number;
  secondRunMilliseconds: number;
}> {
  const inputValues = new Float32Array(CASE_COUNT * 6);
  for (let row = 0; row < CASE_COUNT; row++) {
    const [first, second] = ARITHMETIC_CASES[row % ARITHMETIC_CASES.length]!;
    const angle = ANGLES[row % ANGLES.length]!;
    inputValues.set([...fp64ify(first), ...fp64ify(second), ...fp64ify(angle)], row * 6);
  }
  const inputBuffer = createInputBuffer(device, inputValues);
  const outputBuffer = createOutputBuffer(device, CASE_COUNT * RESULTS_PER_ROW * 2);
  const graph = new GPUCommandGraph(device, {id: 'fp64-kernel-node'});
  const constructionStart = performance.now();
  graph.add(
    createWGSLKernelNode(graph, {
      id: 'fp64-kernel',
      operation: 'WGSLKernelTest',
      variant: 'fp64',
      bindings: [
        {
          name: 'terms',
          view: importGraphBuffer(graph, 'terms', inputBuffer, 'float32', inputValues.length),
          type: 'f32',
          access: 'read'
        },
        {
          name: 'results',
          view: importGraphBuffer(
            graph,
            'results',
            outputBuffer,
            'float32',
            CASE_COUNT * RESULTS_PER_ROW * 2
          ),
          type: 'f32',
          access: 'read_write'
        }
      ],
      invocationCount: CASE_COUNT,
      modules,
      declarations: `fn readPair(offset: u32) -> vec2f {
  return vec2f(terms[termsOffset + offset], terms[termsOffset + offset + 1u]);
}
fn writePair(offset: u32, value: vec2f) {
  results[resultsOffset + offset] = value.x;
  results[resultsOffset + offset + 1u] = value.y;
}`,
      body: `let first = readPair(index * 6u);
  let second = readPair(index * 6u + 2u);
  let angle = readPair(index * 6u + 4u);
  let base = index * ${RESULTS_PER_ROW * 2}u;
  writePair(base, sum_fp64(first, second));
  writePair(base + 2u, mul_fp64(first, second));
  writePair(base + 4u, div_fp64(first, second));
  ${
    withTrigonometry
      ? `writePair(base + 6u, sin_fp64(angle));
  writePair(base + 8u, cos_fp64(angle));`
      : '_ = angle;'
  }`
    })
  );
  const constructionMilliseconds = performance.now() - constructionStart;
  let start = performance.now();
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const values = await readFloat32(outputBuffer);
  const firstRunMilliseconds = performance.now() - start;
  start = performance.now();
  submitGraph(device, compiled, undefined);
  await readFloat32(outputBuffer);
  const secondRunMilliseconds = performance.now() - start;
  compiled.destroy();
  inputBuffer.destroy();
  outputBuffer.destroy();
  return {values, constructionMilliseconds, firstRunMilliseconds, secondRunMilliseconds};
}

/** Largest error per operation: relative for arithmetic, absolute for sin and cos. */
function getWorstErrors(values: number[], withTrigonometry: boolean): number[] {
  const worst = new Array<number>(withTrigonometry ? 5 : 3).fill(0);
  for (let row = 0; row < CASE_COUNT; row++) {
    const [first, second] = ARITHMETIC_CASES[row % ARITHMETIC_CASES.length]!.map(getDoubleSingle);
    const angle = getDoubleSingle(ANGLES[row % ANGLES.length]!);
    const expected = [first! + second!, first! * second!, first! / second!];
    if (withTrigonometry) expected.push(Math.sin(angle), Math.cos(angle));
    expected.forEach((value, operation) => {
      const offset = row * RESULTS_PER_ROW * 2 + operation * 2;
      const actual = values[offset]! + values[offset + 1]!;
      const error =
        operation < 3 ? Math.abs(actual - value) / Math.abs(value) : Math.abs(actual - value);
      worst[operation] = Math.max(worst[operation]!, error);
    });
  }
  return worst;
}

it('createWGSLKernelNode runs the published fp64 modules at double-single accuracy', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const full = await runKernel(device, [fp64], true);
  const arithmetic = await runKernel(device, [fp64arithmetic], false);
  const fullErrors = getWorstErrors(full.values, true);
  const arithmeticErrors = getWorstErrors(arithmetic.values, false);
  // Measured about 3e-15 at worst; float32 would be near 6e-8. The bounds leave a factor of 30.
  const [sumError, productError, quotientError, sineError, cosineError] = fullErrors;
  expect(sumError, 'fp64 add, relative').toBeLessThan(1e-13);
  expect(productError, 'fp64 multiply, relative').toBeLessThan(1e-13);
  expect(quotientError, 'fp64 divide, relative').toBeLessThan(1e-13);
  expect(sineError, 'fp64 sin, absolute').toBeLessThan(1e-13);
  expect(cosineError, 'fp64 cos, absolute').toBeLessThan(1e-13);
  expect(arithmeticErrors, 'fp64arithmetic errors').toEqual(fullErrors.slice(0, 3));
  for (let row = 0; row < CASE_COUNT; row++) {
    const offset = row * RESULTS_PER_ROW * 2;
    expect(
      arithmetic.values.slice(offset, offset + 6),
      `fp64arithmetic alone gives the same arithmetic result words for row ${row}`
    ).toEqual(full.values.slice(offset, offset + 6));
  }
});
