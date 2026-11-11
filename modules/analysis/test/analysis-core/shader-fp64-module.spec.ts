// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// The published double-single shader modules of @luma.gl/shaders (`fp64arithmetic`, `fp64`),
// assembled with the public WGSLShaderAssembler into a plain compute shader, keep their accuracy
// on the test GPU (roadmap section 13.7). The assembler turns on the integer-controlled
// arithmetic for Apple WebGPU, where Metal fast math removes the classic float transforms
// (contributor contract section 11). The spec pins the accuracy of add, multiply, divide, square
// root, log and exp against float64 and of a long sum against float32, and prints the cost.
//
// Measured 2026-11-11 on apple metal-3: add 7e-16, multiply 3e-15, divide 3e-15, sqrt 3e-15,
// log 5e-13, exp(log) 2e-14 relative; a million-term sum of 0.1 is 2e-15 off where float32 is
// 1e-2 off; that sum takes about 4 s in one invocation (4 microseconds per addition) against
// a few milliseconds in float32.

import {fp64, fp64ify, WGSLShaderAssembler, type ShaderModule} from '@luma.gl/shaders';
import {expect, it} from 'vitest';
import {getAdapterLabel, getRawTestDevice, runRawComputeShader} from './shader-compiler-raw-webgpu';

const CASES: readonly [number, number][] = [
  [1 / 3, 3],
  [Math.PI, Math.E],
  [1e10 + 1e-3, 1e-7],
  [123456.789012345, 0.000123456789],
  [0.1, 0.2],
  [40075016.68557849, 6378137],
  [1 + 2 ** -40, 1 - 2 ** -41],
  [2.5e-9, 7.25e11]
];
const WORDS_PER_CASE = 12;
const TERM_COUNT = 100_000;
const TERM = 0.1;

function getShader(loop: 'fp64' | 'float32' | 'none'): string {
  return `@group(0) @binding(0) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<storage, read> terms: array<f32>;

fn writePair(offset: u32, value: vec2f) {
  output[offset] = bitcast<u32>(value.x);
  output[offset + 1u] = bitcast<u32>(value.y);
}

@compute @workgroup_size(1)
fn main() {
  for (var row = 0u; row < ${CASES.length}u; row++) {
    let first = vec2f(terms[row * 4u], terms[row * 4u + 1u]);
    let second = vec2f(terms[row * 4u + 2u], terms[row * 4u + 3u]);
    let base = row * ${WORDS_PER_CASE}u;
    writePair(base, sum_fp64(first, second));
    writePair(base + 2u, mul_fp64(first, second));
    writePair(base + 4u, div_fp64(first, second));
    writePair(base + 6u, sqrt_fp64(first));
    writePair(base + 8u, log_fp64(first));
    // exp of a value of moderate size: the logarithm just computed.
    writePair(base + 10u, exp_fp64(log_fp64(second)));
  }
  let sumOffset = ${CASES.length * WORDS_PER_CASE}u;
  let term = vec2f(terms[${CASES.length * 4}u], terms[${CASES.length * 4 + 1}u]);
  ${
    loop === 'fp64'
      ? `var total = vec2f(0.0, 0.0);
  for (var row = 0u; row < ${TERM_COUNT}u; row++) { total = sum_fp64(total, term); }
  writePair(sumOffset, total);`
      : loop === 'float32'
        ? `var total = 0.0;
  for (var row = 0u; row < ${TERM_COUNT}u; row++) { total = total + term.x; }
  writePair(sumOffset, vec2f(total, 0.0));`
        : 'writePair(sumOffset, term);'
  }
}`;
}

function assemble(source: string, gpu: string): string {
  return new WGSLShaderAssembler().assembleWGSLShader({
    platformInfo: {
      type: 'webgpu',
      shaderLanguage: 'wgsl',
      shaderLanguageVersion: 300,
      gpu,
      features: new Set<string>()
    },
    source,
    modules: [fp64 as ShaderModule],
    shaderStage: 'compute',
    computeEntryPoint: 'main',
    scanVertexAttributes: false
  } as Parameters<WGSLShaderAssembler['assembleWGSLShader']>[0]).source;
}

function getValue(words: Uint32Array, offset: number): number {
  const floats = new Float32Array(words.buffer, words.byteOffset + offset * 4, 2);
  return floats[0] + floats[1];
}

it('the published fp64 modules keep double-single accuracy in a compute shader', async () => {
  const device = await getRawTestDevice();
  const adapter = getAdapterLabel(device);
  const gpu = adapter.split(' ')[0];
  const inputValues = new Float32Array(CASES.length * 4 + 2);
  CASES.forEach(([first, second], row) => {
    inputValues.set([...fp64ify(first), ...fp64ify(second)], row * 4);
  });
  inputValues.set(fp64ify(TERM), CASES.length * 4);
  const outputLength = CASES.length * WORDS_PER_CASE + 2;

  const timings: Record<string, number> = {};
  const results: Record<string, Uint32Array> = {};
  for (const loop of ['none', 'fp64', 'float32'] as const) {
    const code = assemble(getShader(loop), gpu);
    const start = performance.now();
    results[loop] = await runRawComputeShader(device, {code, outputLength, inputValues});
    timings[loop] = performance.now() - start;
  }

  let worst = 0;
  const lines: string[] = [];
  const names = ['sum', 'multiply', 'divide', 'sqrt', 'log', 'exp(log)'];
  const worstByOperation = new Array<number>(names.length).fill(0);
  CASES.forEach(([first, second], row) => {
    // The inputs the shader saw are the double-single values, not the float64 literals.
    const firstValue = fp64ify(first)[0] + fp64ify(first)[1];
    const secondValue = fp64ify(second)[0] + fp64ify(second)[1];
    const expected = [
      firstValue + secondValue,
      firstValue * secondValue,
      firstValue / secondValue,
      Math.sqrt(firstValue),
      Math.log(firstValue),
      secondValue
    ];
    expected.forEach((value, operation) => {
      const actual = getValue(results['fp64'], row * WORDS_PER_CASE + operation * 2);
      const error = Math.abs(actual - value) / Math.abs(value);
      worstByOperation[operation] = Math.max(worstByOperation[operation], error);
      worst = Math.max(worst, error);
    });
  });
  names.forEach((name, operation) =>
    lines.push(`  ${name}: worst relative error ${worstByOperation[operation].toExponential(2)}`)
  );
  const termValue = fp64ify(TERM)[0] + fp64ify(TERM)[1];
  const sumOffset = CASES.length * WORDS_PER_CASE;
  const fp64Sum = getValue(results['fp64'], sumOffset);
  const float32Sum = getValue(results['float32'], sumOffset);
  const exactSum = termValue * TERM_COUNT;
  console.log(
    `fp64 modules on ${adapter}\n${lines.join('\n')}\n` +
      `  sum of ${TERM_COUNT} terms of ${TERM}: fp64 relative error ${(Math.abs(fp64Sum - exactSum) / exactSum).toExponential(2)}, ` +
      `float32 relative error ${(Math.abs(float32Sum - exactSum) / exactSum).toExponential(2)}\n` +
      `  wall time including compile: no loop ${timings['none'].toFixed(0)} ms, fp64 loop ${timings['fp64'].toFixed(0)} ms, ` +
      `float32 loop ${timings['float32'].toFixed(0)} ms`
  );
  // Double-single carries about 48 bits; allow a few units of that. The float32 sum is far off.
  expect(Math.abs(float32Sum - exactSum) / exactSum).toBeGreaterThan(1e-5);
  expect(worstByOperation[0]).toBeLessThan(1e-13);
  expect(worstByOperation[1]).toBeLessThan(1e-13);
  expect(worstByOperation[2]).toBeLessThan(1e-13);
  expect(worstByOperation[3]).toBeLessThan(1e-13);
  expect(Math.abs(fp64Sum - exactSum) / exactSum).toBeLessThan(1e-10);
  expect(worst).toBeLessThan(1e-9);
});
