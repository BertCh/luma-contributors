// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {WgslExec, WgslParser} from 'wgsl_reflect';
import {TERRAIN_TILED_WIDE_SUM_WGSL} from '../../../src/gpu-terrain/terrain-processing/terrain-tiled-wide-sum';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';

it('preserves float32 terms across 320-bit addition and prefix subtraction', () => {
  const cases = [
    [0, 0, 0],
    [1e30, 1, 1e-20],
    [1e-30, 1e30, 1],
    [3.4028234663852886e38, 1, 1.401298464324817e-45],
    [16777216, 1, 1],
    [1, 2 ** -24, 2 ** -24],
    [1e-20, 1e-20, 1e-20]
  ].map(values => values.map(Math.fround));
  const input = Float32Array.from(cases.flat());
  const output = new Float32Array(cases.length * 3);
  const limbs = new Uint32Array(cases.length * 30);
  const source = `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read_write> limbs: array<u32>;
${TERRAIN_TILED_WIDE_SUM_WGSL}
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
 let offset = invocation.x * 3u;
 let encodedFirst = terrainEncodeRunoff(input[offset], 1u);
 let encodedSecond = terrainEncodeRunoff(input[offset + 1u], 1u);
 let encodedThird = terrainEncodeRunoff(input[offset + 2u], 1u);
 let total = terrainAddRunoff(terrainAddRunoff(encodedFirst, encodedSecond), encodedThird);
 var totalWords = total;
 var remainingWords = terrainSubtractRunoff(total, encodedFirst);
 var finalWords = terrainSubtractRunoff(remainingWords, encodedSecond);
 for (var word = 0u; word < 10u; word++) {
  limbs[invocation.x * 30u + word] = totalWords[word];
  limbs[invocation.x * 30u + 10u + word] = remainingWords[word];
  limbs[invocation.x * 30u + 20u + word] = finalWords[word];
 }
 output[offset] = terrainDecodeRunoff(total);
 output[offset + 1u] = terrainDecodeRunoff(terrainSubtractRunoff(total, encodedFirst));
 output[offset + 2u] = terrainDecodeRunoff(terrainSubtractRunoff(terrainSubtractRunoff(total, encodedFirst), encodedSecond));
}`;
  expect(auditWGSLModule(source)).toEqual([]);
  new WgslExec(new WgslParser().parse(source)).dispatchWorkgroups('main', [cases.length, 1, 1], {
    0: {0: input, 1: output, 2: limbs}
  });
  const encode = (value: number) => {
    const bits = new Uint32Array(Float32Array.of(value).buffer)[0];
    const exponent = (bits >>> 23) & 255;
    const significand = BigInt((bits & 0x7fffff) | (exponent === 0 ? 0 : 0x800000));
    return significand << BigInt(exponent === 0 ? 0 : exponent - 1);
  };
  expect(Array.from(limbs)).toEqual(
    cases.flatMap(values => {
      const [first, second, third] = values.map(encode);
      return [first + second + third, second + third, third].flatMap(value =>
        Array.from({length: 10}, (_, word) => Number((value >> BigInt(word * 32)) & 0xffffffffn))
      );
    })
  );
  // wgsl_reflect interprets unsigned >> as signed when the high bit is set. The last case
  // exercises those exact limbs above; real GPU decoding is covered by the dynamic-range spec.
  expect(Array.from(output.slice(0, -3))).toEqual(
    cases
      .slice(0, -1)
      .flatMap(([first, second, third]) => [
        Math.fround(first + second + third),
        Math.fround(second + third),
        third
      ])
  );
});
