// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {expectMetricReadback} from './r19-readback-assertions';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {
  EXACT_ORIENTATION_WGSL,
  EXACT_ORIENTATION_MAGNITUDE_WGSL
} from '../../../src/analysis-core/kernels/exact-orientation-wgsl';
import {
  GPU_UPPER_INCOMPLETE_GAMMA_WGSL,
  GPU_CENTERED_INCOMPLETE_GAMMA_WGSL
} from '../../../src/analysis-core/kernels/incomplete-gamma-wgsl';
import {GET_NAN_WGSL, IS_NAN_BITS_WGSL} from '../../../src/analysis-core/kernels/wgsl-common';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

async function readProbe(
  declarations: string,
  body: string,
  length: number,
  inputValues?: Float32Array | Uint32Array
): Promise<number[]> {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const output = createOutputBuffer(device, length);
  const input = inputValues ? createInputBuffer(device, inputValues) : undefined;
  const graph = new GPUCommandGraph(device, {id: 'r19-shared-numerical-probe'});
  const outputView = importGraphBuffer(graph, 'output', output, 'float32', length);
  const inputFormat = inputValues instanceof Uint32Array ? 'uint32' : 'float32';
  const inputType = inputValues instanceof Uint32Array ? 'u32' : 'f32';
  const inputView =
    input && inputValues
      ? importGraphBuffer(graph, 'input', input, inputFormat, inputValues.length)
      : undefined;
  const bindings: WGSLKernelBinding[] = [
    {name: 'output', view: outputView, type: 'f32', access: 'read_write'}
  ];
  if (inputView) bindings.push({name: 'input', view: inputView, type: inputType, access: 'read'});
  graph.add({
    getCommandNodes: target => [
      createWGSLKernelNode(target, {
        id: 'probe',
        operation: 'SharedNumericalProbe',
        invocationCount: 1,
        bindings,
        declarations,
        body
      })
    ]
  });
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    return await readFloat32(output);
  } finally {
    compiled.destroy();
    output.destroy();
    input?.destroy();
  }
}

it('W3-01 exact metric magnitude readback includes cancellation and unsupported domains', async () => {
  // Exact integer arithmetic: 16384*16380 - 16382^2 = -4, reversed = +4.
  // These expectations are determinant magnitudes, independent of orientSign.
  const result = await readProbe(
    `${EXACT_ORIENTATION_WGSL}\n${EXACT_ORIENTATION_MAGNITUDE_WGSL}`,
    `
  let origin = vec2f(0.0);
  let second = vec2f(16384.0, 16382.0);
  let third = vec2f(16382.0, 16380.0);
  output[outputOffset] = orientExactDeterminant(origin, second, third);
  output[outputOffset+1u] = orientExactDeterminant(origin, third, second);
  output[outputOffset+2u] = orientExactDeterminant(origin, vec2f(1.0), vec2f(2.0));
  // Normal finite coordinates with a subnormal determinant (2^-140) are unsupported.
  output[outputOffset+3u] = orientExactDeterminant(origin, vec2f(8.470329472543003e-22,0.0), vec2f(0.0,8.470329472543003e-22));
  var subnormalBits = 1u;
  output[outputOffset+4u] = orientExactDeterminant(origin, vec2f(bitcast<f32>(subnormalBits),0.0),vec2f(0.0,1.0));
  // Products have exponent span 240 (>200), although both coordinates are normal.
  output[outputOffset+5u] = orientExactDeterminant(vec2f(1.152921504606847e18,8.673617379884035e-19),vec2f(8.673617379884035e-19,1.152921504606847e18),vec2f(1.0));
  `,
    6
  );
  expectMetricReadback(result);
}, 60000);

it('W3-02 centered gamma tail readback preserves SciPy precision', async () => {
  const result = await readProbe(
    `${GET_NAN_WGSL}\n${IS_NAN_BITS_WGSL}\n${GPU_UPPER_INCOMPLETE_GAMMA_WGSL}\n${GPU_CENTERED_INCOMPLETE_GAMMA_WGSL}`,
    `
  output[outputOffset] = getCenteredChiSquareSurvivalValue(4800.0,4096.0);
  output[outputOffset+1u] = getCenteredChiSquareSurvivalValue(11000.0,10000.0);
  output[outputOffset+2u] = getCenteredChiSquareSurvivalValue(10000.0,10000.0);
  output[outputOffset+3u] = getCenteredChiSquareSurvivalValue(65535.0,65535.0);
  `,
    4
  );
  // /usr/bin/python3: scipy.stats.chi2.sf(statistic,df).
  const expected = [
    7.928264752204537e-14, 3.61832955809647e-12, 0.49811936596618267, 0.4992653724170944
  ];
  expect(result).toHaveLength(expected.length);
  expect(Array.from(result).every(Number.isFinite)).toBe(true);
  expected.forEach((value, index) =>
    expect(Math.abs(result[index] / value - 1)).toBeLessThan(0.0002)
  );
}, 60000);

it('S3 exact determinant rejects magnitudes above float32 maximum in both signs', async () => {
  const result = await readProbe(
    `${EXACT_ORIENTATION_WGSL}\n${EXACT_ORIENTATION_MAGNITUDE_WGSL}`,
    `
  let origin = vec2f(0.0);
  let large = input[inputOffset];
  let small = input[inputOffset+1u];
  let boundary = input[inputOffset+2u];
  output[outputOffset] = orientExactDeterminant(origin,vec2f(large,small),vec2f(small,large));
  output[outputOffset+1u] = orientExactDeterminant(origin,vec2f(small,large),vec2f(large,small));
  output[outputOffset+2u] = orientExactDeterminant(origin,vec2f(large,boundary),vec2f(boundary,large));
  output[outputOffset+3u] = orientExactDeterminant(origin,vec2f(boundary,large),vec2f(large,boundary));
  `,
    4,
    Float32Array.from([2 ** 64, 2 ** 32, 2 ** 52])
  );
  // Exact determinants: +/- (2^128-2^64) are out of range;
  // +/- (2^128-2^104) are exactly the largest finite magnitudes.
  expect(result).toHaveLength(4);
  expect(
    Number.isNaN(result[0]) && Number.isNaN(result[1]),
    `overflow readback: [${result.join(', ')}]`
  ).toBe(true);
  expect(result.slice(2)).toEqual([3.4028234663852886e38, -3.4028234663852886e38]);
}, 60000);

it('B2 S4 centered gamma refuses exhausted series and handles runtime infinity', async () => {
  const result = await readProbe(
    `${GET_NAN_WGSL}\n${IS_NAN_BITS_WGSL}\n${GPU_UPPER_INCOMPLETE_GAMMA_WGSL}\n${GPU_CENTERED_INCOMPLETE_GAMMA_WGSL}`,
    `
  let value = input[inputOffset];
  let endpoints = getCenteredGammaTails(5.0,value);
  output[outputOffset] = getCenteredChiSquareSurvivalValue(value,10.0);
  output[outputOffset+1u] = endpoints.x;
  output[outputOffset+2u] = endpoints.y;
  let exhausted = getCenteredGammaTails(1000000.0,1000000.0);
  output[outputOffset+3u] = exhausted.x;
  output[outputOffset+4u] = exhausted.y;
  output[outputOffset+5u] = getCenteredChiSquareSurvivalValue(2000000.0,2000000.0);
  let invalid = getCenteredGammaTails(0.0,value);
  output[outputOffset+6u] = invalid.x;
  output[outputOffset+7u] = invalid.y;
  `,
    8,
    Float32Array.from([Infinity])
  );
  expect(result).toHaveLength(8);
  expect(result.slice(0, 3)).toEqual([0, 1, 0]);
  expect(result.slice(3).every(Number.isNaN)).toBe(true);
}, 60000);

it('T1 metric conversion isolates signed word reconstruction from coordinate accumulation', async () => {
  // Exact accumulator layouts of the existing failing coordinate fixtures, not observed GPU
  // values. Small case: abs(det)=4=(2^22)*2^-20. Large coordinates have product exponents
  // 82 and 18: abs(det)=(2^110-2^46)*2^18. The finite boundary uses exponents
  // 82 and 58: abs(det)=(2^70-2^46)*2^58, exactly maximum finite f32.
  const fixtures = [
    {magnitude: 1n << 22n, minimumExponent: -20, highest: 0, negative: true, expected: -4},
    {magnitude: 1n << 22n, minimumExponent: -20, highest: 0, negative: false, expected: 4},
    {
      magnitude: (1n << 110n) - (1n << 46n),
      minimumExponent: 18,
      highest: 3,
      negative: false,
      expected: NaN
    },
    {
      magnitude: (1n << 110n) - (1n << 46n),
      minimumExponent: 18,
      highest: 3,
      negative: true,
      expected: NaN
    },
    {
      magnitude: (1n << 70n) - (1n << 46n),
      minimumExponent: 58,
      highest: 2,
      negative: false,
      expected: 3.4028234663852886e38
    },
    {
      magnitude: (1n << 70n) - (1n << 46n),
      minimumExponent: 58,
      highest: 2,
      negative: true,
      expected: -3.4028234663852886e38
    }
  ];
  const start = EXACT_ORIENTATION_MAGNITUDE_WGSL.indexOf('  let highestBit =');
  expect(start).toBeGreaterThanOrEqual(0);
  const conversion = EXACT_ORIENTATION_MAGNITUDE_WGSL.slice(
    start,
    EXACT_ORIENTATION_MAGNITUDE_WGSL.lastIndexOf('}')
  );
  const words = Uint32Array.from(
    fixtures.flatMap(fixture =>
      Array.from({length: 8}, (_, index) =>
        Number((fixture.magnitude >> BigInt(index * 32)) & 0xffffffffn)
      )
    )
  );
  const declarations = `fn probeMetricConversion(base: u32, highest: u32, minimumExponent: i32, negativeResult: bool, lowestBit: u32) -> f32 {
    var invalidBits = 0x7fc00000u;
    let invalid = bitcast<f32>(invalidBits);
    var difference: array<u32, 8>;
    for (var slot = 0u; slot < 8u; slot++) { difference[slot] = input[base + slot]; }
    let highestWord = difference[highest];
    var lowerWord = 0u;
    if (highest > 0u) { lowerWord = difference[highest - 1u]; }
    ${conversion}
  }`;
  const body = fixtures
    .map((fixture, index) => {
      // For these exact integer fixtures, isolate the least set bit before measuring its index.
      const lowestBit = (fixture.magnitude & -fixture.magnitude).toString(2).length - 1;
      return `output[outputOffset + ${index}u] = probeMetricConversion(inputOffset + ${index * 8}u, ${fixture.highest}u, ${fixture.minimumExponent}, ${fixture.negative}, ${lowestBit}u);`;
    })
    .join('\n');
  const result = await readProbe(declarations, body, fixtures.length, words);
  expect(result).toHaveLength(fixtures.length);
  fixtures.forEach((fixture, index) => {
    if (Number.isNaN(fixture.expected))
      expect(Number.isNaN(result[index]), `conversion row ${index}`).toBe(true);
    else {
      expect(Number.isFinite(result[index]), `conversion row ${index}`).toBe(true);
      expect(result[index], `conversion row ${index}`).toBe(fixture.expected);
    }
  });
}, 60000);
