// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUSpatialWeightsTracesOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-weights-traces-operation';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';

const float32Words = (values: ArrayLike<number>) =>
  Array.from(new Uint32Array(Float32Array.from(values).buffer));

async function runGram(
  device: Device,
  offsets: Uint32Array,
  neighbors: Uint32Array,
  weights: Float32Array,
  mode: 'exact' | 'stochastic',
  tileRowCount: number
) {
  const result = await runOperationGraph(device, {
    inputs: {
      offsets: {format: 'uint32', values: offsets},
      neighbors: {format: 'uint32', values: neighbors},
      weights: {format: 'float32', values: weights}
    },
    operations: [
      new GPUSpatialWeightsTracesOperation({
        rowCount: offsets.length - 1,
        maximumPower: 4,
        mode,
        probeCount: 3,
        probeBatchSize: 2,
        tileRowCount,
        includeGramTraces: true,
        weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
        output: {powerTraces: 'traces', powerSums: 'sums', gramTraces: 'gram'}
      })
    ],
    outputs: ['gram']
  });
  expect(result['gram']).toHaveLength(4);
  return result['gram'];
}

it.each([
  [2, 2],
  [1e38, 9.999999680285692e37],
  [3.2e38, 3.19999997882106e38]
])('N1 GPU traces retain scale and bounded powers for edge weight %s', async (weight, norm) => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const result = await runOperationGraph(device, {
    inputs: {
      offsets: {format: 'uint32', values: Uint32Array.of(0, 1, 2)},
      neighbors: {format: 'uint32', values: Uint32Array.of(1, 0)},
      weights: {format: 'float32', values: Float32Array.of(weight, weight)}
    },
    operations: [
      new GPUSpatialWeightsTracesOperation({
        rowCount: 2,
        maximumPower: 4,
        mode: 'exact',
        weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
        output: {
          powerTraces: 'traces',
          powerSums: 'sums',
          infinityNorm: 'norm',
          status: {approximation: 'approximation'}
        }
      })
    ],
    outputs: ['norm', 'traces', 'sums', 'approximation']
  });
  // In Python: convert float32 weights to NumPy float64, divide the matrix by its
  // infinity norm, and take powers. The normalized matrix swaps the two rows.
  expect(Array.from(result['norm'])).toEqual([norm]);
  expect(Array.from(result['approximation'])).toEqual([0]);
  const expectedTraces = [0, 2, 0, 2];
  if (weight === 2) {
    expect(float32Words(result['traces'])).toEqual([0, 0x40000000, 0, 0x40000000]);
    expect(float32Words(result['sums'])).toEqual(Array(4).fill(0x40000000));
    return;
  }
  expect(result['traces']).toHaveLength(expectedTraces.length);
  expect(result['sums']).toHaveLength(expectedTraces.length);
  expectedTraces.forEach((trace, index) => {
    expect(Number.isFinite(result['traces'][index])).toBe(true);
    expect(Number.isFinite(result['sums'][index])).toBe(true);
    // Four powers can accumulate float32 division and multiplication roundoff.
    expect(Math.abs(result['traces'][index] - trace)).toBeLessThan(2e-6);
    expect(Math.abs(result['sums'][index] - 2)).toBeLessThan(2e-6);
  });
});

it.each([
  'exact',
  'stochastic'
] as const)('B1 GPU preserves all four Gram traces in %s mode', async mode => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // /usr/bin/python3, NumPy: g = w.T @ w on float64-converted float32 weights.
  // Round [trace(g), trace(g @ w), trace(g @ g), sum(diag(g)**2)] to float32.
  for (const {edge, loop, expected} of [
    {edge: 1.875 * 2 ** 127, loop: 2, expected: [Infinity, 8, Infinity, Infinity]},
    {edge: 2 ** 30, loop: 2 ** -30, expected: [2 ** 60, 2 ** -90, 2 ** 120, 2 ** 120]},
    {edge: 4, loop: 2, expected: [20, 8, 272, 272]}
  ]) {
    const gram = await runGram(
      device,
      Uint32Array.of(0, 1, 1, 2),
      Uint32Array.of(1, 2),
      Float32Array.of(edge, loop),
      mode,
      1
    );
    if (edge === 4) {
      // Exact integer moments [20, 8, 272, 272] also reproduce the baseline GPU words.
      expect(float32Words(gram)).toEqual([0x41a00000, 0x41000000, 0x43880000, 0x43880000]);
      continue;
    }
    expected.forEach((value, index) => {
      const actual = gram[index];
      if (value === Infinity) expect(actual).toBe(Infinity);
      else expect(Math.abs(actual / value - 1)).toBeLessThan(2e-6);
    });
    if (mode === 'exact' && loop === 2) expect(gram[1]).toBe(8);
  }
});

it.each([
  'exact',
  'stochastic'
] as const)('B1 cubic retains cancellation terms in %s mode', async mode => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const epsilon = 2 ** -12;
  const gram = await runGram(
    device,
    Uint32Array.of(0, 1, 3, 4),
    Uint32Array.of(0, 0, 2, 0),
    Float32Array.of(epsilon, -1, 1, epsilon),
    mode,
    3
  );
  // W² has one nonzero column [epsilon², 0, epsilon²]. Each probe gives 2 epsilon³.
  expect(float32Words(gram)[1]).toBe(0x2e000000);
  expect(Array.from(gram).every(Number.isFinite)).toBe(true);
});

it('B2 exactly retains the baseline quartic word for W = [[0.1, 0.3], [0, 0]]', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const gram = await runGram(
    device,
    Uint32Array.of(0, 2, 2),
    Uint32Array.of(0, 1),
    Float32Array.of(0.1, 0.3),
    'exact',
    2
  );
  // Exact rational (w0² + w1²)² using the two stored float32 weights.
  expect(float32Words(gram)[2]).toBe(0x3c23d70c);
});
