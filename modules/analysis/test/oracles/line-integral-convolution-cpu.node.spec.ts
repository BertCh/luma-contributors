// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {convolveLineIntegralOnCPU} from './line-integral-convolution-cpu';
import {getGPULineIntegralConvolutionParameterValues} from '../../src/raster-analysis/flow-texture/line-integral-convolution-parameters';

it('B1 FLT-10 rounds oblique normalization before flooring the landing', () => {
  const result = convolveLineIntegralOnCPU(
    {
      width: 4,
      height: 4,
      velocities: Float32Array.from({length: 32}, (_, index) => (index % 2 ? 4 : 3))
    },
    4,
    4,
    1,
    [0, 0, 3, 3, 0, 0, 3, 3, 2.5, 0, 0, 0],
    31
  );
  // The 3-4-5 direction travels [1.5,2] cells from [.5,.5], reaching [2,2.5].
  // Reverse travel leaves the field; weights 1 and .5 combine noise pixels 0 and 10.
  expect(Number.isFinite(result.values[0])).toBe(true);
  expect(result.values[0]).toBe(0.33969104290008545);
  expect(result.speeds[0]).toBe(5);
});

it.each([
  'world',
  'local'
] as const)('FLT-10 traces decimal cells in output coordinates with a %s packet', frame => {
  const parameters =
    frame === 'world'
      ? Float32Array.of(13, -5, 0.1, 1, 13, -5, 0.1, 1, 0.5, 0, 0, 0)
      : getGPULineIntegralConvolutionParameterValues({
          fieldPlacement: {origin: [13, -5], cellSize: [0.1, 1]},
          outputPlacement: {origin: [13, -5], cellSize: [0.1, 1]},
          localOrigin: [13, -5]
        });
  const result = convolveLineIntegralOnCPU(
    {
      width: 7,
      height: 1,
      velocities: Float32Array.from({length: 14}, (_, index) => (index % 2 ? 0 : 1))
    },
    7,
    1,
    1,
    parameters,
    3
  );
  // Independent integer Philox4x32-10 (Python unbounded integer products) gives
  // noise [.3374575972557068,.47517913579940796,.15463626384735107,
  // .7811055183410645,.3611893653869629,.7109882831573486,.21382975578308105].
  // Centres c+.5 land at c and c+1 with Hann weights .5 each; the final
  // forward landing leaves the field. f32((1.5*noise[c]+.5*noise[c+1])/2):
  expect(Array.from(result.values)).toEqual([
    0.3718879818916321, 0.39504343271255493, 0.3112535774707794, 0.6761264801025391,
    0.4486390948295593, 0.5866986513137817, 0.21382975578308105
  ]);
  expect(Array.from(result.speeds)).toEqual([1, 1, 1, 1, 1, 1, 1]);
});

it('FLT-10 preserves the seven-cell reviewer fixture at local origin [22,-8]', () => {
  const result = convolveLineIntegralOnCPU(
    {
      width: 7,
      height: 1,
      velocities: Float32Array.from({length: 14}, (_, index) => (index % 2 ? 0 : 1))
    },
    7,
    1,
    1,
    getGPULineIntegralConvolutionParameterValues({
      fieldPlacement: {origin: [22, -8], cellSize: [0.1, 1]},
      outputPlacement: {origin: [22, -8], cellSize: [0.1, 1]},
      localOrigin: [22, -8]
    }),
    51
  );
  // Independent Python integer Philox4x32-10 gives noise[3]=.24642854928970337
  // and noise[4]=.8115411996841431. Half-pixel steps from 3.5 land in cells 3/4;
  // Hann weights are 1 at the centre and .5 per step. f32((1.5*n3+.5*n4)/2):
  expect(Number.isFinite(result.values[3])).toBe(true);
  expect(result.values[3]).toBe(0.3877066969871521);
  expect(result.speeds[3]).toBe(1);
});

function expectThresholdLICValues(values: Float32Array): void {
  expect(values).toHaveLength(4);
  expect(Number.isFinite(values[0])).toBe(true);
  expect(values[0]).toBe(0.782850980758667);
}

it('RT-124 rejects a slow RK2 landing before adding its noise weight', () => {
  const result = convolveLineIntegralOnCPU(
    {width: 4, height: 1, velocities: new Float32Array([1, 0, 1, 0, 0, 0, 0, 0])},
    4,
    1,
    1,
    [0, 0, 1, 1, 0, 0, 1, 1, 1.375, 0.75, 0, 0],
    17
  );
  // Centre x=.5 and midpoint x=1.1875 both have speed 1. The landing x=1.875
  // interpolates 1 and 0 with fraction .375, so speed .625 is below .75.
  // The reverse landing is outside the field. Only the pinned Philox centre remains.
  expectThresholdLICValues(result.values);
});

it.each([1, 5])('S1 rejects a threshold LIC result with %i pixels instead of four', length => {
  const values = new Float32Array(length);
  values[0] = 0.782850980758667;
  expect(() => expectThresholdLICValues(values)).toThrow();
});

it('B2 RT-124 admits a landing whose float32 speed equals minimumSpeed', () => {
  const result = convolveLineIntegralOnCPU(
    {
      width: 3,
      height: 1,
      velocities: Float32Array.of(2, 0, 2, 0, 0.5, Math.fround(Math.sqrt(3) / 2))
    },
    3,
    1,
    1,
    [0, 0, 1, 1, 0, 0, 1, 1, 2, 1, 0, 0],
    9
  );
  // Landing x=2.5 is the third centre. In binary32, .5^2 + f32(sqrt(3)/2)^2
  // and its square root both round to 1, admitting the equality boundary.
  // Hann weights 1 and .5 combine the centre and admitted landing's Philox noise
  // into the independently pinned binary32 weighted result below.
  expect(result.values).toHaveLength(3);
  expect(Number.isFinite(result.values[0])).toBe(true);
  expect(result.values[0]).toBe(0.26045188307762146);
});
