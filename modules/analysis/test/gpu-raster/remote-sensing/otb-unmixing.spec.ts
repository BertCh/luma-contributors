// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPURasterSpectralUnmixing} from '../../../src/raster-analysis/remote-sensing/index';
import fixture from '../../oracles/fixtures/raster-otb-unmixing.json';
import {requireTestDevice} from '../../utils/require-test-device';
import {createRemoteSensingHarness} from './remote-sensing-test-utils';

function decodeFloat32(encoded: string): number[] {
  const bytes = Uint8Array.from(atob(encoded), character => character.charCodeAt(0));
  const view = new DataView(bytes.buffer);
  return Array.from({length: bytes.byteLength / 4}, (_, index) => view.getFloat32(index * 4, true));
}

it('matches all 256 original 224-band pixels against OTB 9.1.1 published UCLS abundances', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createRemoteSensingHarness(device);
  const result = await harness.run(
    new GPURasterSpectralUnmixing({
      cellCount: fixture.cellCount,
      bandCount: fixture.bandCount,
      endmemberCount: fixture.endmemberCount,
      values: harness.input(decodeFloat32(fixture.values)),
      endmembers: harness.input(decodeFloat32(fixture.endmembers)),
      constraint: 'unconstrained',
      output: {
        abundances: harness.output('abundances', 'float32', fixture.expectedAbundances.length),
        residual: harness.output('residual', 'float32', fixture.cellCount),
        status: harness.output('status', 'uint32', fixture.cellCount),
        iterationCounts: harness.output('iterations', 'uint32', fixture.cellCount)
      }
    })
  );
  expect(result['status']).toEqual(Array(fixture.cellCount).fill(0));
  expect(result['iterations']).toEqual(Array(fixture.cellCount).fill(0));
  const maximumError = Math.max(
    ...result['abundances'].map((value, index) =>
      Math.abs(value - fixture.expectedAbundances[index])
    )
  );
  expect(maximumError).toBeLessThan(2e-5);
  // Upstream fixture includes noisy spectra and negative UCLS coefficients.
  expect(Math.min(...result['abundances'])).toBeLessThan(-0.01);
  expect(result['residual'].every(value => value > 0.003 && value < 0.006)).toBe(true);
  console.log(JSON.stringify({reference: fixture.provenance.reference, maximumError}));
});

it.each([
  {bandCount: 256, endmemberCount: 6},
  {bandCount: 219, endmemberCount: 8}
])('solves maximum supported $bandCount-band / $endmemberCount-endmember signed spectra', async ({
  bandCount,
  endmemberCount
}) => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Repeated orthogonal columns, with every band carrying information.
  const endmembers = Array.from({length: bandCount * endmemberCount}, (_, index) => {
    const member = Math.floor(index / bandCount);
    const band = index % bandCount;
    if (bandCount !== 256) return band % endmemberCount === member ? 1 : 0;
    const overlap = member & band;
    const parity = (overlap & 1) + ((overlap >> 1) & 1) + ((overlap >> 2) & 1);
    return parity % 2 ? -1 : 1;
  });
  for (const suppliedCoefficients of [
    [0.25, -0.5, 0.125, 0.75, 1, -0.25, 0.5, 0.0625],
    [-0.5, 0.25, -0.75, 0.125, -0.0625, 1, 0.5, 0]
  ]) {
    const coefficients = suppliedCoefficients.slice(0, endmemberCount);
    const harness = createRemoteSensingHarness(device);
    const values = Array.from({length: bandCount}, (_, band) =>
      coefficients.reduce(
        (sum, value, member) => sum + value * endmembers[member * bandCount + band],
        0
      )
    );
    const result = await harness.run(
      new GPURasterSpectralUnmixing({
        cellCount: 1,
        bandCount,
        endmemberCount,
        values: harness.input(values),
        endmembers: harness.input(endmembers),
        output: {
          abundances: harness.output('abundances', 'float32', endmemberCount),
          residual: harness.output('residual', 'float32', 1),
          status: harness.output('status', 'uint32', 1),
          iterationCounts: harness.output('iterations', 'uint32', 1)
        }
      })
    );
    expect(result['status']).toEqual([0]);
    for (let member = 0; member < endmemberCount; member++) {
      expect(result['abundances'][member]).toBeCloseTo(coefficients[member], 6);
    }
    expect(result['residual'][0]).toBeLessThan(1e-8);
  }
});

it.each(
  (['nonnegative', 'simplex'] as const).flatMap(constraint => [
    {constraint, bandCount: 256, endmemberCount: 5},
    {constraint, bandCount: 195, endmemberCount: 8}
  ])
)('preserves $constraint convergence at $bandCount bands and $endmemberCount endmembers', async ({
  constraint,
  bandCount,
  endmemberCount
}) => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const expected =
    endmemberCount === 5
      ? [0.125, 0.25, 0.375, 0.25, 0]
      : [0.0625, 0.125, 0.1875, 0.25, 0, 0.125, 0.25, 0];
  const endmembers = Array.from({length: bandCount * endmemberCount}, (_, index) =>
    Math.floor(index / bandCount) === (index % bandCount) % endmemberCount ? 1 : 0
  );
  const values = Array.from({length: bandCount}, (_, band) => expected[band % endmemberCount]);
  const harness = createRemoteSensingHarness(device);
  const result = await harness.run(
    new GPURasterSpectralUnmixing({
      cellCount: 1,
      bandCount,
      endmemberCount,
      constraint,
      maximumIterations: 512,
      tolerance: 1e-7,
      values: harness.input(values),
      endmembers: harness.input(endmembers),
      output: {
        abundances: harness.output('abundances', 'float32', endmemberCount),
        residual: harness.output('residual', 'float32', 1),
        status: harness.output('status', 'uint32', 1),
        iterationCounts: harness.output('iterations', 'uint32', 1)
      }
    })
  );
  expect(result['status']).toEqual([0]);
  expect(result['iterations'][0]).toBeGreaterThan(0);
  expect(result['iterations'][0]).toBeLessThan(512);
  for (let member = 0; member < endmemberCount; member++) {
    expect(result['abundances'][member]).toBeCloseTo(expected[member], 6);
  }
  expect(result['residual'][0]).toBeLessThan(1e-10);
});

for (const constraint of ['nonnegative', 'simplex'] as const) {
  for (const scale of [2 ** -40, 1, 2 ** 40]) {
    it(`RT-140 ${constraint} identity fit is invariant at scale ${scale}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const harness = createRemoteSensingHarness(device);
      // Identity design: unique least-squares minimizer [.75, .25, 0], feasible for
      // both cones. Third coefficient is the active nonnegativity constraint; SSE = 0.
      const result = await harness.run(
        new GPURasterSpectralUnmixing({
          cellCount: 1,
          bandCount: 3,
          endmemberCount: 3,
          constraint,
          maximumIterations: 512,
          tolerance: 1e-7,
          values: harness.input([0.75 * scale, 0.25 * scale, 0]),
          endmembers: harness.input([scale, 0, 0, 0, scale, 0, 0, 0, scale]),
          output: {
            abundances: harness.output('abundances', 'float32', 3),
            residual: harness.output('residual', 'float32', 1),
            status: harness.output('status', 'uint32', 1),
            iterationCounts: harness.output('iterations', 'uint32', 1)
          }
        })
      );
      expect(result['status']).toEqual([0]);
      expect(result['abundances']).toHaveLength(3);
      for (const [index, expected] of [0.75, 0.25, 0].entries()) {
        expect(Number.isFinite(result['abundances'][index])).toBe(true);
        expect(result['abundances'][index]).toBeCloseTo(expected, 6);
        expect(result['abundances'][index]).toBeGreaterThanOrEqual(0);
      }
      expect(Number.isFinite(result['residual'][0])).toBe(true);
      expect(result['residual'][0]).toBeGreaterThanOrEqual(0);
      expect(result['residual'][0] / (scale * scale)).toBeLessThan(1e-12);
      expect(result['iterations'][0]).toBeLessThan(512);
    });
  }
}

for (const constraint of ['nonnegative', 'simplex'] as const) {
  for (const scale of [2 ** -40, 1, 2 ** 40]) {
    it(`RT-140 S2 exhausted ${constraint} keeps squared residual units at scale ${scale}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const harness = createRemoteSensingHarness(device);
      const result = await harness.run(
        new GPURasterSpectralUnmixing({
          cellCount: 1,
          bandCount: 3,
          endmemberCount: 3,
          constraint,
          maximumIterations: 1,
          tolerance: 1e-7,
          values: harness.input([0.75 * scale, 0.25 * scale, 0]),
          endmembers: harness.input([scale, 0, 0, 0, scale, 0, 0, 0, scale]),
          output: {
            abundances: harness.output('abundances', 'float32', 3),
            residual: harness.output('residual', 'float32', 1),
            status: harness.output('status', 'uint32', 1),
            iterationCounts: harness.output('iterations', 'uint32', 1)
          }
        })
      );
      expect(result['status']).toEqual([4]);
      expect(result['iterations']).toEqual([1]);
      expect(result['abundances']).toHaveLength(3);
      // Identity design has L=3*s^2. One step gives [1/4,1/12,0] for NNLS,
      // with squared residual s^2*(1/4+1/36)=s^2*5/18. Starting simplex at
      // thirds instead gives [17/36,11/36,2/9], residual s^2*7/54.
      const expected =
        constraint === 'nonnegative' ? [1 / 4, 1 / 12, 0] : [17 / 36, 11 / 36, 2 / 9];
      for (const [index, value] of expected.entries()) {
        expect(Number.isFinite(result['abundances'][index])).toBe(true);
        expect(result['abundances'][index]).toBeCloseTo(value, 6);
      }
      const expectedResidual = scale * scale * (constraint === 'nonnegative' ? 5 / 18 : 7 / 54);
      expect(result['residual']).toHaveLength(1);
      expect(Number.isFinite(result['residual'][0])).toBe(true);
      expect(result['residual'][0]).toBeGreaterThan(0);
      expect(Math.abs(result['residual'][0] - expectedResidual) / expectedResidual).toBeLessThan(
        1e-6
      );
    });
  }
}

for (const scale of [2 ** -40, 1, 2 ** 40]) {
  it(`RT-140 B2 dark-pixel simplex solves at scale ${scale}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = createRemoteSensingHarness(device);
    const result = await harness.run(
      new GPURasterSpectralUnmixing({
        cellCount: 1,
        bandCount: 2,
        endmemberCount: 2,
        constraint: 'simplex',
        tolerance: 1e-7,
        values: harness.input([0, 0]),
        endmembers: harness.input([scale, 0.125 * scale, 0.625 * scale, scale]),
        output: {
          abundances: harness.output('abundances', 'float32', 2),
          residual: harness.output('residual', 'float32', 1),
          status: harness.output('status', 'uint32', 1),
          iterationCounts: harness.output('iterations', 'uint32', 1)
        }
      })
    );
    // Minimize (.625+.375*a)^2+(1-.875*a)^2 on [0,1]: a=41/58,
    // other abundance 17/58; the squared norm at that point is 3481/3712.
    expect(result['status']).toEqual([0]);
    expect(result['iterations'][0]).toBeGreaterThan(0);
    expect(result['iterations'][0]).toBeLessThan(256);
    expect(result['abundances']).toHaveLength(2);
    for (const [index, expected] of [41 / 58, 17 / 58].entries()) {
      expect(Number.isFinite(result['abundances'][index])).toBe(true);
      expect(result['abundances'][index]).toBeCloseTo(expected, 6);
    }
    expect(Number.isFinite(result['residual'][0])).toBe(true);
    expect(result['residual'][0] / (scale * scale)).toBeCloseTo(3481 / 3712, 6);
  });
}
