// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPURasterRegistration} from '../../../src/raster-analysis/remote-sensing/index';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid/index';
import {requireTestDevice} from '../../utils/require-test-device';
import {createRemoteSensingHarness} from './remote-sensing-test-utils';
import oracle from '../../oracles/fixtures/raster-spectral-filter.json';

for (const fixture of oracle.registration) {
  it(`NumPy phase correlation peak ratio and existing refinement: ${fixture.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = createRemoteSensingHarness(device);
    const result = await harness.run(
      new GPURasterRegistration({
        grid: makeGPURasterMetadata({width: fixture.width, height: fixture.height}),
        reference: harness.input(fixture.reference),
        moving: harness.input(fixture.moving),
        upsampleFactor: 20,
        output: {
          shift: harness.output('shift', 'float32x2', 1),
          peak: harness.output('peak', 'float32', 1),
          status: harness.output('status', 'uint32', 1),
          secondPeakRatio: harness.output('ratio', 'float32', 2)
        }
      })
    );
    expect(result['status']).toEqual([fixture.status]);
    expect(result['ratio'][1]).toBe(0);
    if (fixture.secondPeakRatio === null) expect(result['ratio'][0]).toBeNaN();
    else expect(result['ratio'][0]).toBeCloseTo(fixture.secondPeakRatio, 4);
    if (fixture.status === 0) {
      expect(Math.abs(result['shift'][0] - fixture.shift[0])).toBeLessThanOrEqual(0.051);
      expect(Math.abs(result['shift'][1] - fixture.shift[1])).toBeLessThanOrEqual(0.051);
    } else {
      for (const value of result['shift']) expect(value).toBeNaN();
      expect(result['peak'][0]).toBeNaN();
    }
  });
}

for (const fixture of oracle.masked) {
  it(`independent centered NCC second-peak ratio: ${fixture.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = createRemoteSensingHarness(device);
    const result = await harness.run(
      new GPURasterRegistration({
        mode: 'masked-linear',
        grid: makeGPURasterMetadata({width: fixture.width, height: fixture.height}),
        reference: harness.input(fixture.reference),
        moving: harness.input(fixture.moving),
        maximumShift: fixture.maximumShift as [number, number],
        minimumOverlapFraction: 0,
        upsampleFactor: 1,
        output: {
          shift: harness.output('shift', 'float32x2', 1),
          peak: harness.output('peak', 'float32', 1),
          status: harness.output('status', 'uint32', 1),
          secondPeakRatio: harness.output('ratio', 'float32', 2)
        }
      })
    );
    expect(result['status']).toEqual([fixture.status]);
    expect(result['ratio'][1]).toBe(0);
    expect(result['ratio'][0]).toBeCloseTo(fixture.secondPeakRatio, 4);
    if (fixture.status === 0) expect(result['shift']).toEqual(fixture.shift);
    else for (const value of result['shift']) expect(value).toBeNaN();
  });
}

it('reports an undefined ratio for masked flat support and nonfinite periodic input', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const mode of ['periodic', 'masked-linear'] as const) {
    const harness = createRemoteSensingHarness(device);
    const reference =
      mode === 'periodic' ? [1, 2, 3, NaN, ...new Array(12).fill(1)] : new Array(16).fill(1);
    const result = await harness.run(
      new GPURasterRegistration({
        mode,
        grid: makeGPURasterMetadata({width: 4, height: 4}),
        reference: harness.input(reference),
        moving: harness.input(new Array(16).fill(1)),
        upsampleFactor: 1,
        output: {
          shift: harness.output('shift', 'float32x2', 1),
          peak: harness.output('peak', 'float32', 1),
          status: harness.output('status', 'uint32', 1),
          secondPeakRatio: harness.output('ratio', 'float32', 1)
        }
      })
    );
    expect(result['status']).toEqual([mode === 'periodic' ? 1 : 3]);
    expect(result['ratio'][0]).toBeNaN();
  }
});
