// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUVariogram} from '../../../src/dataframe-analysis/pair-statistics/gpu-variogram';
import {GPUSpatialCorrelogram} from '../../../src/dataframe-analysis/pair-statistics/gpu-spatial-correlogram';
import {getGPUSpatialCorrelogramParameterValues} from '../../../src/dataframe-analysis/pair-statistics/spatial-correlogram-parameters';
import {getGPUVariogramParameterValues} from '../../../src/dataframe-analysis/pair-statistics/variogram-parameters';
import {
  GPUSpaceTimeVariogram,
  getGPUSpaceTimeVariogramParameterValues
} from '../../../src/dataframe-analysis/pair-statistics/gpu-space-time-variogram';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {makeSpatialFixture, readIntegers, readFloats} from '../../spatial-test-utils';

for (const route of ['ordinary', 'space-time-single', 'space-time-cross'] as const) {
  it(`W5-14 ${route} preserves one-pair semivariance bits with a representable reciprocal`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = makeSpatialFixture(device);
    try {
      const coordinates = fixture.view('float32', Float32Array.of(0.25, 0.25, 0.75, 0.25));
      const positions = fixture.graph.createDataView(coordinates.view.buffer, {
        format: 'float32x2',
        length: 2
      });
      const values = fixture.view('float32', Float32Array.of(-179 * 2 ** -43, 179 * 2 ** -43));
      const semivariances = fixture.view('float32', new Float32Array(route === 'ordinary' ? 1 : 4));
      const pairCounts = fixture.view(
        'uint32',
        new Uint32Array(route === 'ordinary' ? 1 : 4).fill(99)
      );
      const status = fixture.view('uint32', new Uint32Array(route === 'ordinary' ? 1 : 4).fill(99));
      if (route === 'ordinary') {
        const parameters = fixture.view(
          'float32',
          getGPUVariogramParameterValues({bounds: [0, 0, 1, 1], maximumDistance: 1})
        );
        fixture.graph.add(
          new GPUVariogram({
            positions,
            values: values.view,
            parameters: parameters.view,
            gridSize: [2, 2],
            lagCount: 1,
            output: {
              semivariances: semivariances.view,
              pairCounts: pairCounts.view,
              status: {overflow: status.view}
            }
          })
        );
      } else {
        const times = fixture.view('float32', Float32Array.of(0, 0));
        const parameters = fixture.view(
          'float32',
          getGPUSpaceTimeVariogramParameterValues({
            bounds: [0, 0, 1, 1],
            maximumDistance: 1,
            maximumTimeLag: 1
          })
        );
        const secondValues =
          route === 'space-time-cross'
            ? fixture.view('float32', Float32Array.of(-179 * 2 ** -43, 179 * 2 ** -43))
            : undefined;
        fixture.graph.add(
          new GPUSpaceTimeVariogram({
            positions,
            values: values.view,
            secondValues: secondValues?.view,
            times: times.view,
            parameters: parameters.view,
            gridSize: [2, 2],
            lagCount: 1,
            temporalLagCount: 1,
            output: {
              semivariances: semivariances.view,
              pairCounts: pairCounts.view,
              status: status.view
            }
          })
        );
      }
      await fixture.run();
      expect(await readIntegers(pairCounts.buffer)).toEqual(
        route === 'ordinary' ? [1] : [0, 0, 1, 0]
      );
      expect(await readIntegers(status.buffer)).toEqual(route === 'ordinary' ? [0] : [1, 1, 0, 1]);
      const results = await readFloats(semivariances.buffer);
      expect(results.length).toBe(route === 'ordinary' ? 1 : 4);
      const result = results[route === 'ordinary' ? 0 : 2];
      expect(Number.isFinite(result)).toBe(true);
      // Closed form gamma=(358*2^-43)^2/2=64082*2^-86. NumPy float32 gives 0x1c7a5200.
      expect(new Uint32Array(Float32Array.of(result).buffer)[0]).toBe(0x1c7a5200);
    } finally {
      fixture.destroy();
    }
  });
}

for (const route of ['ordinary', 'space-time-single', 'space-time-cross', 'correlogram'] as const) {
  it(`B1 ${route} preserves a normal result when the selected reciprocal is subnormal`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = makeSpatialFixture(device);
    try {
      const coordinates = fixture.view(
        'float32',
        Float32Array.from(Array.from({length: 8}, (_, row) => [0.1 + 0.1 * row, 0.5]).flat())
      );
      const positions = fixture.graph.createDataView(coordinates.view.buffer, {
        format: 'float32x2',
        length: 8
      });
      const values = fixture.view(
        'float32',
        Float32Array.of(0, 0, 0, 0, 0, 0, -1.5 * 2 ** -45, 1.5 * 2 ** -45)
      );
      const spaceTime = route === 'space-time-single' || route === 'space-time-cross';
      const length = spaceTime ? 4 : 1;
      const results = fixture.view('float32', new Float32Array(length));
      const pairCounts = fixture.view('uint32', new Uint32Array(length).fill(99));
      const status = fixture.view('uint32', new Uint32Array(length).fill(99));
      const common = {positions, values: values.view, gridSize: [2, 2] as const};
      if (route === 'ordinary') {
        const parameters = fixture.view(
          'float32',
          getGPUVariogramParameterValues({bounds: [0, 0, 1, 1], maximumDistance: 1})
        );
        fixture.graph.add(
          new GPUVariogram({
            ...common,
            parameters: parameters.view,
            lagCount: 1,
            output: {
              semivariances: results.view,
              pairCounts: pairCounts.view,
              status: {overflow: status.view}
            }
          })
        );
      } else if (route === 'correlogram') {
        const parameters = fixture.view(
          'float32',
          getGPUSpatialCorrelogramParameterValues({bounds: [0, 0, 1, 1], maximumDistance: 1})
        );
        fixture.graph.add(
          new GPUSpatialCorrelogram({
            ...common,
            parameters: parameters.view,
            bandCount: 1,
            output: {
              moransI: results.view,
              pairCounts: pairCounts.view,
              status: {overflow: status.view}
            }
          })
        );
      } else {
        const parameters = fixture.view(
          'float32',
          getGPUSpaceTimeVariogramParameterValues({
            bounds: [0, 0, 1, 1],
            maximumDistance: 1,
            maximumTimeLag: 1
          })
        );
        const times = fixture.view('float32', new Float32Array(8));
        const secondValues =
          route === 'space-time-cross'
            ? fixture.view(
                'float32',
                Float32Array.of(0, 0, 0, 0, 0, 0, -1.5 * 2 ** -45, 1.5 * 2 ** -45)
              )
            : undefined;
        fixture.graph.add(
          new GPUSpaceTimeVariogram({
            ...common,
            parameters: parameters.view,
            times: times.view,
            secondValues: secondValues?.view,
            lagCount: 1,
            temporalLagCount: 1,
            output: {semivariances: results.view, pairCounts: pairCounts.view, status: status.view}
          })
        );
      }
      await fixture.run();
      expect(await readIntegers(pairCounts.buffer)).toEqual(spaceTime ? [0, 0, 28, 0] : [28]);
      expect(await readIntegers(status.buffer)).toEqual(spaceTime ? [1, 1, 0, 1] : [0]);
      const output = await readFloats(results.buffer);
      expect(output.length).toBe(length);
      const value = output[spaceTime ? 2 : 0];
      expect(Number.isFinite(value)).toBe(true);
      // Complete graph: sum gamma_ij=8a^2, 28 pairs => gamma=(2/7)a^2.
      // Moran I=-1/(8-1). Literals follow from a=1.5*2^-45, not a CPU twin.
      const expected = route === 'correlogram' ? -1 / 7 : 5.192958575871677e-28;
      expect(Math.abs(value / expected - 1)).toBeLessThan(3e-6);
    } finally {
      fixture.destroy();
    }
  });
}
