// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {
  GPUAdaptiveBandwidthKernelDensity,
  getGPUAdaptiveBandwidthKernelDensityParameterValues
} from '../../../src/gpu-spatial-analysis/point-density/gpu-adaptive-bandwidth-kernel-density';
import {GPUSpaceTimeKernelDensity} from '../../../src/gpu-spatial-analysis/point-density/gpu-space-time-kernel-density';
import {getGPUSpaceTimeKernelDensityParameterValues} from '../../../src/gpu-spatial-analysis/point-density/space-time-kernel-density-math';
import {createGeometryFixture} from '../outline-geometry/geometry-fixture';
import {makeExtentGrid} from '../spatial-interpolation/extent-grid';
import {
  ADAPTIVE_INDEPENDENT_FIXTURE,
  SPACE_TIME_INDEPENDENT_VALUES
} from './density-independent-fixtures';

it('PTS-140 pins adaptive pilot densities, distinct bandwidths and final cell densities', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const grid = makeExtentGrid(3, 1, [-1, -0.5, 5, 0.5]);
  const fixture = createGeometryFixture(device, {
    parameterLength: 10,
    inputs: {
      positions: {
        format: 'float32x2',
        values: new Float32Array(ADAPTIVE_INDEPENDENT_FIXTURE.positions)
      }
    },
    outputs: {
      values: {format: 'float32', length: 3},
      pilot: {format: 'float32', length: 4},
      bandwidths: {format: 'float32', length: 4}
    },
    create: ({inputs, outputs, parameters}) =>
      new GPUAdaptiveBandwidthKernelDensity({
        positions: inputs['positions'] as never,
        parameters,
        grid,
        indexBounds: [-1, -1, 5, 1],
        indexGridSize: [6, 2],
        output: {
          values: outputs['values'] as never,
          pilotDensities: outputs['pilot'] as never,
          pointBandwidths: outputs['bandwidths'] as never
        }
      })
  });
  try {
    const result = await fixture.run(
      getGPUAdaptiveBandwidthKernelDensityParameterValues({bandwidth: 1, sensitivity: 0.5}, grid)
    );
    for (const name of ['values', 'pilot', 'bandwidths'] as const) {
      expect(result[name]).toHaveLength(ADAPTIVE_INDEPENDENT_FIXTURE[name].length);
      result[name].forEach((value, index) => {
        expect(Number.isFinite(value), `${name} ${index}`).toBe(true);
        expect(value, `${name} ${index}`).toBeCloseTo(ADAPTIVE_INDEPENDENT_FIXTURE[name][index], 5);
      });
    }
  } finally {
    fixture.destroy();
  }
});

it('PTS-140 pins space-time product-kernel interior and temporal boundary densities', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const grid = makeExtentGrid(1, 1, [0, -1, 2, 1]);
  const fixture = createGeometryFixture(device, {
    parameterLength: 14,
    inputs: {
      positions: {format: 'float32x2', values: new Float32Array([0, 0, 0, 0])},
      times: {format: 'float32', values: new Float32Array([0.5, 1.5])}
    },
    outputs: {values: {format: 'float32', length: 2}},
    create: ({inputs, outputs, parameters}) =>
      new GPUSpaceTimeKernelDensity({
        positions: inputs['positions'] as never,
        times: inputs['times'] as never,
        parameters,
        grid,
        indexBounds: [0, -1, 2, 1],
        indexGridSize: [2, 2],
        sliceCount: 2,
        spatialKernel: 'epanechnikov',
        temporalKernel: 'epanechnikov',
        spatialEdgeCorrection: 'none',
        temporalEdgeCorrection: 'uniform',
        output: {values: outputs['values'] as never}
      })
  });
  try {
    const result = await fixture.run(
      getGPUSpaceTimeKernelDensityParameterValues(
        {
          bandwidth: 2,
          temporalBandwidth: 2,
          firstSliceTime: 0,
          sliceInterval: 2,
          timeWindow: [0, 4]
        },
        grid,
        2
      )
    );
    expect(result['values']).toHaveLength(SPACE_TIME_INDEPENDENT_VALUES.length);
    result['values'].forEach((value, index) => {
      expect(Number.isFinite(value)).toBe(true);
      expect(value).toBeCloseTo(SPACE_TIME_INDEPENDENT_VALUES[index], 6);
    });
  } finally {
    fixture.destroy();
  }
});
