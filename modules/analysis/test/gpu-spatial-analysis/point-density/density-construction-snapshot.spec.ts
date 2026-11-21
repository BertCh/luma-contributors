// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUKernelDensity,
  getGPUKernelDensityParameterValues
} from '../../../src/gpu-spatial-analysis/point-density/gpu-kernel-density';
import {
  GPUAdaptiveBandwidthKernelDensity,
  getGPUAdaptiveBandwidthKernelDensityParameterValues
} from '../../../src/gpu-spatial-analysis/point-density/gpu-adaptive-bandwidth-kernel-density';
import {GPUSpaceTimeKernelDensity} from '../../../src/gpu-spatial-analysis/point-density/gpu-space-time-kernel-density';
import {GPUSpaceTimeRelativeRisk} from '../../../src/gpu-spatial-analysis/point-density/gpu-space-time-relative-risk';
import {getGPUSpaceTimeKernelDensityParameterValues} from '../../../src/gpu-spatial-analysis/point-density/space-time-kernel-density-math';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createGeometryFixture} from '../outline-geometry/geometry-fixture';
import {makeExtentGrid} from '../spatial-interpolation/extent-grid';

for (const kind of ['fixed', 'adaptive', 'space-time', 'risk'] as const) {
  it(`W2-02 ${kind} retains construction routing and resubmits changed buffer contents`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const grid = makeExtentGrid(1, 1, [-0.5, -0.5, 0.5, 0.5]);
    const fixture = createGeometryFixture(device, {
      parameterLength: kind === 'fixed' ? 9 : kind === 'adaptive' ? 10 : 14,
      inputs: {
        positions: {format: 'float32x2', values: Float32Array.of(0, 0)},
        controls: {format: 'float32x2', values: Float32Array.of(0, 0)},
        times: {format: 'float32', values: Float32Array.of(0)},
        weights: {format: 'float32', values: Float32Array.of(1)}
      },
      outputs: {values: {format: 'float32', length: 1}},
      create: ({inputs, outputs, parameters}) => {
        const props = {
          positions: inputs['positions'] as GraphDataView<'float32x2'>,
          times: inputs['times'] as GraphDataView<'float32'>,
          weights: inputs['weights'] as GraphDataView<'float32'>,
          casePositions: inputs['positions'] as GraphDataView<'float32x2'>,
          controlPositions: inputs['controls'] as GraphDataView<'float32x2'>,
          caseTimes: inputs['times'] as GraphDataView<'float32'>,
          controlTimes: inputs['times'] as GraphDataView<'float32'>,
          parameters,
          grid: {...grid, affine: [...grid.affine] as typeof grid.affine},
          sliceCount: 1,
          indexBounds: [-2, -2, 2, 2] as [number, number, number, number],
          indexGridSize: [2, 2] as [number, number],
          edgeCorrection: 'none' as const,
          spatialEdgeCorrection: 'none' as const,
          temporalEdgeCorrection: 'none' as const,
          scale: 'ratio' as const,
          output: {values: outputs['values'] as GraphDataView<'float32'>}
        };
        const contributor =
          kind === 'fixed'
            ? new GPUKernelDensity(props)
            : kind === 'adaptive'
              ? new GPUAdaptiveBandwidthKernelDensity(props)
              : kind === 'space-time'
                ? new GPUSpaceTimeKernelDensity(props)
                : new GPUSpaceTimeRelativeRisk(props);
        props.grid.width = 99;
        props.indexBounds[0] = 999;
        props.indexGridSize[0] = 999;
        props.output.values = parameters;
        return contributor;
      }
    });
    try {
      const parameters =
        kind === 'fixed'
          ? getGPUKernelDensityParameterValues({bandwidth: 1}, grid)
          : kind === 'adaptive'
            ? getGPUAdaptiveBandwidthKernelDensityParameterValues(
                {bandwidth: 1, sensitivity: 0},
                grid
              )
            : getGPUSpaceTimeKernelDensityParameterValues(
                {bandwidth: 1, temporalBandwidth: 1, firstSliceTime: 0, sliceInterval: 1},
                grid,
                1
              );
      const firstValues = (await fixture.run(parameters))['values'];
      expect(firstValues).toHaveLength(1);
      const first = firstValues[0];
      // Closed forms at the Gaussian center: 1/(2*pi), (2*pi)^(-3/2); identical risk = 1.
      const expected =
        kind === 'risk' ? 1 : kind === 'space-time' ? 0.06349363593424097 : 0.15915494309189535;
      expect(Number.isFinite(first)).toBe(true);
      expect(first).toBeCloseTo(expected, 5);
      if (kind === 'risk') fixture.writeInput('positions', Float32Array.of(1, 0));
      else fixture.writeInput('weights', Float32Array.of(2));
      const secondValues = (await fixture.run())['values'];
      expect(secondValues).toHaveLength(1);
      const second = secondValues[0];
      expect(Number.isFinite(second)).toBe(true);
      // Moving the sole case one spatial bandwidth from the center gives exp(-1/2).
      expect(second).toBeCloseTo(kind === 'risk' ? 0.6065306597126334 : 2 * expected, 5);
      expect(fixture.getCompileCount()).toBe(0);
    } finally {
      fixture.destroy();
    }
  });
}
