// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUSpaceTimeRelativeRisk} from '../../../src/gpu-spatial-analysis/point-density/gpu-space-time-relative-risk';
import {getGPUSpaceTimeKernelDensityParameterValues} from '../../../src/gpu-spatial-analysis/point-density/space-time-kernel-density-math';
import {requireTestDevice} from '../../utils/require-test-device';
import {makeExtentGrid} from '../spatial-interpolation/extent-grid';
import {createGeometryFixture} from '../outline-geometry/geometry-fixture';

it('relative risk counts case and control exclusions separately and clears both on replay', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const grid = makeExtentGrid(1, 1, [0, 0, 1, 1]);
  const parameters = getGPUSpaceTimeKernelDensityParameterValues(
    {bandwidth: 1, temporalBandwidth: 1, firstSliceTime: 0, sliceInterval: 1},
    grid,
    1
  );
  const fixture = createGeometryFixture(device, {
    parameterLength: parameters.length,
    inputs: {
      positions: {format: 'float32x2', values: Float32Array.of(0.5, 0.5, 0.5, 0.5, 0.5, 0.5)},
      times: {format: 'float32', values: Float32Array.of(0, 0, 0)},
      cases: {format: 'float32', values: Float32Array.of(1, NaN, 1)},
      controls: {format: 'float32', values: Float32Array.of(1, -1, Infinity)}
    },
    outputs: {
      values: {format: 'float32', length: 1},
      caseInvalid: {format: 'uint32', length: 1},
      controlInvalid: {format: 'uint32', length: 1}
    },
    create: ({inputs, outputs, parameters: parameterView}) =>
      new GPUSpaceTimeRelativeRisk({
        casePositions: inputs['positions'] as never,
        controlPositions: inputs['positions'] as never,
        caseTimes: inputs['times'] as never,
        controlTimes: inputs['times'] as never,
        caseWeights: inputs['cases'] as never,
        controlWeights: inputs['controls'] as never,
        parameters: parameterView,
        grid,
        sliceCount: 1,
        indexGridSize: [1, 1],
        indexBounds: [0, 0, 1, 1],
        spatialEdgeCorrection: 'none',
        temporalEdgeCorrection: 'none',
        output: {
          values: outputs['values'] as never,
          status: {
            caseInvalidCount: outputs['caseInvalid'] as never,
            controlInvalidCount: outputs['controlInvalid'] as never
          }
        }
      })
  });
  try {
    // NaN case weight excludes one row; negative/infinite control weights exclude two.
    const dirty = await fixture.run(parameters);
    expect(dirty['caseInvalid']).toEqual([1]);
    expect(dirty['controlInvalid']).toEqual([2]);
    fixture.writeInput('cases', Float32Array.of(1, 1, 1));
    fixture.writeInput('controls', Float32Array.of(1, 1, 1));
    const clean = await fixture.run(parameters);
    expect(clean['caseInvalid']).toEqual([0]);
    expect(clean['controlInvalid']).toEqual([0]);
  } finally {
    fixture.destroy();
  }
});
