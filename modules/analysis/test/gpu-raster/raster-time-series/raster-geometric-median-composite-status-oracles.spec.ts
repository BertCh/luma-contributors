// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPURasterGeometricMedianCompositeOperation} from '../../../src/raster-analysis/raster-time-series/cpu-raster-geometric-median-composite-operation';
import {GPURasterGeometricMedianCompositeOperation} from '../../../src/raster-analysis/raster-time-series/gpu-raster-geometric-median-composite-operation';
import {
  GPURasterGeometricMedianComposite,
  getGPURasterGeometricMedianCompositeParameterValues
} from '../../../src/raster-analysis/raster-time-series/gpu-raster-geometric-median-composite';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {createRasterTimeSeriesHarness} from './raster-time-series-test-utils';

// Each cell has three one-band observations. Cells 0/2 retain observations;
// cell 1 is entirely masked; cell 3 is entirely nonfinite. Status reports support,
// not convergence: the nonzero 12/11 first step of cell 0 still means status 0.
const VALUES = Float32Array.of(0, 2, 10, 0, 2, 10, NaN, 2, NaN, NaN, Infinity, -Infinity);
const MASK = Uint32Array.of(1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1);

it('derives all geometric median operation statuses from usable observations', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const Operation of [
    GPURasterGeometricMedianCompositeOperation,
    CPURasterGeometricMedianCompositeOperation
  ]) {
    const operation = new Operation({
      cellCount: 4,
      sliceCount: 3,
      bandCount: 1,
      iterationCount: 1,
      values: 'values',
      mask: 'mask',
      parameters: 'parameters',
      output: {composite: 'composite', status: 'status'}
    });
    try {
      const result = await runOperationGraph(device, {
        inputs: {
          values: {format: 'float32', values: VALUES},
          mask: {format: 'uint32', values: MASK},
          parameters: {
            format: 'float32',
            values: getGPURasterGeometricMedianCompositeParameterValues()
          }
        },
        operations: [operation],
        outputs: ['status']
      });
      expect(Array.from(result['status'])).toEqual([0, 1, 0, 1]);
    } finally {
      if (operation instanceof GPURasterGeometricMedianCompositeOperation) operation.destroy();
    }
  }
});

it('derives all geometric median contributor statuses independently of iteration convergence', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createRasterTimeSeriesHarness(device);
  const result = await harness.run(
    new GPURasterGeometricMedianComposite({
      cellCount: 4,
      sliceCount: 3,
      bandCount: 1,
      iterationCount: 1,
      values: harness.floats(VALUES),
      mask: harness.integers(MASK),
      parameters: harness.floats(getGPURasterGeometricMedianCompositeParameterValues()),
      output: {
        composite: harness.output('composite', 'float32', 4),
        status: harness.output('status', 'uint32', 4)
      }
    })
  );
  expect(result['status']).toEqual([0, 1, 0, 1]);
});
