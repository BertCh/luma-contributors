// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPURasterGeometricMedianCompositeOperation} from '../../../src/raster-analysis/raster-time-series/cpu-raster-geometric-median-composite-operation';
import {
  GPURasterGeometricMedianCompositeOperation,
  type RasterGeometricMedianCompositeOperationProps
} from '../../../src/raster-analysis/raster-time-series/gpu-raster-geometric-median-composite-operation';
import {getGPURasterGeometricMedianCompositeParameterValues} from '../../../src/raster-analysis/raster-time-series/gpu-raster-geometric-median-composite';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';

for (const missing of [false, true]) {
  for (const distanceEpsilon of [1e-5, 10]) {
    it(`compares every geometric median output with missing=${missing}, epsilon=${distanceEpsilon}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const props: RasterGeometricMedianCompositeOperationProps = {
        cellCount: 1,
        sliceCount: 3,
        bandCount: 2,
        iterationCount: 32,
        values: 'values',
        mask: 'mask',
        parameters: 'parameters',
        output: {composite: 'composite', status: 'status', lastStep: 'lastStep'}
      };
      const outputs = ['composite', 'status', 'lastStep'];
      const results = [];
      for (const Operation of [
        GPURasterGeometricMedianCompositeOperation,
        CPURasterGeometricMedianCompositeOperation
      ])
        results.push(
          await runOperationGraph(device, {
            inputs: {
              values: {format: 'float32', values: new Float32Array([0, 0, 4, 0, 2, 3])},
              mask: {format: 'uint32', values: new Uint32Array(3).fill(missing ? 0 : 5)},
              parameters: {
                format: 'float32',
                values: getGPURasterGeometricMedianCompositeParameterValues({distanceEpsilon})
              }
            },
            operations: [new Operation(props)],
            outputs
          })
        );
      for (const output of outputs) {
        expect(results[0][output].length).toBe(results[1][output].length);
        for (let index = 0; index < results[1][output].length; index++) {
          const expected = results[1][output][index];
          // Absolute tolerance 2e-5 for composite/step; statuses and NaNs exact.
          if (output === 'status' || Number.isNaN(expected))
            expect(results[0][output][index]).toBe(expected);
          else expect(Math.abs(results[0][output][index] - expected)).toBeLessThanOrEqual(2e-5);
        }
      }
    });
  }
}

it('compares a nonzero lastStep after one Weiszfeld iteration', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const props: RasterGeometricMedianCompositeOperationProps = {
    cellCount: 1,
    sliceCount: 3,
    bandCount: 1,
    iterationCount: 1,
    values: 'values',
    parameters: 'parameters',
    output: {composite: 'composite', status: 'status', lastStep: 'lastStep'}
  };
  const outputs = ['composite', 'status', 'lastStep'];
  const results = [];
  for (const Operation of [
    GPURasterGeometricMedianCompositeOperation,
    CPURasterGeometricMedianCompositeOperation
  ]) {
    results.push(
      await runOperationGraph(device, {
        inputs: {
          values: {format: 'float32', values: new Float32Array([0, 2, 10])},
          parameters: {
            format: 'float32',
            values: getGPURasterGeometricMedianCompositeParameterValues()
          }
        },
        operations: [new Operation(props)],
        outputs
      })
    );
  }
  // Mean 4 gives weights 1/4, 1/2, 1/6: next=32/11 and step=12/11.
  // Absolute tolerance 2e-5 for each float; status exact. A constant zero step cannot pass.
  for (const result of results) {
    expect(Math.abs(result['composite'][0] - 32 / 11)).toBeLessThanOrEqual(2e-5);
    expect(Math.abs(result['lastStep'][0] - 12 / 11)).toBeLessThanOrEqual(2e-5);
    expect(result['lastStep'][0]).toBeGreaterThan(1);
    expect(Array.from(result['status'])).toEqual([0]);
  }
  for (const output of outputs) {
    expect(results[0][output].length).toBe(results[1][output].length);
    if (output === 'status') expect(results[0][output]).toEqual(results[1][output]);
    else expect(Math.abs(results[0][output][0] - results[1][output][0])).toBeLessThanOrEqual(2e-5);
  }
});

for (const lastStep of [false, true]) {
  it(`publishes the geometric median layout with two cells, three bands and lastStep=${lastStep}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const props: RasterGeometricMedianCompositeOperationProps = {
      cellCount: 2,
      sliceCount: 1,
      bandCount: 3,
      iterationCount: 1,
      values: 'values',
      parameters: 'parameters',
      mask: 'mask',
      output: {
        composite: 'composite',
        status: 'status',
        ...(lastStep ? {lastStep: 'lastStep'} : {})
      }
    };
    const expected = {
      composite: [10, 11, 12, NaN, NaN, NaN],
      status: [0, 1],
      ...(lastStep ? {lastStep: [0, NaN]} : {})
    };
    for (const Operation of [
      GPURasterGeometricMedianCompositeOperation,
      CPURasterGeometricMedianCompositeOperation
    ]) {
      const result = await runOperationGraph(device, {
        inputs: {
          values: {format: 'float32', values: Float32Array.of(10, 11, 12, 20, 21, 22)},
          mask: {format: 'uint32', values: Uint32Array.of(1, 0)},
          parameters: {
            format: 'float32',
            values: getGPURasterGeometricMedianCompositeParameterValues()
          }
        },
        operations: [new Operation(props)],
        outputs: Object.keys(expected)
      });
      expect(Object.keys(result).sort()).toEqual(Object.keys(expected).sort());
      for (const [name, values] of Object.entries(expected)) {
        expect(result[name]).toHaveLength(values.length);
        for (const [index, value] of values.entries()) {
          if (Number.isNaN(value)) expect(result[name][index]).toBeNaN();
          else {
            expect(Number.isFinite(result[name][index])).toBe(true);
            expect(result[name][index]).toBeCloseTo(value, 5);
          }
        }
      }
    }
  });
}

for (const sliceCount of [1, 3]) {
  it(`TWN-41 preserves ${sliceCount} identical nonzero observations with exact zero step`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    for (const Operation of [
      GPURasterGeometricMedianCompositeOperation,
      CPURasterGeometricMedianCompositeOperation
    ]) {
      const result = await runOperationGraph(device, {
        inputs: {
          values: {format: 'float32', values: new Float32Array(sliceCount).fill(456787.5)},
          parameters: {
            format: 'float32',
            values: getGPURasterGeometricMedianCompositeParameterValues()
          }
        },
        operations: [
          new Operation({
            cellCount: 1,
            sliceCount,
            bandCount: 1,
            iterationCount: 1,
            values: 'values',
            parameters: 'parameters',
            output: {composite: 'composite', status: 'status', lastStep: 'lastStep'}
          })
        ],
        outputs: ['composite', 'status', 'lastStep']
      });
      expect(Array.from(result['composite'])).toEqual([456787.5]);
      expect(Array.from(result['status'])).toEqual([0]);
      expect(Array.from(result['lastStep'])).toEqual([0]);
    }
  });
}
