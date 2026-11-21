// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  GPUContinuousSpatialSampleOperation,
  type ContinuousSpatialSampleOperationProps
} from '../../../src/gpu-spatial-analysis/point-process/gpu-continuous-spatial-sample-operation';
import {CPUContinuousSpatialSampleOperation} from '../../../src/gpu-spatial-analysis/point-process/cpu-continuous-spatial-sample-operation';
import {
  CONTINUOUS_PROPS,
  CONTINUOUS_INPUTS,
  expectContinuousOracle
} from './continuous-sample-output-oracles-fixture';
import {constructSamplingContributor} from './validation-sampling-operation-test-utils';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';

for (const Operation of [
  GPUContinuousSpatialSampleOperation,
  CPUContinuousSpatialSampleOperation
]) {
  for (const changingType of [false, true]) {
    it(`W5-12 ${Operation.name} captures line once, changing=${changingType}`, async () => {
      let reads = 0;
      const frame: ContinuousSpatialSampleOperationProps['frame'] = {
        type: 'line',
        positions: 'positions'
      };
      Object.defineProperty(frame, 'type', {
        get: () => (++reads === 1 || !changingType ? 'line' : 'area')
      });
      const guardedFrame = new Proxy(frame, {
        get(target, key, receiver) {
          if (key === 'bounds') throw new Error('unused bounds read');
          return Reflect.get(target, key, receiver);
        }
      });
      const operation = new Operation({...CONTINUOUS_PROPS, frame: guardedFrame});
      expect(reads).toBe(1);
      if (operation instanceof GPUContinuousSpatialSampleOperation) {
        expect(() =>
          constructSamplingContributor(operation, {positions: {format: 'float32x2', length: 2}})
        ).not.toThrow();
      } else {
        const device = createNullWebGPUDevice();
        try {
          expectContinuousOracle(
            await runOperationGraph(device, {
              inputs: CONTINUOUS_INPUTS,
              operations: [operation],
              outputs: Object.values(CONTINUOUS_PROPS.output)
            })
          );
        } finally {
          device.destroy();
        }
      }
      expect(reads).toBe(1);
    });
  }
  it(`W5-12 ${Operation.name} reads and copies area bounds once`, async () => {
    let reads = 0;
    const bounds: [number, number, number, number] = [0, 0, 1, 1];
    const frame: ContinuousSpatialSampleOperationProps['frame'] = {
      type: 'area',
      positions: 'positions',
      get bounds() {
        reads++;
        return bounds;
      }
    };
    const operation = new Operation({...CONTINUOUS_PROPS, frameRowCount: 3, frame});
    expect(reads).toBe(1);
    bounds[2] = -1;
    if (operation instanceof GPUContinuousSpatialSampleOperation) {
      expect(() =>
        constructSamplingContributor(operation, {positions: {format: 'float32x2', length: 3}})
      ).not.toThrow();
    } else {
      const device = createNullWebGPUDevice();
      try {
        const result = await runOperationGraph(device, {
          inputs: {positions: {format: 'float32x2', values: Float32Array.of(0, 0, 1, 0, 0, 1)}},
          operations: [operation],
          outputs: Object.values(CONTINUOUS_PROPS.output)
        });
        // Independent Random123 integer oracle (scratch/area.py): only candidate (0,0)
        // lies in the unit right triangle, at these two Philox phases. Area weight is .5^2.
        expect(Array.from(result['status'])).toEqual([1, 1, 1, 0]);
        expect(result['selected'].length).toBe(10);
        expect(Array.from(result['selected'])).toEqual([
          0.3629690706729889, 0.3902064263820648, 0, 0, 0, 0, 0, 0, 0, 0
        ]);
        expect(Array.from(result['weights'])).toEqual([0.25, 0, 0, 0, 0]);
      } finally {
        device.destroy();
      }
    }
    expect(reads).toBe(1);
  });
}
