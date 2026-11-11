// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPULocationChangePointsOperation} from '../../../src/gpu-spatial-analysis/change-points/cpu-location-change-points-operation';
import {
  GPULocationChangePointsOperation,
  type LocationChangePointsOperationProps
} from '../../../src/gpu-spatial-analysis/change-points/gpu-location-change-points-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

const SCENES = {
  mean: [0, 1, 0, 1, 9, 10, 9, 10],
  variance: [0, 0.5, -0.5, 0, 5, -5, 6, -6],
  'mean-variance': [0, 1, 0, 1, 8, 10, 8, 10],
  poisson: [1, 2, 1, 2, 12, 14, 12, 14],
  slope: [0, 1, 2, 3, 8, 6, 4, 2]
};

for (const cost of ['mean', 'variance', 'mean-variance', 'poisson', 'slope'] as const) {
  for (const layout of ['cell-major', 'period-major'] as const) {
    it(`change operations agree for ${cost}, ${layout}, including missing and overflowing cells`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const ordinary = SCENES[cost];
      const series = [
        ordinary,
        ordinary.map((value, slice) => (slice === 1 ? NaN : value * 2)),
        new Array<number>(8).fill(NaN)
      ];
      const values = Float32Array.from(
        layout === 'cell-major'
          ? series.flat()
          : Array.from({length: 8}, (_, slice) => series.map(row => row[slice])).flat()
      );
      const parameters = new Float32Array([cost === 'variance' ? 5 : 2, cost === 'slope' ? 3 : 2]);
      const props: LocationChangePointsOperationProps = {
        values: 'values',
        parameters: 'parameters',
        cellCount: 3,
        sliceCount: 8,
        layout,
        cost,
        maximumChangePoints: 1,
        output: {
          changePointCount: 'counts',
          changePoints: 'changes',
          requiredChangePointCounts: 'required',
          optimalCost: 'optimum',
          status: {overflow: 'overflow'}
        }
      };
      const inputs = {
        values: {format: 'float32' as const, values},
        parameters: {format: 'float32' as const, values: parameters}
      };
      const outputs = ['counts', 'changes', 'required', 'optimum', 'overflow'];
      const cpu = await runOperationGraph(device, {
        inputs,
        outputs,
        operations: [new CPULocationChangePointsOperation(props)]
      });
      const gpu = await runOperationGraph(device, {
        inputs,
        outputs,
        operations: [new GPULocationChangePointsOperation(props)]
      });
      expect(cpu['counts'][0]).toBeGreaterThan(0);
      for (const output of outputs) {
        expect(cpu[output].length).toBe(gpu[output].length);
        for (let index = 0; index < cpu[output].length; index++) {
          const expected = cpu[output][index];
          if (Number.isNaN(expected)) expect(gpu[output][index]).toBeNaN();
          else if (output !== 'optimum' || expected === 0)
            expect(gpu[output][index] === expected).toBe(true);
          else
            expect(Math.abs(gpu[output][index] - expected)).toBeLessThan(3e-4 * Math.abs(expected));
        }
      }
    });
  }
}

it('change operations publish earliest changes and an unclamped requirement on overflow', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const props: LocationChangePointsOperationProps = {
    values: 'values',
    parameters: 'parameters',
    cellCount: 1,
    sliceCount: 8,
    layout: 'cell-major',
    cost: 'mean',
    maximumChangePoints: 2,
    output: {
      changePointCount: 'counts',
      changePoints: 'changes',
      requiredChangePointCounts: 'required',
      optimalCost: 'optimum',
      status: {overflow: 'overflow'}
    }
  };
  for (const Operation of [GPULocationChangePointsOperation, CPULocationChangePointsOperation]) {
    const rows = await runOperationGraph(device, {
      inputs: {
        values: {format: 'float32', values: new Float32Array([0, 8, 0, 8, 0, 8, 0, 8])},
        parameters: {format: 'float32', values: new Float32Array([0.25, 1])}
      },
      operations: [new Operation(props)],
      outputs: ['counts', 'changes', 'required', 'optimum', 'overflow']
    });
    expect(Array.from(rows['changes'])).toEqual([1, 2]);
    expect(rows['counts'][0]).toBe(2);
    expect(rows['required'][0]).toBe(7);
    expect(rows['overflow'][0]).toBe(1);
    expect(rows['optimum'][0]).toBeCloseTo(1.75, 5);
  }
});

for (const cost of ['poisson', 'mean'] as const) {
  it(`change operations exercise the ${cost} numerical counterexample`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const props: LocationChangePointsOperationProps = {
      values: 'values',
      parameters: 'parameters',
      cellCount: 1,
      sliceCount: 2,
      layout: 'cell-major',
      cost,
      maximumChangePoints: 1,
      output: {
        changePointCount: 'counts',
        changePoints: 'changes',
        requiredChangePointCounts: 'required',
        optimalCost: 'optimum',
        status: {overflow: 'overflow'}
      }
    };
    const inputs = {
      values: {
        format: 'float32' as const,
        values:
          cost === 'poisson' ? new Float32Array([1073741824, 1073742080]) : new Float32Array([0, 1])
      },
      parameters: {
        format: 'float32' as const,
        values: new Float32Array([cost === 'poisson' ? 2 ** -16 : 2 ** 60, 1])
      }
    };
    const outputs = ['counts', 'changes', 'required', 'optimum', 'overflow'];
    const cpu = await runOperationGraph(device, {
      inputs,
      outputs,
      operations: [new CPULocationChangePointsOperation(props)]
    });
    const gpu = await runOperationGraph(device, {
      inputs,
      outputs,
      operations: [new GPULocationChangePointsOperation(props)]
    });
    const changes = cost === 'poisson' ? 1 : 0;
    for (const rows of [cpu, gpu]) {
      expect(Array.from(rows['counts'])).toEqual([changes]);
      expect(Array.from(rows['required'])).toEqual([changes]);
      expect(Array.from(rows['changes'])).toEqual([changes ? 1 : 0xffffffff]);
      expect(Array.from(rows['overflow'])).toEqual([0]);
    }
    if (cost === 'poisson') {
      // The split's deviance improvement is 3.05175745e-5, above penalty 2^-16.
      // Its tiny penalized improvement cannot change the f32 rounding of the common likelihood.
      expect(cpu['optimum'][0]).toBe(-89311346688);
      expect(Math.abs(gpu['optimum'][0] - cpu['optimum'][0])).toBeLessThan(
        3e-6 * Math.abs(cpu['optimum'][0])
      );
    } else {
      // Exact unsplit RSS is 1/2. Step-rounded GPU cancellation predicts 0; reassociation can
      // retain 1/2. This probe records the device result without requiring that compiler choice.
      expect(cpu['optimum'][0]).toBe(0.5);
      console.info(
        'large-penalty GPU optimalCost (0 = cancellation, 0.5 = retained residual):',
        gpu['optimum'][0]
      );
      expect(gpu['optimum'][0] === 0 || gpu['optimum'][0] === 0.5).toBe(true);
    }
  });
}
