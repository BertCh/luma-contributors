// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUSpaceTimeHawkesOperation,
  type SpaceTimeHawkesOperationProps
} from '../../../src/gpu-spatial-analysis/space-time-hawkes/gpu-space-time-hawkes-operation';
import {CPUSpaceTimeHawkesOperation} from '../../../src/gpu-spatial-analysis/space-time-hawkes/cpu-space-time-hawkes-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import type {OperationGraphRows} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import fixture from './hawkes-operation-fixture.json';

/** All integer/status outputs are exact; each finite float keeps the existing relative 5e-4 bound. */
function expectHawkesOutputs(
  gpu: Record<string, OperationGraphRows>,
  cpu: Record<string, OperationGraphRows>
): void {
  expect(Object.keys(gpu).sort()).toEqual(Object.keys(cpu).sort());
  for (const port of Object.keys(cpu)) {
    expect(gpu[port].length).toBe(cpu[port].length);
    if (cpu[port] instanceof Uint32Array)
      expect(Array.from(gpu[port])).toEqual(Array.from(cpu[port]));
    else
      for (let index = 0; index < cpu[port].length; index++) {
        const expected = cpu[port][index];
        if (Number.isNaN(expected)) expect(Number.isNaN(gpu[port][index])).toBe(true);
        else
          expect(Math.abs(gpu[port][index] - expected)).toBeLessThanOrEqual(
            5e-4 * Math.abs(expected)
          );
      }
  }
}

for (const edge of [false, true]) {
  it(`compares every Hawkes output on ${edge ? 'an entirely excluded catalogue' : 'three triggered clusters in unsorted row order'}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const props: SpaceTimeHawkesOperationProps = {
      id: 'hawkes-graph',
      rowCount: 8,
      maximumIterations: 100,
      maximumParents: 8,
      positions: 'positions',
      times: 'times',
      parameters: 'parameters',
      output: {
        estimates: 'estimates',
        logLikelihood: 'logLikelihood',
        backgroundProbability: 'backgroundProbability',
        status: {
          converged: 'converged',
          iterationCount: 'iterationCount',
          iterationLimitReached: 'iterationLimitReached',
          candidateOverflow: 'candidateOverflow',
          invalidCount: 'invalidCount',
          approximation: 'approximation'
        }
      }
    };
    const order = [5, 2, 7, 0, 4, 1, 6, 3];
    const parameters = Float32Array.from(fixture.parameters);
    // Stop well inside the GPU's budget, with a tolerance away from a rounding-sensitive limit.
    parameters[4] = 0.02;
    const inputs = {
      positions: {
        format: 'float32x2' as const,
        values: Float32Array.from(
          order.flatMap(row => fixture.positions.slice(row * 2, row * 2 + 2))
        )
      },
      times: {
        format: 'float32' as const,
        values: Float32Array.from(order, row => (edge ? -1 : fixture.times[row]))
      },
      parameters: {format: 'float32' as const, values: parameters}
    };
    const outputs = [
      'estimates',
      'logLikelihood',
      'backgroundProbability',
      ...Object.values(props.output.status!)
    ];
    const gpu = await runOperationGraph(device, {
      inputs,
      operations: [new GPUSpaceTimeHawkesOperation(props)],
      outputs
    });
    const cpu = await runOperationGraph(device, {
      inputs,
      operations: [new CPUSpaceTimeHawkesOperation(props)],
      outputs
    });
    expectHawkesOutputs(gpu, cpu);
    expect(cpu['candidateOverflow'][0]).toBe(0);
    expect(cpu['iterationLimitReached'][0]).toBe(0);
    expect(cpu['converged'][0]).toBe(edge ? 0 : 1);
    if (!edge) expect(cpu['backgroundProbability'][1]).toBeLessThan(0.01);
  });
}

it('compares the reported cycling input with a one-iteration budget and latest-fit publication', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const props: SpaceTimeHawkesOperationProps = {
    id: 'hawkes-cycle-budget',
    rowCount: 4,
    maximumIterations: 1,
    maximumParents: 4,
    positions: 'positions',
    times: 'times',
    parameters: 'parameters',
    output: {
      estimates: 'estimates',
      logLikelihood: 'logLikelihood',
      backgroundProbability: 'backgroundProbability',
      status: {
        converged: 'converged',
        iterationLimitReached: 'iterationLimitReached',
        iterationCount: 'iterationCount',
        candidateOverflow: 'candidateOverflow',
        invalidCount: 'invalidCount',
        approximation: 'approximation'
      }
    }
  };
  const inputs = {
    positions: {
      format: 'float32x2' as const,
      values: Float32Array.from(fixture.cyclingInput.positions)
    },
    times: {format: 'float32' as const, values: Float32Array.from(fixture.cyclingInput.times)},
    parameters: {
      format: 'float32' as const,
      values: Float32Array.from(fixture.cyclingInput.parameters)
    }
  };
  const outputs = [
    'estimates',
    'logLikelihood',
    'backgroundProbability',
    ...Object.values(props.output.status!)
  ];
  const gpu = await runOperationGraph(device, {
    inputs,
    operations: [new GPUSpaceTimeHawkesOperation(props)],
    outputs
  });
  const cpu = await runOperationGraph(device, {
    inputs,
    operations: [new CPUSpaceTimeHawkesOperation(props)],
    outputs
  });
  expectHawkesOutputs(gpu, cpu);
  for (const rows of [gpu, cpu]) {
    expect(rows['iterationCount'][0]).toBe(1);
    expect(rows['converged'][0]).toBe(0);
    expect(rows['iterationLimitReached'][0]).toBe(1);
    expect(rows['estimates'].every(Number.isFinite)).toBe(true);
    // Direct six-pair intensity enumeration at the starting parameters, including both
    // compensators. This is the pre-update likelihood, not the density at the new estimates.
    expect(Math.abs(rows['logLikelihood'][0] - -15.259916788653317)).toBeLessThanOrEqual(
      5e-4 * 15.259916788653317
    );
  }
});
