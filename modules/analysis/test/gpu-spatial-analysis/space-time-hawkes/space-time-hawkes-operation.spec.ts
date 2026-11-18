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
        fittedLogLikelihood: 'fittedLogLikelihood',
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
      'fittedLogLikelihood',
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
      fittedLogLikelihood: 'fittedLogLikelihood',
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
    'fittedLogLikelihood',
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

for (const withProbabilities of [false, true]) {
  it(`publishes the one-update fitted likelihood ${withProbabilities ? 'with' : 'without'} final probabilities`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const props: SpaceTimeHawkesOperationProps = {
      id: 'hawkes-fitted-likelihood',
      rowCount: 2,
      maximumIterations: 1,
      positions: 'positions',
      times: 'times',
      parameters: 'parameters',
      output: {
        estimates: 'estimates',
        fittedLogLikelihood: 'fittedLogLikelihood',
        ...(withProbabilities ? {backgroundProbability: 'backgroundProbability'} : {})
      }
    };
    const result = await runOperationGraph(device, {
      inputs: {
        positions: {format: 'float32x2', values: Float32Array.of(0, 0, 1, 1)},
        times: {format: 'float32', values: Float32Array.of(2, 2)},
        parameters: {format: 'float32', values: Float32Array.of(1, 10, 10, 10, 1e-8, 1, 0.5, 1, 1)}
      },
      operations: [new GPUSpaceTimeHawkesOperation(props)],
      outputs: Object.values(props.output) as string[]
    });
    expect(result['estimates'][0]).toBeCloseTo(0.2, 6);
    // No strictly earlier parents: the fitted Poisson intensity is 0.2, not the starting 1.
    expect(result['fittedLogLikelihood'][0]).toBeCloseTo(2 * Math.log(0.2) - 2, 5);
    if (withProbabilities) expect(Array.from(result['backgroundProbability'])).toEqual([1, 1]);
  });
}

for (const scene of ['endpoint', 'extreme-window'] as const) {
  it(`publishes consistent fitted Hawkes diagnostics for ${scene}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rowCount = scene === 'endpoint' ? 2 : 4;
    const props: SpaceTimeHawkesOperationProps = {
      id: 'hawkes-review-diagnostics',
      rowCount,
      maximumIterations: 1,
      positions: 'positions',
      times: 'times',
      parameters: 'parameters',
      output: {
        estimates: 'estimates',
        fittedLogLikelihood: 'fittedLogLikelihood',
        backgroundProbability: 'backgroundProbability'
      }
    };
    const inputs = {
      positions: {format: 'float32x2' as const, values: new Float32Array(2 * rowCount)},
      times: {
        format: 'float32' as const,
        values: new Float32Array(rowCount).fill(scene === 'endpoint' ? 10 : 0)
      },
      parameters: {
        format: 'float32' as const,
        values:
          scene === 'endpoint'
            ? Float32Array.of(1, 10, 10, 10, 1e-8, 1, 0.5, 1, 1)
            : Float32Array.of(2 ** 100, 2 ** -126, 10, 10, 1e-8, 1, 0.5, 2 ** 126, 1)
      }
    };
    for (const Operation of [GPUSpaceTimeHawkesOperation, CPUSpaceTimeHawkesOperation]) {
      const result = await runOperationGraph(device, {
        inputs,
        operations: [new Operation(props)],
        outputs: Object.values(props.output) as string[]
      });
      if (scene === 'endpoint') {
        for (const output of Object.values(result))
          expect(Array.from(output).every(Number.isNaN)).toBe(true);
      } else {
        expect(result['estimates'][0]).toBe(2 ** 28);
        expect(Number.isFinite(result['fittedLogLikelihood'][0])).toBe(true);
        expect(Math.abs(result['fittedLogLikelihood'][0] - 73.6324844)).toBeLessThan(1e-4);
      }
    }
  });
}
