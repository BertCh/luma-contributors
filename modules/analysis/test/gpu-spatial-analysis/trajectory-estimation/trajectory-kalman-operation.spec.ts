// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUTrajectoryKalmanOperation} from '../../../src/gpu-spatial-analysis/trajectory-estimation/cpu-trajectory-kalman-operation';
import {
  GPUTrajectoryKalmanOperation,
  type TrajectoryKalmanOperationProps
} from '../../../src/gpu-spatial-analysis/trajectory-estimation/gpu-trajectory-kalman-operation';
import {runOperationGraph, type OperationGraphInput} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import fixture from './trajectory-kalman-batch.fixture.json';

const OUTPUT = {
  filteredState: 'state',
  filteredCovariance: 'covariance',
  status: 'status',
  smoothedState: 'smoothState',
  smoothedCovariance: 'smoothCovariance'
};

async function runComparison(
  inputs: Record<string, OperationGraphInput>,
  covarianceMode: 'full' | 'isotropic',
  rowCount: number,
  wordTime = false
) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const props: TrajectoryKalmanOperationProps = {
    rowCount,
    positions: 'positions',
    times: 'times',
    offsets: 'offsets',
    parameters: 'parameters',
    timestampFormat: wordTime ? 'uint32x2' : 'float32',
    timeUnitsPerSecond: wordTime ? 1000 : undefined,
    covarianceMode,
    output: OUTPUT
  };
  const gpu = await runOperationGraph(device, {
    inputs,
    operations: [new GPUTrajectoryKalmanOperation(props)],
    outputs: Object.values(OUTPUT)
  });
  const cpu = await runOperationGraph(device, {
    inputs,
    operations: [new CPUTrajectoryKalmanOperation(props)],
    outputs: Object.values(OUTPUT)
  });
  for (const name of Object.values(OUTPUT)) {
    expect(gpu[name].length).toBe(cpu[name].length);
    for (let index = 0; index < cpu[name].length; index++) {
      const actual = gpu[name][index];
      const expected = cpu[name][index];
      if (name === 'status') expect(actual).toBe(expected);
      else if (Number.isNaN(expected)) expect(Number.isNaN(actual), `${name}[${index}]`).toBe(true);
      // Positive variances compare relatively, so small variances cannot pass as zero.
      // Signed states and cross-covariances also allow absolute 3e-5 near cancellation.
      // Numeric equality handles signed zero without asserting its sign.
      else if (actual !== expected) {
        const isVariance =
          (name === 'covariance' || name === 'smoothCovariance') &&
          (covarianceMode === 'full' ? (index % 16) % 5 === 0 : index % 4 === 0 || index % 4 === 3);
        const tolerance =
          isVariance && expected > 0 ? 3e-4 * expected : 3e-5 + 3e-4 * Math.abs(expected);
        expect(Math.abs(actual - expected), `${name}[${index}]`).toBeLessThanOrEqual(tolerance);
      }
    }
  }
  return cpu;
}

for (const covarianceMode of ['isotropic', 'full'] as const) {
  it(`compares every ${covarianceMode} filtered and smoothed output across tracks, missing fixes and a gap`, async () => {
    const testCase = fixture.cases.find(testCase => testCase.covarianceMode === covarianceMode)!;
    const rows = await runComparison(
      {
        positions: {
          format: 'float32x2',
          values: Float32Array.from(testCase.positions, value => (value === null ? NaN : value))
        },
        times: {format: 'float32', values: Float32Array.from(testCase.timestamps)},
        offsets: {format: 'uint32', values: Uint32Array.from(testCase.trackOffsets)},
        parameters: {format: 'float32', values: Float32Array.from(testCase.parameters)}
      },
      covarianceMode,
      8
    );
    expect(Array.from(rows['status'])).toEqual(testCase.status);
    expect(Math.abs(rows['smoothState'][0] - rows['state'][0])).toBeGreaterThan(0.1);
    expect(rows['smoothCovariance'][0]).toBeLessThan(rows['covariance'][0]);
  });

  it(`compares ${covarianceMode} invalid rows, missing initial fixes, decreasing time and resets`, async () => {
    const rows = await runComparison(
      {
        positions: {
          format: 'float32x2',
          values: Float32Array.of(NaN, 0, 0, 0, NaN, NaN, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0)
        },
        times: {format: 'float32', values: Float32Array.of(0, 1, 2, 1, 4, 10, NaN, 11)},
        offsets: {format: 'uint32', values: Uint32Array.of(0, 5, 8)},
        parameters: {
          format: 'float32',
          values:
            covarianceMode === 'full'
              ? Float32Array.of(1, 0.2, 2, 2, 0.3, 3, 3, -0.1, 2, 1)
              : Float32Array.of(1, 2, 3, 1)
        }
      },
      covarianceMode,
      8
    );
    expect(Array.from(rows['status'])).toEqual([6, 1, 2, 4, 1, 1, 4, 1]);
  });

  it(`compares ${covarianceMode} singular smoothing and invalid parameters`, async () => {
    const inputs: Record<string, OperationGraphInput> = {
      positions: {format: 'float32x2', values: Float32Array.of(0, 2, 4, 0, 2, 4)},
      times: {format: 'float32', values: Float32Array.of(0, 1, 2)},
      offsets: {format: 'uint32', values: Uint32Array.of(0, 3)},
      parameters: {
        format: 'float32',
        values:
          covarianceMode === 'full'
            ? Float32Array.of(0, 0, 0, 6, 0, 6, 0, 0, 0, 0)
            : Float32Array.of(0, 6, 0, 0)
      }
    };
    const rows = await runComparison(inputs, covarianceMode, 3);
    expect(Array.from(rows['status'])).toEqual([9, 8, 0]);
    const invalid = await runComparison(
      {
        ...inputs,
        parameters: {
          format: 'float32',
          values:
            covarianceMode === 'full'
              ? Float32Array.of(1, 2, 1, 6, 0, 6, 0, 0, 0, 0)
              : Float32Array.of(-1, 6, 0, 0)
        }
      },
      covarianceMode,
      3
    );
    expect(Array.from(invalid['status'])).toEqual([4, 4, 4]);
  });
}

it('compares the reviewed simultaneous-fix singular smoothing fallback', async () => {
  const rows = await runComparison(
    {
      positions: {format: 'float32x2', values: Float32Array.of(2, 4, 6, 8)},
      times: {format: 'float32', values: Float32Array.of(0, 0)},
      offsets: {format: 'uint32', values: Uint32Array.of(0, 2)},
      parameters: {format: 'float32', values: Float32Array.of(0, 0, 0, 2, 0, 2, 0, 0, 0, 0)}
    },
    'full',
    2
  );
  // Equal independent measurement variances give the arithmetic mean and halve the variance.
  // The deterministic velocity block makes smoothing singular, so forward values are retained.
  const expectedState = [2, 4, 0, 0, 4, 6, 0, 0];
  const expectedCovariance = [
    2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  ];
  expect(Array.from(rows['status'])).toEqual([9, 0]);
  for (const name of ['state', 'smoothState']) {
    for (let index = 0; index < expectedState.length; index++)
      expect(rows[name][index] === expectedState[index]).toBe(true);
  }
  for (const name of ['covariance', 'smoothCovariance']) {
    for (let index = 0; index < expectedCovariance.length; index++)
      expect(rows[name][index] === expectedCovariance[index]).toBe(true);
  }
});

it('compares Int64 time differences near the signed endpoint with millisecond scaling', async () => {
  const testCase = fixture.cases[1];
  const origin = 2n ** 63n - 10000n;
  const words = Uint32Array.from(
    testCase.timestamps.flatMap(time => {
      const ticks = origin + BigInt(time * 1000);
      return [Number(BigInt.asUintN(32, ticks)), Number(BigInt.asUintN(32, ticks >> 32n))];
    })
  );
  const parameters = Float32Array.from(testCase.parameters);
  parameters[9] *= 1000;
  const rows = await runComparison(
    {
      positions: {
        format: 'float32x2',
        values: Float32Array.from(testCase.positions, value => (value === null ? NaN : value))
      },
      times: {format: 'uint32x2', values: words},
      offsets: {format: 'uint32', values: Uint32Array.from(testCase.trackOffsets)},
      parameters: {format: 'float32', values: parameters}
    },
    'full',
    8,
    true
  );
  expect(Array.from(rows['status'])).toEqual(testCase.status);
});
