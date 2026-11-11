// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {invertCPUScaledMatrix} from '../../../src/analysis-core/kernels/cpu-scaled-inverse';
import {
  CPUTrajectoryKalmanOperation,
  computeTrajectoryKalmanOnCPU,
  type CPUTrajectoryKalmanInput
} from '../../../src/gpu-spatial-analysis/trajectory-estimation/cpu-trajectory-kalman-operation';
import {
  GPUTrajectoryKalmanOperation,
  type TrajectoryKalmanOperationProps
} from '../../../src/gpu-spatial-analysis/trajectory-estimation/gpu-trajectory-kalman-operation';
import fixture from './trajectory-kalman-batch.fixture.json';

const NUMERIC_PORTS = [
  'filteredState',
  'filteredCovariance',
  'smoothedState',
  'smoothedCovariance'
] as const;

function createInput(testCase = fixture.cases[0]): CPUTrajectoryKalmanInput {
  return {
    positions: Float32Array.from(testCase.positions, value => (value === null ? NaN : value)),
    timestamps: Float32Array.from(testCase.timestamps),
    trackOffsets: Uint32Array.from(testCase.trackOffsets),
    parameters: Float32Array.from(testCase.parameters)
  };
}

function packTimeWords(ticks: readonly bigint[]): Uint32Array {
  return Uint32Array.from(
    ticks.flatMap(value => [
      Number(BigInt.asUintN(32, value)),
      Number(BigInt.asUintN(32, value >> 32n))
    ])
  );
}

it('matches independent NumPy joint Gaussian solves for every filtered and smoothed moment', () => {
  for (const testCase of fixture.cases) {
    const result = computeTrajectoryKalmanOnCPU(createInput(testCase), {
      covarianceMode: testCase.covarianceMode as 'full' | 'isotropic',
      smooth: true
    });
    expect(Array.from(result.status)).toEqual(testCase.status);
    for (const port of NUMERIC_PORTS) {
      const actual = result[port]!;
      expect(actual.length).toBe(testCase[port].length);
      for (let index = 0; index < actual.length; index++) {
        const expected = testCase[port][index];
        expect(
          Math.abs(actual[index] - expected),
          `${testCase.covarianceMode}.${port}[${index}]`
        ).toBeLessThanOrEqual(1e-6 * Math.max(1, Math.abs(expected)));
      }
    }
    expect(Math.abs(result.smoothedState![0] - result.filteredState[0])).toBeGreaterThan(0.1);
    expect(result.smoothedCovariance![0]).toBeLessThan(result.filteredCovariance[0]);
  }
});

it('agrees with the closed-form static posterior and retains forward values on singular smoothing', () => {
  for (const full of [false, true]) {
    const result = computeTrajectoryKalmanOnCPU(
      {
        positions: Float32Array.of(0, 2, 4, 0, 2, 4),
        timestamps: Float32Array.of(0, 1, 2),
        trackOffsets: Uint32Array.of(0, 3),
        parameters: full
          ? Float32Array.of(0, 0, 0, 6, 0, 6, 0, 0, 0, 0)
          : Float32Array.of(0, 6, 0, 0)
      },
      {covarianceMode: full ? 'full' : 'isotropic', smooth: true}
    );
    expect(Array.from(result.status)).toEqual([9, 8, 0]);
    // Constant position, three independent variance-6 measurements: mean (2,2), variance 2.
    expect(Array.from(result.filteredState.slice(8))).toEqual([2, 2, 0, 0]);
    expect(result.filteredCovariance[2 * (full ? 16 : 4)]).toBe(2);
    for (const port of ['State', 'Covariance'] as const) {
      const filtered = result[`filtered${port}`];
      const smoothed = result[`smoothed${port}`]!;
      for (let index = 0; index < filtered.length; index++)
        expect(smoothed[index] === filtered[index]).toBe(true);
    }
  }
});

it('retains forward values for the reviewed full-covariance case with simultaneous fixes', () => {
  const result = computeTrajectoryKalmanOnCPU(
    {
      positions: Float32Array.of(2, 4, 6, 8),
      timestamps: Float32Array.of(0, 0),
      trackOffsets: Uint32Array.of(0, 2),
      parameters: Float32Array.of(0, 0, 0, 2, 0, 2, 0, 0, 0, 0)
    },
    {covarianceMode: 'full', smooth: true}
  );
  // No elapsed time or process noise: two independent variance-2 measurements give mean (4,6)
  // and position variance 1. Velocity is deterministic zero, making the predicted 4x4 covariance
  // singular. The smoother keeps both forward moments and adds bit 8 to the first row only.
  const expectedState = [2, 4, 0, 0, 4, 6, 0, 0];
  const expectedCovariance = [
    2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  ];
  expect(Array.from(result.status)).toEqual([9, 0]);
  for (const values of [result.filteredState, result.smoothedState!]) {
    for (let index = 0; index < expectedState.length; index++)
      expect(values[index] === expectedState[index]).toBe(true);
  }
  for (const values of [result.filteredCovariance, result.smoothedCovariance!]) {
    for (let index = 0; index < expectedCovariance.length; index++)
      expect(values[index] === expectedCovariance[index]).toBe(true);
  }
});

it('the shared inverse preserves pivoting, scaling and the exact-zero singularity policy', () => {
  // A scaled permutation/diagonal matrix has the inverse given by reciprocal nonzero entries
  // in transposed positions. Its 1e-8 pivot must remain accepted without a relative cutoff.
  const matrix = [
    [0, 2, 0, 0],
    [4, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1e-8]
  ];
  const expected = [
    [0, 0.25, 0, 0],
    [0.5, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1e8]
  ];
  for (const scale of [1, 1e-30, 1e30]) {
    const scaled = matrix.map(row => row.map(value => value * scale));
    const snapshot = scaled.map(row => row.slice());
    const inverse = invertCPUScaledMatrix(scaled);
    expect(inverse).toBeDefined();
    expect(scaled).toEqual(snapshot);
    for (let row = 0; row < 4; row++) {
      for (let column = 0; column < 4; column++) {
        const value = expected[row][column] / scale;
        if (value === 0) expect(inverse![row][column] === 0).toBe(true);
        else
          expect(Math.abs(inverse![row][column] - value)).toBeLessThanOrEqual(
            5e-15 * Math.abs(value)
          );
      }
    }
  }
  expect(
    invertCPUScaledMatrix([
      [1, 2],
      [2, 4]
    ])
  ).toBeUndefined();
  expect(invertCPUScaledMatrix([[0]])).toBeUndefined();
  expect(() => invertCPUScaledMatrix([[1, 2]])).toThrow('matrix');
});

it('keeps cross-axis covariance and x/y symmetry in full mode', () => {
  const input = createInput(fixture.cases[1]);
  const original = computeTrajectoryKalmanOnCPU(input, {covarianceMode: 'full', smooth: true});
  const parameters = Float32Array.from(input.parameters);
  for (const offset of [0, 3, 6])
    [parameters[offset], parameters[offset + 2]] = [parameters[offset + 2], parameters[offset]];
  const positions = Float32Array.from(input.positions);
  for (let row = 0; row < positions.length / 2; row++)
    [positions[row * 2], positions[row * 2 + 1]] = [positions[row * 2 + 1], positions[row * 2]];
  const swapped = computeTrajectoryKalmanOnCPU(
    {...input, parameters, positions},
    {covarianceMode: 'full', smooth: true}
  );
  const permutation = [1, 0, 3, 2];
  for (let row = 0; row < input.timestamps.length; row++) {
    for (let component = 0; component < 4; component++) {
      expect(swapped.filteredState[row * 4 + component]).toBeCloseTo(
        original.filteredState[row * 4 + permutation[component]],
        6
      );
      for (let column = 0; column < 4; column++)
        expect(swapped.smoothedCovariance![row * 16 + column * 4 + component]).toBeCloseTo(
          original.smoothedCovariance![row * 16 + permutation[column] * 4 + permutation[component]],
          6
        );
    }
  }
  expect(Math.abs(original.filteredCovariance[1])).toBeGreaterThan(0.1);
});

it('preserves the Gaussian scaling invariant at large and small covariance magnitudes', () => {
  for (const testCase of fixture.cases) {
    const input = createInput(testCase);
    const options = {covarianceMode: testCase.covarianceMode as 'full' | 'isotropic', smooth: true};
    const reference = computeTrajectoryKalmanOnCPU(input, options);
    for (const scale of [1e-15, 1e15]) {
      const parameters = Float32Array.from(input.parameters, (value, index) =>
        index + 1 === input.parameters.length ? value : value * scale * scale
      );
      const result = computeTrajectoryKalmanOnCPU(
        {
          ...input,
          parameters,
          positions: Float32Array.from(input.positions, value => value * scale)
        },
        options
      );
      expect(Array.from(result.status)).toEqual(Array.from(reference.status));
      for (const port of NUMERIC_PORTS) {
        const outputScale = port.endsWith('Covariance') ? scale * scale : scale;
        for (let index = 0; index < result[port]!.length; index++) {
          const expected = reference[port]![index];
          expect(
            Math.abs(result[port]![index] / outputScale - expected),
            `${port}[${index}] scale ${scale}`
          ).toBeLessThanOrEqual(2e-6 * Math.max(1, Math.abs(expected)));
        }
      }
    }
  }
});

it('subtracts signed Int64 timestamps exactly near either endpoint and honors units and gaps', () => {
  for (const testCase of fixture.cases) {
    const input = createInput(testCase);
    const covarianceMode = testCase.covarianceMode as 'isotropic' | 'full';
    const reference = computeTrajectoryKalmanOnCPU(input, {covarianceMode, smooth: true});
    for (const origin of [2n ** 63n - 10000n, -(2n ** 63n)]) {
      const timestamps = packTimeWords(
        testCase.timestamps.map(time => origin + BigInt(time * 1000))
      );
      const parameters = Float32Array.from(input.parameters);
      parameters[parameters.length - 1] *= 1000;
      const result = computeTrajectoryKalmanOnCPU(
        {...input, timestamps, parameters},
        {covarianceMode, smooth: true, timeUnitsPerSecond: 1000}
      );
      for (const port of [...NUMERIC_PORTS, 'status'] as const)
        expect(Array.from(result[port]!)).toEqual(Array.from(reference[port]!));
    }
  }
});

it('classifies missing measurements, decreasing/nonfinite times, gaps and independent tracks', () => {
  const result = computeTrajectoryKalmanOnCPU(
    {
      positions: Float32Array.of(NaN, 0, 0, 0, NaN, NaN, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0),
      timestamps: Float32Array.of(0, 1, 2, 1, 4, 10, NaN, 11),
      trackOffsets: Uint32Array.of(0, 5, 8),
      parameters: Float32Array.of(1, 2, 3, 1)
    },
    {smooth: true}
  );
  expect(Array.from(result.status)).toEqual([6, 1, 2, 4, 1, 1, 4, 1]);
  for (const row of [0, 3, 6]) {
    for (const port of NUMERIC_PORTS)
      expect(result[port]!.slice(row * 4, (row + 1) * 4).every(Number.isNaN)).toBe(true);
  }
  expect(Array.from(result.filteredState.slice(16, 20))).toEqual([5, 0, 0, 0]);
  expect(result.filteredCovariance[2 * 4]).toBeGreaterThan(result.filteredCovariance[1 * 4]);
});

it('returns status 4 for invalid parameter values, rather than throwing', () => {
  const input = createInput();
  for (const parameters of [
    Float32Array.of(-1, 2, 3, 0),
    Float32Array.of(1, 0, 3, 0),
    Float32Array.of(1, 2, 3, NaN)
  ]) {
    const result = computeTrajectoryKalmanOnCPU({...input, parameters}, {smooth: true});
    expect(Array.from(result.status)).toEqual(new Array(8).fill(4));
    for (const port of NUMERIC_PORTS) expect(result[port]!.every(Number.isNaN)).toBe(true);
  }
  const fullParameters = Float32Array.of(1, 2, 1, 2, 0, 2, 3, 0, 3, 0);
  expect(
    computeTrajectoryKalmanOnCPU(
      {...input, parameters: fullParameters},
      {covarianceMode: 'full'}
    ).status.every(value => value === 4)
  ).toBe(true);
});

it('clamps track offsets and leaves unowned rows zero, including no-track input', () => {
  const input = createInput();
  for (const trackOffsets of [Uint32Array.of(999), Uint32Array.of(999, 1000)]) {
    const result = computeTrajectoryKalmanOnCPU({...input, trackOffsets}, {smooth: true});
    for (const values of Object.values(result))
      expect(values!.every(value => value === 0)).toBe(true);
  }
  const empty = computeTrajectoryKalmanOnCPU({
    positions: new Float32Array(),
    timestamps: new Float32Array(),
    trackOffsets: Uint32Array.of(0),
    parameters: Float32Array.of(1, 1, 1, 0)
  });
  expect(empty.filteredState.length).toBe(0);
});

it('rejects inconsistent arrays with argument names', () => {
  const input = createInput();
  expect(() => computeTrajectoryKalmanOnCPU({...input, positions: Float32Array.of(0)})).toThrow(
    'positions'
  );
  expect(() => computeTrajectoryKalmanOnCPU({...input, timestamps: new Float32Array()})).toThrow(
    'timestamps'
  );
  expect(() => computeTrajectoryKalmanOnCPU({...input, trackOffsets: new Uint32Array()})).toThrow(
    'trackOffsets'
  );
  expect(() =>
    computeTrajectoryKalmanOnCPU({...input, trackOffsets: Uint32Array.of(3, 2)})
  ).toThrow('trackOffsets');
  expect(() =>
    computeTrajectoryKalmanOnCPU({...input, parameters: Float32Array.of(1, 1, 1)})
  ).toThrow('parameters');
});

const PROPS: TrajectoryKalmanOperationProps = {
  id: 'kalman-check',
  rowCount: 3,
  positions: 'positions',
  timestamps: 'timestamps',
  trackOffsets: 'offsets',
  parameters: 'parameters',
  output: {filteredState: 'state', filteredCovariance: 'covariance', status: 'status'}
};

describe('construction-time validation', () => {
  for (const Operation of [GPUTrajectoryKalmanOperation, CPUTrajectoryKalmanOperation]) {
    it(`${Operation.name} rejects bad output lengths and graph aliases`, () => {
      expect(() => new Operation({...PROPS, rowCount: 0})).toThrow(
        /^kalman-check.*(length|rowCount)/
      );
      expect(() => new Operation({...PROPS, rowCount: 1.5})).toThrow(
        /^kalman-check.*(length|rowCount)/
      );
      expect(
        () => new Operation({...PROPS, output: {...PROPS.output, filteredState: 'positions'}})
      ).toThrow(/^kalman-check.*outputs/);
      expect(() => new Operation({...PROPS, output: {...PROPS.output, status: 'state'}})).toThrow(
        /^kalman-check.*outputs/
      );
    });
  }
  it('CPU operation checks smoothing pairing, units and coordinate context', () => {
    expect(
      () =>
        new CPUTrajectoryKalmanOperation({
          ...PROPS,
          output: {...PROPS.output, smoothedState: 'smooth'}
        })
    ).toThrow(/^kalman-check.*smoothedState/);
    expect(() => new CPUTrajectoryKalmanOperation({...PROPS, timeUnitsPerSecond: 1000})).toThrow(
      /^kalman-check.*timeUnitsPerSecond/
    );
    expect(
      () =>
        new CPUTrajectoryKalmanOperation({
          ...PROPS,
          timestampFormat: 'uint32x2',
          timeUnitsPerSecond: 0
        })
    ).toThrow(/^kalman-check.*timeUnitsPerSecond/);
    expect(
      () =>
        new CPUTrajectoryKalmanOperation({
          ...PROPS,
          spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}
        })
    ).toThrow(/^kalman-check/);
    expect(
      () =>
        new CPUTrajectoryKalmanOperation({
          ...PROPS,
          temporalContext: {
            unit: 'millisecond',
            epoch: 0n,
            encoding: 'float32-relative',
            axis: {kind: 'irregular'}
          },
          timeUnitsPerSecond: 1
        })
    ).toThrow(/^kalman-check.*timeUnitsPerSecond/);
  });
});
