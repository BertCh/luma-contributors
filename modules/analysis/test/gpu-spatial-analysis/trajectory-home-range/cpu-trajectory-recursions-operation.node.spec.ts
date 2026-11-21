// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeTrajectoryRecursionsOnCPU,
  CPUTrajectoryRecursionsOperation
} from '../../../src/gpu-spatial-analysis/trajectory-recursion/cpu-trajectory-recursions-operation';
import {
  GPUTrajectoryRecursionsOperation,
  type TrajectoryRecursionsOperationProps
} from '../../../src/gpu-spatial-analysis/trajectory-recursion/gpu-trajectory-recursions-operation';

function getInput() {
  return {
    storage: 'resident-capacity' as const,
    positions: new Float32Array([0, 0, 2, 0, 0, 0, 8, 0, 8, 0, 20, 0, 99, 0]),
    timestamps: new Float32Array([0, 2, 4, 100, 103, 200, 999]),
    trackOffsets: new Uint32Array([0, 3, 3, 5, 6]),
    radii: new Float32Array([1, 2, 0])
  };
}

it('clamps terminal offsets for a two-fix stationary track', () => {
  const result = computeTrajectoryRecursionsOnCPU({
    storage: 'resident-capacity',
    positions: new Float32Array(4),
    timestamps: new Float32Array([0, 1]),
    trackOffsets: new Uint32Array([0, 3]),
    radii: new Float32Array([1])
  });
  expect(Array.from(result.revisits)).toEqual([0, 0]);
  expect(Array.from(result.residenceTimes)).toEqual([1, 1]);
  expect(Array.from(result.firstPassageTimes)).toEqual([1, 0]);
  expect(Array.from(result.firstPassageCensored)).toEqual([1, 1]);
  expect(Array.from(result.candidateOverflow)).toEqual([0]);
});

it('executes empty trajectories in the pure function and CPU named form', async () => {
  const result = computeTrajectoryRecursionsOnCPU({
    positions: new Float32Array(),
    timestamps: new Float32Array(),
    trackOffsets: new Uint32Array([0, 0]),
    radii: new Float32Array([1])
  });
  expect(result.revisits.length).toBe(0);
  expect(result.residenceTimes.length).toBe(0);
  expect(result.firstPassageTimes.length).toBe(0);
  expect(result.firstPassageCensored.length).toBe(0);
  expect(Array.from(result.candidateOverflow)).toEqual([0]);
  const operation = new CPUTrajectoryRecursionsOperation({
    positions: 'positions',
    timestamps: 'timestamps',
    trackOffsets: 'trackOffsets',
    radii: 'radii',
    rowCount: 0,
    radiusCount: 1,
    output: {
      revisits: 'revisits',
      residenceTimes: 'residence',
      firstPassageTimes: 'passage',
      firstPassageCensored: 'censored',
      status: {candidateOverflow: 'loss'}
    }
  });
  const column = (
    format: 'float32' | 'float32x2' | 'uint32',
    values: Float32Array | Uint32Array
  ) => {
    const length = values.length / (format === 'float32x2' ? 2 : 1);
    return {format, length, chunks: [{values, length}], pack: () => values};
  };
  const outputs = await operation.execute({
    inputs: {
      positions: column('float32x2', new Float32Array()),
      timestamps: column('float32', new Float32Array()),
      trackOffsets: column('uint32', Uint32Array.of(0, 0)),
      radii: column('float32', Float32Array.of(1))
    }
  });
  for (const port of ['revisits', 'residence', 'passage', 'censored'])
    expect(Array.from(outputs[port] as Float32Array | Uint32Array)).toEqual([]);
  expect(Array.from(outputs['loss'] as Uint32Array)).toEqual([0]);
});

it('accepts empty radii only in the pure function', () => {
  const result = computeTrajectoryRecursionsOnCPU({...getInput(), radii: new Float32Array()});
  expect(result.revisits.length).toBe(0);
  expect(result.residenceTimes.length).toBe(0);
  expect(result.firstPassageTimes.length).toBe(0);
  expect(result.firstPassageCensored.length).toBe(0);
  expect(Array.from(result.candidateOverflow)).toEqual([0]);
  expect(
    () =>
      new CPUTrajectoryRecursionsOperation({
        positions: 'positions',
        timestamps: 'timestamps',
        trackOffsets: 'trackOffsets',
        radii: 'radii',
        rowCount: 7,
        radiusCount: 0,
        output: {revisits: 'revisits'}
      })
  ).toThrow(/length|radiusCount/);
});

it('uses no inside interval for NaN geometry and the segment start for an unresolvable exit', () => {
  const result = computeTrajectoryRecursionsOnCPU({
    positions: new Float32Array([0, 0, NaN, 0, 2, 0]),
    timestamps: new Float32Array([0, 1, 2]),
    trackOffsets: new Uint32Array([0, 3]),
    radii: new Float32Array([1])
  });
  // Both segments touch NaN, so residence is zero. For focal row 0, the known outside
  // endpoint at t=2 ends passage on segment 1; its missing interval uses the start t=1.
  expect(Array.from(result.revisits)).toEqual([0, 0, 0]);
  expect(Array.from(result.residenceTimes)).toEqual([0, 0, 0]);
  expect(Array.from(result.firstPassageTimes)).toEqual([1, 1, 0]);
  expect(Array.from(result.firstPassageCensored)).toEqual([0, 1, 1]);
  expect(Array.from(result.candidateOverflow)).toEqual([0]);
});

it('matches hand-computed crossings for out-and-back, stationary, singleton and unassigned fixes', () => {
  const result = computeTrajectoryRecursionsOnCPU(getInput());
  expect(Array.from(result.revisits)).toEqual([
    1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  ]);
  expect(Array.from(result.residenceTimes)).toEqual([
    2, 4, 0, 2, 4, 0, 2, 4, 0, 3, 3, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0
  ]);
  expect(Array.from(result.firstPassageTimes)).toEqual([
    1, 4, 0, 1, 2, 0, 0, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  ]);
  expect(Array.from(result.firstPassageCensored)).toEqual([
    0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
  ]);
  expect(Array.from(result.candidateOverflow)).toEqual([0]);
});

it('is translation invariant and uses exact signed Int64 differences', () => {
  const input = getInput();
  const expected = computeTrajectoryRecursionsOnCPU(input);
  const shifted = Float32Array.from(input.positions, value => value + 2 ** 20);
  const timestamps = new Uint32Array(input.timestamps.length * 2);
  const origin = -(1n << 62n) - 1n;
  input.timestamps.forEach((timestamp, row) => {
    const words = BigInt.asUintN(64, origin + BigInt(timestamp));
    timestamps[row * 2] = Number(words & 0xffffffffn);
    timestamps[row * 2 + 1] = Number(words >> 32n);
  });
  const actual = computeTrajectoryRecursionsOnCPU({...input, positions: shifted, timestamps});
  expect(actual).toEqual(expected);
});

it('examines the complete track beyond the GPU default segment cap', () => {
  const count = 4100;
  const result = computeTrajectoryRecursionsOnCPU({
    positions: new Float32Array(count * 2),
    timestamps: Float32Array.from({length: count}, (_, row) => row),
    trackOffsets: new Uint32Array([0, count]),
    radii: new Float32Array([1])
  });
  expect(result.residenceTimes[0]).toBe(count - 1);
  expect(result.firstPassageTimes[0]).toBe(count - 1);
  expect(result.firstPassageCensored[0]).toBe(1);
  expect(result.candidateOverflow[0]).toBe(0);
});

it('counts a tangency as a closed-disc visit and ignores non-positive radii', () => {
  const result = computeTrajectoryRecursionsOnCPU({
    positions: new Float32Array([0, 0, 2, 2, -2, 2]),
    timestamps: new Float32Array([0, 1, 2]),
    trackOffsets: new Uint32Array([0, 3]),
    radii: new Float32Array([2, -1, NaN])
  });
  // The first segment exits at fraction 1/sqrt(2); the next touches (0,2) once.
  expect(result.revisits[0]).toBe(1);
  expect(result.residenceTimes[0]).toBe(Math.fround(1 / Math.sqrt(2)));
  expect(result.firstPassageTimes[0]).toBe(Math.fround(1 / Math.sqrt(2)));
  for (const slot of [1, 2, 4, 5, 7, 8]) {
    expect(result.revisits[slot]).toBe(0);
    expect(result.residenceTimes[slot]).toBe(0);
    expect(result.firstPassageTimes[slot]).toBe(0);
    expect(result.firstPassageCensored[slot]).toBe(1);
  }
});

it('validates array consistency and both operation output shapes at construction', () => {
  const input = getInput();
  expect(() =>
    computeTrajectoryRecursionsOnCPU({...input, timestamps: new Float32Array(1)})
  ).toThrow(/timestamps/);
  expect(() =>
    computeTrajectoryRecursionsOnCPU({...input, trackOffsets: new Uint32Array([7, 0])})
  ).toThrow(/trackOffsets/);
  const props: TrajectoryRecursionsOperationProps = {
    id: 'recursion-test',
    positions: 'positions',
    timestamps: 'timestamps',
    trackOffsets: 'trackOffsets',
    radii: 'radii',
    rowCount: 7,
    radiusCount: 3,
    output: {revisits: 'revisits'}
  };
  for (const Operation of [GPUTrajectoryRecursionsOperation, CPUTrajectoryRecursionsOperation]) {
    expect(() => new Operation({...props, rowCount: -1})).toThrow(/^recursion-test .*rowCount/);
    expect(() => new Operation({...props, output: {revisits: 'positions'}})).toThrow(
      /^recursion-test /
    );
  }
  expect(
    () =>
      new CPUTrajectoryRecursionsOperation({...props, output: {firstPassageCensored: 'censored'}})
  ).toThrow(/^recursion-test output/);
  expect(
    () =>
      new CPUTrajectoryRecursionsOperation({
        ...props,
        radiusCount: 33,
        maximumTrackSegments: 100000
      })
  ).not.toThrow();
});

it('rejects clamping and unowned tails under explicit compact storage', () => {
  const input = getInput();
  expect(() => computeTrajectoryRecursionsOnCPU({...input, storage: 'compact'})).toThrow(
    /terminal offset/
  );
  expect(() =>
    computeTrajectoryRecursionsOnCPU({
      storage: 'compact',
      positions: new Float32Array(4),
      timestamps: Float32Array.of(0, 1),
      trackOffsets: Uint32Array.of(0, 3),
      radii: Float32Array.of(1)
    })
  ).toThrow(/trackOffsets/);
});
