// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';

it('honors an explicit one-row search budget on a stationary track', () => {
  const result = computeTrajectoryThinningOnCPU(
    {
      positions: new Float32Array(8),
      timestamps: Float32Array.of(0, 1, 2, 3),
      trackOffsets: Uint32Array.of(0, 4),
      parameters: Float32Array.of(10, 0, 0, 0)
    },
    {mode: 'minimum-time', capacity: 4, maximumSearchRows: 1}
  );
  expect(Array.from(result.rowIds)).toEqual([0, 2, 3, 0xffffffff]);
  expect(result.candidateOverflow[0]).toBe(1);
});
import {
  computeTrajectoryThinningOnCPU,
  CPUTrajectoryThinningOperation
} from '../../../src/gpu-spatial-analysis/trajectory-cleaning/cpu-trajectory-thinning-operation';
import {
  GPUTrajectoryThinningOperation,
  type TrajectoryThinningOperationProps
} from '../../../src/gpu-spatial-analysis/trajectory-cleaning/gpu-trajectory-thinning-operation';
import {getGPUTrajectoryThinningParameterValues} from '../../../src/gpu-spatial-analysis/trajectory-cleaning/gpu-trajectory-thinning';

const INPUT = {
  positions: new Float32Array([0, 0, 1, 0, 0, 0, 3, 4, 3, 4, 6, 8, 10, 0, 11, 0, 12, 0, 13, 0]),
  timestamps: new Float32Array([0, 1, 2, 3, 7, 8, 0, 2, 5, 6]),
  trackOffsets: new Uint32Array([0, 6, 6, 10]),
  parameters: getGPUTrajectoryThinningParameterValues({tolerance: 5})
};

it('follows hand-computed sequential chains, including equal tolerance and forced last fixes', () => {
  for (const [mode, tolerance, rows, offsets] of [
    ['minimum-distance', 5, [0, 3, 5, 6, 9], [0, 3, 3, 5]],
    ['minimum-time', 3, [0, 3, 4, 5, 6, 8, 9], [0, 4, 4, 7]]
  ] as const) {
    const result = computeTrajectoryThinningOnCPU(
      {...INPUT, parameters: getGPUTrajectoryThinningParameterValues({tolerance})},
      {mode, capacity: 8}
    );
    expect(Array.from(result.rowIds)).toEqual([
      ...rows,
      ...new Array(8 - rows.length).fill(0xffffffff)
    ]);
    expect(Array.from(result.outputTrackOffsets)).toEqual(offsets);
    expect(Array.from(result.validity)).toEqual(
      Array.from({length: 10}, (_, row) => Number((rows as readonly number[]).includes(row)))
    );
    expect(result.requiredCount[0]).toBe(rows.length);
    expect(result.count[0]).toBe(rows.length);
    expect(result.overflow[0]).toBe(0);
    expect(result.candidateOverflow[0]).toBe(0);
    expect(Array.from(result.outputPositions)).toEqual([
      ...rows.flatMap(row => Array.from(INPUT.positions.subarray(row * 2, row * 2 + 2))),
      ...new Array(2 * (8 - rows.length)).fill(0)
    ]);
    expect(Array.from(result.outputTimestamps)).toEqual([
      ...rows.map(row => INPUT.timestamps[row]),
      ...new Array(8 - rows.length).fill(0)
    ]);
  }
  const overflow = computeTrajectoryThinningOnCPU(INPUT, {mode: 'minimum-distance', capacity: 2});
  expect(Array.from(overflow.rowIds)).toEqual([0, 3]);
  expect(Array.from(overflow.outputTrackOffsets)).toEqual([0, 2, 2, 2]);
  expect(overflow.requiredCount[0]).toBe(5);
  expect(overflow.overflow[0]).toBe(1);
  expect(Array.from(overflow.validity)).toEqual([1, 0, 0, 1, 0, 1, 1, 0, 0, 1]);
});

it('uses spherical arc lengths and identifies equivalent dateline and pole fixes', () => {
  for (const [positions, tolerance, expected] of [
    [[0, 0, 30, 0, 90, 0, 180, 0], 1, [0, 2, 3]],
    [[-180, 0, 180, 0, -179, 0], 1e-10, [0, 2]],
    [[0, 90, 180, 90, 10, 90], 1e-10, [0, 2]],
    [[0, -90, 180, -90, 10, -90], 1e-10, [0, 2]],
    [[179.999, 0, -179.999, 0, -179.998, 0], 0.00002, [0, 1, 2]]
  ] as const) {
    const rowCount = positions.length / 2;
    const result = computeTrajectoryThinningOnCPU(
      {
        positions: new Float32Array(positions),
        timestamps: new Float32Array(rowCount),
        trackOffsets: new Uint32Array([0, rowCount]),
        parameters: getGPUTrajectoryThinningParameterValues({tolerance})
      },
      {
        mode: 'minimum-distance',
        capacity: expected.length,
        spatialContext: {
          coordinateSpace: 'longitude-latitude',
          metric: 'great-circle',
          units: 'meters',
          sphereRadius: 1
        }
      }
    );
    // On the equator the central angle is the absolute longitude separation in radians.
    expect(Array.from(result.rowIds)).toEqual(expected);
  }
});

it('keeps only endpoints on long stationary tracks instead of introducing search-cap fixes', () => {
  const rowCount = 5000;
  const result = computeTrajectoryThinningOnCPU(
    {
      positions: new Float32Array(rowCount * 2),
      timestamps: new Float32Array(rowCount),
      trackOffsets: new Uint32Array([0, rowCount]),
      parameters: INPUT.parameters
    },
    {mode: 'minimum-distance', capacity: 2}
  );
  expect(Array.from(result.rowIds)).toEqual([0, rowCount - 1]);
  expect(result.candidateOverflow[0]).toBe(0);
});

it('handles exact Int64 differences across a low-word carry, partial tracks and zero tolerance', () => {
  const origin = (1n << 60n) - 1n;
  const timestamps = new Uint32Array(8);
  [0n, 1n, 3n, 4n].forEach((delta, row) => {
    const timestamp = origin + delta;
    timestamps[row * 2] = Number(timestamp & 0xffffffffn);
    timestamps[row * 2 + 1] = Number(timestamp >> 32n);
  });
  const input = {
    positions: new Float32Array(8),
    timestamps,
    trackOffsets: new Uint32Array([0, 99]),
    parameters: getGPUTrajectoryThinningParameterValues({tolerance: 3})
  };
  const result = computeTrajectoryThinningOnCPU(input, {mode: 'minimum-time', capacity: 3});
  expect(Array.from(result.rowIds)).toEqual([0, 2, 3]);
  expect(Array.from(result.outputTimestamps)).toEqual([
    ...timestamps.subarray(0, 2),
    ...timestamps.subarray(4)
  ]);
  const partial = computeTrajectoryThinningOnCPU(
    {
      ...input,
      trackOffsets: new Uint32Array([1, 3]),
      parameters: getGPUTrajectoryThinningParameterValues({tolerance: 0})
    },
    {mode: 'minimum-time', capacity: 2}
  );
  expect(Array.from(partial.validity)).toEqual([0, 1, 1, 0]);
});

it('rejects inconsistent arrays and validates construction-time output contracts', () => {
  for (const [argument, replacement] of [
    ['positions', new Float32Array(3)],
    ['timestamps', new Float32Array(1)],
    ['trackOffsets', new Uint32Array([4, 2])],
    ['parameters', new Float32Array(3)]
  ] as const)
    expect(() =>
      computeTrajectoryThinningOnCPU(
        {...INPUT, [argument]: replacement},
        {mode: 'minimum-time', capacity: 2}
      )
    ).toThrow(new RegExp(argument));
  const props: TrajectoryThinningOperationProps = {
    id: 'thinning-test',
    mode: 'minimum-distance',
    positions: 'positions',
    timestamps: 'timestamps',
    trackOffsets: 'trackOffsets',
    parameters: 'parameters',
    rowCount: 10,
    trackCount: 3,
    capacity: 5,
    output: {
      rowIds: 'rowIds',
      trackOffsets: 'outputTrackOffsets',
      status: {count: 'count', overflow: 'overflow'}
    }
  };
  for (const Operation of [GPUTrajectoryThinningOperation, CPUTrajectoryThinningOperation]) {
    expect(() => new Operation({...props, capacity: 0})).toThrow(/^thinning-test /);
    expect(() => new Operation({...props, output: {...props.output, rowIds: 'positions'}})).toThrow(
      /^thinning-test /
    );
  }
  expect(() => new CPUTrajectoryThinningOperation({...props, trackCount: 0})).toThrow(
    /^thinning-test trackCount/
  );
  expect(
    () => new CPUTrajectoryThinningOperation({...props, maximumSearchRows: 1e9})
  ).not.toThrow();
});
