// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeTrajectorySplitOnCPU,
  CPUTrajectorySplitOperation
} from '../../../src/gpu-spatial-analysis/trajectory-cleaning/cpu-trajectory-split-operation';
import {
  GPUTrajectorySplitOperation,
  type TrajectorySplitOperationProps
} from '../../../src/gpu-spatial-analysis/trajectory-cleaning/gpu-trajectory-split-operation';
import {getGPUTrajectorySplitParameterValues} from '../../../src/gpu-spatial-analysis/trajectory-cleaning/gpu-trajectory-split';

const INPUT = {
  positions: Float32Array.from({length: 20}, (_, component) => component),
  timestamps: new Float32Array([0, 1, 3, 4, 5, 0, 1, 2, 10, 11]),
  trackOffsets: new Uint32Array([0, 5, 10]),
  parameters: getGPUTrajectorySplitParameterValues({maximumGap: 2})
};

it('matches hand-partitioned gap, stop and overlapping value-change runs on two tracks', () => {
  for (const [mode, extra, expectedRows, expectedOffsets, expectedTracks] of [
    ['gap', {}, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 5, 8, 10], [0, 1, 1]],
    [
      'stop',
      {
        staypointIds: new Uint32Array([
          0xffffffff, 0xffffffff, 0, 0xffffffff, 0xffffffff, 0xffffffff, 1, 0xffffffff, 0xffffffff,
          0xffffffff
        ])
      },
      [0, 1, 3, 4, 7, 8, 9],
      [0, 2, 4, 7],
      [0, 0, 1]
    ],
    [
      'value-change',
      {values: new Uint32Array([1, 1, 2, 2, 3, 9, 8, 8, 8, 8])},
      [0, 1, 2, 2, 3, 4, 5, 6, 6, 7, 8, 9],
      [0, 3, 6, 8, 12],
      [0, 0, 1, 1]
    ]
  ] as const) {
    const result = computeTrajectorySplitOnCPU(
      {...INPUT, ...extra},
      {mode, segmentCapacity: 5, rowCapacity: 14}
    );
    expect(Array.from(result.rowIds)).toEqual([
      ...expectedRows,
      ...new Array(14 - expectedRows.length).fill(0)
    ]);
    expect(Array.from(result.segmentOffsets)).toEqual([
      ...expectedOffsets,
      ...new Array(6 - expectedOffsets.length).fill(0xffffffff)
    ]);
    expect(Array.from(result.segmentTracks)).toEqual([
      ...expectedTracks,
      ...new Array(5 - expectedTracks.length).fill(0xffffffff)
    ]);
    expect(result.count[0]).toBe(expectedTracks.length);
    expect(result.requiredCount[0]).toBe(expectedTracks.length);
    expect(result.overflow[0]).toBe(0);
    expect(Array.from(result.outputPositions)).toEqual([
      ...expectedRows.flatMap(row => [2 * row, 2 * row + 1]),
      ...new Array((14 - expectedRows.length) * 2).fill(0)
    ]);
    expect(Array.from(result.outputTimestamps)).toEqual([
      ...expectedRows.map(row => INPUT.timestamps[row]),
      ...new Array(14 - expectedRows.length).fill(0)
    ]);
  }
});

it('preserves the independent row/segment overflow layout, even its nonmonotonic terminal offset', () => {
  const result = computeTrajectorySplitOnCPU(INPUT, {
    mode: 'gap',
    segmentCapacity: 2,
    rowCapacity: 3
  });
  // GPU writes start 5 for the second segment even though only three rows fit, then terminal 3.
  expect(Array.from(result.segmentOffsets)).toEqual([0, 5, 3]);
  expect(Array.from(result.segmentTracks)).toEqual([0, 1]);
  expect(Array.from(result.rowIds)).toEqual([0, 1, 2]);
  expect(Array.from(result.count)).toEqual([2]);
  expect(Array.from(result.requiredCount)).toEqual([3]);
  expect(Array.from(result.overflow)).toEqual([1]);
});

it('subtracts Int64 words before conversion and copies both words unchanged', () => {
  const timestamps = new Uint32Array(8);
  const origin = (1n << 60n) - 1n;
  [0n, 2n, 5n, 7n].forEach((delta, row) => {
    const timestamp = origin + delta;
    timestamps[row * 2] = Number(timestamp & 0xffffffffn);
    timestamps[row * 2 + 1] = Number(timestamp >> 32n);
  });
  const result = computeTrajectorySplitOnCPU(
    {...INPUT, positions: new Float32Array(8), timestamps, trackOffsets: new Uint32Array([0, 99])},
    {mode: 'gap', segmentCapacity: 2, rowCapacity: 4}
  );
  expect(Array.from(result.segmentOffsets)).toEqual([0, 2, 4]);
  expect(Array.from(result.outputTimestamps)).toEqual(Array.from(timestamps));
});

it('drops singleton runs and rejects inconsistent input arrays', () => {
  const result = computeTrajectorySplitOnCPU(
    {...INPUT, trackOffsets: new Uint32Array([0, 0, 1])},
    {mode: 'gap', segmentCapacity: 1, rowCapacity: 1}
  );
  expect(Array.from(result.segmentOffsets)).toEqual([0, 0xffffffff]);
  expect(result.requiredCount[0]).toBe(0);
  for (const [argument, replacement] of [
    ['positions', new Float32Array(3)],
    ['timestamps', new Float32Array(1)],
    ['trackOffsets', new Uint32Array([4, 2])],
    ['parameters', new Float32Array(3)],
    ['values', new Uint32Array(1)],
    ['staypointIds', new Uint32Array(1)]
  ] as const) {
    expect(() =>
      computeTrajectorySplitOnCPU(
        {...INPUT, [argument]: replacement},
        {mode: 'gap', segmentCapacity: 1, rowCapacity: 1}
      )
    ).toThrow(new RegExp(argument));
  }
});

it('validates construction in both classes; contributor-specific GPU checks remain at lowering', () => {
  const props: TrajectorySplitOperationProps = {
    id: 'split-test',
    mode: 'gap',
    positions: 'positions',
    timestamps: 'timestamps',
    trackOffsets: 'trackOffsets',
    parameters: 'parameters',
    rowCount: 10,
    segmentCapacity: 3,
    rowCapacity: 10,
    output: {
      segmentOffsets: 'segmentOffsets',
      segmentTracks: 'segmentTracks',
      rowIds: 'rowIds',
      status: {count: 'count', overflow: 'overflow'}
    }
  };
  for (const Operation of [GPUTrajectorySplitOperation, CPUTrajectorySplitOperation]) {
    expect(() => new Operation({...props, segmentCapacity: 0})).toThrow(/^split-test /);
    expect(() => new Operation({...props, rowCapacity: 0})).toThrow(/^split-test /);
    expect(() => new Operation({...props, output: {...props.output, rowIds: 'positions'}})).toThrow(
      /^split-test /
    );
  }
  expect(() => new CPUTrajectorySplitOperation({...props, parameters: undefined})).toThrow(
    /^split-test .*parameters/
  );
  expect(() => new CPUTrajectorySplitOperation({...props, rowCount: -1})).toThrow(
    /^split-test rowCount/
  );
});
