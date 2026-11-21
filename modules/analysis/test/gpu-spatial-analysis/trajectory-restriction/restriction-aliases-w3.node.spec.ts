// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import type {CPUTrajectoryRestrictionGraphInput} from '../../../src/gpu-spatial-analysis/index';
import {
  computeTrajectoryRestrictionOnCPU,
  type CPUTrajectoryRestrictionInput
} from '../../../src/gpu-spatial-analysis/trajectory-restriction/cpu-trajectory-restriction-operation';
import {createRectangleScene} from './trajectory-restriction-operation-cases';
import {
  GPUTrajectoryRestrictionOperation,
  type TrajectoryRestrictionOperationProps
} from '../../../src/gpu-spatial-analysis/trajectory-restriction/gpu-trajectory-restriction-operation';
import {CPUTrajectoryRestrictionOperation} from '../../../src/gpu-spatial-analysis/trajectory-restriction/cpu-trajectory-restriction-operation';

it('rejects each removed restriction operation key and names its replacement', () => {
  const props: TrajectoryRestrictionOperationProps = {
    positions: 'positions',
    timestamps: 'timestamps',
    trackOffsets: 'trackOffsets',
    parameters: 'parameters',
    rowCount: 2,
    trackCount: 1,
    output: {
      intervals: 'intervals',
      positions: 'clipped',
      status: 'status',
      relationships: 'relationships'
    }
  };
  for (const Operation of [GPUTrajectoryRestrictionOperation, CPUTrajectoryRestrictionOperation]) {
    expect(() => new Operation(props)).not.toThrow();
    for (const [removed, replacement] of [
      ['times', 'timestamps'],
      ['offsets', 'trackOffsets']
    ]) {
      expect(() => new Operation({...props, [removed]: 'legacy'})).toThrow(
        new RegExp(`${removed} was removed.*${replacement}`)
      );
      expect(
        () =>
          new Operation({...props, [replacement]: undefined} as TrajectoryRestrictionOperationProps)
      ).toThrow(/timestamps and trackOffsets are required/);
    }
  }
});

it('clips the canonical CPU input to the hand-computed box interval', () => {
  const {input, options} = createRectangleScene();
  const graphInput: CPUTrajectoryRestrictionGraphInput = input;
  const result = computeTrajectoryRestrictionOnCPU(graphInput, options);
  // x=-2..2 in four seconds intersects [-1,1] at t=[1,3].
  expect(Array.from(result.intervals.slice(0, 2))).toEqual([1, 3]);
});

it('rejects each removed CPU restriction input key and names its replacement', () => {
  const {input, options} = createRectangleScene();
  for (const [removed, replacement] of [
    ['times', 'timestamps'],
    ['offsets', 'trackOffsets']
  ] as const) {
    expect(() =>
      computeTrajectoryRestrictionOnCPU({...input, [removed]: input[replacement]}, options)
    ).toThrow(new RegExp(`${removed} was removed.*${replacement}`));
    expect(() =>
      computeTrajectoryRestrictionOnCPU(
        {...input, [replacement]: undefined} as CPUTrajectoryRestrictionInput,
        options
      )
    ).toThrow(/timestamps and trackOffsets are required/);
  }
});
