// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it} from 'vitest';
import {computeTrajectoryStaypointsOnCPU} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/cpu-trajectory-staypoints-operation';

it('breaks observed dwell at a missing fix without reporting capacity loss', () => {
  for (const masked of [false, true]) {
    const result = computeTrajectoryStaypointsOnCPU(
      {
        positions: Float32Array.of(0, 0, masked ? 0 : NaN, masked ? 0 : NaN, 1000, 0),
        timestamps: Float32Array.of(0, 300000, 600000),
        trackOffsets: Uint32Array.of(0, 3),
        parameters: Float32Array.of(100, 300000, 900000, 1),
        ...(masked ? {mask: Uint32Array.of(1, 0, 1)} : {})
      },
      {capacity: 3}
    );
    // Each contiguous observed run has one fix, hence zero observed duration.
    expect(result.count[0]).toBe(0);
    expect(result.missingFixCount[0]).toBe(1);
    expect(result.overflow[0]).toBe(0);
    expect(result.candidateOverflow[0]).toBe(0);
  }
});
it('finishes observed dwell at the last observation and restarts after the missing row', () => {
  const result = computeTrajectoryStaypointsOnCPU(
    {
      positions: Float32Array.of(0, 0, 0, 0, NaN, NaN, 10, 0, 10, 0),
      timestamps: Float32Array.of(0, 5, 100, 200, 205),
      trackOffsets: Uint32Array.of(0, 5),
      parameters: Float32Array.of(1, 5, 0, 1)
    },
    {capacity: 2}
  );
  // Observed runs [0,1] and [3,4] each last five time units; the dropout contributes none.
  expect(Array.from(result.durations)).toEqual([5, 5]);
  expect(Array.from(result.startRows)).toEqual([0, 3]);
  expect(Array.from(result.endRows)).toEqual([2, 5]);
  expect(Array.from(result.finishRows)).toEqual([1, 4]);
});
