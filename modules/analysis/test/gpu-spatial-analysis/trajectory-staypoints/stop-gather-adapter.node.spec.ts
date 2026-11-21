// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it} from 'vitest';
import {computeTrajectoryStaypointsOnCPU} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/cpu-trajectory-staypoints-operation';
it('exposes padded stop rows that require the published gather mask adapter', () => {
  const result = computeTrajectoryStaypointsOnCPU(
    {
      positions: Float32Array.of(12, 30, 12, 30, 12, 30),
      timestamps: Float32Array.of(100, 102, 104),
      trackOffsets: Uint32Array.of(0, 3),
      parameters: Float32Array.of(1, 1, 0, 1)
    },
    {capacity: 3}
  );
  // One observed stay occupies the compact prefix; the two (0,0) pairs are allocation padding.
  expect(result.count[0]).toBe(1);
  expect(Array.from(result.startRows)).toEqual([0, 0xffffffff, 0xffffffff]);
  expect(Array.from(result.centroids)).toEqual([12, 30, 0, 0, 0, 0]);
  // GPUGather's public contract zero-fills out-of-range indices. Gathering ones therefore
  // supplies exactly one event, without treating the two padded origins as observations.
  const eventMask = result.startRows.map(row => (row < 3 ? 1 : 0));
  expect(Array.from(eventMask)).toEqual([1, 0, 0]);
  expect(eventMask.reduce((sum, value) => sum + value, 0)).toBe(1);
});
