// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeTrajectoryThinningOnCPU} from '../../../src/gpu-spatial-analysis/trajectory-cleaning/cpu-trajectory-thinning-operation';

it('TRJ-108 distinguishes the bounded 4096-row chain from exact thinning', () => {
  const positions = new Float32Array(12000);
  positions[11998] = 10;
  const input = {
    positions,
    timestamps: new Float32Array(6000),
    trackOffsets: Uint32Array.of(0, 6000),
    parameters: Float32Array.of(5, 0, 0, 0)
  };
  for (const maximumSearchRows of [undefined, 4096]) {
    const result = computeTrajectoryThinningOnCPU(input, {
      mode: 'minimum-distance',
      capacity: 4,
      maximumSearchRows
    });
    // Exact: only the last point reaches distance 5 from the origin. Bounded:
    // after scanning 1..4096, retain 4097 before finding the final displaced fix.
    const expected = maximumSearchRows === undefined ? [0, 5999] : [0, 4097, 5999];
    expect(Array.from(result.sourceRows).slice(0, result.count[0])).toEqual(expected);
    expect(result.count[0]).toBe(expected.length);
  }
});
