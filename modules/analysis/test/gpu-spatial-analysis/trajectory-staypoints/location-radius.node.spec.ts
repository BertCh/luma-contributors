// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it} from 'vitest';
import {computeTrajectoryLocationsOnCPU} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/cpu-trajectory-locations-operation';
it('uses inclusive stored-coordinate radius membership before rounding distance', () => {
  for (const offset of [0, 2 ** -13]) {
    const result = computeTrajectoryLocationsOnCPU(
      {
        centroids: Float32Array.of(0, 0, 1, offset),
        tracks: Uint32Array.of(0, 0),
        count: Uint32Array.of(2),
        parameters: Float32Array.of(1, 0, 0, 0)
      },
      {maximumIterations: 2}
    );
    // Squared distance is exactly 1 or 1 + 2^-26, respectively.
    expect(Array.from(result.locationIds)).toEqual(offset === 0 ? [0, 0] : [0, 1]);
    expect(result.locationCount[0]).toBe(offset === 0 ? 1 : 2);
    expect(result.converged[0]).toBe(1);
  }
});
it('B5 compares exact stored squared distances before subtracting or taking a square root', () => {
  for (const maximumIterations of [undefined, 0, 2])
    for (const centroids of [
      Float32Array.of(-(2 ** -25), 0, 1, 0),
      Float32Array.of(0, 0, 1, 2 ** -26)
    ]) {
      // Exact squared distances exceed one by 2^-24+2^-50, or by 2^-52, respectively.
      const result = computeTrajectoryLocationsOnCPU(
        {
          centroids,
          tracks: Uint32Array.of(0, 0),
          count: Uint32Array.of(2),
          parameters: Float32Array.of(1, 0, 0, 0)
        },
        {maximumIterations}
      );
      expect(Array.from(result.locationIds)).toEqual([0, 1]);
      expect(result.converged[0]).toBe(1);
    }
});
