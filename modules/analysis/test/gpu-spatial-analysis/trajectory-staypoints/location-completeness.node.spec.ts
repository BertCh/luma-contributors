// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it} from 'vitest';
import {computeTrajectoryStaypointsOnCPU} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/cpu-trajectory-staypoints-operation';
import {computeTrajectoryLocationsOnCPU} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/cpu-trajectory-locations-operation';
it('separates location completeness from solver convergence and recovers', () => {
  for (const capacity of [1, 2]) {
    const stays = computeTrajectoryStaypointsOnCPU(
      {
        positions: Float32Array.of(0, 0, 0, 0, 10, 0, 10, 0),
        timestamps: Float32Array.of(0, 1, 0, 1),
        trackOffsets: Uint32Array.of(0, 2, 4),
        parameters: Float32Array.of(1, 1, 0, 1)
      },
      {capacity}
    );
    for (const flag of ['overflow', 'candidateOverflow'] as const) {
      const loss = Uint32Array.of(1);
      const input = {
        centroids: stays.centroids,
        tracks: stays.ids,
        count: stays.count,
        parameters: Float32Array.of(1, 0, 0, 0),
        staypointStatus: {[flag]: loss}
      };
      const missing = computeTrajectoryLocationsOnCPU(input);
      expect(missing.converged[0]).toBe(1);
      expect(missing.complete[0]).toBe(0);
      expect(missing.locationCount[0]).toBe(0);
      expect(Array.from(missing.locationIds)).toEqual(new Array(capacity).fill(0xffffffff));
      loss[0] = 0;
      const recovered = computeTrajectoryLocationsOnCPU(input);
      expect(recovered.complete[0]).toBe(1);
      expect(recovered.locationCount[0]).toBe(capacity);
    }
  }
});
