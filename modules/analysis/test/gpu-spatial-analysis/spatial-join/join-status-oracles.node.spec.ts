// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computePointInPolygonJoinOnCPU} from '../../../src/gpu-spatial-analysis/spatial-join/cpu-point-in-polygon-join-operation';

it('ordinary point geometry counts one uncertain ring then repairs it', () => {
  // The disjoint envelopes [0,2]x[0,2] and [2,4]x[0,2] each contain one point.
  // A NaN shell vertex invalidates only the first candidate; after repair both
  // points are strictly inside their squares. These literals also drive the zonal GPU case.
  for (const scenario of [
    {corner: NaN, counts: [0, 1], uncertain: [1]},
    {corner: 2, counts: [1, 1], uncertain: [0]}
  ]) {
    const result = computePointInPolygonJoinOnCPU(
      {
        points: {kind: 'points', positions: Float32Array.from([1, 1, 3, 1])},
        polygons: {
          kind: 'polygons',
          positions: Float32Array.from([
            0,
            0,
            2,
            0,
            scenario.corner,
            2,
            0,
            2,
            2,
            0,
            4,
            0,
            4,
            2,
            2,
            2
          ]),
          featureOffsets: Uint32Array.from([0, 1, 2]),
          polygonOffsets: Uint32Array.from([0, 1, 2]),
          ringOffsets: Uint32Array.from([0, 4, 8])
        }
      },
      {candidateCapacity: 4, matchCapacity: 2, boundaryRule: 'closed'}
    );
    expect(Array.from(result.polygonAssignmentCounts)).toEqual(scenario.counts);
    expect(Array.from(result.uncertainCount)).toEqual(scenario.uncertain);
  }
});
