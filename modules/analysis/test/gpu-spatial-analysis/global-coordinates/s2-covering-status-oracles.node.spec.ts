// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeS2CapCoveringOnCPU} from '../../../src/gpu-spatial-analysis/cell-topology/cpu-s2-cap-covering-operation';
import {computeS2PolygonCoveringOnCPU} from '../../../src/gpu-spatial-analysis/cell-topology/cpu-s2-polygon-covering-operation';

// Independent derivation is repeated in the adjacent GPU spec: the +x point
// cap intersects one root, the globe six; the xyz octant intersects three roots.
// Six visits at level 1 leave intersecting roots/children unresolved, independently
// forcing traversal bit 4 with spare output capacity. These are complete words.
for (const refinement of ['uniform', 'level-synchronous'] as const) {
  it(`CPU cap status matches root-face cardinalities (${refinement})`, () => {
    const result = computeS2CapCoveringOnCPU(
      Float32Array.of(1, 0, 0, 0, 1, 0, 0, Math.PI, 0, 0, 0, 1),
      {level: 0, capacity: 4, mode: 'exterior', refinement}
    );
    expect(Array.from(result.status)).toEqual([0, 2, 1]);
    expect(Array.from(result.counts)).toEqual([1, 1, 4, 6, 0, 0]);
  });
}
for (const refinement of ['serial', 'level-synchronous'] as const) {
  it(`CPU polygon status matches complete, capacity, invalid and unfinished roots (${refinement})`, () => {
    for (const scenario of [
      {maximumLevel: 0, maximumVisitedCells: 32, capacity: 6, end: 3, status: 0},
      {maximumLevel: 0, maximumVisitedCells: 32, capacity: 1, end: 3, status: 2},
      {maximumLevel: 0, maximumVisitedCells: 32, capacity: 6, end: 2, status: 1},
      {maximumLevel: 1, maximumVisitedCells: 6, capacity: 24, end: 3, status: 4},
      // Serial retains four +x children before stopping; synchronous traversal
      // discards the unfinished frontier after classifying the six roots.
      {
        maximumLevel: 1,
        maximumVisitedCells: 6,
        capacity: 1,
        end: 3,
        status: refinement === 'serial' ? 6 : 4,
        counts: refinement === 'serial' ? [1, 4] : [0, 0]
      }
    ]) {
      const result = computeS2PolygonCoveringOnCPU(
        {
          positions: Float32Array.of(1, 0, 0, 0, 1, 0, 0, 0, 1),
          ringOffsets: Uint32Array.of(0, scenario.end)
        },
        {...scenario, refinement}
      );
      expect(Array.from(result.status)).toEqual([scenario.status]);
      if (scenario.counts) expect(Array.from(result.counts)).toEqual(scenario.counts);
    }
  });
}
