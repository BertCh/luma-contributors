// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeTrajectoryThinningOnCPU} from '../../../src/gpu-spatial-analysis/trajectory-cleaning/cpu-trajectory-thinning-operation';

// Hand-derived rows: tolerance 2 -> [0,2,3]; tolerance 10, budget 1 -> [0,2,3]
// with rows 2/3 beyond anchor 0's search; tolerance 10, budget 4 -> [0,3].
// Same literal statuses as the headless spec, never derived from another run.
// These fixtures use a four-row resident allocation. Empty offsets [0,0] own no rows,
// so count/required are zero and neither capacity nor a search budget can be exceeded.
for (const scenario of [
  {name: 'exact fit', capacity: 3, tolerance: 2, budget: 4, empty: false, status: [3, 3, 0, 0]},
  {name: 'output loss', capacity: 2, tolerance: 2, budget: 4, empty: false, status: [2, 3, 1, 0]},
  {name: 'search loss', capacity: 3, tolerance: 10, budget: 1, empty: false, status: [3, 3, 0, 1]},
  {
    name: 'complete search',
    capacity: 3,
    tolerance: 10,
    budget: 4,
    empty: false,
    status: [2, 2, 0, 0]
  },
  {name: 'empty track', capacity: 3, tolerance: 2, budget: 4, empty: true, status: [0, 0, 0, 0]}
]) {
  it(`CPU thinning publishes independently derived status: ${scenario.name}`, () => {
    const result = computeTrajectoryThinningOnCPU(
      {
        storage: 'resident-capacity',
        positions: new Float32Array(8),
        timestamps: new Float32Array([0, 1, 2, 3]),
        trackOffsets: new Uint32Array(scenario.empty ? [0, 0] : [0, 4]),
        parameters: new Float32Array([scenario.tolerance, 0, 0, 0])
      },
      {mode: 'minimum-time', capacity: scenario.capacity, maximumSearchRows: scenario.budget}
    );
    expect([
      ...result.count,
      ...result.requiredCount,
      ...result.overflow,
      ...result.candidateOverflow
    ]).toEqual(scenario.status);
  });
}
