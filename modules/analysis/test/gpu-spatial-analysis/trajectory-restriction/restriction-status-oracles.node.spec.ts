// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeTrajectoryRestrictionOnCPU} from '../../../src/gpu-spatial-analysis/trajectory-restriction/cpu-trajectory-restriction-operation';
import {RESTRICTION_STATUS_CASES} from './restriction-status-oracles-fixture';

for (const scenario of [
  {name: 'active singleton pair', active: 1, first: 0, status: [50, 0]},
  {name: 'inactive pair', active: 0, first: 0, status: [0, 0]},
  {name: 'invalid track', active: 1, first: 99, status: [4, 0]}
]) {
  it(`CPU restriction exact pair status: ${scenario.name}`, () => {
    // Two one-fix tracks at t=0 separated by 0.5, distance limit 1:
    // temporal support 32 + full containment 16 + singleton 2 = 50.
    const result = computeTrajectoryRestrictionOnCPU(
      {
        positions: new Float32Array([0, 0, 0, 0.5]),
        timestamps: new Float32Array([0, 0]),
        trackOffsets: new Uint32Array([0, 1, 2]),
        parameters: new Float32Array([0, 4, 1, 0, 0, 0, 0]),
        pairs: {
          first: new Uint32Array([scenario.first]),
          second: new Uint32Array([1]),
          count: new Uint32Array([scenario.active])
        }
      },
      {intervalsPerPair: 2}
    );
    expect(Array.from(result.status)).toEqual(scenario.status);
  });
}

for (const scenario of RESTRICTION_STATUS_CASES) {
  it(`CPU restriction whole status words: ${scenario.name}`, () => {
    const result = computeTrajectoryRestrictionOnCPU(
      {
        positions: Float32Array.from(scenario.positions),
        timestamps: new Float32Array([0, 4]),
        trackOffsets: new Uint32Array([0, 2]),
        parameters: Float32Array.from(scenario.parameters)
      },
      {
        geometryBoundary: scenario.boundary,
        polygonRingVertexCounts: scenario.rings,
        intervalsPerSegment: 2
      }
    );
    expect(Array.from(result.status)).toEqual(scenario.status);
  });
}
