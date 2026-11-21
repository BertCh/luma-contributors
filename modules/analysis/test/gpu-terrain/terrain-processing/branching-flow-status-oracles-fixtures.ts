// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {TerrainBranchingFlowOperationProps} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-branching-flow-operation';

export const BRANCHING_STATUS_GRID = {
  width: 3,
  height: 1,
  affine: [1, 0, 0, 0, 1, 0] as const,
  pixelInterpretation: 'area' as const
};

// A two-cell cycle cannot publish either dependency; disconnected cell 2 is an outlet.
// Removing both edges makes three independent resolved cells. Masking cell 2 then gives
// local-invalid code 0 (not dependency-failure code 2). unresolvedCount counts both 0 and 2.
// None of these expectations depends on a positive iteration cap being exhausted.
export const BRANCHING_STATUS_SCENES = [
  {
    name: 'cycle-and-outlet',
    cycle: true,
    validity: [1, 1, 1],
    expectedStatus: [2, 2, 1],
    expectedCount: [2]
  },
  {
    name: 'three-outlets',
    cycle: false,
    validity: [1, 1, 1],
    expectedStatus: [1, 1, 1],
    expectedCount: [0]
  },
  {
    name: 'masked-outlet',
    cycle: false,
    validity: [1, 1, 0],
    expectedStatus: [1, 1, 0],
    expectedCount: [1]
  }
] as const;

export function createBranchingStatusFractions(cycle: boolean): Float32Array {
  const fractions = new Float32Array(24);
  if (cycle) {
    fractions[0] = 1; // cell 0 east to cell 1
    fractions[12] = 1; // cell 1 west to cell 0
  }
  return fractions;
}

export function createBranchingStatusProps(): TerrainBranchingFlowOperationProps {
  return {
    grid: BRANCHING_STATUS_GRID,
    mode: 'transport',
    maximumIterations: 4,
    fractions: 'fractions',
    values: 'values',
    validity: 'validity',
    output: {values: 'result', status: 'status', unresolvedCount: 'unresolvedCount'}
  };
}
