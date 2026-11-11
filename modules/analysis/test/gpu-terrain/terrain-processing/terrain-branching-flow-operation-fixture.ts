// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {TerrainBranchingFlowOperationProps} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-branching-flow-operation';

export const BRANCHING_GRID = {
  width: 3,
  height: 2,
  affine: [2, 0, 0, 0, -4, 0] as const,
  pixelInterpretation: 'area' as const
};

/** Two weighted branches merge at cell 4; cells 2 and 5 are separate outlets. */
export function makeBranchingInput() {
  const fractions = new Float32Array(48);
  const edgeLengths = new Float32Array(48).fill(NaN);
  for (const [cell, direction, fraction, length] of [
    [0, 0, 0.25, 2],
    [0, 2, 0.75, 4],
    [1, 0, 0.5, 3],
    [1, 2, 0.5, 5],
    [3, 0, 1, 7],
    [4, 0, 1, 11]
  ]) {
    fractions[cell * 8 + direction] = fraction;
    edgeLengths[cell * 8 + direction] = length;
  }
  return {fractions, edgeLengths, values: Float32Array.of(2, 3, 5, 7, 11, 13)};
}

export function makeBranchingProps(
  mode: TerrainBranchingFlowOperationProps['mode']
): TerrainBranchingFlowOperationProps {
  return {
    id: 'branching-test',
    grid: BRANCHING_GRID,
    mode,
    fractions: 'fractions',
    values: 'values',
    edgeLengths: 'edgeLengths',
    maximumIterations: 8,
    maximumWalkLength: 2,
    output: {values: 'result', status: 'status', unresolvedCount: 'unresolvedCount'}
  };
}
