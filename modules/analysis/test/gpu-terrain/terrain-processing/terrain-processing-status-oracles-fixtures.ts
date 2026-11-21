// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUTerrainBreachOperation} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-breach-operation';
import {CPUTerrainBreachOperation} from '../../../src/gpu-terrain/terrain-processing/cpu-terrain-breach-operation';
import type {TerrainStatusOracle} from '../terrain-status-oracles-test-utils';

export function getTerrainProcessingStatusOracles(): TerrainStatusOracle[] {
  const breachProps = {
    grid: {
      width: 3,
      height: 1,
      affine: [1, 0, 0, 0, 1, 0] as const,
      pixelInterpretation: 'area' as const
    },
    elevation: 'elevation',
    pitCells: 'pits',
    pitCount: 1,
    maximumSteps: 2,
    maximumCutDepth: 4,
    elevationDecrement: 0.5,
    output: {
      elevation: 'edited',
      cutDepth: 'depth',
      modificationMask: 'mask',
      pathCells: 'paths',
      pathLengths: 'lengths',
      pitStatus: 'pitStatus',
      unresolvedCount: 'unresolvedCount'
    }
  };
  // [2,5,0] has the unique outlet path 0→1→2: cut 5 down to 1.5, then reach 0.
  // [2,5,2] has no original cell below the descending target levels 1.5/1, so no
  // bounded path can reach an outlet. Pit row 3 is outside this three-cell grid.
  // The final success frame must clear prior unresolved counts and invalid pit status.
  return [
    {
      name: 'breach-success-no-outlet-invalid-and-replay',
      gpu: () => new GPUTerrainBreachOperation(breachProps),
      cpu: () => new CPUTerrainBreachOperation(breachProps),
      frames: [
        {elevation: [2, 5, 0], pit: 0, status: [1], count: [0]},
        {elevation: [2, 5, 2], pit: 0, status: [2], count: [1]},
        {elevation: [2, 5, 0], pit: 3, status: [0], count: [1]},
        {elevation: [2, 5, 0], pit: 0, status: [1], count: [0]}
      ].map(frame => ({
        inputs: {
          elevation: {format: 'float32', values: new Float32Array(frame.elevation)},
          pits: {format: 'uint32', values: new Uint32Array([frame.pit])}
        },
        expected: {pitStatus: frame.status, unresolvedCount: frame.count}
      }))
    },
    {
      name: 'breach-counts-two-simultaneous-unresolved-pits-and-clears',
      gpu: () => new GPUTerrainBreachOperation({...breachProps, pitCount: 2}),
      cpu: () => new CPUTerrainBreachOperation({...breachProps, pitCount: 2}),
      // Both end pits have height 2. No cell is lower in [2,5,2], so neither
      // search can escape: count is 2, not Boolean 1. With the middle at 0,
      // each pit has its own one-step outlet and both statuses become success.
      frames: [true, false, true, false].map(blocked => ({
        inputs: {
          elevation: {format: 'float32', values: new Float32Array(blocked ? [2, 5, 2] : [2, 0, 2])},
          pits: {format: 'uint32', values: new Uint32Array([0, 2])}
        },
        expected: {pitStatus: blocked ? [2, 2] : [1, 1], unresolvedCount: [blocked ? 2 : 0]}
      }))
    }
  ];
}
