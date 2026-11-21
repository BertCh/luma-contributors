// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeStreamlinesOnCPU} from '../../../src/raster-analysis/flow-texture/cpu-streamlines-operation';
import {
  getGPUStreamlinesParameterValues,
  getGPUStreamlinesUint32ParameterValues
} from '../../../src/raster-analysis/flow-texture/streamlines-parameters';
const CASES = [
  {
    name: 'pending',
    maximumIterations: 1,
    pointCapacity: 3,
    velocity: 1,
    expected: [1, 1, 3, 0, 1, 1]
  },
  {
    name: 'complete',
    maximumIterations: 2,
    pointCapacity: 3,
    velocity: 1,
    expected: [1, 1, 3, 0, 0, 0]
  },
  {
    name: 'short-points',
    maximumIterations: 2,
    pointCapacity: 2,
    velocity: 1,
    expected: [0, 1, 0, 1, 0, 0]
  },
  {name: 'empty', maximumIterations: 2, pointCapacity: 3, velocity: 0, expected: [0, 0, 0, 0, 0, 0]}
];
const SETTINGS = {
  fieldPlacement: {origin: [0, 0], cellSize: [1, 1]} as const,
  gridPlacement: {origin: [0, 0], cellSize: [1, 1]} as const,
  stepLength: 0,
  seed: 1,
  minimumPoints: 3
};
// Closed form: four seeds compete for the single occupancy cell. A zero integration step
// gives exactly three identical points per valid seed. The first pruning round accepts only
// the priority winner and leaves other valid seeds undecided; round two rejects those occupied
// starts. Two point slots cannot hold the accepted three-point line. A zero field rejects all.

for (const scene of CASES) {
  it(`streamlines have independent status/count literals for ${scene.name}`, () => {
    const result = computeStreamlinesOnCPU(
      {
        velocities: Float32Array.of(scene.velocity, 0),
        parameters: getGPUStreamlinesParameterValues(SETTINGS),
        uint32Parameters: getGPUStreamlinesUint32ParameterValues(SETTINGS)
      },
      {
        fieldWidth: 1,
        fieldHeight: 1,
        gridWidth: 1,
        gridHeight: 1,
        seedColumns: 4,
        seedRows: 1,
        stepsPerDirection: 1,
        lineCapacity: 4,
        pointCapacity: scene.pointCapacity,
        maximumIterations: scene.maximumIterations
      }
    );
    expect([
      result.count[0],
      result.requiredCount[0],
      result.positionCount[0],
      result.overflow[0],
      result.candidateOverflow[0],
      result.iterationLimitReached[0]
    ]).toEqual(scene.expected);
    // OUT-09: increasing capacity adds empty CSR intervals without changing the KI status oracle.
    expect(Array.from(result.lineOffsets)).toEqual([
      0,
      scene.expected[2],
      scene.expected[2],
      scene.expected[2],
      scene.expected[2]
    ]);
  });
}
