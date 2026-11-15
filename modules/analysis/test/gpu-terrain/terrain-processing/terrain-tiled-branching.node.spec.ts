// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeTerrainBranchingFlowOnCPU} from '../../../src/gpu-terrain/terrain-processing/cpu-terrain-branching-flow-operation';
import reference from '../../oracles/terrain-tiled-branching.json';

for (const scene of reference.cases) {
  it(`matches the NumPy linear-system oracle for ${scene.routing} transport`, () => {
    const count = reference.width * reference.height;
    const result = computeTerrainBranchingFlowOnCPU(
      {fractions: Float32Array.from(scene.fractions), values: Float32Array.from(reference.runoff)},
      {
        grid: {
          width: reference.width,
          height: reference.height,
          affine: [1, 0, 0, 0, 1, 0],
          pixelInterpretation: 'area'
        },
        mode: 'transport'
      }
    );
    expect(Array.from(result.status)).toEqual(Array(count).fill(1));
    expect(Array.from(result.unresolvedCount)).toEqual([0]);
    result.values.forEach((value, cell) => {
      // The CPU twin computes in float64, then rounds once to the float32 output.
      expect(value).toBe(Math.fround(scene.accumulation[cell]));
    });
    expect(result.values[count - 1]).toBeCloseTo(
      reference.runoff.reduce((total, value) => total + value, 0),
      5
    );
  });
}
