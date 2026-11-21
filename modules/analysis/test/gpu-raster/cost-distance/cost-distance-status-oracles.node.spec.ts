// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeCostDistanceOnCPU} from '../../../src/raster-analysis/cost-distance/cpu-cost-distance-operation';
import {getGPUCostDistanceParameterValues} from '../../../src/raster-analysis/cost-distance/gpu-cost-distance';

for (const maximumIterations of [1, 2]) {
  it(`cost-distance derives stopping words from a single edge (budget=${maximumIterations})`, () => {
    const grid = {
      width: 2,
      height: 1,
      affine: [1, 0, 0, 0, 1, 0] as const,
      pixelInterpretation: 'area' as const
    };
    const result = computeCostDistanceOnCPU(
      {
        friction: {id: 'friction', format: 'float32', values: Float32Array.of(1, 1)},
        sources: Uint32Array.of(0),
        parameters: getGPUCostDistanceParameterValues({}, grid),
        costBreaks: Float32Array.of(0.5, Infinity)
      },
      {grid, maximumIterations, maximumTieIterations: 2}
    );
    // Round one changes the neighbor from infinity to 1; an unchanged second pass certifies
    // the result. Exact costs 0,1 belong to distinct bands, one cell per band.
    expect(Array.from(result.iterationCount)).toEqual([maximumIterations]);
    expect(Array.from(result.converged)).toEqual(maximumIterations === 1 ? [0] : [1]);
    expect(Array.from(result.iterationLimitReached)).toEqual(maximumIterations === 1 ? [1] : [0]);
    expect(Array.from(result.bandCounts)).toEqual(
      maximumIterations === 1 ? [0xfffffffe, 0xfffffffe] : [1, 1]
    );
    expect(Array.from(result.costs)).toEqual(maximumIterations === 1 ? [NaN, NaN] : [0, 1]);
  });
}
