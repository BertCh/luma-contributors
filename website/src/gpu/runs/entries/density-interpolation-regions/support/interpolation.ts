// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Inputs shared by the interpolation runs: scattered samples and a query grid.

import {createUniformPoints, type PointsInput} from '../../../inputs';
import {createRunGrid, type RunGrid} from './grid';

export type InterpolationInputs = {
  grid: RunGrid;
  samples: PointsInput;
  /** Observation rows [value, errorVariance, 0, 1] of the samples. */
  observations: Float32Array;
};

export function createInterpolationInputs(
  random: () => number,
  world: {width: number; height: number},
  rows = 40,
  sampleCount = 60
): InterpolationInputs {
  const samples = createUniformPoints(random, world, sampleCount, 0.05);
  const observations = new Float32Array(sampleCount * 4);
  samples.values.forEach((value, index) => {
    observations.set([value, 0.01, 0, 1], index * 4);
  });
  return {grid: createRunGrid(world, rows), samples, observations};
}
