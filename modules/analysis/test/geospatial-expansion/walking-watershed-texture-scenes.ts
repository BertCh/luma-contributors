// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {makeGPURasterMetadata} from '../../src/raster-analysis/raster-grid';

/** Input values and independently derived expectations used by the texture resubmission tests. */
export function createWalkingWatershedTextureScene(
  route: 'walking' | 'fifo' | 'minimum-spanning-forest'
) {
  const model = {
    kind: 'grass' as const,
    coefficients: [1, 2, 0, 0] as const,
    downhillSlopeThreshold: -1
  };
  if (route === 'walking')
    return {
      grid: makeGPURasterMetadata({width: 5, height: 1}),
      seeds: [1, 0, 0, 1, 0],
      model,
      // Cost = distance 1 + 2 * rise 2 + mean calibrated friction 10.75 / 12.75.
      frames: [
        {
          elevation: [1, 2, -9, 4, 5],
          elevationValidity: [1, 1, 1, 1, 1],
          friction: [1, 2, 3, 4, 5],
          frictionValidity: [1, 1, 1, 1, 0],
          expected: [0, 15.75, NaN, 0, NaN],
          validity: [1, 1, 0, 1, 0]
        },
        {
          elevation: [2, 3, 6, 7, 8],
          elevationValidity: [1, 0, 1, 1, 1],
          friction: [2, 3, -9, 5, 6],
          frictionValidity: [1, 1, 1, 1, 1],
          expected: [0, NaN, NaN, 0, 17.75],
          validity: [1, 0, 0, 1, 1]
        }
      ]
    };
  return {
    grid: makeGPURasterMetadata({width: 7, height: 1}),
    seeds: [5, 0, 9, 0, 7, 0, 0],
    model,
    // Positive calibration preserves the strict edge ordering on [1,3,8] / [8,3,1].
    // Raw nodata at 3 and masked cells independently prevent traversal elsewhere.
    frames: [
      {
        elevation: [1, 3, 8, -9, 4, 5, 6],
        elevationValidity: [1, 1, 1, 1, 1, 1, 0],
        friction: [1, 2, 3, 4, 5, 6, 7],
        frictionValidity: [1, 1, 1, 1, 1, 1, 0],
        expected: [5, 5, 9, 0, 7, 7, 0],
        validity: [1, 1, 1, 0, 1, 1, 0]
      },
      {
        elevation: [8, 3, 1, 7, 4, 5, 6],
        elevationValidity: [1, 1, 1, 0, 1, 0, 1],
        friction: [2, 3, -9, 5, 6, 7, 8],
        frictionValidity: [1, 1, 1, 1, 1, 1, 1],
        expected: [5, 9, 9, 0, 7, 0, 0],
        validity: [1, 1, 1, 0, 1, 0, 1]
      }
    ]
  };
}
