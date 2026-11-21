// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid/index';

/** Rectangles cover [0,4] x [0,2]; every source overlaps the target by half its area. */
export function createArealRecipeRectangleFixture() {
  return {
    // North-up rows cover y=[0,2] only when their top edge is y=2.
    grid: makeGPURasterMetadata({width: 4, height: 2, origin: [0, 2]}),
    source: {
      positions: new Float32Array([0, 0, 2, 0, 2, 2, 0, 2, 2, 0, 4, 0, 4, 2, 2, 2]),
      featureOffsets: new Uint32Array([0, 1, 2]),
      polygonOffsets: new Uint32Array([0, 1, 2]),
      ringOffsets: new Uint32Array([0, 4, 8]),
      // Two vertical edges per rectangle, two raster rows, two source rectangles.
      crossingCapacity: 8
    },
    target: {
      positions: new Float32Array([1, 0, 3, 0, 3, 2, 1, 2]),
      featureOffsets: new Uint32Array([0, 1]),
      polygonOffsets: new Uint32Array([0, 1]),
      ringOffsets: new Uint32Array([0, 4]),
      crossingCapacity: 4
    }
  };
}
