// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Four corners of a 3 by 4 rectangle: centre (1.5, 2), every centre distance 2.5,
// nearest distance 3, third-nearest distance 5. Population RMS distance is 2.5,
// not the sample-divisor result sqrt(25 / 3).
export const RECTANGLE_POSITIONS = new Float32Array([0, 0, 3, 0, 0, 4, 3, 4]);
export const RECTANGLE_STATISTICS = {
  pointCount: 4,
  invalidCount: 0,
  sampleSize: 4,
  sampled: false,
  coordinateModel: 'planar',
  unit: 'coordinate-unit',
  extent: [0, 0, 3, 4],
  extentWidth: 3,
  extentHeight: 4,
  extentDiagonal: 5,
  center: [1.5, 2],
  standardDistance: 2.5,
  medianDistanceFromCenter: 2.5,
  duplicateLocationCount: 0,
  nearestNeighbor: {
    minimum: 3,
    lowerQuartile: 3,
    median: 3,
    upperQuartile: 3,
    mean: 3,
    maximum: 3,
    upperFence: 3,
    aboveUpperFenceCount: 0,
    nonZeroMedian: 3
  },
  kDistance: {k: 3, distances: [5, 5, 5, 5]}
};
