// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Shapely intersection in z=1, inverse gnomonic projection, then signed spherical-triangle
// area about (0,0,1). Independent oracle: .typecheck/r19/scratch/
// cells-trajectories-recipes-spherical-and-datum/arc-scaling.py (Python math + GEOS).
export const ARC_SCALING_CASES = [
  {
    totalVertices: 8,
    intersectionVertices: 4,
    area: 0.37017390823804425,
    boundaryVertex: [-0.039842970246373954, 0.42502785832745915, 0.9043029676870024],
    rotatedBoundaryVertex: [-0.08207590461910481, 0.4189268295459671, 0.9043029676870024]
  },
  {
    totalVertices: 16,
    intersectionVertices: 8,
    area: 0.5828920515285267,
    boundaryVertex: [-0.0271169350317581, 0.4781443369167823, 0.8778625546797709],
    rotatedBoundaryVertex: [-0.07471624611090744, 0.47304843056308765, 0.8778625546797709]
  },
  {
    totalVertices: 32,
    intersectionVertices: 15,
    area: 0.635995888846276,
    boundaryVertex: [-0.0002169045786668385, 0.497935091662533, 0.8672142742387422],
    rotatedBoundaryVertex: [-0.0499263824282636, 0.4954258359171413, 0.8672142742387422]
  },
  {
    totalVertices: 64,
    intersectionVertices: 29,
    area: 0.6491748505523837,
    boundaryVertex: [-0.00016555124052371447, 0.5005980319733485, 0.8656798386107862],
    rotatedBoundaryVertex: [-0.05014113607246554, 0.49808059939749255, 0.8656798386107862]
  },
  {
    totalVertices: 128,
    intersectionVertices: 56,
    area: 0.6524591162002682,
    boundaryVertex: [0.00011990128208737684, 0.5019086474736922, 0.8649206294307076],
    rotatedBoundaryVertex: [-0.04998795284678799, 0.4994131649800352, 0.8649206294307076]
  }
] as const;

/** Strictly convex noncollinear chart vertices; distinct phases avoid coincident edges. */
export function getArcScalingOperand(totalVertices: number, right: boolean): number[][] {
  const count = totalVertices / 2;
  return Array.from({length: count}, (_, row) => {
    const angle = (right ? 0.19 : 0.07) + (2 * Math.PI * row) / count;
    const point = [(right ? 0.15 : -0.15) + 0.6 * Math.cos(angle), 0.6 * Math.sin(angle), 1];
    const length = Math.hypot(...point);
    return point.map(value => value / length);
  });
}

// Grid count changes while the four resident samples and four query rows stay fixed.
// All grids can reference the same sample block; metadata count is independent of sample count.
export const DATUM_SCALING_GRID_COUNTS = [1, 8, 32, 128, 512] as const;
export function getDatumScalingGrids(count: number) {
  return Array.from({length: count}, (_, grid) => ({
    width: 2,
    height: 2,
    sampleOffset: 0,
    origin: [grid + 10, 0] as const,
    spacing: [1, 1] as const
  }));
}
