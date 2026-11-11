// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {S2_POSITION_TO_IJ, S2_POSITION_TO_ORIENTATION} from '../cell-indexing/cell-index-wgsl';

/** S2 keys allocate sixty Hilbert bits and therefore at most thirty levels. */
export const CPU_S2_COVERING_MAXIMUM_LEVEL = 30;

/** Unit-vector arithmetic shared by the two covering twins. @internal */
export type CPUCoveringVector = readonly [number, number, number];

/** Dot product in float64. @internal */
export function getCoveringDot(first: CPUCoveringVector, second: CPUCoveringVector): number {
  return first[0] * second[0] + first[1] * second[1] + first[2] * second[2];
}

/** Cross product in float64. @internal */
export function getCoveringCross(
  first: CPUCoveringVector,
  second: CPUCoveringVector
): CPUCoveringVector {
  return [
    first[1] * second[2] - first[2] * second[1],
    first[2] * second[0] - first[0] * second[2],
    first[0] * second[1] - first[1] * second[0]
  ];
}

/** Normalizes a nonzero vector. @internal */
export function normalizeCoveringVector(vector: CPUCoveringVector): CPUCoveringVector {
  const length = Math.hypot(...vector);
  return [vector[0] / length, vector[1] / length, vector[2] / length];
}

/** Antipodal vector. @internal */
export function negateCoveringVector(vector: CPUCoveringVector): CPUCoveringVector {
  return [-vector[0], -vector[1], -vector[2]];
}

/** Spherical quadrilateral and inward edge normals. @internal */
export type CPUCoveringCellGeometry = {
  vertices: CPUCoveringVector[];
  normals: CPUCoveringVector[];
};

/**
 * Corners in the contributor's (00, 10, 11, 01) order. Edge normals come directly from the
 * gnomonic bounds, avoiding cancellation between almost coincident vertices at deep levels.
 * @internal
 */
export function getCoveringCellGeometry(cell: bigint, level: number): CPUCoveringCellGeometry {
  const face = Number(cell >> 61n);
  let orientation = face & 1;
  let firstCoordinate = 0;
  let secondCoordinate = 0;
  for (let step = 0; step < level; step++) {
    const position = Number((cell >> BigInt(59 - 2 * step)) & 3n);
    const coordinates = S2_POSITION_TO_IJ[orientation][position];
    firstCoordinate = firstCoordinate * 2 + (coordinates >> 1);
    secondCoordinate = secondCoordinate * 2 + (coordinates & 1);
    orientation ^= S2_POSITION_TO_ORIENTATION[position];
  }
  const convertSurface = (surface: number) =>
    surface >= 0.5 ? (4 * surface * surface - 1) / 3 : (1 - 4 * (1 - surface) ** 2) / 3;
  const lowerFirst = convertSurface(firstCoordinate / 2 ** level);
  const upperFirst = convertSurface((firstCoordinate + 1) / 2 ** level);
  const lowerSecond = convertSurface(secondCoordinate / 2 ** level);
  const upperSecond = convertSurface((secondCoordinate + 1) / 2 ** level);
  const rotateVector = ([first, second, third]: CPUCoveringVector): CPUCoveringVector => {
    switch (face) {
      case 0:
        return [first, second, third];
      case 1:
        return [-second, first, third];
      case 2:
        return [-second, -third, first];
      case 3:
        return [-first, -third, -second];
      case 4:
        return [third, -first, -second];
      default:
        return [third, second, -first];
    }
  };
  const vertices: CPUCoveringVector[] = [
    [1, lowerFirst, lowerSecond],
    [1, upperFirst, lowerSecond],
    [1, upperFirst, upperSecond],
    [1, lowerFirst, upperSecond]
  ];
  const normals: CPUCoveringVector[] = [
    [-lowerSecond, 0, 1],
    [upperFirst, -1, 0],
    [upperSecond, 0, -1],
    [-lowerFirst, 1, 0]
  ];
  return {
    vertices: vertices.map(vector => normalizeCoveringVector(rotateVector(vector))),
    normals: normals.map(vector => normalizeCoveringVector(rotateVector(vector)))
  };
}

/** The kernel's minor-arc tolerance predicates, optionally reusing its edge normal. @internal */
export function isPointOnCoveringArc(
  first: CPUCoveringVector,
  second: CPUCoveringVector,
  point: CPUCoveringVector,
  tolerance: number,
  normal = normalizeCoveringVector(getCoveringCross(first, second))
): boolean {
  return (
    Math.abs(getCoveringDot(normal, point)) <= tolerance &&
    getCoveringDot(getCoveringCross(first, point), normal) >= -tolerance &&
    getCoveringDot(getCoveringCross(point, second), normal) >= -tolerance
  );
}

/** Distance to the spherical quadrilateral, using the GPU's double-cross projection. @internal */
export function getCoveringCellDistance(
  geometry: CPUCoveringCellGeometry,
  point: CPUCoveringVector
): number {
  const {vertices, normals} = geometry;
  const getAngle = (first: CPUCoveringVector, second: CPUCoveringVector) =>
    Math.atan2(Math.hypot(...getCoveringCross(first, second)), getCoveringDot(first, second));
  let minimum = Math.PI;
  let inside = true;
  for (let edge = 0; edge < 4; edge++) {
    const first = vertices[edge];
    const second = vertices[(edge + 1) % 4];
    const normal = normals[edge];
    inside &&= getCoveringDot(normal, point) >= 0;
    minimum = Math.min(minimum, getAngle(point, first));
    const projected = getCoveringCross(normal, getCoveringCross(point, normal));
    if (Math.hypot(...projected) > 1e-10) {
      const nearest = normalizeCoveringVector(projected);
      if (
        getCoveringDot(getCoveringCross(first, nearest), normal) >= 0 &&
        getCoveringDot(getCoveringCross(nearest, second), normal) >= 0
      )
        minimum = Math.min(minimum, getAngle(point, nearest));
    }
  }
  return inside ? 0 : minimum;
}
