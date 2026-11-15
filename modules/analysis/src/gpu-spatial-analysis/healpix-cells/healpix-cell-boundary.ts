// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {HEALPIX_MAXIMUM_ORDER} from './healpix-index-wgsl';

const FACE_RING = [2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4];
const FACE_LONGITUDE = [1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7];

/**
 * Samples a spherical HEALPix nested cell's four curved edges in longitude/latitude degrees.
 * The inverse face projection uses the same nested x/y bit layout as GPUHealpixPointToCell.
 * This is display geometry, not an ellipsoidal or OGC ZIRS boundary. Longitudes are unwrapped
 * onto a continuous branch, which can extend beyond [-180, 180] at the antimeridian.
 * A closed ring has 4 * edgeSegments + 1 vertices; polar vertices use the face meridian.
 */
export function getHealpixCellBoundary(
  cell: bigint,
  order: number,
  edgeSegments = 8
): [number, number][] {
  const operation = 'getHealpixCellBoundary';
  validateCapacity(operation, 'order', order, 0, HEALPIX_MAXIMUM_ORDER);
  validateCapacity(operation, 'edgeSegments', edgeSegments, 1, 64);
  validateRequirement(
    operation,
    cell >= 0n && cell < 12n * 4n ** BigInt(order),
    'cell must be a nested key at order'
  );
  const sideLength = 2 ** order;
  const face = Number(cell >> BigInt(2 * order));
  let column = 0;
  let row = 0;
  for (let bit = 0; bit < order; bit++) {
    column += Number((cell >> BigInt(2 * bit)) & 1n) * 2 ** bit;
    row += Number((cell >> BigInt(2 * bit + 1)) & 1n) * 2 ** bit;
  }
  const corners = [
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 1],
    [0, 0]
  ];
  const ring: [number, number][] = [];
  for (let edge = 0; edge < 4; edge++) {
    for (let step = 0; step < edgeSegments; step++) {
      const fraction = step / edgeSegments;
      const horizontal =
        column + corners[edge][0] * (1 - fraction) + corners[edge + 1][0] * fraction;
      const vertical = row + corners[edge][1] * (1 - fraction) + corners[edge + 1][1] * fraction;
      const ringIndex = FACE_RING[face] * sideLength - horizontal - vertical;
      const polarRadius = Math.min(ringIndex, 4 * sideLength - ringIndex, sideLength);
      // Form colatitude directly in the caps; asin(1 - tiny) collapses fine polar cells.
      const latitude =
        ringIndex < sideLength
          ? 90 - (2 * Math.asin(ringIndex / (Math.sqrt(6) * sideLength)) * 180) / Math.PI
          : ringIndex > 3 * sideLength
            ? -90 +
              (2 * Math.asin((4 * sideLength - ringIndex) / (Math.sqrt(6) * sideLength)) * 180) /
                Math.PI
            : (Math.asin(((2 * sideLength - ringIndex) * 2) / (3 * sideLength)) * 180) / Math.PI;
      let longitude =
        (FACE_LONGITUDE[face] + (polarRadius > 0 ? (horizontal - vertical) / polarRadius : 0)) * 45;
      const previousLongitude = ring.at(-1)?.[0] ?? 0;
      while (longitude - previousLongitude > 180) longitude -= 360;
      while (longitude - previousLongitude < -180) longitude += 360;
      ring.push([longitude, latitude]);
    }
  }
  ring.push([...ring[0]]);
  return ring;
}
