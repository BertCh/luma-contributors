// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  getCoveringCellGeometry,
  getCoveringCross,
  getCoveringDot,
  type CPUCoveringVector
} from '../cell-topology/cpu-s2-covering-geometry';

/** Unit-sphere measures before the CPU twin's single output rounding. @internal */
export type RectangularCellMeasures = {area: number; edges: number[]};

/** Invalid keys return undefined, matching the GPU's zero rows. @internal */
export function getRectangularCellMeasures(
  key: bigint,
  family: 's2' | 'quadbin'
): RectangularCellMeasures | undefined {
  if (family === 'quadbin') {
    const high = Number(key >> 32n);
    const resolution = (high >>> 20) & 31;
    if (high >>> 31 || !(high & 0x40000000) || ((high >>> 27) & 7) !== 1 || resolution > 26)
      return undefined;
    const unused = (1n << BigInt(52 - 2 * resolution)) - 1n;
    if ((key & unused) !== unused) return undefined;
    const morton = key >> BigInt(52 - 2 * resolution);
    let row = 0;
    for (let bit = 0; bit < resolution; bit++)
      row += Number((morton >> BigInt(2 * bit + 1)) & 1n) * 2 ** bit;
    const step = (2 * Math.PI) / 2 ** resolution;
    const north = Math.PI - row * step;
    const south = north - step;
    const meridian = 2 * Math.atan2(Math.sinh(step / 2), Math.cosh(north - step / 2));
    return {
      area: (step * Math.sinh(step)) / (Math.cosh(north) * Math.cosh(south)),
      edges: [step / Math.cosh(north), meridian, step / Math.cosh(south), meridian]
    };
  }
  if (key === 0n || key >> 61n >= 6n) return undefined;
  let markerBit = 0;
  while (((key >> BigInt(markerBit)) & 1n) === 0n) markerBit++;
  if (markerBit > 60 || markerBit % 2 !== 0) return undefined;
  const level = 30 - markerBit / 2;
  const {vertices} = getCoveringCellGeometry(key, level);
  // Covering geometry uses (00, 10, 11, 01); GPUCellGeometry uses (00, 01, 11, 10).
  const boundaryVertices = [vertices[0], vertices[3], vertices[2], vertices[1]];
  const subtract = (first: CPUCoveringVector, second: CPUCoveringVector): CPUCoveringVector => [
    first[0] - second[0],
    first[1] - second[1],
    first[2] - second[2]
  ];
  // Binary64 vector differences give an independent evaluation of the GPU's local increments.
  const triangleArea = (
    first: CPUCoveringVector,
    second: CPUCoveringVector,
    third: CPUCoveringVector
  ) =>
    2 *
    Math.atan2(
      Math.abs(
        getCoveringDot(first, getCoveringCross(subtract(second, first), subtract(third, first)))
      ),
      1 +
        getCoveringDot(first, second) +
        getCoveringDot(second, third) +
        getCoveringDot(third, first)
    );
  return {
    area:
      triangleArea(vertices[0], vertices[1], vertices[2]) +
      triangleArea(vertices[0], vertices[2], vertices[3]),
    edges: boundaryVertices.map((first, edge) => {
      const second = boundaryVertices[(edge + 1) % 4];
      return Math.atan2(
        Math.hypot(...getCoveringCross(first, subtract(second, first))),
        getCoveringDot(first, second)
      );
    })
  };
}
