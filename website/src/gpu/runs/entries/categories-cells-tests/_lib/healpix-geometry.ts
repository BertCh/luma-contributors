// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Drawing helper: the four corners of a HEALPix nested cell in longitude/latitude degrees, from the
// closed-form pixel to sphere formulas of Gorski et al. (2005). The analysis operations only produce
// keys; luma.gl has no HEALPix cell decoder, so the runs decode the keys read back from the GPU here,
// on the CPU, only to draw them.

const FACE_ROW = [2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4];
const FACE_COLUMN = [1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7];

function compactBits(value: bigint): number {
  let result = 0;
  for (let bit = 0; bit < 30; bit++) {
    if ((value >> BigInt(2 * bit)) & 1n) result |= 1 << bit;
  }
  return result;
}

/** Sphere position of a point (x, y) in face-local units of one pixel side = 1/nside. */
function getFacePoint(x: number, y: number, face: number): [number, number] {
  const jr = FACE_ROW[face] - x - y;
  let nr: number;
  let z: number;
  if (jr < 1) {
    nr = jr;
    z = 1 - (nr * nr) / 3;
  } else if (jr > 3) {
    nr = 4 - jr;
    z = (nr * nr) / 3 - 1;
  } else {
    nr = 1;
    z = ((2 - jr) * 2) / 3;
  }
  let tmp = FACE_COLUMN[face] * nr + x - y;
  if (tmp < 0) tmp += 8;
  if (tmp >= 8) tmp -= 8;
  const phi = nr < 1e-12 ? Number.NaN : (Math.PI / 4) * (tmp / nr);
  return [(phi * 180) / Math.PI, (Math.asin(Math.max(-1, Math.min(1, z))) * 180) / Math.PI];
}

/** Center and corners (south, east, north, west) of a nested pixel, longitudes in [-180, 180]. */
export function getHealpixCell(
  pixel: bigint,
  order: number
): {center: [number, number]; corners: [number, number][]} {
  const nside = 2 ** order;
  const face = Number(pixel >> BigInt(2 * order));
  const path = pixel & ((1n << BigInt(2 * order)) - 1n);
  const ix = compactBits(path);
  const iy = compactBits(path >> 1n);
  const x = (ix + 0.5) / nside;
  const y = (iy + 0.5) / nside;
  const half = 0.5 / nside;
  const wrap = (longitude: number) => ((((longitude + 180) % 360) + 360) % 360) - 180;
  const [centerLongitude, centerLatitude] = getFacePoint(x, y, face);
  const center: [number, number] = [wrap(centerLongitude), centerLatitude];
  const corners = (
    [
      [-half, -half],
      [half, -half],
      [half, half],
      [-half, half]
    ] as const
  ).map(([dx, dy]): [number, number] => {
    const [longitude, latitude] = getFacePoint(x + dx, y + dy, face);
    // Pole corners have no longitude of their own, and corners must stay near the center.
    let value = Number.isNaN(longitude) ? center[0] : wrap(longitude);
    while (value - center[0] > 180) value -= 360;
    while (value - center[0] < -180) value += 360;
    return [value, latitude];
  });
  return {center, corners};
}
