// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Binary64 CPU reference for the HEALPix nested scheme, written from the C++ `healpix_base`
// formulas (ang2pix_nest, pix2ang_nest). It is independent of the WGSL kernel and is checked
// against healpix-geo by `rts08-healpix-point-to-cell.spec.ts`.

const FACE_ROW = [2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4];
const FACE_COLUMN = [1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7];

function spreadBits(value: number): bigint {
  let result = 0n;
  for (let bit = 0; bit < 30; bit++) {
    if ((value >> bit) & 1) result |= 1n << BigInt(2 * bit);
  }
  return result;
}

function compactBits(value: bigint): number {
  let result = 0;
  for (let bit = 0; bit < 30; bit++) {
    if ((value >> BigInt(2 * bit)) & 1n) result |= 1 << bit;
  }
  return result;
}

/** Nested pixel of a longitude/latitude (degrees) at `order`, in binary64 on the sphere. */
export function getHealpixNestedPixel(longitude: number, latitude: number, order: number): bigint {
  const nside = 2 ** order;
  const z = Math.sin((latitude * Math.PI) / 180);
  const absoluteZ = Math.abs(z);
  const wrapped = ((longitude % 360) + 360) % 360;
  const tt = wrapped / 90;
  let face: number;
  let ix: number;
  let iy: number;
  if (absoluteZ <= 2 / 3) {
    const temp1 = nside * (0.5 + tt);
    const temp2 = nside * z * 0.75;
    const jp = Math.floor(temp1 - temp2);
    const jm = Math.floor(temp1 + temp2);
    const ifp = Math.floor(jp / nside);
    const ifm = Math.floor(jm / nside);
    face = ifp === ifm ? ifp | 4 : ifp < ifm ? ifp : ifm + 8;
    ix = jm % nside;
    iy = nside - (jp % nside) - 1;
  } else {
    const ntt = Math.min(Math.floor(tt), 3);
    const tp = tt - ntt;
    const factor = nside * Math.sqrt(3 * (1 - absoluteZ));
    const jp = Math.min(Math.floor(tp * factor), nside - 1);
    const jm = Math.min(Math.floor((1 - tp) * factor), nside - 1);
    if (z >= 0) {
      face = ntt;
      ix = nside - jm - 1;
      iy = nside - jp - 1;
    } else {
      face = ntt + 8;
      ix = jp;
      iy = jm;
    }
  }
  return (BigInt(face) << BigInt(2 * order)) + spreadBits(ix) + (spreadBits(iy) << 1n);
}

/** Center longitude/latitude (degrees) of a nested pixel at `order`. */
export function getHealpixNestedCenter(
  pixel: bigint,
  order: number
): {longitude: number; latitude: number} {
  const nside = 2 ** order;
  const face = Number(pixel >> BigInt(2 * order));
  const path = pixel & ((1n << BigInt(2 * order)) - 1n);
  const ix = compactBits(path);
  const iy = compactBits(path >> 1n);
  const ring = FACE_ROW[face] * nside - ix - iy - 1;
  let rowLength: number;
  let z: number;
  let shift = 0;
  if (ring < nside) {
    rowLength = ring;
    z = 1 - (rowLength * rowLength) / (3 * nside * nside);
  } else if (ring > 3 * nside) {
    rowLength = 4 * nside - ring;
    z = -(1 - (rowLength * rowLength) / (3 * nside * nside));
  } else {
    rowLength = nside;
    z = ((2 * nside - ring) * 2) / (3 * nside);
    shift = (ring - nside) & 1;
  }
  let column = (FACE_COLUMN[face] * rowLength + ix - iy + 1 + shift) / 2;
  if (column > 4 * nside) column -= 4 * nside;
  if (column < 1) column += 4 * nside;
  const phi = ((column - (shift + 1) * 0.5) * (Math.PI / 2)) / rowLength;
  const longitude = (((phi * 180) / Math.PI + 180) % 360) - 180;
  return {longitude, latitude: (Math.asin(z) * 180) / Math.PI};
}

/** Great-circle angle in degrees between two longitude/latitude points. */
export function getAngleDegrees(
  first: {longitude: number; latitude: number},
  second: {longitude: number; latitude: number}
): number {
  const radians = Math.PI / 180;
  const latitude1 = first.latitude * radians;
  const latitude2 = second.latitude * radians;
  const delta = (second.longitude - first.longitude) * radians;
  const cosine =
    Math.sin(latitude1) * Math.sin(latitude2) +
    Math.cos(latitude1) * Math.cos(latitude2) * Math.cos(delta);
  return Math.acos(Math.min(1, Math.max(-1, cosine))) / radians;
}
