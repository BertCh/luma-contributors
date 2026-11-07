// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {CatalogPosition} from './catalog-geojson';

/**
 * CPU references for the discrete global grid keys of GPUPointToCell and GPUCellGeometry
 * (Quadbin, quadkey, geohash, S2). H3 has no CPU reference here: it needs the GPU contributors.
 * Inputs are rounded to float32 first, matching the GPU kernels, which read float32 positions.
 */
export type DggsCpuFamily = 'quadbin' | 'quadkey' | 'geohash' | 's2';

export const DGGS_CPU_FAMILIES: readonly DggsCpuFamily[] = ['quadbin', 'quadkey', 'geohash', 's2'];

const MAXIMUM_MERCATOR_LATITUDE = 85.051129;
const QUADBIN_HEADER = 0x4000000000000000n | (1n << 59n);
const QUADBIN_FOOTER = 0xfffffffffffffn;
const S2_MAXIMUM_LEVEL = 30;
const S2_SWAP_MASK = 1;
const S2_INVERT_MASK = 2;
const S2_POSITION_TO_IJ = [
  [0, 1, 3, 2],
  [0, 2, 3, 1],
  [3, 2, 0, 1],
  [3, 1, 0, 2]
];
const S2_POSITION_TO_ORIENTATION = [S2_SWAP_MASK, 0, 0, S2_SWAP_MASK | S2_INVERT_MASK];

function getTileX(longitude: number, zoom: number): number {
  const clipped = Math.min(Math.max(longitude, -180), 180);
  const tileCount = 2 ** zoom;
  return Math.floor(((clipped + 180) * tileCount) / 360) % tileCount;
}

function getTileY(latitude: number, zoom: number): number {
  const clipped = Math.min(
    Math.max(latitude, -MAXIMUM_MERCATOR_LATITUDE),
    MAXIMUM_MERCATOR_LATITUDE
  );
  const sine = Math.sin((clipped * Math.PI) / 180);
  const tileCount = 2 ** zoom;
  const row = tileCount * (0.5 - (0.25 * Math.log((1 + sine) / (1 - sine))) / Math.PI);
  return Math.min(Math.max(Math.floor(row), 0), tileCount - 1);
}

function interleaveTile(x: number, y: number, zoom: number): bigint {
  let digits = 0n;
  for (let bit = 0; bit < zoom; bit++) {
    digits |= BigInt((x >>> bit) & 1) << BigInt(2 * bit);
    digits |= BigInt((y >>> bit) & 1) << BigInt(2 * bit + 1);
  }
  return digits;
}

function makeQuadbinCell(x: number, y: number, zoom: number): bigint {
  const zoomBig = BigInt(zoom);
  return (
    QUADBIN_HEADER |
    (zoomBig << 52n) |
    (interleaveTile(x, y, zoom) << (52n - 2n * zoomBig)) |
    (QUADBIN_FOOTER >> (2n * zoomBig))
  );
}

function readQuadbinTile(cell: bigint): {x: number; y: number; z: number} {
  const z = Number((cell >> 52n) & 0x1fn);
  const digits = (cell >> BigInt(52 - 2 * z)) & ((1n << BigInt(2 * z)) - 1n);
  let x = 0;
  let y = 0;
  for (let bit = 0; bit < z; bit++) {
    x |= Number((digits >> BigInt(2 * bit)) & 1n) << bit;
    y |= Number((digits >> BigInt(2 * bit + 1)) & 1n) << bit;
  }
  return {x: x >>> 0, y: y >>> 0, z};
}

/** Quadbin parent of a cell at a coarser zoom. */
export function getQuadbinParent(cell: bigint, zoom: number): bigint {
  const zoomBig = BigInt(zoom);
  return (cell & ~(0x1fn << 52n)) | (zoomBig << 52n) | (QUADBIN_FOOTER >> (2n * zoomBig));
}

function readQuadkeyTile(cell: bigint): {x: number; y: number; z: number} {
  const z = Number(cell >> 58n);
  let x = 0;
  let y = 0;
  for (let bit = 0; bit < z; bit++) {
    x |= Number((cell >> BigInt(2 * bit)) & 1n) << bit;
    y |= Number((cell >> BigInt(2 * bit + 1)) & 1n) << bit;
  }
  return {x: x >>> 0, y: y >>> 0, z};
}

function getGeohashCell(longitude: number, latitude: number, length: number): bigint {
  let west = -180;
  let east = 180;
  let south = -90;
  let north = 90;
  let bits = 0n;
  for (let bit = 0; bit < 5 * length; bit++) {
    let isOne: boolean;
    if (bit % 2 === 0) {
      const midpoint = (west + east) / 2;
      isOne = longitude >= midpoint;
      if (isOne) west = midpoint;
      else east = midpoint;
    } else {
      const midpoint = (south + north) / 2;
      isOne = latitude >= midpoint;
      if (isOne) south = midpoint;
      else north = midpoint;
    }
    bits = (bits << 1n) | (isOne ? 1n : 0n);
  }
  return (BigInt(length) << 60n) | bits;
}

function getGeohashBounds(cell: bigint): [number, number, number, number] {
  const length = Number(cell >> 60n);
  let west = -180;
  let east = 180;
  let south = -90;
  let north = 90;
  for (let bit = 0; bit < 5 * length; bit++) {
    const isOne = ((cell >> BigInt(5 * length - 1 - bit)) & 1n) === 1n;
    if (bit % 2 === 0) {
      const midpoint = (west + east) / 2;
      if (isOne) west = midpoint;
      else east = midpoint;
    } else {
      const midpoint = (south + north) / 2;
      if (isOne) south = midpoint;
      else north = midpoint;
    }
  }
  return [west, south, east, north];
}

export function getS2Face(x: number, y: number, z: number): number {
  const absolute = [Math.abs(x), Math.abs(y), Math.abs(z)];
  const axis =
    absolute[0] > absolute[1]
      ? absolute[0] > absolute[2]
        ? 0
        : 2
      : absolute[1] > absolute[2]
        ? 1
        : 2;
  return axis + ([x, y, z][axis] < 0 ? 3 : 0);
}

export function getS2FaceUV(face: number, x: number, y: number, z: number): [number, number] {
  switch (face) {
    case 0:
      return [y / x, z / x];
    case 1:
      return [-x / y, z / y];
    case 2:
      return [-x / z, -y / z];
    case 3:
      return [z / x, y / x];
    case 4:
      return [z / y, -x / y];
    default:
      return [-y / z, -x / z];
  }
}

export function getS2FaceXYZ(face: number, u: number, v: number): [number, number, number] {
  switch (face) {
    case 0:
      return [1, u, v];
    case 1:
      return [-u, 1, v];
    case 2:
      return [-u, -v, 1];
    case 3:
      return [-1, -v, -u];
    case 4:
      return [v, -1, -u];
    default:
      return [v, u, -1];
  }
}

export function getS2ST(uv: number): number {
  return uv >= 0 ? 0.5 * Math.sqrt(1 + 3 * uv) : 1 - 0.5 * Math.sqrt(1 - 3 * uv);
}

export function getS2UV(st: number): number {
  return st >= 0.5 ? (4 * st * st - 1) / 3 : (1 - 4 * (1 - st) * (1 - st)) / 3;
}

function getS2Leaf(st: number): number {
  return Math.max(0, Math.min(2 ** S2_MAXIMUM_LEVEL - 1, Math.floor(2 ** S2_MAXIMUM_LEVEL * st)));
}

export function getS2Cell(face: number, i: number, j: number, level: number): bigint {
  let orientation = face & S2_SWAP_MASK;
  let position = 0n;
  for (let step = 0; step < level; step++) {
    const bit = S2_MAXIMUM_LEVEL - 1 - step;
    const ij = (((i >>> bit) & 1) << 1) | ((j >>> bit) & 1);
    const digit = S2_POSITION_TO_IJ[orientation].indexOf(ij);
    position = (position << 2n) | BigInt(digit);
    orientation ^= S2_POSITION_TO_ORIENTATION[digit];
  }
  const shift = BigInt(2 * (S2_MAXIMUM_LEVEL - level));
  return (BigInt(face) << 61n) | (position << (shift + 1n)) | (1n << shift);
}

export function readS2Cell(cell: bigint): {face: number; i: number; j: number; level: number} {
  const face = Number(cell >> 61n);
  let trailingZeros = 0;
  while (((cell >> BigInt(trailingZeros)) & 1n) === 0n) trailingZeros++;
  const level = S2_MAXIMUM_LEVEL - trailingZeros / 2;
  let orientation = face & S2_SWAP_MASK;
  let i = 0;
  let j = 0;
  for (let step = 0; step < level; step++) {
    const digit = Number((cell >> BigInt(60 - 2 * step - 1)) & 3n);
    const ij = S2_POSITION_TO_IJ[orientation][digit];
    i = (i << 1) | (ij >> 1);
    j = (j << 1) | (ij & 1);
    orientation ^= S2_POSITION_TO_ORIENTATION[digit];
  }
  return {face, i, j, level};
}

function getS2Cell3D(longitude: number, latitude: number, level: number): bigint {
  const lng = (longitude * Math.PI) / 180;
  const lat = (latitude * Math.PI) / 180;
  const x = Math.cos(lat) * Math.cos(lng);
  const y = Math.cos(lat) * Math.sin(lng);
  const z = Math.sin(lat);
  const face = getS2Face(x, y, z);
  const [u, v] = getS2FaceUV(face, x, y, z);
  return getS2Cell(face, getS2Leaf(getS2ST(u)), getS2Leaf(getS2ST(v)), level);
}

/** Cell key of a longitude/latitude; float32-rounded inputs, zero for a non-finite position. */
export function getDggsCpuCell(
  family: DggsCpuFamily,
  longitude: number,
  latitude: number,
  resolution: number
): bigint {
  const lng = Math.fround(longitude);
  const lat = Math.fround(latitude);
  if (!Number.isFinite(lng) || !Number.isFinite(lat)) return 0n;
  switch (family) {
    case 'quadbin':
      return makeQuadbinCell(getTileX(lng, resolution), getTileY(lat, resolution), resolution);
    case 'quadkey':
      return (
        (BigInt(resolution) << 58n) |
        interleaveTile(getTileX(lng, resolution), getTileY(lat, resolution), resolution)
      );
    case 'geohash':
      return getGeohashCell(lng, lat, resolution);
    case 's2':
      return getS2Cell3D(lng, lat, resolution);
  }
}

function getMercatorBounds(x: number, y: number, z: number): [number, number, number, number] {
  const scale = 2 ** z;
  const latitude = (row: number) =>
    (Math.atan(Math.sinh(Math.PI * (1 - (2 * row) / scale))) * 180) / Math.PI;
  return [(x / scale) * 360 - 180, latitude(y + 1), ((x + 1) / scale) * 360 - 180, latitude(y)];
}

function makeRectangleRing(bounds: [number, number, number, number]): CatalogPosition[] {
  const [west, south, east, north] = bounds;
  return [
    [west, south],
    [east, south],
    [east, north],
    [west, north],
    [west, south]
  ];
}

/** Closed boundary ring of a cell in longitude/latitude degrees (great circle edges for S2 are not densified). */
export function getDggsCpuBoundary(family: DggsCpuFamily, cell: bigint): CatalogPosition[] {
  switch (family) {
    case 'quadbin': {
      const {x, y, z} = readQuadbinTile(cell);
      return makeRectangleRing(getMercatorBounds(x, y, z));
    }
    case 'quadkey': {
      const {x, y, z} = readQuadkeyTile(cell);
      return makeRectangleRing(getMercatorBounds(x, y, z));
    }
    case 'geohash':
      return makeRectangleRing(getGeohashBounds(cell));
    case 's2': {
      const {face, i, j, level} = readS2Cell(cell);
      const scale = 2 ** level;
      const ring = [
        [0, 0],
        [0, 1],
        [1, 1],
        [1, 0]
      ].map(([deltaI, deltaJ]): CatalogPosition => {
        const [x, y, z] = getS2FaceXYZ(
          face,
          getS2UV((i + deltaI) / scale),
          getS2UV((j + deltaJ) / scale)
        );
        return [
          (Math.atan2(y, x) * 180) / Math.PI,
          (Math.atan2(z, Math.hypot(x, y)) * 180) / Math.PI
        ];
      });
      return [...ring, ring[0]];
    }
  }
}

/** Sixteen lowercase hex digits of a 64-bit key. */
export function formatDggsCellId(cell: bigint): string {
  return cell.toString(16).padStart(16, '0');
}

/** Little-endian (low, high) words of a key to a BigInt. */
export function joinDggsWords(low: number, high: number): bigint {
  return (BigInt(high) << 32n) | BigInt(low);
}
