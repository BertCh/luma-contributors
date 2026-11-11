// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Shared drawing support for the HEALPix runs: the world coastline vertices as the input points, an
// equirectangular fit of the globe into the canvas, and a quad mesh of the cells whose keys the GPU
// wrote (decoded on the CPU by `healpix-geometry`, for drawing only).

import coastline from '../../../../data/world-coastline.json';
import {createCoastlineSegments} from '../../../../data/coastline';
import type {World} from '../../../inputs';
import {getHealpixCell} from './healpix-geometry';

/** Longitude/latitude pairs (degrees) of every vertex of the 1:110m world coastline. */
export function getCoastPoints(): Float32Array {
  const points: number[] = [];
  for (const line of coastline) {
    for (const [longitude, latitude] of line) points.push(longitude, latitude);
  }
  return Float32Array.from(points);
}

export type GlobeFit = {
  scale: number;
  toWorld(longitude: number, latitude: number): [number, number];
  toLongitudeLatitude(x: number, y: number): [number, number];
  coast: Float32Array;
};

/** Fits the whole globe (360 x 180 degrees) into the left `share` of the canvas. */
export function createGlobeFit(world: World, share = 1): GlobeFit {
  const width = world.width * share;
  const scale = 0.96 * Math.min(width / 360, world.height / 180);
  const offsetX = (width - 360 * scale) / 2 + 180 * scale;
  const offsetY = (world.height - 180 * scale) / 2 + 90 * scale;
  return {
    scale,
    toWorld: (longitude, latitude) => [offsetX + longitude * scale, offsetY + latitude * scale],
    toLongitudeLatitude: (x, y) => [(x - offsetX) / scale, (y - offsetY) / scale],
    coast: createCoastlineSegments([scale, scale], [offsetX, offsetY])
  };
}

/** The 64-bit keys of `count` rows of a `uint32x2` (low, high) column, with the UNIQ marker removed. */
export function decodeKeys(words: ArrayLike<number>, count: number, marker = 0n): bigint[] {
  return Array.from(
    {length: count},
    (_, row) => ((BigInt(words[row * 2 + 1]) << 32n) | BigInt(words[row * 2])) - marker
  );
}

export type CellMesh = {
  count: number;
  /** Four corners per cell as world (x, y) pairs. */
  positions: Float32Array;
  indices: Uint32Array;
  owners: Uint32Array;
  triangleCount: number;
  /** Cell centers as world (x, y) pairs. */
  centers: Float32Array;
  /** Closed outline of each cell as four world segments (x0, y0, x1, y1). */
  segments: Float32Array;
};

/** A quad mesh of nested HEALPix cells at `order`, placed with `fit`. */
export function createCellMesh(pixels: readonly bigint[], order: number, fit: GlobeFit): CellMesh {
  const count = pixels.length;
  const positions = new Float32Array(count * 8);
  const segments = new Float32Array(count * 16);
  const centers = new Float32Array(count * 2);
  const indices = new Uint32Array(count * 6);
  const owners = new Uint32Array(count * 2);
  pixels.forEach((pixel, cell) => {
    const {corners, center} = getHealpixCell(pixel, order);
    centers.set(fit.toWorld(center[0], center[1]), cell * 2);
    const placed = corners.map(([longitude, latitude]) => fit.toWorld(longitude, latitude));
    placed.forEach(([x, y], corner) => positions.set([x, y], cell * 8 + corner * 2));
    placed.forEach(([x, y], corner) => {
      const [nextX, nextY] = placed[(corner + 1) % 4];
      segments.set([x, y, nextX, nextY], cell * 16 + corner * 4);
    });
    indices.set(
      [0, 1, 2, 0, 2, 3].map(corner => cell * 4 + corner),
      cell * 6
    );
    owners.set([cell, cell], cell * 2);
  });
  return {count, positions, centers, indices, owners, triangleCount: count * 2, segments};
}
