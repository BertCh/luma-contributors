// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Helpers for the pointer gestures of the cell, index and join runs: formatting the keys the GPU
// wrote and turning CPU copies of decoded boundaries into glyphs.

import type {Point, RunGlyph, RunPeek} from '../../../interaction';
import type {PointerCell} from './cells';

type Words = ArrayLike<number>;

/** The 64-bit cell key stored as (low, high) words, as the hexadecimal string cell tools print. */
export function formatCellKey(keys: Words | null, row = 0): string {
  if (!keys) return '…';
  const low = BigInt(keys[row * 2]);
  const high = BigInt(keys[row * 2 + 1]);
  return ((high << 32n) | low).toString(16);
}

/** The first `count` rows of a `uint32x2` key column whose key equals the key at `row` of `keys`. */
export function findKeyRows(column: Words, count: number, keys: Words, row = 0): number[] {
  const rows: number[] = [];
  for (let index = 0; index < count; index++) {
    if (column[index * 2] === keys[row * 2] && column[index * 2 + 1] === keys[row * 2 + 1]) {
      rows.push(index);
    }
  }
  return rows;
}

/** The closed outline of cell `row` of a decoded boundary column (`stride` slots per cell). */
export function getBoundaryGlyph(
  boundaries: Words | null,
  row: number,
  stride: number,
  vertexCount: number
): RunGlyph | null {
  if (!boundaries || vertexCount < 3) return null;
  const points: number[] = [];
  for (let corner = 0; corner < vertexCount; corner++) {
    points.push(
      boundaries[(row * stride + corner) * 2],
      boundaries[(row * stride + corner) * 2 + 1]
    );
  }
  return points.every(Number.isFinite) ? {shape: 'polyline', points, closed: true} : null;
}

/** The mean of a cell's boundary slots, for a cell whose center was not decoded. */
export function getBoundaryCenter(
  boundaries: Words,
  row: number,
  stride: number,
  vertexCount: number
): Point {
  let sumX = 0;
  let sumY = 0;
  for (let corner = 0; corner < vertexCount; corner++) {
    sumX += boundaries[(row * stride + corner) * 2];
    sumY += boundaries[(row * stride + corner) * 2 + 1];
  }
  return [sumX / vertexCount, sumY / vertexCount];
}

/** `141.9°E 37.4°N`. */
export function formatLongitudeLatitude(longitude: number, latitude: number): string {
  const east = longitude >= 0 ? 'E' : 'W';
  const north = latitude >= 0 ? 'N' : 'S';
  return `${Math.abs(longitude).toFixed(1)}°${east} ${Math.abs(latitude).toFixed(1)}°${north}`;
}

/** Window center in degrees, where a geographic pointer rests by default. */
export function getWindowCenter(window: readonly [number, number, number, number]): Point {
  return [(window[0] + window[2]) / 2, (window[1] + window[3]) / 2];
}

/** CPU copies of the one-row pointer cell: its key, decoded corners and center, and its outline. */
export function readPointerCell(peek: RunPeek, cell: PointerCell) {
  const keys = peek(cell.cells);
  const boundaries = peek(cell.boundaries);
  const centers = peek(cell.centers);
  const counts = peek(cell.vertexCounts);
  const vertexCount = counts && counts[0] > 0 ? counts[0] : cell.vertexCount;
  return {
    keys,
    boundaries,
    centers,
    vertexCount,
    key: formatCellKey(keys),
    outline: getBoundaryGlyph(boundaries, 0, cell.stride, vertexCount)
  };
}

/**
 * A point whose nearest segment is `distance` away, found by bisecting along the normal of the
 * segments of `outline` (packed x0, y0, x1, y1); null when no side reaches it. For a `rest` that
 * reproduces a default distance.
 */
export function findPointAtDistance(
  outline: ArrayLike<number>,
  distance: number,
  findNearest: (x: number, y: number) => number
): Point | null {
  const count = outline.length / 4;
  for (let attempt = 0; attempt < count; attempt++) {
    const segment = Math.floor((attempt * 7) % count);
    const [x0, y0, x1, y1] = [0, 1, 2, 3].map(offset => outline[segment * 4 + offset]);
    const length = Math.hypot(x1 - x0, y1 - y0);
    if (length < distance) continue;
    for (const side of [1, -1]) {
      const middleX = (x0 + x1) / 2;
      const middleY = (y0 + y1) / 2;
      const normalX = (-(y1 - y0) / length) * side;
      const normalY = ((x1 - x0) / length) * side;
      let near = 0;
      let far = distance * 4;
      for (let step = 0; step < 30; step++) {
        const reach = (near + far) / 2;
        if (findNearest(middleX + normalX * reach, middleY + normalY * reach) < distance)
          near = reach;
        else far = reach;
      }
      const reach = (near + far) / 2;
      const x = middleX + normalX * reach;
      const y = middleY + normalY * reach;
      if (Math.abs(findNearest(x, y) - distance) < 1e-3) return [x, y];
    }
  }
  return null;
}
