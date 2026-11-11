// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// A small ear-clipping triangulation of a simple ring, only for drawing input cells as fills. It
// returns vertex indices of the ring, so one triangulation serves every uniformly scaled copy of
// the ring. It is not an operation of the library: the pictures use it for their inputs.

function getCross(positions: ArrayLike<number>, from: number, to: number, through: number): number {
  return (
    (positions[to * 2] - positions[from * 2]) *
      (positions[through * 2 + 1] - positions[from * 2 + 1]) -
    (positions[to * 2 + 1] - positions[from * 2 + 1]) *
      (positions[through * 2] - positions[from * 2])
  );
}

function isInsideTriangle(
  positions: ArrayLike<number>,
  a: number,
  b: number,
  c: number,
  point: number
): boolean {
  return (
    getCross(positions, a, b, point) >= 0 &&
    getCross(positions, b, c, point) >= 0 &&
    getCross(positions, c, a, point) >= 0
  );
}

/**
 * Triangles (three vertex rows each) of the ring `first` to `end` (exclusive) in packed
 * `positions`, in either winding. Falls back to a fan for a ring that is not simple.
 */
export function clipEars(positions: ArrayLike<number>, first: number, end: number): number[] {
  const rows: number[] = [];
  for (let row = first; row < end; row++) rows.push(row);
  let area = 0;
  for (let index = 0; index < rows.length; index++) {
    const current = rows[index];
    const next = rows[(index + 1) % rows.length];
    area +=
      positions[current * 2] * positions[next * 2 + 1] -
      positions[next * 2] * positions[current * 2 + 1];
  }
  const orientation = area >= 0 ? 1 : -1;
  const triangles: number[] = [];
  let guard = rows.length * rows.length;
  while (rows.length > 3 && guard-- > 0) {
    let clipped = false;
    for (let index = 0; index < rows.length; index++) {
      const previous = rows[(index + rows.length - 1) % rows.length];
      const current = rows[index];
      const next = rows[(index + 1) % rows.length];
      if (getCross(positions, previous, current, next) * orientation <= 0) continue;
      const hasInside = rows.some(
        row =>
          row !== previous &&
          row !== current &&
          row !== next &&
          (orientation > 0
            ? isInsideTriangle(positions, previous, current, next, row)
            : isInsideTriangle(positions, next, current, previous, row))
      );
      if (hasInside) continue;
      triangles.push(previous, current, next);
      rows.splice(index, 1);
      clipped = true;
      break;
    }
    if (!clipped) break;
  }
  if (rows.length === 3) triangles.push(rows[0], rows[1], rows[2]);
  else if (rows.length > 3) {
    for (let index = 1; index + 1 < rows.length; index++) {
      triangles.push(rows[0], rows[index], rows[index + 1]);
    }
  }
  return triangles;
}
