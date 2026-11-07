// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** One projected label rectangle in screen pixels (x, y is the top-left corner). */
export type LabelCollisionCandidate = {
  rowIndex: number;
  x: number;
  y: number;
  width: number;
  height: number;
  /** Larger priorities are placed first (for example a size or weight field). */
  priority: number;
};

export type LabelCollisionOptions = {
  /** Candidates beyond this many (lowest priority first dropped) are hidden; bounds CPU time. */
  maximumCandidates?: number;
  /** Extra pixels around every rectangle that other labels must keep clear. */
  padding?: number;
  /** Spatial hash cell size in pixels. */
  cellSize?: number;
};

export const LABEL_COLLISION_DEFAULTS = Object.freeze({
  maximumCandidates: 20000,
  padding: 2,
  cellSize: 64
});

/**
 * Bounded greedy label placement: candidates are visited by descending priority (ties by row index)
 * and accepted when they overlap no already accepted rectangle. A spatial hash keeps the work close
 * to linear in the candidate count. Returns accepted row indices in placement order.
 */
export function placeLabelsGreedy(
  candidates: readonly LabelCollisionCandidate[],
  options: LabelCollisionOptions = {}
): number[] {
  const maximumCandidates = options.maximumCandidates ?? LABEL_COLLISION_DEFAULTS.maximumCandidates;
  const padding = options.padding ?? LABEL_COLLISION_DEFAULTS.padding;
  const cellSize = options.cellSize ?? LABEL_COLLISION_DEFAULTS.cellSize;
  const ordered = candidates
    .filter(
      candidate =>
        Number.isFinite(candidate.x) &&
        Number.isFinite(candidate.y) &&
        candidate.width > 0 &&
        candidate.height > 0
    )
    .sort((left, right) => right.priority - left.priority || left.rowIndex - right.rowIndex)
    .slice(0, maximumCandidates);
  const grid = new Map<string, LabelCollisionCandidate[]>();
  const accepted: number[] = [];
  const half = padding / 2;
  for (const candidate of ordered) {
    const left = candidate.x - half;
    const top = candidate.y - half;
    const right = candidate.x + candidate.width + half;
    const bottom = candidate.y + candidate.height + half;
    const firstColumn = Math.floor(left / cellSize);
    const lastColumn = Math.floor(right / cellSize);
    const firstRow = Math.floor(top / cellSize);
    const lastRow = Math.floor(bottom / cellSize);
    let blocked = false;
    for (let row = firstRow; row <= lastRow && !blocked; row++) {
      for (let column = firstColumn; column <= lastColumn && !blocked; column++) {
        for (const other of grid.get(`${column},${row}`) ?? []) {
          if (
            left < other.x + other.width + half &&
            right > other.x - half &&
            top < other.y + other.height + half &&
            bottom > other.y - half
          ) {
            blocked = true;
            break;
          }
        }
      }
    }
    if (blocked) continue;
    accepted.push(candidate.rowIndex);
    for (let row = firstRow; row <= lastRow; row++) {
      for (let column = firstColumn; column <= lastColumn; column++) {
        const key = `${column},${row}`;
        const cell = grid.get(key);
        if (cell) cell.push(candidate);
        else grid.set(key, [candidate]);
      }
    }
  }
  return accepted;
}

/** Packs accepted rows into the uint32 visibility words (1 bit per row) a label mask consumes. */
export function packAcceptedLabelRows(
  rowCount: number,
  acceptedRows: readonly number[]
): Uint32Array {
  const words = new Uint32Array(Math.ceil(rowCount / 32));
  for (const rowIndex of acceptedRows) words[rowIndex >>> 5] |= 1 << (rowIndex & 31);
  return words;
}
