// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * CPU setup of a Vecchia likelihood: orderings and ordered nearest-neighbour conditioning sets. The
 * sets depend only on the coordinates and the ordering, so they are built once and reused across
 * every parameter evaluation. This runs before the GPU work, as a graph topology does.
 */

/** Grid of a uniform bucketed index over the coordinates. */
type PointGrid = {
  minimumX: number;
  minimumY: number;
  cellSize: number;
  columns: number;
  rows: number;
  heads: Int32Array;
  next: Int32Array;
};

/** Mean rows per cell at full occupancy. */
const ROWS_PER_CELL = 2;

function createPointGrid(positions: ArrayLike<number>, rowCount: number): PointGrid {
  let minimumX = Infinity;
  let minimumY = Infinity;
  let maximumX = -Infinity;
  let maximumY = -Infinity;
  for (let row = 0; row < rowCount; row++) {
    minimumX = Math.min(minimumX, positions[row * 2]);
    maximumX = Math.max(maximumX, positions[row * 2]);
    minimumY = Math.min(minimumY, positions[row * 2 + 1]);
    maximumY = Math.max(maximumY, positions[row * 2 + 1]);
  }
  const width = Math.max(maximumX - minimumX, Number.MIN_VALUE);
  const height = Math.max(maximumY - minimumY, Number.MIN_VALUE);
  const cellCount = Math.max(1, Math.round(rowCount / ROWS_PER_CELL));
  const cellSize = Math.max(
    Math.sqrt((width * height) / cellCount),
    Math.max(width, height) / 4096
  );
  const columns = Math.floor(width / cellSize) + 1;
  const rows = Math.floor(height / cellSize) + 1;
  return {
    minimumX,
    minimumY,
    cellSize,
    columns,
    rows,
    heads: new Int32Array(columns * rows).fill(-1),
    next: new Int32Array(rowCount).fill(-1)
  };
}

function getCellColumn(grid: PointGrid, x: number): number {
  return Math.min(grid.columns - 1, Math.floor((x - grid.minimumX) / grid.cellSize));
}

function getCellRow(grid: PointGrid, y: number): number {
  return Math.min(grid.rows - 1, Math.floor((y - grid.minimumY) / grid.cellSize));
}

function insertPoint(grid: PointGrid, positions: ArrayLike<number>, row: number): void {
  const cell =
    getCellRow(grid, positions[row * 2 + 1]) * grid.columns +
    getCellColumn(grid, positions[row * 2]);
  grid.next[row] = grid.heads[cell];
  grid.heads[cell] = row;
}

/**
 * Visits the indexed rows in square rings around `(x, y)`, nearest ring first, while `limit` (the
 * largest squared distance still of interest) permits. A ring at Chebyshev offset `ring` is at least
 * `(ring - 1) * cellSize` from a point inside the centre cell.
 */
function visitRings(
  grid: PointGrid,
  x: number,
  y: number,
  limit: () => number,
  visit: (row: number) => void
): void {
  const column = getCellColumn(grid, x);
  const row = getCellRow(grid, y);
  const maximumRing = Math.max(column, grid.columns - 1 - column, row, grid.rows - 1 - row);
  for (let ring = 0; ring <= maximumRing; ring++) {
    const gap = Math.max(ring - 1, 0) * grid.cellSize;
    if (gap * gap > limit()) {
      break;
    }
    for (
      let cellRow = Math.max(row - ring, 0);
      cellRow <= Math.min(row + ring, grid.rows - 1);
      cellRow++
    ) {
      const isEdgeRow = Math.abs(cellRow - row) === ring;
      const firstColumn = column - ring;
      const lastColumn = column + ring;
      if (isEdgeRow) {
        for (
          let cellColumn = Math.max(firstColumn, 0);
          cellColumn <= Math.min(lastColumn, grid.columns - 1);
          cellColumn++
        ) {
          for (
            let sample = grid.heads[cellRow * grid.columns + cellColumn];
            sample >= 0;
            sample = grid.next[sample]
          ) {
            visit(sample);
          }
        }
      } else {
        if (firstColumn >= 0) {
          for (
            let sample = grid.heads[cellRow * grid.columns + firstColumn];
            sample >= 0;
            sample = grid.next[sample]
          ) {
            visit(sample);
          }
        }
        if (lastColumn < grid.columns) {
          for (
            let sample = grid.heads[cellRow * grid.columns + lastColumn];
            sample >= 0;
            sample = grid.next[sample]
          ) {
            visit(sample);
          }
        }
      }
    }
  }
}

/** Compressed sparse rows of conditioning sets: row `r` conditions on `neighbors[offsets[r]..offsets[r+1]]`. */
export type VecchiaConditioningSets = {
  /** `rowCount + 1` offsets; the row order is the input row order, not the Vecchia order. */
  offsets: Uint32Array;
  /** Earlier rows of the ordering, nearest first (`GPULocalKriging` re-sorts them by distance). */
  neighbors: Uint32Array;
};

/** Options of {@link buildVecchiaConditioningSets}. */
export type VecchiaConditioningSetProps = {
  /** Packed x, y pairs (`Float64Array`, `Float32Array` or an array). */
  positions: ArrayLike<number>;
  /** Number of rows. Defaults to `positions.length / 2`. */
  rowCount?: number;
  /** Largest conditioning set, 1..32 as for `GPULocalKriging`. */
  neighborCount: number;
  /** Vecchia order: `order[k]` is the row at position k. Defaults to the row order. */
  order?: ArrayLike<number>;
  /** Optional per-row validity; invalid rows neither condition nor are conditioned. */
  validity?: ArrayLike<number>;
  /** Optional per-row times (space-time). Requires `timeScale`. */
  times?: ArrayLike<number>;
  /** Spatial units per unit of time of the metric `sqrt(h^2 + (timeScale u)^2)`. */
  timeScale?: number;
};

/**
 * Builds ordered conditioning sets: for each row, up to `neighborCount` of the rows earlier in the
 * Vecchia order that are nearest in the plane, or in the space-time metric when `times` is given
 * (Euclidean distance on the input precision; the 2-D grid stays a valid bound for the
 * 3-D distance, so a time axis costs extra ring visits, not correctness). Equal distances resolve
 * to the row earlier in the Vecchia order. The first row of the order has no
 * neighbours, so its likelihood term is the marginal one.
 *
 * A uniform grid keeps setup near-linear in the row count for spread data.
 */
export function buildVecchiaConditioningSets(
  props: VecchiaConditioningSetProps
): VecchiaConditioningSets {
  const {positions, neighborCount, order, validity, times, timeScale = 0} = props;
  const rowCount = props.rowCount ?? positions.length / 2;
  const grid = createPointGrid(positions, rowCount);
  const counts = new Uint32Array(rowCount);
  const neighbors = new Uint32Array(rowCount * neighborCount);
  const orderPositions = new Uint32Array(rowCount);
  for (let position = 0; position < rowCount; position++) {
    orderPositions[order ? order[position] : position] = position;
  }
  const bestRows = new Uint32Array(neighborCount);
  const bestDistances = new Float64Array(neighborCount);
  for (let position = 0; position < rowCount; position++) {
    const row = order ? order[position] : position;
    if (validity && !validity[row]) {
      continue;
    }
    const x = positions[row * 2];
    const y = positions[row * 2 + 1];
    let found = 0;
    visitRings(
      grid,
      x,
      y,
      () => (found < neighborCount ? Infinity : bestDistances[neighborCount - 1]),
      sample => {
        if (validity && !validity[sample]) {
          return;
        }
        const offsetX = x - positions[sample * 2];
        const offsetY = y - positions[sample * 2 + 1];
        const scaledTime = times ? timeScale * (times[row] - times[sample]) : 0;
        const separation = offsetX * offsetX + offsetY * offsetY + scaledTime * scaledTime;
        let slot = found < neighborCount ? found : neighborCount;
        while (
          slot > 0 &&
          (bestDistances[slot - 1] > separation ||
            (bestDistances[slot - 1] === separation &&
              orderPositions[bestRows[slot - 1]] > orderPositions[sample]))
        ) {
          if (slot < neighborCount) {
            bestDistances[slot] = bestDistances[slot - 1];
            bestRows[slot] = bestRows[slot - 1];
          }
          slot--;
        }
        if (slot < neighborCount) {
          bestDistances[slot] = separation;
          bestRows[slot] = sample;
          if (found < neighborCount) {
            found++;
          }
        }
      }
    );
    counts[row] = found;
    for (let slot = 0; slot < found; slot++) {
      neighbors[row * neighborCount + slot] = bestRows[slot];
    }
    insertPoint(grid, positions, row);
  }
  const offsets = new Uint32Array(rowCount + 1);
  for (let row = 0; row < rowCount; row++) {
    offsets[row + 1] = offsets[row] + counts[row];
  }
  const packed = new Uint32Array(offsets[rowCount]);
  for (let row = 0; row < rowCount; row++) {
    for (let slot = 0; slot < counts[row]; slot++) {
      packed[offsets[row] + slot] = neighbors[row * neighborCount + slot];
    }
  }
  return {offsets, neighbors: packed};
}

/** CSR of the data rows nearest each query, `[offsets, neighbors]` as for the conditioning sets. */
export function buildVecchiaPredictionNeighborhoods(props: {
  /** Packed data x, y pairs. */
  positions: ArrayLike<number>;
  /** Packed query x, y pairs. */
  queries: ArrayLike<number>;
  neighborCount: number;
  /** Optional data validity. */
  validity?: ArrayLike<number>;
}): VecchiaConditioningSets {
  const {positions, queries, neighborCount, validity} = props;
  const rowCount = positions.length / 2;
  const queryCount = queries.length / 2;
  const grid = createPointGrid(positions, rowCount);
  for (let row = 0; row < rowCount; row++) {
    if (!validity || validity[row]) {
      insertPoint(grid, positions, row);
    }
  }
  const offsets = new Uint32Array(queryCount + 1);
  const neighbors = new Uint32Array(queryCount * neighborCount);
  const bestDistances = new Float64Array(neighborCount);
  const bestRows = new Uint32Array(neighborCount);
  let total = 0;
  for (let query = 0; query < queryCount; query++) {
    const x = queries[query * 2];
    const y = queries[query * 2 + 1];
    let found = 0;
    visitRings(
      grid,
      x,
      y,
      () => (found < neighborCount ? Infinity : bestDistances[neighborCount - 1]),
      sample => {
        const offsetX = x - positions[sample * 2];
        const offsetY = y - positions[sample * 2 + 1];
        const separation = offsetX * offsetX + offsetY * offsetY;
        let slot = found < neighborCount ? found : neighborCount;
        while (
          slot > 0 &&
          (bestDistances[slot - 1] > separation ||
            (bestDistances[slot - 1] === separation && bestRows[slot - 1] > sample))
        ) {
          if (slot < neighborCount) {
            bestDistances[slot] = bestDistances[slot - 1];
            bestRows[slot] = bestRows[slot - 1];
          }
          slot--;
        }
        if (slot < neighborCount) {
          bestDistances[slot] = separation;
          bestRows[slot] = sample;
          if (found < neighborCount) {
            found++;
          }
        }
      }
    );
    offsets[query + 1] = offsets[query] + found;
    for (let slot = 0; slot < found; slot++) {
      neighbors[total++] = bestRows[slot];
    }
  }
  return {offsets, neighbors: neighbors.subarray(0, total)};
}

/** Binary max-heap of `[key, row]` keyed by larger key, then lower row. */
class MaxMinHeap {
  readonly keys: Float64Array;
  readonly rows: Uint32Array;
  size = 0;

  constructor(capacity: number) {
    this.keys = new Float64Array(capacity);
    this.rows = new Uint32Array(capacity);
  }

  private isBefore(first: number, second: number): boolean {
    return (
      this.keys[first] > this.keys[second] ||
      (this.keys[first] === this.keys[second] && this.rows[first] < this.rows[second])
    );
  }

  private swap(first: number, second: number): void {
    const key = this.keys[first];
    const row = this.rows[first];
    this.keys[first] = this.keys[second];
    this.rows[first] = this.rows[second];
    this.keys[second] = key;
    this.rows[second] = row;
  }

  push(key: number, row: number): void {
    let child = this.size++;
    this.keys[child] = key;
    this.rows[child] = row;
    while (child > 0) {
      const parent = (child - 1) >> 1;
      if (!this.isBefore(child, parent)) {
        break;
      }
      this.swap(child, parent);
      child = parent;
    }
  }

  pop(): void {
    this.size--;
    if (this.size > 0) {
      this.keys[0] = this.keys[this.size];
      this.rows[0] = this.rows[this.size];
      let parent = 0;
      for (;;) {
        const left = 2 * parent + 1;
        if (left >= this.size) {
          break;
        }
        const right = left + 1;
        const child = right < this.size && this.isBefore(right, left) ? right : left;
        if (!this.isBefore(child, parent)) {
          break;
        }
        this.swap(child, parent);
        parent = child;
      }
    }
  }
}

/**
 * Exact max-min ordering (Guinness 2018): each row in turn is the one farthest from all rows chosen
 * so far, starting from the row nearest the centroid. Equal distances resolve to the lower row.
 *
 * Distances to the chosen set only fall as rows are added, so a lazy max-heap of upper bounds
 * chooses each row after re-evaluating only the candidates it pops, each by a nearest-chosen query
 * on a grid. Expected near-linear for spread data.
 *
 * @param positions Packed x, y pairs.
 * @param rowCount Number of rows. Defaults to `positions.length / 2`.
 * @returns `order`, where `order[k]` is the row at position k.
 */
export function getVecchiaMaxMinOrdering(
  positions: ArrayLike<number>,
  rowCount: number = positions.length / 2
): Uint32Array {
  const order = new Uint32Array(rowCount);
  if (rowCount === 0) {
    return order;
  }
  let centreX = 0;
  let centreY = 0;
  for (let row = 0; row < rowCount; row++) {
    centreX += positions[row * 2] / rowCount;
    centreY += positions[row * 2 + 1] / rowCount;
  }
  let first = 0;
  let firstDistance = Infinity;
  for (let row = 0; row < rowCount; row++) {
    const offsetX = positions[row * 2] - centreX;
    const offsetY = positions[row * 2 + 1] - centreY;
    const separation = offsetX * offsetX + offsetY * offsetY;
    if (separation < firstDistance) {
      firstDistance = separation;
      first = row;
    }
  }
  const grid = createPointGrid(positions, rowCount);
  const chosen = new Uint8Array(rowCount);
  order[0] = first;
  chosen[first] = 1;
  insertPoint(grid, positions, first);
  const heap = new MaxMinHeap(rowCount);
  for (let row = 0; row < rowCount; row++) {
    if (row !== first) {
      const offsetX = positions[row * 2] - positions[first * 2];
      const offsetY = positions[row * 2 + 1] - positions[first * 2 + 1];
      heap.push(offsetX * offsetX + offsetY * offsetY, row);
    }
  }
  const nearestChosen = (row: number): number => {
    const x = positions[row * 2];
    const y = positions[row * 2 + 1];
    let nearest = Infinity;
    visitRings(
      grid,
      x,
      y,
      () => nearest,
      sample => {
        const offsetX = x - positions[sample * 2];
        const offsetY = y - positions[sample * 2 + 1];
        nearest = Math.min(nearest, offsetX * offsetX + offsetY * offsetY);
      }
    );
    return nearest;
  };
  let position = 1;
  while (position < rowCount) {
    const row = heap.rows[0];
    heap.pop();
    if (chosen[row]) {
      continue;
    }
    const refreshed = nearestChosen(row);
    const isBest =
      heap.size === 0 ||
      refreshed > heap.keys[0] ||
      (refreshed === heap.keys[0] && row < heap.rows[0]);
    if (isBest) {
      order[position++] = row;
      chosen[row] = 1;
      insertPoint(grid, positions, row);
    } else {
      heap.push(refreshed, row);
    }
  }
  return order;
}
