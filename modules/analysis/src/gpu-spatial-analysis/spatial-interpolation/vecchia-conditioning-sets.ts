// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * CPU setup of a Vecchia likelihood: orderings and ordered nearest-neighbour conditioning sets. The
 * sets depend only on the coordinates and the ordering, so they are built once and reused across
 * every parameter evaluation. This runs before the GPU work, as a graph topology does.
 */

import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Maximum conditioning width supported by the Vecchia GPU routes. */
export const MAXIMUM_VECCHIA_NEIGHBOR_COUNT = 32;
/** Signed linked-list row IDs and uint32 CSR offsets must both remain representable. */
export const MAXIMUM_VECCHIA_ENTRY_COUNT = 0x7fffffff;

function isEligible(
  positions: ArrayLike<number>,
  row: number,
  validity?: ArrayLike<number>,
  times?: ArrayLike<number>
): boolean {
  return (
    (!validity || Boolean(validity[row])) &&
    Number.isFinite(positions[row * 2]) &&
    Number.isFinite(positions[row * 2 + 1]) &&
    (!times || Number.isFinite(times[row]))
  );
}

function hasEligiblePoint(
  positions: ArrayLike<number>,
  rowCount: number,
  validity?: ArrayLike<number>,
  times?: ArrayLike<number>
): boolean {
  for (let row = 0; row < rowCount; row++)
    if (isEligible(positions, row, validity, times)) return true;
  return false;
}

function validatePositions(
  id: string,
  positions: ArrayLike<number>,
  rowCount: number,
  port = 'positions'
): void {
  validateRequirement(
    id,
    positions.length % 2 === 0,
    `${port} must contain complete XY pairs; received ${positions.length}`
  );
  validateCapacity(id, 'rowCount', rowCount, 0, MAXIMUM_VECCHIA_ENTRY_COUNT - 1);
  validateRequirement(
    id,
    positions.length % 2 === 0 && positions.length >= rowCount * 2,
    `${port} must contain complete XY pairs and at least ${rowCount * 2} elements; received ${positions.length}`
  );
}

function validateNeighborhoodCapacity(
  id: string,
  rowCount: number,
  neighborCount: number,
  maximumNeighborCount = MAXIMUM_VECCHIA_NEIGHBOR_COUNT
): void {
  validateCapacity(id, 'neighborCount', neighborCount, 1, maximumNeighborCount);
  validateCapacity(
    id,
    'rowCount * neighborCount',
    rowCount * neighborCount,
    0,
    MAXIMUM_VECCHIA_ENTRY_COUNT
  );
}

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

function createPointGrid(
  positions: ArrayLike<number>,
  rowCount: number,
  validity?: ArrayLike<number>,
  times?: ArrayLike<number>
): PointGrid {
  let minimumX = Infinity;
  let minimumY = Infinity;
  let maximumX = -Infinity;
  let maximumY = -Infinity;
  let eligibleCount = 0;
  for (let row = 0; row < rowCount; row++) {
    if (!isEligible(positions, row, validity, times)) continue;
    eligibleCount++;
    minimumX = Math.min(minimumX, positions[row * 2]);
    maximumX = Math.max(maximumX, positions[row * 2]);
    minimumY = Math.min(minimumY, positions[row * 2 + 1]);
    maximumY = Math.max(maximumY, positions[row * 2 + 1]);
  }
  if (eligibleCount === 0) {
    minimumX = minimumY = maximumX = maximumY = 0;
  }
  const width = maximumX - minimumX;
  const height = maximumY - minimumY;
  validateRequirement(
    'VecchiaPointGrid',
    Number.isFinite(width) && Number.isFinite(height),
    `positions extent must be finite; received ${width} by ${height}`
  );
  const cellCount = Math.max(1, Math.round(eligibleCount / ROWS_PER_CELL));
  const span = Math.max(width, height);
  // Scale before multiplying to avoid overflow/underflow for large/tiny finite extents.
  const cellSize =
    span === 0
      ? 1
      : Math.max(
          span * Math.sqrt(((width / span) * (height / span)) / cellCount),
          span / 4096,
          Number.MIN_VALUE
        );
  const columns = Math.floor(width / cellSize) + 1;
  const rows = Math.floor(height / cellSize) + 1;
  validateCapacity(
    'VecchiaPointGrid',
    'columns * rows',
    columns * rows,
    1,
    MAXIMUM_VECCHIA_ENTRY_COUNT
  );
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
  return Math.max(0, Math.min(grid.columns - 1, Math.floor((x - grid.minimumX) / grid.cellSize)));
}

function getCellRow(grid: PointGrid, y: number): number {
  return Math.max(0, Math.min(grid.rows - 1, Math.floor((y - grid.minimumY) / grid.cellSize)));
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
        if (firstColumn >= 0 && firstColumn < grid.columns) {
          for (
            let sample = grid.heads[cellRow * grid.columns + firstColumn];
            sample >= 0;
            sample = grid.next[sample]
          ) {
            visit(sample);
          }
        }
        if (lastColumn >= 0 && lastColumn < grid.columns) {
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
  /** Logical row count, requested width (even for short spans), and readable CSR entry count. */
  counts?: {rowCount: number; neighborCount: number; entryCount: number};
  formats?: {offsets: 'uint32'; neighbors: 'uint32'};
};

/** Builder result with guaranteed metadata; legacy CSR values remain assignable to VecchiaConditioningSets. */
export type PreparedVecchiaConditioningSets = VecchiaConditioningSets & {
  counts: NonNullable<VecchiaConditioningSets['counts']>;
  formats: NonNullable<VecchiaConditioningSets['formats']>;
};

function prepareSets(
  offsets: Uint32Array,
  neighbors: Uint32Array,
  neighborCount: number
): PreparedVecchiaConditioningSets {
  return {
    offsets,
    neighbors,
    counts: {rowCount: offsets.length - 1, neighborCount, entryCount: neighbors.length},
    formats: {offsets: 'uint32', neighbors: 'uint32'}
  };
}

/** Maps CPU CSR arrays to the two named uint32 consumer ports using the caller's uploader. */
export function uploadVecchiaNeighborhoods<View>(
  sets: PreparedVecchiaConditioningSets,
  uploadUint32: (values: Uint32Array, format: 'uint32') => View
): {neighborhoodOffsets: View; neighborhoods: View; neighborCount: number} {
  return {
    neighborhoodOffsets: uploadUint32(sets.offsets, 'uint32'),
    neighborhoods: uploadUint32(sets.neighbors, 'uint32'),
    neighborCount: sets.counts.neighborCount
  };
}

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
  /** Optional setup mask: nonzero includes (including negative integers); zero or NaN excludes. */
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
 * The setup route accepts at most 2,147,483,647 scratch entries (rowCount * neighborCount)
 * and grid buckets, and at most 2,147,483,646 rows, bounded by MAXIMUM_VECCHIA_ENTRY_COUNT.
 */
export function buildVecchiaConditioningSets(
  props: VecchiaConditioningSetProps
): PreparedVecchiaConditioningSets {
  const {positions, neighborCount, order, validity, times, timeScale = 0} = props;
  const rowCount = props.rowCount ?? positions.length / 2;
  const id = 'buildVecchiaConditioningSets';
  validatePositions(id, positions, rowCount);
  validateNeighborhoodCapacity(id, rowCount, neighborCount);
  for (const [name, values] of [
    ['validity', validity],
    ['times', times]
  ] as const) {
    validateRequirement(
      id,
      !values || values.length >= rowCount,
      `${name} length must cover selected rowCount ${rowCount}; received ${values?.length}`
    );
  }
  validateRequirement(
    id,
    !times || (props.timeScale !== undefined && Number.isFinite(timeScale)),
    `timeScale must be explicit and finite when times is supplied; received ${props.timeScale}`
  );
  if (order) {
    validateRequirement(
      id,
      order.length === rowCount,
      `order length must equal selected rowCount ${rowCount}; received ${order.length}`
    );
    const seen = new Set<number>();
    for (let position = 0; position < rowCount; position++) {
      const row = order[position];
      validateRequirement(
        id,
        Number.isInteger(row) && row >= 0 && row < rowCount && !seen.has(row),
        `order must be an exact permutation of [0, ${rowCount}); received ${row} at position ${position}`
      );
      seen.add(row);
    }
  }
  if (!hasEligiblePoint(positions, rowCount, validity, times))
    return prepareSets(new Uint32Array(rowCount + 1), new Uint32Array(), neighborCount);
  const grid = createPointGrid(positions, rowCount, validity, times);
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
    if (!isEligible(positions, row, validity, times)) {
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
  return prepareSets(offsets, packed, neighborCount);
}

/**
 * CSR of the data rows nearest each query, `[offsets, neighbors]` as for the conditioning sets.
 * This setup route accepts at most 2,147,483,647 scratch entries (queryCount * neighborCount)
 * and grid buckets, and at most 2,147,483,646 data or query rows, bounded by
 * MAXIMUM_VECCHIA_ENTRY_COUNT. Prediction setup supports positive integer widths beyond 32,
 * including conformal calibration widths through 256; each downstream GPU solve enforces its own limit.
 */
export function buildVecchiaPredictionNeighborhoods(props: {
  /** Packed data x, y pairs. */
  positions: ArrayLike<number>;
  /** Packed query x, y pairs. */
  queries: ArrayLike<number>;
  neighborCount: number;
  /** Optional setup mask: nonzero includes (including negative integers); zero or NaN excludes. */
  validity?: ArrayLike<number>;
}): PreparedVecchiaConditioningSets {
  const {positions, queries, neighborCount, validity} = props;
  const rowCount = positions.length / 2;
  const queryCount = queries.length / 2;
  const id = 'buildVecchiaPredictionNeighborhoods';
  validatePositions(id, positions, rowCount);
  validatePositions(id, queries, queryCount, 'queries');
  validateNeighborhoodCapacity(id, queryCount, neighborCount, MAXIMUM_VECCHIA_ENTRY_COUNT);
  validateRequirement(
    id,
    !validity || validity.length === rowCount,
    `validity length must equal data rowCount ${rowCount}; received ${validity?.length}`
  );
  for (let query = 0; query < queryCount; query++)
    validateRequirement(
      id,
      isEligible(queries, query),
      `queries must contain finite XY at row ${query}; received ${queries[query * 2]}, ${queries[query * 2 + 1]}`
    );
  if (queryCount === 0 || !hasEligiblePoint(positions, rowCount, validity))
    return prepareSets(new Uint32Array(queryCount + 1), new Uint32Array(), neighborCount);
  const grid = createPointGrid(positions, rowCount, validity);
  for (let row = 0; row < rowCount; row++) {
    if (isEligible(positions, row, validity)) {
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
  return prepareSets(offsets, neighbors.subarray(0, total), neighborCount);
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
  validatePositions('getVecchiaMaxMinOrdering', positions, rowCount);
  for (let row = 0; row < rowCount; row++)
    validateRequirement(
      'getVecchiaMaxMinOrdering',
      isEligible(positions, row),
      `positions must contain finite XY at row ${row}; received ${positions[row * 2]}, ${positions[row * 2 + 1]}`
    );
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
