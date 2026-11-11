// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// A small table of cells drawn as polygons inside a world rectangle, so a distance band by time
// band table sits beside the event map. Row 0 is at the top, cell = row * columns + column, the
// layout of the near-repeat and space-time K tables (distance-major).

import type {RunBuilder} from '../../../define-run';

export type CellTable = {
  columns: number;
  rows: number;
  /** [left, bottom, right, top] in world units. */
  rectangle: readonly [number, number, number, number];
  positions: Float32Array;
  indices: Uint32Array;
  owners: Uint32Array;
  triangleCount: number;
};

export function createCellTable(
  columns: number,
  rows: number,
  rectangle: readonly [number, number, number, number],
  gap = 0.004
): CellTable {
  const [left, bottom, right, top] = rectangle;
  const cellWidth = (right - left) / columns;
  const cellHeight = (top - bottom) / rows;
  const positions: number[] = [];
  const indices: number[] = [];
  const owners: number[] = [];
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      const x0 = left + column * cellWidth + gap;
      const x1 = left + (column + 1) * cellWidth - gap;
      const y1 = top - row * cellHeight - gap;
      const y0 = top - (row + 1) * cellHeight + gap;
      const base = positions.length / 2;
      positions.push(x0, y0, x1, y0, x1, y1, x0, y1);
      indices.push(base, base + 1, base + 2, base, base + 2, base + 3);
      owners.push(row * columns + column, row * columns + column);
    }
  }
  return {
    columns,
    rows,
    rectangle,
    positions: Float32Array.from(positions),
    indices: Uint32Array.from(indices),
    owners: Uint32Array.from(owners),
    triangleCount: owners.length
  };
}

/** The table's geometry buffers for a `polygons` layer. */
export function uploadCellTable(builder: RunBuilder, table: CellTable, id = 'table') {
  return {
    positions: builder.upload(`${id}-positions`, 'float32x2', table.positions),
    indices: builder.upload(`${id}-indices`, 'uint32', table.indices),
    owners: builder.upload(`${id}-owners`, 'uint32', table.owners)
  };
}

/** The cell under (x, y), clamped to the table. */
export function getTableCell(table: CellTable, x: number, y: number) {
  const [left, bottom, right, top] = table.rectangle;
  const column = Math.min(
    table.columns - 1,
    Math.max(0, Math.floor(((x - left) / (right - left)) * table.columns))
  );
  const row = Math.min(
    table.rows - 1,
    Math.max(0, Math.floor(((top - y) / (top - bottom)) * table.rows))
  );
  return {row, column, index: row * table.columns + column};
}

/** The world box of a cell, for a `box` glyph. */
export function getCellBox(table: CellTable, row: number, column: number) {
  const [left, bottom, right, top] = table.rectangle;
  const cellWidth = (right - left) / table.columns;
  const cellHeight = (top - bottom) / table.rows;
  return {
    minimum: [left + column * cellWidth, top - (row + 1) * cellHeight] as [number, number],
    maximum: [left + (column + 1) * cellWidth, top - row * cellHeight] as [number, number]
  };
}
