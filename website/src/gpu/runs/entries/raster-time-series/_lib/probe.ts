// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// The pixel under the pointer, kept inside the map, and the glyph that outlines it.

import {getRasterCell, getRasterCellCenter} from '../../../pick';
import type {RunGlyph} from '../../../interaction';
import type {MapGrid} from './stacks';

type World = {width: number; height: number};

export type ProbeCell = {column: number; row: number; index: number};

/** The map pixel under the pointer; the chart area snaps to the nearest map column. */
export function getProbeCell(
  grid: MapGrid,
  world: World,
  pointer: {x: number; y: number}
): ProbeCell {
  const cell = getRasterCell(
    grid.columns,
    grid.rows,
    [0, 0, world.width, world.height],
    pointer.x,
    pointer.y
  );
  const column = Math.max(grid.chartColumns, cell.column);
  return {column, row: cell.row, index: cell.row * grid.columns + column};
}

export function getCellOutline(grid: MapGrid, world: World, cell: ProbeCell): RunGlyph {
  const [x, y] = getRasterCellCenter(
    grid.columns,
    grid.rows,
    [0, 0, world.width, world.height],
    cell.column,
    cell.row
  );
  const halfWidth = world.width / grid.columns / 2;
  const halfHeight = world.height / grid.rows / 2;
  return {
    shape: 'box',
    minimum: [x - halfWidth, y - halfHeight],
    maximum: [x + halfWidth, y + halfHeight]
  };
}

/** Pointer position (0..1 across the canvas, y down) that lands on a map cell, for `rest`. */
export function getRestPosition(grid: MapGrid, mapUnitX: number, unitY: number): [number, number] {
  return [(grid.chartColumns + mapUnitX * grid.mapColumns) / grid.columns, unitY];
}

/** The pixel as `cell [column, row]` within the map, for the end of a readout. */
export function formatCell(grid: MapGrid, cell: ProbeCell): string {
  return `cell [${cell.column - grid.chartColumns}, ${cell.row}]`;
}
