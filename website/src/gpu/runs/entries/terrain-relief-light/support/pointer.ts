// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Pointer helpers for the terrain, relief and light runs: the observer cell under the pointer and
// the glyph geometry of cell windows.

import type {RasterInput, World} from '../../../inputs';
import type {Point, RunPointer} from '../../../interaction';
import {clamp} from '../../../interaction';
import {getCellOfPercentage, getCellWorldPosition} from './terrain';

/**
 * The observer under the pointer, as the percentage parameters the sight-line runs bind to and as
 * the snapped cell with its world position.
 */
export function getObserverUnderPointer(
  terrain: RasterInput,
  world: World,
  pointer: RunPointer
): {
  observerColumn: number;
  observerRow: number;
  column: number;
  row: number;
  position: [number, number];
} {
  const observerColumn = clamp(Math.round((pointer.x / world.width) * 100), 0, 100);
  const observerRow = clamp(Math.round((1 - pointer.y / world.height) * 100), 0, 100);
  const [column, row] = getCellOfPercentage(terrain, observerColumn, observerRow);
  return {
    observerColumn,
    observerRow,
    column,
    row,
    position: getCellWorldPosition(terrain, world, column, row)
  };
}

/** The pointer position that reproduces the default observer percentages. */
export function getObserverRest(
  parameters: readonly {id: string; default: number}[]
): [number, number] {
  const get = (id: string) => parameters.find(parameter => parameter.id === id)?.default ?? 50;
  return [get('observerColumn') / 100, get('observerRow') / 100];
}

/** A square window of `halfCells` cells around a cell center. */
export function getCellWindow(
  center: Point,
  halfCells: number,
  worldPerCell: number
): {minimum: Point; maximum: Point} {
  const half = (halfCells + 0.5) * worldPerCell;
  return {
    minimum: [center[0] - half, center[1] - half],
    maximum: [center[0] + half, center[1] + half]
  };
}
