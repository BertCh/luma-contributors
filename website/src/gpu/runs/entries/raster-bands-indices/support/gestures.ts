// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Pointer helpers shared by the raster runs: swipe curtains, the raster cell under the pointer,
// cell outlines, and display-only graph colouring of label rasters.

import type {RunGlyph, RunInteractionResult, RunPointer} from '../../../interaction';
import {getRasterCell, getRasterCellCenter, type RasterCell} from '../../../pick';

type World = {width: number; height: number};
type Grid = {columns: number; rows: number};

/** World x of the swipe curtain in `draw`: the pointer, or the middle at rest. */
export function getCurtainX(interaction: RunInteractionResult, world: World): number {
  return interaction.curtain ?? world.width / 2;
}

/** The swipe result for a pointer: the curtain follows its x, with a readout and extra glyphs. */
export function getSwipeResult(
  pointer: RunPointer,
  readout: string,
  glyphs: readonly RunGlyph[] = []
): RunInteractionResult {
  return {curtain: pointer.x, glyphs: [{shape: 'curtain', x: pointer.x}, ...glyphs], readout};
}

/** The raster cell under the pointer; the raster fills the drawing world. */
export function getCellAt(grid: Grid, world: World, pointer: {x: number; y: number}): RasterCell {
  return getRasterCell(
    grid.columns,
    grid.rows,
    [0, 0, world.width, world.height],
    pointer.x,
    pointer.y
  );
}

export function getCellCenter(
  grid: Grid,
  world: World,
  column: number,
  row: number
): [number, number] {
  return getRasterCellCenter(
    grid.columns,
    grid.rows,
    [0, 0, world.width, world.height],
    column,
    row
  );
}

/** Outline of the (2 * radius + 1) cells square centred on a cell. */
export function getCellBox(
  grid: Grid,
  world: World,
  column: number,
  row: number,
  radius = 0
): RunGlyph {
  const cellWidth = world.width / grid.columns;
  const cellHeight = world.height / grid.rows;
  const [x, y] = getCellCenter(grid, world, column, row);
  return {
    shape: 'box',
    minimum: [x - (radius + 0.5) * cellWidth, y - (radius + 0.5) * cellHeight],
    maximum: [x + (radius + 0.5) * cellWidth, y + (radius + 0.5) * cellHeight]
  };
}

/**
 * Display colours for the labels of a raster, as a table indexed by label: labels that touch (4-
 * neighbour adjacency) get different colours by greedy graph colouring, so neighbouring patches
 * never share a hue. Colour 0 is for label 0 (background); the others run 1 to `colorCount`.
 */
export function createLabelColors(
  labels: Uint32Array,
  columns: number,
  rows: number,
  colorCount = 7
): Uint32Array {
  let maximum = 0;
  for (const label of labels) maximum = Math.max(maximum, label);
  const neighbors = new Map<number, Set<number>>();
  const link = (first: number, second: number) => {
    if (first === second || first === 0 || second === 0) return;
    if (!neighbors.has(first)) neighbors.set(first, new Set());
    if (!neighbors.has(second)) neighbors.set(second, new Set());
    neighbors.get(first)!.add(second);
    neighbors.get(second)!.add(first);
  };
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      const label = labels[row * columns + column];
      if (column + 1 < columns) link(label, labels[row * columns + column + 1]);
      if (row + 1 < rows) link(label, labels[(row + 1) * columns + column]);
    }
  }
  const table = new Uint32Array(maximum + 1).fill(1);
  table[0] = 0;
  const assigned = new Set<number>();
  const ordered = [...neighbors.keys()].sort(
    (first, second) => neighbors.get(second)!.size - neighbors.get(first)!.size
  );
  for (const label of ordered) {
    const taken = new Set<number>();
    for (const other of neighbors.get(label)!) if (assigned.has(other)) taken.add(table[other]);
    let color = 1;
    while (taken.has(color) && color < colorCount) color++;
    table[label] = color;
    assigned.add(label);
  }
  return table;
}

/** The display colour of every cell of a label raster. */
export function colorLabels(labels: Uint32Array, table: Uint32Array): Uint32Array {
  return Uint32Array.from(labels, label => (label < table.length ? table[label] : 1));
}

/** Copies a mapped GPU read into a typed array. */
export function toUint32Array(bytes: Uint8Array): Uint32Array {
  return new Uint32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
}
