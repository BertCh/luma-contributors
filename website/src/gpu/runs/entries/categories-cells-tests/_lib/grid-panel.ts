// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Quad meshes for chart panels the raster layer cannot place: a square grid for a table, bars with a
// category per bar, and a translucent backing panel under a chart that sits on a map.

import type {RunBuffer, RunBuilder, RunLayer} from '../../../define-run';

export type Rectangle = readonly [number, number, number, number];
type World = {width: number; height: number};

function writeQuad(
  positions: Float32Array,
  quad: number,
  x0: number,
  y0: number,
  x1: number,
  y1: number
) {
  positions.set([x0, y0, x1, y0, x1, y1, x0, y1], quad * 8);
}

function createQuadIndices(count: number): {indices: Uint32Array; owners: Uint32Array} {
  const indices = new Uint32Array(count * 6);
  const owners = new Uint32Array(count * 2);
  for (let quad = 0; quad < count; quad++) {
    indices.set(
      [0, 1, 2, 0, 2, 3].map(corner => quad * 4 + corner),
      quad * 6
    );
    owners.set([quad, quad], quad * 2);
  }
  return {indices, owners};
}

export type GridPanel = {
  size: number;
  positions: Float32Array;
  indices: Uint32Array;
  owners: Uint32Array;
  /** World rectangle of one table cell (row 0 at the top). */
  getCell(row: number, column: number): {minimum: [number, number]; maximum: [number, number]};
  /** The table cell at a world position, or null. */
  findCell(x: number, y: number): {row: number; column: number} | null;
};

/** A `size` x `size` table of squares centered in `rectangle` (fractions of the view). */
export function createGridPanel(size: number, rectangle: Rectangle, world: World): GridPanel {
  const width = (rectangle[2] - rectangle[0]) * world.width;
  const height = (rectangle[3] - rectangle[1]) * world.height;
  const side = Math.min(width, height) / size;
  const left = rectangle[0] * world.width + (width - side * size) / 2;
  const top = rectangle[3] * world.height - (height - side * size) / 2;
  const positions = new Float32Array(size * size * 8);
  const getCell = (row: number, column: number) => ({
    minimum: [left + column * side, top - (row + 1) * side] as [number, number],
    maximum: [left + (column + 1) * side, top - row * side] as [number, number]
  });
  for (let row = 0; row < size; row++) {
    for (let column = 0; column < size; column++) {
      const {minimum, maximum} = getCell(row, column);
      writeQuad(positions, row * size + column, minimum[0], minimum[1], maximum[0], maximum[1]);
    }
  }
  return {
    size,
    positions,
    ...createQuadIndices(size * size),
    getCell,
    findCell(x, y) {
      const column = Math.floor((x - left) / side);
      const row = Math.floor((top - y) / side);
      return row >= 0 && row < size && column >= 0 && column < size ? {row, column} : null;
    }
  };
}

/** Positions of `count` bars in `rectangle`; `heights` are fractions (0 to 1) of the plot height. */
export function getBarPositions(
  count: number,
  heights: ArrayLike<number>,
  rectangle: Rectangle,
  world: World
): Float32Array {
  const positions = new Float32Array(count * 8);
  const slot = ((rectangle[2] - rectangle[0]) * world.width) / count;
  const bottom = rectangle[1] * world.height;
  const height = (rectangle[3] - rectangle[1]) * world.height;
  for (let bar = 0; bar < count; bar++) {
    const x = rectangle[0] * world.width + bar * slot;
    writeQuad(
      positions,
      bar,
      x + slot * 0.1,
      bottom,
      x + slot * 0.9,
      bottom + Math.max(0, Math.min(1, heights[bar])) * height
    );
  }
  return positions;
}

export type BarMesh = {
  positions: RunBuffer<'float32x2'>;
  indices: RunBuffer<'uint32'>;
  owners: RunBuffer<'uint32'>;
  /** One value per bar, drawn with the categorical ramp: the bar's class. */
  classes: RunBuffer<'uint32'>;
  count: number;
};

/** Bars whose heights are rewritten with `positions.write(getBarPositions(...))`. */
export function allocateBarMesh(builder: RunBuilder, id: string, count: number): BarMesh {
  const {indices, owners} = createQuadIndices(count);
  return {
    positions: builder.allocate(`${id}-positions`, 'float32x2', count * 4),
    indices: builder.upload(`${id}-indices`, 'uint32', indices),
    owners: builder.upload(`${id}-owners`, 'uint32', owners),
    classes: builder.upload(
      'classes',
      'uint32',
      Uint32Array.from({length: count}, (_, bar) => bar)
    ),
    count
  };
}

export type Panel = {
  positions: RunBuffer<'float32x2'>;
  indices: RunBuffer<'uint32'>;
  owners: RunBuffer<'uint32'>;
  values: RunBuffer<'float32'>;
};

/** A backing quad slightly larger than `rectangle`. */
export function uploadPanel(
  builder: RunBuilder,
  rectangle: Rectangle,
  world: World,
  margin = 0.02
): Panel {
  const positions = new Float32Array(8);
  writeQuad(
    positions,
    0,
    (rectangle[0] - margin) * world.width,
    (rectangle[1] - margin) * world.height,
    (rectangle[2] + margin) * world.width,
    (rectangle[3] + margin) * world.height
  );
  return {
    positions: builder.upload('panel-positions', 'float32x2', positions),
    indices: builder.upload('panel-indices', 'uint32', Uint32Array.of(0, 1, 2, 0, 2, 3)),
    owners: builder.upload('panel-owners', 'uint32', Uint32Array.of(0, 0)),
    values: builder.upload('panel-values', 'float32', Float32Array.of(0))
  };
}

/** The panel in the theme background at 88% opacity, drawn under a chart. */
export function drawPanel(panel: Panel, background: readonly number[]): RunLayer {
  return {
    layer: 'polygons',
    positions: panel.positions,
    indices: panel.indices,
    owners: panel.owners,
    values: panel.values,
    triangleCount: 2,
    color: [background[0], background[1], background[2], 0.88]
  };
}
