// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Display adapters for dense (cell, period) cubes. Small GPU kernels rearrange the operation's
// output buffers into one float32 raster the framework draws (a cells-by-periods heatmap with a
// blank block beside it for a map) and a per-cell slice of one period. They read the cube on the
// GPU; nothing is read back. Empty bins and the blank block hold NaN, which the raster leaves muted.

import {createWGSLKernelNode} from '@luma-contributors/analysis/analysis-core';
import type {RunBuffer, RunBuilder} from '../../../define-run';
import type {World} from '../../../inputs';

/** Blank raster columns left of the heatmap, where the map of the cells is drawn. */
export const MAP_COLUMNS = 12;

/** Shape of a cube heatmap: one raster row per cell, one column per period, a blank block left. */
export type CubeFrame = {
  cellCount: number;
  periodCount: number;
  /** Raster columns: the map block plus the periods. */
  columns: number;
  /** Raster rows. */
  rows: number;
  /** Blank columns before the first period. */
  left: number;
};

export function getCubeFrame(cellCount: number, periodCount: number): CubeFrame {
  return {
    cellCount,
    periodCount,
    columns: MAP_COLUMNS + periodCount,
    rows: cellCount,
    left: MAP_COLUMNS
  };
}

/** World rectangle of the heatmap block: [minimumX, minimumY, maximumX, maximumY]. */
export function getHeatmapBounds(frame: CubeFrame, world: World): [number, number, number, number] {
  return [(world.width * frame.left) / frame.columns, 0, world.width, 1];
}

/** The (cell, period) bin under a world position, clamped to the heatmap block. */
export function getBinAt(frame: CubeFrame, world: World, x: number, y: number) {
  const bounds = getHeatmapBounds(frame, world);
  const period = Math.min(
    frame.periodCount - 1,
    Math.max(0, Math.floor(((x - bounds[0]) / (bounds[2] - bounds[0])) * frame.periodCount))
  );
  const cell = Math.min(
    frame.cellCount - 1,
    Math.max(0, Math.floor(((1 - y) / (bounds[3] - bounds[1])) * frame.cellCount))
  );
  return {cell, period, isInside: x >= bounds[0]};
}

/** World rectangle of one bin, [[minimumX, minimumY], [maximumX, maximumY]]. */
export function getBinBox(
  frame: CubeFrame,
  world: World,
  cell: number,
  period: number
): {minimum: [number, number]; maximum: [number, number]} {
  const bounds = getHeatmapBounds(frame, world);
  const width = (bounds[2] - bounds[0]) / frame.periodCount;
  const height = (bounds[3] - bounds[1]) / frame.cellCount;
  return {
    minimum: [bounds[0] + period * width, 1 - (cell + 1) * height],
    maximum: [bounds[0] + (period + 1) * width, 1 - cell * height]
  };
}

/** Cell centers of a `columns` by `rows` lattice laid over the map block, row 0 at the north. */
export function createMapCenters(
  frame: CubeFrame,
  world: World,
  columns: number,
  rows: number
): Float32Array {
  const [minimumX, maximumX] = [
    world.width * 0.035,
    world.width * (frame.left / frame.columns) * 0.94
  ];
  const [minimumY, maximumY] = [0.14, 0.86];
  const centers = new Float32Array(columns * rows * 2);
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      const cell = row * columns + column;
      centers[cell * 2] = minimumX + ((column + 0.5) / columns) * (maximumX - minimumX);
      centers[cell * 2 + 1] = maximumY - ((row + 0.5) / rows) * (maximumY - minimumY);
    }
  }
  return centers;
}

/** Statistic shown: row 0 of the selection buffer. */
export const CUBE_STATISTICS = ['counts', 'sums', 'means'] as const;

/** The selection buffer: statistic, selected period, selected cell. */
export function addCubeSelection(builder: RunBuilder): RunBuffer<'uint32'> {
  return builder.allocate('selection', 'uint32', 4);
}

const NAN_WGSL = 'bitcast<f32>(0x7fc00000u | (selection[selectionOffset] & 0u))';

/**
 * Cube heatmap raster: row `cell`, column `left + period` holds the statistic of the cube row
 * `cell * periodCount + period` (counts, sums or means by `selection[0]`); empty bins are NaN.
 */
export function addCubeHeatmap(
  builder: RunBuilder,
  options: {
    frame: CubeFrame;
    counts: RunBuffer<'uint32'>;
    sums: RunBuffer<'float32'>;
    means: RunBuffer<'float32'>;
    selection: RunBuffer<'uint32'>;
  }
): RunBuffer<'float32'> {
  const {frame} = options;
  const raster = builder.allocate('heatmap', 'float32', frame.columns * frame.rows);
  builder.graph.add(
    createWGSLKernelNode(builder.graph, {
      id: 'cube-heatmap',
      operation: 'display-cube-heatmap',
      invocationCount: frame.columns * frame.rows,
      bindings: [
        {name: 'counts', view: options.counts.view, type: 'u32', access: 'read'},
        {name: 'sums', view: options.sums.view, type: 'f32', access: 'read'},
        {name: 'means', view: options.means.view, type: 'f32', access: 'read'},
        {name: 'selection', view: options.selection.view, type: 'u32', access: 'read'},
        {name: 'raster', view: raster.view, type: 'f32', access: 'read_write'}
      ],
      body: `
  let row = index / ${frame.columns}u;
  let column = index % ${frame.columns}u;
  var result = ${NAN_WGSL};
  if (column >= ${frame.left}u) {
    let bin = row * ${frame.periodCount}u + (column - ${frame.left}u);
    let count = counts[countsOffset + bin];
    if (count > 0u) {
      let statistic = selection[selectionOffset];
      if (statistic == 0u) {
        result = f32(count);
      } else if (statistic == 1u) {
        result = sums[sumsOffset + bin];
      } else {
        result = means[meansOffset + bin];
      }
    }
  }
  raster[rasterOffset + index] = result;`
    })
  );
  return raster;
}

/** The selected period of the cube for every cell: the values of the map dots. */
export function addPeriodSlice(
  builder: RunBuilder,
  options: {
    frame: CubeFrame;
    counts: RunBuffer<'uint32'>;
    sums: RunBuffer<'float32'>;
    means: RunBuffer<'float32'>;
    selection: RunBuffer<'uint32'>;
  }
): RunBuffer<'float32'> {
  const {frame} = options;
  const slice = builder.allocate('period-slice', 'float32', frame.cellCount);
  builder.graph.add(
    createWGSLKernelNode(builder.graph, {
      id: 'period-slice',
      operation: 'display-period-slice',
      invocationCount: frame.cellCount,
      bindings: [
        {name: 'counts', view: options.counts.view, type: 'u32', access: 'read'},
        {name: 'sums', view: options.sums.view, type: 'f32', access: 'read'},
        {name: 'means', view: options.means.view, type: 'f32', access: 'read'},
        {name: 'selection', view: options.selection.view, type: 'u32', access: 'read'},
        {name: 'slice', view: slice.view, type: 'f32', access: 'read_write'}
      ],
      body: `
  let bin = index * ${frame.periodCount}u + selection[selectionOffset + 1u];
  let count = counts[countsOffset + bin];
  var result = ${NAN_WGSL};
  if (count > 0u) {
    let statistic = selection[selectionOffset];
    if (statistic == 0u) {
      result = f32(count);
    } else if (statistic == 1u) {
      result = sums[sumsOffset + bin];
    } else {
      result = means[meansOffset + bin];
    }
  }
  slice[sliceOffset + index] = result;`
    })
  );
  return slice;
}

/** Blank columns between the two blocks of a layout comparison. */
export const LAYOUT_GAP_COLUMNS = 2;

export type LayoutFrame = {
  cellCount: number;
  periodCount: number;
  /** Raster columns: the cell-major block, a gap, then the period-major block. */
  columns: number;
  rows: number;
};

export function getLayoutFrame(cellCount: number, periodCount: number): LayoutFrame {
  return {
    cellCount,
    periodCount,
    columns: periodCount + LAYOUT_GAP_COLUMNS + cellCount,
    rows: Math.max(cellCount, periodCount)
  };
}

/** World rectangle of one bin in the cell-major block (left) or the period-major block (right). */
export function getLayoutBinBox(
  frame: LayoutFrame,
  world: World,
  cell: number,
  period: number,
  block: 'cell-major' | 'period-major'
): {minimum: [number, number]; maximum: [number, number]} {
  const width = world.width / frame.columns;
  const height = 1 / frame.rows;
  const column = block === 'cell-major' ? period : frame.periodCount + LAYOUT_GAP_COLUMNS + cell;
  const row = block === 'cell-major' ? cell : period;
  return {
    minimum: [column * width, 1 - (row + 1) * height],
    maximum: [(column + 1) * width, 1 - row * height]
  };
}

/** Which block and bin a world position falls in, or null in the gap and below a block. */
export function getLayoutBinAt(frame: LayoutFrame, world: World, x: number, y: number) {
  const column = Math.floor((x / world.width) * frame.columns);
  const row = Math.floor((1 - y) * frame.rows);
  if (column < frame.periodCount) {
    return row < frame.cellCount ? {cell: row, period: column} : null;
  }
  const cell = column - frame.periodCount - LAYOUT_GAP_COLUMNS;
  if (cell >= 0 && cell < frame.cellCount && row < frame.periodCount) {
    return {cell, period: row};
  }
  return null;
}

/**
 * Both layouts of one cube in one raster: the cell-major buffer on the left (a row per cell) and the
 * period-major buffer on the right (a row per period). Empty bins are NaN.
 */
export function addLayoutComposite(
  builder: RunBuilder,
  options: {
    frame: LayoutFrame;
    cellMajor: RunBuffer<'uint32'> | RunBuffer<'float32'>;
    periodMajor: RunBuffer<'uint32'> | RunBuffer<'float32'>;
  }
): RunBuffer<'float32'> {
  const {frame} = options;
  const isInteger = options.cellMajor.format === 'uint32';
  const raster = builder.allocate('layouts', 'float32', frame.columns * frame.rows);
  const read = (name: string, bin: string) =>
    isInteger ? `f32(${name}[${name}Offset + ${bin}])` : `${name}[${name}Offset + ${bin}]`;
  const empty = (name: string, bin: string) =>
    isInteger
      ? `${name}[${name}Offset + ${bin}] == 0u`
      : `!(${name}[${name}Offset + ${bin}] == ${name}[${name}Offset + ${bin}])`;
  builder.graph.add(
    createWGSLKernelNode(builder.graph, {
      id: 'layout-composite',
      operation: 'display-layout-composite',
      invocationCount: frame.columns * frame.rows,
      bindings: [
        {
          name: 'cellMajor',
          view: options.cellMajor.view,
          type: isInteger ? 'u32' : 'f32',
          access: 'read'
        },
        {
          name: 'periodMajor',
          view: options.periodMajor.view,
          type: isInteger ? 'u32' : 'f32',
          access: 'read'
        },
        {name: 'raster', view: raster.view, type: 'f32', access: 'read_write'}
      ],
      body: `
  let row = index / ${frame.columns}u;
  let column = index % ${frame.columns}u;
  var result = bitcast<f32>(0x7fc00000u | (row & 0u));
  if (column < ${frame.periodCount}u) {
    if (row < ${frame.cellCount}u) {
      let bin = row * ${frame.periodCount}u + column;
      if (!(${empty('cellMajor', 'bin')})) { result = ${read('cellMajor', 'bin')}; }
    }
  } else if (column >= ${frame.periodCount + LAYOUT_GAP_COLUMNS}u) {
    let cell = column - ${frame.periodCount + LAYOUT_GAP_COLUMNS}u;
    if (row < ${frame.periodCount}u) {
      let bin = row * ${frame.cellCount}u + cell;
      if (!(${empty('periodMajor', 'bin')})) { result = ${read('periodMajor', 'bin')}; }
    }
  }
  raster[rasterOffset + index] = result;`
    })
  );
  return raster;
}
