// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Shared inputs for the terrain and hydrology runs: the Grand Canyon DEM whose ground coordinates are the
// drawing world (so reach polylines can be drawn directly), and the real GPUTerrainFlow that the
// downstream operations (watersheds, stream order, HAND, ...) consume.

import {
  getGPUTerrainFlowParameterValues,
  GPUTerrainFlow,
  type GPUTerrainFlowAccumulationUnits,
  type GPUTerrainFlowSettings,
  type GPUTerrainFlowRouting
} from '@luma-contributors/analysis/gpu-terrain';
import {
  makeGPURasterBand,
  makeGPURasterMetadata
} from '@luma-contributors/analysis/raster-analysis';
import type {RunBuffer, RunBuilder, RunInputContext} from '../../../define-run';
import {getRasterCell, type RasterCell} from '../../../pick';
import {sampleElevation, normalizeElevation} from '../../../../data/elevation';

type RasterGrid = ReturnType<typeof makeGPURasterMetadata>;

export type TerrainInputs = {
  columns: number;
  rows: number;
  cellCount: number;
  /** Cell size in world units (ground coordinates equal the drawing world). */
  cellSize: number;
  grid: RasterGrid;
  /** Elevation in world units, row 0 at the top. Includes closed depressions. */
  elevation: Float32Array;
  /** Strict local minima away from the border, as cell indices. */
  pitCells: Uint32Array;
  /** Highest elevation, for color ranges. */
  elevationRange: [number, number];
};

/** Grand Canyon terrain scaled into the flow graph’s local coordinate system. */
export function createTerrainInputs(
  context: RunInputContext,
  rows = 48,
  fixedColumns?: number
): TerrainInputs {
  const {world} = context;
  const columns = fixedColumns ?? Math.max(24, Math.round(rows * world.width));
  const cellSize = world.height / rows;
  const measured = sampleElevation(columns, rows);
  const elevation = measured.elevation.map(value => (value * cellSize) / measured.cellSize);
  return assembleTerrainInputs(columns, rows, cellSize, elevation);
}

/**
 * A gentle slope dotted with craters: each is a closed pit ringed by a raised rim, so draining it
 * takes a real cut through the rim (breaching) or a fill up to the spill level.
 */
export function createCraterTerrainInputs(
  context: RunInputContext,
  rows = 40,
  craterColumns = 5,
  craterRows = 3,
  tilt = 0.4,
  noise = 0.015
): TerrainInputs {
  const {random, world} = context;
  const columns = Math.max(24, Math.round(rows * world.width));
  const cellSize = world.height / rows;
  const relief = normalizeElevation(sampleElevation(columns, rows).elevation);
  const elevation = new Float32Array(columns * rows);
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      elevation[row * columns + column] =
        noise * relief[row * columns + column] + tilt * (1 - (column + 0.5) / columns);
    }
  }
  for (let craterRow = 0; craterRow < craterRows; craterRow++) {
    for (let craterColumn = 0; craterColumn < craterColumns; craterColumn++) {
      const centerColumn =
        columns * ((craterColumn + 0.5 + 0.3 * (random() - 0.5)) / craterColumns);
      const centerRow = rows * ((craterRow + 0.5 + 0.3 * (random() - 0.5)) / craterRows);
      const ringRadius = 3 + 1.5 * random();
      const depth = 0.012 + 0.008 * random();
      for (let row = 0; row < rows; row++) {
        for (let column = 0; column < columns; column++) {
          const distance = Math.hypot(column - centerColumn, row - centerRow);
          elevation[row * columns + column] +=
            0.025 * Math.exp(-(((distance - ringRadius) / 2) ** 2)) -
            depth * Math.exp(-(distance * distance) / (2 * 2.2 * 2.2));
        }
      }
    }
  }
  return assembleTerrainInputs(columns, rows, cellSize, elevation);
}

/**
 * A gentle slope with a few large craters, each holding smaller pits on its floor: draining a small
 * pit first spills into the big bowl, which spills over its rim, so the depressions nest two deep.
 */
export function createNestedCraterTerrainInputs(
  context: RunInputContext,
  rows = 40,
  craterColumns = 3,
  craterRows = 2
): TerrainInputs & {craterCenters: [number, number][]; subPitCenters: [number, number][]} {
  const {random, world} = context;
  const columns = Math.max(24, Math.round(rows * world.width));
  const cellSize = world.height / rows;
  const elevation = new Float32Array(columns * rows);
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      elevation[row * columns + column] = 0.05 * (1 - (column + 0.5) / columns);
    }
  }
  const craterCenters: [number, number][] = [];
  const subPitCenters: [number, number][] = [];
  const addBump = (
    centerColumn: number,
    centerRow: number,
    getHeight: (distance: number) => number
  ) => {
    for (let row = 0; row < rows; row++) {
      for (let column = 0; column < columns; column++) {
        elevation[row * columns + column] += getHeight(
          Math.hypot(column - centerColumn, row - centerRow)
        );
      }
    }
  };
  for (let craterRow = 0; craterRow < craterRows; craterRow++) {
    for (let craterColumn = 0; craterColumn < craterColumns; craterColumn++) {
      const centerColumn =
        columns * ((craterColumn + 0.5 + 0.2 * (random() - 0.5)) / craterColumns);
      const centerRow = rows * ((craterRow + 0.5 + 0.2 * (random() - 0.5)) / craterRows);
      const ringRadius = 7 + 2 * random();
      craterCenters.push([centerColumn, centerRow]);
      // The bowl and its raised rim.
      addBump(
        centerColumn,
        centerRow,
        distance =>
          0.04 * Math.exp(-(((distance - ringRadius) / 1.8) ** 2)) -
          0.035 * Math.exp(-(distance * distance) / (2 * (ringRadius * 0.55) ** 2))
      );
      // Two smaller pits in the bowl.
      for (const side of [-1, 1]) {
        subPitCenters.push([
          centerColumn + side * ringRadius * 0.35,
          centerRow + side * ringRadius * 0.15
        ]);
        addBump(
          centerColumn + side * ringRadius * 0.35,
          centerRow + side * ringRadius * 0.15,
          distance => -0.02 * Math.exp(-(distance * distance) / (2 * 1.6 * 1.6))
        );
      }
    }
  }
  return {
    ...assembleTerrainInputs(columns, rows, cellSize, elevation),
    craterCenters,
    subPitCenters
  };
}

export function assembleTerrainInputs(
  columns: number,
  rows: number,
  cellSize: number,
  elevation: Float32Array
): TerrainInputs {
  const cellCount = columns * rows;
  const pitCells: number[] = [];
  for (let row = 2; row < rows - 2; row++) {
    for (let column = 2; column < columns - 2; column++) {
      const cell = row * columns + column;
      let isPit = true;
      for (let offsetRow = -1; offsetRow <= 1 && isPit; offsetRow++) {
        for (let offsetColumn = -1; offsetColumn <= 1; offsetColumn++) {
          if (
            (offsetRow || offsetColumn) &&
            elevation[cell + offsetRow * columns + offsetColumn] <= elevation[cell]
          ) {
            isPit = false;
            break;
          }
        }
      }
      if (isPit) pitCells.push(cell);
    }
  }
  return {
    columns,
    rows,
    cellCount,
    cellSize,
    grid: makeGPURasterMetadata({
      width: columns,
      height: rows,
      cellSize,
      origin: [0, rows * cellSize]
    }),
    elevation,
    pitCells: Uint32Array.from(pitCells),
    elevationRange: [Math.min(...elevation), Math.max(...elevation)]
  };
}

export type TerrainFlowBuffers = {
  elevation: RunBuffer<'float32'>;
  filledElevation: RunBuffer<'float32'>;
  flowDirections: RunBuffer<'uint32'>;
  accumulation: RunBuffer<'float32'>;
  streams: RunBuffer<'uint32'>;
  /** The packed GPUTerrainFlow parameters, which the operations that mirror its routing share. */
  parameters: RunBuffer<'float32'>;
  /** Writes GPUTerrainFlow settings into the per-frame parameter buffer. */
  writeSettings(settings: GPUTerrainFlowSettings): void;
  /** Writes the stream threshold (in cells) into the operation's per-frame parameter buffer. */
  writeStreamThreshold(streamThreshold: number): void;
};

/** Uploads the DEM and adds the real GPUTerrainFlow (fill, D8 routing, accumulation, streams). */
export function addTerrainFlow(
  builder: RunBuilder,
  terrain: TerrainInputs,
  options: {
    flowRouting?: GPUTerrainFlowRouting;
    accumulationUnits?: GPUTerrainFlowAccumulationUnits;
  } = {}
): TerrainFlowBuffers {
  const {cellCount, grid} = terrain;
  const elevation = builder.upload('dem', 'float32', terrain.elevation);
  const parameterBuffer = builder.allocate('flow-parameters', 'float32', 8);
  const filledElevation = builder.allocate('filled-elevation', 'float32', cellCount);
  const flowDirections = builder.allocate('flow-directions', 'uint32', cellCount);
  const accumulation = builder.allocate('accumulation', 'float32', cellCount);
  const streams = builder.allocate('streams', 'uint32', cellCount);
  builder.graph.add(
    new GPUTerrainFlow({
      grid,
      elevation: makeGPURasterBand(elevation.view, grid, {id: 'dem'}),
      parameters: parameterBuffer.view,
      fillDepressions: true,
      flowRouting: options.flowRouting,
      accumulationUnits: options.accumulationUnits,
      output: {
        filledElevation: filledElevation.view,
        flowDirections: flowDirections.view,
        accumulation: accumulation.view,
        streams: streams.view
      }
    })
  );
  return {
    elevation,
    filledElevation,
    flowDirections,
    accumulation,
    streams,
    parameters: parameterBuffer,
    writeSettings: settings =>
      parameterBuffer.write(
        getGPUTerrainFlowParameterValues({fillEpsilon: 1e-5, ...settings}, grid)
      ),
    writeStreamThreshold: streamThreshold =>
      parameterBuffer.write(
        getGPUTerrainFlowParameterValues(
          {fillEpsilon: 1e-5, streamThreshold: Math.round(streamThreshold)},
          grid
        )
      )
  };
}

/** Layers shared by terrain runs: the filled DEM as a hillshaded context raster. */
export function getTerrainContextLayer(terrain: TerrainInputs, elevation: RunBuffer<'float32'>) {
  return {
    layer: 'raster',
    context: true,
    values: elevation,
    columns: terrain.columns,
    rows: terrain.rows,
    hillshade: 0.8,
    elevation: 30
  } as const;
}

/** The lowest cell of the trunk valley near a column (fraction of the width), as a cell index. */
export function findValleyCell(terrain: TerrainInputs, columnFraction: number): number {
  const column = Math.min(terrain.columns - 1, Math.floor(terrain.columns * columnFraction));
  let bestCell = column;
  for (let row = 1; row < terrain.rows; row++) {
    const cell = row * terrain.columns + column;
    if (terrain.elevation[cell] < terrain.elevation[bestCell]) bestCell = cell;
  }
  return bestCell;
}

/** [column, row] offsets of the eight D8 neighbors, in the order of the fractions: E, SE, S, SW, W, NW, N, NE. */
export const D8_OFFSETS = [
  [1, 0],
  [1, 1],
  [0, 1],
  [-1, 1],
  [-1, 0],
  [-1, -1],
  [0, -1],
  [1, -1]
] as const;

export const D8_NAMES = ['E', 'SE', 'S', 'SW', 'W', 'NW', 'N', 'NE'] as const;

type World = {width: number; height: number};

/** The raster cell under a drawing-world position (the raster layer fills the drawing world). */
export function getTerrainCell(
  terrain: TerrainInputs,
  world: World,
  x: number,
  y: number
): RasterCell {
  return getRasterCell(terrain.columns, terrain.rows, [0, 0, world.width, world.height], x, y);
}

/** Drawing-world position of a cell's center. */
export function getTerrainCellCenter(
  terrain: TerrainInputs,
  world: World,
  cell: number
): [number, number] {
  const column = cell % terrain.columns;
  const row = Math.floor(cell / terrain.columns);
  return [
    ((column + 0.5) / terrain.columns) * world.width,
    (1 - (row + 0.5) / terrain.rows) * world.height
  ];
}

/**
 * The cell of highest accumulation within `radius` cells of `cell`: the pointer's snap to the
 * channel nearest to it. Without an accumulation copy yet, the cell itself.
 */
export function snapToChannelCell(
  terrain: TerrainInputs,
  accumulation: ArrayLike<number> | null,
  cell: number,
  radius = 3
): number {
  if (!accumulation) return cell;
  const column = cell % terrain.columns;
  const row = Math.floor(cell / terrain.columns);
  let best = cell;
  for (let offsetRow = -radius; offsetRow <= radius; offsetRow++) {
    for (let offsetColumn = -radius; offsetColumn <= radius; offsetColumn++) {
      const nextColumn = column + offsetColumn;
      const nextRow = row + offsetRow;
      if (
        nextColumn < 0 ||
        nextRow < 0 ||
        nextColumn >= terrain.columns ||
        nextRow >= terrain.rows
      ) {
        continue;
      }
      const candidate = nextRow * terrain.columns + nextColumn;
      if (accumulation[candidate] > accumulation[best]) best = candidate;
    }
  }
  return best;
}

/**
 * The cells a drop of water visits from `cell`: follows the D8 codes of GPUTerrainFlow until a
 * terminal cell, a stream cell (when `streams` is given) or `maximumSteps`. Includes `cell` and
 * the cell where it stops.
 */
export function traceDownstream(
  terrain: TerrainInputs,
  flowDirections: ArrayLike<number>,
  cell: number,
  streams: ArrayLike<number> | null = null,
  maximumSteps = terrain.cellCount
): number[] {
  const path = [cell];
  let current = cell;
  for (let step = 0; step < maximumSteps; step++) {
    if (streams && streams[current] !== 0 && streams[current] !== 0xffffffff) break;
    const code = flowDirections[current];
    if (code === 0 || code >= 0xfffffffe) break;
    const direction = Math.round(Math.log2(code));
    const column = (current % terrain.columns) + D8_OFFSETS[direction][0];
    const row = Math.floor(current / terrain.columns) + D8_OFFSETS[direction][1];
    if (column < 0 || row < 0 || column >= terrain.columns || row >= terrain.rows) break;
    current = row * terrain.columns + column;
    path.push(current);
  }
  return path;
}

/** Packed (x, y) pairs of the cell centers, for a polyline glyph. */
export function getCellPolyline(
  terrain: TerrainInputs,
  world: World,
  cells: readonly number[]
): Float32Array {
  const points = new Float32Array(cells.length * 2);
  cells.forEach((cell, index) => {
    const [x, y] = getTerrainCellCenter(terrain, world, cell);
    points[index * 2] = x;
    points[index * 2 + 1] = y;
  });
  return points;
}
