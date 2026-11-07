// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Shared inputs for the terrain and hydrology runs: a seeded DEM whose ground coordinates are the
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
import {createSmoothField} from '../../../inputs';

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

/** A tilted, fractal landscape with a central valley and a few pits. */
export function createTerrainInputs(
  context: RunInputContext,
  rows = 48,
  fixedColumns?: number
): TerrainInputs {
  const {random, world} = context;
  const columns = fixedColumns ?? Math.max(24, Math.round(rows * world.width));
  const cellSize = world.height / rows;
  const field = createSmoothField(random, 5, 3);
  const cellCount = columns * rows;
  const elevation = new Float32Array(cellCount);
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      const x = (column + 0.5) / columns;
      const y = (row + 0.5) / rows;
      // A tributary valley joins a trunk valley that drains to the right edge.
      const valley = 0.2 * Math.abs(y - (0.55 + 0.08 * Math.sin(x * 6))) ** 0.8;
      const tributary = 0.1 * Math.abs(y - (0.55 + 0.35 * Math.sin(x * 3.2 + 1.5))) ** 0.9;
      const tilt = 0.2 * (1 - x);
      elevation[row * columns + column] = 0.11 * field(x, y) + tilt + 0.6 * valley + tributary;
    }
  }
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
  const field = createSmoothField(random, 3, 3);
  const elevation = new Float32Array(columns * rows);
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      elevation[row * columns + column] =
        noise * field((column + 0.5) / columns, (row + 0.5) / rows) +
        tilt * (1 - (column + 0.5) / columns);
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

function assembleTerrainInputs(
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
  const centerRow = Math.round(
    (0.55 + 0.08 * Math.sin(((column + 0.5) / terrain.columns) * 6)) * terrain.rows - 0.5
  );
  let bestCell = centerRow * terrain.columns + column;
  for (let row = centerRow - 3; row <= centerRow + 3; row++) {
    const cell = row * terrain.columns + column;
    if (terrain.elevation[cell] < terrain.elevation[bestCell]) bestCell = cell;
  }
  return bestCell;
}
