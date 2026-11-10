// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Shared helpers for the terrain, relief and light runs: the Grand Canyon elevation tile in meters, the
// grid and band the operations take, and drawing helpers that place cells in the run's world.

import type {GPURasterBand, GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {
  makeGPURasterBand,
  makeGPURasterMetadata
} from '@luma-contributors/analysis/raster-analysis';
import type {RasterRunLayer, RunBuffer, RunBuilder} from '../../../define-run';
import type {RasterInput, World} from '../../../inputs';
import {normalizeElevation, sampleElevation} from '../../../../data/elevation';

export type TerrainInput = RasterInput & {
  /** Elevation in meters, row 0 at the top. */
  elevation: Float32Array;
  /** The drawing world the terrain covers. */
  world: World;
};

/** Measured elevation and projected ground spacing of the centered Grand Canyon crop. */
export function createTerrainInput(world: World, rows = 64): TerrainInput {
  const columns = Math.max(16, Math.round((rows * world.width) / world.height));
  const {elevation, cellSize} = sampleElevation(columns, rows);
  return {columns, rows, cellSize, elevation, values: normalizeElevation(elevation), world};
}

/**
 * The grid of a terrain input with the origin at the top-left corner. Meters by default; with
 * `inWorldUnits` one cell spans `world.height / rows`, so positions an operation writes (contour
 * vertices, snapped peaks) are directly in the run's drawing world.
 */
export function makeTerrainGrid(terrain: RasterInput, worldHeight?: number): GPURasterMetadata {
  const cellSize = worldHeight === undefined ? terrain.cellSize : worldHeight / terrain.rows;
  return makeGPURasterMetadata({
    width: terrain.columns,
    height: terrain.rows,
    cellSize,
    origin: [0, terrain.rows * cellSize]
  });
}

export type UploadedTerrain = {
  grid: GPURasterMetadata;
  band: GPURasterBand;
  elevation: RunBuffer<'float32'>;
  cellCount: number;
  /** Ground size of one cell in the grid's units. */
  cellSize: number;
};

export type UploadTerrainOptions = {
  /** Elevation values to upload instead of the terrain's own. */
  values?: Float32Array;
  id?: string;
  /** Put the grid in world units (see makeTerrainGrid). */
  worldHeight?: number;
};

/** Uploads the elevation and wraps it as a band on the terrain grid. */
export function uploadTerrain(
  builder: RunBuilder,
  terrain: TerrainInput,
  options: UploadTerrainOptions = {}
): UploadedTerrain {
  const grid = makeTerrainGrid(terrain, options.worldHeight);
  const id = options.id ?? 'elevation';
  const elevation = builder.upload(id, 'float32', options.values ?? terrain.elevation);
  return {
    grid,
    band: makeGPURasterBand(elevation.view, grid, {id}),
    elevation,
    cellCount: terrain.elevation.length,
    cellSize:
      options.worldHeight === undefined ? terrain.cellSize : options.worldHeight / terrain.rows
  };
}

/** Elevation plus seeded noise (meters), for runs whose slider changes the input surface. */
export function addTerrainNoise(
  terrain: TerrainInput,
  noise: Float32Array,
  amplitude: number,
  target: Float32Array = new Float32Array(terrain.elevation.length)
): Float32Array {
  for (let index = 0; index < target.length; index++) {
    target[index] = terrain.elevation[index] + (noise[index] - 0.5) * 2 * amplitude;
  }
  return target;
}

/** A raster layer on the terrain grid. */
export function drawGrid(
  terrain: {columns: number; rows: number},
  values: RunBuffer,
  look: Partial<RasterRunLayer> = {}
): RasterRunLayer {
  return {layer: 'raster', values, columns: terrain.columns, rows: terrain.rows, ...look};
}

/** World position (x right, y up) of the center of a cell given in fractional column and row. */
export function getCellWorldPosition(
  terrain: RasterInput,
  world: World,
  column: number,
  row: number
): [number, number] {
  return [
    ((column + 0.5) / terrain.columns) * world.width,
    world.height - ((row + 0.5) / terrain.rows) * world.height
  ];
}

/** Local maxima of the elevation (strictly above the 8 neighbours), highest first. */
export function findLocalMaxima(terrain: TerrainInput, limit: number): number[] {
  const found: {index: number; height: number}[] = [];
  for (let row = 1; row < terrain.rows - 1; row++) {
    for (let column = 1; column < terrain.columns - 1; column++) {
      const index = row * terrain.columns + column;
      const height = terrain.elevation[index];
      let isMaximum = true;
      for (let delta = 0; delta < 9 && isMaximum; delta++) {
        if (delta === 4) continue;
        const neighbor = index + (Math.floor(delta / 3) - 1) * terrain.columns + (delta % 3) - 1;
        isMaximum = terrain.elevation[neighbor] < height;
      }
      if (isMaximum) found.push({index, height});
    }
  }
  return found
    .sort((a, b) => b.height - a.height)
    .slice(0, limit)
    .map(entry => entry.index);
}

/** The input elevation drawn as a muted, hillshaded context layer. */
export function drawElevationContext(
  terrain: {columns: number; rows: number},
  elevation: RunBuffer
): RasterRunLayer {
  return drawGrid(terrain, elevation, {context: true, hillshade: 0.55, elevation: 50});
}

/** World positions (float32x2) of every cell center, row-major like the raster. */
export function createCellCenters(terrain: TerrainInput): Float32Array {
  const centers = new Float32Array(terrain.elevation.length * 2);
  for (let index = 0; index < terrain.elevation.length; index++) {
    const [x, y] = getCellWorldPosition(
      terrain,
      terrain.world,
      index % terrain.columns,
      Math.floor(index / terrain.columns)
    );
    centers[index * 2] = x;
    centers[index * 2 + 1] = y;
  }
  return centers;
}

/** Column and row of a position given as percentages of the raster width and height. */
export function getCellOfPercentage(
  terrain: RasterInput,
  columnPercentage: number,
  rowPercentage: number
): [number, number] {
  return [
    Math.round((columnPercentage / 100) * (terrain.columns - 1)),
    Math.round((rowPercentage / 100) * (terrain.rows - 1))
  ];
}

/** The cell under a world position, clamped to the raster; the inverse of getCellWorldPosition. */
export function getCellOfWorldPosition(
  terrain: RasterInput,
  world: World,
  x: number,
  y: number
): [number, number] {
  return [
    Math.min(terrain.columns - 1, Math.max(0, Math.floor((x / world.width) * terrain.columns))),
    Math.min(terrain.rows - 1, Math.max(0, Math.floor((1 - y / world.height) * terrain.rows)))
  ];
}

/** World units per meter of ground: one world height spans rows times the cell size. */
export function getWorldPerMeter(terrain: RasterInput, world: World): number {
  return world.height / (terrain.rows * terrain.cellSize);
}

/** World units spanned by a number of cells. */
export function getWorldPerCell(terrain: RasterInput, world: World): number {
  return world.height / terrain.rows;
}

/**
 * Local maxima spread over the frame: the highest first, each at least `minimumSpacingCells`
 * (Chebyshev distance) from every one already kept.
 */
export function findSpreadLocalMaxima(
  terrain: TerrainInput,
  count: number,
  minimumSpacingCells: number
): number[] {
  const kept: number[] = [];
  for (const index of findLocalMaxima(terrain, terrain.elevation.length)) {
    const column = index % terrain.columns;
    const row = Math.floor(index / terrain.columns);
    const isSpread = kept.every(
      other =>
        Math.max(
          Math.abs((other % terrain.columns) - column),
          Math.abs(Math.floor(other / terrain.columns) - row)
        ) >= minimumSpacingCells
    );
    if (isSpread) kept.push(index);
    if (kept.length === count) break;
  }
  return kept;
}

/** Visibility codes of the sight-line operations: hidden, visible, outOfRange (clear). */
export const VISIBILITY_COLORS = [
  [0.27, 0.1, 0.4, 1],
  [0.2, 0.85, 0.4, 1],
  [0, 0, 0, 0]
] as const;
