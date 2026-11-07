// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Shared helpers for the terrain, relief and light runs: a seeded elevation tile in meters, the
// grid and band the operations take, and drawing helpers that place cells in the run's world.

import type {GPURasterBand, GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {
  makeGPURasterBand,
  makeGPURasterMetadata
} from '@luma-contributors/analysis/raster-analysis';
import type {RasterRunLayer, RunBuffer, RunBuilder} from '../../../define-run';
import {createSmoothField, type RasterInput, type Random, type World} from '../../../inputs';

/** Ground size of one cell in meters. */
export const TERRAIN_CELL_SIZE = 30;

export type TerrainInput = RasterInput & {
  /** Elevation in meters, row 0 at the top. */
  elevation: Float32Array;
  /** The drawing world the terrain covers. */
  world: World;
};

/** Fractal terrain in meters with a few added summits so peak operations have something to find. */
export function createTerrainInput(
  random: Random,
  world: World,
  rows = 64,
  reliefMeters = 700
): TerrainInput {
  const columns = Math.max(16, Math.round(rows * world.width));
  const field = createSmoothField(random, 4, 3);
  const raster: RasterInput = {
    columns,
    rows,
    cellSize: TERRAIN_CELL_SIZE,
    // Isotropic features: the unit square of the field is cropped vertically to the aspect ratio.
    values: Float32Array.from({length: columns * rows}, (_, index) =>
      field(
        ((index % columns) + 0.5) / columns,
        (Math.floor(index / columns) + 0.5) / (rows * world.width)
      )
    )
  };
  let low = Infinity;
  let high = -Infinity;
  for (const value of raster.values) {
    low = Math.min(low, value);
    high = Math.max(high, value);
  }
  for (let index = 0; index < raster.values.length; index++) {
    raster.values[index] = (raster.values[index] - low) / (high - low);
  }
  const summits = Array.from({length: 5}, () => ({
    x: 0.1 + 0.8 * random(),
    y: 0.1 + 0.8 * random(),
    height: 150 + 200 * random(),
    spread: 0.04 + 0.04 * random()
  }));
  const elevation = new Float32Array(raster.values.length);
  for (let index = 0; index < elevation.length; index++) {
    const x = ((index % raster.columns) + 0.5) / raster.columns;
    const y = (Math.floor(index / raster.columns) + 0.5) / raster.rows;
    let height = raster.values[index] * reliefMeters;
    for (const summit of summits) {
      const distance =
        ((x * world.width - summit.x * world.width) ** 2 + (y - summit.y) ** 2) /
        summit.spread ** 2;
      height += summit.height * Math.exp(-distance);
    }
    elevation[index] = height;
  }
  return {...raster, cellSize: TERRAIN_CELL_SIZE, elevation, world};
}

/**
 * The grid of a terrain input with the origin at the top-left corner. Meters by default; with
 * `inWorldUnits` one cell spans `world.height / rows`, so positions an operation writes (contour
 * vertices, snapped peaks) are directly in the run's drawing world.
 */
export function makeTerrainGrid(terrain: RasterInput, worldHeight?: number): GPURasterMetadata {
  const cellSize = worldHeight === undefined ? TERRAIN_CELL_SIZE : worldHeight / terrain.rows;
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
      options.worldHeight === undefined ? TERRAIN_CELL_SIZE : options.worldHeight / terrain.rows
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
