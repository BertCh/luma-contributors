// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// The raster grid the zone-transfer and raster recipes use: square cells covering the drawing
// world with row 0 at the top (largest y), the orientation the raster layer draws.

import {
  getGPUPolygonRasterizationParameterValues,
  makeGPURasterMetadata
} from '@luma-contributors/analysis/raster-analysis';

export const GRID_HEIGHT = 40;

/** A grid of square cells that covers the world; `cellSize` is in world units. */
export function createWorldGrid(world: {width: number; height: number}) {
  const cellSize = world.height / GRID_HEIGHT;
  const width = Math.ceil(world.width / cellSize);
  const grid = makeGPURasterMetadata({
    width,
    height: GRID_HEIGHT,
    cellSize,
    origin: [0, world.height],
    rowDirection: 'south'
  });
  return {
    grid,
    width,
    height: GRID_HEIGHT,
    cellSize,
    worldHeight: world.height,
    parameters: getGPUPolygonRasterizationParameterValues(grid)
  };
}

export const NO_ZONE = 0xffffffff;

/** Zone id of every grid cell by the cell-center rule (`NO_ZONE` outside all zones). */
export function rasterizeZones(
  positions: Float32Array,
  zoneCount: number,
  verticesPerZone: number,
  world: ReturnType<typeof createWorldGrid>
): Uint32Array {
  const zones = new Uint32Array(world.width * world.height).fill(NO_ZONE);
  for (let row = 0; row < world.height; row++) {
    for (let column = 0; column < world.width; column++) {
      const x = (column + 0.5) * world.cellSize;
      const y = world.worldHeight - (row + 0.5) * world.cellSize;
      for (let zone = 0; zone < zoneCount; zone++) {
        let inside = false;
        const first = zone * verticesPerZone;
        for (let vertex = 0; vertex < verticesPerZone; vertex++) {
          const a = (first + vertex) * 2;
          const b = (first + ((vertex + 1) % verticesPerZone)) * 2;
          const ax = positions[a];
          const ay = positions[a + 1];
          const bx = positions[b];
          const by = positions[b + 1];
          if (ay > y !== by > y && x < ax + ((y - ay) / (by - ay)) * (bx - ax)) inside = !inside;
        }
        if (inside) {
          zones[row * world.width + column] = zone;
          break;
        }
      }
    }
  }
  return zones;
}

/** Cell center (x, y) of grid cell `index`. */
export function getCellCenter(
  world: ReturnType<typeof createWorldGrid>,
  index: number
): [number, number] {
  return [
    ((index % world.width) + 0.5) * world.cellSize,
    world.worldHeight - (Math.floor(index / world.width) + 0.5) * world.cellSize
  ];
}
