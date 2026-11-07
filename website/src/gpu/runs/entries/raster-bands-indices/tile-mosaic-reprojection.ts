// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterTileReprojection stitches separate source tiles (three by two, each its own buffer with
// its own calibration offset) into one mosaic and resamples it into a rotated destination grid in a
// single pass. A tile that is not in `sources` leaves a nodata hole. Tiles and the destination grid
// are construction props; the calibration offset rewrites the tile buffers.

import {
  GPURasterTileReprojection,
  makeGPURasterBand,
  type GPURasterReprojectionSourceTile
} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun, type RunBuffer} from '../../define-run';
import {createFractalRaster} from '../../inputs';
import {parameters} from './tile-mosaic-reprojection.parameters';
import {drawRaster, makeGridOfRaster} from './support/bands';
import {makeRotatedGrid} from './support/reproject';

const TILE_COLUMNS = 3;
const TILE_ROWS = 2;

export default defineOperationRun({
  parameters,
  rebuildOn: ['rotation', 'missingTile'],
  caption: 'Separate tiles in; one reprojected mosaic out. A missing tile leaves a hole.',
  createInputs: ({random, world}) => createFractalRaster(random, world, 48, 5),
  build(builder, inputs, values) {
    const cellCount = inputs.values.length;
    const sourceGrid = makeGridOfRaster(inputs);
    const destinationGrid = makeRotatedGrid(inputs, values.rotation, 0.8);
    const tileWidth = Math.ceil(inputs.columns / TILE_COLUMNS);
    const tileHeight = Math.ceil(inputs.rows / TILE_ROWS);
    const tiles: {buffer: RunBuffer; getTile: (offset: number) => Float32Array}[] = [];
    const sources: GPURasterReprojectionSourceTile[] = [];
    for (let tile = 0; tile < TILE_COLUMNS * TILE_ROWS; tile++) {
      if (values.missingTile === tile + 1) continue;
      const left = (tile % TILE_COLUMNS) * tileWidth;
      const top = Math.floor(tile / TILE_COLUMNS) * tileHeight;
      const right = Math.min(inputs.columns, left + tileWidth);
      const bottom = Math.min(inputs.rows, top + tileHeight);
      // Every other tile carries a calibration offset, so tile seams show in the mosaic.
      const getTile = (offset: number) => {
        const data = new Float32Array((right - left) * (bottom - top));
        for (let row = top; row < bottom; row++) {
          for (let column = left; column < right; column++) {
            data[(row - top) * (right - left) + column - left] =
              inputs.values[row * inputs.columns + column] + (tile % 2 === 0 ? offset : 0);
          }
        }
        return data;
      };
      const buffer = builder.upload(`tile-${tile}`, 'float32', getTile(0));
      tiles.push({buffer, getTile});
      sources.push({
        pixelBounds: [left, top, right, bottom],
        input: makeGPURasterBand(
          buffer.view,
          {...sourceGrid, width: right - left, height: bottom - top},
          {id: `tile-${tile}`}
        )
      });
    }
    const result = builder.allocate('values', 'float32', cellCount);
    const validity = builder.allocate('validity', 'uint32', cellCount);
    builder.graph.add(
      new GPURasterTileReprojection({
        sourceGrid,
        destinationGrid,
        sources,
        method: 'bilinear',
        output: {values: result.view, validity: validity.view}
      })
    );
    return {
      outputs: {result},
      update: current => tiles.forEach(tile => tile.buffer.write(tile.getTile(current.tileOffset)))
    };
  },
  draw: ({inputs, outputs}) => [drawRaster(inputs, outputs.result, {range: [0, 1.2]})]
});
