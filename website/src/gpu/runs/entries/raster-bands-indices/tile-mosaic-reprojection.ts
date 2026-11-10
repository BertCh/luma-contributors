// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterTileReprojection stitches separate source tiles (three by two, each its own buffer with
// its own calibration offset) into one mosaic and resamples it into a rotated destination grid in a
// single pass. A tile that is not in `sources` leaves a nodata hole. Tiles and the destination grid
// are construction props; the calibration offset rewrites the tile buffers. The pointer probes a
// tile: its footprint is outlined in the rotated mosaic, with all six footprints dashed.

import {
  GPURasterTileReprojection,
  makeGPURasterBand,
  type GPURasterReprojectionSourceTile
} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun, type RunBuffer} from '../../define-run';
import {createElevationRaster} from '../../inputs';
import {parameters} from './tile-mosaic-reprojection.parameters';
import {drawRaster, makeGridOfRaster} from './support/bands';
import {makeRotatedGrid} from './support/reproject';

/** Destination cells are 1.45 source cells wide: the rotated mosaic fits inside the frame. */
const MOSAIC_SCALE = 1.45;
const TILE_COLUMNS = 3;
const TILE_ROWS = 2;

export default defineOperationRun({
  parameters,
  rebuildOn: ['rotation', 'missingTile'],
  caption: 'Separate tiles in; one reprojected mosaic out. A missing tile leaves a hole.',
  createInputs: ({world}) => createElevationRaster(world, 48),
  build(builder, inputs, values) {
    const cellCount = inputs.values.length;
    const sourceGrid = makeGridOfRaster(inputs);
    const destinationGrid = makeRotatedGrid(inputs, values.rotation, MOSAIC_SCALE);
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
  interaction: {
    kind: 'probe',
    hint: 'probe a tile',
    rest: [0.5, 0.5],
    resolve: (pointer, {inputs, values, world}) => {
      const destination = makeRotatedGrid(inputs, values.rotation, MOSAIC_SCALE);
      const [a, b, c, d, e, f] = destination.affine as [
        number,
        number,
        number,
        number,
        number,
        number
      ];
      const determinant = a * e - b * d;
      const cellSize = inputs.cellSize;
      // Drawn position <-> destination pixel <-> source-world position <-> source pixel.
      const toDrawn = (x: number, y: number): [number, number] => {
        const pixelColumn = (e * (x - c) - b * (y - f)) / determinant;
        const pixelRow = (-d * (x - c) + a * (y - f)) / determinant;
        return [
          (pixelColumn / inputs.columns) * world.width,
          world.height - (pixelRow / inputs.rows) * world.height
        ];
      };
      const toSource = (drawnX: number, drawnY: number): [number, number] => {
        const pixelColumn = (drawnX / world.width) * inputs.columns;
        const pixelRow = ((world.height - drawnY) / world.height) * inputs.rows;
        const x = a * pixelColumn + b * pixelRow + c;
        const y = d * pixelColumn + e * pixelRow + f;
        return [x / cellSize, inputs.rows - y / cellSize];
      };
      const tileWidth = Math.ceil(inputs.columns / TILE_COLUMNS);
      const tileHeight = Math.ceil(inputs.rows / TILE_ROWS);
      const [sourceColumn, sourceRow] = toSource(pointer.x, pointer.y);
      const inside =
        sourceColumn >= 0 &&
        sourceColumn < inputs.columns &&
        sourceRow >= 0 &&
        sourceRow < inputs.rows;
      const tile =
        Math.floor(sourceRow / tileHeight) * TILE_COLUMNS + Math.floor(sourceColumn / tileWidth);
      const footprint = (index: number) => {
        const left = (index % TILE_COLUMNS) * tileWidth;
        const top = Math.floor(index / TILE_COLUMNS) * tileHeight;
        const right = Math.min(inputs.columns, left + tileWidth);
        const bottom = Math.min(inputs.rows, top + tileHeight);
        return [
          ...toDrawn(left * cellSize, (inputs.rows - top) * cellSize),
          ...toDrawn(right * cellSize, (inputs.rows - top) * cellSize),
          ...toDrawn(right * cellSize, (inputs.rows - bottom) * cellSize),
          ...toDrawn(left * cellSize, (inputs.rows - bottom) * cellSize)
        ];
      };
      const glyphs = Array.from({length: TILE_COLUMNS * TILE_ROWS}, (_, index) => ({
        shape: 'polyline' as const,
        closed: true,
        dashed: index !== tile || !inside,
        points: footprint(index)
      }));
      const missing = values.missingTile - 1;
      const offset = tile % 2 === 0 ? values.tileOffset : 0;
      return {
        glyphs,
        readout: !inside
          ? 'outside the mosaic'
          : tile === missing
            ? `tile ${tile + 1} of 6 | left out`
            : `tile ${tile + 1} of 6 | offset +${offset.toFixed(2)}${missing >= 0 ? ` | tile ${missing + 1} missing` : ''}`
      };
    }
  },
  draw: ({inputs, outputs}) => [drawRaster(inputs, outputs.result, {range: [0, 1.2]})]
});
