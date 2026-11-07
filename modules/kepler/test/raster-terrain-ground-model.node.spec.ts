// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  computeRasterTerrainGrid,
  getRasterRowGroundCellSize
} from '../../../examples/deck/kepler-compute/src/analysis2-raster-terrain';
import {
  createNumericRasterTile,
  type NumericRasterTile
} from '../../../examples/deck/kepler-compute/src/raster-tile-ingest';

function makeRampTile(south: number, north: number): NumericRasterTile {
  const size = 6;
  const values = new Float32Array(size * size);
  for (let row = 0; row < size; row++)
    for (let column = 0; column < size; column++) values[row * size + column] = column * 10;
  return createNumericRasterTile({
    id: 'ramp',
    width: size,
    height: size,
    extent: [0, south, size * 0.001, north],
    bands: [{index: 1, name: 'elevation', values, noDataValue: -9999}]
  });
}

test('per-row geographic ground cell size matches WGS84 meters per degree', () => {
  const tile = createNumericRasterTile({
    id: 'equator',
    width: 1,
    height: 1,
    extent: [0, -0.5, 1, 0.5],
    bands: [{index: 1, name: 'e', values: new Float32Array([0]), noDataValue: -9999}]
  });
  const {cellSizeX, cellSizeY} = getRasterRowGroundCellSize(tile, 'EPSG:4326', 0);
  // One degree of longitude at the equator is 111319.49 m; one degree of latitude there is 110574.39 m.
  expect(cellSizeX).toBeCloseTo(111319.49, 0);
  expect(cellSizeY).toBeCloseTo(110574.39, 0);
});

test('Web Mercator rows shrink with cos(latitude) by the WGS84 factor', () => {
  const tile = createNumericRasterTile({
    id: 'merc',
    width: 1,
    height: 1,
    extent: [0, -0.5, 1, 0.5],
    bands: [{index: 1, name: 'e', values: new Float32Array([0]), noDataValue: -9999}]
  });
  const equator = getRasterRowGroundCellSize(tile, 'EPSG:4326', 0);
  const mercator = getRasterRowGroundCellSize(tile, 'EPSG:3857', 0);
  expect(mercator.cellSizeX).toBeCloseTo(1, 6);
  expect(mercator.cellSizeY).toBeCloseTo(0.99330562, 5);
  expect(equator.cellSizeX).toBeGreaterThan(1000);
});

test('per-row model varies slope with latitude while tile-center model does not', () => {
  const tile = makeRampTile(0, 60);
  const tileCenter = computeRasterTerrainGrid(tile, 'EPSG:4326', {operation: 'slope'});
  const perRow = computeRasterTerrainGrid(tile, 'EPSG:4326', {
    operation: 'slope',
    groundModel: 'wgs84-per-row'
  });
  const size = tile.width;
  const upperRow = 1 * size + 2;
  const lowerRow = 4 * size + 2;
  expect(tileCenter.values[upperRow]).toBeCloseTo(tileCenter.values[lowerRow], 6);
  // Rows nearer the pole have narrower east-west ground cells, so the same ramp is steeper.
  expect(perRow.values[upperRow]).toBeGreaterThan(perRow.values[lowerRow]);
});
