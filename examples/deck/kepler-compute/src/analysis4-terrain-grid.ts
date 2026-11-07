// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {encodeCloudOptimizedGeoTIFF} from './interpolation-cog';
import {
  createNumericRasterSource,
  createNumericRasterTile,
  type NumericRasterCrs,
  type NumericRasterSource
} from './raster-tile-ingest';

import {toDegrees} from '@math.gl/core';
import {WGS84_SEMI_MAJOR_AXIS_METERS} from './earth-constants';

/**
 * Bridges between derived grids, NumericRasterSource and the native outputs of the terrain
 * panel: a mosaic of same-zoom tiles, a GeoTIFF export and a native cell table (one polygon row
 * per valid cell). Pure functions, no GPU and no DOM.
 */
export const TERRAIN_MOSAIC_CELL_LIMIT = 67_108_864;
export const TERRAIN_CELL_TABLE_LIMIT = 500_000;
const WEB_MERCATOR_RADIUS = WGS84_SEMI_MAJOR_AXIS_METERS;
const PIXEL_SIZE_TOLERANCE = 1e-6;

export type TerrainMosaic = Readonly<{
  crs: NumericRasterCrs;
  width: number;
  height: number;
  /** [west, south, east, north] in CRS units. */
  bounds: readonly [number, number, number, number];
  /** Row-major, north-up; NaN where no valid sample exists. */
  values: Float32Array;
  validCellCount: number;
  zoom: number;
}>;

export type DerivedGridInput = Readonly<{
  id: string;
  width: number;
  height: number;
  bounds: readonly [number, number, number, number];
  values: Float32Array;
}>;

/** Wraps a derived EPSG:4326 grid (IDW or kriging prediction) as a one-tile numeric source. */
export function createNumericRasterSourceFromGrid(grid: DerivedGridInput): NumericRasterSource {
  return createNumericRasterSource({
    id: grid.id,
    kind: 'single-raster',
    crs: 'EPSG:4326',
    tiles: [
      createNumericRasterTile({
        id: `${grid.id}/grid`,
        width: grid.width,
        height: grid.height,
        extent: grid.bounds,
        bands: [{index: 1, name: 'value', values: grid.values, noDataValue: null}]
      })
    ]
  });
}

/** Mosaics band 1 of the highest-zoom tiles into one grid. Throws when pixel sizes disagree. */
export function createTerrainMosaic(source: NumericRasterSource): TerrainMosaic {
  const maximumZoom = Math.max(...source.tiles.map(tile => tile.zoom));
  const tiles = source.tiles.filter(tile => tile.zoom === maximumZoom);
  // Binding error: an empty source has nothing to mosaic.
  if (tiles.length === 0) throw new Error('Raster source has no tiles');
  const pixelWidth = (tiles[0].extent[2] - tiles[0].extent[0]) / tiles[0].width;
  const pixelHeight = (tiles[0].extent[3] - tiles[0].extent[1]) / tiles[0].height;
  for (const tile of tiles) {
    const tileWidth = (tile.extent[2] - tile.extent[0]) / tile.width;
    const tileHeight = (tile.extent[3] - tile.extent[1]) / tile.height;
    // Binding error: tiles of one zoom must share a pixel size to form a single grid.
    if (
      Math.abs(tileWidth - pixelWidth) > pixelWidth * PIXEL_SIZE_TOLERANCE ||
      Math.abs(tileHeight - pixelHeight) > pixelHeight * PIXEL_SIZE_TOLERANCE
    )
      throw new Error('Tiles of one zoom level disagree on pixel size');
  }
  const west = Math.min(...tiles.map(tile => tile.extent[0]));
  const south = Math.min(...tiles.map(tile => tile.extent[1]));
  const east = Math.max(...tiles.map(tile => tile.extent[2]));
  const north = Math.max(...tiles.map(tile => tile.extent[3]));
  const width = Math.round((east - west) / pixelWidth);
  const height = Math.round((north - south) / pixelHeight);
  // Capacity limit: one Float32 grid.
  if (width * height > TERRAIN_MOSAIC_CELL_LIMIT)
    throw new Error(`Mosaic exceeds ${TERRAIN_MOSAIC_CELL_LIMIT} cells`);
  const values = new Float32Array(width * height).fill(Number.NaN);
  let validCellCount = 0;
  for (const tile of tiles) {
    const band = tile.bands[0];
    if (!band) continue;
    const columnStart = Math.round((tile.extent[0] - west) / pixelWidth);
    const rowStart = Math.round((north - tile.extent[3]) / pixelHeight);
    for (let row = 0; row < tile.height; row++)
      for (let column = 0; column < tile.width; column++) {
        const sourceIndex = row * tile.width + column;
        if (!band.validity[sourceIndex]) continue;
        const targetColumn = columnStart + column;
        const targetRow = rowStart + row;
        if (targetColumn >= width || targetRow >= height) continue;
        const targetIndex = targetRow * width + targetColumn;
        if (Number.isNaN(values[targetIndex])) validCellCount++;
        values[targetIndex] = band.values[sourceIndex];
      }
  }
  return {
    crs: source.crs,
    width,
    height,
    bounds: [west, south, east, north],
    values,
    validCellCount,
    zoom: maximumZoom
  };
}

/** Float32 GeoTIFF of the mosaic with provenance; tiled, deflated, with overviews. */
export function encodeTerrainMosaicGeoTIFF(
  mosaic: TerrainMosaic,
  bandName: string,
  provenance: Readonly<Record<string, unknown>>
): Promise<Uint8Array<ArrayBuffer>> {
  return encodeCloudOptimizedGeoTIFF({
    width: mosaic.width,
    height: mosaic.height,
    bounds: mosaic.bounds,
    bands: [{name: bandName, values: mosaic.values}],
    crs:
      mosaic.crs === 'EPSG:3857'
        ? {kind: 'projected', epsg: 3857}
        : {kind: 'geographic', epsg: 4326},
    provenance
  });
}

function toLongitudeLatitude(
  crs: NumericRasterCrs,
  x: number,
  y: number
): readonly [number, number] {
  if (crs === 'EPSG:4326') return [x, y];
  return [
    toDegrees(x / WEB_MERCATOR_RADIUS),
    toDegrees(2 * Math.atan(Math.exp(y / WEB_MERCATOR_RADIUS)) - Math.PI / 2)
  ];
}

/**
 * Native cell table: one row per valid cell with a GeoJSON polygon, the cell center and the
 * value in a column named after the operation. Returns undefined above the publication limit
 * (the GeoTIFF export has no such limit).
 */
export function createTerrainCellTable(
  mosaic: TerrainMosaic,
  columnName: string
): arrow.Table | undefined {
  if (mosaic.validCellCount === 0 || mosaic.validCellCount > TERRAIN_CELL_TABLE_LIMIT)
    return undefined;
  const [west, , , north] = mosaic.bounds;
  const pixelWidth = (mosaic.bounds[2] - west) / mosaic.width;
  const pixelHeight = (north - mosaic.bounds[1]) / mosaic.height;
  const cellIndex: number[] = [];
  const cellColumn: number[] = [];
  const cellRow: number[] = [];
  const longitude: number[] = [];
  const latitude: number[] = [];
  const value: number[] = [];
  const geometry: string[] = [];
  for (let row = 0; row < mosaic.height; row++)
    for (let column = 0; column < mosaic.width; column++) {
      const index = row * mosaic.width + column;
      const cellValue = mosaic.values[index];
      if (Number.isNaN(cellValue)) continue;
      const x0 = west + column * pixelWidth;
      const x1 = x0 + pixelWidth;
      const y1 = north - row * pixelHeight;
      const y0 = y1 - pixelHeight;
      const lowerLeft = toLongitudeLatitude(mosaic.crs, x0, y0);
      const upperRight = toLongitudeLatitude(mosaic.crs, x1, y1);
      const center = toLongitudeLatitude(mosaic.crs, (x0 + x1) / 2, (y0 + y1) / 2);
      cellIndex.push(index);
      cellColumn.push(column);
      cellRow.push(row);
      longitude.push(center[0]);
      latitude.push(center[1]);
      value.push(cellValue);
      geometry.push(
        JSON.stringify({
          type: 'Polygon',
          coordinates: [
            [
              [lowerLeft[0], lowerLeft[1]],
              [upperRight[0], lowerLeft[1]],
              [upperRight[0], upperRight[1]],
              [lowerLeft[0], upperRight[1]],
              [lowerLeft[0], lowerLeft[1]]
            ]
          ]
        })
      );
    }
  return new arrow.Table({
    cell_index: arrow.vectorFromArray(cellIndex, new arrow.Int32()),
    cell_column: arrow.vectorFromArray(cellColumn, new arrow.Int32()),
    cell_row: arrow.vectorFromArray(cellRow, new arrow.Int32()),
    longitude: arrow.vectorFromArray(longitude, new arrow.Float64()),
    latitude: arrow.vectorFromArray(latitude, new arrow.Float64()),
    [columnName]: arrow.vectorFromArray(value, new arrow.Float64()),
    geometry: arrow.vectorFromArray(geometry, new arrow.Utf8())
  });
}
