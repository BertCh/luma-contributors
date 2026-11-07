// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createNumericRasterSource,
  createNumericRasterTile,
  type NumericRasterSource,
  type NumericRasterTile
} from './raster-tile-ingest';
import {
  computeRasterZonalStatistics,
  type RasterZonalOptions,
  type RasterZonalPolygon,
  type RasterZonalStatistics
} from './raster-zonal';

import {clamp, safeMod, toDegrees, toRadians} from '@math.gl/core';
import {WGS84_SEMI_MAJOR_AXIS_METERS} from './earth-constants';

/**
 * CPU reference terrain operators over a NumericRasterSource tile (Horn 3x3 kernels).
 * The result is a derived single-band source, so zonal statistics and band math bind to it
 * exactly like to any other source. Border samples and samples with an invalid neighbor are nodata.
 */
export type RasterTerrainOperation = 'slope' | 'aspect' | 'hillshade' | 'curvature';

export type RasterTerrainOptions = Readonly<{
  operation: RasterTerrainOperation;
  bandIndex?: number;
  /** Vertical exaggeration; elevation units are assumed to equal ground meters times this factor. */
  zFactor?: number;
  /** Degrees clockwise from north, used by hillshade. */
  azimuthDegrees?: number;
  /** Degrees above the horizon, used by hillshade. */
  altitudeDegrees?: number;
  /**
   * Ground cell size model. 'tile-center' (default) uses one spherical approximation at the tile
   * center; 'wgs84-per-row' evaluates the WGS84 ellipsoid at every row center, the same model
   * the GPU contributors use, so the CPU fallback and oracle agree with the GPU result.
   */
  groundModel?: 'tile-center' | 'wgs84-per-row';
}>;

const METERS_PER_DEGREE_LATITUDE = 110574;
const METERS_PER_DEGREE_LONGITUDE_AT_EQUATOR = 111320;
const WEB_MERCATOR_HALF_CIRCUMFERENCE = 20037508.342789244;
export const RASTER_TERRAIN_NODATA = -9999;

/** Ground cell size in meters at the tile center, for either supported CRS. */
export function getRasterGroundCellSize(
  tile: NumericRasterTile,
  crs: NumericRasterSource['crs']
): {cellSizeX: number; cellSizeY: number} {
  const [west, south, east, north] = tile.extent;
  const pixelWidth = (east - west) / tile.width;
  const pixelHeight = (north - south) / tile.height;
  if (crs === 'EPSG:3857') {
    const centerNorth = (north + south) / 2;
    const latitudeRadians =
      2 * Math.atan(Math.exp((centerNorth / WEB_MERCATOR_HALF_CIRCUMFERENCE) * Math.PI)) -
      Math.PI / 2;
    const scale = Math.cos(latitudeRadians);
    return {cellSizeX: pixelWidth * scale, cellSizeY: pixelHeight * scale};
  }
  const latitudeRadians = toRadians((north + south) / 2);
  return {
    cellSizeX: pixelWidth * METERS_PER_DEGREE_LONGITUDE_AT_EQUATOR * Math.cos(latitudeRadians),
    cellSizeY: pixelHeight * METERS_PER_DEGREE_LATITUDE
  };
}

const WGS84_ECCENTRICITY_SQUARED = 0.0066943799901413165;

/** Ground cell size at one row center on the WGS84 ellipsoid (float64 twin of the GPU model). */
export function getRasterRowGroundCellSize(
  tile: NumericRasterTile,
  crs: NumericRasterSource['crs'],
  row: number
): {cellSizeX: number; cellSizeY: number} {
  const [west, south, east, north] = tile.extent;
  const pixelWidth = (east - west) / tile.width;
  const pixelHeight = (north - south) / tile.height;
  const rowCenter = north - (row + 0.5) * pixelHeight;
  if (crs === 'EPSG:3857') {
    const mercatorY = rowCenter / WGS84_SEMI_MAJOR_AXIS_METERS;
    const sinLatitude = Math.tanh(mercatorY);
    const curvature = 1 - WGS84_ECCENTRICITY_SQUARED * sinLatitude * sinLatitude;
    const cosLatitude = 1 / Math.cosh(mercatorY);
    return {
      cellSizeX: pixelWidth * (cosLatitude / Math.sqrt(curvature)),
      cellSizeY:
        pixelHeight *
        (((1 - WGS84_ECCENTRICITY_SQUARED) / (curvature * Math.sqrt(curvature))) * cosLatitude)
    };
  }
  const latitude = toRadians(rowCenter);
  const sinLatitude = Math.sin(latitude);
  const curvature = 1 - WGS84_ECCENTRICITY_SQUARED * sinLatitude * sinLatitude;
  const primeVertical = WGS84_SEMI_MAJOR_AXIS_METERS / Math.sqrt(curvature);
  return {
    cellSizeX: toRadians(pixelWidth) * primeVertical * Math.cos(latitude),
    cellSizeY:
      toRadians(pixelHeight) *
      ((WGS84_SEMI_MAJOR_AXIS_METERS * (1 - WGS84_ECCENTRICITY_SQUARED)) /
        (curvature * Math.sqrt(curvature)))
  };
}

export type RasterTerrainGrid = Readonly<{
  width: number;
  height: number;
  values: Float32Array;
  validity: Uint8Array;
}>;

export function computeRasterTerrainGrid(
  tile: NumericRasterTile,
  crs: NumericRasterSource['crs'],
  options: RasterTerrainOptions
): RasterTerrainGrid {
  const band = tile.bands.find(candidate => candidate.index === (options.bandIndex ?? 1));
  // Binding error: the requested elevation band is not part of this tile.
  if (!band) throw new Error(`Raster tile ${tile.id} has no band ${options.bandIndex ?? 1}`);
  const {width, height} = tile;
  const tileCellSize = getRasterGroundCellSize(tile, crs);
  const perRow = options.groundModel === 'wgs84-per-row';
  const zFactor = options.zFactor ?? 1;
  const zenith = toRadians(90 - (options.altitudeDegrees ?? 45));
  const azimuth = toRadians(safeMod(360 - (options.azimuthDegrees ?? 315) + 90, 360));
  const values = new Float32Array(width * height);
  const validity = new Uint8Array(width * height);
  const elevations = band.values;
  for (let row = 1; row < height - 1; row++) {
    const {cellSizeX, cellSizeY} = perRow
      ? getRasterRowGroundCellSize(tile, crs, row)
      : tileCellSize;
    for (let column = 1; column < width - 1; column++) {
      const center = row * width + column;
      const indices = [
        center - width - 1,
        center - width,
        center - width + 1,
        center - 1,
        center,
        center + 1,
        center + width - 1,
        center + width,
        center + width + 1
      ];
      if (indices.some(index => !band.validity[index])) continue;
      const [a, b, c, d, e, f, g, h, i] = indices.map(index => elevations[index] * zFactor);
      const rateX = (c + 2 * f + i - (a + 2 * d + g)) / (8 * cellSizeX);
      const rateY = (g + 2 * h + i - (a + 2 * b + c)) / (8 * cellSizeY);
      let output: number;
      if (options.operation === 'slope') {
        output = toDegrees(Math.atan(Math.hypot(rateX, rateY)));
      } else if (options.operation === 'aspect') {
        // Flat cells have no aspect; ESRI convention marks them -1, which stays valid.
        if (rateX === 0 && rateY === 0) output = -1;
        else {
          let degrees = toDegrees(Math.atan2(rateY, -rateX));
          degrees = degrees < 0 ? 90 - degrees : degrees > 90 ? 360 - degrees + 90 : 90 - degrees;
          output = degrees;
        }
      } else if (options.operation === 'hillshade') {
        const slopeRadians = Math.atan(Math.hypot(rateX, rateY));
        const aspectRadians =
          rateX !== 0
            ? Math.atan2(rateY, -rateX)
            : rateY > 0
              ? Math.PI / 2
              : rateY < 0
                ? (3 * Math.PI) / 2
                : 0;
        const shade =
          Math.cos(zenith) * Math.cos(slopeRadians) +
          Math.sin(zenith) * Math.sin(slopeRadians) * Math.cos(azimuth - aspectRadians);
        output = clamp(255 * shade, 0, 255);
      } else {
        // ESRI curvature: second derivative of the surface, scaled by 100.
        const secondD = ((d + f) / 2 - e) / (cellSizeX * cellSizeX);
        const secondE = ((b + h) / 2 - e) / (cellSizeY * cellSizeY);
        output = -2 * (secondD + secondE) * 100;
      }
      values[center] = output;
      validity[center] = 1;
    }
  }
  return {width, height, values, validity};
}

/** Wraps the derived grid as a one-band NumericRasterSource sharing the input extent and CRS. */
export function createTerrainRasterSource(
  source: NumericRasterSource,
  options: RasterTerrainOptions
): NumericRasterSource {
  return createNumericRasterSource({
    id: `${source.id}:${options.operation}`,
    kind: source.kind,
    crs: source.crs,
    tiles: source.tiles.map(tile => {
      const grid = computeRasterTerrainGrid(tile, source.crs, options);
      const stored = new Float32Array(grid.values);
      for (let sampleIndex = 0; sampleIndex < stored.length; sampleIndex++)
        if (!grid.validity[sampleIndex]) stored[sampleIndex] = RASTER_TERRAIN_NODATA;
      return createNumericRasterTile({
        id: `${tile.id}:${options.operation}`,
        width: tile.width,
        height: tile.height,
        extent: tile.extent,
        zoom: tile.zoom,
        tileIndex: tile.tileIndex,
        bands: [
          {index: 1, name: options.operation, values: stored, noDataValue: RASTER_TERRAIN_NODATA}
        ]
      });
    })
  });
}

/** Zonal statistics of a terrain derivative over polygons. */
export function computeRasterTerrainZonalStatistics(
  source: NumericRasterSource,
  polygons: readonly RasterZonalPolygon[],
  terrain: RasterTerrainOptions,
  zonalOptions: Omit<RasterZonalOptions, 'bandIndex' | 'expression'> = {}
): RasterZonalStatistics {
  return computeRasterZonalStatistics(createTerrainRasterSource(source, terrain), polygons, {
    ...zonalOptions,
    bandIndex: 1
  });
}
