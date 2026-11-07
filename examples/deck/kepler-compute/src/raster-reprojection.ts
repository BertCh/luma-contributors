// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {NumericRasterCrs} from './raster-tile-ingest';

import {clamp, toDegrees, toRadians} from '@math.gl/core';
import {WGS84_SEMI_MAJOR_AXIS_METERS} from './earth-constants';

const EARTH_RADIUS = WGS84_SEMI_MAJOR_AXIS_METERS;
const MAXIMUM_MERCATOR_LATITUDE = 85.0511287798066;

export class UnsupportedRasterCrsError extends Error {
  readonly crs: string;
  constructor(crs: string) {
    super(
      `Raster CRS ${crs} is not supported; only EPSG:4326 and EPSG:3857 can be sampled. ` +
        'Reproject the source before ingestion.'
    );
    this.name = 'UnsupportedRasterCrsError';
    this.crs = crs;
  }
}

export type RasterReprojectionPlan = Readonly<{
  sourceCrs: NumericRasterCrs;
  targetCrs: NumericRasterCrs;
  kind: 'identity' | 'geographic-to-mercator' | 'mercator-to-geographic';
  /** Maps one coordinate pair from the target CRS into the source CRS. */
  transformTargetToSource: (x: number, y: number) => [number, number];
  /** Maps one coordinate pair from the source CRS into the target CRS. */
  transformSourceToTarget: (x: number, y: number) => [number, number];
}>;

export function parseRasterCrs(crs: string | number): NumericRasterCrs {
  const normalized = String(crs)
    .toUpperCase()
    .replace(/^EPSG:?/, '');
  if (normalized === '4326') return 'EPSG:4326';
  if (normalized === '3857' || normalized === '900913') return 'EPSG:3857';
  throw new UnsupportedRasterCrsError(String(crs));
}

export function transformLngLatToMercator(lng: number, lat: number): [number, number] {
  const clampedLatitude = clamp(lat, -MAXIMUM_MERCATOR_LATITUDE, MAXIMUM_MERCATOR_LATITUDE);
  return [
    EARTH_RADIUS * toRadians(lng),
    EARTH_RADIUS * Math.log(Math.tan(Math.PI / 4 + toRadians(clampedLatitude) / 2))
  ];
}

export function transformMercatorToLngLat(x: number, y: number): [number, number] {
  return [
    toDegrees(x / EARTH_RADIUS),
    toDegrees(2 * Math.atan(Math.exp(y / EARTH_RADIUS)) - Math.PI / 2)
  ];
}

/**
 * Plans point sampling from a target CRS (lng/lat or Web Mercator) into a source CRS.
 * Only these two CRSs are supported; every other CRS is refused explicitly.
 */
export function planRasterReprojection(
  sourceCrs: string | number,
  targetCrs: string | number = 'EPSG:4326'
): RasterReprojectionPlan {
  const source = parseRasterCrs(sourceCrs);
  const target = parseRasterCrs(targetCrs);
  if (source === target)
    return {
      sourceCrs: source,
      targetCrs: target,
      kind: 'identity',
      transformTargetToSource: (x, y) => [x, y],
      transformSourceToTarget: (x, y) => [x, y]
    };
  if (source === 'EPSG:3857')
    return {
      sourceCrs: source,
      targetCrs: target,
      kind: 'geographic-to-mercator',
      transformTargetToSource: transformLngLatToMercator,
      transformSourceToTarget: transformMercatorToLngLat
    };
  return {
    sourceCrs: source,
    targetCrs: target,
    kind: 'mercator-to-geographic',
    transformTargetToSource: transformMercatorToLngLat,
    transformSourceToTarget: transformLngLatToMercator
  };
}
