// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUSpatialContext} from '@luma-contributors/analysis/analysis-core';

/** Existing geographic recipe fields are preserved without adding defaults or projection metadata. */
export type KeplerGeographicPointWeightCoordinates = Readonly<{
  longitudeColumn: string;
  latitudeColumn: string;
  coordinateSpace: 'longitude-latitude';
  metric: 'great-circle';
  units: 'meters';
}>;

/** Already-projected XY coordinates; this declaration never performs a CRS transformation. */
export type KeplerProjectedPointWeightCoordinates = Readonly<{
  xColumn: string;
  yColumn: string;
  coordinateSpace: 'planar';
  metric: 'native';
  units: 'meters';
  /** Caller-declared projected CRS identity, with native coordinate units already in meters. */
  crs: string;
  /** Inclusive source-wide search extent, in the declared projected meters. */
  bounds: readonly [number, number, number, number];
}>;

export type KeplerPointWeightCoordinates =
  | KeplerGeographicPointWeightCoordinates
  | KeplerProjectedPointWeightCoordinates;

const GEOGRAPHIC_KEYS = [
  'longitudeColumn',
  'latitudeColumn',
  'coordinateSpace',
  'metric',
  'units'
] as const;
const PROJECTED_KEYS = [
  'xColumn',
  'yColumn',
  'coordinateSpace',
  'metric',
  'units',
  'crs',
  'bounds'
] as const;
const WORLD_BOUNDS = Object.freeze([-180, -90, 180, 90] as const);

/** Coordinate keys only; a surrounding weights parser separately admits k/radius/capacity fields. */
export function getKeplerPointWeightCoordinateKeys(
  coordinateSpace: KeplerPointWeightCoordinates['coordinateSpace']
): readonly string[] {
  return coordinateSpace === 'planar' ? PROJECTED_KEYS : GEOGRAPHIC_KEYS;
}

/** Validate a coordinate-only object and detach its mutable bounds before graph allocation. */
export function parseKeplerPointWeightCoordinates(value: unknown): KeplerPointWeightCoordinates {
  if (!value || typeof value !== 'object' || Array.isArray(value))
    throw new Error('Point-weight coordinates must be an object');
  const settings = value as Record<string, unknown>;
  const coordinateSpace = settings['coordinateSpace'];
  if (coordinateSpace !== 'longitude-latitude' && coordinateSpace !== 'planar')
    throw new Error('Choose geographic longitude/latitude or declared projected XY coordinates');
  const keys = getKeplerPointWeightCoordinateKeys(coordinateSpace);
  if (Object.keys(settings).some(key => !keys.includes(key)))
    throw new Error('Unexpected point-weight coordinate setting');
  if (
    settings['units'] !== 'meters' ||
    settings['metric'] !== (coordinateSpace === 'planar' ? 'native' : 'great-circle')
  )
    throw new Error('Point-weight metric and meter units must match the declared coordinate space');
  if (coordinateSpace === 'longitude-latitude') {
    return Object.freeze({
      longitudeColumn: readColumn(settings['longitudeColumn']),
      latitudeColumn: readColumn(settings['latitudeColumn']),
      coordinateSpace,
      metric: 'great-circle',
      units: 'meters'
    });
  }
  const declaredCRS = settings['crs'];
  if (typeof declaredCRS !== 'string' || !declaredCRS.trim() || declaredCRS.length > 1024)
    throw new Error('Projected point weights require a declared CRS identity');
  const rawBounds = settings['bounds'];
  if (!Array.isArray(rawBounds) || rawBounds.length !== 4)
    throw new Error('Projected bounds require minX, minY, maxX and maxY');
  const bounds = Object.freeze([
    readFloat32(rawBounds[0]),
    readFloat32(rawBounds[1]),
    readFloat32(rawBounds[2]),
    readFloat32(rawBounds[3])
  ] as const);
  const width = Math.fround(bounds[2] - bounds[0]);
  const height = Math.fround(bounds[3] - bounds[1]);
  const squaredDiagonal = Math.fround(Math.fround(width * width) + Math.fround(height * height));
  if (
    width < 0 ||
    height < 0 ||
    !Number.isFinite(width) ||
    !Number.isFinite(height) ||
    !Number.isFinite(squaredDiagonal)
  )
    throw new Error('Projected bounds must be ordered and have finite Float32 squared distances');
  return Object.freeze({
    xColumn: readColumn(settings['xColumn']),
    yColumn: readColumn(settings['yColumn']),
    coordinateSpace,
    metric: 'native',
    units: 'meters',
    crs: declaredCRS.trim(),
    bounds
  });
}

/** Original source roles, independent of any geographic geometry used to render the map. */
export function getKeplerPointWeightCoordinateColumns(
  settings: KeplerPointWeightCoordinates
): readonly [string, string] {
  return settings.coordinateSpace === 'planar'
    ? [settings.xColumn, settings.yColumn]
    : [settings.longitudeColumn, settings.latitudeColumn];
}

/** Core metadata declares Euclidean distances in already-projected meters; it never reprojects. */
export function getKeplerPointWeightSpatialContext(
  settings: KeplerPointWeightCoordinates
): GPUSpatialContext {
  return settings.coordinateSpace === 'planar'
    ? {
        coordinateSpace: 'planar',
        metric: 'native',
        units: 'meters',
        frame: {kind: 'crs', id: settings.crs, crs: settings.crs}
      }
    : {coordinateSpace: 'longitude-latitude', metric: 'great-circle', units: 'meters'};
}

export function getKeplerPointWeightBounds(
  settings: KeplerPointWeightCoordinates
): readonly [number, number, number, number] {
  return settings.coordinateSpace === 'planar' ? settings.bounds : WORLD_BOUNDS;
}

/** Radius is a linear distance in meters; planar kernels also need a representable positive square. */
export function parseKeplerPointWeightRadius(
  value: unknown,
  settings: KeplerPointWeightCoordinates
): number {
  const radius = readFloat32(value);
  if (radius <= 0) throw new Error('Point-weight radius must be positive Float32 meters');
  if (settings.coordinateSpace === 'planar') {
    const squaredRadius = Math.fround(radius * radius);
    const cellRadius = Math.fround(radius * 1.0009765625);
    if (!(squaredRadius > 0) || !Number.isFinite(squaredRadius) || !Number.isFinite(cellRadius))
      throw new Error(
        'Projected radius must have a positive finite Float32 square and grid margin'
      );
  }
  return radius;
}

function readColumn(value: unknown): string {
  // Preserve the existing geographic column-name contract, including literal whitespace in names.
  if (typeof value !== 'string' || !value.length || value.length > 1024)
    throw new Error('Invalid point-weight coordinate column');
  return value;
}
function readFloat32(value: unknown): number {
  if (typeof value !== 'number' || !Number.isFinite(Math.fround(value)))
    throw new Error('Point-weight coordinates require finite Float32 values');
  return Math.fround(value);
}

/** Exact serialized coordinate identity used by topology caches and borrowed binding validation. */
export function areKeplerPointWeightCoordinatesEqual(
  left: KeplerPointWeightCoordinates,
  right: KeplerPointWeightCoordinates
): boolean {
  if (
    left.coordinateSpace !== right.coordinateSpace ||
    left.metric !== right.metric ||
    left.units !== right.units
  )
    return false;
  if (left.coordinateSpace === 'planar' && right.coordinateSpace === 'planar')
    return (
      left.xColumn === right.xColumn &&
      left.yColumn === right.yColumn &&
      left.crs === right.crs &&
      left.bounds.every((value, index) => value === right.bounds[index])
    );
  return (
    left.coordinateSpace === 'longitude-latitude' &&
    right.coordinateSpace === 'longitude-latitude' &&
    left.longitudeColumn === right.longitudeColumn &&
    left.latitudeColumn === right.latitudeColumn
  );
}

/** Require the producer's actual metric/frame declaration to match the saved analysis coordinates. */
export function matchesKeplerPointWeightSpatialContext(
  settings: KeplerPointWeightCoordinates,
  context: GPUSpatialContext | undefined
): boolean {
  if (
    !context ||
    context.coordinateSpace !== settings.coordinateSpace ||
    context.metric !== settings.metric ||
    context.units !== settings.units
  )
    return false;
  return settings.coordinateSpace === 'planar'
    ? context.frame?.kind === 'crs' &&
        context.frame.id === settings.crs &&
        context.frame.crs === settings.crs
    : true;
}
