// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Reference geometry on a sphere (G24): geodesic circles and great-circle arcs that survive the
 * antimeridian and the poles, rhumb lines, graticules, a world-anchored metric grid and the
 * "square mile" frame. Every function returns `[longitude, latitude]` coordinates ready for a
 * line or ring annotation. Pure TypeScript.
 *
 * All distances use a spherical earth of mean radius {@link EARTH_RADIUS_METERS}
 * (6,371,008.8 m). Against the WGS84 ellipsoid this is within about 0.5 percent, far below what
 * a screen can show at national scales; do not use it for survey work.
 */

import {GPU_EARTH_MEAN_RADIUS_METERS} from '@luma-contributors/analysis/analysis-core';
import {
  clampLatitude,
  getLatitudeFromMercatorY,
  getMercatorY,
  getWebMercatorMetersPerPixel,
  WEB_MERCATOR_RADIUS_METERS
} from '../engine/web-mercator';
import {safeMod, toDegrees, toRadians} from '@math.gl/core';
import type {LngLat} from './types';

/** Mean earth radius in metres (IUGG), the sphere every function here assumes. */
export const EARTH_RADIUS_METERS = GPU_EARTH_MEAN_RADIUS_METERS;

export {GPU_EARTH_MEAN_RADIUS_METERS, WEB_MERCATOR_RADIUS_METERS};

/** Metres in one international mile. */
export const METERS_PER_MILE = 1609.344;

const DEGREES = Math.PI / 180;

/**
 * Canonical longitude normalization for the showcase: wraps a longitude (degrees) into the closed
 * range `[-180, 180]`. Values already inside the range, including exactly -180 and 180, are
 * returned unchanged so the function is idempotent and antimeridian splitting can keep the sign
 * of an edge vertex; other values wrap so that 180 + e becomes -180 + e.
 */
export function normalizeLongitude(longitude: number): number {
  if (longitude >= -180 && longitude <= 180) return longitude;
  return safeMod(longitude + 180, 360) - 180;
}

// ---------------------------------------------------------------------------------------------
// Distance and bearing on the mean-radius sphere
// ---------------------------------------------------------------------------------------------
// Closed-form spherical formulas: these run per vertex in track and migration scenes, where
// math.gl's EllipsoidGeodesicExact is ~190x slower for identical sphere results. The node spec
// checks these against EllipsoidGeodesicExact as an independent oracle.

/**
 * Great-circle (haversine) distance in metres on the mean-radius sphere.
 *
 * @example
 * getGreatCircleDistanceMeters([0, 0], [0, 1]); // 111,195 m: one degree of latitude
 */
export function getGreatCircleDistanceMeters(from: LngLat, to: LngLat): number {
  const fromLatitude = toRadians(from[1]);
  const toLatitude = toRadians(to[1]);
  const sinHalfLatitude = Math.sin((toLatitude - fromLatitude) / 2);
  const sinHalfLongitude = Math.sin(toRadians(to[0] - from[0]) / 2);
  const haversine =
    sinHalfLatitude * sinHalfLatitude +
    Math.cos(fromLatitude) * Math.cos(toLatitude) * sinHalfLongitude * sinHalfLongitude;
  return 2 * EARTH_RADIUS_METERS * Math.asin(Math.min(1, Math.sqrt(haversine)));
}

/**
 * Initial great-circle bearing from `from` to `to` in degrees clockwise from north, in `[0, 360)`.
 *
 * @example
 * getInitialBearingDegrees([0, 0], [10, 0]); // 90 (due east)
 */
export function getInitialBearingDegrees(from: LngLat, to: LngLat): number {
  const fromLatitude = toRadians(from[1]);
  const toLatitude = toRadians(to[1]);
  const longitudeDelta = toRadians(to[0] - from[0]);
  const y = Math.sin(longitudeDelta) * Math.cos(toLatitude);
  const x =
    Math.cos(fromLatitude) * Math.sin(toLatitude) -
    Math.sin(fromLatitude) * Math.cos(toLatitude) * Math.cos(longitudeDelta);
  return safeMod(toDegrees(Math.atan2(y, x)), 360);
}

/**
 * The point reached from `lngLat` travelling `distanceMeters` along a great circle on
 * `bearingDegrees` (clockwise from north, DEGREES; convert radians with `toDegrees`). Longitude
 * is normalized with {@link normalizeLongitude}.
 *
 * @example
 * getGeodesicDestination([0, 0], 90, 111_195); // about [1, 0]
 */
export function getGeodesicDestination(
  lngLat: LngLat,
  bearingDegrees: number,
  distanceMeters: number
): LngLat {
  const angularDistance = distanceMeters / EARTH_RADIUS_METERS;
  const bearing = toRadians(bearingDegrees);
  const latitude = toRadians(lngLat[1]);
  const sinLatitude = Math.sin(latitude);
  const cosLatitude = Math.cos(latitude);
  const sinDistance = Math.sin(angularDistance);
  const cosDistance = Math.cos(angularDistance);
  const destinationLatitude = Math.asin(
    Math.min(
      1,
      Math.max(-1, sinLatitude * cosDistance + cosLatitude * sinDistance * Math.cos(bearing))
    )
  );
  const destinationLongitude =
    toRadians(lngLat[0]) +
    Math.atan2(
      Math.sin(bearing) * sinDistance * cosLatitude,
      cosDistance - sinLatitude * Math.sin(destinationLatitude)
    );
  return [normalizeLongitude(toDegrees(destinationLongitude)), toDegrees(destinationLatitude)];
}

/** Alias of {@link getGreatCircleDistanceMeters}, kept for existing callers. */
export const haversineDistance = getGreatCircleDistanceMeters;

/** Alias of {@link getInitialBearingDegrees}, kept for existing callers. */
export const initialBearing = getInitialBearingDegrees;

// ---------------------------------------------------------------------------------------------
// Antimeridian splitting
// ---------------------------------------------------------------------------------------------

/**
 * Splits a polyline whose longitudes are wrapped to `[-180, 180]` into pieces that never jump
 * across the antimeridian; the crossing latitude is interpolated and both pieces end exactly on
 * longitude +-180.
 */
function splitPolylineAtAntimeridian(points: readonly LngLat[]): LngLat[][] {
  const pieces: LngLat[][] = [];
  let current: LngLat[] = [];
  for (let i = 0; i < points.length; i++) {
    const point = points[i];
    if (i > 0) {
      const previous = points[i - 1];
      if (Math.abs(point[0] - previous[0]) > 180) {
        const edge = previous[0] > 0 ? 180 : -180;
        const unwrapped = point[0] + (previous[0] > 0 ? 360 : -360);
        const t = (edge - previous[0]) / (unwrapped - previous[0]);
        const lat = previous[1] + t * (point[1] - previous[1]);
        current.push([edge, lat]);
        pieces.push(current);
        current = [[-edge, lat]];
      }
    }
    current.push(point);
  }
  if (current.length > 0) pieces.push(current);
  return pieces.filter(piece => piece.length > 1);
}

// ---------------------------------------------------------------------------------------------
// Geodesic circle
// ---------------------------------------------------------------------------------------------

/**
 * A circle of true ground radius as polygon rings.
 *
 * - A circle that crosses the antimeridian becomes two closed rings, one on each side, joined
 *   along longitude +-180.
 * - A circle that encloses a pole (the radius exceeds the distance to it) becomes one ring that
 *   runs from longitude -180 to 180 along the circle and closes along the pole's latitude line
 *   (`[180, +-90]` to `[-180, +-90]`), which fills correctly in Web Mercator.
 *
 * Radii of half the earth's circumference or more are clamped just below it.
 *
 * @example
 * // The pole-enclosure threshold of an airport at 52.31 N: (90 - 52.31) * 111.195 = 4,191 km.
 * const ams: LngLat = [4.76, 52.31];
 * geodesicCircle(ams, 4_100_000); // a loop that stays south of the pole (split at +-180 if it crosses)
 * geodesicCircle(ams, 4_300_000); // one ring from -180 to 180, closed along the pole at 90 N
 */
export function geodesicCircle(
  center: LngLat,
  radiusMeters: number,
  vertexCount = 120
): LngLat[][] {
  const count = Math.max(8, Math.floor(vertexCount));
  const maxRadius = Math.PI * EARTH_RADIUS_METERS * 0.999;
  const radius = Math.max(0, Math.min(radiusMeters, maxRadius));
  const ring: LngLat[] = [];
  for (let i = 0; i < count; i++) {
    ring.push(getGeodesicDestination(center, (i * 360) / count, radius));
  }
  const angularDegrees = radius / (EARTH_RADIUS_METERS * DEGREES);
  const poleLatitude =
    center[1] + angularDegrees > 90 ? 90 : center[1] - angularDegrees < -90 ? -90 : 0;

  if (poleLatitude !== 0) return [closeRingAroundPole(ring, poleLatitude)];

  // Closed loop: split where it crosses the antimeridian, then rejoin the end to the start.
  const loop = [...ring, ring[0]];
  const pieces = splitPolylineAtAntimeridian(loop);
  if (pieces.length <= 1) return [loop];
  const first = pieces[0];
  const last = pieces[pieces.length - 1];
  const merged = [...last, ...first.slice(1)];
  const middle = pieces.slice(1, -1);
  return [merged, ...middle].map(piece => [...piece, piece[0]]);
}

/** Builds the pole-enclosing ring: circle points sorted along longitude plus the pole edge. */
function closeRingAroundPole(ring: readonly LngLat[], poleLatitude: number): LngLat[] {
  // Find the one place where longitude wraps and rotate the ring to start just after it.
  let start = 0;
  let crossingLat = ring[0][1];
  for (let i = 0; i < ring.length; i++) {
    const previous = ring[(i + ring.length - 1) % ring.length];
    const point = ring[i];
    if (Math.abs(point[0] - previous[0]) > 180) {
      start = i;
      const edge = previous[0] > 0 ? 180 : -180;
      const unwrapped = point[0] + (previous[0] > 0 ? 360 : -360);
      const t = (edge - previous[0]) / (unwrapped - previous[0]);
      crossingLat = previous[1] + t * (point[1] - previous[1]);
      break;
    }
  }
  let path = [...ring.slice(start), ...ring.slice(0, start)];
  if (path[0][0] > path[path.length - 1][0]) path = path.reverse();
  return [
    [-180, crossingLat],
    ...path,
    [180, crossingLat],
    [180, poleLatitude],
    [-180, poleLatitude],
    [-180, crossingLat]
  ];
}

// ---------------------------------------------------------------------------------------------
// Great circle and rhumb line
// ---------------------------------------------------------------------------------------------

/**
 * The great-circle arc from `a` to `b` as polylines, split at the antimeridian (one piece in
 * most cases, two when the arc crosses it). Antipodal endpoints have no unique arc and return the
 * straight pair.
 *
 * @example
 * greatCircleArc([-122.4, 37.8], [139.7, 35.7]).length; // 2: San Francisco to Tokyo crosses +-180
 */
export function greatCircleArc(a: LngLat, b: LngLat, segments = 64): LngLat[][] {
  const lat1 = a[1] * DEGREES;
  const lng1 = a[0] * DEGREES;
  const lat2 = b[1] * DEGREES;
  const lng2 = b[0] * DEGREES;
  const v1 = [Math.cos(lat1) * Math.cos(lng1), Math.cos(lat1) * Math.sin(lng1), Math.sin(lat1)];
  const v2 = [Math.cos(lat2) * Math.cos(lng2), Math.cos(lat2) * Math.sin(lng2), Math.sin(lat2)];
  const dot = Math.max(-1, Math.min(1, v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]));
  const omega = Math.acos(dot);
  if (omega < 1e-9 || Math.PI - omega < 1e-9) return [[a, b]];
  const sinOmega = Math.sin(omega);
  const count = Math.max(1, Math.floor(segments));
  const points: LngLat[] = [];
  for (let i = 0; i <= count; i++) {
    const t = i / count;
    const w1 = Math.sin((1 - t) * omega) / sinOmega;
    const w2 = Math.sin(t * omega) / sinOmega;
    const x = w1 * v1[0] + w2 * v2[0];
    const y = w1 * v1[1] + w2 * v2[1];
    const z = w1 * v1[2] + w2 * v2[2];
    points.push([Math.atan2(y, x) / DEGREES, Math.atan2(z, Math.hypot(x, y)) / DEGREES]);
  }
  points[0] = [normalizeLongitude(a[0]), a[1]];
  points[count] = [normalizeLongitude(b[0]), b[1]];
  return splitPolylineAtAntimeridian(points);
}

function mercatorY(latitude: number): number {
  return getMercatorY(clampLatitude(latitude, 89.9999));
}

function inverseMercatorY(y: number): number {
  return getLatitudeFromMercatorY(y);
}

/**
 * The rhumb line (constant bearing, a straight line in Web Mercator) from `a` to `b`, taking the
 * shorter way around in longitude, as polylines split at the antimeridian. Compare it with
 * {@link greatCircleArc} to show why flights bow poleward.
 *
 * @example
 * rhumbLine([-74, 40.7], [0, 51.5]); // one polyline, straight in Web Mercator
 */
export function rhumbLine(a: LngLat, b: LngLat, segments = 64): LngLat[][] {
  let dLng = b[0] - a[0];
  if (dLng > 180) dLng -= 360;
  if (dLng < -180) dLng += 360;
  const y1 = mercatorY(a[1]);
  const y2 = mercatorY(b[1]);
  const count = Math.max(1, Math.floor(segments));
  const points: LngLat[] = [];
  for (let i = 0; i <= count; i++) {
    const t = i / count;
    points.push([normalizeLongitude(a[0] + t * dLng), inverseMercatorY(y1 + t * (y2 - y1))]);
  }
  points[0] = [normalizeLongitude(a[0]), a[1]];
  points[count] = [normalizeLongitude(b[0]), b[1]];
  return splitPolylineAtAntimeridian(points);
}

// ---------------------------------------------------------------------------------------------
// Parallels, graticule
// ---------------------------------------------------------------------------------------------

/** Named parallels of the sun's declination and the polar circles, in degrees. */
export const REFERENCE_LATITUDES = {
  equator: {latitude: 0, name: 'Equator'},
  tropicOfCancer: {latitude: 23.44, name: 'Tropic of Cancer'},
  tropicOfCapricorn: {latitude: -23.44, name: 'Tropic of Capricorn'},
  arcticCircle: {latitude: 66.56, name: 'Arctic Circle'},
  antarcticCircle: {latitude: -66.56, name: 'Antarctic Circle'}
} as const;

/** Options of {@link latitudeLine}. */
export type LatitudeLineOptions = {
  /** Western end, default -180. */
  west?: number;
  /** Eastern end, default 180. */
  east?: number;
  /** Vertex spacing in degrees of longitude, default 10 (for projections where it curves). */
  stepDegrees?: number;
};

/**
 * A parallel as one polyline from `west` to `east` (default the whole world).
 *
 * @example
 * latitudeLine(REFERENCE_LATITUDES.tropicOfCancer.latitude);
 */
export function latitudeLine(latitude: number, options: LatitudeLineOptions = {}): LngLat[] {
  const west = options.west ?? -180;
  const east = options.east ?? 180;
  const step = Math.max(0.1, options.stepDegrees ?? 10);
  const points: LngLat[] = [];
  for (let lng = west; lng < east; lng += step) points.push([lng, latitude]);
  points.push([east, latitude]);
  return points;
}

/** Options of {@link graticuleLines}. */
export type GraticuleOptions = {
  /** `[west, south, east, north]` to cover, default the world. */
  bounds?: readonly [number, number, number, number];
  /**
   * Meridians stop at this latitude, default 85 (Web Mercator cannot show the poles). Parallels
   * are never drawn on a pole.
   */
  maxLatitude?: number;
  /** Vertex spacing along each line in degrees, default 5. */
  segmentDegrees?: number;
};

/**
 * Meridians and parallels every `stepDegrees`, anchored at 0 (so 30-degree lines always sit on
 * -60, -30, 0, 30, 60), as polylines: meridians first, then parallels.
 *
 * @example
 * const lines = graticuleLines(30); // 13 meridians and 5 parallels
 */
export function graticuleLines(stepDegrees: number, options: GraticuleOptions = {}): LngLat[][] {
  const step = Math.max(0.01, stepDegrees);
  const [west, south, east, north] = options.bounds ?? [-180, -90, 180, 90];
  const maxLat = options.maxLatitude ?? 85;
  const segment = Math.max(0.1, options.segmentDegrees ?? 5);
  const lines: LngLat[][] = [];
  const bottom = Math.max(south, -maxLat);
  const top = Math.min(north, maxLat);
  const epsilon = step * 1e-9;
  for (let lng = Math.ceil((west - epsilon) / step) * step; lng <= east + epsilon; lng += step) {
    const line: LngLat[] = [];
    for (let lat = bottom; lat < top; lat += segment) line.push([lng, lat]);
    line.push([lng, top]);
    if (line.length > 1) lines.push(line);
  }
  for (let lat = Math.ceil((south - epsilon) / step) * step; lat <= north + epsilon; lat += step) {
    if (Math.abs(lat) >= 90) continue;
    lines.push(latitudeLine(lat, {west, east, stepDegrees: segment}));
  }
  return lines;
}

// ---------------------------------------------------------------------------------------------
// Metric grid
// ---------------------------------------------------------------------------------------------

/** The 1-2-5 ladder used for grid and scale spacing, as mantissas. */
const NICE_MANTISSAS = [1, 2, 5];

/**
 * Web Mercator metres per screen pixel at a zoom (MapLibre 512 px tiles), independent of
 * latitude. Ground metres per pixel are this times `cos(latitude)`.
 *
 * @example
 * getMercatorMetersPerPixel(10); // about 38.2
 */
export function getMercatorMetersPerPixel(zoom: number): number {
  return getWebMercatorMetersPerPixel(0, zoom);
}

/**
 * The smallest 1-2-5 spacing (1, 2, 5, 10, 20, 50 ... metres) that is at least
 * `metersPerPixel * targetPixels`, so grid lines are never denser than `targetPixels` apart.
 *
 * @example
 * getNiceGridSpacing(38.2, 120); // 5000 (4,584 m rounds up to the next step)
 */
export function getNiceGridSpacing(metersPerPixel: number, targetPixels = 100): number {
  const target = Math.max(1e-9, metersPerPixel * targetPixels);
  const exponent = Math.floor(Math.log10(target));
  for (const power of [exponent, exponent + 1]) {
    for (const mantissa of NICE_MANTISSAS) {
      const spacing = mantissa * 10 ** power;
      if (spacing >= target * (1 - 1e-12)) return spacing;
    }
  }
  return 10 ** (exponent + 1);
}

/** Options of {@link metricGridLines}. */
export type MetricGridOptions = {
  /** Coarsen the spacing along the 1-2-5 ladder until there are at most this many lines. Default 200. */
  maxLines?: number;
};

/**
 * Grid lines every `spacingMeters` of Web Mercator metres (EPSG:3857, sphere radius
 * 6,378,137 m), anchored at the world origin (0, 0) so the lines do not slide as the camera
 * moves, as straight two-point polylines inside `bounds` (`[west, south, east, north]`):
 * meridians first, then parallels. The lines are straight in Web Mercator by construction.
 *
 * These are Mercator metres, not ground metres: a "1 km" cell is `cos(latitude)` km on the
 * ground (see `projection-notes.ts`). Label such a grid as a map grid, never as a distance.
 * Use {@link getNiceGridSpacing} to pick `spacingMeters` from the zoom.
 *
 * @example
 * const spacing = getNiceGridSpacing(getMercatorMetersPerPixel(zoom), 120);
 * const lines = metricGridLines([-88, 41.6, -87.4, 42.1], spacing);
 */
export function metricGridLines(
  bounds: readonly [number, number, number, number],
  spacingMeters: number,
  options: MetricGridOptions = {}
): LngLat[][] {
  const [west, south, east, north] = bounds;
  const radius = WEB_MERCATOR_RADIUS_METERS;
  const xWest = radius * west * DEGREES;
  const xEast = radius * east * DEGREES;
  const yBottom = radius * mercatorY(south);
  const yTop = radius * mercatorY(north);
  const maxLines = Math.max(2, options.maxLines ?? 200);
  let spacing = Math.max(1e-3, spacingMeters);
  const lineCount = (step: number) =>
    Math.ceil((xEast - xWest) / step) + Math.ceil((yTop - yBottom) / step) + 2;
  while (lineCount(spacing) > maxLines) spacing = getNiceGridSpacing(spacing * 1.001, 1);

  const lines: LngLat[][] = [];
  for (let k = Math.ceil(xWest / spacing); k * spacing <= xEast; k++) {
    const lng = (k * spacing) / radius / DEGREES;
    lines.push([
      [lng, south],
      [lng, north]
    ]);
  }
  for (let k = Math.ceil(yBottom / spacing); k * spacing <= yTop; k++) {
    const lat = inverseMercatorY((k * spacing) / radius);
    lines.push([
      [west, lat],
      [east, lat]
    ]);
  }
  return lines;
}

// ---------------------------------------------------------------------------------------------
// Square mile
// ---------------------------------------------------------------------------------------------

/**
 * A closed ring (five vertices) around `center` that is one mile on each side on the ground, so
 * it encloses one square mile (2.59 km2) wherever it is drawn. The corners are found with
 * {@link getGeodesicDestination}: half a mile east or west, then half a mile north or south.
 * In Web Mercator it looks `1 / cos(latitude)` times larger on screen at high latitudes.
 *
 * @example
 * squareMileFrame([-87.63, 41.88]); // the Loop in a one-mile frame, for a size comparison
 */
export function squareMileFrame(center: LngLat): LngLat[] {
  const half = METERS_PER_MILE / 2;
  const north = getGeodesicDestination(center, 0, half)[1];
  const south = getGeodesicDestination(center, 180, half)[1];
  const east = getGeodesicDestination(center, 90, half)[0];
  const west = getGeodesicDestination(center, 270, half)[0];
  return [
    [west, south],
    [east, south],
    [east, north],
    [west, north],
    [west, south]
  ];
}
