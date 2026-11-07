// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Feature, Geometry} from 'geojson';

import {clamp, toRadians} from '@math.gl/core';
import {MEAN_EARTH_RADIUS_METERS} from './earth-constants';

type Position = readonly number[];
type Ring = readonly Position[];

export type DrawingIssueCode =
  | 'unsupported-geometry'
  | 'empty-geometry'
  | 'non-finite-coordinate'
  | 'too-few-vertices'
  | 'collapsed-ring'
  | 'zero-area'
  | 'self-intersection'
  | 'hole-outside-shell'
  | 'hole-crosses-shell'
  | 'vertex-budget'
  | 'longitude-out-of-range'
  | 'latitude-out-of-range'
  | 'antimeridian-edge'
  | 'self-intersection-unchecked';

export type DrawingIssue = {code: DrawingIssueCode; message: string};

/** Invalid shapes cannot become filters; warnings document native planar-degree semantics. */
export type DrawingValidation =
  | {status: 'valid'; polygons: Ring[][]; warnings: DrawingIssue[]}
  | {status: 'invalid'; reasons: DrawingIssue[]; warnings: DrawingIssue[]};

export type DrawingValidationOptions = {
  maximumVertices?: number;
  /** Rings with more vertices skip the quadratic simplicity check and emit a warning. */
  maximumSimplicityVertices?: number;
};

const EARTH_RADIUS_METERS = MEAN_EARTH_RADIUS_METERS;

/**
 * Validates a drawn polygon-like feature for use as a Kepler polygon filter. The accepted frame is
 * native Kepler's raw planar [longitude, latitude] degrees: antimeridian-spanning edges, longitudes
 * beyond +-180 and polar latitudes are ordinary vertices and are reported as warnings, never split.
 * Rectangles are Polygon features and use the same path.
 */
export function validateDrawnPolygon(
  feature: Feature<Geometry> | Geometry | null | undefined,
  options: DrawingValidationOptions = {}
): DrawingValidation {
  const reasons: DrawingIssue[] = [];
  const warnings: DrawingIssue[] = [];
  const maximumVertices = options.maximumVertices ?? 100000;
  const maximumSimplicityVertices = options.maximumSimplicityVertices ?? 2048;
  const geometry = feature && feature.type === 'Feature' ? feature.geometry : feature;
  const fail = (code: DrawingIssueCode, message: string) => reasons.push({code, message});
  let polygons: Ring[][] = [];
  if (!geometry) fail('empty-geometry', 'The drawn shape has no geometry.');
  else if (geometry.type === 'Polygon') polygons = [readCoordinates(geometry.coordinates)];
  else if (geometry.type === 'MultiPolygon')
    polygons = geometry.coordinates.map(coordinates => readCoordinates(coordinates));
  else
    fail(
      'unsupported-geometry',
      `Only Polygon and MultiPolygon shapes can filter data; this shape is ${geometry.type}.`
    );
  if (reasons.length) return {status: 'invalid', reasons, warnings};
  if (!polygons.length || polygons.some(polygon => !polygon.length)) {
    fail('empty-geometry', 'The drawn shape has no rings.');
    return {status: 'invalid', reasons, warnings};
  }
  let vertexCount = 0;
  const seen = new Set<DrawingIssueCode>();
  const once = (target: DrawingIssue[], code: DrawingIssueCode, message: string) => {
    if (seen.has(code)) return;
    seen.add(code);
    target.push({code, message});
  };
  for (const polygon of polygons) {
    for (const ring of polygon) vertexCount += ring.length;
  }
  if (vertexCount > maximumVertices) {
    fail(
      'vertex-budget',
      `The drawn shape has ${vertexCount} vertices; the limit is ${maximumVertices}.`
    );
    return {status: 'invalid', reasons, warnings};
  }
  for (const polygon of polygons) {
    const cleanedRings: Ring[] = [];
    for (const ring of polygon) {
      if (ring.some(([x, y]) => !Number.isFinite(x) || !Number.isFinite(y))) {
        once(reasons, 'non-finite-coordinate', 'A drawn vertex is not a finite number.');
        continue;
      }
      // Compare in the Float32 frame the GPU mask uses, so collapsed vertices are reported here.
      const cleaned = removeClosure(
        ring.map(([x, y]): Position => [Math.fround(x), Math.fround(y)])
      );
      const distinct = new Set(cleaned.map(([x, y]) => `${x},${y}`)).size;
      if (cleaned.length < 3) {
        once(reasons, 'too-few-vertices', 'A ring needs at least three vertices.');
        continue;
      }
      if (distinct < 3) {
        once(
          reasons,
          'collapsed-ring',
          'A ring has fewer than three distinct vertices at GPU precision.'
        );
        continue;
      }
      if (isRingCollinear(cleaned)) {
        once(reasons, 'zero-area', 'A ring has zero area; its vertices are collinear.');
        continue;
      }
      if (cleaned.length <= maximumSimplicityVertices) {
        if (isRingSelfIntersecting(cleaned))
          once(reasons, 'self-intersection', 'A ring crosses or touches itself.');
      } else
        once(
          warnings,
          'self-intersection-unchecked',
          'A ring is too large for the interactive simplicity check.'
        );
      for (const [x, y] of cleaned) {
        if (x < -180 || x > 180)
          once(
            warnings,
            'longitude-out-of-range',
            'Longitude beyond +-180 is treated as an ordinary planar vertex, as in native Kepler.'
          );
        if (y < -90 || y > 90)
          once(
            warnings,
            'latitude-out-of-range',
            'Latitude beyond +-90 is treated as an ordinary planar vertex, as in native Kepler.'
          );
      }
      for (let index = 0; index < cleaned.length; index++) {
        if (Math.abs(cleaned[index][0] - cleaned[(index + 1) % cleaned.length][0]) >= 180)
          once(
            warnings,
            'antimeridian-edge',
            'An edge spans 180 degrees of longitude and is filtered as a straight planar edge; no antimeridian split is applied.'
          );
      }
      cleanedRings.push(cleaned);
    }
    if (cleanedRings.length === polygon.length) {
      const [shell, ...holes] = cleanedRings;
      for (const hole of holes) {
        if (ringsCrossProperly(shell, hole))
          once(reasons, 'hole-crosses-shell', 'A hole crosses the polygon boundary.');
        else if (!hole.every(position => isPointInRing(position, shell)))
          once(reasons, 'hole-outside-shell', 'A hole lies outside its polygon shell.');
      }
    }
  }
  return reasons.length
    ? {status: 'invalid', reasons, warnings}
    : {status: 'valid', polygons, warnings};
}

function readCoordinates(coordinates: unknown): Ring[] {
  if (!Array.isArray(coordinates)) return [];
  return coordinates.map((ring: unknown) =>
    Array.isArray(ring)
      ? ring.map(
          (position: unknown): Position =>
            Array.isArray(position)
              ? [Number(position[0]), Number(position[1])]
              : [Number.NaN, Number.NaN]
        )
      : []
  );
}

function removeClosure(ring: Ring): Ring {
  const last = ring.length - 1;
  return ring.length > 1 && ring[0][0] === ring[last][0] && ring[0][1] === ring[last][1]
    ? ring.slice(0, last)
    : ring;
}

function isRingCollinear(ring: Ring): boolean {
  const second = ring.find(position => position[0] !== ring[0][0] || position[1] !== ring[0][1]);
  return !second || ring.every(position => getOrientation(ring[0], second, position) === 0);
}

function getOrientation(a: Position, b: Position, c: Position): number {
  const value = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
  return value > 0 ? 1 : value < 0 ? -1 : 0;
}

function isOnSegment(a: Position, b: Position, point: Position): boolean {
  return (
    Math.min(a[0], b[0]) <= point[0] &&
    point[0] <= Math.max(a[0], b[0]) &&
    Math.min(a[1], b[1]) <= point[1] &&
    point[1] <= Math.max(a[1], b[1])
  );
}

/** True when segments share any point, including endpoint touches and collinear overlap. */
function doSegmentsIntersect(a: Position, b: Position, c: Position, d: Position): boolean {
  const o1 = getOrientation(a, b, c);
  const o2 = getOrientation(a, b, d);
  const o3 = getOrientation(c, d, a);
  const o4 = getOrientation(c, d, b);
  if (o1 !== o2 && o3 !== o4) return true;
  return (
    (o1 === 0 && isOnSegment(a, b, c)) ||
    (o2 === 0 && isOnSegment(a, b, d)) ||
    (o3 === 0 && isOnSegment(c, d, a)) ||
    (o4 === 0 && isOnSegment(c, d, b))
  );
}

function doSegmentsCrossProperly(a: Position, b: Position, c: Position, d: Position): boolean {
  const o1 = getOrientation(a, b, c);
  const o2 = getOrientation(a, b, d);
  const o3 = getOrientation(c, d, a);
  const o4 = getOrientation(c, d, b);
  return o1 !== 0 && o2 !== 0 && o3 !== 0 && o4 !== 0 && o1 !== o2 && o3 !== o4;
}

function isRingSelfIntersecting(ring: Ring): boolean {
  const count = ring.length;
  for (let first = 0; first < count; first++) {
    const a = ring[first];
    const b = ring[(first + 1) % count];
    for (let second = first + 1; second < count; second++) {
      const adjacent = second === first + 1 || (first === 0 && second === count - 1);
      const c = ring[second];
      const d = ring[(second + 1) % count];
      if (adjacent) {
        // Adjacent edges may only share their joint; collinear backtracking overlaps further.
        const [previous, joint, following] = second === first + 1 ? [a, b, d] : [c, a, b];
        if (
          getOrientation(previous, joint, following) === 0 &&
          (joint[0] - previous[0]) * (following[0] - joint[0]) +
            (joint[1] - previous[1]) * (following[1] - joint[1]) <
            0
        )
          return true;
        continue;
      }
      if (doSegmentsIntersect(a, b, c, d)) return true;
    }
  }
  return false;
}

function ringsCrossProperly(first: Ring, second: Ring): boolean {
  for (let a = 0; a < first.length; a++)
    for (let b = 0; b < second.length; b++)
      if (
        doSegmentsCrossProperly(
          first[a],
          first[(a + 1) % first.length],
          second[b],
          second[(b + 1) % second.length]
        )
      )
        return true;
  return false;
}

/** Boundary counts as inside, matching the closed boundary rule of the GPU mask. */
export function isPointInRing(point: Position, ring: Ring): boolean {
  let inside = false;
  for (let index = 0, previous = ring.length - 1; index < ring.length; previous = index++) {
    const a = ring[index];
    const b = ring[previous];
    if (getOrientation(a, b, point) === 0 && isOnSegment(a, b, point)) return true;
    if (a[1] > point[1] !== b[1] > point[1]) {
      const crossing = ((b[0] - a[0]) * (point[1] - a[1])) / (b[1] - a[1]) + a[0];
      if (point[0] < crossing) inside = !inside;
    }
  }
  return inside;
}

export type DrawingMeasurement = {
  /** Geodesic length of a line, or perimeter of a polygon, in meters. */
  lengthMeters: number;
  /** Spherical polygon area in square meters; holes subtract. Zero for lines and points. */
  areaSquareMeters: number;
  vertexCount: number;
};

/** Haversine length and Chamberlain-Duquette area, matching turf-style web measurements. */
export function measureDrawnGeometry(geometry: Geometry): DrawingMeasurement {
  const result: DrawingMeasurement = {lengthMeters: 0, areaSquareMeters: 0, vertexCount: 0};
  const visit = (value: Geometry): void => {
    switch (value.type) {
      case 'Point':
        result.vertexCount += 1;
        break;
      case 'MultiPoint':
        result.vertexCount += value.coordinates.length;
        break;
      case 'LineString':
        measurePath(value.coordinates, false, result);
        break;
      case 'MultiLineString':
        for (const line of value.coordinates) measurePath(line, false, result);
        break;
      case 'Polygon':
        measurePolygon(value.coordinates, result);
        break;
      case 'MultiPolygon':
        for (const polygon of value.coordinates) measurePolygon(polygon, result);
        break;
      case 'GeometryCollection':
        for (const child of value.geometries) visit(child);
    }
  };
  visit(geometry);
  return result;
}

function measurePath(path: Ring, closed: boolean, result: DrawingMeasurement): void {
  result.vertexCount += path.length;
  const count = closed ? path.length : path.length - 1;
  for (let index = 0; index < count; index++)
    result.lengthMeters += getHaversineMeters(path[index], path[(index + 1) % path.length]);
}

function measurePolygon(rings: Position[][], result: DrawingMeasurement): void {
  rings.forEach((ring, ringIndex) => {
    const cleaned = removeClosure(ring);
    measurePath(cleaned, true, result);
    const area = getSphericalRingArea(cleaned);
    result.areaSquareMeters += ringIndex === 0 ? area : -area;
  });
}

function getHaversineMeters(a: Position, b: Position): number {
  const deltaLatitude = toRadians(b[1] - a[1]);
  const deltaLongitude = toRadians(b[0] - a[0]);
  const h =
    Math.sin(deltaLatitude / 2) ** 2 +
    Math.cos(toRadians(a[1])) * Math.cos(toRadians(b[1])) * Math.sin(deltaLongitude / 2) ** 2;
  return 2 * EARTH_RADIUS_METERS * Math.asin(Math.min(1, Math.sqrt(h)));
}

function getSphericalRingArea(ring: Ring): number {
  let total = 0;
  for (let index = 0; index < ring.length; index++) {
    const a = ring[index];
    const b = ring[(index + 1) % ring.length];
    total += toRadians(b[0] - a[0]) * (2 + Math.sin(toRadians(a[1])) + Math.sin(toRadians(b[1])));
  }
  return Math.abs((total * EARTH_RADIUS_METERS * EARTH_RADIUS_METERS) / 2);
}

/**
 * Select-by-click for drawn editor features. Hit-tests in the same planar degree frame as the
 * filter masks (no antimeridian wrapping); the last (top-most) matching feature wins.
 * Returns -1 when nothing is within tolerance.
 */
export function findDrawnFeatureAtPoint(
  features: readonly Feature<Geometry>[],
  point: Position,
  toleranceDegrees: number
): number {
  for (let index = features.length - 1; index >= 0; index--) {
    if (isGeometryHit(features[index].geometry, point, toleranceDegrees)) return index;
  }
  return -1;
}

function isGeometryHit(geometry: Geometry | null, point: Position, tolerance: number): boolean {
  if (!geometry) return false;
  switch (geometry.type) {
    case 'Point':
      return getDistance(geometry.coordinates, point) <= tolerance;
    case 'MultiPoint':
      return geometry.coordinates.some(c => getDistance(c, point) <= tolerance);
    case 'LineString':
      return isPathHit(geometry.coordinates, point, tolerance);
    case 'MultiLineString':
      return geometry.coordinates.some(line => isPathHit(line, point, tolerance));
    case 'Polygon':
      return isPolygonHit(geometry.coordinates, point, tolerance);
    case 'MultiPolygon':
      return geometry.coordinates.some(polygon => isPolygonHit(polygon, point, tolerance));
    case 'GeometryCollection':
      return geometry.geometries.some(child => isGeometryHit(child, point, tolerance));
  }
}

function isPolygonHit(rings: Position[][], point: Position, tolerance: number): boolean {
  if (rings.some(ring => isPathHit(ring, point, tolerance))) return true;
  if (!rings.length || !isPointInRing(point, removeClosure(rings[0]))) return false;
  return !rings.slice(1).some(hole => isPointInRing(point, removeClosure(hole)));
}

function isPathHit(path: Position[], point: Position, tolerance: number): boolean {
  for (let index = 0; index + 1 < path.length; index++)
    if (getSegmentDistance(path[index], path[index + 1], point) <= tolerance) return true;
  return false;
}

function getDistance(a: Position, b: Position): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

function getSegmentDistance(a: Position, b: Position, point: Position): number {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const lengthSquared = dx * dx + dy * dy;
  const t = lengthSquared
    ? clamp(((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / lengthSquared, 0, 1)
    : 0;
  return getDistance([a[0] + t * dx, a[1] + t * dy], point);
}
