// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {LocalFrame} from './catalog-local-frame';
import type {
  CatalogInputSpec,
  CatalogOutputSpec,
  CatalogParameterSpec,
  CatalogParameterValue
} from './catalog-binding-types';
import {
  getFeatureIdentity,
  getLineParts,
  getPointPositions,
  getPolygonParts,
  type CatalogFeature,
  type CatalogFeatureCollection,
  type CatalogPosition
} from './catalog-geojson';

export type Inputs = Readonly<Record<string, CatalogFeatureCollection>>;
export type Values = Readonly<Record<string, CatalogParameterValue>>;
export type PlanarPoint = [number, number];

export function toPlanar(position: CatalogPosition): PlanarPoint {
  return [position[0], position[1]];
}
export type Bounds = [number, number, number, number];

export const CPU_CLEANUP =
  'CPU only; no retained GPU resources. Published results are removed through native Sources.';

export const GPU_OR_CPU_CLEANUP =
  'CPU engine retains nothing. The GPU engine creates storage buffers and compiled graphs per run and destroys them before the result returns; results are plain arrays and publication goes through native Sources.';

export const DEFAULT_MAXIMUM_FEATURES = 20000;
export const DEFAULT_MAXIMUM_VERTICES = 200000;

export function makeInputSpec(
  name: string,
  label: string,
  accepts: CatalogInputSpec['accepts'],
  maximumFeatures = DEFAULT_MAXIMUM_FEATURES,
  maximumVertices = DEFAULT_MAXIMUM_VERTICES
): CatalogInputSpec {
  return {name, label, accepts, maximumFeatures, maximumVertices};
}

export function makeOutputSpec(
  tableId: string,
  label: string,
  geometry: CatalogOutputSpec['geometry'],
  identityColumns: readonly string[],
  maximumRows = DEFAULT_MAXIMUM_FEATURES
): CatalogOutputSpec {
  return {tableId, label, geometry, identityColumns, maximumRows};
}

export function makeNumberParameter(
  name: string,
  label: string,
  defaultValue: number,
  minimum: number,
  maximum: number,
  unit?: string
): CatalogParameterSpec {
  return {name, label, kind: 'number', unit, minimum, maximum, defaultValue};
}

export function makeIntegerParameter(
  name: string,
  label: string,
  defaultValue: number,
  minimum: number,
  maximum: number
): CatalogParameterSpec {
  return {name, label, kind: 'integer', minimum, maximum, defaultValue};
}

export function makeStringParameter(
  name: string,
  label: string,
  defaultValue = ''
): CatalogParameterSpec {
  return {name, label, kind: 'string', maximumLength: 64, defaultValue};
}

export const SEED_PARAMETER_SPEC: CatalogParameterSpec = {
  name: 'seed',
  label: 'Random seed',
  kind: 'integer',
  minimum: 0,
  maximum: 4294967295,
  defaultValue: 1
};

export function getSourceColumns(
  feature: CatalogFeature,
  sourceIndex: number
): {source_index: number; source_id: string | null} {
  return {source_index: sourceIndex, source_id: getFeatureIdentity(feature)};
}

/** Every finite vertex of the collection, in feature order. */
export function collectPositions(collection: CatalogFeatureCollection): CatalogPosition[] {
  const positions: CatalogPosition[] = [];
  for (const feature of collection.features) {
    for (const position of getPointPositions(feature.geometry)) positions.push(position);
    for (const part of getLineParts(feature.geometry)) positions.push(...part);
    for (const polygon of getPolygonParts(feature.geometry))
      for (const ring of polygon) positions.push(...ring);
  }
  return positions.filter(position => Number.isFinite(position[0]) && Number.isFinite(position[1]));
}

export function getFeatureVertices(feature: CatalogFeature): CatalogPosition[] {
  return collectPositions({type: 'FeatureCollection', features: [feature]});
}

export function getPositionBounds(positions: readonly CatalogPosition[]): Bounds | null {
  if (!positions.length) return null;
  let minimumX = Infinity;
  let minimumY = Infinity;
  let maximumX = -Infinity;
  let maximumY = -Infinity;
  for (const [x, y] of positions) {
    if (x < minimumX) minimumX = x;
    if (x > maximumX) maximumX = x;
    if (y < minimumY) minimumY = y;
    if (y > maximumY) maximumY = y;
  }
  return [minimumX, minimumY, maximumX, maximumY];
}

export function getFeatureBounds(feature: CatalogFeature): Bounds | null {
  return getPositionBounds(getFeatureVertices(feature));
}

export function readPropertyText(feature: CatalogFeature, name: string): string {
  const value = name ? feature.properties?.[name] : undefined;
  return value === undefined || value === null ? '' : String(value);
}

export function readPropertyNumber(feature: CatalogFeature, name: string): number | null {
  const value = name ? feature.properties?.[name] : undefined;
  const number = typeof value === 'string' && value.trim() ? Number(value) : value;
  return typeof number === 'number' && Number.isFinite(number) ? number : null;
}

export function getSignedArea(ring: readonly PlanarPoint[]): number {
  let sum = 0;
  for (let index = 0; index < ring.length; index++) {
    const [firstX, firstY] = ring[index];
    const [secondX, secondY] = ring[(index + 1) % ring.length];
    sum += firstX * secondY - secondX * firstY;
  }
  return sum / 2;
}

export function isPointInRing(x: number, y: number, ring: readonly CatalogPosition[]): boolean {
  let inside = false;
  for (let index = 0, previous = ring.length - 1; index < ring.length; previous = index++) {
    const [firstX, firstY] = ring[index];
    const [secondX, secondY] = ring[previous];
    if (
      firstY > y !== secondY > y &&
      x < ((secondX - firstX) * (y - firstY)) / (secondY - firstY) + firstX
    )
      inside = !inside;
  }
  return inside;
}

/** Even-odd containment against a polygon (shell first, then holes). */
export function isPointInPolygonRings(
  x: number,
  y: number,
  rings: readonly (readonly PlanarPoint[])[]
): boolean {
  let inside = false;
  for (const ring of rings) if (isPointInRing(x, y, ring)) inside = !inside;
  return inside;
}

export function getDistanceToSegment(
  x: number,
  y: number,
  firstX: number,
  firstY: number,
  secondX: number,
  secondY: number
): {distance: number; fraction: number; x: number; y: number} {
  const deltaX = secondX - firstX;
  const deltaY = secondY - firstY;
  const lengthSquared = deltaX * deltaX + deltaY * deltaY;
  const fraction =
    lengthSquared === 0
      ? 0
      : Math.max(0, Math.min(1, ((x - firstX) * deltaX + (y - firstY) * deltaY) / lengthSquared));
  const nearestX = firstX + fraction * deltaX;
  const nearestY = firstY + fraction * deltaY;
  return {distance: Math.hypot(x - nearestX, y - nearestY), fraction, x: nearestX, y: nearestY};
}

export type SegmentIntersection = {
  firstFraction: number;
  secondFraction: number;
  x: number;
  y: number;
};

/** Intersection of two non-parallel segments, endpoints included; null for parallel or disjoint. */
export function intersectSegments(
  first: readonly [PlanarPoint, PlanarPoint],
  second: readonly [PlanarPoint, PlanarPoint]
): SegmentIntersection | null {
  const [[ax, ay], [bx, by]] = first;
  const [[cx, cy], [dx, dy]] = second;
  const firstDeltaX = bx - ax;
  const firstDeltaY = by - ay;
  const secondDeltaX = dx - cx;
  const secondDeltaY = dy - cy;
  const denominator = firstDeltaX * secondDeltaY - firstDeltaY * secondDeltaX;
  if (denominator === 0) return null;
  const firstFraction = ((cx - ax) * secondDeltaY - (cy - ay) * secondDeltaX) / denominator;
  const secondFraction = ((cx - ax) * firstDeltaY - (cy - ay) * firstDeltaX) / denominator;
  if (firstFraction < 0 || firstFraction > 1 || secondFraction < 0 || secondFraction > 1)
    return null;
  return {
    firstFraction,
    secondFraction,
    x: ax + firstFraction * firstDeltaX,
    y: ay + firstFraction * firstDeltaY
  };
}

/** Counter-clockwise convex hull (monotone chain) without a repeated closing point. */
export function getConvexHull(points: readonly PlanarPoint[]): PlanarPoint[] {
  const sorted = [...points].sort((first, second) => first[0] - second[0] || first[1] - second[1]);
  const unique = sorted.filter(
    (point, index) =>
      index === 0 || point[0] !== sorted[index - 1][0] || point[1] !== sorted[index - 1][1]
  );
  if (unique.length < 3) return unique;
  const cross = (origin: PlanarPoint, first: PlanarPoint, second: PlanarPoint) =>
    (first[0] - origin[0]) * (second[1] - origin[1]) -
    (first[1] - origin[1]) * (second[0] - origin[0]);
  const lower: PlanarPoint[] = [];
  for (const point of unique) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], point) <= 0)
      lower.pop();
    lower.push(point);
  }
  const upper: PlanarPoint[] = [];
  for (let index = unique.length - 1; index >= 0; index--) {
    const point = unique[index];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], point) <= 0)
      upper.pop();
    upper.push(point);
  }
  lower.pop();
  upper.pop();
  return [...lower, ...upper];
}

export function closeRing<Point extends readonly [number, number]>(ring: Point[]): Point[] {
  if (!ring.length) return ring;
  const first = ring[0];
  const last = ring[ring.length - 1];
  return first[0] === last[0] && first[1] === last[1] ? ring : [...ring, first];
}

/** Ring without its closing vertex, projected into the frame. */
export function projectRing(ring: readonly CatalogPosition[], frame: LocalFrame): PlanarPoint[] {
  const projected = ring.map(position => frame.forward(position));
  if (
    projected.length > 1 &&
    projected[0][0] === projected[projected.length - 1][0] &&
    projected[0][1] === projected[projected.length - 1][1]
  )
    projected.pop();
  return projected;
}

/** Polygon parts of a feature projected into the frame: parts -> rings -> open rings. */
export function projectPolygonParts(feature: CatalogFeature, frame: LocalFrame): PlanarPoint[][][] {
  return getPolygonParts(feature.geometry)
    .map(polygon => polygon.map(ring => projectRing(ring, frame)).filter(ring => ring.length >= 3))
    .filter(polygon => polygon.length);
}

export function makeCirclePositions(
  centerX: number,
  centerY: number,
  radius: number,
  segmentCount: number
): PlanarPoint[] {
  const ring: PlanarPoint[] = [];
  for (let index = 0; index < segmentCount; index++) {
    const angle = (2 * Math.PI * index) / segmentCount;
    ring.push([centerX + radius * Math.cos(angle), centerY + radius * Math.sin(angle)]);
  }
  return ring;
}

export function sortByNumber<Item>(items: Item[], getKey: (item: Item) => number): Item[] {
  return items.sort((first, second) => getKey(first) - getKey(second));
}
