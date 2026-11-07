// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {clamp, toDegrees, toRadians, vec3} from '@math.gl/core';
import {KeplerAllocationError} from '../runtime/dataset-registry';
import type {CatalogRunOutput} from './catalog-binding-types';
import {
  EARTH_RADIUS_METERS,
  addCatalogDiagnostic,
  appendCatalogRow,
  getLineParts,
  getPointPositions,
  getPolygonParts,
  makeCatalogGeometryText,
  makeCatalogTable,
  measurePathLength,
  type CatalogCoordinateSystem,
  type CatalogDiagnostic,
  type CatalogFeature,
  type CatalogFeatureCollection,
  type CatalogGeometry,
  type CatalogPosition
} from './catalog-geojson';
import {makeSeededRandom} from './catalog-extra-geometry';
import {
  closeRing,
  collectPositions,
  getDistanceToSegment,
  getSignedArea,
  getSourceColumns,
  intersectSegments,
  isPointInPolygonRings,
  makeCirclePositions,
  projectPolygonParts,
  readPropertyNumber,
  type PlanarPoint,
  toPlanar
} from './catalog-platform3-a-shared';
import {addLocalFrameDiagnostic, makeLocalFrame} from './catalog-local-frame';

function isSamePosition(first: CatalogPosition, second: CatalogPosition): boolean {
  return first[0] === second[0] && first[1] === second[1];
}

// ---------------------------------------------------------------------------------------------
// Line split at points
// ---------------------------------------------------------------------------------------------

function interpolateAlong(
  positions: readonly CatalogPosition[],
  cumulative: readonly number[],
  distance: number
): CatalogPosition {
  for (let index = 1; index < positions.length; index++)
    if (distance <= cumulative[index] || index === positions.length - 1) {
      const span = cumulative[index] - cumulative[index - 1];
      const fraction =
        span > 0 ? Math.max(0, Math.min(1, (distance - cumulative[index - 1]) / span)) : 0;
      const first = positions[index - 1];
      const second = positions[index];
      return [
        first[0] + (second[0] - first[0]) * fraction,
        first[1] + (second[1] - first[1]) * fraction
      ];
    }
  return positions[0];
}

function sliceAlong(
  positions: readonly CatalogPosition[],
  cumulative: readonly number[],
  from: number,
  to: number
): CatalogPosition[] {
  const result: CatalogPosition[] = [interpolateAlong(positions, cumulative, from)];
  for (let index = 1; index < positions.length - 1; index++)
    if (cumulative[index] > from && cumulative[index] < to) result.push(positions[index]);
  result.push(interpolateAlong(positions, cumulative, to));
  return result;
}

export function runLineSplit(
  lines: CatalogFeatureCollection,
  splitters: CatalogFeatureCollection,
  parameters: {tolerance: number; coordinateSystem: CatalogCoordinateSystem}
): CatalogRunOutput {
  const diagnostics: CatalogDiagnostic[] = [];
  const frame = makeLocalFrame(parameters.coordinateSystem, collectPositions(lines));
  const splitPositions = splitters.features.flatMap(feature => getPointPositions(feature.geometry));
  const matched = new Set<number>();
  const table = makeCatalogTable(
    'line-pieces',
    'Line pieces',
    {
      piece_id: 'string',
      source_index: 'int32',
      source_id: 'string',
      part_index: 'int32',
      piece_index: 'int32',
      piece_count: 'int32',
      length: 'float64',
      geometry: 'string'
    },
    'geometry'
  );
  lines.features.forEach((feature, sourceIndex) => {
    getLineParts(feature.geometry).forEach((part, partIndex) => {
      const positions = part.filter(
        (position, index) => index === 0 || !isSamePosition(position, part[index - 1])
      );
      if (positions.length < 2) return;
      const projected = positions.map(position => frame.forward(position));
      const cumulative = [0];
      for (let index = 1; index < projected.length; index++)
        cumulative.push(
          cumulative[index - 1] +
            Math.hypot(
              projected[index][0] - projected[index - 1][0],
              projected[index][1] - projected[index - 1][1]
            )
        );
      const total = cumulative[cumulative.length - 1];
      const cuts: number[] = [];
      splitPositions.forEach((position, splitterIndex) => {
        const [x, y] = frame.forward(position);
        let bestDistance = Infinity;
        let bestAlong = 0;
        for (let index = 1; index < projected.length; index++) {
          const found = getDistanceToSegment(
            x,
            y,
            projected[index - 1][0],
            projected[index - 1][1],
            projected[index][0],
            projected[index][1]
          );
          if (found.distance < bestDistance) {
            bestDistance = found.distance;
            bestAlong =
              cumulative[index - 1] + found.fraction * (cumulative[index] - cumulative[index - 1]);
          }
        }
        if (bestDistance <= parameters.tolerance) {
          matched.add(splitterIndex);
          if (
            bestAlong > 1e-9 * Math.max(total, 1) &&
            bestAlong < total - 1e-9 * Math.max(total, 1)
          )
            cuts.push(bestAlong);
        }
      });
      cuts.sort((first, second) => first - second);
      const bounds = [
        0,
        ...cuts.filter((cut, index) => index === 0 || cut - cuts[index - 1] > 1e-9),
        total
      ];
      const pieceCount = bounds.length - 1;
      for (let pieceIndex = 0; pieceIndex < pieceCount; pieceIndex++) {
        const coordinates = sliceAlong(
          positions,
          cumulative,
          bounds[pieceIndex],
          bounds[pieceIndex + 1]
        );
        appendCatalogRow(table, {
          piece_id: `${sourceIndex}:${partIndex}:${pieceIndex}`,
          ...getSourceColumns(feature, sourceIndex),
          part_index: partIndex,
          piece_index: pieceIndex,
          piece_count: pieceCount,
          length: measurePathLength(coordinates, parameters.coordinateSystem),
          geometry: makeCatalogGeometryText('LineString', coordinates)
        });
      }
    });
  });
  if (matched.size < splitPositions.length)
    addCatalogDiagnostic(
      diagnostics,
      'splitter-unmatched',
      'warning',
      'Splitter points farther than the tolerance from every line cut nothing.',
      splitPositions.length - matched.size
    );
  return {
    tables: [table],
    diagnostics,
    summary: {
      pieceCount: table.rowCount,
      splitterCount: splitPositions.length,
      matchedCount: matched.size
    }
  };
}

// ---------------------------------------------------------------------------------------------
// Line clip by polygon
// ---------------------------------------------------------------------------------------------

export function runLineClipByPolygon(
  lines: CatalogFeatureCollection,
  polygons: CatalogFeatureCollection,
  parameters: {keep: 'inside' | 'outside' | 'both'; coordinateSystem: CatalogCoordinateSystem}
): CatalogRunOutput {
  const diagnostics: CatalogDiagnostic[] = [];
  type Part = {rings: PlanarPoint[][]; bounds: [number, number, number, number]};
  const parts: Part[] = [];
  for (const feature of polygons.features)
    for (const polygon of getPolygonParts(feature.geometry)) {
      const rings = polygon.map(ring => ring.map(toPlanar)).filter(ring => ring.length >= 4);
      if (!rings.length) continue;
      const shell = rings[0];
      parts.push({
        rings,
        bounds: [
          Math.min(...shell.map(point => point[0])),
          Math.min(...shell.map(point => point[1])),
          Math.max(...shell.map(point => point[0])),
          Math.max(...shell.map(point => point[1]))
        ]
      });
    }
  const isInside = (x: number, y: number) =>
    parts.some(
      part =>
        x >= part.bounds[0] &&
        x <= part.bounds[2] &&
        y >= part.bounds[1] &&
        y <= part.bounds[3] &&
        isPointInPolygonRings(x, y, part.rings)
    );
  const table = makeCatalogTable(
    'clipped-lines',
    'Clipped lines',
    {
      piece_id: 'string',
      source_index: 'int32',
      source_id: 'string',
      part_index: 'int32',
      piece_index: 'int32',
      side: 'string',
      length: 'float64',
      geometry: 'string'
    },
    'geometry'
  );
  lines.features.forEach((feature, sourceIndex) => {
    getLineParts(feature.geometry).forEach((part, partIndex) => {
      const pieces: {side: 'inside' | 'outside'; coordinates: CatalogPosition[]}[] = [];
      for (let index = 1; index < part.length; index++) {
        const start = part[index - 1];
        const end = part[index];
        if (isSamePosition(start, end)) continue;
        const fractions = [0, 1];
        const minimumX = Math.min(start[0], end[0]);
        const maximumX = Math.max(start[0], end[0]);
        const minimumY = Math.min(start[1], end[1]);
        const maximumY = Math.max(start[1], end[1]);
        for (const polygon of parts) {
          if (
            polygon.bounds[2] < minimumX ||
            polygon.bounds[0] > maximumX ||
            polygon.bounds[3] < minimumY ||
            polygon.bounds[1] > maximumY
          )
            continue;
          for (const ring of polygon.rings)
            for (let ringIndex = 1; ringIndex < ring.length; ringIndex++) {
              const hit = intersectSegments(
                [toPlanar(start), toPlanar(end)],
                [ring[ringIndex - 1], ring[ringIndex]]
              );
              if (hit && hit.firstFraction > 0 && hit.firstFraction < 1)
                fractions.push(hit.firstFraction);
            }
        }
        fractions.sort((first, second) => first - second);
        for (let fractionIndex = 1; fractionIndex < fractions.length; fractionIndex++) {
          const from = fractions[fractionIndex - 1];
          const to = fractions[fractionIndex];
          if (to - from < 1e-12) continue;
          const middle = (from + to) / 2;
          const side = isInside(
            start[0] + (end[0] - start[0]) * middle,
            start[1] + (end[1] - start[1]) * middle
          )
            ? 'inside'
            : 'outside';
          const first: CatalogPosition = [
            start[0] + (end[0] - start[0]) * from,
            start[1] + (end[1] - start[1]) * from
          ];
          const second: CatalogPosition = [
            start[0] + (end[0] - start[0]) * to,
            start[1] + (end[1] - start[1]) * to
          ];
          const last = pieces[pieces.length - 1];
          if (
            last &&
            last.side === side &&
            isSamePosition(last.coordinates[last.coordinates.length - 1], first)
          )
            last.coordinates.push(second);
          else pieces.push({side, coordinates: [first, second]});
        }
      }
      pieces
        .filter(piece => parameters.keep === 'both' || piece.side === parameters.keep)
        .forEach((piece, pieceIndex) => {
          appendCatalogRow(table, {
            piece_id: `${sourceIndex}:${partIndex}:${pieceIndex}`,
            ...getSourceColumns(feature, sourceIndex),
            part_index: partIndex,
            piece_index: pieceIndex,
            side: piece.side,
            length: measurePathLength(piece.coordinates, parameters.coordinateSystem),
            geometry: makeCatalogGeometryText('LineString', piece.coordinates)
          });
        });
    });
  });
  addCatalogDiagnostic(
    diagnostics,
    'planar-clip',
    'info',
    'Crossings are computed on the raw coordinates as planar segments (also for EPSG:4326); lengths follow the declared coordinate system. Overlapping polygons count as one region.'
  );
  return {
    tables: [table],
    diagnostics,
    summary: {pieceCount: table.rowCount, polygonPartCount: parts.length}
  };
}

// ---------------------------------------------------------------------------------------------
// Segment intersections
// ---------------------------------------------------------------------------------------------

export const MAXIMUM_INTERSECTION_POINTS = 50000;

export function runSegmentIntersections(lines: CatalogFeatureCollection): CatalogRunOutput {
  type Segment = {
    featureIndex: number;
    partIndex: number;
    segmentIndex: number;
    segmentCount: number;
    closed: boolean;
    start: PlanarPoint;
    end: PlanarPoint;
    minimumX: number;
    maximumX: number;
    minimumY: number;
    maximumY: number;
  };
  const segments: Segment[] = [];
  lines.features.forEach((feature, featureIndex) => {
    getLineParts(feature.geometry).forEach((part, partIndex) => {
      const closed = part.length > 3 && isSamePosition(part[0], part[part.length - 1]);
      for (let index = 1; index < part.length; index++) {
        if (isSamePosition(part[index - 1], part[index])) continue;
        const start = toPlanar(part[index - 1]);
        const end = toPlanar(part[index]);
        segments.push({
          featureIndex,
          partIndex,
          segmentIndex: index - 1,
          segmentCount: part.length - 1,
          closed,
          start,
          end,
          minimumX: Math.min(start[0], end[0]),
          maximumX: Math.max(start[0], end[0]),
          minimumY: Math.min(start[1], end[1]),
          maximumY: Math.max(start[1], end[1])
        });
      }
    });
  });
  segments.sort((first, second) => first.minimumX - second.minimumX);
  const table = makeCatalogTable(
    'segment-intersections',
    'Segment intersections',
    {
      intersection_id: 'string',
      first_index: 'int32',
      first_id: 'string',
      second_index: 'int32',
      second_id: 'string',
      kind: 'string',
      geometry: 'string'
    },
    'geometry'
  );
  const diagnostics: CatalogDiagnostic[] = [];
  for (let first = 0; first < segments.length; first++) {
    const a = segments[first];
    for (let second = first + 1; second < segments.length; second++) {
      const b = segments[second];
      if (b.minimumX > a.maximumX) break;
      if (b.maximumY < a.minimumY || b.minimumY > a.maximumY) continue;
      const sameLine = a.featureIndex === b.featureIndex && a.partIndex === b.partIndex;
      if (sameLine) {
        const gap = Math.abs(a.segmentIndex - b.segmentIndex);
        if (gap <= 1) continue;
        if (a.closed && gap === a.segmentCount - 1) continue;
      }
      const hit = intersectSegments([a.start, a.end], [b.start, b.end]);
      if (!hit) continue;
      if (table.rowCount >= MAXIMUM_INTERSECTION_POINTS)
        throw new KeplerAllocationError('capacity', {message: 'Intersection capacity exceeded'});
      const interior =
        hit.firstFraction > 0 &&
        hit.firstFraction < 1 &&
        hit.secondFraction > 0 &&
        hit.secondFraction < 1;
      const [firstIdentity, secondIdentity] = [a, b].map(
        segment => lines.features[segment.featureIndex]
      );
      appendCatalogRow(table, {
        intersection_id: `${table.rowCount}`,
        first_index: a.featureIndex,
        first_id:
          firstIdentity.id === undefined || firstIdentity.id === null
            ? null
            : String(firstIdentity.id),
        second_index: b.featureIndex,
        second_id:
          secondIdentity.id === undefined || secondIdentity.id === null
            ? null
            : String(secondIdentity.id),
        kind: interior ? 'crossing' : 'touch',
        geometry: makeCatalogGeometryText('Point', [hit.x, hit.y])
      });
    }
  }
  addCatalogDiagnostic(
    diagnostics,
    'collinear-overlaps-ignored',
    'info',
    'Parallel segments, including collinear overlaps, produce no intersection point; use shared paths for overlaps.'
  );
  return {
    tables: [table],
    diagnostics,
    summary: {intersectionCount: table.rowCount, segmentCount: segments.length}
  };
}

// ---------------------------------------------------------------------------------------------
// Shared paths
// ---------------------------------------------------------------------------------------------

export const MAXIMUM_SHARED_PATHS = 50000;

export function runSharedPaths(
  left: CatalogFeatureCollection,
  right: CatalogFeatureCollection,
  parameters: {
    tolerance: number;
    angleToleranceDegrees: number;
    coordinateSystem: CatalogCoordinateSystem;
  }
): CatalogRunOutput {
  const frame = makeLocalFrame(parameters.coordinateSystem, [
    ...collectPositions(left),
    ...collectPositions(right)
  ]);
  const sine = Math.sin(toRadians(parameters.angleToleranceDegrees));
  type Segment = {
    featureIndex: number;
    start: PlanarPoint;
    end: PlanarPoint;
    startRaw: CatalogPosition;
    endRaw: CatalogPosition;
  };
  const collect = (collection: CatalogFeatureCollection): Segment[] => {
    const result: Segment[] = [];
    collection.features.forEach((feature, featureIndex) => {
      for (const part of getLineParts(feature.geometry))
        for (let index = 1; index < part.length; index++)
          if (!isSamePosition(part[index - 1], part[index]))
            result.push({
              featureIndex,
              start: frame.forward(part[index - 1]),
              end: frame.forward(part[index]),
              startRaw: part[index - 1],
              endRaw: part[index]
            });
    });
    return result;
  };
  const leftSegments = collect(left);
  const rightSegments = collect(right);
  const table = makeCatalogTable(
    'shared-paths',
    'Shared paths',
    {
      shared_id: 'string',
      left_index: 'int32',
      left_id: 'string',
      right_index: 'int32',
      right_id: 'string',
      direction: 'string',
      length: 'float64',
      geometry: 'string'
    },
    'geometry'
  );
  const tolerance = parameters.tolerance;
  for (const a of leftSegments) {
    const directionX = a.end[0] - a.start[0];
    const directionY = a.end[1] - a.start[1];
    const length = Math.hypot(directionX, directionY);
    for (const b of rightSegments) {
      if (
        Math.max(b.start[0], b.end[0]) < Math.min(a.start[0], a.end[0]) - tolerance ||
        Math.min(b.start[0], b.end[0]) > Math.max(a.start[0], a.end[0]) + tolerance ||
        Math.max(b.start[1], b.end[1]) < Math.min(a.start[1], a.end[1]) - tolerance ||
        Math.min(b.start[1], b.end[1]) > Math.max(a.start[1], a.end[1]) + tolerance
      )
        continue;
      const otherX = b.end[0] - b.start[0];
      const otherY = b.end[1] - b.start[1];
      const otherLength = Math.hypot(otherX, otherY);
      const crossValue = (directionX * otherY - directionY * otherX) / (length * otherLength);
      if (Math.abs(crossValue) > sine) continue;
      const offsetOf = (point: PlanarPoint) =>
        Math.abs(directionX * (point[1] - a.start[1]) - directionY * (point[0] - a.start[0])) /
        length;
      if (offsetOf(b.start) > tolerance || offsetOf(b.end) > tolerance) continue;
      const alongOf = (point: PlanarPoint) =>
        ((point[0] - a.start[0]) * directionX + (point[1] - a.start[1]) * directionY) / length;
      const first = Math.max(0, Math.min(alongOf(b.start), alongOf(b.end)));
      const last = Math.min(length, Math.max(alongOf(b.start), alongOf(b.end)));
      if (last - first <= Math.max(tolerance * 0.5, 1e-12)) continue;
      if (table.rowCount >= MAXIMUM_SHARED_PATHS)
        throw new KeplerAllocationError('capacity', {message: 'Shared path capacity exceeded'});
      const toRaw = (distance: number): CatalogPosition => [
        a.startRaw[0] + ((a.endRaw[0] - a.startRaw[0]) * distance) / length,
        a.startRaw[1] + ((a.endRaw[1] - a.startRaw[1]) * distance) / length
      ];
      const coordinates = [toRaw(first), toRaw(last)];
      const leftFeature = left.features[a.featureIndex];
      const rightFeature = right.features[b.featureIndex];
      appendCatalogRow(table, {
        shared_id: `${table.rowCount}`,
        left_index: a.featureIndex,
        left_id:
          leftFeature.id === undefined || leftFeature.id === null ? null : String(leftFeature.id),
        right_index: b.featureIndex,
        right_id:
          rightFeature.id === undefined || rightFeature.id === null
            ? null
            : String(rightFeature.id),
        direction: directionX * otherX + directionY * otherY >= 0 ? 'same' : 'opposite',
        length: last - first,
        geometry: makeCatalogGeometryText('LineString', coordinates)
      });
    }
  }
  return {
    tables: [table],
    diagnostics: [
      {
        code: 'segment-pairs',
        severity: 'info',
        count: 1,
        message:
          'Shared paths are collinear overlaps between segment pairs within the tolerance; adjacent overlapping segments are reported as separate rows.'
      }
    ],
    summary: {
      sharedCount: table.rowCount,
      leftSegments: leftSegments.length,
      rightSegments: rightSegments.length
    }
  };
}

// ---------------------------------------------------------------------------------------------
// Great-circle arcs
// ---------------------------------------------------------------------------------------------

export function runGreatCircleArcs(
  origins: CatalogFeatureCollection,
  destinations: CatalogFeatureCollection,
  segmentCount: number
): CatalogRunOutput {
  const diagnostics: CatalogDiagnostic[] = [];
  const table = makeCatalogTable(
    'great-circle-arcs',
    'Great-circle arcs',
    {
      arc_id: 'string',
      origin_index: 'int32',
      origin_id: 'string',
      destination_index: 'int32',
      destination_id: 'string',
      distance: 'float64',
      geometry: 'string'
    },
    'geometry'
  );
  const pairCount = Math.min(origins.features.length, destinations.features.length);
  if (origins.features.length !== destinations.features.length)
    addCatalogDiagnostic(
      diagnostics,
      'unpaired-features',
      'warning',
      'Origins and destinations pair by feature order; surplus features are ignored.',
      Math.abs(origins.features.length - destinations.features.length)
    );
  const toVector = (position: CatalogPosition): [number, number, number] => [
    Math.cos(toRadians(position[1])) * Math.cos(toRadians(position[0])),
    Math.cos(toRadians(position[1])) * Math.sin(toRadians(position[0])),
    Math.sin(toRadians(position[1]))
  ];
  for (let pairIndex = 0; pairIndex < pairCount; pairIndex++) {
    const origin = getPointPositions(origins.features[pairIndex].geometry)[0];
    const destination = getPointPositions(destinations.features[pairIndex].geometry)[0];
    if (!origin || !destination) {
      addCatalogDiagnostic(
        diagnostics,
        'pair-skipped',
        'warning',
        'Pairs without two points are skipped.'
      );
      continue;
    }
    const first = toVector(origin);
    const second = toVector(destination);
    const dot = clamp(vec3.dot(first, second), -1, 1);
    const angle = Math.acos(dot);
    if (angle < 1e-9 || Math.PI - angle < 1e-9) {
      addCatalogDiagnostic(
        diagnostics,
        'degenerate-arc',
        'warning',
        'Coincident or antipodal pairs have no unique great circle and are skipped.'
      );
      continue;
    }
    const coordinates: CatalogPosition[] = [];
    let previousLongitude = origin[0];
    for (let step = 0; step <= segmentCount; step++) {
      const fraction = step / segmentCount;
      const weightFirst = Math.sin((1 - fraction) * angle) / Math.sin(angle);
      const weightSecond = Math.sin(fraction * angle) / Math.sin(angle);
      const x = weightFirst * first[0] + weightSecond * second[0];
      const y = weightFirst * first[1] + weightSecond * second[1];
      const z = weightFirst * first[2] + weightSecond * second[2];
      let longitude = toDegrees(Math.atan2(y, x));
      while (longitude - previousLongitude > 180) longitude -= 360;
      while (longitude - previousLongitude < -180) longitude += 360;
      previousLongitude = longitude;
      coordinates.push(
        step === 0
          ? [origin[0], origin[1]]
          : step === segmentCount
            ? [longitude, destination[1]]
            : [longitude, toDegrees(Math.atan2(z, Math.hypot(x, y)))]
      );
    }
    if (coordinates.some(position => Math.abs(position[0]) > 180))
      addCatalogDiagnostic(
        diagnostics,
        'antimeridian-unwrapped',
        'info',
        'Arcs that cross the antimeridian keep continuous longitudes beyond +-180 instead of being split.'
      );
    appendCatalogRow(table, {
      arc_id: `${pairIndex}`,
      origin_index: pairIndex,
      origin_id:
        origins.features[pairIndex].id === undefined
          ? null
          : String(origins.features[pairIndex].id),
      destination_index: pairIndex,
      destination_id:
        destinations.features[pairIndex].id === undefined
          ? null
          : String(destinations.features[pairIndex].id),
      distance: angle * EARTH_RADIUS_METERS,
      geometry: makeCatalogGeometryText('LineString', coordinates)
    });
  }
  return {tables: [table], diagnostics, summary: {arcCount: table.rowCount}};
}

// ---------------------------------------------------------------------------------------------
// Offset curve
// ---------------------------------------------------------------------------------------------

export function offsetPolyline(
  points: readonly PlanarPoint[],
  distance: number,
  mitreLimit: number
): PlanarPoint[] {
  const normals: PlanarPoint[] = [];
  for (let index = 1; index < points.length; index++) {
    const deltaX = points[index][0] - points[index - 1][0];
    const deltaY = points[index][1] - points[index - 1][1];
    const length = Math.hypot(deltaX, deltaY);
    normals.push([-deltaY / length, deltaX / length]);
  }
  const result: PlanarPoint[] = [
    [points[0][0] + normals[0][0] * distance, points[0][1] + normals[0][1] * distance]
  ];
  for (let index = 1; index < points.length - 1; index++) {
    const previousNormal = normals[index - 1];
    const nextNormal = normals[index];
    const sum: PlanarPoint = [previousNormal[0] + nextNormal[0], previousNormal[1] + nextNormal[1]];
    const sumLengthSquared = sum[0] * sum[0] + sum[1] * sum[1];
    const vertex = points[index];
    // Mitre length relative to the offset: 2 / |n1 + n2|.
    if (sumLengthSquared > 1e-12 && 2 / Math.sqrt(sumLengthSquared) <= mitreLimit) {
      const scale = (2 * distance) / sumLengthSquared;
      result.push([vertex[0] + sum[0] * scale, vertex[1] + sum[1] * scale]);
    } else {
      result.push(
        [vertex[0] + previousNormal[0] * distance, vertex[1] + previousNormal[1] * distance],
        [vertex[0] + nextNormal[0] * distance, vertex[1] + nextNormal[1] * distance]
      );
    }
  }
  const lastNormal = normals[normals.length - 1];
  const last = points[points.length - 1];
  result.push([last[0] + lastNormal[0] * distance, last[1] + lastNormal[1] * distance]);
  return result;
}

export function runOffsetCurve(
  lines: CatalogFeatureCollection,
  parameters: {distance: number; mitreLimit: number; coordinateSystem: CatalogCoordinateSystem}
): CatalogRunOutput {
  const frame = makeLocalFrame(parameters.coordinateSystem, collectPositions(lines));
  const table = makeCatalogTable(
    'offset-curves',
    'Offset curves',
    {
      offset_id: 'string',
      source_index: 'int32',
      source_id: 'string',
      part_index: 'int32',
      offset_distance: 'float64',
      length: 'float64',
      geometry: 'string'
    },
    'geometry'
  );
  const diagnostics: CatalogDiagnostic[] = [];
  lines.features.forEach((feature, sourceIndex) => {
    getLineParts(feature.geometry).forEach((part, partIndex) => {
      const projected = part
        .filter((position, index) => index === 0 || !isSamePosition(position, part[index - 1]))
        .map(position => frame.forward(position));
      if (projected.length < 2) {
        addCatalogDiagnostic(
          diagnostics,
          'degenerate-line',
          'warning',
          'Lines with fewer than two distinct vertices are skipped.'
        );
        return;
      }
      const offset = offsetPolyline(projected, parameters.distance, parameters.mitreLimit);
      const coordinates = offset.map(point => frame.inverse(point[0], point[1]));
      appendCatalogRow(table, {
        offset_id: `${sourceIndex}:${partIndex}`,
        ...getSourceColumns(feature, sourceIndex),
        part_index: partIndex,
        offset_distance: parameters.distance,
        length: measurePathLength(coordinates, parameters.coordinateSystem),
        geometry: makeCatalogGeometryText('LineString', coordinates)
      });
    });
  });
  addCatalogDiagnostic(
    diagnostics,
    'inner-side-loops',
    'info',
    'Offsets are not cleaned: concave corners on the inner side can produce small loops, and curves are not trimmed against each other.'
  );
  return {tables: [table], diagnostics, summary: {offsetCount: table.rowCount}};
}

// ---------------------------------------------------------------------------------------------
// Line loop removal
// ---------------------------------------------------------------------------------------------

export function removeLineLoops(positions: readonly CatalogPosition[]): {
  coordinates: CatalogPosition[];
  loopCount: number;
} {
  let coordinates = [...positions];
  let loopCount = 0;
  const closed =
    coordinates.length > 3 && isSamePosition(coordinates[0], coordinates[coordinates.length - 1]);
  for (let iteration = 0; iteration < 1000; iteration++) {
    let found: {first: number; second: number; x: number; y: number} | null = null;
    for (let first = 0; first < coordinates.length - 1 && !found; first++)
      for (let second = first + 2; second < coordinates.length - 1; second++) {
        if (closed && first === 0 && second === coordinates.length - 2) continue;
        const hit = intersectSegments(
          [toPlanar(coordinates[first]), toPlanar(coordinates[first + 1])],
          [toPlanar(coordinates[second]), toPlanar(coordinates[second + 1])]
        );
        if (hit) {
          found = {first, second, x: hit.x, y: hit.y};
          break;
        }
      }
    if (!found) break;
    const hit = found;
    const crossing: CatalogPosition = [hit.x, hit.y];
    coordinates = [
      ...coordinates.slice(0, hit.first + 1),
      crossing,
      ...coordinates.slice(hit.second + 1)
    ].filter((position, index, all) => index === 0 || !isSamePosition(position, all[index - 1]));
    loopCount++;
  }
  return {coordinates, loopCount};
}

export function runLineLoopRemoval(
  lines: CatalogFeatureCollection,
  coordinateSystem: CatalogCoordinateSystem
): CatalogRunOutput {
  const table = makeCatalogTable(
    'loop-free-lines',
    'Loop-free lines',
    {
      line_id: 'string',
      source_index: 'int32',
      source_id: 'string',
      part_index: 'int32',
      removed_loops: 'int32',
      removed_vertices: 'int32',
      length: 'float64',
      geometry: 'string'
    },
    'geometry'
  );
  lines.features.forEach((feature, sourceIndex) => {
    getLineParts(feature.geometry).forEach((part, partIndex) => {
      if (part.length < 2) return;
      const {coordinates, loopCount} = removeLineLoops(part);
      appendCatalogRow(table, {
        line_id: `${sourceIndex}:${partIndex}`,
        ...getSourceColumns(feature, sourceIndex),
        part_index: partIndex,
        removed_loops: loopCount,
        removed_vertices: part.length - coordinates.length,
        length: measurePathLength(coordinates, coordinateSystem),
        geometry: makeCatalogGeometryText('LineString', coordinates)
      });
    });
  });
  return {
    tables: [table],
    diagnostics: [
      {
        code: 'planar-loops',
        severity: 'info',
        count: 1,
        message:
          'Loops are cut at the first self-intersection, repeatedly, on raw planar coordinates; the crossing point replaces the loop vertices.'
      }
    ],
    summary: {lineCount: table.rowCount}
  };
}

// ---------------------------------------------------------------------------------------------
// Geometry cleanup and polygon precision reduction
// ---------------------------------------------------------------------------------------------

function mapGeometryPositions(
  geometry: CatalogGeometry,
  transform: (position: CatalogPosition) => CatalogPosition
): CatalogGeometry {
  if (!geometry) return geometry;
  const visit = (value: unknown): unknown =>
    Array.isArray(value) && typeof value[0] === 'number'
      ? transform(value as unknown as CatalogPosition)
      : Array.isArray(value)
        ? value.map(visit)
        : value;
  return {type: geometry.type, coordinates: visit(geometry.coordinates)};
}

type RingRebuilder = (ring: CatalogPosition[], isPolygon: boolean) => CatalogPosition[] | null;

function rebuildFeatureGeometry(
  feature: CatalogFeature,
  rebuild: RingRebuilder
): {geometry: CatalogGeometry; removedVertices: number; removedRings: number} | null {
  const geometry = feature.geometry;
  if (!geometry) return null;
  let removedVertices = 0;
  let removedRings = 0;
  const cleanLine = (part: CatalogPosition[]): CatalogPosition[] | null => {
    const cleaned = rebuild(part, false);
    if (!cleaned) {
      removedRings++;
      removedVertices += part.length;
      return null;
    }
    removedVertices += part.length - cleaned.length;
    return cleaned;
  };
  const cleanPolygon = (polygon: CatalogPosition[][]): CatalogPosition[][] | null => {
    const rings: CatalogPosition[][] = [];
    polygon.forEach((ring, ringIndex) => {
      const cleaned = rebuild(ring, true);
      if (!cleaned) {
        removedRings++;
        removedVertices += ring.length;
        if (ringIndex === 0) rings.length = 0;
        return;
      }
      if (ringIndex > 0 && !rings.length) return;
      removedVertices += ring.length - cleaned.length;
      rings.push(cleaned);
    });
    return rings.length ? rings : null;
  };
  if (geometry.type === 'LineString') {
    const part = getLineParts(geometry)[0];
    const cleaned = cleanLine(part);
    return {
      geometry: cleaned ? {type: 'LineString', coordinates: cleaned} : null,
      removedVertices,
      removedRings
    };
  }
  if (geometry.type === 'MultiLineString') {
    const parts = getLineParts(geometry)
      .map(cleanLine)
      .filter((part): part is CatalogPosition[] => part !== null);
    return {
      geometry: parts.length ? {type: 'MultiLineString', coordinates: parts} : null,
      removedVertices,
      removedRings
    };
  }
  if (geometry.type === 'Polygon') {
    const cleaned = cleanPolygon(getPolygonParts(geometry)[0]);
    return {
      geometry: cleaned ? {type: 'Polygon', coordinates: cleaned} : null,
      removedVertices,
      removedRings
    };
  }
  if (geometry.type === 'MultiPolygon') {
    const parts = getPolygonParts(geometry)
      .map(cleanPolygon)
      .filter((part): part is CatalogPosition[][] => part !== null);
    return {
      geometry: parts.length ? {type: 'MultiPolygon', coordinates: parts} : null,
      removedVertices,
      removedRings
    };
  }
  return {geometry, removedVertices: 0, removedRings: 0};
}

function makeRebuildTable(id: string, label: string) {
  return makeCatalogTable(
    id,
    label,
    {
      source_index: 'int32',
      source_id: 'string',
      status: 'string',
      removed_vertices: 'int32',
      removed_rings: 'int32',
      geometry: 'string'
    },
    'geometry'
  );
}

function runRebuild(
  features: CatalogFeatureCollection,
  tableId: string,
  label: string,
  rebuild: RingRebuilder
): CatalogRunOutput {
  const table = makeRebuildTable(tableId, label);
  const diagnostics: CatalogDiagnostic[] = [];
  features.features.forEach((feature, sourceIndex) => {
    const result = rebuildFeatureGeometry(feature, rebuild);
    if (!result || !result.geometry) {
      appendCatalogRow(table, {
        ...getSourceColumns(feature, sourceIndex),
        status: 'dropped',
        removed_vertices: result?.removedVertices ?? 0,
        removed_rings: result?.removedRings ?? 0
      });
      addCatalogDiagnostic(
        diagnostics,
        'feature-dropped',
        'warning',
        'Features that collapse (no usable ring or line) keep a row with status dropped and no geometry.'
      );
      return;
    }
    appendCatalogRow(table, {
      ...getSourceColumns(feature, sourceIndex),
      status:
        JSON.stringify(result.geometry) === JSON.stringify(feature.geometry)
          ? 'unchanged'
          : 'cleaned',
      removed_vertices: result.removedVertices,
      removed_rings: result.removedRings,
      geometry: JSON.stringify(result.geometry)
    });
  });
  return {
    tables: [table],
    diagnostics,
    summary: {
      featureCount: table.rowCount,
      cleanedCount: table.columns['status'].filter(status => status === 'cleaned').length
    }
  };
}

export function runGeometryCleanup(
  features: CatalogFeatureCollection,
  parameters: {tolerance: number; removeSpikes: boolean; coordinateSystem: CatalogCoordinateSystem}
): CatalogRunOutput {
  const frame = makeLocalFrame(parameters.coordinateSystem, collectPositions(features));
  const rebuild: RingRebuilder = (ring, isPolygon) => {
    let positions = ring.filter(
      position => Number.isFinite(position[0]) && Number.isFinite(position[1])
    );
    if (
      isPolygon &&
      positions.length > 1 &&
      isSamePosition(positions[0], positions[positions.length - 1])
    )
      positions = positions.slice(0, -1);
    const near = (first: CatalogPosition, second: CatalogPosition) => {
      const [firstX, firstY] = frame.forward(first);
      const [secondX, secondY] = frame.forward(second);
      return Math.hypot(firstX - secondX, firstY - secondY) <= parameters.tolerance;
    };
    positions = positions.filter(
      (position, index) => index === 0 || !near(position, positions[index - 1])
    );
    if (isPolygon)
      while (positions.length > 1 && near(positions[0], positions[positions.length - 1]))
        positions.pop();
    if (parameters.removeSpikes) {
      let changed = true;
      while (changed && positions.length > 2) {
        changed = false;
        for (let index = 1; index < positions.length - 1; index++)
          if (near(positions[index - 1], positions[index + 1])) {
            positions.splice(index, 2);
            changed = true;
            break;
          }
      }
    }
    if (isPolygon) {
      if (positions.length < 3) return null;
      return closeRing(positions);
    }
    return positions.length >= 2 ? positions : null;
  };
  return runRebuild(features, 'cleaned-geometry', 'Cleaned geometry', rebuild);
}

export function runPolygonPrecisionReduction(
  polygons: CatalogFeatureCollection,
  gridSize: number
): CatalogRunOutput {
  const snap = (value: number) => Math.round(value / gridSize) * gridSize;
  const rebuild: RingRebuilder = ring => {
    const snapped = ring
      .map((position): CatalogPosition => [snap(position[0]), snap(position[1])])
      .filter((position, index, all) => index === 0 || !isSamePosition(position, all[index - 1]));
    const open =
      snapped.length > 1 && isSamePosition(snapped[0], snapped[snapped.length - 1])
        ? snapped.slice(0, -1)
        : snapped;
    const distinct = new Set(open.map(position => `${position[0]}:${position[1]}`));
    if (open.length < 3 || distinct.size < 3) return null;
    const planar = open.map(toPlanar);
    if (getSignedArea(planar) === 0) return null;
    return closeRing(open);
  };
  const output = runRebuild(polygons, 'reduced-polygons', 'Precision-reduced polygons', rebuild);
  output.diagnostics.push({
    code: 'grid-size-units',
    severity: 'info',
    count: 1,
    message:
      'The grid size is in coordinate units (degrees for EPSG:4326); rings that collapse below three distinct vertices or zero area are removed, and rings may self-intersect after snapping.'
  });
  return output;
}

// ---------------------------------------------------------------------------------------------
// Polygon outline
// ---------------------------------------------------------------------------------------------

export function runPolygonOutline(
  polygons: CatalogFeatureCollection,
  coordinateSystem: CatalogCoordinateSystem
): CatalogRunOutput {
  const table = makeCatalogTable(
    'polygon-outlines',
    'Polygon outlines',
    {
      outline_id: 'string',
      source_index: 'int32',
      source_id: 'string',
      part_index: 'int32',
      ring_index: 'int32',
      is_hole: 'boolean',
      length: 'float64',
      geometry: 'string'
    },
    'geometry'
  );
  polygons.features.forEach((feature, sourceIndex) => {
    getPolygonParts(feature.geometry).forEach((rings, partIndex) => {
      rings.forEach((ring, ringIndex) => {
        if (ring.length < 4) return;
        appendCatalogRow(table, {
          outline_id: `${sourceIndex}:${partIndex}:${ringIndex}`,
          ...getSourceColumns(feature, sourceIndex),
          part_index: partIndex,
          ring_index: ringIndex,
          is_hole: ringIndex > 0,
          length: measurePathLength(ring, coordinateSystem),
          geometry: makeCatalogGeometryText('LineString', ring)
        });
      });
    });
  });
  return {tables: [table], diagnostics: [], summary: {outlineCount: table.rowCount}};
}

// ---------------------------------------------------------------------------------------------
// Shape generator
// ---------------------------------------------------------------------------------------------

const SHAPE_SIDES: Record<string, number> = {square: 4, triangle: 3, pentagon: 5, hexagon: 6};

export function runShapeGenerator(
  points: CatalogFeatureCollection,
  parameters: {
    shape: string;
    radius: number;
    radiusProperty: string;
    rotationDegrees: number;
    circleSegments: number;
    coordinateSystem: CatalogCoordinateSystem;
  }
): CatalogRunOutput {
  const diagnostics: CatalogDiagnostic[] = [];
  const table = makeCatalogTable(
    'generated-shapes',
    'Generated shapes',
    {
      shape_id: 'string',
      source_index: 'int32',
      source_id: 'string',
      point_index: 'int32',
      radius: 'float64',
      area: 'float64',
      geometry: 'string'
    },
    'geometry'
  );
  points.features.forEach((feature, sourceIndex) => {
    const radius = parameters.radiusProperty
      ? readPropertyNumber(feature, parameters.radiusProperty)
      : parameters.radius;
    getPointPositions(feature.geometry).forEach((position, pointIndex) => {
      if (radius === null || !(radius > 0)) {
        addCatalogDiagnostic(
          diagnostics,
          'invalid-radius',
          'warning',
          'Points whose radius property is missing, non-numeric or not positive produce no shape.'
        );
        return;
      }
      const frame = makeLocalFrame(parameters.coordinateSystem, [position]);
      const [centerX, centerY] = frame.forward(position);
      const rotation = (parameters.rotationDegrees * Math.PI) / 180;
      let ring: PlanarPoint[];
      if (parameters.shape === 'circle')
        ring = makeCirclePositions(centerX, centerY, radius, parameters.circleSegments);
      else if (parameters.shape === 'star') {
        ring = [];
        for (let index = 0; index < 10; index++) {
          const angle = rotation + (Math.PI * index) / 5 + Math.PI / 2;
          const reach = index % 2 === 0 ? radius : radius * 0.4;
          ring.push([centerX + reach * Math.cos(angle), centerY + reach * Math.sin(angle)]);
        }
      } else {
        const sides = SHAPE_SIDES[parameters.shape];
        ring = [];
        for (let index = 0; index < sides; index++) {
          const angle = rotation + (2 * Math.PI * index) / sides + Math.PI / 2;
          ring.push([centerX + radius * Math.cos(angle), centerY + radius * Math.sin(angle)]);
        }
      }
      appendCatalogRow(table, {
        shape_id: `${sourceIndex}:${pointIndex}`,
        ...getSourceColumns(feature, sourceIndex),
        point_index: pointIndex,
        radius,
        area: Math.abs(getSignedArea(ring)),
        geometry: makeCatalogGeometryText('Polygon', [
          closeRing(ring.map(point => frame.inverse(point[0], point[1])))
        ])
      });
    });
  });
  return {tables: [table], diagnostics, summary: {shapeCount: table.rowCount}};
}

// ---------------------------------------------------------------------------------------------
// Dot density
// ---------------------------------------------------------------------------------------------

export function runDotDensity(
  polygons: CatalogFeatureCollection,
  parameters: {
    valueProperty: string;
    dotValue: number;
    maximumDots: number;
    seed: number;
    coordinateSystem: CatalogCoordinateSystem;
  }
): CatalogRunOutput {
  const diagnostics: CatalogDiagnostic[] = [];
  const frame = makeLocalFrame(parameters.coordinateSystem, collectPositions(polygons), {
    kind: 'lambert-equal-area'
  });
  const table = makeCatalogTable(
    'dots',
    'Dots',
    {
      dot_id: 'string',
      source_index: 'int32',
      source_id: 'string',
      dot_index: 'int32',
      geometry: 'string'
    },
    'geometry'
  );
  let total = 0;
  polygons.features.forEach((feature, sourceIndex) => {
    const value = readPropertyNumber(feature, parameters.valueProperty);
    if (value === null || value < 0) {
      addCatalogDiagnostic(
        diagnostics,
        'invalid-value',
        'warning',
        'Features whose value property is missing, non-numeric or negative get no dots.'
      );
      return;
    }
    const count = Math.floor(value / parameters.dotValue);
    total += count;
    // Dot capacity exceeded.
    if (total > parameters.maximumDots)
      throw new KeplerAllocationError('capacity', {message: 'Dot capacity exceeded'});
    const parts = projectPolygonParts(feature, frame)
      .map(rings => ({
        rings,
        area:
          Math.abs(getSignedArea(rings[0])) -
          rings.slice(1).reduce((sum, ring) => sum + Math.abs(getSignedArea(ring)), 0)
      }))
      .filter(part => part.area > 0);
    if (!parts.length || !count) return;
    const totalArea = parts.reduce((sum, part) => sum + part.area, 0);
    const random = makeSeededRandom(
      (parameters.seed ^ Math.imul(sourceIndex + 1, 0x9e3779b1)) >>> 0
    );
    for (let dotIndex = 0; dotIndex < count; dotIndex++) {
      let pick = random() * totalArea;
      let part = parts[parts.length - 1];
      for (const candidate of parts) {
        pick -= candidate.area;
        if (pick <= 0) {
          part = candidate;
          break;
        }
      }
      const xs = part.rings[0].map(point => point[0]);
      const ys = part.rings[0].map(point => point[1]);
      const minimumX = Math.min(...xs);
      const minimumY = Math.min(...ys);
      const width = Math.max(...xs) - minimumX;
      const height = Math.max(...ys) - minimumY;
      let placed: PlanarPoint | null = null;
      for (let attempt = 0; attempt < 1000 && !placed; attempt++) {
        const x = minimumX + random() * width;
        const y = minimumY + random() * height;
        if (isPointInPolygonRings(x, y, part.rings)) placed = [x, y];
      }
      if (!placed) {
        addCatalogDiagnostic(
          diagnostics,
          'placement-failed',
          'warning',
          'Dots that could not be placed after 1000 attempts are dropped.'
        );
        continue;
      }
      appendCatalogRow(table, {
        dot_id: `${sourceIndex}:${dotIndex}`,
        ...getSourceColumns(feature, sourceIndex),
        dot_index: dotIndex,
        geometry: makeCatalogGeometryText('Point', frame.inverse(placed[0], placed[1]))
      });
    }
  });
  return {tables: [table], diagnostics, summary: {dotCount: table.rowCount}};
}

// ---------------------------------------------------------------------------------------------
// Local-frame projection (azimuthal equidistant, shared makeLocalFrame)
// ---------------------------------------------------------------------------------------------

export function runLocalFrameProjection(
  features: CatalogFeatureCollection,
  parameters: {
    direction: 'forward' | 'inverse';
    originLongitude: number;
    originLatitude: number;
    useDataCenter: boolean;
  }
): CatalogRunOutput {
  const positions = collectPositions(features);
  const useDataCenter =
    parameters.direction === 'forward' && parameters.useDataCenter && positions.length > 0;
  // The data center is antimeridian aware; an explicit origin is used as given.
  const frame = makeLocalFrame('EPSG:4326', positions, {
    origin: useDataCenter ? undefined : [parameters.originLongitude, parameters.originLatitude]
  });
  const origin: CatalogPosition = [...(frame.gpuFrame?.origin ?? [0, 0])] as CatalogPosition;
  const diagnostics: CatalogDiagnostic[] = [];
  const table = makeCatalogTable(
    'local-frame-features',
    'Local-frame features',
    {
      source_index: 'int32',
      source_id: 'string',
      origin_longitude: 'float64',
      origin_latitude: 'float64',
      geometry: 'string'
    },
    'geometry'
  );
  features.features.forEach((feature, sourceIndex) => {
    const geometry = mapGeometryPositions(feature.geometry, position =>
      parameters.direction === 'forward'
        ? frame.forward(position)
        : frame.inverse(position[0], position[1])
    );
    appendCatalogRow(table, {
      ...getSourceColumns(feature, sourceIndex),
      origin_longitude: origin[0],
      origin_latitude: origin[1],
      geometry: geometry ? JSON.stringify(geometry) : null
    });
  });
  addLocalFrameDiagnostic(diagnostics, frame);
  diagnostics.unshift({
    code: 'local-frame',
    severity: 'info',
    count: 1,
    message:
      'Forward output is in meters east and north of the origin in a spherical azimuthal-equidistant frame (distances from the origin are exact); publish it as planar data. Inverse expects that frame.'
  });
  return {
    tables: [table],
    diagnostics,
    summary: {featureCount: table.rowCount, originLongitude: origin[0], originLatitude: origin[1]}
  };
}
