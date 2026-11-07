// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {toDegrees, toRadians} from '@math.gl/core';
import {KeplerAllocationError} from '../runtime/dataset-registry';
import {
  EARTH_RADIUS_METERS,
  addCatalogDiagnostic,
  appendCatalogRow,
  getFeatureIdentity,
  getLineParts,
  getPointPositions,
  getPolygonParts,
  makeCatalogGeometryText,
  makeCatalogTable,
  measureDistance,
  measurePathLength,
  measureRingSignedArea,
  type CatalogCoordinateSystem,
  type CatalogDiagnostic,
  type CatalogFeatureCollection,
  type CatalogOutputTable,
  type CatalogPosition
} from './catalog-geojson';
import {isPointInRing} from './catalog-platform3-a-shared';

export type CatalogGeometryResult = {
  tables: CatalogOutputTable[];
  diagnostics: CatalogDiagnostic[];
  summary: Record<string, string | number>;
};

const VALIDITY_VERTEX_LIMIT = 2000;

function orient(a: CatalogPosition, b: CatalogPosition, c: CatalogPosition): number {
  return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
}

function isOnSegment(a: CatalogPosition, b: CatalogPosition, point: CatalogPosition): boolean {
  return (
    point[0] >= Math.min(a[0], b[0]) &&
    point[0] <= Math.max(a[0], b[0]) &&
    point[1] >= Math.min(a[1], b[1]) &&
    point[1] <= Math.max(a[1], b[1])
  );
}

/** True when segments touch or cross (collinear overlap included). */
function doSegmentsTouch(
  a: CatalogPosition,
  b: CatalogPosition,
  c: CatalogPosition,
  d: CatalogPosition
): boolean {
  const o1 = orient(a, b, c);
  const o2 = orient(a, b, d);
  const o3 = orient(c, d, a);
  const o4 = orient(c, d, b);
  if (o1 * o2 < 0 && o3 * o4 < 0) return true;
  return (
    (o1 === 0 && isOnSegment(a, b, c)) ||
    (o2 === 0 && isOnSegment(a, b, d)) ||
    (o3 === 0 && isOnSegment(c, d, a)) ||
    (o4 === 0 && isOnSegment(c, d, b))
  );
}

function doSegmentsCross(
  a: CatalogPosition,
  b: CatalogPosition,
  c: CatalogPosition,
  d: CatalogPosition
): boolean {
  return orient(a, b, c) * orient(a, b, d) < 0 && orient(c, d, a) * orient(c, d, b) < 0;
}

/**
 * Detect-only validity: closure, length, area, finiteness, self-touch, ring crossings and hole
 * containment. Nothing is repaired; invalid features are reported, never altered or dropped.
 */
export function runGeometryValidity(polygons: CatalogFeatureCollection): CatalogGeometryResult {
  const diagnostics: CatalogDiagnostic[] = [];
  const table = makeCatalogTable(
    'validity',
    'Geometry validity',
    {
      source_index: 'int32',
      source_id: 'string',
      status: 'string',
      error_count: 'int32',
      warning_count: 'int32',
      issues: 'string',
      geometry: 'string'
    },
    'geometry'
  );
  let invalidCount = 0;
  let skippedCount = 0;
  polygons.features.forEach((feature, sourceIndex) => {
    const parts = getPolygonParts(feature.geometry);
    const errors = new Set<string>();
    const warnings = new Set<string>();
    let skipped = false;
    if (!parts.length) errors.add('not-polygon');
    for (const rings of parts) {
      if (rings.reduce((count, ring) => count + ring.length, 0) > VALIDITY_VERTEX_LIMIT) {
        skipped = true;
        continue;
      }
      rings.forEach((ring, ringIndex) => {
        if (ring.some(position => !Number.isFinite(position[0]) || !Number.isFinite(position[1])))
          errors.add('non-finite-coordinate');
        if (ring.length < 4) errors.add('ring-too-short');
        const first = ring[0];
        const last = ring[ring.length - 1];
        if (first && last && (first[0] !== last[0] || first[1] !== last[1]))
          errors.add('ring-not-closed');
        if (ring.length >= 4 && Math.abs(measureRingSignedArea(ring, 'planar')) === 0)
          errors.add('zero-area-ring');
        for (let index = 1; index < ring.length; index++)
          if (ring[index][0] === ring[index - 1][0] && ring[index][1] === ring[index - 1][1])
            warnings.add('duplicate-consecutive-vertex');
        if (ring.length >= 4 && ringIndex === 0 && measureRingSignedArea(ring, 'planar') < 0)
          warnings.add('shell-not-counterclockwise');
        const segmentCount = ring.length - 1;
        for (let a = 0; a < segmentCount; a++)
          for (let b = a + 2; b < segmentCount; b++) {
            if (a === 0 && b === segmentCount - 1) continue;
            if (doSegmentsTouch(ring[a], ring[a + 1], ring[b], ring[b + 1]))
              errors.add('self-intersection');
          }
      });
      for (let left = 0; left < rings.length; left++)
        for (let right = left + 1; right < rings.length; right++)
          for (let a = 0; a < rings[left].length - 1; a++)
            for (let b = 0; b < rings[right].length - 1; b++)
              if (
                doSegmentsCross(
                  rings[left][a],
                  rings[left][a + 1],
                  rings[right][b],
                  rings[right][b + 1]
                )
              )
                errors.add('ring-crossing');
      for (let holeIndex = 1; holeIndex < rings.length; holeIndex++)
        if (
          rings[holeIndex].length &&
          !isPointInRing(rings[holeIndex][0][0], rings[holeIndex][0][1], rings[0]) &&
          !rings[0].some(
            position =>
              position[0] === rings[holeIndex][0][0] && position[1] === rings[holeIndex][0][1]
          )
        )
          errors.add('hole-outside-shell');
    }
    if (skipped) warnings.add('skipped-vertex-limit');
    if (skipped) skippedCount++;
    if (errors.size) invalidCount++;
    appendCatalogRow(table, {
      source_index: sourceIndex,
      source_id: getFeatureIdentity(feature),
      status: errors.size ? 'invalid' : skipped ? 'skipped' : 'valid',
      error_count: errors.size,
      warning_count: warnings.size,
      issues: [...errors, ...warnings].join(','),
      geometry: feature.geometry ? JSON.stringify(feature.geometry) : null
    });
  });
  if (invalidCount)
    addCatalogDiagnostic(
      diagnostics,
      'invalid-geometry',
      'warning',
      'Features failed validity; none were repaired.',
      invalidCount
    );
  if (skippedCount)
    addCatalogDiagnostic(
      diagnostics,
      'skipped-vertex-limit',
      'warning',
      `Features over ${VALIDITY_VERTEX_LIMIT} vertices skip pairwise intersection checks.`,
      skippedCount
    );
  return {
    tables: [table],
    diagnostics,
    summary: {featureCount: polygons.features.length, invalidCount, skippedCount}
  };
}

function getCentroid(coordinates: readonly CatalogPosition[]): [number, number] {
  let x = 0;
  let y = 0;
  for (const position of coordinates) {
    x += position[0];
    y += position[1];
  }
  const count = Math.max(1, coordinates.length);
  return [x / count, y / count];
}

/** Area/length/perimeter with explicit CRS: EPSG:4326 gives m, m2; planar gives coordinate units. */
export function runGeometryMeasures(
  features: CatalogFeatureCollection,
  coordinateSystem: CatalogCoordinateSystem
): CatalogGeometryResult {
  const diagnostics: CatalogDiagnostic[] = [];
  const table = makeCatalogTable(
    'measures',
    'Geometry measures',
    {
      source_index: 'int32',
      source_id: 'string',
      geometry_type: 'string',
      area: 'float64',
      length: 'float64',
      vertex_count: 'int32',
      minimum_x: 'float64',
      minimum_y: 'float64',
      maximum_x: 'float64',
      maximum_y: 'float64',
      centroid_x: 'float64',
      centroid_y: 'float64',
      geometry: 'string'
    },
    'geometry'
  );
  let emptyCount = 0;
  features.features.forEach((feature, sourceIndex) => {
    const polygons = getPolygonParts(feature.geometry);
    const lines = getLineParts(feature.geometry);
    const points = getPointPositions(feature.geometry);
    const all: CatalogPosition[] = [];
    let area = 0;
    let length = 0;
    for (const rings of polygons)
      rings.forEach((ring, ringIndex) => {
        const ringArea = Math.abs(measureRingSignedArea(ring, coordinateSystem));
        area += ringIndex === 0 ? ringArea : -ringArea;
        length += measurePathLength(ring, coordinateSystem);
        all.push(...ring);
      });
    for (const line of lines) {
      length += measurePathLength(line, coordinateSystem);
      all.push(...line);
    }
    all.push(...points);
    if (!all.length) emptyCount++;
    const centroid = all.length ? getCentroid(all) : [null, null];
    appendCatalogRow(table, {
      source_index: sourceIndex,
      source_id: getFeatureIdentity(feature),
      geometry_type: feature.geometry?.type ?? null,
      area: polygons.length ? Math.max(0, area) : null,
      length: polygons.length || lines.length ? length : null,
      vertex_count: all.length,
      minimum_x: all.length ? Math.min(...all.map(position => position[0])) : null,
      minimum_y: all.length ? Math.min(...all.map(position => position[1])) : null,
      maximum_x: all.length ? Math.max(...all.map(position => position[0])) : null,
      maximum_y: all.length ? Math.max(...all.map(position => position[1])) : null,
      centroid_x: centroid[0],
      centroid_y: centroid[1],
      geometry: feature.geometry ? JSON.stringify(feature.geometry) : null
    });
  });
  if (emptyCount)
    addCatalogDiagnostic(
      diagnostics,
      'empty-geometry',
      'warning',
      'Features without coordinates have null measures.',
      emptyCount
    );
  addCatalogDiagnostic(
    diagnostics,
    'centroid-definition',
    'info',
    'Centroid is the planar vertex average, not an area-weighted or geodesic centroid.'
  );
  return {
    tables: [table],
    diagnostics,
    summary: {
      featureCount: features.features.length,
      lengthUnit: coordinateSystem === 'planar' ? 'coordinate-units' : 'meters',
      areaUnit: coordinateSystem === 'planar' ? 'coordinate-units^2' : 'square-meters'
    }
  };
}

/** Circle polygon around each point. EPSG:4326 radius is meters (spherical destination formula). */
export function runPointBuffer(
  points: CatalogFeatureCollection,
  parameters: {
    radius: number;
    segmentsPerQuarter: number;
    coordinateSystem: CatalogCoordinateSystem;
  }
): CatalogGeometryResult {
  const diagnostics: CatalogDiagnostic[] = [];
  const table = makeCatalogTable(
    'buffers',
    'Point buffers',
    {
      buffer_id: 'string',
      source_index: 'int32',
      source_id: 'string',
      point_index: 'int32',
      radius: 'float64',
      area: 'float64',
      geometry: 'string'
    },
    'geometry'
  );
  const vertexCount = parameters.segmentsPerQuarter * 4;
  let droppedCount = 0;
  let antimeridianCount = 0;
  points.features.forEach((feature, sourceIndex) => {
    const positions = getPointPositions(feature.geometry);
    if (!positions.length) droppedCount++;
    positions.forEach((center, pointIndex) => {
      const ring: CatalogPosition[] = [];
      let crossesAntimeridian = false;
      let reachesPole = false;
      for (let vertex = 0; vertex < vertexCount; vertex++) {
        const bearing = (2 * Math.PI * vertex) / vertexCount;
        if (parameters.coordinateSystem === 'planar') {
          ring.push([
            center[0] + parameters.radius * Math.sin(bearing),
            center[1] + parameters.radius * Math.cos(bearing)
          ]);
          continue;
        }
        const delta = parameters.radius / EARTH_RADIUS_METERS;
        const latitude = toRadians(center[1]);
        const nextLatitude = Math.asin(
          Math.sin(latitude) * Math.cos(delta) +
            Math.cos(latitude) * Math.sin(delta) * Math.cos(bearing)
        );
        const nextLongitude =
          toRadians(center[0]) +
          Math.atan2(
            Math.sin(bearing) * Math.sin(delta) * Math.cos(latitude),
            Math.cos(delta) - Math.sin(latitude) * Math.sin(nextLatitude)
          );
        if (Math.abs(toDegrees(nextLatitude)) > 89.999) reachesPole = true;
        const longitude = toDegrees(nextLongitude);
        if (Math.abs(longitude) > 180) crossesAntimeridian = true;
        ring.push([longitude, toDegrees(nextLatitude)]);
      }
      if (reachesPole || delta180(parameters)) {
        droppedCount++;
        return;
      }
      if (crossesAntimeridian) antimeridianCount++;
      ring.push(ring[0]);
      const bufferId = `${sourceIndex}:${pointIndex}`;
      appendCatalogRow(table, {
        buffer_id: bufferId,
        source_index: sourceIndex,
        source_id: getFeatureIdentity(feature),
        point_index: pointIndex,
        radius: parameters.radius,
        area: Math.abs(measureRingSignedArea(ring, parameters.coordinateSystem)),
        geometry: makeCatalogGeometryText('Polygon', [ring])
      });
    });
  });
  if (droppedCount)
    addCatalogDiagnostic(
      diagnostics,
      'feature-dropped',
      'warning',
      'Non-point features or buffers reaching a pole produce no output.',
      droppedCount
    );
  if (antimeridianCount)
    addCatalogDiagnostic(
      diagnostics,
      'antimeridian-unwrapped',
      'warning',
      'Buffer longitudes exceed +-180 and are not split.',
      antimeridianCount
    );
  addCatalogDiagnostic(
    diagnostics,
    'inscribed-polygon',
    'info',
    'The circle is approximated by an inscribed polygon, slightly smaller than the true radius.'
  );
  return {
    tables: [table],
    diagnostics,
    summary: {outputFeatureCount: table.rowCount, verticesPerBuffer: vertexCount + 1}
  };
}

function delta180(parameters: {
  radius: number;
  coordinateSystem: CatalogCoordinateSystem;
}): boolean {
  return (
    parameters.coordinateSystem === 'EPSG:4326' &&
    parameters.radius / EARTH_RADIUS_METERS >= Math.PI / 2
  );
}

/** Inserts evenly spaced vertices so no segment exceeds the maximum. Source features keep identity. */
export function runLineSegmentize(
  lines: CatalogFeatureCollection,
  parameters: {maximumSegmentLength: number; coordinateSystem: CatalogCoordinateSystem},
  maximumOutputVertices: number
): CatalogGeometryResult {
  const diagnostics: CatalogDiagnostic[] = [];
  const table = makeCatalogTable(
    'segmentized',
    'Segmentized lines',
    {
      output_id: 'string',
      source_index: 'int32',
      source_id: 'string',
      added_vertex_count: 'int32',
      length: 'float64',
      geometry: 'string'
    },
    'geometry'
  );
  let totalVertices = 0;
  let droppedCount = 0;
  lines.features.forEach((feature, sourceIndex) => {
    const parts = getLineParts(feature.geometry);
    if (!parts.length) {
      droppedCount++;
      return;
    }
    let added = 0;
    const output = parts.map(part => {
      const densified: CatalogPosition[] = part.length ? [part[0]] : [];
      for (let index = 1; index < part.length; index++) {
        const segmentLength = measureDistance(
          part[index - 1],
          part[index],
          parameters.coordinateSystem
        );
        const pieces = Math.max(1, Math.ceil(segmentLength / parameters.maximumSegmentLength));
        totalVertices += pieces;
        // Output vertex capacity exceeded; raise the segment length or reduce the input.
        if (totalVertices > maximumOutputVertices)
          throw new KeplerAllocationError('capacity', {message: 'Output capacity exceeded'});
        for (let piece = 1; piece < pieces; piece++) {
          const fraction = piece / pieces;
          densified.push([
            part[index - 1][0] + (part[index][0] - part[index - 1][0]) * fraction,
            part[index - 1][1] + (part[index][1] - part[index - 1][1]) * fraction
          ]);
          added++;
        }
        densified.push(part[index]);
      }
      return densified;
    });
    appendCatalogRow(table, {
      output_id: String(sourceIndex),
      source_index: sourceIndex,
      source_id: getFeatureIdentity(feature),
      added_vertex_count: added,
      length: output.reduce(
        (sum, part) => sum + measurePathLength(part, parameters.coordinateSystem),
        0
      ),
      geometry:
        output.length === 1
          ? makeCatalogGeometryText('LineString', output[0])
          : makeCatalogGeometryText('MultiLineString', output)
    });
  });
  if (droppedCount)
    addCatalogDiagnostic(
      diagnostics,
      'feature-dropped',
      'warning',
      'Non-line features produce no output.',
      droppedCount
    );
  addCatalogDiagnostic(
    diagnostics,
    'linear-interpolation',
    'info',
    'New vertices are linear in coordinate space, not along a great circle.'
  );
  return {
    tables: [table],
    diagnostics,
    summary: {outputVertexCount: totalVertices + 0, outputFeatureCount: table.rowCount}
  };
}

function isRingConvex(ring: readonly CatalogPosition[]): boolean {
  let sign = 0;
  const count = ring.length - 1;
  for (let index = 0; index < count; index++) {
    const turn = orient(ring[index], ring[(index + 1) % count], ring[(index + 2) % count]);
    if (turn !== 0) {
      if (sign !== 0 && Math.sign(turn) !== sign) return false;
      sign = Math.sign(turn);
    }
  }
  return true;
}

function clipRingByConvex(
  ring: readonly CatalogPosition[],
  clip: readonly CatalogPosition[]
): CatalogPosition[] {
  let output = ring.slice(0, -1);
  for (let index = 0; index < clip.length && output.length; index++) {
    const a = clip[index];
    const b = clip[(index + 1) % clip.length];
    const input = output;
    output = [];
    for (let vertex = 0; vertex < input.length; vertex++) {
      const current = input[vertex];
      const previous = input[(vertex + input.length - 1) % input.length];
      const currentInside = orient(a, b, current) >= 0;
      const previousInside = orient(a, b, previous) >= 0;
      if (currentInside !== previousInside) {
        const denominator =
          (previous[0] - current[0]) * (a[1] - b[1]) - (previous[1] - current[1]) * (a[0] - b[0]);
        const t =
          ((previous[0] - a[0]) * (a[1] - b[1]) - (previous[1] - a[1]) * (a[0] - b[0])) /
          denominator;
        output.push([
          previous[0] + (current[0] - previous[0]) * t,
          previous[1] + (current[1] - previous[1]) * t
        ]);
      }
      if (currentInside) output.push(current);
    }
  }
  return output.length >= 3 ? [...output, output[0]] : [];
}

/** Cyrus-Beck: the portion of a segment inside a CCW convex polygon, or null. */
function clipSegmentByConvex(
  start: CatalogPosition,
  end: CatalogPosition,
  clip: readonly CatalogPosition[]
): [CatalogPosition, CatalogPosition] | null {
  let enter = 0;
  let leave = 1;
  const dx = end[0] - start[0];
  const dy = end[1] - start[1];
  for (let index = 0; index < clip.length; index++) {
    const a = clip[index];
    const b = clip[(index + 1) % clip.length];
    const numerator = orient(a, b, start);
    const denominator = (b[0] - a[0]) * dy - (b[1] - a[1]) * dx;
    if (denominator === 0) {
      if (numerator < 0) return null;
      continue;
    }
    const t = -numerator / denominator;
    if (denominator > 0) enter = Math.max(enter, t);
    else leave = Math.min(leave, t);
    if (enter > leave) return null;
  }
  return [
    [start[0] + dx * enter, start[1] + dy * enter],
    [start[0] + dx * leave, start[1] + dy * leave]
  ];
}

/**
 * Overlay by clipping polygons and lines against one convex clip polygon (intersection only).
 * Holes clip independently; a concave source shell may gain degenerate bridge edges and is flagged.
 */
export function runConvexClip(
  features: CatalogFeatureCollection,
  clipCollection: CatalogFeatureCollection
): CatalogGeometryResult {
  const diagnostics: CatalogDiagnostic[] = [];
  const clipRing = getPolygonParts(clipCollection.features[0]?.geometry ?? null)[0]?.[0];
  // Clip input must be one polygon with a convex shell; holes in it are ignored.
  if (
    clipCollection.features.length !== 1 ||
    !clipRing ||
    clipRing.length < 4 ||
    !isRingConvex(clipRing)
  )
    throw new Error('Clip polygon invalid');
  const clip = (
    measureRingSignedArea(clipRing, 'planar') < 0 ? [...clipRing].reverse() : clipRing
  ).slice(0, -1);
  const table = makeCatalogTable(
    'clipped',
    'Clipped features',
    {
      output_id: 'string',
      source_index: 'int32',
      source_id: 'string',
      part_index: 'int32',
      geometry_type: 'string',
      geometry: 'string'
    },
    'geometry'
  );
  let outsideCount = 0;
  let concaveCount = 0;
  let outputIndex = 0;
  features.features.forEach((feature, sourceIndex) => {
    const polygons = getPolygonParts(feature.geometry);
    const lines = getLineParts(feature.geometry);
    const emit = (type: string, coordinates: unknown, partIndex: number) => {
      appendCatalogRow(table, {
        output_id: `${sourceIndex}:${outputIndex++}`,
        source_index: sourceIndex,
        source_id: getFeatureIdentity(feature),
        part_index: partIndex,
        geometry_type: type,
        geometry: makeCatalogGeometryText(type, coordinates)
      });
    };
    if (polygons.length) {
      const surviving: CatalogPosition[][][] = [];
      for (const rings of polygons) {
        const shell = clipRingByConvex(rings[0], clip);
        if (!shell.length) continue;
        if (!isRingConvex(rings[0]) && shell.length !== rings[0].length) concaveCount++;
        const holes = rings
          .slice(1)
          .map(hole => clipRingByConvex(hole, clip))
          .filter(hole => hole.length);
        surviving.push([shell, ...holes]);
      }
      if (!surviving.length) outsideCount++;
      else if (surviving.length === 1) emit('Polygon', surviving[0], 0);
      else emit('MultiPolygon', surviving, 0);
    } else if (lines.length) {
      const pieces: CatalogPosition[][] = [];
      for (const line of lines) {
        let current: CatalogPosition[] = [];
        for (let index = 1; index < line.length; index++) {
          const clipped = clipSegmentByConvex(line[index - 1], line[index], clip);
          if (!clipped) {
            if (current.length) pieces.push(current);
            current = [];
            continue;
          }
          const continues =
            current.length &&
            current[current.length - 1][0] === clipped[0][0] &&
            current[current.length - 1][1] === clipped[0][1];
          if (!continues) {
            if (current.length) pieces.push(current);
            current = [clipped[0]];
          }
          current.push(clipped[1]);
        }
        if (current.length) pieces.push(current);
      }
      if (!pieces.length) outsideCount++;
      pieces.forEach((piece, partIndex) => emit('LineString', piece, partIndex));
    } else outsideCount++;
  });
  if (outsideCount)
    addCatalogDiagnostic(
      diagnostics,
      'feature-outside-clip',
      'info',
      'Features with no part inside the clip polygon (or unsupported geometry) produce no output.',
      outsideCount
    );
  if (concaveCount)
    addCatalogDiagnostic(
      diagnostics,
      'concave-shell-clipped',
      'warning',
      'A clipped concave shell may contain degenerate bridge edges.',
      concaveCount
    );
  addCatalogDiagnostic(
    diagnostics,
    'planar-edges',
    'info',
    'Edges are straight in coordinate space; the clip does not follow great circles.'
  );
  return {
    tables: [table],
    diagnostics,
    summary: {outputFeatureCount: table.rowCount, droppedFeatureCount: outsideCount}
  };
}
