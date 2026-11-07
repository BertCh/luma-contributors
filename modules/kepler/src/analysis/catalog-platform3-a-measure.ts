// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {CatalogRunOutput} from './catalog-binding-types';
import {
  addCatalogDiagnostic,
  appendCatalogRow,
  getLineParts,
  getPointPositions,
  makeCatalogGeometryText,
  makeCatalogTable,
  type CatalogCoordinateSystem,
  type CatalogDiagnostic,
  type CatalogFeatureCollection,
  type CatalogPosition
} from './catalog-geojson';
import {
  closeRing,
  collectPositions,
  getConvexHull,
  getDistanceToSegment,
  getFeatureBounds,
  getFeatureVertices,
  getSignedArea,
  getSourceColumns,
  isPointInPolygonRings,
  makeCirclePositions,
  projectPolygonParts,
  readPropertyNumber,
  readPropertyText,
  type Bounds,
  type PlanarPoint
} from './catalog-platform3-a-shared';
import {makeLocalFrame, type LocalFrame} from './catalog-local-frame';

export const MAXIMUM_CLEARANCE_VERTICES = 5000;

function makePolygonText(frame: LocalFrame, ring: readonly PlanarPoint[]): string {
  const positions = closeRing(ring.map(point => frame.inverse(point[0], point[1])));
  return makeCatalogGeometryText('Polygon', [positions]);
}

// ---------------------------------------------------------------------------------------------
// Shape descriptors
// ---------------------------------------------------------------------------------------------

type RotatedRectangle = {
  area: number;
  width: number;
  length: number;
  corners: PlanarPoint[];
  angle: number;
};

/** Minimum-area rotated rectangle over the hull edge directions (rotating calipers). */
export function getMinimumRectangle(points: readonly PlanarPoint[]): RotatedRectangle | null {
  const hull = getConvexHull(points);
  if (hull.length < 3) return null;
  let best: RotatedRectangle | null = null;
  for (let index = 0; index < hull.length; index++) {
    const first = hull[index];
    const second = hull[(index + 1) % hull.length];
    const angle = Math.atan2(second[1] - first[1], second[0] - first[0]);
    const cosine = Math.cos(angle);
    const sine = Math.sin(angle);
    let minimumU = Infinity;
    let maximumU = -Infinity;
    let minimumV = Infinity;
    let maximumV = -Infinity;
    for (const [x, y] of hull) {
      const u = x * cosine + y * sine;
      const v = -x * sine + y * cosine;
      if (u < minimumU) minimumU = u;
      if (u > maximumU) maximumU = u;
      if (v < minimumV) minimumV = v;
      if (v > maximumV) maximumV = v;
    }
    const sizeU = maximumU - minimumU;
    const sizeV = maximumV - minimumV;
    const area = sizeU * sizeV;
    if (!best || area < best.area - 1e-12) {
      const toPoint = (u: number, v: number): PlanarPoint => [
        u * cosine - v * sine,
        u * sine + v * cosine
      ];
      best = {
        area,
        width: Math.min(sizeU, sizeV),
        length: Math.max(sizeU, sizeV),
        corners: [
          toPoint(minimumU, minimumV),
          toPoint(maximumU, minimumV),
          toPoint(maximumU, maximumV),
          toPoint(minimumU, maximumV)
        ],
        angle: (angle * 180) / Math.PI
      };
    }
  }
  return best;
}

export function runShapeDescriptors(
  polygons: CatalogFeatureCollection,
  coordinateSystem: CatalogCoordinateSystem
): CatalogRunOutput {
  const diagnostics: CatalogDiagnostic[] = [];
  const frame = makeLocalFrame(coordinateSystem, collectPositions(polygons), {
    kind: 'lambert-equal-area'
  });
  const table = makeCatalogTable(
    'shape-descriptors',
    'Shape descriptors',
    {
      source_index: 'int32',
      source_id: 'string',
      area: 'float64',
      perimeter: 'float64',
      compactness: 'float64',
      convexity: 'float64',
      rectangularity: 'float64',
      elongation: 'float64',
      vertex_count: 'int32',
      geometry: 'string'
    },
    'geometry'
  );
  polygons.features.forEach((feature, sourceIndex) => {
    const parts = projectPolygonParts(feature, frame);
    if (!parts.length) {
      addCatalogDiagnostic(
        diagnostics,
        'feature-skipped',
        'warning',
        'Features without a usable polygon ring produce no descriptor row.'
      );
      return;
    }
    let area = 0;
    let perimeter = 0;
    let vertexCount = 0;
    const shellPoints: PlanarPoint[] = [];
    for (const rings of parts)
      rings.forEach((ring, ringIndex) => {
        const ringArea = Math.abs(getSignedArea(ring));
        area += ringIndex === 0 ? ringArea : -ringArea;
        vertexCount += ring.length;
        for (let index = 0; index < ring.length; index++) {
          const next = ring[(index + 1) % ring.length];
          perimeter += Math.hypot(next[0] - ring[index][0], next[1] - ring[index][1]);
        }
        if (ringIndex === 0) shellPoints.push(...ring);
      });
    const hull = getConvexHull(shellPoints);
    const hullArea = hull.length >= 3 ? Math.abs(getSignedArea(hull)) : 0;
    const rectangle = getMinimumRectangle(shellPoints);
    appendCatalogRow(table, {
      ...getSourceColumns(feature, sourceIndex),
      area,
      perimeter,
      compactness: perimeter > 0 ? (4 * Math.PI * area) / (perimeter * perimeter) : null,
      convexity: hullArea > 0 ? area / hullArea : null,
      rectangularity: rectangle && rectangle.area > 0 ? area / rectangle.area : null,
      elongation: rectangle && rectangle.length > 0 ? 1 - rectangle.width / rectangle.length : null,
      vertex_count: vertexCount,
      geometry: JSON.stringify(feature.geometry)
    });
  });
  addCatalogDiagnostic(
    diagnostics,
    'metric-frame',
    'info',
    coordinateSystem === 'planar'
      ? 'Areas and lengths are in coordinate units.'
      : 'Areas and lengths are computed in a local Lambert equal-area meter frame around the data center.'
  );
  return {tables: [table], diagnostics, summary: {descriptorCount: table.rowCount}};
}

// ---------------------------------------------------------------------------------------------
// Minimum bounding rectangle or circle
// ---------------------------------------------------------------------------------------------

type PlanarCircle = {x: number; y: number; radius: number};

function makeCircleFromTwo(first: PlanarPoint, second: PlanarPoint): PlanarCircle {
  return {
    x: (first[0] + second[0]) / 2,
    y: (first[1] + second[1]) / 2,
    radius: Math.hypot(first[0] - second[0], first[1] - second[1]) / 2
  };
}

function makeCircleFromThree(
  first: PlanarPoint,
  second: PlanarPoint,
  third: PlanarPoint
): PlanarCircle {
  const bx = second[0] - first[0];
  const by = second[1] - first[1];
  const cx = third[0] - first[0];
  const cy = third[1] - first[1];
  const determinant = 2 * (bx * cy - by * cx);
  if (Math.abs(determinant) < 1e-18) {
    const candidates = [
      makeCircleFromTwo(first, second),
      makeCircleFromTwo(first, third),
      makeCircleFromTwo(second, third)
    ];
    return candidates.reduce((best, candidate) =>
      candidate.radius > best.radius ? candidate : best
    );
  }
  const centerX = (cy * (bx * bx + by * by) - by * (cx * cx + cy * cy)) / determinant;
  const centerY = (bx * (cx * cx + cy * cy) - cx * (bx * bx + by * by)) / determinant;
  return {x: first[0] + centerX, y: first[1] + centerY, radius: Math.hypot(centerX, centerY)};
}

/** Smallest enclosing circle (incremental construction on the hull points). */
export function getMinimumCircle(points: readonly PlanarPoint[]): PlanarCircle | null {
  const hull = getConvexHull(points);
  if (!hull.length) return null;
  const contains = (circle: PlanarCircle, point: PlanarPoint) =>
    Math.hypot(point[0] - circle.x, point[1] - circle.y) <= circle.radius * (1 + 1e-12) + 1e-12;
  let circle: PlanarCircle = {x: hull[0][0], y: hull[0][1], radius: 0};
  for (let first = 1; first < hull.length; first++) {
    if (contains(circle, hull[first])) continue;
    circle = {x: hull[first][0], y: hull[first][1], radius: 0};
    for (let second = 0; second < first; second++) {
      if (contains(circle, hull[second])) continue;
      circle = makeCircleFromTwo(hull[first], hull[second]);
      for (let third = 0; third < second; third++)
        if (!contains(circle, hull[third]))
          circle = makeCircleFromThree(hull[first], hull[second], hull[third]);
    }
  }
  return circle;
}

export function runMinimumBounds(
  features: CatalogFeatureCollection,
  shape: 'rectangle' | 'circle',
  coordinateSystem: CatalogCoordinateSystem
): CatalogRunOutput {
  const diagnostics: CatalogDiagnostic[] = [];
  const frame = makeLocalFrame(coordinateSystem, collectPositions(features));
  const table = makeCatalogTable(
    'minimum-bounds',
    'Minimum bounds',
    {
      bounds_id: 'string',
      source_index: 'int32',
      source_id: 'string',
      shape: 'string',
      area: 'float64',
      width: 'float64',
      length: 'float64',
      angle_degrees: 'float64',
      radius: 'float64',
      geometry: 'string'
    },
    'geometry'
  );
  features.features.forEach((feature, sourceIndex) => {
    const points = getFeatureVertices(feature).map(position => frame.forward(position));
    const base = {bounds_id: `${sourceIndex}`, ...getSourceColumns(feature, sourceIndex), shape};
    if (shape === 'rectangle') {
      const rectangle = getMinimumRectangle(points);
      if (!rectangle) {
        addCatalogDiagnostic(
          diagnostics,
          'degenerate-feature',
          'warning',
          'Features with fewer than three non-collinear vertices produce no bounds row.'
        );
        return;
      }
      appendCatalogRow(table, {
        ...base,
        area: rectangle.area,
        width: rectangle.width,
        length: rectangle.length,
        angle_degrees: rectangle.angle,
        geometry: makePolygonText(frame, rectangle.corners)
      });
    } else {
      const circle = getMinimumCircle(points);
      if (!circle || !(circle.radius > 0)) {
        addCatalogDiagnostic(
          diagnostics,
          'degenerate-feature',
          'warning',
          'Features with a single distinct vertex produce no bounds row.'
        );
        return;
      }
      appendCatalogRow(table, {
        ...base,
        area: Math.PI * circle.radius ** 2,
        radius: circle.radius,
        geometry: makePolygonText(frame, makeCirclePositions(circle.x, circle.y, circle.radius, 64))
      });
    }
  });
  if (shape === 'circle')
    addCatalogDiagnostic(
      diagnostics,
      'inscribed-polygon',
      'info',
      'The circle is written as a 64-gon inscribed in the true circle; radius and area are exact.'
    );
  return {tables: [table], diagnostics, summary: {boundsCount: table.rowCount}};
}

// ---------------------------------------------------------------------------------------------
// Label point and maximum inscribed circle (pole of inaccessibility)
// ---------------------------------------------------------------------------------------------

function getSignedDistanceToPolygon(
  x: number,
  y: number,
  rings: readonly (readonly PlanarPoint[])[]
): number {
  let minimum = Infinity;
  for (const ring of rings)
    for (let index = 0; index < ring.length; index++) {
      const next = ring[(index + 1) % ring.length];
      const distance = getDistanceToSegment(
        x,
        y,
        ring[index][0],
        ring[index][1],
        next[0],
        next[1]
      ).distance;
      if (distance < minimum) minimum = distance;
    }
  return isPointInPolygonRings(x, y, rings) ? minimum : -minimum;
}

type LabelCell = {x: number; y: number; half: number; distance: number; maximum: number};

/** Quad-tree refinement of the point with the largest distance to the boundary. */
export function findPoleOfInaccessibility(
  rings: readonly (readonly PlanarPoint[])[],
  precision: number
): {x: number; y: number; distance: number} {
  const shell = rings[0];
  let minimumX = Infinity;
  let minimumY = Infinity;
  let maximumX = -Infinity;
  let maximumY = -Infinity;
  for (const [x, y] of shell) {
    if (x < minimumX) minimumX = x;
    if (x > maximumX) maximumX = x;
    if (y < minimumY) minimumY = y;
    if (y > maximumY) maximumY = y;
  }
  const size = Math.min(maximumX - minimumX, maximumY - minimumY);
  const makeCell = (x: number, y: number, half: number): LabelCell => {
    const distance = getSignedDistanceToPolygon(x, y, rings);
    return {x, y, half, distance, maximum: distance + half * Math.SQRT2};
  };
  if (size === 0) return {x: minimumX, y: minimumY, distance: 0};
  const half = size / 2;
  const queue: LabelCell[] = [];
  for (let x = minimumX; x < maximumX; x += size)
    for (let y = minimumY; y < maximumY; y += size) queue.push(makeCell(x + half, y + half, half));
  const centroidX = shell.reduce((sum, point) => sum + point[0], 0) / shell.length;
  const centroidY = shell.reduce((sum, point) => sum + point[1], 0) / shell.length;
  let best = makeCell(centroidX, centroidY, 0);
  const boxCell = makeCell(
    minimumX + (maximumX - minimumX) / 2,
    minimumY + (maximumY - minimumY) / 2,
    0
  );
  if (boxCell.distance > best.distance) best = boxCell;
  let iterations = 0;
  while (queue.length && iterations++ < 20000) {
    let bestIndex = 0;
    for (let index = 1; index < queue.length; index++)
      if (queue[index].maximum > queue[bestIndex].maximum) bestIndex = index;
    const cell = queue.splice(bestIndex, 1)[0];
    if (cell.distance > best.distance) best = cell;
    if (cell.maximum - best.distance <= precision) continue;
    const quarter = cell.half / 2;
    queue.push(
      makeCell(cell.x - quarter, cell.y - quarter, quarter),
      makeCell(cell.x + quarter, cell.y - quarter, quarter),
      makeCell(cell.x - quarter, cell.y + quarter, quarter),
      makeCell(cell.x + quarter, cell.y + quarter, quarter)
    );
  }
  return {x: best.x, y: best.y, distance: best.distance};
}

export function runLabelPoints(
  polygons: CatalogFeatureCollection,
  asCircle: boolean,
  precisionFraction: number,
  coordinateSystem: CatalogCoordinateSystem
): CatalogRunOutput {
  const diagnostics: CatalogDiagnostic[] = [];
  const frame = makeLocalFrame(coordinateSystem, collectPositions(polygons));
  const table = makeCatalogTable(
    asCircle ? 'inscribed-circles' : 'label-points',
    asCircle ? 'Maximum inscribed circles' : 'Label points',
    {
      label_id: 'string',
      source_index: 'int32',
      source_id: 'string',
      part_index: 'int32',
      clearance: 'float64',
      geometry: 'string'
    },
    'geometry'
  );
  polygons.features.forEach((feature, sourceIndex) => {
    const parts = projectPolygonParts(feature, frame);
    if (!parts.length) {
      addCatalogDiagnostic(
        diagnostics,
        'feature-skipped',
        'warning',
        'Features without a usable polygon ring produce no row.'
      );
      return;
    }
    let bestPart = -1;
    let best = {x: 0, y: 0, distance: -Infinity};
    parts.forEach((rings, partIndex) => {
      const shellBounds = rings[0].reduce(
        (extent, point) => Math.max(extent, Math.abs(point[0]), Math.abs(point[1])),
        0
      );
      const found = findPoleOfInaccessibility(
        rings,
        Math.max(shellBounds * precisionFraction, 1e-9)
      );
      if (found.distance > best.distance) {
        best = found;
        bestPart = partIndex;
      }
    });
    const center = frame.inverse(best.x, best.y);
    appendCatalogRow(table, {
      label_id: `${sourceIndex}:${bestPart}`,
      ...getSourceColumns(feature, sourceIndex),
      part_index: bestPart,
      clearance: best.distance,
      geometry: asCircle
        ? makePolygonText(
            frame,
            makeCirclePositions(best.x, best.y, Math.max(best.distance, 0), 64)
          )
        : makeCatalogGeometryText('Point', center)
    });
  });
  if (
    polygons.features.some(
      feature => (feature.geometry as {type?: string})?.type === 'MultiPolygon'
    )
  )
    addCatalogDiagnostic(
      diagnostics,
      'largest-part',
      'info',
      'MultiPolygon features report the part with the largest clearance.'
    );
  return {tables: [table], diagnostics, summary: {outputCount: table.rowCount}};
}

// ---------------------------------------------------------------------------------------------
// Minimum clearance
// ---------------------------------------------------------------------------------------------

export function runMinimumClearance(
  features: CatalogFeatureCollection,
  coordinateSystem: CatalogCoordinateSystem
): CatalogRunOutput {
  const diagnostics: CatalogDiagnostic[] = [];
  const frame = makeLocalFrame(coordinateSystem, collectPositions(features));
  const table = makeCatalogTable(
    'minimum-clearance',
    'Minimum clearance',
    {
      source_index: 'int32',
      source_id: 'string',
      clearance: 'float64',
      status: 'string',
      geometry: 'string'
    },
    'geometry'
  );
  features.features.forEach((feature, sourceIndex) => {
    const base = getSourceColumns(feature, sourceIndex);
    const rings: PlanarPoint[][] = [];
    const type = (feature.geometry as {type?: string} | null)?.type ?? '';
    if (type.includes('Polygon'))
      for (const polygon of projectPolygonParts(feature, frame)) rings.push(...polygon);
    else
      for (const part of getLineParts(feature.geometry))
        if (part.length >= 3) rings.push(part.map(position => frame.forward(position)));
    const vertexCount = rings.reduce((sum, ring) => sum + ring.length, 0);
    if (vertexCount < 3) {
      appendCatalogRow(table, {...base, status: 'degenerate'});
      return;
    }
    if (vertexCount > MAXIMUM_CLEARANCE_VERTICES) {
      appendCatalogRow(table, {...base, status: 'skipped-too-many-vertices'});
      addCatalogDiagnostic(
        diagnostics,
        'clearance-skipped',
        'warning',
        `Features over ${MAXIMUM_CLEARANCE_VERTICES} vertices are not measured.`
      );
      return;
    }
    const isPolygon = type.includes('Polygon');
    let bestDistance = Infinity;
    let bestPair: [PlanarPoint, PlanarPoint] | null = null;
    rings.forEach((ring, ringIndex) => {
      ring.forEach((vertex, vertexIndex) => {
        rings.forEach((otherRing, otherRingIndex) => {
          const segmentCount = isPolygon ? otherRing.length : otherRing.length - 1;
          for (let segmentIndex = 0; segmentIndex < segmentCount; segmentIndex++) {
            const next = (segmentIndex + 1) % otherRing.length;
            if (
              ringIndex === otherRingIndex &&
              (segmentIndex === vertexIndex || next === vertexIndex)
            )
              continue;
            const found = getDistanceToSegment(
              vertex[0],
              vertex[1],
              otherRing[segmentIndex][0],
              otherRing[segmentIndex][1],
              otherRing[next][0],
              otherRing[next][1]
            );
            // Coincident vertices do not count: clearance measures separation, not touching.
            if (found.distance > 1e-12 && found.distance < bestDistance) {
              bestDistance = found.distance;
              bestPair = [vertex, [found.x, found.y]];
            }
          }
        });
      });
    });
    const pair = bestPair as [PlanarPoint, PlanarPoint] | null;
    if (!pair) {
      appendCatalogRow(table, {...base, status: 'degenerate'});
      return;
    }
    appendCatalogRow(table, {
      ...base,
      clearance: bestDistance,
      status: 'ok',
      geometry: makeCatalogGeometryText('LineString', [
        frame.inverse(pair[0][0], pair[0][1]),
        frame.inverse(pair[1][0], pair[1][1])
      ])
    });
  });
  addCatalogDiagnostic(
    diagnostics,
    'vertex-to-segment',
    'info',
    'Clearance is the smallest distance from a vertex to a segment that does not own it, as in the JTS definition restricted to vertex-segment pairs.'
  );
  return {
    tables: [table],
    diagnostics,
    summary: {measuredCount: table.columns['status'].filter(status => status === 'ok').length}
  };
}

// ---------------------------------------------------------------------------------------------
// Geographic distribution
// ---------------------------------------------------------------------------------------------

type WeightedPoint = {x: number; y: number; weight: number};

function getMedianCenter(points: readonly WeightedPoint[], start: PlanarPoint): PlanarPoint {
  let [x, y] = start;
  for (let iteration = 0; iteration < 200; iteration++) {
    let numeratorX = 0;
    let numeratorY = 0;
    let denominator = 0;
    for (const point of points) {
      const distance = Math.max(Math.hypot(point.x - x, point.y - y), 1e-12);
      numeratorX += (point.weight * point.x) / distance;
      numeratorY += (point.weight * point.y) / distance;
      denominator += point.weight / distance;
    }
    const nextX = numeratorX / denominator;
    const nextY = numeratorY / denominator;
    const moved = Math.hypot(nextX - x, nextY - y);
    x = nextX;
    y = nextY;
    if (moved < 1e-9) break;
  }
  return [x, y];
}

export function runGeographicDistribution(
  features: CatalogFeatureCollection,
  parameters: {
    groupProperty: string;
    weightProperty: string;
    standardDeviations: number;
    coordinateSystem: CatalogCoordinateSystem;
  }
): CatalogRunOutput {
  const diagnostics: CatalogDiagnostic[] = [];
  const frame = makeLocalFrame(parameters.coordinateSystem, collectPositions(features));
  const groups = new Map<string, WeightedPoint[]>();
  let skippedCount = 0;
  features.features.forEach(feature => {
    const weight = parameters.weightProperty
      ? readPropertyNumber(feature, parameters.weightProperty)
      : 1;
    if (weight === null || weight < 0) {
      skippedCount++;
      return;
    }
    let representatives: CatalogPosition[] = getPointPositions(feature.geometry);
    if (!representatives.length) {
      const vertices = getFeatureVertices(feature);
      if (vertices.length)
        representatives = [
          [
            vertices.reduce((sum, vertex) => sum + vertex[0], 0) / vertices.length,
            vertices.reduce((sum, vertex) => sum + vertex[1], 0) / vertices.length
          ]
        ];
    }
    const key = readPropertyText(feature, parameters.groupProperty);
    for (const position of representatives) {
      const [x, y] = frame.forward(position);
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key)!.push({x, y, weight});
    }
  });
  if (skippedCount)
    addCatalogDiagnostic(
      diagnostics,
      'invalid-weight',
      'warning',
      'Features with a missing, non-numeric or negative weight are skipped.',
      skippedCount
    );
  const ellipseTable = makeCatalogTable(
    'distribution-ellipses',
    'Standard deviational ellipses',
    {
      group: 'string',
      count: 'int32',
      total_weight: 'float64',
      standard_distance: 'float64',
      major_radius: 'float64',
      minor_radius: 'float64',
      rotation_degrees: 'float64',
      geometry: 'string'
    },
    'geometry'
  );
  const centerTable = makeCatalogTable(
    'distribution-centers',
    'Mean and median centers',
    {center_id: 'string', group: 'string', kind: 'string', count: 'int32', geometry: 'string'},
    'geometry'
  );
  for (const [group, points] of [...groups.entries()].sort(([first], [second]) =>
    first < second ? -1 : first > second ? 1 : 0
  )) {
    const totalWeight = points.reduce((sum, point) => sum + point.weight, 0);
    if (!(totalWeight > 0)) {
      addCatalogDiagnostic(
        diagnostics,
        'zero-weight-group',
        'warning',
        'Groups with zero total weight are skipped.'
      );
      continue;
    }
    const meanX = points.reduce((sum, point) => sum + point.weight * point.x, 0) / totalWeight;
    const meanY = points.reduce((sum, point) => sum + point.weight * point.y, 0) / totalWeight;
    const median = getMedianCenter(points, [meanX, meanY]);
    let sumXX = 0;
    let sumYY = 0;
    let sumXY = 0;
    for (const point of points) {
      const deltaX = point.x - meanX;
      const deltaY = point.y - meanY;
      sumXX += point.weight * deltaX * deltaX;
      sumYY += point.weight * deltaY * deltaY;
      sumXY += point.weight * deltaX * deltaY;
    }
    const angle = 0.5 * Math.atan2(2 * sumXY, sumXX - sumYY);
    const cosine = Math.cos(angle);
    const sine = Math.sin(angle);
    let major = 0;
    let minor = 0;
    for (const point of points) {
      const deltaX = point.x - meanX;
      const deltaY = point.y - meanY;
      major += point.weight * (deltaX * cosine + deltaY * sine) ** 2;
      minor += point.weight * (-deltaX * sine + deltaY * cosine) ** 2;
    }
    const majorRadius = Math.sqrt(major / totalWeight) * parameters.standardDeviations;
    const minorRadius = Math.sqrt(minor / totalWeight) * parameters.standardDeviations;
    const hasEllipse = points.length >= 3 && majorRadius > 0 && minorRadius > 0;
    const ring: PlanarPoint[] = [];
    if (hasEllipse)
      for (let index = 0; index < 64; index++) {
        const phase = (2 * Math.PI * index) / 64;
        const localX = majorRadius * Math.cos(phase);
        const localY = minorRadius * Math.sin(phase);
        ring.push([
          meanX + localX * cosine - localY * sine,
          meanY + localX * sine + localY * cosine
        ]);
      }
    appendCatalogRow(ellipseTable, {
      group,
      count: points.length,
      total_weight: totalWeight,
      standard_distance: Math.sqrt((sumXX + sumYY) / totalWeight),
      major_radius: hasEllipse ? majorRadius : null,
      minor_radius: hasEllipse ? minorRadius : null,
      rotation_degrees: hasEllipse ? (angle * 180) / Math.PI : null,
      geometry: hasEllipse ? makePolygonText(frame, ring) : null
    });
    appendCatalogRow(centerTable, {
      center_id: `${group}:mean`,
      group,
      kind: 'mean',
      count: points.length,
      geometry: makeCatalogGeometryText('Point', frame.inverse(meanX, meanY))
    });
    appendCatalogRow(centerTable, {
      center_id: `${group}:median`,
      group,
      kind: 'median',
      count: points.length,
      geometry: makeCatalogGeometryText('Point', frame.inverse(median[0], median[1]))
    });
    if (!hasEllipse)
      addCatalogDiagnostic(
        diagnostics,
        'ellipse-undefined',
        'warning',
        'Groups with fewer than three points or no spread have no ellipse polygon.'
      );
  }
  return {
    tables: [ellipseTable, centerTable],
    diagnostics,
    summary: {groupCount: ellipseTable.rowCount}
  };
}

// ---------------------------------------------------------------------------------------------
// Hilbert order
// ---------------------------------------------------------------------------------------------

const toFloat32 = Math.fround;

/** Classic xy2d Hilbert index of a cell; the curve starts lower-left and ends lower-right. */
export function getHilbertIndex(cellX: number, cellY: number, order: number): number {
  const size = 2 ** order;
  let x = cellX;
  let y = cellY;
  let index = 0;
  for (let step = size / 2; step >= 1; step = Math.floor(step / 2)) {
    const rx = (x & step) > 0 ? 1 : 0;
    const ry = (y & step) > 0 ? 1 : 0;
    index += step * step * ((3 * rx) ^ ry);
    if (ry === 0) {
      if (rx === 1) {
        x = size - 1 - x;
        y = size - 1 - y;
      }
      const swap = x;
      x = y;
      y = swap;
    }
  }
  return index;
}

export type HilbertItems = {
  /** float32 box minima and maxima per feature, NaN for features without geometry. */
  minima: Float32Array;
  maxima: Float32Array;
  /** float32 bounds [minX, minY, maxX, maxY] of all valid boxes. */
  bounds: Float32Array;
  validCount: number;
};

export function getHilbertItems(features: CatalogFeatureCollection): HilbertItems {
  const minima = new Float32Array(features.features.length * 2).fill(NaN);
  const maxima = new Float32Array(features.features.length * 2).fill(NaN);
  const bounds = new Float32Array([Infinity, Infinity, -Infinity, -Infinity]);
  let validCount = 0;
  features.features.forEach((feature, index) => {
    const box: Bounds | null = getFeatureBounds(feature);
    if (!box) return;
    validCount++;
    minima[2 * index] = box[0];
    minima[2 * index + 1] = box[1];
    maxima[2 * index] = box[2];
    maxima[2 * index + 1] = box[3];
    bounds[0] = Math.min(bounds[0], minima[2 * index]);
    bounds[1] = Math.min(bounds[1], minima[2 * index + 1]);
    bounds[2] = Math.max(bounds[2], maxima[2 * index]);
    bounds[3] = Math.max(bounds[3], maxima[2 * index + 1]);
  });
  return {minima, maxima, bounds, validCount};
}

/** CPU keys mirroring the GPUHilbertKeys float32 box-center quantization. */
export function getHilbertKeys(items: HilbertItems, order: number): Uint32Array {
  const count = items.minima.length / 2;
  const keys = new Uint32Array(count);
  const invalid = order >= 16 ? 0xffffffff : 4 ** order;
  const cells = 2 ** order;
  const quantize = (value: number, minimum: number, maximum: number): number => {
    const extent = toFloat32(maximum - minimum);
    if (!(extent > 0) || !Number.isFinite(extent)) return 0;
    const normalized = Math.min(1, Math.max(0, toFloat32(toFloat32(value - minimum) / extent)));
    return Math.min(Math.floor(toFloat32(normalized * cells)), cells - 1);
  };
  for (let index = 0; index < count; index++) {
    const minimumX = items.minima[2 * index];
    const minimumY = items.minima[2 * index + 1];
    const maximumX = items.maxima[2 * index];
    const maximumY = items.maxima[2 * index + 1];
    if (![minimumX, minimumY, maximumX, maximumY].every(Number.isFinite)) {
      keys[index] = invalid;
      continue;
    }
    const centerX = toFloat32(toFloat32(minimumX * 0.5) + toFloat32(maximumX * 0.5));
    const centerY = toFloat32(toFloat32(minimumY * 0.5) + toFloat32(maximumY * 0.5));
    keys[index] = getHilbertIndex(
      quantize(centerX, items.bounds[0], items.bounds[2]),
      quantize(centerY, items.bounds[1], items.bounds[3]),
      order
    );
  }
  return keys;
}

/** Stable ascending permutation of keys, invalid keys last by their sentinel value. */
export function sortHilbertRows(keys: Uint32Array): number[] {
  return Array.from(keys.keys()).sort(
    (first, second) => keys[first] - keys[second] || first - second
  );
}

export function makeHilbertOutput(
  features: CatalogFeatureCollection,
  keys: Uint32Array,
  sortedRows: readonly number[],
  order: number
): CatalogRunOutput {
  const table = makeCatalogTable(
    'hilbert-order',
    'Hilbert order',
    {
      rank: 'int32',
      source_index: 'int32',
      source_id: 'string',
      hilbert_key: 'float64',
      geometry: 'string'
    },
    'geometry'
  );
  const invalid = order >= 16 ? 0xffffffff : 4 ** order;
  const diagnostics: CatalogDiagnostic[] = [];
  sortedRows.forEach((sourceIndex, rank) => {
    const feature = features.features[sourceIndex];
    const isInvalid = keys[sourceIndex] === invalid;
    const bounds = getFeatureBounds(feature);
    if (isInvalid)
      addCatalogDiagnostic(
        diagnostics,
        'feature-without-geometry',
        'warning',
        'Features without finite geometry sort last with the invalid key.'
      );
    appendCatalogRow(table, {
      rank,
      ...getSourceColumns(feature, sourceIndex),
      hilbert_key: keys[sourceIndex],
      geometry: bounds
        ? makeCatalogGeometryText('Point', [
            (bounds[0] + bounds[2]) / 2,
            (bounds[1] + bounds[3]) / 2
          ])
        : null
    });
  });
  return {tables: [table], diagnostics, summary: {featureCount: table.rowCount, order}};
}

export function runHilbertOrder(
  features: CatalogFeatureCollection,
  order: number
): CatalogRunOutput {
  const items = getHilbertItems(features);
  const keys = getHilbertKeys(items, order);
  return makeHilbertOutput(features, keys, sortHilbertRows(keys), order);
}
