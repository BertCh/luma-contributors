// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  CATALOG_POLYGON_TRIANGLE_COLUMNS,
  appendCatalogTriangleRows,
  projectCatalogPolygonParts
} from './catalog-triangle-output';
import type {CatalogRunOutput} from './catalog-binding-types';
import {
  addCatalogDiagnostic,
  appendCatalogRow,
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
  getFeatureVertices,
  getSignedArea,
  getSourceColumns,
  intersectSegments,
  isPointInPolygonRings,
  readPropertyText,
  type PlanarPoint
} from './catalog-platform3-a-shared';
import {makeLocalFrame, type LocalFrame} from './catalog-local-frame';

export type TriangleIndices = [number, number, number];

// ---------------------------------------------------------------------------------------------
// Delaunay triangulation (Bowyer-Watson)
// ---------------------------------------------------------------------------------------------

type WorkingTriangle = {
  a: number;
  b: number;
  c: number;
  centerX: number;
  centerY: number;
  radiusSquared: number;
};

function makeWorkingTriangle(
  vertices: readonly PlanarPoint[],
  a: number,
  b: number,
  c: number
): WorkingTriangle {
  const [ax, ay] = vertices[a];
  const [bx, by] = vertices[b];
  const [cx, cy] = vertices[c];
  const determinant = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
  const centerX =
    ((ax * ax + ay * ay) * (by - cy) +
      (bx * bx + by * by) * (cy - ay) +
      (cx * cx + cy * cy) * (ay - by)) /
    determinant;
  const centerY =
    ((ax * ax + ay * ay) * (cx - bx) +
      (bx * bx + by * by) * (ax - cx) +
      (cx * cx + cy * cy) * (bx - ax)) /
    determinant;
  return {
    a,
    b,
    c,
    centerX,
    centerY,
    radiusSquared: (ax - centerX) ** 2 + (ay - centerY) ** 2
  };
}

/** Counter-clockwise triangles over distinct points; collinear inputs return no triangles. */
export function triangulatePoints(points: readonly PlanarPoint[]): TriangleIndices[] {
  const count = points.length;
  if (count < 3) return [];
  let minimumX = Infinity;
  let minimumY = Infinity;
  let maximumX = -Infinity;
  let maximumY = -Infinity;
  for (const [x, y] of points) {
    minimumX = Math.min(minimumX, x);
    maximumX = Math.max(maximumX, x);
    minimumY = Math.min(minimumY, y);
    maximumY = Math.max(maximumY, y);
  }
  const span = Math.max(maximumX - minimumX, maximumY - minimumY);
  if (!(span > 0)) return [];
  const centerX = (minimumX + maximumX) / 2;
  const centerY = (minimumY + maximumY) / 2;
  const vertices: PlanarPoint[] = points.map(([x, y]) => [
    (x - centerX) / span,
    (y - centerY) / span
  ]);
  const reach = 200;
  vertices.push([-reach, -reach], [reach, -reach], [0, reach]);
  let triangles = [makeWorkingTriangle(vertices, count, count + 1, count + 2)];
  for (let pointIndex = 0; pointIndex < count; pointIndex++) {
    const [x, y] = vertices[pointIndex];
    const kept: WorkingTriangle[] = [];
    const edgeCounts = new Map<string, {from: number; to: number; count: number}>();
    for (const triangle of triangles) {
      if (
        (x - triangle.centerX) ** 2 + (y - triangle.centerY) ** 2 <
        triangle.radiusSquared * (1 - 1e-12)
      ) {
        for (const [from, to] of [
          [triangle.a, triangle.b],
          [triangle.b, triangle.c],
          [triangle.c, triangle.a]
        ]) {
          const key = from < to ? `${from}:${to}` : `${to}:${from}`;
          const existing = edgeCounts.get(key);
          if (existing) existing.count++;
          else edgeCounts.set(key, {from, to, count: 1});
        }
      } else kept.push(triangle);
    }
    for (const edge of edgeCounts.values())
      if (edge.count === 1) {
        const [ax, ay] = vertices[edge.from];
        const [bx, by] = vertices[edge.to];
        if (Math.abs((bx - ax) * (y - ay) - (by - ay) * (x - ax)) > 1e-18)
          kept.push(makeWorkingTriangle(vertices, edge.from, edge.to, pointIndex));
      }
    triangles = kept;
  }
  const result: TriangleIndices[] = [];
  for (const triangle of triangles) {
    if (triangle.a >= count || triangle.b >= count || triangle.c >= count) continue;
    const [ax, ay] = points[triangle.a];
    const [bx, by] = points[triangle.b];
    const [cx, cy] = points[triangle.c];
    const cross = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
    if (cross === 0) continue;
    result.push(
      cross > 0 ? [triangle.a, triangle.b, triangle.c] : [triangle.a, triangle.c, triangle.b]
    );
  }
  return result;
}

type SitePoint = {position: CatalogPosition; featureIndex: number; pointIndex: number};

/** Distinct sites (first occurrence wins) in feature order, with duplicate and non-finite counts. */
export function collectSites(
  features: CatalogFeatureCollection,
  useAllVertices: boolean
): {sites: SitePoint[]; duplicateCount: number} {
  const seen = new Set<string>();
  const sites: SitePoint[] = [];
  let duplicateCount = 0;
  features.features.forEach((feature, featureIndex) => {
    const positions = useAllVertices
      ? getFeatureVertices(feature)
      : getPointPositions(feature.geometry);
    positions.forEach((position, pointIndex) => {
      if (!Number.isFinite(position[0]) || !Number.isFinite(position[1])) return;
      const key = `${position[0]}:${position[1]}`;
      if (seen.has(key)) duplicateCount++;
      else {
        seen.add(key);
        sites.push({position, featureIndex, pointIndex});
      }
    });
  });
  return {sites, duplicateCount};
}

export function runDelaunayTriangulation(
  points: CatalogFeatureCollection,
  coordinateSystem: CatalogCoordinateSystem
): CatalogRunOutput {
  const diagnostics: CatalogDiagnostic[] = [];
  const {sites, duplicateCount} = collectSites(points, false);
  const frame = makeLocalFrame(
    coordinateSystem,
    sites.map(site => site.position)
  );
  const projected = sites.map(site => frame.forward(site.position));
  const triangles = triangulatePoints(projected);
  const triangleTable = makeCatalogTable(
    'delaunay-triangles',
    'Delaunay triangles',
    {
      triangle_id: 'string',
      point_a: 'int32',
      point_b: 'int32',
      point_c: 'int32',
      source_a: 'int32',
      source_b: 'int32',
      source_c: 'int32',
      area: 'float64',
      geometry: 'string'
    },
    'geometry'
  );
  const edgeTable = makeCatalogTable(
    'delaunay-edges',
    'Delaunay edges',
    {
      edge_id: 'string',
      point_from: 'int32',
      point_to: 'int32',
      length: 'float64',
      geometry: 'string'
    },
    'geometry'
  );
  const edgeSet = new Set<string>();
  const edges: [number, number][] = [];
  triangles.forEach((triangle, triangleIndex) => {
    const [a, b, c] = triangle;
    appendCatalogRow(triangleTable, {
      triangle_id: `${triangleIndex}`,
      point_a: a,
      point_b: b,
      point_c: c,
      source_a: sites[a].featureIndex,
      source_b: sites[b].featureIndex,
      source_c: sites[c].featureIndex,
      area: Math.abs(getSignedArea([projected[a], projected[b], projected[c]])),
      geometry: makeCatalogGeometryText('Polygon', [
        closeRing([sites[a].position, sites[b].position, sites[c].position])
      ])
    });
    for (const [from, to] of [
      [a, b],
      [b, c],
      [c, a]
    ]) {
      const key = from < to ? `${from}:${to}` : `${to}:${from}`;
      if (!edgeSet.has(key)) {
        edgeSet.add(key);
        edges.push(from < to ? [from, to] : [to, from]);
      }
    }
  });
  edges
    .sort((first, second) => first[0] - second[0] || first[1] - second[1])
    .forEach(([from, to], edgeIndex) => {
      appendCatalogRow(edgeTable, {
        edge_id: `${edgeIndex}`,
        point_from: from,
        point_to: to,
        length: Math.hypot(
          projected[to][0] - projected[from][0],
          projected[to][1] - projected[from][1]
        ),
        geometry: makeCatalogGeometryText('LineString', [sites[from].position, sites[to].position])
      });
    });
  if (duplicateCount)
    addCatalogDiagnostic(
      diagnostics,
      'duplicate-points',
      'warning',
      'Points at an already seen position are ignored; the first occurrence is the site.',
      duplicateCount
    );
  if (!triangles.length)
    addCatalogDiagnostic(
      diagnostics,
      'no-triangles',
      'warning',
      'Fewer than three distinct points, or all points collinear: no triangles.'
    );
  return {
    tables: [triangleTable, edgeTable],
    diagnostics,
    summary: {siteCount: sites.length, triangleCount: triangles.length, edgeCount: edges.length}
  };
}

// ---------------------------------------------------------------------------------------------
// Voronoi diagram (bounded half-plane clipping)
// ---------------------------------------------------------------------------------------------

function clipConvexByBisector(
  cell: readonly PlanarPoint[],
  site: PlanarPoint,
  other: PlanarPoint
): PlanarPoint[] {
  // Keep points closer to the site: f(p) = (p - midpoint) . (site - other) >= 0.
  const directionX = site[0] - other[0];
  const directionY = site[1] - other[1];
  const midpointX = (site[0] + other[0]) / 2;
  const midpointY = (site[1] + other[1]) / 2;
  const side = (point: PlanarPoint) =>
    (point[0] - midpointX) * directionX + (point[1] - midpointY) * directionY;
  const result: PlanarPoint[] = [];
  for (let index = 0; index < cell.length; index++) {
    const current = cell[index];
    const next = cell[(index + 1) % cell.length];
    const currentSide = side(current);
    const nextSide = side(next);
    if (currentSide >= 0) result.push(current);
    if (currentSide >= 0 !== nextSide >= 0) {
      const fraction = currentSide / (currentSide - nextSide);
      result.push([
        current[0] + fraction * (next[0] - current[0]),
        current[1] + fraction * (next[1] - current[1])
      ]);
    }
  }
  return result;
}

export function runVoronoiDiagram(
  points: CatalogFeatureCollection,
  parameters: {padding: number; coordinateSystem: CatalogCoordinateSystem}
): CatalogRunOutput {
  const diagnostics: CatalogDiagnostic[] = [];
  const {sites, duplicateCount} = collectSites(points, false);
  const frame = makeLocalFrame(
    parameters.coordinateSystem,
    sites.map(site => site.position)
  );
  const projected = sites.map(site => frame.forward(site.position));
  const table = makeCatalogTable(
    'voronoi-cells',
    'Voronoi cells',
    {
      cell_id: 'string',
      source_index: 'int32',
      source_id: 'string',
      point_index: 'int32',
      area: 'float64',
      vertex_count: 'int32',
      geometry: 'string'
    },
    'geometry'
  );
  if (projected.length) {
    let minimumX = Infinity;
    let minimumY = Infinity;
    let maximumX = -Infinity;
    let maximumY = -Infinity;
    for (const [x, y] of projected) {
      minimumX = Math.min(minimumX, x);
      maximumX = Math.max(maximumX, x);
      minimumY = Math.min(minimumY, y);
      maximumY = Math.max(maximumY, y);
    }
    const margin = Math.max(
      Math.max(maximumX - minimumX, maximumY - minimumY) * parameters.padding,
      1e-9
    );
    const box: PlanarPoint[] = [
      [minimumX - margin, minimumY - margin],
      [maximumX + margin, minimumY - margin],
      [maximumX + margin, maximumY + margin],
      [minimumX - margin, maximumY + margin]
    ];
    projected.forEach((site, siteIndex) => {
      let cell = box;
      const order = projected
        .map((other, otherIndex) => ({
          otherIndex,
          distance: Math.hypot(other[0] - site[0], other[1] - site[1])
        }))
        .filter(item => item.otherIndex !== siteIndex)
        .sort(
          (first, second) =>
            first.distance - second.distance || first.otherIndex - second.otherIndex
        );
      let reach = Infinity;
      for (const item of order) {
        if (item.distance / 2 > reach) break;
        cell = clipConvexByBisector(cell, site, projected[item.otherIndex]);
        if (cell.length < 3) break;
        reach = Math.max(
          ...cell.map(vertex => Math.hypot(vertex[0] - site[0], vertex[1] - site[1]))
        );
      }
      if (cell.length < 3) return;
      const source = sites[siteIndex];
      appendCatalogRow(table, {
        cell_id: `${source.featureIndex}:${source.pointIndex}`,
        ...getSourceColumns(points.features[source.featureIndex], source.featureIndex),
        point_index: source.pointIndex,
        area: Math.abs(getSignedArea(cell)),
        vertex_count: cell.length,
        geometry: makeCatalogGeometryText('Polygon', [
          closeRing(cell.map(vertex => frame.inverse(vertex[0], vertex[1])))
        ])
      });
    });
  }
  if (duplicateCount)
    addCatalogDiagnostic(
      diagnostics,
      'duplicate-points',
      'warning',
      'Points at an already seen position get no cell; the first occurrence owns it.',
      duplicateCount
    );
  addCatalogDiagnostic(
    diagnostics,
    'clipped-to-padded-extent',
    'info',
    'Cells on the hull are clipped to the point extent expanded by the padding fraction.'
  );
  return {
    tables: [table],
    diagnostics,
    summary: {cellCount: table.rowCount, siteCount: sites.length}
  };
}

// ---------------------------------------------------------------------------------------------
// Concave hull (Delaunay edge-length trimming)
// ---------------------------------------------------------------------------------------------

function getEdgeKey(from: number, to: number): string {
  return `${from}:${to}`;
}

/** Boundary loops (directed, counter-clockwise outer) of a set of counter-clockwise triangles. */
function getBoundaryLoops(triangles: readonly TriangleIndices[]): number[][] {
  const directed = new Set<string>();
  for (const [a, b, c] of triangles)
    for (const [from, to] of [
      [a, b],
      [b, c],
      [c, a]
    ])
      directed.add(getEdgeKey(from, to));
  const outgoing = new Map<number, number[]>();
  for (const key of directed) {
    const [from, to] = key.split(':').map(Number);
    if (directed.has(getEdgeKey(to, from))) continue;
    if (!outgoing.has(from)) outgoing.set(from, []);
    outgoing.get(from)!.push(to);
  }
  const loops: number[][] = [];
  for (const start of [...outgoing.keys()].sort((first, second) => first - second)) {
    while (outgoing.get(start)?.length) {
      const loop = [start];
      let current = start;
      for (;;) {
        const targets = outgoing.get(current);
        if (!targets?.length) break;
        const next = targets.shift()!;
        if (next === start) break;
        loop.push(next);
        current = next;
      }
      if (loop.length >= 3) loops.push(loop);
    }
  }
  return loops;
}

/** Counter-clockwise boundary loops of the Delaunay triangulation trimmed by `cutoff`. */
export function getConcaveBoundary(
  projected: readonly PlanarPoint[],
  triangles: readonly TriangleIndices[],
  cutoff: number
): number[][] {
  const kept = new Set<number>(triangles.keys());
  const getLength = (from: number, to: number) =>
    Math.hypot(projected[to][0] - projected[from][0], projected[to][1] - projected[from][1]);
  for (;;) {
    const directed = new Map<string, number>();
    kept.forEach(triangleIndex => {
      const [a, b, c] = triangles[triangleIndex];
      for (const [from, to] of [
        [a, b],
        [b, c],
        [c, a]
      ])
        directed.set(getEdgeKey(from, to), triangleIndex);
    });
    const boundaryVertices = new Set<number>();
    const boundaryEdges: {from: number; to: number; triangleIndex: number}[] = [];
    directed.forEach((triangleIndex, key) => {
      const [from, to] = key.split(':').map(Number);
      if (!directed.has(getEdgeKey(to, from))) {
        boundaryEdges.push({from, to, triangleIndex});
        boundaryVertices.add(from);
        boundaryVertices.add(to);
      }
    });
    boundaryEdges.sort(
      (first, second) => getLength(second.from, second.to) - getLength(first.from, first.to)
    );
    let removed = false;
    for (const edge of boundaryEdges) {
      if (getLength(edge.from, edge.to) <= cutoff) break;
      const triangle = triangles[edge.triangleIndex];
      const apex = triangle.find(vertex => vertex !== edge.from && vertex !== edge.to)!;
      const boundaryEdgeCount = [
        [triangle[0], triangle[1]],
        [triangle[1], triangle[2]],
        [triangle[2], triangle[0]]
      ].filter(([from, to]) => !directed.has(getEdgeKey(to, from))).length;
      // Removing a triangle whose apex is already on the boundary would pinch the hull, except for an ear.
      if (boundaryVertices.has(apex) && boundaryEdgeCount < 2) continue;
      if (kept.size <= 1) break;
      kept.delete(edge.triangleIndex);
      removed = true;
      break;
    }
    if (!removed) break;
  }
  return getBoundaryLoops([...kept].map(index => triangles[index]));
}

export function getDelaunayEdgeLengths(
  projected: readonly PlanarPoint[],
  triangles: readonly TriangleIndices[]
): {minimum: number; maximum: number} {
  let minimum = Infinity;
  let maximum = 0;
  for (const [a, b, c] of triangles)
    for (const [from, to] of [
      [a, b],
      [b, c],
      [c, a]
    ]) {
      const length = Math.hypot(
        projected[to][0] - projected[from][0],
        projected[to][1] - projected[from][1]
      );
      minimum = Math.min(minimum, length);
      maximum = Math.max(maximum, length);
    }
  return {minimum, maximum};
}

export function runConcaveHull(
  features: CatalogFeatureCollection,
  parameters: {
    ratio: number;
    maximumEdgeLength: number;
    groupProperty: string;
    coordinateSystem: CatalogCoordinateSystem;
  }
): CatalogRunOutput {
  const diagnostics: CatalogDiagnostic[] = [];
  const frame = makeLocalFrame(parameters.coordinateSystem, collectPositions(features));
  const groups = new Map<string, CatalogPosition[]>();
  features.features.forEach(feature => {
    const key = readPropertyText(feature, parameters.groupProperty);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key)!.push(...getFeatureVertices(feature));
  });
  const table = makeCatalogTable(
    'concave-hulls',
    'Concave hulls',
    {
      hull_id: 'string',
      group: 'string',
      point_count: 'int32',
      ring_count: 'int32',
      area: 'float64',
      edge_cutoff: 'float64',
      geometry: 'string'
    },
    'geometry'
  );
  for (const [group, positions] of [...groups.entries()].sort(([first], [second]) =>
    first < second ? -1 : first > second ? 1 : 0
  )) {
    const seen = new Set<string>();
    const unique = positions.filter(position => {
      const key = `${position[0]}:${position[1]}`;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
    const projected = unique.map(position => frame.forward(position));
    const triangles = triangulatePoints(projected);
    if (!triangles.length) {
      addCatalogDiagnostic(
        diagnostics,
        'degenerate-group',
        'warning',
        'Groups with fewer than three non-collinear distinct points produce no hull.'
      );
      continue;
    }
    const {minimum, maximum} = getDelaunayEdgeLengths(projected, triangles);
    const cutoff =
      parameters.maximumEdgeLength > 0
        ? parameters.maximumEdgeLength
        : minimum + parameters.ratio * (maximum - minimum);
    const loops = getConcaveBoundary(projected, triangles, cutoff);
    const outer = loops.filter(loop => getSignedArea(loop.map(index => projected[index])) > 0);
    const holes = loops.filter(loop => getSignedArea(loop.map(index => projected[index])) < 0);
    const polygons = outer.map(loop => [loop.map(index => unique[index])]);
    for (const hole of holes) {
      const probe = projected[hole[0]];
      const owner = outer.findIndex(loop =>
        isPointInPolygonRings(probe[0], probe[1], [loop.map(index => projected[index])])
      );
      if (owner >= 0) polygons[owner].push(hole.map(index => unique[index]).reverse());
    }
    const area =
      outer.reduce((sum, loop) => sum + getSignedArea(loop.map(index => projected[index])), 0) +
      holes.reduce((sum, loop) => sum + getSignedArea(loop.map(index => projected[index])), 0);
    const closed = polygons.map(rings => rings.map(ring => closeRing([...ring])));
    appendCatalogRow(table, {
      hull_id: group || `${table.rowCount}`,
      group,
      point_count: unique.length,
      ring_count: loops.length,
      area,
      edge_cutoff: cutoff,
      geometry:
        closed.length === 1
          ? makeCatalogGeometryText('Polygon', closed[0])
          : makeCatalogGeometryText('MultiPolygon', closed)
    });
  }
  addCatalogDiagnostic(
    diagnostics,
    'ratio-semantics',
    'info',
    'Boundary triangles are trimmed while their outer edge is longer than min + ratio * (max - min) of the Delaunay edge lengths; ratio 1 is the convex hull. A positive maximumEdgeLength overrides the ratio.'
  );
  return {
    tables: [table],
    diagnostics,
    summary: {hullCount: table.rowCount, groupCount: groups.size}
  };
}

// ---------------------------------------------------------------------------------------------
// Polygon triangulation (ear clipping with hole bridges)
// ---------------------------------------------------------------------------------------------

export const MAXIMUM_TRIANGULATION_VERTICES = 1500;

function isProperlyCrossing(
  first: readonly [PlanarPoint, PlanarPoint],
  second: readonly [PlanarPoint, PlanarPoint]
): boolean {
  const hit = intersectSegments(first, second);
  if (!hit) return false;
  const epsilon = 1e-9;
  return (
    hit.firstFraction > epsilon &&
    hit.firstFraction < 1 - epsilon &&
    hit.secondFraction > epsilon &&
    hit.secondFraction < 1 - epsilon
  );
}

/** Triangles of a polygon (shell first, then holes) as point triples; `complete` is false when ear clipping stalls. */
export function triangulatePolygonRings(rings: readonly (readonly PlanarPoint[])[]): {
  triangles: [PlanarPoint, PlanarPoint, PlanarPoint][];
  complete: boolean;
} {
  const shell = [...rings[0]];
  if (getSignedArea(shell) < 0) shell.reverse();
  const holes = rings
    .slice(1)
    .map(ring => (getSignedArea(ring) > 0 ? [...ring].reverse() : [...ring]))
    .sort(
      (first, second) =>
        Math.max(...second.map(point => point[0])) - Math.max(...first.map(point => point[0]))
    );
  let polygon = shell;
  const allRings = rings.map(ring => [...ring]);
  const remainingHoles = [...holes];
  for (const hole of holes) {
    remainingHoles.shift();
    let holeIndex = 0;
    for (let index = 1; index < hole.length; index++)
      if (hole[index][0] > hole[holeIndex][0]) holeIndex = index;
    const holePoint = hole[holeIndex];
    const candidates = polygon
      .map((point, index) => ({
        index,
        distance: Math.hypot(point[0] - holePoint[0], point[1] - holePoint[1])
      }))
      .sort((first, second) => first.distance - second.distance);
    let bridge = -1;
    for (const candidate of candidates) {
      const target = polygon[candidate.index];
      const segment: [PlanarPoint, PlanarPoint] = [holePoint, target];
      const midpointX = (holePoint[0] + target[0]) / 2;
      const midpointY = (holePoint[1] + target[1]) / 2;
      if (!isPointInPolygonRings(midpointX, midpointY, allRings)) continue;
      const blocked = [polygon, hole, ...remainingHoles].some(ring =>
        ring.some((point, index) =>
          isProperlyCrossing(segment, [point, ring[(index + 1) % ring.length]])
        )
      );
      if (!blocked) {
        bridge = candidate.index;
        break;
      }
    }
    if (bridge < 0) return {triangles: [], complete: false};
    const rotatedHole = [...hole.slice(holeIndex), ...hole.slice(0, holeIndex)];
    polygon = [
      ...polygon.slice(0, bridge + 1),
      ...rotatedHole,
      rotatedHole[0],
      polygon[bridge],
      ...polygon.slice(bridge + 1)
    ];
  }
  const triangles: [PlanarPoint, PlanarPoint, PlanarPoint][] = [];
  const working = [...polygon];
  const isConvex = (previous: PlanarPoint, point: PlanarPoint, next: PlanarPoint) =>
    (point[0] - previous[0]) * (next[1] - previous[1]) -
      (point[1] - previous[1]) * (next[0] - previous[0]) >
    1e-14;
  const isInsideTriangle = (
    point: PlanarPoint,
    a: PlanarPoint,
    b: PlanarPoint,
    c: PlanarPoint
  ): boolean => {
    const first = (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0]);
    const second = (c[0] - b[0]) * (point[1] - b[1]) - (c[1] - b[1]) * (point[0] - b[0]);
    const third = (a[0] - c[0]) * (point[1] - c[1]) - (a[1] - c[1]) * (point[0] - c[0]);
    return first >= 0 && second >= 0 && third >= 0;
  };
  let guard = working.length * working.length + 10;
  while (working.length > 3 && guard-- > 0) {
    let clipped = false;
    for (let index = 0; index < working.length; index++) {
      const previous = working[(index + working.length - 1) % working.length];
      const point = working[index];
      const next = working[(index + 1) % working.length];
      if (!isConvex(previous, point, next)) continue;
      const blocked = working.some(
        other =>
          other !== previous &&
          other !== point &&
          other !== next &&
          !(other[0] === previous[0] && other[1] === previous[1]) &&
          !(other[0] === point[0] && other[1] === point[1]) &&
          !(other[0] === next[0] && other[1] === next[1]) &&
          isInsideTriangle(other, previous, point, next)
      );
      if (blocked) continue;
      triangles.push([previous, point, next]);
      working.splice(index, 1);
      clipped = true;
      break;
    }
    if (!clipped) return {triangles, complete: false};
  }
  if (working.length === 3 && isConvex(working[0], working[1], working[2]))
    triangles.push([working[0], working[1], working[2]]);
  return {triangles, complete: true};
}

export function runPolygonTriangulation(
  polygons: CatalogFeatureCollection,
  coordinateSystem: CatalogCoordinateSystem
): CatalogRunOutput {
  const diagnostics: CatalogDiagnostic[] = [];
  const frame: LocalFrame = makeLocalFrame(coordinateSystem, collectPositions(polygons), {
    kind: 'lambert-equal-area'
  });
  const table = makeCatalogTable(
    'polygon-triangles',
    'Polygon triangles',
    CATALOG_POLYGON_TRIANGLE_COLUMNS,
    'geometry'
  );
  let incompleteCount = 0;
  polygons.features.forEach((feature, sourceIndex) => {
    projectCatalogPolygonParts(feature, frame).forEach(({rings, partIndex}) => {
      const vertexCount = rings.reduce((sum, ring) => sum + ring.length, 0);
      if (vertexCount > MAXIMUM_TRIANGULATION_VERTICES) {
        addCatalogDiagnostic(
          diagnostics,
          'part-too-large',
          'warning',
          `Polygon parts over ${MAXIMUM_TRIANGULATION_VERTICES} vertices are skipped.`
        );
        return;
      }
      const {triangles, complete} = triangulatePolygonRings(rings);
      if (!complete) incompleteCount++;
      appendCatalogTriangleRows(table, triangles, feature, sourceIndex, partIndex, (x, y) =>
        frame.inverse(x, y)
      );
    });
  });
  if (incompleteCount)
    addCatalogDiagnostic(
      diagnostics,
      'triangulation-incomplete',
      'warning',
      'Ear clipping stalled on invalid or self-intersecting rings; the triangles found so far are kept.',
      incompleteCount
    );
  return {tables: [table], diagnostics, summary: {triangleCount: table.rowCount}};
}
