// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {KeplerAllocationError} from '../runtime/dataset-registry';
import type {CatalogRunOutput} from './catalog-binding-types';
import {
  addCatalogDiagnostic,
  appendCatalogRow,
  getFeatureIdentity,
  getLineParts,
  getPointPositions,
  getPolygonParts,
  makeCatalogGeometryText,
  makeCatalogTable,
  measureRingSignedArea,
  type CatalogCoordinateSystem,
  type CatalogDiagnostic,
  type CatalogFeatureCollection,
  type CatalogPosition
} from './catalog-geojson';
import {
  closeRing,
  collectPositions,
  getDistanceToSegment,
  getFeatureBounds,
  getSignedArea,
  intersectSegments,
  isPointInRing,
  type Bounds,
  type PlanarPoint,
  toPlanar
} from './catalog-platform3-a-shared';
import {makeLocalFrame} from './catalog-local-frame';

export const MAXIMUM_JOIN_PAIRS = 200000;

// ---------------------------------------------------------------------------------------------
// Bounds candidates
// ---------------------------------------------------------------------------------------------

export function runBoundsCandidates(
  left: CatalogFeatureCollection,
  right: CatalogFeatureCollection,
  parameters: {distance: number; coordinateSystem: CatalogCoordinateSystem}
): CatalogRunOutput {
  const frame = makeLocalFrame(parameters.coordinateSystem, [
    ...collectPositions(left),
    ...collectPositions(right)
  ]);
  const getProjectedBounds = (collection: CatalogFeatureCollection, expand: number) =>
    collection.features.map((feature): Bounds | null => {
      const raw = getFeatureBounds(feature);
      if (!raw) return null;
      const [minimumX, minimumY] = frame.forward([raw[0], raw[1]]);
      const [maximumX, maximumY] = frame.forward([raw[2], raw[3]]);
      return [minimumX - expand, minimumY - expand, maximumX + expand, maximumY + expand];
    });
  const leftBounds = getProjectedBounds(left, parameters.distance);
  const rightBounds = getProjectedBounds(right, 0);
  const rightOrder = rightBounds
    .map((bounds, index) => ({bounds, index}))
    .filter((item): item is {bounds: Bounds; index: number} => item.bounds !== null)
    .sort((first, second) => first.bounds[0] - second.bounds[0]);
  const table = makeCatalogTable('join-candidates', 'Join candidate pairs', {
    left_index: 'int32',
    left_id: 'string',
    right_index: 'int32',
    right_id: 'string'
  });
  const diagnostics: CatalogDiagnostic[] = [];
  let emptyCount = 0;
  leftBounds.forEach((bounds, leftIndex) => {
    if (!bounds) {
      emptyCount++;
      return;
    }
    const pairs: number[] = [];
    for (const item of rightOrder) {
      if (item.bounds[0] > bounds[2]) break;
      if (item.bounds[2] < bounds[0] || item.bounds[3] < bounds[1] || item.bounds[1] > bounds[3])
        continue;
      pairs.push(item.index);
    }
    pairs.sort((first, second) => first - second);
    for (const rightIndex of pairs) {
      if (table.rowCount >= MAXIMUM_JOIN_PAIRS)
        throw new KeplerAllocationError('capacity', {message: 'Candidate pair capacity exceeded'});
      appendCatalogRow(table, {
        left_index: leftIndex,
        left_id: getFeatureIdentity(left.features[leftIndex]),
        right_index: rightIndex,
        right_id: getFeatureIdentity(right.features[rightIndex])
      });
    }
  });
  if (emptyCount)
    addCatalogDiagnostic(
      diagnostics,
      'empty-geometry',
      'warning',
      'Left features without geometry have no candidates.',
      emptyCount
    );
  addCatalogDiagnostic(
    diagnostics,
    'bounds-only',
    'info',
    'Candidates only share overlapping bounding boxes (left boxes grown by the distance); refine with the predicate join.'
  );
  return {tables: [table], diagnostics, summary: {pairCount: table.rowCount}};
}

// ---------------------------------------------------------------------------------------------
// Exact predicates against polygons
// ---------------------------------------------------------------------------------------------

type PreparedGeometry = {
  points: PlanarPoint[];
  /** Open or closed vertex chains whose segments take part in tests. */
  chains: PlanarPoint[][];
  polygons: PlanarPoint[][][];
  bounds: Bounds | null;
};

function prepareGeometry(feature: CatalogFeatureCollection['features'][number]): PreparedGeometry {
  const points = getPointPositions(feature.geometry).map(toPlanar);
  const chains = getLineParts(feature.geometry)
    .filter(part => part.length >= 2)
    .map(part => part.map(toPlanar));
  const polygons = getPolygonParts(feature.geometry)
    .map(polygon => polygon.filter(ring => ring.length >= 4).map(ring => ring.map(toPlanar)))
    .filter(polygon => polygon.length);
  for (const polygon of polygons) for (const ring of polygon) chains.push(ring);
  return {points, chains, polygons, bounds: getFeatureBounds(feature)};
}

function isOnBoundary(x: number, y: number, polygon: readonly PlanarPoint[][]): boolean {
  for (const ring of polygon)
    for (let index = 1; index < ring.length; index++)
      if (
        getDistanceToSegment(
          x,
          y,
          ring[index - 1][0],
          ring[index - 1][1],
          ring[index][0],
          ring[index][1]
        ).distance <=
        1e-12 * Math.max(1, Math.abs(x), Math.abs(y))
      )
        return true;
  return false;
}

function isInsidePolygon(x: number, y: number, polygon: readonly PlanarPoint[][]): boolean {
  let inside = false;
  for (const ring of polygon) if (isPointInRing(x, y, ring)) inside = !inside;
  return inside;
}

function isInsideOrOn(x: number, y: number, polygons: readonly PlanarPoint[][][]): boolean {
  return polygons.some(polygon => isInsidePolygon(x, y, polygon) || isOnBoundary(x, y, polygon));
}

function isStrictlyInside(x: number, y: number, polygons: readonly PlanarPoint[][][]): boolean {
  return polygons.some(polygon => isInsidePolygon(x, y, polygon) && !isOnBoundary(x, y, polygon));
}

function doesIntersect(geometry: PreparedGeometry, target: PreparedGeometry): boolean {
  for (const [x, y] of geometry.points) if (isInsideOrOn(x, y, target.polygons)) return true;
  for (const chain of geometry.chains) {
    for (const [x, y] of chain) if (isInsideOrOn(x, y, target.polygons)) return true;
    for (let index = 1; index < chain.length; index++)
      for (const polygon of target.polygons)
        for (const ring of polygon)
          for (let ringIndex = 1; ringIndex < ring.length; ringIndex++)
            if (
              intersectSegments(
                [chain[index - 1], chain[index]],
                [ring[ringIndex - 1], ring[ringIndex]]
              )
            )
              return true;
  }
  for (const polygon of geometry.polygons)
    for (const ring of target.polygons.flat())
      for (const [x, y] of ring) if (isInsidePolygon(x, y, polygon)) return true;
  return false;
}

/** True when every part of the geometry lies inside or on the target polygons (boundary contact allowed). */
function isWithin(geometry: PreparedGeometry, target: PreparedGeometry): boolean {
  if (!target.polygons.length) return false;
  if (!geometry.points.length && !geometry.chains.length) return false;
  for (const [x, y] of geometry.points) if (!isInsideOrOn(x, y, target.polygons)) return false;
  for (const chain of geometry.chains) {
    for (let index = 1; index < chain.length; index++) {
      const start = chain[index - 1];
      const end = chain[index];
      if (!isInsideOrOn(start[0], start[1], target.polygons)) return false;
      const fractions = [0, 1];
      for (const polygon of target.polygons)
        for (const ring of polygon)
          for (let ringIndex = 1; ringIndex < ring.length; ringIndex++) {
            const hit = intersectSegments([start, end], [ring[ringIndex - 1], ring[ringIndex]]);
            if (hit) fractions.push(hit.firstFraction);
          }
      fractions.sort((first, second) => first - second);
      for (let fractionIndex = 1; fractionIndex < fractions.length; fractionIndex++) {
        const middle = (fractions[fractionIndex - 1] + fractions[fractionIndex]) / 2;
        if (
          !isInsideOrOn(
            start[0] + (end[0] - start[0]) * middle,
            start[1] + (end[1] - start[1]) * middle,
            target.polygons
          )
        )
          return false;
      }
    }
  }
  // A target hole fully inside a polygon geometry would be covered by it.
  for (const polygon of geometry.polygons)
    for (const targetPolygon of target.polygons)
      for (const hole of targetPolygon.slice(1))
        if (hole.some(([x, y]) => isStrictlyInside(x, y, [polygon]))) return false;
  return true;
}

export type SpatialPredicate = 'intersects' | 'within' | 'contains';

export function runSpatialPredicateJoin(
  left: CatalogFeatureCollection,
  right: CatalogFeatureCollection,
  predicate: SpatialPredicate
): CatalogRunOutput {
  const preparedRight = right.features.map(prepareGeometry);
  const table = makeCatalogTable('spatial-join-pairs', 'Spatial join pairs', {
    join_id: 'string',
    left_index: 'int32',
    left_id: 'string',
    right_index: 'int32',
    right_id: 'string',
    predicate: 'string'
  });
  const diagnostics: CatalogDiagnostic[] = [];
  left.features.forEach((leftFeature, leftIndex) => {
    const geometry = prepareGeometry(leftFeature);
    if (!geometry.bounds) return;
    preparedRight.forEach((target, rightIndex) => {
      if (!target.bounds) return;
      if (
        geometry.bounds![2] < target.bounds[0] ||
        geometry.bounds![0] > target.bounds[2] ||
        geometry.bounds![3] < target.bounds[1] ||
        geometry.bounds![1] > target.bounds[3]
      )
        return;
      const matches =
        predicate === 'intersects'
          ? doesIntersect(geometry, target)
          : predicate === 'within'
            ? isWithin(geometry, target)
            : geometry.polygons.length > 0 && isWithin(target, geometry);
      if (!matches) return;
      if (table.rowCount >= MAXIMUM_JOIN_PAIRS)
        throw new KeplerAllocationError('capacity', {message: 'Join pair capacity exceeded'});
      appendCatalogRow(table, {
        join_id: `${leftIndex}:${rightIndex}`,
        left_index: leftIndex,
        left_id: getFeatureIdentity(leftFeature),
        right_index: rightIndex,
        right_id: getFeatureIdentity(right.features[rightIndex]),
        predicate
      });
    });
  });
  addCatalogDiagnostic(
    diagnostics,
    'planar-predicates',
    'info',
    'Predicates use planar tests on the raw coordinates, boundary contact counts as inside; within means every part of the left geometry lies in the right polygons, contains needs a polygon on the left.'
  );
  return {tables: [table], diagnostics, summary: {pairCount: table.rowCount}};
}

// ---------------------------------------------------------------------------------------------
// Coverage validity (overlaps)
// ---------------------------------------------------------------------------------------------

export function runCoverageValidity(polygons: CatalogFeatureCollection): CatalogRunOutput {
  const prepared = polygons.features.map(prepareGeometry);
  const pairTable = makeCatalogTable('coverage-overlaps', 'Coverage overlaps', {
    overlap_id: 'string',
    first_index: 'int32',
    first_id: 'string',
    second_index: 'int32',
    second_id: 'string',
    kind: 'string'
  });
  const overlapCounts = new Array<number>(polygons.features.length).fill(0);
  const getProbes = (geometry: PreparedGeometry): PlanarPoint[] => {
    const probes: PlanarPoint[] = [];
    for (const polygon of geometry.polygons) {
      const shell = polygon[0];
      for (let index = 0; index < shell.length - 1; index++) {
        probes.push(shell[index]);
        probes.push([
          (shell[index][0] + shell[index + 1][0]) / 2,
          (shell[index][1] + shell[index + 1][1]) / 2
        ]);
      }
      const open = shell.slice(0, -1);
      const centroid: PlanarPoint = [
        open.reduce((sum, point) => sum + point[0], 0) / open.length,
        open.reduce((sum, point) => sum + point[1], 0) / open.length
      ];
      if (isInsidePolygon(centroid[0], centroid[1], polygon)) probes.push(centroid);
    }
    return probes;
  };
  const probes = prepared.map(getProbes);
  for (let first = 0; first < prepared.length; first++)
    for (let second = first + 1; second < prepared.length; second++) {
      const a = prepared[first];
      const b = prepared[second];
      if (!a.bounds || !b.bounds) continue;
      if (
        a.bounds[2] < b.bounds[0] ||
        a.bounds[0] > b.bounds[2] ||
        a.bounds[3] < b.bounds[1] ||
        a.bounds[1] > b.bounds[3]
      )
        continue;
      let kind: string | null = null;
      for (const polygonA of a.polygons) {
        for (const ringA of polygonA)
          for (let indexA = 1; indexA < ringA.length && !kind; indexA++)
            for (const polygonB of b.polygons)
              for (const ringB of polygonB)
                for (let indexB = 1; indexB < ringB.length && !kind; indexB++) {
                  const hit = intersectSegments(
                    [ringA[indexA - 1], ringA[indexA]],
                    [ringB[indexB - 1], ringB[indexB]]
                  );
                  if (
                    hit &&
                    hit.firstFraction > 1e-9 &&
                    hit.firstFraction < 1 - 1e-9 &&
                    hit.secondFraction > 1e-9 &&
                    hit.secondFraction < 1 - 1e-9
                  )
                    kind = 'crossing';
                }
      }
      if (
        !kind &&
        (probes[first].some(([x, y]) => isStrictlyInside(x, y, b.polygons)) ||
          probes[second].some(([x, y]) => isStrictlyInside(x, y, a.polygons)))
      )
        kind = 'contained';
      if (!kind) continue;
      if (pairTable.rowCount >= MAXIMUM_JOIN_PAIRS)
        throw new KeplerAllocationError('capacity', {message: 'Overlap pair capacity exceeded'});
      overlapCounts[first]++;
      overlapCounts[second]++;
      appendCatalogRow(pairTable, {
        overlap_id: `${first}:${second}`,
        first_index: first,
        first_id: getFeatureIdentity(polygons.features[first]),
        second_index: second,
        second_id: getFeatureIdentity(polygons.features[second]),
        kind
      });
    }
  const featureTable = makeCatalogTable(
    'coverage-features',
    'Coverage feature status',
    {
      source_index: 'int32',
      source_id: 'string',
      overlap_count: 'int32',
      status: 'string',
      geometry: 'string'
    },
    'geometry'
  );
  polygons.features.forEach((feature, sourceIndex) =>
    appendCatalogRow(featureTable, {
      source_index: sourceIndex,
      source_id: getFeatureIdentity(feature),
      overlap_count: overlapCounts[sourceIndex],
      status: overlapCounts[sourceIndex] ? 'overlaps' : 'valid',
      geometry: JSON.stringify(feature.geometry)
    })
  );
  return {
    tables: [pairTable, featureTable],
    diagnostics: [
      {
        code: 'gaps-not-detected',
        severity: 'info',
        count: 1,
        message:
          'Only interior overlaps are reported (proper edge crossings, or a boundary, edge midpoint or interior probe strictly inside another polygon); gaps between polygons are not detected, and shared-edge mismatches are not reported.'
      }
    ],
    summary: {
      overlapCount: pairTable.rowCount,
      overlappingFeatureCount: overlapCounts.filter(count => count > 0).length
    }
  };
}

// ---------------------------------------------------------------------------------------------
// Polygonize
// ---------------------------------------------------------------------------------------------

export const MAXIMUM_POLYGONIZE_SEGMENTS = 20000;

export function runPolygonize(
  lines: CatalogFeatureCollection,
  parameters: {
    snapTolerance: number;
    nodeCrossings: boolean;
    coordinateSystem: CatalogCoordinateSystem;
  }
): CatalogRunOutput {
  const diagnostics: CatalogDiagnostic[] = [];
  let segments: [PlanarPoint, PlanarPoint][] = [];
  for (const feature of lines.features)
    for (const part of getLineParts(feature.geometry))
      for (let index = 1; index < part.length; index++)
        if (part[index - 1][0] !== part[index][0] || part[index - 1][1] !== part[index][1])
          segments.push([toPlanar(part[index - 1]), toPlanar(part[index])]);
  if (parameters.nodeCrossings) {
    const cuts = segments.map(() => [0, 1]);
    for (let first = 0; first < segments.length; first++)
      for (let second = first + 1; second < segments.length; second++) {
        const hit = intersectSegments(segments[first], segments[second]);
        if (hit) {
          cuts[first].push(hit.firstFraction);
          cuts[second].push(hit.secondFraction);
        }
      }
    const noded: [PlanarPoint, PlanarPoint][] = [];
    segments.forEach(([start, end], index) => {
      const fractions = cuts[index].sort((first, second) => first - second);
      for (let cut = 1; cut < fractions.length; cut++)
        if (fractions[cut] - fractions[cut - 1] > 1e-12)
          noded.push([
            [
              start[0] + (end[0] - start[0]) * fractions[cut - 1],
              start[1] + (end[1] - start[1]) * fractions[cut - 1]
            ],
            [
              start[0] + (end[0] - start[0]) * fractions[cut],
              start[1] + (end[1] - start[1]) * fractions[cut]
            ]
          ]);
    });
    segments = noded;
  }
  // Polygonize segment capacity exceeded.
  if (segments.length > MAXIMUM_POLYGONIZE_SEGMENTS)
    throw new KeplerAllocationError('capacity', {message: 'Segment capacity exceeded'});
  const vertexIndex = new Map<string, number>();
  const vertices: PlanarPoint[] = [];
  const getVertex = (point: PlanarPoint): number => {
    const key =
      parameters.snapTolerance > 0
        ? `${Math.round(point[0] / parameters.snapTolerance)}:${Math.round(point[1] / parameters.snapTolerance)}`
        : `${point[0]}:${point[1]}`;
    let index = vertexIndex.get(key);
    if (index === undefined) {
      index = vertices.length;
      vertexIndex.set(key, index);
      vertices.push(point);
    }
    return index;
  };
  const edgeKeys = new Set<string>();
  const neighbors: Set<number>[] = [];
  const addNeighbor = (from: number, to: number) => {
    while (neighbors.length <= Math.max(from, to)) neighbors.push(new Set());
    neighbors[from].add(to);
    neighbors[to].add(from);
  };
  for (const [start, end] of segments) {
    const from = getVertex(start);
    const to = getVertex(end);
    const key = from < to ? `${from}:${to}` : `${to}:${from}`;
    if (from === to || edgeKeys.has(key)) continue;
    edgeKeys.add(key);
    addNeighbor(from, to);
  }
  // Prune dangles.
  const queue = neighbors
    .map((set, index) => (set.size < 2 ? index : -1))
    .filter(index => index >= 0);
  let danglingEdgeCount = 0;
  while (queue.length) {
    const vertex = queue.pop()!;
    for (const other of [...neighbors[vertex]]) {
      neighbors[vertex].delete(other);
      neighbors[other].delete(vertex);
      danglingEdgeCount++;
      if (neighbors[other].size < 2) queue.push(other);
    }
  }
  const sorted = neighbors.map((set, vertex) =>
    [...set].sort(
      (first, second) =>
        Math.atan2(
          vertices[first][1] - vertices[vertex][1],
          vertices[first][0] - vertices[vertex][0]
        ) -
        Math.atan2(
          vertices[second][1] - vertices[vertex][1],
          vertices[second][0] - vertices[vertex][0]
        )
    )
  );
  const visited = new Set<string>();
  const faces: {ring: number[]; area: number}[] = [];
  sorted.forEach((targets, from) =>
    targets.forEach(to => {
      if (visited.has(`${from}:${to}`)) return;
      const ring: number[] = [];
      let currentFrom = from;
      let currentTo = to;
      while (!visited.has(`${currentFrom}:${currentTo}`)) {
        visited.add(`${currentFrom}:${currentTo}`);
        ring.push(currentFrom);
        const around = sorted[currentTo];
        const position = around.indexOf(currentFrom);
        const next = around[(position - 1 + around.length) % around.length];
        currentFrom = currentTo;
        currentTo = next;
      }
      faces.push({ring, area: getSignedArea(ring.map(vertex => vertices[vertex]))});
    })
  );
  const interior = faces.filter(face => face.area > 0);
  const outerBoundaries = faces.filter(face => face.area < 0);
  const holesByFace = new Map<number, number[][]>();
  for (const boundary of outerBoundaries) {
    const probe = vertices[boundary.ring[0]];
    let owner = -1;
    interior.forEach((face, faceIndex) => {
      if (face.ring.includes(boundary.ring[0])) return;
      if (
        isPointInRing(
          probe[0],
          probe[1],
          face.ring.map(vertex => vertices[vertex])
        ) &&
        (owner < 0 || face.area < interior[owner].area)
      )
        owner = faceIndex;
    });
    if (owner >= 0) {
      if (!holesByFace.has(owner)) holesByFace.set(owner, []);
      holesByFace.get(owner)!.push(boundary.ring);
    }
  }
  const table = makeCatalogTable(
    'polygonized-faces',
    'Polygonized faces',
    {
      face_id: 'string',
      ring_count: 'int32',
      vertex_count: 'int32',
      area: 'float64',
      geometry: 'string'
    },
    'geometry'
  );
  interior.forEach((face, faceIndex) => {
    const rings = [face.ring, ...(holesByFace.get(faceIndex) ?? [])].map(ring =>
      closeRing(ring.map((vertex): CatalogPosition => vertices[vertex]))
    );
    appendCatalogRow(table, {
      face_id: `${faceIndex}`,
      ring_count: rings.length,
      vertex_count: face.ring.length,
      area:
        Math.abs(measureRingSignedArea(rings[0], parameters.coordinateSystem)) -
        rings
          .slice(1)
          .reduce(
            (sum, ring) => sum + Math.abs(measureRingSignedArea(ring, parameters.coordinateSystem)),
            0
          ),
      geometry: makeCatalogGeometryText('Polygon', rings)
    });
  });
  if (danglingEdgeCount)
    addCatalogDiagnostic(
      diagnostics,
      'dangles-removed',
      'info',
      'Edges that cannot bound a face (dangles and cut edges to degree-one vertices) were pruned.',
      danglingEdgeCount
    );
  addCatalogDiagnostic(
    diagnostics,
    'planar-faces',
    'info',
    'Faces are traced on the raw planar coordinates; lines must meet at shared vertices (or set nodeCrossings), and islands are attached as holes of the smallest enclosing face.'
  );
  return {
    tables: [table],
    diagnostics,
    summary: {faceCount: table.rowCount, segmentCount: segments.length}
  };
}
