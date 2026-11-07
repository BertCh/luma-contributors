// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUSphericalPolygonContains} from '@luma-contributors/graph/gpu-spatial-analysis';
import {toRadians, vec3} from '@math.gl/core';
import type {CatalogOperationBinding, CatalogRunOutput} from './catalog-binding-types';
import {
  addCatalogDiagnostic,
  appendCatalogRow,
  getPointPositions,
  getPolygonParts,
  makeCatalogGeometryText,
  makeCatalogTable,
  type CatalogDiagnostic,
  type CatalogFeatureCollection
} from './catalog-geojson';
import {CatalogGpuUnsupportedError, withCatalogGpuSession} from './catalog-gpu-session';
import {
  PLATFORM3_GPU_OR_CPU_CLEANUP,
  makeInputSpec,
  makeOutputSpec,
  type Values
} from './catalog-platform3-b-common';

const MAXIMUM_PAIRS = 500000;
const TOLERANCE = 1e-6;

export type Vector3 = readonly [number, number, number];

export function cross(a: Vector3, b: Vector3): Vector3 {
  return vec3.cross([0, 0, 0], a, b) as unknown as Vector3;
}

export function dot(a: Vector3, b: Vector3): number {
  return vec3.dot(a, b);
}

/** Unit vector. Unlike `vec3.normalize`, a zero vector yields NaN so degenerate arcs stay invalid. */
export function normalize(a: Vector3): Vector3 {
  const length = vec3.length(a);
  return [a[0] / length, a[1] / length, a[2] / length];
}

function negate(a: Vector3): Vector3 {
  return vec3.negate([0, 0, 0], a) as unknown as Vector3;
}

/** Unit vector of a longitude/latitude, rounded to float32 like the GPU input. */
export function getUnitVector(longitude: number, latitude: number): Vector3 {
  const lng = toRadians(longitude);
  const lat = toRadians(latitude);
  return [
    Math.fround(Math.cos(lat) * Math.cos(lng)),
    Math.fround(Math.cos(lat) * Math.sin(lng)),
    Math.fround(Math.sin(lat))
  ];
}

export function isOnArc(first: Vector3, second: Vector3, point: Vector3, margin: number): boolean {
  const normal = normalize(cross(first, second));
  return (
    Math.abs(dot(normal, point)) <= margin &&
    dot(cross(first, point), normal) >= -margin &&
    dot(cross(point, second), normal) >= -margin
  );
}

/** Ring validity independent of the query: the GPU kernel reports status 1 for these rings. */
export function isRingValid(ring: readonly Vector3[]): boolean {
  if (ring.length < 3) return false;
  const sum = ring.reduce<Vector3>(
    (total, vertex) => [total[0] + vertex[0], total[1] + vertex[1], total[2] + vertex[2]],
    [0, 0, 0]
  );
  if (Math.hypot(...sum) <= TOLERANCE) return false;
  const center = normalize(sum);
  for (let edge = 0; edge < ring.length; edge++) {
    const next = (edge + 1) % ring.length;
    const first = ring[edge];
    const second = ring[next];
    const normal = cross(first, second);
    if (Math.hypot(...normal) <= TOLERANCE || dot(center, first) <= TOLERANCE) return false;
    for (let other = edge + 1; other < ring.length; other++) {
      const otherNext = (other + 1) % ring.length;
      if (other === next || otherNext === edge) continue;
      const third = ring[other];
      const fourth = ring[otherNext];
      const crossing = cross(normal, cross(third, fourth));
      if (Math.hypot(...crossing) > TOLERANCE * TOLERANCE) {
        const intersection = normalize(crossing);
        const opposite = negate(intersection);
        if (
          (isOnArc(first, second, intersection, TOLERANCE) &&
            isOnArc(third, fourth, intersection, TOLERANCE)) ||
          (isOnArc(first, second, opposite, TOLERANCE) &&
            isOnArc(third, fourth, opposite, TOLERANCE))
        )
          return false;
      } else if (
        isOnArc(first, second, third, TOLERANCE) ||
        isOnArc(first, second, fourth, TOLERANCE)
      )
        return false;
    }
  }
  return true;
}

type SphericalPolygon = {
  rings: Vector3[][];
  featureIndex: number;
  isValid: boolean;
};

type SphericalPoint = {vector: Vector3; featureIndex: number; coordinates: [number, number]};

/** Containment of a point in one valid polygon (first ring outer, later rings holes). */
function testContainment(
  polygon: SphericalPolygon,
  point: Vector3
): {contains: boolean; onBoundary: boolean} {
  let inside = false;
  let onBoundary = false;
  polygon.rings.forEach((ring, ringIndex) => {
    let winding = 0;
    for (let edge = 0; edge < ring.length; edge++) {
      const first = ring[edge];
      const second = ring[(edge + 1) % ring.length];
      onBoundary = onBoundary || isOnArc(first, second, point, TOLERANCE);
      const normal = cross(first, second);
      const numerator = dot(point, normal);
      const denominator = dot(first, second) - dot(point, first) * dot(point, second);
      if (Math.abs(numerator) + Math.abs(denominator) > 1e-12)
        winding += Math.atan2(numerator, denominator);
    }
    const ringInside = winding > Math.PI;
    inside = ringIndex === 0 ? ringInside : inside && !ringInside;
  });
  return {contains: inside, onBoundary};
}

function readSphericalInputs(points: CatalogFeatureCollection, polygons: CatalogFeatureCollection) {
  const queries: SphericalPoint[] = [];
  let droppedCount = 0;
  points.features.forEach((feature, featureIndex) => {
    const positions = getPointPositions(feature.geometry);
    if (!positions.length) droppedCount++;
    for (const [longitude, latitude] of positions)
      if (Number.isFinite(longitude) && Number.isFinite(latitude) && Math.abs(latitude) <= 90)
        queries.push({
          vector: getUnitVector(longitude, latitude),
          featureIndex,
          coordinates: [longitude, latitude]
        });
      else droppedCount++;
  });
  const shapes: SphericalPolygon[] = [];
  polygons.features.forEach((feature, featureIndex) => {
    for (const polygon of getPolygonParts(feature.geometry)) {
      const rings = polygon.map(ring => {
        const open = ring.slice(
          0,
          ring.length > 1 &&
            ring[0][0] === ring[ring.length - 1][0] &&
            ring[0][1] === ring[ring.length - 1][1]
            ? -1
            : undefined
        );
        return open.map(([longitude, latitude]) => getUnitVector(longitude, latitude));
      });
      shapes.push({rings, featureIndex, isValid: rings.every(isRingValid)});
    }
  });
  // Every point is tested against every polygon part.
  if (queries.length * shapes.length > MAXIMUM_PAIRS)
    throw new RangeError(`More than ${MAXIMUM_PAIRS} point-polygon pairs`);
  return {queries, shapes, droppedCount};
}

type PairResult = {contains: Uint8Array; status: Uint8Array};

function buildContainmentOutput(
  queries: readonly SphericalPoint[],
  shapes: readonly SphericalPolygon[],
  result: PairResult,
  droppedCount: number,
  dispatchedContributors?: readonly string[]
): CatalogRunOutput {
  const diagnostics: CatalogDiagnostic[] = [];
  if (droppedCount)
    addCatalogDiagnostic(
      diagnostics,
      'feature-dropped',
      'warning',
      'Features without a valid longitude/latitude point position are skipped.',
      droppedCount
    );
  const invalidCount = shapes.filter(shape => !shape.isValid).length;
  if (invalidCount)
    addCatalogDiagnostic(
      diagnostics,
      'invalid-spherical-polygon',
      'warning',
      'Polygons that are self-crossing, degenerate or wider than a hemisphere contain nothing.',
      invalidCount
    );
  const table = makeCatalogTable(
    'spherical-containment',
    'Spherical point in polygon',
    {
      point_feature_index: 'int32',
      polygon_feature_index: 'int32',
      on_boundary: 'boolean',
      geometry: 'string'
    },
    'geometry'
  );
  queries.forEach((query, queryIndex) =>
    shapes.forEach((shape, shapeIndex) => {
      const pair = queryIndex * shapes.length + shapeIndex;
      if (!result.contains[pair]) return;
      appendCatalogRow(table, {
        point_feature_index: query.featureIndex,
        polygon_feature_index: shape.featureIndex,
        on_boundary: result.status[pair] === 2,
        geometry: makeCatalogGeometryText('Point', query.coordinates)
      });
    })
  );
  return {
    tables: [table],
    diagnostics,
    summary: {
      pointCount: queries.length,
      polygonCount: shapes.length,
      containedCount: table.rowCount
    },
    ...(dispatchedContributors ? {dispatchedContributors} : {})
  };
}

function computeContainmentCpu(
  queries: readonly SphericalPoint[],
  shapes: readonly SphericalPolygon[],
  values: Values
): PairResult {
  const pairs = queries.length * shapes.length;
  const result: PairResult = {contains: new Uint8Array(pairs), status: new Uint8Array(pairs)};
  const isClosed = values['boundary'] === 'closed';
  queries.forEach((query, queryIndex) =>
    shapes.forEach((shape, shapeIndex) => {
      const pair = queryIndex * shapes.length + shapeIndex;
      if (!shape.isValid) {
        result.status[pair] = 1;
        return;
      }
      const {contains, onBoundary} = testContainment(shape, query.vector);
      if (onBoundary) {
        result.status[pair] = 2;
        result.contains[pair] = isClosed ? 1 : 0;
      } else result.contains[pair] = contains ? 1 : 0;
    })
  );
  return result;
}

const SPHERICAL_BINDING: CatalogOperationBinding = {
  id: 'spherical-point-in-polygon',
  title: 'Spherical point in polygon',
  domain: 'geometry',
  summary:
    'Tests points against polygons whose edges are great-circle arcs on the sphere (no planar longitude/latitude assumption, antimeridian and polar polygons work).',
  capabilityIds: ['coverage-global-coordinates'],
  contributorNames: ['GPUSphericalPolygonContains'],
  gpuContributorNames: ['GPUSphericalPolygonContains'],
  engine: 'gpu-or-cpu',
  inputs: [
    makeInputSpec('points', 'Points', ['point']),
    makeInputSpec('polygons', 'Polygons', ['polygon'], 2000, 100000)
  ],
  parameters: [
    {
      name: 'boundary',
      label: 'Points on an edge',
      kind: 'enum',
      options: ['closed', 'open'],
      defaultValue: 'closed'
    }
  ],
  outputs: [
    makeOutputSpec(
      'spherical-containment',
      'Spherical point in polygon',
      'point',
      ['point_feature_index', 'polygon_feature_index'],
      MAXIMUM_PAIRS
    )
  ],
  cleanup: PLATFORM3_GPU_OR_CPU_CLEANUP,
  declaredBehavior: [
    'Coordinates are WGS84 degrees treated as a unit sphere; polygon edges are the shorter great-circle arc between consecutive vertices, the first ring is the outer ring (its smaller region) and later rings are holes.',
    'Polygons that cross themselves, touch non-adjacent edges, have a zero-length edge or span a full hemisphere contain nothing and are counted as invalid; every point is tested against every polygon part (at most 500,000 pairs).',
    'A point within 1e-6 of an edge is on_boundary and is contained for boundary=closed only. Positions are float32, so points closer than about 1e-6 radians to an edge can differ between engines.'
  ],
  limitation:
    'Containment only: the spherical overlay, convex-intersection and arc-overlay contributors are not bound, and complement (outside-of-polygon) mode and hemisphere-centered large rings are not exposed.',
  run: (inputs, values) => {
    const {queries, shapes, droppedCount} = readSphericalInputs(
      inputs['points'],
      inputs['polygons']
    );
    return buildContainmentOutput(
      queries,
      shapes,
      computeContainmentCpu(queries, shapes, values),
      droppedCount
    );
  },
  runGpu: async (context, inputs, values) => {
    const {queries, shapes, droppedCount} = readSphericalInputs(
      inputs['points'],
      inputs['polygons']
    );
    const pairs = queries.length * shapes.length;
    const usable = shapes.filter(shape => shape.isValid);
    if (!pairs || !usable.length) throw new CatalogGpuUnsupportedError('Nothing to test');
    const positions: number[] = [];
    const ringOffsets: number[] = [0];
    const polygonOffsets: number[] = [0];
    let ringCount = 0;
    for (const shape of shapes) {
      for (const ring of shape.rings) {
        for (const vertex of ring) positions.push(...vertex);
        ringOffsets.push(positions.length / 3);
        ringCount++;
      }
      polygonOffsets.push(ringCount);
    }
    const queryVectors: number[] = [];
    const polygonIndices: number[] = [];
    for (const query of queries)
      shapes.forEach((_, shapeIndex) => {
        queryVectors.push(...query.vector);
        polygonIndices.push(shapeIndex);
      });
    const result = await withCatalogGpuSession(
      context.device,
      'catalog-spherical-contains',
      async session => {
        const contains = session.output('contains', 'uint32', pairs);
        const status = session.output('status', 'uint32', pairs);
        session.graph.add(
          new GPUSphericalPolygonContains({
            id: 'catalog-spherical-contains',
            boundary: values['boundary'] as 'closed' | 'open',
            positions: session.input('positions', Float32Array.from(positions), 'float32x3'),
            ringOffsets: session.input('ring-offsets', Uint32Array.from(ringOffsets), 'uint32'),
            polygonOffsets: session.input(
              'polygon-offsets',
              Uint32Array.from(polygonOffsets),
              'uint32'
            ),
            queries: session.input('queries', Float32Array.from(queryVectors), 'float32x3'),
            polygonIndices: session.input(
              'polygon-indices',
              Uint32Array.from(polygonIndices),
              'uint32'
            ),
            output: {contains: contains.view, status: status.view}
          })
        );
        session.run();
        return {
          contains: Uint8Array.from(await session.readUint32(contains)),
          status: Uint8Array.from(await session.readUint32(status))
        };
      }
    );
    return buildContainmentOutput(queries, shapes, result, droppedCount, [
      'GPUSphericalPolygonContains'
    ]);
  }
};

export const PLATFORM4_SPHERICAL_BINDINGS: readonly CatalogOperationBinding[] = [SPHERICAL_BINDING];
