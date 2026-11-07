// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUDelaunayTessellation,
  GPUPolygonTriangulation,
  getDelaunayMaximumTriangleCount,
  getPolygonTriangulationIndexCount
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {CatalogGpuContext, CatalogRunOutput} from './catalog-binding-types';
import {
  addCatalogDiagnostic,
  appendCatalogRow,
  makeCatalogGeometryText,
  makeCatalogTable,
  type CatalogCoordinateSystem,
  type CatalogDiagnostic,
  type CatalogFeatureCollection
} from './catalog-geojson';
import {CatalogGpuUnsupportedError, withCatalogGpuSession} from './catalog-gpu-session';
import {
  closeRing,
  collectPositions,
  getSignedArea,
  getSourceColumns,
  projectPolygonParts,
  type PlanarPoint
} from './catalog-platform3-a-shared';
import {makeLocalFrame} from './catalog-local-frame';
import {MAXIMUM_TRIANGULATION_VERTICES, collectSites} from './catalog-platform3-a-tessellation';

/**
 * GPUDelaunayTessellation on the metric local frame of the CPU reference, shifted to the data
 * center. Sites are deduplicated on the CPU exactly like the reference. The GPU triangulation is
 * exact (symbolic super vertices), so it can keep hull triangles that the CPU reference's large
 * bounding triangle clips; interior triangles agree except for cocircular ties. Triangles come in
 * the contributor's canonical order (smallest site first) instead of insertion order.
 */
export async function runGpuDelaunayTriangulation(
  context: CatalogGpuContext,
  points: CatalogFeatureCollection,
  coordinateSystem: CatalogCoordinateSystem
): Promise<CatalogRunOutput> {
  const {sites, duplicateCount} = collectSites(points, false);
  if (sites.length < 3) throw new CatalogGpuUnsupportedError('Fewer than three distinct points');
  const frame = makeLocalFrame(
    coordinateSystem,
    sites.map(site => site.position)
  );
  const projected: PlanarPoint[] = sites.map(site => frame.forward(site.position));
  let centerX = 0;
  let centerY = 0;
  for (const [x, y] of projected) {
    centerX += x / projected.length;
    centerY += y / projected.length;
  }
  const positions = new Float32Array(projected.length * 2);
  projected.forEach(([x, y], index) => {
    positions[2 * index] = x - centerX;
    positions[2 * index + 1] = y - centerY;
  });
  const capacity = Math.max(getDelaunayMaximumTriangleCount(sites.length), 8);
  const result = await withCatalogGpuSession(context.device, 'catalog-delaunay', async session => {
    const triangles = session.output('triangles', 'uint32x3', capacity);
    const count = session.output('count', 'uint32', 1);
    const overflow = session.output('overflow', 'uint32', 1);
    const requiredCount = session.output('required-count', 'uint32', 1);
    session.graph.add(
      new GPUDelaunayTessellation({
        positions: session.input('positions', positions, 'float32x2'),
        output: {
          triangles: triangles.view,
          status: {count: count.view, overflow: overflow.view, requiredCount: requiredCount.view}
        }
      })
    );
    session.run();
    return {
      triangles: await session.readUint32(triangles),
      count: await session.readScalar(count),
      overflow: await session.readScalar(overflow)
    };
  });
  // The capacity is the general-position maximum, so overflow means a contributor-side limit.
  if (result.overflow) throw new CatalogGpuUnsupportedError('GPUDelaunayTessellation overflowed');
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
  for (let triangleIndex = 0; triangleIndex < result.count; triangleIndex++) {
    const [a, b, c] = [0, 1, 2].map(corner => result.triangles[3 * triangleIndex + corner]);
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
  }
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
  const diagnostics: CatalogDiagnostic[] = [];
  if (duplicateCount)
    addCatalogDiagnostic(
      diagnostics,
      'duplicate-points',
      'warning',
      'Points at an already seen position are ignored; the first occurrence is the site.',
      duplicateCount
    );
  if (!result.count)
    addCatalogDiagnostic(
      diagnostics,
      'no-triangles',
      'warning',
      'Fewer than three distinct points, or all points collinear: no triangles.'
    );
  return {
    tables: [triangleTable, edgeTable],
    diagnostics,
    summary: {siteCount: sites.length, triangleCount: result.count, edgeCount: edges.length},
    dispatchedContributors: ['GPUDelaunayTessellation']
  };
}

/**
 * GPUPolygonTriangulation (Earcut per polygon part, monotone fallback) on the shifted metric frame.
 * Triangulations of a polygon are not unique, so triangles differ from the CPU ear clipper; the
 * triangle count per part (vertices plus twice the holes minus two) and the total area agree.
 * Triangle vertices are the float64 source positions of the rows the GPU indexed.
 */
export async function runGpuPolygonTriangulation(
  context: CatalogGpuContext,
  polygons: CatalogFeatureCollection,
  coordinateSystem: CatalogCoordinateSystem
): Promise<CatalogRunOutput> {
  const frame = makeLocalFrame(coordinateSystem, collectPositions(polygons), {
    kind: 'lambert-equal-area'
  });
  const diagnostics: CatalogDiagnostic[] = [];
  const parts: {sourceIndex: number; partIndex: number; rings: PlanarPoint[][]}[] = [];
  let oversized = 0;
  polygons.features.forEach((feature, sourceIndex) =>
    projectPolygonParts(feature, frame).forEach((rings, partIndex) => {
      if (rings.reduce((sum, ring) => sum + ring.length, 0) > MAXIMUM_TRIANGULATION_VERTICES)
        oversized++;
      else parts.push({sourceIndex, partIndex, rings});
    })
  );
  if (oversized)
    addCatalogDiagnostic(
      diagnostics,
      'part-too-large',
      'warning',
      `Polygon parts over ${MAXIMUM_TRIANGULATION_VERTICES} vertices are skipped.`
    );
  if (!parts.length) throw new CatalogGpuUnsupportedError('No polygon part to triangulate');
  let centerX = 0;
  let centerY = 0;
  let total = 0;
  for (const part of parts)
    for (const ring of part.rings)
      for (const [x, y] of ring) {
        centerX += x;
        centerY += y;
        total++;
      }
  centerX /= total;
  centerY /= total;
  const vertices: PlanarPoint[] = [];
  const positions: number[] = [];
  const polygonOffsets = [0];
  const ringOffsets = [0];
  const featureOffsets = [0];
  for (const part of parts) {
    for (const ring of part.rings) {
      for (const point of ring) {
        vertices.push(point);
        positions.push(point[0] - centerX, point[1] - centerY);
      }
      ringOffsets.push(vertices.length);
    }
    polygonOffsets.push(ringOffsets.length - 1);
    featureOffsets.push(polygonOffsets.length - 1);
  }
  const ringCount = ringOffsets.length - 1;
  const result = await withCatalogGpuSession(
    context.device,
    'catalog-polygon-triangulation',
    async session => {
      const indices = session.output(
        'indices',
        'uint32',
        getPolygonTriangulationIndexCount(vertices.length, ringCount, parts.length)
      );
      const valid = session.output('valid', 'uint32', parts.length);
      const triangleCount = session.output('triangle-count', 'uint32', parts.length);
      const triangleOffsets = session.output('triangle-offsets', 'uint32', parts.length + 1);
      session.graph.add(
        new GPUPolygonTriangulation({
          positions: session.input('positions', Float32Array.from(positions), 'float32x2'),
          featureOffsets: session.input(
            'feature-offsets',
            Uint32Array.from(featureOffsets),
            'uint32'
          ),
          polygonOffsets: session.input(
            'polygon-offsets',
            Uint32Array.from(polygonOffsets),
            'uint32'
          ),
          ringOffsets: session.input('ring-offsets', Uint32Array.from(ringOffsets), 'uint32'),
          output: {
            indices: indices.view,
            valid: valid.view,
            triangleCount: triangleCount.view,
            triangleOffsets: triangleOffsets.view
          }
        })
      );
      session.run();
      return {
        indices: await session.readUint32(indices),
        valid: await session.readUint32(valid),
        triangleCount: await session.readUint32(triangleCount),
        triangleOffsets: await session.readUint32(triangleOffsets)
      };
    }
  );
  const table = makeCatalogTable(
    'polygon-triangles',
    'Polygon triangles',
    {
      triangle_id: 'string',
      source_index: 'int32',
      source_id: 'string',
      part_index: 'int32',
      triangle_index: 'int32',
      area: 'float64',
      geometry: 'string'
    },
    'geometry'
  );
  let incompleteCount = 0;
  parts.forEach((part, partRow) => {
    if (!result.valid[partRow]) incompleteCount++;
    for (let triangle = 0; triangle < result.triangleCount[partRow]; triangle++) {
      const base = 3 * (result.triangleOffsets[partRow] + triangle);
      const corners = [0, 1, 2].map(corner => vertices[result.indices[base + corner]]);
      appendCatalogRow(table, {
        triangle_id: `${part.sourceIndex}:${part.partIndex}:${triangle}`,
        ...getSourceColumns(polygons.features[part.sourceIndex], part.sourceIndex),
        part_index: part.partIndex,
        triangle_index: triangle,
        area: Math.abs(getSignedArea(corners)),
        geometry: makeCatalogGeometryText('Polygon', [
          closeRing(corners.map(point => frame.inverse(point[0], point[1])))
        ])
      });
    }
  });
  if (incompleteCount)
    addCatalogDiagnostic(
      diagnostics,
      'triangulation-incomplete',
      'warning',
      'Ear clipping stalled on invalid or self-intersecting rings; the triangles found so far are kept.',
      incompleteCount
    );
  return {
    tables: [table],
    diagnostics,
    summary: {triangleCount: table.rowCount},
    dispatchedContributors: ['GPUPolygonTriangulation']
  };
}
