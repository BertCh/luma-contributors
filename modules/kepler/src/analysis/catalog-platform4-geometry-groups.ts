// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUGridGenerator,
  GPUGroupConvexHull,
  GPUKMeans,
  getGPUGridGeneratorParameterValues,
  getGPUKMeansParameterValues
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {CatalogGpuContext, CatalogRunOutput} from './catalog-binding-types';
import {
  addCatalogDiagnostic,
  appendCatalogRow,
  getFeatureIdentity,
  getPointPositions,
  makeCatalogGeometryText,
  makeCatalogTable,
  measureRingSignedArea,
  type CatalogCoordinateSystem,
  type CatalogDiagnostic,
  type CatalogFeatureCollection,
  type CatalogPosition
} from './catalog-geojson';
import {CatalogGpuUnsupportedError, withCatalogGpuSession} from './catalog-gpu-session';
import {collectCatalogPositions} from './catalog-platform4-geometry-common';
import {makeLocalFrame} from './catalog-local-frame';

/**
 * GPUGroupConvexHull: points are grouped on the CPU by property value (sorted like the CPU
 * reference), hull vertex rows come from the GPU, and the ring and area are rebuilt from the float64
 * source positions. Hull membership uses the contributor's exact lattice predicate, so a nearly
 * collinear point can differ from the float64 CPU hull; the vertex order may start elsewhere.
 */
export async function runGpuGroupConvexHull(
  context: CatalogGpuContext,
  points: CatalogFeatureCollection,
  parameters: {groupProperty: string; coordinateSystem: CatalogCoordinateSystem}
): Promise<CatalogRunOutput> {
  const groups = new Map<string, CatalogPosition[]>();
  for (const feature of points.features) {
    const value = parameters.groupProperty
      ? feature.properties?.[parameters.groupProperty]
      : undefined;
    const key = value === undefined || value === null ? '' : String(value);
    const list = groups.get(key) ?? [];
    list.push(...getPointPositions(feature.geometry));
    groups.set(key, list);
  }
  const ordered = [...groups.entries()].sort((a, b) => (a[0] < b[0] ? -1 : 1));
  const rowCount = ordered.reduce((sum, [, positions]) => sum + positions.length, 0);
  if (!rowCount) throw new CatalogGpuUnsupportedError('No points');
  const frame = makeLocalFrame('planar', collectCatalogPositions([points]), {shift: 'center'});
  const positions = new Float32Array(rowCount * 2);
  const labels = new Uint32Array(rowCount);
  const groupStarts: number[] = [];
  let cursor = 0;
  ordered.forEach(([, groupPositions], group) => {
    groupStarts.push(cursor);
    for (const position of groupPositions) {
      const [x, y] = frame.forward(position);
      positions[2 * cursor] = x;
      positions[2 * cursor + 1] = y;
      labels[cursor++] = group;
    }
  });
  const largestGroup = Math.max(...ordered.map(([, groupPositions]) => groupPositions.length));
  const {vertexIndices, offsets, counts, overflow} = await withCatalogGpuSession(
    context.device,
    'catalog-group-convex-hull',
    async session => {
      const vertexIndicesOutput = session.output('vertex-indices', 'uint32', rowCount);
      const offsetsOutput = session.output('offsets', 'uint32', ordered.length + 1);
      const countsOutput = session.output('counts', 'uint32', ordered.length);
      const overflowOutput = session.output('overflow', 'uint32', 1);
      session.graph.add(
        new GPUGroupConvexHull({
          positions: session.input('positions', positions, 'float32x2'),
          labels: session.input('labels', labels, 'uint32'),
          groupCount: ordered.length,
          maximumVerticesPerGroup: Math.max(3, largestGroup),
          vertexCapacity: rowCount,
          output: {
            vertexIndices: vertexIndicesOutput.view,
            offsets: offsetsOutput.view,
            counts: countsOutput.view,
            status: {overflow: overflowOutput.view}
          }
        })
      );
      session.run();
      return {
        vertexIndices: await session.readUint32(vertexIndicesOutput),
        offsets: await session.readUint32(offsetsOutput),
        counts: await session.readUint32(countsOutput),
        overflow: await session.readScalar(overflowOutput)
      };
    }
  );
  // Capacities cover every group, so overflow is a contributor-side limit.
  if (overflow) throw new CatalogGpuUnsupportedError('GPUGroupConvexHull overflowed');
  const table = makeCatalogTable(
    'group-hulls',
    'Group convex hulls',
    {
      hull_id: 'string',
      group_value: 'string',
      point_count: 'int32',
      hull_vertex_count: 'int32',
      area: 'float64',
      geometry: 'string'
    },
    'geometry'
  );
  const flatPositions = ordered.flatMap(([, groupPositions]) => groupPositions);
  let degenerate = 0;
  ordered.forEach(([group, groupPositions], index) => {
    const hull: CatalogPosition[] = [];
    for (let slot = offsets[index]; slot < offsets[index] + counts[index]; slot++)
      hull.push(flatPositions[vertexIndices[slot]]);
    if (hull.length < 3) {
      degenerate++;
      return;
    }
    const ring = [...hull, hull[0]];
    appendCatalogRow(table, {
      hull_id: `h${table.rowCount}`,
      group_value: group,
      point_count: groupPositions.length,
      hull_vertex_count: hull.length,
      area: Math.abs(measureRingSignedArea(ring, parameters.coordinateSystem)),
      geometry: makeCatalogGeometryText('Polygon', [ring])
    });
  });
  const diagnostics: CatalogDiagnostic[] = [];
  if (degenerate)
    addCatalogDiagnostic(
      diagnostics,
      'degenerate-group',
      'warning',
      'Groups with fewer than three distinct or only collinear points have no hull and are omitted.',
      degenerate
    );
  return {
    tables: [table],
    diagnostics,
    summary: {hullCount: table.rowCount, groupCount: groups.size},
    dispatchedContributors: ['GPUGroupConvexHull']
  };
}

/**
 * GPUGridGenerator (square grid). The lattice is generated from the extent's lower-left corner at
 * the origin and shifted back in float64, so cell corners carry float32 error relative to the cell
 * size only (about 1e-7 of one cell per lattice step).
 */
export async function runGpuGridGenerator(
  context: CatalogGpuContext,
  extent: CatalogFeatureCollection,
  parameters: {cellWidth: number; cellHeight: number},
  maximumCells: number
): Promise<CatalogRunOutput> {
  const positions = collectCatalogPositions([extent]);
  // An empty extent is an error row set on the CPU engine.
  if (!positions.length) throw new CatalogGpuUnsupportedError('The extent has no coordinates');
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const [x, y] of positions) {
    minX = Math.min(minX, x);
    maxX = Math.max(maxX, x);
    minY = Math.min(minY, y);
    maxY = Math.max(maxY, y);
  }
  const columns = Math.max(1, Math.ceil((maxX - minX) / parameters.cellWidth - 1e-9));
  const rows = Math.max(1, Math.ceil((maxY - minY) / parameters.cellHeight - 1e-9));
  // The CPU engine raises the allocation error for oversized grids.
  if (columns * rows > maximumCells) throw new CatalogGpuUnsupportedError('Grid capacity exceeded');
  const vertices = await withCatalogGpuSession(
    context.device,
    'catalog-grid-generator',
    async session => {
      const output = session.output('cells', 'float32x2', columns * rows * 4);
      session.graph.add(
        new GPUGridGenerator({
          gridType: 'square',
          columns,
          rows,
          parameters: session.input(
            'parameters',
            getGPUGridGeneratorParameterValues({
              minX: 0,
              minY: 0,
              cellWidth: parameters.cellWidth,
              cellHeight: parameters.cellHeight
            }),
            'float32'
          ),
          output: {positions: output.view}
        })
      );
      session.run();
      return session.readFloat32(output);
    }
  );
  const table = makeCatalogTable(
    'grid-cells',
    'Grid cells',
    {cell_id: 'string', row: 'int32', column: 'int32', geometry: 'string'},
    'geometry'
  );
  for (let row = 0; row < rows; row++)
    for (let column = 0; column < columns; column++) {
      const cell = row * columns + column;
      const ring: [number, number][] = [];
      for (let corner = 0; corner < 4; corner++)
        ring.push([
          minX + vertices[2 * (4 * cell + corner)],
          minY + vertices[2 * (4 * cell + corner) + 1]
        ]);
      ring.push(ring[0]);
      appendCatalogRow(table, {
        cell_id: `${row}:${column}`,
        row,
        column,
        geometry: makeCatalogGeometryText('Polygon', [ring])
      });
    }
  return {
    tables: [table],
    diagnostics: [],
    summary: {cellCount: table.rowCount, rows, columns},
    dispatchedContributors: ['GPUGridGenerator']
  };
}

const KMEANS_RESTART_COUNT = 4;
const KMEANS_MAXIMUM_ITERATIONS = 64;

/**
 * GPUKMeans on the metric local frame. The contributor seeds with k-means++ (best of four seeded
 * restarts) instead of the CPU's farthest-first start, so for well separated data both reach the
 * same partition while overlapping clusters may settle in different local optima. Clusters are
 * renumbered by first appearance among the points. At most 64 iterations; more use the CPU engine.
 */
export async function runGpuKMeans(
  context: CatalogGpuContext,
  points: CatalogFeatureCollection,
  parameters: {
    clusterCount: number;
    maximumIterations: number;
    coordinateSystem: CatalogCoordinateSystem;
  }
): Promise<CatalogRunOutput> {
  if (parameters.maximumIterations > KMEANS_MAXIMUM_ITERATIONS)
    throw new CatalogGpuUnsupportedError('GPUKMeans runs at most 64 iterations');
  const entries = points.features.flatMap((feature, pointIndex) =>
    getPointPositions(feature.geometry).map(position => ({feature, pointIndex, position}))
  );
  if (!entries.length) throw new CatalogGpuUnsupportedError('No points');
  const frame = makeLocalFrame(parameters.coordinateSystem, collectCatalogPositions([points]), {
    shift: 'center'
  });
  const clusterCount = Math.min(parameters.clusterCount, entries.length);
  const packed = new Float32Array(entries.length * 2);
  entries.forEach(({position}, index) => {
    const [x, y] = frame.forward(position);
    packed[2 * index] = x;
    packed[2 * index + 1] = y;
  });
  const result = await withCatalogGpuSession(context.device, 'catalog-k-means', async session => {
    const labels = session.output('labels', 'uint32', entries.length);
    const centers = session.output('centers', 'float32x2', clusterCount);
    const squaredDistances = session.output('squared-distances', 'float32', entries.length);
    const iterationCount = session.output('iteration-count', 'uint32', 1);
    session.graph.add(
      new GPUKMeans({
        positions: session.input('positions', packed, 'float32x2'),
        k: clusterCount,
        maximumIterations: parameters.maximumIterations,
        restartCount: KMEANS_RESTART_COUNT,
        parameters: session.input('parameters', getGPUKMeansParameterValues({seed: 1}), 'uint32'),
        output: {
          labels: labels.view,
          centers: centers.view,
          squaredDistances: squaredDistances.view,
          status: {iterationCount: iterationCount.view}
        }
      })
    );
    session.run();
    return {
      labels: await session.readUint32(labels),
      squaredDistances: await session.readFloat32(squaredDistances),
      iterations: await session.readScalar(iterationCount)
    };
  });
  const relabel = new Map<number, number>();
  const assignmentTable = makeCatalogTable(
    'kmeans-assignments',
    'Cluster assignments',
    {
      point_index: 'int32',
      point_source_id: 'string',
      cluster: 'int32',
      distance: 'float64',
      geometry: 'string'
    },
    'geometry'
  );
  const centroidTable = makeCatalogTable(
    'kmeans-centroids',
    'Cluster centroids',
    {cluster: 'int32', point_count: 'int32', geometry: 'string'},
    'geometry'
  );
  const sums = new Map<number, [number, number, number]>();
  let inertia = 0;
  entries.forEach((entry, index) => {
    const label = result.labels[index];
    if (!relabel.has(label)) relabel.set(label, relabel.size);
    const cluster = relabel.get(label)!;
    const sum = sums.get(cluster) ?? [0, 0, 0];
    sum[0] += entry.position[0];
    sum[1] += entry.position[1];
    sum[2]++;
    sums.set(cluster, sum);
    inertia += result.squaredDistances[index];
    appendCatalogRow(assignmentTable, {
      point_index: entry.pointIndex,
      point_source_id: getFeatureIdentity(entry.feature),
      cluster,
      distance: Math.sqrt(result.squaredDistances[index]),
      geometry: makeCatalogGeometryText('Point', entry.position)
    });
  });
  for (let cluster = 0; cluster < clusterCount; cluster++) {
    const sum = sums.get(cluster);
    appendCatalogRow(centroidTable, {
      cluster,
      point_count: sum ? sum[2] : 0,
      geometry: sum ? makeCatalogGeometryText('Point', [sum[0] / sum[2], sum[1] / sum[2]]) : null
    });
  }
  const diagnostics: CatalogDiagnostic[] = [];
  if (clusterCount < parameters.clusterCount)
    addCatalogDiagnostic(
      diagnostics,
      'cluster-count-reduced',
      'warning',
      'Fewer points than requested clusters; the cluster count was reduced to the point count.'
    );
  return {
    tables: [assignmentTable, centroidTable],
    diagnostics,
    summary: {clusterCount, iterations: result.iterations, inertia},
    dispatchedContributors: ['GPUKMeans']
  };
}
