// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUPolygonRasterization,
  GPURasterJoin,
  getGPUPolygonRasterizationParameterValues,
  makeGPURasterMetadata
} from '@luma-contributors/analysis/raster-analysis';
import type {CatalogOperationBinding, CatalogRunOutput} from './catalog-binding-types';
import {COORDINATE_SYSTEM_PARAMETER} from './catalog-binding-types';
import {
  addCatalogDiagnostic,
  appendCatalogRow,
  getPolygonParts,
  makeCatalogTable,
  type CatalogCoordinateSystem,
  type CatalogDiagnostic,
  type CatalogFeatureCollection
} from './catalog-geojson';
import {CatalogGpuUnsupportedError, withCatalogGpuSession} from './catalog-gpu-session';
import {
  PLATFORM3_GPU_OR_CPU_CLEANUP,
  collectAllPositions,
  getFeatureGeometryText,
  makeGridSpec,
  makeInputSpec,
  makeNumberParameter,
  makeOutputSpec,
  makeStringParameter,
  readPointSamples
} from './catalog-platform3-b-common';
import {makeLocalFrame, type LocalFrame} from './catalog-local-frame';

type Edge = {x0: number; y0: number; x1: number; y1: number};

type JoinInputs = {
  frame: LocalFrame;
  /** Float32 frame coordinates of the polygon vertices, with feature, polygon and ring offsets. */
  positions: Float32Array;
  featureOffsets: Uint32Array;
  polygonOffsets: Uint32Array;
  ringOffsets: Uint32Array;
  /** Per feature, per polygon, the implicitly closed edges. */
  edges: Edge[][][];
  featureCount: number;
  extentX: number;
  extentY: number;
  pointPositions: Float32Array;
  pointValues: Float32Array;
  pointCount: number;
  droppedCount: number;
};

function readJoinInputs(
  points: CatalogFeatureCollection,
  polygons: CatalogFeatureCollection,
  valueProperty: string,
  coordinateSystem: CatalogCoordinateSystem
): JoinInputs {
  const frame = makeLocalFrame(coordinateSystem, collectAllPositions(polygons), {shift: 'minimum'});
  const positions: number[] = [];
  const featureOffsets = [0];
  const polygonOffsets = [0];
  const ringOffsets = [0];
  const edges: Edge[][][] = [];
  let extentX = 0;
  let extentY = 0;
  for (const feature of polygons.features) {
    const featureEdges: Edge[][] = [];
    for (const polygon of getPolygonParts(feature.geometry)) {
      const polygonEdges: Edge[] = [];
      for (const ring of polygon) {
        const vertices = ring.map(position => frame.forward(position));
        for (const [x, y] of vertices) {
          positions.push(x, y);
          extentX = Math.max(extentX, x);
          extentY = Math.max(extentY, y);
        }
        ringOffsets.push(positions.length / 2);
        const rounded = vertices.map(([x, y]) => [Math.fround(x), Math.fround(y)]);
        rounded.forEach(([x0, y0], index) => {
          const [x1, y1] = rounded[(index + 1) % rounded.length];
          polygonEdges.push({x0, y0, x1, y1});
        });
      }
      polygonOffsets.push(ringOffsets.length - 1);
      featureEdges.push(polygonEdges);
    }
    featureOffsets.push(polygonOffsets.length - 1);
    edges.push(featureEdges);
  }
  const sample = readPointSamples(points, valueProperty, frame);
  const pointPositions = new Float32Array(2 * sample.count);
  const pointValues = new Float32Array(sample.count);
  for (let index = 0; index < sample.count; index++) {
    pointPositions[2 * index] = sample.x[index];
    pointPositions[2 * index + 1] = sample.y[index];
    pointValues[index] = sample.value[index];
  }
  return {
    frame,
    positions: Float32Array.from(positions),
    featureOffsets: Uint32Array.from(featureOffsets),
    polygonOffsets: Uint32Array.from(polygonOffsets),
    ringOffsets: Uint32Array.from(ringOffsets),
    edges,
    featureCount: polygons.features.length,
    extentX,
    extentY,
    pointPositions,
    pointValues,
    pointCount: sample.count,
    droppedCount: sample.droppedCount
  };
}

/** Even-odd test with the contributor's half-open rule. */
function isInsideEdges(edges: readonly Edge[], x: number, y: number): boolean {
  let inside = false;
  for (const {x0, y0, x1, y1} of edges) {
    const lowerIsFirst = y0 < y1;
    const lowerX = lowerIsFirst ? x0 : x1;
    const lowerY = lowerIsFirst ? y0 : y1;
    const upperX = lowerIsFirst ? x1 : x0;
    const upperY = lowerIsFirst ? y1 : y0;
    if (lowerY <= y && y < upperY) {
      const crossingX = lowerX + ((upperX - lowerX) * (y - lowerY)) / (upperY - lowerY);
      if (crossingX <= x) inside = !inside;
    }
  }
  return inside;
}

type ZoneTotals = {
  counts: number[];
  sums: number[];
  boundaryCounts: number[];
  unassignedBoundaryCount: number;
  outsideCount: number;
};

function computeExactJoin(inputs: JoinInputs): ZoneTotals {
  const totals: ZoneTotals = {
    counts: new Array(inputs.featureCount).fill(0),
    sums: new Array(inputs.featureCount).fill(0),
    boundaryCounts: new Array(inputs.featureCount).fill(0),
    unassignedBoundaryCount: 0,
    outsideCount: 0
  };
  for (let point = 0; point < inputs.pointCount; point++) {
    const x = inputs.pointPositions[2 * point];
    const y = inputs.pointPositions[2 * point + 1];
    const zone = inputs.edges.findIndex(feature =>
      feature.some(polygonEdges => isInsideEdges(polygonEdges, x, y))
    );
    if (zone < 0) totals.outsideCount++;
    else {
      totals.counts[zone]++;
      totals.sums[zone] += inputs.pointValues[point];
    }
  }
  return totals;
}

function buildJoinOutput(
  polygons: CatalogFeatureCollection,
  inputs: JoinInputs,
  totals: ZoneTotals,
  hasValues: boolean,
  isApproximate: boolean,
  cellSize: number,
  dispatchedContributors?: readonly string[]
): CatalogRunOutput {
  const diagnostics: CatalogDiagnostic[] = [];
  if (inputs.droppedCount)
    addCatalogDiagnostic(
      diagnostics,
      'feature-dropped',
      'warning',
      'Features without a point position or numeric value are skipped.',
      inputs.droppedCount
    );
  const table = makeCatalogTable(
    'raster-join',
    'Points per polygon',
    {
      feature_index: 'int32',
      count: 'int32',
      sum: 'float64',
      mean: 'float64',
      boundary_count: 'int32',
      geometry: 'string'
    },
    'geometry'
  );
  polygons.features.forEach((feature, index) =>
    appendCatalogRow(table, {
      feature_index: index,
      count: totals.counts[index],
      sum: hasValues ? totals.sums[index] : null,
      mean: hasValues && totals.counts[index] ? totals.sums[index] / totals.counts[index] : null,
      boundary_count: totals.boundaryCounts[index],
      geometry: getFeatureGeometryText(feature)
    })
  );
  return {
    tables: [table],
    diagnostics,
    summary: {
      polygonCount: polygons.features.length,
      pointCount: inputs.pointCount,
      outsideCount: totals.outsideCount,
      unassignedBoundaryCount: totals.unassignedBoundaryCount,
      isApproximate: isApproximate ? 1 : 0,
      cellSize
    },
    ...(dispatchedContributors ? {dispatchedContributors} : {})
  };
}

const RASTER_JOIN_BINDING: CatalogOperationBinding = {
  id: 'raster-join-points-in-polygons',
  title: 'Raster join: points per polygon',
  domain: 'raster',
  summary:
    'Counts and sums points per polygon by rasterizing the polygons once and binning each point into its raster cell, with a per-polygon bound on boundary cell points.',
  capabilityIds: ['spatial-joins'],
  contributorNames: ['GPUPolygonRasterization', 'GPURasterJoin'],
  gpuContributorNames: ['GPUPolygonRasterization', 'GPURasterJoin'],
  engine: 'gpu-or-cpu',
  inputs: [
    makeInputSpec('points', 'Points', ['point']),
    makeInputSpec('polygons', 'Polygons', ['polygon'], 20000, 200000)
  ],
  parameters: [
    makeStringParameter('valueProperty', 'Value property to sum (empty = count only)'),
    makeNumberParameter('cellSize', 'Raster cell size', 10, 1e-6, 1e7, 'crs'),
    COORDINATE_SYSTEM_PARAMETER
  ],
  outputs: [
    makeOutputSpec('raster-join', 'Points per polygon', 'polygon', ['feature_index'], 20000)
  ],
  cleanup: PLATFORM3_GPU_OR_CPU_CLEANUP,
  declaredBehavior: [
    'Polygons are rasterized in a local meter frame (planar units for planar data) on a grid covering their extent, capped at 250,000 cells (a coarser cell size is required beyond that).',
    'Each point joins the smallest-index polygon feature whose even-odd interior holds its cell center (holes respected); points outside the raster join nothing and are counted.',
    'The CPU engine is the exact point-in-polygon join. The GPU engine is approximate only for points in boundary cells: boundary_count per polygon bounds how many of its points may differ, and the summary carries unassignedBoundaryCount.',
    'Points without a position or (when set) a numeric value are skipped and counted.'
  ],
  limitation:
    'Approximate on the GPU by design (error bound published, not re-joined exactly); planar float32 frame coordinates; sums are float32. The raster-only contributors (terrain, zonal raster statistics) stay behind the raster panels because they take raster tiles, not feature collections.',
  run: (inputs, values) => {
    const joinInputs = readJoinInputs(
      inputs['points'],
      inputs['polygons'],
      values['valueProperty'] as string,
      values['coordinateSystem'] as CatalogCoordinateSystem
    );
    return buildJoinOutput(
      inputs['polygons'],
      joinInputs,
      computeExactJoin(joinInputs),
      Boolean(values['valueProperty']),
      false,
      values['cellSize'] as number
    );
  },
  runGpu: async (context, inputs, values) => {
    const joinInputs = readJoinInputs(
      inputs['points'],
      inputs['polygons'],
      values['valueProperty'] as string,
      values['coordinateSystem'] as CatalogCoordinateSystem
    );
    const cellSize = values['cellSize'] as number;
    if (!joinInputs.featureCount || !joinInputs.pointCount || joinInputs.positions.length < 2)
      throw new CatalogGpuUnsupportedError('Nothing to join');
    const hasValues = Boolean(values['valueProperty']);
    const grid = makeGridSpec(joinInputs.extentX, joinInputs.extentY, cellSize);
    const raster = makeGPURasterMetadata({
      width: grid.columns,
      height: grid.rows,
      cellSize: [cellSize, cellSize],
      origin: [0, 0],
      rowDirection: 'north'
    });
    // Every (edge, row) crossing needs a slot; one more row per edge covers the rounding.
    let crossingCapacity = 64;
    for (let edge = 0; edge + 1 < joinInputs.positions.length / 2; edge++)
      crossingCapacity +=
        Math.ceil(
          Math.abs(joinInputs.positions[2 * edge + 3] - joinInputs.positions[2 * edge + 1]) /
            cellSize
        ) + 1;
    const totals = await withCatalogGpuSession(
      context.device,
      'catalog-raster-join',
      async session => {
        const cellCount = grid.columns * grid.rows;
        const zones = session.output('zones', 'uint32', cellCount);
        const boundary = session.output('boundary', 'uint32', cellCount);
        const overflow = session.output('overflow', 'uint32', 1);
        const parameters = session.input(
          'parameters',
          getGPUPolygonRasterizationParameterValues(raster),
          'float32'
        );
        session.graph.add(
          new GPUPolygonRasterization({
            id: 'catalog-raster-join-polygons',
            grid: raster,
            parameters,
            positions: session.input('positions', joinInputs.positions, 'float32x2'),
            featureOffsets: session.input('feature-offsets', joinInputs.featureOffsets, 'uint32'),
            polygonOffsets: session.input('polygon-offsets', joinInputs.polygonOffsets, 'uint32'),
            ringOffsets: session.input('ring-offsets', joinInputs.ringOffsets, 'uint32'),
            crossingCapacity,
            output: {zones: zones.view, boundary: boundary.view, status: {overflow: overflow.view}}
          })
        );
        const counts = session.output('counts', 'uint32', joinInputs.featureCount);
        const sums = session.output('sums', 'float32', joinInputs.featureCount);
        const boundaryCounts = session.output('boundary-counts', 'uint32', joinInputs.featureCount);
        const unassigned = session.output('unassigned', 'uint32', 1);
        const outside = session.output('outside', 'uint32', 1);
        session.graph.add(
          new GPURasterJoin({
            id: 'catalog-raster-join',
            grid: raster,
            parameters,
            points: session.input('points', joinInputs.pointPositions, 'float32x2'),
            ...(hasValues
              ? {values: session.input('values', joinInputs.pointValues, 'float32')}
              : {}),
            zones: zones.view,
            boundary: boundary.view,
            zoneCount: joinInputs.featureCount,
            output: {
              counts: counts.view,
              ...(hasValues ? {sums: sums.view} : {}),
              boundaryCounts: boundaryCounts.view,
              unassignedBoundaryCount: unassigned.view,
              outsideCount: outside.view
            }
          })
        );
        session.run();
        // The crossing capacity covers every edge row, so an overflow flag cannot be set.
        return {
          counts: Array.from(await session.readUint32(counts)),
          sums: hasValues
            ? Array.from(await session.readFloat32(sums))
            : new Array<number>(joinInputs.featureCount).fill(0),
          boundaryCounts: Array.from(await session.readUint32(boundaryCounts)),
          unassignedBoundaryCount: await session.readScalar(unassigned),
          outsideCount: await session.readScalar(outside)
        };
      }
    );
    return buildJoinOutput(inputs['polygons'], joinInputs, totals, hasValues, true, cellSize, [
      'GPUPolygonRasterization',
      'GPURasterJoin'
    ]);
  }
};

export const PLATFORM4_RASTER_BINDINGS: readonly CatalogOperationBinding[] = [RASTER_JOIN_BINDING];
