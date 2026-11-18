// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

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
  type CatalogCoordinateSystem,
  type CatalogDiagnostic,
  type CatalogFeatureCollection,
  type CatalogPosition
} from './catalog-geojson';
import type {CatalogGpuContext, CatalogRunOutput} from './catalog-binding-types';
import {
  CatalogGpuSession,
  CatalogGpuUnsupportedError,
  withCatalogGpuSession,
  type CatalogGpuOutput
} from './catalog-gpu-session';
import {
  closeCatalogRing,
  createCatalogFrame,
  getCatalogPolygonArea,
  getCatalogRectangle,
  packCatalogLines,
  packCatalogPoints,
  packCatalogPolygons,
  readCatalogPolygons,
  type CatalogFrame
} from './catalog-gpu-pack';
import {
  GPUBufferSurface,
  GPUGeometryMeasures,
  GPUGeometryValidity,
  GPULineSegmentize,
  GPUMakeValid,
  GPUPolygonOverlay,
  GPURectangleClip,
  GPU_GEOMETRY_VALIDITY_BIT,
  GPU_GEOMETRY_VALIDITY_STRUCTURAL_MASK,
  getGPUBufferSurfaceMaximumBoundaryRowCount,
  getGPULineSegmentizeParameterValues,
  getGPUOffsetCurveRowsPerVertex,
  getGPURectangleClipParameterValues,
  type GPUPolygonOverlayOperation,
  type GPUPolygonOverlayOutput
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {type GPUSpatialContext} from '@luma-contributors/analysis/analysis-core';

type FeatureKind = 'polygon' | 'line' | 'point' | 'none';

function getFeatureKind(feature: {
  geometry: CatalogFeatureCollection['features'][number]['geometry'];
}): FeatureKind {
  if (getPolygonParts(feature.geometry).length) return 'polygon';
  if (getLineParts(feature.geometry).length) return 'line';
  if (getPointPositions(feature.geometry).length) return 'point';
  return 'none';
}

function getMeasureContext(coordinateSystem: CatalogCoordinateSystem): GPUSpatialContext {
  return coordinateSystem === 'planar'
    ? {coordinateSpace: 'planar', metric: 'native', units: 'native'}
    : {
        coordinateSpace: 'longitude-latitude',
        metric: 'great-circle',
        units: 'meters',
        sphereRadius: EARTH_RADIUS_METERS
      };
}

// ---------------------------------------------------------------------------------------------
// Measures
// ---------------------------------------------------------------------------------------------

type MeasureRow = {
  area: number | null;
  length: number | null;
  vertexCount: number;
  bounds: [number, number, number, number];
  centroid: [number, number];
};

/** GPUGeometryMeasures per geometry kind; centroids are area- or length-weighted, unlike the CPU vertex average. */
export async function runGpuGeometryMeasures(
  context: CatalogGpuContext,
  features: CatalogFeatureCollection,
  coordinateSystem: CatalogCoordinateSystem
): Promise<CatalogRunOutput> {
  const spatialContext = getMeasureContext(coordinateSystem);
  const rows = new Map<number, MeasureRow>();
  const invalid = new Set<number>();
  const readRows = async (
    session: CatalogGpuSession,
    sourceIndices: readonly number[],
    outputs: {
      lengths: CatalogGpuOutput<'float32'>;
      areas?: CatalogGpuOutput<'float32'>;
      centroids: CatalogGpuOutput<'float32x2'>;
      bounds: CatalogGpuOutput<'float32x4'>;
      vertexCounts: CatalogGpuOutput<'uint32'>;
    },
    hasLength: boolean
  ) => {
    const lengths = await session.readFloat32(outputs.lengths);
    const areas = outputs.areas ? await session.readFloat32(outputs.areas) : undefined;
    const centroids = await session.readFloat32(outputs.centroids);
    const bounds = await session.readFloat32(outputs.bounds);
    const vertexCounts = await session.readUint32(outputs.vertexCounts);
    sourceIndices.forEach((sourceIndex, row) => {
      const area = areas ? areas[row] : null;
      if (area !== null && Number.isNaN(area)) invalid.add(sourceIndex);
      rows.set(sourceIndex, {
        area: area === null || Number.isNaN(area) ? null : Math.max(0, area),
        length: hasLength && !Number.isNaN(lengths[row]) ? lengths[row] : null,
        vertexCount: vertexCounts[row],
        bounds: [bounds[4 * row], bounds[4 * row + 1], bounds[4 * row + 2], bounds[4 * row + 3]],
        centroid: [centroids[2 * row], centroids[2 * row + 1]]
      });
    });
  };
  const polygons = packCatalogPolygons(features.features, {closing: 'keep'});
  if (polygons.sourceIndices.length)
    await withCatalogGpuSession(context.device, 'catalog-measures-polygons', async session => {
      const count = polygons.sourceIndices.length;
      const outputs = {
        lengths: session.output('lengths', 'float32', count),
        areas: session.output('areas', 'float32', count),
        centroids: session.output('centroids', 'float32x2', count),
        bounds: session.output('bounds', 'float32x4', count),
        vertexCounts: session.output('vertex-counts', 'uint32', count)
      };
      session.graph.add(
        new GPUGeometryMeasures({
          geometryType: 'polygons',
          positions: session.input('positions', polygons.positions, 'float32x2'),
          featureOffsets: session.input('feature-offsets', polygons.featureOffsets, 'uint32'),
          polygonOffsets: session.input('polygon-offsets', polygons.polygonOffsets, 'uint32'),
          ringOffsets: session.input('ring-offsets', polygons.ringOffsets, 'uint32'),
          spatialContext,
          output: {
            lengths: outputs.lengths.view,
            areas: outputs.areas.view,
            centroids: outputs.centroids.view,
            bounds: outputs.bounds.view,
            vertexCounts: outputs.vertexCounts.view
          }
        })
      );
      session.run();
      await readRows(session, polygons.sourceIndices, outputs, true);
    });
  const lines = packCatalogLines(features.features);
  if (lines.sourceIndices.length)
    await withCatalogGpuSession(context.device, 'catalog-measures-lines', async session => {
      const count = lines.sourceIndices.length;
      const outputs = {
        lengths: session.output('lengths', 'float32', count),
        centroids: session.output('centroids', 'float32x2', count),
        bounds: session.output('bounds', 'float32x4', count),
        vertexCounts: session.output('vertex-counts', 'uint32', count)
      };
      session.graph.add(
        new GPUGeometryMeasures({
          geometryType: 'lines',
          positions: session.input('positions', lines.positions, 'float32x2'),
          lineOffsets: session.input('line-offsets', lines.lineOffsets, 'uint32'),
          featureOffsets: session.input('feature-offsets', lines.featureOffsets, 'uint32'),
          spatialContext,
          output: {
            lengths: outputs.lengths.view,
            centroids: outputs.centroids.view,
            bounds: outputs.bounds.view,
            vertexCounts: outputs.vertexCounts.view
          }
        })
      );
      session.run();
      await readRows(session, lines.sourceIndices, outputs, true);
    });
  const points = packCatalogPoints(features.features);
  if (points.sourceIndices.length)
    await withCatalogGpuSession(context.device, 'catalog-measures-points', async session => {
      const count = points.sourceIndices.length;
      const outputs = {
        lengths: session.output('lengths', 'float32', count),
        centroids: session.output('centroids', 'float32x2', count),
        bounds: session.output('bounds', 'float32x4', count),
        vertexCounts: session.output('vertex-counts', 'uint32', count)
      };
      session.graph.add(
        new GPUGeometryMeasures({
          geometryType: 'points',
          positions: session.input('positions', points.positions, 'float32x2'),
          featureOffsets: session.input('feature-offsets', points.featureOffsets, 'uint32'),
          spatialContext,
          output: {
            lengths: outputs.lengths.view,
            centroids: outputs.centroids.view,
            bounds: outputs.bounds.view,
            vertexCounts: outputs.vertexCounts.view
          }
        })
      );
      session.run();
      await readRows(session, points.sourceIndices, outputs, false);
    });
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
    const row = rows.get(sourceIndex);
    if (!row) emptyCount++;
    appendCatalogRow(table, {
      source_index: sourceIndex,
      source_id: getFeatureIdentity(feature),
      geometry_type: feature.geometry?.type ?? null,
      area: row?.area ?? null,
      length: row?.length ?? null,
      vertex_count: row?.vertexCount ?? 0,
      minimum_x: row?.bounds[0] ?? null,
      minimum_y: row?.bounds[1] ?? null,
      maximum_x: row?.bounds[2] ?? null,
      maximum_y: row?.bounds[3] ?? null,
      centroid_x: row?.centroid[0] ?? null,
      centroid_y: row?.centroid[1] ?? null,
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
  if (invalid.size)
    addCatalogDiagnostic(
      diagnostics,
      'gpu-measure-rejected',
      'warning',
      'GPUGeometryMeasures rejected these polygons (ambiguous edges, more than 360 degrees of longitude or a pole); area is null.',
      invalid.size
    );
  addCatalogDiagnostic(
    diagnostics,
    'centroid-definition',
    'info',
    'GPU centroid is area-weighted (polygons), length-weighted (lines) or the vertex mean (points); the CPU reference uses the planar vertex average.'
  );
  return {
    tables: [table],
    diagnostics,
    summary: {
      featureCount: features.features.length,
      lengthUnit: coordinateSystem === 'planar' ? 'coordinate-units' : 'meters',
      areaUnit: coordinateSystem === 'planar' ? 'coordinate-units^2' : 'square-meters'
    },
    dispatchedContributors: ['GPUGeometryMeasures']
  };
}

// ---------------------------------------------------------------------------------------------
// Validity
// ---------------------------------------------------------------------------------------------

const VALIDITY_ERROR_NAMES: readonly [number, string][] = [
  [GPU_GEOMETRY_VALIDITY_BIT.nonFinite, 'non-finite-coordinate'],
  [GPU_GEOMETRY_VALIDITY_BIT.unclosedRing, 'ring-not-closed'],
  [GPU_GEOMETRY_VALIDITY_BIT.shortRing, 'ring-too-short'],
  [GPU_GEOMETRY_VALIDITY_BIT.selfIntersection, 'self-intersection'],
  [GPU_GEOMETRY_VALIDITY_BIT.crossingRings, 'ring-crossing'],
  [GPU_GEOMETRY_VALIDITY_BIT.holeOutsideShell, 'hole-outside-shell'],
  [GPU_GEOMETRY_VALIDITY_BIT.uncertain, 'predicate-uncertain']
];

const VALIDITY_WARNING_NAMES: readonly [number, string][] = [
  [GPU_GEOMETRY_VALIDITY_BIT.repeatedVertex, 'duplicate-consecutive-vertex'],
  [GPU_GEOMETRY_VALIDITY_BIT.badOrientation, 'shell-not-counterclockwise']
];

/** Converts a GPUGeometryValidity bit mask into the catalog's issue names. */
export function describeValidityMask(mask: number): {errors: string[]; warnings: string[]} {
  return {
    errors: VALIDITY_ERROR_NAMES.filter(([bit]) => mask & bit).map(([, name]) => name),
    warnings: VALIDITY_WARNING_NAMES.filter(([bit]) => mask & bit).map(([, name]) => name)
  };
}

export async function runGpuGeometryValidity(
  context: CatalogGpuContext,
  polygons: CatalogFeatureCollection
): Promise<CatalogRunOutput> {
  const packed = packCatalogPolygons(polygons.features, {closing: 'keep'});
  const masks = new Map<number, number>();
  let overflowed = false;
  if (packed.sourceIndices.length) {
    // Valid polygons produce almost no same-feature intersections; grow once on overflow.
    let intersectionCapacity = Math.max(1024, 4 * packed.vertexCount);
    for (let attempt = 0; attempt < 2; attempt++) {
      const result = await withCatalogGpuSession(
        context.device,
        'catalog-validity',
        async session => {
          const count = packed.sourceIndices.length;
          const mask = session.output('mask', 'uint32', count);
          const overflow = session.output('overflow', 'uint32', 1);
          session.graph.add(
            new GPUGeometryValidity({
              polygons: {
                kind: 'polygons',
                positions: session.input('positions', packed.positions, 'float32x2'),
                featureOffsets: session.input('feature-offsets', packed.featureOffsets, 'uint32'),
                polygonOffsets: session.input('polygon-offsets', packed.polygonOffsets, 'uint32'),
                ringOffsets: session.input('ring-offsets', packed.ringOffsets, 'uint32')
              },
              ringClosure: 'explicit',
              orientation: 'counter-clockwise-shell',
              intersectionCapacity,
              output: {issueCode: mask.view, status: {overflow: overflow.view}}
            })
          );
          session.run();
          return {
            mask: await session.readUint32(mask),
            overflow: await session.readScalar(overflow)
          };
        }
      );
      overflowed = result.overflow !== 0;
      packed.sourceIndices.forEach((sourceIndex, row) => masks.set(sourceIndex, result.mask[row]));
      if (!overflowed) break;
      intersectionCapacity *= 8;
    }
  }
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
  polygons.features.forEach((feature, sourceIndex) => {
    const mask = masks.get(sourceIndex);
    const {errors, warnings} =
      mask === undefined ? {errors: ['not-polygon'], warnings: []} : describeValidityMask(mask);
    const isInvalid = mask === undefined || (mask & GPU_GEOMETRY_VALIDITY_STRUCTURAL_MASK) !== 0;
    if (isInvalid) invalidCount++;
    appendCatalogRow(table, {
      source_index: sourceIndex,
      source_id: getFeatureIdentity(feature),
      status: isInvalid ? 'invalid' : overflowed ? 'incomplete' : 'valid',
      error_count: errors.length,
      warning_count: warnings.length,
      issues: [...errors, ...warnings].join(','),
      geometry: feature.geometry ? JSON.stringify(feature.geometry) : null
    });
  });
  if (invalidCount)
    addCatalogDiagnostic(
      diagnostics,
      'invalid-geometry',
      'warning',
      'Features failed validity; none were repaired. Use the topology repair operation.',
      invalidCount
    );
  if (overflowed)
    addCatalogDiagnostic(
      diagnostics,
      'intersection-capacity-overflow',
      'warning',
      'The intersection list overflowed even after growth; self-intersection and ring-crossing flags may be missing.'
    );
  addCatalogDiagnostic(
    diagnostics,
    'gpu-validity-scope',
    'info',
    'GPU validity checks closure, short rings, repeated vertices, orientation, self-intersection, ring crossings and holes outside shells; zero-area rings are not a separate flag.'
  );
  return {
    tables: [table],
    diagnostics,
    summary: {featureCount: polygons.features.length, invalidCount, skippedCount: 0},
    dispatchedContributors: ['GPUGeometryValidity']
  };
}

// ---------------------------------------------------------------------------------------------
// Segmentize
// ---------------------------------------------------------------------------------------------

const MAXIMUM_PIECES_PER_SEGMENT = 1024;

export async function runGpuLineSegmentize(
  context: CatalogGpuContext,
  lines: CatalogFeatureCollection,
  parameters: {maximumSegmentLength: number; coordinateSystem: CatalogCoordinateSystem},
  maximumOutputVertices: number
): Promise<CatalogRunOutput> {
  const packed = packCatalogLines(lines.features);
  const spatialContext = getMeasureContext(parameters.coordinateSystem);
  // Exact output requirement: each segment becomes clamp(ceil(length / maximum), 1, cap) pieces.
  let requiredVertices = 0;
  for (let line = 0; line < packed.lineSourceIndices.length; line++) {
    const first = packed.lineOffsets[line];
    const last = packed.lineOffsets[line + 1];
    if (last > first) requiredVertices += 1;
    for (let vertex = first + 1; vertex < last; vertex++) {
      const length = measureDistance(
        [packed.positions[2 * vertex - 2], packed.positions[2 * vertex - 1]],
        [packed.positions[2 * vertex], packed.positions[2 * vertex + 1]],
        parameters.coordinateSystem
      );
      requiredVertices += Math.min(
        MAXIMUM_PIECES_PER_SEGMENT,
        Math.max(1, Math.ceil(length / parameters.maximumSegmentLength))
      );
    }
  }
  // Output vertex capacity exceeded; raise the segment length or reduce the input.
  if (requiredVertices > maximumOutputVertices)
    throw new KeplerAllocationError('capacity', {message: 'Output capacity exceeded'});
  const output = await withCatalogGpuSession(
    context.device,
    'catalog-segmentize',
    async session => {
      const capacity = Math.max(requiredVertices + 16, 16);
      const lineCount = packed.lineSourceIndices.length;
      const positions = session.output('positions', 'float32x2', capacity);
      const lineOffsets = session.output('line-offsets', 'uint32', lineCount + 1);
      const count = session.output('count', 'uint32', 1);
      const overflow = session.output('overflow', 'uint32', 1);
      session.graph.add(
        new GPULineSegmentize({
          positions: session.input('input-positions', packed.positions, 'float32x2'),
          lineOffsets: session.input('input-offsets', packed.lineOffsets, 'uint32'),
          spatialContext,
          maximumPieceCount: MAXIMUM_PIECES_PER_SEGMENT,
          parameters: session.input(
            'parameters',
            getGPULineSegmentizeParameterValues({
              maximumSegmentLength: parameters.maximumSegmentLength,
              spatialContext
            }),
            'float32'
          ),
          output: {
            positions: positions.view,
            lineOffsets: lineOffsets.view,
            status: {count: count.view, overflow: overflow.view}
          }
        })
      );
      session.run();
      return {
        positions: await session.readFloat32(positions),
        lineOffsets: await session.readUint32(lineOffsets),
        overflow: await session.readScalar(overflow)
      };
    }
  );
  // The exact requirement was sized above, so overflow means a contributor-side limit.
  if (output.overflow) throw new CatalogGpuUnsupportedError('GPULineSegmentize output overflowed');
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
  const partsByFeature = new Map<number, CatalogPosition[][]>();
  const addedByFeature = new Map<number, number>();
  packed.lineSourceIndices.forEach((sourceIndex, line) => {
    const part: CatalogPosition[] = [];
    for (let vertex = output.lineOffsets[line]; vertex < output.lineOffsets[line + 1]; vertex++)
      part.push([output.positions[2 * vertex], output.positions[2 * vertex + 1]]);
    const parts = partsByFeature.get(sourceIndex) ?? [];
    parts.push(part);
    partsByFeature.set(sourceIndex, parts);
    addedByFeature.set(
      sourceIndex,
      (addedByFeature.get(sourceIndex) ?? 0) +
        part.length -
        (packed.lineOffsets[line + 1] - packed.lineOffsets[line])
    );
  });
  let droppedCount = 0;
  lines.features.forEach((feature, sourceIndex) => {
    const parts = partsByFeature.get(sourceIndex);
    if (!parts) {
      droppedCount++;
      return;
    }
    appendCatalogRow(table, {
      output_id: String(sourceIndex),
      source_index: sourceIndex,
      source_id: getFeatureIdentity(feature),
      added_vertex_count: addedByFeature.get(sourceIndex) ?? 0,
      length: parts.reduce(
        (sum, part) => sum + measurePathLength(part, parameters.coordinateSystem),
        0
      ),
      geometry:
        parts.length === 1
          ? makeCatalogGeometryText('LineString', parts[0])
          : makeCatalogGeometryText('MultiLineString', parts)
    });
  });
  const diagnostics: CatalogDiagnostic[] = [];
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
    'interpolation-model',
    'info',
    parameters.coordinateSystem === 'planar'
      ? 'New vertices are linear in coordinate space.'
      : 'New vertices lie on the great circle between the input vertices (the CPU reference interpolates linearly in degrees).'
  );
  return {
    tables: [table],
    diagnostics,
    summary: {outputVertexCount: requiredVertices, outputFeatureCount: table.rowCount},
    dispatchedContributors: ['GPULineSegmentize']
  };
}

// ---------------------------------------------------------------------------------------------
// Rectangle clip
// ---------------------------------------------------------------------------------------------

export async function runGpuRectangleClip(
  context: CatalogGpuContext,
  features: CatalogFeatureCollection,
  clipCollection: CatalogFeatureCollection
): Promise<CatalogRunOutput> {
  const clipRing = getPolygonParts(clipCollection.features[0]?.geometry ?? null)[0]?.[0];
  const rectangle = clipRing ? getCatalogRectangle(clipRing) : null;
  if (!rectangle || clipCollection.features.length !== 1)
    throw new CatalogGpuUnsupportedError(
      'GPURectangleClip needs an axis-aligned rectangle; the CPU engine clips by any convex polygon'
    );
  const parameters = (session: CatalogGpuSession) =>
    session.input('rectangle', getGPURectangleClipParameterValues(rectangle), 'float32');
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
  const rowsBySource = new Map<number, {type: string; coordinates: unknown; part: number}[]>();
  const addRow = (sourceIndex: number, type: string, coordinates: unknown, part: number) => {
    const rows = rowsBySource.get(sourceIndex) ?? [];
    rows.push({type, coordinates, part});
    rowsBySource.set(sourceIndex, rows);
  };
  const polygons = packCatalogPolygons(features.features, {closing: 'drop'});
  if (polygons.sourceIndices.length)
    await withCatalogGpuSession(context.device, 'catalog-clip-polygons', async session => {
      const capacity = 8 * polygons.vertexCount + 16;
      const positions = session.output('positions', 'float32x2', capacity);
      const ringOffsets = session.output('ring-offsets', 'uint32', polygons.ringOffsets.length);
      const lineCount = session.output('line-count', 'uint32', 1);
      const overflow = session.output('overflow', 'uint32', 1);
      const requiredVertexCount = session.output('required-vertex-count', 'uint32', 1);
      session.graph.add(
        new GPURectangleClip({
          geometryType: 'polygons',
          positions: session.input('input-positions', polygons.positions, 'float32x2'),
          featureOffsets: session.input('feature-offsets', polygons.featureOffsets, 'uint32'),
          polygonOffsets: session.input('polygon-offsets', polygons.polygonOffsets, 'uint32'),
          ringOffsets: session.input('input-ring-offsets', polygons.ringOffsets, 'uint32'),
          parameters: parameters(session),
          output: {
            positions: positions.view,
            lineOffsets: ringOffsets.view,
            status: {
              count: lineCount.view,
              overflow: overflow.view,
              requiredVertexCount: requiredVertexCount.view
            }
          }
        })
      );
      session.run();
      if (await session.readScalar(overflow))
        throw new CatalogGpuUnsupportedError('GPURectangleClip output capacity overflowed');
      const outputPositions = await session.readFloat32(positions);
      const outputRings = await session.readUint32(ringOffsets);
      polygons.sourceIndices.forEach((sourceIndex, featureRow) => {
        const result: CatalogPosition[][][] = [];
        for (
          let polygon = polygons.featureOffsets[featureRow];
          polygon < polygons.featureOffsets[featureRow + 1];
          polygon++
        ) {
          const rings: CatalogPosition[][] = [];
          for (
            let ring = polygons.polygonOffsets[polygon];
            ring < polygons.polygonOffsets[polygon + 1];
            ring++
          ) {
            const vertices: CatalogPosition[] = [];
            for (let vertex = outputRings[ring]; vertex < outputRings[ring + 1]; vertex++)
              vertices.push([outputPositions[2 * vertex], outputPositions[2 * vertex + 1]]);
            // An empty clipped ring has fewer than three vertices and is dropped; an empty shell drops the polygon.
            if (vertices.length >= 3) rings.push(closeCatalogRing(vertices));
            else if (ring === polygons.polygonOffsets[polygon]) break;
          }
          if (rings.length) result.push(rings);
        }
        if (result.length === 1) addRow(sourceIndex, 'Polygon', result[0], 0);
        else if (result.length > 1) addRow(sourceIndex, 'MultiPolygon', result, 0);
      });
    });
  const lines = packCatalogLines(features.features);
  if (lines.sourceIndices.length)
    await withCatalogGpuSession(context.device, 'catalog-clip-lines', async session => {
      const lineCapacity = lines.vertexCount + 1;
      const capacity = 2 * lines.vertexCount + 16;
      const positions = session.output('positions', 'float32x2', capacity);
      const lineOffsets = session.output('line-offsets', 'uint32', lineCapacity + 1);
      const sourceLines = session.output('source-lines', 'uint32', lineCapacity);
      const lineCount = session.output('line-count', 'uint32', 1);
      const overflow = session.output('overflow', 'uint32', 1);
      const requiredVertexCount = session.output('required-vertex-count', 'uint32', 1);
      session.graph.add(
        new GPURectangleClip({
          geometryType: 'lines',
          positions: session.input('input-positions', lines.positions, 'float32x2'),
          lineOffsets: session.input('input-offsets', lines.lineOffsets, 'uint32'),
          parameters: parameters(session),
          output: {
            positions: positions.view,
            lineOffsets: lineOffsets.view,
            sourceLineIndices: sourceLines.view,
            status: {
              count: lineCount.view,
              overflow: overflow.view,
              requiredVertexCount: requiredVertexCount.view
            }
          }
        })
      );
      session.run();
      if (await session.readScalar(overflow))
        throw new CatalogGpuUnsupportedError('GPURectangleClip output capacity overflowed');
      const outputLineCount = await session.readScalar(lineCount);
      const outputPositions = await session.readFloat32(positions);
      const outputOffsets = await session.readUint32(lineOffsets);
      const outputSources = await session.readUint32(sourceLines);
      const partCounters = new Map<number, number>();
      for (let line = 0; line < outputLineCount; line++) {
        const sourceIndex = lines.lineSourceIndices[outputSources[line]];
        const piece: CatalogPosition[] = [];
        for (let vertex = outputOffsets[line]; vertex < outputOffsets[line + 1]; vertex++)
          piece.push([outputPositions[2 * vertex], outputPositions[2 * vertex + 1]]);
        if (piece.length < 2) continue;
        const part = partCounters.get(sourceIndex) ?? 0;
        partCounters.set(sourceIndex, part + 1);
        addRow(sourceIndex, 'LineString', piece, part);
      }
    });
  let outsideCount = 0;
  let outputIndex = 0;
  features.features.forEach((feature, sourceIndex) => {
    const rows = rowsBySource.get(sourceIndex);
    if (!rows?.length) {
      outsideCount++;
      return;
    }
    for (const row of rows)
      appendCatalogRow(table, {
        output_id: `${sourceIndex}:${outputIndex++}`,
        source_index: sourceIndex,
        source_id: getFeatureIdentity(feature),
        part_index: row.part,
        geometry_type: row.type,
        geometry: makeCatalogGeometryText(row.type, row.coordinates)
      });
  });
  const diagnostics: CatalogDiagnostic[] = [];
  if (outsideCount)
    addCatalogDiagnostic(
      diagnostics,
      'feature-outside-clip',
      'info',
      'Features with no part inside the clip rectangle (or unsupported geometry) produce no output.',
      outsideCount
    );
  addCatalogDiagnostic(
    diagnostics,
    'planar-edges',
    'info',
    'Edges are straight in coordinate space; the clip does not follow great circles. Concave polygons can keep zero-width bridges along the clip edges.'
  );
  return {
    tables: [table],
    diagnostics,
    summary: {outputFeatureCount: table.rowCount, droppedFeatureCount: outsideCount},
    dispatchedContributors: ['GPURectangleClip']
  };
}

// ---------------------------------------------------------------------------------------------
// Polygon overlay outputs shared by buffer and overlay
// ---------------------------------------------------------------------------------------------

type OverlayCapacities = {
  ringCapacity: number;
  vertexCapacity: number;
  boundaryCapacity: number;
  intersections: number;
  nodedSegments: number;
};

function createOverlayOutputs(session: CatalogGpuSession, capacities: OverlayCapacities) {
  const positions = session.output('result-positions', 'float32x2', capacities.vertexCapacity);
  const featureOffsets = session.output('result-features', 'uint32', capacities.ringCapacity + 1);
  const polygonOffsets = session.output('result-polygons', 'uint32', capacities.ringCapacity + 1);
  const ringOffsets = session.output('result-rings', 'uint32', capacities.ringCapacity + 1);
  const scalar = (name: string) => session.output(name, 'uint32', 1);
  const count = scalar('count');
  const requiredRingCount = scalar('required-ring-count');
  const overflow = scalar('overflow');
  const candidateOverflow = scalar('candidate-overflow');
  const boundaryCount = scalar('boundary-count');
  const boundaryRequired = scalar('boundary-required');
  const boundaryOverflow = scalar('boundary-overflow');
  const boundaryCandidateOverflow = scalar('boundary-candidate-overflow');
  const output: GPUPolygonOverlayOutput = {
    geometry: {
      kind: 'polygons',
      positions: positions.view,
      featureOffsets: featureOffsets.view,
      polygonOffsets: polygonOffsets.view,
      ringOffsets: ringOffsets.view
    },
    status: {
      count: count.view,
      requiredRingCount: requiredRingCount.view,
      overflow: overflow.view,
      candidateOverflow: candidateOverflow.view
    },
    boundary: {
      endpoints: session.output('boundary-endpoints', 'float32x4', capacities.boundaryCapacity)
        .view,
      operandIds: session.output('boundary-operands', 'uint32', capacities.boundaryCapacity).view,
      sourceFeatureIds: session.output('boundary-features', 'uint32', capacities.boundaryCapacity)
        .view,
      status: {
        count: boundaryCount.view,
        requiredCount: boundaryRequired.view,
        overflow: boundaryOverflow.view,
        candidateOverflow: boundaryCandidateOverflow.view
      }
    }
  };
  return {
    output,
    positions,
    featureOffsets,
    polygonOffsets,
    ringOffsets,
    count,
    requiredRingCount,
    overflow,
    candidateOverflow,
    boundaryRequired,
    boundaryOverflow,
    boundaryCandidateOverflow
  };
}

type OverlayRead = {
  polygons: CatalogPosition[][][];
  complete: boolean;
  requiredRings: number;
  requiredBoundary: number;
};

async function readOverlayOutputs(
  session: CatalogGpuSession,
  outputs: ReturnType<typeof createOverlayOutputs>,
  frame: CatalogFrame
): Promise<OverlayRead> {
  const flags = await Promise.all([
    session.readScalar(outputs.overflow),
    session.readScalar(outputs.candidateOverflow),
    session.readScalar(outputs.boundaryOverflow),
    session.readScalar(outputs.boundaryCandidateOverflow)
  ]);
  const requiredRings = await session.readScalar(outputs.requiredRingCount);
  const requiredBoundary = await session.readScalar(outputs.boundaryRequired);
  const complete = flags.every(flag => flag === 0);
  if (!complete) return {polygons: [], complete, requiredRings, requiredBoundary};
  const featureCount = await session.readScalar(outputs.count);
  if (featureCount === 0) return {polygons: [], complete, requiredRings, requiredBoundary};
  const featureOffsets = await session.readUint32(outputs.featureOffsets);
  const polygonCount = featureOffsets[featureCount];
  const polygonOffsets = await session.readUint32(outputs.polygonOffsets);
  const ringOffsets = await session.readUint32(outputs.ringOffsets);
  const positions = await session.readFloat32(outputs.positions);
  return {
    polygons: readCatalogPolygons(
      positions,
      ringOffsets,
      polygonOffsets,
      [0, polygonCount],
      frame.unproject
    ),
    complete,
    requiredRings,
    requiredBoundary
  };
}

/**
 * Runs `build` on a fresh session until the arrangement fits its capacities. Capacities grow from
 * the contributor's unclamped required counts; they are bounded by the device's storage binding.
 */
async function runOverlayGraph(
  context: CatalogGpuContext,
  id: string,
  frame: CatalogFrame,
  inputRows: number,
  build: (
    session: CatalogGpuSession,
    outputs: ReturnType<typeof createOverlayOutputs>,
    capacities: OverlayCapacities
  ) => void
): Promise<OverlayRead> {
  const maximumRows = getGPUBufferSurfaceMaximumBoundaryRowCount(context.device);
  let capacities: OverlayCapacities = {
    ringCapacity: Math.max(64, Math.ceil(inputRows / 8)),
    vertexCapacity: Math.max(1024, 2 * inputRows),
    boundaryCapacity: Math.max(1024, 2 * inputRows),
    intersections: Math.max(1024, 2 * inputRows),
    nodedSegments: Math.max(4096, 6 * inputRows)
  };
  for (let attempt = 0; attempt < 4; attempt++) {
    // Boundary rows plus intersections must fit one snap-rounding segment table.
    if (inputRows + capacities.intersections > maximumRows)
      throw new CatalogGpuUnsupportedError(
        'Input exceeds the snap-rounding capacity of this device'
      );
    const read = await withCatalogGpuSession(context.device, id, async session => {
      const outputs = createOverlayOutputs(session, capacities);
      build(session, outputs, capacities);
      session.run();
      return readOverlayOutputs(session, outputs, frame);
    });
    if (read.complete) return read;
    capacities = {
      ringCapacity: Math.max(capacities.ringCapacity * 2, read.requiredRings + 1),
      boundaryCapacity: Math.max(capacities.boundaryCapacity * 2, read.requiredBoundary + 1),
      vertexCapacity: Math.max(
        capacities.vertexCapacity * 2,
        read.requiredBoundary + read.requiredRings + 1
      ),
      intersections: capacities.intersections * 2,
      nodedSegments: capacities.nodedSegments * 2
    };
  }
  throw new CatalogGpuUnsupportedError('The polygon arrangement overflowed its capacities');
}

function getFrameTolerance(
  collections: readonly CatalogFeatureCollection[],
  frame: CatalogFrame
): number {
  let minimum = Infinity;
  let maximum = -Infinity;
  for (const collection of collections)
    for (const feature of collection.features)
      for (const position of [
        ...getPolygonParts(feature.geometry).flat(2),
        ...getLineParts(feature.geometry).flat()
      ]) {
        const [x, y] = frame.project(position);
        minimum = Math.min(minimum, x, y);
        maximum = Math.max(maximum, x, y);
      }
  const extent = Number.isFinite(minimum) ? maximum - minimum : 1;
  // The snap lattice is a power of two of at least 2^-22 of the extent; ask for a 2^-20 lattice.
  return Math.max(extent * 2 ** -20, 1e-9);
}

function getFramedArea(
  rings: readonly (readonly CatalogPosition[])[],
  frame: CatalogFrame
): number {
  return getCatalogPolygonArea(
    rings.map(ring => ring.map(position => frame.project(position) as CatalogPosition))
  );
}

// ---------------------------------------------------------------------------------------------
// Buffers
// ---------------------------------------------------------------------------------------------

export type CatalogBufferParameters = {
  distance: number;
  quadSegments: number;
  joinStyle: 'round' | 'mitre' | 'bevel';
  capStyle: 'round' | 'flat' | 'square';
  mitreLimit: number;
  dissolve: boolean;
  coordinateSystem: CatalogCoordinateSystem;
};

const MAXIMUM_UNDISSOLVED_BUFFER_FEATURES = 256;

/** Buffers lines or polygons with GPUBufferSurface (planar; degrees go through a local azimuthal frame). */
export async function runGpuBuffer(
  context: CatalogGpuContext,
  geometryKind: 'line' | 'polygon',
  collection: CatalogFeatureCollection,
  parameters: CatalogBufferParameters
): Promise<CatalogRunOutput> {
  const frame = createCatalogFrame(parameters.coordinateSystem, [collection]);
  const diagnostics: CatalogDiagnostic[] = [];
  const table = makeCatalogTable(
    'buffers',
    geometryKind === 'line' ? 'Line buffers' : 'Polygon buffers',
    {
      buffer_id: 'string',
      component_index: 'int32',
      source_index: 'int32',
      source_id: 'string',
      distance: 'float64',
      area: 'float64',
      ring_count: 'int32',
      hole_count: 'int32',
      geometry: 'string'
    },
    'geometry'
  );
  const tolerance = getFrameTolerance([collection], frame);
  const rowsPerVertex = getGPUOffsetCurveRowsPerVertex(
    parameters.joinStyle,
    parameters.quadSegments,
    'buffer'
  );
  const capRows =
    parameters.capStyle === 'round'
      ? 2 * parameters.quadSegments - 1
      : parameters.capStyle === 'square'
        ? 2
        : 0;
  const bufferGroup = async (featureIndices: readonly number[]): Promise<CatalogPosition[][][]> => {
    const subset = {
      type: 'FeatureCollection' as const,
      features: featureIndices.map(index => collection.features[index])
    };
    if (geometryKind === 'line') {
      const packed = packCatalogLines(subset.features, {project: frame.project});
      if (!packed.lineSourceIndices.length) return [];
      const rows =
        2 * rowsPerVertex * packed.vertexCount +
        (2 * capRows + 1) * packed.lineSourceIndices.length;
      const read = await runOverlayGraph(
        context,
        'catalog-line-buffer',
        frame,
        rows,
        (session, outputs, capacities) => {
          session.graph.add(
            new GPUBufferSurface({
              geometry: {
                kind: 'lines',
                positions: session.input('positions', packed.positions, 'float32x2'),
                lineOffsets: session.input('line-offsets', packed.lineOffsets, 'uint32')
              },
              parameters: session.input(
                'parameters',
                Float32Array.of(parameters.distance, parameters.mitreLimit, 0, 0),
                'float32'
              ),
              joinStyle: parameters.joinStyle,
              capStyle: parameters.capStyle,
              quadSegments: parameters.quadSegments,
              vertexTolerance: tolerance,
              capacity: {
                intersections: capacities.intersections,
                nodedSegments: capacities.nodedSegments
              },
              output: outputs.output
            })
          );
        }
      );
      return read.polygons;
    }
    const packed = packCatalogPolygons(subset.features, {project: frame.project, closing: 'drop'});
    if (!packed.sourceIndices.length) return [];
    const rows = (1 + 2 * rowsPerVertex) * packed.vertexCount + 3 * packed.ringCount;
    const read = await runOverlayGraph(
      context,
      'catalog-polygon-buffer',
      frame,
      rows,
      (session, outputs, capacities) => {
        session.graph.add(
          new GPUBufferSurface({
            geometry: {
              kind: 'polygons',
              positions: session.input('positions', packed.positions, 'float32x2'),
              featureOffsets: session.input('feature-offsets', packed.featureOffsets, 'uint32'),
              polygonOffsets: session.input('polygon-offsets', packed.polygonOffsets, 'uint32'),
              ringOffsets: session.input('ring-offsets', packed.ringOffsets, 'uint32')
            },
            parameters: session.input(
              'parameters',
              Float32Array.of(parameters.distance, parameters.mitreLimit, 0, 0),
              'float32'
            ),
            joinStyle: parameters.joinStyle,
            capStyle: parameters.capStyle,
            quadSegments: parameters.quadSegments,
            vertexTolerance: tolerance,
            capacity: {
              intersections: capacities.intersections,
              nodedSegments: capacities.nodedSegments
            },
            output: outputs.output
          })
        );
      }
    );
    return read.polygons;
  };
  const eligible = collection.features
    .map((feature, index) => ({index, kind: getFeatureKind(feature)}))
    .filter(entry => entry.kind === geometryKind)
    .map(entry => entry.index);
  const skipped = collection.features.length - eligible.length;
  if (skipped)
    addCatalogDiagnostic(
      diagnostics,
      'feature-skipped',
      'warning',
      `Features that are not ${geometryKind} geometry produce no buffer.`,
      skipped
    );
  let componentCount = 0;
  if (parameters.dissolve) {
    const polygons = await bufferGroup(eligible);
    polygons.forEach((rings, componentIndex) => {
      componentCount++;
      appendCatalogRow(table, {
        buffer_id: `all:${componentIndex}`,
        component_index: componentIndex,
        distance: parameters.distance,
        area: getFramedArea(rings, frame),
        ring_count: rings.length,
        hole_count: rings.length - 1,
        geometry: makeCatalogGeometryText('Polygon', rings)
      });
    });
    addCatalogDiagnostic(
      diagnostics,
      'buffers-dissolved',
      'info',
      'All buffers are dissolved into one union; component rows do not map to single sources. Turn off dissolve for per-feature buffers.'
    );
  } else {
    // Undissolved buffers need one arrangement per source feature.
    if (eligible.length > MAXIMUM_UNDISSOLVED_BUFFER_FEATURES)
      throw new KeplerAllocationError('capacity', {
        message: 'Feature capacity exceeded for per-feature buffers'
      });
    for (const sourceIndex of eligible) {
      const polygons = await bufferGroup([sourceIndex]);
      polygons.forEach((rings, componentIndex) => {
        componentCount++;
        appendCatalogRow(table, {
          buffer_id: `${sourceIndex}:${componentIndex}`,
          component_index: componentIndex,
          source_index: sourceIndex,
          source_id: getFeatureIdentity(collection.features[sourceIndex]),
          distance: parameters.distance,
          area: getFramedArea(rings, frame),
          ring_count: rings.length,
          hole_count: rings.length - 1,
          geometry: makeCatalogGeometryText('Polygon', rings)
        });
      });
    }
  }
  addCatalogDiagnostic(
    diagnostics,
    'buffer-frame',
    'info',
    frame.kind === 'identity'
      ? 'Distances are in coordinate units.'
      : 'Distances are meters in a local azimuthal-equidistant frame around the input center; area is measured in that frame.'
  );
  if (geometryKind === 'line' && parameters.distance <= 0)
    addCatalogDiagnostic(
      diagnostics,
      'non-positive-line-distance',
      'warning',
      'A non-positive line buffer distance produces an empty surface.'
    );
  return {
    tables: [table],
    diagnostics,
    summary: {componentCount, distance: parameters.distance},
    dispatchedContributors: ['GPUBufferSurface', 'GPUPolygonOverlay']
  };
}

// ---------------------------------------------------------------------------------------------
// Overlay
// ---------------------------------------------------------------------------------------------

/** Boolean overlay of two polygon layers; each layer is treated as one dissolved operand. */
export async function runGpuPolygonOverlay(
  context: CatalogGpuContext,
  left: CatalogFeatureCollection,
  right: CatalogFeatureCollection,
  parameters: {
    operation: GPUPolygonOverlayOperation;
    coordinateSystem: CatalogCoordinateSystem;
    fillRule: 'even-odd' | 'nonzero';
  }
): Promise<CatalogRunOutput> {
  const frame = createCatalogFrame(parameters.coordinateSystem, [left, right]);
  const packedLeft = packCatalogPolygons(left.features, {project: frame.project, closing: 'drop'});
  const packedRight = packCatalogPolygons(right.features, {
    project: frame.project,
    closing: 'drop'
  });
  const diagnostics: CatalogDiagnostic[] = [];
  const table = makeCatalogTable(
    'overlay',
    `Overlay ${parameters.operation}`,
    {
      overlay_id: 'string',
      component_index: 'int32',
      area: 'float64',
      ring_count: 'int32',
      hole_count: 'int32',
      geometry: 'string'
    },
    'geometry'
  );
  let polygons: CatalogPosition[][][] = [];
  const copyOperand = (operand: CatalogFeatureCollection) => {
    for (const feature of operand.features)
      for (const rings of getPolygonParts(feature.geometry))
        polygons.push(rings as CatalogPosition[][]);
  };
  if (packedLeft.sourceIndices.length && packedRight.sourceIndices.length) {
    const read = await runOverlayGraph(
      context,
      `catalog-overlay-${parameters.operation}`,
      frame,
      packedLeft.vertexCount + packedRight.vertexCount,
      (session, outputs, capacities) => {
        session.graph.add(
          new GPUPolygonOverlay({
            operation: parameters.operation,
            fillRule: parameters.fillRule,
            left: {
              kind: 'polygons',
              positions: session.input('left-positions', packedLeft.positions, 'float32x2'),
              featureOffsets: session.input('left-features', packedLeft.featureOffsets, 'uint32'),
              polygonOffsets: session.input('left-polygons', packedLeft.polygonOffsets, 'uint32'),
              ringOffsets: session.input('left-rings', packedLeft.ringOffsets, 'uint32')
            },
            right: {
              kind: 'polygons',
              positions: session.input('right-positions', packedRight.positions, 'float32x2'),
              featureOffsets: session.input('right-features', packedRight.featureOffsets, 'uint32'),
              polygonOffsets: session.input('right-polygons', packedRight.polygonOffsets, 'uint32'),
              ringOffsets: session.input('right-rings', packedRight.ringOffsets, 'uint32')
            },
            capacity: {
              intersections: capacities.intersections,
              nodedSegments: capacities.nodedSegments
            },
            vertexTolerance: getFrameTolerance([left, right], frame),
            output: outputs.output
          })
        );
      }
    );
    polygons = read.polygons;
  } else if (parameters.operation !== 'intersection') {
    // One empty operand: union and symmetric difference keep the other layer, difference keeps the left.
    if (packedLeft.sourceIndices.length) copyOperand(left);
    else if (parameters.operation !== 'difference') copyOperand(right);
    if (polygons.length)
      addCatalogDiagnostic(
        diagnostics,
        'operand-empty',
        'warning',
        'One layer has no polygons; the other layer is returned as is and is not dissolved.'
      );
  }
  polygons.forEach((rings, componentIndex) =>
    appendCatalogRow(table, {
      overlay_id: `${parameters.operation}:${componentIndex}`,
      component_index: componentIndex,
      area: getFramedArea(rings, frame),
      ring_count: rings.length,
      hole_count: rings.length - 1,
      geometry: makeCatalogGeometryText('Polygon', rings)
    })
  );
  addCatalogDiagnostic(
    diagnostics,
    'layers-dissolved',
    'info',
    'Each layer is one operand: overlapping polygons within a layer merge, and result components do not carry source feature identity.'
  );
  addCatalogDiagnostic(
    diagnostics,
    'overlay-lattice',
    'info',
    frame.kind === 'identity'
      ? 'Vertices snap to a lattice about 2^-20 of the extent.'
      : 'Computed in a local azimuthal-equidistant frame; vertices snap to a lattice about 2^-20 of the extent.'
  );
  return {
    tables: [table],
    diagnostics,
    summary: {componentCount: polygons.length, operation: parameters.operation},
    dispatchedContributors: ['GPUPolygonOverlay']
  };
}

// ---------------------------------------------------------------------------------------------
// Topology repair
// ---------------------------------------------------------------------------------------------

/** Rebuilds valid polygons per source feature with GPUMakeValid (even-odd source parity). */
export async function runGpuTopologyRepair(
  context: CatalogGpuContext,
  polygons: CatalogFeatureCollection,
  parameters: {coordinateSystem: CatalogCoordinateSystem}
): Promise<CatalogRunOutput> {
  const frame = createCatalogFrame(parameters.coordinateSystem, [polygons]);
  // GPUMakeValid consumes explicit closing vertices; dropping them leaves every ring open.
  const packed = packCatalogPolygons(polygons.features, {project: frame.project, closing: 'keep'});
  const repairedByFeature = new Map<number, CatalogPosition[][][]>();
  let attemptCapacity = Math.max(256, 4 * packed.vertexCount);
  let complete = !packed.sourceIndices.length;
  const tolerance = getFrameTolerance([polygons], frame);
  for (let attempt = 0; attempt < 4 && !complete; attempt++) {
    const segments = attemptCapacity;
    const intersections = Math.max(256, attemptCapacity / 2);
    const rings = Math.max(64, 2 * packed.ringCount + 16);
    const vertices = 4 * attemptCapacity;
    const result = await withCatalogGpuSession(
      context.device,
      'catalog-make-valid',
      async session => {
        const directed = 2 * segments;
        const scalar = (name: string) => session.output(name, 'uint32', 1);
        const nodedStatus = {
          count: scalar('noded-count'),
          requiredCount: scalar('noded-required'),
          overflow: scalar('noded-overflow'),
          candidateOverflow: scalar('noded-candidate-overflow')
        };
        const directedStatus = {
          count: scalar('directed-count'),
          requiredCount: scalar('directed-required'),
          overflow: scalar('directed-overflow'),
          candidateOverflow: scalar('directed-candidate-overflow')
        };
        const status = {
          count: scalar('count'),
          requiredRingCount: scalar('required-rings'),
          overflow: scalar('overflow'),
          candidateOverflow: scalar('candidate-overflow')
        };
        const diagnosticScalars = [
          scalar('closed-rings'),
          scalar('open-chains'),
          scalar('cut-edges'),
          scalar('dangles'),
          scalar('unused-edges')
        ];
        const outPositions = session.output('out-positions', 'float32x2', vertices);
        const outFeatures = session.output('out-features', 'uint32', rings + 1);
        const outPolygons = session.output('out-polygons', 'uint32', rings + 1);
        const outRings = session.output('out-rings', 'uint32', rings + 1);
        const outSources = session.output('out-sources', 'uint32', rings);
        session.graph.add(
          new GPUMakeValid({
            intersectionCapacity: intersections,
            precision: {
              vertexTolerance: tolerance,
              predicates: 'exact-or-uncertain',
              storage: 'float32'
            },
            polygons: {
              kind: 'polygons',
              positions: session.input('positions', packed.positions, 'float32x2'),
              featureOffsets: session.input('feature-offsets', packed.featureOffsets, 'uint32'),
              polygonOffsets: session.input('polygon-offsets', packed.polygonOffsets, 'uint32'),
              ringOffsets: session.input('ring-offsets', packed.ringOffsets, 'uint32'),
              sourceIds: session.input(
                'source-ids',
                Uint32Array.from({length: packed.sourceIndices.length}, (_, index) => index),
                'uint32'
              )
            },
            output: {
              nodedSegments: {
                endpoints: session.output('noded-endpoints', 'float32x4', segments).view,
                sourceFeatureIds: session.output('noded-features', 'uint32', segments).view,
                sourceRingIds: session.output('noded-rings', 'uint32', segments).view,
                sourceEdgeIds: session.output('noded-edges', 'uint32', segments).view,
                sourceStartParameters: session.output('noded-starts', 'float32', segments).view,
                sourceEndParameters: session.output('noded-ends', 'float32', segments).view,
                status: {
                  count: nodedStatus.count.view,
                  requiredCount: nodedStatus.requiredCount.view,
                  overflow: nodedStatus.overflow.view,
                  candidateOverflow: nodedStatus.candidateOverflow.view
                }
              },
              directedEdges: {
                endpoints: session.output('directed-endpoints', 'float32x4', directed).view,
                sourceSegmentIds: session.output('directed-segments', 'uint32', directed).view,
                sourceFeatureIds: session.output('directed-features', 'uint32', directed).view,
                twinIds: session.output('directed-twins', 'uint32', directed).view,
                ringIds: session.output('directed-rings', 'uint32', directed).view,
                flags: session.output('directed-flags', 'uint32', directed).view,
                status: {
                  count: directedStatus.count.view,
                  requiredCount: directedStatus.requiredCount.view,
                  overflow: directedStatus.overflow.view,
                  candidateOverflow: directedStatus.candidateOverflow.view
                }
              },
              polygons: {
                kind: 'polygons',
                positions: outPositions.view,
                featureOffsets: outFeatures.view,
                polygonOffsets: outPolygons.view,
                ringOffsets: outRings.view,
                sourceIds: outSources.view
              },
              diagnostics: {
                edgeClasses: session.output('edge-classes', 'uint32', segments).view,
                closedRingCount: diagnosticScalars[0].view,
                openChainCount: diagnosticScalars[1].view,
                cutEdgeCount: diagnosticScalars[2].view,
                dangleCount: diagnosticScalars[3].view,
                unusedEdgeCount: diagnosticScalars[4].view
              },
              status: {
                count: status.count.view,
                requiredRingCount: status.requiredRingCount.view,
                overflow: status.overflow.view,
                candidateOverflow: status.candidateOverflow.view
              }
            }
          })
        );
        session.run();
        const overflow =
          (await session.readScalar(status.overflow)) |
          (await session.readScalar(status.candidateOverflow));
        const polygonCount = await session.readScalar(status.count);
        const polygonRows = await session.readUint32(outPolygons);
        const ringCount = polygonRows[polygonCount];
        const ringRows = await session.readUint32(outRings);
        const vertexCount = ringRows[ringCount];
        const sources = await session.readUint32(outSources);
        const positions = await session.readFloat32(outPositions);
        return {
          overflow: overflow !== 0 || vertexCount > vertices || ringCount > rings,
          polygonCount,
          polygonRows,
          ringRows,
          sources,
          positions
        };
      }
    );
    if (result.overflow) {
      attemptCapacity *= 4;
      continue;
    }
    complete = true;
    for (let polygon = 0; polygon < result.polygonCount; polygon++) {
      const rings = readCatalogPolygons(
        result.positions,
        result.ringRows,
        result.polygonRows,
        [polygon, polygon + 1],
        frame.unproject
      )[0];
      if (!rings) continue;
      const featureRow = result.sources[polygon];
      const sourceIndex = packed.sourceIndices[featureRow] ?? packed.sourceIndices[0];
      const list = repairedByFeature.get(sourceIndex) ?? [];
      list.push(rings);
      repairedByFeature.set(sourceIndex, list);
    }
  }
  if (!complete) throw new CatalogGpuUnsupportedError('GPUMakeValid overflowed its capacities');
  const table = makeCatalogTable(
    'repaired',
    'Repaired polygons',
    {
      source_index: 'int32',
      source_id: 'string',
      status: 'string',
      polygon_count: 'int32',
      input_area: 'float64',
      repaired_area: 'float64',
      geometry: 'string'
    },
    'geometry'
  );
  const diagnostics: CatalogDiagnostic[] = [];
  let repairedCount = 0;
  let emptyCount = 0;
  polygons.features.forEach((feature, sourceIndex) => {
    if (!packed.sourceIndices.includes(sourceIndex)) return;
    const repaired = repairedByFeature.get(sourceIndex) ?? [];
    const inputArea = getPolygonParts(feature.geometry).reduce(
      (sum, rings) => sum + getFramedArea(rings as CatalogPosition[][], frame),
      0
    );
    const repairedArea = repaired.reduce((sum, rings) => sum + getFramedArea(rings, frame), 0);
    const changed =
      repaired.length !== getPolygonParts(feature.geometry).length ||
      Math.abs(repairedArea - inputArea) > 1e-6 * Math.max(1, Math.abs(inputArea));
    if (!repaired.length) emptyCount++;
    else if (changed) repairedCount++;
    appendCatalogRow(table, {
      source_index: sourceIndex,
      source_id: getFeatureIdentity(feature),
      status: !repaired.length ? 'empty' : changed ? 'repaired' : 'unchanged',
      polygon_count: repaired.length,
      input_area: inputArea,
      repaired_area: repairedArea,
      geometry: repaired.length
        ? repaired.length === 1
          ? makeCatalogGeometryText('Polygon', repaired[0])
          : makeCatalogGeometryText('MultiPolygon', repaired)
        : null
    });
  });
  const skipped = polygons.features.length - packed.sourceIndices.length;
  if (skipped)
    addCatalogDiagnostic(
      diagnostics,
      'feature-skipped',
      'warning',
      'Features without polygon parts are not repaired.',
      skipped
    );
  if (emptyCount)
    addCatalogDiagnostic(
      diagnostics,
      'repair-empty',
      'warning',
      'These features have no area covered an odd number of times after noding.',
      emptyCount
    );
  addCatalogDiagnostic(
    diagnostics,
    'repair-rule',
    'info',
    'Rings are noded and rebuilt with an even-odd source parity rule (GEOS linework make_valid); bow ties split into polygons and doubly covered lenses are dropped.'
  );
  return {
    tables: [table],
    diagnostics,
    summary: {featureCount: packed.sourceIndices.length, repairedCount, emptyCount},
    dispatchedContributors: ['GPUMakeValid']
  };
}

export {getFeatureKind};
