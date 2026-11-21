// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUPointBuffer,
  getGPUPointBufferParameterValues
} from '@luma-contributors/analysis/geospatial-analysis';
import type {
  CatalogGpuContext,
  CatalogRunOutput,
  CatalogPreviewPlan
} from './catalog-binding-types';
import type {CatalogDataInput} from './catalog-data-input';
import {isCatalogFeatureInput} from './catalog-data-input';
import type {CatalogParameterValue} from './catalog-binding-types';
import {validateRequirement} from '@luma-contributors/analysis/analysis-core';
import {
  EARTH_RADIUS_METERS,
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
import {CatalogGpuSession, withCatalogGpuSession} from './catalog-gpu-session';

const POLE_LATITUDE_DEGREES = 89.999;

/**
 * GPU point buffer: GPUPointBuffer computes the ring vertex offsets (float32, relative to the
 * center); centers are added in float64 here, so rows match the CPU reference to float32 offset
 * precision. Row identity, drop rules and diagnostics mirror runPointBuffer.
 */
async function executeGpuPointBuffer(
  points: CatalogFeatureCollection,
  parameters: {
    radius: number;
    segmentsPerQuarter: number;
    coordinateSystem: CatalogCoordinateSystem;
  },
  readOffsets: (centers: readonly CatalogPosition[], vertexCount: number) => Promise<Float32Array>
): Promise<CatalogRunOutput> {
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
  const centers: CatalogPosition[] = [];
  const sourceIndices: number[] = [];
  const pointIndices: number[] = [];
  let droppedCount = 0;
  points.features.forEach((feature, sourceIndex) => {
    const positions = getPointPositions(feature.geometry);
    if (!positions.length) droppedCount++;
    positions.forEach((center, pointIndex) => {
      centers.push(center);
      sourceIndices.push(sourceIndex);
      pointIndices.push(pointIndex);
    });
  });
  const isGeographic = parameters.coordinateSystem === 'EPSG:4326';
  const radiusReachesEquatorialBand =
    isGeographic && parameters.radius / EARTH_RADIUS_METERS >= Math.PI / 2;
  let offsets: Float32Array = new Float32Array(0);
  if (centers.length && !radiusReachesEquatorialBand) {
    offsets = await readOffsets(centers, vertexCount);
  }
  let antimeridianCount = 0;
  centers.forEach((center, centerIndex) => {
    if (radiusReachesEquatorialBand) {
      droppedCount++;
      return;
    }
    const ring: CatalogPosition[] = [];
    let crossesAntimeridian = false;
    let reachesPole = false;
    for (let vertex = 0; vertex < vertexCount; vertex++) {
      const offsetIndex = 2 * (centerIndex * vertexCount + vertex);
      const longitude = center[0] + offsets[offsetIndex];
      const latitude = center[1] + offsets[offsetIndex + 1];
      if (isGeographic) {
        if (Math.abs(latitude) > POLE_LATITUDE_DEGREES) reachesPole = true;
        if (Math.abs(longitude) > 180) crossesAntimeridian = true;
      }
      ring.push([longitude, latitude]);
    }
    if (reachesPole) {
      droppedCount++;
      return;
    }
    if (crossesAntimeridian) antimeridianCount++;
    ring.push(ring[0]);
    appendCatalogRow(table, {
      buffer_id: `${sourceIndices[centerIndex]}:${pointIndices[centerIndex]}`,
      source_index: sourceIndices[centerIndex],
      source_id: getFeatureIdentity(points.features[sourceIndices[centerIndex]]),
      point_index: pointIndices[centerIndex],
      radius: parameters.radius,
      area: Math.abs(measureRingSignedArea(ring, parameters.coordinateSystem)),
      geometry: makeCatalogGeometryText('Polygon', [ring])
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
    summary: {outputFeatureCount: table.rowCount, verticesPerBuffer: vertexCount + 1},
    dispatchedContributors: ['GPUPointBuffer']
  };
}

type PointBufferParameters = {
  radius: number;
  segmentsPerQuarter: number;
  coordinateSystem: CatalogCoordinateSystem;
};

/** One shared graph layout for one-shot and retained point buffering; GPUPointBuffer has no scratch. */
function createPointBufferResources(
  session: CatalogGpuSession,
  centers: readonly CatalogPosition[],
  parameters: PointBufferParameters
) {
  const centerValues = Float32Array.from(centers.flatMap(center => [center[0], center[1]]));
  const output = session.output(
    'offsets',
    'float32x2',
    centers.length * parameters.segmentsPerQuarter * 4
  );
  const parameterView = session.input(
    'parameters',
    getGPUPointBufferParameterValues(parameters.radius),
    'float32'
  );
  session.graph.add(
    new GPUPointBuffer({
      points: session.input('points', centerValues, 'float32x2'),
      parameters: parameterView,
      offsets: output.view,
      verticesPerRing: parameters.segmentsPerQuarter * 4,
      spatialContext:
        parameters.coordinateSystem === 'EPSG:4326'
          ? {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}
          : {coordinateSpace: 'planar', metric: 'native'}
    })
  );
  return {output, parameterView};
}

export async function runGpuPointBuffer(
  context: CatalogGpuContext,
  points: CatalogFeatureCollection,
  parameters: PointBufferParameters
): Promise<CatalogRunOutput> {
  context.signal?.throwIfAborted();
  const result = await executeGpuPointBuffer(points, parameters, centers =>
    withCatalogGpuSession(context.device, 'catalog-point-buffer', async session => {
      const {output} = createPointBufferResources(session, centers, parameters);
      session.run(context.signal);
      const offsets = await session.readFloat32(output);
      context.signal?.throwIfAborted();
      return offsets;
    })
  );
  context.signal?.throwIfAborted();
  return result;
}

/** Radius is live; input revision, ring resolution and coordinate system require a new compilation. */
export function planGpuPointBufferPreview(
  inputs: Readonly<Record<string, CatalogDataInput>>,
  values: Readonly<Record<string, CatalogParameterValue>>
): CatalogPreviewPlan {
  const points = inputs['points'];
  validateRequirement(
    'PointBufferPreview',
    isCatalogFeatureInput(points),
    'points must be features'
  );
  const features = points as CatalogFeatureCollection;
  const pointCount = features.features.reduce(
    (total, feature) => total + getPointPositions(feature.geometry).length,
    0
  );
  const parameters: PointBufferParameters = {
    radius: values['radius'] as number,
    segmentsPerQuarter: values['segmentsPerQuarter'] as number,
    coordinateSystem: values['coordinateSystem'] as CatalogCoordinateSystem
  };
  const vertexCount = parameters.segmentsPerQuarter * 4;
  const outputBytes = Math.max(pointCount * vertexCount * 8, 4);
  // readAsync allocates a GPU staging buffer as large as the offsets, in addition to storage.
  const gpuByteLength = Math.max(pointCount * 8, 8) + 8 + 2 * outputBytes;
  // Two CPU table copies, coordinates, JSON geometry and one readback plus its aligned copy.
  // A JS number string needs at most 25 UTF-16 code units including separators. The row
  // allowance covers properties, identities and array storage; source strings are included below.
  const identityBytes = features.features.reduce(
    (total, feature) =>
      total +
      String(getFeatureIdentity(feature)).length * 2 * getPointPositions(feature.geometry).length,
    0
  );
  const cpuByteLength =
    2 * outputBytes + pointCount * (vertexCount * 256 + 4096) + 2 * identityBytes;
  return {
    key: JSON.stringify([parameters.segmentsPerQuarter, parameters.coordinateSystem]),
    gpuByteLength,
    cpuByteLength,
    create: async context => {
      const session = new CatalogGpuSession(
        context.device,
        'catalog-point-buffer-preview',
        gpuByteLength
      );
      let resources: ReturnType<typeof createPointBufferResources> | undefined;
      return {
        initialize: async signal => {
          signal.throwIfAborted();
          const centers = features.features.flatMap(feature => getPointPositions(feature.geometry));
          resources = createPointBufferResources(session, centers, parameters);
          await session.compile(signal);
        },
        execute: async (nextValues, signal) => {
          signal.throwIfAborted();
          validateRequirement(
            'PointBufferPreview',
            Boolean(resources),
            'initialize must complete before execute'
          );
          const {parameterView, output} = resources!;
          const nextParameters = {...parameters, radius: nextValues['radius'] as number};
          session.writeInput(
            parameterView,
            getGPUPointBufferParameterValues(nextParameters.radius)
          );
          const result = await executeGpuPointBuffer(features, nextParameters, async () => {
            session.run(signal);
            try {
              const offsets = await session.readFloat32(output);
              signal.throwIfAborted();
              return offsets;
            } finally {
              await session.waitForCompletion();
            }
          });
          signal.throwIfAborted();
          return result;
        },
        destroy: () => session.dispose()
      };
    }
  };
}
