// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUPointBuffer,
  getGPUPointBufferParameterValues
} from '@luma-contributors/analysis/geospatial-analysis';
import type {CatalogGpuContext, CatalogRunOutput} from './catalog-binding-types';
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
import {withCatalogGpuSession} from './catalog-gpu-session';

const POLE_LATITUDE_DEGREES = 89.999;

/**
 * GPU point buffer: GPUPointBuffer computes the ring vertex offsets (float32, relative to the
 * center); centers are added in float64 here, so rows match the CPU reference to float32 offset
 * precision. Row identity, drop rules and diagnostics mirror runPointBuffer.
 */
export async function runGpuPointBuffer(
  context: CatalogGpuContext,
  points: CatalogFeatureCollection,
  parameters: {
    radius: number;
    segmentsPerQuarter: number;
    coordinateSystem: CatalogCoordinateSystem;
  }
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
    const centerValues = new Float32Array(centers.length * 2);
    centers.forEach((center, index) => {
      centerValues[2 * index] = center[0];
      centerValues[2 * index + 1] = center[1];
    });
    offsets = await withCatalogGpuSession(context.device, 'catalog-point-buffer', async session => {
      const output = session.output('offsets', 'float32x2', centers.length * vertexCount);
      session.graph.add(
        new GPUPointBuffer({
          points: session.input('points', centerValues, 'float32x2'),
          parameters: session.input(
            'parameters',
            getGPUPointBufferParameterValues(parameters.radius),
            'float32'
          ),
          offsets: output.view,
          verticesPerRing: vertexCount,
          coordinateSystem: isGeographic ? 'geographic' : 'planar'
        })
      );
      session.run();
      return session.readFloat32(output);
    });
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
