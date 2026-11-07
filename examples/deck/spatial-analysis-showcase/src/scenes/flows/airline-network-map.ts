// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {COORDINATE_SYSTEM, type Layer} from '@deck.gl/core';
import {buildPolygonMesh} from '../../cartography/polygon-mesh';
import type {GeoJsonCollection} from '../../data/loaders';
import {SpatialAnalysisPolygonLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {createPolygonMeshBuffers} from '../../engine/polygon-buffers';
import type {SpatialAnalysisResources} from '../../engine/resources';
import {projectNetworkDisplayPosition} from './airline-network-projection';

/** Natural Earth ground in the same display projection as airports and routes. */
export function createNetworkMap(resources: SpatialAnalysisResources, land: GeoJsonCollection) {
  const boundary: number[][] = [];
  for (let latitude = -90; latitude <= 90; latitude += 2) boundary.push([-180, latitude]);
  for (let longitude = -178; longitude <= 180; longitude += 2) boundary.push([longitude, 90]);
  for (let latitude = 88; latitude >= -90; latitude -= 2) boundary.push([180, latitude]);
  for (let longitude = 178; longitude >= -180; longitude -= 2) boundary.push([longitude, -90]);
  const ocean: GeoJsonCollection = {
    type: 'FeatureCollection',
    features: [
      {type: 'Feature', properties: {}, geometry: {type: 'Polygon', coordinates: [boundary]}}
    ]
  };
  const oceanBuffers = createPolygonMeshBuffers(
    resources,
    buildPolygonMesh(ocean, projectNetworkDisplayPosition),
    'map-ocean'
  );
  const landBuffers = createPolygonMeshBuffers(
    resources,
    buildPolygonMesh(land, projectNetworkDisplayPosition),
    'map-land'
  );
  return (): Layer[] => {
    const style = {coordinateSystem: COORDINATE_SYSTEM.LNGLAT};
    return [
      new SpatialAnalysisPolygonLayer({
        id: 'network-ocean',
        ...style,
        triangles: oceanBuffers.triangles,
        features: oceanBuffers.triangleFeatures,
        vertexCount: oceanBuffers.vertexCount,
        color: [10, 16, 24, 255]
      }),
      new SpatialAnalysisPolygonLayer({
        id: 'network-land',
        ...style,
        triangles: landBuffers.triangles,
        features: landBuffers.triangleFeatures,
        vertexCount: landBuffers.vertexCount,
        color: [31, 41, 52, 255]
      }),
      ...[landBuffers, oceanBuffers].map(
        (buffers, index) =>
          new SpatialAnalysisSegmentLayer({
            id: `network-map-outline-${index}`,
            ...style,
            segments: buffers.outline,
            instanceCount: buffers.outlineCount,
            widthPixels: 0.65,
            color: [75, 93, 111, 150]
          })
      )
    ];
  };
}
