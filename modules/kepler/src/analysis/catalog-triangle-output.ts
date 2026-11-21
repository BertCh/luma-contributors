// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {
  makeCatalogGeometryText,
  appendCatalogRow,
  getFeatureIdentity,
  getPolygonParts,
  type CatalogFeature,
  type CatalogOutputTable,
  measureRingSignedArea,
  type CatalogPosition
} from './catalog-geojson';
import {projectRing, type PlanarPoint} from './catalog-platform3-a-shared';
import type {LocalFrame} from './catalog-local-frame';

/** Shared CPU/GPU publication schema; counted degenerate rows retain their identities. */
export const CATALOG_POLYGON_TRIANGLE_COLUMNS = {
  triangle_id: 'string',
  source_index: 'int32',
  source_id: 'string',
  part_index: 'int32',
  triangle_index: 'int32',
  area: 'float64',
  degeneracy: 'string',
  geometry: 'string'
} as const;

/** Classifies the original planar coordinates before restoring their output frame. */
export function serializeCatalogTriangle(
  corners: readonly CatalogPosition[],
  unproject: (x: number, y: number) => [number, number]
) {
  const area = Math.abs(measureRingSignedArea(corners, 'planar'));
  const restored = corners.map(point => unproject(point[0], point[1]));
  const distinct = restored.filter(
    (point, index) =>
      restored.findIndex(other => other[0] === point[0] && other[1] === point[1]) === index
  );
  const geometryType = area > 0 ? 'Polygon' : distinct.length > 1 ? 'LineString' : 'Point';
  return {
    area,
    degeneracy: area > 0 ? 'none' : distinct.length > 1 ? 'line' : 'point',
    geometry: makeCatalogGeometryText(
      geometryType,
      area > 0 ? [[...restored, restored[0]]] : distinct.length > 1 ? distinct : distinct[0]
    )
  };
}

/** Projects polygon parts without renumbering nonempty parts after an empty source part. */
export function projectCatalogPolygonParts(
  feature: CatalogFeature,
  frame: LocalFrame
): {
  partIndex: number;
  rings: PlanarPoint[][];
}[] {
  return getPolygonParts(feature.geometry)
    .map((polygon, partIndex) => ({
      partIndex,
      rings: polygon.map(ring => projectRing(ring, frame)).filter(ring => ring.length >= 3)
    }))
    .filter(part => part.rings.length > 0);
}

/** CPU triangle row publication: every counted triple, including lines and points, keeps its IDs. */
export function appendCatalogTriangleRows(
  table: CatalogOutputTable,
  triangles: readonly (readonly CatalogPosition[])[],
  feature: CatalogFeature,
  sourceIndex: number,
  partIndex: number,
  unproject: (x: number, y: number) => [number, number]
): void {
  triangles.forEach((triangle, triangleIndex) =>
    appendCatalogRow(table, {
      triangle_id: `${sourceIndex}:${partIndex}:${triangleIndex}`,
      source_index: sourceIndex,
      source_id: getFeatureIdentity(feature),
      part_index: partIndex,
      triangle_index: triangleIndex,
      ...serializeCatalogTriangle(triangle, unproject)
    })
  );
}
