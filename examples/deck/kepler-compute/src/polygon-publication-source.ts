// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type * as arrow from 'apache-arrow';
import {
  POLYGON_CONTIGUITY_MAXIMUM_ROWS,
  POLYGON_CONTIGUITY_MAXIMUM_VERTICES,
  type PreparedKeplerGeometrySource
} from './geometry-preparation';
import type {PolygonAnalysisSource} from './analysis-polygon-weights-cache';

/** Retains existing original-coordinate Arrow polygon vectors without decoding or uploading again. */
export function getPolygonAnalysisSource(
  geometry: PreparedKeplerGeometrySource,
  column: string
): {source?: PolygonAnalysisSource; byteLength: number; reason?: string} {
  if (!geometry.metadata.featureTypes.polygon)
    return {byteLength: 0, reason: 'Polygon contiguity requires polygon geometry in the source.'};
  if (
    geometry.metadata.rowCount < 1 ||
    geometry.metadata.rowCount > POLYGON_CONTIGUITY_MAXIMUM_ROWS ||
    geometry.batches.length > POLYGON_CONTIGUITY_MAXIMUM_ROWS
  )
    return {
      byteLength: 0,
      reason: `Polygon analysis supports at most ${POLYGON_CONTIGUITY_MAXIMUM_ROWS} source rows and batches.`
    };
  // Outlines carry the original rings, including closure vertices; this is a conservative bound.
  const vertexCount = geometry.batches.reduce(
    (count, batch) => count + batch.polygonOutlines.positions.length / 3,
    0
  );
  if (vertexCount > POLYGON_CONTIGUITY_MAXIMUM_VERTICES)
    return {
      byteLength: 0,
      reason: `Polygon analysis supports at most ${POLYGON_CONTIGUITY_MAXIMUM_VERTICES} original ring vertices.`
    };
  const batches = geometry.batches.map(({id, rowCount, sourceRowOffset, polygons}) => ({
    id,
    rowCount,
    sourceRowOffset,
    polygons
  }));
  const byteLength = batches.reduce(
    (bytes, batch) =>
      bytes +
      64 +
      batch.id.length * 2 +
      batch.polygons.data.reduce((sum, data) => sum + getPolygonDataBytes(data), 0),
    0
  );
  return {source: {column, source: {metadata: geometry.metadata, batches}}, byteLength};
}

function getPolygonDataBytes(data: arrow.Data): number {
  return (
    (data.values?.byteLength ?? 0) +
    (data.valueOffsets?.byteLength ?? 0) +
    (data.nullBitmap?.byteLength ?? 0) +
    (data.typeIds?.byteLength ?? 0) +
    (data.children ?? []).reduce((bytes, child) => bytes + getPolygonDataBytes(child), 0)
  );
}
