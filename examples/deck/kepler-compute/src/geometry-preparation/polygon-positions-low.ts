// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Vector} from 'apache-arrow';
import {makeGeoArrowColumnFromArrowVector} from '@luma.gl/graph/arrow';
import {materializeGeoArrowRows} from '@math.gl/arrow/geoarrow-utils';
import type {ArrowMultiPolygonType} from '@luma.gl/engine/arrow';

/**
 * Published `convertArrowPolygonToGPUVectors` emits float32 positions only. Kepler's fp64 sources
 * need the float32 residual of each tessellated vertex, so this walks the polygons in
 * tessellation order (rows, polygons, rings, closing vertex removed) and returns
 * `value - fround(value)` for XY, two values per vertex. Rings under three vertices are skipped
 * like the tessellator skips them. The walk must reproduce the tessellator vertex order exactly;
 * on a mismatch it returns zeros, which keeps float32 behaviour instead of misaligned residuals.
 */
export function makeKeplerPolygonPositionsLow(
  polygons: Vector<ArrowMultiPolygonType>,
  vertexCount: number
): Float32Array {
  const column = makeGeoArrowColumnFromArrowVector(polygons, {encoding: 'geoarrow.multipolygon'});
  const positionsLow = new Float32Array(vertexCount * 2);
  let vertexIndex = 0;
  const visitRing = (ring: number[][]) => {
    const closed =
      ring.length > 1 &&
      ring[0].length === ring[ring.length - 1].length &&
      ring[0].every((value, index) => value === ring[ring.length - 1][index]);
    const ringLength = closed ? ring.length - 1 : ring.length;
    if (ringLength < 3) return;
    for (let index = 0; index < ringLength; index++) {
      for (let component = 0; component < 2; component++) {
        const value = ring[index][component] ?? 0;
        if (vertexIndex < vertexCount) {
          positionsLow[vertexIndex * 2 + component] = value - Math.fround(value);
        }
      }
      vertexIndex++;
    }
  };
  const visitGeometry = (geometry: {type: string; [key: string]: unknown}) => {
    if (geometry.type === 'Polygon') {
      for (const ring of geometry['coordinates'] as number[][][]) visitRing(ring);
    } else if (geometry.type === 'MultiPolygon') {
      for (const polygon of geometry['coordinates'] as number[][][][])
        for (const ring of polygon) visitRing(ring);
    } else if (geometry.type === 'GeometryCollection') {
      for (const child of geometry['geometries'] as {type: string}[]) visitGeometry(child);
    }
  };
  for (const row of materializeGeoArrowRows(column)) if (row) visitGeometry(row as {type: string});
  if (vertexIndex !== vertexCount) return new Float32Array(vertexCount * 2);
  return positionsLow;
}
