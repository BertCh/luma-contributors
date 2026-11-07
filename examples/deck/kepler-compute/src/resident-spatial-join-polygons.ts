// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {DataType, Vector} from 'apache-arrow';
import type {PreparedPolygonTopologySource} from '@luma.gl/engine/arrow';

export type PackedSpatialJoinPolygons = {
  /** Original XY rounded to Float32; rings close implicitly. */
  positions: Float32Array;
  ringOffsets: Uint32Array;
  polygonOffsets: Uint32Array;
  featureOffsets: Uint32Array;
  rowCount: number;
  vertexCount: number;
};

/**
 * Packs the retained original-coordinate polygon rows into the four-level GPU layout. Mixed
 * geometry or invalid rows are rejected because native Spatial Join would treat them differently
 * (non-polygons become centroid points; invalid values become empty).
 */
export function packSpatialJoinPolygons(
  source: PreparedPolygonTopologySource
): PackedSpatialJoinPolygons {
  const {metadata} = source;
  const rowCount = metadata.rowCount;
  // Polygon row and vertex counts are bounded by memory reservation, not by a fixed cap.
  if (!Number.isSafeInteger(rowCount) || rowCount < 1)
    throw new Error('Spatial Join polygons must have at least 1 row. Use SQL for empty inputs.');
  if (metadata.featureTypes.point || metadata.featureTypes.line || metadata.invalidRows.length)
    throw new Error(
      'Spatial Join polygons must be valid polygon or multipolygon rows only. Use SQL for mixed or invalid geometry.'
    );
  const positions: number[] = [];
  const ringOffsets = [0];
  const polygonOffsets = [0];
  const featureOffsets = [0];
  let expectedOffset = 0;
  for (const batch of source.batches) {
    if (batch.sourceRowOffset !== expectedOffset || batch.polygons.length !== batch.rowCount)
      throw new Error('Spatial Join polygon batches must be contiguous source rows');
    expectedOffset += batch.rowCount;
    for (let row = 0; row < batch.rowCount; row++) {
      const parts = readList(batch.polygons, row);
      for (let partIndex = 0; parts && partIndex < parts.length; partIndex++) {
        const rings = readList(parts, partIndex);
        if (!rings || !rings.length) throw new Error('Invalid polygon part');
        for (let ringIndex = 0; ringIndex < rings.length; ringIndex++) {
          const vertices = readList(rings, ringIndex);
          if (!vertices || !DataType.isFixedSizeList(vertices.type) || vertices.type.listSize < 2)
            throw new Error('Invalid polygon ring');
          let count = vertices.length;
          const read = (index: number): [number, number] => {
            const coordinate = vertices.get(index);
            const x: unknown = coordinate instanceof Vector ? coordinate.get(0) : undefined;
            const y: unknown = coordinate instanceof Vector ? coordinate.get(1) : undefined;
            if (typeof x !== 'number' || typeof y !== 'number' || !Number.isFinite(x + y))
              throw new Error('Nonfinite polygon coordinate');
            return [x, y];
          };
          const first = read(0);
          const last = read(count - 1);
          if (count > 1 && first[0] === last[0] && first[1] === last[1]) count--;
          if (count < 3) throw new Error('Polygon ring needs three vertices');
          // Offsets are uint32 words on the GPU; vertex indices must stay addressable.
          if (positions.length / 2 + count > 0xffffffff)
            throw new Error('Spatial Join polygons exceed 2^32 vertices. Use SQL.');
          for (let index = 0; index < count; index++) positions.push(...read(index));
          ringOffsets.push(positions.length / 2);
        }
        polygonOffsets.push(ringOffsets.length - 1);
      }
      featureOffsets.push(polygonOffsets.length - 1);
    }
  }
  if (expectedOffset !== rowCount) throw new Error('Spatial Join polygon row count mismatch');
  return {
    positions: Float32Array.from(positions),
    ringOffsets: Uint32Array.from(ringOffsets),
    polygonOffsets: Uint32Array.from(polygonOffsets),
    featureOffsets: Uint32Array.from(featureOffsets),
    rowCount,
    vertexCount: positions.length / 2
  };
}

function readList(vector: Vector, rowIndex: number): Vector | null {
  if (!DataType.isList(vector.type)) throw new Error('Invalid polygon list type');
  const value = vector.get(rowIndex);
  if (value === null) return null;
  if (!(value instanceof Vector)) throw new Error('Invalid polygon list value');
  return value;
}
