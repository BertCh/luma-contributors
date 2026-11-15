// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Vector, type DataType} from 'apache-arrow';
import {parseWKB, type WellKnownGeometry} from '@math.gl/arrow/wkb';
import type {GeometrySourceBatch} from './geometry-gpu-vectors';

/** Existing WKB reader handles endian/dialect/topology; this adapter retains Arrow row identity. */
export function decodeGeometrySourceWKB(vector: Vector<DataType>): {
  batches: GeometrySourceBatch[];
  invalidRows: {sourceRowIndex: number; reason: string}[];
  spatialReferences: {sourceRowIndex: number; srids: readonly number[]}[];
} {
  const invalidRows: {sourceRowIndex: number; reason: string}[] = [];
  const spatialReferences: {sourceRowIndex: number; srids: readonly number[]}[] = [];
  let sourceRowOffset = 0;
  const batches = vector.data.map((chunk, chunkIndex) => {
    const values = new Vector([chunk]);
    const rows: unknown[] = [];
    for (let rowIndex = 0; rowIndex < chunk.length; rowIndex++) {
      const sourceRowIndex = sourceRowOffset + rowIndex;
      const bytes: unknown = values.get(rowIndex);
      if (
        bytes === null ||
        bytes === undefined ||
        (bytes instanceof Uint8Array && bytes.byteLength === 0)
      ) {
        rows.push(null);
        continue;
      }
      try {
        if (!(bytes instanceof Uint8Array))
          throw new Error('WKB geometry requires a binary Arrow row');
        // One pass: nested geometries only occur below a root GeometryCollection or Multi* type.
        const parsed = parseWKB(bytes, {maximumDepth: 32});
        if (parsed.geometry.type === 'GeometryCollection')
          throw new Error('Pinned Kepler WKBLoader does not support GeometryCollection');
        if (parsed.dimension !== 'xy' && parsed.dimension !== 'xyz')
          throw new Error('WKB measure coordinates require a dedicated XYM/XYZM adapter');
        if (parsed.srid !== undefined)
          spatialReferences.push({sourceRowIndex, srids: [parsed.srid]});
        rows.push(normalizeEmptyWKBPoints(parsed.geometry));
      } catch (error) {
        rows.push(null);
        invalidRows.push({
          sourceRowIndex,
          reason: error instanceof Error ? error.message : String(error)
        });
      }
    }
    sourceRowOffset += chunk.length;
    return {id: `geometry-batch-${chunkIndex}`, rows};
  });
  return {batches, invalidRows, spatialReferences};
}

export function normalizeEmptyWKBPoints(geometry: WellKnownGeometry): WellKnownGeometry {
  switch (geometry.type) {
    case 'Point':
      return geometry.coordinates.every(Number.isNaN) ? {type: 'Point', coordinates: []} : geometry;
    case 'MultiPoint':
      return {
        ...geometry,
        coordinates: geometry.coordinates.filter(coordinate => !coordinate.every(Number.isNaN))
      };
    case 'GeometryCollection':
      return {...geometry, geometries: geometry.geometries.map(normalizeEmptyWKBPoints)};
    default:
      return geometry;
  }
}
