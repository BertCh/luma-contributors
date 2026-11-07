// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import type {GPUVector} from '@luma-contributors/graph/gpu-data';
import type {KeplerTableDerivedSnapshot} from '@luma-contributors/kepler';
import type {ResidentTooltipValue} from './resident-tooltip-fields';

/** Neighbor, core-distance, p-value and similar companion columns shown beside the primary result. */
export const MAXIMUM_NEIGHBOR_TOOLTIP_COLUMNS = 8;

type ScalarVector = GPUVector<'float32' | 'uint32' | 'sint32'>;

/** Four-byte readback of one row from a chunked, borrowed result vector; null when out of range. */
export async function readVectorRowValue(
  vector: GPUVector<any>,
  rowIndex: number
): Promise<number | null> {
  let rowOffset = 0;
  for (const chunk of vector.data) {
    if (rowIndex < rowOffset + chunk.length) {
      const buffer = chunk.buffer instanceof Buffer ? chunk.buffer : chunk.buffer.buffer;
      const bytes = await buffer.readAsync(chunk.byteOffset + (rowIndex - rowOffset) * 4, 4);
      const view = new DataView(bytes.buffer, bytes.byteOffset, 4);
      if (vector.format === 'float32') return view.getFloat32(0, true);
      if (vector.format === 'uint32') return view.getUint32(0, true);
      if (vector.format === 'sint32') return view.getInt32(0, true);
      return null;
    }
    rowOffset += chunk.length;
  }
  return null;
}

/** "analysis_local_moran_neighbor_count" -> "Local moran neighbor count". */
export function getNeighborTooltipLabel(columnName: string): string {
  const words = columnName.replace(/^analysis_/, '').replace(/_/g, ' ');
  return words.charAt(0).toUpperCase() + words.slice(1);
}

/**
 * Bounded readback of the companion columns of the active neighbor-based analysis (hot spots,
 * local Moran, quantile LISA, DBSCAN/OPTICS core distance and reachability) for one picked row.
 * Zero validity marks null. Skips the primary result column, which the base provider already shows.
 */
export async function readNeighborTooltipValues(
  snapshot: KeplerTableDerivedSnapshot | undefined,
  primaryColumn: string,
  rowIndex: number
): Promise<Record<string, ResidentTooltipValue>> {
  if (!snapshot) return {};
  const names = Object.keys(snapshot.columns)
    .filter(name => name !== primaryColumn)
    .slice(0, MAXIMUM_NEIGHBOR_TOOLTIP_COLUMNS);
  const entries = await Promise.all(
    names.map(async (name): Promise<[string, ResidentTooltipValue] | null> => {
      const vector = snapshot.columns[name] as ScalarVector;
      if (!vector.format || !['float32', 'uint32', 'sint32'].includes(vector.format)) return null;
      const validity = snapshot.validity?.[name];
      const [value, valid] = await Promise.all([
        readVectorRowValue(vector, rowIndex),
        validity ? readVectorRowValue(validity, rowIndex) : Promise.resolve(1)
      ]);
      const finite = value !== null && valid !== 0 && Number.isFinite(value);
      return [getNeighborTooltipLabel(name), finite ? value : null];
    })
  );
  // A stale snapshot must not label values from another result revision.
  if (!snapshot.isCurrent()) return {};
  return Object.fromEntries(entries.filter(entry => entry !== null));
}

/** Eight-byte readback of one point's [longitude, latitude] from the resident positions. */
export async function readPointPosition(
  positions: GPUVector<'float32x2'>,
  rowIndex: number
): Promise<readonly [number, number] | null> {
  let rowOffset = 0;
  for (const chunk of positions.data) {
    if (rowIndex < rowOffset + chunk.length) {
      const buffer = chunk.buffer instanceof Buffer ? chunk.buffer : chunk.buffer.buffer;
      const bytes = await buffer.readAsync(chunk.byteOffset + (rowIndex - rowOffset) * 8, 8);
      const view = new DataView(bytes.buffer, bytes.byteOffset, 8);
      const longitude = view.getFloat32(0, true);
      const latitude = view.getFloat32(4, true);
      return Number.isFinite(longitude) && Number.isFinite(latitude) ? [longitude, latitude] : null;
    }
    rowOffset += chunk.length;
  }
  return null;
}
