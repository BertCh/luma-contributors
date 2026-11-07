// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {DGGSFamily} from './dggs-source';

/** Native Kepler layer types whose `token` column is a DGGS cell ID with derived polygon footprints. */
export function isCellLayerType(type: string | null | undefined): type is DGGSFamily {
  return type === 's2' || type === 'a5' || type === 'geohash';
}

/** Native layers whose resident rendering, styling and picking use the published polygon path. */
export function isPolygonFootprintLayerType(type: string | null | undefined): boolean {
  return type === 'geojson' || isCellLayerType(type);
}

/** Recorded cell index: the native layer `token` column and its DGGS family. */
export type NativeCellIndex = {family: DGGSFamily; column: string};
export const CELL_INDEX_METADATA_KEY = 'nativeCellIndex';

export function getNativeCellIndex(
  metadata: Readonly<Record<string, unknown>> | undefined
): NativeCellIndex | undefined {
  const value = metadata?.[CELL_INDEX_METADATA_KEY] as NativeCellIndex | undefined;
  return value && isCellLayerType(value.family) && typeof value.column === 'string'
    ? {family: value.family, column: value.column}
    : undefined;
}

export function parseNativeCellIndex(
  value: unknown,
  fieldNames: readonly string[]
): NativeCellIndex | undefined {
  if (value === undefined || value === null) return undefined;
  const record = value as Record<string, unknown>;
  if (
    typeof value !== 'object' ||
    Array.isArray(value) ||
    typeof record['family'] !== 'string' ||
    !isCellLayerType(record['family']) ||
    typeof record['column'] !== 'string' ||
    !fieldNames.includes(record['column'])
  )
    throw new Error('Invalid saved cell index');
  return {family: record['family'], column: record['column']};
}

/** The native column role that selects a polygon-footprint layer's source: GeoJSON or cell token. */
export function getFootprintLayerColumn(layer: {
  config: {columns: object};
}): string | null | undefined {
  const columns = layer.config.columns as Record<string, {value: string | null} | undefined>;
  return (columns['geojson'] ?? columns['token'])?.value;
}
