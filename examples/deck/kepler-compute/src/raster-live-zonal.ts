// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import type {RasterLiveController} from './raster-live-controller';
import type {RasterZonalPolygon, RasterZonalStatistics} from './raster-zonal';

export const ZONAL_COLUMN_NAMES = [
  'zonal_count',
  'zonal_sum',
  'zonal_mean',
  'zonal_min',
  'zonal_max',
  'zonal_nodata',
  'zonal_scanned',
  'zonal_truncated'
] as const;

function toNested(value: unknown): unknown {
  if (value === null || value === undefined) return value;
  if (typeof value === 'number') return value;
  if (typeof value === 'string') return value;
  return Array.from(value as Iterable<unknown>, toNested);
}

function isCoordinate(value: unknown): boolean {
  return Array.isArray(value) && value.length >= 2 && typeof value[0] === 'number';
}

/**
 * Decodes one GeoArrow polygon or multipolygon value (nested lists of [x, y] coordinates) or a
 * GeoJSON geometry string into zonal polygons. Returns undefined for anything else (null rows,
 * points, lines, WKB), which the caller reports as skipped.
 */
export function decodeZonalPolygons(value: unknown): RasterZonalPolygon[] | undefined {
  let nested: unknown = value;
  if (typeof value === 'string') {
    try {
      let geometry = JSON.parse(value) as {
        type?: string;
        coordinates?: unknown;
        geometry?: unknown;
      };
      // GeoJSON Feature strings (as in CSV geometry columns) wrap the geometry.
      if (geometry.type === 'Feature') geometry = (geometry.geometry ?? {}) as typeof geometry;
      if (geometry.type !== 'Polygon' && geometry.type !== 'MultiPolygon') return undefined;
      nested = geometry.coordinates;
    } catch {
      return undefined;
    }
  } else if (value && typeof value === 'object') nested = toNested(value);
  if (!Array.isArray(nested) || nested.length === 0) return undefined;
  // polygon: rings of coordinates; multipolygon: polygons of rings of coordinates.
  const ringFirst = (nested[0] as unknown[])?.[0];
  if (isCoordinate(ringFirst)) return [nested as RasterZonalPolygon];
  const coordinateFirst = (ringFirst as unknown[] | undefined)?.[0];
  if (isCoordinate(coordinateFirst)) return nested as RasterZonalPolygon[];
  return undefined;
}

export type RasterLiveZonalResult = Readonly<{
  table: arrow.Table;
  statistics: readonly (RasterZonalStatistics | null)[];
  skippedRowCount: number;
  /** Set when no output was produced. */
  reason?: string;
}>;

/** Appends the `zonal_*` columns to a table batch by batch, preserving schema and field metadata. */
export function appendZonalColumns(
  table: arrow.Table,
  statistics: readonly (RasterZonalStatistics | null)[]
): arrow.Table {
  const columns: Record<(typeof ZONAL_COLUMN_NAMES)[number], (number | null)[]> = {
    zonal_count: statistics.map(stats => stats?.count ?? null),
    zonal_sum: statistics.map(stats => (stats && stats.count > 0 ? stats.sum : null)),
    zonal_mean: statistics.map(stats => stats?.mean ?? null),
    zonal_min: statistics.map(stats => stats?.minimum ?? null),
    zonal_max: statistics.map(stats => stats?.maximum ?? null),
    zonal_nodata: statistics.map(stats => stats?.nodataCount ?? null),
    zonal_scanned: statistics.map(stats => stats?.scannedCount ?? null),
    zonal_truncated: statistics.map(stats => (stats ? Number(stats.isTruncated) : null))
  };
  const vectors = ZONAL_COLUMN_NAMES.map(name =>
    arrow.vectorFromArray(columns[name], new arrow.Float64())
  );
  const fields = [
    ...table.schema.fields,
    ...ZONAL_COLUMN_NAMES.map(name => new arrow.Field(name, new arrow.Float64(), true))
  ];
  const schema = new arrow.Schema(fields, table.schema.metadata);
  const structType = new arrow.Struct(fields);
  let offset = 0;
  const batches = table.batches.map(batch => {
    const length = batch.numRows;
    const children = [
      ...batch.data.children,
      ...vectors.map(vector => vector.slice(offset, offset + length).data[0])
    ];
    offset += length;
    return new arrow.RecordBatch(
      schema,
      arrow.makeData({type: structType, length, nullCount: 0, children})
    );
  });
  return new arrow.Table(schema, batches);
}

/**
 * Runs zonal statistics for every row of a polygon table against the live raster layer and
 * returns the same table with `zonal_*` columns appended (original columns and field metadata,
 * including GeoArrow extension metadata, are preserved batch by batch).
 */
export function computeZonalTable(options: {
  controller: RasterLiveController;
  rasterLayerId: string;
  table: arrow.Table;
  geometryColumn: string;
  bandIndex?: number;
  maximumPixelCount?: number;
}): RasterLiveZonalResult {
  const {controller, rasterLayerId, table, geometryColumn} = options;
  const geometry = table.getChild(geometryColumn);
  if (!geometry)
    return {table, statistics: [], skippedRowCount: 0, reason: 'Geometry column missing'};
  const rowCount = table.numRows;
  const statistics: (RasterZonalStatistics | null)[] = new Array(rowCount).fill(null);
  let skippedRowCount = 0;
  for (let rowIndex = 0; rowIndex < rowCount; rowIndex++) {
    const polygons = decodeZonalPolygons(geometry.get(rowIndex));
    const stats = polygons
      ? controller.computeZonal(rasterLayerId, polygons, {
          bandIndex: options.bandIndex,
          maximumPixelCount: options.maximumPixelCount
        })
      : undefined;
    if (stats) statistics[rowIndex] = stats;
    else skippedRowCount++;
  }
  if (skippedRowCount === rowCount)
    return {
      table,
      statistics,
      skippedRowCount,
      reason: 'No polygon row intersects a loaded raster tile'
    };
  return {table: appendZonalColumns(table, statistics), statistics, skippedRowCount};
}
