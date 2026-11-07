// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type * as arrow from 'apache-arrow';
import {parseGeoJsonRawFeature} from '@kepler.gl/layers';

/** Arrow field metadata key that carries the GeoArrow extension name (Kepler GEOARROW_METADATA_KEY). */
export const FLOW_FIELD_GEOARROW_METADATA_KEY = 'ARROW:extension:name';
export const FLOW_FIELD_CENTROID_ROW_LIMIT = 500000;

export type FlowFieldCentroids = {
  longitudes: Float64Array;
  latitudes: Float64Array;
  /** Rows beyond the row limit; they are NaN and never contribute to the field. */
  truncatedCount: number;
};

export type FlowFieldCentroidOptions = {
  rowLimit?: number;
  /** Injectable for tests; defaults to Kepler's parser (GeoJSON, WKT, WKB, GeoArrow). */
  parseFeature?: (
    value: unknown,
    geoArrowEncoding?: string
  ) => {geometry?: unknown} | null | undefined;
};

type GeometryLike = {type?: string; coordinates?: unknown; geometries?: GeometryLike[]};

function accumulatePositions(coordinates: unknown, sums: [number, number, number]): void {
  if (!Array.isArray(coordinates)) return;
  if (typeof coordinates[0] === 'number') {
    sums[0] += coordinates[0] as number;
    sums[1] += coordinates[1] as number;
    sums[2]++;
    return;
  }
  for (const child of coordinates) accumulatePositions(child, sums);
}

function accumulateGeometry(geometry: GeometryLike, sums: [number, number, number]): void {
  if (geometry.geometries) for (const child of geometry.geometries) accumulateGeometry(child, sums);
  else accumulatePositions(geometry.coordinates, sums);
}

/** Native `getCentroidFromGeometry`: the average of all vertex positions, or null. */
export function getFlowFieldGeometryCentroid(geometry: unknown): [number, number] | null {
  if (!geometry) return null;
  const sums: [number, number, number] = [0, 0, 0];
  accumulateGeometry(geometry as GeometryLike, sums);
  if (!sums[2]) return null;
  const longitude = sums[0] / sums[2];
  const latitude = sums[1] / sums[2];
  return Number.isFinite(longitude) && Number.isFinite(latitude) ? [longitude, latitude] : null;
}

function getGeoArrowEncoding(table: arrow.Table, columnName: string): string | undefined {
  const field = table.schema.fields.find(candidate => candidate.name === columnName);
  return field?.metadata?.get(FLOW_FIELD_GEOARROW_METADATA_KEY) || undefined;
}

/** Reads one GeoJSON/GeoArrow column as per-row centroid longitudes and latitudes (NaN on failure). */
export function readFlowFieldCentroids(
  table: arrow.Table,
  columnName: string,
  options: FlowFieldCentroidOptions = {}
): FlowFieldCentroids {
  const rowLimit = options.rowLimit ?? FLOW_FIELD_CENTROID_ROW_LIMIT;
  const parseFeature = options.parseFeature ?? parseGeoJsonRawFeature;
  const longitudes = new Float64Array(table.numRows).fill(NaN);
  const latitudes = new Float64Array(table.numRows).fill(NaN);
  const vector = table.getChild(columnName);
  if (!vector) return {longitudes, latitudes, truncatedCount: 0};
  const geoArrowEncoding = getGeoArrowEncoding(table, columnName);
  const count = Math.min(table.numRows, rowLimit);
  for (let index = 0; index < count; index++) {
    try {
      const centroid = getFlowFieldGeometryCentroid(
        parseFeature(vector.get(index), geoArrowEncoding)?.geometry
      );
      if (centroid) {
        longitudes[index] = centroid[0];
        latitudes[index] = centroid[1];
      }
    } catch {
      // Unparseable cells stay NaN, as in the native layer.
    }
  }
  return {longitudes, latitudes, truncatedCount: Math.max(0, table.numRows - count)};
}

type CachedCentroids = {table: arrow.Table; columnName: string; centroids: FlowFieldCentroids};

/** Parses each layer's centroid column once per table and column. */
export class FlowFieldCentroidCache {
  private readonly entries = new Map<string, CachedCentroids>();

  get(layerId: string, table: arrow.Table, columnName: string): FlowFieldCentroids {
    const cached = this.entries.get(layerId);
    if (cached && cached.table === table && cached.columnName === columnName)
      return cached.centroids;
    const centroids = readFlowFieldCentroids(table, columnName);
    this.entries.set(layerId, {table, columnName, centroids});
    return centroids;
  }

  release(layerId: string): void {
    this.entries.delete(layerId);
  }
}
