// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Pure planning for resident filtering and styling of decoded vector tiles.
 *
 * Scope honesty: this file converts decoded GeoJSON tile features into columnar batches grouped by
 * geometry family, with a stable feature id column and Float32 property columns, and computes a
 * filter mask per batch on the CPU. It does NOT upload to the GPU and is not wired into
 * resident-layers.ts; the batches are shaped so the existing resident point, line and polygon
 * adapters can ingest them, which remains follow-up work (see backlog TILES items).
 */

import {
  resolveVectorTileFeatureIdentity,
  type VectorTileFeatureIdOptions
} from './vector-tile-feature-ids';

export type DecodedTileGeometryFamily = 'point' | 'line' | 'polygon';

export type DecodedTileFeature = Readonly<{
  id?: string | number | null;
  properties?: Readonly<Record<string, unknown>> | null;
  geometry: Readonly<{type: string; coordinates: unknown}> | null;
}>;

export type DecodedTileBatch = Readonly<{
  family: DecodedTileGeometryFamily;
  featureIds: readonly string[];
  /** Flattened lng,lat pairs for every vertex of every feature. */
  positions: Float32Array;
  /** Vertex count per feature (parallel to featureIds), for path/polygon offsets. */
  vertexCounts: Uint32Array;
  numericColumns: ReadonlyMap<string, Float32Array>;
}>;

const FAMILY_BY_TYPE: Readonly<Record<string, DecodedTileGeometryFamily>> = {
  Point: 'point',
  MultiPoint: 'point',
  LineString: 'line',
  MultiLineString: 'line',
  Polygon: 'polygon',
  MultiPolygon: 'polygon'
};

function collectVertices(coordinates: unknown, output: number[]): void {
  if (!Array.isArray(coordinates)) return;
  if (typeof coordinates[0] === 'number') output.push(coordinates[0], coordinates[1] as number);
  else for (const child of coordinates) collectVertices(child, output);
}

/** Groups features by geometry family; features with unsupported or null geometry are skipped. */
export function buildDecodedTileBatches(
  features: readonly DecodedTileFeature[],
  options: VectorTileFeatureIdOptions & {numericFieldNames?: readonly string[]}
): readonly DecodedTileBatch[] {
  const groups = new Map<
    DecodedTileGeometryFamily,
    {ids: string[]; vertices: number[]; counts: number[]; columns: Map<string, number[]>}
  >();
  for (const feature of features) {
    const family = feature.geometry ? FAMILY_BY_TYPE[feature.geometry.type] : undefined;
    if (!family || !feature.geometry) continue;
    let group = groups.get(family);
    if (!group) {
      group = {ids: [], vertices: [], counts: [], columns: new Map()};
      groups.set(family, group);
    }
    const before = group.vertices.length;
    collectVertices(feature.geometry.coordinates, group.vertices);
    group.counts.push((group.vertices.length - before) / 2);
    group.ids.push(resolveVectorTileFeatureIdentity(feature, options).featureId);
    for (const name of options.numericFieldNames ?? []) {
      const value = feature.properties?.[name];
      const column = group.columns.get(name) ?? [];
      column.push(typeof value === 'number' ? value : Number.NaN);
      group.columns.set(name, column);
    }
  }
  return [...groups].map(([family, group]) => ({
    family,
    featureIds: group.ids,
    positions: Float32Array.from(group.vertices),
    vertexCounts: Uint32Array.from(group.counts),
    numericColumns: new Map(
      [...group.columns].map(([name, values]) => [name, Float32Array.from(values)])
    )
  }));
}

export type DecodedTileFilter =
  | Readonly<{kind: 'range'; field: string; min: number; max: number}>
  | Readonly<{kind: 'select'; field: string; values: readonly number[]}>;

/** 1 = feature passes all filters. NaN (missing) values fail every filter on that field. */
export function computeDecodedTileFilterMask(
  batch: DecodedTileBatch,
  filters: readonly DecodedTileFilter[]
): Uint8Array {
  const mask = new Uint8Array(batch.featureIds.length).fill(1);
  for (const filter of filters) {
    const column = batch.numericColumns.get(filter.field);
    for (let index = 0; index < mask.length; index++) {
      const value = column ? column[index] : Number.NaN;
      const passes =
        filter.kind === 'range'
          ? value >= filter.min && value <= filter.max
          : filter.values.includes(value);
      if (!passes) mask[index] = 0;
    }
  }
  return mask;
}

/** Linear color-domain extent over passing features; undefined when nothing passes. */
export function computeDecodedTileStyleDomain(
  batch: DecodedTileBatch,
  field: string,
  mask?: Uint8Array
): [number, number] | undefined {
  const column = batch.numericColumns.get(field);
  if (!column) return undefined;
  let minimum = Infinity;
  let maximum = -Infinity;
  for (let index = 0; index < column.length; index++) {
    const value = column[index];
    if ((mask && !mask[index]) || Number.isNaN(value)) continue;
    if (value < minimum) minimum = value;
    if (value > maximum) maximum = value;
  }
  return minimum <= maximum ? [minimum, maximum] : undefined;
}
