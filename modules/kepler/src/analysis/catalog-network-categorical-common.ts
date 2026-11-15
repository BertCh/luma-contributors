// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  CatalogOperationBinding,
  CatalogParameterSpec,
  CatalogRunOutput
} from './catalog-binding-types';
import {
  appendCatalogRow,
  getFeatureIdentity,
  getPointPositions,
  makeCatalogTable,
  type CatalogColumnType,
  type CatalogFeatureCollection
} from './catalog-geojson';
import type {CatalogGpuSession} from './catalog-gpu-session';
import {PLATFORM3_GPU_OR_CPU_CLEANUP, type Values} from './catalog-platform3-b-common';

export const MAXIMUM_ROWS = 1024;
export const CLEANUP = PLATFORM3_GPU_OR_CPU_CLEANUP;

export function makeBinding(
  binding: Omit<
    CatalogOperationBinding,
    'engine' | 'gpuContributorNames' | 'cleanup' | 'capabilityIds'
  >
): CatalogOperationBinding {
  return {
    ...binding,
    engine: 'gpu',
    gpuContributorNames: binding.contributorNames,
    cleanup: CLEANUP,
    capabilityIds: []
  };
}

export function makeEnum(
  name: string,
  label: string,
  options: readonly string[],
  defaultValue = options[0]
): CatalogParameterSpec {
  return {name, label, kind: 'enum', options, defaultValue};
}

/** Keep original rows, including missing categories and invalid points; never shift source identities. */
export function readPositions(
  collection: CatalogFeatureCollection,
  geographic = false
): Float32Array {
  return Float32Array.from(
    collection.features.flatMap(feature => {
      const positions = getPointPositions(feature.geometry);
      if (
        positions.length !== 1 ||
        !positions[0].slice(0, 2).every(Number.isFinite) ||
        (geographic && Math.abs(positions[0][1]) > 90)
      )
        throw new Error(
          'Each feature must contain one finite point in the declared coordinate system'
        );
      if (positions[0].slice(0, 2).some(value => !Number.isFinite(Math.fround(value))))
        throw new Error('Coordinates exceed Float32 range');
      return [positions[0][0], positions[0][1]];
    })
  );
}

export function readNumbers(
  collection: CatalogFeatureCollection,
  property: string,
  nonnegative = false
): Float32Array {
  return Float32Array.from(
    collection.features.map(feature => {
      const value = feature.properties?.[property];
      if (
        typeof value !== 'number' ||
        !Number.isFinite(Math.fround(value)) ||
        (nonnegative && value < 0)
      )
        throw new Error(
          `Property ${property} must be finite${nonnegative ? ' and nonnegative' : ''} on every row`
        );
      return value;
    })
  );
}

export function readCategories(collection: CatalogFeatureCollection, property: string) {
  const labels: string[] = [];
  const indices = new Map<string, number>();
  const categories = Uint32Array.from(
    collection.features.map(feature => {
      const value = feature.properties?.[property];
      if (value === null || value === undefined) return 0xffffffff;
      if (
        (typeof value !== 'string' && typeof value !== 'number' && typeof value !== 'boolean') ||
        (typeof value === 'number' && !Number.isFinite(value))
      )
        throw new Error(`Property ${property} must contain scalar categories`);
      const key = JSON.stringify([typeof value, value]);
      let index = indices.get(key);
      if (index === undefined) {
        index = labels.length;
        indices.set(key, index);
        labels.push(`${typeof value}:${String(value)}`);
      }
      return index;
    })
  );
  if (!labels.length || labels.length > 32) throw new Error('Category count must be in [1, 32]');
  return {categories, labels};
}

export function uploadWeights(
  session: CatalogGpuSession,
  weights: {offsets: Uint32Array; neighbors: Uint32Array; weights: Float32Array}
) {
  return {
    offsets: session.input('offsets', weights.offsets, 'uint32'),
    neighbors: session.input('neighbors', weights.neighbors, 'uint32'),
    weights: session.input('weights', weights.weights, 'float32')
  };
}

/** Non-finite contributor statistics stay null in native tables, with status retained separately. */
export function makeResult(
  id: string,
  label: string,
  columns: Record<string, CatalogColumnType>,
  rows: readonly Record<string, unknown>[],
  summary: CatalogRunOutput['summary'] = {}
): CatalogRunOutput {
  const table = makeCatalogTable(id, label, columns, columns['geometry'] ? 'geometry' : undefined);
  for (const row of rows)
    appendCatalogRow(
      table,
      Object.fromEntries(
        Object.entries(row).map(([name, value]) => [
          name,
          typeof value === 'number' && !Number.isFinite(value) ? null : value
        ])
      ) as Parameters<typeof appendCatalogRow>[1]
    );
  return {tables: [table], diagnostics: [], summary: {rowCount: rows.length, ...summary}};
}

export function getSourceRow(collection: CatalogFeatureCollection, index: number) {
  const feature = collection.features[index];
  return {
    source_index: index,
    source_id: getFeatureIdentity(feature),
    geometry: JSON.stringify(feature.geometry)
  };
}

export function getNumber(values: Values, name: string): number {
  return values[name] as number;
}
export function getString(values: Values, name: string): string {
  return values[name] as string;
}
