// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type * as arrow from 'apache-arrow';
import {validateCapacity, validateRequirement} from '@luma-contributors/analysis/analysis-core';
import type {CatalogFeatureCollection} from '@luma-contributors/kepler';

export type PointCloudNativeDataset = {
  id: string;
  label: string;
  fields: readonly {name: string}[];
  dataContainer: {numRows(): number; valueAt(row: number, column: number): unknown};
};
type PointCloudDatasetLike = Pick<PointCloudNativeDataset, 'fields' | 'dataContainer'>;
type PointCloudPublication = {
  label: string;
  dataset: {
    id: string;
    population?: {state: string; partial?: unknown};
    value: {resident?: unknown; source: arrow.Table};
  };
};
const POINT_CLOUD_FIELD_PREFIX = 'XYZ columns: ';
const MAXIMUM_POINT_CLOUD_ROWS = 2048;

/** Discover explicit XYZ role sets by case-insensitive source names; users choose the set in the field menu. */
export function findPointCloudCatalogFields(
  dataset: Pick<PointCloudDatasetLike, 'fields'>
): string[] {
  // Canonical generated coordinates must win over retained source aliases, regardless of
  // Arrow schema order. Prefer an exact spelling before a case-insensitive source alias.
  const find = (aliases: string[]) =>
    aliases
      .map(
        alias =>
          dataset.fields.find(field => field.name === alias) ??
          dataset.fields.find(field => field.name.toLowerCase() === alias)
      )
      .find(Boolean)?.name;
  const alternatives = [
    [
      find(['longitude', 'lon', 'lng']),
      find(['latitude', 'lat']),
      find(['altitude', 'alt', 'elevation', 'height', 'z'])
    ],
    [find(['x']), find(['y']), find(['z'])]
  ];
  return alternatives
    .filter((names): names is string[] => names.every((name): name is string => Boolean(name)))
    .map(names => `${POINT_CLOUD_FIELD_PREFIX}${JSON.stringify(names)}`);
}

/** Bounded explicit read, shared by retained Arrow and native CSV/LAS-like point tables. */
export function readPointCloudCatalogDataset(
  dataset: PointCloudDatasetLike,
  field: string
): CatalogFeatureCollection | undefined {
  if (!field.startsWith(POINT_CLOUD_FIELD_PREFIX)) return undefined;
  const coordinateFields: unknown = JSON.parse(field.slice(POINT_CLOUD_FIELD_PREFIX.length));
  validateRequirement(
    'catalog-pointcloud-source',
    Array.isArray(coordinateFields) &&
      coordinateFields.length === 3 &&
      coordinateFields.every(name => typeof name === 'string') &&
      new Set(coordinateFields).size === 3,
    'coordinate fields must name three distinct XYZ columns'
  );
  const names = coordinateFields as string[];
  const indices = names.map(name => dataset.fields.findIndex(candidate => candidate.name === name));
  validateRequirement(
    'catalog-pointcloud-source',
    indices.every(index => index >= 0),
    'coordinate fields must exist in the source'
  );
  const count = dataset.dataContainer.numRows();
  validateCapacity('catalog-pointcloud-source', 'rows', count, 1, MAXIMUM_POINT_CLOUD_ROWS);
  const features = Array.from({length: count}, (_, row) => {
    const coordinates = indices.map(column => {
      const value = dataset.dataContainer.valueAt(row, column);
      return typeof value === 'number'
        ? value
        : typeof value === 'string' && value.trim()
          ? Number(value)
          : Number.NaN;
    });
    const properties: Record<string, unknown> = {};
    dataset.fields.forEach((sourceField, column) => {
      const value = dataset.dataContainer.valueAt(row, column);
      if (typeof value === 'number' || typeof value === 'string' || typeof value === 'boolean')
        properties[sourceField.name] = value;
      else if (typeof value === 'bigint') properties[sourceField.name] = value.toString();
    });
    return {id: row, geometry: {type: 'Point', coordinates}, properties};
  });
  validateRequirement(
    'catalog-pointcloud-source',
    features.every(feature => feature.geometry.coordinates.every(Number.isFinite)),
    'XYZ columns must contain finite numeric values; null/empty coordinates are not zero'
  );
  return {type: 'FeatureCollection', features};
}

/** Original complete retained CPU sources only. Never page a GPU result or consume a stream prefix. */
export function listPointCloudCatalogSources(
  publications: Iterable<PointCloudPublication>
): PointCloudNativeDataset[] {
  return Array.from(publications)
    .filter(
      published =>
        !published.dataset.value.resident &&
        (!published.dataset.population ||
          (published.dataset.population.state === 'complete' &&
            !published.dataset.population.partial))
    )
    .map(published => {
      const table = published.dataset.value.source;
      const columns = table.schema.fields.map(field => table.getChild(field.name));
      return {
        id: published.dataset.id,
        label: published.label,
        fields: table.schema.fields.map(field => ({name: field.name})),
        dataContainer: {
          numRows: () => table.numRows,
          valueAt: (row: number, column: number) => columns[column]?.get(row)
        }
      };
    })
    .filter(dataset => findPointCloudCatalogFields(dataset).length > 0);
}

export function getPointCloudCatalogSamples(operationId: string): Record<string, string> {
  if (!operationId.startsWith('pointcloud-')) return {};
  const make = (offset: number) =>
    JSON.stringify({
      type: 'FeatureCollection',
      features: Array.from({length: 25}, (_, row) => ({
        type: 'Feature',
        id: `point-${row}`,
        geometry: {
          type: 'Point',
          coordinates: [
            -87.65 + (row % 5) * 0.00001,
            41.88 + Math.floor(row / 5) * 0.00001,
            100 + offset + (row === 12 ? 3 : 0)
          ]
        },
        properties: {
          classification: row === 12 ? 1 : 2,
          intensity: row * 10,
          voxel_id: row,
          count: 1,
          representative: row
        }
      }))
    });
  if (operationId === 'pointcloud-icp') return {source: make(0.1), target: make(0)};
  if (operationId === 'pointcloud-change') return {reference: make(0), comparison: make(0.2)};
  if (operationId === 'pointcloud-voxel-merge') {
    const planar = (offset: number) => {
      const collection = JSON.parse(make(offset));
      for (const [row, feature] of collection.features.entries())
        feature.geometry.coordinates = [row % 5, Math.floor(row / 5), offset];
      return JSON.stringify(collection);
    };
    return {left: planar(0), right: planar(0.2)};
  }
  return {points: make(0)};
}

/** Suggest all three numeric render roles when a point-cloud result is staged in Sources. */
export function getPointCloudCatalogCoordinates(table: arrow.Table, label: string) {
  return (label.startsWith('Point cloud ') || label === 'Point-cloud surfaces') &&
    ['longitude', 'latitude', 'altitude'].every(name => table.getChild(name))
    ? {longitude: 'longitude', latitude: 'latitude', altitude: 'altitude'}
    : undefined;
}
