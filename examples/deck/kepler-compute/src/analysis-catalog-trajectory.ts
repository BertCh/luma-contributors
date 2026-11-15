// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {validateCapacity} from '@luma-contributors/analysis/analysis-core';
import type {CatalogFeatureCollection, CatalogOutputTable} from '@luma-contributors/kepler';
import type {SandboxTripOptions} from './trip-source-options';

export const TRAJECTORY_POINT_FIELD = 'Point rows (mapped longitude/latitude columns)';
export type CatalogTrajectorySource = {id: string; label: string; table: arrow.Table};
export type CatalogTrajectoryNativeDataset = {
  id: string;
  label: string;
  fields: readonly {name: string}[];
  dataContainer: {numRows(): number; valueAt(row: number, column: number): unknown};
};

export function isTrajectoryCatalogOperation(operationId: string): boolean {
  return operationId.startsWith('trajectory-');
}

/** Original retained Arrow only: never trigger GPU paging or read a lazy resident table. */
export function makeCatalogTrajectoryNativeSource(
  source: CatalogTrajectorySource
): CatalogTrajectoryNativeDataset {
  const columns = source.table.schema.fields.map(field => source.table.getChild(field.name));
  return {
    id: source.id,
    label: source.label,
    fields: source.table.schema.fields.map(field => ({name: field.name})),
    dataContainer: {
      numRows: () => source.table.numRows,
      // Discovery reads raw cells. Conversion belongs to explicit input preparation, where
      // failures are reported by the Run handler rather than escaping a React render.
      valueAt: (row, column) => columns[column]?.get(row)
    }
  };
}

/** Convert only numeric roles to Numbers; unrelated integers and track IDs stay lossless. */
export function normalizeCatalogTrajectoryProperties(
  properties: Readonly<Record<string, unknown>>,
  parameters: Readonly<Record<string, unknown>> = {},
  coordinateProperties: readonly string[] = []
): Record<string, unknown> {
  const numericProperties = new Set([
    ...coordinateProperties,
    (parameters['timeProperty'] as string | undefined) ?? 'timestamp',
    ...(parameters['target'] === 'property'
      ? [(parameters['valueProperty'] as string | undefined) ?? 'value']
      : [])
  ]);
  const normalizeValue = (value: unknown, property: string): unknown => {
    if (typeof value === 'bigint') {
      if (!numericProperties.has(property)) return value.toString();
      const number = Number(value);
      if (!Number.isSafeInteger(number))
        throw new Error(`Trajectory property ${property} integer exceeds JavaScript exact range.`);
      return number;
    }
    if (value instanceof Date) return value.toISOString();
    if (
      Array.isArray(value) ||
      value instanceof arrow.Vector ||
      (ArrayBuffer.isView(value) && !(value instanceof DataView))
    )
      return Array.from(value as ArrayLike<unknown>, entry => normalizeValue(entry, property));
    // Native structs and other objects are not trajectory scalar/array properties.
    return value === null || ['string', 'number', 'boolean', 'undefined'].includes(typeof value)
      ? value
      : undefined;
  };
  return Object.fromEntries(
    Object.entries(properties).map(([property, value]) => [
      property,
      normalizeValue(value, property)
    ])
  );
}

/** Explicit point column mapping, bounded before cells are inspected. Track/time roles remain parameters. */
export function readCatalogTrajectoryPoints(
  dataset: CatalogTrajectoryNativeDataset,
  longitudeProperty: string,
  latitudeProperty: string,
  parameters: Readonly<Record<string, unknown>> = {}
): CatalogFeatureCollection {
  const rowCount = dataset.dataContainer.numRows();
  validateCapacity('catalog-trajectory-native', 'rowCount', rowCount, 0, 2048);
  const longitudeIndex = dataset.fields.findIndex(field => field.name === longitudeProperty);
  const latitudeIndex = dataset.fields.findIndex(field => field.name === latitudeProperty);
  if (longitudeIndex < 0 || latitudeIndex < 0)
    throw new Error('Choose existing point longitude and latitude columns.');
  return {
    type: 'FeatureCollection',
    features: Array.from({length: rowCount}, (_, row) => {
      const properties: Record<string, unknown> = {};
      dataset.fields.forEach((field, column) => {
        properties[field.name] = dataset.dataContainer.valueAt(row, column);
      });
      const normalized = normalizeCatalogTrajectoryProperties(properties, parameters, [
        longitudeProperty,
        latitudeProperty
      ]);
      return {
        id: row,
        properties: normalized,
        geometry: {
          type: 'Point',
          coordinates: [normalized[longitudeProperty], normalized[latitudeProperty]]
        }
      };
    })
  };
}

/** Native Trips publication uses exact epoch milliseconds with an explicit, editable error budget. */
export function getCatalogTrajectoryTripOptions(
  table: CatalogOutputTable
): SandboxTripOptions | undefined {
  if (table.id !== 'trajectory-tracks' && table.id !== 'trajectory-trips') return undefined;
  const timestamps = table.columns['start_time'].filter(
    (value): value is number => typeof value === 'number'
  );
  if (!timestamps.length) return undefined;
  return {
    timestampEncoding: 'unix-millisecond',
    originMilliseconds: Math.min(...timestamps),
    maximumTimeErrorMilliseconds: 0
  };
}

export function getCatalogTrajectorySampleInputs(
  operationId: string
): Record<string, string> | undefined {
  if (!isTrajectoryCatalogOperation(operationId)) return undefined;
  const origin = Date.UTC(2026, 0, 1);
  // Two tracks, each with two stays separated by a moving section and one isolated spike.
  const features = [0, 1].map(track => ({
    type: 'Feature',
    id: `track-${track}`,
    properties: {track_id: `track-${track}`, value: 1},
    geometry: {
      type: 'LineString',
      coordinates: [
        [0, 0],
        [0, 0],
        [0, 0],
        [0.002, 0],
        [0.004, 0],
        [0.006, 0],
        [0.006, 0],
        [0.006, 0],
        [0.03, 0.02],
        [0.006, 0],
        [0.006, 0]
      ].map(([longitude, latitude], index) => [
        -74 + longitude,
        40.74 + latitude + track * 0.01,
        0,
        origin + index * 60000
      ])
    }
  }));
  return {tracks: JSON.stringify({type: 'FeatureCollection', features})};
}
