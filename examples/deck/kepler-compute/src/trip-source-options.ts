// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type * as arrow from 'apache-arrow';
import type {GeoJSONTripPreparationOptions} from '@luma.gl/engine/arrow';
import {preparePublishedTrips} from './source-publication';

/**
 * Grouped tabular trips: the layer's geometry-role column is the trip ID column, and these
 * explicit columns supply one timed longitude/latitude vertex per source row.
 */
export type TripGrouping = {
  timestampColumn: string;
  longitudeColumn: string;
  latitudeColumn: string;
  altitudeColumn?: string;
};
/** The sandbox accepts geographic input; CRS transformations require separate preparation. */
export type SandboxTripOptions = Pick<
  GeoJSONTripPreparationOptions,
  'timestampEncoding' | 'originMilliseconds' | 'maximumTimeErrorMilliseconds'
> & {grouping?: TripGrouping};
/** Preparation options accepted by published sources, including optional crs retention. */
export type TripPreparationOptions = GeoJSONTripPreparationOptions & {grouping?: TripGrouping};

const GROUPING_KEYS = ['timestampColumn', 'longitudeColumn', 'latitudeColumn', 'altitudeColumn'];

function readTripGrouping(value: unknown): TripGrouping {
  if (!value || typeof value !== 'object' || Array.isArray(value))
    throw new Error('Invalid saved trip grouping');
  const record = value as Record<string, unknown>;
  if (Object.keys(record).some(key => !GROUPING_KEYS.includes(key)))
    throw new Error('Unsupported trip grouping option');
  for (const key of GROUPING_KEYS.slice(0, 3))
    if (typeof record[key] !== 'string' || !record[key])
      throw new Error('Trip grouping requires explicit timestamp, longitude and latitude columns');
  const altitude = record['altitudeColumn'];
  if (altitude !== undefined && (typeof altitude !== 'string' || !altitude))
    throw new Error('Trip altitude column must be a column name');
  return {
    timestampColumn: record['timestampColumn'] as string,
    longitudeColumn: record['longitudeColumn'] as string,
    latitudeColumn: record['latitudeColumn'] as string,
    ...(altitude === undefined ? {} : {altitudeColumn: altitude})
  };
}

export function readSandboxTripOptions(value: unknown): SandboxTripOptions {
  if (!value || typeof value !== 'object' || Array.isArray(value))
    throw new Error('Invalid saved trip options');
  if (
    Object.keys(value).some(
      key =>
        ![
          'timestampEncoding',
          'originMilliseconds',
          'maximumTimeErrorMilliseconds',
          'grouping'
        ].includes(key)
    )
  )
    throw new Error('Unsupported sandbox trip option');
  const encoding = 'timestampEncoding' in value ? value.timestampEncoding : undefined;
  if (encoding !== 'unix-millisecond' && encoding !== 'unix-second' && encoding !== 'iso8601')
    throw new Error('Choose an explicit trip timestamp encoding');
  const origin = 'originMilliseconds' in value ? value.originMilliseconds : undefined;
  if (origin !== undefined && (typeof origin !== 'number' || !Number.isSafeInteger(origin)))
    throw new Error('Trip origin must be exact integral epoch milliseconds');
  const error = 'maximumTimeErrorMilliseconds' in value ? value.maximumTimeErrorMilliseconds : 0;
  if (typeof error !== 'number' || !Number.isFinite(error) || error < 0)
    throw new Error('Trip time error budget must be finite and nonnegative');
  const grouping = 'grouping' in value ? readTripGrouping(value.grouping) : undefined;
  return {
    timestampEncoding: encoding,
    maximumTimeErrorMilliseconds: error,
    ...(grouping ? {grouping} : {}),
    ...(origin === undefined ? {} : {originMilliseconds: origin})
  };
}

/** CPU-only preflight shared by source review and portable-workspace validation. */
export function prepareSandboxTrips(
  source: arrow.Table,
  geometryColumn: string,
  options: SandboxTripOptions
) {
  return preparePublishedTrips(source, geometryColumn, readSandboxTripOptions(options));
}
