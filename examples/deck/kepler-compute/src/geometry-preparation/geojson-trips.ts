// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Vector} from 'apache-arrow';
import {
  getGeometryFeatureProperties,
  ensureGeometryProperty,
  writeGeometryProperty,
  type PreparedGeometryProperties,
  type GeometryStyleDiagnostic
} from './geometry-properties';
import {getTripTypedStorageByteLength, quantizeTripTimes, validateTripOptions} from './trip-time';

export type TripTimestampEncoding = 'unix-millisecond' | 'unix-second' | 'iso8601';
export type GeoJSONTripSourceBatch = {id: string; rows: readonly unknown[]};
export type KeplerGeoJSONTripPreparationOptions = {
  timestampEncoding: TripTimestampEncoding;
  /** Explicit epoch-ms origin, or the minimum accepted source timestamp when omitted. */
  originMilliseconds?: number;
  /** Zero requires exact Float32 relative milliseconds. Larger error is an explicit choice. */
  maximumTimeErrorMilliseconds?: number;
  crs?: unknown;
};
export type PreparedGeoJSONTripBatch = {
  id: string;
  rowCount: number;
  sourceRowOffset: number;
  positions: Float32Array;
  valueOffsets: Int32Array;
  /** Global source-row index per trip primitive; invalid/null rows never renumber valid rows. */
  sourceRows: Uint32Array;
  /** Original timestamp values, aligned to flattened vertices, before any parsing/rebasing. */
  sourceTimestamps: readonly (number | string)[];
  /** Integral epoch milliseconds are exactly represented in Float64, never uploaded as Float32. */
  epochMilliseconds: Float64Array;
  timestamps: Float32Array;
  properties?: Pick<PreparedGeometryProperties, 'lineColor' | 'lineWidth'>;
};
export type PreparedKeplerGeoJSONTrips = {
  kind: 'geojson-linestring-trips' | 'tabular-trips';
  timestampEncoding: TripTimestampEncoding;
  originMilliseconds: number;
  maximumTimeErrorMilliseconds: number;
  actualTimeErrorMilliseconds: number;
  animationDomainMilliseconds: readonly [number, number] | null;
  rowCount: number;
  tripCount: number;
  vertexCount: number;
  crs?: unknown;
  batches: readonly PreparedGeoJSONTripBatch[];
  invalidRows: readonly {sourceRowIndex: number; reason: string}[];
  sourceStyleProperties: readonly ('lineColor' | 'lineWidth')[];
  invalidStyleProperties: readonly GeometryStyleDiagnostic[];
  /** Retained typed storage and offsets; excludes original string/JS object overhead. */
  typedStorageByteLength: number;
};
type ParsedTrip = {
  sourceRowIndex: number;
  positions: number[];
  sourceTimestamps: (number | string)[];
  epochMilliseconds: number[];
};

/**
 * Explicit ingestion for GeoJSON LineString or MultiLineString [x,y,z,time] rows.
 * Each part is an independent timed primitive with the same source identity; parts never connect.
 * A malformed part rejects the whole row. Collections and WKB measures need separate adapters.
 * Parsing never sorts vertices, drops individual vertices, or rewrites source coordinates.
 */
export function prepareKeplerGeoJSONTrips(
  batches: readonly GeoJSONTripSourceBatch[],
  options: KeplerGeoJSONTripPreparationOptions
): PreparedKeplerGeoJSONTrips {
  const maximumTimeErrorMilliseconds = validateTripOptions(options);
  const invalidRows: {sourceRowIndex: number; reason: string}[] = [];
  const sourceStyleProperties = new Set<'lineColor' | 'lineWidth'>();
  const invalidStyleProperties: GeometryStyleDiagnostic[] = [];
  const ids = new Set<string>();
  let rowCount = 0;
  let minimumSourceTime = Infinity;
  const parsed = batches.map(batch => {
    if (!batch.id || ids.has(batch.id)) throw new Error('Trip batch IDs must be unique');
    ids.add(batch.id);
    const sourceRowOffset = rowCount;
    rowCount += batch.rows.length;
    if (!Number.isSafeInteger(rowCount) || rowCount > 0xffffffff)
      throw new Error('Trip source rows exceed uint32 identity capacity');
    const trips: ParsedTrip[][] = [];
    batch.rows.forEach((row, localRowIndex) => {
      if (row === null || row === undefined) return;
      const sourceRowIndex = sourceRowOffset + localRowIndex;
      try {
        const parts = parseTrips(row, sourceRowIndex, options.timestampEncoding);
        for (const part of parts)
          minimumSourceTime = Math.min(minimumSourceTime, part.epochMilliseconds[0]);
        trips.push(parts);
      } catch (error) {
        invalidRows.push({
          sourceRowIndex,
          reason: error instanceof Error ? error.message : String(error)
        });
      }
    });
    return {batch, sourceRowOffset, trips};
  });
  const originMilliseconds =
    options.originMilliseconds ?? (Number.isFinite(minimumSourceTime) ? minimumSourceTime : 0);
  let actualTimeErrorMilliseconds = 0;
  let minimumTime = Infinity;
  let maximumTime = -Infinity;
  let tripCount = 0;
  let vertexCount = 0;
  const preparedBatches = parsed.map(({batch, sourceRowOffset, trips}) => {
    const positions: number[] = [];
    const sourceRows: number[] = [];
    const sourceTimestamps: (number | string)[] = [];
    const epochMilliseconds: number[] = [];
    const timestamps: number[] = [];
    const valueOffsets = [0];
    const properties: Pick<PreparedGeometryProperties, 'lineColor' | 'lineWidth'> = {};
    for (const parts of trips) {
      const quantizedParts = parts.map(trip =>
        quantizeTripTimes(trip.epochMilliseconds, originMilliseconds, maximumTimeErrorMilliseconds)
      );
      const rejected = quantizedParts.find(result => !result.valid);
      if (rejected && !rejected.valid) {
        invalidRows.push({sourceRowIndex: parts[0].sourceRowIndex, reason: rejected.reason});
        continue;
      }
      for (const [partIndex, trip] of parts.entries()) {
        const quantized = quantizedParts[partIndex];
        if (!quantized.valid) continue;
        const {rounded, error} = quantized;
        const localRowIndex = trip.sourceRowIndex - sourceRowOffset;
        const featureProperties = getGeometryFeatureProperties(batch.rows[localRowIndex]);
        if (partIndex === 0)
          for (const property of ['lineColor', 'lineWidth'] as const) {
            const value = featureProperties?.[property];
            if (value === undefined) continue;
            sourceStyleProperties.add(property);
            const reason = writeGeometryProperty(
              properties,
              property,
              value,
              localRowIndex,
              batch.rows.length
            );
            if (reason)
              invalidStyleProperties.push({sourceRowIndex: trip.sourceRowIndex, property, reason});
          }
        if (timestamps.length + rounded.length > 0x7fffffff)
          throw new Error('Trip batch vertices exceed Int32 offset capacity');
        for (const position of trip.positions) positions.push(position);
        sourceRows.push(trip.sourceRowIndex);
        for (const value of trip.sourceTimestamps) sourceTimestamps.push(value);
        for (const value of trip.epochMilliseconds) epochMilliseconds.push(value);
        for (const value of rounded) timestamps.push(value);
        valueOffsets.push(timestamps.length);
        minimumTime = Math.min(minimumTime, trip.epochMilliseconds[0]);
        maximumTime = Math.max(
          maximumTime,
          trip.epochMilliseconds[trip.epochMilliseconds.length - 1]
        );
        actualTimeErrorMilliseconds = Math.max(actualTimeErrorMilliseconds, error);
      }
    }
    tripCount += sourceRows.length;
    vertexCount += timestamps.length;
    return {
      id: batch.id,
      rowCount: batch.rows.length,
      sourceRowOffset,
      positions: Float32Array.from(positions),
      valueOffsets: Int32Array.from(valueOffsets),
      sourceRows: Uint32Array.from(sourceRows),
      sourceTimestamps,
      epochMilliseconds: Float64Array.from(epochMilliseconds),
      timestamps: Float32Array.from(timestamps),
      properties
    };
  });
  for (const batch of preparedBatches) {
    for (const property of sourceStyleProperties)
      ensureGeometryProperty(batch.properties, property, batch.rowCount);
  }
  return {
    kind: 'geojson-linestring-trips',
    timestampEncoding: options.timestampEncoding,
    originMilliseconds,
    maximumTimeErrorMilliseconds,
    actualTimeErrorMilliseconds,
    animationDomainMilliseconds: tripCount ? [minimumTime, maximumTime] : null,
    rowCount,
    tripCount,
    vertexCount,
    crs: options.crs,
    batches: preparedBatches,
    invalidRows,
    sourceStyleProperties: [...sourceStyleProperties],
    invalidStyleProperties,
    typedStorageByteLength: getTripTypedStorageByteLength(preparedBatches)
  };
}

/** Serialized GeoJSON Arrow rows retain each physical chunk/slice as a separate source batch. */
export function prepareKeplerGeoJSONTripsFromArrow(
  vector: Vector,
  options: KeplerGeoJSONTripPreparationOptions & {batchIds?: readonly string[]}
): PreparedKeplerGeoJSONTrips {
  if (options.batchIds && options.batchIds.length !== vector.data.length)
    throw new Error('Trip batch IDs must match Arrow chunks');
  return prepareKeplerGeoJSONTrips(
    vector.data.map((chunk, index) => {
      const values = new Vector([chunk]);
      return {
        id: options.batchIds?.[index] ?? `trip-batch-${index}`,
        rows: Array.from({length: values.length}, (_, rowIndex) => values.get(rowIndex))
      };
    }),
    options
  );
}

function parseTrips(
  row: unknown,
  sourceRowIndex: number,
  encoding: TripTimestampEncoding
): ParsedTrip[] {
  const value: unknown = typeof row === 'string' ? JSON.parse(row) : row;
  if (!isRecord(value)) throw new Error('Trip row must be GeoJSON');
  const geometry = value['type'] === 'Feature' ? value['geometry'] : value;
  if (!isRecord(geometry) || !['LineString', 'MultiLineString'].includes(String(geometry['type'])))
    throw new Error('Timed geometry requires a GeoJSON LineString or MultiLineString');
  const parts =
    geometry['type'] === 'LineString' ? [geometry['coordinates']] : geometry['coordinates'];
  if (!Array.isArray(parts) || !parts.length) throw new Error('Trip requires at least one path');
  return parts.map(coordinates => parseTripPart(coordinates, sourceRowIndex, encoding));
}

function parseTripPart(
  coordinates: unknown,
  sourceRowIndex: number,
  encoding: TripTimestampEncoding
): ParsedTrip {
  if (!Array.isArray(coordinates) || coordinates.length < 2)
    throw new Error('Trip LineString requires at least two vertices');
  const trip: ParsedTrip = {
    sourceRowIndex,
    positions: [],
    sourceTimestamps: [],
    epochMilliseconds: []
  };
  for (const coordinate of coordinates) {
    if (!Array.isArray(coordinate) || coordinate.length !== 4)
      throw new Error('Trip vertices require exactly [x, y, z, time]');
    for (const position of coordinate.slice(0, 3)) {
      if (typeof position !== 'number' || !Number.isFinite(Math.fround(position)))
        throw new Error('Trip XYZ coordinates must be finite Float32-representable values');
      trip.positions.push(position);
    }
    const sourceTime: unknown = coordinate[3];
    const time = parseTripTime(sourceTime, encoding);
    if (
      trip.epochMilliseconds.length &&
      time < trip.epochMilliseconds[trip.epochMilliseconds.length - 1]
    )
      throw new Error('Trip timestamps must be nondecreasing in source vertex order');
    // parseTime validates the original value and never rewrites it.
    trip.sourceTimestamps.push(sourceTime as number | string);
    trip.epochMilliseconds.push(time);
  }
  return trip;
}
export function parseTripTime(value: unknown, encoding: TripTimestampEncoding): number {
  let milliseconds: number;
  if (encoding === 'iso8601') {
    if (
      typeof value !== 'string' ||
      !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,3})?(?:Z|[+-]\d{2}:\d{2})$/.test(value)
    )
      throw new Error(
        'ISO trip times require an explicit timezone and at most millisecond precision'
      );
    const year = Number(value.slice(0, 4));
    const month = Number(value.slice(5, 7));
    const day = Number(value.slice(8, 10));
    const days = [
      31,
      year % 4 === 0 && (year % 100 !== 0 || year % 400 === 0) ? 29 : 28,
      31,
      30,
      31,
      30,
      31,
      31,
      30,
      31,
      30,
      31
    ];
    if (
      month < 1 ||
      month > 12 ||
      day < 1 ||
      day > days[month - 1] ||
      Number(value.slice(11, 13)) > 23
    )
      throw new Error('Invalid ISO trip date');
    milliseconds = Date.parse(value);
  } else {
    if (typeof value !== 'number' || !Number.isFinite(value))
      throw new Error('Numeric trip timestamps require finite numbers');
    milliseconds = encoding === 'unix-second' ? value * 1000 : value;
    if (encoding === 'unix-second' && milliseconds / 1000 !== value)
      throw new Error('Trip seconds must round-trip exactly through epoch milliseconds');
  }
  if (!Number.isSafeInteger(milliseconds))
    throw new Error('Trip timestamp cannot be represented as exact integral epoch milliseconds');
  return milliseconds;
}
function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}
