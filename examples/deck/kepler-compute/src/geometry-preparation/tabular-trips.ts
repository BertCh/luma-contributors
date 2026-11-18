// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {DataType, type Table} from 'apache-arrow';
import {convertKeplerTimeInstant, type GPUKeplerTimeUnit} from '@luma-contributors/kepler';
import {readOriginalTimestampTicks, resolveOriginalColumnRow} from '../original-column-values';
import {
  parseTripTime,
  type KeplerGeoJSONTripPreparationOptions,
  type PreparedGeoJSONTripBatch,
  type PreparedKeplerGeoJSONTrips
} from './geojson-trips';
import {getTripTypedStorageByteLength, quantizeTripTimes, validateTripOptions} from './trip-time';

/** One physical source batch of tabular trip columns; arrays align by local row. */
export type TabularTripSourceBatch = {
  id: string;
  tripIds: readonly unknown[];
  timestamps: readonly unknown[];
  longitudes: readonly unknown[];
  latitudes: readonly unknown[];
  altitudes?: readonly unknown[];
  /** Optional explicit part identity within each trip; parts never join across this value. */
  partIds?: readonly unknown[];
};
export type TabularTripColumns = {
  tripId: string;
  timestamp: string;
  longitude: string;
  latitude: string;
  altitude?: string;
  partId?: string;
};
export type TabularTripPreparationOptions = KeplerGeoJSONTripPreparationOptions;
/** Explicit grouping index: per-trip member source rows in timestamp order (CSR layout). */
export type PreparedTabularTripBatch = PreparedGeoJSONTripBatch & {
  /** `sourceRows.length + 1` offsets into `memberRows`. */
  memberOffsets: Uint32Array;
  /** Global source rows contributing vertices, ordered by timestamp then source row. */
  memberRows: Uint32Array;
};
export type PreparedTabularTrips = Omit<PreparedKeplerGeoJSONTrips, 'kind' | 'batches'> & {
  kind: 'tabular-trips';
  batches: readonly PreparedTabularTripBatch[];
};

type Vertex = {
  row: number;
  time: number;
  sourceTime: number | string;
  x: number;
  y: number;
  z: number;
};
type Group = {ownerBatch: number; vertices: Vertex[]};

/**
 * Groups timed points by an explicit trip ID and orders each group by timestamp.
 *
 * Matching the pinned native tabular Trips: rows with a null ID are skipped, rows with a
 * non-finite coordinate or missing timestamp contribute no vertex, vertices sort ascending by
 * time, and a trip is visible when at least one member row is selected. Native also removes
 * unselected vertices from the path; the runtime reproduces that on the GPU from the membership
 * index without changing these prepared buffers. Unlike native, ties retain source row order and IDs are compared as strings in first-valid-row
 * order. The primary source row of a trip is its first valid source row. A trip belongs to the
 * physical batch of that primary row; source buffers are never repacked.
 */
export function prepareKeplerTabularTrips(
  batches: readonly TabularTripSourceBatch[],
  options: TabularTripPreparationOptions
): PreparedTabularTrips {
  return prepareTabularTripBatches(batches, options);
}

function prepareTabularTripBatches(
  batches: readonly TabularTripSourceBatch[],
  options: TabularTripPreparationOptions,
  validatedMilliseconds?: readonly (readonly (number | undefined)[])[]
): PreparedTabularTrips {
  const maximumTimeErrorMilliseconds = validateTripOptions(options);
  const invalidRows: {sourceRowIndex: number; reason: string}[] = [];
  const tripGroups = new Map<string, Map<string | null, Group>>();
  const groups: Group[] = [];
  const batchIds = new Set<string>();
  const batchOffsets: number[] = [];
  let rowCount = 0;
  batches.forEach((batch, batchIndex) => {
    if (!batch.id || batchIds.has(batch.id)) throw new Error('Trip batch IDs must be unique');
    batchIds.add(batch.id);
    batchOffsets.push(rowCount);
    const length = batch.tripIds.length;
    for (const values of [
      batch.timestamps,
      batch.longitudes,
      batch.latitudes,
      batch.altitudes,
      batch.partIds
    ])
      if (values && values.length !== length)
        throw new Error('Trip source columns must have matching lengths');
    rowCount += length;
    if (!Number.isSafeInteger(rowCount) || rowCount > 0xffffffff)
      throw new Error('Trip source rows exceed uint32 identity capacity');
    for (let local = 0; local < length; local++) {
      const rawId = batch.tripIds[local];
      if (rawId === null || rawId === undefined) continue;
      const row = batchOffsets[batchIndex] + local;
      const x = batch.longitudes[local];
      const y = batch.latitudes[local];
      const altitude = batch.altitudes ? batch.altitudes[local] : 0;
      const z = altitude === null || altitude === undefined ? 0 : altitude;
      const sourceTime = readSourceTime(batch.timestamps[local]);
      let time: number;
      try {
        if (sourceTime === null) throw new Error('Trip row has no timestamp');
        time =
          validatedMilliseconds?.[batchIndex]?.[local] ??
          parseTripTime(sourceTime, options.timestampEncoding);
        for (const value of [x, y, z]) {
          if (typeof value !== 'number' || !Number.isFinite(Math.fround(value)))
            throw new Error('Trip XYZ coordinates must be finite Float32-representable values');
        }
      } catch (error) {
        invalidRows.push({
          sourceRowIndex: row,
          reason: error instanceof Error ? error.message : String(error)
        });
        continue;
      }
      const key = String(rawId);
      let parts = tripGroups.get(key);
      if (!parts) {
        parts = new Map();
        tripGroups.set(key, parts);
      }
      const partId = batch.partIds?.[local];
      const partKey = partId === null || partId === undefined ? null : String(partId);
      let group = parts.get(partKey);
      if (!group) {
        group = {ownerBatch: batchIndex, vertices: []};
        parts.set(partKey, group);
        groups.push(group);
      }
      group.vertices.push({
        row,
        time,
        sourceTime: sourceTime as number | string,
        x: x as number,
        y: y as number,
        z: z as number
      });
    }
  });
  let minimumSourceTime = Infinity;
  const accepted: Group[] = [];
  for (const group of groups) {
    if (group.vertices.length < 2) {
      for (const vertex of group.vertices)
        invalidRows.push({
          sourceRowIndex: vertex.row,
          reason: 'Trip requires at least two valid timed vertices'
        });
      continue;
    }
    // Array sort is stable, so equal timestamps retain source row order.
    group.vertices.sort((first, second) => first.time - second.time);
    minimumSourceTime = Math.min(minimumSourceTime, group.vertices[0].time);
    accepted.push(group);
  }
  const originMilliseconds =
    options.originMilliseconds ?? (Number.isFinite(minimumSourceTime) ? minimumSourceTime : 0);
  let actualTimeErrorMilliseconds = 0;
  let minimumTime = Infinity;
  let maximumTime = -Infinity;
  let tripCount = 0;
  let vertexCount = 0;
  const builders = batches.map(() => ({
    positions: [] as number[],
    sourceRows: [] as number[],
    sourceTimestamps: [] as (number | string)[],
    epochMilliseconds: [] as number[],
    timestamps: [] as number[],
    valueOffsets: [0],
    memberRows: [] as number[],
    memberOffsets: [0]
  }));
  for (const group of accepted) {
    const quantized = quantizeTripTimes(
      group.vertices.map(vertex => vertex.time),
      originMilliseconds,
      maximumTimeErrorMilliseconds
    );
    if (!quantized.valid) {
      for (const vertex of group.vertices)
        invalidRows.push({sourceRowIndex: vertex.row, reason: quantized.reason});
      continue;
    }
    const {rounded, error} = quantized;
    const builder = builders[group.ownerBatch];
    if (builder.timestamps.length + rounded.length > 0x7fffffff)
      throw new Error('Trip batch vertices exceed Int32 offset capacity');
    // The primary row is the first valid source row, which is not necessarily the first in time.
    builder.sourceRows.push(
      group.vertices.reduce((minimum, vertex) => Math.min(minimum, vertex.row), Infinity)
    );
    for (const vertex of group.vertices) {
      builder.positions.push(vertex.x, vertex.y, vertex.z);
      builder.sourceTimestamps.push(vertex.sourceTime);
      builder.epochMilliseconds.push(vertex.time);
      builder.memberRows.push(vertex.row);
    }
    for (const value of rounded) builder.timestamps.push(value);
    builder.valueOffsets.push(builder.timestamps.length);
    builder.memberOffsets.push(builder.memberRows.length);
    minimumTime = Math.min(minimumTime, group.vertices[0].time);
    maximumTime = Math.max(maximumTime, group.vertices[group.vertices.length - 1].time);
    actualTimeErrorMilliseconds = Math.max(actualTimeErrorMilliseconds, error);
    tripCount++;
    vertexCount += rounded.length;
  }
  const preparedBatches = builders.map(
    (builder, index): PreparedTabularTripBatch => ({
      id: batches[index].id,
      rowCount: batches[index].tripIds.length,
      sourceRowOffset: batchOffsets[index],
      positions: Float32Array.from(builder.positions),
      valueOffsets: Int32Array.from(builder.valueOffsets),
      sourceRows: Uint32Array.from(builder.sourceRows),
      sourceTimestamps: builder.sourceTimestamps,
      epochMilliseconds: Float64Array.from(builder.epochMilliseconds),
      timestamps: Float32Array.from(builder.timestamps),
      memberOffsets: Uint32Array.from(builder.memberOffsets),
      memberRows: Uint32Array.from(builder.memberRows)
    })
  );
  return {
    kind: 'tabular-trips',
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
    sourceStyleProperties: [],
    invalidStyleProperties: [],
    typedStorageByteLength: getTripTypedStorageByteLength(preparedBatches)
  };
}

/**
 * Reads named Arrow columns per physical record batch, preserving batch boundaries.
 * Plain and dictionary timestamps use retained ticks and reject sub-millisecond precision loss
 * before conversion to the selected trip encoding. Validated integer milliseconds are carried
 * separately so formatting the source timestamp as numeric seconds cannot alter the trip clock.
 */
export function prepareKeplerTabularTripsFromArrow(
  table: Table,
  columns: TabularTripColumns,
  options: TabularTripPreparationOptions & {batchIds?: readonly string[]}
): PreparedTabularTrips {
  if (options.batchIds && options.batchIds.length !== table.batches.length)
    throw new Error('Trip batch IDs must match Arrow batches');
  const validatedMilliseconds: (number | undefined)[][] = table.batches.map(() => []);
  const readColumnValues = (
    batch: Table['batches'][number],
    name: string,
    timestampMilliseconds?: (number | undefined)[]
  ): unknown[] => {
    const column = batch.getChild(name);
    if (!column) throw new Error('Trip column is missing from the retained source');
    if (name === columns.timestamp) {
      return Array.from({length: column.length}, (_, row) => {
        const original = resolveOriginalColumnRow(column, row);
        if (!original) return null;
        const {vector: sourceVector, row: sourceRow} = original;
        if (!DataType.isTimestamp(sourceVector.type)) return sourceVector.get(sourceRow);
        const unit = (['second', 'millisecond', 'microsecond', 'nanosecond'] as const)[
          sourceVector.type.unit
        ] as GPUKeplerTimeUnit;
        // Arrow's getter narrows nanoseconds to Number before scaling; a modern +1 ns instant
        // can become an apparently integral millisecond. Refuse that loss before narrowing.
        const ticks = readOriginalTimestampTicks(sourceVector, sourceRow);
        const milliseconds = Number(convertKeplerTimeInstant(ticks, unit, 'millisecond'));
        if (!Number.isSafeInteger(milliseconds))
          throw new Error('Trip timestamp requires exact integral epoch milliseconds');
        // Preserve validated milliseconds separately: encoding numeric seconds and parsing them
        // again can lose a vertex even when the Arrow source is an exact integer millisecond.
        if (timestampMilliseconds) timestampMilliseconds[row] = milliseconds;
        if (options.timestampEncoding === 'unix-second') return milliseconds / 1000;
        if (options.timestampEncoding === 'iso8601') return new Date(milliseconds).toISOString();
        return milliseconds;
      });
    }
    return Array.from({length: column.length}, (_, index) => column.get(index));
  };
  return prepareTabularTripBatches(
    table.batches.map((batch, index) => ({
      id: options.batchIds?.[index] ?? `trip-batch-${index}`,
      tripIds: readColumnValues(batch, columns.tripId),
      timestamps: readColumnValues(batch, columns.timestamp, validatedMilliseconds[index]),
      longitudes: readColumnValues(batch, columns.longitude),
      latitudes: readColumnValues(batch, columns.latitude),
      ...(columns.altitude ? {altitudes: readColumnValues(batch, columns.altitude)} : {}),
      ...(columns.partId ? {partIds: readColumnValues(batch, columns.partId)} : {})
    })),
    options,
    validatedMilliseconds
  );
}

function readSourceTime(value: unknown): number | string | null {
  if (typeof value === 'bigint') return Number.isSafeInteger(Number(value)) ? Number(value) : null;
  return typeof value === 'number' || typeof value === 'string' ? value : null;
}

export function isKeplerTabularTrips(
  trips: PreparedKeplerGeoJSONTrips
): trips is PreparedTabularTrips {
  return trips.kind === 'tabular-trips';
}
