// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {KeplerGeoJSONTripPreparationOptions, PreparedGeoJSONTripBatch} from './geojson-trips';

/** Validates the options shared by GeoJSON and tabular trips and returns the time error budget. */
export function validateTripOptions(options: KeplerGeoJSONTripPreparationOptions): number {
  if (!['unix-millisecond', 'unix-second', 'iso8601'].includes(options.timestampEncoding))
    throw new Error('Choose an explicit trip timestamp encoding');
  const maximumTimeErrorMilliseconds = options.maximumTimeErrorMilliseconds ?? 0;
  if (!Number.isFinite(maximumTimeErrorMilliseconds) || maximumTimeErrorMilliseconds < 0)
    throw new Error('Trip time error budget must be finite and nonnegative');
  if (options.originMilliseconds !== undefined && !Number.isSafeInteger(options.originMilliseconds))
    throw new Error('Trip origin must be exact integral epoch milliseconds');
  return maximumTimeErrorMilliseconds;
}

export type QuantizedTripTimes =
  | {valid: true; rounded: number[]; error: number}
  | {valid: false; reason: string};

/**
 * Rebases epoch milliseconds on the origin and rounds them to Float32. Rejects the trip when the
 * rounding error exceeds the budget; the reason states the budget the trip would need.
 */
export function quantizeTripTimes(
  epochMilliseconds: readonly number[],
  originMilliseconds: number,
  maximumTimeErrorMilliseconds: number
): QuantizedTripTimes {
  const relative = epochMilliseconds.map(value => value - originMilliseconds);
  const rounded = relative.map(Math.fround);
  const error = relative.reduce(
    (maximum, value, index) => Math.max(maximum, Math.abs(rounded[index] - value)),
    0
  );
  if (
    relative.some(value => !Number.isSafeInteger(value)) ||
    !Number.isFinite(error) ||
    error > maximumTimeErrorMilliseconds
  ) {
    const requirement = Number.isFinite(error)
      ? ` (needs at least ${error} ms, budget is ${maximumTimeErrorMilliseconds} ms)`
      : '';
    return {
      valid: false,
      reason: `Relative trip times exceed the explicit Float32 error budget${requirement}`
    };
  }
  return {valid: true, rounded, error};
}

/** Retained typed storage of prepared trip batches, including optional extra typed arrays. */
export function getTripTypedStorageByteLength(
  batches: readonly (PreparedGeoJSONTripBatch & {
    memberOffsets?: Uint32Array;
    memberRows?: Uint32Array;
  })[]
): number {
  return batches.reduce(
    (total, batch) =>
      total +
      batch.positions.byteLength +
      batch.valueOffsets.byteLength +
      batch.sourceRows.byteLength +
      batch.epochMilliseconds.byteLength +
      batch.timestamps.byteLength +
      (batch.memberOffsets?.byteLength ?? 0) +
      (batch.memberRows?.byteLength ?? 0) +
      Object.values(batch.properties ?? {}).reduce(
        (bytes, channel) => bytes + channel.values.byteLength + channel.validity.byteLength,
        0
      ),
    0
  );
}
