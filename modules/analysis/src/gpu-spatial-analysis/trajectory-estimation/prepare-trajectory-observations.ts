// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {splitTimeWords} from '../../dataframe-analysis/time-window-filter/time-words';
import {
  createLocalFrameBasis,
  eastNorthUpBasis,
  fixedToLocal,
  spheroidToCartesian
} from '@math.gl/core';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';
import type {GPUSpatialContext} from '../../analysis-core/contracts/spatial-context';
import type {GPUTemporalContext} from '../../analysis-core/contracts/temporal-context';

const OPERATION = 'prepareGPUTrajectoryObservations';

const WGS84_SPHEROID = {
  semiMajorAxis: 6378137,
  semiMinorAxis: 6356752.314245179,
  eccentricitySquared: 6.6943799901413165e-3
};

export type GPUTrajectoryObservationPreparation = {
  /** WGS84 longitude/latitude degrees, kept in host f64 until local-origin subtraction. */
  longitudeLatitude: ArrayLike<number>;
  /** Absolute integer timestamps; bigint preserves nanosecond epochs before subtraction. */
  timestamps: ArrayLike<bigint>;
  epoch: bigint;
  unitsPerSecond: bigint;
  /** Explicit local tangent-plane origin [longitude,latitude] in degrees. */
  origin: readonly [number, number];
};

/**
 * Explicit host coordinate/epoch preparation, never an estimator or implicit GPU fallback.
 * WGS84 ellipsoidal ECEF is rotated into the origin's east/north tangent plane. Height is zero;
 * curvature is projected away. Retain these metadata to interpret metre-space covariance.
 * Callers wanting a map projection can instead upload their already projected coordinates.
 */
export function prepareGPUTrajectoryObservations(props: GPUTrajectoryObservationPreparation) {
  validateRequirement(
    OPERATION,
    props.longitudeLatitude.length % 2 === 0 &&
      props.timestamps.length === props.longitudeLatitude.length / 2,
    'longitudeLatitude must hold two values per timestamp'
  );
  validateRequirement(
    OPERATION,
    props.unitsPerSecond > 0n && Number.isFinite(Number(props.unitsPerSecond)),
    'unitsPerSecond must be a positive finite integer'
  );
  const [longitude, latitude] = props.origin;
  // The origin feeds float64 host math, so it is compared as a double; the named-limit helper
  // rounds to float32 and would accept 180.000001.
  validateRequirement(
    OPERATION,
    Number.isFinite(longitude) && longitude >= -180 && longitude <= 180,
    'origin longitude must be a finite number in [-180, 180]'
  );
  validateRequirement(
    OPERATION,
    Number.isFinite(latitude) && latitude >= -90 && latitude <= 90,
    'origin latitude must be a finite number in [-90, 90]'
  );
  const radians = Math.PI / 180;
  const anchor = {x: 0, y: 0, z: 0};
  const location = {x: 0, y: 0, z: 0};
  const local = {x: 0, y: 0, z: 0};
  const basis = createLocalFrameBasis();
  validateRequirement(
    OPERATION,
    Boolean(
      spheroidToCartesian(
        {x: longitude * radians, y: latitude * radians, z: 0},
        WGS84_SPHEROID,
        anchor
      ) && eastNorthUpBasis(longitude * radians, latitude * radians, basis)
    ),
    'origin must project onto the WGS84 spheroid'
  );
  const positions = new Float32Array(props.longitudeLatitude.length);
  const times = new Float32Array(props.timestamps.length);
  const timeWords = new Uint32Array(props.timestamps.length * 2);
  for (let row = 0; row < times.length; row++) {
    const inputLongitude = props.longitudeLatitude[row * 2];
    const inputLatitude = props.longitudeLatitude[row * 2 + 1];
    // Nonfinite observations remain missing measurements for the filter.
    validateRequirement(
      OPERATION,
      !Number.isFinite(inputLongitude) || (inputLongitude >= -180 && inputLongitude <= 180),
      `longitudeLatitude longitude at row ${row} must be within [-180, 180] degrees`
    );
    validateRequirement(
      OPERATION,
      !Number.isFinite(inputLatitude) || (inputLatitude >= -90 && inputLatitude <= 90),
      `longitudeLatitude latitude at row ${row} must be within [-90, 90] degrees`
    );
    location.x = inputLongitude * radians;
    location.y = inputLatitude * radians;
    location.z = 0;
    if (spheroidToCartesian(location, WGS84_SPHEROID, location)) {
      location.x -= anchor.x;
      location.y -= anchor.y;
      location.z -= anchor.z;
      fixedToLocal(location, basis, local);
      positions[row * 2] = local.x;
      positions[row * 2 + 1] = local.y;
    } else {
      positions[row * 2] = NaN;
      positions[row * 2 + 1] = NaN;
    }
    const delta = props.timestamps[row] - props.epoch;
    // Integer quotient/remainder prevents a large absolute epoch from consuming the f32 mantissa.
    times[row] =
      Number(delta / props.unitsPerSecond) +
      Number(delta % props.unitsPerSecond) / Number(props.unitsPerSecond);
    validateRequirement(
      OPERATION,
      Number.isFinite(times[row]),
      `timestamps at row ${row} minus epoch must convert to a finite number of seconds`
    );
    // Word rows keep every timestamp exact; `times` is only precise within weeks of the epoch.
    const words = splitTimeWords(props.timestamps[row]);
    timeWords[row * 2] = words.low;
    timeWords[row * 2 + 1] = words.high;
  }
  const origin = Object.freeze([longitude, latitude] as const);
  const wordUnit =
    props.unitsPerSecond === 1n
      ? 'second'
      : props.unitsPerSecond === 1000n
        ? 'millisecond'
        : props.unitsPerSecond === 1000000n
          ? 'microsecond'
          : props.unitsPerSecond === 1000000000n
            ? 'nanosecond'
            : undefined;
  const timeWordsTemporalContext: GPUTemporalContext | undefined = wordUnit
    ? Object.freeze({
        encoding: 'int64-words',
        unit: wordUnit,
        epoch: 'unix',
        axis: Object.freeze({kind: 'irregular'})
      })
    : undefined;
  const epochSeconds = props.epoch / props.unitsPerSecond;
  const timesTemporalContext: GPUTemporalContext | undefined =
    props.epoch % props.unitsPerSecond === 0n &&
    epochSeconds >= -(1n << 63n) &&
    epochSeconds < 1n << 63n
      ? Object.freeze({
          encoding: 'float32-relative',
          unit: 'second',
          epoch: epochSeconds,
          axis: Object.freeze({kind: 'irregular'})
        })
      : undefined;
  const spatialContext: GPUSpatialContext = Object.freeze({
    coordinateSpace: 'planar',
    metric: 'native',
    units: 'meters',
    datum: 'EPSG:4326',
    frame: Object.freeze({
      id: `wgs84-local-east-north:${longitude}:${latitude}:height-zero`,
      kind: 'crs',
      crs: 'LOCAL:WGS84-EAST-NORTH',
      origin: Object.freeze([longitude, latitude]) as unknown as [number, number],
      anchor: 'projection-center',
      axisOrder: 'enu',
      unit: Object.freeze({name: 'meter', metersPerUnit: 1})
    })
  });
  return {
    positions,
    /** Elapsed seconds after the exact epochTicks / epochUnitsPerSecond origin in timesMetadata. */
    times,
    /** Absolute source ticks as Int64 `(low, high)` words, two per row; not elapsed seconds. */
    timeWords,
    /** Exact source epoch and rate remain meaningful when the epoch is a fractional second. */
    timesMetadata: Object.freeze({
      encoding: 'float32-relative' as const,
      unit: 'second' as const,
      epochTicks: props.epoch,
      epochUnitsPerSecond: props.unitsPerSecond
    }),
    /** Available only when the epoch is an integral number of seconds. */
    timesTemporalContext,
    /** Available for source rates 1, 1000, 1000000 and 1000000000 ticks per second. */
    timeWordsTemporalContext,
    spatialContext,
    projection: {
      kind: 'wgs84-local-east-north' as const,
      origin,
      units: 'metres' as const
    },
    /** Source epoch ticks; use timesMetadata to interpret the elapsed-seconds stream. */
    epoch: props.epoch,
    /** Source tick rate, used by timeWords; times is already in seconds. */
    unitsPerSecond: props.unitsPerSecond
  };
}
