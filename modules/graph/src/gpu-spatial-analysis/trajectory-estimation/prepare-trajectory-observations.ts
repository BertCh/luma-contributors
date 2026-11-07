// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import {
  createLocalFrameBasis,
  eastNorthUpBasis,
  fixedToLocal,
  spheroidToCartesian
} from '@math.gl/core';

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
  assert(
    props.longitudeLatitude.length % 2 === 0 &&
      props.timestamps.length === props.longitudeLatitude.length / 2
  );
  assert(props.unitsPerSecond > 0n && Number.isFinite(Number(props.unitsPerSecond)));
  const [longitude, latitude] = props.origin;
  assert(
    Number.isFinite(longitude) &&
      longitude >= -180 &&
      longitude <= 180 &&
      Number.isFinite(latitude) &&
      latitude >= -90 &&
      latitude <= 90
  );
  const radians = Math.PI / 180;
  const anchor = {x: 0, y: 0, z: 0};
  const location = {x: 0, y: 0, z: 0};
  const local = {x: 0, y: 0, z: 0};
  const basis = createLocalFrameBasis();
  assert(
    spheroidToCartesian(
      {x: longitude * radians, y: latitude * radians, z: 0},
      WGS84_SPHEROID,
      anchor
    ) && eastNorthUpBasis(longitude * radians, latitude * radians, basis)
  );
  const positions = new Float32Array(props.longitudeLatitude.length);
  const times = new Float32Array(props.timestamps.length);
  for (let row = 0; row < times.length; row++) {
    const inputLongitude = props.longitudeLatitude[row * 2];
    const inputLatitude = props.longitudeLatitude[row * 2 + 1];
    // Nonfinite observations remain missing measurements for the filter.
    assert(!Number.isFinite(inputLongitude) || (inputLongitude >= -180 && inputLongitude <= 180));
    assert(!Number.isFinite(inputLatitude) || (inputLatitude >= -90 && inputLatitude <= 90));
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
    assert(Number.isFinite(times[row]));
  }
  return {
    positions,
    times,
    projection: {
      kind: 'wgs84-local-east-north' as const,
      origin: props.origin,
      units: 'metres' as const
    },
    epoch: props.epoch,
    unitsPerSecond: props.unitsPerSecond
  };
}
