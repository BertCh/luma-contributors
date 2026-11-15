// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

const DEGREES_TO_RADIANS = Math.PI / 180;

function roundEven(value: number): number {
  const lower = Math.floor(value);
  const fraction = value - lower;
  return fraction < 0.5 || (fraction === 0.5 && lower % 2 === 0) ? lower : lower + 1;
}

/** Binary64 mirror of geodesicWrapLongitudeDelta; half-turn ties use round-to-even. @internal */
export function wrapLongitude(delta: number): number {
  return delta - 360 * roundEven(delta / 360);
}

/** Binary64 mirror of geodesicLongitudeDelta, shared by CPU geometry families. @internal */
export function computeLongitudeDelta(source: number, destination: number): number {
  // Same seam branch as geodesicLongitudeDelta. Float32 endpoints in these quadrants
  // are multiples of 2^-17, so these scaled differences are exact integers in binary64.
  if (
    Math.abs(source) > 90 &&
    Math.abs(source) <= 180 &&
    Math.abs(destination) > 90 &&
    Math.abs(destination) <= 180 &&
    source < 0 !== destination < 0
  ) {
    return (destination * 131072 - source * 131072 + (source > 0 ? 47185920 : -47185920)) / 131072;
  }
  const halfTurn = 180 * roundEven((destination - source) / 360);
  return destination - halfTurn - (source + halfTurn);
}

/** Binary64 mirror of geodesicCosLatitude. @internal */
export function computeCosLatitude(latitude: number): number {
  return Math.sin((90 - Math.abs(latitude)) * DEGREES_TO_RADIANS);
}

/** Binary64 mirror of geodesicBearingComponents. @internal */
export function computeBearingComponents(
  longitudeDelta: number,
  latitude: number,
  targetLatitude: number
): [number, number, number] {
  if (latitude === targetLatitude && (longitudeDelta === 0 || Math.abs(latitude) === 90))
    return [0, 0, 1];
  const sinLatitude = Math.sin(latitude * DEGREES_TO_RADIANS);
  const cosine = computeCosLatitude(targetLatitude);
  const halfVersine = 2 * Math.sin(longitudeDelta / 2) ** 2;
  let east = cosine * Math.sin(longitudeDelta);
  let north =
    Math.sin((targetLatitude - latitude) * DEGREES_TO_RADIANS) + sinLatitude * cosine * halfVersine;
  let longitudeCosine = 1 - halfVersine;
  const deltaDegrees = longitudeDelta / DEGREES_TO_RADIANS;
  if (Math.abs(deltaDegrees) > 90 && latitude * targetLatitude <= 0) {
    const supplement = (180 - Math.abs(deltaDegrees)) * DEGREES_TO_RADIANS;
    const halfSine = Math.sin(supplement / 2);
    east = cosine * Math.sign(deltaDegrees) * Math.sin(supplement);
    north =
      Math.sin((latitude + targetLatitude) * DEGREES_TO_RADIANS) -
      2 * sinLatitude * cosine * halfSine * halfSine;
    longitudeCosine = 2 * halfSine * halfSine - 1;
  }
  const vertical =
    sinLatitude * Math.sin(targetLatitude * DEGREES_TO_RADIANS) +
    computeCosLatitude(latitude) * cosine * longitudeCosine;
  return [east, north, vertical];
}

/** Binary64 mirror of geodesicCentralAngle, with a precomputed longitude delta. @internal */
export function computeCentralAngle(
  longitudeDelta: number,
  latitude: number,
  targetLatitude: number
): number {
  const [east, north, vertical] = computeBearingComponents(
    longitudeDelta,
    latitude,
    targetLatitude
  );
  return Math.atan2(Math.hypot(east, north), vertical);
}
