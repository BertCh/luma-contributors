// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Shared trajectory rounding: WGSL round uses nearest-even at a half-integer. @internal */
export function roundTrajectoryEven(value: number): number {
  const lower = Math.floor(value);
  const fraction = value - lower;
  return lower + Number(fraction > 0.5 || (fraction === 0.5 && lower % 2 !== 0));
}

/** Float64 transcription of geodesicLongitudeDelta, including its exact seam branch. @internal */
export function getCPUTrajectoryLongitudeDelta(source: number, destination: number): number {
  if (
    Math.abs(source) > 90 &&
    Math.abs(source) <= 180 &&
    Math.abs(destination) > 90 &&
    Math.abs(destination) <= 180 &&
    source < 0 !== destination < 0
  ) {
    const sourceUnits = source * 131072;
    const destinationUnits = destination * 131072;
    const turnUnits = source > 0 ? 47185920 : -47185920;
    return (destinationUnits - sourceUnits + turnUnits) / 131072;
  }
  const halfTurn = 180 * roundTrajectoryEven((destination - source) / 360);
  return destination - halfTurn - (source + halfTurn);
}

/**
 * Trajectory distance in float64, following movement-common and geodesicCentralAngle step by step.
 * Undefined radius selects the planar metric. Positions are packed float32x2 input rows or
 * float64 arithmetic centers; no intermediate center rounding is performed. @internal
 */
export function getCPUTrajectoryDistance(
  positions: ArrayLike<number>,
  first: number,
  second: number,
  radius: number | undefined
): number {
  const firstLongitude = positions[first * 2];
  const secondLongitude = positions[second * 2];
  const firstLatitude = positions[first * 2 + 1];
  const secondLatitude = positions[second * 2 + 1];
  if (radius === undefined)
    return Math.hypot(secondLongitude - firstLongitude, secondLatitude - firstLatitude);
  const radians = Math.PI / 180;
  const longitudeAngle = getCPUTrajectoryLongitudeDelta(firstLongitude, secondLongitude) * radians;
  const latitudeAngle = (secondLatitude - firstLatitude) * radians;
  const firstSine = Math.sin(firstLatitude * radians);
  const firstCosine = Math.sin((90 - Math.abs(firstLatitude)) * radians);
  const secondCosine = Math.sin((90 - Math.abs(secondLatitude)) * radians);
  const halfSine = Math.sin(0.5 * longitudeAngle);
  const halfVersine = 2 * halfSine * halfSine;
  const east = secondCosine * Math.sin(longitudeAngle);
  const north = Math.sin(latitudeAngle) + firstSine * secondCosine * halfVersine;
  const up =
    firstSine * Math.sin(secondLatitude * radians) + firstCosine * secondCosine * (1 - halfVersine);
  return radius * Math.atan2(Math.sqrt(east * east + north * north), up);
}
