// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** WGS84/meters acceptance envelopes, checked by the headless spec against PROJ fixtures. */
export const CR01_DISTANCE_ERROR_METERS = 12;
export const CR01_BEARING_ERROR_DEGREES = 0.004;
export const CR01_MIDPOINT_ERROR_METERS = 128;
export const CR01_DESTINATION_ERROR_METERS = 12;

/** Double haversine of coordinate differences; independent of the shader's local-frame form. */
export function getPositionErrorMeters(
  actual: readonly number[],
  expected: readonly number[]
): number {
  const degreesToRadians = Math.PI / 180;
  const latitudeDelta = (actual[1] - expected[1]) * degreesToRadians;
  const longitudeDelta = (actual[0] - expected[0]) * degreesToRadians;
  const term =
    Math.sin(latitudeDelta / 2) ** 2 +
    Math.cos(actual[1] * degreesToRadians) *
      Math.cos(expected[1] * degreesToRadians) *
      Math.sin(longitudeDelta / 2) ** 2;
  return 2 * 6371008.8 * Math.asin(Math.sqrt(Math.min(1, Math.max(0, term))));
}

/** Azimuths compare modulo a turn, including -180/+180 and signed zero. */
export function getBearingErrorDegrees(actual: number, expected: number): number {
  const difference = actual - expected;
  return Math.abs(difference - 360 * Math.round(difference / 360));
}
