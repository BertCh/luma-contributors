// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Icon source-row visibility, mirroring tabular point altitude semantics: longitude and latitude
 * must be finite numbers; a null/undefined altitude renders at zero; a non-null nonfinite or
 * non-numeric altitude hides only that icon (the source row keeps its identity).
 */
export function isIconCoordinateVisible(
  longitude: unknown,
  latitude: unknown,
  altitude: unknown,
  hasAltitudeColumn: boolean
): boolean {
  if (!Number.isFinite(longitude) || !Number.isFinite(latitude)) return false;
  if (!hasAltitudeColumn || altitude === null || altitude === undefined) return true;
  return Number.isFinite(altitude);
}
