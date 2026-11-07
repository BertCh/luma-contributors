// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Native Kepler formats timestamp labels without a display format as the date-time tooltip format
 * `TOOLTIP_FORMATS.DATE_TIME_L_LTS` (`format: 'L LTS'`, for example 09/05/2016 12:00:00 AM).
 */
export const DEFAULT_TIMESTAMP_LABEL_FORMAT = 'L LTS';

/** True for the Kepler field type whose label values are Unix milliseconds. */
export function isTimestampLabelFieldType(type: string | undefined): boolean {
  return type?.toLowerCase() === 'timestamp';
}

/**
 * Normalizes an Arrow timestamp cell (milliseconds number, bigint or Date) to the Unix
 * milliseconds native `valueAccessor` returns; unparseable cells stay null (an empty label).
 */
export function getTimestampMilliseconds(value: unknown): number | null {
  if (value === null || value === undefined) return null;
  const milliseconds =
    value instanceof Date ? value.getTime() : typeof value === 'bigint' ? Number(value) : value;
  return typeof milliseconds === 'number' && Number.isFinite(milliseconds) ? milliseconds : null;
}
