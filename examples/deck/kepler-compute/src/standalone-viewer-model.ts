// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {clamp} from '@math.gl/core';

/** Pure helpers for the offline standalone viewer (no DOM, no GPU). */

const LONGITUDE_NAMES = ['lng', 'lon', 'long', 'longitude', 'x'];
const LATITUDE_NAMES = ['lat', 'latitude', 'y'];

/** Prefer the first Kepler point layer's lat/lng columns, else well-known column names. */
export function findPointColumns(
  fieldNames: readonly string[],
  nativeConfig: unknown
): {longitude: string; latitude: string} | null {
  const queue: unknown[] = [nativeConfig];
  while (queue.length) {
    const value = queue.shift();
    if (!value || typeof value !== 'object') continue;
    const record = value as Record<string, any>;
    const columns = record['config']?.columns ?? record['columns'];
    const longitude = columns?.lng?.value ?? columns?.longitude?.value;
    const latitude = columns?.lat?.value ?? columns?.latitude?.value;
    if (fieldNames.includes(longitude) && fieldNames.includes(latitude))
      return {longitude, latitude};
    queue.push(...Object.values(record));
  }
  const find = (names: string[]) =>
    fieldNames.find(name => names.includes(name.toLowerCase().replace(/[^a-z]/g, '')));
  const longitude = find(LONGITUDE_NAMES);
  const latitude = find(LATITUDE_NAMES);
  return longitude && latitude ? {longitude, latitude} : null;
}

/** Web-mercator view that fits [west, south, east, north] in a viewport. */
export function getViewStateForBounds(
  bounds: readonly [number, number, number, number],
  width: number,
  height: number
): {longitude: number; latitude: number; zoom: number} {
  const [west, south, east, north] = bounds;
  const span = Math.max((east - west) / 360, (north - south) / 180, 1e-6);
  const size = Math.max(Math.min(width, height), 1);
  const zoom = clamp(Math.log2(size / 512 / span), 0, 20);
  return {longitude: (west + east) / 2, latitude: (south + north) / 2, zoom};
}
