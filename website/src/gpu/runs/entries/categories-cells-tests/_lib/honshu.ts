// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// USGS earthquakes inside the Honshu window with their time (days since 2020-01-01), decoded from the
// checked-in extract `data/japan-earthquakes.json` (see data/point-sources.md). The shared point
// helpers expose positions and magnitudes only, so the time column is read here.

import seismicity from '../../../../data/japan-earthquakes.json';
import {HONSHU_WINDOW} from '../../../../data/point-observations';

export type HonshuEvents = {
  count: number;
  /** Longitude/latitude pairs in degrees. */
  positions: Float32Array;
  magnitude: Float32Array;
  /** Whole days since January 1, 2020. */
  day: Float32Array;
};

let cached: HonshuEvents | undefined;

/** Every event of the sample inside the Honshu window, in catalog order. */
export function getHonshuEventsWithTime(): HonshuEvents {
  if (cached) return cached;
  const bytes = Uint8Array.from(atob(seismicity.data), character => character.charCodeAt(0));
  const view = new DataView(bytes.buffer);
  const [west, south, east, north] = seismicity.bounds;
  const [windowWest, windowSouth, windowEast, windowNorth] = HONSHU_WINDOW;
  const positions: number[] = [];
  const magnitude: number[] = [];
  const day: number[] = [];
  for (let row = 0; row < bytes.length / 8; row++) {
    const longitude = west + ((east - west) * view.getUint16(row * 8, true)) / 10000;
    const latitude = south + ((north - south) * view.getUint16(row * 8 + 2, true)) / 10000;
    if (
      longitude < windowWest ||
      longitude > windowEast ||
      latitude < windowSouth ||
      latitude > windowNorth
    ) {
      continue;
    }
    positions.push(longitude, latitude);
    magnitude.push(view.getUint16(row * 8 + 4, true) / 100);
    day.push(view.getUint16(row * 8 + 6, true));
  }
  cached = {
    count: magnitude.length,
    positions: Float32Array.from(positions),
    magnitude: Float32Array.from(magnitude),
    day: Float32Array.from(day)
  };
  return cached;
}
