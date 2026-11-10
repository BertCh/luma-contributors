// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Independent CPU reference of trackintel's `sliding` rule in binary64, written from the trackintel
// source and checked against the trackintel fixture in trackintel-parity.spec.ts.

export type ReferenceStaypoint = {
  track: number;
  startRow: number;
  endRow: number;
  finishRow: number;
  centroid: [number, number];
  duration: number;
};

export type ReferenceSettings = {
  distanceThreshold: number;
  timeThreshold: number;
  maximumGap?: number;
  includeLast?: boolean;
  metric: (a: number[], b: number[]) => number;
  /** Angular (longitude-wrapping) centroid when positions are degrees. */
  degrees?: boolean;
};

function makeCentroid(rows: number[][], degrees: boolean): [number, number] {
  const seen = new Set<string>();
  const distinct: number[][] = [];
  for (const row of rows) {
    const key = `${row[0]},${row[1]}`;
    if (!seen.has(key)) {
      seen.add(key);
      distinct.push(row);
    }
  }
  const count = distinct.length;
  const y = distinct.reduce((sum, row) => sum + row[1], 0) / count;
  if (!degrees) {
    return [distinct.reduce((sum, row) => sum + row[0], 0) / count, y];
  }
  let sine = 0;
  let cosine = 0;
  for (const row of distinct) {
    const angle = (row[0] * Math.PI) / 180;
    sine += Math.sin(angle);
    cosine += Math.cos(angle);
  }
  return [(Math.atan2(sine / count, cosine / count) * 180) / Math.PI, y];
}

/** Sequential trackintel-style sliding window per track. */
export function detectStaypointsReference(
  positions: number[][],
  times: number[],
  trackOffsets: number[],
  settings: ReferenceSettings
): ReferenceStaypoint[] {
  const {distanceThreshold, timeThreshold, maximumGap = 0, includeLast = false, metric} = settings;
  const result: ReferenceStaypoint[] = [];
  for (let track = 0; track + 1 < trackOffsets.length; track++) {
    const begin = trackOffsets[track];
    const end = trackOffsets[track + 1];
    if (end - begin === 0) {
      continue;
    }
    let start = begin;
    let current = begin;
    for (current = begin + 1; current < end; current++) {
      if (maximumGap > 0 && times[current] - times[current - 1] > maximumGap) {
        start = current;
        continue;
      }
      if (metric(positions[start], positions[current]) >= distanceThreshold) {
        if (times[current] - times[start] >= timeThreshold) {
          result.push({
            track,
            startRow: start,
            endRow: current,
            finishRow: current,
            centroid: makeCentroid(positions.slice(start, current), settings.degrees ?? false),
            duration: times[current] - times[start]
          });
        }
        start = current;
      }
    }
    if (includeLast && end - begin > 1) {
      const last = end - 1;
      if (times[last] - times[start] >= timeThreshold) {
        result.push({
          track,
          startRow: start,
          endRow: end,
          finishRow: last,
          centroid: makeCentroid(positions.slice(start, end), settings.degrees ?? false),
          duration: times[last] - times[start]
        });
      }
    }
  }
  return result;
}

/** Euclidean distance. */
export function euclideanDistance(a: number[], b: number[]): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

/** Haversine distance in the units of `radius`. */
export function haversineDistance(a: number[], b: number[], radius: number): number {
  const toRadians = Math.PI / 180;
  const latitudeA = a[1] * toRadians;
  const latitudeB = b[1] * toRadians;
  const deltaLatitude = (b[1] - a[1]) * toRadians;
  const deltaLongitude = (b[0] - a[0]) * toRadians;
  const sinLatitude = Math.sin(deltaLatitude / 2);
  const sinLongitude = Math.sin(deltaLongitude / 2);
  const haversine =
    sinLatitude * sinLatitude +
    Math.cos(latitudeA) * Math.cos(latitudeB) * sinLongitude * sinLongitude;
  return 2 * radius * Math.atan2(Math.sqrt(haversine), Math.sqrt(1 - haversine));
}
