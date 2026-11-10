// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// CPU lookups on a fleet: the track under the pointer, a track's polyline and where it is at a time.

import {findNearestOnPolyline} from '../../../pick';
import type {FleetInput} from './fleet';

/** The track whose polyline passes nearest to (x, y), with the distance to it. */
export function findNearestTrack(
  fleet: FleetInput,
  x: number,
  y: number
): {track: number; distance: number} {
  let best = {track: -1, distance: Infinity};
  for (let track = 0; track < fleet.trackCount; track++) {
    const start = fleet.trackOffsets[track];
    const hit = findNearestOnPolyline(
      fleet.positions,
      x,
      y,
      start,
      fleet.trackOffsets[track + 1] - start
    );
    if (hit && hit.distance < best.distance) best = {track, distance: hit.distance};
  }
  return best;
}

/** The packed (x, y) vertices of one track, for a `polyline` glyph. */
export function getTrackVertices(fleet: FleetInput, track: number): Float32Array {
  return fleet.positions.subarray(fleet.trackOffsets[track] * 2, fleet.trackOffsets[track + 1] * 2);
}

/** Position of a track at `time` (playback seconds), interpolated between its fixes; null outside its span. */
export function getTrackPositionAt(
  fleet: FleetInput,
  track: number,
  time: number
): [number, number] | null {
  const start = fleet.trackOffsets[track];
  const end = fleet.trackOffsets[track + 1];
  if (time < fleet.timestamps[start] || time > fleet.timestamps[end - 1]) return null;
  for (let row = start; row < end - 1; row++) {
    const first = fleet.timestamps[row];
    const second = fleet.timestamps[row + 1];
    if (time >= first && time <= second) {
      const fraction = second > first ? (time - first) / (second - first) : 0;
      return [
        fleet.positions[row * 2] +
          (fleet.positions[row * 2 + 2] - fleet.positions[row * 2]) * fraction,
        fleet.positions[row * 2 + 1] +
          (fleet.positions[row * 2 + 3] - fleet.positions[row * 2 + 1]) * fraction
      ];
    }
  }
  return null;
}

/** The fix with the most other fixes within `radius`: a busy spot to rest a pointer or an anchor on. */
export function findBusiestFix(fleet: FleetInput, radius: number): [number, number] {
  let best: [number, number] = [fleet.worldWidth / 2, 0.5];
  let bestCount = -1;
  for (let row = 0; row < fleet.rowCount; row++) {
    const x = fleet.positions[row * 2];
    const y = fleet.positions[row * 2 + 1];
    let count = 0;
    for (let other = 0; other < fleet.rowCount; other++) {
      if (Math.hypot(fleet.positions[other * 2] - x, fleet.positions[other * 2 + 1] - y) < radius) {
        count++;
      }
    }
    if (count > bestCount) {
      bestCount = count;
      best = [x, y];
    }
  }
  return best;
}
