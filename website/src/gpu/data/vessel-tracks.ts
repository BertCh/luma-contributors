// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import tracks from './sf-bay-ais.json';
import {fitLocalMap} from './fit-local-map';

const DAY_SECONDS = 86400;
export const AIS_PLAYBACK_SECONDS = 600;

/**
 * Resample observed San Francisco Bay vessel legs by fix index. Linear interpolation keeps each
 * scene's bounded buffer layout while preserving the positions and relative times of the legs.
 */
export function sampleVesselTracks(
  world: {width: number; height: number},
  trackCount: number,
  minimumFixes: number,
  maximumFixes: number,
  playbackSeconds = AIS_PLAYBACK_SECONDS
) {
  const selected = tracks.slice(0, trackCount);
  const project = fitLocalMap(selected.flat(), world);
  const points: number[] = [];
  const timestamps: number[] = [];
  const segments: number[] = [];
  const trackOffsets = [0];
  for (const track of selected) {
    const count = Math.max(minimumFixes, Math.min(maximumFixes, track.length));
    for (let fix = 0; fix < count; fix++) {
      const position = (fix * (track.length - 1)) / (count - 1);
      const index = Math.floor(position);
      const fraction = position - index;
      const first = track[index];
      const next = track[Math.min(index + 1, track.length - 1)];
      const [x, y] = project(
        first[0] + (next[0] - first[0]) * fraction,
        first[1] + (next[1] - first[1]) * fraction
      );
      if (fix > 0) segments.push(points[points.length - 2], points[points.length - 1], x, y);
      points.push(x, y);
      timestamps.push(
        ((first[2] + (next[2] - first[2]) * fraction) / DAY_SECONDS) * playbackSeconds
      );
    }
    trackOffsets.push(points.length / 2);
  }
  return {
    points: Float32Array.from(points),
    timestamps: Float32Array.from(timestamps),
    segments: Float32Array.from(segments),
    trackOffsets: Uint32Array.from(trackOffsets),
    count: segments.length / 4
  };
}
