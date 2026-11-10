// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {sampleVesselTracks} from '../../../../data/vessel-tracks';

export const TRACK_COUNT = 24;
export const VERTEX_COUNT = 12;

export type TrackPairs = {
  lineOffsets: Uint32Array;
  pairA: Uint32Array;
  pairB: Uint32Array;
  /** Builds vertex positions for a noise amount (world units). */
  getPositions(noise: number): Float32Array;
};

/** Compare observed vessel paths, ordered by starting longitude. Noise is an optional scenario. */
export function createTrackPairs(random: () => number): TrackPairs {
  const unitNoise = Float32Array.from(
    {length: TRACK_COUNT * VERTEX_COUNT * 2},
    () => random() * 2 - 1
  );
  const {points} = sampleVesselTracks(
    {width: 1.6, height: 1},
    TRACK_COUNT,
    VERTEX_COUNT,
    VERTEX_COUNT
  );
  const order = Array.from({length: TRACK_COUNT}, (_, index) => index).sort(
    (first, second) => points[first * VERTEX_COUNT * 2] - points[second * VERTEX_COUNT * 2]
  );
  const ordered = Float32Array.from(
    order.flatMap(track =>
      Array.from(points.subarray(track * VERTEX_COUNT * 2, (track + 1) * VERTEX_COUNT * 2))
    )
  );
  const lineOffsets = Uint32Array.from(
    {length: TRACK_COUNT + 1},
    (_, index) => index * VERTEX_COUNT
  );
  const pairA: number[] = [];
  const pairB: number[] = [];
  for (let first = 0; first < TRACK_COUNT; first++) {
    for (let second = 0; second < TRACK_COUNT; second++) {
      pairA.push(first);
      pairB.push(second);
    }
  }
  return {
    lineOffsets,
    pairA: Uint32Array.from(pairA),
    pairB: Uint32Array.from(pairB),
    getPositions(noise) {
      return Float32Array.from(
        ordered,
        (coordinate, index) => coordinate + unitNoise[index] * noise
      );
    }
  };
}
