// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Tracks in families of similar shapes, for the similarity runs. Every track has the same vertex
// count; all pairs are listed so a similarity table can be drawn as a matrix.

export const FAMILY_COUNT = 3;
export const TRACKS_PER_FAMILY = 8;
export const TRACK_COUNT = FAMILY_COUNT * TRACKS_PER_FAMILY;
export const VERTEX_COUNT = 12;

export type TrackFamilies = {
  /** Unit noise (x, y) per vertex. */
  unitNoise: Float32Array;
  lineOffsets: Uint32Array;
  pairA: Uint32Array;
  pairB: Uint32Array;
  /** Builds vertex positions for a noise amount (world units). */
  getPositions(noise: number): Float32Array;
};

/** Tracks of one family follow the same curve (a different one per family) plus noise. */
export function createTrackFamilies(random: () => number): TrackFamilies {
  const unitNoise = Float32Array.from(
    {length: TRACK_COUNT * VERTEX_COUNT * 2},
    () => random() * 2 - 1
  );
  const phases = Float32Array.from({length: TRACK_COUNT}, () => (random() - 0.5) * 0.25);
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
    unitNoise,
    lineOffsets,
    pairA: Uint32Array.from(pairA),
    pairB: Uint32Array.from(pairB),
    getPositions(noise) {
      const positions = new Float32Array(TRACK_COUNT * VERTEX_COUNT * 2);
      for (let track = 0; track < TRACK_COUNT; track++) {
        const family = Math.floor(track / TRACKS_PER_FAMILY);
        for (let vertex = 0; vertex < VERTEX_COUNT; vertex++) {
          const progress = vertex / (VERTEX_COUNT - 1);
          const row = (track * VERTEX_COUNT + vertex) * 2;
          const x = progress + phases[track] * (family === 1 ? 0 : 1);
          const y =
            family === 0
              ? 0.5 * Math.sin(progress * Math.PI * 2)
              : family === 1
                ? progress * progress * 1.6 - 0.3
                : 0.5 - Math.abs(progress - 0.5) * 2;
          positions[row] = x + unitNoise[row] * noise;
          positions[row + 1] = y + unitNoise[row + 1] * noise;
        }
      }
      return positions;
    }
  };
}
