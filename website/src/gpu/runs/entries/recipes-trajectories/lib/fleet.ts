// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {AIS_PLAYBACK_SECONDS} from '../../../../data/vessel-tracks';
import tracks from '../../../../data/sf-bay-ais.json';

const DAY_SECONDS = 86400;
/** A step longer than this (local meters) is a reporting gap; the track is split there. */
const GAP_METERS = 4000;
/** Pieces with fewer raw fixes than this are dropped. */
const MINIMUM_PIECE_FIXES = 14;
/** Fraction of each world axis the fleet fills. */
const FRAME_FILL = 0.9;

export type FleetInput = {
  trackCount: number;
  rowCount: number;
  /** (x, y) per fix, sorted by track then time. */
  positions: Float32Array;
  /** Playback seconds since midnight, with one UTC day scaled to ten minutes, one per fix. */
  timestamps: Float32Array;
  /** trackCount + 1 row offsets. */
  trackOffsets: Uint32Array;
  /** Consecutive fixes of a track as (x0, y0, x1, y1). */
  segments: Float32Array;
  segmentCount: number;
  /** Largest timestamp. */
  duration: number;
  /** Width of the world the tracks live in. */
  worldWidth: number;
};

export type FleetOptions = {
  trackCount?: number;
  minimumFixes?: number;
  maximumFixes?: number;
};

type Fix = readonly number[];

/** Splits every vessel track where two consecutive reports are far apart, so no chord crosses the bay. */
function splitTracksAtGaps(): Fix[][] {
  const pieces: Fix[][] = [];
  for (const track of tracks as Fix[][]) {
    let piece: Fix[] = [track[0]];
    for (let index = 1; index < track.length; index++) {
      const previous = track[index - 1];
      const fix = track[index];
      if (Math.hypot(fix[0] - previous[0], fix[1] - previous[1]) > GAP_METERS) {
        pieces.push(piece);
        piece = [];
      }
      piece.push(fix);
    }
    pieces.push(piece);
  }
  return pieces
    .filter(piece => piece.length >= MINIMUM_PIECE_FIXES)
    .sort((first, second) => second.length - first.length);
}

/**
 * Vessel reports keep a common UTC clock, so pairwise encounters are evaluated concurrently. Tracks
 * are split at long reporting gaps and each axis is fitted to the frame.
 */
export function createFleet(
  world: {width: number; height: number},
  options: FleetOptions = {}
): FleetInput {
  const selected = splitTracksAtGaps().slice(0, options.trackCount ?? 12);
  const minimumFixes = options.minimumFixes ?? 18;
  const maximumFixes = options.maximumFixes ?? 36;
  let west = Infinity;
  let south = Infinity;
  let east = -Infinity;
  let north = -Infinity;
  for (const [x, y] of selected.flat()) {
    west = Math.min(west, x);
    east = Math.max(east, x);
    south = Math.min(south, y);
    north = Math.max(north, y);
  }
  const project = (x: number, y: number): [number, number] => [
    world.width * (0.5 + ((x - (west + east) / 2) / (east - west || 1)) * FRAME_FILL),
    0.5 + ((y - (south + north) / 2) / (north - south || 1)) * FRAME_FILL
  ];
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
        ((first[2] + (next[2] - first[2]) * fraction) / DAY_SECONDS) * AIS_PLAYBACK_SECONDS
      );
    }
    trackOffsets.push(points.length / 2);
  }
  return {
    trackCount: selected.length,
    rowCount: points.length / 2,
    positions: Float32Array.from(points),
    timestamps: Float32Array.from(timestamps),
    trackOffsets: Uint32Array.from(trackOffsets),
    segments: Float32Array.from(segments),
    segmentCount: segments.length / 4,
    duration: AIS_PLAYBACK_SECONDS,
    worldWidth: world.width
  };
}

/** Edges joining consecutive rows inside each track, as node-index pairs. */
export function createTrackEdges(trackOffsets: Uint32Array, rowsPerTrack?: number): Uint32Array {
  const edges: number[] = [];
  for (let track = 0; track < trackOffsets.length - 1; track++) {
    const start = rowsPerTrack === undefined ? trackOffsets[track] : track * rowsPerTrack;
    const end = rowsPerTrack === undefined ? trackOffsets[track + 1] : start + rowsPerTrack;
    for (let row = start; row < end - 1; row++) edges.push(row, row + 1);
  }
  return Uint32Array.from(edges);
}

/** Squeezes a fleet into the upper part of the view (y from `offset` to 1), leaving room for a chart. */
export function raiseFleet(fleet: FleetInput, offset: number): FleetInput {
  const scale = 1 - offset;
  const positions = Float32Array.from(fleet.positions, (value, index) =>
    index % 2 === 1 ? offset + value * scale : value
  );
  const segments = Float32Array.from(fleet.segments, (value, index) =>
    index % 2 === 1 ? offset + value * scale : value
  );
  return {...fleet, positions, segments};
}
