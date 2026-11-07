// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Seeded vehicle tracks for the trajectory runs: irregular GPS fixes packed track after track.

export type FleetInput = {
  trackCount: number;
  rowCount: number;
  /** (x, y) per fix, sorted by track then time. */
  positions: Float32Array;
  /** Seconds since the start, one per fix. */
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
  /** Seconds of the longest track. */
  duration?: number;
  /** Tracks start and end at different times. */
  staggered?: boolean;
  /** Stop for a while now and then (speed 0). */
  stops?: boolean;
  /** Position noise as a fraction of the world height. */
  jitter?: number;
};

/** Wandering tracks with irregular fix spacing; everything lives in the world rectangle. */
export function createFleet(
  random: () => number,
  world: {width: number; height: number},
  options: FleetOptions = {}
): FleetInput {
  const trackCount = options.trackCount ?? 12;
  const minimumFixes = options.minimumFixes ?? 18;
  const maximumFixes = options.maximumFixes ?? 36;
  const duration = options.duration ?? 600;
  const positions: number[] = [];
  const timestamps: number[] = [];
  const segments: number[] = [];
  const trackOffsets = new Uint32Array(trackCount + 1);
  for (let track = 0; track < trackCount; track++) {
    const fixCount = minimumFixes + Math.floor(random() * (maximumFixes - minimumFixes + 1));
    // Irregular gaps, normalized to span the duration.
    const gaps = Array.from({length: fixCount - 1}, () => 0.3 + random() ** 2.5 * 3);
    const gapTotal = gaps.reduce((sum, gap) => sum + gap, 0);
    let x = (0.1 + 0.8 * random()) * world.width;
    let y = (0.1 + 0.8 * random()) * world.height;
    let heading = random() * Math.PI * 2;
    const speed = (0.35 + 0.5 * random()) / fixCount;
    const startTime = options.staggered ? random() * duration * 0.35 : 0;
    const trackDuration = options.staggered ? duration * (0.4 + 0.25 * random()) : duration;
    let time = startTime;
    let stoppedUntil = -1;
    for (let fix = 0; fix < fixCount; fix++) {
      const jitter = options.jitter ?? 0;
      positions.push(x + (random() - 0.5) * jitter, y + (random() - 0.5) * jitter);
      timestamps.push(time);
      const previousX = x;
      const previousY = y;
      if (fix < fixCount - 1) {
        const gap = (gaps[fix] / gapTotal) * trackDuration;
        time += gap;
        if (options.stops && stoppedUntil < fix && random() < 0.1)
          stoppedUntil = fix + 3 + Math.floor(random() * 4);
        heading += (random() - 0.5) * 0.9;
        const step =
          fix <= stoppedUntil
            ? 0.12 * speed * (gap / (trackDuration / fixCount)) * 1.6
            : speed * (gap / (trackDuration / fixCount)) * 1.6;
        x += Math.cos(heading) * step * world.height;
        y += Math.sin(heading) * step * world.height;
        if (x < 0.03 || x > world.width - 0.03) {
          heading = Math.PI - heading;
          x = Math.min(Math.max(x, 0.03), world.width - 0.03);
        }
        if (y < 0.03 || y > world.height - 0.03) {
          heading = -heading;
          y = Math.min(Math.max(y, 0.03), world.height - 0.03);
        }
        segments.push(previousX, previousY, x, y);
      }
    }
    trackOffsets[track + 1] = positions.length / 2;
  }
  return {
    trackCount,
    rowCount: positions.length / 2,
    positions: Float32Array.from(positions),
    timestamps: Float32Array.from(timestamps),
    trackOffsets,
    segments: Float32Array.from(segments),
    segmentCount: segments.length / 4,
    duration,
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
