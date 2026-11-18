// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Shared inputs and pipeline pieces of the trajectory-mobility runs: San Francisco Bay AIS vessel
// legs with their original fixes (no resampling), uploaded once, and the staypoint chain that the
// segment, location and mobility runs read.

import {
  getGPUTrajectoryStaypointsParameterValues,
  GPUTrajectorySegments,
  GPUTrajectoryStaypoints
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {RunBuffer, RunBuilder} from '../../../define-run';
import tracks from '../../../../data/sf-bay-ais.json';
import {createFleet, createTrackEdges, type FleetInput} from '../../recipes-trajectories/lib/fleet';

/** Staypoint table capacity shared by the runs that read it. */
export const STAYPOINT_CAPACITY = 512;

/** Playback seconds between two AIS reports is about 25 s per hour of vessel time. */
export const PLAYBACK_SECONDS_PER_HOUR = 25;

/** A vessel fleet with its scale: meters per world unit, to say distances in km. */
export type MovementFleet = FleetInput & {metersPerUnit: number};

/** The longest legs, split at long reporting gaps as createFleet does (4 km, 14 fixes). */
function getSelectedLegs(trackCount: number): (readonly number[])[][] {
  const pieces: (readonly number[])[][] = [];
  for (const track of tracks as (readonly number[])[][]) {
    let piece = [track[0]];
    for (let index = 1; index < track.length; index++) {
      const previous = track[index - 1];
      if (Math.hypot(track[index][0] - previous[0], track[index][1] - previous[1]) > 4000) {
        pieces.push(piece);
        piece = [];
      }
      piece.push(track[index]);
    }
    pieces.push(piece);
  }
  return pieces
    .filter(piece => piece.length >= 14)
    .sort((a, b) => b.length - a.length)
    .slice(0, trackCount);
}

/** A vessel fleet with every original fix kept. */
export function createMovementFleet(
  world: {width: number; height: number},
  trackCount: number
): MovementFleet {
  const fleet = createFleet(world, {trackCount, minimumFixes: 0, maximumFixes: 999});
  const fixes = getSelectedLegs(trackCount).flat();
  const xs = fixes.map(fix => fix[0]);
  const ys = fixes.map(fix => fix[1]);
  const extentX = Math.max(...xs) - Math.min(...xs);
  const extentY = Math.max(...ys) - Math.min(...ys);
  // The frame fits each axis to 90% of its side, so the scales differ a little; use their mean.
  const metersPerUnit = Math.sqrt((extentX / (0.9 * world.width)) * (extentY / 0.9));
  return {...fleet, metersPerUnit};
}

/** Uploaded columns of a fleet: positions, timestamps, offsets and the drawn track edges. */
export type FleetBuffers = {
  positions: RunBuffer<'float32x2'>;
  timestamps: RunBuffer<'float32'>;
  trackOffsets: RunBuffer<'uint32'>;
  edges: RunBuffer<'uint32x2'>;
};

export function uploadFleet(builder: RunBuilder, fleet: FleetInput): FleetBuffers {
  return {
    positions: builder.upload('positions', 'float32x2', fleet.positions),
    timestamps: builder.upload('timestamps', 'float32', fleet.timestamps),
    trackOffsets: builder.upload('offsets', 'uint32', fleet.trackOffsets),
    edges: builder.upload('edges', 'uint32x2', createTrackEdges(fleet.trackOffsets))
  };
}

/** Everything GPUTrajectoryStaypoints wrote, as run buffers. */
export type StaypointBuffers = {
  parameters: RunBuffer<'float32'>;
  tracks: RunBuffer<'uint32'>;
  count: RunBuffer<'uint32'>;
  overflow: RunBuffer<'uint32'>;
  candidateOverflow: RunBuffer<'uint32'>;
  startRows: RunBuffer<'uint32'>;
  endRows: RunBuffer<'uint32'>;
  finishRows: RunBuffer<'uint32'>;
  centroids: RunBuffer<'float32x2'>;
  durations: RunBuffer<'float32'>;
  rowStaypointIds: RunBuffer<'uint32'>;
};

/** Adds GPUTrajectoryStaypoints with every optional column, so later operations can read them. */
export function addStaypoints(
  builder: RunBuilder,
  fleet: FleetInput,
  buffers: FleetBuffers,
  maximumSearchRows?: number
): StaypointBuffers {
  const staypoints: StaypointBuffers = {
    parameters: builder.allocate('staypoint-parameters', 'float32', 4),
    tracks: builder.allocate('staypoint-tracks', 'uint32', STAYPOINT_CAPACITY),
    count: builder.allocate('staypoint-count', 'uint32', 1),
    overflow: builder.allocate('staypoint-overflow', 'uint32', 1),
    candidateOverflow: builder.allocate('staypoint-candidate-overflow', 'uint32', 1),
    startRows: builder.allocate('staypoint-start-rows', 'uint32', STAYPOINT_CAPACITY),
    endRows: builder.allocate('staypoint-end-rows', 'uint32', STAYPOINT_CAPACITY),
    finishRows: builder.allocate('staypoint-finish-rows', 'uint32', STAYPOINT_CAPACITY),
    centroids: builder.allocate('staypoint-centroids', 'float32x2', STAYPOINT_CAPACITY),
    durations: builder.allocate('staypoint-durations', 'float32', STAYPOINT_CAPACITY),
    rowStaypointIds: builder.allocate('row-staypoint-ids', 'uint32', fleet.rowCount)
  };
  builder.graph.add(
    new GPUTrajectoryStaypoints({
      positions: buffers.positions.view,
      timestamps: buffers.timestamps.view,
      trackOffsets: buffers.trackOffsets.view,
      parameters: staypoints.parameters.view,
      maximumSearchRows,
      output: {
        ids: staypoints.tracks.view,
        status: {
          count: staypoints.count.view,
          overflow: staypoints.overflow.view,
          candidateOverflow: staypoints.candidateOverflow.view
        },
        startRows: staypoints.startRows.view,
        endRows: staypoints.endRows.view,
        finishRows: staypoints.finishRows.view,
        centroids: staypoints.centroids.view,
        durations: staypoints.durations.view,
        staypointIds: staypoints.rowStaypointIds.view
      }
    })
  );
  return staypoints;
}

/** Writes the per-frame staypoint settings of the slider values. */
export function writeStaypointSettings(
  staypoints: StaypointBuffers,
  values: Record<string, number>
): void {
  staypoints.parameters.write(
    getGPUTrajectoryStaypointsParameterValues({
      distanceThreshold: values.distanceThreshold,
      timeThreshold: values.timeThreshold,
      maximumGap: values.maximumGap ?? 0
    })
  );
}

/** Reads a run buffer back as uint32 rows. */
export async function readUint32(source: RunBuffer): Promise<Uint32Array> {
  const bytes = await source.buffer.readAsync();
  return new Uint32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
}

/** Reads a run buffer back as float32 rows. */
export async function readFloat32(source: RunBuffer): Promise<Float32Array> {
  const bytes = await source.buffer.readAsync();
  return new Float32Array(
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)
  );
}

/** Index of the track that owns `row`. */
export function getTrackOfRow(trackOffsets: Uint32Array, row: number): number {
  let low = 0;
  let high = trackOffsets.length - 2;
  while (low < high) {
    const middle = (low + high + 1) >> 1;
    if (trackOffsets[middle] <= row) low = middle;
    else high = middle - 1;
  }
  return low;
}

/** One staypoint of the CPU sliding-window reference (trackintel semantics, planar). */
export type ReferenceStaypoint = {
  track: number;
  startRow: number;
  endRow: number;
  centroid: [number, number];
  duration: number;
};

/**
 * CPU sliding-window staypoints of a fleet. Runs use it only to place the pointer's rest position
 * on a real staypoint; the picture is what GPUTrajectoryStaypoints wrote.
 */
export function findReferenceStaypoints(
  fleet: FleetInput,
  distanceThreshold: number,
  timeThreshold: number
): ReferenceStaypoint[] {
  const staypoints: ReferenceStaypoint[] = [];
  const {positions, timestamps, trackOffsets} = fleet;
  for (let track = 0; track < trackOffsets.length - 1; track++) {
    const end = trackOffsets[track + 1];
    let anchor = trackOffsets[track];
    for (let row = anchor + 1; row < end; row++) {
      const distance = Math.hypot(
        positions[row * 2] - positions[anchor * 2],
        positions[row * 2 + 1] - positions[anchor * 2 + 1]
      );
      if (distance < distanceThreshold) continue;
      const duration = timestamps[row] - timestamps[anchor];
      if (duration >= timeThreshold) {
        let x = 0;
        let y = 0;
        for (let member = anchor; member < row; member++) {
          x += positions[member * 2];
          y += positions[member * 2 + 1];
        }
        staypoints.push({
          track,
          startRow: anchor,
          endRow: row,
          centroid: [x / (row - anchor), y / (row - anchor)],
          duration
        });
      }
      anchor = row;
    }
  }
  return staypoints;
}

/** Consecutive fixes of each track as (x0, y0, x1, y1), for drawing positions that were edited. */
export function createSegmentsOfPositions(
  positions: Float32Array,
  trackOffsets: Uint32Array
): Float32Array {
  const segments: number[] = [];
  for (let track = 0; track < trackOffsets.length - 1; track++) {
    for (let row = trackOffsets[track]; row < trackOffsets[track + 1] - 1; row++) {
      segments.push(
        positions[row * 2],
        positions[row * 2 + 1],
        positions[row * 2 + 2],
        positions[row * 2 + 3]
      );
    }
  }
  return Float32Array.from(segments);
}

/** Speed (world units per playback second) of the step that ends at each row; 0 at a track start. */
export function getStepSpeeds(fleet: FleetInput, positions = fleet.positions): Float32Array {
  const speeds = new Float32Array(fleet.rowCount);
  for (let track = 0; track < fleet.trackCount; track++) {
    for (let row = fleet.trackOffsets[track] + 1; row < fleet.trackOffsets[track + 1]; row++) {
      const duration = fleet.timestamps[row] - fleet.timestamps[row - 1];
      speeds[row] =
        Math.hypot(
          positions[row * 2] - positions[row * 2 - 2],
          positions[row * 2 + 1] - positions[row * 2 - 1]
        ) / Math.max(duration, 1e-6);
    }
  }
  return speeds;
}

export const TRIPLEG_CAPACITY = 512;
export const TRIP_CAPACITY = 512;

/** Everything GPUTrajectorySegments wrote, as run buffers. */
export type SegmentBuffers = {
  parameters: RunBuffer<'float32'>;
  triplegTracks: RunBuffer<'uint32'>;
  triplegCount: RunBuffer<'uint32'>;
  triplegOverflow: RunBuffer<'uint32'>;
  triplegStartRows: RunBuffer<'uint32'>;
  triplegEndRows: RunBuffer<'uint32'>;
  rowTriplegIds: RunBuffer<'uint32'>;
  tripCount: RunBuffer<'uint32'>;
  tripCandidateOverflow: RunBuffer<'uint32'>;
  triplegCandidateOverflow: RunBuffer<'uint32'>;
  triplegTripIds: RunBuffer<'uint32'>;
};

/** Adds GPUTrajectorySegments after the staypoints, with triplegs and trips. */
export function addSegments(
  builder: RunBuilder,
  fleet: FleetInput,
  buffers: FleetBuffers,
  staypoints: StaypointBuffers
): SegmentBuffers {
  const segments: SegmentBuffers = {
    parameters: builder.allocate('segment-parameters', 'float32', 4),
    triplegTracks: builder.allocate('tripleg-tracks', 'uint32', TRIPLEG_CAPACITY),
    triplegCount: builder.allocate('tripleg-count', 'uint32', 1),
    triplegOverflow: builder.allocate('tripleg-overflow', 'uint32', 1),
    triplegStartRows: builder.allocate('tripleg-start-rows', 'uint32', TRIPLEG_CAPACITY),
    triplegEndRows: builder.allocate('tripleg-end-rows', 'uint32', TRIPLEG_CAPACITY),
    rowTriplegIds: builder.allocate('row-tripleg-ids', 'uint32', fleet.rowCount),
    tripCount: builder.allocate('trip-count', 'uint32', 1),
    tripCandidateOverflow: builder.allocate('trip-candidate-overflow', 'uint32', 1),
    triplegCandidateOverflow: builder.allocate('tripleg-candidate-overflow', 'uint32', 1),
    triplegTripIds: builder.allocate('tripleg-trip-ids', 'uint32', TRIPLEG_CAPACITY)
  };
  const tripTracks = builder.allocate('trip-tracks', 'uint32', TRIP_CAPACITY);
  const triplegOverflow = segments.triplegOverflow;
  const tripOverflow = builder.allocate('trip-overflow', 'uint32', 1);
  builder.graph.add(
    new GPUTrajectorySegments({
      positions: buffers.positions.view,
      timestamps: buffers.timestamps.view,
      trackOffsets: buffers.trackOffsets.view,
      staypointIds: staypoints.rowStaypointIds.view,
      staypoints: {
        tracks: staypoints.tracks.view,
        startRows: staypoints.startRows.view,
        finishRows: staypoints.finishRows.view,
        durations: staypoints.durations.view,
        status: {
          count: staypoints.count.view,
          overflow: staypoints.overflow.view,
          candidateOverflow: staypoints.candidateOverflow.view
        }
      },
      parameters: segments.parameters.view,
      output: {
        triplegs: {
          ids: segments.triplegTracks.view,
          status: {
            count: segments.triplegCount.view,
            overflow: triplegOverflow.view,
            candidateOverflow: segments.triplegCandidateOverflow.view
          },
          startRows: segments.triplegStartRows.view,
          endRows: segments.triplegEndRows.view,
          rowTriplegIds: segments.rowTriplegIds.view
        },
        trips: {
          ids: tripTracks.view,
          status: {
            count: segments.tripCount.view,
            overflow: tripOverflow.view,
            candidateOverflow: segments.tripCandidateOverflow.view
          },
          triplegTripIds: segments.triplegTripIds.view
        }
      }
    })
  );
  return segments;
}
