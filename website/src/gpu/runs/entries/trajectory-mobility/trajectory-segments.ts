// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTrajectorySegments after GPUTrajectoryStaypoints (distanceThreshold 0.006, timeThreshold 12 s):
// the runs of fixes between staypoints are triplegs, and triplegs between two activities form a
// trip. Lines are colored by trip; rings mark the staypoints that last longer than
// activityMinimumDuration (activities).

import {getGPUTrajectorySegmentsParameterValues} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun, type RunBuffer} from '../../define-run';
import type {RunGlyph} from '../../interaction';
import {
  addSegments,
  addStaypoints,
  createMovementFleet,
  readFloat32,
  readUint32,
  uploadFleet,
  writeStaypointSettings
} from './_lib/movement';
import {formatDuration} from './_lib/units';
import {parameters} from './trajectory-segments.parameters';

const TRACK_COUNT = 8;
const STAYPOINT_SETTINGS = {distanceThreshold: 0.006, timeThreshold: 12, maximumGap: 0};
const NONE = 0xffffffff;

type Outputs = {
  rowTriplegIds: RunBuffer;
  segments: RunBuffer;
  positions: RunBuffer;
  triplegCount: RunBuffer;
  tripCount: RunBuffer;
  triplegStartRows: RunBuffer;
  triplegEndRows: RunBuffer;
  triplegTripIds: RunBuffer;
  staypointCount: RunBuffer;
  centroids: RunBuffer;
  durations: RunBuffer;
};

export default defineOperationRun({
  parameters,
  caption:
    'Eight Bay vessel tracks. Lines are colored by trip (grey: not in any trip); rings are activities, staypoints longer than activityMinimumDuration.',
  createInputs: ({world}) => {
    const fleet = createMovementFleet(world, TRACK_COUNT);
    // The first row of every drawn step, to find the tripleg under the pointer.
    const stepRows: number[] = [];
    for (let track = 0; track < fleet.trackCount; track++) {
      for (let row = fleet.trackOffsets[track]; row < fleet.trackOffsets[track + 1] - 1; row++) {
        stepRows.push(row);
      }
    }
    return {...fleet, stepRows};
  },
  build(builder, fleet) {
    const buffers = uploadFleet(builder, fleet);
    const staypoints = addStaypoints(builder, fleet, buffers);
    const segments = addSegments(builder, fleet, buffers, staypoints);
    const outputs: Outputs = {
      rowTriplegIds: segments.rowTriplegIds,
      segments: builder.upload('segments', 'float32x4', fleet.segments),
      positions: buffers.positions,
      triplegCount: segments.triplegCount,
      tripCount: segments.tripCount,
      triplegStartRows: segments.triplegStartRows,
      triplegEndRows: segments.triplegEndRows,
      triplegTripIds: segments.triplegTripIds,
      staypointCount: staypoints.count,
      centroids: staypoints.centroids,
      durations: staypoints.durations
    };
    return {
      outputs,
      update: current => {
        writeStaypointSettings(staypoints, STAYPOINT_SETTINGS);
        segments.parameters.write(
          getGPUTrajectorySegmentsParameterValues({
            maximumGap: current.maximumGap,
            activityMinimumDuration: current.activityMinimumDuration
          })
        );
      }
    };
  },
  // Triplegs are joined row to row and labeled with their trip; activities are gathered.
  async afterRun({inputs, outputs, values, upload}) {
    const o = outputs;
    const [triplegCount, starts, ends, tripIds, stayCount, centroids, durations] =
      await Promise.all([
        readUint32(o.triplegCount),
        readUint32(o.triplegStartRows),
        readUint32(o.triplegEndRows),
        readUint32(o.triplegTripIds),
        readUint32(o.staypointCount),
        readFloat32(o.centroids),
        readFloat32(o.durations)
      ]);
    const labels = new Uint32Array(inputs.rowCount).fill(NONE);
    const edges: number[] = [];
    for (let tripleg = 0; tripleg < Math.min(triplegCount[0], starts.length); tripleg++) {
      for (let row = starts[tripleg]; row < ends[tripleg]; row++) {
        labels[row] = tripIds[tripleg];
        if (row + 1 < ends[tripleg]) edges.push(row, row + 1);
      }
    }
    const activities: number[] = [];
    for (let stay = 0; stay < stayCount[0]; stay++) {
      if (durations[stay] > values.activityMinimumDuration) {
        activities.push(centroids[stay * 2], centroids[stay * 2 + 1]);
      }
    }
    return {
      labels: upload('trip-labels', 'uint32', labels),
      edges: upload('tripleg-edges', 'uint32x2', Uint32Array.from(edges.length ? edges : [0, 0])),
      activityCount: activities.length / 2,
      activities: upload(
        'activities',
        'float32x2',
        Float32Array.from(activities.length ? activities : [0, 0])
      )
    };
  },
  // Probe: the tripleg under the pointer lights up with the other triplegs of its trip, and the
  // readout says how long the trip lasted. The two sliders set what counts as an activity and a gap.
  interaction: {
    kind: 'probe',
    hint: 'probe a tripleg',
    rest: ({toUnit, inputs}) => toUnit([inputs.segments[40 * 4], inputs.segments[40 * 4 + 1]]),
    resolve(pointer, {inputs, outputs, peek}) {
      const count = peek(outputs.triplegCount);
      const starts = peek(outputs.triplegStartRows);
      const ends = peek(outputs.triplegEndRows);
      const tripIds = peek(outputs.triplegTripIds);
      const trips = peek(outputs.tripCount);
      if (!count || !starts || !ends || !tripIds || !trips) return {};
      // The nearest drawn step that lies inside a tripleg.
      let best = -1;
      let bestDistance = Infinity;
      for (let step = 0; step < inputs.segmentCount; step++) {
        const row = inputs.stepRows[step];
        const inside = Array.from({length: count[0]}, (_, leg) => leg).some(
          leg => row >= starts[leg] && row + 1 < ends[leg]
        );
        if (!inside) continue;
        const distance = Math.hypot(
          (inputs.segments[step * 4] + inputs.segments[step * 4 + 2]) / 2 - pointer.x,
          (inputs.segments[step * 4 + 1] + inputs.segments[step * 4 + 3]) / 2 - pointer.y
        );
        if (distance < bestDistance) {
          best = step;
          bestDistance = distance;
        }
      }
      if (best < 0) return {readout: `${trips[0]} trips, ${count[0]} triplegs`};
      const row = inputs.stepRows[best];
      const leg = Array.from({length: count[0]}, (_, index) => index).find(
        index => row >= starts[index] && row + 1 < ends[index]
      ) as number;
      const trip = tripIds[leg];
      const glyphs: RunGlyph[] = [];
      let legCount = 0;
      let first = Infinity;
      let last = -Infinity;
      for (let other = 0; other < count[0]; other++) {
        if (tripIds[other] !== trip) continue;
        legCount++;
        const points: number[] = [];
        for (let fix = starts[other]; fix < ends[other]; fix++) {
          points.push(inputs.positions[fix * 2], inputs.positions[fix * 2 + 1]);
        }
        glyphs.push({shape: 'polyline', points});
        first = Math.min(first, inputs.timestamps[starts[other]]);
        last = Math.max(last, inputs.timestamps[ends[other] - 1]);
      }
      return {
        glyphs,
        readout: `trip ${trip} · ${legCount} triplegs · ${formatDuration(last - first)} · ${trips[0]} trips`
      };
    }
  },
  draw: ({inputs, outputs, extra}) => [
    {layer: 'lines', context: true, segments: outputs.segments, count: inputs.segmentCount},
    ...(extra
      ? [
          {
            layer: 'segments' as const,
            nodes: outputs.positions,
            edges: extra.edges,
            values: extra.labels,
            radius: 3.5
          },
          {
            layer: 'dots' as const,
            positions: extra.activities,
            count: extra.activityCount,
            radius: 12,
            color: [1, 1, 1, 0.45] as const
          }
        ]
      : [])
  ]
});
