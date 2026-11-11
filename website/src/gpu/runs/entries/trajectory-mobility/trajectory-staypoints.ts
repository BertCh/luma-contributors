// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTrajectoryStaypoints on San Francisco Bay vessel tracks: the fixes that stay within
// distanceThreshold of an anchor for at least timeThreshold are one staypoint. Stay fixes are
// colored by staypoint, rings are the centroids sized by duration.

import {defineOperationRun, type RunBuffer} from '../../define-run';
import {bindAxes} from '../../interaction';
import {
  createMovementFleet,
  addStaypoints,
  uploadFleet,
  writeStaypointSettings
} from './_lib/movement';
import {formatDistance, formatDuration} from './_lib/units';
import {parameters} from './trajectory-staypoints.parameters';

type Outputs = {
  segments: RunBuffer;
  positions: RunBuffer;
  rowStaypointIds: RunBuffer;
  centroids: RunBuffer;
  durations: RunBuffer;
  count: RunBuffer;
};

export default defineOperationRun({
  parameters,
  caption:
    'Vessel fixes from the San Francisco Bay AIS extract. Colored fixes belong to a staypoint; rings mark its centroid and fade in with its duration. Distances and speeds are approximate: the frame stretches x and y differently.',
  createInputs: ({world}) => createMovementFleet(world, 16),
  build(builder, fleet) {
    const buffers = uploadFleet(builder, fleet);
    const staypoints = addStaypoints(builder, fleet, buffers);
    const outputs: Outputs = {
      segments: builder.upload('segments', 'float32x4', fleet.segments),
      positions: buffers.positions,
      rowStaypointIds: staypoints.rowStaypointIds,
      centroids: staypoints.centroids,
      durations: staypoints.durations,
      count: staypoints.count
    };
    return {outputs, update: current => writeStaypointSettings(staypoints, current)};
  },
  // A pad: x is distanceThreshold, y is timeThreshold. A ring of the distance is drawn around the
  // longest staypoint, so the reader sees the radius that holds it together.
  interaction: bindAxes<{metersPerUnit: number}, Outputs>(parameters, {
    kind: 'pad',
    hint: 'distance across, time up',
    x: 'distanceThreshold',
    y: 'timeThreshold',
    decorate: (values, _pointer, {inputs, outputs, peek}) => {
      const count = peek(outputs.count);
      const durations = peek(outputs.durations);
      const centroids = peek(outputs.centroids);
      const staypointCount = count ? count[0] : 0;
      let longest = -1;
      if (durations) {
        for (let row = 0; row < staypointCount; row++) {
          if (longest < 0 || durations[row] > durations[longest]) longest = row;
        }
      }
      return {
        glyphs:
          centroids && longest >= 0
            ? [
                {
                  shape: 'ring',
                  at: [centroids[longest * 2], centroids[longest * 2 + 1]],
                  radius: values.distanceThreshold
                }
              ]
            : [],
        readout: `distanceThreshold ${formatDistance(values.distanceThreshold, inputs.metersPerUnit)} · timeThreshold ${formatDuration(values.timeThreshold)} · ${staypointCount} stays`
      };
    }
  }),
  draw: ({inputs, outputs}) => [
    {layer: 'lines', context: true, segments: outputs.segments, count: inputs.segmentCount},
    {layer: 'dots', positions: outputs.positions, values: outputs.rowStaypointIds, radius: 3},
    {
      layer: 'dots',
      positions: outputs.centroids,
      values: outputs.durations,
      radius: 11,
      fade: true,
      range: [0, 200],
      alpha: 0.85
    }
  ]
});
