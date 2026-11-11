// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTrajectoryRecursions on twelve San Francisco Bay vessel tracks: for the circle of one radius
// around every fix, how often the track comes back (revisits), how long it spends inside
// (residenceTimes) and how long it takes to leave (firstPassageTimes).

import {GPUTrajectoryRecursions} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun, type RunBuffer} from '../../define-run';
import {clamp, findParameter, snapToParameter} from '../../interaction';
import {createMovementFleet, readFloat32, readUint32, uploadFleet} from './_lib/movement';
import {formatDistance, formatDuration} from './_lib/units';
import {parameters} from './trajectory-recursions.parameters';

const TRACK_COUNT = 6;

type Outputs = {
  segments: RunBuffer;
  positions: RunBuffer;
  revisits: RunBuffer;
  residenceTimes: RunBuffer;
  firstPassageTimes: RunBuffer;
  censored: RunBuffer;
};

export default defineOperationRun({
  parameters,
  caption:
    'Six Bay vessel tracks. Each fix is colored by the chosen column for the circle of radii[0] around it; the ring is drawn around the busiest fix. Distances and speeds are approximate: the frame stretches x and y differently.',
  createInputs: ({world}) => {
    const fleet = createMovementFleet(world, TRACK_COUNT);
    // The anchor is the fix with the most other fixes around it (a place the vessels keep coming back to).
    let anchor = 0;
    let anchorNeighbors = -1;
    for (let row = 0; row < fleet.rowCount; row++) {
      let neighbors = 0;
      for (let other = 0; other < fleet.rowCount; other++) {
        neighbors +=
          Math.hypot(
            fleet.positions[row * 2] - fleet.positions[other * 2],
            fleet.positions[row * 2 + 1] - fleet.positions[other * 2 + 1]
          ) < 0.06
            ? 1
            : 0;
      }
      if (neighbors > anchorNeighbors) {
        anchor = row;
        anchorNeighbors = neighbors;
      }
    }
    return {...fleet, anchor};
  },
  build(builder, fleet) {
    const buffers = uploadFleet(builder, fleet);
    const radii = builder.allocate('radii', 'float32', 1);
    const revisits = builder.allocate('revisits', 'uint32', fleet.rowCount);
    const residenceTimes = builder.allocate('residence-times', 'float32', fleet.rowCount);
    const firstPassageTimes = builder.allocate('first-passage-times', 'float32', fleet.rowCount);
    const censored = builder.allocate('censored', 'uint32', fleet.rowCount);
    builder.graph.add(
      new GPUTrajectoryRecursions({
        positions: buffers.positions.view,
        timestamps: buffers.timestamps.view,
        trackOffsets: buffers.trackOffsets.view,
        radii: radii.view,
        output: {
          revisits: revisits.view,
          residenceTimes: residenceTimes.view,
          firstPassageTimes: firstPassageTimes.view,
          firstPassageCensored: censored.view
        }
      })
    );
    const outputs: Outputs = {
      segments: builder.upload('segments', 'float32x4', fleet.segments),
      positions: buffers.positions,
      revisits,
      residenceTimes,
      firstPassageTimes,
      censored
    };
    return {outputs, update: current => radii.write(Float32Array.of(current.radius))};
  },
  // The chosen column as float32, so one ramp draws all three.
  async afterRun({outputs, values, upload}) {
    const metric = Math.round(values.metric);
    const column =
      metric === 0
        ? Float32Array.from(await readUint32(outputs.revisits))
        : await readFloat32(metric === 1 ? outputs.residenceTimes : outputs.firstPassageTimes);
    return {shown: upload('shown', 'float32', column)};
  },
  // Reach: the distance from the pointer to the busiest fix is the radius. The ring grows until it
  // catches another pass of the track, and the revisit count ticks up.
  interaction: {
    kind: 'reach',
    hint: 'reach from the anchor',
    rest: ({toUnit, inputs}) =>
      toUnit([
        inputs.positions[inputs.anchor * 2] + parameters[0].default,
        inputs.positions[inputs.anchor * 2 + 1]
      ]),
    resolve(pointer, {inputs, outputs, peek}) {
      const at: [number, number] = [
        inputs.positions[inputs.anchor * 2],
        inputs.positions[inputs.anchor * 2 + 1]
      ];
      const parameter = findParameter(parameters, 'radius');
      const radius = snapToParameter(
        parameter,
        clamp(Math.hypot(pointer.x - at[0], pointer.y - at[1]), parameter.min, parameter.max)
      );
      const revisits = peek(outputs.revisits);
      const residence = peek(outputs.residenceTimes);
      const passage = peek(outputs.firstPassageTimes);
      const censored = peek(outputs.censored);
      return {
        values: {radius},
        glyphs: [
          {shape: 'ring', at, radius},
          {shape: 'line', from: at, to: [pointer.x, pointer.y], dashed: true}
        ],
        readout:
          revisits && residence && passage && censored
            ? `radius ${formatDistance(radius, inputs.metersPerUnit)} · ${revisits[inputs.anchor]} revisits · ${formatDuration(residence[inputs.anchor])} inside · leaves in ${formatDuration(passage[inputs.anchor])}${censored[inputs.anchor] ? '+' : ''}`
            : `radius ${formatDistance(radius, inputs.metersPerUnit)}`
      };
    }
  },
  draw: ({inputs, outputs, extra, values}) => [
    {layer: 'lines', context: true, segments: outputs.segments, count: inputs.segmentCount},
    ...(extra
      ? [
          {
            layer: 'dots' as const,
            positions: outputs.positions,
            values: extra.shown,
            radius: 8,
            range: [0, [6, 120, 60][Math.round(values.metric)]] as const
          }
        ]
      : [])
  ]
});
