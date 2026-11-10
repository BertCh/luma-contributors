// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTrajectoryPlayhead: every track interpolated at one instant. Dots are the live positions,
// colored by track status (active, not started, finished, inside a gap).

import {
  getGPUTrajectoryPlayheadParameterValues,
  GPUTrajectoryPlayhead
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun, type RunBuffer} from '../../define-run';
import {bindAxes} from '../../interaction';
import {createFleet} from './lib/fleet';
import {parameters} from './trajectory-playhead.parameters';

export default defineOperationRun({
  parameters,
  caption: 'San Francisco Bay vessel positions at the playhead; gray tracks are not running.',
  createInputs: ({world}) =>
    createFleet(world, {
      trackCount: 16,
      minimumFixes: 12,
      maximumFixes: 22
    }),
  build(builder, fleet) {
    const parameterBuffer = builder.allocate('parameters', 'float32', 4);
    const currentPositions = builder.allocate('current', 'float32x2', fleet.trackCount);
    const speeds = builder.allocate('speeds', 'float32', fleet.trackCount);
    const trackStatus = builder.allocate('status', 'uint32', fleet.trackCount);
    builder.graph.add(
      new GPUTrajectoryPlayhead({
        positions: builder.upload('positions', 'float32x2', fleet.positions).view,
        timestamps: builder.upload('timestamps', 'float32', fleet.timestamps).view,
        trackOffsets: builder.upload('offsets', 'uint32', fleet.trackOffsets).view,
        parameters: parameterBuffer.view,
        output: {
          currentPositions: currentPositions.view,
          speeds: speeds.view,
          trackStatus: trackStatus.view
        }
      })
    );
    return {
      outputs: {
        currentPositions,
        speeds,
        trackStatus,
        segments: builder.upload('segments', 'float32x4', fleet.segments)
      },
      update: current =>
        parameterBuffer.write(
          getGPUTrajectoryPlayheadParameterValues({
            playhead: current.playhead,
            maximumGap: current.maximumGap
          })
        )
    };
  },
  // x is the playhead: the time strip along the bottom carries its tick.
  interaction: bindAxes<unknown, {trackStatus: RunBuffer<'uint32'>}>(parameters, {
    kind: 'time',
    hint: 'scrub the clock',
    x: 'playhead',
    decorate: (values, _pointer, {outputs, world, peek}) => {
      const status = peek(outputs.trackStatus);
      let activeCount = 0;
      let finishedCount = 0;
      for (const code of status ?? []) {
        if (code === 1) activeCount++;
        if (code === 3) finishedCount++;
      }
      const minutes = Math.round((values.playhead / 600) * 24 * 60);
      const clock = `${String(Math.floor(minutes / 60) % 24).padStart(2, '0')}:${String(minutes % 60).padStart(2, '0')}`;
      const stripX = (seconds: number) => world.width * (0.06 + 0.88 * (seconds / 600));
      return {
        glyphs: [
          {shape: 'line', from: [stripX(0), 0.03], to: [stripX(600), 0.03]},
          {
            shape: 'line',
            from: [stripX(values.playhead), 0.01],
            to: [stripX(values.playhead), 0.05]
          }
        ],
        readout: `playhead ${clock} UTC · ${activeCount} active, ${finishedCount} finished`
      };
    }
  }),
  draw: ({inputs, outputs}) => [
    {layer: 'lines', context: true, segments: outputs.segments, count: inputs.segmentCount},
    {layer: 'dots', positions: outputs.currentPositions, values: outputs.trackStatus, radius: 11}
  ]
});
