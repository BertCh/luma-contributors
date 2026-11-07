// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTrajectoryPlayhead: every track interpolated at one instant. Dots are the live positions,
// colored by track status (active, not started, finished, inside a gap).

import {
  getGPUTrajectoryPlayheadParameterValues,
  GPUTrajectoryPlayhead
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createFleet} from './lib/fleet';
import {parameters} from './trajectory-playhead.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Live vehicle positions at the playhead; gray tracks are not running.',
  createInputs: ({random, world}) =>
    createFleet(random, world, {
      trackCount: 16,
      staggered: true,
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
  draw: ({inputs, outputs}) => [
    {layer: 'lines', context: true, segments: outputs.segments, count: inputs.segmentCount},
    {layer: 'dots', positions: outputs.currentPositions, values: outputs.trackStatus, radius: 12}
  ]
});
