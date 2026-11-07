// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTrajectoryMetrics: per-step speed along every track (segment color) and the stops it
// detected (rings sized by dwell time).

import {
  getGPUTrajectoryMetricsParameterValues,
  GPUTrajectoryMetrics
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createFleet, createTrackEdges} from './lib/fleet';
import {parameters} from './trajectory-metrics.parameters';

const STOP_CAPACITY = 64;

export default defineOperationRun({
  parameters,
  caption: 'Step speed along each track, and the stops the metrics found.',
  createInputs: ({random, world}) =>
    createFleet(random, world, {trackCount: 14, stops: true, minimumFixes: 24, maximumFixes: 40}),
  build(builder, fleet) {
    const parameterBuffer = builder.allocate('parameters', 'float32', 4);
    const stepSpeeds = builder.allocate('step-speeds', 'float32', fleet.rowCount);
    const stopIds = builder.allocate('stop-ids', 'uint32', STOP_CAPACITY);
    const stopCount = builder.allocate('stop-count', 'uint32', 1);
    const stopOverflow = builder.allocate('stop-overflow', 'uint32', 1);
    const centroids = builder.allocate('centroids', 'float32x2', STOP_CAPACITY);
    const durations = builder.allocate('durations', 'float32', STOP_CAPACITY);
    builder.graph.add(
      new GPUTrajectoryMetrics({
        positions: builder.upload('positions', 'float32x2', fleet.positions).view,
        timestamps: builder.upload('timestamps', 'float32', fleet.timestamps).view,
        trackOffsets: builder.upload('offsets', 'uint32', fleet.trackOffsets).view,
        parameters: parameterBuffer.view,
        output: {
          stepSpeeds: stepSpeeds.view,
          stops: {
            ids: stopIds.view,
            status: {count: stopCount.view, overflow: stopOverflow.view},
            centroids: centroids.view,
            durations: durations.view
          }
        }
      })
    );
    return {
      outputs: {
        stepSpeeds,
        centroids,
        durations,
        nodes: builder.upload('nodes', 'float32x2', fleet.positions),
        edges: builder.upload('edges', 'uint32x2', createTrackEdges(fleet.trackOffsets))
      },
      update: current =>
        parameterBuffer.write(
          getGPUTrajectoryMetricsParameterValues({
            stopSpeedThreshold: current.stopSpeedThreshold,
            stopMinimumDuration: current.stopMinimumDuration
          })
        )
    };
  },
  draw: ({outputs}) => [
    {
      layer: 'segments',
      nodes: outputs.nodes,
      edges: outputs.edges,
      values: outputs.stepSpeeds,
      radius: 2.5,
      range: [0, 0.003]
    },
    // Unused stop slots are zero, so a faded ramp hides them.
    {
      layer: 'dots',
      positions: outputs.centroids,
      values: outputs.durations,
      radius: 11,
      fade: true,
      range: [0, 150],
      alpha: 0.85
    }
  ]
});
