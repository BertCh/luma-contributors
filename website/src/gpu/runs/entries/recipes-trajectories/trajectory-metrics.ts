// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTrajectoryMetrics: per-step speed along every track (segment color) and the stops it
// detected (rings sized by dwell time).

import {
  getGPUTrajectoryMetricsParameterValues,
  GPUTrajectoryMetrics
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {findParameter, snapToParameter} from '../../interaction';
import {findNearestSegment} from '../../pick';
import {createFleet, createTrackEdges} from './lib/fleet';
import {parameters} from './trajectory-metrics.parameters';

const STOP_CAPACITY = 64;

export default defineOperationRun({
  parameters,
  caption: 'Step speed along each track, and the stops the metrics found.',
  createInputs: ({world}) => {
    const fleet = createFleet(world, {trackCount: 14, minimumFixes: 24, maximumFixes: 40});
    // Speed of every segment (world units per playback second), in the order of fleet.segments.
    const speeds: number[] = [];
    for (let track = 0; track < fleet.trackCount; track++) {
      for (let row = fleet.trackOffsets[track]; row < fleet.trackOffsets[track + 1] - 1; row++) {
        const distance = Math.hypot(
          fleet.positions[row * 2 + 2] - fleet.positions[row * 2],
          fleet.positions[row * 2 + 3] - fleet.positions[row * 2 + 1]
        );
        speeds.push(distance / Math.max(1e-6, fleet.timestamps[row + 1] - fleet.timestamps[row]));
      }
    }
    // Rest on the segment whose speed is nearest the default threshold.
    const defaultThreshold = parameters[0].default;
    let restSegment = 0;
    for (let segment = 1; segment < speeds.length; segment++) {
      if (
        Math.abs(speeds[segment] - defaultThreshold) <
        Math.abs(speeds[restSegment] - defaultThreshold)
      ) {
        restSegment = segment;
      }
    }
    const restPoint: [number, number] = [
      (fleet.segments[restSegment * 4] + fleet.segments[restSegment * 4 + 2]) / 2,
      (fleet.segments[restSegment * 4 + 1] + fleet.segments[restSegment * 4 + 3]) / 2
    ];
    return {...fleet, speeds, restPoint};
  },
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
        stopCount,
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
  // The speed of the segment under the pointer becomes the threshold: slower segments turn into stops.
  interaction: {
    kind: 'level',
    hint: 'pick a speed',
    rest: ({toUnit, inputs}) => toUnit(inputs.restPoint),
    resolve(pointer, {inputs, outputs, peek}) {
      const hit = findNearestSegment(inputs.segments, pointer.x, pointer.y, inputs.segmentCount);
      if (!hit) return {};
      const stopSpeedThreshold = snapToParameter(
        findParameter(parameters, 'stopSpeedThreshold'),
        inputs.speeds[hit.index]
      );
      const stopCount = peek(outputs.stopCount);
      return {
        values: {stopSpeedThreshold},
        glyphs: [
          {
            shape: 'line',
            from: [inputs.segments[hit.index * 4], inputs.segments[hit.index * 4 + 1]],
            to: [inputs.segments[hit.index * 4 + 2], inputs.segments[hit.index * 4 + 3]]
          },
          {shape: 'crosshair', at: hit.point}
        ],
        readout: `stopSpeedThreshold ${stopSpeedThreshold.toFixed(4)} · ${stopCount ? stopCount[0] : 0} stops`
      };
    }
  },
  draw: ({outputs}) => [
    {
      layer: 'segments',
      nodes: outputs.nodes,
      edges: outputs.edges,
      values: outputs.stepSpeeds,
      radius: 2.5,
      range: [0, 0.03]
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
