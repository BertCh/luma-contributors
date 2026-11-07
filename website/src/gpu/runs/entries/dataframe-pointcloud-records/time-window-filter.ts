// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTimeWindowFilter over timestamped events that spread outward from a center. Faint dots are
// every event; bright dots are the ones inside the window, faded in over startFadeDuration by the
// GPU-written fadeWeights.

import {
  GPUTimeWindowFilter,
  GPU_TIME_WINDOW_FILTER_PARAMETER_LENGTH,
  getGPUTimeWindowFilterParameterValues
} from '@luma-contributors/analysis/dataframe-analysis';
import {defineOperationRun} from '../../define-run';
import {createUniformPoints} from '../../inputs';
import {parameters} from './time-window-filter.parameters';

const EVENT_COUNT = 2500;

export default defineOperationRun({
  parameters,
  caption: 'Events inside the time window, faded in from its start.',
  createInputs: ({random, world}) => {
    const points = createUniformPoints(random, world, EVENT_COUNT, 0.1);
    const hours = Float32Array.from({length: EVENT_COUNT}, (_, index) => {
      const distance = Math.hypot(
        points.positions[index * 2] - world.width / 2,
        points.positions[index * 2 + 1] - 0.5
      );
      return Math.min(24, distance * 22 + random() * 2);
    });
    return {points, hours};
  },
  build(builder, inputs) {
    const timestamps = builder.upload('timestamps', 'float32', inputs.hours);
    const positions = builder.upload('positions', 'float32x2', inputs.points.positions);
    const window = builder.allocate('window', 'float32', GPU_TIME_WINDOW_FILTER_PARAMETER_LENGTH);
    const fadeWeights = builder.allocate('fade-weights', 'float32', EVENT_COUNT);
    builder.graph.add(
      new GPUTimeWindowFilter({
        timestamps: timestamps.view,
        window: window.view,
        output: {
          ids: builder.allocate('ids', 'uint32', EVENT_COUNT).view,
          status: {
            count: builder.allocate('count', 'uint32', 1).view,
            overflow: builder.allocate('overflow', 'uint32', 1).view
          },
          fadeWeights: fadeWeights.view
        }
      })
    );
    return {
      outputs: {positions, timestamps, fadeWeights},
      update: current =>
        window.write(
          getGPUTimeWindowFilterParameterValues({
            start: current.start,
            end: Math.max(current.end, current.start),
            startFadeDuration: current.startFadeDuration
          })
        )
    };
  },
  rebuildOn: [],
  draw: ({outputs}) => [
    {
      layer: 'dots',
      context: true,
      positions: outputs.positions,
      color: [0.4, 0.45, 0.55, 0.45],
      radius: 2.4
    },
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.fadeWeights,
      ramp: 'sequential',
      range: [0, 1],
      fade: true,
      radius: 3.6
    }
  ]
});
