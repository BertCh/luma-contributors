// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUKnoxTest: events that are close in space and also close in time, against the same count when
// the times are shuffled. Dots are events colored by time. The bars are the GPU summary: observed
// close pairs, expected, shuffled mean, pseudo p-values and the z-score on the right.

import {
  getGPUSpaceTimeParameterValues,
  GPUKnoxTest,
  GPU_SPACE_TIME_PARAMETER_LENGTH,
  GPU_SPACE_TIME_SUMMARY_LENGTH
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {uploadSpatialWeights} from '../../inputs';
import {createBandGraph, createSpaceTimeEvents, getEventTimes} from './shared/point-field';
import {parameters} from './knox-test.parameters';

const MAXIMUM_PERMUTATIONS = 499;

export default defineOperationRun({
  parameters,
  caption: 'Knox test: close in space and in time more often than shuffled times allow.',
  createInputs: ({random, world}) => createSpaceTimeEvents(random, world),
  build(builder, events, values) {
    const graph = createBandGraph(events.positions, values.radius);
    const times = builder.allocate('times', 'float32', events.count);
    const parameterBuffer = builder.allocate(
      'parameters',
      'uint32',
      GPU_SPACE_TIME_PARAMETER_LENGTH
    );
    const statistics = builder.allocate('statistics', 'uint32', MAXIMUM_PERMUTATIONS + 1);
    const summary = builder.allocate('summary', 'float32', GPU_SPACE_TIME_SUMMARY_LENGTH);
    builder.graph.add(
      new GPUKnoxTest({
        pairs: uploadSpatialWeights(builder, graph.weights),
        times: times.view,
        parameters: parameterBuffer.view,
        maximumPermutations: MAXIMUM_PERMUTATIONS,
        output: {statistics: statistics.view, summary: summary.view}
      })
    );
    return {
      outputs: {
        times,
        summary,
        positions: builder.upload('positions', 'float32x2', events.positions)
      },
      update: current => {
        times.write(getEventTimes(events, current.clustering));
        parameterBuffer.write(
          getGPUSpaceTimeParameterValues({
            seed: 1,
            permutationCount: current.permutationCount,
            timeThreshold: current.timeThreshold
          })
        );
      }
    };
  },
  rebuildOn: ['radius'],
  draw: ({outputs}) => [
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.times,
      range: [0, 120],
      radius: 4
    },
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.summary,
      color: [0.4, 0.68, 1, 1],
      range: [0, 3000],
      rectangle: [0.3, 0.04, 0.96, 0.32]
    }
  ]
});
