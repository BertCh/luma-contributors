// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUMantelTest: correlation between spatial closeness and time difference over event pairs. The
// dots on the left are the statistic of each shuffle (the reference distribution); the bars are
// the GPU summary, whose first bar is the observed statistic. Events are colored by time.

import {
  getGPUSpaceTimeParameterValues,
  GPUMantelTest,
  GPU_SPACE_TIME_PARAMETER_LENGTH,
  GPU_SPACE_TIME_SUMMARY_LENGTH
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {uploadSpatialWeights} from '../../inputs';
import {createBandGraph, createSpaceTimeEvents, getEventTimes} from './shared/point-field';
import {parameters} from './mantel-test.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Mantel test: spatial closeness against time difference, with the shuffled reference.',
  createInputs: ({random, world}) => createSpaceTimeEvents(random, world),
  build(builder, events, values) {
    const graph = createBandGraph(events.positions, values.radius);
    const times = builder.allocate('times', 'float32', events.count);
    const parameterBuffer = builder.allocate(
      'parameters',
      'uint32',
      GPU_SPACE_TIME_PARAMETER_LENGTH
    );
    const statistics = builder.allocate('statistics', 'float32', values.permutationCount + 1);
    const summary = builder.allocate('summary', 'float32', GPU_SPACE_TIME_SUMMARY_LENGTH);
    builder.graph.add(
      new GPUMantelTest({
        pairs: uploadSpatialWeights(builder, graph.weights),
        times: times.view,
        parameters: parameterBuffer.view,
        maximumPermutations: values.permutationCount,
        output: {statistics: statistics.view, summary: summary.view}
      })
    );
    return {
      outputs: {
        times,
        statistics,
        summary,
        indices: builder.upload(
          'indices',
          'float32',
          Float32Array.from({length: values.permutationCount + 1}, (_, index) => index)
        ),
        positions: builder.upload('positions', 'float32x2', events.positions)
      },
      update: current => {
        times.write(getEventTimes(events, current.clustering));
        parameterBuffer.write(
          getGPUSpaceTimeParameterValues({
            seed: current.seed,
            permutationCount: current.permutationCount
          })
        );
      }
    };
  },
  rebuildOn: ['radius', 'permutationCount'],
  draw: ({outputs, values}) => [
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.times,
      range: [0, 120],
      radius: 4
    },
    {
      layer: 'chart',
      kind: 'scatter',
      values: outputs.statistics,
      xValues: outputs.indices,
      xRange: [0, values.permutationCount],
      color: [0.4, 0.68, 1, 0.8],
      radius: 2,
      range: [-0.05, 0.1],
      rectangle: [0.04, 0.04, 0.62, 0.32]
    },
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.summary,
      ramp: 'diverging',
      range: [-0.05, 0.1],
      rectangle: [0.7, 0.04, 0.96, 0.32]
    }
  ]
});
