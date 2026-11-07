// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUGlobalPermutationTest: the statistic of the observed values against its distribution when the
// values are shuffled over the points. The bars are the GPU-written histogram of the shuffled
// statistics; the small chart is the summary (observed value, pseudo p-value, exceedances, ...).

import {
  getGPUPermutationParameterValues,
  GPUGlobalPermutationTest,
  GPU_GLOBAL_PERMUTATION_SUMMARY,
  GPU_PERMUTATION_PARAMETER_LENGTH
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {uploadSpatialWeights} from '../../inputs';
import {createNeighborGraph, createPointField, mixNoise} from './shared/point-field';
import {parameters} from './global-permutation-test.parameters';

const STATISTICS = ['moran', 'geary', 'getisOrdG'] as const;
const MAXIMUM_PERMUTATIONS = 999;
const HISTOGRAM_BINS = 24;
// Axis range per statistic: wide enough for the shuffled values and the observed value.
const RANGES = [
  [-0.1, 0.3],
  [0.6, 1.15],
  [0.0118, 0.0125]
] as const;

export default defineOperationRun({
  parameters,
  caption: 'Observed statistic against 999 shuffles of the values.',
  createInputs: ({random, world}) => createPointField(random, world, 500),
  build(builder, points, values) {
    const graph = createNeighborGraph(points.positions, 6);
    const valueBuffer = builder.allocate('values', 'float32', points.count);
    const parameterBuffer = builder.allocate(
      'permutation',
      'uint32',
      GPU_PERMUTATION_PARAMETER_LENGTH
    );
    const summary = builder.allocate('summary', 'float32', GPU_GLOBAL_PERMUTATION_SUMMARY.length);
    const histogram = builder.allocate('histogram', 'uint32', HISTOGRAM_BINS);
    const reference = builder.allocate('reference', 'float32', MAXIMUM_PERMUTATIONS);
    builder.graph.add(
      new GPUGlobalPermutationTest({
        weights: uploadSpatialWeights(builder, graph.weights),
        values: valueBuffer.view,
        statistic: STATISTICS[values.statistic],
        parameters: parameterBuffer.view,
        maximumPermutations: MAXIMUM_PERMUTATIONS,
        output: {
          summary: summary.view,
          histogram: histogram.view,
          referenceDistribution: reference.view
        }
      })
    );
    return {
      outputs: {
        valueBuffer,
        summary,
        reference,
        indices: builder.upload(
          'indices',
          'float32',
          Float32Array.from({length: MAXIMUM_PERMUTATIONS}, (_, index) => index)
        ),
        positions: builder.upload('positions', 'float32x2', points.positions)
      },
      update: current => {
        valueBuffer.write(mixNoise(points.field, points.noise, current.noise));
        parameterBuffer.write(
          getGPUPermutationParameterValues(
            {seed: current.seed, permutationCount: current.permutationCount},
            {maximumPermutations: MAXIMUM_PERMUTATIONS}
          )
        );
      }
    };
  },
  rebuildOn: ['statistic'],
  draw: ({outputs, values}) => [
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.valueBuffer,
      radius: 4,
      range: [0.1, 0.9]
    },
    {
      layer: 'chart',
      kind: 'scatter',
      values: outputs.reference,
      xValues: outputs.indices,
      xRange: [0, MAXIMUM_PERMUTATIONS],
      color: [0.4, 0.68, 1, 0.8],
      radius: 2,
      range: RANGES[values.statistic],
      rectangle: [0.04, 0.04, 0.62, 0.32]
    },
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.summary,
      ramp: 'diverging',
      range: RANGES[values.statistic],
      rectangle: [0.7, 0.04, 0.96, 0.32]
    }
  ]
});
