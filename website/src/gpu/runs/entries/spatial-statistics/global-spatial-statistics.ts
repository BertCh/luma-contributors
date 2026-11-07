// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUGlobalSpatialStatistics: Moran's I, Geary's C and Getis-Ord G with their moments and
// p-values in one summary buffer. The bars are that buffer: the first block is the weights
// summary, then eight values (statistic, expected, variances, z-scores, p-values) per statistic.

import {GPUGlobalSpatialStatistics} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {uploadSpatialWeights} from '../../inputs';
import {createBandGraph, createPointField, mixNoise} from './shared/point-field';
import {parameters} from './global-spatial-statistics.parameters';

const ALTERNATIVES = ['two-sided', 'greater', 'lesser'] as const;
const SUMMARY_LENGTH = 52;

export default defineOperationRun({
  parameters,
  caption: 'Global Moran, Geary and Getis-Ord G with expectation, z-score and p-value.',
  createInputs: ({random, world}) => createPointField(random, world, 700),
  build(builder, points, values) {
    const graph = createBandGraph(points.positions, values.radius);
    const valueBuffer = builder.allocate('values', 'float32', points.count);
    const summary = builder.allocate('summary', 'float32', SUMMARY_LENGTH);
    builder.graph.add(
      new GPUGlobalSpatialStatistics({
        weights: uploadSpatialWeights(builder, graph.weights),
        values: valueBuffer.view,
        statistics: ['moran', 'geary', 'getisOrdG'],
        alternative: ALTERNATIVES[values.alternative],
        output: {summary: summary.view}
      })
    );
    return {
      outputs: {
        valueBuffer,
        summary,
        positions: builder.upload('positions', 'float32x2', points.positions)
      },
      update: current => valueBuffer.write(mixNoise(points.field, points.noise, current.noise))
    };
  },
  rebuildOn: ['alternative', 'radius'],
  draw: ({outputs}) => [
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.valueBuffer,
      radius: 4,
      range: [0.1, 0.9]
    },
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.summary,
      ramp: 'diverging',
      range: [-3, 3],
      rectangle: [0.3, 0.04, 0.96, 0.32]
    }
  ]
});
