// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUGammaStatistic: the sum over neighbor pairs of a pair function (cross-product,
// squared-difference or absolute-difference) of their values. The map shows the values; the bars
// are the statistic the GPU wrote. A smooth field gives small differences, noise gives large ones.

import {GPUGammaStatistic} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {uploadSpatialWeights} from '../../inputs';
import {createBandGraph, createPointField, mixNoise} from './shared/point-field';
import {parameters} from './gamma-statistic.parameters';

// One scale for every operation: the statistic is normalized, and absolute differences are larger.
const GAUGE_LIMIT = 0.08;
const OPERATIONS = ['cross-product', 'squared-difference', 'absolute-difference'] as const;

export default defineOperationRun({
  parameters,
  caption: 'Gamma statistic: a pair function summed over every neighbor pair.',
  createInputs: ({random, world}) => createPointField(random, world, 700),
  build(builder, points, values) {
    const graph = createBandGraph(points.positions, values.radius);
    const valueBuffer = builder.allocate('values', 'float32', points.count);
    const statistic = builder.allocate('statistic', 'float32', 1);
    builder.graph.add(
      new GPUGammaStatistic({
        weights: uploadSpatialWeights(builder, graph.weights),
        values: valueBuffer.view,
        operation: OPERATIONS[values.operation],
        output: {statistic: statistic.view}
      })
    );
    return {
      outputs: {
        valueBuffer,
        statistic,
        positions: builder.upload('positions', 'float32x2', points.positions)
      },
      update: current => valueBuffer.write(mixNoise(points.field, points.noise, current.noise))
    };
  },
  rebuildOn: ['operation', 'radius'],
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
      values: outputs.statistic,
      range: [0, GAUGE_LIMIT],
      rectangle: [0.4, 0.04, 0.6, 0.32]
    }
  ]
});
