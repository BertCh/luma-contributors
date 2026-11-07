// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSpatialPearson between a variable and a second variable that follows it plus noise. The
// map shows the first variable; the bar is the statistic the GPU wrote (one float32).

import {GPUSpatialPearson} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {uploadSpatialWeights} from '../../inputs';
import {createBandGraph, createPointField, mixNoise} from './shared/point-field';
import {parameters} from './spatial-pearson.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Spatial Pearson correlation of two variables over the neighbor graph.',
  createInputs: ({random, world}) => createPointField(random, world, 700),
  build(builder, points, values) {
    const graph = createBandGraph(points.positions, values.radius);
    const first = builder.upload('first', 'float32', points.field);
    const second = builder.allocate('second', 'float32', points.count);
    const statistic = builder.allocate('statistic', 'float32', 1);
    builder.graph.add(
      new GPUSpatialPearson({
        weights: uploadSpatialWeights(builder, graph.weights),
        values: first.view,
        secondValues: second.view,
        output: {statistic: statistic.view}
      })
    );
    return {
      outputs: {
        first,
        statistic,
        positions: builder.upload('positions', 'float32x2', points.positions)
      },
      update: current => second.write(mixNoise(points.field, points.noise, current.noise))
    };
  },
  rebuildOn: ['radius'],
  draw: ({outputs}) => [
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.first,
      radius: 4,
      range: [0.1, 0.9]
    },
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.statistic,
      ramp: 'diverging',
      range: [-1, 1],
      rectangle: [0.4, 0.04, 0.6, 0.32]
    }
  ]
});
