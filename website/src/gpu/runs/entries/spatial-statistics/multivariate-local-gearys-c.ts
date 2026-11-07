// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUMultivariateLocalGeary: one local Geary statistic over several variables at once, averaged
// across the variables. Variables share one spatial pattern; the noise slider breaks it
// independently in each variable, so more variables give a more reliable picture of the cluster.

import {GPUMultivariateLocalGeary} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {uploadSpatialWeights} from '../../inputs';
import {createBandGraph, createPointField} from './shared/point-field';
import {parameters} from './multivariate-local-gearys-c.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Local Geary averaged over several variables.',
  createInputs: ({random, world}) => createPointField(random, world, 700, false),
  build(builder, points, values) {
    const graph = createBandGraph(points.positions, values.radius);
    const variableCount = values.variableCount;
    const valueBuffer = builder.allocate('values', 'float32', points.count * variableCount);
    const statistic = builder.allocate('statistic', 'float32', points.count);
    const status = builder.allocate('status', 'uint32', points.count);
    builder.graph.add(
      new GPUMultivariateLocalGeary({
        weights: uploadSpatialWeights(builder, graph.weights),
        values: valueBuffer.view,
        variableCount,
        output: {statistic: statistic.view, status: status.view}
      })
    );
    return {
      outputs: {
        statistic,
        positions: builder.upload('positions', 'float32x2', points.positions)
      },
      update: current => {
        const interleaved = new Float32Array(points.count * variableCount);
        for (let row = 0; row < points.count; row++) {
          for (let variable = 0; variable < variableCount; variable++) {
            const jitter = Math.sin((row * 7 + variable * 131) * 12.9898) * 43758.5453;
            const noise = jitter - Math.floor(jitter);
            interleaved[row * variableCount + variable] =
              points.field[row] * (1 - current.noise) + noise * current.noise;
          }
        }
        valueBuffer.write(interleaved);
      }
    };
  },
  rebuildOn: ['variableCount', 'radius'],
  draw: ({outputs}) => [
    {layer: 'dots', context: true, positions: outputs.positions, radius: 2},
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.statistic,
      radius: 4.5,
      range: [0, 1.4]
    }
  ]
});
