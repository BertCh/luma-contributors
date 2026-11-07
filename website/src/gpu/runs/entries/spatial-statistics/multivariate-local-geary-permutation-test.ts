// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUMultivariateLocalGearyPermutationTest: classifies each point as a cluster, a negative
// association or not significant, from the multivariate local Geary statistic and shuffles of
// the neighbors. The exact strategy handles up to 256 rows, so this run uses 240 points.

import {GPUMultivariateLocalGearyPermutationTest} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {uploadSpatialWeights} from '../../inputs';
import {createNeighborGraph, createPointField} from './shared/point-field';
import {parameters} from './multivariate-local-geary-permutation-test.parameters';

const TAILS = ['equal-tail', 'esda-min-tail'] as const;

export default defineOperationRun({
  parameters,
  caption: 'Multivariate local Geary classes: high-high, low-low, negative, not significant.',
  createInputs: ({random, world}) => createPointField(random, world, 240, false),
  build(builder, points, values) {
    const graph = createNeighborGraph(points.positions, 6);
    const variableCount = values.variableCount;
    const valueBuffer = builder.allocate('values', 'float32', points.count * variableCount);
    const statistic = builder.allocate('statistic', 'float32', points.count);
    const pValues = builder.allocate('p-values', 'float32', points.count);
    const classes = builder.allocate('classes', 'uint32', points.count);
    builder.graph.add(
      new GPUMultivariateLocalGearyPermutationTest({
        weights: uploadSpatialWeights(builder, graph.weights),
        values: valueBuffer.view,
        variableCount,
        maximumPermutations: values.maximumPermutations,
        seed: [1, 2],
        tail: TAILS[values.tail],
        output: {statistic: statistic.view, pValues: pValues.view, classes: classes.view}
      })
    );
    return {
      outputs: {
        classes,
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
  rebuildOn: ['variableCount', 'tail', 'maximumPermutations'],
  draw: ({outputs}) => [
    {layer: 'dots', context: true, positions: outputs.positions, radius: 2},
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.classes,
      ramp: 'categorical',
      radius: 6
    }
  ]
});
