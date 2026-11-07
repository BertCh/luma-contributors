// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPULOSHResidualPermutationTest: LOSH (local spatial heteroscedasticity) plus a conditional
// permutation test of its residuals. Points whose neighborhood variability is significant are
// drawn in color. The noise sits in a vertical band, so that band is where the test fires.

import {
  getGPUPermutationParameterValues,
  GPULOSHResidualPermutationTest,
  GPU_PERMUTATION_PARAMETER_LENGTH
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {uploadSpatialWeights} from '../../inputs';
import {createNeighborGraph, createPointField} from './shared/point-field';
import {parameters} from './losh-residual-permutation-test.parameters';

const ALTERNATIVES = ['two-sided', 'greater', 'lesser'] as const;
const MAXIMUM_PERMUTATIONS = 499;

export default defineOperationRun({
  parameters,
  caption: 'Significant local variability, from permuting the LOSH residuals.',
  createInputs: ({random, world}) => createPointField(random, world, 600, false),
  build(builder, points, values) {
    const graph = createNeighborGraph(points.positions, 6);
    const valueBuffer = builder.allocate('values', 'float32', points.count);
    const parameterBuffer = builder.allocate(
      'permutation',
      'uint32',
      GPU_PERMUTATION_PARAMETER_LENGTH
    );
    const statistic = builder.allocate('statistic', 'float32', points.count);
    const status = builder.allocate('status', 'uint32', points.count);
    const exceedances = builder.allocate('exceedances', 'uint32', points.count);
    const pseudoPValues = builder.allocate('pseudo-p-values', 'float32', points.count);
    const adjustedPValues = builder.allocate('adjusted-p-values', 'float32', points.count);
    const significant = builder.allocate('significant', 'uint32', points.count);
    const overflow = builder.allocate('overflow', 'uint32', 1);
    builder.graph.add(
      new GPULOSHResidualPermutationTest({
        weights: uploadSpatialWeights(builder, graph.weights),
        values: valueBuffer.view,
        alternative: ALTERNATIVES[values.alternative],
        parameters: parameterBuffer.view,
        maximumPermutations: MAXIMUM_PERMUTATIONS,
        output: {
          statistic: statistic.view,
          status: status.view,
          inference: {
            exceedances: exceedances.view,
            pseudoPValues: pseudoPValues.view,
            adjustedPValues: adjustedPValues.view,
            significant: significant.view,
            status: {overflow: overflow.view}
          }
        }
      })
    );
    return {
      outputs: {
        significant,
        statistic,
        positions: builder.upload('positions', 'float32x2', points.positions)
      },
      update: current => {
        const mixed = Float32Array.from(points.field, (value, index) => {
          const x = points.positions[index * 2] / points.worldWidth;
          return Math.abs(x - 0.5) < 0.18
            ? value + (points.noise[index] - 0.5) * current.noise * 2
            : value;
        });
        valueBuffer.write(mixed);
        parameterBuffer.write(
          getGPUPermutationParameterValues(
            {seed: 5, permutationCount: current.permutationCount},
            {maximumPermutations: MAXIMUM_PERMUTATIONS}
          )
        );
      }
    };
  },
  rebuildOn: ['alternative'],
  draw: ({outputs}) => [
    {layer: 'dots', context: true, positions: outputs.positions, radius: 2},
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.significant,
      ramp: 'categorical',
      radius: 4.5
    }
  ]
});
