// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPULocalPermutationTest: each point's local statistic against the same statistic when its
// neighbors' values are drawn at random. Points whose pseudo p-value passes the significance
// level are drawn in color; the rest stay dark. More permutations sharpen the p-values.

import {
  getGPUPermutationParameterValues,
  GPULocalPermutationTest,
  GPU_PERMUTATION_PARAMETER_LENGTH
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {uploadSpatialWeights} from '../../inputs';
import {
  createNeighborGraph,
  createPointField,
  createProbeInputs,
  mixNoise,
  probeNeighbors
} from './shared/point-field';
import {parameters} from './local-permutation-test.parameters';

const STATISTICS = ['localMoran', 'localGeary', 'localG', 'localGStar'] as const;
const ALTERNATIVES = ['two-sided', 'greater', 'lesser'] as const;
const MAXIMUM_PERMUTATIONS = 499;

export default defineOperationRun({
  parameters,
  caption: 'Points whose local statistic beats the shuffled neighbors are significant.',
  createInputs: ({random, world}) => {
    const points = createPointField(random, world, 600, 0);
    return {...createProbeInputs(points), graph: createNeighborGraph(points.positions, 6)};
  },
  build(builder, points, values) {
    const {graph} = points;
    const valueBuffer = builder.allocate('values', 'float32', points.count);
    const parameterBuffer = builder.allocate(
      'permutation',
      'uint32',
      GPU_PERMUTATION_PARAMETER_LENGTH
    );
    const exceedances = builder.allocate('exceedances', 'uint32', points.count);
    const pseudoPValues = builder.allocate('pseudo-p-values', 'float32', points.count);
    const significant = builder.allocate('significant', 'uint32', points.count);
    const overflow = builder.allocate('overflow', 'uint32', 1);
    builder.graph.add(
      new GPULocalPermutationTest({
        weights: uploadSpatialWeights(builder, graph.weights),
        values: valueBuffer.view,
        statistic: STATISTICS[values.statistic],
        alternative: ALTERNATIVES[values.alternative],
        parameters: parameterBuffer.view,
        maximumPermutations: MAXIMUM_PERMUTATIONS,
        output: {
          exceedances: exceedances.view,
          pseudoPValues: pseudoPValues.view,
          significant: significant.view,
          status: {overflow: overflow.view}
        }
      })
    );
    return {
      outputs: {
        significant,
        pseudoPValues,
        positions: builder.upload('positions', 'float32x2', points.positions)
      },
      update: current => {
        valueBuffer.write(mixNoise(points.field, points.noise, current.noise));
        parameterBuffer.write(
          getGPUPermutationParameterValues(
            {seed: 11, permutationCount: current.permutationCount},
            {maximumPermutations: MAXIMUM_PERMUTATIONS}
          )
        );
      }
    };
  },
  rebuildOn: ['statistic', 'alternative'],
  interaction: {
    kind: 'probe',
    hint: 'hover a point',
    rest: [0.5, 0.5],
    resolve(pointer, {inputs, outputs, values, peek}) {
      const probe = probeNeighbors(inputs, pointer, inputs.graph.weights);
      if (probe.row < 0) return {};
      const pseudoP = peek(outputs.pseudoPValues)?.[probe.row];
      const isSignificant = peek(outputs.significant)?.[probe.row] === 1;
      return {
        glyphs: probe.glyphs,
        readout: `${STATISTICS[values.statistic]} · p ${
          pseudoP === undefined ? '…' : pseudoP.toFixed(3)
        } (${values.permutationCount} shuffles)${isSignificant ? ' · significant' : ''}`
      };
    }
  },
  draw: ({outputs, theme}) => [
    {layer: 'dots', positions: outputs.positions, color: [...theme.ink, 0.4], radius: 2.5},
    // Only significant points are drawn in color: the faded mask hides the rest.
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.significant,
      ramp: 'sequential',
      range: [0, 1],
      fade: true,
      radius: 6
    }
  ]
});
