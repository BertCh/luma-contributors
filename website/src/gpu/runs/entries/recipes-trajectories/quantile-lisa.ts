// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUQuantileLISA: tracts in the selected quantile class are the dots; the fill marks tracts where
// that class clusters significantly (conditional local join counts with permutation p-values).

import {
  getGPUPermutationParameterValues,
  GPUQuantileLISA
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createTracts, uploadSpatialWeights} from '../../inputs';
import {getTractFill, uploadTractMesh} from './lib/tracts';
import {parameters} from './quantile-lisa.parameters';

const MAXIMUM_PERMUTATIONS = 99;

export default defineOperationRun({
  parameters,
  caption: 'Where the chosen quantile class clusters (filled) among its members (dots).',
  createInputs: ({random, world}) => {
    const tracts = createTracts(random, world);
    return {tracts, draws: Float32Array.from({length: tracts.count}, () => random() - 0.5)};
  },
  build(builder, {tracts, draws}) {
    const valueBuffer = builder.allocate('values', 'float32', tracts.count);
    const quantileParameters = builder.allocate('quantile-parameters', 'uint32', 2);
    const permutationParameters = builder.allocate('permutation-parameters', 'uint32', 4);
    const membership = builder.allocate('membership', 'uint32', tracts.count);
    const boundaries = builder.allocate('boundaries', 'float32', 2);
    const validCount = builder.allocate('valid-count', 'uint32', 1);
    const classes = builder.allocate('classes', 'uint32', tracts.count);
    const invalidQuantile = builder.allocate('invalid-quantile', 'uint32', 1);
    const exceedances = builder.allocate('exceedances', 'uint32', tracts.count);
    const pseudoPValues = builder.allocate('pseudo-p-values', 'float32', tracts.count);
    const overflow = builder.allocate('overflow', 'uint32', 1);
    builder.graph.add(
      new GPUQuantileLISA({
        values: valueBuffer.view,
        weights: uploadSpatialWeights(builder, tracts.contiguity),
        quantileParameters: quantileParameters.view,
        parameters: permutationParameters.view,
        maximumPermutations: MAXIMUM_PERMUTATIONS,
        maximumNeighbors: 8,
        output: {
          membership: membership.view,
          boundaries: boundaries.view,
          validCount: validCount.view,
          classes: classes.view,
          invalidQuantile: invalidQuantile.view,
          exceedances: exceedances.view,
          pseudoPValues: pseudoPValues.view,
          status: {overflow: overflow.view}
        }
      })
    );
    return {
      outputs: {classes, membership, mesh: uploadTractMesh(builder, tracts)},
      update: current => {
        const quantileCount = current.numberOfQuantiles;
        quantileParameters.write(
          Uint32Array.of(quantileCount, Math.min(current.selectedQuantile, quantileCount))
        );
        permutationParameters.write(
          getGPUPermutationParameterValues({
            seed: 7,
            permutationCount: MAXIMUM_PERMUTATIONS,
            significanceLevel: current.significanceLevel
          })
        );
        valueBuffer.write(
          Float32Array.from(tracts.values, (value, index) => value + draws[index] * current.noise)
        );
      }
    };
  },
  draw: ({inputs, outputs}) => [
    getTractFill(outputs.mesh, inputs.tracts, outputs.classes),
    {
      layer: 'lines',
      context: true,
      segments: outputs.mesh.outline,
      count: inputs.tracts.outlineCount
    },
    {
      layer: 'dots',
      positions: outputs.mesh.centers,
      values: outputs.membership,
      ramp: 'sequential',
      range: [0, 1],
      fade: true,
      radius: 3.5
    }
  ]
});
