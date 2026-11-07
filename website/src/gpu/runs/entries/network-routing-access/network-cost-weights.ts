// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUNetworkCostWeights turns a network cost matrix into spatial weights with a distance decay.
// Every slot it writes is one (cost, weight) pair, drawn as a scatter: the decay curve itself.

import {getGPUNetworkCostWeightsParameterValues} from '@luma-contributors/analysis/gpu-network';
import {defineOperationRun} from '../../define-run';
import {
  addCostMatrix,
  addCostWeights,
  createAccessScene,
  DECAY_NAMES,
  uploadSearch
} from './shared/access';
import {parameters} from './network-cost-weights.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Each point is one origin-facility pair: network cost against its decay weight.',
  createInputs: ({random, world}) => createAccessScene(random, world, 12, 8),
  build(builder, scene) {
    const {network} = scene;
    const search = uploadSearch(builder, network);
    const matrix = addCostMatrix(builder, search, scene.facilityNodes, network.nodeCount);
    const parameterBuffer = builder.allocate('parameters', 'float32', 4);
    const result = addCostWeights(
      builder,
      matrix,
      scene.facilityNodes.length,
      undefined,
      'rows',
      parameterBuffer
    );
    return {
      outputs: {weights: result.weights, distances: result.distances},
      update: values => {
        search.costLimit.write(Float32Array.of(2));
        // Clear the slot tail so a smaller catchment leaves no stale points behind.
        result.weights.write(new Float32Array(result.weights.length));
        result.distances.write(new Float32Array(result.distances.length));
        parameterBuffer.write(
          getGPUNetworkCostWeightsParameterValues({
            costLimit: values.costLimit,
            decay: DECAY_NAMES[values.decay],
            beta: values.beta,
            minimumCost: 0.1
          })
        );
      }
    };
  },
  draw: ({outputs, values}) => [
    {
      layer: 'chart',
      kind: 'scatter',
      values: outputs.weights,
      xValues: outputs.distances,
      xRange: [0, values.costLimit],
      range: [0, 1.2],
      radius: 6,
      alpha: 0.5
    }
  ]
});
