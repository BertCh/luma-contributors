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
import {findParameter, snapToParameter} from '../../interaction';
import {getChartPosition, getChartWorldX} from '../../pick';
import {parameters} from './network-cost-weights.parameters';

const COST_LIMIT = findParameter(parameters, 'costLimit');
const BETA = findParameter(parameters, 'beta');
const AXIS_MAXIMUM = 1.6;
const RECTANGLE = [0.07, 0.14, 0.95, 0.88] as const;

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
        result.weights.write(new Float32Array(result.weights.length).fill(Number.NaN));
        result.distances.write(new Float32Array(result.distances.length).fill(Number.NaN));
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
  // Level: x on the cost axis is the cutoff, y is the decay steepness.
  interaction: {
    kind: 'level',
    hint: 'set the cutoff',
    rest: ({world}) => {
      const x = getChartWorldX(RECTANGLE, world, COST_LIMIT.default / AXIS_MAXIMUM);
      const v = (BETA.default - BETA.min) / (BETA.max - BETA.min);
      const y = (RECTANGLE[1] + v * (RECTANGLE[3] - RECTANGLE[1])) * world.height;
      return [x / world.width, 1 - y / world.height];
    },
    resolve(pointer, {values, world}) {
      const hit = getChartPosition(RECTANGLE, world, pointer.x, pointer.y);
      const costLimit = snapToParameter(COST_LIMIT, hit.u * AXIS_MAXIMUM);
      const beta = snapToParameter(BETA, BETA.min + hit.v * (BETA.max - BETA.min));
      return {
        values: {costLimit, beta},
        glyphs: [{shape: 'curtain', x: getChartWorldX(RECTANGLE, world, costLimit / AXIS_MAXIMUM)}],
        readout: `costLimit ${costLimit.toFixed(2)} · beta ${beta.toFixed(2)} · ${parameters[1].options?.[values.decay]}`
      };
    }
  },
  draw: ({outputs}) => [
    {
      layer: 'chart',
      kind: 'scatter',
      values: outputs.weights,
      xValues: outputs.distances,
      xRange: [0, AXIS_MAXIMUM],
      rectangle: RECTANGLE,
      range: [0, 1.2],
      radius: 6,
      alpha: 0.5
    }
  ]
});
