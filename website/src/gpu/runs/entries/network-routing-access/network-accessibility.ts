// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUNetworkAccessibility scores every intersection by the opportunities (facility supply) it can
// reach: gravity with distance decay, or a plain cumulative count inside the threshold. The cost
// matrix comes from GPUNetworkCostMatrix, one row per facility.

import {
  GPUNetworkAccessibility,
  getGPUNetworkAccessibilityParameterValues
} from '@luma-contributors/analysis/gpu-network';
import {defineOperationRun} from '../../define-run';
import {addCostMatrix, createAccessScene, DECAY_NAMES, uploadSearch} from './shared/access';
import {getRoadContext, getRoadValues, uploadRoads} from './shared/roads';
import {parameters} from './network-accessibility.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Opportunities reachable from each intersection, within the threshold.',
  createInputs: ({random, world}) => createAccessScene(random, world, 8, 8),
  build(builder, scene) {
    const {network} = scene;
    const search = uploadSearch(builder, network);
    const matrix = addCostMatrix(builder, search, scene.facilityNodes, network.nodeCount);
    const parameterBuffer = builder.allocate('parameters', 'float32', 4);
    const gravity = builder.allocate('gravity', 'float32', network.nodeCount);
    const cumulative = builder.allocate('cumulative', 'float32', network.nodeCount);
    builder.graph.add(
      new GPUNetworkAccessibility({
        costs: matrix.view,
        opportunityWeights: builder.upload('supply', 'float32', scene.supply).view,
        parameters: parameterBuffer.view,
        gravity: gravity.view,
        cumulative: cumulative.view
      })
    );
    return {
      outputs: {
        ...uploadRoads(builder, network),
        gravity,
        cumulative,
        facilities: builder.upload('facilities', 'float32x2', scene.facilityPositions)
      },
      update: values => {
        search.costLimit.write(Float32Array.of(2.5));
        parameterBuffer.write(
          getGPUNetworkAccessibilityParameterValues({
            threshold: values.threshold,
            decay: DECAY_NAMES[values.decay] as 'none' | 'exponential' | 'power',
            beta: values.beta,
            minimumCost: 0.1
          })
        );
      }
    };
  },
  draw: ({outputs, values}) => [
    getRoadContext(outputs),
    getRoadValues(outputs, values.measure === 0 ? outputs.gravity : outputs.cumulative, {
      radius: 2.4
    }),
    {layer: 'dots', positions: outputs.facilities, radius: 6, color: [1, 1, 1, 1]}
  ]
});
