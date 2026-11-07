// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUHuffTradeAreas: each demand location patronises facilities in proportion to attractiveness
// and distance decay; its trade area is the most probable facility. Costs are network costs.

import {getGPUNetworkCostWeightsParameterValues} from '@luma-contributors/analysis/gpu-network';
import {
  GPUHuffTradeAreas,
  getGPUHuffTradeAreasParameterValues
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {addCostMatrix, addCostWeights, createAccessScene, uploadSearch} from './shared/access';
import {getRoadContext, uploadRoads} from './shared/roads';
import {parameters} from './huff-trade-areas.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Each demand location is colored by the facility with the highest Huff probability.',
  createInputs: ({random, world}) => createAccessScene(random, world, 6, 360),
  build(builder, scene) {
    const {network} = scene;
    const search = uploadSearch(builder, network);
    const matrix = addCostMatrix(builder, search, scene.facilityNodes, network.nodeCount);
    const decayParameters = builder.allocate('decay-parameters', 'float32', 4);
    const exponentBuffer = builder.allocate('exponent', 'float32', 1);
    const facilityCount = scene.facilityNodes.length;
    // Demand-major weights: row = demand location, neighbors = facilities.
    const demandWeights = addCostWeights(
      builder,
      matrix,
      facilityCount,
      scene.demandNodes,
      'columns',
      decayParameters
    );
    const tradeArea = builder.allocate('trade-area', 'uint32', scene.demandNodes.length);
    builder.graph.add(
      new GPUHuffTradeAreas({
        attractiveness: builder.upload('attractiveness', 'float32', scene.supply).view,
        demandWeights: demandWeights.spatialWeights,
        parameters: exponentBuffer.view,
        output: {tradeArea: tradeArea.view}
      })
    );
    return {
      outputs: {
        ...uploadRoads(builder, network),
        tradeArea,
        demands: builder.upload('demands', 'float32x2', scene.demandPositions),
        facilities: builder.upload('facilities', 'float32x2', scene.facilityPositions)
      },
      update: values => {
        search.costLimit.write(Float32Array.of(2.5));
        decayParameters.write(
          getGPUNetworkCostWeightsParameterValues({
            costLimit: values.costLimit,
            decay: 'exponential',
            beta: values.beta
          })
        );
        exponentBuffer.write(getGPUHuffTradeAreasParameterValues({exponent: values.exponent}));
      }
    };
  },
  draw: ({outputs}) => [
    getRoadContext(outputs),
    {layer: 'dots', positions: outputs.demands, values: outputs.tradeArea, radius: 5},
    {layer: 'dots', positions: outputs.facilities, radius: 9, color: [1, 1, 1, 1]}
  ]
});
