// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUCatchmentAccessibility (2SFCA and 3SFCA) on network costs: facility supply is shared among the
// demand inside each catchment, then summed back to every demand location.

import {getGPUNetworkCostWeightsParameterValues} from '@luma-contributors/analysis/gpu-network';
import {GPUCatchmentAccessibility} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {addCostMatrix, addCostWeights, createAccessScene, uploadSearch} from './shared/access';
import {getRoadContext, uploadRoads} from './shared/roads';
import {parameters} from './catchment-accessibility.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Supply-to-demand access per demand location, from network catchments.',
  createInputs: ({random, world}) => createAccessScene(random, world, 8, 240),
  build(builder, scene, values) {
    const {network} = scene;
    const search = uploadSearch(builder, network);
    const matrix = addCostMatrix(builder, search, scene.facilityNodes, network.nodeCount);
    const parameterBuffer = builder.allocate('parameters', 'float32', 4);
    const facilityCount = scene.facilityNodes.length;
    const facilityWeights = addCostWeights(
      builder,
      matrix,
      facilityCount,
      scene.demandNodes,
      'rows',
      parameterBuffer
    );
    const demandWeights = addCostWeights(
      builder,
      matrix,
      facilityCount,
      scene.demandNodes,
      'columns',
      parameterBuffer
    );
    const accessibility = builder.allocate('accessibility', 'float32', scene.demandNodes.length);
    const ratios = builder.allocate('ratios', 'float32', facilityCount);
    builder.graph.add(
      new GPUCatchmentAccessibility({
        method: values.method === 0 ? '2sfca' : '3sfca',
        supply: builder.upload('supply', 'float32', scene.supply).view,
        demand: builder.upload('demand', 'float32', scene.population).view,
        facilityWeights: facilityWeights.spatialWeights,
        demandWeights: demandWeights.spatialWeights,
        output: {accessibility: accessibility.view, ratios: ratios.view}
      })
    );
    return {
      outputs: {
        ...uploadRoads(builder, network),
        accessibility,
        demands: builder.upload('demands', 'float32x2', scene.demandPositions),
        facilities: builder.upload('facilities', 'float32x2', scene.facilityPositions)
      },
      update: current => {
        search.costLimit.write(Float32Array.of(2));
        parameterBuffer.write(
          getGPUNetworkCostWeightsParameterValues({
            costLimit: current.costLimit,
            decay: current.decay === 0 ? 'none' : 'gaussian'
          })
        );
      }
    };
  },
  rebuildOn: ['method'],
  draw: ({outputs}) => [
    getRoadContext(outputs),
    {layer: 'dots', positions: outputs.demands, values: outputs.accessibility, radius: 5},
    {layer: 'dots', positions: outputs.facilities, radius: 8, color: [1, 1, 1, 1]}
  ]
});
