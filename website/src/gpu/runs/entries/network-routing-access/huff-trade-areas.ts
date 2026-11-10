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
import {
  addCostMatrix,
  addCostWeights,
  createAccessScene,
  createFacilityDrag,
  createFacilityIds,
  getFacilityMarkers,
  moveFacility,
  uploadSearch
} from './shared/access';
import {getRoadContext, uploadRoads} from './shared/roads';
import {parameters} from './huff-trade-areas.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Each demand location is colored by the facility with the highest Huff probability.',
  createInputs: ({random, world}) => {
    const scene = createAccessScene(random, world, 6, 360);
    return {...scene, drag: createFacilityDrag(scene)};
  },
  build(builder, scene) {
    const {network} = scene;
    const search = uploadSearch(builder, network);
    const facilityNodes = builder.upload('facility-nodes', 'uint32', scene.facilityNodes);
    const facilities = builder.upload('facilities', 'float32x2', scene.facilityPositions);
    const matrix = addCostMatrix(builder, search, facilityNodes, network.nodeCount);
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
        facilities,
        facilityIds: builder.upload('facility-ids', 'uint32', createFacilityIds(facilityCount))
      },
      update: values => {
        facilityNodes.write(scene.facilityNodes);
        facilities.write(scene.facilityPositions);
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
  interaction: {
    kind: 'drag',
    hint: 'drag a facility',
    rest: ({inputs, toUnit}) =>
      toUnit([inputs.drag.defaultPositions[0], inputs.drag.defaultPositions[1]]),
    resolve(pointer, {inputs, values, peek, outputs}) {
      const move = moveFacility(inputs, inputs.drag, pointer.x, pointer.y);
      const at: [number, number] = [
        inputs.network.nodes[move.node * 2],
        inputs.network.nodes[move.node * 2 + 1]
      ];
      return {
        rerun: move.changed,
        glyphs: [{shape: 'crosshair', at, size: 14}],
        readout: `facility ${move.facility + 1} · exponent ${values.exponent} · beta ${values.beta}`
      };
    }
  },
  draw: ({outputs, theme}) => [
    getRoadContext(outputs),
    {layer: 'dots', positions: outputs.demands, values: outputs.tradeArea, radius: 5},
    ...getFacilityMarkers(outputs.facilities, theme, {ids: outputs.facilityIds, radius: 9})
  ]
});
