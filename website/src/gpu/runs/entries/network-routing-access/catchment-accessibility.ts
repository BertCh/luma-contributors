// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUCatchmentAccessibility (2SFCA and 3SFCA) on network costs: facility supply is shared among the
// demand inside each catchment, then summed back to every demand location.

import {getGPUNetworkCostWeightsParameterValues} from '@luma-contributors/analysis/gpu-network';
import {GPUCatchmentAccessibility} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {
  addCostMatrix,
  addCostWeights,
  createAccessScene,
  createFacilityDrag,
  getFacilityMarkers,
  moveFacility,
  uploadSearch
} from './shared/access';
import {getRoadContext, uploadRoads} from './shared/roads';
import {parameters} from './catchment-accessibility.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Supply-to-demand access per demand location, from network catchments.',
  createInputs: ({random, world}) => {
    const scene = createAccessScene(random, world, 8, 120);
    return {...scene, drag: createFacilityDrag(scene)};
  },
  build(builder, scene, values) {
    const {network} = scene;
    const search = uploadSearch(builder, network);
    const facilityNodes = builder.upload('facility-nodes', 'uint32', scene.facilityNodes);
    const facilities = builder.upload('facilities', 'float32x2', scene.facilityPositions);
    const matrix = addCostMatrix(builder, search, facilityNodes, network.nodeCount);
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
        facilities
      },
      update: current => {
        facilityNodes.write(scene.facilityNodes);
        facilities.write(scene.facilityPositions);
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
        glyphs: [
          {shape: 'ring', at, radius: values.costLimit, dashed: true},
          {shape: 'crosshair', at, size: 14}
        ],
        readout: `facility ${move.facility + 1} · costLimit ${values.costLimit.toFixed(2)}`
      };
    }
  },
  draw: ({outputs, theme}) => [
    getRoadContext(outputs),
    {
      layer: 'dots',
      positions: outputs.demands,
      values: outputs.accessibility,
      range: [0, 0.05],
      radius: 6
    },
    ...getFacilityMarkers(outputs.facilities, theme, {radius: 8})
  ]
});
