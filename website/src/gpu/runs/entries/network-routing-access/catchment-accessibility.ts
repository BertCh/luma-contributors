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
import {createLegend} from '../../text';
import {addMatrixFootprint, addZoneSegments} from './shared/display';
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
    // The dragged facility's catchment: the roads within costLimit of it, from its row of the same
    // cost matrix, read on the GPU.
    const activeFacility = builder.allocate('active-facility', 'uint32', 1);
    const reachLimit = builder.allocate('reach-limit', 'float32', 1);
    const footprint = addMatrixFootprint(builder, {
      id: 'catchment',
      costs: matrix,
      nodeCount: network.nodeCount,
      columnCount: network.nodeCount,
      mode: 'row',
      selector: activeFacility,
      limit: reachLimit,
      floor: 1
    });
    const roads = uploadRoads(builder, network);
    const reach = addZoneSegments(builder, {
      id: 'catchment',
      nodes: roads.nodes,
      edges: roads.edges,
      values: footprint
    });
    return {
      outputs: {
        ...roads,
        reach,
        accessibility,
        demands: builder.upload('demands', 'float32x2', scene.demandPositions),
        facilities
      },
      update: current => {
        facilityNodes.write(scene.facilityNodes);
        facilities.write(scene.facilityPositions);
        search.costLimit.write(Float32Array.of(2));
        activeFacility.write(Uint32Array.of(Math.max(scene.drag.active, 0)));
        reachLimit.write(Float32Array.of(current.costLimit));
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
        glyphs: [{shape: 'crosshair', at, size: 14}],
        readout: `facility ${move.facility + 1} · costLimit ${values.costLimit.toFixed(2)}`
      };
    }
  },
  draw: ({outputs, theme}) => [
    getRoadContext(outputs),
    // The roads inside the dragged facility's catchment (within costLimit of it).
    {layer: 'lines', segments: outputs.reach, radius: 4, color: [...theme.categories[3], 0.7]},
    {
      layer: 'dots',
      positions: outputs.demands,
      values: outputs.accessibility,
      range: [0, 0.05],
      radius: 7
    },
    ...getFacilityMarkers(outputs.facilities, theme, {radius: 8}),
    {
      layer: 'text',
      ...createLegend({
        corner: 'top-left',
        entries: [
          {
            label: 'roads within costLimit of the dragged facility',
            swatch: [...theme.categories[3]]
          },
          {label: 'demand, by accessibility (output)', swatch: [...theme.ramp[4]]}
        ]
      })
    }
  ]
});
