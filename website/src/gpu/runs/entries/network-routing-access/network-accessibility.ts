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
import {
  addCostMatrix,
  createAccessScene,
  createFacilityDrag,
  DECAY_NAMES,
  getFacilityMarkers,
  moveFacility,
  uploadSearch
} from './shared/access';
import {getRoadContext, getRoadValues, uploadRoads} from './shared/roads';
import {parameters} from './network-accessibility.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Opportunities reachable from each intersection, within the threshold.',
  createInputs: ({random, world}) => {
    const scene = createAccessScene(random, world, 8, 8);
    return {...scene, drag: createFacilityDrag(scene)};
  },
  build(builder, scene) {
    const {network} = scene;
    const search = uploadSearch(builder, network);
    const facilityNodes = builder.upload('facility-nodes', 'uint32', scene.facilityNodes);
    const facilities = builder.upload('facilities', 'float32x2', scene.facilityPositions);
    const matrix = addCostMatrix(builder, search, facilityNodes, network.nodeCount);
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
        facilities
      },
      update: values => {
        facilityNodes.write(scene.facilityNodes);
        facilities.write(scene.facilityPositions);
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
          {shape: 'ring', at, radius: values.threshold, dashed: true},
          {shape: 'crosshair', at, size: 14}
        ],
        readout: `opportunity ${move.facility + 1} · threshold ${values.threshold.toFixed(2)}`
      };
    }
  },
  draw: ({outputs, values, theme}) => [
    getRoadContext(outputs),
    getRoadValues(outputs, values.measure === 0 ? outputs.gravity : outputs.cumulative, {
      radius: 2.4,
      range: [0, values.measure === 0 ? 8 : 8]
    }),
    ...getFacilityMarkers(outputs.facilities, theme, {radius: 7})
  ]
});
