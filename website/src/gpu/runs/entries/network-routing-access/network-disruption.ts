// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// addNetworkDisruptionRecipe: close roads (edgeMask) and compare origin-destination costs and
// accessibility before and after. Closed roads are drawn in the second color; origins show how
// much reachable opportunity each one lost.

import {
  addNetworkDisruptionRecipe,
  getGPUNetworkAccessibilityParameterValues
} from '@luma-contributors/analysis/gpu-network';
import {defineOperationRun} from '../../define-run';
import {uploadNetworkTopology} from '../../inputs';
import {
  gatherPositions,
  createRoadNetwork,
  createRoads,
  getSlotValues,
  pickNodes,
  uploadSlotLines
} from './shared/roads';

const ORIGIN_COUNT = 40;

import {parameters} from './network-disruption.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Closed roads, and the accessibility each origin loses (diverging).',
  createInputs: ({random, world}) => {
    const network = createRoadNetwork(createRoads(random, world, 12), 0);
    const origins = pickNodes(network, world, ORIGIN_COUNT, 5);
    const jobs = Float32Array.from({length: network.nodeCount}, () =>
      random() < 0.15 ? 1 + random() * 4 : 0
    );
    return {network, origins, jobs};
  },
  build(builder, {network, origins, jobs}) {
    const {topology} = uploadNetworkTopology(builder, network, network.lengths);
    const edgeMask = builder.allocate('edge-mask', 'uint32', network.slotCount);
    const parameterBuffer = builder.allocate('parameters', 'float32', 4);
    const accessibilityDifferences = builder.allocate(
      'accessibility-differences',
      'float32',
      ORIGIN_COUNT
    );
    addNetworkDisruptionRecipe(builder.graph, {
      search: {
        topology,
        startNodes: builder.upload('origins', 'uint32', origins).view,
        maximumIterations: 64
      },
      rowCount: ORIGIN_COUNT,
      edgeMask: edgeMask.view,
      opportunityWeights: builder.upload('jobs', 'float32', jobs).view,
      parameters: parameterBuffer.view,
      outputs: {accessibilityDifferences: accessibilityDifferences.view}
    });
    return {
      outputs: {
        slots: uploadSlotLines(builder, network),
        edgeMask,
        accessibilityDifferences,
        origins: builder.upload(
          'origin-positions',
          'float32x2',
          gatherPositions(network.nodes, origins)
        )
      },
      update: values => {
        edgeMask.write(
          Uint32Array.from(network.reverseSlots, (reverse, slot) => {
            const road = Math.min(slot, reverse);
            return (Math.imul(road + 1, 2654435761) >>> 0) / 4294967296 >= values.closures ? 1 : 0;
          })
        );
        parameterBuffer.write(
          getGPUNetworkAccessibilityParameterValues({threshold: values.threshold})
        );
      }
    };
  },
  draw: ({outputs}) => [
    getSlotValues(outputs.slots, outputs.edgeMask, {context: false, radius: 1.6} as never),
    {
      layer: 'dots',
      positions: outputs.origins,
      values: outputs.accessibilityDifferences,
      ramp: 'diverging',
      radius: 10
    }
  ]
});
