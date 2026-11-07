// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUNetworkStructure: bridges, articulation points and strongly connected components of a road
// network with closures. A bridge is a road whose loss disconnects the network.

import {GPUNetworkStructure} from '@luma-contributors/analysis/gpu-network';
import {defineOperationRun} from '../../define-run';
import {
  createRoadNetwork,
  createRoads,
  getRoadContext,
  getRoadValues,
  getSlotValues,
  uploadRoadTopology,
  uploadRoads,
  uploadSlotLines
} from './shared/roads';
import {parameters} from './network-structure.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Bridges (roads whose loss disconnects), articulation points, strong components.',
  createInputs: ({random, world}) => ({base: createRoads(random, world)}),
  build(builder, {base}, values) {
    const network = createRoadNetwork(base, values.closures);
    const {topology} = uploadRoadTopology(builder, network);
    const bridges = builder.allocate('bridges', 'uint32', network.slotCount);
    const articulationPoints = builder.allocate('articulation-points', 'uint32', network.nodeCount);
    const strongComponents = builder.allocate('strong-components', 'uint32', network.nodeCount);
    builder.graph.add(
      new GPUNetworkStructure({
        topology,
        reverseEdgeSlots: builder.upload('reverse-slots', 'uint32', network.reverseSlots).view,
        output: {
          bridges: bridges.view,
          articulationPoints: articulationPoints.view,
          strongComponents: strongComponents.view
        }
      })
    );
    return {
      outputs: {
        ...uploadRoads(builder, network),
        slots: uploadSlotLines(builder, network),
        bridges,
        articulationPoints,
        strongComponents
      }
    };
  },
  draw: ({outputs, values}) => [
    getRoadContext(outputs),
    values.output === 0
      ? getSlotValues(outputs.slots, outputs.bridges, {radius: 2.4})
      : getRoadValues(
          outputs,
          values.output === 1 ? outputs.articulationPoints : outputs.strongComponents,
          {radius: 2.4}
        )
  ]
});
