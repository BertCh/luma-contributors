// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUNetworkSubgraphFilter: keeps the roads whose speed lies in a range. Surviving roads are
// colored, the rest stay muted; intersections left without a road drop out.

import {
  GPUNetworkSubgraphFilter,
  getGPUNetworkSubgraphFilterParameterValues
} from '@luma-contributors/analysis/gpu-network';
import {defineOperationRun} from '../../define-run';
import {
  createRoadNetwork,
  createRoads,
  getRoadContext,
  getSlotValues,
  uploadRoadTopology,
  uploadRoads,
  uploadSlotLines
} from './shared/roads';
import {parameters} from './network-subgraph-filter.parameters';

const LAYOUT = {nodeColumnCount: 0, edgeColumnCount: 1, hasEdgeTimes: false} as const;

export default defineOperationRun({
  parameters,
  caption: 'Roads kept by the speed filter, colored; dropped roads are muted.',
  createInputs: ({random, world}) => {
    const network = createRoadNetwork(createRoads(random, world), 0);
    // One speed per road, shared by both directed slots.
    const speeds = Float32Array.from({length: network.slotCount}, (_, slot) => {
      const road = Math.min(slot, network.reverseSlots[slot]);
      return (Math.imul(road + 1, 2654435761) >>> 0) / 4294967296;
    });
    return {network, speeds};
  },
  build(builder, {network, speeds}) {
    const {topology} = uploadRoadTopology(builder, network);
    const parameterBuffer = builder.allocate('parameters', 'float32', 4);
    const nodeMask = builder.allocate('node-mask', 'uint32', network.nodeCount);
    const edgeMask = builder.allocate('edge-mask', 'uint32', network.slotCount);
    builder.graph.add(
      new GPUNetworkSubgraphFilter({
        topology,
        edgeColumns: [builder.upload('speed', 'float32', speeds).view],
        parameters: parameterBuffer.view,
        dropIsolated: true,
        output: {nodeMask: nodeMask.view, edgeMask: edgeMask.view}
      })
    );
    return {
      outputs: {
        ...uploadRoads(builder, network),
        slots: uploadSlotLines(builder, network),
        edgeMask,
        nodeMask
      },
      update: values =>
        parameterBuffer.write(
          getGPUNetworkSubgraphFilterParameterValues(LAYOUT, {
            edgeRanges: [[values.minimumSpeed, values.maximumSpeed]]
          })
        )
    };
  },
  draw: ({outputs}) => [
    getRoadContext(outputs),
    getSlotValues(outputs.slots, outputs.edgeMask, {radius: 2.2})
  ]
});
