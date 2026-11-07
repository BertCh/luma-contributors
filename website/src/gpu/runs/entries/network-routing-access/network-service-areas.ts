// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUNetworkServiceAreas on a street grid: every intersection takes the color of the facility that
// is cheapest to reach. Sliders write the operation's per-frame views (active facility count, cost
// limit) and the road costs, so the service areas re-form without recompiling.

import {GPUNetworkServiceAreas} from '@luma-contributors/analysis/gpu-network';
import {defineOperationRun} from '../../define-run';
import {createNetwork, uploadNetworkTopology} from '../../inputs';
import {parameters} from './network-service-areas.parameters';

const MAXIMUM_FACILITIES = 8;

export default defineOperationRun({
  parameters,
  caption: 'Each road takes the color of its nearest facility by travel cost.',
  createInputs: ({random, world}) => {
    const network = createNetwork(random, world);
    // Facilities spread over the network: evenly strided intersections.
    const facilityNodes = Uint32Array.from({length: MAXIMUM_FACILITIES}, (_, index) =>
      network.findNearestNode(world.width * ((index * 0.37 + 0.13) % 1), (index * 0.61 + 0.2) % 1)
    );
    return {network, facilityNodes, costs: new Float32Array(network.slotCount)};
  },
  build(builder, inputs) {
    const {network, facilityNodes} = inputs;
    const {topology, weights} = uploadNetworkTopology(builder, network);
    const facilities = builder.upload('facilities', 'uint32', facilityNodes);
    const facilityCount = builder.allocate('facility-count', 'uint32', 1);
    const costLimit = builder.allocate('cost-limit', 'float32', 1);
    const assignments = builder.allocate('assignments', 'uint32', network.nodeCount);
    const costs = builder.allocate('costs', 'float32', network.nodeCount);
    builder.graph.add(
      new GPUNetworkServiceAreas({
        topology,
        facilities: facilities.view,
        facilityCount: facilityCount.view,
        costLimit: costLimit.view,
        maximumIterations: 32,
        output: {assignments: assignments.view, costs: costs.view}
      })
    );
    return {
      outputs: {
        assignments,
        nodes: builder.upload('nodes', 'float32x2', network.nodes),
        edges: builder.upload('edges', 'uint32x2', network.edgePairs),
        facilityPositions: builder.upload(
          'facility-positions',
          'float32x2',
          gatherFacilityPositions(network.nodes, facilityNodes)
        )
      },
      update: values => {
        network.fillWeights(inputs.costs, values.congestion);
        weights.write(inputs.costs);
        facilityCount.write(Uint32Array.of(Math.round(values.facilityCount)));
        costLimit.write(Float32Array.of(values.costLimit));
      }
    };
  },
  draw: ({outputs, values}) => [
    {layer: 'segments', context: true, nodes: outputs.nodes, edges: outputs.edges, radius: 1},
    {
      layer: 'segments',
      nodes: outputs.nodes,
      edges: outputs.edges,
      values: outputs.assignments,
      radius: 1.8
    },
    {
      layer: 'dots',
      positions: outputs.facilityPositions,
      count: Math.round(values.facilityCount),
      radius: 7,
      color: [1, 1, 1, 1]
    }
  ]
});

function gatherFacilityPositions(nodes: Float32Array, facilityNodes: Uint32Array): Float32Array {
  const positions = new Float32Array(facilityNodes.length * 2);
  facilityNodes.forEach((node, index) => {
    positions[index * 2] = nodes[node * 2];
    positions[index * 2 + 1] = nodes[node * 2 + 1];
  });
  return positions;
}
