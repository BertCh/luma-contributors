// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUNetworkCoarsening: collapse a road network into compound nodes by a group label (here a
// spatial cell). Dots are the group centroids, sized by node count through color; the lines are the
// superedges between neighboring groups.

import {GPUNetworkCoarsening} from '@luma-contributors/analysis/gpu-network';
import {defineOperationRun} from '../../define-run';
import {addEdgePairs} from './shared/display';
import {
  createRoadNetwork,
  createRoads,
  getRoadContext,
  uploadRoads,
  uploadRoadTopology
} from './shared/roads';
import {parameters} from './network-coarsening.parameters';

const GROUP_CAPACITY = 64;
const EDGE_CAPACITY = 256;

export default defineOperationRun({
  parameters,
  caption: 'Group centroids and superedges after labelling nodes by spatial cell.',
  createInputs: ({random, world}) => ({base: createRoads(random, world, 12), world}),
  build(builder, {base, world}, values) {
    const network = createRoadNetwork(base, values.closures);
    const {topology} = uploadRoadTopology(builder, network);
    const nodePositions = builder.upload('node-positions', 'float32x2', network.nodes);
    const labels = builder.allocate('labels', 'uint32', network.nodeCount);
    const groupNodeCount = builder.allocate('group-node-count', 'uint32', GROUP_CAPACITY);
    const groupCentroid = builder.allocate('group-centroid', 'float32x2', GROUP_CAPACITY);
    const edgeIds = builder.allocate('edge-ids', 'uint32', EDGE_CAPACITY);
    const edgeCount = builder.allocate('edge-count', 'uint32', 1);
    const edgeOverflow = builder.allocate('edge-overflow', 'uint32', 1);
    const edgeTargets = builder.allocate('edge-targets', 'uint32', EDGE_CAPACITY);
    const edgeCounts = builder.allocate('edge-counts', 'uint32', EDGE_CAPACITY);
    builder.graph.add(
      new GPUNetworkCoarsening({
        topology: {...topology, nodePositions: nodePositions.view},
        labels: labels.view,
        groupCapacity: GROUP_CAPACITY,
        output: {
          groupNodeCount: groupNodeCount.view,
          groupCentroid: groupCentroid.view,
          edges: {ids: edgeIds.view, status: {count: edgeCount.view, overflow: edgeOverflow.view}},
          edgeTargetGroups: edgeTargets.view,
          edgeCounts: edgeCounts.view
        }
      })
    );
    return {
      outputs: {
        ...uploadRoads(builder, network),
        groupNodeCount,
        groupCentroid,
        superedges: addEdgePairs(builder, {
          id: 'superedges',
          from: edgeIds,
          to: edgeTargets,
          fallbackNode: GROUP_CAPACITY - 1
        })
      },
      update: current => {
        const columns = Math.ceil(world.width / current.cellSize);
        labels.write(
          Uint32Array.from({length: network.nodeCount}, (_, node) => {
            const column = Math.min(
              columns - 1,
              Math.floor(network.nodes[node * 2] / current.cellSize)
            );
            return column + Math.floor(network.nodes[node * 2 + 1] / current.cellSize) * columns;
          })
        );
      }
    };
  },
  rebuildOn: ['closures'],
  draw: ({outputs}) => [
    getRoadContext(outputs),
    {
      layer: 'segments',
      nodes: outputs.groupCentroid,
      edges: outputs.superedges,
      radius: 2.5,
      color: [0.5, 0.8, 1, 0.9]
    },
    {
      layer: 'dots',
      positions: outputs.groupCentroid,
      values: outputs.groupNodeCount,
      ramp: 'sequential',
      radius: 9
    }
  ]
});
