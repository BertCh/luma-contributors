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
  getRoadValues,
  uploadRoads,
  uploadRoadTopology
} from './shared/roads';
import type {RunGlyph} from '../../interaction';
import {findParameter} from '../../interaction';
import {parameters} from './network-coarsening.parameters';

const CELL_SIZE = findParameter(parameters, 'cellSize');

const GROUP_CAPACITY = 128;
const EDGE_CAPACITY = 256;

export default defineOperationRun({
  parameters,
  caption: 'Group centroids and superedges after labelling nodes by spatial cell.',
  createInputs: ({random, world}) => ({
    base: createRoads(random, world, 12),
    world,
    offset: [0, 0] as [number, number]
  }),
  build(builder, inputs, values) {
    const {base, world} = inputs;
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
        labels,
        edgeCount,
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
        // The pointer slides the label grid; cells are numbered from the cell left of and below it.
        const columns = Math.ceil(world.width / current.cellSize) + 2;
        labels.write(
          Uint32Array.from({length: network.nodeCount}, (_, node) => {
            const column = Math.floor(
              (network.nodes[node * 2] - inputs.offset[0]) / current.cellSize
            );
            const row = Math.floor(
              (network.nodes[node * 2 + 1] - inputs.offset[1]) / current.cellSize
            );
            return (
              Math.min(GROUP_CAPACITY - 2, column + 1 + (row + 1) * columns) % (GROUP_CAPACITY - 1)
            );
          })
        );
      }
    };
  },
  rebuildOn: ['closures'],
  // Window: the pointer slides the label grid, so the cell under it lines up with its corner.
  interaction: {
    kind: 'window',
    hint: 'slide the grid',
    rest: ({world, toUnit}) => {
      const cell = CELL_SIZE.default;
      return toUnit([cell * Math.floor(world.width / 2 / cell), cell * Math.floor(0.5 / cell)]);
    },
    resolve(pointer, {inputs, outputs, values, world, peek}) {
      const cell = values.cellSize;
      const offset: [number, number] = [pointer.x % cell, pointer.y % cell];
      const changed = offset[0] !== inputs.offset[0] || offset[1] !== inputs.offset[1];
      inputs.offset = offset;
      const glyphs: RunGlyph[] = [];
      for (let x = offset[0]; x <= world.width; x += cell) {
        glyphs.push({shape: 'line', from: [x, 0], to: [x, 1], dashed: true});
      }
      for (let y = offset[1]; y <= 1; y += cell) {
        glyphs.push({shape: 'line', from: [0, y], to: [world.width, y], dashed: true});
      }
      const groups = peek(outputs.groupNodeCount);
      const superedges = peek(outputs.edgeCount);
      let groupCount = 0;
      if (groups) for (const count of groups) if (count > 0) groupCount++;
      return {
        rerun: changed,
        glyphs,
        readout: `cellSize ${cell.toFixed(2)} · ${groupCount} groups · ${superedges ? superedges[0] : 0} superedges`
      };
    }
  },
  draw: ({outputs}) => [
    getRoadContext(outputs),
    getRoadValues(outputs, outputs.labels, {radius: 1.8}),
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
