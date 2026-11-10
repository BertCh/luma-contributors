// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUNetworkNeighborhood: the k-hop ego network around the start nodes. Roads are colored by hop
// distance; intersections beyond `hops` stay unreached.

import {GPUNetworkNeighborhood} from '@luma-contributors/analysis/gpu-network';
import {defineOperationRun} from '../../define-run';
import {uploadNetworkTopology} from '../../inputs';
import {chooseSticky} from '../../pick';
import {
  createNodeIndex,
  createRoads,
  gatherPositions,
  getRoadContext,
  getRoadValues,
  pickNodes,
  uploadRoads
} from './shared/roads';
import {parameters} from './network-neighborhood.parameters';

const MAXIMUM_STARTS = 4;
const MAXIMUM_HOPS = 12;

export default defineOperationRun({
  parameters,
  caption: 'Hop distance from the start nodes, out to the hops limit.',
  createInputs: ({random, world}) => {
    const network = createRoads(random, world);
    const starts = pickNodes(network, world, MAXIMUM_STARTS, 3);
    return {
      network,
      starts,
      defaultStart: starts[0],
      index: createNodeIndex(network)
    };
  },
  build(builder, {network, starts}) {
    const {topology} = uploadNetworkTopology(builder, network);
    const hops = builder.allocate('hops', 'uint32', 1);
    const startCount = builder.allocate('start-count', 'uint32', 1);
    const hopDistances = builder.allocate('hop-distances', 'uint32', network.nodeCount);
    const startNodes = builder.upload('start-nodes', 'uint32', starts);
    const startPositions = builder.upload(
      'start-positions',
      'float32x2',
      gatherPositions(network.nodes, starts)
    );
    builder.graph.add(
      new GPUNetworkNeighborhood({
        topology,
        startNodes: startNodes.view,
        startCount: startCount.view,
        hops: hops.view,
        maximumHops: MAXIMUM_HOPS,
        output: {hopDistances: hopDistances.view}
      })
    );
    return {
      outputs: {
        ...uploadRoads(builder, network),
        hopDistances,
        startPositions
      },
      update: values => {
        startNodes.write(starts);
        startPositions.write(gatherPositions(network.nodes, starts));
        hops.write(Uint32Array.of(values.hops));
        startCount.write(Uint32Array.of(values.startCount));
      }
    };
  },
  // The pointer moves the first start node; the hop rings regrow from it.
  interaction: {
    kind: 'place',
    hint: 'place the start',
    rest: ({inputs, toUnit}) =>
      toUnit([
        inputs.network.nodes[inputs.defaultStart * 2],
        inputs.network.nodes[inputs.defaultStart * 2 + 1]
      ]),
    resolve(pointer, {inputs, outputs, values, peek}) {
      const node = chooseSticky(inputs.index, pointer.x, pointer.y, inputs.starts[0]);
      const changed = node !== inputs.starts[0];
      inputs.starts[0] = node;
      const hops = peek(outputs.hopDistances);
      let reached = 0;
      if (hops) for (const hop of hops) if (hop <= values.hops) reached++;
      return {
        rerun: changed,
        glyphs: [
          {
            shape: 'crosshair',
            at: [inputs.network.nodes[node * 2], inputs.network.nodes[node * 2 + 1]],
            size: 14
          }
        ],
        readout: `start node ${node} · hops ${values.hops} · ${reached} intersections`
      };
    }
  },
  draw: ({outputs, values}) => [
    getRoadContext(outputs),
    getRoadValues(outputs, outputs.hopDistances, {ramp: 'sequential', range: [-4, 12]}),
    {
      layer: 'dots',
      positions: outputs.startPositions,
      count: values.startCount,
      radius: 7,
      color: [1, 1, 1, 1]
    }
  ]
});
