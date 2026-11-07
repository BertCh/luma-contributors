// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUNetworkNeighborhood: the k-hop ego network around the start nodes. Roads are colored by hop
// distance; intersections beyond `hops` stay unreached.

import {GPUNetworkNeighborhood} from '@luma-contributors/analysis/gpu-network';
import {defineOperationRun} from '../../define-run';
import {uploadNetworkTopology} from '../../inputs';
import {
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
    return {network, starts: pickNodes(network, world, MAXIMUM_STARTS, 3)};
  },
  build(builder, {network, starts}) {
    const {topology} = uploadNetworkTopology(builder, network);
    const hops = builder.allocate('hops', 'uint32', 1);
    const startCount = builder.allocate('start-count', 'uint32', 1);
    const hopDistances = builder.allocate('hop-distances', 'uint32', network.nodeCount);
    builder.graph.add(
      new GPUNetworkNeighborhood({
        topology,
        startNodes: builder.upload('start-nodes', 'uint32', starts).view,
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
        startPositions: builder.upload(
          'start-positions',
          'float32x2',
          gatherPositions(network.nodes, starts)
        )
      },
      update: values => {
        hops.write(Uint32Array.of(values.hops));
        startCount.write(Uint32Array.of(values.startCount));
      }
    };
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
