// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUNetworkKernelDensity: event intensity along roads, from shortest-path distances. Every
// intersection is a query; roads are colored by the triangular-kernel intensity of the events
// (dots) within the network bandwidth.

import {GPUNetworkKernelDensity} from '@luma-contributors/analysis/gpu-network';
import {defineOperationRun} from '../../define-run';
import {addCostMatrix, uploadSearch} from './shared/access';
import {
  createRoads,
  getRoadContext,
  getRoadValues,
  gatherPositions,
  uploadRoads
} from './shared/roads';
import {parameters} from './network-kernel-density.parameters';

const EVENT_COUNT = 40;

export default defineOperationRun({
  parameters,
  caption: 'Network kernel density of the events (dots), per intersection.',
  createInputs: ({random, world}) => {
    const network = createRoads(random, world, 8);
    // Each intersection represents half the length of its roads.
    const quadratureLengths = new Float32Array(network.nodeCount);
    for (let node = 0; node < network.nodeCount; node++) {
      for (let slot = network.offsets[node]; slot < network.offsets[node + 1]; slot++) {
        quadratureLengths[node] += network.lengths[slot] / 2;
      }
    }
    const pick = (clustered: boolean) =>
      Uint32Array.from({length: EVENT_COUNT}, (_, index) =>
        clustered
          ? network.findNearestNode(
              world.width * (0.25 + 0.5 * (index % 2)) + (random() - 0.5) * 0.2,
              0.5 + (random() - 0.5) * 0.4
            )
          : network.findNearestNode(random() * world.width, random())
      );
    return {
      network,
      quadratureLengths,
      uniform: pick(false),
      clustered: pick(true),
      events: new Uint32Array(EVENT_COUNT)
    };
  },
  build(builder, inputs, values) {
    const {network} = inputs;
    const search = uploadSearch(builder, network);
    const queries = Uint32Array.from({length: network.nodeCount}, (_, index) => index);
    // Events change with the clustering slider, so their nodes are rewritten each frame.
    const eventNodes = builder.allocate('event-nodes', 'uint32', EVENT_COUNT);
    const distances = addCostMatrix(builder, search, queries, network.nodeCount, eventNodes);
    const intensity = builder.allocate('intensity', 'float32', network.nodeCount);
    const kernelMass = builder.allocate('kernel-mass', 'float32', EVENT_COUNT);
    const status = builder.allocate('status', 'uint32', network.nodeCount);
    builder.graph.add(
      new GPUNetworkKernelDensity({
        distances: distances.view,
        eventCount: EVENT_COUNT,
        quadratureLengths: builder.upload('quadrature-lengths', 'float32', inputs.quadratureLengths)
          .view,
        bandwidth: values.bandwidth,
        output: {intensity: intensity.view, kernelMass: kernelMass.view, status: status.view}
      })
    );
    const events = builder.upload(
      'events',
      'float32x2',
      gatherPositions(network.nodes, inputs.uniform)
    );
    return {
      outputs: {
        ...uploadRoads(builder, network),
        intensity,
        events
      },
      update: current => {
        const mix = Uint32Array.from(inputs.uniform, (uniformNode, index) =>
          index / EVENT_COUNT < current.clustering ? inputs.clustered[index] : uniformNode
        );
        eventNodes.write(mix);
        events.write(gatherPositions(network.nodes, mix));
        search.costLimit.write(Float32Array.of(4));
      }
    };
  },
  rebuildOn: ['bandwidth'],
  draw: ({outputs}) => [
    getRoadContext(outputs),
    getRoadValues(outputs, outputs.intensity, {radius: 2.4}),
    {layer: 'dots', positions: outputs.events, radius: 5, color: [1, 1, 1, 1]}
  ]
});
