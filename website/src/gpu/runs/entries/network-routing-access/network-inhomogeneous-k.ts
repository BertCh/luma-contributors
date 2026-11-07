// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUNetworkInhomogeneousK: linear K for events whose intensity varies along the network, with
// Ang's correction for junctions. Distances come from GPUNetworkCostMatrix (event by node).
// The line is K(r) for 24 radii; the bars are its numerator.

import {GPUNetworkInhomogeneousK} from '@luma-contributors/analysis/gpu-network';
import {defineOperationRun} from '../../define-run';
import {addCostMatrix, uploadSearch} from './shared/access';
import {createRoads} from './shared/roads';
import {parameters} from './network-inhomogeneous-k.parameters';

const EVENT_COUNT = 32;
const RADIUS_COUNT = 24;

export default defineOperationRun({
  parameters,
  caption: 'Inhomogeneous K(r): pair counts weighted by inverse intensity, per radius.',
  createInputs: ({random, world}) => {
    const network = createRoads(random, world, 6);
    const events = new Set<number>();
    while (events.size < EVENT_COUNT)
      events.add(network.findNearestNode(random() * world.width, random()));
    const eventNodes = Uint32Array.from(events);
    const edgeLengths = Float32Array.from({length: network.edgeCount}, (_, edge) =>
      Math.hypot(
        network.nodes[network.edgePairs[edge * 2] * 2] -
          network.nodes[network.edgePairs[edge * 2 + 1] * 2],
        network.nodes[network.edgePairs[edge * 2] * 2 + 1] -
          network.nodes[network.edgePairs[edge * 2 + 1] * 2 + 1]
      )
    );
    return {network, world, eventNodes, edgeLengths};
  },
  build(builder, inputs, values) {
    const {network} = inputs;
    const search = uploadSearch(builder, network);
    const eventBuffer = builder.upload('event-nodes', 'uint32', inputs.eventNodes);
    const distances = addCostMatrix(builder, search, eventBuffer, network.nodeCount);
    const intensity = builder.allocate('intensity', 'float32', EVENT_COUNT);
    const radii = builder.allocate('radii', 'float32', RADIUS_COUNT);
    const k = builder.allocate('k', 'float32', RADIUS_COUNT);
    const numerator = builder.allocate('numerator', 'float32', RADIUS_COUNT);
    const normalization = builder.allocate('normalization', 'float32', 3);
    const status = builder.allocate('status', 'uint32', RADIUS_COUNT);
    builder.graph.add(
      new GPUNetworkInhomogeneousK({
        eventNodeIds: eventBuffer.view,
        intensity: intensity.view,
        nodeDistances: distances.view,
        nodeCount: network.nodeCount,
        edgeNodes: builder.upload('edge-nodes', 'uint32x2', network.edgePairs).view,
        edgeLengths: builder.upload('edge-lengths', 'float32', inputs.edgeLengths).view,
        radii: radii.view,
        correction: values.correction === 1 ? 'ang' : 'none',
        output: {
          k: k.view,
          numerator: numerator.view,
          normalization: normalization.view,
          status: status.view
        }
      })
    );
    return {
      outputs: {k, numerator},
      update: current => {
        search.costLimit.write(Float32Array.of(8));
        radii.write(
          Float32Array.from(
            {length: RADIUS_COUNT},
            (_, index) => ((index + 1) / RADIUS_COUNT) * current.radii
          )
        );
        intensity.write(
          Float32Array.from(
            inputs.eventNodes,
            node => 1 + (current.gradient * network.nodes[node * 2]) / inputs.world.width
          )
        );
      }
    };
  },
  rebuildOn: ['correction'],
  draw: ({outputs}) => [
    {layer: 'chart', kind: 'bars', values: outputs.numerator, alpha: 0.35},
    {layer: 'chart', kind: 'line', values: outputs.k, radius: 3}
  ]
});
