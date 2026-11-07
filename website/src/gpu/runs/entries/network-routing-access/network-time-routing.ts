// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUNetworkTimeRouting: earliest arrival with time-dependent (affine) travel durations. Roads are
// colored by the arrival time at each intersection for a given departure time; later departures
// meet slower roads.

import {GPUNetworkTimeRouting} from '@luma-contributors/analysis/gpu-network';
import {defineOperationRun} from '../../define-run';
import {
  createRoadNetwork,
  createRoads,
  getRoadContext,
  getRoadValues,
  uploadRoads
} from './shared/roads';
import {parameters} from './network-time-routing.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Earliest arrival time at each intersection; later departures meet slower roads.',
  createInputs: ({random, world}) => {
    const network = createRoadNetwork(createRoads(random, world, 8), 0);
    const transitions = new Uint32Array(network.slotCount * 2);
    for (let slot = 0; slot < network.slotCount; slot++) {
      transitions[slot * 2] = network.slotSources[slot];
      transitions[slot * 2 + 1] = network.neighbors[slot];
    }
    return {
      network,
      transitions,
      origin: network.findNearestNode(world.width * 0.15, 0.5),
      destination: network.findNearestNode(world.width * 0.9, 0.5),
      costs: new Float32Array(network.slotCount * 2)
    };
  },
  build(builder, inputs) {
    const {network} = inputs;
    const stateVertices = Uint32Array.from({length: network.nodeCount}, (_, index) => index);
    const departure = builder.allocate('departure', 'float32', 1);
    const costs = builder.allocate('costs', 'float32x2', network.slotCount);
    const arrivals = builder.allocate('arrivals', 'float32', network.nodeCount);
    const predecessors = builder.allocate('predecessors', 'uint32', network.nodeCount);
    const summary = builder.allocate('summary', 'uint32', 4);
    builder.graph.add(
      new GPUNetworkTimeRouting({
        stateVertices: builder.upload('state-vertices', 'uint32', stateVertices).view,
        transitions: builder.upload('transitions', 'uint32x2', inputs.transitions).view,
        costModel: 'affine-duration',
        costs: costs.view,
        endpoints: builder.upload(
          'endpoints',
          'uint32',
          Uint32Array.of(inputs.origin, inputs.destination)
        ).view,
        departure: departure.view,
        output: {
          arrivals: arrivals.view,
          predecessorTransitions: predecessors.view,
          summary: summary.view
        }
      })
    );
    return {
      outputs: {...uploadRoads(builder, network), arrivals},
      update: values => {
        // Duration = length * (1 + slope * departure-time factor): base and slope per transition.
        for (let slot = 0; slot < network.slotCount; slot++) {
          inputs.costs[slot * 2] = network.lengths[slot];
          inputs.costs[slot * 2 + 1] =
            network.lengths[slot] * values.slope * (0.3 + ((slot * 37) % 10) / 10);
        }
        costs.write(inputs.costs);
        departure.write(Float32Array.of(values.departure));
      }
    };
  },
  draw: ({outputs}) => [
    getRoadContext(outputs),
    getRoadValues(outputs, outputs.arrivals, {radius: 2.4, range: [0, 4]})
  ]
});
