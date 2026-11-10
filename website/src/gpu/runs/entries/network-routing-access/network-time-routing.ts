// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUNetworkTimeRouting: earliest arrival with time-dependent (affine) travel durations. Roads are
// colored by the arrival time at each intersection for a given departure time; later departures
// meet slower roads.

import {GPUNetworkTimeRouting} from '@luma-contributors/analysis/gpu-network';
import {defineOperationRun} from '../../define-run';
import {findParameter, getFractionOf, getParameterAt} from '../../interaction';
import {
  createRoadNetwork,
  createRoads,
  getEndpointMarkers,
  getRoadContext,
  getRoadValues,
  uploadRoads
} from './shared/roads';
import {parameters} from './network-time-routing.parameters';

const DEPARTURE = findParameter(parameters, 'departure');

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
      world,
      origin: network.findNearestNode(world.width * 0.15, 0.5),
      destination: network.findNearestNode(world.width * 0.9, 0.5),
      costs: new Float32Array(network.slotCount * 2)
    };
  },
  build(builder, inputs) {
    const {network} = inputs;
    const origin = builder.upload(
      'origin-position',
      'float32x2',
      Float32Array.of(network.nodes[inputs.origin * 2], network.nodes[inputs.origin * 2 + 1])
    );
    const destination = builder.upload(
      'destination-position',
      'float32x2',
      Float32Array.of(
        network.nodes[inputs.destination * 2],
        network.nodes[inputs.destination * 2 + 1]
      )
    );
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
      outputs: {...uploadRoads(builder, network), arrivals, origin, destination},
      update: values => {
        // Duration = length * (1 + slope * departure-time factor): base and slope per transition.
        // The slope is largest in the middle of the picture, so later departures meet a slow centre.
        for (let slot = 0; slot < network.slotCount; slot++) {
          const source = network.slotSources[slot];
          const target = network.neighbors[slot];
          const middleX = (network.nodes[source * 2] + network.nodes[target * 2]) / 2;
          const middleY = (network.nodes[source * 2 + 1] + network.nodes[target * 2 + 1]) / 2;
          const distance = Math.hypot(middleX - inputs.world.width / 2, middleY - 0.5);
          const congestion = 0.1 + 3 * Math.exp(-((distance / 0.3) ** 2));
          inputs.costs[slot * 2] = network.lengths[slot];
          inputs.costs[slot * 2 + 1] = network.lengths[slot] * values.slope * congestion;
        }
        costs.write(inputs.costs);
        departure.write(Float32Array.of(values.departure));
      }
    };
  },
  // x is the departure time: the colors are the travel time since departing, on a fixed scale.
  interaction: {
    kind: 'time',
    hint: 'scrub the departure',
    rest: [getFractionOf(DEPARTURE, DEPARTURE.default), 0.5],
    resolve(pointer, {inputs, outputs, world, peek}) {
      const departure = getParameterAt(DEPARTURE, pointer.unitX);
      const arrivals = peek(outputs.arrivals);
      const arrival = arrivals ? arrivals[inputs.destination] : null;
      const left = world.width * 0.06;
      const right = world.width * 0.94;
      const at = left + (right - left) * (departure / DEPARTURE.max);
      return {
        values: {departure},
        glyphs: [
          {shape: 'line', from: [left, 0.04], to: [right, 0.04]},
          {shape: 'crosshair', at: [at, 0.04], size: 7}
        ],
        readout:
          arrival !== null && arrival < 1e20
            ? `departure ${departure.toFixed(2)} · arrive ${arrival.toFixed(2)}`
            : `departure ${departure.toFixed(2)}`
      };
    }
  },
  draw: ({outputs, values, theme}) => [
    getRoadContext(outputs),
    getRoadValues(outputs, outputs.arrivals, {
      radius: 2.4,
      range: [values.departure, values.departure + 5]
    }),
    ...getEndpointMarkers(outputs.origin, theme, {colorIndex: 4}),
    ...getEndpointMarkers(outputs.destination, theme, {colorIndex: 2})
  ]
});
