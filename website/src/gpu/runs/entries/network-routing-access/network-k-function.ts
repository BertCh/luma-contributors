// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUNetworkKFunction: Ripley's K on a road network. The bright curve is the observed K(d); the
// thin curves are the minimum and maximum of random patterns on the same network. A curve above
// the envelope means the events are clustered along the roads.

import {
  GPUNetworkKFunction,
  getGPUNetworkKFunctionParameterValues
} from '@luma-contributors/analysis/gpu-network';
import {defineOperationRun} from '../../define-run';
import {uploadNetworkTopology} from '../../inputs';
import {createRoads} from './shared/roads';
import {addSlice} from './shared/display';
import {parameters} from './network-k-function.parameters';

const EVENT_COUNT = 90;
const BAND_COUNT = 48;
const SIMULATION_COUNT = 19;

export default defineOperationRun({
  parameters,
  caption: 'Observed K(d) against the envelope of random patterns on the same network.',
  createInputs: ({random, world}) => {
    const network = createRoads(random, world, 8);
    let networkLength = 0;
    for (let slot = 0; slot < network.slotCount; slot++) networkLength += network.lengths[slot] / 2;
    return {
      network,
      networkLength,
      uniform: Float32Array.from({length: EVENT_COUNT * 2}, (_, index) =>
        index % 2 === 0 ? random() * world.width : random()
      ),
      hotspots: Float32Array.from({length: EVENT_COUNT * 2}, (_, index) =>
        index % 2 === 0
          ? world.width * (0.25 + 0.5 * (Math.floor(index / 2) % 2)) + (random() - 0.5) * 0.18
          : 0.5 + (random() - 0.5) * 0.5
      ),
      events: new Float32Array(EVENT_COUNT * 2)
    };
  },
  build(builder, inputs) {
    const {network} = inputs;
    const {topology} = uploadNetworkTopology(builder, network);
    const nodePositions = builder.upload('node-positions', 'float32x2', network.nodes);
    const events = builder.allocate('events', 'float32x2', EVENT_COUNT);
    const maximumDistance = builder.allocate('maximum-distance', 'float32', 1);
    const networkLength = builder.allocate('network-length', 'float32', 1);
    const parameterBuffer = builder.allocate('parameters', 'uint32', 4);
    const kValues = builder.allocate('k-values', 'float32', (1 + SIMULATION_COUNT) * BAND_COUNT);
    const envelope = builder.allocate('envelope', 'float32', 3 * BAND_COUNT);
    const snapped = builder.allocate('snapped', 'uint32', 1);
    const converged = builder.allocate('converged', 'uint32', 1);
    builder.graph.add(
      new GPUNetworkKFunction({
        positions: events.view,
        topology: {...topology, nodePositions: nodePositions.view},
        maximumDistance: maximumDistance.view,
        networkLength: networkLength.view,
        parameters: parameterBuffer.view,
        bandCount: BAND_COUNT,
        simulationCount: SIMULATION_COUNT,
        maximumIterations: 48,
        output: {
          kValues: kValues.view,
          envelope: envelope.view,
          snappedEventCount: snapped.view,
          status: {converged: converged.view}
        }
      })
    );
    return {
      outputs: {
        observed: addSlice(builder, {
          id: 'observed',
          source: kValues,
          offset: 0,
          length: BAND_COUNT
        }),
        lower: addSlice(builder, {id: 'lower', source: envelope, offset: 0, length: BAND_COUNT}),
        upper: addSlice(builder, {
          id: 'upper',
          source: envelope,
          offset: 2 * BAND_COUNT,
          length: BAND_COUNT
        })
      },
      update: values => {
        for (let index = 0; index < inputs.events.length; index++) {
          inputs.events[index] =
            inputs.uniform[index] * (1 - values.clustering) +
            inputs.hotspots[index] * values.clustering;
        }
        events.write(inputs.events);
        maximumDistance.write(Float32Array.of(values.maximumDistance));
        networkLength.write(Float32Array.of(inputs.networkLength));
        parameterBuffer.write(
          getGPUNetworkKFunctionParameterValues({
            seed: 4306,
            activeSimulations: Math.round(values.activeSimulations)
          })
        );
      }
    };
  },
  draw: ({inputs, outputs}) => {
    const range = [0, inputs.networkLength * 1.05] as const;
    return [
      {
        layer: 'chart',
        kind: 'line',
        values: outputs.upper,
        range,
        color: [0.5, 0.7, 1, 0.6],
        radius: 1.5
      },
      {
        layer: 'chart',
        kind: 'line',
        values: outputs.lower,
        range,
        color: [0.5, 0.7, 1, 0.6],
        radius: 1.5
      },
      {
        layer: 'chart',
        kind: 'line',
        values: outputs.observed,
        range,
        color: [1, 0.78, 0.25, 1],
        radius: 3
      }
    ];
  }
});
