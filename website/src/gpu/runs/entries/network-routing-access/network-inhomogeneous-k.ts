// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUNetworkInhomogeneousK: linear K for events whose intensity varies along the network, with
// Ang's correction for junctions. Distances come from GPUNetworkCostMatrix (event by node).
// The line is K(r) for 24 radii; the bars are its numerator.

import {GPUNetworkInhomogeneousK} from '@luma-contributors/analysis/gpu-network';
import {defineOperationRun} from '../../define-run';
import {addCostMatrix, uploadSearch} from './shared/access';
import {createInsetMap, createRoads, gatherPositions, getRoadContext} from './shared/roads';
import {getChartPosition, getChartWorldX} from '../../pick';
import {parameters} from './network-inhomogeneous-k.parameters';

const RECTANGLE = [0.56, 0.14, 0.97, 0.88] as const;

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
    return {network, world, eventNodes, edgeLengths, inset: createInsetMap(world)};
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
      outputs: {
        k,
        numerator,
        nodes: builder.upload('display-nodes', 'float32x2', inputs.inset.transform(network.nodes)),
        edges: builder.upload('road-edges', 'uint32x2', network.edgePairs),
        eventDots: builder.upload(
          'event-dots',
          'float32x2',
          inputs.inset.transform(gatherPositions(network.nodes, inputs.eventNodes))
        )
      },
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
  // Level: x on the chart is the radius; a rule marks it and the readout gives K there.
  interaction: {
    kind: 'level',
    hint: 'read K at a radius',
    rest: [0.5, 0.5],
    resolve(pointer, {inputs, outputs, values, world, peek}) {
      // Over the chart the rule follows the pointer; over the map the pointer sweeps the whole axis.
      const chartHit = getChartPosition(RECTANGLE, world, pointer.x, pointer.y);
      const hit = {u: pointer.unitX < RECTANGLE[0] ? pointer.unitX / RECTANGLE[0] : chartHit.u};
      const index = Math.min(RADIUS_COUNT - 1, Math.floor(hit.u * RADIUS_COUNT));
      const k = peek(outputs.k);
      const radius = ((index + 1) / RADIUS_COUNT) * values.radii;
      // The same radius as a ring around the event nearest the middle of the map.
      let central = inputs.eventNodes[0];
      const distanceOf = (candidate: number) =>
        Math.hypot(
          inputs.network.nodes[candidate * 2] - world.width / 2,
          inputs.network.nodes[candidate * 2 + 1] - 0.5
        );
      for (const node of inputs.eventNodes)
        if (distanceOf(node) < distanceOf(central)) central = node;
      return {
        glyphs: [
          {shape: 'curtain', x: getChartWorldX(RECTANGLE, world, (index + 0.5) / RADIUS_COUNT)},
          {
            shape: 'ring',
            at: inputs.inset.toMap(
              inputs.network.nodes[central * 2],
              inputs.network.nodes[central * 2 + 1]
            ),
            radius: radius * inputs.inset.scale
          }
        ],
        readout: `r ${radius.toFixed(2)} · K ${k ? k[index].toFixed(2) : '-'}`
      };
    }
  },
  draw: ({outputs}) => [
    getRoadContext(outputs),
    {layer: 'dots', positions: outputs.eventDots, radius: 4, color: [1, 0.78, 0.25, 1]},
    {
      layer: 'chart',
      kind: 'bars',
      rectangle: RECTANGLE,
      values: outputs.numerator,
      range: [0, 20],
      alpha: 0.35
    },
    {
      layer: 'chart',
      kind: 'line',
      rectangle: RECTANGLE,
      values: outputs.k,
      range: [0, 2.5],
      radius: 3
    }
  ]
});
