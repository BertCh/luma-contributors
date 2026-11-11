// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUNetworkPotentialPathArea: the space-time prism of a traveller who leaves the origin and must be
// at the destination within a budget. Streets are colored by how tight the trip is: brightest where
// a trip through them has the least time to spare; streets that no trip within the budget could use stay muted. Two reachability runs
// supply the trees (the streets are two-way with equal lengths, so the tree to the destination is the
// tree from it).

import {
  GPUNetworkPotentialPathArea,
  GPUNetworkReachability
} from '@luma-contributors/analysis/gpu-network';
import {defineOperationRun, type RunBuffer} from '../../define-run';
import {addPrismBudget, addPrismTightness} from './_lib/network-kernels';
import {
  createNodeIndex,
  createRoadNetwork,
  createRoads,
  getEndpointMarkers,
  getRoadContext,
  getSlotValues,
  uploadRoads,
  uploadRoadTopology,
  uploadSlotLines
} from '../network-routing-access/shared/roads';
import {parameters} from './network-potential-path-area.parameters';

const ORIGIN_SITES = [
  [0.1, 0.5],
  [0.3, 0.9],
  [0.3, 0.1]
] as const;

/** Street lengths are world units: one unit is about 3 km, driven at 30 km/h. */
const MINUTES_PER_UNIT = 6;
const formatMinutes = (units: number | undefined, divisor = 1) =>
  units === undefined ? '…' : `${Math.round((units * MINUTES_PER_UNIT) / divisor)} min`;

type Inputs = ReturnType<typeof createInputs>;
type Outputs = ReturnType<typeof uploadRoads> & {
  slack: RunBuffer;
  budget: RunBuffer;
  nodeInPrism: RunBuffer;
  summary: RunBuffer;
  slackLines: ReturnType<typeof uploadSlotLines>;
  originMarker: RunBuffer;
  destinationMarker: RunBuffer;
};

function createInputs({
  random,
  world
}: {
  random: () => number;
  world: {width: number; height: number};
}) {
  const network = createRoadNetwork(createRoads(random, world, 14), 0);
  const destination = network.findNearestNode(world.width * 0.88, 0.5);
  return {
    network,
    world,
    nodeIndex: createNodeIndex(network),
    originNodes: Uint32Array.from(ORIGIN_SITES, ([x, y]) =>
      network.findNearestNode(x * world.width, y)
    ),
    defaultDestination: destination,
    destination
  };
}

export default defineOperationRun<Inputs, Outputs>({
  parameters,
  caption:
    'Streets a traveller could use between the two anchors within the budget, brightest where the trip has the least time to spare. Street lengths stand in for travel time on a San Francisco TIGER/Line extract of about 350 intersections.',
  createInputs,
  build(builder, inputs) {
    const {network} = inputs;
    const {topology} = uploadRoadTopology(builder, network);
    const originNode = builder.allocate('origin-node', 'uint32', 1);
    const destinationNode = builder.allocate('destination-node', 'uint32', 1);
    const forward = builder.allocate('forward-costs', 'float32', network.nodeCount);
    const reverse = builder.allocate('reverse-costs', 'float32', network.nodeCount);
    const search = {maximumIterations: 24, localIterations: 16};
    builder.graph.add(
      new GPUNetworkReachability({
        id: 'forward',
        topology,
        startNodes: originNode.view,
        ...search,
        output: {costs: forward.view}
      })
    );
    builder.graph.add(
      new GPUNetworkReachability({
        id: 'reverse',
        topology,
        startNodes: destinationNode.view,
        ...search,
        output: {costs: reverse.view}
      })
    );
    const spare = builder.allocate('spare', 'float32', 1);
    const budget = addPrismBudget(builder, {forward, destination: destinationNode, spare});
    const edgeInPrism = builder.allocate('edge-in-prism', 'uint32', network.slotCount);
    const edgeSlack = builder.allocate('edge-slack', 'float32', network.slotCount);
    const nodeInPrism = builder.allocate('node-in-prism', 'uint32', network.nodeCount);
    const summary = builder.allocate('summary', 'uint32', 2);
    builder.graph.add(
      new GPUNetworkPotentialPathArea({
        topology,
        forwardCosts: forward.view,
        reverseCosts: reverse.view,
        budget: budget.view,
        output: {
          edgeInPrism: edgeInPrism.view,
          edgeSlack: edgeSlack.view,
          nodeInPrism: nodeInPrism.view,
          summary: summary.view
        }
      })
    );
    const originMarker = builder.allocate('origin-marker', 'float32x2', 1);
    const destinationMarker = builder.allocate('destination-marker', 'float32x2', 1);
    return {
      outputs: {
        ...uploadRoads(builder, network),
        slack: addPrismTightness(builder, {
          inPrism: edgeInPrism,
          slack: edgeSlack,
          budget,
          spare,
          slotCount: network.slotCount
        }),
        nodeInPrism,
        budget,
        summary,
        slackLines: uploadSlotLines(builder, network, 0.004),
        originMarker,
        destinationMarker
      },
      update: values => {
        const origin = inputs.originNodes[values.origin];
        originNode.write(Uint32Array.of(origin));
        destinationNode.write(Uint32Array.of(inputs.destination));
        spare.write(Float32Array.of(values.spare));
        originMarker.write(network.nodes.slice(origin * 2, origin * 2 + 2));
        destinationMarker.write(
          network.nodes.slice(inputs.destination * 2, inputs.destination * 2 + 2)
        );
      }
    };
  },
  // Drag: the destination anchor follows the pointer to the nearest intersection; the prism reshapes.
  interaction: {
    kind: 'drag',
    hint: 'drag the destination',
    rest: ({inputs, toUnit}) =>
      toUnit([
        inputs.network.nodes[inputs.defaultDestination * 2],
        inputs.network.nodes[inputs.defaultDestination * 2 + 1]
      ]),
    resolve(pointer, {inputs, outputs, values, peek}) {
      const node = inputs.nodeIndex.nearest(pointer.x, pointer.y);
      const changed = node !== inputs.destination;
      inputs.destination = node;
      const summary = peek(outputs.summary);
      return {
        rerun: changed,
        spotlight: null,
        glyphs: [
          {
            shape: 'crosshair',
            at: [inputs.network.nodes[node * 2], inputs.network.nodes[node * 2 + 1]],
            size: 9
          }
        ],
        readout: `budget ${formatMinutes(peek(outputs.budget)?.[0])} (+${Math.round(values.spare * 100)}%) · shortest ${formatMinutes(peek(outputs.budget)?.[0], 1 + values.spare)}${
          summary
            ? summary[1] === 0
              ? ' · unreachable within budget'
              : ` · ${summary[0]} nodes · ${summary[1]} edges`
            : ''
        }`
      };
    }
  },
  draw: ({outputs, theme}) => [
    {...getRoadContext(outputs), radius: 0.8, color: [0.45, 0.5, 0.6, 0.35]},
    getSlotValues(outputs.slackLines, outputs.slack, {
      radius: 1.8,
      ramp: 'sequential',
      range: [0, 1]
    }),
    ...getEndpointMarkers(outputs.originMarker, theme, {colorIndex: 4}),
    ...getEndpointMarkers(outputs.destinationMarker, theme, {colorIndex: 2})
  ]
});
