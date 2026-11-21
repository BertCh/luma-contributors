// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUMapMatchedEdgeVolumes: noisy GPS tracks are matched to the street network by GPUMapMatching, and
// every edge on each matched route, including the edges between fixes, counts one traversal. Streets
// are drawn per direction, colored by their traversal count.

import {
  GPUMapMatchedEdgeVolumes,
  GPUMapMatching,
  getGPUMapMatchingParameterValues
} from '@luma-contributors/analysis/gpu-network';
import {defineOperationRun, type RunBuffer} from '../../define-run';
import {addSlotValues} from '../network-routing-access/shared/display';
import {
  createNodeIndex,
  createRoadNetwork,
  createRoads,
  createSlotSegments,
  findNearestSlot,
  getRoadContext,
  getSlotValues,
  uploadRoads,
  uploadRoadTopology,
  uploadSlotLines
} from '../network-routing-access/shared/roads';
import {parameters} from './map-matched-edge-volumes.parameters';

const TRACK_COUNTS = [24, 48, 96];
const FIXES_PER_TRACK = 12;
const WALK_NODES = 2 * FIXES_PER_TRACK;
const TRAVERSED_EDGE_CAPACITY = 48;
const HUB_SITES = [
  [0.5, 0.5],
  [0.22, 0.3],
  [0.8, 0.7]
] as const;

type Inputs = ReturnType<typeof createInputs>;
type Outputs = ReturnType<typeof uploadRoads> & {
  volumes: RunBuffer;
  volumeLines: RunBuffer;
  slotLines: ReturnType<typeof uploadSlotLines>;
  fixes: RunBuffer;
  truncated: RunBuffer;
};

function createInputs({
  random,
  world
}: {
  random: () => number;
  world: {width: number; height: number};
}) {
  const network = createRoadNetwork(createRoads(random, world, 14), 0);
  const hubs = HUB_SITES.map(([x, y]) => network.findNearestNode(x * world.width, y));
  const maximumTracks = TRACK_COUNTS[TRACK_COUNTS.length - 1];
  // Each track drives toward one of the hubs from a random start, mostly choosing the neighbor that
  // gets closer, so trips share corridors the way real traffic does.
  const truth = new Float32Array(maximumTracks * FIXES_PER_TRACK * 2);
  for (let track = 0; track < maximumTracks; track++) {
    const hub = hubs[track % hubs.length];
    let node = network.findNearestNode(random() * world.width, random());
    let previous = -1;
    for (let step = 0; step < WALK_NODES; step++) {
      if (step % 2 === 0) {
        const fix = track * FIXES_PER_TRACK + step / 2;
        truth[fix * 2] = network.nodes[node * 2];
        truth[fix * 2 + 1] = network.nodes[node * 2 + 1];
      }
      const options: number[] = [];
      for (let slot = network.offsets[node]; slot < network.offsets[node + 1]; slot++) {
        if (network.neighbors[slot] !== previous) options.push(network.neighbors[slot]);
      }
      if (!options.length) options.push(previous);
      const distanceToHub = (candidate: number) =>
        Math.hypot(
          network.nodes[candidate * 2] - network.nodes[hub * 2],
          network.nodes[candidate * 2 + 1] - network.nodes[hub * 2 + 1]
        );
      options.sort((a, b) => distanceToHub(a) - distanceToHub(b));
      const next = random() < 0.75 ? options[0] : options[Math.floor(random() * options.length)];
      previous = node;
      node = next;
    }
  }
  const noise = Float32Array.from(
    {length: truth.length},
    () => (random() + random() + random() - 1.5) * 1.3
  );
  return {
    network,
    world,
    truth,
    noise,
    slotSegments: createSlotSegments(network),
    nodeIndex: createNodeIndex(network),
    fixes: new Float32Array(truth.length),
    focus: {slot: -1}
  };
}

export default defineOperationRun<Inputs, Outputs>({
  parameters,
  caption:
    'Traversals per street direction from GPS tracks matched to the network. Tracks are simulated trips toward three hubs, with added GPS noise, on a San Francisco TIGER/Line extract of about 350 intersections.',
  createInputs,
  build(builder, inputs, values) {
    const {network} = inputs;
    const trackCount = TRACK_COUNTS[values.trackCount];
    const fixCount = trackCount * FIXES_PER_TRACK;
    const {topology} = uploadRoadTopology(builder, network);
    const positions = builder.allocate('fixes', 'float32x2', fixCount);
    const parameterBuffer = builder.allocate('parameters', 'float32', 8);
    const routeEdges = builder.allocate(
      'route-edges',
      'uint32',
      trackCount * TRAVERSED_EDGE_CAPACITY
    );
    const routeCounts = builder.allocate('route-counts', 'uint32', trackCount);
    const routeOverflow = builder.allocate('route-overflow', 'uint32', 1);
    const routeCandidateOverflow = builder.allocate('route-candidate-overflow', 'uint32', 1);
    builder.graph.add(
      new GPUMapMatching({
        positions: positions.view,
        trackOffsets: builder.upload(
          'track-offsets',
          'uint32',
          Uint32Array.from({length: trackCount + 1}, (_, index) => index * FIXES_PER_TRACK)
        ).view,
        topology: {
          ...topology,
          nodePositions: builder.upload('node-positions', 'float32x2', network.nodes).view
        },
        parameters: parameterBuffer.view,
        candidateCount: 8,
        // The default 64-node route search drops nodes on some hops of this network, which marks
        // every picture with 48 or more tracks as incomplete; 128 keeps the default settings complete.
        routeNodeBudget: 128,
        cellSize: 0.06,
        bounds: {minimum: [-0.2, -0.2], maximum: [inputs.world.width + 0.2, 1.2]},
        traversedEdgeCapacity: TRAVERSED_EDGE_CAPACITY,
        output: {
          matchedEdges: builder.allocate('matched-edges', 'uint32', fixCount).view,
          snapDistances: builder.allocate('snap-distances', 'float32', fixCount).view,
          snappedPositions: builder.allocate('snapped-positions', 'float32x2', fixCount).view,
          traversedEdges: {
            edges: routeEdges.view,
            counts: routeCounts.view,
            status: {overflow: routeOverflow.view, candidateOverflow: routeCandidateOverflow.view}
          }
        }
      })
    );
    const volumes = builder.allocate('edge-volumes', 'uint32', network.slotCount);
    const truncated = builder.allocate('truncated', 'uint32', 1);
    builder.graph.add(
      new GPUMapMatchedEdgeVolumes({
        traversedEdges: {
          edges: routeEdges.view,
          counts: routeCounts.view,
          status: {overflow: routeOverflow.view, candidateOverflow: routeCandidateOverflow.view}
        },
        trackCount,
        traversedEdgeCapacity: TRAVERSED_EDGE_CAPACITY,
        edgeCount: network.slotCount,
        output: {edgeVolumes: volumes.view, status: {candidateOverflow: truncated.view}}
      })
    );
    return {
      outputs: {
        ...uploadRoads(builder, network),
        volumes,
        volumeLines: addSlotValues(builder, {
          id: 'volume',
          source: volumes,
          count: network.slotCount
        }),
        slotLines: uploadSlotLines(builder, network, 0.004),
        fixes: positions,
        truncated
      },
      update: current => {
        for (let fix = 0; fix < fixCount * 2; fix++) {
          inputs.fixes[fix] = inputs.truth[fix] + inputs.noise[fix] * current.noise;
        }
        positions.write(inputs.fixes.subarray(0, fixCount * 2));
        parameterBuffer.write(
          getGPUMapMatchingParameterValues({
            sigma: current.sigma,
            beta: 0.05,
            searchRadius: current.searchRadius
          })
        );
      }
    };
  },
  rebuildOn: ['trackCount'],
  // Probe: the directed street nearest the pointer, with its traversal count.
  interaction: {
    kind: 'probe',
    hint: 'hover a street',
    rest: [0.5, 0.5],
    resolve(pointer, {inputs, outputs, values, peek}) {
      const hit = findNearestSlot(inputs.slotSegments, pointer.x, pointer.y);
      const volumes = peek(outputs.volumes);
      const segment = inputs.slotSegments.subarray(hit.slot * 4, hit.slot * 4 + 4);
      const truncated = peek(outputs.truncated)?.[0];
      return {
        spotlight: {at: hit.point, radius: 0.12},
        glyphs: [
          {
            shape: 'polyline',
            points: [segment[0], segment[1], segment[2], segment[3]]
          },
          {shape: 'ring', at: hit.point, radius: 0.025}
        ],
        readout: `edge ${hit.slot} · ${volumes ? `${volumes[hit.slot]} ${volumes[hit.slot] === 1 ? 'traversal' : 'traversals'}` : '…'} · ${TRACK_COUNTS[values.trackCount]} tracks${
          truncated ? ' · incomplete traversals' : ''
        }`
      };
    }
  },
  draw: ({outputs}) => [
    getRoadContext(outputs),
    {
      layer: 'dots',
      context: true,
      positions: outputs.fixes,
      radius: 2,
      color: [0.9, 0.55, 0.3, 0.5]
    },
    getSlotValues(outputs.slotLines, outputs.volumeLines, {
      radius: 2.6,
      ramp: 'sequential',
      logarithmic: true,
      range: [0, 3.5]
    })
  ]
});
