// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUNetworkTravelTimePercentiles: one earliest-arrival routing per departure of a window, then
// per-intersection percentiles of the travel time. Streets are colored by the chosen percentile of
// the time to reach them from the origin. Congestion builds with the clock, so a wider window makes
// the tail percentiles slower than the median.

import {GPUNetworkTravelTimePercentiles} from '@luma-contributors/analysis/gpu-network';
import {defineOperationRun, type RunBuffer} from '../../define-run';
import {addTableColumn} from './_lib/network-kernels';
import {
  createNodeIndex,
  createRoadNetwork,
  createRoads,
  getEndpointMarkers,
  getRoadContext,
  getRoadValues,
  uploadRoads
} from '../network-routing-access/shared/roads';
import {parameters} from './network-travel-time-percentiles.parameters';

const PERCENTILES = [5, 50, 80, 95];
const LABELS = ['p5', 'p50', 'p80', 'p95', 'p95 - p50'];
/** Street lengths are world units: one unit is about 3 km, driven at 30 km/h. */
const MINUTES_PER_UNIT = 6;
const BARS = [0.04, 0.06, 0.3, 0.3] as const;
const formatMinutes = (units: number) => `${(units * MINUTES_PER_UNIT).toFixed(1)} min`;

type Inputs = ReturnType<typeof createInputs>;
type Outputs = ReturnType<typeof uploadRoads> & {
  travelTimes: RunBuffer;
  reachedCounts: RunBuffer;
  column: RunBuffer;
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
  const network = createRoadNetwork(createRoads(random, world, 8), 0);
  const transitions = new Uint32Array(network.slotCount * 2);
  for (let slot = 0; slot < network.slotCount; slot++) {
    transitions[slot * 2] = network.slotSources[slot];
    transitions[slot * 2 + 1] = network.neighbors[slot];
  }
  const origin = network.findNearestNode(world.width * 0.22, 0.5);
  return {
    network,
    world,
    transitions,
    nodeIndex: createNodeIndex(network),
    defaultOrigin: origin,
    destination: network.findNearestNode(world.width * 0.8, 0.2),
    origin,
    costs: new Float32Array(network.slotCount * 2)
  };
}

export default defineOperationRun<Inputs, Outputs, {bars: RunBuffer; limit: number}>({
  parameters,
  caption:
    'Travel time from one intersection over a window of departures, as a percentile. Street lengths and a congestion model that builds with the clock stand in for travel times on a 128-intersection San Francisco TIGER/Line extract.',
  createInputs,
  build(builder, inputs, values) {
    const {network} = inputs;
    const vertexCount = network.nodeCount;
    const initialState = builder.allocate('initial-state', 'uint32', 1);
    const window = builder.allocate('window', 'float32', 2);
    const costs = builder.allocate('costs', 'float32x2', network.slotCount);
    const travelTimes = builder.allocate(
      'travel-times',
      'float32',
      vertexCount * PERCENTILES.length
    );
    const reachedCounts = builder.allocate('reached-counts', 'uint32', vertexCount);
    builder.graph.add(
      new GPUNetworkTravelTimePercentiles({
        stateVertices: builder.upload(
          'state-vertices',
          'uint32',
          Uint32Array.from({length: vertexCount}, (_, index) => index)
        ).view,
        transitions: builder.upload('transitions', 'uint32x2', inputs.transitions).view,
        costModel: 'affine-duration',
        costs: costs.view,
        vertexCount,
        initialState: initialState.view,
        departureCount: values.departureCount,
        percentiles: PERCENTILES,
        window: window.view,
        output: {travelTimes: travelTimes.view, reachedCounts: reachedCounts.view}
      })
    );
    const selection = builder.allocate('selection', 'uint32', 1);
    const originMarker = builder.allocate('origin-marker', 'float32x2', 1);
    // Congestion is strongest in the middle of the picture, so a later departure meets a slower center.
    for (let slot = 0; slot < network.slotCount; slot++) {
      const source = network.slotSources[slot];
      const target = network.neighbors[slot];
      const middleX = (network.nodes[source * 2] + network.nodes[target * 2]) / 2;
      const middleY = (network.nodes[source * 2 + 1] + network.nodes[target * 2 + 1]) / 2;
      const congestion =
        0.1 +
        3 * Math.exp(-((Math.hypot(middleX - inputs.world.width / 2, middleY - 0.5) / 0.3) ** 2));
      inputs.costs[slot * 2] = network.lengths[slot];
      inputs.costs[slot * 2 + 1] = network.lengths[slot] * congestion;
    }
    costs.write(inputs.costs);
    return {
      outputs: {
        ...uploadRoads(builder, network),
        travelTimes,
        reachedCounts,
        column: addTableColumn(builder, {
          id: 'percentile',
          table: travelTimes,
          stride: PERCENTILES.length,
          vertexCount,
          selection
        }),
        originMarker,
        destinationMarker: builder.upload(
          'destination-marker',
          'float32x2',
          network.nodes.slice(inputs.destination * 2, inputs.destination * 2 + 2)
        )
      },
      update: current => {
        initialState.write(Uint32Array.of(inputs.origin));
        window.write(Float32Array.of(current.firstDeparture, current.step));
        selection.write(Uint32Array.of(current.percentile));
        originMarker.write(network.nodes.slice(inputs.origin * 2, inputs.origin * 2 + 2));
      }
    };
  },
  rebuildOn: ['departureCount'],
  async afterRun({inputs, outputs, upload}) {
    // The four percentiles of the trip to the marked destination, drawn as bars.
    const bytes = await outputs.travelTimes.buffer.readAsync();
    const table = new Float32Array(
      bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)
    );
    const row = inputs.destination * PERCENTILES.length;
    const bars = Float32Array.from(PERCENTILES, (_, index) =>
      Number.isFinite(table[row + index]) ? table[row + index] : 0
    );
    return {bars: upload('bars', 'float32', bars), limit: Math.max(...bars, 0.05) * 1.15};
  },
  // Place: the origin is the intersection under the pointer.
  interaction: {
    kind: 'place',
    hint: 'move the origin',
    rest: ({inputs, toUnit}) =>
      toUnit([
        inputs.network.nodes[inputs.defaultOrigin * 2],
        inputs.network.nodes[inputs.defaultOrigin * 2 + 1]
      ]),
    resolve(pointer, {inputs, outputs, values, peek}) {
      const node = inputs.nodeIndex.nearest(pointer.x, pointer.y);
      const changed = node !== inputs.origin;
      inputs.origin = node;
      const times = peek(outputs.travelTimes);
      const stride = PERCENTILES.length;
      const world = inputs.world;
      let readout = `${LABELS[values.percentile]} drawn`;
      if (times) {
        const [low, median, high, tail] = [0, 1, 2, 3].map(
          index => times[inputs.destination * stride + index]
        );
        readout = `to marked destination: p5 ${formatMinutes(low)} · p50 ${formatMinutes(median)} · p95 ${formatMinutes(tail)} (p80 ${formatMinutes(high)}) · ${LABELS[values.percentile]} drawn`;
      }
      const lastDeparture = values.firstDeparture + (values.departureCount - 1) * values.step;
      const barWidth = ((BARS[2] - BARS[0]) * world.width) / PERCENTILES.length;
      const chosen = Math.min(values.percentile, PERCENTILES.length - 1);
      const ruleX = BARS[0] * world.width + (chosen + 0.5) * barWidth;
      return {
        rerun: changed,
        spotlight: null,
        glyphs: [
          {
            shape: 'crosshair',
            at: [inputs.network.nodes[node * 2], inputs.network.nodes[node * 2 + 1]],
            size: 9
          },
          {
            shape: 'box',
            minimum: [BARS[0] * world.width, BARS[1]],
            maximum: [BARS[2] * world.width, BARS[3]]
          },
          {shape: 'line', from: [ruleX, BARS[1]], to: [ruleX, BARS[3]], dashed: true}
        ],
        readout: `${readout} · departures ${formatMinutes(values.firstDeparture)} to ${formatMinutes(lastDeparture)}`
      };
    }
  },
  draw: ({outputs, theme, values, extra}) => [
    getRoadContext(outputs),
    getRoadValues(outputs, outputs.column, {
      radius: 2.4,
      range: values.percentile === 4 ? 'auto' : [0, 4]
    }),
    ...getEndpointMarkers(outputs.originMarker, theme, {colorIndex: 4}),
    ...getEndpointMarkers(outputs.destinationMarker, theme, {colorIndex: 2}),
    {
      layer: 'chart',
      kind: 'bars',
      values: extra.bars,
      range: [0, extra.limit],
      rectangle: BARS,
      color: [...theme.categories[0], 0.95]
    }
  ]
});
