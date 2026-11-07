// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUFlowAggregation: bin origin-destination records into grid zones and sum the flow between
// zone pairs. The grid shows each zone's outgoing weight; lines join zone centers of the heaviest
// flows.

import {GPUFlowAggregation} from '@luma-contributors/analysis/gpu-network';
import {defineOperationRun} from '../../define-run';
import {addEdgePairs} from './shared/display';
import {parameters} from './flow-aggregation.parameters';

const RECORD_COUNT = 1200;
const TOP_COUNT = 160;
const COLUMN_OPTIONS = [8, 12, 16];
const HUBS = [
  [0.22, 0.3],
  [0.7, 0.72],
  [0.5, 0.35],
  [0.85, 0.25]
] as const;

export default defineOperationRun({
  parameters,
  caption: 'Outgoing flow per zone (grid) and the heaviest zone-to-zone flows (lines).',
  createInputs: ({random, world}) => {
    // Each record starts at one hub and ends at another, with a spread around both.
    const hubs = Float32Array.from({length: RECORD_COUNT * 2}, () =>
      Math.floor(random() * HUBS.length)
    );
    const jitter = Float32Array.from(
      {length: RECORD_COUNT * 4},
      () => (random() + random() + random() - 1.5) * 0.12
    );
    const weights = Float32Array.from({length: RECORD_COUNT}, () => 1 + Math.floor(random() * 4));
    return {
      world,
      hubs,
      jitter,
      weights,
      origins: new Float32Array(RECORD_COUNT * 2),
      destinations: new Float32Array(RECORD_COUNT * 2)
    };
  },
  build(builder, inputs, values) {
    const columns = COLUMN_OPTIONS[values.gridSize];
    const rows = Math.max(2, Math.round((columns * 9) / 16));
    const zoneCount = columns * rows;
    const {world} = inputs;
    const origins = builder.allocate('origins', 'float32x2', RECORD_COUNT);
    const destinations = builder.allocate('destinations', 'float32x2', RECORD_COUNT);
    const ids = builder.allocate('ids', 'uint32', TOP_COUNT);
    const count = builder.allocate('count', 'uint32', 1);
    const overflow = builder.allocate('overflow', 'uint32', 1);
    const pairOverflow = builder.allocate('pair-overflow', 'uint32', 1);
    const originZones = builder.allocate('flow-origin-zones', 'uint32', TOP_COUNT);
    const destinationZones = builder.allocate('flow-destination-zones', 'uint32', TOP_COUNT);
    const flowWeights = builder.allocate('flow-weights', 'float32', TOP_COUNT);
    const zoneOutWeights = builder.allocate('zone-out-weights', 'float32', zoneCount);
    const zoneInWeights = builder.allocate('zone-in-weights', 'float32', zoneCount);
    builder.graph.add(
      new GPUFlowAggregation({
        zones: {kind: 'grid', bounds: [0, 0, world.width, 1], gridSize: [columns, rows]},
        origins: origins.view,
        destinations: destinations.view,
        weights: builder.upload('weights', 'float32', inputs.weights).view,
        excludeSelfFlows: values.excludeSelfFlows === 1,
        pairCapacity: 4096,
        output: {
          ids: ids.view,
          status: {count: count.view, overflow: overflow.view, pairOverflow: pairOverflow.view},
          flowOriginZoneIds: originZones.view,
          flowDestinationZoneIds: destinationZones.view,
          flowWeights: flowWeights.view,
          zoneOutWeights: zoneOutWeights.view,
          zoneInWeights: zoneInWeights.view
        }
      })
    );
    // Zone centers, drawn with row 0 at the top like the grid; one spare node stays offscreen.
    const centers = new Float32Array((zoneCount + 1) * 2).fill(-100);
    for (let zone = 0; zone < zoneCount; zone++) {
      centers[zone * 2] = ((zone % columns) + 0.5) * (world.width / columns);
      centers[zone * 2 + 1] = 1 - (Math.floor(zone / columns) + 0.5) / rows;
    }
    return {
      outputs: {
        zoneOutWeights,
        centers: builder.upload('centers', 'float32x2', centers),
        flows: addEdgePairs(builder, {
          id: 'flows',
          from: originZones,
          to: destinationZones,
          fallbackNode: zoneCount
        }),
        originZones,
        destinationZones
      },
      update: current => {
        for (let record = 0; record < RECORD_COUNT; record++) {
          const origin = HUBS[inputs.hubs[record * 2]];
          const destination = HUBS[inputs.hubs[record * 2 + 1]];
          // The grid's row 0 is the lowest y; world y is flipped so row 0 draws at the top.
          inputs.origins[record * 2] =
            (origin[0] + inputs.jitter[record * 4] * current.spread) * world.width;
          inputs.origins[record * 2 + 1] =
            1 - (origin[1] + inputs.jitter[record * 4 + 1] * current.spread);
          inputs.destinations[record * 2] =
            (destination[0] + inputs.jitter[record * 4 + 2] * current.spread) * world.width;
          inputs.destinations[record * 2 + 1] =
            1 - (destination[1] + inputs.jitter[record * 4 + 3] * current.spread);
        }
        origins.write(inputs.origins);
        destinations.write(inputs.destinations);
        originZones.write(new Uint32Array(TOP_COUNT).fill(0xffffffff));
        destinationZones.write(new Uint32Array(TOP_COUNT).fill(0xffffffff));
      }
    };
  },
  rebuildOn: ['gridSize', 'excludeSelfFlows'],
  draw: ({outputs, values}) => {
    const columns = COLUMN_OPTIONS[values.gridSize];
    return [
      {
        layer: 'raster',
        values: outputs.zoneOutWeights,
        columns,
        rows: Math.max(2, Math.round((columns * 9) / 16)),
        ramp: 'sequential'
      },
      {
        layer: 'segments',
        nodes: outputs.centers,
        edges: outputs.flows,
        radius: 1.4,
        color: [1, 1, 1, 0.5]
      }
    ];
  }
});
