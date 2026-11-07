// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUZoneEvents: where every track crosses a zone boundary. Dots are the interpolated crossing
// positions colored by the zone crossed; trackEventCapacity truncates each track's event list.

import {GPUZoneEvents} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createFleet} from './lib/fleet';
import {
  createZoneRectangles,
  ZONE_COUNT,
  ZONE_FEATURE_OFFSETS,
  ZONE_RING_OFFSETS
} from './lib/zones';
import {parameters} from './zone-events.parameters';

const EVENT_CAPACITY = 256;

export default defineOperationRun({
  parameters,
  caption: 'Enter and exit crossings of every track against three zones.',
  createInputs: ({random, world}) => ({
    fleet: createFleet(random, world, {trackCount: 16, minimumFixes: 20, maximumFixes: 30}),
    world
  }),
  build(builder, {fleet}, values) {
    const zonePositions = builder.allocate('zone-positions', 'float32x2', ZONE_COUNT * 4);
    const zoneOutline = builder.allocate('zone-outline', 'float32x4', ZONE_COUNT * 4);
    const ids = builder.allocate('event-ids', 'uint32', EVENT_CAPACITY);
    const count = builder.allocate('event-count', 'uint32', 1);
    const overflow = builder.allocate('event-overflow', 'uint32', 1);
    const eventZones = builder.allocate('event-zones', 'uint32', EVENT_CAPACITY);
    const eventPositions = builder.allocate('event-positions', 'float32x2', EVENT_CAPACITY);
    builder.graph.add(
      new GPUZoneEvents({
        tracks: {
          kind: 'trajectories',
          positions: builder.upload('positions', 'float32x2', fleet.positions).view,
          timestamps: builder.upload('timestamps', 'float32', fleet.timestamps).view,
          trackOffsets: builder.upload('offsets', 'uint32', fleet.trackOffsets).view
        },
        zones: {
          kind: 'polygons',
          positions: zonePositions.view,
          featureOffsets: builder.upload('zone-features', 'uint32', ZONE_FEATURE_OFFSETS).view,
          polygonOffsets: builder.upload('zone-polygons', 'uint32', ZONE_FEATURE_OFFSETS).view,
          ringOffsets: builder.upload('zone-rings', 'uint32', ZONE_RING_OFFSETS).view
        },
        candidateCapacity: 8192,
        trackEventCapacity: values.trackEventCapacity,
        output: {
          events: {
            ids: ids.view,
            status: {count: count.view, overflow: overflow.view},
            eventZones: eventZones.view,
            eventPositions: eventPositions.view
          }
        }
      })
    );
    return {
      outputs: {
        zoneOutline,
        eventZones,
        eventPositions,
        segments: builder.upload('segments', 'float32x4', fleet.segments)
      },
      update: current => {
        const zones = createZoneRectangles(inputsWorld(fleet.worldWidth), current.zoneSize);
        zonePositions.write(zones.positions);
        zoneOutline.write(zones.outline);
      }
    };
  },
  rebuildOn: ['trackEventCapacity'],
  draw: ({inputs, outputs}) => [
    {layer: 'lines', context: true, segments: outputs.segments, count: inputs.fleet.segmentCount},
    {
      layer: 'lines',
      context: true,
      segments: outputs.zoneOutline,
      count: ZONE_COUNT * 4,
      radius: 2,
      alpha: 1
    },
    {layer: 'dots', positions: outputs.eventPositions, values: outputs.eventZones, radius: 10}
  ]
});

function inputsWorld(width: number): {width: number; height: number} {
  return {width, height: 1};
}
