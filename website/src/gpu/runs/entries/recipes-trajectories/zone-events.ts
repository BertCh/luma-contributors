// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUZoneEvents: where every track crosses a zone boundary. Dots are the interpolated crossing
// positions colored by the zone crossed; trackEventCapacity truncates each track's event list.

import {GPUZoneEvents} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import type {RunGlyph} from '../../interaction';
import {createFleet} from './lib/fleet';
import {
  chooseZoneCenters,
  createZoneRectangles,
  dragNearestZone,
  getZoneHalfSize,
  type ZoneCenter,
  ZONE_COUNT,
  ZONE_FEATURE_OFFSETS,
  ZONE_RING_OFFSETS
} from './lib/zones';
import {parameters} from './zone-events.parameters';

const EVENT_CAPACITY = 256;

export default defineOperationRun({
  parameters,
  caption: 'Enter and exit crossings of every track against three zones.',
  createInputs: ({world}) => {
    const fleet = createFleet(world, {trackCount: 16, minimumFixes: 20, maximumFixes: 30});
    const homes = chooseZoneCenters(fleet);
    return {fleet, world, homes, centers: homes as readonly ZoneCenter[]};
  },
  build(builder, inputs, values) {
    const {fleet} = inputs;
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
        count,
        eventZones,
        eventPositions,
        segments: builder.upload('segments', 'float32x4', fleet.segments)
      },
      update: current => {
        const zones = createZoneRectangles(inputs.world, current.zoneSize, inputs.centers);
        zonePositions.write(zones.positions);
        zoneOutline.write(zones.outline);
      }
    };
  },
  rebuildOn: ['trackEventCapacity'],
  // The pointer carries the nearest zone: its boundary crossings re-form on the new boundary.
  interaction: {
    kind: 'drag',
    hint: 'move a zone',
    rest: ({toUnit, inputs}) => toUnit(inputs.homes[0]),
    resolve(pointer, {inputs, outputs, values, peek}) {
      const {centers, zone} = dragNearestZone(inputs.homes, pointer);
      inputs.centers = centers;
      const [halfWidth, halfHeight] = getZoneHalfSize(inputs.world, values.zoneSize);
      const [centerX, centerY] = centers[zone];
      const eventPositions = peek(outputs.eventPositions);
      const eventZones = peek(outputs.eventZones);
      const eventCount = peek(outputs.count);
      const glyphs: RunGlyph[] = [
        {
          shape: 'box',
          minimum: [centerX - halfWidth, centerY - halfHeight],
          maximum: [centerX + halfWidth, centerY + halfHeight]
        }
      ];
      let crossingCount = 0;
      if (eventPositions && eventZones && eventCount) {
        for (let event = 0; event < Math.min(eventCount[0], eventZones.length); event++) {
          if (eventZones[event] === zone) crossingCount++;
        }
      }
      return {
        rerun: true,
        glyphs,
        readout: `zone ${'ABC'[zone]} · ${crossingCount} crossings`
      };
    }
  },
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
