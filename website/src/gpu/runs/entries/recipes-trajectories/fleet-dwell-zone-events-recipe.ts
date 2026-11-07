// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// addFleetDwellZoneEventsRecipe: enter and exit events of every track against three zones, with
// dwell per (track, zone) rolled up per zone. Dots are the crossings; zones are filled by the
// total dwell time of the visiting tracks.

import {addFleetDwellZoneEventsRecipe} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createFleet} from './lib/fleet';
import {
  createZoneRectangles,
  createZoneTriangles,
  ZONE_COUNT,
  ZONE_FEATURE_OFFSETS,
  ZONE_RING_OFFSETS,
  ZONE_TRIANGLE_COUNT
} from './lib/zones';
import {parameters} from './fleet-dwell-zone-events-recipe.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Crossings per zone and the dwell they add up to.',
  createInputs: ({random, world}) => ({
    fleet: createFleet(random, world, {trackCount: 16, minimumFixes: 20, maximumFixes: 30}),
    world
  }),
  build(builder, {fleet}, values) {
    const capacity = fleet.trackCount * values.trackEventCapacity;
    const zonePositions = builder.allocate('zone-positions', 'float32x2', ZONE_COUNT * 4);
    const zoneOutline = builder.allocate('zone-outline', 'float32x4', ZONE_COUNT * 4);
    const eventPositions = builder.allocate('event-positions', 'float32x2', capacity);
    const eventZones = builder.allocate('event-zones', 'uint32', capacity);
    const zoneTotals = builder.allocate('zone-totals', 'float32', ZONE_COUNT);
    const triangles = createZoneTriangles();
    addFleetDwellZoneEventsRecipe(builder.graph, {
      trajectories: {
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
      outputs: {
        events: {eventPositions: eventPositions.view, eventZones: eventZones.view},
        table: {sumValues: zoneTotals.view}
      }
    });
    return {
      outputs: {
        eventPositions,
        eventZones,
        zoneTotals,
        zonePositions,
        zoneOutline,
        indices: builder.upload('zone-indices', 'uint32', triangles.indices),
        owners: builder.upload('zone-owners', 'uint32', triangles.owners),
        segments: builder.upload('segments', 'float32x4', fleet.segments)
      },
      update: current => {
        const zones = createZoneRectangles({width: fleet.worldWidth, height: 1}, current.zoneSize);
        zonePositions.write(zones.positions);
        zoneOutline.write(zones.outline);
      }
    };
  },
  rebuildOn: ['trackEventCapacity'],
  draw: ({inputs, outputs}) => [
    {
      layer: 'polygons',
      positions: outputs.zonePositions,
      indices: outputs.indices,
      owners: outputs.owners,
      values: outputs.zoneTotals,
      triangleCount: ZONE_TRIANGLE_COUNT,
      alpha: 0.7
    },
    {layer: 'lines', context: true, segments: outputs.segments, count: inputs.fleet.segmentCount},
    {
      layer: 'lines',
      context: true,
      segments: outputs.zoneOutline,
      count: ZONE_COUNT * 4,
      radius: 2
    },
    {layer: 'dots', positions: outputs.eventPositions, values: outputs.eventZones, radius: 10}
  ]
});
