// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// addFleetDwellZoneEventsRecipe: enter and exit events of every track against three zones, with
// dwell per (track, zone) rolled up per zone. Dots are the crossings; zones are filled by the
// total dwell time of the visiting tracks.

import {addFleetDwellZoneEventsRecipe} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import type {RunGlyph} from '../../interaction';
import {createFleet} from './lib/fleet';
import {formatZoneEventReadout} from './lib/zone-event-readout';
import {
  chooseZoneCenters,
  createZoneRectangles,
  dragNearestZone,
  getZoneHalfSize,
  type ZoneCenter,
  createZoneTriangles,
  ZONE_COUNT,
  ZONE_FEATURE_OFFSETS,
  ZONE_RING_OFFSETS,
  ZONE_TRIANGLE_COUNT
} from './lib/zones';
import {parameters} from './fleet-dwell-zone-events-recipe.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Crossings per zone (small dots; exits ringed) and the dwell they add up to.',
  createInputs: ({world}) => {
    const fleet = createFleet(world, {trackCount: 16, minimumFixes: 20, maximumFixes: 30});
    const homes = chooseZoneCenters(fleet);
    return {fleet, world, homes, centers: homes as readonly ZoneCenter[]};
  },
  build(builder, inputs, values) {
    const {fleet} = inputs;
    const capacity = fleet.trackCount * values.trackEventCapacity;
    const zonePositions = builder.allocate('zone-positions', 'float32x2', ZONE_COUNT * 4);
    const zoneOutline = builder.allocate('zone-outline', 'float32x4', ZONE_COUNT * 4);
    const eventPositions = builder.allocate('event-positions', 'float32x2', capacity);
    const eventZones = builder.allocate('event-zones', 'uint32', capacity);
    const eventTypes = builder.allocate('event-types', 'uint32', capacity);
    const zoneTotals = builder.allocate('zone-totals', 'float32', ZONE_COUNT);
    const eventCount = builder.allocate('event-count', 'uint32', 1);
    const eventOverflow = builder.allocate('event-overflow', 'uint32', 1);
    const eventCandidateOverflow = builder.allocate('event-candidate-overflow', 'uint32', 1);
    const scratchCandidateOverflow = builder.allocate('scratch-candidate-overflow', 'uint32', 1);
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
      output: {
        events: {
          status: {
            count: eventCount.view,
            overflow: eventOverflow.view,
            candidateOverflow: eventCandidateOverflow.view
          },
          eventPositions: eventPositions.view,
          eventZones: eventZones.view,
          eventTypes: eventTypes.view
        },
        table: {sumValues: zoneTotals.view},
        status: {candidateOverflow: scratchCandidateOverflow.view}
      }
    });
    return {
      outputs: {
        eventPositions,
        eventZones,
        eventTypes,
        zoneTotals,
        eventCount,
        eventOverflow,
        eventCandidateOverflow,
        scratchCandidateOverflow,
        zonePositions,
        zoneOutline,
        indices: builder.upload('zone-indices', 'uint32', triangles.indices),
        owners: builder.upload('zone-owners', 'uint32', triangles.owners),
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
  // The pointer carries the nearest zone: the crossings and the dwell fill re-form around it.
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
      const totals = peek(outputs.zoneTotals);
      const eventCount = peek(outputs.eventCount);
      const glyphs: RunGlyph[] = [
        {
          shape: 'box',
          minimum: [centerX - halfWidth, centerY - halfHeight],
          maximum: [centerX + halfWidth, centerY + halfHeight]
        }
      ];
      let crossingCount: number | undefined;
      if (
        eventZones &&
        eventCount &&
        Number.isInteger(eventCount[0]) &&
        eventCount[0] >= 0 &&
        eventCount[0] <= eventZones.length &&
        eventCount[0] !== 0xffffffff
      ) {
        crossingCount = 0;
        for (let event = 0; event < eventCount[0]; event++) {
          if (eventZones[event] === 0xffffffff) {
            crossingCount = undefined;
            break;
          }
          if (eventZones[event] !== zone) continue;
          crossingCount++;
          if (
            eventPositions &&
            eventPositions.length >= 2 * event + 2 &&
            Number.isFinite(eventPositions[2 * event]) &&
            Number.isFinite(eventPositions[2 * event + 1]) &&
            glyphs.length < 30
          ) {
            glyphs.push({
              shape: 'dot',
              at: [eventPositions[event * 2], eventPositions[event * 2 + 1]]
            });
          }
        }
      }
      const dwell = totals && Number.isFinite(totals[zone]) ? Math.round(totals[zone]) : null;
      return {
        rerun: true,
        glyphs,
        readout: formatZoneEventReadout(
          'ABC'[zone],
          crossingCount,
          {
            eventOverflow: peek(outputs.eventOverflow)?.[0],
            eventCandidateOverflow: peek(outputs.eventCandidateOverflow)?.[0],
            scratchCandidateOverflow: peek(outputs.scratchCandidateOverflow)?.[0]
          },
          dwell
        )
      };
    }
  },
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
    // Every crossing is a small dot by zone; exits get a larger bright ring (enter is type 0, and
    // fade hides it).
    {layer: 'dots', positions: outputs.eventPositions, values: outputs.eventZones, radius: 6},
    {
      layer: 'dots',
      positions: outputs.eventPositions,
      values: outputs.eventTypes,
      ramp: 'sequential',
      range: [0, 1],
      fade: true,
      radius: 12,
      alpha: 0.7
    }
  ]
});
