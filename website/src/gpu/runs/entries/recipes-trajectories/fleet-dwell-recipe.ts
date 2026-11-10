// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// addFleetDwellRecipe: stops detected on every track, joined to zones and rolled up. Dots are the
// stops (by dwell time); each zone is filled by the total dwell time of the stops inside it.

import {
  addFleetDwellRecipe,
  getGPUTrajectoryMetricsParameterValues
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import type {RunGlyph} from '../../interaction';
import {createFleet} from './lib/fleet';
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
import {parameters} from './fleet-dwell-recipe.parameters';

const STOP_CAPACITY = 128;

export default defineOperationRun({
  parameters,
  caption: 'Stops per zone: total dwell time fills each zone.',
  createInputs: ({world}) => {
    const fleet = createFleet(world, {trackCount: 18, minimumFixes: 26, maximumFixes: 40});
    const homes = chooseZoneCenters(fleet);
    return {fleet, world, homes, centers: homes as readonly ZoneCenter[]};
  },
  build(builder, inputs) {
    const {fleet} = inputs;
    const parameterBuffer = builder.allocate('parameters', 'float32', 4);
    const zonePositions = builder.allocate('zone-positions', 'float32x2', ZONE_COUNT * 4);
    const zoneOutline = builder.allocate('zone-outline', 'float32x4', ZONE_COUNT * 4);
    const centroids = builder.allocate('centroids', 'float32x2', STOP_CAPACITY);
    const durations = builder.allocate('durations', 'float32', STOP_CAPACITY);
    const zoneTotals = builder.allocate('zone-totals', 'float32', ZONE_COUNT);
    const triangles = createZoneTriangles();
    addFleetDwellRecipe(builder.graph, {
      trajectories: {
        kind: 'trajectories',
        positions: builder.upload('positions', 'float32x2', fleet.positions).view,
        timestamps: builder.upload('timestamps', 'float32', fleet.timestamps).view,
        trackOffsets: builder.upload('offsets', 'uint32', fleet.trackOffsets).view
      },
      parameters: parameterBuffer.view,
      stopCapacity: STOP_CAPACITY,
      zones: {
        kind: 'polygons',
        positions: zonePositions.view,
        featureOffsets: builder.upload('zone-features', 'uint32', ZONE_FEATURE_OFFSETS).view,
        polygonOffsets: builder.upload('zone-polygons', 'uint32', ZONE_FEATURE_OFFSETS).view,
        ringOffsets: builder.upload('zone-rings', 'uint32', ZONE_RING_OFFSETS).view,
        candidateCapacity: 4096
      },
      outputs: {
        stops: {centroids: centroids.view, durations: durations.view},
        table: {sumValues: zoneTotals.view}
      }
    });
    return {
      outputs: {
        centroids,
        durations,
        zoneTotals,
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
        parameterBuffer.write(
          getGPUTrajectoryMetricsParameterValues({
            stopSpeedThreshold: current.stopSpeedThreshold,
            stopMinimumDuration: current.stopMinimumDuration
          })
        );
      }
    };
  },
  // The pointer carries the nearest zone: stops that fall inside it add to its dwell fill.
  interaction: {
    kind: 'drag',
    hint: 'move a zone',
    rest: ({toUnit, inputs}) => toUnit(inputs.homes[0]),
    resolve(pointer, {inputs, outputs, values, peek}) {
      const {centers, zone} = dragNearestZone(inputs.homes, pointer);
      inputs.centers = centers;
      const [halfWidth, halfHeight] = getZoneHalfSize(inputs.world, values.zoneSize);
      const [centerX, centerY] = centers[zone];
      const totals = peek(outputs.zoneTotals);
      const centroids = peek(outputs.centroids);
      const durations = peek(outputs.durations);
      const glyphs: RunGlyph[] = [
        {
          shape: 'box',
          minimum: [centerX - halfWidth, centerY - halfHeight],
          maximum: [centerX + halfWidth, centerY + halfHeight]
        }
      ];
      let stopCount = 0;
      if (centroids && durations) {
        for (let stop = 0; stop < durations.length; stop++) {
          const x = centroids[stop * 2];
          const y = centroids[stop * 2 + 1];
          if (
            durations[stop] > 0 &&
            Math.abs(x - centerX) <= halfWidth &&
            Math.abs(y - centerY) <= halfHeight
          ) {
            stopCount++;
            glyphs.push({shape: 'dot', at: [x, y]});
          }
        }
      }
      const dwell = totals ? Math.round(totals[zone]) : 0;
      return {
        rerun: true,
        glyphs,
        readout: `zone ${'ABC'[zone]} · ${stopCount} stops · dwell ${dwell} s`
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
      alpha: 0.75
    },
    {layer: 'lines', context: true, segments: outputs.segments, count: inputs.fleet.segmentCount},
    {
      layer: 'lines',
      context: true,
      segments: outputs.zoneOutline,
      count: ZONE_COUNT * 4,
      radius: 2
    },
    {
      layer: 'dots',
      positions: outputs.centroids,
      values: outputs.durations,
      radius: 8,
      fade: true,
      range: [0, 150]
    }
  ]
});
