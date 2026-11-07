// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// addFleetDwellRecipe: stops detected on every track, joined to zones and rolled up. Dots are the
// stops (by dwell time); each zone is filled by the total dwell time of the stops inside it.

import {
  addFleetDwellRecipe,
  getGPUTrajectoryMetricsParameterValues
} from '@luma-contributors/analysis/gpu-spatial-analysis';
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
import {parameters} from './fleet-dwell-recipe.parameters';

const STOP_CAPACITY = 128;

export default defineOperationRun({
  parameters,
  caption: 'Stops per zone: total dwell time fills each zone.',
  createInputs: ({random, world}) => ({
    fleet: createFleet(random, world, {
      trackCount: 18,
      stops: true,
      minimumFixes: 26,
      maximumFixes: 40
    }),
    world
  }),
  build(builder, {fleet}) {
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
        const zones = createZoneRectangles({width: fleet.worldWidth, height: 1}, current.zoneSize);
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
