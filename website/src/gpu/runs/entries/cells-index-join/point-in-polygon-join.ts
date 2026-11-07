// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUPointInPolygonJoin writes, for every point, the id of the zone that contains it. Points are
// colored by that zone id; points in no zone keep the muted input color.

import {GPUPointInPolygonJoin} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createRandomPoints, createZones, scaleZones, uploadZones} from './support/join';
import {parameters} from './point-in-polygon-join.parameters';

const POINT_COUNT = 900;

export default defineOperationRun({
  parameters,
  caption: 'Each point colored by the zone that contains it.',
  createInputs: ({random, world}) => ({
    zones: createZones(random, world),
    points: createRandomPoints(random, world, POINT_COUNT)
  }),
  build(builder, inputs) {
    const points = builder.allocate('points', 'float32x2', POINT_COUNT);
    const zones = uploadZones(builder, inputs.zones);
    const pointFeatureIds = builder.allocate('point-feature-ids', 'uint32', POINT_COUNT);
    const featureCounts = builder.allocate('feature-counts', 'uint32', inputs.zones.featureCount);
    const overflow = builder.allocate('overflow', 'uint32', 1);
    builder.graph.add(
      new GPUPointInPolygonJoin({
        points: {kind: 'points', positions: points.view},
        polygons: zones.geometry,
        candidateCapacity: 1 << 14,
        output: {
          pointFeatureIds: pointFeatureIds.view,
          featureCounts: featureCounts.view,
          status: {overflow: overflow.view}
        }
      })
    );
    return {
      outputs: {points, pointFeatureIds, zones},
      update: current => {
        points.write(
          Float32Array.from(inputs.points, (value, index) =>
            index % 2 === 0 ? value + current.shift : value
          )
        );
        const scaled = scaleZones(inputs.zones, current.size);
        zones.positions.write(scaled.positions);
        zones.outline.write(scaled.outline);
      }
    };
  },
  draw: ({outputs}) => [
    {layer: 'dots', context: true, positions: outputs.points, radius: 3},
    {layer: 'dots', positions: outputs.points, values: outputs.pointFeatureIds, radius: 3.5},
    {
      layer: 'lines',
      context: true,
      segments: outputs.zones.outline,
      radius: 1.4,
      color: [1, 1, 1, 0.8]
    }
  ]
});
