// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUPointCloudNeighborhoods on a lidar-like cloud (top view). Each point is colored by how many
// neighbors the GPU found within `radius` (counts); the table keeps at most `capacity` of them.

import {GPUPointCloudNeighborhoods} from '@luma-contributors/analysis/pointcloud-analysis';
import {defineOperationRun} from '../../define-run';
import {addIntegerToFloat} from './_lib/convert';
import {createTerrainCloud, getJitteredPositions} from './_lib/helpers';
import {parameters} from './pointcloud-neighborhoods.parameters';

const POINT_COUNT = 3000;

export default defineOperationRun({
  parameters,
  caption: 'Neighbors within the radius, per point.',
  createInputs: ({random, world}) => createTerrainCloud(random, world.width, POINT_COUNT),
  build(builder, cloud, values) {
    const positions = builder.allocate('positions', 'float32x3', POINT_COUNT);
    const counts = builder.allocate('counts', 'uint32', POINT_COUNT);
    builder.graph.add(
      new GPUPointCloudNeighborhoods({
        positions: positions.view,
        radius: values.radius,
        capacity: values.capacity,
        indices: builder.allocate('indices', 'uint32', POINT_COUNT * values.capacity).view,
        squaredDistances: builder.allocate(
          'squared-distances',
          'float32',
          POINT_COUNT * values.capacity
        ).view,
        counts: counts.view
      })
    );
    return {
      outputs: {
        footprint: builder.upload('footprint', 'float32x2', cloud.footprint),
        counts: addIntegerToFloat(builder, 'neighbor-counts', counts)
      },
      update: current => positions.write(getJitteredPositions(cloud, current.noise))
    };
  },
  rebuildOn: ['radius', 'capacity'],
  draw: ({outputs, values}) => [
    {
      layer: 'dots',
      positions: outputs.footprint,
      values: outputs.counts,
      range: [0, values.capacity * 0.6],
      radius: 2.6
    }
  ]
});
