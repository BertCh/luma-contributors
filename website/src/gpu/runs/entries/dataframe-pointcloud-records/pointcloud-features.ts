// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUPointCloudFeatures after GPUPointCloudNeighborhoods: the eigenvalues of each point's local
// covariance become shape descriptors. Ground is planar (bright with `planarity`); tree crowns
// scatter in all directions (bright with `scattering`).

import {
  GPUPointCloudFeatures,
  GPUPointCloudNeighborhoods
} from '@luma-contributors/analysis/pointcloud-analysis';
import {defineOperationRun} from '../../define-run';
import {addFloatCopy} from './_lib/convert';
import {createTerrainCloud, getJitteredPositions} from './_lib/helpers';
import {parameters} from './pointcloud-features.parameters';

const POINT_COUNT = 5000;

export default defineOperationRun({
  parameters,
  caption: 'Local shape descriptor per point from covariance eigenvalues.',
  createInputs: ({random, world}) => createTerrainCloud(random, world.width, POINT_COUNT),
  build(builder, cloud, values) {
    const capacity = values.capacity;
    const positions = builder.allocate('positions', 'float32x3', POINT_COUNT);
    const indices = builder.allocate('indices', 'uint32', POINT_COUNT * capacity);
    const counts = builder.allocate('counts', 'uint32', POINT_COUNT);
    builder.graph.add(
      new GPUPointCloudNeighborhoods({
        positions: positions.view,
        radius: 0.05,
        capacity,
        indices: indices.view,
        squaredDistances: builder.allocate('squared-distances', 'float32', POINT_COUNT * capacity)
          .view,
        counts: counts.view
      })
    );
    const shapes = builder.allocate('shapes', 'float32x3', POINT_COUNT);
    builder.graph.add(
      new GPUPointCloudFeatures({
        positions: positions.view,
        indices: indices.view,
        counts: counts.view,
        capacity,
        normals: builder.allocate('normals', 'float32x3', POINT_COUNT).view,
        eigenvalues: builder.allocate('eigenvalues', 'float32x3', POINT_COUNT).view,
        shapes: shapes.view,
        status: builder.allocate('status', 'uint32', POINT_COUNT).view
      })
    );
    return {
      outputs: {
        footprint: builder.upload('footprint', 'float32x2', cloud.footprint),
        linearity: addFloatCopy(builder, 'linearity', shapes, POINT_COUNT, {componentStart: 0}),
        planarity: addFloatCopy(builder, 'planarity', shapes, POINT_COUNT, {componentStart: 1}),
        scattering: addFloatCopy(builder, 'scattering', shapes, POINT_COUNT, {componentStart: 2})
      },
      update: current => positions.write(getJitteredPositions(cloud, current.noise))
    };
  },
  rebuildOn: ['capacity'],
  draw: ({outputs, values}) => [
    {
      layer: 'dots',
      positions: outputs.footprint,
      values: [outputs.linearity, outputs.planarity, outputs.scattering][values.shape],
      range: [0, 1],
      radius: 2.8
    }
  ]
});
