// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUPointCloudHeight after GPUPointCloudSurfaces: each point's height above the terrain grid under
// it. Ground points are near zero (muted base dots); tree crowns climb to the top of the ramp.

import {
  GPUPointCloudHeight,
  GPUPointCloudSurfaces
} from '@luma-contributors/analysis/pointcloud-analysis';
import {defineOperationRun} from '../../define-run';
import {createCloudGrid, createTerrainCloud, getJitteredPositions} from './_lib/helpers';
import {parameters} from './pointcloud-height.parameters';

const POINT_COUNT = 9000;

export default defineOperationRun({
  parameters,
  caption: 'Height above ground per point.',
  createInputs: ({random, world}) => createTerrainCloud(random, world.width, POINT_COUNT),
  build(builder, cloud, values) {
    const grid = createCloudGrid(cloud.width, values.cellSize);
    const cellCount = grid.width * grid.height;
    const positions = builder.allocate('positions', 'float32x3', POINT_COUNT);
    const terrain = builder.allocate('terrain', 'float32', cellCount);
    const coverage = builder.allocate('coverage', 'uint32', cellCount);
    builder.graph.add(
      new GPUPointCloudSurfaces({
        positions: positions.view,
        classifications: builder.upload('classes', 'uint32', cloud.classes).view,
        groundClass: 2,
        grid,
        terrain: terrain.view,
        surface: builder.allocate('surface', 'float32', cellCount).view,
        canopy: builder.allocate('canopy', 'float32', cellCount).view,
        coverage: coverage.view
      })
    );
    const heights = builder.allocate('heights', 'float32', POINT_COUNT);
    builder.graph.add(
      new GPUPointCloudHeight({
        positions: positions.view,
        grid,
        terrain: terrain.view,
        coverage: coverage.view,
        coverageBit: 2,
        interpolation:
          values.interpolation === 0
            ? undefined // the prop's default: sample the containing cell
            : values.interpolation === 1
              ? {method: 'bilinear'}
              : {method: 'idw', radius: values.cellSize * 3},
        heights: heights.view,
        status: builder.allocate('status', 'uint32', POINT_COUNT).view
      })
    );
    return {
      outputs: {footprint: builder.upload('footprint', 'float32x2', cloud.footprint), heights},
      update: current => positions.write(getJitteredPositions(cloud, current.noise))
    };
  },
  rebuildOn: ['interpolation', 'cellSize'],
  draw: ({outputs}) => [
    {layer: 'dots', context: true, positions: outputs.footprint, radius: 2.4},
    {
      layer: 'dots',
      positions: outputs.footprint,
      values: outputs.heights,
      range: [0, 0.2],
      fade: true,
      radius: 2.8
    }
  ]
});
