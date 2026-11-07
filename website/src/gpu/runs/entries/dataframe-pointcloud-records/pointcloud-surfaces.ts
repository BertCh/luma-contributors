// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUPointCloudSurfaces rasterizes a classified cloud into terrain (lowest ground), surface (highest
// point) and canopy (surface minus terrain). The picture is the canopy height grid: tree crowns
// stand out over bare ground. With groundClass 1 every point counts as ground and the canopy flattens.

import {GPUPointCloudSurfaces} from '@luma-contributors/analysis/pointcloud-analysis';
import {defineOperationRun} from '../../define-run';
import {addFloatCopy} from './_lib/convert';
import {createCloudGrid, createTerrainCloud, getJitteredPositions} from './_lib/helpers';
import {parameters} from './pointcloud-surfaces.parameters';

const POINT_COUNT = 9000;

export default defineOperationRun({
  parameters,
  caption: 'Canopy height (surface minus terrain) per grid cell.',
  createInputs: ({random, world}) => createTerrainCloud(random, world.width, POINT_COUNT),
  build(builder, cloud, values) {
    const grid = createCloudGrid(cloud.width, values.cellSize);
    const cellCount = grid.width * grid.height;
    const positions = builder.allocate('positions', 'float32x3', POINT_COUNT);
    const canopy = builder.allocate('canopy', 'float32', cellCount);
    builder.graph.add(
      new GPUPointCloudSurfaces({
        positions: positions.view,
        classifications: builder.upload('classes', 'uint32', cloud.classes).view,
        groundClass: values.groundClass,
        grid,
        terrain: builder.allocate('terrain', 'float32', cellCount).view,
        surface: builder.allocate('surface', 'float32', cellCount).view,
        canopy: canopy.view,
        coverage: builder.allocate('coverage', 'uint32', cellCount).view
      })
    );
    return {
      outputs: {
        canopy: addFloatCopy(builder, 'canopy-grid', canopy, cellCount, {flipColumns: grid.width})
      },
      update: current => positions.write(getJitteredPositions(cloud, current.noise))
    };
  },
  rebuildOn: ['groundClass', 'cellSize'],
  draw: ({inputs, outputs, values}) => {
    const grid = createCloudGrid(inputs.width, values.cellSize);
    return [
      {
        layer: 'raster',
        values: outputs.canopy,
        columns: grid.width,
        rows: grid.height,
        range: [0, 0.25]
      }
    ];
  }
});
