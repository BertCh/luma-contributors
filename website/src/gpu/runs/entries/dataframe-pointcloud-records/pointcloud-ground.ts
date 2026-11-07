// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUPointCloudGround (progressive morphological filter) on the lowest-point grid from
// GPUPointCloudSurfaces. Points kept as ground are orange; points the filter removes (trees) are
// blue. Larger windows and thresholds remove more.

import {
  GPUPointCloudGround,
  GPUPointCloudSurfaces
} from '@luma-contributors/analysis/pointcloud-analysis';
import {defineOperationRun} from '../../define-run';
import {createCloudGrid, createTerrainCloud} from './_lib/helpers';
import {parameters} from './pointcloud-ground.parameters';

const POINT_COUNT = 9000;
const CELL_SIZE = 0.03;

export default defineOperationRun({
  parameters,
  caption: 'Ground points (orange) kept by the morphological filter; others (blue) removed.',
  createInputs: ({random, world}) => createTerrainCloud(random, world.width, POINT_COUNT),
  build(builder, cloud, values) {
    const grid = createCloudGrid(cloud.width, CELL_SIZE);
    const cellCount = grid.width * grid.height;
    const positions = builder.upload('positions', 'float32x3', cloud.positions);
    const classes = builder.upload('classes', 'uint32', new Uint32Array(POINT_COUNT).fill(1));
    const minimum = builder.allocate('minimum', 'float32', cellCount);
    const coverage = builder.allocate('coverage', 'uint32', cellCount);
    builder.graph.add(
      new GPUPointCloudSurfaces({
        positions: positions.view,
        classifications: classes.view,
        groundClass: 1,
        grid,
        terrain: minimum.view,
        surface: builder.allocate('surface', 'float32', cellCount).view,
        canopy: builder.allocate('canopy', 'float32', cellCount).view,
        coverage: coverage.view
      })
    );
    const groundMask = builder.allocate('ground-mask', 'uint32', POINT_COUNT);
    builder.graph.add(
      new GPUPointCloudGround({
        positions: positions.view,
        classifications: classes.view,
        grid,
        minimumSurface: minimum.view,
        coverage: coverage.view,
        radii: [1, 2, 4, 8].slice(0, values.radiusCount),
        initialThreshold: values.initialThreshold,
        slope: values.slope,
        maximumThreshold: values.maximumThreshold,
        groundMask: groundMask.view,
        outputClassifications: builder.allocate('output-classes', 'uint32', POINT_COUNT).view,
        status: builder.allocate('status', 'uint32', POINT_COUNT).view
      })
    );
    return {
      outputs: {footprint: builder.upload('footprint', 'float32x2', cloud.footprint), groundMask}
    };
  },
  rebuildOn: ['radiusCount', 'initialThreshold', 'slope', 'maximumThreshold'],
  draw: ({outputs}) => [
    {layer: 'dots', positions: outputs.footprint, values: outputs.groundMask, radius: 2.4}
  ]
});
