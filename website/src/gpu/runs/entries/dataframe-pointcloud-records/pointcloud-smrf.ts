// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUPointCloudSMRF (simple morphological filter). The raster is the provisional ground surface the
// GPU inpainted under the cloud; the dots are the points it classified: ground orange, removed
// (trees and other objects) blue.

import {GPUPointCloudSMRF} from '@luma-contributors/analysis/pointcloud-analysis';
import {defineOperationRun} from '../../define-run';
import {addFloatCopy} from './_lib/convert';
import {createCloudGrid, createTerrainCloud} from './_lib/helpers';
import {parameters} from './pointcloud-smrf.parameters';

const POINT_COUNT = 9000;
const CELL_SIZE = 0.03;

export default defineOperationRun({
  parameters,
  caption: 'Inpainted ground surface (raster) and ground / non-ground points (dots).',
  createInputs: ({random, world}) => createTerrainCloud(random, world.width, POINT_COUNT),
  build(builder, cloud, values) {
    const grid = createCloudGrid(cloud.width, CELL_SIZE);
    const cellCount = grid.width * grid.height;
    const groundSurface = builder.allocate('ground-surface', 'float32', cellCount);
    const groundMask = builder.allocate('ground-mask', 'uint32', POINT_COUNT);
    builder.graph.add(
      new GPUPointCloudSMRF({
        positions: builder.upload('positions', 'float32x3', cloud.positions).view,
        classifications: builder.upload('classes', 'uint32', new Uint32Array(POINT_COUNT).fill(1))
          .view,
        grid,
        slope: values.slope,
        window: values.window,
        threshold: values.threshold,
        cut: values.cut,
        groundSurface: groundSurface.view,
        groundMask: groundMask.view,
        outputClassifications: builder.allocate('output-classes', 'uint32', POINT_COUNT).view,
        status: builder.allocate('status', 'uint32', POINT_COUNT).view
      })
    );
    return {
      outputs: {
        footprint: builder.upload('footprint', 'float32x2', cloud.footprint),
        groundMask,
        surface: addFloatCopy(builder, 'surface-grid', groundSurface, cellCount, {
          flipColumns: grid.width
        })
      }
    };
  },
  rebuildOn: ['slope', 'window', 'threshold', 'cut'],
  draw: ({inputs, outputs}) => {
    const grid = createCloudGrid(inputs.width, CELL_SIZE);
    return [
      {
        layer: 'raster',
        values: outputs.surface,
        columns: grid.width,
        rows: grid.height,
        range: [0, 0.2]
      },
      {
        layer: 'dots',
        positions: outputs.footprint,
        values: outputs.groundMask,
        radius: 1.8,
        alpha: 0.85
      }
    ];
  }
});
