// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUNaturalNeighbor interpolates each grid cell with Sibson (Voronoi-area) weights of the
// surrounding samples; the surface is continuous and passes through every sample.

import {GPUNaturalNeighbor} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createUniformPoints} from '../../inputs';
import {parameters} from './natural-neighbor.parameters';
import {addChannel} from './support/channel';
import {createRunGrid} from './support/grid';

export default defineOperationRun({
  parameters,
  caption: 'Sibson weights from the Voronoi cells of the samples (dots).',
  createInputs: ({random, world}) => ({
    grid: createRunGrid(world, 36),
    samples: createUniformPoints(random, world, 60, 0.05)
  }),
  build(builder, {grid, samples}, values) {
    const cellCount = grid.columns * grid.rows;
    const sampleCount = values.sampleCount;
    const results = builder.allocate('results', 'float32x4', cellCount);
    builder.graph.add(
      new GPUNaturalNeighbor({
        positions: builder.upload(
          'positions',
          'float32x2',
          samples.positions.slice(0, sampleCount * 2)
        ).view,
        values: builder.upload('values', 'float32', samples.values.slice(0, sampleCount)).view,
        queries: builder.upload('queries', 'float32x2', grid.centers).view,
        maximumVertexCount: values.maximumVertexCount,
        maximumNeighborCount: 64,
        output: {values: results.view, status: builder.allocate('status', 'uint32', cellCount).view}
      })
    );
    return {
      outputs: {
        prediction: addChannel(builder, results, 0, 'prediction'),
        positions: builder.upload('sample-positions', 'float32x2', samples.positions),
        sampleCount
      }
    };
  },
  draw: ({inputs, outputs}) => [
    {
      layer: 'raster',
      values: outputs.prediction,
      columns: inputs.grid.columns,
      rows: inputs.grid.rows,
      range: [0, 1]
    },
    {
      layer: 'dots',
      context: true,
      positions: outputs.positions,
      count: outputs.sampleCount,
      radius: 2.5,
      color: [1, 1, 1, 0.9]
    }
  ]
});
