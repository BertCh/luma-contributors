// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUKriging: ordinary kriging of scattered samples with a spherical, exponential or gaussian
// variogram. `output` switches between the prediction (`values`) and the kriging variance, which
// is lowest at the samples.

import {
  getGPUKrigingParameterValues,
  GPUKriging
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createUniformPoints} from '../../inputs';
import {parameters} from './kriging.parameters';
import {createRunGrid} from './support/grid';

const MODELS = ['spherical', 'exponential', 'gaussian'] as const;

export default defineOperationRun({
  parameters,
  caption: 'Kriged surface from scattered samples (dots).',
  createInputs: ({random, world}) => ({
    world,
    grid: createRunGrid(world, 64),
    samples: createUniformPoints(random, world, 70, 0.05)
  }),
  build(builder, {grid, samples}) {
    const positions = builder.upload('positions', 'float32x2', samples.positions);
    const sampleValues = builder.upload('values', 'float32', samples.values);
    const parameterBuffer = builder.allocate('parameters', 'float32', 12);
    const prediction = builder.allocate('prediction', 'float32', grid.columns * grid.rows);
    const variance = builder.allocate('variance', 'float32', grid.columns * grid.rows);
    builder.graph.add(
      new GPUKriging({
        positions: positions.view,
        values: sampleValues.view,
        parameters: parameterBuffer.view,
        grid: grid.grid,
        indexGridSize: [12, 8],
        indexBounds: grid.window,
        maximumNeighborCount: 16,
        output: {values: prediction.view, variance: variance.view}
      })
    );
    return {
      outputs: {prediction, variance, positions, sampleValues},
      update: values =>
        parameterBuffer.write(
          getGPUKrigingParameterValues(
            {
              searchRadius: Infinity,
              neighborCount: 16,
              variogram: {
                model: MODELS[values.model],
                nugget: values.nugget,
                sill: 1,
                range: values.range
              }
            },
            grid.grid
          )
        )
    };
  },
  draw: ({inputs, outputs, values}) => [
    {
      layer: 'raster',
      values: values.output === 1 ? outputs.variance : outputs.prediction,
      columns: inputs.grid.columns,
      rows: inputs.grid.rows,
      range: values.output === 1 ? 'auto' : [0, 1],
      smoothing: true
    },
    {layer: 'dots', context: true, positions: outputs.positions, radius: 2.5, color: [1, 1, 1, 0.9]}
  ]
});
