// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUConditionalGaussianSimulation draws random fields that honor the observed samples. A
// realization differs from the next one between the samples; `moments` summarize all replicates
// (mean and standard deviation).

import {GPUConditionalGaussianSimulation} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {parameters} from './conditional-gaussian-simulation.parameters';
import {addChannel} from './support/channel';
import {createInterpolationInputs} from './support/interpolation';

const REPLICATE_COUNT = 16;

export default defineOperationRun({
  parameters,
  caption: 'One simulated field that honors the samples; switch to the mean or spread of 16 draws.',
  createInputs: ({random, world}) => createInterpolationInputs(random, world, 28, 40),
  build(builder, {grid, samples, observations}, values) {
    const cellCount = grid.columns * grid.rows;
    const realizations = builder.allocate('realizations', 'float32', cellCount * REPLICATE_COUNT);
    const moments = builder.allocate('moments', 'float32x2', cellCount);
    builder.graph.add(
      new GPUConditionalGaussianSimulation({
        positions: builder.upload('positions', 'float32x2', samples.positions).view,
        observations: builder.upload('observations', 'float32x4', observations).view,
        queries: builder.upload('queries', 'float32x2', grid.centers).view,
        covariance: {model: 'exponential', range: values.range, sill: 0.06, nugget: 0},
        mean: 0.5,
        seed: values.seed,
        replicateCount: REPLICATE_COUNT,
        output: {
          values: realizations.view,
          moments: moments.view,
          status: builder.allocate('status', 'uint32', REPLICATE_COUNT).view
        }
      })
    );
    return {
      outputs: {
        realizations,
        mean: addChannel(builder, moments, 0, 'mean', 2),
        spread: addChannel(builder, moments, 1, 'spread', 2),
        positions: builder.upload('sample-positions', 'float32x2', samples.positions)
      }
    };
  },
  draw: ({inputs, outputs, values}) => [
    {
      layer: 'raster',
      values: [outputs.realizations, outputs.mean, outputs.spread][values.output],
      columns: inputs.grid.columns,
      rows: inputs.grid.rows,
      range: values.output === 2 ? 'auto' : [0, 1],
      smoothing: true
    },
    {layer: 'dots', context: true, positions: outputs.positions, radius: 2.5, color: [1, 1, 1, 0.9]}
  ]
});
