// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUConditionalGaussianSimulation draws random fields that honor the observed samples. A
// realization differs from the next one between the samples; `moments` summarize all replicates
// (mean and standard deviation).

import {GPUConditionalGaussianSimulation} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {bindAxes} from '../../interaction';
import {parameters} from './conditional-gaussian-simulation.parameters';
import {addChannel, addReplicateSelect} from './support/channel';
import {createInterpolationInputs, drawSampleDots} from './support/interpolation';

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
    const {selected, index} = addReplicateSelect(builder, realizations, cellCount, 'realization');
    return {
      update: current => index.write(Uint32Array.of(current.replicate)),
      outputs: {
        realization: selected,
        mean: addChannel(builder, moments, 0, 'mean', 2),
        spread: addChannel(builder, moments, 1, 'spread', 2),
        positions: builder.upload('sample-positions', 'float32x2', samples.positions),
        sampleValues: builder.upload('sample-values', 'float32', samples.values)
      }
    };
  },
  rebuildOn: ['seed', 'range'],
  // All 16 replicates are in one run; the pointer plays them back in order.
  interaction: bindAxes(parameters, {
    kind: 'time',
    hint: 'play the replicates',
    x: 'replicate',
    decorate: values => ({
      readout: `replicate ${values.replicate + 1} of ${REPLICATE_COUNT} · seed ${values.seed}`
    })
  }),
  draw: ({inputs, outputs, values}) => [
    {
      layer: 'raster',
      values: [outputs.realization, outputs.mean, outputs.spread][values.output],
      columns: inputs.grid.columns,
      rows: inputs.grid.rows,
      range: values.output === 2 ? 'auto' : [0, 1],
      smoothing: true
    },
    ...drawSampleDots(outputs.positions, outputs.sampleValues)
  ]
});
