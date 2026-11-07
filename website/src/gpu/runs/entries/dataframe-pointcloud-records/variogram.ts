// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUVariogram of observations of a spatial field (left panel, colored by value). Each bar is the
// semivariance of the point pairs in one distance bin: it rises with distance until the sill, and
// its intercept at zero distance is the nugget (noise).

import {
  GPUVariogram,
  GPU_VARIOGRAM_PARAMETER_LENGTH,
  getGPUVariogramParameterValues
} from '@luma-contributors/analysis/dataframe-analysis';
import {defineOperationRun} from '../../define-run';
import {createFieldSample, getFieldValues} from './_lib/helpers';
import {parameters} from './variogram.parameters';

const POINT_COUNT = 700;
const PLOT = [0.58, 0.12, 0.96, 0.88] as const;

export default defineOperationRun({
  parameters,
  caption: 'Semivariance per distance bin of a noisy spatial field.',
  createInputs: ({random, world}) => createFieldSample(random, world.width * 0.5, POINT_COUNT),
  build(builder, sample, values) {
    const positions = builder.upload('positions', 'float32x2', sample.positions);
    const observed = builder.allocate('values', 'float32', POINT_COUNT);
    const parameterBuffer = builder.allocate(
      'parameters',
      'float32',
      GPU_VARIOGRAM_PARAMETER_LENGTH
    );
    const semivariances = builder.allocate('semivariances', 'float32', values.lagCount);
    builder.graph.add(
      new GPUVariogram({
        positions: positions.view,
        values: observed.view,
        parameters: parameterBuffer.view,
        gridSize: [32, 32],
        lagCount: values.lagCount,
        output: {semivariances: semivariances.view}
      })
    );
    return {
      outputs: {positions, observed, semivariances},
      update: current => {
        observed.write(getFieldValues(sample, current.noise));
        parameterBuffer.write(
          getGPUVariogramParameterValues({
            bounds: [0, 0, sample.windowWidth, 1],
            maximumDistance: current.maximumDistance
          })
        );
      }
    };
  },
  rebuildOn: ['lagCount'],
  draw: ({outputs}) => [
    {
      layer: 'dots',
      context: true,
      positions: outputs.positions,
      values: outputs.observed,
      range: [0, 10],
      radius: 2.8
    },
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.semivariances,
      range: [0, 5],
      rectangle: PLOT,
      color: [0.3, 0.75, 1, 0.9]
    }
  ]
});
