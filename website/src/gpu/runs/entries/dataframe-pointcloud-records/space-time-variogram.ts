// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSpaceTimeVariogram of a field that also changes over time. The grid is the semivariance per
// (spatial lag, temporal lag) bin: columns are time lags, rows are distance bins from near (top)
// to far. It grows along both axes when the field varies in space and in time.

import {GPUSpaceTimeVariogram} from '@luma-contributors/analysis/dataframe-analysis';
import {defineOperationRun} from '../../define-run';
import {createFieldSample, drawNormal} from './_lib/helpers';
import {parameters} from './space-time-variogram.parameters';

const POINT_COUNT = 900;
const LAG_COUNT = 16;
const TEMPORAL_LAG_COUNT = 8;

export default defineOperationRun({
  parameters,
  caption: 'Semivariance per distance (rows) and time lag (columns).',
  createInputs: ({random, world}) => ({
    sample: createFieldSample(random, world.width, POINT_COUNT),
    times: Float32Array.from({length: POINT_COUNT}, () => random() * 24),
    phases: Float32Array.from({length: POINT_COUNT}, () => drawNormal(random) * 0.2)
  }),
  build(builder, inputs) {
    const {sample} = inputs;
    const positions = builder.upload('positions', 'float32x2', sample.positions);
    const values = builder.allocate('values', 'float32', POINT_COUNT);
    const times = builder.upload('times', 'float32', inputs.times);
    const parameterBuffer = builder.allocate('parameters', 'float32', 2);
    const semivariances = builder.allocate(
      'semivariances',
      'float32',
      LAG_COUNT * TEMPORAL_LAG_COUNT
    );
    builder.graph.add(
      new GPUSpaceTimeVariogram({
        positions: positions.view,
        values: values.view,
        times: times.view,
        parameters: parameterBuffer.view,
        lagCount: LAG_COUNT,
        temporalLagCount: TEMPORAL_LAG_COUNT,
        output: {
          semivariances: semivariances.view,
          pairCounts: builder.allocate('pair-counts', 'uint32', LAG_COUNT * TEMPORAL_LAG_COUNT)
            .view,
          status: builder.allocate('status', 'uint32', LAG_COUNT * TEMPORAL_LAG_COUNT).view
        }
      })
    );
    return {
      outputs: {semivariances},
      update: current => {
        values.write(
          Float32Array.from(
            sample.field,
            (value, index) =>
              value +
              Math.sin((inputs.times[index] / 24) * Math.PI * 2 + inputs.phases[index]) *
                current.drift
          )
        );
        parameterBuffer.write(Float32Array.of(current.maximumDistance, current.maximumTimeLag));
      }
    };
  },
  rebuildOn: [],
  draw: ({outputs}) => [
    {
      layer: 'raster',
      values: outputs.semivariances,
      columns: TEMPORAL_LAG_COUNT,
      rows: LAG_COUNT,
      range: [0, 8]
    }
  ]
});
