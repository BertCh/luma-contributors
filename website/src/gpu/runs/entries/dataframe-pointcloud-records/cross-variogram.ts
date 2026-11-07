// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUCrossVariogram of two variables observed at the same points (dots: first variable). Each
// bar is half the mean product of the two variables' differences over the pairs in a distance
// bin: positive when they vary together, negative when they oppose.

import {GPUCrossVariogram} from '@luma-contributors/analysis/dataframe-analysis';
import {defineOperationRun} from '../../define-run';
import {createFieldSample, getFieldValues} from './_lib/helpers';
import {parameters} from './cross-variogram.parameters';

const POINT_COUNT = 700;
const PLOT = [0.58, 0.12, 0.96, 0.88] as const;

export default defineOperationRun({
  parameters,
  caption: 'Cross semivariance of two variables per distance bin.',
  createInputs: ({random, world}) => createFieldSample(random, world.width * 0.5, POINT_COUNT),
  build(builder, sample, values) {
    const positions = builder.upload('positions', 'float32x2', sample.positions);
    const first = builder.upload('first', 'float32', getFieldValues(sample, 0.3));
    const second = builder.allocate('second', 'float32', POINT_COUNT);
    const parameterBuffer = builder.allocate('parameters', 'float32', 2);
    const semivariances = builder.allocate('semivariances', 'float32', values.lagCount);
    builder.graph.add(
      new GPUCrossVariogram({
        positions: positions.view,
        values: first.view,
        secondValues: second.view,
        parameters: parameterBuffer.view,
        lagCount: values.lagCount,
        output: {
          semivariances: semivariances.view,
          pairCounts: builder.allocate('pair-counts', 'uint32', values.lagCount).view,
          status: builder.allocate('status', 'uint32', values.lagCount).view
        }
      })
    );
    return {
      outputs: {positions, first, semivariances},
      update: current => {
        const mix = Math.sqrt(1 - current.correlation ** 2);
        second.write(
          Float32Array.from(
            sample.field,
            (value, index) =>
              5 + current.correlation * (value - 5) + mix * (sample.secondField[index] - 5)
          )
        );
        parameterBuffer.write(Float32Array.of(current.maximumDistance, 1));
      }
    };
  },
  rebuildOn: ['lagCount'],
  draw: ({outputs}) => [
    {
      layer: 'dots',
      context: true,
      positions: outputs.positions,
      values: outputs.first,
      range: [0, 10],
      radius: 2.8
    },
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.semivariances,
      range: [-3, 3],
      rectangle: PLOT,
      color: [0.3, 0.75, 1, 0.45]
    },
    {
      layer: 'chart',
      kind: 'line',
      values: outputs.semivariances,
      range: [-3, 3],
      rectangle: PLOT,
      radius: 2.6,
      color: [0.3, 0.75, 1, 1]
    }
  ]
});
