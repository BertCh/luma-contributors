// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSpatialCorrelogram of a spatial field. Bars are Moran's I of the points within each distance
// band (positive means similar values sit together); the line is its z-score. Both fall as the
// bands reach farther.

import {
  GPUSpatialCorrelogram,
  GPU_SPATIAL_CORRELOGRAM_PARAMETER_LENGTH,
  getGPUSpatialCorrelogramParameterValues
} from '@luma-contributors/analysis/dataframe-analysis';
import {defineOperationRun} from '../../define-run';
import {createFieldSample, getFieldValues} from './_lib/helpers';
import {parameters} from './spatial-correlogram.parameters';

const POINT_COUNT = 600;
const PLOT = [0.58, 0.12, 0.96, 0.88] as const;

export default defineOperationRun({
  parameters,
  caption: 'Moran’s I (bars) and its z-score (line) per distance band.',
  createInputs: ({random, world}) => createFieldSample(random, world.width * 0.5, POINT_COUNT),
  build(builder, sample, values) {
    const positions = builder.upload('positions', 'float32x2', sample.positions);
    const observed = builder.allocate('values', 'float32', POINT_COUNT);
    const parameterBuffer = builder.allocate(
      'parameters',
      'float32',
      GPU_SPATIAL_CORRELOGRAM_PARAMETER_LENGTH
    );
    const moransI = builder.allocate('morans-i', 'float32', values.bandCount);
    const zScores = builder.allocate('z-scores', 'float32', values.bandCount);
    builder.graph.add(
      new GPUSpatialCorrelogram({
        positions: positions.view,
        values: observed.view,
        parameters: parameterBuffer.view,
        gridSize: [32, 32],
        bandCount: values.bandCount,
        bandMode: values.bandMode === 0 ? 'cumulative' : 'annulus',
        output: {moransI: moransI.view, zScores: zScores.view}
      })
    );
    return {
      outputs: {positions, observed, moransI, zScores},
      update: current => {
        observed.write(getFieldValues(sample, current.noise));
        parameterBuffer.write(
          getGPUSpatialCorrelogramParameterValues({
            bounds: [0, 0, sample.windowWidth, 1],
            maximumDistance: current.maximumDistance
          })
        );
      }
    };
  },
  rebuildOn: ['bandCount', 'bandMode'],
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
      values: outputs.moransI,
      range: [-0.2, 1],
      rectangle: PLOT,
      color: [0.3, 0.75, 1, 0.8]
    },
    {
      layer: 'chart',
      kind: 'line',
      values: outputs.zScores,
      range: [-5, 40],
      rectangle: PLOT,
      radius: 2,
      color: [1, 0.6, 0.2, 1]
    }
  ]
});
