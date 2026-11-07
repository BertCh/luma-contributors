// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUVariogramModelFit on the bins of a GPUVariogram. The dots are the empirical semivariances;
// the three curves are the spherical (cyan), exponential (orange) and gaussian (white) models the
// GPU fitted, evaluated from the fit's parameter rows.

import {
  GPUVariogram,
  GPUVariogramModelFit,
  GPU_VARIOGRAM_PARAMETER_LENGTH,
  getGPUVariogramParameterValues
} from '@luma-contributors/analysis/dataframe-analysis';
import {defineOperationRun} from '../../define-run';
import {addFittedCurve} from './_lib/fitted-curve';
import {createFieldSample, getFieldValues} from './_lib/helpers';
import {parameters} from './variogram-model-fit.parameters';

const POINT_COUNT = 700;
const LAG_COUNT = 20;
const CURVE_COUNT = 48;
const PLOT = [0.58, 0.12, 0.96, 0.88] as const;
const WEIGHTINGS = ['cressie', 'pairs', 'none'] as const;

export default defineOperationRun({
  parameters,
  caption: 'Empirical semivariance bins and the three models fitted to them.',
  createInputs: ({random, world}) => createFieldSample(random, world.width * 0.5, POINT_COUNT),
  build(builder, sample, values) {
    const positions = builder.upload('positions', 'float32x2', sample.positions);
    const observed = builder.allocate('values', 'float32', POINT_COUNT);
    const parameterBuffer = builder.allocate(
      'parameters',
      'float32',
      GPU_VARIOGRAM_PARAMETER_LENGTH
    );
    const semivariances = builder.allocate('semivariances', 'float32', LAG_COUNT);
    const pairCounts = builder.allocate('pair-counts', 'uint32', LAG_COUNT);
    const meanDistances = builder.allocate('mean-distances', 'float32', LAG_COUNT);
    builder.graph.add(
      new GPUVariogram({
        positions: positions.view,
        values: observed.view,
        parameters: parameterBuffer.view,
        gridSize: [32, 32],
        lagCount: LAG_COUNT,
        output: {
          semivariances: semivariances.view,
          pairCounts: pairCounts.view,
          meanDistances: meanDistances.view
        }
      })
    );
    const fitted = builder.allocate('fitted', 'float32x4', 3);
    builder.graph.add(
      new GPUVariogramModelFit({
        distances: meanDistances.view,
        semivariances: semivariances.view,
        pairCounts: pairCounts.view,
        models: ['spherical', 'exponential', 'gaussian'],
        weighting: WEIGHTINGS[values.weighting],
        output: {
          parameters: fitted.view,
          diagnostics: builder.allocate('diagnostics', 'float32x4', 3).view,
          status: builder.allocate('status', 'uint32', 3).view
        }
      })
    );
    const curveDistances = builder.allocate('curve-distances', 'float32', CURVE_COUNT);
    return {
      outputs: {
        positions,
        observed,
        semivariances,
        meanDistances,
        spherical: addFittedCurve(builder, 'spherical', fitted, curveDistances, 0),
        exponential: addFittedCurve(builder, 'exponential', fitted, curveDistances, 1),
        gaussian: addFittedCurve(builder, 'gaussian', fitted, curveDistances, 2)
      },
      update: current => {
        observed.write(getFieldValues(sample, current.noise));
        curveDistances.write(
          Float32Array.from(
            {length: CURVE_COUNT},
            (_, index) => (index / (CURVE_COUNT - 1)) * current.maximumDistance
          )
        );
        parameterBuffer.write(
          getGPUVariogramParameterValues({
            bounds: [0, 0, sample.windowWidth, 1],
            maximumDistance: current.maximumDistance
          })
        );
      }
    };
  },
  rebuildOn: ['weighting'],
  draw: ({outputs, values}) => [
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
      kind: 'scatter',
      values: outputs.semivariances,
      xValues: outputs.meanDistances,
      xRange: [0, values.maximumDistance],
      range: [0, 5],
      rectangle: PLOT,
      radius: 5,
      color: [0.85, 0.85, 0.9, 0.9]
    },
    {
      layer: 'chart',
      kind: 'line',
      values: outputs.spherical,
      range: [0, 5],
      rectangle: PLOT,
      radius: 2.4,
      color: [0.3, 0.75, 1, 1]
    },
    {
      layer: 'chart',
      kind: 'line',
      values: outputs.exponential,
      range: [0, 5],
      rectangle: PLOT,
      radius: 2.4,
      color: [1, 0.6, 0.2, 1]
    },
    {
      layer: 'chart',
      kind: 'line',
      values: outputs.gaussian,
      range: [0, 5],
      rectangle: PLOT,
      radius: 2.4,
      color: [0.7, 1, 0.7, 1]
    }
  ]
});
