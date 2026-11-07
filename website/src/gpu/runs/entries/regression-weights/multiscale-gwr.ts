// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUMultiscaleGeographicallyWeightedRegression gives every coefficient its own bandwidth. Left:
// the surface of the first slope, which truly drifts from west to east (a small bandwidth follows
// it). Right: the surface of the second slope, which is constant (a large bandwidth keeps it flat;
// a small one adds noise).

import {GPUMultiscaleGeographicallyWeightedRegression} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {parameters} from './multiscale-gwr.parameters';
import {
  createRegressionInputs,
  createResponse,
  drawTractMap,
  DUAL_MAP_SHARE,
  getDriftingCoefficients,
  uploadRegressionMap
} from './shared/regression';
import {addTableColumn} from './shared/weights';

export default defineOperationRun({
  parameters,
  caption:
    'Two coefficient surfaces fitted at their own scales: drifting slope and constant slope.',
  createInputs: ({random, world}) => createRegressionInputs(random, world, 8, DUAL_MAP_SHARE, true),
  build(builder, inputs) {
    const rowCount = inputs.tracts.count;
    const positions = builder.upload('positions', 'float32x2', inputs.tracts.centers);
    const predictors = builder.upload('predictors', 'float32', inputs.predictors);
    const response = builder.allocate('response', 'float32', rowCount);
    const parameterBuffer = builder.allocate('parameters', 'float32', 4);
    const coefficients = builder.allocate('coefficients', 'float32', rowCount * 3);
    builder.graph.add(
      new GPUMultiscaleGeographicallyWeightedRegression({
        positions: positions.view,
        predictors: predictors.view,
        predictorCount: 2,
        response: response.view,
        kernel: 'bisquare',
        maximumIterations: 50,
        parameters: parameterBuffer.view,
        output: {
          coefficients: coefficients.view,
          fitted: builder.allocate('fitted', 'float32', rowCount).view,
          status: {
            codes: builder.allocate('codes', 'uint32', rowCount).view,
            iterationCount: builder.allocate('iteration-count', 'uint32', 1).view
          }
        }
      })
    );
    return {
      outputs: {
        meshes: [
          uploadRegressionMap(builder, inputs),
          uploadRegressionMap(builder, inputs, inputs.mapWidth * 1.14)
        ],
        first: addTableColumn(builder, 'first-slope', coefficients, 3, 1, rowCount),
        second: addTableColumn(builder, 'second-slope', coefficients, 3, 2, rowCount)
      },
      update: current => {
        response.write(
          createResponse(inputs, {
            noise: 0.3,
            coefficients: getDriftingCoefficients(inputs, 2.5)
          })
        );
        parameterBuffer.write(
          Float32Array.of(
            1e-2,
            current.interceptBandwidth,
            current.firstBandwidth,
            current.secondBandwidth
          )
        );
      }
    };
  },
  draw: ({inputs, outputs}) => [
    ...drawTractMap(inputs, outputs.meshes[0], outputs.first, {
      ramp: 'sequential',
      range: [0, 3.2]
    }),
    ...drawTractMap(inputs, outputs.meshes[1], outputs.second, {
      ramp: 'diverging',
      range: [-3, 0]
    })
  ]
});
