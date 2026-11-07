// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUGeographicallyWeightedRegression fits a separate regression at every tract center. Left: the
// local slope of the first predictor, which drifts from west to east in the data. Right: the
// local R2. A small bandwidth follows the drift but is noisy; a large one flattens it.

import {
  GPUGeographicallyWeightedRegression,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY_LENGTH,
  getGPUGeographicallyWeightedRegressionParameterLength,
  getGPUGeographicallyWeightedRegressionParameterValues
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {parameters} from './geographically-weighted-regression.parameters';
import {
  createRegressionInputs,
  createResponse,
  drawTractMap,
  DUAL_MAP_SHARE,
  getDriftingCoefficients,
  uploadRegressionMap
} from './shared/regression';
import {addTableColumn} from './shared/weights';

const LADDER_CAPACITY = 4;

export default defineOperationRun({
  parameters,
  caption: 'Local slope of the first predictor (left) and local R2 (right).',
  createInputs: ({random, world}) =>
    createRegressionInputs(random, world, 10, DUAL_MAP_SHARE, true),
  build(builder, inputs) {
    const rowCount = inputs.tracts.count;
    const positions = builder.upload('positions', 'float32x2', inputs.tracts.centers);
    const predictors = builder.upload('predictors', 'float32', inputs.predictors);
    const response = builder.allocate('response', 'float32', rowCount);
    const parameterBuffer = builder.allocate(
      'parameters',
      'float32',
      getGPUGeographicallyWeightedRegressionParameterLength(LADDER_CAPACITY)
    );
    const coefficients = builder.allocate('coefficients', 'float32', rowCount * 3);
    const localR2 = builder.allocate('local-r2', 'float32', rowCount);
    builder.graph.add(
      new GPUGeographicallyWeightedRegression({
        positions: positions.view,
        predictors: predictors.view,
        predictorCount: 2,
        response: response.view,
        parameters: parameterBuffer.view,
        maximumBandwidthCount: LADDER_CAPACITY,
        maximumNeighborCount: 64,
        refinement: false,
        output: {
          coefficients: coefficients.view,
          localR2: localR2.view,
          selectedBandwidth: builder.allocate('selected-bandwidth', 'float32', 2).view,
          summary: builder.allocate(
            'summary',
            'float32',
            GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY_LENGTH
          ).view
        }
      })
    );
    return {
      outputs: {
        meshes: [
          uploadRegressionMap(builder, inputs),
          uploadRegressionMap(builder, inputs, inputs.mapWidth * 1.14)
        ],
        slope: addTableColumn(builder, 'local-slope', coefficients, 3, 1, rowCount),
        localR2
      },
      update: current => {
        response.write(
          createResponse(inputs, {
            noise: 0.3,
            coefficients: getDriftingCoefficients(inputs, current.drift)
          })
        );
        parameterBuffer.write(
          getGPUGeographicallyWeightedRegressionParameterValues(
            {
              bandwidths: [current.bandwidth],
              kernel: current.kernel === 0 ? 'gaussian' : 'bisquare'
            },
            LADDER_CAPACITY
          )
        );
      }
    };
  },
  draw: ({inputs, outputs}) => [
    ...drawTractMap(inputs, outputs.meshes[0], outputs.slope, {
      ramp: 'sequential',
      range: [0, 3.2]
    }),
    ...drawTractMap(inputs, outputs.meshes[1], outputs.localR2, {ramp: 'sequential', range: [0, 1]})
  ]
});
