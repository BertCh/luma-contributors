// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPULocalCollinearity over k-nearest-neighbor neighborhoods. The second predictor follows the first
// more and more toward the east, so the local variance inflation factor (left) and the local
// condition number (right) grow from west to east.

import {GPULocalCollinearity} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createKnnWeights} from '../../inputs';
import {parameters} from './local-collinearity.parameters';
import {
  createRegressionInputs,
  drawTractMap,
  DUAL_MAP_SHARE,
  uploadRegressionMap
} from './shared/regression';
import {addTableColumn, uploadWeightsBuffers} from './shared/weights';

export default defineOperationRun({
  parameters,
  caption: 'Local variance inflation (left) and condition number (right).',
  createInputs: ({random, world}) =>
    createRegressionInputs(random, world, 10, DUAL_MAP_SHARE, true),
  build(builder, inputs, values) {
    const rowCount = inputs.tracts.count;
    const predictors = builder.allocate('predictors', 'float32', rowCount * 2);
    const weights = uploadWeightsBuffers(
      builder,
      'neighborhood',
      createKnnWeights(inputs.tracts.centers, values.neighbors)
    );
    const varianceInflationFactors = builder.allocate('vif', 'float32', rowCount * 2);
    const conditionNumbers = builder.allocate('condition-numbers', 'float32', rowCount);
    builder.graph.add(
      new GPULocalCollinearity({
        weights: weights.input,
        predictors: predictors.view,
        predictorCount: 2,
        output: {
          varianceInflationFactors: varianceInflationFactors.view,
          conditionNumbers: conditionNumbers.view,
          status: builder.allocate('status', 'uint32', rowCount).view
        }
      })
    );
    return {
      outputs: {
        meshes: [
          uploadRegressionMap(builder, inputs),
          uploadRegressionMap(builder, inputs, inputs.mapWidth * 1.14)
        ],
        inflation: addTableColumn(builder, 'inflation', varianceInflationFactors, 2, 0, rowCount),
        conditionNumbers
      },
      update: current => {
        const mixed = new Float32Array(rowCount * 2);
        for (let tract = 0; tract < rowCount; tract++) {
          const east = inputs.tracts.centers[tract * 2] / inputs.mapWidth;
          const strength = Math.min(1, Math.max(0, east)) * current.collinearity;
          const first = inputs.predictors[tract * 2];
          mixed[tract * 2] = first;
          mixed[tract * 2 + 1] =
            strength * first + (1 - strength) * inputs.predictors[tract * 2 + 1];
        }
        predictors.write(mixed);
      }
    };
  },
  rebuildOn: ['neighbors'],
  draw: ({inputs, outputs}) => [
    ...drawTractMap(inputs, outputs.meshes[0], outputs.inflation, {
      ramp: 'sequential',
      range: [1, 25]
    }),
    ...drawTractMap(inputs, outputs.meshes[1], outputs.conditionNumbers, {
      ramp: 'sequential',
      range: [1, 12]
    })
  ]
});
