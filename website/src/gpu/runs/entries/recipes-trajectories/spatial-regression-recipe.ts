// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// addSpatialRegressionRecipe: OLS, diagnostics, residual local Moran and geographically weighted
// fits chained on tracts. Fill is the local R-squared of the weighted fit; dots are the OLS
// residuals (clusters of one color mean the errors are spatially dependent).

import {
  addSpatialRegressionRecipe,
  getGPUGeographicallyWeightedRegressionParameterValues,
  getGPUSpatialAutocorrelationParameterValues,
  GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH,
  GPUParameterBuffer
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createTracts} from '../../inputs';
import {getTractFill, uploadTractMesh} from './lib/tracts';
import {parameters} from './spatial-regression-recipe.parameters';

const PREDICTOR_COUNT = 2;
const LADDER_CAPACITY = 4;

export default defineOperationRun({
  parameters,
  caption: 'Local fit quality (fill) and OLS residuals (dots).',
  createInputs: ({random, world}) => {
    const tracts = createTracts(random, world, 11);
    const draw = () => Float32Array.from({length: tracts.count}, () => random() * 2 - 1);
    return {tracts, world, firstPredictor: draw(), secondPredictor: draw(), noise: draw()};
  },
  build(builder, inputs, values) {
    const {tracts} = inputs;
    const predictors = builder.allocate('predictors', 'float32', tracts.count * PREDICTOR_COUNT);
    const response = builder.allocate('response', 'float32', tracts.count);
    const residuals = builder.allocate('residuals', 'float32', tracts.count);
    const coefficients = builder.allocate('coefficients', 'float32', PREDICTOR_COUNT + 1);
    const olsStatus = builder.allocate('ols-status', 'uint32', 1);
    const localCoefficients = builder.allocate(
      'local-coefficients',
      'float32',
      tracts.count * (PREDICTOR_COUNT + 1)
    );
    const localR2 = builder.allocate('local-r2', 'float32', tracts.count);
    // Row-standardized contiguity weights.
    const rowWeights = new Float32Array(tracts.contiguity.neighbors.length);
    for (let row = 0; row < tracts.count; row++) {
      const first = tracts.contiguity.offsets[row];
      const last = tracts.contiguity.offsets[row + 1];
      rowWeights.fill(1 / Math.max(1, last - first), first, last);
    }
    const ladder = (scale: number) => [0.6 * scale, scale, 1.6 * scale];
    const gwrValues = getGPUGeographicallyWeightedRegressionParameterValues(
      {kernel: 'bisquare', bandwidthMode: 'fixed', bandwidths: ladder(values.bandwidth)},
      LADDER_CAPACITY
    );
    const gwrParameters = builder.allocate('gwr-parameters', 'float32', gwrValues.length);
    const autocorrelation = builder.allocate(
      'autocorrelation',
      'float32',
      GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH
    );
    addSpatialRegressionRecipe(builder.graph, {
      predictors: predictors.view,
      response: response.view,
      predictorCount: PREDICTOR_COUNT,
      weights: {
        offsets: builder.upload('weights-offsets', 'uint32', tracts.contiguity.offsets).view,
        neighbors: builder.upload('weights-neighbors', 'uint32', tracts.contiguity.neighbors).view,
        weights: builder.upload('weights-values', 'float32', rowWeights).view
      },
      parameters: new GPUParameterBuffer(builder.device, {
        id: 'regression-parameters',
        format: 'float32',
        length: GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH
      }),
      outputs: {
        ols: {
          coefficients: coefficients.view,
          status: {code: olsStatus.view},
          residuals: residuals.view
        },
        localFits: {coefficients: localCoefficients.view, localR2: localR2.view}
      },
      localFits: {
        positions: builder.upload('positions', 'float32x2', tracts.centers).view,
        parameters: gwrParameters.view,
        maximumBandwidthCount: LADDER_CAPACITY
      }
    });
    return {
      outputs: {localR2, residuals, mesh: uploadTractMesh(builder, tracts)},
      update: current => {
        const rows = tracts.count;
        const stacked = new Float32Array(rows * PREDICTOR_COUNT);
        const responses = new Float32Array(rows);
        // Errors smoothed over the contiguity graph, so they are spatially dependent.
        let errors = Float32Array.from(inputs.noise);
        for (let pass = 0; pass < 4; pass++) {
          errors = Float32Array.from(inputs.noise, (own, row) => {
            let sum = 0;
            const first = tracts.contiguity.offsets[row];
            const last = tracts.contiguity.offsets[row + 1];
            for (let slot = first; slot < last; slot++)
              sum += errors[tracts.contiguity.neighbors[slot]];
            return (
              own * (1 - current.dependence) +
              (current.dependence * sum) / Math.max(1, last - first)
            );
          });
        }
        for (let row = 0; row < rows; row++) {
          const x = tracts.centers[row * 2] / inputs.world.width;
          stacked[row * 2] = 3 + 4 * inputs.firstPredictor[row];
          stacked[row * 2 + 1] = inputs.secondPredictor[row];
          const slope = 2 + current.variation * (x - 0.5) * 3;
          responses[row] = 1 + slope * stacked[row * 2] - stacked[row * 2 + 1] + errors[row] * 8;
        }
        predictors.write(stacked);
        response.write(responses);
        autocorrelation.write(getGPUSpatialAutocorrelationParameterValues());
        gwrParameters.write(
          getGPUGeographicallyWeightedRegressionParameterValues(
            {kernel: 'bisquare', bandwidthMode: 'fixed', bandwidths: ladder(current.bandwidth)},
            LADDER_CAPACITY
          )
        );
      }
    };
  },
  draw: ({inputs, outputs}) => [
    getTractFill(outputs.mesh, inputs.tracts, outputs.localR2),
    {
      layer: 'lines',
      context: true,
      segments: outputs.mesh.outline,
      count: inputs.tracts.outlineCount
    },
    {
      layer: 'dots',
      positions: outputs.mesh.centers,
      values: outputs.residuals,
      ramp: 'diverging',
      radius: 4
    }
  ]
});
