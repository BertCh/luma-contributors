// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// addSpatialRegressionRecipe: OLS, diagnostics, residual local Moran and geographically weighted
// fits chained on tracts. Fill is the local R-squared of the weighted fit; dots are the OLS
// residuals of an illustrative response to two observed Chicago tract covariates.

import {
  addSpatialRegressionRecipe,
  getGPUGeographicallyWeightedRegressionParameterValues,
  GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
import {defineOperationRun} from '../../define-run';
import {findParameter, snapToParameter} from '../../interaction';
import {createRegressionInputs, createResponse} from '../regression-weights/shared/regression';
import {getTractFill, uploadTractMesh} from './lib/tracts';
import {findZoneAt, getTractGeometry} from './lib/zone-lookup';
import {parameters} from './spatial-regression-recipe.parameters';

const PREDICTOR_COUNT = 2;
const LADDER_CAPACITY = 4;
/** The tract near the middle of the map that anchors the kernel ring. */
const ANCHOR_TRACT = 50;

export default defineOperationRun({
  parameters,
  caption: 'Local fit quality of a modeled response on Chicago census tracts, by kernel bandwidth.',
  createInputs: ({random, world}) => createRegressionInputs(random, world, 11, 1),
  build(builder, inputs, values) {
    const {tracts, weights} = inputs;
    const predictors = builder.upload('predictors', 'float32', inputs.predictors);
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
    const ladder = (scale: number) => [0.6 * scale, scale, 1.6 * scale];
    const gwrValues = getGPUGeographicallyWeightedRegressionParameterValues(
      {kernel: 'bisquare', bandwidthMode: 'fixed', bandwidths: ladder(values.bandwidth)},
      LADDER_CAPACITY
    );
    const gwrParameters = builder.allocate('gwr-parameters', 'float32', gwrValues.length);
    addSpatialRegressionRecipe(builder.graph, {
      predictors: predictors.view,
      response: response.view,
      predictorCount: PREDICTOR_COUNT,
      weights: {
        offsets: builder.upload('weights-offsets', 'uint32', weights.offsets).view,
        neighbors: builder.upload('weights-neighbors', 'uint32', weights.neighbors).view,
        weights: builder.upload('weights-values', 'float32', weights.weights).view
      },
      parameters: builder.own(
        new GPUParameterBuffer(builder.device, {
          id: 'regression-parameters',
          format: 'float32',
          length: GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH
        })
      ),
      output: {
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
      outputs: {localR2, mesh: uploadTractMesh(builder, tracts)},
      update: current => {
        response.write(
          createResponse(inputs, {
            noise: 0.3,
            errorLambda: current.dependence,
            coefficients: tract => {
              const east = tracts.centers[tract * 2] / inputs.mapWidth;
              return [1, 2 + current.variation * (east - 0.5) * 3, -1.5];
            }
          })
        );
        gwrParameters.write(
          getGPUGeographicallyWeightedRegressionParameterValues(
            {kernel: 'bisquare', bandwidthMode: 'fixed', bandwidths: ladder(current.bandwidth)},
            LADDER_CAPACITY
          )
        );
      }
    };
  },
  // The distance from the anchor tract to the pointer is the kernel bandwidth: the ring is its reach.
  interaction: {
    kind: 'reach',
    hint: 'stretch the kernel',
    rest: ({toUnit, inputs}) =>
      toUnit([
        inputs.tracts.centers[ANCHOR_TRACT * 2] + parameters[0].default,
        inputs.tracts.centers[ANCHOR_TRACT * 2 + 1]
      ]),
    resolve(pointer, {inputs, outputs, peek}) {
      const anchor: [number, number] = [
        inputs.tracts.centers[ANCHOR_TRACT * 2],
        inputs.tracts.centers[ANCHOR_TRACT * 2 + 1]
      ];
      const bandwidth = snapToParameter(
        findParameter(parameters, 'bandwidth'),
        Math.hypot(pointer.x - anchor[0], pointer.y - anchor[1])
      );
      const localR2 = peek(outputs.localR2);
      const tract = findZoneAt(getTractGeometry(inputs.tracts), anchor[0], anchor[1]);
      return {
        values: {bandwidth},
        glyphs: [
          {shape: 'ring', at: anchor, radius: bandwidth},
          {shape: 'ring', at: anchor, radius: bandwidth * 1.6, dashed: true},
          {shape: 'dot', at: anchor}
        ],
        readout: `bandwidth ${bandwidth.toFixed(2)} · local R2 ${localR2 ? localR2[tract].toFixed(2) : '...'}`
      };
    }
  },
  draw: ({inputs, outputs}) => [
    getTractFill(outputs.mesh, inputs.tracts, outputs.localR2, {range: [0, 1]}),
    {
      layer: 'lines',
      context: true,
      segments: outputs.mesh.outline,
      count: inputs.tracts.outlineCount
    }
  ]
});
