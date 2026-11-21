// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// addInterpolationDiagnosticsRecipe: kriging plus leave-one-out cross validation. The surface is
// the kriged prediction; dots are the samples colored by their leave-one-out residual (blue
// over-predicted, red under-predicted).

import {
  addInterpolationDiagnosticsRecipe,
  getGPUKrigingParameterValues
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {findParameter, snapToParameter} from '../../interaction';
import {createUniformPoints} from '../../inputs';
import {findBusiestPoint} from './lib/points';
import {createWorldGrid} from './lib/rasterize';
import {parameters} from './interpolation-diagnostics-recipe.parameters';

const MODELS = ['spherical', 'exponential', 'gaussian'] as const;

export default defineOperationRun({
  parameters,
  caption: 'Kriged surface with leave-one-out residuals at the samples.',
  createInputs: ({random, world}) => {
    const samples = createUniformPoints(random, world, 60, 0.35);
    return {
      grid: createWorldGrid(world),
      samples,
      anchor: findBusiestPoint(samples.positions, 0.3)
    };
  },
  build(builder, {grid, samples}) {
    const sampleCount = samples.count;
    const positions = builder.upload('positions', 'float32x2', samples.positions);
    const sampleValues = builder.upload('values', 'float32', samples.values);
    const parameterBuffer = builder.allocate('parameters', 'float32', 12);
    const surface = builder.allocate('surface', 'float32', grid.width * grid.height);
    const variance = builder.allocate('variance', 'float32', grid.width * grid.height);
    const predictions = builder.allocate('predictions', 'float32', sampleCount);
    const predictionVariance = builder.allocate('prediction-variance', 'float32', sampleCount);
    const residuals = builder.allocate('residuals', 'float32', sampleCount);
    const standardized = builder.allocate('standardized', 'float32', sampleCount);
    const validCount = builder.allocate('valid-count', 'uint32', 1);
    const standardizedCount = builder.allocate('standardized-count', 'uint32', 1);
    const bias = builder.allocate('bias', 'float32', 1);
    const meanAbsoluteError = builder.allocate('mean-absolute-error', 'float32', 1);
    const rootMeanSquaredError = builder.allocate('root-mean-squared-error', 'float32', 1);
    addInterpolationDiagnosticsRecipe(builder.graph, {
      kriging: {
        positions: positions.view,
        values: sampleValues.view,
        parameters: parameterBuffer.view,
        grid: grid.grid,
        indexGridSize: [12, 8],
        indexBounds: [0, 0, grid.width * grid.cellSize, grid.worldHeight],
        maximumNeighborCount: 16
      },
      output: {
        values: surface.view,
        variance: variance.view,
        predictions: predictions.view,
        predictionVariance: predictionVariance.view,
        residuals: residuals.view,
        standardizedResiduals: standardized.view,
        validCount: validCount.view,
        standardizedCount: standardizedCount.view,
        bias: bias.view,
        meanAbsoluteError: meanAbsoluteError.view,
        rootMeanSquaredError: rootMeanSquaredError.view
      }
    });
    return {
      outputs: {surface, residuals, positions, rootMeanSquaredError},
      update: current =>
        parameterBuffer.write(
          getGPUKrigingParameterValues(
            {
              searchRadius: Infinity,
              neighborCount: 16,
              variogram: {
                model: MODELS[current.model],
                nugget: current.nugget,
                sill: 1,
                range: current.range
              }
            },
            grid.grid
          )
        )
    };
  },
  // The distance from a sample to the pointer is the variogram range: the ring is its reach.
  interaction: {
    kind: 'reach',
    hint: 'stretch the range',
    rest: ({toUnit, inputs}) =>
      toUnit([inputs.anchor[0] + parameters[1].default, inputs.anchor[1]]),
    resolve(pointer, {inputs, outputs, values, peek}) {
      const range = snapToParameter(
        findParameter(parameters, 'range'),
        Math.hypot(pointer.x - inputs.anchor[0], pointer.y - inputs.anchor[1])
      );
      const error = peek(outputs.rootMeanSquaredError);
      return {
        values: {range},
        glyphs: [
          {shape: 'ring', at: inputs.anchor, radius: range},
          {shape: 'line', from: inputs.anchor, to: [pointer.x, pointer.y], dashed: true},
          {shape: 'dot', at: inputs.anchor}
        ],
        readout: `variogram.range ${range.toFixed(2)} · ${MODELS[values.model]} · RMSE ${error ? error[0].toFixed(4) : '...'}`
      };
    }
  },
  draw: ({inputs, outputs}) => [
    {
      layer: 'raster',
      values: outputs.surface,
      columns: inputs.grid.width,
      rows: inputs.grid.height,
      range: [0, 1],
      smoothing: true
    },
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.residuals,
      ramp: 'diverging',
      range: [-0.4, 0.4],
      radius: 8
    }
  ]
});
