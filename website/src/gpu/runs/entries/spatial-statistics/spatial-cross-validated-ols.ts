// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSpatialCrossValidatedOLS: fits ordinary least squares on each fold's training blocks and
// predicts everything. Left: points colored by the fold-0 prediction. Right: the observed
// response. The bars are the fitted coefficients of every fold (intercept first); when they
// agree across folds the model is stable across space.

import {GPUSpatialCrossValidatedOLS} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createSmoothField, createUniformPoints} from '../../inputs';
import {CHART_BAND} from './shared/point-field';
import {parameters} from './spatial-cross-validated-ols.parameters';

const POINT_COUNT = 400;
const PREDICTOR_COUNT = 3;
const BLOCK_COLUMNS = 6;
const BLOCK_ROWS = 3;

export default defineOperationRun({
  parameters,
  caption: 'Fold-0 predictions, observed response and per-fold coefficients.',
  createInputs: ({random, world}) => {
    const mapHeight = 1 - CHART_BAND;
    const panelWidth = world.width / 2;
    const points = createUniformPoints(
      random,
      {width: panelWidth, height: mapHeight},
      POINT_COUNT,
      0
    );
    const field = createSmoothField(random, 3);
    const positions = Float32Array.from(points.positions, (value, index) =>
      index % 2 === 1 ? value + (1 - mapHeight) : value
    );
    const mirrored = Float32Array.from(positions, (value, index) =>
      index % 2 === 0 ? value + panelWidth : value
    );
    const predictors = new Float32Array(POINT_COUNT * PREDICTOR_COUNT);
    const blocks = new Uint32Array(POINT_COUNT);
    const noise = Float32Array.from({length: POINT_COUNT}, () => random() - 0.5);
    for (let row = 0; row < POINT_COUNT; row++) {
      const x = points.positions[row * 2] / panelWidth;
      const y = points.positions[row * 2 + 1] / mapHeight;
      predictors[row * PREDICTOR_COUNT] = field(x, y);
      predictors[row * PREDICTOR_COUNT + 1] = x;
      predictors[row * PREDICTOR_COUNT + 2] = y;
      blocks[row] =
        Math.min(Math.floor(y * BLOCK_ROWS), BLOCK_ROWS - 1) * BLOCK_COLUMNS +
        Math.min(Math.floor(x * BLOCK_COLUMNS), BLOCK_COLUMNS - 1);
    }
    return {positions, mirrored, predictors, blocks, noise};
  },
  build(builder, inputs, values) {
    const foldCount = values.foldCount;
    const response = builder.allocate('response', 'float32', POINT_COUNT);
    const folds = builder.allocate('folds', 'uint32', POINT_COUNT);
    const training = builder.allocate('training', 'uint32', foldCount * POINT_COUNT);
    const testing = builder.allocate('testing', 'uint32', foldCount * POINT_COUNT);
    const counts = builder.allocate('counts', 'uint32', foldCount * 3);
    const foldStatus = builder.allocate('fold-status', 'uint32', foldCount);
    const coefficients = builder.allocate(
      'coefficients',
      'float32',
      foldCount * (PREDICTOR_COUNT + 1)
    );
    const predictions = builder.allocate('predictions', 'float32', foldCount * POINT_COUNT);
    const fitStatus = builder.allocate('fit-status', 'uint32', foldCount);
    const metrics = builder.allocate('metrics', 'float32', foldCount * 6);
    const metricStatus = builder.allocate('metric-status', 'uint32', foldCount);
    builder.graph.add(
      new GPUSpatialCrossValidatedOLS({
        folds: {
          positions: builder.upload('positions', 'float32x2', inputs.positions).view,
          blocks: builder.upload('blocks', 'uint32', inputs.blocks).view,
          foldCount,
          seed: 3,
          bufferDistance: values.bufferDistance,
          output: {
            folds: folds.view,
            training: training.view,
            testing: testing.view,
            counts: counts.view,
            status: foldStatus.view
          }
        },
        predictors: builder.upload('predictors', 'float32', inputs.predictors).view,
        predictorCount: PREDICTOR_COUNT,
        response: response.view,
        output: {
          coefficients: coefficients.view,
          predictions: predictions.view,
          fitStatus: fitStatus.view,
          metrics: metrics.view,
          metricStatus: metricStatus.view
        }
      })
    );
    return {
      outputs: {
        response,
        predictions,
        coefficients,
        left: builder.upload('left', 'float32x2', inputs.positions),
        right: builder.upload('right', 'float32x2', inputs.mirrored)
      },
      update: current => {
        response.write(
          Float32Array.from({length: POINT_COUNT}, (_, row) => {
            const base = inputs.predictors.subarray(
              row * PREDICTOR_COUNT,
              (row + 1) * PREDICTOR_COUNT
            );
            return (
              0.2 +
              0.6 * base[0] +
              0.3 * base[1] -
              0.2 * base[2] +
              current.noise * inputs.noise[row]
            );
          })
        );
      }
    };
  },
  rebuildOn: ['foldCount', 'bufferDistance'],
  draw: ({outputs}) => [
    {
      layer: 'dots',
      positions: outputs.right,
      values: outputs.response,
      count: POINT_COUNT,
      radius: 3.5,
      range: [0, 1],
      context: true
    },
    {
      layer: 'dots',
      positions: outputs.left,
      values: outputs.predictions,
      count: POINT_COUNT,
      radius: 3.5,
      range: [0, 1]
    },
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.coefficients,
      ramp: 'diverging',
      range: [-1, 1],
      rectangle: [0.2, 0.04, 0.8, 0.3]
    }
  ]
});
