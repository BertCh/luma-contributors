// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUNestedInterpolationValidation: for every outer fold, picks the interpolation settings (a
// kriging model or Gaussian RBF bandwidths) by an inner cross-validation on the training blocks
// only, then predicts the held-out outer block. Each dot is a held-out point: the observed
// response (x) against the nested prediction (y). Points on the diagonal were predicted well.

import {
  GPUNestedInterpolationValidation,
  GPUSpatialCrossValidation
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createSmoothField, createUniformPoints} from '../../inputs';
import {parameters} from './nested-interpolation-validation.parameters';

const POINT_COUNT = 240;
const PLOT = [0.12, 0.1, 0.88, 0.92] as const;

export default defineOperationRun({
  parameters,
  caption: 'Held-out response against nested-validation predictions.',
  createInputs: ({random, world}) => {
    const points = createUniformPoints(random, world, POINT_COUNT, 0);
    const field = createSmoothField(random, 3);
    const signal = Float32Array.from({length: POINT_COUNT}, (_, row) =>
      field(points.positions[row * 2] / world.width, points.positions[row * 2 + 1])
    );
    const outerBlocks = new Uint32Array(POINT_COUNT);
    const innerBlocks = new Uint32Array(POINT_COUNT);
    for (let row = 0; row < POINT_COUNT; row++) {
      const column = Math.min(Math.floor((points.positions[row * 2] / world.width) * 5), 4);
      const line = Math.min(Math.floor(points.positions[row * 2 + 1] * 3), 2);
      outerBlocks[row] = line * 5 + column;
      innerBlocks[row] =
        Math.min(Math.floor((points.positions[row * 2] / world.width) * 10), 9) +
        10 * Math.min(Math.floor(points.positions[row * 2 + 1] * 6), 5);
    }
    return {
      positions: points.positions,
      signal,
      outerBlocks,
      innerBlocks,
      noise: Float32Array.from({length: POINT_COUNT}, () => random() - 0.5),
      diagonal: Float32Array.of(
        PLOT[0] * world.width,
        PLOT[1] * world.height,
        PLOT[2] * world.width,
        PLOT[3] * world.height
      )
    };
  },
  build(builder, inputs, values) {
    const foldCount = values.foldCount;
    const positions = builder.upload('positions', 'float32x2', inputs.positions);
    const response = builder.allocate('response', 'float32', POINT_COUNT);
    const folds = builder.allocate('folds', 'uint32', POINT_COUNT);
    const training = builder.allocate('training', 'uint32', foldCount * POINT_COUNT);
    const testing = builder.allocate('testing', 'uint32', foldCount * POINT_COUNT);
    const counts = builder.allocate('counts', 'uint32', foldCount * 3);
    const foldStatus = builder.allocate('fold-status', 'uint32', foldCount);
    const candidates = [
      {mode: 'rbf', kernel: 'gaussian', epsilon: 2, smoothing: 0.1},
      {mode: 'rbf', kernel: 'gaussian', epsilon: 6, smoothing: 0.1},
      {mode: 'rbf', kernel: 'gaussian', epsilon: 18, smoothing: 0.1}
    ] as const;
    const selected = builder.allocate('selected', 'uint32', foldCount);
    const scores = builder.allocate('scores', 'float32', foldCount * candidates.length * 3);
    const predictions = builder.allocate('predictions', 'float32', foldCount * POINT_COUNT);
    const metrics = builder.allocate('metrics', 'float32', foldCount * 6);
    const status = builder.allocate('status', 'uint32', foldCount);
    // The response repeated once per fold, to sit under the fold-major predictions.
    const observed = builder.allocate('observed', 'float32', foldCount * POINT_COUNT);
    builder.graph.add(
      new GPUSpatialCrossValidation({
        id: 'outer-folds',
        positions: positions.view,
        blocks: builder.upload('outer-blocks', 'uint32', inputs.outerBlocks).view,
        foldCount,
        seed: 5,
        output: {
          folds: folds.view,
          training: training.view,
          testing: testing.view,
          counts: counts.view,
          status: foldStatus.view
        }
      })
    );
    builder.graph.add(
      new GPUNestedInterpolationValidation({
        positions: positions.view,
        response: response.view,
        training: training.view,
        testing: testing.view,
        foldCount,
        innerBlocks: builder.upload('inner-blocks', 'uint32', inputs.innerBlocks).view,
        innerFoldCount: 3,
        seed: 2,
        neighborCount: values.neighborCount,
        candidates,
        output: {
          selected: selected.view,
          scores: scores.view,
          predictions: predictions.view,
          metrics: metrics.view,
          status: status.view
        }
      })
    );
    return {
      outputs: {
        predictions,
        observed,
        response,
        diagonal: builder.upload('diagonal', 'float32x4', inputs.diagonal)
      },
      update: current => {
        const values = Float32Array.from(
          inputs.signal,
          (value, row) => value + current.noise * inputs.noise[row]
        );
        response.write(values);
        observed.write(
          Float32Array.from(
            {length: foldCount * POINT_COUNT},
            (_, index) => values[index % POINT_COUNT]
          )
        );
      }
    };
  },
  rebuildOn: ['foldCount', 'neighborCount'],
  draw: ({outputs}) => [
    {layer: 'lines', context: true, segments: outputs.diagonal, radius: 1},
    {
      layer: 'chart',
      kind: 'scatter',
      values: outputs.predictions,
      xValues: outputs.observed,
      xRange: [0, 1],
      range: [0, 1],
      color: [0.4, 0.68, 1, 0.85],
      radius: 4,
      rectangle: PLOT
    }
  ]
});
