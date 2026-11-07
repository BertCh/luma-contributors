// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSpatialCrossValidation: assigns whole spatial blocks to folds and excludes a buffer around
// the test block from training. Left: each point colored by its fold. Right: fold 0, where orange
// points train the model and blue points are held out (the test blocks plus the buffer ring).

import {GPUSpatialCrossValidation} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createUniformPoints} from '../../inputs';
import {parameters} from './spatial-cross-validation.parameters';

const POINT_COUNT = 500;
const BLOCK_COLUMNS = 6;
const BLOCK_ROWS = 4;

export default defineOperationRun({
  parameters,
  caption: 'Spatial block folds (left) and fold 0 training set with its buffer (right).',
  createInputs: ({random, world}) => {
    const panelWidth = world.width / 2;
    const points = createUniformPoints(
      random,
      {width: panelWidth, height: world.height},
      POINT_COUNT,
      0
    );
    const blocks = new Uint32Array(POINT_COUNT);
    const mirrored = new Float32Array(POINT_COUNT * 2);
    for (let index = 0; index < POINT_COUNT; index++) {
      const column = Math.min(
        Math.floor((points.positions[index * 2] / panelWidth) * BLOCK_COLUMNS),
        BLOCK_COLUMNS - 1
      );
      const row = Math.min(
        Math.floor(points.positions[index * 2 + 1] * BLOCK_ROWS),
        BLOCK_ROWS - 1
      );
      blocks[index] = row * BLOCK_COLUMNS + column;
      mirrored[index * 2] = points.positions[index * 2] + panelWidth;
      mirrored[index * 2 + 1] = points.positions[index * 2 + 1];
    }
    return {positions: points.positions, mirrored, blocks};
  },
  build(builder, inputs, values) {
    const foldCount = values.foldCount;
    const folds = builder.allocate('folds', 'uint32', POINT_COUNT);
    const training = builder.allocate('training', 'uint32', foldCount * POINT_COUNT);
    const testing = builder.allocate('testing', 'uint32', foldCount * POINT_COUNT);
    const counts = builder.allocate('counts', 'uint32', foldCount * 3);
    const status = builder.allocate('status', 'uint32', foldCount);
    builder.graph.add(
      new GPUSpatialCrossValidation({
        positions: builder.upload('positions', 'float32x2', inputs.positions).view,
        blocks: builder.upload('blocks', 'uint32', inputs.blocks).view,
        foldCount,
        seed: values.seed,
        bufferDistance: values.bufferDistance,
        output: {
          folds: folds.view,
          training: training.view,
          testing: testing.view,
          counts: counts.view,
          status: status.view
        }
      })
    );
    return {
      outputs: {
        folds,
        training,
        left: builder.upload('left', 'float32x2', inputs.positions),
        right: builder.upload('right', 'float32x2', inputs.mirrored)
      }
    };
  },
  draw: ({outputs}) => [
    {layer: 'dots', positions: outputs.left, values: outputs.folds, count: POINT_COUNT, radius: 4},
    {
      layer: 'dots',
      positions: outputs.right,
      values: outputs.training,
      count: POINT_COUNT,
      radius: 4
    }
  ]
});
