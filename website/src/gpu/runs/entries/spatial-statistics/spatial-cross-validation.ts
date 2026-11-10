// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSpatialCrossValidation: assigns whole spatial blocks to folds and excludes a buffer around
// the test block from training. Left: each point colored by its fold. Right: one fold's split,
// where one color trains the model, one is the test block and one is the buffer ring that is
// excluded from training. Hovering a point picks its fold (fold 1 until then).

import {GPUSpatialCrossValidation} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createUniformPoints} from '../../inputs';
import {chooseSticky, createPointIndex} from '../../pick';
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
    return {
      positions: points.positions,
      mirrored,
      blocks,
      index: createPointIndex(points.positions),
      focus: {row: -1},
      panelWidth
    };
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
        testing,
        counts,
        foldCount,
        // Render-only: the shown fold's split per point: 0 buffered out, 1 training, 2 test.
        split: builder.allocate('split', 'uint32', POINT_COUNT),
        left: builder.upload('left', 'float32x2', inputs.positions),
        right: builder.upload('right', 'float32x2', inputs.mirrored)
      }
    };
  },
  interaction: {
    kind: 'probe',
    hint: 'hover a point',
    rest: [0.25, 0.5],
    resolve(pointer, {inputs, outputs, peek}) {
      const folds = peek(outputs.folds);
      const training = peek(outputs.training);
      const testing = peek(outputs.testing);
      const counts = peek(outputs.counts);
      // Either panel picks a point: the right panel is the left one shifted by its width.
      const x = pointer.x >= inputs.panelWidth ? pointer.x - inputs.panelWidth : pointer.x;
      const row = chooseSticky(inputs.index, x, pointer.y, inputs.focus.row);
      inputs.focus.row = row;
      if (!folds || !training || !testing) return {};
      const fold = row >= 0 ? folds[row] : 0;
      const split = new Uint32Array(POINT_COUNT);
      for (let point = 0; point < POINT_COUNT; point++) {
        split[point] =
          testing[fold * POINT_COUNT + point] === 1 ? 2 : training[fold * POINT_COUNT + point];
      }
      outputs.split.write(split);
      const glyphs =
        row < 0
          ? []
          : ([
              {
                shape: 'crosshair',
                at: [inputs.positions[row * 2], inputs.positions[row * 2 + 1]],
                size: 12
              },
              {
                shape: 'crosshair',
                at: [inputs.mirrored[row * 2], inputs.mirrored[row * 2 + 1]],
                size: 12
              }
            ] as const);
      return {
        glyphs,
        readout: `fold ${fold + 1}/${outputs.foldCount} · ${counts ? counts[fold * 3 + 1] : '…'} test · ${
          counts ? counts[fold * 3 + 2] : '…'
        } buffered out`
      };
    }
  },
  draw: ({outputs}) => [
    {layer: 'dots', positions: outputs.left, values: outputs.folds, count: POINT_COUNT, radius: 4},
    {
      layer: 'dots',
      positions: outputs.right,
      values: outputs.split,
      count: POINT_COUNT,
      radius: 4
    }
  ]
});
