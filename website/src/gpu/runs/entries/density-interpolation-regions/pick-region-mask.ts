// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUPickRegionMask turns a pick result (object index, layer batch pairs) into a 0/1 mask over
// the rows of one layer. Points of three layers are drawn in three colors; the bright halos mark
// the picked rows of the chosen `batchIndex` only. The pick list is built on the CPU, as a pointer
// pick would deliver it; the mask is the GPU output.

import {GPUPickRegionMask} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createUniformPoints} from '../../inputs';
import {parameters} from './pick-region-mask.parameters';

const BATCH_COUNT = 3;
// One fixed color per layer batch; picked rows get a bright halo and are redrawn on top as rings.
const BATCH_COLORS = [
  [0.15, 0.4, 0.85],
  [0.9, 0.45, 0.1],
  [0.1, 0.65, 0.55]
] as const;
const HALO_COLORS = [
  [0, 0, 0, 0],
  [1, 0.85, 0.15, 0.95]
] as const;

export default defineOperationRun({
  parameters,
  caption: 'Rows of the chosen layer that a pick touched.',
  createInputs: ({random, world}) => ({
    world,
    points: createUniformPoints(random, world, 900, 0.2)
  }),
  build(builder, {world, points}, current) {
    // The pick result: [pair count, reserved, (object index, batch)...] for every point within
    // the pick radius of the pointer; point i belongs to layer i % 3.
    const picked: number[] = [];
    for (let row = 0; row < points.count; row++) {
      const distance = Math.hypot(
        points.positions[row * 2] - 0.5 * world.width,
        points.positions[row * 2 + 1] - 0.5
      );
      if (distance < current.pickRadius) picked.push(row, row % BATCH_COUNT);
    }
    const picks = Uint32Array.from([picked.length / 2, 0, ...picked]);
    const mask = builder.allocate('mask', 'uint32', points.count);
    builder.graph.add(
      new GPUPickRegionMask({
        picks: builder.upload('picks', 'uint32', picks).view,
        batchIndex: current.batchIndex,
        output: {
          mask: mask.view,
          status: {overflow: builder.allocate('overflow', 'uint32', 1).view}
        }
      })
    );
    const batches = Uint32Array.from({length: points.count}, (_, row) => row % BATCH_COUNT);
    return {
      outputs: {
        mask,
        batches: builder.upload('batches', 'uint32', batches),
        positions: builder.upload('points', 'float32x2', points.positions)
      }
    };
  },
  draw: ({outputs}) => [
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.batches,
      categories: BATCH_COLORS,
      radius: 4.5,
      alpha: 0.85
    },
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.mask,
      categories: HALO_COLORS,
      radius: 9
    },
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.batches,
      categories: BATCH_COLORS,
      radius: 4.5
    }
  ]
});
