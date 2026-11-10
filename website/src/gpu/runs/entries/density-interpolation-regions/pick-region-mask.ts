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
    points: createUniformPoints(random, world, 600, 0.2),
    // The pick position is owned by the pointer; it rests at the middle of the picture.
    center: [0.5 * world.width, 0.5] as [number, number]
  }),
  build(builder, {points, center}, current) {
    // The pick result: [pair count, reserved, (object index, batch)...] for every point within
    // the pick radius of the pointer; point i belongs to layer i % 3. The buffer has room for
    // every point, so the pointer rewrites it without a rebuild.
    const picks = builder.allocate('picks', 'uint32', 2 + points.count * 2);
    const mask = builder.allocate('mask', 'uint32', points.count);
    builder.graph.add(
      new GPUPickRegionMask({
        picks: picks.view,
        batchIndex: current.batchIndex,
        output: {
          mask: mask.view,
          status: {overflow: builder.allocate('overflow', 'uint32', 1).view}
        }
      })
    );
    const batches = Uint32Array.from({length: points.count}, (_, row) => row % BATCH_COUNT);
    return {
      update: values => {
        const picked: number[] = [];
        for (let row = 0; row < points.count; row++) {
          const distance = Math.hypot(
            points.positions[row * 2] - center[0],
            points.positions[row * 2 + 1] - center[1]
          );
          if (distance < values.pickRadius) picked.push(row, row % BATCH_COUNT);
        }
        picks.write(Uint32Array.from([picked.length / 2, 0, ...picked]));
      },
      outputs: {
        mask,
        batches: builder.upload('batches', 'uint32', batches),
        positions: builder.upload('points', 'float32x2', points.positions)
      }
    };
  },
  rebuildOn: ['batchIndex'],
  // The pointer is the pick position; the pick radius is the ring around it.
  interaction: {
    kind: 'place',
    hint: 'move the pick',
    rest: [0.5, 0.5],
    resolve(pointer, {inputs, outputs, values, peek}) {
      inputs.center[0] = pointer.x;
      inputs.center[1] = pointer.y;
      const mask = peek(outputs.mask);
      let pickedCount = 0;
      if (mask) for (const flag of mask) pickedCount += flag;
      return {
        rerun: true,
        glyphs: [{shape: 'ring', at: inputs.center, radius: values.pickRadius}],
        readout: `batchIndex ${values.batchIndex} · ${mask ? pickedCount : '...'} rows picked`
      };
    }
  },
  draw: ({outputs}) => [
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.batches,
      categories: BATCH_COLORS,
      radius: 3.4,
      alpha: 0.85
    },
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.mask,
      categories: HALO_COLORS,
      radius: 7
    },
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.batches,
      categories: BATCH_COLORS,
      radius: 3.4
    }
  ]
});
