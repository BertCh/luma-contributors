// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTransferMask carrying a block selection (bright dots) up to the tracts that contain them,
// with no readback. In `any` mode a tract is lit when any of its blocks is selected; in `count`
// mode the fill is the number of selected blocks.

import {GPUTransferMask} from '@luma-contributors/analysis/computation-record';
import {defineOperationRun, type RunBuffer} from '../../define-run';
import {findParameter, snapToParameter} from '../../interaction';
import {getWorkplaceRows} from '../../../data/workplace-rows';
import {addIntegerToFloat} from './_lib/convert';
import {uploadTractGeometry} from './_lib/helpers';
import {parameters} from './transfer-mask.parameters';

function createInputs({
  world,
  random
}: {
  world: {width: number; height: number};
  random: () => number;
}) {
  return {...getWorkplaceRows(random, world), world};
}
type Inputs = ReturnType<typeof createInputs>;
type Outputs = ReturnType<typeof uploadTractGeometry> &
  Record<'blocks' | 'blockFlags' | 'tractFill', RunBuffer>;

export default defineOperationRun<Inputs, Outputs>({
  parameters,
  caption: 'Tracts touched by the selected blocks.',
  createInputs,
  build(builder, inputs, values) {
    const {tracts} = inputs;
    const blockFlags = builder.allocate('block-flags', 'uint32', inputs.jobs.length);
    const tractFlags = builder.allocate('tract-flags', 'uint32', tracts.count);
    builder.graph.add(
      new GPUTransferMask({
        id: 'carry-selection',
        source: blockFlags.view,
        relation: {
          layout: 'index',
          targets: builder.upload('tract-of-block', 'uint32', inputs.keys).view
        },
        target: tractFlags.view,
        mode: values.mode === 0 ? 'any' : 'count'
      })
    );
    return {
      outputs: {
        ...uploadTractGeometry(builder, tracts),
        blocks: builder.upload('blocks', 'float32x2', inputs.positions),
        blockFlags,
        tractFill: addIntegerToFloat(builder, 'tract-fill', tractFlags)
      },
      update: current => {
        blockFlags.write(
          Uint32Array.from({length: inputs.jobs.length}, (_, row) =>
            Math.hypot(
              inputs.positions[row * 2] - current.selectionCenter,
              inputs.positions[row * 2 + 1] - current.selectionCenterY
            ) < current.selectionRadius
              ? 1
              : 0
          )
        );
      }
    };
  },
  rebuildOn: ['mode'],
  // The pointer is the center of the circle that selects blocks (the operation's source flags).
  interaction: {
    kind: 'place',
    hint: 'move the selection',
    rest: ({values, world}) => [values.selectionCenter / world.width, 1 - values.selectionCenterY],
    resolve(pointer, {inputs, values}) {
      const centerX = snapToParameter(findParameter(parameters, 'selectionCenter'), pointer.x);
      const centerY = snapToParameter(findParameter(parameters, 'selectionCenterY'), pointer.y);
      let blockCount = 0;
      const tracts = new Set<number>();
      for (let row = 0; row < inputs.jobs.length; row++) {
        if (
          Math.hypot(inputs.positions[row * 2] - centerX, inputs.positions[row * 2 + 1] - centerY) <
          values.selectionRadius
        ) {
          blockCount++;
          tracts.add(inputs.keys[row]);
        }
      }
      return {
        values: {selectionCenter: centerX, selectionCenterY: centerY},
        glyphs: [
          {shape: 'ring', at: [centerX, centerY], radius: values.selectionRadius},
          {shape: 'crosshair', at: [centerX, centerY], size: 6}
        ],
        readout: `${values.mode === 0 ? 'any' : 'count'} · ${blockCount} blocks -> ${tracts.size} tracts`
      };
    }
  },
  draw: ({inputs, outputs, values}) => [
    {
      layer: 'polygons',
      positions: outputs.vertices,
      indices: outputs.indices,
      owners: outputs.owners,
      values: outputs.tractFill,
      triangleCount: inputs.tracts.triangleCount,
      range: [0, values.mode === 0 ? 1 : 25]
    },
    {layer: 'lines', context: true, segments: outputs.outline, count: inputs.tracts.outlineCount},
    {
      layer: 'dots',
      positions: outputs.blocks,
      values: outputs.blockFlags,
      radius: 1.6,
      alpha: 0.9
    }
  ]
});
