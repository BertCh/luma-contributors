// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUKeyJoin of block points (left, keyed by tract) to a tract table (right). Matched blocks take
// the tract's gathered value; blocks whose key is missing from the right table stay muted gray.

import {GPUKeyJoin} from '@luma-contributors/analysis/dataframe-analysis';
import {defineOperationRun, type RunBuffer} from '../../define-run';
import type {RunGlyph} from '../../interaction';
import {chooseSticky, createPointIndex} from '../../pick';
import {getWorkplaceRows} from '../../../data/workplace-rows';
import {createTractProbe, hashToUnit, uploadTractGeometry, writeRowPositions} from './_lib/helpers';
import {parameters} from './key-join.parameters';

const getRightKept = (tract: number, share: number) => hashToUnit(tract) * 100 < share;
const getLeftKept = (row: number, share: number) => hashToUnit(row + 5000) * 100 < share;

function createInputs({
  world,
  random
}: {
  world: {width: number; height: number};
  random: () => number;
}) {
  const rows = getWorkplaceRows(random, world);
  const probe = createTractProbe(rows.tracts, rows.keys);
  return {
    ...rows,
    world,
    probe,
    index: createPointIndex(rows.positions),
    focus: {row: -1, tract: probe.richest}
  };
}
type Inputs = ReturnType<typeof createInputs>;
type Outputs = ReturnType<typeof uploadTractGeometry> &
  Record<'joined' | 'blocks' | 'tractValues' | 'links' | 'members', RunBuffer>;

export default defineOperationRun<Inputs, Outputs>({
  parameters,
  caption:
    'Blocks colored by the value joined from their tract; tracts missing from the right table stay empty.',
  createInputs,
  build(builder, inputs) {
    const {tracts} = inputs;
    const rightMask = builder.allocate('right-mask', 'uint32', tracts.count);
    const leftMask = builder.allocate('left-mask', 'uint32', inputs.jobs.length);
    const joined = builder.allocate('joined-value', 'float32', inputs.jobs.length);
    builder.graph.add(
      new GPUKeyJoin({
        leftKeys: builder.upload('left-keys', 'uint32', inputs.keys).view,
        rightKeys: builder.upload(
          'right-keys',
          'uint32',
          Uint32Array.from({length: tracts.count}, (_, row) => row)
        ).view,
        leftMask: leftMask.view,
        rightMask: rightMask.view,
        kind: 'left',
        gather: [
          {
            column: builder.upload('right-value', 'float32', Float32Array.from(tracts.values)).view,
            output: joined.view
          }
        ],
        output: {
          matched: builder.allocate('matched', 'uint32', inputs.jobs.length).view,
          rightRows: builder.allocate('right-rows', 'uint32', inputs.jobs.length).view
        }
      })
    );
    const tractValues = builder.allocate('tract-values', 'float32', tracts.count);
    const memberLimit = Math.max(...inputs.probe.members.map(rows => rows.length));
    return {
      outputs: {
        ...uploadTractGeometry(builder, tracts),
        joined,
        blocks: builder.upload('blocks', 'float32x2', inputs.positions),
        tractValues,
        links: builder.allocate('links', 'float32x4', memberLimit),
        members: builder.allocate('members', 'float32x2', memberLimit)
      },
      update: current => {
        rightMask.write(
          Uint32Array.from({length: tracts.count}, (_, row) =>
            getRightKept(row, current.rightMask) ? 1 : 0
          )
        );
        leftMask.write(
          Uint32Array.from({length: inputs.jobs.length}, (_, row) =>
            getLeftKept(row, current.leftMask) ? 1 : 0
          )
        );
        // Tracts present in the right table carry their value faintly; the others stay empty.
        tractValues.write(
          Float32Array.from({length: tracts.count}, (_, row) =>
            getRightKept(row, current.rightMask) ? tracts.values[row] : Number.NaN
          )
        );
      }
    };
  },
  rebuildOn: [],
  // The pointer is a block or a tract: the blocks keyed to the tract are linked to its right-table
  // row (the tract center), and the readout says whether the join matched.
  interaction: {
    kind: 'probe',
    hint: 'hover a block or tract',
    rest: ({inputs}) => {
      const center = inputs.tracts.centers.subarray(
        inputs.probe.richest * 2,
        inputs.probe.richest * 2 + 2
      );
      return [center[0] / inputs.world.width, 1 - center[1] / inputs.world.height];
    },
    resolve(pointer, {inputs, outputs, values, pixelSize, peek}) {
      const block = chooseSticky(inputs.index, pointer.x, pointer.y, inputs.focus.row);
      const isBlock =
        block >= 0 && inputs.index.distanceTo(block, pointer.x, pointer.y) < pixelSize * 8;
      let tract = isBlock ? inputs.keys[block] : inputs.probe.findTract(pointer.x, pointer.y);
      if (tract < 0) tract = inputs.focus.tract;
      inputs.focus.tract = tract;
      inputs.focus.row = isBlock ? block : -1;
      const center = [
        inputs.tracts.centers[tract * 2],
        inputs.tracts.centers[tract * 2 + 1]
      ] as const;
      const rows = isBlock ? [block] : inputs.probe.members[tract];
      const count = writeRowPositions(outputs.members, inputs.positions, rows);
      const links = new Float32Array(outputs.links.length * 4);
      for (let index = 0; index < count; index++) {
        links.set(
          [
            inputs.positions[rows[index] * 2],
            inputs.positions[rows[index] * 2 + 1],
            center[0],
            center[1]
          ],
          index * 4
        );
      }
      outputs.links.write(links);
      const hasRightRow = getRightKept(tract, values.rightMask);
      const joinedValue = isBlock ? peek(outputs.joined)?.[block] : undefined;
      const status = isBlock
        ? !getLeftKept(block, values.leftMask)
          ? 'masked out'
          : hasRightRow
            ? 'matched'
            : 'no right row'
        : hasRightRow
          ? 'right row present'
          : 'no right row';
      const glyphs: RunGlyph[] = [
        {shape: 'polyline', points: inputs.probe.getRing(tract), closed: true},
        {shape: 'crosshair', at: center, size: 7}
      ];
      return {
        focus: count,
        glyphs,
        readout: isBlock
          ? `block ${block} -> tract ${inputs.tracts.geoids[tract].slice(-6)} · ${
              joinedValue !== undefined && hasRightRow ? joinedValue.toFixed(2) : '-'
            } (${status})`
          : `tract ${inputs.tracts.geoids[tract].slice(-6)} · ${inputs.probe.members[tract].length} blocks · ${status}`
      };
    }
  },
  draw: ({inputs, outputs, interaction}) => [
    {
      layer: 'polygons',
      positions: outputs.vertices,
      indices: outputs.indices,
      owners: outputs.owners,
      values: outputs.tractValues,
      triangleCount: inputs.tracts.triangleCount,
      range: [0, 1],
      alpha: 0.35
    },
    {layer: 'lines', context: true, segments: outputs.outline, count: inputs.tracts.outlineCount},
    {layer: 'dots', positions: outputs.blocks, values: outputs.joined, range: [0, 1], radius: 2.6},
    {
      layer: 'lines',
      segments: outputs.links,
      count: typeof interaction.focus === 'number' ? Math.max(1, interaction.focus) : 1,
      color: [1, 0.75, 0.2, 0.7]
    },
    {
      layer: 'dots',
      positions: outputs.members,
      count: typeof interaction.focus === 'number' ? Math.max(1, interaction.focus) : 1,
      color: [1, 0.75, 0.2, 1],
      radius: 3.4
    }
  ]
});
