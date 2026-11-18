// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSegmentIntersection finds every crossing between west-east lines (left) and south-north
// lines (right). The dots are the intersection points the operation wrote, up to its pair count.

import {GPUSegmentIntersection} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {makeRandom} from '../../../demos/kit';
import {defineOperationRun} from '../../define-run';
import {createCrossingLines, pickCrossingLines} from './shared/polygon-set';
import {addMaskedPoints} from './shared/shape-segments';
import {parameters} from './segment-intersection.parameters';

const CAPACITY = 256;

export default defineOperationRun({
  parameters,
  caption: 'Every crossing between two sets of lines, found in one pass.',
  createInputs: ({seed, world}) => ({
    seed,
    world,
    lines: createCrossingLines(makeRandom(seed), world, 5, 6, 12, 0.07),
    shifts: [] as Array<{line: number; offset: number}>
  }),
  build(builder, inputs, values) {
    const lines = createCrossingLines(
      makeRandom(inputs.seed),
      inputs.world,
      5,
      6,
      12,
      values.wobble
    );
    inputs.lines = lines;
    inputs.shifts = [];
    const leftPositions = lines.positions.slice(0, lines.firstVertexCount * 2);
    const leftOffsets = lines.lineOffsets.slice(0, lines.firstLineCount + 1);
    const rightPositions = lines.positions.slice(lines.firstVertexCount * 2);
    const rightOffsets = lines.lineOffsets
      .slice(lines.firstLineCount)
      .map(offset => offset - lines.firstVertexCount);
    const left = {
      kind: 'lines' as const,
      positions: builder.upload('left-positions', 'float32x2', leftPositions),
      lineOffsets: builder.upload('left-offsets', 'uint32', leftOffsets)
    };
    const right = {
      kind: 'lines' as const,
      positions: builder.upload('right-positions', 'float32x2', rightPositions),
      lineOffsets: builder.upload('right-offsets', 'uint32', rightOffsets)
    };
    const leftRows = builder.allocate('left-ids', 'uint32', CAPACITY);
    const rightRows = builder.allocate('right-ids', 'uint32', CAPACITY);
    const count = builder.allocate('count', 'uint32', 1);
    const overflow = builder.allocate('overflow', 'uint32', 1);
    const requiredCount = builder.allocate('required-count', 'uint32', 1);
    const points = builder.allocate('points', 'float32x2', CAPACITY);
    builder.graph.add(
      new GPUSegmentIntersection({
        left: {
          kind: 'lines',
          positions: left.positions.view,
          lineOffsets: left.lineOffsets.view
        },
        ...(values.right === 1
          ? {
              right: {
                kind: 'lines',
                positions: right.positions.view,
                lineOffsets: right.lineOffsets.view
              }
            }
          : {}),
        output: {
          pairs: {
            leftRows: leftRows.view,
            rightRows: rightRows.view,
            status: {count: count.view, overflow: overflow.view, requiredCount: requiredCount.view}
          },
          points: points.view
        }
      })
    );
    const leftSegments = builder.upload('left-segments', 'float32x4', lines.firstSegments);
    const rightSegments = builder.upload('right-segments', 'float32x4', lines.secondSegments);
    return {
      update: () => {
        lines.regenerate(inputs.shifts);
        left.positions.write(lines.positions.slice(0, lines.firstVertexCount * 2));
        right.positions.write(lines.positions.slice(lines.firstVertexCount * 2));
        leftSegments.write(lines.firstSegments);
        rightSegments.write(lines.secondSegments);
      },
      outputs: {
        leftSegments,
        rightSegments,
        count,
        crossings: addMaskedPoints(builder, 'crossings', points, count)
      }
    };
  },
  rebuildOn: ['right', 'wobble'],
  // The pointer is where one left and one right line cross: every crossing slides with them.
  interaction: {
    kind: 'drag',
    hint: 'move the crossing',
    rest: ({inputs, toUnit}) => {
      const {lines} = inputs;
      return toUnit([
        lines.lineBases[
          lines.firstLineCount + Math.floor((lines.lineCount - lines.firstLineCount) / 2)
        ],
        lines.lineBases[Math.floor(lines.firstLineCount / 2)]
      ]);
    },
    resolve(pointer, {inputs, outputs, peek}) {
      inputs.shifts = pickCrossingLines(inputs.lines, pointer.x, pointer.y, inputs.shifts);
      const count = peek(outputs.count)?.[0];
      return {
        rerun: true,
        glyphs: [{shape: 'crosshair', at: [pointer.x, pointer.y]}],
        readout: `${count ?? '...'} crossings`
      };
    }
  },
  draw: ({outputs}) => [
    {layer: 'lines', context: true, segments: outputs.leftSegments, color: [0.4, 0.75, 1, 1]},
    {layer: 'lines', context: true, segments: outputs.rightSegments, color: [0.78, 0.57, 0.92, 1]},
    {layer: 'dots', positions: outputs.crossings, radius: 9, color: [1, 0.6, 0.2, 1]}
  ]
});
