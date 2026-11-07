// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSharedPaths finds the stretches that two sets of lines share exactly. The left lines are muted,
// the right lines are orange; each right line follows its left partner for `overlap` vertices (the
// middle one in the opposite direction) and then detours. The colored runs are the shared stretches.

import {GPUSharedPaths} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {addPackedSegments, createLineSet, createPackedSegments} from './shared/lines';
import {parameters} from './shared-paths.parameters';

const LINE_COUNT = 3;
const VERTEX_COUNT = 10;

export default defineOperationRun({
  parameters,
  caption: 'Stretches shared exactly by two line sets, one color per shared run.',
  createInputs: ({random, world}) => {
    const left = createLineSet(random, world, LINE_COUNT, VERTEX_COUNT, 1.2);
    const getRight = (overlap: number, detour: number) => {
      const positions = new Float32Array(left.positions);
      for (let line = 0; line < LINE_COUNT; line++) {
        const first = line * VERTEX_COUNT;
        const reversed = line === 1;
        for (let vertex = 0; vertex < VERTEX_COUNT; vertex++) {
          const source = first + (reversed ? VERTEX_COUNT - 1 - vertex : vertex);
          const isShared = reversed ? vertex >= VERTEX_COUNT - overlap : vertex < overlap;
          positions[(first + vertex) * 2] = left.positions[source * 2];
          positions[(first + vertex) * 2 + 1] =
            left.positions[source * 2 + 1] + (isShared ? 0 : detour);
        }
      }
      return positions;
    };
    return {left, getRight};
  },
  build(builder, inputs, values) {
    const leftPositions = builder.upload('left-positions', 'float32x2', inputs.left.positions);
    const leftOffsets = builder.upload('left-offsets', 'uint32', inputs.left.lineOffsets);
    const initialRight = inputs.getRight(values.overlap, values.detour);
    const rightPositions = builder.upload('right-positions', 'float32x2', initialRight);
    const rightOffsets = builder.upload('right-offsets', 'uint32', inputs.left.lineOffsets);
    const runCapacity = 64;
    const vertexCapacity = 512;
    const leftLineIds = builder.allocate('run-left-lines', 'uint32', runCapacity);
    const rightLineIds = builder.allocate('run-right-lines', 'uint32', runCapacity);
    const forward = builder.allocate('run-forward', 'uint32', runCapacity);
    const offsets = builder.allocate('run-offsets', 'uint32', runCapacity + 1);
    const positions = builder.allocate('run-positions', 'float32x2', vertexCapacity);
    const count = builder.allocate('run-count', 'uint32', 1);
    const vertexCount = builder.allocate('run-vertex-count', 'uint32', 1);
    const overflow = builder.allocate('run-overflow', 'uint32', 1);
    const requiredCount = builder.allocate('run-required', 'uint32', 1);
    builder.graph.add(
      new GPUSharedPaths({
        left: {kind: 'lines', positions: leftPositions.view, lineOffsets: leftOffsets.view},
        right: {kind: 'lines', positions: rightPositions.view, lineOffsets: rightOffsets.view},
        intersectionCapacity: 1024,
        output: {
          runs: {
            leftLineIds: leftLineIds.view,
            rightLineIds: rightLineIds.view,
            forward: forward.view,
            offsets: offsets.view,
            positions: positions.view
          },
          status: {
            count: count.view,
            vertexCount: vertexCount.view,
            overflow: overflow.view,
            requiredCount: requiredCount.view
          }
        }
      })
    );
    const drawing = addPackedSegments(builder, 'shared', positions, offsets, {lineCount: count});
    const leftSegments = builder.upload(
      'left-segments',
      'float32x4',
      createPackedSegments(inputs.left.positions, inputs.left.lineOffsets, false)
    );
    const rightSegments = builder.allocate('right-segments', 'float32x4', leftSegments.length);
    return {
      outputs: {leftSegments, rightSegments, ...drawing},
      update: current => {
        const next = inputs.getRight(current.overlap, current.detour);
        rightPositions.write(next);
        rightSegments.write(createPackedSegments(next, inputs.left.lineOffsets, false));
      }
    };
  },
  draw: ({outputs}) => [
    {layer: 'lines', context: true, segments: outputs.leftSegments, radius: 4},
    {
      layer: 'lines',
      context: true,
      segments: outputs.rightSegments,
      color: [1, 0.58, 0.25, 0.7],
      radius: 1.8
    },
    {
      layer: 'segments',
      nodes: outputs.points,
      edges: outputs.edges,
      values: outputs.lineIds,
      radius: 3
    }
  ]
});
