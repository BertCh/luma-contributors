// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPULineSplit nodes a set of lines: wherever lines cross, every line is cut into pieces. A family
// of horizontal and a family of vertical lines cross; each output piece has its own color, and the
// muted lines are the unsplit inputs. Moving the vertical family moves the cuts.

import {GPULineSplit} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {addPackedSegments, createPackedSegments} from './shared/lines';
import {parameters} from './line-split.parameters';

const FAMILY_SIZE = 4;
const VERTICES = 6;
const PIECE_CAPACITY = 256;
const VERTEX_CAPACITY = 2048;

export default defineOperationRun({
  parameters,
  caption: 'Lines cut at every crossing; each piece gets its own color.',
  createInputs: ({random, world}) => {
    const jitter = Array.from({length: FAMILY_SIZE * 2 * VERTICES}, () => random() - 0.5);
    const lineOffsets = Uint32Array.from(
      {length: FAMILY_SIZE * 2 + 1},
      (_, line) => line * VERTICES
    );
    const getPositions = (sweep: number, tilt: number) => {
      const positions = new Float32Array(FAMILY_SIZE * 2 * VERTICES * 2);
      for (let line = 0; line < FAMILY_SIZE * 2; line++) {
        const isHorizontal = line < FAMILY_SIZE;
        const slot = (isHorizontal ? line : line - FAMILY_SIZE) + 0.5;
        for (let vertex = 0; vertex < VERTICES; vertex++) {
          const along = vertex / (VERTICES - 1);
          const noise = jitter[line * VERTICES + vertex] * 0.07;
          const row = line * VERTICES + vertex;
          if (isHorizontal) {
            const baseY = (slot / FAMILY_SIZE) * world.height * 0.9 + world.height * 0.05;
            positions[row * 2] = (0.03 + 0.94 * along) * world.width;
            positions[row * 2 + 1] = baseY + noise + (along - 0.5) * tilt * 0.3;
          } else {
            const baseX =
              ((slot / FAMILY_SIZE + sweep * 0.25) % 1) * world.width * 0.94 + 0.03 * world.width;
            positions[row * 2] = baseX + noise + (along - 0.5) * tilt * 0.3;
            positions[row * 2 + 1] = (0.03 + 0.94 * along) * world.height;
          }
        }
      }
      return positions;
    };
    return {lineOffsets, getPositions};
  },
  build(builder, inputs, values) {
    const initial = inputs.getPositions(values.sweep, values.tilt);
    const positions = builder.upload('positions', 'float32x2', initial);
    const lineOffsets = builder.upload('line-offsets', 'uint32', inputs.lineOffsets);
    const pieceSources = builder.allocate('piece-sources', 'uint32', PIECE_CAPACITY);
    const pieceOffsets = builder.allocate('piece-offsets', 'uint32', PIECE_CAPACITY + 1);
    const piecePositions = builder.allocate('piece-positions', 'float32x2', VERTEX_CAPACITY);
    const count = builder.allocate('piece-count', 'uint32', 1);
    const overflow = builder.allocate('piece-overflow', 'uint32', 1);
    const requiredCount = builder.allocate('piece-required', 'uint32', 1);
    const vertexCount = builder.allocate('piece-vertex-count', 'uint32', 1);
    const requiredVertexCount = builder.allocate('piece-required-vertices', 'uint32', 1);
    builder.graph.add(
      new GPULineSplit({
        lines: {kind: 'lines', positions: positions.view, lineOffsets: lineOffsets.view},
        intersectionCapacity: 512,
        output: {
          pieces: {
            geometry: {
              kind: 'lines',
              positions: piecePositions.view,
              lineOffsets: pieceOffsets.view
            },
            sourceIds: pieceSources.view,
            status: {
              count: count.view,
              overflow: overflow.view,
              requiredCount: requiredCount.view,
              vertexCount: vertexCount.view,
              requiredVertexCount: requiredVertexCount.view
            }
          }
        }
      })
    );
    const drawing = addPackedSegments(builder, 'pieces', piecePositions, pieceOffsets, {
      lineCount: count
    });
    const original = builder.allocate('original', 'float32x4', FAMILY_SIZE * 2 * (VERTICES - 1));
    return {
      outputs: {original, ...drawing},
      update: current => {
        const next = inputs.getPositions(current.sweep, current.tilt);
        positions.write(next);
        original.write(createPackedSegments(next, inputs.lineOffsets, false));
      }
    };
  },
  draw: ({outputs}) => [
    {layer: 'lines', context: true, segments: outputs.original, radius: 4},
    {
      layer: 'segments',
      nodes: outputs.points,
      edges: outputs.edges,
      values: outputs.lineIds,
      radius: 2
    },
    {layer: 'dots', positions: outputs.points, values: outputs.lineIds, radius: 3}
  ]
});
