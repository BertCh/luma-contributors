// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPULineSplit nodes a set of lines: wherever lines cross, every line is cut into pieces. A family
// of horizontal and a family of vertical lines cross; each output piece has its own color, and the
// muted lines are the unsplit inputs. Moving the vertical family moves the cuts.

import {GPULineSplit} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {createCrossingLines} from '../geometry-processing/shared/polygon-set';
import {defineOperationRun} from '../../define-run';
import {bindAxes} from '../../interaction';
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
    const lines = createCrossingLines(random, world, FAMILY_SIZE, FAMILY_SIZE, VERTICES);
    const {lineOffsets} = lines;
    const getPositions = (sweep: number, tilt: number) => {
      const positions = new Float32Array(lines.positions);
      for (let vertex = 0; vertex < lines.vertexCount; vertex++) {
        const x = positions[vertex * 2];
        const y = positions[vertex * 2 + 1];
        if (vertex < lines.firstVertexCount) {
          positions[vertex * 2 + 1] += (x / world.width - 0.5) * tilt * 0.3;
        } else {
          positions[vertex * 2] += (sweep - 0.3) * 0.25 * world.width + (y - 0.5) * tilt * 0.3;
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
  // The pointer carries the vertical family of input lines: x sweeps it sideways, y tilts both
  // families, and the cuts move with the crossings.
  interaction: bindAxes(parameters, {
    kind: 'drag',
    hint: 'drag the crossing lines',
    x: 'sweep',
    y: 'tilt'
  }),
  draw: ({outputs}) => [
    {layer: 'lines', context: true, segments: outputs.original, radius: 4},
    {
      layer: 'segments',
      nodes: outputs.points,
      edges: outputs.edges,
      values: outputs.lineIds,
      radius: 2
    }
  ]
});
