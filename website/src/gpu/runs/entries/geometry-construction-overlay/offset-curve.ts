// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUOffsetCurve shifts every line sideways by a signed distance (positive is left of the line
// direction), with round, mitre or bevel corners on the outside of turns. The muted line is the
// input; the blue line is the offset the operation wrote.

import {
  getGPUOffsetCurveParameterValues,
  getGPUOffsetCurveRowsPerVertex,
  GPUOffsetCurve
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {
  addPackedSegments,
  createZigzagLines,
  createPackedSegments,
  OUTPUT_COLOR
} from './shared/lines';
import {parameters} from './offset-curve.parameters';

const JOIN_STYLES = ['round', 'mitre', 'bevel'] as const;
const QUAD_SEGMENTS = 8;

export default defineOperationRun({
  parameters,
  caption: 'Offset curves of lines; the muted line is the input.',
  createInputs: ({random, world}) => createZigzagLines(random, world, 2, 9, 0.14),
  build(builder, lines, values) {
    const joinStyle = JOIN_STYLES[values.joinStyle];
    const rowsPerVertex = getGPUOffsetCurveRowsPerVertex(joinStyle, QUAD_SEGMENTS);
    const vertexCount = lines.positions.length / 2;
    const positions = builder.upload('positions', 'float32x2', lines.positions);
    const lineOffsets = builder.upload('line-offsets', 'uint32', lines.lineOffsets);
    const offsetPositions = builder.allocate(
      'offset-positions',
      'float32x2',
      vertexCount * rowsPerVertex
    );
    const parameterBuffer = builder.allocate('offset-parameters', 'float32', 4);
    builder.graph.add(
      new GPUOffsetCurve({
        positions: positions.view,
        lineOffsets: lineOffsets.view,
        joinStyle,
        quadSegments: QUAD_SEGMENTS,
        parameters: parameterBuffer.view,
        output: {positions: offsetPositions.view}
      })
    );
    // Vertex i owns rows [i * k, (i + 1) * k), so line offsets scale by k.
    const blockOffsets = builder.upload(
      'block-offsets',
      'uint32',
      lines.lineOffsets.map(offset => offset * rowsPerVertex)
    );
    const drawing = addPackedSegments(builder, 'offset', offsetPositions, blockOffsets);
    const original = builder.upload(
      'original',
      'float32x4',
      createPackedSegments(lines.positions, lines.lineOffsets, false)
    );
    return {
      outputs: {original, ...drawing},
      update: current =>
        parameterBuffer.write(
          getGPUOffsetCurveParameterValues({
            distance: current.distance,
            mitreLimit: current.mitreLimit
          })
        )
    };
  },
  rebuildOn: ['joinStyle'],
  draw: ({outputs}) => [
    {layer: 'lines', context: true, segments: outputs.original, radius: 3},
    {layer: 'lines', segments: outputs.segments, color: OUTPUT_COLOR, radius: 2}
  ]
});
