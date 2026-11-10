// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUOffsetCurve shifts every line sideways by a signed distance (positive is left of the line
// direction), with round, mitre or bevel corners on the outside of turns. The muted line is the
// input; the blue line is the offset the operation wrote.

import {createRiverBend} from '../../../data/hydrography';
import {
  getGPUOffsetCurveParameterValues,
  getGPUOffsetCurveRowsPerVertex,
  GPUOffsetCurve
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {findParameter, snapToParameter} from '../../interaction';
import {
  addPackedSegments,
  createPackedSegments,
  findNearestLine,
  findRestBesideLines,
  OUTPUT_COLOR
} from './shared/lines';
import {parameters} from './offset-curve.parameters';

const JOIN_STYLES = ['round', 'mitre', 'bevel'] as const;
const QUAD_SEGMENTS = 8;

export default defineOperationRun({
  parameters,
  caption: 'Offset curves of lines; the muted line is the input.',
  createInputs: ({world}) => {
    const vertexCount = 7;
    const lines = [
      createRiverBend(0, vertexCount, world, 0, 2),
      createRiverBend(1, vertexCount, world, 1, 2)
    ];
    return {
      world,
      positions: Float32Array.from(lines.flat(2)),
      lineOffsets: Uint32Array.of(0, vertexCount, vertexCount * 2)
    };
  },
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
  // The signed distance from the pointer to the nearest line (left of its direction is positive)
  // is the offset, so the blue curve passes under the cursor.
  interaction: {
    kind: 'reach',
    hint: 'move to either side',
    rest: ({inputs, toUnit}) => {
      const target = findParameter(parameters, 'distance').default;
      return toUnit(
        findRestBesideLines(
          inputs,
          target,
          (x, y) => findNearestLine(inputs, x, y)?.signedDistance ?? 0,
          inputs.world
        )
      );
    },
    resolve(pointer, {inputs, values}) {
      const hit = findNearestLine(inputs, pointer.x, pointer.y);
      if (!hit) return {};
      const distance = snapToParameter(findParameter(parameters, 'distance'), hit.signedDistance);
      const side = distance >= 0 ? 'left' : 'right';
      return {
        values: {distance},
        glyphs: [
          {shape: 'line', from: hit.point, to: [pointer.x, pointer.y]},
          {shape: 'dot', at: hit.point}
        ],
        readout: `distance ${distance >= 0 ? '+' : ''}${distance.toFixed(3)} (${side}) · ${JOIN_STYLES[values.joinStyle]}`
      };
    }
  },
  draw: ({outputs}) => [
    {layer: 'lines', context: true, segments: outputs.original, radius: 3},
    {layer: 'lines', segments: outputs.segments, color: OUTPUT_COLOR, radius: 3}
  ]
});
