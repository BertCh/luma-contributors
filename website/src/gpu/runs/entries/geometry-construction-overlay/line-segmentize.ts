// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPULineSegmentize densifies polylines so that no segment is longer than maximumSegmentLength. The
// original vertices are the coarse context; the output vertices (dots) and lines are what it wrote.

import {
  getGPULineSegmentizeParameterValues,
  GPULineSegmentize
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {findParameter, snapToParameter} from '../../interaction';
import {
  addPackedSegments,
  allocateLineOutput,
  createLineSet,
  createPackedSegments,
  findNearestVertex,
  findRestAround,
  HIGHLIGHT_COLOR,
  OUTPUT_COLOR
} from './shared/lines';
import {createLegend} from '../../text';
import {parameters} from './line-segmentize.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Lines densified to a maximum segment length; dots are the vertices written.',
  createInputs: ({random, world}) => ({...createLineSet(random, world, 4, 5), world}),
  build(builder, lines, values) {
    const closedLines = values.closedLines === 1;
    const positions = builder.upload('positions', 'float32x2', lines.positions);
    const lineOffsets = builder.upload('line-offsets', 'uint32', lines.lineOffsets);
    const result = allocateLineOutput(builder, 'segmentized', 4000, lines.lineCount);
    const parameterBuffer = builder.allocate('segmentize-parameters', 'float32', 4);
    builder.graph.add(
      new GPULineSegmentize({
        positions: positions.view,
        lineOffsets: lineOffsets.view,
        closedLines,
        parameters: parameterBuffer.view,
        output: result.output
      })
    );
    const drawing = addPackedSegments(
      builder,
      'segmentized',
      result.positions,
      result.lineOffsets,
      {
        closed: closedLines,
        lineCount: result.lineCount
      }
    );
    const originalPoints = builder.upload('original-points', 'float32x2', lines.positions);
    const original = builder.upload(
      'original',
      'float32x4',
      createPackedSegments(lines.positions, lines.lineOffsets, closedLines)
    );
    return {
      outputs: {original, originalPoints, vertexCount: result.count, ...drawing},
      update: current =>
        parameterBuffer.write(
          getGPULineSegmentizeParameterValues({maximumSegmentLength: current.maximumSegmentLength})
        )
    };
  },
  rebuildOn: ['closedLines'],
  // The distance from the pointer to the nearest original vertex is the longest segment allowed:
  // the ring around that vertex is the reach of its segments.
  interaction: {
    kind: 'reach',
    hint: 'move off a vertex',
    rest: ({inputs, toUnit}) =>
      toUnit(
        findRestAround(
          [inputs.positions[0], inputs.positions[1]],
          findParameter(parameters, 'maximumSegmentLength').default,
          (x, y) => findNearestVertex(inputs.positions, x, y)?.distance ?? 0,
          inputs.world
        )
      ),
    resolve(pointer, {inputs, outputs, peek}) {
      const hit = findNearestVertex(inputs.positions, pointer.x, pointer.y);
      if (!hit) return {};
      const vertexCount = peek(outputs.vertexCount)?.[0];
      const maximumSegmentLength = snapToParameter(
        findParameter(parameters, 'maximumSegmentLength'),
        hit.distance
      );
      return {
        values: {maximumSegmentLength},
        glyphs: [
          {shape: 'ring', at: hit.point, radius: maximumSegmentLength, dashed: true},
          {shape: 'line', from: hit.point, to: [pointer.x, pointer.y], dashed: true}
        ],
        readout: `maximumSegmentLength ${maximumSegmentLength.toFixed(2)}${vertexCount === undefined ? '' : ` · ${vertexCount} vertices`}`
      };
    }
  },
  // Original vertices are large muted discs; every vertex the operation wrote is an orange dot on
  // top, so a disc with an orange core is an original vertex and a lone orange dot an inserted one.
  draw: ({outputs, theme}) => [
    {layer: 'lines', context: true, segments: outputs.original, radius: 3},
    {layer: 'dots', context: true, positions: outputs.originalPoints, radius: 13},
    {layer: 'lines', segments: outputs.segments, color: OUTPUT_COLOR, radius: 3},
    {layer: 'dots', positions: outputs.points, radius: 6.5, color: HIGHLIGHT_COLOR},
    {
      layer: 'text',
      ...createLegend({
        corner: 'top-left',
        entries: [
          {label: 'original vertex', swatch: theme.muted},
          {
            label: 'written vertex (lone dot: inserted)',
            swatch: [HIGHLIGHT_COLOR[0], HIGHLIGHT_COLOR[1], HIGHLIGHT_COLOR[2]]
          }
        ]
      })
    }
  ]
});
