// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPULineSmooth runs Chaikin corner cutting: each pass replaces every corner by two points at
// `ratio` along its edges. The coarse polyline is the muted context; the smoothed lines are output.

import {
  getGPULineSmoothParameterValues,
  GPULineSmooth
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {
  addPackedSegments,
  allocateLineOutput,
  createLineSet,
  createPackedSegments,
  OUTPUT_COLOR
} from './shared/lines';
import {parameters} from './line-smooth.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Chaikin corner cutting: coarse polylines (muted) and their smoothed output.',
  createInputs: ({random, world}) => createLineSet(random, world, 3, 7, 2),
  build(builder, lines, values) {
    const positions = builder.upload('positions', 'float32x2', lines.positions);
    const lineOffsets = builder.upload('line-offsets', 'uint32', lines.lineOffsets);
    const result = allocateLineOutput(builder, 'smoothed', 4000, lines.lineCount);
    const parameterBuffer = builder.allocate('smooth-parameters', 'float32', 4);
    builder.graph.add(
      new GPULineSmooth({
        positions: positions.view,
        lineOffsets: lineOffsets.view,
        iterationCount: values.iterationCount,
        parameters: parameterBuffer.view,
        output: result.output
      })
    );
    const drawing = addPackedSegments(builder, 'smoothed', result.positions, result.lineOffsets);
    const original = builder.upload(
      'original',
      'float32x4',
      createPackedSegments(lines.positions, lines.lineOffsets, false)
    );
    const originalPoints = builder.upload('original-points', 'float32x2', lines.positions);
    return {
      outputs: {original, originalPoints, ...drawing},
      update: current =>
        parameterBuffer.write(getGPULineSmoothParameterValues({ratio: current.ratio}))
    };
  },
  rebuildOn: ['iterationCount'],
  draw: ({outputs}) => [
    {layer: 'lines', context: true, segments: outputs.original},
    {layer: 'dots', context: true, positions: outputs.originalPoints, radius: 5},
    {layer: 'lines', segments: outputs.segments, color: OUTPUT_COLOR, radius: 1.8}
  ]
});
