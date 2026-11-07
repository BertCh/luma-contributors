// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPULineSimplification (Douglas-Peucker) keeps the vertices whose importance exceeds the
// tolerance. The original line is muted; kept vertices are the output ids, drawn as the simplified
// line. Dropped vertices stay as muted dots.

import {
  getGPULineSimplificationParameterValues,
  GPULineSimplification
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {
  addSelectedSegments,
  createLineSet,
  createPackedSegments,
  createRowLines,
  OUTPUT_COLOR
} from './shared/lines';
import {parameters} from './line-simplification.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Douglas-Peucker: the kept vertices (orange) and the simplified line (blue).',
  createInputs: ({random, world}) => createLineSet(random, world, 3, 48, 0.9),
  build(builder, lines) {
    const rowCount = lines.positions.length / 2;
    const positions = builder.upload('positions', 'float32x2', lines.positions);
    const lineOffsets = builder.upload('line-offsets', 'uint32', lines.lineOffsets);
    const importance = builder.allocate('importance', 'float32', rowCount);
    const ids = builder.allocate('kept-ids', 'uint32', rowCount);
    const count = builder.allocate('kept-count', 'uint32', 1);
    const overflow = builder.allocate('kept-overflow', 'uint32', 1);
    const requiredCount = builder.allocate('kept-required', 'uint32', 1);
    const keptMask = builder.allocate('kept-mask', 'uint32', rowCount);
    const parameterBuffer = builder.allocate('tolerance', 'float32', 4);
    builder.graph.add(
      new GPULineSimplification({
        positions: positions.view,
        lineOffsets: lineOffsets.view,
        importance: importance.view,
        parameters: parameterBuffer.view,
        output: {
          ids: ids.view,
          outputMask: keptMask.view,
          status: {count: count.view, overflow: overflow.view, requiredCount: requiredCount.view}
        }
      })
    );
    const rowLines = builder.upload('row-lines', 'uint32', createRowLines(lines.lineOffsets));
    const simplified = addSelectedSegments(builder, 'simplified', positions, rowLines, ids, count);
    const original = builder.upload(
      'original',
      'float32x4',
      createPackedSegments(lines.positions, lines.lineOffsets, false)
    );
    return {
      outputs: {original, simplified, positions, keptMask},
      update: current =>
        parameterBuffer.write(
          getGPULineSimplificationParameterValues({tolerance: current.tolerance})
        )
    };
  },
  draw: ({outputs}) => [
    {layer: 'lines', context: true, segments: outputs.original},
    {layer: 'dots', context: true, positions: outputs.positions, radius: 2},
    {layer: 'lines', segments: outputs.simplified, color: OUTPUT_COLOR, radius: 2},
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.keptMask,
      ramp: 'sequential',
      range: [0.5, 1],
      radius: 3.6,
      alpha: 1
    }
  ]
});
