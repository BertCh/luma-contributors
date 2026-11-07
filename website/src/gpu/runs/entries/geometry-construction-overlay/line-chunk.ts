// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPULineChunk cuts every line into pieces of a fixed length (mode 'chunk'). Pieces are colored by
// their output line.

import {
  getGPULineChunkParameterValues,
  GPULineChunk
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {
  addPackedSegments,
  allocateLineOutput,
  createLineSet,
  createPackedSegments
} from './shared/lines';
import {parameters} from './line-chunk.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Lines cut into equal pieces; each output line has its own color.',
  createInputs: ({random, world}) => createLineSet(random, world, 4, 8, 1),
  build(builder, lines) {
    const positions = builder.upload('positions', 'float32x2', lines.positions);
    const lineOffsets = builder.upload('line-offsets', 'uint32', lines.lineOffsets);
    const result = allocateLineOutput(builder, 'pieces', 3000, 400);
    const parameterBuffer = builder.allocate('chunk-parameters', 'float32', 4);
    builder.graph.add(
      new GPULineChunk({
        positions: positions.view,
        lineOffsets: lineOffsets.view,
        mode: 'chunk',
        parameters: parameterBuffer.view,
        output: result.output
      })
    );
    const drawing = addPackedSegments(builder, 'pieces', result.positions, result.lineOffsets, {
      lineCount: result.lineCount
    });
    const original = builder.upload(
      'original',
      'float32x4',
      createPackedSegments(lines.positions, lines.lineOffsets, false)
    );
    return {
      outputs: {original, ...drawing},
      update: current =>
        parameterBuffer.write(getGPULineChunkParameterValues({chunkLength: current.chunkLength}))
    };
  },
  draw: ({outputs}) => [
    {layer: 'lines', context: true, segments: outputs.original, radius: 3},
    {
      layer: 'segments',
      nodes: outputs.points,
      edges: outputs.edges,
      values: outputs.lineIds,
      radius: 2.2
    },
    {layer: 'dots', positions: outputs.points, values: outputs.lineIds, radius: 3.2}
  ]
});
