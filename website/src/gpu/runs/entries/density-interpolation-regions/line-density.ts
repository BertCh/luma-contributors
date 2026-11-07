// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPULineDensity accumulates segment length per grid cell (divided by cell area); the white lines
// are the input polylines, the raster is the `densities` output.

import {
  getGPULineDensityParameterValues,
  GPULineDensity
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createPolylines} from '../../inputs';
import {parameters} from './line-density.parameters';
import {flipVertical} from './support/geometry';

export default defineOperationRun({
  parameters,
  caption: 'Length of lines per cell area; the lines are the input.',
  createInputs: ({random, world}) => ({
    width: world.width,
    random,
    lines: createPolylines(random, world, 40, 14)
  }),
  build(builder, inputs, values) {
    const rows = values.rows;
    const columns = Math.round(inputs.width * rows);
    const {lines} = inputs;
    const lineCount = values.lineCount;
    const vertexEnd = lines.offsets[lineCount];
    const positions = builder.upload(
      'positions',
      'float32x2',
      flipVertical(lines.vertices.slice(0, vertexEnd * 2))
    );
    const lineOffsets = builder.upload(
      'line-offsets',
      'uint32',
      lines.offsets.slice(0, lineCount + 1)
    );
    const parameterBuffer = builder.allocate('parameters', 'float32', 4);
    parameterBuffer.write(
      getGPULineDensityParameterValues({
        minX: 0,
        minY: 0,
        cellWidth: inputs.width / columns,
        cellHeight: 1 / rows
      })
    );
    const lengths = builder.allocate('lengths', 'float32', columns * rows);
    const densities = builder.allocate('densities', 'float32', columns * rows);
    const overflow = builder.allocate('overflow', 'uint32', 1);
    builder.graph.add(
      new GPULineDensity({
        positions: positions.view,
        lineOffsets: lineOffsets.view,
        columns,
        rows,
        parameters: parameterBuffer.view,
        output: {
          lengths: lengths.view,
          densities: densities.view,
          status: {overflow: overflow.view}
        }
      })
    );
    const segments = builder.upload('segments', 'float32x4', lines.segments);
    return {outputs: {densities, columns, rows, segments, segmentCount: vertexEnd - lineCount}};
  },
  draw: ({outputs}) => [
    {layer: 'raster', values: outputs.densities, columns: outputs.columns, rows: outputs.rows},
    {layer: 'lines', context: true, segments: outputs.segments, count: outputs.segmentCount}
  ]
});
