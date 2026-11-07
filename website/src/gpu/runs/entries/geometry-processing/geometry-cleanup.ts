// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUGeometryCleanup snaps a densely sampled line to a grid and drops the repeated vertices that
// creates. The muted path is the input; the bright path is the compacted output, drawn with the
// vertex count the operation wrote.

import {
  getGPUGeometryCleanupParameterValues,
  GPUGeometryCleanup
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {addJitter, createIdentityIds, createSingleLine} from './shared/polygon-set';
import {parameters} from './geometry-cleanup.parameters';

export default defineOperationRun({
  parameters,
  caption: 'A dense line snapped to a grid, with repeated vertices removed.',
  createInputs: ({random, world}) => {
    const line = createSingleLine(random, world, 160);
    return {line, noisy: addJitter(line.positions, random, 0.002)};
  },
  build(builder, {line, noisy}) {
    const positions = builder.upload('positions', 'float32x2', noisy);
    const lineOffsets = builder.upload('line-offsets', 'uint32', line.lineOffsets);
    const parameterBuffer = builder.allocate('parameters', 'float32', 4);
    const cleaned = builder.allocate('cleaned', 'float32x2', line.count);
    const cleanedOffsets = builder.allocate('cleaned-offsets', 'uint32', 2);
    const count = builder.allocate('count', 'uint32', 1);
    const overflow = builder.allocate('overflow', 'uint32', 1);
    const requiredCount = builder.allocate('required-count', 'uint32', 1);
    builder.graph.add(
      new GPUGeometryCleanup({
        geometryType: 'lines',
        positions: positions.view,
        lineOffsets: lineOffsets.view,
        parameters: parameterBuffer.view,
        output: {
          positions: cleaned.view,
          lineOffsets: cleanedOffsets.view,
          status: {count: count.view, overflow: overflow.view, requiredCount: requiredCount.view}
        }
      })
    );
    return {
      outputs: {
        positions,
        cleaned,
        count,
        ids: builder.upload('ids', 'uint32', createIdentityIds(line.count))
      },
      update: values =>
        parameterBuffer.write(
          getGPUGeometryCleanupParameterValues({
            gridSize: values.gridSize,
            tolerance: values.tolerance,
            removeRepeatedPoints: true
          })
        )
    };
  },
  draw: ({inputs, outputs}) => [
    {
      layer: 'paths',
      context: true,
      nodes: outputs.positions,
      ids: outputs.ids,
      count: inputs.line.count
    },
    {layer: 'paths', nodes: outputs.cleaned, ids: outputs.ids, count: outputs.count, radius: 2}
  ]
});
