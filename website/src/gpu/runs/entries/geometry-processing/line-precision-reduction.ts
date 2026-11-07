// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPULinePrecisionReduction rounds a line to a grid. The output vertex count is the last line
// offset the operation wrote, copied into a count buffer so the path draws exactly what was kept.

import {GPULinePrecisionReduction} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createIdentityIds, createSingleLine} from './shared/polygon-set';
import {addKernel} from './shared/shape-segments';
import {parameters} from './line-precision-reduction.parameters';

export default defineOperationRun({
  parameters,
  caption: 'The line rounded to the precision grid; collapsed lines follow the chosen mode.',
  createInputs: ({random, world}) => createSingleLine(random, world, 120),
  build(builder, line, values) {
    const positions = builder.upload('positions', 'float32x2', line.positions);
    const lineOffsets = builder.upload('line-offsets', 'uint32', line.lineOffsets);
    const reduced = builder.allocate('reduced', 'float32x2', line.count);
    const reducedOffsets = builder.allocate('reduced-offsets', 'uint32', 2);
    const status = builder.allocate('status', 'uint32', 1);
    builder.graph.add(
      new GPULinePrecisionReduction({
        positions: positions.view,
        lineOffsets: lineOffsets.view,
        gridSize: values.gridSize,
        mode: 'valid-output',
        output: {positions: reduced.view, lineOffsets: reducedOffsets.view, status: status.view}
      })
    );
    const count = builder.allocate('count', 'uint32', 1);
    addKernel(builder, {
      id: 'copy-count',
      invocationCount: 1,
      bindings: [
        {name: 'offsets', buffer: reducedOffsets, access: 'read'},
        {name: 'count', buffer: count, access: 'read_write'}
      ],
      body: 'count[countOffset] = offsets[offsetsOffset + 1u];'
    });
    return {
      outputs: {
        positions,
        reduced,
        count,
        ids: builder.upload('ids', 'uint32', createIdentityIds(line.count))
      }
    };
  },
  draw: ({inputs, outputs}) => [
    {
      layer: 'paths',
      context: true,
      nodes: outputs.positions,
      ids: outputs.ids,
      count: inputs.count
    },
    {layer: 'paths', nodes: outputs.reduced, ids: outputs.ids, count: outputs.count, radius: 2}
  ]
});
