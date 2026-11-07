// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURectangleClip cuts wandering lines to an axis-aligned rectangle. Muted lines are the input,
// the outlined box is the per-frame clip rectangle, and the bright edges are the clipped
// positions the operation wrote, linked by the output line offsets.

import {
  getGPURectangleClipParameterValues,
  GPURectangleClip
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {makeRandom} from '../../../demos/kit';
import {defineOperationRun} from '../../define-run';
import {createCrossingLines} from './shared/polygon-set';
import {addBoxSegments, addOutputEdges} from './shared/shape-segments';
import {parameters} from './rectangle-clip.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Lines clipped to the rectangle; the pieces inside are drawn from the output buffers.',
  createInputs: ({seed, world}) => ({
    world,
    lines: createCrossingLines(makeRandom(seed), world, 4, 5, 16, 0.08)
  }),
  build(builder, {lines, world}) {
    const positions = builder.upload('positions', 'float32x2', lines.positions);
    const lineOffsets = builder.upload('line-offsets', 'uint32', lines.lineOffsets);
    const parameterBuffer = builder.allocate('parameters', 'float32', 8);
    const capacity = lines.vertexCount * 2;
    const clipped = builder.allocate('clipped', 'float32x2', capacity);
    const clippedOffsets = builder.allocate('clipped-offsets', 'uint32', capacity);
    builder.graph.add(
      new GPURectangleClip({
        geometryType: 'lines',
        positions: positions.view,
        lineOffsets: lineOffsets.view,
        parameters: parameterBuffer.view,
        output: {
          positions: clipped.view,
          lineOffsets: clippedOffsets.view,
          status: {
            count: builder.allocate('count', 'uint32', 1).view,
            overflow: builder.allocate('overflow', 'uint32', 1).view
          }
        }
      })
    );
    const box = builder.allocate('box', 'float32x4', 1);
    return {
      outputs: {
        box,
        boxSegments: addBoxSegments(builder, 'box', box),
        clipped,
        edges: addOutputEdges(builder, 'clipped', clippedOffsets, capacity, false),
        inputSegments: builder.upload(
          'input-segments',
          'float32x4',
          Float32Array.from([...lines.firstSegments, ...lines.secondSegments])
        )
      },
      update: values => {
        const half = values.size * 0.5;
        const settings = {
          minX: values.centerX - half * world.height * 1.4,
          minY: values.centerY - half,
          maxX: values.centerX + half * world.height * 1.4,
          maxY: values.centerY + half
        };
        parameterBuffer.write(getGPURectangleClipParameterValues(settings));
        box.write(Float32Array.of(settings.minX, settings.minY, settings.maxX, settings.maxY));
      }
    };
  },
  draw: ({outputs}) => [
    {layer: 'lines', context: true, segments: outputs.inputSegments},
    {layer: 'lines', segments: outputs.boxSegments, color: [1, 0.6, 0.2, 1]},
    {layer: 'segments', nodes: outputs.clipped, edges: outputs.edges, radius: 2.6}
  ]
});
