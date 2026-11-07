// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPULineLengthPerPolygon splits every line at tract boundaries and sums the length inside each
// tract; tracts are filled by that `lengths` output with the lines on top.

import {GPULineLengthPerPolygon} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createPolylines, createTracts} from '../../inputs';
import {parameters} from './line-length-per-polygon.parameters';
import {getPackedPolygons, uploadPackedPolygons, uploadTractMesh} from './support/geometry';

export default defineOperationRun({
  parameters,
  caption: 'Tracts colored by the length of line inside them.',
  createInputs: ({random, world}) => ({
    random,
    world,
    lines: createPolylines(random, world, 40, 14)
  }),
  build(builder, inputs, values) {
    const tracts = createTracts(() => 0.5, inputs.world, values.tractRows, 0);
    const {lines} = inputs;
    const lineCount = values.lineCount;
    const vertexEnd = lines.offsets[lineCount];
    const lengths = builder.allocate('lengths', 'float32', tracts.count);
    const overflow = builder.allocate('overflow', 'uint32', 1);
    builder.graph.add(
      new GPULineLengthPerPolygon({
        positions: builder.upload('positions', 'float32x2', lines.vertices.slice(0, vertexEnd * 2))
          .view,
        lineOffsets: builder.upload('line-offsets', 'uint32', lines.offsets.slice(0, lineCount + 1))
          .view,
        polygons: {kind: 'polygons', ...uploadPackedPolygons(builder, getPackedPolygons(tracts))},
        output: {lengths: lengths.view, status: {overflow: overflow.view}}
      })
    );
    return {
      outputs: {
        lengths,
        ...uploadTractMesh(builder, tracts),
        segments: builder.upload('segments', 'float32x4', lines.segments),
        segmentCount: vertexEnd - lineCount,
        triangleCount: tracts.triangleCount,
        outlineCount: tracts.outlineCount
      }
    };
  },
  draw: ({outputs}) => [
    {
      layer: 'polygons',
      positions: outputs.vertices,
      indices: outputs.indices,
      owners: outputs.owners,
      values: outputs.lengths,
      triangleCount: outputs.triangleCount
    },
    {
      layer: 'lines',
      context: true,
      segments: outputs.outline,
      count: outputs.outlineCount,
      alpha: 0.5
    },
    {
      layer: 'lines',
      context: true,
      color: [1, 1, 1, 1],
      segments: outputs.segments,
      count: outputs.segmentCount
    }
  ]
});
