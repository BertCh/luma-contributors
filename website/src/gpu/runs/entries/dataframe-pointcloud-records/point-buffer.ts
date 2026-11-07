// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUPointBuffer generating the ring vertices around points (planar mode, radius in map units). The
// GPU writes the offsets of each ring vertex from its center; the outlines are the rings joined
// vertex to vertex around each point.

import {
  GPUPointBuffer,
  getGPUPointBufferParameterValues
} from '@luma-contributors/analysis/geospatial-analysis';
import {defineOperationRun} from '../../define-run';
import {addRingSegments} from './_lib/ring-segments';
import {parameters} from './point-buffer.parameters';

const MAXIMUM_POINT_COUNT = 24;

export default defineOperationRun({
  parameters,
  caption: 'Buffer rings generated around points.',
  createInputs: ({random, world}) => ({
    centers: Float32Array.from({length: MAXIMUM_POINT_COUNT * 2}, (_, index) =>
      index % 2 === 0 ? (0.1 + 0.8 * random()) * world.width : 0.1 + 0.8 * random()
    )
  }),
  build(builder, inputs, values) {
    const pointCount = values.pointCount;
    const centers = builder.upload('centers', 'float32x2', inputs.centers.slice(0, pointCount * 2));
    const parameterBuffer = builder.allocate('parameters', 'float32', 2);
    const offsets = builder.allocate('offsets', 'float32x2', pointCount * values.verticesPerRing);
    builder.graph.add(
      new GPUPointBuffer({
        points: centers.view,
        parameters: parameterBuffer.view,
        offsets: offsets.view,
        verticesPerRing: values.verticesPerRing,
        coordinateSystem: 'planar'
      })
    );
    return {
      outputs: {
        centers,
        rings: addRingSegments(builder, 'rings', centers, offsets, values.verticesPerRing)
      },
      update: current => parameterBuffer.write(getGPUPointBufferParameterValues(current.radius))
    };
  },
  rebuildOn: ['verticesPerRing', 'pointCount'],
  draw: ({outputs}) => [
    {
      layer: 'dots',
      context: true,
      positions: outputs.centers,
      color: [0.9, 0.9, 1, 0.9],
      radius: 3
    },
    {layer: 'lines', segments: outputs.rings, color: [0.3, 0.75, 1, 1], radius: 1.8}
  ]
});
