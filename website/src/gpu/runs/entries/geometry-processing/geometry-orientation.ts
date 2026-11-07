// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUGeometryOrientation forces polygon winding. Half the input rings wind each way; the edges
// of the output are colored by the `reversedRings` flag the operation wrote for their ring.

import {
  getGPUGeometryOrientationParameterValues,
  GPUGeometryOrientation
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {makeRandom} from '../../../demos/kit';
import {defineOperationRun} from '../../define-run';
import {createPolygonSet, createRingOfVertex, uploadPolygonSet} from './shared/polygon-set';
import {addKernel} from './shared/shape-segments';
import {parameters} from './geometry-orientation.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Rings the operation had to reverse are highlighted.',
  createInputs: ({seed, world}) => ({
    set: createPolygonSet(makeRandom(seed), world, {mixedWinding: true, hasHoles: true})
  }),
  build(builder, inputs) {
    const polygons = uploadPolygonSet(builder, inputs.set);
    const parameterBuffer = builder.allocate('parameters', 'float32', 4);
    const ringCount = inputs.set.ringOffsets.length - 1;
    const oriented = builder.allocate('oriented', 'float32x2', inputs.set.positions.length / 2);
    const reversedRings = builder.allocate('reversed-rings', 'uint32', ringCount);
    builder.graph.add(
      new GPUGeometryOrientation({
        mode: 'orient-polygons',
        positions: polygons.positions.view,
        featureOffsets: polygons.featureOffsets.view,
        polygonOffsets: polygons.polygonOffsets.view,
        ringOffsets: polygons.ringOffsets.view,
        parameters: parameterBuffer.view,
        output: {positions: oriented.view, reversedRings: reversedRings.view}
      })
    );
    // Spread each ring's flag onto its vertices so the edges can be colored by it.
    const ringOfVertex = builder.upload(
      'ring-of-vertex',
      'uint32',
      createRingOfVertex(inputs.set.ringOffsets)
    );
    const reversedVertices = builder.allocate('reversed-vertices', 'uint32', ringOfVertex.length);
    addKernel(builder, {
      id: 'spread-reversed',
      invocationCount: ringOfVertex.length,
      bindings: [
        {name: 'rings', buffer: reversedRings, access: 'read'},
        {name: 'ringOfVertex', buffer: ringOfVertex, access: 'read'},
        {name: 'flags', buffer: reversedVertices, access: 'read_write'}
      ],
      body: 'flags[flagsOffset + index] = rings[ringsOffset + ringOfVertex[ringOfVertexOffset + index]];'
    });
    return {
      outputs: {...polygons, oriented, reversedVertices},
      update: values =>
        parameterBuffer.write(
          getGPUGeometryOrientationParameterValues({
            exteriorClockwise: values.exteriorClockwise === 1
          })
        )
    };
  },
  draw: ({inputs, outputs}) => [
    {layer: 'lines', context: true, segments: outputs.outline, count: inputs.set.outlineCount},
    {
      layer: 'segments',
      nodes: outputs.oriented,
      edges: outputs.edges,
      values: outputs.reversedVertices,
      radius: 2.4
    }
  ]
});
