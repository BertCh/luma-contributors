// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUOutlineGeometry writes a render-ready triangle outline of a given distance around points,
// lines or polygon rings: discs at points, round-capped strokes along lines, round-joined strokes
// along rings. The blue fill is the triangle list the operation wrote; the muted input is on top.

import {
  getGPUOutlineGeometryParameterValues,
  getGPUOutlineGeometryVerticesPerInput,
  GPUOutlineGeometry
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createLineSet, createPackedSegments, OUTPUT_COLOR} from './shared/lines';
import {
  createBlobSet,
  createRingSegments,
  createTriangleListDrawing,
  getPolygonFields,
  uploadPolygonSet
} from './shared/polygons';
import {parameters} from './outline-geometry.parameters';

const JOIN_SEGMENTS = 12;

export default defineOperationRun({
  parameters,
  caption: 'Triangle outlines around points, lines or polygon rings (input drawn on top).',
  createInputs: ({random, world}) => {
    const points = Float32Array.from({length: 14}, (_, index) =>
      index % 2 === 0 ? (0.08 + 0.84 * random()) * world.width : 0.08 + 0.84 * random()
    );
    return {
      points,
      lines: createLineSet(random, world, 3, 5, 2.5),
      blobs: createBlobSet(random, world, {
        rows: 1,
        columns: 3,
        vertexCount: 9,
        irregularity: 0.4,
        fill: 0.6
      })
    };
  },
  build(builder, inputs, values) {
    const type = (['points', 'lines', 'polygons'] as const)[values.geometryType];
    const pointBuffer = builder.upload('points', 'float32x2', inputs.points);
    const linePositions = builder.upload('line-positions', 'float32x2', inputs.lines.positions);
    const lineOffsets = builder.upload('line-offsets', 'uint32', inputs.lines.lineOffsets);
    const blobs = uploadPolygonSet(builder, 'blobs', inputs.blobs);
    const parameterBuffer = builder.allocate('outline-parameters', 'float32', 4);
    const vertexCount = (
      type === 'points' ? pointBuffer : type === 'lines' ? linePositions : blobs.positions
    ).length;
    const triangles = builder.allocate(
      'outline-triangles',
      'float32x2',
      vertexCount * getGPUOutlineGeometryVerticesPerInput(JOIN_SEGMENTS)
    );
    const common = {
      joinSegments: JOIN_SEGMENTS,
      parameters: parameterBuffer.view,
      output: {positions: triangles.view}
    };
    builder.graph.add(
      new GPUOutlineGeometry(
        type === 'points'
          ? {geometryType: 'points', positions: pointBuffer.view, ...common}
          : type === 'lines'
            ? {
                geometryType: 'lines',
                positions: linePositions.view,
                lineOffsets: lineOffsets.view,
                ...common
              }
            : {geometryType: 'polygons', ...getPolygonFields(blobs), ...common}
      )
    );
    const input = builder.upload(
      'input',
      'float32x4',
      type === 'lines'
        ? createPackedSegments(inputs.lines.positions, inputs.lines.lineOffsets, false)
        : type === 'polygons'
          ? createRingSegments(inputs.blobs)
          : new Float32Array(4)
    );
    const inputPoints =
      type === 'points'
        ? pointBuffer
        : builder.upload('no-points', 'float32x2', Float32Array.of(-9, -9));
    return {
      outputs: {
        triangles,
        input,
        inputPoints,
        mesh: createTriangleListDrawing(builder, 'outline', triangles.length)
      },
      update: current =>
        parameterBuffer.write(getGPUOutlineGeometryParameterValues({distance: current.distance}))
    };
  },
  rebuildOn: ['geometryType'],
  draw: ({outputs}) => [
    {
      layer: 'polygons',
      positions: outputs.triangles,
      indices: outputs.mesh.indices,
      owners: outputs.mesh.owners,
      values: outputs.mesh.values,
      triangleCount: outputs.mesh.triangleCount,
      color: OUTPUT_COLOR,
      alpha: 0.5
    },
    {layer: 'lines', context: true, segments: outputs.input, radius: 2},
    {layer: 'dots', context: true, positions: outputs.inputPoints, radius: 4}
  ]
});
